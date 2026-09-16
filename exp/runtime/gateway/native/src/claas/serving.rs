//! Scope-bound private vLLM dispatch and cross-process admission leases.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::response::Response;
use futures_util::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::Scope;
use crate::admission::Admission;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::settlement::AttemptGuard;

#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "UncheckedConfiguration")]
pub(crate) struct Configuration {
    pub bindings: Vec<Binding>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedConfiguration {
    bindings: Vec<Binding>,
}

impl TryFrom<UncheckedConfiguration> for Configuration {
    type Error = &'static str;

    fn try_from(value: UncheckedConfiguration) -> Result<Self, Self::Error> {
        let mut origins = std::collections::HashSet::new();
        for binding in &value.bindings {
            let Some(key) = private_origin(&binding.private_base_url) else {
                return Err("CLaaS private vLLM origin must use loopback HTTP on 127.0.0.1, localhost, or [::1], with no credentials, query, fragment, or non-root path");
            };
            if !origins.insert(key) {
                return Err("CLaaS supports one application per private vLLM origin; start a separate private server for each application");
            }
        }
        Ok(Self {
            bindings: value.bindings,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    scope: Scope,
    alias: String,
    registry_path: String,
    private_base_url: String,
    state_path: String,
    admission_lock_path: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Revision {
    scope: Scope,
    policy_revision: String,
    model_id: String,
    model_revision: String,
    tokenizer_id: String,
    tokenizer_revision: String,
    adapter_directory: Option<String>,
    manifest_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    scope: Scope,
    generation: u64,
    active: Revision,
    previous: Option<Revision>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    schema_version: u8,
    scope: Scope,
    binding_sha256: String,
    paused: bool,
    generation: u64,
    policy_revision: String,
    model_name: String,
}

/// Shared lock lifetime follows the caller response body, including cancellation.
#[derive(Clone)]
pub(crate) struct RequestLease(Arc<File>);

/// An exclusive native lease uses exactly the gateway's lock primitive on every OS.
#[pyclass]
pub struct ExclusiveLease {
    file: Mutex<Option<File>>,
}

#[pymethods]
impl ExclusiveLease {
    /// Release ownership without unlinking or replacing the coordination inode.
    fn release(&self) {
        if let Ok(mut file) = self.file.lock() {
            file.take();
        }
    }
}

#[pyfunction]
pub fn claas_acquire_exclusive(
    py: Python<'_>,
    path: String,
    timeout_seconds: f64,
) -> PyResult<ExclusiveLease> {
    if !Path::new(&path).is_absolute()
        || !timeout_seconds.is_finite()
        || !(0.0..=3600.0).contains(&timeout_seconds)
        || timeout_seconds == 0.0
    {
        return Err(PyValueError::new_err(
            "an absolute lock path and finite positive timeout are required",
        ));
    }
    py.detach(move || {
        let file = open_lock(&path).map_err(PyRuntimeError::new_err)?;
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
        loop {
            match file.try_lock() {
                Ok(()) => {
                    return Ok(ExclusiveLease {
                        file: Mutex::new(Some(file)),
                    })
                }
                Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "gateway drain lock unavailable: {error}"
                    )))
                }
            }
        }
    })
}

fn open_lock(path: &str) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(|error| error.to_string())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, PublicError> {
    let mut text = String::new();
    File::open(path)
        .map_err(|_| unavailable())?
        .take(65_537)
        .read_to_string(&mut text)
        .map_err(|_| unavailable())?;
    if text.len() > 65_536 {
        return Err(unavailable());
    }
    serde_json::from_str(&text).map_err(|_| unavailable())
}

fn digest(value: &Value) -> String {
    fn canonical(value: &Value) -> Value {
        match value {
            Value::Object(fields) => {
                let mut names: Vec<_> = fields.keys().collect();
                names.sort();
                Value::Object(
                    names
                        .into_iter()
                        .map(|name| (name.clone(), canonical(&fields[name])))
                        .collect(),
                )
            }
            Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
            _ => value.clone(),
        }
    }
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical(value)).unwrap_or_default())
    )
}

fn model_name(revision: &Revision) -> String {
    let mut value = serde_json::to_value(revision).unwrap_or_default();
    if revision.adapter_directory.is_none() {
        value.as_object_mut().unwrap().remove("policy_revision");
    }
    format!("claas-{}", digest(&value))
}

fn unavailable() -> PublicError {
    let mut error = PublicError::new(503, "claas_serving_paused", "This application's model is paused or its serving revision is unavailable. Retry after recovery.", "api_error");
    error.retry_after_seconds = Some(1);
    error
}

fn ready(binding: &Binding) -> Result<State, PublicError> {
    let state: State = read_json(&binding.state_path)?;
    if state.schema_version != 1
        || state.paused
        || state.scope != binding.scope
        || state.binding_sha256
            != digest(&serde_json::to_value(binding).map_err(|_| unavailable())?)
    {
        return Err(unavailable());
    }
    Ok(state)
}

fn private_origin(raw: &str) -> Option<(String, String, u16)> {
    let authority = raw.strip_prefix("http://")?;
    let authority = authority.strip_suffix('/').unwrap_or(authority);
    let suffix = ["127.0.0.1", "localhost", "[::1]"]
        .iter()
        .find_map(|host| authority.strip_prefix(host))?;
    let port = if suffix.is_empty() {
        80
    } else {
        let digits = suffix.strip_prefix(':')?;
        if digits.is_empty()
            || digits.len() > 5
            || !digits.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        digits.parse::<u16>().ok()?
    };
    (port > 0).then(|| ("http".to_owned(), "loopback".to_owned(), port))
}

fn origin(raw: &str) -> Option<(String, String, u16)> {
    let url = reqwest::Url::parse(raw).ok()?;
    let host = url.host_str()?.trim_matches(['[', ']']);
    let host = match host {
        "127.0.0.1" | "localhost" | "::1" => "loopback",
        other => other,
    };
    Some((
        url.scheme().to_owned(),
        host.to_owned(),
        url.port_or_known_default()?,
    ))
}

fn uses_private_origin(config: &Configuration, admission: &Admission) -> bool {
    admission.route.iter().any(|wire| {
        let Some(target) = origin(&wire.url) else {
            return false;
        };
        config
            .bindings
            .iter()
            .any(|binding| origin(&binding.private_base_url).is_some_and(|bound| bound == target))
    })
}

fn bind(
    config: &Configuration,
    admission: &mut Admission,
) -> Result<Option<RequestLease>, PublicError> {
    let Some((_, user)) = admission
        .caller_scope
        .as_deref()
        .and_then(|scope| scope.split_once(':'))
    else {
        if uses_private_origin(config, admission)
            || config
                .bindings
                .iter()
                .any(|binding| binding.alias == admission.alias)
        {
            return Err(unavailable());
        }
        return Ok(None);
    };
    let Some(binding) = config
        .bindings
        .iter()
        .find(|binding| binding.scope.user_id == user && binding.alias == admission.alias)
    else {
        return if uses_private_origin(config, admission) {
            Err(unavailable())
        } else {
            Ok(None)
        };
    };
    ready(binding)?;
    let file = open_lock(&binding.admission_lock_path).map_err(|_| unavailable())?;
    file.try_lock_shared().map_err(|_| unavailable())?;
    let state = ready(binding)?;
    let registry: Registry = read_json(&binding.registry_path)?;
    let active = &registry.active;
    if registry.scope != binding.scope
        || active.scope != binding.scope
        || registry
            .previous
            .as_ref()
            .is_some_and(|previous| previous.scope != binding.scope)
        || registry.generation != state.generation
        || active.policy_revision != state.policy_revision
        || model_name(active) != state.model_name
        || admission.route.len() != 1
        || admission.refusal_failover
    {
        return Err(unavailable());
    }
    let wire = &mut admission.route[0];
    let mut base = binding.private_base_url.trim_end_matches('/').to_owned();
    if !base.ends_with("/v1") {
        base.push_str("/v1");
    }
    if wire.dialect != "openai_compatible"
        || wire.url != format!("{base}/chat/completions")
        || wire.upstream_body.is_some()
        || wire.model_id != active.model_id
        || wire
            .model_revision
            .as_ref()
            .is_some_and(|revision| revision != &active.model_revision)
    {
        return Err(unavailable());
    }
    let payload = wire
        .upstream_payload
        .as_object_mut()
        .ok_or_else(unavailable)?;
    if payload.get("model").and_then(Value::as_str) != Some(active.model_id.as_str()) {
        return Err(unavailable());
    }
    payload.insert("model".to_owned(), json!(state.model_name));
    wire.model_id = state.model_name;
    Ok(Some(RequestLease(Arc::new(file))))
}

/// Reject paused authorized aliases before keyed replay or new durable admission.
pub(crate) async fn authenticate(
    state: &crate::server::AppState,
    raw_key: &str,
    body: &[u8],
) -> Result<(), PublicError> {
    let argument = json!({"raw_key":raw_key}).to_string();
    state.bridge.call("authenticate", argument.clone()).await?;
    let Some(config) = &state.serving else {
        return Ok(());
    };
    let body: Value = serde_json::from_slice(body).map_err(|_| PublicError::invalid_json())?;
    let Some(alias) = body.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    if !config.bindings.iter().any(|binding| binding.alias == alias) {
        return Ok(());
    }
    let authority = state
        .bridge
        .call(
            "claas_authority",
            json!({"raw_key": raw_key, "alias": alias}).to_string(),
        )
        .await?;
    let identity: Value = serde_json::from_str(&authority).map_err(|_| unavailable())?;
    if identity.get("alias_granted").and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    let user = identity
        .get("user_id")
        .and_then(Value::as_str)
        .ok_or_else(unavailable)?;
    if let Some(binding) = config
        .bindings
        .iter()
        .find(|binding| binding.alias == alias && binding.scope.user_id == user)
    {
        ready(binding)?;
    }
    Ok(())
}

/// Bind only the certified private rung; abandon accepted accounting on any failure.
pub(crate) async fn prepare(
    config: &Option<Configuration>,
    admission: &mut Admission,
    guard: &mut AttemptGuard,
) -> Result<Option<RequestLease>, PublicError> {
    let Some(config) = config else {
        return Ok(None);
    };
    match bind(config, admission) {
        Ok(lease) => {
            guard.retain_serving_lease(lease.clone());
            Ok(lease)
        }
        Err(error) => {
            guard
                .abandon(&Failure::new(
                    FailureClass::Unavailable,
                    "CLaaS serving admission is paused or inconsistent",
                ))
                .await;
            Err(error)
        }
    }
}

/// Preserve the shared lock until the caller finishes or drops the final body.
pub(crate) fn hold(lease: Option<RequestLease>, response: Response) -> Response {
    let Some(lease) = lease else { return response };
    let (parts, body) = response.into_parts();
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), lease),
        |(mut body, lease)| async move {
            let _file = &lease.0;
            body.next().await.map(|item| (item, (body, lease)))
        },
    );
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
#[path = "serving_test.rs"]
mod tests;
