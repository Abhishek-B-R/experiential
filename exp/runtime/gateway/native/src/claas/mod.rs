//! Opt-in local application capture, isolated from content-free accounting.

pub(crate) mod feedback;
mod feedback_contracts;
mod feedback_store;
pub(crate) mod serving;
mod store;
mod stream;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, HttpBody};
use axum::response::Response;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::admission::Admission;

pub(crate) use store::CaptureStore;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub user_id: String,
    pub application_id: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Policy {
    pub scope: Scope,
    pub enabled: bool,
    pub maximum_experiences: usize,
    pub maximum_storage_bytes: usize,
    pub maximum_experience_bytes: usize,
    pub retention_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    pub alias: String,
    pub policy: Policy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureConfiguration {
    pub database_path: String,
    pub bindings: Vec<Binding>,
    pub queue_capacity: usize,
}

/// Bounded content state exists only after a configured scope matches admission.
pub(crate) struct CaptureSession {
    store: Arc<CaptureStore>,
    policy: Policy,
    request_id: String,
    model_id: String,
    protocol: &'static str,
    request: Value,
    deployment_id: Option<String>,
}

impl CaptureSession {
    pub(crate) fn begin(
        store: &Option<Arc<CaptureStore>>,
        admission: &Admission,
        body: &str,
        protocol: &'static str,
    ) -> Option<Self> {
        let store = store.as_ref()?;
        // This scope was derived from the authenticated virtual key, never
        // from caller-supplied application headers or request metadata.
        let user_id = admission.caller_identity_id.as_deref()?;
        let policy = store.policy(user_id, &admission.alias)?;
        if body.len() > policy.maximum_experience_bytes {
            store.skip();
            return None;
        }
        let mut request: Value = serde_json::from_str(body).ok()?;
        let object = request.as_object_mut()?;
        object.retain(|name, _| {
            matches!(
                name.as_str(),
                "model"
                    | "messages"
                    | "input"
                    | "instructions"
                    | "tools"
                    | "tool_choice"
                    | "parallel_tool_calls"
                    | "response_format"
                    | "text"
                    | "temperature"
                    | "top_p"
                    | "max_tokens"
                    | "max_completion_tokens"
                    | "max_output_tokens"
                    | "stop"
                    | "previous_response_id"
                    | "reasoning"
                    | "stream"
            )
        });
        Some(Self {
            store: store.clone(),
            policy: policy.clone(),
            request_id: admission.request_id.clone(),
            model_id: admission.exact_model_id.clone(),
            protocol,
            request,
            deployment_id: None,
        })
    }

    fn finish(self, response: Value) {
        let Some(response_id) = response.get("id").and_then(Value::as_str) else {
            self.store.skip();
            return;
        };
        let captured_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_secs_f64())
            .unwrap_or(0.0);
        let experience_id = format!(
            "experience-{:x}",
            Sha256::digest(
                serde_json::to_vec(&json!([
                    self.policy.scope.user_id,
                    self.policy.scope.application_id,
                    self.request_id,
                    self.protocol
                ]))
                .unwrap_or_default()
            )
        );
        let parent = self.request.get("previous_response_id").cloned();
        let experience = json!({
            "schema_version": 1,
            "experience_id": experience_id,
            "response_id": response_id,
            "episode_id": null,
            "parent_response_id": parent,
            "scope": {"user_id": self.policy.scope.user_id,
                "application_id": self.policy.scope.application_id},
            "protocol": self.protocol,
            "captured_at": captured_at,
            "request": self.request,
            "response": response,
            "provenance": {"source_kind": "traffic", "source_id": self.request_id,
                "model_id": self.model_id, "model_revision": null,
                "deployment_id": self.deployment_id, "policy_revision": null,
                "source_experience_ids": []},
            "exact_tokens": null
        });
        self.store
            .submit(self.policy, experience, captured_at as u64);
    }
}

/// Tee bounded bytes as the caller consumes them; never wait for storage or training.
pub(crate) fn capture_response(session: Option<CaptureSession>, response: Response) -> Response {
    let Some(mut session) = session else {
        return response;
    };
    if !response.status().is_success() {
        return response;
    }
    session.deployment_id = response
        .headers()
        .get("x-gateway-deployment")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let sse = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    let expected_bytes = response.body().size_hint().exact().or_else(|| {
        response
            .headers()
            .get("content-length")?
            .to_str()
            .ok()?
            .parse::<u64>()
            .ok()
    });
    let (parts, body) = response.into_parts();
    let maximum = session.policy.maximum_experience_bytes;
    let body = futures_util::stream::unfold(
        (body.into_data_stream(), Vec::new(), Some(session)),
        move |(mut body, mut bytes, mut session)| async move {
            match body.next().await {
                Some(item) => {
                    if let Some(active) = session.as_ref() {
                        match &item {
                            Ok(chunk) if bytes.len() + chunk.len() <= maximum => {
                                bytes.extend_from_slice(chunk)
                            }
                            _ => {
                                active.store.skip();
                                session = None;
                                bytes.clear();
                            }
                        }
                    }
                    if expected_bytes == Some(bytes.len() as u64) {
                        if let Some(active) = session.take() {
                            finalize(active, &bytes, sse);
                        }
                        bytes.clear();
                    }
                    Some((item, (body, bytes, session)))
                }
                None => {
                    if let Some(active) = session {
                        finalize(active, &bytes, sse);
                    }
                    None
                }
            }
        },
    );
    Response::from_parts(parts, Body::from_stream(body))
}

fn finalize(session: CaptureSession, bytes: &[u8], sse: bool) {
    let result = if sse {
        stream::completed_response(bytes, session.protocol)
    } else {
        serde_json::from_slice(bytes).ok()
    };
    if let Some(value) = result {
        session.finish(value);
    } else {
        session.store.skip();
    }
}

#[cfg(test)]
#[path = "capture_test.rs"]
mod tests;
