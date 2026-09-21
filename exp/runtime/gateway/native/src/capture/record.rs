//! Versioned content records shared by local and hosted capture destinations.

use std::borrow::Cow;

use serde::ser::{Error, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::value::{to_raw_value, RawValue};
use serde_json::Value;

pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Authority-derived tenancy; none of these values comes from a caller's metadata.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub organization_id: String,
    pub identity_id: String,
    pub application_id: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Protocol {
    ChatCompletions,
    Responses,
    Messages,
}

/// Effective input supplied by the authenticated, post-guardrail admission seam.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub request_id: String,
    pub scope: Scope,
    pub protocol: Protocol,
    pub model_id: Option<String>,
    pub context: Value,
}

/// Exact public content, distinguished from a successfully reconstructed completion.
/// A disconnected or truncated stream remains evidence, never a complete rollout.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Response {
    Json {
        status: u16,
        body: Value,
        /// Escaped exact JSON when the query projection contains unstorable text.
        source_json: Option<String>,
    },
    Sse {
        status: u16,
        frames: Vec<Value>,
        truncated: bool,
        client_disconnected: bool,
        source_json: Option<String>,
    },
}

/// Valid JSON encoded once, reused verbatim by sizing, settlement and delivery.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub(crate) struct EncodedResponse(Box<RawValue>);

impl EncodedResponse {
    pub(crate) fn len(&self) -> usize {
        self.0.get().len()
    }
}

impl Response {
    pub(crate) fn encode(&self) -> Option<EncodedResponse> {
        to_raw_value(self).ok().map(EncodedResponse)
    }
}

/// One idempotent request update. A later update may supply its response.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record<R = Response> {
    pub schema_version: u32,
    pub request: Request,
    pub response: Option<R>,
    /// Provider-returned plaintext from an explicitly exposure-enabled winning rung.
    pub provider_reasoning: Option<String>,
    pub provider_reasoning_source_json: Option<String>,
    /// Exact completed tool calls, escaped once so JSONB cannot alter their text.
    pub provider_tool_calls_json: Option<String>,
    pub deployment_id: Option<String>,
    pub captured_at: f64,
}

impl<R: Serialize> Serialize for Record<R> {
    /// Project only exceptional reasoning text; never clone the request or response.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let source = self
            .provider_reasoning
            .as_ref()
            .filter(|text| text.contains('\0'))
            .map(serde_json::to_string)
            .transpose()
            .map_err(S::Error::custom)?;
        let reasoning = self.provider_reasoning.as_ref().map(|text| {
            if source.is_some() {
                Cow::Owned(text.replace('\0', "\u{fffd}"))
            } else {
                Cow::Borrowed(text.as_str())
            }
        });
        let reasoning_source = source
            .as_deref()
            .or(self.provider_reasoning_source_json.as_deref());
        let mut record = serializer.serialize_struct("Record", 8)?;
        record.serialize_field("schema_version", &self.schema_version)?;
        record.serialize_field("request", &self.request)?;
        record.serialize_field("response", &self.response)?;
        record.serialize_field("provider_reasoning", &reasoning)?;
        record.serialize_field("provider_reasoning_source_json", &reasoning_source)?;
        record.serialize_field("provider_tool_calls_json", &self.provider_tool_calls_json)?;
        record.serialize_field("deployment_id", &self.deployment_id)?;
        record.serialize_field("captured_at", &self.captured_at)?;
        record.end()
    }
}

impl<R: Serialize> Record<R> {
    /// Validate both the version and the content budget before destination admission.
    pub(crate) fn encode(&self, maximum_bytes: usize) -> Option<String> {
        if self.schema_version != SCHEMA_VERSION
            || !self.captured_at.is_finite()
            || self.captured_at < 0.0
            || [
                &self.request.request_id,
                &self.request.scope.organization_id,
                &self.request.scope.identity_id,
                &self.request.scope.application_id,
            ]
            .iter()
            .any(|value| value.trim().is_empty() || value.len() > 512)
            || self
                .request
                .model_id
                .as_ref()
                .is_some_and(|value| value.trim().is_empty() || value.len() > 512)
            || self
                .request
                .context
                .get("schema_version")
                .and_then(Value::as_u64)
                != Some(1)
            || !self
                .request
                .context
                .get("request")
                .is_some_and(Value::is_object)
        {
            return None;
        }
        let encoded = serde_json::to_string(self).ok()?;
        (encoded.len() <= maximum_bytes).then_some(encoded)
    }
}

#[cfg(test)]
#[path = "record_test.rs"]
mod tests;
