//! Borrowed database payload projection; only the final writer encodes content.
use super::record::{Protocol, Record, Response};
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;

#[derive(Serialize)]
struct Scope<'a> {
    user_id: &'a str,
    application_id: &'a str,
}

#[derive(Serialize)]
struct Output<'a> {
    response: &'a Option<Response>,
    provider_reasoning: Option<Cow<'a, str>>,
    provider_reasoning_source_json: Option<Cow<'a, str>>,
    provider_tool_calls_json: &'a Option<String>,
}

#[derive(Serialize)]
struct Request<'a> {
    exp_context: &'a Value,
    exp_capture_output: Output<'a>,
    previous_response_id: &'a Value,
}

#[derive(Serialize)]
struct Provenance<'a> {
    source_kind: &'static str,
    source_id: &'a str,
    model_id: &'a Option<String>,
    model_revision: Option<&'a str>,
    deployment_id: &'a Option<String>,
    policy_revision: Option<&'a str>,
    source_experience_ids: &'a [&'a str],
}

#[derive(Serialize)]
struct Experience<'a> {
    schema_version: u32,
    experience_id: &'a str,
    response_id: &'a str,
    episode_id: Option<&'a str>,
    parent_response_id: &'a Value,
    scope: Scope<'a>,
    protocol: Protocol,
    captured_at: f64,
    request: Request<'a>,
    response: &'a Value,
    provenance: Provenance<'a>,
    exact_tokens: Option<u64>,
}

pub(super) fn encode(
    record: &Record,
    response: &Value,
    experience_id: &str,
    maximum: usize,
) -> Option<String> {
    let context = &record.request.context;
    let scope = &record.request.scope;
    let reasoning = record.durable_reasoning().ok()?;
    let parent = &context["request"]["previous_response_id"];
    let experience = Experience {
        schema_version: 1,
        experience_id,
        response_id: response["id"].as_str()?,
        episode_id: context["request"]["metadata"]["conversation_id"]
            .as_str()
            .filter(|value| !value.trim().is_empty() && value.len() <= 512),
        parent_response_id: parent,
        scope: Scope {
            user_id: &scope.identity_id,
            application_id: &scope.application_id,
        },
        protocol: record.request.protocol,
        captured_at: record.captured_at,
        request: Request {
            exp_context: context,
            exp_capture_output: Output {
                response: &record.response,
                provider_reasoning: reasoning.text,
                provider_reasoning_source_json: reasoning.source_json,
                provider_tool_calls_json: &record.provider_tool_calls_json,
            },
            previous_response_id: parent,
        },
        response,
        provenance: Provenance {
            source_kind: "traffic",
            source_id: &record.request.request_id,
            model_id: &record.request.model_id,
            model_revision: None,
            deployment_id: &record.deployment_id,
            policy_revision: None,
            source_experience_ids: &[],
        },
        exact_tokens: None,
    };
    super::budget::encode(&experience, maximum)
}

#[cfg(test)]
#[path = "local_payload_test.rs"]
mod tests;
