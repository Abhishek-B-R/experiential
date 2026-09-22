use super::*;
use serde_json::json;

fn record() -> Record {
    serde_json::from_value(json!({
        "schema_version":1,"request":{"request_id":"request",
            "scope":{"organization_id":"org","identity_id":"user","application_id":"app"},
            "protocol":"responses","model_id":"model",
            "context":{"schema_version":1,"request":{"messages":[
                {"role":"user","content":"prompt 雪"},
                {"role":"tool","content":" environment ","tool_call_id":"call"}
            ],"tools":[{"name":"lookup","parameters":{"type":"object"}}],
                "previous_response_id":"parent","metadata":{"conversation_id":"episode"}}}},
        "response":{"kind":"json","status":200,"body":{
            "id":"response","status":"completed","output":[]},"source_json":null},
        "provider_reasoning":"first\0second雪", "provider_reasoning_source_json":null,
        "provider_tool_calls_json":"[{\"raw_arguments\":\"{  }\"}]",
        "deployment_id":"deployment","captured_at":1.0
    }))
    .unwrap()
}

#[test]
fn borrowed_payload_preserves_schema_content_sidecars_and_exact_limit() {
    let record = record();
    let response = super::super::projection::completed_response(&record).unwrap();
    assert!(matches!(response, Cow::Borrowed(_)));
    let payload = encode(&record, &response, "experience-id", 8192).unwrap();
    let actual: Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        actual,
        json!({
            "schema_version":1,"experience_id":"experience-id","response_id":"response",
            "episode_id":"episode","parent_response_id":"parent",
            "scope":{"user_id":"user","application_id":"app"},
            "protocol":"responses","captured_at":1.0,
            "request":{
                "exp_context":record.request.context,
                "exp_capture_output":{
                    "response":record.response,"provider_reasoning":"first�second雪",
                    "provider_reasoning_source_json":"\"first\\u0000second雪\"",
                    "provider_tool_calls_json":record.provider_tool_calls_json,
                },"previous_response_id":"parent"
            },"response":response,
            "provenance":{"source_kind":"traffic","source_id":"request","model_id":"model",
                "model_revision":null,"deployment_id":"deployment","policy_revision":null,
                "source_experience_ids":[]},"exact_tokens":null,
        })
    );
    assert_eq!(
        encode(&record, &response, "experience-id", payload.len()),
        Some(payload.clone())
    );
    assert!(encode(&record, &response, "experience-id", payload.len() - 1).is_none());
    assert_eq!(
        record.provider_reasoning.as_deref(),
        Some("first\0second雪")
    );
}

#[test]
fn completed_responses_event_is_borrowed_from_retained_frames() {
    let mut record = record();
    record.response = Some(Response::Sse {
        status: 200,
        frames: vec![json!({"type":"response.completed", "response":{
            "id":"response","status":"completed","output":[]}})],
        truncated: false,
        client_disconnected: false,
        source_json: None,
    });
    let response = super::super::projection::completed_response(&record).unwrap();
    assert!(matches!(response, Cow::Borrowed(_)));
    assert_eq!(response["id"], "response");
}
