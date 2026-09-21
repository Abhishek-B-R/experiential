use super::*;
use serde_json::json;
use std::cell::Cell;

fn record<R>(response: R) -> Record<R> {
    Record {
        schema_version: 1,
        request: Request {
            request_id: "request".into(),
            scope: Scope {
                organization_id: "org".into(),
                identity_id: "identity".into(),
                application_id: "alias".into(),
            },
            protocol: Protocol::ChatCompletions,
            model_id: None,
            context: json!({"schema_version":1,"request":{"messages":[]}}),
        },
        response: Some(response),
        provider_reasoning: None,
        provider_reasoning_source_json: None,
        provider_tool_calls_json: None,
        deployment_id: None,
        captured_at: 1.0,
    }
}

#[test]
fn preencoded_response_is_embedded_verbatim_without_reformatting() {
    // Whitespace inside raw JSON survives only if the envelope reuses the encoded bytes.
    let raw = r#"{"kind":"json", "status":200, "body": {"text": "雪"}, "source_json":null}"#;
    let encoded = EncodedResponse(RawValue::from_string(raw.into()).unwrap());
    assert_eq!(encoded.len(), raw.len());
    let record = record(encoded);
    let first = record.encode(4096).unwrap();
    let second = record.encode(4096).unwrap();
    assert_eq!(first, second);
    assert!(first.contains(raw));
    let decoded: Record = serde_json::from_str(&first).unwrap();
    let Response::Json { body, .. } = decoded.response.unwrap() else {
        panic!()
    };
    assert_eq!(body["text"], "雪");
    assert!(record.encode(first.len() - 1).is_none());
    assert_eq!(record.encode(first.len()).unwrap(), first);
}

struct SerializeOnce<'a>(&'a Cell<usize>);

impl Serialize for SerializeOnce<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.set(self.0.get() + 1);
        serializer.serialize_none()
    }
}

#[test]
fn exceptional_reasoning_borrows_the_record_and_serializes_the_response_once() {
    let calls = Cell::new(0);
    // The response deliberately does not implement Clone.
    let mut record = record(SerializeOnce(&calls));
    record.provider_reasoning = Some("first\0second雪".into());
    let encoded = record.encode(4096).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        record.provider_reasoning.as_deref(),
        Some("first\0second雪")
    );
    let decoded: Record = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        decoded.provider_reasoning.as_deref(),
        Some("first\u{fffd}second雪")
    );
    let source: String =
        serde_json::from_str(decoded.provider_reasoning_source_json.as_deref().unwrap()).unwrap();
    assert_eq!(source, "first\0second雪");
}
