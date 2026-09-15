//! Azure annotation-only frames share the compatible content and finish lifecycle.

use crate::dialects::{drain_stream_fixture, Dialect, Normalizer};
use crate::errors::FailureClass;
use crate::events::Event;
use crate::sse::SseEvent;
use serde_json::{json, Value};

fn frame(choice: Value) -> SseEvent {
    SseEvent {
        event: None,
        data: json!({"choices": [choice]}).to_string(),
    }
}

fn annotation(finish: Option<&str>) -> Value {
    let filtered = finish == Some("content_filter");
    json!({
        "index": 0, "finish_reason": finish,
        "content_filter_results": {
            "hate": {"filtered": filtered, "severity": if filtered { "high" } else { "safe" }}
        },
        "content_filter_offsets": {"check_offset": 49, "start_offset": 47, "end_offset": 49}
    })
}

#[test]
fn azure_annotations_preserve_text_finish_and_trailing_usage() {
    for after_stop in [false, true] {
        let mut frames = vec![
            json!({"choices": [], "prompt_filter_results": []}),
            json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            json!({"choices": [annotation(None)]}),
            json!({"choices": [{"index": 0, "delta": {"content": "OK"}}]}),
        ];
        let stop = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        let annotation = json!({"choices": [annotation(None)]});
        frames.extend(if after_stop {
            vec![stop, annotation]
        } else {
            vec![annotation, stop]
        });
        frames.push(json!({"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14}}));
        let wire = frames
            .iter()
            .map(|value| format!("data: {value}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n";
        // Decode across arbitrary transport boundaries, including inside JSON.
        let chunks = wire
            .as_bytes()
            .chunks(7)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &chunks);
        assert!(failure.is_none(), "{failure:?}");
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "OK"}),
                json!({"kind": "usage", "input_tokens": 13, "output_tokens": 1, "cached_input_tokens": null, "reasoning_tokens": null}),
                json!({"kind": "completed"}),
            ]
        );
    }
}

#[test]
fn azure_filter_annotation_preserves_refusal_instead_of_completion() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let events = normalizer
        .feed(&frame(annotation(Some("content_filter"))))
        .unwrap();
    assert!(matches!(events.as_slice(), [Event::RefusalDelta(_)]));
    normalizer.feed(&frame(annotation(None))).unwrap();
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".into(),
        })
        .unwrap();
    assert!(
        matches!(events.as_slice(), [Event::Failed(failure)] if failure.failure_class == FailureClass::Refusal)
    );
}

#[test]
fn compatible_missing_or_invalid_deltas_still_fail_without_valid_annotations() {
    let mut cases = vec![
        json!({"index": 0}),
        json!({"index": 0, "finish_reason": "stop"}),
    ];
    for delta in [Value::Null, json!("bad"), json!([]), json!(0)] {
        let mut choice = annotation(None);
        choice["delta"] = delta;
        cases.push(choice);
    }
    for field in ["content_filter_results", "content_filter_offsets"] {
        for value in [Value::Null, json!("bad"), json!([])] {
            let mut choice = annotation(None);
            choice[field] = value;
            cases.push(choice);
        }
        let mut choice = annotation(None);
        choice.as_object_mut().unwrap().remove(field);
        cases.push(choice);
    }
    for choice in cases {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        assert!(
            normalizer.feed(&frame(choice.clone())).is_err(),
            "accepted {choice}"
        );
    }
}

/// An OpenAI-compatible lane that answers HTTP 200 with an error-shaped body
/// (no `choices`) declares its failure in whichever envelope spelling it
/// uses; each reaches the ledger with its sentence instead of dying as a
/// malformed "choices must be an array" frame.
#[test]
fn error_shaped_success_frames_declare_the_provider_failure_with_detail() {
    for (payload, expected_class, expected_detail) in [
        (
            json!({"object": "error", "message": "Tool 'g' not found in tools list.",
                   "type": "BadRequestError", "param": null, "code": 400}),
            FailureClass::InvalidRequest,
            "400: Tool 'g' not found in tools list.",
        ),
        (
            json!({"code": "invalid-argument", "error": "Argument not supported on this model: presencePenalty"}),
            FailureClass::InvalidRequest,
            "invalid-argument: Argument not supported on this model: presencePenalty",
        ),
        (
            json!({"code": 400, "reason": "INVALID_PARAMETER",
                   "message": "tools is not supported by this model", "metadata": {}}),
            FailureClass::InvalidRequest,
            "INVALID_PARAMETER: tools is not supported by this model",
        ),
        (
            json!({"detail": "An image input is required for this model."}),
            FailureClass::ProviderInternal,
            "An image input is required for this model.",
        ),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        let events = normalizer
            .feed(&SseEvent {
                event: None,
                data: payload.to_string(),
            })
            .expect("an error-shaped frame is a declared failure, never malformed");
        let failure = match events.as_slice() {
            [Event::Failed(failure)] => failure,
            other => panic!("expected one failed terminal for {payload}, got {other:?}"),
        };
        assert_eq!(failure.failure_class, expected_class, "{payload}");
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some(expected_detail),
            "{payload}"
        );
    }
}
