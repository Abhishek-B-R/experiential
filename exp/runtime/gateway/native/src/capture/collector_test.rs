use super::*;
use crate::capture::record::{Protocol, Response, Scope};
use serde_json::json;
use std::sync::{mpsc, Arc};

struct MemorySink(mpsc::Sender<Record>);

impl Sink for MemorySink {
    fn write(&mut self, record: &Record, maximum_bytes: usize) -> Result<(), ()> {
        let encoded = record.encode(maximum_bytes).ok_or(())?;
        self.0
            .send(serde_json::from_str(&encoded).map_err(|_| ())?)
            .map_err(|_| ())
    }
}

fn config() -> Configuration {
    Configuration {
        delivery: Limits {
            maximum_records: 8,
            maximum_bytes: 65536,
            maximum_record_bytes: 8192,
        },
        maximum_pending_records: 4,
        maximum_pending_bytes: 65536,
        maximum_request_bytes: 4096,
        maximum_response_bytes: 4096,
        ttl_seconds: 30,
        settlement_required: true,
    }
}

fn request(id: &str) -> Request {
    Request {
        request_id: id.to_owned(),
        scope: Scope {
            organization_id: "org".into(),
            identity_id: "identity".into(),
            application_id: "alias".into(),
        },
        protocol: Protocol::ChatCompletions,
        model_id: Some("model".into()),
        context: Arc::new(
            json!({"schema_version":1,"request":{"messages":[{"role":"user","content":"task"}],"tools":[{"name":"search"}]}}),
        ),
    }
}

fn response() -> Response {
    Response::Json {
        status: 200,
        body: json!({"id":"completion","choices":[]}),
        source_json: None,
    }
}

fn collector(config: Configuration) -> (Arc<Collector>, mpsc::Receiver<Record>) {
    let (sender, receiver) = mpsc::channel();
    (
        Arc::new(Collector::new(config, MemorySink(sender)).unwrap()),
        receiver,
    )
}

fn drain(collector: &Collector, receiver: mpsc::Receiver<Record>) -> Vec<Record> {
    assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
    receiver.try_iter().collect()
}

#[test]
fn routing_provenance_is_optional_until_selected_and_then_immutable() {
    let (collector, receiver) = collector(config());
    let mut input = request("request");
    input.model_id = None;
    assert!(collector.begin(input.clone()));
    collector.select_model("request", "selected");
    collector.select_model("request", "replacement");
    collector.settle("request", true, false);
    input.request_id = "rejected".into();
    assert!(collector.begin(input));
    collector.settle("rejected", true, false);
    let records = drain(&collector, receiver);
    assert_eq!(records[0].request.model_id.as_deref(), Some("selected"));
    assert_eq!(records[1].request.model_id, None);
}

#[test]
fn collector_forwards_destination_cleanup_failure_without_losing_write_success() {
    struct CleanupFailure;
    impl Sink for CleanupFailure {
        fn write(&mut self, _: &str) -> Result<(), ()> {
            Ok(())
        }
        fn take_maintenance_failures(&mut self) -> u64 {
            1
        }
    }
    let collector = Collector::new(config(), CleanupFailure).unwrap();
    assert!(collector.begin(request("saved")));
    collector.settle("saved", true, false);
    assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(collector.counts(), [0, 0, 1, 0, 0, 0]);
    assert_eq!(collector.maintenance_failures(), 1);
}

#[test]
fn selected_model_uses_cached_request_size_including_json_escapes() {
    let mut configuration = config();
    configuration.maximum_request_bytes = 1024;
    let (collector, receiver) = collector(configuration);
    let mut input = request("request");
    input.model_id = None;
    assert!(collector.begin(input));
    collector.select_model("request", "snow-雪-\"quoted\"");
    {
        let pending = collector.pending.lock().unwrap();
        let entry = &pending.entries["request"];
        assert_eq!(
            entry.request_bytes,
            serde_json::to_string(&entry.record.request).unwrap().len()
        );
        assert_eq!(
            entry.bytes,
            entry.record.heap_bytes() + "request".len() + 512
        );
    }
    collector.settle("request", true, false);
    let mut input = request("overflow");
    input.model_id = None;
    assert!(collector.begin(input));
    collector.select_model("overflow", &"\"".repeat(512));
    collector.settle("overflow", true, false);
    assert_eq!(drain(&collector, receiver).len(), 1);
    assert_eq!(collector.counts()[5], 1);
}

#[test]
fn structured_response_size_is_enforced_at_collector_boundary() {
    let mut configuration = config();
    let exact = response().json_bytes();
    configuration.maximum_response_bytes = exact;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, true);
    assert_eq!(drain(&collector, receiver).len(), 1);

    let mut configuration = config();
    configuration.maximum_response_bytes = exact - 1;
    let (collector, receiver) = self::collector(configuration);
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, true);
    assert!(drain(&collector, receiver).is_empty());
    assert_eq!(collector.counts()[5], 1);
}

#[test]
fn idle_maintenance_expires_pending_content_without_another_request() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector
        .pending
        .lock()
        .unwrap()
        .entries
        .get_mut("request")
        .unwrap()
        .expires = Instant::now();
    let until = Instant::now() + Duration::from_secs(3);
    while !collector.pending.lock().unwrap().entries.is_empty() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(collector.pending.lock().unwrap().entries.is_empty());
    assert_eq!(collector.counts()[5], 1);
    assert!(drain(&collector, receiver).is_empty());
}

#[test]
fn response_before_settlement_is_held_and_byok_never_writes_any_content() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    assert!(collector.attach("request"));
    collector.finish("request", Some(response()), Some("deployment".into()));
    assert!(receiver.try_recv().is_err());
    collector.settle("request", false, false);
    assert!(drain(&collector, receiver).is_empty());
}

#[test]
fn settlement_before_response_captures_prompt_then_one_response_update() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.settle("request", true, true);
    assert!(collector.attach("request"));
    assert!(!collector.attach("request"));
    collector.finish("request", Some(response()), Some("deployment".into()));
    collector.finish("request", Some(response()), None);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    assert!(records[0].response.is_none());
    assert!(records[1].response.is_some());
    assert_eq!(records[1].request.scope.identity_id, "identity");
    assert_eq!(records[1].deployment_id.as_deref(), Some("deployment"));
}

#[test]
fn response_before_eligible_settlement_emits_one_complete_record() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_some());
}

#[test]
fn failed_host_requests_keep_only_the_permitted_prompt() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    collector.settle("request", true, false);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 1);
    assert!(records[0].response.is_none());
}

#[test]
fn provider_reasoning_is_lossless_bounded_and_requires_response_eligibility() {
    let (collector, receiver) = collector(config());
    assert!(collector.begin(request("kept")));
    collector.reasoning("kept", "first\0");
    collector.reasoning("kept", "second雪");
    collector.finish("kept", Some(response()), None);
    collector.settle("kept", true, true);
    assert!(collector.begin(request("discarded")));
    collector.reasoning("discarded", "must not persist");
    collector.settle("discarded", true, false);
    assert!(collector.begin(request("overflow")));
    collector.reasoning("overflow", &"x".repeat(4097));
    collector.finish("overflow", Some(response()), None);
    collector.settle("overflow", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    let restored: String =
        serde_json::from_str(records[0].provider_reasoning_source_json.as_ref().unwrap()).unwrap();
    assert_eq!(restored, "first\0second雪");
    assert!(records[1].provider_reasoning.is_none());
}

#[test]
fn raw_tool_arguments_survive_projection_but_never_response_denial() {
    let (collector, receiver) = collector(config());
    let mut call = crate::events::CompletedToolCall {
        call_id: "call-1".into(),
        name: "lookup".into(),
        namespace: None,
        caller: None,
        provider_item_id: None,
        provider_status: None,
        raw_arguments: "{  \"x\" : \"雪\"  }".into(),
        custom: false,
    };
    assert!(collector.begin(request("kept")));
    collector.tool_call("kept", &call);
    call.call_id = "call-2".into();
    call.raw_arguments = "freeform\0text".into();
    call.custom = true;
    collector.tool_call("kept", &call);
    collector.finish("kept", Some(response()), None);
    collector.settle("kept", true, true);
    assert!(collector.begin(request("denied")));
    collector.tool_call("denied", &call);
    collector.settle("denied", true, false);
    assert!(collector.begin(request("overflow")));
    call.raw_arguments = "x".repeat(4096);
    collector.tool_call("overflow", &call);
    collector.settle("overflow", true, true);
    let records = drain(&collector, receiver);
    assert_eq!(records.len(), 2);
    let calls: serde_json::Value =
        serde_json::from_str(records[0].provider_tool_calls_json.as_ref().unwrap()).unwrap();
    assert_eq!(calls[0]["raw_arguments"], "{  \"x\" : \"雪\"  }");
    assert_eq!(calls[1]["raw_arguments"], "freeform\0text");
    assert!(records[1].provider_tool_calls_json.is_none());
}

#[test]
fn unscoped_unknown_duplicate_and_expired_content_is_not_persisted() {
    let (collector, receiver) = collector(config());
    let mut invalid = request("invalid");
    invalid.scope.identity_id.clear();
    assert!(!collector.begin(invalid));
    assert!(collector.begin(request("request")));
    assert!(!collector.begin(request("request")));
    collector
        .pending
        .lock()
        .unwrap()
        .entries
        .get_mut("request")
        .unwrap()
        .expires = Instant::now();
    collector.settle("request", true, true);
    collector.finish("unknown", Some(response()), None);
    assert!(drain(&collector, receiver).is_empty());
    assert_eq!(collector.counts()[5], 3);
}

#[test]
fn pending_count_and_bytes_are_bounded_without_evicting_other_live_requests() {
    let mut configuration = config();
    configuration.maximum_pending_records = 1;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("first")));
    assert!(!collector.begin(request("overflow")));
    collector.settle("first", true, false);
    assert!(collector.begin(request("next")));
    collector.settle("next", false, false);
    assert_eq!(drain(&collector, receiver).len(), 1);
}

#[test]
fn local_capture_does_not_require_hosted_settlement() {
    let mut configuration = config();
    configuration.settlement_required = false;
    let (collector, receiver) = collector(configuration);
    assert!(collector.begin(request("request")));
    collector.finish("request", Some(response()), None);
    assert_eq!(drain(&collector, receiver).len(), 1);
}
