use super::*;

struct PausedSink {
    entered: mpsc::Sender<String>,
    resume: mpsc::Receiver<()>,
    fail: bool,
}

impl Sink for PausedSink {
    fn write(&mut self, record: &Record, _maximum_bytes: usize) -> Result<(), ()> {
        self.entered
            .send(record.request.request_id.clone())
            .map_err(|_| ())?;
        self.resume
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| ())?;
        if self.fail {
            Err(())
        } else {
            Ok(())
        }
    }
}

fn paused(limits: Limits, fail: bool) -> (Delivery, mpsc::Receiver<String>, mpsc::Sender<()>) {
    let (entered, observer) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    let delivery = Delivery::new(
        limits,
        PausedSink {
            entered,
            resume: paused,
            fail,
        },
    )
    .unwrap();
    (delivery, observer, resume)
}

fn limits() -> Limits {
    Limits {
        maximum_records: 2,
        maximum_bytes: record("12345678").heap_bytes() * 2,
        maximum_record_bytes: 1024,
    }
}

fn record(id: &str) -> Record {
    serde_json::from_value(serde_json::json!({
        "schema_version":1,
        "request": {"request_id":id,
            "scope":{"organization_id":"org","identity_id":"identity","application_id":"app"},
            "protocol":"chat_completions","model_id":null,
            "context":{"schema_version":1,"request":{}}},
        "response":null,"provider_reasoning":null,"provider_reasoning_source_json":null,
        "provider_tool_calls_json":null,"deployment_id":null,"captured_at":1.0,
    }))
    .unwrap()
}

#[test]
fn saturated_destination_never_blocks_serving_or_exceeds_total_byte_budget() {
    let (delivery, entered, resume) = paused(limits(), false);
    assert!(delivery.submit(record("12345678")));
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "12345678"
    );
    assert!(delivery.submit(record("abcdefgh")));
    for _ in 0..1000 {
        assert!(!delivery.submit(record("x")));
    }
    assert_eq!(
        delivery.counts(),
        [2, limits().maximum_bytes as u64, 0, 0, 1000]
    );
    resume.send(()).unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "abcdefgh"
    );
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 2, 0, 1000]);
    assert!(!delivery.submit(record("closed")));
    assert_eq!(delivery.counts(), [0, 0, 2, 0, 1001]);
}

#[test]
fn record_count_is_bounded_even_for_tiny_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    assert!(delivery.submit(record("a")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit(record("b")));
    assert!(!delivery.submit(record("c")));
    assert_eq!(
        delivery.counts(),
        [2, (record("a").heap_bytes() * 2) as u64, 0, 0, 1]
    );
    resume.send(()).unwrap();
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
}

#[test]
fn allocated_capacity_not_just_json_length_is_charged() {
    let (delivery, _, _) = paused(limits(), false);
    let mut large_allocation = record("x");
    let mut text = String::with_capacity(limits().maximum_bytes + 1);
    text.push('x');
    large_allocation.provider_reasoning = Some(text);
    assert!(!delivery.submit(large_allocation));
    assert_eq!(delivery.counts(), [0, 0, 0, 0, 1]);
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
}

#[test]
fn failed_destination_releases_budget_and_records_no_sensitive_error() {
    let (delivery, entered, resume) = paused(limits(), true);
    assert!(delivery.submit(record("private")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 0, 1, 0]);
}

#[test]
fn shutdown_returns_while_sink_is_blocked_and_expires_queued_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    assert!(delivery.submit(record("active")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit(record("queued")));
    assert!(!delivery.close_until(Instant::now()));
    assert!(!delivery.submit(record("late")));
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 1, 0, 2]);
    assert!(entered.try_recv().is_err());
}

#[test]
fn limits_reject_zero_unbounded_and_incoherent_configuration() {
    for bounds in [
        Limits {
            maximum_records: 0,
            ..limits()
        },
        Limits {
            maximum_records: 4097,
            ..limits()
        },
        Limits {
            maximum_bytes: 7,
            ..limits()
        },
        Limits {
            maximum_bytes: usize::MAX,
            ..limits()
        },
        Limits {
            maximum_record_bytes: 0,
            ..limits()
        },
        Limits {
            maximum_record_bytes: 9 * 1024 * 1024,
            maximum_bytes: 10 * 1024 * 1024,
            ..limits()
        },
    ] {
        assert!(bounds.validate().is_err());
    }
}
