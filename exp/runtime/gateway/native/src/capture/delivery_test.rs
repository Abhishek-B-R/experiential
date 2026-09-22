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
fn saturated_destination_waits_without_losing_records_or_exceeding_queued_budget() {
    let (delivery, entered, resume) = paused(limits(), false);
    let delivery = Arc::new(delivery);
    assert!(delivery.submit(record("12345678")));
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "12345678"
    );
    assert!(delivery.submit(record("abcdefgh")));
    let (returned, result) = mpsc::channel();
    let producer = delivery.clone();
    let waiting = std::thread::spawn(move || returned.send(producer.submit(record("x"))).unwrap());
    assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
    assert_eq!(
        delivery.counts(),
        [2, limits().maximum_bytes as u64, 0, 0, 0]
    );
    resume.send(()).unwrap();
    assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
    waiting.join().unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "abcdefgh"
    );
    resume.send(()).unwrap();
    assert_eq!(entered.recv_timeout(Duration::from_secs(1)).unwrap(), "x");
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 3, 0, 0]);
    assert!(!delivery.submit(record("closed")));
    assert_eq!(delivery.counts(), [0, 0, 3, 0, 1]);
}

#[test]
fn record_count_is_bounded_even_for_tiny_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    let delivery = Arc::new(delivery);
    assert!(delivery.submit(record("a")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit(record("b")));
    let (returned, result) = mpsc::channel();
    let producer = delivery.clone();
    let waiting = std::thread::spawn(move || returned.send(producer.submit(record("c"))).unwrap());
    assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
    assert_eq!(
        delivery.counts(),
        [2, (record("a").heap_bytes() * 2) as u64, 0, 0, 0]
    );
    resume.send(()).unwrap();
    assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
    waiting.join().unwrap();
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
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
fn committed_write_and_cleanup_failures_have_separate_counters() {
    struct CleanupFailure;
    impl Sink for CleanupFailure {
        fn write(&mut self, _: &Record, _: usize) -> Result<(), ()> {
            Ok(())
        }
        fn take_maintenance_failures(&mut self) -> u64 {
            1
        }
        fn maintain(&mut self) -> Result<(), ()> {
            Err(())
        }
    }
    let delivery = Delivery::new(limits(), CleanupFailure).unwrap();
    assert!(delivery.submit(record("saved")));
    let until = Instant::now() + Duration::from_secs(5);
    while delivery.maintenance_failures() < 2 && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(delivery.close_until(until));
    assert_eq!(delivery.counts(), [0, 0, 1, 0, 0]);
    assert!(delivery.maintenance_failures() >= 2);
}

#[test]
fn shutdown_timeout_reports_incomplete_drain_without_purging_accepted_records() {
    let (delivery, entered, resume) = paused(limits(), false);
    let delivery = Arc::new(delivery);
    assert!(delivery.submit(record("active")));
    entered.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(delivery.submit(record("queued")));
    let (returned, result) = mpsc::channel();
    let producer = delivery.clone();
    let waiting =
        std::thread::spawn(move || returned.send(producer.submit(record("waiting"))).unwrap());
    assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
    assert!(!delivery.close_until(Instant::now()));
    assert!(!delivery.submit(record("late")));
    resume.send(()).unwrap();
    assert!(result.recv_timeout(Duration::from_secs(1)).unwrap());
    waiting.join().unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "queued"
    );
    resume.send(()).unwrap();
    assert_eq!(
        entered.recv_timeout(Duration::from_secs(1)).unwrap(),
        "waiting"
    );
    resume.send(()).unwrap();
    assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
    assert_eq!(delivery.counts(), [0, 0, 3, 0, 1]);
    assert!(entered.try_recv().is_err());
}

#[test]
fn synchronous_completion_waits_for_durable_success_and_reports_writer_failure() {
    for fail in [false, true] {
        let (delivery, entered, resume) = paused(limits(), fail);
        let delivery = Arc::new(delivery);
        let (returned, result) = mpsc::channel();
        let producer = delivery.clone();
        let waiting = std::thread::spawn(move || {
            returned
                .send(producer.submit_wait(record("ack"), None))
                .unwrap();
        });
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
        resume.send(()).unwrap();
        assert_eq!(result.recv_timeout(Duration::from_secs(1)).unwrap(), !fail);
        waiting.join().unwrap();
        assert!(delivery.close_until(Instant::now() + Duration::from_secs(1)));
        assert_eq!(delivery.counts()[4], 0);
    }
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
