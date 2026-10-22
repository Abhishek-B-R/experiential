//! The pre-first-byte budget: a chain of slow dials ends inside the budget
//! with the gateway's own retryable timeout, never past the edge's origin limit.
use super::*;

/// A rung that accepts every connection and never answers (a stalled lane).
async fn spawn_stalled_rung() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            held.push(socket);
        }
    });
    format!("http://{address}/v1/chat/completions")
}

async fn run_budgeted(
    harness: &Harness,
    route: &[DeploymentWire],
    budget: Option<Duration>,
) -> (Won, AttemptGuard) {
    let mut guard = AttemptGuard::new(
        harness.bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "request-budget".to_string(),
        Instant::now(),
    );
    let started = Instant::now();
    let context = WaterfallContext {
        pre_first_byte_deadline: budget.map(|budget| started + budget),
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "request-budget",
        raw_key: "key",
        caller_scope: None,
        route,
        policy: RoutePolicy {
            maximum_total_attempts: 4,
            maximum_same_deployment_attempts: 1,
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        },
        deadline: started + Duration::from_secs(30),
        time_to_first_byte: Duration::from_secs(2),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(120),
        approximate_input_tokens: 1.0,
        chat_logprobs: false,
        capture_logprobs: false,
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
        output_guardrails: None,
    };
    (acquire_attempt(&context, &mut guard).await, guard)
}

#[test]
fn stalled_dials_stop_at_the_budget_with_a_retryable_timeout() {
    block_on(async {
        let harness = Harness::new();
        let route = [
            wire("a", &spawn_stalled_rung().await, 0),
            wire("b", &spawn_stalled_rung().await, 0),
            wire("c", &spawn_stalled_rung().await, 0),
        ];
        let started = Instant::now();
        let (won, guard) = run_budgeted(&harness, &route, Some(Duration::from_secs(3))).await;
        let elapsed = started.elapsed();
        let Won::Failed(error) = finish(guard, won).await else {
            panic!("a fully stalled chain must fail");
        };
        // Two 2 s header windows would take 4 s and a third 6 s; the budget
        // ends the chain at 3 s instead, as our own retryable 503.
        assert!(elapsed < Duration::from_millis(3_900), "{elapsed:?}");
        assert!(elapsed >= Duration::from_millis(2_900), "{elapsed:?}");
        assert_eq!(error.status_code, 503);
        assert_eq!(error.code, "deadline_exceeded");
        assert_eq!(error.retry_after_seconds, Some(2));
    });
}

#[test]
fn without_a_budget_each_stalled_dial_gets_its_own_window() {
    block_on(async {
        let harness = Harness::new();
        let route = [
            wire("a", &spawn_stalled_rung().await, 0),
            wire("b", &spawn_stalled_rung().await, 0),
        ];
        let started = Instant::now();
        let (won, guard) = run_budgeted(&harness, &route, None).await;
        let elapsed = started.elapsed();
        assert!(matches!(finish(guard, won).await, Won::Failed(_)));
        assert!(elapsed >= Duration::from_millis(3_900), "{elapsed:?}");
    });
}

/// A rung whose authored header allowance is far longer than the budget.
fn slow_header_wire(deployment_id: &str, url: &str) -> DeploymentWire {
    let mut slow = wire(deployment_id, url, 0);
    slow.time_to_first_byte_base_seconds = Some(300.0);
    slow
}

#[test]
fn a_long_header_allowance_still_leaves_room_to_fail_over() {
    block_on(async {
        // The 2026-10-21 shape: a lane authoring 300 s of header wait ahead
        // of a live sibling. Capped at the budget minus the reserve, the
        // stall ends at 1.5 s and the sibling serves inside the budget.
        let harness = Harness::new();
        let healthy = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [
            slow_header_wire("a", &spawn_stalled_rung().await),
            wire("b", &healthy.url, 0),
        ];
        let budget =
            super::super::successor::FAILOVER_FIRST_BYTE_RESERVE + Duration::from_millis(1_500);
        let started = Instant::now();
        let (won, guard) = run_budgeted(&harness, &route, Some(budget)).await;
        let elapsed = started.elapsed();
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the header timeout must fail over to the live sibling");
        };
        assert_eq!(committed.depth, 1);
        assert!(elapsed >= Duration::from_millis(1_400), "{elapsed:?}");
        assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    });
}

/// A context over `route` whose pre-first-byte budget ends `budget` after `start`.
fn cap_context<'a>(
    harness: &'a Harness,
    route: &'a [DeploymentWire],
    start: Instant,
    budget: Duration,
    maximum_total_attempts: u32,
) -> WaterfallContext<'a> {
    WaterfallContext {
        pre_first_byte_deadline: Some(start + budget),
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "request-cap",
        raw_key: "key",
        caller_scope: None,
        route,
        policy: RoutePolicy {
            maximum_total_attempts,
            maximum_same_deployment_attempts: 1,
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        },
        deadline: start + Duration::from_secs(600),
        time_to_first_byte: Duration::from_secs(2),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(120),
        approximate_input_tokens: 1.0,
        chat_logprobs: false,
        capture_logprobs: false,
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
        output_guardrails: None,
    }
}

#[test]
fn only_a_dial_with_an_untried_successor_gives_up_its_allowance() {
    let harness = Harness::new();
    let pair = [wire("a", "http://a.test", 0), wire("b", "http://b.test", 0)];
    let single = [wire("a", "http://a.test", 0)];
    let reserve = super::super::successor::FAILOVER_FIRST_BYTE_RESERVE;
    let budget = Duration::from_secs(90);
    let now = Instant::now();
    let context = cap_context(&harness, &pair, now, budget, 8);
    // Depth 0 with depth 1 untried, and depth 1 revisiting a skipped depth
    // 0: both stop waiting for headers at the budget minus the reserve.
    let capped = Some(now + budget - reserve);
    assert_eq!(failover_first_byte_cap(&context, 0, &[1, 0], 1), capped);
    assert_eq!(failover_first_byte_cap(&context, 1, &[0, 1], 1), capped);
    // The last untried rung keeps its full allowance.
    assert_eq!(failover_first_byte_cap(&context, 1, &[1, 1], 2), None);
    // A single-rung route keeps it too.
    let context = cap_context(&harness, &single, now, budget, 8);
    assert_eq!(failover_first_byte_cap(&context, 0, &[1], 1), None);
    // No attempt left after this one: nothing to save the budget for.
    let context = cap_context(&harness, &pair, now, budget, 1);
    assert_eq!(failover_first_byte_cap(&context, 0, &[1, 0], 1), None);
    // A budget already inside the reserve is left alone, never cut to nothing.
    let context = cap_context(&harness, &pair, now, reserve - Duration::from_secs(1), 8);
    assert_eq!(failover_first_byte_cap(&context, 0, &[1, 0], 1), None);
}
