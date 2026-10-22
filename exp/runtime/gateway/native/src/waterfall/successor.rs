//! The waterfall's successor rules: whether a classified pre-commit failure
//! leaves any later dispatch possible, the wait before a throttled rung is
//! re-dialed, and how much of the pre-first-byte budget a rung's header wait
//! must leave for its successor.

use std::time::{Duration, Instant};

use super::{fallback_rules, first_byte_allowance, DeploymentWire, RoutePolicy, WaterfallContext};
use crate::errors::{Failure, FailureClass};
use crate::relay::remaining;
use crate::throttle_backoff::{jitter_unit, BackoffQuery, ThrottleRedial};

/// Whether the classified failure leaves any successor dispatch possible
/// under the rust-side facts (caps, flags, untried rungs, deadline). The
/// control plane re-checks with health and budget state and may still answer
/// with exhaustion. `dispatched` counts physical dispatches per route index:
/// a failover may land on ANY rung not yet dispatched in this request,
/// earlier skipped rungs included (`fallback_rules::successor_available`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn successor_possible(
    policy: RoutePolicy,
    route: &[DeploymentWire],
    deadline: Instant,
    total_attempts: u32,
    same_deployment_attempts: u32,
    depth: usize,
    dispatched: &[u32],
    failure: &Failure,
    refusal_eligible: bool,
) -> bool {
    if total_attempts >= policy.maximum_total_attempts || remaining(deadline).is_zero() {
        return false;
    }
    let same = failure.retryable_same_deployment
        && same_deployment_attempts < policy.maximum_same_deployment_attempts;
    // An unrestricted later rung takes the failure when the route policy
    // advances it; a failover-only rung takes it when its own set names it,
    // whatever the policy says (`fallback_rules`).
    let failover = fallback_rules::successor_available(
        route,
        depth,
        dispatched,
        failure,
        failure.failover_eligible || refusal_eligible,
    );
    same || failover
}

/// What a rung's first-byte (header) wait leaves of the pre-commit budget
/// for a successor: enough for a fresh dial to answer its headers.
pub(crate) const FAILOVER_FIRST_BYTE_RESERVE: Duration = Duration::from_secs(15);

/// The instant a dial at `depth` must stop waiting for headers so a header
/// timeout can still fail over, or `None` to keep the authored allowance.
///
/// A rung's authored first-byte allowance can exceed the whole pre-first-byte
/// budget (production, 2026-10-21: a self-hosted lane authored 300 s against
/// the 90 s budget, so every header timeout ended the request with nothing
/// left to dial). While an untried rung could take a header timeout (a
/// failover-eligible `timeout`, never redialed on the same rung) and the
/// attempt cap allows one more dispatch, the wait is capped at the pre-commit
/// deadline minus `FAILOVER_FIRST_BYTE_RESERVE`. The last untried rung (and a
/// single-rung route) keeps its full allowance, and a budget already inside
/// the reserve is left alone rather than cut to nothing. `dispatched` must
/// already count this dial; `total_attempts` likewise.
pub(super) fn failover_first_byte_cap(
    ctx: &WaterfallContext<'_>,
    depth: usize,
    dispatched: &[u32],
    total_attempts: u32,
) -> Option<Instant> {
    if !ctx.policy.permits(total_attempts, 0) {
        return None;
    }
    let stall = Failure::new(
        FailureClass::Timeout,
        "provider did not send response headers in time",
    )
    .with_retry(false, true);
    if !fallback_rules::successor_available(ctx.route, depth, dispatched, &stall, true) {
        return None;
    }
    let cap = ctx
        .pre_commit_deadline()
        .checked_sub(FAILOVER_FIRST_BYTE_RESERVE)?;
    (cap > Instant::now()).then_some(cap)
}

/// The wait before re-dialing a throttled rung, or `None` when the ladder
/// should advance instead: the pool authors no schedule, the rung's redial
/// budget on this request is spent (or zero), the failure is not a throttle,
/// the total cap is reached, or the schedule itself declines (`Retry-After`
/// beyond the ceiling, or no room under the deadline).
pub(super) fn throttle_backoff_delay(
    ctx: &WaterfallContext<'_>,
    wire: &DeploymentWire,
    failure: &Failure,
    throttle_redials_at_depth: u32,
    total_attempts: u32,
) -> Option<Duration> {
    let schedule = wire.throttle_redial.or(ctx.policy.throttle_redial)?;
    if wire.throttle_redial_budget == 0
        || failure.failure_class != FailureClass::Throttled
        || total_attempts >= ctx.policy.maximum_total_attempts
    {
        return None;
    }
    if throttle_redials_at_depth >= schedule.max_attempts.min(wire.throttle_redial_budget) {
        return None;
    }
    if let Some(backoff) = ctx.policy.backoff {
        return backoff.delay(
            throttle_redials_at_depth,
            failure.retry_after_seconds,
            remaining(ctx.deadline).saturating_sub(first_byte_allowance(
                wire,
                ctx.time_to_first_byte,
                ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
                ctx.approximate_input_tokens,
            )),
            ctx.request_id,
        );
    }
    BackoffQuery {
        // The rung's per-request budget never exceeds the schedule's cap.
        schedule: ThrottleRedial {
            max_attempts: schedule.max_attempts.min(wire.throttle_redial_budget),
            ..schedule
        },
        redials_so_far: throttle_redials_at_depth,
        retry_after_seconds: failure.retry_after_seconds,
        remaining_deadline: remaining(ctx.deadline),
        first_byte_allowance: first_byte_allowance(
            wire,
            ctx.time_to_first_byte,
            ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
            ctx.approximate_input_tokens,
        ),
        jitter_unit: jitter_unit(ctx.request_id, total_attempts),
    }
    .delay()
}
