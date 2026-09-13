"""Per-reservation dispatch-policy decisions for the native waterfall.

The accounting bridge reserves every physical dispatch immediately before
network work; these helpers make the two policy decisions it needs at that
moment without owning state of their own. ``reserve_rung_slot`` asks the
worker's load registry whether a policy-bounded rung admits the dispatch or
sheds it sideways, folding in the affinity pool's warm-session standing.
``failed_dispatch_candidate`` turns a classified failure into the ladder's
next candidate, reading the requesting organization's observed cached
fraction on the failed rung so a pool authoring ``throttle_cache_threshold``
can dispose of a throttle by the cache actually at stake, and honoring a
post-backoff redial under an authored ``throttle_redial`` schedule.
``throttle_backoff_eligibility`` applies the same cache-stakes gate at
admission, per rung, so the data plane knows which rungs are worth waiting
for before the first throttle arrives.
"""

from __future__ import annotations

import logging

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import (
    THROTTLE_BACKOFF,
    THROTTLE_FAILOVER_COLD,
    InflightRequest,
    ThrottleDisposition,
    next_route_candidate,
    rung_load_key,
    throttle_disposition,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry

_logger = logging.getLogger(__name__)


def reserve_rung_slot(
    loads: RungLoadRegistry,
    sticky: StickySpillRegistry,
    entry: InflightRequest,
    deployment: ExactModelDeployment,
    *,
    reserved_tokens: int,
    force: bool,
) -> str | RungShed | None:
    """Reserve one policy-bounded slot on a rung, or report the shed.

    Args:
        loads: The worker's per-rung in-flight and rate-window registry.
        sticky: The worker-local conversation-to-rung bindings.
        entry: The owning in-flight request (organization and weight).
        deployment: The claimed rung about to dispatch.
        reserved_tokens: Worst-case tokens this dispatch reserves, counted
            against the rung's token window when one is authored.
        force: Admit past every policy limit because no other rung can
            serve.

    Returns:
        An opaque reservation ticket, the shed disclosure, or ``None``
        when the rung authors no admission policy (the untouched default).
    """
    policy = deployment.gateway.dispatch
    if policy is None or (
        policy.concurrency_bound is None
        and policy.requests_per_minute is None
        and policy.tokens_per_minute is None
    ):
        return None
    # Warm standing: the request's affinity fingerprint holds a live sticky
    # binding on THIS rung, so its provider cache lives here and the
    # fresh-session early threshold does not apply to it. The early threshold
    # only exists on affinity pools AND for requests that carry a fingerprint
    # (chat/Responses admission): a surface with no session concept
    # (embeddings, images) must never be classed fresh wholesale.
    fresh_fraction = (
        policy.fresh_session_spill_fraction
        if entry.route.snapshot.failover_mode == "maximize_cache_affinity"
        and entry.affinity_fingerprint is not None
        else None
    )
    warm_session = True
    if fresh_fraction is not None and entry.affinity_fingerprint is not None:
        warm_session = sticky.bound_deployment(entry.affinity_fingerprint) == (
            deployment.deployment_id
        )
    result = loads.reserve(
        rung_load_key(deployment),
        organization_id=entry.authorization.organization_id,
        weight=entry.authorization.fair_share_weight,
        bound=policy.concurrency_bound,
        fair_share=policy.fair_share,
        requests_per_minute=policy.requests_per_minute,
        tokens_per_minute=policy.tokens_per_minute,
        cache_priority_alpha=policy.cache_priority_alpha,
        reserved_tokens=reserved_tokens if policy.tokens_per_minute is not None else 0,
        warm_session=warm_session,
        fresh_spill_fraction=fresh_fraction,
        force=force,
    )
    if isinstance(result, RungShed) and result.reason == "rate_limit":
        _logger.debug(
            "gateway rate-limit shed on deployment %r (learned ceiling %s/min)",
            deployment.deployment_id,
            result.learned_requests_per_minute,
        )
    return result


def failed_dispatch_candidate(
    *,
    health: DeploymentHealthRegistry,
    loads: RungLoadRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    entry: InflightRequest,
    failure: GatewayFailure,
    current_depth: int,
    throttle_backoff: bool = False,
) -> tuple[int | None, ThrottleDisposition | None]:
    """Choose the ladder's next candidate after one classified failure.

    Reads the cache at stake on the failed rung (the requesting organization's
    EWMA of its settled cached fraction there, zero without evidence) and
    hands it with the pool's authored ``throttle_cache_threshold`` to the
    frozen candidate policy, so a throttle is surfaced or failed over by the
    warm cache it would abandon. Without a threshold the fraction is inert.
    On a pool authoring ``throttle_redial`` a throttle instead redials the
    warm rung when the data plane has waited the backoff, advances cold once
    the redials are spent, and the disposition names which happened.

    Args:
        health: Revision-isolated circuit and throttle registry.
        loads: The worker's per-rung load registry holding the cache EWMA.
        keys: One health key per ordered route deployment.
        entry: The owning in-flight request.
        failure: The classified failure that ended the previous dispatch.
        current_depth: Route position of the failed dispatch.
        throttle_backoff: Whether the data plane waited the pool's backoff
            and asks to redial the throttled rung.

    Returns:
        ``(candidate, disposition)``: the claimed route index or ``None``
        when the ladder is exhausted, and the throttle disposition when the
        failure was a throttle on a threshold- or schedule-authoring pool
        (else ``None``).
    """
    route = entry.route
    threshold = route.snapshot.throttle_cache_threshold
    redial = route.snapshot.throttle_redial
    deployment = route.deployments[current_depth]
    cached_fraction = loads.cached_fraction(
        rung_load_key(deployment), entry.authorization.organization_id
    )
    candidate = next_route_candidate(
        health=health,
        keys=keys,
        failure=failure,
        current_depth=current_depth,
        attempt_counts=entry.attempt_counts,
        total_attempts=entry.total_attempts,
        refusal_failover=entry.authorization.refusal_failover,
        failover_mode=route.snapshot.failover_mode,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
        throttle_redial=redial,
        throttle_backoff=throttle_backoff,
        throttle_redials_so_far=entry.throttle_redials[current_depth],
    )
    disposition = throttle_disposition(
        failure,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
    )
    if redial is not None and failure.failure_class == GatewayFailureClass.THROTTLED:
        # With a schedule authored a throttle never surfaces mid-ladder: it
        # either redials the warm rung or advances cold past it, and an
        # exhausted ladder is a plain exhausted throttle.
        if candidate == current_depth:
            disposition = THROTTLE_BACKOFF
        elif candidate is not None:
            disposition = THROTTLE_FAILOVER_COLD
        else:
            disposition = None
    if disposition is not None:
        _logger.debug(
            "gateway throttle on deployment %r disposed %s (cached fraction %.3f, threshold %s)",
            deployment.deployment_id,
            disposition,
            cached_fraction,
            threshold,
        )
    return candidate, disposition


def throttle_backoff_eligibility(
    loads: RungLoadRegistry,
    route: GatewayRoute,
    organization_id: str,
) -> tuple[bool, ...]:
    """Decide, per rung, whether a throttle there is worth backing off for.

    Read once at admission so the data plane knows before the first throttle
    which rungs to redial with backoff and which to fail over cold at once.
    Every rung is ineligible on a pool without a ``throttle_redial``
    schedule (the historical failover-only throttle). With a schedule and no
    ``throttle_cache_threshold`` every rung is eligible: the operator asked
    for backoff on this pool. With both, exactly the rungs where the
    requesting organization's observed cached fraction meets the threshold
    are eligible, the same cache-stakes gate that would otherwise surface
    the throttle, now meaning "wait here" instead of "return the 429". The
    fraction is the admission-time EWMA, at most seconds older than the
    reading a failure-time decision would take.

    Args:
        loads: The worker's per-rung load registry holding the cache EWMA.
        route: The resolved ordered route about to be admitted.
        organization_id: The requesting organization.

    Returns:
        One flag per route deployment, in route order.
    """
    snapshot = route.snapshot
    if snapshot.throttle_redial is None:
        return tuple(False for _ in route.deployments)
    threshold = snapshot.throttle_cache_threshold
    if threshold is None:
        return tuple(True for _ in route.deployments)
    return tuple(
        loads.cached_fraction(rung_load_key(deployment), organization_id) >= threshold
        for deployment in route.deployments
    )
