"""Tests for the failover reservation walk: revisit, sideways spill, and the forced pass.

The route mirrors the 2026-10-21 production shape: two self-hosted lanes,
each with a refusing ``concurrency_bound``.
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_test import (
    _admit,
    _deployment,
    _RecordingLedger,
    _settle,
    _start,
)
from exp.runtime.gateway.native_execution import deployment_health_key

_BAD_GATEWAY: JsonObject = {
    "failure_class": "provider_internal",
    "safe_message": "provider service failed; retry after a short delay",
    "retryable_same_deployment": True,
    "failover_eligible": True,
}


def _refusing_pair(
    bound_a: int = 2, bound_b: int = 8
) -> tuple[ExactModelDeployment, ExactModelDeployment]:
    """Two lanes authoring refusing bounds (``saturation="refuse"``)."""
    return (
        _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=bound_a, saturation="refuse"),
        ),
        _deployment(
            "deployment-b",
            connection_sha256="c" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=bound_b, saturation="refuse"),
        ),
    )


def _fill(
    registry: NativeAttemptAccounting,
    route: tuple[ExactModelDeployment, ...],
    count: int,
    prefix: str,
) -> None:
    """Hold ``count`` in-flight dispatches on the route's lead rung."""
    for index in range(count):
        request_id = f"{prefix}-{index}"
        _admit(registry, route, request_id=request_id)
        assert _start(registry, ordinal=0, request_id=request_id)["route_depth"] == 0


def test_a_failover_onto_a_full_refusing_rung_is_admitted_as_a_forced_pass() -> None:
    """B fails at depth 0 and A is full: A is forced past its bound, not the original 502."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    _admit(registry, (lane_b, lane_a), request_id="failing")
    first = _start(registry, ordinal=0, request_id="failing")
    assert first["route_depth"] == 0
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id="failing",
    )
    # The retryable 502 would redial B; one dispatch per rung keeps the test on failover.
    entry = registry.entry("failing")
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    failover = _start(
        registry, ordinal=1, current_depth=0, failure=_BAD_GATEWAY, request_id="failing"
    )
    assert failover.get("exhausted") is not True, failover
    assert failover["route_depth"] == 1
    assert ledger.started[-1]["deployment_id"] == "deployment-a"
    assert ledger.started[-1]["dispatch_reason"] == "saturated_overflow"


def test_a_first_dial_onto_the_same_full_rung_is_still_refused() -> None:
    """The forced pass is a failover's alone: a fresh request keeps the 429."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, _lane_b = _refusing_pair()
    _fill(registry, (lane_a,), 2, "occupied")
    _admit(registry, (lane_a,), request_id="fresh")
    refused = _start(registry, ordinal=0, request_id="fresh")
    assert refused["exhausted"] is True
    assert cast("JsonObject", refused["failure"])["failure_class"] == "throttled"


def _fail_over(registry: NativeAttemptAccounting, request_id: str) -> JsonObject:
    """Start ``request_id`` on B (route B, A), fail it with a 502, and reserve its failover."""
    started = _start(registry, ordinal=0, request_id=request_id)
    assert started["route_depth"] == 0
    _settle(
        registry,
        attempt_id=str(started["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id=request_id,
    )
    entry = registry.entry(request_id)
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    return _start(registry, ordinal=1, current_depth=0, failure=_BAD_GATEWAY, request_id=request_id)


def test_forced_failover_passes_are_capped_and_then_refused_with_a_retryable_429() -> None:
    """Half the bound of forced passes may ride on A; past that, a fast 429, never the 502."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair(bound_a=4)
    _fill(registry, (lane_a, lane_b), 4, "occupied")
    for request_id in ("failover-1", "failover-2", "failover-3"):
        _admit(registry, (lane_b, lane_a), request_id=request_id)
    # ceil(4 x 0.5) = 2 forced failover passes on top of A's bound of 4.
    assert _fail_over(registry, "failover-1")["route_depth"] == 1
    second = _fail_over(registry, "failover-2")
    assert second["route_depth"] == 1
    refused = _fail_over(registry, "failover-3")
    assert refused["exhausted"] is True
    failure = cast("JsonObject", refused["failure"])
    assert failure["failure_class"] == "throttled"
    assert failure["retry_after_seconds"] == 5
    assert registry.rung_admission_counters()[2] == 1
    # A settled forced pass frees its slot under the cap for the next failover.
    _settle(
        registry,
        attempt_id=str(second["attempt_id"]),
        outcome="completed",
        finalize=True,
        request_id="failover-2",
    )
    _admit(registry, (lane_b, lane_a), request_id="failover-4")
    assert _fail_over(registry, "failover-4")["route_depth"] == 1


def test_a_failover_with_no_untried_rung_keeps_the_original_failure() -> None:
    """Nothing untried and nothing shed: the dead rung's own failure is the answer."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    _lane_a, lane_b = _refusing_pair()
    _admit(registry, (lane_b,), request_id="alone")
    exhausted = _fail_over(registry, "alone")
    assert exhausted["exhausted"] is True
    assert cast("JsonObject", exhausted["failure"])["failure_class"] == "provider_internal"


def test_an_attempt_dialed_past_an_ejected_lane_discloses_lane_ejected() -> None:
    """The ejected lead is ordered last; the attempt that replaced it names it."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    dead = _deployment("mp-dead", connection_sha256="b" * 64)
    live = _deployment("mp-live", connection_sha256="c" * 64)
    entry = _admit(registry, (dead, live), request_id="past-ejected")
    dead_key = deployment_health_key(entry.authorization, dead)
    for _ in range(3):
        registry.health.failed(
            dead_key,
            GatewayFailure(failure_class=GatewayFailureClass.TRANSPORT, safe_message="refused"),
        )
    started = _start(registry, ordinal=0, request_id="past-ejected")
    assert started["route_depth"] == 1
    assert ledger.started[-1]["dispatch_reason"] == "lane_ejected"
    assert ledger.started[-1]["preferred_deployment_id"] == "mp-dead"


def test_a_rung_skipped_at_the_first_dial_is_revisited_when_its_sibling_fails() -> None:
    """A is full so the request starts on B; B's 502 sends it back to A, forced."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    _admit(registry, (lane_a, lane_b), request_id="skipped")
    first = _start(registry, ordinal=0, request_id="skipped")
    assert first["route_depth"] == 1
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id="skipped",
    )
    entry = registry.entry("skipped")
    assert entry is not None
    entry.ordinary_attempt_counts[1] = 2
    revisit = _start(
        registry, ordinal=1, current_depth=1, failure=_BAD_GATEWAY, request_id="skipped"
    )
    assert revisit["route_depth"] == 0
    assert ledger.started[-1]["deployment_id"] == "deployment-a"


def test_a_failover_spills_sideways_before_it_forces_anything() -> None:
    """With a free untried rung left, a full one is skipped, never forced."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    lane_c = _deployment("deployment-c", connection_sha256="e" * 64)
    _fill(registry, (lane_a,), 2, "occupied")
    _admit(registry, (lane_b, lane_a, lane_c), request_id="spill")
    first = _start(registry, ordinal=0, request_id="spill")
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id="spill",
    )
    entry = registry.entry("spill")
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    spilled = _start(registry, ordinal=1, current_depth=0, failure=_BAD_GATEWAY, request_id="spill")
    assert spilled["route_depth"] == 2
    assert ledger.started[-1]["dispatch_reason"] == "queue_bound"
