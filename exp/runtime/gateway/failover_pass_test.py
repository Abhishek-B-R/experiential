"""Tests for the failover reservation walk: revisit, sideways spill, and the forced pass.

The route mirrors the 2026-10-21 production shape: two self-hosted lanes,
each with a refusing ``concurrency_bound``.
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_test import (
    _admit,
    _deployment,
    _RecordingLedger,
    _settle,
    _start,
)

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


def test_the_forced_pass_stops_at_the_pro_ceiling_and_keeps_the_original_failure() -> None:
    """A rung already at 1.5x its bound refuses even a failover: the 502 is the answer."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    # A third request already overflowed A as a failover (2 x 1.5 = 3 slots).
    for request_id, ordinal in (("first-failover", 1), ("second-failover", 1)):
        _admit(registry, (lane_b, lane_a), request_id=request_id)
        started = _start(registry, ordinal=0, request_id=request_id)
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
        result = _start(
            registry,
            ordinal=ordinal,
            current_depth=0,
            failure=_BAD_GATEWAY,
            request_id=request_id,
        )
        if request_id == "first-failover":
            assert result["route_depth"] == 1
        else:
            assert result["exhausted"] is True
            failure = cast("JsonObject", result["failure"])
            assert failure["failure_class"] == "provider_internal"


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
