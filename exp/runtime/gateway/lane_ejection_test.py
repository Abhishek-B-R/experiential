"""Tests for per-worker, per-lane dead-box ejection."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayFailureClass
from exp.runtime.gateway.lane_ejection import (
    EJECTION_COOLDOWN_ENV,
    EJECTION_FAILURES_ENV,
    EJECTION_WINDOW_ENV,
    LaneEjectionPolicy,
    LaneEjections,
)

_LANE = ("deployment-a", "connection-a")
_OTHER = ("deployment-b", "connection-b")
_DEAD = GatewayFailureClass.PROVIDER_INTERNAL


def _ejected_after_three(tracker: LaneEjections, lane: tuple[str, str], start: float) -> None:
    """Feed three dead-lane failures one second apart starting at ``start``."""
    for offset in (0.0, 1.0, 2.0):
        tracker.failed(lane, _DEAD, start + offset)


def test_policy_reads_operator_overrides_and_keeps_defaults() -> None:
    """Unset variables keep the defaults; set ones replace them; bad ones fail loudly."""
    assert LaneEjectionPolicy.from_environ({}) == LaneEjectionPolicy()
    policy = LaneEjectionPolicy.from_environ(
        {EJECTION_FAILURES_ENV: "5", EJECTION_WINDOW_ENV: "4.5", EJECTION_COOLDOWN_ENV: "60"}
    )
    assert policy == LaneEjectionPolicy(failures=5, window_seconds=4.5, cooldown_seconds=60.0)
    with pytest.raises(ValueError):
        LaneEjectionPolicy.from_environ({EJECTION_COOLDOWN_ENV: "0"})


def test_three_dead_lane_failures_inside_the_window_eject_for_the_cooldown() -> None:
    """Refused or 5xx three times in ten seconds: ordered last for thirty seconds."""
    tracker = LaneEjections(LaneEjectionPolicy())
    tracker.failed(_LANE, GatewayFailureClass.TRANSPORT, 0.0)
    tracker.failed(_LANE, _DEAD, 1.0)
    assert not tracker.ejected(_LANE, 1.5)
    tracker.failed(_LANE, _DEAD, 2.0)
    assert tracker.ejected(_LANE, 2.0)
    assert tracker.ejected(_LANE, 31.9)
    # Per lane: the sibling on the same worker is untouched.
    assert not tracker.ejected(_OTHER, 2.0)


def test_failures_spread_past_the_window_never_eject() -> None:
    """Three failures over more than ten seconds are not a dead box."""
    tracker = LaneEjections(LaneEjectionPolicy())
    for at in (0.0, 6.0, 12.0):
        tracker.failed(_LANE, _DEAD, at)
    assert not tracker.ejected(_LANE, 12.0)
    # The sliding streak still ejects once three land inside the window.
    tracker.failed(_LANE, _DEAD, 13.0)
    assert tracker.ejected(_LANE, 13.0)


@pytest.mark.parametrize(
    "neutral",
    [
        GatewayFailureClass.INVALID_REQUEST,
        GatewayFailureClass.THROTTLED,
        GatewayFailureClass.TIMEOUT,
    ],
)
def test_client_errors_throttles_and_timeouts_neither_count_nor_reset(
    neutral: GatewayFailureClass,
) -> None:
    """A 4xx, a 429, or a timeout says nothing about the box: the streak is unchanged."""
    tracker = LaneEjections(LaneEjectionPolicy())
    for _ in range(5):
        tracker.failed(_LANE, neutral, 0.0)
    assert not tracker.ejected(_LANE, 0.0)
    tracker.failed(_LANE, _DEAD, 0.0)
    tracker.failed(_LANE, _DEAD, 1.0)
    tracker.failed(_LANE, neutral, 1.5)
    tracker.failed(_LANE, _DEAD, 2.0)
    assert tracker.ejected(_LANE, 2.0)


def test_a_success_resets_the_streak() -> None:
    """An answered request between failures means the box is up."""
    tracker = LaneEjections(LaneEjectionPolicy())
    tracker.failed(_LANE, _DEAD, 0.0)
    tracker.failed(_LANE, _DEAD, 1.0)
    tracker.succeeded(_LANE)
    tracker.failed(_LANE, _DEAD, 2.0)
    assert not tracker.ejected(_LANE, 2.0)


def test_half_open_lets_one_probe_through_and_a_success_closes() -> None:
    """After the cooldown the first claim probes; others keep ordering it last until it lands."""
    tracker = LaneEjections(LaneEjectionPolicy())
    _ejected_after_three(tracker, _LANE, 0.0)
    assert not tracker.ejected(_LANE, 32.5)
    tracker.claimed(_LANE, 32.5)
    assert tracker.ejected(_LANE, 33.0)
    tracker.succeeded(_LANE)
    assert not tracker.ejected(_LANE, 33.0)
    # Closed for good: one new failure does not re-eject.
    tracker.failed(_LANE, _DEAD, 34.0)
    assert not tracker.ejected(_LANE, 34.0)


def test_a_failed_probe_re_ejects_for_another_cooldown() -> None:
    """One dead-lane failure on the half-open probe ejects again at once."""
    tracker = LaneEjections(LaneEjectionPolicy())
    _ejected_after_three(tracker, _LANE, 0.0)
    tracker.claimed(_LANE, 33.0)
    tracker.failed(_LANE, _DEAD, 33.5)
    assert tracker.ejected(_LANE, 34.0)
    assert tracker.ejected(_LANE, 63.4)
    assert not tracker.ejected(_LANE, 63.6)


def test_a_released_or_lost_probe_frees_the_next_probe() -> None:
    """A probe claim that never dispatched, or never reported back, does not pin the lane."""
    tracker = LaneEjections(LaneEjectionPolicy())
    _ejected_after_three(tracker, _LANE, 0.0)
    tracker.claimed(_LANE, 33.0)
    tracker.released(_LANE)
    assert not tracker.ejected(_LANE, 33.1)
    tracker.claimed(_LANE, 34.0)
    assert tracker.ejected(_LANE, 63.9)
    assert not tracker.ejected(_LANE, 64.0)


def test_zero_failures_disables_ejection() -> None:
    """The operator switch: ``failures=0`` never ejects."""
    tracker = LaneEjections(LaneEjectionPolicy(failures=0))
    for at in range(10):
        tracker.failed(_LANE, _DEAD, float(at))
    assert not tracker.ejected(_LANE, 10.0)
