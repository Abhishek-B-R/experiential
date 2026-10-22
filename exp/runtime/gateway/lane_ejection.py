"""Per-worker, per-lane ejection of a rung whose box answers nothing but dead-lane failures.

On 2026-10-21 a self-hosted rung's box refused every connection (or answered
an instant 502) for twenty minutes, and every request on the route dialed it
first: 7,000+ wasted dials. The revision-scoped health circuit
(``DeploymentHealthRegistry``) keys on the authorization's catalog digest, so
its memory restarts per revision view, and it counts timeouts and malformed
streams alike. This tracker keys on the PHYSICAL lane (deployment id and
connection digest, the ``rung_load_key``), so every request on the worker
learns from every other one, and it counts only the failures that say the box
itself is down: transport failures (refused, reset, unreachable) and provider
5xx (``provider_internal``: 500, 502, 503, 504). A client 4xx, a throttle, or
a timeout says nothing here (each has its own handling) and neither extends
nor resets a streak; a success, or an open stream, closes it.

``failures`` such failures inside ``window_seconds`` (no success between them)
eject the lane for ``cooldown_seconds``. An ejected lane is never removed:
the claim walk orders it LAST among a route's candidates, so a single-rung
route still dials it and fails fast, and a route whose other rungs are all
unclaimable still reaches it. After the cooldown the lane is half-open: the
first request that claims it is its probe (other requests keep ordering it
last), a success closes the ejection, and a counted failure re-ejects it for
another cooldown. A probe that never reports back expires after one cooldown
so the lane cannot stay ejected forever on a lost settlement.

State is in memory, per worker, and never published to the catalog. Never
account-wide or provider-wide: a dead lane ejects only itself.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from exp.runtime.gateway.contracts import GatewayFailureClass

LaneKey = tuple[str, str]
"""``(deployment_id, connection_sha256)``: one physical lane on this worker."""

DEAD_LANE_FAILURES = frozenset(
    {GatewayFailureClass.TRANSPORT, GatewayFailureClass.PROVIDER_INTERNAL}
)
"""The failure classes that count toward an ejection: the box itself is down."""

DEFAULT_EJECTION_FAILURES = 3
DEFAULT_EJECTION_WINDOW_SECONDS = 10.0
DEFAULT_EJECTION_COOLDOWN_SECONDS = 30.0

EJECTION_FAILURES_ENV = "EXP_GATEWAY_LANE_EJECTION_FAILURES"
EJECTION_WINDOW_ENV = "EXP_GATEWAY_LANE_EJECTION_WINDOW_SECONDS"
EJECTION_COOLDOWN_ENV = "EXP_GATEWAY_LANE_EJECTION_COOLDOWN_SECONDS"


@dataclass(frozen=True)
class LaneEjectionPolicy:
    """When a lane is ejected and for how long.

    Attributes:
        failures: Consecutive dead-lane failures that eject the lane; zero
            disables ejection entirely.
        window_seconds: The span those failures must fall inside.
        cooldown_seconds: How long an ejected lane is ordered last before
            one request may probe it.
    """

    failures: int = DEFAULT_EJECTION_FAILURES
    window_seconds: float = DEFAULT_EJECTION_WINDOW_SECONDS
    cooldown_seconds: float = DEFAULT_EJECTION_COOLDOWN_SECONDS

    def __post_init__(self) -> None:
        """Reject a policy that could never eject or never recover.

        Raises:
            ValueError: A negative failure count or a non-positive duration.
        """
        if self.failures < 0:
            raise ValueError("lane ejection failures must not be negative")
        if self.window_seconds <= 0 or self.cooldown_seconds <= 0:
            raise ValueError("lane ejection window and cooldown must be positive")

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> LaneEjectionPolicy:
        """Read the operator overrides, keeping the defaults for unset variables.

        Args:
            environ: The environment to read; ``None`` reads ``os.environ``.

        Returns:
            The policy with every set override applied.

        Raises:
            ValueError: An override does not parse or is out of range.
        """
        mapping = os.environ if environ is None else environ
        failures = mapping.get(EJECTION_FAILURES_ENV)
        window = mapping.get(EJECTION_WINDOW_ENV)
        cooldown = mapping.get(EJECTION_COOLDOWN_ENV)
        return cls(
            failures=DEFAULT_EJECTION_FAILURES if not failures else int(failures),
            window_seconds=DEFAULT_EJECTION_WINDOW_SECONDS if not window else float(window),
            cooldown_seconds=DEFAULT_EJECTION_COOLDOWN_SECONDS if not cooldown else float(cooldown),
        )


@dataclass
class _LaneState:
    """One lane's streak and ejection, mutated only under the owner's lock.

    Attributes:
        failures: Monotonic times of the current streak's counted failures.
        ejected_until: Monotonic instant the cooldown ends; 0 when not ejected.
        probe_started: Monotonic instant a half-open probe was claimed, or
            ``None`` when no probe is in flight.
    """

    failures: deque[float] = field(default_factory=deque)
    ejected_until: float = 0.0
    probe_started: float | None = None


class LaneEjections:
    """Track dead-lane streaks and ejections per physical lane.

    Not thread-safe on its own: ``DeploymentHealthRegistry`` calls every
    method under its lock.
    """

    def __init__(self, policy: LaneEjectionPolicy) -> None:
        """Bind the ejection policy.

        Args:
            policy: Thresholds and cooldown.
        """
        self._policy = policy
        self._lanes: dict[LaneKey, _LaneState] = {}

    def ejected(self, lane: LaneKey, now: float) -> bool:
        """Whether a claim walk should order the lane last right now.

        True inside the cooldown, and after it while another request's
        half-open probe is still in flight (and not yet expired).

        Args:
            lane: The physical lane.
            now: The current monotonic time.

        Returns:
            Whether the lane is ejected for this claim walk.
        """
        state = self._lanes.get(lane)
        if state is None or not state.ejected_until:
            return False
        if now < state.ejected_until:
            return True
        return (
            state.probe_started is not None
            and now - state.probe_started < self._policy.cooldown_seconds
        )

    def claimed(self, lane: LaneKey, now: float) -> None:
        """Note that a request claimed the lane, making it the probe when half-open.

        Args:
            lane: The physical lane.
            now: The current monotonic time.
        """
        state = self._lanes.get(lane)
        if state is None or not state.ejected_until or now < state.ejected_until:
            return
        if (
            state.probe_started is None
            or now - state.probe_started >= self._policy.cooldown_seconds
        ):
            state.probe_started = now

    def released(self, lane: LaneKey) -> None:
        """Forget an in-flight probe whose claim never reached dispatch.

        Args:
            lane: The physical lane.
        """
        state = self._lanes.get(lane)
        if state is not None:
            state.probe_started = None

    def succeeded(self, lane: LaneKey) -> None:
        """Close the lane's streak and ejection: the box answered.

        Args:
            lane: The physical lane.
        """
        self._lanes.pop(lane, None)

    def failed(self, lane: LaneKey, failure_class: GatewayFailureClass, now: float) -> None:
        """Apply one settled failure to the lane.

        A dead-lane failure extends the streak, ejects at the threshold, and
        re-ejects a half-open lane at once; any other class only frees the
        probe slot so the next request may probe.

        Args:
            lane: The physical lane.
            failure_class: The settled attempt's failure class.
            now: The current monotonic time.
        """
        if self._policy.failures == 0:
            return
        state = self._lanes.setdefault(lane, _LaneState())
        probing = bool(state.ejected_until) and now >= state.ejected_until
        state.probe_started = None
        if failure_class not in DEAD_LANE_FAILURES:
            return
        if probing:
            state.ejected_until = now + self._policy.cooldown_seconds
            return
        state.failures.append(now)
        while len(state.failures) > self._policy.failures:
            state.failures.popleft()
        if (
            len(state.failures) == self._policy.failures
            and now - state.failures[0] <= self._policy.window_seconds
        ):
            state.ejected_until = now + self._policy.cooldown_seconds
            state.failures.clear()
