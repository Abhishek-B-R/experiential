"""Native admission consumes session evidence while respecting stage and host gates."""

import pytest

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest, GatewayUsage
from exp.runtime.gateway.model_plan import model_execution_snapshot
from exp.runtime.gateway.model_plan_test import catalog
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_test import _RecordingLedger
from exp.runtime.gateway.native_admission import (
    _affinity_ordered_rungs,
    _prefer_cache_capable_rungs,
)
from exp.runtime.gateway.native_admission_test import _affinity_fixture, _marked_request
from exp.runtime.gateway.native_execution import InflightRequest, select_route_deployments
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_recovery import record_session_outcome, session_cache_key
from exp.runtime.gateway.native_stage_admission import stage_affinity_ordered_rungs
from exp.runtime.gateway.recovery import RecoveryScope, RecoverySnapshot, SessionRecoveryRegistry
from exp.runtime.gateway.recovery_test import Clock
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile


class Host:
    """In-memory immutable observation host with explicit credential identity."""

    def scope_for(self, deployment: ExactModelDeployment, organization_id: str) -> RecoveryScope:
        """Freeze a known credential scope per destination model."""
        return RecoveryScope(
            provider=deployment.provider,
            exact_model_id=deployment.exact_model_id,
            endpoint_scope=deployment.connection_sha256,
            region_scope="region",
            credential_scope="credential",
            organization_id=organization_id,
        )

    def snapshot(self) -> RecoverySnapshot:
        """Return no fleet recovery evidence, so a healthy warm fallback stays retained."""
        return RecoverySnapshot(loaded_at=1000)


def test_settlement_records_successful_session_cache_once_and_never_dispatch_only() -> None:
    """The native settlement hook owns evidence, independent of aggregate fairness samples."""
    normalized = catalog()
    auth = _route().snapshot.authorization
    snapshot = model_execution_snapshot(normalized, auth, normalized.pools[0])
    by_id = {d.deployment_id: d for d in normalized.deployments}
    deployments = tuple(by_id[d] for d in snapshot.deployment_ids)
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    request = GatewayRequest(
        surface=auth.surface,
        messages=(
            GatewayMessage(role="system", content="prefix"),
            GatewayMessage(role="user", content="turn"),
        ),
        provider_prompt_cache_key="xpl-test",
    )
    entry = InflightRequest(authorization=auth, route=route, request=request, deadline_monotonic=10)
    key = session_cache_key(entry)
    assert key is not None
    registry = SessionRecoveryRegistry(clock=Clock())
    host = Host()
    scope = host.scope_for(route.deployment, auth.organization_id)
    assert (
        registry.choose(
            key, ((route.deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    entry.attempt_depths["attempt"] = 0
    usage = GatewayUsage(input_tokens=100, output_tokens=1, cached_input_tokens=80)
    record_session_outcome(registry, host, entry, "attempt", usage, None)
    # Unknown declared retention remains no evidence even with a cache read.
    assert (
        registry.choose(
            key, ((route.deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    assert entry.recovery_recorded_attempts == {"attempt"}


@pytest.mark.parametrize("has_stages", [False, True])
@pytest.mark.parametrize("with_host", [False, True])
def test_live_reasoning_pin_precedes_stage_cache_and_recovery_ordering(
    has_stages: bool,
    with_host: bool,
) -> None:
    """A live child issuer is preserved, while a removed issuer is never resurrected."""
    route, wires = _affinity_fixture()
    snapshot = route.snapshot
    if has_stages:
        stage = snapshot.stage_for_depth(0).model_copy(
            update={"stage_index": 1, "ancestry": ("root", "child")}
        )
        snapshot = snapshot.model_copy(update={"exact_model_id": "root", "model_stages": (stage,)})
    route = route.model_copy(
        update={
            "snapshot": snapshot,
            "reasoning_pinned_deployment_id": route.deployment.deployment_id,
        }
    )
    wires = (
        wires[0],
        (GatewayWireProfile(dialect="anthropic_messages", url="https://cache.test"), wires[1][1]),
        wires[2],
    )
    accounting = NativeAttemptAccounting(
        _RecordingLedger(), recovery_host=Host() if with_host else None
    )
    request = _marked_request()
    marker_route, marker_wires = _prefer_cache_capable_rungs(route, wires, request)
    assert marker_route is route and marker_wires is wires
    ordered, ordered_wires, placement = _affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=snapshot.authorization,
        continuation=None,
    )
    assert ordered is route and ordered_wires is wires
    assert placement.fingerprint is not None
    accounting.sticky.bind(
        placement.fingerprint, route.deployments[-1].deployment_id, ttl_seconds=60
    )
    repeated, _, _ = stage_affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=snapshot.authorization,
        continuation=None,
    )
    assert repeated is route
    surviving = select_route_deployments(route, (1, 2))
    admitted, _, _ = stage_affinity_ordered_rungs(
        surviving,
        wires[1:],
        request,
        accounting=accounting,
        authorization=snapshot.authorization,
        continuation=None,
    )
    assert route.deployment.deployment_id not in admitted.snapshot.deployment_ids
    if has_stages:
        assert admitted.snapshot.exact_model_id == "root"
        assert all(s.ancestry == ("root", "child") for s in admitted.snapshot.model_stages)
