"""Immutable static-wire credentials survive rotation without exporting account identity."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast
from uuid import uuid4

import pytest

from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import normalize_gateway_catalog
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayUsage
from exp.runtime.gateway.execution_resolution import _resolved_wire_profile
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_accounting_test import _registry, _start
from exp.runtime.gateway.native_execution import InflightRequest, deployment_wire_entry
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_recovery import record_session_outcome, session_cache_key
from exp.runtime.gateway.native_recovery_test import request
from exp.runtime.gateway.recovery import (
    FrozenRecoveryBinding,
    OperationalScope,
    RecoveryScope,
    RecoverySnapshot,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.recovery_binding import (
    bind_recovery_profiles,
    frozen_scope,
    validated_recovery_binding,
)
from exp.runtime.gateway.recovery_test import Clock
from exp.runtime.models.credentials import CredentialResolution, DispatchCredentialReceipt
from exp.runtime.models.credentials_test import AtomicEnvironment
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.messages_payloads import anthropic_messages_stream_payload
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.registry_test import _catalog


class Host:
    """Record only detached operational evidence."""

    def __init__(self) -> None:
        """Start with no topology observations."""
        self.observed: list[OperationalScope] = []

    def observe_scope(self, scope: OperationalScope) -> None:
        """Capture the exact detached scope."""
        self.observed.append(scope)

    def attempt_started(self, attempt_id: str, scope: OperationalScope) -> None:
        """Accept an already-bound attempt without any credential read."""
        self.observed.append(scope)

    def snapshot(self) -> RecoverySnapshot:
        """Provide an empty snapshot without authorizing recovery."""
        return RecoverySnapshot(loaded_at=0)


@pytest.mark.parametrize("swept", [False, True])
def test_k1_scope_survives_rotation_through_actual_accounting_settlement(swept: bool) -> None:
    """Both settlement paths retain K1 and cannot create K2 cache evidence."""
    accounting, ledger, entry = _registry()
    host = Host()
    accounting.recovery_host = host
    accounting.recovery = SessionRecoveryRegistry(clock=Clock())
    deployment = entry.route.deployment.model_copy(
        update={
            "gateway": entry.route.deployment.gateway.model_copy(
                update={
                    "cache_retention_seconds": 100,
                    "dispatch": GatewayRungDispatchPolicy(sticky_spill_seconds=60),
                }
            )
        }
    )
    entry.route = entry.route.model_copy(
        update={
            "deployment": deployment,
            "snapshot": entry.route.snapshot.model_copy(
                update={"failover_mode": "maximize_cache_affinity"}
            ),
        }
    )
    entry.request = request()
    first, second = DispatchCredentialReceipt(uuid4()), DispatchCredentialReceipt(uuid4())
    scope = RecoveryScope(
        provider=deployment.provider,
        exact_model_id=deployment.exact_model_id,
        endpoint_scope="endpoint",
        region_scope="region",
        organization_id=entry.authorization.organization_id,
        credential_scope=str(first.binding_id),
    )
    binding = FrozenRecoveryBinding(
        deployment.deployment_id,
        deployment.connection_sha256,
        "https://test.invalid",
        deployment.provider_model,
        scope,
    )
    entry.recovery_bindings = {deployment.deployment_id: binding}
    started = _start(accounting, ordinal=0)
    rotated = scope.model_copy(update={"credential_scope": str(second.binding_id)})
    settlement = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_id": started["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 1, "cached_input_tokens": 80},
            "finalize": True,
        }
    )
    if swept:
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            accounting.settle(settlement)
        accounting.sweep_expired()
    else:
        accounting.settle(settlement)
    key = session_cache_key(entry)
    assert key is not None
    assert (
        accounting.recovery.choose(
            key, ((deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        == deployment.deployment_id
    )
    assert (
        accounting.recovery.choose(
            key, ((deployment.deployment_id, rotated),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    assert accounting.entry(entry.authorization.request_id) is None
    assert len(ledger.finished) == 1 and accounting.accounting_healthy
    assert host.observed == [scope.operational()]


def test_actual_runtime_client_auth_and_receipt_rotate_together() -> None:
    """One atomic lookup produces each client's actual static authorization header."""
    first, second = DispatchCredentialReceipt(uuid4()), DispatchCredentialReceipt(uuid4())
    environment = AtomicEnvironment(CredentialResolution("K1", "environment", receipt=first))
    catalog = _catalog(api_key_env="OPENAI_API_KEY")
    runtime = RuntimeModelCatalog(catalog, environment=environment)
    deployment = normalize_gateway_catalog(catalog).deployments[0]
    k1 = _resolved_wire_profile(deployment, runtime.resolve("fixture-model"))
    assert environment.calls == 1 and k1.credential_receipt is first
    environment.resolved = CredentialResolution("K2", "environment", receipt=second)
    k2 = _resolved_wire_profile(deployment, runtime.resolve("fixture-model"))
    assert environment.calls == 2 and k2.credential_receipt is second
    assert {name.lower(): value for name, value in k1.headers.items()}[
        "authorization"
    ] == "Bearer K1"
    assert {name.lower(): value for name, value in k2.headers.items()}[
        "authorization"
    ] == "Bearer K2"
    assert k1.credential_receipt is first


def test_receipt_is_frozen_before_choice_and_never_serialized() -> None:
    """K1 settles under K1 after current auth rotates to K2; shared topology is identical."""
    route = _route()
    deployment = route.deployment.model_copy(
        update={
            "gateway": route.deployment.gateway.model_copy(update={"cache_retention_seconds": 100})
        }
    )
    route = route.model_copy(update={"deployment": deployment})
    host = Host()
    first, second = DispatchCredentialReceipt(uuid4()), DispatchCredentialReceipt(uuid4())
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.openai.com/v1/chat/completions",
        model_id=deployment.provider_model,
        headers={"authorization": "Bearer test-K1"},
        credential_receipt=first,
    )
    client = cast(NativeWireClient, object())
    bound = bind_recovery_profiles(
        (deployment,), ((profile, client),), route.snapshot.authorization.organization_id, host
    )[0][0]
    binding = validated_recovery_binding(
        deployment, bound, route.snapshot.authorization.organization_id
    )
    assert binding is not None
    rotated = bind_recovery_profiles(
        (deployment,),
        (
            (
                replace(
                    profile, headers={"authorization": "Bearer test-K2"}, credential_receipt=second
                ),
                client,
            ),
        ),
        route.snapshot.authorization.organization_id,
        host,
    )[0][0]
    assert rotated.recovery_binding is not None
    assert binding.scope != rotated.recovery_binding.scope
    assert binding.scope.operational() == rotated.recovery_binding.scope.operational()
    entry = InflightRequest(
        route.snapshot.authorization,
        route,
        request(),
        10,
        recovery_bindings={deployment.deployment_id: binding},
        attempt_depths={"attempt": 0},
    )
    registry = SessionRecoveryRegistry()
    record_session_outcome(
        registry,
        host,
        entry,
        "attempt",
        GatewayUsage(input_tokens=100, cached_input_tokens=80, output_tokens=1),
        None,
    )
    retained = frozen_scope(
        entry.recovery_bindings, deployment, entry.authorization.organization_id
    )
    assert retained is not None and retained.credential_scope == str(first.binding_id)
    for encoded in (
        repr(bound),
        repr(binding),
        binding.scope.model_dump_json(),
        json.dumps(binding.scope.model_dump()),
        json.dumps(deployment_wire_entry(route, deployment, bound, {})),
        route.snapshot.model_dump_json(),
    ):
        assert str(first.binding_id) not in encoded
    assert "test-K1" not in repr(bound)
    assert all(type(scope) is OperationalScope for scope in host.observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "https://other.test/v1"),
        ("dialect", "anthropic_messages"),
        ("signs_request_body", True),
        ("model_id", "another-model"),
        ("credential_receipt", None),
        ("operational_region", "another-region"),
    ],
)
def test_reassigned_wire_binding_is_rejected(field: str, value: object) -> None:
    """A copied receipt cannot be attached to a different actual wire."""
    route = _route()
    host = Host()
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.openai.com/v1/chat/completions",
        credential_receipt=DispatchCredentialReceipt(uuid4()),
    )
    bound = bind_recovery_profiles(
        (route.deployment,),
        ((profile, cast(NativeWireClient, object())),),
        route.snapshot.authorization.organization_id,
        host,
    )[0][0]
    changed = replace(bound, **{field: value})
    with pytest.raises(ValueError, match="differs"):
        validated_recovery_binding(
            route.deployment, changed, route.snapshot.authorization.organization_id
        )


def test_unknown_receipt_or_region_does_not_invent_shared_recovery() -> None:
    """Ordinary serving survives missing binding; custom unknown regions publish no facts."""
    route = _route()
    host = Host()
    client = cast(NativeWireClient, object())
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://private.test/v1")
    assert (
        bind_recovery_profiles((route.deployment,), ((profile, client),), "org", host)[0][
            0
        ].recovery_binding
        is None
    )
    bound = bind_recovery_profiles(
        (route.deployment,),
        ((replace(profile, credential_receipt=DispatchCredentialReceipt(uuid4())), client),),
        "org",
        host,
    )[0][0]
    assert bound.recovery_binding is not None and bound.recovery_binding.scope.region_scope is None
    assert not host.observed


@pytest.mark.parametrize("region", ["us", "eu", "future-region"])
def test_request_geography_never_becomes_global_recovery(region: str) -> None:
    """A payload region not proven by the wire cannot inherit global transport recovery."""
    route = _route()
    host = Host()
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id=route.deployment.provider_model,
        credential_receipt=DispatchCredentialReceipt(uuid4()),
    )
    bound = bind_recovery_profiles(
        (route.deployment,),
        ((profile, cast(NativeWireClient, object())),),
        route.snapshot.authorization.organization_id,
        host,
        request_region=region,
    )[0][0]
    assert bound.recovery_binding is None
    assert not host.observed
    scoped_request = request().model_copy(
        update={"surface": GatewayApiSurface.MESSAGES, "inference_geo": region}
    )
    assert (
        anthropic_messages_stream_payload(profile.model_id, scoped_request)["inference_geo"]
        == region
    )
    global_profile = bind_recovery_profiles(
        (route.deployment,),
        ((profile, cast(NativeWireClient, object())),),
        route.snapshot.authorization.organization_id,
        host,
    )[0][0]
    assert global_profile.recovery_binding is not None
    reused = bind_recovery_profiles(
        (route.deployment,),
        ((global_profile, cast(NativeWireClient, object())),),
        route.snapshot.authorization.organization_id,
        host,
        request_region=region,
    )[0][0]
    assert reused.recovery_binding is None
