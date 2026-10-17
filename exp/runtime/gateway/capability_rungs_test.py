"""Capability-first rung selection and sampling clamps through route admission."""

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.capability_rungs import narrow_to_capable_rungs
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_admission import admitted_route_requests
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _Accounting:
    """Count coercion recordings without a live ledger."""

    recovery_host = None

    def __init__(self) -> None:
        """Start at zero recorded coercions."""
        self.recorded = 0

    def record_admission_coercions(self, count: int) -> None:
        """Accumulate one admission's disclosure count."""
        self.recorded += count


def _deployment(deployment_id: str, *, structured: bool) -> ExactModelDeployment:
    """Build one rung that does or does not declare structured output."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(
            maximum_output_tokens=8_192, supports_structured_output=structured
        ),
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_structured_text=True,
                supports_stop_sequences=True,
            )
        ),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...],
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
) -> GatewayRoute:
    """Build one route over ``deployments`` in the given order."""
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=surface,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _qwen_route(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
) -> tuple[GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...]]:
    """Mirror qwen3.8-27b: a lead without structured output, a schema rung without temperature."""
    client = cast(NativeWireClient, object())
    route = _route(
        (_deployment("novita", structured=False), _deployment("cerebras", structured=True)),
        surface,
    )
    wires = (
        (GatewayWireProfile(dialect="openai_compatible", url="https://n.test"), client),
        (
            GatewayWireProfile(
                dialect="openai_compatible", url="https://c.test", supports_temperature=False
            ),
            client,
        ),
    )
    return route, wires


def _structured_chat(temperature: float | None = None) -> GatewayRequest:
    """Decode one Chat request with a strict json_schema response format."""
    payload: JsonObject = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "answer"}],
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": _SCHEMA},
        },
    }
    if temperature is not None:
        payload["temperature"] = temperature
    return decode_chat(payload).request


def _admit(
    route: GatewayRoute,
    wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...],
    request: GatewayRequest,
) -> tuple[GatewayRoute, GatewayRequest, GatewayRequest, int]:
    """Admit one request and return the narrowed route, both requests, and the coercions."""
    accounting = _Accounting()
    narrowed, _wires, public, provider, _placement = admitted_route_requests(
        route,
        wires,
        request,
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    return narrowed, public, provider, accounting.recorded


def test_structured_output_routes_past_a_lead_that_cannot_carry_it() -> None:
    """The lead kept the caller's temperature exactly, so preference ranking chose it
    and the capability check then refused a schema the second rung serves (7,210
    qwen3.8-27b refusals in three days, 2026-10-17). The schema rung now serves,
    dropping the temperature it does not take, and both changes are disclosed."""
    route, wires = _qwen_route()
    narrowed, public, provider, recorded = _admit(route, wires, _structured_chat(temperature=0.2))
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("cerebras",)
    assert provider.structured_text is not None
    assert provider.temperature is None
    assert "response_format->routed(structured_output_rung)" in public.ignored_parameters
    assert "temperature->dropped(unsupported_by_provider)" in public.ignored_parameters
    assert recorded == 1


def test_responses_structured_output_names_text_format() -> None:
    """The same reorder on the Responses surface names the caller's ``text.format``."""
    route, wires = _qwen_route(GatewayApiSurface.RESPONSES)
    request = decode_responses(
        {
            "model": "public-model",
            "input": "answer",
            "stream": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": _SCHEMA,
                }
            },
        }
    ).request
    narrowed, public, _provider, _recorded = _admit(route, wires, request)
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("cerebras",)
    assert "text.format->routed(structured_output_rung)" in public.ignored_parameters


def test_a_capable_lead_keeps_the_route_and_discloses_nothing() -> None:
    """Narrowing drops the incapable fallback; the lead is unchanged, so no reorder is told."""
    deployments = (_deployment("lead", structured=True), _deployment("plain", structured=False))
    wire = (
        GatewayWireProfile(dialect="openai_compatible", url="https://l.test"),
        cast(NativeWireClient, object()),
    )
    narrowed, _wires, disclosures = narrow_to_capable_rungs(
        _route(deployments), (wire, wire), _structured_chat()
    )
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("lead",)
    assert disclosures == ()


def test_no_rung_with_structured_output_still_fails_closed() -> None:
    """The route stays whole and the per-rung check refuses on the field, as before."""
    client = cast(NativeWireClient, object())
    route = _route((_deployment("a", structured=False), _deployment("b", structured=False)))
    wire = GatewayWireProfile(dialect="openai_compatible", url="https://a.test")
    with pytest.raises(ProviderCapabilityError):
        _admit(route, ((wire, client), (wire, client)), _structured_chat())


def test_a_live_reasoning_pin_never_moves() -> None:
    """A continuation pinned to its issuing rung keeps it first; a stale pin does not."""
    route, wires = _qwen_route()
    pinned = route.model_copy(update={"reasoning_pinned_deployment_id": "novita"})
    kept, _wires, disclosures = narrow_to_capable_rungs(pinned, wires, _structured_chat())
    assert kept is pinned
    assert disclosures == ()
    stale = route.model_copy(update={"reasoning_pinned_deployment_id": "issuer"})
    moved, _wires, disclosures = narrow_to_capable_rungs(stale, wires, _structured_chat())
    assert tuple(item.deployment_id for item in moved.deployments) == ("cerebras",)
    assert disclosures == ("response_format->routed(structured_output_rung)",)


def test_a_locked_temperature_route_clamps_instead_of_refusing() -> None:
    """claude-haiku-5.5 accepts only temperature 1.0; a caller's 0.1 now serves at 1.0
    with ``temperature->1.0(route_minimum)`` instead of a 400 (41 refusals a day)."""
    client = cast(NativeWireClient, object())
    deployment = _deployment("claude", structured=True)
    route = _route((deployment,), GatewayApiSurface.MESSAGES)
    wire = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-haiku-5-5",
        minimum_temperature=1.0,
        maximum_temperature=1.0,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        maximum_output_tokens=1_024,
        temperature=0.1,
    )
    _narrowed, public, provider, recorded = _admit(route, ((wire, client),), request)
    assert provider.temperature == 1.0
    assert public.ignored_parameters == ("temperature->1.0(route_minimum)",)
    assert recorded == 1
