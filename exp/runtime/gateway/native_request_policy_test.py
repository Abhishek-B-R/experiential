"""Request-selected routes remain aligned with credentials and typed authority."""

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import GatewayDeploymentCapabilities
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_admission import resolve_admission_route
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway, _pool_control_plane
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import DispatchableRoute, dispatchable_route_profiles
from exp.runtime.gateway.native_request_policy import require_route_authority
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.web_search.backend import WebSearchBackend
from exp.runtime.gateway.web_search.plan import WebSearchPlan, plan_web_search
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


def _body(policy: JsonObject) -> str:
    """Encode one policy-bearing chat request for the bridge."""
    return json.dumps(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}], "gateway": policy}
    )


def test_no_fallback_selects_unrestricted_wire_and_credentials(tmp_path: Path) -> None:
    """A conditional lead cannot leave its wire attached to the ordinary successor."""
    _manager, key = _configured_pool_gateway(
        tmp_path,
        api_key_envs=("ALPHA_KEY", "BETA_KEY"),
        gateway_capabilities=(
            GatewayDeploymentCapabilities(
                supports_streaming=True, failover_only_on=("provider_internal",)
            ),
            GatewayDeploymentCapabilities(supports_streaming=True),
        ),
    )
    plane = NativeControlPlane(
        load_gateway_components(
            tmp_path, environment={"ALPHA_KEY": "alpha-secret", "BETA_KEY": "beta-secret"}
        )
    )
    admitted = json.loads(
        plane.admit(
            json.dumps({"raw_key": key, "body": _body({"routing": {"allow_fallbacks": False}})})
        )
    )
    assert len(admitted["route"]) == 1
    wire = admitted["route"][0]
    assert wire["deployment_id"] == "beta"
    assert "127.0.0.1:10" in wire["url"]
    assert wire["headers"]["Authorization"] == "Bearer beta-secret"
    assert wire["upstream_payload"]["model"] == "beta-model-exact"


def test_selected_only_route_plans_search_without_discarded_fallback(tmp_path: Path) -> None:
    """No-fallback planning retains provider-native search on the chosen wire."""
    plane, key = _pool_control_plane(tmp_path)
    components = plane._components  # noqa: SLF001
    selector = "route_" + "b" * 64
    body: JsonObject = {
        "model": "coding",
        "input": "search",
        "tools": [{"type": "web_search"}],
        "gateway": {"routing": {"route_id": selector, "allow_fallbacks": False}},
    }
    request = decode_responses(body).request
    authorization = components.store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=999999999
    )
    route = components.routes.resolve_direct(authorization)
    profiles = dispatchable_route_profiles(components.runtime_catalogs, route)
    first_profile, first_client = profiles.resolved_wires[0]
    profiles = DispatchableRoute(
        profiles.indexes,
        (
            (dataclasses.replace(first_profile, dialect="openai_responses"), first_client),
            *profiles.resolved_wires[1:],
        ),
        profiles.dead,
    )
    seen: list[str] = []

    def resolve(
        bound: NativeGatewayComponents,
        authority: AuthorizationSnapshot,
        incoming: GatewayRequest,
        *,
        continuation: ContinuationContext | None = None,
    ) -> GatewayRoute:
        """Stand in for a host resolving its public selector through the normal route seam."""
        return resolve_admission_route(
            bound, authority, incoming, continuation=continuation
        ).model_copy(update={"resolved_route_id": selector})

    def search(
        incoming: GatewayRequest,
        dialects: Sequence[str],
        backend: WebSearchBackend | None,
        *,
        deadline_monotonic: float,
    ) -> WebSearchPlan:
        """Observe the planner's actual inputs while running its normal implementation."""
        seen.extend(dialects)
        return plan_web_search(incoming, dialects, backend, deadline_monotonic=deadline_monotonic)

    with (
        patch("exp.runtime.gateway.native_bridge.resolve_admission_route", resolve),
        patch(
            "exp.runtime.gateway.native_bridge.dispatchable_route_profiles", return_value=profiles
        ),
        patch("exp.runtime.gateway.native_bridge.plan_web_search", search),
    ):
        admitted = json.loads(
            plane.admit(
                json.dumps({"raw_key": key, "surface": "responses", "body": json.dumps(body)})
            )
        )
    assert seen == ["openai_responses"]
    assert "web_search" not in admitted
    assert admitted["route"][0]["upstream_payload"]["tools"] == [{"type": "web_search"}]


def test_unhandled_standalone_route_is_a_field_error(tmp_path: Path) -> None:
    """A valid opaque selector is never ignored by the standalone resolver."""
    plane, key = _pool_control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as caught:
        plane.admit(
            json.dumps(
                {"raw_key": key, "body": _body({"routing": {"route_id": "route_" + "a" * 64}})}
            )
        )
    error = json.loads(caught.value.public_error_json)
    assert error["status_code"] == 400
    assert error["param"] == "gateway.routing.route_id"


def test_selected_conditional_route_fails_even_with_an_ordinary_fallback(tmp_path: Path) -> None:
    """An explicit route preference never widens conditional first-dial authority."""
    plane, key = _pool_control_plane(tmp_path)
    request: GatewayRequest = decode_chat(
        json.loads(_body({"routing": {"route_id": "route_" + "a" * 64}}))
    ).request
    components = plane._components  # noqa: SLF001
    authorization = components.store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=999999999
    )
    route: GatewayRoute = components.routes.resolve_direct(authorization)
    conditional = route.deployment.model_copy(
        update={
            "gateway": route.deployment.gateway.model_copy(
                update={
                    "capabilities": route.deployment.gateway.capabilities.model_copy(
                        update={"failover_only_on": ("provider_internal",)}
                    )
                }
            )
        }
    )
    selected = route.model_copy(
        update={"deployment": conditional, "resolved_route_id": authorization.requested_route_id}
    )
    with pytest.raises(ProviderParameterError, match="requested route"):
        require_route_authority(authorization, request, selected)
