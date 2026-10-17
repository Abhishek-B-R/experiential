"""Capability-first rung selection for hard request features some rungs lack.

Admission ranks a route's rungs by how exactly they keep the caller's soft
preferences (sampling, thinking) before it checks the hard capabilities each
rung's catalog declares. When the rung that keeps the preferences exactly is
the one that cannot carry a hard feature, that ranking narrows the route to it
and the capability check then refuses a request another rung could serve
(qwen3.8-27b: a structured-output request with a ``temperature`` its
schema-enforcing rung does not take, 2026-10-17). A hard feature therefore
narrows the route FIRST, to the rungs that declare it, in route order; the
soft preferences are judged among those rungs only.

Only structured output (``response_format`` / ``text.format`` json_schema) is
handled here: its requirement is decided by the decoded request alone, so no
later shaping or coercion can remove it. When no rung declares it the route is
left whole and the existing per-rung check refuses the request by field. A
reasoning-pinned continuation whose issuing rung is still on the route, and a
caller-pinned route, never move.
"""

from __future__ import annotations

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient

STRUCTURED_OUTPUT_RUNG_SUFFIX = "->routed(structured_output_rung)"
"""Disclosure suffix recorded when structured output moves the request off the lead rung."""

_ResolvedWires = tuple[tuple[GatewayWireProfile, NativeWireClient], ...]


def serves_structured_output(deployment: ExactModelDeployment) -> bool:
    """Whether one rung's catalog declares structured output, as preflight judges it.

    Mirrors ``preflight_gateway_request``: the deployment must declare
    structured text, and its model capabilities (when the catalog carries
    them) must declare structured output.
    """
    model = deployment.capabilities
    return deployment.gateway.capabilities.supports_structured_text and (
        model is None or model.supports_structured_output
    )


def narrow_to_capable_rungs(
    route: GatewayRoute, resolved_wires: _ResolvedWires, request: GatewayRequest
) -> tuple[GatewayRoute, _ResolvedWires, tuple[str, ...]]:
    """Keep only the rungs that can carry the request's structured output.

    Args:
        route: Frozen ordered route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        request: Decoded public request.

    Returns:
        The narrowed route and wires plus the reorder disclosure, which is
        recorded only when the lead rung was passed over. The inputs come back
        unchanged when the request asks for no structured output, every rung
        or no rung carries it, or the route is pinned to a live rung.
    """
    pinned = route.reasoning_pinned_deployment_id
    if (
        request.structured_text is None
        or route.resolved_route_id is not None
        or any(deployment.deployment_id == pinned for deployment in route.deployments)
    ):
        return route, resolved_wires, ()
    indexes = tuple(
        index
        for index, deployment in enumerate(route.deployments)
        if serves_structured_output(deployment)
    )
    if not indexes or len(indexes) == len(route.deployments):
        return route, resolved_wires, ()
    field = "text.format" if request.surface == GatewayApiSurface.RESPONSES else "response_format"
    disclosures = () if indexes[0] == 0 else (f"{field}{STRUCTURED_OUTPUT_RUNG_SUFFIX}",)
    return (
        select_route_deployments(route, indexes),
        tuple(resolved_wires[index] for index in indexes),
        disclosures,
    )
