"""Cache duration admission for the hosted gateway's write-price contract."""

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError


def cache_markers(request: GatewayRequest) -> tuple[JsonObject, ...]:
    """Return all explicit cache hints without serializing prompt content."""
    markers: list[JsonObject | None] = [request.provider_cache_control]
    markers.extend(tool.cache_control for tool in request.tools)
    for tool in request.provider_server_tools:
        marker = tool.get("cache_control")
        if isinstance(marker, dict):
            markers.append(marker)
    for message in request.messages:
        markers.append(message.cache_control)
        markers.extend(call.cache_control for call in message.tool_calls)
        for part in message.content_parts:
            if part.kind == "text" or part.kind == "image" or part.kind == "document":
                markers.append(part.cache_control)
        blocks = (*message.provider_text_blocks, *(message.provider_anthropic_blocks or ()))
        if message.provider_anthropic_block is not None:
            blocks = (*blocks, message.provider_anthropic_block)
        for block in blocks:
            marker = block.get("cache_control")
            if isinstance(marker, dict):
                markers.append(marker)
    return tuple(marker for marker in markers if marker is not None)


def require_priceable_cache_duration(profile: GatewayWireProfile, request: GatewayRequest) -> None:
    """Refuse hosted one-hour writes until settlement carries the TTL-specific price.

    The hosted ledger has a single five-minute write rate. A one-hour write
    costs twice ordinary input and cannot be charged accurately from that
    aggregate count. BYOK pays the upstream directly and preserves its TTL.
    """
    if profile.billing_customer_managed or not profile.preserves_cache_control:
        return
    if any(marker.get("ttl") == "1h" for marker in cache_markers(request)):
        raise ProviderParameterError(
            message=(
                "This hosted route supports 5-minute cache writes. "
                "Use ttl '5m' or a BYOK route for 1-hour caching."
            ),
            param="cache_control.ttl",
            code="invalid_parameter",
        )
