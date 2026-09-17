"""Caller cache TTL evidence used only to reserve a sufficient write budget."""

from __future__ import annotations

from collections.abc import Iterator

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import DocumentContentPart, ImageContentPart, TextContentPart
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.decisions_contracts import DecisionRequest


def requests_hour_cache(request: GatewayRequest | DecisionRequest) -> bool:
    """Whether any carried prompt-cache marker requests the one-hour write rate.

    Args:
        request: Canonical request whose excluded cache carriers can reach the provider.

    Returns:
        True when at least one marker requests one hour. This is reservation evidence,
        never a substitute for provider-reported TTL counts at settlement.
    """
    return isinstance(request, GatewayRequest) and any(
        marker is not None and marker.get("ttl") == "1h" for marker in _markers(request)
    )


def _markers(request: GatewayRequest) -> Iterator[JsonObject | None]:
    """Yield cache controls from every supported Anthropic request carrier.

    Args:
        request: Request containing typed tools, messages, and opaque provider blocks.

    Yields:
        A carried cache marker or None when a carrier has no marker.
    """
    yield request.provider_cache_control
    for tool in request.tools:
        yield tool.cache_control
    for block in request.provider_server_tools:
        marker = block.get("cache_control")
        if isinstance(marker, dict):
            yield marker
    for message in request.messages:
        yield message.cache_control
        blocks = (*message.provider_text_blocks, *(message.provider_anthropic_blocks or ()))
        if message.provider_anthropic_block is not None:
            blocks = (*blocks, message.provider_anthropic_block)
        for block in blocks:
            marker = block.get("cache_control")
            if isinstance(marker, dict):
                yield marker
        for call in message.tool_calls:
            yield call.cache_control
        for part in message.content_parts:
            if isinstance(part, (TextContentPart, ImageContentPart, DocumentContentPart)):
                yield part.cache_control
