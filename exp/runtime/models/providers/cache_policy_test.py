"""Hosted write pricing must match the requested cache duration."""

import pytest

from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import require_priceable_cache_duration
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.errors import ProviderParameterError


@pytest.mark.parametrize("customer_managed", [True, False])
def test_one_hour_cache_requires_direct_provider_billing(customer_managed: bool) -> None:
    """One-hour writes cannot silently use a hosted five-minute rate."""
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 32,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://example.invalid",
        billing_customer_managed=customer_managed,
    )
    if customer_managed:
        require_priceable_cache_duration(profile, request)
    else:
        with pytest.raises(ProviderParameterError, match="5-minute"):
            require_priceable_cache_duration(profile, request)


@pytest.mark.parametrize("customer_managed", [True, False])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_server_tool_cache_duration_is_checked_before_dispatch(
    customer_managed: bool, ttl: str
) -> None:
    """Verbatim server-tool declarations cannot bypass hosted write pricing."""
    marker = {"type": "ephemeral", "ttl": ttl}
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 32,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "cache_control": marker,
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://example.invalid",
        billing_customer_managed=customer_managed,
    )
    if ttl == "1h" and not customer_managed:
        with pytest.raises(ProviderParameterError, match="5-minute"):
            dialect_stream_payload(profile, request)
    else:
        payload = dialect_stream_payload(profile, request)
        assert payload["tools"] == [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "cache_control": marker,
            }
        ]
