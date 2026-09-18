"""Cache breakpoints survive public decoding and supported provider adapters."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.openai_protocol.requests import decode_chat


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
def test_explicit_prefix_reaches_cache_capable_adapters(surface: str, wire: str) -> None:
    """The customer's two API formats retain the same exact prefix breakpoint."""
    marked: JsonObject = {
        "type": "text",
        "text": "a stable prefix",
        "cache_control": {"type": "ephemeral"},
    }
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "answer briefly"}],
    }
    if surface == "messages":
        body["system"] = [marked]
        request = decode_messages(body).request
    else:
        body["messages"] = [
            {"role": "system", "content": [marked]},
            {"role": "user", "content": "answer briefly"},
        ]
        request = decode_chat(body).request
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        model_id="claude-sonnet-4-6",
        url="https://example.invalid",
        forwards_cache_control=wire == "openrouter",
    )
    payload = dialect_stream_payload(profile, request)
    if wire == "bedrock_converse_stream":
        assert payload["system"] == [
            {"text": "a stable prefix"},
            {"cachePoint": {"type": "default"}},
        ]
    elif wire == "anthropic_messages":
        assert payload["system"] == [marked]
    else:
        messages = payload["messages"]
        assert isinstance(messages, list)
        first = messages[0]
        assert isinstance(first, dict)
        assert first["content"] == [marked]


def test_unknown_chat_adapter_does_not_receive_unsupported_markers() -> None:
    """A generic endpoint is not declared cache-capable merely for speaking Chat."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hello", "cache_control": {"type": "ephemeral"}}
            ],
        }
    ).request
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.invalid")
    payload = dialect_stream_payload(profile, request)
    assert not profile.preserves_cache_control
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_marker_keeps_exact_text_boundaries() -> None:
    """A breakpoint before a dynamic suffix stays at that prefix boundary."""
    blocks: list[JsonObject] = [
        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "dynamic suffix"},
    ]
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": blocks}]}
    ).request
    assert request.messages[0].content == "prefixdynamic suffix"
    assert request.messages[0].provider_text_blocks == tuple(blocks)


@pytest.mark.parametrize("wire", ["bedrock_converse_stream", "openrouter"])
def test_folded_instruction_keeps_prior_checkpoint(wire: str) -> None:
    """A dynamic system reminder must not erase or extend the user's cached prefix."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": "stable prefix",
                    "cache_control": {"type": "ephemeral"},
                },
                {"role": "system", "content": "dynamic reminder"},
            ],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        model_id="claude-sonnet-4-6",
        url="https://example.invalid",
        forwards_cache_control=wire == "openrouter",
        system_messages_leading_only=wire == "openrouter",
    )
    payload = dialect_stream_payload(profile, request)
    messages = payload["messages"]
    assert isinstance(messages, list)
    first = messages[0]
    assert isinstance(first, dict)
    if wire == "bedrock_converse_stream":
        assert first["content"] == [
            {"text": "stable prefix"},
            {"cachePoint": {"type": "default"}},
            {"text": "\n\n"},
            {"text": "dynamic reminder"},
        ]
    else:
        assert first["content"] == [
            {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "dynamic reminder"},
        ]


@pytest.mark.parametrize(
    "media",
    [
        {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ],
)
def test_unsupported_media_marker_drops_with_disclosure(media: JsonObject) -> None:
    """A cache hint cannot turn an accepted media request into a 400."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
                        media,
                    ],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ).request
    assert request.ignored_parameters == ("messages.0.cache_control->dropped(unsupported_media)",)
    assert len(request.messages[0].content_parts) == 2
    assert request.messages[0].provider_text_blocks == (
        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
    )
