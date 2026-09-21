"""Effective context capture preserves tools and never mutates serving input."""

import json

from exp.runtime.gateway.capture_context import capture_request_context, restore_capture_context
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.openai_protocol.requests import decode_chat


def test_capture_context_preserves_tools_and_generation_settings() -> None:
    """The saved context includes definitions, not only observed tool calls."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "lookup"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a record.",
                        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                    },
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 0.2,
            "max_tokens": 128,
        }
    ).request
    before = request.model_dump_json()
    context = capture_request_context(request)
    assert context is not None
    assert context["request"] == request.model_dump(mode="json", exclude_none=True)
    assert request.tools[0].parameters["type"] == "object"
    assert request.model_dump_json() == before
    assert capture_request_context(request, maximum_bytes=1) is None


def test_excluded_provider_carriers_are_retained_separately() -> None:
    """A provider's native tool declaration survives the capture-only projection."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).request.model_copy(
        update={"provider_thinking_config": {"type": "enabled", "budget_tokens": 32}}
    )
    context = capture_request_context(request)
    assert context is not None
    provider = context["provider_context"]
    assert isinstance(provider, dict)
    assert provider["provider_thinking_config"] == {"type": "enabled", "budget_tokens": 32}


def test_capture_context_is_storable_and_omits_transport_replay_key() -> None:
    """Normalization touches the stored copy, not the served prompt or opaque key."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "a\x00b\ud800"}]}
    ).request.model_copy(update={"idempotency_key": "private-header"})
    before = request.model_dump()
    context = capture_request_context(request)
    assert context is not None
    serialized = json.dumps(context)
    assert "\x00" not in serialized
    restored = restore_capture_context(context)
    assert GatewayRequest.model_validate(restored["request"]).messages[0].content == "a\x00b\ud800"
    # Escapes inside source_json are ordinary text after JSONB decodes the envelope.
    source = context["source_json"]
    assert isinstance(source, str) and "\x00" not in source
    assert "private-header" not in serialized
    assert request.model_dump() == before


def test_lossless_context_retains_colliding_keys_and_enforces_total_budget() -> None:
    """A lossless sidecar never bypasses the admission memory ceiling."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "a\0b"}]}
    ).request
    context = capture_request_context(request)
    assert context is not None
    size = len(json.dumps(context, ensure_ascii=True, separators=(",", ":")))
    assert capture_request_context(request, maximum_bytes=size - 1) is None
    restored = GatewayRequest.model_validate(restore_capture_context(context)["request"])
    assert restored.messages[0].content == "a\0b"
