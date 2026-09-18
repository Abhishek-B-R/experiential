"""OpenCode-style ``cache_control`` normalization for the Chat surface.

The @ai-sdk stack attaches Anthropic-style ephemeral cache hints to recent
messages for Claude-family model ids. Placements are classified in
``CHAT_CACHE_CONTROL_PLACEMENTS``: message-level and text-part hints are
validated and temporarily removed for official SDK validation, then retained
on canonical message carriers. Tool-call hints ride canonical tool calls.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, field_validator

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.openai_protocol.errors import invalid_field

_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


class _HintWireModel(BaseModel):
    """Strict private wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class EphemeralCacheControl(_HintWireModel):
    """Validated OpenCode/Anthropic cache breakpoint.

    The object form is ``{"type": "ephemeral"}`` with an optional ``ttl`` of
    ``5m`` or ``1h``. An explicit ``ttl: null`` is not in that allowlist.
    """

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None

    @field_validator("ttl", mode="before")
    @classmethod
    def _reject_null_ttl(cls, value: object) -> object:
        """Reject an explicit null TTL while still allowing the key to be omitted."""
        if value is None:
            raise ValueError("ttl must be 5m or 1h when present")
        return value


def drop_opencode_cache_control(payload: JsonObject) -> JsonObject:
    """Remove supported OpenCode ``cache_control`` annotations from Chat messages.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload, or a shallow copy whose messages no longer carry
        a supported ``cache_control`` annotation.

    Raises:
        OpenAIProtocolError: A ``cache_control`` value is malformed or unsupported.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload
    cleaned_messages: list[JsonValue] = []
    changed = False
    for index, raw_message in enumerate(raw_messages):
        message, message_changed = _without_message_hint(raw_message, index)
        cleaned_messages.append(message)
        changed = changed or message_changed
    if not changed:
        return payload
    cleaned_payload = dict(payload)
    cleaned_payload["messages"] = cleaned_messages
    return cleaned_payload


def restore_chat_cache_control(
    messages: tuple[GatewayMessage, ...], payload: JsonObject
) -> tuple[GatewayMessage, ...]:
    """Carry validated Chat hints past official SDK validation onto wire metadata.

    The decoder validates the cleaned body first. Its one-to-one message mapping
    lets the original markers ride the same carriers used by Messages requests.
    """
    raw_messages = cast(list[JsonObject], payload["messages"])
    restored: list[GatewayMessage] = []
    for message, raw in zip(messages, raw_messages, strict=True):
        marker = raw.get("cache_control")
        content = raw.get("content")
        blocks: list[JsonObject] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
                    block: JsonObject = {"type": "text", "text": part["text"]}
                    if isinstance(part.get("cache_control"), dict):
                        block["cache_control"] = part["cache_control"]
                    blocks.append(block)
        elif isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        if message.role == "tool":
            # A tool result is one upstream block; its last text breakpoint
            # therefore belongs to the whole result.
            markers = [b["cache_control"] for b in blocks if "cache_control" in b]
            tool_marker = marker if isinstance(marker, dict) else markers[-1] if markers else None
            restored.append(message.model_copy(update={"cache_control": tool_marker}))
            continue
        if isinstance(marker, dict):
            if message.tool_calls:
                calls = (
                    *message.tool_calls[:-1],
                    message.tool_calls[-1].model_copy(update={"cache_control": marker}),
                )
                message = message.model_copy(update={"tool_calls": calls})
            elif message.content_parts and message.content_parts[-1].kind != "text":
                last = message.content_parts[-1]
                if last.kind != "image" and last.kind != "document":
                    raise invalid_field("messages.cache_control")
                message = message.model_copy(
                    update={
                        "content_parts": (
                            *message.content_parts[:-1],
                            last.model_copy(update={"cache_control": marker}),
                        )
                    }
                )
            elif blocks:
                blocks[-1]["cache_control"] = marker
        if any("cache_control" in block for block in blocks):
            message = message.model_copy(update={"provider_text_blocks": tuple(blocks)})
        restored.append(message)
    return tuple(restored)


def _without_message_hint(raw_message: JsonValue, index: int) -> tuple[JsonValue, bool]:
    """Drop a supported ``cache_control`` annotation from one Chat message.

    Args:
        raw_message: One ``messages`` entry.
        index: Zero-based message index used in public error paths.

    Returns:
        The message (copied when an annotation is removed) and whether it changed.

    Raises:
        OpenAIProtocolError: The annotation is present but not a supported form.
    """
    if not isinstance(raw_message, dict):
        return raw_message, False
    message = cast(JsonObject, raw_message)
    changed = False
    if "cache_control" in message:
        require_supported_cache_control(message["cache_control"], f"messages.{index}.cache_control")
        message = {key: value for key, value in message.items() if key != "cache_control"}
        changed = True
    content = message.get("content")
    if isinstance(content, list):
        cleaned_content, content_changed = _without_text_part_hint(
            cast(list[JsonValue], content), index
        )
        if content_changed:
            if not changed:
                message = dict(message)
            message["content"] = cleaned_content
            changed = True
    return message, changed


def _without_text_part_hint(
    parts: list[JsonValue], message_index: int
) -> tuple[list[JsonValue], bool]:
    """Drop supported ``cache_control`` from OpenCode text content parts.

    Args:
        parts: Message ``content`` array.
        message_index: Zero-based parent message index used in public error paths.

    Returns:
        The content array (copied when an annotation is removed) and whether it changed.

    Raises:
        OpenAIProtocolError: A text-part annotation is present but not a supported form.
    """
    cleaned: list[JsonValue] = []
    changed = False
    for part_index, raw_part in enumerate(parts):
        if not isinstance(raw_part, dict) or "cache_control" not in raw_part:
            cleaned.append(raw_part)
            continue
        part = cast(JsonObject, raw_part)
        if part.get("type") not in _TEXT_PART_TYPES:
            cleaned.append(raw_part)
            continue
        require_supported_cache_control(
            part["cache_control"],
            f"messages.{message_index}.content.{part_index}.cache_control",
        )
        cleaned.append({key: value for key, value in part.items() if key != "cache_control"})
        changed = True
    return (cleaned, True) if changed else (parts, False)


def require_supported_cache_control(value: JsonValue, param: str) -> None:
    """Accept null or a supported ephemeral ``cache_control`` object.

    Args:
        value: Raw ``cache_control`` annotation.
        param: Public dotted field path used in the error.

    Raises:
        OpenAIProtocolError: The annotation is malformed or unsupported.
    """
    if value is None:
        return
    try:
        EphemeralCacheControl.model_validate(value)
    except ValidationError as exc:
        raise invalid_field(param) from exc
