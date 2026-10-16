"""Disclosed repairs for client fields that carry no meaning a provider can act on.

Some widely deployed OpenAI-surface clients attach fields the gateway cannot
forward but whose removal cannot change the answer: GitHub Copilot Chat tags
Chat messages with its own ``cot_id``, LiteLLM echoes assistant output items
with a lifecycle ``status`` or ``phase`` but without the item ``id`` those
markers belong to, and the OpenAI SDK sends Responses ``stream_options``. A
named 400 for any of them fails every request the client makes, so each is
removed here, before the compatibility manifest and the wire models read the
body, and every removal that changes what the caller asked for is named in
``ignored_parameters``. Any other unknown field keeps its named rejection.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.responses_probe import drop_gateway_web_search_items

CHAT_COT_ID_DROPPED = "messages.cot_id->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's client-side ``cot_id`` tag is removed."""

OUTPUT_STATUS_DROPPED = "input.status->dropped(no_item_id)"
"""Disclosure recorded when an id-less assistant output item loses its ``status``."""

OUTPUT_PHASE_DROPPED = "input.phase->dropped(no_item_id)"
"""Disclosure recorded when an id-less message item loses its ``phase``."""

STREAM_OPTIONS_NOT_STREAMING = "stream_options->ignored(stream_false)"
"""Disclosure recorded when Responses ``stream_options`` arrive on a non-streaming call."""

STREAM_OBFUSCATION_IGNORED = "stream_options.include_obfuscation->ignored(not_emitted)"
"""Disclosure recorded when stream obfuscation padding is requested but never emitted."""

_STREAM_OPTION_FIELDS = frozenset({"include_obfuscation"})


def drop_chat_message_cot_ids(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove the client-side ``cot_id`` tag from every Chat message.

    GitHub Copilot Chat stamps ``cot_id`` on messages it replays. No provider
    defines the field, so it is client bookkeeping with nothing to forward.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when no message carries the tag; otherwise a copy
        without it plus the one disclosure naming the removal.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not any(
        isinstance(message, dict) and "cot_id" in message for message in raw_messages
    ):
        return payload, ()
    messages: list[JsonValue] = [
        {key: value for key, value in message.items() if key != "cot_id"}
        if isinstance(message, dict)
        else message
        for message in raw_messages
    ]
    return {**payload, "messages": messages}, (CHAT_COT_ID_DROPPED,)


def drop_unanchored_output_markers(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove output lifecycle markers that arrive without their item id.

    ``status`` on an assistant message and ``phase`` on any message describe a
    provider-issued output item, which the gateway can replay only by its
    ``id``. Without the id the item is an ordinary input message whose text is
    unchanged, so the markers are removed and the message decodes as one.

    Args:
        payload: Parsed Responses body.

    Returns:
        The original payload when nothing needs repair; otherwise a copy with
        the markers removed plus one disclosure per removed marker kind.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    dropped_status = False
    dropped_phase = False
    items: list[JsonValue] = []
    for item in raw_input:
        if (
            not isinstance(item, dict)
            or item.get("type", "message") != "message"
            or item.get("id") is not None
        ):
            items.append(item)
            continue
        strip_status = item.get("role") == "assistant" and item.get("status") is not None
        strip_phase = item.get("phase") is not None
        if not strip_status and not strip_phase:
            items.append(item)
            continue
        dropped_status = dropped_status or strip_status
        dropped_phase = dropped_phase or strip_phase
        removed = {"status"} if strip_status else set()
        if strip_phase:
            removed.add("phase")
        items.append({key: value for key, value in item.items() if key not in removed})
    if not dropped_status and not dropped_phase:
        return payload, ()
    disclosures = (
        *((OUTPUT_STATUS_DROPPED,) if dropped_status else ()),
        *((OUTPUT_PHASE_DROPPED,) if dropped_phase else ()),
    )
    return {**payload, "input": items}, disclosures


def accept_responses_stream_options(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Consume the Responses ``stream_options`` object the OpenAI API defines.

    Its only field, ``include_obfuscation``, asks the server to pad stream
    deltas with random characters. The gateway never emits that padding, so
    ``false`` is honored exactly and ``true`` is disclosed as ignored. The
    object on a non-streaming call has nothing to configure and is disclosed
    as ignored. Unknown keys inside the object keep a named rejection.

    Args:
        payload: Parsed Responses body.

    Returns:
        The payload without ``stream_options`` and any disclosures.

    Raises:
        OpenAIProtocolError: The object is malformed or names an unknown key.
    """
    if "stream_options" not in payload:
        return payload, ()
    options = payload["stream_options"]
    remaining = {key: value for key, value in payload.items() if key != "stream_options"}
    if options is None:
        return remaining, ()
    if not isinstance(options, dict):
        raise invalid_field("stream_options", "'stream_options' must be an object.")
    unknown = sorted(set(options) - _STREAM_OPTION_FIELDS)
    if unknown:
        raise invalid_field(
            f"stream_options.{unknown[0]}",
            f"Unknown parameter: 'stream_options.{unknown[0]}'. "
            "Responses stream_options accepts only include_obfuscation.",
        )
    obfuscation = options.get("include_obfuscation")
    if obfuscation is not None and not isinstance(obfuscation, bool):
        raise invalid_field(
            "stream_options.include_obfuscation",
            "'stream_options.include_obfuscation' must be a boolean.",
        )
    if payload.get("stream") is not True:
        return remaining, (STREAM_OPTIONS_NOT_STREAMING,)
    if obfuscation is True:
        return remaining, (STREAM_OBFUSCATION_IGNORED,)
    return remaining, ()


def repair_chat(
    payload: JsonObject, disclosures: tuple[str, ...]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every Chat repair, appending its disclosures to ``disclosures``."""
    payload, cot = drop_chat_message_cot_ids(payload)
    return payload, (*disclosures, *cot)


def repair_responses(
    payload: JsonObject, disclosures: tuple[str, ...]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every Responses repair, appending its disclosures to ``disclosures``.

    The gateway's own rendered web search items are removed first
    (``responses_probe.drop_gateway_web_search_items``).

    Raises:
        OpenAIProtocolError: ``stream_options`` is malformed.
    """
    payload, markers = drop_unanchored_output_markers(drop_gateway_web_search_items(payload))
    payload, stream = accept_responses_stream_options(payload)
    return payload, (*disclosures, *markers, *stream)
