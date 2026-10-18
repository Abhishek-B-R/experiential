"""Disclosed repairs for client fields that carry no meaning a provider can act on.

Some widely deployed OpenAI-surface clients attach fields the gateway cannot
forward but whose removal cannot change the answer: GitHub Copilot Chat tags
Chat messages with its own ``cot_id`` and ``cot_summary``, other agents stamp
``timestamp`` or ``startsNewUserRequest`` on them, LiteLLM echoes assistant
output items with a lifecycle ``status`` or ``phase`` but without the item
``id`` those markers belong to, Codex Desktop tags Responses input items with
``internal_chat_message_metadata_passthrough``, and the OpenAI SDK sends
Responses ``stream_options``. A
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

CHAT_COT_SUMMARY_DROPPED = "messages.cot_summary->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's client-side ``cot_summary`` is removed."""

CHAT_TIMESTAMP_DROPPED = "messages.timestamp->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's client-side ``timestamp`` is removed."""

CHAT_STARTS_NEW_USER_REQUEST_DROPPED = "messages.startsNewUserRequest->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's ``startsNewUserRequest`` flag is removed."""

CHAT_PARTICIPANT_NAME_DROPPED = "messages.name->dropped(participant_name)"
"""Disclosure recorded when a non-tool Chat message's participant ``name`` is removed."""

_CHAT_CLIENT_ANNOTATIONS: tuple[tuple[str, str], ...] = (
    ("cot_id", CHAT_COT_ID_DROPPED),
    ("cot_summary", CHAT_COT_SUMMARY_DROPPED),
    ("timestamp", CHAT_TIMESTAMP_DROPPED),
    ("startsNewUserRequest", CHAT_STARTS_NEW_USER_REQUEST_DROPPED),
)
"""Client bookkeeping keys no provider defines on a Chat message, in disclosure order."""

_PARTICIPANT_NAME_ROLES = frozenset({"system", "developer", "user", "assistant"})
"""Roles whose ``name`` is OpenAI's optional participant label.

On ``tool`` and ``function`` messages ``name`` attributes the result to its
function and is carried by the wire model, so it is never removed there."""

INPUT_METADATA_PASSTHROUGH_DROPPED = (
    "input.internal_chat_message_metadata_passthrough->dropped(client_annotation)"
)
"""Disclosure recorded when Codex Desktop's per-item metadata passthrough is removed."""

REASONING_WITHOUT_IDENTITY_DROPPED = "input.reasoning->dropped(no_id_or_encrypted_content)"
"""Disclosure recorded when a replayed reasoning item carries only display summary.

Without an ``id`` or ``encrypted_content`` no provider can resolve the item,
and its summary and content parts are display-only (they never reach provider
history even on an identified item), so dropping it changes nothing sent."""

STREAM_REASONING_SUMMARY_DELIVERY_IGNORED = (
    "stream_options.reasoning_summary_delivery->ignored(not_supported)"
)
"""Disclosure recorded when Codex's reasoning-summary delivery option is ignored."""

OUTPUT_STATUS_DROPPED = "input.status->dropped(no_item_id)"
"""Disclosure recorded when an id-less assistant output item loses its ``status``."""

OUTPUT_PHASE_DROPPED = "input.phase->dropped(no_item_id)"
"""Disclosure recorded when an id-less message item loses its ``phase``."""

STREAM_OPTIONS_NOT_STREAMING = "stream_options->ignored(stream_false)"
"""Disclosure recorded when Responses ``stream_options`` arrive on a non-streaming call."""

STREAM_OBFUSCATION_IGNORED = "stream_options.include_obfuscation->ignored(not_emitted)"
"""Disclosure recorded when stream obfuscation padding is requested but never emitted."""

_STREAM_OPTION_FIELDS = frozenset({"include_obfuscation", "reasoning_summary_delivery"})


def drop_chat_message_client_annotations(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove client bookkeeping keys and participant names from Chat messages.

    GitHub Copilot Chat stamps ``cot_id`` and ``cot_summary`` on messages it
    replays, and other agents stamp ``timestamp`` or ``startsNewUserRequest``.
    No provider defines them, so they are client bookkeeping with nothing to
    forward. ``name`` on a system, developer, user, or assistant message is
    OpenAI's optional participant label: no other provider wire carries it,
    so it is removed with its own disclosure rather than refused.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when no message carries any of them; otherwise a
        copy without them plus one disclosure per removed key kind.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload, ()
    removed: set[str] = set()
    messages: list[JsonValue] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        keys = {key for key, _ in _CHAT_CLIENT_ANNOTATIONS if key in message}
        if message.get("role") in _PARTICIPANT_NAME_ROLES and "name" in message:
            keys.add("name")
        if not keys:
            messages.append(message)
            continue
        removed |= keys
        messages.append({key: value for key, value in message.items() if key not in keys})
    if not removed:
        return payload, ()
    disclosures = tuple(
        disclosure for key, disclosure in _CHAT_CLIENT_ANNOTATIONS if key in removed
    ) + ((CHAT_PARTICIPANT_NAME_DROPPED,) if "name" in removed else ())
    return {**payload, "messages": messages}, disclosures


def drop_responses_client_items(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove Responses input the gateway can neither forward nor resolve.

    Codex Desktop tags input items with
    ``internal_chat_message_metadata_passthrough``, client bookkeeping no
    provider defines. ai-sdk, Bun, codex-tui and LiteLLM replay reasoning items
    that carry neither ``id`` nor ``encrypted_content``: nothing can resolve
    such an item, and its summary and content are display-only parts that never
    reach provider history, so the whole item goes.

    Args:
        payload: Parsed Responses body.

    Returns:
        The original payload when nothing needs repair; otherwise a copy plus
        one disclosure per repair kind.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    dropped_metadata = False
    dropped_reasoning = False
    items: list[JsonValue] = []
    for item in raw_input:
        if not isinstance(item, dict):
            items.append(item)
            continue
        if (
            item.get("type") == "reasoning"
            and item.get("id") is None
            and item.get("encrypted_content") is None
        ):
            dropped_reasoning = True
            continue
        if "internal_chat_message_metadata_passthrough" in item:
            dropped_metadata = True
            item = {
                key: value
                for key, value in item.items()
                if key != "internal_chat_message_metadata_passthrough"
            }
        items.append(item)
    if not dropped_metadata and not dropped_reasoning:
        return payload, ()
    disclosures = (
        *((INPUT_METADATA_PASSTHROUGH_DROPPED,) if dropped_metadata else ()),
        *((REASONING_WITHOUT_IDENTITY_DROPPED,) if dropped_reasoning else ()),
    )
    return {**payload, "input": items}, disclosures


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

    Its documented field, ``include_obfuscation``, asks the server to pad
    stream deltas with random characters. The gateway never emits that
    padding, so ``false`` is honored exactly and ``true`` is disclosed as
    ignored. Codex Desktop also sends ``reasoning_summary_delivery``, a
    delivery preference for summary events the gateway streams the same way
    regardless, so any value is disclosed as ignored. The object on a
    non-streaming call has nothing to configure and is disclosed as ignored.
    Unknown keys inside the object keep a named rejection.

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
            "Responses stream_options accepts only include_obfuscation and "
            "reasoning_summary_delivery.",
        )
    obfuscation = options.get("include_obfuscation")
    if obfuscation is not None and not isinstance(obfuscation, bool):
        raise invalid_field(
            "stream_options.include_obfuscation",
            "'stream_options.include_obfuscation' must be a boolean.",
        )
    if payload.get("stream") is not True:
        return remaining, (STREAM_OPTIONS_NOT_STREAMING,)
    disclosures = (
        *((STREAM_OBFUSCATION_IGNORED,) if obfuscation is True else ()),
        *(
            (STREAM_REASONING_SUMMARY_DELIVERY_IGNORED,)
            if options.get("reasoning_summary_delivery") is not None
            else ()
        ),
    )
    return remaining, disclosures


def repair_chat(
    payload: JsonObject, disclosures: tuple[str, ...]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every Chat repair, appending its disclosures to ``disclosures``."""
    payload, annotations = drop_chat_message_client_annotations(payload)
    return payload, (*disclosures, *annotations)


def repair_responses(
    payload: JsonObject, disclosures: tuple[str, ...]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every Responses repair, appending its disclosures to ``disclosures``.

    The gateway's own rendered web search items are removed first
    (``responses_probe.drop_gateway_web_search_items``).

    Raises:
        OpenAIProtocolError: ``stream_options`` is malformed.
    """
    payload, items = drop_responses_client_items(drop_gateway_web_search_items(payload))
    payload, markers = drop_unanchored_output_markers(payload)
    payload, stream = accept_responses_stream_options(payload)
    return payload, (*disclosures, *items, *markers, *stream)
