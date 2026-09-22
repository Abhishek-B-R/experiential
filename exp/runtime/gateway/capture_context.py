"""Bounded, credential-free effective request context for trace consumers."""

from __future__ import annotations

import json
import re

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.core.durable_json import normalize_durable_object
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.replay_identity import provider_replay_authority

_UNSTORABLE_ESCAPE = re.compile(r"\\u(?:0000|d[89a-f][0-9a-f]{2})")


def capture_request_context(
    request: GatewayRequest, *, maximum_bytes: int = 1_048_576
) -> JsonObject | None:
    """Snapshot post-guardrail, expanded context without changing the served request.

    Provider-significant carriers excluded from normal model serialization are
    retained separately. No transport headers, resolved credentials, or provider
    connection configuration enters this document. A prompt can itself contain
    sensitive text; this function does not promise content redaction.
    """
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    document: JsonObject = {
        "schema_version": 1,
        "request": request.model_dump(mode="json", exclude_none=True, exclude={"idempotency_key"}),
        "provider_context": _captured_provider_context(request),
    }
    encoded = json.dumps(document, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > maximum_bytes:
        return None
    if not _UNSTORABLE_ESCAPE.search(encoded):
        return document
    cleaned, replacements = normalize_durable_object(document)
    if not replacements:
        return document
    # JSONB cannot represent NUL or lone surrogates. Keep a queryable
    # projection and the exact escaped JSON, rather than destroy evidence.
    cleaned["source_json"] = encoded
    return (
        cleaned
        if len(json.dumps(cleaned, ensure_ascii=True, separators=(",", ":"))) <= maximum_bytes
        else None
    )


def _captured_provider_context(request: GatewayRequest) -> JsonObject | None:
    """Add caller-visible evidence to the stored copy, never to replay authority."""
    provider = provider_replay_authority(request)
    captured: dict[int, list[JsonValue]] = {
        index: [block.model_dump(mode="json") for block in message.capture_only_reasoning]
        for index, message in enumerate(request.messages)
        if message.capture_only_reasoning
    }
    if not captured:
        return provider
    if provider is None:
        provider = {"provider_replay": []}
    replay = provider["provider_replay"]
    assert isinstance(replay, list)
    for entry in replay:
        assert isinstance(entry, dict)
        index = entry["message_index"]
        assert isinstance(index, int)
        visible = captured.pop(index, [])
        if not visible:
            continue
        blocks = entry.setdefault("provider_reasoning", [])
        assert isinstance(blocks, list)
        blocks.extend(visible)
    replay.extend(
        {"message_index": index, "provider_reasoning": blocks} for index, blocks in captured.items()
    )
    return provider


def restore_capture_context(context: JsonObject) -> JsonObject:
    """Recover exact captured strings from the explicitly lossless JSON sidecar."""
    source = context.get("source_json")
    if source is None:
        return context
    if not isinstance(source, str):
        raise ValueError("capture source_json must be text")
    document = json.loads(source)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid lossless capture context")
    return document
