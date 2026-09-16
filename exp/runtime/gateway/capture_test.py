"""Capture serialization and lifecycle behavior independent of hosted storage."""

from __future__ import annotations

import json
import logging

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.capture import (
    CapturePayload,
    PromptCaptureBuffer,
    PromptCapturePayload,
    ResponseCaptureHandoff,
    ResponseCapturePayload,
    ResponseCaptureRegistry,
    count_unstorable_text,
    enqueue_capture_response,
    sanitize_capture_message,
    serialize_capture_messages,
    serialize_capture_response,
    sse_capture_document,
)
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest


class _Writer:
    """Record the destination boundary without any host dependencies."""

    def __init__(self) -> None:
        """Start with an empty ordered destination."""
        self.payloads: list[CapturePayload] = []

    def enqueue(self, payload: CapturePayload) -> None:
        """Record one handoff in arrival order."""
        self.payloads.append(payload)


def _request(text: str = "hello") -> GatewayRequest:
    """Build the canonical request whose messages capture preserves."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=text),),
        stream=False,
    )


def _payload(request_id: str) -> PromptCapturePayload:
    """Return a small capture already attributed by the host."""
    return PromptCapturePayload(
        request_id=request_id,
        org_id="organization-a",
        prompt_sha256=None,
        messages_json='[{"role":"user","content":"hello"}]',
    )


def test_messages_preserve_canonical_wire_shape_without_changing_request() -> None:
    """The shared serializer emits exactly the existing stored message array."""
    request = _request()
    original = request.model_dump_json()
    assert serialize_capture_messages(request, request_id="r") == json.dumps(
        [message.model_dump(mode="json", exclude_none=True) for message in request.messages],
        separators=(",", ":"),
    )
    assert request.model_dump_json() == original


@pytest.mark.parametrize("text", ["a\x00b", "a\ud800b", "a\udfffb"])
def test_unstorable_strings_are_normalized_only_in_saved_copy(text: str) -> None:
    """Stored text is durable while the caller's request stays byte-identical."""
    request = _request(text)
    saved = serialize_capture_messages(request, request_id="r")
    assert saved is not None
    assert json.loads(saved)[0]["content"] == "a\ufffdb"
    assert request.messages[0].content == text


def test_literal_escapes_and_valid_surrogate_pairs_preserve_meaning() -> None:
    """Already-storable escape spelling is not treated as damaged text."""
    saved = serialize_capture_messages(_request("\\u0000 \ud83d\ude00"), request_id="r")
    assert saved is not None
    assert json.loads(saved)[0]["content"] == "\\u0000 \U0001f600"


def test_sanitizer_preserves_colliding_object_keys() -> None:
    """Normalizing a key cannot overwrite a caller's other tool arguments."""
    source: JsonObject = {"a\x00": "first", "a\ufffd": "second", "a\ufffd~2": "third"}
    saved, count = sanitize_capture_message(source)
    # One replaced code point and two collision-disambiguation steps.
    assert count == 3
    assert sorted(saved.values()) == ["first", "second", "third"]
    assert saved["a\ufffd"] == "second"
    assert source["a\x00"] == "first"


def test_unstorable_count_covers_keys_values_and_nested_tool_arguments() -> None:
    """All content branches share the same durable-text counting contract."""
    assert count_unstorable_text({"x\x00": ["\ud800", {"v": "\ud83d\ude00"}]}) == 2


def test_message_size_cap_drops_instead_of_truncating() -> None:
    """An incomplete prompt is never represented as a complete stored message array."""
    assert serialize_capture_messages(_request("x" * 1_100_000), request_id="r") is None


def test_buffer_is_pop_once_and_oldest_first_bounded() -> None:
    """Accepted requests can consume only their own most recent buffered capture."""
    buffer = PromptCaptureBuffer(capacity=2)
    for request_id in ("a", "b", "c"):
        buffer.remember(_payload(request_id))
    assert buffer.pop("a") is None
    assert buffer.pop("b") == _payload("b")
    assert buffer.pop("b") is None
    assert buffer.pop("c") == _payload("c")


def test_response_retains_every_field_and_uses_utf8_size() -> None:
    """Tool calls, logprobs, and Unicode survive without a projection whitelist."""
    document: JsonObject = {
        "kind": "json",
        "status": 200,
        "body": {"text": "\u4e2d" * 100_000, "logprobs": [0.1], "tool_calls": [{"id": "t"}]},
    }
    saved = serialize_capture_response(document, request_id="r")
    assert saved is not None
    assert json.loads(saved) == document
    assert serialize_capture_response({"text": "x" * 4_000_000}, request_id="r") is None


def test_sse_shape_retains_non_json_frames_and_explicit_partial_status() -> None:
    """Raw data frames stay ordered and disconnect metadata is not invented."""
    complete = sse_capture_document(
        [b'{"x":1}', b"other"], truncated=False, client_disconnected=False
    )
    assert complete == {
        "kind": "sse",
        "status": 200,
        "frames": [{"x": 1}, "other"],
        "truncated": False,
    }
    partial = sse_capture_document([], truncated=True, client_disconnected=True)
    assert partial["client_disconnected"] is True
    assert partial["truncated"] is True


def test_registry_expires_and_reports_unclaimed_entries(caplog: pytest.LogCaptureFixture) -> None:
    """Expired response permissions cannot authorize a later capture."""
    now = [0.0]
    registry = ResponseCaptureRegistry(capacity=2, ttl_seconds=1.0, clock=lambda: now[0])
    registry.record(request_id="r", org_id="o")
    now[0] = 2.0
    with caplog.at_level(logging.INFO, logger="exp.runtime.gateway.capture"):
        assert registry.pop("r") is None
    assert "expired unclaimed" in caplog.text


@pytest.mark.parametrize("settle_first", [False, True])
def test_disconnect_and_settlement_orders_preserve_identical_frames(settle_first: bool) -> None:
    """Either side of the relay/settle race hands off one identical partial response."""
    registry = ResponseCaptureRegistry()
    writer = _Writer()
    handoff = ResponseCaptureHandoff(registry, writer)
    if settle_first:
        registry.record(request_id="r", org_id="o")
    handoff.park("r", [b'{"delta":"hi"}'], truncated=False)
    if not settle_first:
        parked = registry.record(request_id="r", org_id="o")
        assert parked is not None
        enqueue_capture_response(
            writer,
            "r",
            "o",
            sse_capture_document(
                parked.frames, truncated=parked.truncated, client_disconnected=True
            ),
        )
    assert len(writer.payloads) == 1
    payload = writer.payloads[0]
    assert isinstance(payload, ResponseCapturePayload)
    assert payload.org_id == "o"
    assert json.loads(payload.response_json) == {
        "kind": "sse",
        "status": 200,
        "frames": [{"delta": "hi"}],
        "truncated": False,
        "client_disconnected": True,
    }
    assert handoff.claim("r") is None


def test_unregistered_response_is_never_authorized_and_forget_releases_claim() -> None:
    """The host alone authorizes capture; a replay or unknown request grants nothing."""
    registry = ResponseCaptureRegistry()
    writer = _Writer()
    handoff = ResponseCaptureHandoff(registry, writer)
    assert handoff.claim("unknown") is None
    registry.record(request_id="r", org_id="o")
    handoff.forget("r")
    assert handoff.claim("r") is None
    assert writer.payloads == []


def test_parked_frames_are_bounded_by_bytes_count_and_age() -> None:
    """Unsettled or capture-disabled streams cannot retain unbounded frame buffers."""
    now = [0.0]
    registry = ResponseCaptureRegistry(
        parked_capacity=1, parked_bytes_cap=1024, parked_ttl_seconds=1.0, clock=lambda: now[0]
    )
    registry.park("a", [b"a"], truncated=False)
    registry.park("b", [b"b"], truncated=False)
    assert registry.record(request_id="a", org_id="o") is None
    assert registry.record(request_id="b", org_id="o") is not None
    registry.park("large", [b"x" * 2048], truncated=False)
    assert registry.record(request_id="large", org_id="o") is None
    registry.park("expired", [b"x"], truncated=False)
    now[0] = 2.0
    assert registry.record(request_id="expired", org_id="o") is None


def test_large_stream_is_explicitly_truncated_before_enqueue() -> None:
    """The response destination receives a bounded prefix, not an oversized write."""
    writer = _Writer()
    document: JsonObject = {"kind": "sse", "frames": ["x" * 1_000_000] * 8, "truncated": False}
    assert enqueue_capture_response(writer, "r", "o", document)
    payload = writer.payloads[0]
    assert isinstance(payload, ResponseCapturePayload)
    saved = json.loads(payload.response_json)
    assert saved["truncated"] is True
    assert 0 < len(saved["frames"]) < 8


def test_large_nonstream_response_is_not_enqueued() -> None:
    """An oversized non-stream response is an explicit capture miss."""
    writer = _Writer()
    assert not enqueue_capture_response(writer, "r", "o", {"body": "x" * 4_000_000})
    assert writer.payloads == []
