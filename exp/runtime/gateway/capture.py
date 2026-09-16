"""Shared content-capture serialization and request/response lifecycle handoff.

The serving engine exposes canonical messages and relayed responses through typed
records. A host supplies capture consent and a CaptureWriter destination. This
module owns content shape, durable-text normalization, bounded in-flight buffers,
and the race between response delivery and settlement. It performs no database or
provider I/O and does not change the request or response sent over the gateway.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.core.text import normalize_durable_text
from exp.runtime.gateway.contracts import GatewayRequest

logger = logging.getLogger(__name__)

# Buffered payloads awaiting accept; small because only capture-on orgs fill
# it and accept follows authorize within the same request task.
_BUFFER_CAPACITY = 5_000

# A cheap "certainly over" pre-filter on the compact JSON text: jsonb's binary
# form is never smaller than the text for these shapes, so a prompt over 1 MiB
# of text cannot fit the column. The authority is the SQL function
#: it measures pg_column_size, the
# column's own yardstick, and refuses quietly -- jsonb runs 1.3-2.1x the text
# for object-heavy arrays, which no fixed text headroom can bound. Either way
# the response half still lands on its own row.
_MAX_MESSAGES_BYTES = 1_048_576
# A response is the whole thing the caller got (a logprobs stream is large).
# The column check is 4 MiB of jsonb TEXT, which renders ~8-10% larger than the
# compact UTF-8 dump measured here (a space after every ',' and ':'), so the
# worker-side cap keeps that headroom: a document that passes here always fits
# the column, and truncation (not a refused write) is the outcome at the edge.
_MAX_RESPONSE_BYTES = 3_670_016
# Settle -> relay handoff entries awaiting the front; the engine awaits the
# settle before it assembles terminal bytes, so a claimed entry lives seconds.
# An entry NOBODY claims (a Responses-over-WebSocket request: same api_surface,
# but its frames never cross the HTTP tee) expires by AGE, so sustained
# WebSocket traffic cannot crowd live HTTP entries out; the count cap is only a
# far safety net against a pathological rate.
_RESPONSE_REGISTRY_TTL_SECONDS = 300.0
_RESPONSE_REGISTRY_CAPACITY = 200_000
# Frames a caller received before it disconnected, parked until the settle
# names the request (ResponseCaptureRegistry.park). The engine keeps serving
# an abandoned stream and settles it when the provider finishes, so the settle
# can arrive up to the engine's request bound later; the age bound is that
# bound (the worker passes it), and total bytes are capped so a burst of
# abandoned streams on capture-off orgs (whose settle never claims) holds a
# bounded slice of memory.
_PARKED_TTL_SECONDS = 1560.0
_PARKED_BYTES_CAP = 64 * 1024 * 1024
_PARKED_CAPACITY = 2_000
# Accounted per parked entry and per frame beside the payload bytes: the id,
# the dataclass, the tuple, each bytes object's header and the OrderedDict
# node, so many tiny frames cannot sit under the byte cap unweighed.
_PARKED_ENTRY_OVERHEAD_BYTES = 512
_PARKED_FRAME_OVERHEAD_BYTES = 64


class PromptCapturePayload(BaseModel):
    """One request's captured content, ready for the background writer."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    org_id: str
    prompt_sha256: str | None
    # Canonical messages serialized to a JSON string (bounded); the writer
    # passes it to Postgres as jsonb.
    messages_json: str


class ResponseCapturePayload(BaseModel):
    """One request's entire response, serialized for the background writer.

    ``response_json`` is a JSON object: ``{"kind": "json", "status": 200,
    "body": <the finished body>}`` or ``{"kind": "sse", "status": 200,
    "frames": [<each data payload in order>], "truncated": bool}``.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    org_id: str
    response_json: str


class CaptureDiscard(BaseModel):
    """Forget one request's captured prompt: the settle proved it a BYOK lane."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    org_id: str


CapturePayload = PromptCapturePayload | ResponseCapturePayload | CaptureDiscard


class CaptureWriter(Protocol):
    """Destination receiving capture records in request lifecycle order.

    Implementations own durable writes and must enqueue without provider or database
    work on the serving caller. Organization consent belongs to the host, not this
    transport-independent content contract.
    """

    def enqueue(self, payload: CapturePayload) -> None:
        """Hand one bounded request, response, or discard to the destination."""
        ...


# ``json.dumps`` (ASCII-escaped) spells every code point jsonb refuses as a
# ``\uXXXX`` escape: NUL as ``\u0000``, a UTF-16 surrogate as ``\ud800``..
# ``\udfff``. Scanning the dumped payload for those spellings is the no-hit
# fast path: a prompt carrying neither (the overwhelming case) is stored as
# dumped, with no per-message walk or copy on the bridge callback thread. A
# hit is only a CANDIDATE , caller text spelling a backslash-u escape
# literally, or a valid surrogate pair (which jsonb decodes to one scalar),
# matches too; the walk then replaces nothing and the dumped document stands.
# NUL and every UTF-16 surrogate (high d800-dbff AND low dc00-dfff): jsonb
# refuses NUL, and a lone surrogate cannot be UTF-8 encoded at all.
_UNSTORABLE_ESCAPE = re.compile(r"\\u(?:0000|d[89a-f][0-9a-f]{2})")
_REPLACEMENT_CHAR = "\ufffd"


def _replace_in_text(text: str) -> tuple[str, int]:
    """Return ``text`` made durable by the engine's normalizer, plus the count.

    ``normalize_durable_text`` replaces NUL and each LONE surrogate with U+FFFD
    and folds a valid surrogate pair into its scalar (the value jsonb decodes
    the pair to anyway), so the replacements are exactly the U+FFFD it added.
    """
    normalized = normalize_durable_text(text)
    return normalized, normalized.count(_REPLACEMENT_CHAR) - text.count(_REPLACEMENT_CHAR)


def count_unstorable_text(value: JsonValue) -> int:
    """Count the code points inside ``value`` that no Postgres text column can hold.

    NUL and each LONE UTF-16 surrogate, in strings AND object keys, anywhere in
    the JSON tree; a valid surrogate pair is one storable scalar and counts as
    nothing. The batch lane's submit boundary refuses a JSONL line or filename
    on a non-zero count (``the host's batch admission``)
    instead of letting the job document's ``jsonb`` insert fail as a 500.
    """
    match value:
        case str():
            return _replace_in_text(value)[1]
        case dict():
            return sum(
                _replace_in_text(key)[1] + count_unstorable_text(item)
                for key, item in value.items()
            )
        case list():
            return sum(count_unstorable_text(item) for item in value)
        case _:
            return 0


def _replace_in_object(value: JsonObject) -> tuple[JsonObject, int]:
    r"""Sanitize one JSON object's keys and values without merging entries.

    Keys that needed no replacement are placed first, verbatim. A sanitized
    key that then collides with an existing key (``"a\x00"`` beside a literal
    ``"a\ufffd"``) is disambiguated with a ``~<n>`` suffix instead of
    overwriting the other entry; the disambiguation counts as a replacement so
    the stamp reflects it.
    """
    entries: JsonObject = {}
    total = 0
    renamed: list[tuple[str, JsonValue]] = []
    for key, item in value.items():
        cleaned_item, item_count = _replace_unstorable(item)
        total += item_count
        cleaned_key, key_count = _replace_in_text(key)
        if key_count == 0:
            entries[key] = cleaned_item
        else:
            total += key_count
            renamed.append((cleaned_key, cleaned_item))
    for cleaned_key, cleaned_item in renamed:
        unique_key = cleaned_key
        suffix = 1
        while unique_key in entries:
            suffix += 1
            unique_key = f"{cleaned_key}~{suffix}"
            total += 1
        entries[unique_key] = cleaned_item
    return entries, total


def _replace_unstorable(value: JsonValue) -> tuple[JsonValue, int]:
    """Return ``value`` with every unstorable code point replaced, plus the count.

    Walks the JSON value ``model_dump(mode="json")`` produces: strings (values
    and object keys) are rewritten, arrays and objects recurse, scalars pass
    through untouched.
    """
    match value:
        case dict():
            return _replace_in_object(value)
        case list():
            items: list[JsonValue] = []
            list_total = 0
            for item in value:
                cleaned_item, count = _replace_unstorable(item)
                items.append(cleaned_item)
                list_total += count
            return items, list_total
        case str():
            return _replace_in_text(value)
        case _:
            return value, 0


def sanitize_capture_message(message: JsonObject) -> tuple[JsonObject, int]:
    """Make one dumped message storable as ``jsonb``.

    Args:
        message: One ``GatewayMessage.model_dump(mode="json")`` object.

    Returns:
        The same object and 0 when nothing needed replacing; otherwise a copy
        with every unstorable code point replaced by U+FFFD (colliding
        sanitized keys disambiguated, never merged) and the replacement count,
        disambiguations included.
    """
    cleaned, total = _replace_in_object(message)
    return (message, 0) if total == 0 else (cleaned, total)


def serialize_capture_messages(request: GatewayRequest, *, request_id: str) -> str | None:
    """Serialize the canonical messages for capture, or None when oversized.

    Args:
        request: The canonical content-bearing request; read only.
        request_id: The minted request id, named in the replacement log line.

    Returns:
        A compact JSON array of the request's messages, every string storable
        as ``jsonb`` (see :func:`sanitize_capture_message`), or None when the
        serialized form exceeds the capture size cap.
    """
    dumped = [message.model_dump(mode="json", exclude_none=True) for message in request.messages]
    payload = json.dumps(dumped, separators=(",", ":"))
    if _UNSTORABLE_ESCAPE.search(payload):
        sanitized = [sanitize_capture_message(message) for message in dumped]
        replaced = sum(count for _, count in sanitized)
        if replaced:
            logger.warning(
                "prompt capture replaced %d unstorable code point(s) for request %s",
                replaced,
                request_id,
            )
            payload = json.dumps([message for message, _ in sanitized], separators=(",", ":"))
    # ``json.dumps`` escapes to ASCII, so the character count is the byte count.
    if len(payload) > _MAX_MESSAGES_BYTES:
        return None
    return payload


def serialize_capture_response(document: JsonObject, *, request_id: str) -> str | None:
    """Serialize one response document for capture, or None when oversized.

    The same storability rule as prompts (every string jsonb-safe, see
    :func:`sanitize_capture_message`) over the whole document; the size cap is
    the response column's 4 MiB. Nothing is summarized or projected out: a
    logprobs stream, tool calls, reasoning summaries, and the usage frame all
    stay exactly as the caller received them.
    """
    payload = json.dumps(document, separators=(",", ":"))
    if _UNSTORABLE_ESCAPE.search(payload):
        cleaned, replaced = _replace_in_object(document)
        if replaced:
            logger.warning(
                "response capture replaced %d unstorable code point(s) for request %s",
                replaced,
                request_id,
            )
            document = cleaned
    # Stored as UTF-8 (jsonb keeps the real code points), so the cap is
    # measured on the UTF-8 form, not on json.dumps' ASCII-escaped spelling ,
    # a CJK or emoji-heavy stream is not penalized for its escapes.
    payload = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    if len(payload.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return None
    return payload


def trim_sse_document(document: JsonObject) -> JsonObject | None:
    """Drop the tail of an oversized stream capture so a prefix is kept, marked truncated.

    Halves the retained frame count per step (frames stay in order); None when
    even a single frame does not fit, or the document is not a stream.
    """
    frames = document.get("frames")
    if document.get("kind") != "sse" or not isinstance(frames, list) or not frames:
        return None
    keep = len(frames)
    while keep > 1:
        keep //= 2
        candidate: JsonObject = {**document, "frames": list(frames[:keep]), "truncated": True}
        encoded = json.dumps(candidate, separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) <= _MAX_RESPONSE_BYTES:
            return candidate
    return None


def frame_payload(data: bytes) -> JsonValue:
    """One SSE data payload as JSON, or its text when it is not JSON."""
    try:
        return cast("JsonValue", json.loads(data))
    except ValueError:
        return data.decode("utf-8", "replace")


def sse_capture_document(
    frames: Sequence[bytes], *, truncated: bool, client_disconnected: bool
) -> JsonObject:
    """The stored shape of one relayed stream: its data payloads in order.

    ``client_disconnected`` marks a stream whose caller left before the
    terminal frames; ``frames`` are then exactly what the caller received.
    The key is present only when true, so a finished stream's document is
    unchanged.
    """
    document: JsonObject = {
        "kind": "sse",
        "status": 200,
        "frames": [frame_payload(data) for data in frames],
        "truncated": truncated,
    }
    if client_disconnected:
        document["client_disconnected"] = True
    return document


def enqueue_capture_response(
    writer: CaptureWriter, request_id: str, org_id: str, document: JsonObject
) -> bool:
    """Serialize one response document and queue it; True when it was handed over.

    A stream over the cap keeps its prefix, marked truncated, rather than
    losing the whole response; a single oversized body is dropped.
    """
    response_json = serialize_capture_response(document, request_id=request_id)
    if response_json is None:
        trimmed = trim_sse_document(document)
        response_json = (
            None if trimmed is None else serialize_capture_response(trimmed, request_id=request_id)
        )
        if response_json is None:
            logger.warning("response capture skipped for request %s: over the size cap", request_id)
            return False
        logger.warning("response capture truncated for request %s: over the size cap", request_id)
    writer.enqueue(
        ResponseCapturePayload(request_id=request_id, org_id=org_id, response_json=response_json)
    )
    return True


@dataclass(frozen=True)
class ParkedResponse:
    """The SSE data payloads a caller received before it disconnected."""

    frames: tuple[bytes, ...]
    truncated: bool
    size_bytes: int
    parked_at: float


class ResponseCaptureRegistry:
    """Bounded settle -> relay handoff naming the requests whose response to keep.

    The ledger records ``request_id -> org_id`` at the finalizing settle for
    capture-on orgs on non-BYOK lanes; the front pops the entry by the
    ``x-request-id`` the engine stamps on its response and hands the relayed
    body to the writer. FIFO-bounded like the cost registry: a keyed replay
    served from another worker finds no entry and captures nothing.

    The reverse order also happens: a caller that disconnects mid-stream ends
    the front's relay at once (Starlette cancels the streaming response on
    ``http.disconnect``), while the engine keeps serving the provider stream
    and settles it , as completed, billed , only when the provider finishes.
    The front then has the frames the caller received but no entry yet, so it
    PARKS them (:meth:`park`); the later settle's :meth:`record` finds the
    parked frames and hands them back instead of registering an entry nobody
    would claim. A settle that never comes (a cancelled or failed attempt, a
    capture-off org, a BYOK lane) leaves the parked frames to age out.
    """

    def __init__(
        self,
        *,
        capacity: int = _RESPONSE_REGISTRY_CAPACITY,
        ttl_seconds: float = _RESPONSE_REGISTRY_TTL_SECONDS,
        parked_ttl_seconds: float = _PARKED_TTL_SECONDS,
        parked_bytes_cap: int = _PARKED_BYTES_CAP,
        parked_capacity: int = _PARKED_CAPACITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty registry bounded by entry age, with a hard count cap behind it."""
        self._capacity = capacity
        self._ttl_seconds = ttl_seconds
        self._parked_ttl_seconds = parked_ttl_seconds
        self._parked_bytes_cap = parked_bytes_cap
        self._parked_capacity = parked_capacity
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._parked: OrderedDict[str, ParkedResponse] = OrderedDict()
        self._parked_bytes = 0

    def record(self, *, request_id: str, org_id: str) -> ParkedResponse | None:
        """Mark one request as response-capturing for its org.

        Returns the frames the front parked for this request when its caller
        disconnected before the settle; the caller then stores them itself
        (no entry is registered, since the relay is already over).
        """
        now = self._clock()
        with self._lock:
            self._expire_parked(now)
            parked = self._parked.pop(request_id, None)
            if parked is not None:
                self._parked_bytes -= parked.size_bytes
            else:
                self._entries[request_id] = (org_id, now)
                self._entries.move_to_end(request_id)
            expired = self._expire(now)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        _log_expired(expired)
        return parked

    def park(self, request_id: str, frames: Sequence[bytes], *, truncated: bool) -> str | None:
        """Keep the frames a disconnected caller received until the settle names the request.

        Both arrival orders are served: when the settle has ALREADY registered
        the request (its terminal frames were assembled just as the caller
        left), that entry is consumed and its org id returned, so the caller
        stores the frames now , nothing is parked. Otherwise the frames wait
        for :meth:`record`, bounded by age (the engine's request bound: after
        it the request has settled or been abandoned), by total accounted
        bytes and by entry count, oldest out first. Only the raw payload bytes
        are kept; nothing is parsed on the relay loop.
        """
        size = _PARKED_ENTRY_OVERHEAD_BYTES + sum(
            len(frame) + _PARKED_FRAME_OVERHEAD_BYTES for frame in frames
        )
        now = self._clock()
        with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry is not None:
                return entry[0]
            self._expire_parked(now)
            previous = self._parked.pop(request_id, None)
            if previous is not None:
                self._parked_bytes -= previous.size_bytes
            self._parked[request_id] = ParkedResponse(
                frames=tuple(frames), truncated=truncated, size_bytes=size, parked_at=now
            )
            self._parked_bytes += size
            while self._parked and (
                self._parked_bytes > self._parked_bytes_cap
                or len(self._parked) > self._parked_capacity
            ):
                _, evicted = self._parked.popitem(last=False)
                self._parked_bytes -= evicted.size_bytes
        return None

    def _expire_parked(self, now: float) -> None:
        """Drop parked frames older than the parked TTL (called under the lock)."""
        while self._parked:
            _, parked = next(iter(self._parked.items()))
            if now - parked.parked_at <= self._parked_ttl_seconds:
                break
            self._parked.popitem(last=False)
            self._parked_bytes -= parked.size_bytes

    def pop(self, request_id: str) -> str | None:
        """Consume the org for one request; None when it does not capture (or expired)."""
        now = self._clock()
        with self._lock:
            expired = self._expire(now)
            entry = self._entries.pop(request_id, None)
        _log_expired(expired)
        return None if entry is None else entry[0]

    def _expire(self, now: float) -> list[tuple[str, str, float]]:
        """Drop entries older than the TTL from the front; return them for logging.

        Called under the lock, so it only collects: an expiry is a served HTTP
        request whose response the front never claimed (the settle registered
        it), so a missing stored response is either a transport the tee does
        not see (Responses over WebSocket) or a front-side miss, and the log
        line the caller emits AFTER releasing the lock is what tells the two
        apart.
        """
        expired: list[tuple[str, str, float]] = []
        while self._entries:
            request_id, (org_id, recorded_at) = next(iter(self._entries.items()))
            if now - recorded_at <= self._ttl_seconds:
                break
            self._entries.popitem(last=False)
            expired.append((request_id, org_id, now - recorded_at))
        return expired


# One aggregated line per sweep, never under the registry lock: a backlog that
# crosses the TTL at once (a front-side claim outage) must not stall the relay
# loop with thousands of synchronous log calls.
_EXPIRED_IDS_LOGGED = 5


def _log_expired(expired: list[tuple[str, str, float]]) -> None:
    """Name the expired unclaimed entries (bounded sample, age range) in one log line.

    The age (seconds since the settle registered the request) is what tells a
    front-side miss from a slow relay: an entry expiring right at the TTL was
    never claimed, one far past it sat behind a stalled sweep.
    """
    if not expired:
        return
    ages = [age for _, _, age in expired]
    sample = ", ".join(
        f"{request_id} (org {org_id}, {age:.0f}s)"
        for request_id, org_id, age in expired[:_EXPIRED_IDS_LOGGED]
    )
    logger.info(
        "response capture: %d registry entr%s expired unclaimed (age %.0f-%.0fs): %s%s",
        len(expired),
        "y" if len(expired) == 1 else "ies",
        min(ages),
        max(ages),
        sample,
        ""
        if len(expired) <= _EXPIRED_IDS_LOGGED
        else f", +{len(expired) - _EXPIRED_IDS_LOGGED} more",
    )


class PromptCaptureBuffer:
    """Bounded authorize->accept handoff for capture payloads."""

    def __init__(self, *, capacity: int = _BUFFER_CAPACITY) -> None:
        """Create an empty buffer with an oldest-first eviction cap."""
        self._capacity = capacity
        self._entries: OrderedDict[str, PromptCapturePayload] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, payload: PromptCapturePayload) -> None:
        """Hold one request's capture payload until accept collects it."""
        with self._lock:
            self._entries[payload.request_id] = payload
            self._entries.move_to_end(payload.request_id)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def pop(self, request_id: str) -> PromptCapturePayload | None:
        """Collect and forget one request's payload; None when never seen."""
        with self._lock:
            return self._entries.pop(request_id, None)


class ResponseCaptureHandoff:
    """The front's one call: keep this relayed response if its request captures.

    Looks the request up in the settle-fed registry; a miss (capture-off org,
    BYOK lane, a replay from another worker, an engine error the ledger never
    finalized) captures nothing. Serialization happens on the relay's event
    loop thread AFTER the last byte left, never ahead of any byte.
    """

    def __init__(self, registry: ResponseCaptureRegistry, writer: CaptureWriter) -> None:
        """Bind the settle registry and the shared background writer."""
        self._registry = registry
        self._writer = writer

    def forget(self, request_id: str) -> None:
        """Release the registry entry of a request whose response yielded nothing to keep.

        An empty or malformed stream, an unparseable body, or a non-200 answer
        would otherwise leave its settle-created entry to age out by eviction,
        crowding out live requests' entries.
        """
        self._registry.pop(request_id)

    def claim(self, request_id: str) -> str | None:
        """Consume the settle's entry: the org id when the request captures, else None.

        The front claims BEFORE it parses or serializes anything, so a stream
        the settle never registered (BYOK, capture-off org, a replay from
        another worker) costs the relay no parsing at all.
        """
        return self._registry.pop(request_id)

    def capture(self, request_id: str, org_id: str, document: JsonObject) -> bool:
        """Enqueue ``document`` for a CLAIMED request; True when it was handed over."""
        return enqueue_capture_response(self._writer, request_id, org_id, document)

    def park(self, request_id: str, frames: Sequence[bytes], *, truncated: bool) -> None:
        """Park the frames a caller received before disconnecting, for the settle to claim.

        When the settle already named the request, the registry hands its org
        back and the frames are stored right away, marked as a disconnect
        (the relay is over, so nobody else will).
        """
        org_id = self._registry.park(request_id, frames, truncated=truncated)
        if org_id is not None:
            enqueue_capture_response(
                self._writer,
                request_id,
                org_id,
                sse_capture_document(frames, truncated=truncated, client_disconnected=True),
            )
