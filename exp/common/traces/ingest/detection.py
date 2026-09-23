"""Conservative trace-format recognition for interactive source selection."""

from pathlib import Path

import ijson

_RECORD_PREFIXES = frozenset(
    {
        "",
        "item",
        "conversations.item",
        "data.item",
        "results.item",
        "captures.item",
        "records.item",
        "spans.item",
    }
)
_CHAT_ROLES = frozenset({"system", "developer", "user", "human", "assistant", "tool"})
_SIGNATURE_FIELDS = frozenset(
    {
        "messages",
        "conversation",
        "resourceSpans",
        "request",
        "response",
        "traceId",
        "spanId",
        "trace_id",
        "span_id",
        "attributes",
        "role",
    }
)


def detect_trace_source(path: Path) -> str | None:
    """Recognize an unambiguous chat, native capture, or OpenTelemetry JSON export.

    The streaming scan retains only field shapes, never complete conversations or the
    corpus. Unknown, mixed, and malformed files need an explicit source choice. The
    selected canonical loader still owns validation and normalization of the full file.

    Args:
        path: Explicit operator-selected JSON or JSONL export.

    Returns:
        Recognized canonical source name, or None when selection is required.

    Raises:
        OSError: The selected file cannot be read.
    """
    matches: set[str] = set()
    records: dict[str, dict[str, str]] = {}
    try:
        with path.open("rb") as stream:
            for prefix, event, value in ijson.parse(stream, multiple_values=True):
                if prefix in _RECORD_PREFIXES:
                    if event == "start_map":
                        records[prefix] = {}
                    elif event == "end_map":
                        matches.update(_record_sources(prefix, records.pop(prefix, {})))
                parent, _, field = prefix.rpartition(".")
                if (
                    parent in records
                    and field in _SIGNATURE_FIELDS
                    and event in {"start_array", "start_map", "string"}
                ):
                    records[parent][field] = event
                    if field == "role" and event == "string" and value in _CHAT_ROLES:
                        records[parent]["chat_role"] = "string"
                if field == "protocol" and event == "string" and value == "chat_completions":
                    record = parent.removesuffix(".request") if parent != "request" else ""
                    if parent == "request" or parent.endswith(".request"):
                        if record in records:
                            records[record]["capture_protocol"] = "string"
    except (ijson.JSONError, UnicodeDecodeError):
        return None
    return next(iter(matches)) if len(matches) == 1 else None


def _record_sources(prefix: str, fields: dict[str, str]) -> set[str]:
    """Recognize supported record signatures without inspecting message or tool content."""
    matches: set[str] = set()
    if any(fields.get(key) == "start_array" for key in ("messages", "conversation")) or (
        prefix == "item" and "chat_role" in fields
    ):
        matches.add("chat-json")
    if fields.get("resourceSpans") == "start_array":
        matches.add("otlp")
    if (
        fields.get("request") == "start_map"
        and fields.get("response") == "start_map"
        and "capture_protocol" in fields
    ):
        matches.add("experiential")
    if all(fields.get(key) == "string" for key in ("traceId", "spanId")):
        matches.add("otel-genai" if fields.get("attributes") == "start_map" else "otlp")
    if all(fields.get(key) == "string" for key in ("trace_id", "span_id")):
        matches.add("otel-genai")
    return matches
