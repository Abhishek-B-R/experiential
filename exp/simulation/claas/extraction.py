"""Narrow, loss-aware extraction from captured Chat Completions and Responses bodies."""

from __future__ import annotations

import json
from dataclasses import dataclass

from exp.common.claas import Experience
from exp.common.core.artifacts import JsonObject, JsonValue, canonical_json_bytes, sha256_json
from exp.common.models import ModelMessage, ToolCall
from exp.common.tasks import ToolSchema
from exp.simulation.claas.contracts import EvidenceReference


@dataclass(frozen=True)
class ObservedToolResult:
    """A tool result whose error state is known only when explicitly reported."""

    call_id: str
    content: str
    is_error: bool | None
    evidence: EvidenceReference


def reference(experience: Experience, pointer: str) -> EvidenceReference:
    """Bind one protocol field to the exact source exchange."""
    return EvidenceReference(
        experience_id=experience.experience_id,
        experience_sha256=sha256_json(experience),
        pointer=pointer,
    )


def initial_messages(experience: Experience) -> tuple[ModelMessage, ...]:
    """Extract the initial text-only prompt, stopping before any observed answer.

    Args:
        experience: An original exchange, rather than an unresolved continuation.

    Returns:
        Initial system and user inputs without later answers or outcome labels.

    Raises:
        ValueError: Input includes unsupported content or no initial user request.
    """
    messages: list[ModelMessage] = []
    if experience.protocol == "responses":
        instructions = experience.request.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append(ModelMessage(role="system", content=instructions))
        raw = experience.request.get("input")
        if isinstance(raw, str):
            messages.append(ModelMessage(role="user", content=raw))
            return tuple(messages)
    else:
        raw = experience.request.get("messages")
    for item in _objects(raw):
        role = item.get("role")
        if role not in ("system", "developer", "user"):
            break
        content = _text(item.get("content"))
        messages.append(ModelMessage(role="user" if role == "user" else "system", content=content))
    if not any(message.role == "user" for message in messages):
        raise ValueError(
            "experience has no initial user text; include its original parent exchange"
        )
    return tuple(messages)


def tool_schemas(experience: Experience) -> tuple[ToolSchema, ...]:
    """Preserve function names, descriptions, and complete parameter schemas.

    Hosted tools and non-object function schemas need a dedicated environment adapter and are
    rejected rather than silently removed from the scenario's action space.
    """
    tools: list[ToolSchema] = []
    for item in _objects(experience.request.get("tools")):
        if item.get("type") != "function":
            raise ValueError(
                "CLaaS practice accepts function tools; supply a function-tool adapter"
            )
        body = item.get("function") if experience.protocol == "chat_completions" else item
        if not isinstance(body, dict):
            raise ValueError("captured function tool has no definition")
        name = body.get("name")
        parameters = body.get("parameters")
        description = body.get("description")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError("captured function tool needs a name and complete parameter schema")
        tools.append(
            ToolSchema(
                name=name,
                description=description if isinstance(description, str) and description else name,
                input_schema=parameters,
            )
        )
    return tuple(tools)


def tool_actions(experience: Experience) -> tuple[tuple[ToolCall, EvidenceReference], ...]:
    """Extract policy tool actions with pointers into the captured response."""
    result: list[tuple[ToolCall, EvidenceReference]] = []
    if experience.protocol == "responses":
        for index, item in enumerate(_objects(experience.response.get("output"))):
            if item.get("type") == "function_call":
                result.append(
                    (_call(item, "call_id"), reference(experience, f"/response/output/{index}"))
                )
    else:
        choices = _objects(experience.response.get("choices"))
        if len(choices) > 1:
            raise ValueError("mining multiple completion choices needs explicit branch selection")
        if choices:
            message = choices[0].get("message")
            if isinstance(message, dict):
                for index, item in enumerate(_objects(message.get("tool_calls"))):
                    function = item.get("function")
                    if not isinstance(function, dict):
                        raise ValueError("captured tool call has no function")
                    result.append(
                        (
                            _call({**function, "id": item.get("id")}, "id"),
                            reference(
                                experience, f"/response/choices/0/message/tool_calls/{index}"
                            ),
                        )
                    )
    return tuple(result)


def request_tool_actions(experience: Experience) -> tuple[tuple[ToolCall, EvidenceReference], ...]:
    """Read actions already present in one request's explicit conversation history.

    This supports standalone captured requests without joining their transcript
    prefixes to another exchange. Repeated history is deduplicated by call ID
    and exact action in the miner.
    """
    key = "messages" if experience.protocol == "chat_completions" else "input"
    raw = experience.request.get(key)
    if experience.protocol == "responses" and isinstance(raw, str):
        return ()
    result: list[tuple[ToolCall, EvidenceReference]] = []
    for index, item in enumerate(_objects(raw)):
        if experience.protocol == "responses" and item.get("type") == "function_call":
            result.append(
                (_call(item, "call_id"), reference(experience, f"/request/{key}/{index}"))
            )
        elif item.get("role") == "assistant":
            for call_index, call in enumerate(_objects(item.get("tool_calls"))):
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("captured history tool call has no function")
                result.append(
                    (
                        _call({**function, "id": call.get("id")}, "id"),
                        reference(experience, f"/request/{key}/{index}/tool_calls/{call_index}"),
                    )
                )
    return tuple(result)


def tool_results(experience: Experience) -> tuple[ObservedToolResult, ...]:
    """Extract tool results without interpreting arbitrary error-looking prose."""
    key = "messages" if experience.protocol == "chat_completions" else "input"
    results: list[ObservedToolResult] = []
    raw = experience.request.get(key)
    if experience.protocol == "responses" and isinstance(raw, str):
        return ()
    for index, item in enumerate(_objects(raw)):
        is_chat = item.get("role") == "tool"
        is_responses = item.get("type") == "function_call_output"
        if not is_chat and not is_responses:
            continue
        call_id = item.get("tool_call_id" if is_chat else "call_id")
        if not isinstance(call_id, str):
            raise ValueError("captured tool result has no call ID")
        content = item.get("content" if is_chat else "output")
        text = content if isinstance(content, str) else canonical_json_bytes(content).decode()
        raw_status = item.get("is_error")
        status: bool | None = raw_status if isinstance(raw_status, bool) else None
        structured = content
        if isinstance(content, str):
            try:
                structured = json.loads(content)
            except ValueError:
                structured = None
        if status is None and isinstance(structured, dict):
            flag = structured.get("is_error")
            if type(flag) is bool:
                status = flag
            elif structured.get("status") == "success":
                status = False
            elif structured.get("status") == "error":
                status = True
            elif structured.get("error"):
                status = True
        results.append(
            ObservedToolResult(
                call_id=call_id,
                content=text,
                is_error=status,
                evidence=reference(experience, f"/request/{key}/{index}"),
            )
        )
    return tuple(results)


def _objects(value: JsonValue | None) -> tuple[JsonObject, ...]:
    """Read a protocol list without silently dropping malformed members."""
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("captured protocol collection must contain only JSON objects")
    return tuple(item for item in value if isinstance(item, dict))


def _call(item: JsonObject, id_key: str) -> ToolCall:
    """Read one provider function call without inventing missing arguments."""
    arguments = item.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as exc:
            raise ValueError("captured tool arguments are not valid JSON") from exc
    name = item.get("name")
    call_id = item.get(id_key)
    if not isinstance(arguments, dict) or not isinstance(name, str) or not isinstance(call_id, str):
        raise ValueError("captured tool action needs its ID, name, and JSON-object arguments")
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _text(content: JsonValue | None) -> str:
    """Read visible text while rejecting unsupported multimodal seed inputs."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in _objects(content):
        text = item.get("text")
        if item.get("type") not in ("text", "input_text") or not isinstance(text, str):
            raise ValueError("CLaaS practice currently needs text inputs; use a multimodal adapter")
        parts.append(text)
    if not parts:
        raise ValueError("captured initial message has no text")
    return "".join(parts)
