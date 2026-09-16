"""Deterministic failure-pattern mining without manufacturing outcome labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.claas import Experience
from exp.common.core.artifacts import ContractModel, canonical_json_bytes, sha256_json, stable_id
from exp.common.models import ToolCall
from exp.simulation.claas.contracts import (
    ClaasScenario,
    EvidenceReference,
    ExperienceSignal,
    MinedScenario,
)
from exp.simulation.claas.extraction import (
    initial_messages,
    reference,
    request_tool_actions,
    tool_actions,
    tool_results,
    tool_schemas,
)


class MiningLimits(ContractModel):
    """Finite local selection work and the minimum repeated unchanged result count."""

    maximum_experiences: int = Field(default=256, ge=1, le=10_000, strict=True)
    maximum_source_bytes: int = Field(default=4_194_304, ge=1, strict=True)
    maximum_signals: int = Field(default=512, ge=1, strict=True)
    repeated_action_threshold: int = Field(default=3, ge=2, le=100, strict=True)


def mine_experiences(
    experiences: Sequence[Experience],
    *,
    partition: Literal["fit", "held_out"],
    limits: MiningLimits | None = None,
) -> tuple[MinedScenario, ...]:
    """Mine bounded, source-bound practice scenarios from explicitly selected traffic.

    Episode membership comes only from an explicit episode ID or Responses parent chain.
    Matching transcript prefixes never join unrelated callers. Selection patterns remain private
    to the result, while scenario messages contain only the initial request-visible prefix.

    Args:
        experiences: Complete source exchanges selected by the caller's fit/held-out split.
        partition: The caller's preassigned evidence partition, preserved on every scenario.
        limits: Explicit local work ceilings. Exceeding them rejects the entire operation.

    Returns:
        Deterministic scenarios with informative patterns and unknown outcome labels.

    Raises:
        ValueError: Sources are duplicated, malformed, too large, or have broken episode links.
    """
    bounds = limits or MiningLimits()
    if any(item.provenance.source_kind != "traffic" for item in experiences):
        raise ValueError(
            "mining requires observed traffic; synthetic or unverified imports are not evidence"
        )
    if len(experiences) > bounds.maximum_experiences:
        raise ValueError("mining experience limit exceeded; select a smaller source batch")
    if sum(len(canonical_json_bytes(item)) for item in experiences) > bounds.maximum_source_bytes:
        raise ValueError("mining source byte limit exceeded; select a smaller source batch")
    if len({item.experience_id for item in experiences}) != len(experiences):
        raise ValueError("mining requires unique experience IDs")
    by_response = {
        (item.scope.user_id, item.scope.application_id, item.response_id): item
        for item in experiences
    }
    if len(by_response) != len(experiences):
        raise ValueError("mining requires unique response IDs within an application scope")
    groups: dict[tuple[str, str, str], list[Experience]] = defaultdict(list)
    for item in experiences:
        root = _episode_root(item, by_response)
        groups[(item.scope.user_id, item.scope.application_id, root)].append(item)
    result: list[MinedScenario] = []
    signal_count = 0
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: (item.captured_at, item.experience_id))
        seed = group[0]
        messages = initial_messages(seed)
        tools = tool_schemas(seed)
        sources = tuple(reference(item, "/request") for item in group)
        scenario = ClaasScenario(
            scenario_id=stable_id(
                "claas-scenario",
                {
                    "scope": seed.scope.model_dump(mode="json"),
                    "partition": partition,
                    "sources": [item.model_dump(mode="json") for item in sources],
                },
            ),
            scope=seed.scope,
            partition=partition,
            messages=messages,
            tools=tools,
            sources=sources,
        )
        signals = _mine_group(group, bounds)
        signal_count += len(signals)
        if signal_count > bounds.maximum_signals:
            raise ValueError("mining signal limit exceeded; select a smaller source batch")
        result.append(MinedScenario(scenario=scenario, signals=signals))
    return tuple(result)


def _episode_root(
    item: Experience,
    by_response: dict[tuple[str, str, str], Experience],
) -> str:
    """Resolve explicit continuation ancestry without crossing application scopes."""
    if item.episode_id is not None:
        return "episode:" + item.episode_id
    visited: set[str] = set()
    while item.parent_response_id is not None:
        if item.response_id in visited:
            raise ValueError("captured Responses parent chain contains a cycle")
        visited.add(item.response_id)
        parent = by_response.get(
            (item.scope.user_id, item.scope.application_id, item.parent_response_id)
        )
        if parent is None:
            raise ValueError(
                "captured continuation is missing its parent; include the parent batch"
            )
        if parent.captured_at > item.captured_at:
            raise ValueError("captured continuation precedes its parent")
        item = parent
        if item.episode_id is not None:
            return "episode:" + item.episode_id
    return "exchange:" + item.experience_id


def _mine_group(group: Sequence[Experience], limits: MiningLimits) -> tuple[ExperienceSignal, ...]:
    """Identify local evidence patterns while retaining unknown task outcomes."""
    signals: list[ExperienceSignal] = []
    actions: dict[str, tuple[ToolCall, EvidenceReference]] = {}
    seen_results: dict[str, str] = {}
    errors: dict[str, EvidenceReference] = {}
    repeated_key: str | None = None
    repeated_refs: list[EvidenceReference] = []
    for experience in group:
        for action, evidence in request_tool_actions(experience):
            previous = actions.get(action.call_id)
            if previous is not None and previous[0] != action:
                raise ValueError("captured history has conflicting tool-call IDs")
            if previous is None:
                actions[action.call_id] = (action, evidence)
        for result in tool_results(experience):
            digest = sha256_json({"content": result.content, "is_error": result.is_error})
            prior = seen_results.get(result.call_id)
            if prior is not None:
                if prior != digest:
                    raise ValueError("a captured tool call has conflicting repeated results")
                continue
            seen_results[result.call_id] = digest
            pair = actions.get(result.call_id)
            if result.is_error is True:
                signals.append(
                    ExperienceSignal(
                        kind="tool_error",
                        evidence=(result.evidence,),
                        description="Tool reports an error; episode outcome remains unknown.",
                    )
                )
                if pair is not None:
                    errors[pair[0].name] = result.evidence
            elif result.is_error is False and pair is not None and pair[0].name in errors:
                signals.append(
                    ExperienceSignal(
                        kind="tool_recovery",
                        evidence=(errors.pop(pair[0].name), result.evidence),
                        description="The same tool later reports a non-error result.",
                    )
                )
            if pair is not None:
                action, action_ref = pair
                key = sha256_json(
                    {"name": action.name, "arguments": action.arguments, "result_sha256": digest}
                )
                if key == repeated_key:
                    repeated_refs.extend((action_ref, result.evidence))
                else:
                    repeated_key = key
                    repeated_refs = [action_ref, result.evidence]
                if len(repeated_refs) == 2 * limits.repeated_action_threshold:
                    signals.append(
                        ExperienceSignal(
                            kind="repeated_action",
                            evidence=tuple(repeated_refs),
                            description="Equivalent actions returned unchanged observations.",
                        )
                    )
        for action, evidence in tool_actions(experience):
            if action.call_id in actions:
                raise ValueError("captured episode repeats a policy tool-call ID")
            actions[action.call_id] = (action, evidence)
        signals.extend(_terminal_signals(experience))
    return tuple(signals)


def _terminal_signals(experience: Experience) -> tuple[ExperienceSignal, ...]:
    """Select explicit truncation, cancellation, failure, or empty terminal output."""
    response = experience.response
    pointer = "/response"
    reason: str | None = None
    kind: Literal["truncation", "early_termination"] = "early_termination"
    if experience.protocol == "responses":
        status = response.get("status")
        if status == "incomplete":
            details = response.get("incomplete_details")
            if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
                kind, reason = "truncation", "Provider exhausted its output-token limit."
            else:
                reason = "Provider marked the response incomplete without output-token exhaustion."
        elif status in ("failed", "cancelled"):
            reason = "Provider ended the response with an explicit failure or cancellation."
        elif status == "completed" and response.get("output") == []:
            reason = "Provider completed without visible output."
    else:
        choices = response.get("choices")
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            choice = choices[0]
            finish = choice.get("finish_reason")
            if finish == "length":
                kind, reason = "truncation", "Provider stopped at its output length limit."
            elif finish == "content_filter":
                reason = "Provider stopped the response through its content filter."
            elif finish == "stop":
                message = choice.get("message")
                if (
                    isinstance(message, dict)
                    and not message.get("content")
                    and not message.get("tool_calls")
                ):
                    reason = "Provider completed without visible output."
            pointer = "/response/choices/0"
    if reason is None:
        return ()
    return (
        ExperienceSignal(kind=kind, evidence=(reference(experience, pointer),), description=reason),
    )
