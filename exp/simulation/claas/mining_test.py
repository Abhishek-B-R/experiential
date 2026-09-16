"""Regression coverage for source-bound failure signals and safe scenario seeds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from exp.common.claas import ClaasScope, Experience, ExperienceProvenance
from exp.common.core.artifacts import JsonObject, sha256_json
from exp.simulation.claas import MiningLimits, mine_experiences


def make_experience(
    index: int = 0,
    *,
    tool_call: str | None = None,
    result_call: str | None = None,
    result_content: str = '{"is_error": false, "balance": 3}',
    finish_reason: str = "stop",
) -> Experience:
    """Build one small protocol exchange with explicit same-episode membership."""
    request: JsonObject = {
        "messages": [{"role": "user", "content": "Look up my claim."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up one claim.",
                    "parameters": {
                        "type": "object",
                        "properties": {"claim_id": {"type": "string"}},
                        "required": ["claim_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    if result_call is not None:
        request["messages"] = [
            {"role": "user", "content": "Look up my claim."},
            {"role": "assistant", "content": "SOURCE ANSWER MUST STAY PRIVATE"},
            {"role": "tool", "tool_call_id": result_call, "content": result_content},
        ]
    message: JsonObject = {"role": "assistant", "content": "Finished looking."}
    if tool_call is not None:
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"claim_id":"c1"}'},
                }
            ],
        }
    return Experience(
        experience_id=f"experience-{index}",
        response_id=f"response-{index}",
        episode_id="episode-1",
        scope=ClaasScope(user_id="user-1", application_id="claims"),
        protocol="chat_completions",
        captured_at=datetime(2026, 9, 16, tzinfo=UTC) + timedelta(seconds=index),
        request=request,
        response={"choices": [{"message": message, "finish_reason": finish_reason}]},
        provenance=ExperienceProvenance(
            source_kind="traffic", source_id="gateway", model_id="policy-1"
        ),
    )


def test_recovered_tool_error_is_useful_without_becoming_an_episode_failure() -> None:
    """A later explicit non-error result adds recovery evidence without success inference."""
    sources = (
        make_experience(0, tool_call="bad"),
        make_experience(
            1, tool_call="retry", result_call="bad", result_content='{"error":"temporary"}'
        ),
        make_experience(2, result_call="retry"),
    )
    mined = mine_experiences(sources, partition="fit")[0]
    assert [signal.kind for signal in mined.signals] == ["tool_error", "tool_recovery"]
    assert mined.outcome == "unknown"
    assert mined.scenario.messages[0].content == "Look up my claim."
    assert "SOURCE ANSWER" not in mined.scenario.model_dump_json()
    assert mined.scenario.tools[0].input_schema["additionalProperties"] is False
    recovery = mined.signals[1]
    assert recovery.evidence[0].experience_sha256 == sha256_json(sources[1])
    assert recovery.evidence[1].pointer == "/request/messages/2"


def test_repeated_equivalent_actions_require_unchanged_observations() -> None:
    """Call IDs do not hide repetitive actions, but changed observations break a run."""
    sources = (
        make_experience(0, tool_call="a"),
        make_experience(1, result_call="a", tool_call="b"),
        make_experience(2, result_call="b", tool_call="c"),
        make_experience(3, result_call="c"),
    )
    mined = mine_experiences(sources, partition="fit")[0]
    assert [signal.kind for signal in mined.signals] == ["repeated_action"]
    assert len(mined.signals[0].evidence) == 6
    changed = list(sources)
    changed[2] = make_experience(
        2, result_call="b", tool_call="c", result_content='{"is_error":false,"balance":4}'
    )
    assert mine_experiences(changed, partition="fit")[0].signals == ()


def test_clean_traffic_is_unknown_and_explicit_truncation_is_selected() -> None:
    """Neither absence of errors nor ordinary stop tokens are task-success evidence."""
    assert mine_experiences((make_experience(),), partition="fit")[0].outcome == "unknown"
    assert mine_experiences((make_experience(),), partition="fit")[0].signals == ()
    truncated = make_experience(finish_reason="length")
    assert mine_experiences((truncated,), partition="fit")[0].signals[0].kind == "truncation"


def test_scope_and_explicit_episode_membership_prevent_prefix_joins() -> None:
    """Identical transcripts remain separate without matching scope and episode authority."""
    first = make_experience().model_copy(update={"episode_id": None})
    second = make_experience(1).model_copy(update={"episode_id": None})
    assert len(mine_experiences((first, second), partition="fit")) == 2
    other_scope = second.model_copy(
        update={"scope": ClaasScope(user_id="user-2", application_id="claims")}
    )
    assert len(mine_experiences((first, other_scope), partition="fit")) == 2


def test_limits_reject_entire_batch_instead_of_truncating_evidence() -> None:
    """Bounded mining never silently emits evidence from a partial source batch."""
    with pytest.raises(ValueError, match="experience limit"):
        mine_experiences(
            (make_experience(), make_experience(1)),
            partition="fit",
            limits=MiningLimits(maximum_experiences=1),
        )
    with pytest.raises(ValueError, match="byte limit"):
        mine_experiences(
            (make_experience(),), partition="fit", limits=MiningLimits(maximum_source_bytes=1)
        )


def test_responses_parent_chain_and_early_termination() -> None:
    """Responses continuations join only their explicit original parent."""
    first = make_experience().model_copy(
        update={
            "protocol": "responses",
            "episode_id": None,
            "request": {"input": "Look up my claim."},
            "response": {"status": "completed", "output": []},
        }
    )
    second = make_experience(1).model_copy(
        update={
            "protocol": "responses",
            "episode_id": None,
            "parent_response_id": first.response_id,
            "request": {"input": "Continue."},
            "response": {"status": "incomplete", "output": []},
        }
    )
    result = mine_experiences((second, first), partition="held_out")
    assert len(result) == 1
    assert result[0].scenario.partition == "held_out"
    assert [signal.kind for signal in result[0].signals] == ["early_termination", "truncation"]
    with pytest.raises(ValueError, match="missing its parent"):
        mine_experiences((second,), partition="fit")
