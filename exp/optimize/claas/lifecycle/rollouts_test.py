"""Exact authored-environment rollouts preserve provenance and teacher/student budgets."""

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from exp.common.claas.scenarios import EnvironmentTransition
from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction
from exp.optimize.claas.lifecycle.cycle import base_revision, training_spec
from exp.optimize.claas.lifecycle.cycle_test import (
    Admission,
    LookupEnvironment,
    LookupSession,
    Serving,
    config,
    scenario,
)
from exp.optimize.claas.lifecycle.rollouts import PracticeReceipt, RevisionPolicy, collect_practice
from exp.optimize.claas.training_contracts import validate_training_batch


@pytest.mark.parametrize("objective", ["sdpo", "hybrid", "reinforce"])
def test_arbitrary_environment_rollouts_preserve_original_tokens_and_budget(
    tmp_path: Path, objective: str
) -> None:
    """No traffic, world model, or source references are needed to produce a valid batch."""
    settings = config()
    settings = settings.model_copy(
        update={"limits": settings.limits.model_copy(update={"maximum_response_tokens": 256})}
    )
    admission = Admission()
    admission.paused = True
    serving = Serving(admission, base_revision(settings))
    environment = LookupEnvironment()
    per_example = 4 if objective == "reinforce" else 10
    spec = training_spec(settings).model_copy(
        update={"objective": objective, "max_batch_tokens": 2 * per_example + 1}
    )
    batch = asyncio.run(
        collect_practice(
            scenarios=(scenario("authored-1"), scenario("authored-2")),
            environment=environment,
            serving=serving,
            revision=base_revision(settings),
            spec=spec,
            limits=settings.limits,
            directory=tmp_path,
            cycle_id="authored-practice",
        )
    )
    validate_training_batch(spec, batch, None)
    assert len(batch.examples) == 2
    for example in batch.examples:
        assert example.experience.exact_tokens is not None
        assert example.experience.exact_tokens.prompt_token_ids == (1, 2)
        assert example.experience.exact_tokens.response_token_ids == (3, 4)
        assert example.experience.provenance.source_kind == "environment"
    assert environment.closed == environment.opened
    assert serving.response_limits
    assert set(serving.response_limits) == {256}


def test_revision_policy_supplies_the_selected_response_cap() -> None:
    """Evaluation forwards its frozen response budget through the serving contract."""
    settings = config()
    admission = Admission()
    admission.paused = True
    serving = Serving(admission, base_revision(settings))
    policy = RevisionPolicy(serving, base_revision(settings), maximum_response_tokens=64)
    task = scenario("evaluation")
    asyncio.run(policy.act(messages=task.messages, tools=task.tools, request_id="evaluation-0"))
    assert serving.response_limits == [64]


@pytest.mark.parametrize("execution_failed", [False, True])
def test_cleanup_failure_preserves_practice_evidence_and_refuses_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution_failed: bool
) -> None:
    """Completed actions survive a failed environment close, without producing a batch."""

    async def fail_close(
        self: LookupSession, reason: Literal["terminal", "step_limit", "failed"]
    ) -> JsonObject:
        """Simulate uncertain resource cleanup after the environment completed its actions."""
        del self, reason
        raise RuntimeError("cleanup unavailable")

    original_step = LookupSession.step

    async def fail_terminal_step(
        self: LookupSession, action: AssistantAction
    ) -> EnvironmentTransition:
        """Keep the first transition and fail the second before cleanup also fails."""
        if not action.tool_calls:
            raise ValueError("execution unavailable")
        return await original_step(self, action)

    monkeypatch.setattr(LookupSession, "close", fail_close)
    if execution_failed:
        monkeypatch.setattr(LookupSession, "step", fail_terminal_step)
    settings = config()
    admission = Admission()
    admission.paused = True
    serving = Serving(admission, base_revision(settings))
    expected_error = ValueError if execution_failed else RuntimeError
    expected_message = "execution unavailable" if execution_failed else "cleanup unavailable"
    with pytest.raises(expected_error, match=expected_message):
        asyncio.run(
            collect_practice(
                scenarios=(scenario("cleanup-failure"),),
                environment=LookupEnvironment(),
                serving=serving,
                revision=base_revision(settings),
                spec=training_spec(settings),
                limits=settings.limits,
                directory=tmp_path,
                cycle_id="practice",
            )
        )
    receipt = PracticeReceipt.model_validate_json((tmp_path / "practice-0-0.json").read_bytes())
    assert len(receipt.samples) == len(receipt.experiences) == 2
    assert len(receipt.episode.steps) == (1 if execution_failed else 2)
    assert receipt.episode.end_reason == "failed"
    expected_evidence: JsonObject = {
        "close_complete": False,
        "cleanup_failure_type": "RuntimeError",
        "execution_end_reason": "failed" if execution_failed else "terminal",
    }
    if execution_failed:
        expected_evidence["execution_failure_type"] = "ValueError"
    assert receipt.episode.evidence == expected_evidence


def test_cancellation_during_failed_episode_cleanup_preserves_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup cancellation overrides prior execution failure and preserves evidence."""
    closing = asyncio.Event()
    original_step = LookupSession.step

    async def fail_terminal_step(
        self: LookupSession, action: AssistantAction
    ) -> EnvironmentTransition:
        """Preserve one complete transition before the next execution fails."""
        if not action.tool_calls:
            raise ValueError("execution unavailable")
        return await original_step(self, action)

    async def wait_for_cancellation(
        self: LookupSession, reason: Literal["terminal", "step_limit", "failed"]
    ) -> JsonObject:
        """Expose a pending cleanup that only caller cancellation will interrupt."""
        del self, reason
        closing.set()
        await asyncio.Future[None]()
        raise AssertionError("cleanup unexpectedly completed")

    monkeypatch.setattr(LookupSession, "step", fail_terminal_step)
    monkeypatch.setattr(LookupSession, "close", wait_for_cancellation)
    settings = config()
    admission = Admission()
    admission.paused = True
    serving = Serving(admission, base_revision(settings))

    async def cancel_cleanup() -> None:
        """Cancel only once execution has failed and the environment is closing."""
        task = asyncio.create_task(
            collect_practice(
                scenarios=(scenario("cancel-cleanup"),),
                environment=LookupEnvironment(),
                serving=serving,
                revision=base_revision(settings),
                spec=training_spec(settings),
                limits=settings.limits,
                directory=tmp_path,
                cycle_id="cancel",
            )
        )
        await asyncio.wait_for(closing.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_cleanup())
    receipt = PracticeReceipt.model_validate_json((tmp_path / "cancel-0-0.json").read_bytes())
    assert len(receipt.samples) == len(receipt.experiences) == 2
    assert len(receipt.episode.steps) == 1
    assert receipt.episode.end_reason == "failed"
    assert receipt.episode.evidence == {
        "close_complete": False,
        "cleanup_failure_type": "CancelledError",
        "execution_end_reason": "failed",
        "execution_failure_type": "ValueError",
    }
