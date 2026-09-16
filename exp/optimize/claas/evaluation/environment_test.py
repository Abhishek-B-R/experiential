"""Environment cleanup failures retain task evidence and prevent scoring or promotion."""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from exp.common.claas import ClaasScope
from exp.common.claas.scenarios import EnvironmentEpisode, EnvironmentTransition, Scenario
from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema
from exp.optimize.claas.configuration import PromotionPolicy
from exp.optimize.claas.evaluation.environment import EnvironmentEvaluator, EpisodeScore
from exp.optimize.claas.evaluation.paired import evaluate_policies
from exp.optimize.claas.evaluation.promotion import decide_promotion
from exp.runtime.claas.registry import RegistryState, ServingRevision


class Policy:
    """Return a deterministic action with a caller-selected immutable revision."""

    def __init__(self, revision: str) -> None:
        """Bind one revision for every action in this attempt."""
        self.policy_revision = revision

    async def act(
        self, *, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> AssistantAction:
        """Produce an action without reading the environment's private failure controls."""
        return AssistantAction(content="Continue the task.")


class UnusedScorer:
    """Record attempts to score an episode whose cleanup did not complete."""

    scorer_id = "cleanup-check-v1"
    score_kind = "executable"

    def __init__(self) -> None:
        """Start without scoring or verification calls."""
        self.calls = 0

    async def score(self, episode: EnvironmentEpisode) -> EpisodeScore:
        """Reject scoring because every fixture has an incomplete lifecycle."""
        self.calls += 1
        raise AssertionError("failed cleanup must prevent scoring")

    def verify(self, episode: EnvironmentEpisode, score: EpisodeScore) -> None:
        """Reject verification because no fixture may produce a score."""
        raise AssertionError("failed cleanup must prevent score verification")


class CleanupEnvironment:
    """Run one retained transition before a terminal or failed execution and bad cleanup."""

    environment_id = "cleanup-evidence-v1"

    def __init__(
        self,
        cleanup: Literal["raise", "timeout"],
        *,
        fail_step: bool = False,
        cancel_step: bool = False,
    ) -> None:
        """Choose cleanup behavior independently of the preceding environment execution."""
        self.cleanup = cleanup
        self.fail_step = fail_step
        self.cancel_step = cancel_step
        self.close_reasons: list[str] = []
        self.close_started = asyncio.Event()
        self.close_cancelled = False

    async def open(self, scenario: Scenario) -> CleanupSession:
        """Reset the transition counter for each policy's independent attempt."""
        return CleanupSession(self, scenario)


class CleanupSession:
    """Preserve a visible first transition while allowing later execution or cleanup failure."""

    def __init__(self, environment: CleanupEnvironment, scenario: Scenario) -> None:
        """Bind the selected failure behavior and private scenario."""
        self.environment = environment
        self.scenario = scenario
        self.steps = 0

    async def step(self, action: AssistantAction) -> EnvironmentTransition:
        """Return one recorded observation before any requested execution failure."""
        self.steps += 1
        if self.environment.cancel_step:
            raise asyncio.CancelledError
        if self.environment.fail_step and self.steps > 1:
            raise ValueError("PRIVATE execution exception")
        return EnvironmentTransition(
            messages=(ModelMessage(role="user", content="Observed progress."),),
            terminal=not self.environment.fail_step,
        )

    async def close(self, reason: Literal["terminal", "step_limit", "failed"]) -> JsonObject:
        """Record the execution ending and fail or wait until the cleanup deadline."""
        self.environment.close_reasons.append(reason)
        self.environment.close_started.set()
        if self.environment.cleanup == "timeout":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.environment.close_cancelled = True
                raise
        raise RuntimeError("PRIVATE cleanup exception")


def scenario() -> Scenario:
    """Provide an authored task independent of traffic, provider clients, or synthetic worlds."""
    return Scenario(
        scope=ClaasScope(user_id="user", application_id="app"),
        scenario_id="task",
        environment_id=CleanupEnvironment.environment_id,
        messages=(ModelMessage(role="user", content="Complete the task."),),
    )


@pytest.mark.parametrize("cleanup", ["raise", "timeout"])
@pytest.mark.parametrize("fail_step", [False, True])
def test_cleanup_failure_preserves_episode_and_blocks_promotion(
    cleanup: Literal["raise", "timeout"], fail_step: bool
) -> None:
    """Keep completed steps and the first failure while recording a separate cleanup error."""

    async def run() -> None:
        """Exercise the paired evaluator and promotion boundary with both failed policies."""
        task = scenario()
        environment = CleanupEnvironment(cleanup, fail_step=fail_step)
        scorer = UnusedScorer()
        evaluator = EnvironmentEvaluator(environment, scorer, operation_timeout_seconds=0.01)
        manifest = evaluator.freeze((task,))
        report = await evaluate_policies(
            manifest, current=Policy("current"), candidate=Policy("candidate"), evaluator=evaluator
        )
        cleanup_error = "RuntimeError" if cleanup == "raise" else "TimeoutError"
        for outcome in (report.pairs[0].current, report.pairs[0].candidate):
            assert outcome.score is None
            assert outcome.failure_type == ("ValueError" if fail_step else cleanup_error)
            episode = EnvironmentEpisode.model_validate(outcome.evidence["episode"])
            assert episode.evidence["cleanup_failure_type"] == cleanup_error
            assert episode.evidence["close_complete"] is False
            assert episode.evidence["execution_end_reason"] == (
                "failed" if fail_step else "terminal"
            )
            assert episode.evidence.get("execution_failure_type") == (
                "ValueError" if fail_step else None
            )
            assert len(episode.steps) == 1
            assert episode.messages[-1].content == "Observed progress."
            assert episode.end_reason == "failed"
            assert "PRIVATE" not in outcome.model_dump_json()
        assert environment.close_reasons == ["failed" if fail_step else "terminal"] * 2
        assert scorer.calls == 0
        assert report.paired_mean_delta is None
        current = ServingRevision(
            scope=task.scope,
            policy_revision="current",
            model_id="model",
            model_revision="revision",
            tokenizer_id="tokenizer",
            tokenizer_revision="revision",
        )
        decision = decide_promotion(
            manifest=manifest,
            report=report,
            baseline=RegistryState(scope=task.scope, generation=0, active=current),
            candidate=current.model_copy(update={"policy_revision": "candidate"}),
            policy=PromotionPolicy(minimum_evaluation_tasks=1),
            evaluator=evaluator,
        )
        assert not decision.approved
        assert decision.reason == "evaluation_failed"

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["execution", "cleanup"])
def test_cancellation_remains_cancellation_even_when_cleanup_fails(
    phase: Literal["execution", "cleanup"],
) -> None:
    """Do not convert cancellation into an ordinary failed or scored task receipt."""

    async def run() -> None:
        """Cancel execution or in-flight cleanup and confirm the scorer stays unused."""
        environment = CleanupEnvironment(
            "raise" if phase == "execution" else "timeout", cancel_step=phase == "execution"
        )
        scorer = UnusedScorer()
        evaluator = EnvironmentEvaluator(environment, scorer)
        task = asyncio.create_task(evaluator.evaluate(scenario(), Policy("current")))
        await asyncio.wait_for(environment.close_started.wait(), timeout=1)
        if phase == "cleanup":
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scorer.calls == 0
        assert len(environment.close_reasons) == 1
        if phase == "cleanup":
            assert environment.close_cancelled

    asyncio.run(run())
