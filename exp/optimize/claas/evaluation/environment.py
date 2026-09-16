"""Paired-evaluation adapter for explicitly supplied environments and episode scorers."""

from __future__ import annotations

import asyncio
from typing import Literal, Protocol

from pydantic import Field

from exp.common.claas.scenarios import EnvironmentEpisode, EnvironmentStep, Scenario
from exp.common.core.artifacts import ContractModel, JsonObject, sha256_json
from exp.optimize.claas.evaluation.paired import (
    EvaluationManifest,
    EvaluationPolicy,
    PairedEvaluationReport,
    PolicyTaskEvaluation,
)
from exp.runtime.environments.learning import Environment


class EpisodeScore(ContractModel):
    """A domain scorer's bounded score and auditable supporting evidence."""

    score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    evidence: JsonObject


class EpisodeScorer(Protocol):
    """An executable verifier or judge selected independently of the learning core."""

    @property
    def scorer_id(self) -> str:
        """Return the immutable scoring configuration identity."""
        ...

    @property
    def score_kind(self) -> str:
        """Distinguish executable correctness, synthetic judgment, or other signals."""
        ...

    async def score(self, episode: EnvironmentEpisode) -> EpisodeScore:
        """Score a completed episode against its private task specification."""
        ...

    def verify(self, episode: EnvironmentEpisode, score: EpisodeScore) -> None:
        """Check the reported score against recorded or independently verifiable evidence."""
        ...


class EnvironmentEvaluator:
    """Run fresh bounded environment sessions and delegate task-specific scoring."""

    def __init__(
        self,
        environment: Environment,
        scorer: EpisodeScorer,
        *,
        maximum_steps: int = 8,
        operation_timeout_seconds: float = 120,
    ) -> None:
        """Freeze bounded lifecycle and scoring choices before either policy runs."""
        if not 1 <= maximum_steps <= 256 or not 0 < operation_timeout_seconds <= 3600:
            raise ValueError("evaluation steps or operation deadline exceeds supported bounds")
        self.environment = environment
        self.scorer = scorer
        self.maximum_steps = maximum_steps
        self.timeout = operation_timeout_seconds

    @property
    def settings(self) -> JsonObject:
        """Return all comparison settings that must remain frozen across revisions."""
        return {
            "environment_id": self.environment.environment_id,
            "scorer_id": self.scorer.scorer_id,
            "maximum_steps": self.maximum_steps,
            "operation_timeout_seconds": self.timeout,
        }

    @property
    def evaluator_id(self) -> str:
        """Bind configuration to a stable evaluator identity."""
        return "environment-" + sha256_json(self.settings)

    def freeze(self, tasks: tuple[Scenario, ...]) -> EvaluationManifest:
        """Freeze caller-supplied tasks without mining, synthesis, or provider selection."""
        if not tasks:
            raise ValueError("evaluation requires at least one scenario")
        if any(task.environment_id != self.environment.environment_id for task in tasks):
            raise ValueError("evaluation scenario selects a different environment")
        return EvaluationManifest(
            scope=tasks[0].scope,
            evaluator_id=self.evaluator_id,
            evaluator_settings=self.settings,
            score_kind=self.scorer.score_kind,
            tasks=tasks,
        )

    async def evaluate(self, task: Scenario, policy: EvaluationPolicy) -> PolicyTaskEvaluation:
        """Own reset, stepping, and cleanup while preserving explicit incomplete outcomes."""
        if task.environment_id != self.environment.environment_id:
            raise ValueError("evaluation scenario selects a different environment")
        revision = policy.policy_revision
        session = await asyncio.wait_for(self.environment.open(task), timeout=self.timeout)
        steps: list[EnvironmentStep] = []
        messages = task.messages
        reason: Literal["terminal", "step_limit", "failed"] = "failed"
        failure: str | None = None
        try:
            for index in range(self.maximum_steps):
                if policy.policy_revision != revision:
                    raise ValueError("policy revision changed during evaluation")
                action = await asyncio.wait_for(
                    policy.act(
                        messages=messages,
                        tools=task.tools,
                        request_id=f"{task.scenario_id}:{revision}:{index}",
                    ),
                    timeout=self.timeout,
                )
                transition = await asyncio.wait_for(session.step(action), timeout=self.timeout)
                steps.append(EnvironmentStep(action=action, transition=transition))
                messages = transition.messages
                if transition.terminal:
                    reason = "terminal"
                    break
            else:
                reason = "step_limit"
                failure = "StepLimitExceeded"
        except Exception as error:  # noqa: BLE001 - preserve failed task denominator
            failure = type(error).__name__
        finally:
            evidence = await asyncio.wait_for(session.close(reason), timeout=self.timeout)
        episode = EnvironmentEpisode(
            scenario=task, steps=tuple(steps), end_reason=reason, evidence=evidence
        )
        result_evidence: JsonObject = {"episode": episode.model_dump(mode="json")}
        if failure is not None:
            return PolicyTaskEvaluation(
                task_id=task.scenario_id,
                policy_revision=revision,
                evidence=result_evidence,
                failure_type=failure,
            )
        scored = await asyncio.wait_for(self.scorer.score(episode), timeout=self.timeout)
        self.scorer.verify(episode, scored)
        result_evidence["score"] = scored.model_dump(mode="json")
        return PolicyTaskEvaluation(
            task_id=task.scenario_id,
            policy_revision=revision,
            score=scored.score,
            evidence=result_evidence,
        )

    def verify(self, manifest: EvaluationManifest, report: PairedEvaluationReport) -> None:
        """Verify exact tasks, terminal transcripts, and domain-specific score evidence."""
        if (
            manifest.evaluator_settings != self.settings
            or manifest.score_kind != self.scorer.score_kind
        ):
            raise ValueError("environment evaluator settings differ from frozen evaluation")
        tasks = {task.scenario_id: task for task in manifest.tasks}
        for pair in report.pairs:
            for outcome in (pair.current, pair.candidate):
                if outcome.score is None:
                    continue
                episode = EnvironmentEpisode.model_validate(outcome.evidence.get("episode"))
                scored = EpisodeScore.model_validate(outcome.evidence.get("score"))
                if (
                    episode.scenario != tasks[pair.task_id]
                    or episode.end_reason != "terminal"
                    or scored.score != outcome.score
                ):
                    raise ValueError("evaluation score differs from complete frozen episode")
                self.scorer.verify(episode, scored)
