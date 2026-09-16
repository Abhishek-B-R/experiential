"""Optional traffic/world-model adapters implementing generic CLaaS environment contracts."""

from __future__ import annotations

from functools import partial
from typing import Literal

from exp.common.claas import Experience
from exp.common.claas.feedback import FeedbackRecord, FinalizedEpisode
from exp.common.claas.scenarios import EnvironmentTransition, Scenario
from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction
from exp.optimize.claas.evaluation.paired import (
    EvaluationManifest as CoreManifest,
)
from exp.optimize.claas.evaluation.paired import (
    EvaluationPolicy,
)
from exp.optimize.claas.evaluation.paired import (
    PairedEvaluationReport as CoreReport,
)
from exp.optimize.claas.evaluation.paired import (
    PolicyTaskEvaluation as CoreOutcome,
)
from exp.optimize.claas.lifecycle.rollouts import run_owned
from exp.optimize.workflows.traffic_learning import evaluation as traffic
from exp.simulation.claas.contracts import ClaasScenario
from exp.simulation.claas.harness import ClaasWorldModel, ClaasWorldSession
from exp.simulation.claas.provider import ClaasBoundedProvider
from exp.simulation.claas.source_feedback import select_source_feedback


def scenario_input(scenario: ClaasScenario, *, scenario_id: str | None = None) -> Scenario:
    """Expose public inputs and retain the traffic-specific scenario privately."""
    return Scenario(
        scenario_id=scenario_id or scenario.scenario_id,
        scope=scenario.scope,
        environment_id="traffic-world-model-v1",
        messages=scenario.messages,
        tools=scenario.tools,
        environment_data={"traffic_scenario": scenario.model_dump(mode="json")},
        source_kind="simulation",
        source_experience_ids=tuple(ref.experience_id for ref in scenario.sources),
    )


class TrafficEnvironment:
    """Resolve source grounding privately and reset the world model for each practice episode."""

    environment_id = "traffic-world-model-v1"

    def __init__(
        self,
        world: ClaasWorldModel,
        grounding: tuple[Experience, ...],
        *,
        feedback: tuple[FeedbackRecord, ...] = (),
        episodes: tuple[FinalizedEpisode, ...] = (),
    ) -> None:
        """Bind the authorized world model and fit-only original source records."""
        self.world = world
        self.sources = {item.experience_id: item for item in grounding}
        self.feedback = feedback
        self.episodes = episodes

    async def open(self, scenario: Scenario) -> TrafficSession:
        """Reset a grounded simulation and withhold historical feedback from the student."""
        source = ClaasScenario.model_validate(scenario.environment_data.get("traffic_scenario"))
        if scenario != scenario_input(source):
            raise ValueError("traffic scenario differs from its exact generic envelope")
        grounding = tuple(self.sources[ref.experience_id] for ref in source.sources)
        session = self.world.open(
            source,
            grounding=grounding,
            source_feedback=select_source_feedback(grounding, self.feedback, self.episodes),
        )
        return TrafficSession(session)


class TrafficSession:
    """Translate world-model steps while keeping private feedback outside visible messages."""

    def __init__(self, session: ClaasWorldSession) -> None:
        """Own one world-model session until close materializes its complete evidence."""
        self.session = session

    async def step(self, action: AssistantAction) -> EnvironmentTransition:
        """Execute a bounded synchronous world call without abandoning in-flight work."""
        result = await run_owned(partial(self.session.step, action))
        return EnvironmentTransition(
            messages=self.session.messages,
            terminal=result.transition.terminal,
            reward=result.transition.reward,
            feedback=result.transition.feedback,
        )

    async def close(self, reason: Literal["terminal", "step_limit", "failed"]) -> JsonObject:
        """Finalize the world receipt regardless of the generic consumer's ending reason."""
        return {"world_episode": self.session.end().model_dump(mode="json")}


class TrafficEvaluator:
    """Wrap frozen synthetic evaluation with its complete provider and replay validation."""

    def __init__(
        self,
        manifest: traffic.EvaluationManifest,
        world: ClaasWorldModel,
        judge: ClaasBoundedProvider,
    ) -> None:
        """Bind exact world/judge models to an already frozen traffic evaluation."""
        self.manifest = manifest
        self.world = world
        self.judge = judge
        if (
            world.purpose != "evaluation"
            or world.model != manifest.world_model
            or judge.model != manifest.judge_model
        ):
            raise ValueError("traffic evaluator providers differ from the frozen manifest")
        self.world.authorize_source(manifest.scope)
        self.judge.authorize_source(manifest.scope)

    @property
    def evaluator_id(self) -> str:
        """Bind traffic sources, world model, and judge to one immutable identity."""
        return "traffic-" + self.manifest.digest

    def freeze(self) -> CoreManifest:
        """Publish generic immutable tasks with the full traffic manifest retained privately."""
        return CoreManifest(
            scope=self.manifest.scope,
            evaluator_id=self.evaluator_id,
            score_kind="synthetic_judge",
            evaluator_settings={"traffic_manifest": self.manifest.model_dump(mode="json")},
            tasks=tuple(
                scenario_input(task.scenario, scenario_id=task.task_id)
                for task in self.manifest.tasks
            ),
        )

    async def evaluate(self, task: Scenario, policy: EvaluationPolicy) -> CoreOutcome:
        """Evaluate one original held-out task using the authorized world model and judge."""
        original = next(item for item in self.manifest.tasks if item.task_id == task.scenario_id)
        if task != scenario_input(original.scenario, scenario_id=original.task_id):
            raise ValueError("evaluation task differs from frozen traffic input")
        outcome = await traffic._evaluate_task(
            original,
            policy,
            policy.policy_revision,
            self.world,
            self.judge,
            self.manifest.rubric,
            120,
        )
        return CoreOutcome(
            task_id=outcome.task_id,
            policy_revision=outcome.policy_revision,
            score=outcome.judgment.score if outcome.judgment else None,
            failure_type=outcome.failure_type,
            evidence={"traffic_evaluation": outcome.model_dump(mode="json")},
        )

    def verify(self, manifest: CoreManifest, report: CoreReport) -> None:
        """Reconstruct and replay complete synthetic receipts before accepting any score."""
        if manifest != self.freeze():
            raise ValueError("generic evaluation differs from its frozen traffic manifest")
        pairs: list[traffic.PairedTaskEvaluation] = []
        for pair in report.pairs:
            outcomes: list[traffic.PolicyTaskEvaluation] = []
            for outcome in (pair.current, pair.candidate):
                if not outcome.evidence and outcome.failure_type:
                    converted = traffic.PolicyTaskEvaluation(
                        task_id=outcome.task_id,
                        policy_revision=outcome.policy_revision,
                        failure_stage="policy",
                        failure_type=outcome.failure_type,
                    )
                else:
                    converted = traffic.PolicyTaskEvaluation.model_validate(
                        outcome.evidence.get("traffic_evaluation")
                    )
                score = converted.judgment.score if converted.judgment else None
                if (
                    score != outcome.score
                    or converted.failure_type != outcome.failure_type
                    or converted.task_id != outcome.task_id
                    or converted.policy_revision != outcome.policy_revision
                ):
                    raise ValueError("generic score differs from synthetic judge receipt")
                outcomes.append(converted)
            pairs.append(
                traffic.PairedTaskEvaluation(
                    task_id=pair.task_id, current=outcomes[0], candidate=outcomes[1]
                )
            )
        original = traffic.PairedEvaluationReport(
            manifest_sha256=self.manifest.digest,
            current_policy_revision=report.current_policy_revision,
            candidate_policy_revision=report.candidate_policy_revision,
            expected_task_ids=report.expected_task_ids,
            pairs=tuple(pairs),
        )
        traffic.verify_evaluation_report(original, self.manifest)
