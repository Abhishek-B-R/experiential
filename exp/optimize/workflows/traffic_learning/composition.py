"""Optional traffic mining and world-model synthesis composed above generic CLaaS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from exp.common.claas import Experience
from exp.common.claas.feedback import FeedbackRecord, FinalizedEpisode
from exp.optimize.claas.configuration import LocalClaasConfig
from exp.optimize.claas.lifecycle.cycle import AdmissionController, CycleState, run_cycle
from exp.optimize.claas.lifecycle.inputs import PreparedCycle
from exp.optimize.claas.lifecycle.rollouts import ServingController
from exp.optimize.claas.training_contracts import ClaasTrainingBackend
from exp.optimize.workflows.traffic_learning.adapters import (
    TrafficEnvironment,
    TrafficEvaluator,
    scenario_input,
)
from exp.optimize.workflows.traffic_learning.calls import run_owned
from exp.optimize.workflows.traffic_learning.configuration import TrafficWorkflowConfig
from exp.optimize.workflows.traffic_learning.sources.preparation import prepare_evidence
from exp.optimize.workflows.traffic_learning.sources.retention import prune_evidence
from exp.simulation.claas.harness import ClaasWorldModel
from exp.simulation.claas.provider import ClaasBoundedProvider
from exp.simulation.claas.synthesis import fit_scenarios, synthesize_scenarios


@dataclass(frozen=True)
class TrafficProviders:
    """Explicit bounded providers selected only by this optional experience workflow."""

    synthesis: ClaasBoundedProvider
    practice: ClaasWorldModel
    evaluation: ClaasWorldModel
    judge: ClaasBoundedProvider


async def run_traffic_cycle(
    *,
    directory: Path,
    config: LocalClaasConfig,
    workflow: TrafficWorkflowConfig,
    experiences: tuple[Experience, ...],
    providers: TrafficProviders,
    serving: ServingController,
    admission: AdmissionController,
    backend_factory: Callable[[str], ClaasTrainingBackend],
    feedback: tuple[FeedbackRecord, ...] = (),
    episodes: tuple[FinalizedEpisode, ...] = (),
    compute_reservation_usd: float = 0.0,
) -> CycleState:
    """Prepare a frozen traffic split, synthesize fit tasks, then call the generic cycle.

    The core's application lock owns preparation through activation, preventing
    hold-out capture and source changes from racing scenario selection.
    Reservations cover synthesis before it dispatches as well as later practice.
    """
    return await run_cycle(
        directory=directory,
        config=config,
        plan=TrafficCycleSource(workflow, experiences, providers, feedback, episodes),
        serving=serving,
        admission=admission,
        backend_factory=backend_factory,
        compute_reservation_usd=compute_reservation_usd,
    )


@dataclass(frozen=True)
class TrafficCycleSource:
    """Prepare traffic-derived tasks while the generic cycle owns its application lock."""

    workflow: TrafficWorkflowConfig
    experiences: tuple[Experience, ...]
    providers: TrafficProviders
    feedback: tuple[FeedbackRecord, ...] = ()
    episodes: tuple[FinalizedEpisode, ...] = ()

    @property
    def external_reservation_usd(self) -> float:
        """Declare all provider work before source preparation or any paid dispatch."""
        return sum(
            item.limits.maximum_total_cost_usd
            for item in (
                self.providers.synthesis,
                self.providers.practice,
                self.providers.evaluation,
                self.providers.judge,
            )
        )

    async def prepare(self, directory: Path, config: LocalClaasConfig) -> PreparedCycle:
        """Mine only retained fit sources, freezing held-out identities before synthesis."""
        if config.scope != self.workflow.scope or any(
            item.scope != config.scope for item in self.experiences
        ):
            raise ValueError("traffic workflow sources or configuration cross application scope")
        if len(self.experiences) > self.workflow.maximum_source_experiences:
            raise ValueError("traffic workflow exceeds its configured source ceiling")
        providers = self.providers
        prune_evidence(
            directory,
            retained_source_ids={item.experience_id for item in self.experiences},
            maximum_cycles=31,
        )
        split, frozen = prepare_evidence(
            directory=directory,
            experiences=self.experiences,
            world_model=providers.evaluation.model,
            judge_model=providers.judge.model,
            minimum_tasks=config.promotion.minimum_evaluation_tasks,
        )
        if providers.practice.model != frozen.world_model:
            raise ValueError("practice and evaluation must use the same frozen world model")
        generated = await run_owned(
            partial(
                synthesize_scenarios,
                split,
                provider=providers.synthesis,
                maximum_scenarios=config.limits.maximum_scenarios,
            )
        )
        synthesized_ids = {item.scenario.scenario_id for item in generated}
        seeds = [
            item
            for item in fit_scenarios(split, generated)
            if item.scenario_id not in synthesized_ids
        ]
        mixed = []
        for index, seed in enumerate(seeds):
            mixed.append(seed)
            if index < len(generated):
                mixed.append(generated[index].scenario)
        evaluator = TrafficEvaluator(frozen, providers.evaluation, providers.judge)
        return PreparedCycle(
            scenarios=tuple(scenario_input(item) for item in mixed),
            environment=TrafficEnvironment(
                providers.practice, split.fit, feedback=self.feedback, episodes=self.episodes
            ),
            evaluation=evaluator.freeze(),
            evaluator=evaluator,
            external_reservation_usd=self.external_reservation_usd,
            evidence={
                "workflow": "traffic-learning-v1",
                "split": split.model_dump(mode="json"),
                "synthesis": [item.model_dump(mode="json") for item in generated],
            },
        )
