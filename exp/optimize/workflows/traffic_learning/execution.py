"""Explicit world-model providers and conservative budgets for traffic learning."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.optimize.claas.configuration import LocalClaasConfig
from exp.optimize.claas.execution import ExecutionSettings, validate_worker_runtime
from exp.optimize.claas.lifecycle.inputs import PreparedCycle
from exp.optimize.workflows.traffic_learning.composition import TrafficCycleSource, TrafficProviders
from exp.optimize.workflows.traffic_learning.configuration import TrafficWorkflowConfig
from exp.optimize.workflows.traffic_learning.sources.buffer import refresh_buffer
from exp.optimize.workflows.traffic_learning.sources.preparation import prepare_evidence
from exp.optimize.workflows.traffic_learning.sources.retention import prune_evidence
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.claas.harness import ClaasWorldModel, SourceDisclosure, WorldModelLimits
from exp.simulation.claas.provider import ClaasBoundedProvider


class TrafficProviderSettings(ContractModel):
    """Operator-declared conservative ceilings, including failures and retries."""

    maximum_world_call_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    maximum_judge_call_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    maximum_request_bytes: int = Field(default=65_536, strict=True, ge=1024, le=1_048_576)
    maximum_provider_output_tokens: int = Field(default=2048, strict=True, ge=1, le=65536)


def save_provider_settings(directory: Path, settings: TrafficProviderSettings) -> None:
    """Persist selected provider ceilings separately from the generic worker runtime."""
    write_text_atomic(
        directory / "traffic-providers.json", settings.model_dump_json(indent=2) + "\n"
    )


def load_provider_settings(directory: Path) -> TrafficProviderSettings:
    """Require explicit world-model call ceilings before provider authorization."""
    try:
        return TrafficProviderSettings.model_validate_json(
            (directory / "traffic-providers.json").read_bytes()
        )
    except FileNotFoundError:
        raise ValueError(
            "traffic providers are not configured; include traffic settings in claas bind"
        ) from None


class ProviderPlan(ContractModel):
    """Four independent reservoirs whose full scheduled worst-case costs are reserved."""

    synthesis: WorldModelLimits
    practice: WorldModelLimits
    evaluation: WorldModelLimits
    judge: WorldModelLimits

    @property
    def maximum_cost_usd(self) -> float:
        """Sum every reservation, including failed and retried calls."""
        return sum(
            item.maximum_total_cost_usd
            for item in (self.synthesis, self.practice, self.evaluation, self.judge)
        )


def provider_plan(
    config: LocalClaasConfig,
    settings: TrafficProviderSettings,
    runtime: ExecutionSettings,
    workflow: TrafficWorkflowConfig,
    *,
    evaluation_tasks: int,
) -> ProviderPlan:
    """Reserve the complete scheduled call counts before resolving any credentials."""
    limits = config.limits

    def bounds(count: int, cost: float) -> WorldModelLimits:
        """Reserve a finite count with exactly one conservative retry-inclusive call price."""
        return WorldModelLimits(
            maximum_steps=limits.maximum_episode_steps,
            maximum_model_calls=count,
            maximum_request_bytes=settings.maximum_request_bytes,
            maximum_output_tokens=settings.maximum_provider_output_tokens,
            maximum_total_cost_usd=count * cost,
            maximum_call_cost_usd=cost,
        )

    if not 1 <= evaluation_tasks <= workflow.maximum_source_experiences:
        raise ValueError(
            "frozen evaluation exceeds source limits; restore the prior configured limit"
        )
    plan = ProviderPlan(
        synthesis=bounds(limits.maximum_scenarios, settings.maximum_world_call_cost_usd),
        practice=bounds(
            limits.maximum_scenarios
            * limits.maximum_rollouts_per_scenario
            * limits.maximum_episode_steps,
            settings.maximum_world_call_cost_usd,
        ),
        evaluation=bounds(
            2 * evaluation_tasks * limits.maximum_episode_steps,
            settings.maximum_world_call_cost_usd,
        ),
        judge=bounds(2 * evaluation_tasks, settings.maximum_judge_call_cost_usd),
    )
    compute_cost = 0.0
    if config.compute == "modal":
        if runtime.modal is None:
            raise ValueError("Modal execution requires explicit modal settings")
        compute_cost = runtime.modal.estimated_maximum_cost_usd
        if runtime.modal.timeout_seconds > limits.maximum_training_seconds:
            raise ValueError("Modal execution timeout exceeds the configured training deadline")
    elif runtime.modal is not None:
        raise ValueError("local execution cannot carry unused Modal authorization")
    if plan.maximum_cost_usd + compute_cost > limits.maximum_cost_usd:
        raise ValueError(
            "scheduled provider and compute reservations exceed maximum_cycle_cost; "
            "reduce cycle work "
            "or explicitly raise the configured ceiling"
        )
    return plan


def build_providers(
    config: LocalClaasConfig,
    workflow: TrafficWorkflowConfig,
    plan: ProviderPlan,
    catalog: RuntimeModelCatalog,
) -> TrafficProviders:
    """Resolve authorized catalog credentials only after the caller obtains spend consent."""
    world = catalog.resolve(workflow.world_model_alias, role="world_model")
    judge = catalog.resolve(workflow.judge_alias, role="judge")
    disclosure = SourceDisclosure(scope=config.scope, model=world.snapshot)
    return TrafficProviders(
        synthesis=ClaasBoundedProvider(
            client=world.client,
            model=world.snapshot,
            limits=plan.synthesis,
            source_disclosure=disclosure,
        ),
        practice=ClaasWorldModel(
            client=world.client,
            model=world.snapshot,
            limits=plan.practice,
            source_disclosure=disclosure,
        ),
        evaluation=ClaasWorldModel(
            client=world.client,
            model=world.snapshot,
            limits=plan.evaluation,
            source_disclosure=disclosure,
            purpose="evaluation",
        ),
        judge=ClaasBoundedProvider(
            client=judge.client,
            model=judge.snapshot,
            limits=plan.judge,
            source_disclosure=SourceDisclosure(scope=config.scope, model=judge.snapshot),
        ),
    )


@dataclass(frozen=True)
class ConfiguredTrafficSource:
    """Refresh and budget one frozen cohort under the core's application cycle lock.

    Construction and reservation access do not resolve credentials. The CLI must
    obtain consent for the full configured cycle ceiling before passing this
    source to run_cycle. Preparation recomputes actual provider reservoirs from
    the current frozen cohort before constructing any provider clients.
    """

    config: LocalClaasConfig
    workflow: TrafficWorkflowConfig
    runtime: ExecutionSettings
    settings: TrafficProviderSettings
    catalog: RuntimeModelCatalog
    database_path: Path
    provider_factory: Callable[
        [LocalClaasConfig, TrafficWorkflowConfig, ProviderPlan, RuntimeModelCatalog],
        TrafficProviders,
    ] = build_providers

    @property
    def external_reservation_usd(self) -> float:
        """Reserve the entire provider allowance before any locked preparation work."""
        compute = self.runtime.modal.estimated_maximum_cost_usd if self.runtime.modal else 0.0
        return self.config.limits.maximum_cost_usd - compute

    async def prepare(self, directory: Path, config: LocalClaasConfig) -> PreparedCycle:
        """Select current retained sources, freeze their cohort, then authorize exact providers."""
        if config != self.config or self.workflow.scope != config.scope:
            raise ValueError("configured traffic source differs from its learning application")
        validate_worker_runtime(self.runtime, config)
        buffer = refresh_buffer(
            directory, self.database_path, config.scope, self.workflow.maximum_source_experiences
        )
        world, _ = self.catalog.snapshot(self.workflow.world_model_alias)
        judge, _ = self.catalog.snapshot(self.workflow.judge_alias)
        prune_evidence(
            directory,
            retained_source_ids={item.experience_id for item in buffer.experiences},
            maximum_cycles=31,
        )
        _, manifest = prepare_evidence(
            directory=directory,
            experiences=buffer.experiences,
            world_model=world,
            judge_model=judge,
            minimum_tasks=config.promotion.minimum_evaluation_tasks,
        )
        plan = provider_plan(
            config, self.settings, self.runtime, self.workflow, evaluation_tasks=len(manifest.tasks)
        )
        providers = self.provider_factory(config, self.workflow, plan, self.catalog)
        source = TrafficCycleSource(
            self.workflow, buffer.experiences, providers, buffer.feedback, buffer.episodes
        )
        if source.external_reservation_usd > self.external_reservation_usd:
            raise ValueError("constructed providers exceed the authorized cycle allowance")
        prepared = await source.prepare(directory, config)
        return replace(prepared, external_reservation_usd=self.external_reservation_usd)
