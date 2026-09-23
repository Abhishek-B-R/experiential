"""Typed evaluation inputs shared with router composition, without policy-fitting fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactId, ArtifactInput, ContractModel
from exp.common.evaluations import EvaluationProtocol, ObservedProductionCell
from exp.common.judging import Judge
from exp.common.models import ModelSnapshot, RoutedCandidateSnapshot
from exp.optimize.evaluation.simulation import SimulatorFactory
from exp.simulation.specs import WorldModelSettings


class EvaluationSetup(ContractModel):
    """Frozen worker, environment, judge and execution inputs independent of router fitting.

    Attributes:
        candidates: Nonempty collection of frozen worker model identities.
        observed_cells: Historical cells, empty for standalone model evaluation.
        production_protocol: Build-bound production evidence protocol.
        simulation_protocol: Shared worker simulation and judging protocol.
        fit_rag_input: Immutable fit-only retrieval index.
        pricing_snapshot_id: Frozen catalog prices used for the comparison.
        judgment_status: Actual provisional or human-calibrated provenance.
        world_model_settings: Grounded environment and retrieval configuration.
        simulation_completion_input: Frozen request reservations, required for quoting.
        agent_id: Selected runtime identity, from 1 through 256 characters.
        seed: Reproducible simulation seed.
        maximum_steps: Positive per-rollout step ceiling.
        continuation_of: Exact parent specification, or None for a fresh evaluation.
        maximum_rollout_output_tokens: Positive cumulative worker output cap, default 1,000,000.
        maximum_concurrency: Positive ceiling on concurrently admitted rollouts.
    """

    candidates: tuple[RoutedCandidateSnapshot, ...] = Field(min_length=1)
    observed_cells: tuple[ObservedProductionCell, ...] = ()
    production_protocol: EvaluationProtocol
    simulation_protocol: EvaluationProtocol
    fit_rag_input: ArtifactInput
    pricing_snapshot_id: ArtifactId
    judgment_status: Literal["provisional", "human_calibrated"]
    world_model_settings: WorldModelSettings
    simulation_completion_input: ArtifactInput | None = None
    agent_id: str = Field(min_length=1, max_length=256)
    seed: int
    maximum_steps: int = Field(gt=0)
    continuation_of: ArtifactInput | None = None
    maximum_rollout_output_tokens: int = Field(default=1_000_000, gt=0)
    maximum_concurrency: int = Field(gt=0)


class EvaluationBudget(ContractModel):
    """Finite ceilings for simulation plus judging; execution never opts out of enforcement.

    Attributes:
        maximum_cost_usd: Positive finite provider-spend ceiling.
        maximum_judgments: Positive ceiling on durable cell judgments.
    """

    maximum_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    maximum_judgments: int = Field(gt=0)


class EvaluationExecutionContract(ArtifactEnvelope):
    """Hash-bound execution settings and authorization included in the evaluation identity.

    Attributes:
        contract_id: Content-derived execution identity.
        setup: Frozen model, environment and judge inputs.
        budget: Authorized finite execution ceilings.
    """

    contract_id: ArtifactId
    setup: EvaluationSetup
    budget: EvaluationBudget


class JudgmentReferences(Protocol):
    """Artifact identities consumed by the shared durable judgment executor."""

    @property
    def rubric_id(self) -> str:
        """Return the frozen rubric artifact identity."""

    @property
    def calibration_id(self) -> str:
        """Return the frozen calibration artifact identity."""


@dataclass(frozen=True)
class EvaluationJudge:
    """Verified persisted judge references, never caller-authored calibration evidence.

    Attributes:
        rubric_id: Verified immutable rubric identity.
        calibration_id: Verified immutable calibration identity.
    """

    rubric_id: str
    calibration_id: str


@dataclass(frozen=True)
class EvaluationServices:
    """Injected model-backed services; evaluation orchestration remains Experiential-owned.

    Attributes:
        simulator_factory: Builds the selected simulation engine for one frozen plan.
        judge: Provider-bound, reservation-enforcing judge.
        plan_inputs: Additional immutable execution inputs, empty by default.
    """

    simulator_factory: SimulatorFactory
    judge: EvaluationRuntimeJudge
    plan_inputs: tuple[ArtifactInput, ...] = ()


class EvaluationRuntimeJudge(Judge, Protocol):
    """Judge whose configured provider identity can be checked before any paid work."""

    @property
    def model(self) -> ModelSnapshot:
        """Return the exact model bound by the runtime's reservation-enforcing client."""
