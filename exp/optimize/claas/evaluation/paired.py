"""Frozen arbitrary scenarios and paired evaluation through an injected task evaluator."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field, model_validator

from exp.common.claas import ClaasScope
from exp.common.claas.scenarios import Scenario
from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema


class EvaluationPolicy(Protocol):
    """A policy bound to one immutable model and adapter revision."""

    @property
    def policy_revision(self) -> str:
        """Return the revision used throughout this attempt."""
        ...

    async def act(
        self, *, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> AssistantAction:
        """Generate an action from visible environment inputs only."""
        ...


class EvaluationManifest(ContractModel):
    """Caller-frozen tasks and evaluator configuration, independent of their origin."""

    scope: ClaasScope
    evaluator_id: str = Field(min_length=1, max_length=512)
    evaluator_settings: JsonObject = Field(default_factory=dict)
    score_kind: str = Field(min_length=1, max_length=128)
    tasks: tuple[Scenario, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_tasks(self) -> EvaluationManifest:
        """Reject duplicate identities and cross-application held-out tasks."""
        if len({task.scenario_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("evaluation task IDs must be unique")
        if any(task.scope != self.scope for task in self.tasks):
            raise ValueError("evaluation tasks cross application scope")
        return self

    @property
    def digest(self) -> str:
        """Bind the exact tasks and evaluator configuration before training."""
        return sha256_json(self)


class PolicyTaskEvaluation(ContractModel):
    """An evaluated task or explicit failure with evaluator-owned supporting evidence."""

    task_id: str
    policy_revision: str
    score: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    evidence: JsonObject = Field(default_factory=dict)
    failure_type: str | None = None

    @model_validator(mode="after")
    def _validate_score(self) -> PolicyTaskEvaluation:
        """Keep failed attempts in the denominator without assigning fabricated scores."""
        if (self.score is None) != bool(self.failure_type):
            raise ValueError("evaluation requires either a score or an explicit failure")
        return self


class PairedTaskEvaluation(ContractModel):
    """Two policy attempts on the same immutable task."""

    task_id: str
    current: PolicyTaskEvaluation
    candidate: PolicyTaskEvaluation

    @model_validator(mode="after")
    def _validate_pair(self) -> PairedTaskEvaluation:
        """Reject pairing outcomes from different tasks."""
        if self.current.task_id != self.task_id or self.candidate.task_id != self.task_id:
            raise ValueError("paired evaluation task IDs differ")
        return self


class PairedEvaluationReport(ContractModel):
    """Exact task coverage and revision-bound scores with no activation authority."""

    manifest_sha256: Sha256
    evaluator_id: str
    score_kind: str
    current_policy_revision: str
    candidate_policy_revision: str
    expected_task_ids: tuple[str, ...] = Field(min_length=1)
    pairs: tuple[PairedTaskEvaluation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pairs(self) -> PairedEvaluationReport:
        """Require complete, ordered task coverage and exact per-policy identities."""
        if self.current_policy_revision == self.candidate_policy_revision:
            raise ValueError("paired evaluation requires distinct policy revisions")
        if len(set(self.expected_task_ids)) != len(self.expected_task_ids):
            raise ValueError("paired evaluation contains duplicate task IDs")
        if tuple(pair.task_id for pair in self.pairs) != self.expected_task_ids:
            raise ValueError("paired evaluation must cover every frozen task in order")
        if any(
            pair.current.policy_revision != self.current_policy_revision
            or pair.candidate.policy_revision != self.candidate_policy_revision
            for pair in self.pairs
        ):
            raise ValueError("paired evaluation policy revisions differ")
        return self

    @property
    def paired_mean_delta(self) -> float | None:
        """Return improvement only when all frozen attempts have valid scores."""
        if any(pair.current.score is None or pair.candidate.score is None for pair in self.pairs):
            return None
        return sum(
            pair.candidate.score - pair.current.score
            for pair in self.pairs
            if pair.current.score is not None and pair.candidate.score is not None
        ) / len(self.pairs)


class TaskEvaluator(Protocol):
    """Explicit evaluator plugin owning environment execution and score evidence checks."""

    @property
    def evaluator_id(self) -> str:
        """Return the immutable implementation/configuration identity."""
        ...

    async def evaluate(self, task: Scenario, policy: EvaluationPolicy) -> PolicyTaskEvaluation:
        """Run one complete task, without disclosing private setup to the policy."""
        ...

    def verify(self, manifest: EvaluationManifest, report: PairedEvaluationReport) -> None:
        """Validate retained scores against exact environment or judge evidence."""
        ...


async def evaluate_policies(
    manifest: EvaluationManifest,
    *,
    current: EvaluationPolicy,
    candidate: EvaluationPolicy,
    evaluator: TaskEvaluator,
) -> PairedEvaluationReport:
    """Evaluate fresh episodes in alternating policy order on every frozen scenario."""
    manifest = EvaluationManifest.model_validate_json(manifest.model_dump_json())
    if evaluator.evaluator_id != manifest.evaluator_id:
        raise ValueError("evaluator differs from its frozen manifest")
    revisions = (current.policy_revision, candidate.policy_revision)
    if not all(item.strip() for item in revisions) or revisions[0] == revisions[1]:
        raise ValueError("evaluation requires two distinct nonempty policy revisions")
    pairs: list[PairedTaskEvaluation] = []
    for index, task in enumerate(manifest.tasks):
        results: dict[int, PolicyTaskEvaluation] = {}
        policies = (current, candidate)
        for side in (0, 1) if index % 2 == 0 else (1, 0):
            policy = policies[side]
            try:
                if policy.policy_revision != revisions[side]:
                    raise ValueError("policy revision changed during evaluation")
                outcome = await evaluator.evaluate(task, policy)
                if policy.policy_revision != revisions[side]:
                    raise ValueError("policy revision changed during evaluation")
                if (
                    outcome.task_id != task.scenario_id
                    or outcome.policy_revision != revisions[side]
                ):
                    raise ValueError("evaluator returned another task or policy identity")
                results[side] = outcome
            except Exception as error:  # noqa: BLE001 - failed tasks remain in the report
                results[side] = PolicyTaskEvaluation(
                    task_id=task.scenario_id,
                    policy_revision=revisions[side],
                    failure_type=type(error).__name__,
                )
        pairs.append(
            PairedTaskEvaluation(task_id=task.scenario_id, current=results[0], candidate=results[1])
        )
    report = PairedEvaluationReport(
        manifest_sha256=manifest.digest,
        evaluator_id=manifest.evaluator_id,
        score_kind=manifest.score_kind,
        current_policy_revision=revisions[0],
        candidate_policy_revision=revisions[1],
        expected_task_ids=tuple(task.scenario_id for task in manifest.tasks),
        pairs=tuple(pairs),
    )
    verify_evaluation_report(report, manifest, evaluator=evaluator)
    return report


def verify_evaluation_report(
    report: PairedEvaluationReport,
    manifest: EvaluationManifest,
    *,
    evaluator: TaskEvaluator,
) -> None:
    """Verify structural identity before delegating domain-specific score validation."""
    report = PairedEvaluationReport.model_validate_json(report.model_dump_json())
    manifest = EvaluationManifest.model_validate_json(manifest.model_dump_json())
    if (
        report.manifest_sha256 != manifest.digest
        or report.expected_task_ids != tuple(task.scenario_id for task in manifest.tasks)
        or report.evaluator_id != manifest.evaluator_id
        or report.score_kind != manifest.score_kind
        or evaluator.evaluator_id != manifest.evaluator_id
    ):
        raise ValueError("evaluation report differs from its frozen manifest")
    evaluator.verify(manifest, report)
