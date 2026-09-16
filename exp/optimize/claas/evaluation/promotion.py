"""Conservative activation decisions from complete frozen paired evaluation evidence."""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.optimize.claas.configuration import PromotionPolicy
from exp.optimize.claas.evaluation.paired import (
    EvaluationManifest,
    PairedEvaluationReport,
    TaskEvaluator,
    verify_evaluation_report,
)
from exp.runtime.claas.registry import RegistryState, ServingRevision


class PromotionDecision(ContractModel):
    """An auditable decision; only the serving lifecycle can actually load an adapter."""

    approved: bool
    reason: Literal["improved", "insufficient_tasks", "evaluation_failed", "no_improvement"]
    score_kind: str
    evaluation_manifest_sha256: Sha256
    report_sha256: Sha256
    baseline_generation: int
    current_policy_revision: str
    candidate_policy_revision: str
    task_count: int
    paired_mean_delta: float | None


def decide_promotion(
    *,
    manifest: EvaluationManifest,
    report: PairedEvaluationReport,
    baseline: RegistryState,
    candidate: ServingRevision,
    policy: PromotionPolicy,
    evaluator: TaskEvaluator,
) -> PromotionDecision:
    """Require complete paired improvement on the exact pretraining held-out manifest.

    Args:
        manifest: Frozen before training and reused across both policies.
        report: Both policies' complete per-task outcomes and evaluator evidence.
        baseline: Active registry generation captured before training.
        candidate: Verified artifact evaluated as the challenger.
        policy: Explicit minimum evidence and improvement requirements.

    Returns:
        A persisted decision with explicit score provenance, without changing serving state.

    Raises:
        ValueError: Report identity, task coverage, model identity, or score evidence has drifted.
    """
    manifest = EvaluationManifest.model_validate_json(manifest.model_dump_json())
    report = PairedEvaluationReport.model_validate_json(report.model_dump_json())
    _validate_identities(manifest, report, baseline, candidate)
    verify_evaluation_report(report, manifest, evaluator=evaluator)
    count = len(report.pairs)
    delta = report.paired_mean_delta
    candidate_failures = sum(pair.candidate.failure_type is not None for pair in report.pairs)
    if count < policy.minimum_evaluation_tasks:
        reason = "insufficient_tasks"
    elif delta is None or candidate_failures > policy.maximum_hard_failures:
        reason = "evaluation_failed"
    elif delta <= policy.minimum_score_improvement:
        reason = "no_improvement"
    else:
        reason = "improved"
    return PromotionDecision(
        approved=reason == "improved",
        reason=reason,
        score_kind=report.score_kind,
        evaluation_manifest_sha256=manifest.digest,
        report_sha256=sha256_json(report),
        baseline_generation=baseline.generation,
        current_policy_revision=baseline.active.policy_revision,
        candidate_policy_revision=candidate.policy_revision,
        task_count=count,
        paired_mean_delta=delta,
    )


def _validate_identities(
    manifest: EvaluationManifest,
    report: PairedEvaluationReport,
    baseline: RegistryState,
    candidate: ServingRevision,
) -> None:
    """Reject cross-scope, missing, duplicate, stale, or differently judged comparisons."""
    if manifest.scope != baseline.scope or candidate.scope != baseline.scope:
        raise ValueError("promotion evidence crosses application scope")
    for field in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
        if getattr(candidate, field) != getattr(baseline.active, field):
            raise ValueError("candidate changes the frozen base model or tokenizer")
    if (
        report.manifest_sha256 != manifest.digest
        or report.current_policy_revision != baseline.active.policy_revision
        or report.candidate_policy_revision != candidate.policy_revision
        or candidate.policy_revision == baseline.active.policy_revision
    ):
        raise ValueError("promotion report is not bound to this baseline and candidate")
    if tuple(pair.task_id for pair in report.pairs) != tuple(
        task.scenario_id for task in manifest.tasks
    ):
        raise ValueError("promotion requires every frozen task exactly once and in manifest order")
    if any(
        pair.current.policy_revision != report.current_policy_revision
        or pair.candidate.policy_revision != report.candidate_policy_revision
        for pair in report.pairs
    ):
        raise ValueError("per-task evaluated policy differs from the paired report")
