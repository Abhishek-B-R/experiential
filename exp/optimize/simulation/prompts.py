"""One-component GEPA experiments with explicit development splits and fixed evaluators."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, sha256_json

if TYPE_CHECKING:
    from gepa.core.adapter import EvaluationBatch

PromptComponent = Literal[
    "behavioral_library",
    "construction",
    "initialization",
    "prediction",
    "jev",
    "research",
    "fidelity_judge",
    "actor_equivalence_judge",
]


class PromptCase(ContractModel):
    """One development example; confirmation data is absent from this optimizer interface.

    Attributes:
        case_id: Stable example identity.
        lineage_id: Shared source or fixture family, disjoint across selection splits.
        input: Permitted development inputs supplied to the declared evaluator.
    """

    case_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    input: JsonObject


class PromptEvaluation(ContractModel):
    """Actual execution outcome and grounded feedback, including failed examples.

    Attributes:
        score: Fixed evaluator metric in [0, 1], with unavailable execution scoring zero.
        status: Actual execution/evaluation status, retained separately from score.
        output: Exact outputs or references retained by the evaluator.
        feedback: Development-only observations for reflective mutation.
    """

    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["completed", "failed", "unassessable"]
    output: JsonObject
    feedback: str

    @model_validator(mode="after")
    def unavailable(self) -> PromptEvaluation:
        """Do not reward failed evaluations by omitting them from the optimizer metric."""
        if self.status != "completed" and self.score != 0:
            raise ValueError("unavailable evaluations must retain zero optimizer credit")
        return self


class PromptStudy(ContractModel):
    """Prospective identity for one prompt's baseline-versus-GEPA comparison.

    Attributes:
        study_id: Unique study identity, not reused for hidden restarts.
        component: Exactly one prompt slot optimized; other prompts remain fixed.
        purpose: Simulator optimization or separately registered judge calibration.
        frozen_pipeline_sha256: Non-target prompts, actor, data policy and runtime identity.
        evaluator_sha256: Fixed metric, rubric and reference-access policy identity.
        maximum_metric_calls: GEPA evaluation ceiling, additional to provider-wide budgets.
        seed: Registered optimizer random seed.
    """

    study_id: str = Field(min_length=1)
    component: PromptComponent
    purpose: Literal["simulator", "judge_calibration"] = "simulator"
    frozen_pipeline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_metric_calls: int = Field(ge=2, strict=True)
    seed: int = Field(default=0, strict=True)

    @model_validator(mode="after")
    def judge_scope(self) -> PromptStudy:
        """Judge prompts can optimize calibration agreement, never favorable simulator ratings."""
        judge = self.component in ("fidelity_judge", "actor_equivalence_judge")
        if judge != (self.purpose == "judge_calibration"):
            raise ValueError("judge and simulator optimization require separate studies")
        return self


class _Adapter:
    """Adapt actual pipeline evaluations to GEPA without taking provider ownership."""

    propose_new_texts = None

    def __init__(
        self,
        study: PromptStudy,
        evaluate: Callable[[str, PromptCase], PromptEvaluation],
        record: Callable[[JsonObject], None],
    ) -> None:
        """Bind the frozen component evaluator and durable experiment receipt sink."""
        self.study, self._evaluate, self._record = study, evaluate, record
        self._metric_calls = 0

    def evaluate(
        self, batch: list[PromptCase], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[JsonObject, JsonObject]:
        """Run every requested development case, preserving all failed-case outcomes."""
        from gepa.core.adapter import EvaluationBatch

        if set(candidate) != {self.study.component}:
            raise ValueError("GEPA changed an undeclared prompt component")
        if self._metric_calls + len(batch) > self.study.maximum_metric_calls:
            raise RuntimeError("GEPA evaluation batch exceeds the hard metric-call ceiling")
        outputs: list[JsonObject] = []
        traces: list[JsonObject] = []
        scores = []
        for case in batch:
            self._metric_calls += 1
            result = self._evaluate(candidate[self.study.component], case)
            receipt: JsonObject = {
                "case": case.model_dump(mode="json"),
                "candidate": candidate,
                "candidate_sha256": sha256_json(candidate),
                "evaluation": result.model_dump(mode="json"),
            }
            self._record(receipt)
            outputs.append(result.output)
            traces.append(receipt)
            scores.append(result.score)
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[JsonObject, JsonObject],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, JsonValue]]]:
        """Expose only actual development traces for the single declared prompt slot."""
        del candidate
        if components_to_update != [self.study.component] or eval_batch.trajectories is None:
            raise ValueError("reflection requires recorded traces for the declared component")
        return {self.study.component: eval_batch.trajectories}


def optimize_prompt(
    study: PromptStudy,
    *,
    baseline: str,
    train: Sequence[PromptCase],
    validation: Sequence[PromptCase],
    evaluate: Callable[[str, PromptCase], PromptEvaluation],
    reflect: Callable[[str], str],
    record: Callable[[JsonObject], None],
    run_directory: Path,
) -> str:
    """Compare a baseline with actual GEPA proposals on one frozen pipeline component.

    Args:
        study: Registered component, metric, budget and non-target pipeline identities.
        baseline: Initial prompt text, kept in GEPA's candidate frontier and receipts.
        train: Reflection examples from development only.
        validation: Disjoint development lineages for candidate selection, not final test data.
        evaluate: Real component or pipeline execution under the fixed metric.
        reflect: Admitted Experiential completion caller; no implicit GEPA provider client.
        record: Durable sink for registration, every evaluation and final selection.
        run_directory: New external directory for GEPA's full optimization history.

    Returns:
        Development-selected prompt; selection makes no held-out improvement claim.
    """
    from gepa import optimize

    if not baseline.strip() or not train or not validation:
        raise ValueError("baseline and nonempty explicit development splits are required")
    if {c.lineage_id for c in train} & {c.lineage_id for c in validation}:
        raise ValueError("GEPA training and selection lineages overlap")
    all_ids = [c.case_id for c in (*train, *validation)]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("prompt study repeats development case identities")
    if study.maximum_metric_calls < 15 * len(validation):
        raise ValueError("GEPA budget must permit meaningful search, at least 15x selection size")
    run_directory.mkdir(parents=True, exist_ok=False)
    record(
        {
            "study": study.model_dump(mode="json"),
            "baseline": baseline,
            "train": [c.model_dump(mode="json") for c in train],
            "validation": [c.model_dump(mode="json") for c in validation],
        }
    )

    def reflection(prompt: str | list[dict[str, JsonValue]]) -> str:
        """Route every GEPA reflection through the caller's shared admission and accounting."""
        return reflect(
            prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        )

    # GEPA checks its own ceiling between batches; reserve a complete proposal iteration.
    batch_size = min(3, len(train))
    batch_reserve = 2 * batch_size + len(validation)
    result = optimize(
        seed_candidate={study.component: baseline},
        trainset=list(train),
        valset=list(validation),
        adapter=_Adapter(study, evaluate, record),
        reflection_lm=reflection,
        max_metric_calls=study.maximum_metric_calls - batch_reserve,
        seed=study.seed,
        reflection_minibatch_size=batch_size,
        skip_perfect_score=False,
        run_dir=str(run_directory),
        display_progress_bar=False,
        raise_on_exception=True,
    )
    best = result.best_candidate
    if not isinstance(best, dict) or set(best) != {study.component}:
        raise ValueError("GEPA returned an unexpected component inventory")
    record(
        {
            "selected": best,
            "candidate_count": len(result.candidates),
            "selection_scores": list(result.val_aggregate_scores),
            "held_out_evaluation": "not performed by the optimizer",
        }
    )
    return best[study.component]
