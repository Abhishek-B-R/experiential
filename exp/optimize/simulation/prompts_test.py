"""Prompt experiment registration and split protection."""

from pathlib import Path

import pytest

from exp.optimize.simulation.prompts import (
    PromptCase,
    PromptEvaluation,
    PromptStudy,
    _Adapter,
    optimize_prompt,
)


def test_judge_and_simulator_prompt_optimization_are_separate() -> None:
    """A simulator study cannot optimize its own rating instrument."""
    with pytest.raises(ValueError, match="separate studies"):
        PromptStudy(
            study_id="study",
            component="fidelity_judge",
            frozen_pipeline_sha256="a" * 64,
            evaluator_sha256="b" * 64,
            maximum_metric_calls=30,
        )


def test_shared_lineages_are_rejected_before_evaluation(tmp_path: Path) -> None:
    """Held-out logs cannot be duplicate episodes under different example IDs."""
    study = PromptStudy(
        study_id="study",
        component="prediction",
        frozen_pipeline_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        maximum_metric_calls=30,
    )
    with pytest.raises(ValueError, match="lineages overlap"):
        optimize_prompt(
            study,
            baseline="Preserve state",
            train=[PromptCase(case_id="a", lineage_id="same", input={})],
            validation=[PromptCase(case_id="b", lineage_id="same", input={})],
            evaluate=lambda text, case: (_ for _ in ()).throw(AssertionError("must not evaluate")),
            reflect=lambda text: "unused",
            record=lambda value: None,
            run_directory=tmp_path / "gepa",
        )


def test_optimizer_cannot_dispatch_a_batch_beyond_the_registered_ceiling() -> None:
    """Library budgets remain hard even when upstream GEPA stops only between iterations."""
    study = PromptStudy(
        study_id="bounded",
        component="prediction",
        frozen_pipeline_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        maximum_metric_calls=2,
    )
    calls = []

    def unexpected(text: str, case: PromptCase) -> PromptEvaluation:
        """Record any execution that should have been refused before dispatch."""
        calls.append(case)
        raise AssertionError("over-budget evaluator was called")

    adapter = _Adapter(study, unexpected, lambda value: None)
    with pytest.raises(RuntimeError, match="hard metric-call ceiling"):
        adapter.evaluate(
            [PromptCase(case_id=str(i), lineage_id=str(i), input={}) for i in range(3)],
            {"prediction": "baseline"},
        )
    assert not calls
