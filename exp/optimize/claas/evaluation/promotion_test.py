"""Promotion remains conservative across arbitrary evaluator score kinds."""

import asyncio

from exp.optimize.claas.configuration import PromotionPolicy
from exp.optimize.claas.evaluation.paired import evaluate_policies
from exp.optimize.claas.evaluation.paired_test import evaluation_fixture
from exp.optimize.claas.evaluation.promotion import decide_promotion
from exp.optimize.claas.lifecycle.cycle_test import scenario
from exp.optimize.claas.lifecycle.rollouts import RevisionPolicy
from exp.runtime.claas.registry import RegistryState


def test_external_evaluator_scores_require_the_configured_evidence_floor() -> None:
    """Positive executable correctness delta cannot bypass minimum frozen task count."""
    evaluator, serving = evaluation_fixture()
    manifest = evaluator.freeze((scenario("held-out"),))
    current = serving.loaded
    candidate = current.model_copy(update={"policy_revision": "claas-candidate"})
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=RevisionPolicy(serving, current),
            candidate=RevisionPolicy(serving, candidate),
            evaluator=evaluator,
        )
    )
    baseline = RegistryState(scope=current.scope, generation=0, active=current)
    decision = decide_promotion(
        manifest=manifest,
        report=report,
        baseline=baseline,
        candidate=candidate,
        policy=PromotionPolicy(minimum_evaluation_tasks=2),
        evaluator=evaluator,
    )
    assert not decision.approved and decision.reason == "insufficient_tasks"
