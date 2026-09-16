"""Optional synthetic traffic evaluation preserves original provider replay validation."""

import asyncio

from exp.optimize.claas.evaluation.paired import evaluate_policies
from exp.optimize.workflows.traffic_learning.adapters import TrafficEvaluator
from exp.optimize.workflows.traffic_learning.evaluation_test import Policy, evaluation_fixture


def test_traffic_evaluator_plugs_into_generic_pairing() -> None:
    """Existing world-model evidence can score policies through the same generic interface."""
    manifest, world, judge = evaluation_fixture()
    evaluator = TrafficEvaluator(manifest, world, judge)
    report = asyncio.run(
        evaluate_policies(
            evaluator.freeze(),
            current=Policy("base"),
            candidate=Policy("candidate"),
            evaluator=evaluator,
        )
    )
    assert report.paired_mean_delta == 0.5 and report.score_kind == "synthetic_judge"
    evaluator.verify(evaluator.freeze(), report)
