"""Arbitrary authored-task evaluation retains all failures and verifies score evidence."""

import asyncio

import pytest

from exp.optimize.claas.evaluation.environment import EnvironmentEvaluator
from exp.optimize.claas.evaluation.paired import evaluate_policies, verify_evaluation_report
from exp.optimize.claas.lifecycle.cycle import base_revision
from exp.optimize.claas.lifecycle.cycle_test import (
    Admission,
    ExactAnswerScorer,
    LookupEnvironment,
    Serving,
    config,
    scenario,
)
from exp.optimize.claas.lifecycle.rollouts import RevisionPolicy


def evaluation_fixture() -> tuple[EnvironmentEvaluator, Serving]:
    """Create a deterministic executable environment without any model provider."""
    admission = Admission()
    admission.paused = True
    serving = Serving(admission, base_revision(config()))
    return EnvironmentEvaluator(LookupEnvironment(), ExactAnswerScorer()), serving


def test_paired_arbitrary_environment_resets_every_attempt_and_scores_both() -> None:
    """A manually authored task uses the same generic pairing/promotion evidence shape."""
    evaluator, serving = evaluation_fixture()
    manifest = evaluator.freeze((scenario("one"), scenario("two")))
    candidate = serving.loaded.model_copy(update={"policy_revision": "claas-candidate"})
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=RevisionPolicy(serving, serving.loaded),
            candidate=RevisionPolicy(serving, candidate),
            evaluator=evaluator,
        )
    )
    assert report.paired_mean_delta == 1.0
    assert report.score_kind == "executable_correctness"
    assert len(report.pairs) == 2
    assert isinstance(evaluator.environment, LookupEnvironment)
    assert evaluator.environment.opened == evaluator.environment.closed == 4


def test_report_rejects_forged_executable_score() -> None:
    """A syntactically valid result cannot override the injected verifier's actual rule."""
    evaluator, serving = evaluation_fixture()
    manifest = evaluator.freeze((scenario("one"),))
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=RevisionPolicy(serving, serving.loaded),
            candidate=RevisionPolicy(
                serving, serving.loaded.model_copy(update={"policy_revision": "claas-candidate"})
            ),
            evaluator=evaluator,
        )
    )
    data = report.model_dump(mode="json")
    data["pairs"][0]["candidate"]["score"] = 0.5
    with pytest.raises(ValueError, match="complete frozen episode"):
        verify_evaluation_report(type(report).model_validate(data), manifest, evaluator=evaluator)


def test_step_limit_keeps_every_failed_attempt_in_paired_denominator() -> None:
    """A partial tool workflow cannot vanish from the comparison or produce a mean score."""
    _, serving = evaluation_fixture()
    environment = LookupEnvironment()
    evaluator = EnvironmentEvaluator(environment, ExactAnswerScorer(), maximum_steps=1)
    manifest = evaluator.freeze((scenario("one"), scenario("two")))
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=RevisionPolicy(serving, serving.loaded),
            candidate=RevisionPolicy(
                serving, serving.loaded.model_copy(update={"policy_revision": "claas-candidate"})
            ),
            evaluator=evaluator,
        )
    )
    assert len(report.pairs) == 2 and report.paired_mean_delta is None
    assert all(
        outcome.failure_type == "StepLimitExceeded"
        for pair in report.pairs
        for outcome in (pair.current, pair.candidate)
    )
    assert environment.opened == environment.closed == 4
