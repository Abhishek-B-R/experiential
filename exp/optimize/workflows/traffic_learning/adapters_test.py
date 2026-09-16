"""Optional synthetic traffic evaluation preserves original provider replay validation."""

import asyncio
from typing import Literal

import pytest

from exp.common.models import AssistantAction
from exp.optimize.claas.evaluation.paired import evaluate_policies
from exp.optimize.workflows.traffic_learning.adapters import (
    TrafficEnvironment,
    TrafficEvaluator,
    TrafficSession,
    scenario_input,
)
from exp.optimize.workflows.traffic_learning.evaluation_test import Policy, evaluation_fixture
from exp.simulation.claas import ClaasWorldModel, SourceDisclosure, mine_experiences
from exp.simulation.claas.harness_test import (
    RecordingClient,
    final_transition,
    limits,
    model_snapshot,
)
from exp.simulation.claas.mining_test import make_experience


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


def practice_session() -> tuple[TrafficSession, RecordingClient]:
    """Open one real world-model harness without issuing a provider request."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
    )
    environment = TrafficEnvironment(world, (source,))
    return asyncio.run(environment.open(scenario_input(scenario))), client


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("step_limit", "limit"), ("failed", "error"), ("terminal", "caller_ended")],
)
def test_generic_close_reason_is_retained_in_world_evidence(
    reason: Literal["terminal", "step_limit", "failed"], expected: str
) -> None:
    """Policy failure and caller step limits remain visible in nested harness receipts."""
    session, client = practice_session()
    receipt = asyncio.run(session.close(reason))
    episode = receipt["world_episode"]
    assert isinstance(episode, dict)
    assert episode["end_reason"] == expected
    assert not client.requests


def test_world_terminal_reason_remains_immutable_after_generic_close() -> None:
    """Closing a completed harness does not relabel its already recorded world outcome."""
    session, client = practice_session()
    transition = asyncio.run(session.step(AssistantAction(content="Complete.")))
    assert transition.terminal
    receipt = asyncio.run(session.close("terminal"))
    episode = receipt["world_episode"]
    assert isinstance(episode, dict)
    assert episode["end_reason"] == "world_terminal"
    assert asyncio.run(session.close("failed")) == receipt
    assert len(client.requests) == 1
