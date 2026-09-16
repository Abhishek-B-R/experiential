"""Offline paired multi-turn evaluation with explicit failures and no feedback leakage."""

import asyncio
import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.claas.evaluation import EvaluationManifest, evaluate_policies, freeze_evaluation
from exp.simulation.claas.harness import ClaasWorldModel, SourceDisclosure
from exp.simulation.claas.harness_test import RecordingClient, limits, model_snapshot
from exp.simulation.claas.partition import split_experiences
from exp.simulation.claas.partition_test import source_batch
from exp.simulation.claas.provider import ClaasBoundedProvider
from exp.simulation.claas.replay import replay_episode


class Policy:
    """Deterministic two-turn tool policy with a fixed adapter identity."""

    def __init__(self, revision: str, *, fail: bool = False) -> None:
        """Bind a fixture outcome without loading a model or calling a provider."""
        self.policy_revision = revision
        self.fail = fail
        self.inputs: list[tuple[ModelMessage, ...]] = []

    async def act(
        self, *, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> AssistantAction:
        """Look up the claim, then report the synthetic observation."""
        self.inputs.append(messages)
        if self.fail:
            raise RuntimeError("private provider exception must not be persisted")
        assert tools[0].name == "lookup"
        if request_id.endswith(":0"):
            return AssistantAction(
                tool_calls=(
                    ToolCall(call_id="lookup-1", name="lookup", arguments={"claim_id": "c1"}),
                )
            )
        assert messages[-1].role == "tool"
        return AssistantAction(content=f"Claim pending. Answer from {self.policy_revision}.")


def world_reply(request: ModelRequest) -> JsonObject:
    """Simulate a tool result and a subsequent terminal response."""
    with pytest.raises(RuntimeError, match="no running event loop"):
        asyncio.get_running_loop()
    payload = json.loads(request.messages[-1].content or "{}")
    calls = payload["latest_action"]["tool_calls"]
    return {
        "observations": [
            {"call_id": call["call_id"], "content": "Claim pending.", "is_error": False}
            for call in calls
        ],
        "user_message": None,
        "terminal": not calls,
        "feedback": "PRIVATE TRAINING FEEDBACK",
        "reward": 1.0,
    }


def judge_reply(request: ModelRequest) -> JsonObject:
    """Score visible terminal responses while checking private feedback is excluded."""
    assert "PRIVATE TRAINING FEEDBACK" not in request.model_dump_json()
    with pytest.raises(RuntimeError, match="no running event loop"):
        asyncio.get_running_loop()
    payload = json.loads(request.messages[-1].content or "{}")
    content = payload["visible_trajectory"][-1]["assistant_action"]["content"]
    return {
        "score": 0.5 if "candidate" in content else 0.0,
        "feedback": "Consistent with simulated observations.",
    }


def evaluation_fixture() -> tuple[EvaluationManifest, ClaasWorldModel, ClaasBoundedProvider]:
    """Freeze two held-out tasks and finite mock world/judge clients."""
    split = split_experiences(source_batch(), seed="one", held_out_fraction=0.5)
    manifest = freeze_evaluation(split, world_model=model_snapshot(), judge_model=model_snapshot())
    disclosure = SourceDisclosure(scope=split.scope, model=model_snapshot())
    bounds = limits(maximum_steps=3, maximum_model_calls=32, maximum_total_cost_usd=1.0)
    world = ClaasWorldModel(
        client=RecordingClient(world_reply),
        model=model_snapshot(),
        limits=bounds,
        source_disclosure=disclosure,
        purpose="evaluation",
    )
    judge = ClaasBoundedProvider(
        client=RecordingClient(judge_reply),
        model=model_snapshot(),
        limits=bounds,
        source_disclosure=disclosure,
    )
    return manifest, world, judge


def test_paired_evaluation_runs_both_policies_on_identical_multiturn_tasks() -> None:
    """Both revisions receive fresh tool sessions and one identical frozen judge per task."""
    manifest, world, judge = evaluation_fixture()
    current, candidate = Policy("current"), Policy("candidate")
    report = asyncio.run(
        evaluate_policies(manifest, current=current, candidate=candidate, world=world, judge=judge)
    )
    assert len(report.pairs) == 2
    assert report.paired_mean_delta == 0.5
    assert world.reserved_calls == 8 and judge.reserved_calls == 4
    assert [pair.task_id for pair in report.pairs] == [task.task_id for task in manifest.tasks]
    for pair in report.pairs:
        assert pair.current.episode is not None and len(pair.current.episode.steps) == 2
        assert pair.candidate.episode is not None and len(pair.candidate.episode.steps) == 2
        assert pair.current.score_kind == pair.candidate.score_kind == "synthetic_judge"
    episode = report.pairs[0].current.episode
    assert episode is not None
    assert (
        replay_episode(episode, grounding=manifest.tasks[0].grounding, limits=world.limits)
        == episode
    )
    assert all(
        "PRIVATE TRAINING FEEDBACK" not in str(messages)
        for messages in current.inputs + candidate.inputs
    )


def test_policy_failures_stay_in_pairs_and_suppress_aggregate() -> None:
    """A failed policy attempt never disappears or receives a fabricated success score."""
    manifest, world, judge = evaluation_fixture()
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=Policy("current"),
            candidate=Policy("candidate", fail=True),
            world=world,
            judge=judge,
        )
    )
    assert len(report.pairs) == len(manifest.tasks)
    assert all(pair.candidate.failure_stage == "policy" for pair in report.pairs)
    assert all(pair.current.judgment is not None for pair in report.pairs)
    assert report.paired_mean_delta is None
    assert "private provider exception" not in report.model_dump_json()


def test_evaluation_world_rejects_fit_and_manifest_rejects_judge_drift() -> None:
    """Held-out execution authority cannot accidentally become training authority."""
    manifest, world, _ = evaluation_fixture()
    task = manifest.tasks[0]
    with pytest.raises(ValueError, match="held_out"):
        world.reset(task.scenario.model_copy(update={"partition": "fit"}), grounding=task.grounding)
    changed = manifest.model_dump(mode="json")
    changed["rubric"] = "Changed after seeing candidate results."
    with pytest.raises(ValueError, match="judge protocol"):
        EvaluationManifest.model_validate(changed)


def test_evaluation_missing_judge_disclosure_fails_before_policy_execution() -> None:
    """A missing judgment authorization cannot consume policy or world compute first."""
    manifest, world, judge = evaluation_fixture()
    judge.source_disclosure = None
    current, candidate = Policy("current"), Policy("candidate")
    with pytest.raises(ValueError, match="disclosure"):
        asyncio.run(
            evaluate_policies(
                manifest, current=current, candidate=candidate, world=world, judge=judge
            )
        )
    assert not current.inputs and not candidate.inputs
    assert world.reserved_calls == judge.reserved_calls == 0


def test_cancellation_joins_an_already_dispatched_provider_call() -> None:
    """Evaluation cannot close while its owned synchronous provider work is still running."""
    from threading import Event

    from exp.optimize.claas.evaluation import _run_blocking

    started, release, finished = Event(), Event(), Event()

    def operation() -> str:
        """Hold an in-flight provider operation until the test releases it."""
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        return "done"

    async def cancel() -> None:
        """Cancel the caller and prove its dispatch remains owned until completion."""
        task = asyncio.create_task(_run_blocking(operation))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(cancel())


def test_manifest_rejects_overlapping_source_groups() -> None:
    """Duplicate membership cannot be hidden by a dictionary's last-value behavior."""
    manifest, _, _ = evaluation_fixture()
    changed = manifest.model_dump(mode="json")
    original = manifest.held_out_groups[0]
    forged = original.model_copy(update={"group_id": "f" * 64})
    changed["held_out_groups"] = [forged.model_dump(mode="json"), *changed["held_out_groups"]]
    with pytest.raises(ValueError, match="disjoint experience"):
        EvaluationManifest.model_validate(changed)


def test_incomplete_saved_report_cannot_publish_a_subset_score() -> None:
    """Removing a completed pair cannot turn incomplete prescribed coverage into a score."""
    from exp.optimize.claas.evaluation import PairedEvaluationReport, verify_evaluation_report

    manifest, world, judge = evaluation_fixture()
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=Policy("current"),
            candidate=Policy("candidate"),
            world=world,
            judge=judge,
        )
    )
    verify_evaluation_report(report, manifest)
    changed = report.model_dump(mode="json")
    changed["pairs"] = changed["pairs"][:-1]
    with pytest.raises(ValueError, match="every frozen task"):
        PairedEvaluationReport.model_validate(changed)
    changed["expected_task_ids"] = changed["expected_task_ids"][:-1]
    partial = PairedEvaluationReport.model_validate(changed)
    with pytest.raises(ValueError, match="authoritative"):
        verify_evaluation_report(partial, manifest)


def test_multiturn_evaluation_uses_real_http_client_off_the_policy_loop() -> None:
    """Exercise provider serialization and sync/async transport adaptation without network I/O."""
    from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
    from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

    snapshot = model_snapshot().model_copy(update={"provider": "openai-compatible"})
    split = split_experiences(source_batch(), seed="s", held_out_fraction=0.5)
    manifest = freeze_evaluation(split, world_model=snapshot, judge_model=snapshot)
    assert len(manifest.tasks) == 1

    def response(payload: JsonObject) -> JsonHttpResponse:
        """Encode a structured world/judge response in the actual provider wire shape."""
        return JsonHttpResponse(
            status_code=200,
            body={
                "model": snapshot.model_id,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 16, "total_tokens": 24},
            },
        )

    observation: JsonObject = {
        "observations": [{"call_id": "lookup-1", "content": "Claim pending.", "is_error": False}],
        "user_message": None,
        "terminal": False,
        "feedback": "PRIVATE FEEDBACK",
        "reward": 0.5,
    }
    terminal: JsonObject = {
        "observations": [],
        "user_message": None,
        "terminal": True,
        "feedback": "PRIVATE FEEDBACK",
        "reward": 0.5,
    }
    world_transport = ScriptedJsonTransport([response(observation), response(terminal)] * 2)
    judge_transport = ScriptedJsonTransport(
        [
            response({"score": 0.0, "feedback": "Current policy score."}),
            response({"score": 0.5, "feedback": "Candidate policy score."}),
        ]
    )
    disclosure = SourceDisclosure(scope=split.scope, model=snapshot)
    bounds = limits(maximum_steps=3, maximum_model_calls=8, maximum_total_cost_usd=1.0)
    world = ClaasWorldModel(
        client=OpenAICompatibleClient(
            model=snapshot,
            api_key="fixture",
            base_url="https://fixture.invalid/v1",
            transport=world_transport,
        ),
        model=snapshot,
        limits=bounds,
        source_disclosure=disclosure,
        purpose="evaluation",
    )
    judge = ClaasBoundedProvider(
        client=OpenAICompatibleClient(
            model=snapshot,
            api_key="fixture",
            base_url="https://fixture.invalid/v1",
            transport=judge_transport,
        ),
        model=snapshot,
        limits=bounds,
        source_disclosure=disclosure,
    )
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=Policy("current"),
            candidate=Policy("candidate"),
            world=world,
            judge=judge,
        )
    )
    assert report.paired_mean_delta == 0.5
    assert len(world_transport.requests) == 4 and len(judge_transport.requests) == 2
    assert all(request.url.endswith("/chat/completions") for request in world_transport.requests)


def test_loaded_report_rejects_score_tampering_and_wrong_judge_identity() -> None:
    """A score must derive from retained provider evidence for the frozen judge model."""
    from exp.optimize.claas.evaluation import PairedEvaluationReport, verify_evaluation_report

    manifest, world, judge = evaluation_fixture()
    report = asyncio.run(
        evaluate_policies(
            manifest,
            current=Policy("current"),
            candidate=Policy("candidate"),
            world=world,
            judge=judge,
        )
    )
    raw = json.loads(report.model_dump_json())
    raw["pairs"][0]["candidate"]["judgment"]["score"] = 1.0
    with pytest.raises(ValueError, match="recorded judge response"):
        PairedEvaluationReport.model_validate(raw)
    raw = json.loads(report.model_dump_json())
    raw["pairs"][0]["candidate"]["judge_response"]["model"]["model_id"] = "other-judge"
    forged = PairedEvaluationReport.model_validate(raw)
    with pytest.raises(ValueError, match="different model snapshot"):
        verify_evaluation_report(forged, manifest)
