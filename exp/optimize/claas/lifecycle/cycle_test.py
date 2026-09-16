"""Generic cycle orchestration with an authored tool environment and receipt-only backend.

These fixtures prove lifecycle and evidence composition, not a veRL optimizer or
learned quality improvement. Actual upstream execution has separate backend tests.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest

from exp.common.claas import ClaasScope, ExactTokenEvidence
from exp.common.claas.scenarios import EnvironmentEpisode, EnvironmentTransition, Scenario
from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.core.locks import FileLockTimeout, file_write_lock
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.claas.backends.checkpoints import CheckpointManifest, hash_file
from exp.optimize.claas.configuration import CycleLimits, LocalClaasConfig, PromotionPolicy
from exp.optimize.claas.evaluation.environment import EnvironmentEvaluator, EpisodeScore
from exp.optimize.claas.lifecycle import cycle
from exp.optimize.claas.lifecycle.cycle import CycleState, base_revision, rollback, run_cycle
from exp.optimize.claas.lifecycle.inputs import PreparedCycle
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingJob,
    TrainingResult,
    next_policy_revision,
    validate_training_batch,
)
from exp.runtime.claas.registry import AdapterRegistry, RegistryState, ServingRevision
from exp.runtime.claas.serving.contracts import PolicySample


class Admission:
    """Record admission ordering independently of the generic core under test."""

    def __init__(self) -> None:
        """Start public admission open."""
        self.paused = False
        self.published: list[tuple[int, str]] = []
        self.closed = False

    async def pause_and_drain(self, *, timeout_seconds: float = 120) -> None:
        """Complete the fixture's public drain."""
        self.paused = True

    async def resume(
        self, *, expected_registry_generation: int, expected_policy_revision: str
    ) -> None:
        """Record exact generation and revision readiness."""
        self.published.append((expected_registry_generation, expected_policy_revision))
        self.paused = False

    async def close(self) -> None:
        """Release fixture ownership without reopening admission."""
        self.closed = True


class Serving:
    """Deterministic policy fixture with explicit lifecycle and original-token receipts."""

    def __init__(self, admission: Admission, base: ServingRevision) -> None:
        """Bind baseline identity and start awake."""
        self.admission, self.loaded = admission, base
        self.asleep = False
        self.sampled: list[str] = []
        self.response_limits: list[int | None] = []
        self.improve = True

    async def pause_and_drain(self) -> None:
        """Require public traffic drain first."""
        assert self.admission.paused

    async def wake(self) -> None:
        """Restore inference memory."""
        self.asleep = False

    async def sleep(self) -> None:
        """Release inference before the backend opens."""
        assert self.admission.paused
        self.asleep = True

    async def load_revision(self, revision: ServingRevision) -> None:
        """Load only while public traffic is paused."""
        assert self.admission.paused and not self.asleep
        self.loaded = revision

    async def resume(self) -> None:
        """Prove memory restoration precedes public readiness."""
        assert self.admission.paused and not self.asleep

    async def tokenize_training_text(self, text: str) -> tuple[int, ...]:
        """Provide deterministic teacher-only fixture token lengths."""
        return (7, 8)

    async def sample_for_evaluation(
        self,
        messages: tuple[ModelMessage, ...],
        tools: tuple[ToolSchema, ...],
        request_id: str,
        *,
        max_tokens: int | None = None,
    ) -> PolicySample:
        """Issue lookup then a revision-controlled answer without private feedback access."""
        assert self.admission.paused and not self.asleep
        assert all("private feedback" not in (message.content or "") for message in messages)
        self.sampled.append(self.loaded.policy_revision)
        self.response_limits.append(max_tokens)
        action = (
            AssistantAction(
                content="4"
                if self.improve and self.loaded.policy_revision.startswith("claas-")
                else "0"
            )
            if messages[-1].role == "tool"
            else AssistantAction(
                tool_calls=(ToolCall(call_id="lookup-1", name="lookup", arguments={}),)
            )
        )
        return PolicySample(
            action=action,
            exact_tokens=ExactTokenEvidence(
                model_id=self.loaded.model_id,
                model_revision=self.loaded.model_revision,
                tokenizer_id=self.loaded.tokenizer_id,
                tokenizer_revision=self.loaded.tokenizer_revision,
                policy_revision=self.loaded.policy_revision,
                prompt_token_ids=(1, 2),
                response_token_ids=(3, 4),
                response_logprobs=(-1.0, -1.0),
                sampling_temperature=1.0,
                sampling_top_p=1.0,
                sampling_top_k=None,
            ),
            request={"messages": [message.model_dump(mode="json") for message in messages]},
            response={"id": request_id, "content": action.content},
            raw_completion={"id": request_id},
        )


class LookupEnvironment:
    """Manually authored deterministic tool workflow, independent of traffic and providers."""

    environment_id = "authored-lookup-v1"

    def __init__(self) -> None:
        """Record reset/close counts across every independent episode."""
        self.opened = 0
        self.closed = 0

    async def open(self, scenario: Scenario) -> LookupSession:
        """Reset private task state for a fresh attempt."""
        self.opened += 1
        return LookupSession(self, scenario)


class LookupSession:
    """An execute-only session that exposes tool results, never its private answer key."""

    def __init__(self, owner: LookupEnvironment, scenario: Scenario) -> None:
        """Bind one scenario's private expected answer."""
        self.owner, self.scenario = owner, scenario
        self.messages = scenario.messages

    async def step(self, action: AssistantAction) -> EnvironmentTransition:
        """Execute lookup or finish with independently computed scalar/text feedback."""
        self.messages += (ModelMessage(role="assistant", assistant_action=action),)
        if action.tool_calls:
            call = action.tool_calls[0]
            self.messages += (
                ModelMessage(role="tool", tool_call_id=call.call_id, content="Observed value: 4"),
            )
        correct = action.content == self.scenario.environment_data["answer"]
        return EnvironmentTransition(
            messages=self.messages,
            terminal=not action.tool_calls,
            reward=1.0 if correct else 0.0,
            feedback="private feedback: report the observed value",
        )

    async def close(self, reason: Literal["terminal", "step_limit", "failed"]) -> JsonObject:
        """Record cleanup independently of cycle success."""
        self.owner.closed += 1
        return {"closed": True, "reason": reason}


class ExactAnswerScorer:
    """Executable task correctness scorer with replayable evidence."""

    scorer_id = "exact-answer-v1"
    score_kind = "executable_correctness"

    async def score(self, episode: EnvironmentEpisode) -> EpisodeScore:
        """Compare the final action with the private authored answer."""
        score = float(
            episode.steps[-1].action.content == episode.scenario.environment_data["answer"]
        )
        return EpisodeScore(score=score, evidence={"rule": self.scorer_id})

    def verify(self, episode: EnvironmentEpisode, score: EpisodeScore) -> None:
        """Recompute correctness without trusting an environment reward label."""
        expected = float(
            episode.steps[-1].action.content == episode.scenario.environment_data["answer"]
        )
        if score.score != expected or score.evidence != {"rule": self.scorer_id}:
            raise ValueError("score differs from deterministic answer verification")


def config() -> LocalClaasConfig:
    """Create a provider-independent application with a tiny deterministic task bound."""
    return LocalClaasConfig(
        scope=ClaasScope(user_id="user", application_id="claims"),
        base_model="fixture/llama",
        base_model_revision="a" * 40,
        tokenizer_id="fixture/llama",
        tokenizer_revision="a" * 40,
        lora_rank=2,
        limits=CycleLimits(
            maximum_scenarios=2,
            maximum_rollouts_per_scenario=1,
            maximum_episode_steps=2,
            maximum_cost_usd=5.0,
        ),
        promotion=PromotionPolicy(minimum_evaluation_tasks=1),
    )


def scenario(identity: str) -> Scenario:
    """Author an arbitrary tool task with no source trace or world-model requirement."""
    return Scenario(
        scenario_id=identity,
        scope=config().scope,
        environment_id=LookupEnvironment.environment_id,
        messages=(ModelMessage(role="user", content="Look up and report the value."),),
        tools=(
            ToolSchema(
                name="lookup",
                description="Read a value",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
        environment_data={"answer": "4"},
    )


class ReceiptBackend:
    """Produce explicit fixture artifacts solely to test the generic transaction boundary."""

    def __init__(self, directory: Path, lineage: str, serving: Serving) -> None:
        """Bind durable artifacts and the lifecycle assertion fixture."""
        self.directory, self.lineage, self.serving = directory, lineage, serving
        self.batches: list[TrainingBatch] = []
        self.closed = False
        self.fail = False

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
    ) -> ReceiptSession:
        """Require sleeping inference before fixture execution begins."""
        assert self.serving.asleep
        return ReceiptSession(self, spec, resume)


class ReceiptSession:
    """A backend protocol fixture, deliberately without optimizer execution."""

    def __init__(
        self, backend: ReceiptBackend, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None
    ) -> None:
        """Bind exact input and optional prior checkpoint identities."""
        self.backend, self.spec, self.previous = backend, spec, resume
        self.result: TrainingResult | None = None

    @property
    def policy_revision(self) -> str:
        """Return the exact input revision required for this fixture update."""
        return self.previous.policy_revision if self.previous else self.spec.initial_policy_revision

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Validate exact samples and write digest-bound marker artifacts for orchestration."""
        if self.backend.fail:
            raise RuntimeError("deliberate backend failure")
        validate_training_batch(self.spec, batch, self.previous)
        self.backend.batches.append(batch)
        job = TrainingJob(
            spec=self.spec,
            batch=batch,
            checkpoint_root=str(self.backend.directory),
            resume_checkpoint=self.previous,
            lineage_id=self.backend.lineage,
        )
        revision = next_policy_revision(job)
        root = self.backend.directory / revision
        files = {}
        for name in (
            "student/adapter_config.json",
            "student/adapter_model.safetensors",
            "teacher/adapter_config.json",
            "teacher/adapter_model.safetensors",
            "verl/actor/model_world_size_1_rank_0.pt",
            "verl/actor/optim_world_size_1_rank_0.pt",
            "verl/actor/extra_state_world_size_1_rank_0.pt",
            "verl/actor/fsdp_config.json",
            "verl/teacher/model_world_size_1_rank_0.pt",
            "verl/teacher/fsdp_config.json",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ORCHESTRATION TEST RECEIPT; NOT MODEL WEIGHTS")
            files[name] = hash_file(path)
        history = (
            revision,
            *(
                self.previous.policy_history
                if self.previous
                else (self.spec.initial_policy_revision,)
            ),
        )[: self.spec.max_policy_lag + 1]
        step = (self.previous.step if self.previous else 0) + 1
        ids = tuple(item.experience.experience_id for item in batch.examples)
        manifest = CheckpointManifest(
            schema_version=2,
            training_backend="verl-fsdp-0.9.0",
            spec=self.spec,
            policy_revision=revision,
            parent_policy_revision=batch.expected_policy_revision,
            policy_history=history,
            step=step,
            batch_id=batch.batch_id,
            consumed_experience_ids=ids,
            files=files,
            lineage_id=self.backend.lineage,
        )
        (root / "manifest.json").write_text(manifest.model_dump_json())
        checkpoint = TrainingCheckpoint(
            scope=self.spec.scope,
            adapter_id=self.spec.adapter_id,
            policy_revision=revision,
            policy_history=history,
            path=str(root),
            manifest_sha256=sha256_json(manifest),
            step=step,
        )
        self.result = TrainingResult(
            checkpoint=checkpoint, consumed_experience_ids=ids, metrics={"fixture_only": 1.0}
        )
        return self.result

    async def checkpoint(self) -> TrainingCheckpoint:
        """Return only a completed fixture receipt."""
        assert self.result is not None
        return self.result.checkpoint

    async def close(self) -> None:
        """Record lifecycle cleanup for success and failure paths."""
        self.backend.closed = True


async def drive_cycle(
    directory: Path,
    *,
    fail: bool = False,
    improve: bool = True,
    maximum_response_tokens: int = 2048,
) -> tuple[CycleState, Serving, Admission, LookupEnvironment, list[ReceiptBackend]]:
    """Drive the actual generic cycle using explicitly injected deterministic adapters."""
    settings = config()
    settings = settings.model_copy(
        update={
            "limits": settings.limits.model_copy(
                update={"maximum_response_tokens": maximum_response_tokens}
            )
        }
    )
    admission = Admission()
    serving = Serving(admission, base_revision(settings))
    serving.improve = improve
    environment = LookupEnvironment()
    evaluator = EnvironmentEvaluator(environment, ExactAnswerScorer(), maximum_steps=2)
    backends: list[ReceiptBackend] = []

    def factory(lineage: str) -> ReceiptBackend:
        """Track each independently closed fixture backend."""
        backend = ReceiptBackend(directory / "checkpoints", lineage, serving)
        backend.fail = fail
        backends.append(backend)
        return backend

    result = await run_cycle(
        directory=directory,
        config=settings,
        plan=PreparedCycle(
            scenarios=(scenario("fit"),),
            environment=environment,
            evaluation=evaluator.freeze((scenario("held-out"),)),
            evaluator=evaluator,
        ),
        serving=serving,
        admission=admission,
        backend_factory=factory,
    )
    return result, serving, admission, environment, backends


def test_authored_environment_drives_generic_cycle_and_rollback(tmp_path: Path) -> None:
    """Real core composes custom environment, evaluator, backend receipts and activation."""
    state, serving, admission, environment, backends = asyncio.run(drive_cycle(tmp_path))
    assert state.stage == "complete" and state.decision and state.decision.approved
    assert state.decision.score_kind == "executable_correctness"
    assert environment.opened == environment.closed == 3
    assert backends[0].closed and len(backends[0].batches[0].examples) == 2
    assert all(
        item.experience.provenance.source_kind == "environment"
        and not item.experience.provenance.source_experience_ids
        for item in backends[0].batches[0].examples
    )
    assert admission.closed and not admission.paused
    assert serving.loaded.policy_revision == state.candidate_revision
    restored = asyncio.run(
        rollback(directory=tmp_path, config=config(), serving=serving, admission=admission)
    )
    assert restored.active == base_revision(config())


def test_configured_response_limit_reaches_practice_and_both_evaluation_policies(
    tmp_path: Path,
) -> None:
    """A non-default cap is attached to every sampled action through the actual cycle."""
    _, serving, _, _, _ = asyncio.run(drive_cycle(tmp_path, maximum_response_tokens=256))
    assert len(serving.response_limits) == 6
    assert serving.response_limits == [256] * 6
    assert len(set(serving.sampled)) == 2


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit", "publish"])
def test_failed_rollback_recovers_the_durable_active_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    """Failed and uncertain pointer writes recover registry truth before reopening traffic."""
    _, serving, admission, _, _ = asyncio.run(drive_cycle(tmp_path))
    registry = AdapterRegistry(tmp_path / "registry.json", config().scope)
    before = registry.read()
    original_rollback = AdapterRegistry.rollback
    original_resume = admission.resume
    failed = False
    admission.closed = False

    def rollback_pointer(store: AdapterRegistry, *, expected_generation: int) -> RegistryState:
        """Fail before or after the real durable registry transaction."""
        assert serving.loaded == before.previous
        if failure_point == "before_commit":
            raise OSError("fixture rollback write failed")
        result = original_rollback(store, expected_generation=expected_generation)
        if failure_point == "after_commit":
            raise OSError("fixture rollback acknowledgment lost")
        return result

    async def publish(*, expected_registry_generation: int, expected_policy_revision: str) -> None:
        """Reject the first rollback publication, then permit verified recovery."""
        nonlocal failed
        if failure_point == "publish" and not failed:
            failed = True
            raise OSError("fixture admission publication failed")
        await original_resume(
            expected_registry_generation=expected_registry_generation,
            expected_policy_revision=expected_policy_revision,
        )

    monkeypatch.setattr(AdapterRegistry, "rollback", rollback_pointer)
    monkeypatch.setattr(admission, "resume", publish)
    with pytest.raises(OSError, match="fixture"):
        asyncio.run(
            rollback(directory=tmp_path, config=config(), serving=serving, admission=admission)
        )
    durable = registry.read()
    expected = before.active if failure_point == "before_commit" else before.previous
    assert serving.loaded == durable.active == expected
    assert admission.published[-1] == (durable.generation, durable.active.policy_revision)
    assert not admission.paused and admission.closed


@pytest.mark.parametrize("failure", ["timeout", "cancellation"])
@pytest.mark.parametrize("close_failure", [False, True])
def test_rollback_initial_drain_failure_never_retries_or_changes_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, close_failure: bool
) -> None:
    """Propagate the first drain timeout or cancellation without entering private recovery."""
    _, serving, admission, _, _ = asyncio.run(drive_cycle(tmp_path))
    registry = AdapterRegistry(tmp_path / "registry.json", config().scope)
    before = registry.read()
    publications = tuple(admission.published)
    admission.closed = False
    attempts = 0
    failures: list[BaseException] = []

    async def run() -> None:
        """Fail an in-flight public drain before private serving ownership is acquired."""
        started = asyncio.Event()

        async def drain() -> None:
            """Expire a real deadline or accept caller cancellation on the first drain."""
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise AssertionError("initial drain failure must not be retried")
            admission.paused = True
            started.set()
            try:
                async with asyncio.timeout(0.01 if failure == "timeout" else 1):
                    await asyncio.Event().wait()
            except BaseException as error:
                failures.append(error)
                raise

        async def forbidden_private_drain() -> None:
            """Reject any private serving transition after an uncompleted public drain."""
            raise AssertionError("private serving must remain untouched")

        async def failed_close() -> None:
            """Expose an admission release error without hiding the original drain failure."""
            admission.closed = True
            raise OSError("fixture admission close failed")

        monkeypatch.setattr(admission, "pause_and_drain", drain)
        monkeypatch.setattr(serving, "pause_and_drain", forbidden_private_drain)
        if close_failure:
            monkeypatch.setattr(admission, "close", failed_close)
        task = asyncio.create_task(
            rollback(directory=tmp_path, config=config(), serving=serving, admission=admission)
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        if failure == "cancellation":
            task.cancel("caller cancelled rollback")
        expected = TimeoutError if failure == "timeout" else asyncio.CancelledError
        with pytest.raises(expected) as raised:
            await asyncio.wait_for(task, timeout=1)
        assert raised.value is failures[0]

    asyncio.run(run())
    assert attempts == 1
    assert registry.read() == before
    assert serving.loaded == before.active
    assert tuple(admission.published) == publications
    assert admission.closed and admission.paused


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
def test_failed_rollback_recovery_keeps_admission_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """A server that cannot reload registry truth never publishes readiness."""
    _, serving, admission, _, _ = asyncio.run(drive_cycle(tmp_path))
    registry = AdapterRegistry(tmp_path / "registry.json", config().scope)
    before = registry.read()
    publications = tuple(admission.published)
    attempted: list[ServingRevision] = []
    admission.closed = False
    original_error = failure("fixture serving unavailable")

    async def unavailable(revision: ServingRevision) -> None:
        """Fail both the requested previous revision and the recovery load."""
        attempted.append(revision)
        if len(attempted) == 1:
            raise original_error
        raise OSError("fixture serving unavailable")

    monkeypatch.setattr(serving, "load_revision", unavailable)
    with pytest.raises(failure, match="serving unavailable") as raised:
        asyncio.run(
            rollback(directory=tmp_path, config=config(), serving=serving, admission=admission)
        )
    if failure is asyncio.CancelledError:
        assert raised.value is original_error
    assert attempted == [before.previous, before.active]
    assert registry.read() == before
    assert tuple(admission.published) == publications
    assert admission.paused and admission.closed


@pytest.mark.parametrize("operation", ["cycle", "rollback"])
def test_admission_close_does_not_rethrow_the_callers_handled_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A successful lifecycle cannot inherit unrelated exception state from its caller."""
    _, serving, admission, _, _ = asyncio.run(drive_cycle(tmp_path))

    async def failed_close(self: Admission) -> None:
        """Expose the lifecycle's own release error after a successful operation."""
        self.closed = True
        raise OSError("fixture admission release failed")

    async def handled_caller() -> None:
        """Call the lifecycle while an unrelated exception is already being handled."""
        try:
            raise ValueError("unrelated caller failure")
        except ValueError:
            if operation == "cycle":
                await drive_cycle(tmp_path)
            else:
                await rollback(
                    directory=tmp_path, config=config(), serving=serving, admission=admission
                )

    monkeypatch.setattr(Admission, "close", failed_close)
    with pytest.raises(OSError, match="admission release failed"):
        asyncio.run(handled_caller())


@pytest.mark.parametrize("phase", ["recovery", "admission_close"])
def test_new_cancellation_overrides_prior_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """A caller cancellation during recovery or release overrides an earlier ordinary failure."""
    _, serving, admission, _, _ = asyncio.run(drive_cycle(tmp_path))
    registry = AdapterRegistry(tmp_path / "registry.json", config().scope)
    before = registry.read()
    original_load = serving.load_revision
    original_wake = serving.wake
    attempted = False
    admission.closed = False

    async def run() -> None:
        """Cancel only after the rollback has failed and entered its cleanup phase."""
        entered = asyncio.Event()

        async def fail_once(revision: ServingRevision) -> None:
            """Fail the attempted rollback before allowing durable-revision recovery."""
            nonlocal attempted
            if not attempted:
                attempted = True
                raise ValueError("fixture rollback load failed")
            await original_load(revision)

        async def recovery_wake() -> None:
            """Expose a pending recovery after the original ordinary rollback failure."""
            if attempted and phase == "recovery":
                entered.set()
                await asyncio.Future()
            await original_wake()

        async def close() -> None:
            """Allow new release cancellation or test preservation over a later close error."""
            admission.closed = True
            if phase == "admission_close":
                entered.set()
                await asyncio.Future()
            raise OSError("fixture admission close failed")

        monkeypatch.setattr(serving, "load_revision", fail_once)
        monkeypatch.setattr(serving, "wake", recovery_wake)
        monkeypatch.setattr(admission, "close", close)
        task = asyncio.create_task(
            rollback(directory=tmp_path, config=config(), serving=serving, admission=admission)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel("new caller cancellation")
        with pytest.raises(asyncio.CancelledError, match="new caller cancellation"):
            await task

    asyncio.run(run())
    assert registry.read() == before
    assert admission.closed
    assert admission.paused == (phase == "recovery")


def test_rejection_resumes_active_revision_and_preserves_checkpoint(tmp_path: Path) -> None:
    """Rejected candidates remain available without changing the active adapter."""
    state, serving, admission, _, backends = asyncio.run(drive_cycle(tmp_path, improve=False))
    assert state.decision and not state.decision.approved
    assert serving.loaded == base_revision(config()) and not admission.paused
    assert backends[0].closed
    assert len(list((tmp_path / "checkpoint-receipts").glob("*.json"))) == 1
    second, *_ = asyncio.run(drive_cycle(tmp_path, improve=False))
    assert second.stage == "complete"


def test_backend_failure_restores_serving_and_persists_journal(tmp_path: Path) -> None:
    """A failed update cannot publish a candidate or leave an unrecorded terminal state."""
    with pytest.raises(RuntimeError, match="deliberate backend failure"):
        asyncio.run(drive_cycle(tmp_path, fail=True))
    state = CycleState.model_validate_json(
        next((tmp_path / "cycles").glob("*/state.json")).read_bytes()
    )
    assert (
        state.stage == "failed" and state.serving_restored and state.failure_type == "RuntimeError"
    )
    assert AdapterRegistry(
        tmp_path / "registry.json", config().scope
    ).read().active == base_revision(config())


def test_private_scenario_setup_never_becomes_student_feedback(tmp_path: Path) -> None:
    """Private answer key stays in environment inputs; only actual tool observations are visible."""
    state, *_ = asyncio.run(drive_cycle(tmp_path))
    receipt = json.loads(
        next((tmp_path / "cycles" / state.cycle_id / "practice").glob("*.json")).read_bytes()
    )
    assert receipt["episode"]["scenario"]["environment_data"] == {"answer": "4"}
    for sample in receipt["samples"]:
        serialized = json.dumps(sample["request"])
        assert "answer" not in serialized and "private feedback" not in serialized


def test_source_preparation_and_training_share_the_application_lock(tmp_path: Path) -> None:
    """An evaluation writer cannot change exclusion state between preparation and training."""

    settings = config()
    admission = Admission()
    serving = Serving(admission, base_revision(settings))
    environment = LookupEnvironment()
    evaluator = EnvironmentEvaluator(environment, ExactAnswerScorer())
    prepared = PreparedCycle(
        scenarios=(scenario("fit"),),
        environment=environment,
        evaluation=evaluator.freeze((scenario("held-out"),)),
        evaluator=evaluator,
    )
    observed: list[str] = []

    class LockedSource:
        """Check the same authoritative lock held by evaluation holdout writes."""

        external_reservation_usd = 0.0

        async def prepare(self, directory: Path, config: LocalClaasConfig) -> PreparedCycle:
            """Prove preparation has exclusive ownership without reacquiring it."""
            with (
                pytest.raises(FileLockTimeout),
                file_write_lock(directory / "cycle", what="evaluation", timeout_s=0),
            ):
                pytest.fail("evaluation exclusion mutation raced scenario preparation")
            observed.append("prepare")
            return prepared

    class LockedBackend(ReceiptBackend):
        """Check source ownership remains unchanged when compute opens."""

        async def open(
            self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
        ) -> ReceiptSession:
            """Prove the original cycle lock still excludes evaluation writers."""
            with (
                pytest.raises(FileLockTimeout),
                file_write_lock(tmp_path / "cycle", what="evaluation", timeout_s=0),
            ):
                pytest.fail("evaluation exclusion mutation raced training")
            observed.append("train")
            return await super().open(spec, resume)

    asyncio.run(
        run_cycle(
            directory=tmp_path,
            config=settings,
            plan=LockedSource(),
            serving=serving,
            admission=admission,
            backend_factory=lambda lineage: LockedBackend(
                tmp_path / "checkpoints", lineage, serving
            ),
        )
    )
    assert observed == ["prepare", "train"]
    with file_write_lock(tmp_path / "cycle", what="evaluation", timeout_s=0):
        assert not admission.paused


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
def test_initial_drain_failure_is_recorded_without_touching_private_inference(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    """A failed public drain leaves serving ownership untouched and records the failure."""

    class FailedAdmission(Admission):
        """Fail at the first lifecycle ownership boundary."""

        async def pause_and_drain(self, *, timeout_seconds: float = 120) -> None:
            """Stop before any private inference operation can be authorized."""
            raise failure()

    admission = FailedAdmission()
    serving = Serving(admission, base_revision(config()))
    environment = LookupEnvironment()
    evaluator = EnvironmentEvaluator(environment, ExactAnswerScorer())
    with pytest.raises(failure):
        asyncio.run(
            run_cycle(
                directory=tmp_path,
                config=config(),
                plan=PreparedCycle(
                    scenarios=(scenario("fit"),),
                    environment=environment,
                    evaluation=evaluator.freeze((scenario("held-out"),)),
                    evaluator=evaluator,
                ),
                serving=serving,
                admission=admission,
                backend_factory=lambda lineage: ReceiptBackend(
                    tmp_path / "checkpoints", lineage, serving
                ),
            )
        )
    state = CycleState.model_validate_json(
        next((tmp_path / "cycles").glob("*/state.json")).read_bytes()
    )
    assert state.stage == "failed" and not state.serving_restored
    assert state.failure_type == failure.__name__ and admission.closed
    assert not serving.sampled and environment.opened == 0


@pytest.mark.parametrize("failure", [None, "backend_close", "recovery", "admission_close"])
def test_training_cancellation_closes_backend_before_active_serving_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """A task cancelled during training closes its session before reopening public traffic."""

    async def exercise() -> None:
        """Cancel an actually waiting backend protocol operation."""
        entered = asyncio.Event()
        settings = config()
        admission = Admission()
        serving = Serving(admission, base_revision(settings))
        environment = LookupEnvironment()
        evaluator = EnvironmentEvaluator(environment, ExactAnswerScorer())
        backends: list[ReceiptBackend] = []
        cancellations: list[asyncio.CancelledError] = []

        class WaitingSession(ReceiptSession):
            """Represent owned compute waiting for completion."""

            async def train(self, batch: TrainingBatch) -> TrainingResult:
                """Remain in flight until the caller cancels the cycle."""
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError as error:
                    cancellations.append(error)
                    raise
                raise AssertionError("unreachable")

            async def close(self) -> None:
                """Make uncertain backend cleanup observable without releasing compute."""
                if failure == "backend_close":
                    raise OSError("fixture backend close failed")
                await super().close()

        class WaitingBackend(ReceiptBackend):
            """Expose a cancellable training operation with observable cleanup."""

            async def open(
                self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
            ) -> WaitingSession:
                """Bind the actual core-created session lifecycle."""
                return WaitingSession(self, spec, resume)

        def factory(lineage: str) -> WaitingBackend:
            """Retain the adapter so cleanup can be asserted after cancellation."""
            backend = WaitingBackend(tmp_path / "checkpoints", lineage, serving)
            backends.append(backend)
            return backend

        task = asyncio.create_task(
            run_cycle(
                directory=tmp_path,
                config=settings,
                plan=PreparedCycle(
                    scenarios=(scenario("fit"),),
                    environment=environment,
                    evaluation=evaluator.freeze((scenario("held-out"),)),
                    evaluator=evaluator,
                ),
                serving=serving,
                admission=admission,
                backend_factory=factory,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1)

        async def failed_recovery() -> None:
            """Fail before inference can safely resume after cancellation."""
            raise OSError("fixture recovery failed")

        async def failed_admission_close() -> None:
            """Fail final admission release after the active serving revision is restored."""
            admission.closed = True
            raise OSError("fixture admission close failed")

        if failure == "recovery":
            monkeypatch.setattr(serving, "wake", failed_recovery)
        elif failure == "admission_close":
            monkeypatch.setattr(admission, "close", failed_admission_close)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value is cancellations[0]
        assert backends[0].closed == (failure != "backend_close")
        restored = failure not in {"backend_close", "recovery"}
        assert serving.asleep != restored
        assert admission.paused != restored
        assert admission.closed
        assert serving.loaded == base_revision(settings)
        state = CycleState.model_validate_json(
            next((tmp_path / "cycles").glob("*/state.json")).read_bytes()
        )
        assert state.failure_type == "CancelledError"
        assert state.serving_restored == restored
        if failure == "backend_close":
            assert state.cleanup_failure_type == "OSError"
        if failure == "recovery":
            assert state.recovery_failure_type == "OSError"

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [ValueError, TimeoutError])
def test_training_and_backend_cleanup_failures_preserve_both_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    """Retain the original training error and separate cleanup evidence without resuming."""
    training_error = failure("fixture training failed")
    sessions: list[ReceiptSession] = []

    async def failed_train(self: ReceiptSession, batch: TrainingBatch) -> TrainingResult:
        """Fail a real cycle's owned training call with an observable original exception."""
        sessions.append(self)
        raise training_error

    async def failed_close(self: ReceiptSession) -> None:
        """Leave backend ownership uncertain and record a different failure type."""
        raise OSError("fixture backend close failed")

    monkeypatch.setattr(ReceiptSession, "train", failed_train)
    monkeypatch.setattr(ReceiptSession, "close", failed_close)
    with pytest.raises(failure, match="fixture training failed") as raised:
        asyncio.run(drive_cycle(tmp_path))
    assert raised.value is training_error
    state = CycleState.model_validate_json(
        next((tmp_path / "cycles").glob("*/state.json")).read_bytes()
    )
    assert state.failure_type == failure.__name__
    assert state.cleanup_failure_type == "OSError"
    assert state.recovery_failure_type is None
    assert not state.serving_restored
    assert len(sessions) == 1
    backend = sessions[0].backend
    assert not backend.closed
    assert backend.serving.asleep
    assert backend.serving.admission.paused and backend.serving.admission.closed


def test_late_journal_failure_restores_committed_adapter_under_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-publication persistence failure reacquires admission and restores registry truth."""

    original = cycle._save_state
    failed = False

    def fail_complete(directory: Path, state: CycleState) -> None:
        """Fail one terminal write after activation has committed."""
        nonlocal failed
        if state.stage == "complete" and not failed:
            failed = True
            raise OSError("fixture journal write failed")
        original(directory, state)

    monkeypatch.setattr(cycle, "_save_state", fail_complete)
    with pytest.raises(OSError, match="fixture journal"):
        asyncio.run(drive_cycle(tmp_path))
    state = CycleState.model_validate_json(
        next((tmp_path / "cycles").glob("*/state.json")).read_bytes()
    )
    active = AdapterRegistry(tmp_path / "registry.json", config().scope).read().active
    assert state.stage == "failed" and state.serving_restored
    assert active.policy_revision == state.candidate_revision


def test_invalid_recipe_fails_before_scenario_source_can_dispatch(tmp_path: Path) -> None:
    """Malformed model pins cannot spend on environment preparation or synthesis."""
    called = False

    class Source:
        """Represent an optional workflow that must not run before recipe validation."""

        external_reservation_usd = 1.0

        async def prepare(self, directory: Path, config: LocalClaasConfig) -> PreparedCycle:
            """Record an impermissible dispatch if reached."""
            nonlocal called
            called = True
            raise AssertionError("source must remain unused")

    settings = config().model_copy(update={"base_model_revision": "mutable-main"})
    admission = Admission()
    serving = Serving(admission, base_revision(settings))
    with pytest.raises(ValueError, match="immutable"):
        asyncio.run(
            run_cycle(
                directory=tmp_path,
                config=settings,
                plan=Source(),
                serving=serving,
                admission=admission,
                backend_factory=lambda lineage: ReceiptBackend(
                    tmp_path / "checkpoints", lineage, serving
                ),
            )
        )
    assert not called
