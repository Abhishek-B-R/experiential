"""Portable wake, practice, train, evaluate, and publication transactions for local CLaaS."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from exp.common.claas.scenarios import Scenario
from exp.common.core.artifacts import ContractModel, canonical_json_bytes, sha256_json
from exp.common.core.files import write_text_atomic
from exp.common.core.locks import file_write_lock
from exp.optimize.claas.backends.checkpoints import verify_checkpoint, verify_training_result
from exp.optimize.claas.configuration import LocalClaasConfig
from exp.optimize.claas.evaluation.paired import EvaluationManifest, evaluate_policies
from exp.optimize.claas.evaluation.promotion import PromotionDecision, decide_promotion
from exp.optimize.claas.lifecycle.inputs import CycleSource, PreparedCycle
from exp.optimize.claas.lifecycle.rollouts import (
    RevisionPolicy,
    ServingController,
    collect_practice,
)
from exp.optimize.claas.training_contracts import (
    ClaasTrainingBackend,
    ClaasTrainingSpec,
    TrainingCheckpoint,
    TrainingJob,
)
from exp.runtime.claas.registry import AdapterRegistry, RegistryState, ServingRevision


class AdmissionController(Protocol):
    """Cross-process request admission tied to the published registry generation."""

    async def pause_and_drain(self, *, timeout_seconds: float = 120) -> None:
        """Durably pause before taking exclusive ownership of admitted traffic."""
        ...

    async def resume(
        self, *, expected_registry_generation: int, expected_policy_revision: str
    ) -> None:
        """Publish readiness only for the already loaded and committed revision."""
        ...

    async def close(self) -> None:
        """Release ownership while preserving a paused state after uncertain failure."""
        ...


class CycleState(ContractModel):
    """Durable cycle progress and terminal decision, without secret or exception contents."""

    cycle_id: str
    stage: Literal["prepared", "practice", "training", "evaluation", "complete", "failed"]
    baseline_generation: int
    baseline_revision: str
    candidate_revision: str | None = None
    decision: PromotionDecision | None = None
    failure_type: str | None = None
    cleanup_failure_type: str | None = None
    recovery_failure_type: str | None = None
    serving_restored: bool = False
    evidence_sha256: str | None = None


def base_revision(config: LocalClaasConfig) -> ServingRevision:
    """Bind a stable application base identity independently of mutable cycle limits."""
    identity = {
        "scope": config.scope.model_dump(mode="json"),
        "model": config.base_model,
        "revision": config.base_model_revision,
        "tokenizer": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
    }
    return ServingRevision(
        scope=config.scope,
        policy_revision="base-" + sha256_json(identity),
        model_id=config.base_model,
        model_revision=config.base_model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=config.tokenizer_revision,
    )


def training_spec(config: LocalClaasConfig) -> ClaasTrainingSpec:
    """Create the immutable optimizer recipe shared by local and remote workers."""
    if (
        not re.fullmatch(r"[0-9a-f]{40}", config.base_model_revision)
        or not re.fullmatch(r"[0-9a-f]{40}", config.tokenizer_revision)
        or Path(config.base_model).is_absolute()
        or Path(config.tokenizer_id).is_absolute()
    ):
        raise ValueError(
            "training requires Hugging Face IDs and immutable 40-character commit revisions"
        )
    return ClaasTrainingSpec(
        scope=config.scope,
        adapter_id="application",
        base_model=config.base_model,
        model_revision=config.base_model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=config.tokenizer_revision,
        initial_policy_revision=base_revision(config).policy_revision,
        objective=config.objective,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        learning_rate=config.learning_rate,
    )


def checkpoint_for_revision(
    directory: Path, revision: ServingRevision, spec: ClaasTrainingSpec
) -> TrainingCheckpoint | None:
    """Verify resumable training state before loading an adapter or acquiring training compute."""
    if revision.adapter_directory is None:
        if revision.policy_revision != spec.initial_policy_revision:
            raise ValueError("base registry revision differs from this application's recipe")
        return None
    checkpoint = TrainingCheckpoint.model_validate_json(
        _checkpoint_receipt_path(directory, revision.policy_revision).read_bytes()
    )
    manifest = verify_checkpoint(checkpoint, spec)
    if (
        checkpoint.policy_revision != revision.policy_revision
        or checkpoint.manifest_sha256 != revision.manifest_sha256
        or str(Path(checkpoint.path) / manifest.serving_adapter_directory)
        != revision.adapter_directory
    ):
        raise ValueError("serving registry and verified training checkpoint differ")
    return checkpoint


async def run_cycle(
    *,
    directory: Path,
    config: LocalClaasConfig,
    plan: PreparedCycle | CycleSource,
    serving: ServingController,
    admission: AdmissionController,
    backend_factory: Callable[[str], ClaasTrainingBackend],
    compute_reservation_usd: float = 0.0,
) -> CycleState:
    """Run one bounded update, then activate only a complete held-out improvement.

    The application lock serializes cycles and rollback. The public gateway stays
    paused from baseline sampling until a verified selected revision is loaded.
    Every checkpoint is retained, including rejected candidates, while a fresh
    cycle lineage can resume the actual active revision after rejection.
    """
    directory = directory.resolve()
    with file_write_lock(directory / "cycle", what="CLaaS learning cycle"):
        external_reservation_usd = plan.external_reservation_usd
        if any(
            not math.isfinite(value) or value < 0
            for value in (external_reservation_usd, compute_reservation_usd)
        ):
            raise ValueError("external and compute reservations must be finite and nonnegative")
        if external_reservation_usd + compute_reservation_usd > config.limits.maximum_cost_usd:
            raise ValueError(
                "combined external and compute reservations exceed the configured cycle ceiling"
            )
        registry = AdapterRegistry(directory / "registry.json", config.scope)
        baseline = registry.initialize(base_revision(config))
        spec = training_spec(config)
        resume = checkpoint_for_revision(directory, baseline.active, spec)
        prepared = (
            plan if isinstance(plan, PreparedCycle) else await plan.prepare(directory, config)
        )
        if prepared.external_reservation_usd != external_reservation_usd:
            raise ValueError("scenario source changed its declared reservation")
        manifest = EvaluationManifest.model_validate_json(prepared.evaluation.model_dump_json())
        evaluator = prepared.evaluator
        scenarios = tuple(
            Scenario.model_validate_json(item.model_dump_json()) for item in prepared.scenarios
        )
        evidence = dict(prepared.evidence)
        context_bytes = canonical_json_bytes(evidence)
        if len(context_bytes) > 268_435_456:
            raise ValueError("cycle context exceeds 256 MiB; bound the supplied evidence")
        if any(item.scope != config.scope for item in scenarios) or manifest.scope != config.scope:
            raise ValueError("cycle scenarios belong to another application")
        if not scenarios or len({item.scenario_id for item in scenarios}) != len(scenarios):
            raise ValueError("practice requires nonempty uniquely identified scenarios")
        if {item.scenario_id for item in scenarios}.intersection(
            task.scenario_id for task in manifest.tasks
        ):
            raise ValueError("practice and evaluation scenarios overlap")
        practice_sources = {
            identity for item in scenarios for identity in item.source_experience_ids
        }
        if practice_sources.intersection(
            identity for task in manifest.tasks for identity in task.source_experience_ids
        ):
            raise ValueError("practice and evaluation source evidence overlaps")
        if evaluator.evaluator_id != manifest.evaluator_id:
            raise ValueError("evaluator differs from frozen evaluation")
        if len(manifest.tasks) < config.promotion.minimum_evaluation_tasks:
            raise ValueError("frozen evaluation does not meet the configured minimum task count")
        cycle_id = uuid4().hex
        run_directory = directory / "cycles" / cycle_id
        state = CycleState(
            cycle_id=cycle_id,
            stage="prepared",
            baseline_generation=baseline.generation,
            baseline_revision=baseline.active.policy_revision,
            evidence_sha256=sha256_json(evidence or {}),
        )
        _save_state(run_directory, state)
        write_text_atomic(run_directory / "context.json", context_bytes.decode() + "\n")
        write_text_atomic(run_directory / "evaluation.json", manifest.model_dump_json() + "\n")
        try:
            await admission.pause_and_drain()
        except BaseException as error:
            state = state.model_copy(
                update={
                    "stage": "failed",
                    "failure_type": type(error).__name__,
                    "serving_restored": False,
                }
            )
            try:
                _save_state(run_directory, state)
            finally:
                await _close_admission(admission, error)
            raise
        cleanup_failure: BaseException | None = None
        pending_failure: BaseException | None = None
        try:
            await serving.pause_and_drain()
            await serving.wake()
            await serving.load_revision(baseline.active)
            state = state.model_copy(update={"stage": "practice"})
            _save_state(run_directory, state)
            batch = await collect_practice(
                scenarios=scenarios,
                environment=prepared.environment,
                serving=serving,
                revision=baseline.active,
                spec=spec,
                limits=config.limits,
                directory=run_directory / "practice",
                cycle_id=cycle_id,
            )
            write_text_atomic(run_directory / "batch.json", batch.model_dump_json() + "\n")
            await serving.sleep()
            state = state.model_copy(update={"stage": "training"})
            _save_state(run_directory, state)
            backend = backend_factory(cycle_id)
            session = await backend.open(spec, resume)
            training_failure: BaseException | None = None
            try:
                async with asyncio.timeout(config.limits.maximum_training_seconds):
                    result = await session.train(batch)
            except BaseException as error:
                training_failure = error
                raise
            finally:
                try:
                    await session.close()
                except BaseException as error:
                    cleanup_failure = error
                    if isinstance(training_failure, asyncio.CancelledError) and isinstance(
                        error, Exception
                    ):
                        raise training_failure from error
                    raise
            job = TrainingJob(
                spec=spec,
                batch=batch,
                checkpoint_root=str(directory / "checkpoints"),
                resume_checkpoint=resume,
                lineage_id=cycle_id,
            )
            checkpoint_manifest = verify_training_result(job, result)
            checkpoint = result.checkpoint
            candidate = baseline.active.model_copy(
                update={
                    "policy_revision": checkpoint.policy_revision,
                    "adapter_directory": str(
                        Path(checkpoint.path) / checkpoint_manifest.serving_adapter_directory
                    ),
                    "manifest_sha256": checkpoint.manifest_sha256,
                }
            )
            write_text_atomic(
                _checkpoint_receipt_path(directory, checkpoint.policy_revision),
                checkpoint.model_dump_json() + "\n",
            )
            state = state.model_copy(
                update={"stage": "evaluation", "candidate_revision": candidate.policy_revision}
            )
            _save_state(run_directory, state)
            await serving.wake()
            report = await evaluate_policies(
                manifest,
                current=RevisionPolicy(
                    serving,
                    baseline.active,
                    maximum_response_tokens=config.limits.maximum_response_tokens,
                ),
                candidate=RevisionPolicy(
                    serving,
                    candidate,
                    maximum_response_tokens=config.limits.maximum_response_tokens,
                ),
                evaluator=evaluator,
            )
            write_text_atomic(run_directory / "report.json", report.model_dump_json() + "\n")
            decision = decide_promotion(
                manifest=manifest,
                report=report,
                baseline=baseline,
                candidate=candidate,
                policy=config.promotion,
                evaluator=evaluator,
            )
            write_text_atomic(run_directory / "decision.json", decision.model_dump_json() + "\n")
            selected = candidate if decision.approved else baseline.active
            await serving.load_revision(selected)
            current = (
                registry.activate(candidate, expected_generation=baseline.generation)
                if decision.approved
                else registry.read()
            )
            if current.active != selected:
                raise ValueError(
                    "active registry changed during evaluation; restore current serving"
                )
            await _resume(serving, admission, current)
            state = state.model_copy(
                update={"stage": "complete", "decision": decision, "serving_restored": True}
            )
            _save_state(run_directory, state)
            return state
        except BaseException as error:
            pending_failure = error
            restored = False
            recovery_failure: BaseException | None = None
            try:
                if cleanup_failure is None:
                    await _restore_active(directory, registry, spec, serving, admission)
                    restored = True
            except BaseException as recovery_error:
                recovery_failure = recovery_error
                if isinstance(error, asyncio.CancelledError) and isinstance(
                    recovery_error, Exception
                ):
                    raise error from recovery_error
                pending_failure = recovery_error
                raise
            finally:
                state = state.model_copy(
                    update={
                        "stage": "failed",
                        "failure_type": type(error).__name__,
                        "cleanup_failure_type": (
                            type(cleanup_failure).__name__ if cleanup_failure is not None else None
                        ),
                        "recovery_failure_type": (
                            type(recovery_failure).__name__
                            if recovery_failure is not None
                            else None
                        ),
                        "serving_restored": restored,
                    }
                )
                _save_state(run_directory, state)
            raise
        finally:
            await _close_admission(admission, pending_failure)


async def rollback(
    *,
    directory: Path,
    config: LocalClaasConfig,
    serving: ServingController,
    admission: AdmissionController,
) -> RegistryState:
    """Load the previous verified revision before changing the active registry pointer."""
    with file_write_lock(directory / "cycle", what="CLaaS rollback"):
        registry = AdapterRegistry(directory / "registry.json", config.scope)
        baseline = registry.read()
        if baseline.previous is None:
            raise ValueError("no previous adapter is available for rollback")
        spec = training_spec(config)
        checkpoint_for_revision(directory, baseline.previous, spec)
        try:
            await admission.pause_and_drain()
        except BaseException as error:
            await _close_admission(admission, error)
            raise
        pending_failure: BaseException | None = None
        try:
            await serving.pause_and_drain()
            await serving.wake()
            await serving.load_revision(baseline.previous)
            current = registry.rollback(expected_generation=baseline.generation)
            await _resume(serving, admission, current)
            return current
        except BaseException as error:
            pending_failure = error
            try:
                await _restore_active(directory, registry, spec, serving, admission)
            except BaseException as recovery_error:
                if isinstance(error, asyncio.CancelledError) and isinstance(
                    recovery_error, Exception
                ):
                    raise error from recovery_error
                pending_failure = recovery_error
                raise
            raise
        finally:
            await _close_admission(admission, pending_failure)


async def _close_admission(admission: AdmissionController, failure: BaseException | None) -> None:
    """Release admission without replacing an existing failure with an ordinary close error."""
    try:
        await admission.close()
    except Exception as close_error:
        if failure is not None:
            raise failure from close_error
        raise


async def _restore_active(
    directory: Path,
    registry: AdapterRegistry,
    spec: ClaasTrainingSpec,
    serving: ServingController,
    admission: AdmissionController,
) -> None:
    """Recover the durable active revision after a failed or uncertain state transition."""
    await admission.pause_and_drain()
    await serving.pause_and_drain()
    current = registry.read()
    checkpoint_for_revision(directory, current.active, spec)
    await serving.wake()
    await serving.load_revision(current.active)
    await _resume(serving, admission, current)


async def _resume(
    serving: ServingController, admission: AdmissionController, state: RegistryState
) -> None:
    """Reopen only the loaded registry generation after all pointer writes complete."""
    await serving.resume()
    await admission.resume(
        expected_registry_generation=state.generation,
        expected_policy_revision=state.active.policy_revision,
    )


def _checkpoint_receipt_path(directory: Path, revision: str) -> Path:
    """Locate a checkpoint receipt without treating revision labels as path segments."""
    return directory / "checkpoint-receipts" / f"{sha256_json(revision)}.json"


def _save_state(directory: Path, state: CycleState) -> None:
    """Replace one complete journal record without exposing partial progress JSON."""
    write_text_atomic(directory / "state.json", state.model_dump_json() + "\n")
