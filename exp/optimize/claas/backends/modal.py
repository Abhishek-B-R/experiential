"""Optional Modal execution of the shared worker with verified local checkpoints.

Constructing or opening this backend performs no paid call. Training validates
one explicit cost authorization before dispatch, with no automatic job retry.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path, PurePosixPath

import modal

from exp.common.core.artifacts import sha256_json
from exp.common.core.files import fsync_directory_best_effort
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    verify_checkpoint,
    verify_training_result,
)
from exp.optimize.claas.backends.modal_configuration import ModalExecutionConfig
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingJob,
    TrainingResult,
    TrainingSession,
    next_policy_revision,
)

REMOTE_ROOT = "/claas/checkpoints"


class ModalVerlBackend:
    """Execute a single authorized update in a predeployed Modal worker.

    Example:
        ``backend = ModalVerlBackend(config=config, checkpoint_root=Path("/durable/adapters"))``
        ``session = await backend.open(spec, resume); result = await session.train(batch)``

    The remote app must be created by ``create_modal_app`` with the same config
    and a pinned worker image. Checkpoints are committed to its existing Volume,
    downloaded within a byte ceiling, and verified before any success receipt.
    """

    def __init__(
        self, *, config: ModalExecutionConfig, checkpoint_root: Path, lineage_id: str = "main"
    ) -> None:
        """Bind explicit deployment settings without looking up cloud resources."""
        if not checkpoint_root.is_absolute():
            raise ValueError("checkpoint_root must be an absolute durable local directory")
        if not lineage_id.strip() or len(lineage_id) > 512:
            raise ValueError("lineage_id must be a nonblank identifier of at most 512 characters")
        self.lineage_id = lineage_id
        self.config = config
        self.checkpoint_root = checkpoint_root.resolve()

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
    ) -> TrainingSession:
        """Validate local resume evidence without acquiring compute or reading credentials."""
        if resume:
            verify_checkpoint(resume, spec)
            _remote_checkpoint(resume, self.checkpoint_root)
        return _ModalSession(self, spec, resume)

    async def execute(self, job: TrainingJob) -> TrainingResult:
        """Dispatch exactly once, commit remotely, then materialize verified local state."""
        config = self.config
        function = modal.Function.from_name(
            config.app_name, config.function_name, environment_name=config.environment_name
        )
        function = function.with_options(
            gpu=config.gpu,
            timeout=config.timeout_seconds,
            retries=0,
            max_containers=1,
        )
        # Shield handle creation so cancellation cannot leave an untracked remote call.
        spawning = asyncio.create_task(function.spawn.aio(job.model_dump_json()))
        try:
            call = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            call = await spawning
            await call.cancel.aio(terminate_containers=True)
            raise
        try:
            payload = await call.get.aio(
                timeout=config.timeout_seconds + config.startup_timeout_seconds
            )
        except BaseException:
            await call.cancel.aio(terminate_containers=True)
            raise
        if not isinstance(payload, str):
            raise ClaasTrainingError(
                "Modal returned an invalid receipt; inspect the durable Volume before retrying"
            )
        result = TrainingResult.model_validate_json(payload)
        expected = tuple(item.experience.experience_id for item in job.batch.examples)
        checkpoint = result.checkpoint
        if (
            checkpoint.scope != job.spec.scope
            or checkpoint.adapter_id != job.spec.adapter_id
            or checkpoint.policy_revision != next_policy_revision(job)
            or checkpoint.step != (job.resume_checkpoint.step if job.resume_checkpoint else 0) + 1
            or result.consumed_experience_ids != expected
        ):
            raise ClaasTrainingError(
                "Modal receipt differs from submitted job; inspect durable state"
            )
        volume = modal.Volume.from_name(
            config.volume_name, environment_name=config.environment_name
        )
        local = await _download_checkpoint(
            volume,
            checkpoint,
            job.spec,
            self.checkpoint_root,
            config.maximum_checkpoint_bytes,
            job.lineage_id,
        )
        local_result = result.model_copy(update={"checkpoint": local})
        verify_training_result(job, local_result)
        return local_result


def _remote_checkpoint(checkpoint: TrainingCheckpoint, local_root: Path) -> TrainingCheckpoint:
    """Translate a verified local receipt to the same immutable remote Volume path."""
    try:
        relative = Path(checkpoint.path).relative_to(local_root)
    except ValueError as exc:
        raise ValueError("resume checkpoint is outside this backend's checkpoint_root") from exc
    if ".." in relative.parts:
        raise ValueError("resume checkpoint contains a parent traversal")
    return checkpoint.model_copy(
        update={"path": str(PurePosixPath(REMOTE_ROOT) / relative.as_posix())}
    )


async def _download_checkpoint(
    volume: modal.Volume,
    checkpoint: TrainingCheckpoint,
    spec: ClaasTrainingSpec,
    local_root: Path,
    maximum_bytes: int,
    lineage_id: str = "main",
) -> TrainingCheckpoint:
    """Materialize only manifest-listed immutable files and validate before atomic rename."""
    expected_scope = sha256_json(
        {"scope": spec.scope.model_dump(mode="json"), "adapter_id": spec.adapter_id}
    )
    lineage = sha256_json({"lineage_id": lineage_id})
    expected_path = (
        PurePosixPath(REMOTE_ROOT) / expected_scope / lineage / checkpoint.policy_revision
    )
    if PurePosixPath(checkpoint.path) != expected_path:
        raise ValueError("Modal checkpoint path does not match the exact application and revision")
    remote = expected_path.relative_to(REMOTE_ROOT)
    directory = local_root / expected_scope / lineage
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / checkpoint.policy_revision
    if destination.exists():
        local = checkpoint.model_copy(update={"path": str(destination)})
        verify_checkpoint(local, spec)
        return local
    total = 0
    with tempfile.TemporaryDirectory(prefix=".download-", dir=directory) as staging:
        root = Path(staging)

        async def download(relative: str) -> None:
            """Download one safe path while counting all transferred bytes."""
            nonlocal total
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts or not parts.parts:
                raise ValueError("checkpoint manifest contains an unsafe file path")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                target.chmod(0o600)
                async for chunk in volume.read_file.aio(str(remote / relative)):
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise ValueError(
                            "Modal checkpoint exceeds the authorized download byte ceiling"
                        )
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())

        await download("manifest.json")
        manifest = CheckpointManifest.model_validate_json((root / "manifest.json").read_bytes())
        if sha256_json(manifest) != checkpoint.manifest_sha256:
            raise ValueError("remote checkpoint manifest changed; do not activate it")
        for relative in manifest.files:
            if relative == "manifest.json":
                raise ValueError("manifest must not include itself")
            await download(relative)
        staged = checkpoint.model_copy(update={"path": str(root)})
        verify_checkpoint(staged, spec)
        for subdirectory in sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            fsync_directory_best_effort(subdirectory)
        fsync_directory_best_effort(root)
        root.rename(destination)
        fsync_directory_best_effort(destination.parent)
    return checkpoint.model_copy(update={"path": str(destination)})


class _ModalSession:
    """One explicit cost authorization and one isolated optimizer update."""

    def __init__(
        self, backend: ModalVerlBackend, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None
    ) -> None:
        """Bind local state and defer remote work until a valid batch arrives."""
        self._backend, self._spec, self._checkpoint = backend, spec, resume
        self._closed = False
        self._spent = False
        self._lock = asyncio.Lock()
        self._running: asyncio.Task[TrainingResult] | None = None

    @property
    def policy_revision(self) -> str:
        """Return the policy expected by this session's one batch."""
        return (
            self._checkpoint.policy_revision
            if self._checkpoint
            else self._spec.initial_policy_revision
        )

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Validate before spending and never silently retry an uncertain update."""
        async with self._lock:
            if self._closed or self._spent:
                raise ClaasTrainingError(
                    "Modal authorization is consumed; inspect completion and open a fresh session"
                )
            job = TrainingJob(
                spec=self._spec,
                batch=batch,
                checkpoint_root=REMOTE_ROOT,
                lineage_id=self._backend.lineage_id,
                resume_checkpoint=_remote_checkpoint(
                    self._checkpoint, self._backend.checkpoint_root
                )
                if self._checkpoint
                else None,
            )
            self._spent = True
            self._running = asyncio.create_task(self._backend.execute(job))
            try:
                result = await self._running
            finally:
                self._running = None
            self._checkpoint = result.checkpoint
            return result

    async def checkpoint(self) -> TrainingCheckpoint:
        """Return only fully downloaded and verified completion evidence."""
        if self._checkpoint is None:
            raise ClaasTrainingError("no checkpoint exists before a completed Modal update")
        verify_checkpoint(self._checkpoint, self._spec)
        return self._checkpoint

    async def close(self) -> None:
        """Cancel an owned remote call and wait for its cancellation request to complete.

        Modal acknowledges cancellation, while its configured hard timeout remains
        the billing bound. Cancellation is not a successful optimizer receipt.
        """
        self._closed = True
        if self._running is not None:
            self._running.cancel()
        async with self._lock:
            self._running = None
