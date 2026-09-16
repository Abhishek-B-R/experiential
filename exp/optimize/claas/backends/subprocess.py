"""Finite local GPU execution with typed receipts and no eager training imports."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

from exp.optimize.claas.backends.checkpoints import verify_checkpoint
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


class SubprocessVerlBackend:
    """Run one bounded CUDA worker process per batch, releasing its GPU on exit.

    Example:
        ``backend = SubprocessVerlBackend(python_executable=Path("/worker/bin/python"),
        checkpoint_root=Path("/durable/adapters"), cuda_visible_device="0")``
        ``session = await backend.open(spec); result = await session.train(batch)``

    The selected interpreter must have ``experiential[claas-verl]`` installed.
    Selecting this backend is explicit authorization to use the named local GPU;
    it never discovers GPUs, downloads dependencies, or creates cloud resources.
    """

    def __init__(
        self,
        *,
        python_executable: Path,
        checkpoint_root: Path,
        cuda_visible_device: str,
        timeout_seconds: float = 1800,
    ) -> None:
        """Bind exact runtime, durable storage, one device, and finite job timeout."""
        if not python_executable.is_absolute() or not python_executable.is_file():
            raise ValueError("python_executable must name an existing absolute worker interpreter")
        if not checkpoint_root.is_absolute():
            raise ValueError("checkpoint_root must be an absolute durable directory")
        if re.fullmatch(r"(?:[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+)", cuda_visible_device) is None:
            raise ValueError("cuda_visible_device must identify exactly one authorized GPU")
        if not 0 < timeout_seconds <= 86400:
            raise ValueError("timeout_seconds must be finite and between zero and 86400")
        self._python = python_executable
        self._root = checkpoint_root
        self._device = cuda_visible_device
        self._timeout = timeout_seconds

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
    ) -> TrainingSession:
        """Validate resume state before returning a session; no GPU is loaded yet."""
        if resume:
            verify_checkpoint(resume, spec)
        return _SubprocessSession(self, spec, resume)


class _SubprocessSession:
    """Serialized optimizer execution and the last verified completion receipt."""

    def __init__(
        self,
        backend: SubprocessVerlBackend,
        spec: ClaasTrainingSpec,
        resume: TrainingCheckpoint | None,
    ) -> None:
        """Bind one application identity without acquiring compute."""
        self._backend = backend
        self._spec = spec
        self._checkpoint = resume
        self._closed = False
        self._failed = False
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None

    @property
    def policy_revision(self) -> str:
        """Return the revision required by the next batch."""
        return (
            self._checkpoint.policy_revision
            if self._checkpoint
            else self._spec.initial_policy_revision
        )

    async def train(self, batch: TrainingBatch) -> TrainingResult:
        """Dispatch one update and verify its identity and state before acknowledging it."""
        async with self._lock:
            if self._closed or self._failed:
                raise ClaasTrainingError(
                    "session is closed or failed; reopen verified checkpoint state"
                )
            job = TrainingJob(
                spec=self._spec,
                batch=batch,
                checkpoint_root=str(self._backend._root),
                resume_checkpoint=self._checkpoint,
            )
            with tempfile.TemporaryDirectory(prefix="claas-job-") as directory:
                root = Path(directory)
                job_path, result_path = root / "job.json", root / "result.json"
                job_path.write_text(job.model_dump_json())
                job_path.chmod(0o600)
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = self._backend._device
                environment["VERL_USE_EXTERNAL_PLUGINS"] = "none"
                environment.pop("VERL_USE_EXTERNAL_MODULES", None)
                with (root / "worker.log").open("wb") as output:
                    process = await asyncio.create_subprocess_exec(
                        str(self._backend._python),
                        "-m",
                        "exp.optimize.claas.backends.verl_worker",
                        "--job",
                        str(job_path),
                        "--result",
                        str(result_path),
                        env=environment,
                        stdout=output,
                        stderr=output,
                    )
                    self._process = process
                    if self._closed:
                        self._failed = True
                        if process.returncode is None:
                            process.kill()
                        await process.wait()
                        self._process = None
                        raise ClaasTrainingError("session closed during worker startup")
                    try:
                        await asyncio.wait_for(process.wait(), timeout=self._backend._timeout)
                    except (TimeoutError, asyncio.CancelledError):
                        self._failed = True
                        if process.returncode is None:
                            process.kill()
                        await process.wait()
                        raise
                    if process.returncode != 0:
                        self._failed = True
                        raise ClaasTrainingError(
                            "veRL worker failed; verify experiential[claas-verl], "
                            "pinned model/tokenizer, "
                            "and one BF16 CUDA device. Do not retry an uncertain optimizer update."
                        )
                self._process = None
                try:
                    result = TrainingResult.model_validate_json(result_path.read_text())
                    expected_ids = tuple(item.experience.experience_id for item in batch.examples)
                    if (
                        result.checkpoint.policy_revision != next_policy_revision(job)
                        or result.consumed_experience_ids != expected_ids
                        or result.checkpoint.step
                        != (self._checkpoint.step if self._checkpoint else 0) + 1
                        or not Path(result.checkpoint.path)
                        .resolve()
                        .is_relative_to(self._backend._root.resolve())
                    ):
                        raise ValueError("worker receipt does not match the submitted batch")
                    verify_checkpoint(result.checkpoint, self._spec)
                except (OSError, ValueError):
                    self._failed = True
                    raise
                self._checkpoint = result.checkpoint
                return result

    async def checkpoint(self) -> TrainingCheckpoint:
        """Return only verified, completed state; initial model weights are not a checkpoint."""
        if self._checkpoint is None:
            raise ClaasTrainingError(
                "no training checkpoint exists before the first completed update"
            )
        verify_checkpoint(self._checkpoint, self._spec)
        return self._checkpoint

    async def close(self) -> None:
        """Stop any owned worker, wait for resource release, and close the session."""
        self._closed = True
        if self._process is not None and self._process.returncode is None:
            self._failed = True
            self._process.kill()
            await self._process.wait()
        # A concurrent train may still be awaiting process creation. It observes
        # _closed immediately after creation and kills that process before this
        # lock is released. Returning from close therefore proves no owned worker.
        async with self._lock:
            self._process = None
