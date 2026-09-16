"""Finite local GPU execution with typed receipts and no eager training imports."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

from exp.optimize.claas.backends.checkpoints import verify_checkpoint, verify_training_result
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingJob,
    TrainingResult,
    TrainingSession,
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
        model_access_token: str | None = None,
        lineage_id: str = "main",
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
        if not lineage_id.strip() or len(lineage_id) > 512:
            raise ValueError("lineage_id must contain 1 to 512 nonblank characters")
        self._python = python_executable
        self._root = checkpoint_root
        self._device = cuda_visible_device
        self._timeout = timeout_seconds
        self._model_access_token = model_access_token
        self._lineage_id = lineage_id

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
                lineage_id=self._backend._lineage_id,
            )
            with tempfile.TemporaryDirectory(prefix="claas-job-") as directory:
                root = Path(directory)
                job_path, result_path = root / "job.json", root / "result.json"
                job_path.write_text(job.model_dump_json())
                job_path.chmod(0o600)
                environment = _worker_environment(
                    self._backend._device, self._backend._model_access_token
                )
                with (root / "worker.log").open("wb") as output:
                    process = await asyncio.create_subprocess_exec(
                        str(self._backend._python),
                        "-m",
                        "exp.optimize.claas.backends.verl.worker",
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
                        output.flush()
                        diagnostic = _worker_diagnostic(root / "worker.log")
                        raise ClaasTrainingError(
                            "veRL worker failed; verify experiential[claas-verl], "
                            "pinned model/tokenizer, "
                            "and one BF16 CUDA device. Do not retry an uncertain optimizer update."
                            f"\nLast worker output (bounded):\n{diagnostic}"
                        )
                self._process = None
                try:
                    result = TrainingResult.model_validate_json(result_path.read_text())
                    if (
                        not Path(result.checkpoint.path)
                        .resolve()
                        .is_relative_to(self._backend._root.resolve())
                    ):
                        raise ValueError("worker receipt does not match the submitted batch")
                    verify_training_result(job, result)
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


def _worker_diagnostic(path: Path) -> str:
    """Keep a bounded failure tail in the raised exception after temporary logs are removed."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 8192))
        return handle.read(8192).decode("utf-8", errors="replace")


def _worker_environment(device: str, model_access_token: str | None) -> dict[str, str]:
    """Forward runtime settings and only an explicitly supplied model credential."""
    allowed = {
        "PATH",
        "HOME",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    }
    environment = {name: value for name, value in os.environ.items() if name in allowed}
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": device,
            "VERL_USE_EXTERNAL_PLUGINS": "none",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
    )
    if model_access_token is not None:
        environment["HF_TOKEN"] = model_access_token
    return environment
