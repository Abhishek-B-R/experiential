"""Portable lifecycle and dependency isolation tests for local execution."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from exp.optimize.claas.backends.subprocess import SubprocessVerlBackend
from exp.optimize.claas.training_contracts import ClaasTrainingError
from exp.optimize.claas.training_contracts_test import job, spec


def test_portable_import_never_loads_gpu_stack() -> None:
    """Gateway processes can import contracts and lifecycle without importing training libraries."""
    source = (
        "import sys; import exp.optimize.claas.backends.subprocess; "
        "assert not {'torch', 'ray', 'verl', 'peft'} & set(sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr.decode()


def test_session_requires_completion_and_can_close_without_compute(tmp_path: Path) -> None:
    """Opening does not acquire a GPU; a closed session cannot later run work."""

    async def run() -> None:
        """Drive the public lifecycle without dispatching a process."""
        backend = SubprocessVerlBackend(
            python_executable=Path(sys.executable),
            checkpoint_root=tmp_path,
            cuda_visible_device="0",
        )
        session = await backend.open(spec())
        assert session.policy_revision == "policy-0"
        with pytest.raises(ClaasTrainingError, match="before the first"):
            await session.checkpoint()
        await session.close()
        with pytest.raises(ClaasTrainingError, match="closed"):
            await session.train(job(tmp_path).batch)

    asyncio.run(run())


@pytest.mark.parametrize("device", ["", "0,1", "-1", "0; shell"])
def test_no_implicit_gpu_selection(tmp_path: Path, device: str) -> None:
    """An execution adapter must bind one exact GPU before it can open."""
    with pytest.raises(ValueError, match="exactly one"):
        SubprocessVerlBackend(
            python_executable=Path(sys.executable),
            checkpoint_root=tmp_path,
            cuda_visible_device=device,
        )


def test_real_worker_entrypoint_fails_closed_without_cuda(tmp_path: Path) -> None:
    """Actually dispatch the optional worker on CPU and refuse a success receipt."""

    async def run() -> None:
        """Run the process boundary without downloading weights or creating cloud compute."""
        backend = SubprocessVerlBackend(
            python_executable=Path(sys.executable),
            checkpoint_root=tmp_path,
            cuda_visible_device="999999",
            timeout_seconds=30,
        )
        session = await backend.open(spec())
        with pytest.raises(ClaasTrainingError, match="worker failed"):
            await session.train(job(tmp_path).batch)
        with pytest.raises(ClaasTrainingError, match="failed"):
            await session.train(job(tmp_path).batch)
        await session.close()

    asyncio.run(run())


def test_close_during_process_creation_waits_for_owned_worker_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing at the spawn await cannot strand a newly created training process."""
    original_spawn = asyncio.create_subprocess_exec

    async def run() -> None:
        """Exercise concurrent training startup and session cleanup."""
        started, release = asyncio.Event(), asyncio.Event()
        processes: list[asyncio.subprocess.Process] = []

        async def delayed_spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            """Hold process creation until the test permits ownership transfer."""
            del args, kwargs
            process = await original_spawn(sys.executable, "-c", "import time; time.sleep(60)")
            processes.append(process)
            started.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        backend = SubprocessVerlBackend(
            python_executable=Path(sys.executable),
            checkpoint_root=tmp_path,
            cuda_visible_device="0",
        )
        session = await backend.open(spec())
        update = asyncio.create_task(session.train(job(tmp_path).batch))
        await asyncio.wait_for(started.wait(), timeout=5)
        cleanup = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not cleanup.done()
        release.set()
        with pytest.raises(ClaasTrainingError, match="closed during worker startup"):
            await asyncio.wait_for(update, timeout=5)
        await asyncio.wait_for(cleanup, timeout=5)
        assert all(process.returncode is not None for process in processes)

    asyncio.run(run())


def test_failure_diagnostic_retains_bounded_tail(tmp_path: Path) -> None:
    """A useful terminal exception survives temporary worker-log cleanup."""
    from exp.optimize.claas.backends.subprocess import _worker_diagnostic

    log = tmp_path / "worker.log"
    log.write_bytes(b"earlier-output" * 10000 + b"\nCUDA out of memory\n")
    diagnostic = _worker_diagnostic(log)
    assert diagnostic.endswith("CUDA out of memory\n")
    assert len(diagnostic.encode()) <= 8192


def test_worker_environment_drops_unrelated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional ML process receives only its explicit model token and runtime settings."""
    from exp.optimize.claas.backends.subprocess import _worker_environment

    monkeypatch.setenv("OPENAI_API_KEY", "gateway-secret-canary")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cloud-secret-canary")
    monkeypatch.setenv("HF_TOKEN", "implicit-token-canary")
    monkeypatch.setenv("VERL_USE_EXTERNAL_MODULES", "unapproved.module")
    environment = _worker_environment("0", None)
    assert "OPENAI_API_KEY" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "HF_TOKEN" not in environment
    assert "VERL_USE_EXTERNAL_MODULES" not in environment
    assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert _worker_environment("0", "explicit-worker-token")["HF_TOKEN"] == "explicit-worker-token"
