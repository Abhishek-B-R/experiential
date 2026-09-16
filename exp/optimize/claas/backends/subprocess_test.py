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
