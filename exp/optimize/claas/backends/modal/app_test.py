"""Construct the real Modal definition without deploying or allocating compute."""

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import modal
import pytest

from exp.optimize.claas.backends.modal import app as modal_app
from exp.optimize.claas.backends.modal.app import create_modal_app
from exp.optimize.claas.backends.modal.backend import REMOTE_ROOT
from exp.optimize.claas.backends.modal.backend_test import config
from exp.optimize.claas.backends.subprocess import SubprocessVerlBackend
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingCheckpoint,
    TrainingSession,
)
from exp.optimize.claas.training_contracts_test import job


def test_app_definition_requires_no_cloud_or_gpu_runtime() -> None:
    """A real SDK app can be built locally with deferred image and Volume handles."""
    app = create_modal_app(config=config(), image=modal.Image.debian_slim())
    assert isinstance(app, modal.App)
    assert app.name == "fixture"


def test_optional_adapter_does_not_import_training_stack() -> None:
    """Importing the explicit execution adapter does not eagerly load model libraries."""
    source = (
        "import sys; import exp.optimize.claas.backends.modal.app; "
        "assert not {'torch', 'ray', 'verl', 'peft'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("device", [None, "GPU-fixture"])
def test_modal_single_gpu_binds_explicit_local_device(
    monkeypatch: pytest.MonkeyPatch, device: str | None
) -> None:
    """Invoke the actual SDK definition without cloud calls or launching its worker."""
    if device is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device)
    observed: list[str] = []

    class ReachedWorkerBoundary(RuntimeError):
        """Stop after the real subprocess constructor validates its selected device."""

    class InspectBackend(SubprocessVerlBackend):
        """Keep real runtime validation but do not create a worker or download weights."""

        async def open(
            self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None
        ) -> TrainingSession:
            """Record the GPU that the Modal definition passed to the common worker."""
            observed.append(self._device)
            raise ReachedWorkerBoundary

    reload_volume = AsyncMock()
    monkeypatch.setattr(modal.Volume, "reload", MagicMock(aio=reload_volume))
    monkeypatch.setattr(
        modal.Client, "from_env", MagicMock(side_effect=AssertionError("unexpected cloud lookup"))
    )
    monkeypatch.setattr(modal_app, "SubprocessVerlBackend", InspectBackend)
    app = create_modal_app(config=config(), image=modal.Image.debian_slim())
    function = app._local_state.functions["train_claas"]
    payload = job(Path(REMOTE_ROOT)).model_dump_json()

    async def invoke() -> None:
        """Execute the registered function locally until the explicit worker boundary."""
        with pytest.raises(ReachedWorkerBoundary):
            await function.local(payload)

    asyncio.run(invoke())
    reload_volume.assert_awaited_once()
    assert observed == [device if device is not None else "0"]
