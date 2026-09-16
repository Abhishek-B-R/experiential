"""Explicit Modal app construction around the shared CLaaS training worker.

Callers supply an already selected image with experiential[claas-verl] installed.
Creating this definition does not deploy it, create a Volume, or launch a GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from exp.optimize.claas.backends.modal import REMOTE_ROOT, ModalExecutionConfig
from exp.optimize.claas.backends.subprocess import SubprocessVerlBackend
from exp.optimize.claas.training_contracts import TrainingJob


def create_modal_app(*, config: ModalExecutionConfig, image: modal.Image) -> modal.App:
    """Define one serialized, finite GPU worker using an existing named Volume.

    Deploy this returned app explicitly with the Modal SDK after authorization.
    Use an immutable image containing the same CLaaS worker version as the caller.
    No retries or warm containers are configured. Volume commit precedes receipt.
    """
    app = modal.App(config.app_name)
    volume = modal.Volume.from_name(config.volume_name, environment_name=config.environment_name)

    @app.function(
        name=config.function_name,
        serialized=True,
        image=image,
        gpu=config.gpu,
        volumes={REMOTE_ROOT: volume},
        timeout=config.timeout_seconds,
        startup_timeout=config.startup_timeout_seconds,
        retries=0,
        min_containers=0,
        max_containers=1,
        single_use_containers=True,
    )
    async def train_claas(payload: str) -> str:
        """Execute the common worker once, then persist before acknowledging completion."""
        job = TrainingJob.model_validate_json(payload)
        if job.checkpoint_root != REMOTE_ROOT:
            raise ValueError("Modal jobs must use the configured durable Volume mount")
        await volume.reload.aio()
        backend = SubprocessVerlBackend(
            python_executable=Path(sys.executable),
            checkpoint_root=Path(REMOTE_ROOT),
            cuda_visible_device=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            timeout_seconds=config.timeout_seconds,
            lineage_id=job.lineage_id,
        )
        session = await backend.open(job.spec, job.resume_checkpoint)
        try:
            result = await session.train(job.batch)
            await volume.commit.aio()
            return result.model_dump_json()
        finally:
            await session.close()

    return app
