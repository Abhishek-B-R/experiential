"""Provider-neutral execution settings and explicit local or Modal worker selection."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.optimize.claas.backends.modal.configuration import ModalExecutionConfig
from exp.optimize.claas.backends.subprocess import SubprocessVerlBackend
from exp.optimize.claas.configuration import LocalClaasConfig
from exp.optimize.claas.training_contracts import ClaasTrainingBackend


class ExecutionSettings(ContractModel):
    """An explicit private inference endpoint and worker environment without credentials."""

    private_base_url: str
    worker_python: Path
    cuda_visible_device: str = Field(default="0", min_length=1, pattern=r"^[0-9]+$")
    decoder: Literal["qwen35", "hermes"] = "qwen35"
    modal: ModalExecutionConfig | None = None

    @model_validator(mode="after")
    def _validate_private_runtime(self) -> ExecutionSettings:
        """Keep the privileged vLLM control API on loopback and process paths explicit."""
        url = urlsplit(self.private_base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("private_base_url must be a loopback HTTP origin without credentials")
        if not self.worker_python.is_absolute():
            raise ValueError("worker_python must name an absolute worker environment executable")
        return self


def validate_worker_runtime(settings: ExecutionSettings, config: LocalClaasConfig) -> None:
    """Reject a missing local interpreter before credentials or provider work are requested."""
    if config.compute == "local" and (
        not settings.worker_python.is_file() or not os.access(settings.worker_python, os.X_OK)
    ):
        raise ValueError(
            "local worker_python must name an existing executable interpreter; "
            "bind the installed claas-verl environment before training"
        )


def create_training_backend(
    settings: ExecutionSettings, config: LocalClaasConfig, directory: Path, lineage_id: str
) -> ClaasTrainingBackend:
    """Construct the selected backend through one explicit optional-plugin boundary.

    Local execution requires no Modal SDK. Selecting Modal loads its fixed adapter
    module immediately and propagates a missing extra or invalid configuration;
    no import failure selects another backend or changes execution semantics.
    """
    validate_worker_runtime(settings, config)
    if config.compute == "local":
        return SubprocessVerlBackend(
            python_executable=settings.worker_python,
            checkpoint_root=directory / "checkpoints",
            cuda_visible_device=settings.cuda_visible_device,
            timeout_seconds=config.limits.maximum_training_seconds,
            lineage_id=lineage_id,
        )
    if settings.modal is None:
        raise ValueError("Modal execution requires explicit modal settings")
    # This is explicit backend loading, never a fallback after an import failure.
    adapter = importlib.import_module("exp.optimize.claas.backends.modal.backend")
    return cast(
        ClaasTrainingBackend,
        adapter.ModalVerlBackend(
            config=settings.modal, checkpoint_root=directory / "checkpoints", lineage_id=lineage_id
        ),
    )


def save_execution_settings(directory: Path, settings: ExecutionSettings) -> None:
    """Persist a complete secret-free selected runtime before any compute call."""
    write_text_atomic(directory / "execution.json", settings.model_dump_json(indent=2) + "\n")


def load_execution_settings(directory: Path) -> ExecutionSettings:
    """Require explicit runtime binding rather than guessing a GPU or inference endpoint."""
    try:
        return ExecutionSettings.model_validate_json((directory / "execution.json").read_bytes())
    except FileNotFoundError:
        raise ValueError("runtime is not bound; run exp optimize claas bind first") from None
