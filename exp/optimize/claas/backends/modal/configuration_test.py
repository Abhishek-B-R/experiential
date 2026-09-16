"""Modal configuration is usable without importing or installing the optional SDK."""

import subprocess
import sys

import pytest

from exp.optimize.claas.backends.modal.configuration import ModalExecutionConfig


def config() -> ModalExecutionConfig:
    """Return a finite authorization without constructing any SDK object."""
    return ModalExecutionConfig(
        app_name="test",
        volume_name="test",
        gpu="H100",
        timeout_seconds=60,
        startup_timeout_seconds=60,
        maximum_container_rate_usd_per_second=0.001,
        authorized_maximum_cost_usd=0.12,
    )


def test_cost_gate_rejects_before_backend_construction() -> None:
    """An underfunded budget and multi-GPU selection fail before any cloud SDK call."""
    assert config().estimated_maximum_cost_usd == 0.12
    with pytest.raises(ValueError, match="exceeds authorization"):
        ModalExecutionConfig.model_validate(
            config().model_dump() | {"authorized_maximum_cost_usd": 0.01}
        )
    with pytest.raises(ValueError):
        ModalExecutionConfig.model_validate(config().model_dump() | {"gpu": "H100:8"})


def test_configuration_import_does_not_load_optional_runtimes() -> None:
    """Fresh CLI-style imports neither require Modal nor load training frameworks."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from exp.optimize.claas.backends.modal.configuration import ModalExecutionConfig; "
            "assert 'modal' not in sys.modules; assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
