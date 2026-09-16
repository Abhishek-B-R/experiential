"""Provider-neutral runtime selection and privileged loopback boundaries."""

import sys
from pathlib import Path

import pytest

from exp.optimize.claas import execution
from exp.optimize.claas.backends.modal.configuration import ModalExecutionConfig
from exp.optimize.claas.backends.subprocess import SubprocessVerlBackend
from exp.optimize.claas.execution import (
    ExecutionSettings,
    create_training_backend,
    validate_worker_runtime,
)
from exp.optimize.claas.lifecycle.cycle_test import config


def settings() -> ExecutionSettings:
    """Return an explicit runtime without a world-model or provider dependency."""
    return ExecutionSettings(
        private_base_url="http://127.0.0.1:8000",
        worker_python=Path(sys.executable),
    )


@pytest.mark.parametrize(
    "url", ["http://remote:8000", "http://127.0.0.1:8000/v1", "http://u:p@localhost"]
)
def test_control_endpoint_is_a_credential_free_private_origin(url: str) -> None:
    """Privileged sleep/load requests cannot target an arbitrary host or malformed path."""
    with pytest.raises(ValueError, match="loopback"):
        ExecutionSettings.model_validate(settings().model_dump() | {"private_base_url": url})


def test_local_worker_requires_an_executable_regular_file(tmp_path: Path) -> None:
    """An absolute path alone is insufficient preflight evidence for a usable worker."""
    path = tmp_path / "python"
    selected = settings().model_copy(update={"worker_python": path})
    with pytest.raises(ValueError, match="existing executable interpreter"):
        validate_worker_runtime(selected, config())
    path.mkdir()
    with pytest.raises(ValueError, match="existing executable interpreter"):
        validate_worker_runtime(selected, config())
    path.rmdir()
    path.write_text("not executable")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="existing executable interpreter"):
        validate_worker_runtime(selected, config())
    validate_worker_runtime(settings(), config())
    validate_worker_runtime(selected, config().model_copy(update={"compute": "modal"}))


def test_local_selection_does_not_load_optional_modal_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local training path works when the Modal extra is absent."""

    def missing_plugin(name: str) -> None:
        """Fail if local selection attempts to load any optional plugin."""
        raise AssertionError(f"unexpected optional plugin: {name}")

    monkeypatch.setattr(execution.importlib, "import_module", missing_plugin)
    backend = create_training_backend(settings(), config(), tmp_path, "local")
    assert isinstance(backend, SubprocessVerlBackend)


def test_selected_modal_extra_fails_immediately_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable selected adapter fails without falling back to local compute."""
    requested: list[str] = []

    def missing_plugin(name: str) -> None:
        """Record the explicit module selection and emulate an absent SDK."""
        requested.append(name)
        raise ModuleNotFoundError("No module named 'modal'", name="modal")

    monkeypatch.setattr(execution.importlib, "import_module", missing_plugin)
    selected = settings().model_copy(
        update={
            "modal": ModalExecutionConfig(
                app_name="test-claas",
                volume_name="test-checkpoints",
                gpu="A100",
                maximum_container_rate_usd_per_second=0.001,
                authorized_maximum_cost_usd=3.0,
            )
        }
    )
    with pytest.raises(ModuleNotFoundError, match="modal"):
        create_training_backend(
            selected, config().model_copy(update={"compute": "modal"}), tmp_path, "remote"
        )
    assert requested == ["exp.optimize.claas.backends.modal.backend"]
