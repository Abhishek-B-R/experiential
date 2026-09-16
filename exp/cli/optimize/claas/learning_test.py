"""Learning CLI exposes bounded cycles and rejects missing runtime before provider construction."""

from pathlib import Path

from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.optimize.claas.app_test import _initialize
from exp.common.claas import ClaasScope
from exp.optimize.claas.configuration import application_directory
from exp.optimize.claas.execution import ExecutionSettings, save_execution_settings


def test_training_requires_explicit_runtime_and_exposes_bounded_cycles(tmp_path: Path) -> None:
    """A new application cannot silently pick a GPU, cloud adapter, or private model server."""
    runner = CliRunner()
    assert runner.invoke(app, _initialize(tmp_path)).exit_code == 0
    result = runner.invoke(app, ["optimize", "claas", "train", "claims", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "runtime is not bound" in result.output
    help_result = runner.invoke(app, ["optimize", "claas", "train", "--help"])
    assert help_result.exit_code == 0 and "cycles" in help_result.output


def test_missing_worker_fails_before_catalog_or_provider_preflight(tmp_path: Path) -> None:
    """A known invalid local runtime cannot consume synthesis or practice reservations."""

    runner = CliRunner()
    assert runner.invoke(app, _initialize(tmp_path)).exit_code == 0
    directory = application_directory(
        tmp_path, ClaasScope(user_id="default", application_id="claims")
    )
    save_execution_settings(
        directory,
        ExecutionSettings(
            private_base_url="http://127.0.0.1:8000",
            worker_python=tmp_path / "missing-python",
        ),
    )
    # No catalog, capture configuration, gateway binding, or credentials exist.
    result = runner.invoke(app, ["optimize", "claas", "train", "claims", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "local worker_python" in result.output and "before training" in result.output
