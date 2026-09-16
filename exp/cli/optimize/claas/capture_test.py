"""Local capture setup binds the same authenticated user as the native gateway."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.optimize.claas.app_test import _initialize
from exp.runtime.claas.capture import load_capture_configuration
from exp.runtime.gateway.tests.launch_test import _configure_gateway


def test_capture_cli_persists_explicit_scope_and_can_disable_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist and disable content capture for the authenticated application."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    manager, key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:12345/v1")
    _, user = manager.store().authenticated_identity(raw_key=key)
    runner = CliRunner()
    result = runner.invoke(app, [*_initialize(tmp_path), "--user", user])
    assert result.exit_code == 0, result.output
    arguments = [
        "optimize",
        "claas",
        "capture",
        "claims",
        "--alias",
        "coding",
        "--user",
        user,
        "--root",
        str(tmp_path),
    ]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "restart the gateway" in result.output
    config = load_capture_configuration(tmp_path)
    assert config is not None
    assert config.bindings[0].policy.scope.user_id == user
    assert config.bindings[0].policy.enabled
    assert not config.database_path.exists()
    assert key not in (tmp_path / "gateway" / "claas.json").read_text()
    result = runner.invoke(app, [*arguments, "--disable"])
    assert result.exit_code == 0, result.output
    config = load_capture_configuration(tmp_path)
    assert config is not None and not config.bindings[0].policy.enabled


def test_capture_requires_an_existing_key_identity_and_alias_grant(tmp_path: Path) -> None:
    """Reject capture setup without an existing authorized gateway identity."""
    runner = CliRunner()
    assert runner.invoke(app, _initialize(tmp_path)).exit_code == 0
    result = runner.invoke(
        app,
        ["optimize", "claas", "capture", "claims", "--alias", "coding", "--root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "active authenticated gateway identity" in result.output
    assert load_capture_configuration(tmp_path) is None
