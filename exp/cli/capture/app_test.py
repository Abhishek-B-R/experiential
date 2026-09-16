"""Capture exposes only foreground collection and independent network reset."""

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.capture import app as capture_module


def test_capture_without_subcommand_runs_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capture command invokes the foreground owner with default domains."""
    calls: list[tuple[str, ...]] = []

    def run_capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Record the requested domains without starting network interception."""
        calls.append(domains)

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", run_capture)
    result = CliRunner().invoke(app, ["capture"])
    assert result.exit_code == 0, result.output
    assert calls == [capture_module.DEFAULT_DOMAINS]


def test_reset_does_not_login_or_start_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline reset only invokes the networking recovery helper."""
    reset_calls: list[str] = []

    def unexpected_capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Fail if offline reset attempts to invoke the capture lifetime."""
        raise AssertionError("reset must never read a login or start capture")

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", unexpected_capture)
    monkeypatch.setattr(capture_module, "reset_capture_system", lambda: reset_calls.append("reset"))
    result = CliRunner().invoke(app, ["capture", "reset"])
    assert result.exit_code == 0, result.output
    assert reset_calls == ["reset"]
    assert "networking restored" in result.output


def test_domain_validation_precedes_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid domain arguments are rejected before authentication."""
    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    result = CliRunner().invoke(app, ["capture", "--domain", "https://example.com"])
    assert result.exit_code == 2
    assert "exact DNS hostname" in result.output


def test_no_background_management_commands() -> None:
    """Background management verbs are outside the public command surface."""
    runner = CliRunner()
    for command in ("status", "stop", "start"):
        result = runner.invoke(app, ["capture", command])
        assert result.exit_code == 2


def test_help_never_attempts_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading command help cannot trigger privileged setup."""

    def reject_setup() -> None:
        """Fail if help attempts to inspect or change system state."""
        raise AssertionError("help must not inspect or change system state")

    monkeypatch.setattr(capture_module, "_require_macos", reject_setup)
    result = CliRunner().invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    assert "reset" in result.output


def test_capture_requires_supported_python_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older SDK interpreter rejects capture before authentication or OS changes."""
    monkeypatch.setattr(capture_module.sys, "version_info", (3, 12, 0))
    with pytest.raises(ValueError, match="Capture requires Python 3.13"):
        capture_module._capture(Console(), domains=("api.openai.com",), root=Path("."))


def test_runner_replaces_foreground_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture keeps one foreground process and passes only non-secret CLI arguments."""
    launches: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(capture_module.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(
        capture_module.os, "execv", lambda executable, args: launches.append((executable, args))
    )
    capture_module._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
    assert launches == [
        (
            capture_module.sys.executable,
            [
                capture_module.sys.executable,
                "-m",
                "exp.cli.capture.runner",
                "--root",
                "/tmp/project",
                "--domain",
                "api.openai.com",
            ],
        )
    ]
