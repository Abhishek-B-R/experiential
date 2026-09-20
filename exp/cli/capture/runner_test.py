"""The foreground runner retains public CLI arguments and reports recoverable errors."""

from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.capture import runner


def test_runner_receives_public_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process entrypoint retains the exact domain set and project root."""
    invocations: list[tuple[tuple[str, ...], Path]] = []

    def capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Record arguments without invoking credentials, trust, or networking."""
        invocations.append((domains, root))

    monkeypatch.setattr(runner, "_capture", capture)
    assert runner.main(["--root", "/tmp/project", "--domain", "api.openai.com"]) == 0
    assert invocations == [(("api.openai.com",), Path("/tmp/project"))]


def test_runner_reports_startup_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed foreground capture returns its actionable startup diagnostic."""

    def capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Simulate a content-free setup error before interception begins."""
        raise RuntimeError("Synthetic startup failure")

    monkeypatch.setattr(runner, "_capture", capture)
    assert runner.main(["--root", "/tmp/project", "--domain", "api.openai.com"]) == 1
    output = capsys.readouterr().out
    assert "Synthetic startup failure" in output
    assert "exp capture reset" not in output


def test_preflight_runs_before_login_or_cloud_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing backend prerequisites cannot open login or create a cloud run."""

    def unavailable() -> None:
        """Reject backend setup before authentication is allowed."""
        raise RuntimeError("Synthetic missing redirector")

    monkeypatch.setattr(runner, "require_local_backend", unavailable)
    with pytest.raises(RuntimeError, match="Synthetic missing redirector"):
        runner._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
