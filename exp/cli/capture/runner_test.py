"""The foreground runner retains public CLI arguments and reports recoverable errors."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.capture import runner
from exp.cli.capture.auth import CaptureCredentials


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


@pytest.mark.parametrize("fail_session", [False, True])
def test_session_ownership_covers_login_and_shutdown(
    monkeypatch: pytest.MonkeyPatch, fail_session: bool
) -> None:
    """The per-user lock prevents competing login or interception until teardown ends."""
    events: list[str] = []
    credentials = CaptureCredentials("https://api.example.com", "https://example.com", "test")

    @contextmanager
    def ownership() -> Iterator[None]:
        """Record the ownership boundary even when shutdown fails."""
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    def authenticate(
        *, console: Console, environment: Mapping[str, str], root: Path
    ) -> CaptureCredentials:
        """Return synthetic login material after the lock has been acquired."""
        assert events == ["preflight", "acquire"]
        events.append("login")
        return credentials

    async def session(
        console: Console, *, domains: tuple[str, ...], credentials: CaptureCredentials
    ) -> None:
        """Represent the complete asynchronous capture and native shutdown lifetime."""
        assert events == ["preflight", "acquire", "login"]
        events.append("shutdown")
        if fail_session:
            raise RuntimeError("Synthetic shutdown failure")

    monkeypatch.setattr(runner, "require_local_backend", lambda: events.append("preflight"))
    monkeypatch.setattr(runner, "capture_instance", ownership)
    monkeypatch.setattr(runner, "capture_credentials", authenticate)
    monkeypatch.setattr(runner, "_capture_authenticated", session)
    if fail_session:
        with pytest.raises(RuntimeError, match="Synthetic shutdown failure"):
            runner._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
    else:
        runner._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
    assert events == ["preflight", "acquire", "login", "shutdown", "release"]
