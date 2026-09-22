"""The foreground runner retains public CLI arguments and reports recoverable errors."""

import asyncio
import io
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from rich.console import Console

from exp.cli.capture import runner
from exp.cli.capture.auth import CaptureCredentials
from exp.runtime.capture.certificates import capture_certificate_directory
from exp.runtime.capture.control import CaptureRun, CaptureRunClient, Organization
from exp.runtime.capture.upload import CaptureUploader, UploadStats


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


def test_rejected_certificate_ends_cloud_run_without_reusing_legacy_ca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real runner uses a fresh scoped identity and preserves a proxy failure."""
    domains = ("chatgpt.com", "api.openai.com")
    data = tmp_path / "capture"
    legacy = data / "ca" / "mitmproxy-ca-cert.pem"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy CA must remain untouched")
    ca_directory = capture_certificate_directory(data, domains)
    organization = Organization(org_id=uuid4(), org_slug="test", org_name="Test")
    run = CaptureRun(
        id=uuid4(),
        org_id=organization.org_id,
        upload_origin="https://storage.example",
        upload_path_prefix="/storage/v1/object/upload/sign/capture/",
    )
    control = Mock(spec=CaptureRunClient)
    control.whoami.return_value = organization
    control.start.return_value = run
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    uploader.pending_current_run = 0
    trust = Mock()
    session = AsyncMock(side_effect=RuntimeError("Synthetic client certificate rejection"))
    monkeypatch.setattr(runner, "provider_data_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "CaptureRunClient", lambda **kwargs: control)
    monkeypatch.setattr(runner, "CaptureUploader", lambda **kwargs: uploader)
    monkeypatch.setattr(runner, "certificate_is_trusted", lambda *args, **kwargs: False)
    monkeypatch.setattr(runner, "trust_certificate", trust)
    monkeypatch.setattr(runner, "run_session", session)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="Synthetic client certificate rejection"):
        asyncio.run(
            runner._capture_authenticated(
                Console(file=output, width=200),
                domains=domains,
                credentials=CaptureCredentials(
                    "https://api.example.com", "https://example.com", "synthetic"
                ),
            )
        )
    certificate = ca_directory / "mitmproxy-ca-cert.pem"
    assert certificate.is_file()
    trust.assert_called_once_with(certificate, domains=domains)
    assert session.call_args.kwargs["ca_directory"] == ca_directory
    assert legacy.read_bytes() == b"legacy CA must remain untouched"
    assert "Public CA:" in output.getvalue()
    assert "Capture stopped. Interception disabled." not in output.getvalue()
    control.end.assert_awaited_once_with(run, pending_batches=0, upload_errors=0)
    control.close.assert_awaited_once()
