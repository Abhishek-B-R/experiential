"""Foreground failures and cancellation release interception before uploads."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from exp.cli.capture import session
from exp.runtime.capture.control import CaptureRun, CaptureRunClient
from exp.runtime.capture.proxy import CaptureProxy
from exp.runtime.capture.upload import CaptureUploader, UploadStats


@pytest.mark.parametrize("fail_before_ready", [False, True])
def test_proxy_failure_releases_interception_and_closes_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_before_ready: bool,
) -> None:
    """Inactive failures never announce capture; active failures drain uploads last."""
    events: list[str] = []
    proxy = Mock(spec=CaptureProxy)
    proxy.dropped_exchanges = 0
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    uploader.start.side_effect = lambda: events.append("upload-start")
    uploader.close.side_effect = lambda **kwargs: events.append("upload-close")
    proxy.shutdown.side_effect = lambda: events.append("proxy-stop")

    async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
        """Fail either before approval or after reporting interception is active."""
        try:
            if fail_before_ready:
                raise RuntimeError("synthetic approval failure")
            ready()
            await asyncio.sleep(0.01)
            raise RuntimeError("synthetic active proxy failure")
        finally:
            events.append("interception-disabled")

    proxy.serve.side_effect = serve
    monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
    with pytest.raises(RuntimeError, match="synthetic"):
        asyncio.run(
            session.run_session(
                domains=("api.openai.com",),
                ca_directory=tmp_path,
                uploader=uploader,
                control=Mock(spec=CaptureRunClient),
                run=CaptureRun(
                    id=uuid4(),
                    org_id=uuid4(),
                    upload_origin="https://storage.example",
                    upload_path_prefix="/storage/v1/object/upload/sign/capture/",
                ),
                on_started=lambda: events.append("active"),
                on_progress=lambda stats: None,
                on_warning=lambda message: None,
            )
        )
    assert "interception-disabled" in events
    if fail_before_ready:
        assert "active" not in events
        assert "upload-start" not in events
        assert "upload-close" not in events
    else:
        assert "active" in events
        assert events[-1] == "upload-close"
        assert events.index("interception-disabled") < events.index("upload-close")


def test_cancellation_does_not_wait_for_macos_approval() -> None:
    """Ctrl+C during pending first-use approval returns promptly to cleanup."""

    async def run() -> None:
        """Keep startup pending while delivering the foreground stop signal."""
        stop = asyncio.Event()
        ready = asyncio.Event()

        async def pending() -> None:
            """Represent a backend waiting for OS approval."""
            await asyncio.Event().wait()

        task = asyncio.create_task(pending())
        stop.set()
        try:
            assert not await asyncio.wait_for(session._wait_for_proxy(task, ready, stop), 0.5)
            assert not ready.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("hang", [False, True])
def test_shutdown_failure_cannot_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hang: bool
) -> None:
    """A failed or stalled redirector stop stays a failure after uploader cleanup."""

    async def run() -> None:
        """Stop a simulated active session and inspect cleanup error propagation."""
        closing = asyncio.Event()
        proxy = Mock(spec=CaptureProxy)
        proxy.dropped_exchanges = 0
        proxy.shutdown.side_effect = closing.set
        uploader = Mock(spec=CaptureUploader)
        uploader.stats = UploadStats(0, 0, 0, 0, 0)
        original_wait = session._wait_for_proxy

        async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
            """Raise or remain pending when the session requests shutdown."""
            ready()
            await closing.wait()
            if hang:
                await asyncio.Event().wait()
            raise RuntimeError("synthetic redirector stop failure")

        async def wait(task: asyncio.Task[None], ready: asyncio.Event, stop: asyncio.Event) -> bool:
            """Deliver the stop signal immediately after a successful startup."""
            result = await original_wait(task, ready, stop)
            asyncio.get_running_loop().call_soon(stop.set)
            return result

        proxy.serve.side_effect = serve
        monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
        monkeypatch.setattr(session, "_wait_for_proxy", wait)
        monkeypatch.setattr(session, "_SHUTDOWN_TIMEOUT", 0.02)
        with pytest.raises(RuntimeError, match="could not be confirmed|stop failure"):
            await session.run_session(
                domains=("api.openai.com",),
                ca_directory=tmp_path,
                uploader=uploader,
                control=Mock(spec=CaptureRunClient),
                run=CaptureRun(
                    id=uuid4(),
                    org_id=uuid4(),
                    upload_origin="https://storage.example",
                    upload_path_prefix="/storage/v1/object/upload/sign/capture/",
                ),
                on_started=lambda: None,
                on_progress=lambda stats: None,
                on_warning=lambda message: None,
            )
        uploader.close.assert_called_once()

    asyncio.run(run())
