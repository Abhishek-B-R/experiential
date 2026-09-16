"""Capture delivery retries safely while keeping cloud failures off inference."""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from exp.runtime.capture.normalization import CapturedExchange, normalize_exchange
from exp.runtime.capture.upload import CaptureUploader


def _exchange() -> CapturedExchange:
    """Return a bounded synthetic request and response for this test."""
    return CapturedExchange(
        protocol="responses",
        host="api.openai.com",
        path="/v1/responses",
        started_ns=1,
        ended_ns=2,
        request=b'{"model":"test","input":"hello","api_key":"secret"}',
        response=b'{"output":[]}',
        status=200,
    )


def _wait(predicate: Callable[[], bool]) -> None:
    """Wait briefly for the bounded background delivery worker to make progress."""
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_recovery_keeps_original_run_batch_and_never_sends_api_key_to_storage(
    tmp_path: Path,
) -> None:
    """Recover an existing batch with stable identity and separate storage credentials."""
    prior_run, run, batch, ingest = (str(uuid4()) for _ in range(4))
    prior = tmp_path / prior_run
    prior.mkdir()
    (prior / f"{batch}.json").write_bytes(normalize_exchange(_exchange(), max_body_bytes=4096))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Emulate the cloud response needed by this delivery scenario."""
        requests.append(request)
        if request.url.host == "storage.example":
            assert "authorization" not in request.headers
            assert b"secret" not in request.content
            return httpx.Response(409)
        assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
        if request.url.path.endswith(f"/{prior_run}/end"):
            assert json.loads(request.content) == {"pending_batches": 0, "upload_errors": 0}
            return httpx.Response(200)
        if request.url.path.endswith("/batches/upload"):
            assert prior_run in request.url.path
            assert json.loads(request.content)["batch_id"] == batch
            return httpx.Response(
                200,
                json={
                    "status": "pending",
                    "signed_url": "https://storage.example/file?token=signed",
                    "ingest_id": ingest,
                },
            )
        assert request.url.path.endswith(f"/{ingest}/finalize")
        return httpx.Response(202)

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "PLATFORM-KEY",
        tmp_path / run,
        transport=httpx.MockTransport(handler),
    )
    uploader.start()
    try:
        assert uploader.pending_current_run == 0
        _wait(lambda: uploader.stats.uploaded_batches == 1)
    finally:
        uploader.close()
    assert len(requests) == 4
    assert not (prior / f"{batch}.json").exists()


def test_slow_cloud_does_not_block_submission_or_local_shutdown_flush(tmp_path: Path) -> None:
    """Keep inference enqueue and local persistence responsive during a cloud stall."""
    entered, release = threading.Event(), threading.Event()
    run = str(uuid4())

    def handler(request: httpx.Request) -> httpx.Response:
        """Emulate the cloud response needed by this delivery scenario."""
        entered.set()
        release.wait(3)
        return httpx.Response(503)

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        transport=httpx.MockTransport(handler),
    )
    uploader.start()
    try:
        assert uploader.submit(_exchange())
        assert entered.wait(2)
        before = time.monotonic()
        assert uploader.submit(_exchange())
        assert time.monotonic() - before < 0.1
        uploader.close(timeout=0.3)
        assert len(list((tmp_path / run).glob("*.json"))) == 2
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in (tmp_path / run).glob("*.json"))
    finally:
        release.set()
        uploader.close()


def test_finite_raw_queue_reports_backpressure_without_starting_worker(tmp_path: Path) -> None:
    """Reject queue overflow immediately and expose a dropped-capture counter."""
    run = str(uuid4())
    uploader = CaptureUploader(
        "https://api.example", "org", run, "KEY", tmp_path / run, max_queue_bytes=1
    )
    assert not uploader.submit(_exchange())
    assert uploader.stats.dropped_exchanges == 1


def test_spool_rejects_symbolic_link_ancestors(tmp_path: Path) -> None:
    """Reject spool paths that could follow a symbolic link into another directory."""
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target)
    run = str(uuid4())
    uploader = CaptureUploader("https://api.example", "org", run, "KEY", linked / run)
    with pytest.raises(ValueError, match="symbolic"):
        uploader.start()
