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

_UPLOAD_ORIGIN = "https://storage.example"
_UPLOAD_PREFIX = "/storage/v1/object/upload/sign/artifacts/orgs/org/telemetry-traces/otlp/"
_UPLOAD_TEMPLATE = f"{_UPLOAD_ORIGIN}{_UPLOAD_PREFIX}{{ingest}}/{{nonce}}?token=signed"


def _signed_url(ingest: str) -> str:
    """Return a synthetic ticket within the run's acknowledged storage scope."""
    return f"{_UPLOAD_ORIGIN}{_UPLOAD_PREFIX}{ingest}/{'a' * 43}?token=signed"


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
                    "signed_url": _signed_url(ingest),
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
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
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
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
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
        "https://api.example",
        "org",
        run,
        "KEY",
        tmp_path / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
        max_queue_bytes=1,
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
    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "KEY",
        linked / run,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    with pytest.raises(ValueError, match="symbolic"):
        uploader.start()


@pytest.mark.parametrize(
    "destination_template",
    [
        _UPLOAD_TEMPLATE.replace("storage.example", "unapproved.example"),
        _UPLOAD_TEMPLATE.replace("storage.example", "storage.example.attacker.example"),
        _UPLOAD_TEMPLATE.replace("storage.example", "storage.example:8443"),
        _UPLOAD_TEMPLATE.replace("https://", "http://"),
        _UPLOAD_TEMPLATE.replace("https://", "https://user:password@"),
        _UPLOAD_TEMPLATE.replace("/orgs/org/", "/orgs/other/"),
        _UPLOAD_TEMPLATE.replace("{ingest}", "00000000-0000-0000-0000-000000000000"),
        _UPLOAD_TEMPLATE.replace("/{nonce}", "/unexpected/{nonce}"),
        _UPLOAD_TEMPLATE.replace("/{nonce}", "/%2e%2e/{nonce}"),
        _UPLOAD_TEMPLATE + "#fragment",
        "https://[malformed",
    ],
)
def test_misrouted_signed_ticket_never_sends_capture_bytes(
    tmp_path: Path, destination_template: str
) -> None:
    """Reject an unapproved origin or object path before making any signed PUT."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    path.write_bytes(b'{"synthetic":"capture-content-canary"}')
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an inconsistent ticket from the otherwise authenticated control API."""
        requests.append(request)
        assert request.method == "POST"
        assert request.url.host == "api.example"
        assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
        assert b"capture-content-canary" not in request.content
        return httpx.Response(
            200,
            json={
                "status": "pending",
                "ingest_id": ingest,
                "signed_url": destination_template.format(ingest=ingest, nonce="a" * 43),
            },
        )

    uploader = CaptureUploader(
        "https://api.example",
        "org",
        run,
        "PLATFORM-KEY",
        directory,
        upload_origin=_UPLOAD_ORIGIN,
        upload_path_prefix=_UPLOAD_PREFIX,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="capture upload destination"):
            uploader._upload(client, path)
    assert len(requests) == 1
    assert path.exists()


@pytest.mark.parametrize(
    ("api_origin", "upload_origin", "path_prefix"),
    [
        ("https://api.example", "https://storage.example:443", _UPLOAD_PREFIX),
        ("https://preview.example", "https://storage.example:8443", "/proxy" + _UPLOAD_PREFIX),
        (
            "https://preview.example",
            "https://storage.example",
            _UPLOAD_PREFIX.replace("artifacts", "artifact%20bucket"),
        ),
        ("http://127.0.0.1:8000", "http://localhost:55421", _UPLOAD_PREFIX),
        ("http://[::1]:8000", "http://[::1]:55421", _UPLOAD_PREFIX),
    ],
)
def test_approved_storage_scope_supports_preview_and_explicit_local_development(
    tmp_path: Path, api_origin: str, upload_origin: str, path_prefix: str
) -> None:
    """Respect explicit deployment origins without sending Platform credentials to Storage."""
    run, ingest = str(uuid4()), str(uuid4())
    directory = tmp_path / run
    directory.mkdir()
    path = directory / f"{uuid4()}.json"
    content = b'{"synthetic":"capture-content-canary"}'
    path.write_bytes(content)
    requests: list[httpx.Request] = []
    signed_url = f"{upload_origin}{path_prefix}{ingest}/{'a' * 43}?token=signed"

    def handler(request: httpx.Request) -> httpx.Response:
        """Accept the approved signed destination and the separate authenticated finalize."""
        requests.append(request)
        if request.method == "PUT":
            assert request.url == httpx.URL(signed_url)
            assert request.content == content
            assert "authorization" not in request.headers
            return httpx.Response(200)
        assert request.headers["authorization"] == "Bearer PLATFORM-KEY"
        if request.url.path.endswith("/batches/upload"):
            return httpx.Response(
                200, json={"status": "pending", "ingest_id": ingest, "signed_url": signed_url}
            )
        assert request.url.path.endswith(f"/{ingest}/finalize")
        return httpx.Response(202)

    uploader = CaptureUploader(
        api_origin,
        "org",
        run,
        "PLATFORM-KEY",
        directory,
        upload_origin=upload_origin,
        upload_path_prefix=path_prefix,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        uploader._upload(client, path)
    assert [request.method for request in requests] == ["POST", "PUT", "POST"]


@pytest.mark.parametrize(
    ("api_origin", "upload_origin", "path_prefix"),
    [
        ("https://api.example", "http://localhost:55421", _UPLOAD_PREFIX),
        ("http://127.0.0.1:8000", "http://storage.example", _UPLOAD_PREFIX),
        ("http://api.example", "http://localhost:55421", _UPLOAD_PREFIX),
        ("https://api.example", "https://user:password@storage.example", _UPLOAD_PREFIX),
        ("https://api.example", "https://storage.example?query=true", _UPLOAD_PREFIX),
        ("https://api.example", "https://storage.example/unexpected", _UPLOAD_PREFIX),
        (
            "https://api.example",
            _UPLOAD_ORIGIN,
            _UPLOAD_PREFIX.replace("/orgs/org/", "/orgs/other/"),
        ),
        ("https://api.example", _UPLOAD_ORIGIN, "/../" + _UPLOAD_PREFIX),
        ("https://api.example", _UPLOAD_ORIGIN, _UPLOAD_PREFIX + "?query=true"),
    ],
)
def test_invalid_run_storage_policy_is_rejected_before_uploading(
    tmp_path: Path, api_origin: str, upload_origin: str, path_prefix: str
) -> None:
    """Reject credential-bearing, cross-organization, or unapproved cleartext policies."""
    run = str(uuid4())
    with pytest.raises(ValueError, match="capture run storage"):
        CaptureUploader(
            api_origin,
            "org",
            run,
            "PLATFORM-KEY",
            tmp_path / run,
            upload_origin=upload_origin,
            upload_path_prefix=path_prefix,
        )
