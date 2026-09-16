"""Bounded asynchronous capture delivery with credential-free durable retry files."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from exp.runtime.capture.normalization import CapturedExchange, normalize_exchange

logger = logging.getLogger(__name__)
_MAX_BATCH_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class UploadStats:
    """Content-free capture delivery counters for status and run heartbeats."""

    pending_batches: int
    upload_errors: int
    dropped_exchanges: int
    captured_exchanges: int
    uploaded_batches: int


class CaptureUploader:
    """Normalize copies off the inference path and retry signed cloud uploads."""

    def __init__(
        self,
        base_url: str,
        org_id: str,
        run_id: str,
        api_key: str,
        spool_dir: Path,
        *,
        max_body_bytes: int = 8 * 1024 * 1024,
        max_queue_bytes: int = 32 * 1024 * 1024,
        max_spool_bytes: int = 64 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Bind one organization and origin-scoped spool to authenticated delivery.

        Args:
            base_url: Validated Platform API origin.
            org_id: Authenticated organization ID.
            run_id: Client-generated UUID for this local capture run.
            api_key: Normal Platform API key, retained only in process memory.
            spool_dir: Private origin/org/run directory supplied by orchestration.
            max_body_bytes: Maximum decompressed bytes for either model body.
            max_queue_bytes: Maximum raw exchange bytes waiting for the worker.
            max_spool_bytes: Maximum sanitized files across runs in this spool's parent.
            transport: Optional HTTP transport for deterministic integration tests.
        """
        UUID(run_id)
        if min(max_body_bytes, max_queue_bytes, max_spool_bytes) < 1:
            raise ValueError("capture upload limits must be positive")
        if spool_dir.name != run_id:
            raise ValueError("capture spool directory must be named for its run UUID")
        self._base = f"{base_url.rstrip('/')}/api/orgs/{org_id}"
        self._api_key = api_key
        self._spool_dir = spool_dir
        self._max_body_bytes = max_body_bytes
        self._max_queue_bytes = max_queue_bytes
        self._max_spool_bytes = max_spool_bytes
        self._transport = transport
        self._queue: queue.Queue[CapturedExchange] = queue.Queue(maxsize=64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._delivery_thread: threading.Thread | None = None
        self._queued_bytes = 0
        self._captured = 0
        self._dropped = 0
        self._errors = 0
        self._uploaded = 0
        self._processing = 0
        self._retry_at: dict[Path, float] = {}

    def start(self) -> None:
        """Start one background worker without performing cloud requests inline."""
        if self._thread is not None:
            raise RuntimeError("capture uploader already started")
        for directory in (self._spool_dir, *self._spool_dir.parents):
            if directory.is_symlink():
                raise ValueError("capture spool cannot use symbolic links")
        self._spool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in (self._spool_dir, self._spool_dir.parent):
            if directory.stat().st_uid != os.getuid():
                raise ValueError("capture spool must belong to the current user")
            directory.chmod(0o700)
        self._thread = threading.Thread(target=self._work, name="exp-capture-upload", daemon=True)
        self._thread.start()
        self._delivery_thread = threading.Thread(
            target=self._deliver, name="exp-capture-delivery", daemon=True
        )
        self._delivery_thread.start()

    def submit(self, exchange: CapturedExchange) -> bool:
        """Accept a finite raw copy immediately, returning false on queue pressure."""
        with self._lock:
            if (
                self._stop.is_set()
                or self._queued_bytes + exchange.byte_count > self._max_queue_bytes
            ):
                self._dropped += 1
                return False
            try:
                self._queue.put_nowait(exchange)
            except queue.Full:
                self._dropped += 1
                return False
            self._queued_bytes += exchange.byte_count
            self._captured += 1
            return True

    @property
    def pending_current_run(self) -> int:
        """Count only this run for its heartbeat, excluding recovered sibling runs."""
        files = sum(path.parent == self._spool_dir for path in self._files())
        with self._lock:
            return self._queue.qsize() + self._processing + files

    @property
    def stats(self) -> UploadStats:
        """Read delivery counters without ever exposing model content or credentials."""
        pending_files = len(self._files())
        with self._lock:
            return UploadStats(
                pending_batches=self._queue.qsize() + self._processing + pending_files,
                upload_errors=self._errors,
                dropped_exchanges=self._dropped,
                captured_exchanges=self._captured,
                uploaded_batches=self._uploaded,
            )

    def close(self, timeout: float = 5.0) -> UploadStats:
        """Stop accepting copies and allow a bounded local drain before returning.

        Sanitized files survive cloud failure and are recovered by the next run.
        A daemon worker never delays application exit beyond the requested timeout.
        """
        self._stop.set()
        deadline = time.monotonic() + max(timeout, 0.0)
        if self._thread is not None:
            self._thread.join(max(deadline - time.monotonic(), 0.0))
        if self._delivery_thread is not None:
            self._delivery_thread.join(max(deadline - time.monotonic(), 0.0))
        return self.stats

    def _work(self) -> None:
        """Drain copied bodies to local storage independently of cloud availability."""
        while not self._stop.is_set() or not self._queue.empty():
            try:
                exchange = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._processing += 1
            try:
                self._persist(exchange)
            except (ValueError, OSError, RecursionError, TypeError):
                with self._lock:
                    self._dropped += 1
            finally:
                with self._lock:
                    self._queued_bytes -= exchange.byte_count
                    self._processing -= 1
                self._queue.task_done()

    def _deliver(self) -> None:
        """Keep slow cloud requests off both the inference and local persistence paths."""
        with httpx.Client(
            timeout=httpx.Timeout(5.0),
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            while not self._stop.wait(0.1):
                self._deliver_one(client)

    def _persist(self, exchange: CapturedExchange) -> None:
        """Write only sanitized OTLP, with finite per-origin/org spool capacity."""
        payload = normalize_exchange(exchange, max_body_bytes=self._max_body_bytes)
        if len(payload) > _MAX_BATCH_BYTES:
            raise ValueError("normalized capture exceeds the cloud batch limit")
        files = self._files()
        occupied = 0
        for path in files:
            try:
                occupied += path.stat().st_size
            except FileNotFoundError:
                # The independent delivery worker can finish a batch during accounting.
                continue
        if occupied + len(payload) > self._max_spool_bytes or len(files) >= 1024:
            raise ValueError("capture spool is full")
        destination = self._spool_dir / f"{uuid4()}.json"
        temporary = destination.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _files(self) -> list[Path]:
        """Find bounded retry files only in UUID-named sibling run directories."""
        result: list[Path] = []
        for directory in self._spool_dir.parent.glob("*"):
            if directory.is_symlink() or not directory.is_dir() or not _uuid(directory.name):
                continue
            for path in directory.glob("*.json"):
                if not path.is_symlink() and path.is_file() and _uuid(path.stem):
                    result.append(path)
                    if len(result) >= 1024:
                        return sorted(result)
        return sorted(result)

    def _deliver_one(self, client: httpx.Client) -> None:
        """Retry one eligible batch while never retaining an unbounded error history."""
        now = time.monotonic()
        files = self._files()
        self._retry_at = {
            path: deadline for path, deadline in self._retry_at.items() if path in files
        }
        for path in files:
            if self._retry_at.get(path, 0) > now:
                continue
            try:
                self._upload(client, path)
            except (httpx.HTTPError, ValueError, OSError, KeyError, TypeError):
                with self._lock:
                    self._errors += 1
                self._retry_at[path] = now + 10.0
                logger.warning("Capture upload deferred; sanitized batch remains queued locally")
            else:
                path.unlink(missing_ok=True)
                self._retry_at.pop(path, None)
                with self._lock:
                    self._uploaded += 1
                if path.parent != self._spool_dir:
                    self._finish_recovered_run(client, path.parent)
            return

    def _finish_recovered_run(self, client: httpx.Client, directory: Path) -> None:
        """Refresh an old run's pending count without reopening its capture lifetime."""
        pending = sum(path.parent == directory for path in self._files())
        try:
            response = client.post(
                f"{self._base}/capture/runs/{directory.name}/end",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"pending_batches": pending, "upload_errors": 0},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            with self._lock:
                self._errors += 1
            logger.warning("Recovered capture was uploaded; its run status could not be refreshed")

    def _upload(self, client: httpx.Client, path: Path) -> None:
        """Reserve idempotently, PUT without Platform credentials, then finalize."""
        if path.stat().st_size > self._max_spool_bytes:
            raise ValueError("capture retry file exceeds the spool limit")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        response = client.post(
            f"{self._base}/capture/runs/{path.parent.name}/batches/upload",
            headers=headers,
            json={"batch_id": path.stem, "source_kind": "otlp"},
        )
        response.raise_for_status()
        ticket = response.json()
        if not isinstance(ticket, dict):
            raise ValueError("capture upload response is not an object")
        if ticket.get("status") in {"running", "done"}:
            return
        if ticket.get("status") == "error":
            raise ValueError("capture batch failed cloud validation; retained locally")
        signed_url = ticket.get("signed_url")
        ingest_id = ticket.get("ingest_id")
        if not isinstance(signed_url, str) or not isinstance(ingest_id, str):
            raise ValueError("capture upload response lacks a ticket")
        parsed = httpx.URL(signed_url)
        if parsed.scheme != "https" or parsed.userinfo:
            raise ValueError("capture upload destination must be credential-free HTTPS")
        UUID(ingest_id)
        uploaded = client.put(
            signed_url,
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        if uploaded.status_code != 409:
            uploaded.raise_for_status()
        finalized = client.post(
            f"{self._base}/telemetry/traces/{ingest_id}/finalize",
            headers=headers,
        )
        finalized.raise_for_status()


def _uuid(value: str) -> bool:
    """Recognize canonical UUID filenames without allowing arbitrary path components."""
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
