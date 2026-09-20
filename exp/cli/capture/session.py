"""Own foreground network interception, cloud heartbeats, and bounded shutdown."""

from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from exp.runtime.capture.control import CaptureCloudError, CaptureRun, CaptureRunClient
from exp.runtime.capture.proxy import CaptureProxy
from exp.runtime.capture.upload import CaptureUploader, UploadStats

_SHUTDOWN_TIMEOUT = 5.0


async def run_session(
    *,
    domains: tuple[str, ...],
    ca_directory: Path,
    uploader: CaptureUploader,
    control: CaptureRunClient,
    run: CaptureRun,
    on_started: Callable[[], None],
    on_progress: Callable[[UploadStats], None],
    on_warning: Callable[[str], None],
) -> UploadStats:
    """Serve provider traffic until interrupted, always releasing interception.

    Args:
        domains: Exact provider hostnames chosen for this capture run.
        ca_directory: Private directory holding the trusted capture CA.
        uploader: Nonblocking bounded trace queue and background uploader.
        control: Normal authenticated Platform capture-run client.
        run: Organization-bound run acknowledged before interception.
        on_started: Terminal callback once network interception is ready.
        on_progress: Terminal callback receiving content-free upload counters.
        on_warning: Terminal callback for recoverable upload failures.

    Returns:
        Final upload counters after bounded flushing.

    Raises:
        RuntimeError: The local network redirector fails to start or stops unexpectedly.
    """
    stop = asyncio.Event()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    proxy = CaptureProxy(sink=uploader.submit, domains=domains)
    original_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in original_signals:
        loop.add_signal_handler(sig, stop.set)
    proxy_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    uploader_started = False
    try:
        proxy_task = asyncio.create_task(proxy.serve(ca_directory=ca_directory, ready=ready.set))
        if not await _wait_for_proxy(proxy_task, ready, stop):
            return uploader.stats
        uploader.start()
        uploader_started = True
        if not stop.is_set():
            on_started()
        next_heartbeat = time.monotonic() + 15.0
        while not stop.is_set():
            if proxy_task.done():
                await proxy_task
                raise RuntimeError(
                    "Capture proxy stopped unexpectedly. Local interception is stopping."
                )
            stats = uploader.stats
            stats = replace(
                stats, dropped_exchanges=stats.dropped_exchanges + proxy.dropped_exchanges
            )
            on_progress(stats)
            if heartbeat_task is not None and heartbeat_task.done():
                try:
                    await heartbeat_task
                except CaptureCloudError:
                    on_warning("Platform is temporarily unavailable; capture uploads will retry.")
                heartbeat_task = None
            if heartbeat_task is None and time.monotonic() >= next_heartbeat:
                heartbeat_task = asyncio.create_task(
                    control.heartbeat(
                        run,
                        pending_batches=uploader.pending_current_run,
                        upload_errors=stats.upload_errors,
                    )
                )
                next_heartbeat = time.monotonic() + 15.0
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except TimeoutError:
                continue
    finally:
        proxy.shutdown()
        try:
            if proxy_task is not None:
                if not ready.is_set() and not proxy_task.done():
                    proxy_task.cancel()
                await _finish_proxy(proxy_task)
        finally:
            try:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    await asyncio.gather(heartbeat_task, return_exceptions=True)
                if uploader_started:
                    await asyncio.to_thread(uploader.close, timeout=5.0)
            finally:
                for sig, handler in original_signals.items():
                    loop.remove_signal_handler(sig)
                    signal.signal(sig, handler)
    stats = uploader.stats
    return replace(stats, dropped_exchanges=stats.dropped_exchanges + proxy.dropped_exchanges)


async def _wait_for_proxy(
    task: asyncio.Task[None], ready: asyncio.Event, stop: asyncio.Event
) -> bool:
    """Allow first-use macOS approval while keeping Ctrl+C immediately cancellable."""
    readiness = asyncio.create_task(ready.wait())
    interrupted = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait(
            {task, readiness, interrupted}, timeout=180.0, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
            raise RuntimeError("Capture proxy stopped before it was ready.")
        if interrupted in done:
            return False
        if readiness not in done:
            raise RuntimeError(
                "Capture is waiting for macOS network extension approval. Approve Mitmproxy "
                "Redirector in System Settings, then run exp capture again."
            )
        return True
    finally:
        readiness.cancel()
        interrupted.cancel()
        await asyncio.gather(readiness, interrupted, return_exceptions=True)


async def _finish_proxy(task: asyncio.Task[None]) -> None:
    """Require confirmed backend shutdown before reporting interception disabled."""
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=_SHUTDOWN_TIMEOUT)
    except TimeoutError as exc:
        task.cancel()
        raise RuntimeError(
            "Network extension shutdown could not be confirmed. Disable Mitmproxy Redirector "
            "in macOS System Settings before retrying Capture."
        ) from exc
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
