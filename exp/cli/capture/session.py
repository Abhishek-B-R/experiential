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
_STARTUP_TIMEOUT = 180.0
_WAITING_NOTICE_DELAY = 3.0


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
    on_waiting: Callable[[], None] | None = None,
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
        on_waiting: Optional callback when network startup remains pending.

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
        if not await _wait_for_proxy(proxy_task, ready, stop, on_waiting=on_waiting):
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
    task: asyncio.Task[None],
    ready: asyncio.Event,
    stop: asyncio.Event,
    on_waiting: Callable[[], None] | None = None,
) -> bool:
    """Report pending startup once without extending approval or cancellation bounds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STARTUP_TIMEOUT
    readiness = asyncio.create_task(ready.wait())
    interrupted = asyncio.create_task(stop.wait())
    notice = asyncio.create_task(asyncio.sleep(_WAITING_NOTICE_DELAY))
    waiting = {task, readiness, interrupted, notice}
    try:
        while True:
            done, _ = await asyncio.wait(
                waiting,
                timeout=max(0.0, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task.done():
                await task
                raise RuntimeError("Capture proxy stopped before it was ready.")
            if stop.is_set():
                return False
            if ready.is_set():
                return True
            if not done:
                raise RuntimeError(
                    "Capture could not start the macOS network extension. Enable Mitmproxy "
                    "Redirector in System Settings > General > Login Items & Extensions > "
                    "Network Extensions, then run exp capture again."
                )
            if notice in done:
                waiting.remove(notice)
                if on_waiting is not None:
                    on_waiting()
    finally:
        readiness.cancel()
        interrupted.cancel()
        notice.cancel()
        await asyncio.gather(readiness, interrupted, notice, return_exceptions=True)


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
