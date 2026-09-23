"""Own a supervised native session and renew its lease from the serving event loop."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Protocol, cast

import mitmproxy_rs
from mitmproxy.proxy.mode_servers import LocalRedirectorInstance

from exp.runtime.capture.policy import validate_domains

_RENEW_INTERVAL = 2.0
_starting = False


class _SupervisedStarter(Protocol):
    """Describe the native Capture API required by the signed safety protocol."""

    def __call__(
        self,
        handle_tcp_stream: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        handle_udp_stream: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        *,
        capture_domains: list[str],
    ) -> Awaitable[mitmproxy_rs.local.LocalRedirector]:
        """Start an inactive redirector with native DNS bypass and independent supervision."""
        ...


@asynccontextmanager
async def capture_watchdog(domains: tuple[str, ...]) -> AsyncIterator[None]:
    """Prepare native safety before mitmproxy can enable any process selector.

    The signed supervisor survives this Python process. Native lease renewal must
    originate on the serving event loop, so an unresponsive loop cannot retain
    interception. No heartbeat work is added to individual requests.

    Args:
        domains: Selected provider hosts checked by the supervisor after shutdown.

    Yields:
        None while the caller owns the supervised native handle.

    Raises:
        RuntimeError: Another local redirector owns this process or startup races it.
    """
    global _starting
    domains = validate_domains(domains)
    cls = LocalRedirectorInstance
    if _starting or cls._server is not None or cls._instance is not None:
        raise RuntimeError("Another local redirector is active. Stop it before running Capture.")
    _starting = True
    native: mitmproxy_rs.local.LocalRedirector | None = None
    renewal: asyncio.Task[None] | None = None
    try:
        start = cast(_SupervisedStarter, mitmproxy_rs.local.start_local_redirector)
        native = await start(
            cls.redirector_handle_stream,
            cls.redirector_handle_stream,
            capture_domains=list(domains),
        )
        if cls._server is not None or cls._instance is not None:
            raise RuntimeError("A local redirector started during Capture setup. Retry Capture.")
        cls._server = native
        renewal = asyncio.create_task(_renew(native))
        yield
    finally:
        if renewal is not None:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
        # Normal cleanup releases the handle first. This also covers cancellation
        # between native startup and mitmproxy registering its local-mode instance.
        try:
            if native is not None:
                if cls._server is native:
                    cls._server = None
                    cls._instance = None
                native.close()
                await native.wait_closed()
        finally:
            _starting = False


async def _renew(native: mitmproxy_rs.local.LocalRedirector) -> None:
    """Renew only this handle's current selector, never resuming a stopped owner."""
    while LocalRedirectorInstance._server is native:
        owner = LocalRedirectorInstance._instance
        if owner is not None:
            data = owner.mode.data
            spec = f"{data},!{os.getpid()}" if data else f"!{os.getpid()}"
            try:
                native.set_intercept(spec)
            except (OSError, RuntimeError):
                # A closed native channel stops renewal. The independent native
                # lease expires without help from this process.
                return
        await asyncio.sleep(_RENEW_INTERVAL)
