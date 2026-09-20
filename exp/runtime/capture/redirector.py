"""Close Capture's mitmproxy servers, including an interrupted local-mode startup."""

from __future__ import annotations

from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy.mode_servers import LocalRedirectorInstance


async def stop_capture_servers(proxyserver: Proxyserver) -> None:
    """Stop each owned server and report cleanup failures after attempting all of them.

    Args:
        proxyserver: The server manager belonging to this foreground Capture process.

    Raises:
        RuntimeError: A server could not confirm shutdown.
    """
    errors: list[Exception] = []
    for server in tuple(proxyserver.servers):
        try:
            if isinstance(server, LocalRedirectorInstance):
                await _stop_local_redirector(server)
            elif server.is_running:
                await server.stop()
        except Exception as exc:  # noqa: BLE001 - Attempt all owned cleanup before reporting failure.
            errors.append(exc)
    if errors:
        raise RuntimeError(
            "Capture could not confirm network interception shutdown. Disable Mitmproxy "
            "Redirector in macOS System Settings if connections fail."
        ) from errors[0]


async def _stop_local_redirector(server: LocalRedirectorInstance) -> None:
    """Release only this instance's native redirector, including partial initialization.

    Mitmproxy 12's local mode stores its native handle and owner in class-level
    fields. Startup cancellation can leave an owner without a native handle, and
    ordinary stop deliberately retains the native daemon. This is the single
    boundary that accesses those internals so foreground Capture can fully close
    its redirector without changing an unrelated owner's state.
    """
    cls = type(server)
    if cls._instance is not server:
        return
    native = cls._server
    if native is None:
        cls._instance = None
        return
    try:
        await server.stop()
    finally:
        # An awaited stop must never confer authority over a replacement owner.
        if cls._server is native and (cls._instance is None or cls._instance is server):
            native.close()
            cls._instance = None
            cls._server = None
            # Clear ownership before awaiting closure so a new capture can own a
            # distinct handle without this cleanup clearing it afterward.
            await native.wait_closed()
