"""Exercise supervised startup and lease ownership without enabling OS interception."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import cast

import mitmproxy_rs
import pytest
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy.mode_servers import LocalRedirectorInstance

from exp.runtime.capture import watchdog
from exp.runtime.capture.redirector import stop_capture_servers


class Native:
    """Model a native session with observable lease renewals and confirmed shutdown."""

    def __init__(self) -> None:
        """Start inactive, exactly like the OS backend before a selector arrives."""
        self.specs: list[str] = []
        self.closed = False
        self.waited = False
        self.fail_close = False

    def set_intercept(self, spec: str) -> None:
        """Reject renewal after shutdown and otherwise retain only process selectors."""
        if self.closed:
            raise OSError("closed")
        self.specs.append(spec)

    def close(self) -> None:
        """Close idempotently, as required by the native handle's contract."""
        self.closed = True

    async def wait_closed(self) -> None:
        """Report cleanup failure without keeping interception alive in this fixture."""
        self.waited = True
        if self.fail_close:
            raise RuntimeError("native shutdown was not confirmed")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use isolated singleton state and short test-only heartbeat intervals."""
    monkeypatch.setattr(LocalRedirectorInstance, "_server", None)
    monkeypatch.setattr(LocalRedirectorInstance, "_instance", None)
    monkeypatch.setattr(watchdog, "_starting", False)
    monkeypatch.setattr(watchdog, "_RENEW_INTERVAL", 0.001)


def install_native(monkeypatch: pytest.MonkeyPatch, native: Native) -> list[list[str]]:
    """Supply the native safety API while recording the required startup policy."""
    requested: list[list[str]] = []

    async def start(
        tcp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        udp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        *,
        capture_domains: list[str],
    ) -> Native:
        """Prove Capture explicitly requests safety before any selector is sent."""
        assert native.specs == []
        requested.append(capture_domains)
        return native

    monkeypatch.setattr(mitmproxy_rs.local, "start_local_redirector", start)
    return requested


def test_safety_precedes_interception_and_renewal_stops_with_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive real mitmproxy start/stop against an explicitly supervised native handle."""
    native = Native()
    requested = install_native(monkeypatch, native)

    async def scenario() -> None:
        """Allow several renewals, then ensure cleanup never re-enables interception."""
        manager = Proxyserver()
        server = LocalRedirectorInstance.make("local:!mDNSResponder", manager)
        manager.servers._instances[server.mode] = server
        async with watchdog.capture_watchdog(("chatgpt.com", "api.openai.com")):
            assert requested == [["api.openai.com", "chatgpt.com"]]
            assert native.specs == []
            await server.start()
            await asyncio.sleep(0.015)
            assert len(native.specs) > 1
            assert set(native.specs) == {f"!mDNSResponder,!{os.getpid()}"}
            await stop_capture_servers(manager)
            stopped = list(native.specs)
            await asyncio.sleep(0.005)
            assert native.specs == stopped
        assert native.closed and native.waited
        assert LocalRedirectorInstance._server is None
        assert not watchdog._starting

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_exit_before_mode_registration_still_closes_native(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    """Partial setup cannot leave an unowned native supervisor running."""
    native = Native()
    install_native(monkeypatch, native)

    async def scenario() -> None:
        """Exit before mitmproxy assigns its mode-instance owner."""
        async with watchdog.capture_watchdog(("chatgpt.com",)):
            if failure:
                raise ValueError("setup failed")

    if failure:
        with pytest.raises(ValueError, match="setup failed"):
            asyncio.run(scenario())
    else:
        asyncio.run(scenario())
    assert native.closed and native.waited
    assert not native.specs
    assert LocalRedirectorInstance._server is None
    assert not watchdog._starting


def test_shutdown_failure_releases_startup_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfirmed shutdown stays visible without poisoning the process-level guard."""
    native = Native()
    native.fail_close = True
    install_native(monkeypatch, native)

    async def scenario() -> None:
        """Surface the native error to the command's existing recovery diagnostic."""
        async with watchdog.capture_watchdog(("chatgpt.com",)):
            pass

    with pytest.raises(RuntimeError, match="shutdown was not confirmed"):
        asyncio.run(scenario())
    assert not watchdog._starting


def test_existing_owner_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture cannot acquire or stop an unrelated local redirector."""
    native = Native()
    requested = install_native(monkeypatch, Native())
    monkeypatch.setattr(LocalRedirectorInstance, "_server", native)

    async def scenario() -> None:
        """Reject ownership before calling any native method."""
        async with watchdog.capture_watchdog(("chatgpt.com",)):
            raise AssertionError("must not acquire an existing redirector")

    with pytest.raises(RuntimeError, match="Another local redirector"):
        asyncio.run(scenario())
    assert not native.closed and requested == []


def test_cancelled_startup_releases_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation while waiting for the signed app does not retain Python ownership."""

    async def scenario() -> None:
        """Cancel a pending native startup without ever enabling interception."""
        entered = asyncio.Event()

        async def start(
            tcp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
            udp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
            *,
            capture_domains: list[str],
        ) -> Native:
            """Model a native approval wait with no initialized handle."""
            entered.set()
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

        async def run() -> None:
            """Attempt to enter the supervised scope."""
            async with watchdog.capture_watchdog(("chatgpt.com",)):
                raise AssertionError("startup must stay pending")

        monkeypatch.setattr(mitmproxy_rs.local, "start_local_redirector", start)
        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not watchdog._starting
        assert LocalRedirectorInstance._server is None

    asyncio.run(scenario())


def test_renewal_does_not_touch_replacement_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A renewal task loses authority immediately when its native handle is replaced."""
    previous, current = Native(), Native()
    monkeypatch.setattr(LocalRedirectorInstance, "_server", current)
    asyncio.run(watchdog._renew(cast(mitmproxy_rs.local.LocalRedirector, previous)))
    assert not previous.specs and not current.specs
