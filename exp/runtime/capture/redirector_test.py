"""Exercise native redirector lifetimes with real mode instances and fake OS handles."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

import mitmproxy_rs
import pytest
from mitmproxy.addons.proxyserver import Proxyserver
from mitmproxy.proxy.mode_servers import LocalRedirectorInstance, RegularInstance

from exp.runtime.capture.redirector import stop_capture_servers


class FakeNative:
    """Record native operations without installing or activating any system component."""

    def __init__(self, *, fail_clear: bool = False) -> None:
        """Configure an optional interception-disable failure."""
        self.events: list[str] = []
        self.fail_clear = fail_clear

    def set_intercept(self, spec: str) -> None:
        """Record the exact specification sent by the real mitmproxy instance."""
        self.events.append(f"intercept:{spec}")
        if spec == "" and self.fail_clear:
            raise OSError("control channel disconnected")

    def close(self) -> None:
        """Record closing the native control channel."""
        self.events.append("close")

    async def wait_closed(self) -> None:
        """Record acknowledgement of native shutdown."""
        self.events.append("closed")


@pytest.fixture(autouse=True)
def isolated_native_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's native handle isolated from other mode-instance tests."""
    monkeypatch.setattr(LocalRedirectorInstance, "_instance", None)
    monkeypatch.setattr(LocalRedirectorInstance, "_server", None)


def _local(proxyserver: Proxyserver) -> LocalRedirectorInstance:
    """Register a real local-mode instance without starting the operating-system backend."""
    server = LocalRedirectorInstance.make("local", proxyserver)
    proxyserver.servers._instances[server.mode] = server
    return server


def _fake_start(monkeypatch: pytest.MonkeyPatch, native: FakeNative) -> None:
    """Replace only native startup; mitmproxy's real start and stop methods still execute."""

    async def start(
        tcp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        udp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
    ) -> FakeNative:
        """Supply an inert native redirector handle."""
        return native

    monkeypatch.setattr(mitmproxy_rs.local, "start_local_redirector", start)


def test_cancelled_startup_releases_incomplete_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation at macOS approval leaves no singleton blocking a future capture."""

    async def scenario() -> None:
        """Cancel the real local-mode startup while its fake native operation is pending."""
        entered = asyncio.Event()

        async def pending(
            tcp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
            udp: Callable[[mitmproxy_rs.Stream], Awaitable[None]],
        ) -> FakeNative:
            """Model the unbounded native approval wait without activating anything."""
            entered.set()
            await asyncio.Future[None]()
            raise AssertionError("pending startup unexpectedly completed")

        monkeypatch.setattr(mitmproxy_rs.local, "start_local_redirector", pending)
        proxyserver = Proxyserver()
        server = _local(proxyserver)
        startup = asyncio.create_task(server.start())
        await entered.wait()
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert LocalRedirectorInstance._instance is server
        assert LocalRedirectorInstance._server is None
        await stop_capture_servers(proxyserver)
        assert LocalRedirectorInstance._instance is None
        assert LocalRedirectorInstance._server is None

    asyncio.run(scenario())


def test_active_stop_disables_and_closes_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful foreground exit disables interception and confirms native closure."""
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Drive the actual mitmproxy local instance through start and complete shutdown."""
        proxyserver = Proxyserver()
        server = _local(proxyserver)
        await server.start()
        await stop_capture_servers(proxyserver)
        assert LocalRedirectorInstance._instance is None
        assert LocalRedirectorInstance._server is None
        await stop_capture_servers(proxyserver)

    asyncio.run(scenario())
    assert native.events == [f"intercept:!{os.getpid()}", "intercept:", "close", "closed"]


def test_failed_disable_still_closes_native_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken interception-control command cannot bypass native handle cleanup."""
    native = FakeNative(fail_clear=True)
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Surface the stop failure only after closing and releasing the native singleton."""
        proxyserver = Proxyserver()
        server = _local(proxyserver)
        await server.start()
        with pytest.raises(RuntimeError, match="could not confirm network interception shutdown"):
            await stop_capture_servers(proxyserver)
        assert LocalRedirectorInstance._instance is None
        assert LocalRedirectorInstance._server is None

    asyncio.run(scenario())
    assert native.events[-3:] == ["intercept:", "close", "closed"]


def test_different_owner_is_not_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale manager must not disable the native handle owned by another capture."""
    native = FakeNative()
    _fake_start(monkeypatch, native)

    async def scenario() -> None:
        """Keep the active owner's singleton and specification unchanged."""
        stale_manager = Proxyserver()
        _local(stale_manager)
        active_manager = Proxyserver()
        active = _local(active_manager)
        await active.start()
        await stop_capture_servers(stale_manager)
        assert LocalRedirectorInstance._instance is active
        assert native.events == [f"intercept:!{os.getpid()}"]
        await stop_capture_servers(active_manager)

    asyncio.run(scenario())


def test_regular_server_is_stopped_only_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test transport follows its ordinary lifecycle without local-mode state access."""
    stopped: list[RegularInstance] = []

    async def stop(server: RegularInstance) -> None:
        """Observe regular-server shutdown without opening a socket."""
        stopped.append(server)

    async def scenario() -> None:
        """Invoke stop only for the regular instance that reports an active listener."""
        manager = Proxyserver()
        server = RegularInstance.make("regular", manager)
        manager.servers._instances[server.mode] = server
        monkeypatch.setattr(RegularInstance, "is_running", property(lambda self: False))
        monkeypatch.setattr(RegularInstance, "stop", stop)
        await stop_capture_servers(manager)
        assert stopped == []
        monkeypatch.setattr(RegularInstance, "is_running", property(lambda self: True))
        await stop_capture_servers(manager)
        assert stopped == [server]

    asyncio.run(scenario())


def test_new_owner_during_old_native_close_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Awaiting old native shutdown must not clear a later capture's singleton."""

    async def scenario() -> None:
        """Start a distinct capture while the old native close acknowledgement is pending."""
        closing = asyncio.Event()
        release = asyncio.Event()

        class WaitingNative(FakeNative):
            """Delay only closure acknowledgement, never operating-system cleanup."""

            async def wait_closed(self) -> None:
                """Allow another owner to start before this shutdown finishes."""
                closing.set()
                await release.wait()
                await super().wait_closed()

        previous = WaitingNative()
        _fake_start(monkeypatch, previous)
        previous_manager = Proxyserver()
        await _local(previous_manager).start()
        cleanup = asyncio.create_task(stop_capture_servers(previous_manager))
        await closing.wait()
        current = FakeNative()
        _fake_start(monkeypatch, current)
        current_manager = Proxyserver()
        current_server = _local(current_manager)
        await current_server.start()
        release.set()
        await cleanup
        assert LocalRedirectorInstance._instance is current_server
        assert current.events == [f"intercept:!{os.getpid()}"]
        await stop_capture_servers(current_manager)

    asyncio.run(scenario())


def test_one_failure_does_not_skip_other_owned_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    """All owned shutdown operations are attempted before reporting a local-mode failure."""
    native = FakeNative(fail_clear=True)
    _fake_start(monkeypatch, native)
    stopped: list[RegularInstance] = []

    async def stop(server: RegularInstance) -> None:
        """Record regular transport cleanup after the local transport failed."""
        stopped.append(server)

    async def scenario() -> None:
        """Put the failing local server first to exercise continued cleanup."""
        manager = Proxyserver()
        await _local(manager).start()
        regular = RegularInstance.make("regular", manager)
        manager.servers._instances[regular.mode] = regular
        monkeypatch.setattr(RegularInstance, "is_running", property(lambda self: True))
        monkeypatch.setattr(RegularInstance, "stop", stop)
        with pytest.raises(RuntimeError, match="could not confirm network interception shutdown"):
            await stop_capture_servers(manager)
        assert stopped == [regular]

    asyncio.run(scenario())
