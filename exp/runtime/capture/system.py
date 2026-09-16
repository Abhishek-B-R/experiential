"""Unprivileged supervision for the temporary macOS capture system helper.

Call ``CaptureSystemSession.start(port, domains)``, send ``heartbeat()`` at least
every five seconds, and always ``close()``. Recovery through ``reset_capture_system``
requires neither an Experiential login nor network access.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

from exp.runtime.capture.system_helper import CaptureSystemError, validate_domains, validate_port

_HELPER_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C"}
_READY_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 25.0


def _authorize() -> None:
    """Request macOS administrator authorization without elevating the foreground CLI."""
    if sys.platform != "darwin":
        raise CaptureSystemError("DNS capture currently supports macOS only.")
    if os.geteuid() == 0:
        raise CaptureSystemError("Run exp capture as your normal user, not with sudo.")
    result = subprocess.run(["/usr/bin/sudo", "-v"], env=_HELPER_ENV, check=False)
    if result.returncode:
        raise CaptureSystemError(
            "Administrator authorization was not granted; capture was not enabled."
        )


def _command(action: str, *, port: int = 0, domains: tuple[str, ...] = ()) -> list[str]:
    """Construct a fixed helper command with isolated imports and no configurable paths."""
    command = [
        "/usr/bin/sudo",
        "-n",
        str(Path(sys.executable).resolve()),
        "-I",
        "-S",
        str(Path(__file__).with_name("system_helper.py").resolve()),
        action,
    ]
    if action == "serve":
        command.extend(("--port", str(validate_port(port))))
        for domain in validate_domains(domains):
            command.extend(("--domain", domain))
    return command


def _read_event(process: subprocess.Popen[bytes], timeout: float) -> tuple[str, str]:
    """Read one bounded JSON line without hiding startup failure or waiting indefinitely."""
    if process.stdout is None:
        raise CaptureSystemError("Capture helper output is unavailable.")
    deadline = time.monotonic() + timeout
    content = bytearray()
    while b"\n" not in content:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([process.stdout], [], [], max(0, remaining))[0]:
            raise CaptureSystemError(
                "Capture helper timed out. Run exp capture reset before retrying."
            )
        chunk = os.read(process.stdout.fileno(), 1)
        if not chunk:
            raise CaptureSystemError(
                "Capture helper exited before confirming cleanup. Run exp capture reset."
            )
        content.extend(chunk)
        if len(content) > 16384:
            raise CaptureSystemError(
                "Capture helper returned invalid output. Run exp capture reset."
            )
    try:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("not an event")
        event, detail = value.get("event"), value.get("detail", "")
        if not isinstance(event, str) or not isinstance(detail, str):
            raise ValueError("invalid event")
    except (ValueError, UnicodeError) as exc:
        raise CaptureSystemError(
            "Capture helper returned invalid output. Run exp capture reset."
        ) from exc
    if event == "error":
        raise CaptureSystemError(detail)
    return event, detail


class CaptureSystemSession:
    """A foreground-owned lease on temporary system routing, supervised across CLI crashes."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        """Store only the helper process, never provider or Experiential credentials."""
        self.process = process
        self._closed = False

    @classmethod
    def start(cls, upstream_port: int, domains: tuple[str, ...]) -> CaptureSystemSession:
        """Authorize and start routing only after the unprivileged proxy is ready.

        Args:
            upstream_port: Listening mitmproxy TCP port on IPv4 loopback.
            domains: Explicit lowercase provider DNS names, never wildcards.

        Returns:
            A session that requires a heartbeat at least every five seconds.
        """
        command = _command("serve", port=upstream_port, domains=domains)
        _authorize()
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=_HELPER_ENV,
            bufsize=0,
        )
        session = cls(process)
        try:
            event, _ = _read_event(process, _READY_TIMEOUT)
            if event != "ready":
                raise CaptureSystemError("Capture helper did not confirm readiness.")
            session.heartbeat()
            return session
        except (CaptureSystemError, OSError):
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=_CLOSE_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                raise CaptureSystemError(
                    "Capture startup failed and cleanup is unconfirmed. Run exp capture reset."
                ) from exc
            finally:
                if process.stdout is not None:
                    process.stdout.close()
            raise

    def heartbeat(self) -> None:
        """Renew the routing lease; callers must stop capture if the helper has failed."""
        if self._closed or self.process.poll() is not None or self.process.stdin is None:
            raise CaptureSystemError(
                "Capture helper stopped. Run exp capture reset before retrying."
            )
        try:
            self.process.stdin.write(b"PING\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise CaptureSystemError("Capture helper disconnected. Run exp capture reset.") from exc

    def close(self, timeout: float = _CLOSE_TIMEOUT) -> None:
        """Release routing and wait for confirmation; never kill the cleanup process."""
        if self._closed:
            return
        self._closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            event, _ = _read_event(self.process, timeout)
            self.process.wait(timeout=timeout)
            if event != "stopped" or self.process.returncode != 0:
                raise CaptureSystemError(
                    "Capture cleanup was not confirmed. Run exp capture reset."
                )
        except subprocess.TimeoutExpired as exc:
            raise CaptureSystemError("Capture cleanup timed out. Run exp capture reset.") from exc
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()


def reset_capture_system() -> None:
    """Recover this feature's marked hosts entries offline, without loading user credentials."""
    _authorize()
    try:
        result = subprocess.run(
            _command("reset"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=_HELPER_ENV,
            check=False,
            timeout=_CLOSE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureSystemError("Capture reset did not finish. Retry exp capture reset.") from exc
    try:
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("invalid reset response")
        detail = value.get("detail", "Capture reset failed; inspect the system recovery state.")
        if not isinstance(detail, str):
            raise ValueError("invalid reset detail")
        if result.returncode or value.get("event") != "reset":
            raise CaptureSystemError(detail)
    except (ValueError, UnicodeError) as exc:
        raise CaptureSystemError(
            "Capture reset did not confirm cleanup. Retry exp capture reset."
        ) from exc
