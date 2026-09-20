"""Preflight checks and per-user ownership of the macOS local redirector."""

from __future__ import annotations

import os
import platform
import plistlib
import re
import stat
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from xml.parsers.expat import ExpatError

from filelock import FileLock, Timeout

_APPLICATIONS = Path("/Applications")
_APP_NAME = "Mitmproxy Redirector.app"
_APP_PLIST = f"{_APP_NAME}/Contents/Info.plist"
_EXTENSION_PLIST = (
    f"{_APP_NAME}/Contents/Library/SystemExtensions/"
    "org.mitmproxy.macos-redirector.network-extension.systemextension/Contents/Info.plist"
)
_REINSTALL = (
    "Capture's macOS redirector package is missing or invalid. "
    "Reinstall Experiential with Python 3.13 or newer to restore its mitmproxy-macos dependency."
)


@contextmanager
def capture_instance() -> Iterator[None]:
    """Hold one foreground Capture session per user, including startup and shutdown.

    This location deliberately ignores profile roots and XDG settings: every session
    controls the same macOS redirector. The OS releases the lock when a process exits
    or crashes. Keeping the file preserves its inode for other waiting processes.

    Yields:
        None while this process owns the user's Capture session.

    Raises:
        RuntimeError: Another session is active or the lock path is unsafe or inaccessible.
    """
    try:
        path = _foreground_lock_path()
        lock = FileLock(path, timeout=0, mode=0o600)
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(
            "Capture is already running for this macOS user. Stop that session before "
            "starting another one."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Capture could not acquire its foreground lock. Check access to "
            "~/Library/Application Support/exp/capture and rerun exp capture."
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _foreground_lock_path() -> Path:
    """Create only user-owned directories and reject redirected or shared lock paths."""
    home = Path.home()
    for ancestor in reversed(home.parents):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise RuntimeError(f"Capture's home directory has an unsafe ancestor: {ancestor}")
    _require_owned_directory(home)
    directory = home
    for name in ("Library", "Application Support", "exp", "capture"):
        directory /= name
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _require_owned_directory(directory)
    # Older installations may have created this application-owned directory with 0755.
    # Normalize this narrow directory only, never the user's Library or home directory.
    directory.chmod(0o700)
    path = directory / "foreground.lock"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return path
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
    ):
        raise RuntimeError(
            f"Capture's foreground lock is unsafe: {path}. Restore a regular file owned "
            "only by your user, then rerun exp capture."
        )
    return path


def _require_owned_directory(path: Path) -> None:
    """Reject symlinks, foreign ownership, and directories writable by other users."""
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise RuntimeError(
            f"Capture's foreground lock directory is unsafe: {path}. Use a directory "
            "owned by your user that other users cannot modify, then rerun exp capture."
        )


def require_local_backend() -> None:
    """Validate prerequisites without installing or activating a Network Extension.

    The actual mitmproxy startup owns app installation and macOS authorization.
    Passing this check does not establish that the user has approved the extension.

    Raises:
        RuntimeError: The platform, packaged app, or installation permissions prevent startup.
    """
    if sys.platform != "darwin":
        raise RuntimeError("System Capture currently supports macOS only.")
    if os.geteuid() == 0:
        raise RuntimeError("Run exp capture as your normal user, not with sudo.")
    archive = _packaged_archive()
    minimum = _minimum_macos(archive)
    current = _version(platform.mac_ver()[0])
    if current is None or current < minimum:
        required = ".".join(str(part) for part in minimum[:2])
        raise RuntimeError(f"Capture's local redirector requires macOS {required} or newer.")
    if os.environ.get("MITMPROXY_KEEP_REDIRECTOR") == "1":
        raise RuntimeError(
            "Unset MITMPROXY_KEEP_REDIRECTOR and rerun exp capture so mitmproxy can manage "
            "its packaged redirector."
        )
    _require_install_access(archive)


def _packaged_archive() -> Path:
    """Find the official dependency's app archive without importing or starting its code."""
    try:
        package = distribution("mitmproxy-macos")
        archive = Path(str(package.locate_file(f"mitmproxy_macos/{_APP_NAME}.tar")))
        if not archive.is_file():
            raise RuntimeError(_REINSTALL)
        return archive
    except (PackageNotFoundError, OSError) as exc:
        raise RuntimeError(_REINSTALL) from exc


def _version(value: str) -> tuple[int, int, int] | None:
    """Parse a numeric macOS release without assuming a marketing version name."""
    if re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]{1,4}){0,2}", value) is None:
        return None
    parts = [int(part) for part in value.split(".")]
    parts.extend([0] * (3 - len(parts)))
    return parts[0], parts[1], parts[2]


def _minimum_macos(archive: Path) -> tuple[int, int, int]:
    """Read both bundle deployment targets in place, never extracting executable files."""
    minimum = (0, 0, 0)
    try:
        with tarfile.open(archive, "r:") as bundle:
            for name in (_APP_PLIST, _EXTENSION_PLIST):
                member = bundle.getmember(name)
                if not member.isfile() or not 0 < member.size <= 65536:
                    raise ValueError("invalid app metadata")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("missing app metadata")
                with source:
                    metadata = plistlib.loads(source.read(65537))
                version = (
                    metadata.get("LSMinimumSystemVersion") if isinstance(metadata, dict) else None
                )
                parsed = _version(version) if isinstance(version, str) else None
                if parsed is None:
                    raise ValueError("invalid app deployment target")
                minimum = max(minimum, parsed)
    except (
        OSError,
        KeyError,
        ValueError,
        tarfile.TarError,
        plistlib.InvalidFileException,
        ExpatError,
    ) as exc:
        raise RuntimeError(_REINSTALL) from exc
    return minimum


def _require_install_access(archive: Path) -> None:
    """Match mitmproxy's mtime reuse rule before requiring app replacement permissions."""
    app = _APPLICATIONS / _APP_NAME
    plist = app / "Contents/Info.plist"
    executable = app / "Contents/MacOS/Mitmproxy Redirector"
    try:
        if plist.is_file() and plist.stat().st_mtime_ns == archive.stat().st_mtime_ns:
            if executable.is_file() and os.access(executable, os.X_OK):
                return
            raise RuntimeError(
                "The installed Mitmproxy Redirector app is incomplete. Ask an administrator "
                "to remove it from /Applications, then rerun exp capture to reinstall it."
            )
        if _writable_directory(_APPLICATIONS) and (not app.exists() or _writable_directory(app)):
            return
    except OSError as exc:
        raise RuntimeError("Capture could not inspect its local redirector installation.") from exc
    raise RuntimeError(
        "Capture needs permission to install or update /Applications/Mitmproxy Redirector.app. "
        "Ask your administrator to grant this user installation access, then rerun exp capture "
        "as your normal user."
    )


def _writable_directory(path: Path) -> bool:
    """Inspect access without creating files or requesting elevated privileges."""
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
