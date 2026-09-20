"""Exercise local-backend diagnostics using only synthetic app archives and paths."""

from __future__ import annotations

import io
import os
import plistlib
import select
import stat
import subprocess
import sys
import tarfile
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path

import pytest

from exp.runtime.capture import local_backend


def _archive(
    directory: Path, *, app_version: str = "12.0", extension_version: str = "12.0"
) -> Path:
    """Create metadata-only dependency packaging without an executable or real extension."""
    package = directory / "mitmproxy_macos"
    package.mkdir()
    archive = package / "Mitmproxy Redirector.app.tar"
    with tarfile.open(archive, "w:") as bundle:
        for name, version in (
            (local_backend._APP_PLIST, app_version),
            (local_backend._EXTENSION_PLIST, extension_version),
        ):
            payload = plistlib.dumps({"LSMinimumSystemVersion": version})
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    return archive


@pytest.fixture
def backend_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Limit every preflight path to the test directory and report a normal macOS user."""
    applications = tmp_path / "Applications"
    applications.mkdir()
    monkeypatch.setattr(local_backend, "_APPLICATIONS", applications)
    monkeypatch.setattr(local_backend.sys, "platform", "darwin")
    monkeypatch.setattr(local_backend.os, "geteuid", lambda: 501)
    monkeypatch.setattr(local_backend.platform, "mac_ver", lambda: ("15.0", ("", "", ""), ""))
    monkeypatch.delenv("MITMPROXY_KEEP_REDIRECTOR", raising=False)
    package = Distribution.at(tmp_path / "mitmproxy_macos-0.12.11.dist-info")
    monkeypatch.setattr(local_backend, "distribution", lambda name: package)
    return applications


def test_preflight_does_not_install_or_activate(backend_environment: Path, tmp_path: Path) -> None:
    """A valid packaged dependency passes without extracting an app or changing any file."""
    _archive(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    local_backend.require_local_backend()
    assert before == {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    assert list(backend_environment.iterdir()) == []


def test_preflight_rejects_other_platform_before_package_lookup(
    backend_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported systems receive the product boundary, not a missing dependency error."""
    monkeypatch.setattr(local_backend.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS only"):
        local_backend.require_local_backend()


def test_preflight_rejects_root(backend_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential-bearing Capture must remain in the normal user's process."""
    monkeypatch.setattr(local_backend.os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="normal user, not with sudo"):
        local_backend.require_local_backend()


def test_missing_dependency_explains_reinstallation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing wheel metadata produces an actionable dependency-specific error."""

    def missing(name: str) -> Distribution:
        """Model an environment where the macOS dependency was not installed."""
        raise PackageNotFoundError(name)

    monkeypatch.setattr(local_backend, "distribution", missing)
    with pytest.raises(RuntimeError, match="Reinstall Experiential.*mitmproxy-macos"):
        local_backend._packaged_archive()


def test_missing_archive_explains_reinstallation(backend_environment: Path) -> None:
    """Installed metadata without the packaged app does not count as readiness."""
    with pytest.raises(RuntimeError, match="redirector package is missing or invalid"):
        local_backend.require_local_backend()


@pytest.mark.parametrize("content", [b"invalid tar", b"\0" * 10240])
def test_invalid_or_empty_archive_is_rejected(tmp_path: Path, content: bytes) -> None:
    """Corrupt archives and archives without the app metadata fail before installation."""
    archive = tmp_path / "invalid.tar"
    archive.write_bytes(content)
    with pytest.raises(RuntimeError, match="Reinstall Experiential"):
        local_backend._minimum_macos(archive)


@pytest.mark.parametrize("content", [b"invalid plist", b"<?xml version='1.0'?><plist><"])
def test_malformed_plist_explains_reinstallation(tmp_path: Path, content: bytes) -> None:
    """Invalid binary or XML metadata produces the same actionable package diagnostic."""
    archive = tmp_path / "malformed.tar"
    with tarfile.open(archive, "w:") as bundle:
        member = tarfile.TarInfo(local_backend._APP_PLIST)
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    with pytest.raises(RuntimeError, match="Reinstall Experiential"):
        local_backend._minimum_macos(archive)


@pytest.mark.parametrize("version", ["", "twelve", "12.-1", "99999999999999"])
def test_invalid_deployment_target_is_rejected(tmp_path: Path, version: str) -> None:
    """Malformed deployment metadata is not silently treated as compatible."""
    archive = _archive(tmp_path, extension_version=version)
    with pytest.raises(RuntimeError, match="Reinstall Experiential"):
        local_backend._minimum_macos(archive)


def test_extension_minimum_is_enforced(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newer of app and extension requirements controls platform compatibility."""
    _archive(tmp_path, app_version="12.0", extension_version="13.1")
    monkeypatch.setattr(local_backend.platform, "mac_ver", lambda: ("13.0", ("", "", ""), ""))
    with pytest.raises(RuntimeError, match="macOS 13.1 or newer"):
        local_backend.require_local_backend()


def test_minimum_os_boundary_passes(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact packaged deployment target is accepted."""
    _archive(tmp_path)
    monkeypatch.setattr(local_backend.platform, "mac_ver", lambda: ("12.0", ("", "", ""), ""))
    local_backend.require_local_backend()


def test_unmanaged_redirector_override_is_explicit(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upstream developer override cannot silently bypass package management checks."""
    _archive(tmp_path)
    monkeypatch.setenv("MITMPROXY_KEEP_REDIRECTOR", "1")
    with pytest.raises(RuntimeError, match="Unset MITMPROXY_KEEP_REDIRECTOR"):
        local_backend.require_local_backend()


def test_install_permission_denied_is_actionable(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonwritable Applications directory fails before login or system authorization."""
    _archive(tmp_path)
    monkeypatch.setattr(local_backend, "_writable_directory", lambda path: False)
    with pytest.raises(RuntimeError, match="Ask your administrator.*normal user"):
        local_backend.require_local_backend()
    assert list(backend_environment.iterdir()) == []


def _installed_app(applications: Path, archive: Path, *, current: bool) -> Path:
    """Create a fake installed bundle whose timestamp controls mitmproxy's update decision."""
    app = applications / local_backend._APP_NAME
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    plist = contents / "Info.plist"
    plist.write_bytes(b"metadata")
    timestamp = archive.stat().st_mtime_ns + (0 if current else 1)
    os.utime(plist, ns=(timestamp, timestamp))
    executable = contents / "MacOS/Mitmproxy Redirector"
    executable.write_bytes(b"not a real executable")
    executable.chmod(0o700)
    return app


def test_current_install_does_not_require_update_permission(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact reusable app can run even when the user cannot replace /Applications apps."""
    archive = _archive(tmp_path)
    _installed_app(backend_environment, archive, current=True)
    monkeypatch.setattr(local_backend, "_writable_directory", lambda path: False)
    local_backend.require_local_backend()


def test_outdated_install_requires_app_replacement_access(
    backend_environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writable Applications alone is insufficient to replace a protected old app."""
    archive = _archive(tmp_path)
    _installed_app(backend_environment, archive, current=False)
    monkeypatch.setattr(
        local_backend, "_writable_directory", lambda path: path == backend_environment
    )
    with pytest.raises(RuntimeError, match="permission to install or update"):
        local_backend.require_local_backend()


def test_matching_timestamp_without_executable_fails(
    backend_environment: Path, tmp_path: Path
) -> None:
    """A partial installation cannot pass merely because the plist timestamp matches."""
    archive = _archive(tmp_path)
    app = _installed_app(backend_environment, archive, current=True)
    (app / "Contents/MacOS/Mitmproxy Redirector").unlink()
    with pytest.raises(RuntimeError, match="installed Mitmproxy Redirector app is incomplete"):
        local_backend.require_local_backend()


@pytest.fixture
def capture_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Replace only the lock's home with a private synthetic directory."""
    home = tmp_path.resolve() / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_foreground_lock_is_private_and_reused(capture_home: Path) -> None:
    """Release keeps one private inode so later processes cannot lock competing files."""
    path = capture_home / "Library/Application Support/exp/capture/foreground.lock"
    with local_backend.capture_instance():
        metadata = path.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.is_file()
    with local_backend.capture_instance():
        assert path.stat().st_ino == metadata.st_ino


def test_foreground_lock_releases_after_body_exception(capture_home: Path) -> None:
    """Application errors retain their meaning and still release ownership."""
    with pytest.raises(OSError, match="application failed"):
        with local_backend.capture_instance():
            raise OSError("application failed")
    with local_backend.capture_instance():
        pass


_LOCK_CHILD = """
import sys
from pathlib import Path
from unittest.mock import patch
from exp.runtime.capture.local_backend import capture_instance

with patch.object(Path, 'home', return_value=Path(sys.argv[1])):
    try:
        with capture_instance():
            sys.stdout.write('locked\\n')
            sys.stdout.flush()
            if sys.argv[2] == 'hold':
                sys.stdin.read()
    except RuntimeError as exc:
        sys.stderr.write(str(exc))
        sys.exit(23)
"""


def test_foreground_lock_rejects_other_process_and_profile(
    capture_home: Path, tmp_path: Path
) -> None:
    """Different XDG profiles still contend for the same user's actual OS lock."""
    with local_backend.capture_instance():
        child = subprocess.run(
            [sys.executable, "-c", _LOCK_CHILD, str(capture_home), "attempt"],
            cwd=Path(local_backend.__file__).parents[3],
            env={**os.environ, "XDG_DATA_HOME": str(tmp_path / "another-profile")},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    assert child.returncode == 23
    assert "Capture is already running for this macOS user" in child.stderr
    assert not (tmp_path / "another-profile").exists()


def test_foreground_lock_releases_after_process_crash(capture_home: Path) -> None:
    """SIGKILL releases kernel ownership without requiring reset or deleting the file."""
    child = subprocess.Popen(
        [sys.executable, "-c", _LOCK_CHILD, str(capture_home), "hold"],
        cwd=Path(local_backend.__file__).parents[3],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0], "lock child did not become ready"
        assert child.stdout.readline() == "locked\n"
        with pytest.raises(RuntimeError, match="Capture is already running"):
            with local_backend.capture_instance():
                pytest.fail("the other process already owns Capture")
        path = capture_home / "Library/Application Support/exp/capture/foreground.lock"
        inode = path.stat().st_ino
        child.kill()
        child.wait(timeout=10)
        with local_backend.capture_instance():
            assert path.stat().st_ino == inode
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize("component", ["Library", "Library/Application Support/exp/capture"])
def test_foreground_lock_rejects_symlink_directory(
    capture_home: Path, tmp_path: Path, component: str
) -> None:
    """No file is created through a substituted application or Library directory."""
    target = tmp_path / "redirected"
    target.mkdir()
    link = capture_home / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("symlink directory was accepted")
    assert list(target.iterdir()) == []


def test_foreground_lock_rejects_symlink_home_ancestor(
    capture_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking the leaf home alone cannot permit an ancestor that redirects its path."""
    link = tmp_path / "redirected"
    link.symlink_to(capture_home.parent, target_is_directory=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: link / "home"))
    with pytest.raises(RuntimeError, match="unsafe ancestor"):
        with local_backend.capture_instance():
            pytest.fail("symlink ancestor was accepted")
    assert list(capture_home.iterdir()) == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "shared"])
def test_foreground_lock_rejects_unsafe_file(capture_home: Path, tmp_path: Path, kind: str) -> None:
    """Reject aliases and shared files before FileLock can truncate or chmod a target."""
    path = local_backend._foreground_lock_path()
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "hardlink":
        path.hardlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_text("shared")
        path.chmod(0o666)
    with pytest.raises(RuntimeError, match="foreground lock is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("unsafe lock file was accepted")
    assert target.read_text() == "preserve"


def test_foreground_lock_rejects_shared_parent(capture_home: Path) -> None:
    """A writable parent would allow another user to replace the private lock directory."""
    library = capture_home / "Library"
    library.mkdir(mode=0o777)
    library.chmod(0o777)
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("shared parent was accepted")
    assert list(library.iterdir()) == []


def test_foreground_lock_rejects_foreign_home_owner(
    capture_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different effective user cannot use or normalize this user's capture directory."""
    monkeypatch.setattr(local_backend.os, "geteuid", lambda: capture_home.stat().st_uid + 1)
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        with local_backend.capture_instance():
            pytest.fail("foreign home was accepted")
    assert list(capture_home.iterdir()) == []


def test_foreground_lock_normalizes_only_capture_directory(capture_home: Path) -> None:
    """Legacy readable directories become private without chmodding the user's Library."""
    directory = capture_home / "Library/Application Support/exp/capture"
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    library = capture_home / "Library"
    library.chmod(0o755)
    with local_backend.capture_instance():
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(library.stat().st_mode) == 0o755
