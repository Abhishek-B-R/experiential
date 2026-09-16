"""Capture CA persistence never follows links or exposes signing keys."""

import os
import stat
from pathlib import Path

import pytest

from exp.runtime.capture.certificates import prepare_certificate


def test_certificate_reused_and_private(tmp_path: Path) -> None:
    """Repeated capture reuses one CA while its key stays owner-readable only."""
    directory = tmp_path / "certificates"
    certificate = prepare_certificate(directory)
    before = certificate.read_bytes()
    assert b"BEGIN CERTIFICATE" in before
    assert b"PRIVATE KEY" not in before
    assert prepare_certificate(directory).read_bytes() == before
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "mitmproxy-ca.pem").stat().st_mode) == 0o600


def test_linked_directory_is_rejected(tmp_path: Path) -> None:
    """A linked CA directory cannot redirect key creation outside its owner."""
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="user-owned directory"):
        prepare_certificate(linked)
    assert not list(original.iterdir())


def test_linked_ancestor_is_rejected_before_creating_files(tmp_path: Path) -> None:
    """A symlink above the CA directory is rejected before writing through it."""
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="without links"):
        prepare_certificate(linked / "capture" / "ca")
    assert not list(original.iterdir())


def test_linked_key_is_rejected_without_touching_target(tmp_path: Path) -> None:
    """An existing hard-linked key is rejected without changing its target."""
    original = tmp_path / "original"
    original.write_text("untouched")
    directory = tmp_path / "certificates"
    directory.mkdir()
    os.link(original, directory / "mitmproxy-ca.pem")
    with pytest.raises(ValueError, match="unsafe files"):
        prepare_certificate(directory)
    assert original.read_text() == "untouched"
