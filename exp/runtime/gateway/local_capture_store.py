"""Read-only identity-scoped consumption of the native gateway capture database."""

import errno
import os
import sqlite3
import stat
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from exp.runtime.gateway.local_capture_contracts import CapturedExchange, LocalCaptureScope

_NOT_A_REGULAR_FILE = (
    "capture database must be a regular file; use a database path without a file symlink"
)


def _pin_regular_file(path: Path) -> int | None:
    """Open the final path entry without following it, and keep it pinned.

    ``O_NOFOLLOW`` refuses a final-entry symlink in the open itself, so there is
    no window between deciding the entry is a regular file and holding it. The
    descriptor stays open for the whole read: it pins the inode, so the identity
    the caller checks against cannot be recycled underneath it.

    A symlink at the database filename would redirect the reader to content the
    operator never named, and every scope check downstream then inspects the
    wrong file and finds nothing wrong. File mode is deliberately not checked:
    native capture owns this database and its permissions, so a read-only
    consumer is not the component that gets to refuse them.

    Args:
        path: Capture database whose final entry has not been resolved.

    Returns:
        A descriptor on the validated entry, or ``None`` when the path is
        absent. Absence is left to the connection, which reports it as the
        actionable "collect fresh traffic" error a caller matches on; it is not
        the substitution this guards against.

    Raises:
        OSError: The entry cannot be opened for a reason other than absence.
        ValueError: The final entry is a symlink or another non-regular file.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as error:
        # ELOOP is how O_NOFOLLOW reports "the final entry is a symlink",
        # including a dangling one, which must not read as an absent file.
        if error.errno not in {errno.ELOOP, errno.EMLINK}:
            raise
        raise ValueError(_NOT_A_REGULAR_FILE) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(_NOT_A_REGULAR_FILE)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_pinned_identity(path: Path, descriptor: int) -> None:
    """Fail closed unless ``path`` still names the pinned entry.

    SQLite is handed a pathname, not this descriptor, so it resolves the name a
    second time and could open a different file than the one validated. Only the
    caller can tell: comparing the name's CURRENT target against the pinned
    inode answers whether the two opens agree, and a mismatch means the entry
    was replaced around the database open. That is refused rather than read.

    ``stat`` here, not ``lstat``, precisely because it must answer the question
    SQLite's own open asked: what does this name resolve to.

    Args:
        path: The database pathname handed to SQLite.
        descriptor: The pinned entry from :func:`_pin_regular_file`.

    Raises:
        OSError: The path's metadata cannot be read.
        ValueError: The name no longer resolves to the validated entry.
    """
    pinned = os.fstat(descriptor)
    current = path.stat()
    if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
        raise ValueError(
            "capture database changed identity while it was being opened; "
            "retry the read once the path is stable"
        )


@dataclass(frozen=True)
class CaptureRow:
    """One durable gateway exchange and its monotonically increasing cursor.

    Attributes:
        sequence: Database-local cursor; retention can leave gaps.
        experience: Validated exchange in the reader's identity/application scope.
    """

    sequence: int
    experience: CapturedExchange


class LocalCaptureStore:
    """Read bounded pages without acquiring write authority over captured traffic.

    Cursors are local to one database. Retention may create gaps; consumers must
    checkpoint the last returned sequence rather than assume contiguous IDs.
    """

    def __init__(self, database_path: Path, scope: LocalCaptureScope) -> None:
        """Bind an existing local content database and an explicit application scope."""
        # Resolve directory aliases without hiding a symlink at the final filename:
        # a whole-path resolve() answers with the link TARGET, which is exactly the
        # substitution the read-time check below exists to catch.
        self._path = database_path.parent.resolve() / database_path.name
        self._scope = scope

    def read_after(self, sequence: int = 0, *, limit: int = 100) -> tuple[CaptureRow, ...]:
        """Return at most one bounded page for the bound application.

        Args:
            sequence: Last consumed row sequence, or zero for retained history.
            limit: Maximum returned rows, between one and one thousand.

        Returns:
            Validated durable records ordered by increasing sequence.
        """
        if sequence < 0 or not 1 <= limit <= 1000:
            raise ValueError("sequence must be nonnegative and limit must be between 1 and 1000")
        return self._read(sequence=sequence, limit=limit)

    def read_snapshot(self) -> tuple[CaptureRow, ...]:
        """Read all retained scoped captures from one consistent SQLite read snapshot.

        This explicit convenience method materializes the snapshot. Importers use
        iter_snapshot to copy it to private disk without retaining the corpus in memory.
        """
        return self._read(sequence=0, limit=None)

    def iter_snapshot(self) -> Generator[CaptureRow]:
        """Yield retained scoped rows from one snapshot; close the iterator on early exit."""
        return self._iter_read(sequence=0, limit=None)

    def _read(self, *, sequence: int, limit: int | None) -> tuple[CaptureRow, ...]:
        """Materialize explicitly requested pages for callers needing random access."""
        return tuple(self._iter_read(sequence=sequence, limit=limit))

    def _iter_read(self, *, sequence: int, limit: int | None) -> Generator[CaptureRow]:
        """Own one snapshot and validate every payload against its durable partition."""
        # Validated per read rather than once at construction: a store outlives
        # the call that built it, so the entry can be replaced in between. Both
        # read paths land here, so neither can be left behind.
        descriptor = _pin_regular_file(self._path)
        try:
            connection = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True, timeout=1.0)
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise
        try:
            if descriptor is not None:
                # Before any query: SQLite resolved the pathname itself, so this
                # is what proves it opened the entry that was validated.
                _require_pinned_identity(self._path, descriptor)
            connection.execute("BEGIN")
            cursor = connection.execute(
                "SELECT sequence, payload FROM gateway_captures "
                "WHERE user_id = ? AND application_id = ? AND sequence > ? "
                "AND expires_at > unixepoch() ORDER BY sequence LIMIT ?",
                (
                    self._scope.user_id,
                    self._scope.application_id,
                    sequence,
                    limit if limit is not None else -1,
                ),
            )
            for row in cursor:
                experience = CapturedExchange.model_validate_json(row[1])
                if experience.scope != self._scope:
                    raise ValueError("experience payload scope differs from its durable partition")
                yield CaptureRow(sequence=row[0], experience=experience)
        finally:
            connection.close()
            if descriptor is not None:
                # Held until the read is done: while it is open the inode cannot
                # be recycled, so the identity checked above stays meaningful.
                os.close(descriptor)
