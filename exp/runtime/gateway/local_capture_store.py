"""Read-only identity-scoped consumption of the native gateway capture database."""

import sqlite3
import stat
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from exp.runtime.gateway.local_capture_contracts import CapturedExchange, LocalCaptureScope


def _require_regular_file(path: Path) -> None:
    """Reject a final path entry that is not a regular file, without following it.

    A symlink at the database filename redirects the reader to content the
    operator never named, and every scope check downstream then inspects the
    wrong file and finds nothing wrong. File mode is deliberately not checked
    here: native capture owns this database and its permissions, so a read-only
    consumer is not the component that gets to refuse them.

    An absent path is left to the connection, which already reports it as the
    actionable "collect fresh traffic" error a caller matches on. Absence is not
    the substitution this guards against, and answering it here first would
    replace that guidance with a bare metadata error.

    Args:
        path: Capture database whose final entry has not been resolved.

    Raises:
        OSError: The path's metadata cannot be read for a reason other than
            absence.
        ValueError: The final entry is a symlink or another non-regular file.
    """
    try:
        # lstat, not stat: stat answers about the TARGET, which is the question
        # that lets a symlink through. A DANGLING link still answers here, so it
        # is refused rather than mistaken for an absent file.
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError(
            "capture database must be a regular file; use a database path without a file symlink"
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
        # Checked per read rather than once at construction: a store outlives the
        # call that built it, so the entry can be replaced with a link in between.
        # Both read paths land here, so neither can be left behind.
        _require_regular_file(self._path)
        connection = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
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
