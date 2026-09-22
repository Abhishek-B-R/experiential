"""Read-only identity-scoped consumption of the native gateway capture database."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from exp.runtime.gateway.local_capture_contracts import CapturedExchange, LocalCaptureScope


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
        self._path = database_path.resolve()
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

        Rows are fetched in pages without a total-record cutoff. The connection closes
        before normalization or writing imports, so capture delivery can continue.
        """
        return self._read(sequence=0, limit=None)

    def _read(self, *, sequence: int, limit: int | None) -> tuple[CaptureRow, ...]:
        """Own one snapshot and validate every payload against its durable partition."""
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
            result: list[CaptureRow] = []
            while rows := cursor.fetchmany(1000):
                for row in rows:
                    experience = CapturedExchange.model_validate_json(row[1])
                    if experience.scope != self._scope:
                        raise ValueError(
                            "experience payload scope differs from its durable partition"
                        )
                    result.append(CaptureRow(sequence=row[0], experience=experience))
        finally:
            connection.close()
        return tuple(result)
