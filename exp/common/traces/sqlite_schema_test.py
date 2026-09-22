"""Traffic-schema ownership is checked before SQLite settings or tables can change."""

import sqlite3
from pathlib import Path

import pytest

from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import TraceStoreError, initialize_schema, validate_schema
from exp.common.traces.sqlite_test import _save
from exp.common.traces.trace_test import _trace


@pytest.mark.parametrize(
    "schema, message",
    [
        ("CREATE TABLE other(payload TEXT)", "Unrecognized"),
        ("CREATE TABLE trace_records(payload TEXT)", "Incomplete"),
    ],
)
def test_unrecognized_database_is_preserved(tmp_path: Path, schema: str, message: str) -> None:
    """Unrelated evidence is not converted or changed into a trace store."""
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as connection:
        connection.execute(schema)
    before = path.read_bytes()
    with pytest.raises(TraceStoreError, match=message):
        _save(SQLiteTraceStore(path), (_trace(),))
    assert path.read_bytes() == before
    assert not Path(f"{path}-wal").exists()


def test_trace_schema_can_join_capture_and_rejects_unknown_version(tmp_path: Path) -> None:
    """Only the complete, supported trace namespace may coexist with captures."""
    with sqlite3.connect(tmp_path / "traffic.db") as connection:
        connection.execute("CREATE TABLE gateway_captures(payload TEXT)")
        connection.execute("INSERT INTO gateway_captures VALUES ('existing capture')")
        initialize_schema(connection)
        validate_schema(connection)
        assert connection.execute("SELECT payload FROM gateway_captures").fetchone() == (
            "existing capture",
        )
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("UPDATE trace_store_schema SET version=2")
        with pytest.raises(TraceStoreError, match="Unsupported"):
            validate_schema(connection)
