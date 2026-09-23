"""Owned SQLite tables for canonical trace imports beside native gateway captures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TRACE_TABLE_SQL = {
    "trace_store_schema": "CREATE TABLE trace_store_schema "
    "(version INTEGER PRIMARY KEY CHECK(version=1)) STRICT",
    "trace_records": """CREATE TABLE trace_records (
        record_sha256 TEXT PRIMARY KEY CHECK(length(record_sha256)=64),
        trace_id TEXT NOT NULL, payload TEXT NOT NULL
    ) STRICT""",
    "trace_imports": """CREATE TABLE trace_imports (
        import_id TEXT PRIMARY KEY, source_format TEXT NOT NULL,
        source TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
    ) STRICT""",
    "trace_import_records": """CREATE TABLE trace_import_records (
        import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        ordinal INTEGER NOT NULL CHECK(ordinal>=0),
        record_sha256 TEXT NOT NULL REFERENCES trace_records(record_sha256),
        source TEXT NOT NULL,
        PRIMARY KEY(import_id,ordinal)
    ) STRICT""",
    "trace_project_imports": """CREATE TABLE trace_project_imports (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL, import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        UNIQUE(project_id,import_id)
    ) STRICT""",
}
TRACE_TABLES = frozenset(TRACE_TABLE_SQL)


class TraceStoreError(ValueError):
    """A trace database cannot be used without changing or losing stored evidence."""


def trace_database_path(root: Path) -> Path:
    """Return the shared content database used by capture and local ingestion."""
    return root.resolve() / "gateway" / "traffic.db"


def validate_schema(connection: sqlite3.Connection) -> None:
    """Reject unrelated or partially initialized databases before any mutation.

    Args:
        connection: Open connection to the proposed content database.

    Raises:
        TraceStoreError: Existing tables or the trace schema version are unsupported.
    """
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if tables - TRACE_TABLES - {"gateway_captures"}:
        raise TraceStoreError(
            "Unrecognized traffic database; preserve it and select another --root."
        )
    present = tables & TRACE_TABLES
    if present and present != TRACE_TABLES:
        raise TraceStoreError(
            "Incomplete trace database schema; preserve it and select another --root."
        )
    for table in present:
        saved_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0]
        if (
            " ".join(saved_sql.split()).casefold()
            != " ".join(TRACE_TABLE_SQL[table].split()).casefold()
        ):
            raise TraceStoreError(
                "Incompatible trace table definition; preserve it and select another --root."
            )
    if present and connection.execute("SELECT version FROM trace_store_schema").fetchall() != [
        (1,)
    ]:
        raise TraceStoreError(
            "Unsupported trace schema version; use a matching Experiential release."
        )


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create trace tables inside the caller's transaction, leaving capture rows untouched."""
    for statement in TRACE_TABLE_SQL.values():
        connection.execute(statement.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1))
    connection.execute("INSERT OR IGNORE INTO trace_store_schema VALUES (1)")
