"""Owned SQLite tables for canonical trace imports beside native gateway captures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TRACE_TABLES = frozenset(
    {
        "trace_store_schema",
        "trace_records",
        "trace_imports",
        "trace_import_records",
        "trace_project_imports",
    }
)
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS trace_store_schema "
    "(version INTEGER PRIMARY KEY CHECK(version=1)) STRICT",
    "INSERT OR IGNORE INTO trace_store_schema VALUES (1)",
    """CREATE TABLE IF NOT EXISTS trace_records (
        record_sha256 TEXT PRIMARY KEY CHECK(length(record_sha256)=64),
        trace_id TEXT NOT NULL, payload TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE IF NOT EXISTS trace_imports (
        import_id TEXT PRIMARY KEY, source_format TEXT NOT NULL,
        source TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE IF NOT EXISTS trace_import_records (
        import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        ordinal INTEGER NOT NULL CHECK(ordinal>=0),
        record_sha256 TEXT NOT NULL REFERENCES trace_records(record_sha256),
        source TEXT NOT NULL,
        PRIMARY KEY(import_id,ordinal)
    ) STRICT""",
    """CREATE TABLE IF NOT EXISTS trace_project_imports (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL, import_id TEXT NOT NULL REFERENCES trace_imports(import_id),
        UNIQUE(project_id,import_id)
    ) STRICT""",
)


class TraceStoreError(ValueError):
    """A trace database cannot be used without changing or losing stored evidence."""


def trace_database_path(root: Path) -> Path:
    """Return the shared content database used by capture and local ingestion."""
    return (root / "gateway" / "traffic.db").resolve()


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
    if present and connection.execute("SELECT version FROM trace_store_schema").fetchall() != [
        (1,)
    ]:
        raise TraceStoreError(
            "Unsupported trace schema version; use a matching Experiential release."
        )


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create trace tables inside the caller's transaction, leaving capture rows untouched."""
    for statement in _SCHEMA:
        connection.execute(statement)
