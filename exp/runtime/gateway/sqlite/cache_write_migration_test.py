"""Tests for preserving durable accounting across cache-write schema extension."""

import sqlite3
from pathlib import Path

import pytest

from exp.runtime.gateway.sqlite import migrations
from exp.runtime.gateway.sqlite.cache_write_migration import CACHE_WRITE_COLUMNS
from exp.runtime.gateway.sqlite.migrations import (
    GatewaySchemaError,
    connect_database,
    initialize_database,
)
from exp.runtime.gateway.sqlite.migrations_test import _replay_history


def test_failed_cache_extension_rolls_back_columns_and_keeps_prior_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial DDL failure preserves schema 21 and the original committed data."""
    path = tmp_path / "gateway.db"
    path.touch(mode=0o600)
    connection = connect_database(path)
    try:
        _replay_history(connection, upto=22)
        connection.execute(
            "INSERT INTO organizations VALUES ('org', 'org', 'Original', 1, 't', 't')"
        )
        connection.execute("PRAGMA user_version = 21")
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setitem(
        migrations._MIGRATIONS, 22, (*migrations._MIGRATIONS[22][:2], "INVALID SQL")
    )
    with pytest.raises(GatewaySchemaError, match="migration failed") as failure:
        initialize_database(path)
    assert isinstance(failure.value.__cause__, sqlite3.OperationalError)
    connection = connect_database(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 21
        assert (
            connection.execute("SELECT display_name FROM organizations").fetchone()[0] == "Original"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(gateway_attempts)")}
        assert columns.isdisjoint(CACHE_WRITE_COLUMNS)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_fresh_extension_and_reinitialization_have_one_nullable_column_set(tmp_path: Path) -> None:
    """Fresh setup and repeated initialization preserve one bounded schema extension."""
    path = tmp_path / "gateway.db"
    initialize_database(path)
    assert initialize_database(path) is None
    connection = connect_database(path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(gateway_attempts)")]
        assert all(columns.count(name) == 1 for name in CACHE_WRITE_COLUMNS)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()
