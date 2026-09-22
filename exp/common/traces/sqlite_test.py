"""Immutable import identity, atomicity, and evidence integrity in real SQLite."""

import sqlite3
from pathlib import Path

import pytest

from exp.common.core.artifacts import SourceIdentity
from exp.common.traces.sqlite import SQLiteTraceStore, TraceImportReceipt
from exp.common.traces.sqlite_schema import TraceStoreError
from exp.common.traces.trace import Trace, TraceSource
from exp.common.traces.trace_test import _trace


def _save(
    store: SQLiteTraceStore, traces: tuple[Trace, ...], project: str = "powerset"
) -> TraceImportReceipt:
    """Persist a synthetic source with retained normalization metadata."""
    return store.write_import(
        project,
        source_format="otlp",
        source=traces[0].source.identity,
        traces=traces,
        metadata={"issues": [{"source_record": "line-2", "message": "invalid record"}]},
    )


def test_repeat_reopen_overlap_and_project_membership(tmp_path: Path) -> None:
    """Repeats reuse evidence, overlaps share content, and projects stay scoped."""
    path = tmp_path / "traffic.db"
    store = SQLiteTraceStore(path)
    assert store.list_imports("powerset") == ()
    assert not path.exists()
    trace = _trace()
    first = _save(store, (trace,))
    saved = store.read_import(first.import_id)
    assert saved.traces == (trace,)
    assert saved.metadata["issues"]
    repeated = _save(SQLiteTraceStore(path), (trace,))
    assert repeated.import_id == first.import_id
    assert repeated.already_linked and repeated.new_records == 0
    assert store.read_import(first.import_id).created_at == saved.created_at
    assert path.stat().st_mode & 0o777 == 0o600
    other_source = TraceSource(
        identity=SourceIdentity(kind="file", source_id="other-export"),
        semantic_convention_version="1.37.0",
    )
    overlap = trace.model_copy(update={"source": other_source})
    second = _save(store, (overlap,))
    assert second.import_id != first.import_id and second.new_records == 0
    assert store.read_import(second.import_id).traces[0].source == other_source
    other = _save(store, (trace,), project="other")
    assert other.import_id == first.import_id and not other.already_linked
    assert store.list_imports("powerset") == (first.import_id, second.import_id)
    assert store.list_imports("other") == (first.import_id,)
    changed = trace.model_copy(update={"task": "Changed evidence with the same source trace ID"})
    third = _save(store, (changed,))
    assert third.new_records == 1 and third.import_id != first.import_id
    assert store.read_import(first.import_id) == saved


def test_write_failure_rolls_back_records_and_membership(tmp_path: Path) -> None:
    """A failure in the last write publishes neither a partial import nor new trace records."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    first = _save(store, (_trace(),))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_link BEFORE INSERT ON trace_project_imports "
            "BEGIN SELECT RAISE(ABORT, 'private payload error'); END"
        )
    with pytest.raises(TraceStoreError) as raised:
        _save(store, (_trace().model_copy(update={"task": "Different task"}),))
    assert "private payload" not in str(raised.value)
    assert store.list_imports("powerset") == (first.import_id,)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM trace_records").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM trace_imports").fetchone() == (1,)


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE trace_records SET payload='{}'",
        "UPDATE trace_records SET payload='not json private-content'",
        "UPDATE trace_records SET trace_id='other'",
        "UPDATE trace_imports SET metadata='{}'",
        "UPDATE trace_import_records SET ordinal=9",
        "DELETE FROM trace_import_records",
    ],
)
def test_corrupt_evidence_is_rejected_on_read_and_repeat(tmp_path: Path, mutation: str) -> None:
    """Neither reading nor idempotent reuse can bless changed bytes or missing memberships."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    first = _save(store, (_trace(),))
    with sqlite3.connect(store.path) as connection:
        connection.execute(mutation)
    with pytest.raises(TraceStoreError) as raised:
        store.read_import(first.import_id)
    assert "private-content" not in str(raised.value)
    with pytest.raises(TraceStoreError):
        _save(store, (_trace(),))
