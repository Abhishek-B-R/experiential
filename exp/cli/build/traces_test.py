"""Builds mine the exact saved import while source acquisition stays resumable and scoped."""

from pathlib import Path

import pytest

from exp.cli.build.traces import load_build_traces
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.ingest.persistence_test import _source
from exp.common.traces.ingest.sources import load_trace_source
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.runtime.gateway.ingest import load_gateway_capture
from exp.runtime.gateway.ingest.conversion_test import _database, _experience


@pytest.mark.parametrize("dry_run", [False, True])
def test_build_file_evidence_matches_saved_import_and_deduplicates(
    tmp_path: Path, dry_run: bool
) -> None:
    """Mining receives all canonical evidence and exclusions from the selected source."""
    source = _source(tmp_path)
    root = tmp_path / "state"
    expected = load_trace_source("chat-json", source)
    result = load_build_traces(
        "powerset", root=root, path=source, source="chat-json", dry_run=dry_run
    )
    assert result == expected
    assert len(result.traces) == 20 and len(result.issues) == 1
    if dry_run:
        assert not root.exists()
        return
    store = SQLiteTraceStore(trace_database_path(root))
    imports = store.list_imports("powerset")
    assert len(imports) == 1
    assert read_ingested_traces(root, imports[0]) == result
    assert load_build_traces("powerset", root=root, path=source, source="chat-json") == result
    assert store.list_imports("powerset") == imports
    source.unlink()
    assert read_ingested_traces(root, imports[0]) == expected


@pytest.mark.parametrize("dry_run", [False, True])
def test_build_gateway_evidence_preserves_identity_and_exact_snapshot(
    tmp_path: Path, dry_run: bool
) -> None:
    """Only the requested identity's retained captures reach the build corpus."""
    source = tmp_path / "traffic.db"
    root = tmp_path / "state"
    _database(source, (_experience("developer"), _experience("other")))
    expected = load_gateway_capture(source, identity_id="developer")
    result = load_build_traces(
        "powerset", root=root, path=source, source="gateway", identity="developer", dry_run=dry_run
    )
    assert result == expected and len(result.traces) == 1
    if dry_run:
        assert not root.exists()
    else:
        imports = SQLiteTraceStore(trace_database_path(root)).list_imports("powerset")
        assert len(imports) == 1
        assert read_ingested_traces(root, imports[0]) == expected


@pytest.mark.parametrize(
    "source,identity,message",
    [
        ("gateway", None, "requires --identity ID"),
        ("chat-json", "default", "requires --source gateway"),
        ("unknown", None, "unsupported trace source"),
    ],
)
def test_invalid_build_source_creates_no_workspace(
    tmp_path: Path, source: str, identity: str | None, message: str
) -> None:
    """Invalid source selection fails before any durable import or project write."""
    root = tmp_path / "state"
    with pytest.raises(ValueError, match=message):
        load_build_traces(
            "powerset", root=root, path=tmp_path / "missing", source=source, identity=identity
        )
    assert not root.exists()


def test_empty_build_source_is_not_published(tmp_path: Path) -> None:
    """Unusable evidence cannot become a selected build corpus."""
    source = tmp_path / "empty.jsonl"
    source.write_text("{}\n")
    root = tmp_path / "state"
    with pytest.raises(ValueError, match="no valid canonical traces"):
        load_build_traces("powerset", root=root, path=source, source="chat-json")
    assert not root.exists()
