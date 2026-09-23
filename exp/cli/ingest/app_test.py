"""The ingest CLI commits trace evidence without building projects or calling providers."""

from pathlib import Path

import pytest
from rich.text import Text
from typer import rich_utils
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.ingest.persistence_test import _source
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.runtime.gateway.ingest import load_gateway_capture
from exp.runtime.gateway.ingest.conversion_test import _database, _experience


@pytest.mark.parametrize("color", [False, True])
def test_ingest_requires_export_in_automation(monkeypatch: pytest.MonkeyPatch, color: bool) -> None:
    """Missing traces show a repair command with plain or colored terminal output."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", color)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "standard")
    result = CliRunner().invoke(app, ["ingest", "powerset", "--non-interactive"], color=color)
    assert result.exit_code != 0
    assert "--traces PATH" in Text.from_ansi(result.output).plain


@pytest.mark.parametrize("dry_run", [False, True])
def test_ingest_stores_all_traces_without_setup_or_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """Fresh installs import 20 scenarios without model setup, embeddings or project files."""
    root = tmp_path / "state"
    path = _source(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Any build, credential or provider setup would violate pure-ingest semantics."""
        pytest.fail("ingest called model setup or the grounded-build pipeline")

    monkeypatch.setattr("exp.cli.build.app.RuntimeModelCatalog", forbidden)
    monkeypatch.setattr("exp.cli.build.app.resolve_setup_providers", forbidden)
    arguments = [
        "ingest",
        "powerset",
        "--traces",
        str(path),
        "--root",
        str(root),
        "--non-interactive",
    ]
    if dry_run:
        arguments.append("--dry-run")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "20 accepted traces" in result.output
    assert "1 excluded records" in result.output
    assert not (root / "projects").exists()
    if dry_run:
        assert "No database or project files written" in result.output
        assert not root.exists()
    else:
        store = SQLiteTraceStore(trace_database_path(root))
        imports = store.list_imports("powerset")
        assert len(imports) == 1
        restored = read_ingested_traces(root, imports[0])
        assert len(restored.traces) == 20 and len(restored.issues) == 1
        repeated = CliRunner().invoke(app, arguments)
        assert repeated.exit_code == 0 and "Already imported" in repeated.output
        assert store.list_imports("powerset") == imports


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--source", "gateway"], "requires --identity ID"),
        (["--identity", "default"], "requires --source gateway"),
        (["--source", "unknown"], "unsupported source"),
    ],
)
@pytest.mark.parametrize("color", [False, True])
def test_bad_source_selection_has_no_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
    color: bool,
) -> None:
    """Source/identity errors stay readable in colored terminals and create no storage."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", color)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "standard" if color else None)
    root = tmp_path / "state"
    result = CliRunner().invoke(
        app,
        ["ingest", "powerset", "--root", str(root), "--non-interactive", *arguments],
        color=color,
    )
    assert ("\x1b[" in result.output) == color
    assert result.exit_code == 2 and message in Text.from_ansi(result.output).plain
    assert not root.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_gateway_source_uses_scoped_runtime_adapter(tmp_path: Path, dry_run: bool) -> None:
    """CLI dispatch retains exact gateway evidence while keeping identities isolated."""
    source = tmp_path / "traffic.db"
    root = tmp_path / "destination"
    _database(source, (_experience("developer"), _experience("other")))
    arguments = [
        "ingest",
        "powerset",
        "--source",
        "gateway",
        "--identity",
        "developer",
        "--traces",
        str(source),
        "--root",
        str(root),
        "--non-interactive",
    ]
    if dry_run:
        arguments.append("--dry-run")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "1 accepted traces" in result.output
    if dry_run:
        assert not root.exists()
    else:
        store = SQLiteTraceStore(trace_database_path(root))
        imports = store.list_imports("powerset")
        assert len(imports) == 1
        assert read_ingested_traces(root, imports[0]) == load_gateway_capture(
            source, identity_id="developer"
        )
        repeated = CliRunner().invoke(app, arguments)
        assert repeated.exit_code == 0 and "Already imported" in repeated.output
        assert store.list_imports("powerset") == imports
