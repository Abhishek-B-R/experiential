"""Trace ingestion command validation."""

import json
from pathlib import Path

import pytest
from rich.text import Text
from typer import rich_utils
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.build.app_test import _catalog, _otlp_export, _RuntimeCatalog
from exp.common.config.settings import set_maximum_command_cost_usd
from exp.common.project import ProjectStore
from exp.common.traces import load_trace_dataset


@pytest.mark.parametrize("color", [False, True])
def test_ingest_requires_export_in_automation(monkeypatch: pytest.MonkeyPatch, color: bool) -> None:
    """Missing traces show the same repair command with plain or colored terminal output."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", color)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "standard")
    result = CliRunner().invoke(app, ["ingest", "powerset", "--non-interactive"], color=color)
    assert result.exit_code != 0
    if color:
        assert "\x1b[" in result.output
    assert "--traces PATH" in Text.from_ansi(result.output).plain


@pytest.mark.parametrize("dry_run", [False, True])
def test_ingest_preserves_tool_trace_evidence_without_router_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """The CLI builds or preflights a small corpus with instructions and standard tool results."""
    root = tmp_path / ".exp"
    root.mkdir()
    _catalog(root)
    monkeypatch.setattr("exp.cli.build.app.RuntimeModelCatalog", _RuntimeCatalog)
    monkeypatch.setattr("exp.cli.build.app.capture_build_completed", lambda **_kwargs: None)
    records = [
        {
            "trace_id": f"research-{name}",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a company",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "messages": [
                {"role": "developer", "content": "Cite sources."},
                {"role": "user", "content": f"Research {name}"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-a", "content": f"{name} is a company."},
                {"role": "assistant", "content": "Company found."},
            ],
        }
        for name in ("Acme", "Beta")
    ]
    source = tmp_path / "research.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    arguments = [
        "ingest",
        "research",
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
    assert "exp eval" not in result.output
    store = ProjectStore(root, "research")
    config = store.load_project()
    assert config.trace_source == "chat-json"
    assert (config.build is None) == dry_run
    manifests = tuple(store.artifacts.read(item).manifest for item in store.artifacts.list_ids())
    assert not any(
        "rollout" in item.artifact_type or "router" in item.artifact_type for item in manifests
    )
    dataset = next(item for item in manifests if item.artifact_type == "trace-dataset")
    traces = load_trace_dataset(store.artifacts, dataset.artifact_id).traces
    assert len(traces) == 2
    assert all(trace.tools[0].name == "lookup" for trace in traces)
    assert all(
        trace.initial_context["instruction_messages"]
        == [{"role": "developer", "content": "Cite sources."}]
        for trace in traces
    )
    if dry_run:
        assert "No provider calls or build selection" in Text.from_ansi(result.output).plain
    else:
        replay = CliRunner().invoke(app, arguments)
        assert replay.exit_code == 0, replay.output
        assert store.load_project().build == config.build


@pytest.mark.parametrize("command_name", ["build", "ingest"])
def test_over_budget_recovery_keeps_invoked_command_and_trace_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command_name: str
) -> None:
    """A rejected ingestion retries the selected source through the command the user invoked."""
    root = tmp_path / ".exp"
    root.mkdir()
    _catalog(root)
    source = _otlp_export(tmp_path, count=2)
    monkeypatch.setattr("exp.cli.build.app.RuntimeModelCatalog", _RuntimeCatalog)
    monkeypatch.setattr("exp.cli.build.app._embedding_cost_ceiling", lambda *_args: 0.75)
    result = CliRunner().invoke(
        app,
        [
            command_name,
            "support",
            "--traces",
            str(source),
            "--source",
            "otlp",
            "--root",
            str(root),
            "--max-build-cost-usd",
            "0.01",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 2, result.output
    output = " ".join(Text.from_ansi(result.output).plain.replace("│", " ").split())
    assert "conservative embedding estimate $0.750000 exceeds" in output
    assert f"exp {command_name} support --traces" in output
    assert "--source otlp" in output
    assert "--max-build-cost-usd 0.750000" in output
    assert ProjectStore(root, "support").load_project().build is None


@pytest.mark.parametrize("command_name", ["build", "ingest"])
def test_spend_confirmation_names_the_invoked_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command_name: str
) -> None:
    """Both CLI entrypoints authorize their own command while sharing grounded-build execution."""
    monkeypatch.chdir(tmp_path)
    root = Path("custom-state")
    root.mkdir()
    _catalog(root)
    source = _otlp_export(Path("."), count=2)
    set_maximum_command_cost_usd(1.0, root)
    monkeypatch.setattr("exp.cli.build.app.RuntimeModelCatalog", _RuntimeCatalog)
    monkeypatch.setattr("exp.cli.build.app.capture_build_completed", lambda **_kwargs: None)
    monkeypatch.setattr("exp.cli.build.app._embedding_cost_ceiling", lambda *_args: 0.75)
    monkeypatch.setattr("exp.cli.shared.consent.can_prompt", lambda _console: True)
    monkeypatch.setattr("exp.cli.ingest.app.can_prompt", lambda _console: True)
    result = CliRunner().invoke(
        app,
        [
            command_name,
            "support",
            "--traces",
            str(source),
            "--source",
            "otlp",
            "--root",
            str(root),
        ],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    output = " ".join(Text.from_ansi(result.output).plain.replace("│", " ").split())
    assert f"Authorize exp {command_name} support --traces traces.jsonl" in output
    assert "--source otlp --root custom-state" in output
    assert "--max-build-cost-usd 5.0" in output
    assert ProjectStore(root, "support").load_project().build is not None
