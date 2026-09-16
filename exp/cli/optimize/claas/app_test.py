"""Application setup and status CLI behavior without provider access."""

import json
from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from exp.cli.app import app

_RUNNER = CliRunner()


def _initialize(root: Path) -> list[str]:
    """Build one explicit application initialization command."""
    return [
        "optimize",
        "claas",
        "init",
        "claims",
        "--base-model",
        "test-model",
        "--revision",
        "a" * 40,
        "--world-model",
        "world",
        "--judge",
        "judge",
        "--root",
        str(root),
    ]


def test_cli_initializes_and_reads_configuration_without_provider_calls(tmp_path: Path) -> None:
    """Round-trip configured ownership and pinned identities through the public CLI."""
    result = _RUNNER.invoke(app, _initialize(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Configured CLaaS" in result.output
    result = _RUNNER.invoke(app, ["optimize", "claas", "status", "claims", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    config = json.loads(result.output)
    assert config["scope"] == {"user_id": "default", "application_id": "claims"}
    assert config["compute"] == "local"
    assert config["tokenizer_revision"] == "a" * 40


def test_cli_rejects_an_implicit_base_model_change(tmp_path: Path) -> None:
    """Reject reuse of an existing adapter with different base weights."""
    assert _RUNNER.invoke(app, _initialize(tmp_path)).exit_code == 0
    arguments = _initialize(tmp_path)
    arguments[arguments.index("a" * 40)] = "b" * 40
    result = _RUNNER.invoke(app, [*arguments, "--replace"])
    assert result.exit_code != 0
    assert "immutable" in result.output


def test_status_never_reads_another_users_application(tmp_path: Path) -> None:
    """Keep application status isolated by authenticated owner."""
    assert _RUNNER.invoke(app, _initialize(tmp_path)).exit_code == 0
    result = _RUNNER.invoke(
        app,
        ["optimize", "claas", "status", "claims", "--user", "other", "--root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "not configured" in result.output


def test_cli_generic_application_requires_no_world_model(tmp_path: Path) -> None:
    """Manual or executable environments need only model and application identity."""
    arguments = _initialize(tmp_path)
    for option in ("--world-model", "--judge"):
        index = arguments.index(option)
        del arguments[index : index + 2]
    result = _RUNNER.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    result = _RUNNER.invoke(app, ["optimize", "claas", "status", "claims", "--root", str(tmp_path)])
    config = json.loads(result.output)
    assert "world_model_alias" not in config
    assert "judge_alias" not in config
    assert not tuple(tmp_path.rglob("traffic-workflow.json"))


def test_cli_traffic_workflow_requires_both_provider_choices(tmp_path: Path) -> None:
    """An incomplete workflow selection fails before persisting learner configuration."""
    arguments = _initialize(tmp_path)
    index = arguments.index("--judge")
    del arguments[index : index + 2]
    result = _RUNNER.invoke(app, arguments, color=True)
    assert result.exit_code != 0
    assert "both --world-model and --judge" in unstyle(result.output)
    assert not tuple(tmp_path.rglob("config.json"))
