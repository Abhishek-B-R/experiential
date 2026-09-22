"""Trace ingestion command validation."""

import pytest
from rich.text import Text
from typer import rich_utils
from typer.testing import CliRunner

from exp.cli.app import app


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
