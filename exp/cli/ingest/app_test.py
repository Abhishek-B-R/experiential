"""Trace ingestion command validation."""

from typer.testing import CliRunner

from exp.cli.app import app


def test_ingest_requires_export_in_automation() -> None:
    """Missing traces never launch the separate router build wizard."""
    result = CliRunner().invoke(app, ["ingest", "powerset", "--non-interactive"])
    assert result.exit_code != 0
    assert "--traces PATH" in result.output
