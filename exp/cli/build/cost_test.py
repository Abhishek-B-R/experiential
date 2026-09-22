"""Build-cost ceiling command tests."""

from pathlib import Path

import pytest

from exp.cli.build.cost import over_ceiling_message, sufficient_ceiling_usd


@pytest.mark.parametrize("command_name", ["build", "ingest"])
@pytest.mark.parametrize("source", ["otlp", "chat-json"])
def test_over_ceiling_command_quotes_paths_and_covers_full_precision(
    command_name: str, source: str
) -> None:
    """The suggested command quotes paths and covers the exact estimate."""
    estimate = 1.2345674
    message = over_ceiling_message(
        estimate=estimate,
        ceiling=0.01,
        project="support",
        trace_file=Path("/tmp/my traces/export.jsonl"),
        source=source,
        root=Path("/tmp/exp root"),
        world_model=None,
        judge=None,
        embedder=None,
        top_k=5,
        command_name=command_name,
    )

    assert "conservative embedding estimate $1.234567 exceeds" in message
    assert f"Re-run with a higher ceiling: exp {command_name} support --traces" in message
    assert f"--source {source}" in message
    assert "--traces '/tmp/my traces/export.jsonl'" in message
    assert "--root '/tmp/exp root'" in message
    sufficient = sufficient_ceiling_usd(estimate)
    assert float(sufficient) >= estimate
    assert f"--max-build-cost-usd {sufficient}" in message
