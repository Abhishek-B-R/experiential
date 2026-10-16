"""Flag rules of the opt-in learning stage, checked before any build work."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app


@pytest.mark.parametrize(
    "flags",
    [
        ["--learn"],
        ["--memory-recall", "jev"],
        ["--max-learning-attempts", "10"],
        ["--hold-out"],
    ],
)
def test_learning_flags_require_each_other(tmp_path: Path, flags: list[str]) -> None:
    """``--learn`` needs a recall alias, and learning options need ``--learn``."""
    root = tmp_path / ".exp"
    result = CliRunner().invoke(
        app,
        ["build", "retail", "--traces", str(tmp_path / "t.jsonl"), "--root", str(root), *flags],
    )
    assert result.exit_code == 2
    assert "--learn requires --memory-recall ALIAS" in " ".join(result.output.split())
    assert not root.exists()
