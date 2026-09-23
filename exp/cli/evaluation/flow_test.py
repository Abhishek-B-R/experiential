"""Project navigation keeps build and provider setup outside the evaluation command."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.evaluation import flow
from exp.cli.shared.picker import PickerOption, PickerResult
from exp.optimize.evaluation.runs_test import _twenty_scenarios


def test_project_home_only_offers_evaluation_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built project starts at New evaluation without import or provider setup controls."""
    project, _, _ = _twenty_scenarios(tmp_path)

    def choose(console: Console, *, title: str, options: Sequence[PickerOption]) -> PickerResult:
        """Assert the menu exposes only meaningful actions for this empty evaluation history."""
        del console, title
        assert [(option.value, option.label) for option in options] == [
            ("new", "New evaluation"),
            ("exit", "Back"),
        ]
        return PickerResult(values=("new",))

    monkeypatch.setattr(flow, "choose_one", choose)
    assert flow._project_screen(project) is None
