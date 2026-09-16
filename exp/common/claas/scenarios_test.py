"""Scenario episode contracts reject inconsistent terminal evidence."""

import pytest

from exp.common.claas import ClaasScope
from exp.common.claas.scenarios import EnvironmentEpisode, Scenario
from exp.common.models import ModelMessage


def test_terminal_requires_recorded_terminal_transition() -> None:
    """A label alone cannot turn an empty episode into a completed task."""
    scenario = Scenario(
        scenario_id="authored",
        scope=ClaasScope(user_id="u", application_id="a"),
        environment_id="calculator-v1",
        messages=(ModelMessage(role="user", content="2+2"),),
    )
    with pytest.raises(ValueError, match="terminal transition"):
        EnvironmentEpisode(scenario=scenario, steps=(), end_reason="terminal")
