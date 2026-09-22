"""Terminal measurements preserve unavailable and positive operating costs."""

import pytest

from exp.cli.evaluation.view import _number


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "unavailable"), (0, "$0.0000"), (0.000016, "$0.000016"), (1e-9, "$1e-09")],
)
def test_cost_display_does_not_round_positive_usage_to_free(
    value: float | None, expected: str
) -> None:
    """A cheap measured rollout remains distinguishable from zero or unavailable usage."""
    assert _number(value, "$") == expected
