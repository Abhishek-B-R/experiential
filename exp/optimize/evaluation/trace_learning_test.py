"""The learning pass stops cleanly at its attempt budget and continues on rerun."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.held_out_fidelity_test import (
    REVISION,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.trace_learning import (
    fit_cases,
    learn_from_fit_traces,
    learning_curve,
    learning_plan,
    learning_spec,
)


def test_budget_stops_between_traces_and_a_rerun_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace starts only when its whole ceiling fits; the next run picks up after it."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider)

    def learn(budget: int):  # noqa: ANN202 - local shorthand
        """Run the pass with one budget."""
        return learn_from_fit_traces(
            store,
            recall_alias="jev",
            maximum_attempts=budget,
            code_revision=REVISION,
            clock=lambda: datetime.now(UTC),
        )

    nothing = learn(1)
    assert nothing.learned == 0 and nothing.stopped is not None and not provider.requests[1:]
    assert learning_curve(store) is None
    one = learning_spec(
        store,
        learning_plan(store, "jev"),
        fit_cases(store)[0],
        deadline=datetime.now(UTC) + timedelta(hours=1),
    ).maximum_physical_attempts
    first = learn(one + one // 2)
    assert 0 < first.learned < first.traces and first.head_version == first.learned
    rest = learn(10_000)
    assert rest.stopped is None and rest.learned == rest.traces == len(fit_cases(store))
    curve = learning_curve(store)
    assert curve is not None and len(curve.points) == rest.traces
    assert [w.last for w in curve.halves][-1] == rest.traces
