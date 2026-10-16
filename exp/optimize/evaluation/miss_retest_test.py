"""Miss retest: learned traces the simulator got wrong, replayed on the final memory."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.optimize.evaluation.held_out_fidelity_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _import_id,
    build_project,
    episode,
)
from exp.optimize.evaluation.trace_learning import (
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    recorded_plan,
)
from exp.optimize.evaluation.trace_learning_test import memory_delivered


def retest(root: Path, *flags: str) -> tuple[int, str]:
    """Invoke ``exp eval PROJECT --retest-misses 5`` and return its exit code and output."""
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--retest-misses", "5", "--non-interactive"]
        + list(flags),
    )
    return result.exit_code, " ".join(result.output.split())


def test_a_missed_turn_is_fixed_on_the_final_memory_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the first trace missed (the learnable tool); the final memory reproduces it."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(
        tmp_path, monkeypatch, provider, "--learn", "--memory-recall", "jev"
    )
    learned, predicted = len(provider.requests), len(provider.predictions())
    own = learning_cases(store, "all")[0].task.lineage_group_id  # the trace that missed

    code, dry = retest(root, "--dry-run")
    assert code == 0, dry
    assert f"1 of {USERS} learned traces had a miss · 3 turns" in dry
    assert "Prepared without provider calls" in dry and len(provider.requests) == learned

    code, output = retest(root, "--yes")
    assert code == 0, output
    assert "Miss retest 3/3 replies exact (100%)" in output
    assert "exact before learning 2/3 (67%) → after 3/3 (100%)" in output
    assert "fixed 1 of 1 missed turns · 0 regressed" in output
    assert "Test-then-train fidelity" in output
    purchased = len(provider.requests)
    assert purchased > learned
    # Learning hid the trace's own task; the retest reads what memory learned from it.
    retested = provider.predictions()[predicted:]
    assert retested and all(own in memory_delivered(prediction) for prediction in retested)

    code, replay = retest(root, "--verbose")
    assert code == 0, replay
    assert "Saved results. No provider calls." in replay and "Retested trace" in replay
    assert len(provider.requests) == purchased


def test_retest_needs_learning_and_a_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused clearly, before any call, with nothing learned or nothing missed."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    root, _, _ = build_project(tmp_path, monkeypatch, provider)
    code, output = retest(root)
    assert code == 2 and "--learn --memory-recall ALIAS" in output

    other = tmp_path / "learned"
    other.mkdir()
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    root, _, _ = build_project(other, monkeypatch, provider, "--learn", "--memory-recall", "jev")
    sent = len(provider.requests)
    code, output = retest(root)
    assert code == 2 and "there is no miss to retest" in output
    assert len(provider.requests) == sent


@pytest.mark.parametrize(
    "flags",
    [
        ["--held-out-fidelity", "2"],
        ["--models", "a"],
        ["--fidelity", "spec.json"],
    ],
)
def test_retest_cannot_combine_with_other_modes(tmp_path: Path, flags: list[str]) -> None:
    """One evaluation mode per command."""
    code, output = retest(tmp_path / ".exp", *flags)
    assert code == 2 and "cannot combine" in output


def test_a_cancelled_preflight_never_hides_learned_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the default --learn records nothing; the --hold-out memory stays active."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    declined = CliRunner().invoke(
        app,
        ["build", PROJECT, "--import-id", _import_id(store), "--root", str(root)]
        + ["--non-interactive", "--learn", "--memory-recall", "jev"],
    )
    assert "learned" not in declined.output, declined.output
    assert not recorded_plan(store, "all"), "a preflight records no plan"
    outcome = learn_from_traces(
        store,
        recall_alias="jev",
        scope="fit",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )
    assert learning_plan(store).scope == "fit"
    curve = learning_curve(store)
    assert curve is not None and len(curve.points) == outcome.learned
    code, output = retest(root, "--dry-run")
    assert code == 0, output
    assert f"of {outcome.learned} learned traces had a miss" in output
