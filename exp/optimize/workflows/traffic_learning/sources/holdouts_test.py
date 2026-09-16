"""External evaluation reservations cannot relabel previously fitted traffic."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from exp.optimize.workflows.traffic_learning.sources import holdouts
from exp.optimize.workflows.traffic_learning.sources.holdouts import (
    acknowledge_evaluation_holdout,
    begin_evaluation_holdout,
    load_evaluation_holdouts,
    reserve_evaluation_holdout,
)


def test_holdout_reservations_are_persistent_idempotent_and_content_free(tmp_path: Path) -> None:
    """Later partial evaluation attempts extend the ledger without duplicate memberships."""
    assert load_evaluation_holdouts(tmp_path) == frozenset()
    reserve_evaluation_holdout(tmp_path, ("second", "first", "second"))
    reserve_evaluation_holdout(tmp_path, ("third", "first"))
    assert load_evaluation_holdouts(tmp_path) == frozenset({"first", "second", "third"})
    state = json.loads((tmp_path / "evaluation-holdouts.json").read_text())
    assert set(state) == {"schema_version", "response_ids", "pending"}
    assert state["pending"] is None


def test_reservation_refuses_existing_fit_assignment_atomically(tmp_path: Path) -> None:
    """Existing training exposure prevents a false claim of independent held-out data."""
    reserve_evaluation_holdout(tmp_path, ("held",))
    (tmp_path / "partitions.json").write_text(
        json.dumps({"seed": "s", "assignments": {"used": "fit"}})
    )
    with pytest.raises(ValueError, match="already assigned to fit"):
        reserve_evaluation_holdout(tmp_path, ("fresh", "used"))
    assert load_evaluation_holdouts(tmp_path) == frozenset({"held"})


def test_holdout_file_symlink_is_never_followed(tmp_path: Path) -> None:
    """Owned ledger writes cannot be redirected to another file."""
    target = tmp_path / "target"
    target.write_text("keep")
    (tmp_path / "evaluation-holdouts.json").symlink_to(target)
    with pytest.raises(ValueError, match="regular owned file"):
        reserve_evaluation_holdout(tmp_path, ("response",))
    assert target.read_text() == "keep"


def test_pending_dispatch_survives_abrupt_process_exit(tmp_path: Path) -> None:
    """A crashed caller leaves a durable learning block even without an acknowledged response."""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; from pathlib import Path; "
                "from exp.optimize.workflows.traffic_learning.sources.holdouts "
                "import begin_evaluation_holdout; "
                "begin_evaluation_holdout(Path(sys.argv[1]), run_id='run-one', "
                "request_id='evaluation:task:1:0'); os._exit(17)"
            ),
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[5],
        check=False,
        timeout=30,
    )
    assert child.returncode == 17
    with pytest.raises(ValueError, match="unresolved held-out evaluation"):
        load_evaluation_holdouts(tmp_path)
    with pytest.raises(ValueError, match="do not delete"):
        begin_evaluation_holdout(tmp_path, run_id="run-two", request_id="evaluation:task:2:0")
    with pytest.raises(ValueError, match="unresolved held-out evaluation"):
        reserve_evaluation_holdout(tmp_path, ("unrelated-response",))


def test_acknowledgment_atomically_excludes_capture_and_clears_pending(tmp_path: Path) -> None:
    """Each restart-visible state is either blocked or contains the acknowledged exclusion."""
    reserve_evaluation_holdout(tmp_path, ("prior-response",))
    begin_evaluation_holdout(tmp_path, run_id="run", request_id="evaluation:task:1:0")
    with pytest.raises(ValueError, match="unresolved held-out evaluation"):
        load_evaluation_holdouts(tmp_path)
    acknowledge_evaluation_holdout(
        tmp_path, run_id="run", request_id="evaluation:task:1:0", response_id="native-response"
    )
    assert load_evaluation_holdouts(tmp_path) == frozenset({"prior-response", "native-response"})
    assert json.loads((tmp_path / "evaluation-holdouts.json").read_bytes())["pending"] is None


@pytest.mark.parametrize("run_id, request_id", [("other", "request"), ("run", "other")])
def test_acknowledgment_must_match_pending_run_and_request(
    tmp_path: Path, run_id: str, request_id: str
) -> None:
    """A response from another attempt cannot remove the unresolved request marker."""
    begin_evaluation_holdout(tmp_path, run_id="run", request_id="request")
    original = (tmp_path / "evaluation-holdouts.json").read_bytes()
    with pytest.raises(ValueError, match="differs from the pending"):
        acknowledge_evaluation_holdout(
            tmp_path, run_id=run_id, request_id=request_id, response_id="response"
        )
    assert (tmp_path / "evaluation-holdouts.json").read_bytes() == original


def test_failed_acknowledgment_write_keeps_learning_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storage error after native capture never clears the pre-dispatch marker."""
    begin_evaluation_holdout(tmp_path, run_id="run", request_id="request")
    original = (tmp_path / "evaluation-holdouts.json").read_bytes()

    def fail_write(path: Path, payload: str) -> None:
        """Simulate a failed durable replacement before its atomic rename."""
        raise OSError("storage unavailable")

    monkeypatch.setattr(holdouts, "write_text_atomic", fail_write)
    with pytest.raises(OSError, match="storage unavailable"):
        acknowledge_evaluation_holdout(
            tmp_path, run_id="run", request_id="request", response_id="native-response"
        )
    assert (tmp_path / "evaluation-holdouts.json").read_bytes() == original
    with pytest.raises(ValueError, match="unresolved held-out evaluation"):
        load_evaluation_holdouts(tmp_path)
