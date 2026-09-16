"""Evidence retention deletes only owned bounded cycles and never follows links."""

import json
import os
from pathlib import Path

import pytest

from exp.optimize.workflows.traffic_learning.sources.retention import prune_evidence
from exp.simulation.claas.partition import ClaasSourceSplit, split_experiences
from exp.simulation.claas.partition_test import source_batch


@pytest.fixture
def split() -> ClaasSourceSplit:
    """Use genuine immutable source partitions without provider execution."""
    return split_experiences(source_batch(), seed="fixed", held_out_fraction=0.5)


def _cycle(directory: Path, number: int, split: ClaasSourceSplit | None = None) -> Path:
    """Create an owned cycle journal with deterministic age and optional source evidence."""
    path = directory / "cycles" / f"{number:032x}"
    path.mkdir(parents=True)
    (path / "state.json").write_text(json.dumps({"cycle_id": path.name, "stage": "complete"}))
    if split is not None:
        (path / "context.json").write_text(
            json.dumps({"workflow": "traffic-learning-v1", "split": split.model_dump(mode="json")})
        )
    (path / "practice").mkdir()
    (path / "practice/rollout.json").write_text('{"private":"synthetic source derivative"}')
    os.utime(path, ns=(number * 1_000_000_000, number * 1_000_000_000))
    return path


def _sources(split: ClaasSourceSplit) -> set[str]:
    """Extract original experience identities instead of response or group identifiers."""
    return {item.experience_id for item in (*split.fit, *split.held_out)}


def test_expired_sources_remove_complete_cycle_but_leave_checkpoint_owners(
    tmp_path: Path, split: ClaasSourceSplit
) -> None:
    """Raw split and derived trajectories disappear when any source loses retention."""
    path = _cycle(tmp_path, 1, split)
    for name in ("evaluation.json", "registry.json", "partitions.json"):
        (tmp_path / name).write_text("owned elsewhere")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/adapter").write_bytes(b"weights")
    receipt = prune_evidence(tmp_path, _sources(split) - {split.fit[0].experience_id})
    assert receipt.deleted_cycle_ids == (path.name,)
    assert not path.exists()
    assert (tmp_path / "checkpoints/adapter").read_bytes() == b"weights"
    assert (tmp_path / "evaluation.json").read_text() == "owned elsewhere"
    assert "source" not in receipt.model_dump_json()
    assert prune_evidence(tmp_path, set()).deleted_cycle_ids == ()


def test_newest_cycle_bound_preserves_foreign_directories(
    tmp_path: Path, split: ClaasSourceSplit
) -> None:
    """Count pruning removes only recognized generated UUID directories, oldest first."""
    cycles = tuple(_cycle(tmp_path, number, split) for number in range(1, 5))
    foreign = tmp_path / "cycles/customer-notes"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep")
    uuid_without_journal = tmp_path / "cycles" / ("f" * 32)
    uuid_without_journal.mkdir()
    (uuid_without_journal / "data").write_text("keep")
    receipt = prune_evidence(tmp_path, _sources(split), maximum_cycles=2)
    assert receipt.deleted_cycle_ids == tuple(path.name for path in cycles[:2])
    assert all(path.exists() for path in cycles[2:])
    assert (foreign / "notes.txt").read_text() == "keep"
    assert (uuid_without_journal / "data").read_text() == "keep"
    if os.name == "posix":
        assert tmp_path.stat().st_mode & 0o777 == 0o700


def test_prepared_journal_without_split_still_obeys_count_bound(tmp_path: Path) -> None:
    """An interrupted initial write cannot leave unlimited empty generated cycles."""
    old, newest = _cycle(tmp_path, 1), _cycle(tmp_path, 2)
    receipt = prune_evidence(tmp_path, set(), maximum_cycles=1)
    assert receipt.deleted_cycle_ids == (old.name,)
    assert newest.exists()


@pytest.mark.parametrize(
    "location", ["application", "cycles", "cycle", "nested-file", "nested-directory"]
)
def test_symlinks_fail_before_deleting_owned_or_foreign_content(
    tmp_path: Path, split: ClaasSourceSplit, location: str
) -> None:
    """Every deletion candidate is inspected before mutations can reach a link target."""
    application = tmp_path / "application"
    cycle = _cycle(application, 1, split)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("foreign data")
    if location == "application":
        selected = tmp_path / "application-link"
        selected.symlink_to(application, target_is_directory=True)
    else:
        selected = application
        if location == "cycles":
            (application / "cycles").rename(application / "real-cycles")
            (application / "cycles").symlink_to(
                application / "real-cycles", target_is_directory=True
            )
        elif location == "cycle":
            (application / "cycles" / ("f" * 32)).symlink_to(outside, target_is_directory=True)
        elif location == "nested-file":
            (cycle / "practice/link").symlink_to(outside / "keep")
        else:
            (cycle / "practice/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        prune_evidence(selected, set())
    assert (outside / "keep").read_text() == "foreign data"
    assert (
        application / ("real-cycles" if location == "cycles" else "cycles") / cycle.name
    ).exists()


def test_mismatched_journal_identity_rejects_whole_prune(
    tmp_path: Path, split: ClaasSourceSplit
) -> None:
    """Malformed UUID directories cannot cause valid evidence to be partially deleted."""
    first, bad = _cycle(tmp_path, 1, split), _cycle(tmp_path, 2, split)
    (bad / "state.json").write_text(json.dumps({"cycle_id": "f" * 32, "stage": "complete"}))
    with pytest.raises(ValueError, match="identity differs"):
        prune_evidence(tmp_path, set())
    assert first.exists() and bad.exists()


@pytest.mark.parametrize("maximum", [0, -1, True, 10001])
def test_invalid_bound_rejects_before_creating_application(tmp_path: Path, maximum: int) -> None:
    """Retention requires a finite positive count rather than guessing deletion intent."""
    application = tmp_path / "absent"
    with pytest.raises(ValueError, match="maximum_cycles"):
        prune_evidence(application, set(), maximum)
    assert not application.exists()
