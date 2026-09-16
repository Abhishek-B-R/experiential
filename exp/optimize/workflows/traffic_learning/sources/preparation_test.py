"""Frozen evaluation identity and cross-cycle partition regression coverage."""

from pathlib import Path

import pytest

from exp.common.claas import Experience
from exp.optimize.workflows.traffic_learning.sources.holdouts import (
    begin_evaluation_holdout,
    reserve_evaluation_holdout,
)
from exp.optimize.workflows.traffic_learning.sources.preparation import (
    PartitionLedger,
    prepare_evidence,
)
from exp.simulation.claas.harness_test import model_snapshot
from exp.simulation.claas.partition_test import source_batch


def traffic() -> tuple[Experience, ...]:
    """Provide enough independent workflows for both deterministic partitions."""
    original = source_batch()[0]
    return tuple(
        original.model_copy(
            update={
                "experience_id": f"experience-{index}",
                "response_id": f"response-{index}",
                "episode_id": None,
                "parent_response_id": None,
            }
        )
        for index in range(32)
    )


def test_evaluation_stays_frozen_when_more_traffic_arrives(tmp_path: Path) -> None:
    """New fit content cannot mutate the previously reserved evaluation tasks."""
    first, manifest = prepare_evidence(
        directory=tmp_path,
        experiences=traffic()[:16],
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    second, saved = prepare_evidence(
        directory=tmp_path,
        experiences=traffic(),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    assert len(second.fit) > len(first.fit)
    assert saved == manifest
    assert (tmp_path / "evaluation.json").read_text().strip() == manifest.model_dump_json()


def test_late_partition_change_fails_before_overwriting_manifest(tmp_path: Path) -> None:
    """An existing response cannot enter fit after belonging to held-out evaluation."""
    split, manifest = prepare_evidence(
        directory=tmp_path,
        experiences=traffic(),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    path = tmp_path / "partitions.json"
    ledger = PartitionLedger.model_validate_json(path.read_bytes())
    assignments = dict(ledger.assignments)
    assignments[split.fit[0].response_id] = "held_out"
    path.write_text(ledger.model_copy(update={"assignments": assignments}).model_dump_json())
    with pytest.raises(ValueError, match="previously partitioned"):
        prepare_evidence(
            directory=tmp_path,
            experiences=traffic(),
            world_model=model_snapshot(),
            judge_model=model_snapshot(),
            minimum_tasks=1,
        )
    assert (tmp_path / "evaluation.json").read_text().strip() == manifest.model_dump_json()


def test_insufficient_evaluation_does_not_freeze_small_manifest(tmp_path: Path) -> None:
    """Collecting more independent traffic can satisfy the initial evidence floor."""
    with pytest.raises(ValueError, match="collect at least"):
        prepare_evidence(
            directory=tmp_path,
            experiences=traffic(),
            world_model=model_snapshot(),
            judge_model=model_snapshot(),
            minimum_tasks=100,
        )
    assert not (tmp_path / "evaluation.json").exists()


def test_retired_sources_rotate_evaluation_before_new_provider_calls(tmp_path: Path) -> None:
    """A prior frozen cohort cannot keep disclosing expired or evicted traffic."""
    _, previous = prepare_evidence(
        directory=tmp_path,
        experiences=traffic(),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    changed = tuple(
        item.model_copy(
            update={
                "experience_id": "new-" + item.experience_id,
                "response_id": "new-" + item.response_id,
            }
        )
        for item in traffic()
    )
    _, current = prepare_evidence(
        directory=tmp_path,
        experiences=changed,
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    assert current.digest != previous.digest
    assert all(
        item.experience_id.startswith("new-") for task in current.tasks for item in task.grounding
    )


def test_expired_evaluation_is_removed_when_new_cohort_cannot_be_built(tmp_path: Path) -> None:
    """An empty retained buffer cannot leave expired raw grounding in the application manifest."""
    prepare_evidence(
        directory=tmp_path,
        experiences=traffic(),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    with pytest.raises(ValueError):
        prepare_evidence(
            directory=tmp_path,
            experiences=(),
            world_model=model_snapshot(),
            judge_model=model_snapshot(),
            minimum_tasks=1,
        )
    assert not (tmp_path / "evaluation.json").exists()


@pytest.mark.parametrize("relationship", ["parent", "episode", "provenance"])
def test_evaluation_holdouts_exclude_entire_linked_groups(
    tmp_path: Path, relationship: str
) -> None:
    """A reserved response excludes linked peers before splitting or synthesis."""

    original = traffic()
    first, second = original[:2]
    if relationship == "parent":
        second = second.model_copy(update={"parent_response_id": first.response_id})
    elif relationship == "episode":
        first = first.model_copy(update={"episode_id": "reserved-evaluation"})
        second = second.model_copy(update={"episode_id": "reserved-evaluation"})
    else:
        second = second.model_copy(
            update={
                "provenance": second.provenance.model_copy(
                    update={"source_experience_ids": (first.experience_id,)}
                )
            }
        )
    reserve_evaluation_holdout(tmp_path, (second.response_id,))
    split, manifest = prepare_evidence(
        directory=tmp_path,
        experiences=(first, second, *original[2:]),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    excluded = {first.response_id, second.response_id}
    assert not excluded.intersection(item.response_id for item in split.fit + split.held_out)
    assert not excluded.intersection(
        item.response_id for task in manifest.tasks for item in task.grounding
    )
    ledger = PartitionLedger.model_validate_json((tmp_path / "partitions.json").read_bytes())
    assert not excluded.intersection(ledger.assignments)


def test_new_evaluation_reservation_retires_existing_synthetic_cohort(tmp_path: Path) -> None:
    """An explicit evaluation reservation also removes the source from synthetic evaluation."""

    split, previous = prepare_evidence(
        directory=tmp_path,
        experiences=traffic(),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    reserved = split.held_out[0].response_id
    reserve_evaluation_holdout(tmp_path, (reserved,))
    _, current = prepare_evidence(
        directory=tmp_path,
        experiences=traffic(),
        world_model=model_snapshot(),
        judge_model=model_snapshot(),
        minimum_tasks=1,
    )
    assert current.digest != previous.digest
    assert reserved not in {item.response_id for task in current.tasks for item in task.grounding}


@pytest.mark.parametrize("already_frozen", [False, True])
def test_pending_evaluation_blocks_preparation_before_partition_or_manifest_changes(
    tmp_path: Path, already_frozen: bool
) -> None:
    """Both CLI preflight and cycles fail closed on a restart-visible unresolved evaluation."""

    if already_frozen:
        prepare_evidence(
            directory=tmp_path,
            experiences=traffic(),
            world_model=model_snapshot(),
            judge_model=model_snapshot(),
            minimum_tasks=1,
        )
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    begin_evaluation_holdout(tmp_path, run_id="interrupted-run", request_id="evaluation:task:0:0")
    pending = (tmp_path / "evaluation-holdouts.json").read_bytes()
    with pytest.raises(ValueError, match="unresolved held-out evaluation"):
        prepare_evidence(
            directory=tmp_path,
            experiences=traffic(),
            world_model=model_snapshot(),
            judge_model=model_snapshot(),
            minimum_tasks=1,
        )
    assert (tmp_path / "evaluation-holdouts.json").read_bytes() == pending
    after = {
        path.name: path.read_bytes()
        for path in tmp_path.glob("*.json")
        if path.name != "evaluation-holdouts.json"
    }
    assert after == before
