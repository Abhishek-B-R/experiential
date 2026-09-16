"""Freeze evaluation before practice and preserve response partitions across learning cycles."""

from pathlib import Path
from typing import Literal

from exp.common.claas import Experience
from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.core.files import write_bytes_atomic
from exp.common.models import ModelSnapshot
from exp.optimize.workflows.traffic_learning.evaluation import (
    DEFAULT_RUBRIC,
    EvaluationManifest,
    freeze_evaluation,
)
from exp.optimize.workflows.traffic_learning.sources.holdouts import load_evaluation_holdouts
from exp.simulation.claas.partition import (
    ClaasSourceSplit,
    exclude_response_groups,
    split_experiences,
)

_MAXIMUM_STATE_BYTES = 64_000_000


def _read_owned_state(path: Path, label: str) -> bytes | None:
    """Reject linked, nonregular, or oversized state before parsing or replacing it."""
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular owned file")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"{label} must be a regular owned file")
    if path.stat().st_size > _MAXIMUM_STATE_BYTES:
        raise ValueError(f"{label} exceeds its 64 MB bound")
    with path.open("rb") as handle:
        payload = handle.read(_MAXIMUM_STATE_BYTES + 1)
    if len(payload) > _MAXIMUM_STATE_BYTES:
        raise ValueError(f"{label} exceeds its 64 MB bound")
    return payload


class PartitionLedger(ContractModel):
    """Durable response assignments prevent late episode links from crossing partitions."""

    seed: str
    assignments: dict[str, Literal["fit", "held_out"]]


def prepare_evidence(
    *,
    directory: Path,
    experiences: tuple[Experience, ...],
    world_model: ModelSnapshot,
    judge_model: ModelSnapshot,
    minimum_tasks: int,
    rubric: str = DEFAULT_RUBRIC,
) -> tuple[ClaasSourceSplit, EvaluationManifest]:
    """Persist stable split assignments and a held-out manifest from retained sources.

    The caller holds the application cycle lock. Incompatible provider choices or
    new links that would move a historical response fail before any provider call.
    A retired source rotates the evaluation cohort before any practice dispatch.
    Each paired comparison uses one frozen cohort. Collect more independent
    traffic when either stable partition is empty.
    """
    experiences = exclude_response_groups(experiences, load_evaluation_holdouts(directory))
    seed = "claas-local-v1"
    ledger_path = directory / "partitions.json"
    ledger_payload = _read_owned_state(ledger_path, "partition ledger")
    ledger = (
        PartitionLedger.model_validate_json(ledger_payload)
        if ledger_payload is not None
        else PartitionLedger(seed=seed, assignments={})
    )
    if ledger.seed != seed:
        raise ValueError("partition seed changed; initialize a new application")
    path = directory / "evaluation.json"
    evaluation_payload = _read_owned_state(path, "evaluation manifest")
    previous_manifest = (
        EvaluationManifest.model_validate_json(evaluation_payload)
        if evaluation_payload is not None
        else None
    )
    if previous_manifest is not None:
        retained = {item.experience_id: sha256_json(item) for item in experiences}
        if any(
            retained.get(item.experience_id) != sha256_json(item)
            for task in previous_manifest.tasks
            for item in task.grounding
        ):
            # Retirement does not depend on having enough new traffic to replace the cohort.
            path.unlink()
            previous_manifest = None
    split = split_experiences(experiences, seed=seed)
    assignments = dict(ledger.assignments)
    for name, sources in (("fit", split.fit), ("held_out", split.held_out)):
        for item in sources:
            previous = assignments.get(item.response_id)
            if previous is not None and previous != name:
                raise ValueError(
                    "new episode links would move previously partitioned responses; "
                    "use a new application for the corrected episode grouping"
                )
            assignments[item.response_id] = "fit" if name == "fit" else "held_out"
    save_manifest = previous_manifest is None
    if previous_manifest is not None:
        manifest = previous_manifest
        if (
            manifest.scope != split.scope
            or manifest.world_model != world_model
            or manifest.judge_model != judge_model
            or manifest.rubric != rubric
        ):
            raise ValueError("frozen evaluation identity changed; initialize a new application")
        held_responses = {item.response_id for task in manifest.tasks for item in task.grounding}
        if any(item.response_id in held_responses for item in split.fit):
            raise ValueError("fit traffic overlaps the frozen held-out evaluation")
    else:
        manifest = freeze_evaluation(
            split, world_model=world_model, judge_model=judge_model, rubric=rubric
        )
    if len(manifest.tasks) < minimum_tasks:
        raise ValueError(
            f"held-out evaluation has {len(manifest.tasks)} tasks; collect at least "
            f"{minimum_tasks} independently held-out workflows before training"
        )
    ledger_payload = (
        PartitionLedger(seed=seed, assignments=assignments).model_dump_json() + "\n"
    ).encode()
    evaluation_payload = (manifest.model_dump_json() + "\n").encode()
    if len(ledger_payload) > _MAXIMUM_STATE_BYTES:
        raise ValueError("partition ledger exceeds its 64 MB bound")
    if len(evaluation_payload) > _MAXIMUM_STATE_BYTES:
        raise ValueError("evaluation manifest exceeds its 64 MB bound")
    write_bytes_atomic(ledger_path, ledger_payload, follow_symlinks=False)
    if save_manifest:
        write_bytes_atomic(path, evaluation_payload, follow_symlinks=False)
    return split, manifest
