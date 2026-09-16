"""Private, bounded retention for source-derived CLaaS cycle evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from exp.common.core.artifacts import ContractModel
from exp.simulation.claas.partition import ClaasSourceSplit

_CYCLE_ID = re.compile(r"^[0-9a-f]{32}$")
_MAXIMUM_SPLIT_BYTES = 268_435_456


class EvidencePruneReceipt(ContractModel):
    """Content-free identities of generated cycles removed by one retention pass."""

    deleted_cycle_ids: tuple[str, ...] = ()


class _CycleIdentity(BaseModel):
    """Recognize owned cycle journals without depending on orchestration implementation."""

    model_config = ConfigDict(extra="ignore")
    cycle_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    stage: Literal["prepared", "practice", "training", "evaluation", "complete", "failed"]


@dataclass(frozen=True)
class _Cycle:
    """Validated generated directory and source-retention status."""

    path: Path
    modified_ns: int
    expired: bool


def prune_evidence(
    directory: Path, retained_source_ids: set[str], maximum_cycles: int = 32
) -> EvidencePruneReceipt:
    """Remove expired generated cycles and bound the newest retained cycle journals.

    The caller holds the application cycle lock throughout this operation. Raw
    cycle evidence is disposable; model checkpoints, registry receipts, partition
    assignments, and the frozen evaluation manifest have separate lifecycle owners.
    A source ID means an original ``Experience.experience_id``, not a response ID.

    Args:
        directory: One application directory, made private on POSIX before inspection.
        retained_source_ids: Original capture identities still eligible for source use.
        maximum_cycles: Maximum generated cycle directories to retain after cleanup.

    Returns:
        Deleted UUID cycle identities, without source IDs, content, or exception text.

    Raises:
        ValueError: Bounds, symlinks, journals, or saved source manifests are invalid.
        OSError: Private permissions, source inspection, or directory removal failed.
    """
    if type(maximum_cycles) is not int or not 1 <= maximum_cycles <= 10_000:
        raise ValueError("maximum_cycles must be an integer between 1 and 10000")
    if any(
        not isinstance(identity, str) or not identity.strip() for identity in retained_source_ids
    ):
        raise ValueError("retained_source_ids must contain nonblank experience IDs")
    _reject_symlink(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        directory.chmod(0o700)
    root = directory / "cycles"
    _reject_symlink(root)
    if not root.exists():
        return EvidencePruneReceipt()
    cycles: list[_Cycle] = []
    for path in root.iterdir():
        _reject_symlink(path)
        if not _CYCLE_ID.fullmatch(path.name) or not path.is_dir():
            continue
        _validate_tree(path)
        state_path = path / "state.json"
        # A UUID name alone does not establish ownership of arbitrary user files.
        if not state_path.is_file():
            continue
        if state_path.stat().st_size > 1_048_576:
            raise ValueError("cycle state exceeds its bound; restore the generated journal")
        identity = _CycleIdentity.model_validate_json(state_path.read_bytes())
        if identity.cycle_id != path.name:
            raise ValueError(
                "cycle journal identity differs from its directory; restore the journal"
            )
        split_path = path / "context.json"
        expired = False
        if split_path.exists():
            if split_path.stat().st_size > _MAXIMUM_SPLIT_BYTES:
                raise ValueError("cycle split exceeds 256 MiB; restore bounded source evidence")
            context = json.loads(split_path.read_bytes())
            if context.get("workflow") != "traffic-learning-v1":
                cycles.append(_Cycle(path, path.stat().st_mtime_ns, False))
                continue
            split = ClaasSourceSplit.model_validate(context.get("split"))
            source_ids = {item.experience_id for item in (*split.fit, *split.held_out)}
            expired = not source_ids.issubset(retained_source_ids)
        cycles.append(_Cycle(path, path.stat().st_mtime_ns, expired))
    retained = sorted(
        (item for item in cycles if not item.expired),
        key=lambda item: (item.modified_ns, item.path.name),
        reverse=True,
    )
    remove = {item.path for item in cycles if item.expired}
    remove.update(item.path for item in retained[maximum_cycles:])
    # Validate all candidates before deleting any, then recheck each tree at dispatch.
    for path in sorted(remove):
        _validate_tree(path)
    deleted: list[str] = []
    for path in sorted(remove):
        _reject_symlink(path)
        shutil.rmtree(path)
        deleted.append(path.name)
    return EvidencePruneReceipt(deleted_cycle_ids=tuple(deleted))


def _reject_symlink(path: Path) -> None:
    """Refuse links before reading, changing permissions, or removing any evidence."""
    if path.is_symlink():
        raise ValueError("evidence retention refuses symlinks; restore regular owned paths")


def _validate_tree(path: Path) -> None:
    """Never traverse nested links when recognizing or deleting generated evidence."""
    _reject_symlink(path)
    for parent, directories, files in os.walk(path, followlinks=False):
        for name in (*directories, *files):
            _reject_symlink(Path(parent) / name)
