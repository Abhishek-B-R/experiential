"""Versioned cross-trajectory memory over the append-only experience library.

A lineage is a numbered sequence of immutable versions. Each learning version consumes one
distinct completed trajectory: its disclosed native replies become a source observation and
its session investigations become unresolved research. Neither becomes a verified rule.
Version identities are derived from the lineage and number, so a concurrent or changed
publication of the same number fails instead of forking the lineage.
"""

import json
from collections.abc import Sequence
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    Sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.reconstruction.library import (
    LibraryEntry,
    LibrarySnapshot,
    extend_library,
    resolve_library_entry,
)

_TYPE = "experience-memory-version"


class MemoryVersion(ArtifactEnvelope):
    """One frozen checkpoint of accumulated experience for a single simulator.

    Attributes:
        lineage: Caller-named memory sequence within one project.
        version: Number of consumed distinct trajectories; 0 is never persisted.
        world_model: Exact simulator whose predictions produced the research.
        selection_policy_sha256: Memory selector used to deliver these records.
        library: Experience library snapshot holding every admitted entry.
        consumed: Ordered digests of consumed trajectories, one per version.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    version: int = Field(ge=1, strict=True)
    world_model: ArtifactInput
    selection_policy_sha256: Sha256
    library: ArtifactInput
    consumed: tuple[Sha256, ...]


def memory_version_id(lineage: str, version: int) -> str:
    """Derive the only artifact identity a lineage version may occupy."""
    return stable_id("experience-memory", {"lineage": lineage, "version": version})


def read_memory_version(store: ArtifactStore, lineage: str, version: int) -> MemoryVersion:
    """Load one verified checkpoint, rejecting a different lineage or number."""
    identity = memory_version_id(lineage, version)
    saved = MemoryVersion.model_validate_json(store.read_bytes(identity, "memory.json"))
    if saved.lineage != lineage or saved.version != version:
        raise ValueError("memory version artifact differs from its lineage position")
    return saved


def memory_head(store: ArtifactStore, lineage: str) -> MemoryVersion | None:
    """Return the latest contiguous version, or None before any trajectory is consumed."""
    head = None
    while store.exists(memory_version_id(lineage, (head.version if head else 0) + 1)):
        head = read_memory_version(store, lineage, (head.version if head else 0) + 1)
    return head


def require_memory_scope(
    version: MemoryVersion | None, *, world_model: ArtifactInput, selection_policy_sha256: str
) -> None:
    """Refuse experience produced by another simulator or delivered by another selector."""
    if version is not None and (
        version.world_model != world_model
        or version.selection_policy_sha256 != selection_policy_sha256
    ):
        raise ValueError("memory lineage belongs to a different simulator or selector")


def memory_records(store: ArtifactStore, version: MemoryVersion | None) -> dict[str, JsonObject]:
    """Resolve active research, retaining superseded records only in the archive.

    Returns:
        Research records keyed by their original identity. Original trajectories are
        retained for provenance and counting but are not delivered as hypotheses.
    """
    if version is None:
        return {}
    snapshot = _library(store, version)
    research = {
        entry.entry_id: resolve_library_entry(store, entry)
        for entry in snapshot.entries
        if entry.kind == "research"
    }
    superseded: set[str] = set()
    for record in research.values():
        investigation = record.get("investigation") if isinstance(record, dict) else None
        if isinstance(investigation, dict):
            superseded.update(str(key) for key in investigation.get("supersedes", ()))
    return {
        key: value
        for key, value in research.items()
        if key not in superseded and isinstance(value, dict)
    }


def consumed_trajectory_digest(trajectory: JsonObject) -> Sha256:
    """Identify a completed trajectory by its public actions and disclosed outcomes only."""
    return sha256_json(trajectory)


def publish_trajectory(
    store: ArtifactStore,
    *,
    lineage: str,
    parent: MemoryVersion | None,
    world_model: ArtifactInput,
    selection_policy_sha256: Sha256,
    trajectory: JsonObject,
    research: Sequence[str],
    created_at: datetime,
    code_revision: str,
) -> tuple[MemoryVersion, bool]:
    """Consume one completed trajectory into the next lineage version.

    Args:
        store: Project artifact owner holding the session research records.
        lineage: Memory sequence to extend.
        parent: Pinned version the trajectory was predicted against, or None for the first.
        world_model: Simulator that produced the predictions.
        selection_policy_sha256: Selector that delivered parent memory.
        trajectory: Public actions plus disclosed native replies, without run identity.
        research: Session research record artifact IDs created from this trajectory.
        created_at: Aware publication time.
        code_revision: Publishing implementation identity.

    Returns:
        The new version and True, or the existing head and False when this exact
        trajectory was already consumed. Repeated draws never increase the count.
    """
    require_memory_scope(
        parent, world_model=world_model, selection_policy_sha256=selection_policy_sha256
    )
    digest = consumed_trajectory_digest(trajectory)
    head = memory_head(store, lineage)
    if head is not None and digest in head.consumed:
        return head, False
    if (head.version if head else 0) != (parent.version if parent else 0):
        raise ValueError("memory advanced after this trajectory was pinned")
    number = 1 if parent is None else parent.version + 1
    source_id = stable_id("consumed-trajectory", {"digest": digest})
    payload: JsonObject = {
        "trajectory": trajectory,
        "authority": "Disclosed native replies after prediction; not a behavioral rule",
    }
    if not store.exists(source_id):
        store.write_json(
            artifact_id=source_id,
            artifact_type="consumed-trajectory",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision
            ),
            files={"trajectory.json": payload},
        )
    entries: list[LibraryEntry] = [
        _entry(store, source_id, "trajectory.json", kind="trajectory", cue=digest)
    ]
    prior = set() if parent is None else {e.entry_id for e in _library(store, parent).entries}
    entries.extend(
        _entry(store, identity, "record.json", kind="research", cue=digest)
        for identity in research
        if identity not in prior
    )
    snapshot = extend_library(
        store,
        entries,
        parent=None if parent is None else _library(store, parent),
        created_at=created_at,
        code_revision=code_revision,
    )
    version = MemoryVersion(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(
            sorted(
                [artifact_input(store.read(snapshot.library_id).manifest)]
                + ([] if parent is None else [_input(store, lineage, parent.version)]),
                key=lambda row: row.artifact_id,
            )
        ),
        code_revision=code_revision,
        lineage=lineage,
        version=number,
        world_model=world_model,
        selection_policy_sha256=selection_policy_sha256,
        library=artifact_input(store.read(snapshot.library_id).manifest),
        consumed=(*(() if parent is None else parent.consumed), digest),
    )
    persisted, _ = store.write_or_replay(
        artifact_id=memory_version_id(lineage, number),
        artifact_type=_TYPE,
        envelope=version,
        envelope_path="memory.json",
        envelope_type=MemoryVersion,
        files={"memory.json": canonical_json_bytes(version)},
    )
    return persisted, True


def _entry(store: ArtifactStore, identity: str, path: str, *, kind: str, cue: str) -> LibraryEntry:
    """Reference one persisted record with its authority-preserving disposition."""
    return LibraryEntry(
        entry_id=identity,
        kind="trajectory" if kind == "trajectory" else "research",
        artifact=artifact_input(store.read(identity).manifest),
        path=path,
        payload_sha256=sha256_bytes(store.read_bytes(identity, path)),
        disposition="source" if kind == "trajectory" else "unresolved",
        cues=("trajectory:" + cue,),
    )


def _input(store: ArtifactStore, lineage: str, version: int) -> ArtifactInput:
    """Reference a persisted version as provenance for its child."""
    return artifact_input(store.read(memory_version_id(lineage, version)).manifest)


def _library(store: ArtifactStore, version: MemoryVersion) -> LibrarySnapshot:
    """Load the exact library snapshot pinned by a version."""
    if artifact_input(store.read(version.library.artifact_id).manifest) != version.library:
        raise ValueError("memory library manifest changed")
    raw: JsonValue = json.loads(store.read_bytes(version.library.artifact_id, "library.json"))
    return LibrarySnapshot.model_validate(raw)
