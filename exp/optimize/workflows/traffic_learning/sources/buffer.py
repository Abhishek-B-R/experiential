"""Bounded durable traffic inputs for repeated CLaaS preparation cycles.

Call ``refresh_buffer(directory, database_path, scope, maximum_experiences)`` before
preparing a cycle. Each refresh consumes one bounded page per source stream. Explicit
finalization changes the projected grouping, never the captured payload. Cross-cycle
fit/held-out assignment belongs to the caller's persistent response partition ledger.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.claas import ClaasScope, Experience
from exp.common.claas.feedback import FeedbackRecord, FinalizedEpisode
from exp.common.core.artifacts import ContractModel, Sha256
from exp.common.core.files import write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.runtime.claas.feedback import FeedbackStore
from exp.runtime.claas.store import ExperienceStore

_MAXIMUM_BUFFER_BYTES = 268_435_456


class BufferCursors(ContractModel):
    """Independent monotonic positions within one bound native database."""

    captures: int = Field(default=0, strict=True, ge=0)
    feedback: int = Field(default=0, strict=True, ge=0)
    episodes: int = Field(default=0, strict=True, ge=0)


@dataclass(frozen=True)
class SourceBuffer:
    """Retained originals and complete inputs eligible for cycle preparation.

    ``experiences`` contains standalone calls and complete Responses ancestry, with
    episode IDs projected only from complete durable finalization receipts. Feedback
    and episode receipts target only these eligible inputs. A completed receipt does
    not imply success. ``captures`` preserves original protocol and provenance fields.
    """

    captures: tuple[Experience, ...]
    experiences: tuple[Experience, ...]
    feedback: tuple[FeedbackRecord, ...]
    episodes: tuple[FinalizedEpisode, ...]
    cursors: BufferCursors


class _BufferState(ContractModel):
    """Atomically persisted source records and their acknowledged consumer positions."""

    schema_version: Literal[1] = 1
    database_identity: Sha256
    scope: ClaasScope
    cursors: BufferCursors = Field(default_factory=BufferCursors)
    captures: tuple[Experience, ...] = ()
    feedback: tuple[FeedbackRecord, ...] = ()
    episodes: tuple[FinalizedEpisode, ...] = ()


def refresh_buffer(
    directory: Path,
    database_path: Path,
    scope: ClaasScope,
    maximum_experiences: int,
) -> SourceBuffer:
    """Consume bounded new pages and atomically persist a retention-checked buffer.

    Args:
        directory: Application state directory; a private source-buffer child is owned here.
        database_path: Existing native capture database, bound by resolved path and inode.
        scope: Explicit local application identity, never inferred from captured content.
        maximum_experiences: Retained count ceiling per stream, between 1 and 1,000,000.

    Returns:
        Original retained captures, eligible projected inputs, and durable cursors.

    Raises:
        ValueError: Scope, database identity, immutable evidence, or size bounds changed.
        OSError: Reading source evidence or atomically publishing consumer state failed.
    """
    if type(maximum_experiences) is not int or not 1 <= maximum_experiences <= 1_000_000:
        raise ValueError("maximum_experiences must be an integer between 1 and 1000000")
    database_path = database_path.resolve(strict=True)
    identity = _database_identity(database_path)
    private = directory / "source-buffer"
    if private.is_symlink():
        raise ValueError(
            "source-buffer directory must not be a symlink; choose a private directory"
        )
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    private.chmod(0o700)
    path = private / "state.json"
    if path.is_symlink():
        raise ValueError("source buffer state must not be a symlink; restore its regular file")
    with file_write_lock(path, what="CLaaS source buffer"):
        state = _load(path, identity, scope)
        limit = min(maximum_experiences, 1000)
        captures = ExperienceStore(database_path, scope).read_after(
            state.cursors.captures, limit=limit
        )
        store = FeedbackStore(database_path, scope)
        feedback = store.read_after(state.cursors.feedback, limit=limit)
        episodes = store.episodes_after(state.cursors.episodes, limit=limit)
        state = _BufferState(
            database_identity=identity,
            scope=scope,
            cursors=BufferCursors(
                captures=captures[-1].sequence if captures else state.cursors.captures,
                feedback=feedback[-1].sequence if feedback else state.cursors.feedback,
                episodes=episodes[-1].sequence if episodes else state.cursors.episodes,
            ),
            captures=_append(
                state.captures,
                tuple(row.experience for row in captures),
                lambda item: item.experience_id,
                maximum_experiences,
            ),
            feedback=_append(
                state.feedback,
                tuple(row.record for row in feedback),
                lambda item: item.feedback.feedback_id,
                maximum_experiences,
            ),
            episodes=_append(
                state.episodes,
                tuple(row.record for row in episodes),
                lambda item: item.episode.episode_id,
                maximum_experiences,
            ),
        )
        state = _retain(database_path, state)
        if _database_identity(database_path) != identity:
            raise ValueError(
                "capture database was replaced during refresh; use a new buffer directory"
            )
        result = _project(state)
        payload = state.model_dump_json().encode()
        if len(payload) > _MAXIMUM_BUFFER_BYTES:
            raise ValueError("source buffer exceeds 256 MiB; reduce maximum_experiences and retry")
        write_bytes_atomic(path, payload, follow_symlinks=False)
        path.chmod(0o600)
        return result


def _database_identity(path: Path) -> str:
    """Bind cursors to a file identity so replacement cannot silently skip new history."""
    info = path.stat()
    return hashlib.sha256(f"{path}\0{info.st_dev}\0{info.st_ino}".encode()).hexdigest()


def _load(path: Path, identity: str, scope: ClaasScope) -> _BufferState:
    """Reject stale bindings and bound deserialization before reading retained content."""
    if not path.exists():
        return _BufferState(database_identity=identity, scope=scope)
    if path.stat().st_size > _MAXIMUM_BUFFER_BYTES:
        raise ValueError("source buffer exceeds 256 MiB; restore a bounded buffer")
    state = _BufferState.model_validate_json(path.read_bytes())
    if state.database_identity != identity or state.scope != scope:
        raise ValueError("source buffer scope or database changed; choose a new buffer directory")
    if any(item.scope != scope for item in (*state.captures, *state.feedback, *state.episodes)):
        raise ValueError("source buffer contains evidence from another scope; restore its state")
    return state


def _append[T: ContractModel](
    previous: tuple[T, ...], new: tuple[T, ...], key: Callable[[T], str], maximum: int
) -> tuple[T, ...]:
    """Preserve insertion order, reject rewritten evidence, and retain the newest bound."""
    records = {key(item): item for item in previous}
    if len(records) != len(previous):
        raise ValueError("source buffer contains duplicate evidence IDs; restore its state")
    for item in new:
        known = records.setdefault(key(item), item)
        if known != item:
            raise ValueError(
                "durable source evidence changed for an existing ID; restore the source"
            )
    return tuple(records.values())[-maximum:]


def _retain(database_path: Path, state: _BufferState) -> _BufferState:
    """Recheck cached source lifetime and immutable payloads in one read transaction."""
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True, timeout=1.0)
    try:
        connection.execute("BEGIN")
        captures = _retained(
            connection,
            state.scope,
            "claas_experiences",
            "experience_id",
            state.captures,
            lambda item: item.experience_id,
            "1",
        )
        episodes = _retained(
            connection,
            state.scope,
            "claas_episodes",
            "episode_id",
            state.episodes,
            lambda item: item.episode.episode_id,
            _EPISODE_RETAINED,
        )
        feedback = _retained(
            connection,
            state.scope,
            "claas_feedback",
            "feedback_id",
            state.feedback,
            lambda item: item.feedback.feedback_id,
            _FEEDBACK_RETAINED,
        )
        return state.model_copy(
            update={"captures": captures, "episodes": episodes, "feedback": feedback}
        )
    finally:
        connection.close()


def _retained[T: ContractModel](
    connection: sqlite3.Connection,
    scope: ClaasScope,
    table: str,
    identifier: str,
    records: tuple[T, ...],
    key: Callable[[T], str],
    condition: str,
) -> tuple[T, ...]:
    """Read only bounded known IDs, checking source deletion, expiry, and payload equality."""
    result: list[T] = []
    for start in range(0, len(records), 400):
        chunk = records[start : start + 400]
        placeholders = ",".join("?" for _ in chunk)
        rows = dict(
            connection.execute(
                f"SELECT {identifier},payload FROM {table} AS r "
                "WHERE user_id=? AND application_id=? "
                f"AND {identifier} IN ({placeholders}) AND expires_at>unixepoch() AND {condition}",
                (scope.user_id, scope.application_id, *(key(item) for item in chunk)),
            ).fetchall()
        )
        for item in chunk:
            payload = rows.get(key(item))
            if payload is not None:
                if type(item).model_validate_json(payload) != item:
                    raise ValueError(
                        "retained source payload changed; restore immutable source evidence"
                    )
                result.append(item)
    return tuple(result)


def _project(state: _BufferState) -> SourceBuffer:
    """Exclude incomplete ancestry and episodes, then attach only durable memberships."""
    by_response = {item.response_id: item for item in state.captures}
    if len(by_response) != len(state.captures):
        raise ValueError("source buffer contains duplicate response IDs; restore source evidence")
    membership: dict[str, str] = {}
    excluded: set[str] = set()
    for receipt in state.episodes:
        episode = receipt.episode
        for response_id in episode.response_ids:
            known = membership.setdefault(response_id, episode.episode_id)
            if known != episode.episode_id:
                raise ValueError(
                    "response belongs to conflicting finalized episodes; restore source evidence"
                )
        if not set(episode.response_ids).issubset(by_response):
            excluded.update(episode.response_ids)
    # Evaluate chains iteratively, avoiding recursion limits for long retained episodes.
    valid: dict[str, bool] = {}
    for response_id in by_response:
        trail: set[str] = set()
        current: str | None = response_id
        while current is not None and current not in valid and current not in trail:
            item = by_response.get(current)
            if item is None or current in excluded:
                break
            trail.add(current)
            current = item.parent_response_id if item.protocol == "responses" else None
        complete = current is None or valid.get(current, False)
        valid.update((member, complete) for member in trail)
    excluded.update(response_id for response_id in by_response if not valid.get(response_id, False))
    # Walk dependency edges once, including all members of an excluded episode.
    children: dict[str, list[str]] = {}
    for item in state.captures:
        if item.protocol == "responses" and item.parent_response_id is not None:
            children.setdefault(item.parent_response_id, []).append(item.response_id)
    episode_members = {
        item.episode.episode_id: item.episode.response_ids for item in state.episodes
    }
    queue = deque(excluded)
    seen_episodes: set[str] = set()
    while queue:
        response_id = queue.popleft()
        dependents = list(children.get(response_id, ()))
        episode_id = membership.get(response_id)
        if episode_id is not None and episode_id not in seen_episodes:
            seen_episodes.add(episode_id)
            dependents.extend(episode_members[episode_id])
        for dependent in dependents:
            if dependent not in excluded:
                excluded.add(dependent)
                queue.append(dependent)
    experiences = tuple(
        item.model_copy(update={"episode_id": membership.get(item.response_id)}, deep=True)
        for item in state.captures
        if item.response_id not in excluded
    )
    eligible = {item.response_id for item in experiences}
    episodes = tuple(
        item for item in state.episodes if set(item.episode.response_ids).issubset(eligible)
    )
    episode_ids = {item.episode.episode_id for item in episodes}
    feedback = tuple(
        item
        for item in state.feedback
        if item.feedback.response_id in eligible or item.feedback.episode_id in episode_ids
    )
    return SourceBuffer(state.captures, experiences, feedback, episodes, state.cursors)


# Cached metadata is still governed by the native source lifetime, including eviction
# before its stored expiry. Conditions use only fixed internal SQL identifiers.
_EPISODE_RETAINED = """NOT EXISTS (
 SELECT 1 FROM claas_episode_members m WHERE m.user_id=r.user_id
 AND m.application_id=r.application_id AND m.episode_id=r.episode_id AND NOT EXISTS (
 SELECT 1 FROM claas_experiences e WHERE e.user_id=m.user_id
 AND e.application_id=m.application_id AND e.response_id=m.response_id
 AND e.expires_at>unixepoch()))"""
_FEEDBACK_RETAINED = """((r.response_id IS NOT NULL AND EXISTS (
 SELECT 1 FROM claas_experiences e WHERE e.user_id=r.user_id
 AND e.application_id=r.application_id AND e.response_id=r.response_id
 AND e.expires_at>unixepoch())) OR (r.episode_id IS NOT NULL AND EXISTS (
 SELECT 1 FROM claas_episodes p WHERE p.user_id=r.user_id
 AND p.application_id=r.application_id AND p.episode_id=r.episode_id
 AND p.expires_at>unixepoch() AND NOT EXISTS (
 SELECT 1 FROM claas_episode_members m WHERE m.user_id=p.user_id
 AND m.application_id=p.application_id AND m.episode_id=p.episode_id AND NOT EXISTS (
 SELECT 1 FROM claas_experiences e WHERE e.user_id=m.user_id
 AND e.application_id=m.application_id AND e.response_id=m.response_id
 AND e.expires_at>unixepoch())))))"""
