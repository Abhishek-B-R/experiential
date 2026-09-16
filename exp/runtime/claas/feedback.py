"""Read-only application-scoped consumption of durable feedback and finalized episodes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exp.common.claas import ClaasScope
from exp.common.claas.feedback import FeedbackRecord, FinalizedEpisode


@dataclass(frozen=True)
class FeedbackRow:
    """A durable feedback record and its monotonic database consumer cursor."""

    sequence: int
    record: FeedbackRecord


@dataclass(frozen=True)
class EpisodeRow:
    """An immutable episode association and its monotonic database consumer cursor."""

    sequence: int
    record: FinalizedEpisode


class FeedbackStore:
    """Consume retained explicit feedback without write authority over captured traffic."""

    def __init__(self, database_path: Path, scope: ClaasScope) -> None:
        """Bind an existing capture database and one explicit local application scope."""
        self._path = database_path.resolve()
        self._scope = scope

    def read_after(self, sequence: int = 0, *, limit: int = 100) -> tuple[FeedbackRow, ...]:
        """Read at most one page of unexpired, still-grounded feedback in sequence order."""
        result = tuple(
            FeedbackRow(sequence=index, record=FeedbackRecord.model_validate_json(payload))
            for index, payload in self._read("claas_feedback", sequence, limit)
        )
        if any(
            row.record.scope != self._scope
            or row.record.feedback.application_id != self._scope.application_id
            for row in result
        ):
            raise ValueError("feedback payload scope differs from its durable partition")
        return result

    def episodes_after(self, sequence: int = 0, *, limit: int = 100) -> tuple[EpisodeRow, ...]:
        """Read a bounded page of finalized episodes whose member evidence is retained."""
        result = tuple(
            EpisodeRow(sequence=index, record=FinalizedEpisode.model_validate_json(payload))
            for index, payload in self._read("claas_episodes", sequence, limit)
        )
        if any(
            row.record.scope != self._scope
            or row.record.episode.application_id != self._scope.application_id
            for row in result
        ):
            raise ValueError("episode payload scope differs from its durable partition")
        return result

    def _read(
        self, table: Literal["claas_feedback", "claas_episodes"], sequence: int, limit: int
    ) -> tuple[tuple[int, str], ...]:
        """Apply expiration and source-retention checks within one read transaction."""
        if sequence < 0 or not 1 <= limit <= 1000:
            raise ValueError("sequence must be nonnegative and limit must be between 1 and 1000")
        condition = _FEEDBACK_RETAINED if table == "claas_feedback" else _EPISODE_RETAINED
        connection = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
            return tuple(
                connection.execute(
                    f"SELECT sequence,payload FROM {table} AS r "
                    "WHERE r.user_id=? AND r.application_id=? AND r.sequence>? "
                    f"AND r.expires_at>unixepoch() AND {condition} ORDER BY sequence LIMIT ?",
                    (self._scope.user_id, self._scope.application_id, sequence, limit),
                ).fetchall()
            )
        finally:
            connection.close()


_EPISODE_RETAINED = """NOT EXISTS (
  SELECT 1 FROM claas_episode_members m WHERE m.user_id=r.user_id
  AND m.application_id=r.application_id AND m.episode_id=r.episode_id AND NOT EXISTS (
    SELECT 1 FROM claas_experiences e WHERE e.user_id=m.user_id
    AND e.application_id=m.application_id AND e.response_id=m.response_id
    AND e.expires_at>unixepoch()))"""
_FEEDBACK_RETAINED = """(
  (r.response_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM claas_experiences e WHERE e.user_id=r.user_id
    AND e.application_id=r.application_id AND e.response_id=r.response_id
    AND e.expires_at>unixepoch())) OR
  (r.episode_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM claas_episodes p WHERE p.user_id=r.user_id
    AND p.application_id=r.application_id AND p.episode_id=r.episode_id
    AND p.expires_at>unixepoch() AND NOT EXISTS (
      SELECT 1 FROM claas_episode_members m WHERE m.user_id=p.user_id
      AND m.application_id=p.application_id AND m.episode_id=p.episode_id AND NOT EXISTS (
        SELECT 1 FROM claas_experiences e WHERE e.user_id=m.user_id
        AND e.application_id=m.application_id AND e.response_id=m.response_id
        AND e.expires_at>unixepoch())))))"""
