"""Persistent source cursors and grouping use actual bounded SQLite evidence."""

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.claas import ClaasScope, Experience, ExperienceProvenance
from exp.common.claas.feedback import (
    FeedbackRecord,
    FeedbackRequest,
    FinalizedEpisode,
    FinalizeEpisodeRequest,
)
from exp.optimize.workflows.traffic_learning.sources import buffer
from exp.optimize.workflows.traffic_learning.sources.buffer import BufferCursors, refresh_buffer

_SCOPE = ClaasScope(user_id="user", application_id="claims")
_NOW = datetime.now(UTC)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """Initialize the production reader schema without a provider or native worker."""
    path = tmp_path / "capture.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
          CREATE TABLE claas_experiences(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            experience_id UNIQUE,user_id,application_id,response_id,expires_at,payload);
          CREATE TABLE claas_feedback(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id,application_id,feedback_id,response_id,episode_id,expires_at,payload);
          CREATE TABLE claas_episodes(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id,application_id,episode_id,expires_at,payload);
          CREATE TABLE claas_episode_members(user_id,application_id,episode_id,response_id);
        """)
    return path


def _capture(
    path: Path,
    response_id: str,
    *,
    parent: str | None = None,
    scope: ClaasScope = _SCOPE,
    chat: bool = False,
) -> Experience:
    """Persist a complete original protocol exchange and return its exact contract."""
    item = Experience(
        experience_id=f"capture-{response_id}",
        response_id=response_id,
        parent_response_id=parent,
        scope=scope,
        protocol="chat_completions" if chat else "responses",
        captured_at=_NOW,
        request={"messages": [{"role": "user", "content": "same transcript"}]},
        response={"id": response_id},
        provenance=ExperienceProvenance(
            source_kind="traffic", source_id="gateway", model_id="model"
        ),
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO claas_experiences(experience_id,user_id,application_id,"
            "response_id,expires_at,payload) "
            "VALUES(?,?,?,?,unixepoch()+3600,?)",
            (
                item.experience_id,
                scope.user_id,
                scope.application_id,
                response_id,
                item.model_dump_json(),
            ),
        )
    return item


def _episode(path: Path, episode_id: str, members: tuple[str, ...]) -> FinalizedEpisode:
    """Write a durable explicit finalization receipt with its native membership index."""
    item = FinalizedEpisode(
        scope=_SCOPE,
        episode=FinalizeEpisodeRequest(
            application_id="claims", episode_id=episode_id, response_ids=members, status="completed"
        ),
        finalized_at=_NOW,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO claas_episodes(user_id,application_id,episode_id,expires_at,payload) "
            "VALUES('user','claims',?,unixepoch()+3600,?)",
            (episode_id, item.model_dump_json()),
        )
        connection.executemany(
            "INSERT INTO claas_episode_members VALUES('user','claims',?,?)",
            ((episode_id, member) for member in members),
        )
    return item


def _feedback(
    path: Path, response_id: str | None = None, episode_id: str | None = None
) -> FeedbackRecord:
    """Write text-only feedback so unknown numeric outcomes must survive buffering."""
    item = FeedbackRecord(
        scope=_SCOPE,
        feedback=FeedbackRequest(
            application_id="claims",
            feedback_id=f"feedback-{response_id or episode_id}",
            response_id=response_id,
            episode_id=episode_id,
            text="Check the address.",
        ),
        created_at=_NOW,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO claas_feedback(user_id,application_id,feedback_id,"
            "response_id,episode_id,expires_at,payload) "
            "VALUES('user','claims',?,?,?,unixepoch()+3600,?)",
            (item.feedback.feedback_id, response_id, episode_id, item.model_dump_json()),
        )
    return item


def test_cursor_resume_scope_and_newest_count_bound(database: Path, tmp_path: Path) -> None:
    """Repeated calls read one new page, never duplicate old evidence or another scope."""
    _capture(database, "other", scope=ClaasScope(user_id="neighbor", application_id="claims"))
    originals = tuple(_capture(database, str(index), chat=True) for index in range(5))
    directory = tmp_path / "state"
    first = refresh_buffer(directory, database, _SCOPE, 2)
    assert first.captures == originals[:2]
    assert first.cursors.captures == 3
    second = refresh_buffer(directory, database, _SCOPE, 2)
    assert second.captures == originals[2:4]
    third = refresh_buffer(directory, database, _SCOPE, 2)
    assert third.captures == originals[3:]
    assert refresh_buffer(directory, database, _SCOPE, 2) == third
    assert third.experiences == third.captures
    assert all(item.episode_id is None for item in third.experiences)
    assert (directory / "source-buffer").stat().st_mode & 0o777 == 0o700
    assert (directory / "source-buffer/state.json").stat().st_mode & 0o777 == 0o600


def test_late_finalization_projects_membership_and_preserves_originals(
    database: Path, tmp_path: Path
) -> None:
    """Finalization may arrive in a later cycle without rewriting capture provenance."""
    first = _capture(database, "first")
    second = _capture(database, "second", parent="first")
    before = refresh_buffer(tmp_path / "state", database, _SCOPE, 10)
    assert before.experiences == (first, second)
    receipt = _episode(database, "episode", ("first", "second"))
    signal = _feedback(database, episode_id="episode")
    after = refresh_buffer(tmp_path / "state", database, _SCOPE, 10)
    assert after.captures == (first, second)
    assert tuple(item.episode_id for item in after.experiences) == ("episode", "episode")
    assert after.episodes == (receipt,)
    assert after.feedback == (signal,)
    assert after.feedback[0].feedback.training_reward is None
    assert after.cursors == BufferCursors(captures=2, feedback=1, episodes=1)
    after.experiences[0].request["changed"] = True
    assert "changed" not in after.captures[0].request


def test_receipt_waits_for_complete_capture_page(database: Path, tmp_path: Path) -> None:
    """A finalization receipt consumed early stays available when the final capture arrives."""
    _capture(database, "unrelated")
    _capture(database, "first")
    _capture(database, "second", parent="first")
    _episode(database, "episode", ("first", "second"))
    first = refresh_buffer(tmp_path / "state", database, _SCOPE, 2)
    assert [item.response_id for item in first.experiences] == ["unrelated"]
    assert first.episodes == ()
    second = refresh_buffer(tmp_path / "state", database, _SCOPE, 2)
    assert [item.response_id for item in second.experiences] == ["first", "second"]
    assert second.episodes[0].episode.episode_id == "episode"


def test_missing_ancestry_cycles_and_partial_episodes_are_excluded(
    database: Path, tmp_path: Path
) -> None:
    """No prefix matching repairs incomplete Responses history or cyclic evidence."""
    _capture(database, "missing-parent-child", parent="missing")
    _capture(database, "cycle-a", parent="cycle-b")
    _capture(database, "cycle-b", parent="cycle-a")
    _capture(database, "same-episode")
    _capture(database, "descendant", parent="same-episode")
    _capture(database, "standalone-chat", chat=True)
    _episode(database, "broken", ("missing-parent-child", "same-episode"))
    _feedback(database, response_id="same-episode")
    result = refresh_buffer(tmp_path / "state", database, _SCOPE, 20)
    assert len(result.captures) == 6
    assert [item.response_id for item in result.experiences] == ["standalone-chat"]
    assert result.feedback == result.episodes == ()


@pytest.mark.parametrize(
    "operation",
    ["DELETE FROM claas_experiences", "UPDATE claas_experiences SET expires_at=unixepoch()-1"],
)
def test_cached_source_expiry_or_eviction_removes_dependent_evidence(
    database: Path, tmp_path: Path, operation: str
) -> None:
    """Retention applies to already buffered payloads even when source cursors are idle."""
    _capture(database, "response")
    _episode(database, "episode", ("response",))
    _feedback(database, response_id="response")
    _feedback(database, episode_id="episode")
    before = refresh_buffer(tmp_path / "state", database, _SCOPE, 10)
    assert len(before.feedback) == 2
    with sqlite3.connect(database) as connection:
        connection.execute(operation)
    after = refresh_buffer(tmp_path / "state", database, _SCOPE, 10)
    assert after.captures == after.experiences == after.feedback == after.episodes == ()
    assert after.cursors == before.cursors
    assert "Check the address." not in (tmp_path / "state/source-buffer/state.json").read_text()


def test_source_replacement_scope_change_and_mutation_fail_closed(
    database: Path, tmp_path: Path
) -> None:
    """A cursor cannot silently move to a new database, scope, or rewritten payload."""
    _capture(database, "response")
    directory = tmp_path / "state"
    refresh_buffer(directory, database, _SCOPE, 10)
    with pytest.raises(ValueError, match="scope or database"):
        refresh_buffer(
            directory, database, ClaasScope(user_id="other", application_id="claims"), 10
        )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE claas_experiences SET "
            "payload=replace(payload,'same transcript','changed content')"
        )
    with pytest.raises(ValueError, match="payload changed"):
        refresh_buffer(directory, database, _SCOPE, 10)
    replacement = tmp_path / "new.sqlite3"
    replacement.write_bytes(database.read_bytes())
    os.replace(replacement, database)
    with pytest.raises(ValueError, match="scope or database"):
        refresh_buffer(directory, database, _SCOPE, 10)


def test_failed_atomic_write_does_not_advance_durable_cursor(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry reads the same new row after persistence fails before publication."""
    _capture(database, "first")
    directory = tmp_path / "state"
    first = refresh_buffer(directory, database, _SCOPE, 10)
    _capture(database, "second")
    writer = buffer.write_bytes_atomic

    def fail(path: Path, payload: bytes, *, follow_symlinks: bool = True) -> None:
        """Inject a disk error before replacing the persisted consumer state."""
        raise OSError("disk full")

    monkeypatch.setattr(buffer, "write_bytes_atomic", fail)
    with pytest.raises(OSError, match="disk full"):
        refresh_buffer(directory, database, _SCOPE, 10)
    state = buffer._BufferState.model_validate_json(
        (directory / "source-buffer/state.json").read_bytes()
    )
    assert state.cursors == first.cursors
    monkeypatch.setattr(buffer, "write_bytes_atomic", writer)
    second = refresh_buffer(directory, database, _SCOPE, 10)
    assert second.cursors.captures == 2
    assert len(second.captures) == 2


def test_count_eviction_never_trains_a_truncated_response_chain(
    database: Path, tmp_path: Path
) -> None:
    """The oldest ancestor falling out of the bound excludes its retained descendants."""
    _capture(database, "first")
    _capture(database, "second", parent="first")
    _capture(database, "third", parent="second")
    directory = tmp_path / "state"
    assert len(refresh_buffer(directory, database, _SCOPE, 2).experiences) == 2
    result = refresh_buffer(directory, database, _SCOPE, 2)
    assert [item.response_id for item in result.captures] == ["second", "third"]
    assert result.experiences == ()


def test_conflicting_durable_episode_membership_is_rejected(database: Path, tmp_path: Path) -> None:
    """Corrupt membership must not silently pick an episode by receipt arrival order."""
    _capture(database, "response")
    _episode(database, "first", ("response",))
    _episode(database, "second", ("response",))
    with pytest.raises(ValueError, match="conflicting finalized episodes"):
        refresh_buffer(tmp_path / "state", database, _SCOPE, 10)
    assert not (tmp_path / "state/source-buffer/state.json").exists()


def test_buffer_byte_bound_fails_before_publishing_cursor(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversize input requires a lower count bound before any consumer state advances."""
    _capture(database, "response")
    monkeypatch.setattr(buffer, "_MAXIMUM_BUFFER_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds 256 MiB"):
        refresh_buffer(tmp_path / "state", database, _SCOPE, 10)
    assert not (tmp_path / "state/source-buffer/state.json").exists()
