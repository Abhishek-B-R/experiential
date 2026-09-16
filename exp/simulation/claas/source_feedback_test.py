"""Historical feedback scope, finalized membership, privacy, and replay tests."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from exp.common.claas import (
    FeedbackRecord,
    FeedbackRequest,
    FinalizedEpisode,
    FinalizeEpisodeRequest,
)
from exp.common.models import AssistantAction
from exp.simulation.claas import (
    ClaasWorldModel,
    SourceDisclosure,
    SourceFeedback,
    mine_experiences,
    replay_episode,
    select_source_feedback,
)
from exp.simulation.claas.harness_test import (
    RecordingClient,
    final_transition,
    limits,
    model_snapshot,
)
from exp.simulation.claas.mining_test import make_experience
from exp.simulation.claas.source_feedback import validate_source_feedback


def response_feedback(
    *, response_id: str = "response-0", identity: str = "feedback-1"
) -> FeedbackRecord:
    """Create caller-authored feedback about a completed observed response."""
    source = make_experience()
    return FeedbackRecord(
        scope=source.scope,
        created_at=source.captured_at + timedelta(minutes=1),
        feedback=FeedbackRequest(
            application_id=source.scope.application_id,
            feedback_id=identity,
            response_id=response_id,
            text="PRIVATE CALLER FEEDBACK: the historical claim lookup used the wrong record.",
            reward=-1.0,
        ),
    )


def episode_feedback(*, responses: tuple[str, ...] = ("response-0",)) -> SourceFeedback:
    """Bind one explicit finalized episode to later caller feedback."""
    source = make_experience()
    record = FeedbackRecord(
        scope=source.scope,
        created_at=source.captured_at + timedelta(minutes=2),
        feedback=FeedbackRequest(
            application_id=source.scope.application_id,
            feedback_id="episode-feedback",
            episode_id="user-episode",
            text="The historical workflow failed.",
            success=False,
        ),
    )
    finalized = FinalizedEpisode(
        scope=source.scope,
        finalized_at=source.captured_at + timedelta(minutes=1),
        episode=FinalizeEpisodeRequest(
            application_id=source.scope.application_id,
            episode_id="user-episode",
            response_ids=responses,
            status="failed",
        ),
    )
    return SourceFeedback(record=record, finalized_episode=finalized)


def test_private_source_feedback_replays_without_copying_historical_reward() -> None:
    """The world sees the caller label; policy messages and new action rewards stay separate."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
    )
    feedback = (SourceFeedback(record=response_feedback()),)
    with world.open(scenario, grounding=(source,), source_feedback=feedback) as session:
        step = session.step(AssistantAction(content="The claim is pending."))
        assert step.transition.reward is None
        assert "PRIVATE CALLER FEEDBACK" not in str(session.messages)
        episode = session.end()
    assert episode.source_feedback == feedback
    payload = json.loads(client.requests[0].messages[-1].content or "{}")
    assert payload["source_feedback"][0]["record"]["feedback"]["reward"] == -1.0
    assert "Do not copy a historical reward" in (client.requests[0].messages[0].content or "")
    assert replay_episode(episode, grounding=(source,), limits=limits()) == episode
    changed = episode.model_copy(update={"source_feedback": ()})
    with pytest.raises(ValueError, match="replay request differs"):
        replay_episode(changed, grounding=(source,), limits=limits())


def test_feedback_is_rejected_for_held_out_without_a_provider_call() -> None:
    """Evaluation-only worlds cannot consume later practice feedback."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="held_out")[0].scenario
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        purpose="evaluation",
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
    )
    with pytest.raises(ValueError, match="only for fit"):
        world.open(
            scenario,
            grounding=(source,),
            source_feedback=(SourceFeedback(record=response_feedback()),),
        )
    assert not client.requests and world.reserved_calls == 0


def test_episode_feedback_requires_complete_explicit_membership() -> None:
    """An episode name or partial source window cannot prove complete feedback membership."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    valid = episode_feedback()
    assert validate_source_feedback(scenario, (source,), (valid,)) == (valid,)
    with pytest.raises(ValueError, match="finalization receipt"):
        validate_source_feedback(scenario, (source,), (SourceFeedback(record=valid.record),))
    with pytest.raises(ValueError, match="membership is missing"):
        validate_source_feedback(
            scenario, (source,), (episode_feedback(responses=("response-0", "missing")),)
        )
    assert valid.finalized_episode is not None
    wrong = valid.model_copy(
        update={
            "finalized_episode": valid.finalized_episode.model_copy(
                update={
                    "episode": valid.finalized_episode.episode.model_copy(
                        update={"episode_id": "other"}
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="scoped finalized episode"):
        validate_source_feedback(scenario, (source,), (wrong,))


def test_selection_excludes_unmatched_records_and_keeps_exact_episode_receipts() -> None:
    """The per-scenario selector does not attach unrelated or partially grounded labels."""
    source = make_experience()
    direct = response_feedback()
    unmatched = response_feedback(response_id="different", identity="other-feedback")
    episode = episode_feedback()
    assert episode.finalized_episode is not None
    selected = select_source_feedback(
        (source,), (direct, unmatched, episode.record), (episode.finalized_episode,)
    )
    assert selected == (SourceFeedback(record=direct), episode)
    partial = episode_feedback(responses=("response-0", "response-1"))
    assert partial.finalized_episode is not None
    assert select_source_feedback((source,), (partial.record,), (partial.finalized_episode,)) == ()
    assert select_source_feedback((source,), (episode.record,), ()) == ()


def test_cross_scope_feedback_rejects_even_when_target_is_unmatched() -> None:
    """A buffer from another user cannot be silently mixed into this application."""
    source = make_experience()
    record = response_feedback(response_id="other").model_copy(
        update={
            "scope": source.scope.model_copy(update={"user_id": "another-user"}),
        }
    )
    with pytest.raises(ValueError, match="another user"):
        select_source_feedback((source,), (record,), ())
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    with pytest.raises(ValueError, match="another user"):
        validate_source_feedback(scenario, (source,), (SourceFeedback(record=record),))


def test_high_feedback_volume_selects_a_bounded_recent_practice_window() -> None:
    """A valid 256-record buffer can open practice while preserving the latest 128 labels."""
    source = make_experience()
    records = tuple(response_feedback(identity=f"feedback-{index}") for index in range(256))
    selected = select_source_feedback((source,), records, ())
    assert tuple(item.record for item in selected) == records[-128:]
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
    )
    with world.open(scenario, grounding=(source,), source_feedback=selected) as session:
        assert session.end().source_feedback == selected
    assert not client.requests


def test_bounded_selection_still_rejects_invalid_records_outside_the_window() -> None:
    """Truncating valid labels cannot hide a cross-scope or duplicate earlier record."""
    source = make_experience()
    records = tuple(response_feedback(identity=f"feedback-{index}") for index in range(129))
    foreign = records[0].model_copy(
        update={"scope": source.scope.model_copy(update={"user_id": "another-user"})}
    )
    with pytest.raises(ValueError, match="another user"):
        select_source_feedback((source,), (foreign, *records[1:]), ())
    with pytest.raises(ValueError, match="unique"):
        select_source_feedback((source,), (*records, records[0]), ())
