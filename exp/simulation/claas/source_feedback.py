"""Exact observed-source binding for private practice feedback."""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.claas import Experience, FeedbackRecord, FinalizedEpisode
from exp.simulation.claas.contracts import ClaasScenario, SourceFeedback


def validate_source_feedback(
    scenario: ClaasScenario,
    grounding: Sequence[Experience],
    feedback: Sequence[SourceFeedback],
) -> tuple[SourceFeedback, ...]:
    """Bind caller labels to named observed responses without assigning new action rewards.

    Episode labels require the explicit immutable finalization receipt and every
    named response in the source grounding. Captured episode IDs alone do not
    prove membership. Additional source responses never inherit the label.
    """
    if not feedback:
        return ()
    if scenario.partition != "fit":
        raise ValueError("source feedback is permitted only for fit practice")
    if len(feedback) > 128:
        raise ValueError("source feedback limit exceeded; select at most 128 records")
    by_response = {item.response_id: item for item in grounding}
    if len(by_response) != len(grounding):
        raise ValueError("feedback grounding must have unique response IDs")
    records = tuple(SourceFeedback.model_validate_json(item.model_dump_json()) for item in feedback)
    if len({item.record.feedback.feedback_id for item in records}) != len(records):
        raise ValueError("source feedback IDs must be unique")
    for item in records:
        record, target = item.record, item.record.feedback
        if record.scope != scenario.scope or target.application_id != scenario.scope.application_id:
            raise ValueError("source feedback belongs to another user or application")
        if target.response_id is not None:
            if item.finalized_episode is not None:
                raise ValueError("response feedback must not carry unrelated episode membership")
            response_ids = (target.response_id,)
        else:
            finalized = item.finalized_episode
            if finalized is None:
                raise ValueError("episode feedback requires an explicit finalization receipt")
            if (
                finalized.scope != scenario.scope
                or finalized.episode.application_id != scenario.scope.application_id
                or finalized.episode.episode_id != target.episode_id
            ):
                raise ValueError("feedback target differs from its scoped finalized episode")
            if finalized.finalized_at > record.created_at:
                raise ValueError("episode feedback predates its finalization receipt")
            response_ids = finalized.episode.response_ids
        for identity in response_ids:
            source = by_response.get(identity)
            if source is None:
                raise ValueError("source feedback membership is missing from scenario grounding")
            if source.scope != scenario.scope or source.provenance.source_kind != "traffic":
                raise ValueError("source feedback requires observed traffic in the exact scope")
    return records


def select_source_feedback(
    grounding: Sequence[Experience],
    feedback: Sequence[FeedbackRecord],
    episodes: Sequence[FinalizedEpisode],
) -> tuple[SourceFeedback, ...]:
    """Select only exactly matched caller labels from one application's source buffer.

    Unmatched responses and incomplete episode memberships are left out. A
    matching captured episode ID alone never replaces the finalization receipt.
    Scope mismatches and contradictory receipt identities fail explicitly.
    """
    if not grounding:
        return ()
    scope = grounding[0].scope
    if any(item.scope != scope or item.provenance.source_kind != "traffic" for item in grounding):
        raise ValueError("feedback selection requires observed traffic in one exact scope")
    responses = {item.response_id for item in grounding}
    if len(responses) != len(grounding):
        raise ValueError("feedback selection requires unique source response IDs")
    by_episode: dict[str, FinalizedEpisode] = {}
    for finalized in episodes:
        if finalized.scope != scope or finalized.episode.application_id != scope.application_id:
            raise ValueError("episode receipt belongs to another user or application")
        if finalized.episode.episode_id in by_episode:
            raise ValueError("episode receipts must have unique immutable identities")
        by_episode[finalized.episode.episode_id] = finalized
    selected: list[SourceFeedback] = []
    identities: set[str] = set()
    for record in feedback:
        target = record.feedback
        if record.scope != scope or target.application_id != scope.application_id:
            raise ValueError("source feedback belongs to another user or application")
        if target.feedback_id in identities:
            raise ValueError("source feedback IDs must be unique")
        identities.add(target.feedback_id)
        finalized = None
        if target.response_id is not None:
            if target.response_id not in responses:
                continue
        else:
            finalized = by_episode.get(target.episode_id or "")
            if finalized is None or not set(finalized.episode.response_ids).issubset(responses):
                continue
            if finalized.finalized_at > record.created_at:
                raise ValueError("episode feedback predates its finalization receipt")
        selected.append(SourceFeedback(record=record, finalized_episode=finalized))
        if len(selected) > 128:
            raise ValueError("source feedback limit exceeded; select at most 128 records")
    return tuple(selected)
