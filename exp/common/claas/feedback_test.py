"""Explicit feedback preserves missing values and refuses caller identity claims."""

import pytest

from exp.common.claas.feedback import FeedbackRequest, FinalizeEpisodeRequest


def test_text_feedback_does_not_create_a_numeric_or_binary_reward() -> None:
    """Missing reward and success remain unknown after protocol round trips."""
    feedback = FeedbackRequest(
        application_id="claims", feedback_id="fb-1", response_id="r-1", text="Ask for the receipt."
    )
    assert feedback.reward is None
    assert feedback.success is None
    assert FeedbackRequest.model_validate_json(feedback.model_dump_json()) == feedback


def test_false_and_zero_are_valid_explicit_feedback() -> None:
    """Falsy values still count as feedback because their presence is explicit."""
    assert (
        FeedbackRequest(
            application_id="claims",
            feedback_id="fb-1",
            response_id="r-1",
            reward=0.0,
            success=False,
        ).success
        is False
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"user_id": "other"},
        {"reward": True},
        {"success": 0},
        {"text": " "},
        {"reward": float("inf")},
    ],
)
def test_feedback_rejects_identity_overrides_and_malformed_signals(
    extra: dict[str, str | float | bool],
) -> None:
    """Caller data cannot claim another user's scope or coerce a feedback signal."""
    with pytest.raises(ValueError):
        FeedbackRequest.model_validate(
            {"application_id": "claims", "feedback_id": "fb-1", "response_id": "r-1", **extra}
        )


def test_feedback_requires_exactly_one_target_and_at_least_one_signal() -> None:
    """Empty payloads and ambiguous associations are invalid."""
    with pytest.raises(ValueError, match="needs text"):
        FeedbackRequest(application_id="claims", feedback_id="fb-1", response_id="r-1")
    with pytest.raises(ValueError, match="exactly one"):
        FeedbackRequest(
            application_id="claims",
            feedback_id="fb-1",
            response_id="r-1",
            episode_id="e-1",
            text="x",
        )


def test_episode_membership_is_explicit_and_unique() -> None:
    """Finalization cannot repeat a response to inflate an episode's evidence."""
    with pytest.raises(ValueError, match="unique"):
        FinalizeEpisodeRequest(
            application_id="claims",
            episode_id="e-1",
            response_ids=("r-1", "r-1"),
            status="completed",
        )


@pytest.mark.parametrize("success,reward", [(True, 1.0), (False, 0.0)])
def test_training_reward_uses_only_explicit_binary_signal(success: bool, reward: float) -> None:
    """The training projection maps explicit binary feedback without altering raw data."""
    feedback = FeedbackRequest(
        application_id="app", feedback_id="fb", response_id="r", success=success
    )
    assert feedback.reward is None
    assert feedback.training_reward == reward


@pytest.mark.parametrize("reward,success", [(-1.0, False), (0.0, True), (2.0, None)])
def test_conflicting_and_out_of_range_scores_are_rejected(
    reward: float, success: bool | None
) -> None:
    """Training can consume the scalar without hidden clipping or conflicting labels."""
    with pytest.raises(ValueError):
        FeedbackRequest(
            application_id="app", feedback_id="fb", response_id="r", reward=reward, success=success
        )
