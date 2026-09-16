"""Strict explicit feedback and episode-finalization requests for continual learning."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from exp.common.claas.contracts import ClaasScope, Identifier
from exp.common.core.artifacts import ContractModel


class FeedbackRequest(ContractModel):
    """Caller-authored feedback for exactly one response or finalized episode.

    Scope ownership comes from the authenticated gateway key. A caller names an application,
    never a user identity. Omitted scores stay absent and are not interpreted as zero or failure.
    """

    application_id: Identifier
    feedback_id: Identifier
    response_id: Identifier | None = None
    episode_id: Identifier | None = None
    text: str | None = Field(default=None, min_length=1, max_length=65_536)
    reward: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False, strict=True)
    success: bool | None = Field(default=None, strict=True)

    @field_validator("application_id", "feedback_id", "response_id", "episode_id")
    @classmethod
    def _require_identifiers(cls, value: str | None) -> str | None:
        """Match the native identifier byte bound and reject blank identifiers."""
        if value is not None and (not value.strip() or len(value.encode()) > 512):
            raise ValueError("identifiers must be nonblank and at most 512 UTF-8 bytes")
        return value

    @model_validator(mode="after")
    def _require_one_target_and_feedback(self) -> FeedbackRequest:
        """Reject ambiguous targets and empty feedback without manufacturing defaults."""
        if (self.response_id is None) == (self.episode_id is None):
            raise ValueError("feedback must target exactly one response_id or episode_id")
        if self.text is None and self.reward is None and self.success is None:
            raise ValueError("feedback needs text, reward, or success")
        if self.text is not None and not self.text.strip():
            raise ValueError("feedback text must contain non-whitespace content")
        if self.text is not None and len(self.text.encode()) > 65_536:
            raise ValueError("feedback text must be at most 65536 UTF-8 bytes")
        if (
            self.reward is not None
            and self.success is not None
            and self.reward != float(self.success)
        ):
            raise ValueError(
                "paired feedback requires reward=1 for success or reward=0 for failure"
            )
        return self

    @property
    def training_reward(self) -> float | None:
        """Map explicit binary feedback to 0/1 without inventing a score for text alone."""
        if self.reward is not None:
            return self.reward
        return None if self.success is None else float(self.success)


class FeedbackRecord(ContractModel):
    """Durably acknowledged feedback with authenticated scope and receipt time."""

    schema_version: Literal[1] = 1
    scope: ClaasScope
    feedback: FeedbackRequest
    created_at: AwareDatetime


class FinalizeEpisodeRequest(ContractModel):
    """Explicit immutable membership and terminal state for one application episode."""

    application_id: Identifier
    episode_id: Identifier
    response_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1_024)
    status: Literal["completed", "failed", "abandoned"]

    @field_validator("application_id", "episode_id")
    @classmethod
    def _require_identifiers(cls, value: str) -> str:
        """Reject blank identifiers and enforce the native UTF-8 byte bound."""
        if not value.strip() or len(value.encode()) > 512:
            raise ValueError("identifiers must be nonblank and at most 512 UTF-8 bytes")
        return value

    @field_validator("response_ids")
    @classmethod
    def _require_unique_responses(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate membership instead of silently changing the episode."""
        if len(value) != len(set(value)):
            raise ValueError("episode response_ids must be unique")
        if any(not item.strip() or len(item.encode()) > 512 for item in value):
            raise ValueError("response identifiers must be nonblank and at most 512 UTF-8 bytes")
        return value


class FinalizedEpisode(ContractModel):
    """Caller-reported episode completion, with no inferred success assertion."""

    schema_version: Literal[1] = 1
    scope: ClaasScope
    episode: FinalizeEpisodeRequest
    finalized_at: AwareDatetime
