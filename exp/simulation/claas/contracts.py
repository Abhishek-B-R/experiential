"""Source-bound scenarios, informative signals, and synthetic world transitions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.claas import ClaasScope, FeedbackRecord, FinalizedEpisode
from exp.common.core.artifacts import ContractModel, Sha256
from exp.common.models import ModelMessage, ModelRequest, ModelResponse
from exp.common.tasks import ToolSchema


class EvidenceReference(ContractModel):
    """A JSON pointer into one immutable captured experience, including its digest."""

    experience_id: str = Field(min_length=1)
    experience_sha256: Sha256
    pointer: str = Field(pattern=r"^/(request|response)(/.*)?$")


class ExperienceSignal(ContractModel):
    """An observed pattern useful for selection, never an inferred success label."""

    kind: Literal[
        "tool_error", "tool_recovery", "repeated_action", "truncation", "early_termination"
    ]
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    description: str = Field(min_length=1)


class ClaasScenario(ContractModel):
    """Only request-visible inputs that a policy may see at the start of practice."""

    scenario_id: str = Field(min_length=1)
    scope: ClaasScope
    partition: Literal["fit", "held_out"]
    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    tools: tuple[ToolSchema, ...] = ()
    sources: tuple[EvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_initial_inputs(self) -> ClaasScenario:
        """Keep observed answers and tool results outside the policy's seed prompt."""
        if any(message.role not in ("system", "user") for message in self.messages):
            raise ValueError("scenario seeds accept only initial system and user messages")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("scenario seeds need an initial user request")
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise ValueError("scenario tool names must be unique")
        return self


class MinedScenario(ContractModel):
    """A safe scenario with separate private selection evidence and unknown outcome."""

    scenario: ClaasScenario
    signals: tuple[ExperienceSignal, ...]
    outcome: Literal["unknown"] = "unknown"


class SyntheticObservation(ContractModel):
    """A simulated result linked to exactly one policy-emitted tool call."""

    call_id: str = Field(min_length=1, max_length=512)
    content: str = Field(max_length=65_536)
    is_error: bool = Field(strict=True)


class WorldTransition(ContractModel):
    """Strict hosted-model output with synthetic feedback kept out of policy inputs."""

    observations: tuple[SyntheticObservation, ...] = Field(max_length=64)
    user_message: str | None = Field(default=None, max_length=65_536)
    terminal: bool = Field(strict=True)
    feedback: str = Field(min_length=1, max_length=65_536)
    reward: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def _require_visible_progress(self) -> WorldTransition:
        """Require a next observation unless the synthetic episode has ended."""
        if not self.terminal and not self.observations and not self.user_message:
            raise ValueError("a nonterminal transition needs a tool result or user message")
        ids = tuple(observation.call_id for observation in self.observations)
        if len(ids) != len(set(ids)):
            raise ValueError("synthetic observations must have unique call IDs")
        return self


class WorldStep(ContractModel):
    """Replayable provider evidence that is always identified as synthetic."""

    index: int = Field(ge=0, strict=True)
    scenario_id: str
    sources: tuple[EvidenceReference, ...]
    request: ModelRequest
    response: ModelResponse
    transition: WorldTransition
    provenance: Literal["synthetic"] = "synthetic"


class SourceFeedback(ContractModel):
    """Private caller feedback about an observed response or explicitly finalized episode."""

    record: FeedbackRecord
    finalized_episode: FinalizedEpisode | None = None


class WorldEpisode(ContractModel):
    """A bounded practice episode, without claims about real task success."""

    scenario: ClaasScenario
    steps: tuple[WorldStep, ...]
    end_reason: Literal["world_terminal", "caller_ended", "limit", "error"]
    outcome: Literal["unverified"] = "unverified"
    source_feedback: tuple[SourceFeedback, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def _isolate_source_feedback(self) -> WorldEpisode:
        """Historical training feedback cannot enter held-out episode evidence."""
        if self.scenario.partition != "fit" and self.source_feedback:
            raise ValueError("source feedback is private practice evidence, not held-out input")
        return self
