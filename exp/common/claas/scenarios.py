"""Provider-independent scenario inputs and recorded environment interaction evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.claas.contracts import ClaasScope
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema


class Scenario(ContractModel):
    """An authored, imported, or generated task with private environment setup.

    Only messages and tools are shown to the student. Environment data can hold
    hidden state, deterministic fixtures, or source references; the core never
    interprets it or forwards it to a policy.
    """

    scenario_id: str = Field(min_length=1, max_length=512)
    scope: ClaasScope
    environment_id: str = Field(min_length=1, max_length=512)
    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    tools: tuple[ToolSchema, ...] = ()
    environment_data: JsonObject = Field(default_factory=dict)
    source_kind: Literal["simulation", "environment", "import"] = "environment"
    source_experience_ids: tuple[str, ...] = ()


class EnvironmentTransition(ContractModel):
    """Private learning feedback and public next state returned after one action."""

    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    terminal: bool
    reward: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    feedback: str | None = Field(default=None, min_length=1, max_length=65_536)


class EnvironmentStep(ContractModel):
    """One student action and its environment-owned transition."""

    action: AssistantAction
    transition: EnvironmentTransition


class EnvironmentEpisode(ContractModel):
    """Complete bounded environment transcript including explicit incomplete endings."""

    scenario: Scenario
    steps: tuple[EnvironmentStep, ...]
    end_reason: Literal["terminal", "step_limit", "failed"]
    evidence: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_terminal(self) -> EnvironmentEpisode:
        """Require terminal endings to match the final recorded transition."""
        if any(step.transition.terminal for step in self.steps[:-1]):
            raise ValueError("episode contains actions after a terminal transition")
        terminal = bool(self.steps and self.steps[-1].transition.terminal)
        if (self.end_reason == "terminal" and not terminal) or (
            self.end_reason == "step_limit" and terminal
        ):
            raise ValueError("episode ending differs from its recorded terminal transition")
        return self

    @property
    def messages(self) -> tuple[ModelMessage, ...]:
        """Return only the environment's visible transcript, excluding private feedback."""
        return self.steps[-1].transition.messages if self.steps else self.scenario.messages
