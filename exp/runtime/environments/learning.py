"""Asynchronous episode transitions and private feedback for continual learning.

These sessions consume complete assistant actions and own terminal state, scalar/text
feedback, and explicit asynchronous cleanup. The execute-only EnvironmentRuntime in
interface.py instead opens a synchronous context manager for individual ToolCall
observations. An application bridging the two contracts supplies its own episode
termination and scoring policy.
"""

from __future__ import annotations

from typing import Literal, Protocol

from exp.common.claas.scenarios import EnvironmentTransition, Scenario
from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction


class EnvironmentSession(Protocol):
    """One freshly reset task instance, owned until explicit close completes."""

    async def step(self, action: AssistantAction) -> EnvironmentTransition:
        """Execute an action and return visible state plus separate learning feedback."""
        ...

    async def close(self, reason: Literal["terminal", "step_limit", "failed"]) -> JsonObject:
        """Release resources and return retained environment-specific evidence."""
        ...


class Environment(Protocol):
    """A reusable factory that resets a separate environment for every policy episode."""

    @property
    def environment_id(self) -> str:
        """Return the immutable environment implementation and configuration identity."""
        ...

    async def open(self, scenario: Scenario) -> EnvironmentSession:
        """Reset one scenario without exposing its private setup to the policy."""
        ...
