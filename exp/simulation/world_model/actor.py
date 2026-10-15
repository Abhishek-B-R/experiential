"""Expose normal Experiential simulator sessions to the maintained whole-episode actor."""

from __future__ import annotations

import json
from types import TracebackType

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments import Observation
from exp.simulation.world_model.application import WorldModel, WorldModelSession


class WorldModelEnvironment:
    """Execute-only actor adapter, with simulator state kept behind its existing owner."""

    def __init__(self, world_model: WorldModel, task: TaskCase) -> None:
        """Bind the normal simulator and public task, without native reference access."""
        self._world, self._task = world_model, task
        self._session: WorldModelSession | None = None
        self._steps: list[WorkflowStep] = []
        self._failed = False

    def __enter__(self) -> WorldModelEnvironment:
        """Open a fresh candidate world using only the permitted initial context."""
        if self._session is not None or self._steps:
            raise ValueError("actor environment is single-use")
        self._session = self._world.new_session(
            task=self._task.instruction,
            initial_context=self._task.initial_context,
            tools=self._task.tools,
        )
        return self

    def execute(self, action: ToolCall) -> Observation:
        """Deliver one original call identity through the normal simulator step protocol."""
        if self._session is None or self._failed:
            raise RuntimeError("actor simulator session is not active")
        public_action: JsonObject = {"name": action.name, "arguments": action.arguments}
        try:
            result = self._world.step(
                self._session.id,
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": action.call_id,
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": json.dumps(action.arguments),
                            },
                        }
                    ],
                },
            )
            if len(result.messages) != 1 or result.messages[0].get("role") != "tool":
                raise ValueError("one actor action requires one simulator tool result")
            content = result.messages[0].get("content")
            if not isinstance(content, str):
                raise ValueError("simulator tool result must contain text")
        except Exception as error:  # noqa: BLE001 - preserve actual failed actor slots.
            self._failed = True
            self._steps.append(
                WorkflowStep(
                    step=len(self._steps),
                    action=public_action,
                    status="failure",
                    error_type=type(error).__name__,
                )
            )
            raise
        self._steps.append(
            WorkflowStep(step=len(self._steps), action=public_action, status="valid", reply=content)
        )
        return Observation(content=content)

    def evidence(
        self, *, workflow_id: str, final_answer: str | None, stop_reason: str
    ) -> WorkflowEvidence:
        """Capture observable actor behavior; unobserved private state remains unavailable."""
        return WorkflowEvidence(
            workflow_id=workflow_id,
            task=self._task.instruction,
            task_context=self._task.initial_context,
            initial_state={},
            steps=tuple(self._steps),
            final_answer=final_answer,
            stop_reason=stop_reason,
            contract={"state_observation": "Private state unavailable in this actor trace"},
        )

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the normal simulator session on success or actor failure."""
        if self._session is not None:
            self._world.end_session(self._session.id)
            self._session = None
