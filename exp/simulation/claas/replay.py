"""Provider-free replay of exact recorded world-model requests and transitions."""

from __future__ import annotations

import json
from collections.abc import Sequence

from exp.common.claas import Experience
from exp.common.core.artifacts import sha256_json
from exp.common.models import AssistantAction, ModelRequest, ModelResponse, ModelSnapshot
from exp.simulation.claas.contracts import WorldEpisode, WorldStep
from exp.simulation.claas.harness import ClaasWorldModel, SourceDisclosure, WorldModelLimits


class ReplayModelClient:
    """Replay recorded responses only when the complete request matches its evidence."""

    def __init__(self, steps: Sequence[WorldStep]) -> None:
        """Retain an ordered immutable sequence without constructing provider clients."""
        self._steps = tuple(steps)
        self.consumed = 0

    @property
    def model_snapshot(self) -> ModelSnapshot:
        """Expose the recorded recipient, without constructing a live provider."""
        if not self._steps:
            raise ValueError("replay has no recorded recipient")
        return self._steps[0].response.model

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the next recorded result, rejecting missing or drifted requests."""
        if self.consumed >= len(self._steps):
            raise ValueError("replay has no recorded call remaining")
        step = self._steps[self.consumed]
        if sha256_json(request) != sha256_json(step.request):
            raise ValueError("replay request differs from its recorded evidence")
        self.consumed += 1
        return step.response


def replay_episode(
    episode: WorldEpisode,
    *,
    grounding: Sequence[Experience],
    limits: WorldModelLimits,
) -> WorldEpisode:
    """Re-execute a completed episode's parsing and state transitions without spending.

    Args:
        episode: A successfully recorded terminal or caller-ended episode.
        grounding: Its exact immutable captured source exchanges.
        limits: The same bounds used for the recorded run.

    Returns:
        Independently reconstructed evidence equal to the recorded episode.

    Raises:
        ValueError: Evidence is incomplete, altered, reordered, or incompatible with this harness.
    """
    if not episode.steps or episode.end_reason not in ("world_terminal", "caller_ended"):
        raise ValueError("episode replay requires recorded steps and a completed lifecycle")
    client = ReplayModelClient(episode.steps)
    world = ClaasWorldModel(
        client=client,
        model=episode.steps[0].response.model,
        limits=limits,
        purpose="evaluation" if episode.scenario.partition == "held_out" else "practice",
        source_disclosure=SourceDisclosure(
            scope=episode.scenario.scope, model=episode.steps[0].response.model
        ),
    )
    with world.open(episode.scenario, grounding=grounding) as session:
        for step in episode.steps:
            content = step.request.messages[-1].content
            if content is None:
                raise ValueError("recorded world-model request has no action payload")
            payload = json.loads(content)
            if not isinstance(payload, dict) or "latest_action" not in payload:
                raise ValueError("recorded world-model request has no latest_action")
            session.step(AssistantAction.model_validate(payload["latest_action"]))
        replayed = session.end()
    if replayed != episode:
        raise ValueError("replayed episode differs from its recorded transitions or provenance")
    return replayed
