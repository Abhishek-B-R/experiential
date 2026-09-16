"""No-spend end-to-end practice sessions, strict validation, and shared limits."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
)
from exp.simulation.claas import (
    ClaasWorldModel,
    SourceDisclosure,
    WorldModelLimitError,
    WorldModelLimits,
    mine_experiences,
    replay_episode,
)
from exp.simulation.claas.mining_test import make_experience


def model_snapshot() -> ModelSnapshot:
    """Return a stable provider identity without any credential-bearing configuration."""
    return ModelSnapshot(
        provider="replay",
        model_id="world-model",
        revision="fixed",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


class RecordingClient:
    """In-process completion fixture that never creates an HTTP provider."""

    def __init__(self, respond: Callable[[ModelRequest], JsonObject]) -> None:
        """Bind deterministic test responses and retain every dispatched request."""
        self.respond = respond
        self.model_snapshot = model_snapshot()
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Produce a typed synthetic response with finite accounting."""
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=json.dumps(self.respond(request))),
            model=model_snapshot(),
            economics=OperationEconomics(
                cost_usd=NumericMeasurement(value=0.01, provenance="observed")
            ),
        )


def limits(**overrides: int | float) -> WorldModelLimits:
    """Create a small explicitly funded fixture budget."""
    values: dict[str, int | float] = {
        "maximum_steps": 2,
        "maximum_model_calls": 2,
        "maximum_total_cost_usd": 0.04,
        "maximum_call_cost_usd": 0.02,
    }
    values.update(overrides)
    return WorldModelLimits.model_validate(values)


def tool_transition(_request: ModelRequest) -> JsonObject:
    """Return one successful tool observation with private training feedback."""
    return {
        "observations": [{"call_id": "lookup-1", "content": "Claim pending.", "is_error": False}],
        "user_message": None,
        "terminal": False,
        "feedback": "PRIVATE LABEL: the claim lookup is useful.",
        "reward": 0.5,
    }


def final_transition(_request: ModelRequest) -> JsonObject:
    """Return a terminal transition whose score is explicitly unknown."""
    return {
        "observations": [],
        "user_message": "Thanks.",
        "terminal": True,
        "feedback": "The conversation has ended; external correctness is unverified.",
        "reward": None,
    }


def lookup_action() -> AssistantAction:
    """Return a policy-generated call using the mined tool schema."""
    return AssistantAction(
        tool_calls=(ToolCall(call_id="lookup-1", name="lookup", arguments={"claim_id": "c1"}),)
    )


def test_tool_episode_replays_exactly_and_never_exposes_private_feedback() -> None:
    """Drive the public API through a tool step and terminal response without spending."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(tool_transition)
    world = ClaasWorldModel(
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
        client=client,
        model=model_snapshot(),
        limits=limits(),
    )
    with world.open(scenario, grounding=(source,)) as session:
        first = session.step(lookup_action())
        assert first.provenance == "synthetic"
        assert first.transition.reward == 0.5
        assert "PRIVATE LABEL" not in " ".join(
            message.content or "" for message in session.messages
        )
        assert session.messages[-1].tool_call_id == "lookup-1"
        client.respond = final_transition
        session.step(AssistantAction(content="Your claim is pending."))
        episode = session.end()
    assert episode.end_reason == "world_terminal"
    assert episode.outcome == "unverified"
    assert world.reserved_calls == 2
    assert replay_episode(episode, grounding=(source,), limits=limits()) == episode
    with pytest.raises(ValueError, match="closed"):
        session.step(lookup_action())


def test_reset_never_replenishes_shared_provider_budget() -> None:
    """Opening new sessions cannot bypass the run-level provider ceiling."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
        client=client,
        model=model_snapshot(),
        limits=limits(maximum_model_calls=1),
    )
    world.reset(scenario, grounding=(source,)).step(AssistantAction(content="Done."))
    session = world.reset(scenario, grounding=(source,))
    with pytest.raises(WorldModelLimitError, match="budget exhausted"):
        session.step(AssistantAction(content="Done."))
    assert len(client.requests) == 1
    assert session.end().end_reason == "limit"


@pytest.mark.parametrize(
    "alteration",
    [
        {"observations": [{"call_id": "wrong", "content": "x", "is_error": False}]},
        {"reward": float("nan")},
        {"terminal": "false"},
        {"secret_answer": "leak"},
    ],
)
def test_world_output_is_strict_and_invalid_calls_keep_their_reservation(
    alteration: JsonObject,
) -> None:
    """Malformed feedback cannot enter policy state or evade call accounting."""

    def respond(request: ModelRequest) -> JsonObject:
        """Alter one part of an otherwise valid output protocol."""
        return {**tool_transition(request), **alteration}

    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    world = ClaasWorldModel(
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
        client=RecordingClient(respond),
        model=model_snapshot(),
        limits=limits(),
    )
    session = world.reset(scenario, grounding=(source,))
    with pytest.raises(ValueError):
        session.step(lookup_action())
    assert session.messages == scenario.messages
    assert session.end().end_reason == "error"
    assert world.reserved_calls == 1


def test_grounding_drift_and_heldout_seeds_fail_before_provider_dispatch() -> None:
    """Only the exact scenario sources and fit partition can ground synthetic practice."""
    source = make_experience()
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
        client=client,
        model=model_snapshot(),
        limits=limits(),
    )
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    altered = source.model_copy(update={"response": {"changed": True}})
    with pytest.raises(ValueError, match="digest"):
        world.reset(scenario, grounding=(altered,))
    heldout = mine_experiences((source,), partition="held_out")[0].scenario
    with pytest.raises(ValueError, match="fit evidence"):
        world.reset(heldout, grounding=(source,))
    assert not client.requests


def test_request_size_limit_is_enforced_before_spend() -> None:
    """An oversized prompt stops before reserving or dispatching a model call."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
        client=client,
        model=model_snapshot(),
        limits=limits(maximum_request_bytes=1),
    )
    with pytest.raises(WorldModelLimitError, match="byte limit"):
        world.reset(scenario, grounding=(source,)).step(AssistantAction(content="Done."))
    assert not client.requests
    assert world.reserved_calls == 0


@pytest.mark.parametrize("authorization", ["absent", "wrong_scope", "wrong_model"])
def test_source_disclosure_requires_exact_scope_and_model(authorization: str) -> None:
    """Capturing traffic never implicitly grants a newly selected provider access."""
    source = make_experience()
    scenario = mine_experiences((source,), partition="fit")[0].scenario
    client = RecordingClient(final_transition)
    disclosure = None
    if authorization != "absent":
        scope = source.scope
        model = model_snapshot()
        if authorization == "wrong_scope":
            scope = scope.model_copy(update={"user_id": "other-user"})
        else:
            model = model.model_copy(update={"provider": "other-provider"})
        disclosure = SourceDisclosure(scope=scope, model=model)
    world = ClaasWorldModel(
        client=client, model=model_snapshot(), limits=limits(), source_disclosure=disclosure
    )
    with pytest.raises(ValueError, match="source disclosure"):
        world.reset(scenario, grounding=(source,))
    assert not client.requests
    assert world.reserved_calls == 0


def test_mismatched_client_recipient_is_rejected_before_source_disclosure() -> None:
    """A declared snapshot cannot authorize a differently configured client."""
    source = make_experience()
    client = RecordingClient(final_transition)
    client.model_snapshot = model_snapshot().model_copy(update={"model_id": "unapproved-recipient"})
    with pytest.raises(ValueError, match="bound.*recipient"):
        ClaasWorldModel(
            client=client,
            model=model_snapshot(),
            limits=limits(),
            source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
        )
    assert not client.requests


def test_recipient_drift_after_reset_cannot_disclose_sources() -> None:
    """Recheck recipient identity at dispatch, not only when a session is created."""
    source = make_experience()
    client = RecordingClient(final_transition)
    world = ClaasWorldModel(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        source_disclosure=SourceDisclosure(scope=source.scope, model=model_snapshot()),
    )
    session = world.open(
        mine_experiences((source,), partition="fit")[0].scenario, grounding=(source,)
    )
    client.model_snapshot = model_snapshot().model_copy(update={"connection_sha256": "c" * 64})
    with pytest.raises(ValueError, match="recipient"):
        session.step(AssistantAction(content="Done."))
    assert not client.requests and world.reserved_calls == 0
