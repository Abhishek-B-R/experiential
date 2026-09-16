"""Source disclosure, finite reservations, and provider response boundary tests."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest
from exp.simulation.claas.harness import SourceDisclosure, WorldModelLimitError
from exp.simulation.claas.harness_test import RecordingClient, limits, model_snapshot
from exp.simulation.claas.mining_test import make_experience
from exp.simulation.claas.provider import ClaasBoundedProvider


def test_denied_disclosure_never_dispatches_and_failed_calls_retain_reservations() -> None:
    """Disclosure and shared budget checks happen before a provider can receive source data."""
    client = RecordingClient(lambda _: {"unused": True})
    scope = make_experience().scope
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="source"),), maximum_output_tokens=16
    )
    provider = ClaasBoundedProvider(
        client=client, model=model_snapshot(), limits=limits(maximum_model_calls=1)
    )
    with pytest.raises(ValueError, match="disclosure"):
        provider.complete(scope, request)
    assert client.requests == []
    provider.source_disclosure = SourceDisclosure(scope=scope, model=model_snapshot())
    provider.complete(scope, request)
    assert provider.reserved_calls == 1
    with pytest.raises(WorldModelLimitError, match="budget"):
        provider.complete(scope, request)
    assert len(client.requests) == 1


def test_provider_exception_does_not_refund_its_reservation() -> None:
    """A transport failure remains charged because remote execution may have begun."""

    def fail(_request: ModelRequest) -> JsonObject:
        """Simulate an uncertain provider failure after dispatch."""
        raise ValueError("uncertain completion")

    scope = make_experience().scope
    client = RecordingClient(fail)
    provider = ClaasBoundedProvider(
        client=client,
        model=model_snapshot(),
        limits=limits(maximum_model_calls=1),
        source_disclosure=SourceDisclosure(scope=scope, model=model_snapshot()),
    )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="source"),), maximum_output_tokens=16
    )
    with pytest.raises(ValueError, match="uncertain"):
        provider.complete(scope, request)
    with pytest.raises(WorldModelLimitError, match="budget"):
        provider.complete(scope, request)
    assert provider.reserved_calls == len(client.requests) == 1


def test_provider_recipient_drift_fails_before_synthesis_or_judgment_dispatch() -> None:
    """A changed client recipient cannot reuse an earlier scope/model disclosure grant."""
    scope = make_experience().scope
    client = RecordingClient(lambda _: {"unused": True})
    provider = ClaasBoundedProvider(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        source_disclosure=SourceDisclosure(scope=scope, model=model_snapshot()),
    )
    client.model_snapshot = model_snapshot().model_copy(update={"model_id": "different"})
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="source"),), maximum_output_tokens=16
    )
    with pytest.raises(ValueError, match="recipient"):
        provider.complete(scope, request)
    assert provider.reserved_calls == 0 and not client.requests
