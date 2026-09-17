"""Tests for response assembly from normalized serving events."""

import pytest

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.openai_protocol.response import completed_body


@pytest.mark.parametrize("terminal", [GatewayEventKind.COMPLETED, GatewayEventKind.INCOMPLETE])
def test_responses_body_truncates_fractional_epoch_seconds(terminal: GatewayEventKind) -> None:
    """A fractional wall clock yields integer timestamps for strict Responses clients.

    Args:
        terminal: The terminal status whose timestamp contract is checked.
    """
    body = completed_body(
        request=GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="hello"),),
        ),
        request_id="request-one",
        model="coding",
        created_at=1_700_000_000.75,
        events=(GatewayEvent(kind=terminal, sequence_number=0),),
    )
    assert type(body["created_at"]) is int
    assert body["created_at"] == 1_700_000_000
    if terminal == GatewayEventKind.COMPLETED:
        assert type(body["completed_at"]) is int
        assert body["completed_at"] == body["created_at"]
    else:
        assert body["completed_at"] is None
