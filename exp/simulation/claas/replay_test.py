"""Replay rejects unrecorded provider requests without contacting a provider."""

import pytest

from exp.common.models import ModelMessage, ModelRequest
from exp.simulation.claas import ReplayModelClient


def test_empty_replay_has_no_fallback_provider() -> None:
    """Running out of evidence is a hard error rather than a paid fallback."""
    client = ReplayModelClient(())
    with pytest.raises(ValueError, match="no recorded"):
        client.complete(ModelRequest(messages=(ModelMessage(role="user", content="hello"),)))
