"""Strict feedback and scenario boundaries."""

import pytest

from exp.simulation.claas import WorldTransition


def test_empty_nonterminal_world_output_is_invalid() -> None:
    """The world must provide another visible observation or explicitly end the episode."""
    with pytest.raises(ValueError, match="nonterminal"):
        WorldTransition(observations=(), terminal=False, feedback="Unknown.", reward=None)
