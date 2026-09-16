"""Fit-only synthesis tests with source-answer isolation and immutable provenance."""

import pytest

from exp.simulation.claas.harness import SourceDisclosure
from exp.simulation.claas.harness_test import RecordingClient, limits, model_snapshot
from exp.simulation.claas.partition import split_experiences
from exp.simulation.claas.partition_test import source_batch
from exp.simulation.claas.provider import ClaasBoundedProvider
from exp.simulation.claas.synthesis import fit_scenarios, synthesize_scenarios


def test_synthesis_sees_only_fit_initial_inputs_and_keeps_tools_fixed() -> None:
    """Held-out requests, historical answers, and output tool definitions never seed generation."""
    split = split_experiences(source_batch(), seed="one", held_out_fraction=0.5)
    client = RecordingClient(lambda _: {"user_message": "Look up a delayed claim."})
    provider = ClaasBoundedProvider(
        client=client,
        model=model_snapshot(),
        limits=limits(),
        source_disclosure=SourceDisclosure(scope=split.scope, model=model_snapshot()),
    )
    generated = synthesize_scenarios(split, provider=provider)
    sent = " ".join(request.model_dump_json() for request in client.requests)
    assert "Finished looking" not in sent
    for source in split.held_out:
        assert f"source-{source.experience_id.rsplit('-', 1)[-1]}" not in sent
    assert all(item.provenance == "synthetic" for item in generated)
    assert all(item.scenario.partition == "fit" for item in generated)
    assert len(fit_scenarios(split, generated)) == 4
    changed = generated[0].model_copy(update={"split_sha256": "0" * 64})
    with pytest.raises(ValueError, match="synthetic scenario|fit-only"):
        fit_scenarios(split, (changed,))
