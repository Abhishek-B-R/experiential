"""Regression tests for conversation-disjoint, source-bound held-out partitions."""

import pytest

from exp.common.claas import Experience
from exp.simulation.claas.mining_test import make_experience
from exp.simulation.claas.partition import ClaasSourceSplit, split_experiences


def source_batch() -> tuple[Experience, ...]:
    """Create four independently identified workflows without provider execution."""
    return tuple(
        make_experience(index).model_copy(
            update={
                "episode_id": f"episode-{index}",
                "request": {
                    **make_experience(index).request,
                    "messages": [{"role": "user", "content": f"Look up claim source-{index}."}],
                },
            }
        )
        for index in range(4)
    )


def test_deterministic_source_groups_precede_any_synthesis() -> None:
    """Input ordering cannot change assignment or leak a linked continuation across sides."""
    sources = source_batch()
    continuation = make_experience(4).model_copy(
        update={
            "episode_id": None,
            "parent_response_id": sources[0].response_id,
        }
    )
    first = split_experiences((*sources, continuation), seed="fixed", held_out_fraction=0.5)
    second = split_experiences(
        tuple(reversed((*sources, continuation))), seed="fixed", held_out_fraction=0.5
    )
    assert first == second
    fit_ids = {item.experience_id for item in first.fit}
    assert (sources[0].experience_id in fit_ids) == (continuation.experience_id in fit_ids)
    assert not {group.group_id for group in first.fit_groups}.intersection(
        group.group_id for group in first.held_out_groups
    )


def test_split_rejects_synthetic_and_single_group_sources() -> None:
    """Generated or same-episode evidence cannot supply an independent held-out set."""
    with pytest.raises(ValueError, match="two independent"):
        split_experiences((make_experience(0), make_experience(1)), seed="s")
    sources = source_batch()
    simulated = sources[0].model_copy(
        update={
            "provenance": sources[0].provenance.model_copy(update={"source_kind": "simulation"}),
        }
    )
    with pytest.raises(ValueError, match="observed traffic"):
        split_experiences((simulated, *sources[1:]), seed="s")


def test_saved_split_rejects_partition_membership_tampering() -> None:
    """Deserialization rechecks complete membership instead of trusting partition labels."""
    split = split_experiences(source_batch(), seed="s", held_out_fraction=0.5)
    changed = split.model_dump(mode="json")
    changed["fit"] = [split.held_out[0].model_dump(mode="json")]
    with pytest.raises(ValueError):
        ClaasSourceSplit.model_validate(changed)


def test_declared_source_ancestry_stays_in_one_partition() -> None:
    """Imported derivatives cannot cross the boundary from their declared source exchange."""
    sources = source_batch()
    linked = sources[1].model_copy(
        update={
            "episode_id": None,
            "provenance": sources[1].provenance.model_copy(
                update={
                    "source_experience_ids": (sources[0].experience_id,),
                }
            ),
        }
    )
    split = split_experiences((sources[0], linked, *sources[2:]), seed="one", held_out_fraction=0.5)
    fit_ids = {item.experience_id for item in split.fit}
    assert (sources[0].experience_id in fit_ids) == (linked.experience_id in fit_ids)


def test_appending_traffic_never_reassigns_existing_groups() -> None:
    """A fixed hash threshold avoids corpus-size-driven train/evaluation leakage."""
    sources = source_batch()
    first = split_experiences(sources, seed="one", held_out_fraction=0.5)
    later = tuple(
        make_experience(index).model_copy(update={"episode_id": f"episode-{index}"})
        for index in range(4, 12)
    )
    next_split = split_experiences((*sources, *later), seed="one", held_out_fraction=0.5)
    initial_held = {item.experience_id for item in first.held_out}
    expanded_held = {item.experience_id for item in next_split.held_out}
    assert initial_held == expanded_held.intersection(item.experience_id for item in sources)
    continuation = make_experience(20).model_copy(update={"episode_id": sources[0].episode_id})
    extended = split_experiences((*sources, continuation), seed="one", held_out_fraction=0.5)
    assert {group.group_id for group in first.held_out_groups} == {
        group.group_id for group in extended.held_out_groups
    }


def test_explicit_episode_does_not_hide_broken_response_ancestry() -> None:
    """Even explicitly grouped traffic must carry complete acyclic parent links."""
    sources = source_batch()
    broken = sources[0].model_copy(update={"parent_response_id": "absent"})
    with pytest.raises(ValueError, match="missing.*parent"):
        split_experiences((broken, *sources[1:]), seed="one")
    cycle = sources[0].model_copy(update={"parent_response_id": sources[0].response_id})
    with pytest.raises(ValueError, match="cycle"):
        split_experiences((cycle, *sources[1:]), seed="one")
