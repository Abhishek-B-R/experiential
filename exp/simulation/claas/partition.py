"""Deterministic episode-disjoint traffic partitions established before synthesis."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, model_validator

from exp.common.claas import ClaasScope, Experience
from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.simulation.claas.mining import MiningLimits, mine_experiences


class SourceGroup(ContractModel):
    """One indivisible set of linked source exchanges and its assigned partition."""

    group_id: Sha256
    experience_ids: tuple[str, ...] = Field(min_length=1)


class ClaasSourceSplit(ContractModel):
    """Frozen source contents and group membership, without any generated data."""

    scope: ClaasScope
    seed: str = Field(min_length=1, max_length=512)
    held_out_fraction: float = Field(gt=0, lt=1, allow_inf_nan=False)
    fit: tuple[Experience, ...] = Field(min_length=1)
    held_out: tuple[Experience, ...] = Field(min_length=1)
    fit_groups: tuple[SourceGroup, ...] = Field(min_length=1)
    held_out_groups: tuple[SourceGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_membership(self) -> ClaasSourceSplit:
        """Recheck episode boundaries when loading a saved partition manifest."""
        sources = self.fit + self.held_out
        expected = _source_groups(sources)
        actual = self.fit_groups + self.held_out_groups
        if {group.group_id: group for group in expected} != {
            group.group_id: group for group in actual
        } or len(actual) != len(expected):
            raise ValueError("source split must preserve every complete linked group exactly once")
        for partition, groups in (
            (self.fit, self.fit_groups),
            (self.held_out, self.held_out_groups),
        ):
            if {item.experience_id for item in partition} != {
                identity for group in groups for identity in group.experience_ids
            }:
                raise ValueError("source partition membership differs from its immutable groups")
        if any(
            _held_out(group, self.seed, self.held_out_fraction) for group in self.fit_groups
        ) or any(
            not _held_out(group, self.seed, self.held_out_fraction)
            for group in self.held_out_groups
        ):
            raise ValueError("source groups differ from their stable hash assignment")
        if any(item.scope != self.scope for item in sources):
            raise ValueError("source split crosses user or application scope")
        return self

    @property
    def digest(self) -> str:
        """Return the content identity binding sources, groups, and split seed."""
        return sha256_json(self)


def split_experiences(
    experiences: Sequence[Experience],
    *,
    seed: str,
    held_out_fraction: float = 0.2,
    limits: MiningLimits | None = None,
) -> ClaasSourceSplit:
    """Split complete source groups before mining, synthesis, or policy optimization.

    Explicit episode IDs, Responses ancestry, and declared source ancestry join
    groups. Similar text never joins unrelated callers. At least two independent
    groups are required; a single workflow cannot provide its own held-out test.
    """
    if not math.isfinite(held_out_fraction) or not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must be finite and strictly between zero and one")
    # The existing miner validates source sizes, request schemas, and episode chronology.
    mine_experiences(experiences, partition="fit", limits=limits)
    groups = _source_groups(experiences)
    if len(groups) < 2:
        raise ValueError("held-out evaluation requires at least two independent source groups")
    held_out_groups = tuple(group for group in groups if _held_out(group, seed, held_out_fraction))
    fit_groups = tuple(group for group in groups if not _held_out(group, seed, held_out_fraction))
    if not held_out_groups or not fit_groups:
        raise ValueError(
            "stable split has an empty partition; collect more independent source groups"
        )
    held_out_ids = {identity for group in held_out_groups for identity in group.experience_ids}
    return ClaasSourceSplit(
        scope=experiences[0].scope,
        seed=seed,
        held_out_fraction=held_out_fraction,
        fit=tuple(
            sorted(
                (item for item in experiences if item.experience_id not in held_out_ids),
                key=lambda item: item.experience_id,
            )
        ),
        held_out=tuple(
            sorted(
                (item for item in experiences if item.experience_id in held_out_ids),
                key=lambda item: item.experience_id,
            )
        ),
        fit_groups=fit_groups,
        held_out_groups=held_out_groups,
    )


def exclude_response_groups(
    experiences: Sequence[Experience], response_ids: frozenset[str]
) -> tuple[Experience, ...]:
    """Exclude every explicitly linked group touching a reserved response.

    Apply this before source partitioning. A benchmark response's parent, episode
    peers, and declared source ancestry cannot become practice through a later link.
    """
    if not experiences or not response_ids:
        return tuple(experiences)
    reserved = {item.experience_id for item in experiences if item.response_id in response_ids}
    excluded = {
        identity
        for group in _source_groups(experiences)
        if reserved.intersection(group.experience_ids)
        for identity in group.experience_ids
    }
    return tuple(item for item in experiences if item.experience_id not in excluded)


def _source_groups(experiences: Sequence[Experience]) -> tuple[SourceGroup, ...]:
    """Union explicit relationships while rejecting missing or synthetic provenance."""
    by_id = {item.experience_id: item for item in experiences}
    if len(by_id) != len(experiences) or not experiences:
        raise ValueError("source splits require nonempty, unique experience IDs")
    if len({item.scope for item in experiences}) != 1:
        raise ValueError("source splits require exactly one user and application scope")
    if any(item.provenance.source_kind != "traffic" for item in experiences):
        raise ValueError("source splits require observed traffic before synthesis")
    responses = {item.response_id: item for item in experiences}
    if len(responses) != len(experiences):
        raise ValueError("source splits require unique response IDs")
    for item in experiences:
        current = item
        visited: set[str] = set()
        while current.parent_response_id is not None:
            if current.response_id in visited:
                raise ValueError("source continuation contains a parent cycle")
            visited.add(current.response_id)
            parent = responses.get(current.parent_response_id)
            if parent is None:
                raise ValueError("source continuation is missing its parent")
            if parent.captured_at > current.captured_at:
                raise ValueError("source continuation precedes its parent")
            current = parent
    parents = {identity: identity for identity in by_id}

    def root(identity: str) -> str:
        """Find and compress one union-set representative."""
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    def join(left: str, right: str) -> None:
        """Join two known source identities deterministically."""
        if right not in by_id:
            raise ValueError("source ancestry is incomplete; include every referenced exchange")
        first, second = sorted((root(left), root(right)))
        parents[second] = first

    episodes: dict[str, str] = {}
    for item in experiences:
        if item.episode_id is not None:
            previous = episodes.setdefault(item.episode_id, item.experience_id)
            join(item.experience_id, previous)
        if item.parent_response_id is not None:
            parent = responses.get(item.parent_response_id)
            if parent is None:
                raise ValueError("source continuation is missing its parent")
            join(item.experience_id, parent.experience_id)
        for ancestor in item.provenance.source_experience_ids:
            join(item.experience_id, ancestor)
    members: dict[str, list[str]] = {}
    for identity in by_id:
        members.setdefault(root(identity), []).append(identity)
    return tuple(
        sorted(
            (
                SourceGroup(
                    group_id=_group_identity(tuple(by_id[identity] for identity in ids)),
                    experience_ids=tuple(sorted(ids)),
                )
                for ids in members.values()
            ),
            key=lambda group: group.group_id,
        )
    )


def _held_out(group: SourceGroup, seed: str, fraction: float) -> bool:
    """Assign one stable group by hash threshold, never by the current corpus size."""
    value = int(sha256_json({"seed": seed, "group": group.group_id})[:16], 16)
    return value / 2**64 < fraction


def _group_identity(members: tuple[Experience, ...]) -> str:
    """Keep an episode's assignment stable when later exchanges are appended."""
    episode_ids = sorted({item.episode_id for item in members if item.episode_id is not None})
    if len(episode_ids) > 1:
        raise ValueError("linked source ancestry has conflicting explicit episode IDs")
    first = min(members, key=lambda item: (item.captured_at, item.experience_id))
    identity = "episode:" + episode_ids[0] if episode_ids else "response:" + first.response_id
    return sha256_json({"scope": first.scope.model_dump(mode="json"), "identity": identity})
