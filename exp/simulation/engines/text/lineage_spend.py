"""Count physical provider work once across retries and retained continuation prefixes."""

import math
from collections.abc import Callable, Sequence

from exp.common.project import ArtifactStore, artifact_input
from exp.common.rollouts import RolloutArtifact
from exp.simulation.engines.text.bindings import rollout_id_for_binding
from exp.simulation.engines.text.errors import SimulationResumeError
from exp.simulation.engines.text.resume import load_rollout
from exp.simulation.engines.text.rollout_support import rollout_spend


def lineage_spend(
    store: ArtifactStore,
    rollouts: Sequence[RolloutArtifact],
    *,
    measure: Callable[[RolloutArtifact], float | None] = rollout_spend,
) -> float | None:
    """Sum verified retries and ancestors, subtracting each child's retained prefix.

    Args:
        store: Immutable store owning all ancestor and retry evidence.
        rollouts: Selected heads or already expanded attempts. Duplicate IDs count once.
        measure: Operation-spend validator used at admission or final reconciliation.

    Returns:
        Unique physical provider spend, or None if any dispatched cost is unknown.

    Raises:
        SimulationResumeError: A parent pointer, retry binding or cumulative cost is inconsistent.
    """
    pending = list(rollouts)
    seen: set[str] = set()
    increments: list[float] = []
    while pending:
        rollout = pending.pop()
        if rollout.rollout_id in seen:
            continue
        seen.add(rollout.rollout_id)
        amount = measure(rollout)
        if amount is None:
            return None
        pointer = rollout.continuation_of
        if pointer is not None:
            if artifact_input(store.read(pointer.artifact_id).manifest) != pointer:
                raise SimulationResumeError("continuation spend parent manifest changed")
            parent = load_rollout(store, pointer.artifact_id)
            prefix = measure(parent)
            if prefix is None:
                return None
            amount -= prefix
            if amount < -1e-12:
                raise SimulationResumeError("continued rollout omitted retained prefix spend")
            pending.append(parent)
        increments.append(max(0.0, amount))
        binding = rollout.simulation_binding
        if rollout.retry_attempt and binding is None:
            raise SimulationResumeError("retry spend requires a complete simulation binding")
        if binding is not None:
            for attempt in range(rollout.retry_attempt):
                prior = load_rollout(store, rollout_id_for_binding(binding, attempt=attempt))
                if prior.simulation_binding != binding or prior.retry_attempt != attempt:
                    raise SimulationResumeError("retry spend binding changed")
                pending.append(prior)
    return math.fsum(increments)


def prefix_retry_credit(parent: RolloutArtifact | None, attempt: int) -> float:
    """Avoid charging a restored prefix twice against the same cell's retry allowance."""
    if parent is None or attempt == 0:
        return 0.0
    value = rollout_spend(parent)
    if value is None:
        raise SimulationResumeError("cannot restore a prefix with unknown provider spend")
    return value
