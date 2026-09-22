"""Independent cells overlap while progress stays on the caller thread."""

from threading import Barrier, get_ident
from typing import cast

from exp.common.evaluations import EvaluationCell
from exp.common.rollouts import RolloutArtifact
from exp.simulation.engines.text.dispatch import dispatch_cells
from exp.simulation.engines.text.simulator_test import _cell


def test_dispatch_overlaps_workers_and_serializes_progress() -> None:
    """A barrier proves parallel execution without timing-dependent assertions."""
    barrier = Barrier(2, timeout=5)
    owner = get_ident()
    worker_threads: set[int] = set()
    progress_threads: list[int] = []
    completed: dict[str, RolloutArtifact] = {}

    def execute(cell: EvaluationCell) -> RolloutArtifact:
        """Meet the other worker before completing an opaque test result."""
        worker_threads.add(get_ident())
        barrier.wait()
        return cast(RolloutArtifact, cell)

    dispatch_cells(
        (_cell("cell-a", "task-a"), _cell("cell-b", "task-b")),
        workers=2,
        execute=execute,
        completed=completed,
        observe=lambda: progress_threads.append(get_ident()),
    )
    assert len(worker_threads) == 2 and owner not in worker_threads
    assert progress_threads == [owner, owner]
    assert set(completed) == {"cell-a", "cell-b"}
