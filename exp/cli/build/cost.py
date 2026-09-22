"""Build-cost ceiling diagnostics for explicit trace automation."""

from __future__ import annotations

import math
import shlex
from pathlib import Path

from exp.common.config import ARTIFACT_DIR


def over_ceiling_message(
    *,
    estimate: float,
    ceiling: float,
    project: str,
    trace_file: Path,
    source: str,
    root: Path,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
    top_k: int,
    command_name: str,
) -> str:
    """Describe an over-ceiling refusal and the exact command that raises the limit.

    Args:
        estimate: Conservative maximum embedding cost in USD.
        ceiling: Configured ``--max-build-cost-usd`` value that the estimate exceeded.
        project: Local project identifier from this invocation.
        trace_file: Trace export path from this invocation.
        source: Selected canonical source format.
        root: Local ``.exp`` artifact root.
        world_model: Optional world-model alias override.
        judge: Optional judge alias override.
        embedder: Optional embedder alias override.
        top_k: Requested retrieval result limit.
        command_name: Invoked CLI command, used unchanged in the recovery instruction.

    Returns:
        Fail-closed message naming both amounts and a sufficient rebuild command.
    """
    command = grounded_build_command(
        command_name=command_name,
        project=project,
        trace_file=trace_file,
        source=source,
        root=root,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
        top_k=top_k,
        maximum_build_cost_usd=sufficient_ceiling_usd(estimate),
    )
    return (
        f"conservative embedding estimate ${estimate:.6f} exceeds "
        f"--max-build-cost-usd ${ceiling:.6f}. "
        f"Re-run with a higher ceiling: {command}"
    )


def grounded_build_command(
    *,
    command_name: str,
    project: str,
    trace_file: Path,
    source: str,
    root: Path,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
    top_k: int,
    maximum_build_cost_usd: str,
) -> str:
    """Render a reusable invocation for spend consent and cost-ceiling recovery.

    Args:
        command_name: Invoked CLI command name.
        project: Local project identifier.
        trace_file: Explicit trace export path.
        source: Selected source, retained even when it matches one command's default.
        root: Local artifact root.
        world_model: Optional world-model alias override.
        judge: Optional judge alias override.
        embedder: Optional embedder alias override.
        top_k: Requested retrieval result limit.
        maximum_build_cost_usd: Requested or recommended ceiling encoded without precision loss.

    Returns:
        Shell-quoted command preserving the selected ingestion workflow and cost ceiling.
    """
    command = ["exp", command_name, project, "--traces", str(trace_file), "--source", source]
    if root != Path(ARTIFACT_DIR):
        command.extend(["--root", str(root)])
    if world_model is not None:
        command.extend(["--world-model", world_model])
    if judge is not None:
        command.extend(["--judge", judge])
    if embedder is not None:
        command.extend(["--embedder", embedder])
    if top_k != 5:
        command.extend(["--top-k", str(top_k)])
    command.extend(["--max-build-cost-usd", maximum_build_cost_usd])
    return shlex.join(command)


def sufficient_ceiling_usd(estimate: float) -> str:
    """Return a six-decimal ceiling that authorizes the full-precision estimate.

    Args:
        estimate: Conservative maximum embedding cost in USD.

    Returns:
        A ``--max-build-cost-usd`` value whose parsed float is at least ``estimate``.
    """
    micros = max(math.ceil(estimate * 1_000_000), 1)
    while True:
        text = f"{micros / 1_000_000:.6f}"
        if float(text) >= estimate:
            return text
        micros += 1
