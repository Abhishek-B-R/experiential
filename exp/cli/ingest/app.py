"""Ingest local traces into a grounded project without running router optimization."""

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

from exp.cli.build.app import build
from exp.cli.shared.consent import can_prompt
from exp.cli.shared.options import ROOT_OPTION
from exp.cli.shared.theme import EXP_THEME
from exp.simulation.ingest.sources import CANONICAL_TRACE_SOURCES

_console = Console(theme=EXP_THEME)
_TRACE_OPTION = typer.Option(None, "--traces", help="JSON, JSONL, or OTel export.")


def ingest(
    project: str = typer.Argument(..., metavar="PROJECT"),
    traces: Path | None = _TRACE_OPTION,
    source: str = typer.Option("chat-json", "--source", help="Declared trace format."),
    root: Path = ROOT_OPTION,
    world_model: str | None = typer.Option(None, "--world-model"),
    judge: str | None = typer.Option(None, "--judge"),
    embedder: str | None = typer.Option(None, "--embedder"),
    maximum_build_cost_usd: float = typer.Option(5.0, "--max-build-cost-usd", min=0.000001),
    yes: bool = typer.Option(False, "--yes", "-y"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Normalize, mine, and ground traces for later evaluations.

    Args:
        project: Local project name.
        traces: Explicit export path, or an interactive file prompt.
        source: Canonical source loader name.
        root: Local artifact root.
        world_model: Project environment model alias.
        judge: Project judge alias.
        embedder: Retrieval embedding model alias.
        maximum_build_cost_usd: Embedding spend ceiling.
        yes: Confirm an allowed estimate.
        dry_run: Stop after provider-free review.
        non_interactive: Never prompt for missing setup.
    """
    interactive = not non_interactive and can_prompt(_console)
    if traces is None:
        if not interactive:
            raise typer.BadParameter("provide --traces PATH, or run exp ingest in a terminal")
        traces = Path(Prompt.ask("Trace file", console=_console)).expanduser()
        source = Prompt.ask(
            "Trace format", choices=list(CANONICAL_TRACE_SOURCES), default=source, console=_console
        )
    build(
        project=project,
        legacy_trace_file=None,
        trace_file=traces,
        source=source,
        root=root,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
        top_k=5,
        maximum_build_cost_usd=maximum_build_cost_usd,
        yes=yes,
        maximum_router_cost_usd=None,
        dry_run=dry_run,
        no_interactive=not interactive,
        provider=None,
    )
