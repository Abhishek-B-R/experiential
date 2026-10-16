"""Held-out simulation fidelity in the existing evaluation terminal conventions."""

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.fidelity import render_fidelity
from exp.cli.evaluation.fidelity_turns import render_turns, turn_marks
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display, qualified
from exp.cli.shared.theme import EXP_THEME
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.fidelity import (
    PreparedFidelity,
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.held_out_fidelity import (
    register_held_out_fidelity,
    saved_held_out_results,
)
from exp.optimize.evaluation.trace_learning import fit_cases, learning_curve
from exp.optimize.evaluation.trace_replay import ReplayAggregate, ReplayResult, aggregate
from exp.runtime.models.providers.transport import ProviderTransportError


def run_held_out_fidelity(
    project: str,
    count: int,
    *,
    root: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    verbose: bool = False,
    retry_incomplete: bool = False,
) -> None:
    """Measure the learned simulator on held-out traces it never learned, or replay the result.

    Uses the hybrid profile: the memory learned with ``--hold-out`` at its final checkpoint,
    with every held-out lineage reserved, plus recorded facts from the fit-only index.

    Args:
        project: Existing project with a completed build.
        count: Maximum held-out traces, selected deterministically.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions.
        dry_run: Register the runs and stop before credentials or provider calls.
        verbose: Also print every trace, every turn, and every criterion.
        retry_incomplete: Replace interrupted runs, buying their calls again.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        store = ProjectStore(root, project)
        panel = register_held_out_fidelity(
            store,
            count,
            profile="memory",
            code_revision=installed_release_revision(),
            clock=lambda: datetime.now(UTC),
            retry_incomplete=retry_incomplete,
        )
        heading(console, project, "Held-out simulation fidelity")
        judge = panel.prepared[0].spec.judge_alias
        steps = sum(len(case.steps) for case in panel.cases)
        console.print(
            Text(
                f"{len(panel.cases)} held-out traces · {steps} turns · hybrid simulator "
                f"{panel.world_model.artifact_id} · memory v{panel.memory_version} · "
                f"judge {judge}",
                style="dim",
            )
        )
        if panel.retired:
            console.print(
                Text(
                    f"{len(panel.retired)} earlier attempts could not complete; "
                    "registered fresh attempts",
                    style="dim",
                )
            )
        fit = len(fit_cases(store))
        if panel.memory_version is not None and panel.memory_version < fit:
            console.print(
                f"[yellow]warning[/yellow] memory v{panel.memory_version} learned "
                f"{panel.memory_version} of {fit} fit traces; finish `exp build --learn` "
                "to measure the fully learned simulator"
            )
        if dry_run:
            for prepared in panel.prepared:
                console.print(Text(f"  {prepared.spec.run_id}"))
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return
        pending = [p for p in panel.prepared if saved_fidelity_report(store, p) is None]
        if not pending:
            console.print(Text("Saved results. No provider calls.", style="dim"))
            # The lifecycle owner replays each report and completes any interrupted publication.
            for prepared in panel.prepared:
                execute_fidelity_evaluation(store, prepared)
        else:
            interactive = not non_interactive and can_prompt(console)
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --held-out-fidelity {count}",
                non_interactive=not interactive,
            ):
                return
            execute_runs(console, store, panel.prepared)
        results = saved_held_out_results(store, panel)
        if verbose:
            render_traces(console, store, panel.prepared, results, "Held-out trace")
        console.print()
        console.print(Text(aggregate_line(aggregate(results), len(panel.cases))))
        curve = learning_curve(store, "fit")
        if curve is not None:
            console.print(Text(curve.line()))
        console.print(
            Text(
                "Held-out lineages were reserved before learning; recorded facts come only "
                "from fit traces.",
                style="dim",
            )
        )


def execute_runs(
    console: Console, store: ProjectStore, registrations: tuple[PreparedFidelity, ...]
) -> None:
    """Run every registered trace through the fidelity owner, one progress row per stage."""
    total = len(registrations)
    try:
        with progress_display(console) as progress:
            for index, prepared in enumerate(registrations, start=1):
                execute_fidelity_evaluation(
                    store, prepared, progress=qualified(progress, f"trace {index}/{total}")
                )
    except ProviderTransportError as exc:
        console.print(f"[red]error[/red] a provider request failed: {exc}")
        console.print(
            "Completed traces are saved. Run the same command again to continue; "
            "the stopped trace starts a fresh attempt."
        )
        raise typer.Exit(code=1) from exc


def aggregate_line(total: ReplayAggregate, planned: int, title: str = "Held-out fidelity") -> str:
    """One line: replies exact, reply-field agreement, and the judge mean.

    Args:
        total: Sums over completed traces.
        planned: Traces in the panel, including any without a saved report.
        title: Leading label naming what was measured.

    Returns:
        Plain summary text with explicit denominators.
    """
    exact = f"{total.exact}/{total.steps} replies exact ({share(total.exact, total.steps)})"
    fields = (
        f"{share(total.fields_matched, total.fields_total)} reply fields match "
        f"({total.fields_matched}/{total.fields_total})"
    )
    judge = "judge -" if total.judge_mean is None else f"judge {total.judge_mean:.1f}/5"
    line = f"{title}  {exact} · {fields} · {judge} · {total.traces}/{planned} traces"
    if total.judge_turns:
        line += (
            f"\n  judge agrees with exact replies on {total.judge_agreements}/{total.judge_turns} "
            f"turns ({share(total.judge_agreements, total.judge_turns)})"
        )
    return line


def render_traces(
    console: Console,
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    results: tuple[ReplayResult, ...],
    label: str,
) -> None:
    """Per-trace rows, then every turn and criterion of each trace."""
    table = Table(box=None, padding=(0, 2))
    for column in (label, "Replies exact", "Field match", "Judge", "Turns"):
        table.add_column(column)
    for result in results:
        table.add_row(
            Text(result.trace_id),
            f"{result.exact}/{result.steps}",
            share(result.fields_matched, result.fields_total),
            _judge(result),
            turn_marks(result.turns),
        )
    console.print()
    console.print(table)
    prepared = {p.spec.run_id: p for p in registrations}
    for result in results:
        registration = prepared[result.run_id]
        console.print()
        console.print(Text(f"{result.trace_id} · {result.run_id}", style="bold"))
        render_turns(console, store, registration, result.report, registration.spec.judge)
        render_fidelity(console, result.report, registration.spec.judge)


def _judge(result: ReplayResult) -> str:
    """Mean assessed score of one trace, or its judge status."""
    if not result.judge_scores:
        return "-" if result.judge_status == "completed" else result.judge_status
    return f"{sum(result.judge_scores) / len(result.judge_scores):.1f}"


def share(part: int, whole: int) -> str:
    """A whole-number percentage, or a dash without a denominator."""
    return f"{part / whole:.0%}" if whole else "-"
