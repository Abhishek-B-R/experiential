"""The opt-in learning stage of ``exp build``: memory grown trace by trace from fit traces."""

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.progress import progress_display
from exp.common.project import ProjectStore
from exp.optimize.evaluation.trace_learning import (
    fit_cases,
    learn_from_fit_traces,
    learning_curve,
    learning_plan,
    learning_spec,
)
from exp.optimize.evaluation.trace_replay import DEADLINE
from exp.runtime.models.providers.transport import ProviderTransportError


def run_learning(
    console: Console,
    store: ProjectStore,
    *,
    project: str,
    root: Path,
    recall_alias: str,
    maximum_attempts: int | None,
    yes: bool,
    non_interactive: bool,
    code_revision: str,
    retry_incomplete: bool = False,
) -> None:
    """Show the learning preflight, ask for spend consent, then learn from every fit trace.

    Args:
        console: Build command's terminal owner.
        store: Built project.
        project: Project name, for the replay command.
        root: Local ``.exp`` root holding the consent settings.
        recall_alias: Native decision alias recalling memory records.
        maximum_attempts: Physical-attempt budget; None allows every trace's full ceiling.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Replace an interrupted trace, buying its calls again.

    Raises:
        typer.Exit: A provider request failed; completed traces stay saved.
    """
    plan = learning_plan(store, recall_alias)
    cases = fit_cases(store)
    now = datetime.now(UTC)
    ceiling = sum(
        learning_spec(store, plan, case, deadline=now + DEADLINE).maximum_physical_attempts
        for case in cases
    )
    budget = ceiling if maximum_attempts is None else maximum_attempts
    console.print("Learning preflight")
    console.print(f"  [dim]traces[/dim]       {len(cases)} fit traces, in export order")
    console.print(f"  [dim]simulator[/dim]    hybrid {plan.world_model.artifact_id}")
    console.print(f"  [dim]memory[/dim]       {plan.lineage} · recall {plan.recall.jev_alias}")
    console.print(
        f"  [dim]attempts[/dim]     at most {budget:,} physical attempts, retries included "
        f"(all traces need at most {ceiling:,})"
    )
    if not require_spend_consent(
        console,
        root=root,
        yes=yes,
        estimated_cost_usd=None,
        cost_is_upper_bound=False,
        command=f"exp build {project} --learn --memory-recall {recall_alias}",
        non_interactive=non_interactive or not can_prompt(console),
    ):
        return
    try:
        with progress_display(console) as progress:
            outcome = learn_from_fit_traces(
                store,
                recall_alias=recall_alias,
                maximum_attempts=budget,
                code_revision=code_revision,
                clock=lambda: datetime.now(UTC),
                progress=progress,
                retry_incomplete=retry_incomplete,
            )
    except ProviderTransportError as exc:
        console.print(f"[red]error[/red] a provider request failed: {exc}")
        console.print(
            "Learned traces are saved. Run the same command again to continue; "
            "the stopped trace starts a fresh attempt."
        )
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]✓[/green] learned {outcome.learned}/{outcome.traces} fit traces · "
        f"memory v{outcome.head_version} · {outcome.attempts_used:,} physical attempts"
    )
    if outcome.incomplete:
        console.print(
            f"[yellow]warning[/yellow] {outcome.incomplete} trace(s) stopped before completing "
            "on every allowed attempt and were not learned"
        )
    if outcome.stopped is not None:
        console.print(f"[yellow]warning[/yellow] {outcome.stopped}")
    curve = learning_curve(store)
    if curve is not None:
        console.print(Text(curve.line()))
    console.print(f"next: exp eval {project} --held-out-fidelity N")
