"""The learning stage of ``exp build``: memory grown trace by trace from the traces.

Learning is the build's default pipeline. It needs a native decision model to recall memory:
the catalog's ``memory_recall`` role, or else the catalog's best SystemOne decision model.
"""

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.progress import progress_display
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.trace_learning import (
    Scope,
    learn_from_traces,
    learned_runs,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_progress,
    learning_spec,
)
from exp.optimize.evaluation.trace_replay import DEADLINE
from exp.runtime.models.providers.transport import ProviderTransportError

# Preferred native decision models for memory recall, best first.
_RECALL_PREFERENCE = ("clef-flash", "clef", "jev")
DECISION_PROVIDER = "typesafe"


def resolve_memory_recall(catalog: ModelCatalog) -> str | None:
    """Choose the decision alias that recalls simulator memory, without a flag.

    The ``memory_recall`` role wins when set. Otherwise the catalog's SystemOne decision models
    are ranked by ``_RECALL_PREFERENCE`` (Clef Flash first), then by alias.

    Args:
        catalog: The project's model catalog.

    Returns:
        The alias, or None when the catalog has no native decision model.

    Raises:
        ValueError: The ``memory_recall`` role names an alias that is not a decision model.
    """

    def decision(alias: str) -> bool:
        record = catalog.models.get(alias)
        connection = None if record is None else catalog.connections.get(record.connection)
        return connection is not None and connection.provider == DECISION_PROVIDER

    chosen = catalog.roles.memory_recall
    if chosen is not None:
        if not decision(chosen):
            raise ValueError(
                f"memory_recall role {chosen!r} is not a native decision model "
                f"({DECISION_PROVIDER} connection)"
            )
        return chosen

    def rank(alias: str) -> tuple[int, str]:
        model = catalog.models[alias].model.casefold()
        hits = [i for i, name in enumerate(_RECALL_PREFERENCE) if name in model]
        return (min(hits) if hits else len(_RECALL_PREFERENCE), alias)

    candidates = sorted((a for a in catalog.models if decision(a)), key=rank)
    return candidates[0] if candidates else None


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
    hold_out: bool = False,
) -> None:
    """Show the learning preflight, ask for spend consent, then learn from every trace.

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
        hold_out: Learn fit traces only, keeping held-out traces for an independent panel.

    Raises:
        typer.Exit: A provider request failed; completed traces stay saved.
    """
    scope: Scope = "fit" if hold_out else "all"
    plan = learning_plan(store, recall_alias, scope)
    cases = learning_cases(store, scope)
    now = datetime.now(UTC)
    ceiling = sum(
        learning_spec(store, plan, case, deadline=now + DEADLINE).maximum_physical_attempts
        for case in cases
    )
    budget = ceiling if maximum_attempts is None else maximum_attempts
    console.print("Learning preflight")
    kept = "fit traces; held-out traces kept out" if hold_out else "traces, every trace"
    console.print(f"  [dim]traces[/dim]       {len(cases)} {kept}, in export order")
    console.print(
        "  [dim]measured[/dim]     test-then-train: whole trace predicted on memory pinned "
        "before it"
    )
    console.print(f"  [dim]simulator[/dim]    hybrid {plan.world_model.artifact_id}")
    console.print(f"  [dim]memory[/dim]       {plan.lineage} · recall {plan.recall.jev_alias}")
    console.print(
        f"  [dim]attempts[/dim]     at most {budget:,} physical attempts, retries included "
        f"(all traces need at most {ceiling:,})"
    )
    scored, _ = learning_progress(store, plan)
    if len(scored) == len(cases) and all(learned for *_, learned in scored):
        # Every trace is already learned: the pass replays saved runs without provider calls.
        console.print(Text("Saved results. No provider calls.", style="dim"))
    elif not require_spend_consent(
        console,
        root=root,
        yes=yes,
        estimated_cost_usd=None,
        cost_is_upper_bound=False,
        command=f"exp eval {project} --audit N" if hold_out else f"exp build {project}",
        non_interactive=non_interactive or not can_prompt(console),
    ):
        return
    try:
        with progress_display(console) as progress:
            outcome = learn_from_traces(
                store,
                recall_alias=recall_alias,
                scope=scope,
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
        f"[green]✓[/green] learned {outcome.learned}/{outcome.traces} "
        f"{'fit traces' if hold_out else 'traces'} · "
        f"memory v{outcome.head_version} · {outcome.attempts_used:,} physical attempts"
    )
    if outcome.incomplete:
        console.print(
            f"[yellow]warning[/yellow] {outcome.incomplete} trace(s) stopped before completing "
            "on every allowed attempt and were not learned"
        )
    if outcome.stopped is not None:
        console.print(f"[yellow]warning[/yellow] {outcome.stopped}")
    curve = learning_curve(store, scope)
    if curve is not None:
        console.print(Text(curve.headline()))
        console.print(Text(curve.line()))
        if not hold_out:
            console.print(
                Text(
                    "This measures learning progress over an evolving memory. For the final "
                    f"simulator on unseen traces, run exp eval {project} --audit N.",
                    style="dim",
                )
            )


def learn_and_retest(
    console: Console,
    store: ProjectStore,
    *,
    project: str,
    root: Path,
    yes: bool,
    non_interactive: bool,
    code_revision: str,
    retry_incomplete: bool = False,
) -> None:
    """The build's default pipeline: learn from every trace, then retest the misses.

    Args:
        console: Build command's terminal owner.
        store: Built project.
        project: Project name.
        root: Local ``.exp`` root holding the consent settings.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Replace an interrupted trace, buying its calls again.
    """
    recall = resolve_memory_recall(load_model_catalog(store.model_catalog_path))
    if recall is None:
        console.print(
            "[yellow]note[/yellow] the simulator was built without learning: learning needs a "
            "native decision model to recall memory (for example cloudflare/clef-flash on the "
            "Experiential gateway). Add one to the model catalog and run the build again."
        )
        return
    run_learning(
        console,
        store,
        project=project,
        root=root,
        recall_alias=recall,
        maximum_attempts=None,
        yes=yes,
        non_interactive=non_interactive,
        code_revision=code_revision,
        retry_incomplete=retry_incomplete,
    )
    plan = learning_plan(store, scope="all")
    missed = [result for _, _, result in learned_runs(store, plan) if result.exact < result.steps]
    if not missed:
        return
    from exp.cli.evaluation.miss_retest import run_miss_retest

    console.print()
    run_miss_retest(
        project,
        len(missed),
        root=root,
        yes=yes,
        non_interactive=non_interactive,
        dry_run=False,
        retry_incomplete=retry_incomplete,
    )
