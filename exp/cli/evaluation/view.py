"""Terminal result tables and side-by-side rollout inspection."""

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from exp.cli.shared.picker import PickerOption, choose_many, choose_one
from exp.common.project import ProjectStore
from exp.optimize.evaluation.export import export_report, load_report_evidence, rollout_transcript
from exp.optimize.evaluation.runs import EvaluationRun


def render_report(console: Console, project: ProjectStore, run: EvaluationRun) -> None:
    """Present measured quality, candidate cost, latency, coverage, and local report paths."""
    evidence = load_report_evidence(project, run)
    report = evidence.report
    console.print(f"\n[bold]Results · {project.paths.project_id}[/bold]")
    console.print(
        f"{len(evidence.tasks)} distinct scenarios · "
        f"{report.compared_cells} shared valid scenario/repeat pairs · "
        f"{report.excluded_cells} excluded"
    )
    table = Table(
        "Model",
        "Quality /100",
        "Assistant / task",
        "Assistant latency",
        "Valid",
        "Invalid",
        "Incomplete",
        box=None,
    )
    for metric in report.models:
        table.add_row(
            metric.candidate.alias,
            _number(metric.quality * 100 if metric.quality is not None else None, ""),
            _number(metric.operating_cost_usd, "$"),
            _number(metric.latency_seconds, "", "s"),
            str(metric.scored_cells),
            str(metric.failed_cells),
            str(metric.incomplete_cells),
        )
    console.print(table)
    json_path, html_path = export_report(project, run)
    console.print(f"Report and Pareto plot: {html_path}", markup=False)
    console.print(f"Portable data: {json_path}", markup=False)
    console.print(
        f"Accounted experiment spend: simulation {_number(run.simulation_cost_usd, '$')} · "
        f"judge {_number(run.judge_cost_usd, '$')}"
    )


def inspect_report(console: Console, project: ProjectStore, run: EvaluationRun) -> None:
    """Browse every scenario with one or two selected model traces in the terminal."""
    evidence = load_report_evidence(project, run)
    aliases = tuple(model.candidate.alias for model in evidence.report.models)
    while True:
        task = choose_one(
            console,
            title="Inspect a scenario (Esc to return)",
            options=tuple(
                PickerOption(item.task_id, item.instruction[:120], item.task_id)
                for item in evidence.tasks
            ),
        )
        if not task.values:
            return
        models = choose_many(
            console,
            title="Compare one or two models",
            minimum=1,
            options=tuple(PickerOption(alias, alias) for alias in aliases),
            preselected=aliases[:2],
        )
        if not models.values:
            continue
        if len(models.values) > 2:
            console.print("Choose at most two models for side-by-side inspection.")
            continue
        repeats = sorted({row.repeat for row in evidence.rows if row.task_id == task.values[0]})
        repeat = choose_one(
            console,
            title="Repeat",
            options=tuple(PickerOption(str(value), f"Run {value + 1}") for value in repeats),
        )
        if not repeat.values:
            continue
        panels = []
        for alias in models.values:
            matching = [
                row
                for row in evidence.rows
                if row.task_id == task.values[0]
                and row.candidate_alias == alias
                and row.repeat == int(repeat.values[0])
            ]
            for row in matching:
                rollout = next(
                    (item for item in evidence.rollouts if item.rollout_id == row.rollout_id), None
                )
                text = rollout_transcript(rollout) if rollout else "No saved rollout"
                panels.append(
                    Panel(Text(text), title=f"{alias} · repeat {row.repeat + 1} · {row.status}")
                )
        console.print(Columns(panels, equal=True, expand=True))


def _number(value: float | None, prefix: str, suffix: str = "") -> str:
    """Format a measurement without turning missing or small positive values into zero."""
    if value is None:
        return "unavailable"
    precision = 6 if 0 < abs(value) < 0.001 else 4
    rendered = f"{value:.{precision}f}"
    if value != 0 and float(rendered) == 0:
        rendered = f"{value:.3g}"
    return f"{prefix}{rendered}{suffix}"
