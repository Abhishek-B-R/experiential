"""Import file or gateway traces into the shared local SQLite content store."""

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

from exp.cli.shared.consent import can_prompt
from exp.cli.shared.options import ROOT_OPTION
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import validate_artifact_id
from exp.common.traces.sqlite_schema import trace_database_path
from exp.simulation.ingest.persistence import ingest_traces
from exp.simulation.ingest.sources import CANONICAL_TRACE_SOURCES

_console = Console(theme=EXP_THEME)
_TRACE_OPTION = typer.Option(
    None, "--traces", help="JSON, JSONL, OTel export, or gateway database."
)


def ingest(
    project: str = typer.Argument(..., metavar="PROJECT"),
    traces: Path | None = _TRACE_OPTION,
    source: str = typer.Option("chat-json", "--source", help="Declared trace format, or gateway."),
    root: Path = ROOT_OPTION,
    identity: str | None = typer.Option(
        None, "--identity", help="Gateway capture identity to import."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without writing SQLite."),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Store canonical traces and provenance without model setup or paid work.

    Args:
        project: Local project namespace receiving the import.
        traces: Explicit source path; gateway sources default to local traffic.db.
        source: Canonical loader name.
        root: Local workspace containing the shared traffic database.
        identity: Required gateway identity; never inferred from project names.
        dry_run: Validate without changing the database or project directory.
        non_interactive: Never prompt for a missing file or source.
    """
    source = source.strip().casefold()
    try:
        validate_artifact_id(project)
        if source not in CANONICAL_TRACE_SOURCES:
            raise ValueError(
                f"unsupported source; choose one of: {', '.join(CANONICAL_TRACE_SOURCES)}"
            )
        if source == "gateway":
            if identity is None:
                raise ValueError("--source gateway requires --identity ID")
            traces = traces or trace_database_path(root)
        elif identity is not None:
            raise ValueError("--identity requires --source gateway")
        if traces is None:
            if non_interactive or not can_prompt(_console):
                raise ValueError("provide --traces PATH, or run exp ingest in a terminal")
            traces = Path(Prompt.ask("Trace file", console=_console)).expanduser()
            source = Prompt.ask(
                "Trace format",
                choices=[name for name in CANONICAL_TRACE_SOURCES if name != "gateway"],
                default=source,
                console=_console,
            )
        result, receipt = ingest_traces(
            project,
            root=root,
            source_format=source,
            path=traces.expanduser(),
            identity_id=identity,
            dry_run=dry_run,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _console.print(f"{len(result.traces)} accepted traces · {len(result.issues)} excluded records")
    for issue in result.issues:
        _console.print(f"  {issue.source_record}: {issue.message}", markup=False)
    if receipt is None:
        _console.print("Dry run complete. No database or project files written.")
    else:
        disposition = "Already imported" if receipt.already_linked else "Imported"
        _console.print(
            f"{disposition} for {project}: {receipt.import_id} · {receipt.new_records} new records",
            markup=False,
        )
        _console.print(f"Database: {trace_database_path(root)}", markup=False)
