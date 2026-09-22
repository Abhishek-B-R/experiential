"""Persist canonical source conversions in SQLite without mining or provider setup."""

from pathlib import Path

from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject
from exp.common.traces.sqlite import SQLiteTraceStore, TraceImportReceipt
from exp.common.traces.sqlite_schema import trace_database_path
from exp.simulation.ingest.model_identity import TraceModelIdentityEvidence
from exp.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from exp.simulation.ingest.sources import load_trace_source

_ISSUES = TypeAdapter(tuple[TraceNormalizationIssue, ...])
_IDENTITY = TypeAdapter(tuple[TraceModelIdentityEvidence, ...] | None)


def ingest_traces(
    project_id: str,
    *,
    root: Path,
    source_format: str,
    path: Path,
    identity_id: str | None = None,
    dry_run: bool = False,
) -> tuple[TraceNormalizationResult, TraceImportReceipt | None]:
    """Normalize one explicit source and durably select it for a local project.

    Args:
        project_id: Project namespace for the committed import.
        root: Local workspace containing the shared gateway content database.
        source_format: Declared file format or gateway capture source.
        path: Source export or gateway SQLite database.
        identity_id: Required authenticated capture scope for gateway reads.
        dry_run: Validate and report normalization without creating any storage.

    Returns:
        Complete normalization evidence and, unless dry_run, its committed SQLite receipt.
    """
    source_format = source_format.strip().casefold()
    result = load_trace_source(source_format, path, identity_id=identity_id)
    if dry_run:
        return result, None
    metadata: JsonObject = {
        "schema_version": 1,
        "issues": _ISSUES.dump_python(result.issues, mode="json"),
        "identity_evidence": _IDENTITY.dump_python(result.identity_evidence, mode="json"),
    }
    if result.source is None:
        raise ValueError("trace source identity is missing; use a canonical source loader")
    receipt = SQLiteTraceStore(trace_database_path(root)).write_import(
        project_id,
        source_format=source_format,
        source=result.source,
        traces=result.traces,
        metadata=metadata,
    )
    return result, receipt


def read_ingested_traces(root: Path, import_id: str) -> TraceNormalizationResult:
    """Load an exact stored import for later task construction without reading its source again."""
    stored = SQLiteTraceStore(trace_database_path(root)).read_import(import_id)
    if stored.metadata.get("schema_version") != 1:
        raise ValueError(
            "unsupported normalized import metadata; use a matching Experiential release"
        )
    return TraceNormalizationResult(
        traces=stored.traces,
        issues=_ISSUES.validate_python(stored.metadata["issues"]),
        identity_evidence=_IDENTITY.validate_python(stored.metadata["identity_evidence"]),
        source=stored.source,
    )
