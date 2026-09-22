"""Transactional, content-addressed trace imports in the shared local traffic database."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    SourceIdentity,
    canonical_json_bytes,
    sha256_json,
    stable_id,
    validate_artifact_id,
)
from exp.common.traces.sqlite_schema import TraceStoreError, initialize_schema, validate_schema
from exp.common.traces.trace import Trace, TraceSource

_JSON_OBJECT = TypeAdapter(JsonObject)


class StoredTraceImport(ContractModel):
    """Immutable normalization evidence shared by every project selecting this import.

    Attributes:
        import_id: Content identity of the complete normalized import.
        source_format: Declared normalizer, such as chat-json or gateway.
        source: Original file or scoped capture provenance.
        traces: Exact normalized records in source order.
        metadata: Producer-owned normalization issues and model-identity provenance.
        created_at: First successful persistence time, unchanged on repeated imports.
    """

    import_id: str
    source_format: str = Field(min_length=1)
    source: SourceIdentity
    traces: tuple[Trace, ...]
    metadata: JsonObject
    created_at: AwareDatetime


class TraceImportReceipt(ContractModel):
    """Committed import identity and counts for one project association.

    Attributes:
        project_id: Local project namespace, independent of gateway authentication.
        import_id: Immutable import selected by this project.
        trace_count: Number of accepted trace records in the import.
        new_records: Number of canonical records newly added to the shared store.
        already_linked: Whether this project already selected the same complete import.
    """

    project_id: str
    import_id: str
    trace_count: int = Field(ge=0)
    new_records: int = Field(ge=0)
    already_linked: bool


def _record(trace: Trace) -> JsonObject:
    """Separate per-import provenance from reusable canonical trace content."""
    return trace.model_dump(mode="json", exclude={"source"})


def _import_id(
    source_format: str, source: SourceIdentity, traces: Sequence[Trace], metadata: JsonObject
) -> str:
    """Bind an import to its source, ordered records, and complete normalization metadata."""
    return stable_id(
        "import",
        {
            "source_format": source_format,
            "source": source.model_dump(mode="json"),
            "records": [
                {
                    "sha256": sha256_json(_record(trace)),
                    "source": trace.source.model_dump(mode="json"),
                }
                for trace in traces
            ],
            "metadata": metadata,
        },
    )


class SQLiteTraceStore:
    """Shared immutable trace storage with atomic, idempotent project membership.

    Construction and reads create no files. Only write_import obtains write authority.
    Native capture retains ownership of gateway_captures and its retention policy.
    Canonical imports preserve selected evidence independently of capture expiration.
    """

    def __init__(self, database_path: Path) -> None:
        """Bind the content database path without opening or creating it."""
        self.path = database_path.resolve()

    @contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Own one connection and report storage errors without exposing trace payloads."""
        try:
            if write:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
            mode = "rw" if write else "ro"
            connection = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=5)
            try:
                connection.execute("PRAGMA foreign_keys=ON")
                validate_schema(connection)
                if write:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.execute("BEGIN IMMEDIATE")
                    initialize_schema(connection)
                else:
                    connection.execute("BEGIN")
                yield connection
                if write:
                    connection.commit()
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exc:
            raise TraceStoreError(
                "Cannot access trace database; check its path, permissions, and free space."
            ) from exc

    def write_import(
        self,
        project_id: str,
        *,
        source_format: str,
        source: SourceIdentity,
        traces: Sequence[Trace],
        metadata: JsonObject,
    ) -> TraceImportReceipt:
        """Atomically retain normalized evidence and associate it with a project.

        Args:
            project_id: Local namespace receiving this import.
            source_format: Explicit source loader name.
            source: Immutable source identity supplied by normalization.
            traces: Ordered accepted records; their original sources remain intact.
            metadata: All exclusions and model-span identity evidence from normalization.

        Returns:
            A receipt committed only after all records and project membership are durable.

        Raises:
            TraceStoreError: Stored evidence conflicts or a write cannot complete.
            ValueError: A project or source name is invalid.
        """
        validate_artifact_id(project_id)
        if not source_format.strip():
            raise ValueError("source_format must not be empty")
        traces = tuple(traces)
        import_id = _import_id(source_format, source, traces, metadata)
        header = (
            source_format,
            canonical_json_bytes(source).decode(),
            canonical_json_bytes(metadata).decode(),
        )
        records = tuple(
            (
                sha256_json(_record(trace)),
                trace.trace_id,
                canonical_json_bytes(_record(trace)).decode(),
                canonical_json_bytes(trace.source).decode(),
            )
            for trace in traces
        )
        new_records = 0
        with self._connect(write=True) as connection:
            exists = connection.execute(
                "SELECT source_format,source,metadata FROM trace_imports WHERE import_id=?",
                (import_id,),
            ).fetchone()
            if exists is None:
                connection.execute(
                    "INSERT INTO trace_imports VALUES (?, ?, ?, ?, ?)",
                    (
                        import_id,
                        *header,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                for ordinal, (digest, trace_id, payload, provenance) in enumerate(records):
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO trace_records VALUES (?, ?, ?)",
                        (digest, trace_id, payload),
                    )
                    new_records += inserted.rowcount
                    saved = connection.execute(
                        "SELECT trace_id,payload FROM trace_records WHERE record_sha256=?",
                        (digest,),
                    ).fetchone()
                    if saved != (trace_id, payload):
                        raise TraceStoreError(
                            "Canonical trace content differs from its stored digest."
                        )
                    connection.execute(
                        "INSERT INTO trace_import_records VALUES (?, ?, ?, ?)",
                        (import_id, ordinal, digest, provenance),
                    )
            else:
                saved = tuple(
                    connection.execute(
                        "SELECT m.ordinal,r.record_sha256,r.trace_id,r.payload,m.source "
                        "FROM trace_import_records m JOIN trace_records r "
                        "ON r.record_sha256=m.record_sha256 WHERE m.import_id=? ORDER BY m.ordinal",
                        (import_id,),
                    )
                )
                expected = tuple((ordinal, *record) for ordinal, record in enumerate(records))
                if exists != header or saved != expected:
                    raise TraceStoreError(
                        "Stored import evidence differs from its immutable identity."
                    )
            linked = connection.execute(
                "INSERT OR IGNORE INTO trace_project_imports(project_id,import_id) VALUES (?, ?)",
                (project_id, import_id),
            ).rowcount
        return TraceImportReceipt(
            project_id=project_id,
            import_id=import_id,
            trace_count=len(traces),
            new_records=new_records,
            already_linked=linked == 0,
        )

    def list_imports(self, project_id: str) -> tuple[str, ...]:
        """Read project import identities in selection order without creating missing storage."""
        validate_artifact_id(project_id)
        if not self.path.exists():
            return ()
        with self._connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trace_project_imports'"
            ).fetchone():
                return ()
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT import_id FROM trace_project_imports "
                    "WHERE project_id=? ORDER BY sequence",
                    (project_id,),
                )
            )

    def read_import(self, import_id: str) -> StoredTraceImport:
        """Read and verify a frozen import, including every source and normalization exclusion."""
        try:
            with self._connect() as connection:
                result = self._read_import(connection, import_id)
                if (
                    _import_id(result.source_format, result.source, result.traces, result.metadata)
                    != import_id
                ):
                    raise TraceStoreError(
                        "Stored import evidence differs from its immutable identity."
                    )
                return result
        except ValidationError as exc:
            raise TraceStoreError(
                "Stored import is corrupt; restore it from a verified source."
            ) from exc

    @staticmethod
    def _read_import(connection: sqlite3.Connection, import_id: str) -> StoredTraceImport:
        """Reconstruct one import through an already owned transaction."""
        row = connection.execute(
            "SELECT source_format,source,metadata,created_at FROM trace_imports WHERE import_id=?",
            (import_id,),
        ).fetchone()
        if row is None:
            raise TraceStoreError("Trace import was not found; select a saved import identity.")
        traces = []
        for ordinal, trace_id, digest, payload, source in connection.execute(
            "SELECT m.ordinal,r.trace_id,r.record_sha256,r.payload,m.source "
            "FROM trace_import_records m "
            "JOIN trace_records r ON r.record_sha256=m.record_sha256 "
            "WHERE m.import_id=? ORDER BY m.ordinal",
            (import_id,),
        ):
            content = _JSON_OBJECT.validate_json(payload)
            if (
                ordinal != len(traces)
                or content.get("trace_id") != trace_id
                or sha256_json(content) != digest
            ):
                raise TraceStoreError("Stored trace bytes do not match their content digest.")
            traces.append(
                Trace.model_validate({**content, "source": TraceSource.model_validate_json(source)})
            )
        return StoredTraceImport(
            import_id=import_id,
            source_format=row[0],
            source=SourceIdentity.model_validate_json(row[1]),
            metadata=_JSON_OBJECT.validate_json(row[2]),
            created_at=row[3],
            traces=tuple(traces),
        )
