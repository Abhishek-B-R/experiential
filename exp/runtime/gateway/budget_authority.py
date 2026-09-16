"""Pinned-graph validation outside budget write locks, with transactional revision fencing."""

from __future__ import annotations

import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path

from exp.common.models.gateway_catalog import (
    NormalizedGatewayCatalog,
    read_pinned_normalized_snapshot,
)
from exp.common.models.gateway_chains import expand_model_chain

MAXIMUM_BUDGET_SNAPSHOT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BudgetAliasRevision:
    """Alias authority read before file I/O and compared under the write transaction."""

    revision_id: str
    pool_id: str
    snapshot_ref: str
    catalog_sha256: str


def active_budget_revision(
    connection: sqlite3.Connection, organization_id: str, alias_id: str
) -> BudgetAliasRevision:
    """Read one active direct alias's exact revision, never mutable current catalog data."""
    row = connection.execute(
        """SELECT r.revision_id,r.target_kind,r.pool_id,r.snapshot_ref,r.catalog_sha256
        FROM gateway_aliases a JOIN alias_revisions r
          ON r.organization_id=a.organization_id AND r.alias_id=a.alias_id
         AND r.revision_id=a.active_revision_id
        WHERE a.organization_id=? AND a.alias_id=? AND a.active=1""",
        (organization_id, alias_id),
    ).fetchone()
    if row is None or row["target_kind"] != "direct":
        raise ValueError("budget pool requires an active direct alias revision")
    return BudgetAliasRevision(
        str(row["revision_id"]),
        str(row["pool_id"]),
        str(row["snapshot_ref"]),
        str(row["catalog_sha256"]),
    )


def read_budget_snapshot(
    database_path: Path, snapshot_ref: str, digest: str
) -> NormalizedGatewayCatalog:
    """Read a bounded regular snapshot without following any symlink path component."""
    if os.open not in os.supports_dir_fd or any(
        not hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ):
        raise ValueError(
            "budget snapshot authoring requires directory-relative no-follow file support; "
            "use a supported local host to configure deployment or child budgets"
        )
    relative = Path(snapshot_ref)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in (".", "..") for part in relative.parts)
    ):
        raise ValueError("budget catalog snapshot reference escapes gateway state")
    directory = os.open(database_path.parent.resolve(), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("budget catalog snapshot must be a regular file")
            if info.st_size > MAXIMUM_BUDGET_SNAPSHOT_BYTES:
                raise ValueError(
                    "budget catalog snapshot exceeds 64 MiB; "
                    "reduce the catalog before authoring its budget"
                )
            payload = stream.read(MAXIMUM_BUDGET_SNAPSHOT_BYTES + 1)
            final = os.fstat(stream.fileno())
            if (
                len(payload) != info.st_size
                or final.st_mtime_ns != info.st_mtime_ns
                or final.st_size != info.st_size
            ):
                raise ValueError("budget catalog snapshot changed or was truncated during read")
    finally:
        os.close(directory)
    return read_pinned_normalized_snapshot(payload, digest)


def require_reachable_budget_target(
    catalog: NormalizedGatewayCatalog,
    root_pool_id: str,
    pool_id: str,
    deployment_id: str | None,
) -> None:
    """Accept selected pools and reachable leaves of the bounded pinned model graph."""
    pools = {pool.pool_id: pool for pool in catalog.pools}
    root = pools.get(root_pool_id)
    if root is None:
        raise ValueError("budget root pool is missing from its pinned catalog")
    authored = next((c for c in catalog.model_chains if c.model_id == root.exact_model_id), None)
    if authored is None:
        allowed = {root_pool_id: set(root.deployment_ids)}
    else:
        if authored.pool_id != root_pool_id:
            raise ValueError("budget graph root differs from its active alias target")
        expanded = expand_model_chain(root.exact_model_id, catalog.chains_by_model())
        allowed: dict[str, set[str]] = {}
        for segment in expanded.segments:
            allowed.setdefault(segment.pool_id, set()).update(segment.deployment_ids)
    if pool_id not in allowed or (
        deployment_id is not None and deployment_id not in allowed[pool_id]
    ):
        raise ValueError("budget target is not reachable in its alias's pinned model chain")
