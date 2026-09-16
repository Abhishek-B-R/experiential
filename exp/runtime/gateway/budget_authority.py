"""Pinned-graph validation outside budget write locks, with transactional revision fencing."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from exp.common.config.settings import GatewayResourceSettings
from exp.common.models.gateway_catalog import (
    NormalizedGatewayCatalog,
    read_pinned_normalized_snapshot,
)
from exp.common.models.gateway_chains import expand_model_chain
from exp.runtime.gateway.snapshot_file import read_snapshot_bytes

MAXIMUM_BUDGET_SNAPSHOT_BYTES = GatewayResourceSettings().budget_snapshot_max_bytes


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
    database_path: Path,
    snapshot_ref: str,
    digest: str,
    maximum_bytes: int = MAXIMUM_BUDGET_SNAPSHOT_BYTES,
) -> NormalizedGatewayCatalog:
    """Read a safe regular snapshot within the operator's authoring resource budget."""
    payload = read_snapshot_bytes(database_path.parent, snapshot_ref, maximum_bytes)
    return read_pinned_normalized_snapshot(payload, digest)


def validate_budget_revision(
    database_path: Path,
    revision: BudgetAliasRevision,
    pool_id: str,
    deployment_id: str | None,
    maximum_bytes: int,
) -> BudgetAliasRevision:
    """Validate a previously read alias revision outside its later write transaction."""
    if revision.pool_id == pool_id and deployment_id is None:
        return revision
    try:
        catalog = read_budget_snapshot(
            database_path, revision.snapshot_ref, revision.catalog_sha256, maximum_bytes
        )
    except OSError as exc:
        raise ValueError("budget scope catalog snapshot is unreadable") from exc
    require_reachable_budget_target(catalog, revision.pool_id, pool_id, deployment_id)
    return revision


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
