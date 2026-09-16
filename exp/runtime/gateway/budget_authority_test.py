"""Pinned descendant budgets reject stale authority and unsafe local snapshot files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models.gateway_catalog import CatalogSnapshotDigestError, NormalizedGatewayCatalog
from exp.runtime.gateway import budget_authority as budgets_module
from exp.runtime.gateway.budget_authority import (
    MAXIMUM_BUDGET_SNAPSHOT_BYTES,
    read_budget_snapshot,
    require_reachable_budget_target,
)
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind
from exp.runtime.gateway.budgets_test import _activate_chain, _authority, _chain_catalog, _Clock


def test_child_authority_checks_only_reachable_pool_leaves() -> None:
    """A pool member omitted from the authored graph is not a new budget target."""
    catalog = _chain_catalog()
    require_reachable_budget_target(catalog, "pool", "child-pool", "child")
    with pytest.raises(ValueError, match="not reachable"):
        require_reachable_budget_target(catalog, "pool", "child-pool", "primary")
    unavailable = catalog.model_copy(
        update={"model_chains": (catalog.model_chains[0].model_copy(update={"available": False}),)}
    )
    with pytest.raises(ValueError, match="not reachable"):
        require_reachable_budget_target(unavailable, "pool", "child-pool", None)


def test_retarget_during_snapshot_read_refuses_budget_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File parsing happens outside the write lock; the transaction checks the exact revision."""
    clock = _Clock()
    store, _ledger, budgets, _key = _authority(tmp_path, clock)
    _activate_chain(store, tmp_path)
    original = budgets_module.read_budget_snapshot

    def retarget(path: Path, ref: str, digest: str, maximum_bytes: int) -> NormalizedGatewayCatalog:
        """Move active authority while the preflight reads its original frozen file."""
        catalog = original(path, ref, digest, maximum_bytes)
        _activate_chain(store, tmp_path, revision_id="changed", pool_id="child-pool")
        return catalog

    monkeypatch.setattr(budgets_module, "read_budget_snapshot", retarget)
    with pytest.raises(ValueError, match="revision changed"):
        budgets.set_limit(
            organization_id="org",
            period="2026-08",
            scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
            limit_nano_usd=100,
        )
    assert budgets.limits(organization_id="org", period="2026-08") == ()


@pytest.mark.parametrize("kind", ["escape", "symlink", "fifo", "oversize", "corrupt"])
def test_snapshot_file_boundary(tmp_path: Path, kind: str) -> None:
    """No device blocking, path escape, unlimited read, or silent malformed-data fallback."""
    path = tmp_path / "snapshot"
    if kind == "symlink":
        path.symlink_to(tmp_path / "elsewhere")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO file type is POSIX-only; Windows rejects device paths separately")
        os.mkfifo(path)
    elif kind == "oversize":
        with path.open("wb") as stream:
            stream.truncate(MAXIMUM_BUDGET_SNAPSHOT_BYTES + 1)
    elif kind == "corrupt":
        path.write_text("{broken")
    with pytest.raises((ValueError, OSError)):
        read_budget_snapshot(
            tmp_path / "gateway.db", "../outside" if kind == "escape" else "snapshot", "a" * 64
        )


def test_snapshot_bound_accepts_exact_limit_and_rejects_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The byte limit is inclusive and cannot replace pinned content validation."""
    catalog = _chain_catalog()
    payload = canonical_json_bytes(catalog.model_dump(mode="json"))
    (tmp_path / "snapshot").write_bytes(payload)
    assert (
        read_budget_snapshot(
            tmp_path / "gateway.db", "snapshot", catalog.identity_sha256(), len(payload)
        )
        == catalog
    )
    with pytest.raises(CatalogSnapshotDigestError):
        read_budget_snapshot(tmp_path / "gateway.db", "snapshot", "f" * 64, len(payload))
    with pytest.raises(ValueError, match="resource budget"):
        read_budget_snapshot(
            tmp_path / "gateway.db", "snapshot", catalog.identity_sha256(), len(payload) - 1
        )


def test_snapshot_directory_symlink_and_directory_file_are_refused(tmp_path: Path) -> None:
    """Neither an intermediate symlink nor a directory may masquerade as a snapshot."""
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        read_budget_snapshot(tmp_path / "gateway.db", "linked/snapshot", "f" * 64)
    with pytest.raises((ValueError, OSError)):
        read_budget_snapshot(tmp_path / "gateway.db", "real", "f" * 64)


@pytest.mark.skipif(os.name == "nt", reason="Windows uses the kernel32 handle backend")
def test_snapshot_loader_fails_explicitly_without_safe_handle_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported systems never fall back to following potentially unsafe paths."""
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match="unsupported on this operating system"):
        read_budget_snapshot(tmp_path / "gateway.db", "snapshot", "f" * 64)
