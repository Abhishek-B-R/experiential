"""Checkpoint digest and scope rejection tests without executing model payloads."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.backends.checkpoints import CheckpointManifest, hash_file, verify_checkpoint
from exp.optimize.claas.training_contracts import TrainingCheckpoint
from exp.optimize.claas.training_contracts_test import spec


def checkpoint(root: Path) -> TrainingCheckpoint:
    """Write inert files with a complete manifest for verification-only tests."""
    paths = (
        "student/adapter_config.json",
        "student/adapter_model.safetensors",
        "teacher/adapter_config.json",
        "teacher/adapter_model.safetensors",
        "optimizer.pt",
    )
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inert-test-payload")
    manifest = CheckpointManifest(
        spec=spec(),
        policy_revision="policy-1",
        parent_policy_revision="policy-0",
        policy_history=("policy-1", "policy-0"),
        step=1,
        batch_id="batch-1",
        consumed_experience_ids=("experience-1",),
        files={name: hash_file(root / name) for name in paths},
    )
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return TrainingCheckpoint(
        scope=spec().scope,
        adapter_id=spec().adapter_id,
        policy_revision="policy-1",
        policy_history=manifest.policy_history,
        step=1,
        path=str(root),
        manifest_sha256=sha256_json(manifest),
    )


def test_verifies_before_loading_and_rejects_changed_payload(tmp_path: Path) -> None:
    """A valid manifest does not authorize a subsequently changed optimizer pickle."""
    receipt = checkpoint(tmp_path)
    assert verify_checkpoint(receipt, spec()).step == 1
    (tmp_path / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_checkpoint(receipt, spec())


def test_rejects_unbound_ancestry_and_extra_payload(tmp_path: Path) -> None:
    """A claimed replay ancestor must be contained in the immutable manifest."""
    receipt = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        verify_checkpoint(
            receipt.model_copy(update={"policy_history": ("policy-1", "foreign")}), spec()
        )
    (tmp_path / "extra.pt").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="outside"):
        verify_checkpoint(receipt, spec())
