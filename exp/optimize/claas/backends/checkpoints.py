"""Content verification for complete CLaaS student, teacher, and optimizer checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.optimize.claas.training_contracts import ClaasTrainingSpec, TrainingCheckpoint


class CheckpointManifest(ContractModel):
    """Exact frozen configuration and digests for every resumable state file."""

    spec: ClaasTrainingSpec
    policy_revision: str = Field(min_length=1)
    parent_policy_revision: str = Field(min_length=1)
    policy_history: tuple[str, ...] = Field(min_length=1)
    step: int = Field(strict=True, ge=1)
    batch_id: str = Field(min_length=1)
    consumed_experience_ids: tuple[str, ...] = Field(min_length=1)
    files: dict[str, Sha256] = Field(min_length=1)


def hash_file(path: Path) -> str:
    """Hash a regular payload without loading model state or executing pickle."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    checkpoint: TrainingCheckpoint, spec: ClaasTrainingSpec
) -> CheckpointManifest:
    """Fail before compute if the checkpoint, scope, recipe, or any payload has drifted."""
    root = Path(checkpoint.path)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("checkpoint must name an absolute immutable directory, without symlinks")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("checkpoint manifest cannot be a symlink")
    manifest = CheckpointManifest.model_validate_json(manifest_path.read_text())
    if (
        sha256_json(manifest) != checkpoint.manifest_sha256
        or manifest.spec != spec
        or manifest.policy_revision != checkpoint.policy_revision
        or manifest.step != checkpoint.step
        or manifest.policy_history != checkpoint.policy_history
        or manifest.policy_history[0] != checkpoint.policy_revision
        or len(set(manifest.policy_history)) != len(manifest.policy_history)
        or len(manifest.policy_history) > min(checkpoint.step + 1, spec.max_policy_lag + 1)
        or checkpoint.scope != spec.scope
        or checkpoint.adapter_id != spec.adapter_id
    ):
        raise ValueError("checkpoint identity, training recipe, or manifest digest does not match")
    for relative, expected in manifest.files.items():
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or relative == "manifest.json":
            raise ValueError("checkpoint contains an invalid payload path")
        payload = root / relative
        if any(part.is_symlink() for part in (payload, *payload.parents)):
            raise ValueError("checkpoint payload cannot be a symlink")
        if not payload.is_file() or hash_file(payload) != expected:
            raise ValueError(f"checkpoint payload is missing or changed: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest.files) | {"manifest.json"}:
        raise ValueError("checkpoint contains files outside its manifest")
    required = {
        "student/adapter_config.json",
        "student/adapter_model.safetensors",
        "teacher/adapter_config.json",
        "teacher/adapter_model.safetensors",
        "optimizer.pt",
    }
    if not required.issubset(manifest.files):
        raise ValueError("checkpoint lacks student, teacher, or resumable optimizer state")
    return manifest
