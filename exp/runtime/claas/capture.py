"""Explicit startup bindings for native local traffic capture."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from exp.common.claas import CapturePolicy
from exp.common.claas.contracts import Identifier
from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.common.core.locks import file_write_lock


class CaptureBinding(ContractModel):
    """Bind one authenticated user and gateway alias to one agent application."""

    alias: Identifier
    policy: CapturePolicy


class CaptureConfiguration(ContractModel):
    """One local content database, independent from content-free gateway accounting.

    The operator supplies a path explicitly. Scope identifiers never become path
    components. Each user/alias pair has at most one application binding.
    """

    database_path: Path
    bindings: tuple[CaptureBinding, ...]
    queue_capacity: int = Field(default=256, strict=True, ge=1, le=4096)

    @model_validator(mode="after")
    def _validate_bindings(self) -> CaptureConfiguration:
        """Require an absolute path and unambiguous capture authority."""
        if not self.database_path.is_absolute():
            raise ValueError("capture database_path must be absolute")
        keys = [(binding.policy.scope.user_id, binding.alias) for binding in self.bindings]
        if len(keys) != len(set(keys)):
            raise ValueError("capture bindings must have unique user_id and alias pairs")
        policies: dict[tuple[str, str], CapturePolicy] = {}
        for binding in self.bindings:
            scope = binding.policy.scope
            key = (scope.user_id, scope.application_id)
            previous = policies.setdefault(key, binding.policy)
            if previous != binding.policy:
                raise ValueError("aliases for the same application must share one capture policy")
        return self


def load_capture_configuration(root: Path) -> CaptureConfiguration | None:
    """Read opt-in bindings for the next gateway startup; missing means disabled."""
    path = root / "gateway" / "claas.json"
    if not path.exists():
        return None
    return CaptureConfiguration.model_validate_json(path.read_bytes())


def save_capture_binding(root: Path, binding: CaptureBinding) -> CaptureConfiguration:
    """Explicitly configure one alias and a consistent policy for its whole application.

    Existing bindings for other users or applications survive the atomic update.
    An alias already assigned to another application is rejected, so this
    operation cannot silently redirect existing learning traffic.
    """

    path = root / "gateway" / "claas.json"
    with file_write_lock(path, what="CLaaS capture configuration"):
        previous = load_capture_configuration(root)
        bindings: list[CaptureBinding] = []
        for item in previous.bindings if previous else ():
            if (
                item.alias == binding.alias
                and item.policy.scope.user_id == binding.policy.scope.user_id
            ):
                if item.policy.scope != binding.policy.scope:
                    raise ValueError("this user's alias is already bound to another application")
                continue
            if item.policy.scope == binding.policy.scope:
                item = CaptureBinding(alias=item.alias, policy=binding.policy)
            bindings.append(item)
        bindings.append(binding)
        content_root = root.resolve() / "claas"
        content_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        config = CaptureConfiguration(
            database_path=previous.database_path if previous else content_root / "traffic.sqlite",
            bindings=tuple(bindings),
            queue_capacity=previous.queue_capacity if previous else 256,
        )
        write_text_atomic(path, config.model_dump_json(indent=2) + "\n")
        return config
