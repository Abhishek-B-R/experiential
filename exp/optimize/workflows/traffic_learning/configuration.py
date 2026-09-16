"""Optional traffic-mining and world-model choices outside the CLaaS learning core."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.claas import ClaasScope
from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.common.core.locks import file_write_lock
from exp.optimize.claas.configuration import application_directory


class TrafficWorkflowConfig(ContractModel):
    """One explicit application workflow that derives simulation tasks from traffic."""

    schema_version: Literal[1] = 1
    scope: ClaasScope
    world_model_alias: str = Field(min_length=1, max_length=512)
    judge_alias: str = Field(min_length=1, max_length=512)
    maximum_source_experiences: int = Field(default=100, strict=True, ge=1, le=256)


def load_workflow(root: Path, scope: ClaasScope) -> TrafficWorkflowConfig:
    """Load only the separately selected traffic workflow for this application."""
    path = application_directory(root, scope) / "traffic-workflow.json"
    try:
        config = TrafficWorkflowConfig.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise ValueError(
            "traffic learning is not configured; use claas init with --world-model and --judge, "
            "or supply scenarios and an environment through the CLaaS Python API"
        ) from None
    if config.scope != scope:
        raise ValueError("traffic workflow belongs to another application")
    return config


def save_workflow(root: Path, config: TrafficWorkflowConfig, *, replace: bool = False) -> Path:
    """Persist an explicit workflow selection without changing generic learning settings."""
    path = application_directory(root, config.scope) / "traffic-workflow.json"
    with file_write_lock(path, what="traffic learning configuration"):
        if path.exists():
            previous = load_workflow(root, config.scope)
            if previous == config:
                return path
            if not replace:
                raise ValueError("traffic workflow already exists; use --replace to edit settings")
        write_text_atomic(path, config.model_dump_json(indent=2) + "\n")
    return path
