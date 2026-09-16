"""Persist bounded local continual-learning applications without provider secrets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.claas.contracts import ClaasScope
from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.core.files import write_text_atomic
from exp.common.core.locks import file_write_lock


class CycleLimits(ContractModel):
    """Finite work and spending ceilings for one learning cycle."""

    maximum_scenarios: int = Field(default=16, strict=True, ge=1, le=10_000)
    maximum_rollouts_per_scenario: int = Field(default=4, strict=True, ge=1, le=128)
    maximum_episode_steps: int = Field(default=8, strict=True, ge=1, le=256)
    maximum_response_tokens: int = Field(default=2048, strict=True, ge=1, le=131_072)
    maximum_training_seconds: int = Field(default=3600, strict=True, ge=1, le=86_400)
    maximum_cost_usd: float = Field(default=5.0, gt=0, allow_inf_nan=False)


class PromotionPolicy(ContractModel):
    """Requirements for activating a candidate measured against the active adapter."""

    minimum_evaluation_tasks: int = Field(default=10, strict=True, ge=1)
    minimum_score_improvement: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    maximum_hard_failures: Literal[0] = 0


class LocalClaasConfig(ContractModel):
    """One local application's immutable model choices and editable cycle settings.

    Scenarios, environments, and evaluation are supplied by callers. This
    configuration requires no traffic source, world model, or judge provider.
    """

    schema_version: Literal[2] = 2
    scope: ClaasScope
    base_model: str = Field(min_length=1, max_length=512)
    base_model_revision: str = Field(min_length=1, max_length=512)
    tokenizer_id: str = Field(min_length=1, max_length=512)
    tokenizer_revision: str = Field(min_length=1, max_length=512)
    backend: Literal["verl"] = "verl"
    compute: Literal["local", "modal"] = "local"
    objective: Literal["sdpo", "reinforce", "hybrid"] = "sdpo"
    lora_rank: int = Field(default=16, strict=True, ge=1, le=256)
    learning_rate: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    interval_seconds: int = Field(default=3600, strict=True, ge=60, le=2_592_000)
    limits: CycleLimits = Field(default_factory=CycleLimits)
    promotion: PromotionPolicy = Field(default_factory=PromotionPolicy)


def application_directory(root: Path, scope: ClaasScope) -> Path:
    """Locate one scope without interpreting user or application IDs as paths.

    Args:
        root: User-selected Experiential artifact root.
        scope: Exact local learning identity.

    Returns:
        A stable directory whose name is a full content hash of the scope.
    """
    return root / "claas" / sha256_json(scope)


def load_configuration(root: Path, scope: ClaasScope) -> LocalClaasConfig:
    """Load a strict configuration and verify its embedded identity.

    Args:
        root: User-selected Experiential artifact root.
        scope: Application to load.

    Returns:
        The persisted, validated local configuration.

    Raises:
        ValueError: The configuration is missing, malformed, or belongs elsewhere.
    """
    path = application_directory(root, scope) / "config.json"
    try:
        config = LocalClaasConfig.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise ValueError(
            f"CLaaS application {scope.application_id!r} is not configured; "
            "run exp optimize claas init first"
        ) from None
    if config.scope != scope:
        raise ValueError(f"CLaaS configuration at {path} belongs to another application")
    return config


def save_configuration(root: Path, config: LocalClaasConfig, *, replace: bool = False) -> Path:
    """Initialize or explicitly update settings without changing an adapter's base.

    Args:
        root: User-selected Experiential artifact root.
        config: Validated model and learning settings without provider credentials.
        replace: Permit changes to cycle settings of an existing application.

    Returns:
        The configuration file written atomically while holding its writer lock.

    Raises:
        ValueError: Initialization conflicts or replacement changes model identity.
    """
    path = application_directory(root, config.scope) / "config.json"
    with file_write_lock(path, what="CLaaS configuration"):
        if path.exists():
            previous = load_configuration(root, config.scope)
            if previous == config:
                return path
            if not replace:
                raise ValueError("CLaaS application already exists; use --replace to edit settings")
            identity_fields = (
                "base_model",
                "base_model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "lora_rank",
            )
            if any(getattr(previous, key) != getattr(config, key) for key in identity_fields):
                raise ValueError(
                    "the application's base model, tokenizer, and LoRA rank are immutable; "
                    "initialize another application for a different model"
                )
        write_text_atomic(path, config.model_dump_json(indent=2) + "\n")
    return path
