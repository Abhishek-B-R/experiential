"""Application configuration boundaries and immutable adapter identities."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.claas.contracts import ClaasScope
from exp.optimize.claas.configuration import (
    CycleLimits,
    LocalClaasConfig,
    application_directory,
    load_configuration,
    save_configuration,
)


def _config(scope: ClaasScope | None = None) -> LocalClaasConfig:
    """Build a minimal local application configuration without provider construction."""
    return LocalClaasConfig(
        scope=scope or ClaasScope(user_id="local", application_id="claims"),
        base_model="test-model",
        base_model_revision="base-v1",
        tokenizer_id="test-model",
        tokenizer_revision="tokenizer-v1",
    )


def test_application_identity_cannot_escape_root(tmp_path: Path) -> None:
    """Hash untrusted scope labels into a bounded application directory."""
    scope = ClaasScope(user_id="../../elsewhere", application_id="/absolute/path")
    path = application_directory(tmp_path, scope)
    assert path.parent == tmp_path / "claas"
    assert len(path.name) == 64
    assert path != application_directory(
        tmp_path, ClaasScope(user_id="another", application_id="/absolute/path")
    )


def test_configuration_roundtrip_and_explicit_replacement(tmp_path: Path) -> None:
    """Require explicit replacement for changed mutable application settings."""
    config = _config()
    path = save_configuration(tmp_path, config)
    assert load_configuration(tmp_path, config.scope) == config
    assert save_configuration(tmp_path, config) == path
    changed = config.model_copy(update={"interval_seconds": 7200})
    with pytest.raises(ValueError, match="already exists"):
        save_configuration(tmp_path, changed)
    save_configuration(tmp_path, changed, replace=True)
    assert load_configuration(tmp_path, config.scope).interval_seconds == 7200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_model", "different"),
        ("base_model_revision", "base-v2"),
        ("tokenizer_revision", "tokenizer-v2"),
        ("lora_rank", 32),
    ],
)
def test_configuration_cannot_reuse_adapters_for_changed_base(
    tmp_path: Path, field: str, value: str | int
) -> None:
    """Preserve the immutable model and LoRA shape of existing application state."""
    config = _config()
    save_configuration(tmp_path, config)
    changed = LocalClaasConfig.model_validate(config.model_dump() | {field: value})
    with pytest.raises(ValueError, match="immutable"):
        save_configuration(tmp_path, changed, replace=True)
    assert load_configuration(tmp_path, config.scope) == config


def test_configuration_rejects_wrong_embedded_scope(tmp_path: Path) -> None:
    """Reject a configuration file whose embedded owner differs from its path."""
    config = _config()
    path = save_configuration(tmp_path, config)
    path.write_text(_config(ClaasScope(user_id="other", application_id="claims")).model_dump_json())
    with pytest.raises(ValueError, match="another application"):
        load_configuration(tmp_path, config.scope)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0.0, -1.0])
def test_cycle_budget_is_positive_and_finite(value: float) -> None:
    """Reject zero, negative, and nonfinite cycle spending ceilings."""
    with pytest.raises(ValidationError):
        CycleLimits(maximum_cost_usd=value)
