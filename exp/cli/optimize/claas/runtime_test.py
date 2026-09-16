"""CLI binding checks pin the same private inference server and gateway registry."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.optimize.claas.runtime import serving_binding
from exp.common.claas import ClaasScope
from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
)
from exp.optimize.claas.configuration import application_directory, save_configuration
from exp.optimize.claas.execution import save_execution_settings
from exp.optimize.claas.execution_test import settings
from exp.optimize.claas.lifecycle.cycle_test import config
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.claas.serving import load_gateway_serving_configuration
from exp.runtime.gateway.tests.launch_test import _configure_gateway


def test_bind_cli_pins_authorized_direct_origin_and_starts_paused(tmp_path: Path) -> None:
    """A real persisted alias is validated without executing private control calls."""
    manager, _ = _configure_gateway(tmp_path, base_url="http://127.0.0.1:8000/v1")
    manager.migrate_legacy_provider_connections()
    alias = manager.aliases()[0]
    assert alias.snapshot_ref and alias.revision_id
    catalog_path = manager.state_dir / alias.snapshot_ref
    manager.ensure_alias_provider_bindings(
        alias_id=alias.alias_id,
        alias_revision_id=alias.revision_id,
        catalog=ModelCatalog.model_validate_json(
            catalog_path.with_suffix(".models.json").read_bytes()
        ),
    )
    scope = ClaasScope(user_id="default", application_id="claims")
    configured = config().model_copy(update={"scope": scope, "base_model": "provider-model-exact"})
    save_configuration(tmp_path, configured)
    execution = tmp_path / "execution-input.json"
    execution.write_text(settings().model_dump_json())
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "claas",
            "bind",
            "claims",
            "--alias",
            "coding",
            "--execution",
            str(execution),
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert "--enable-sleep-mode" in output["vllm_command"]
    configuration = load_gateway_serving_configuration(tmp_path)
    assert configuration is not None
    binding = serving_binding(tmp_path, scope)
    assert json.loads(binding.state_path.read_text())["paused"]
    assert alias.catalog_sha256
    manager.activate_direct_alias(
        alias_id="replacement",
        alias_name="replacement",
        revision_id="replacement-revision",
        pool_id="coding",
        snapshot_ref=alias.snapshot_ref,
        catalog_sha256=alias.catalog_sha256,
    )
    manager.add_grant(identity_id="default", alias_id="replacement")
    manager.ensure_alias_provider_bindings(
        alias_id="replacement",
        alias_revision_id="replacement-revision",
        catalog=ModelCatalog.model_validate_json(
            catalog_path.with_suffix(".models.json").read_bytes()
        ),
    )
    rotated = CliRunner().invoke(
        app,
        [
            "optimize",
            "claas",
            "bind",
            "claims",
            "--alias",
            "replacement",
            "--execution",
            str(execution),
            "--root",
            str(tmp_path),
        ],
    )
    assert rotated.exit_code == 0, rotated.output
    selected = serving_binding(tmp_path, scope)
    assert selected.alias == "replacement"
    assert json.loads(selected.state_path.read_text())["paused"]
    current = load_gateway_serving_configuration(tmp_path)
    assert current is not None and current.bindings == (selected,)
    save_execution_settings(
        application_directory(tmp_path, scope),
        settings().model_copy(update={"private_base_url": "http://127.0.0.1:9000"}),
    )
    with pytest.raises(ValueError, match="runtime and gateway binding differ"):
        serving_binding(tmp_path, scope)


def test_bind_rejects_other_model_before_writing_readiness(tmp_path: Path) -> None:
    """A granted alias cannot be repurposed as an unrelated student's serving path."""
    manager, _ = _configure_gateway(tmp_path, base_url="http://127.0.0.1:8000/v1")
    manager.migrate_legacy_provider_connections()
    alias = manager.aliases()[0]
    assert alias.snapshot_ref and alias.revision_id
    catalog_path = manager.state_dir / alias.snapshot_ref
    manager.ensure_alias_provider_bindings(
        alias_id=alias.alias_id,
        alias_revision_id=alias.revision_id,
        catalog=ModelCatalog.model_validate_json(
            catalog_path.with_suffix(".models.json").read_bytes()
        ),
    )
    scope = ClaasScope(user_id="default", application_id="claims")
    save_configuration(tmp_path, config().model_copy(update={"scope": scope}))
    execution = tmp_path / "execution-input.json"
    execution.write_text(settings().model_dump_json())
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "claas",
            "bind",
            "claims",
            "--alias",
            "coding",
            "--execution",
            str(execution),
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "provider model differs" in result.output
    assert load_gateway_serving_configuration(tmp_path) is None


@pytest.mark.parametrize("provider_revision", [None, "a" * 40, "b" * 40])
def test_bind_hugging_face_provider_identity_is_distinct_from_logical_slug(
    tmp_path: Path, provider_revision: str | None
) -> None:
    """A real Qwen model path binds through a slug, but a conflicting weight revision fails."""
    manager, _ = _configure_gateway(tmp_path, base_url="http://127.0.0.1:8000/v1")
    manager.migrate_legacy_provider_connections()
    normalized, snapshot, _ = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="Qwen/Qwen3.5-4B",
        exact_model_id="qwen3.5-4b",
        revision=provider_revision,
        capabilities=ModelCapabilities(supports_tools=True),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_streaming_tool_arguments=True
        ),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=0, output_nano_usd_per_million_tokens=0
        ),
        pricing_source="explicit private fixture",
        replace=True,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="qwen-revision",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.ensure_alias_provider_bindings(
        alias_id="coding",
        alias_revision_id="qwen-revision",
        catalog=ModelCatalog.model_validate_json(snapshot.with_suffix(".models.json").read_bytes()),
    )
    scope = ClaasScope(user_id="default", application_id="claims")
    configured = config().model_copy(
        update={"scope": scope, "base_model": "Qwen/Qwen3.5-4B", "base_model_revision": "a" * 40}
    )
    save_configuration(tmp_path, configured)
    execution = tmp_path / "execution-input.json"
    execution.write_text(settings().model_dump_json())
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "claas",
            "bind",
            "claims",
            "--alias",
            "coding",
            "--execution",
            str(execution),
            "--root",
            str(tmp_path),
        ],
    )
    if provider_revision == "b" * 40:
        assert result.exit_code != 0
        assert "deployment revision differs" in result.output
        assert load_gateway_serving_configuration(tmp_path) is None
    else:
        assert result.exit_code == 0, result.output
        command = json.loads(result.output)["vllm_command"]
        assert command[2] == "Qwen/Qwen3.5-4B"
        assert "--enable-auto-tool-choice" in command
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
        assert command[command.index("--reasoning-parser") + 1] == "qwen3"
