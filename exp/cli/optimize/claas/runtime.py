"""Bind and start the private student inference path used by local CLaaS."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import httpx
import typer
from rich.console import Console

from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.claas import ClaasScope
from exp.common.core.locks import FileLockTimeout, file_write_lock
from exp.common.models.gateway_catalog import read_pinned_normalized_snapshot
from exp.optimize.claas.configuration import application_directory, load_configuration
from exp.optimize.claas.execution import (
    ExecutionSettings,
    load_execution_settings,
    save_execution_settings,
)
from exp.optimize.claas.lifecycle.cycle import base_revision, checkpoint_for_revision, training_spec
from exp.optimize.workflows.traffic_learning.execution import (
    TrafficProviderSettings,
    save_provider_settings,
)
from exp.runtime.claas.registry import AdapterRegistry
from exp.runtime.claas.serving.configuration import VllmServerConfig
from exp.runtime.claas.serving.decoding import HermesCompletionDecoder, Qwen35CompletionDecoder
from exp.runtime.claas.serving.lifecycle import VllmServingLifecycle
from exp.runtime.gateway.claas.serving import (
    GatewayAdmissionLease,
    GatewayServingBinding,
    load_gateway_serving_configuration,
    save_gateway_serving_binding,
)
from exp.runtime.gateway.management import GatewayManagement

_console = Console()


class RuntimeBindingInput(ExecutionSettings):
    """CLI composition of a generic runtime and optional traffic-provider reservations."""

    traffic: TrafficProviderSettings | None = None


def bind_runtime(
    application: str = typer.Argument(help="Configured CLaaS application."),
    alias: str = typer.Option(..., "--alias", help="Granted direct gateway alias for the student."),
    execution: Annotated[
        Path,
        typer.Option(
            "--execution",
            exists=True,
            dir_okay=False,
            help="Secret-free ExecutionSettings JSON file.",
        ),
    ] = Path("execution.json"),
    user: str = typer.Option("default", "--user"),
    root: Path = ROOT_OPTION,
) -> None:
    """Bind a private student runtime; the gateway starts paused until activation succeeds.

    Args:
        application: Previously configured local learning application.
        alias: Granted direct alias pointing at the private vLLM origin.
        execution: Explicit worker environment, decoder, and conservative call-cost settings.
        user: Authenticated gateway identity that owns this adapter.
        root: Experiential artifact root.
    """
    with usage_error(ValueError, FileLockTimeout, OSError, RuntimeError):
        scope = ClaasScope(user_id=user, application_id=application)
        config = load_configuration(root, scope)
        selected = RuntimeBindingInput.model_validate_json(execution.read_bytes())
        settings = ExecutionSettings.model_validate(selected.model_dump(exclude={"traffic"}))
        manager = GatewayManagement(root)
        if not any(item.identity_id == user and item.active for item in manager.identities()):
            raise ValueError("--user must name an active authenticated gateway identity")
        if not any(item.alias_name == alias for item in manager.grants(identity_id=user)):
            raise ValueError("this gateway identity has no grant for the requested alias")
        validate_alias(manager, alias, settings, config.base_model, config.base_model_revision)
        directory = application_directory(root, scope).resolve()
        with file_write_lock(directory / "cycle", what="CLaaS runtime binding"):
            registry = AdapterRegistry(directory / "registry.json", scope)
            registry.initialize(base_revision(config))
            binding = GatewayServingBinding(
                scope=scope,
                alias=alias,
                registry_path=registry.path,
                private_base_url=settings.private_base_url,
                state_path=directory / "serving.json",
                admission_lock_path=directory / "admission.lock",
            )
            save_gateway_serving_binding(root, binding)
            save_execution_settings(directory, settings)
            if selected.traffic is not None:
                save_provider_settings(directory, selected.traffic)
        server = VllmServerConfig(
            base=base_revision(config),
            port=urlsplit(settings.private_base_url).port or 80,
            max_lora_rank=config.lora_rank,
            decoder=settings.decoder,
        )
    _console.print(
        json.dumps(
            {
                "vllm_command": server.command(),
                "vllm_environment": {
                    **server.environment(),
                    "CUDA_VISIBLE_DEVICES": settings.cuda_visible_device,
                },
                "next": (
                    "Start private vLLM, run exp optimize claas activate, then restart the gateway."
                ),
            },
            indent=2,
        ),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def serving_binding(root: Path, scope: ClaasScope) -> GatewayServingBinding:
    """Require one exact application binding rather than inferring an unrelated alias."""
    configuration = load_gateway_serving_configuration(root)
    if configuration is None:
        raise ValueError("student serving is not bound; run exp optimize claas bind first")
    selected = [item for item in configuration.bindings if item.scope == scope]
    if len(selected) != 1:
        raise ValueError("application needs exactly one student serving binding; run claas bind")
    binding = selected[0]
    directory = application_directory(root, scope).resolve()
    settings = load_execution_settings(directory)
    if (
        binding.private_base_url.rstrip("/") != settings.private_base_url.rstrip("/")
        or binding.registry_path != directory / "registry.json"
        or binding.state_path != directory / "serving.json"
        or binding.admission_lock_path != directory / "admission.lock"
    ):
        raise ValueError("runtime and gateway binding differ; run claas bind before activation")
    return binding


def make_serving(
    client: httpx.AsyncClient, settings: ExecutionSettings, config_root: Path, scope: ClaasScope
) -> VllmServingLifecycle:
    """Construct a paused runtime without performing inference or control calls."""
    config = load_configuration(config_root, scope)
    decoder = (
        Qwen35CompletionDecoder() if settings.decoder == "qwen35" else HermesCompletionDecoder()
    )
    return VllmServingLifecycle(
        client=client,
        base=base_revision(config),
        decoder=decoder,
        max_tokens=config.limits.maximum_response_tokens,
    )


def activate(
    application: str = typer.Argument(help="Configured local application."),
    user: str = typer.Option("default", "--user"),
    root: Path = ROOT_OPTION,
) -> None:
    """Load the active verified revision and open its already configured gateway binding.

    Args:
        application: Application whose current registry revision should serve traffic.
        user: Authenticated application owner.
        root: Experiential artifact root.
    """
    with usage_error(ValueError, FileLockTimeout, OSError, RuntimeError, httpx.HTTPError):
        scope = ClaasScope(user_id=user, application_id=application)
        config = load_configuration(root, scope)
        directory = application_directory(root, scope).resolve()
        settings = load_execution_settings(directory)
        binding = serving_binding(root, scope)

        async def run() -> None:
            """Hold the application lock through verified load and readiness publication."""
            with file_write_lock(directory / "cycle", what="CLaaS activation"):
                registry = AdapterRegistry(directory / "registry.json", scope)
                state = registry.read()
                checkpoint_for_revision(directory, state.active, training_spec(config))
                lease = GatewayAdmissionLease(binding)
                lease.initialize()
                async with httpx.AsyncClient(
                    base_url=settings.private_base_url, timeout=120
                ) as client:
                    serving = make_serving(client, settings, root, scope)
                    try:
                        await lease.pause_and_drain()
                        await serving.pause_and_drain()
                        await serving.wake()
                        await serving.load_revision(state.active)
                        await serving.resume()
                        await lease.resume(
                            expected_registry_generation=state.generation,
                            expected_policy_revision=state.active.policy_revision,
                        )
                    finally:
                        await lease.close()

        asyncio.run(run())
    _console.print(f"Active adapter for {application!r} is loaded and ready.", markup=False)


def validate_alias(
    manager: GatewayManagement,
    alias: str,
    settings: ExecutionSettings,
    base_model: str,
    base_revision: str,
) -> None:
    """Verify one pinned direct route reaches the configured private student origin."""
    selected = [item for item in manager.aliases() if item.alias_name == alias and item.active]
    if len(selected) != 1:
        raise ValueError("student alias is not active; configure a direct gateway alias first")
    active = selected[0]
    if active.target_kind != "direct" or not active.snapshot_ref or not active.catalog_sha256:
        raise ValueError("student alias must select one direct private vLLM deployment")
    path = (manager.state_dir / active.snapshot_ref).resolve()
    if not path.is_relative_to(manager.state_dir.resolve()):
        raise ValueError("student alias catalog escapes gateway state")
    catalog = read_pinned_normalized_snapshot(path.read_bytes(), active.catalog_sha256)
    pool = next((item for item in catalog.pools if item.pool_id == active.pool_id), None)
    if pool is None or len(pool.deployment_ids) != 1:
        raise ValueError("student alias must have one deployment for the configured base model")
    deployment = next(
        item for item in catalog.deployments if item.deployment_id == pool.deployment_ids[0]
    )
    if deployment.provider_model != base_model:
        raise ValueError("student deployment provider model differs from the configured base model")
    if deployment.revision is not None and deployment.revision != base_revision:
        raise ValueError("student deployment revision differs from the configured base revision")
    connections = manager.alias_provider_connections(
        alias_id=active.alias_id, alias_revision_id=active.revision_id or ""
    )
    connection = next(
        (item.config for item in connections if item.connection_id == deployment.connection), None
    )
    if (
        connection is None
        or connection.provider != "openai-compatible"
        or (connection.base_url or "").rstrip("/").removesuffix("/v1")
        != settings.private_base_url.rstrip("/")
    ):
        raise ValueError(
            "student alias must point to the configured private OpenAI-compatible origin"
        )
