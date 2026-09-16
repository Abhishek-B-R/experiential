"""Bounded local learning and explicit rollback commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import typer
from rich.console import Console

from exp.cli.optimize.claas.runtime import make_serving, serving_binding
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.claas import ClaasScope
from exp.common.config import resolve_command_budget_usd
from exp.common.core.locks import FileLockTimeout, file_write_lock
from exp.common.models.catalog import load_model_catalog
from exp.optimize.claas.configuration import application_directory, load_configuration
from exp.optimize.claas.execution import (
    create_training_backend,
    load_execution_settings,
    validate_worker_runtime,
)
from exp.optimize.claas.lifecycle.cycle import rollback, run_cycle, training_spec
from exp.optimize.workflows.traffic_learning.configuration import load_workflow
from exp.optimize.workflows.traffic_learning.execution import (
    ConfiguredTrafficSource,
    load_provider_settings,
    provider_plan,
)
from exp.optimize.workflows.traffic_learning.sources.buffer import refresh_buffer
from exp.optimize.workflows.traffic_learning.sources.preparation import prepare_evidence
from exp.optimize.workflows.traffic_learning.sources.retention import prune_evidence
from exp.runtime.claas.capture import load_capture_configuration
from exp.runtime.gateway.claas.serving import GatewayAdmissionLease
from exp.runtime.models.registry import RuntimeModelCatalog

_console = Console()


def train(
    application: str = typer.Argument(help="Configured local application."),
    cycles: int = typer.Option(
        1, "--cycles", min=1, max=10_000, help="Finite wake/sleep cycles; default runs once."
    ),
    user: str = typer.Option("default", "--user"),
    yes: bool = typer.Option(False, "--yes", help="Confirm an in-budget command estimate."),
    root: Path = ROOT_OPTION,
) -> None:
    """Learn from retained traffic, test each candidate, and publish measured improvements.

    Args:
        application: Configured local agent application.
        cycles: Finite number of updates, separated by the configured interval.
        user: Authenticated owner of traffic and adapter state.
        yes: Explicit confirmation within the configured per-command spending ceiling.
        root: Experiential artifact root.
    """
    with usage_error(ValueError, FileLockTimeout, OSError, RuntimeError, httpx.HTTPError):
        scope = ClaasScope(user_id=user, application_id=application)
        config = load_configuration(root, scope)
        training_spec(config)
        directory = application_directory(root, scope).resolve()
        settings = load_execution_settings(directory)
        validate_worker_runtime(settings, config)
        workflow = load_workflow(root, scope)
        provider_settings = load_provider_settings(directory)
        binding = serving_binding(root, scope)
        capture = load_capture_configuration(root)
        if capture is None or not any(
            item.policy.scope == scope and item.policy.enabled for item in capture.bindings
        ):
            raise ValueError("traffic capture is not enabled; run exp optimize claas capture first")
        catalog = RuntimeModelCatalog(load_model_catalog(root / "models.toml"))
        world, _ = catalog.snapshot(workflow.world_model_alias)
        judge, _ = catalog.snapshot(workflow.judge_alias)
        with file_write_lock(directory / "cycle", what="CLaaS learning preflight"):
            buffer = refresh_buffer(
                directory, capture.database_path, scope, workflow.maximum_source_experiences
            )
            prune_evidence(
                directory, retained_source_ids={item.experience_id for item in buffer.experiences}
            )
            _, manifest = prepare_evidence(
                directory=directory,
                experiences=buffer.experiences,
                world_model=world,
                judge_model=judge,
                minimum_tasks=config.promotion.minimum_evaluation_tasks,
            )
        provider_plan(
            config, provider_settings, settings, workflow, evaluation_tasks=len(manifest.tasks)
        )
        estimate = cycles * config.limits.maximum_cost_usd
        if estimate > resolve_command_budget_usd(root, None):
            raise ValueError(
                "learning estimate exceeds the hard per-command budget; "
                "reduce cycles or change budget settings"
            )
        # The complete per-cycle cap also reserves optional compute, before SDK loading.
        if not require_spend_consent(
            _console,
            root=root,
            yes=yes,
            estimated_cost_usd=cycles * config.limits.maximum_cost_usd,
            command=f"exp optimize claas train {application} --cycles {cycles}",
            non_interactive=False,
        ):
            return

        async def run() -> None:
            """Run finite cycles with fresh reservations and independent training lineages."""
            async with httpx.AsyncClient(base_url=settings.private_base_url, timeout=120) as client:
                serving = make_serving(client, settings, root, scope)
                for index in range(cycles):
                    lease = GatewayAdmissionLease(binding)
                    lease.initialize()
                    result = await run_cycle(
                        directory=directory,
                        config=config,
                        plan=ConfiguredTrafficSource(
                            config=config,
                            workflow=workflow,
                            runtime=settings,
                            settings=provider_settings,
                            catalog=catalog,
                            database_path=capture.database_path,
                        ),
                        serving=serving,
                        admission=lease,
                        compute_reservation_usd=(
                            settings.modal.estimated_maximum_cost_usd if settings.modal else 0.0
                        ),
                        backend_factory=lambda lineage: create_training_backend(
                            settings, config, directory, lineage
                        ),
                    )
                    _console.print(
                        result.model_dump_json(indent=2),
                        markup=False,
                        highlight=False,
                        soft_wrap=True,
                    )
                    if index + 1 < cycles:
                        await asyncio.sleep(config.interval_seconds)

        asyncio.run(run())


def rollback_adapter(
    application: str = typer.Argument(
        help="Application whose previous adapter should be restored."
    ),
    user: str = typer.Option("default", "--user"),
    root: Path = ROOT_OPTION,
) -> None:
    """Restore the previous verified adapter while pausing and draining public requests.

    Args:
        application: Application with an existing rollback revision.
        user: Authenticated adapter owner.
        root: Experiential artifact root.
    """
    with usage_error(ValueError, FileLockTimeout, OSError, RuntimeError, httpx.HTTPError):
        scope = ClaasScope(user_id=user, application_id=application)
        config = load_configuration(root, scope)
        directory = application_directory(root, scope).resolve()
        settings = load_execution_settings(directory)
        binding = serving_binding(root, scope)

        async def run() -> None:
            """Load and publish the rollback target on the configured private server."""
            async with httpx.AsyncClient(base_url=settings.private_base_url, timeout=120) as client:
                state = await rollback(
                    directory=directory,
                    config=config,
                    serving=make_serving(client, settings, root, scope),
                    admission=GatewayAdmissionLease(binding),
                )
                _console.print(
                    state.model_dump_json(indent=2), markup=False, highlight=False, soft_wrap=True
                )

        asyncio.run(run())
