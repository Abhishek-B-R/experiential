"""Configure and inspect one local continually learned application."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from exp.cli.optimize.claas.capture import configure_capture
from exp.cli.optimize.claas.learning import rollback_adapter, train
from exp.cli.optimize.claas.runtime import activate, bind_runtime
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.claas.contracts import ClaasScope
from exp.common.core.locks import FileLockTimeout, file_write_lock
from exp.optimize.claas.configuration import (
    CycleLimits,
    LocalClaasConfig,
    application_directory,
    load_configuration,
    save_configuration,
)
from exp.optimize.workflows.traffic_learning.configuration import (
    TrafficWorkflowConfig,
    load_workflow,
    save_workflow,
)


class Objective(StrEnum):
    """Explicit feedback objective offered by the local application setup command."""

    SDPO = "sdpo"
    REINFORCE = "reinforce"
    HYBRID = "hybrid"


claas_app = typer.Typer(
    help="Learn a local application's LoRA from experience.", no_args_is_help=True
)
_console = Console()
claas_app.command("capture")(configure_capture)
claas_app.command("bind")(bind_runtime)
claas_app.command("activate")(activate)
claas_app.command("train")(train)
claas_app.command("rollback")(rollback_adapter)


@claas_app.command("init")
def initialize(
    application: str = typer.Argument(help="Local application or agent name."),
    base_model: str = typer.Option(..., "--base-model", help="Trainable Hugging Face model ID."),
    revision: str = typer.Option(..., "--revision", help="Exact base-model revision."),
    world_model: str | None = typer.Option(
        None,
        "--world-model",
        help="Opt into traffic synthesis with this authorized world-model alias.",
    ),
    judge: str | None = typer.Option(
        None,
        "--judge",
        help="Judge for the optional traffic-learning workflow; requires --world-model.",
    ),
    tokenizer: str | None = typer.Option(None, "--tokenizer", help="Defaults to the base model."),
    tokenizer_revision: str | None = typer.Option(
        None, "--tokenizer-revision", help="Defaults to the base-model revision."
    ),
    user: str = typer.Option("default", "--user", help="Local user who owns this adapter."),
    modal: bool = typer.Option(False, "--modal", help="Use the optional Modal compute adapter."),
    objective: Annotated[Objective, typer.Option("--objective")] = Objective.SDPO,
    lora_rank: int = typer.Option(16, "--lora-rank", min=1, max=256),
    interval: int = typer.Option(3600, "--interval-seconds", min=60, max=2_592_000),
    maximum_cost: float = typer.Option(5.0, "--maximum-cycle-cost", min=0.01),
    replace: bool = typer.Option(False, "--replace", help="Explicitly update existing settings."),
    root: Path = ROOT_OPTION,
) -> None:
    """Persist model choices and finite cycle limits without launching resources.

    Args:
        application: Local agent or application identity.
        base_model: Trainable base model ID.
        revision: Immutable base-model revision.
        world_model: Configured provider alias used by the harness.
        judge: Configured provider alias used to judge evaluation episodes.
        tokenizer: Optional tokenizer ID distinct from the base model.
        tokenizer_revision: Exact tokenizer revision when it differs from the base.
        user: Local owner of the application's experience and adapter state.
        modal: Whether cycles should execute on Modal GPUs.
        objective: Text SDPO, scalar REINFORCE, or both feedback objectives.
        lora_rank: Rank of this application's low-rank adapter.
        interval: Minimum period between scheduled training cycles.
        maximum_cost: Finite spending ceiling for one cycle.
        replace: Explicitly permit updating an existing application's settings.
        root: Experiential artifact root.
    """
    with usage_error(ValueError, FileLockTimeout):
        if (world_model is None) != (judge is None):
            raise ValueError("traffic learning requires both --world-model and --judge")
        config = LocalClaasConfig(
            scope=ClaasScope(user_id=user, application_id=application),
            base_model=base_model,
            base_model_revision=revision,
            tokenizer_id=tokenizer or base_model,
            tokenizer_revision=tokenizer_revision or revision,
            compute="modal" if modal else "local",
            objective=objective.value,
            lora_rank=lora_rank,
            interval_seconds=interval,
            limits=CycleLimits(maximum_cost_usd=maximum_cost),
        )
        workflow = (
            TrafficWorkflowConfig(
                scope=config.scope, world_model_alias=world_model, judge_alias=judge
            )
            if world_model is not None and judge is not None
            else None
        )
        directory = application_directory(root, config.scope)
        with file_write_lock(directory / "cycle", what="CLaaS application setup"):
            if workflow is not None and (directory / "traffic-workflow.json").exists():
                if load_workflow(root, config.scope) != workflow and not replace:
                    raise ValueError(
                        "traffic workflow already exists; use --replace to edit settings"
                    )
            path = save_configuration(root, config, replace=replace)
            if workflow is not None:
                save_workflow(root, workflow, replace=replace)
    _console.print(f"Configured CLaaS application {application!r} at {path}", markup=False)


@claas_app.command("status")
def status(
    application: str = typer.Argument(help="Local application or agent name."),
    user: str = typer.Option("default", "--user", help="Local adapter owner."),
    root: Path = ROOT_OPTION,
) -> None:
    """Show persisted application settings without inference or training calls.

    Args:
        application: Local application to inspect.
        user: Local owner whose configuration is selected.
        root: Experiential artifact root.
    """
    with usage_error(ValueError):
        config = load_configuration(root, ClaasScope(user_id=user, application_id=application))
    _console.print(config.model_dump_json(indent=2), markup=False, highlight=False, soft_wrap=True)
