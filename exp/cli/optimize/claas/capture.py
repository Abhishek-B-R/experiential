"""Opt a configured application's authenticated gateway traffic into local learning."""

from pathlib import Path

import typer

from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.claas import CapturePolicy, ClaasScope
from exp.common.core.locks import FileLockTimeout
from exp.optimize.claas.configuration import load_configuration
from exp.runtime.claas.capture import CaptureBinding, save_capture_binding
from exp.runtime.gateway.management import GatewayManagement


def configure_capture(
    application: str = typer.Argument(help="Configured local application."),
    alias: str = typer.Option(..., "--alias", help="Existing gateway alias used by this agent."),
    user: str = typer.Option("default", "--user", help="Authenticated gateway identity ID."),
    disable: bool = typer.Option(False, "--disable", help="Disable capture for this application."),
    maximum_experiences: int = typer.Option(10_000, "--maximum-experiences", min=1, max=1_000_000),
    retention_seconds: int = typer.Option(604_800, "--retention-seconds", min=1),
    root: Path = ROOT_OPTION,
) -> None:
    """Persist explicit content-capture consent; restart the gateway to apply it.

    Args:
        application: Application whose configuration and scope are already fixed.
        alias: Existing granted gateway alias whose HTTP traffic provides source evidence.
        user: Exact identity derived from the agent's authenticated gateway key.
        disable: Disable capture on every alias belonging to this application.
        maximum_experiences: Maximum retained exchanges for this scope.
        retention_seconds: Maximum source retention period.
        root: Experiential artifact root.
    """
    with usage_error(ValueError, FileLockTimeout):
        scope = ClaasScope(user_id=user, application_id=application)
        load_configuration(root, scope)
        manager = GatewayManagement(root)
        if not any(item.identity_id == user and item.active for item in manager.identities()):
            raise ValueError("--user must name an active authenticated gateway identity")
        if not disable and not any(
            item.alias_name == alias for item in manager.grants(identity_id=user)
        ):
            raise ValueError("this gateway identity has no grant for the requested alias")
        config = save_capture_binding(
            root,
            CaptureBinding(
                alias=alias,
                policy=CapturePolicy(
                    scope=scope,
                    enabled=not disable,
                    maximum_experiences=maximum_experiences,
                    retention_seconds=retention_seconds,
                ),
            ),
        )
    state = "disabled" if disable else "enabled"
    typer.echo(
        f"Content capture {state} for {application!r}; restart the gateway to apply. "
        f"Local source store: {config.database_path}"
    )
