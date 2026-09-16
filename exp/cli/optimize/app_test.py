"""Deferred CLaaS manifests expose every implemented command in help and completion."""

from typer import _click
from typer.main import get_group
from typer.testing import CliRunner

from exp.cli.optimize.app import optimize_app
from exp.cli.optimize.claas.app import claas_app
from exp.cli.shared.defer import DeferredTyperGroup


def test_deferred_claas_manifest_covers_loaded_commands() -> None:
    """Adding a loaded command without updating deferred discovery is a regression."""
    parent = get_group(optimize_app)
    context = _click.Context(parent)
    deferred = parent.get_command(context, "claas")
    assert isinstance(deferred, DeferredTyperGroup)
    loaded = get_group(claas_app)
    expected = {"activate", "bind", "capture", "init", "rollback", "status", "train"}
    assert set(loaded.list_commands(_click.Context(loaded))) == expected
    assert set(deferred.list_commands(_click.Context(deferred))) == expected
    completions = deferred.shell_complete(_click.Context(deferred), "")
    assert {item.value for item in completions} == expected


def test_claas_help_exposes_supported_runtime_workflow() -> None:
    """The actual parent CLI help lists capture, bind, activate, train, and rollback."""
    result = CliRunner().invoke(optimize_app, ["claas", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("init", "status", "capture", "bind", "activate", "train", "rollback"):
        assert command in result.output
