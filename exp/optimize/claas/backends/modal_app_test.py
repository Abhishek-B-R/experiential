"""Construct the real Modal definition without deploying or allocating compute."""

import subprocess
import sys

import modal

from exp.optimize.claas.backends.modal_app import create_modal_app
from exp.optimize.claas.backends.modal_test import config


def test_app_definition_requires_no_cloud_or_gpu_runtime() -> None:
    """A real SDK app can be built locally with deferred image and Volume handles."""
    app = create_modal_app(config=config(), image=modal.Image.debian_slim())
    assert isinstance(app, modal.App)
    assert app.name == "fixture"


def test_optional_adapter_does_not_import_training_stack() -> None:
    """Importing the explicit execution adapter does not eagerly load model libraries."""
    source = (
        "import sys; import exp.optimize.claas.backends.modal_app; "
        "assert not {'torch', 'ray', 'verl', 'peft'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
