"""Traffic workflow selection stays independent from generic learner configuration."""

from pathlib import Path

import pytest

from exp.common.claas import ClaasScope
from exp.optimize.claas.configuration import LocalClaasConfig, save_configuration
from exp.optimize.workflows.traffic_learning.configuration import (
    TrafficWorkflowConfig,
    load_workflow,
    save_workflow,
)


def test_generic_application_needs_no_traffic_workflow(tmp_path: Path) -> None:
    """An application can persist its learner settings without any model harness aliases."""
    scope = ClaasScope(user_id="local", application_id="tools")
    learner = LocalClaasConfig(
        scope=scope,
        base_model="student",
        base_model_revision="a" * 40,
        tokenizer_id="student",
        tokenizer_revision="a" * 40,
    )
    path = save_configuration(tmp_path, learner)
    assert path.exists()
    with pytest.raises(ValueError, match="traffic learning is not configured"):
        load_workflow(tmp_path, scope)
    workflow = TrafficWorkflowConfig(scope=scope, world_model_alias="world", judge_alias="judge")
    workflow_path = save_workflow(tmp_path, workflow)
    assert workflow_path != path
    assert load_workflow(tmp_path, scope) == workflow
    assert LocalClaasConfig.model_validate_json(path.read_bytes()) == learner


def test_workflow_replacement_is_explicit_and_scoped(tmp_path: Path) -> None:
    """Provider choices cannot change silently or be read under another application."""
    scope = ClaasScope(user_id="local", application_id="tools")
    original = TrafficWorkflowConfig(scope=scope, world_model_alias="world", judge_alias="judge")
    path = save_workflow(tmp_path, original)
    assert save_workflow(tmp_path, original) == path
    changed = original.model_copy(update={"world_model_alias": "another-world"})
    with pytest.raises(ValueError, match="--replace"):
        save_workflow(tmp_path, changed)
    save_workflow(tmp_path, changed, replace=True)
    assert load_workflow(tmp_path, scope) == changed
    path.write_text(
        changed.model_copy(
            update={"scope": ClaasScope(user_id="other", application_id="tools")}
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="another application"):
        load_workflow(tmp_path, scope)
