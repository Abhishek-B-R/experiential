"""Inject arbitrary prepared scenarios or a bounded scenario supplier into CLaaS."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from exp.common.claas.scenarios import Scenario
from exp.common.core.artifacts import JsonValue
from exp.optimize.claas.configuration import LocalClaasConfig
from exp.optimize.claas.evaluation.paired import EvaluationManifest, TaskEvaluator
from exp.runtime.environments.learning import Environment


@dataclass(frozen=True)
class PreparedCycle:
    """Explicit learning inputs, with no required traffic source or model provider."""

    scenarios: tuple[Scenario, ...]
    environment: Environment
    evaluation: EvaluationManifest
    evaluator: TaskEvaluator
    external_reservation_usd: float = 0.0
    evidence: Mapping[str, JsonValue] = field(default_factory=dict)


class CycleSource(Protocol):
    """Optional scenario preparation strategy executed under the application cycle lock."""

    @property
    def external_reservation_usd(self) -> float:
        """Declare the complete bounded preparation, practice, and evaluation reservation."""
        ...

    async def prepare(self, directory: Path, config: LocalClaasConfig) -> PreparedCycle:
        """Freeze inputs while the caller exclusively owns source and lifecycle state."""
        ...
