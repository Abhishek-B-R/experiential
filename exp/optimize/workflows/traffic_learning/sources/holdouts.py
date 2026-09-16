"""Persist explicit evaluation response exclusions before captured traffic reaches training."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from exp.common.claas.contracts import Identifier
from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic


class PendingEvaluationRequest(ContractModel):
    """A dispatched held-out request whose durable captured identity is not yet acknowledged."""

    run_id: Identifier
    request_id: Identifier


class EvaluationHoldouts(ContractModel):
    """Content-free response identities permanently reserved for external held-out evaluation."""

    schema_version: Literal[1] = 1
    response_ids: tuple[Identifier, ...] = Field(default=(), max_length=1_000_000)
    pending: PendingEvaluationRequest | None = None

    @field_validator("response_ids")
    @classmethod
    def _unique_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate or noncanonical persisted exclusions instead of repairing them."""
        if value != tuple(sorted(set(value))):
            raise ValueError("evaluation holdout response_ids must be unique and sorted")
        return value


class _PartitionLedger(ContractModel):
    """Read the authoritative split ledger without importing cycle orchestration."""

    seed: str
    assignments: dict[str, Literal["fit", "held_out"]]


def _read_holdouts(directory: Path) -> EvaluationHoldouts:
    """Read the strict owned ledger, including any unresolved request after process restart."""
    path = directory / "evaluation-holdouts.json"
    if path.is_symlink():
        raise ValueError("evaluation holdouts must be a regular owned file")
    if not path.exists():
        return EvaluationHoldouts()
    if path.stat().st_size > 64_000_000:
        raise ValueError("evaluation holdout ledger exceeds its 64 MB bound")
    return EvaluationHoldouts.model_validate_json(path.read_bytes())


def _require_no_pending(state: EvaluationHoldouts) -> None:
    """Block all source use until an outstanding evaluation capture is conclusively identified."""
    if state.pending is not None:
        raise ValueError(
            "unresolved held-out evaluation request blocks learning and further evaluation calls; "
            "preserve evaluation-holdouts.json and the capture database for reconciliation, "
            "or initialize a new application with a fresh alias; do not delete the pending marker"
        )


def load_evaluation_holdouts(directory: Path) -> frozenset[str]:
    """Return exclusions only when every prior held-out dispatch has a durable response identity."""
    state = _read_holdouts(directory)
    _require_no_pending(state)
    return frozenset(state.response_ids)


def _write_holdouts(directory: Path, state: EvaluationHoldouts) -> None:
    """Atomically replace the single pending/exclusion state before permitting further work."""
    payload = state.model_dump_json() + "\n"
    if len(payload.encode()) > 64_000_000:
        raise ValueError("evaluation holdout ledger exceeds its 64 MB bound")
    write_text_atomic(directory / "evaluation-holdouts.json", payload)


def _validate_not_fit(directory: Path, selected: tuple[str, ...]) -> None:
    """Reject retroactive held-out labels for traffic already exposed to training."""
    path = directory / "partitions.json"
    if path.is_symlink():
        raise ValueError("partition ledger must be a regular owned file")
    if path.exists():
        if path.stat().st_size > 64_000_000:
            raise ValueError("partition ledger exceeds its 64 MB bound")
        assignments = _PartitionLedger.model_validate_json(path.read_bytes()).assignments
        if any(assignments.get(identity) == "fit" for identity in selected):
            raise ValueError(
                "evaluation response was already assigned to fit; use fresh held-out tasks"
            )


def reserve_evaluation_holdout(directory: Path, response_ids: tuple[str, ...]) -> None:
    """Reserve known responses while the caller holds the application cycle lock.

    Every linked source group touching these identities must be excluded before
    source partitioning. This helper cannot clear an unresolved dispatched request.
    """
    selected = EvaluationHoldouts(response_ids=tuple(sorted(set(response_ids)))).response_ids
    previous = _read_holdouts(directory)
    _require_no_pending(previous)
    _validate_not_fit(directory, selected)
    state = EvaluationHoldouts(
        response_ids=tuple(sorted(set(previous.response_ids).union(selected)))
    )
    _write_holdouts(directory, state)


def begin_evaluation_holdout(directory: Path, *, run_id: str, request_id: str) -> None:
    """Durably block learning before dispatch while the caller holds the application cycle lock.

    A crash or unknown HTTP result leaves this marker intact. No timeout or next
    run can implicitly clear it because the gateway may already have committed
    capture before the client received its response ID.
    """
    previous = _read_holdouts(directory)
    _require_no_pending(previous)
    pending = PendingEvaluationRequest(run_id=run_id, request_id=request_id)
    _write_holdouts(
        directory, EvaluationHoldouts(response_ids=previous.response_ids, pending=pending)
    )


def acknowledge_evaluation_holdout(
    directory: Path, *, run_id: str, request_id: str, response_id: str
) -> None:
    """Reserve the exact acknowledged capture and clear its matching pending marker atomically.

    The caller owns the cycle lock and supplies only the response ID received from
    that same dispatched request. A write failure preserves the old blocking state;
    a crash after replacement leaves the response excluded without a blocking gap.
    """
    previous = _read_holdouts(directory)
    pending = PendingEvaluationRequest(run_id=run_id, request_id=request_id)
    if previous.pending != pending:
        raise ValueError("evaluation acknowledgment differs from the pending run or request")
    selected = EvaluationHoldouts(response_ids=(response_id,)).response_ids
    _validate_not_fit(directory, selected)
    state = EvaluationHoldouts(
        response_ids=tuple(sorted(set(previous.response_ids).union(selected)))
    )
    _write_holdouts(directory, state)
