"""SDK-free configuration and finite cost authorization for optional Modal training."""

from __future__ import annotations

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel


class ModalExecutionConfig(ContractModel):
    """An explicitly selected deployment and finite, caller-authorized cost ceiling.

    The rate must conservatively include GPU, CPU, and memory billing. It is an
    input supplied by the caller's pricing/consent layer, never a live price claim.
    One backend session allows one job; open another with a fresh authorization
    for another update. Retries require inspecting durable completion first.
    """

    app_name: str = Field(min_length=1)
    function_name: str = Field(default="train_claas", min_length=1)
    volume_name: str = Field(min_length=1)
    environment_name: str | None = None
    gpu: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9-]*$")
    timeout_seconds: int = Field(default=1800, strict=True, ge=1, le=86400)
    startup_timeout_seconds: int = Field(default=600, strict=True, ge=1, le=3600)
    maximum_container_rate_usd_per_second: float = Field(gt=0, allow_inf_nan=False, strict=True)
    authorized_maximum_cost_usd: float = Field(gt=0, allow_inf_nan=False, strict=True)
    maximum_checkpoint_bytes: int = Field(default=4_294_967_296, strict=True, ge=1)

    @property
    def estimated_maximum_cost_usd(self) -> float:
        """Conservatively reserve startup plus execution at the caller's ceiling rate."""
        return (
            self.timeout_seconds + self.startup_timeout_seconds
        ) * self.maximum_container_rate_usd_per_second

    @model_validator(mode="after")
    def _require_cost_authorization(self) -> ModalExecutionConfig:
        """Reject an underfunded job before any SDK lookup or paid request."""
        if self.estimated_maximum_cost_usd > self.authorized_maximum_cost_usd:
            raise ValueError(
                "Modal estimate exceeds authorization; raise the ceiling or reduce duration"
            )
        return self
