"""Pinned private vLLM launch parameters for the exact-token sampling boundary."""

from __future__ import annotations

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.runtime.claas.registry import ServingRevision
from exp.runtime.claas.vllm import serving_model_name


class VllmServerConfig(ContractModel):
    """Explicit base identity and finite capacity for a privately owned vLLM process.

    Run the returned arguments in the chosen pinned vLLM environment. The control
    API binds loopback because development sleep and dynamic adapter endpoints
    must be reachable only by the owning CLaaS lifecycle.
    """

    base: ServingRevision
    port: int = Field(default=8000, strict=True, ge=1, le=65535)
    max_model_len: int = Field(default=8192, strict=True, ge=2, le=131072)
    max_lora_rank: int = Field(default=16, strict=True, ge=1, le=256)
    gpu_memory_utilization: float = Field(default=0.8, strict=True, gt=0, lt=1)

    @model_validator(mode="after")
    def _require_base(self) -> VllmServerConfig:
        """Keep adapter selection separate from pinned base process construction."""
        if self.base.adapter_directory is not None:
            raise ValueError(
                "vLLM launch requires the frozen base; load adapters through lifecycle"
            )
        return self

    def command(self) -> tuple[str, ...]:
        """Return shell-free arguments with generation defaults disabled for exact evidence."""
        return (
            "vllm",
            "serve",
            self.base.model_id,
            "--revision",
            self.base.model_revision,
            "--tokenizer",
            self.base.tokenizer_id,
            "--tokenizer-revision",
            self.base.tokenizer_revision,
            "--served-model-name",
            serving_model_name(self.base),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            "bfloat16",
            "--generation-config",
            "vllm",
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--enable-sleep-mode",
            "--enable-lora",
            "--max-loras",
            "1",
            "--max-lora-rank",
            str(self.max_lora_rank),
        )

    def environment(self) -> dict[str, str]:
        """Return only required server switches, leaving credentials and GPU selection to caller."""
        return {"VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
