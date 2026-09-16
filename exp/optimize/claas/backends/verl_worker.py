"""Standalone, one-rank upstream veRL training worker for portable CLaaS jobs.

The published veRL TrainingWorker owns model/LoRA construction, backward,
optimizer steps, schedulers, and native checkpoint resume. CLaaS supplies exact
rollouts, the SDPO objective callback, EMA teacher updates and immutable receipts.
Local and Modal execution invoke this same module; there is no custom trainer or
CPU optimizer fallback.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import math
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
from filelock import FileLock
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import (
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    checkpoint_snapshot,
    verify_checkpoint,
)
from exp.optimize.claas.backends.verl_engine import ENGINE_MODEL_TYPE
from exp.optimize.claas.backends.verl_inputs import build_engine_batch
from exp.optimize.claas.backends.verl_objective import FeedbackLoss
from exp.optimize.claas.backends.verl_state import publish_checkpoint, update_teacher
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    TrainingJob,
    TrainingResult,
    next_policy_revision,
)

logger = logging.getLogger(__name__)


def require_worker_runtime() -> None:
    """Reject incompatible dependencies and ambiguous GPU placement before loading weights."""
    if importlib.metadata.version("verl") != "0.9.0":
        raise ClaasTrainingError(
            "this worker requires verl==0.9.0; install experiential[claas-verl]"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise ClaasTrainingError("set CUDA_VISIBLE_DEVICES to exactly one authorized GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ClaasTrainingError("the CLaaS veRL worker requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ClaasTrainingError("the selected GPU must support BF16; no implicit dtype fallback")


def _validate_model_reference(identifier: str, revision: str) -> None:
    """Require immutable remote model and tokenizer revisions."""
    if Path(identifier).is_absolute() or Path(identifier).exists():
        raise ValueError("mutable local model/tokenizer directories are not revision-bound")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            "remote model and tokenizer revisions must be immutable 40-character commits"
        )


@contextmanager
def single_rank_process_group() -> Iterator[None]:
    """Own one finite, private distributed process group without starting a Ray cluster."""
    if torch.distributed.is_initialized():
        raise ClaasTrainingError(
            "run the CLaaS worker in its own process, outside an existing process group"
        )
    with tempfile.TemporaryDirectory(prefix="claas-verl-rendezvous-") as directory:
        previous = {
            name: os.environ.get(name)
            for name in (
                "WORLD_SIZE",
                "RANK",
                "LOCAL_WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            )
        }
        os.environ.update(
            WORLD_SIZE="1",
            RANK="0",
            LOCAL_WORLD_SIZE="1",
            LOCAL_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="0",
        )
        try:
            torch.cuda.set_device(0)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                rank=0,
                world_size=1,
                init_method=(Path(directory) / "store").as_uri(),
                timeout=timedelta(seconds=120),
            )
            yield
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def worker_config(
    job: TrainingJob,
    model_path: Path,
    tokenizer_path: Path,
    *,
    teacher: bool,
) -> TrainingWorkerConfig:
    """Configure original-model LoRA and native state using veRL's typed public API."""
    model = HFModelConfig(
        path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        trust_remote_code=False,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=True,
        lora_rank=job.spec.lora_rank,
        lora_alpha=job.spec.lora_alpha,
        target_modules=list(job.spec.target_modules),
        override_config={"attn_implementation": "sdpa"},
    )
    if model.hf_config.is_encoder_decoder or getattr(model.hf_config, "quantization_config", None):
        raise ValueError("CLaaS veRL requires an unquantized causal language model")
    if (
        str(model.hf_config.model_type).startswith("qwen3_5")
        and model.hf_config.model_type != "qwen3_5"
    ):
        raise ValueError("CLaaS requires the original dense Qwen3.5 wrapper checkpoint")
    contents = ["model"] if teacher else ["model", "optimizer", "extra"]
    return TrainingWorkerConfig(
        model_type=ENGINE_MODEL_TYPE,
        model_config=model,
        engine_config=FSDPEngineConfig(
            strategy="fsdp",
            forward_only=teacher,
            fsdp_size=1,
            use_orig_params=True,
            wrap_policy={"disable": True},
            model_dtype="bf16",
            dtype="bfloat16",
            mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
            use_torch_compile=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1,
            use_remove_padding=False,
            seed=job.spec.seed,
        ),
        optimizer_config=FSDPOptimizerConfig(
            lr=job.spec.learning_rate,
            weight_decay=0.0,
            clip_grad=job.spec.max_gradient_norm,
            lr_scheduler_type="constant",
            total_training_steps=1,
        ),
        checkpoint_config=CheckpointConfig(
            save_contents=contents,
            load_contents=contents,
            save_lora_only=True,
        ),
    )


def execute_training_job(job: TrainingJob) -> TrainingResult:
    """Resolve immutable inputs then delegate the complete optimizer update to veRL."""
    require_worker_runtime()
    _validate_model_reference(job.spec.base_model, job.spec.model_revision)
    _validate_model_reference(job.spec.tokenizer_id, job.spec.tokenizer_revision)
    if job.resume_checkpoint:
        verify_checkpoint(job.resume_checkpoint, job.spec)
    model_path = Path(
        snapshot_download(repo_id=job.spec.base_model, revision=job.spec.model_revision)
    )
    tokenizer_path = Path(
        snapshot_download(repo_id=job.spec.tokenizer_id, revision=job.spec.tokenizer_revision)
    )
    return train_local_snapshots(job, model_path, tokenizer_path)


def train_local_snapshots(
    job: TrainingJob, model_path: Path, tokenizer_path: Path
) -> TrainingResult:
    """Run actual CUDA veRL workers using already resolved snapshots or offline test fixtures.

    The production entrypoint alone resolves immutable remote revisions. This
    lower-level seam lets the CUDA integration test create tiny local weights;
    it never substitutes a CPU or synthetic optimizer for upstream veRL.
    """
    require_worker_runtime()
    with checkpoint_snapshot(job.resume_checkpoint, job.spec) as resume:
        staged = job.model_copy(update={"resume_checkpoint": resume})
        root = (
            Path(job.checkpoint_root).resolve()
            / sha256_json(
                {
                    "scope": job.spec.scope.model_dump(mode="json"),
                    "adapter_id": job.spec.adapter_id,
                }
            )
            / sha256_json({"lineage_id": job.lineage_id})
        )
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / ".training.lock", timeout=0):
            destination = root / next_policy_revision(staged)
            _validate_lineage(staged, root, destination)
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "model tokenizer must implement the Hugging Face text tokenizer contract"
                )
            actor_batch = build_engine_batch(staged, tokenizer)
            teacher_batch = (
                build_engine_batch(staged, tokenizer, teacher=True)
                if staged.spec.objective != "reinforce"
                else None
            )
            with single_rank_process_group():
                torch.manual_seed(staged.spec.seed)
                actor = TrainingWorker(
                    worker_config(staged, model_path, tokenizer_path, teacher=False)
                )
                actor.reset()
                teacher = TrainingWorker(
                    worker_config(staged, model_path, tokenizer_path, teacher=True)
                )
                teacher.reset()
                if resume is not None:
                    teacher.load_checkpoint(
                        str(Path(resume.path) / "verl" / "teacher"), del_local_after_load=False
                    )
                    actor.load_checkpoint(
                        str(Path(resume.path) / "verl" / "actor"), del_local_after_load=False
                    )
                else:
                    update_teacher(actor, teacher, 1.0)
                loss = FeedbackLoss(staged)
                if teacher_batch is not None:
                    target = FeedbackLoss(staged, teacher=True)
                    teacher.set_loss_fn(target)
                    teacher.infer_batch(teacher_batch)
                    loss.teacher_logits = target.teacher_logits
                actor.set_loss_fn(loss)
                output = actor.train_batch(actor_batch)
                reported = tu.get(output, "metrics")
                metrics = {name: float(reported[name]) for name in ("loss", "grad_norm")}
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ClaasTrainingError(
                        "veRL returned nonfinite training metrics; checkpoint publication refused"
                    )
                metrics["response_tokens"] = float(loss.response_tokens)
                update_teacher(actor, teacher, staged.spec.teacher_update_rate)
                return publish_checkpoint(staged, actor, teacher, metrics, destination)


def _validate_lineage(job: TrainingJob, root: Path, destination: Path) -> None:
    """Reject duplicate updates or stale resume within the selected candidate lineage."""
    if destination.exists():
        raise ClaasTrainingError("this batch already has a checkpoint; do not replay the update")
    completed = [
        CheckpointManifest.model_validate_json(path.read_text())
        for path in root.glob("claas-*/manifest.json")
    ]
    if completed:
        latest = max(completed, key=lambda item: item.step)
        if (
            job.resume_checkpoint is None
            or latest.policy_revision != job.resume_checkpoint.policy_revision
        ):
            raise ClaasTrainingError(
                "adapter has newer state; resume its latest checkpoint before training"
            )


def main() -> None:
    """Execute one serialized job and write its typed completion receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    result = execute_training_job(TrainingJob.model_validate_json(args.job.read_text()))
    args.result.write_text(result.model_dump_json())
    logger.info("Completed upstream veRL optimizer step %s", result.checkpoint.step)


if __name__ == "__main__":
    main()
