"""Explicit single-device veRL objective worker for full-vocabulary SDPO LoRA.

Run only inside the optional ``experiential[claas-verl]`` environment. One job
loads one frozen base with student and EMA teacher adapters, applies one update,
atomically saves resumable state, and exits. It never starts Ray, a serving
process, or paid cloud resources. CUDA placement belongs to the execution adapter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import cast

import torch
from filelock import FileLock
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.algorithms.sdpo import feedback_objective
from exp.optimize.claas.backends.checkpoints import CheckpointManifest, hash_file, verify_checkpoint
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    TrainingCheckpoint,
    TrainingExample,
    TrainingJob,
    TrainingResult,
    next_policy_revision,
)

logger = logging.getLogger(__name__)


def require_worker_runtime() -> None:
    """Reject unpinned libraries and ambiguous GPU placement before loading weights."""
    if importlib.metadata.version("verl") != "0.9.0":
        raise ClaasTrainingError(
            "this worker requires verl==0.9.0; install experiential[claas-verl]"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise ClaasTrainingError("set CUDA_VISIBLE_DEVICES to exactly one authorized GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ClaasTrainingError("the CLaaS worker requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ClaasTrainingError("the selected GPU must support BF16; no implicit dtype fallback")


def _validate_model_reference(identifier: str, revision: str) -> None:
    """Require pinned Hugging Face commits or an explicit existing local model path."""
    if Path(identifier).is_absolute():
        if not Path(identifier).is_dir():
            raise ValueError("local model or tokenizer directory does not exist")
    elif re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            "remote model and tokenizer revisions must be immutable 40-character commits"
        )


def execute_training_job(job: TrainingJob) -> TrainingResult:
    """Load and train one authorized CUDA job using CLaaS-owned algorithm code.

    GPU execution remains a separate validation gate from CPU optimizer tests.
    Supported models must fit their frozen base, two LoRA adapters, activations,
    and full-vocabulary logits on the selected device. No quantization, FSDP,
    multi-node execution, top-k distillation, or remote model code is enabled.
    """
    require_worker_runtime()
    _validate_model_reference(job.spec.base_model, job.spec.model_revision)
    _validate_model_reference(job.spec.tokenizer_id, job.spec.tokenizer_revision)
    if job.resume_checkpoint:
        verify_checkpoint(job.resume_checkpoint, job.spec)
    torch.manual_seed(job.spec.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        job.spec.tokenizer_id, revision=job.spec.tokenizer_revision, trust_remote_code=False
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("model tokenizer must implement the Hugging Face text tokenizer contract")
    base = AutoModelForCausalLM.from_pretrained(
        job.spec.base_model,
        revision=job.spec.model_revision,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    if base.config.is_encoder_decoder or getattr(base.config, "quantization_config", None):
        raise ValueError("CLaaS requires an unquantized text causal language model")
    return train_loaded_model(job, base, tokenizer, torch.device("cuda:0"))


def _open_adapters(job: TrainingJob, base: PreTrainedModel) -> PeftModel:
    """Create or restore two LoRA adapters sharing one frozen base model."""
    if job.resume_checkpoint:
        root = Path(job.resume_checkpoint.path)
        model = PeftModel.from_pretrained(
            base, root / "student", adapter_name="student", is_trainable=True
        )
        model.load_adapter(root / "teacher", adapter_name="teacher", is_trainable=False)
    else:
        config = LoraConfig(
            r=job.spec.lora_rank,
            lora_alpha=job.spec.lora_alpha,
            lora_dropout=0.0,
            target_modules=list(job.spec.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = cast(PeftModel, get_peft_model(base, config, adapter_name="student"))
        model.add_adapter("teacher", config)
        set_peft_model_state_dict(
            model, get_peft_model_state_dict(model, adapter_name="student"), adapter_name="teacher"
        )
    model.set_adapter("student")
    # Evaluation mode disables all dropout without disabling autograd for student LoRA weights.
    model.eval()
    return model


def _response_logits(
    model: PeftModel, context: tuple[int, ...], response: tuple[int, ...], device: torch.device
) -> Tensor:
    """Forward original token IDs and return logits predicting exactly the response."""
    ids = torch.tensor([context + response], dtype=torch.long, device=device)
    output = cast(CausalLMOutputWithPast, model(input_ids=ids, use_cache=False))
    if output.logits is None:
        raise ValueError("model must expose full-vocabulary logits for CLaaS training")
    return output.logits[0, len(context) - 1 : -1]


def _teacher_context(
    item: TrainingExample, tokenizer: PreTrainedTokenizerBase, max_sequence_tokens: int
) -> tuple[int, ...] | None:
    """Append newly tokenized feedback context without re-tokenizing sampled IDs."""
    tokens = item.experience.exact_tokens
    if tokens is None:
        raise ValueError("training requires original token evidence")
    if item.text_feedback is None:
        return None
    hint = (
        "\n\nFeedback from a previous attempt:\n"
        + item.text_feedback
        + "\n\nUsing this feedback, produce the best response to the original request.\n"
    )
    added = tuple(tokenizer.encode(hint, add_special_tokens=False))
    context = tokens.prompt_token_ids + added
    if len(context) + len(tokens.response_token_ids) > max_sequence_tokens:
        raise ValueError(
            "feedback-conditioned sequence exceeds max_sequence_tokens; shorten feedback"
        )
    return context


def _update_teacher(model: PeftModel, rate: float) -> None:
    """Update only teacher LoRA parameters by EMA, leaving the shared base frozen."""
    student = get_peft_model_state_dict(model, adapter_name="student")
    teacher = get_peft_model_state_dict(model, adapter_name="teacher")
    if student.keys() != teacher.keys():
        raise ValueError("student and teacher adapters have incompatible parameter sets")
    with torch.no_grad():
        updated = {
            name: teacher[name] * (1 - rate) + value.detach() * rate
            for name, value in student.items()
        }
        set_peft_model_state_dict(model, updated, adapter_name="teacher")


def train_loaded_model(
    job: TrainingJob,
    base: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> TrainingResult:
    """Train an already loaded model; CPU tests use locally generated tiny weights.

    This seam performs the same forward, backward, optimizer, EMA, and checkpoint
    operations as the CUDA entrypoint. It does not download or select compute.
    """
    if job.resume_checkpoint:
        verify_checkpoint(job.resume_checkpoint, job.spec)
    root = Path(job.checkpoint_root).resolve()
    adapter_root = root / sha256_json(
        {"scope": job.spec.scope.model_dump(mode="json"), "adapter_id": job.spec.adapter_id}
    )
    adapter_root.mkdir(parents=True, exist_ok=True)
    with FileLock(adapter_root / ".training.lock", timeout=0):
        destination = adapter_root / next_policy_revision(job)
        if destination.exists():
            raise ClaasTrainingError(
                "this batch already has a checkpoint; do not replay the update"
            )
        completed = [
            CheckpointManifest.model_validate_json(path.read_text())
            for path in adapter_root.glob("claas-*/manifest.json")
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
        contexts = [
            _teacher_context(item, tokenizer, job.spec.max_sequence_tokens)
            if job.spec.objective != "reinforce"
            else None
            for item in job.batch.examples
        ]
        total_context_tokens = sum(
            len(item.experience.exact_tokens.prompt_token_ids)
            + len(item.experience.exact_tokens.response_token_ids)
            + (
                len(context) + len(item.experience.exact_tokens.response_token_ids)
                if context
                else 0
            )
            for item, context in zip(job.batch.examples, contexts, strict=True)
            if item.experience.exact_tokens is not None
        )
        if total_context_tokens > job.spec.max_batch_tokens:
            raise ValueError("student and teacher inputs exceed max_batch_tokens; split the batch")
        model = _open_adapters(job, base).to(device)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=job.spec.learning_rate,
            weight_decay=0.0,
        )
        if job.resume_checkpoint:
            state = torch.load(
                Path(job.resume_checkpoint.path) / "optimizer.pt",
                map_location=device,
                weights_only=True,
            )
            optimizer.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)
        token_count = sum(
            len(item.experience.exact_tokens.response_token_ids)
            for item in job.batch.examples
            if item.experience.exact_tokens
        )
        metrics: dict[str, float] = {"loss": 0.0, "response_tokens": float(token_count)}
        for item, teacher_context in zip(job.batch.examples, contexts, strict=True):
            tokens = item.experience.exact_tokens
            if tokens is None:
                raise ValueError("training requires original token evidence")
            teacher_logits = None
            if teacher_context is not None:
                model.set_adapter("teacher")
                with torch.no_grad():
                    teacher_logits = _response_logits(
                        model, teacher_context, tokens.response_token_ids, device
                    )
            model.set_adapter("student")
            student_logits = _response_logits(
                model, tokens.prompt_token_ids, tokens.response_token_ids, device
            )
            loss, observed = feedback_objective(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                response_tokens=torch.tensor(
                    tokens.response_token_ids, dtype=torch.long, device=device
                ),
                rollout_logprobs=torch.tensor(tokens.response_logprobs, device=device),
                scalar_reward=item.scalar_reward,
                spec=job.spec,
                batch_response_tokens=token_count,
            )
            loss.backward()
            metrics["loss"] += float(loss.detach())
            for name, value in observed.items():
                metrics[name] = metrics.get(name, 0.0) + value / len(job.batch.examples)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            job.spec.max_gradient_norm,
            error_if_nonfinite=True,
        )
        metrics["gradient_norm"] = float(gradient_norm)
        optimizer.step()
        _update_teacher(model, job.spec.teacher_update_rate)
        return _save_result(job, model, optimizer, metrics, destination)


def _save_result(
    job: TrainingJob,
    model: PeftModel,
    optimizer: torch.optim.AdamW,
    metrics: dict[str, float],
    destination: Path,
) -> TrainingResult:
    """Publish complete student/teacher/optimizer files with one atomic directory rename."""
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        model.save_pretrained(
            str(temporary), selected_adapters=["student", "teacher"], safe_serialization=True
        )
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        files = {
            str(path.relative_to(temporary)): hash_file(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        manifest = CheckpointManifest(
            spec=job.spec,
            policy_revision=destination.name,
            parent_policy_revision=job.batch.expected_policy_revision,
            policy_history=(
                destination.name,
                *(
                    job.resume_checkpoint.policy_history
                    if job.resume_checkpoint
                    else (job.spec.initial_policy_revision,)
                ),
            )[: job.spec.max_policy_lag + 1],
            step=(job.resume_checkpoint.step if job.resume_checkpoint else 0) + 1,
            batch_id=job.batch.batch_id,
            consumed_experience_ids=tuple(
                item.experience.experience_id for item in job.batch.examples
            ),
            files=files,
        )
        (temporary / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    checkpoint = TrainingCheckpoint(
        scope=job.spec.scope,
        adapter_id=job.spec.adapter_id,
        policy_revision=manifest.policy_revision,
        policy_history=manifest.policy_history,
        step=manifest.step,
        path=str(destination),
        manifest_sha256=sha256_json(manifest),
    )
    verify_checkpoint(checkpoint, job.spec)
    return TrainingResult(
        checkpoint=checkpoint,
        metrics=metrics,
        consumed_experience_ids=manifest.consumed_experience_ids,
    )


def main() -> None:
    """Execute an explicit serialized job and write its typed completion receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    job = TrainingJob.model_validate_json(args.job.read_text())
    result = execute_training_job(job)
    args.result.write_text(result.model_dump_json())
    logger.info("Completed CLaaS optimizer step %s", result.checkpoint.step)


if __name__ == "__main__":
    main()
