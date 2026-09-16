"""Upstream worker configuration contracts and explicit opt-in CUDA training proof.

CPU tests do not claim to execute veRL's CUDA-only FSDP optimizer. The separate
GPU test runs real worker initialization, updates, native resume and PEFT export.
"""

import os
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)
from verl.workers.engine import BaseEngine, FSDPEngineWithLMHead

from exp.optimize.claas.backends.checkpoints import verify_checkpoint
from exp.optimize.claas.backends.verl_engine import ClaasFeedbackEngine
from exp.optimize.claas.backends.verl_worker import (
    _validate_lineage,
    _validate_model_reference,
    require_worker_runtime,
    train_local_snapshots,
    worker_config,
)
from exp.optimize.claas.training_contracts import ClaasTrainingError, TrainingBatch, TrainingJob
from exp.optimize.claas.training_contracts_test import example, job


def tokenizer() -> PreTrainedTokenizerFast:
    """Create a local eight-token tokenizer that cannot download model assets."""
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


def tiny_snapshot(path: Path) -> Path:
    """Save deterministic tiny full-model and tokenizer fixtures without a download."""
    torch.manual_seed(1)
    config = GPT2Config.from_dict(
        {
            "vocab_size": 8,
            "n_positions": 128,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "bos_token_id": 1,
            "eos_token_id": 7,
            "attn_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "resid_pdrop": 0.0,
        }
    )
    GPT2LMHeadModel(config).save_pretrained(path)
    tokenizer().save_pretrained(path)
    return path


def test_worker_config_selects_native_model_optimizer_and_checkpoint_ownership(
    tmp_path: Path,
) -> None:
    """Exercise actual upstream typed configuration and inherited execution methods."""
    snapshot = tiny_snapshot(tmp_path / "base")
    training = job(tmp_path / "results")
    actor = worker_config(training, snapshot, snapshot, teacher=False)
    teacher = worker_config(training, snapshot, snapshot, teacher=True)
    assert actor.model_config.local_path == str(snapshot)
    assert actor.model_config.lora_rank == training.spec.lora_rank
    assert actor.optimizer_config.lr == training.spec.learning_rate
    assert actor.checkpoint_config.save_contents == ["model", "optimizer", "extra"]
    assert actor.checkpoint_config.load_contents == ["model", "optimizer", "extra"]
    assert actor.checkpoint_config.save_lora_only
    assert teacher.engine_config.forward_only
    assert teacher.checkpoint_config.save_contents == ["model"]
    assert ClaasFeedbackEngine.initialize is FSDPEngineWithLMHead.initialize
    assert ClaasFeedbackEngine.train_batch is BaseEngine.train_batch
    assert ClaasFeedbackEngine.forward_backward_batch is FSDPEngineWithLMHead.forward_backward_batch
    assert ClaasFeedbackEngine.optimizer_step is FSDPEngineWithLMHead.optimizer_step
    assert ClaasFeedbackEngine.save_checkpoint is FSDPEngineWithLMHead.save_checkpoint
    assert ClaasFeedbackEngine.load_checkpoint is FSDPEngineWithLMHead.load_checkpoint


def test_gpu_placement_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU host or multiple visible GPUs cannot silently become a training run."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ClaasTrainingError, match="exactly one authorized"):
        require_worker_runtime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ClaasTrainingError, match="exactly one authorized"):
        require_worker_runtime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ClaasTrainingError, match="visible CUDA"):
        require_worker_runtime()


def test_mutable_local_model_references_are_not_treated_as_pinned(tmp_path: Path) -> None:
    """A revision string cannot freeze model or tokenizer files in a mutable directory."""
    with pytest.raises(ValueError, match="not revision-bound"):
        _validate_model_reference(str(tmp_path), "a" * 40)
    with pytest.raises(ValueError, match="immutable"):
        _validate_model_reference("owner/model", "main")
    _validate_model_reference("owner/model", "a" * 40)


def test_duplicate_checkpoint_rejected_before_worker_launch(tmp_path: Path) -> None:
    """An existing complete update cannot be replayed into its immutable directory."""
    destination = tmp_path / "complete"
    destination.mkdir()
    with pytest.raises(ClaasTrainingError, match="already has a checkpoint"):
        _validate_lineage(job(tmp_path), tmp_path, destination)


@pytest.mark.skipif(
    os.environ.get("CLAAS_RUN_CUDA_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires explicit CLAAS_RUN_CUDA_INTEGRATION=1 and an authorized CUDA GPU",
)
def test_cuda_upstream_verl_update_native_resume_and_export(tmp_path: Path) -> None:
    """Run unmocked veRL optimizer updates and prove its native Adam state resumes."""
    snapshot = tiny_snapshot(tmp_path / "base")
    first_job = job(tmp_path / "results")
    first = train_local_snapshots(first_job, snapshot, snapshot)
    manifest = verify_checkpoint(first.checkpoint, first_job.spec)
    assert manifest.training_backend == "verl-fsdp-0.9.0"
    assert first.metrics["grad_norm"] > 0
    root = Path(first.checkpoint.path)
    student = load_file(str(root / "student/adapter_model.safetensors"))
    teacher = load_file(str(root / "teacher/adapter_model.safetensors"))
    b_name = next(name for name in student if "lora_B" in name)
    assert student[b_name].abs().sum() > 0
    torch.testing.assert_close(
        teacher[b_name].float(),
        student[b_name].float() * first_job.spec.teacher_update_rate,
        atol=1e-5,
        rtol=0.02,
    )
    second_job = TrainingJob(
        spec=first_job.spec,
        checkpoint_root=first_job.checkpoint_root,
        resume_checkpoint=first.checkpoint,
        batch=TrainingBatch(
            batch_id="batch-2",
            expected_policy_revision=first.checkpoint.policy_revision,
            examples=(example(policy=first.checkpoint.policy_revision),),
        ),
    )
    second = train_local_snapshots(second_job, snapshot, snapshot)
    assert second.checkpoint.step == 2
    state = torch.load(
        Path(second.checkpoint.path) / "verl/actor/optim_world_size_1_rank_0.pt",
        weights_only=True,
    )
    assert state["state"]
    assert all(float(cast(torch.Tensor, value["step"])) == 2 for value in state["state"].values())
    assert (
        verify_checkpoint(second.checkpoint, first_job.spec).serving_adapter_directory == "student"
    )
    resumed_student = load_file(
        str(Path(second.checkpoint.path) / "student/adapter_model.safetensors")
    )
    resumed_teacher = load_file(
        str(Path(second.checkpoint.path) / "teacher/adapter_model.safetensors")
    )
    rate = first_job.spec.teacher_update_rate
    for name in resumed_teacher:
        torch.testing.assert_close(
            resumed_teacher[name].float(),
            teacher[name].float() * (1 - rate) + resumed_student[name].float() * rate,
            atol=1e-5,
            rtol=0.02,
        )


def tiny_qwen() -> Qwen3_5ForConditionalGeneration:
    """Create the actual hybrid text architecture plus tiny unused vision weights locally."""
    config = Qwen3_5Config.from_dict(
        {
            "text_config": {
                "vocab_size": 8,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "max_position_embeddings": 128,
                "tie_word_embeddings": True,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [1, 1, 2],
                },
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
                "patch_size": 2,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
            },
            "tie_word_embeddings": True,
        }
    )
    config._attn_implementation = "eager"
    return Qwen3_5ForConditionalGeneration(config)


def test_upstream_lora_construction_preserves_original_qwen_wrapper_names(tmp_path: Path) -> None:
    """Run upstream LoRA construction on CPU without claiming a CUDA optimizer update."""
    model = tiny_qwen()
    snapshot = tmp_path / "qwen"
    model.save_pretrained(snapshot)
    tokenizer().save_pretrained(snapshot)
    training = job(tmp_path / "results")
    training = training.model_copy(
        update={"spec": training.spec.model_copy(update={"target_modules": ("q_proj", "v_proj")})}
    )
    config = worker_config(training, snapshot, snapshot, teacher=False)
    engine = object.__new__(ClaasFeedbackEngine)
    engine.model_config = config.model_config
    adapted = engine._build_lora_module(model)
    exported = tmp_path / "adapter"
    adapted.save_pretrained(exported, safe_serialization=True)
    weights = load_file(str(exported / "adapter_model.safetensors"))
    assert weights
    assert all(name.startswith("base_model.model.model.language_model.") for name in weights)
    assert all("lora_" in name for name in weights)
