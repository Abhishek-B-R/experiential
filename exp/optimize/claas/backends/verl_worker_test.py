"""Real CPU LoRA optimizer, EMA, exact-token, checkpoint, and GPU-boundary tests."""

import copy
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from exp.optimize.claas.backends.checkpoints import verify_checkpoint
from exp.optimize.claas.backends.verl_worker import (
    _teacher_context,
    require_worker_runtime,
    train_loaded_model,
)
from exp.optimize.claas.training_contracts import ClaasTrainingError, TrainingBatch, TrainingJob
from exp.optimize.claas.training_contracts_test import example, job


def tokenizer() -> PreTrainedTokenizerFast:
    """Create a local eight-token tokenizer that cannot download model assets."""
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


def test_real_sdpo_update_resume_and_ema_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backprop changes student LoRA weights and preserves teacher/optimizer on resume."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
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
        }
    )
    base = GPT2LMHeadModel(config)
    initial = copy.deepcopy(base)
    first_job = job(tmp_path)
    result = train_loaded_model(first_job, base, tokenizer(), torch.device("cpu"))
    assert result.checkpoint.step == 1
    assert result.metrics["gradient_norm"] > 0
    verify_checkpoint(result.checkpoint, first_job.spec)
    student = load_file(str(Path(result.checkpoint.path) / "student/adapter_model.safetensors"))
    teacher = load_file(str(Path(result.checkpoint.path) / "teacher/adapter_model.safetensors"))
    b_name = next(name for name in student if "lora_B" in name)
    assert student[b_name].abs().sum() > 0
    torch.testing.assert_close(
        teacher[b_name], student[b_name] * first_job.spec.teacher_update_rate
    )
    second_batch = TrainingBatch(
        batch_id="batch-2",
        expected_policy_revision=result.checkpoint.policy_revision,
        examples=(example(policy=result.checkpoint.policy_revision),),
    )
    second_job = TrainingJob(
        spec=first_job.spec,
        batch=second_batch,
        checkpoint_root=str(tmp_path),
        resume_checkpoint=result.checkpoint,
    )
    second = train_loaded_model(second_job, initial, tokenizer(), torch.device("cpu"))
    assert second.checkpoint.step == 2
    verify_checkpoint(second.checkpoint, first_job.spec)
    state = torch.load(Path(second.checkpoint.path) / "optimizer.pt", weights_only=True)
    assert all(float(cast(torch.Tensor, value["step"])) == 2 for value in state["state"].values())
    with pytest.raises(ClaasTrainingError, match="already has a checkpoint"):
        train_loaded_model(second_job, copy.deepcopy(initial), tokenizer(), torch.device("cpu"))


def test_feedback_context_keeps_original_token_ids() -> None:
    """Feedback tokenization appends context and leaves sampled IDs unchanged."""
    item = example()
    context = _teacher_context(item, tokenizer(), 128)
    assert context is not None and context[:2] == (1, 2)
    assert item.experience.exact_tokens is not None
    assert item.experience.exact_tokens.response_token_ids == (3, 4)
    with pytest.raises(ValueError, match="shorten feedback"):
        _teacher_context(item, tokenizer(), 4)


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
    from exp.optimize.claas.backends.verl_worker import _validate_model_reference

    with pytest.raises(ValueError, match="not revision-bound"):
        _validate_model_reference(str(tmp_path), "a" * 40)
    with pytest.raises(ValueError, match="immutable"):
        _validate_model_reference("owner/model", "main")
    _validate_model_reference("owner/model", "a" * 40)


def test_checkpoint_flush_failure_cannot_publish_a_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disk flush failure leaves no complete directory that could be acknowledged."""
    import os

    def fail_flush(_fd: int) -> None:
        """Simulate failure before the atomic checkpoint publication boundary."""
        raise OSError("checkpoint device failed")

    config = GPT2Config.from_dict(
        {"vocab_size": 8, "n_positions": 128, "n_embd": 16, "n_layer": 1, "n_head": 2}
    )
    monkeypatch.setattr(os, "fsync", fail_flush)
    with pytest.raises(OSError, match="checkpoint device failed"):
        train_loaded_model(job(tmp_path), GPT2LMHeadModel(config), tokenizer(), torch.device("cpu"))
    assert not list(tmp_path.glob("*/claas-*/manifest.json"))
    assert not list(tmp_path.glob("*/.checkpoint-*"))
