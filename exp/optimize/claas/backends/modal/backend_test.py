"""No-spend Modal authorization, durable checkpoint transfer, and lifecycle tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import modal
import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.backends.checkpoints import (
    CheckpointManifest,
    hash_file,
    verify_checkpoint,
)
from exp.optimize.claas.backends.checkpoints_test import checkpoint
from exp.optimize.claas.backends.modal.backend import (
    REMOTE_ROOT,
    ModalVerlBackend,
    _download_checkpoint,
)
from exp.optimize.claas.backends.modal.configuration import ModalExecutionConfig
from exp.optimize.claas.training_contracts import (
    ClaasTrainingError,
    TrainingJob,
    TrainingResult,
    next_policy_revision,
)
from exp.optimize.claas.training_contracts_test import job, spec


def config() -> ModalExecutionConfig:
    """Return one explicit 12-cent fixture authorization, with no live rate claim."""
    return ModalExecutionConfig(
        app_name="fixture",
        volume_name="existing-fixture",
        gpu="A10G",
        timeout_seconds=60,
        startup_timeout_seconds=60,
        maximum_container_rate_usd_per_second=0.001,
        authorized_maximum_cost_usd=0.12,
    )


class _FileReader:
    """An SDK-shaped bounded byte source backed by inert local fixture files."""

    def __init__(self, root: Path) -> None:
        """Bind the remote fixture checkpoint directory."""
        self.root = root

    async def aio(self, path: str) -> AsyncIterator[bytes]:
        """Yield chunks from the requested manifest-listed relative file."""
        relative = Path(*Path(path).parts[3:])
        content = (self.root / relative).read_bytes()
        yield content[:10]
        yield content[10:]


class _Volume:
    """Expose the SDK read_file method without contacting Modal."""

    def __init__(self, root: Path) -> None:
        """Bind local test payloads."""
        self.read_file = _FileReader(root)


@pytest.mark.parametrize("serving_export", [False, True])
def test_download_verifies_manifest_and_preserves_immutable_scope(
    tmp_path: Path, serving_export: bool
) -> None:
    """A remote receipt becomes local activation evidence only after every digest passes."""
    source = tmp_path / "remote"
    receipt = checkpoint(source)
    if serving_export:
        manifest = verify_checkpoint(receipt, spec())
        files = dict(manifest.files)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            target = source / "serving" / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(b"inert-serving-payload")
            files[f"serving/{name}"] = hash_file(target)
        manifest = manifest.model_copy(
            update={"serving_adapter_directory": "serving", "files": files}
        )
        (source / "manifest.json").write_text(manifest.model_dump_json())
        receipt = receipt.model_copy(update={"manifest_sha256": sha256_json(manifest)})
    scope_id = sha256_json(
        {"scope": spec().scope.model_dump(mode="json"), "adapter_id": spec().adapter_id}
    )
    lineage = sha256_json({"lineage_id": "main"})
    remote = receipt.model_copy(
        update={"path": f"{REMOTE_ROOT}/{scope_id}/{lineage}/{receipt.policy_revision}"}
    )

    async def run() -> None:
        """Transfer a complete inert checkpoint then detect changed source bytes."""
        local = await _download_checkpoint(
            cast(modal.Volume, _Volume(source)), remote, spec(), tmp_path / "local", 1_000_000
        )
        verified = verify_checkpoint(local, spec())
        assert verified.step == 1
        assert verified.serving_adapter_directory == ("serving" if serving_export else "student")
        if serving_export:
            assert (Path(local.path) / "serving/adapter_model.safetensors").read_bytes() == (
                b"inert-serving-payload"
            )
        assert Path(local.path).is_dir()
        assert local.path.startswith(str(tmp_path / "local" / scope_id))
        (source / "verl/actor/optim_world_size_1_rank_0.pt").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="missing or changed"):
            await _download_checkpoint(
                cast(modal.Volume, _Volume(source)), remote, spec(), tmp_path / "second", 1_000_000
            )
        assert not (tmp_path / "second" / scope_id / receipt.policy_revision).exists()
        with pytest.raises(ValueError, match="byte ceiling"):
            await _download_checkpoint(
                cast(modal.Volume, _Volume(source)), remote, spec(), tmp_path / "limited", 1
            )

    asyncio.run(run())


def test_one_authorization_never_silently_retries(tmp_path: Path) -> None:
    """An uncertain remote update consumes its reservation and cannot be retried in place."""

    class FailingBackend(ModalVerlBackend):
        """Simulate uncertainty after the cloud boundary without allocating compute."""

        calls = 0

        async def execute(self, job: TrainingJob) -> TrainingResult:
            """Record the dispatch and fail before a verifiable receipt arrives."""
            self.calls += 1
            raise ClaasTrainingError("uncertain remote completion")

    async def run() -> None:
        """Validate before dispatch, consume once, and close without retries."""
        backend = FailingBackend(config=config(), checkpoint_root=tmp_path)
        session = await backend.open(spec())
        assert backend.calls == 0
        bad = job(tmp_path).batch.model_copy(update={"expected_policy_revision": "foreign"})
        with pytest.raises(ValueError, match="stale"):
            await session.train(bad)
        assert backend.calls == 0
        with pytest.raises(ClaasTrainingError, match="uncertain"):
            await session.train(job(tmp_path).batch)
        with pytest.raises(ClaasTrainingError, match="consumed"):
            await session.train(job(tmp_path).batch)
        assert backend.calls == 1
        await session.close()

    asyncio.run(run())


def test_close_waits_for_execution_cancellation(tmp_path: Path) -> None:
    """Concurrent close cannot leave a locally tracked dispatch task running."""

    async def run() -> None:
        """Hold execution until close cancels it and confirm cleanup ran."""
        entered, cleanup = asyncio.Event(), asyncio.Event()

        class BlockingBackend(ModalVerlBackend):
            """Simulate a remote task whose cancellation performs cleanup."""

            async def execute(self, job: TrainingJob) -> TrainingResult:
                """Wait until cancellation then acknowledge local cleanup."""
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup.set()
                raise AssertionError("unreachable")

        backend = BlockingBackend(config=config(), checkpoint_root=tmp_path)
        session = await backend.open(spec())
        training = asyncio.create_task(session.train(job(tmp_path).batch))
        await entered.wait()
        await session.close()
        assert cleanup.is_set()
        with pytest.raises(asyncio.CancelledError):
            await training

    asyncio.run(run())


def test_named_lineage_survives_dispatch_download_and_exact_job_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful remote job downloads its own lineage rather than the default lineage."""
    submitted = job(tmp_path).model_copy(
        update={"lineage_id": "cycle-claims", "checkpoint_root": REMOTE_ROOT}
    )
    source = tmp_path / "remote"
    receipt = checkpoint(source)
    policy = next_policy_revision(submitted)
    ids = tuple(item.experience.experience_id for item in submitted.batch.examples)
    manifest = CheckpointManifest.model_validate_json((source / "manifest.json").read_bytes())
    manifest = manifest.model_copy(
        update={
            "policy_revision": policy,
            "policy_history": (policy, submitted.spec.initial_policy_revision),
            "batch_id": submitted.batch.batch_id,
            "consumed_experience_ids": ids,
            "lineage_id": submitted.lineage_id,
        }
    )
    (source / "manifest.json").write_text(manifest.model_dump_json())
    scope_hash = sha256_json(
        {"scope": submitted.spec.scope.model_dump(mode="json"), "adapter_id": spec().adapter_id}
    )
    lineage_hash = sha256_json({"lineage_id": submitted.lineage_id})
    receipt = receipt.model_copy(
        update={
            "policy_revision": policy,
            "policy_history": manifest.policy_history,
            "manifest_sha256": sha256_json(manifest),
            "path": f"{REMOTE_ROOT}/{scope_hash}/{lineage_hash}/{policy}",
        }
    )
    result = TrainingResult(checkpoint=receipt, consumed_experience_ids=ids, metrics={})
    call = MagicMock()
    call.get.aio = AsyncMock(return_value=result.model_dump_json())
    function = MagicMock()
    function.with_options.return_value = function
    function.spawn.aio = AsyncMock(return_value=call)
    monkeypatch.setattr(modal.Function, "from_name", MagicMock(return_value=function))
    monkeypatch.setattr(modal.Volume, "from_name", MagicMock(return_value=_Volume(source)))
    backend = ModalVerlBackend(
        config=config(), checkpoint_root=tmp_path / "local", lineage_id=submitted.lineage_id
    )
    local = asyncio.run(backend.execute(submitted))
    assert local.checkpoint.path == str(tmp_path / "local" / scope_hash / lineage_hash / policy)
    assert verify_checkpoint(local.checkpoint, spec()).lineage_id == submitted.lineage_id
    function.spawn.aio.assert_awaited_once_with(submitted.model_dump_json())
