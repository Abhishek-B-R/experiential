"""Cross-process gateway admission leases bound to immutable local adapter pointers."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

import exp_gateway_native
from pydantic import Field, model_validator

from exp.common.claas import ClaasScope
from exp.common.claas.contracts import Identifier
from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.core.files import write_text_atomic
from exp.common.core.locks import file_write_lock
from exp.runtime.claas.registry import AdapterRegistry
from exp.runtime.claas.serving.vllm import serving_model_name


class GatewayServingBinding(ContractModel):
    """Bind one authorized user/alias to a private vLLM route and local registry."""

    scope: ClaasScope
    alias: Identifier
    registry_path: Path
    private_base_url: str
    state_path: Path
    admission_lock_path: Path

    @model_validator(mode="after")
    def _validate_paths(self) -> GatewayServingBinding:
        """Require distinct absolute coordination files and a local HTTP server origin."""
        paths = (self.registry_path, self.state_path, self.admission_lock_path)
        if any(not path.is_absolute() for path in paths) or len(set(paths)) != len(paths):
            raise ValueError(
                "registry, state, and admission lock paths must be distinct absolute paths"
            )
        if (
            re.fullmatch(
                r"http://(?:127\.0\.0\.1|localhost|\[::1\])(?::[0-9]{1,5})?/?",
                self.private_base_url,
            )
            is None
            or urlsplit(self.private_base_url).port == 0
        ):
            raise ValueError(
                "private_base_url must be a loopback HTTP origin on 127.0.0.1, localhost, "
                "or [::1], with no credentials, query, fragment, or non-root path"
            )
        return self

    @property
    def binding_sha256(self) -> str:
        """Bind coordination state to this exact scope, alias, endpoint, and files."""
        return sha256_json(self)


class GatewayServingConfiguration(ContractModel):
    """Opt-in private serving bindings, limited to one gateway alias per application."""

    bindings: tuple[GatewayServingBinding, ...]

    @model_validator(mode="after")
    def _unique_bindings(self) -> GatewayServingConfiguration:
        """Prevent duplicate authority, application leases, or coordination file reuse."""
        pairs = [(item.scope.user_id, item.alias) for item in self.bindings]
        scopes = [(item.scope.user_id, item.scope.application_id) for item in self.bindings]
        files = [
            path for item in self.bindings for path in (item.state_path, item.admission_lock_path)
        ]
        origins = [_private_origin(item.private_base_url) for item in self.bindings]
        if len(set(origins)) != len(origins):
            raise ValueError(
                "CLaaS supports one application per private vLLM origin; "
                "start a separate private server for each application"
            )
        if (
            len(set(pairs)) != len(pairs)
            or len(set(scopes)) != len(scopes)
            or len(set(files)) != len(files)
        ):
            raise ValueError(
                "each application, user/alias, and serving coordination file must be unique"
            )
        return self


def _private_origin(value: str) -> tuple[str, str, int]:
    """Canonicalize loopback aliases and default ports for exclusive server ownership."""
    url = urlsplit(value)
    host = url.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        host = "loopback"
    return url.scheme, host, url.port or (443 if url.scheme == "https" else 80)


class GatewayServingState(ContractModel):
    """Durable admission state which remains paused after a controller crash."""

    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    scope: ClaasScope
    binding_sha256: Sha256
    paused: bool = Field(strict=True)
    generation: int = Field(strict=True, ge=0)
    policy_revision: Identifier
    model_name: Identifier


class _NativeLease(Protocol):
    """Minimal native exclusive OS-lock ownership."""

    def release(self) -> None:
        """Release the held OS lock without unlinking its shared coordination inode."""
        ...


class GatewayAdmissionLease:
    """Pause all bound gateway requests before changing the shared GPU or adapter.

    The native gateway holds shared locks until each caller response body closes.
    This controller durably pauses before waiting for the exclusive lock, so new
    readers cannot starve the drain. Timeout or cancellation leaves the gate paused.
    """

    def __init__(self, binding: GatewayServingBinding) -> None:
        """Bind one explicit scope without touching admission or creating a model."""
        self.binding = binding
        self.registry = AdapterRegistry(binding.registry_path, binding.scope)
        self._lease: _NativeLease | None = None
        self._controller: _NativeLease | None = None
        self._control = asyncio.Lock()

    def initialize(self) -> GatewayServingState:
        """Create paused coordination state once; never reopen an existing gate implicitly."""
        self.binding.admission_lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with file_write_lock(self.binding.state_path, what="CLaaS gateway admission"):
            if self.binding.state_path.exists():
                return self._read()
            return self._write(paused=True)

    def _read(self) -> GatewayServingState:
        """Fail closed if state belongs to another scope or endpoint binding."""
        state = GatewayServingState.model_validate_json(self.binding.state_path.read_bytes())
        if state.scope != self.binding.scope or state.binding_sha256 != self.binding.binding_sha256:
            raise ValueError("gateway admission state belongs to another serving binding")
        return state

    def _write(self, *, paused: bool) -> GatewayServingState:
        """Atomically persist the exact currently selected pointer and admission state."""
        registry = self.registry.read()
        state = GatewayServingState(
            scope=self.binding.scope,
            binding_sha256=self.binding.binding_sha256,
            paused=paused,
            generation=registry.generation,
            policy_revision=registry.active.policy_revision,
            model_name=serving_model_name(registry.active),
        )
        write_text_atomic(self.binding.state_path, state.model_dump_json() + "\n")
        return state

    async def pause_and_drain(self, *, timeout_seconds: float = 120) -> None:
        """Persist pause before bounded exclusive acquisition; failure leaves admission closed."""
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600:
            raise ValueError("drain timeout must be finite, positive, and at most one hour")
        async with self._control:
            if self._lease is not None:
                return
            # Serialize controllers separately: one controller's resume must not
            # erase another controller's pending pause while readers are draining.
            self._controller = await self._acquire(
                Path(str(self.binding.admission_lock_path) + ".controller"), timeout_seconds
            )
            try:
                with file_write_lock(self.binding.state_path, what="CLaaS gateway admission"):
                    self._read()
                    self._write(paused=True)
                self._lease = await self._acquire(self.binding.admission_lock_path, timeout_seconds)
            except BaseException:
                self._controller.release()
                self._controller = None
                raise

    async def _acquire(self, path: Path, timeout_seconds: float) -> _NativeLease:
        """Track a pending native lock acquisition until success or bounded cleanup."""
        acquiring = asyncio.create_task(
            asyncio.to_thread(
                exp_gateway_native.claas_acquire_exclusive, str(path), timeout_seconds
            )
        )
        try:
            return cast(_NativeLease, await asyncio.shield(acquiring))
        except asyncio.CancelledError:
            lease = cast(_NativeLease, await acquiring)
            lease.release()
            raise

    async def resume(
        self, *, expected_registry_generation: int, expected_policy_revision: str
    ) -> None:
        """Reopen only after the loaded model and durable registry have been confirmed by caller."""
        async with self._control:
            if self._lease is None:
                raise ValueError("pause and drain before resuming gateway admission")
            with (
                file_write_lock(self.binding.state_path, what="CLaaS gateway admission"),
                file_write_lock(self.binding.registry_path, what="CLaaS adapter registry"),
            ):
                self._read()
                registry = self.registry.read()
                if (
                    registry.generation != expected_registry_generation
                    or registry.active.policy_revision != expected_policy_revision
                ):
                    raise ValueError(
                        "registry changed before resume; restore the selected revision while paused"
                    )
                self._write(paused=False)
            self._lease.release()
            self._lease = None
            if self._controller is not None:
                self._controller.release()
                self._controller = None

    async def close(self) -> None:
        """Release controller ownership without reopening a paused gate."""
        async with self._control:
            if self._lease is not None:
                self._lease.release()
                self._lease = None
            if self._controller is not None:
                self._controller.release()
                self._controller = None


def load_gateway_serving_configuration(root: Path) -> GatewayServingConfiguration | None:
    """Read explicit private-serving bindings; absence leaves ordinary aliases unchanged."""
    path = root / "gateway" / "claas-serving.json"
    return (
        GatewayServingConfiguration.model_validate_json(path.read_bytes())
        if path.exists()
        else None
    )


def save_gateway_serving_binding(
    root: Path, binding: GatewayServingBinding, *, drain_timeout_seconds: float = 120
) -> GatewayServingConfiguration:
    """Publish a scoped binding, draining and retiring its previous runtime before rotation."""
    if not math.isfinite(drain_timeout_seconds) or not 0 < drain_timeout_seconds <= 3600:
        raise ValueError("drain timeout must be finite, positive, and at most one hour")
    path = root / "gateway" / "claas-serving.json"
    with file_write_lock(path, what="CLaaS gateway serving configuration"):
        previous = load_gateway_serving_configuration(root)
        previous_items = previous.bindings if previous else ()
        old = next((item for item in previous_items if item.scope == binding.scope), None)
        items = [item for item in previous_items if item.scope != binding.scope]
        items.append(binding)
        configuration = GatewayServingConfiguration(bindings=tuple(items))
        if old is None or old == binding:
            GatewayAdmissionLease(binding).initialize()
            write_text_atomic(path, configuration.model_dump_json(indent=2) + "\n")
        else:
            with _rotate_binding(old, binding, timeout_seconds=drain_timeout_seconds):
                write_text_atomic(path, configuration.model_dump_json(indent=2) + "\n")
        return configuration


@contextmanager
def _rotate_binding(
    old: GatewayServingBinding, new: GatewayServingBinding, *, timeout_seconds: float
) -> Iterator[None]:
    """Keep both controllers excluded until old traffic drains and replacement state is paused.

    Retired coordination files contain the replacement digest. Stale gateways
    and controllers therefore fail closed even after the lock paths move. A
    partially published rotation remains paused and can be retried explicitly.
    """
    previous, replacement = GatewayAdmissionLease(old), GatewayAdmissionLease(new)
    fields = ("registry_path", "state_path", "admission_lock_path")
    if any(
        getattr(old, left) == getattr(new, right)
        for left in fields
        for right in fields
        if left != right
    ):
        raise ValueError("coordination file roles cannot be swapped; choose fresh runtime paths")
    replacement.registry.read()
    if new.state_path != old.state_path and new.state_path.exists():
        _rotation_state(new)
    with ExitStack() as locks:
        lock_paths = tuple(dict.fromkeys((old.admission_lock_path, new.admission_lock_path)))
        for lock_path in lock_paths:
            lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            controller = exp_gateway_native.claas_acquire_exclusive(
                str(lock_path) + ".controller", timeout_seconds
            )
            locks.callback(controller.release)
        with file_write_lock(old.state_path, what="CLaaS retiring gateway admission"):
            state = GatewayServingState.model_validate_json(old.state_path.read_bytes())
            if state.scope != old.scope:
                raise ValueError("retiring admission state belongs to another application")
            if state.binding_sha256 == old.binding_sha256:
                previous._write(paused=True)
            elif not state.paused:
                raise ValueError("changed admission state must be paused before runtime rotation")
        if new.state_path != old.state_path and new.state_path.exists():
            with file_write_lock(new.state_path, what="CLaaS replacement gateway admission"):
                _rotation_state(new)
                replacement._write(paused=True)
        for lock_path in lock_paths:
            lease = exp_gateway_native.claas_acquire_exclusive(str(lock_path), timeout_seconds)
            locks.callback(lease.release)
        with file_write_lock(new.state_path, what="CLaaS replacement gateway admission"):
            state = replacement._write(paused=True)
        if new.state_path != old.state_path:
            with file_write_lock(old.state_path, what="CLaaS retiring gateway admission"):
                write_text_atomic(old.state_path, state.model_dump_json() + "\n")
        yield


def _rotation_state(binding: GatewayServingBinding) -> GatewayServingState:
    """Allow reuse of a retired paused file belonging to the same application only."""
    state = GatewayServingState.model_validate_json(binding.state_path.read_bytes())
    if state.scope != binding.scope or (
        state.binding_sha256 != binding.binding_sha256 and not state.paused
    ):
        raise ValueError("replacement admission state is active or belongs to another application")
    return state
