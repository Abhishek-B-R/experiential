"""Real native traffic uses promoted private routes and cross-process admission locks."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.claas import ClaasScope
from exp.common.models import GatewayDeploymentCapabilities, GatewayTokenPrices, ModelCapabilities
from exp.runtime.claas.registry import AdapterRegistry, ServingRevision
from exp.runtime.claas.serving.vllm import serving_model_name
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.claas import serving
from exp.runtime.gateway.claas.serving import (
    GatewayAdmissionLease,
    GatewayServingBinding,
    GatewayServingConfiguration,
    GatewayServingState,
    load_gateway_serving_configuration,
    save_gateway_serving_binding,
)
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import _configure_gateway, _unused_port, _wait_ready


class _Provider(BaseHTTPRequestHandler):
    """Record the real wire model and optionally hold one upstream request open."""

    models: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    hold = False
    hold_after_first = False

    def do_POST(self) -> None:  # noqa: N802
        """Serve finite SSE after recording the actual requested adapter route."""
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).models.append(payload["model"])
        if type(self).hold:
            type(self).entered.set()
            assert type(self).release.wait(timeout=10)
        body = (
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
            b"data: [DONE]\n\n"
        )
        prefix = (
            b'data: {"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}\n\n'
            if type(self).hold_after_first
            else b""
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(prefix) + len(body)))
        self.end_headers()
        if prefix:
            self.wfile.write(prefix)
            self.wfile.flush()
            type(self).entered.set()
            assert type(self).release.wait(timeout=10)
        self.wfile.write(body)

    def log_message(self, format: str, *args: str) -> None:
        """Keep request content out of test logs."""


def _binding(root: Path, url: str) -> tuple[GatewayServingBinding, ServingRevision]:
    """Initialize a scope and immutable base matching the test gateway catalog."""
    scope = ClaasScope(user_id="default", application_id="claims")
    base = ServingRevision(
        scope=scope,
        policy_revision="base",
        model_id="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        tokenizer_id="tokenizer",
        tokenizer_revision="b" * 40,
    )
    binding = GatewayServingBinding(
        scope=scope,
        alias="coding",
        private_base_url=url,
        registry_path=root / "active.json",
        state_path=root / "admission.json",
        admission_lock_path=root / "admission.lock",
    )
    AdapterRegistry(binding.registry_path, scope).initialize(base)
    return binding, base


def test_native_gateway_drains_and_dispatches_promoted_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual socket traffic stops before sleep and changes adapter only after durable resume."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "fixture-only-key")
    _Provider.models = []
    _Provider.hold = False
    _Provider.hold_after_first = False
    _Provider.entered = threading.Event()
    _Provider.release = threading.Event()
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    url = f"http://127.0.0.1:{provider.server_port}/v1"
    manager, raw_key = _configure_gateway(tmp_path, base_url=url)
    normalized, snapshot, _ = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="Qwen/Qwen3.5-4B",
        exact_model_id="qwen3.5-4b",
        revision="a" * 40,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000,
            output_nano_usd_per_million_tokens=2_000_000,
        ),
        pricing_source="loopback-test",
        replace=True,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="qwen-provider-revision",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    binding, base = _binding(tmp_path, url.removesuffix("/v1"))
    config = save_gateway_serving_binding(tmp_path, binding)
    components = load_gateway_components(tmp_path)
    port = _unused_port()
    shutdown = exp_gateway_native.shutdown_handle()
    failures: list[BaseException] = []

    def run_gateway() -> None:
        """Serve native requests with a real admission registry binding."""
        try:
            serve_native_gateway(
                NativeControlPlane(components),
                host="127.0.0.1",
                port=port,
                serving=config,
                shutdown=shutdown,
            )
        except BaseException as exc:  # noqa: BLE001 - propagate thread failures below.
            failures.append(exc)

    gateway_thread = threading.Thread(target=run_gateway, daemon=True)
    gateway_thread.start()

    def request(surface: str = "chat/completions", replay_key: str = "") -> httpx.Response:
        """Send a valid bounded Chat or Responses request through native admission."""
        body = {
            "model": "coding",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        }
        if surface == "responses":
            body = {"model": "coding", "max_output_tokens": 64, "input": "hello"}
        headers = {"Authorization": f"Bearer {raw_key}"}
        if replay_key:
            headers["Idempotency-Key"] = replay_key
        return httpx.post(
            f"http://127.0.0.1:{port}/v1/{surface}",
            json=body,
            headers=headers,
            timeout=10,
        )

    async def cycle() -> None:
        """Exercise pause, outstanding request drain, promotion, and subsequent traffic."""
        lease = GatewayAdmissionLease(binding)
        assert (await asyncio.to_thread(request)).status_code == 503
        await lease.pause_and_drain()
        await lease.resume(expected_registry_generation=0, expected_policy_revision="base")
        served = await asyncio.to_thread(request, "chat/completions", "cached")
        assert served.status_code == 200
        assert served.headers["x-gateway-canonical-model"] == "qwen3.5-4b"
        assert _Provider.models == [serving_model_name(base)]
        _Provider.hold = True
        active = asyncio.create_task(asyncio.to_thread(request))
        assert await asyncio.to_thread(_Provider.entered.wait, 5)
        draining = asyncio.create_task(lease.pause_and_drain())
        for _ in range(100):
            if GatewayServingState.model_validate_json(binding.state_path.read_bytes()).paused:
                break
            await asyncio.sleep(0.01)
        assert not draining.done()
        rejected = await asyncio.to_thread(request)
        assert rejected.status_code == 503
        assert rejected.headers["retry-after"] == "1"
        assert (await asyncio.to_thread(request, "chat/completions", "cached")).status_code == 503
        _Provider.release.set()
        assert (await active).status_code == 200
        await draining
        _Provider.hold = False
        await lease.resume(expected_registry_generation=0, expected_policy_revision="base")
        _Provider.entered.clear()
        _Provider.release.clear()
        _Provider.hold_after_first = True

        def disconnect() -> None:
            """Drop a live public stream while its provider still has outstanding generation."""
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {raw_key}"},
                json={
                    "model": "coding",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "abort"}],
                },
                timeout=10,
            ) as response:
                assert response.status_code == 200
                assert next(response.iter_lines()).startswith("data:")

        await asyncio.to_thread(disconnect)
        assert await asyncio.to_thread(_Provider.entered.wait, 5)
        draining = asyncio.create_task(lease.pause_and_drain())
        await asyncio.sleep(0.1)
        assert not draining.done()
        _Provider.release.set()
        await draining
        _Provider.hold_after_first = False
        candidate = base.model_copy(
            update={
                "policy_revision": "trained",
                "adapter_directory": str(tmp_path / "trained" / "student"),
                "manifest_sha256": "c" * 64,
            }
        )
        registry = AdapterRegistry(binding.registry_path, binding.scope)
        state = registry.activate(candidate, expected_generation=0)
        with pytest.raises(ValueError, match="registry changed"):
            await lease.resume(expected_registry_generation=0, expected_policy_revision="base")
        assert (await asyncio.to_thread(request)).status_code == 503
        await lease.resume(
            expected_registry_generation=state.generation, expected_policy_revision="trained"
        )
        await lease.close()
        assert (await asyncio.to_thread(request, "responses")).status_code == 200
        assert _Provider.models[-1] == serving_model_name(candidate)
        registry.rollback(expected_generation=1)
        assert (await asyncio.to_thread(request)).status_code == 503

    try:
        _wait_ready(port, gateway_thread)
        asyncio.run(cycle())
    finally:
        _Provider.release.set()
        shutdown.request_shutdown()
        gateway_thread.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not gateway_thread.is_alive()


def test_failed_drain_leaves_paused_and_controller_can_recover(tmp_path: Path) -> None:
    """Native lock contention and explicit cleanup never silently reopen the gate."""
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    lease = GatewayAdmissionLease(binding)
    lease.initialize()
    held = exp_gateway_native.claas_acquire_exclusive(str(binding.admission_lock_path), 0.1)

    async def run() -> None:
        """Force a drain timeout, release the old holder, then explicitly resume."""
        with pytest.raises(RuntimeError, match="lock unavailable"):
            await lease.pause_and_drain(timeout_seconds=0.02)
        assert GatewayServingState.model_validate_json(binding.state_path.read_bytes()).paused
        held.release()
        await lease.close()
        await lease.pause_and_drain()
        await lease.resume(expected_registry_generation=0, expected_policy_revision="base")
        await lease.close()
        assert not GatewayServingState.model_validate_json(binding.state_path.read_bytes()).paused

    asyncio.run(run())


@pytest.mark.parametrize("release_before_timeout", [False, True])
def test_cancelled_native_drain_preserves_cancellation_and_releases_ownership(
    tmp_path: Path, release_before_timeout: bool
) -> None:
    """Cancellation survives native timeout or late success without leaking either lock."""
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    lease = GatewayAdmissionLease(binding)
    lease.initialize()
    held = exp_gateway_native.claas_acquire_exclusive(str(binding.admission_lock_path), 0.1)

    async def run() -> None:
        """Cancel during actual lock contention, then prove another controller can recover."""
        draining = asyncio.create_task(lease.pause_and_drain(timeout_seconds=0.2))
        for _ in range(100):
            if lease._controller is not None:
                break
            await asyncio.sleep(0.001)
        assert lease._controller is not None
        draining.cancel()
        await asyncio.sleep(0.01)
        draining.cancel()
        if release_before_timeout:
            held.release()
        with pytest.raises(asyncio.CancelledError):
            await draining
        assert GatewayServingState.model_validate_json(binding.state_path.read_bytes()).paused
        held.release()
        replacement = GatewayAdmissionLease(binding)
        await replacement.pause_and_drain(timeout_seconds=0.2)
        await replacement.resume(expected_registry_generation=0, expected_policy_revision="base")
        await replacement.close()

    try:
        asyncio.run(run())
    finally:
        held.release()


@pytest.mark.parametrize(
    "surface", ["chat/completions", "responses", "messages", "messages/count_tokens"]
)
def test_revoked_alias_grant_cannot_observe_serving_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """Real native requests return identical authorization errors while ready or paused."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "fixture-only-key")
    manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:8001/v1")
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    configuration = save_gateway_serving_binding(tmp_path, binding)
    components = load_gateway_components(tmp_path)
    port = _unused_port()
    shutdown = exp_gateway_native.shutdown_handle()
    failures: list[BaseException] = []

    def run_gateway() -> None:
        """Serve the actual Rust authorization/readiness path without an upstream provider."""
        try:
            serve_native_gateway(
                NativeControlPlane(components),
                host="127.0.0.1",
                port=port,
                serving=configuration,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - propagate thread failures below.
            failures.append(error)

    gateway_thread = threading.Thread(target=run_gateway, daemon=True)
    gateway_thread.start()

    def request(replay: bool) -> httpx.Response:
        """Exercise both new and replay-keyed requests through the chosen API surface."""
        body = {
            "model": "coding",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        }
        if surface == "responses":
            body = {"model": "coding", "max_output_tokens": 64, "input": "hello"}
        headers = {"Authorization": f"Bearer {raw_key}"}
        if surface.startswith("messages"):
            headers["x-api-key"] = raw_key
            headers["anthropic-version"] = "2023-06-01"
        if replay:
            headers["Idempotency-Key"] = "revoked-grant"
        return httpx.post(
            f"http://127.0.0.1:{port}/v1/{surface}", json=body, headers=headers, timeout=5
        )

    async def compare() -> None:
        """Observe the same no-grant response before and after the controller resumes."""
        paused = [await asyncio.to_thread(request, replay) for replay in (False, True)]
        lease = GatewayAdmissionLease(binding)
        await lease.pause_and_drain()
        await lease.resume(expected_registry_generation=0, expected_policy_revision="base")
        await lease.close()
        ready = [await asyncio.to_thread(request, replay) for replay in (False, True)]
        for before, after in zip(paused, ready, strict=True):
            expected_status = 404 if surface == "messages/count_tokens" else 403
            assert before.status_code == after.status_code == expected_status
            assert before.json() == after.json()
            assert "paused" not in before.text

    try:
        _wait_ready(port, gateway_thread)
        assert manager.remove_grant(identity_id="default", alias_id="coding")
        asyncio.run(compare())
    finally:
        shutdown.request_shutdown()
        gateway_thread.join(timeout=10)
        components.write_ledger.close()
    assert not failures
    assert not gateway_thread.is_alive()


@pytest.mark.parametrize(
    "url",
    [
        "https://provider.example",
        "http://provider.example",
        "https://127.0.0.1:8001",
        "http://user:secret@localhost:8001",
        "http://@localhost:8001",
        "http://localhost:8001?secret=value",
        "http://localhost:8001?",
        "http://localhost:8001#fragment",
        "http://localhost:8001#",
        "http://localhost:8001/v1",
        "http://localhost:8001//",
        "http://localhost:8001/other/..",
        "http://127.1:8001",
        "http://localhost.example:8001",
        "http://localhost:0",
        "http://localhost:65536",
        " http://localhost:8001",
    ],
)
def test_private_origin_rejected_before_persistence_and_on_load(tmp_path: Path, url: str) -> None:
    """Both public binding construction and persisted configuration loading reject unsafe URLs."""
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    payload = binding.model_dump(mode="json") | {"private_base_url": url}
    path = tmp_path / "gateway" / "claas-serving.json"
    with pytest.raises(ValueError):
        save_gateway_serving_binding(tmp_path, GatewayServingBinding.model_validate(payload))
    assert not path.exists()
    assert not binding.state_path.exists()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"bindings": [payload]}))
    with pytest.raises(ValueError):
        load_gateway_serving_configuration(tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://localhost",
        "http://[::1]",
        "http://127.0.0.1:8001/",
        "http://localhost:8001/",
        "http://[::1]:8001/",
    ],
)
def test_loopback_origins_persist_and_reload(tmp_path: Path, url: str) -> None:
    """Each supported loopback spelling accepts root paths and explicit or default ports."""
    binding, _ = _binding(tmp_path, url)
    saved = save_gateway_serving_binding(tmp_path, binding)
    assert load_gateway_serving_configuration(tmp_path) == saved


@pytest.mark.parametrize(
    "other_url", ["http://localhost:8001", "http://127.0.0.1:8001", "http://[::1]:8001/"]
)
def test_configuration_rejects_shared_private_server(tmp_path: Path, other_url: str) -> None:
    """Separate application locks cannot control one private inference server independently."""
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    other = binding.model_copy(
        update={
            "scope": ClaasScope(user_id="other", application_id="other"),
            "alias": "other",
            "private_base_url": other_url,
            "state_path": tmp_path / "other-state.json",
            "admission_lock_path": tmp_path / "other.lock",
        }
    )
    with pytest.raises(ValueError, match="one application per private"):
        GatewayServingConfiguration(bindings=(binding, other))


@pytest.mark.parametrize("rotation", ["alias", "origin", "paths"])
def test_binding_rotation_drains_replaces_scope_and_retires_old_controller(
    tmp_path: Path, rotation: str
) -> None:
    """An application can rotate runtime without deleting files or reviving stale gateways."""
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    save_gateway_serving_binding(tmp_path, binding)
    previous = GatewayAdmissionLease(binding)

    async def activate(lease: GatewayAdmissionLease) -> None:
        """Explicitly open one initialized base pointer."""
        await lease.pause_and_drain()
        await lease.resume(expected_registry_generation=0, expected_policy_revision="base")

    asyncio.run(activate(previous))
    if rotation == "alias":
        replacement = binding.model_copy(update={"alias": "new-alias"})
    elif rotation == "origin":
        replacement = binding.model_copy(update={"private_base_url": "http://127.0.0.1:8002"})
    else:
        replacement = binding.model_copy(
            update={
                "state_path": tmp_path / "new-state.json",
                "admission_lock_path": tmp_path / "new-admission.lock",
            }
        )
    updated = save_gateway_serving_binding(tmp_path, replacement)
    assert updated.bindings == (replacement,)
    assert load_gateway_serving_configuration(tmp_path) == updated
    with pytest.raises(ValueError, match="another serving binding"):
        previous.initialize()
    selected = GatewayAdmissionLease(replacement)
    assert selected.initialize().paused
    asyncio.run(activate(selected))
    assert not selected.initialize().paused
    with pytest.raises(ValueError, match="another serving binding"):
        asyncio.run(previous.pause_and_drain())
    reverted = save_gateway_serving_binding(tmp_path, binding)
    assert reverted.bindings == (binding,)
    assert GatewayAdmissionLease(binding).initialize().paused
    with pytest.raises(ValueError, match="another serving binding"):
        selected.initialize()


def test_rotation_timeout_preserves_config_and_leaves_old_admission_paused(tmp_path: Path) -> None:
    """A busy old runtime cannot be replaced or reopened by a timed-out rotation."""
    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    original = save_gateway_serving_binding(tmp_path, binding)
    active = GatewayAdmissionLease(binding)

    async def activate() -> None:
        """Ensure rotation must actively pause the previously serving runtime."""
        await active.pause_and_drain()
        await active.resume(expected_registry_generation=0, expected_policy_revision="base")

    asyncio.run(activate())
    assert not active.initialize().paused
    held = exp_gateway_native.claas_acquire_exclusive(str(binding.admission_lock_path), 0.1)
    replacement = binding.model_copy(update={"alias": "new-alias"})
    try:
        with pytest.raises(RuntimeError, match="lock unavailable"):
            save_gateway_serving_binding(tmp_path, replacement, drain_timeout_seconds=0.02)
        assert load_gateway_serving_configuration(tmp_path) == original
        assert GatewayAdmissionLease(binding).initialize().paused
    finally:
        held.release()
    assert save_gateway_serving_binding(tmp_path, replacement).bindings == (replacement,)


def test_rotation_cannot_take_another_applications_alias(tmp_path: Path) -> None:
    """Replacing by application scope never deletes another scope with the requested alias."""
    binding, base = _binding(tmp_path, "http://127.0.0.1:8001")
    save_gateway_serving_binding(tmp_path, binding)
    scope = binding.scope.model_copy(update={"application_id": "another-app"})
    other = binding.model_copy(
        update={
            "scope": scope,
            "alias": "other-alias",
            "private_base_url": "http://127.0.0.1:8002",
            "registry_path": tmp_path / "other-registry.json",
            "state_path": tmp_path / "other-state.json",
            "admission_lock_path": tmp_path / "other-admission.lock",
        }
    )
    AdapterRegistry(other.registry_path, scope).initialize(base.model_copy(update={"scope": scope}))
    original = save_gateway_serving_binding(tmp_path, other)
    with pytest.raises(ValueError, match="unique"):
        save_gateway_serving_binding(tmp_path, binding.model_copy(update={"alias": other.alias}))
    assert load_gateway_serving_configuration(tmp_path) == original


@pytest.mark.parametrize("move_paths", [False, True])
def test_failed_rotation_publication_can_retry_without_reopening_old_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, move_paths: bool
) -> None:
    """Partial writes leave both generations closed and allow explicit completion on retry."""

    binding, _ = _binding(tmp_path, "http://127.0.0.1:8001")
    original = save_gateway_serving_binding(tmp_path, binding)
    replacement = binding.model_copy(update={"alias": "new-alias"})
    if move_paths:
        replacement = replacement.model_copy(
            update={
                "state_path": tmp_path / "new-state.json",
                "admission_lock_path": tmp_path / "new-admission.lock",
            }
        )
    configuration_path = tmp_path / "gateway" / "claas-serving.json"
    write = serving.write_text_atomic

    def fail_configuration(path: Path, text: str) -> None:
        """Simulate a durable configuration publication failure after readiness was retired."""
        if path == configuration_path:
            raise OSError("fixture storage failure")
        write(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(serving, "write_text_atomic", fail_configuration)
        with pytest.raises(OSError, match="fixture storage failure"):
            save_gateway_serving_binding(tmp_path, replacement)
    assert load_gateway_serving_configuration(tmp_path) == original
    with pytest.raises(ValueError, match="another serving binding"):
        GatewayAdmissionLease(binding).initialize()
    assert GatewayAdmissionLease(replacement).initialize().paused
    assert save_gateway_serving_binding(tmp_path, replacement).bindings == (replacement,)
