"""Shared native capture contracts and real-socket serving isolation."""

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureDeliveryLimits,
    CaptureRecord,
    CaptureRecordV1,
    CaptureSseResponse,
    read_capture_record_json,
)
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.tests.chain_authority_fixture_test import (
    chain_components,
    publish_authored_chain_fixture,
)
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)
from exp.runtime.gateway.tests.native_chat_images_test import _PNG_BASE64
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames

native = pytest.importorskip("exp_gateway_native")


@pytest.mark.parametrize("winner", ["child", "root_suffix"])
@pytest.mark.parametrize("policy", ["keep", "off", "byok", "prompt_only", "cancel"])
def test_nested_capture_separates_root_from_winner_and_does_not_recapture_replay(
    tmp_path: Path,
    winner: str,
    policy: str,
) -> None:
    """Actual HTTP traversal keeps root input identity and permission-gated winning provenance."""
    calls: list[str] = []
    provider_stopped = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        """Serve two root providers around a conditional child using one local listener."""

        def do_POST(self) -> None:  # noqa: N802 - standard HTTP handler contract.
            """Fail pre-output until the configured semantic winner is reached."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            child = self.path.startswith("/child/")
            first = self.path.startswith("/first/")
            if first or (child and winner == "root_suffix"):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"fixture unavailable"}}')
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            if policy == "cancel":
                try:
                    self.wfile.write(_content_chunk("winner"))
                    self.wfile.flush()
                    while True:
                        self.wfile.write(_content_chunk("more"))
                        self.wfile.flush()
                        time.sleep(0.02)
                except OSError:
                    provider_stopped.set()
            else:
                self.wfile.write(_content_chunk("winner") + _terminal_frames())

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{provider.server_port}"
    manager, key = _configured_pool_gateway(
        tmp_path, base_urls=(origin + "/first/v1", origin + "/child/v1")
    )
    authored = load_model_catalog(tmp_path / "models.toml")
    models = dict(authored.models)
    alpha, beta = models["alpha"], models["beta"]
    assert alpha.gateway is not None and beta.gateway is not None
    models["beta"] = beta.model_copy(
        update={"gateway": beta.gateway.model_copy(update={"exact_model_id": "child-exact"})}
    )
    models["suffix"] = alpha.model_copy(update={"connection": "suffix"})
    connections = dict(authored.connections)
    connections["suffix"] = connections[alpha.connection].model_copy(
        update={"base_url": origin + "/suffix/v1"}
    )
    authored = authored.model_copy(
        update={
            "models": models,
            "connections": connections,
            "gateway_pools": {
                "alpha": authored.gateway_pools["coding"].model_copy(
                    update={"deployment_aliases": ("alpha", "suffix")}
                )
            },
            "gateway_model_chains": {
                "model-revision-exact": GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="alpha",
                    revision="capture-chain",
                    rungs=(
                        GatewayDeploymentRung(deployment_id="alpha"),
                        GatewayModelReferenceRung(model_id="child-exact"),
                        GatewayDeploymentRung(deployment_id="suffix"),
                    ),
                )
            },
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    publish_authored_chain_fixture(tmp_path, revision_id="capture-chain", pool_id="alpha")
    components = chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    capture = CaptureController(
        collector, application_for=lambda _auth: None if policy == "off" else "app"
    )
    port, shutdown = _unused_port(), native.shutdown_handle()
    errors: list[BaseException] = []

    def serve() -> None:
        """Run the real data plane with a hosted capture sink and explicit shutdown."""
        try:
            serve_native_gateway(
                NativeControlPlane(components, capture=capture),
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surface worker failure below.
            errors.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        headers = {"authorization": f"Bearer {key}", "Idempotency-Key": "capture-once"}
        body = {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}
        expected = "child-exact" if winner == "child" else "model-revision-exact"
        if policy == "cancel":
            headers.pop("Idempotency-Key")
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json={**body, "stream": True},
                timeout=20,
            ) as response:
                assert response.status_code == 200
                assert response.headers["x-gateway-canonical-model"] == expected
                request_id = response.headers["x-request-id"]
                for line in response.iter_lines():
                    if "winner" in line:
                        break
            count = len(calls)
            assert provider_stopped.wait(3)
            collector.settle(request_id, True, True)
            assert len(calls) == count
        else:
            response = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=20,
            )
            assert response.status_code == 200, response.text
            assert response.headers["x-gateway-canonical-model"] == expected
            count = len(calls)
            collector.settle(response.headers["x-request-id"], policy != "byok", policy == "keep")
            replay = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=20,
            )
            assert replay.content == response.content and len(calls) == count
    finally:
        shutdown.request_shutdown()
        worker.join(10)
        manager.close()
        provider.shutdown()
        provider.server_close()
        thread.join(5)
    assert not errors and not worker.is_alive()
    assert collector.close(1)
    if policy in {"off", "byok"}:
        assert records == []
    else:
        assert len(records) == 1
        record = json.loads(records[0])
        assert record["request"]["model_id"] == "model-revision-exact"
        assert record.get("canonical_model_id") == expected
        assert record["deployment_id"] == ("beta" if winner == "child" else "suffix")
        assert (record["response"] is not None) is (policy in {"keep", "cancel"})
        assert (record["metrics"] is not None) is (policy in {"keep", "cancel"})
        if policy == "cancel":
            assert record["response"]["client_disconnected"]
            assert not record["metrics"]["usage_complete"]
            assert record["metrics"]["terminal_at"] is None


@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.parametrize("shape", ["large", "fragments", "signed_text", "signed_image", "oversize"])
def test_gemini_capture_only_evidence_never_retries_or_fails_visible_inference(
    tmp_path: Path,
    capture_enabled: bool,
    shape: str,
) -> None:
    """Optional telemetry limits never consume the refusal or semantic output budgets."""
    calls: list[str] = []

    class Gemini(BaseHTTPRequestHandler):
        """Emit finite valid private evidence before visible text and exact usage."""

        def do_POST(self) -> None:  # noqa: N802 - standard handler contract.
            """Serve one genuine Gemini stream with no outbound provider connection."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            if shape == "fragments":
                parts = [{"thought": True, "text": "summary"} for _ in range(257)]
            elif shape == "signed_text":
                parts = [{"text": "ok", "thoughtSignature": "s" * 65_537}]
            elif shape == "signed_image":
                parts = [
                    {
                        "inlineData": {"mimeType": "image/png", "data": _PNG_BASE64},
                        "thoughtSignature": "s" * 65_537,
                    }
                ]
            else:
                parts = [{"thought": True, "text": "x" * (65_537 if shape == "large" else 200_000)}]
            if shape != "signed_text":
                parts.append({"text": "ok"})
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                (
                    "data: "
                    + json.dumps(
                        {
                            "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
                        }
                    )
                    + "\n\n"
                ).encode()
            )

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Gemini)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, key = _configured_pool_gateway(tmp_path, provider="gemini")
    authored = load_model_catalog(tmp_path / "models.toml")
    models = dict(authored.models)
    beta = models["beta"]
    assert beta.gateway is not None
    models["beta"] = beta.model_copy(
        update={"gateway": beta.gateway.model_copy(update={"exact_model_id": "eligible-child"})}
    )
    authored = authored.model_copy(
        update={
            "models": models,
            "gateway_pools": {},
            "gateway_model_chains": {
                "model-revision-exact": GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="alpha",
                    revision="gemini-capture",
                    rungs=(
                        GatewayDeploymentRung(deployment_id="alpha"),
                        GatewayModelReferenceRung(model_id="eligible-child"),
                    ),
                )
            },
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    publish_authored_chain_fixture(tmp_path, revision_id="gemini-capture", pool_id="alpha")
    components = chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    records: list[str] = []
    configuration = CaptureConfiguration(maximum_response_bytes=100_000, settlement_required=False)
    collector = (
        native.CaptureCollector(configuration.model_dump_json(), records.append)
        if capture_enabled
        else None
    )
    capture = (
        CaptureController(collector, application_for=lambda _auth: "app") if collector else None
    )

    class Plane(NativeControlPlane):
        """Keep real Gemini preflight and alter only its test transport destination."""

        def admit(self, argument: str) -> str:
            """Redirect the already-validated Gemini provider request to loopback."""
            admitted = json.loads(super().admit(argument))
            for wire in admitted["route"]:
                assert wire["dialect"] == "gemini_generate_content"
                wire["url"] = (
                    f"http://127.0.0.1:{provider.server_port}/{wire['deployment_id']}/generate"
                )
            return json.dumps(admitted)

    port, shutdown = _unused_port(), native.shutdown_handle()
    errors: list[BaseException] = []

    def serve() -> None:
        """Run capture-on/off through the same actual native HTTP path."""
        try:
            serve_native_gateway(
                Plane(components, capture=capture),
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surface worker failure below.
            errors.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"authorization": f"Bearer {key}"},
            json={"model": "coding", "messages": [{"role": "user", "content": "hello"}]},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "ok"
        assert response.json()["usage"]["prompt_tokens"] == 3
        assert response.json()["usage"]["completion_tokens"] == 2
        if shape == "signed_image":
            assert response.json()["choices"][0]["message"]["images"]
    finally:
        shutdown.request_shutdown()
        worker.join(10)
        manager.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(5)
    assert not errors and not worker.is_alive()
    assert calls == ["/alpha/generate"]
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute(
            "SELECT deployment_id,state FROM gateway_attempts"
        ).fetchall() == [("alpha", "completed")]
    if collector:
        assert collector.close(1) and len(records) == 1
        record = CaptureRecord.model_validate_json(records[0])
        assert record.response is not None and record.metrics is not None
        assert record.gemini_thought_parts_truncated is (shape == "oversize")
        assert len(record.gemini_thought_parts) == (
            0 if shape == "oversize" else 257 if shape == "fragments" else 1
        )
    else:
        assert records == []


def _request_json() -> str:
    """Return the versioned boundary's minimum authenticated request."""
    return json.dumps(
        {
            "request_id": "request",
            "scope": {"organization_id": "org", "identity_id": "identity", "application_id": "app"},
            "protocol": "chat_completions",
            "model_id": "model",
            "context": {"schema_version": 1, "request": {"messages": []}},
        }
    )


def test_persisted_capture_schema_reader_is_explicit_strict_and_never_invents_winner() -> None:
    """StrictV1 archives migrate only through the explicit reader; new sinks consume schema2."""
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1) and len(records) == 1
    payload = json.loads(records[0])
    assert payload["schema_version"] == 2
    parsed = CaptureRecord.model_validate_json(records[0])
    assert parsed.canonical_model_id is None
    assert read_capture_record_json(records[0]) == parsed
    assert json.loads(parsed.model_dump_json()) == payload
    legacy = {
        key: value
        for key, value in payload.items()
        if key not in {"canonical_model_id", "gemini_thought_parts_truncated"}
    }
    legacy["schema_version"] = 1
    CaptureRecordV1.model_validate(legacy)
    upgraded = read_capture_record_json(json.dumps(legacy))
    assert upgraded.schema_version == 2 and upgraded.canonical_model_id is None
    assert upgraded.gemini_thought_parts_truncated is None
    assert upgraded.request.model_id == "model"
    with pytest.raises(ValueError):
        CaptureRecord.model_validate(legacy)
    with pytest.raises(ValueError):
        CaptureRecordV1.model_validate(payload)
    with pytest.raises(ValueError):
        read_capture_record_json(json.dumps({**legacy, "canonical_model_id": "forged"}))
    for version in (True, False, 1.0, 2.0, "1", "2", None, 0, 3):
        with pytest.raises(ValueError):
            read_capture_record_json(json.dumps({**payload, "schema_version": version}))
    for document in (payload, legacy):
        with pytest.raises(ValueError):
            read_capture_record_json(json.dumps({**document, "unexpected": "no"}))


def test_python_sink_runs_off_caller_thread_and_close_releases_gil() -> None:
    """A sink requiring Python can finish while the caller waits on the Rust drain."""
    records: list[str] = []
    threads: list[int] = []

    def write(record: str) -> None:
        """Observe destination execution without any provider or SQL dependency."""
        records.append(record)
        threads.append(threading.get_ident())

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), write)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert threads and threads[0] != threading.get_ident()
    assert CaptureRecord.model_validate_json(records[0]).request.scope.identity_id == "identity"
    assert collector.counts() == (0, 0, 1, 0, 0, 0)


def test_close_timeout_preserves_accepted_content_for_later_host_settlement() -> None:
    """Timing out the Python boundary cannot purge accepted but undecided content."""
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert collector.begin(_request_json())
    assert not collector.close(0)
    assert collector.counts() == (0, 0, 0, 0, 0, 0)
    collector.settle("request", True, False)
    assert collector.close(1)
    assert len(records) == 1
    assert CaptureRecord.model_validate_json(records[0]).request.request_id == "request"
    assert collector.counts() == (0, 0, 1, 0, 0, 0)


def test_python_sink_failure_never_logs_exception_content(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Database exception strings may carry private parameters and must be discarded."""

    def fail(_record: str) -> None:
        """Simulate a storage rejection containing sensitive context."""
        raise RuntimeError("private SQL parameter that must not be logged")

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), fail)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert collector.counts() == (0, 0, 0, 1, 0, 0)
    assert "private SQL" not in "".join(capfd.readouterr())


def test_python_and_rust_configuration_fail_closed() -> None:
    """Both entry points reject invalid bounds and unknown configuration."""
    with pytest.raises(ValueError):
        CaptureDeliveryLimits(maximum_bytes=1)
    with pytest.raises(ValueError):
        CaptureConfiguration(maximum_pending_bytes=1)
    with pytest.raises(ValueError):
        native.CaptureCollector('{"unknown": true}', lambda _: None)


def test_accepted_routing_failure_keeps_effective_prompt_without_inventing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture begins after acceptance but before a route can fail without dispatch."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:1/v1")
    components = load_gateway_components(tmp_path)
    records: list[str] = []
    authorized: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)

    def application_for(authorization: AuthorizationSnapshot) -> str:
        """Record the authenticated request for an explicit hosted terminal verdict."""
        authorized.append(authorization.request_id)
        return "application"

    capture = CaptureController(collector, application_for=application_for)
    control = NativeControlPlane(components, capture=capture)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        """Reject route construction without making a provider call."""
        raise GatewayRoutingError("unavailable route")

    monkeypatch.setattr(control, "_resolve_route", unavailable)
    try:
        with pytest.raises(NativeBridgeError):
            control.admit(
                json.dumps(
                    {
                        "raw_key": raw_key,
                        "body": json.dumps(
                            {
                                "model": "coding",
                                "messages": [{"role": "user", "content": "retained task"}],
                            }
                        ),
                    }
                )
            )
        assert len(authorized) == 1
        collector.settle(authorized[0], True, False)
        assert collector.close(1)
    finally:
        collector.close(1)
        components.write_ledger.close()
    parsed = CaptureRecord.model_validate_json(records[0])
    assert parsed.request.model_id is None
    assert parsed.response is None
    assert "retained task" in records[0]


@pytest.mark.parametrize("policy", ["local", "hosted", "hosted-late", "off", "broken", "full"])
def test_real_http_surfaces_capture_or_fail_before_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Collect Chat, Responses and Messages JSON/SSE through native HTTP, not a fixture tap."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _LoopbackProvider.calls = 0
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    records: list[str] = []
    configuration = CaptureConfiguration(settlement_required=policy in {"hosted", "hosted-late"})
    collector = native.CaptureCollector(configuration.model_dump_json(), records.append)
    if policy == "full":
        assert collector.close(1)

    def application_for(authorization: AuthorizationSnapshot) -> str | None:
        """Exercise host policy separately from content assembly and persistence."""
        assert authorization.identity_id == "default"
        if policy == "broken":
            raise RuntimeError("private policy details")
        return None if policy == "off" else "application"

    capture = CaptureController(collector, application_for=application_for)
    port = _unused_port()
    shutdown = native.shutdown_handle()
    failures: list[BaseException] = []

    def run() -> None:
        """Serve the real data plane and preserve startup failures for assertions."""
        try:
            serve_native_gateway(
                NativeControlPlane(components, capture=capture),
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surfaced after bounded shutdown.
            failures.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    awaiting_settlement: list[str] = []
    try:
        _wait_ready(port, worker)
        for surface in ("chat/completions", "responses", "messages"):
            for stream in (False, True):
                payload: dict[str, object] = {"model": "coding", "stream": stream}
                if surface == "responses":
                    payload["input"] = "capture task"
                else:
                    payload["messages"] = [{"role": "user", "content": "capture task"}]
                if surface == "messages":
                    payload["max_tokens"] = 128
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/{surface}",
                    headers={
                        "authorization": f"Bearer {raw_key}",
                        "X-Session-Id": "real-harness-session",
                        "X-Other-Private-Header": "do-not-capture-me",
                    },
                    json=payload,
                    timeout=10,
                )
                if policy in {"broken", "full"}:
                    assert response.status_code == 503
                    assert (
                        "overloaded_error" if surface == "messages" else "capture_unavailable"
                    ) in response.text
                    assert "private policy" not in response.text
                    continue
                assert response.status_code == 200, response.text
                assert "hello " in response.text and "world" in response.text
                if policy == "hosted":
                    collector.settle(response.headers["x-request-id"], True, True)
                elif policy == "hosted-late":
                    awaiting_settlement.append(response.headers["x-request-id"])
        if policy == "hosted-late":
            assert not collector.close(0)
            assert not records
            assert collector.counts()[5] == 0
            for request_id in awaiting_settlement:
                collector.settle(request_id, True, True)
    finally:
        shutdown.request_shutdown()
        worker.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not worker.is_alive()
    assert collector.close(1)
    assert _LoopbackProvider.calls == (0 if policy in {"broken", "full"} else 6)
    if policy in {"off", "broken", "full"}:
        assert records == []
        return
    parsed = [CaptureRecord.model_validate_json(value) for value in records]
    completed = [record for record in parsed if record.response is not None]
    assert len(completed) == 6
    assert sum(record.response.kind == "json" for record in completed if record.response) == 3
    assert all(
        not record.response.truncated and not record.response.client_disconnected
        for record in completed
        if isinstance(record.response, CaptureSseResponse)
    )
    assert {record.request.protocol for record in completed} == {
        "chat_completions",
        "responses",
        "messages",
    }
    assert all(record.request.scope.identity_id == "default" for record in completed)
    assert all(record.request.model_id is not None for record in completed)
    for record in completed:
        assert record.request.context["session_id"] == "real-harness-session"
        assert record.provider_reasoning is None
        assert record.metrics is not None
        assert record.metrics.terminal_at is not None
        assert record.metrics.terminal_at >= record.metrics.started_at
        assert record.metrics.first_token_at is not None
        assert record.metrics.first_token_at >= record.metrics.started_at
        assert record.metrics.duration_ms is not None and record.metrics.duration_ms > 0
        assert record.metrics.usage_complete
        assert record.metrics.usage is not None
        assert record.metrics.usage.input_tokens is not None
        assert record.metrics.usage.output_tokens is not None
    assert "provider-secret" not in "".join(records)
    assert raw_key not in "".join(records)
    assert "do-not-capture-me" not in "".join(records)
