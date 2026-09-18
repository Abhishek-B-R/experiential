"""Real Rust loopback execution preserves cross-model identity and request accounting."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.catalog_authority import snapshot_current_catalog
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.tests.native_waterfall_test import (
    _DRIVER_SOURCE,
    _attempt_rows,
    _content_chunk,
    _PrimaryUpstream,
    _SecondaryUpstream,
    _ServingEngine,
    _sse_frame,
    _terminal_frames,
)


class _StageSecondaryUpstream(_SecondaryUpstream):
    """Capture the real child wire while retaining the ordinary loopback response."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Require JSON mode on the child wire and record it in the scripted JSON reply."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if "provider" in payload:
            text = json.dumps(
                {
                    "from-secondary": True,
                    "provider": payload["provider"],
                    "metadata": self.headers.get("X-OpenRouter-Metadata"),
                }
            )
        elif "response_format" in payload:
            text = json.dumps(
                {
                    "from-secondary": True,
                    "response_format": payload["response_format"],
                    "verbosity_sent": "verbosity" in payload,
                }
            )
        else:
            text = "from-secondary"
        try:
            self.wfile.write(
                _sse_frame(
                    {
                        "provider": "fixture-upstream",
                        "choices": [
                            {"index": 0, "delta": {"content": text}, "finish_reason": None}
                        ],
                    }
                )
                if "provider" in payload
                else _content_chunk(text)
            )
            self.wfile.write(_terminal_frames(prompt_tokens=3, completion_tokens=1))
        except OSError:
            return


def test_installed_native_exports_ordered_stage_contract() -> None:
    """Read the compiled extension's feature marker, independently of package version labels."""
    marker = getattr(exp_gateway_native, "MODEL_STAGE_CONTRACT_VERSION", None)
    assert type(marker) is int
    assert marker == 1


@pytest.fixture(name="engine")
def stage_engine(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[_ServingEngine]:
    """Serve a genuinely different fallback model, with no false pool equivalence."""
    primary = ThreadingHTTPServer(("127.0.0.1", 0), _PrimaryUpstream)
    secondary = ThreadingHTTPServer(("127.0.0.1", 0), _StageSecondaryUpstream)
    for server in (primary, secondary):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    manager, raw_key = _configured_pool_gateway(
        tmp_path,
        base_urls=(
            f"http://127.0.0.1:{primary.server_port}/v1",
            f"http://127.0.0.1:{secondary.server_port}/v1",
        ),
    )
    catalog = load_model_catalog(tmp_path / "models.toml")
    models = dict(catalog.models)
    beta = models["beta"]
    assert beta.gateway is not None
    capabilities = beta.gateway.capabilities
    zdr_child = getattr(request, "param", None) == "zdr-child"
    if hasattr(request, "param") and not zdr_child:
        capabilities = capabilities.model_copy(update={"failover_only_on": (request.param,)})
    models["beta"] = beta.model_copy(
        update={
            "gateway": beta.gateway.model_copy(
                update={"exact_model_id": "secondary-exact", "capabilities": capabilities}
            )
        }
    )
    chain = GatewayModelChain(
        model_id="model-revision-exact",
        pool_id="alpha",
        revision="chain-one",
        rungs=(
            GatewayDeploymentRung(deployment_id="alpha"),
            GatewayModelReferenceRung(model_id="secondary-exact"),
        ),
    )
    connections = dict(catalog.connections)
    if zdr_child:
        connections[beta.connection] = connections[beta.connection].model_copy(
            update={"provider": "openrouter", "base_url": None}
        )
    authored = catalog.model_copy(
        update={
            "models": models,
            "connections": connections,
            "gateway_pools": {},
            "gateway_model_chains": {"model-revision-exact": chain},
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    _, normalized, snapshot = snapshot_current_catalog(tmp_path)
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-stage",
        pool_id="alpha",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    driver = tmp_path / "driver.py"
    source = _DRIVER_SOURCE
    if zdr_child:
        source = source.replace(
            "    components = load_gateway_components(",
            "    from exp.runtime.models import registry\n"
            "    factory, _origin = registry._HTTP_PROVIDERS['openrouter']\n"
            "    registry._HTTP_PROVIDERS['openrouter'] = (factory, config['child_origin'])\n"
            "    components = load_gateway_components(",
        )
        source = source.replace(
            "    control_plane = NativeControlPlane(",
            "    from exp.runtime.gateway.routing import CatalogRouteResolver\n"
            "    original = CatalogRouteResolver.resolve_direct\n"
            "    def constrained(self, authorization):\n"
            '        """Apply only the fixture host\'s frozen child ZDR policy."""\n'
            "        route = original(self, authorization)\n"
            "        return route.model_copy(update={'snapshot': route.snapshot.model_copy(\n"
            "            update={'zdr_constrained_deployment_ids': ('beta',)})})\n"
            "    CatalogRouteResolver.resolve_direct = constrained\n"
            "    control_plane = NativeControlPlane(",
        )
    driver.write_text(source + "\n")
    log_path = tmp_path / "driver.log"
    environment = {**os.environ, "TEST_PROVIDER_KEY": "loopback-secret"}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(driver),
                json.dumps(
                    {
                        "root": str(tmp_path),
                        "request_timeout_seconds": 10,
                        "child_origin": f"http://127.0.0.1:{secondary.server_port}/v1",
                    }
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=log,
            env=environment,
            text=True,
        )  # noqa: S603 - generated test driver.
        ports: list[int] = []

        def collect() -> None:
            """Read port announcements without blocking the readiness deadline."""
            assert process.stdout is not None
            for line in process.stdout:
                ports.append(int(json.loads(line)["port"]))

        threading.Thread(target=collect, daemon=True).start()
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, log_path.read_text()
                if ports:
                    try:
                        response = httpx.get(
                            f"http://127.0.0.1:{ports[-1]}/health/live", timeout=0.5
                        )
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                time.sleep(0.05)
            yield _ServingEngine(ports[-1], raw_key, manager.database_path)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)
            for server in (primary, secondary):
                server.shutdown()
                server.server_close()
            assert process.returncode == 0, log_path.read_text()


@pytest.mark.parametrize("engine", ["zdr-child"], indirect=True)
def test_actual_child_zdr_constraint_and_upstream_identity_survive_fallback(
    engine: _ServingEngine,
) -> None:
    """The actual child wire, committed identity and settled upstream retain host ZDR policy."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": "always-500"}],
            "provider": {"zdr": False, "data_collection": "allow", "order": ["fixture-upstream"]},
        },
        timeout=30,
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert response.headers["x-gateway-zdr-constrained"] == "true"
    assert response.headers["x-gateway-deployment"] == "beta"
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert content == {
        "from-secondary": True,
        "metadata": "enabled",
        "provider": {"zdr": True, "data_collection": "deny", "order": ["fixture-upstream"]},
    }
    with sqlite3.connect(engine.database_path) as db:
        assert db.execute(
            "SELECT exact_model_id,upstream_provider,state FROM gateway_attempts "
            "ORDER BY attempt_ordinal"
        ).fetchall() == [
            ("model-revision-exact", None, "failed"),
            ("model-revision-exact", None, "failed"),
            ("secondary-exact", "fixture-upstream", "completed"),
        ]


@pytest.mark.parametrize(
    "engine,allowed", [("provider_internal", True), ("refusal", False)], indirect=["engine"]
)
def test_native_child_stage_honors_conditional_failover(
    engine: _ServingEngine, allowed: bool
) -> None:
    """The actual native child wire is reachable only for its authored upstream failure token."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "messages": [{"role": "user", "content": "always-500"}]},
        timeout=30,
    )
    with sqlite3.connect(engine.database_path) as db:
        attempts = db.execute(
            "SELECT deployment_id,state,fallback_reason FROM gateway_attempts "
            "ORDER BY attempt_ordinal"
        ).fetchall()
    assert attempts[:2] == [("alpha", "failed", None), ("alpha", "failed", None)]
    assert response.status_code == (200 if allowed else 502)
    if allowed:
        assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
        assert attempts[2:] == [("beta", "completed", "failover_only_on:provider_internal")]
    else:
        assert response.status_code == 502
        assert len(attempts) == 2


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_actual_stage_header_alias_and_single_request_ledger(
    engine: _ServingEngine, surface: str, stream: bool
) -> None:
    """One request redials root once then commits different exact model on every surface."""
    payload: JsonObject = {"model": "coding", "stream": stream}
    if surface == "chat/completions":
        payload["response_format"] = {"type": "json_object"}
        payload["verbosity"] = "low"
    if surface == "responses":
        payload["input"] = "always-500"
    else:
        payload["messages"] = [{"role": "user", "content": "always-500"}]
        if surface == "messages":
            payload["max_tokens"] = 32
    response = httpx.post(
        f"{engine.base}/v1/{surface}",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=payload,
        timeout=30,
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-canonical-model"] == "secondary-exact"
    assert response.headers["x-gateway-alias"] == "coding"
    assert "from-secondary" in response.text
    if surface == "chat/completions":
        if stream:
            chunks = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            text = "".join(
                choice.get("delta", {}).get("content", "")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
            assert any(
                chunk.get("x-experiential-ignored-parameters") == ["verbosity"] for chunk in chunks
            )
        else:
            text = response.json()["choices"][0]["message"]["content"]
            assert response.json()["x-experiential-ignored-parameters"] == ["verbosity"]
        assert json.loads(text) == {
            "from-secondary": True,
            "response_format": {"type": "json_object"},
            "verbosity_sent": False,
        }
    if not stream:
        assert response.json()["model"] == "coding"
    request_id = response.headers["x-request-id"]
    assert _attempt_rows(engine, request_id) == [
        (0, 0, "failed"),
        (1, 0, "failed"),
        (2, 1, "completed"),
    ]
    with sqlite3.connect(engine.database_path) as db:
        rows = db.execute(
            "SELECT exact_model_id,pool_id FROM gateway_attempts "
            "WHERE request_id=? ORDER BY attempt_ordinal",
            (request_id,),
        ).fetchall()
        assert rows == [
            ("model-revision-exact", "alpha"),
            ("model-revision-exact", "alpha"),
            ("secondary-exact", "beta"),
        ]
        assert db.execute("SELECT count(*) FROM gateway_requests").fetchone() == (1,)
        assert db.execute(
            "SELECT count(*) FROM gateway_attempts WHERE state IN ('dispatched','running')"
        ).fetchone() == (0,)
