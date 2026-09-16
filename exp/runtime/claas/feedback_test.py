"""Retained feedback readers preserve unknown scores and enforce scoped source lifetime."""

import json
import sqlite3
from pathlib import Path

import pytest

from exp.common.claas import ClaasScope
from exp.runtime.claas.feedback import FeedbackStore


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """Create realistic durable rows including expired and neighboring application evidence."""
    path = tmp_path / "capture.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
      CREATE TABLE claas_experiences(user_id,application_id,response_id,expires_at);
      CREATE TABLE claas_feedback(sequence INTEGER PRIMARY KEY,user_id,application_id,
        feedback_id,response_id,episode_id,expires_at,payload);
      CREATE TABLE claas_episodes(sequence INTEGER PRIMARY KEY,user_id,application_id,
        episode_id,expires_at,payload);
      CREATE TABLE claas_episode_members(user_id,application_id,episode_id,response_id);
      INSERT INTO claas_experiences VALUES('user','app','response',unixepoch()+60);
      INSERT INTO claas_episode_members VALUES('user','app','episode','response');
    """)
    scope = {"user_id": "user", "application_id": "app"}
    episode = {
        "schema_version": 1,
        "scope": scope,
        "finalized_at": 1_800_000_000,
        "episode": {
            "application_id": "app",
            "episode_id": "episode",
            "response_ids": ["response"],
            "status": "completed",
        },
    }
    connection.execute(
        "INSERT INTO claas_episodes VALUES(1,'user','app','episode',unixepoch()+60,?)",
        (json.dumps(episode),),
    )
    for index, target in enumerate(("response", "episode"), start=1):
        feedback = {
            "application_id": "app",
            "feedback_id": f"fb-{index}",
            f"{target}_id": target,
            "text": "Correct the address.",
        }
        record = {
            "schema_version": 1,
            "scope": scope,
            "feedback": feedback,
            "created_at": 1_800_000_000,
        }
        connection.execute(
            "INSERT INTO claas_feedback VALUES(?,'user','app',?,?,?,unixepoch()+60,?)",
            (
                index,
                f"fb-{index}",
                "response" if target == "response" else None,
                "episode" if target == "episode" else None,
                json.dumps(record),
            ),
        )
    connection.commit()
    connection.close()
    return path


def test_scoped_pagination_preserves_unknown_and_excludes_other_scopes(database: Path) -> None:
    """Read independent monotonic cursors without treating text as a numeric outcome."""
    store = FeedbackStore(database, ClaasScope(user_id="user", application_id="app"))
    first = store.read_after(limit=1)
    assert len(first) == 1
    assert first[0].record.feedback.training_reward is None
    assert [row.sequence for row in store.read_after(first[0].sequence)] == [2]
    assert store.episodes_after()[0].record.episode.response_ids == ("response",)
    assert (
        FeedbackStore(database, ClaasScope(user_id="other", application_id="app")).read_after()
        == ()
    )
    with pytest.raises(ValueError, match="sequence"):
        store.read_after(-1)


@pytest.mark.parametrize("delete", [True, False])
def test_source_eviction_or_expiration_hides_dependent_records(
    database: Path, delete: bool
) -> None:
    """Even before the native prune tick, deleted evidence is unavailable to consumers."""
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM claas_experiences"
            if delete
            else "UPDATE claas_experiences SET expires_at=unixepoch()-1"
        )
    store = FeedbackStore(database, ClaasScope(user_id="user", application_id="app"))
    assert store.read_after() == ()
    assert store.episodes_after() == ()


def test_corrupt_scope_payload_is_not_relabelled(database: Path) -> None:
    """A durable row cannot launder a mismatched payload into the selected partition."""
    with sqlite3.connect(database) as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM claas_feedback WHERE sequence=1").fetchone()[0]
        )
        payload["scope"]["user_id"] = "forged"
        connection.execute(
            "UPDATE claas_feedback SET payload=? WHERE sequence=1", (json.dumps(payload),)
        )
    with pytest.raises(ValueError, match="scope"):
        FeedbackStore(database, ClaasScope(user_id="user", application_id="app")).read_after()


def test_real_native_feedback_is_durable_scoped_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use local HTTP and SQLite to verify acknowledgement, key authority, and finalization."""
    import threading
    import time
    from http.server import ThreadingHTTPServer

    import httpx

    from exp.common.claas import CapturePolicy
    from exp.runtime.claas.capture import CaptureBinding, CaptureConfiguration
    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.native_bridge import NativeControlPlane
    from exp.runtime.gateway.native_server import serve_native_gateway
    from exp.runtime.gateway.tests.launch_test import (
        _configure_gateway,
        _LoopbackProvider,
        _unused_port,
        _wait_ready,
    )

    native = pytest.importorskip("exp_gateway_native")
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    manager.create_identity(identity_id="neighbor", display_name="Neighbor")
    neighbor_key = manager.issue_key(identity_id="neighbor", key_id="neighbor-key").raw_key
    components = load_gateway_components(tmp_path)
    scope = ClaasScope(
        user_id=components.store.authenticated_identity(raw_key=raw_key)[1], application_id="claims"
    )
    path = tmp_path / "feedback.sqlite3"
    capture = CaptureConfiguration(
        database_path=path,
        bindings=(CaptureBinding(alias="coding", policy=CapturePolicy(scope=scope, enabled=True)),),
    )
    shutdown = native.shutdown_handle()
    port = _unused_port()
    failures: list[BaseException] = []

    def run() -> None:
        """Serve the actual Rust gateway with local-only provider and capture storage."""
        try:
            serve_native_gateway(
                NativeControlPlane(components),
                host="127.0.0.1",
                port=port,
                capture=capture,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - reassert server failures after cleanup.
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        _wait_ready(port, thread)
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=10,
            headers={"Authorization": f"Bearer {raw_key}"},
        ) as client:
            completion = client.post(
                "/v1/chat/completions",
                json={
                    "model": "coding",
                    "messages": [{"role": "user", "content": "hello feedback"}],
                },
            )
            assert completion.status_code == 200, completion.text
            response_id = completion.json()["id"]
            request = {
                "application_id": "claims",
                "feedback_id": "fb-1",
                "response_id": response_id,
                "text": "Please confirm the address first.",
            }
            for _ in range(100):
                acknowledged = client.post("/v1/claas/feedback", json=request)
                if acknowledged.status_code != 409:
                    break
                time.sleep(0.01)
            assert acknowledged.status_code == 200, acknowledged.text
            store = FeedbackStore(path, scope)
            assert store.read_after()[0].record.feedback.training_reward is None
            replay = client.post("/v1/claas/feedback", json=request)
            assert replay.json()["replayed"] is True
            assert replay.json()["record"] == acknowledged.json()["record"]
            assert (
                client.post("/v1/claas/feedback", json={**request, "success": True}).status_code
                == 409
            )
            assert (
                client.post(
                    "/v1/claas/feedback", json={**request, "user_id": scope.user_id}
                ).status_code
                == 400
            )
            assert (
                client.post(
                    "/v1/claas/feedback",
                    json=request,
                    headers={"Authorization": f"Bearer {neighbor_key}"},
                ).status_code
                == 404
            )
            assert (
                client.post(
                    "/v1/claas/feedback", json=request, headers={"Authorization": "Bearer invalid"}
                ).status_code
                == 401
            )
            missing = client.post(
                "/v1/claas/feedback",
                json={**request, "feedback_id": "missing", "response_id": "not-retained"},
            )
            assert missing.status_code == 409
            assert missing.headers["retry-after"] == "1"
            episode = {
                "application_id": "claims",
                "episode_id": "episode",
                "response_ids": [response_id],
                "status": "completed",
            }
            assert client.post("/v1/claas/episodes/finalize", json=episode).status_code == 200
            assert len(store.episodes_after()) == 1
            assert (
                client.post("/v1/claas/episodes/finalize", json=episode).json()["replayed"] is True
            )
            assert (
                client.post(
                    "/v1/claas/episodes/finalize", json={**episode, "status": "failed"}
                ).status_code
                == 409
            )
    finally:
        shutdown.request_shutdown()
        thread.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not thread.is_alive()
