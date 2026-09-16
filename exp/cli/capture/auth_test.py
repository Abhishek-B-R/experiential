"""Capture uses the same endpoint-bound login as other CLI services."""

from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.capture.auth import capture_credentials
from exp.cli.providers.experiential_cloud import hosted_credential_binding
from exp.common.auth import ProviderAuthStore, StoredCredentialEndpointMismatch


def test_saved_login_reused_without_browser(tmp_path: Path) -> None:
    """Capture reuses an endpoint-bound login without initiating authentication."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("experiential-cloud", "xpl_saved", binding=hosted_credential_binding({}))
    credentials = capture_credentials(console=Console(), environment={}, root=tmp_path, store=store)
    assert credentials.api_key == "xpl_saved"
    assert credentials.api_url == "https://api.experientiallabs.ai"
    assert "xpl_saved" not in repr(credentials)


def test_preview_does_not_receive_production_credential(tmp_path: Path) -> None:
    """A preview endpoint cannot reuse the saved production credential."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("experiential-cloud", "xpl_saved", binding=hosted_credential_binding({}))
    with pytest.raises(StoredCredentialEndpointMismatch):
        capture_credentials(
            console=Console(),
            environment={"EXP_GATEWAY_URL": "https://preview.example/v1"},
            root=tmp_path,
            store=store,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://public.example/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/other",
    ],
)
def test_unsafe_endpoint_rejected_before_login(tmp_path: Path, endpoint: str) -> None:
    """Invalid API origins fail before credentials or login are consulted."""
    with pytest.raises(ValueError, match="EXP_GATEWAY_URL"):
        capture_credentials(
            console=Console(),
            environment={"EXP_GATEWAY_URL": endpoint},
            root=tmp_path,
            store=ProviderAuthStore(tmp_path / "absent.json"),
        )


def test_loopback_preview_environment_key_supported(tmp_path: Path) -> None:
    """An explicit local development endpoint accepts its ordinary environment key."""
    credentials = capture_credentials(
        console=Console(),
        environment={
            "EXP_GATEWAY_URL": "http://127.0.0.1:8000/v1",
            "EXPLABS_API_KEY": "xpl_local",
        },
        root=tmp_path,
        store=ProviderAuthStore(tmp_path / "absent.json"),
    )
    assert credentials.api_url == "http://127.0.0.1:8000"
    assert credentials.api_key == "xpl_local"
