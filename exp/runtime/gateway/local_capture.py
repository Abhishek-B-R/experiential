"""Local identity-scoped gateway capture composition over the native collector."""

from __future__ import annotations

from pathlib import Path

from exp.common.claas import CapturePolicy, ClaasScope
from exp.runtime.claas.capture import CaptureBinding, CaptureConfiguration
from exp.runtime.gateway.management import GatewayManagement

GATEWAY_CAPTURE_APPLICATION = "gateway"


def local_capture_path(root: Path) -> Path:
    """Return the separate content database, never the accounting database."""
    return (root / "gateway" / "traffic.db").resolve()


def local_capture_configuration(root: Path, *, ghost: bool = False) -> CaptureConfiguration | None:
    """Enable bounded capture for current granted identities unless explicitly disabled.

    This local policy includes own-provider-key traffic. Hosted consent, BYOK
    exclusions and tenant persistence are deliberately not decided here.
    Bindings are a startup snapshot; restart after changing identity grants.
    """
    if ghost:
        return None
    manager = GatewayManagement(root)
    identities = {identity.identity_id for identity in manager.identities() if identity.active}
    aliases = {alias.alias_name for alias in manager.aliases() if alias.active}
    bindings = tuple(
        CaptureBinding(
            alias=grant.alias_name,
            policy=CapturePolicy(
                scope=ClaasScope(
                    user_id=grant.identity_id,
                    application_id=GATEWAY_CAPTURE_APPLICATION,
                ),
                enabled=True,
            ),
        )
        for grant in manager.grants()
        if grant.identity_id in identities and grant.alias_name in aliases
    )
    if not bindings:
        return None
    return CaptureConfiguration(database_path=local_capture_path(root), bindings=bindings)
