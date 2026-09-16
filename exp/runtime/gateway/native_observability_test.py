"""Key-derived CLaaS authority checks at the real native control-plane boundary."""

import json
from pathlib import Path

import pytest

from exp.runtime.gateway.native_bridge import NativeBridgeError
from exp.runtime.gateway.native_bridge_test import _control_plane


def test_claas_authority_uses_authenticated_identity(tmp_path: Path) -> None:
    """Ignore caller identity claims and reject an invalid key without model admission."""
    control, raw_key = _control_plane(tmp_path)
    expected = control._components.store.authenticated_identity(raw_key=raw_key)[1]
    result = json.loads(
        control.claas_authority(json.dumps({"raw_key": raw_key, "user_id": "forged"}))
    )
    assert result == {"user_id": expected}
    with pytest.raises(NativeBridgeError):
        control.claas_authority(json.dumps({"raw_key": "invalid"}))
