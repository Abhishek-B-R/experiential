"""Billing-source upgrades preserve authored input and reject mixed schema records."""

from copy import deepcopy

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog_upgrade import upgrade_billing_source


def test_upgrade_preserves_input_and_assigns_nested_billing_ownership() -> None:
    """Add billing ownership to a model and its training base without mutating either."""
    document: JsonObject = {
        "schema_version": 1,
        "models": {"agent": {"sft_provenance": {"base_model": {"model_id": "base"}}}},
    }
    original = deepcopy(document)
    assert upgrade_billing_source(document) == {
        "schema_version": 2,
        "models": {
            "agent": {
                "billing_source": "customer_managed",
                "sft_provenance": {
                    "base_model": {"model_id": "base", "billing_source": "customer_managed"}
                },
            }
        },
    }
    assert document == original


@pytest.mark.parametrize("nested", [False, True])
def test_upgrade_rejects_current_billing_fields_in_schema_one(nested: bool) -> None:
    """Reject mixed authored schemas at either the model or its training base."""
    record: JsonObject = {"billing_source": "host_managed"}
    if nested:
        record = {"sft_provenance": {"base_model": record}}
    with pytest.raises(ValueError, match="must not declare current billing_source"):
        upgrade_billing_source({"schema_version": 1, "models": {"agent": record}})


def test_current_schema_is_returned_without_billing_inference() -> None:
    """Leave missing current billing fields for strict catalog validation to reject."""
    document: JsonObject = {"schema_version": 3, "models": {"agent": {}}}
    assert upgrade_billing_source(document) is document
