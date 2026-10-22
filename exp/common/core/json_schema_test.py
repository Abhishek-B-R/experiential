"""Reference expansion is lossless, schema-position aware and fails closed."""

import json

import jsonschema
import pytest
from pydantic import BaseModel

from exp.common.core.artifacts import JsonObject
from exp.common.core.json_schema import expand_local_references


class Effect(BaseModel):
    """Nested object that pydantic emits behind a reference."""

    kind: str
    state: dict[str, int]


class Proposal(BaseModel):
    """Root model with a referenced nested object."""

    effect: Effect
    others: list[Effect] = []


def test_expansion_removes_references_and_preserves_validation() -> None:
    """Every sample validates identically before and after expansion."""
    schema = Proposal.model_json_schema()
    assert "$ref" in json.dumps(schema)
    expanded = expand_local_references(schema)
    assert "$ref" not in json.dumps(expanded) and "$defs" not in expanded
    samples: list[JsonObject] = [
        {"effect": {"kind": "replace", "state": {"a": 1}}},
        {"effect": json.dumps({"kind": "replace", "state": {"a": 1}})},
        {"effect": {"kind": "replace"}},
        {"effect": {"kind": "x", "state": {}}, "others": [{"kind": "y", "state": {"b": 2}}]},
        {"effect": {"kind": "x", "state": {}}, "others": ["not an object"]},
    ]
    for sample in samples:
        before = jsonschema.Draft202012Validator(schema).is_valid(sample)
        after = jsonschema.Draft202012Validator(expanded).is_valid(sample)
        assert before == after, sample
    assert schema == Proposal.model_json_schema()


def test_instance_values_property_names_and_siblings_are_kept() -> None:
    """Defaults, enums and a property literally named $ref are not schema positions."""
    schema: JsonObject = {
        "$defs": {"Leaf": {"type": "string"}},
        "type": "object",
        "properties": {
            "$ref": {"type": "integer"},
            "leaf": {"$ref": "#/$defs/Leaf", "description": "kept", "maxLength": 3},
            "fixed": {"const": {"$ref": "#/not/a/schema"}, "default": {"$ref": "x"}},
        },
    }
    expanded = expand_local_references(schema)
    properties = expanded["properties"]
    assert isinstance(properties, dict)
    assert properties["$ref"] == {"type": "integer"}
    assert properties["leaf"] == {
        "allOf": [{"type": "string"}, {"description": "kept", "maxLength": 3}]
    }
    assert properties["fixed"] == {"const": {"$ref": "#/not/a/schema"}, "default": {"$ref": "x"}}


@pytest.mark.parametrize(
    "reference",
    ["#/$defs/Node", "https://example.test/schema.json", "#/$defs/Missing"],
)
def test_recursive_external_and_missing_references_fail_closed(reference: str) -> None:
    """No finite lossless inline form exists, so expansion refuses."""
    schema: JsonObject = {
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "$ref": reference,
    }
    with pytest.raises(ValueError, match="reference"):
        expand_local_references(schema)
