"""Lossless expansion of local JSON Schema references for reference-free tool schemas.

Some served models follow a tool schema's nested object shapes only when those shapes are
written inline: behind a ``$ref`` they may return the object as a JSON-encoded string.
Expanding acyclic local references produces an equivalent schema without that indirection.
"""

import copy

from exp.common.core.artifacts import JsonObject, JsonValue

# Keywords whose value is one subschema, a name-to-subschema map, or a subschema list.
_SINGLE = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_MAP = frozenset({"dependentSchemas", "patternProperties", "properties"})
_ARRAY = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_DEFINITIONS = frozenset({"$defs", "definitions"})


def expand_local_references(schema: JsonObject) -> JsonObject:
    """Return an equivalent schema with every acyclic ``#/...`` reference written inline.

    Only schema positions are traversed. Instance values such as ``default``, ``const``,
    ``enum`` and ``examples`` and property names (including a property named ``$ref``) are
    copied unchanged. A reference with sibling keywords becomes ``allOf`` of the target and
    the siblings, preserving both constraints. Definition tables are dropped afterwards.

    Args:
        schema: Root JSON Schema object.

    Returns:
        A new reference-free schema; the input is never mutated.

    Raises:
        ValueError: A reference is recursive, dynamic, external or unresolvable, so it has
            no finite inline equivalent.
    """

    def resolve(reference: str) -> JsonValue:
        """Follow one JSON Pointer from the root."""
        target: JsonValue = schema
        for token in reference[2:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and token in target:
                target = target[token]
            elif isinstance(target, list) and token.isdigit() and int(token) < len(target):
                target = target[int(token)]
            else:
                raise ValueError(f"unresolvable schema reference {reference!r}")
        return target

    def walk(node: JsonValue, stack: tuple[str, ...]) -> JsonValue:
        """Expand one subschema position."""
        if not isinstance(node, dict):
            return copy.deepcopy(node)
        if "$dynamicRef" in node or "$recursiveRef" in node:
            raise ValueError("dynamic schema references have no inline equivalent")
        if "$ref" in node:
            reference = node["$ref"]
            if not isinstance(reference, str) or not reference.startswith("#/"):
                raise ValueError("only local schema references can be expanded")
            if reference in stack:
                raise ValueError(f"recursive schema reference {reference!r}")
            target = walk(resolve(reference), (*stack, reference))
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            if not siblings:
                return target
            return {"allOf": [target, walk(siblings, stack)]}
        result: JsonObject = {}
        for key, value in node.items():
            if key in _DEFINITIONS:
                continue
            if key in _SINGLE:
                result[key] = walk(value, stack)
            elif key in _MAP and isinstance(value, dict):
                result[key] = {name: walk(child, stack) for name, child in value.items()}
            elif key in _ARRAY and isinstance(value, list):
                result[key] = [walk(child, stack) for child in value]
            else:
                result[key] = copy.deepcopy(value)
        return result

    expanded = walk(schema, ())
    if not isinstance(expanded, dict):
        raise ValueError("schema root must be an object")
    return expanded
