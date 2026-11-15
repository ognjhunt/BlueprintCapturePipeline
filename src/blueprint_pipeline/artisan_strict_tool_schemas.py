"""Every tool ARTISAN sends to OpenAI as ``strict`` must be expressible under strict function calling.

On 2026-11-13 a paid stage 3 stopped in ARTISAN's review phase: the API refused the request
(HTTP 400 ``invalid_function_parameters``) because ``request_render``'s ``pose``, a free-form
joint-to-value map, was sent ``strict: true`` and strict mode cannot express a map
(``patches/artisan/0005-request-render-non-strict.patch`` sends that one tool non-strict).

This module holds the rule so the next such tool is found without a paid call:

- ``strict_tool_violations`` reads converted tools (the codec's output) and names every rule a
  strict tool breaks: every object schema defines ``properties``, sets ``additionalProperties``
  to false (no free-form map) and requires every property; optional values are written as the
  codec's nullable transform (``type`` lists ending in ``"null"``); arrays carry ``items``; only
  JSON types are used; ``oneOf`` and ``$ref`` are not used.
- ``upstream_tool_schemas`` computes, from a patched upstream source tree and without importing
  it, the exact tools the OpenAI codec sends for the design, coding and review agents: it
  executes only the three tool-list functions and the codec's conversion functions, read from
  the source by their syntax tree, with any module constant they name read as a literal. A
  function holding an import, a ``global`` or ``nonlocal`` statement or an underscore-prefixed
  attribute is refused before it runs; the rest sees only ``PURE_BUILTINS`` and ``copy.deepcopy``.
- ``SNAPSHOT`` (``config/artisan_openai_tool_schemas.v1.json``) is that output for the pinned
  upstream with every patch applied. CI checks the snapshot; the component builder recomputes it
  from the real patched source and refuses to seal a component whose tools differ from the
  snapshot or break a rule, so a new or changed tool fails at build, never in a paid run.
  Regenerate it in the change that alters a tool, from a tree with every patch applied:
  ``python -m blueprint_pipeline.artisan_strict_tool_schemas --source <tree> --revision <sha>``
  (docs/task_object_artisan_authoring.md, "Strict tool schemas and their snapshot").

Nothing here calls a network, a model or a provider.
"""
from __future__ import annotations

import __future__
import ast
import builtins
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEMA_VERSION = "artisan_openai_tool_schemas.v1"
SNAPSHOT = Path(__file__).resolve().parents[2] / "config" / "artisan_openai_tool_schemas.v1.json"
#: The agent role, the upstream file and the function returning that agent's tool list.
TOOL_SOURCES = (
    ("design", "pipeline/agents/design_harness.py", "_design_tool_schemas"),
    ("coding", "pipeline/harness_sdk.py", "_tool_schemas"),
    ("review", "pipeline/harness_review.py", "_review_tool_schemas"),
)
CODEC_SOURCE = "pipeline/llm/openai_codec.py"
CODEC_FUNCTIONS = ("make_json_schema_nullable", "normalize_tool_json_schema_for_responses",
                   "normalize_function_parameters_for_responses", "convert_tools")
_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
#: The schema keywords a strict tool may use: the ones the pinned upstream's tools use today
#: (2026-11-15), plus ``anyOf``. Any other keyword fails the build by name; widening this list is a
#: reviewed change that checks OpenAI's documentation for that keyword first.
ALLOWED_KEYWORDS = frozenset({"type", "properties", "required", "additionalProperties", "items", "enum",
                              "description", "minimum", "maximum", "anyOf"})
#: OpenAI's structured-output size limits as we read them (2026-11): at most 10 levels of object
#: nesting, at most 5,000 object properties in one schema, at most 1,000 enum values in one schema.
#: These are our reading of the published rule, not values the API reports; the pinned tools use
#: depth 5 and at most 35 properties.
MAX_NESTING_DEPTH = 10
MAX_TOTAL_PROPERTIES = 5_000
MAX_TOTAL_ENUM_VALUES = 1_000
_MAX_NAME_RESOLUTIONS = 32
#: Pure built-ins only (no import, open, eval, exec, compile, getattr, reflection or I/O); widening is reviewed.
PURE_BUILTINS = ("any", "all", "bool", "dict", "enumerate", "float", "frozenset", "int", "isinstance", "len",
                 "list", "max", "min", "range", "set", "sorted", "str", "tuple", "zip")
_NOT_PLAIN = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


class ArtisanToolSchemaError(ValueError):
    """The tool lists could not be read from the source, or break the strict rules."""


def _types(schema: Mapping[str, Any]) -> list[str]:
    value = schema.get("type")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object" if isinstance(value, Mapping) else "unknown"


def _schema_violations(schema: Any, path: str, *, depth: int = 0) -> list[str]:
    if not isinstance(schema, Mapping):
        return [f"{path}:schema_not_object"]
    found: list[str] = []
    for keyword in ("oneOf", "$ref"):
        if keyword in schema:
            found.append(f"{path}:{keyword}_unsupported")
    for keyword in sorted(set(schema) - ALLOWED_KEYWORDS - {"oneOf", "$ref"}):
        found.append(f"{path}:keyword_unsupported:{keyword}")
    branches = schema.get("anyOf")
    if branches is not None:
        if not isinstance(branches, list) or not branches:
            found.append(f"{path}:anyOf_invalid")
        else:
            for index, branch in enumerate(branches):
                found.extend(_schema_violations(branch, f"{path}/anyOf/{index}", depth=depth))
        return found
    types = _types(schema)
    if not types:
        return [*found, f"{path}:type_missing"]
    if any(item not in _JSON_TYPES for item in types) or len(set(types)) != len(types):
        found.append(f"{path}:type_invalid")
    if "null" in types and types[-1] != "null":
        found.append(f"{path}:nullable_type_not_last")
    enum = schema.get("enum")
    if enum is not None:
        # An integer is a number; nothing else crosses types.
        if not isinstance(enum, list) or not enum or any(
                _json_type(value) not in types and not (_json_type(value) == "integer" and "number" in types)
                for value in enum):
            found.append(f"{path}:enum_value_type_mismatch")
    if "object" in types:
        if depth >= MAX_NESTING_DEPTH:
            found.append(f"{path}:nesting_depth_exceeded")
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            found.append(f"{path}:object_without_properties")
            properties = {}
        if schema.get("additionalProperties") is not False:
            found.append(f"{path}:additional_properties_not_false")
        required = schema.get("required")
        # A repeated name fails the comparison too: sorted(["a", "a"]) != ["a"].
        if not isinstance(required, list) or sorted(required) != sorted(properties):
            found.append(f"{path}:not_every_property_required")
        for name, child in properties.items():
            found.extend(_schema_violations(child, f"{path}/properties/{name}", depth=depth + 1))
    if "array" in types:
        if "items" not in schema:
            found.append(f"{path}:array_without_items")
        else:
            found.extend(_schema_violations(schema["items"], f"{path}/items", depth=depth))
    return found


def _counts(schema: Any) -> tuple[int, int]:
    """Object properties and enum values in one schema, every level included."""
    if isinstance(schema, list):
        totals = [_counts(item) for item in schema]
        return sum(row[0] for row in totals), sum(row[1] for row in totals)
    if not isinstance(schema, Mapping):
        return 0, 0
    properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    enum = schema.get("enum") if isinstance(schema.get("enum"), list) else []
    nested = [_counts(child) for child in properties.values()]
    nested += [_counts(schema[key]) for key in ("items", "anyOf") if key in schema]
    return (len(properties) + sum(row[0] for row in nested), len(enum) + sum(row[1] for row in nested))


def strict_tool_violations(tools: list[Mapping[str, Any]]) -> list[str]:
    """Every strict-rule violation of the tools sent ``strict: true``, named ``<tool>:<path>:<rule>``."""
    found: list[str] = []
    for tool in tools:
        if tool.get("strict") is not True:
            continue
        name = str(tool.get("name") or "")
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping) or _types(parameters) != ["object"]:
            found.append(f"{name}::root_not_a_plain_object")
            continue
        found.extend(f"{name}:{item}" for item in _schema_violations(parameters, ""))
        properties, enum_values = _counts(parameters)
        if properties > MAX_TOTAL_PROPERTIES:
            found.append(f"{name}::total_properties_exceeded")
        if enum_values > MAX_TOTAL_ENUM_VALUES:
            found.append(f"{name}::total_enum_values_exceeded")
    return found


def _module_constant(source_root: Path, relative: str, name: str) -> Any:
    """The literal value of module constant ``name`` in ``relative`` or a ``pipeline`` module it imports it from."""
    tree = ast.parse((source_root / relative).read_text(encoding="utf-8"))
    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign) else
                   node.targets if isinstance(node, ast.Assign) else [])
        if any(isinstance(target, ast.Name) and target.id == name for target in targets) and node.value is not None:
            try:
                return ast.literal_eval(node.value)
            except ValueError as error:
                raise ArtisanToolSchemaError(f"artisan_tool_schema_name_not_literal:{relative}:{name}") from error
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("pipeline.")
                and any(alias.name == name and alias.asname is None for alias in node.names)):
            return _module_constant(source_root, node.module.replace(".", "/") + ".py", name)
    raise ArtisanToolSchemaError(f"artisan_tool_schema_name_unresolved:{relative}:{name}")


def _functions(source_root: Path, relative: str, names: tuple[str, ...]) -> dict[str, Any]:
    path = source_root / relative
    if path.is_symlink() or not path.is_file():
        raise ArtisanToolSchemaError(f"artisan_tool_schema_source_missing:{relative}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(node.name for node in found) != sorted(names):
        raise ArtisanToolSchemaError(f"artisan_tool_schema_function_missing:{relative}")
    for node in found:
        if any(isinstance(item, _NOT_PLAIN) or (isinstance(item, ast.Attribute) and item.attr.startswith("_"))
               for item in ast.walk(node)):
            raise ArtisanToolSchemaError(f"artisan_tool_schema_function_not_plain:{relative}:{node.name}")
    namespace: dict[str, Any] = {"copy": SimpleNamespace(deepcopy=copy.deepcopy), "Any": Any,
                                 "__builtins__": {name: getattr(builtins, name) for name in PURE_BUILTINS}}
    try:
        # Annotations stay unevaluated, as in the upstream module (``from __future__ import annotations``).
        code = compile(ast.Module(body=found, type_ignores=[]), str(path), "exec",
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)
        exec(code, namespace)  # only the checked functions of the pinned upstream
    except (NameError, SyntaxError, TypeError) as error:
        raise ArtisanToolSchemaError(f"artisan_tool_schema_function_unreadable:{relative}") from error
    return {"namespace": namespace, "relative": relative}


def _call(source_root: Path, loaded: Mapping[str, Any], name: str, *args: Any) -> Any:
    namespace = loaded["namespace"]
    for _ in range(_MAX_NAME_RESOLUTIONS):
        try:
            return namespace[name](*args)
        except NameError as error:
            missing = getattr(error, "name", None)
            if not missing or missing in namespace:
                raise ArtisanToolSchemaError(f"artisan_tool_schema_name_unresolved:{name}") from error
            namespace[missing] = _module_constant(source_root, loaded["relative"], missing)
        except AttributeError as error:  # e.g. copy.copy: only copy.deepcopy is exposed
            raise ArtisanToolSchemaError(f"artisan_tool_schema_attribute_unavailable:{name}:{error.name}") from error
    raise ArtisanToolSchemaError(f"artisan_tool_schema_name_unresolved:{name}")


def upstream_tool_schemas(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    """The tools the OpenAI codec sends for each agent role, computed from a (patched) source tree."""
    root = Path(source_root)
    codec = _functions(root, CODEC_SOURCE, CODEC_FUNCTIONS)
    result: dict[str, list[dict[str, Any]]] = {}
    for role, relative, function in TOOL_SOURCES:
        declared = _call(root, _functions(root, relative, (function,)), function)
        result[role] = json.loads(json.dumps(_call(root, codec, "convert_tools", declared)))
    return result


def snapshot_value(tools: Mapping[str, list[Mapping[str, Any]]], *, revision: str) -> dict[str, Any]:
    """The snapshot record: the upstream revision and the converted tools, by agent role."""
    return {"schema_version": SCHEMA_VERSION, "upstream_revision": revision, "tools": json.loads(json.dumps(tools))}


def read_snapshot(path: Path = SNAPSHOT) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION or not isinstance(value.get("tools"), dict):
        raise ArtisanToolSchemaError("artisan_tool_schema_snapshot_invalid")
    return value


def require_strict_tools(source_root: Path, *, revision: str, snapshot: Path = SNAPSHOT) -> dict[str, Any]:
    """The builder's check: the patched source's tools equal the snapshot and break no strict rule."""
    tools = upstream_tool_schemas(source_root)
    violations = [f"{role}:{item}" for role, rows in sorted(tools.items()) for item in strict_tool_violations(rows)]
    if violations:
        raise ArtisanToolSchemaError("artisan_strict_tool_schema_invalid:" + ",".join(violations[:8]))
    expected = read_snapshot(snapshot)
    if snapshot_value(tools, revision=revision) != expected:
        raise ArtisanToolSchemaError("artisan_tool_schema_snapshot_stale")
    return {"status": "strict_tools_valid", "strict_tool_count": sum(
        1 for rows in tools.values() for row in rows if row.get("strict") is True)}


def main(argv: list[str] | None = None) -> int:
    """Write the snapshot from a patched source tree: ``--source <tree> --revision <sha>``."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", default=str(SNAPSHOT))
    args = parser.parse_args(argv)
    tools = upstream_tool_schemas(Path(args.source))
    Path(args.output).write_text(json.dumps(snapshot_value(tools, revision=args.revision),
                                            indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
