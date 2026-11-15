"""Every tool ARTISAN sends to OpenAI as strict follows strict function calling's rules (no paid call).

2026-11-13: a paid stage 3 stopped when the API refused the review agent's ``request_render``
(HTTP 400 ``invalid_function_parameters`` at ``/properties/pose/type/0``): a free-form map sent
strict. ``patches/artisan/0005-request-render-non-strict.patch`` sends that tool non-strict;
``config/artisan_openai_tool_schemas.v1.json`` is every tool the pinned, patched upstream sends,
and the component builder refuses a source whose tools differ from it or break a rule.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline import artisan_strict_tool_schemas as tools
from blueprint_pipeline.task_object_authoring_backend import ARTISAN_REVISION

ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/artisan/0005-request-render-non-strict.patch").read_text()
BUILDER = ROOT / "scripts/build_task_evaluation_scene_configuration_artisan_component.py"


def _tool(name: str, parameters: dict, *, strict: bool = True) -> dict:
    return {"type": "function", "name": name, "description": "", "parameters": parameters, "strict": strict}


def _strict_object(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": sorted(properties), "additionalProperties": False}


def test_every_strict_tool_in_the_snapshot_follows_the_strict_rules():
    snapshot = tools.read_snapshot()
    assert snapshot["upstream_revision"] == ARTISAN_REVISION
    by_role = snapshot["tools"]
    assert sorted(by_role) == ["coding", "design", "review"]
    assert all(by_role[role] for role in by_role)
    assert {role: tools.strict_tool_violations(rows) for role, rows in by_role.items()} == {
        "coding": [], "design": [], "review": []}
    # Most tools stay strict: the fix is one tool, not strictness switched off.
    strict = {role: sorted(row["name"] for row in rows if row["strict"] is True) for role, rows in by_role.items()}
    assert "request_render" in strict["coding"] and "submit_review" in strict["review"]
    assert sum(len(names) for names in strict.values()) == 21


def test_the_review_request_render_is_non_strict_and_otherwise_unchanged():
    by_role = tools.read_snapshot()["tools"]
    review = {row["name"]: row for row in by_role["review"]}["request_render"]
    coding = {row["name"]: row for row in by_role["coding"]}["request_render"]
    assert review["strict"] is False and coding["strict"] is True
    # The model still sees the joint -> value map its prompts describe (``pose={joint: value}``).
    pose = review["parameters"]["properties"]["pose"]
    assert pose["type"] == ["object", "null"] and pose["additionalProperties"] == {"type": "number"}
    assert "properties" not in pose
    assert sorted(review["parameters"]["required"]) == sorted(review["parameters"]["properties"])
    # Sent strict, exactly this schema is what the API refused on 2026-11-13.
    refused = tools.strict_tool_violations([{**review, "strict": True}])
    assert refused == ["request_render:/properties/pose:object_without_properties",
                       "request_render:/properties/pose:additional_properties_not_false",
                       "request_render:/properties/pose:not_every_property_required"]


def _changed(patch: str) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = files.setdefault(line[len("+++ b/"):], [])
        elif current is not None and line[:1] in {"+", "-"} and not line.startswith(("+++", "---")):
            current.append(line)
    return files


def test_the_patch_marks_only_the_review_tool_and_the_codec_honours_the_mark():
    changed = _changed(PATCH)
    assert sorted(changed) == ["pipeline/harness_review.py", "pipeline/harness_sdk.py", "pipeline/llm/openai_codec.py"]
    # The coding harness only gains its read_file argument check: nothing removed, one guard added.
    sdk = changed["pipeline/harness_sdk.py"]
    assert not [line for line in sdk if line.startswith("-")]
    assert [line for line in sdk if "isinstance" in line] == [
        '+                if not isinstance(args, dict) or not isinstance(args.get("path", ""), str):']
    codec = "\n".join(changed["pipeline/llm/openai_codec.py"])
    assert '-                    "strict": False if is_read_file else True,' in codec
    assert '+                    "strict": not (is_read_file or tool.get("strict") is False),' in codec
    review = "\n".join(changed["pipeline/harness_review.py"])
    assert review.count("non_strict(fn(") == 1 and '"strict": False' in review
    assert "physicalvision/Artisan@" + ARTISAN_REVISION in PATCH


@pytest.mark.parametrize(("schema", "rule"), [
    ({"type": "object", "additionalProperties": False, "required": []}, "object_without_properties"),
    ({"type": "object", "properties": {}, "required": [], "additionalProperties": {"type": "number"}},
     "additional_properties_not_false"),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}, "additional_properties_not_false"),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False},
     "not_every_property_required"),
    ({"type": "array"}, "array_without_items"),
    ({"oneOf": [{"type": "string"}]}, "oneOf_unsupported"),
    ({"type": "string", "$ref": "#/x"}, "$ref_unsupported"),
    ({"type": ["null", "string"]}, "nullable_type_not_last"),
    ({"type": "map"}, "type_invalid"),
    ({"description": "untyped"}, "type_missing"),
])
def test_each_strict_rule_names_the_property_that_breaks_it(schema, rule):
    found = tools.strict_tool_violations([_tool("t", _strict_object({"field": schema}))])
    assert found and all(item.startswith("t:/properties/field") for item in found)
    assert any(item.endswith(":" + rule) for item in found)
    # The same schema inside an array or an anyOf branch is found there.
    nested = tools.strict_tool_violations([_tool("t", _strict_object(
        {"list": {"type": "array", "items": schema}, "either": {"anyOf": [schema, {"type": "null"}]}}))])
    assert any(item.startswith("t:/properties/list/items") for item in nested)
    assert any(item.startswith("t:/properties/either/anyOf/0") for item in nested)


def test_valid_strict_tools_and_non_strict_tools_pass():
    nullable = {"type": ["object", "null"], **{k: v for k, v in _strict_object({"x": {"type": "number"}}).items() if k != "type"}}
    valid = _tool("ok", _strict_object({"views": {"type": "array", "items": {"type": "string"}},
                                        "camera": nullable, "segment": {"type": ["boolean", "null"]}}))
    free_map = _tool("loose", _strict_object({"pose": {"type": "object", "additionalProperties": {"type": "number"}}}),
                     strict=False)
    assert tools.strict_tool_violations([valid, free_map]) == []
    assert tools.strict_tool_violations([_tool("root", {"type": ["object", "null"], "properties": {}})]) == [
        "root::root_not_a_plain_object"]


_CODEC = '''
def make_json_schema_nullable(schema):
    out = copy.deepcopy(schema)
    out["type"] = [out["type"], "null"]
    return out


def normalize_tool_json_schema_for_responses(schema, *, nullable=False):
    out = copy.deepcopy(schema)
    if isinstance(out.get("properties"), dict):
        required = set(out.get("required") or [])
        out["properties"] = {key: normalize_tool_json_schema_for_responses(value, nullable=key not in required)
                             for key, value in out["properties"].items()}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return make_json_schema_nullable(out) if nullable else out


def normalize_function_parameters_for_responses(parameters, *, required_override=None):
    return normalize_tool_json_schema_for_responses(parameters)


def convert_tools(tools):
    return [{"type": "function", "name": t["function"]["name"], "description": "",
             "parameters": normalize_function_parameters_for_responses(t["function"]["parameters"]),
             "strict": t.get("strict") is not False} for t in tools]
'''


def _fake_upstream(root: Path, *, pose_strict: bool) -> Path:
    (root / "pipeline/llm").mkdir(parents=True)
    (root / "pipeline/agents").mkdir()
    (root / "pipeline/llm/openai_codec.py").write_text("import copy\n" + _CODEC)
    (root / "pipeline/policies.py").write_text('DIMS: tuple[str, ...] = ("a", "b")\n')
    marker = "" if pose_strict else ', "strict": False'
    (root / "pipeline/harness_review.py").write_text(
        "from pipeline.policies import DIMS\n\n"
        "def _review_tool_schemas():\n"
        "    return [{\"type\": \"function\", \"function\": {\"name\": \"rubric\", \"parameters\": {\"type\": \"object\", "
        "\"properties\": {d: {\"type\": \"integer\"} for d in DIMS}, \"required\": list(DIMS)}}},\n"
        "            {\"type\": \"function\"" + marker + ", \"function\": {\"name\": \"request_render\", \"parameters\": "
        "{\"type\": \"object\", \"properties\": {\"pose\": {\"type\": \"object\", \"additionalProperties\": "
        "{\"type\": \"number\"}}}, \"required\": []}}}]\n")
    for relative, name in (("pipeline/harness_sdk.py", "_tool_schemas"),
                           ("pipeline/agents/design_harness.py", "_design_tool_schemas")):
        (root / relative).write_text(
            f"def {name}():\n    return [{{\"type\": \"function\", \"function\": {{\"name\": \"read\", "
            "\"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, "
            "\"required\": [\"path\"]}}}]\n")
    return root


def test_tools_are_computed_from_the_source_without_importing_it(tmp_path):
    source = _fake_upstream(tmp_path / "fixed", pose_strict=False)
    computed = tools.upstream_tool_schemas(source)
    assert sorted(computed) == ["coding", "design", "review"]
    rubric = computed["review"][0]["parameters"]
    assert sorted(rubric["properties"]) == ["a", "b"]  # a module constant imported from pipeline.policies
    assert computed["review"][1]["strict"] is False


def test_the_builder_check_refuses_a_bad_tool_and_a_stale_snapshot(tmp_path):
    fixed = _fake_upstream(tmp_path / "fixed", pose_strict=False)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(tools.snapshot_value(tools.upstream_tool_schemas(fixed), revision="r")))
    assert tools.require_strict_tools(fixed, revision="r", snapshot=snapshot) == {
        "status": "strict_tools_valid", "strict_tool_count": 3}
    broken = _fake_upstream(tmp_path / "broken", pose_strict=True)
    with pytest.raises(tools.ArtisanToolSchemaError, match="^artisan_strict_tool_schema_invalid:review:request_render:"):
        tools.require_strict_tools(broken, revision="r", snapshot=snapshot)
    changed = copy.deepcopy(json.loads(snapshot.read_text()))
    changed["tools"]["coding"][0]["description"] = "edited"
    snapshot.write_text(json.dumps(changed))
    with pytest.raises(tools.ArtisanToolSchemaError, match="^artisan_tool_schema_snapshot_stale$"):
        tools.require_strict_tools(fixed, revision="r", snapshot=snapshot)
    snapshot.write_text(json.dumps(tools.snapshot_value(tools.upstream_tool_schemas(fixed), revision="r")))
    with pytest.raises(tools.ArtisanToolSchemaError, match="^artisan_tool_schema_snapshot_stale$"):
        tools.require_strict_tools(fixed, revision="another upstream", snapshot=snapshot)


def test_the_builder_checks_the_patched_source_before_anything_is_sealed():
    spec = importlib.util.spec_from_file_location("artisan_builder_strict_tools_under_test", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.UPSTREAM_PATCHES[-1] == "0005-request-render-non-strict.patch"
    build = next(node for node in ast.parse(BUILDER.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "build_artisan_scene_configuration_component")
    calls = [node.func.id for node in ast.walk(build) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.index("apply_upstream_patches") < calls.index("require_strict_tools") < calls.index("_runtime_preflight")
    check = next(node for node in ast.walk(build) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "require_strict_tools")
    assert ast.unparse(check.args[0]) == "source"
    assert "config/artisan_openai_tool_schemas.v1.json" in ast.unparse(check)


# --- the non-strict tools' handlers check their arguments before using them (0005) -------------

def _added_lines(path: str) -> list[str]:
    section = PATCH.split(f"+++ b/{path}\n", 1)[1].split("\ndiff --git ", 1)[0]
    return [line[1:] for line in section.splitlines() if line.startswith("+")]


def _render_check():
    added = "\n".join(_added_lines("pipeline/harness_review.py"))
    start = added.index("REQUEST_RENDER_SHAPE_ERROR = (")
    end = added.index("    return None\n", added.index("def _request_render_arguments_error(")) + len("    return None\n")
    namespace = {"math": __import__("math"), "Any": object}
    exec(added[start:end], namespace)  # noqa: S102 - the patch's own added lines
    return namespace["_request_render_arguments_error"], namespace["REQUEST_RENDER_SHAPE_ERROR"]


#: The independent review's cases (S/rv-scene-a/probes/render_args.py) plus valid calls.
RENDER_CASES = {
    "ok": ({"views": ["front"], "pose": {"j": 0.5}}, True),
    "ok minimal": ({"views": ["front", "perspective_open"]}, True),
    "ok every field": ({"views": ["front"], "focus": ["hinge"], "pose": {"j": 1}, "camera": {"azimuth_deg": 30},
                        "segment": True}, True),
    "pose null value": ({"views": ["front"], "pose": {"j": None}}, False),
    "pose string value": ({"views": ["front"], "pose": {"j": "open"}}, False),
    "pose numeric string": ({"views": ["front"], "pose": {"j": "0.5"}}, False),
    "pose boolean": ({"views": ["front"], "pose": {"j": True}}, False),
    "pose infinite": ({"views": ["front"], "pose": {"j": float("inf")}}, False),
    "pose list": ({"views": ["front"], "pose": [{"joint": "j", "value": 0.5}]}, False),
    "pose string": ({"views": ["front"], "pose": "j=0.5"}, False),
    "pose nested": ({"views": ["front"], "pose": {"j": {"value": 0.5}}}, False),
    "views string": ({"views": "front"}, False),
    "views int": ({"views": 3}, False),
    "focus int": ({"views": ["front"], "focus": 3}, False),
    "args list": (["front"], False),
    "args string": ("front", False),
}


@pytest.mark.parametrize("case", sorted(RENDER_CASES))
def test_a_malformed_request_render_call_gets_an_error_never_an_exception(case):
    check, shape = _render_check()
    args, well_formed = RENDER_CASES[case]
    assert check(args) == (None if well_formed else shape)
    assert shape.startswith("ERROR: request_render takes an object: views=[view names]")


def test_the_render_handler_checks_before_it_converts_and_otherwise_is_unchanged():
    section = PATCH.split("+++ b/pipeline/harness_review.py\n", 1)[1].split("\ndiff --git ", 1)[0]
    assert ("     def _request_render(self, args: dict[str, Any]) -> tuple[str, list[Path]]:\n"
            "+        malformed = _request_render_arguments_error(args)\n"
            "+        if malformed is not None:\n"
            "+            return malformed, []\n"
            "         views = [str(v) for v in (args.get(\"views\") or [])]") in section
    # Nothing that converts a well-formed call is removed or edited.
    assert not [line for line in section.splitlines() if line.startswith("-") and "pose" in line]


@pytest.mark.parametrize("path", ["pipeline/harness_review.py", "pipeline/harness_sdk.py"])
def test_a_malformed_read_file_call_gets_an_error_in_both_harnesses(path):
    lines = _added_lines(path)
    guard = next(line.strip() for line in lines if 'isinstance(args.get("path", ""), str)' in line)
    assert guard.startswith("if ") and guard.endswith(":")
    condition = guard[3:-1]
    assert any(line.strip().startswith("return READ_FILE_SHAPE_ERROR") for line in lines)
    for args, refused in (({"path": "model.py"}, False), ({"path": "model.py", "offset": 2}, False),
                          ({}, False),  # a missing path keeps its existing "missing arg" error
                          ({"path": 3}, True), ({"path": None}, True), ({"path": ["model.py"]}, True),
                          (["model.py"], True), ("model.py", True)):
        assert eval(condition, {"args": args}) is refused, args  # noqa: S307 - the patch's own line


# --- checker rules added in the review round -------------------------------------------------

@pytest.mark.parametrize(("field", "rule"), [
    ({"anyOf": []}, "anyOf_invalid"),
    ({"anyOf": "string"}, "anyOf_invalid"),
    ({"type": ["string", "string"]}, "type_invalid"),
    ({"type": "string", "allOf": [{"type": "string"}]}, "keyword_unsupported:allOf"),
    ({"type": "string", "default": "x"}, "keyword_unsupported:default"),
    ({"type": "string", "minLength": 1}, "keyword_unsupported:minLength"),
    ({"type": "object", "properties": {}, "required": [], "additionalProperties": False,
      "patternProperties": {".*": {"type": "string"}}}, "keyword_unsupported:patternProperties"),
    ({"type": "string", "enum": ["a", 1]}, "enum_value_type_mismatch"),
    ({"type": "integer", "enum": [1, 2.5]}, "enum_value_type_mismatch"),
    ({"type": "string", "enum": []}, "enum_value_type_mismatch"),
])
def test_each_added_rule_names_what_breaks_it(field, rule):
    found = tools.strict_tool_violations([_tool("t", _strict_object({"field": field}))])
    assert f"t:/properties/field:{rule}" in found


def test_valid_enums_pass_including_integers_for_a_number():
    assert tools.strict_tool_violations([_tool("t", _strict_object({
        "a": {"type": "string", "enum": ["x", "y"]}, "b": {"type": "number", "enum": [1, 2.5]},
        "c": {"type": ["string", "null"], "enum": ["x", None]}}))]) == []


def test_duplicate_required_names_are_refused():
    parameters = _strict_object({"a": {"type": "string"}})
    parameters["required"] = ["a", "a"]
    assert tools.strict_tool_violations([_tool("t", parameters)]) == ["t::not_every_property_required"]


def test_nesting_depth_and_size_limits():
    deep = {"type": "string"}
    for _ in range(tools.MAX_NESTING_DEPTH + 1):
        deep = _strict_object({"child": deep})
    assert any(item.endswith(":nesting_depth_exceeded") for item in
               tools.strict_tool_violations([_tool("deep", deep)]))
    shallow = {"type": "string"}
    for _ in range(tools.MAX_NESTING_DEPTH - 1):
        shallow = _strict_object({"child": shallow})
    assert tools.strict_tool_violations([_tool("shallow", shallow)]) == []
    wide = _strict_object({f"p{i}": {"type": "string"} for i in range(tools.MAX_TOTAL_PROPERTIES + 1)})
    assert tools.strict_tool_violations([_tool("wide", wide)]) == ["wide::total_properties_exceeded"]
    enums = _strict_object({f"e{i}": {"type": "string", "enum": [str(j) for j in range(100)]} for i in range(11)})
    assert tools.strict_tool_violations([_tool("enums", enums)]) == ["enums::total_enum_values_exceeded"]


def test_a_snapshot_of_another_schema_version_is_refused(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({**tools.read_snapshot(), "schema_version": "artisan_openai_tool_schemas.v0"}))
    with pytest.raises(tools.ArtisanToolSchemaError, match="^artisan_tool_schema_snapshot_invalid$"):
        tools.read_snapshot(snapshot)


def test_an_unknown_name_in_a_tool_function_is_a_named_refusal(tmp_path):
    source = _fake_upstream(tmp_path / "fixed", pose_strict=False)
    path = source / "pipeline/harness_sdk.py"
    # A default argument naming something the extraction cannot see: evaluated at definition.
    path.write_text(path.read_text().replace("def _tool_schemas():", "def _tool_schemas(x=UNKNOWN_NAME):"))
    with pytest.raises(tools.ArtisanToolSchemaError, match="^artisan_tool_schema_function_unreadable:"):
        tools.upstream_tool_schemas(source)
    # An annotation naming it is not evaluated, as in the upstream module itself.
    path.write_text(path.read_text().replace("def _tool_schemas(x=UNKNOWN_NAME):",
                                             "def _tool_schemas() -> UnknownType:"))
    assert tools.upstream_tool_schemas(source)["coding"][0]["name"] == "read"
