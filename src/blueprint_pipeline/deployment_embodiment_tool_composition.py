"""Retain pinned arm/tool sources and a transparent, deterministic UR assembly.

The resulting body is a generic public UR5e/Robotiq2F85 simulation assembly.
Its existence does not establish Dream hardware or policy compatibility.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
import tempfile

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .immutable_directory_publication import publish_staged_immutable_directory

COMPOSITE_SCHEMA = "embodiment_composite_source_bundle.v1"
IDENTITY = "universal_robots_ur5e_bimanual"
ENTRY = "assembly.urdf.xacro"


def _require(value, reason):
    if not value:
        from .deployment_embodiment_assets import AssetPreparationError
        raise AssetPreparationError("asset_tool_composition_" + reason)


def _sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def component_recipe(recipe: dict) -> dict:
    composition = recipe["composition"]
    source = composition["tool_source"]
    from .deployment_embodiment_assets import _path
    _require(source["repository_url"] == "https://github.com/PickNikRobotics/ros2_robotiq_gripper"
             and re.fullmatch(r"[0-9a-f]{40}", source["revision"])
             and source["license_id"] == "BSD-3-Clause" and source["license_path"] == "LICENSE"
             and source["license_url"] == source["repository_url"] + "/blob/" + source["revision"] + "/LICENSE"
             and source["format"] == "xacro" and source["source_arm_count"] == 0
             and source["import_arguments"] == {}
             and source["package_roots"] == {"robotiq_description": "robotiq_description"}, "tool_pin_or_license_invalid")
    _require(isinstance(source["include_paths"], list) and 1 <= len(source["include_paths"]) <= 64
             and source["license_path"] in source["include_paths"]
             and source["entry_path"] in source["include_paths"], "tool_source_closure_invalid")
    for path in source["include_paths"]:
        _path(path)
    _require(recipe["embodiment_id"] == IDENTITY and composition["schema_version"] == "ur5e_robotiq2f85_composition.v1"
             and composition["arm_embodiment_id"] == "universal_robots_ur5e"
             and composition["tool_component_id"] == "robotiq_2f85_ur_adapter", "identity_invalid")
    return {"embodiment_id": composition["tool_component_id"], "arm_count": 0,
            "status": "source_available", "source": deepcopy(composition["tool_source"]),
            "composition_required": [], "missing": [], "reason": "Public tool component; no robot or policy authority."}


def assembly_bytes() -> bytes:
    # UR adapter geometry/mount come directly from the retained tool macro.
    # The grasp frame is an explicit simulation convention from PickNik's
    # pinned public UR example, retained as tcp_reference in the recipe.
    return b'''<?xml version="1.0"?>
<robot name="blueprint_public_ur5e_robotiq2f85" xmlns:xacro="http://ros.org/wiki/xacro">
  <xacro:include filename="$(find ur_description)/urdf/ur.urdf.xacro"/>
  <xacro:include filename="$(find robotiq_description)/urdf/ur_to_robotiq_adapter.urdf.xacro"/>
  <xacro:include filename="$(find robotiq_description)/urdf/robotiq_2f_85_macro.urdf.xacro"/>
  <xacro:ur_to_robotiq prefix="" parent="tool0" child="gripper_mount_link" rotation="0"/>
  <xacro:robotiq_gripper name="blueprint_simulation_gripper" prefix="" parent="gripper_mount_link" include_ros2_control="false">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:robotiq_gripper>
  <link name="grasp_link"/>
  <joint name="blueprint_grasp_frame" type="fixed">
    <parent link="robotiq_85_base_link"/><child link="grasp_link"/>
    <origin xyz="0 0 0.134" rpy="0 0 3.141592653589793"/>
  </joint>
</robot>
'''


def effective_description(recipe: dict, receipt: dict) -> dict:
    component_recipe(recipe)
    _require(receipt["schema_version"] == COMPOSITE_SCHEMA, "source_receipt_required")
    return {**recipe["source"], "entry_path": ENTRY, "format": "xacro",
            "package_roots": {"ur_description": "arm/source", "robotiq_description": "tool/source/robotiq_description"}}


def _record(root: Path, recipe: dict) -> dict:
    from .deployment_embodiment_assets import verify_source_bundle, _verify_source_recipe, recipe_for
    _require(recipe["source"] == recipe_for(recipe["composition"]["arm_embodiment_id"])["source"], "arm_pin_changed")
    arm = verify_source_bundle(root / "source/arm", embodiment_id=recipe["composition"]["arm_embodiment_id"])
    tool = _verify_source_recipe(root / "source/tool", _recipe=component_recipe(recipe))
    _require((root / "source" / ENTRY).stat().st_size == len(assembly_bytes())
             and (root / "source" / ENTRY).read_bytes() == assembly_bytes(), "generated_assembly_changed")
    files = []
    for path in sorted((root / "source").rglob("*")):
        _require(not path.is_symlink(), "symlink_refused")
        if path.is_file():
            files.append({"path": path.relative_to(root / "source").as_posix(),
                          "size_bytes": path.stat().st_size, "sha256": _sha(path)})
    expected = {ENTRY, "arm/source-bundle.json", "tool/source-bundle.json"}
    for name, source in (("arm", arm), ("tool", tool)):
        expected.update(name + "/source/" + row["path"] for row in source["files"])
    _require({row["path"] for row in files} == expected, "unrecorded_source_file")
    _require(len(files) <= 2500 and sum(row["size_bytes"] for row in files) <= 512 * 1024**2, "source_bound_exceeded")
    licenses = [{**value["license"], "path": name + "/source/" + value["license"]["path"]}
                for name, value in (("arm", arm), ("tool", tool))]
    record = {"schema_version": COMPOSITE_SCHEMA, "claim_ceiling": "source_asset_not_qualified",
        "embodiment_id": recipe["embodiment_id"], "recipe_digest": digest(recipe),
        "repository_url": arm["repository_url"], "revision": arm["revision"], "entry_path": ENTRY, "format": "xacro",
        "source_arm_count": 1, "files": files, "license": licenses[0], "licenses": licenses,
        "components": [{"name": name, "repository_url": value["repository_url"], "revision": value["revision"],
                        "bundle_digest": value["bundle_digest"], "recipe_digest": value["recipe_digest"]}
                       for name, value in (("arm", arm), ("tool", tool))],
        "generated_assembly": {"path": ENTRY, "sha256": _sha(root / "source" / ENTRY),
                               "generator_sha256": _sha(Path(__file__)), "composition": recipe["composition"]},
        "composition_required": recipe["composition_required"], "missing": recipe["missing"],
        "execution_authorized": False, "policy_compatibility_established": False}
    record["bundle_digest"] = digest(record, digest_field="bundle_digest")
    return record


def verify_composite(root: Path, *, embodiment_id=None) -> dict:
    from .deployment_embodiment_assets import recipe_for
    value = json.loads((root / "source-bundle.json").read_text())
    _require(value.get("embodiment_id") == (embodiment_id or IDENTITY), "identity_changed")
    recipe = recipe_for(value["embodiment_id"])
    _require(value == _record(root, recipe), "receipt_or_source_changed")
    return value


def materialize_composite(recipe: dict, destination: Path, *, get=None) -> dict:
    from .deployment_embodiment_assets import materialize_sources, _materialize_source_recipe, _refuse_symlink_ancestors
    _refuse_symlink_ancestors(destination)
    if destination.exists():
        return verify_composite(destination, embodiment_id=recipe["embodiment_id"])
    _require(destination.parent.is_dir(), "destination_parent_invalid")
    with tempfile.TemporaryDirectory(prefix=".tool-assembly-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        (staged / "source").mkdir()
        materialize_sources(recipe["composition"]["arm_embodiment_id"], staged / "source/arm", get=get)
        _materialize_source_recipe(component_recipe(recipe), staged / "source/tool", get=get)
        (staged / "source" / ENTRY).write_bytes(assembly_bytes())
        receipt = _record(staged, recipe)
        (staged / "source-bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
        publish_staged_immutable_directory(staging=staged, destination=destination, manifest_name="source-bundle.json",
                                          output_exists_code="asset_tool_composition_destination_appeared")
    return receipt
