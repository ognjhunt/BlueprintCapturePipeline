"""Pinned single FR3 + Robotiq2F85 simulation assembly, not publisher hardware.

OpenWAM identifies FR3 and a Robotiq parallel gripper, but does not publish the
SKU, adapter, TCP or camera calibration. The direct flange attachment and
nominal grasp reference below are declared simulation choices. The original
publisher files and both licenses remain separate and unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .deployment_embodiment_tool_composition import _sha
from .immutable_directory_publication import publish_staged_immutable_directory

IDENTITY = "franka_research_3_robotiq_2f85_sim"
COMPOSITE_SCHEMA = "embodiment_fr3_robotiq_source_bundle.v1"
ENTRY = "fr3_robotiq_sim.urdf.xacro"


def _require(condition, reason):
    if not condition:
        from .deployment_embodiment_assets import AssetPreparationError
        raise AssetPreparationError("asset_fr3_composition_" + reason)


def component_recipe(recipe: dict) -> dict:
    """Validate this exact composition and reuse the existing licensed tool contract."""
    from .deployment_embodiment_tool_composition import component_recipe as tool_recipe

    composition, source = recipe["composition"], recipe["source"]
    _require(recipe["embodiment_id"] == IDENTITY and recipe["arm_count"] == 1
             and composition["schema_version"] == "fr3_robotiq2f85_sim_composition.v1"
             and composition["tool_component_id"] == "robotiq_2f85_source_component"
             and all(composition.get(key) is False for key in ("publisher_hardware_match_established",
                 "publisher_calibration_match_established", "policy_compatibility_established"))
             and isinstance(composition.get("simulation_assumptions"), dict)
             and set(composition["simulation_assumptions"]) == {"robot", "tool_sku", "mount", "tcp", "cameras"},
             "identity_or_assumptions_invalid")
    _require(source["repository_url"] == "https://github.com/frankarobotics/franka_description"
             and source["revision"] == "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
             and source["entry_path"] == "robots/fr3/fr3.urdf.xacro" and source["format"] == "xacro"
             and source["license_id"] == "Apache-2.0" and source["license_path"] == "LICENSE"
             and "NOTICE" in source["include_paths"]
             and source["import_arguments"] == {"robot_type": "fr3", "hand": "false", "ee_id": "none"}
             and source["package_roots"] == {"franka_description": "."} and source["source_arm_count"] == 1,
             "arm_source_invalid")
    # Validate the identical public component without changing the existing
    # UR composite module (its retained generator digest must stay stable).
    tool = tool_recipe({"embodiment_id": "universal_robots_ur5e_bimanual", "composition": {
        "schema_version": "ur5e_robotiq2f85_composition.v1", "arm_embodiment_id": "universal_robots_ur5e",
        "tool_component_id": "robotiq_2f85_ur_adapter", "tool_source": composition["tool_source"]}})
    _require(tool["source"]["revision"] == "a74d007d8f2f06dc6a503ad21038ba869d4999a6", "tool_source_changed")
    tool["embodiment_id"] = composition["tool_component_id"]
    return tool


def arm_component_recipe(recipe: dict) -> dict:
    component_recipe(recipe)
    return {"embodiment_id": "franka_research_3_bare_source_component", "arm_count": 1,
            "status": "source_available", "source": deepcopy(recipe["source"]),
            "composition_required": [], "missing": [], "reason": "Pinned bare FR3 source component, not a policy."}


def assembly_bytes() -> bytes:
    return b'''<?xml version="1.0"?>
<robot name="blueprint_fr3_robotiq2f85_sim" xmlns:xacro="http://ros.org/wiki/xacro">
  <xacro:include filename="$(find franka_description)/robots/fr3/fr3.urdf.xacro"/>
  <xacro:include filename="$(find robotiq_description)/urdf/robotiq_2f_85_macro.urdf.xacro"/>
  <xacro:robotiq_gripper name="blueprint_simulation_gripper" prefix="" parent="fr3_link8" include_ros2_control="false">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:robotiq_gripper>
  <link name="grasp_link"/>
  <joint name="blueprint_nominal_grasp_frame" type="fixed">
    <parent link="robotiq_85_base_link"/><child link="grasp_link"/>
    <origin xyz="0 0 0.134" rpy="0 0 3.141592653589793"/>
  </joint>
</robot>
'''


def effective_description(recipe: dict, receipt: dict) -> dict:
    component_recipe(recipe)
    _require(receipt["schema_version"] == COMPOSITE_SCHEMA, "source_receipt_required")
    return {**recipe["source"], "entry_path": ENTRY, "package_roots": {
        "franka_description": "arm/source", "robotiq_description": "tool/source/robotiq_description"}}


def _record(root: Path, recipe: dict) -> dict:
    from .deployment_embodiment_assets import _verify_source_recipe
    arm = _verify_source_recipe(root / "source/arm", _recipe=arm_component_recipe(recipe))
    tool = _verify_source_recipe(root / "source/tool", _recipe=component_recipe(recipe))
    assembly = root / "source" / ENTRY
    _require(assembly.stat().st_size == len(assembly_bytes()) and assembly.read_bytes() == assembly_bytes(),
             "assembly_changed")
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
        "embodiment_id": IDENTITY, "recipe_digest": digest(recipe),
        "repository_url": arm["repository_url"], "revision": arm["revision"], "entry_path": ENTRY, "format": "xacro",
        "source_arm_count": 1, "files": files, "license": licenses[0], "licenses": licenses,
        "components": [{"name": name, "repository_url": value["repository_url"], "revision": value["revision"],
                        "bundle_digest": value["bundle_digest"], "recipe_digest": value["recipe_digest"]}
                       for name, value in (("arm", arm), ("tool", tool))],
        "generated_assembly": {"path": ENTRY, "sha256": _sha(assembly), "generator_sha256": _sha(Path(__file__)),
                               "composition": recipe["composition"]},
        "composition_required": recipe["composition_required"], "missing": recipe["missing"],
        "execution_authorized": False, "policy_compatibility_established": False}
    record["bundle_digest"] = digest(record, digest_field="bundle_digest")
    return record


def verify_composite(root: Path, *, embodiment_id=None) -> dict:
    from .deployment_embodiment_assets import recipe_for
    value = json.loads((root / "source-bundle.json").read_text())
    _require(value.get("embodiment_id") == IDENTITY and embodiment_id in (None, IDENTITY), "identity_changed")
    _require(value == _record(root, recipe_for(IDENTITY)), "receipt_or_source_changed")
    return value


def materialize_composite(recipe: dict, destination: Path, *, get=None) -> dict:
    from .deployment_embodiment_assets import _materialize_source_recipe, _refuse_symlink_ancestors
    _refuse_symlink_ancestors(destination)
    if destination.exists():
        return verify_composite(destination, embodiment_id=recipe["embodiment_id"])
    _require(destination.parent.is_dir(), "destination_parent_invalid")
    with tempfile.TemporaryDirectory(prefix=".fr3-tool-assembly-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        (staged / "source").mkdir()
        for name, component in (("arm", arm_component_recipe(recipe)), ("tool", component_recipe(recipe))):
            _materialize_source_recipe(component, staged / "source" / name, get=get)
        (staged / "source" / ENTRY).write_bytes(assembly_bytes())
        receipt = _record(staged, recipe)
        (staged / "source-bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
        publish_staged_immutable_directory(staging=staged, destination=destination, manifest_name="source-bundle.json",
                                          output_exists_code="asset_fr3_composition_destination_appeared")
    return receipt
