"""Retain pinned arm/tool sources and a transparent, deterministic iiwa14 assembly.

Each resulting body is the exact pinned iiwa14/Robotiq2F85 simulation assembly.
Its existence does not establish publisher hardware or policy compatibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .immutable_directory_publication import publish_staged_immutable_directory

COMPOSITE_SCHEMA = "embodiment_kuka_composite_source_bundle.v1"
from . import deployment_embodiment_kuka_bridge as bridge
VARIANTS = {"kuka_lbr_iiwa_14_robotiq_2f85_sim": "kuka_lbr_iiwa_14"}
ENTRY = "assembly.urdf.xacro"
ARM_SOURCE_DIGEST = "sha256:df88cbae30a06109def7366679be5cac311786280b3988fa022646a08033a4e1"


def _require(value, reason):
    if not value:
        from .deployment_embodiment_assets import AssetPreparationError
        raise AssetPreparationError("asset_kuka_composition_" + reason)


def _sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def component_recipe(recipe: dict) -> dict:
    from .deployment_embodiment_tool_composition import component_recipe as tool_recipe
    composition, identity = recipe["composition"], recipe["embodiment_id"]
    _require(identity in VARIANTS and recipe["arm_count"] == 1
             and composition["schema_version"] == "iiwa14_robotiq2f85_sim_composition.v1"
             and composition["arm_embodiment_id"] == VARIANTS[identity]
             and composition["tool_component_id"] == "robotiq_2f85_direct_sim_mount"
             and composition["policy_compatibility_established"] is False
             and composition["publisher_hardware_match_established"] is False, "identity_invalid")
    expected = recipe["source"]
    _require(digest(expected) == ARM_SOURCE_DIGEST, "arm_pin_changed")
    _require(composition["mount"] == mount_descriptor(identity), "mount_changed")
    tool = tool_recipe({"embodiment_id": "universal_robots_ur5e_bimanual", "composition": {
        "schema_version": "ur5e_robotiq2f85_composition.v1", "arm_embodiment_id": "universal_robots_ur5e",
        "tool_component_id": "robotiq_2f85_ur_adapter", "tool_source": composition["tool_source"]}})
    _require(tool["source"]["revision"] == "a74d007d8f2f06dc6a503ad21038ba869d4999a6", "tool_pin_changed")
    return tool


def mount_descriptor(identity):
    _require(identity in VARIANTS, "identity_invalid")
    return {"parent_link": "attachment_site",
            "translation_m": [0, 0, 0], "rpy_rad": [0, 0, 0],
            "adapter_geometry_present": False, "mount_kind": "ideal_fixed_simulation_mount"}


def assembly_bytes(identity: str) -> bytes:
    _require(identity in VARIANTS, "identity_invalid")
    arm = f'  <xacro:include filename="{bridge.ARM_ENTRY}"/>'
    parent = mount_descriptor(identity)["parent_link"]
    # No adapter geometry or UR-only 11 mm spacer. The source attachment
    # frame remains link7 + 45 mm Z. This ideal mount is not hardware fit.
    return f'''<?xml version="1.0"?>
<robot name="blueprint_{identity}" xmlns:xacro="http://ros.org/wiki/xacro">
{arm}
  <xacro:include filename="$(find robotiq_description)/urdf/robotiq_2f_85_macro.urdf.xacro"/>
  <xacro:robotiq_gripper name="blueprint_simulation_gripper" prefix="" parent="{parent}" include_ros2_control="false">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:robotiq_gripper>
  <link name="grasp_link"/>
  <joint name="blueprint_grasp_frame" type="fixed">
    <parent link="robotiq_85_base_link"/><child link="grasp_link"/>
    <origin xyz="0 0 0.134" rpy="0 0 3.141592653589793"/>
  </joint>
</robot>
'''.encode()


def effective_description(recipe: dict, receipt: dict) -> dict:
    component_recipe(recipe)
    _require(receipt["schema_version"] == COMPOSITE_SCHEMA, "source_receipt_required")
    packages = {name: "arm/source" + ("/" + path if path != "." else "")
                for name, path in recipe["source"]["package_roots"].items()}
    packages["robotiq_description"] = "tool/source/robotiq_description"
    return {**recipe["source"], "entry_path": ENTRY, "format": "xacro", "package_roots": packages}


def _record(root: Path, recipe: dict) -> dict:
    from .deployment_embodiment_assets import verify_source_bundle, _verify_source_recipe, recipe_for
    _require(recipe["source"] == recipe_for(recipe["composition"]["arm_embodiment_id"])["source"], "arm_pin_changed")
    arm = verify_source_bundle(root / "source/arm", embodiment_id=recipe["composition"]["arm_embodiment_id"])
    tool = _verify_source_recipe(root / "source/tool", _recipe=component_recipe(recipe))
    generated, metadata = bridge.generate(root / "source/arm/source", arm)
    _require((root / "source" / bridge.ARM_ENTRY).read_bytes() == generated
             and json.loads((root / "source" / bridge.REPORT).read_text()) == metadata,
             "source_bridge_changed")
    _require((root / "source" / ENTRY).stat().st_size == len(assembly_bytes(recipe["embodiment_id"]))
             and (root / "source" / ENTRY).read_bytes() == assembly_bytes(recipe["embodiment_id"]), "generated_assembly_changed")
    files = []
    for path in sorted((root / "source").rglob("*")):
        _require(not path.is_symlink(), "symlink_refused")
        if path.is_file():
            files.append({"path": path.relative_to(root / "source").as_posix(),
                          "size_bytes": path.stat().st_size, "sha256": _sha(path)})
    expected = {ENTRY, bridge.ARM_ENTRY, bridge.REPORT, "arm/source-bundle.json", "tool/source-bundle.json"}
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
    _require(value.get("embodiment_id") in VARIANTS and embodiment_id in (None, value["embodiment_id"]), "identity_changed")
    recipe = recipe_for(value["embodiment_id"])
    _require(value == _record(root, recipe), "receipt_or_source_changed")
    return value


def materialize_composite(recipe: dict, destination: Path, *, get=None) -> dict:
    from .deployment_embodiment_assets import materialize_sources, _materialize_source_recipe, _refuse_symlink_ancestors
    _refuse_symlink_ancestors(destination)
    if destination.exists():
        return verify_composite(destination, embodiment_id=recipe["embodiment_id"])
    _require(destination.parent.is_dir(), "destination_parent_invalid")
    with tempfile.TemporaryDirectory(prefix=".kuka-tool-assembly-", dir=destination.parent) as temporary:
        staged = Path(temporary)
        (staged / "source").mkdir()
        materialize_sources(recipe["composition"]["arm_embodiment_id"], staged / "source/arm", get=get)
        _materialize_source_recipe(component_recipe(recipe), staged / "source/tool", get=get)
        arm = json.loads((staged / "source/arm/source-bundle.json").read_text())
        generated, metadata = bridge.generate(staged / "source/arm/source", arm)
        (staged / "source" / bridge.ARM_ENTRY).write_bytes(generated)
        (staged / "source" / bridge.REPORT).write_text(json.dumps(metadata, indent=2) + "\n")
        (staged / "source" / ENTRY).write_bytes(assembly_bytes(recipe["embodiment_id"]))
        receipt = _record(staged, recipe)
        (staged / "source-bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
        publish_staged_immutable_directory(staging=staged, destination=destination, manifest_name="source-bundle.json",
                                          output_exists_code="asset_kuka_composition_destination_appeared")
    return receipt
