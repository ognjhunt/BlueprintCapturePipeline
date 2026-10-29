"""Development-only multi-source contracts; native source CPU proof is external."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_tool_composition as composition
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_station as station


@pytest.fixture
def composed(tmp_path, monkeypatch):
    recipes = {r["embodiment_id"]: r for r in assets.asset_recipes()["recipes"]}
    recipe = recipes[composition.IDENTITY]
    monkeypatch.setattr(assets, "recipe_for", lambda identity: deepcopy(recipes[identity]))
    monkeypatch.setattr(runtime, "recipe_for", assets.recipe_for)
    monkeypatch.setattr(importer, "recipe_for", assets.recipe_for)
    monkeypatch.setattr(station, "recipe_for", assets.recipe_for)
    links = '<link name="base_link_inertia"/>'
    for number in range(1, 7):
        parent = "base_link_inertia" if number == 1 else f"link{number-1}"
        links += f'''<link name="link{number}"/><joint name="joint{number}" type="revolute">
          <parent link="{parent}"/><child link="link{number}"/><origin xyz="0 0 .1"/>
          <axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>'''
    links += '<link name="tool0"/><joint name="flange" type="fixed"><parent link="link6"/><child link="tool0"/></joint>'
    arm = {"LICENSE": b"Development-only BSD3 arm fixture", "urdf/ur.urdf.xacro": (
        '<robot xmlns:xacro="http://ros.org/wiki/xacro">'+links+'</robot>').encode()}
    tool = {"LICENSE": b"Development-only BSD3 tool fixture",
        "robotiq_description/urdf/ur_to_robotiq_adapter.urdf.xacro": b'''<robot xmlns:xacro="http://ros.org/wiki/xacro">
        <xacro:macro name="ur_to_robotiq" params="prefix parent child rotation">
          <link name="${child}"/><joint name="adapter" type="fixed"><parent link="${parent}"/>
          <child link="${child}"/><origin xyz="0 0 .011" rpy="0 0 ${rotation}"/></joint>
        </xacro:macro></robot>''',
        "robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro": b'''<robot xmlns:xacro="http://ros.org/wiki/xacro">
        <xacro:macro name="robotiq_gripper" params="name prefix parent include_ros2_control *origin">
          <link name="robotiq_85_base_link"/>
          <joint name="tool_mount" type="fixed"><parent link="${parent}"/>
            <child link="robotiq_85_base_link"/><xacro:insert_block name="origin"/></joint>
          <link name="robotiq_85_left_finger_tip_link"/><link name="robotiq_85_right_finger_tip_link"/>
          <joint name="robotiq_85_left_knuckle_joint" type="revolute"><parent link="robotiq_85_base_link"/>
            <child link="robotiq_85_left_finger_tip_link"/><axis xyz="0 1 0"/><limit lower="0" upper=".8" velocity=".5" effort="50"/></joint>
          <joint name="right_finger" type="revolute"><parent link="robotiq_85_base_link"/>
            <child link="robotiq_85_right_finger_tip_link"/><axis xyz="0 1 0"/><limit lower="-.8" upper="0" velocity=".5" effort="50"/>
            <mimic joint="robotiq_85_left_knuckle_joint" multiplier="-1"/></joint>
        </xacro:macro></robot>'''}
    def get(url, *, maximum):
        source = recipe["composition"]["tool_source"] if "PickNikRobotics" in url else recipe["source"]
        files = tool if "PickNikRobotics" in url else arm
        if "/git/trees/" in url:
            rows = [{"path": name, "type": "blob", "mode": "100644", "size": len(value),
                     "sha": hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()}
                    for name, value in files.items()]
            return json.dumps({"truncated": False, "tree": rows}).encode()
        return files[url.split("/" + source["revision"] + "/", 1)[1]]
    bundle = tmp_path / "sources"
    record = assets.materialize_sources(composition.IDENTITY, bundle, get=get)
    return bundle, record, get


def test_sources_not_flattened_or_relicensed_and_retries_keep_exact_bytes(composed):
    bundle, record, get = composed
    assert record["schema_version"] == composition.COMPOSITE_SCHEMA
    assert [row["name"] for row in record["components"]] == ["arm", "tool"]
    assert len(record["licenses"]) == 2
    assert record["components"][0]["revision"] == "d4042c1f5123c86645cc5dbd650b4f461f697ee9"
    assert record["components"][1]["revision"] == "a74d007d8f2f06dc6a503ad21038ba869d4999a6"
    assert (bundle / "source/arm/source/LICENSE").read_bytes() != (bundle / "source/tool/source/LICENSE").read_bytes()
    assert assets.materialize_sources(composition.IDENTITY, bundle, get=lambda **k: pytest.fail("network retry")) == record
    assert record["execution_authorized"] is False and record["policy_compatibility_established"] is False
    assert record["generated_assembly"]["composition"]["dream_hardware_match_established"] is False


@pytest.mark.parametrize("path", ["arm/source/LICENSE", "tool/source/LICENSE", "assembly.urdf.xacro",
    "tool/source/robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro", "unexpected.urdf"])
def test_changed_either_publisher_license_assembly_or_extra_file_is_rejected(composed, path):
    bundle, _, _ = composed
    target = bundle / "source" / path
    target.parent.chmod(0o755)
    target.write_bytes(b"development-only changed bytes")
    with pytest.raises(assets.AssetPreparationError):
        assets.verify_source_bundle(bundle)


def test_two_instances_keep_independent_tools_exact_mount_and_tcp(composed, tmp_path):
    bundle, record, _ = composed
    config = runtime.prepare_runtime_configuration(bundle, tmp_path / "configured")
    fields = config["manifest_fields"]
    assert len(fields["joints"]) == 12 and len(fields["grippers"]) == 2
    assert config["source_bundle_digest"] == record["bundle_digest"]
    names = set()
    for side, arm, gripper in zip(("left", "right"), fields["arms"], fields["grippers"], strict=True):
        assert arm["tool_from_tcp"]["position_m"] == pytest.approx([0, 0, .134])
        assert arm["tool_from_tcp"]["orientation_xyzw"] == pytest.approx([0, 0, 1, 0], abs=1e-14)
        assert len(arm["joint_names"]) == 6
        assert gripper["max_width_m"] == .085
        assert gripper["joints"] == [
            {"name": side + "_robotiq_85_left_knuckle_joint", "open": 0., "closed": .7929},
            {"name": side + "_right_finger", "open": 0., "closed": -.7929}]
        assert not names.intersection(j["name"] for j in gripper["joints"])
        names.update(j["name"] for j in gripper["joints"])
    assert runtime.verify_runtime_configuration(tmp_path / "configured") == config
    assert "dream_hardware_and_policy_match_not_established" in config["source_limitations"]


def test_composed_sources_use_existing_usd_import_and_manifest_binding(composed, tmp_path, monkeypatch):
    from tests.test_deployment_embodiment_runtime_configuration import _fake_sdk_from_urdf
    bundle, _, _ = composed
    configured = tmp_path / "configured"
    config = runtime.prepare_runtime_configuration(bundle, configured)
    _fake_sdk_from_urdf(monkeypatch)
    prepared = tmp_path / "prepared"
    imported = importer.import_source_asset(configured, prepared, fix_base=True, station_parameters=config["station_parameters"])
    binding = runtime.bind_prepared_configuration(configured, prepared, "https://models.example.test/development-only.usda")
    assert imported["composed_arm_count"] == 2 and binding["execution_authorized"] is False
    assert len(binding["robot_embodiment"]["grippers"]) == 2
    assert binding["robot_embodiment"]["embodiment_id"] == composition.IDENTITY
    assert importer.verify_prepared_asset_directory(prepared) == imported


def test_composite_component_license_and_pin_cannot_be_unbound():
    recipe = assets.recipe_for(composition.IDENTITY)
    recipe["composition"]["tool_source"]["license_id"] = "unknown"
    with pytest.raises(assets.AssetPreparationError, match="license_invalid"):
        composition.component_recipe(recipe)
