"""Synthetic import-seam tests; actual public-source CPU receipts stay separate."""
from copy import deepcopy
import json
import xml.etree.ElementTree as XML

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_fr3_composition as composition
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from tests.test_deployment_embodiment_tool_composition import composed  # noqa: F401


@pytest.fixture
def fr3_composed(composed, tmp_path):  # noqa: F811 - imported pytest fixture
    ur_bundle, ur_record, tool_get = composed
    recipe = assets.recipe_for(composition.IDENTITY)
    robot = '<link name="fr3_link0"/>'
    for index in range(1, 8):
        robot += f'''<link name="fr3_link{index}"/><joint name="fr3_joint{index}" type="revolute">
          <parent link="fr3_link{index-1}"/><child link="fr3_link{index}"/><origin xyz="0 0 .1"/>
          <axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>'''
    robot += '<link name="fr3_link8"/><joint name="flange" type="fixed"><parent link="fr3_link7"/><child link="fr3_link8"/></joint>'
    robot += '<xacro:if value="$(arg hand)"><link name="fr3_hand"/></xacro:if>'
    files = {"LICENSE": b"Development-only Apache2 arm license fixture", "NOTICE": b"Development-only preserved arm notice",
             "robots/fr3/fr3.urdf.xacro": ('<robot xmlns:xacro="http://ros.org/wiki/xacro"><xacro:arg name="hand" default="true"/>' + robot + '</robot>').encode()}
    def get(url, *, maximum):
        if "PickNikRobotics" in url:
            return tool_get(url, maximum=maximum)
        if "/git/trees/" in url:
            return json.dumps({"truncated": False, "tree": [{"path": path, "type": "blob", "mode": "100644",
                "size": len(data), "sha": assets._git_blob_sha(data)} for path, data in files.items()]}).encode()
        return files[url.split("/" + recipe["source"]["revision"] + "/", 1)[1]]
    bundle = tmp_path / "fr3-sources"
    record = assets.materialize_sources(composition.IDENTITY, bundle, get=get)
    return bundle, record, ur_bundle, ur_record


def test_single_fr3_preserves_separate_licenses_and_old_ur_composite(fr3_composed):
    bundle, record, ur_bundle, ur_record = fr3_composed
    assert [row["identifier"] for row in record["licenses"]] == ["Apache-2.0", "BSD-3-Clause"]
    assert (bundle / "source/arm/source/NOTICE").is_file()
    assert record["source_arm_count"] == 1 and record["schema_version"] == composition.COMPOSITE_SCHEMA
    assert record["execution_authorized"] is False and record["policy_compatibility_established"] is False
    assumption = record["generated_assembly"]["composition"]
    assert assumption["publisher_hardware_match_established"] is False
    assert assumption["publisher_calibration_match_established"] is False
    assert set(assumption["simulation_assumptions"]) == {"robot", "tool_sku", "mount", "tcp", "cameras"}
    assert assets.verify_source_bundle(ur_bundle) == ur_record
    assert assets.materialize_sources(composition.IDENTITY, bundle, get=lambda **kwargs: pytest.fail("retry fetch")) == record


def test_seven_fr3_joints_one_real_robotiq_no_franka_hand_and_explicit_mount(fr3_composed, tmp_path):
    bundle, record, _, _ = fr3_composed
    folder = tmp_path / "configured"
    config = runtime.prepare_runtime_configuration(bundle, folder)
    manifest = config["manifest_fields"]
    assert manifest["embodiment_id"] == composition.IDENTITY and manifest["form"] == "fixed_base_arm"
    assert [row["name"] for row in manifest["joints"]] == [f"fr3_joint{n}" for n in range(1, 8)]
    assert [arm["name"] for arm in manifest["arms"]] == ["arm"]
    assert len(manifest["grippers"]) == 1 and manifest["grippers"][0]["max_width_m"] == .085
    assert manifest["arms"][0]["tool_from_tcp"]["position_m"] == pytest.approx([0, 0, .134])
    assert manifest["arms"][0]["tool_from_tcp"]["orientation_xyzw"] == pytest.approx([0, 0, 1, 0], abs=1e-14)
    assert [c["name"] for c in manifest["cameras"]] == ["overhead", "arm_wrist"]
    assert config["source_bundle_digest"] == record["bundle_digest"]
    assert "simulation_assumption_direct_zero_offset_flange_mount" in config["source_limitations"]
    tree = XML.parse(folder / "robot.urdf").getroot()
    assert tree.find("link[@name='fr3_hand']") is None
    mount = next(j for j in tree.findall("joint") if j.find("child").get("link") == "robotiq_85_base_link")
    assert mount.get("type") == "fixed" and mount.find("parent").get("link") == "fr3_link8"
    assert mount.find("origin").get("xyz") == "0 0 0" and mount.find("origin").get("rpy") == "0 0 0"
    assert runtime.verify_runtime_configuration(folder) == config


def test_fr3_configuration_reaches_usd_import_and_manifest_binding(fr3_composed, tmp_path, monkeypatch):
    from tests.test_deployment_embodiment_runtime_configuration import _fake_sdk_from_urdf
    bundle, _, _, _ = fr3_composed
    configured, prepared = tmp_path / "configured", tmp_path / "prepared"
    config = runtime.prepare_runtime_configuration(bundle, configured)
    _fake_sdk_from_urdf(monkeypatch)
    result = importer.import_source_asset(configured, prepared, fix_base=True, station_parameters=config["station_parameters"])
    binding = runtime.bind_prepared_configuration(configured, prepared, "https://models.example/development-only.usda")
    assert result["composed_arm_count"] == 1
    assert len(binding["robot_embodiment"]["joints"]) == 7
    assert binding["robot_embodiment"]["embodiment_id"] == composition.IDENTITY
    assert binding["execution_authorized"] is False
    assert importer.verify_prepared_asset_directory(prepared) == result


@pytest.mark.parametrize("path", ["arm/source/NOTICE", "tool/source/LICENSE", composition.ENTRY, "unrecorded"])
def test_changed_sources_license_notice_assembly_and_extra_files_refuse_reuse(fr3_composed, path):
    bundle, _, _, _ = fr3_composed
    target = bundle / "source" / path
    target.parent.chmod(0o755)
    target.write_bytes(b"changed-development-only")
    with pytest.raises(assets.AssetPreparationError):
        assets.verify_source_bundle(bundle)


@pytest.mark.parametrize("defect", ["arm", "tool", "identity", "claim"])
def test_wrong_source_or_publisher_hardware_claim_cannot_reuse_variant(defect):
    recipe = deepcopy(assets.recipe_for(composition.IDENTITY))
    if defect == "arm":
        recipe["source"]["import_arguments"]["hand"] = "true"
    elif defect == "tool":
        recipe["composition"]["tool_source"]["revision"] = "f" * 40
    elif defect == "identity":
        recipe["embodiment_id"] = "franka_research_3_bimanual"
    else:
        recipe["composition"]["publisher_hardware_match_established"] = True
    with pytest.raises(assets.AssetPreparationError):
        composition.component_recipe(recipe)
