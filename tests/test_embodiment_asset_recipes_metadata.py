"""Public source recipes must preserve exact bodies and unavailable facts."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re


DATA = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline/_catalog_data"


def _recipes():
    value = json.loads((DATA / "embodiment_asset_recipes.v1.json").read_text())
    assert value["schema_version"] == "embodiment_asset_recipes.v1"
    assert value["claim_ceiling"] == "source_asset_not_qualified"
    return {row["embodiment_id"]: row for row in value["recipes"]}


def test_catalog_and_existing_compatibility_identities_have_explicit_source_state():
    rows = _recipes()
    catalog = json.loads((DATA / "deployment_embodiment_support.v1.json").read_text())
    assert set(rows) == {row["embodiment_id"] for row in catalog["embodiments"]} | {
        "droid_franka_robotiq_2f85", "unitree_g1_dex3_v1", "boston_dynamics_atlas",
        "universal_robots_ur10e", "fanuc_crx_10ia", "abb_gofa_crb15000",
        "kinova_gen3", "ufactory_xarm7",
    }
    assert len(rows) == 24
    for row in rows.values():
        assert row["status"] in {"source_available", "source_unavailable", "existing_runtime"}
        assert bool(row["source"]) == (row["status"] == "source_available")
        assert row["reason"]


def test_public_sources_use_immutable_commits_not_invented_compiled_model_digests():
    for row in _recipes().values():
        source = row["source"]
        if source is None:
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        assert re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repository_url"])
        assert source["format"] in {"urdf", "xacro", "mjcf"}
        assert source["source_arm_count"] in {1, 2}
        assert source["source_arm_count"] <= row["arm_count"]
        assert source["license_id"] in {"MIT", "Apache-2.0", "BSD-3-Clause"}
        assert source["license_url"] == (
            f'{source["repository_url"]}/blob/{source["revision"]}/{source["license_path"]}'
        )
        for key in ("entry_path", "license_path"):
            path = PurePosixPath(source[key])
            assert not path.is_absolute() and ".." not in path.parts
        assert source["license_path"] in source["include_paths"]
        assert any(source["entry_path"] == path or source["entry_path"].startswith(path + "/")
                   for path in source["include_paths"])
        assert all(isinstance(value, str) for value in source["import_arguments"].values())
        assert not {"model_digest", "checkpoint_digest", "qualified", "execution_authorized"} & set(source)


def test_close_names_never_select_an_unverified_variant_or_collapse_existing_droid():
    rows = _recipes()
    assert rows["droid_franka_robotiq_2f85"]["status"] == "existing_runtime"
    gp20 = rows["yaskawa_gp20"]["source"]
    assert gp20["entry_path"] == "motoman_gp25_support/urdf/gp25_20.xacro"
    assert gp20["revision"] == "846fbbc57c60223f12bf4795ea6404982d57709e"
    assert gp20["license_path"] == "motoman_gp25_support/package.xml"
    assert gp20["package_roots"]["motoman_resources"] == "motoman_resources"
    assert "motoman_gp25_support/meshes/gp25_20" in gp20["include_paths"]
    assert "motoman_gp25_support/meshes/gp25" in gp20["include_paths"]
    assert "GP20HL is a different variant" in rows["yaskawa_gp20"]["reason"]
    for identity in ("standard_bots_ro1", "standard_bots_core", "abb_gofa_crb15000", "boston_dynamics_atlas"):
        assert rows[identity]["source"] is None
    assert "/crx10ia_l." in rows["fanuc_crx_10ia_l"]["source"]["entry_path"]
    assert "/crx10ia." in rows["fanuc_crx_10ia"]["source"]["entry_path"]
    assert rows["universal_robots_ur5e"]["source"]["import_arguments"]["ur_type"] == "ur5e"
    assert rows["universal_robots_ur10e"]["source"]["import_arguments"]["ur_type"] == "ur10e"


def test_source_specific_physics_tools_and_camera_gaps_remain_visible():
    rows = _recipes()
    assert "inertial_parameters_not_present_in_gp12_macro" in rows["yaskawa_gp12"]["missing"]
    assert "inertial_parameters_not_present_in_gp25_20_macro" in rows["yaskawa_gp20"]["missing"]
    assert "confirm_exact_gp25_20_variant_YR_1_06VX25_20_A0" in rows["yaskawa_gp20"]["composition_required"]
    assert "left_and_right_gripper_urdf_joints_are_fixed" in rows["almond_axol"]["missing"]
    yam = rows["i2rt_yam_bimanual"]
    assert yam["source"]["source_arm_count"] == 2
    assert rows["trlc_dk1_bimanual"]["source"]["source_arm_count"] == 1
    assert "/station/yam_station_linear_4310_d405/" in yam["source"]["entry_path"]
    assert "camera_intrinsics_and_site_calibration" in yam["missing"]
    assert "independent_tool_action_mapping" in yam["missing"]
    xarm = rows["ufactory_xarm7_bimanual"]["source"]["import_arguments"]
    assert xarm["dof_1"] == xarm["dof_2"] == "7"
    assert xarm["add_gripper_1"] == xarm["add_gripper_2"] == "true"
    assert "reviewed_checkpoint_and_sonic_terms" in rows["unitree_g1_dex3_v1"]["missing"]
