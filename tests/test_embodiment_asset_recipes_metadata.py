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
        "kinova_gen3", "ufactory_xarm7", "franka_research_3_robotiq_2f85_sim",
        "abb_gofa_crb15000_5_95", "abb_gofa_crb15000_12_127",
        "universal_robots_ur5e_robotiq_2f85_sim", "universal_robots_ur10e_robotiq_2f85_sim",
        'kuka_lbr_iiwa_14_robotiq_2f85_sim',
        'yaskawa_gp12_robotiq_2f85_sim', 'yaskawa_gp20_robotiq_2f85_sim',
        'fanuc_crx_10ia_robotiq_2f85_sim',
        'fanuc_crx_10ia_l_robotiq_2f85_sim',
        'fanuc_m710ic_45m_robotiq_2f85_sim',
        'abb_gofa_crb15000_5_95_robotiq_2f85_sim',
        'abb_gofa_crb15000_12_127_robotiq_2f85_sim',
        "fanuc_m10id_12", "fanuc_m10id_12_robotiq_2f85_sim", "yaskawa_hc20", "yaskawa_hc20_robotiq_2f85_sim",
        "agile_robots_diana_7", "agile_robots_diana_7_robotiq_2f85_sim", "kinova_jaco2_j2n6s300",
        "kuka_kr20_r3100", "kuka_kr30_r2100", "kuka_kr50_r2100", "kuka_kr50_r2500", "kuka_kr70_r2100",
    }
    assert len(rows) == 49
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
    for identity in ("standard_bots_ro1", "standard_bots_core"):
        assert rows[identity]["status"] == "source_unavailable"
        assert "referenced_standard_bots_description_mesh_closure" in rows[identity]["missing"]
        assert "explicit_asset_reuse_license" in rows[identity]["missing"]
        assert "8a2164e041d3a4e75b2eeffe6351f3c89b292f5c" in rows[identity]["reason"]
    assert "/crx10ia_l." in rows["fanuc_crx_10ia_l"]["source"]["entry_path"]
    assert "/crx10ia." in rows["fanuc_crx_10ia"]["source"]["entry_path"]
    assert rows["universal_robots_ur5e"]["source"]["import_arguments"]["ur_type"] == "ur5e"
    assert rows["universal_robots_ur10e"]["source"]["import_arguments"]["ur_type"] == "ur10e"


def test_source_specific_physics_tools_and_camera_gaps_remain_visible():
    rows = _recipes()
    assert "inertial_parameters_not_present_in_gp12_macro" in rows["yaskawa_gp12"]["missing"]
    assert "inertial_parameters_not_present_in_gp25_20_macro" in rows["yaskawa_gp20"]["missing"]
    assert "confirm_exact_gp25_20_variant_YR_1_06VX25_20_A0" in rows["yaskawa_gp20"]["composition_required"]
    axol = rows["almond_axol"]
    assert axol["source"]["entry_path"].endswith("/axol_standalone.urdf")
    assert axol["source"]["revision"] == "3ccc69cf3adaea09845456148d6a4f16ef059f27"
    assert axol["source"]["package_roots"] == {}
    for mesh_directory in ("jelly", "standalone"):
        assert f"almond_axol/kinematics/urdf/meshes/{mesh_directory}" in axol["source"]["include_paths"]
    assert "left_and_right_gripper_urdf_joints_are_fixed" not in axol["missing"]
    assert "source_estimated_non_arm_masses_and_inertias" in axol["missing"]
    assert "independent_finger_pair_commands_and_mimic_mapping" in axol["composition_required"]
    assert "apply_source_custom_drive_gains_after_import" in axol["composition_required"]
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
