"""Research metadata must preserve exact variants and the integration boundary.

Covers: src/blueprint_pipeline/_catalog_data/deployment_embodiment_support.v1.json
"""

from __future__ import annotations

from blueprint_pipeline import task_evaluation_embodiment_registry as registry
from blueprint_pipeline.task_evaluation_policy_candidate_registry import (
    BIMANUAL_EEF_POSE_ACTION,
    DREAM1_NATIVE_JOINT_ACTION,
)


def _rows():
    return {
        row["embodiment_id"]: row
        for row in registry.deployment_embodiment_support_catalog()["embodiments"]
    }


def test_catalog_covers_exact_requested_variants_without_promoting_them() -> None:
    rows = _rows()
    assert set(rows) == {
        "universal_robots_ur5e", "universal_robots_ur5e_bimanual",
        "trlc_dk1_bimanual", "almond_axol", "fanuc_crx_10ia_l", "fanuc_m710ic_45m",
        "yaskawa_gp12", "yaskawa_gp20", "i2rt_yam_bimanual", "agilex_piper_bimanual",
        "standard_bots_ro1", "standard_bots_core", "franka_research_3_bimanual",
        "kinova_gen3_bimanual", "ufactory_xarm7_bimanual", "kuka_lbr_iiwa_14",
    }
    for row in rows.values():
        profile = registry.embodiment(row["embodiment_id"])
        assert profile is not None
        assert profile.status == "not_integrated" and not profile.available
        assert profile.controller is None and profile.simulator_runtime_id is None
        assert profile.sensor_reference is None
        assert set(row["required_artifacts"]) <= set(profile.missing)
        assert profile.arm_count == row["arm_count"]
    assert registry.embodiment("fanuc_crx_10ia").embodiment_id != "fanuc_crx_10ia_l"
    assert registry.embodiment_registry_violations() == []


def test_dream_publisher_evidence_is_limited_to_its_actual_paired_models() -> None:
    demonstrated = set()
    for row in _rows().values():
        for route in row["policy_routes"]:
            if route["route_id"] == "dreamlabs_dream1":
                assert route["source_url"] == "https://dreamlabs.ai/research/dream-1/"
                assert route["evidence"] == "publisher_demonstrated"
                assert row["arm_count"] == 2
                demonstrated.add(row["embodiment_id"])
    assert demonstrated == {
        "universal_robots_ur5e_bimanual", "i2rt_yam_bimanual", "agilex_piper_bimanual",
        "franka_research_3_bimanual", "kinova_gen3_bimanual", "ufactory_xarm7_bimanual",
    }
    for embodiment_id in demonstrated:
        profile = registry.embodiment(embodiment_id)
        assert BIMANUAL_EEF_POSE_ACTION in profile.action_schema_ids
        assert "independent_bimanual_tool_channels_required" in profile.missing
        assert "eef_pose_ik_adapter" in profile.missing
    native_joint = [
        row.embodiment_id for row in registry.EMBODIMENT_REGISTRY
        if DREAM1_NATIVE_JOINT_ACTION in row.action_schema_ids
    ]
    assert native_joint == ["i2rt_yam_bimanual"]
    for embodiment_id in ("universal_robots_ur5e", "trlc_dk1_bimanual", "almond_axol",
                          "fanuc_crx_10ia_l", "kuka_lbr_iiwa_14"):
        assert registry.embodiment(embodiment_id).action_schema_ids == ()


def test_platform_evidence_does_not_fabricate_a_model_or_isaac_integration() -> None:
    catalog = registry.deployment_embodiment_support_catalog()
    assert catalog["schema_version"] == "deployment_embodiment_support.v1"
    assert catalog["claim_ceiling"] == "planning_only"
    assert "catalog_digest" not in catalog
    routes = {row["route_id"]: row for row in catalog["policy_routes"]}
    assert set(routes) == {
        "dreamlabs_dream1", "pi_api", "pi05_custom_checkpoint", "customer_controller",
        "planning_endpoint", "skill_trace", "runway_praxis1", "groot_custom_checkpoint",
        "act_custom_checkpoint",
    }
    assert routes["dreamlabs_dream1"]["model_id"] == "dreamlabs/dream-1"
    assert routes["runway_praxis1"]["model_id"] == "praxis-1"
    for route_id in set(routes) - {"dreamlabs_dream1", "runway_praxis1"}:
        assert routes[route_id]["model_id"] is None
    for route_id in set(routes) - {"dreamlabs_dream1"}:
        assert routes[route_id]["observation_schema_id"] is None
        assert routes[route_id]["action_schema_id"] is None
    axol = _rows()["almond_axol"]
    assert axol["arm_dof"] == [7, 7]
    assert "validated_actuated_gripper_model" in axol["required_artifacts"]
    assert "fixed gripper joints" in axol["tool_requirement"]
    assert [(route["route_id"], route["evidence"]) for route in axol["policy_routes"]
            if route["route_id"] == "pi_api"] == [("pi_api", "publisher_demonstrated")]
    assert _rows()["trlc_dk1_bimanual"]["arm_dof"] == [6, 6]


def test_early_partner_and_custom_checkpoint_routes_do_not_inherit_deployment_claims() -> None:
    catalog = registry.deployment_embodiment_support_catalog()
    routes = {row["route_id"]: row for row in catalog["policy_routes"]}
    praxis = routes["runway_praxis1"]
    assert praxis["kind"] == "early_access"
    assert "partner_access_and_usage_terms" in praxis["required_artifacts"]
    assert "runtime_adapter_and_readiness_receipt" in praxis["required_artifacts"]
    linked = {
        row["embodiment_id"]: {route["route_id"]: route for route in row["policy_routes"]}
        for row in catalog["embodiments"]
    }
    assert [body for body, links in linked.items() if "runway_praxis1" in links] == [
        "standard_bots_ro1"
    ]
    assert linked["standard_bots_ro1"]["runway_praxis1"]["evidence"] == "publisher_demonstrated"
    assert "runway_praxis1" not in linked["standard_bots_core"]
    for route_id in ("groot_custom_checkpoint", "act_custom_checkpoint"):
        assert routes[route_id]["kind"] == "checkpoint"
        assert routes[route_id]["model_id"] is None
        assert "checkpoint_digest_and_license" in routes[route_id]["required_artifacts"]
        assert "embodiment_specific_normalization_and_action_mapping" in routes[route_id]["required_artifacts"]
        assert "checkpoint_admission_receipt" in routes[route_id]["required_artifacts"]
        assert [body for body, links in linked.items() if route_id in links] == [
            "universal_robots_ur5e"
        ]
        assert linked["universal_robots_ur5e"][route_id]["evidence"] == "customer_supplied"
        assert "droid" not in str(routes[route_id]).lower()
    for body in ("standard_bots_ro1", "standard_bots_core", "universal_robots_ur5e"):
        assert registry.embodiment(body).action_schema_ids == ()


def test_sources_preserve_exact_variants_pins_and_asset_gaps() -> None:
    rows = _rows()
    assert rows["franka_research_3_bimanual"]["asset_sources"][0]["relative_path"] == (
        "FrankaRobotics/FrankaFR3/fr3.usd"
    )
    assert rows["fanuc_crx_10ia_l"]["asset_sources"][0]["relative_path"] == (
        "Fanuc/crx10ia_l/crx10ia_l.usd"
    )
    for embodiment_id in ("trlc_dk1_bimanual", "almond_axol", "agilex_piper_bimanual",
                          "ufactory_xarm7_bimanual", "kuka_lbr_iiwa_14"):
        source = rows[embodiment_id]["asset_sources"][0]
        assert source["revision"] and len(source["revision"]) == 40
        assert source["revision"] in source["url"]
    for embodiment_id in ("yaskawa_gp12", "yaskawa_gp20"):
        assert rows[embodiment_id]["asset_sources"][0]["format"] == "documentation"
        assert "exact_robot_variant_and_licensed_asset" in rows[embodiment_id]["required_artifacts"]
    for embodiment_id in ("standard_bots_ro1", "standard_bots_core"):
        assert "manufacturer_asset_access_and_license" in rows[embodiment_id]["required_artifacts"]
    assert rows["standard_bots_core"]["related_teams"] == ["Standard Bots", "Path Machining"]
    assert rows["fanuc_m710ic_45m"]["related_teams"] == ["Jacobi Robotics", "Delta Technology"]


def test_each_catalog_read_is_independent_of_consumer_mutation() -> None:
    value = registry.deployment_embodiment_support_catalog()
    value["embodiments"][0]["display_name"] = "changed by a consumer"
    value["embodiments"].clear()
    assert len(_rows()) == 16
