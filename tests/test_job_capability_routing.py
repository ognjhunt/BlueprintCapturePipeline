"""Capability routing lists what can run; it never invents a configuration."""

from __future__ import annotations

from blueprint_pipeline.job_capability_routing import (
    EMBODIMENT_PROFILES,
    route_job_capabilities,
    routing_limitations,
)
from blueprint_pipeline.task_evaluation_policy_candidate_registry import runnable_candidate_ids


def _policies(value):
    return [row["policy"]["candidate_id"] for row in value["configurations"]]


def test_routing_reexports_the_same_pure_task_profile_metadata():
    from blueprint_pipeline import job_capability_routing, task_profile_contracts
    assert job_capability_routing.TASK_PROFILES is task_profile_contracts.TASK_PROFILES
    assert job_capability_routing.TaskProfile is task_profile_contracts.TaskProfile
    assert job_capability_routing.task_profile_for is task_profile_contracts.task_profile_for


def test_supported_profile_lists_complete_runnable_configurations():
    value = route_job_capabilities(
        "rigid_object_pick_place", provided_inputs=("metric_object_geometry", "target_region")
    )
    assert value["status"] == "supported"
    assert value["task_profile_id"] == "rigid_pick_place"
    assert _policies(value) == list(runnable_candidate_ids())
    row = value["configurations"][0]
    for field in ("robot", "tool", "observation_schema_id", "action_schema_id", "controller",
                  "simulator_runtime_id", "runtime_adapter", "success_predicate"):
        assert row[field]
    assert row["policy"]["checkpoint_revision"] and row["policy"]["checkpoint_digest"].startswith("sha256:")
    assert value["routing_digest"] == route_job_capabilities(
        "rigid_object_pick_place", provided_inputs=("metric_object_geometry", "target_region")
    )["routing_digest"]


def test_missing_measurements_are_requested_not_assumed():
    value = route_job_capabilities("articulated_open_close")
    assert value["status"] == "needs_input"
    assert value["missing_inputs"] == ["qualified_articulation_graph"]
    assert _policies(value)  # Supported once the articulation graph is qualified.


def test_unsupported_family_is_a_gap_not_coerced_into_pick_place():
    for family in ("laundry_folding", "manipulation", "pick"):
        value = route_job_capabilities(family)
        assert value["status"] == "capability_gap"
        assert value["configurations"] == []
        assert "independent_task_scorer" in value["capability_gap"]["missing"]
        assert routing_limitations(value)[0].startswith(f"Capability gap for task family '{family}'")
    assert route_job_capabilities("")["missing_inputs"] == ["task_family"]


def test_unavailable_robots_stay_listed_with_reasons_and_never_run():
    atlas = next(row for row in EMBODIMENT_PROFILES if row.embodiment_id == "boston_dynamics_atlas")
    assert atlas.available is False and atlas.reason
    value = route_job_capabilities("rigid_pick_place", embodiment_ids=["boston_dynamics_atlas", "unknown_bot"],
                                   provided_inputs=("metric_object_geometry", "target_region"))
    assert value["status"] == "capability_gap" and value["configurations"] == []
    reasons = {row["id"]: row["reason"] for row in value["unavailable"]}
    assert set(reasons) == {"boston_dynamics_atlas", "unknown_bot"}
    # Atlas is never a prerequisite: the default route still runs supported robots.
    assert route_job_capabilities("rigid_pick_place", provided_inputs=(
        "metric_object_geometry", "target_region"))["status"] == "supported"


def test_default_route_shows_which_robot_would_unlock_each_unavailable_policy():
    value = route_job_capabilities("rigid_pick_place", provided_inputs=("metric_object_geometry", "target_region"))
    rows = {(row["kind"], row["id"]): row for row in value["unavailable"]}
    dream = rows[("policy", "dreamlabs_dream1")]
    assert "provider_account_access_unverified" in dream["blockers"]
    assert "verified_robot_asset" in rows[("embodiment", "i2rt_yam_bimanual")]["missing"]
    assert ("embodiment", "universal_robots_ur5e") in rows
    # Three-camera DROID checkpoints match no listed robot rig yet.
    assert {"cosmos3_nano_policy_droid", "flux3_action_droid"} <= {row["id"] for row in value["unmatched_policies"]}
    assert all(row["policy"]["kind"] == "checkpoint" for row in value["configurations"])
    assert all(row["embodiment_id"] == "droid_franka_robotiq_2f85" for row in value["configurations"])
