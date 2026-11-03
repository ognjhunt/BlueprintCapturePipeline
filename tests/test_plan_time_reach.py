"""Reach is decided at planning, before anything is reserved, from evidence the packet already has.

The geometry below follows the website job capture-447 (a tray carried along a dining table): the
packet's station stands about a metre from both ends, beyond the DROID Franka's chain reach.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan, validate_screening_plan
from blueprint_pipeline.robot_arm_path_clearance import (
    arm_reach_bound,
    carry_grasps,
    reach_insufficient_code,
    rigid_reach_shortfall,
)
from blueprint_pipeline.website_candidate_feasibility import (
    _decide_fixed,
    bound_station,
    candidate_feasibility,
    infeasible_blockers,
    is_measured_refusal,
)
from tests.test_beta_policy_screening_plan import NOW, inputs, scene

STATION = {"position_world_m": [0.4674, -0.7031, 0.8166], "orientation_xyzw": [0.0, 0.0, 0.9782156, 0.2075915]}
START = [-0.4326, -0.3031, 0.8244, 0.0, 0.0, 0.0, 1.0]
TARGET = [-0.2277, 0.0998, 0.8143]
CONTACT = [0.0, 0.0, 0.0225]
TRAY = {"minimum": [-0.1567, -0.2606, -0.0104], "maximum": [0.1567, 0.2606, 0.0346]}
DROID = "droid_franka_panda_robotiq_2f85_v1"


def _plan(station=STATION, strategy="pick_and_place"):
    return {"plan_digest": "sha256:" + "a" * 64, "task_kind": "rigid_pick_place",
            "robot": {"robot_id": "franka_panda", "base_pose_world": station},
            "objects": [{"semantic_role": "task_object", "object_type": "RIGID", "task_subject": True}],
            "task_spec": {"manipulation_strategy": strategy, "start_pose_world": START,
                          "target_position_world_m": TARGET, "subject_collision_bounds_scoring_frame_m": TRAY,
                          "interaction_affordance": {"contact_point_scoring_frame_m": CONTACT}}}


def _catalog(*keys):
    return {"candidates": [{"candidate_id": key, "interface": {"robot_profile_id": DROID}} for key in keys]}


def test_the_franka_reach_bound_holds_at_every_joint_value_and_is_attained():
    from blueprint_pipeline.robot_arm_path_clearance import ARM_LINK_BOUNDS, _frames

    shoulder, bound = arm_reach_bound("franka_panda")
    assert shoulder.tolist() == pytest.approx([0.0, 0.0, 0.333]) and bound == pytest.approx(0.9903, abs=1e-4)
    grasp = np.asarray(ARM_LINK_BOUNDS["franka_panda"]["grasp_point_flange_m"])
    farthest = 0.0
    for q in np.random.default_rng(7).uniform(-math.pi, math.pi, (4000, 7)):
        flange = _frames(q)[8]
        farthest = max(farthest, float(np.linalg.norm(flange[:3, 3] + flange[:3, :3] @ grasp - shoulder)))
    assert farthest <= bound + 1e-9 and farthest > bound - 0.01


def test_a_task_contact_beyond_the_chain_bound_is_a_proof_named_by_its_farthest_phase():
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=STATION, start_pose_world=START,
                                  target_position_world=TARGET, contact_point_m=CONTACT, subject_bounds_m=TRAY)
    distances = {row["phase"]: row["distance_m"] for row in reach["phases"]}
    assert reach["status"] == "out_of_reach" and reach["beyond"] == ["place", "pick"]
    assert distances["place"] == pytest.approx(1.107, abs=2e-3) and distances["pick"] == pytest.approx(1.030, abs=2e-3)
    # The tray's near corner is within the bound at both ends: a grasp off the task's contact is not ruled out.
    assert reach["grasp_independent"] is False
    assert all(row["object_nearest_distance_m"] < reach["reach_bound_m"] for row in reach["phases"])
    assert reach["claim_ceiling"] == "cpu_bounded_estimate" and reach["simulator_executed"] is False
    near = {**STATION, "position_world_m": [0.1, -0.5, 0.8166]}
    assert rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=near, start_pose_world=START,
                                 target_position_world=TARGET, contact_point_m=CONTACT)["beyond"] == []


def test_droid_candidates_proven_out_of_reach_at_the_packet_station_are_infeasible_before_any_quote():
    feasibility = candidate_feasibility(scene_plan=_plan(), catalog=_catalog("pi05_droid", "groot_n17_droid"),
                                        profile={})
    row = feasibility["candidates"]["pi05_droid"]
    assert row["status"] == "infeasible" and row["reach_source"] == "franka_chain_reach_bound"
    assert row["blockers"] == [reach_insufficient_code("place", row["reach"]["phases"][1]["distance_m"],
                                                       row["reach"]["reach_bound_m"])]
    assert row["blockers"][0].startswith("robot_reach_insufficient:place:1.10") and row["blockers"][0].endswith("m_vs_0.990m")
    assert infeasible_blockers(feasibility) == {"pi05_droid": row["blockers"], "groot_n17_droid": row["blockers"]}
    assert is_measured_refusal(row["blockers"][0]) and not is_measured_refusal("screening_gpu_memory_floor_unrentable")


@pytest.mark.parametrize("status, expected, blockers", [
    ("clear", "feasible", []),
    ("blocked", "unassessed", ["packet_station_search_blocked:robot_rigid_pick_place_unreachable:lift"]),
    (None, "unassessed", ["packet_station_search_not_recorded"])])
def test_within_the_bound_the_packet_gates_own_search_decides_and_a_miss_is_never_a_refusal(
        monkeypatch, status, expected, blockers):
    from blueprint_pipeline import team_scene_packet

    near = {**STATION, "position_world_m": [0.1, -0.5, 0.8166], "stance_digest": "sha256:" + "b" * 64}
    recorded = {} if status is None else {"robot_station": near, "robot_station_search": {
        "cleared": status == "clear", "results": [{"stance_digest": near["stance_digest"], "status": status,
            "blockers": [] if status == "clear" else ["robot_rigid_pick_place_unreachable:lift"]}]}}
    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: recorded)
    plan = _plan(station={key: near[key] for key in ("position_world_m", "orientation_xyzw")})
    row = candidate_feasibility(scene_plan=plan, catalog=_catalog("pi05_droid"), profile={})["candidates"]["pi05_droid"]
    assert (row["status"], row["blockers"]) == (expected, blockers)
    assert row["reach_source"] == ("packet_station_search" if status == "clear" else None)


def test_a_candidate_on_another_arm_an_articulated_task_or_the_g1_lane_is_not_measured_with_the_franka():
    catalog = {"candidates": [{"candidate_id": "other", "interface": {"robot_profile_id": "someone_elses_arm"}},
                              {"candidate_id": "g1", "interface": {"robot_profile_id": DROID}}]}
    rows = candidate_feasibility(scene_plan=_plan(), catalog=catalog,
                                 profile={"screening_candidates": {"g1": {"native_g1": {}}}})["candidates"]
    assert rows["other"] == rows["g1"] == {"status": "unassessed",
                                           "blockers": ["embodiment_reach_manifest_not_registered"]}
    push = candidate_feasibility(scene_plan=_plan(strategy="planar_push"), catalog=_catalog("pi05_droid"), profile={})
    assert push["candidates"]["pi05_droid"]["blockers"] == ["packet_arm_reach_task_unmodeled"]


def test_a_measured_own_arm_solve_outranks_the_declared_reach_either_way():
    declared_ok = {"status": "feasible", "blockers": [], "reach_m": 1.453, "station_basis": "scene_packet_station"}
    proven = {"status": "out_of_reach", "beyond_reach": {"arm": "arm", "phase_id": "place", "distance_m": 1.0619,
                                                          "reach_bound_m": 0.9}}
    row = _decide_fixed(declared_ok, proven)
    assert (row["status"], row["blockers"], row["reach_source"]) == (
        "infeasible", ["robot_reach_insufficient:place:1.062m_vs_0.900m"], "own_arm_model")
    declared_short = {**declared_ok, "status": "infeasible", "blockers": ["team_embodiment_station_out_of_reach"]}
    row = _decide_fixed(declared_short, {"status": "passed"})
    assert (row["status"], row["blockers"], row["reach_source"]) == ("feasible", [], "own_arm_model")
    # Its run binds it where it was solved, so the declared figure no longer vetoes the binding.
    plan = _plan()
    assert bound_station(row, plan) == ({key: STATION[key] for key in ("position_world_m", "orientation_xyzw")}, True)
    # A search miss or no model leaves the declared verdict, and the row says which source decided.
    for arm in ({"status": "flagged"}, {"status": "unassessed", "reason": "task_phases_unavailable"}, None):
        row = _decide_fixed(declared_ok, arm)
        assert (row["status"], row["reach_source"]) == ("feasible", "declared_reach_m")
        assert bound_station(row, plan) == (None, False)


def test_an_infeasible_reach_is_listed_never_quoted_and_the_site_reads_it_as_measured(tmp_path):
    catalog, capabilities, quotes = inputs()
    code = "robot_reach_insufficient:place:1.107m_vs_0.990m"
    capabilities["candidate_infeasible"] = {row["candidate_id"]: [code] for row in catalog["candidates"]}
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                 cost_quotes=quotes, now=NOW)
    assert {row["status"] for row in value["candidates"]} == {"infeasible"}
    assert value["maximum_planned_cost_usd"] == "0.00" and validate_screening_plan(value) == value
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    [published] = coordinator.pending_publications(queue_root=tmp_path)
    # Nothing is reserved, the campaign ends blocked, and every row carries the measured-geometry marker
    # the site's outcome reads as infeasible (WebApp screeningCandidateState), with the measured numbers.
    assert published["status"] == "blocked" and published["budget"]["committed_upper_bound_usd"] == "0"
    assert all(row["status"] == "unavailable" and row["blockers"] == [coordinator.INFEASIBLE_BLOCKER, code]
               for row in published["candidates"])


def test_the_carry_grasps_are_the_tasks_own_then_the_intakes_tilted_and_side_on_ones():
    task = {"approach_unit": [0.0, 0.0, 1.0], "gripper_orientation_xyzw": [0.0, 1.0, 0.0, 0.0]}
    grasps = carry_grasps(task, TRAY)
    assert grasps[0] == {"name": "task", **task} and grasps[1]["name"] == "top_down"
    assert {row["name"] for row in grasps[2:]} >= {"tilt45_yaw0", "tilt90_yaw270"} and len(grasps) == 10
    assert carry_grasps(task, None) == [grasps[0]]


def test_a_station_search_that_carries_from_nowhere_keeps_its_best_attempt_and_the_measured_shortfall(monkeypatch):
    from blueprint_pipeline import robot_arm_path_clearance as clearance
    from blueprint_pipeline.team_scene_packet import _carrying_station

    seen = {}

    def nothing_carries(stances, *, grasps, **task):
        seen["grasps"] = [row["name"] for row in grasps]
        attempts = [(stance, {"status": "blocked", "blockers": ["robot_rigid_pick_place_unreachable:approach"],
                              "furthest_sample": index, "path_samples": 60, "grasp": grasp["name"]})
                    for index, (stance, grasp) in enumerate((s, g) for g in grasps for s in stances)]
        return None, max((r for _, r in attempts), key=lambda r: r["furthest_sample"]), attempts
    monkeypatch.setattr(clearance, "first_carrying_stance", nothing_carries)
    stances = [{**STATION, "stance_digest": "sha256:" + str(i) * 64} for i in range(2)]
    task_spec = {"start_pose_world": START, "target_position_world_m": TARGET,
                 "subject_collision_bounds_scoring_frame_m": TRAY,
                 "interaction_affordance": {"contact_point_scoring_frame_m": CONTACT,
                                            "approach_unit_scoring_frame": [0.0, 0.0, 1.0],
                                            "gripper_orientation_scoring_frame_xyzw": [0.0, 1.0, 0.0, 0.0],
                                            "pregrasp_clearance_m": 0.1}}
    profile = {"robot_joint_reset_positions_rad": {f"panda_joint{i}": 0.0 for i in range(1, 8)}}
    station, record = _carrying_station(stances, task_spec=task_spec, collision="unused.usda", profile=profile)
    assert station is stances[0] and record["cleared"] is False and seen["grasps"][0] == "task"
    assert record["grasps_tried"] == len(seen["grasps"]) == 10
    assert record["best_attempt"]["stance_digest"] == stances[1]["stance_digest"]
    assert record["best_attempt"]["grasp"] == seen["grasps"][-1]
    assert record["reach_shortfall"]["beyond"] == ["place", "pick"]


def test_the_packet_gate_admits_a_carry_only_a_tilted_grasp_clears(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    from blueprint_pipeline import robot_arm_path_clearance as clearance
    from tests.test_robot_arm_path_clearance import _rigid_packet

    tried = []

    def only_tilted(**kwargs):
        # Say the top-down carry cannot lift here and every tilted one clears.
        tried.append(kwargs["approach_unit"])
        top_down = kwargs["approach_unit"] == [0.0, 0.0, 1.0]
        return {"schema_version": clearance.SCHEMA_VERSION, "task_kind": "rigid_pick_place", "robot_id": "franka_panda",
                "status": "blocked" if top_down else "clear", "furthest_sample": 3, "path_samples": 40,
                "blockers": ["robot_rigid_pick_place_unreachable:lift"] if top_down else []}
    monkeypatch.setattr(clearance, "rigid_pick_place_clearance", only_tilted)
    packet = _rigid_packet(tmp_path / "floor", target=[0.5, 0.3, 0.0325])
    import json
    plan_path = packet / "native_task_arena_scene_plan.v1.json"
    plan = json.loads(plan_path.read_text())
    plan["task_spec"]["subject_collision_bounds_scoring_frame_m"] = {"minimum": [-0.02, -0.03, -0.03],
                                                                     "maximum": [0.02, 0.03, 0.03]}
    plan_path.write_text(json.dumps(plan))
    report = clearance.packet_arm_path_clearance(packet)
    # The task's grasp and the intake's top-down one are refused; the first tilted grasp clears, and is named.
    assert report["status"] == "clear" and report["grasp"] == "tilt45_yaw0" and len(tried) == 3
    # With only the task's grasp blocked and no bounds to derive others from, the refusal stands.
    del plan["task_spec"]["subject_collision_bounds_scoring_frame_m"]
    plan_path.write_text(json.dumps(plan))
    refused = clearance.packet_arm_path_clearance(packet)
    assert refused["status"] == "blocked" and refused["grasp"] == "task"
