"""Reach is decided at planning, before anything is reserved, from evidence the packet already has.

The geometry below follows the website job capture-447 (a tray carried along a dining table): the
packet's station stands about a metre from the tray's centre at both ends, beyond the DROID Franka's chain
reach, while the tray's near edge is within it. The team half's station and the bundle's gate need the
task's grasp contact in reach at both ends (capture-447 ran 39 paid episodes with it out of reach at the
place); the scripted carry's miss with the contact in reach is a flag.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import (
    build_screening_plan,
    validate_screening_plan,
)
from blueprint_pipeline.robot_arm_path_clearance import (
    SCRIPTED_CARRY_FLAG,
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
#: A station across the room: the whole tray is beyond the chain bound at both ends.
FAR = {"position_world_m": [1.3, -0.7031, 0.8166], "orientation_xyzw": [0.0, 0.0, 0.9782156, 0.2075915]}
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
                          "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
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


def test_a_task_contact_beyond_the_chain_bound_is_evidence_and_the_objects_nearest_point_decides():
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=STATION, start_pose_world=START,
                                  target_position_world=TARGET, contact_point_m=CONTACT, subject_bounds_m=TRAY,
                                  destination_orientation_xyzw=START[3:])
    distances = {row["phase"]: row["distance_m"] for row in reach["phases"]}
    # The task's contact (the tray's centre) is beyond the bound at both ends: the scripted carry's contact.
    assert reach["task_contact_beyond"] == ["place", "pick"]
    assert distances["place"] == pytest.approx(1.107, abs=2e-3) and distances["pick"] == pytest.approx(1.030, abs=2e-3)
    # The tray's near edge is within the bound at both ends: a policy's own grasp there is not ruled out.
    assert reach["status"] == "within_bound" and reach["beyond"] == [] and reach["grasp_independent"] is False
    assert reach["rule"] == "object_nearest_point_within_arm_chain_reach_bound_of_shoulder"
    assert all(row["object_nearest_distance_m"] < reach["reach_bound_m"] for row in reach["phases"])
    assert reach["claim_ceiling"] == "cpu_bounded_estimate" and reach["simulator_executed"] is False
    # From across the room the whole tray is out of reach: a proof for any grasp, named by its farthest phase.
    far = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=FAR, start_pose_world=START,
                                target_position_world=TARGET, contact_point_m=CONTACT, subject_bounds_m=TRAY,
                                destination_orientation_xyzw=START[3:])
    nearest = {row["phase"]: row["object_nearest_distance_m"] for row in far["phases"]}
    assert far["status"] == "out_of_reach" and far["beyond"] == ["pick", "place"] and far["grasp_independent"] is True
    assert nearest["pick"] > nearest["place"] > far["reach_bound_m"]
    # Without the object's box only the contact is known: in reach it is a point of the object; beyond, nothing
    # is proved.
    near = {**STATION, "position_world_m": [0.1, -0.5, 0.8166]}
    assert rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=near, start_pose_world=START,
                                 target_position_world=TARGET, contact_point_m=CONTACT)["status"] == "within_bound"
    unknown = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=STATION, start_pose_world=START,
                                    target_position_world=TARGET, contact_point_m=CONTACT)
    assert unknown["status"] == "unassessed" and unknown["beyond"] == [] and unknown["task_contact_beyond"]


def test_the_place_is_measured_wherever_the_task_accepts_the_object():
    # A long box ending broadside to the arm, just out of reach held as it started.
    box = {"minimum": [-0.05, -0.3, -0.02], "maximum": [0.05, 0.3, 0.02]}
    base = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    common = {"robot_id": "franka_panda", "base_pose_world": base, "start_pose_world": [0.5, 0.0, 0.02, 0, 0, 0, 1],
              "destination_orientation_xyzw": [0, 0, 0, 1],
              "target_position_world": [1.05, 0.0, 0.02], "contact_point_m": [0.0, 0.0, 0.0], "subject_bounds_m": box}
    held = rigid_reach_shortfall(**common)
    assert held["beyond"] == ["place"]
    # Its origin may end 10 cm nearer: within reach.
    region = {"minimum": [0.95, -0.05, 0.01], "maximum": [1.1, 0.05, 0.03]}
    assert rigid_reach_shortfall(**common, destination_bounds_world_m=region)["status"] == "within_bound"
    # Or turned end-on toward the arm, when the contract lets it end turned: within reach again.
    turned = rigid_reach_shortfall(**common, place_yaw_free_rad=math.pi)
    assert turned["status"] == "within_bound" and turned["place_region"]["yaw_free_rad"] == pytest.approx(math.pi)
    nearest = {row["phase"]: row["object_nearest_distance_m"] for row in turned["phases"]}
    assert nearest["place"] < {row["phase"]: row["object_nearest_distance_m"] for row in held["phases"]}["place"]


def test_droid_candidates_proven_out_of_reach_at_the_packet_station_are_infeasible_before_any_quote():
    feasibility = candidate_feasibility(scene_plan=_plan(station=FAR),
                                        catalog=_catalog("pi05_droid", "groot_n17_droid"), profile={})
    row = feasibility["candidates"]["pi05_droid"]
    assert row["status"] == "infeasible" and row["reach_source"] == "franka_chain_reach_bound"
    # Named by the object's nearest point at its farthest phase, not the task's contact.
    pick = row["reach"]["phases"][0]
    assert row["blockers"] == [reach_insufficient_code("pick", pick["object_nearest_distance_m"],
                                                       row["reach"]["reach_bound_m"])]
    assert row["blockers"][0].startswith("robot_reach_insufficient:pick:1.") and row["blockers"][0].endswith("m_vs_0.990m")
    assert infeasible_blockers(feasibility) == {"pi05_droid": row["blockers"], "groot_n17_droid": row["blockers"]}
    assert is_measured_refusal(row["blockers"][0]) and not is_measured_refusal("screening_gpu_memory_floor_unrentable")


def test_a_tray_whose_centre_is_out_of_reach_but_near_edge_is_in_reach_is_infeasible_as_the_gate_refuses_it(
        monkeypatch):
    # capture-447: the task's contact (the tray's centre) is beyond the bound at both ends from the packet's
    # station, though its near edge is in reach. The bundle's gate refuses that packet, so planning does too.
    from blueprint_pipeline import team_scene_packet
    from blueprint_pipeline.robot_arm_path_clearance import reach_refusal_code

    station = {**STATION, "stance_digest": "sha256:" + "c" * 64}
    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: {
        "robot_station": station, "robot_station_search": {"cleared": False, "results": [
            {"stance_digest": station["stance_digest"], "status": "blocked",
             "blockers": ["robot_rigid_pick_place_unreachable:approach"]}]}})
    feasibility = candidate_feasibility(scene_plan=_plan(), catalog=_catalog("pi05_droid", "groot_n17_droid"),
                                        profile={})
    row = feasibility["candidates"]["pi05_droid"]
    assert row["status"] == "infeasible" and row["reach_source"] == "franka_chain_reach_bound"
    assert row["reach"]["task_contact_beyond"] == ["place", "pick"] and row["reach"]["beyond"] == []
    place = next(item for item in row["reach"]["phases"] if item["phase"] == "place")
    assert row["blockers"] == [reach_refusal_code(row["reach"])] == [
        reach_insufficient_code("place", place["distance_m"], row["reach"]["reach_bound_m"])]
    assert infeasible_blockers(feasibility) == {"pi05_droid": row["blockers"], "groot_n17_droid": row["blockers"]}
    assert is_measured_refusal(row["blockers"][0])


@pytest.mark.parametrize("status, expected, blockers, flags", [
    ("clear", "feasible", [], None),
    ("blocked", "feasible", [], [SCRIPTED_CARRY_FLAG]),
    (None, "feasible", [], ["packet_station_search_not_recorded"])])
def test_within_the_bound_the_packet_gates_own_search_is_evidence_and_a_miss_is_never_a_refusal(
        monkeypatch, status, expected, blockers, flags):
    from blueprint_pipeline import team_scene_packet

    near = {**STATION, "position_world_m": [0.1, -0.5, 0.8166], "stance_digest": "sha256:" + "b" * 64}
    recorded = {} if status is None else {"robot_station": near, "robot_station_search": {
        "cleared": status == "clear", "results": [{"stance_digest": near["stance_digest"], "status": status,
            "blockers": [] if status == "clear" else ["robot_rigid_pick_place_unreachable:lift"]}]}}
    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: recorded)
    plan = _plan(station={key: near[key] for key in ("position_world_m", "orientation_xyzw")})
    row = candidate_feasibility(scene_plan=plan, catalog=_catalog("pi05_droid"), profile={})["candidates"]["pi05_droid"]
    assert (row["status"], row["blockers"], row.get("flags")) == (expected, blockers, flags)
    assert row["reach_source"] == ("packet_station_search" if status == "clear" else "franka_chain_reach_bound")


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
                 "subject_collision_bounds_scoring_frame_m": TRAY, "destination_orientation_xyzw": START[3:],
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
    # The tray's near edge is in reach from both, its contact from neither: no stance is in reach, the closest
    # stands with its measured shortfall, and the bundle's gate refuses it (``rigid_carry_verdict``).
    assert record["stances_in_reach"] == 0 and record["chosen_basis"] == "no_clear_stance_in_reach"
    assert record["flags"] == [] and record["reach_shortfall"]["status"] == "within_bound"
    assert record["reach_shortfall"]["task_contact_beyond"] == ["place", "pick"]
    assert record["object_reach"]["task_contact_beyond"] == ["place", "pick"]
    # From stances across the room no point of the tray is in reach: the closest stands, out of reach.
    far = [{**FAR, "stance_digest": "sha256:" + str(i) * 64} for i in range(2)]
    station, record = _carrying_station(far, task_spec=task_spec, collision="unused.usda", profile=profile)
    assert station is far[0] and record["chosen_basis"] == "no_clear_stance_in_reach" and record["flags"] == []
    assert record["stances_in_reach"] == 0 and record["reach_shortfall"]["beyond"] == ["pick", "place"]


def test_the_station_is_the_in_reach_stance_whose_contact_is_nearest_when_no_scripted_carry_clears(monkeypatch):
    from blueprint_pipeline import robot_arm_path_clearance as clearance
    from blueprint_pipeline.team_scene_packet import _carrying_station

    tried = {}

    def nothing_carries(stances, *, grasps, **task):
        tried["stances"] = [row["stance_digest"] for row in stances]
        attempts = [(stance, {"status": "blocked", "blockers": ["robot_rigid_pick_place_unreachable:approach"],
                              "furthest_sample": 0, "path_samples": 60, "grasp": "task"}) for stance in stances]
        return None, attempts[0][1], attempts
    monkeypatch.setattr(clearance, "first_carrying_stance", nothing_carries)
    # Ranked best first by the search: out of reach, the tray's edge only in reach, its contact in reach
    # (0.903 m at the place), its contact nearer still (0.752 m).
    rows = [FAR, STATION, {**STATION, "position_world_m": [0.25, -0.6, 0.8166]},
            {**STATION, "position_world_m": [0.1, -0.5, 0.8166]}]
    stances = [{**row, "stance_digest": "sha256:" + str(i) * 64} for i, row in enumerate(rows)]
    task_spec = {"start_pose_world": START, "target_position_world_m": TARGET,
                 "subject_collision_bounds_scoring_frame_m": TRAY, "destination_orientation_xyzw": START[3:],
                 "interaction_affordance": {"contact_point_scoring_frame_m": CONTACT,
                                            "approach_unit_scoring_frame": [0.0, 0.0, 1.0],
                                            "gripper_orientation_scoring_frame_xyzw": [0.0, 1.0, 0.0, 0.0],
                                            "pregrasp_clearance_m": 0.1}}
    profile = {"robot_joint_reset_positions_rad": {f"panda_joint{i}": 0.0 for i in range(1, 8)}}
    station, record = _carrying_station(stances, task_spec=task_spec, collision="unused.usda", profile=profile)
    # The scripted carry is tried only from stances with the contact in reach, nearest contact first; it stands.
    assert tried["stances"] == [stances[3]["stance_digest"], stances[2]["stance_digest"]]
    assert station is stances[3] and record["chosen_rank"] == 3 and record["stances_in_reach"] == 2
    assert record["chosen_basis"] == "nearest_graspable_point_in_reach" and record["flags"] == [SCRIPTED_CARRY_FLAG]


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
    # With only the task's grasp blocked and no bounds to derive others from, the miss is flagged, never refused.
    del plan["task_spec"]["subject_collision_bounds_scoring_frame_m"]
    plan_path.write_text(json.dumps(plan))
    flagged = clearance.packet_arm_path_clearance(packet)
    assert flagged["status"] == "flagged" and flagged["grasp"] == "task" and flagged["blockers"] == []
    assert flagged["flags"] == [SCRIPTED_CARRY_FLAG]


# capture-447 as the tracker records it (2026-11-06 reach pass): the task's grasp contact 0.970 m from the
# shoulder at the pick and 1.086 m at the place, against a 0.990 m bound, while the tray's nearest point is
# 0.735 m and 0.729 m away. Laid out in the plane of the shoulder: the base at the origin (shoulder 0.333 m
# up), the pick ahead of it along x, the place to its left along y, the contact 0.10 m along the tray's x.
SHOULDER_Z = 0.333
WIDE_TRAY = {"minimum": [-0.135, -0.35239, -0.02], "maximum": [0.135, 0.35239, 0.02]}
WIDE_CONTACT = [0.10, 0.0, 0.0]
WIDE_START = [0.87, 0.0, SHOULDER_Z, 0.0, 0.0, 0.0, 1.0]
WIDE_TARGET = [0.0, math.sqrt(1.086 ** 2 - 0.10 ** 2), SHOULDER_Z]
OLD_STATION = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
               "stance_digest": "sha256:" + "e" * 64}
#: About midway between the pick and the place.
MIDWAY = {"position_world_m": [0.435, WIDE_TARGET[1] / 2.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
          "stance_digest": "sha256:" + "f" * 64}
CODE_447 = "robot_reach_insufficient:place:1.086m_vs_0.990m"


def _wide_spec(start=WIDE_START, target=WIDE_TARGET, contact=WIDE_CONTACT, bounds=WIDE_TRAY):
    return {"manipulation_strategy": "pick_and_place", "start_pose_world": list(start),
            "target_position_world_m": list(target), "subject_collision_bounds_scoring_frame_m": bounds,
            "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "interaction_affordance": {"contact_point_scoring_frame_m": list(contact),
                                       "approach_unit_scoring_frame": [0.0, 0.0, 1.0],
                                       "gripper_orientation_scoring_frame_xyzw": [0.0, 1.0, 0.0, 0.0],
                                       "pregrasp_clearance_m": 0.1}}


PANDA_RESET = {"robot_joint_reset_positions_rad": {f"panda_joint{i}": 0.0 for i in range(1, 8)}}


def _scripted(monkeypatch, *, clears):
    """Replace the scripted carry: it clears from the first stance it is given, or from none."""
    from blueprint_pipeline import robot_arm_path_clearance as clearance

    tried: list[str] = []

    def carry(stances, *, grasps, **task):
        tried.extend(row["stance_digest"] for row in stances)
        attempts = [(row, {"status": "clear" if clears and i == 0 else "blocked", "grasp": "task",
                           "blockers": [] if clears and i == 0 else ["robot_rigid_pick_place_unreachable:carry"],
                           "furthest_sample": 15, "path_samples": 42}) for i, row in enumerate(stances)]
        return (stances[0] if clears else None), attempts[0][1], attempts
    monkeypatch.setattr(clearance, "first_carrying_stance", carry)
    return tried


def test_capture_447s_station_has_some_of_the_tray_in_reach_but_not_its_contact_and_is_out_of_reach(monkeypatch):
    from blueprint_pipeline.robot_arm_path_clearance import (
        object_reach_inputs,
        task_contact_in_reach,
    )
    from blueprint_pipeline.team_scene_packet import _carrying_station

    spec = _wide_spec()
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    rows = {row["phase"]: row for row in reach["phases"]}
    assert rows["pick"]["distance_m"] == pytest.approx(0.970, abs=5e-4)
    assert rows["place"]["distance_m"] == pytest.approx(1.086, abs=5e-4)
    assert rows["pick"]["object_nearest_distance_m"] == pytest.approx(0.735, abs=5e-4)
    assert rows["place"]["object_nearest_distance_m"] == pytest.approx(0.729, abs=5e-4)
    assert reach["reach_bound_m"] == pytest.approx(0.990, abs=5e-4)
    # The old rule's pass (some point of the tray in reach at both ends) still reads as before ...
    assert reach["status"] == "within_bound" and reach["beyond"] == []
    # ... but the contact is beyond the bound at the place: the stance is out of reach.
    assert reach["task_contact_beyond"] == ["place"] and task_contact_in_reach(reach) is False
    _scripted(monkeypatch, clears=False)
    station, record = _carrying_station([OLD_STATION], task_spec=spec, collision="unused.usda", profile=PANDA_RESET)
    assert station is OLD_STATION and record["stances_in_reach"] == 0
    assert record["chosen_basis"] == "no_clear_stance_in_reach" and record["flags"] == []
    assert record["reach_shortfall"]["task_contact_beyond"] == ["place"]


@pytest.mark.parametrize("clears, basis", [(True, "scripted_carry_cleared"),
                                           (False, "nearest_graspable_point_in_reach")])
def test_a_stance_midway_between_pick_and_place_is_chosen_with_both_contacts_in_reach(monkeypatch, clears, basis):
    from blueprint_pipeline.robot_arm_path_clearance import object_reach_inputs
    from blueprint_pipeline.team_scene_packet import _carrying_station

    spec = _wide_spec()
    tried = _scripted(monkeypatch, clears=clears)
    # The search ranks the old station first; it is not in reach, so the carry is tried from the midway one only.
    station, record = _carrying_station([OLD_STATION, MIDWAY], task_spec=spec, collision="unused.usda",
                                        profile=PANDA_RESET)
    assert station is MIDWAY and record["chosen_rank"] == 1 and record["chosen_basis"] == basis
    assert tried == [MIDWAY["stance_digest"]] and record["stances_in_reach"] == 1
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=MIDWAY, **object_reach_inputs(spec))
    assert reach["task_contact_beyond"] == [] and record["object_reach"]["task_contact_beyond"] == []
    assert all(row["distance_m"] < reach["reach_bound_m"] for row in reach["phases"])


def test_with_no_stance_bringing_the_contact_to_the_place_the_gate_holds_the_job_before_any_paid_run(
        tmp_path, monkeypatch):
    import json

    from blueprint_pipeline import robot_arm_path_clearance as clearance
    from blueprint_pipeline.robot_arm_path_clearance import (
        RobotArmPathAdmissionBlocked,
        object_reach_inputs,
        reach_refusal_code,
        require_robot_arm_path_admission,
        rigid_carry_verdict,
    )

    spec = _wide_spec()
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    miss = {"status": "blocked", "blockers": ["robot_rigid_pick_place_unreachable:carry"], "furthest_sample": 15}
    verdict = rigid_carry_verdict(miss, reach)
    assert verdict["status"] == "blocked" and verdict["blockers"] == [CODE_447] == [reach_refusal_code(reach)]
    assert verdict["scripted_carry_blockers"] == ["robot_rigid_pick_place_unreachable:carry"]
    assert verdict["object_reach"]["status"] == "within_bound"
    assert verdict["object_reach"]["task_contact_beyond"] == ["place"]
    # The packet's own gate, from its scene plan, refuses it by the same name.
    pytest.importorskip("pxr")
    from tests.test_robot_arm_path_clearance import _rigid_packet

    monkeypatch.setattr(clearance, "rigid_pick_place_clearance", lambda **_: dict(miss))
    packet = _rigid_packet(tmp_path / "packet", target=WIDE_TARGET)
    plan_path = packet / "native_task_arena_scene_plan.v1.json"
    plan = json.loads(plan_path.read_text())
    plan["task_spec"].update({key: spec[key] for key in (
        "start_pose_world", "target_position_world_m", "subject_collision_bounds_scoring_frame_m",
        "destination_orientation_xyzw")})
    plan["task_spec"]["interaction_affordance"]["contact_point_scoring_frame_m"] = WIDE_CONTACT
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(RobotArmPathAdmissionBlocked) as refused:
        require_robot_arm_path_admission(packet)
    assert refused.value.blockers == [CODE_447]


@pytest.mark.parametrize("clears", [True, False])
def test_a_carry_comfortably_in_reach_keeps_its_station_and_passes_the_gate(monkeypatch, clears):
    from blueprint_pipeline.robot_arm_path_clearance import object_reach_inputs, rigid_carry_verdict
    from blueprint_pipeline.team_scene_packet import _carrying_station

    box = {"minimum": [-0.03, -0.03, -0.03], "maximum": [0.03, 0.03, 0.03]}
    spec = _wide_spec(start=[0.5, 0.0, SHOULDER_Z, 0.0, 0.0, 0.0, 1.0], target=[0.4, 0.3, SHOULDER_Z],
                      contact=[0.0, 0.0, 0.0], bounds=box)
    behind = {**OLD_STATION, "position_world_m": [-0.2, 0.0, 0.0], "stance_digest": "sha256:" + "d" * 64}
    stances = [OLD_STATION, behind]
    reaches = [rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=row, **object_reach_inputs(spec))
               for row in stances]
    # Both stances were in reach by the old rule and are by the new one.
    assert all(row["status"] == "within_bound" and row["task_contact_beyond"] == [] for row in reaches)
    tried = _scripted(monkeypatch, clears=clears)
    station, record = _carrying_station(stances, task_spec=spec, collision="unused.usda", profile=PANDA_RESET)
    assert station is OLD_STATION and record["chosen_rank"] == 0 and record["stances_in_reach"] == 2
    assert tried == [OLD_STATION["stance_digest"], behind["stance_digest"]]
    carry = {"status": "clear", "blockers": []} if clears else {
        "status": "blocked", "blockers": ["robot_rigid_pick_place_unreachable:carry"]}
    verdict = rigid_carry_verdict(carry, reaches[0])
    assert verdict["status"] == ("clear" if clears else "flagged")
    assert verdict["blockers"] == [] and verdict.get("flags", []) == ([] if clears else [SCRIPTED_CARRY_FLAG])


def _wide_plan(station, spec):
    return {"plan_digest": "sha256:" + "a" * 64, "task_kind": "rigid_pick_place",
            "robot": {"robot_id": "franka_panda",
                      "base_pose_world": {key: station[key] for key in ("position_world_m", "orientation_xyzw")}},
            "objects": [{"semantic_role": "task_object", "object_type": "RIGID", "task_subject": True}],
            "task_spec": spec}


def test_planning_refuses_the_recorded_capture_447_station_by_the_gates_own_reason(monkeypatch):
    # Nearest point 0.729 m at the place, contact 1.086 m, bound 0.990 m: infeasible, named as the gate names it.
    from blueprint_pipeline import team_scene_packet

    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: {
        "robot_station": OLD_STATION, "robot_station_search": {"cleared": False, "results": [
            {"stance_digest": OLD_STATION["stance_digest"], "status": "blocked",
             "blockers": ["robot_rigid_pick_place_unreachable:carry"]}]}})
    row = candidate_feasibility(scene_plan=_wide_plan(OLD_STATION, _wide_spec()), catalog=_catalog("pi05_droid"),
                                profile={})["candidates"]["pi05_droid"]
    assert row["reach"]["status"] == "within_bound" and row["reach"]["task_contact_beyond"] == ["place"]
    assert (row["status"], row["blockers"], row["reach_source"]) == (
        "infeasible", [CODE_447], "franka_chain_reach_bound")


@pytest.mark.parametrize("status, source, flags", [("clear", "packet_station_search", None),
                                                   ("blocked", "franka_chain_reach_bound", [SCRIPTED_CARRY_FLAG])])
def test_planning_a_carry_comfortably_in_reach_is_unchanged(monkeypatch, status, source, flags):
    from blueprint_pipeline import team_scene_packet

    box = {"minimum": [-0.03, -0.03, -0.03], "maximum": [0.03, 0.03, 0.03]}
    spec = _wide_spec(start=[0.5, 0.0, SHOULDER_Z, 0.0, 0.0, 0.0, 1.0], target=[0.4, 0.3, SHOULDER_Z],
                      contact=[0.0, 0.0, 0.0], bounds=box)
    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: {
        "robot_station": OLD_STATION, "robot_station_search": {"cleared": status == "clear", "results": [
            {"stance_digest": OLD_STATION["stance_digest"], "status": status,
             "blockers": [] if status == "clear" else ["robot_rigid_pick_place_unreachable:carry"]}]}})
    row = candidate_feasibility(scene_plan=_wide_plan(OLD_STATION, spec), catalog=_catalog("pi05_droid"),
                                profile={})["candidates"]["pi05_droid"]
    assert (row["status"], row["blockers"], row["reach_source"], row.get("flags")) == ("feasible", [], source, flags)


# A put-down the success contract accepts anywhere in a region: its nominal target 1.02 m from the shoulder,
# beyond the 0.990 m bound, while the region's nearest edge is 0.88 m away.
SMALL_BOX = {"minimum": [-0.03, -0.03, -0.03], "maximum": [0.03, 0.03, 0.03]}


def _region_spec():
    spec = _wide_spec(start=[0.6, 0.0, SHOULDER_Z, 0.0, 0.0, 0.0, 1.0], target=[1.02, 0.0, SHOULDER_Z],
                      contact=[0.0, 0.0, 0.0], bounds=SMALL_BOX)
    region = {"minimum": [0.88, -0.05, SHOULDER_Z - 0.01], "maximum": [1.02, 0.05, SHOULDER_Z + 0.01]}
    criteria = {"destination_containment": {"mode": "required", "position_bounds_world_m": region},
                "orientation": {"mode": "required", "reference_xyzw": [0.0, 0.0, 0.0, 1.0], "tolerance_rad": 0.0}}
    return {**spec, "task_success_contract": {"criteria": criteria}}


def test_a_nominal_put_down_beyond_reach_inside_an_accepted_region_in_reach_is_flagged_never_held(monkeypatch):
    from blueprint_pipeline.robot_arm_path_clearance import (
        NOMINAL_CONTACT_FLAG,
        contact_reach_status,
        object_reach_inputs,
        reach_refused,
        rigid_carry_verdict,
    )
    from blueprint_pipeline.team_scene_packet import _carrying_station

    spec = _region_spec()
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    place = next(row for row in reach["phases"] if row["phase"] == "place")
    assert place["distance_m"] == pytest.approx(1.02, abs=5e-4) and reach["task_contact_beyond"] == ["place"]
    assert place["contact_admissible_distance_m"] == pytest.approx(0.88, abs=5e-4)
    assert reach["task_contact_admissible_beyond"] == [] and not reach_refused(reach)
    assert contact_reach_status(reach) == "flagged"
    miss = {"status": "blocked", "blockers": ["robot_rigid_pick_place_unreachable:carry"]}
    verdict = rigid_carry_verdict(miss, reach)
    assert verdict["status"] == "flagged" and verdict["blockers"] == []
    assert verdict["flags"] == [SCRIPTED_CARRY_FLAG, NOMINAL_CONTACT_FLAG]
    assert verdict["advisories"][-1] == f"{NOMINAL_CONTACT_FLAG}:place:1.020m_vs_0.990m"
    # Planning agrees: feasible, flagged.
    from blueprint_pipeline import team_scene_packet

    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: {
        "robot_station": OLD_STATION, "robot_station_search": {"cleared": False, "results": [
            {"stance_digest": OLD_STATION["stance_digest"], **miss}]}})
    row = candidate_feasibility(scene_plan=_wide_plan(OLD_STATION, spec), catalog=_catalog("pi05_droid"),
                                profile={})["candidates"]["pi05_droid"]
    assert (row["status"], row["flags"]) == ("feasible", [SCRIPTED_CARRY_FLAG, NOMINAL_CONTACT_FLAG])
    # A stance that reaches the nominal target too ranks first, whatever the search's own rank.
    ahead = {**OLD_STATION, "position_world_m": [0.3, 0.0, 0.0], "stance_digest": "sha256:" + "a" * 64}
    tried = _scripted(monkeypatch, clears=False)
    station, record = _carrying_station([OLD_STATION, ahead], task_spec=spec, collision="unused.usda",
                                        profile=PANDA_RESET)
    assert tried == [ahead["stance_digest"], OLD_STATION["stance_digest"]] and station is ahead
    assert record["stances_in_reach"] == 2 and NOMINAL_CONTACT_FLAG not in record["flags"]
    # Alone, the region-only stance stands, flagged.
    station, record = _carrying_station([OLD_STATION], task_spec=spec, collision="unused.usda", profile=PANDA_RESET)
    assert station is OLD_STATION and record["flags"] == [SCRIPTED_CARRY_FLAG, NOMINAL_CONTACT_FLAG]


def test_without_the_objects_box_a_contact_beyond_the_bound_is_unassessed_and_flagged_never_held(monkeypatch):
    from blueprint_pipeline.robot_arm_path_clearance import (
        contact_reach_status,
        object_reach_inputs,
        rigid_carry_verdict,
    )

    spec = _wide_spec(bounds=None)
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    assert reach["status"] == "unassessed" and reach["task_contact_beyond"] == ["place"]
    assert contact_reach_status(reach) == "unassessed"
    verdict = rigid_carry_verdict({"status": "blocked", "blockers": ["robot_rigid_pick_place_unreachable:carry"]},
                                  reach)
    assert verdict["status"] == "flagged" and verdict["blockers"] == [] and verdict["flags"] == [SCRIPTED_CARRY_FLAG]
    from blueprint_pipeline import team_scene_packet

    monkeypatch.setattr(team_scene_packet, "recorded_station_search", lambda *_, **__: {})
    row = candidate_feasibility(scene_plan=_wide_plan(OLD_STATION, spec), catalog=_catalog("pi05_droid"),
                                profile={})["candidates"]["pi05_droid"]
    assert row["status"] == "unassessed" and row["blockers"] == ["packet_station_search_not_recorded"]


def test_the_recorded_capture_447_numbers_are_still_held_by_name():
    from blueprint_pipeline.robot_arm_path_clearance import (
        object_reach_inputs,
        reach_refusal_code,
        reach_refused,
    )

    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION,
                                  **object_reach_inputs(_wide_spec()))
    assert reach_refused(reach) and reach_refusal_code(reach) == CODE_447 == (
        "robot_reach_insufficient:place:1.086m_vs_0.990m")


def test_the_put_down_bound_never_exceeds_the_true_contact_distance_for_any_accepted_pose():
    from scipy.spatial.transform import Rotation

    from blueprint_pipeline.robot_arm_path_clearance import (
        _place_contact_distance,
        _place_contact_offset,
    )

    rng = np.random.default_rng(447)
    shoulder = np.array([0.0, 0.0, 0.333])
    for _ in range(300):
        target = rng.uniform([-1.2, -1.2, 0.0], [1.2, 1.2, 1.0])
        half = rng.uniform(0.0, 0.15, 3) * (rng.random() < 0.7)
        region = (target - half, target + half)
        contact = rng.uniform(-0.3, 0.3, 3)
        reference = Rotation.random(random_state=rng)
        angle = float(rng.choice([0.0, rng.uniform(0.0, math.pi), math.pi]))
        offset, grow = _place_contact_offset(contact, reference.as_quat().tolist(), angle)
        bound = _place_contact_distance(shoulder, target, offset, grow, region if half.any() else None)
        axes = rng.normal(size=(200, 3))
        turns = Rotation.from_rotvec(axes / np.linalg.norm(axes, axis=1, keepdims=True)
                                     * rng.uniform(0.0, angle, (200, 1)))
        origins = rng.uniform(region[0], region[1], (200, 3))
        for order in ((reference * turns), (turns * reference)):
            true = np.linalg.norm(origins + order.apply(contact) - shoulder, axis=1)
            assert bound <= true.min() + 1e-9


def _tilt_spec(tolerance_rad):
    # The contact 0.10 m above the object's origin; its nominal put-down 1.000 m from the shoulder, beyond the
    # bound. Tilting the object 0.3 rad toward the arm, which the contract accepts, brings it to 0.970 m.
    spec = _wide_spec(start=[0.6, 0.0, SHOULDER_Z - 0.1, 0.0, 0.0, 0.0, 1.0], target=[1.0, 0.0, SHOULDER_Z - 0.1],
                      contact=[0.0, 0.0, 0.1], bounds=SMALL_BOX)
    target = spec["target_position_world_m"]
    criteria = {"destination_containment": {"mode": "required",
                                            "position_bounds_world_m": {"minimum": target, "maximum": target}},
                "orientation": {"mode": "required", "reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                                "tolerance_rad": tolerance_rad}}
    return {**spec, "task_success_contract": {"criteria": criteria}}


def test_a_put_down_a_tilt_the_contract_accepts_brings_into_reach_is_flagged_not_held():
    from blueprint_pipeline.robot_arm_path_clearance import (
        contact_reach_status,
        object_reach_inputs,
        reach_refused,
    )

    tilted = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION,
                                   **object_reach_inputs(_tilt_spec(0.3)))
    place = next(row for row in tilted["phases"] if row["phase"] == "place")
    assert place["distance_m"] == pytest.approx(1.0, abs=5e-4) and tilted["task_contact_beyond"] == ["place"]
    # The horizontal-only swing measured 1.000 m here and held it; any axis moves it 2 * 0.1 * sin(0.15).
    assert place["contact_admissible_distance_m"] == pytest.approx(1.0 - 2 * 0.1 * math.sin(0.15), abs=2e-4)
    assert not reach_refused(tilted) and contact_reach_status(tilted) == "flagged"
    # No tilt accepted: the nominal put-down is the only one, and it is beyond the bound.
    upright = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION,
                                    **object_reach_inputs(_tilt_spec(0.0)))
    assert reach_refused(upright)


def test_the_put_down_contact_is_turned_by_the_destination_orientation_or_proves_nothing():
    from blueprint_pipeline.robot_arm_path_clearance import (
        contact_reach_status,
        object_reach_inputs,
        reach_refused,
    )

    # Picked up facing +x, put down turned half a turn: the contact 0.10 m along the object's x then faces the arm.
    spec = _wide_spec(start=[0.6, 0.0, SHOULDER_Z, 0.0, 0.0, 0.0, 1.0], target=[1.0, 0.0, SHOULDER_Z],
                      contact=[0.1, 0.0, 0.0], bounds=SMALL_BOX)
    turned = {**spec, "destination_orientation_xyzw": [0.0, 0.0, 1.0, 0.0]}
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(turned))
    place = next(row for row in reach["phases"] if row["phase"] == "place")
    assert place["distance_m"] == pytest.approx(1.10, abs=5e-4)  # turned as it started: the nominal figure
    assert place["contact_admissible_distance_m"] == pytest.approx(0.90, abs=5e-4)
    assert not reach_refused(reach) and contact_reach_status(reach) == "flagged"
    # Held as it started, the same contact is 1.10 m away at the put-down: refused.
    same = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    assert reach_refused(same)
    # With no destination orientation the contact cannot be located at the put-down: no proof there.
    unknown = {key: value for key, value in spec.items() if key != "destination_orientation_xyzw"}
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(unknown))
    place = next(row for row in reach["phases"] if row["phase"] == "place")
    assert place["place_contact_unlocated"] is True and "contact_admissible_distance_m" not in place
    assert not reach_refused(reach) and contact_reach_status(reach) == "flagged"


def test_capture_447_with_the_websites_planar_tolerance_is_held_at_its_least_accepted_distance():
    from blueprint_pipeline.robot_arm_path_clearance import (
        object_reach_inputs,
        reach_refusal_code,
        reach_refused,
    )

    target = np.asarray(WIDE_TARGET)
    spec = {**_wide_spec(), "destination_position_bounds_world_m": {
        "minimum": (target - [0.05, 0.05, 0.0]).tolist(), "maximum": (target + [0.05, 0.05, 0.0]).tolist()}}
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    # The nearest accepted put-down, 5 cm nearer on each planar axis: sqrt(0.05^2 + 1.0314^2) = 1.0326 m.
    assert reach_refused(reach)
    assert reach_refusal_code(reach) == "robot_reach_insufficient:place:1.032m_vs_0.990m"


def test_the_put_down_nearest_point_bound_never_exceeds_the_true_distance_for_any_accepted_pose():
    from scipy.spatial.transform import Rotation

    from blueprint_pipeline.robot_arm_path_clearance import (
        _nearest_box_point,
        _place_nearest_distance,
    )

    rng = np.random.default_rng(4470)
    shoulder = np.array([0.0, 0.0, 0.333])
    for _ in range(300):
        target = rng.uniform([-1.2, -1.2, 0.0], [1.2, 1.2, 1.0])
        half = rng.uniform(0.0, 0.15, 3) * (rng.random() < 0.7)
        region = (target - half, target + half) if half.any() else None
        lower = -rng.uniform(0.01, 0.4, 3) + rng.uniform(-0.1, 0.1, 3)
        upper = lower + rng.uniform(0.02, 0.6, 3)
        reference = Rotation.random(random_state=rng)
        angle = float(rng.choice([0.0, rng.uniform(0.0, math.pi), math.pi]))
        bound = _place_nearest_distance(shoulder, target, reference.as_matrix(), lower, upper, region, angle)
        axes = rng.normal(size=(100, 3))
        turns = Rotation.from_rotvec(axes / np.linalg.norm(axes, axis=1, keepdims=True)
                                     * rng.uniform(0.0, angle, (100, 1)))
        lo, hi = (target, target) if region is None else region
        origins = rng.uniform(lo, hi, (100, 3))
        for order in ((reference * turns), (turns * reference)):
            for origin, rotation in zip(origins, order.as_matrix()):
                true = np.linalg.norm(_nearest_box_point(shoulder, origin, rotation, lower, upper) - shoulder)
                assert bound <= true + 1e-9
    # Without a destination orientation and less than any turn accepted, the put-down proves nothing.
    assert _place_nearest_distance(shoulder, target, None, lower, upper, None, 0.2) is None


def test_a_put_down_nearest_point_a_tilt_or_the_destination_orientation_brings_into_reach_is_not_held():
    from blueprint_pipeline.robot_arm_path_clearance import object_reach_inputs, reach_refused

    # A 0.6 m bar pointing away from the arm, its origin at its near end 1.00 m from the shoulder (beyond the
    # bound). Picked up pointing away; put down turned half a turn, it points back toward the arm.
    bar = {"minimum": [0.0, -0.02, -0.02], "maximum": [0.6, 0.02, 0.02]}
    spec = _wide_spec(start=[0.3, 0.0, SHOULDER_Z, 0.0, 0.0, 0.0, 1.0], target=[1.0, 0.0, SHOULDER_Z],
                      contact=[0.3, 0.0, 0.0], bounds=bar)
    held = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(spec))
    assert held["beyond"] == ["place"] and reach_refused(held)
    turned = {**spec, "destination_orientation_xyzw": [0.0, 0.0, 1.0, 0.0]}
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(turned))
    place = next(row for row in reach["phases"] if row["phase"] == "place")
    assert place["object_nearest_distance_m"] == pytest.approx(0.40, abs=1e-3) and reach["beyond"] == []
    # Unknown destination orientation: the put-down proves nothing; the pick, where the bar is, still measures.
    unknown = {key: value for key, value in spec.items() if key != "destination_orientation_xyzw"}
    reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=OLD_STATION, **object_reach_inputs(unknown))
    place = next(row for row in reach["phases"] if row["phase"] == "place")
    assert place["place_nearest_unlocated"] is True and reach["beyond"] == [] and not reach_refused(reach)
