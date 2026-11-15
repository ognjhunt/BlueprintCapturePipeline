"""Before any construction, whether an enabled embodiment can carry a prepared rigid task."""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.website_rigid_feasibility import DEFAULT_ROBOT_PROFILES, rigid_task_verdict
from tests.test_team_scene_packet import _box, _stage

FLOOR = {"description": "floor", "aabb_min_xyz": [-2.0, -2.0, 0.0], "aabb_max_xyz": [4.0, 4.0, 0.0]}
DESK_TOP = {"description": "desk top", "aabb_min_xyz": [0.6, -0.5, 0.74], "aabb_max_xyz": [1.2, 0.5, 0.74]}
TABLE_TOP = {"description": "table top", "aabb_min_xyz": [0.40, -0.6, 0.40], "aabb_max_xyz": [0.90, 0.6, 0.40]}
DROID = "droid_franka_panda_robotiq_2f85_v1"


def _room(tmp_path: Path) -> Path:
    room = _stage(tmp_path / "collision.usda", "/Root")
    _box(room, "/Root/Floor", (-2.0, -2.0, 0.0), (4.0, 4.0, 0.0))
    # A desk with cabinets under it, standing on the floor.
    _box(room, "/Root/Desk", (0.6, -0.5, 0.0), (1.2, 0.5, 0.74))
    room.GetRootLayer().Save()
    return tmp_path / "collision.usda"


def _preparation(destination, *, support=FLOOR, subject=([0.2675, -0.0325, 0.0], [0.3325, 0.0325, 0.065])):
    return {"digest": "sha256:" + "1" * 64,
            "intake_request": {"task": {"strategy": "pick_and_place", "support": support,
                                        "success": {"pregrasp_clearance_m": 0.1}}},
            "subject": {"aabb_min_xyz": subject[0], "aabb_max_xyz": subject[1]},
            "destination": destination}


DESK = {"relation": "on", "position_world_m": [0.9, 0.0, 0.74], "support": DESK_TOP}


def test_a_ball_the_scripted_carry_cannot_lift_onto_a_desk_is_flagged_never_refused(tmp_path):
    verdict = rigid_task_verdict(preparation=_preparation(DESK), collision_path=_room(tmp_path),
                                 robot_profiles=[DROID])
    assert verdict["status"] == "flagged" and verdict["mount_rule"] == "level_with_lower_support"
    # Several stations and every grasp were tried; none lifts it to the desk. The ball and the desk are within
    # the arm's reach, so a learned policy may still: the miss is a flag, and the job goes on.
    assert verdict["blockers"] == [] and verdict["flags"] == ["scripted_carry_path_not_found"]
    assert len(verdict["scripted_carry_blockers"]) == 1 and verdict["scripted_carry_blockers"][0].endswith(":lift")
    assert verdict["stations_tried"] > 1 and verdict["grasps_tried"] == 9
    assert verdict["object_reach"]["status"] == "within_bound" and verdict["stances_in_reach"] > 0
    assert verdict["embodiments"] == [{"robot_id": "franka_panda", "status": "blocked",
                                       "blockers": verdict["scripted_carry_blockers"],
                                       "grasp": verdict["reach"]["grasp"], "object_reach": "within_bound",
                                       "stances_in_reach": verdict["stances_in_reach"],
                                       "flags": ["scripted_carry_path_not_found"]}]
    # The mount stands on the floor: level with the lower of the two surfaces.
    assert verdict["station"]["position_world_m"][2] == pytest.approx(0.0, abs=1e-6)
    assert verdict["verdict_digest"].startswith("sha256:")


def test_the_same_ball_moved_across_the_floor_is_cleared_top_down_first(tmp_path):
    verdict = rigid_task_verdict(preparation=_preparation({"relation": "on", "position_world_m": [0.3, 0.35, 0.0]}),
                                 collision_path=_room(tmp_path), robot_profiles=[DROID])
    assert verdict["status"] == "clear" and verdict["blockers"] == [], verdict["blockers"]
    assert verdict["reach"]["cleared_grip"] is not None and verdict["reach"]["grasp"] == "top_down"


def test_a_region_evaluation_refused_in_preparation_rides_the_verdict_as_a_risk(tmp_path):
    collision = _room(tmp_path)
    preparation = _preparation({"relation": "on", "position_world_m": [0.3, 0.35, 0.0]})
    plain = rigid_task_verdict(preparation=preparation, collision_path=collision, robot_profiles=[DROID])
    assert plain["status"] == "clear" and "risks" not in plain
    preparation["surface_target_admission"] = {"status": "refused", "code": "surface_target_initial_overlap",
                                               "basis": "evaluation_admission_rule"}
    flagged = rigid_task_verdict(preparation=preparation, collision_path=collision, robot_profiles=[DROID])
    # The carry still clears; the verdict is no longer a plain "clear" over a known evaluation refusal.
    assert flagged["status"] == "clear" and flagged["blockers"] == []
    assert flagged["risks"] == [{"code": "surface_target_initial_overlap", "basis": "evaluation_admission_rule"}]
    assert {key: value for key, value in flagged.items() if key not in {"risks", "verdict_digest"}} == {
        key: value for key, value in plain.items() if key != "verdict_digest"}


def test_a_lip_over_the_object_refuses_top_down_and_a_tilted_grasp_carries_it(tmp_path):
    room = _stage(tmp_path / "collision.usda", "/Root")
    _box(room, "/Root/Floor", (-2.0, -2.0, 0.0), (4.0, 4.0, 0.0))
    _box(room, "/Root/Table", (0.40, -0.6, 0.0), (0.90, 0.6, 0.40))
    # A cabinet lip 14 cm above a 16 cm bottle: the hand cannot come down to it from above.
    _box(room, "/Root/Lip", (0.50, -0.34, 0.70), (0.66, -0.16, 0.72))
    room.GetRootLayer().Save()
    verdict = rigid_task_verdict(
        preparation=_preparation({"relation": "on", "position_world_m": [0.55, 0.30, 0.40]}, support=TABLE_TOP,
                                 subject=([0.52, -0.28, 0.40], [0.58, -0.22, 0.56])),
        collision_path=tmp_path / "collision.usda", robot_profiles=[DROID])
    assert verdict["status"] == "clear" and verdict["reach"]["grasp"] not in (None, "top_down")
    # Top-down clears from no station: the lip strikes the hand on the way down; the stations a footprint
    # apart on the table's far side are out of the arm's reach for it.
    top_down = [row for row in verdict["station_results"] if row["grasp"] == "top_down"]
    assert top_down and all(row["status"] == "blocked" for row in top_down)
    assert ("robot_rigid_pick_place_scene_strike:robotiq_2f85_body:approach",) in {
        tuple(row["blockers"]) for row in top_down}


def test_an_embodiment_the_gate_cannot_model_is_unassessed_never_refused(tmp_path):
    collision = _room(tmp_path)
    mixed = rigid_task_verdict(preparation=_preparation(DESK), collision_path=collision,
                               robot_profiles=[DROID, "team_mobile_manipulator_v1"])
    # The Franka's scripted carry misses it (a flag); the other robot was never measured: nothing is refused.
    assert mixed["status"] == "flagged" and mixed["unassessed_robot_profiles"] == ["team_mobile_manipulator_v1"]
    assert mixed["blockers"] == []
    far = rigid_task_verdict(preparation=_far_preparation(), collision_path=_wide_table(tmp_path / "wide"),
                             robot_profiles=[DROID, "team_mobile_manipulator_v1"])
    # Out of the Franka's reach from every clear stance; the other robot was never measured: unassessed.
    assert far["status"] == "unassessed"
    assert far["blockers"][-1] == "robot_profile_unmodeled:team_mobile_manipulator_v1"
    only = rigid_task_verdict(preparation=_preparation(DESK), collision_path=collision,
                              robot_profiles=["team_mobile_manipulator_v1"])
    assert only["status"] == "unassessed" and only["station"] is None and only["robot_id"] is None
    none = rigid_task_verdict(preparation=_preparation(DESK), collision_path=collision, robot_profiles=[])
    assert none["status"] == "abstained" and none["blockers"] == ["robot_profile_none_enabled"]


def _wide_table(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    room = _stage(tmp_path / "collision.usda", "/Root")
    _box(room, "/Root/Floor", (-3.0, -3.0, 0.0), (3.0, 3.0, 0.0))
    _box(room, "/Root/Table", (-1.2, -1.2, 0.0), (1.2, 1.2, 0.74))
    room.GetRootLayer().Save()
    return tmp_path / "collision.usda"


def _far_preparation():
    # A box in the middle of a 2.4 m table, moved 20 cm along it: no point of it is ever within the arm's reach
    # from a pedestal beside the table.
    wide = {"description": "wide table top", "aabb_min_xyz": [-1.2, -1.2, 0.74], "aabb_max_xyz": [1.2, 1.2, 0.74]}
    return _preparation({"relation": "on", "position_world_m": [0.2, 0.0, 0.74]}, support=wide,
                        subject=([-0.05, -0.05, 0.74], [0.05, 0.05, 0.80]))


def test_an_object_out_of_reach_from_every_clear_stance_abstains_by_the_hard_reach_rule(tmp_path):
    verdict = rigid_task_verdict(preparation=_far_preparation(), collision_path=_wide_table(tmp_path),
                                 robot_profiles=[DROID])
    assert verdict["status"] == "abstained" and verdict["stances_in_reach"] == 0
    assert verdict["object_reach"]["status"] == "out_of_reach" and verdict["object_reach"]["grasp_independent"]
    assert len(verdict["blockers"]) == 1 and verdict["blockers"][0].startswith("robot_reach_insufficient:")
    assert verdict["blockers"][0].endswith("m_vs_0.990m") and "flags" not in verdict


def test_the_default_embodiment_is_the_droid_arm():
    assert DEFAULT_ROBOT_PROFILES == (DROID,)


def test_a_room_with_no_station_abstains_by_name_instead_of_crashing(tmp_path, monkeypatch):
    from blueprint_pipeline import team_embodiment_stance
    from blueprint_pipeline.team_embodiment import TeamEmbodimentError

    def nowhere(**_kwargs):
        raise TeamEmbodimentError(["stance_no_clear_floor"])

    monkeypatch.setattr(team_embodiment_stance, "ground_stance", nowhere)
    verdict = rigid_task_verdict(preparation=_preparation(DESK), collision_path=_room(tmp_path), robot_profiles=[DROID])
    assert verdict["status"] == "abstained" and verdict["blockers"] == ["robot_station_stance_no_clear_floor"]


def _bar():
    # A 1.2 m bar on the wide table, its far end nearest the table's edge: some of it is within the arm's reach
    # from a pedestal beside the table, its centre (the intake's assumed contact) is not.
    wide = {"description": "wide table top", "aabb_min_xyz": [-1.2, -1.2, 0.74], "aabb_max_xyz": [1.2, 1.2, 0.74]}
    return _preparation({"relation": "on", "position_world_m": [0.1, 0.0, 0.74]}, support=wide,
                        subject=([-0.6, -0.03, 0.74], [0.6, 0.03, 0.80]))


def test_a_long_object_with_its_near_end_in_reach_and_its_centre_beyond_is_flagged_and_goes_on(tmp_path):
    # No task contact is sealed at intake: a centre beyond the bound proves nothing, so the job is not held.
    # The advisory names the assumed contact's numbers; the team half, with the task's own contact, decides.
    from blueprint_pipeline.robot_arm_path_clearance import SCRIPTED_CARRY_FLAG
    from blueprint_pipeline.website_rigid_feasibility import ASSUMED_CONTACT_FLAG

    verdict = rigid_task_verdict(preparation=_bar(), collision_path=_wide_table(tmp_path), robot_profiles=[DROID])
    assert verdict["status"] == "flagged" and verdict["blockers"] == [] and verdict["stances_in_reach"] > 0
    reach = verdict["object_reach"]
    assert reach["status"] == "within_bound" and reach["beyond"] == [] and reach["task_contact_beyond"]
    assert verdict["flags"] == [SCRIPTED_CARRY_FLAG, ASSUMED_CONTACT_FLAG]
    assert verdict["advisories"] and all(code.startswith("robot_reach_assumed_contact_beyond:")
                                         and code.endswith("m_vs_0.990m") for code in verdict["advisories"])
    [entry] = verdict["embodiments"]
    assert entry["assumed_contact_beyond"] == reach["task_contact_beyond"] and "reach_held" not in entry
    # The robot-fit report reads it as the verdict does: flagged, so unknown, never a pass or a refusal.
    from blueprint_pipeline.website_robot_fit import _verdict_reach

    assert _verdict_reach(verdict, "franka_panda")["result"] == "unknown"


def test_the_robot_fit_report_fails_only_where_the_verdict_holds_on_its_reach_proof(tmp_path):
    from blueprint_pipeline.website_robot_fit import _verdict_reach

    held = rigid_task_verdict(preparation=_far_preparation(), collision_path=_wide_table(tmp_path / "far"),
                              robot_profiles=[DROID])
    assert held["status"] == "abstained" and held["embodiments"][0]["reach_held"] is True
    assert _verdict_reach(held, "franka_panda")["result"] == "fail"
    # In reach with the centre too: the scripted carry clears or is only flagged, and both reports pass.
    clear = rigid_task_verdict(preparation=_preparation({"relation": "on", "position_world_m": [0.5, 0.3, 0.0]}),
                               collision_path=_room(tmp_path), robot_profiles=[DROID])
    assert clear["status"] in {"clear", "flagged"} and "assumed_contact_beyond" not in clear["embodiments"][0]
    assert _verdict_reach(clear, "franka_panda")["result"] == "pass"
