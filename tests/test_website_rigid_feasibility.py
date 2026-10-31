"""Before any construction, whether an enabled embodiment can carry a prepared rigid task."""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.website_rigid_feasibility import enabled_robot_profiles, rigid_task_verdict
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


def test_a_ball_on_the_floor_cannot_be_carried_onto_a_desk_from_a_floor_mount(tmp_path):
    verdict = rigid_task_verdict(preparation=_preparation(DESK), collision_path=_room(tmp_path),
                                 robot_profiles=[DROID])
    assert verdict["status"] == "abstained" and verdict["mount_rule"] == "level_with_lower_support"
    # Several stations and every grasp were tried; none lifts it to the desk: the robot's verdict.
    assert len(verdict["blockers"]) == 1 and verdict["blockers"][0].endswith(":lift"), verdict["blockers"]
    assert verdict["stations_tried"] > 1 and verdict["grasps_tried"] == 9
    assert verdict["embodiments"] == [{"robot_id": "franka_panda", "status": "blocked",
                                       "blockers": verdict["blockers"], "grasp": verdict["reach"]["grasp"]}]
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
    top_down = [row for row in verdict["station_results"] if row["grasp"] == "top_down"]
    assert top_down and {tuple(row["blockers"]) for row in top_down} == {
        ("robot_rigid_pick_place_scene_strike:robotiq_2f85_body:approach",)}


def test_an_embodiment_the_gate_cannot_model_is_unassessed_never_refused(tmp_path):
    collision = _room(tmp_path)
    mixed = rigid_task_verdict(preparation=_preparation(DESK), collision_path=collision,
                               robot_profiles=[DROID, "team_mobile_manipulator_v1"])
    # The Franka cannot do it; the other robot was never measured, so nothing is refused.
    assert mixed["status"] == "unassessed" and mixed["unassessed_robot_profiles"] == ["team_mobile_manipulator_v1"]
    assert mixed["blockers"][-1] == "robot_profile_unmodeled:team_mobile_manipulator_v1"
    only = rigid_task_verdict(preparation=_preparation(DESK), collision_path=collision,
                              robot_profiles=["team_mobile_manipulator_v1"])
    assert only["status"] == "unassessed" and only["station"] is None and only["robot_id"] is None
    none = rigid_task_verdict(preparation=_preparation(DESK), collision_path=collision, robot_profiles=[])
    assert none["status"] == "abstained" and none["blockers"] == ["robot_profile_none_enabled"]


def test_the_default_embodiments_are_the_enabled_screening_candidates():
    assert DROID in enabled_robot_profiles()


def test_a_room_with_no_station_abstains_by_name_instead_of_crashing(tmp_path, monkeypatch):
    from blueprint_pipeline import team_embodiment_stance
    from blueprint_pipeline.team_embodiment import TeamEmbodimentError

    def nowhere(**_kwargs):
        raise TeamEmbodimentError(["stance_no_clear_floor"])

    monkeypatch.setattr(team_embodiment_stance, "ground_stance", nowhere)
    verdict = rigid_task_verdict(preparation=_preparation(DESK), collision_path=_room(tmp_path), robot_profiles=[DROID])
    assert verdict["status"] == "abstained" and verdict["blockers"] == ["robot_station_stance_no_clear_floor"]
