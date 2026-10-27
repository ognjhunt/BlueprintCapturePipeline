"""Before any construction, whether the DROID Franka can carry a prepared rigid task."""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.website_rigid_feasibility import rigid_task_verdict
from tests.test_team_scene_packet import _box, _stage

FLOOR = {"description": "floor", "aabb_min_xyz": [-2.0, -2.0, 0.0], "aabb_max_xyz": [4.0, 4.0, 0.0]}
DESK_TOP = {"description": "desk top", "aabb_min_xyz": [0.6, -0.5, 0.74], "aabb_max_xyz": [1.2, 0.5, 0.74]}


def _room(tmp_path: Path) -> Path:
    room = _stage(tmp_path / "collision.usda", "/Root")
    _box(room, "/Root/Floor", (-2.0, -2.0, 0.0), (4.0, 4.0, 0.0))
    # A desk with cabinets under it, standing on the floor.
    _box(room, "/Root/Desk", (0.6, -0.5, 0.0), (1.2, 0.5, 0.74))
    room.GetRootLayer().Save()
    return tmp_path / "collision.usda"


def _preparation(destination):
    return {"digest": "sha256:" + "1" * 64,
            "intake_request": {"task": {"strategy": "pick_and_place", "support": FLOOR,
                                        "success": {"pregrasp_clearance_m": 0.1}}},
            "subject": {"aabb_min_xyz": [0.2675, -0.0325, 0.0], "aabb_max_xyz": [0.3325, 0.0325, 0.065]},
            "destination": destination}


def test_a_ball_on_the_floor_cannot_be_carried_onto_a_desk_from_a_floor_mount(tmp_path):
    verdict = rigid_task_verdict(preparation=_preparation(
        {"relation": "on", "position_world_m": [0.9, 0.0, 0.74], "support": DESK_TOP}), collision_path=_room(tmp_path))
    assert verdict["status"] == "abstained" and verdict["mount_rule"] == "level_with_lower_support"
    # Several stations reach the ball; none can lift it to the desk: the robot's verdict, not one station's.
    assert verdict["blockers"] == ["robot_rigid_pick_place_unreachable:lift"] and verdict["stations_tried"] > 1
    # The mount stands on the floor: level with the lower of the two surfaces.
    assert verdict["station"]["position_world_m"][2] == pytest.approx(0.0, abs=1e-6)
    assert verdict["verdict_digest"].startswith("sha256:")


def test_the_same_ball_moved_across_the_floor_is_cleared(tmp_path):
    verdict = rigid_task_verdict(preparation=_preparation({"relation": "on", "position_world_m": [0.3, 0.35, 0.0]}),
                                 collision_path=_room(tmp_path))
    assert verdict["status"] == "clear" and verdict["blockers"] == [], verdict["blockers"]
    assert verdict["reach"]["cleared_grip"] is not None
