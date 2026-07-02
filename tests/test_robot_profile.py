"""Hermetic tests for scene_placement.robot_profile.

NO GPU, NO network, NO isaacsim. A RobotProfile makes robot embodiment data,
not code: footprint, pelvis height, reach geometry, and spawn/prim metadata all
live in one swappable object so placement works for robots other than the G1.
"""
from __future__ import annotations

import json
import math

import pytest

from blueprint_pipeline.scene_placement.robot_profile import (
    RobotProfile,
    get_robot_profile,
    known_robot_ids,
    register_robot_profile,
    robot_profile_from_dict,
    robot_profile_from_json_file,
)
from blueprint_pipeline.scene_placement.placement import compute_stand_pose
from blueprint_pipeline.scene_placement.types import SceneObject
from blueprint_pipeline.scene_placement.validation import validate_stand_pose


def _obj(id_, label, *, cx=0.0, cy=0.0, cz=0.0, sx=1.0, sy=1.0, sz=1.0) -> SceneObject:
    return SceneObject(
        id=id_,
        label=label,
        bbox_min=(cx - sx / 2, cy - sy / 2, cz - sz / 2),
        bbox_max=(cx + sx / 2, cy + sy / 2, cz + sz / 2),
        centroid=(cx, cy, cz),
    )


def _always_clear_probe(pose, yaw):  # noqa: ANN001 - Probe protocol: 0 = clear floor
    return 0


# ----------------------------- registry -----------------------------

def test_g1_profile_matches_runner_constants():
    """The built-in G1 profile must pin the values the GPU runner uses today."""
    p = get_robot_profile("unitree_g1")
    assert p.robot_id == "unitree_g1"
    assert p.pelvis_height_m == pytest.approx(0.79)
    assert p.footprint_half_extent_xyz == pytest.approx((0.12, 0.23, 0.62))
    assert p.arm_span_m == pytest.approx(0.45)
    assert p.shoulder_lateral_offset_m == pytest.approx(0.16)
    assert p.shoulder_above_root_m == pytest.approx(0.29)
    assert p.usd_prim_path == "/World/G1"


def test_unknown_robot_id_raises_with_known_ids():
    with pytest.raises(KeyError) as ei:
        get_robot_profile("totally_unknown_bot")
    assert "unitree_g1" in str(ei.value)


def test_register_and_fetch_custom_profile():
    p = RobotProfile(robot_id="test_wide_bot", pelvis_height_m=1.1,
                     footprint_half_extent_xyz=(0.4, 0.5, 0.9))
    register_robot_profile(p)
    assert get_robot_profile("test_wide_bot") is p
    assert "test_wide_bot" in known_robot_ids()


# ----------------------------- dict / json loading -----------------------------

def test_profile_from_dict_defaults_and_overrides():
    p = robot_profile_from_dict({"robot_id": "h1_like", "pelvis_height_m": 1.05})
    assert p.robot_id == "h1_like"
    assert p.pelvis_height_m == pytest.approx(1.05)
    # unspecified fields fall back to sane defaults, not G1-specific prim paths
    assert p.arm_span_m > 0


def test_profile_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError) as ei:
        robot_profile_from_dict({"robot_id": "x", "pelvis_heigth_m": 1.0})
    assert "pelvis_heigth_m" in str(ei.value)


def test_profile_from_dict_requires_robot_id():
    with pytest.raises(ValueError):
        robot_profile_from_dict({"pelvis_height_m": 1.0})


def test_profile_from_json_file(tmp_path):
    f = tmp_path / "bot.json"
    f.write_text(json.dumps({"robot_id": "file_bot", "arm_span_m": 0.8}))
    p = robot_profile_from_json_file(f)
    assert p.robot_id == "file_bot"
    assert p.arm_span_m == pytest.approx(0.8)


def test_profile_round_trips_via_to_dict():
    p = get_robot_profile("unitree_g1")
    q = robot_profile_from_dict(p.to_dict())
    assert q == p


# ----------------------------- derived reach -----------------------------

def test_max_shoulder_to_affordance_derives_from_arm_span():
    short = robot_profile_from_dict({"robot_id": "short_arms", "arm_span_m": 0.3})
    long = robot_profile_from_dict({"robot_id": "long_arms", "arm_span_m": 0.9})
    assert long.max_shoulder_to_affordance_m() > short.max_shoulder_to_affordance_m()


# ----------------------------- placement integration -----------------------------

def test_compute_stand_pose_uses_profile_pelvis_height():
    target = _obj("sink", "sink", cx=2.0, cy=2.0, cz=0.9)
    tall = robot_profile_from_dict({"robot_id": "tall_bot", "pelvis_height_m": 1.2})
    pose = compute_stand_pose(target, probe=_always_clear_probe, floor_z=0.0,
                              robot_profile=tall)
    assert pose.position[2] == pytest.approx(1.2)


def test_compute_stand_pose_kwarg_overrides_profile():
    target = _obj("sink", "sink", cx=2.0, cy=2.0, cz=0.9)
    tall = robot_profile_from_dict({"robot_id": "tall_bot2", "pelvis_height_m": 1.2})
    pose = compute_stand_pose(target, probe=_always_clear_probe, floor_z=0.0,
                              pelvis_height=0.5, robot_profile=tall)
    assert pose.position[2] == pytest.approx(0.5)


def test_bigger_robot_stands_farther_out():
    target = _obj("counter", "counter", cx=0.0, cy=0.0, cz=0.5, sx=1.0, sy=1.0, sz=1.0)
    small = robot_profile_from_dict(
        {"robot_id": "small_bot", "standing_distance_m": 0.3})
    big = robot_profile_from_dict(
        {"robot_id": "big_bot", "standing_distance_m": 0.9})
    pose_small = compute_stand_pose(target, probe=_always_clear_probe, floor_z=0.0,
                                    robot_profile=small)
    pose_big = compute_stand_pose(target, probe=_always_clear_probe, floor_z=0.0,
                                  robot_profile=big)
    d_small = math.dist(pose_small.position[:2], (0.0, 0.0))
    d_big = math.dist(pose_big.position[:2], (0.0, 0.0))
    assert d_big > d_small


# ----------------------------- validation integration -----------------------------

def test_validate_stand_pose_uses_profile_footprint():
    """A wide robot clips an obstacle that a narrow robot clears."""
    target = _obj("sink", "sink", cx=2.0, cy=0.0, cz=0.9)
    # floor-reaching obstacle 0.45m to the robot's side
    obstacle = _obj("cab", "cabinet", cx=0.0, cy=0.60, cz=0.4, sx=0.4, sy=0.4, sz=0.8)
    narrow = robot_profile_from_dict(
        {"robot_id": "narrow_bot", "footprint_half_extent_xyz": (0.12, 0.23, 0.62)})
    wide = robot_profile_from_dict(
        {"robot_id": "wide_bot", "footprint_half_extent_xyz": (0.12, 0.75, 0.62)})
    common = dict(target=target, obstacles=[obstacle], floor_z=0.0,
                  standoff_range=(0.1, 5.0))
    v_narrow = validate_stand_pose((0.0, 0.0, 0.79), 0.0, robot_profile=narrow, **common)
    v_wide = validate_stand_pose((0.0, 0.0, 0.79), 0.0, robot_profile=wide, **common)
    assert v_narrow.ok
    assert not v_wide.ok
    assert any(obst_id == "cab" for obst_id, _area in v_wide.clipping)


def test_validate_stand_pose_uses_profile_pelvis_height():
    target = _obj("sink", "sink", cx=2.0, cy=0.0, cz=0.9)
    tall = robot_profile_from_dict({"robot_id": "tall_bot3", "pelvis_height_m": 1.2})
    # standing at G1 pelvis height with a tall-bot profile must fail on-floor check
    v = validate_stand_pose((0.0, 0.0, 0.79), 0.0, target=target, obstacles=[],
                            floor_z=0.0, robot_profile=tall,
                            standoff_range=(0.1, 5.0))
    assert not v.ok
    v2 = validate_stand_pose((0.0, 0.0, 1.2), 0.0, target=target, obstacles=[],
                             floor_z=0.0, robot_profile=tall,
                             standoff_range=(0.1, 5.0))
    assert v2.ok
