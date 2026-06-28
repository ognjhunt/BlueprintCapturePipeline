"""Hermetic tests for the geometric placement validator (no GPU/render/network)."""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline.scene_placement import (
    PlacementVerdict,
    SceneObject,
    StandPose,
    validate_placement,
    validate_stand_pose,
)
from blueprint_pipeline.scene_placement.validation import (
    _angle_diff_deg,
    _xy_box_gap,
    _xy_overlap_area,
)


def _obj(oid, cx, cy, cz, sx=0.4, sy=0.4, sz=0.4):
    return SceneObject(
        id=oid, label=oid,
        bbox_min=(cx - sx / 2, cy - sy / 2, cz - sz / 2),
        bbox_max=(cx + sx / 2, cy + sy / 2, cz + sz / 2),
        centroid=(cx, cy, cz),
    )


# ----------------------------- pure helpers -----------------------------

def test_xy_overlap_area():
    assert _xy_overlap_area((0, 0), (2, 2), (1, 1), (3, 3)) == pytest.approx(1.0)   # 1x1 overlap
    assert _xy_overlap_area((0, 0), (1, 1), (2, 2), (3, 3)) == 0.0                   # disjoint
    assert _xy_overlap_area((0, 0), (1, 1), (1, 0), (2, 1)) == 0.0                   # edge-touch -> 0


def test_xy_box_gap():
    assert _xy_box_gap((0, 0), (1, 1), (2, 0), (3, 1)) == pytest.approx(1.0)         # 1m gap in x
    assert _xy_box_gap((0, 0), (1, 1), (0.5, 0.5), (2, 2)) == 0.0                     # overlapping -> 0
    assert _xy_box_gap((0, 0), (1, 1), (4, 5), (5, 6)) == pytest.approx(5.0)          # diagonal gap (3,4)->5


def test_angle_diff_deg():
    assert _angle_diff_deg(0.0, 0.0) == pytest.approx(0.0)
    assert _angle_diff_deg(math.pi, 0.0) == pytest.approx(180.0)
    assert _angle_diff_deg(0.1, -0.1) == pytest.approx(math.degrees(0.2))
    # wraparound: +179 vs -179 is 2 degrees apart, not 358
    assert _angle_diff_deg(math.radians(179), math.radians(-179)) == pytest.approx(2.0, abs=1e-6)


# ----------------------------- the four checks -----------------------------

def _sink_and_counter():
    target = _obj("sink", 2.28, 1.33, 0.85, sx=0.4, sy=0.4, sz=0.3)
    counter = _obj("counter", 2.28, 1.33, 0.45, sx=1.5, sy=0.6, sz=0.9)  # floor->0.9, y in [1.03,1.63]
    return target, counter


def test_clip_detected_when_footprint_overlaps_counter():
    target, counter = _sink_and_counter()
    # robot standing AT the counter (y=1.2): footprint y[0.92,1.48] overlaps counter y[1.03,1.63]
    v = validate_stand_pose((2.28, 1.2, 0.84), math.pi / 2, target, [target, counter], floor_z=0.05)
    assert v.ok is False
    assert any(f.startswith("clips:") for f in v.failures)
    assert any(oid == "counter" for oid, _ in v.clipping)


def test_no_clip_when_standing_in_front():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [target, counter], floor_z=0.05)
    assert v.clipping == []
    assert "clips" not in " ".join(v.failures)


def test_wall_box_above_pelvis_is_not_a_clip():
    target, _ = _sink_and_counter()
    # a box mounted high on the wall (min_z 1.5 > pelvis 0.84) directly above the robot
    wall_box = _obj("kitchen_box", 2.28, 0.55, 1.6, sx=0.4, sy=0.4, sz=0.2)
    v = validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [target, wall_box], floor_z=0.05)
    assert v.clipping == []          # stood UNDER it, not into it


def test_off_floor_flagged():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((2.28, 0.55, 1.5), math.pi / 2, target, [target, counter], floor_z=0.05)
    assert v.on_floor is False
    assert any(f.startswith("off_floor") for f in v.failures)


def test_facing_away_flagged():
    target, counter = _sink_and_counter()
    # standing in front but facing +x instead of +y toward the sink
    v = validate_stand_pose((2.28, 0.55, 0.84), 0.0, target, [target, counter], floor_z=0.05)
    assert v.facing_error_deg == pytest.approx(90.0, abs=1.0)
    assert any(f.startswith("facing_off") for f in v.failures)


def test_standoff_too_far_flagged():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((2.28, -1.0, 0.84), math.pi / 2, target, [target, counter], floor_z=0.05)
    assert v.standoff_m > 1.3
    assert any(f.startswith("standoff_out") for f in v.failures)


def test_full_pass_in_front_facing_sink():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [target, counter], floor_z=0.05)
    assert v.ok is True
    assert v.failures == []
    assert v.on_floor is True and 0.30 <= v.standoff_m <= 1.30 and v.facing_error_deg < 5


# ------ the render-#9 diagnostic: intent valid, actual clipping -> isolates the coord-frame bug ------

def test_render9_intent_passes_but_counter_clipping_actual_fails():
    target, counter = _sink_and_counter()
    obstacles = [target, counter]
    yaw = math.atan2(1.33 - 0.03, 2.28 - 2.01)   # the stance's facing toward the sink
    # INTENDED pose from the stance plan: (2.01, 0.03) in the aisle -> should PASS
    intent = validate_stand_pose((2.01, 0.03, 0.84), yaw, target, obstacles, floor_z=0.05)
    assert intent.ok is True, intent.notes
    # ACTUAL placed pose if the robot really landed at the counter (the bug) -> must FAIL on clip
    actual = validate_stand_pose((2.28, 1.0, 0.84), yaw, target, obstacles, floor_z=0.05)
    assert actual.ok is False
    assert any(f.startswith("clips:") for f in actual.failures)
    # -> intent valid + actual invalid pinpoints a placement-application (coordinate) bug, not a bad pose


def test_validate_placement_accepts_standpose():
    target, counter = _sink_and_counter()
    sp = StandPose(position=(2.28, 0.55, 0.84), yaw=math.pi / 2, target_id="sink",
                   clear=True, standoff_m=0.55)
    v = validate_placement(sp, target, [target, counter], floor_z=0.05)
    assert isinstance(v, PlacementVerdict) and v.ok is True


# --------- Finding 1: overhead-fixture clip in the (pelvis, head] band must be caught ---------

def test_overhead_cabinet_in_torso_band_is_a_clip():
    # Robot at floor_z=0.0 -> pelvis_z=0.79, collision box z in [0.17, 1.41].
    # An upper cabinet z[0.9,1.6] dips into the head/torso band AND overlaps the footprint xy.
    target = _obj("sink", 5.0, 5.0, 0.85)  # far away (no standoff/facing interference)
    upper = SceneObject(
        id="upper_cabinet", label="upper_cabinet",
        bbox_min=(1.0, 1.0, 0.9), bbox_max=(2.0, 2.0, 1.6), centroid=(1.5, 1.5, 1.25),
    )
    v = validate_stand_pose((1.5, 1.5, 0.79), 0.0, target, [upper], floor_z=0.0)
    assert any(oid == "upper_cabinet" for oid, _ in v.clipping)
    assert v.ok is False


def test_range_hood_just_above_pelvis_is_a_clip():
    # Bottom exactly at z=1.0 (above the 0.79 pelvis cutoff, below the 1.41 head top) -> must clip.
    target = _obj("stove", 5.0, 5.0, 0.85)
    hood = SceneObject(
        id="range_hood", label="range_hood",
        bbox_min=(1.0, 1.0, 1.0), bbox_max=(2.0, 2.0, 1.5), centroid=(1.5, 1.5, 1.25),
    )
    v = validate_stand_pose((1.5, 1.5, 0.79), 0.0, target, [hood], floor_z=0.0)
    assert any(oid == "range_hood" for oid, _ in v.clipping)


def test_box_above_head_clears_and_is_not_a_clip():
    # min_z 1.5 > robot_top_z 1.41 -> genuinely high, stood under, no clip (regression guard for the
    # band edge: we must NOT start over-flagging boxes the robot actually clears).
    target = _obj("sink", 5.0, 5.0, 0.85)
    high = SceneObject(
        id="wall_box", label="wall_box",
        bbox_min=(1.0, 1.0, 1.5), bbox_max=(2.0, 2.0, 2.0), centroid=(1.5, 1.5, 1.75),
    )
    v = validate_stand_pose((1.5, 1.5, 0.79), 0.0, target, [high], floor_z=0.0)
    assert v.clipping == []


# --------- Finding 2: floor_z is required; a wrong frame would mis-classify the clip ---------

def test_floor_z_is_required():
    target, counter = _sink_and_counter()
    with pytest.raises(TypeError):
        validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [target, counter])


def test_correct_floor_z_catches_overhead_clip_that_default_frame_would_miss():
    # Raised-floor scene (floor_z=0.5 -> pelvis_z=1.29, head top=1.91). An upper cabinet at z[0.85,1.7]
    # sits in the torso band ONLY when the right floor_z is supplied; with the wrong frame the robot
    # would be modeled 0.5 m too low and the clip mis-classified.
    target = _obj("sink", 5.0, 5.0, 1.35)
    upper = SceneObject(
        id="upper_cabinet", label="upper_cabinet",
        bbox_min=(1.0, 1.0, 0.85), bbox_max=(2.0, 2.0, 1.7), centroid=(1.5, 1.5, 1.275),
    )
    v = validate_stand_pose((1.5, 1.5, 1.29), 0.0, target, [upper], floor_z=0.5)
    assert any(oid == "upper_cabinet" for oid, _ in v.clipping)
    assert v.on_floor is True  # pelvis 1.29 == floor_z 0.5 + pelvis_height 0.79


# --------- Finding 3: non-finite inputs must fail loud, never silently pass ---------

def test_nan_yaw_fails_not_passes():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((2.28, 0.55, 0.84), float("nan"), target, [target, counter], floor_z=0.05)
    assert v.ok is False
    assert any(f.startswith("non_finite_pose") for f in v.failures)


def test_nan_position_fails_with_explicit_reason():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((float("nan"), 0.55, 0.84), math.pi / 2, target, [target, counter], floor_z=0.05)
    assert v.ok is False
    assert any(f.startswith("non_finite_pose") for f in v.failures)


def test_nan_obstacle_bbox_is_flagged_not_skipped():
    target, _ = _sink_and_counter()
    bad = SceneObject(
        id="corrupt", label="corrupt",
        bbox_min=(float("nan"), 1.0, 0.0), bbox_max=(2.0, 2.0, 1.0), centroid=(1.5, 1.5, 0.5),
    )
    v = validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [bad], floor_z=0.05)
    assert v.ok is False
    assert any(f.startswith("non_finite_box") and "corrupt" in f for f in v.failures)


def test_inf_yaw_fails():
    target, counter = _sink_and_counter()
    v = validate_stand_pose((2.28, 0.55, 0.84), float("inf"), target, [target, counter], floor_z=0.05)
    assert v.ok is False
    assert any(f.startswith("non_finite_pose") for f in v.failures)


# --------- Finding 4: rotation-frame blind spot is documented, gross errors are caught ---------

def test_subthreshold_rotation_frame_error_is_a_known_blind_spot():
    # Correct position, but yaw rotated 30deg in a flipped frame -> under the 35deg facing tol, so the
    # validator (by design) does NOT catch a pure-rotation frame bug. This locks the documented limit.
    target, counter = _sink_and_counter()
    yaw = math.pi / 2 + math.radians(30)
    v = validate_stand_pose((2.28, 0.55, 0.84), yaw, target, [target, counter], floor_z=0.05)
    assert v.facing_error_deg == pytest.approx(30.0, abs=1.0)
    assert v.ok is True  # blind spot: sub-threshold rotation passes


def test_gross_rotation_frame_error_is_caught():
    target, counter = _sink_and_counter()
    yaw = math.pi / 2 + math.radians(90)  # 90deg off -> well over tolerance
    v = validate_stand_pose((2.28, 0.55, 0.84), yaw, target, [target, counter], floor_z=0.05)
    assert v.facing_error_deg == pytest.approx(90.0, abs=1.0)
    assert any(f.startswith("facing_off") for f in v.failures)
    assert v.ok is False


# --------- Finding 5: a small target recessed behind a deep counter must be reachable ---------

def test_recessed_target_reachable_with_fixture_aware_standoff():
    # Faucet (small) recessed at the back of a deep counter; counter front edge at y=0.98.
    faucet = _obj("faucet", 2.49, 1.15, 1.02, sx=0.1, sy=0.1, sz=0.1)   # box y in [1.10, 1.20]
    counter = SceneObject(
        id="counter", label="counter",
        bbox_min=(1.5, 0.98, 0.0), bbox_max=(3.0, 1.20, 0.9), centroid=(2.25, 1.09, 0.45),
    )
    # Closest counter-clearing pose: footprint front edge just shy of the counter (py=0.70 -> y_max=0.98).
    pose = (2.49, 0.70, 0.84)
    yaw = math.pi / 2
    # Without fixture awareness, standoff is measured to the tiny faucet box -> too far (~0.40 from faucet
    # front 1.10 to footprint front 0.98... actually the gap to the faucet here is 1.10-0.98=0.12 BELOW
    # the min) so naive standoff over-flags. With the counter as a reach surface, the gap collapses to 0.
    naive = validate_stand_pose(pose, yaw, faucet, [counter], floor_z=0.0)
    assert any(f.startswith("standoff_out") for f in naive.failures)  # documents the over-flag
    fixture_aware = validate_stand_pose(
        pose, yaw, faucet, [counter], floor_z=0.0, standoff_obstacles=[counter]
    )
    # Standoff now measured to the nearest of (faucet, counter): the robot is at the counter edge.
    assert fixture_aware.standoff_m == pytest.approx(0.0, abs=1e-6)
    # Still not OK here because 0.0 < standoff_range lower bound, but the metric is now fixture-correct;
    # a pose backed off slightly should pass both clip and standoff.
    back = validate_stand_pose(
        (2.49, 0.30, 0.84), yaw, faucet, [counter], floor_z=0.0, standoff_obstacles=[counter]
    )
    assert back.clipping == []
    assert 0.30 <= back.standoff_m <= 1.30
    assert back.ok is True


# --------- Finding 6: adversarial / edge coverage ---------

def test_empty_obstacles_no_clip():
    target, _ = _sink_and_counter()
    v = validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [], floor_z=0.05)
    assert v.clipping == []
    assert v.ok is True  # nothing to clip; other checks pass for the canonical good pose


def test_zero_size_target_has_defined_verdict():
    # A degenerate zero-extent target box: standoff is just the distance to the point; verdict defined.
    target = _obj("sink", 2.28, 1.33, 0.85, sx=0.0, sy=0.0, sz=0.0)
    counter = _obj("counter", 2.28, 1.33, 0.45, sx=1.5, sy=0.6, sz=0.9)
    v = validate_stand_pose((2.28, 0.55, 0.84), math.pi / 2, target, [counter], floor_z=0.05)
    # gap from footprint front (y_max=0.83) to the point at y=1.33 is 0.50 -> within [0.30, 1.30].
    assert v.standoff_m == pytest.approx(0.50, abs=1e-6)
    assert v.ok is True


def test_footprint_wider_than_gap_clips():
    # Target-to-obstacle gap is smaller than the footprint half-extent -> the footprint must clip.
    target = _obj("sink", 0.0, 2.0, 0.85)
    # Counter only 0.2m in front of the pose center; footprint (hy=0.28) reaches into it.
    counter = SceneObject(
        id="counter", label="counter",
        bbox_min=(-1.0, 0.2, 0.0), bbox_max=(1.0, 1.0, 0.9), centroid=(0.0, 0.6, 0.45),
    )
    v = validate_stand_pose((0.0, 0.0, 0.84), math.pi / 2, target, [counter], floor_z=0.0)
    assert any(oid == "counter" for oid, _ in v.clipping)


def test_multiple_clipping_obstacles_all_reported_with_areas():
    target = _obj("sink", 5.0, 5.0, 0.85)
    a = SceneObject(id="cab_a", label="cab_a", bbox_min=(1.0, 1.0, 0.0),
                    bbox_max=(1.6, 2.0, 0.9), centroid=(1.3, 1.5, 0.45))
    b = SceneObject(id="cab_b", label="cab_b", bbox_min=(1.4, 1.0, 0.0),
                    bbox_max=(2.0, 2.0, 0.9), centroid=(1.7, 1.5, 0.45))
    v = validate_stand_pose((1.5, 1.5, 0.79), 0.0, target, [a, b], floor_z=0.0)
    ids = {oid for oid, _ in v.clipping}
    assert ids == {"cab_a", "cab_b"}
    for _, area in v.clipping:
        assert area > 0.0
    assert v.ok is False
