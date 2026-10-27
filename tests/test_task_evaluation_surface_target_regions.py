"""A region on another surface than the subject's own (floor to desk)."""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline.task_evaluation_surface_target import (
    derive_surface_target,
    score_surface_target,
    surface_execution_limits,
)

FLOOR = {"sage_prim_path": "/Root", "top_z_m": 0.0,
         "bounds_min_xyz_m": [-2.0, -2.0, -0.01], "bounds_max_xyz_m": [2.0, 2.0, 0.0]}
DESK = {"sage_prim_path": "/Root", "top_z_m": 0.74,
        "bounds_min_xyz_m": [-0.6, -0.4, 0.72], "bounds_max_xyz_m": [0.6, 0.4, 0.74]}
BALL = ([-0.0325, -0.0325, 0.0], [0.0325, 0.0325, 0.065])  # resting on the floor under the desk


def _target(*, start_support):
    return derive_surface_target(destination={"kind": "green_region", "relation": "on", "visible_label": "desk",
        "position_world_m": [0.0, 0.0, 0.74], "orientation_xyzw": [0, 0, 0, 1], "radius_m": 0.08},
        support=DESK, source_min=BALL[0], source_max=BALL[1], support_instance_id="desk",
        start_support=start_support)


def test_a_subject_under_the_region_on_another_surface_is_not_already_in_it():
    target = _target(start_support=FLOOR)
    assert target["support_source_instance_id"] == "desk"
    # Level surfaces keep the plan-only rule: the same footprint is an initial overlap.
    with pytest.raises(ValueError, match="surface_target_initial_overlap"):
        _target(start_support={**FLOOR, "bounds_max_xyz_m": [2.0, 2.0, 0.72]})
    with pytest.raises(ValueError, match="surface_target_initial_overlap"):
        _target(start_support=None)


def test_the_workspace_spans_both_surfaces():
    limits = surface_execution_limits(success={}, support=DESK, start_support=FLOOR)
    bounds = limits["robot_workspace_position_bounds_world_m"]
    assert bounds["minimum"] == [-3.0, -3.0, -1.01] and bounds["maximum"] == [3.0, 3.0, 0.74 + 1.5]
    assert surface_execution_limits(success={}, support=DESK)["robot_workspace_position_bounds_world_m"][
        "minimum"] == [-1.6, -1.4, -0.28]


@pytest.mark.parametrize("start_z, outside", [(0.0325, True), (0.7725, False)])
def test_scoring_reads_the_start_height_not_the_plan_position_alone(start_z, outside):
    target = _target(start_support=FLOOR)
    rows = [{"step_index": k, "task_object_pose_world": [0.0, 0.0, start_z, 0.0, 0.0, 0.0, 1.0],
             "destination_pose_world": [0.0, 0.0, 0.741, 0.0, 0.0, 0.0, 1.0],
             "support_contact_active": True, "task_contact_active": False} for k in range(40)]
    half = [[-0.0325] * 3, [0.0325] * 3]
    result = score_surface_target(target=target, bounds={"minimum": half[0], "maximum": half[1]},
                                  samples=rows, frequency_hz=20.0, minimum_lift_m=0.05)
    assert result["readback_complete"] and result["initially_outside"] is outside
    assert math.isfinite(target["radius_m"])
