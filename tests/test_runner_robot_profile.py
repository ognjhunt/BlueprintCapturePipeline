"""Hermetic tests: the GPU runner's robot constants become profile-driven.

Loads a FRESH copy of the runner module per test (apply_robot_profile mutates
module globals) — no isaacsim, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement.robot_profile import robot_profile_from_dict

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("parity_runner_rp_test", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _custom_profile():
    return robot_profile_from_dict({
        "robot_id": "test_big_bot",
        "pelvis_height_m": 1.10,
        "footprint_half_extent_xyz": (0.30, 0.40, 0.70),
        "arm_span_m": 0.80,
        "max_effector_to_affordance_m": 0.35,
        "shoulder_lateral_offset_m": 0.25,
        "shoulder_above_root_m": 0.40,
        "standoff_range_m": (0.5, 1.6),
    })


def test_default_constants_are_g1():
    m = _load()
    assert m.ROBOT_FOOTPRINT_HALF_EXTENT == pytest.approx((0.12, 0.23, 0.62))
    assert m.ROBOT_PELVIS_HEIGHT_M == pytest.approx(0.79)
    assert m.G1_APPROX_ARM_SPAN_M == pytest.approx(0.45)


def test_apply_robot_profile_updates_module_constants():
    m = _load()
    m.apply_robot_profile(_custom_profile())
    assert m.ROBOT_FOOTPRINT_HALF_EXTENT == pytest.approx((0.30, 0.40, 0.70))
    assert m.ROBOT_PELVIS_HEIGHT_M == pytest.approx(1.10)
    assert m.G1_APPROX_ARM_SPAN_M == pytest.approx(0.80)
    assert m.G1_APPROX_SHOULDER_LATERAL_OFFSET_M == pytest.approx(0.25)
    assert m.G1_APPROX_SHOULDER_ABOVE_ROOT_M == pytest.approx(0.40)
    # derived reach envelope must be recomputed, not stale
    assert m.MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M == pytest.approx(0.80 + 0.35)
    assert m.TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M == pytest.approx((0.5, 1.6))
    # applied profile is recorded for result manifests
    assert m.ACTIVE_ROBOT_PROFILE.robot_id == "test_big_bot"


def test_apply_robot_profile_default_is_noop_for_g1():
    m = _load()
    before = (m.ROBOT_FOOTPRINT_HALF_EXTENT, m.ROBOT_PELVIS_HEIGHT_M,
              m.G1_APPROX_ARM_SPAN_M, m.MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M)
    m.apply_robot_profile(m.resolve_robot_profile_from_args(
        m.build_arg_parser().parse_args(["--out-dir", "/tmp/x"])))
    after = (m.ROBOT_FOOTPRINT_HALF_EXTENT, m.ROBOT_PELVIS_HEIGHT_M,
             m.G1_APPROX_ARM_SPAN_M, m.MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M)
    assert before == after
    assert m.ACTIVE_ROBOT_PROFILE.robot_id == "unitree_g1"


def test_cli_robot_id_resolves_registry_profile():
    m = _load()
    args = m.build_arg_parser().parse_args(["--out-dir", "/tmp/x", "--robot-id", "unitree_g1"])
    profile = m.resolve_robot_profile_from_args(args)
    assert profile.robot_id == "unitree_g1"


def test_cli_robot_profile_json_wins_over_robot_id(tmp_path):
    f = tmp_path / "bot.json"
    f.write_text(json.dumps({"robot_id": "json_bot", "pelvis_height_m": 1.3}))
    m = _load()
    args = m.build_arg_parser().parse_args(
        ["--out-dir", "/tmp/x", "--robot-id", "unitree_g1", "--robot-profile-json", str(f)])
    profile = m.resolve_robot_profile_from_args(args)
    assert profile.robot_id == "json_bot"
    assert profile.pelvis_height_m == pytest.approx(1.3)


def test_plan_task_stance_uses_applied_profile_footprint():
    """The stance planner must see the applied profile, not an import-time
    default-arg snapshot of the G1 footprint."""
    m = _load()
    m.apply_robot_profile(_custom_profile())
    plan = m.plan_task_stance(
        scenario={
            "scenario_id": "t",
            "task": "reach the target",
            "task_target_position_xyz": [1.0, 0.0, 1.0],
            "task_target_bounds": {"min": [0.9, -0.1, 0.8], "max": [1.1, 0.1, 1.2]},
        },
        probe_collision=lambda pose, yaw: 0,
        floor_z_hint=0.0,
    )
    assert plan["robot_footprint_half_extent"] == pytest.approx([0.30, 0.40, 0.70])


def test_apply_robot_profile_scales_close_reach_gap_with_arm_span():
    m = _load()
    g1_max = m.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M[1]
    long_arms = robot_profile_from_dict({"robot_id": "long_arms", "arm_span_m": 0.9})
    m.apply_robot_profile(long_arms)
    assert m.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M[1] > g1_max
    assert m.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M[0] == pytest.approx(0.08)


def test_empty_stance_ladder_falls_back_to_footprint_derived():
    """An empty stance_distance_candidates_m list (e.g. from a request builder)
    must mean 'derive from the robot footprint', never 'no candidates'."""
    m = _load()
    scenario = {"task": "open the microwave door",
                "stance_distance_candidates_m": []}
    distances = m.task_stance_distance_candidates(scenario)
    assert distances, "empty explicit ladder must fall back to derived candidates"


def test_cli_unknown_robot_id_raises_helpfully():
    m = _load()
    args = m.build_arg_parser().parse_args(["--out-dir", "/tmp/x", "--robot-id", "nope_bot"])
    with pytest.raises(KeyError) as ei:
        m.resolve_robot_profile_from_args(args)
    assert "unitree_g1" in str(ei.value)
