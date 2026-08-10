from __future__ import annotations

import math

import pytest

from blueprint_pipeline.articulated_scripted_ik_preflight import (
    SCRIPTED_IK_PREFLIGHT_SCHEMA_VERSION,
    ScriptedIkPreflightError,
    resolve_scripted_joint_trajectory,
)


def _linear_solver(gain: float = 1.0, unreachable_beyond_m: float | None = None):
    """A stand-in arm whose joints track the target position proportionally.

    Real IK is a numeric solver behind a GPU-adjacent model; what this module
    owns is everything around it, so the solver is injected and the tests can
    drive exactly the cases that matter.
    """

    def solve(*, position_world_m, quaternion_world_xyzw, seed_joint_positions):
        reach = math.sqrt(sum(value * value for value in position_world_m))
        if unreachable_beyond_m is not None and reach > unreachable_beyond_m:
            return {"solved": False, "position_error_m": reach, "joint_positions": None}
        joints = [gain * position_world_m[index % 3] for index in range(7)]
        return {"solved": True, "position_error_m": 0.0, "joint_positions": joints}

    return solve


def _poses():
    return [
        {"phase_id": "approach", "position_world_m": [0.10, 0.10, 0.10],
         "gripper_command": 1.0},
        {"phase_id": "grasp", "position_world_m": [0.10, 0.12, 0.10],
         "gripper_command": 0.0},
        {"phase_id": "sweep_01", "position_world_m": [0.10, 0.16, 0.10],
         "gripper_command": 0.0},
        {"phase_id": "release", "position_world_m": [0.10, 0.16, 0.10],
         "gripper_command": 1.0},
    ]


def _resolve(**overrides):
    arguments = {
        "poses": _poses(),
        "initial_joint_positions": [0.0] * 7,
        "solve_pose": _linear_solver(),
        "max_joint_delta_rad": 0.05,
    }
    arguments.update(overrides)
    return resolve_scripted_joint_trajectory(**arguments)


def test_every_action_is_the_eight_the_action_space_expects() -> None:
    result = _resolve()

    assert all(len(row["isaac_action"]) == 8 for row in result["actions"])
    assert result["schema_version"] == SCRIPTED_IK_PREFLIGHT_SCHEMA_VERSION


def test_no_step_asks_a_joint_to_jump_further_than_it_may() -> None:
    """An open-loop replay of a large jump is a teleport, not a motion.

    The arm's controller would chase it at maximum effort and either miss the
    handle or hurl the door, and either way the run would not be measuring the
    program under test.
    """

    result = _resolve()

    previous = [0.0] * 7
    for row in result["actions"]:
        joints = row["isaac_action"][:7]
        assert max(abs(a - b) for a, b in zip(joints, previous)) <= 0.05 + 1e-9
        previous = joints


def test_a_long_move_is_subdivided_rather_than_refused() -> None:
    """The waypoints describe where to go, not how finely to get there."""

    result = _resolve(max_joint_delta_rad=0.01)

    assert len(result["actions"]) > len(_poses())
    assert result["interpolated_step_count"] > 0
    assert [row["phase_id"] for row in result["actions"]][0] == "approach"


def test_the_gripper_closes_at_grasp_and_opens_at_release() -> None:
    result = _resolve()

    by_phase = {}
    for row in result["actions"]:
        by_phase.setdefault(row["phase_id"], []).append(row["isaac_action"][7])
    assert by_phase["approach"][-1] == 1.0
    assert by_phase["grasp"][-1] == 0.0
    assert by_phase["release"][-1] == 1.0


def test_an_unreachable_pose_fails_closed_and_names_the_phase() -> None:
    """A silently dropped waypoint would look like a control that simply missed."""

    with pytest.raises(ScriptedIkPreflightError) as excinfo:
        _resolve(solve_pose=_linear_solver(unreachable_beyond_m=0.2))

    assert any("unreachable" in e and "sweep_01" in e for e in excinfo.value.errors)


def test_each_solve_is_seeded_from_the_previous_one() -> None:
    """Unseeded IK is free to flip the elbow between neighbouring waypoints."""

    seeds = []

    def recording(*, position_world_m, quaternion_world_xyzw, seed_joint_positions):
        seeds.append(list(seed_joint_positions))
        return {
            "solved": True,
            "position_error_m": 0.0,
            "joint_positions": [0.01 * len(seeds)] * 7,
        }

    _resolve(solve_pose=recording)

    assert seeds[0] == [0.0] * 7
    assert seeds[1] == [0.01] * 7


def test_resolution_is_deterministic() -> None:
    assert _resolve() == _resolve()


def test_an_empty_pose_list_fails_closed() -> None:
    with pytest.raises(ScriptedIkPreflightError) as excinfo:
        _resolve(poses=[])

    assert any("poses_missing" in e for e in excinfo.value.errors)
