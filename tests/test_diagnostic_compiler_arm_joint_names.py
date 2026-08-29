"""The compiler must read arm joint names from the profile, not guess them."""

from __future__ import annotations

import pytest

from blueprint_pipeline import (
    task_evaluation_diagnostic_native_arena_compiler as mod,
)
from blueprint_pipeline.scene_placement.robot_profile import (
    RobotProfile,
    get_robot_profile,
    register_robot_profile,
)


def test_franka_profile_declares_its_arm_joint_chain() -> None:
    profile = get_robot_profile("franka_panda")
    assert profile.arm_joint_names == (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )


def test_reset_derivation_refuses_an_embodiment_with_no_declared_chain() -> None:
    """A profile that declares no arm chain must fail closed, not fall back.

    The previous form regenerated `panda_joint1..7` from a format string, which
    silently produced Franka names for any embodiment and then raised a reset
    error that pointed at the wrong thing.
    """

    register_robot_profile(
        RobotProfile(robot_id="test_armless_embodiment", arm_joint_names=())
    )
    with pytest.raises(
        mod.TaskEvaluationDiagnosticNativeArenaCompilerError
    ) as excinfo:
        mod._derive_task_aware_franka_reset(
            profile={"robot_joint_reset_positions_rad": {"panda_joint1": 0.0}},
            base_pose={"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            task_trajectory={},
            robot_id="test_armless_embodiment",
        )
    assert "arm_joint_names_undeclared" in str(
        excinfo.value
    ) or "task_trajectory_invalid" in str(excinfo.value)


def test_reset_derivation_refuses_an_unregistered_robot_id() -> None:
    with pytest.raises(mod.TaskEvaluationDiagnosticNativeArenaCompilerError):
        mod._derive_task_aware_franka_reset(
            profile={"robot_joint_reset_positions_rad": {}},
            base_pose={"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            task_trajectory={},
            robot_id="no_such_robot_profile_at_all",
        )
