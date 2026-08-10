from __future__ import annotations

import pytest

from blueprint_pipeline.native_task_robot_contact_topology import (
    NativeTaskRobotContactTopologyError,
    resolve_native_task_robot_contact_topology,
)


def test_franka_droid_profile_binds_every_contact_body_to_an_exact_path() -> None:
    topology = resolve_native_task_robot_contact_topology("franka_panda")

    assert topology["runtime_asset"]["sha256"] == (
        "sha256:c8d72259834e2e5290754f8580b37efbc0dec079ac6a98b27b167efe6461eb2c"
    )
    assert topology["task_contact_body_paths"] == [
        (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/"
            "left_inner_finger"
        ),
        (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/"
            "right_inner_finger"
        ),
    ]
    assert len(topology["protected_collision_body_paths"]) == 18
    assert all(
        "*" not in path for path in topology["protected_collision_body_paths"]
    )


def test_unknown_robot_has_no_guessed_contact_topology() -> None:
    with pytest.raises(NativeTaskRobotContactTopologyError) as excinfo:
        resolve_native_task_robot_contact_topology("unknown_robot")

    assert excinfo.value.errors == (
        "native_task_robot_contact_topology_unavailable:unknown_robot",
    )
