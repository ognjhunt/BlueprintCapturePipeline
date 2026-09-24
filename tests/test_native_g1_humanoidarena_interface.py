"""The G1 policy vector must keep the publisher's exact named-joint layout."""

import math

import pytest

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_BODY_JOINT_NAMES
from blueprint_pipeline.native_g1_humanoidarena_interface import (
    ACTION_WIDTH,
    CANONICAL_BODY_JOINT_NAMES_29,
    build_semantic_v3_state,
    parse_semantic_v3_action,
)


def _joints() -> dict[str, float]:
    return {name: float(index) / 100 for index, name in enumerate(PROTOCOL_V4_BODY_JOINT_NAMES)}


def test_heading_canonical_state_uses_named_upstream_order() -> None:
    heading = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    positions = _joints()
    velocities = {name: -value for name, value in positions.items()}
    state = build_semantic_v3_state(
        initial_root_orientation_xyzw=heading,
        current_root_orientation_xyzw=heading,
        body_joint_positions_rad=positions,
        body_joint_velocities_rad_s=velocities,
    )
    assert len(state) == 64
    assert state[:6] == pytest.approx([1, 0, 0, 1, 0, 0])
    assert state[6:35] == [positions[name] for name in CANONICAL_BODY_JOINT_NAMES_29]
    assert state[35:] == [velocities[name] for name in CANONICAL_BODY_JOINT_NAMES_29]
    assert state[7] != positions[PROTOCOL_V4_BODY_JOINT_NAMES[1]]


def test_state_refuses_missing_joint_and_bad_quaternion() -> None:
    positions = _joints()
    positions.pop("left_knee_joint")
    with pytest.raises(ValueError, match="joint_positions_invalid"):
        build_semantic_v3_state(
            initial_root_orientation_xyzw=[0, 0, 0, 1],
            current_root_orientation_xyzw=[0, 0, 0, 1],
            body_joint_positions_rad=positions,
            body_joint_velocities_rad_s=_joints(),
        )
    with pytest.raises(ValueError, match="root_quaternion_invalid"):
        build_semantic_v3_state(
            initial_root_orientation_xyzw=[0, 0, 0, 0],
            current_root_orientation_xyzw=[0, 0, 0, 1],
            body_joint_positions_rad=_joints(),
            body_joint_velocities_rad_s=_joints(),
        )


def test_semantic_action_stays_controller_reference() -> None:
    action = [0.0] * ACTION_WIDTH
    action[3:9] = [1, 0, 0, 1, 0, 0]
    action[9:38] = list(range(29))
    action[38:40] = [0, 1]
    parsed = parse_semantic_v3_action(action)
    assert parsed["body_joint_reference_rad"]["right_hip_pitch_joint"] == 1
    assert parsed["requires_sonic_controller"] is True
    assert "joint_targets" not in parsed
    for invalid in ([0.0] * 39, [0.0] * 78, [math.nan] * 40):
        with pytest.raises(ValueError, match="shape_or_value_invalid"):
            parse_semantic_v3_action(invalid)
    action[3:9] = [0] * 6
    with pytest.raises(ValueError, match="rotation_invalid"):
        parse_semantic_v3_action(action)
    action[3:9] = [1, 0, 0, 1, 0, 0]
    action[38] = 1.2
    with pytest.raises(ValueError, match="hand_range_invalid"):
        parse_semantic_v3_action(action)
