from __future__ import annotations

import pytest

from blueprint_pipeline import gear_sonic_joint_order_contract as contract


def _valid_controller_state() -> dict:
    return {
        "joint_order_schema_version": contract.JOINT_ORDER_SCHEMA_VERSION,
        "body_joint_names": list(contract.PROTOCOL_V4_BODY_JOINT_NAMES),
        "left_hand_joint_names": list(contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        "right_hand_joint_names": list(contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        "mapping_digest": contract.PROTOCOL_V4_MAPPING_DIGEST,
    }


def test_pinned_protocol_v4_joint_order_shape_and_uniqueness() -> None:
    assert len(contract.PROTOCOL_V4_BODY_JOINT_NAMES) == 29
    assert len(contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES) == 7
    assert len(contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES) == 7
    assert len(contract.PROTOCOL_V4_FULL_JOINT_ORDER) == 43
    assert len(set(contract.PROTOCOL_V4_FULL_JOINT_ORDER)) == 43
    assert contract.PROTOCOL_V4_FULL_JOINT_ORDER == (
        contract.PROTOCOL_V4_BODY_JOINT_NAMES
        + contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES
        + contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES
    )
    # Legs precede waist, waist precedes arms, in the official 29-DOF layout.
    body = contract.PROTOCOL_V4_BODY_JOINT_NAMES
    assert body[0] == "left_hip_pitch_joint"
    assert body[6] == "right_hip_pitch_joint"
    assert body[12:15] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    assert body[15] == "left_shoulder_pitch_joint"
    assert body[22] == "right_shoulder_pitch_joint"
    assert all(name.endswith("_joint") for name in contract.PROTOCOL_V4_FULL_JOINT_ORDER)


def test_mapping_digest_is_stable_and_order_sensitive() -> None:
    digest = contract.compute_mapping_digest(
        schema_version=contract.JOINT_ORDER_SCHEMA_VERSION,
        body_joint_names=contract.PROTOCOL_V4_BODY_JOINT_NAMES,
        left_hand_joint_names=contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES,
        right_hand_joint_names=contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES,
    )
    assert digest == contract.PROTOCOL_V4_MAPPING_DIGEST
    swapped = list(contract.PROTOCOL_V4_BODY_JOINT_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert (
        contract.compute_mapping_digest(
            schema_version=contract.JOINT_ORDER_SCHEMA_VERSION,
            body_joint_names=swapped,
            left_hand_joint_names=contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES,
            right_hand_joint_names=contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES,
        )
        != digest
    )


def test_validate_controller_joint_order_accepts_pinned_contract() -> None:
    validated = contract.validate_controller_joint_order(_valid_controller_state())
    assert validated["schema_version"] == contract.JOINT_ORDER_SCHEMA_VERSION
    assert validated["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert tuple(validated["body_joint_names"]) == contract.PROTOCOL_V4_BODY_JOINT_NAMES


def test_validate_controller_joint_order_rejects_missing_schema_version() -> None:
    state = _valid_controller_state()
    del state["joint_order_schema_version"]
    with pytest.raises(ValueError, match="joint_order_schema_version_missing"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_unsupported_schema_version() -> None:
    state = _valid_controller_state()
    state["joint_order_schema_version"] = "gear_sonic_joint_order.protocol_v3.v1"
    with pytest.raises(ValueError, match="joint_order_schema_version_unsupported"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_positional_only_results() -> None:
    state = _valid_controller_state()
    del state["body_joint_names"]
    with pytest.raises(ValueError, match="positional_only_rejected"):
        contract.validate_controller_joint_order(state)
    state = _valid_controller_state()
    del state["left_hand_joint_names"]
    with pytest.raises(ValueError, match="positional_only_rejected"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_missing_digest() -> None:
    state = _valid_controller_state()
    del state["mapping_digest"]
    with pytest.raises(ValueError, match="mapping_digest_missing"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_wrong_digest() -> None:
    state = _valid_controller_state()
    state["mapping_digest"] = "0" * 64
    with pytest.raises(ValueError, match="mapping_digest_mismatch"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_duplicate_joints() -> None:
    state = _valid_controller_state()
    state["body_joint_names"][1] = state["body_joint_names"][0]
    with pytest.raises(ValueError, match="controller_body_joint_names_duplicate"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_unknown_joints() -> None:
    state = _valid_controller_state()
    state["body_joint_names"][3] = "left_knee_motor_joint"
    with pytest.raises(ValueError, match="controller_body_joint_names_unknown"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_missing_joints() -> None:
    state = _valid_controller_state()
    state["body_joint_names"] = state["body_joint_names"][:-1]
    with pytest.raises(ValueError, match="controller_body_joint_names_missing"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_left_right_permutation() -> None:
    state = _valid_controller_state()
    names = state["body_joint_names"]
    left = names.index("left_shoulder_pitch_joint")
    right = names.index("right_shoulder_pitch_joint")
    names[left], names[right] = names[right], names[left]
    with pytest.raises(ValueError, match="controller_body_joint_names_permuted"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_adjacent_permutation() -> None:
    state = _valid_controller_state()
    names = state["body_joint_names"]
    index = names.index("left_wrist_roll_joint")
    names[index], names[index + 1] = names[index + 1], names[index]
    with pytest.raises(ValueError, match="controller_body_joint_names_permuted"):
        contract.validate_controller_joint_order(state)


def test_validate_controller_joint_order_rejects_hand_permutation() -> None:
    state = _valid_controller_state()
    names = state["left_hand_joint_names"]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="controller_left_hand_joint_names_permuted"):
        contract.validate_controller_joint_order(state)


def test_validate_full_joint_order_rejects_permuted_and_accepts_pinned() -> None:
    contract.validate_full_joint_order(
        list(contract.PROTOCOL_V4_FULL_JOINT_ORDER), source="executor"
    )
    permuted = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    permuted[0], permuted[6] = permuted[6], permuted[0]
    with pytest.raises(ValueError, match="executor_joint_names_permuted"):
        contract.validate_full_joint_order(permuted, source="executor")


def test_validate_model_joint_names_is_set_exact() -> None:
    shuffled = list(reversed(contract.PROTOCOL_V4_FULL_JOINT_ORDER))
    contract.validate_model_joint_names(shuffled)
    with pytest.raises(ValueError, match="mujoco_model_joint_names_missing"):
        contract.validate_model_joint_names(shuffled[:-1])
    with pytest.raises(ValueError, match="mujoco_model_joint_names_unknown"):
        contract.validate_model_joint_names(shuffled + ["mystery_joint"])
    with pytest.raises(ValueError, match="mujoco_model_joint_names_duplicate"):
        contract.validate_model_joint_names(shuffled + [shuffled[0]])


def test_build_isaac_dof_mapping_returns_permutation() -> None:
    live = list(reversed(contract.PROTOCOL_V4_FULL_JOINT_ORDER))
    mapping = contract.build_isaac_dof_mapping(live)
    assert len(mapping) == 43
    for row in mapping:
        assert live[row["articulation_dof_index"]] == row["joint_name"]
        assert (
            contract.PROTOCOL_V4_FULL_JOINT_ORDER[row["protocol_index"]]
            == row["joint_name"]
        )


def test_build_isaac_dof_mapping_rejects_bad_articulations() -> None:
    live = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    with pytest.raises(ValueError, match="isaac_articulation_joint_names_missing"):
        contract.build_isaac_dof_mapping(live[:-1])
    with pytest.raises(ValueError, match="isaac_articulation_joint_names_unknown"):
        contract.build_isaac_dof_mapping(live + ["gripper_joint"])
    with pytest.raises(ValueError, match="isaac_articulation_joint_names_duplicate"):
        contract.build_isaac_dof_mapping(live + [live[0]])
