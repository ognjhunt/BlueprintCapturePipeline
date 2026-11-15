"""Zero-GPU conformance fixture for a second registered embodiment.

These tests exercise the supported Franka/G1 default hierarchy and a separate
conformance profile so placement, export, and action paths cannot quietly
hardcode one embodiment. The fixed-base fixture differs from the humanoid on
base mobility, action dimensionality, and camera layout and runs entirely on
CPU.
"""

from __future__ import annotations

import pytest

from blueprint_pipeline.scene_placement.robot_profile import (
    DEFAULT_HUMANOID_ROBOT_ID,
    DEFAULT_ROBOT_ID,
    FIXED_BASE_SINGLE_ARM_PROFILE,
    FRANKA_PANDA_PROFILE,
    UNITREE_G1_PROFILE,
    UnknownRobotProfileError,
    default_robot_id_for_embodiment,
    get_robot_profile,
    known_robot_ids,
    robot_embodiment_pack_contract,
    robot_profile_from_dict,
)


# -- registry ------------------------------------------------------------


def test_supported_defaults_and_conformance_embodiment_are_registered() -> None:
    ids = known_robot_ids()
    assert "franka_panda" in ids
    assert "unitree_g1" in ids
    assert "fixed_base_single_arm_reference" in ids
    assert DEFAULT_ROBOT_ID == "franka_panda"
    assert DEFAULT_HUMANOID_ROBOT_ID == "unitree_g1"
    assert get_robot_profile(DEFAULT_ROBOT_ID) is FRANKA_PANDA_PROFILE
    assert get_robot_profile(DEFAULT_HUMANOID_ROBOT_ID) is UNITREE_G1_PROFILE


@pytest.mark.parametrize(
    ("embodiment_type", "expected"),
    [
        (None, "franka_panda"),
        ("fixed_base_single_arm_manipulator", "franka_panda"),
        ("mobile_manipulator", "franka_panda"),
        ("humanoid", "unitree_g1"),
        ("bipedal-humanoid", "unitree_g1"),
        ("Humanoid Robot", "unitree_g1"),
    ],
)
def test_embodiment_default_hierarchy(embodiment_type: str | None, expected: str) -> None:
    assert default_robot_id_for_embodiment(embodiment_type) == expected


def test_unknown_robot_id_raises_a_typed_error_callers_can_catch() -> None:
    with pytest.raises(UnknownRobotProfileError):
        get_robot_profile("some_robot_from_a_job_request")
    # Still a KeyError, so existing handlers keep working.
    with pytest.raises(KeyError):
        get_robot_profile("some_robot_from_a_job_request")


def test_the_two_embodiments_differ_on_the_axes_that_matter() -> None:
    """Not a cosmetic second entry: the evaluation-relevant axes all differ."""

    g1 = get_robot_profile("unitree_g1")
    arm = get_robot_profile("fixed_base_single_arm_reference")

    assert g1.embodiment_type != arm.embodiment_type
    # Action interface: 78-D whole-body command versus 7-D delta end-effector.
    assert g1.action_interface["lerobot_export"]["action_dim"] == 78
    assert arm.action_interface["dim"] == 7
    # Observation interface.
    g1_mounts = {rig["mount"] for rig in g1.camera_rigs}
    arm_mounts = {rig["mount"] for rig in arm.camera_rigs}
    assert "head" in g1_mounts
    assert "head" not in arm_mounts
    assert "external_static" in arm_mounts
    # Kinematics/morphology.
    assert arm.kinematics["has_legs"] is False
    assert arm.kinematics["arm_count"] == 1
    assert arm.pelvis_height_m == 0.0
    assert g1.pelvis_height_m > 0.0


def test_second_profile_builds_its_embodiment_pack_contract() -> None:
    contract = robot_embodiment_pack_contract(FIXED_BASE_SINGLE_ARM_PROFILE)
    assert contract["robot_id"] == "fixed_base_single_arm_reference"
    assert contract["claim_boundaries"][
        "profile_is_a_conformance_fixture_not_a_supported_product_embodiment"
    ] is True


def test_second_profile_round_trips_through_the_json_shaped_loader() -> None:
    payload = {
        "robot_id": "third_party_arm",
        "embodiment_type": "fixed_base_single_arm_manipulator",
        "arm_span_m": 0.9,
        "standoff_range_m": [0.2, 0.9],
        "footprint_half_extent_xyz": [0.2, 0.2, 0.4],
    }
    profile = robot_profile_from_dict(payload)
    assert profile.robot_id == "third_party_arm"
    assert profile.standoff_range_m == (0.2, 0.9)


def test_profile_loader_still_rejects_typos() -> None:
    with pytest.raises(ValueError, match="unknown robot profile key"):
        robot_profile_from_dict({"robot_id": "x", "pelvis_heigth_m": 1.0})
