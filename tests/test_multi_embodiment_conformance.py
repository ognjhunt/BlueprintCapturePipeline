"""Zero-GPU conformance fixture for a second registered embodiment.

These tests exercise the supported Franka/G1 default hierarchy and a separate
conformance profile so placement, export, and action paths cannot quietly
hardcode one embodiment. The fixed-base fixture differs from the humanoid on
base mobility, action dimensionality, and camera layout and runs entirely on
CPU.
"""

from __future__ import annotations

import pytest

from blueprint_pipeline import action_space_registry as spaces
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


# -- action spaces -------------------------------------------------------


def test_registered_action_spaces_cover_both_embodiments() -> None:
    ids = spaces.registered_action_space_ids()
    assert spaces.SC3_7D_DELTA_EE in ids
    assert spaces.UNITREE_G1_WHOLE_BODY_78D in ids

    sc3 = spaces.get_action_space(spaces.SC3_7D_DELTA_EE)
    g1 = spaces.get_action_space(spaces.UNITREE_G1_WHOLE_BODY_78D)
    assert sc3.dim == 7
    # The executing G1 action really is 78-D; the platform can now say so.
    assert g1.dim == 78


def test_unregistered_action_space_fails_closed() -> None:
    with pytest.raises(spaces.UnknownActionSpaceError):
        spaces.get_action_space("some_vendor_action_space")


def test_default_action_space_is_still_sc3_so_existing_callers_are_unchanged() -> None:
    assert spaces.get_action_space().action_schema_id == spaces.SC3_7D_DELTA_EE
    assert spaces.get_action_space().dim_blocker == "action_space_dim_must_equal_7"


def test_vectors_are_validated_against_their_own_space_not_a_global_seven() -> None:
    seven = [0.0] * 7
    seventy_eight = [0.0] * 78

    assert spaces.validate_action_vector(seven, action_schema_id=spaces.SC3_7D_DELTA_EE) == []
    assert (
        spaces.validate_action_vector(
            seventy_eight, action_schema_id=spaces.UNITREE_G1_WHOLE_BODY_78D
        )
        == []
    )
    # And crucially, they are not interchangeable just because both are arrays.
    assert "action_space_dim_must_equal_7" in spaces.validate_action_vector(
        seventy_eight, action_schema_id=spaces.SC3_7D_DELTA_EE
    )
    assert "action_space_dim_must_equal_78" in spaces.validate_action_vector(
        seven, action_schema_id=spaces.UNITREE_G1_WHOLE_BODY_78D
    )


def test_non_numeric_and_non_finite_vectors_are_rejected() -> None:
    assert "action_vector_non_numeric" in spaces.validate_action_vector(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True]
    )
    assert "action_vector_non_finite" in spaces.validate_action_vector(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf")]
    )
    assert spaces.validate_action_vector("not a vector") == [
        "action_vector_missing_or_not_a_sequence"
    ]


def test_action_space_contract_validation_matches_the_registered_layout() -> None:
    sc3 = spaces.get_action_space(spaces.SC3_7D_DELTA_EE)
    good = {
        "dim": 7,
        "representation": "7d_delta_end_effector_pose",
        "order": list(sc3.order),
        "units": list(sc3.units),
    }
    assert spaces.validate_action_space_contract(good) == []

    wrong_dim = {**good, "dim": 78}
    assert "action_space_dim_must_equal_7" in spaces.validate_action_space_contract(wrong_dim)

    assert spaces.validate_action_space_contract({}) == ["action_space_contract_missing"]


def test_action_space_layouts_are_internally_consistent() -> None:
    """Every registered space's order/units must match its own dimension."""

    for name in spaces.registered_action_space_ids():
        space = spaces.get_action_space(name)
        assert len(space.order) == space.dim, name
        assert len(space.units) == space.dim, name
        assert space.dim_blocker.endswith(str(space.dim))


def test_g1_whole_body_space_matches_the_executing_controller_layout() -> None:
    """64 motion tokens plus two 7-DOF hands is what the controller actually sends."""

    space = spaces.get_action_space(spaces.UNITREE_G1_WHOLE_BODY_78D)
    motion_tokens = [name for name in space.order if name.startswith("motion_token_")]
    left = [name for name in space.order if name.startswith("left_hand_")]
    right = [name for name in space.order if name.startswith("right_hand_")]

    assert len(motion_tokens) == 64
    assert len(left) == 7
    assert len(right) == 7
    assert len(motion_tokens) + len(left) + len(right) == space.dim


def test_profile_action_interface_names_a_registered_space() -> None:
    """A profile may not reference an action space nobody registered."""

    for robot_id in known_robot_ids():
        profile = get_robot_profile(robot_id)
        schema_id = profile.action_interface.get("action_schema_id")
        if not schema_id:
            continue
        space = spaces.get_action_space(str(schema_id))
        assert space.dim == profile.action_interface.get("dim"), robot_id


def test_g1_profile_declares_the_whole_body_action_space() -> None:
    """The G1's declared export width must match the registered 78-D space.

    This is the concrete reason the platform-wide 7 was wrong: the embodiment
    the pipeline actually drives could not be described by its own contract.
    """

    export = UNITREE_G1_PROFILE.action_interface["lerobot_export"]
    registered = spaces.get_action_space(spaces.UNITREE_G1_WHOLE_BODY_78D)

    assert export["action_dim"] == registered.dim == 78
    # Its segment layout must tile the full width without gaps or overlap.
    segments = sorted(export["segments"], key=lambda row: row["start"])
    assert segments[0]["start"] == 0
    assert segments[-1]["end"] == registered.dim
    for earlier, later in zip(segments, segments[1:]):
        assert earlier["end"] == later["start"]
