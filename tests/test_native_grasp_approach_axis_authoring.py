"""Author the missing gripper approach axis, and prove the frame it produces.

PR #799 established the convention and then refused, correctly, to invent the
input it was missing.  ``target_driven_link_far_edge_pinch`` gave the panel
normal to BOTH ``approach_unit_registered_stage`` and
``pinch_axis_registered_stage``, so the sealed affordance carried one axis
twice: ``ee_x = ee_y x ee_z`` is the zero vector and no quaternion exists.

The missing axis is not a preference, it is a measurement the producer already
had and discarded.  ``radial`` -- the in-plane direction from the hinge toward
the panel centre -- is computed to place the contact at the free edge and then
thrown away.  The gripper travels along the negative of it: inward from beyond
the free edge, toward the hinge.

Two things about that choice are worth stating plainly, because both were
argued from geometry rather than picked:

  * The *sign* is forced.  The contact sits at ``far_radial``, the free edge.
    Hinge-ward is the only in-plane direction that leaves the palm and the rear
    finger outside the appliance; the opposite sign buries the wrist in the
    cabinet, and the two hinge-parallel directions put the palm inside the
    panel because the contact is at mid-height, not at an edge.
  * The *jaw sign* is the producer's own.  ``pinch_axis`` already points from
    the panel toward the robot base, and it is consumed unnegated.  The negated
    branch is not merely a different roll: it is unreachable, because it asks
    the wrist to sit behind the door (see the reachability witnesses below).

What that costs is real and is not hidden: the resulting frame is 180 degrees
from the arm's measured reset pose, which is further than the 120 degrees the
identity placeholder cost.  Distance from reset is not the criterion --
reachability is -- so the witnesses below carry that burden.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.franka_kinematics import (
    FRANKA_JOINT_LIMITS_RAD,
    forward_kinematics,
)
from blueprint_pipeline.native_franka_action_math import (
    GRASP_AXIS_DEGENERACY_TOLERANCE,
    is_unauthored_identity_quaternion_xyzw,
)
from blueprint_pipeline.paired_target_interaction_affordance_candidate import (
    GRIPPER_FRAME_INDEPENDENCE_TOLERANCE,
    PairedTargetInteractionAffordanceError,
    materialize_paired_target_interaction_affordance_candidate,
    refuse_degenerate_gripper_frame,
)
from blueprint_pipeline.paired_target_native_arena_request import (
    PairedTargetNativeArenaRequestError,
    _grasp_orientation_contact_xyzw,
)
from blueprint_pipeline.rigid_frame_transforms import (
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
)
from tests.test_paired_target_interaction_affordance_candidate import (
    _freeze,
    _registered,
    _usd,
)


# The r22/r23 packet, committed.  Everything the reachability witnesses are
# checked against is read from it rather than restated here.
SEALED_PACKET = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "paired_target_native_arena_requests_v2_8f181229"
    / "task_a_washer_door_open"
    / "native_task_arena_packet_request.v1.json"
)

# Sealed 840920 geometry, in the registered stage frame.
SEALED_HINGE_POINT = [3.2704862952316285, 9.456716013828277, 0.42999999415255735]
SEALED_CONTACT_POINT = [3.7634863044329236, 9.456664008775391, 0.40499998738356097]
SEALED_PINCH_AXIS = [-0.0, -1.0, -0.0]
# What the producer now measures for this door: the panel-plane radial, negated.
AUTHORED_APPROACH_AXIS = [-1.0, 0.0, 0.0]
# The controlled body's measured reset orientation (r22 readback).
MEASURED_RESET_BODY_QUAT = [0.5, 0.5, 0.5, 0.5]

# Robotiq 2F-85 base to finger-midpoint, along the tool axis.  Modelled, not
# measured from the asset, so the reachability sweep behind these witnesses was
# repeated at 0.10 and 0.22 m and reached every phase at both.
TOOL_OFFSET_M = 0.16

# One continuous IK branch: every waypoint solved from its predecessor, no
# re-seeding, so these are the same solution family throughout.  Consecutive
# sweep waypoints differ by at most 0.0755 rad on that branch -- the r22/r23
# failure mode was the differential IK alternating between families every step,
# and this shows the authored frame does not require it.
WITNESS_JOINT_POSITIONS_RAD = {
    "approach_standoff": [
        -2.085056, -0.385802, -0.085682, -2.91491, 1.051613, 1.266986, 0.539788
    ],
    "joint_path_00": [
        -2.044628, -0.600374, 0.3817, -3.062988, 1.573228, 1.30943, 0.628555
    ],
    "joint_path_14": [
        -1.636365, -0.17238, 0.649625, -2.680082, 1.764876, 1.578576, 0.607089
    ],
    "joint_path_28": [
        -1.245413, 0.289364, 0.718316, -2.218437, 1.602889, 1.841555, 0.679479
    ],
    "retreat": [
        -1.290992, -0.013369, 0.543619, -2.531538, 1.573508, 1.564234, 0.621499
    ],
}


def _packet() -> dict:
    return json.loads(SEALED_PACKET.read_text(encoding="utf-8"))


def _rotation(quaternion_xyzw) -> np.ndarray:
    """Columns are the frame's axes, using the repository's own rotation."""

    return np.column_stack(
        [
            rotate_vector_xyzw(quaternion_xyzw, axis)
            for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ]
    )


def _quaternion_angle_rad(left, right) -> float:
    return 2.0 * math.acos(
        min(1.0, abs(sum(a * b for a, b in zip(left, right, strict=True))))
    )


def _articulated_candidate(tmp_path: Path) -> dict:
    """A washer-door-shaped asset: vertical hinge, thin panel, robot in front."""

    freeze = _freeze(tmp_path / "freeze.json", task_kind="articulated_interaction")
    usd = _usd(tmp_path / "asset.usda", articulated=True)
    registered = _registered(tmp_path / "registered.json", freeze, usd, task_id="task_a")
    return materialize_paired_target_interaction_affordance_candidate(
        task_freeze_path=freeze,
        registered_asset_receipt_path=registered,
        robot_base_position_world_m=[0.0, -1.0, 0.0],
        output_path=tmp_path / "result.json",
    )


# ---------------------------------------------------------------------------
# The producer: measuring the axis instead of reusing the panel normal.
# ---------------------------------------------------------------------------


def test_producer_measures_an_approach_independent_of_the_jaw_axis(
    tmp_path: Path,
) -> None:
    result = _articulated_candidate(tmp_path)
    candidate = result["candidate"]
    approach = candidate["gripper_approach_axis_registered_stage"]
    pinch = candidate["pinch_axis_registered_stage"]

    # The defect was that these two were the same vector.
    assert approach != pytest.approx(pinch, abs=1e-6)
    assert np.dot(approach, pinch) == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(approach) == pytest.approx(1.0, abs=1e-9)
    assert (
        result["selection_contract"]["gripper_approach_axis_source"]
        == "panel_plane_radial_inward_from_free_edge"
    )


def test_producer_approach_points_from_the_free_edge_toward_the_hinge(
    tmp_path: Path,
) -> None:
    """The sign, checked against the geometry rather than asserted."""

    candidate = _articulated_candidate(tmp_path)["candidate"]
    hinge = np.asarray(candidate["hinge_point_registered_stage_m"])
    contact = np.asarray(candidate["contact_point_registered_stage_m"])
    approach = np.asarray(candidate["gripper_approach_axis_registered_stage"])

    # Travelling along the approach from the contact reduces the radius, i.e.
    # the gripper closes on the hinge rather than reaching away past the edge.
    assert float(np.dot(approach, contact - hinge)) < 0.0


def test_standoff_direction_is_left_alone(tmp_path: Path) -> None:
    """``approach_unit`` is a translation and must keep meaning that.

    The plan places the pre-contact standoff at ``contact + approach_unit *
    clearance``.  Repurposing that field as the gripper axis would have moved
    the standoff off the panel face, so the new axis is a new field.
    """

    candidate = _articulated_candidate(tmp_path)["candidate"]

    assert candidate["approach_unit_registered_stage"] == pytest.approx(
        candidate["pinch_axis_registered_stage"], abs=1e-9
    )


def test_rigid_branch_keeps_its_own_approach(tmp_path: Path) -> None:
    """A free body has no privileged in-plane direction; nothing changes there."""

    freeze = _freeze(tmp_path / "freeze.json", task_kind="rigid_object_manipulation")
    usd = _usd(tmp_path / "asset.usda", articulated=False)
    registered = _registered(tmp_path / "registered.json", freeze, usd, task_id="task_b")
    result = materialize_paired_target_interaction_affordance_candidate(
        task_freeze_path=freeze,
        registered_asset_receipt_path=registered,
        robot_base_position_world_m=[0.0, -1.0, 0.0],
        output_path=tmp_path / "result.json",
    )

    candidate = result["candidate"]
    assert candidate["gripper_approach_axis_registered_stage"] == pytest.approx(
        candidate["approach_unit_registered_stage"], abs=1e-12
    )
    assert (
        result["selection_contract"]["gripper_approach_axis_source"]
        == "base_to_contact_direction"
    )


def test_a_collapsed_axis_pair_is_refused_where_it_is_sealed() -> None:
    with pytest.raises(
        PairedTargetInteractionAffordanceError,
        match="paired_target_affordance_gripper_frame_axes_degenerate",
    ):
        refuse_degenerate_gripper_frame(SEALED_PINCH_AXIS, SEALED_PINCH_AXIS)


def test_the_producer_refusal_matches_the_author_refusal() -> None:
    """Two tolerances for one collapse would leave a pair only one side rejects."""

    assert GRIPPER_FRAME_INDEPENDENCE_TOLERANCE == GRASP_AXIS_DEGENERACY_TOLERANCE


def test_an_independent_pair_is_accepted() -> None:
    refuse_degenerate_gripper_frame(AUTHORED_APPROACH_AXIS, SEALED_PINCH_AXIS)


# ---------------------------------------------------------------------------
# The author: what the sealed washer door now produces.
# ---------------------------------------------------------------------------


def _sealed_candidate() -> dict:
    return {
        "gripper_approach_axis_registered_stage": AUTHORED_APPROACH_AXIS,
        "pinch_axis_registered_stage": SEALED_PINCH_AXIS,
    }


def _identity_contact_path() -> dict:
    return {
        "joint_contact_path": [
            {"contact_pose_asset_root": {"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
        ]
    }


def test_sealed_washer_door_now_authors_a_real_grasp_frame() -> None:
    authored = _grasp_orientation_contact_xyzw(
        _sealed_candidate(), _identity_contact_path()
    )

    half = math.sqrt(0.5)
    assert authored == pytest.approx([-half, 0.0, half, 0.0], abs=1e-12)
    assert is_unauthored_identity_quaternion_xyzw(authored) is False


def test_the_authored_frame_carries_the_axes_that_went_in() -> None:
    authored = _grasp_orientation_contact_xyzw(
        _sealed_candidate(), _identity_contact_path()
    )
    columns = _rotation(authored)

    assert columns[:, 2] == pytest.approx(AUTHORED_APPROACH_AXIS, abs=1e-12)
    assert columns[:, 1] == pytest.approx(SEALED_PINCH_AXIS, abs=1e-12)


def test_a_receipt_that_predates_the_authored_axis_is_refused_by_name() -> None:
    """Silently falling back to ``approach_unit`` would restore the collapse."""

    stale = {"pinch_axis_registered_stage": SEALED_PINCH_AXIS}

    with pytest.raises(PairedTargetNativeArenaRequestError) as excinfo:
        _grasp_orientation_contact_xyzw(stale, _identity_contact_path())

    assert "gripper_approach_axis_missing" in str(excinfo.value)


def test_the_authored_frame_is_180_degrees_from_the_measured_reset() -> None:
    """The price of the choice, recorded rather than discovered on a GPU."""

    authored = _grasp_orientation_contact_xyzw(
        _sealed_candidate(), _identity_contact_path()
    )

    assert math.degrees(
        _quaternion_angle_rad(authored, MEASURED_RESET_BODY_QUAT)
    ) == pytest.approx(180.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reachability: the part an authored quaternion cannot be sealed without.
# ---------------------------------------------------------------------------


def _commanded_body_quaternion(row: dict, authored: list[float]) -> list[float]:
    """What ``native_task_construction_plan`` will command for this waypoint.

    The asset root is world-aligned in this packet, so the plan's
    ``root (x) contact (x) authored`` composition reduces to the second factor
    onward; the assertion below only needs the rotation, not the root offset.
    """

    return quaternion_multiply_xyzw(
        row["contact_pose_asset_root"]["orientation_xyzw"], authored
    )


def _asset_root_position() -> np.ndarray:
    packet = _packet()
    rows = [
        row
        for row in packet["assets"]
        if row.get("semantic_role") == "replacement"
        and row.get("object_type") == "ARTICULATION"
        and str(row["filename"]).endswith("washer_candidate.usda")
    ]
    assert len(rows) == 1
    return np.asarray(rows[0]["pose_world"]["position_world_m"], dtype=np.float64)


def _witness_targets() -> dict[str, tuple[np.ndarray, list[float]]]:
    packet = _packet()
    path = packet["task_spec"]["interaction_affordance"]["joint_contact_path"]
    root = _asset_root_position()
    authored = _grasp_orientation_contact_xyzw(
        _sealed_candidate(), {"joint_contact_path": path}
    )
    standoff = np.asarray(
        packet["task_spec"]["interaction_affordance"]["approach_unit_asset_root"],
        dtype=np.float64,
    ) * float(
        packet["task_spec"]["interaction_affordance"]["precontact_clearance_m"]
    )

    def contact(index: int) -> np.ndarray:
        return root + np.asarray(
            path[index]["contact_pose_asset_root"]["position_m"], dtype=np.float64
        )

    return {
        "approach_standoff": (
            contact(0) + standoff,
            _commanded_body_quaternion(path[0], authored),
        ),
        "joint_path_00": (contact(0), _commanded_body_quaternion(path[0], authored)),
        "joint_path_14": (contact(14), _commanded_body_quaternion(path[14], authored)),
        "joint_path_28": (contact(28), _commanded_body_quaternion(path[28], authored)),
        "retreat": (
            contact(28) + standoff,
            _commanded_body_quaternion(path[28], authored),
        ),
    }


def _flange_world(joints) -> tuple[np.ndarray, np.ndarray]:
    packet = _packet()
    base = packet["robot_base_pose_world"]
    position_base, rotation_base = forward_kinematics(joints)
    base_rotation = _rotation(base["orientation_xyzw"])
    position = base_rotation @ np.asarray(position_base) + np.asarray(
        base["position_world_m"]
    )
    return position, base_rotation @ np.asarray(rotation_base)


def test_the_sealed_reset_points_the_tool_axis_straight_down() -> None:
    """The anchor the orientation claim is measured against.

    Reachability here is a claim about a *rotation from the reset*, so the reset
    itself has to come from the packet rather than from an assumption.  The
    sealed DROID reset joints put the flange's +Z -- the tool axis -- along
    world -Z, which is also what ``adp009d_approach_capture`` pins as the
    tool-down orientation.
    """

    packet = _packet()
    reset = [
        packet["robot_joint_reset_positions_rad"][f"panda_joint{index}"]
        for index in range(1, 8)
    ]
    _, rotation = _flange_world(reset)

    assert rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


@pytest.mark.parametrize("phase", sorted(WITNESS_JOINT_POSITIONS_RAD))
def test_every_witnessed_phase_orientation_is_achievable(phase: str) -> None:
    """A constructive existence proof, mount-free.

    The commanded quantity is the *controlled body's* orientation, and that body
    is the Robotiq base, whose frame is a fixed rotation off the flange.  That
    fixed rotation cancels out of a difference of two orientations, so the claim
    is stated as one: the rotation taking the reset flange orientation to the
    witnessed one is exactly the rotation taking the measured reset body
    orientation to the commanded one.  Nothing here depends on knowing the
    coupler.
    """

    packet = _packet()
    reset = [
        packet["robot_joint_reset_positions_rad"][f"panda_joint{index}"]
        for index in range(1, 8)
    ]
    _, reset_flange = _flange_world(reset)
    _, witness_flange = _flange_world(WITNESS_JOINT_POSITIONS_RAD[phase])
    _, commanded = _witness_targets()[phase]

    flange_delta = witness_flange @ reset_flange.T
    body_delta = _rotation(commanded) @ _rotation(MEASURED_RESET_BODY_QUAT).T

    assert flange_delta == pytest.approx(body_delta, abs=1e-5)


@pytest.mark.parametrize("phase", sorted(WITNESS_JOINT_POSITIONS_RAD))
def test_every_witnessed_phase_position_is_achievable(phase: str) -> None:
    """The grasp frame is the finger midpoint, which rides the tool axis."""

    position, rotation = _flange_world(WITNESS_JOINT_POSITIONS_RAD[phase])
    reached = position + TOOL_OFFSET_M * rotation[:, 2]
    target, _ = _witness_targets()[phase]

    assert reached == pytest.approx(target, abs=1e-5)


@pytest.mark.parametrize("phase", sorted(WITNESS_JOINT_POSITIONS_RAD))
def test_every_witness_stays_inside_the_published_joint_limits(phase: str) -> None:
    """Reachable in principle is not reachable: r22 pinned j1, j6 and j7.

    The tightest phase is the grasp itself, where joint 4 sits 0.0088 rad off
    its lower limit on this branch.  That is inside, and it is thin -- recorded
    here so a placement change that eats it fails a test rather than a run.
    """

    joints = WITNESS_JOINT_POSITIONS_RAD[phase]
    margins = [
        min(value - lower, upper - value)
        for value, (lower, upper) in zip(joints, FRANKA_JOINT_LIMITS_RAD, strict=True)
    ]

    assert min(margins) > 0.0
    assert min(margins) == pytest.approx(
        {
            "approach_standoff": 0.1569,
            "joint_path_00": 0.0088,
            "joint_path_14": 0.3917,
            "joint_path_28": 0.8534,
            "retreat": 0.5403,
        }[phase],
        abs=5e-4,
    )
