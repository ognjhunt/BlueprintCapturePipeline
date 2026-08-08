"""Deterministic scoring for the frozen ADP-009D pick-lift-translate-place task.

Goal-prompt section 3 freezes the task the two candidates are adjudicated on:
start at the exact sealed can pose, grasp with the DROID Robotiq 2F-85, lift the
can at least ``0.08`` m above its settled start height, translate it to one
preregistered destination patch on the same support at least ``0.15`` m from the
start, and release it so it settles upright within ``0.05`` m of the destination
at no more than ``15`` degrees of tilt.  That section also fixes the grader:
"The primary success metric is deterministic simulator state, never a policy,
VLM, Cosmos, or human aesthetic judgment."

This module is that grader, and only that grader.  It consumes already-extracted
rigid-body poses and gripper state -- the numbers Isaac writes, not pictures of
them -- so it imports no Isaac, torch, or pxr, runs without a GPU, and can be
re-executed by a reviewer against a retained receipt.  Nothing here looks at a
rendered image, a learned score, or a caller-asserted success flag.

Three properties are load-bearing:

* **Fail closed.**  Malformed evidence raises :class:`TaskScoringError`.  Evidence
  that is well-formed but cannot decide a rung yields the explicit
  ``undetermined`` status; it never silently reads as failure or as success.
* **Monotone ladder.**  ``never_moved < moved < grasped < lifted < translated <
  placed``.  Resolution walks up from the bottom and stops at the first rung the
  evidence does not support, so a run cannot claim a rung it skipped.
* **Re-derivable receipt.**  Every measurement a predicate consumed and every
  threshold it was compared against is emitted and digest-bound, so the verdict
  can be checked without rerunning the simulator.

Quaternion convention
---------------------
The exact pinned IsaacLab revision represents ``root_pose_w`` as
``(x, y, z, w)``.  This scorer therefore defaults to explicit ``xyzw`` while
retaining a named ``wxyz`` compatibility path.  Getting this wrong is not
hypothetical: the same four numbers describe different rotations, and a
shifted payload can turn a real topple into a phantom upright can.

A settled can on a flat support is upright, so :func:`normalize_object_samples`
rejects a start pose whose tilt exceeds the canonical hold tolerance.  Be
precise about what that buys.  It catches a payload whose components have been
shifted a slot -- identity landing in ``x`` reads as 180 degrees -- and it
catches a genuinely non-canonical start.  It does **not** catch an ``(x, y, z,
w)`` payload of an upright can: identity is ``(0, 0, 0, 1)`` there, which read
as wxyz is a 180 degree yaw, and a yawed can is still upright.  No pose check
can separate those two, because they describe the same physical can.  The
convention is therefore a contract with the caller, not something this module
can fully verify.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def canonical_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    """Self-contained twin of ``decision_evidence_contracts.canonical_digest``.

    Scoring runs both inside the provider bundle and on a reviewer's laptop, so
    it must not depend on the wider package.  Parity with the repository
    contract is pinned by a test.
    """

    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


TASK_SCORING_SCHEMA_VERSION = "adp009d_task_scoring.v1"

# ---------------------------------------------------------------------------
# Measured scene facts.
#
# These are mirrored from the sealed scene, not re-derived here.  They match
# ``adp009d_isaac_runtime`` (ROBOT_BASE_POSITION_M, CAN_START_POSITION_M),
# ``adp009d_approach_capture`` (SUPPORT_HEIGHT_M,
# APPROVED_CAN_TOP_ABOVE_SUPPORT_M), and the SAGE collision ROI the task
# collision derivative was cut from.
# ---------------------------------------------------------------------------

SUPPORT_PLANE_Z_M = 0.5264650138348479
CAN_START_POSITION_M = (3.4681748, -3.3100837, 0.5264650138348479)
# Observed top of the approved can above the support plane.  The can's root
# origin sits *at* the support plane at rest -- CAN_START_POSITION_M[2] is
# SUPPORT_PLANE_Z_M exactly -- so the root z is the height of the can's base and
# this value is the can's full height, not a half-extent.
APPROVED_CAN_TOP_ABOVE_SUPPORT_M = 0.169
ROBOT_BASE_POSITION_M = (3.4681748, -2.8100837, 0.2766791)
ROBOT_BASE_YAW_RAD = -math.pi / 2
# Franka Panda maximum reach from the base.  Beyond it no arm motion can recover
# the can, so a pose outside this radius has left the task.
FRANKA_REACH_M = 0.855
# The SAGE collision ROI the task collision derivative was cut from.  Outside
# this box the sealed scene carries no collision geometry at all, so an object
# pose out there is not adjudicable: nothing would have stopped it falling.
SAGE_COLLISION_ROI_MIN_M = (2.4681748, -4.3100837, -0.1)
SAGE_COLLISION_ROI_MAX_M = (4.4681748, -1.9100837, 1.8)

# ---------------------------------------------------------------------------
# Canonical hold tolerances.
#
# Verbatim from ``adp009d_isaac_runtime`` (CAN_HOLD_XY_TOLERANCE_M,
# CAN_HOLD_Z_TOLERANCE_M, CAN_HOLD_TILT_TOLERANCE_DEG).  They are this program's
# already-measured floor for "this is real motion, not settle noise", so every
# derived threshold below that needs such a floor uses these rather than a fresh
# guess.
# ---------------------------------------------------------------------------

HOLD_XY_TOLERANCE_M = 5.0e-3
HOLD_Z_TOLERANCE_M = 5.0e-3
HOLD_TILT_TOLERANCE_DEG = 2.0

# ---------------------------------------------------------------------------
# Frozen task thresholds.
#
# Verbatim from goal-prompt section 3, and already pinned as
# ``required_thresholds`` in ``adp009d_franka_evaluation_harness``
# (minimum_lift_m, minimum_translation_m, maximum_center_error_m,
# maximum_tilt_degrees).  Changing one here without changing it there breaks the
# manifest contract, so they are duplicated deliberately and pinned by a test.
# ---------------------------------------------------------------------------

LIFT_CLEARANCE_M = 0.08
DESTINATION_MIN_DISTANCE_FROM_START_M = 0.15
PLACE_RADIUS_M = 0.05
PLACE_MAX_TILT_DEG = 15.0

# ---------------------------------------------------------------------------
# Derived thresholds.  Each states what it is derived from.
# ---------------------------------------------------------------------------

# Robotiq 2F-85 nameplate stroke: the "85" in the model name is its maximum
# opening in millimetres.  Goal-prompt section 3 names this exact gripper, so
# this is a property of the frozen embodiment, not a tuning knob.
GRIPPER_FULL_OPENING_M = 0.085
# How far the fingers must have travelled from fully open before the gripper
# counts as closed on something.  Set to the canonical hold tolerance: this
# program's floor for real motion versus noise.
GRASP_MIN_CLOSURE_M = HOLD_XY_TOLERANCE_M
GRIPPER_CLOSED_WIDTH_MAX_M = GRIPPER_FULL_OPENING_M - GRASP_MIN_CLOSURE_M
# The can's axis must lie between the fingers, so it must be within half the
# gripper's full stroke of the grasp frame.  Derived, not chosen.
GRASP_CAPTURE_RADIUS_XY_M = GRIPPER_FULL_OPENING_M / 2.0
# PhysX reports exactly 0.0 N when no contact pair exists, so any strictly
# positive normal force is real contact.  No magnitude floor is asserted: the
# sealed can's mass was never measured in this program, and a force floor
# without a mass would be an invented number.
GRASP_CONTACT_FORCE_MIN_N = 0.0
# The gripper must present at least this many contacting bodies before contact
# counts as a grasp.  A parallel jaw holds an object between two fingers; one
# finger touching is a nudge.
GRASP_MIN_CONTACT_BODIES = 2
# Below this the can is still resting on its support: the canonical hold z
# tolerance is the smallest z change this program treats as real.
SUPPORT_CLEARANCE_EPSILON_M = HOLD_Z_TOLERANCE_M
# A settled start can is upright.  Reusing the canonical hold tilt tolerance
# gives the tightest defensible upright test and doubles as the wxyz-convention
# guard described in the module docstring.
START_UPRIGHT_MAX_TILT_DEG = HOLD_TILT_TOLERANCE_DEG
# The runtime judges its canonical hold over a 40-step window (``range(40)``,
# retained as ``camera_warmup_40_frames``).  Reusing that length keeps "settled"
# meaning the same thing at the end of an episode as it does at the start.
SETTLE_WINDOW_SAMPLES = 40
# Quaternions arrive from a simulator that normalizes them.  A payload that
# misses unit norm by more than this is a wiring error, not float drift.
QUATERNION_NORM_TOLERANCE = 1.0e-3

# ---------------------------------------------------------------------------
# Vocabularies.
# ---------------------------------------------------------------------------

OUTCOME_NEVER_MOVED = "never_moved"
OUTCOME_MOVED = "moved"
OUTCOME_GRASPED = "grasped"
OUTCOME_LIFTED = "lifted"
OUTCOME_TRANSLATED = "translated"
OUTCOME_PLACED = "placed"
#: Monotone outcome ladder.  A run may only claim a rung whose every lower rung
#: the evidence also supports.
OUTCOME_LADDER = (
    OUTCOME_NEVER_MOVED,
    OUTCOME_MOVED,
    OUTCOME_GRASPED,
    OUTCOME_LIFTED,
    OUTCOME_TRANSLATED,
    OUTCOME_PLACED,
)

STATUS_SCORED = "scored"
STATUS_UNDETERMINED = "undetermined"

FAILURE_DROPPED = "dropped"
FAILURE_KNOCKED_OVER = "knocked_over"
FAILURE_PUSHED_OUT_OF_TASK_ENVELOPE = "pushed_out_of_task_envelope"
FAILURE_NEVER_MOVED = "never_moved"
FAILURE_MODES = (
    FAILURE_DROPPED,
    FAILURE_KNOCKED_OVER,
    FAILURE_PUSHED_OUT_OF_TASK_ENVELOPE,
    FAILURE_NEVER_MOVED,
)

#: Both fingers report contact with the object: the strongest grasp evidence.
GRASP_EVIDENCE_CONTACT = "both_finger_contact"
#: The gripper is closed around the can and the can is clear of its support.  A
#: closed gripper cannot hold a can off a support it is not gripping, and a push
#: cannot lift, so this is grasp evidence without a contact sensor.
GRASP_EVIDENCE_HELD_CLEAR = "closed_gripper_holding_can_clear_of_support"
#: Gripper evidence was present and says this sample is not a grasp.
GRASP_EVIDENCE_NOT_GRASPED = "not_grasped"
#: Neither contact forces nor gripper width were supplied for this sample.
GRASP_EVIDENCE_UNAVAILABLE = "unavailable"

#: Recorded on every receipt.  Scoring reads simulator object state only.
JUDGEMENT_SOURCE = "deterministic_simulator_object_state"


class TaskScoringError(ValueError):
    """Stable fail-closed task-scoring contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------


def _finite_vector(value: Any, length: int) -> np.ndarray | None:
    """Coerce ``value`` to a finite float vector of ``length``, or ``None``."""

    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        return None
    try:
        vector = np.asarray([float(item) for item in value], dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not bool(np.isfinite(vector).all()):
        return None
    return vector


QUATERNION_ORDER_WXYZ = "wxyz"
QUATERNION_ORDER_XYZW = "xyzw"
QUATERNION_ORDERS = (QUATERNION_ORDER_WXYZ, QUATERNION_ORDER_XYZW)


def tilt_degrees_from_quaternion(
    quaternion: Sequence[float], *, order: str
) -> float:
    """Angle between the body's +z axis and world up, in degrees.

    For a unit quaternion the third diagonal entry of the rotation matrix --
    the world-z component of the body's z axis -- is ``1 - 2 * (x**2 + y**2)``.
    Which slots hold x and y is the whole question, and it is not safe to
    assume: retained live data for this scene shows the approved can's
    ``root_pose_w`` quaternion as ``(-1.2e-05, -8.2e-05, -0.0, 1.0)`` for an
    upright can, i.e. w in the LAST slot -- xyzw, not the wxyz this function
    originally hardcoded.

    Guessing wrong is silent rather than loud, which is why the order is a
    required argument.  Near identity both readings agree to four decimals, so
    a settled can looks correct either way; but a can knocked 90 degrees about
    x is ``(0.707, 0, 0, 0.707)`` in xyzw, which read as wxyz gives x=0, y=0
    and therefore 0 degrees of tilt.  The knocked-over predicate would never
    fire on exactly the case it exists to catch.
    """

    if order not in QUATERNION_ORDERS:
        raise TaskScoringError([f"task_scoring_quaternion_order_unknown:{order}"])
    vector = _finite_vector(quaternion, 4)
    if vector is None:
        raise TaskScoringError(["task_scoring_quaternion_invalid"])
    values = [float(item) for item in vector]
    if order == QUATERNION_ORDER_WXYZ:
        x, y = values[1], values[2]
    else:
        x, y = values[0], values[1]
    alignment = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(alignment))


def tilt_degrees_from_quaternion_wxyz(quaternion_wxyz: Sequence[float]) -> float:
    """Backwards-compatible wxyz wrapper.  Prefer the explicit-order form."""

    return tilt_degrees_from_quaternion(
        quaternion_wxyz, order=QUATERNION_ORDER_WXYZ
    )


def _horizontal_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.hypot(left[0] - right[0], left[1] - right[1]))


def _inside_sage_collision_roi(position: np.ndarray) -> bool:
    low = np.asarray(SAGE_COLLISION_ROI_MIN_M, dtype=float)
    high = np.asarray(SAGE_COLLISION_ROI_MAX_M, dtype=float)
    return bool(np.all(position >= low) and np.all(position <= high))


def inside_task_envelope(position_world_m: Sequence[float]) -> bool:
    """The task envelope is the SAGE collision ROI intersected with arm reach.

    Both bounds are measured, and both are hard: outside the ROI the sealed
    scene has no collision geometry, and beyond ``FRANKA_REACH_M`` from the base
    the arm cannot touch the object again whatever the policy does.
    """

    position = _finite_vector(position_world_m, 3)
    if position is None:
        raise TaskScoringError(["task_scoring_position_invalid"])
    base = np.asarray(ROBOT_BASE_POSITION_M, dtype=float)
    within_reach = float(np.linalg.norm(position - base)) <= FRANKA_REACH_M
    return _inside_sage_collision_roi(position) and within_reach


# ---------------------------------------------------------------------------
# Input normalization.  Everything that can raise lives here.
# ---------------------------------------------------------------------------


def _normalize_gripper_width(raw: Any, errors: list[str], index: int) -> float | None:
    if raw is None:
        return None
    try:
        width = float(raw)
    except (TypeError, ValueError):
        errors.append(f"task_scoring_sample_{index}_gripper_width_invalid")
        return None
    if not math.isfinite(width) or width < 0.0:
        errors.append(f"task_scoring_sample_{index}_gripper_width_invalid")
        return None
    if width > GRIPPER_FULL_OPENING_M + GRASP_MIN_CLOSURE_M:
        # Wider than the 2F-85 can physically open: the channel is not this
        # gripper's width, or it is in the wrong unit.
        errors.append(f"task_scoring_sample_{index}_gripper_width_exceeds_stroke")
        return None
    return width


def _normalize_contact_forces(raw: Any, errors: list[str], index: int) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (Sequence, np.ndarray)):
        errors.append(f"task_scoring_sample_{index}_contact_forces_invalid")
        return None
    try:
        forces = [float(item) for item in raw]
    except (TypeError, ValueError):
        errors.append(f"task_scoring_sample_{index}_contact_forces_invalid")
        return None
    if not forces or any(not math.isfinite(item) or item < 0.0 for item in forces):
        errors.append(f"task_scoring_sample_{index}_contact_forces_invalid")
        return None
    return tuple(forces)


def normalize_object_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    require_sealed_start_pose: bool = True,
    quaternion_order: str = QUATERNION_ORDER_XYZW,
) -> tuple[dict[str, Any], ...]:
    """Validate an episode's extracted state and raise on anything malformed.

    Each sample carries ``step_index`` and ``can_pose_world`` -- seven numbers,
    ``(x, y, z, qx, qy, qz, qw)``, exactly Isaac Lab's ``root_pose_w`` row -- and
    optionally ``gripper_width_m``, ``grasp_frame_position_world_m``, and
    ``finger_contact_forces_n``.

    ``require_sealed_start_pose`` gates the canonical-condition check that the
    episode began at ``CAN_START_POSITION_M``.  It defaults to on because
    goal-prompt section 3 freezes that start; a scenario family that
    deliberately relocates the object must switch it off explicitly and say so
    in its own manifest.  The upright check is never gated: it is the quaternion
    convention guard, and no settled can on a flat support starts on its side.
    """

    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TaskScoringError(["task_scoring_samples_not_a_sequence"])
    if not samples:
        raise TaskScoringError(["task_scoring_samples_empty"])

    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    previous_step: int | None = None

    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append(f"task_scoring_sample_{index}_not_a_mapping")
            continue

        try:
            step_index = int(sample["step_index"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"task_scoring_sample_{index}_step_index_invalid")
            step_index = index
        else:
            if previous_step is not None and step_index <= previous_step:
                errors.append(f"task_scoring_sample_{index}_step_index_not_increasing")
            previous_step = step_index

        pose = _finite_vector(sample.get("can_pose_world"), 7)
        if pose is None:
            errors.append(f"task_scoring_sample_{index}_can_pose_world_invalid")
            continue
        quaternion = pose[3:7]
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            errors.append(f"task_scoring_sample_{index}_quaternion_not_unit_norm")
            continue

        normalized.append(
            {
                "step_index": step_index,
                "position_m": pose[0:3],
                "quaternion": quaternion,
                "quaternion_order": quaternion_order,
                "tilt_deg": tilt_degrees_from_quaternion(
                    quaternion, order=quaternion_order
                ),
                "gripper_width_m": _normalize_gripper_width(
                    sample.get("gripper_width_m"), errors, index
                ),
                "grasp_frame_position_m": _finite_vector(
                    sample.get("grasp_frame_position_world_m"), 3
                ),
                "finger_contact_forces_n": _normalize_contact_forces(
                    sample.get("finger_contact_forces_n"), errors, index
                ),
            }
        )

    if errors:
        raise TaskScoringError(errors)

    start = normalized[0]
    if start["tilt_deg"] > START_UPRIGHT_MAX_TILT_DEG:
        # A settled can on a flat support is upright.  The usual cause of a
        # tilted start is a shifted quaternion payload rather than a real
        # topple, so the error names that first -- see the module docstring for
        # what this check can and cannot detect.
        raise TaskScoringError(["task_scoring_start_pose_not_upright_check_quaternion_order"])
    if require_sealed_start_pose:
        sealed = np.asarray(CAN_START_POSITION_M, dtype=float)
        if _horizontal_distance(start["position_m"], sealed) > HOLD_XY_TOLERANCE_M:
            raise TaskScoringError(["task_scoring_start_pose_not_at_sealed_can_position"])
        if abs(float(start["position_m"][2]) - float(sealed[2])) > HOLD_Z_TOLERANCE_M:
            raise TaskScoringError(["task_scoring_start_pose_not_at_sealed_support_height"])

    return tuple(normalized)


def validate_destination(
    destination_position_world_m: Sequence[float],
    *,
    start_position_world_m: Sequence[float],
) -> dict[str, Any]:
    """Validate the preregistered destination patch against the frozen task.

    The destination is an *input*, never chosen here.  Goal-prompt section 3
    requires it to be derived from SAGE support triangles plus splat/semantic
    occupancy before any learned-policy outcome; this function only refuses a
    destination the frozen task could not have used.
    """

    destination = _finite_vector(destination_position_world_m, 3)
    if destination is None:
        raise TaskScoringError(["task_scoring_destination_invalid"])
    start = _finite_vector(start_position_world_m, 3)
    if start is None:
        raise TaskScoringError(["task_scoring_destination_start_invalid"])

    errors: list[str] = []
    # "on the same support": the harness manifest contract pins the destination
    # to the start height exactly, so any deviation beyond the canonical hold z
    # tolerance is a different surface.
    same_support_offset_m = abs(float(destination[2]) - float(start[2]))
    if same_support_offset_m > HOLD_Z_TOLERANCE_M:
        errors.append("task_scoring_destination_not_on_same_support")
    distance_from_start_m = _horizontal_distance(destination, start)
    if distance_from_start_m < DESTINATION_MIN_DISTANCE_FROM_START_M:
        errors.append("task_scoring_destination_below_minimum_translation")
    base = np.asarray(ROBOT_BASE_POSITION_M, dtype=float)
    distance_from_base_m = float(np.linalg.norm(destination - base))
    if distance_from_base_m > FRANKA_REACH_M:
        errors.append("task_scoring_destination_outside_reachable_workspace")
    if not _inside_sage_collision_roi(destination):
        errors.append("task_scoring_destination_outside_sage_collision_roi")
    if errors:
        raise TaskScoringError(errors)

    return {
        "position_world_m": [float(item) for item in destination],
        "distance_from_start_m": distance_from_start_m,
        "distance_from_robot_base_m": distance_from_base_m,
        "same_support_offset_m": same_support_offset_m,
    }


# ---------------------------------------------------------------------------
# Measurement.  The single place raw samples are read.
# ---------------------------------------------------------------------------


def grasp_evidence_for_sample(sample: Mapping[str, Any], *, support_plane_z_m: float) -> str:
    """Classify one sample's grasp evidence into the module's four-way vocabulary.

    Contact is preferred.  Failing that, a closed gripper wrapped around a can
    that is clear of its support is grasp evidence on its own: nothing else in
    this scene holds a can up, and a push cannot lift.  With neither channel the
    sample is ``unavailable`` -- not "not grasped".
    """

    forces = sample.get("finger_contact_forces_n")
    width = sample.get("gripper_width_m")
    if forces is None and width is None:
        return GRASP_EVIDENCE_UNAVAILABLE

    if forces is not None:
        contacting = sum(1 for force in forces if force > GRASP_CONTACT_FORCE_MIN_N)
        if contacting >= GRASP_MIN_CONTACT_BODIES:
            return GRASP_EVIDENCE_CONTACT

    if width is not None and width <= GRIPPER_CLOSED_WIDTH_MAX_M:
        position = sample["position_m"]
        clearance = float(position[2]) - float(support_plane_z_m)
        if clearance >= SUPPORT_CLEARANCE_EPSILON_M:
            grasp_frame = sample.get("grasp_frame_position_m")
            if grasp_frame is None:
                return GRASP_EVIDENCE_HELD_CLEAR
            # When the grasp frame is known, also require the can's axis between
            # the fingers and the fingers somewhere along the can's body.
            height_above_can_base = float(grasp_frame[2]) - float(position[2])
            if (
                _horizontal_distance(grasp_frame, position) <= GRASP_CAPTURE_RADIUS_XY_M
                and 0.0 <= height_above_can_base <= APPROVED_CAN_TOP_ABOVE_SUPPORT_M
            ):
                return GRASP_EVIDENCE_HELD_CLEAR

    return GRASP_EVIDENCE_NOT_GRASPED


def measure_episode(
    normalized_samples: Sequence[Mapping[str, Any]],
    *,
    destination_position_world_m: Sequence[float],
    support_plane_z_m: float = SUPPORT_PLANE_Z_M,
    settle_window_samples: int = SETTLE_WINDOW_SAMPLES,
) -> dict[str, Any]:
    """Reduce an episode to the scalars every predicate is computed from.

    Emitting this dict is what makes a verdict re-derivable: a reviewer can
    re-run the predicates against these numbers without the simulator.
    """

    if int(settle_window_samples) < 1:
        raise TaskScoringError(["task_scoring_settle_window_invalid"])
    settle_window_samples = int(settle_window_samples)
    destination = _finite_vector(destination_position_world_m, 3)
    if destination is None:
        raise TaskScoringError(["task_scoring_destination_invalid"])

    positions = np.asarray([sample["position_m"] for sample in normalized_samples], dtype=float)
    tilts = np.asarray([float(sample["tilt_deg"]) for sample in normalized_samples], dtype=float)
    start = positions[0]
    final = positions[-1]

    horizontal_from_start = np.hypot(positions[:, 0] - start[0], positions[:, 1] - start[1])
    z_from_start = positions[:, 2] - start[2]
    clearance_above_support = positions[:, 2] - float(support_plane_z_m)
    horizontal_to_destination = np.hypot(
        positions[:, 0] - destination[0], positions[:, 1] - destination[1]
    )
    base = np.asarray(ROBOT_BASE_POSITION_M, dtype=float)
    distance_from_base = np.linalg.norm(positions - base, axis=1)

    evidence = [
        grasp_evidence_for_sample(sample, support_plane_z_m=support_plane_z_m)
        for sample in normalized_samples
    ]
    grasped_flags = [
        item in (GRASP_EVIDENCE_CONTACT, GRASP_EVIDENCE_HELD_CLEAR) for item in evidence
    ]
    grasped_indices = [index for index, flag in enumerate(grasped_flags) if flag]
    unavailable_indices = [
        index for index, item in enumerate(evidence) if item == GRASP_EVIDENCE_UNAVAILABLE
    ]

    lifted_indices = [
        index for index, value in enumerate(z_from_start) if float(value) >= LIFT_CLEARANCE_M
    ]
    first_lifted_index = lifted_indices[0] if lifted_indices else None

    # A landing is a post-lift sample back in the support-contact band with no
    # grasp evidence.  Where it lands separates a drop from a release.
    landing_index: int | None = None
    landing_inside_destination: bool | None = None
    if first_lifted_index is not None:
        for index in range(first_lifted_index + 1, len(normalized_samples)):
            if float(clearance_above_support[index]) > SUPPORT_CLEARANCE_EPSILON_M:
                continue
            if evidence[index] == GRASP_EVIDENCE_UNAVAILABLE:
                landing_index = index
                landing_inside_destination = None
                break
            if grasped_flags[index]:
                continue
            landing_index = index
            landing_inside_destination = bool(
                float(horizontal_to_destination[index]) <= PLACE_RADIUS_M
            )
            break

    window = min(settle_window_samples, len(normalized_samples))
    settle_available = len(normalized_samples) >= settle_window_samples
    settle_positions = positions[-window:]
    settle_tilts = tilts[-window:]
    settle_evidence = evidence[-window:]
    settle_grasped_flags = grasped_flags[-window:]
    settle_anchor = settle_positions[0]
    settle_xy_span_m = float(
        np.max(np.hypot(settle_positions[:, 0] - settle_anchor[0],
                        settle_positions[:, 1] - settle_anchor[1]))
    )
    settle_z_span_m = float(np.max(np.abs(settle_positions[:, 2] - settle_anchor[2])))
    settle_tilt_span_deg = float(np.max(np.abs(settle_tilts - settle_tilts[0])))

    if any(item == GRASP_EVIDENCE_UNAVAILABLE for item in settle_evidence):
        settle_grasped: bool | None = None
    else:
        settle_grasped = bool(any(settle_grasped_flags))

    return {
        "sample_count": len(normalized_samples),
        "first_step_index": int(normalized_samples[0]["step_index"]),
        "final_step_index": int(normalized_samples[-1]["step_index"]),
        "start_position_m": [float(item) for item in start],
        "final_position_m": [float(item) for item in final],
        "destination_position_m": [float(item) for item in destination],
        "support_plane_z_m": float(support_plane_z_m),
        "start_tilt_deg": float(tilts[0]),
        "final_tilt_deg": float(tilts[-1]),
        "max_tilt_deg": float(np.max(tilts)),
        "max_horizontal_displacement_from_start_m": float(np.max(horizontal_from_start)),
        "final_horizontal_displacement_from_start_m": float(horizontal_from_start[-1]),
        "max_abs_z_displacement_from_start_m": float(np.max(np.abs(z_from_start))),
        "max_lift_above_start_m": float(np.max(z_from_start)),
        "final_lift_above_start_m": float(z_from_start[-1]),
        "min_clearance_above_support_m": float(np.min(clearance_above_support)),
        "final_clearance_above_support_m": float(clearance_above_support[-1]),
        "min_horizontal_distance_to_destination_m": float(np.min(horizontal_to_destination)),
        "final_horizontal_distance_to_destination_m": float(horizontal_to_destination[-1]),
        "final_distance_from_robot_base_m": float(distance_from_base[-1]),
        "max_distance_from_robot_base_m": float(np.max(distance_from_base)),
        "final_inside_task_envelope": inside_task_envelope(final),
        "any_sample_outside_task_envelope": bool(
            not all(inside_task_envelope(position) for position in positions)
        ),
        "grasp_evidence_by_sample": list(evidence),
        "grasped_sample_indices": grasped_indices,
        "grasp_evidence_unavailable_sample_indices": unavailable_indices,
        "first_lifted_sample_index": first_lifted_index,
        "post_lift_landing_sample_index": landing_index,
        "post_lift_landing_inside_destination": landing_inside_destination,
        "settle_window_samples_requested": settle_window_samples,
        "settle_window_samples_used": int(window),
        "settle_window_available": bool(settle_available),
        "settle_xy_span_m": settle_xy_span_m,
        "settle_z_span_m": settle_z_span_m,
        "settle_tilt_span_deg": settle_tilt_span_deg,
        "settle_max_tilt_deg": float(np.max(settle_tilts)),
        "settle_min_clearance_above_support_m": float(
            np.min(settle_positions[:, 2] - float(support_plane_z_m))
        ),
        "settle_grasped": settle_grasped,
    }


# ---------------------------------------------------------------------------
# Predicates.  Each takes a ``measure_episode`` mapping and returns
# ``True`` / ``False`` / ``None``, where ``None`` means "the evidence cannot
# decide this" -- never "no".
# ---------------------------------------------------------------------------


def never_moved(measurements: Mapping[str, Any]) -> bool:
    """The can never left the canonical hold tolerances: nothing happened."""

    return (
        float(measurements["max_horizontal_displacement_from_start_m"]) <= HOLD_XY_TOLERANCE_M
        and float(measurements["max_abs_z_displacement_from_start_m"]) <= HOLD_Z_TOLERANCE_M
        and float(measurements["max_tilt_deg"]) <= HOLD_TILT_TOLERANCE_DEG
    )


def moved(measurements: Mapping[str, Any]) -> bool:
    """The can left the canonical hold tolerances at least once."""

    return not never_moved(measurements)


def grasped(measurements: Mapping[str, Any]) -> bool | None:
    """The gripper held the can at some point in the episode.

    ``None`` when no sample carried either contact forces or a gripper width and
    none of the samples that did carry them showed a grasp: with no gripper
    channel at all there is no evidence either way.
    """

    if measurements["grasped_sample_indices"]:
        return True
    if measurements["grasp_evidence_unavailable_sample_indices"]:
        return None
    return False


def lifted(measurements: Mapping[str, Any]) -> bool:
    """The can rose at least ``LIFT_CLEARANCE_M`` above its settled start height.

    Because the sealed can's root origin rests exactly on the support plane,
    "above its settled start height" and "clear of the support" are the same
    measurement here; both are recorded so a reviewer need not take that on
    trust.
    """

    return float(measurements["max_lift_above_start_m"]) >= LIFT_CLEARANCE_M


def translated(measurements: Mapping[str, Any]) -> bool:
    """The can reached the preregistered destination patch.

    Horizontal distance only.  The destination is a patch *on the support*, and
    the can's root origin sits on the support plane at rest, so its z is gated
    separately by :func:`placed`; folding z in here would double-count it and
    make the ``0.05`` m tolerance unreachable for any can with height.
    """

    return float(measurements["min_horizontal_distance_to_destination_m"]) <= PLACE_RADIUS_M


def placed(measurements: Mapping[str, Any]) -> bool | None:
    """The can was released and settled upright on the support at the destination.

    The settle window must be entirely post-release: a single sample inside it
    showing the gripper still on the can fails the predicate.  That is
    deliberate.  "Release it so it settles" means the episode has to keep
    running after the fingers open, and a window that straddles the release
    would not show the can at rest anyway.

    ``None`` when the episode is shorter than the settle window, or when the
    settle window has no gripper evidence: neither "it came to rest" nor "the
    gripper let go" can be established, and guessing either way would be exactly
    the silent default this module exists to prevent.
    """

    if not measurements["settle_window_available"]:
        return None
    if measurements["settle_grasped"] is None:
        return None
    settled = (
        float(measurements["settle_xy_span_m"]) <= HOLD_XY_TOLERANCE_M
        and float(measurements["settle_z_span_m"]) <= HOLD_Z_TOLERANCE_M
        and float(measurements["settle_tilt_span_deg"]) <= HOLD_TILT_TOLERANCE_DEG
    )
    on_support = abs(float(measurements["final_clearance_above_support_m"])) <= HOLD_Z_TOLERANCE_M
    return bool(
        settled
        and on_support
        and measurements["settle_grasped"] is False
        and float(measurements["settle_max_tilt_deg"]) <= PLACE_MAX_TILT_DEG
        and float(measurements["final_horizontal_distance_to_destination_m"]) <= PLACE_RADIUS_M
    )


def dropped(measurements: Mapping[str, Any]) -> bool | None:
    """The can lost support: it fell off, or it fell out of the gripper.

    Falling below the support plane is unconditional.  Otherwise a drop is a
    post-lift return to the support-contact band with no grasp evidence and
    outside the destination patch -- the same motion as a release, separated
    from it by *where* it lands rather than by re-deriving :func:`placed`.
    """

    fell_off_support = (
        float(measurements["min_clearance_above_support_m"]) < -HOLD_Z_TOLERANCE_M
    )
    if fell_off_support:
        return True
    if measurements["post_lift_landing_sample_index"] is None:
        # No post-lift landing at all: either it never got lifted, or it is
        # still held or still airborne at the end.  Not a drop.
        return False
    inside = measurements["post_lift_landing_inside_destination"]
    if inside is None:
        return None
    return not bool(inside)


def knocked_over(measurements: Mapping[str, Any]) -> bool | None:
    """The can came to rest tilted beyond the task's own placement tolerance.

    Tilt is judged at rest, not in flight: a carried can may legitimately be
    reoriented mid-episode, and goal-prompt section 3 states the ``15`` degree
    limit as a property of the settled result.  ``None`` when the episode is too
    short to establish rest.
    """

    if not measurements["settle_window_available"]:
        return None
    at_rest = (
        float(measurements["settle_min_clearance_above_support_m"])
        <= SUPPORT_CLEARANCE_EPSILON_M
    )
    return bool(at_rest and float(measurements["settle_max_tilt_deg"]) > PLACE_MAX_TILT_DEG)


def pushed_out_of_task_envelope(measurements: Mapping[str, Any]) -> bool | None:
    """The can was displaced without ever being grasped and left the envelope.

    "Pushed" is the distinguishing part: this fires only when the gripper never
    held the can, which is what separates a swipe from a carry that ended badly.
    A run that grasped the can and then flung it out is recorded through
    ``final_inside_task_envelope`` and ``max_distance_from_robot_base_m`` in the
    receipt rather than through this predicate.
    """

    was_grasped = grasped(measurements)
    if was_grasped is None:
        return None
    return bool(
        was_grasped is False
        and moved(measurements)
        and not bool(measurements["final_inside_task_envelope"])
    )


# ---------------------------------------------------------------------------
# Ladder resolution.
# ---------------------------------------------------------------------------


def resolve_outcome_ladder(predicates: Mapping[str, bool | None]) -> dict[str, Any]:
    """Walk the monotone ladder and stop at the first rung evidence does not support.

    ``never_moved`` is the floor and is always reachable, so the walk starts at
    rung 0 and only ever climbs through rungs whose predicate is definitely
    true.  A ``None`` anywhere on the path stops the climb and marks the whole
    verdict ``undetermined``: the run keeps whatever rung it had already proven
    and cannot borrow a higher one from a later predicate.
    """

    outcome = OUTCOME_NEVER_MOVED
    rank = 0
    status = STATUS_SCORED
    reasons: list[str] = []
    truncated_at: str | None = None

    for index in range(1, len(OUTCOME_LADDER)):
        rung = OUTCOME_LADDER[index]
        value = predicates.get(rung)
        if value is None:
            status = STATUS_UNDETERMINED
            reasons.append(f"{rung}_undetermined")
            truncated_at = rung
            break
        if not value:
            truncated_at = rung
            break
        outcome = rung
        rank = index

    return {
        "outcome": outcome,
        "outcome_rank": rank,
        "ladder": list(OUTCOME_LADDER),
        "ladder_truncated_at": truncated_at,
        "status": status,
        "undetermined_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def thresholds_used(*, settle_window_samples: int = SETTLE_WINDOW_SAMPLES) -> dict[str, Any]:
    """Every threshold a verdict was compared against, for the receipt."""

    return {
        "lift_clearance_m": LIFT_CLEARANCE_M,
        "destination_min_distance_from_start_m": DESTINATION_MIN_DISTANCE_FROM_START_M,
        "place_radius_m": PLACE_RADIUS_M,
        "place_max_tilt_deg": PLACE_MAX_TILT_DEG,
        "hold_xy_tolerance_m": HOLD_XY_TOLERANCE_M,
        "hold_z_tolerance_m": HOLD_Z_TOLERANCE_M,
        "hold_tilt_tolerance_deg": HOLD_TILT_TOLERANCE_DEG,
        "gripper_full_opening_m": GRIPPER_FULL_OPENING_M,
        "gripper_closed_width_max_m": GRIPPER_CLOSED_WIDTH_MAX_M,
        "grasp_capture_radius_xy_m": GRASP_CAPTURE_RADIUS_XY_M,
        "grasp_contact_force_min_n": GRASP_CONTACT_FORCE_MIN_N,
        "grasp_min_contact_bodies": GRASP_MIN_CONTACT_BODIES,
        "support_clearance_epsilon_m": SUPPORT_CLEARANCE_EPSILON_M,
        "start_upright_max_tilt_deg": START_UPRIGHT_MAX_TILT_DEG,
        "settle_window_samples": int(settle_window_samples),
        "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE,
        "approved_can_top_above_support_m": APPROVED_CAN_TOP_ABOVE_SUPPORT_M,
        "franka_reach_m": FRANKA_REACH_M,
        "sage_collision_roi_min_m": list(SAGE_COLLISION_ROI_MIN_M),
        "sage_collision_roi_max_m": list(SAGE_COLLISION_ROI_MAX_M),
        "robot_base_position_m": list(ROBOT_BASE_POSITION_M),
        "support_plane_z_m": SUPPORT_PLANE_Z_M,
    }


def score_task_episode(
    *,
    samples: Sequence[Mapping[str, Any]],
    destination_position_world_m: Sequence[float],
    support_plane_z_m: float = SUPPORT_PLANE_Z_M,
    settle_window_samples: int = SETTLE_WINDOW_SAMPLES,
    require_sealed_start_pose: bool = True,
) -> dict[str, Any]:
    """Score one episode of the frozen task and return a digest-bound receipt.

    Raises :class:`TaskScoringError` for malformed evidence.  Returns a receipt
    whose ``status`` is ``"scored"`` when every rung on the resolved path was
    decided, and ``"undetermined"`` when it was not.
    """

    normalized = normalize_object_samples(
        samples, require_sealed_start_pose=require_sealed_start_pose
    )
    destination = validate_destination(
        destination_position_world_m,
        start_position_world_m=normalized[0]["position_m"],
    )
    measurements = measure_episode(
        normalized,
        destination_position_world_m=destination["position_world_m"],
        support_plane_z_m=support_plane_z_m,
        settle_window_samples=settle_window_samples,
    )

    predicates: dict[str, bool | None] = {
        OUTCOME_NEVER_MOVED: never_moved(measurements),
        OUTCOME_MOVED: moved(measurements),
        OUTCOME_GRASPED: grasped(measurements),
        OUTCOME_LIFTED: lifted(measurements),
        OUTCOME_TRANSLATED: translated(measurements),
        OUTCOME_PLACED: placed(measurements),
    }
    failures: dict[str, bool | None] = {
        FAILURE_DROPPED: dropped(measurements),
        FAILURE_KNOCKED_OVER: knocked_over(measurements),
        FAILURE_PUSHED_OUT_OF_TASK_ENVELOPE: pushed_out_of_task_envelope(measurements),
        FAILURE_NEVER_MOVED: predicates[OUTCOME_NEVER_MOVED],
    }
    ladder = resolve_outcome_ladder(predicates)

    # The ladder's ``status`` is a claim about the outcome rung.  A failure mode
    # can be undecidable while the ladder still resolves cleanly, so that is
    # reported alongside rather than folded into ``status`` -- but it is always
    # reported, never dropped.
    undetermined_reasons = list(ladder["undetermined_reasons"])
    undetermined_reasons.extend(
        f"failure_mode_{name}_undetermined" for name, value in failures.items() if value is None
    )

    report: dict[str, Any] = {
        "schema_version": TASK_SCORING_SCHEMA_VERSION,
        "status": ladder["status"],
        "outcome": ladder["outcome"],
        "outcome_rank": ladder["outcome_rank"],
        "ladder": ladder["ladder"],
        "ladder_truncated_at": ladder["ladder_truncated_at"],
        "task_succeeded": ladder["outcome"] == OUTCOME_PLACED,
        "predicates": dict(predicates),
        "failure_modes": dict(failures),
        "failure_modes_fully_determined": all(value is not None for value in failures.values()),
        "undetermined_reasons": sorted(set(undetermined_reasons)),
        "measurements": measurements,
        "destination": destination,
        "thresholds": thresholds_used(settle_window_samples=settle_window_samples),
        "sealed_start_pose_required": bool(require_sealed_start_pose),
        "quaternion_convention": "wxyz",
        "judgement_source": JUDGEMENT_SOURCE,
        "rendered_image_consulted": False,
        "learned_judge_consulted": False,
        "candidate_policy_queried": False,
        "caller_asserted_success_accepted": False,
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


__all__ = [
    "APPROVED_CAN_TOP_ABOVE_SUPPORT_M",
    "CAN_START_POSITION_M",
    "DESTINATION_MIN_DISTANCE_FROM_START_M",
    "FAILURE_DROPPED",
    "FAILURE_KNOCKED_OVER",
    "FAILURE_MODES",
    "FAILURE_NEVER_MOVED",
    "FAILURE_PUSHED_OUT_OF_TASK_ENVELOPE",
    "FRANKA_REACH_M",
    "GRASP_CAPTURE_RADIUS_XY_M",
    "GRASP_CONTACT_FORCE_MIN_N",
    "GRASP_EVIDENCE_CONTACT",
    "GRASP_EVIDENCE_HELD_CLEAR",
    "GRASP_EVIDENCE_NOT_GRASPED",
    "GRASP_EVIDENCE_UNAVAILABLE",
    "GRASP_MIN_CLOSURE_M",
    "GRASP_MIN_CONTACT_BODIES",
    "GRIPPER_CLOSED_WIDTH_MAX_M",
    "GRIPPER_FULL_OPENING_M",
    "HOLD_TILT_TOLERANCE_DEG",
    "HOLD_XY_TOLERANCE_M",
    "HOLD_Z_TOLERANCE_M",
    "JUDGEMENT_SOURCE",
    "LIFT_CLEARANCE_M",
    "OUTCOME_GRASPED",
    "OUTCOME_LADDER",
    "OUTCOME_LIFTED",
    "OUTCOME_MOVED",
    "OUTCOME_NEVER_MOVED",
    "OUTCOME_PLACED",
    "OUTCOME_TRANSLATED",
    "PLACE_MAX_TILT_DEG",
    "PLACE_RADIUS_M",
    "QUATERNION_NORM_TOLERANCE",
    "ROBOT_BASE_POSITION_M",
    "ROBOT_BASE_YAW_RAD",
    "SAGE_COLLISION_ROI_MAX_M",
    "SAGE_COLLISION_ROI_MIN_M",
    "SETTLE_WINDOW_SAMPLES",
    "START_UPRIGHT_MAX_TILT_DEG",
    "STATUS_SCORED",
    "STATUS_UNDETERMINED",
    "SUPPORT_CLEARANCE_EPSILON_M",
    "SUPPORT_PLANE_Z_M",
    "TASK_SCORING_SCHEMA_VERSION",
    "TaskScoringError",
    "canonical_digest",
    "dropped",
    "grasp_evidence_for_sample",
    "grasped",
    "inside_task_envelope",
    "knocked_over",
    "lifted",
    "measure_episode",
    "moved",
    "never_moved",
    "normalize_object_samples",
    "placed",
    "pushed_out_of_task_envelope",
    "resolve_outcome_ladder",
    "score_task_episode",
    "thresholds_used",
    "QUATERNION_ORDERS",
    "QUATERNION_ORDER_WXYZ",
    "QUATERNION_ORDER_XYZW",
    "tilt_degrees_from_quaternion",
    "tilt_degrees_from_quaternion_wxyz",
    "translated",
    "validate_destination",
]
