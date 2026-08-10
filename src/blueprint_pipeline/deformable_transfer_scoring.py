"""Pure deterministic scoring for a deformable-to-receptacle transfer.

The scorer consumes native numeric state, keyed by the frozen task entity IDs.
It does not accept caller-authored success predicates.  Its only geometry input
is the frozen destination interior oriented bounding box (OBB); its only task
inputs are numeric thresholds that are emitted again in the result receipt.

The deformable is represented by its native nodal positions, velocities,
deformation gradients, kinematic flags, reset-write counter, and solver
diagnostics.  Release comes from native gripper contact-pair/force readback,
retreat comes from gripper clearance sample points, and receptacle stability
comes from its pose and velocity trace.  Images, policy outputs, and learned
scores are intentionally outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "deformable_transfer_scoring.v1"

# This is a numerical guard for an inclusive geometric boundary, not a task
# tolerance.  It only absorbs round-off introduced by the OBB rotation.
OBB_BOUNDARY_EPSILON_M = 1.0e-9
QUATERNION_NORM_TOLERANCE = 1.0e-6
FREE_KINEMATIC_FLAG = 1.0

OUTCOME_LADDER = (
    "native_state_observed",
    "integrity_preserved",
    "finite_without_divergence",
    "contained",
    "released",
    "settled",
    "strain_within_bound",
    "robot_retreated",
    "receptacle_stable",
    "succeeded",
)


class DeformableTransferScoringError(ValueError):
    """Fail-closed structural or frozen-spec validation error."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _raise(error: str) -> None:
    raise DeformableTransferScoringError([error])


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _mapping(value: Any, *, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(error)
    return value


def _identifier(value: Any, *, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(error)
    return value.strip()


def _has_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return bool(np.issubdtype(value.dtype, np.bool_))
    if isinstance(value, Mapping):
        return any(_has_boolean(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_boolean(item) for item in value)
    return False


def _array(
    value: Any,
    *,
    ndim: int,
    trailing_shape: tuple[int, ...],
    nonempty: bool,
    error: str,
) -> np.ndarray:
    if _has_boolean(value):
        _raise(error)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DeformableTransferScoringError([error]) from exc
    if (
        array.ndim != ndim
        or tuple(array.shape[-len(trailing_shape) :]) != trailing_shape
        or (nonempty and array.shape[0] < 1)
    ):
        _raise(error)
    return array


def _vector(value: Any, *, size: int, error: str) -> np.ndarray:
    return _array(
        value,
        ndim=1,
        trailing_shape=(size,),
        nonempty=True,
        error=error,
    )


def _finite_number(
    value: Any,
    *,
    error: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        _raise(error)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DeformableTransferScoringError([error]) from exc
    if not math.isfinite(result):
        _raise(error)
    if minimum is not None:
        if minimum_inclusive and result < minimum:
            _raise(error)
        if not minimum_inclusive and result <= minimum:
            _raise(error)
    if maximum is not None and result > maximum:
        _raise(error)
    return result


def _nonnegative_integer(value: Any, *, error: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _raise(error)
    result = int(value)
    if result < 0:
        _raise(error)
    return result


def _positive_integer(value: Any, *, error: str) -> int:
    result = _nonnegative_integer(value, error=error)
    if result < 1:
        _raise(error)
    return result


def _quaternion_is_valid(quaternion_xyzw: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(quaternion_xyzw))
        and abs(float(np.linalg.norm(quaternion_xyzw)) - 1.0)
        <= QUATERNION_NORM_TOLERANCE
    )


def _rotation_from_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    if not _quaternion_is_valid(quaternion_xyzw):
        _raise("deformable_transfer_destination_obb_orientation_invalid")
    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _normalized_spec(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(task_spec, error="deformable_transfer_task_spec_invalid")
    deformable_id = _identifier(
        source.get("deformable_entity_id"),
        error="deformable_transfer_deformable_entity_id_invalid",
    )
    destination_id = _identifier(
        source.get("destination_entity_id"),
        error="deformable_transfer_destination_entity_id_invalid",
    )
    robot_id = _identifier(
        source.get("robot_entity_id"),
        error="deformable_transfer_robot_entity_id_invalid",
    )
    if len({deformable_id, destination_id, robot_id}) != 3:
        _raise("deformable_transfer_entity_ids_not_distinct")

    obb = _mapping(
        source.get("destination_interior_obb"),
        error="deformable_transfer_destination_obb_invalid",
    )
    obb_center = _vector(
        obb.get("center_world_m"),
        size=3,
        error="deformable_transfer_destination_obb_center_invalid",
    )
    obb_half_extents = _vector(
        obb.get("half_extents_m"),
        size=3,
        error="deformable_transfer_destination_obb_half_extents_invalid",
    )
    obb_orientation = _vector(
        obb.get("orientation_xyzw"),
        size=4,
        error="deformable_transfer_destination_obb_orientation_invalid",
    )
    if (
        not np.all(np.isfinite(obb_center))
        or not np.all(np.isfinite(obb_half_extents))
        or np.any(obb_half_extents <= 0.0)
        or not _quaternion_is_valid(obb_orientation)
    ):
        _raise("deformable_transfer_destination_obb_invalid")
    obb_orientation = obb_orientation / np.linalg.norm(obb_orientation)

    reference_pose = _mapping(
        source.get("receptacle_reference_pose_world"),
        error="deformable_transfer_receptacle_reference_pose_invalid",
    )
    reference_position = _vector(
        reference_pose.get("position_m"),
        size=3,
        error="deformable_transfer_receptacle_reference_position_invalid",
    )
    reference_orientation = _vector(
        reference_pose.get("orientation_xyzw"),
        size=4,
        error="deformable_transfer_receptacle_reference_orientation_invalid",
    )
    if not np.all(np.isfinite(reference_position)) or not _quaternion_is_valid(
        reference_orientation
    ):
        _raise("deformable_transfer_receptacle_reference_pose_invalid")
    reference_orientation = reference_orientation / np.linalg.norm(reference_orientation)

    thresholds = {
        "minimum_particle_fraction_inside": _finite_number(
            source.get("minimum_particle_fraction_inside"),
            error="deformable_transfer_minimum_fraction_invalid",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        "settle_window_samples": _positive_integer(
            source.get("settle_window_samples"),
            error="deformable_transfer_settle_window_invalid",
        ),
        "maximum_node_speed_mps": _finite_number(
            source.get("maximum_node_speed_mps"),
            error="deformable_transfer_maximum_node_speed_invalid",
            minimum=0.0,
        ),
        "maximum_principal_strain": _finite_number(
            source.get("maximum_principal_strain"),
            error="deformable_transfer_maximum_principal_strain_invalid",
            minimum=0.0,
        ),
        "maximum_release_contact_force_n": _finite_number(
            source.get("maximum_release_contact_force_n"),
            error="deformable_transfer_release_contact_force_invalid",
            minimum=0.0,
        ),
        "minimum_robot_clearance_m": _finite_number(
            source.get("minimum_robot_clearance_m"),
            error="deformable_transfer_minimum_robot_clearance_invalid",
            minimum=0.0,
        ),
        "maximum_receptacle_translation_drift_m": _finite_number(
            source.get("maximum_receptacle_translation_drift_m"),
            error="deformable_transfer_receptacle_translation_drift_invalid",
            minimum=0.0,
        ),
        "maximum_receptacle_rotation_drift_rad": _finite_number(
            source.get("maximum_receptacle_rotation_drift_rad"),
            error="deformable_transfer_receptacle_rotation_drift_invalid",
            minimum=0.0,
            maximum=math.pi,
        ),
        "maximum_receptacle_linear_speed_mps": _finite_number(
            source.get("maximum_receptacle_linear_speed_mps"),
            error="deformable_transfer_receptacle_linear_speed_invalid",
            minimum=0.0,
        ),
        "maximum_receptacle_angular_speed_radps": _finite_number(
            source.get("maximum_receptacle_angular_speed_radps"),
            error="deformable_transfer_receptacle_angular_speed_invalid",
            minimum=0.0,
        ),
    }
    return {
        "deformable_entity_id": deformable_id,
        "destination_entity_id": destination_id,
        "robot_entity_id": robot_id,
        "destination_interior_obb": {
            "center_world_m": obb_center.tolist(),
            "half_extents_m": obb_half_extents.tolist(),
            "orientation_xyzw": obb_orientation.tolist(),
        },
        "receptacle_reference_pose_world": {
            "position_m": reference_position.tolist(),
            "orientation_xyzw": reference_orientation.tolist(),
        },
        **thresholds,
    }


def _sample_entity(
    entities: Mapping[str, Any], entity_id: str, *, sample_index: int
) -> Mapping[str, Any]:
    return _mapping(
        entities.get(entity_id),
        error=f"deformable_transfer_sample_entity_missing:{sample_index}:{entity_id}",
    )


def _contact_value(
    source: Any,
    *,
    deformable_id: str,
    integer: bool,
    error: str,
) -> int | float:
    values = _mapping(source, error=error)
    if deformable_id not in values:
        _raise(error)
    if integer:
        return _nonnegative_integer(values[deformable_id], error=error)
    value = values[deformable_id]
    if isinstance(value, (bool, np.bool_)):
        _raise(error)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DeformableTransferScoringError([error]) from exc


def _normalized_samples(
    samples: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if isinstance(samples, (str, bytes, bytearray)) or not isinstance(samples, Sequence):
        _raise("deformable_transfer_samples_invalid")
    if not samples:
        _raise("deformable_transfer_samples_empty")

    normalized: list[dict[str, Any]] = []
    previous_sample_index: int | None = None
    previous_time: float | None = None
    nodal_shape: tuple[int, ...] | None = None
    gradient_shape: tuple[int, ...] | None = None

    for sequence_index, raw_sample in enumerate(samples):
        sample = _mapping(
            raw_sample, error=f"deformable_transfer_sample_invalid:{sequence_index}"
        )
        sample_index = _nonnegative_integer(
            sample.get("sample_index"),
            error=f"deformable_transfer_sample_index_invalid:{sequence_index}",
        )
        time_seconds = _finite_number(
            sample.get("time_seconds"),
            error=f"deformable_transfer_sample_time_invalid:{sample_index}",
            minimum=0.0,
        )
        if previous_sample_index is not None and sample_index <= previous_sample_index:
            _raise("deformable_transfer_sample_indices_not_increasing")
        if previous_time is not None and time_seconds <= previous_time:
            _raise("deformable_transfer_sample_times_not_increasing")
        previous_sample_index = sample_index
        previous_time = time_seconds

        entities = _mapping(
            sample.get("entities"),
            error=f"deformable_transfer_sample_entities_invalid:{sample_index}",
        )
        deformable = _sample_entity(
            entities, spec["deformable_entity_id"], sample_index=sample_index
        )
        destination = _sample_entity(
            entities, spec["destination_entity_id"], sample_index=sample_index
        )
        robot = _sample_entity(
            entities, spec["robot_entity_id"], sample_index=sample_index
        )

        positions = _array(
            deformable.get("nodal_positions_world_m"),
            ndim=2,
            trailing_shape=(3,),
            nonempty=True,
            error=f"deformable_transfer_nodal_positions_invalid:{sample_index}",
        )
        velocities = _array(
            deformable.get("nodal_velocities_world_mps"),
            ndim=2,
            trailing_shape=(3,),
            nonempty=True,
            error=f"deformable_transfer_nodal_velocities_invalid:{sample_index}",
        )
        gradients = _array(
            deformable.get("deformation_gradients"),
            ndim=3,
            trailing_shape=(3, 3),
            nonempty=True,
            error=f"deformable_transfer_deformation_gradients_invalid:{sample_index}",
        )
        flags = _array(
            deformable.get("nodal_kinematic_flags"),
            ndim=1,
            trailing_shape=(positions.shape[0],),
            nonempty=True,
            error=f"deformable_transfer_kinematic_flags_invalid:{sample_index}",
        )
        if velocities.shape != positions.shape:
            _raise(f"deformable_transfer_nodal_state_shape_mismatch:{sample_index}")
        if nodal_shape is None:
            nodal_shape = positions.shape
            gradient_shape = gradients.shape
        elif positions.shape != nodal_shape or gradients.shape != gradient_shape:
            _raise("deformable_transfer_native_tensor_shape_changed")

        write_count = _nonnegative_integer(
            deformable.get("state_write_count_after_episode_start"),
            error=f"deformable_transfer_state_write_count_invalid:{sample_index}",
        )
        divergence_count = _nonnegative_integer(
            deformable.get("solver_divergence_count"),
            error=f"deformable_transfer_solver_divergence_count_invalid:{sample_index}",
        )

        pose = _mapping(
            destination.get("pose_world"),
            error=f"deformable_transfer_receptacle_pose_invalid:{sample_index}",
        )
        receptacle_position = _vector(
            pose.get("position_m"),
            size=3,
            error=f"deformable_transfer_receptacle_position_invalid:{sample_index}",
        )
        receptacle_orientation = _vector(
            pose.get("orientation_xyzw"),
            size=4,
            error=f"deformable_transfer_receptacle_orientation_invalid:{sample_index}",
        )
        receptacle_linear_velocity = _vector(
            destination.get("linear_velocity_world_mps"),
            size=3,
            error=f"deformable_transfer_receptacle_linear_velocity_invalid:{sample_index}",
        )
        receptacle_angular_velocity = _vector(
            destination.get("angular_velocity_world_radps"),
            size=3,
            error=f"deformable_transfer_receptacle_angular_velocity_invalid:{sample_index}",
        )

        clearance_points = _array(
            robot.get("gripper_clearance_points_world_m"),
            ndim=2,
            trailing_shape=(3,),
            nonempty=True,
            error=f"deformable_transfer_robot_clearance_points_invalid:{sample_index}",
        )
        contact_pairs = _contact_value(
            robot.get("gripper_contact_pair_count_by_entity_id"),
            deformable_id=spec["deformable_entity_id"],
            integer=True,
            error=f"deformable_transfer_gripper_contact_pairs_invalid:{sample_index}",
        )
        contact_force = _contact_value(
            robot.get("gripper_contact_normal_force_n_by_entity_id"),
            deformable_id=spec["deformable_entity_id"],
            integer=False,
            error=f"deformable_transfer_gripper_contact_force_invalid:{sample_index}",
        )

        numeric_arrays = (
            positions,
            velocities,
            gradients,
            flags,
            receptacle_position,
            receptacle_orientation,
            receptacle_linear_velocity,
            receptacle_angular_velocity,
            clearance_points,
        )
        all_numeric_finite = bool(
            all(np.all(np.isfinite(value)) for value in numeric_arrays)
            and math.isfinite(float(contact_force))
        )
        flags_in_native_domain = bool(
            np.all(
                np.isclose(flags, 0.0, atol=OBB_BOUNDARY_EPSILON_M)
                | np.isclose(flags, FREE_KINEMATIC_FLAG, atol=OBB_BOUNDARY_EPSILON_M)
            )
        )
        native_values_valid = bool(
            all_numeric_finite
            and _quaternion_is_valid(receptacle_orientation)
            and float(contact_force) >= 0.0
            and flags_in_native_domain
        )

        normalized.append(
            {
                "sample_index": sample_index,
                "time_seconds": time_seconds,
                "positions": positions,
                "velocities": velocities,
                "gradients": gradients,
                "kinematic_flags": flags,
                "write_count": write_count,
                "divergence_count": divergence_count,
                "receptacle_position": receptacle_position,
                "receptacle_orientation": receptacle_orientation,
                "receptacle_linear_velocity": receptacle_linear_velocity,
                "receptacle_angular_velocity": receptacle_angular_velocity,
                "clearance_points": clearance_points,
                "contact_pairs": int(contact_pairs),
                "contact_force": float(contact_force),
                "all_numeric_finite": all_numeric_finite,
                "native_values_valid": native_values_valid,
            }
        )
    return normalized


def _inside_obb(
    points_world: np.ndarray,
    *,
    center_world: np.ndarray,
    half_extents: np.ndarray,
    world_from_local: np.ndarray,
) -> np.ndarray:
    # With row vectors, world = local @ R.T and local = world @ R.
    local = (points_world - center_world) @ world_from_local
    return np.all(np.abs(local) <= half_extents + OBB_BOUNDARY_EPSILON_M, axis=1)


def _distance_points_to_obb(
    points_world: np.ndarray,
    *,
    center_world: np.ndarray,
    half_extents: np.ndarray,
    world_from_local: np.ndarray,
) -> np.ndarray:
    local = (points_world - center_world) @ world_from_local
    outside = np.maximum(np.abs(local) - half_extents, 0.0)
    return np.linalg.norm(outside, axis=1)


def _quaternion_angle_rad(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> float | None:
    if not _quaternion_is_valid(left_xyzw) or not _quaternion_is_valid(right_xyzw):
        return None
    left = left_xyzw / np.linalg.norm(left_xyzw)
    right = right_xyzw / np.linalg.norm(right_xyzw)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _maximum_principal_strain(samples: Sequence[Mapping[str, Any]]) -> float | None:
    maximum = 0.0
    for sample in samples:
        gradients = sample["gradients"]
        if not np.all(np.isfinite(gradients)):
            return None
        try:
            singular_values = np.linalg.svd(gradients, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        maximum = max(maximum, float(np.max(np.abs(singular_values - 1.0))))
    return maximum


def _minimum_robot_clearance(
    samples: Sequence[Mapping[str, Any]],
    *,
    center_world: np.ndarray,
    half_extents: np.ndarray,
    world_from_local: np.ndarray,
) -> float | None:
    minimum = math.inf
    for sample in samples:
        positions = sample["positions"]
        clearance_points = sample["clearance_points"]
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(clearance_points)):
            return None
        node_distances = np.linalg.norm(
            clearance_points[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2
        )
        obb_distances = _distance_points_to_obb(
            clearance_points,
            center_world=center_world,
            half_extents=half_extents,
            world_from_local=world_from_local,
        )
        minimum = min(
            minimum,
            float(np.min(node_distances)),
            float(np.min(obb_distances)),
        )
    return minimum if math.isfinite(minimum) else None


def _resolved_ladder(predicates: Mapping[str, bool]) -> dict[str, Any]:
    outcome = OUTCOME_LADDER[0]
    rank = 0
    truncated_at: str | None = None
    for index, rung in enumerate(OUTCOME_LADDER[1:], start=1):
        if not predicates[rung]:
            truncated_at = rung
            break
        outcome = rung
        rank = index
    return {
        "outcome": outcome,
        "outcome_rank": rank,
        "outcome_ladder": list(OUTCOME_LADDER),
        "ladder_truncated_at": truncated_at,
    }


def score_deformable_transfer(
    *,
    task_spec: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score raw native entity-state samples against one frozen transfer spec.

    Structurally malformed inputs raise :class:`DeformableTransferScoringError`.
    Well-shaped native evidence containing NaNs, solver divergence, direct state
    writes, or kinematic attachment produces a deterministic non-success receipt
    so that a simulator failure cannot be mistaken for a learned-policy failure.
    """

    spec = _normalized_spec(task_spec)
    state_samples = _normalized_samples(samples, spec)
    window_count = int(spec["settle_window_samples"])
    settle_window_available = len(state_samples) >= window_count
    settle_samples = state_samples[-window_count:] if settle_window_available else state_samples

    obb = spec["destination_interior_obb"]
    obb_center = np.asarray(obb["center_world_m"], dtype=np.float64)
    obb_half_extents = np.asarray(obb["half_extents_m"], dtype=np.float64)
    obb_orientation = np.asarray(obb["orientation_xyzw"], dtype=np.float64)
    world_from_local = _rotation_from_xyzw(obb_orientation)

    all_numeric_finite = all(sample["all_numeric_finite"] for sample in state_samples)
    native_values_valid = all(sample["native_values_valid"] for sample in state_samples)
    # Native integrations expose this as a cumulative counter.  The maximum is
    # therefore the exact observed count; summing samples would double-count a
    # single divergence retained across subsequent readbacks.
    divergence_count = max(sample["divergence_count"] for sample in state_samples)
    post_start_write_count = max(sample["write_count"] for sample in state_samples)
    kinematic_node_count = max(
        int(
            np.count_nonzero(
                ~np.isclose(
                    sample["kinematic_flags"],
                    FREE_KINEMATIC_FLAG,
                    atol=OBB_BOUNDARY_EPSILON_M,
                )
            )
        )
        for sample in state_samples
    )

    final_positions = state_samples[-1]["positions"]
    if np.all(np.isfinite(final_positions)):
        inside = _inside_obb(
            final_positions,
            center_world=obb_center,
            half_extents=obb_half_extents,
            world_from_local=world_from_local,
        )
        particle_fraction_inside: float | None = float(np.mean(inside))
        centroid_world = np.mean(final_positions, axis=0)
        centroid_inside: bool | None = bool(
            _inside_obb(
                centroid_world[np.newaxis, :],
                center_world=obb_center,
                half_extents=obb_half_extents,
                world_from_local=world_from_local,
            )[0]
        )
        centroid_world_m: list[float] | None = centroid_world.tolist()
    else:
        particle_fraction_inside = None
        centroid_inside = None
        centroid_world_m = None

    settle_speed: float | None
    if all(np.all(np.isfinite(sample["velocities"])) for sample in settle_samples):
        settle_speed = max(
            float(np.max(np.linalg.norm(sample["velocities"], axis=1)))
            for sample in settle_samples
        )
    else:
        settle_speed = None

    release_contact_pair_count = max(
        sample["contact_pairs"] for sample in settle_samples
    )
    release_contact_force_n: float | None
    if all(math.isfinite(sample["contact_force"]) for sample in settle_samples):
        release_contact_force_n = max(sample["contact_force"] for sample in settle_samples)
    else:
        release_contact_force_n = None

    maximum_principal_strain = _maximum_principal_strain(state_samples)
    minimum_robot_clearance_m = _minimum_robot_clearance(
        settle_samples,
        center_world=obb_center,
        half_extents=obb_half_extents,
        world_from_local=world_from_local,
    )

    reference = spec["receptacle_reference_pose_world"]
    reference_position = np.asarray(reference["position_m"], dtype=np.float64)
    reference_orientation = np.asarray(reference["orientation_xyzw"], dtype=np.float64)
    translation_drifts: list[float] = []
    rotation_drifts: list[float] = []
    receptacle_linear_speeds: list[float] = []
    receptacle_angular_speeds: list[float] = []
    for sample in settle_samples:
        if np.all(np.isfinite(sample["receptacle_position"])):
            translation_drifts.append(
                float(np.linalg.norm(sample["receptacle_position"] - reference_position))
            )
        rotation = _quaternion_angle_rad(
            sample["receptacle_orientation"], reference_orientation
        )
        if rotation is not None:
            rotation_drifts.append(rotation)
        if np.all(np.isfinite(sample["receptacle_linear_velocity"])):
            receptacle_linear_speeds.append(
                float(np.linalg.norm(sample["receptacle_linear_velocity"]))
            )
        if np.all(np.isfinite(sample["receptacle_angular_velocity"])):
            receptacle_angular_speeds.append(
                float(np.linalg.norm(sample["receptacle_angular_velocity"]))
            )

    maximum_receptacle_translation_drift_m = (
        max(translation_drifts) if len(translation_drifts) == len(settle_samples) else None
    )
    maximum_receptacle_rotation_drift_rad = (
        max(rotation_drifts) if len(rotation_drifts) == len(settle_samples) else None
    )
    maximum_receptacle_linear_speed_mps = (
        max(receptacle_linear_speeds)
        if len(receptacle_linear_speeds) == len(settle_samples)
        else None
    )
    maximum_receptacle_angular_speed_radps = (
        max(receptacle_angular_speeds)
        if len(receptacle_angular_speeds) == len(settle_samples)
        else None
    )

    no_post_start_direct_writes = post_start_write_count == 0
    no_kinematic_attachment = kinematic_node_count == 0
    finite_without_divergence = bool(
        all_numeric_finite and native_values_valid and divergence_count == 0
    )
    contained = bool(
        particle_fraction_inside is not None
        and particle_fraction_inside >= spec["minimum_particle_fraction_inside"]
        and centroid_inside is True
    )
    released = bool(
        settle_window_available
        and release_contact_pair_count == 0
        and release_contact_force_n is not None
        and release_contact_force_n <= spec["maximum_release_contact_force_n"]
    )
    settled = bool(
        settle_window_available
        and settle_speed is not None
        and settle_speed <= spec["maximum_node_speed_mps"]
    )
    strain_within_bound = bool(
        maximum_principal_strain is not None
        and maximum_principal_strain <= spec["maximum_principal_strain"]
    )
    robot_retreated = bool(
        settle_window_available
        and minimum_robot_clearance_m is not None
        and minimum_robot_clearance_m >= spec["minimum_robot_clearance_m"]
    )
    receptacle_stable = bool(
        settle_window_available
        and maximum_receptacle_translation_drift_m is not None
        and maximum_receptacle_translation_drift_m
        <= spec["maximum_receptacle_translation_drift_m"]
        and maximum_receptacle_rotation_drift_rad is not None
        and maximum_receptacle_rotation_drift_rad
        <= spec["maximum_receptacle_rotation_drift_rad"]
        and maximum_receptacle_linear_speed_mps is not None
        and maximum_receptacle_linear_speed_mps
        <= spec["maximum_receptacle_linear_speed_mps"]
        and maximum_receptacle_angular_speed_radps is not None
        and maximum_receptacle_angular_speed_radps
        <= spec["maximum_receptacle_angular_speed_radps"]
    )

    deterministic_success = bool(
        no_post_start_direct_writes
        and no_kinematic_attachment
        and finite_without_divergence
        and contained
        and released
        and settled
        and strain_within_bound
        and robot_retreated
        and receptacle_stable
    )
    predicates = {
        "native_state_observed": True,
        "integrity_preserved": no_post_start_direct_writes
        and no_kinematic_attachment,
        "finite_without_divergence": finite_without_divergence,
        "contained": contained,
        "released": released,
        "settled": settled,
        "strain_within_bound": strain_within_bound,
        "robot_retreated": robot_retreated,
        "receptacle_stable": receptacle_stable,
        "succeeded": deterministic_success,
    }

    failure_reasons: list[str] = []
    if not no_post_start_direct_writes:
        failure_reasons.append("post_start_direct_state_write_observed")
    if not no_kinematic_attachment:
        failure_reasons.append("kinematic_attachment_observed")
    if not all_numeric_finite or not native_values_valid:
        failure_reasons.append("non_finite_or_invalid_native_state")
    if divergence_count > 0:
        failure_reasons.append("solver_divergence_observed")
    if particle_fraction_inside is None or (
        particle_fraction_inside < spec["minimum_particle_fraction_inside"]
    ):
        failure_reasons.append("insufficient_particle_containment")
    if centroid_inside is not True:
        failure_reasons.append("centroid_outside_destination")
    if not settle_window_available:
        failure_reasons.append("settle_window_incomplete")
    if not released:
        failure_reasons.append("gripper_contact_not_released")
    if not settled:
        failure_reasons.append("settle_velocity_exceeded")
    if not strain_within_bound:
        failure_reasons.append("maximum_principal_strain_exceeded")
    if not robot_retreated:
        failure_reasons.append("robot_retreat_clearance_not_met")
    if not receptacle_stable:
        failure_reasons.append("receptacle_pose_or_drift_unstable")

    measurements = {
        "sample_count": len(state_samples),
        "settle_window_available": settle_window_available,
        "settle_window_samples_used": len(settle_samples),
        "particle_count": int(final_positions.shape[0]),
        "particle_fraction_inside": particle_fraction_inside,
        "centroid_world_m": centroid_world_m,
        "centroid_inside": centroid_inside,
        "maximum_settle_node_speed_mps": settle_speed,
        "release_contact_pair_count": release_contact_pair_count,
        "maximum_release_contact_force_n": release_contact_force_n,
        "maximum_absolute_principal_engineering_strain": maximum_principal_strain,
        "minimum_robot_clearance_m": minimum_robot_clearance_m,
        "maximum_receptacle_translation_drift_m": maximum_receptacle_translation_drift_m,
        "maximum_receptacle_rotation_drift_rad": maximum_receptacle_rotation_drift_rad,
        "maximum_receptacle_linear_speed_mps": maximum_receptacle_linear_speed_mps,
        "maximum_receptacle_angular_speed_radps": maximum_receptacle_angular_speed_radps,
        "post_start_state_write_count": post_start_write_count,
        "maximum_kinematic_node_count": kinematic_node_count,
        "solver_divergence_count": divergence_count,
        "all_numeric_state_finite": all_numeric_finite,
        "native_values_valid": native_values_valid,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "entity_ids": {
            "deformable": spec["deformable_entity_id"],
            "destination": spec["destination_entity_id"],
            "robot": spec["robot_entity_id"],
        },
        "frozen_destination_interior_obb": spec["destination_interior_obb"],
        "receptacle_reference_pose_world": spec[
            "receptacle_reference_pose_world"
        ],
        "thresholds": {
            key: value
            for key, value in spec.items()
            if key
            not in {
                "deformable_entity_id",
                "destination_entity_id",
                "robot_entity_id",
                "destination_interior_obb",
                "receptacle_reference_pose_world",
            }
        },
        "measurements": measurements,
        "predicates": predicates,
        **_resolved_ladder(predicates),
        "failure_reasons": failure_reasons,
        "deterministic_success": deterministic_success,
    }
    result["result_digest"] = _canonical_digest(result)
    return result


__all__ = [
    "DeformableTransferScoringError",
    "OBB_BOUNDARY_EPSILON_M",
    "OUTCOME_LADDER",
    "SCHEMA_VERSION",
    "score_deformable_transfer",
]
