"""Bounded command-orientation candidates for native contact acquisition.

The command pose and the scientific arrival pose are deliberately separate.
Each candidate may ask IK for a small body-frame orientation bias, but native
physics is always scored against the unchanged variant-authoritative target.
No candidate changes position, tolerances, collision policy, or contact gates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "native_task_arena_bounded_orientation_search.v1"

# C81/C82 measured a repeatable target-local orientation residual whose unit
# direction is approximately (0.803, 0.596, 0.009).  Cartesian directions keep
# the search general; this combined direction makes the measured miss an actual
# candidate instead of hoping independent-axis probes compose by accident.
MEASURED_RESIDUAL_DIRECTION_BODY = (0.803, 0.596, 0.009)
DEFAULT_BIAS_MAGNITUDES_RAD = (0.008, 0.016)


class BoundedOrientationError(ValueError):
    """Typed fail-closed input error."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _finite_vector(
    value: Any,
    *,
    length: int,
    error: str,
    require_nonzero: bool = False,
) -> list[float]:
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError
        vector = [float(component) for component in value]
    except (TypeError, ValueError) as exc:
        raise BoundedOrientationError([error]) from exc
    if (
        len(vector) != length
        or not all(math.isfinite(component) for component in vector)
        or (
            require_nonzero
            and math.sqrt(sum(component * component for component in vector))
            <= 1.0e-12
        )
    ):
        raise BoundedOrientationError([error])
    return vector


def _optional_pose_override(
    row: Mapping[str, Any],
    *,
    override_field: str,
    fallback: Sequence[float],
    length: int,
    error: str,
    require_nonzero: bool = False,
) -> list[float]:
    value = row.get(override_field)
    if value is None:
        return list(fallback)
    return _finite_vector(
        value,
        length=length,
        error=error,
        require_nonzero=require_nonzero,
    )


def _unit(values: Sequence[float]) -> tuple[float, float, float]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise BoundedOrientationError(
            ["bounded_orientation_direction_invalid"]
        ) from exc
    norm = math.sqrt(sum(value * value for value in vector))
    if (
        len(vector) != 3
        or not all(math.isfinite(value) for value in vector)
        or norm <= 1.0e-12
    ):
        raise BoundedOrientationError(["bounded_orientation_direction_invalid"])
    return tuple(value / norm for value in vector)


def default_body_rotation_vectors_rad() -> tuple[tuple[float, float, float], ...]:
    """Return a symmetric 20-vector shell covering all three rotation axes."""

    residual = _unit(MEASURED_RESIDUAL_DIRECTION_BODY)
    residual_tangent = _unit((-residual[1], residual[0], 0.0))
    directions = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        residual,
        residual_tangent,
    )
    vectors: list[tuple[float, float, float]] = []
    for magnitude in DEFAULT_BIAS_MAGNITUDES_RAD:
        for direction in directions:
            for sign in (-1.0, 1.0):
                vectors.append(
                    tuple(sign * magnitude * value for value in direction)
                )
    return tuple(vectors)


def apply_body_rotation_vector_xyzw(
    quaternion_xyzw: Sequence[float],
    rotation_vector_body_rad: Sequence[float],
) -> list[float]:
    """Right-multiply one orientation by a target-local rotation vector."""

    try:
        base = [float(value) for value in quaternion_xyzw]
        vector = [float(value) for value in rotation_vector_body_rad]
    except (TypeError, ValueError) as exc:
        raise BoundedOrientationError(
            ["bounded_orientation_quaternion_invalid"]
        ) from exc
    if len(base) != 4 or len(vector) != 3 or not all(
        math.isfinite(value) for value in [*base, *vector]
    ):
        raise BoundedOrientationError(["bounded_orientation_quaternion_invalid"])
    base_norm = math.sqrt(sum(value * value for value in base))
    angle = math.sqrt(sum(value * value for value in vector))
    if base_norm <= 1.0e-12 or angle <= 1.0e-12:
        raise BoundedOrientationError(["bounded_orientation_quaternion_invalid"])
    bx, by, bz, bw = (value / base_norm for value in base)
    scale = math.sin(angle / 2.0) / angle
    dx, dy, dz = (value * scale for value in vector)
    dw = math.cos(angle / 2.0)
    product = [
        bw * dx + bx * dw + by * dz - bz * dy,
        bw * dy - bx * dz + by * dw + bz * dx,
        bw * dz + bx * dy - by * dx + bz * dw,
        bw * dw - bx * dx - by * dy - bz * dz,
    ]
    norm = math.sqrt(sum(value * value for value in product))
    normalized = [value / norm for value in product]
    for value in normalized:
        if abs(value) <= 1.0e-15:
            continue
        if value < 0.0:
            normalized = [-component for component in normalized]
        break
    return [0.0 if abs(value) <= 1.0e-15 else value for value in normalized]


def _phase(plan: Mapping[str, Any], phase_id: str) -> Mapping[str, Any] | None:
    actions = plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        return None
    return next(
        (
            row
            for row in actions
            if isinstance(row, Mapping)
            and row.get("mode") == "ik_pose"
            and str(row.get("phase_id") or "") == phase_id
        ),
        None,
    )


def build_bounded_orientation_postures(
    *,
    variant_plans: Sequence[tuple[str, Mapping[str, Any]]],
    solve_phase: Callable[
        [str, Sequence[float], Sequence[float], Sequence[Sequence[float]]],
        Mapping[str, Any] | None,
    ],
    reference_joint_seeds: Sequence[Sequence[float]],
    rotation_vectors_body_rad: Sequence[Sequence[float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve open and close for each small command bias under both jaw bases."""

    raw_vectors = (
        default_body_rotation_vectors_rad()
        if rotation_vectors_body_rad is None
        else rotation_vectors_body_rad
    )
    try:
        vectors = tuple(
            tuple(
                _finite_vector(
                    vector,
                    length=3,
                    error="bounded_orientation_rotation_vector_invalid",
                    require_nonzero=True,
                )
            )
            for vector in raw_vectors
        )
    except TypeError as exc:
        raise BoundedOrientationError(
            ["bounded_orientation_rotation_vectors_invalid"]
        ) from exc
    if not vectors:
        raise BoundedOrientationError(
            ["bounded_orientation_rotation_vectors_invalid"]
        )
    seeds: list[list[float]] = []
    seen_seeds: set[tuple[float, ...]] = set()
    for raw in reference_joint_seeds:
        try:
            seed = [float(value) for value in raw]
        except (TypeError, ValueError):
            continue
        key = tuple(round(value, 8) for value in seed)
        if (
            len(seed) == 7
            and all(math.isfinite(value) for value in seed)
            and key not in seen_seeds
        ):
            seen_seeds.add(key)
            seeds.append(seed)
    attempts: list[dict[str, Any]] = []
    postures: list[dict[str, Any]] = []
    seen_commands: set[tuple[Any, ...]] = set()
    for variant_id, plan in variant_plans:
        opened = _phase(plan, "contact_open")
        closed = _phase(plan, "contact_close")
        if opened is None or closed is None:
            attempts.append(
                {
                    "variant_id": variant_id,
                    "status": "refused",
                    "reason": "contact_open_or_close_phase_missing",
                }
            )
            continue
        try:
            open_position = _finite_vector(
                opened["target_position_world_m"],
                length=3,
                error="bounded_orientation_contact_open_position_invalid",
            )
            close_position = _finite_vector(
                closed["target_position_world_m"],
                length=3,
                error="bounded_orientation_contact_close_position_invalid",
            )
            open_authority = _finite_vector(
                opened["target_quaternion_world_xyzw"],
                length=4,
                error="bounded_orientation_contact_open_quaternion_invalid",
                require_nonzero=True,
            )
            close_authority = _finite_vector(
                closed["target_quaternion_world_xyzw"],
                length=4,
                error="bounded_orientation_contact_close_quaternion_invalid",
                require_nonzero=True,
            )
            authoritative_open_position = _optional_pose_override(
                opened,
                override_field="arrival_target_position_world_m",
                fallback=open_position,
                length=3,
                error=(
                    "bounded_orientation_contact_open_arrival_position_invalid"
                ),
            )
            authoritative_close_position = _optional_pose_override(
                closed,
                override_field="arrival_target_position_world_m",
                fallback=close_position,
                length=3,
                error=(
                    "bounded_orientation_contact_close_arrival_position_invalid"
                ),
            )
            authoritative_open_quaternion = _optional_pose_override(
                opened,
                override_field="arrival_target_quaternion_world_xyzw",
                fallback=open_authority,
                length=4,
                error=(
                    "bounded_orientation_contact_open_arrival_quaternion_invalid"
                ),
                require_nonzero=True,
            )
            authoritative_close_quaternion = _optional_pose_override(
                closed,
                override_field="arrival_target_quaternion_world_xyzw",
                fallback=close_authority,
                length=4,
                error=(
                    "bounded_orientation_contact_close_arrival_quaternion_invalid"
                ),
                require_nonzero=True,
            )
        except KeyError:
            target_error = BoundedOrientationError(
                ["bounded_orientation_contact_target_missing"]
            )
        except BoundedOrientationError as exc:
            target_error = exc
        else:
            target_error = None
        if target_error is not None:
            attempts.append(
                {
                    "variant_id": variant_id,
                    "status": "refused",
                    "reason": "contact_target_invalid",
                    "blockers": list(target_error.errors),
                }
            )
            continue
        for vector in vectors:
            open_command = apply_body_rotation_vector_xyzw(open_authority, vector)
            close_command = apply_body_rotation_vector_xyzw(close_authority, vector)
            command_key = (
                variant_id,
                *(round(value, 9) for value in open_command),
                *(round(value, 9) for value in close_command),
            )
            if command_key in seen_commands:
                continue
            seen_commands.add(command_key)
            attempt: dict[str, Any] = {
                "candidate_index": len(attempts),
                "variant_id": variant_id,
                "rotation_vector_body_rad": list(vector),
                "open_command_quaternion_world_xyzw": open_command,
                "close_command_quaternion_world_xyzw": close_command,
                "authoritative_open_quaternion_world_xyzw": open_authority,
                "authoritative_close_quaternion_world_xyzw": close_authority,
                "authoritative_open_position_world_m": authoritative_open_position,
                "authoritative_close_position_world_m": authoritative_close_position,
                "authoritative_arrival_open_quaternion_world_xyzw": (
                    authoritative_open_quaternion
                ),
                "authoritative_arrival_close_quaternion_world_xyzw": (
                    authoritative_close_quaternion
                ),
                "position_offset_world_m": [0.0, 0.0, 0.0],
            }
            open_solution = solve_phase(
                "contact_open", open_position, open_command, seeds
            )
            if not isinstance(open_solution, Mapping):
                attempts.append(
                    {**attempt, "status": "refused", "reason": "contact_open_unsolved"}
                )
                continue
            try:
                open_joints = [
                    float(value) for value in open_solution["joint_positions_rad"]
                ]
            except (KeyError, TypeError, ValueError):
                attempts.append(
                    {**attempt, "status": "refused", "reason": "contact_open_invalid"}
                )
                continue
            close_solution = solve_phase(
                "contact_close",
                close_position,
                close_command,
                [open_joints, *seeds],
            )
            if not isinstance(close_solution, Mapping):
                attempts.append(
                    {**attempt, "status": "refused", "reason": "contact_close_unsolved"}
                )
                continue
            try:
                close_joints = [
                    float(value) for value in close_solution["joint_positions_rad"]
                ]
            except (KeyError, TypeError, ValueError):
                attempts.append(
                    {**attempt, "status": "refused", "reason": "contact_close_invalid"}
                )
                continue
            if len(open_joints) != 7 or len(close_joints) != 7:
                attempts.append(
                    {**attempt, "status": "refused", "reason": "joint_vector_invalid"}
                )
                continue
            margins = [
                float(solution.get("minimum_joint_limit_margin_rad") or 0.0)
                for solution in (open_solution, close_solution)
            ]
            metadata = {
                **attempt,
                "status": "solved",
                "open_joint_positions_rad": open_joints,
                "close_joint_positions_rad": close_joints,
                "minimum_joint_limit_margin_rad": min(margins),
            }
            attempts.append(metadata)
            postures.append(
                {
                    "posture_index": len(postures),
                    "seed_index": open_solution.get("seed_index"),
                    "variant_id": variant_id,
                    "posture_source": "bounded_commanded_orientation_search",
                    "joint_positions_rad": open_joints,
                    "minimum_joint_limit_margin_rad": min(margins),
                    "candidate_command_target_position_world_m": open_position,
                    "candidate_command_target_quaternion_world_xyzw": open_command,
                    "authoritative_target_position_world_m": (
                        authoritative_open_position
                    ),
                    "authoritative_target_quaternion_world_xyzw": (
                        authoritative_open_quaternion
                    ),
                    "bounded_orientation_candidate": metadata,
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidates_found" if postures else "unavailable",
        "reason": None if postures else "no_open_close_candidate_solved",
        "represented_candidate_count": len(variant_plans) * len(vectors),
        "solved_candidate_count": len(postures),
        "executed_cell_count": 0,
        "position_offset_search_enabled": False,
        "single_gain_required": True,
        "reference_seed_count": len(seeds),
        "attempts": attempts,
        "claim_boundary": (
            "command_orientation_candidates_only;position_and_authoritative_"
            "arrival_targets_unchanged;native_physics_position_orientation_"
            "collision_joint_contact_and_continuous_episode_gates_remain_"
            "authoritative"
        ),
    }
    return postures, report


__all__ = [
    "BoundedOrientationError",
    "DEFAULT_BIAS_MAGNITUDES_RAD",
    "MEASURED_RESIDUAL_DIRECTION_BODY",
    "SCHEMA_VERSION",
    "apply_body_rotation_vector_xyzw",
    "build_bounded_orientation_postures",
    "default_body_rotation_vectors_rad",
]
