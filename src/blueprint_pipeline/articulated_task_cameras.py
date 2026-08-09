"""Resolve policy-input and review cameras for the articulated door task.

Two policy-input views (external, wrist) and one review-only overview are
resolved analytically against the real Franka base, the replacement door
geometry, and the frozen door-state matrix: the handle must project inside
the frame margin with enough pixels at every state, the moving door must stay
visible, and the sight line must not be occluded by a bound obstacle. The
wrist camera tracks the handle as it swings, so a fixed aim cannot pass.

This is an analytic projection screen, not rendered evidence. It selects and
rejects camera candidates cheaply before native render time; the native
render gate still owns lossless frames, manifests, timestamps, sync, and the
final observability verdict. The overview stream is review-only by contract:
it can never enter policy inputs or scoring.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


CAMERA_RESOLUTION_SCHEMA_VERSION = "articulated_task_camera_resolution.v1"


class ArticulatedTaskCameraError(ValueError):
    """Stable, sorted camera-resolution input failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite_vector(value: Any, length: int, error: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ArticulatedTaskCameraError([error])
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ArticulatedTaskCameraError([error]) from exc
    if any(not math.isfinite(item) for item in result):
        raise ArticulatedTaskCameraError([error])
    return result


def _basis(position: Sequence[float], target: Sequence[float]):
    forward = [target[axis] - position[axis] for axis in range(3)]
    norm = math.sqrt(sum(value * value for value in forward))
    if norm <= 1e-9:
        raise ArticulatedTaskCameraError(["articulated_task_camera_degenerate_aim"])
    forward = [value / norm for value in forward]
    world_up = [0.0, 0.0, 1.0]
    right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ]
    right_norm = math.sqrt(sum(value * value for value in right))
    if right_norm <= 1e-9:
        raise ArticulatedTaskCameraError(["articulated_task_camera_degenerate_aim"])
    right = [value / right_norm for value in right]
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return forward, right, up


def _project(
    point: Sequence[float],
    *,
    position: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
    tangent: float,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    relative = [point[axis] - position[axis] for axis in range(3)]
    depth = sum(relative[axis] * forward[axis] for axis in range(3))
    if depth <= 1e-6:
        return None
    aspect = width / height
    ndc_x = sum(relative[axis] * right[axis] for axis in range(3)) / (depth * tangent * aspect)
    ndc_y = sum(relative[axis] * up[axis] for axis in range(3)) / (depth * tangent)
    return ((ndc_x + 1.0) * width / 2.0, (1.0 - ndc_y) * height / 2.0, depth)


def _segment_hits_aabb(
    start: Sequence[float],
    end: Sequence[float],
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> bool:
    low, high = 0.0, 1.0
    for axis in range(3):
        delta = end[axis] - start[axis]
        if abs(delta) <= 1e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        axis_low = (minimum[axis] - start[axis]) / delta
        axis_high = (maximum[axis] - start[axis]) / delta
        if axis_low > axis_high:
            axis_low, axis_high = axis_high, axis_low
        low = max(low, axis_low)
        high = min(high, axis_high)
        if low > high:
            return False
    return True


def resolve_articulated_task_cameras(
    *,
    hinge_origin_world_m: Sequence[float],
    handle_closed_midpoint_world_m: Sequence[float],
    door_state_angles_degrees: Sequence[float],
    franka_base_xy_world_m: Sequence[float],
    task_door_closed_aabb_m: Mapping[str, Any],
    obstacles: Sequence[Mapping[str, Any]],
    external_camera_candidates: Sequence[Mapping[str, Any]],
    overview_camera_candidates: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    vertical_fov_degrees: float,
    frame_margin_fraction: float,
    minimum_handle_pixels: int,
    handle_half_extents_m: Sequence[float] = (0.136, 0.022, 0.026),
    wrist_grasp_half_extents_m: Sequence[float] = (0.030, 0.022, 0.026),
    wrist_standoff_m: float = 0.22,
    wrist_height_offset_m: float = 0.10,
) -> dict[str, Any]:
    """Select external/wrist policy cameras and one review-only overview."""

    hinge = _finite_vector(hinge_origin_world_m, 3, "articulated_task_camera_hinge_invalid")
    handle = _finite_vector(
        handle_closed_midpoint_world_m, 3, "articulated_task_camera_handle_invalid"
    )
    base = _finite_vector(
        franka_base_xy_world_m, 2, "articulated_task_camera_base_invalid"
    )
    half = _finite_vector(
        handle_half_extents_m, 3, "articulated_task_camera_handle_extent_invalid"
    )
    # At grasp standoff the whole bar cannot fit the frame, and it does not
    # need to: the wrist view must keep the grasp region observable while the
    # external view frames the whole handle. Evaluating the same full-bar box
    # for both would reject every physically sensible wrist mount.
    grasp_half = _finite_vector(
        wrist_grasp_half_extents_m, 3, "articulated_task_camera_grasp_extent_invalid"
    )
    if (
        not isinstance(image_width, int)
        or not isinstance(image_height, int)
        or image_width < 16
        or image_height < 16
    ):
        raise ArticulatedTaskCameraError(["articulated_task_camera_image_size_invalid"])
    fov = float(vertical_fov_degrees)
    if not 0.0 < fov < 180.0:
        raise ArticulatedTaskCameraError(["articulated_task_camera_fov_invalid"])
    margin = float(frame_margin_fraction)
    if not 0.0 <= margin < 0.4:
        raise ArticulatedTaskCameraError(["articulated_task_camera_frame_margin_invalid"])
    if not isinstance(minimum_handle_pixels, int) or minimum_handle_pixels < 1:
        raise ArticulatedTaskCameraError(
            ["articulated_task_camera_minimum_pixels_invalid"]
        )
    states: list[float] = []
    for value in door_state_angles_degrees:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ArticulatedTaskCameraError(["articulated_task_camera_states_invalid"])
        states.append(float(value))
    if len(states) < 2:
        raise ArticulatedTaskCameraError(["articulated_task_camera_states_invalid"])

    door_min = _finite_vector(
        task_door_closed_aabb_m.get("aabb_min"), 3, "articulated_task_camera_door_aabb_invalid"
    )
    door_max = _finite_vector(
        task_door_closed_aabb_m.get("aabb_max"), 3, "articulated_task_camera_door_aabb_invalid"
    )
    obstacle_rows = []
    for index, row in enumerate(obstacles):
        obstacle_rows.append(
            {
                "obstacle_id": str(row.get("obstacle_id") or f"obstacle_{index}"),
                "minimum": _finite_vector(
                    row.get("world_aabb_min_m"), 3, f"articulated_task_camera_obstacle_invalid:{index}"
                ),
                "maximum": _finite_vector(
                    row.get("world_aabb_max_m"), 3, f"articulated_task_camera_obstacle_invalid:{index}"
                ),
            }
        )

    radius = math.hypot(handle[0] - hinge[0], handle[1] - hinge[1])
    base_angle = math.atan2(handle[1] - hinge[1], handle[0] - hinge[0])
    tangent = math.tan(math.radians(fov / 2.0))
    margin_x = margin * image_width
    margin_y = margin * image_height

    def _door_outward_normal(state: float) -> list[float]:
        angle = math.radians(state)
        return [-math.sin(angle), math.cos(angle), 0.0]

    def _handle_at(state: float) -> list[float]:
        angle = base_angle + math.radians(state)
        return [
            hinge[0] + radius * math.cos(angle),
            hinge[1] + radius * math.sin(angle),
            handle[2],
        ]

    def _door_corners_at(state: float) -> list[list[float]]:
        angle = math.radians(state)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = []
        for x in (door_min[0], door_max[0]):
            for y in (door_min[1], door_max[1]):
                for z in (door_min[2], door_max[2]):
                    dx, dy = x - hinge[0], y - hinge[1]
                    corners.append(
                        [
                            hinge[0] + cos_a * dx - sin_a * dy,
                            hinge[1] + sin_a * dx + cos_a * dy,
                            z,
                        ]
                    )
        return corners

    def _evaluate(
        position: list[float],
        *,
        track_handle: bool,
        fixed_target: list[float] | None,
        require_handle_contract: bool = True,
    ):
        rows: list[dict[str, Any]] = []
        reasons: list[str] = []
        for state in states:
            handle_point = _handle_at(state)
            aim = handle_point if track_handle else (fixed_target or handle_point)
            if track_handle:
                # A wrist mount approaches along the door's outward normal, so
                # it stays in front of the face it is servoing to.
                normal = _door_outward_normal(state)
                camera_position = [
                    handle_point[0] + wrist_standoff_m * normal[0],
                    handle_point[1] + wrist_standoff_m * normal[1],
                    handle_point[2] + wrist_height_offset_m,
                ]
            else:
                camera_position = position
            forward, right, up = _basis(camera_position, aim)
            extents = grasp_half if track_handle else half
            corners = [
                [
                    handle_point[0] + sx * extents[0],
                    handle_point[1] + sy * extents[1],
                    handle_point[2] + sz * extents[2],
                ]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
            projected = [
                _project(
                    corner,
                    position=camera_position,
                    forward=forward,
                    right=right,
                    up=up,
                    tangent=tangent,
                    width=image_width,
                    height=image_height,
                )
                for corner in corners
            ]
            if any(point is None for point in projected):
                reasons.append(f"handle_behind_camera:{state}")
                break
            xs = [point[0] for point in projected]
            ys = [point[1] for point in projected]
            pixels = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
            inside = (
                min(xs) >= margin_x
                and max(xs) <= image_width - margin_x
                and min(ys) >= margin_y
                and max(ys) <= image_height - margin_y
            )
            if require_handle_contract and not inside:
                reasons.append(f"handle_outside_frame_margin:{state}")
                break
            if require_handle_contract and pixels < minimum_handle_pixels:
                reasons.append(f"handle_pixels_below_minimum:{state}")
                break
            # The door slab itself occludes its own handle: a camera behind
            # the door face would otherwise "see" the handle through the door.
            outward = _door_outward_normal(state)
            to_camera = [
                camera_position[axis] - handle_point[axis] for axis in range(3)
            ]
            if (
                require_handle_contract
                and sum(to_camera[axis] * outward[axis] for axis in range(3)) <= 0.0
            ):
                reasons.append(f"handle_behind_door_face:{state}")
                break
            occluder = None
            for row in obstacle_rows:
                if _segment_hits_aabb(
                    camera_position, handle_point, row["minimum"], row["maximum"]
                ):
                    occluder = row["obstacle_id"]
                    break
            if occluder is not None:
                reasons.append(f"handle_occluded:{state}:{occluder}")
                break
            door_projected = [
                _project(
                    corner,
                    position=camera_position,
                    forward=forward,
                    right=right,
                    up=up,
                    tangent=tangent,
                    width=image_width,
                    height=image_height,
                )
                for corner in _door_corners_at(state)
            ]
            visible_door = [point for point in door_projected if point is not None]
            door_visible = any(
                0 <= point[0] <= image_width and 0 <= point[1] <= image_height
                for point in visible_door
            )
            if not door_visible:
                reasons.append(f"moving_door_not_visible:{state}")
                break
            rows.append(
                {
                    "angle_degrees": state,
                    "handle_world_m": [round(value, 6) for value in handle_point],
                    "camera_position_world_m": [
                        round(value, 6) for value in camera_position
                    ],
                    "handle_pixels": round(pixels, 3),
                    "handle_inside_frame_margin": bool(inside),
                    "moving_door_visible": True,
                }
            )
        return rows, reasons

    rejected: list[dict[str, Any]] = []
    external: dict[str, Any] | None = None
    for candidate in external_camera_candidates:
        camera_id = str(candidate.get("camera_id") or "")
        position = _finite_vector(
            candidate.get("position_world_m"), 3, f"articulated_task_camera_candidate_invalid:{camera_id}"
        )
        rows, reasons = _evaluate(position, track_handle=False, fixed_target=None)
        if reasons or len(rows) != len(states):
            rejected.append({"camera_id": camera_id, "role": "policy_input", "reasons": sorted(set(reasons))})
            continue
        score = min(row["handle_pixels"] for row in rows)
        if external is None or score > external["minimum_handle_pixels"]:
            external = {
                "camera_id": camera_id,
                "role": "policy_input",
                "policy_input": True,
                "scores_episode": False,
                "position_world_m": position,
                "aim_policy": "fixed_aim_at_closed_handle_midpoint",
                "minimum_handle_pixels": score,
                "per_state_visibility": rows,
            }

    wrist_rows, wrist_reasons = _evaluate(
        [0.0, 0.0, 0.0], track_handle=True, fixed_target=None
    )
    wrist = None
    if not wrist_reasons and len(wrist_rows) == len(states):
        wrist = {
            "camera_id": "wrist",
            "role": "policy_input",
            "policy_input": True,
            "scores_episode": False,
            "aim_policy": "rigid_wrist_mount_tracking_the_moving_handle",
            "standoff_m": float(wrist_standoff_m),
            "grasp_region_half_extents_m": grasp_half,
            "height_offset_m": float(wrist_height_offset_m),
            "minimum_handle_pixels": min(row["handle_pixels"] for row in wrist_rows),
            "per_state_visibility": wrist_rows,
        }
    else:
        rejected.append(
            {"camera_id": "wrist", "role": "policy_input", "reasons": sorted(set(wrist_reasons))}
        )

    overview = None
    for candidate in overview_camera_candidates:
        camera_id = str(candidate.get("camera_id") or "")
        position = _finite_vector(
            candidate.get("position_world_m"), 3, f"articulated_task_camera_candidate_invalid:{camera_id}"
        )
        rows, reasons = _evaluate(
            position,
            track_handle=False,
            fixed_target=[
                0.5 * (base[0] + handle[0]),
                0.5 * (base[1] + handle[1]),
                handle[2],
            ],
            require_handle_contract=False,
        )
        if reasons or len(rows) != len(states):
            rejected.append({"camera_id": camera_id, "role": "review_only", "reasons": sorted(set(reasons))})
            continue
        overview = {
            "camera_id": camera_id,
            "role": "review_only",
            "policy_input": False,
            "scores_episode": False,
            "position_world_m": position,
            "aim_policy": "fixed_aim_at_base_handle_midpoint_for_human_review",
            "handle_contract_applied": False,
            "requirement": "moving_door_visible_at_every_state_for_human_review",
            "per_state_visibility": rows,
        }
        break

    blockers: list[str] = []
    if external is None:
        blockers.append("articulated_task_external_camera_unresolved")
    if wrist is None:
        blockers.append("articulated_task_wrist_camera_unresolved")
    if overview is None:
        blockers.append("articulated_task_overview_camera_unresolved")

    receipt: dict[str, Any] = {
        "schema_version": CAMERA_RESOLUTION_SCHEMA_VERSION,
        "status": (
            "cameras_locally_resolved" if not blockers else "articulated_task_cameras_unresolved"
        ),
        "hinge_origin_world_m": hinge,
        "handle_closed_midpoint_world_m": handle,
        "handle_arc_radius_m": round(radius, 6),
        "franka_base_xy_world_m": base,
        "door_state_angles_degrees": states,
        "image": {
            "width": image_width,
            "height": image_height,
            "vertical_fov_degrees": fov,
            "frame_margin_fraction": margin,
            "minimum_handle_pixels": minimum_handle_pixels,
        },
        "external_camera": external,
        "wrist_camera": wrist,
        "overview_camera": overview,
        "rejected_candidates": sorted(rejected, key=lambda row: row["camera_id"]),
        "blockers": blockers,
        "claim_boundary": {
            "analytic_projection_not_rendered_frames": True,
            "native_render_verification_required": True,
            "overview_is_review_only_and_never_scores": True,
            "lossless_media_binding_owned_by_episode_contract": True,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ArticulatedTaskCameraError",
    "CAMERA_RESOLUTION_SCHEMA_VERSION",
    "resolve_articulated_task_cameras",
]
