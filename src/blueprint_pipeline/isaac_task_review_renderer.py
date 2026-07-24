"""Attempt-bound overview and robot-POV renderer for persistent Isaac tasks."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REVIEW_WIDTH = 640
REVIEW_HEIGHT = 480
REVIEW_RENDERER_PREWARM_SCHEMA_VERSION = "isaac_review_renderer_prewarm.v1"
ARTICULATED_HANDLE_FOCUS_SCHEMA_VERSION = "articulated_handle_focus.v1"
REVIEW_RENDERER_PREWARM_UPDATE_COUNT = 16
REVIEW_DLSS_EXEC_MODE = 2
REVIEW_CAPTURE_RT_SUBFRAMES = 8
# A deliberately wide, square-pixel robot POV keeps both arms and the nearby
# kitchen affordance visible.  These are authored onto the live USD cameras;
# they are not inherited from a prior rendered frame.
REVIEW_FOCAL_LENGTH_MM = 20.0
REVIEW_HORIZONTAL_APERTURE_MM = 76.16789245605469
REVIEW_VERTICAL_APERTURE_MM = 57.125919342041016
# USD cameras default to a 1 m near plane.  A manipulation camera is much
# closer than that to the robot's hands and the task affordance, so retaining
# the USD default clips the entire workspace and leaves only distant walls or
# the environment dome in the image.
REVIEW_NEAR_CLIP_M = 0.05
REVIEW_FAR_CLIP_M = 50.0
# G1's articulation root is its pelvis. The shipped USD can report head_link
# at that same transform, so the camera needs an explicit physical lens-height
# floor plus a small face-forward offset. These values restore the validated
# late-June manipulation camera: roughly 1.35 m at the paid run's 0.84 m root,
# with the lens just in front of the head but still behind the forearms.
ROBOT_POV_HEIGHT_ABOVE_ROOT_M = 0.51
ROBOT_POV_LENS_FORWARD_M = 0.08
ROBOT_POV_HEAD_SURFACE_CLEARANCE_M = 0.03
ROBOT_POV_MAX_LENS_FORWARD_M = 0.22
ROBOT_POV_FALLBACK_HEAD_FORWARD_EXTENT_M = 0.09
ROBOT_POV_MAX_PITCH_DOWN_DEG = 24.0
ROBOT_POV_MAX_HEAD_XY_OFFSET_FROM_ROOT_M = 0.35
ROBOT_POV_ACTIVE_ARM_CALIBRATION_LINK_NAMES = (
    "right_elbow_link",
    "right_wrist_yaw_link",
)
OVERVIEW_BACK_FROM_ROBOT_M = 1.45
OVERVIEW_SIDE_FROM_ROBOT_M = 0.60
OVERVIEW_HEIGHT_ABOVE_ROOT_M = 1.10
MIN_TASK_HORIZONTAL_STANDOFF_M = 0.15
MAX_TASK_HORIZONTAL_STANDOFF_M = 2.0


def configure_review_render_quality(settings: Any | None = None) -> dict[str, Any]:
    """Pin the persistent camera renderer to NVIDIA's SDG quality settings."""

    if settings is None:
        import carb.settings  # type: ignore

        settings = carb.settings.get_settings()
    requested = {
        "/rtx/post/aa/op": 3,
        "/rtx/post/dlss/execMode": REVIEW_DLSS_EXEC_MODE,
    }
    for path, value in requested.items():
        settings.set(path, value)
    effective = {path: settings.get(path) for path in requested}
    if effective != requested:
        raise RuntimeError("review_renderer_quality_setting_readback_mismatch")
    return {
        "schema_version": "isaac_review_render_quality.v1",
        "status": "passed",
        "renderer_mode": "RayTracedLighting",
        "anti_aliasing": "DLSS",
        "dlss_exec_mode": "quality",
        "dlss_exec_mode_value": REVIEW_DLSS_EXEC_MODE,
        "capture_rt_subframes": REVIEW_CAPTURE_RT_SUBFRAMES,
        "settings": effective,
        "claim_boundary": (
            "These settings reduce low-resolution DLSS smearing and settle moved "
            "camera/material history; they do not prove scene-asset fidelity, policy "
            "quality, or task success."
        ),
    }


def _norm(value: Sequence[float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(float(item) ** 2 for item in value)) or 1.0
    return tuple(float(item) / magnitude for item in value)  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _finite_xyz(value: Sequence[float], *, error_code: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise RuntimeError(error_code)
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(error_code)
    return result  # type: ignore[return-value]


def task_camera_plan(
    *,
    robot_root: Sequence[float],
    target: Sequence[float],
    robot_head: Sequence[float] | None = None,
    robot_head_forward_extent_m: float | None = None,
    validate_robot_head_mount: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Build a robot-mounted egocentric POV and a third-person review view.

    The plan is based on the live robot root and target in one world frame.  It
    deliberately does not use the whole-scene bounds (which place the camera
    outside this open-walled kitchen) or a raw ``head_link`` transform (the G1
    asset can expose that link at the articulation root without a lens-height
    offset).
    """

    root = _finite_xyz(robot_root, error_code="review_renderer_robot_root_invalid")
    task = _finite_xyz(target, error_code="review_renderer_target_center_invalid")
    dx = task[0] - root[0]
    dy = task[1] - root[1]
    horizontal_standoff = math.hypot(dx, dy)
    if not MIN_TASK_HORIZONTAL_STANDOFF_M <= horizontal_standoff <= MAX_TASK_HORIZONTAL_STANDOFF_M:
        raise RuntimeError("review_renderer_robot_target_standoff_invalid")
    forward = (dx / horizontal_standoff, dy / horizontal_standoff)
    right = (-forward[1], forward[0])

    head_hint = _finite_xyz(
        robot_head
        if robot_head is not None
        else (root[0], root[1], root[2] + ROBOT_POV_HEIGHT_ABOVE_ROOT_M),
        error_code="review_renderer_robot_head_invalid",
    )
    # A large pelvis-to-head XY offset is invalid while calibrating a standing
    # robot, but becomes expected if the live articulation later tips or falls.
    # Once the camera has a signed head-local mount, rejecting that pose here
    # prevents the backend from finishing the action measurement and reporting
    # the unsafe stance that should terminate the episode.  The caller disables
    # this initial-calibration guard only for an already-calibrated rigid mount.
    if validate_robot_head_mount and math.hypot(
        head_hint[0] - root[0], head_hint[1] - root[1]
    ) > ROBOT_POV_MAX_HEAD_XY_OFFSET_FROM_ROOT_M:
        raise RuntimeError("review_renderer_robot_head_mount_offset_invalid")
    robot_head_mount = (
        head_hint[0],
        head_hint[1],
        max(head_hint[2], root[2] + ROBOT_POV_HEIGHT_ABOVE_ROOT_M),
    )
    head_forward_extent_m = max(0.0, float(robot_head_forward_extent_m or 0.0))
    lens_forward_m = max(
        ROBOT_POV_LENS_FORWARD_M,
        head_forward_extent_m + ROBOT_POV_HEAD_SURFACE_CLEARANCE_M,
    )
    if not math.isfinite(lens_forward_m) or lens_forward_m > ROBOT_POV_MAX_LENS_FORWARD_M:
        raise RuntimeError("review_renderer_robot_head_forward_extent_invalid")
    robot_pov_eye = (
        robot_head_mount[0] + forward[0] * lens_forward_m,
        robot_head_mount[1] + forward[1] * lens_forward_m,
        robot_head_mount[2],
    )
    robot_head_front = (
        robot_head_mount[0] + forward[0] * head_forward_extent_m,
        robot_head_mount[1] + forward[1] * head_forward_extent_m,
        robot_head_mount[2],
    )
    lens_to_task_horizontal_m = math.hypot(
        task[0] - robot_pov_eye[0], task[1] - robot_pov_eye[1]
    )
    minimum_look_at_z = robot_pov_eye[2] - math.tan(
        math.radians(ROBOT_POV_MAX_PITCH_DOWN_DEG)
    ) * lens_to_task_horizontal_m
    robot_pov_target = (task[0], task[1], max(task[2], minimum_look_at_z))
    # This three-quarter view stays behind the robot, but offsets to one side
    # so both the G1 and the microwave face are visible rather than mutually
    # occluding one another.
    overview_eye = (
        root[0]
        - forward[0] * OVERVIEW_BACK_FROM_ROBOT_M
        - right[0] * OVERVIEW_SIDE_FROM_ROBOT_M,
        root[1]
        - forward[1] * OVERVIEW_BACK_FROM_ROBOT_M
        - right[1] * OVERVIEW_SIDE_FROM_ROBOT_M,
        root[2] + OVERVIEW_HEIGHT_ABOVE_ROOT_M,
    )
    robot_torso = (root[0], root[1], root[2] + 0.18)
    overview_target = tuple(
        robot_torso[index] * 0.48 + task[index] * 0.52 for index in range(3)
    )
    return {
        "robot_root": root,
        "robot_head_mount": robot_head_mount,
        "robot_head_front": robot_head_front,
        "robot_pov_lens_offset": (
            forward[0] * lens_forward_m,
            forward[1] * lens_forward_m,
            0.0,
        ),
        "target": task,
        "overview_eye": overview_eye,
        "overview_target": overview_target,  # type: ignore[dict-item]
        "robot_pov_eye": robot_pov_eye,
        "robot_pov_target": robot_pov_target,
    }


def project_world_point(
    contract: Mapping[str, Any], point: Sequence[float]
) -> dict[str, Any]:
    """Project one world point through the exact authored camera contract."""

    world = _finite_xyz(point, error_code="review_renderer_projection_point_invalid")
    eye = _finite_xyz(
        contract.get("camera_world_xyz_m") or (),
        error_code="review_renderer_projection_camera_invalid",
    )
    rows = contract.get("camera_xmat_row_major")
    if not isinstance(rows, Sequence) or len(rows) != 3:
        raise RuntimeError("review_renderer_projection_rotation_invalid")
    axes = [
        _finite_xyz(row, error_code="review_renderer_projection_rotation_invalid")
        for row in rows
    ]
    intrinsics = contract.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise RuntimeError("review_renderer_projection_intrinsics_invalid")
    rel = tuple(world[index] - eye[index] for index in range(3))
    x_camera = _dot(axes[0], rel)
    y_camera = _dot(axes[1], rel)
    depth_m = -_dot(axes[2], rel)
    near_m, far_m = [float(item) for item in contract.get("clipping_range_m") or ()]
    fx = float(intrinsics.get("fx"))
    fy = float(intrinsics.get("fy"))
    cx = float(intrinsics.get("cx"))
    cy = float(intrinsics.get("cy"))
    width = int(intrinsics.get("image_width"))
    height = int(intrinsics.get("image_height"))
    finite = all(
        math.isfinite(item)
        for item in (x_camera, y_camera, depth_m, near_m, far_m, fx, fy, cx, cy)
    )
    in_depth_range = finite and near_m < depth_m < far_m
    if in_depth_range:
        u_px = cx + fx * x_camera / depth_m
        v_px = cy - fy * y_camera / depth_m
    else:
        u_px = math.nan
        v_px = math.nan
    in_frame = bool(
        in_depth_range
        and math.isfinite(u_px)
        and math.isfinite(v_px)
        and 0.0 <= u_px < width
        and 0.0 <= v_px < height
    )
    return {
        "world_xyz_m": list(world),
        "u_px": float(u_px) if math.isfinite(u_px) else None,
        "v_px": float(v_px) if math.isfinite(v_px) else None,
        "depth_m": float(depth_m),
        "in_depth_range": bool(in_depth_range),
        "in_frame": in_frame,
    }


def _pitch_limited_target(
    eye: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float]:
    """Retain the validated late-June head-forward pitch limit."""

    camera = _finite_xyz(eye, error_code="review_renderer_camera_eye_invalid")
    candidate = _finite_xyz(
        target, error_code="review_renderer_camera_target_invalid"
    )
    horizontal_m = math.hypot(
        candidate[0] - camera[0], candidate[1] - camera[1]
    )
    if horizontal_m <= 1e-6:
        return candidate
    minimum_z = camera[2] - math.tan(
        math.radians(ROBOT_POV_MAX_PITCH_DOWN_DEG)
    ) * horizontal_m
    return (candidate[0], candidate[1], max(candidate[2], minimum_z))


def _look_at_projection_contract(
    eye: Sequence[float], target: Sequence[float]
) -> dict[str, Any]:
    """Build the same pinhole contract as the authored USD review camera."""

    camera = _finite_xyz(eye, error_code="review_renderer_camera_eye_invalid")
    look_at = _finite_xyz(
        target, error_code="review_renderer_camera_target_invalid"
    )
    forward = _norm(tuple(look_at[index] - camera[index] for index in range(3)))
    z_axis = tuple(-value for value in forward)
    x_axis = _norm(_cross((0.0, 0.0, 1.0), z_axis))
    y_axis = _cross(z_axis, x_axis)
    return {
        "camera_world_xyz_m": list(camera),
        "camera_xmat_row_major": [list(x_axis), list(y_axis), list(z_axis)],
        "clipping_range_m": [REVIEW_NEAR_CLIP_M, REVIEW_FAR_CLIP_M],
        "intrinsics": {
            "fx": REVIEW_FOCAL_LENGTH_MM * REVIEW_WIDTH / REVIEW_HORIZONTAL_APERTURE_MM,
            "fy": REVIEW_FOCAL_LENGTH_MM * REVIEW_HEIGHT / REVIEW_VERTICAL_APERTURE_MM,
            "cx": REVIEW_WIDTH / 2.0,
            "cy": REVIEW_HEIGHT / 2.0,
            "image_width": REVIEW_WIDTH,
            "image_height": REVIEW_HEIGHT,
        },
    }


def select_robot_pov_calibration_target(
    *,
    eye: Sequence[float],
    task_target: Sequence[float],
    registration_landmarks: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Select the late-June task-and-forearm head-camera calibration target.

    The policy camera remains physically mounted to the head.  This selection
    happens only for its initial head-local calibration; subsequent frames
    inherit head translation and rotation without task re-aiming.
    """

    task = _finite_xyz(
        task_target, error_code="review_renderer_camera_target_invalid"
    )
    positions = {
        str(row.get("landmark_id") or ""): _finite_xyz(
            row.get("world_position_xyz") or (),
            error_code="review_renderer_arm_calibration_landmark_invalid",
        )
        for row in registration_landmarks
        if str(row.get("landmark_id") or "")
        in ROBOT_POV_ACTIVE_ARM_CALIBRATION_LINK_NAMES
    }
    missing = sorted(
        set(ROBOT_POV_ACTIVE_ARM_CALIBRATION_LINK_NAMES) - set(positions)
    )
    if missing:
        raise RuntimeError(
            "review_renderer_arm_calibration_landmarks_missing:" + ",".join(missing)
        )
    elbow = positions["right_elbow_link"]
    wrist = positions["right_wrist_yaw_link"]

    def weighted(*rows: tuple[Sequence[float], float]) -> tuple[float, float, float]:
        total = sum(weight for _, weight in rows)
        return tuple(
            sum(float(point[index]) * weight for point, weight in rows) / total
            for index in range(3)
        )  # type: ignore[return-value]

    # The first weighted target reproduces the final June 29 task/forearm
    # blend.  The stronger forearm candidate handles the same geometry at a
    # wider task stance while the projection scorer keeps the task in frame.
    candidates = (
        ("task_only", task),
        (
            "late_june_task_forearm_context",
            weighted((task, 0.64), (wrist, 0.24), (elbow, 0.12)),
        ),
        (
            "late_june_forearm_weighted",
            weighted((task, 0.38), (wrist, 0.38), (elbow, 0.24)),
        ),
    )
    scored: list[dict[str, Any]] = []
    best: tuple[float, tuple[float, float, float], str] | None = None
    for name, raw_target in candidates:
        target = _pitch_limited_target(eye, raw_target)
        contract = _look_at_projection_contract(eye, target)
        projections = {
            "task_target": project_world_point(contract, task),
            "right_elbow_link": project_world_point(contract, elbow),
            "right_wrist_yaw_link": project_world_point(contract, wrist),
        }
        in_frame_names = sorted(
            key for key, value in projections.items() if value["in_frame"]
        )
        required_in_frame = all(value["in_frame"] for value in projections.values())
        margins = []
        for projection in projections.values():
            if not projection["in_frame"]:
                continue
            u_px = float(projection["u_px"])
            v_px = float(projection["v_px"])
            margins.append(
                min(u_px, REVIEW_WIDTH - u_px, v_px, REVIEW_HEIGHT - v_px)
            )
        minimum_margin_px = min(margins, default=-1.0)
        score = (
            (1000.0 if required_in_frame else 0.0)
            + 100.0 * len(in_frame_names)
            + minimum_margin_px
        )
        scored.append(
            {
                "candidate": name,
                "target_world_xyz_m": list(target),
                "required_geometry_in_frame": required_in_frame,
                "in_frame_names": in_frame_names,
                "minimum_in_frame_margin_px": minimum_margin_px,
                "score": score,
                "projections": projections,
            }
        )
        row = (score, target, name)
        if best is None or row[0] > best[0]:
            best = row
    assert best is not None
    return best[1], {
        "schema_version": "robot_pov_arm_aware_calibration.v1",
        "selected_candidate": best[2],
        "selected_score": best[0],
        "candidates": scored,
        "camera_remains_rigid_head_local_after_initial_calibration": True,
        "policy_camera_reaims_at_task_each_frame": False,
    }


def look_at_quaternion(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0, 0, 1)
) -> tuple[float, float, float, float]:
    forward = _norm(tuple(target[i] - eye[i] for i in range(3)))
    z_axis = tuple(-item for item in forward)
    x_axis = _norm(_cross(up, z_axis))
    y_axis = _cross(z_axis, x_axis)
    return _camera_axes_quaternion((x_axis, y_axis, z_axis))


def _camera_axes_quaternion(
    axes: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Convert USD camera local axes expressed in world space to a quaternion."""

    if len(axes) != 3:
        raise RuntimeError("review_renderer_camera_axes_invalid")
    x_axis, y_axis, z_axis = [
        _finite_xyz(axis, error_code="review_renderer_camera_axes_invalid")
        for axis in axes
    ]
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        return (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)


def rigid_head_camera_mount(
    *,
    head_origin: Sequence[float],
    head_axes: Sequence[Sequence[float]],
    camera_eye: Sequence[float],
    camera_axes: Sequence[Sequence[float]],
) -> dict[str, list[Any]]:
    """Calibrate a camera pose once in the robot head's local coordinates."""

    origin = _finite_xyz(
        head_origin, error_code="review_renderer_head_mount_origin_invalid"
    )
    head_rows = [
        _norm(_finite_xyz(row, error_code="review_renderer_head_mount_axes_invalid"))
        for row in head_axes
    ]
    if len(head_rows) != 3 or any(
        abs(_dot(head_rows[left], head_rows[right])) > 1e-5
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("review_renderer_head_mount_axes_invalid")
    eye = _finite_xyz(camera_eye, error_code="review_renderer_camera_eye_invalid")
    camera_rows = [
        _norm(_finite_xyz(row, error_code="review_renderer_camera_axes_invalid"))
        for row in camera_axes
    ]
    if len(camera_rows) != 3:
        raise RuntimeError("review_renderer_camera_axes_invalid")
    eye_delta = tuple(eye[index] - origin[index] for index in range(3))
    return {
        "camera_origin_head_local_m": [
            _dot(eye_delta, head_axis) for head_axis in head_rows
        ],
        "camera_axes_head_local": [
            [_dot(camera_axis, head_axis) for head_axis in head_rows]
            for camera_axis in camera_rows
        ],
    }


def rigid_head_camera_pose(
    *,
    head_origin: Sequence[float],
    head_axes: Sequence[Sequence[float]],
    mount: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """Compose a fixed head-local camera mount with the current live head pose."""

    origin = _finite_xyz(
        head_origin, error_code="review_renderer_head_mount_origin_invalid"
    )
    head_rows = [
        _norm(_finite_xyz(row, error_code="review_renderer_head_mount_axes_invalid"))
        for row in head_axes
    ]
    local_origin = _finite_xyz(
        mount.get("camera_origin_head_local_m") or (),
        error_code="review_renderer_head_local_camera_origin_invalid",
    )
    local_axes = mount.get("camera_axes_head_local")
    if not isinstance(local_axes, Sequence) or len(local_axes) != 3:
        raise RuntimeError("review_renderer_head_local_camera_axes_invalid")
    local_axis_rows = [
        _finite_xyz(
            row, error_code="review_renderer_head_local_camera_axes_invalid"
        )
        for row in local_axes
    ]

    def to_world(local: Sequence[float]) -> list[float]:
        return [
            sum(float(local[index]) * head_rows[index][world] for index in range(3))
            for world in range(3)
        ]

    world_offset = to_world(local_origin)
    world_axes = [_norm(to_world(local_axis)) for local_axis in local_axis_rows]
    return {
        "camera_world_xyz_m": [
            origin[index] + world_offset[index] for index in range(3)
        ],
        "camera_xmat_row_major": [list(axis) for axis in world_axes],
    }


def articulated_handle_focus_from_mesh(
    *,
    points_world: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    hinge_world_xyz: Sequence[float],
) -> dict[str, Any]:
    """Resolve a disconnected handle component away from an articulated hinge.

    Some kitchen assets model a door panel and its handle as disconnected
    topology inside one mesh prim.  A prim-level bound therefore points at the
    panel center.  This helper identifies the farthest substantial components
    from the hinge and returns their union center without depending on names.
    """

    points = [
        _finite_xyz(point, error_code="review_renderer_handle_mesh_point_invalid")
        for point in points_world
    ]
    counts = [int(value) for value in face_vertex_counts]
    indices = [int(value) for value in face_vertex_indices]
    hinge = _finite_xyz(
        hinge_world_xyz, error_code="review_renderer_handle_hinge_invalid"
    )
    if not points or not counts or any(count < 3 for count in counts):
        return {"status": "fallback", "blockers": ["handle_mesh_topology_empty"]}
    if sum(counts) != len(indices) or any(
        index < 0 or index >= len(points) for index in indices
    ):
        return {"status": "fallback", "blockers": ["handle_mesh_topology_invalid"]}
    parents = list(range(len(points)))
    sizes = [1 for _ in points]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        for vertex in face[1:]:
            union(face[0], vertex)
    component_indices: dict[int, list[int]] = {}
    for index in range(len(points)):
        component_indices.setdefault(find(index), []).append(index)
    components: list[dict[str, Any]] = []
    for members in component_indices.values():
        if len(members) < 4:
            continue
        minimum = [min(points[index][axis] for index in members) for axis in range(3)]
        maximum = [max(points[index][axis] for index in members) for axis in range(3)]
        center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
        extents = [maximum[axis] - minimum[axis] for axis in range(3)]
        components.append(
            {
                "vertex_count": len(members),
                "bbox_min_xyz_m": minimum,
                "bbox_max_xyz_m": maximum,
                "center_xyz_m": center,
                "extent_xyz_m": extents,
                "distance_from_hinge_m": math.dist(center, hinge),
                "bbox_volume_m3": extents[0] * extents[1] * extents[2],
            }
        )
    if len(components) < 2:
        return {
            "status": "fallback",
            "blockers": ["handle_mesh_has_fewer_than_two_components"],
        }
    panel = max(
        components,
        key=lambda row: (int(row["vertex_count"]), float(row["bbox_volume_m3"])),
    )
    candidates = [row for row in components if row is not panel]
    farthest = max(float(row["distance_from_hinge_m"]) for row in candidates)
    minimum_distance = max(
        0.08,
        float(panel["distance_from_hinge_m"]) * 1.35,
        farthest - 0.025,
    )
    selected = [
        row
        for row in candidates
        if float(row["distance_from_hinge_m"]) >= minimum_distance
        and max(float(value) for value in row["extent_xyz_m"]) >= 0.04
    ]
    if not selected:
        return {
            "status": "fallback",
            "blockers": ["disconnected_handle_component_not_resolved"],
            "component_count": len(components),
        }
    minimum = [
        min(float(row["bbox_min_xyz_m"][axis]) for row in selected)
        for axis in range(3)
    ]
    maximum = [
        max(float(row["bbox_max_xyz_m"][axis]) for row in selected)
        for axis in range(3)
    ]
    center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
    return {
        "schema_version": ARTICULATED_HANDLE_FOCUS_SCHEMA_VERSION,
        "status": "resolved_disconnected_articulated_handle",
        "target_world_xyz_m": center,
        "handle_bbox_min_xyz_m": minimum,
        "handle_bbox_max_xyz_m": maximum,
        "hinge_world_xyz_m": list(hinge),
        "component_count": len(components),
        "selected_component_count": len(selected),
        "panel_component": panel,
        "selected_components": selected,
        "claim_boundary": (
            "Disconnected topology and hinge distance identify a manipulation "
            "focus; this is not contact, grasp, or articulation-transition proof."
        ),
    }


class IsaacTaskReviewRenderer:
    def __init__(
        self,
        *,
        stage: Any,
        app: Any,
        robot_prim_path: str,
        output_dir: Path,
        heartbeat_callback: Callable[[], Any] | None = None,
    ):
        import omni.replicator.core as rep  # type: ignore
        from pxr import UsdGeom  # type: ignore

        self.render_quality_contract = configure_review_render_quality()
        self.stage = stage
        self.app = app
        self.robot_prim_path = robot_prim_path
        self.output_dir = Path(output_dir)
        if heartbeat_callback is not None:
            raise RuntimeError(
                "review_renderer_heartbeat_must_attach_after_prewarm"
            )
        self.heartbeat_callback: Callable[[], Any] | None = None
        self._prewarm_completed = False
        self._prewarm_evidence: dict[str, Any] | None = None
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.overview_path = "/World/BlueprintReview/OverviewCamera"
        self.robot_pov_path = "/World/BlueprintReview/RobotPOVCamera"
        self._target_prim_path: str | None = None
        self._task_target_center: tuple[float, float, float] | None = None
        self._robot_head_origin_pose_local_m: list[float] | None = None
        self._robot_pov_head_local_mount: dict[str, list[Any]] | None = None
        self._robot_pov_calibration_target: tuple[float, float, float] | None = None
        self._robot_pov_calibration_landmarks: list[dict[str, Any]] | None = None
        self._robot_pov_arm_aware_calibration: dict[str, Any] | None = None
        self.camera_contracts: dict[str, dict[str, Any]] = {}
        for camera_path in (self.overview_path, self.robot_pov_path):
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
            camera.GetFocalLengthAttr().Set(REVIEW_FOCAL_LENGTH_MM)
            camera.GetHorizontalApertureAttr().Set(REVIEW_HORIZONTAL_APERTURE_MM)
            camera.GetVerticalApertureAttr().Set(REVIEW_VERTICAL_APERTURE_MM)
            camera.GetClippingRangeAttr().Set(
                (REVIEW_NEAR_CLIP_M, REVIEW_FAR_CLIP_M)
            )
        self.rep = rep
        self.annotators = {}
        self.render_products = {}
        for role, camera in (
            ("overview", self.overview_path),
            ("robot_pov", self.robot_pov_path),
        ):
            product = rep.create.render_product(camera, (REVIEW_WIDTH, REVIEW_HEIGHT))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            self.render_products[role] = product
            self.annotators[role] = annotator

    def prewarm(
        self, *, update_count: int = REVIEW_RENDERER_PREWARM_UPDATE_COUNT
    ) -> dict[str, Any]:
        """Realize Replicator products before any Isaac tensor view exists.

        The persistent backend calls this while the timeline is stopped and
        before constructing ``SingleArticulation``.  Reading every attached RGB
        annotator after the updates is deliberate: merely creating a render
        product does not prove Replicator has completed its lazy stage edits.
        """

        if self.heartbeat_callback is not None:
            raise RuntimeError("review_renderer_prewarm_heartbeat_already_attached")
        if self._prewarm_completed:
            raise RuntimeError("review_renderer_prewarm_already_completed")
        cycles = int(update_count)
        if cycles <= 0 or cycles > 32:
            raise RuntimeError("review_renderer_prewarm_update_count_invalid")
        expected_roles = {"overview", "robot_pov"}
        if set(self.annotators) != expected_roles:
            raise RuntimeError("review_renderer_prewarm_annotator_inventory_invalid")
        if set(getattr(self, "render_products", {})) != expected_roles:
            raise RuntimeError("review_renderer_prewarm_render_product_inventory_invalid")

        import numpy as np  # type: ignore

        # ``SimulationApp.update`` advances Kit but does not schedule an
        # attached Replicator annotator capture.  On a cold real kitchen stage
        # that leaves ``get_data`` empty even though both render products have
        # been created.  Use Replicator's synchronous standalone capture seam
        # with a zero timeline delta so the products are genuinely realized
        # without advancing physics.  The backend independently verifies the
        # physics-step delta is zero around this entire call.
        rgb_shapes: dict[str, list[int]] = {}
        render_steps_executed = 0
        invalid_roles = set(expected_roles)
        for _ in range(cycles):
            self.rep.orchestrator.step(
                delta_time=0.0,
                pause_timeline=True,
                wait_for_render=True,
            )
            render_steps_executed += 1
            rgb_shapes = {}
            invalid_roles = set()
            for role, annotator in self.annotators.items():
                data = np.asarray(annotator.get_data())
                if data.ndim != 3 or data.shape[2] not in (3, 4):
                    invalid_roles.add(role)
                    continue
                if data.shape[:2] != (REVIEW_HEIGHT, REVIEW_WIDTH):
                    raise RuntimeError(
                        f"review_renderer_prewarm_{role}_resolution_mismatch"
                    )
                rgb_shapes[role] = [int(value) for value in data.shape]
            if not invalid_roles:
                break
        if invalid_roles:
            role = sorted(invalid_roles)[0]
            raise RuntimeError(f"review_renderer_prewarm_{role}_rgb_invalid")

        evidence = {
            "schema_version": REVIEW_RENDERER_PREWARM_SCHEMA_VERSION,
            "status": "passed",
            "update_count": cycles,
            "render_steps_executed": render_steps_executed,
            "render_step_delta_time_seconds": 0.0,
            "render_step_wait_for_render": True,
            "camera_roles": sorted(expected_roles),
            "render_product_count": len(self.render_products),
            "rgb_shapes": rgb_shapes,
            "render_products_realized": True,
            "render_quality": deepcopy(self.render_quality_contract),
            "heartbeat_callback_attached_during_prewarm": False,
            "heartbeat_callback_attached_after_prewarm": False,
            "blockers": [],
            "claim_boundary": (
                "This proves the attempt-bound Replicator render products returned "
                "live RGB while no heartbeat callback was attached; the backend "
                "separately proves this occurred while physics was stopped and before "
                "constructing any SingleArticulation tensor view."
            ),
        }
        self._prewarm_evidence = evidence
        self._prewarm_completed = True
        return deepcopy(evidence)

    def attach_heartbeat_callback(
        self, callback: Callable[[], Any]
    ) -> dict[str, Any]:
        """Attach live-state refresh only after prewarm and robot standing setup."""

        if not self._prewarm_completed or self._prewarm_evidence is None:
            raise RuntimeError("review_renderer_heartbeat_before_prewarm")
        if not callable(callback):
            raise RuntimeError("review_renderer_heartbeat_callback_invalid")
        if self.heartbeat_callback is not None:
            raise RuntimeError("review_renderer_heartbeat_already_attached")
        self.heartbeat_callback = callback
        self._prewarm_evidence["heartbeat_callback_attached_after_prewarm"] = True
        return self.prewarm_contract()

    def prewarm_contract(self) -> dict[str, Any]:
        if not self._prewarm_completed or self._prewarm_evidence is None:
            raise RuntimeError("review_renderer_prewarm_incomplete")
        return deepcopy(self._prewarm_evidence)

    def _heartbeat(self) -> None:
        if self.heartbeat_callback is not None:
            self.heartbeat_callback()

    def _center(self, prim_path: str) -> tuple[float, float, float]:
        from pxr import Usd, UsdGeom  # type: ignore

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"review_renderer_prim_missing:{prim_path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        candidate = prim
        # A target mesh may delegate its bound to one immediate object parent.
        # Never continue into /World or /root: that silently turns a missing
        # task/robot bound into the center of the entire kitchen.
        for _ in range(2):
            aligned = cache.ComputeWorldBound(candidate).ComputeAlignedRange()
            minimum, maximum = aligned.GetMin(), aligned.GetMax()
            mins = [float(minimum[i]) for i in range(3)]
            maxs = [float(maximum[i]) for i in range(3)]
            values = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
            extents = [maxs[i] - mins[i] for i in range(3)]
            if (
                all(math.isfinite(value) for value in (*values, *extents))
                and all(extent >= 0.0 for extent in extents)
                and max(extents, default=0.0) <= 20.0
                and max((abs(value) for value in values), default=0.0) <= 100.0
            ):
                return tuple(values)  # type: ignore[return-value]
            candidate = candidate.GetParent()
            if str(candidate.GetPath()) in {"/", "/World", "/root"}:
                break
        raise RuntimeError(f"review_renderer_bound_missing:{prim_path}")

    def _articulated_handle_focus(self, prim_path: str) -> dict[str, Any]:
        """Resolve a handle embedded as disconnected topology in one door mesh."""

        from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

        stage = getattr(self, "stage", None)
        if stage is None:
            return {"status": "fallback", "blockers": ["stage_unavailable"]}
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            return {"status": "fallback", "blockers": ["target_is_not_mesh"]}
        joint = None
        for child in prim.GetChildren():
            if child.IsA(UsdPhysics.RevoluteJoint):
                candidate = UsdPhysics.RevoluteJoint(child)
                body1 = candidate.GetBody1Rel().GetTargets()
                if body1 and str(body1[0]) == str(prim.GetPath()):
                    joint = candidate
                    break
        if joint is None:
            return {
                "status": "fallback",
                "blockers": ["target_revolute_joint_not_resolved"],
            }
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            return {"status": "fallback", "blockers": ["target_mesh_data_missing"]}
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        points_world = [
            tuple(float(value) for value in transform.Transform(Gf.Vec3d(point)))
            for point in points
        ]
        local_hinge = Gf.Vec3d(joint.GetLocalPos1Attr().Get())
        hinge_world = transform.Transform(local_hinge)
        result = articulated_handle_focus_from_mesh(
            points_world=points_world,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            hinge_world_xyz=tuple(float(value) for value in hinge_world),
        )
        axis_token = str(joint.GetAxisAttr().Get())
        local_axis = {
            "X": Gf.Vec3d(1.0, 0.0, 0.0),
            "Y": Gf.Vec3d(0.0, 1.0, 0.0),
            "Z": Gf.Vec3d(0.0, 0.0, 1.0),
        }.get(axis_token)
        if local_axis is None:
            return {
                "status": "fallback",
                "blockers": ["target_revolute_joint_axis_invalid"],
            }
        local_rotation = joint.GetLocalRot1Attr().Get()
        joint_rotation = Gf.Rotation(
            Gf.Quatd(
                float(local_rotation.GetReal()),
                Gf.Vec3d(
                    *[float(value) for value in local_rotation.GetImaginary()]
                ),
            )
        )
        world_axis = transform.TransformDir(
            joint_rotation.TransformDir(local_axis)
        ).GetNormalized()
        result["target_prim_path"] = str(prim.GetPath())
        result["joint_prim_path"] = str(joint.GetPrim().GetPath())
        result["joint_axis_token"] = axis_token
        result["joint_world_axis_xyz"] = [float(value) for value in world_axis]
        result["joint_lower_limit_degrees"] = float(
            joint.GetLowerLimitAttr().Get()
        )
        result["joint_upper_limit_degrees"] = float(
            joint.GetUpperLimitAttr().Get()
        )
        result["resolution_source"] = "live_isaac_target_mesh_topology_and_joint"
        return result

    def _world_translation(self, prim_path: str) -> tuple[float, float, float]:
        from pxr import UsdGeom  # type: ignore

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"review_renderer_prim_missing:{prim_path}")
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        return _finite_xyz(
            [float(translation[index]) for index in range(3)],
            error_code=f"review_renderer_world_translation_invalid:{prim_path}",
        )

    def _head_center(self) -> tuple[float, float, float]:
        from pxr import Usd, UsdGeom  # type: ignore

        root = self.stage.GetPrimAtPath(self.robot_prim_path)
        robot_root = self._world_translation(self.robot_prim_path)
        minimum_lens_z = robot_root[2] + ROBOT_POV_HEIGHT_ABOVE_ROOT_M
        pose_prim = root
        for prim in Usd.PrimRange(root):
            if prim.GetName().lower() in {"head", "head_link"}:
                pose_prim = prim
                break
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(pose_prim)
        translation = matrix.ExtractTranslation()
        raw_origin = tuple(float(translation[index]) for index in range(3))
        rotation = matrix.ExtractRotationMatrix()
        rows = [
            _norm([float(rotation[row][column]) for column in range(3)])
            for row in range(3)
        ]
        local_offset = getattr(self, "_robot_head_origin_pose_local_m", None)
        if local_offset is None:
            # The shipped G1 head_link can coincide with the articulation root.
            # Calibrate the physical standing-height correction once, expressed
            # in the live head/root frame; never reapply it along world +Z.
            calibrated_origin = (
                raw_origin[0],
                raw_origin[1],
                max(raw_origin[2], minimum_lens_z),
            )
            delta = tuple(
                calibrated_origin[index] - raw_origin[index] for index in range(3)
            )
            local_offset = [_dot(delta, axis) for axis in rows]
            self._robot_head_origin_pose_local_m = local_offset
        world_offset = [
            sum(float(local_offset[index]) * rows[index][world] for index in range(3))
            for world in range(3)
        ]
        return tuple(
            raw_origin[index] + world_offset[index] for index in range(3)
        )  # type: ignore[return-value]

    def _head_rotation_rows(self) -> list[list[float]]:
        """Read the live head orientation, falling back to the articulation root."""

        from pxr import Usd, UsdGeom  # type: ignore

        root = self.stage.GetPrimAtPath(self.robot_prim_path)
        pose_prim = root
        for prim in Usd.PrimRange(root):
            if prim.GetName().lower() in {"head", "head_link"}:
                pose_prim = prim
                break
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(pose_prim)
        rotation = matrix.ExtractRotationMatrix()
        rows = [
            _norm([float(rotation[row][column]) for column in range(3)])
            for row in range(3)
        ]
        if any(
            abs(_dot(rows[left], rows[right])) > 1e-5
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise RuntimeError("review_renderer_robot_head_rotation_invalid")
        return [list(row) for row in rows]

    def _head_forward_extent(
        self, forward_xy: Sequence[float]
    ) -> float:
        """Return the head bound's support distance toward the task.

        A head-link transform is commonly at the link origin, which can sit
        several centimetres behind the rendered face shell.  Mounting the lens
        from that origin alone put the camera inside the G1 head in attempt 7.
        Use the live rendered bound when available and retain the late-June
        12 cm face-plane fallback when this asset exposes no usable bound.
        """

        from pxr import Usd, UsdGeom  # type: ignore

        root = self.stage.GetPrimAtPath(self.robot_prim_path)
        fallback = ROBOT_POV_FALLBACK_HEAD_FORWARD_EXTENT_M
        fx, fy = _norm((float(forward_xy[0]), float(forward_xy[1]), 0.0))[:2]
        for prim in Usd.PrimRange(root):
            if prim.GetName().lower() not in {"head", "head_link"}:
                continue
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum, maximum = aligned.GetMin(), aligned.GetMax()
            head_origin = self._world_translation(str(prim.GetPath()))
            center_x = (float(maximum[0]) + float(minimum[0])) / 2.0
            center_y = (float(maximum[1]) + float(minimum[1])) / 2.0
            half_x = (float(maximum[0]) - float(minimum[0])) / 2.0
            half_y = (float(maximum[1]) - float(minimum[1])) / 2.0
            extent = (
                fx * (center_x - head_origin[0])
                + fy * (center_y - head_origin[1])
                + abs(fx) * half_x
                + abs(fy) * half_y
            )
            if math.isfinite(extent) and 0.01 <= extent <= (
                ROBOT_POV_MAX_LENS_FORWARD_M - ROBOT_POV_HEAD_SURFACE_CLEARANCE_M
            ):
                return extent
            return fallback
        return fallback

    def _place(
        self, camera_path: str, eye: Sequence[float], target: Sequence[float]
    ) -> dict[str, Any]:
        authored_eye = _finite_xyz(eye, error_code="review_renderer_camera_eye_invalid")
        authored_target = _finite_xyz(
            target, error_code="review_renderer_camera_target_invalid"
        )
        forward = _norm(
            tuple(authored_target[i] - authored_eye[i] for i in range(3))
        )
        if math.dist(authored_eye, authored_target) <= 1e-6:
            raise RuntimeError("review_renderer_camera_eye_target_coincident")
        z_axis = tuple(-item for item in forward)
        x_axis = _norm(_cross((0.0, 0.0, 1.0), z_axis))
        y_axis = _cross(z_axis, x_axis)
        return self._place_axes(
            camera_path,
            authored_eye,
            (x_axis, y_axis, z_axis),
            calibration_target=authored_target,
        )

    def _place_axes(
        self,
        camera_path: str,
        eye: Sequence[float],
        camera_axes: Sequence[Sequence[float]],
        *,
        calibration_target: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Author an exact camera pose without re-aiming its inherited axes."""

        from pxr import Gf, Usd, UsdGeom  # type: ignore

        authored_eye = _finite_xyz(eye, error_code="review_renderer_camera_eye_invalid")
        authored_rows = [
            _norm(_finite_xyz(row, error_code="review_renderer_camera_axes_invalid"))
            for row in camera_axes
        ]
        if len(authored_rows) != 3 or any(
            abs(_dot(authored_rows[left], authored_rows[right])) > 1e-5
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise RuntimeError("review_renderer_camera_axes_invalid")
        quaternion = _camera_axes_quaternion(authored_rows)
        xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(camera_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*authored_eye))
        xform.AddOrientOp().Set(Gf.Quatf(*[float(item) for item in quaternion]))
        # Read back the composed USD transform instead of claiming the
        # requested pose.  This catches op-order, quaternion-convention, and
        # inherited-parent mistakes before a paid render is accepted.
        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        actual_eye = tuple(float(translation[index]) for index in range(3))
        rotation = matrix.ExtractRotationMatrix()
        camera_rows = [
            [float(rotation[row][column]) for column in range(3)]
            for row in range(3)
        ]
        eye_error_m = math.dist(authored_eye, actual_eye)
        axis_alignment = [
            _dot(camera_rows[index], authored_rows[index]) for index in range(3)
        ]
        camera_up_world_z = float(camera_rows[1][2])
        if eye_error_m > 1e-5:
            raise RuntimeError("review_renderer_camera_position_readback_mismatch")
        if min(axis_alignment) < 0.99999:
            raise RuntimeError("review_renderer_camera_axes_readback_mismatch")
        fx = REVIEW_FOCAL_LENGTH_MM * REVIEW_WIDTH / REVIEW_HORIZONTAL_APERTURE_MM
        fy = REVIEW_FOCAL_LENGTH_MM * REVIEW_HEIGHT / REVIEW_VERTICAL_APERTURE_MM
        contract = {
            "available": True,
            "camera_path": camera_path,
            "projection_token": "perspective",
            "resolution": [REVIEW_WIDTH, REVIEW_HEIGHT],
            "camera_world_xyz_m": [float(item) for item in actual_eye],
            # Rows of the world-to-camera rotation.  Camera forward is local
            # -Z in USD, matching the projection used by the controller/FK
            # adapter.
            "camera_xmat_row_major": camera_rows,
            "clipping_range_m": [REVIEW_NEAR_CLIP_M, REVIEW_FAR_CLIP_M],
            "pose_readback": {
                "status": "PASS",
                "position_error_m": eye_error_m,
                "axis_alignment_cosines": axis_alignment,
                "camera_up_world_z": camera_up_world_z,
                "usd_camera_local_forward_axis": "-Z",
                "usd_camera_local_up_axis": "+Y",
            },
            "intrinsics": {
                "available": True,
                "fx": fx,
                "fy": fy,
                "cx": REVIEW_WIDTH / 2.0,
                "cy": REVIEW_HEIGHT / 2.0,
                "image_width": REVIEW_WIDTH,
                "image_height": REVIEW_HEIGHT,
                "focal_length_mm": REVIEW_FOCAL_LENGTH_MM,
                "horizontal_aperture_mm": REVIEW_HORIZONTAL_APERTURE_MM,
                "vertical_aperture_mm": REVIEW_VERTICAL_APERTURE_MM,
                "projection_method": "live_isaac_usd_camera_pinhole_from_focal_aperture",
            },
            "source": "live_isaac_review_camera_authored_for_this_simulator_session",
        }
        if calibration_target is not None:
            target = _finite_xyz(
                calibration_target,
                error_code="review_renderer_camera_target_invalid",
            )
            contract["look_at_target_world_xyz_m"] = list(target)
            desired_forward = _norm(
                tuple(target[index] - actual_eye[index] for index in range(3))
            )
            contract["pose_readback"]["forward_alignment_cosine"] = _dot(
                tuple(-value for value in camera_rows[2]), desired_forward
            )
        return contract

    def camera_contract(self, role: str) -> dict[str, Any]:
        contract = getattr(self, "camera_contracts", {}).get(str(role))
        if not isinstance(contract, Mapping) or contract.get("available") is not True:
            raise RuntimeError(f"review_renderer_{role}_camera_contract_unavailable")
        return deepcopy(dict(contract))

    def set_initial_robot_pov_calibration_landmarks(
        self, landmarks: Sequence[Mapping[str, Any]]
    ) -> None:
        """Bind exact live Isaac arm links before the first policy render."""

        if self._robot_pov_head_local_mount is not None:
            raise RuntimeError("review_renderer_robot_pov_already_calibrated")
        self._robot_pov_calibration_landmarks = [deepcopy(dict(row)) for row in landmarks]

    def follow_live_robot(
        self, *, target_prim_path: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Re-author both review cameras from the live articulation pose.

        The persistent evaluator advances the same Isaac articulation for many
        controller frames.  Camera transforms therefore cannot be treated as
        one-time episode setup: the robot-head POV must follow the live head on
        every physics update.  Its mount is calibrated once in head-local
        coordinates and then inherits head translation and rotation without
        re-aiming at the task.

        This method only authors USD camera transforms.  It deliberately does
        not call ``app.update`` or Replicator, so the caller retains the exact
        one-physics-update-per-controller-frame contract.
        """

        current_target = getattr(self, "_target_prim_path", None)
        requested_target = str(target_prim_path or current_target or "").strip()
        if requested_target:
            if (
                requested_target != current_target
                or getattr(self, "_task_target_center", None) is None
            ):
                self._heartbeat()
                focus = self._articulated_handle_focus(requested_target)
                if focus.get("status") == "resolved_disconnected_articulated_handle":
                    self._task_target_center = _finite_xyz(
                        focus.get("target_world_xyz_m") or (),
                        error_code="review_renderer_handle_focus_invalid",
                    )
                else:
                    self._task_target_center = self._center(requested_target)
                    focus["fallback_target_world_xyz_m"] = list(
                        self._task_target_center
                    )
                self._task_target_focus_evidence = focus
                self._heartbeat()
            self._target_prim_path = requested_target
        target = getattr(self, "_task_target_center", None)
        if target is None:
            raise RuntimeError("review_renderer_task_target_not_initialized")
        robot_root = self._world_translation(self.robot_prim_path)
        robot_head = self._head_center()
        robot_head_axes = self._head_rotation_rows()
        task_dx = float(target[0]) - float(robot_root[0])
        task_dy = float(target[1]) - float(robot_root[1])
        task_horizontal = math.hypot(task_dx, task_dy)
        if task_horizontal <= 1e-6:
            raise RuntimeError("review_renderer_robot_target_standoff_invalid")
        head_forward_extent_m = self._head_forward_extent(
            (task_dx / task_horizontal, task_dy / task_horizontal)
        )
        robot_pov_mount = getattr(self, "_robot_pov_head_local_mount", None)
        mount_calibrated_now = robot_pov_mount is None
        plan = task_camera_plan(
            robot_root=robot_root,
            robot_head=robot_head,
            target=target,
            robot_head_forward_extent_m=head_forward_extent_m,
            validate_robot_head_mount=mount_calibrated_now,
        )
        overview_contract = self._place(
            self.overview_path, plan["overview_eye"], plan["overview_target"]
        )
        if mount_calibrated_now:
            calibration_target = plan["robot_pov_target"]
            calibration_evidence = None
            calibration_landmarks = getattr(
                self, "_robot_pov_calibration_landmarks", None
            )
            if calibration_landmarks:
                calibration_target, calibration_evidence = (
                    select_robot_pov_calibration_target(
                        eye=plan["robot_pov_eye"],
                        task_target=plan["target"],
                        registration_landmarks=calibration_landmarks,
                    )
                )
            robot_pov_contract = self._place(
                self.robot_pov_path,
                plan["robot_pov_eye"],
                calibration_target,
            )
            robot_pov_mount = rigid_head_camera_mount(
                head_origin=robot_head,
                head_axes=robot_head_axes,
                camera_eye=robot_pov_contract["camera_world_xyz_m"],
                camera_axes=robot_pov_contract["camera_xmat_row_major"],
            )
            self._robot_pov_head_local_mount = robot_pov_mount
            self._robot_pov_calibration_target = tuple(plan["target"])
            self._robot_pov_arm_aware_calibration = calibration_evidence
        else:
            robot_pov_pose = rigid_head_camera_pose(
                head_origin=robot_head,
                head_axes=robot_head_axes,
                mount=robot_pov_mount,
            )
            robot_pov_contract = self._place_axes(
                self.robot_pov_path,
                robot_pov_pose["camera_world_xyz_m"],
                robot_pov_pose["camera_xmat_row_major"],
            )
        overview_contract.update(
            {
                "camera_role": "overview",
                "viewpoint_mode": "task_framed_third_person_review",
                "robot_mounted": False,
                "policy_observation_eligible": False,
            }
        )
        robot_pov_contract.update(
            {
                "camera_role": "robot_pov",
                "viewpoint_mode": "robot_head_mounted_egocentric",
                "robot_mounted": True,
                "policy_observation_eligible": True,
                "mount_motion_model": "rigid_head_local_transform",
                "gaze_motion_model": "inherits_head_orientation_no_task_reaim",
                "mount_calibration": "initial_task_facing_then_fixed_for_episode",
                "mount_calibrated_this_update": mount_calibrated_now,
                "head_local_mount": deepcopy(robot_pov_mount),
                "calibration_target_world_xyz_m": list(
                    getattr(self, "_robot_pov_calibration_target", plan["target"])
                ),
                "head_forward_extent_m": head_forward_extent_m,
                "arm_aware_initial_calibration": deepcopy(
                    getattr(self, "_robot_pov_arm_aware_calibration", None)
                ),
                "task_target_focus": deepcopy(
                    getattr(self, "_task_target_focus_evidence", {})
                ),
            }
        )
        projection_requirements = {
            "overview": {
                "robot_root": plan["robot_root"],
                "robot_head_lens": robot_pov_contract["camera_world_xyz_m"],
                "task_target": plan["target"],
            },
            "robot_pov": {
                "task_target": plan["target"],
            },
        }
        for role, contract in (
            ("overview", overview_contract),
            ("robot_pov", robot_pov_contract),
        ):
            projections = {
                name: project_world_point(contract, point)
                for name, point in projection_requirements[role].items()
            }
            required_in_frame = role == "overview" or mount_calibrated_now
            all_points_in_frame = all(row["in_frame"] for row in projections.values())
            if required_in_frame and not all_points_in_frame:
                raise RuntimeError(
                    f"review_renderer_{role}_required_geometry_not_in_frame"
                )
            contract["framing_validation"] = {
                "status": "PASS",
                "required_world_points": projections,
                "all_required_points_in_frame": all_points_in_frame,
                "task_target_required_in_frame_this_update": required_in_frame,
                "validation_basis": (
                    "initial_mount_task_and_self_geometry"
                    if role == "robot_pov" and mount_calibrated_now
                    else "rigid_head_local_pose_readback"
                    if role == "robot_pov"
                    else "task_framed_overview"
                ),
                "task_local_camera_plan": {
                    key: list(value) for key, value in plan.items()
                },
            }
            if role == "robot_pov" and mount_calibrated_now:
                head_front_projection = project_world_point(
                    contract, plan["robot_head_front"]
                )
                root_projection = project_world_point(contract, plan["robot_root"])
                if head_front_projection["in_depth_range"]:
                    raise RuntimeError(
                        "review_renderer_robot_pov_head_surface_in_front_of_lens"
                    )
                # A pitched-down head camera can put the root point on the
                # positive side of the optical plane even though it projects
                # far below the image.  Reject visible self-geometry, not an
                # out-of-frame point that cannot contaminate the observation.
                if root_projection["in_frame"]:
                    raise RuntimeError("review_renderer_robot_root_visible_in_head_pov")
                contract["framing_validation"]["excluded_self_geometry"] = {
                    "robot_head_front": head_front_projection,
                    "robot_root": root_projection,
                }
                contract["framing_validation"][
                    "head_surface_and_root_behind_lens"
                ] = True
        self.camera_contracts = {
            "overview": dict(overview_contract or {}),
            "robot_pov": dict(robot_pov_contract or {}),
        }
        return deepcopy(self.camera_contracts)

    def render(self, *, step_index: int, target_prim_path: str) -> list[dict[str, Any]]:
        self.follow_live_robot(target_prim_path=target_prim_path)
        for _ in range(3):
            self._heartbeat()
            self.app.update()
            self._heartbeat()
        # NVIDIA recommends multiple RT subframes after moving cameras or
        # materials so DLSS history cannot smear a prior view into this frame.
        self.rep.orchestrator.step(
            delta_time=0.0,
            pause_timeline=True,
            wait_for_render=True,
            rt_subframes=REVIEW_CAPTURE_RT_SUBFRAMES,
        )
        return self.capture_current(step_index=step_index)

    def capture_current(self, *, step_index: int) -> list[dict[str, Any]]:
        """Persist the latest rendered state without advancing simulation time."""

        import hashlib
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        for role in self.annotators:
            self.camera_contract(role)
        artifacts = []
        for role, annotator in self.annotators.items():
            self._heartbeat()
            data = np.asarray(annotator.get_data())
            self._heartbeat()
            if data.ndim != 3 or data.shape[2] not in (3, 4):
                raise RuntimeError(f"review_renderer_{role}_rgb_invalid")
            rgb = data[:, :, :3].astype("uint8")
            if rgb.shape[:2] != (REVIEW_HEIGHT, REVIEW_WIDTH):
                raise RuntimeError(f"review_renderer_{role}_resolution_mismatch")
            channel_stddev = [float(value) for value in rgb.std(axis=(0, 1))]
            non_uniform = max(channel_stddev, default=0.0) >= 1.0
            path = self.frames_dir / f"{role}_{step_index:04d}.png"
            self._heartbeat()
            Image.fromarray(rgb).save(path)
            self._heartbeat()
            encoded = path.read_bytes()
            self._heartbeat()
            artifacts.append(
                {
                    "camera_role": role,
                    "frame_index": step_index,
                    "path": str(path),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "camera_contract": self.camera_contract(role),
                    "visual_signal": {
                        "status": "completed" if non_uniform else "blocked",
                        "rgb_channel_stddev": channel_stddev,
                        "non_uniform": non_uniform,
                        "blockers": (
                            [] if non_uniform else [f"review_renderer_{role}_visual_signal_too_low"]
                        ),
                    },
                }
            )
        self._heartbeat()
        return artifacts
