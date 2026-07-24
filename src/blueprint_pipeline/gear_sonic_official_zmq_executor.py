"""One-step client for the official GEAR-SONIC protocol-v4 controller stack.

The official C++ deployment process remains persistent. This client publishes
one UNITREE_G1_SONIC latent action to its documented ``pose`` endpoint, waits
for the corresponding ``g1_debug`` controller result, and derives FK landmarks
with the official G1 MuJoCo model. It never decodes the latent action itself.

Controller results must carry the pinned protocol-v4 joint-order schema
(:mod:`blueprint_pipeline.gear_sonic_joint_order_contract`); positional-only
results are rejected fail-closed and FK targets are applied by joint name.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .gear_sonic_joint_order_contract import (
    PROTOCOL_V4_BODY_JOINT_NAMES,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PINNED_WBC_SOURCE_REVISION,
    build_isaac_dof_mapping,
    controller_frame_sequence_start,
    validate_model_joint_names,
    pinned_controller_joint_order,
)

ROOT_ENV = "BLUEPRINT_GEAR_SONIC_ROOT"
MODEL_ENV = "BLUEPRINT_GEAR_SONIC_ROBOT_MODEL"
INPUT_ENV = "BLUEPRINT_GEAR_SONIC_INPUT"
OUTPUT_ENV = "BLUEPRINT_GEAR_SONIC_OUTPUT"
ACTION_HOST_ENV = "BLUEPRINT_GEAR_SONIC_ACTION_HOST"
ACTION_PORT_ENV = "BLUEPRINT_GEAR_SONIC_ACTION_PORT"
STATE_HOST_ENV = "BLUEPRINT_GEAR_SONIC_STATE_HOST"
STATE_PORT_ENV = "BLUEPRINT_GEAR_SONIC_STATE_PORT"
DEFAULT_ROOT = "/opt/wbc"
SEALED_REVISION_FILE = ".blueprint-source-revision"
DEFAULT_MODEL = "/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
MOTION_TOKEN_DIM = 64
HAND_DIM = 7
BODY_DIM = len(PROTOCOL_V4_BODY_JOINT_NAMES)
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM
SONIC_TO_PROTOCOL_V4_HAND_INDICES = (4, 5, 6, 0, 1, 2, 3)
STATE_TOPIC = b"g1_debug"
CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION = "gear_sonic_controller_action_sequence.v1"
CAMERA_PROJECTION_SCHEMA_VERSION = "controller_fk_camera_projection_context.v1"
CAMERA_PROJECTION_TRANSFORM = "mujoco_pelvis_relative_to_live_isaac_pelvis_wxyz"
CAMERA_PROJECTION_LIVE_STATUS = "captured_from_live_persistent_isaac_session"
REGISTRATION_MAX_TOLERANCE_M = 0.05


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_detached_head_revision(root: Path) -> str:
    """Read the immutable legacy checkout revision without invoking Git.

    The released monolithic worker checked out the pinned WBC commit as root
    and retained ``.git``.  Episodes run as an unprivileged user, so Git's
    repository ownership policy can reject even a read-only ``rev-parse``.
    The image build used a detached checkout; accepting only a regular,
    non-symlink ``.git/HEAD`` containing one full commit id preserves the same
    fail-closed provenance boundary without changing global Git trust.
    """

    git_dir = root / ".git"
    head = git_dir / "HEAD"
    if (
        git_dir.is_symlink()
        or not git_dir.is_dir()
        or head.is_symlink()
        or not head.is_file()
    ):
        return ""
    try:
        revision = head.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError):
        return ""
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        return ""
    return revision


def _pinned_controller_revision(root: Path) -> str:
    marker = root / SEALED_REVISION_FILE
    if marker.is_file():
        revision = marker.read_text(encoding="utf-8").strip().lower()
        if revision != PINNED_WBC_SOURCE_REVISION:
            raise RuntimeError("official_gear_sonic_controller_revision_mismatch")
        return revision
    detached_revision = _legacy_detached_head_revision(root)
    if detached_revision:
        if detached_revision != PINNED_WBC_SOURCE_REVISION:
            raise RuntimeError("official_gear_sonic_controller_revision_mismatch")
        return detached_revision
    # Legacy monolithic images retain the repository. Thin foundations use the
    # immutable marker above and intentionally contain neither Git nor .git.
    # The legacy image cloned /opt/wbc as root, then runs the episode as the
    # unprivileged ``blueprint`` user.  Scope Git's ownership exception to this
    # exact resolved checkout for this read-only revision query; a global Git
    # configuration mutation would make the provenance check less trustworthy.
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    revision = completed.stdout.strip().lower() if completed.returncode == 0 else ""
    if revision != PINNED_WBC_SOURCE_REVISION:
        raise RuntimeError("official_gear_sonic_controller_revision_mismatch")
    return revision


def _finite_vector(value: Any, *, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}_missing")
    result = [float(item) for item in value]
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_dimension_or_value_invalid")
    return result


def _protocol_v4_action_frame(value: Any) -> list[float]:
    frame = _finite_vector(value, size=ACTION_DIM, name="unitree_g1_sonic_action_frame")
    left = frame[MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM]
    right = frame[MOTION_TOKEN_DIM + HAND_DIM :]
    reorder = SONIC_TO_PROTOCOL_V4_HAND_INDICES
    return frame[:MOTION_TOKEN_DIM] + [left[index] for index in reorder] + [
        right[index] for index in reorder
    ]


def _validated_action_frames(
    action: Mapping[str, Any],
) -> tuple[list[list[float]], dict[str, Any]]:
    selected = _finite_vector(
        action.get("sonic_action_chunk") or action.get("action_chunk"),
        size=ACTION_DIM,
        name="unitree_g1_sonic_action",
    )
    raw_contract = action.get("controller_action")
    if not isinstance(raw_contract, Mapping):
        frames = [selected]
        return frames, {
            "schema_version": CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION,
            "execution_mode": "single_frame_receding_horizon",
            "execution_frame_count": 1,
            "source_horizon_frame_count": 1,
            "frame_dimension": ACTION_DIM,
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "execution_duration_seconds": 0.02,
            "frames_sha256": _canonical(frames),
            "source_frames_sha256": _canonical(frames),
            "legacy_single_frame_contract_synthesized": True,
        }
    contract = dict(raw_contract)
    if contract.get("schema_version") != CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION:
        raise ValueError("official_gear_sonic_controller_action_sequence_schema_mismatch")
    raw_frames = contract.get("frames")
    if isinstance(raw_frames, (str, bytes)) or not isinstance(raw_frames, Sequence):
        raise ValueError("official_gear_sonic_controller_action_sequence_missing")
    frames = [
        _finite_vector(
            frame,
            size=ACTION_DIM,
            name=f"unitree_g1_sonic_action_frame_{index}",
        )
        for index, frame in enumerate(raw_frames)
    ]
    frame_count = int(contract.get("execution_frame_count") or 0)
    source_count = int(contract.get("source_horizon_frame_count") or 0)
    if (
        not frames
        or frame_count != len(frames)
        or source_count < frame_count
        or int(contract.get("frame_dimension") or 0) != ACTION_DIM
    ):
        raise ValueError("official_gear_sonic_controller_action_sequence_shape_invalid")
    frames_sha256 = _canonical(frames)
    if str(contract.get("frames_sha256") or "") != frames_sha256:
        raise ValueError("official_gear_sonic_controller_action_sequence_sha256_mismatch")
    if frames[0] != selected:
        raise ValueError("official_gear_sonic_selected_action_sequence_frame_zero_mismatch")
    control_hz = float(contract.get("control_hz") or 0.0)
    sample_period = float(contract.get("sample_period_seconds") or 0.0)
    if (
        not math.isfinite(control_hz)
        or control_hz <= 0.0
        or not math.isfinite(sample_period)
        or abs(sample_period - 1.0 / control_hz) > 1e-9
    ):
        raise ValueError("official_gear_sonic_controller_action_sequence_timing_invalid")
    expected_duration = frame_count / control_hz
    if abs(float(contract.get("execution_duration_seconds") or 0.0) - expected_duration) > 1e-9:
        raise ValueError("official_gear_sonic_controller_action_sequence_duration_invalid")
    return frames, contract


def _finite_matrix3(value: Any, *, name: str) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name}_missing_or_invalid")
    rows = [_finite_vector(row, size=3, name=name) for row in value]
    for index, row in enumerate(rows):
        norm = math.sqrt(sum(item * item for item in row))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"{name}_row_not_unit:{index}")
    for left in range(3):
        for right in range(left + 1, 3):
            dot = sum(rows[left][index] * rows[right][index] for index in range(3))
            if abs(dot) > 1e-4:
                raise ValueError(f"{name}_rows_not_orthogonal:{left}:{right}")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError(f"{name}_not_proper_rotation")
    return rows


def _normalized_quaternion_wxyz(value: Any, *, name: str) -> list[float]:
    quaternion = _finite_vector(value, size=4, name=name)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name}_norm_invalid")
    return [item / norm for item in quaternion]


def _rotate_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    w, x, y, z = _normalized_quaternion_wxyz(
        quaternion, name="official_projection_pelvis_quaternion"
    )
    vx, vy, vz = _finite_vector(
        vector, size=3, name="official_projection_model_root_relative_xyz"
    )
    rows = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    return [
        rows[row][0] * vx + rows[row][1] * vy + rows[row][2] * vz
        for row in range(3)
    ]


def _validated_live_projection_context(
    projection_context: Mapping[str, Any],
) -> dict[str, Any]:
    context = dict(projection_context)
    if context.get("schema_version") != CAMERA_PROJECTION_SCHEMA_VERSION:
        raise ValueError("official_gear_sonic_camera_projection_schema_mismatch")
    if context.get("status") != CAMERA_PROJECTION_LIVE_STATUS:
        raise ValueError("official_gear_sonic_camera_projection_not_live_session_capture")
    for field in (
        "attempt_id",
        "launch_nonce",
        "simulator_session_id",
        "stage_id",
    ):
        if not str(context.get(field) or "").strip():
            raise ValueError(f"official_gear_sonic_camera_projection_{field}_missing")
    if context.get("coordinate_transform") != CAMERA_PROJECTION_TRANSFORM:
        raise ValueError("official_gear_sonic_camera_projection_transform_mismatch")
    source_frame = dict(context.get("source_frame_artifact") or {})
    source_path = Path(str(source_frame.get("path") or "")).expanduser().resolve()
    expected_sha256 = str(source_frame.get("sha256") or "").strip().lower()
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or len(expected_sha256) != 64
        or _sha256_file(source_path) != expected_sha256
    ):
        raise ValueError("official_gear_sonic_camera_source_frame_missing_or_hash_mismatch")
    pelvis = dict(context.get("live_isaac_pelvis_world_pose") or {})
    if not str(pelvis.get("prim_path") or "").rstrip("/").endswith("/pelvis"):
        raise ValueError("official_gear_sonic_live_isaac_pelvis_prim_invalid")
    _finite_vector(
        pelvis.get("position_xyz"), size=3, name="official_projection_pelvis_world_xyz"
    )
    _normalized_quaternion_wxyz(
        pelvis.get("quaternion_wxyz"), name="official_projection_pelvis_quaternion"
    )
    registration = dict(context.get("standing_cross_simulator_registration") or {})
    if registration.get("status") != (
        "pending_official_mujoco_named_link_residual_verification"
    ) or registration.get("surrogate") is not False:
        raise ValueError("official_gear_sonic_cross_simulator_registration_invalid")
    return context


def _model_relative_to_live_isaac_world(
    relative_xyz: Any, context: Mapping[str, Any]
) -> list[float]:
    pelvis = dict(context.get("live_isaac_pelvis_world_pose") or {})
    origin = _finite_vector(
        pelvis.get("position_xyz"), size=3, name="official_projection_pelvis_world_xyz"
    )
    rotated = _rotate_wxyz(pelvis.get("quaternion_wxyz"), relative_xyz)
    return [origin[index] + rotated[index] for index in range(3)]


def _verify_standing_cross_simulator_registration(
    *, context: Mapping[str, Any], standing_landmarks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    registration = dict(context.get("standing_cross_simulator_registration") or {})
    required_names = [str(name) for name in registration.get("required_landmark_names") or []]
    if len(required_names) < 6 or len(set(required_names)) != len(required_names):
        raise ValueError("official_gear_sonic_registration_required_landmarks_invalid")
    isaac_rows = {
        str(row.get("landmark_id") or ""): dict(row)
        for row in registration.get("isaac_named_link_world_poses") or []
        if isinstance(row, Mapping)
    }
    mujoco_rows = {
        str(row.get("landmark_id") or row.get("name") or ""): dict(row)
        for row in standing_landmarks
        if isinstance(row, Mapping)
    }
    if any(name not in isaac_rows or name not in mujoco_rows for name in required_names):
        raise ValueError("official_gear_sonic_registration_named_link_missing")
    tolerance = float(registration.get("maximum_residual_tolerance_m") or 0.0)
    if not math.isfinite(tolerance) or not 0.0 < tolerance <= REGISTRATION_MAX_TOLERANCE_M:
        raise ValueError("official_gear_sonic_registration_tolerance_invalid")
    residual_rows: list[dict[str, Any]] = []
    for name in required_names:
        expected = _finite_vector(
            isaac_rows[name].get("world_position_xyz"),
            size=3,
            name=f"official_registration_isaac_world_xyz_{name}",
        )
        observed = _model_relative_to_live_isaac_world(
            mujoco_rows[name].get("model_root_relative_xyz"), context
        )
        residual = math.sqrt(
            sum((observed[index] - expected[index]) ** 2 for index in range(3))
        )
        residual_rows.append(
            {
                "landmark_id": name,
                "isaac_world_xyz": expected,
                "transformed_mujoco_world_xyz": observed,
                "residual_m": residual,
            }
        )
    maximum = max(row["residual_m"] for row in residual_rows)
    rms = math.sqrt(
        sum(row["residual_m"] ** 2 for row in residual_rows) / len(residual_rows)
    )
    if maximum > tolerance:
        raise RuntimeError(
            "official_gear_sonic_cross_simulator_registration_residual_exceeded:"
            f"{maximum:.9f}>{tolerance:.9f}"
        )
    return {
        "schema_version": "gear_sonic_isaac_named_link_registration.v1",
        "status": "passed",
        "method": "named_link_residual_after_live_isaac_pelvis_wxyz_transform",
        "landmark_count": len(residual_rows),
        "maximum_residual_m": maximum,
        "rms_residual_m": rms,
        "maximum_residual_tolerance_m": tolerance,
        "residuals": residual_rows,
        "surrogate": False,
    }


def _project_fk_landmarks(
    landmarks: Sequence[Mapping[str, Any]],
    projection_context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Project action-derived MuJoCo FK through the exact live Isaac camera.

    MuJoCo landmarks are expressed relative to the pelvis, then placed through
    the exact live Isaac pelvis WXYZ pose captured with the RGB frame before
    applying the recorded USD camera intrinsics and world-to-camera rotation.
    """

    context = _validated_live_projection_context(projection_context)
    source_frame = dict(context.get("source_frame_artifact") or {})
    source_frame_sha256 = str(source_frame.get("sha256") or "").strip().lower()
    if len(source_frame_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_frame_sha256
    ):
        raise ValueError("official_gear_sonic_camera_source_frame_sha256_invalid")

    camera = dict(context.get("camera_contract") or {})
    if camera.get("available") is not True or camera.get("projection_token") != "perspective":
        raise ValueError("official_gear_sonic_camera_contract_unavailable_or_nonperspective")
    intrinsics = dict(camera.get("intrinsics") or {})
    if intrinsics.get("available") is not True:
        raise ValueError("official_gear_sonic_camera_intrinsics_unavailable")
    fx = float(intrinsics.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or 0.0)
    cx = float(intrinsics.get("cx"))
    cy = float(intrinsics.get("cy"))
    width = int(intrinsics.get("image_width") or 0)
    height = int(intrinsics.get("image_height") or 0)
    if (
        not all(math.isfinite(value) for value in (fx, fy, cx, cy))
        or fx <= 0.0
        or fy <= 0.0
        or width <= 0
        or height <= 0
        or list(camera.get("resolution") or []) != [width, height]
        or int(source_frame.get("width") or 0) != width
        or int(source_frame.get("height") or 0) != height
    ):
        raise ValueError("official_gear_sonic_camera_intrinsics_or_resolution_invalid")
    clipping_range = camera.get("clipping_range_m")
    try:
        near_m, far_m = [float(value) for value in clipping_range]
    except (TypeError, ValueError) as exc:
        raise ValueError("official_gear_sonic_camera_clipping_range_invalid") from exc
    if (
        not isinstance(clipping_range, Sequence)
        or isinstance(clipping_range, (str, bytes, bytearray))
        or len(clipping_range) != 2
        or not all(math.isfinite(value) for value in (near_m, far_m))
        or not 0.0 < near_m < far_m
    ):
        raise ValueError("official_gear_sonic_camera_clipping_range_invalid")
    camera_world = _finite_vector(
        camera.get("camera_world_xyz_m"), size=3, name="official_camera_world_xyz"
    )
    camera_rows = _finite_matrix3(
        camera.get("camera_xmat_row_major"), name="official_camera_xmat"
    )
    context_sha256 = _canonical(context)
    projected: list[dict[str, Any]] = []
    for index, raw_landmark in enumerate(landmarks):
        landmark = dict(raw_landmark)
        relative = _finite_vector(
            landmark.get("model_root_relative_xyz"),
            size=3,
            name=f"official_fk_landmark_root_relative_xyz_{index}",
        )
        world = _model_relative_to_live_isaac_world(relative, context)
        camera_delta = [world[i] - camera_world[i] for i in range(3)]
        camera_xyz = [
            sum(camera_rows[row][column] * camera_delta[column] for column in range(3))
            for row in range(3)
        ]
        depth = -camera_xyz[2]
        projection: dict[str, Any] = {
            "available": False,
            "unavailable_reason": "behind_or_on_live_camera_plane",
            "projection_context_sha256": context_sha256,
            "source_frame_sha256": source_frame_sha256,
        }
        if math.isfinite(depth) and depth > 1e-6:
            if not near_m < depth < far_m:
                projection.update(
                    {
                        "unavailable_reason": "outside_live_camera_depth_range",
                        "depth_m": round(depth, 6),
                    }
                )
                landmark["landmark_id"] = str(
                    landmark.get("landmark_id")
                    or landmark.get("name")
                    or f"landmark_{index}"
                )
                landmark["world_xyz"] = [round(value, 9) for value in world]
                landmark["camera_xyz"] = [round(value, 9) for value in camera_xyz]
                landmark["image_projection"] = projection
                projected.append(landmark)
                continue
            u_px = cx + fx * camera_xyz[0] / depth
            v_px = cy - fy * camera_xyz[1] / depth
            if (
                math.isfinite(u_px)
                and math.isfinite(v_px)
                and 0.0 <= u_px < float(width)
                and 0.0 <= v_px < float(height)
            ):
                projection = {
                    "available": True,
                    "u_px": round(u_px, 6),
                    "v_px": round(v_px, 6),
                    "depth_m": round(depth, 6),
                    "projection_context_sha256": context_sha256,
                    "source_frame_sha256": source_frame_sha256,
                }
            else:
                projection.update(
                    {
                        "unavailable_reason": "outside_live_camera_viewport",
                        # Preserve the finite pinhole projection even though it
                        # lies outside the RGB viewport.  The egocentric WAM
                        # conditioning renderer can then encode its direction
                        # at the corresponding image edge without pretending
                        # the robot link was visible in the source frame.
                        "u_px": round(u_px, 6),
                        "v_px": round(v_px, 6),
                        "depth_m": round(depth, 6),
                    }
                )
        landmark["landmark_id"] = str(
            landmark.get("landmark_id") or landmark.get("name") or f"landmark_{index}"
        )
        landmark["world_xyz"] = [round(value, 9) for value in world]
        landmark["camera_xyz"] = [round(value, 9) for value in camera_xyz]
        landmark["image_projection"] = projection
        projected.append(landmark)
    # A head/robot-POV camera can legitimately frame the task target while all
    # standing arm landmarks are behind the lens or outside the viewport.  The
    # controller/FK boundary is still valid in that case: every landmark is
    # bound to this exact live camera and carries an explicit unavailable
    # reason.  Visibility is a review-composition property, not a prerequisite
    # for accepting and applying an official controller action.
    return projected, context_sha256


def _token_matches(observed: Any, motion_token: Sequence[float]) -> bool:
    if isinstance(observed, (str, bytes)) or not isinstance(observed, Sequence):
        return False
    try:
        values = [float(item) for item in observed]
    except (TypeError, ValueError):
        return False
    if len(values) != len(motion_token):
        return False
    return all(
        math.isfinite(value) and abs(value - float(expected)) <= 1e-6
        for value, expected in zip(values, motion_token)
    )


def _controller_timestamp(value: Any) -> Decimal | None:
    try:
        timestamp = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return timestamp if timestamp.is_finite() else None


def _controller_index(value: Any) -> Decimal | None:
    """Parse the pinned controller's monotonic state-logger entry index."""

    index = _controller_timestamp(value)
    if index is None or index < 0 or index != index.to_integral_value():
        return None
    return index


def _controller_freshness_mode(
    *,
    current_index: Decimal | None,
    last_controller_index: Decimal | None,
    current_timestamp: Decimal | None,
    last_controller_timestamp: Decimal | None,
) -> str | None:
    """Return the official signal proving a controller row is newer.

    ``g1_debug.index`` is the pinned controller's always-present monotonic
    StateLogger entry index. In simulator mode ``ros_timestamp`` is documented
    to remain ``0.0`` when ROS 2 wall-clock is unavailable, so it is only a
    secondary freshness source.
    """

    if (
        current_index is not None
        and last_controller_index is not None
        and current_index > last_controller_index
    ):
        return "strict_monotonic_state_index"
    if (
        current_timestamp is not None
        and last_controller_timestamp is not None
        and current_timestamp > last_controller_timestamp
    ):
        return "strict_ros_timestamp"
    return None


def _controller_frame_matches(value: Any, expected: int) -> bool:
    if not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        value = value[0] if len(value) == 1 else None
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    return numeric.is_finite() and numeric == expected


def _controller_frame_match_mode(
    state: Mapping[str, Any],
    *,
    expected_frame_index: int,
    current_index: Decimal | None = None,
    last_controller_index: Decimal | None = None,
    current_timestamp: Decimal | None,
    last_controller_timestamp: Decimal | None,
    allow_local_frame_fallback: bool = True,
) -> str | None:
    """Return the proven frame-freshness mode for one controller reply.

    The pinned controller treats each streamed protocol-v4 pose as a new
    one-frame temporary motion and therefore reports controller-local
    ``frame_index == 0`` even when the pose carries a monotonic Blueprint
    sequence number.  Exact token and hand echoes bind the reply to the sent
    pose. A requested sequence-frame echo is sufficient with a strictly newer
    official state index (or advancing ROS timestamp when available). A
    controller-local-zero or absent-frame reply is
    accepted only when the caller has separately proven the exact token/hand
    action is unique in the horizon and drained pre-send controller states.
    """

    freshness_mode = _controller_freshness_mode(
        current_index=current_index,
        last_controller_index=last_controller_index,
        current_timestamp=current_timestamp,
        last_controller_timestamp=last_controller_timestamp,
    )
    if freshness_mode is None:
        return None
    if "frame_index" not in state:
        return (
            f"{freshness_mode}_unique_action_without_reported_frame"
            if allow_local_frame_fallback
            else None
        )
    value = state.get("frame_index")
    if _controller_frame_matches(value, expected_frame_index):
        return f"{freshness_mode}_and_requested_sequence_frame"
    if allow_local_frame_fallback and _controller_frame_matches(value, 0):
        return f"{freshness_mode}_unique_action_and_controller_local_frame_zero"
    return None


def _controller_state_matches_expected_frame(
    state: Mapping[str, Any],
    *,
    expected_frame_index: int,
    current_index: Decimal | None = None,
    last_controller_index: Decimal | None = None,
    current_timestamp: Decimal | None,
    last_controller_timestamp: Decimal | None,
    allow_local_frame_fallback: bool = True,
) -> bool:
    """Require a fresh reply with a supported controller frame convention."""

    return (
        _controller_frame_match_mode(
            state,
            expected_frame_index=expected_frame_index,
            current_index=current_index,
            last_controller_index=last_controller_index,
            current_timestamp=current_timestamp,
            last_controller_timestamp=last_controller_timestamp,
            allow_local_frame_fallback=allow_local_frame_fallback,
        )
        is not None
    )


def _well_formed_controller_state(value: Any) -> bool:
    """Return whether a ``g1_debug`` row proves the controller is in CONTROL.

    The pinned controller publishes this topic only from its CONTROL loop. A
    row need not yet echo the current action, but it must carry the finite live
    robot state that the executor will later bind to that action.
    """

    if not isinstance(value, Mapping):
        return False
    for field, size in (
        ("token_state", MOTION_TOKEN_DIM),
        ("body_q_target", BODY_DIM),
        ("body_q_measured", BODY_DIM),
        ("base_quat_measured", 4),
    ):
        try:
            _finite_vector(value.get(field), size=size, name=field)
        except (TypeError, ValueError):
            return False
    return True


def _controller_mode_command_messages(
    builder: Callable[..., bytes],
) -> tuple[bytes, bytes]:
    """Build the planner-start and streamed-control commands in FSM order."""

    return (
        builder(start=True, stop=False, planner=True),
        builder(start=False, stop=False, planner=False),
    )


def _zmq_pubsub_roundtrip(
    *,
    pose_message: bytes,
    command_message: bytes | None = None,
    planner_start_command_message: bytes | None = None,
    stream_command_message: bytes | None = None,
    motion_token: Sequence[float],
    timeout_seconds: float,
    action_endpoint: str | None = None,
    state_endpoint: str | None = None,
    state_topic: bytes = STATE_TOPIC,
    slow_joiner_grace_seconds: float = 0.3,
    left_hand: Sequence[float] | None = None,
    right_hand: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Real PUB/SUB roundtrip: bind the action PUB, connect the state SUB.

    Stale controller states are discarded unless ``token_state`` matches this
    attempt's motion token AND, when hand targets were sent, the echoed
    ``last_left_hand_action``/``last_right_hand_action`` match them; only the
    fully matching reply returns.
    """

    import msgpack  # type: ignore
    import zmq  # type: ignore

    sequenced_controller_start = (
        planner_start_command_message is not None or stream_command_message is not None
    )
    if sequenced_controller_start and (
        planner_start_command_message is None or stream_command_message is None
    ):
        raise ValueError("official_gear_sonic_controller_command_sequence_incomplete")
    if not sequenced_controller_start and command_message is None:
        raise ValueError("official_gear_sonic_controller_command_missing")

    action = action_endpoint or (
        f"tcp://{os.getenv(ACTION_HOST_ENV, '127.0.0.1')}:"
        f"{int(os.getenv(ACTION_PORT_ENV, '5556'))}"
    )
    state_address = state_endpoint or (
        f"tcp://{os.getenv(STATE_HOST_ENV, '127.0.0.1')}:"
        f"{int(os.getenv(STATE_PORT_ENV, '5557'))}"
    )
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    subscriber = context.socket(zmq.SUB)
    publisher.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.SUBSCRIBE, state_topic)
    publisher.bind(action)
    subscriber.connect(state_address)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    control_mode_observed = not sequenced_controller_start
    last_planner_start_send = 0.0
    stream_command_sent = False
    try:
        time.sleep(max(0.0, float(slow_joiner_grace_seconds)))  # PUB/SUB slow joiner.
        if not sequenced_controller_start:
            publisher.send(command_message)
        while time.monotonic() < deadline:
            now = time.monotonic()
            if not control_mode_observed:
                if now - last_planner_start_send >= 0.1:
                    publisher.send(planner_start_command_message)
                    last_planner_start_send = now
            else:
                if sequenced_controller_start and not stream_command_sent:
                    publisher.send(stream_command_message)
                    stream_command_sent = True
                publisher.send(pose_message)
            events = dict(poller.poll(100))
            if subscriber not in events:
                continue
            raw = subscriber.recv()
            if not raw.startswith(state_topic):
                continue
            state = msgpack.unpackb(raw[len(state_topic):], raw=False)
            if not isinstance(state, Mapping):
                continue
            if not control_mode_observed:
                if not _well_formed_controller_state(state):
                    continue
                # A valid g1_debug row proves planner start moved the pinned
                # controller FSM into CONTROL. Switch to streamed mode, then
                # require a fresh exact action/hand echo before returning.
                control_mode_observed = True
                publisher.send(stream_command_message)
                stream_command_sent = True
                publisher.send(pose_message)
                continue
            if not _token_matches(state.get("token_state"), motion_token):
                continue
            if left_hand is not None and not _token_matches(
                state.get("last_left_hand_action"), left_hand
            ):
                continue
            if right_hand is not None and not _token_matches(
                state.get("last_right_hand_action"), right_hand
            ):
                continue
            return dict(state)
        if not control_mode_observed:
            raise TimeoutError("official_gear_sonic_control_mode_entry_timeout")
        raise TimeoutError("official_gear_sonic_matching_controller_state_timeout")
    finally:
        publisher.close()
        subscriber.close()
        context.term()


def _zmq_pubsub_horizon_roundtrip(
    *,
    pose_messages: Sequence[bytes],
    motion_tokens: Sequence[Sequence[float]],
    left_hands: Sequence[Sequence[float]],
    right_hands: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    planner_start_command_message: bytes,
    stream_command_message: bytes,
    control_hz: float,
    timeout_seconds: float,
    action_endpoint: str | None = None,
    state_endpoint: str | None = None,
    state_topic: bytes = STATE_TOPIC,
    slow_joiner_grace_seconds: float = 0.3,
) -> list[dict[str, Any]]:
    """Stream one bounded horizon through one controller PUB/SUB session.

    The planner transition happens once. Each subsequent pose is sent no
    earlier than its declared 50 Hz slot and must receive its own exact token
    and hand echo before the next frame is published.
    """

    import msgpack  # type: ignore
    import zmq  # type: ignore

    frame_count = len(pose_messages)
    if (
        frame_count < 2
        or len(motion_tokens) != frame_count
        or len(left_hands) != frame_count
        or len(right_hands) != frame_count
        or len(frame_indices) != frame_count
    ):
        raise ValueError("official_gear_sonic_horizon_transport_shape_invalid")
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("official_gear_sonic_horizon_transport_control_hz_invalid")
    action_echo_sha256s = [
        _canonical(
            {
                "motion_token": motion_tokens[index],
                "left_hand": left_hands[index],
                "right_hand": right_hands[index],
            }
        )
        for index in range(frame_count)
    ]
    action_echo_counts = Counter(action_echo_sha256s)
    action = action_endpoint or (
        f"tcp://{os.getenv(ACTION_HOST_ENV, '127.0.0.1')}:"
        f"{int(os.getenv(ACTION_PORT_ENV, '5556'))}"
    )
    state_address = state_endpoint or (
        f"tcp://{os.getenv(STATE_HOST_ENV, '127.0.0.1')}:"
        f"{int(os.getenv(STATE_PORT_ENV, '5557'))}"
    )
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    subscriber = context.socket(zmq.SUB)
    publisher.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.SUBSCRIBE, state_topic)
    publisher.bind(action)
    subscriber.connect(state_address)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    try:
        time.sleep(max(0.0, float(slow_joiner_grace_seconds)))
        last_planner_start_send = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_planner_start_send >= 0.1:
                publisher.send(planner_start_command_message)
                last_planner_start_send = now
            events = dict(poller.poll(100))
            if subscriber not in events:
                continue
            raw = subscriber.recv()
            if not raw.startswith(state_topic):
                continue
            state = msgpack.unpackb(raw[len(state_topic) :], raw=False)
            if isinstance(state, Mapping) and _well_formed_controller_state(state):
                last_controller_index = _controller_index(state.get("index"))
                last_controller_timestamp = _controller_timestamp(
                    state.get("ros_timestamp")
                )
                break
        else:
            raise TimeoutError("official_gear_sonic_control_mode_entry_timeout")

        publisher.send(stream_command_message)
        sample_period = 1.0 / control_hz
        sequence_start = time.monotonic()
        previous_send_time: float | None = None
        results: list[dict[str, Any]] = []
        for index, pose_message in enumerate(pose_messages):
            target_send_time = (
                sequence_start
                if previous_send_time is None
                else previous_send_time + sample_period
            )
            remaining_until_slot = target_send_time - time.monotonic()
            if remaining_until_slot > 0.0:
                time.sleep(remaining_until_slot)
            drained_controller_state_count = 0
            # The controller can publish more than one g1_debug row for a
            # streamed pose. Clear every row already available immediately
            # before this send so a queued response from the prior frame cannot
            # satisfy the current frame. Advance both official freshness
            # watermarks while draining to retain the fail-closed boundary.
            while True:
                try:
                    raw = subscriber.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                drained_controller_state_count += 1
                if not raw.startswith(state_topic):
                    continue
                drained_state = msgpack.unpackb(
                    raw[len(state_topic) :], raw=False
                )
                if not isinstance(drained_state, Mapping):
                    continue
                drained_index = _controller_index(drained_state.get("index"))
                drained_timestamp = _controller_timestamp(
                    drained_state.get("ros_timestamp")
                )
                if drained_index is not None and (
                    last_controller_index is None
                    or drained_index > last_controller_index
                ):
                    last_controller_index = drained_index
                if drained_timestamp is not None and (
                    last_controller_timestamp is None
                    or drained_timestamp > last_controller_timestamp
                ):
                    last_controller_timestamp = drained_timestamp
            send_time = time.monotonic()
            previous_send_time = send_time
            publisher.send(pose_message)
            while time.monotonic() < deadline:
                remaining_ms = max(1, min(100, int((deadline - time.monotonic()) * 1000)))
                events = dict(poller.poll(remaining_ms))
                if subscriber not in events:
                    continue
                raw = subscriber.recv()
                if not raw.startswith(state_topic):
                    continue
                state = msgpack.unpackb(raw[len(state_topic) :], raw=False)
                if not isinstance(state, Mapping):
                    continue
                current_index = _controller_index(state.get("index"))
                current_timestamp = _controller_timestamp(state.get("ros_timestamp"))
                frame_match_mode = _controller_frame_match_mode(
                    state,
                    expected_frame_index=frame_indices[index],
                    current_index=current_index,
                    last_controller_index=last_controller_index,
                    current_timestamp=current_timestamp,
                    last_controller_timestamp=last_controller_timestamp,
                    allow_local_frame_fallback=(
                        action_echo_counts[action_echo_sha256s[index]] == 1
                    ),
                )
                if frame_match_mode is None:
                    continue
                if not _token_matches(state.get("token_state"), motion_tokens[index]):
                    continue
                if not _token_matches(
                    state.get("last_left_hand_action"), left_hands[index]
                ) or not _token_matches(
                    state.get("last_right_hand_action"), right_hands[index]
                ):
                    continue
                row = dict(state)
                row["_blueprint_command_send_offset_seconds"] = (
                    send_time - sequence_start
                )
                row["_blueprint_controller_frame_match_mode"] = frame_match_mode
                row["_blueprint_controller_state_queue_drain_count"] = (
                    drained_controller_state_count
                )
                results.append(row)
                if current_index is not None and (
                    last_controller_index is None
                    or current_index > last_controller_index
                ):
                    last_controller_index = current_index
                if current_timestamp is not None and (
                    last_controller_timestamp is None
                    or current_timestamp > last_controller_timestamp
                ):
                    last_controller_timestamp = current_timestamp
                break
            else:
                raise TimeoutError(
                    "official_gear_sonic_matching_controller_state_timeout:"
                    f"frame_{index}"
                )
        return results
    finally:
        publisher.close()
        subscriber.close()
        context.term()


def _zmq_roundtrip(
    *,
    motion_token: Sequence[float],
    left_hand: Sequence[float],
    right_hand: Sequence[float],
    frame_index: int,
    timeout_seconds: float,
    action_frames: Sequence[Sequence[float]] | None = None,
    control_hz: float = 50.0,
) -> dict[str, Any] | list[dict[str, Any]]:
    import numpy as np  # type: ignore
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # type: ignore
        build_command_message,
        pack_pose_message,
    )

    planner_start, stream = _controller_mode_command_messages(build_command_message)
    if action_frames is not None and len(action_frames) > 1:
        frames = [
            _finite_vector(
                frame,
                size=ACTION_DIM,
                name=f"unitree_g1_sonic_transport_frame_{index}",
            )
            for index, frame in enumerate(action_frames)
        ]
        tokens = [frame[:MOTION_TOKEN_DIM] for frame in frames]
        left_targets = [
            frame[MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM] for frame in frames
        ]
        right_targets = [frame[MOTION_TOKEN_DIM + HAND_DIM :] for frame in frames]
        poses = [
            pack_pose_message(
                {
                    "token_state": np.asarray(token, dtype=np.float32).reshape(1, -1),
                    "frame_index": np.asarray([frame_index + index], dtype=np.int64),
                    "left_hand_joints": np.asarray(
                        left_targets[index], dtype=np.float32
                    ).reshape(1, -1),
                    "right_hand_joints": np.asarray(
                        right_targets[index], dtype=np.float32
                    ).reshape(1, -1),
                },
                topic="pose",
                version=4,
            )
            for index, token in enumerate(tokens)
        ]
        return _zmq_pubsub_horizon_roundtrip(
            pose_messages=poses,
            motion_tokens=tokens,
            left_hands=left_targets,
            right_hands=right_targets,
            frame_indices=[frame_index + index for index in range(len(frames))],
            planner_start_command_message=planner_start,
            stream_command_message=stream,
            control_hz=control_hz,
            timeout_seconds=timeout_seconds,
        )

    token = np.asarray(motion_token, dtype=np.float32)
    pose = pack_pose_message(
        {
            "token_state": token.reshape(1, -1),
            "frame_index": np.asarray([frame_index], dtype=np.int64),
            "left_hand_joints": np.asarray(left_hand, dtype=np.float32).reshape(1, -1),
            "right_hand_joints": np.asarray(right_hand, dtype=np.float32).reshape(1, -1),
        },
        topic="pose",
        version=4,
    )
    return _zmq_pubsub_roundtrip(
        pose_message=pose,
        planner_start_command_message=planner_start,
        stream_command_message=stream,
        motion_token=[float(item) for item in token.reshape(-1)],
        timeout_seconds=timeout_seconds,
        left_hand=[float(item) for item in left_hand],
        right_hand=[float(item) for item in right_hand],
    )


def _official_mujoco_fk(
    *, model_path: Path, body_positions: Sequence[float],
    left_hand: Sequence[float], right_hand: Sequence[float]
) -> tuple[list[str], list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply protocol-v4 targets to the pinned model by joint name.

    The model must expose exactly the pinned 43-joint set; each value is
    written to the qpos address of its named joint, never positionally.
    """

    import mujoco  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model_joint_names: list[str] = []
    for index in range(model.njnt):
        if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        model_joint_names.append(str(name) if name else "")
    validate_model_joint_names(model_joint_names)
    body = _finite_vector(body_positions, size=BODY_DIM, name="official_body_q_target")
    left = _finite_vector(left_hand, size=HAND_DIM, name="official_left_hand_target")
    right = _finite_vector(right_hand, size=HAND_DIM, name="official_right_hand_target")
    names = list(PROTOCOL_V4_FULL_JOINT_ORDER)
    positions = body + left + right
    applied: list[dict[str, Any]] = []
    for protocol_index, (joint_name, value) in enumerate(zip(names, positions)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError("official_gear_sonic_mujoco_model_joint_names_missing")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = float(value)
        applied.append(
            {
                "joint_name": joint_name,
                "protocol_index": protocol_index,
                "model_qpos_address": qpos_address,
                "applied_value": float(value),
            }
        )
    mujoco.mj_forward(model, data)
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if root_body_id < 0:
        raise RuntimeError("official_gear_sonic_fk_pelvis_body_missing")
    root_xyz = [float(value) for value in data.xpos[root_body_id]]
    landmarks: list[dict[str, Any]] = []
    for body_index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_index) or ""
        lower = name.lower()
        if not any(term in lower for term in ("shoulder", "elbow", "wrist", "hand")):
            continue
        xyz = data.xpos[body_index]
        landmarks.append(
            {
                "name": name,
                "landmark_id": name,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "model_xyz": [float(value) for value in xyz],
                "model_root_relative_xyz": [
                    float(xyz[index]) - root_xyz[index] for index in range(3)
                ],
            }
        )
    if not landmarks:
        raise RuntimeError("official_gear_sonic_fk_landmarks_missing")
    return names, positions, landmarks, applied


def validate_live_isaac_articulation(joint_names: Sequence[str]) -> list[dict[str, Any]]:
    """Hook for validating a live Isaac articulation joint list before use."""

    return build_isaac_dof_mapping(joint_names)


def execute(
    request: Mapping[str, Any], *,
    transport: Callable[..., Any] = _zmq_roundtrip,
    fk_solver: Callable[
        ..., tuple[list[str], list[float], list[dict[str, Any]], list[dict[str, Any]]]
    ] = _official_mujoco_fk,
    isaac_joint_names: Sequence[str] | None = None,
    controller_revision_resolver: Callable[[Path], str] = _pinned_controller_revision,
) -> dict[str, Any]:
    action = dict(request.get("action") or {})
    expected_sha = str(request.get("source_action_sha256") or "")
    if _canonical(action) != expected_sha:
        raise ValueError("official_gear_sonic_request_action_sha256_mismatch")
    action_frames, action_sequence_contract = _validated_action_frames(action)
    protocol_frames = [_protocol_v4_action_frame(frame) for frame in action_frames]
    control_hz = float(action_sequence_contract["control_hz"])
    explicit_horizon = isinstance(action.get("controller_action"), Mapping)
    controller_frame_start = controller_frame_sequence_start(
        outer_step_index=int(request.get("step_index") or 0),
        source_horizon_frame_count=int(
            action_sequence_contract["source_horizon_frame_count"]
        ),
        explicit_horizon=explicit_horizon,
    )
    first = protocol_frames[0]
    transport_started = time.monotonic()
    transport_result = transport(
        motion_token=first[:MOTION_TOKEN_DIM],
        left_hand=first[MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM],
        right_hand=first[MOTION_TOKEN_DIM + HAND_DIM :],
        frame_index=controller_frame_start,
        timeout_seconds=120.0,
        **(
            {"action_frames": protocol_frames, "control_hz": control_hz}
            if len(action_frames) > 1
            else {}
        ),
    )
    transport_elapsed_seconds = time.monotonic() - transport_started
    if len(action_frames) == 1:
        if not isinstance(transport_result, Mapping):
            raise RuntimeError("official_gear_sonic_controller_state_missing")
        states = [dict(transport_result)]
    else:
        if (
            isinstance(transport_result, (str, bytes, Mapping))
            or not isinstance(transport_result, Sequence)
        ):
            raise RuntimeError(
                "official_gear_sonic_controller_state_sequence_missing"
            )
        states = [dict(state) for state in transport_result if isinstance(state, Mapping)]
        if len(states) != len(action_frames):
            raise RuntimeError(
                "official_gear_sonic_controller_state_sequence_count_mismatch"
            )
    root = Path(os.getenv(ROOT_ENV, DEFAULT_ROOT)).resolve()
    model = Path(os.getenv(MODEL_ENV, DEFAULT_MODEL)).resolve()
    if root.name != "wbc" or not (root / "gear_sonic_deploy").is_dir():
        raise RuntimeError("official_gear_sonic_repository_missing")
    if not model.is_file() or root not in model.parents:
        raise RuntimeError("official_gear_sonic_robot_model_missing_or_outside_repository")
    controller_revision = controller_revision_resolver(root)
    joint_order = pinned_controller_joint_order(controller_revision)
    camera_projection_context = request.get("camera_projection_context")
    camera_projection_context_sha256 = None
    camera_source_frame_sha256 = None
    cross_simulator_registration = None
    validated_context: dict[str, Any] | None = None
    if isinstance(camera_projection_context, Mapping):
        validated_context = _validated_live_projection_context(
            camera_projection_context
        )
        camera_source_frame_sha256 = str(
            dict(validated_context.get("source_frame_artifact") or {}).get("sha256")
            or ""
        ).lower()
        registration = dict(
            validated_context.get("standing_cross_simulator_registration") or {}
        )
        standing_names = [
            str(name) for name in registration.get("standing_joint_names") or []
        ]
        if standing_names != list(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise ValueError(
                "official_gear_sonic_registration_standing_joint_order_invalid"
            )
        standing_positions = _finite_vector(
            registration.get("standing_joint_positions"),
            size=len(PROTOCOL_V4_FULL_JOINT_ORDER),
            name="official_registration_standing_joint_positions",
        )
        _, _, standing_landmarks, _ = fk_solver(
            model_path=model,
            body_positions=standing_positions[:BODY_DIM],
            left_hand=standing_positions[BODY_DIM : BODY_DIM + HAND_DIM],
            right_hand=standing_positions[BODY_DIM + HAND_DIM :],
        )
        cross_simulator_registration = _verify_standing_cross_simulator_registration(
            context=validated_context,
            standing_landmarks=standing_landmarks,
        )
    isaac_dof_mapping = (
        build_isaac_dof_mapping(isaac_joint_names)
        if isaac_joint_names is not None
        else None
    )
    controller_fk_sequence: list[dict[str, Any]] = []
    for horizon_index, (frame, protocol_frame, state) in enumerate(
        zip(action_frames, protocol_frames, states)
    ):
        motion = protocol_frame[:MOTION_TOKEN_DIM]
        left = protocol_frame[MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM]
        right = protocol_frame[MOTION_TOKEN_DIM + HAND_DIM :]
        controller_motion = _finite_vector(
            state.get("token_state"),
            size=MOTION_TOKEN_DIM,
            name=f"official_motion_token_state_{horizon_index}",
        )
        controller_left = _finite_vector(
            state.get("last_left_hand_action"),
            size=HAND_DIM,
            name=f"official_left_hand_target_{horizon_index}",
        )
        controller_right = _finite_vector(
            state.get("last_right_hand_action"),
            size=HAND_DIM,
            name=f"official_right_hand_target_{horizon_index}",
        )
        if any(abs(a - b) > 1e-6 for a, b in zip(motion, controller_motion)):
            raise RuntimeError(
                "official_gear_sonic_controller_motion_echo_mismatch:"
                f"frame_{horizon_index}"
            )
        for side, sent, echoed in (
            ("left", left, controller_left),
            ("right", right, controller_right),
        ):
            if any(abs(a - b) > 1e-6 for a, b in zip(sent, echoed)):
                suffix = f":frame_{horizon_index}" if len(action_frames) > 1 else ""
                raise RuntimeError(
                    f"official_gear_sonic_controller_hand_echo_mismatch:{side}{suffix}"
                )
        body_target = _finite_vector(
            state.get("body_q_target"),
            size=BODY_DIM,
            name=f"official_body_q_target_{horizon_index}",
        )
        names, positions, landmarks, applied_dof_mapping = fk_solver(
            model_path=model,
            body_positions=body_target,
            left_hand=controller_left,
            right_hand=controller_right,
        )
        if validated_context is not None:
            landmarks, camera_projection_context_sha256 = _project_fk_landmarks(
                landmarks,
                validated_context,
            )
        proprioceptive_state = {
            "body_q_measured": state.get("body_q_measured"),
            "base_quat_measured": state.get("base_quat_measured"),
            "official_controller_protocol": 4,
        }
        state_timestamp = str(state.get("ros_timestamp") or time.time_ns())
        controller_state_evidence = {
            "token_state": controller_motion,
            "body_q_target": body_target,
            "body_q_measured": proprioceptive_state["body_q_measured"],
            "left_hand_action": controller_left,
            "right_hand_action": controller_right,
            "base_quat_measured": proprioceptive_state["base_quat_measured"],
            "state_timestamp": state_timestamp,
            "controller_frame_match_mode": state.get(
                "_blueprint_controller_frame_match_mode"
            ),
            "controller_state_queue_drain_count": state.get(
                "_blueprint_controller_state_queue_drain_count"
            ),
        }
        send_offset = state.get("_blueprint_command_send_offset_seconds")
        controller_fk_sequence.append(
            {
                "horizon_frame_index": horizon_index,
                "controller_frame_index": controller_frame_start + horizon_index,
                "controller_reported_frame_index": state.get("frame_index"),
                "controller_frame_match_mode": state.get(
                    "_blueprint_controller_frame_match_mode"
                ),
                "controller_state_queue_drain_count": state.get(
                    "_blueprint_controller_state_queue_drain_count"
                ),
                "source_action_frame_sha256": _canonical(frame),
                "controller_state_sha256": _canonical(controller_state_evidence),
                "command_send_offset_seconds": (
                    float(send_offset) if send_offset is not None else None
                ),
                "joint_positions": positions,
                "joint_names": names,
                "applied_dof_mapping": applied_dof_mapping,
                "landmarks": landmarks,
                "proprioceptive_state": proprioceptive_state,
                "state_timestamp": state_timestamp,
            }
        )
    final_frame = controller_fk_sequence[-1]
    controller_fk_sequence_sha256 = _canonical(controller_fk_sequence)
    controller_state_sequence_sha256 = _canonical(
        [row["controller_state_sha256"] for row in controller_fk_sequence]
    )
    command_send_offsets = [
        row["command_send_offset_seconds"] for row in controller_fk_sequence
    ]
    send_timing_measured = all(value is not None for value in command_send_offsets)
    measured_send_intervals: list[float] = []
    if send_timing_measured:
        measured_offsets = [float(value) for value in command_send_offsets]
        measured_send_intervals = [
            measured_offsets[index] - measured_offsets[index - 1]
            for index in range(1, len(measured_offsets))
        ]
        if any(interval + 1e-6 < 1.0 / control_hz for interval in measured_send_intervals):
            raise RuntimeError(
                "official_gear_sonic_controller_horizon_sent_faster_than_declared"
            )
    execution_contract = {
        "schema_version": "gear_sonic_controller_horizon_execution.v1",
        "execution_mode": action_sequence_contract["execution_mode"],
        "controller_session_count": 1,
        "execution_frame_count": len(action_frames),
        "source_horizon_frame_count": int(
            action_sequence_contract["source_horizon_frame_count"]
        ),
        "frame_dimension": ACTION_DIM,
        "control_hz": control_hz,
        "sample_period_seconds": 1.0 / control_hz,
        "declared_execution_duration_seconds": len(action_frames) / control_hz,
        "transport_elapsed_seconds": transport_elapsed_seconds,
        "command_send_timing_measured": send_timing_measured,
        "command_send_offsets_seconds": command_send_offsets,
        "measured_send_intervals_seconds": measured_send_intervals,
        "never_faster_than_declared_control_hz": (
            True if send_timing_measured else None
        ),
        "input_action_frames_sha256": action_sequence_contract["frames_sha256"],
        "protocol_v4_action_frames_sha256": _canonical(protocol_frames),
        "sonic_to_protocol_v4_hand_indices": list(
            SONIC_TO_PROTOCOL_V4_HAND_INDICES
        ),
        "source_action_frames_sha256": action_sequence_contract[
            "source_frames_sha256"
        ],
        "controller_state_sequence_sha256": controller_state_sequence_sha256,
        "controller_fk_sequence_sha256": controller_fk_sequence_sha256,
        "final_controller_fk_frame_sha256": _canonical(final_frame),
    }
    return {
        "status": "completed",
        "runtime_result_id": f"gear-sonic-zmq-{uuid.uuid4().hex}",
        "source_action_sha256": expected_sha,
        "landmarks": final_frame["landmarks"],
        "camera_projection_context_sha256": camera_projection_context_sha256,
        "camera_source_frame_sha256": camera_source_frame_sha256,
        "cross_simulator_registration": cross_simulator_registration,
        "joint_positions": final_frame["joint_positions"],
        "joint_names": final_frame["joint_names"],
        "joint_order_schema_version": joint_order["schema_version"],
        "mapping_digest": joint_order["mapping_digest"],
        "controller_revision": controller_revision,
        "mapping_source": joint_order["mapping_source"],
        "robot_model_sha256": _sha256_file(model),
        "applied_dof_mapping": final_frame["applied_dof_mapping"],
        "isaac_dof_mapping": isaac_dof_mapping,
        "proprioceptive_state": final_frame["proprioceptive_state"],
        "state_timestamp": final_frame["state_timestamp"],
        "controller_fk_sequence": controller_fk_sequence,
        "controller_fk_sequence_sha256": controller_fk_sequence_sha256,
        "execution_contract": execution_contract,
    }


def main() -> int:
    request = json.loads(Path(os.environ[INPUT_ENV]).read_text(encoding="utf-8"))
    result = execute(request)
    Path(os.environ[OUTPUT_ENV]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
