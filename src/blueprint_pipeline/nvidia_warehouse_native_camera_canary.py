"""Native Isaac camera canary for the pinned NVIDIA Warehouse workcell."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import posixpath
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import write_json
from .nvidia_warehouse_workcell import CANARY_SPEC_SCHEMA_VERSION
from .policy_ranking_thesis import canonical_sha256, file_sha256


RESULT_SCHEMA_VERSION = "nvidia_warehouse_native_camera_canary_result.v1"
MIN_NONBLANK_SPATIAL_STD = 4.0
MIN_WRIST_WORLD_DISPLACEMENT_M = 0.001
# Isaac Sim 6's unified pose view returns float-backed translation and
# quaternion tensors.  Reconstructing a parent-local 4x4 transform from those
# values can therefore differ by a few float32 ULPs even when the authored
# rigid mount is unchanged.  One micrometre / 1e-6 matrix-element precision is
# still far below any meaningful camera-mount slip while remaining measurable
# through that public API.
MAX_WRIST_LOCAL_TRANSFORM_DELTA = 1e-6
MAX_KINEMATIC_JOINT_POSITION_ERROR_RAD = 0.02
# A projected centroid can be fully occluded. Require at least an 8x8-pixel
# render-derived semantic footprint before treating a task entity as visible.
# This camera-validity threshold is frozen independently of policy rankings.
MIN_REQUIRED_ENTITY_VISIBLE_PIXELS = 64
# V30's returned wrist RGB proved that the initial optical ray was blocked by
# workcell/crate geometry at the object-height mount. Raise the one-time rigid
# mount above that obstruction; this is fixed before any policy/WAM execution.
WRIST_CAMERA_WORLD_CLEARANCE_M = 0.25
REQUIRED_VIEWS = ("external", "wrist")
SUPPORTED_OPTIONAL_VIEWS = ("external_2",)


def _required_views_from_spec(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve a prospectively declared camera set without changing v1 defaults."""

    declared = spec.get("required_views")
    if declared is None:
        return REQUIRED_VIEWS
    if not isinstance(declared, list) or not all(isinstance(view, str) for view in declared):
        raise ValueError("native_camera_required_views_invalid")
    views = tuple(declared)
    if len(views) != len(set(views)):
        raise ValueError("native_camera_required_views_duplicate")
    if not all(view in views for view in REQUIRED_VIEWS):
        raise ValueError("native_camera_required_views_baseline_missing")
    allowed = set(REQUIRED_VIEWS + SUPPORTED_OPTIONAL_VIEWS)
    if any(view not in allowed for view in views):
        raise ValueError("native_camera_required_views_unsupported")
    cameras = spec.get("cameras")
    if not isinstance(cameras, Mapping) or any(view not in cameras for view in views):
        raise ValueError("native_camera_required_view_spec_missing")
    if "external_2" in views:
        external = cameras.get("external")
        external_2 = cameras.get("external_2")
        if not isinstance(external, Mapping) or not isinstance(external_2, Mapping):
            raise ValueError("native_camera_external_pair_spec_invalid")
        first_pose = (external.get("position_m"), external.get("look_at_m"))
        second_pose = (external_2.get("position_m"), external_2.get("look_at_m"))
        if first_pose == second_pose:
            raise ValueError("native_camera_external_pair_pose_not_distinct")
    return views


def _simulation_app_launch_config() -> dict[str, Any]:
    """Return a fresh launcher config that lets the evidence wrapper resume."""

    return {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": 640,
        "height": 480,
        # Isaac Sim 6 defaults this to true, which terminates the process from
        # SimulationApp.close() before the outer worker can archive and upload
        # the camera evidence.
        "fast_shutdown": False,
    }


def import_simulation_app() -> Any:
    """Resolve the Isaac launcher while rejecting non-callable API shims."""

    try:
        from isaacsim import SimulationApp

        if callable(SimulationApp):
            return SimulationApp
    except Exception:
        pass
    from omni.isaac.kit import SimulationApp

    if not callable(SimulationApp):
        raise ImportError("isaac_simulation_app_not_callable")
    return SimulationApp


def _add_prim_semantic_label(prim: Any, semantic_label: str) -> str:
    """Author one class label through the first supported Isaac semantics API."""

    candidates = (
        (
            "isaacsim.core.experimental.utils.semantics",
            "add_labels",
            "isaacsim_core_experimental_labels_api",
        ),
        (
            "isaacsim.core.utils.semantics",
            "add_update_semantics",
            "isaacsim_core_legacy_semantics_api",
        ),
        (
            "semantics.schema_editor",
            "add_prim_semantics",
            "semantics_schema_editor",
        ),
        (
            "omni.isaac.core.utils.semantics",
            "add_update_semantics",
            "omni_isaac_legacy_semantics_api",
        ),
    )
    import_failures: list[str] = []
    for module_name, function_name, api_name in candidates:
        try:
            function = getattr(importlib.import_module(module_name), function_name)
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            import_failures.append(f"{module_name}:{type(exc).__name__}")
            continue
        if function_name == "add_labels":
            function(prim, labels=[semantic_label], taxonomy="class")
        else:
            try:
                function(
                    prim,
                    semantic_label=semantic_label,
                    type_label="class",
                )
            except TypeError:
                function(prim, semantic_label, "class")
        return api_name
    raise ImportError("native_semantics_api_unavailable:" + ",".join(import_failures))


def _author_renderable_semantic_label_tree(
    *,
    root_prim: Any,
    semantic_label: str,
    prim_range: Callable[[Any], Any],
    is_renderable: Callable[[Any], bool],
    add_label: Callable[[Any, str], str] = _add_prim_semantic_label,
) -> dict[str, Any]:
    """Label every rendered descendant so RTX pixels inherit the entity class."""

    targets = [prim for prim in prim_range(root_prim) if is_renderable(prim)]
    if not targets:
        targets = [root_prim]
    apis = [add_label(prim, semantic_label) for prim in targets]
    return {
        "root_label": str(semantic_label),
        "renderable_prim_count": len(targets),
        "api_names": sorted(set(apis)),
        "root_fallback_used": targets == [root_prim],
    }


def _camera_quaternion_wxyz(forward: Any, up: Any) -> np.ndarray:
    """Quaternion for a USD camera whose local forward axis is negative Z."""

    forward_array = np.asarray(forward, dtype=float)
    up_array = np.asarray(up, dtype=float)
    forward_array /= np.linalg.norm(forward_array)
    right = np.cross(forward_array, up_array)
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, forward_array)
    corrected_up /= np.linalg.norm(corrected_up)
    rotation = np.column_stack((right, corrected_up, -forward_array))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        final_index = (index + 2) % 3
        scale = (
            math.sqrt(
                1.0
                + rotation[index, index]
                - rotation[next_index, next_index]
                - rotation[final_index, final_index]
            )
            * 2.0
        )
        xyz = np.zeros(3)
        xyz[index] = 0.25 * scale
        xyz[next_index] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        xyz[final_index] = (rotation[final_index, index] + rotation[index, final_index]) / scale
        w = (rotation[final_index, next_index] - rotation[next_index, final_index]) / scale
        quaternion = np.concatenate(([w], xyz))
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _project_world_points(
    *, camera_to_world: Any, points: Mapping[str, Any], width: int, height: int, vfov_deg: float
) -> dict[str, bool]:
    """Project named world points with the same negative-Z USD camera convention."""

    matrix = np.asarray(camera_to_world, dtype=float).reshape(4, 4)
    world_to_camera = np.linalg.inv(matrix)
    focal = float(height) / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    result: dict[str, bool] = {}
    for name, value in points.items():
        homogeneous = np.concatenate((np.asarray(value, dtype=float), [1.0]))
        camera = homogeneous @ world_to_camera
        depth = -float(camera[2])
        if depth <= 1e-9:
            result[str(name)] = False
            continue
        u = focal * float(camera[0]) / depth + (float(width) - 1.0) / 2.0
        v = -focal * float(camera[1]) / depth + (float(height) - 1.0) / 2.0
        result[str(name)] = 0.0 <= u < width and 0.0 <= v < height
    return result


def _project_required_external_entities(
    *,
    camera_to_world: Any,
    task_points: Mapping[str, Any],
    franka_link_points: Mapping[str, Any],
    width: int,
    height: int,
    vfov_deg: float,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Require task points and at least one Franka link origin in the image."""

    if not franka_link_points:
        raise ValueError("native_warehouse_franka_link_projection_points_missing")
    task_projection = _project_world_points(
        camera_to_world=camera_to_world,
        points=task_points,
        width=width,
        height=height,
        vfov_deg=vfov_deg,
    )
    link_projection = _project_world_points(
        camera_to_world=camera_to_world,
        points=franka_link_points,
        width=width,
        height=height,
        vfov_deg=vfov_deg,
    )
    required = {
        "franka": any(link_projection.values()),
        **task_projection,
    }
    return required, {
        "franka_visibility_rule": "at_least_one_articulation_link_origin_in_frame",
        "franka_link_origins_projected_in_frame": link_projection,
    }


def _semantic_entity_visibility(
    *, semantic_frame: Any, entity_labels: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Count render-derived semantic pixels for required entities, fail closed."""

    frame = semantic_frame if isinstance(semantic_frame, Mapping) else {}
    payload = frame.get("data")
    info_value = frame.get("info")
    info = info_value if isinstance(info_value, Mapping) else {}
    labels_value = info.get("idToLabels")
    labels = labels_value if isinstance(labels_value, Mapping) else {}
    try:
        pixels = np.asarray(payload)
        if pixels.ndim == 3 and pixels.shape[-1] == 1:
            pixels = pixels[:, :, 0]
        if pixels.ndim != 2 or pixels.size == 0 or not np.issubdtype(pixels.dtype, np.integer):
            raise ValueError
    except (TypeError, ValueError):
        pixels = np.asarray([], dtype=np.uint32)

    observed_labels = {
        str(raw_id): {
            str(key): str(value) for key, value in raw_labels.items() if isinstance(key, str)
        }
        for raw_id, raw_labels in list(labels.items())[:64]
        if isinstance(raw_labels, Mapping)
    }
    result: dict[str, dict[str, Any]] = {}
    for entity_id, expected_label in entity_labels.items():
        matching_ids: list[int] = []
        for raw_id, raw_labels in labels.items():
            label_map = raw_labels if isinstance(raw_labels, Mapping) else {}
            if str(label_map.get("class") or "") != str(expected_label):
                continue
            try:
                matching_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue
        visible_pixels = (
            int(np.isin(pixels, matching_ids).sum()) if pixels.size and matching_ids else 0
        )
        result[str(entity_id)] = {
            "semantic_class": str(expected_label),
            "semantic_ids": sorted(set(matching_ids)),
            "visible_pixel_count": visible_pixels,
            "minimum_visible_pixel_count": MIN_REQUIRED_ENTITY_VISIBLE_PIXELS,
            "visible": visible_pixels >= MIN_REQUIRED_ENTITY_VISIBLE_PIXELS,
            "render_derived": True,
            "observed_id_to_labels": observed_labels,
        }
    return result


def _matrix_array(matrix: Any) -> np.ndarray:
    return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)])


def _backend_array_to_numpy(value: Any) -> np.ndarray:
    """Copy a NumPy/Torch backend value to a CPU NumPy array."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value)


def _camera_sensor_annotator_frame(*, sensor: Any, annotator: str) -> dict[str, Any]:
    """Normalize Isaac 6's experimental RTX sensor output for assessment."""

    data, info_value = sensor.get_data(annotator)
    if data is None:
        raise ValueError(f"native_camera_annotator_data_unavailable:{annotator}")
    info = dict(info_value) if isinstance(info_value, Mapping) else {}
    return {
        "data": _backend_array_to_numpy(data),
        "info": info,
    }


def _world_pose_matrix_from_backend_pose(position: Any, orientation_wxyz: Any) -> np.ndarray:
    """Build the row-vector local-to-world matrix used by USD projection helpers."""

    translation = _backend_array_to_numpy(position).astype(float).reshape(-1)
    quaternion = _backend_array_to_numpy(orientation_wxyz).astype(float).reshape(-1)
    if (
        translation.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(translation).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError("native_fabric_world_pose_invalid")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("native_fabric_world_orientation_invalid")
    w, x, y, z = quaternion / norm
    column_rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = column_rotation.T
    matrix[3, :3] = translation
    return matrix


def _quaternion_wxyz_from_world_pose_matrix(matrix_value: Any) -> np.ndarray:
    """Extract a scalar-first quaternion from a row-vector world-pose matrix."""

    matrix = np.asarray(matrix_value, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3].T
    if not np.isfinite(rotation).all():
        raise ValueError("native_world_pose_rotation_invalid")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        final_index = (index + 2) % 3
        scale = (
            math.sqrt(
                1.0
                + rotation[index, index]
                - rotation[next_index, next_index]
                - rotation[final_index, final_index]
            )
            * 2.0
        )
        xyz = np.zeros(3)
        xyz[index] = 0.25 * scale
        xyz[next_index] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        xyz[final_index] = (rotation[final_index, index] + rotation[index, final_index]) / scale
        w = (rotation[final_index, next_index] - rotation[next_index, final_index]) / scale
        quaternion = np.concatenate(([w], xyz))
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("native_world_pose_orientation_invalid")
    return quaternion / norm


def _articulation_link_world_pose_matrices(
    *,
    robot: Any,
    simulation_view: Any,
) -> dict[str, np.ndarray]:
    """Read all live link poses from the articulation physics tensor view."""

    update_kinematics = getattr(simulation_view, "update_articulations_kinematic", None)
    if not callable(update_kinematics):
        raise ValueError("native_articulation_kinematic_update_api_missing")
    articulation = getattr(robot, "_articulation_view", None)
    physics_view = getattr(articulation, "_physics_view", None)
    get_link_transforms = getattr(physics_view, "get_link_transforms", None)
    if not callable(get_link_transforms):
        raise ValueError("native_articulation_link_transform_api_missing")
    names_value = getattr(articulation, "body_names", None)
    if not isinstance(names_value, (list, tuple)) or not names_value:
        raise ValueError("native_articulation_link_names_missing")
    names = [str(name) for name in names_value]
    try:
        update_kinematics()
        transforms = _backend_array_to_numpy(get_link_transforms()).astype(float)
    except Exception as exc:
        raise ValueError(
            "native_articulation_link_transform_query_failed:" + type(exc).__name__
        ) from exc
    if transforms.size != len(names) * 7:
        raise ValueError("native_articulation_link_transform_cardinality_invalid")
    transforms = transforms.reshape(1, len(names), 7)
    if not np.isfinite(transforms).all():
        raise ValueError("native_articulation_link_transform_invalid")
    return {
        name: _world_pose_matrix_from_backend_pose(
            transforms[0, index, :3],
            [
                transforms[0, index, 6],
                transforms[0, index, 3],
                transforms[0, index, 4],
                transforms[0, index, 5],
            ],
        )
        for index, name in enumerate(names)
    }


def _synchronize_camera_to_rigid_link(
    *,
    pose_view: Any,
    parent_to_world: Any,
    mount_translation_parent: Any,
    mount_orientation_parent_wxyz: Any,
) -> np.ndarray:
    """Apply a fixed parent-local mount to a world-space camera explicitly."""

    mount_to_parent = _world_pose_matrix_from_backend_pose(
        mount_translation_parent,
        mount_orientation_parent_wxyz,
    )
    camera_to_world = mount_to_parent @ np.asarray(parent_to_world, dtype=float).reshape(4, 4)
    set_world_poses = getattr(pose_view, "set_world_poses", None)
    if not callable(set_world_poses):
        raise ValueError("native_wrist_camera_unified_pose_write_api_missing")
    try:
        set_world_poses(
            positions=np.asarray([camera_to_world[3, :3]], dtype=float),
            orientations=np.asarray(
                [_quaternion_wxyz_from_world_pose_matrix(camera_to_world)], dtype=float
            ),
        )
    except Exception as exc:
        raise ValueError(
            "native_wrist_camera_unified_pose_write_failed:" + type(exc).__name__
        ) from exc
    return camera_to_world


def _unified_world_pose_matrix(pose_view: Any) -> np.ndarray:
    """Read one pose through Isaac 6's unified USD/USDRT/Fabric API."""

    get_world_poses = getattr(pose_view, "get_world_poses", None)
    if not callable(get_world_poses):
        raise ValueError("native_unified_world_pose_api_missing")
    try:
        positions, orientations = get_world_poses()
    except Exception as exc:
        raise ValueError("native_unified_world_pose_query_failed:" + type(exc).__name__) from exc
    positions_array = _backend_array_to_numpy(positions).astype(float).reshape(-1, 3)
    orientations_array = _backend_array_to_numpy(orientations).astype(float).reshape(-1, 4)
    if positions_array.shape[0] != 1 or orientations_array.shape[0] != 1:
        raise ValueError("native_unified_world_pose_cardinality_invalid")
    return _world_pose_matrix_from_backend_pose(positions_array[0], orientations_array[0])


def _camera_pose_backend_congruence(
    *,
    requested_camera_to_world: Any,
    authoring_camera_to_world: Any,
    usd_camera_to_world: Any,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare the requested camera pose with both runtime and USD truth."""

    matrices = {
        "requested": np.asarray(requested_camera_to_world, dtype=float).reshape(4, 4),
        "authoring": np.asarray(authoring_camera_to_world, dtype=float).reshape(4, 4),
        "usd": np.asarray(usd_camera_to_world, dtype=float).reshape(4, 4),
    }
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("native_wrist_camera_pose_backend_matrix_invalid")
    deltas = {
        "requested_to_authoring_max_abs": float(
            np.max(np.abs(matrices["requested"] - matrices["authoring"]))
        ),
        "requested_to_usd_max_abs": float(np.max(np.abs(matrices["requested"] - matrices["usd"]))),
        "authoring_to_usd_max_abs": float(np.max(np.abs(matrices["authoring"] - matrices["usd"]))),
    }
    threshold = float(tolerance)
    congruent = all(value <= threshold for value in deltas.values())
    return {
        **deltas,
        "tolerance_max_abs": threshold,
        "congruent": congruent,
    }


def _render_world_without_physics_advance(world: Any) -> None:
    """Update articulation/Fabric transforms and render without stepping physics."""

    render = getattr(world, "render", None)
    if not callable(render):
        raise ValueError("native_franka_zero_time_scene_update_api_missing")
    device = str(getattr(world, "device", "") or "").lower()
    if "cuda" not in device:
        raise ValueError("native_franka_zero_time_scene_update_cuda_backend_required")
    if getattr(world, "physics_sim_view", None) is None:
        raise ValueError("native_franka_zero_time_scene_update_physics_view_missing")
    is_playing = getattr(world, "is_playing", None)
    if not callable(is_playing) or not bool(is_playing()):
        raise ValueError("native_franka_zero_time_scene_update_simulation_not_playing")
    before = int(world.current_time_step_index)
    try:
        # Isaac Sim 6.0 World.render() calls the CUDA physics view's
        # update_articulations_kinematic() before its zero-time app update.
        render()
    except Exception as exc:
        raise ValueError(
            "native_franka_zero_time_scene_update_failed:" + type(exc).__name__
        ) from exc
    if int(world.current_time_step_index) != before:
        raise ValueError("native_franka_zero_time_scene_update_advanced_physics")


def _apply_and_measure_render_only_joint_pose(
    *,
    robot: Any,
    joint_positions: Any,
    phase: str,
    render: Callable[[], None],
    render_count: int,
) -> dict[str, Any]:
    """Teleport one joint state, update/render without physics, and measure it."""

    requested = np.asarray(joint_positions, dtype=float).reshape(-1)
    zeros = np.zeros_like(requested)
    backend_utils = getattr(robot, "_backend_utils", None)
    backend_convert = getattr(backend_utils, "convert", None)
    backend_device = getattr(robot, "_device", None)
    if callable(backend_convert):
        requested_for_backend = backend_convert(requested, backend_device)
        zeros_for_backend = backend_convert(zeros, backend_device)
    else:
        requested_for_backend = requested
        zeros_for_backend = zeros
    required = {
        "set_joint_positions": getattr(robot, "set_joint_positions", None),
        "set_joint_velocities": getattr(robot, "set_joint_velocities", None),
        "get_joint_positions": getattr(robot, "get_joint_positions", None),
    }
    missing = sorted(name for name, method in required.items() if not callable(method))
    if missing:
        raise ValueError("native_franka_render_only_joint_state_api_missing:" + ",".join(missing))
    try:
        required["set_joint_positions"](requested_for_backend)
        required["set_joint_velocities"](zeros_for_backend)
        for _ in range(int(render_count)):
            render()
    except Exception as exc:
        raise ValueError(
            "native_franka_render_only_joint_state_failed:" + type(exc).__name__
        ) from exc
    measured = _backend_array_to_numpy(required["get_joint_positions"]()).astype(float).reshape(-1)
    if measured.shape != requested.shape or not np.isfinite(measured).all():
        raise ValueError("native_franka_render_only_joint_measurement_invalid")
    return {
        "phase": str(phase),
        "mode": "render_only_kinematic_joint_state_transition",
        "requested_joint_positions_rad": requested.tolist(),
        "measured_joint_positions_rad": measured.tolist(),
        "max_abs_position_error_rad": float(np.max(np.abs(measured - requested))),
        "zero_velocity_state_applied": True,
        "joint_position_state_applied": True,
        "zero_time_scene_update_requested": True,
        "render_count": int(render_count),
        "physics_steps_requested": 0,
    }


def _rigid_wrist_mount_from_initial_task_framing(
    *,
    parent_to_world: Any,
    mount_translation_parent: Any,
    target_world_points: Mapping[str, Any],
    world_up: Any = (0.0, 0.0, 1.0),
    camera_eye_world_offset: Any = (0.0, 0.0, 0.0),
    target_relative_camera_eye_world_offset: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate one rigid parent-local wrist gaze from initial task geometry."""

    matrix = np.asarray(parent_to_world, dtype=float).reshape(4, 4)
    mount = np.asarray(mount_translation_parent, dtype=float).reshape(3)
    up_world = np.asarray(world_up, dtype=float).reshape(3)
    eye_offset_world = np.asarray(camera_eye_world_offset, dtype=float).reshape(3)
    target_eye_offset_world = (
        None
        if target_relative_camera_eye_world_offset is None
        else np.asarray(target_relative_camera_eye_world_offset, dtype=float).reshape(3)
    )
    points = {
        str(name): np.asarray(value, dtype=float).reshape(3)
        for name, value in target_world_points.items()
    }
    if (
        not points
        or not np.isfinite(matrix).all()
        or not np.isfinite(mount).all()
        or not np.isfinite(up_world).all()
        or not np.isfinite(eye_offset_world).all()
        or (target_eye_offset_world is not None and not np.isfinite(target_eye_offset_world).all())
    ):
        raise ValueError("native_wrist_mount_calibration_input_invalid")
    target_world = np.mean(np.stack(list(points.values())), axis=0)
    world_to_parent = np.linalg.inv(matrix)
    base_eye_world = (np.concatenate((mount, [1.0])) @ matrix)[:3]
    if target_eye_offset_world is None:
        eye_world = base_eye_world + eye_offset_world
        eye_placement_mode = "base_mount_world_offset"
    else:
        eye_world = target_world + target_eye_offset_world
        eye_placement_mode = "target_relative_world_offset"
    resolved_mount = (np.concatenate((eye_world, [1.0])) @ world_to_parent)[:3]
    target_parent = np.concatenate((target_world, [1.0])) @ world_to_parent
    forward_parent = target_parent[:3] - resolved_mount
    forward_norm = float(np.linalg.norm(forward_parent))
    if not math.isfinite(forward_norm) or forward_norm <= 1e-9:
        raise ValueError("native_wrist_mount_calibration_target_degenerate")
    forward_parent /= forward_norm
    up_parent = (np.concatenate((up_world, [0.0])) @ world_to_parent)[:3]
    up_norm = float(np.linalg.norm(up_parent))
    if not math.isfinite(up_norm) or up_norm <= 1e-9:
        raise ValueError("native_wrist_mount_calibration_up_degenerate")
    up_parent /= up_norm
    if float(np.linalg.norm(np.cross(forward_parent, up_parent))) <= 1e-6:
        candidates = np.eye(3)
        up_parent = max(
            candidates,
            key=lambda candidate: float(np.linalg.norm(np.cross(forward_parent, candidate))),
        )
    quaternion = _camera_quaternion_wxyz(forward_parent, up_parent)
    return quaternion, {
        "mode": "one_time_initial_task_framing_rigid_parent_local_mount",
        "target_entity_ids": sorted(points),
        "target_centroid_world_m": target_world.tolist(),
        "base_camera_eye_world_m": base_eye_world.tolist(),
        "camera_eye_world_m": eye_world.tolist(),
        "camera_eye_placement_mode": eye_placement_mode,
        "camera_eye_world_offset_m": eye_offset_world.tolist(),
        "target_relative_camera_eye_world_offset_m": (
            None if target_eye_offset_world is None else target_eye_offset_world.tolist()
        ),
        "resolved_mount_translation_parent_m": resolved_mount.tolist(),
        "mount_forward_parent": forward_parent.tolist(),
        "mount_up_parent": np.asarray(up_parent, dtype=float).tolist(),
        "per_frame_task_reaim_performed": False,
    }


def _wrist_calibration_target_world_points(
    *, calibration: Mapping[str, Any], entity_world_points: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the prospectively specified task entities used for wrist framing."""

    available = {str(key): value for key, value in entity_world_points.items()}
    requested_value = calibration.get("target_entity_ids")
    if not isinstance(requested_value, list) or not requested_value:
        raise ValueError("native_wrist_mount_target_entity_ids_missing")
    requested = [str(value).strip() for value in requested_value]
    if any(not value for value in requested) or len(set(requested)) != len(requested):
        raise ValueError("native_wrist_mount_target_entity_ids_invalid")
    unsupported = sorted(set(requested) - set(available))
    if unsupported:
        raise ValueError(
            "native_wrist_mount_target_entity_ids_unsupported:" + ",".join(unsupported)
        )
    resolved = {entity_id: available[entity_id] for entity_id in requested}
    if any(value is None for value in resolved.values()):
        raise ValueError("native_wrist_mount_target_placement_missing")
    return resolved


def _world_bound_center_xyz(world_bound: Any) -> np.ndarray:
    """Return the finite center of one USD world-space aligned bound."""

    aligned = world_bound.ComputeAlignedBox()
    minimum = np.asarray(aligned.GetMin(), dtype=float).reshape(3)
    maximum = np.asarray(aligned.GetMax(), dtype=float).reshape(3)
    center = (minimum + maximum) / 2.0
    if not np.isfinite(center).all() or np.any(maximum < minimum):
        raise ValueError("native_renderable_world_bound_invalid")
    return center


def _aabb_intersection_report(
    *,
    subject_min_xyz: Any,
    subject_max_xyz: Any,
    obstacle_bounds: Mapping[str, tuple[Any, Any]],
    allowed_enclosing_obstacle_paths: tuple[str, ...] = (),
    minimum_intersection_volume_m3: float = 1e-9,
) -> dict[str, Any]:
    """Report conservative 3D placement intersections without render labels.

    The check deliberately uses world-space aligned bounds.  It can reject a
    valid placement near a concave/open receptacle, but it must never approve a
    target that is wholly embedded in another scene object.  A task that needs
    receptacle containment must declare a separately validated containment
    contract instead of silently bypassing this initial-object check.
    """

    subject_min = np.asarray(subject_min_xyz, dtype=float).reshape(3)
    subject_max = np.asarray(subject_max_xyz, dtype=float).reshape(3)
    if (
        not np.isfinite(subject_min).all()
        or not np.isfinite(subject_max).all()
        or np.any(subject_max <= subject_min)
        or not math.isfinite(float(minimum_intersection_volume_m3))
        or minimum_intersection_volume_m3 < 0.0
    ):
        raise ValueError("native_task_placement_subject_bound_invalid")
    subject_volume = float(np.prod(subject_max - subject_min))
    allowed = {str(path) for path in allowed_enclosing_obstacle_paths}
    if not all(path.startswith("/World/WarehouseWorkcell/") for path in allowed):
        raise ValueError("native_task_placement_allowed_enclosure_path_invalid")
    intersections: list[dict[str, Any]] = []
    allowed_enclosing_intersections: list[dict[str, Any]] = []
    for obstacle_path, bound in sorted(obstacle_bounds.items()):
        if not isinstance(bound, tuple) or len(bound) != 2:
            raise ValueError("native_task_placement_obstacle_bound_invalid")
        obstacle_min = np.asarray(bound[0], dtype=float).reshape(3)
        obstacle_max = np.asarray(bound[1], dtype=float).reshape(3)
        if (
            not np.isfinite(obstacle_min).all()
            or not np.isfinite(obstacle_max).all()
            or np.any(obstacle_max <= obstacle_min)
        ):
            raise ValueError("native_task_placement_obstacle_bound_invalid")
        intersection_extent = np.maximum(
            0.0,
            np.minimum(subject_max, obstacle_max) - np.maximum(subject_min, obstacle_min),
        )
        intersection_volume = float(np.prod(intersection_extent))
        if intersection_volume <= minimum_intersection_volume_m3:
            continue
        overlap_fraction = intersection_volume / subject_volume
        row = {
            "obstacle_prim_path": str(obstacle_path),
            "intersection_extent_m": intersection_extent.tolist(),
            "intersection_volume_m3": intersection_volume,
            "subject_overlap_fraction": overlap_fraction,
        }
        if str(obstacle_path) in allowed and overlap_fraction >= 1.0 - 1e-9:
            allowed_enclosing_intersections.append(row)
        else:
            intersections.append(row)
    return {
        "status": "passed" if not intersections else "failed",
        "method": "world_aligned_aabb_intersection_fail_closed",
        "subject_min_world_m": subject_min.tolist(),
        "subject_max_world_m": subject_max.tolist(),
        "subject_volume_m3": subject_volume,
        "minimum_intersection_volume_m3": float(minimum_intersection_volume_m3),
        "intersections": intersections,
        "allowed_enclosing_obstacle_paths": sorted(allowed),
        "allowed_enclosing_intersections": allowed_enclosing_intersections,
        "rankings_or_policy_outcomes_accessed": False,
    }


def _summarize_required_entity_projections(
    *, view_id: str, projections_by_phase: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    """Apply the frozen per-view projection contract without over-constraining it.

    The external view is the scene-grounding view, so every required entity
    must remain projected in both observations.  The wrist contract requires
    the manipulated task object (the spray can) in the initial observation;
    the commanded wrist frame is separately checked for validity and rigid
    motion, but is not re-aimed at a world-fixed object after the hand moves.
    """

    phases = {
        str(phase): dict(values)
        for phase, values in projections_by_phase.items()
        if isinstance(values, Mapping)
    }
    if view_id == "external":
        names = sorted({str(name) for values in phases.values() for name in values})
        return {
            name: all(bool(phases.get(phase, {}).get(name)) for phase in ("initial", "commanded"))
            for name in names
        }
    if view_id == "wrist":
        return {"spraycan_at_initial_pose": bool(phases.get("initial", {}).get("spraycan"))}
    raise ValueError(f"native_camera_projection_view_invalid:{view_id}")


def _summarize_required_entity_visibility(
    *, view_id: str, visibility_by_phase: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    """Apply the same frozen per-view contract to render-derived visibility."""

    phases = {
        str(phase): dict(values)
        for phase, values in visibility_by_phase.items()
        if isinstance(values, Mapping)
    }

    def visible(phase: str, entity_id: str) -> bool:
        value = phases.get(phase, {}).get(entity_id)
        return bool(value.get("visible")) if isinstance(value, Mapping) else False

    if view_id == "external":
        return {
            entity_id: all(visible(phase, entity_id) for phase in ("initial", "commanded"))
            for entity_id in ("franka", "spraycan", "tray")
        }
    if view_id == "wrist":
        return {"spraycan_at_initial_pose": visible("initial", "spraycan")}
    raise ValueError(f"native_camera_visibility_view_invalid:{view_id}")


def _relocate_layer_asset_path(
    layer_path: Path, source_asset_uri: str, replacement_authored_path: str
) -> int:  # pragma: no cover - exercised inside the pinned Isaac/OpenUSD image
    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        raise ValueError(f"native_warehouse_asset_relocation_layer_missing:{layer_path}")
    replacements = 0

    def relocate(value: str) -> str:
        nonlocal replacements
        if value == source_asset_uri:
            replacements += 1
            return replacement_authored_path
        return value

    UsdUtils.ModifyAssetPaths(layer, relocate)
    if replacements <= 0:
        raise ValueError("native_warehouse_asset_relocation_source_not_authored")
    layer.Save()
    return replacements


def _apply_runtime_asset_relocations(
    *,
    assets_root: Path,
    manifest: Mapping[str, Any],
    layer_relocator: Callable[[Path, str, str], int] = _relocate_layer_asset_path,
) -> dict[str, Any]:
    """Apply hash-bound local mirrors before the workcell composition is loaded."""

    root = assets_root.expanduser().resolve()
    rows = manifest.get("runtime_asset_relocations") or []
    if not isinstance(rows, list):
        raise ValueError("native_warehouse_asset_relocations_invalid")
    authored_replacement_count = 0
    applied: list[dict[str, Any]] = []
    for value in rows:
        if not isinstance(value, Mapping):
            raise ValueError("native_warehouse_asset_relocation_invalid")
        owner_relative = str(value.get("owner_relative_path") or "")
        replacement_relative = str(value.get("replacement_relative_path") or "")
        source_asset_uri = str(value.get("source_asset_uri") or "")
        replacement_authored = str(value.get("replacement_authored_path") or "")
        owner = (root / owner_relative).resolve()
        replacement = (root / replacement_relative).resolve()
        if (
            not owner.is_relative_to(root)
            or not replacement.is_relative_to(root)
            or not owner.is_file()
            or not replacement.is_file()
            or not source_asset_uri
            or not replacement_authored.startswith(("./", "../"))
        ):
            raise ValueError("native_warehouse_asset_relocation_path_invalid")
        expected_authored = posixpath.relpath(
            replacement_relative, posixpath.dirname(owner_relative)
        )
        if not expected_authored.startswith(("./", "../")):
            expected_authored = "./" + expected_authored
        if replacement_authored != expected_authored:
            raise ValueError("native_warehouse_asset_relocation_binding_invalid")
        count = int(layer_relocator(owner, source_asset_uri, replacement_authored))
        if count <= 0:
            raise ValueError("native_warehouse_asset_relocation_not_applied")
        authored_replacement_count += count
        applied.append(
            {
                "owner_relative_path": owner_relative,
                "replacement_relative_path": replacement_relative,
                "authored_replacement_count": count,
            }
        )
    return {
        "relocation_count": len(applied),
        "authored_replacement_count": authored_replacement_count,
        "relocations": applied,
    }


def _load_materialization_manifest(assets_root: Path) -> tuple[Path, dict[str, Any]]:
    """Load the manifest from a materialization root or extracted bundle layout."""

    root = assets_root.expanduser().resolve()
    direct = root / "materialization_manifest.json"
    extracted = root.parent / "materialization_manifest.json"
    candidates = [direct]
    if root.name == "assets":
        candidates.append(extracted)
    manifest_path = next(
        (path for path in candidates if path.is_file() and not path.is_symlink()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError("native_warehouse_materialization_manifest_missing")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("native_warehouse_materialization_manifest_not_object")
    return manifest_path, dict(value)


def isaac_sim_6_backend(
    *, spec: Mapping[str, Any], assets_root: Path, output_dir: Path
) -> dict[str, Any]:  # pragma: no cover - requires the pinned Isaac GPU image
    """Load the selected workcell and render synchronized external/wrist frames."""

    SimulationApp = import_simulation_app()
    simulation_app = SimulationApp(_simulation_app_launch_config())
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        output_dir.mkdir(parents=True, exist_ok=True)
        _manifest_path, manifest = _load_materialization_manifest(assets_root)
        relocation_evidence = _apply_runtime_asset_relocations(
            assets_root=assets_root, manifest=manifest
        )
        # World.render() only refreshes articulation kinematics from the physics
        # view for CUDA-backed worlds in Isaac Sim 6.0.
        world = World(stage_units_in_meters=1.0, backend="torch", device="cuda:0")
        stage = world.stage
        scene = spec["scene"]
        placements = scene["placements"]
        workcell_path = assets_root / str(scene["workcell_usd"])
        spraycan_path = assets_root / str(scene["spraycan_usd"])
        if not workcell_path.is_file() or not spraycan_path.is_file():
            raise FileNotFoundError("native_warehouse_required_usd_missing")

        add_reference_to_stage(str(workcell_path), "/World/WarehouseWorkcell")
        add_reference_to_stage(str(spraycan_path), "/World/Task/SprayCan")
        native_assets = get_assets_root_path() or ""
        franka_path = native_assets.rstrip("/") + str(scene["franka_asset"])
        add_reference_to_stage(franka_path, "/World/Franka")

        def set_pose(prim_path: str, translation: Any, quaternion: Any | None = None) -> None:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"native_warehouse_prim_missing:{prim_path}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
            if quaternion is not None:
                w, x, y, z = map(float, quaternion)
                xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(w, Gf.Vec3d(x, y, z))
                )

        set_pose("/World/WarehouseWorkcell", placements["workcell_translation_m"])
        set_pose("/World/Franka", placements["franka_base_translation_m"])
        set_pose("/World/Task/SprayCan", placements["spraycan_translation_m"])

        spray_prim = stage.GetPrimAtPath("/World/Task/SprayCan")
        rigid = UsdPhysics.RigidBodyAPI.Apply(spray_prim)
        rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(spray_prim).CreateMassAttr(0.25)
        collision_count = sum(
            1 for prim in Usd.PrimRange(spray_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)
        )

        tray_center = placements["tray_center_translation_m"]
        tray = UsdGeom.Cube.Define(stage, "/World/Task/Tray")
        tray.CreateSizeAttr(1.0)
        tray.CreateDisplayColorAttr([(0.05, 0.1, 0.9)])
        tray_xform = UsdGeom.Xformable(tray.GetPrim())
        tray_xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, tray_center)))
        tray_xform.AddScaleOp().Set(Gf.Vec3f(0.36, 0.28, 0.03))
        UsdPhysics.CollisionAPI.Apply(tray.GetPrim())

        semantic_targets = {
            "franka": stage.GetPrimAtPath("/World/Franka"),
            "spraycan": spray_prim,
            "tray": tray.GetPrim(),
        }
        semantic_labeling_apis: dict[str, dict[str, Any]] = {}
        for semantic_label, semantic_prim in semantic_targets.items():
            if not semantic_prim.IsValid():
                raise ValueError(f"native_semantic_target_prim_missing:{semantic_label}")
            semantic_labeling_apis[semantic_label] = _author_renderable_semantic_label_tree(
                root_prim=semantic_prim,
                semantic_label=semantic_label,
                prim_range=Usd.PrimRange,
                is_renderable=lambda prim: prim.IsA(UsdGeom.Gprim),
            )
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        entity_world_points = {
            entity_id: _world_bound_center_xyz(bbox_cache.ComputeWorldBound(prim))
            for entity_id, prim in semantic_targets.items()
            if entity_id in {"spraycan", "tray"}
        }
        spraycan_world_bound = bbox_cache.ComputeWorldBound(spray_prim).ComputeAlignedBox()
        workcell_root = stage.GetPrimAtPath("/World/WarehouseWorkcell")
        workcell_child_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for child in workcell_root.GetChildren():
            aligned = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
            child_min = np.asarray(aligned.GetMin(), dtype=float).reshape(3)
            child_max = np.asarray(aligned.GetMax(), dtype=float).reshape(3)
            if np.isfinite(child_min).all() and np.isfinite(child_max).all() and np.all(
                child_max > child_min
            ):
                workcell_child_bounds[str(child.GetPath())] = (child_min, child_max)
        initial_task_placement_validation = _aabb_intersection_report(
            subject_min_xyz=spraycan_world_bound.GetMin(),
            subject_max_xyz=spraycan_world_bound.GetMax(),
            obstacle_bounds=workcell_child_bounds,
            allowed_enclosing_obstacle_paths=tuple(
                str(path)
                for path in placements.get("initial_target_allowed_enclosing_prim_paths", [])
            ),
        )

        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(1000.0)
        distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        distant.CreateIntensityAttr(2500.0)

        required_views = _required_views_from_spec(spec)
        camera_objects: dict[str, CameraSensor] = {}
        camera_paths: dict[str, str] = {}
        for view_id in (view for view in required_views if view != "wrist"):
            camera_spec = spec["cameras"][view_id]
            eye = np.asarray(camera_spec["position_m"], dtype=float)
            forward = np.asarray(camera_spec["look_at_m"], dtype=float) - eye
            quaternion = _camera_quaternion_wxyz(forward, (0.0, 0.0, 1.0))
            camera_path = f"/World/Cameras/{''.join(part.title() for part in view_id.split('_'))}"
            camera_paths[view_id] = camera_path
            camera_prim = UsdGeom.Camera.Define(stage, camera_path)
            # RtxCamera may create a missing camera and apply OmniSensorAPI, but
            # when wrapping an existing UsdGeom.Camera Isaac Sim 6 validates
            # that the schema is already present.  We author the USD camera
            # ourselves so its calibrated optics and pose exist before reset.
            camera_prim.GetPrim().ApplyAPI("OmniSensorAPI")
            camera_prim.CreateVerticalApertureAttr(15.2908)
            camera_prim.CreateHorizontalApertureAttr(15.2908 * 640.0 / 480.0)
            camera_prim.CreateFocalLengthAttr(
                15.2908
                / (2.0 * math.tan(math.radians(float(camera_spec["vertical_fov_deg"])) / 2.0))
            )
            set_pose(camera_path, eye, quaternion)

        # Define the wrist render sensor before World.reset(), but keep it outside
        # the articulation hierarchy. Isaac's render-only joint teleport updates
        # physics link tensors without reliably propagating a newly-authored child
        # camera through Fabric. We therefore synchronize this world-space sensor
        # from the live panda_hand tensor pose and one fixed parent-local mount.
        wrist = spec["cameras"]["wrist"]
        calibration = wrist["rigid_mount_orientation"]
        wrist_path = "/World/Cameras/Wrist"
        camera_paths["wrist"] = wrist_path
        wrist_prim = UsdGeom.Camera.Define(stage, wrist_path)
        wrist_prim.GetPrim().ApplyAPI("OmniSensorAPI")
        wrist_prim.CreateVerticalApertureAttr(15.2908)
        wrist_prim.CreateHorizontalApertureAttr(15.2908 * 640.0 / 480.0)
        wrist_prim.CreateFocalLengthAttr(
            15.2908 / (2.0 * math.tan(math.radians(float(wrist["vertical_fov_deg"])) / 2.0))
        )
        set_pose(wrist_path, wrist["mount_translation_m"], (1.0, 0.0, 0.0, 0.0))

        robot = SingleArticulation(prim_path="/World/Franka", name="native_warehouse_franka")
        world.scene.add(robot)
        world.reset()
        physics_sim_view = world.physics_sim_view
        if physics_sim_view is None:
            raise ValueError("native_articulation_physics_view_missing")

        initial_joints = np.asarray(
            [0.2897, 0.50732, -0.140016, -2.176, -0.0310497, 2.51592, -0.49251, 0.04, 0.04]
        )
        commanded_joints = initial_joints.copy()
        commanded_joints[0] += 0.12
        physics_step_after_reset = int(world.current_time_step_index)
        precalibration_joint_state = _apply_and_measure_render_only_joint_pose(
            robot=robot,
            joint_positions=initial_joints,
            phase="initial_precalibration",
            render=lambda: _render_world_without_physics_advance(world),
            render_count=2,
        )

        def usd_camera_matrix(path: str) -> np.ndarray:
            cache = UsdGeom.XformCache()
            return _matrix_array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))

        precalibration_link_matrices = _articulation_link_world_pose_matrices(
            robot=robot,
            simulation_view=physics_sim_view,
        )
        hand_initial_matrix = precalibration_link_matrices.get("panda_hand")
        if hand_initial_matrix is None:
            raise ValueError("native_articulation_panda_hand_link_missing")
        target_relative_eye_offset = calibration.get("target_relative_camera_eye_world_offset_m")
        wrist_quaternion, wrist_mount_calibration = _rigid_wrist_mount_from_initial_task_framing(
            parent_to_world=hand_initial_matrix,
            mount_translation_parent=wrist["mount_translation_m"],
            target_world_points=_wrist_calibration_target_world_points(
                calibration=calibration,
                entity_world_points=entity_world_points,
            ),
            world_up=calibration["world_up"],
            camera_eye_world_offset=(
                (0.0, 0.0, WRIST_CAMERA_WORLD_CLEARANCE_M)
                if target_relative_eye_offset is None
                else (0.0, 0.0, 0.0)
            ),
            target_relative_camera_eye_world_offset=target_relative_eye_offset,
        )
        resolved_wrist_mount_translation = wrist_mount_calibration[
            "resolved_mount_translation_parent_m"
        ]

        # Isaac Sim 6 deprecates the legacy Camera frame-dict path in favor of
        # CameraSensor. Bind both RGB and semantic segmentation to the same
        # render product so visibility evidence comes from the exact RGB view.
        # The new API spells the color annotator ``rgb`` and declares resolution
        # in OpenCV/NumPy order: (height, width).
        def create_camera_authoring(view_id: str, camera_path: str) -> Any:
            try:
                return RtxCamera(
                    camera_path,
                    reset_xform_op_properties=False,
                )
            except Exception as exc:
                raise ValueError(
                    f"native_camera_rtx_authoring_setup_failed:{view_id}:{type(exc).__name__}"
                ) from exc

        def create_camera_sensor(view_id: str, authored_camera: Any) -> Any:
            try:
                sensor = CameraSensor(
                    authored_camera,
                    resolution=(480, 640),
                    annotators=["rgb", "semantic_segmentation"],
                )
            except Exception as exc:
                raise ValueError(
                    f"native_camera_rtx_runtime_setup_failed:{view_id}:{type(exc).__name__}"
                ) from exc
            if sensor.authoring_object is not authored_camera:
                raise ValueError(f"native_camera_rtx_authoring_identity_lost:{view_id}")
            return sensor

        camera_authoring = {
            view_id: create_camera_authoring(view_id, camera_paths[view_id])
            for view_id in required_views
        }
        # Position the exact authoring object that CameraSensor will bind before
        # creating its render product. CameraSensor lazily creates an optical
        # Camera wrapper (which normalizes xform ops), so the same RtxCamera
        # object is also reused for every subsequent wrist pose write and read.
        wrist_pose_view = camera_authoring["wrist"]
        _synchronize_camera_to_rigid_link(
            pose_view=wrist_pose_view,
            parent_to_world=hand_initial_matrix,
            mount_translation_parent=resolved_wrist_mount_translation,
            mount_orientation_parent_wxyz=wrist_quaternion,
        )
        for view_id in (view for view in required_views if view != "wrist"):
            camera_objects[view_id] = create_camera_sensor(view_id, camera_authoring[view_id])
        camera_objects["wrist"] = create_camera_sensor("wrist", camera_authoring["wrist"])
        _synchronize_camera_to_rigid_link(
            pose_view=wrist_pose_view,
            parent_to_world=hand_initial_matrix,
            mount_translation_parent=resolved_wrist_mount_translation,
            mount_orientation_parent_wxyz=wrist_quaternion,
        )
        for _ in range(2):
            _render_world_without_physics_advance(world)
        initial_joint_state = _apply_and_measure_render_only_joint_pose(
            robot=robot,
            joint_positions=initial_joints,
            phase="initial_observation",
            render=lambda: _render_world_without_physics_advance(world),
            render_count=4,
        )

        initial_link_matrices = _articulation_link_world_pose_matrices(
            robot=robot,
            simulation_view=physics_sim_view,
        )
        hand_world_initial = initial_link_matrices.get("panda_hand")
        if hand_world_initial is None:
            raise ValueError("native_articulation_panda_hand_link_missing")
        requested_wrist_world_initial = _synchronize_camera_to_rigid_link(
            pose_view=wrist_pose_view,
            parent_to_world=hand_world_initial,
            mount_translation_parent=resolved_wrist_mount_translation,
            mount_orientation_parent_wxyz=wrist_quaternion,
        )
        for _ in range(2):
            _render_world_without_physics_advance(world)
        wrist_world_initial = _unified_world_pose_matrix(wrist_pose_view)
        wrist_pose_congruence = {
            "initial": _camera_pose_backend_congruence(
                requested_camera_to_world=requested_wrist_world_initial,
                authoring_camera_to_world=wrist_world_initial,
                usd_camera_to_world=usd_camera_matrix(wrist_path),
            )
        }
        if not wrist_pose_congruence["initial"]["congruent"]:
            raise ValueError("native_wrist_camera_pose_backend_divergence:initial")
        wrist_local_initial = wrist_world_initial @ np.linalg.inv(hand_world_initial)
        requested_wrist_local_initial = requested_wrist_world_initial @ np.linalg.inv(
            hand_world_initial
        )
        actual_wrist_eye = wrist_world_initial[3, :3]
        actual_wrist_forward = (np.asarray([0.0, 0.0, -1.0, 0.0]) @ wrist_world_initial)[:3]
        actual_wrist_forward /= np.linalg.norm(actual_wrist_forward)
        target_centroid = np.asarray(
            wrist_mount_calibration["target_centroid_world_m"], dtype=float
        )
        actual_target_direction = target_centroid - actual_wrist_eye
        actual_target_direction /= np.linalg.norm(actual_target_direction)
        wrist_mount_calibration.update(
            {
                "calibrated_after_initial_joint_hold": True,
                "actual_initial_camera_eye_world_m": actual_wrist_eye.tolist(),
                "actual_initial_optical_axis_world": actual_wrist_forward.tolist(),
                "actual_initial_target_alignment_cosine": float(
                    np.dot(actual_wrist_forward, actual_target_direction)
                ),
            }
        )
        entity_points = entity_world_points
        franka_link_points = {
            link_name: matrix[3, :3]
            for link_name, matrix in initial_link_matrices.items()
            if link_name.startswith("panda_")
        }
        if not franka_link_points:
            raise ValueError("native_warehouse_franka_link_projection_points_missing")

        frames: dict[str, dict[str, Any]] = {view: {} for view in required_views}

        def save_frames(phase: str) -> int:
            step = int(world.current_time_step_index)
            for view_id, camera in camera_objects.items():
                rgb_frame = _camera_sensor_annotator_frame(sensor=camera, annotator="rgb")
                rgba = np.asarray(rgb_frame["data"])
                if rgba.ndim != 3 or rgba.shape[0:2] != (480, 640):
                    shape_code = "x".join(str(int(value)) for value in rgba.shape)
                    raise ValueError(f"native_camera_frame_shape_invalid:{view_id}:{shape_code}")
                path = output_dir / f"{view_id}_{phase}.png"
                Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(path)
                frames[view_id][f"{phase}_frame_path"] = str(path)
                frames[view_id][f"{phase}_physics_step"] = step
                semantic_frame = _camera_sensor_annotator_frame(
                    sensor=camera,
                    annotator="semantic_segmentation",
                )
                frames[view_id].setdefault("required_entities_visible_pixels_by_phase", {})[
                    phase
                ] = _semantic_entity_visibility(
                    semantic_frame=semantic_frame,
                    entity_labels={
                        entity_id: entity_id
                        for entity_id in (
                            ("spraycan",) if view_id == "wrist" else ("franka", "spraycan", "tray")
                        )
                    },
                )
            return step

        def record_entity_projections(phase: str) -> None:
            for view_id in required_views:
                path = camera_paths[view_id]
                camera_to_world = (
                    usd_camera_matrix(path)
                    if view_id != "wrist"
                    else _unified_world_pose_matrix(wrist_pose_view)
                )
                if view_id != "wrist":
                    required, projection_evidence = _project_required_external_entities(
                        camera_to_world=camera_to_world,
                        task_points=entity_points,
                        franka_link_points=franka_link_points,
                        width=640,
                        height=480,
                        vfov_deg=float(spec["cameras"][view_id]["vertical_fov_deg"]),
                    )
                    frames[view_id].setdefault("franka_projection_evidence_by_phase", {})[phase] = (
                        projection_evidence
                    )
                else:
                    required = _project_world_points(
                        camera_to_world=camera_to_world,
                        points=entity_points,
                        width=640,
                        height=480,
                        vfov_deg=float(spec["cameras"][view_id]["vertical_fov_deg"]),
                    )
                frames[view_id].setdefault("required_entities_projected_in_frame_by_phase", {})[
                    phase
                ] = required

        def finalize_entity_projections() -> None:
            for view_id in required_views:
                phase_values = frames[view_id]["required_entities_projected_in_frame_by_phase"]
                frames[view_id]["required_entities_projected_in_frame"] = (
                    _summarize_required_entity_projections(
                        view_id=view_id,
                        projections_by_phase=phase_values,
                    )
                )
                visibility_values = frames[view_id]["required_entities_visible_pixels_by_phase"]
                frames[view_id]["required_entities_visible_pixels"] = (
                    _summarize_required_entity_visibility(
                        view_id=view_id,
                        visibility_by_phase=visibility_values,
                    )
                )

        initial_step = save_frames("initial")
        record_entity_projections("initial")

        commanded_joint_state = _apply_and_measure_render_only_joint_pose(
            robot=robot,
            joint_positions=commanded_joints,
            phase="commanded_observation",
            render=lambda: _render_world_without_physics_advance(world),
            render_count=4,
        )
        commanded_link_matrices = _articulation_link_world_pose_matrices(
            robot=robot,
            simulation_view=physics_sim_view,
        )
        hand_world_commanded = commanded_link_matrices.get("panda_hand")
        if hand_world_commanded is None:
            raise ValueError("native_articulation_panda_hand_link_missing")
        requested_wrist_world_commanded = _synchronize_camera_to_rigid_link(
            pose_view=wrist_pose_view,
            parent_to_world=hand_world_commanded,
            mount_translation_parent=resolved_wrist_mount_translation,
            mount_orientation_parent_wxyz=wrist_quaternion,
        )
        for _ in range(2):
            _render_world_without_physics_advance(world)
        commanded_step = save_frames("commanded")
        record_entity_projections("commanded")
        finalize_entity_projections()
        wrist_world_commanded = _unified_world_pose_matrix(wrist_pose_view)
        wrist_pose_congruence["commanded"] = _camera_pose_backend_congruence(
            requested_camera_to_world=requested_wrist_world_commanded,
            authoring_camera_to_world=wrist_world_commanded,
            usd_camera_to_world=usd_camera_matrix(wrist_path),
        )
        if not wrist_pose_congruence["commanded"]["congruent"]:
            raise ValueError("native_wrist_camera_pose_backend_divergence:commanded")
        wrist_local_commanded = wrist_world_commanded @ np.linalg.inv(hand_world_commanded)
        requested_wrist_local_commanded = requested_wrist_world_commanded @ np.linalg.inv(
            hand_world_commanded
        )

        missing = [
            row["relative_path"]
            for row in manifest.get("files") or []
            if not (assets_root / str(row.get("relative_path") or "")).is_file()
        ]
        dof_count = getattr(robot, "num_dof", None)
        if dof_count is None:
            dof_count = getattr(robot, "num_dofs", -1)
        return {
            "isaac_sim_major_version": 6,
            "scene_loaded": stage.GetPrimAtPath("/World/WarehouseWorkcell").IsValid(),
            "missing_dataset_local_dependencies": missing,
            "runtime_asset_relocations": relocation_evidence,
            "franka_dof_count": int(dof_count),
            "franka_dof_names": list(robot.dof_names),
            "spraycan_collision_mesh_count": collision_count,
            "spraycan_runtime_rigid_body": spray_prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "spraycan_kinematic_for_camera_canary": True,
            "semantic_labeling_apis": semantic_labeling_apis,
            "entity_renderable_world_centers_m": {
                entity_id: np.asarray(point, dtype=float).tolist()
                for entity_id, point in entity_world_points.items()
            },
            "initial_task_placement_validation": initial_task_placement_validation,
            "camera_runtime_api": {
                "module": "isaacsim.sensors.experimental.rtx",
                "authoring_class": "RtxCamera",
                "runtime_class": "CameraSensor",
                "annotators": ["rgb", "semantic_segmentation"],
                "resolution_width_height": [640, 480],
                "shared_render_product_per_view": True,
            },
            "views": frames,
            "wrist_mount_calibration": wrist_mount_calibration,
            "wrist_mount_implementation": {
                "mode": "explicit_live_backend_world_sensor_sync_from_physics_link_tensor",
                "parent_link": "panda_hand",
                "physics_link_pose_source": "get_link_transforms_after_update_articulations_kinematic",
                "camera_pose_write_backend": "bound_rtx_camera_authoring_object_world_pose_api",
                "render_product_bound_to_same_authoring_object": True,
                "pose_congruence_by_phase": wrist_pose_congruence,
                "per_frame_task_reaim_performed": False,
            },
            "franka_render_only_joint_state": {
                "mode": "render_only_kinematic_joint_state_transition",
                "physics_dynamics_claimed": False,
                "precalibration": precalibration_joint_state,
                "initial": initial_joint_state,
                "commanded": commanded_joint_state,
            },
            "camera_transition_physics_steps_advanced": int(world.current_time_step_index)
            - physics_step_after_reset,
            "wrist_parent_world_displacement_m": float(
                np.linalg.norm(hand_world_commanded[3, :3] - hand_world_initial[3, :3])
            ),
            "wrist_camera_world_displacement_m": float(
                np.linalg.norm(wrist_world_commanded[3, :3] - wrist_world_initial[3, :3])
            ),
            "wrist_camera_local_transform_delta": float(
                np.max(np.abs(wrist_local_commanded - wrist_local_initial))
            ),
            "wrist_camera_observed_local_transform_delta": float(
                np.max(np.abs(wrist_local_commanded - wrist_local_initial))
            ),
            "wrist_camera_requested_local_transform_delta": float(
                np.max(
                    np.abs(
                        requested_wrist_local_commanded - requested_wrist_local_initial
                    )
                )
            ),
            "external_wrist_timestamp_pairs_exact": all(
                frames["external"][f"{phase}_physics_step"]
                == frames["wrist"][f"{phase}_physics_step"]
                for phase in ("initial", "commanded")
            ),
            "camera_timestamps_exact": all(
                len({frames[view_id][f"{phase}_physics_step"] for view_id in required_views}) == 1
                for phase in ("initial", "commanded")
            ),
            "initial_physics_step": initial_step,
            "commanded_physics_step": commanded_step,
            "rankings_or_policy_outcomes_accessed": False,
        }
    finally:
        simulation_app.close()


def _validated_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != CANARY_SPEC_SCHEMA_VERSION:
        raise ValueError("nvidia_warehouse_native_camera_canary_spec_invalid")
    spec = dict(value)
    declared = spec.pop("spec_sha256", None)
    if declared != canonical_sha256(spec):
        raise ValueError("nvidia_warehouse_native_camera_canary_spec_sha256_invalid")
    spec["spec_sha256"] = declared
    return spec


def _image_evidence(path_value: Any, expected_resolution: list[int]) -> dict[str, Any]:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("nvidia_warehouse_native_camera_frame_missing_or_unsafe")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "resolution": [width, height],
        "spatial_std": float(np.std(rgb)),
        "nonblank": bool(float(np.std(rgb)) >= MIN_NONBLANK_SPATIAL_STD),
        "resolution_matches": [width, height] == expected_resolution,
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_safe_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_evidence(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe_evidence(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def assess_native_camera_backend_result(
    *, spec: Mapping[str, Any], backend_result: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        required_views = _required_views_from_spec(spec)
    except ValueError as exc:
        return {
            "status": "failed",
            "blockers": [str(exc)],
            "views": {},
            "camera_motion_and_mount_checks_passed": False,
        }
    if backend_result.get("isaac_sim_major_version") != 6:
        blockers.append("isaac_sim_major_version_not_6")
    if backend_result.get("scene_loaded") is not True:
        blockers.append("native_workcell_scene_not_loaded")
    if backend_result.get("missing_dataset_local_dependencies") not in ([], ()):
        blockers.append("native_workcell_dataset_local_dependencies_missing")
    if backend_result.get("franka_dof_count") != 9:
        blockers.append("native_franka_dof_count_invalid")
    if backend_result.get("spraycan_collision_mesh_count", 0) < 1:
        blockers.append("native_spraycan_collision_missing")
    if backend_result.get("spraycan_runtime_rigid_body") is not True:
        blockers.append("native_spraycan_rigid_body_missing")
    placement_value = backend_result.get("initial_task_placement_validation")
    placement = placement_value if isinstance(placement_value, Mapping) else {}
    if placement.get("status") != "passed":
        blockers.append("native_spraycan_initial_placement_intersects_workcell")

    views_value = backend_result.get("views")
    views = views_value if isinstance(views_value, Mapping) else {}
    view_evidence: dict[str, Any] = {}
    for view_id in required_views:
        view = views.get(view_id)
        if not isinstance(view, Mapping):
            blockers.append(f"native_camera_view_missing:{view_id}")
            continue
        expected = list(spec["cameras"][view_id]["resolution"])
        frames = {}
        for phase in ("initial", "commanded"):
            try:
                frame = _image_evidence(view.get(f"{phase}_frame_path"), expected)
            except (OSError, ValueError):
                blockers.append(f"native_camera_frame_invalid:{view_id}:{phase}")
                continue
            frames[phase] = frame
            if not frame["nonblank"]:
                blockers.append(f"native_camera_frame_blank:{view_id}:{phase}")
            if not frame["resolution_matches"]:
                blockers.append(f"native_camera_resolution_invalid:{view_id}:{phase}")
        projected = view.get("required_entities_projected_in_frame")
        if not isinstance(projected, Mapping) or not all(projected.values()):
            blockers.append(f"native_camera_required_entity_projection_failed:{view_id}")
        visible = view.get("required_entities_visible_pixels")
        if not isinstance(visible, Mapping) or not all(visible.values()):
            blockers.append(f"native_camera_required_entity_visibility_failed:{view_id}")
        view_evidence[view_id] = {
            "frames": frames,
            "required_entities_projected_in_frame": dict(projected or {}),
            "required_entities_projected_in_frame_by_phase": dict(
                view.get("required_entities_projected_in_frame_by_phase") or {}
            ),
            "required_entities_visible_pixels": dict(visible or {}),
            "required_entities_visible_pixels_by_phase": dict(
                view.get("required_entities_visible_pixels_by_phase") or {}
            ),
        }

    if "external_2" in required_views:
        for phase in ("initial", "commanded"):
            first = view_evidence.get("external", {}).get("frames", {}).get(phase, {})
            second = view_evidence.get("external_2", {}).get("frames", {}).get(phase, {})
            if first.get("rgb_sha256") and first.get("rgb_sha256") == second.get("rgb_sha256"):
                blockers.append(f"native_camera_external_pair_frame_not_distinct:{phase}")

    wrist_world_delta = _finite_float(backend_result.get("wrist_camera_world_displacement_m"))
    wrist_observed_local_delta = _finite_float(
        backend_result.get("wrist_camera_observed_local_transform_delta")
    )
    wrist_requested_local_delta = _finite_float(
        backend_result.get("wrist_camera_requested_local_transform_delta")
    )
    # Compatibility for injected/older backends. New Isaac evidence always
    # reports both values and gates the mathematically fixed requested mount;
    # per-phase backend congruence separately observes that the public camera
    # API honored those requested poses within its existing tolerance.
    if wrist_requested_local_delta is None:
        wrist_requested_local_delta = _finite_float(
            backend_result.get("wrist_camera_local_transform_delta")
        )
    if wrist_observed_local_delta is None:
        wrist_observed_local_delta = _finite_float(
            backend_result.get("wrist_camera_local_transform_delta")
        )
    if wrist_world_delta is None:
        blockers.append("native_wrist_camera_world_displacement_missing_or_invalid")
    elif wrist_world_delta <= MIN_WRIST_WORLD_DISPLACEMENT_M:
        blockers.append("native_wrist_camera_did_not_move_with_hand")
    if wrist_requested_local_delta is None:
        blockers.append("native_wrist_camera_local_transform_missing_or_invalid")
    elif wrist_requested_local_delta > MAX_WRIST_LOCAL_TRANSFORM_DELTA:
        blockers.append("native_wrist_camera_mount_not_rigid")
    if wrist_observed_local_delta is None:
        blockers.append("native_wrist_camera_observed_local_transform_missing_or_invalid")
    mount_value = backend_result.get("wrist_mount_calibration")
    mount = mount_value if isinstance(mount_value, Mapping) else {}
    if mount.get("mode") != "one_time_initial_task_framing_rigid_parent_local_mount":
        blockers.append("native_wrist_mount_calibration_missing_or_invalid")
    if mount.get("calibrated_after_initial_joint_hold") is not True:
        blockers.append("native_wrist_mount_not_calibrated_after_joint_hold")
    if mount.get("per_frame_task_reaim_performed") is not False:
        blockers.append("native_wrist_camera_per_frame_reaim_not_forbidden")
    joint_state_value = backend_result.get("franka_render_only_joint_state")
    joint_state = joint_state_value if isinstance(joint_state_value, Mapping) else {}
    if (
        joint_state.get("mode") != "render_only_kinematic_joint_state_transition"
        or joint_state.get("physics_dynamics_claimed") is not False
    ):
        blockers.append("native_franka_render_only_joint_state_missing_or_invalid")
    if backend_result.get("camera_transition_physics_steps_advanced") != 0:
        blockers.append("native_camera_transition_advanced_physics")
    joint_state_evidence: dict[str, Any] = {}
    for phase in ("initial", "commanded"):
        phase_value = joint_state.get(phase)
        phase_state = phase_value if isinstance(phase_value, Mapping) else {}
        error = _finite_float(phase_state.get("max_abs_position_error_rad"))
        joint_state_evidence[phase] = {
            **dict(phase_state),
            "max_abs_position_error_rad": error,
        }
        if error is None:
            blockers.append(f"native_franka_joint_state_error_missing_or_invalid:{phase}")
        elif error > MAX_KINEMATIC_JOINT_POSITION_ERROR_RAD:
            blockers.append(f"native_franka_joint_state_error_exceeded:{phase}")
        if phase_state.get("zero_time_scene_update_requested") is not True:
            blockers.append(f"native_franka_zero_time_scene_update_not_proven:{phase}")
    timestamp_key = (
        "camera_timestamps_exact"
        if len(required_views) > len(REQUIRED_VIEWS)
        else "external_wrist_timestamp_pairs_exact"
    )
    if backend_result.get(timestamp_key) is not True:
        blockers.append("native_camera_timestamps_not_synchronized")

    return {
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "views": view_evidence,
        "required_views": list(required_views),
        "wrist_camera_world_displacement_m": wrist_world_delta,
        "wrist_camera_world_displacement_min_m": MIN_WRIST_WORLD_DISPLACEMENT_M,
        "wrist_camera_local_transform_delta": wrist_requested_local_delta,
        "wrist_camera_requested_local_transform_delta": wrist_requested_local_delta,
        "wrist_camera_observed_local_transform_delta": wrist_observed_local_delta,
        "wrist_camera_local_transform_delta_max": MAX_WRIST_LOCAL_TRANSFORM_DELTA,
        "initial_task_placement_validation": dict(placement),
        "wrist_mount_calibration": dict(mount),
        "franka_render_only_joint_state": {
            "mode": joint_state.get("mode"),
            "physics_dynamics_claimed": joint_state.get("physics_dynamics_claimed"),
            "max_abs_position_error_rad": MAX_KINEMATIC_JOINT_POSITION_ERROR_RAD,
            **joint_state_evidence,
        },
        "camera_transition_physics_steps_advanced": backend_result.get(
            "camera_transition_physics_steps_advanced"
        ),
        "camera_motion_and_mount_checks_passed": not any(
            blocker.startswith("native_wrist_camera")
            or blocker.startswith("native_wrist_mount")
            or blocker.startswith("native_franka_joint")
            or blocker.startswith("native_franka_render")
            or blocker.startswith("native_camera_transition")
            for blocker in blockers
        ),
    }


def run_native_camera_canary(
    *,
    spec_path: str | Path,
    assets_root: str | Path,
    output_dir: str | Path,
    backend: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one label-free native camera canary through an injected Isaac backend."""

    spec = _validated_spec(spec_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("nvidia_warehouse_native_camera_canary_output_exists")
    output.mkdir(parents=True)
    backend_result = backend(
        spec=spec,
        assets_root=Path(assets_root).expanduser().resolve(),
        output_dir=output / "runtime",
    )
    if not isinstance(backend_result, Mapping):
        raise ValueError("nvidia_warehouse_native_camera_backend_result_invalid")
    assessment = assess_native_camera_backend_result(spec=spec, backend_result=backend_result)
    for view in assessment["views"].values():
        for frame in view["frames"].values():
            try:
                frame["relative_path"] = (
                    Path(str(frame["path"])).resolve().relative_to(output).as_posix()
                )
            except (KeyError, ValueError):
                assessment["blockers"].append("native_camera_frame_outside_canary_output")
    assessment["blockers"] = sorted(set(assessment["blockers"]))
    assessment["status"] = "passed" if not assessment["blockers"] else "failed"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": assessment["status"],
        "blockers": assessment["blockers"],
        "spec_path": str(Path(spec_path).expanduser().resolve()),
        "spec_sha256": spec["spec_sha256"],
        "assets_root": str(Path(assets_root).expanduser().resolve()),
        "backend_result": _json_safe_evidence(backend_result),
        "assessment": assessment,
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_policy_or_wam_model_invoked": False,
        "claim_boundary": {
            "native_scene_and_camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json(output / "native_camera_canary_result.json", result)
    return result


def main() -> int:  # pragma: no cover - requires the pinned Isaac GPU image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_native_camera_canary(
        spec_path=args.spec,
        assets_root=args.assets_root,
        output_dir=args.output_dir,
        backend=isaac_sim_6_backend,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 1


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "assess_native_camera_backend_result",
    "isaac_sim_6_backend",
    "run_native_camera_canary",
]


if __name__ == "__main__":
    raise SystemExit(main())
