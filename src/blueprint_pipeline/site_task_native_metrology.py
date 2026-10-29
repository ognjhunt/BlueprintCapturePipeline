"""Read bounded rigid/DROID metrology from existing native Arena handles.

This leaf never constructs an environment, steps physics, authorizes execution,
or accepts a serialized snapshot as engine proof. Missing native mechanisms
remain typed unproven observations. The existing qualification producer owns
paired samples/review; raw native readbacks are not site qualification.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_runtime import NativeTaskArenaEnvironment

SELECTOR_SCHEMA = "blueprint.site_task_native_metrology_selectors.v1"
CAPTURE_SCHEMA = "blueprint.site_task_native_metrology_capture.v1"
DEPTH_METHOD = "isaaclab_ffff603_range_to_rectified_optical_axis.v1"
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_SELECTORS = 128
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")


class NativeMetrologyError(ValueError):
    """Fixed refusal codes, never native remote bodies or private paths."""


def _require(ok: Any, code: str) -> None:
    if not ok:
        raise NativeMetrologyError("native_metrology_" + code)


def _array(value: Any) -> np.ndarray:
    # The pinned Lab exposes Warp ProxyArrays; .torch is its public read view.
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value)
    _require(not result.dtype.hasobject and 0 < result.nbytes <= MAX_ARRAY_BYTES,
             "native_array_invalid")
    return result.copy()


def optical_axis_depth(range_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Convert the pinned SDK's optical-center range to optical-axis metres.

    IsaacLab ffff603 camera.py:91-93 distinguishes these quantities. Each
    rectified pixel's ray is K^-1[u,v,1]; range / ||ray|| is Z. The original
    range array is retained separately, never relabeled or overwritten.
    """
    distance, k = np.asarray(range_m), np.asarray(intrinsics)
    _require(distance.ndim == 2 and np.issubdtype(distance.dtype, np.floating)
             and k.shape == (3, 3) and np.isfinite(k).all()
             and k[0, 0] > 0 and k[1, 1] > 0 and k[0, 1] == 0 and k[1, 0] == 0
             and np.array_equal(k[2], [0, 0, 1]) and not (distance < 0).any(), "depth_calibration_invalid")
    valid = np.isfinite(distance) & (distance > 0)
    _require(valid.any(), "depth_samples_missing")
    v, u = np.indices(distance.shape, dtype=np.float64)
    rays = np.sqrt(((u - k[0, 2]) / k[0, 0]) ** 2
                   + ((v - k[1, 2]) / k[1, 1]) ** 2 + 1)
    # Zero is a missing-sample sentinel, never a measured match. Original bytes
    # and the validity mask are retained and selected feature pixels must exist.
    return np.divide(distance, rays, out=np.zeros(distance.shape, dtype=np.float64), where=valid)


def appearance_endpoints(rgb: np.ndarray, feature: Mapping[str, Any]) -> list[list[int]]:
    """Locate preregistered appearance pixels in RGB, independently of colliders.

    A color-band scan is deliberately a bounded extraction method, not a claim
    that every real task supplies that feature. Unsupported/missing/ambiguous
    features refuse. Pixel positions are observed, never supplied as results.
    """
    _require(isinstance(feature, Mapping) and set(feature) == {
        "kind", "lower_rgb", "upper_rgb", "scan_axis", "scan_line"}
        and feature["kind"] == "rgb_band_endpoints", "appearance_selector_invalid")
    _require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3,
             "rgb_invalid")
    low, high = np.asarray(feature["lower_rgb"]), np.asarray(feature["upper_rgb"])
    _require(low.shape == high.shape == (3,) and low.dtype.kind in "iu" and high.dtype.kind in "iu"
             and (0 <= low).all() and (low <= high).all() and (high <= 255).all(),
             "appearance_selector_invalid")
    axis, line = feature["scan_axis"], feature["scan_line"]
    _require(axis in {"u", "v"} and type(line) is int
             and 0 <= line < rgb.shape[0 if axis == "u" else 1], "appearance_selector_invalid")
    scan = rgb[line] if axis == "u" else rgb[:, line]
    selected = np.flatnonzero(np.all((scan >= low) & (scan <= high), axis=1))
    _require(len(selected) >= 2 and np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)),
             "appearance_feature_missing_or_ambiguous")
    return ([[int(selected[0]), line], [int(selected[-1]), line]] if axis == "u"
            else [[line, int(selected[0])], [line, int(selected[-1])]])


def _stage():
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    _require(stage is not None, "active_stage_unavailable")
    return stage


def _asset(built: NativeTaskArenaEnvironment, selector: Mapping[str, Any]):
    role = selector.get("asset_role")
    _require(role in built.scene_asset_names, "asset_role_unavailable")
    return built.env.unwrapped.scene[built.scene_asset_names[role]]


def _prim(built: NativeTaskArenaEnvironment, selector: Mapping[str, Any], stage: Any):
    asset = _asset(built, selector)
    scene = built.env.unwrapped.scene
    path = str(asset.cfg.prim_path)
    paths = scene.env_prim_paths
    _require(len(paths) == 1, "single_environment_required")
    path = path.replace("{ENV_REGEX_NS}", paths[0])
    expression = getattr(scene, "env_regex_ns", None)
    if expression:
        path = path.replace(str(expression), paths[0])
    relative = selector.get("relative_prim_path", "")
    _require(isinstance(relative, str) and "\\" not in relative and "\x00" not in relative
             and not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts
             and (not relative or PurePosixPath(relative).as_posix() == relative), "prim_selector_invalid")
    path = path.rstrip("/") + ("/" + relative if relative else "")
    _require(path.startswith(paths[0] + "/") and not any(c in path for c in "*{}[]"),
             "prim_identity_unresolved")
    prim = stage.GetPrimAtPath(path)
    _require(prim and prim.IsValid(), "prim_unavailable")
    return prim


def _bounds(prim: Any, *, collision: bool) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = prim.GetStage()
    _require(UsdGeom.GetStageMetersPerUnit(stage) == 1.0, "native_stage_units_unproven")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy,
                                                    UsdGeom.Tokens.render], useExtentsHint=False)
    targets = list(itertools.islice(Usd.PrimRange(prim), 4097)) if collision else [prim]
    _require(len(targets) <= 4096, "prim_budget_exceeded")
    if collision:
        targets = [p for p in targets if p.HasAPI(UsdPhysics.CollisionAPI)]
    _require(bool(targets), "collision_representation_unavailable" if collision else "geometry_unavailable")
    bounds = [cache.ComputeWorldBound(p).ComputeAlignedRange() for p in targets]
    low = np.min([np.asarray(b.GetMin(), dtype=float) for b in bounds], axis=0)
    high = np.max([np.asarray(b.GetMax(), dtype=float) for b in bounds], axis=0)
    _require(np.isfinite(low).all() and np.isfinite(high).all() and (high >= low).all(),
             "native_bounds_invalid")
    return low, high


def _floor_plane(prim: Any) -> list[float]:
    from pxr import Usd, UsdGeom
    _require(UsdGeom.GetStageMetersPerUnit(prim.GetStage()) == 1.0,
             "native_stage_units_unproven")
    mesh = UsdGeom.Mesh(prim)
    _require(bool(mesh), "registered_floor_mesh_unavailable")
    source = mesh.GetPointsAttr().Get()
    _require(source is not None and 3 <= len(source) <= 4096, "registered_floor_mesh_invalid")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = np.asarray([transform.Transform(p) for p in source], dtype=float)
    _require(np.isfinite(points).all(), "registered_floor_mesh_invalid")
    normal = None
    for index in range(1, len(points) - 1):
        candidate = np.cross(points[index] - points[0], points[index + 1] - points[0])
        length = np.linalg.norm(candidate)
        if length > np.finfo(float).eps:
            normal = candidate / length
            break
    _require(normal is not None, "registered_floor_mesh_invalid")
    # Numerical coplanarity is representation validity, not a fidelity threshold.
    _require(np.allclose((points - points[0]) @ normal, 0, rtol=0, atol=1e-9),
             "registered_floor_not_planar")
    up = UsdGeom.GetStageUpAxis(prim.GetStage())
    _require(up in {"Y", "Z"}, "native_up_axis_unproven")
    if normal[1 if up == "Y" else 2] < 0:
        normal = -normal
    return [*normal.tolist(), -float(np.dot(normal, points[0]))]


def _raycast(origin: list[float], direction: list[float], distance: float) -> Mapping[str, Any]:
    from omni.physx import get_physx_scene_query_interface
    return get_physx_scene_query_interface().raycast_closest(origin, direction, distance)


def _observed_rays(built, selections, points, stage):
    result = {}
    for name, selector in selections.items():
        origin = points[selector["origin_anchor"]]
        direction = np.asarray(selector["direction"], dtype=float)
        distance = selector["maximum_distance_m"]
        _require(direction.shape == (3,) and np.isfinite(direction).all()
                 and math.isclose(float(np.linalg.norm(direction)), 1.0, abs_tol=1e-9)
                 and type(distance) in (int, float) and math.isfinite(distance) and 0 < distance <= 10000,
                 "ray_selector_invalid")
        hit = _raycast(origin, direction.tolist(), float(distance))
        _require(isinstance(hit, Mapping) and hit.get("hit") is True, "collision_ray_missing")
        position = np.asarray(hit.get("position"), dtype=float)
        _require(position.shape == (3,) and np.isfinite(position).all(), "collision_ray_invalid")
        expected = str(_prim(built, selector, stage).GetPath())
        actual = hit.get("collision")
        _require(isinstance(actual, str) and (actual == expected or actual.startswith(expected + "/")),
                 "collision_ray_unexpected_collider")
        result[name] = {"query": "nearest_scene_collision", "hit": True, "origin_m": list(origin),
                        "hit_point_m": position.tolist(), "collision_prim_path": actual,
                        "direction": direction.tolist(), "maximum_distance_m": float(distance)}
    return result


def _dynamics(built, selections):
    masses, raw = {}, {}
    for name, selector in selections.items():
        subject = _asset(built, selector)
        view = subject.root_view  # pinned ffff603 PhysX RigidObject public view
        mass = _array(view.get_masses()).reshape(-1)
        material = _array(view.get_material_properties())
        _require(len(mass) == 1 and np.isfinite(mass).all() and mass[0] > 0
                 and 0 < material.size <= 4096 * 3 and np.isfinite(material).all() and material.shape[-1] == 3,
                 "rigid_native_dynamics_invalid")
        masses[name] = float(mass[0])
        raw[name] = {"mass_kg": masses[name], "native_material_properties": material.tolist(),
                     "method": "root_view.get_masses_and_get_material_properties"}
    return masses, raw


def _bound_adapter(built, adapter):
    from .adp009d_isaac_episode_adapter import IsaacEpisodeAdapter
    _require(isinstance(adapter, IsaacEpisodeAdapter), "native_episode_adapter_required")
    adapter_env = getattr(adapter, "_env", None)
    _require(adapter_env is not None and adapter_env.unwrapped is built.env.unwrapped,
             "native_episode_environment_mismatch")


def _bound_camera(built, adapter, role):
    _bound_adapter(built, adapter)
    names = getattr(adapter, "_camera_scene_names", None)
    _require(isinstance(names, Mapping) and role in names and role in built.camera_scene_names,
             "native_camera_unavailable")
    camera = built.env.unwrapped.scene[built.camera_scene_names[role]]
    _require(adapter._env.unwrapped.scene[names[role]] is camera,
             "native_episode_camera_mismatch")
    return camera


def _camera(built, adapter, selector):
    role = selector["camera_role"]
    camera = _bound_camera(built, adapter, role)
    metadata = adapter.read_control_observation_metadata()
    _require(_bound_camera(built, adapter, role) is camera, "native_episode_camera_mismatch")
    before = _array(camera.frame)
    output = camera.data.output
    rgb = _array(output["rgb"])[0]
    k = _array(camera.data.intrinsic_matrices)[0]
    source_name = next((name for name in ("distance_to_image_plane", "depth", "distance_to_camera")
                        if name in output), None)
    _require(source_name is not None, "native_depth_unavailable")
    original = _array(output[source_name])[0]
    if original.ndim == 3 and original.shape[-1] == 1:
        original = original[..., 0]
    depth = optical_axis_depth(original, k) if source_name == "distance_to_camera" else original.copy()
    # Run the same calibration/value validation even for a native Z-depth buffer.
    if source_name != "distance_to_camera":
        optical_axis_depth(depth, k)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    _require(depth.shape == rgb.shape[:2] and np.array_equal(before, _array(camera.frame)),
             "native_frame_generation_changed")
    pixels = appearance_endpoints(rgb, selector["appearance_feature"])
    _require(all(depth[v, u] > 0 for u, v in pixels), "native_depth_feature_missing")
    sync = metadata.get("synchronizations", {}).get(role)
    calibration = metadata.get("calibrations", {}).get(role)
    _require(isinstance(sync, Mapping) and sync.get("host_bytes_ready") is True
             and isinstance(sync.get("render"), Mapping) and isinstance(calibration, Mapping)
             and all(type(sync["render"].get(key)) is int and sync["render"][key] >= 0
                     for key in ("reset_generation", "control_step_index")),
             "native_render_generation_unproven")
    freshness = sync.get("sensor_freshness")
    _require(isinstance(freshness, Mapping) and freshness.get("status") == "observed"
             and type(freshness.get("frame_index")) is int and before.size == 1
             and freshness["frame_index"] == before.reshape(-1)[0]
             and all(freshness.get(key) == sync["render"][key]
                     for key in ("reset_generation", "control_step_index"))
             and calibration.get("resolution") == [rgb.shape[1], rgb.shape[0]]
             and np.array_equal(np.asarray(calibration.get("intrinsic_matrix")), k),
             "native_camera_metadata_mismatch")
    valid_depth = np.isfinite(original) & (original > 0)
    return {"rgb": rgb, "depth_m": depth, "intrinsics": k, "original_depth": original,
            "valid_depth": valid_depth}, {
        "pixels": pixels, "appearance_method": "rgb_band_endpoints.v1", "source_depth_quantity": source_name,
        "depth_quantity": "optical_axis_m", "depth_method": DEPTH_METHOD if source_name == "distance_to_camera"
        else "native_optical_axis_buffer.v1", "camera_frame": before.tolist(),
        "invalid_depth_pixels": int(np.count_nonzero(~valid_depth)),
        "missing_depth_encoding": "zero_sentinel_with_original_depth_and_validity_mask",
        "calibration": dict(calibration), "synchronization": dict(sync)}


def _capture_epoch(built, adapter):
    _bound_adapter(built, adapter)
    metadata = adapter.read_control_observation_metadata()
    syncs = metadata.get("synchronizations")
    _require(isinstance(syncs, Mapping) and 1 <= len(syncs) <= 8, "native_render_generation_unproven")
    epochs = set()
    for sync in syncs.values():
        render = sync.get("render") if isinstance(sync, Mapping) else None
        _require(isinstance(render, Mapping)
                 and all(type(render.get(key)) is int and render[key] >= 0
                         for key in ("reset_generation", "control_step_index")),
                 "native_render_generation_unproven")
        epochs.add((render["reset_generation"], render["control_step_index"]))
    stamp, seconds = metadata.get("timestamp_ns"), metadata.get("simulation_time_s")
    _require(len(epochs) == 1 and type(stamp) is int and stamp >= 0
             and type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0,
             "native_render_generation_unproven")
    reset, step = epochs.pop()
    return {"reset_generation": reset, "control_step_index": step,
            "timestamp_ns": stamp, "simulation_time_s": seconds}


@dataclass(frozen=True)
class NativeMetrologyCapture:
    report: Mapping[str, Any]
    rgbd: Mapping[str, Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class VerifiedDefaultDroidModel:
    """Local staged-byte capability, never execution or native-open authority."""
    model_path: Path
    delivery: Mapping[str, Any]
    delivery_digest: str

    def recheck(self) -> None:
        from .blueprint_robot_model_delivery import _verify_runtime_closure
        from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
        from .team_embodiment import validate_model_dependency_closure
        row = self.delivery
        _require(isinstance(row, Mapping) and row.get("schema_version") == "blueprint.default_robot_model_delivery.v1"
                 and row.get("embodiment_preset_id") == DROID_POLICY_CANARY_PRESET_ID
                 and row.get("execution_authorized") is False
                 and row.get("claim_ceiling") == "source_asset_not_qualified"
                 and _SHA.fullmatch(str(row.get("source_import_digest")))
                 and canonical_digest(row) == self.delivery_digest, "default_model_delivery_changed")
        closure = row.get("dependency_closure")
        _require(isinstance(closure, Mapping), "default_model_closure_missing")
        root = next((entry for entry in closure.get("files", [])
                     if entry.get("path") == closure.get("model_path")), {})
        validate_model_dependency_closure(closure, model_record={"sha256": row.get("model_sha256"),
                                                               "size_bytes": root.get("size_bytes")})
        _require(self.model_path.name == closure["model_path"] and self.model_path.is_absolute(),
                 "default_model_path_changed")
        _verify_runtime_closure(self.model_path.parent, closure)


def prepare_default_droid_model(*, operator_profile: Mapping[str, Any],
                                target_directory: Path) -> VerifiedDefaultDroidModel:
    """Use existing operator selection/staging; this never grants an execution."""
    from .blueprint_robot_model_delivery import stage_default_robot_model
    delivery = stage_default_robot_model(profile=operator_profile, target_directory=target_directory)
    selected = VerifiedDefaultDroidModel(Path(target_directory) / delivery["dependency_closure"]["model_path"],
                                         delivery, canonical_digest(delivery))
    selected.recheck()
    return selected


def bind_selected_droid_model(embodiment: Any, selected: VerifiedDefaultDroidModel) -> None:
    """Set only the existing DROID spawn asset after checking selected bytes."""
    _require(isinstance(selected, VerifiedDefaultDroidModel), "verified_default_model_required")
    selected.recheck()
    _require(hasattr(embodiment.scene_config.robot.spawn, "usd_path"), "droid_spawn_asset_unavailable")
    embodiment.scene_config.robot.spawn.usd_path = str(selected.model_path)


def read_default_droid_model_use(built: NativeTaskArenaEnvironment,
                                selected: VerifiedDefaultDroidModel) -> dict[str, Any]:
    """Report actual spawn/loaded-layer observations without inventing byte-open proof."""
    _require(isinstance(built, NativeTaskArenaEnvironment)
             and isinstance(selected, VerifiedDefaultDroidModel), "native_model_handles_required")
    selected.recheck()
    robot = built.env.unwrapped.scene["robot"]
    _require(str(robot.cfg.spawn.usd_path) == str(selected.model_path), "native_spawn_model_mismatch")
    view = robot.root_view
    _require(view is not None and int(view.count) == 1, "native_robot_view_unavailable")
    opened_layers = set()
    for layer in _stage().GetUsedLayers():
        path = getattr(layer, "realPath", "")
        if path:
            opened_layers.add(str(Path(path)))
    root_seen = str(selected.model_path) in opened_layers
    selected.recheck()
    return {"status": "native_root_layer_observed" if root_seen else "unproven",
            "reason": "native_dependency_open_byte_receipts_unavailable", "runtime_open_proven": False,
            "model_sha256": selected.delivery["model_sha256"],
            "dependency_closure_digest": selected.delivery["dependency_closure"]["closure_digest"],
            "source_import_digest": selected.delivery["source_import_digest"],
            "delivery_digest": selected.delivery_digest, "current_dependency_bytes_verified": True,
            "native_root_layer_observed": root_seen,
            "proof_boundary": "current_closed_bytes_and_native_layer_identity_not_asset_stream_receipts"}


def read_rigid_native_metrology(built: NativeTaskArenaEnvironment, *, selectors: Mapping[str, Any],
                               episode_adapter: Any = None,
                               default_robot_model: VerifiedDefaultDroidModel | None = None) -> NativeMetrologyCapture:
    """Read existing rigid DROID handles only; never initialize or advance them."""
    _require(isinstance(built, NativeTaskArenaEnvironment), "native_environment_handle_required")
    _require(built.plan.get("task_kind") == "rigid_pick_place"
             and built.plan.get("robot", {}).get("robot_id") == "franka_panda", "existing_droid_rigid_required")
    _require(isinstance(selectors, Mapping) and set(selectors) == {
        "schema_version", "scene_plan_digest", "geometry_points", "floor_plane", "collision_bounds",
        "collision_rays", "rigid_bodies", "cameras"}
        and selectors["schema_version"] == SELECTOR_SCHEMA
        and selectors["scene_plan_digest"] == built.plan.get("plan_digest")
        == canonical_digest(built.plan, digest_field="plan_digest"), "selector_binding_invalid")
    for key in ("geometry_points", "collision_bounds", "collision_rays", "rigid_bodies", "cameras"):
        rows = selectors[key]
        _require(isinstance(rows, Mapping) and 1 <= len(rows) <= MAX_SELECTORS
                 and all(isinstance(name, str) and _ID.fullmatch(name) and isinstance(row, Mapping)
                         for name, row in rows.items()), "selector_budget_or_identity_invalid")
    report = {"schema_version": CAPTURE_SCHEMA, "scene_plan_digest": built.plan["plan_digest"],
              "selector_digest": canonical_digest(selectors), "observed_at_iso": datetime.now(timezone.utc).isoformat(),
              "source_kind": "native_handle_readback", "dimensions": {}, "values": {},
              "qualification_claimed": False, "physical_performance_claimed": False,
              "runtime_open_proven": False,
              "model_open_status": {"status": "unproven", "reason": "native_dependency_open_receipts_unavailable"}}
    values, status, frames = report["values"], report["dimensions"], {}
    try:
        epoch_before = _capture_epoch(built, episode_adapter)
    except (NativeMetrologyError, ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError):
        epoch_before = None
    try:
        stage = _stage()
    except (ImportError, AttributeError, NativeMetrologyError, RuntimeError, OSError):
        stage = None
    def observe(dimension, callback):
        try:
            callback()
            status[dimension] = {"status": "observed", "proof_boundary": "native_readback_not_site_qualification"}
        except (NativeMetrologyError, ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            reason = str(exc) if isinstance(exc, NativeMetrologyError) else "native_metrology_native_capability_unavailable"
            status[dimension] = {"status": "unproven", "reason": reason}
    def geometry():
        _require(stage is not None, "active_stage_unavailable")
        points = {}
        for name, selector in selectors["geometry_points"].items():
            low, high = _bounds(_prim(built, selector, stage), collision=False)
            corner = selector.get("corner")
            _require(isinstance(corner, list) and len(corner) == 3
                     and all(type(v) is int and v in (0, 1) for v in corner), "geometry_corner_invalid")
            points[name] = np.where(corner, high, low).tolist()
        values["geometry_points_m"] = points
    def registration():
        _require(stage is not None, "active_stage_unavailable")
        values["registered_floor_plane"] = _floor_plane(_prim(built, selectors["floor_plane"], stage))
    def collision():
        _require(stage is not None, "active_stage_unavailable")
        values["collision_rays"] = _observed_rays(built, selectors["collision_rays"],
                                                   values["geometry_points_m"], stage)
    def task_state():
        _require(stage is not None, "active_stage_unavailable")
        bounds, reset_states = {}, {}
        for name, selector in selectors["collision_bounds"].items():
            low, high = _bounds(_prim(built, selector, stage), collision=True)
            pose = _array(_asset(built, selector).data.root_pose_w)
            _require(pose.shape == (1, 7) and np.isfinite(pose).all()
                     and math.isclose(float(np.linalg.norm(pose[0, 3:])), 1.0, abs_tol=1e-5),
                     "native_reset_pose_unavailable")
            bounds[name] = {"min": low.tolist(), "max": high.tolist(),
                            "representation": "loaded_usd_collision_geometry_not_cooked_aabb"}
            reset_states[name] = {"pose_world_m_xyzw": pose[0].tolist(), "source": "native_rigid_object_root_pose_w"}
        values["collision_bounds_m"] = bounds
        values["native_object_reset_states"] = reset_states
    def dynamics():
        values["rigid_body_masses_kg"], values["native_rigid_body_attributes"] = _dynamics(built, selectors["rigid_bodies"])
    def observation():
        cameras = {}
        for name, selector in selectors["cameras"].items():
            frame, evidence = _camera(built, episode_adapter, selector)
            _require(sum(array.nbytes for rows in frames.values() for array in rows.values())
                     + sum(array.nbytes for array in frame.values()) <= MAX_ARRAY_BYTES,
                     "capture_array_budget_exceeded")
            frames[name], cameras[name] = frame, evidence
        values["rendered_rgbd"] = cameras
    for name, callback in (("scale", geometry), ("registration", registration), ("collision", collision),
                           ("task_state", task_state), ("observation", observation), ("task_dynamics", dynamics)):
        observe(name, callback)
    try:
        epoch_after = _capture_epoch(built, episode_adapter)
    except (NativeMetrologyError, ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError):
        epoch_after = None
    stable_epoch = epoch_before is not None and epoch_before == epoch_after
    report["reset_render_generation"] = ({"status": "observed", **epoch_before} if stable_epoch
        else {"status": "unproven", "reason": "native_reset_render_generation_missing_or_changed"})
    report["complete_native_dimensions"] = stable_epoch and all(row["status"] == "observed" for row in status.values())
    if default_robot_model is not None:
        try:
            report["model_open_status"] = read_default_droid_model_use(built, default_robot_model)
        except (NativeMetrologyError, ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            report["model_open_status"] = {"status": "unproven", "runtime_open_proven": False,
                "reason": str(exc) if isinstance(exc, NativeMetrologyError)
                else "native_metrology_native_model_capability_unavailable"}
    return NativeMetrologyCapture(report, frames)


def retain_native_metrology(capture: NativeMetrologyCapture, destination: Path) -> dict[str, Any]:
    """Retain a readback and original/derived RGB-D bytes exclusively and fsynced."""
    _require(isinstance(capture, NativeMetrologyCapture), "native_capture_required")
    _require(len(capture.rgbd) <= MAX_SELECTORS and all(_ID.fullmatch(name) for name in capture.rgbd),
             "retained_render_identity_invalid")
    _require(sum(array.nbytes for rows in capture.rgbd.values() for array in rows.values()) <= MAX_ARRAY_BYTES,
             "capture_array_budget_exceeded")
    _require(capture.report.get("schema_version") == CAPTURE_SCHEMA
             and capture.report.get("source_kind") == "native_handle_readback"
             and all(capture.report.get(key) is False for key in
                     ("runtime_open_proven", "qualification_claimed", "physical_performance_claimed")),
             "capture_claim_boundary_invalid")
    root = Path(destination)
    _require(not root.exists() and not root.is_symlink(), "retained_capture_conflict")
    root.mkdir(parents=True, mode=0o700)
    report = json.loads(json.dumps(capture.report, allow_nan=False))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identity = os.fstat(root_fd)
    def check_identity():
        current = root.lstat()
        _require((current.st_dev, current.st_ino, current.st_mode)
                 == (identity.st_dev, identity.st_ino, identity.st_mode), "capture_directory_changed")
    def write(name, raw):
        check_identity()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        with os.fdopen(fd, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        check_identity()
    try:
        artifacts = {}
        for name, arrays in capture.rgbd.items():
            stream = io.BytesIO()
            np.savez(stream, **arrays)
            raw = stream.getvalue()
            _require(0 < len(raw) <= MAX_ARRAY_BYTES, "retained_render_budget_exceeded")
            filename = name + ".npz"
            write(filename, raw)
            artifacts[name] = {"path": filename, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                               "size_bytes": len(raw)}
        report["render_artifacts"] = artifacts
        report["capture_digest"] = canonical_digest(report, digest_field="capture_digest")
        write("capture.json", (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        os.fsync(root_fd)
        check_identity()
    finally:
        os.close(root_fd)
    return report
