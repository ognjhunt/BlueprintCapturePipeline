"""Build a scene-grounded WAM/policy episode packet.

The packet is a setup artifact for simulator/generated-world loops. It can
prepare a robot/head-POV policy observation from a local USD scene when OpenUSD
rendering is available, but it never upgrades physics, safety, deployment, or
physical-robot readiness claims.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json, utc_now_iso, write_json
from .episode_spec import build_episode_specs
from .local_capture import resolve_local_capture_context
from .scene_asset_preflight import build_scene_asset_preflight


SCHEMA_VERSION = "scene_wam_policy_episode_packet.v1"
OBSERVATION_SCHEMA_VERSION = "scene_wam_policy_initial_observation.v1"
TASK_MANIFEST_SCHEMA_VERSION = "scene_episode_task_manifest.v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "scene_policy_wam_claim_boundary.v1"
RENDER_SCHEMA_VERSION = "scene_policy_observation_render_manifest.v1"
DEFAULT_USD_VISUAL_MJCF_MAX_MESHES = 40
DEFAULT_USD_VISUAL_MJCF_MAX_TRIANGLES_PER_MESH = 2000
DEFAULT_POLICY_RENDER_WIDTH = 960
DEFAULT_POLICY_RENDER_HEIGHT = 540
FALLBACK_MUJOCO_RENDER_WIDTH = 640
FALLBACK_MUJOCO_RENDER_HEIGHT = 480
DEFAULT_UNITREE_G1_FOOTPRINT_RADIUS_M = 0.55
DEFAULT_UNITREE_G1_CLEARANCE_MARGIN_M = 0.05

UNITREE_G1_SONIC_ZERO_STATE: dict[str, list[float]] = {
    "left_leg": [0.0] * 6,
    "right_leg": [0.0] * 6,
    "waist": [0.0] * 3,
    "left_arm": [0.0] * 7,
    "right_arm": [0.0] * 7,
    "left_hand": [0.0] * 7,
    "right_hand": [0.0] * 7,
    "projected_gravity": [0.0, 0.0, -1.0],
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _float_list(value: Any, *, fallback: Sequence[float]) -> list[float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(fallback)
        try:
            parsed = json.loads(text)
            return _float_list(parsed, fallback=fallback)
        except json.JSONDecodeError:
            parts = [part.strip() for part in text.split(",")]
            values: list[float] = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    return list(fallback)
            return values if values else list(fallback)
    if isinstance(value, Mapping):
        return _float_list(value.get("xyz"), fallback=fallback)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return list(fallback)
            if not math.isfinite(number):
                return list(fallback)
            values.append(number)
        return values if values else list(fallback)
    return list(fallback)


def _pose(value: Any, *, fallback_xyz: Sequence[float], source: str) -> dict[str, Any]:
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    xyz = _float_list(value, fallback=fallback_xyz)[:3]
    if len(xyz) < 3:
        xyz = list(fallback_xyz)
    rpy = _float_list(_mapping(value).get("rpy"), fallback=[0.0, 0.0, 0.0])[:3]
    if len(rpy) < 3:
        rpy = [0.0, 0.0, 0.0]
    return {"xyz": xyz, "rpy": rpy, "source": source}


def _slug(value: str, *, fallback: str) -> str:
    out = []
    for char in value.lower():
        out.append(char if char.isalnum() else "_")
    text = "_".join(part for part in "".join(out).split("_") if part)
    return text or fallback


def _load_episode(capture_root: Path, *, task_id: str) -> dict[str, Any]:
    path = capture_root / "pipeline" / "simulation_automation" / "episode_specs.json"
    manifest = _read_optional_json(path)
    episodes = manifest.get("episodes") if isinstance(manifest.get("episodes"), list) else []
    for episode in episodes:
        if isinstance(episode, Mapping) and _string(episode.get("task_id")) == task_id:
            return dict(episode)
    return dict(episodes[0]) if episodes and isinstance(episodes[0], Mapping) else {}


def _load_task_anchor(capture_root: Path, *, task_id: str, target_object_id: str) -> dict[str, Any]:
    path = capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json"
    manifest = _read_optional_json(path)
    tasks = manifest.get("tasks") if isinstance(manifest.get("tasks"), list) else []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        target_ids = [str(item) for item in task.get("target_object_ids") or []]
        if _string(task.get("task_id")) == task_id or target_object_id in target_ids:
            return dict(task)
    return dict(tasks[0]) if tasks and isinstance(tasks[0], Mapping) else {}


def _mujoco_scene_bounds_and_target(scene_asset: Path, *, target_object_id: str) -> dict[str, Any]:
    try:
        import mujoco  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"mujoco_import_failed:{type(exc).__name__}"],
        }
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_asset))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"mujoco_scene_open_failed:{type(exc).__name__}"],
        }
    center = [float(item) for item in model.stat.center]
    extent = max(0.0, float(model.stat.extent))
    scene_bounds = {
        "min": [round(item - extent, 9) for item in center],
        "max": [round(item + extent, 9) for item in center],
    }
    target_tokens = [
        token
        for token in _slug(target_object_id, fallback="target").split("_")
        if token
    ]
    target_rows: list[dict[str, Any]] = []
    for geom_index in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_index) or ""
        haystack = _slug(name, fallback="")
        if target_tokens and not all(token in haystack for token in target_tokens):
            continue
        center_xyz = [float(item) for item in data.geom_xpos[geom_index]]
        target_rows.append(
            {
                "geom_name": name,
                "center": center_xyz,
            }
        )
    target = target_rows[0] if target_rows else {}
    return {
        "status": "complete",
        "scene_bounds": scene_bounds,
        "scene_center": center,
        "matched_target_count": len(target_rows),
        "selected_target_prim": target or None,
        "target_anchor_xyz": target.get("center") or center,
        "mujoco_model_mesh_count": int(model.nmesh),
        "mujoco_model_texture_count": int(model.ntex),
        "mujoco_model_geom_count": int(model.ngeom),
        "blockers": [],
    }


def _scene_bounds_and_target(scene_asset: Path, *, target_object_id: str) -> dict[str, Any]:
    if _is_mjcf_asset(scene_asset):
        return _mujoco_scene_bounds_and_target(
            scene_asset,
            target_object_id=target_object_id,
        )
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"pxr_import_failed:{type(exc).__name__}"],
        }
    stage = Usd.Stage.Open(str(scene_asset))
    if stage is None:
        return {"status": "blocked", "blockers": ["usd_stage_open_failed"]}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    target_tokens = [
        token
        for token in _slug(target_object_id, fallback="target").split("_")
        if token
    ]
    scene_min: list[float] | None = None
    scene_max: list[float] | None = None
    target_rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        name = prim.GetName()
        path_text = str(prim.GetPath())
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            minimum = [float(item) for item in box.GetMin()]
            maximum = [float(item) for item in box.GetMax()]
        except Exception:
            continue
        if scene_min is None:
            scene_min = minimum
            scene_max = maximum
        else:
            scene_min = [min(a, b) for a, b in zip(scene_min, minimum)]
            scene_max = [max(a, b) for a, b in zip(scene_max or maximum, maximum)]
        haystack = _slug(f"{name} {path_text}", fallback="")
        if target_tokens and all(token in haystack for token in target_tokens):
            center = [(a + b) / 2.0 for a, b in zip(minimum, maximum)]
            target_rows.append(
                {
                    "prim_path": path_text,
                    "name": name,
                    "bounds": {"min": minimum, "max": maximum},
                    "center": center,
                }
            )
    scene_center = (
        [(a + b) / 2.0 for a, b in zip(scene_min, scene_max or scene_min)]
        if scene_min is not None
        else [0.0, 0.0, 0.0]
    )
    target = target_rows[0] if target_rows else {}
    return {
        "status": "complete" if scene_min is not None else "blocked",
        "scene_bounds": {"min": scene_min, "max": scene_max} if scene_min is not None else {},
        "scene_center": scene_center,
        "matched_target_count": len(target_rows),
        "selected_target_prim": target or None,
        "target_anchor_xyz": target.get("center") or scene_center,
        "blockers": [] if scene_min is not None else ["usd_scene_bounds_unavailable"],
    }


def _scenario_manifest_search_dirs(*, capture_root: Path, scene_asset: Path) -> list[Path]:
    dirs: list[Path] = []
    for candidate in [
        scene_asset.parent,
        *scene_asset.parents,
        capture_root / "pipeline",
        capture_root,
    ]:
        resolved = candidate.resolve()
        if resolved not in dirs:
            dirs.append(resolved)
        if resolved == capture_root.resolve():
            break
    return dirs


def _load_scene_scenario_specs(*, capture_root: Path, scene_asset: Path) -> dict[str, Any]:
    filenames = (
        "lightwheel_kitchen_scenarios.json",
        "isaac_execution_request.provider.json",
    )
    loaded: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_scenarios: set[tuple[str, tuple[float, float, float]]] = set()
    for directory in _scenario_manifest_search_dirs(capture_root=capture_root, scene_asset=scene_asset):
        for filename in filenames:
            path = directory / filename
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            manifest = _read_optional_json(path)
            scenarios = manifest.get("scenarios") if isinstance(manifest.get("scenarios"), list) else []
            for scenario in scenarios:
                if not isinstance(scenario, Mapping):
                    continue
                spawn = _float_list(scenario.get("spawn_position_xyz"), fallback=[])
                if len(spawn) < 3:
                    continue
                scenario_id = _string(scenario.get("scenario_id"))
                scenario_key = (
                    scenario_id,
                    (round(float(spawn[0]), 6), round(float(spawn[1]), 6), round(float(spawn[2]), 6)),
                )
                if scenario_key in seen_scenarios:
                    continue
                seen_scenarios.add(scenario_key)
                target = _float_list(scenario.get("target_position_xyz"), fallback=[])
                loaded.append(
                    {
                        "source_path": str(path),
                        "scenario_id": scenario_id,
                        "description": _string(scenario.get("description")),
                        "scenario_status": _string(scenario.get("scenario_status")),
                        "execution_proven": bool(scenario.get("execution_proven")),
                        "robot_profile_id": _string(scenario.get("robot_profile_id")),
                        "spawn_position_xyz": spawn[:3],
                        "target_position_xyz": target[:3] if len(target) >= 3 else None,
                        "waypoints_xyz": scenario.get("waypoints_xyz") or [],
                        "navigation_policy_boundary": _string(
                            scenario.get("navigation_policy_boundary")
                        ),
                    }
                )
    return {
        "status": "available" if loaded else "missing",
        "scenario_count": len(loaded),
        "scenarios": loaded,
        "source_paths": sorted({str(row["source_path"]) for row in loaded}),
        "scenario_metadata_is_execution_proof": False,
    }


def _scenario_relevance_score(
    scenario: Mapping[str, Any],
    *,
    task_id: str,
    target_object_id: str,
    target_pose: Mapping[str, Any],
) -> float:
    haystack = " ".join(
        [
            _string(scenario.get("scenario_id")),
            _string(scenario.get("description")),
            _string(scenario.get("robot_profile_id")),
        ]
    ).lower()
    target_tokens = [
        token
        for token in [
            *_slug(task_id, fallback="").split("_"),
            *_slug(target_object_id, fallback="").split("_"),
        ]
        if len(token) >= 3
    ]
    score = 1000.0
    if any(token in haystack for token in target_tokens):
        score -= 300.0
    if "sink" in haystack:
        score -= 250.0
    if "entry" in haystack:
        score -= 80.0
    if "open" in haystack:
        score -= 40.0
    if "narrow" in haystack:
        score += 120.0
    target_xyz = _vec3(target_pose.get("xyz"), fallback=[0.0, 0.0, 0.0])
    scenario_target = _float_list(scenario.get("target_position_xyz"), fallback=[])
    if len(scenario_target) >= 2:
        score += 10.0 * math.hypot(
            float(scenario_target[0]) - target_xyz[0],
            float(scenario_target[1]) - target_xyz[1],
        )
    return round(score, 6)


def _scenario_pose_candidates(
    *,
    capture_root: Path,
    scene_asset: Path,
    task_id: str,
    target_object_id: str,
    target_pose: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario_specs = _load_scene_scenario_specs(capture_root=capture_root, scene_asset=scene_asset)
    rows: list[dict[str, Any]] = []
    for scenario in scenario_specs["scenarios"]:
        spawn = _float_list(scenario.get("spawn_position_xyz"), fallback=[])
        if len(spawn) < 3:
            continue
        score = _scenario_relevance_score(
            scenario,
            task_id=task_id,
            target_object_id=target_object_id,
            target_pose=target_pose,
        )
        rows.append(
            {
                "candidate_id": f"scenario_spawn:{scenario.get('scenario_id')}",
                "pose": {
                    "xyz": [float(spawn[0]), float(spawn[1]), float(spawn[2])],
                    "rpy": [0.0, 0.0, 0.0],
                    "source": "scene_scenario_spawn_metadata",
                },
                "source": "scene_scenario_spawn_metadata",
                "selection_score": score,
                "scenario_id": scenario.get("scenario_id"),
                "scenario_description": scenario.get("description"),
                "scenario_source_path": scenario.get("source_path"),
                "scenario_execution_proven": bool(scenario.get("execution_proven")),
                "scenario_status": scenario.get("scenario_status"),
                "scenario_target_position_xyz": scenario.get("target_position_xyz"),
                "scenario_metadata_is_execution_proof": False,
            }
        )
    rows.sort(key=lambda row: (float(row["selection_score"]), _string(row.get("candidate_id"))))
    return rows, scenario_specs


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _usd_sink_front_pose_candidates(
    *,
    scene_asset: Path,
    target_pose: Mapping[str, Any],
    target_object_id: str,
) -> list[dict[str, Any]]:
    if not _is_usd_asset(scene_asset):
        return []
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-untyped]
    except Exception:
        return []
    stage = Usd.Stage.Open(str(scene_asset))
    if stage is None:
        return []
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    target_xyz = _vec3(target_pose.get("xyz"), fallback=[0.0, 0.0, 1.0])
    target_slug = _slug(target_object_id, fallback="")
    target_tokens = [token for token in target_slug.split("_") if token]
    sink_tokens = [token for token in target_tokens if token.startswith("sink")]
    sink_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        try:
            if not prim.IsA(UsdGeom.Gprim):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            minimum = [float(item) for item in box.GetMin()]
            maximum = [float(item) for item in box.GetMax()]
        except Exception:
            continue
        path_text = str(prim.GetPath())
        path_slug = _slug(path_text, fallback="")
        x_extent = maximum[0] - minimum[0]
        y_extent = maximum[1] - minimum[1]
        z_extent = maximum[2] - minimum[2]
        if x_extent > 15.0 or y_extent > 15.0 or z_extent > 15.0:
            continue
        center = [(minimum[index] + maximum[index]) / 2.0 for index in range(3)]
        if sink_tokens and any(token in path_slug for token in sink_tokens):
            sink_rows.append(
                {
                    "prim_path": path_text,
                    "bounds": {"min": minimum, "max": maximum},
                    "center": center,
                }
            )
        near_target_xy = math.hypot(center[0] - target_xyz[0], center[1] - target_xyz[1]) <= 1.8
        panel_like = (
            minimum[2] <= 0.35
            and maximum[2] >= 0.5
            and near_target_xy
            and any(token in path_slug for token in ("door", "drawer", "dishwasher"))
            and (min(x_extent, y_extent) <= 0.16)
            and (max(x_extent, y_extent) >= 0.35)
        )
        if panel_like:
            panel_rows.append(
                {
                    "prim_path": path_text,
                    "bounds": {"min": minimum, "max": maximum},
                    "center": center,
                    "x_extent": x_extent,
                    "y_extent": y_extent,
                    "target_xy_distance_m": math.hypot(
                        center[0] - target_xyz[0],
                        center[1] - target_xyz[1],
                    ),
                }
            )
    if not sink_rows or not panel_rows:
        return []
    sink_min = [
        min(_vec3(row["bounds"]["min"], fallback=[0.0, 0.0, 0.0])[index] for row in sink_rows)
        for index in range(3)
    ]
    sink_max = [
        max(_vec3(row["bounds"]["max"], fallback=[0.0, 0.0, 0.0])[index] for row in sink_rows)
        for index in range(3)
    ]
    sink_center = [(sink_min[index] + sink_max[index]) / 2.0 for index in range(3)]
    rows: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(
        sorted(panel_rows, key=lambda row: float(row["target_xy_distance_m"]))[:8]
    ):
        bounds = panel["bounds"]
        minimum = _vec3(bounds["min"], fallback=[0.0, 0.0, 0.0])
        maximum = _vec3(bounds["max"], fallback=[0.0, 0.0, 0.0])
        center = _vec3(panel["center"], fallback=[0.0, 0.0, 0.0])
        x_extent = float(panel["x_extent"])
        y_extent = float(panel["y_extent"])
        if x_extent <= y_extent:
            normal_axis = 0
            tangent_axis = 1
            outward_sign = -1.0 if center[0] <= sink_center[0] else 1.0
            face_value = minimum[0] if outward_sign < 0 else maximum[0]
            tangent_min = minimum[1]
            tangent_max = maximum[1]
        else:
            normal_axis = 1
            tangent_axis = 0
            outward_sign = -1.0 if center[1] <= sink_center[1] else 1.0
            face_value = minimum[1] if outward_sign < 0 else maximum[1]
            tangent_min = minimum[0]
            tangent_max = maximum[0]
        tangent_span = max(0.0, tangent_max - tangent_min)
        if tangent_span >= 0.25:
            base_tangent = _clamp_float(
                target_xyz[tangent_axis],
                tangent_min + 0.15,
                tangent_max - 0.15,
            )
        else:
            base_tangent = center[tangent_axis]
        tangent_offsets = (0.15, 0.0, -0.15, 0.3, -0.3)
        distances = (0.9, 1.05, 0.75, 1.2)
        for distance_index, distance in enumerate(distances):
            for offset_index, offset in enumerate(tangent_offsets):
                xyz = [target_xyz[0], target_xyz[1], 0.05]
                xyz[normal_axis] = face_value + outward_sign * distance
                xyz[tangent_axis] = base_tangent + offset
                yaw = math.atan2(target_xyz[1] - xyz[1], target_xyz[0] - xyz[0])
                rows.append(
                    {
                        "candidate_id": (
                            "sink_front_panel:"
                            f"{panel_index}:d{distance_index}:o{offset_index}"
                        ),
                        "pose": {
                            "xyz": [round(float(item), 9) for item in xyz],
                            "rpy": [0.0, 0.0, round(float(yaw), 9)],
                            "source": "usd_sink_front_panel_clearance_candidate",
                        },
                        "source": "usd_sink_front_panel_clearance_candidate",
                        "selection_score": round(
                            5.0
                            + panel_index * 5.0
                            + distance_index * 0.4
                            + offset_index * 0.02,
                            6,
                        ),
                        "front_panel_prim_path": panel["prim_path"],
                        "front_panel_bounds": bounds,
                        "sink_union_bounds": {"min": sink_min, "max": sink_max},
                        "front_normal_axis": "x" if normal_axis == 0 else "y",
                        "front_outward_sign": outward_sign,
                    }
                )
    rows.sort(key=lambda row: (float(row["selection_score"]), _string(row.get("candidate_id"))))
    return rows


def _target_ring_pose_candidates(
    *,
    target_pose: Mapping[str, Any],
    scene_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_xyz = _vec3(target_pose.get("xyz"), fallback=[0.0, 0.0, 1.0])
    scene_center = _vec3(scene_summary.get("scene_center"), fallback=[0.0, 0.0, 0.0])
    preferred_angle = math.atan2(target_xyz[1] - scene_center[1], target_xyz[0] - scene_center[0])
    angles = [preferred_angle]
    for offset_degrees in (30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        angles.append(preferred_angle + math.radians(offset_degrees))
    rows: list[dict[str, Any]] = []
    for radius in (0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0):
        for angle_index, angle in enumerate(angles):
            xyz = [
                target_xyz[0] + radius * math.cos(angle),
                target_xyz[1] + radius * math.sin(angle),
                max(0.0, min(0.08, target_xyz[2] if target_xyz[2] < 0.2 else 0.05)),
            ]
            yaw = math.atan2(target_xyz[1] - xyz[1], target_xyz[0] - xyz[0])
            rows.append(
                {
                    "candidate_id": f"target_ring:r{radius:.1f}:a{angle_index}",
                    "pose": {
                        "xyz": [round(float(item), 9) for item in xyz],
                        "rpy": [0.0, 0.0, round(float(yaw), 9)],
                        "source": "usd_target_ring_clearance_candidate",
                    },
                    "source": "usd_target_ring_clearance_candidate",
                    "selection_score": round(100.0 + radius * 10.0 + angle_index * 0.01, 6),
                    "target_ring_radius_m": radius,
                    "target_ring_angle_rad": round(float(angle), 9),
                }
            )
    return rows


def _usd_placement_obstacle_aabbs(scene_asset: Path) -> dict[str, Any]:
    if not _is_usd_asset(scene_asset):
        return {
            "status": "unsupported",
            "obstacles": [],
            "blockers": [f"unsupported_clearance_scene_asset_format:{_asset_suffix(scene_asset) or 'none'}"],
        }
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "status": "blocked",
            "obstacles": [],
            "blockers": [f"pxr_import_failed:{type(exc).__name__}"],
        }
    stage = Usd.Stage.Open(str(scene_asset))
    if stage is None:
        return {
            "status": "blocked",
            "obstacles": [],
            "blockers": ["usd_stage_open_failed"],
        }
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    obstacles: list[dict[str, Any]] = []
    skipped_floor_like = 0
    skipped_outlier = 0
    skipped_non_obstacle_height = 0
    skipped_collision_named = 0
    skipped_invisible = 0
    skipped_broad_aggregate_bounds: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        try:
            if not prim.IsA(UsdGeom.Gprim):
                continue
        except Exception:
            continue
        path_text = str(prim.GetPath())
        slug = path_text.lower()
        if any(token in slug for token in ("collision", "collisions", "collider")):
            skipped_collision_named += 1
            continue
        try:
            if UsdGeom.Imageable(prim).ComputeVisibility(Usd.TimeCode.Default()) == "invisible":
                skipped_invisible += 1
                continue
        except Exception:
            pass
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            minimum = [float(item) for item in box.GetMin()]
            maximum = [float(item) for item in box.GetMax()]
        except Exception:
            continue
        height = maximum[2] - minimum[2]
        x_extent = maximum[0] - minimum[0]
        y_extent = maximum[1] - minimum[1]
        if maximum[2] <= 0.08 and height <= 0.08:
            skipped_floor_like += 1
            continue
        if minimum[2] > 1.4 or maximum[2] < 0.05:
            skipped_non_obstacle_height += 1
            continue
        if x_extent > 15.0 or y_extent > 15.0:
            skipped_outlier += 1
            continue
        path_is_broad_aggregate = (
            (x_extent > 1.5 and y_extent > 1.5)
            or ("Kitchen_Wall" in path_text and x_extent > 1.0 and y_extent > 1.0)
            or (
                "Kitchen_Cabinet002/Kitchen_Cabinet002/Kitchen_Cabinet002" in path_text
            )
        )
        if path_is_broad_aggregate:
            skipped_broad_aggregate_bounds.append(
                {
                    "prim_path": path_text,
                    "bounds": {
                        "min": [round(float(item), 9) for item in minimum],
                        "max": [round(float(item), 9) for item in maximum],
                    },
                    "reason": "broad_aggregate_aabb_overapproximates_concave_or_compound_fixture",
                }
            )
            continue
        obstacles.append(
            {
                "prim_path": path_text,
                "bounds": {
                    "min": [round(float(item), 9) for item in minimum],
                    "max": [round(float(item), 9) for item in maximum],
                },
            }
        )
    return {
        "status": "available",
        "obstacles": obstacles,
        "obstacle_count": len(obstacles),
        "skipped_floor_like_count": skipped_floor_like,
        "skipped_outlier_bounds_count": skipped_outlier,
        "skipped_non_obstacle_height_count": skipped_non_obstacle_height,
        "skipped_collision_named_count": skipped_collision_named,
        "skipped_invisible_count": skipped_invisible,
        "skipped_broad_aggregate_bounds_count": len(skipped_broad_aggregate_bounds),
        "skipped_broad_aggregate_bounds_sample": skipped_broad_aggregate_bounds[:8],
        "blockers": [],
        "clearance_source": "usd_gprim_world_aabb_proxy",
        "broad_aggregate_aabbs_skipped": bool(skipped_broad_aggregate_bounds),
        "broad_aggregate_aabb_skip_is_not_collision_proof": True,
        "real_collision_geometry_validated": False,
    }


def _xy_clearance_to_aabb(
    *,
    xy: Sequence[float],
    bounds: Mapping[str, Any],
    footprint_radius_m: float,
) -> float:
    minimum = _vec3(bounds.get("min"), fallback=[0.0, 0.0, 0.0])
    maximum = _vec3(bounds.get("max"), fallback=[0.0, 0.0, 0.0])
    x = float(xy[0])
    y = float(xy[1])
    dx = max(minimum[0] - x, 0.0, x - maximum[0])
    dy = max(minimum[1] - y, 0.0, y - maximum[1])
    return math.hypot(dx, dy) - footprint_radius_m


def _evaluate_pose_clearance(
    *,
    pose: Mapping[str, Any],
    obstacles_manifest: Mapping[str, Any],
    footprint_radius_m: float = DEFAULT_UNITREE_G1_FOOTPRINT_RADIUS_M,
    clearance_margin_m: float = DEFAULT_UNITREE_G1_CLEARANCE_MARGIN_M,
) -> dict[str, Any]:
    xyz = _vec3(pose.get("xyz"), fallback=[0.0, 0.0, 0.0])
    obstacles = (
        obstacles_manifest.get("obstacles")
        if isinstance(obstacles_manifest.get("obstacles"), list)
        else []
    )
    if obstacles_manifest.get("status") != "available":
        return {
            "status": "not_evaluated",
            "accepted": False,
            "blockers": list(obstacles_manifest.get("blockers", [])) or ["placement_obstacles_unavailable"],
            "clearance_source": obstacles_manifest.get("clearance_source"),
            "real_collision_geometry_validated": False,
        }
    nearest: list[dict[str, Any]] = []
    min_clearance = float("inf")
    for obstacle in obstacles:
        bounds = obstacle.get("bounds") if isinstance(obstacle, Mapping) else None
        if not isinstance(bounds, Mapping):
            continue
        clearance = _xy_clearance_to_aabb(
            xy=xyz[:2],
            bounds=bounds,
            footprint_radius_m=footprint_radius_m,
        )
        min_clearance = min(min_clearance, clearance)
        if clearance < clearance_margin_m:
            nearest.append(
                {
                    "prim_path": obstacle.get("prim_path"),
                    "clearance_m": round(float(clearance), 6),
                    "bounds": bounds,
                }
            )
    nearest.sort(key=lambda row: float(row["clearance_m"]))
    accepted = not nearest
    return {
        "status": "passed" if accepted else "failed",
        "accepted": accepted,
        "pose_xyz": [round(float(item), 9) for item in xyz],
        "minimum_clearance_m": round(float(min_clearance), 6) if math.isfinite(min_clearance) else None,
        "required_clearance_margin_m": float(clearance_margin_m),
        "robot_footprint_radius_m": float(footprint_radius_m),
        "blocking_obstacle_count": len(nearest),
        "blocking_obstacles_sample": nearest[:8],
        "obstacle_count_evaluated": int(obstacles_manifest.get("obstacle_count") or len(obstacles)),
        "clearance_source": obstacles_manifest.get("clearance_source"),
        "real_collision_geometry_validated": False,
        "physics_contact_validated": False,
        "blockers": [] if accepted else ["robot_start_pose_static_usd_aabb_clearance_failed"],
    }


def _resolve_robot_start_pose(
    *,
    capture_root: Path,
    scene_asset: Path,
    scene_summary: Mapping[str, Any],
    task_id: str,
    target_object_id: str,
    target_pose: Mapping[str, Any],
    robot_start_pose: str | Sequence[float] | Mapping[str, Any] | None,
    episode: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    robot_fallback = [
        float(target_pose["xyz"][0]),
        float(target_pose["xyz"][1]) - 1.0,
        max(0.0, float(target_pose["xyz"][2]) - 1.0),
    ]
    candidates: list[dict[str, Any]] = []
    if robot_start_pose is not None:
        candidates.append(
            {
                "candidate_id": "provided_robot_start_pose",
                "pose": _pose(
                    robot_start_pose,
                    fallback_xyz=robot_fallback,
                    source="provided_robot_start_pose",
                ),
                "source": "provided_robot_start_pose",
                "selection_score": 0.0,
            }
        )
    if episode.get("robot_spawn_pose") is not None:
        candidates.append(
            {
                "candidate_id": "episode_robot_spawn_pose",
                "pose": _pose(
                    episode.get("robot_spawn_pose"),
                    fallback_xyz=robot_fallback,
                    source="episode_robot_spawn_pose",
                ),
                "source": "episode_robot_spawn_pose",
                "selection_score": 10.0,
            }
        )
    candidates.extend(
        _usd_sink_front_pose_candidates(
            scene_asset=scene_asset,
            target_pose=target_pose,
            target_object_id=target_object_id,
        )
    )
    candidates.extend(
        _target_ring_pose_candidates(target_pose=target_pose, scene_summary=scene_summary)
    )
    scenario_candidates, scenario_specs = _scenario_pose_candidates(
        capture_root=capture_root,
        scene_asset=scene_asset,
        task_id=task_id,
        target_object_id=target_object_id,
        target_pose=target_pose,
    )
    candidates.extend(
        {
            **candidate,
            "selection_score": 500.0 + float(candidate.get("selection_score") or 0.0),
        }
        for candidate in scenario_candidates
    )
    candidates.append(
        {
            "candidate_id": "target_relative_default",
            "pose": _pose(
                None,
                fallback_xyz=robot_fallback,
                source="target_relative_default_after_clearance_candidates",
            ),
            "source": "target_relative_default_after_clearance_candidates",
            "selection_score": 10000.0,
        }
    )
    candidates.sort(key=lambda row: (float(row["selection_score"]), _string(row.get("candidate_id"))))
    obstacles = _usd_placement_obstacle_aabbs(scene_asset)
    evaluated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        pose = dict(candidate["pose"])
        clearance = _evaluate_pose_clearance(pose=pose, obstacles_manifest=obstacles)
        row = {
            key: value
            for key, value in candidate.items()
            if key not in {"pose"}
        }
        row["pose"] = pose
        row["clearance_check"] = clearance
        row["accepted"] = bool(clearance.get("accepted"))
        evaluated.append(row)
        if row["accepted"]:
            selected = row
            break
    if selected is None:
        fallback_pose = dict(candidates[0]["pose"]) if candidates else _pose(
            None,
            fallback_xyz=robot_fallback,
            source="unresolved_robot_start_pose",
        )
        selected = {
            "candidate_id": "blocked_no_clear_robot_spawn_pose",
            "source": "blocked_no_clear_robot_spawn_pose",
            "pose": fallback_pose,
            "accepted": False,
            "clearance_check": {
                "status": "failed",
                "accepted": False,
                "blockers": ["blocked_no_clear_robot_spawn_pose_from_scene_metadata_or_target_ring"],
                "real_collision_geometry_validated": False,
                "physics_contact_validated": False,
            },
        }
    pose = dict(selected["pose"])
    pose["source"] = _string(selected.get("source")) or _string(pose.get("source"))
    pose["placement_candidate_id"] = selected.get("candidate_id")
    pose["placement_clearance_status"] = selected["clearance_check"].get("status")
    pose["placement_clearance_source"] = selected["clearance_check"].get("clearance_source")
    manifest = {
        "schema_version": "scene_robot_start_pose_resolution.v1",
        "status": "resolved" if selected.get("accepted") else "blocked",
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_source": selected.get("source"),
        "selected_pose": pose,
        "input_robot_start_pose_provided": robot_start_pose is not None,
        "input_robot_start_pose_rejected": bool(
            robot_start_pose is not None
            and selected.get("candidate_id") != "provided_robot_start_pose"
        ),
        "episode_robot_spawn_pose_available": episode.get("robot_spawn_pose") is not None,
        "target_relative_default_xyz": [round(float(item), 9) for item in robot_fallback],
        "scenario_specs": scenario_specs,
        "placement_obstacle_manifest": {
            key: value for key, value in obstacles.items() if key != "obstacles"
        },
        "evaluated_candidates": evaluated[:32],
        "selected_clearance_check": selected.get("clearance_check"),
        "real_collision_geometry_validated": False,
        "physics_contact_validated": False,
        "claim_boundary": {
            "robot_start_pose_selected_from_scene_metadata_or_usd_proxy": bool(
                selected.get("accepted")
            ),
            "static_usd_aabb_clearance_proxy_used": obstacles.get("status") == "available",
            "clearance_proxy_is_not_physics_contact_proof": True,
            "broad_aggregate_aabbs_skipped": bool(
                obstacles.get("broad_aggregate_aabbs_skipped")
            ),
            "broad_aggregate_aabb_skip_is_not_collision_proof": True,
            "scenario_metadata_is_execution_proof": False,
            "real_collision_geometry_validated": False,
            "physics_contact_validated": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
        },
        "blockers": [] if selected.get("accepted") else ["blocked_no_clear_robot_spawn_pose"],
    }
    return pose, manifest


def _write_head_pov_usd(
    *,
    scene_asset: Path,
    output_path: Path,
    robot_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
) -> tuple[bool, str | None]:
    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore[import-untyped]
    except Exception as exc:
        return False, f"pxr_import_failed:{type(exc).__name__}"
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    scene = stage.DefinePrim("/World/Scene", "Xform")
    scene.GetReferences().AddReference(str(scene_asset))
    camera = UsdGeom.Camera.Define(stage, "/World/BlueprintHeadPovCamera")
    robot_xyz = _float_list(robot_pose.get("xyz"), fallback=[0.0, -1.0, 0.0])[:3]
    target_xyz = _float_list(target_pose.get("xyz"), fallback=[0.0, 0.0, 1.0])[:3]
    head_height = max(1.35, target_xyz[2] + 0.75) if target_xyz[2] > 0.25 else 1.35
    eye = Gf.Vec3d(robot_xyz[0], robot_xyz[1], robot_xyz[2] + head_height)
    look_z = target_xyz[2] if target_xyz[2] > 0.25 else target_xyz[2] + 0.65
    look_at = Gf.Vec3d(target_xyz[0], target_xyz[1], look_z)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    view = Gf.Matrix4d().SetLookAt(eye, look_at, up)
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(20.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr((0.01, 1000.0))
    stage.Save()
    return True, None


def _renderer_availability() -> dict[str, Any]:
    commands = {
        "usdrecord": shutil.which("usdrecord") is not None,
        "usdview": shutil.which("usdview") is not None,
        "blender": shutil.which("blender") is not None,
    }
    packages: dict[str, bool] = {}
    for name in ("pxr", "open3d", "trimesh", "PIL", "mujoco"):
        try:
            __import__(name)
            packages[name] = True
        except Exception:
            packages[name] = False
    return {"commands": commands, "python_packages": packages}


def _rendered_image_content_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "contentful": False,
            "blockers": ["rendered_image_missing"],
        }
    try:
        from PIL import Image, ImageStat  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "contentful": True,
            "blockers": [f"rendered_image_content_check_unavailable:{type(exc).__name__}"],
        }
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            stat = ImageStat.Stat(rgb)
            sample_height = max(1, round(64 * rgb.height / max(1, rgb.width)))
            sample = rgb.resize((64, sample_height))
            pixels = list(sample.getdata())
        sampled_count = max(1, len(pixels))
        unique_sampled_color_count = len(set(pixels))
        non_white_count = sum(1 for pixel in pixels if any(channel < 245 for channel in pixel))
        non_black_count = sum(1 for pixel in pixels if any(channel > 10 for channel in pixel))
        contentful = bool(
            unique_sampled_color_count >= 4
            and max(float(value) for value in stat.stddev) >= 2.0
            and non_white_count / sampled_count >= 0.01
            and non_black_count / sampled_count >= 0.01
        )
        return {
            "path": str(path),
            "exists": True,
            "width": rgb.width,
            "height": rgb.height,
            "contentful": contentful,
            "mean_rgb": [round(float(value), 4) for value in stat.mean],
            "stddev_rgb": [round(float(value), 4) for value in stat.stddev],
            "unique_sampled_color_count": unique_sampled_color_count,
            "non_white_pixel_fraction_sampled": round(non_white_count / sampled_count, 6),
            "non_black_pixel_fraction_sampled": round(non_black_count / sampled_count, 6),
            "blockers": [] if contentful else ["rendered_image_blank_or_uniform"],
        }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "contentful": False,
            "blockers": [f"rendered_image_content_check_failed:{type(exc).__name__}"],
        }


def _safe_xml_name(value: str, *, fallback: str) -> str:
    out = []
    for char in value:
        out.append(char if char.isalnum() or char == "_" else "_")
    text = "_".join(part for part in "".join(out).split("_") if part)
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"_{text}"
    return text[:80]


def _asset_suffix(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _is_usd_asset(path: Path) -> bool:
    return _asset_suffix(path) in {"usd", "usda", "usdc"}


def _is_mjcf_asset(path: Path) -> bool:
    return _asset_suffix(path) in {"xml", "mjcf"}


def _mjcf_contains_unitree_g1(path: Path) -> bool:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return False
    if root.find(".//freejoint[@name='floating_base_joint']") is not None:
        return True
    if root.find(".//joint[@name='floating_base_joint']") is not None:
        return True
    for include in root.findall(".//include"):
        include_file = _string(include.get("file"))
        if include_file and "g1" in Path(include_file).name.lower():
            return True
    return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_unitree_g1_mjcf_path(value: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(value))
    for env_name in ("BLUEPRINT_MUJOCO_G1_XML", "UNITREE_G1_MJCF_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))
    model_root = os.environ.get("BLUEPRINT_MUJOCO_G1_MODEL_ROOT")
    if model_root:
        candidates.append(Path(model_root) / "g1.xml")
    candidates.append(_repo_root() / "output" / "external_assets" / "mujoco_menagerie" / "unitree_g1" / "g1.xml")
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = (_repo_root() / resolved).resolve()
        if resolved.is_file():
            return resolved
    return None


def _vec3(value: Any, *, fallback: Sequence[float]) -> list[float]:
    values = _float_list(value, fallback=fallback)
    if len(values) < 3:
        return list(fallback)[:3]
    return values[:3]


def _format_float_list(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _mjcf_compiler_dir(root: ET.Element, xml_path: Path, *, attr_name: str) -> Path:
    compiler = root.find("compiler")
    directory = _string(compiler.get(attr_name)) if compiler is not None else ""
    base = xml_path.parent / directory if directory else xml_path.parent
    return base.resolve()


def _camera_axes_for_lookat(
    *, eye: Sequence[float], look: Sequence[float]
) -> tuple[list[float], list[float]]:
    forward = [float(look[index]) - float(eye[index]) for index in range(3)]
    length = math.sqrt(sum(component * component for component in forward))
    if length <= 1e-9:
        forward = [0.0, 1.0, -0.2]
        length = math.sqrt(sum(component * component for component in forward))
    forward = [component / length for component in forward]
    up = [0.0, 0.0, 1.0]
    right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ]
    right_length = math.sqrt(sum(component * component for component in right))
    if right_length <= 1e-9:
        right = [1.0, 0.0, 0.0]
        right_length = 1.0
    right = [component / right_length for component in right]
    camera_up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    up_length = math.sqrt(sum(component * component for component in camera_up))
    if up_length <= 1e-9:
        camera_up = [0.0, 0.0, 1.0]
    else:
        camera_up = [component / up_length for component in camera_up]
    return right, camera_up


def _camera_attrs_for_lookat(
    *, name: str, eye: Sequence[float], look: Sequence[float], fovy: float
) -> dict[str, str]:
    right, camera_up = _camera_axes_for_lookat(eye=eye, look=look)
    return {
        "name": name,
        "pos": _format_float_list([float(item) for item in eye[:3]]),
        "xyaxes": _format_float_list([*right, *camera_up]),
        "fovy": f"{float(fovy):.9g}",
    }


def _pose_yaw(robot_pose: Mapping[str, Any]) -> float:
    return float(_vec3(robot_pose.get("rpy"), fallback=[0.0, 0.0, 0.0])[2])


def _forward_offset_xy(robot_pose: Mapping[str, Any], distance_m: float) -> tuple[float, float]:
    yaw = _pose_yaw(robot_pose)
    return math.cos(yaw) * distance_m, math.sin(yaw) * distance_m


def _mujoco_camera_from_poses(
    *,
    mujoco_module: Any,
    robot_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
    video_camera: str = "head_pov",
) -> Any:
    camera = mujoco_module.MjvCamera()
    camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    robot_xyz = _vec3(robot_pose.get("xyz"), fallback=[0.0, -1.0, 0.0])
    target_xyz = _vec3(target_pose.get("xyz"), fallback=[0.0, 0.0, 1.0])
    camera_role = str(video_camera or "head_pov").strip()
    if camera_role == "torso_pov":
        camera_height = max(1.0, target_xyz[2] + 0.35) if target_xyz[2] > 0.25 else 1.0
        look_z = target_xyz[2] + 0.18 if target_xyz[2] > 0.25 else target_xyz[2] + 0.55
        forward_offset = 0.16
    else:
        camera_height = max(1.35, target_xyz[2] + 0.75) if target_xyz[2] > 0.25 else 1.35
        look_z = target_xyz[2] + 0.24 if target_xyz[2] > 0.25 else target_xyz[2] + 0.75
        forward_offset = 0.22
    offset_x, offset_y = _forward_offset_xy(robot_pose, forward_offset)
    eye = [robot_xyz[0] + offset_x, robot_xyz[1] + offset_y, robot_xyz[2] + camera_height]
    look = [target_xyz[0], target_xyz[1], look_z]
    direction = [look[index] - eye[index] for index in range(3)]
    horizontal = math.hypot(direction[0], direction[1])
    distance = max(0.25, math.sqrt(sum(component * component for component in direction)))
    camera.lookat[:] = look
    camera.distance = distance
    camera.azimuth = math.degrees(math.atan2(direction[0], direction[1] or 1e-9))
    camera.elevation = -math.degrees(math.atan2(direction[2], horizontal or 1e-9))
    return camera


def _rgba_from_value(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except TypeError:
        return None
    if len(values) < 3:
        return None
    return (
        max(0.0, min(1.0, values[0])),
        max(0.0, min(1.0, values[1])),
        max(0.0, min(1.0, values[2])),
        max(0.0, min(1.0, values[3])) if len(values) > 3 else 1.0,
    )


def _stable_rgba(name: str) -> tuple[float, float, float, float]:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return (
        0.35 + ((seed >> 0) % 100) / 220.0,
        0.35 + ((seed >> 7) % 100) / 220.0,
        0.35 + ((seed >> 14) % 100) / 220.0,
        1.0,
    )


def _usd_material_info(material: Any) -> dict[str, Any]:
    if not material:
        return {"texture_path": None, "rgba": None}
    try:
        from pxr import Sdf, Usd, UsdShade  # type: ignore[import-untyped]
    except Exception:
        return {"texture_path": None, "rgba": None}

    asset_candidates: list[tuple[int, str, str]] = []
    rgba: tuple[float, float, float, float] | None = None
    prim = material.GetPrim()
    for child in Usd.PrimRange(prim):
        if child == prim:
            continue
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        for shader_input in shader.GetInputs():
            value = shader_input.Get()
            input_name = shader_input.GetBaseName().lower()
            if rgba is None and input_name in {"diffusecolor", "basecolor", "color"}:
                rgba = _rgba_from_value(value)
            if isinstance(value, Sdf.AssetPath):
                resolved = value.resolvedPath or value.path
                if not resolved:
                    continue
                haystack = f"{input_name} {Path(value.path).name.lower()}"
                score = 10
                if any(token in haystack for token in ("diffuse", "albedo", "basecolor", "_bc")):
                    score = 0
                elif "color" in haystack:
                    score = 1
                elif "normal" in haystack:
                    score = 8
                elif any(token in haystack for token in ("rough", "metal", "spec")):
                    score = 9
                asset_candidates.append((score, value.path, resolved))
    asset_candidates.sort(key=lambda row: row[0])
    texture_path = None
    for _score, _authored, resolved in asset_candidates:
        candidate = Path(resolved)
        if candidate.is_file():
            texture_path = candidate
            break
    return {
        "texture_path": str(texture_path) if texture_path else None,
        "rgba": rgba,
    }


def _usd_mesh_uv_lookup(mesh: Any) -> tuple[Any, Any, str]:
    try:
        from pxr import UsdGeom  # type: ignore[import-untyped]
    except Exception:
        return None, None, ""
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
    if not primvar or not primvar.HasValue():
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("UVMap")
    if not primvar or not primvar.HasValue():
        return None, None, ""
    return primvar.Get(), primvar.GetIndices(), str(primvar.GetInterpolation())


def _usd_uv_for_face_vertex(
    *,
    values: Any,
    indices: Any,
    interpolation: str,
    face_vertex_index: int,
    vertex_index: int,
    fallback_uv: tuple[float, float],
) -> tuple[float, float]:
    if values is None:
        return fallback_uv
    try:
        if interpolation == "faceVarying":
            uv_index = int(indices[face_vertex_index]) if indices is not None else face_vertex_index
        elif interpolation == "vertex":
            uv_index = int(indices[vertex_index]) if indices is not None else vertex_index
        else:
            uv_index = int(indices[0]) if indices is not None and len(indices) else 0
        uv = values[uv_index]
        return float(uv[0]), float(uv[1])
    except Exception:
        return fallback_uv


def _bounds_from_mapping(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(value, Mapping):
        return None
    minimum = _vec3(value.get("min"), fallback=[0.0, 0.0, 0.0])
    maximum = _vec3(value.get("max"), fallback=[0.0, 0.0, 0.0])
    if minimum == maximum:
        return None
    return (tuple(minimum), tuple(maximum))


def _write_bbox_proxy_obj(
    *,
    output_path: Path,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    estimated_source_triangle_count: int,
    max_triangles: int,
) -> dict[str, Any]:
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    epsilon = 1e-4
    if abs(max_x - min_x) < epsilon:
        min_x -= epsilon
        max_x += epsilon
    if abs(max_y - min_y) < epsilon:
        min_y -= epsilon
        max_y += epsilon
    if abs(max_z - min_z) < epsilon:
        min_z -= epsilon
        max_z += epsilon
    vertices = [
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (max_x, max_y, min_z),
        (min_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z),
        (min_x, max_y, max_z),
    ]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    faces = [
        ((1, 1), (2, 2), (3, 3)),
        ((1, 1), (3, 3), (4, 4)),
        ((5, 1), (7, 3), (6, 2)),
        ((5, 1), (8, 4), (7, 3)),
        ((1, 1), (6, 3), (2, 2)),
        ((1, 1), (5, 4), (6, 3)),
        ((4, 1), (3, 2), (7, 3)),
        ((4, 1), (7, 3), (8, 4)),
        ((1, 1), (4, 2), (8, 3)),
        ((1, 1), (8, 3), (5, 4)),
        ((2, 1), (6, 2), (7, 3)),
        ((2, 1), (7, 3), (3, 4)),
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Blueprint bbox proxy mesh exported from USD for MuJoCo rendering\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uvs:
            handle.write(f"vt {u:.9g} {v:.9g}\n")
        for face in faces:
            handle.write(
                "f "
                + " ".join(f"{vertex_index}/{uv_index}" for vertex_index, uv_index in face)
                + "\n"
            )
    return {
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "uv_count": len(uvs),
        "has_uvs": True,
        "triangle_limit": int(max_triangles),
        "triangle_limit_reached": estimated_source_triangle_count > max(1, int(max_triangles)),
        "bbox_proxy_used": True,
        "estimated_source_triangle_count": int(estimated_source_triangle_count),
        "world_bounds": {
            "min": [round(float(item), 9) for item in (min_x, min_y, min_z)],
            "max": [round(float(item), 9) for item in (max_x, max_y, max_z)],
        },
    }


def _copy_texture_for_mujoco(
    *,
    source_path: Path,
    texture_dir: Path,
    destination_stem: str,
) -> dict[str, Any] | None:
    destination_name = f"{destination_stem}.png"
    destination_path = texture_dir / destination_name
    try:
        from PIL import Image  # type: ignore[import-untyped]

        with Image.open(source_path) as image:
            mode = "RGBA" if "A" in image.getbands() else "RGB"
            image.convert(mode).save(destination_path)
        return {
            "source_path": str(source_path),
            "copied_path": str(destination_path),
            "destination_name": destination_name,
            "converted_to_png_for_mujoco": True,
        }
    except Exception:
        if source_path.suffix.lower() != ".png":
            return None
        try:
            shutil.copy2(source_path, destination_path)
        except Exception:
            return None
        return {
            "source_path": str(source_path),
            "copied_path": str(destination_path),
            "destination_name": destination_name,
            "converted_to_png_for_mujoco": False,
        }


def _write_unitree_g1_mjcf_for_composition(
    *,
    source_mjcf: Path,
    output_mjcf: Path,
    robot_pose: Mapping[str, Any],
) -> dict[str, Any]:
    tree = ET.parse(source_mjcf)
    root = tree.getroot()
    assets_dir = _mjcf_compiler_dir(root, source_mjcf, attr_name="meshdir")
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(assets_dir))
    for mesh in root.findall(".//mesh"):
        mesh_file = _string(mesh.get("file"))
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.set("file", str((assets_dir / mesh_file).resolve()))
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("unitree_g1_pelvis_body_not_found")
    original_pos = _vec3(pelvis.get("pos"), fallback=[0.0, 0.0, 0.793])
    robot_xyz = _vec3(robot_pose.get("xyz"), fallback=[0.0, -1.0, 0.0])
    robot_rpy = _vec3(robot_pose.get("rpy"), fallback=[0.0, 0.0, 0.0])
    pelvis_pos = [robot_xyz[0], robot_xyz[1], robot_xyz[2] + original_pos[2]]
    pelvis.set("pos", _format_float_list(pelvis_pos))
    pelvis.set("euler", _format_float_list(robot_rpy))
    output_mjcf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_mjcf, encoding="utf-8", xml_declaration=True)
    return {
        "source_unitree_g1_mjcf_path": str(source_mjcf),
        "generated_unitree_g1_mjcf_path": str(output_mjcf),
        "unitree_g1_root_body_name": "pelvis",
        "unitree_g1_root_body_pos": pelvis_pos,
        "unitree_g1_root_body_euler": robot_rpy,
        "unitree_g1_original_root_body_pos": original_pos,
    }


def _copy_scene_mjcf_assets_and_worldbody(
    *,
    scene_root: ET.Element,
    scene_mjcf_path: Path,
    combined_asset: ET.Element,
    combined_worldbody: ET.Element,
) -> dict[str, Any]:
    mesh_dir = _mjcf_compiler_dir(scene_root, scene_mjcf_path, attr_name="meshdir")
    texture_dir = _mjcf_compiler_dir(scene_root, scene_mjcf_path, attr_name="texturedir")
    rename_maps: dict[str, dict[str, str]] = {
        "mesh": {},
        "material": {},
        "texture": {},
    }
    asset_root = scene_root.find("asset")
    scene_asset_count = 0
    if asset_root is not None:
        for child in list(asset_root):
            copied = copy.deepcopy(child)
            if copied.tag in rename_maps:
                name = _string(copied.get("name"))
                if name:
                    new_name = _safe_xml_name(f"scene_{name}", fallback=f"scene_{copied.tag}")
                    rename_maps[copied.tag][name] = new_name
                    copied.set("name", new_name)
            if copied.tag == "mesh":
                mesh_file = _string(copied.get("file"))
                if mesh_file and not Path(mesh_file).is_absolute():
                    copied.set("file", str((mesh_dir / mesh_file).resolve()))
            if copied.tag == "texture":
                texture_file = _string(copied.get("file"))
                if texture_file and not Path(texture_file).is_absolute():
                    copied.set("file", str((texture_dir / texture_file).resolve()))
            if copied.tag == "material":
                texture_name = _string(copied.get("texture"))
                if texture_name in rename_maps["texture"]:
                    copied.set("texture", rename_maps["texture"][texture_name])
            combined_asset.append(copied)
            scene_asset_count += 1
    worldbody_root = scene_root.find("worldbody")
    scene_geom_count = 0
    if worldbody_root is not None:
        for child in list(worldbody_root):
            copied = copy.deepcopy(child)
            for geom in [copied, *list(copied.findall(".//geom"))]:
                if geom.tag != "geom":
                    continue
                name = _string(geom.get("name"))
                if name:
                    geom.set("name", _safe_xml_name(f"scene_{name}", fallback="scene_geom"))
                mesh_name = _string(geom.get("mesh"))
                if mesh_name in rename_maps["mesh"]:
                    geom.set("mesh", rename_maps["mesh"][mesh_name])
                material_name = _string(geom.get("material"))
                if material_name in rename_maps["material"]:
                    geom.set("material", rename_maps["material"][material_name])
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                scene_geom_count += 1
            combined_worldbody.append(copied)
    return {
        "scene_asset_count": scene_asset_count,
        "scene_geom_count": scene_geom_count,
        "scene_visual_geom_collision_disabled": True,
        "scene_asset_name_prefix": "scene_",
    }


def _render_mujoco_named_camera_frame(
    *,
    scene_model_path: Path,
    camera_name: str,
    output_path: Path,
    width: int = DEFAULT_POLICY_RENDER_WIDTH,
    height: int = DEFAULT_POLICY_RENDER_HEIGHT,
) -> dict[str, Any]:
    try:
        import mujoco  # type: ignore[import-untyped]
        from PIL import Image  # type: ignore[import-untyped]
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"mujoco_named_camera_import_failed:{type(exc).__name__}"]}
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_model_path))
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
            return {
                "status": "blocked",
                "blockers": [f"mujoco_named_camera_missing:{camera_name}"],
            }
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            renderer.update_scene(data, camera=camera_name)
            frame = renderer.render()
        finally:
            renderer.close()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).convert("RGB").save(output_path, quality=92)
        content_summary = _rendered_image_content_summary(output_path)
        return {
            "status": "completed" if content_summary.get("contentful") else "blocked",
            "camera_name": camera_name,
            "frame_path": str(output_path),
            "render_width": width,
            "render_height": height,
            "rendered_image_content_summary": content_summary,
            "blockers": list(content_summary.get("blockers", [])),
        }
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"mujoco_named_camera_render_failed:{type(exc).__name__}"]}


def _compose_scene_with_unitree_g1_mjcf(
    *,
    scene_mjcf_path: Path,
    output_dir: Path,
    generated_at: str,
    robot_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
    g1_mjcf_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_g1 = _resolve_unitree_g1_mjcf_path(g1_mjcf_path)
    composition_dir = output_dir / "combined_mujoco_scene"
    composition_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = composition_dir / "scene_unitree_g1_mujoco_composition.json"
    if resolved_g1 is None:
        manifest = {
            "schema_version": "scene_unitree_g1_mujoco_composition.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "blockers": ["missing_unitree_g1_mjcf_asset"],
            "combined_mjcf_path": None,
            "unitree_g1_asset_spawned": False,
            "physics_contact_validated": False,
        }
        write_json(manifest_path, manifest)
        return manifest
    try:
        scene_tree = ET.parse(scene_mjcf_path)
        scene_root = scene_tree.getroot()
        generated_g1_mjcf = composition_dir / "unitree_g1_absolute_meshes_placed.xml"
        g1_info = _write_unitree_g1_mjcf_for_composition(
            source_mjcf=resolved_g1,
            output_mjcf=generated_g1_mjcf,
            robot_pose=robot_pose,
        )
        robot_xyz = _vec3(robot_pose.get("xyz"), fallback=[0.0, -1.0, 0.0])
        target_xyz = _vec3(target_pose.get("xyz"), fallback=[0.0, 0.0, 1.0])
        head_height = max(1.35, target_xyz[2] + 0.75) if target_xyz[2] > 0.25 else 1.35
        head_offset_x, head_offset_y = _forward_offset_xy(robot_pose, 0.22)
        head_eye = [
            robot_xyz[0] + head_offset_x,
            robot_xyz[1] + head_offset_y,
            robot_xyz[2] + head_height,
        ]
        look_z = target_xyz[2] + 0.24 if target_xyz[2] > 0.25 else target_xyz[2] + 0.75
        target_look = [target_xyz[0], target_xyz[1], look_z]
        overview_eye = [
            robot_xyz[0] - 2.4,
            min(robot_xyz[1], target_xyz[1]) - 2.4,
            max(2.4, target_xyz[2] + 2.1),
        ]
        overview_look = [
            (robot_xyz[0] + target_xyz[0]) / 2.0,
            (robot_xyz[1] + target_xyz[1]) / 2.0,
            max(0.8, target_xyz[2]),
        ]
        root = ET.Element("mujoco", {"model": "blueprint_kitchen_unitree_g1_visual_scene"})
        ET.SubElement(root, "include", {"file": str(generated_g1_mjcf)})
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "headlight",
            {
                "diffuse": "0.85 0.85 0.82",
                "ambient": "0.22 0.22 0.2",
                "specular": "0.35 0.35 0.35",
            },
        )
        ET.SubElement(
            visual,
            "global",
            {
                "offwidth": str(DEFAULT_POLICY_RENDER_WIDTH),
                "offheight": str(DEFAULT_POLICY_RENDER_HEIGHT),
                "azimuth": "140",
                "elevation": "-20",
            },
        )
        ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "200"})
        asset = ET.SubElement(root, "asset")
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "blueprint_key_light",
                "pos": _format_float_list([target_xyz[0] - 1.5, target_xyz[1] - 2.0, 4.0]),
                "dir": "0.25 0.35 -1",
                "directional": "true",
                "diffuse": "0.9 0.86 0.78",
            },
        )
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "blueprint_fill_light",
                "pos": _format_float_list([target_xyz[0] + 2.0, target_xyz[1] + 1.5, 2.4]),
                "diffuse": "0.35 0.42 0.55",
            },
        )
        ET.SubElement(
            worldbody,
            "camera",
            _camera_attrs_for_lookat(
                name="blueprint_head_pov",
                eye=head_eye,
                look=target_look,
                fovy=55.0,
            ),
        )
        torso_height = max(1.0, target_xyz[2] + 0.35) if target_xyz[2] > 0.25 else 1.0
        torso_offset_x, torso_offset_y = _forward_offset_xy(robot_pose, 0.16)
        torso_eye = [
            robot_xyz[0] + torso_offset_x,
            robot_xyz[1] + torso_offset_y,
            robot_xyz[2] + torso_height,
        ]
        torso_look_z = target_xyz[2] + 0.18 if target_xyz[2] > 0.25 else target_xyz[2] + 0.55
        ET.SubElement(
            worldbody,
            "camera",
            _camera_attrs_for_lookat(
                name="blueprint_torso_pov",
                eye=torso_eye,
                look=[target_xyz[0], target_xyz[1], torso_look_z],
                fovy=62.0,
            ),
        )
        ET.SubElement(
            worldbody,
            "camera",
            _camera_attrs_for_lookat(
                name="blueprint_overview",
                eye=overview_eye,
                look=overview_look,
                fovy=60.0,
            ),
        )
        scene_copy_info = _copy_scene_mjcf_assets_and_worldbody(
            scene_root=scene_root,
            scene_mjcf_path=scene_mjcf_path,
            combined_asset=asset,
            combined_worldbody=worldbody,
        )
        combined_mjcf_path = composition_dir / "blueprint_kitchen_unitree_g1_scene.xml"
        ET.ElementTree(root).write(combined_mjcf_path, encoding="utf-8", xml_declaration=True)
        import mujoco  # type: ignore[import-untyped]

        model = mujoco.MjModel.from_xml_path(str(combined_mjcf_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        floating_base_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "floating_base_joint",
        )
        scene_geom_indices = [
            index
            for index in range(int(model.ngeom))
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("scene_")
        ]
        scene_collision_enabled = any(
            int(model.geom_contype[index]) != 0 or int(model.geom_conaffinity[index]) != 0
            for index in scene_geom_indices
        )
        overview_render = _render_mujoco_named_camera_frame(
            scene_model_path=combined_mjcf_path,
            camera_name="blueprint_overview",
            output_path=composition_dir / "blueprint_kitchen_unitree_g1_overview.jpg",
        )
        manifest = {
            "schema_version": "scene_unitree_g1_mujoco_composition.v1",
            "generated_at": generated_at,
            "status": "completed" if floating_base_joint_id >= 0 else "blocked",
            "blockers": [] if floating_base_joint_id >= 0 else ["unitree_g1_floating_base_joint_not_found"],
            "source_scene_mjcf_path": str(scene_mjcf_path),
            "combined_mjcf_path": str(combined_mjcf_path),
            **g1_info,
            **scene_copy_info,
            "robot_start_pose": dict(robot_pose),
            "robot_start_pose_source": robot_pose.get("source"),
            "robot_start_pose_placement_candidate_id": robot_pose.get("placement_candidate_id"),
            "robot_start_pose_static_clearance_status": robot_pose.get(
                "placement_clearance_status"
            ),
            "robot_start_pose_static_clearance_source": robot_pose.get(
                "placement_clearance_source"
            ),
            "unitree_g1_asset_spawned": floating_base_joint_id >= 0,
            "unitree_g1_floating_base_joint_found": floating_base_joint_id >= 0,
            "combined_mujoco_model_mesh_count": int(model.nmesh),
            "combined_mujoco_model_texture_count": int(model.ntex),
            "combined_mujoco_model_geom_count": int(model.ngeom),
            "scene_visual_geom_count_loaded": len(scene_geom_indices),
            "scene_visual_collision_enabled": scene_collision_enabled,
            "explicit_lights_authored": True,
            "explicit_cameras_authored": True,
            "head_pov_camera_name": "blueprint_head_pov",
            "torso_pov_camera_name": "blueprint_torso_pov",
            "overview_camera_name": "blueprint_overview",
            "overview_render": overview_render,
            "claim_boundary": {
                "combined_scene_visualizes_unitree_g1_asset": floating_base_joint_id >= 0,
                "scene_visual_mjcf_is_generated_from_usd_for_rendering": True,
                "scene_visual_collision_disabled": not scene_collision_enabled,
                "scene_collision_geometry_validated": False,
                "static_usd_aabb_clearance_proxy_used": bool(
                    robot_pose.get("placement_clearance_source")
                ),
                "static_usd_aabb_clearance_proxy_passed": (
                    robot_pose.get("placement_clearance_status") == "passed"
                ),
                "physics_contact_validated": False,
                "physical_robot_sensor_proof": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
                "safety_validation_proven": False,
                "real_world_manipulation_success_proven": False,
            },
        }
        write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest = {
            "schema_version": "scene_unitree_g1_mujoco_composition.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "blockers": [f"scene_unitree_g1_mjcf_composition_failed:{type(exc).__name__}"],
            "combined_mjcf_path": None,
            "unitree_g1_asset_spawned": False,
            "physics_contact_validated": False,
        }
        write_json(manifest_path, manifest)
        return manifest


def _write_obj_from_usd_mesh(
    *,
    mesh: Any,
    output_path: Path,
    max_triangles: int = DEFAULT_USD_VISUAL_MJCF_MAX_TRIANGLES_PER_MESH,
    world_bounds: Any | None = None,
) -> dict[str, Any] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-untyped]
    except Exception:
        return None
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not counts or not indices:
        return None
    triangle_limit = max(1, int(max_triangles))
    estimated_source_triangle_count = sum(max(0, int(face_count) - 2) for face_count in counts)
    mapped_bounds = _bounds_from_mapping(world_bounds)
    if estimated_source_triangle_count > triangle_limit and mapped_bounds is not None:
        return _write_bbox_proxy_obj(
            output_path=output_path,
            bounds=mapped_bounds,
            estimated_source_triangle_count=estimated_source_triangle_count,
            max_triangles=triangle_limit,
        )
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    transform = UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_points: list[tuple[float, float, float]] = []
    for point in points:
        transformed = transform.Transform(point)
        world_points.append((float(transformed[0]), float(transformed[1]), float(transformed[2])))
    min_x = min(point[0] for point in world_points)
    max_x = max(point[0] for point in world_points)
    min_y = min(point[1] for point in world_points)
    max_y = max(point[1] for point in world_points)
    range_x = max(1e-6, max_x - min_x)
    range_y = max(1e-6, max_y - min_y)
    if estimated_source_triangle_count > triangle_limit:
        bounds = (
            (min_x, min_y, min(point[2] for point in world_points)),
            (max_x, max_y, max(point[2] for point in world_points)),
        )
        return _write_bbox_proxy_obj(
            output_path=output_path,
            bounds=bounds,
            estimated_source_triangle_count=estimated_source_triangle_count,
            max_triangles=triangle_limit,
        )
    uv_values, uv_indices, uv_interpolation = _usd_mesh_uv_lookup(mesh)
    obj_vertices: list[tuple[float, float, float]] = []
    obj_uvs: list[tuple[float, float]] = []
    obj_faces: list[tuple[int, int, int]] = []
    cursor = 0
    for face_count in counts:
        face_refs: list[int] = []
        for local_index in range(int(face_count)):
            face_vertex_index = cursor + local_index
            vertex_index = int(indices[face_vertex_index])
            point = world_points[vertex_index]
            fallback_uv = ((point[0] - min_x) / range_x, (point[1] - min_y) / range_y)
            uv = _usd_uv_for_face_vertex(
                values=uv_values,
                indices=uv_indices,
                interpolation=uv_interpolation,
                face_vertex_index=face_vertex_index,
                vertex_index=vertex_index,
                fallback_uv=fallback_uv,
            )
            obj_vertices.append(point)
            obj_uvs.append(uv)
            face_refs.append(len(obj_vertices))
        cursor += int(face_count)
        if len(face_refs) < 3:
            continue
        for index in range(1, len(face_refs) - 1):
            obj_faces.append((face_refs[0], face_refs[index], face_refs[index + 1]))
            if len(obj_faces) >= triangle_limit:
                break
        if len(obj_faces) >= triangle_limit:
            break
    if not obj_faces:
        return None
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Blueprint visual MJCF mesh exported from USD for MuJoCo rendering\n")
        for x, y, z in obj_vertices:
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in obj_uvs:
            handle.write(f"vt {u:.9g} {v:.9g}\n")
        for a, b, c in obj_faces:
            handle.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    return {
        "vertex_count": len(obj_vertices),
        "triangle_count": len(obj_faces),
        "uv_count": len(obj_uvs),
        "has_uvs": bool(obj_uvs),
        "triangle_limit": int(triangle_limit),
        "triangle_limit_reached": len(obj_faces) >= triangle_limit,
        "bbox_proxy_used": False,
        "estimated_source_triangle_count": int(estimated_source_triangle_count),
    }


def _usd_mesh_visual_candidates(
    *,
    stage: Any,
    target_pose: Mapping[str, Any],
    robot_pose: Mapping[str, Any],
    max_meshes: int,
) -> list[dict[str, Any]]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-untyped]
    except Exception:
        return []
    target_xyz = _vec3(target_pose.get("xyz"), fallback=[0.0, 0.0, 1.0])
    robot_xyz = _vec3(robot_pose.get("xyz"), fallback=[0.0, -1.0, 0.0])
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    candidates: list[dict[str, Any]] = []
    skipped_collision_count = 0
    skipped_invisible_count = 0
    skipped_degenerate_mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        path_text = str(prim.GetPath())
        slug = path_text.lower()
        if any(token in slug for token in ("collision", "collisions", "collider")):
            skipped_collision_count += 1
            continue
        try:
            if UsdGeom.Imageable(prim).ComputeVisibility(Usd.TimeCode.Default()) == "invisible":
                skipped_invisible_count += 1
                continue
        except Exception:
            pass
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            minimum = [float(item) for item in box.GetMin()]
            maximum = [float(item) for item in box.GetMax()]
        except Exception:
            minimum = [0.0, 0.0, 0.0]
            maximum = [0.0, 0.0, 0.0]
        center = [(a + b) / 2.0 for a, b in zip(minimum, maximum)]
        dimensions = [maximum[index] - minimum[index] for index in range(3)]
        if min(dimensions) <= 1e-6 and (
            max(dimensions) >= 2.0 or Path(path_text).name.lower() == "plane"
        ):
            skipped_degenerate_mesh_count += 1
            continue
        target_distance = math.sqrt(sum((center[index] - target_xyz[index]) ** 2 for index in range(3)))
        robot_distance = math.sqrt(sum((center[index] - robot_xyz[index]) ** 2 for index in range(3)))
        candidates.append(
            {
                "prim": prim,
                "prim_path": path_text,
                "center": center,
                "bounds": {"min": minimum, "max": maximum},
                "dimensions": dimensions,
                "target_distance_m": target_distance,
                "robot_distance_m": robot_distance,
                "priority": min(target_distance, robot_distance),
            }
        )
    candidates.sort(key=lambda row: (float(row["priority"]), str(row["prim_path"])))
    selected = candidates[: max(1, int(max_meshes))]
    for row in selected:
        row["total_candidate_count"] = len(candidates)
        row["skipped_collision_mesh_count"] = skipped_collision_count
        row["skipped_invisible_mesh_count"] = skipped_invisible_count
        row["skipped_degenerate_mesh_count"] = skipped_degenerate_mesh_count
    return selected


def _build_visual_mjcf_from_usd(
    *,
    scene_asset: Path,
    output_dir: Path,
    generated_at: str,
    target_pose: Mapping[str, Any],
    robot_pose: Mapping[str, Any],
    max_meshes: int = DEFAULT_USD_VISUAL_MJCF_MAX_MESHES,
) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom, UsdShade  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "schema_version": "usd_to_mujoco_visual_mjcf.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "blockers": [f"pxr_import_failed:{type(exc).__name__}"],
        }
    stage = Usd.Stage.Open(str(scene_asset))
    if stage is None:
        return {
            "schema_version": "usd_to_mujoco_visual_mjcf.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "blockers": ["usd_stage_open_failed"],
        }
    visual_dir = output_dir / "mujoco_visual_scene"
    mesh_dir = visual_dir / "visual_meshes"
    texture_dir = visual_dir / "textures"
    ensure_dir(mesh_dir)
    ensure_dir(texture_dir)
    root = ET.Element("mujoco", {"model": "blueprint_usd_visual_scene"})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "meshdir": "visual_meshes", "texturedir": "textures"},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {
            "offwidth": str(DEFAULT_POLICY_RENDER_WIDTH),
            "offheight": str(DEFAULT_POLICY_RENDER_HEIGHT),
        },
    )
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    mesh_rows: list[dict[str, Any]] = []
    texture_rows: list[dict[str, Any]] = []
    copied_textures: dict[str, str] = {}
    skipped_texture_sources: set[str] = set()
    candidates = _usd_mesh_visual_candidates(
        stage=stage,
        target_pose=target_pose,
        robot_pose=robot_pose,
        max_meshes=max_meshes,
    )
    for mesh_index, candidate in enumerate(candidates):
        prim = candidate["prim"]
        mesh = UsdGeom.Mesh(prim)
        mesh_name = _safe_xml_name(
            f"{mesh_index:04d}_{str(prim.GetPath()).strip('/').replace('/', '_')}",
            fallback=f"mesh_{mesh_index:04d}",
        )
        obj_path = mesh_dir / f"{mesh_name}.obj"
        obj_info = _write_obj_from_usd_mesh(
            mesh=mesh,
            output_path=obj_path,
            world_bounds=candidate.get("bounds"),
        )
        if obj_info is None:
            continue
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        material_info = _usd_material_info(material)
        texture_file = material_info.get("texture_path")
        texture_name = None
        if texture_file:
            texture_source = Path(str(texture_file))
            if texture_source.is_file():
                texture_source_key = str(texture_source)
                destination_name = copied_textures.get(texture_source_key)
                if destination_name is None and texture_source_key not in skipped_texture_sources:
                    texture_copy = _copy_texture_for_mujoco(
                        source_path=texture_source,
                        texture_dir=texture_dir,
                        destination_stem=f"texture_{len(copied_textures):04d}",
                    )
                    if texture_copy is None:
                        skipped_texture_sources.add(texture_source_key)
                    else:
                        destination_name = str(texture_copy["destination_name"])
                        texture_rows.append(texture_copy)
                        copied_textures[texture_source_key] = destination_name
                if destination_name:
                    texture_name = _safe_xml_name(
                        f"tex_{Path(destination_name).stem}",
                        fallback=f"tex_{mesh_index:04d}",
                    )
                if texture_name and not any(
                    child.get("name") == texture_name for child in asset.findall("texture")
                ):
                    ET.SubElement(
                        asset,
                        "texture",
                        {
                            "name": texture_name,
                            "type": "2d",
                            "file": destination_name,
                        },
                    )
        rgba = material_info.get("rgba") or _stable_rgba(mesh_name)
        material_name = _safe_xml_name(f"mat_{mesh_name}", fallback=f"mat_{mesh_index:04d}")
        material_attrs = {
            "name": material_name,
            "rgba": " ".join(f"{float(component):.6g}" for component in rgba),
        }
        if texture_name:
            material_attrs["texture"] = texture_name
        ET.SubElement(asset, "material", material_attrs)
        mjcf_mesh_name = _safe_xml_name(f"mesh_{mesh_name}", fallback=f"mesh_{mesh_index:04d}")
        ET.SubElement(asset, "mesh", {"name": mjcf_mesh_name, "file": obj_path.name})
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": _safe_xml_name(f"geom_{mesh_name}", fallback=f"geom_{mesh_index:04d}"),
                "type": "mesh",
                "mesh": mjcf_mesh_name,
                "material": material_name,
                "contype": "0",
                "conaffinity": "0",
            },
        )
        mesh_rows.append(
            {
                "prim_path": str(prim.GetPath()),
                "obj_path": str(obj_path),
                "material_name": material_name,
                "texture_name": texture_name,
                "target_distance_m": round(float(candidate.get("target_distance_m") or 0.0), 6),
                "robot_distance_m": round(float(candidate.get("robot_distance_m") or 0.0), 6),
                **obj_info,
            }
        )
    if not mesh_rows:
        return {
            "schema_version": "usd_to_mujoco_visual_mjcf.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "blockers": ["usd_visual_mesh_export_produced_no_meshes"],
            "visual_mjcf_path": None,
        }
    xml_path = visual_dir / "usd_visual_scene.xml"
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return {
        "schema_version": "usd_to_mujoco_visual_mjcf.v1",
        "generated_at": generated_at,
        "status": "completed",
        "blockers": [],
        "visual_mjcf_path": str(xml_path),
        "mesh_count": len(mesh_rows),
        "total_visual_mesh_candidate_count": int(candidates[0].get("total_candidate_count") or len(candidates))
        if candidates
        else 0,
        "max_meshes": int(max_meshes),
        "selected_target_or_robot_proximity_meshes": True,
        "skipped_collision_mesh_count": int(candidates[0].get("skipped_collision_mesh_count") or 0)
        if candidates
        else 0,
        "skipped_invisible_mesh_count": int(candidates[0].get("skipped_invisible_mesh_count") or 0)
        if candidates
        else 0,
        "skipped_degenerate_mesh_count": int(
            candidates[0].get("skipped_degenerate_mesh_count") or 0
        )
        if candidates
        else 0,
        "texture_asset_count": len(texture_rows),
        "textured_mesh_count": sum(1 for row in mesh_rows if row.get("texture_name")),
        "triangle_limited_mesh_count": sum(1 for row in mesh_rows if row.get("triangle_limit_reached")),
        "mesh_rows_sample": mesh_rows[:24],
        "texture_rows_sample": texture_rows[:24],
        "claim_boundary": {
            "visual_mjcf_is_generated_from_usd_for_rendering": True,
            "physics_contact_validated": False,
            "usd_material_texture_binding_attempted": True,
        },
    }


def _render_mujoco_scene_observation(
    *,
    scene_model_path: Path,
    output_dir: Path,
    robot_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
    video_camera: str,
    generated_at: str,
    render_backend: str,
) -> dict[str, Any]:
    try:
        import mujoco  # type: ignore[import-untyped]
        from PIL import Image  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "schema_version": RENDER_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "video_camera": video_camera,
            "render_backend": render_backend,
            "blockers": [f"mujoco_render_import_failed:{type(exc).__name__}"],
            "real_scene_observation_rendered": False,
            "frame_path": None,
        }
    render_dir = output_dir / "rendered_observations"
    ensure_dir(render_dir)
    jpg_path = render_dir / f"initial_policy_observation_{render_backend}.jpg"
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = _mujoco_camera_from_poses(
            mujoco_module=mujoco,
            robot_pose=robot_pose,
            target_pose=target_pose,
            video_camera=video_camera,
        )
        render_width = DEFAULT_POLICY_RENDER_WIDTH
        render_height = DEFAULT_POLICY_RENDER_HEIGHT
        framebuffer_retry_used = False

        def _render_frame(*, width: int, height: int) -> Any:
            renderer = mujoco.Renderer(model, height=height, width=width)
            try:
                renderer.update_scene(data, camera=camera)
                return renderer.render()
            finally:
                renderer.close()

        try:
            frame = _render_frame(width=render_width, height=render_height)
        except ValueError as exc:
            if "framebuffer" not in str(exc).lower():
                raise
            render_width = FALLBACK_MUJOCO_RENDER_WIDTH
            render_height = FALLBACK_MUJOCO_RENDER_HEIGHT
            framebuffer_retry_used = True
            frame = _render_frame(width=render_width, height=render_height)
        Image.fromarray(frame).convert("RGB").save(jpg_path, quality=92)
        content_summary = _rendered_image_content_summary(jpg_path)
        if not content_summary.get("contentful"):
            return {
                "schema_version": RENDER_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "blocked",
                "video_camera": video_camera,
                "render_backend": render_backend,
                "mujoco_model_path": str(scene_model_path),
                "frame_path": str(jpg_path),
                "blockers": list(content_summary.get("blockers", [])),
                "real_scene_observation_rendered": False,
                "rendered_image_content_summary": content_summary,
                "mujoco_model_mesh_count": int(getattr(model, "nmesh", 0)),
                "mujoco_model_texture_count": int(getattr(model, "ntex", 0)),
                "textures_bound_in_loaded_mujoco_model": int(getattr(model, "ntex", 0)) > 0,
                "render_width": render_width,
                "render_height": render_height,
                "framebuffer_retry_used": framebuffer_retry_used,
            }
        return {
            "schema_version": RENDER_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed",
            "video_camera": video_camera,
            "render_backend": render_backend,
            "mujoco_model_path": str(scene_model_path),
            "frame_path": str(jpg_path),
            "rendered_image_content_summary": content_summary,
            "png_path": None,
            "blockers": [],
            "real_scene_observation_rendered": True,
            "mujoco_model_mesh_count": int(getattr(model, "nmesh", 0)),
            "mujoco_model_texture_count": int(getattr(model, "ntex", 0)),
            "textures_bound_in_loaded_mujoco_model": int(getattr(model, "ntex", 0)) > 0,
            "render_width": render_width,
            "render_height": render_height,
            "framebuffer_retry_used": framebuffer_retry_used,
            "claim_boundary": {
                "mujoco_visual_render_used": True,
                "physics_contact_validated": False,
                "physical_robot_sensor_proof": False,
            },
        }
    except Exception as exc:
        return {
            "schema_version": RENDER_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "video_camera": video_camera,
            "render_backend": render_backend,
            "mujoco_model_path": str(scene_model_path),
            "blockers": [f"mujoco_render_failed:{type(exc).__name__}"],
            "real_scene_observation_rendered": False,
            "frame_path": None,
        }


def _render_initial_observation(
    *,
    scene_asset: Path,
    output_dir: Path,
    robot_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
    video_camera: str,
    generated_at: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    availability = _renderer_availability()
    usdrecord = shutil.which("usdrecord")
    blockers: list[str] = []
    render_dir = output_dir / "rendered_observations"
    ensure_dir(render_dir)
    if _is_mjcf_asset(scene_asset):
        composition: dict[str, Any] | None = None
        scene_model_path = scene_asset
        render_backend = "mujoco_mjcf"
        if not _mjcf_contains_unitree_g1(scene_asset):
            composition = _compose_scene_with_unitree_g1_mjcf(
                scene_mjcf_path=scene_asset,
                output_dir=render_dir,
                generated_at=generated_at,
                robot_pose=robot_pose,
                target_pose=target_pose,
            )
            if composition.get("status") != "completed" or not composition.get("combined_mjcf_path"):
                return {
                    "schema_version": RENDER_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "blocked",
                    "video_camera": video_camera,
                    "renderer_discovery": availability,
                    "scene_asset_format": _asset_suffix(scene_asset),
                    "render_backend": "mujoco_mjcf_with_g1",
                    "unitree_g1_scene_composition": composition,
                    "blockers": list(composition.get("blockers", [])),
                    "real_scene_observation_rendered": False,
                    "frame_path": None,
                }
            scene_model_path = Path(str(composition["combined_mjcf_path"]))
            render_backend = "mujoco_mjcf_with_g1"
        render = _render_mujoco_scene_observation(
            scene_model_path=scene_model_path,
            output_dir=output_dir,
            robot_pose=robot_pose,
            target_pose=target_pose,
            video_camera=video_camera,
            generated_at=generated_at,
            render_backend=render_backend,
        )
        return {
            **render,
            "renderer_discovery": availability,
            "scene_asset_format": _asset_suffix(scene_asset),
            "unitree_g1_scene_composition": composition,
        }
    if not availability["python_packages"].get("pxr"):
        blockers.append("missing_python_package_pxr")
    if not _is_usd_asset(scene_asset):
        blockers.append(f"unsupported_scene_asset_format:{_asset_suffix(scene_asset) or 'none'}")
    if not usdrecord:
        blockers.append("missing_renderer_command_usdrecord")
    overlay_path = render_dir / "scene_with_blueprint_head_pov_camera.usda"
    png_path = render_dir / "initial_policy_observation.png"
    metal_png_path = render_dir / "initial_policy_observation_metal.png"
    jpg_path = render_dir / "initial_policy_observation.jpg"
    if blockers:
        return {
            "schema_version": RENDER_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "video_camera": video_camera,
            "renderer_discovery": availability,
            "blockers": blockers,
            "real_scene_observation_rendered": False,
            "frame_path": None,
        }
    ok, overlay_blocker = _write_head_pov_usd(
        scene_asset=scene_asset,
        output_path=overlay_path,
        robot_pose=robot_pose,
        target_pose=target_pose,
    )
    if not ok:
        return {
            "schema_version": RENDER_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "video_camera": video_camera,
            "renderer_discovery": availability,
            "head_pov_usd_path": str(overlay_path),
            "blockers": [overlay_blocker or "head_pov_usd_write_failed"],
            "real_scene_observation_rendered": False,
            "frame_path": None,
        }
    command_attempts: list[dict[str, Any]] = []

    def _run_usdrecord(command: list[str], expected_path: Path) -> tuple[str, Any, float]:
        started = time.monotonic()
        try:
            completed_process = subprocess.run(
                command,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            duration = round(time.monotonic() - started, 6)
            content_summary = _rendered_image_content_summary(expected_path)
            status_text = (
                "completed"
                if (
                    completed_process.returncode == 0
                    and expected_path.is_file()
                    and content_summary.get("contentful")
                )
                else "blocked"
            )
            command_attempts.append(
                {
                    "command": [shlex.quote(part) for part in command],
                    "returncode": completed_process.returncode,
                    "duration_seconds": duration,
                    "stdout_size_bytes": len(completed_process.stdout or ""),
                    "stderr_size_bytes": len(completed_process.stderr or ""),
                    "stdout_omitted_to_avoid_secret_leakage": bool(completed_process.stdout),
                    "stderr_omitted_to_avoid_secret_leakage": bool(completed_process.stderr),
                    "expected_output_path": str(expected_path),
                    "output_created": expected_path.is_file(),
                    "rendered_image_content_summary": content_summary,
                    "blockers": list(content_summary.get("blockers", [])),
                    "status": status_text,
                }
            )
            return status_text, completed_process, duration
        except Exception as exc:
            duration = round(time.monotonic() - started, 6)
            command_attempts.append(
                {
                    "command": [shlex.quote(part) for part in command],
                    "duration_seconds": duration,
                    "expected_output_path": str(expected_path),
                    "output_created": False,
                    "status": "blocked",
                    "blockers": [f"usdrecord_failed:{type(exc).__name__}"],
                }
            )
            return "blocked", None, duration

    cpu_command = [
        usdrecord or "usdrecord",
        "--disableGpu",
        "--camera",
        "/World/BlueprintHeadPovCamera",
        "-w",
        "960",
        str(overlay_path),
        str(png_path),
    ]
    status, completed, _duration = _run_usdrecord(cpu_command, png_path)
    selected_png_path = png_path
    if status != "completed":
        metal_command = [
            usdrecord or "usdrecord",
            "--camera",
            "/World/BlueprintHeadPovCamera",
            "-w",
            "960",
            str(overlay_path),
            str(metal_png_path),
        ]
        status, completed, _duration = _run_usdrecord(metal_command, metal_png_path)
        selected_png_path = metal_png_path
    if status != "completed":
        blockers.append("usdrecord_render_failed")
        visual_mjcf = _build_visual_mjcf_from_usd(
            scene_asset=scene_asset,
            output_dir=render_dir,
            generated_at=generated_at,
            target_pose=target_pose,
            robot_pose=robot_pose,
        )
        if visual_mjcf.get("status") == "completed" and visual_mjcf.get("visual_mjcf_path"):
            composition = _compose_scene_with_unitree_g1_mjcf(
                scene_mjcf_path=Path(str(visual_mjcf["visual_mjcf_path"])),
                output_dir=render_dir,
                generated_at=generated_at,
                robot_pose=robot_pose,
                target_pose=target_pose,
            )
            if composition.get("status") != "completed" or not composition.get("combined_mjcf_path"):
                blockers.extend(str(item) for item in composition.get("blockers", []))
                return {
                    "schema_version": RENDER_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "status": "blocked",
                    "video_camera": video_camera,
                    "renderer_discovery": availability,
                    "scene_asset_format": _asset_suffix(scene_asset),
                    "render_backend": "usd_to_mujoco_visual_mjcf_with_g1",
                    "head_pov_usd_path": str(overlay_path),
                    "usdrecord_command_attempts": command_attempts,
                    "usdrecord_blockers": list(blockers),
                    "usd_to_mujoco_visual_mjcf": visual_mjcf,
                    "unitree_g1_scene_composition": composition,
                    "blockers": blockers,
                    "real_scene_observation_rendered": False,
                    "frame_path": None,
                }
            mujoco_render = _render_mujoco_scene_observation(
                scene_model_path=Path(str(composition["combined_mjcf_path"])),
                output_dir=output_dir,
                robot_pose=robot_pose,
                target_pose=target_pose,
                video_camera=video_camera,
                generated_at=generated_at,
                render_backend="usd_to_mujoco_visual_mjcf_with_g1",
            )
            if mujoco_render.get("status") == "completed":
                return {
                    **mujoco_render,
                    "renderer_discovery": availability,
                    "scene_asset_format": _asset_suffix(scene_asset),
                    "head_pov_usd_path": str(overlay_path),
                    "usdrecord_command_attempts": command_attempts,
                    "usdrecord_blockers": list(blockers),
                    "usd_to_mujoco_visual_mjcf": visual_mjcf,
                    "unitree_g1_scene_composition": composition,
                    "real_scene_observation_rendered": True,
                    "render_backend": "usd_to_mujoco_visual_mjcf_with_g1",
                    "blockers": [],
                }
            blockers.extend(str(item) for item in mujoco_render.get("blockers", []))
        else:
            blockers.extend(str(item) for item in visual_mjcf.get("blockers", []))
    if status == "completed":
        try:
            from PIL import Image  # type: ignore[import-untyped]

            with Image.open(selected_png_path) as image:
                image.convert("RGB").save(jpg_path, quality=92)
        except Exception:
            shutil.copy2(selected_png_path, jpg_path)
    return {
        "schema_version": RENDER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "video_camera": video_camera,
        "renderer_discovery": availability,
        "scene_asset_format": _asset_suffix(scene_asset),
        "render_backend": "usdrecord",
        "head_pov_usd_path": str(overlay_path),
        "png_path": str(selected_png_path) if selected_png_path.is_file() else None,
        "frame_path": str(jpg_path) if jpg_path.is_file() else None,
        "command_attempts": command_attempts,
        "command": command_attempts[-1]["command"] if command_attempts else None,
        "returncode": completed.returncode if completed is not None else None,
        "duration_seconds": command_attempts[-1]["duration_seconds"] if command_attempts else None,
        "stdout_size_bytes": command_attempts[-1].get("stdout_size_bytes", 0)
        if command_attempts
        else 0,
        "stderr_size_bytes": command_attempts[-1].get("stderr_size_bytes", 0)
        if command_attempts
        else 0,
        "stdout_omitted_to_avoid_secret_leakage": command_attempts[-1].get(
            "stdout_omitted_to_avoid_secret_leakage", False
        )
        if command_attempts
        else False,
        "stderr_omitted_to_avoid_secret_leakage": command_attempts[-1].get(
            "stderr_omitted_to_avoid_secret_leakage", False
        )
        if command_attempts
        else False,
        "blockers": blockers,
        "real_scene_observation_rendered": status == "completed",
    }


def build_scene_wam_policy_episode_packet(
    *,
    capture_root: str | Path,
    scene_asset: str | Path,
    task_id: str,
    target_object_id: str,
    target_anchor_pose: str | Sequence[float] | Mapping[str, Any] | None = None,
    robot_start_pose: str | Sequence[float] | Mapping[str, Any] | None = None,
    video_camera: str = "head_pov",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    resolved_scene_asset = Path(scene_asset).expanduser().resolve()
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else context.pipeline_root / "scene_wam_policy_episode_packet"
    )
    ensure_dir(resolved_output_dir)
    generated_at = utc_now_iso()
    if not (context.pipeline_root / "simulation_automation" / "scene_asset_preflight.json").is_file():
        build_scene_asset_preflight(
            capture_root=context.capture_root,
            scene_assets=[resolved_scene_asset],
        )
    if not (context.pipeline_root / "simulation_automation" / "episode_specs.json").is_file():
        build_episode_specs(capture_root=context.capture_root)
    scene_summary = _scene_bounds_and_target(
        resolved_scene_asset,
        target_object_id=target_object_id,
    )
    task_anchor = _load_task_anchor(
        context.capture_root,
        task_id=task_id,
        target_object_id=target_object_id,
    )
    episode = _load_episode(context.capture_root, task_id=task_id)
    target_fallback = scene_summary.get("target_anchor_xyz") or [0.0, 0.0, 1.0]
    target_pose = _pose(
        target_anchor_pose or task_anchor.get("goal_zone") or episode.get("target_region"),
        fallback_xyz=target_fallback,
        source="cli_or_task_anchor_or_usd_target_bounds",
    )
    robot_pose, robot_placement = _resolve_robot_start_pose(
        capture_root=context.capture_root,
        scene_asset=resolved_scene_asset,
        scene_summary=scene_summary,
        task_id=task_id,
        target_object_id=target_object_id,
        target_pose=target_pose,
        robot_start_pose=robot_start_pose,
        episode=episode,
    )
    render = _render_initial_observation(
        scene_asset=resolved_scene_asset,
        output_dir=resolved_output_dir,
        robot_pose=robot_pose,
        target_pose=target_pose,
        video_camera=video_camera,
        generated_at=generated_at,
    )
    frame_path = render.get("frame_path") if render.get("status") == "completed" else None
    g1_composition = (
        dict(render.get("unitree_g1_scene_composition"))
        if isinstance(render.get("unitree_g1_scene_composition"), Mapping)
        else {}
    )
    g1_composition_path = (
        str(
            Path(str(g1_composition["combined_mjcf_path"])).parent
            / "scene_unitree_g1_mujoco_composition.json"
        )
        if g1_composition.get("combined_mjcf_path")
        else None
    )
    unitree_g1_spawned_in_mujoco = bool(g1_composition.get("unitree_g1_asset_spawned"))
    initial_observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "robot_profile_id": "unitree_g1_sonic",
        "selected_candidate_id": "unitree_groot_n17_sonic_policy",
        "camera_frame_path": frame_path,
        "visual_observation": {
            "available": bool(frame_path),
            "camera_frame_path": frame_path,
            "camera_id": video_camera,
            "first_person_policy_observation_candidate": bool(frame_path),
            "scene_observation_rendered_from_usd": bool(frame_path),
            "unitree_g1_asset_spawned_in_mujoco_scene": unitree_g1_spawned_in_mujoco,
            "unitree_g1_scene_composition_path": g1_composition_path,
            "blank_or_placeholder_image_used": False,
            "physical_robot_sensor_proof": False,
            "blockers": [] if frame_path else list(render.get("blockers", [])),
        },
        "robot_start_pose": robot_pose,
        "robot_start_pose_resolution": robot_placement,
        "target_anchor_pose": target_pose,
        "state": {
            "root_position": robot_pose["xyz"],
            "root_yaw_rad": float(_vec3(robot_pose.get("rpy"), fallback=[0.0, 0.0, 0.0])[2]),
            "target_waypoint": target_pose["xyz"][:2],
            "target_object_id": target_object_id,
        },
        "unitree_g1_sonic_state": UNITREE_G1_SONIC_ZERO_STATE,
        "unitree_g1_sonic_state_source": "scene_packet_contract_probe_zero_state",
        "claim_boundary": {
            "simulator_generated_world_observation_only": True,
            "blank_or_placeholder_image_used": False,
            "physical_robot_sensor_proof": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
    }
    task_manifest = {
        "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_asset": str(resolved_scene_asset),
        "scene_asset_exists": resolved_scene_asset.is_file(),
        "scene_summary": scene_summary,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "target_anchor_pose": target_pose,
        "robot_profile_id": "unitree_g1_sonic",
        "robot_spawn_pose": robot_pose,
        "robot_start_pose_resolution": robot_placement,
        "video_camera": video_camera,
        "scene_physics_required_for_wam_loop": False,
        "physics_contact_validated": False,
        "static_usd_aabb_clearance_proxy_passed": (
            _mapping(robot_placement.get("selected_clearance_check")).get("status") == "passed"
        ),
        "input_robot_start_pose_rejected": bool(
            robot_placement.get("input_robot_start_pose_rejected")
        ),
        "unitree_g1_asset_spawned_in_mujoco_scene": unitree_g1_spawned_in_mujoco,
        "unitree_g1_scene_composition_path": g1_composition_path,
        "scene_collision_geometry_validated": False,
        "scene_visual_collision_disabled": bool(
            _mapping(g1_composition.get("claim_boundary")).get("scene_visual_collision_disabled")
        ),
        "task_anchor_source": task_anchor.get("anchor_source"),
        "episode_source": episode.get("episode_id"),
        "blockers": [
            *([] if resolved_scene_asset.is_file() else ["scene_asset_missing"]),
            *list(scene_summary.get("blockers", [])),
            *list(robot_placement.get("blockers", [])),
            *([] if frame_path else ["initial_policy_observation_render_not_available"]),
        ],
    }
    claim_boundary = {
        "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_physics_required_for_wam_loop": False,
        "physics_contact_validated": False,
        "scene_packet_is_policy_loop_input_not_success_proof": True,
        "wam_evaluator_is_not_robot_policy": True,
        "generated_wam_outputs_are_not_raw_capture_evidence": True,
        "unitree_g1_asset_spawned_in_mujoco_scene": unitree_g1_spawned_in_mujoco,
        "scene_collision_geometry_validated": False,
        "static_usd_aabb_clearance_proxy_passed": (
            _mapping(robot_placement.get("selected_clearance_check")).get("status") == "passed"
        ),
        "input_robot_start_pose_rejected": bool(
            robot_placement.get("input_robot_start_pose_rejected")
        ),
        "real_collision_geometry_validated": False,
        "scene_visual_collision_disabled": bool(
            _mapping(g1_composition.get("claim_boundary")).get("scene_visual_collision_disabled")
        ),
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
    }
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_for_policy_wam_loop" if frame_path else "blocked",
        "capture_root": str(context.capture_root),
        "scene_asset": str(resolved_scene_asset),
        "task_id": task_id,
        "target_object_id": target_object_id,
        "robot_profile_id": "unitree_g1_sonic",
        "selected_candidate_id": "unitree_groot_n17_sonic_policy",
        "initial_policy_observation_path": str(
            resolved_output_dir / "initial_policy_observation.json"
        ),
        "scene_episode_task_manifest_path": str(
            resolved_output_dir / "scene_episode_task_manifest.json"
        ),
        "robot_start_pose_resolution_path": str(
            resolved_output_dir / "robot_start_pose_resolution.json"
        ),
        "scene_policy_wam_claim_boundary_path": str(
            resolved_output_dir / "scene_policy_wam_claim_boundary.json"
        ),
        "render_manifest_path": str(resolved_output_dir / "initial_policy_observation_render.json"),
        "unitree_g1_scene_composition_path": g1_composition_path,
        "unitree_g1_asset_spawned_in_mujoco_scene": unitree_g1_spawned_in_mujoco,
        "initial_policy_observation_frame_path": frame_path,
        "scene_physics_required_for_wam_loop": False,
        "physics_contact_validated": False,
        "scene_collision_geometry_validated": False,
        "static_usd_aabb_clearance_proxy_passed": (
            _mapping(robot_placement.get("selected_clearance_check")).get("status") == "passed"
        ),
        "input_robot_start_pose_rejected": bool(
            robot_placement.get("input_robot_start_pose_rejected")
        ),
        "robot_start_pose": robot_pose,
        "blockers": task_manifest["blockers"],
        "claim_boundary": claim_boundary,
    }
    write_json(resolved_output_dir / "initial_policy_observation_render.json", render)
    write_json(resolved_output_dir / "initial_policy_observation.json", initial_observation)
    write_json(resolved_output_dir / "scene_episode_task_manifest.json", task_manifest)
    write_json(resolved_output_dir / "robot_start_pose_resolution.json", robot_placement)
    write_json(resolved_output_dir / "scene_policy_wam_claim_boundary.json", claim_boundary)
    write_json(resolved_output_dir / "scene_wam_policy_episode_packet.json", packet)
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--scene-asset", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--target-object-id", required=True)
    parser.add_argument("--target-anchor-pose")
    parser.add_argument("--robot-start-pose")
    parser.add_argument("--video-camera", default="head_pov")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    packet = build_scene_wam_policy_episode_packet(
        capture_root=args.capture_root,
        scene_asset=args.scene_asset,
        task_id=args.task_id,
        target_object_id=args.target_object_id,
        target_anchor_pose=args.target_anchor_pose,
        robot_start_pose=args.robot_start_pose,
        video_camera=args.video_camera,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": packet.get("status"),
                "scene_wam_policy_episode_packet": str(
                    Path(packet["initial_policy_observation_path"]).parent
                    / "scene_wam_policy_episode_packet.json"
                ),
                "initial_policy_observation_frame_path": packet.get(
                    "initial_policy_observation_frame_path"
                ),
                "blockers": packet.get("blockers", []),
            },
            sort_keys=True,
        )
    )
    return 0 if packet.get("status") == "ready_for_policy_wam_loop" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
