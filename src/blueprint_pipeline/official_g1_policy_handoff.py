"""Build robot-team handoff artifacts from the official Unitree G1 MuJoCo policy."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .common import ensure_dir, optional_read_json, read_json_any, utc_now_iso, write_json
from .mujoco_g1_simulator_command import (
    _blank_scene_checks,
    _convert_glb_to_obj,
    _find_scene_glb,
    _glb_visual_summary,
    _render_capture_steps,
)


OFFICIAL_G1_HANDOFF_SCHEMA_VERSION = "official_unitree_g1_robot_team_handoff.v1"
OFFICIAL_G1_ENRICHED_TRACE_SCHEMA_VERSION = "official_unitree_g1_policy_trace_enriched.v1"
OFFICIAL_G1_RENDER_MANIFEST_SCHEMA_VERSION = "official_unitree_g1_policy_rendered_motion.v2"
ROBOT_POV_MANIFEST_SCHEMA_VERSION = "official_unitree_g1_robot_pov.v1"
WORLDLABS_OVERLAY_SCHEMA_VERSION = "worldlabs_visual_overlay_manifest.v1"
DEFAULT_POLICY_RELATIVE = (
    "pipeline/sim_only_beta_rehearsal/official_unitree_g1_policy_execution"
)
DEFAULT_MATRIX_RELATIVE = "pipeline/robot_eval_dataset/sim_only_scenario_eval_matrix.json"
SIGNED_URL_SIGNATURE_PARAM = "x-goog-" + "signature="
SIGNED_URL_QUERY_PATTERN = re.compile(
    rf"([?&]){SIGNED_URL_SIGNATURE_PARAM}[^\s\"'&]+",
    flags=re.IGNORECASE,
)
SIGNED_URL_SIGNATURE_REPLACEMENT = "x-goog-redacted-signature-param=<redacted:signed-url-signature>"
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
CAMERA_SPECS = {
    "robot_pov_head": {
        "body_name": "pelvis",
        "mount_label": "head_proxy_on_pelvis_body",
        "pos": "0.16 0 0.58",
        "xyaxes": "0 -1 0 0 0 1",
        "fovy": "82",
    },
    "robot_pov_torso": {
        "body_name": "pelvis",
        "mount_label": "torso_proxy_on_pelvis_body",
        "pos": "0.10 0 0.34",
        "xyaxes": "0 -1 0 0 0 1",
        "fovy": "88",
    },
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) for value in array.tolist()]


def _number(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pose_triplet(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        x = _number(value.get("x") or value.get("X") or value.get("pos_x"))
        y = _number(value.get("y") or value.get("Y") or value.get("pos_y"))
        z = _number(value.get("z") or value.get("Z") or value.get("pos_z"))
        if x is not None and y is not None:
            return (x, y, z if z is not None else 0.793)
        position = (
            value.get("position")
            or value.get("position_xyz")
            or value.get("translation")
            or value.get("xyz")
        )
        return _pose_triplet(position)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if len(parts) >= 2:
            x = _number(parts[0])
            y = _number(parts[1])
            z = _number(parts[2]) if len(parts) >= 3 else 0.793
            if x is not None and y is not None:
                return (x, y, z if z is not None else 0.793)
    return None


def _nested_pose(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[float, float, float] | None:
    for key in keys:
        if key in mapping:
            pose = _pose_triplet(mapping.get(key))
            if pose is not None:
                return pose
    for nested_key in (
        "navigation",
        "route",
        "robot_route",
        "concrete_mutation",
        "engine_mutations",
        "mujoco",
        "mujoco_mutation",
    ):
        nested = mapping.get(nested_key)
        if isinstance(nested, Mapping):
            pose = _nested_pose(nested, keys)
            if pose is not None:
                return pose
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _redact_signed_url_text(text: str) -> str:
    if SIGNED_URL_SIGNATURE_PARAM not in text.lower():
        return text
    return SIGNED_URL_QUERY_PATTERN.sub(rf"\1{SIGNED_URL_SIGNATURE_REPLACEMENT}", text)


def _redact_runtime_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_signed_url_text(value)
    if isinstance(value, list):
        return [_redact_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_runtime_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redact_runtime_value(item) for key, item in value.items()}
    return value


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    redacted = _redact_runtime_value(payload)
    write_json(path, dict(redacted) if isinstance(redacted, Mapping) else dict(payload))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            redacted = _redact_runtime_value(row)
            handle.write(json.dumps(redacted, sort_keys=True) + "\n")
            count += 1
    return count


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _first_matrix_run(path: Path) -> dict[str, Any]:
    payload = optional_read_json(path) or {}
    runs = payload.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
        for run in runs:
            if isinstance(run, Mapping):
                return dict(run)
    return {}


def _scenario_context(path: Path) -> dict[str, Any]:
    payload = optional_read_json(path) or {}
    run = _first_matrix_run(path)
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    return {
        "matrix_path": str(path),
        "matrix_status": payload.get("status"),
        "scenario_eval_run_count": payload.get("scenario_eval_run_count") or len(runs),
        "variation_instance_count": payload.get("variation_instance_count"),
        "selected_run": {
            key: run.get(key)
            for key in (
                "scenario_eval_run_id",
                "task_id",
                "scenario_id",
                "scenario_variation_instance_id",
                "variation_name",
                "robot_pov_required",
                "policy_attempt_required",
                "simulator_rollout_required",
                "review_required",
            )
        },
        "scenario_eval_run_ids": [
            _string(item.get("scenario_eval_run_id"))
            for item in runs
            if isinstance(item, Mapping) and _string(item.get("scenario_eval_run_id"))
        ],
    }


def _resolve_policy_root(
    *,
    explicit_root: str | Path | None,
    manifest: Mapping[str, Any],
    handoff_dir: Path,
) -> Path:
    candidates: list[Any] = [
        explicit_root,
        handoff_dir / "policy_source_snapshot" / "unitree_rl_gym",
        _mapping(manifest.get("source_repository")).get("local_inspection_root"),
        "/private/tmp/blueprint-unitree-policy-inspection/unitree_rl_gym",
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if (path / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml").is_file():
            return path
    searched = [str(value) for value in candidates if value]
    raise FileNotFoundError("missing Unitree RL Gym root; searched " + ", ".join(searched))


def _copy_tree_files(source_root: Path, relative_paths: Sequence[str], target_root: Path) -> None:
    for relative in relative_paths:
        source = source_root / relative
        target = target_root / relative
        if source.is_dir():
            for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
                destination = target / file_path.relative_to(source)
                ensure_dir(destination.parent)
                shutil.copy2(file_path, destination)
        elif source.is_file():
            ensure_dir(target.parent)
            shutil.copy2(source, target)


def _materialize_policy_snapshot(policy_root: Path, handoff_dir: Path) -> dict[str, Any]:
    snapshot_root = handoff_dir / "policy_source_snapshot" / "unitree_rl_gym"
    if policy_root.resolve() == snapshot_root.resolve():
        files = sorted(path for path in snapshot_root.rglob("*") if path.is_file())
        return {
            "status": "complete",
            "source_root": str(policy_root),
            "snapshot_root": str(snapshot_root),
            "file_count": len(files),
            "snapshot_sha256": hashlib.sha256(
                "\n".join(
                    f"{file.relative_to(snapshot_root)}:{_sha256(file)}" for file in files
                ).encode("utf-8")
            ).hexdigest(),
            "portable_provider_rerun_ready": True,
        }
    relative_paths = (
        "LICENSE",
        "deploy/deploy_mujoco/configs/g1.yaml",
        "deploy/pre_train/g1/motion.pt",
        "resources/robots/g1_description",
    )
    _copy_tree_files(policy_root, relative_paths, snapshot_root)
    files = sorted(path for path in snapshot_root.rglob("*") if path.is_file())
    return {
        "status": "complete",
        "source_root": str(policy_root),
        "snapshot_root": str(snapshot_root),
        "file_count": len(files),
        "snapshot_sha256": hashlib.sha256(
            "\n".join(f"{file.relative_to(snapshot_root)}:{_sha256(file)}" for file in files).encode(
                "utf-8"
            )
        ).hexdigest(),
        "portable_provider_rerun_ready": True,
    }


def _policy_paths(policy_root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    official = _mapping(manifest.get("official_artifacts"))
    config_path = Path(
        _string(official.get("config_path"))
        or policy_root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
    )
    policy_path = Path(
        _string(official.get("policy_path")) or policy_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    )
    xml_path = Path(
        _string(official.get("xml_path"))
        or policy_root / "resources" / "robots" / "g1_description" / "scene.xml"
    )
    if not config_path.is_file() or not str(config_path).startswith(str(policy_root)):
        config_path = policy_root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
    if not policy_path.is_file() or not str(policy_path).startswith(str(policy_root)):
        policy_path = policy_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    if not xml_path.is_file() or not str(xml_path).startswith(str(policy_root)):
        xml_path = policy_root / "resources" / "robots" / "g1_description" / "scene.xml"
    for label, path in {
        "config": config_path,
        "policy": policy_path,
        "xml": xml_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing official Unitree {label} artifact: {path}")
    return {"config": config_path, "policy": policy_path, "xml": xml_path}


def _load_policy_config(config_path: Path, policy_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = dict(payload) if isinstance(payload, Mapping) else {}
    for key in ("policy_path", "xml_path"):
        value = _string(config.get(key))
        if value:
            config[key] = value.replace("{LEGGED_GYM_ROOT_DIR}", str(policy_root))
    return config


def _xml_escape(value: Path | str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _xml_float(value: Any) -> str:
    return f"{float(value):.6g}"


def _xml_vec(values: Sequence[Any]) -> str:
    return " ".join(_xml_float(value) for value in values)


def _proxy_xy_distance(point_xy: Sequence[float], proxy: Mapping[str, Any]) -> float | None:
    pos = proxy.get("pos")
    size = proxy.get("size")
    if not (
        isinstance(pos, Sequence)
        and not isinstance(pos, (str, bytes))
        and isinstance(size, Sequence)
        and not isinstance(size, (str, bytes))
        and len(pos) >= 2
        and len(size) >= 2
    ):
        return None
    x, y = float(point_xy[0]), float(point_xy[1])
    px, py = float(pos[0]), float(pos[1])
    sx, sy = float(size[0]), float(size[1])
    dx = max(abs(x - px) - sx, 0.0)
    dy = max(abs(y - py) - sy, 0.0)
    return float(math.hypot(dx, dy))


def _base_path_clearance_audit(
    *,
    base_positions: Sequence[Sequence[float]],
    collision_proxies: Sequence[Mapping[str, Any]],
    required_clearance_m: float,
) -> dict[str, Any]:
    if not base_positions:
        return {
            "status": "failed",
            "reason": "missing_base_positions",
            "passed": False,
        }
    if not collision_proxies:
        return {
            "status": "failed",
            "reason": "missing_collision_proxies",
            "passed": False,
        }
    minimum: dict[str, Any] | None = None
    for step, base in enumerate(base_positions):
        point_xy = [float(base[0]), float(base[1])]
        for proxy_index, proxy in enumerate(collision_proxies):
            distance = _proxy_xy_distance(point_xy, proxy)
            if distance is None:
                continue
            if minimum is None or distance < float(minimum["clearance_m"]):
                minimum = {
                    "clearance_m": distance,
                    "step_index": step,
                    "base_xy": point_xy,
                    "proxy_index": proxy_index,
                    "proxy_name": proxy.get("name"),
                    "proxy_pos": proxy.get("pos"),
                    "proxy_size": proxy.get("size"),
                }
    endpoint_clearance = None
    endpoint_proxy_index = None
    endpoint = base_positions[-1]
    for proxy_index, proxy in enumerate(collision_proxies):
        distance = _proxy_xy_distance([float(endpoint[0]), float(endpoint[1])], proxy)
        if distance is None:
            continue
        if endpoint_clearance is None or distance < endpoint_clearance:
            endpoint_clearance = distance
            endpoint_proxy_index = proxy_index
    min_clearance = float(minimum["clearance_m"]) if minimum is not None else None
    passed = (
        min_clearance is not None
        and min_clearance >= float(required_clearance_m)
        and endpoint_clearance is not None
        and endpoint_clearance >= float(required_clearance_m)
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "required_clearance_m": float(required_clearance_m),
        "minimum_clearance_m": min_clearance,
        "minimum_clearance_sample": minimum,
        "endpoint_clearance_m": endpoint_clearance,
        "endpoint_proxy_index": endpoint_proxy_index,
        "base_sample_count": len(base_positions),
        "collision_proxy_count": len(collision_proxies),
    }


def _rounded_pose(pose: Sequence[float]) -> tuple[float, float, float]:
    return (round(float(pose[0]), 6), round(float(pose[1]), 6), round(float(pose[2]), 6))


def _dedupe_route_points(
    points: Sequence[Sequence[float]], *, min_distance_m: float = 0.05
) -> list[tuple[float, float, float]]:
    route: list[tuple[float, float, float]] = []
    for point in points:
        rounded = _rounded_pose(point)
        if not route:
            route.append(rounded)
            continue
        last = route[-1]
        distance = math.sqrt(
            (rounded[0] - last[0]) ** 2
            + (rounded[1] - last[1]) ** 2
            + (rounded[2] - last[2]) ** 2
        )
        if distance >= min_distance_m:
            route.append(rounded)
    return route


def _route_distance(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt(
            (float(b[0]) - float(a[0])) ** 2
            + (float(b[1]) - float(a[1])) ** 2
            + (float(b[2]) - float(a[2])) ** 2
        )
    return total


def _mesh_xy_bounds(
    *,
    mesh_info: Mapping[str, Any],
    collision_proxies: Sequence[Mapping[str, Any]],
    start: Sequence[float],
    goal: Sequence[float],
    margin_m: float,
) -> tuple[float, float, float, float]:
    xs = [float(start[0]), float(goal[0])]
    ys = [float(start[1]), float(goal[1])]
    bounds = mesh_info.get("bounds")
    if isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)) and len(bounds) >= 2:
        lower = bounds[0]
        upper = bounds[1]
        if (
            isinstance(lower, Sequence)
            and not isinstance(lower, (str, bytes))
            and isinstance(upper, Sequence)
            and not isinstance(upper, (str, bytes))
            and len(lower) >= 2
            and len(upper) >= 2
        ):
            min_x = _number(lower[0])
            min_y = _number(lower[1])
            max_x = _number(upper[0])
            max_y = _number(upper[1])
            if None not in (min_x, min_y, max_x, max_y):
                xs.extend([float(min_x), float(max_x)])
                ys.extend([float(min_y), float(max_y)])
    for proxy in collision_proxies:
        pos = proxy.get("pos")
        size = proxy.get("size")
        if not (
            isinstance(pos, Sequence)
            and not isinstance(pos, (str, bytes))
            and isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(pos) >= 2
            and len(size) >= 2
        ):
            continue
        xs.extend([float(pos[0]) - float(size[0]), float(pos[0]) + float(size[0])])
        ys.extend([float(pos[1]) - float(size[1]), float(pos[1]) + float(size[1])])
    return (
        min(xs) - margin_m,
        min(ys) - margin_m,
        max(xs) + margin_m,
        max(ys) + margin_m,
    )


def _clearance_sample(
    *,
    point_xy: Sequence[float],
    collision_proxies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    minimum: dict[str, Any] | None = None
    for proxy_index, proxy in enumerate(collision_proxies):
        distance = _proxy_xy_distance(point_xy, proxy)
        if distance is None:
            continue
        if minimum is None or distance < float(minimum["clearance_m"]):
            minimum = {
                "clearance_m": distance,
                "xy": [float(point_xy[0]), float(point_xy[1])],
                "proxy_index": proxy_index,
                "proxy_name": proxy.get("name"),
                "proxy_pos": proxy.get("pos"),
                "proxy_size": proxy.get("size"),
            }
    return minimum or {
        "clearance_m": None,
        "xy": [float(point_xy[0]), float(point_xy[1])],
        "proxy_index": None,
        "proxy_name": None,
    }


def _route_clearance_audit(
    *,
    route_waypoints: Sequence[Sequence[float]],
    collision_proxies: Sequence[Mapping[str, Any]],
    required_clearance_m: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    if len(route_waypoints) < 2:
        return {
            "status": "failed",
            "passed": False,
            "reason": "route_requires_at_least_two_waypoints",
        }
    minimum: dict[str, Any] | None = None
    sample_count = 0
    for segment_index, (start, end) in enumerate(zip(route_waypoints, route_waypoints[1:])):
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        segment_length = math.hypot(ex - sx, ey - sy)
        steps = max(1, int(math.ceil(segment_length / max(0.05, sample_spacing_m))))
        for sample_index in range(steps + 1):
            alpha = sample_index / float(steps)
            point = [sx + (ex - sx) * alpha, sy + (ey - sy) * alpha]
            sample_count += 1
            sample = _clearance_sample(point_xy=point, collision_proxies=collision_proxies)
            clearance = sample.get("clearance_m")
            if clearance is None:
                continue
            candidate = {**sample, "segment_index": segment_index, "sample_index": sample_index}
            if minimum is None or float(clearance) < float(minimum["clearance_m"]):
                minimum = candidate
    min_clearance = float(minimum["clearance_m"]) if minimum is not None else None
    passed = min_clearance is not None and min_clearance >= float(required_clearance_m)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "required_clearance_m": float(required_clearance_m),
        "minimum_clearance_m": min_clearance,
        "minimum_clearance_sample": minimum,
        "route_sample_count": sample_count,
        "collision_proxy_count": len(collision_proxies),
    }


def _segment_is_clear(
    *,
    start: Sequence[float],
    end: Sequence[float],
    collision_proxies: Sequence[Mapping[str, Any]],
    required_clearance_m: float,
    sample_spacing_m: float,
) -> bool:
    audit = _route_clearance_audit(
        route_waypoints=[start, end],
        collision_proxies=collision_proxies,
        required_clearance_m=required_clearance_m,
        sample_spacing_m=sample_spacing_m,
    )
    return audit.get("passed") is True


def _smooth_route(
    *,
    route_waypoints: Sequence[Sequence[float]],
    collision_proxies: Sequence[Mapping[str, Any]],
    required_clearance_m: float,
    sample_spacing_m: float,
) -> list[tuple[float, float, float]]:
    if len(route_waypoints) <= 2:
        return _dedupe_route_points(route_waypoints)
    smoothed: list[tuple[float, float, float]] = [_rounded_pose(route_waypoints[0])]
    current_index = 0
    while current_index < len(route_waypoints) - 1:
        next_index = current_index + 1
        for candidate_index in range(len(route_waypoints) - 1, current_index, -1):
            if _segment_is_clear(
                start=route_waypoints[current_index],
                end=route_waypoints[candidate_index],
                collision_proxies=collision_proxies,
                required_clearance_m=required_clearance_m,
                sample_spacing_m=sample_spacing_m,
            ):
                next_index = candidate_index
                break
        smoothed.append(_rounded_pose(route_waypoints[next_index]))
        current_index = next_index
    return _dedupe_route_points(smoothed)


def _plan_occupancy_grid_route(
    *,
    start: Sequence[float],
    goal: Sequence[float],
    collision_proxies: Sequence[Mapping[str, Any]],
    mesh_info: Mapping[str, Any],
    required_clearance_m: float,
    grid_resolution_m: float = 0.35,
) -> dict[str, Any]:
    start_pose = _rounded_pose(start)
    goal_pose = _rounded_pose(goal)
    blockers: list[str] = []
    if not collision_proxies:
        blockers.append("missing_collision_proxies_for_occupancy_map")

    start_clearance = _clearance_sample(
        point_xy=start_pose[:2],
        collision_proxies=collision_proxies,
    )
    goal_clearance = _clearance_sample(
        point_xy=goal_pose[:2],
        collision_proxies=collision_proxies,
    )
    if start_clearance.get("clearance_m") is None:
        blockers.append("start_clearance_unavailable")
    elif float(start_clearance["clearance_m"]) < float(required_clearance_m):
        blockers.append("start_occupied_or_below_clearance")
    if goal_clearance.get("clearance_m") is None:
        blockers.append("goal_clearance_unavailable")
    elif float(goal_clearance["clearance_m"]) < float(required_clearance_m):
        blockers.append("goal_occupied_or_below_clearance")
    if blockers:
        return {
            "schema_version": "official_unitree_g1_navigation_plan.v1",
            "status": "blocked",
            "planned": False,
            "blockers": blockers,
            "start_pose": list(start_pose),
            "goal_pose": list(goal_pose),
            "start_clearance": start_clearance,
            "goal_clearance": goal_clearance,
            "route_waypoints": [],
            "route_distance_m": None,
            "route_clearance_audit": None,
            "occupancy_map": {
                "source": "external_scene_collision_proxy_geoms",
                "collision_proxy_count": len(collision_proxies),
            },
        }

    resolution = max(0.10, float(grid_resolution_m))
    margin = max(float(required_clearance_m) + resolution * 2.0, 1.0)
    min_x, min_y, max_x, max_y = _mesh_xy_bounds(
        mesh_info=mesh_info,
        collision_proxies=collision_proxies,
        start=start_pose,
        goal=goal_pose,
        margin_m=margin,
    )
    width = max_x - min_x
    height = max_y - min_y
    max_cells_axis = 220
    if width / resolution > max_cells_axis or height / resolution > max_cells_axis:
        resolution = max(width, height) / float(max_cells_axis)
    x_count = max(2, int(math.ceil(width / resolution)) + 1)
    y_count = max(2, int(math.ceil(height / resolution)) + 1)

    def world_to_cell(point: Sequence[float]) -> tuple[int, int]:
        ix = int(round((float(point[0]) - min_x) / resolution))
        iy = int(round((float(point[1]) - min_y) / resolution))
        return max(0, min(x_count - 1, ix)), max(0, min(y_count - 1, iy))

    def cell_center(cell: tuple[int, int]) -> tuple[float, float, float]:
        return (
            min_x + cell[0] * resolution,
            min_y + cell[1] * resolution,
            start_pose[2],
        )

    free_cache: dict[tuple[int, int], bool] = {}

    def cell_is_free(cell: tuple[int, int]) -> bool:
        if cell in free_cache:
            return free_cache[cell]
        center = cell_center(cell)
        sample = _clearance_sample(point_xy=center[:2], collision_proxies=collision_proxies)
        clearance = sample.get("clearance_m")
        free = clearance is not None and float(clearance) >= float(required_clearance_m)
        free_cache[cell] = free
        return free

    start_cell = world_to_cell(start_pose)
    goal_cell = world_to_cell(goal_pose)
    free_cache[start_cell] = True
    free_cache[goal_cell] = True
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(goal_cell[0] - cell[0], goal_cell[1] - cell[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = [(heuristic(start_cell), 0.0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    closed: set[tuple[int, int]] = set()
    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            break
        closed.add(current)
        for dx, dy in neighbors:
            candidate = (current[0] + dx, current[1] + dy)
            if not (0 <= candidate[0] < x_count and 0 <= candidate[1] < y_count):
                continue
            if not cell_is_free(candidate):
                continue
            step_cost = math.sqrt(2.0) if dx and dy else 1.0
            tentative = current_cost + step_cost
            if tentative >= g_score.get(candidate, float("inf")):
                continue
            came_from[candidate] = current
            g_score[candidate] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(candidate), tentative, candidate))

    if goal_cell not in came_from and goal_cell != start_cell:
        return {
            "schema_version": "official_unitree_g1_navigation_plan.v1",
            "status": "blocked",
            "planned": False,
            "blockers": ["no_collision_free_occupancy_grid_route"],
            "start_pose": list(start_pose),
            "goal_pose": list(goal_pose),
            "start_clearance": start_clearance,
            "goal_clearance": goal_clearance,
            "route_waypoints": [],
            "route_distance_m": None,
            "route_clearance_audit": None,
            "occupancy_map": {
                "source": "external_scene_collision_proxy_geoms",
                "resolution_m": resolution,
                "bounds_xy": [min_x, min_y, max_x, max_y],
                "grid_shape_xy": [x_count, y_count],
                "collision_proxy_count": len(collision_proxies),
                "expanded_clearance_m": float(required_clearance_m),
                "visited_cell_count": len(closed),
            },
        }

    cell_path = [goal_cell]
    while cell_path[-1] != start_cell:
        cell_path.append(came_from[cell_path[-1]])
    cell_path.reverse()
    raw_route: list[tuple[float, float, float]] = [start_pose]
    raw_route.extend(cell_center(cell) for cell in cell_path[1:-1])
    raw_route.append(goal_pose)
    route = _smooth_route(
        route_waypoints=raw_route,
        collision_proxies=collision_proxies,
        required_clearance_m=required_clearance_m,
        sample_spacing_m=max(0.05, resolution * 0.5),
    )
    route_audit = _route_clearance_audit(
        route_waypoints=route,
        collision_proxies=collision_proxies,
        required_clearance_m=required_clearance_m,
        sample_spacing_m=max(0.05, resolution * 0.5),
    )
    if route_audit.get("passed") is not True:
        blockers.append("planned_route_clearance_audit_failed")
    return {
        "schema_version": "official_unitree_g1_navigation_plan.v1",
        "status": "planned" if not blockers else "blocked",
        "planned": not blockers,
        "blockers": blockers,
        "start_pose": list(start_pose),
        "goal_pose": list(goal_pose),
        "start_clearance": start_clearance,
        "goal_clearance": goal_clearance,
        "route_waypoints": [list(point) for point in route],
        "raw_grid_route_waypoint_count": len(raw_route),
        "route_waypoint_count": len(route),
        "route_distance_m": round(_route_distance(route), 6),
        "route_clearance_audit": route_audit,
        "occupancy_map": {
            "source": "external_scene_collision_proxy_geoms",
            "resolution_m": resolution,
            "bounds_xy": [min_x, min_y, max_x, max_y],
            "grid_shape_xy": [x_count, y_count],
            "collision_proxy_count": len(collision_proxies),
            "expanded_clearance_m": float(required_clearance_m),
            "visited_cell_count": len(closed),
        },
        "planner_boundary": (
            "A deterministic 2D occupancy-grid route is planned from generated MuJoCo "
            "collision proxies, then converted to velocity commands for the official G1 "
            "policy. This is simulated support evidence, not physical robot readiness."
        ),
    }


def _default_navigation_goal(
    *,
    start: Sequence[float],
    mesh_info: Mapping[str, Any],
    collision_proxies: Sequence[Mapping[str, Any]],
    required_clearance_m: float,
) -> tuple[float, float, float] | None:
    min_x, min_y, max_x, max_y = _mesh_xy_bounds(
        mesh_info=mesh_info,
        collision_proxies=collision_proxies,
        start=start,
        goal=start,
        margin_m=0.0,
    )
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    inset = max(1.0, required_clearance_m * 2.0)
    candidates = [
        (sx, max_y - inset if sy <= center_y else min_y + inset, sz),
        (center_x, max_y - inset if sy <= center_y else min_y + inset, sz),
        (max_x - inset if sx <= center_x else min_x + inset, sy, sz),
        (max_x - inset if sx <= center_x else min_x + inset, center_y, sz),
    ]
    for candidate in candidates:
        sample = _clearance_sample(point_xy=candidate[:2], collision_proxies=collision_proxies)
        clearance = sample.get("clearance_m")
        if clearance is not None and float(clearance) >= float(required_clearance_m):
            return _rounded_pose(candidate)
    return None


def _navigation_command(
    *,
    route_waypoints: Sequence[Sequence[float]],
    base_position: Sequence[float],
    base_yaw: float,
    waypoint_index: int,
    max_speed_mps: float,
    waypoint_tolerance_m: float,
    yaw_gain: float,
    max_yaw_rate: float,
) -> dict[str, Any]:
    if len(route_waypoints) < 2:
        return {
            "command_xyz": [0.0, 0.0, 0.0],
            "waypoint_index": 0,
            "active_waypoint": None,
            "goal_reached": False,
            "distance_to_active_waypoint_m": None,
            "distance_to_goal_m": None,
        }
    index = max(1, min(int(waypoint_index), len(route_waypoints) - 1))
    bx, by = float(base_position[0]), float(base_position[1])
    while index < len(route_waypoints) - 1:
        waypoint = route_waypoints[index]
        if math.hypot(float(waypoint[0]) - bx, float(waypoint[1]) - by) > waypoint_tolerance_m:
            break
        index += 1
    waypoint = route_waypoints[index]
    goal = route_waypoints[-1]
    dx = float(waypoint[0]) - bx
    dy = float(waypoint[1]) - by
    distance = math.hypot(dx, dy)
    distance_to_goal = math.hypot(float(goal[0]) - bx, float(goal[1]) - by)
    goal_reached = index == len(route_waypoints) - 1 and distance <= waypoint_tolerance_m
    if goal_reached:
        return {
            "command_xyz": [0.0, 0.0, 0.0],
            "waypoint_index": index,
            "active_waypoint": list(_rounded_pose(waypoint)),
            "goal_reached": True,
            "distance_to_active_waypoint_m": round(distance, 6),
            "distance_to_goal_m": round(distance_to_goal, 6),
        }
    desired_yaw = math.atan2(dy, dx) if distance > 1e-9 else float(base_yaw)
    yaw_error = math.atan2(math.sin(desired_yaw - base_yaw), math.cos(desired_yaw - base_yaw))
    speed = min(float(max_speed_mps), max(0.08, distance * 0.8))
    world_vx = math.cos(desired_yaw) * speed
    world_vy = math.sin(desired_yaw) * speed
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    body_vx = cos_yaw * world_vx + sin_yaw * world_vy
    body_vy = -sin_yaw * world_vx + cos_yaw * world_vy
    if abs(yaw_error) > 0.75:
        body_vx *= 0.55
        body_vy *= 0.55
    body_vy = max(-0.30, min(0.30, body_vy))
    yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_gain * yaw_error))
    return {
        "command_xyz": [float(body_vx), float(body_vy), float(yaw_rate)],
        "waypoint_index": index,
        "active_waypoint": list(_rounded_pose(waypoint)),
        "goal_reached": False,
        "desired_yaw_rad": round(desired_yaw, 6),
        "yaw_error_rad": round(yaw_error, 6),
        "distance_to_active_waypoint_m": round(distance, 6),
        "distance_to_goal_m": round(distance_to_goal, 6),
    }


def _write_camera_robot_xml(source_robot_xml: Path, output_robot_xml: Path) -> None:
    tree = ET.parse(source_robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_dir = source_robot_xml.parent / "meshes"
    if compiler is not None:
        compiler.set("meshdir", str(mesh_dir))
    for mesh in root.findall(".//mesh"):
        mesh_file = _string(mesh.get("file"))
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.set("file", str(mesh_dir / mesh_file))
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError("official Unitree G1 XML does not contain body name 'pelvis'")
    existing = {_string(camera.get("name")) for camera in pelvis.findall("camera")}
    for name, spec in CAMERA_SPECS.items():
        if name in existing:
            continue
        ET.SubElement(
            pelvis,
            "camera",
            {
                "name": name,
                "pos": spec["pos"],
                "xyaxes": spec["xyaxes"],
                "fovy": spec["fovy"],
            },
        )
    ensure_dir(output_robot_xml.parent)
    tree.write(output_robot_xml, encoding="utf-8", xml_declaration=False)


def _write_camera_scene_xml(
    source_scene_xml: Path,
    robot_xml: Path,
    output_scene_xml: Path,
    *,
    render_width: int,
    render_height: int,
    external_scene_obj: Path | None = None,
    external_collision_proxies: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    external_asset_block = ""
    external_worldbody_block = ""
    proxy_worldbody_block = ""
    for index, proxy in enumerate(external_collision_proxies or []):
        pos = proxy.get("pos")
        size = proxy.get("size")
        if not (
            isinstance(pos, Sequence)
            and not isinstance(pos, (str, bytes))
            and isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(pos) >= 3
            and len(size) >= 3
        ):
            continue
        proxy_worldbody_block += (
            "\n"
            f'    <geom name="blueprint_external_collision_proxy_{index:03d}" '
            f'type="box" pos="{_xml_vec(pos[:3])}" size="{_xml_vec(size[:3])}" '
            'rgba="0.05 0.75 0.35 0.24" contype="1" conaffinity="1" group="3"/>'
        )
    if external_scene_obj is not None:
        external_asset_block = f"""
    <mesh name="blueprint_external_scene_mesh" file="{_xml_escape(external_scene_obj)}"/>
    <material name="blueprint_external_scene_mat" rgba="0.45 0.50 0.55 1"/>
    <material name="blueprint_external_scene_collision_mat" rgba="0.05 0.75 0.35 0.24"/>"""
        collision_worldbody_block = proxy_worldbody_block or """
    <geom name="blueprint_external_scene_collision" type="mesh" mesh="blueprint_external_scene_mesh"
      material="blueprint_external_scene_collision_mat" contype="1" conaffinity="1" group="3"/>"""
        external_worldbody_block = """
    <geom name="blueprint_external_scene_visual" type="mesh" mesh="blueprint_external_scene_mesh"
      material="blueprint_external_scene_mat" contype="0" conaffinity="0"/>
""" + collision_worldbody_block
    wrapper = f"""<mujoco model="blueprint_official_unitree_g1_policy_handoff">
  <include file="{_xml_escape(robot_xml)}"/>
  <statistic center="1.0 0.7 1.0" extent="0.8"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.20 0.20 0.20" specular="0.6 0.6 0.6"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20" offwidth="{int(render_width)}" offheight="{int(render_height)}"/>
    <map znear="0.01" zfar="200"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
{external_asset_block}
  </asset>
  <worldbody>
    <light name="blueprint_policy_key" pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{external_worldbody_block}
    <camera name="overview" pos="-2.8 -4.2 2.4" xyaxes="0.83 -0.56 0 0.27 0.40 0.88" fovy="55"/>
  </worldbody>
</mujoco>
"""
    ensure_dir(output_scene_xml.parent)
    output_scene_xml.write_text(wrapper, encoding="utf-8")
    _ = source_scene_xml


def _gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def _pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def _yaw_from_quat_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = [float(value) for value in quaternion]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _joint_addresses(model: Any, mujoco: Any) -> list[dict[str, Any]]:
    addresses: list[dict[str, Any]] = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"official Unitree G1 model missing joint: {name}")
        addresses.append(
            {
                "name": name,
                "joint_id": int(joint_id),
                "qpos_addr": int(model.jnt_qposadr[joint_id]),
                "dof_addr": int(model.jnt_dofadr[joint_id]),
            }
        )
    return addresses


def _body_id(model: Any, mujoco: Any, name: str) -> int | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return int(body_id) if body_id >= 0 else None


def _body_record(model: Any, data: Any, mujoco: Any, name: str) -> dict[str, Any] | None:
    body_id = _body_id(model, mujoco, name)
    if body_id is None:
        return None
    return {
        "body_name": name,
        "position_xyz": _as_float_list(data.xpos[body_id]),
        "orientation_quat_wxyz": _as_float_list(data.xquat[body_id]),
        "cvel": _as_float_list(data.cvel[body_id]),
    }


def _contact_records(model: Any, data: Any, mujoco: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    foot_summary: dict[str, dict[str, Any]] = {
        name: {
            "in_contact": False,
            "contact_count": 0,
            "normal_force_estimate_sum": 0.0,
            "slip_indicator": False,
        }
        for name in FOOT_BODY_NAMES
    }
    foot_body_ids = {
        name: _body_id(model, mujoco, name)
        for name in FOOT_BODY_NAMES
        if _body_id(model, mujoco, name) is not None
    }
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        body_ids = [int(model.geom_bodyid[geom_id]) for geom_id in geom_ids]
        body_names = [
            _string(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            or f"body_{body_id}"
            for body_id in body_ids
        ]
        geom_names = [
            _string(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            or f"geom_{geom_id}"
            for geom_id in geom_ids
        ]
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        record = {
            "contact_index": index,
            "geom_ids": geom_ids,
            "geom_names": geom_names,
            "body_ids": body_ids,
            "body_names": body_names,
            "distance": float(contact.dist),
            "position_xyz": _as_float_list(contact.pos),
            "frame": _as_float_list(contact.frame),
            "contact_force_6d": _as_float_list(force),
            "normal_force_estimate": float(force[0]),
            "foot_contact": any(body_id in foot_body_ids.values() for body_id in body_ids),
            "scene_collision_contact": any(
                name == "blueprint_external_scene_collision"
                or name.startswith("blueprint_external_collision_proxy_")
                for name in geom_names
            ),
        }
        records.append(record)
        for foot_name, foot_body_id in foot_body_ids.items():
            if foot_body_id not in body_ids:
                continue
            linear_velocity = np.asarray(data.cvel[foot_body_id][3:6], dtype=float)
            horizontal_speed = float(np.linalg.norm(linear_velocity[:2]))
            foot_summary[foot_name]["in_contact"] = True
            foot_summary[foot_name]["contact_count"] += 1
            foot_summary[foot_name]["normal_force_estimate_sum"] += float(force[0])
            foot_summary[foot_name]["slip_indicator"] = (
                bool(foot_summary[foot_name]["slip_indicator"]) or horizontal_speed > 0.20
            )
            foot_summary[foot_name]["horizontal_speed_mps"] = horizontal_speed
    return records, {
        name: {
            **value,
            "normal_force_estimate_sum": round(float(value["normal_force_estimate_sum"]), 6),
        }
        for name, value in foot_summary.items()
    }


def _observation(
    *,
    data: Any,
    action: np.ndarray,
    default_angles: np.ndarray,
    dof_pos_scale: float,
    dof_vel_scale: float,
    ang_vel_scale: float,
    cmd: np.ndarray,
    cmd_scale: np.ndarray,
    counter: int,
    simulation_dt: float,
    num_actions: int,
    num_obs: int,
) -> np.ndarray:
    obs = np.zeros(num_obs, dtype=np.float32)
    qj = np.asarray(data.qpos[7 : 7 + num_actions], dtype=np.float32)
    dqj = np.asarray(data.qvel[6 : 6 + num_actions], dtype=np.float32)
    quat = np.asarray(data.qpos[3:7], dtype=np.float32)
    omega = np.asarray(data.qvel[3:6], dtype=np.float32)
    qj = (qj - default_angles) * dof_pos_scale
    dqj = dqj * dof_vel_scale
    gravity_orientation = _gravity_orientation(quat)
    omega = omega * ang_vel_scale
    period = 0.8
    count = counter * simulation_dt
    phase = count % period / period
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)
    obs[:3] = omega
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd * cmd_scale
    obs[9 : 9 + num_actions] = qj
    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
        [sin_phase, cos_phase],
        dtype=np.float32,
    )
    return obs


def _render_side_camera(mujoco: Any) -> Any:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.9, 0.0, 0.85]
    camera.distance = 3.2
    camera.azimuth = 90
    camera.elevation = -12
    return camera


def _render_follow_camera(mujoco: Any, position: Sequence[float]) -> Any:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(position[0]) + 0.55, float(position[1]), float(position[2]) + 0.55]
    camera.distance = 1.45
    camera.azimuth = 115
    camera.elevation = -10
    return camera


def _camera_output_path(
    *,
    camera: str,
    sample_index: int,
    frames_dir: Path,
    robot_pov_frames_dir: Path,
) -> Path:
    if camera in {"robot_pov_head", "robot_pov_torso"}:
        return robot_pov_frames_dir / camera.replace("robot_pov_", "") / f"{camera}_{sample_index:04d}.png"
    return frames_dir / f"official_policy_{camera}_{sample_index:04d}.png"


def _render_frame(
    *,
    renderer: Any,
    mujoco: Any,
    data: Any,
    camera: str,
    base_position: Sequence[float],
) -> np.ndarray:
    if camera == "overview":
        renderer.update_scene(data, camera="overview")
    elif camera == "side":
        renderer.update_scene(data, camera=_render_side_camera(mujoco))
    elif camera == "follow":
        renderer.update_scene(data, camera=_render_follow_camera(mujoco, base_position))
    elif camera in {"robot_pov_head", "robot_pov_torso"}:
        renderer.update_scene(data, camera=camera)
    else:
        raise ValueError(f"unsupported camera: {camera}")
    return renderer.render()


def _video_encoding_settings(*, render_fps: int, video_crf: int) -> dict[str, Any]:
    return {
        "fps": int(render_fps),
        "video_crf": int(video_crf),
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "movflags": "+faststart",
    }


def _frame_durations_for_realtime_video(
    *,
    frame_count: int,
    render_fps: int,
    frame_times_s: Sequence[float] | None = None,
    video_duration_s: float | None = None,
) -> tuple[list[float], dict[str, Any]]:
    fallback_duration = 1.0 / max(1, int(render_fps))
    if frame_count <= 0:
        return [], {"mode": "no_frames"}
    if not frame_times_s or len(frame_times_s) != frame_count:
        return [fallback_duration] * frame_count, {
            "mode": "fixed_render_fps",
            "expected_video_duration_s": round(fallback_duration * frame_count, 9),
        }
    times = [float(value) for value in frame_times_s]
    monotonic = all(next_time > time for time, next_time in zip(times, times[1:]))
    if not monotonic:
        return [fallback_duration] * frame_count, {
            "mode": "fixed_render_fps_non_monotonic_source_times",
            "expected_video_duration_s": round(fallback_duration * frame_count, 9),
        }
    durations = [
        max(0.000001, next_time - time) for time, next_time in zip(times, times[1:])
    ]
    if frame_count == 1:
        durations.append(fallback_duration)
    elif video_duration_s is not None:
        final_duration = float(video_duration_s) - (times[-1] - times[0])
        durations.append(max(0.000001, final_duration))
    else:
        durations.append(durations[-1])
    expected_duration = sum(durations)
    return durations, {
        "mode": "source_sim_time_realtime",
        "first_frame_sim_time_s": round(times[0], 9),
        "last_frame_sim_time_s": round(times[-1], 9),
        "expected_video_duration_s": round(expected_duration, 9),
        "source_video_duration_s": round(float(video_duration_s), 9)
        if video_duration_s is not None
        else None,
    }


def _write_frame_video(
    *,
    camera: str,
    frame_paths: Sequence[str],
    output_dir: Path,
    render_fps: int,
    video_crf: int,
    frame_times_s: Sequence[float] | None = None,
    video_duration_s: float | None = None,
) -> dict[str, Any]:
    if len(frame_paths) < 2:
        return {
            "status": "not_generated",
            "reason": "requires_at_least_two_frames",
            "frame_count": len(frame_paths),
            "encoding": _video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
        }
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "status": "not_generated",
            "reason": "ffmpeg_unavailable",
            "frame_count": len(frame_paths),
            "encoding": _video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
        }
    ensure_dir(output_dir)
    concat_path = output_dir / f"{camera}_video_frames.txt"
    video_path = output_dir / f"{camera}.mp4"
    durations, timing = _frame_durations_for_realtime_video(
        frame_count=len(frame_paths),
        render_fps=render_fps,
        frame_times_s=frame_times_s,
        video_duration_s=video_duration_s,
    )
    lines: list[str] = []
    for frame_path, duration in zip(frame_paths, durations):
        escaped = str(Path(frame_path)).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {duration:.9f}")
    escaped_last = str(Path(frame_paths[-1])).replace("'", "'\\''")
    lines.append(f"file '{escaped_last}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={int(render_fps)},format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        str(int(video_crf)),
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        return {
            "status": "not_generated",
            "reason": "ffmpeg_failed",
            "stderr_tail": completed.stderr[-1000:],
            "frame_count": len(frame_paths),
            "encoding": _video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
            "video_timing": timing,
        }
    return {
        "status": "complete",
        "path": str(video_path),
        "concat_list_path": str(concat_path),
        "frame_count": len(frame_paths),
        "encoding": _video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
        "video_timing": timing,
        "size_bytes": video_path.stat().st_size,
        "source_frames": list(frame_paths),
    }


def _ffprobe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.is_file():
        return {"status": "not_checked", "reason": "ffprobe_unavailable_or_missing_video"}
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return {
            "status": "not_checked",
            "reason": "ffprobe_failed",
            "stderr_tail": completed.stderr[-1000:],
        }
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    stream = streams[0] if isinstance(streams, list) and streams else {}
    return {"status": "checked", **dict(stream)}


def _camera_set(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        raw = ["overview", "side", "follow", "robot_pov"]
    elif isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw = [part.strip() for item in value for part in str(item).split(",") if part.strip()]
    expanded: list[str] = []
    for camera in raw:
        if camera == "robot_pov":
            expanded.extend(["robot_pov_head", "robot_pov_torso"])
        else:
            expanded.append(camera)
    allowed = {"overview", "side", "follow", "robot_pov_head", "robot_pov_torso"}
    unsupported = [camera for camera in expanded if camera not in allowed]
    if unsupported:
        raise ValueError(f"unsupported camera-set value(s): {', '.join(unsupported)}")
    return list(dict.fromkeys(expanded))


def _stream_gate(rows: Sequence[Mapping[str, Any]], *, control_update_count: int) -> dict[str, Any]:
    blockers: list[str] = []
    if not rows:
        blockers.append("missing_timeseries_rows")
        sample = {}
    else:
        sample = rows[0]
    required = {
        "qpos": "missing_qpos_stream",
        "qvel": "missing_qvel_stream",
        "joint_positions": "missing_joint_position_stream",
        "joint_velocities": "missing_joint_velocity_stream",
        "actuator_controls": "missing_actuator_control_stream",
        "actuator_forces": "missing_actuator_force_stream",
        "foot_contact_states": "missing_foot_contact_stream",
        "command_xyz": "missing_command_stream",
    }
    for field, blocker in required.items():
        if field not in sample:
            blockers.append(blocker)
    if control_update_count <= 0:
        blockers.append("missing_policy_observation_stream")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "required_streams": sorted(required),
        "policy_observation_stream_required": True,
        "control_update_count": control_update_count,
    }


def _build_sensor_stream_manifest(
    *,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    control_update_count: int,
    row_count: int,
) -> dict[str, Any]:
    gate = _stream_gate(rows, control_update_count=control_update_count)
    manifest = {
        "schema_version": "official_unitree_g1_sensor_stream_manifest.v1",
        "status": "complete" if gate["passed"] else "blocked",
        "row_count": row_count,
        "control_update_count": control_update_count,
        "streams": {
            "qpos": {"status": "complete", "sample_hz": 500},
            "qvel": {"status": "complete", "sample_hz": 500},
            "joint_positions": {"status": "complete", "joint_names": list(JOINT_NAMES)},
            "joint_velocities": {"status": "complete", "joint_names": list(JOINT_NAMES)},
            "actuator_controls": {"status": "complete", "actuator_names": list(JOINT_NAMES)},
            "actuator_forces": {"status": "complete", "actuator_names": list(JOINT_NAMES)},
            "policy_observations": {
                "status": "available_at_control_updates",
                "control_update_count": control_update_count,
                "missing_policy_observation_stream": control_update_count <= 0,
            },
            "contacts": {
                "status": "complete",
                "contact_pairs_included": True,
                "contact_force_estimates_included": True,
                "slip_indicators_derivable": True,
            },
            "body_kinematics": {
                "status": "partial",
                "available_bodies": ["pelvis"],
                "missing_named_bodies": ["torso", "head"],
                "boundary": (
                    "The official 12-DoF XML represents torso/head meshes under the pelvis body; "
                    "no separate named torso/head body exists for body kinematics."
                ),
            },
        },
        "handoff_gate": gate,
        "timeseries_path": str(path),
    }
    return manifest


def _robot_pov_manifest(
    *,
    camera_records: Sequence[Mapping[str, Any]],
    robot_pov_frames: Mapping[str, Sequence[str]],
    robot_pov_videos: Mapping[str, Mapping[str, Any]],
    render_width: int,
    render_height: int,
    render_fps: int,
    nonblank_checks: Mapping[str, Any],
    calibration_path: Path,
) -> dict[str, Any]:
    frame_count = sum(len(paths) for paths in robot_pov_frames.values())
    return {
        "schema_version": ROBOT_POV_MANIFEST_SCHEMA_VERSION,
        "status": "complete" if frame_count else "not_recorded",
        "simulated_robot_pov": bool(frame_count),
        "real_robot_pov": False,
        "physical_sensor_data": False,
        "camera_body_name": "pelvis",
        "camera_body_mount_boundary": (
            "Cameras are MuJoCo fixed cameras mounted on the named pelvis body. "
            "Head/torso labels are proxy offsets because the official G1 XML has no "
            "separate named head or torso body."
        ),
        "camera_pose_relative_to_body": {
            name: {
                "body_name": spec["body_name"],
                "mount_label": spec["mount_label"],
                "pos": spec["pos"],
                "xyaxes": spec["xyaxes"],
                "fovy": spec["fovy"],
            }
            for name, spec in CAMERA_SPECS.items()
        },
        "render_resolution": [int(render_width), int(render_height)],
        "render_fps": int(render_fps),
        "frame_count": frame_count,
        "frames": {key: list(paths) for key, paths in robot_pov_frames.items()},
        "videos": {key: dict(value) for key, value in robot_pov_videos.items()},
        "camera_records": [dict(record) for record in camera_records],
        "nonblank_checks": dict(nonblank_checks),
        "all_frames_nonblank": nonblank_checks.get("all_frames_nonblank") is True,
        "calibration_path": str(calibration_path),
        "proof_boundary": {
            "simulated_robot_pov": True,
            "real_robot_pov": False,
            "physical_sensor_data": False,
            "physical_robot_readiness_proven": False,
        },
    }


def _worldlabs_asset_overlay_manifest(capture_root: Path, output_path: Path) -> dict[str, Any]:
    assets_dir = capture_root / "pipeline" / "worldlabs_assets"
    materialized_path = assets_dir / "materialized_assets_manifest.json"
    materialized = optional_read_json(materialized_path) or {}
    downloads = materialized.get("downloads") if isinstance(materialized.get("downloads"), list) else []
    collider = next(
        (
            dict(item)
            for item in downloads
            if isinstance(item, Mapping) and _string(item.get("kind")) == "collider_mesh_glb"
        ),
        {},
    )
    spz_assets = [
        dict(item)
        for item in downloads
        if isinstance(item, Mapping) and _string(item.get("kind")) == "splat_spz"
    ]
    collider_path = Path(_string(collider.get("local_path")))
    glb_summary = _glb_visual_summary(collider_path) if collider_path.is_file() else {}
    textured_mesh_available = bool(
        glb_summary.get("textures_count") or glb_summary.get("images_count")
    )
    blockers = []
    if not textured_mesh_available:
        blockers.append("blocked_missing_high_quality_textured_mesh_export")
    if not spz_assets:
        blockers.append("missing_spz_visual_overlay_assets")
    manifest = {
        "schema_version": WORLDLABS_OVERLAY_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "complete" if spz_assets else "blocked",
        "source_materialized_assets_manifest": str(materialized_path),
        "world_id": materialized.get("world_id"),
        "collider_glb": collider,
        "collider_glb_visual_summary": glb_summary,
        "spz_assets": spz_assets,
        "website_viewer_overlay_ready": bool(spz_assets and collider),
        "mujoco_textured_visuals_proven": textured_mesh_available,
        "mujoco_texture_claim_allowed": textured_mesh_available,
        "mujoco_visual_boundary": (
            "SPZ assets are suitable for website/viewer overlay. MuJoCo textured rendering "
            "is not claimed unless a textured mesh GLB with image/texture evidence exists."
        ),
        "blockers": blockers,
    }
    _safe_write_json(output_path, manifest)
    return manifest


def build_official_g1_policy_handoff(
    *,
    capture_root: str | Path,
    policy_manifest_path: str | Path | None = None,
    unitree_rl_gym_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    render_width: int = 1280,
    render_height: int = 720,
    render_fps: int = 24,
    video_crf: int = 18,
    max_frames: int = 120,
    camera_set: Sequence[str] | str | None = None,
    duration_seconds: float | None = None,
    target_displacement_m: float | None = None,
    fall_height_threshold_m: float = 0.45,
    command_xyz: Sequence[float] | None = None,
    collision_proxy_limit: int = 512,
    base_path_clearance_m: float = 0.38,
    initial_root_xy: Sequence[float] | None = None,
    initial_root_yaw: float = 0.0,
    navigation_goal_xyz: Sequence[float] | None = None,
    navigation_grid_resolution_m: float = 0.35,
    navigation_max_speed_mps: float = 0.55,
    navigation_waypoint_tolerance_m: float = 0.35,
    navigation_yaw_gain: float = 1.2,
    navigation_max_yaw_rate: float = 0.9,
    enable_navigation_planner: bool = True,
    copy_policy_source_snapshot: bool = True,
) -> dict[str, Any]:
    if platform.system().lower() == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        import mujoco  # type: ignore[import-not-found]
        import torch
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency guard.
        raise RuntimeError("mujoco, torch, and Pillow are required for G1 handoff") from exc

    root = Path(capture_root).expanduser().resolve()
    policy_dir = root / DEFAULT_POLICY_RELATIVE
    handoff_dir = Path(output_dir).expanduser().resolve() if output_dir else policy_dir / "robot_team_handoff"
    rendered_dir = handoff_dir / "rendered_policy_motion"
    frames_dir = rendered_dir / "frames"
    robot_pov_frames_dir = handoff_dir / "robot_pov_frames"
    generated_mjcf_dir = handoff_dir / "generated_mjcf"
    ensure_dir(handoff_dir)
    ensure_dir(frames_dir)
    ensure_dir(robot_pov_frames_dir)
    ensure_dir(generated_mjcf_dir)

    manifest_path = (
        Path(policy_manifest_path).expanduser().resolve()
        if policy_manifest_path
        else policy_dir / "official_unitree_g1_policy_execution_manifest.json"
    )
    policy_manifest = _load_json(manifest_path)
    execution = _mapping(policy_manifest.get("execution"))
    metrics = _mapping(policy_manifest.get("metrics"))
    original_trace_path = Path(_string(execution.get("trace_path")) or policy_dir / "policy_execution_trace.jsonl")
    original_trace_rows = _trace_rows(original_trace_path)

    policy_root = _resolve_policy_root(
        explicit_root=unitree_rl_gym_root,
        manifest=policy_manifest,
        handoff_dir=handoff_dir,
    )
    snapshot_manifest = (
        _materialize_policy_snapshot(policy_root, handoff_dir)
        if copy_policy_source_snapshot
        else {"status": "skipped", "portable_provider_rerun_ready": False}
    )
    if snapshot_manifest.get("status") == "complete":
        policy_root = Path(str(snapshot_manifest["snapshot_root"])).resolve()
    paths = _policy_paths(policy_root, policy_manifest)
    if not _string(policy_manifest.get("policy_id")):
        policy_manifest["policy_id"] = "unitree_rl_gym_g1_pretrain_motion"
    if not _string(policy_manifest.get("robot_profile_id")):
        policy_manifest["robot_profile_id"] = "unitree_g1_humanoid"
    if not _string(policy_manifest.get("robot_make_model")):
        policy_manifest["robot_make_model"] = "Unitree G1"
    source_repository = {
        "name": "unitree_rl_gym",
        "url": "https://github.com/unitreerobotics/unitree_rl_gym",
        "local_inspection_root": str(policy_root),
        **_mapping(policy_manifest.get("source_repository")),
    }
    source_repository["resolved_local_root"] = str(policy_root)
    policy_manifest["source_repository"] = source_repository
    official_artifacts = {
        **_mapping(policy_manifest.get("official_artifacts")),
        "config_path": str(paths["config"]),
        "policy_path": str(paths["policy"]),
        "xml_path": str(paths["xml"]),
    }
    policy_manifest["official_artifacts"] = official_artifacts
    config = _load_policy_config(paths["config"], policy_root)
    source_robot_xml = paths["xml"].parent / "g1_12dof.xml"
    generated_robot_xml = generated_mjcf_dir / "g1_12dof_with_robot_pov_cameras.xml"
    generated_scene_xml = generated_mjcf_dir / "official_unitree_g1_policy_camera_scene.xml"
    external_scene_mesh_info: dict[str, Any] = {
        "status": "not_loaded",
        "reason": "capture_scene_glb_not_found",
    }
    external_scene_obj: Path | None = None
    try:
        external_scene_glb = _find_scene_glb(root)
        external_scene_obj = handoff_dir / "capture_scene_for_official_g1_policy.obj"
        external_scene_mesh_info = _convert_glb_to_obj(
            external_scene_glb,
            external_scene_obj,
            collision_proxy_limit=collision_proxy_limit,
        )
        external_collision_proxies = list(
            external_scene_mesh_info.get("collision_proxy_geoms") or []
        )
        external_scene_mesh_info["collision_geometry_contract"] = {
            "visual_geom": "blueprint_external_scene_visual",
            "collision_geom": "blueprint_external_scene_collision"
            if not external_collision_proxies
            else None,
            "same_mesh_for_visual_and_collision": not bool(external_collision_proxies),
            "proxy_collision_model_used": bool(external_collision_proxies),
            "collision_proxy_geom_count": len(external_collision_proxies),
        }
    except FileNotFoundError:
        external_scene_obj = None
        external_collision_proxies = []
    _write_camera_robot_xml(source_robot_xml, generated_robot_xml)
    _write_camera_scene_xml(
        paths["xml"],
        generated_robot_xml,
        generated_scene_xml,
        render_width=render_width,
        render_height=render_height,
        external_scene_obj=external_scene_obj,
        external_collision_proxies=external_collision_proxies,
    )

    matrix_path = root / DEFAULT_MATRIX_RELATIVE
    scenario_context = _scenario_context(matrix_path)
    selected_matrix_run = _first_matrix_run(matrix_path)
    model = mujoco.MjModel.from_xml_path(str(generated_scene_xml))
    data = mujoco.MjData(model)
    default_start_z = float(data.qpos[2]) if model.nq >= 3 else 0.793
    matrix_start = _nested_pose(
        selected_matrix_run,
        (
            "spawn_pose",
            "start_pose",
            "initial_pose",
            "robot_spawn_pose",
            "robot_start_pose",
            "start_xyz",
            "spawn_xyz",
        ),
    )
    matrix_goal = _nested_pose(
        selected_matrix_run,
        (
            "target_pose",
            "goal_pose",
            "navigation_target_pose",
            "robot_target_pose",
            "target_xyz",
            "goal_xyz",
        ),
    )
    explicit_goal = _pose_triplet(navigation_goal_xyz)
    if initial_root_xy is not None and len(initial_root_xy) >= 2:
        initial_root_pose = (
            float(initial_root_xy[0]),
            float(initial_root_xy[1]),
            matrix_start[2] if matrix_start is not None else default_start_z,
        )
    else:
        initial_root_pose = matrix_start or (
            float(data.qpos[0]) if model.nq >= 1 else 0.0,
            float(data.qpos[1]) if model.nq >= 2 else 0.0,
            default_start_z,
        )
    goal_pose = explicit_goal or matrix_goal
    if goal_pose is None and enable_navigation_planner:
        goal_pose = _default_navigation_goal(
            start=initial_root_pose,
            mesh_info=external_scene_mesh_info,
            collision_proxies=external_collision_proxies,
            required_clearance_m=base_path_clearance_m,
        )
    navigation_plan = (
        _plan_occupancy_grid_route(
            start=initial_root_pose,
            goal=goal_pose,
            collision_proxies=external_collision_proxies,
            mesh_info=external_scene_mesh_info,
            required_clearance_m=base_path_clearance_m,
            grid_resolution_m=navigation_grid_resolution_m,
        )
        if enable_navigation_planner and goal_pose is not None
        else {
            "schema_version": "official_unitree_g1_navigation_plan.v1",
            "status": "not_requested",
            "planned": False,
            "blockers": ["navigation_goal_not_available"],
            "start_pose": list(_rounded_pose(initial_root_pose)),
            "goal_pose": None,
            "route_waypoints": [],
        }
    )
    if navigation_plan.get("planned") is True:
        planned_start = navigation_plan.get("start_pose") or initial_root_pose
        data.qpos[0] = float(planned_start[0])
        data.qpos[1] = float(planned_start[1])
        if model.nq >= 3:
            data.qpos[2] = float(planned_start[2])
    elif initial_root_xy is not None and len(initial_root_xy) >= 2:
        data.qpos[0] = float(initial_root_pose[0])
        data.qpos[1] = float(initial_root_pose[1])
    if model.nq >= 7:
        yaw = float(initial_root_yaw)
        data.qpos[3:7] = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
        mujoco.mj_forward(model, data)
    model.opt.timestep = float(config.get("simulation_dt") or 0.002)
    policy = torch.jit.load(str(paths["policy"]), map_location="cpu")
    policy.eval()

    simulation_dt = float(config.get("simulation_dt") or model.opt.timestep or 0.002)
    control_decimation = int(config.get("control_decimation") or 10)
    requested_duration = float(
        duration_seconds
        if duration_seconds is not None
        else metrics.get("duration_seconds_requested")
        or metrics.get("sim_time_s")
        or 4.0
    )
    total_steps = max(1, int(round(requested_duration / simulation_dt)))
    kps = np.asarray(config.get("kps") or [100] * len(JOINT_NAMES), dtype=np.float32)
    kds = np.asarray(config.get("kds") or [2] * len(JOINT_NAMES), dtype=np.float32)
    default_angles = np.asarray(config.get("default_angles") or [0] * len(JOINT_NAMES), dtype=np.float32)
    cmd = np.asarray(
        command_xyz
        if command_xyz is not None
        else config.get("cmd_init") or metrics.get("command_xyz") or [0.5, 0.0, 0.0],
        dtype=np.float32,
    )
    cmd_scale = np.asarray(config.get("cmd_scale") or [2.0, 2.0, 0.25], dtype=np.float32)
    dof_pos_scale = float(config.get("dof_pos_scale") or 1.0)
    dof_vel_scale = float(config.get("dof_vel_scale") or 0.05)
    ang_vel_scale = float(config.get("ang_vel_scale") or 0.25)
    action_scale = float(config.get("action_scale") or 0.25)
    num_actions = int(config.get("num_actions") or len(JOINT_NAMES))
    num_obs = int(config.get("num_obs") or 47)
    joint_addresses = _joint_addresses(model, mujoco)
    actuator_names = [
        _string(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index))
        or JOINT_NAMES[index]
        for index in range(model.nu)
    ]
    cameras = _camera_set(camera_set)
    sample_indices = sorted(_render_capture_steps(total_steps, max_rendered_steps=max_frames))
    renderer = mujoco.Renderer(model, height=int(render_height), width=int(render_width))

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    control_update_index: int | None = None
    control_update_count = 0
    rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    camera_records: list[dict[str, Any]] = []
    camera_frame_paths: dict[str, list[str]] = {camera: [] for camera in cameras}
    camera_frame_times: dict[str, list[float]] = {camera: [] for camera in cameras}
    base_positions: list[list[float]] = []
    navigation_active = navigation_plan.get("planned") is True
    navigation_waypoints = [
        _rounded_pose(point)
        for point in navigation_plan.get("route_waypoints", [])
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 3
    ]
    navigation_waypoint_index = 1 if len(navigation_waypoints) > 1 else 0
    navigation_goal_reached = False
    navigation_command_samples: list[dict[str, Any]] = []
    navigation_clearance_samples: list[dict[str, Any]] = []
    navigation_clearance_violation_count = 0
    navigation_min_runtime_clearance: dict[str, Any] | None = None
    finite_state = True
    finite_actions = True
    initial_base_position_xy: np.ndarray | None = None
    episode_termination_reason = "timeout"
    episode_termination_step: int | None = None
    episode_termination_time_s: float | None = None
    try:
        with torch.no_grad():
            for step in range(total_steps):
                q = np.asarray(data.qpos[7 : 7 + num_actions], dtype=np.float32)
                dq = np.asarray(data.qvel[6 : 6 + num_actions], dtype=np.float32)
                tau = _pd_control(target_dof_pos, q, kps, np.zeros_like(kds), dq, kds)
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)
                counter = step + 1
                policy_update_applied = counter % control_decimation == 0
                base_position = _as_float_list(data.qpos[0:3])
                base_yaw_rad = _yaw_from_quat_wxyz(_as_float_list(data.qpos[3:7]))
                navigation_command_record: dict[str, Any] | None = None
                runtime_clearance = _clearance_sample(
                    point_xy=base_position[:2],
                    collision_proxies=external_collision_proxies,
                )
                runtime_clearance_m = runtime_clearance.get("clearance_m")
                if runtime_clearance_m is not None:
                    runtime_clearance = {
                        **runtime_clearance,
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                    }
                    if (
                        navigation_min_runtime_clearance is None
                        or float(runtime_clearance_m)
                        < float(navigation_min_runtime_clearance["clearance_m"])
                    ):
                        navigation_min_runtime_clearance = dict(runtime_clearance)
                    if float(runtime_clearance_m) < float(base_path_clearance_m):
                        navigation_clearance_violation_count += 1
                    if policy_update_applied or float(runtime_clearance_m) < float(base_path_clearance_m):
                        navigation_clearance_samples.append(dict(runtime_clearance))
                if navigation_active:
                    navigation_command_record = _navigation_command(
                        route_waypoints=navigation_waypoints,
                        base_position=base_position,
                        base_yaw=base_yaw_rad,
                        waypoint_index=navigation_waypoint_index,
                        max_speed_mps=navigation_max_speed_mps,
                        waypoint_tolerance_m=navigation_waypoint_tolerance_m,
                        yaw_gain=navigation_yaw_gain,
                        max_yaw_rate=navigation_max_yaw_rate,
                    )
                    navigation_waypoint_index = int(navigation_command_record["waypoint_index"])
                    navigation_goal_reached = (
                        navigation_goal_reached
                        or navigation_command_record.get("goal_reached") is True
                    )
                    cmd = np.asarray(navigation_command_record["command_xyz"], dtype=np.float32)
                    if policy_update_applied or navigation_command_record.get("goal_reached") is True:
                        navigation_command_samples.append(
                            {
                                "step": step,
                                "sim_time_s": round(float(data.time), 9),
                                **navigation_command_record,
                            }
                        )
                policy_observation: list[float] | None = None
                if policy_update_applied:
                    obs = _observation(
                        data=data,
                        action=action,
                        default_angles=default_angles,
                        dof_pos_scale=dof_pos_scale,
                        dof_vel_scale=dof_vel_scale,
                        ang_vel_scale=ang_vel_scale,
                        cmd=cmd,
                        cmd_scale=cmd_scale,
                        counter=counter,
                        simulation_dt=simulation_dt,
                        num_actions=num_actions,
                        num_obs=num_obs,
                    )
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().cpu().numpy().squeeze().astype(np.float32)
                    target_dof_pos = action * action_scale + default_angles
                    control_update_index = control_update_count
                    control_update_count += 1
                    policy_observation = _as_float_list(obs)
                    finite_actions = finite_actions and bool(np.all(np.isfinite(action)))
                finite_state = finite_state and bool(
                    np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
                )
                contacts, foot_contact_states = _contact_records(model, data, mujoco)
                if initial_base_position_xy is None:
                    initial_base_position_xy = np.asarray(base_position[:2], dtype=float)
                base_displacement_xy = float(
                    np.linalg.norm(
                        np.asarray(base_position[:2], dtype=float) - initial_base_position_xy
                    )
                )
                scene_contact_count = sum(
                    1 for contact in contacts if contact.get("scene_collision_contact") is True
                )
                step_termination_reason: str | None = None
                if scene_contact_count > 0:
                    step_termination_reason = "scene_collision"
                elif (
                    navigation_active
                    and runtime_clearance_m is not None
                    and float(runtime_clearance_m) < float(base_path_clearance_m)
                ):
                    step_termination_reason = "clearance_below_threshold"
                elif float(base_position[2]) < float(fall_height_threshold_m):
                    step_termination_reason = "fall_height_below_threshold"
                elif navigation_active and navigation_goal_reached:
                    step_termination_reason = "navigation_goal_reached"
                elif (
                    target_displacement_m is not None
                    and base_displacement_xy >= float(target_displacement_m)
                ):
                    step_termination_reason = "target_displacement_reached"
                base_positions.append(base_position)
                joint_positions = {
                    item["name"]: float(data.qpos[item["qpos_addr"]]) for item in joint_addresses
                }
                joint_velocities = {
                    item["name"]: float(data.qvel[item["dof_addr"]]) for item in joint_addresses
                }
                body_records = {
                    name: record
                    for name in ("pelvis", "torso", "head")
                    if (record := _body_record(model, data, mujoco, name)) is not None
                }
                row = {
                    "schema_version": OFFICIAL_G1_ENRICHED_TRACE_SCHEMA_VERSION,
                    "sim_time_s": float(data.time),
                    "step": step,
                    "control_update_index": control_update_index,
                    "policy_update_applied": policy_update_applied,
                    "qpos": _as_float_list(data.qpos),
                    "qvel": _as_float_list(data.qvel),
                    "base_position_xyz": base_position,
                    "base_orientation_quat_wxyz": _as_float_list(data.qpos[3:7]),
                    "base_yaw_rad": base_yaw_rad,
                    "base_linear_velocity_xyz": _as_float_list(data.qvel[0:3]),
                    "base_angular_velocity_xyz": _as_float_list(data.qvel[3:6]),
                    "joint_positions": joint_positions,
                    "joint_velocities": joint_velocities,
                    "actuator_names": actuator_names,
                    "actuator_controls": _as_float_list(data.ctrl),
                    "actuator_forces": _as_float_list(data.actuator_force),
                    "policy_action": _as_float_list(action),
                    "target_dof_pos": _as_float_list(target_dof_pos),
                    "body_kinematics": body_records,
                    "missing_body_kinematics": [
                        body for body in ("torso", "head") if body not in body_records
                    ],
                    "contact_pairs": contacts,
                    "foot_contact_states": foot_contact_states,
                    "policy_observation": policy_observation,
                    "policy_observation_stream": (
                        "available_at_control_update" if policy_update_applied else "not_sampled_this_step"
                    ),
                    "command_xyz": _as_float_list(cmd),
                    "command_source": (
                        "navigation_planner_waypoint_velocity"
                        if navigation_active
                        else "static_policy_command"
                    ),
                    "navigation": {
                        "active": navigation_active,
                        "planner_status": navigation_plan.get("status"),
                        "waypoint_index": navigation_waypoint_index,
                        "active_waypoint": navigation_command_record.get("active_waypoint")
                        if navigation_command_record
                        else None,
                        "goal_pose": navigation_plan.get("goal_pose"),
                        "goal_reached": navigation_goal_reached,
                        "distance_to_active_waypoint_m": navigation_command_record.get(
                            "distance_to_active_waypoint_m"
                        )
                        if navigation_command_record
                        else None,
                        "distance_to_goal_m": navigation_command_record.get("distance_to_goal_m")
                        if navigation_command_record
                        else None,
                        "runtime_clearance_m": runtime_clearance_m,
                    },
                    "scenario_context": scenario_context["selected_run"],
                    "rendered_frame": step in sample_indices,
                    "base_displacement_xy_m": base_displacement_xy,
                    "scene_collision_contact_count": scene_contact_count,
                    "fall_height_threshold_m": float(fall_height_threshold_m),
                    "target_displacement_m": float(target_displacement_m)
                    if target_displacement_m is not None
                    else None,
                    "episode_termination_candidate": step_termination_reason,
                }
                rows.append(row)
                if policy_update_applied:
                    enriched_rows.append(row)
                if step in sample_indices:
                    sample_index = sample_indices.index(step)
                    for camera in cameras:
                        frame_path = _camera_output_path(
                            camera=camera,
                            sample_index=sample_index,
                            frames_dir=frames_dir,
                            robot_pov_frames_dir=robot_pov_frames_dir,
                        )
                        ensure_dir(frame_path.parent)
                        image = _render_frame(
                            renderer=renderer,
                            mujoco=mujoco,
                            data=data,
                            camera=camera,
                            base_position=base_position,
                        )
                        Image.fromarray(image).save(frame_path)
                        record = {
                            "camera": camera,
                            "path": str(frame_path),
                            "trace_step": step,
                            "sim_time_s": float(data.time),
                            "resolution": [int(render_width), int(render_height)],
                        }
                        frame_records.append(record)
                        camera_frame_paths.setdefault(camera, []).append(str(frame_path))
                        camera_frame_times.setdefault(camera, []).append(float(data.time))
                        if camera.startswith("robot_pov_"):
                            spec = CAMERA_SPECS[camera]
                            camera_records.append(
                                {
                                    **record,
                                    "camera_body_name": spec["body_name"],
                                    "camera_mount_label": spec["mount_label"],
                                    "camera_pose_relative_to_body": {
                                        "pos": spec["pos"],
                                        "xyaxes": spec["xyaxes"],
                                        "fovy": spec["fovy"],
                                    },
                                    "simulated_robot_pov": True,
                                    "real_robot_pov": False,
                                    "physical_sensor_data": False,
                                }
                            )
                if step_termination_reason is not None:
                    episode_termination_reason = step_termination_reason
                    episode_termination_step = step
                    episode_termination_time_s = float(data.time)
                    break
    finally:
        renderer.close()

    if episode_termination_step is None and rows:
        last_row = rows[-1]
        episode_termination_reason = "timeout"
        episode_termination_step = int(last_row.get("step") or 0)
        episode_termination_time_s = float(last_row.get("sim_time_s") or 0.0)
    video_duration_s = (
        float(episode_termination_time_s)
        if episode_termination_time_s is not None
        else requested_duration
    )

    timeseries_path = handoff_dir / "robot_team_timeseries.jsonl"
    enriched_trace_path = handoff_dir / "policy_execution_trace_enriched.jsonl"
    _write_jsonl(timeseries_path, rows)
    _write_jsonl(enriched_trace_path, enriched_rows)

    video_groups = {
        "official_policy_overview": camera_frame_paths.get("overview", []),
        "official_policy_side": camera_frame_paths.get("side", []),
        "official_policy_follow": camera_frame_paths.get("follow", []),
        "robot_pov_head": camera_frame_paths.get("robot_pov_head", []),
        "robot_pov_torso": camera_frame_paths.get("robot_pov_torso", []),
    }
    videos: dict[str, dict[str, Any]] = {}
    for video_name, frame_paths in video_groups.items():
        out_dir = handoff_dir if video_name.startswith("robot_pov_") else rendered_dir
        video = _write_frame_video(
            camera=video_name,
            frame_paths=frame_paths,
            output_dir=out_dir,
            render_fps=render_fps,
            video_crf=video_crf,
            frame_times_s=camera_frame_times.get(
                video_name.replace("official_policy_", ""), []
            )
            if video_name.startswith("official_policy_")
            else camera_frame_times.get(video_name, []),
            video_duration_s=video_duration_s,
        )
        video["ffprobe"] = _ffprobe_video(Path(_string(video.get("path"))))
        video["blank_scene_checks"] = _blank_scene_checks(frame_paths)
        videos[video_name] = video

    robot_pov_frame_paths = {
        "head": camera_frame_paths.get("robot_pov_head", []),
        "torso": camera_frame_paths.get("robot_pov_torso", []),
    }
    robot_pov_all_frames = [path for paths in robot_pov_frame_paths.values() for path in paths]
    calibration = {
        "schema_version": "official_unitree_g1_robot_pov_camera_calibration.v1",
        "simulated_robot_pov": True,
        "real_robot_pov": False,
        "physical_sensor_data": False,
        "camera_model": "mujoco_fixed_camera",
        "render_resolution": [int(render_width), int(render_height)],
        "render_fps": int(render_fps),
        "cameras": {
            name: {
                "camera_body_name": spec["body_name"],
                "mount_label": spec["mount_label"],
                "pos": spec["pos"],
                "xyaxes": spec["xyaxes"],
                "fovy": spec["fovy"],
            }
            for name, spec in CAMERA_SPECS.items()
        },
        "intrinsics_boundary": (
            "MuJoCo fovy and image dimensions define simulated camera calibration; this is "
            "not a calibrated physical Unitree camera."
        ),
    }
    calibration_path = handoff_dir / "robot_pov_camera_calibration.json"
    _safe_write_json(calibration_path, calibration)
    robot_pov_manifest = _robot_pov_manifest(
        camera_records=camera_records,
        robot_pov_frames=robot_pov_frame_paths,
        robot_pov_videos={
            "head": videos["robot_pov_head"],
            "torso": videos["robot_pov_torso"],
        },
        render_width=render_width,
        render_height=render_height,
        render_fps=render_fps,
        nonblank_checks=_blank_scene_checks(robot_pov_all_frames),
        calibration_path=calibration_path,
    )
    robot_pov_manifest_path = handoff_dir / "robot_pov_manifest.json"
    _safe_write_json(robot_pov_manifest_path, robot_pov_manifest)

    first_base = np.asarray(base_positions[0], dtype=float)
    last_base = np.asarray(base_positions[-1], dtype=float)
    base_displacement = float(np.linalg.norm(last_base[:2] - first_base[:2]))
    motion_range = {
        "base_position_min_xyz": _as_float_list(np.min(np.asarray(base_positions), axis=0)),
        "base_position_max_xyz": _as_float_list(np.max(np.asarray(base_positions), axis=0)),
        "base_displacement_xy_m": base_displacement,
    }
    base_path_clearance = _base_path_clearance_audit(
        base_positions=base_positions,
        collision_proxies=external_collision_proxies,
        required_clearance_m=base_path_clearance_m,
    )
    navigation_runtime_audit = {
        "status": "complete" if navigation_active else "not_active",
        "navigation_active": navigation_active,
        "goal_reached": navigation_goal_reached,
        "final_waypoint_index": navigation_waypoint_index,
        "waypoint_count": len(navigation_waypoints),
        "runtime_clearance_violation_count": navigation_clearance_violation_count,
        "minimum_runtime_clearance": navigation_min_runtime_clearance,
        "command_sample_count": len(navigation_command_samples),
        "clearance_sample_count": len(navigation_clearance_samples),
        "command_samples": navigation_command_samples[:250],
        "clearance_samples": navigation_clearance_samples[:250],
        "termination_reason": episode_termination_reason,
        "termination_step": episode_termination_step,
        "termination_time_s": episode_termination_time_s,
        "control_policy_feed": (
            "waypoint_velocity_commands_in_policy_observation"
            if navigation_active
            else "static_policy_command"
        ),
    }
    navigation_manifest = {
        "schema_version": "official_unitree_g1_navigation_manifest.v1",
        "generated_at": utc_now_iso(),
        "status": (
            "complete"
            if navigation_active
            and navigation_plan.get("status") == "planned"
            and navigation_clearance_violation_count == 0
            else "blocked"
            if enable_navigation_planner
            else "not_requested"
        ),
        "planner_enabled": enable_navigation_planner,
        "planner": navigation_plan,
        "runtime_audit": navigation_runtime_audit,
        "settings": {
            "grid_resolution_m": float(navigation_grid_resolution_m),
            "required_clearance_m": float(base_path_clearance_m),
            "max_speed_mps": float(navigation_max_speed_mps),
            "waypoint_tolerance_m": float(navigation_waypoint_tolerance_m),
            "yaw_gain": float(navigation_yaw_gain),
            "max_yaw_rate": float(navigation_max_yaw_rate),
        },
        "proof_boundary": {
            "simulated_mujoco_navigation": navigation_active,
            "official_policy_command_stream_integrated": navigation_active,
            "continuous_contact_checks": True,
            "continuous_clearance_checks": True,
            "physical_robot_readiness_proven": False,
            "real_world_safety_contact_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    navigation_manifest_path = handoff_dir / "navigation_plan_manifest.json"
    _safe_write_json(navigation_manifest_path, navigation_manifest)
    stream_manifest = _build_sensor_stream_manifest(
        path=timeseries_path,
        rows=rows,
        control_update_count=control_update_count,
        row_count=len(rows),
    )
    stream_manifest_path = handoff_dir / "sensor_stream_manifest.json"
    _safe_write_json(stream_manifest_path, stream_manifest)

    contact_rows_with_pairs = [row for row in rows if row.get("contact_pairs")]
    scene_contact_rows = [
        row
        for row in contact_rows_with_pairs
        if any(
            bool(contact.get("scene_collision_contact"))
            for contact in row.get("contact_pairs", [])
            if isinstance(contact, Mapping)
        )
    ]
    scene_contact_pairs = [
        contact
        for row in scene_contact_rows
        for contact in row.get("contact_pairs", [])
        if isinstance(contact, Mapping) and contact.get("scene_collision_contact") is True
    ]
    contact_manifest = {
        "schema_version": "official_unitree_g1_contact_manifest.v1",
        "status": "complete",
        "row_count": len(rows),
        "rows_with_contacts": len(contact_rows_with_pairs),
        "rows_with_external_scene_contacts": len(scene_contact_rows),
        "external_scene_contact_pair_count": len(scene_contact_pairs),
        "external_scene_collision_geom": "blueprint_external_scene_collision",
        "external_scene_contacts_sample": [dict(contact) for contact in scene_contact_pairs[:10]],
        "contact_stream_available": True,
        "contact_force_estimates_available": True,
        "foot_body_names": list(FOOT_BODY_NAMES),
        "slip_indicator_derivation": "foot body contact plus horizontal body speed greater than 0.20 m/s",
        "sample_contact_pairs": [
            contact
            for row in contact_rows_with_pairs[:5]
            for contact in row.get("contact_pairs", [])[:2]
        ],
    }
    contact_manifest_path = handoff_dir / "contact_manifest.json"
    _safe_write_json(contact_manifest_path, contact_manifest)

    camera_manifest = {
        "schema_version": "official_unitree_g1_camera_manifest.v1",
        "status": "complete" if frame_records else "not_recorded",
        "render_resolution": [int(render_width), int(render_height)],
        "render_fps": int(render_fps),
        "video_crf": int(video_crf),
        "camera_set": cameras,
        "fixed_scene_cameras": [camera for camera in cameras if camera in {"overview"}],
        "virtual_review_cameras": [camera for camera in cameras if camera in {"side", "follow"}],
        "simulated_robot_pov_cameras": [
            camera for camera in cameras if camera in {"robot_pov_head", "robot_pov_torso"}
        ],
        "body_mounted_camera_body_name": "pelvis",
        "real_robot_pov": False,
        "physical_sensor_data": False,
        "frames": frame_records,
        "videos": videos,
        "robot_pov_manifest": str(robot_pov_manifest_path),
        "robot_pov_camera_calibration": str(calibration_path),
    }
    camera_manifest_path = handoff_dir / "camera_manifest.json"
    _safe_write_json(camera_manifest_path, camera_manifest)

    scene_assets_manifest_path = root / "pipeline" / "worldlabs_assets" / "materialized_assets_manifest.json"
    scene_assets = optional_read_json(scene_assets_manifest_path) or {}
    provenance_manifest = {
        "schema_version": "official_unitree_g1_provenance_manifest.v1",
        "generated_at": utc_now_iso(),
        "capture_root": str(root),
        "source_policy_manifest": str(manifest_path),
        "source_policy_manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "source_trace_path": str(original_trace_path),
        "source_trace_sha256": _sha256(original_trace_path) if original_trace_path.is_file() else None,
        "source_trace_row_count": len(original_trace_rows),
        "enriched_trace_path": str(enriched_trace_path),
        "enriched_trace_sha256": _sha256(enriched_trace_path),
        "timeseries_path": str(timeseries_path),
        "timeseries_sha256": _sha256(timeseries_path),
        "policy_id": policy_manifest.get("policy_id"),
        "source_repository": {
            **_mapping(policy_manifest.get("source_repository")),
            "resolved_local_root": str(policy_root),
            "resolved_git_commit": _git_commit(policy_root),
        },
        "official_artifact_hashes": {
            "config_sha256": _sha256(paths["config"]),
            "policy_sha256": _sha256(paths["policy"]),
            "xml_sha256": _sha256(paths["xml"]),
            "generated_robot_xml_sha256": _sha256(generated_robot_xml),
            "generated_camera_scene_xml_sha256": _sha256(generated_scene_xml),
        },
        "scene_asset_manifest": str(scene_assets_manifest_path),
        "scene_asset_hashes": [
            {
                "kind": item.get("kind"),
                "quality": item.get("quality"),
                "local_path": item.get("local_path"),
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
            for item in scene_assets.get("downloads", [])
            if isinstance(item, Mapping)
        ],
        "scenario_context": scenario_context,
        "policy_source_snapshot": snapshot_manifest,
        "proof_boundary": {
            "simulator_official_policy_trace": True,
            "physical_robot_readiness_proven": False,
            "real_robot_pov": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    provenance_manifest_path = handoff_dir / "provenance_manifest.json"
    _safe_write_json(provenance_manifest_path, provenance_manifest)

    overlay_manifest = _worldlabs_asset_overlay_manifest(
        root,
        root / "pipeline" / "worldlabs_assets" / "website_viewer_visual_overlay_manifest.json",
    )

    all_render_frames = [record["path"] for record in frame_records]
    render_manifest = {
        "schema_version": OFFICIAL_G1_RENDER_MANIFEST_SCHEMA_VERSION,
        "status": "complete" if frame_records else "not_recorded",
        "source_policy_manifest": str(manifest_path),
        "source_trace_path": str(original_trace_path),
        "source_enriched_trace_path": str(enriched_trace_path),
        "source_generated_mjcf": str(generated_scene_xml),
        "rendered_from_official_policy_trace": False,
        "fresh_policy_rollout": True,
        "render_boundary": (
            "Frames are rendered during a fresh local MuJoCo rollout using the official "
            "Unitree RL Gym G1 pretrain policy and generated body-mounted cameras. This "
            "is simulated policy motion, not physical robot footage."
        ),
        "policy_id": policy_manifest.get("policy_id"),
        "command_xyz": _as_float_list(cmd),
        "command_source": (
            "navigation_planner_waypoint_velocity"
            if navigation_active
            else "static_policy_command"
        ),
        "navigation_manifest": str(navigation_manifest_path),
        "navigation_plan_status": navigation_plan.get("status"),
        "navigation_goal_reached": navigation_goal_reached,
        "navigation_route_distance_m": navigation_plan.get("route_distance_m"),
        "source_trace_rows": len(original_trace_rows),
        "timeseries_rows": len(rows),
        "rendered_sample_count_per_camera": {
            camera: len(paths) for camera, paths in camera_frame_paths.items()
        },
        "planned_render_sample_count_per_camera": len(sample_indices),
        "render_resolution": [int(render_width), int(render_height)],
        "render_fps": int(render_fps),
        "video_crf": int(video_crf),
        "codec_settings": _video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
        "videos": videos,
        "frames": frame_records,
        "blank_scene_checks": _blank_scene_checks(all_render_frames),
        "all_frames_nonblank": _blank_scene_checks(all_render_frames).get("all_frames_nonblank")
        is True,
        "episode_termination_reason": episode_termination_reason,
        "episode_termination_step": episode_termination_step,
        "episode_termination_time_s": episode_termination_time_s,
        "motion_range": motion_range,
    }
    render_manifest_path = rendered_dir / "official_policy_rendered_motion_manifest.json"
    _safe_write_json(render_manifest_path, render_manifest)

    handoff_gate = stream_manifest["handoff_gate"]
    required_video_names = (
        "official_policy_overview",
        "official_policy_side",
        "official_policy_follow",
        "robot_pov_head",
        "robot_pov_torso",
    )
    videos_complete = all(
        videos[name].get("status") == "complete"
        for name in required_video_names
    )
    robot_pov_complete = robot_pov_manifest.get("status") == "complete" and robot_pov_manifest.get(
        "all_frames_nonblank"
    ) is True
    high_quality_complete = videos_complete and all(
        int(video.get("ffprobe", {}).get("width") or render_width) >= 1280
        and int(video.get("ffprobe", {}).get("height") or render_height) >= 720
        for video in videos.values()
        if video.get("status") == "complete"
    )
    blockers = []
    if not handoff_gate["passed"]:
        blockers.extend(handoff_gate["blockers"])
    if not robot_pov_complete:
        blockers.append("simulated_robot_pov_incomplete")
    if not high_quality_complete:
        blockers.append("high_quality_video_incomplete")
    if not finite_state:
        blockers.append("non_finite_state")
    if not finite_actions:
        blockers.append("non_finite_policy_actions")
    if base_displacement <= 0.10:
        blockers.append("base_displacement_below_walking_threshold")
    if episode_termination_reason == "scene_collision":
        blockers.append("episode_terminated_by_scene_collision")
    if episode_termination_reason == "fall_height_below_threshold":
        blockers.append("episode_terminated_by_fall")
    if episode_termination_reason == "clearance_below_threshold":
        blockers.append("episode_terminated_by_clearance_violation")
    if enable_navigation_planner and navigation_plan.get("status") != "planned":
        blockers.append("navigation_planner_route_not_available")
        blockers.extend(
            f"navigation_planner_{blocker}"
            for blocker in navigation_plan.get("blockers", [])
            if isinstance(blocker, str)
        )
    if navigation_active and navigation_clearance_violation_count > 0:
        blockers.append("navigation_runtime_clearance_violation")
    if navigation_active and not navigation_goal_reached and episode_termination_reason == "timeout":
        blockers.append("navigation_goal_not_reached_before_timeout")
    if not base_path_clearance.get("passed"):
        blockers.append("base_path_or_endpoint_occupancy_not_clear")
    external_scene_collision_loaded = bool(external_scene_obj is not None)
    if not external_scene_collision_loaded:
        blockers.append("external_scene_collision_mesh_not_loaded")
    prior_policy_execution_manifest_proven = bool(
        policy_manifest.get("status") == "completed"
        and metrics.get("finite_state") is True
        and metrics.get("finite_actions") is True
    )
    official_policy_execution_proven = bool(
        finite_state and finite_actions and control_update_count > 0
    )
    handoff_manifest = {
        "schema_version": OFFICIAL_G1_HANDOFF_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "complete" if not blockers else "blocked",
        "capture_root": str(root),
        "policy_id": policy_manifest.get("policy_id"),
        "robot_profile_id": policy_manifest.get("robot_profile_id"),
        "robot_make_model": policy_manifest.get("robot_make_model"),
        "robot_team_handoff_dataset_status": "complete" if handoff_gate["passed"] else "blocked",
        "simulated_robot_pov_status": "complete" if robot_pov_complete else "blocked",
        "high_quality_video_status": "complete" if high_quality_complete else "blocked",
        "official_policy_execution_proven": official_policy_execution_proven,
        "prior_policy_execution_manifest_proven": prior_policy_execution_manifest_proven,
        "fresh_policy_rollout_proven": True,
        "walking_motion_proven": bool(base_displacement > 0.10 and finite_state and finite_actions),
        "training_grade_policy_rollout_proven": bool(
            handoff_gate["passed"] and robot_pov_complete and high_quality_complete
        ),
        "robot_team_handoff_gate_passed": not blockers,
        "locomotion_controller_integrated": True,
        "planner_navigation_layer_integrated": navigation_active,
        "navigation_planner_status": navigation_plan.get("status"),
        "navigation_goal_reached": navigation_goal_reached,
        "navigation_runtime_clearance_violation_count": navigation_clearance_violation_count,
        "navigation_route_distance_m": navigation_plan.get("route_distance_m"),
        "navigation_waypoint_count": len(navigation_waypoints),
        "physical_robot_readiness_proven": False,
        "real_robot_pov": False,
        "real_robot_pov_evidence_proven": False,
        "real_world_safety_contact_validation_proven": False,
        "customer_delivery_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "duration_seconds": requested_duration,
        "actual_duration_seconds": episode_termination_time_s,
        "episode_termination_reason": episode_termination_reason,
        "episode_termination_step": episode_termination_step,
        "episode_termination_time_s": episode_termination_time_s,
        "episode_termination_contract": {
            "target_displacement_m": float(target_displacement_m)
            if target_displacement_m is not None
            else None,
            "scene_collision_terminates_episode": True,
            "fall_height_threshold_m": float(fall_height_threshold_m),
            "timeout_seconds": requested_duration,
        },
        "simulation_dt": simulation_dt,
        "steps": len(rows),
        "planned_steps": total_steps,
        "control_updates": control_update_count,
        "command_xyz": _as_float_list(cmd),
        "command_source": (
            "navigation_planner_waypoint_velocity"
            if navigation_active
            else "static_policy_command"
        ),
        "initial_root_xy": [float(value) for value in initial_root_xy[:2]]
        if initial_root_xy is not None and len(initial_root_xy) >= 2
        else None,
        "initial_root_yaw": float(initial_root_yaw),
        "finite_state": finite_state,
        "finite_actions": finite_actions,
        "motion_range": motion_range,
        "external_scene_collision_loaded": external_scene_collision_loaded,
        "external_scene_visual_collision_same_mesh": bool(
            external_scene_collision_loaded
            and _mapping(
                external_scene_mesh_info.get("collision_geometry_contract")
            ).get("same_mesh_for_visual_and_collision")
            is True
        ),
        "external_scene_proxy_collision_model_used": bool(
            _mapping(
                external_scene_mesh_info.get("collision_geometry_contract")
            ).get("proxy_collision_model_used")
        ),
        "external_scene_collision_proxy_geom_count": int(
            _mapping(
                external_scene_mesh_info.get("collision_geometry_contract")
            ).get("collision_proxy_geom_count")
            or 0
        ),
        "collision_proxy_limit": int(collision_proxy_limit),
        "base_path_clearance_m": float(base_path_clearance_m),
        "base_path_clearance": base_path_clearance,
        "navigation_plan": navigation_plan,
        "navigation_runtime_audit": {
            key: value
            for key, value in navigation_runtime_audit.items()
            if key not in {"command_samples", "clearance_samples"}
        },
        "external_scene_mesh": external_scene_mesh_info,
        "external_scene_contact_rows": len(scene_contact_rows),
        "external_scene_contact_pair_count": len(scene_contact_pairs),
        "render_settings": {
            "render_width": int(render_width),
            "render_height": int(render_height),
            "render_fps": int(render_fps),
            "video_crf": int(video_crf),
            "max_frames": int(max_frames),
            "camera_set": cameras,
        },
        "artifacts": {
            "robot_team_timeseries": str(timeseries_path),
            "policy_execution_trace_enriched": str(enriched_trace_path),
            "sensor_stream_manifest": str(stream_manifest_path),
            "navigation_plan_manifest": str(navigation_manifest_path),
            "camera_manifest": str(camera_manifest_path),
            "contact_manifest": str(contact_manifest_path),
            "provenance_manifest": str(provenance_manifest_path),
            "robot_pov_manifest": str(robot_pov_manifest_path),
            "robot_pov_camera_calibration": str(calibration_path),
            "rendered_motion_manifest": str(render_manifest_path),
            "worldlabs_visual_overlay_manifest": str(
                root
                / "pipeline"
                / "worldlabs_assets"
                / "website_viewer_visual_overlay_manifest.json"
            ),
            "generated_camera_scene_mjcf": str(generated_scene_xml),
        },
        "videos": videos,
        "worldlabs_visual_overlay": overlay_manifest,
        "blockers": blockers,
        "proof_boundary": {
            "official_policy_simulated_motion": True,
            "simulated_mujoco_navigation": navigation_active,
            "continuous_contact_checks": True,
            "continuous_clearance_checks": True,
            "physical_robot_readiness_proven": False,
            "real_robot_pov": False,
            "physical_sensor_data": False,
            "real_world_safety_contact_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    handoff_manifest_path = handoff_dir / "robot_team_handoff_manifest.json"
    _safe_write_json(handoff_manifest_path, handoff_manifest)
    return {**handoff_manifest, "manifest_path": str(handoff_manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root")
    parser.add_argument("--policy-manifest")
    parser.add_argument("--unitree-rl-gym-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-fps", type=int, default=24)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--camera-set",
        default="overview,side,follow,robot_pov",
        help="Comma-separated cameras: overview,side,follow,robot_pov,robot_pov_head,robot_pov_torso.",
    )
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--target-displacement-m", type=float)
    parser.add_argument("--fall-height-threshold-m", type=float, default=0.45)
    parser.add_argument("--command-x", type=float)
    parser.add_argument("--command-y", type=float)
    parser.add_argument("--command-yaw", type=float)
    parser.add_argument("--collision-proxy-limit", type=int, default=512)
    parser.add_argument("--base-path-clearance-m", type=float, default=0.38)
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-z", type=float, default=0.793)
    parser.add_argument("--navigation-grid-resolution-m", type=float, default=0.35)
    parser.add_argument("--navigation-max-speed-mps", type=float, default=0.55)
    parser.add_argument("--navigation-waypoint-tolerance-m", type=float, default=0.35)
    parser.add_argument("--navigation-yaw-gain", type=float, default=1.2)
    parser.add_argument("--navigation-max-yaw-rate", type=float, default=0.9)
    parser.add_argument("--disable-navigation-planner", action="store_true")
    parser.add_argument("--no-policy-source-snapshot", action="store_true")
    args = parser.parse_args(argv)
    capture_root = args.capture_root or os.environ.get("BLUEPRINT_CAPTURE_ROOT")
    if not capture_root:
        parser.error("--capture-root or BLUEPRINT_CAPTURE_ROOT is required")
    command_xyz = None
    if (
        args.command_x is not None
        or args.command_y is not None
        or args.command_yaw is not None
    ):
        command_xyz = [
            0.5 if args.command_x is None else args.command_x,
            0.0 if args.command_y is None else args.command_y,
            0.0 if args.command_yaw is None else args.command_yaw,
        ]
    navigation_goal_xyz = (
        [args.goal_x, args.goal_y, args.goal_z]
        if args.goal_x is not None and args.goal_y is not None
        else None
    )
    result = build_official_g1_policy_handoff(
        capture_root=capture_root,
        policy_manifest_path=args.policy_manifest,
        unitree_rl_gym_root=args.unitree_rl_gym_root,
        output_dir=args.output_dir,
        render_width=args.render_width,
        render_height=args.render_height,
        render_fps=args.render_fps,
        video_crf=args.video_crf,
        max_frames=args.max_frames,
        camera_set=args.camera_set,
        duration_seconds=args.duration_seconds,
        target_displacement_m=args.target_displacement_m,
        fall_height_threshold_m=args.fall_height_threshold_m,
        command_xyz=command_xyz,
        collision_proxy_limit=args.collision_proxy_limit,
        base_path_clearance_m=args.base_path_clearance_m,
        initial_root_xy=[args.start_x, args.start_y]
        if args.start_x is not None and args.start_y is not None
        else None,
        initial_root_yaw=args.start_yaw,
        navigation_goal_xyz=navigation_goal_xyz,
        navigation_grid_resolution_m=args.navigation_grid_resolution_m,
        navigation_max_speed_mps=args.navigation_max_speed_mps,
        navigation_waypoint_tolerance_m=args.navigation_waypoint_tolerance_m,
        navigation_yaw_gain=args.navigation_yaw_gain,
        navigation_max_yaw_rate=args.navigation_max_yaw_rate,
        enable_navigation_planner=not args.disable_navigation_planner,
        copy_policy_source_snapshot=not args.no_policy_source_snapshot,
    )
    simulator_output_path = _string(os.environ.get("BLUEPRINT_SIMULATOR_OUTPUT"))
    if simulator_output_path:
        _safe_write_json(
            Path(simulator_output_path).expanduser().resolve(),
            {
                "schema_version": "official_unitree_g1_policy_handoff_simulator_output.v1",
                "generated_at": utc_now_iso(),
                "status": "completed" if result.get("status") == "complete" else "blocked",
                "simulator_backend": "mujoco",
                "simulator_execution_proven": result.get("status") == "complete",
                "official_policy_execution_proven": result.get(
                    "official_policy_execution_proven"
                )
                is True,
                "fresh_policy_rollout_proven": result.get("fresh_policy_rollout_proven") is True,
                "unitree_g1_asset_spawned": True,
                "mujoco_g1_asset_execution_proven": True,
                "robot_policy_execution_proven": result.get("walking_motion_proven") is True,
                "robot_team_handoff_dataset_status": result.get(
                    "robot_team_handoff_dataset_status"
                ),
                "simulated_robot_pov_status": result.get("simulated_robot_pov_status"),
                "high_quality_video_status": result.get("high_quality_video_status"),
                "walking_motion_proven": result.get("walking_motion_proven") is True,
                "planner_navigation_layer_integrated": result.get(
                    "planner_navigation_layer_integrated"
                )
                is True,
                "navigation_planner_status": result.get("navigation_planner_status"),
                "navigation_goal_reached": result.get("navigation_goal_reached") is True,
                "navigation_runtime_clearance_violation_count": result.get(
                    "navigation_runtime_clearance_violation_count"
                ),
                "navigation_route_distance_m": result.get("navigation_route_distance_m"),
                "training_grade_policy_rollout_proven": result.get(
                    "training_grade_policy_rollout_proven"
                )
                is True,
                "physical_robot_readiness_proven": False,
                "real_robot_pov": False,
                "physical_sensor_data": False,
                "public_claim_upgrade_allowed": False,
                "artifact_paths": result.get("artifacts"),
                "manifest_path": result.get("manifest_path"),
                "blockers": result.get("blockers") or [],
                "proof_boundary": result.get("proof_boundary"),
            },
        )
    print(result["manifest_path"])
    print(result["status"])
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
