"""Optional CPU-only MuJoCo/PyBullet preflight setup and smoke runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import traceback
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .episode_spec import build_episode_specs
from .local_capture import resolve_local_capture_context


EPISODE_SETUP_SCHEMA_VERSION = "episode_setup_manifest.v1"
CPU_SIMULATOR_PREFLIGHT_SCHEMA_VERSION = "cpu_simulator_preflight_manifest.v1"
CPU_SIMULATOR_RESULT_SCHEMA_VERSION = "cpu_simulator_preflight_result.v1"
SPAWN_POSE_VALIDATION_SCHEMA_VERSION = "spawn_pose_validation_manifest.v1"
CPU_PREFLIGHT_MANIFEST_SCHEMA_VERSION = "cpu_preflight_manifest.v1"
PRE_GPU_READINESS_SUMMARY_SCHEMA_VERSION = "pre_gpu_readiness_summary.v1"

CPU_BACKENDS = ("mujoco", "pybullet")

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "optional_local_cpu_simulator_preflight_only",
    "repo_local_only": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "gpu_required": False,
    "gpu_simulator_execution_proven": False,
    "owner_system_simulator_execution_proven": False,
    "simulator_execution_proven": False,
    "robot_readiness_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "safety_validated": False,
    "public_claim_upgrade_allowed": False,
    "allowed_phrase": "local CPU preflight smoke",
    "disallowed_claims": [
        "simulator_execution_completed",
        "robot_ready",
        "deployment_ready",
        "policy_success",
        "physics_contact_validated",
        "safety_validated",
    ],
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _indent_xml(root: ET.Element) -> None:
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover
        pass


def _float_list(value: Any, *, fallback: Sequence[float]) -> List[float]:
    out: List[float] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value[:3]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(float(fallback[len(out)]))
    while len(out) < 3:
        out.append(float(fallback[len(out)]))
    return out[:3]


def _finite_xyz(value: Any) -> List[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    out: List[float] = []
    for item in list(value)[:3]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _frame_bounds(automation_dir: Path) -> Dict[str, Any]:
    frame_manifest = _read_optional_mapping(automation_dir / "scene_frame_estimate.json")
    frame = _mapping(frame_manifest.get("frame"))
    bounds = _mapping(frame.get("bounds"))
    low = _finite_xyz(bounds.get("min"))
    high = _finite_xyz(bounds.get("max"))
    floor = frame.get("floor_z_estimate")
    try:
        floor_z = float(floor)
    except (TypeError, ValueError):
        floor_z = low[2] if low else 0.0
    return {
        "manifest": frame_manifest,
        "bounds": {"min": low, "max": high} if low and high else None,
        "floor_z": floor_z,
        "source": frame.get("source_asset"),
        "confidence": frame.get("confidence") or "low",
    }


def _candidate_spawn_poses(episode: Mapping[str, Any], frame: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    primary = _mapping(episode.get("robot_spawn_pose"))
    if primary:
        candidates.append(
            {
                "candidate_id": "episode_robot_spawn_pose",
                "source": primary.get("source") or "episode_spec",
                "xyz": _finite_xyz(primary.get("xyz")) or _float_list(primary.get("xyz"), fallback=(0, 0, 0)),
                "rpy": _float_list(primary.get("rpy"), fallback=(0, 0, 0)),
            }
        )
    bounds = _mapping(frame.get("bounds"))
    low = _finite_xyz(bounds.get("min"))
    high = _finite_xyz(bounds.get("max"))
    floor_z = float(frame.get("floor_z") or 0.0)
    if low and high:
        center = [(low[index] + high[index]) * 0.5 for index in range(3)]
        span = [max(0.0, high[index] - low[index]) for index in range(3)]
        offsets = [
            ("frame_center_floor", [center[0], center[1], floor_z + 0.05]),
            ("near_min_corner_floor", [low[0] + span[0] * 0.2, low[1] + span[1] * 0.2, floor_z + 0.05]),
            ("near_max_corner_floor", [low[0] + span[0] * 0.8, low[1] + span[1] * 0.8, floor_z + 0.05]),
        ]
        for candidate_id, xyz in offsets:
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source": "scene_frame_estimate",
                    "xyz": xyz,
                    "rpy": [0.0, 0.0, 0.0],
                }
            )
    seen: set[tuple[float, float, float]] = set()
    out: List[Dict[str, Any]] = []
    for candidate in candidates:
        xyz = _finite_xyz(candidate.get("xyz"))
        if xyz is None:
            out.append(candidate)
            continue
        key = tuple(round(value, 4) for value in xyz)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _inside_aabb(xyz: Sequence[float], low: Sequence[float], high: Sequence[float], *, margin: float = 0.0) -> bool:
    return all(
        float(low[index]) - margin <= float(xyz[index]) <= float(high[index]) + margin
        for index in range(3)
    )


def _spawn_intersects_proxy_obstacle(xyz: Sequence[float], obstacle: Mapping[str, Any]) -> bool:
    low = _finite_xyz(obstacle.get("min_xyz"))
    high = _finite_xyz(obstacle.get("max_xyz"))
    return bool(low and high and _inside_aabb(xyz, low, high, margin=0.05))


def _validate_spawn_candidate(
    candidate: Mapping[str, Any],
    *,
    frame: Mapping[str, Any],
    proxy_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []
    xyz = _finite_xyz(candidate.get("xyz"))
    if xyz is None:
        blockers.append("spawn_pose_not_finite")
        xyz = [0.0, 0.0, 0.0]
    bounds = _mapping(frame.get("bounds"))
    low = _finite_xyz(bounds.get("min"))
    high = _finite_xyz(bounds.get("max"))
    if not low or not high:
        blockers.append("scene_bounds_missing_or_invalid")
        dimensions = None
    else:
        dimensions = [high[index] - low[index] for index in range(3)]
        if any(value <= 0 for value in dimensions):
            blockers.append("scene_bounds_empty_or_inverted")
        if any(value < 0.05 for value in dimensions):
            warnings.append("scene_scale_suspiciously_small")
        if any(value > 1_000 for value in dimensions):
            warnings.append("scene_scale_suspiciously_large")
        if not _inside_aabb(xyz, low, high, margin=0.05):
            blockers.append("spawn_outside_scene_bounds")
    floor_z = float(frame.get("floor_z") or 0.0)
    if xyz[2] < floor_z - 0.05:
        blockers.append("spawn_below_floor_estimate")
    if xyz[2] > floor_z + 2.5:
        warnings.append("spawn_height_far_above_floor_estimate")
    proxy_obstacles = [
        item for item in proxy_manifest.get("proxy_obstacles") or [] if isinstance(item, Mapping)
    ]
    intersecting = [
        _string(item.get("obstacle_id"))
        for item in proxy_obstacles
        if _spawn_intersects_proxy_obstacle(xyz, item)
    ]
    if intersecting:
        blockers.append("spawn_inside_known_or_proxy_geometry")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source"),
        "xyz": xyz,
        "rpy": _float_list(candidate.get("rpy"), fallback=(0, 0, 0)),
        "status": "blocked" if blockers else "valid_review_required",
        "blockers": blockers,
        "warnings": warnings,
        "floor_z_estimate": floor_z,
        "scene_dimensions": dimensions,
        "inside_scene_bounds": "spawn_outside_scene_bounds" not in blockers,
        "intersecting_proxy_obstacles": [item for item in intersecting if item],
        "review_required": True,
    }


def _build_spawn_pose_validation(automation_dir: Path, *, generated_at: str, scene_id: str, capture_id: str) -> Dict[str, Any]:
    frame = _frame_bounds(automation_dir)
    proxy_manifest = _read_optional_mapping(automation_dir / "cpu_scene_proxy_manifest.json")
    episodes = _episode_specs(automation_dir)
    validations: List[Dict[str, Any]] = []
    for episode in episodes:
        episode_id = _string(episode.get("episode_id"))
        candidates = _candidate_spawn_poses(episode, frame)
        candidate_results = [
            _validate_spawn_candidate(
                candidate,
                frame=frame,
                proxy_manifest=proxy_manifest,
            )
            for candidate in candidates
        ]
        validations.append(
            {
                "episode_id": episode_id,
                "task_id": episode.get("task_id"),
                "scenario_id": episode.get("scenario_id"),
                "candidate_count": len(candidate_results),
                "valid_candidate_count": sum(
                    1 for item in candidate_results if item.get("status") == "valid_review_required"
                ),
                "candidates": candidate_results,
                "status": "blocked"
                if not candidate_results
                or all(item.get("status") == "blocked" for item in candidate_results)
                else "valid_candidates_review_required",
            }
        )
    blockers = sorted(
        {
            blocker
            for validation in validations
            for candidate in validation.get("candidates", [])
            for blocker in _string_list(candidate.get("blockers"))
        }
    )
    manifest = {
        "schema_version": SPAWN_POSE_VALIDATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": "blocked" if not validations or "scene_bounds_missing_or_invalid" in blockers else "review_required",
        "episode_count": len(validations),
        "validations": validations,
        "blockers": blockers,
        "checks": [
            "finite_spawn_xyz",
            "scene_bounds_nonempty",
            "scale_sanity",
            "floor_height_consistency",
            "spawn_inside_scene_bounds",
            "spawn_outside_known_or_proxy_geometry_where_available",
        ],
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["deterministic_fingerprint"] = _sha_payload(
        {"validations": validations, "blockers": blockers}
    )
    return manifest


def _episode_specs(automation_dir: Path) -> List[Dict[str, Any]]:
    spec = _read_optional_mapping(automation_dir / "episode_spec.v1.json")
    episodes = spec.get("episodes")
    return [dict(item) for item in episodes if isinstance(item, Mapping)] if isinstance(episodes, list) else []


def _primary_episode(automation_dir: Path) -> Dict[str, Any]:
    episodes = _episode_specs(automation_dir)
    return episodes[0] if episodes else {}


def _write_mujoco_fixture(path: Path, episode: Mapping[str, Any]) -> None:
    root = ET.Element("mujoco", {"model": "blueprint_cpu_preflight"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"timestep": "0.01", "gravity": "0 0 -9.81"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "cpu_preflight_light", "pos": "0 0 4"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor_proxy",
            "type": "plane",
            "pos": "0 0 0",
            "size": "8 8 0.05",
            "rgba": "0.72 0.74 0.76 1",
        },
    )
    spawn = _mapping(episode.get("robot_spawn_pose"))
    xyz = _float_list(spawn.get("xyz"), fallback=(0.0, 0.0, 0.25))
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "robot_proxy", "pos": f"{xyz[0]:.6f} {xyz[1]:.6f} {max(0.15, xyz[2]):.6f}"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "robot_proxy_geom",
            "type": "box",
            "size": "0.25 0.20 0.30",
            "rgba": "0.12 0.34 0.82 0.65",
        },
    )
    ET.SubElement(
        body,
        "joint",
        {"name": "robot_proxy_free_joint", "type": "free", "limited": "false"},
    )
    _indent_xml(root)
    ensure_dir(path.parent)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_pybullet_fixture(path: Path, episode: Mapping[str, Any]) -> None:
    robot = ET.Element("robot", {"name": "blueprint_cpu_preflight"})
    ET.SubElement(robot, "link", {"name": "world"})
    spawn = _mapping(episode.get("robot_spawn_pose"))
    xyz = _float_list(spawn.get("xyz"), fallback=(0.0, 0.0, 0.25))
    floor = ET.SubElement(robot, "link", {"name": "floor_proxy"})
    for section in ("visual", "collision"):
        node = ET.SubElement(floor, section)
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "box", {"size": "8 8 0.05"})
    floor_joint = ET.SubElement(robot, "joint", {"name": "world_to_floor_proxy", "type": "fixed"})
    ET.SubElement(floor_joint, "parent", {"link": "world"})
    ET.SubElement(floor_joint, "child", {"link": "floor_proxy"})
    ET.SubElement(floor_joint, "origin", {"xyz": "0 0 -0.025", "rpy": "0 0 0"})

    body = ET.SubElement(robot, "link", {"name": "robot_proxy"})
    inertial = ET.SubElement(body, "inertial")
    ET.SubElement(inertial, "mass", {"value": "5.0"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "0.1", "ixy": "0", "ixz": "0", "iyy": "0.1", "iyz": "0", "izz": "0.1"},
    )
    for section in ("visual", "collision"):
        node = ET.SubElement(body, section)
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "box", {"size": "0.5 0.4 0.6"})
    joint = ET.SubElement(robot, "joint", {"name": "world_to_robot_proxy", "type": "floating"})
    ET.SubElement(joint, "parent", {"link": "world"})
    ET.SubElement(joint, "child", {"link": "robot_proxy"})
    ET.SubElement(
        joint,
        "origin",
        {"xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {max(0.3, xyz[2]):.6f}", "rpy": "0 0 0"},
    )
    _indent_xml(robot)
    ensure_dir(path.parent)
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def _write_setup_artifacts(automation_dir: Path, episode: Mapping[str, Any]) -> Dict[str, str]:
    mujoco_path = automation_dir / "mujoco_cpu_preflight" / "episode_scene.xml"
    pybullet_path = automation_dir / "pybullet_cpu_preflight" / "episode_scene.urdf"
    _write_mujoco_fixture(mujoco_path, episode)
    _write_pybullet_fixture(pybullet_path, episode)
    return {
        "mujoco_mjcf": _relative_to(automation_dir, mujoco_path),
        "pybullet_urdf": _relative_to(automation_dir, pybullet_path),
    }


def _install_instructions(backend: str) -> Dict[str, str]:
    package = "pybullet" if backend == "pybullet" else "mujoco"
    return {
        "install": f"python -m pip install {package}",
        "run": (
            "BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true "
            "blueprint-run-cpu-simulator-preflight --capture-root <capture-root> "
            "--allow-cpu-simulator-preflight"
        ),
    }


def _blocked_backend_result(
    *,
    backend: str,
    reason: str,
    blockers: Sequence[str],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": CPU_SIMULATOR_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "backend": backend,
        "status": "blocked",
        "reason": reason,
        "blockers": list(blockers),
        "optional_dependency": backend,
        "install_and_run": _install_instructions(backend),
        "local_cpu_smoke_ran": False,
        "local_cpu_smoke_completed": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _run_pybullet_smoke(
    *,
    urdf_path: Path,
    steps: int,
    allow_render: bool,
    generated_at: str,
) -> Dict[str, Any]:
    try:
        import pybullet as pybullet  # type: ignore[import-untyped]

        client = pybullet.connect(pybullet.DIRECT)
        try:
            pybullet.resetSimulation(physicsClientId=client)
            pybullet.setGravity(0, 0, -9.81, physicsClientId=client)
            body_id = pybullet.loadURDF(str(urdf_path), useFixedBase=False, physicsClientId=client)
            for _ in range(max(1, steps)):
                pybullet.stepSimulation(physicsClientId=client)
            camera_result = None
            if allow_render:
                camera_result = pybullet.getCameraImage(
                    32,
                    32,
                    renderer=pybullet.ER_TINY_RENDERER,
                    physicsClientId=client,
                )[:2]
            return {
                "schema_version": CPU_SIMULATOR_RESULT_SCHEMA_VERSION,
                "generated_at": generated_at,
                "backend": "pybullet",
                "status": "completed_local_cpu_smoke",
                "reason": None,
                "blockers": [],
                "urdf_path": str(urdf_path),
                "body_id": body_id,
                "steps": max(1, steps),
                "render_attempted": bool(allow_render),
                "render_result_shape": list(camera_result) if camera_result else None,
                "local_cpu_smoke_ran": True,
                "local_cpu_smoke_completed": True,
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        finally:
            pybullet.disconnect(physicsClientId=client)
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {
            "schema_version": CPU_SIMULATOR_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "backend": "pybullet",
            "status": "failed_local_cpu_smoke",
            "reason": exc.__class__.__name__,
            "blockers": [str(exc) or exc.__class__.__name__],
            "traceback": traceback.format_exc(limit=8),
            "local_cpu_smoke_ran": True,
            "local_cpu_smoke_completed": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


def _run_mujoco_smoke(*, mjcf_path: Path, steps: int, generated_at: str) -> Dict[str, Any]:
    try:
        import mujoco  # type: ignore[import-untyped]

        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        for _ in range(max(1, steps)):
            mujoco.mj_step(model, data)
        return {
            "schema_version": CPU_SIMULATOR_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "backend": "mujoco",
            "status": "completed_local_cpu_smoke",
            "reason": None,
            "blockers": [],
            "mjcf_path": str(mjcf_path),
            "steps": max(1, steps),
            "render_attempted": False,
            "local_cpu_smoke_ran": True,
            "local_cpu_smoke_completed": True,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {
            "schema_version": CPU_SIMULATOR_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "backend": "mujoco",
            "status": "failed_local_cpu_smoke",
            "reason": exc.__class__.__name__,
            "blockers": [str(exc) or exc.__class__.__name__],
            "traceback": traceback.format_exc(limit=8),
            "local_cpu_smoke_ran": True,
            "local_cpu_smoke_completed": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }


def _backend_result(
    *,
    backend: str,
    automation_dir: Path,
    allow_cpu_simulator_preflight: bool,
    env_allowed: bool,
    steps: int,
    allow_render: bool,
    generated_at: str,
) -> Dict[str, Any]:
    if backend not in CPU_BACKENDS:
        return _blocked_backend_result(
            backend=backend,
            reason="unsupported_cpu_backend",
            blockers=[f"unsupported_cpu_backend:{backend}"],
            generated_at=generated_at,
        )
    if not (allow_cpu_simulator_preflight and env_allowed):
        return _blocked_backend_result(
            backend=backend,
            reason="approval_required",
            blockers=[
                "Set BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true and pass --allow-cpu-simulator-preflight.",
            ],
            generated_at=generated_at,
        )
    if importlib.util.find_spec(backend) is None:
        return _blocked_backend_result(
            backend=backend,
            reason="missing_optional_dependency",
            blockers=[f"missing_python_package:{backend}"],
            generated_at=generated_at,
        )
    if backend == "pybullet":
        return _run_pybullet_smoke(
            urdf_path=automation_dir / "pybullet_cpu_preflight" / "episode_scene.urdf",
            steps=steps,
            allow_render=allow_render,
            generated_at=generated_at,
        )
    return _run_mujoco_smoke(
        mjcf_path=automation_dir / "mujoco_cpu_preflight" / "episode_scene.xml",
        steps=steps,
        generated_at=generated_at,
    )


def build_cpu_simulator_preflight(
    *,
    capture_root: str | Path,
    allow_cpu_simulator_preflight: bool = False,
    backends: Sequence[str] = CPU_BACKENDS,
    smoke_steps: int = 10,
    allow_render: bool = False,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    automation_dir = context.pipeline_root / "simulation_automation"
    ensure_dir(automation_dir)
    if not (automation_dir / "episode_spec.v1.json").is_file():
        build_episode_specs(capture_root=context.capture_root)
    generated_at = utc_now_iso()
    episode = _primary_episode(automation_dir)
    setup_artifacts = _write_setup_artifacts(automation_dir, episode)
    spawn_validation = _build_spawn_pose_validation(
        automation_dir,
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
    )
    episode_setup = {
        "schema_version": EPISODE_SETUP_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_optional_cpu_smoke" if episode else "blocked_missing_episode_spec",
        "episode_spec_path": "episode_spec.v1.json",
        "episode_specs_path": "episode_specs.json",
        "spawn_pose_validation_manifest_path": "spawn_pose_validation_manifest.json",
        "primary_episode_id": episode.get("episode_id"),
        "generated_fixtures": setup_artifacts,
        "backends": {
            "mujoco": {
                "fixture_path": setup_artifacts["mujoco_mjcf"],
                "mode": "CPU_compile_step_optional",
                "gpu_required": False,
            },
            "pybullet": {
                "fixture_path": setup_artifacts["pybullet_urdf"],
                "mode": "DIRECT_cpu_optional",
                "gpu_required": False,
            },
        },
        "missing_proof_labels": sorted(
            set(_string_list(episode.get("missing_proof_labels")) if episode else ["missing_episode_spec"])
        ),
        "spawn_validation_status": spawn_validation.get("status"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    env_allowed = _env_truthy("BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT")
    selected_backends = [backend for backend in backends if backend in CPU_BACKENDS]
    if not selected_backends:
        selected_backends = list(CPU_BACKENDS)
    backend_results: Dict[str, Dict[str, Any]] = {}
    for backend in selected_backends:
        backend_results[backend] = _backend_result(
            backend=backend,
            automation_dir=automation_dir,
            allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
            env_allowed=env_allowed,
            steps=smoke_steps,
            allow_render=allow_render,
            generated_at=generated_at,
        )
        result_path = automation_dir / f"{backend}_cpu_preflight" / "smoke_result.json"
        write_json(result_path, backend_results[backend])
        if backend_results[backend]["status"] == "blocked":
            write_json(
                automation_dir / f"{backend}_cpu_preflight" / "blocked_manifest.json",
                backend_results[backend],
            )

    completed = [
        backend
        for backend, result in backend_results.items()
        if result.get("status") == "completed_local_cpu_smoke"
    ]
    failed = [
        backend
        for backend, result in backend_results.items()
        if result.get("status") == "failed_local_cpu_smoke"
    ]
    blocked = [
        backend
        for backend, result in backend_results.items()
        if result.get("status") == "blocked"
    ]
    overall_status = (
        "failed_local_cpu_smoke"
        if failed
        else "completed_local_cpu_smoke"
        if completed and not blocked
        else "ready_blocked_optional_dependencies_or_gates"
    )
    manifest = {
        "schema_version": CPU_SIMULATOR_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": overall_status,
        "episode_setup_manifest_path": "episode_setup_manifest.json",
        "episode_spec_path": "episode_spec.v1.json",
        "episode_specs_path": "episode_specs.json",
        "spawn_pose_validation_manifest_path": "spawn_pose_validation_manifest.json",
        "selected_backends": selected_backends,
        "execution_gate": {
            "allow_cpu_simulator_preflight_flag": bool(allow_cpu_simulator_preflight),
            "env_BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT": env_allowed,
            "allow_render": bool(allow_render),
            "smoke_steps": max(1, smoke_steps),
        },
        "backend_results": backend_results,
        "local_cpu_smoke_completed_backends": completed,
        "blocked_backends": blocked,
        "failed_backends": failed,
        "blockers": sorted(
            {
                blocker
                for result in backend_results.values()
                for blocker in _string_list(result.get("blockers"))
            }
        ),
        "install_and_run": {
            backend: _install_instructions(backend)
            for backend in selected_backends
            if backend in blocked
        },
        "local_cpu_smoke_ran": any(result.get("local_cpu_smoke_ran") for result in backend_results.values()),
        "local_cpu_smoke_completed": bool(completed) and not failed,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    scene_preflight = _read_optional_mapping(automation_dir / "scene_asset_preflight.json")
    dependency_audit = _read_optional_mapping(automation_dir / "scene_asset_dependency_audit.json")
    collider_proxy_plan = _read_optional_mapping(automation_dir / "collider_proxy_plan.json")
    spawn_hard_blockers = (
        _string_list(spawn_validation.get("blockers"))
        if spawn_validation.get("status") == "blocked"
        else []
    )
    hard_preflight_blockers = sorted(
        set(_string_list(scene_preflight.get("blockers")) + spawn_hard_blockers)
        - {
            "portable_collider_glb_missing",
            "isaac_usd_collision_unverified",
            "simulator_execution_not_run",
        }
    )
    ready_for_owner_gpu_preflight = bool(scene_preflight) and not hard_preflight_blockers
    cpu_preflight_manifest = {
        "schema_version": CPU_PREFLIGHT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_owner_gpu_preflight_handoff"
        if ready_for_owner_gpu_preflight
        else "blocked_for_owner_gpu_preflight_handoff",
        "ready_for_owner_gpu_preflight": ready_for_owner_gpu_preflight,
        "artifact_paths": {
            "scene_asset_inventory": "scene_asset_inventory.json",
            "scene_asset_dependency_audit": "scene_asset_dependency_audit.json",
            "scene_asset_preflight": "scene_asset_preflight.json",
            "collider_proxy_plan": "collider_proxy_plan.json",
            "cpu_scene_proxy_manifest": "cpu_scene_proxy_manifest.json",
            "task_anchor_proposal_manifest": "task_anchor_proposal_manifest.json",
            "episode_specs": "episode_specs.json",
            "spawn_pose_validation_manifest": "spawn_pose_validation_manifest.json",
            "episode_setup_manifest": "episode_setup_manifest.json",
            "cpu_simulator_preflight_manifest": "cpu_simulator_preflight_manifest.json",
        },
        "checks_performed": [
            "asset_inventory",
            "dependency_audit",
            "collider_or_proxy_planning",
            "task_anchor_proposals",
            "episode_spec_compilation",
            "spawn_pose_validation",
            "optional_cpu_mujoco_pybullet_fixture_generation",
        ],
        "dependency_summary": {
            "missing_local_file_count": dependency_audit.get("missing_local_file_count", 0),
            "hard_missing_local_file_count": dependency_audit.get(
                "hard_missing_local_file_count",
                dependency_audit.get("missing_local_file_count", 0),
            ),
            "owner_system_material_warning_count": dependency_audit.get(
                "owner_system_material_warning_count",
                0,
            ),
            "remote_ref_count": dependency_audit.get("remote_ref_count", 0),
            "unresolved_ref_count": dependency_audit.get("unresolved_ref_count", 0),
        },
        "collider_summary": {
            "real_collider_proven": bool(collider_proxy_plan.get("real_collider_proven")),
            "proxy_estimated": bool(collider_proxy_plan.get("proxy_estimated")),
            "missing_collider": bool(collider_proxy_plan.get("missing_collider")),
            "review_required": True,
            "labels": collider_proxy_plan.get("labels") or [],
        },
        "spawn_validation_status": spawn_validation.get("status"),
        "hard_preflight_blockers": hard_preflight_blockers,
        "review_required": True,
        "owner_gpu_simulator_execution_required": True,
        "owner_gpu_simulator_execution_proven": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "physics_contact_validated": False,
        "safety_validated": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    pre_gpu_summary = {
        "schema_version": PRE_GPU_READINESS_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": cpu_preflight_manifest["status"],
        "ready_for_owner_gpu_preflight": ready_for_owner_gpu_preflight,
        "ready_for_robot_evaluation": False,
        "cpu_checked": cpu_preflight_manifest["checks_performed"],
        "cpu_artifacts": cpu_preflight_manifest["artifact_paths"],
        "remaining_unproven_step": "actual_owner_system_gpu_simulator_execution",
        "remaining_owner_system_blockers": [
            {
                "blocker_id": "owner_gpu_simulator_execution_not_run",
                "owner": "robot_team_or_owner_system_operator",
                "required_input": "Run accepted simulator backend on owner GPU system and upload proof schema artifacts.",
                "safe_next_command": (
                    "BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true "
                    "blueprint-run-simulation-automation --capture-root <capture-root> "
                    "--allow-simulator-execution --allow-simulator isaac_sim "
                    "--simulator-command isaac_sim='<owner-system command>'"
                ),
            }
        ],
        "disallowed_claims": list(CLAIM_BOUNDARY["disallowed_claims"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    episode_setup["deterministic_fingerprint"] = _sha_payload(
        {"episode_setup": {k: v for k, v in episode_setup.items() if k != "generated_at"}}
    )
    manifest["deterministic_fingerprint"] = _sha_payload(
        {"manifest": {k: v for k, v in manifest.items() if k != "generated_at"}}
    )
    cpu_preflight_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "status": cpu_preflight_manifest["status"],
            "dependency_summary": cpu_preflight_manifest["dependency_summary"],
            "collider_summary": cpu_preflight_manifest["collider_summary"],
            "hard_preflight_blockers": hard_preflight_blockers,
        }
    )
    pre_gpu_summary["deterministic_fingerprint"] = _sha_payload(
        {
            "status": pre_gpu_summary["status"],
            "ready_for_owner_gpu_preflight": ready_for_owner_gpu_preflight,
            "remaining_owner_system_blockers": pre_gpu_summary["remaining_owner_system_blockers"],
        }
    )
    write_json(automation_dir / "spawn_pose_validation_manifest.json", spawn_validation)
    write_json(automation_dir / "episode_setup_manifest.json", episode_setup)
    write_json(automation_dir / "cpu_simulator_preflight_manifest.json", manifest)
    write_json(automation_dir / "cpu_preflight_manifest.json", cpu_preflight_manifest)
    write_json(automation_dir / "pre_gpu_readiness_summary.json", pre_gpu_summary)
    write_text(
        automation_dir / "cpu_simulator_preflight_README.txt",
        (
            "CPU simulator preflight is optional and local-only.\n"
            "Install optional packages with `python -m pip install mujoco pybullet`.\n"
            "Run with `BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true "
            "blueprint-run-cpu-simulator-preflight --capture-root <capture-root> "
            "--allow-cpu-simulator-preflight`.\n"
            "A passing local CPU smoke does not prove robot readiness, safety, contact, "
            "policy success, or owner-system simulator execution.\n"
        ),
    )
    return {
        "schema_version": "cpu_simulator_preflight_result.v1",
        "capture_root": str(context.capture_root),
        "automation_dir": str(automation_dir),
        "status": overall_status,
        "episode_setup_manifest_path": str((automation_dir / "episode_setup_manifest.json").resolve()),
        "spawn_pose_validation_manifest_path": str(
            (automation_dir / "spawn_pose_validation_manifest.json").resolve()
        ),
        "cpu_simulator_preflight_manifest_path": str(
            (automation_dir / "cpu_simulator_preflight_manifest.json").resolve()
        ),
        "cpu_preflight_manifest_path": str((automation_dir / "cpu_preflight_manifest.json").resolve()),
        "pre_gpu_readiness_summary_path": str(
            (automation_dir / "pre_gpu_readiness_summary.json").resolve()
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate CPU MuJoCo/PyBullet setup manifests and optionally run local CPU smoke checks"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument(
        "--allow-cpu-simulator-preflight",
        action="store_true",
        help="Run optional CPU smoke only when BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true is also set",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=CPU_BACKENDS,
        default=[],
        help="CPU backend to preflight; repeatable. Defaults to MuJoCo and PyBullet.",
    )
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument(
        "--allow-render",
        action="store_true",
        help="Allow optional TinyRenderer path for PyBullet. MuJoCo rendering is not attempted.",
    )
    args = parser.parse_args(argv)
    try:
        result = build_cpu_simulator_preflight(
            capture_root=args.capture_root,
            allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
            backends=args.backend or CPU_BACKENDS,
            smoke_steps=args.smoke_steps,
            allow_render=args.allow_render,
        )
    except (OSError, ValueError, PipelineError) as exc:
        print(f"[cpu-simulator-preflight] FAILED: {exc}")
        return 1
    print(f"[cpu-simulator-preflight] manifest={result['cpu_simulator_preflight_manifest_path']}")
    print(f"[cpu-simulator-preflight] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
