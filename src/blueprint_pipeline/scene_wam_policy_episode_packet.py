"""Build a scene-grounded WAM/policy episode packet.

The packet is a setup artifact for simulator/generated-world loops. It can
prepare a robot/head-POV policy observation from a local USD scene when OpenUSD
rendering is available, but it never upgrades physics, safety, deployment, or
physical-robot readiness claims.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import time
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


def _scene_bounds_and_target(scene_asset: Path, *, target_object_id: str) -> dict[str, Any]:
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
    eye = Gf.Vec3d(robot_xyz[0], robot_xyz[1], robot_xyz[2] + 1.35)
    look_at = Gf.Vec3d(target_xyz[0], target_xyz[1], target_xyz[2] + 0.65)
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
    for name in ("pxr", "open3d", "trimesh", "PIL"):
        try:
            __import__(name)
            packages[name] = True
        except Exception:
            packages[name] = False
    return {"commands": commands, "python_packages": packages}


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
    if not availability["python_packages"].get("pxr"):
        blockers.append("missing_python_package_pxr")
    if not usdrecord:
        blockers.append("missing_renderer_command_usdrecord")
    render_dir = output_dir / "rendered_observations"
    ensure_dir(render_dir)
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
            status_text = (
                "completed"
                if completed_process.returncode == 0 and expected_path.is_file()
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
    robot_fallback = [
        float(target_pose["xyz"][0]),
        float(target_pose["xyz"][1]) - 1.0,
        max(0.0, float(target_pose["xyz"][2]) - 1.0),
    ]
    robot_pose = _pose(
        robot_start_pose or episode.get("robot_spawn_pose"),
        fallback_xyz=robot_fallback,
        source="cli_or_episode_spec_or_target_relative_default",
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
            "blank_or_placeholder_image_used": False,
            "physical_robot_sensor_proof": False,
            "blockers": [] if frame_path else list(render.get("blockers", [])),
        },
        "robot_start_pose": robot_pose,
        "target_anchor_pose": target_pose,
        "state": {
            "root_position": robot_pose["xyz"],
            "root_yaw_rad": 0.0,
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
        "video_camera": video_camera,
        "scene_physics_required_for_wam_loop": False,
        "physics_contact_validated": False,
        "task_anchor_source": task_anchor.get("anchor_source"),
        "episode_source": episode.get("episode_id"),
        "blockers": [
            *([] if resolved_scene_asset.is_file() else ["scene_asset_missing"]),
            *list(scene_summary.get("blockers", [])),
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
        "scene_policy_wam_claim_boundary_path": str(
            resolved_output_dir / "scene_policy_wam_claim_boundary.json"
        ),
        "render_manifest_path": str(resolved_output_dir / "initial_policy_observation_render.json"),
        "initial_policy_observation_frame_path": frame_path,
        "scene_physics_required_for_wam_loop": False,
        "physics_contact_validated": False,
        "blockers": task_manifest["blockers"],
        "claim_boundary": claim_boundary,
    }
    write_json(resolved_output_dir / "initial_policy_observation_render.json", render)
    write_json(resolved_output_dir / "initial_policy_observation.json", initial_observation)
    write_json(resolved_output_dir / "scene_episode_task_manifest.json", task_manifest)
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
