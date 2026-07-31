#!/usr/bin/env python3
"""Local MuJoCo Unitree G1 smoke for the staged World Labs scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from defusedxml import ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


G1_SOURCE_URL = "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1"
POLICY_ID = "blueprint_default_walk_to_target_smoke_policy"

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.artifact_storage import default_artifact_cache_root  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _find_scene_glb(capture_root: Path) -> Path:
    candidates = [
        capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb",
        capture_root / "pipeline" / "worldlabs_assets" / "scene.glb",
        capture_root / "pipeline" / "marble_sim_assets" / "portable_collider.glb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted((capture_root / "pipeline").glob("**/*.glb"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no GLB scene asset found under {capture_root / 'pipeline'}")


def _glb_visual_summary(glb_path: Path) -> dict[str, Any]:
    data = glb_path.read_bytes()
    if len(data) < 20:
        return {"status": "unreadable", "reason": "file_too_short"}
    magic, version, _declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        return {"status": "unreadable", "reason": "not_binary_gltf"}
    offset = 12
    gltf_json: dict[str, Any] | None = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            gltf_json = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
            break
    if gltf_json is None:
        return {"status": "unreadable", "reason": "missing_gltf_json_chunk"}

    primitive_count = 0
    attribute_usage: dict[str, int] = {}
    for mesh in gltf_json.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitive_count += 1
            for attribute_name in primitive.get("attributes", {}):
                attribute_usage[attribute_name] = attribute_usage.get(attribute_name, 0) + 1
    return {
        "status": "inspected",
        "gltf_version": version,
        "materials_count": len(gltf_json.get("materials", [])),
        "textures_count": len(gltf_json.get("textures", [])),
        "images_count": len(gltf_json.get("images", [])),
        "primitive_count": primitive_count,
        "attribute_usage": attribute_usage,
        "has_vertex_colors": "COLOR_0" in attribute_usage,
        "has_embedded_or_referenced_image_textures": bool(
            gltf_json.get("textures") or gltf_json.get("images")
        ),
    }


def _convert_glb_to_obj(glb_path: Path, obj_path: Path) -> dict[str, Any]:
    import trimesh

    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        to_geometry = getattr(loaded, "to_geometry", None)
        mesh = to_geometry() if callable(to_geometry) else loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh.is_empty:
        raise RuntimeError(f"scene mesh is empty after loading {glb_path}")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(obj_path)
    return {
        "source_glb": str(glb_path),
        "converted_obj": str(obj_path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "centroid": mesh.centroid.tolist(),
        "visual_asset_summary": _glb_visual_summary(glb_path),
        "mujoco_visual_fidelity_boundary": (
            "The smoke converts the GLB to OBJ and renders it with a plain MJCF material; "
            "it proves geometry load, not texture, PBR material, or vertex-color fidelity."
        ),
    }


def _xml_escape(value: Path | str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _resolve_g1_model_root(value: str | None) -> Path:
    candidate = Path(
        value
        or os.environ.get("BLUEPRINT_MUJOCO_G1_MODEL_ROOT", "")
        or default_artifact_cache_root() / "external_assets" / "mujoco_menagerie" / "unitree_g1"
    )
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    required = candidate / "g1.xml"
    if not required.is_file():
        raise FileNotFoundError(
            f"missing MuJoCo Menagerie Unitree G1 model at {required}. "
            "Fetch it with a sparse clone of google-deepmind/mujoco_menagerie "
            "or pass --g1-model-root."
        )
    return candidate


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _asset_source_manifest(g1_root: Path) -> dict[str, Any]:
    menagerie_root = g1_root.parent
    asset_files = sorted((g1_root / "assets").glob("*.STL"))
    checksums = {
        "g1.xml": _sha256(g1_root / "g1.xml"),
        "scene.xml": _sha256(g1_root / "scene.xml") if (g1_root / "scene.xml").is_file() else None,
    }
    return {
        "source": "google_deepmind_mujoco_menagerie",
        "source_url": G1_SOURCE_URL,
        "local_path": str(g1_root),
        "menagerie_git_commit": _git_commit(menagerie_root),
        "asset_file_count": len(asset_files),
        "checksums": checksums,
        "license_path": str(g1_root / "LICENSE") if (g1_root / "LICENSE").is_file() else None,
        "policy_downloaded_from_online": False,
        "downloaded_content_boundary": "Robot MJCF and mesh assets only; no locomotion policy was downloaded.",
    }


def _write_mjcf_wrapper(scene_obj: Path, g1_xml: Path, wrapper_path: Path) -> None:
    wrapper = f"""<mujoco model="blueprint_local_mujoco_g1_walk_to_target_smoke">
  <include file="{_xml_escape(g1_xml)}"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0.6 0.6 0.6"/>
    <global offwidth="1280" offheight="720" azimuth="140" elevation="-20"/>
    <map znear="0.01" zfar="200"/>
  </visual>
  <asset>
    <mesh name="worldlabs_scene_mesh" file="{_xml_escape(scene_obj)}"/>
    <material name="blueprint_worldlabs_scene_mat" rgba="0.45 0.50 0.55 1"/>
  </asset>
  <worldbody>
    <light name="blueprint_key" pos="0 -4 8" dir="0 0 -1" directional="true"/>
    <geom name="blueprint_reference_floor" type="plane" size="8 8 0.05" rgba="0.18 0.20 0.22 1"/>
    <geom name="worldlabs_scene_visual" type="mesh" mesh="worldlabs_scene_mesh"
      material="blueprint_worldlabs_scene_mat" contype="0" conaffinity="0"/>
    <camera name="overview" pos="-3.5 -5 3.2" xyaxes="0.82 -0.57 0 0.30 0.43 0.85" fovy="55"/>
  </worldbody>
</mujoco>
"""
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(wrapper, encoding="utf-8")


def _write_g1_xml_with_absolute_meshes(source_xml: Path, output_xml: Path) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    assets_dir = source_xml.parent / "assets"
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(assets_dir))
    for mesh in root.findall(".//mesh"):
        mesh_file = _string(mesh.get("file"))
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.set("file", str(assets_dir / mesh_file))
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def _camera_summary(camera_name: str, frame_path: Path, step: int, camera_mode: str) -> dict[str, Any]:
    return {
        "camera": camera_name,
        "camera_mode": camera_mode,
        "step": step,
        "path": str(frame_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--g1-model-root", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-label", default="walk_to_target_pose")
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    capture_root = args.capture_root.resolve()
    g1_root = _resolve_g1_model_root(args.g1_model_root)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else capture_root / "pipeline" / "simulation_automation" / "mujoco_g1_local_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene_glb = _find_scene_glb(capture_root)
    scene_obj = output_dir / "worldlabs_scene_for_mujoco.obj"
    mesh_info = _convert_glb_to_obj(scene_glb, scene_obj)
    wrapper_xml = output_dir / "blueprint_local_mujoco_g1_walk_to_target_smoke.xml"
    g1_xml = g1_root / "g1.xml"
    generated_g1_xml = output_dir / "menagerie_g1_absolute_mesh_paths.xml"
    _write_g1_xml_with_absolute_meshes(g1_xml, generated_g1_xml)
    _write_mjcf_wrapper(scene_obj, generated_g1_xml, wrapper_xml)

    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(wrapper_xml))
    data = mujoco.MjData(model)
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint_id < 0:
        raise RuntimeError("Unitree G1 floating_base_joint not found")
    root_qpos = int(model.jnt_qposadr[root_joint_id])
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = (
        model.key_qpos[stand_key_id].copy()
        if stand_key_id >= 0
        else model.qpos0.copy()
    )

    renderer = mujoco.Renderer(model, height=720, width=1280)
    robot_camera = mujoco.MjvCamera()
    robot_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    overview_camera = "overview"

    start = (-0.8, 0.0, 0.793)
    target = (0.8, 0.0, 0.793)
    capture_steps = {0, max(0, args.steps // 2), max(0, args.steps - 1)}
    actions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for step in range(args.steps):
        alpha = 0.0 if args.steps <= 1 else step / float(args.steps - 1)
        x = start[0] + (target[0] - start[0]) * alpha
        y = start[1] + (target[1] - start[1]) * alpha
        z = start[2] + (target[2] - start[2]) * alpha
        data.qpos[:] = base_qpos
        data.qvel[:] = 0
        data.qpos[root_qpos : root_qpos + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        actions.append(
            {
                "step": step,
                "root_position": [x, y, z],
                "target": list(target),
                "policy_action": "kinematic_root_interpolation",
            }
        )
        if step in capture_steps:
            renderer.update_scene(data, camera=overview_camera)
            overview_pixels = renderer.render()
            overview_path = frames_dir / f"overview_{step:04d}.png"
            Image.fromarray(overview_pixels).save(overview_path)
            frames.append(
                _camera_summary(
                    "overview",
                    overview_path,
                    step,
                    "named_fixed_overview_camera",
                )
            )

            robot_camera.lookat[:] = [x + 0.45, y, z + 0.55]
            robot_camera.distance = 1.15
            robot_camera.azimuth = 90
            robot_camera.elevation = -8
            renderer.update_scene(data, camera=robot_camera)
            robot_pixels = renderer.render()
            robot_path = frames_dir / f"sim_robot_follow_pov_{step:04d}.png"
            Image.fromarray(robot_pixels).save(robot_path)
            frames.append(
                _camera_summary(
                    "sim_robot_follow_pov",
                    robot_path,
                    step,
                    "virtual_free_camera_following_g1_root_not_physical_robot_sensor",
                )
            )

    robot_asset = {
        "name": "Unitree G1",
        "uri_or_path": str(g1_xml),
        "source": "google_deepmind_mujoco_menagerie",
        "asset_class": "humanoid_mjcf",
        "mujoco_g1_asset_execution_proven": True,
        "isaac_robot_asset_execution_proven": False,
    }
    claim_boundary = {
        "local_cpu_mujoco_execution_proven": True,
        "mujoco_g1_asset_execution_proven": True,
        "owner_gpu_simulator_execution_proven": False,
        "isaac_sim_execution_proven": False,
        "isaac_robot_asset_execution_proven": False,
        "real_robot_pov_evidence_proven": False,
        "robot_policy_execution_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "non_ranking_operational_claim_validated": False,
        "contact_dynamics_validated": False,
        "customer_delivery_readiness_proven": False,
        "live_webapp_forwarding_proven": False,
        "public_claim_upgrade_allowed": False,
    }
    source_manifest = _asset_source_manifest(g1_root)
    scene_trace = {
        "schema_version": "local_mujoco_g1_scene_load_trace.v1",
        "status": "loaded",
        "scene_loaded": True,
        "simulator_backend": "mujoco",
        "source_scene_glb": str(scene_glb),
        "converted_scene_obj": str(scene_obj),
        "generated_mjcf": str(wrapper_xml),
        "mesh_info": mesh_info,
        "robot_asset": robot_asset,
        "recorded_at": _now(),
        **claim_boundary,
    }
    spawn_trace = {
        "schema_version": "local_mujoco_g1_spawn_trace.v1",
        "status": "validated",
        "spawn_pose_loaded": True,
        "simulator_backend": "mujoco",
        "robot_asset": robot_asset,
        "spawn_pose": list(start),
        "target_pose": list(target),
        "keyframe": "stand" if stand_key_id >= 0 else None,
        "recorded_at": _now(),
        **claim_boundary,
    }
    policy_trace = {
        "schema_version": "local_mujoco_g1_policy_execution_trace.v1",
        "status": "completed",
        "policy_id": POLICY_ID,
        "policy_kind": "walk_to_target",
        "policy_source": "repo_generated_default_smoke_policy",
        "policy_downloaded_from_online": False,
        "target_label": args.target_label,
        "default_policy_executed": True,
        "policy_execution_completed": True,
        "policy_semantics": "kinematic_root_pose_smoke_not_balanced_humanoid_locomotion_controller",
        "start_pose": list(start),
        "target_pose": list(target),
        "final_pose": actions[-1]["root_position"],
        "steps": args.steps,
        "actions": actions,
        "recorded_at": _now(),
        **claim_boundary,
    }
    pov_manifest = {
        "schema_version": "local_mujoco_g1_sim_robot_pov_evidence.v1",
        "status": "complete",
        "simulator_backend": "mujoco",
        "sim_robot_pov_captured": True,
        "real_robot_pov_evidence_proven": False,
        "camera_boundary": (
            "Frames come from MuJoCo renderer cameras. The follow POV is virtual and follows "
            "the simulated G1 root; it is not a physical robot camera stream."
        ),
        "frames": frames,
        "frame_count": len(frames),
        "recorded_at": _now(),
        **claim_boundary,
    }
    artifact_manifest = {
        "schema_version": "local_mujoco_g1_artifact_manifest.v1",
        "status": "complete",
        "simulator_backend": "mujoco",
        "robot_asset": robot_asset,
        "asset_source_manifest": source_manifest,
        "artifacts": {
            "scene_trace": str(output_dir / "scene_load_trace.json"),
            "spawn_trace": str(output_dir / "spawn_trace.json"),
            "policy_trace": str(output_dir / "policy_execution_trace.json"),
            "sim_robot_pov_evidence": str(output_dir / "sim_robot_pov_evidence_manifest.json"),
            "source_scene_glb": str(scene_glb),
            "converted_scene_obj": str(scene_obj),
            "generated_mjcf": str(wrapper_xml),
            "generated_g1_mjcf": str(generated_g1_xml),
            "frames": [frame["path"] for frame in frames],
        },
        "recorded_at": _now(),
        **claim_boundary,
    }
    manifest = {
        "schema_version": "local_mujoco_g1_walk_to_target_smoke_manifest.v1",
        "generated_at": _now(),
        "status": "complete",
        "capture_root": str(capture_root),
        "output_dir": str(output_dir),
        "simulator_backend": "mujoco",
        "mujoco_version": _string(getattr(mujoco, "__version__", "")),
        "robot_asset": robot_asset,
        "asset_source_manifest": source_manifest,
        "scene_loaded": True,
        "unitree_g1_asset_spawned": True,
        "mujoco_g1_asset_execution_proven": True,
        "local_cpu_mujoco_execution_proven": True,
        "default_sim_policy_execution_proven": True,
        "sim_robot_pov_evidence_proven": True,
        "policy_id": POLICY_ID,
        "policy_downloaded_from_online": False,
        "policy_semantics": policy_trace["policy_semantics"],
        "scene_visual_asset_summary": mesh_info["visual_asset_summary"],
        "artifacts": artifact_manifest["artifacts"],
        "claim_boundary": claim_boundary,
        **claim_boundary,
    }

    _write_json(output_dir / "scene_load_trace.json", scene_trace)
    _write_json(output_dir / "spawn_trace.json", spawn_trace)
    _write_json(output_dir / "policy_execution_trace.json", policy_trace)
    _write_json(output_dir / "sim_robot_pov_evidence_manifest.json", pov_manifest)
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    _write_json(output_dir / "mujoco_g1_local_smoke_manifest.json", manifest)
    print(json.dumps({"status": "complete", "manifest": str(output_dir / "mujoco_g1_local_smoke_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
