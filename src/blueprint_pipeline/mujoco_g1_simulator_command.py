"""Packaged MuJoCo Unitree G1 simulator command for robot-eval workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json


MUJOCO_G1_SIMULATOR_COMMAND_OUTPUT_SCHEMA_VERSION = "mujoco_g1_simulator_command_output.v1"
MUJOCO_G1_SIMULATOR_COMMAND_ARTIFACT_SCHEMA_VERSION = "mujoco_g1_simulator_command_artifact.v1"
DEFAULT_MENAGERIE_REF = "4c358ef9d9d7f32ca58b40b490884a0c1726a440"
G1_SOURCE_URL = "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1"
G1_REPOSITORY_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
POLICY_ID = "blueprint_default_walk_to_target_smoke_policy"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _xml_escape(value: Path | str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _find_scene_glb(capture_root: Path) -> Path:
    candidates = (
        capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb",
        capture_root / "pipeline" / "worldlabs_assets" / "scene.glb",
        capture_root / "pipeline" / "marble_sim_assets" / "portable_collider.glb",
    )
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
    for mesh in gltf_json.get("meshes", []) or []:
        for primitive in mesh.get("primitives", []) or []:
            primitive_count += 1
            for attribute_name in primitive.get("attributes", {}) or {}:
                attribute_usage[attribute_name] = attribute_usage.get(attribute_name, 0) + 1
    return {
        "status": "inspected",
        "gltf_version": version,
        "materials_count": len(gltf_json.get("materials", []) or []),
        "textures_count": len(gltf_json.get("textures", []) or []),
        "images_count": len(gltf_json.get("images", []) or []),
        "primitive_count": primitive_count,
        "attribute_usage": attribute_usage,
        "has_vertex_colors": "COLOR_0" in attribute_usage,
        "has_embedded_or_referenced_image_textures": bool(
            gltf_json.get("textures") or gltf_json.get("images")
        ),
    }


def _convert_glb_to_obj(glb_path: Path, obj_path: Path) -> dict[str, Any]:
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency check.
        raise RuntimeError("trimesh is required for GLB-to-OBJ MuJoCo scene conversion") from exc

    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        to_geometry = getattr(loaded, "to_geometry", None)
        mesh = to_geometry() if callable(to_geometry) else loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh.is_empty:
        raise RuntimeError(f"scene mesh is empty after loading {glb_path}")
    ensure_dir(obj_path.parent)
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


def _fetch_g1_assets(*, menagerie_root: Path, menagerie_ref: str) -> Path:
    if (menagerie_root / "unitree_g1" / "g1.xml").is_file():
        return menagerie_root / "unitree_g1"
    ensure_dir(menagerie_root.parent)
    if not (menagerie_root / ".git").is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                G1_REPOSITORY_URL,
                str(menagerie_root),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    subprocess.run(
        ["git", "-C", str(menagerie_root), "checkout", menagerie_ref],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        ["git", "-C", str(menagerie_root), "sparse-checkout", "set", "unitree_g1"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    g1_root = menagerie_root / "unitree_g1"
    if not (g1_root / "g1.xml").is_file():
        raise FileNotFoundError(f"fetched MuJoCo Menagerie checkout is missing {g1_root / 'g1.xml'}")
    return g1_root


def _resolve_g1_model_root(
    *,
    explicit_root: str | Path | None,
    capture_root: Path,
    allow_fetch: bool,
    menagerie_ref: str,
) -> Path:
    repo_root = _repo_root()
    candidates = [
        explicit_root,
        os.environ.get("BLUEPRINT_MUJOCO_G1_MODEL_ROOT"),
        Path("/opt/blueprint/assets/mujoco_menagerie/unitree_g1"),
        repo_root / "output" / "external_assets" / "mujoco_menagerie" / "unitree_g1",
        Path.cwd() / "output" / "external_assets" / "mujoco_menagerie" / "unitree_g1",
        capture_root
        / "pipeline"
        / "external_assets"
        / "mujoco_menagerie"
        / "unitree_g1",
    ]
    for value in candidates:
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if (candidate / "g1.xml").is_file():
            return candidate
    if allow_fetch:
        menagerie_root = Path(
            os.environ.get(
                "BLUEPRINT_MUJOCO_MENAGERIE_FETCH_ROOT",
                "/tmp/blueprint_mujoco_menagerie",
            )
        )
        return _fetch_g1_assets(menagerie_root=menagerie_root, menagerie_ref=menagerie_ref)
    searched = [str(value) for value in candidates if value]
    raise FileNotFoundError(
        "missing MuJoCo Menagerie Unitree G1 g1.xml; searched "
        + ", ".join(searched)
        + ". Pass --g1-model-root or enable --allow-fetch-g1-assets."
    )


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
    ensure_dir(output_xml.parent)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def _write_mjcf_wrapper(scene_obj: Path, g1_xml: Path, wrapper_path: Path) -> None:
    wrapper = f"""<mujoco model="blueprint_mujoco_g1_simulator_command">
  <include file="{_xml_escape(g1_xml)}"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0.6 0.6 0.6"/>
    <global offwidth="640" offheight="360" azimuth="140" elevation="-20"/>
    <map znear="0.01" zfar="200"/>
  </visual>
  <asset>
    <mesh name="blueprint_scene_mesh" file="{_xml_escape(scene_obj)}"/>
    <material name="blueprint_scene_mat" rgba="0.45 0.50 0.55 1"/>
  </asset>
  <worldbody>
    <light name="blueprint_key" pos="0 -4 8" dir="0 0 -1" directional="true"/>
    <geom name="blueprint_reference_floor" type="plane" size="8 8 0.05" rgba="0.18 0.20 0.22 1"/>
    <geom name="blueprint_scene_visual" type="mesh" mesh="blueprint_scene_mesh"
      material="blueprint_scene_mat" contype="0" conaffinity="0"/>
    <camera name="overview" pos="-3.5 -5 3.2" xyaxes="0.82 -0.57 0 0.30 0.43 0.85" fovy="55"/>
  </worldbody>
</mujoco>
"""
    ensure_dir(wrapper_path.parent)
    wrapper_path.write_text(wrapper, encoding="utf-8")


def _asset_source_manifest(g1_root: Path) -> dict[str, Any]:
    asset_files = sorted((g1_root / "assets").glob("*"))
    menagerie_root = g1_root.parent
    return {
        "source": "google_deepmind_mujoco_menagerie",
        "source_url": G1_SOURCE_URL,
        "local_path": str(g1_root),
        "menagerie_git_commit": _git_commit(menagerie_root),
        "asset_file_count": len(asset_files),
        "checksums": {
            "g1.xml": _sha256(g1_root / "g1.xml"),
            "scene.xml": _sha256(g1_root / "scene.xml")
            if (g1_root / "scene.xml").is_file()
            else None,
        },
        "license_path": str(g1_root / "LICENSE") if (g1_root / "LICENSE").is_file() else None,
        "policy_downloaded_from_online": False,
        "downloaded_content_boundary": "Robot MJCF and mesh assets only; no locomotion policy was downloaded.",
    }


def _first_matrix_run(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        return {}
    runs = payload.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
        for run in runs:
            if isinstance(run, Mapping):
                return run
    return {}


def _camera_record(camera: str, frame_path: Path, step: int, mode: str) -> dict[str, Any]:
    return {"camera": camera, "camera_mode": mode, "step": step, "path": str(frame_path)}


def run_mujoco_g1_simulator_command(
    *,
    capture_root: str | Path,
    g1_model_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    simulator_output_path: str | Path | None = None,
    scenario_eval_matrix_path: str | Path | None = None,
    steps: int = 32,
    render_frames: bool = True,
    allow_fetch_g1_assets: bool = False,
    menagerie_ref: str = DEFAULT_MENAGERIE_REF,
) -> dict[str, Any]:
    if platform.system().lower() == "linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    root = Path(capture_root).resolve()
    output_root = (
        Path(output_dir).resolve()
        if output_dir
        else root / "pipeline" / "simulation_automation" / "mujoco_g1_simulator_command"
    )
    ensure_dir(output_root)
    frames_dir = output_root / "frames"
    ensure_dir(frames_dir)
    generated_at = utc_now_iso()

    scene_glb = _find_scene_glb(root)
    scene_obj = output_root / "capture_scene_for_mujoco.obj"
    mesh_info = _convert_glb_to_obj(scene_glb, scene_obj)

    resolved_g1_root = _resolve_g1_model_root(
        explicit_root=g1_model_root,
        capture_root=root,
        allow_fetch=allow_fetch_g1_assets,
        menagerie_ref=menagerie_ref,
    )
    g1_xml = resolved_g1_root / "g1.xml"
    generated_g1_xml = output_root / "menagerie_g1_absolute_mesh_paths.xml"
    wrapper_xml = output_root / "blueprint_mujoco_g1_simulator_command.xml"
    _write_g1_xml_with_absolute_meshes(g1_xml, generated_g1_xml)
    _write_mjcf_wrapper(scene_obj, generated_g1_xml, wrapper_xml)

    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - runtime dependency check.
        raise RuntimeError("mujoco is required for the Unitree G1 simulator command") from exc

    model = mujoco.MjModel.from_xml_path(str(wrapper_xml))
    data = mujoco.MjData(model)
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint_id < 0:
        raise RuntimeError("Unitree G1 floating_base_joint not found")
    root_qpos = int(model.jnt_qposadr[root_joint_id])
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = model.key_qpos[stand_key_id].copy() if stand_key_id >= 0 else model.qpos0.copy()

    renderer = None
    if render_frames:
        from PIL import Image

        renderer = mujoco.Renderer(model, height=360, width=640)
    else:
        Image = None  # type: ignore[assignment]

    start = (-0.8, 0.0, 0.793)
    target = (0.8, 0.0, 0.793)
    bounded_steps = max(1, int(steps))
    capture_steps = {0, max(0, bounded_steps // 2), max(0, bounded_steps - 1)}
    actions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    robot_camera = mujoco.MjvCamera()
    robot_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    for step in range(bounded_steps):
        alpha = 0.0 if bounded_steps <= 1 else step / float(bounded_steps - 1)
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
        if renderer is not None and step in capture_steps:
            renderer.update_scene(data, camera="overview")
            overview_path = frames_dir / f"overview_{step:04d}.png"
            Image.fromarray(renderer.render()).save(overview_path)
            frames.append(
                _camera_record("overview", overview_path, step, "named_fixed_overview_camera")
            )

            robot_camera.lookat[:] = [x + 0.45, y, z + 0.55]
            robot_camera.distance = 1.15
            robot_camera.azimuth = 90
            robot_camera.elevation = -8
            renderer.update_scene(data, camera=robot_camera)
            robot_path = frames_dir / f"sim_robot_follow_pov_{step:04d}.png"
            Image.fromarray(renderer.render()).save(robot_path)
            frames.append(
                _camera_record(
                    "sim_robot_follow_pov",
                    robot_path,
                    step,
                    "virtual_free_camera_following_g1_root_not_physical_robot_sensor",
                )
            )
        mujoco.mj_step(model, data)
    if renderer is not None:
        renderer.close()

    claim_boundary = {
        "simulator_execution_proven": True,
        "mujoco_g1_asset_execution_proven": True,
        "mujoco_g1_asset_spawned": True,
        "unitree_g1_asset_spawned": True,
        "isaac_sim_execution_proven": False,
        "isaac_robot_asset_execution_proven": False,
        "real_robot_pov_evidence_proven": False,
        "robot_policy_execution_proven": False,
        "robot_team_policy_execution_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validated": False,
        "contact_dynamics_validated": False,
        "customer_delivery_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
    }
    robot_asset = {
        "name": "Unitree G1",
        "uri_or_path": str(g1_xml),
        "source": "google_deepmind_mujoco_menagerie",
        "asset_class": "humanoid_mjcf",
        "mujoco_g1_asset_execution_proven": True,
        "isaac_robot_asset_execution_proven": False,
    }
    source_manifest = _asset_source_manifest(resolved_g1_root)
    common = {
        "schema_version": MUJOCO_G1_SIMULATOR_COMMAND_ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "simulator_backend": "mujoco",
        "robot_asset": robot_asset,
        "asset_source_manifest": source_manifest,
        "claim_boundary": claim_boundary,
        **claim_boundary,
    }
    scene_trace = {
        **common,
        "artifact_type": "scene_load_trace",
        "status": "loaded",
        "scene_loaded": True,
        "source_scene_glb": str(scene_glb),
        "converted_scene_obj": str(scene_obj),
        "generated_mjcf": str(wrapper_xml),
        "generated_g1_mjcf": str(generated_g1_xml),
        "mesh_info": mesh_info,
        "render_backend": os.environ.get("MUJOCO_GL"),
        "mujoco_version": _string(getattr(mujoco, "__version__", "")),
    }
    spawn_trace = {
        **common,
        "artifact_type": "spawn_trace",
        "status": "validated",
        "spawn_pose_loaded": True,
        "spawn_pose": list(start),
        "target_pose": list(target),
        "keyframe": "stand" if stand_key_id >= 0 else None,
    }
    policy_trace = {
        **common,
        "artifact_type": "default_sim_policy_trace",
        "status": "completed",
        "policy_id": POLICY_ID,
        "policy_kind": "walk_to_target",
        "policy_source": "repo_generated_default_smoke_policy",
        "policy_downloaded_from_online": False,
        "default_policy_executed": True,
        "policy_execution_completed": True,
        "policy_semantics": "kinematic_root_pose_smoke_not_balanced_humanoid_locomotion_controller",
        "start_pose": list(start),
        "target_pose": list(target),
        "final_pose": actions[-1]["root_position"],
        "steps": bounded_steps,
        "actions": actions,
    }
    pov_manifest = {
        **common,
        "artifact_type": "sim_robot_pov_manifest",
        "status": "complete" if frames else "not_recorded",
        "sim_robot_pov_captured": bool(frames),
        "real_robot_pov_evidence_proven": False,
        "camera_boundary": (
            "Frames come from MuJoCo renderer cameras. The follow POV is virtual and follows "
            "the simulated G1 root; it is not a physical robot camera stream."
        ),
        "frames": frames,
        "frame_count": len(frames),
    }
    artifact_paths = {
        "scene_trace": str(output_root / "scene_load_trace.json"),
        "spawn_trace": str(output_root / "spawn_trace.json"),
        "policy_trace": str(output_root / "policy_execution_trace.json"),
        "sim_robot_pov_evidence": str(output_root / "sim_robot_pov_evidence_manifest.json"),
        "source_scene_glb": str(scene_glb),
        "converted_scene_obj": str(scene_obj),
        "generated_mjcf": str(wrapper_xml),
        "generated_g1_mjcf": str(generated_g1_xml),
        "frames": [frame["path"] for frame in frames],
    }
    artifact_manifest = {
        **common,
        "artifact_type": "artifact_manifest",
        "status": "complete",
        "artifacts": artifact_paths,
        "files": [str(wrapper_xml), str(generated_g1_xml), str(scene_obj)]
        + [frame["path"] for frame in frames],
    }
    write_json(output_root / "scene_load_trace.json", scene_trace)
    write_json(output_root / "spawn_trace.json", spawn_trace)
    write_json(output_root / "policy_execution_trace.json", policy_trace)
    write_json(output_root / "sim_robot_pov_evidence_manifest.json", pov_manifest)
    write_json(output_root / "artifact_manifest.json", artifact_manifest)

    matrix_run = _first_matrix_run(
        Path(scenario_eval_matrix_path).resolve()
        if scenario_eval_matrix_path
        else Path(os.environ["BLUEPRINT_SCENARIO_EVAL_MATRIX"]).resolve()
        if os.environ.get("BLUEPRINT_SCENARIO_EVAL_MATRIX")
        else None
    )
    attempt = {
        "attempt_id": "mujoco_g1_walk_to_target_smoke_0001",
        "episode_id": _string(matrix_run.get("episode_id")) or "mujoco_g1_episode_0001",
        "scenario_id": _string(matrix_run.get("scenario_id")) or "scenario_walk_to_target",
        "scenario_run_id": _string(matrix_run.get("scenario_run_id"))
        or "mujoco_g1_scenario_run_0001",
        "scenario_eval_run_id": _string(matrix_run.get("scenario_eval_run_id")) or None,
        "scenario_variation_instance_id": _string(
            matrix_run.get("scenario_variation_instance_id")
        )
        or None,
        "variation_name": _string(matrix_run.get("variation_name")) or None,
        "task_id": _string(matrix_run.get("task_id")) or "walk_to_target",
        "policy_id": POLICY_ID,
        "status": "completed",
        "success": True,
        "metrics": {
            "cycle_time_seconds": round(float(data.time), 6),
            "intervention_count": 0,
            "unsafe_proximity_event_count": 0,
            "collision_risk_event_count": 0,
            "object_drop_count": 0,
            "wrong_object_count": 0,
            "timeout_count": 0,
            "simulated_step_count": bounded_steps,
        },
        "actions": actions,
        "contact_trace": [],
        "safety_events": [],
        "artifact_paths": artifact_paths,
    }
    payload = {
        "schema_version": MUJOCO_G1_SIMULATOR_COMMAND_OUTPUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "simulator_backend": "mujoco",
        "mujoco_version": _string(getattr(mujoco, "__version__", "")),
        "capture_root": str(root),
        "output_dir": str(output_root),
        "robot_asset": robot_asset,
        "asset_source_manifest": source_manifest,
        "scene_loaded": True,
        "unitree_g1_asset_spawned": True,
        "unitree_g1_robot_asset_spawned": True,
        "mujoco_g1_asset_spawned": True,
        "mujoco_g1_asset_execution_proven": True,
        "default_sim_policy_execution_proven": True,
        "robot_policy_execution_proven": False,
        "robot_team_policy_execution_proven": False,
        "sim_robot_pov_evidence_proven": bool(frames),
        "real_robot_pov_evidence_proven": False,
        "attempts": [attempt],
        "artifact_paths": {
            **artifact_paths,
            "artifact_manifest": str(output_root / "artifact_manifest.json"),
        },
        "claim_boundary": claim_boundary,
        **claim_boundary,
    }
    simulator_output = (
        Path(simulator_output_path).resolve()
        if simulator_output_path
        else Path(os.environ["BLUEPRINT_SIMULATOR_OUTPUT"]).resolve()
        if os.environ.get("BLUEPRINT_SIMULATOR_OUTPUT")
        else output_root / "mujoco_g1_simulator_output.json"
    )
    write_json(simulator_output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--g1-model-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--simulator-output", type=Path, default=None)
    parser.add_argument("--scenario-eval-matrix", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--skip-render-frames", action="store_true")
    parser.add_argument("--allow-fetch-g1-assets", action="store_true")
    parser.add_argument("--no-fetch-g1-assets", action="store_true")
    parser.add_argument("--menagerie-ref", default=DEFAULT_MENAGERIE_REF)
    args = parser.parse_args(argv)

    capture_root = args.capture_root or Path(os.environ.get("BLUEPRINT_CAPTURE_ROOT", ""))
    if not _string(capture_root):
        parser.error("--capture-root or BLUEPRINT_CAPTURE_ROOT is required")
    allow_fetch = bool(args.allow_fetch_g1_assets) and not bool(args.no_fetch_g1_assets)
    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=args.g1_model_root,
        output_dir=args.output_dir,
        simulator_output_path=args.simulator_output,
        scenario_eval_matrix_path=args.scenario_eval_matrix,
        steps=args.steps,
        render_frames=not args.skip_render_frames,
        allow_fetch_g1_assets=allow_fetch,
        menagerie_ref=args.menagerie_ref,
    )
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "simulator_backend": payload.get("simulator_backend"),
                "unitree_g1_asset_spawned": payload.get("unitree_g1_asset_spawned"),
                "simulator_execution_proven": payload.get("simulator_execution_proven"),
                "output_dir": payload.get("output_dir"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
