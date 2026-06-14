"""Packaged MuJoCo Unitree G1 simulator command for robot-eval workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
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
RENDER_CAMERA_CONTRACT = [
    {
        "camera": "overview",
        "mode": "named_fixed_overview_camera",
        "proof_boundary": "virtual_mujoco_renderer_camera_not_physical_sensor",
    },
    {
        "camera": "sim_robot_follow_pov",
        "mode": "virtual_free_camera_following_g1_root_not_physical_robot_sensor",
        "proof_boundary": "virtual_mujoco_renderer_camera_not_physical_sensor",
    },
    {
        "camera": "side",
        "mode": "virtual_side_profile_camera_tracking_g1_root",
        "proof_boundary": "virtual_mujoco_renderer_camera_not_physical_sensor",
    },
]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or fallback


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


def _obj_vertex_color_summary(obj_path: Path) -> dict[str, Any]:
    vertex_count = 0
    vertex_rgb_count = 0
    face_count = 0
    rgb_min = [1.0, 1.0, 1.0]
    rgb_max = [0.0, 0.0, 0.0]
    sample_rgb: list[list[float]] = []
    with obj_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertex_count += 1
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    rgb = [float(parts[4]), float(parts[5]), float(parts[6])]
                except ValueError:
                    continue
                vertex_rgb_count += 1
                for index, channel in enumerate(rgb):
                    rgb_min[index] = min(rgb_min[index], channel)
                    rgb_max[index] = max(rgb_max[index], channel)
                if len(sample_rgb) < 5:
                    sample_rgb.append([round(channel, 6) for channel in rgb])
            elif line.startswith("f "):
                face_count += 1
    return {
        "status": "inspected",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "vertex_rgb_count": vertex_rgb_count,
        "vertex_rgb_fraction": round(vertex_rgb_count / vertex_count, 6)
        if vertex_count
        else 0.0,
        "has_vertex_rgb": vertex_rgb_count > 0,
        "rgb_min": [round(value, 6) for value in rgb_min] if vertex_rgb_count else None,
        "rgb_max": [round(value, 6) for value in rgb_max] if vertex_rgb_count else None,
        "sample_rgb": sample_rgb,
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
    obj_vertex_color_summary = _obj_vertex_color_summary(obj_path)
    glb_visual_summary = _glb_visual_summary(glb_path)
    collision_proxy_geoms, collision_proxy_summary = _collision_proxy_geoms_from_mesh(mesh)
    return {
        "source_glb": str(glb_path),
        "converted_obj": str(obj_path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "centroid": mesh.centroid.tolist(),
        "visual_asset_summary": glb_visual_summary,
        "obj_vertex_color_summary": obj_vertex_color_summary,
        "collision_proxy_geoms": collision_proxy_geoms,
        "collision_proxy_summary": collision_proxy_summary,
        "mujoco_visual_fidelity_boundary": (
            "The command converts the World Labs/Marble collider GLB to OBJ for MuJoCo. "
            "The OBJ carries vertex-color evidence when present, but the MuJoCo view is "
            "still simulator visual evidence, not photoreal Marble/SPZ or PBR texture proof."
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


def _collision_proxy_geoms_from_mesh(
    mesh: Any, *, max_proxies: int = 160
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]
    proxies: list[dict[str, Any]] = []
    skipped = {
        "floor_like": 0,
        "overhead": 0,
        "scene_shell": 0,
        "degenerate": 0,
    }
    for component_index, component in enumerate(components):
        try:
            bounds = component.bounds
            lower = [float(value) for value in bounds[0][:3]]
            upper = [float(value) for value in bounds[1][:3]]
        except Exception:
            skipped["degenerate"] += 1
            continue
        extents = [upper[index] - lower[index] for index in range(3)]
        if any(value <= 0.0 for value in extents):
            skipped["degenerate"] += 1
            continue
        z_min = lower[2]
        z_max = upper[2]
        xy_area = extents[0] * extents[1]
        volume = xy_area * extents[2]
        if z_max <= 0.14 and xy_area >= 2.0:
            skipped["floor_like"] += 1
            continue
        if z_min >= 2.35:
            skipped["overhead"] += 1
            continue
        if extents[0] >= 8.0 and extents[1] >= 8.0 and extents[2] >= 2.0:
            skipped["scene_shell"] += 1
            continue
        if volume <= 0.001:
            skipped["degenerate"] += 1
            continue
        margin = 0.035
        pos = [(lower[index] + upper[index]) / 2.0 for index in range(3)]
        size = [max(0.025, extents[index] / 2.0 + margin) for index in range(3)]
        proxies.append(
            {
                "source_component_index": component_index,
                "name": f"component_{component_index:04d}",
                "pos": [round(value, 6) for value in pos],
                "size": [round(value, 6) for value in size],
                "bounds": [
                    [round(value, 6) for value in lower],
                    [round(value, 6) for value in upper],
                ],
                "extents": [round(value, 6) for value in extents],
                "volume_m3_estimate": round(volume, 6),
            }
        )
    proxies.sort(key=lambda item: float(item["volume_m3_estimate"]), reverse=True)
    bounded = proxies[: max(0, max_proxies)]
    summary = {
        "status": "generated" if bounded else "not_generated",
        "source_component_count": len(components),
        "proxy_count": len(bounded),
        "max_proxy_count": max_proxies,
        "skipped": skipped,
        "generation_method": "component_aabb_obstacle_proxies_excluding_floor_overhead_and_scene_shell",
        "proof_boundary": (
            "Obstacle proxies are conservative MuJoCo box colliders derived from scene "
            "components. They are better than colliding with the entire visual mesh, but "
            "still need robot-team review before customer safety claims."
        ),
    }
    return bounded, summary


def _xml_float(value: Any) -> str:
    return f"{float(value):.6g}"


def _xml_vec(values: Sequence[Any]) -> str:
    return " ".join(_xml_float(value) for value in values)


def _write_mjcf_wrapper(
    scene_obj: Path,
    g1_xml: Path,
    wrapper_path: Path,
    *,
    collision_proxies: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    proxy_geoms = []
    for index, proxy in enumerate(collision_proxies or []):
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
        proxy_id = _safe_id(proxy.get("name"), fallback=f"proxy_{index:03d}")
        proxy_geoms.append(
            "    "
            f'<geom name="blueprint_collision_proxy_{index:03d}_{_xml_escape(proxy_id)}" '
            f'type="box" pos="{_xml_vec(pos[:3])}" size="{_xml_vec(size[:3])}" '
            'rgba="0.05 0.75 0.35 0.18" contype="1" conaffinity="1" group="3"/>'
        )
    if proxy_geoms:
        collision_geometry_block = "\n".join(proxy_geoms)
    else:
        collision_geometry_block = (
            '    <geom name="blueprint_scene_collision" type="mesh" mesh="blueprint_scene_mesh"\n'
            '      material="blueprint_scene_collision_mat" contype="1" conaffinity="1" group="3"/>'
        )
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
    <material name="blueprint_scene_collision_mat" rgba="0.05 0.75 0.35 0.18"/>
  </asset>
  <worldbody>
    <light name="blueprint_key" pos="0 -4 8" dir="0 0 -1" directional="true"/>
    <geom name="blueprint_reference_floor" type="plane" size="8 8 0.05" rgba="0.18 0.20 0.22 1"
      contype="1" conaffinity="1"/>
    <geom name="blueprint_scene_visual" type="mesh" mesh="blueprint_scene_mesh"
      material="blueprint_scene_mat" contype="0" conaffinity="0"/>
{collision_geometry_block}
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


def _matrix_runs(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None or not path.is_file():
        run = {
            "scenario_eval_run_id": "mujoco_g1_default_eval_run_0001",
            "episode_id": "mujoco_g1_episode_0001",
            "scenario_id": "scenario_walk_to_target",
            "scenario_run_id": "mujoco_g1_scenario_run_0001",
            "scenario_variation_instance_id": None,
            "variation_name": "default_walk_to_target_smoke",
            "task_id": "walk_to_target",
            "baseline_capture_layout": True,
            "claim_boundary": "standalone_default_run_not_job_matrix_proof",
        }
        return [
            run
        ], {
            "status": "synthesized_default_run",
            "scenario_eval_matrix_path": str(path) if path else None,
            "scenario_eval_run_count": 1,
            "required_scenario_eval_run_ids": [run["scenario_eval_run_id"]],
            "source": "no_scenario_eval_matrix_supplied",
        }
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        return [], {
            "status": "blocked_invalid_scenario_eval_matrix",
            "scenario_eval_matrix_path": str(path),
            "scenario_eval_run_count": 0,
            "required_scenario_eval_run_ids": [],
            "reason": "matrix_payload_not_mapping",
        }
    runs: list[dict[str, Any]] = []
    missing_run_id_indexes: list[int] = []
    raw_runs = payload.get("runs")
    if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, (str, bytes)):
        for index, raw_run in enumerate(raw_runs, start=1):
            if not isinstance(raw_run, Mapping):
                continue
            run = dict(raw_run)
            run_id = _string(run.get("scenario_eval_run_id"))
            if not run_id:
                missing_run_id_indexes.append(index)
            runs.append(run)
    required_ids = [
        _string(run.get("scenario_eval_run_id"))
        for run in runs
        if _string(run.get("scenario_eval_run_id"))
    ]
    duplicate_ids = sorted(
        {run_id for run_id in required_ids if required_ids.count(run_id) > 1}
    )
    declared_count = payload.get("scenario_eval_run_count")
    declared_count_matches_rows = True
    if isinstance(declared_count, int) and not isinstance(declared_count, bool):
        declared_count_matches_rows = declared_count == len(runs)
    return runs, {
        "status": _string(payload.get("status")) or "loaded",
        "scenario_eval_matrix_path": str(path),
        "scenario_eval_run_count": len(runs),
        "required_scenario_eval_run_ids": required_ids,
        "missing_scenario_eval_run_id_indexes": missing_run_id_indexes,
        "duplicate_scenario_eval_run_ids": duplicate_ids,
        "scenario_eval_run_ids_unique": not duplicate_ids,
        "matrix_declared_count_matches_rows": declared_count_matches_rows,
        "source_matrix_scenario_eval_run_count": declared_count,
        "matrix_blockers": payload.get("blockers") if isinstance(payload.get("blockers"), list) else [],
    }


def _first_matrix_run(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.is_file():
        return {}
    runs, _summary = _matrix_runs(path)
    return runs[0] if runs else {}


def _stable_episode_seed(run: Mapping[str, Any], index: int) -> int:
    stable_fields = {
        "index": index,
        "scenario_eval_run_id": run.get("scenario_eval_run_id"),
        "scenario_variation_instance_id": run.get("scenario_variation_instance_id"),
        "variation_name": run.get("variation_name"),
        "task_id": run.get("task_id"),
        "scenario_id": run.get("scenario_id"),
        "concrete_mutation": run.get("concrete_mutation"),
        "engine_mutations": run.get("engine_mutations"),
    }
    digest = hashlib.sha256(
        json.dumps(stable_fields, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


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


def _scene_route_frame(mesh_info: Mapping[str, Any]) -> tuple[float, float, float, float]:
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
                center_x = float(min_x + max_x) / 2.0
                center_y = float(min_y + max_y) / 2.0
                radius_x = max(0.45, min(1.25, abs(float(max_x - min_x)) * 0.35))
                radius_y = max(0.45, min(1.25, abs(float(max_y - min_y)) * 0.35))
                return center_x, center_y, radius_x, radius_y
    centroid = mesh_info.get("centroid")
    if isinstance(centroid, Sequence) and not isinstance(centroid, (str, bytes)) and len(centroid) >= 2:
        center_x = _number(centroid[0]) or 0.0
        center_y = _number(centroid[1]) or 0.0
        return center_x, center_y, 0.8, 0.8
    return 0.0, 0.0, 0.8, 0.8


def _rounded_pose(pose: Sequence[float]) -> tuple[float, float, float]:
    return (round(float(pose[0]), 6), round(float(pose[1]), 6), round(float(pose[2]), 6))


def _episode_navigation_spec(
    *,
    run: Mapping[str, Any],
    mesh_info: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    seed = _stable_episode_seed(run, index)
    start_keys = (
        "spawn_pose",
        "start_pose",
        "initial_pose",
        "robot_spawn_pose",
        "robot_start_pose",
        "start_xyz",
        "spawn_xyz",
    )
    target_keys = (
        "target_pose",
        "goal_pose",
        "navigation_target_pose",
        "robot_target_pose",
        "target_xyz",
        "goal_xyz",
    )
    explicit_start = _nested_pose(run, start_keys)
    explicit_target = _nested_pose(run, target_keys)
    center_x, center_y, radius_x, radius_y = _scene_route_frame(mesh_info)
    angle = math.radians((seed % 3600) / 10.0)
    skew = math.radians(((seed >> 8) % 41) - 20)
    derived_start = (
        center_x + radius_x * math.cos(angle),
        center_y + radius_y * math.sin(angle),
        0.793,
    )
    derived_target = (
        center_x + radius_x * math.cos(angle + math.pi + skew),
        center_y + radius_y * math.sin(angle + math.pi + skew),
        0.793,
    )
    start = explicit_start or derived_start
    target = explicit_target or derived_target
    if explicit_start and not explicit_target:
        target = (
            center_x - (explicit_start[0] - center_x),
            center_y - (explicit_start[1] - center_y),
            explicit_start[2],
        )
    if explicit_target and not explicit_start:
        start = (
            center_x - (explicit_target[0] - center_x),
            center_y - (explicit_target[1] - center_y),
            explicit_target[2],
        )
    route_source = (
        "matrix_explicit_spawn_and_target"
        if explicit_start and explicit_target
        else "matrix_explicit_spawn_deterministic_target"
        if explicit_start
        else "deterministic_spawn_matrix_explicit_target"
        if explicit_target
        else "deterministic_seeded_scene_route"
    )
    start = _rounded_pose(start)
    target = _rounded_pose(target)
    distance = math.sqrt(
        (target[0] - start[0]) ** 2
        + (target[1] - start[1]) ** 2
        + (target[2] - start[2]) ** 2
    )
    return {
        "seed": seed,
        "start": start,
        "target": target,
        "route_source": route_source,
        "route_distance_m": round(distance, 6),
        "scene_route_frame": {
            "center_xy": [round(center_x, 6), round(center_y, 6)],
            "radius_xy": [round(radius_x, 6), round(radius_y, 6)],
        },
    }


def _route_waypoints_from_run(
    *, run: Mapping[str, Any], start: Sequence[float], target: Sequence[float]
) -> tuple[list[tuple[float, float, float]], str]:
    for key in ("route_waypoints", "navigation_waypoints", "waypoints"):
        value = run.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            points = [_pose_triplet(item) for item in value]
            route = [point for point in points if point is not None]
            if route:
                normalized = [_rounded_pose(start)]
                normalized.extend(_rounded_pose(point) for point in route)
                normalized.append(_rounded_pose(target))
                return _dedupe_route_points(normalized), f"matrix_{key}"
    return _warehouse_aisle_preview_route(start=start, target=target), (
        "deterministic_warehouse_aisle_preview_route_not_navmesh"
    )


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


def _warehouse_aisle_preview_route(
    *, start: Sequence[float], target: Sequence[float]
) -> list[tuple[float, float, float]]:
    start_pose = _rounded_pose(start)
    target_pose = _rounded_pose(target)
    z = start_pose[2]
    central_x = 0.0
    cross_aisles = (8.8, 2.3, -6.7, -9.0)
    target_cross_y = min(cross_aisles, key=lambda value: abs(value - target_pose[1]))
    points = [
        start_pose,
        (central_x, start_pose[1], z),
        (central_x, target_cross_y, z),
        (target_pose[0], target_cross_y, z),
        target_pose,
    ]
    return _dedupe_route_points(points)


def _route_distance(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt(
            (float(b[0]) - float(a[0])) ** 2
            + (float(b[1]) - float(a[1])) ** 2
            + (float(b[2]) - float(a[2])) ** 2
        )
    return total


def _interpolate_route(
    points: Sequence[Sequence[float]], alpha: float
) -> tuple[tuple[float, float, float], float, int]:
    if not points:
        return (0.0, 0.0, 0.793), 0.0, 0
    if len(points) == 1:
        point = points[0]
        return (float(point[0]), float(point[1]), float(point[2])), 0.0, 0
    total = _route_distance(points)
    if total <= 0:
        point = points[-1]
        return (float(point[0]), float(point[1]), float(point[2])), 0.0, len(points) - 1
    remaining = max(0.0, min(1.0, alpha)) * total
    for segment_index, (a, b) in enumerate(zip(points, points[1:])):
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        segment_distance = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        if segment_distance <= 0:
            continue
        if remaining <= segment_distance:
            local_alpha = remaining / segment_distance
            x = ax + (bx - ax) * local_alpha
            y = ay + (by - ay) * local_alpha
            z = az + (bz - az) * local_alpha
            yaw = math.atan2(by - ay, bx - ax)
            return (x, y, z), yaw, segment_index
        remaining -= segment_distance
    a = points[-2]
    b = points[-1]
    yaw = math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))
    return (float(b[0]), float(b[1]), float(b[2])), yaw, len(points) - 2


def _yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _g1_preview_joint_addresses(model: Any, mujoco_module: Any) -> dict[str, int]:
    names = [
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_elbow_joint",
    ]
    addresses: dict[str, int] = {}
    for name in names:
        joint_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            address = int(model.jnt_qposadr[joint_id])
            addresses[name] = address
    return addresses


def _apply_preview_gait_pose(
    *,
    qpos: Any,
    base_qpos: Any,
    joint_addresses: Mapping[str, int],
    phase: float,
    moving: bool,
) -> None:
    if not moving:
        return

    def set_joint(name: str, delta: float) -> None:
        address = joint_addresses.get(name)
        if address is None or address >= len(qpos) or address >= len(base_qpos):
            return
        qpos[address] = base_qpos[address] + delta

    stride = math.sin(phase)
    counter = math.sin(phase + math.pi)
    knee_left = max(0.0, math.sin(phase + 0.45)) * 0.35
    knee_right = max(0.0, math.sin(phase + math.pi + 0.45)) * 0.35
    set_joint("left_hip_pitch_joint", 0.24 * stride)
    set_joint("right_hip_pitch_joint", 0.24 * counter)
    set_joint("left_knee_joint", knee_left)
    set_joint("right_knee_joint", knee_right)
    set_joint("left_ankle_pitch_joint", -0.10 * stride)
    set_joint("right_ankle_pitch_joint", -0.10 * counter)
    set_joint("left_shoulder_pitch_joint", -0.20 * stride)
    set_joint("right_shoulder_pitch_joint", -0.20 * counter)
    set_joint("left_elbow_joint", 0.08 * max(0.0, -stride))
    set_joint("right_elbow_joint", 0.08 * max(0.0, -counter))
    set_joint("waist_yaw_joint", 0.04 * math.sin(phase * 0.5))


def _set_preview_pose(
    *,
    data: Any,
    base_qpos: Any,
    root_qpos: int,
    pose: Sequence[float],
    yaw: float,
    joint_addresses: Mapping[str, int],
    phase: float,
    moving: bool,
) -> None:
    data.qpos[:] = base_qpos
    data.qvel[:] = 0
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    data.qpos[root_qpos : root_qpos + 7] = [x, y, z, *_yaw_quaternion(yaw)]
    _apply_preview_gait_pose(
        qpos=data.qpos,
        base_qpos=base_qpos,
        joint_addresses=joint_addresses,
        phase=phase,
        moving=moving,
    )


def _candidate_pose_specs(
    *,
    desired_pose: Sequence[float],
    previous_pose: Sequence[float] | None,
    yaw: float,
    previous_yaw: float | None = None,
    previous_phase: float | None = None,
    previous_moving: bool | None = None,
) -> list[dict[str, Any]]:
    x, y, z = float(desired_pose[0]), float(desired_pose[1]), float(desired_pose[2])
    normal = (-math.sin(yaw), math.cos(yaw))
    specs: list[dict[str, Any]] = [
        {"candidate_kind": "direct", "pose": (x, y, z), "lateral_offset_m": 0.0}
    ]
    for offset in (0.18, -0.18, 0.36, -0.36, 0.6, -0.6):
        specs.append(
            {
                "candidate_kind": "redirect",
                "pose": (x + normal[0] * offset, y + normal[1] * offset, z),
                "lateral_offset_m": offset,
            }
        )
    if previous_pose is not None:
        specs.append(
            {
                "candidate_kind": "stop",
                "pose": (
                    float(previous_pose[0]),
                    float(previous_pose[1]),
                    float(previous_pose[2]),
                ),
                "yaw": previous_yaw if previous_yaw is not None else yaw,
                "phase": previous_phase if previous_phase is not None else 0.0,
                "moving": bool(previous_moving),
                "lateral_offset_m": 0.0,
            }
        )
    else:
        for radius in (0.35, 0.7, 1.05, 1.4, 1.8):
            for angle_index in range(8):
                angle = yaw + angle_index * (math.pi / 4.0)
                specs.append(
                    {
                        "candidate_kind": "spawn_relocation",
                        "pose": (
                            x + math.cos(angle) * radius,
                            y + math.sin(angle) * radius,
                            z,
                        ),
                        "lateral_offset_m": None,
                        "relocation_radius_m": radius,
                    }
                )
    return specs


def _evaluate_preview_candidate(
    *,
    model: Any,
    data: Any,
    mujoco_module: Any,
    base_qpos: Any,
    root_qpos: int,
    joint_addresses: Mapping[str, int],
    candidate: Mapping[str, Any],
    yaw: float,
    phase: float,
    moving: bool,
) -> dict[str, Any]:
    pose = candidate["pose"]
    candidate_yaw = float(candidate.get("yaw", yaw))
    candidate_phase = float(candidate.get("phase", phase))
    candidate_moving = bool(candidate.get("moving", moving))
    _set_preview_pose(
        data=data,
        base_qpos=base_qpos,
        root_qpos=root_qpos,
        pose=pose,
        yaw=candidate_yaw,
        joint_addresses=joint_addresses,
        phase=candidate_phase,
        moving=candidate_moving,
    )
    mujoco_module.mj_forward(model, data)
    contacts = _contact_records(model, data, mujoco_module)
    scene_contact_count = _scene_collision_contact_count(contacts)
    return {
        "candidate_kind": _string(candidate.get("candidate_kind")),
        "pose": _rounded_pose(pose),
        "yaw": candidate_yaw,
        "phase": candidate_phase,
        "moving": candidate_moving,
        "lateral_offset_m": candidate.get("lateral_offset_m"),
        "relocation_radius_m": candidate.get("relocation_radius_m"),
        "contacts": contacts,
        "contact_count": len(contacts),
        "scene_collision_contact_count": scene_contact_count,
        "accepted": scene_contact_count == 0,
    }


def _render_episode_indexes(total_episodes: int, max_rendered_episodes: int) -> set[int]:
    if total_episodes <= 0 or max_rendered_episodes <= 0:
        return set()
    rendered_count = min(total_episodes, max_rendered_episodes)
    if rendered_count == total_episodes:
        return set(range(total_episodes))
    if rendered_count == 1:
        return {0}
    return {
        round(index * (total_episodes - 1) / float(rendered_count - 1))
        for index in range(rendered_count)
    }


def _camera_record(camera: str, frame_path: Path, step: int, mode: str) -> dict[str, Any]:
    return {"camera": camera, "camera_mode": mode, "step": step, "path": str(frame_path)}


def _camera_record_with_time(
    camera: str,
    frame_path: Path,
    step: int,
    mode: str,
    *,
    sim_time_s: float | None,
) -> dict[str, Any]:
    record = _camera_record(camera, frame_path, step, mode)
    record["sim_time_s"] = round(float(sim_time_s), 9) if sim_time_s is not None else None
    return record


def _frame_groups(frames: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups = {
        "overview": [],
        "sim_robot_follow_pov": [],
        "side": [],
    }
    for frame in frames:
        camera = _string(frame.get("camera"))
        path = _string(frame.get("path"))
        if camera in groups and path:
            groups[camera].append(path)
    return groups


def _frame_time_groups(frames: Sequence[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    groups: dict[str, list[float | None]] = {
        "overview": [],
        "sim_robot_follow_pov": [],
        "side": [],
    }
    for frame in frames:
        camera = _string(frame.get("camera"))
        if camera not in groups:
            continue
        time_value = _number(frame.get("sim_time_s"))
        groups[camera].append(round(float(time_value), 9) if time_value is not None else None)
    return groups


def _is_scene_collision_contact(record: Mapping[str, Any]) -> bool:
    if record.get("scene_collision_contact") is True:
        return True
    geom_names = record.get("geom_names")
    if not isinstance(geom_names, Sequence) or isinstance(geom_names, (str, bytes)):
        return False
    for raw_name in geom_names:
        name = _string(raw_name)
        if name == "blueprint_scene_collision" or name.startswith("blueprint_collision_proxy_"):
            return True
    return False


def _scene_collision_contact_count(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for record in records if _is_scene_collision_contact(record))


def _contact_records(model: Any, data: Any, mujoco_module: Any) -> list[dict[str, Any]]:
    contact_count = int(getattr(data, "ncon", 0) or 0)
    if contact_count <= 0:
        return []
    records: list[dict[str, Any]] = []
    for index in range(contact_count):
        try:
            contact = data.contact[index]
        except Exception:
            continue
        geom_ids = [int(getattr(contact, "geom1", -1)), int(getattr(contact, "geom2", -1))]
        geom_names: list[str] = []
        body_names: list[str] = []
        for geom_id in geom_ids:
            geom_name = ""
            body_name = ""
            try:
                geom_name = _string(
                    mujoco_module.mj_id2name(
                        model,
                        mujoco_module.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    )
                )
            except Exception:
                geom_name = ""
            try:
                body_id = int(model.geom_bodyid[geom_id])
                body_name = _string(
                    mujoco_module.mj_id2name(
                        model,
                        mujoco_module.mjtObj.mjOBJ_BODY,
                        body_id,
                    )
                )
            except Exception:
                body_name = ""
            geom_names.append(geom_name or f"geom_{geom_id}")
            body_names.append(body_name or "unknown_body")
        force = None
        try:
            import numpy as np

            force_vector = np.zeros(6, dtype=np.float64)
            mujoco_module.mj_contactForce(model, data, index, force_vector)
            force = [round(float(value), 6) for value in force_vector.tolist()]
        except Exception:
            force = None
        raw_position = getattr(contact, "pos", [])
        try:
            position_xyz = [round(float(value), 6) for value in list(raw_position)[:3]]
        except Exception:
            position_xyz = []
        scene_collision_contact = any(
            name == "blueprint_scene_collision" or name.startswith("blueprint_collision_proxy_")
            for name in geom_names
        )
        records.append(
            {
                "contact_index": index,
                "geom_ids": geom_ids,
                "geom_names": geom_names,
                "body_names": body_names,
                "distance": round(float(getattr(contact, "dist", 0.0) or 0.0), 9),
                "position_xyz": position_xyz,
                "contact_force_6d": force,
                "scene_collision_contact": scene_collision_contact,
                "reference_floor_contact": "blueprint_reference_floor" in geom_names,
            }
        )
    return records


def _collision_summary(
    contact_trace: Sequence[Mapping[str, Any]],
    *,
    collision_probe_trace: Sequence[Mapping[str, Any]] | None = None,
    collision_response_events: Sequence[Mapping[str, Any]] | None = None,
    collision_proxy_count: int = 0,
) -> dict[str, Any]:
    probe_trace = list(collision_probe_trace or [])
    response_events = list(collision_response_events or [])
    scene_contacts = [
        record for record in contact_trace if _is_scene_collision_contact(record)
    ]
    floor_contacts = [
        record for record in contact_trace if record.get("reference_floor_contact") is True
    ]
    rejected_scene_contacts = [
        record for record in probe_trace if _is_scene_collision_contact(record)
    ]
    scene_visual_mesh_collision_twin_enabled = collision_proxy_count == 0
    scene_collision_mesh_geom_enabled = collision_proxy_count == 0
    scene_collision_proxy_geoms_enabled = collision_proxy_count > 0
    proxy_collision_governed_preview_proven = (
        scene_collision_proxy_geoms_enabled and len(scene_contacts) == 0
    )
    visible_scene_collision_alignment_validated = scene_collision_mesh_geom_enabled
    collision_avoidance_validated = len(scene_contacts) == 0
    collision_dynamics_validated = (
        collision_avoidance_validated and visible_scene_collision_alignment_validated
    )
    physics_controlled_preview_proven = collision_dynamics_validated
    collision_response_validated = any(
        _string(event.get("event_type"))
        in {
            "candidate_rejected_scene_collision",
            "motion_redirected_by_collision_probe",
            "motion_stopped_by_collision_probe",
        }
        for event in response_events
    )
    return {
        "collision_geometry_loaded": True,
        "scene_visual_mesh_collision_twin_enabled": scene_visual_mesh_collision_twin_enabled,
        "scene_visual_mesh_collisions_enabled": False,
        "scene_collision_mesh_geom_enabled": scene_collision_mesh_geom_enabled,
        "scene_collision_proxy_geoms_enabled": scene_collision_proxy_geoms_enabled,
        "scene_collision_proxy_geom_count": collision_proxy_count,
        "visible_scene_collision_alignment_validated": (
            visible_scene_collision_alignment_validated
        ),
        "proxy_collision_model_used": scene_collision_proxy_geoms_enabled,
        "proxy_collision_governed_preview_proven": (
            proxy_collision_governed_preview_proven
        ),
        "reference_floor_collisions_enabled": True,
        "contact_detection_sampled": True,
        "contact_sample_count": len(contact_trace) + len(probe_trace),
        "committed_contact_sample_count": len(contact_trace),
        "collision_probe_contact_sample_count": len(probe_trace),
        "robot_scene_contact_event_count": len(scene_contacts),
        "robot_floor_contact_event_count": len(floor_contacts),
        "rejected_scene_collision_probe_count": len(rejected_scene_contacts),
        "collision_response_event_count": len(response_events),
        "sample_contacts": [dict(record) for record in contact_trace[:20]],
        "sample_rejected_collision_contacts": [
            dict(record) for record in rejected_scene_contacts[:20]
        ],
        "sample_collision_response_events": [dict(event) for event in response_events[:20]],
        "collision_dynamics_validated": collision_dynamics_validated,
        "collision_avoidance_validated": collision_avoidance_validated,
        "collision_response_validated": collision_response_validated,
        "physics_controlled_preview_proven": physics_controlled_preview_proven,
        "validation_boundary": (
            "The preview controller probes MuJoCo contact before committing each root pose. "
            "A proxy-only run can prove collision-governed preview motion against generated "
            "obstacle proxies, but it does not validate that the robot avoided every visible "
            "mesh surface unless visible_scene_collision_alignment_validated is true. This is "
            "still not a balanced Unitree locomotion policy."
        ),
    }


def _blank_scene_checks(frame_paths: Sequence[str]) -> dict[str, Any]:
    if not frame_paths:
        return {
            "status": "not_applicable",
            "frame_count": 0,
            "all_frames_nonblank": False,
            "checks": [],
        }
    try:
        from PIL import Image, ImageStat
    except Exception as exc:  # pragma: no cover - dependency guard.
        return {
            "status": "not_checked",
            "reason": f"pillow_unavailable:{type(exc).__name__}",
            "frame_count": len(frame_paths),
            "all_frames_nonblank": False,
            "checks": [],
        }

    checks: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        path = Path(frame_path)
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image)
        extrema = image.getextrema()
        sample_width = min(160, image.width)
        sample_height = min(90, image.height)
        sampled = image.resize((sample_width, sample_height))
        colors = sampled.getcolors(maxcolors=sample_width * sample_height)
        sampled_unique_colors = len(colors or [])
        channel_ranges = [high - low for low, high in extrema]
        mean = [round(value, 3) for value in stat.mean]
        nonblank = bool(
            sampled_unique_colors > 16
            and max(channel_ranges) > 24
            and not all(value >= 248 for value in stat.mean)
        )
        checks.append(
            {
                "path": str(path),
                "size": [image.width, image.height],
                "mean_rgb": mean,
                "channel_ranges": channel_ranges,
                "sampled_unique_colors": sampled_unique_colors,
                "nonblank": nonblank,
            }
        )
    return {
        "status": "checked",
        "frame_count": len(checks),
        "all_frames_nonblank": all(check["nonblank"] for check in checks),
        "checks": checks,
    }


def _rendered_array_nonblank(frame: Any) -> bool:
    return _rendered_array_scene_score(frame) > 0


def _rendered_array_scene_score(frame: Any) -> float:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a runtime dependency.
        return 1.0
    array = np.asarray(frame)
    if array.ndim < 3 or array.size == 0:
        return 0.0
    rgb = array[:, :, :3]
    flat = rgb.reshape(-1, 3)
    channel_ranges = flat.max(axis=0) - flat.min(axis=0)
    mean = flat.mean(axis=0)
    sample_stride_y = max(1, rgb.shape[0] // 90)
    sample_stride_x = max(1, rgb.shape[1] // 160)
    sample = rgb[::sample_stride_y, ::sample_stride_x, :].reshape(-1, 3)
    sampled_unique_colors = np.unique(sample, axis=0).shape[0]
    if sampled_unique_colors <= 16 or int(channel_ranges.max()) <= 24 or bool((mean >= 248).all()):
        return 0.0
    return float(sampled_unique_colors) + float(channel_ranges.max()) + float(flat.std())


def _write_frame_video(
    *,
    camera: str,
    frame_paths: Sequence[str],
    frame_times_s: Sequence[float | None],
    output_root: Path,
    fallback_frame_duration_s: float,
) -> dict[str, Any]:
    if len(frame_paths) < 2:
        return {
            "status": "not_generated",
            "reason": "requires_at_least_two_frames",
            "frame_count": len(frame_paths),
            "realtime_timing_from_sim_time": False,
        }
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "status": "not_generated",
            "reason": "ffmpeg_unavailable",
            "frame_count": len(frame_paths),
            "realtime_timing_from_sim_time": False,
        }
    concat_path = output_root / f"{camera}_video_frames.txt"
    video_path = output_root / f"{camera}.mp4"
    clean_times = [
        float(value) if value is not None else None
        for value in list(frame_times_s)[: len(frame_paths)]
    ]
    if len(clean_times) < len(frame_paths):
        clean_times.extend([None] * (len(frame_paths) - len(clean_times)))
    fallback_duration = max(1.0 / 240.0, float(fallback_frame_duration_s or 0.0) or 1.0 / 30.0)
    durations: list[float] = []
    for index in range(len(frame_paths)):
        current = clean_times[index]
        next_time = clean_times[index + 1] if index + 1 < len(clean_times) else None
        if current is not None and next_time is not None and next_time > current:
            durations.append(max(1.0 / 240.0, next_time - current))
        elif durations:
            durations.append(durations[-1])
        else:
            durations.append(fallback_duration)
    simulated_duration = sum(durations)
    realtime_timing = all(value is not None for value in clean_times[:2])
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
        "format=yuv420p",
        "-fps_mode",
        "vfr",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        return {
            "status": "not_generated",
            "reason": "ffmpeg_failed",
            "frame_count": len(frame_paths),
            "stderr_tail": completed.stderr[-1000:],
            "realtime_timing_from_sim_time": realtime_timing,
            "simulated_duration_s": round(simulated_duration, 9),
        }
    return {
        "status": "complete",
        "path": str(video_path),
        "concat_list_path": str(concat_path),
        "frame_count": len(frame_paths),
        "realtime_timing_from_sim_time": realtime_timing,
        "simulated_duration_s": round(simulated_duration, 9),
        "frame_durations_s": [round(value, 9) for value in durations],
        "playback_speed": "realtime_sim_time",
        "fps_filter_applied": False,
        "source_frames": list(frame_paths),
    }


def _visual_artifact_summary(
    *,
    frames: Sequence[Mapping[str, Any]],
    output_root: Path,
    mesh_info: Mapping[str, Any],
    model_timestep_s: float,
) -> dict[str, Any]:
    groups = _frame_groups(frames)
    time_groups = _frame_time_groups(frames)
    fallback_frame_duration_s = max(1.0 / 240.0, float(model_timestep_s or 0.0) or 1.0 / 30.0)
    videos = {
        "overview_video": _write_frame_video(
            camera="overview",
            frame_paths=groups["overview"],
            frame_times_s=time_groups["overview"],
            output_root=output_root,
            fallback_frame_duration_s=fallback_frame_duration_s,
        ),
        "robot_pov_video": _write_frame_video(
            camera="sim_robot_follow_pov",
            frame_paths=groups["sim_robot_follow_pov"],
            frame_times_s=time_groups["sim_robot_follow_pov"],
            output_root=output_root,
            fallback_frame_duration_s=fallback_frame_duration_s,
        ),
        "side_video": _write_frame_video(
            camera="side",
            frame_paths=groups["side"],
            frame_times_s=time_groups["side"],
            output_root=output_root,
            fallback_frame_duration_s=fallback_frame_duration_s,
        ),
    }
    all_frame_paths = [path for paths in groups.values() for path in paths]
    obj_summary = dict(mesh_info.get("obj_vertex_color_summary") or {})
    glb_summary = dict(mesh_info.get("visual_asset_summary") or {})
    texture_material_evidence = {
        "status": "materialized_vertex_color_scene_evidence_present"
        if obj_summary.get("has_vertex_rgb")
        else "plain_geometry_material_only",
        "source_scene_glb": mesh_info.get("source_glb"),
        "converted_scene_obj": mesh_info.get("converted_obj"),
        "glb_materials_count": glb_summary.get("materials_count"),
        "glb_textures_count": glb_summary.get("textures_count"),
        "glb_images_count": glb_summary.get("images_count"),
        "glb_has_vertex_colors": glb_summary.get("has_vertex_colors"),
        "obj_has_vertex_rgb": obj_summary.get("has_vertex_rgb"),
        "obj_vertex_rgb_fraction": obj_summary.get("vertex_rgb_fraction"),
        "white_scene_success_allowed": False,
        "fidelity_boundary": mesh_info.get("mujoco_visual_fidelity_boundary"),
    }
    blank_scene_checks = _blank_scene_checks(all_frame_paths)
    limitations = []
    if videos["overview_video"]["status"] != "complete":
        limitations.append(f"overview_video:{videos['overview_video'].get('reason')}")
    if videos["robot_pov_video"]["status"] != "complete":
        limitations.append(f"robot_pov_video:{videos['robot_pov_video'].get('reason')}")
    if videos["side_video"]["status"] != "complete":
        limitations.append(f"side_video:{videos['side_video'].get('reason')}")
    if texture_material_evidence["status"] != "materialized_vertex_color_scene_evidence_present":
        limitations.append("texture_material_evidence:vertex_rgb_not_detected")
    if (
        blank_scene_checks["status"] == "checked"
        and not blank_scene_checks["all_frames_nonblank"]
    ):
        limitations.append("blank_scene_checks:one_or_more_frames_blank")
    return {
        "status": "complete" if all_frame_paths else "not_recorded",
        "overview_frames": groups["overview"],
        "robot_pov_frames": groups["sim_robot_follow_pov"],
        "side_frames": groups["side"],
        **videos,
        "blank_scene_checks": blank_scene_checks,
        "texture_material_evidence": texture_material_evidence,
        "limitations": limitations,
    }


def _render_capture_steps(bounded_steps: int, *, max_rendered_steps: int = 24) -> set[int]:
    rendered_step_count = min(max(1, bounded_steps), max(1, max_rendered_steps))
    if rendered_step_count <= 1:
        return {0}
    return {
        round(index * (bounded_steps - 1) / float(rendered_step_count - 1))
        for index in range(rendered_step_count)
    }


def run_mujoco_g1_simulator_command(
    *,
    capture_root: str | Path,
    g1_model_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    simulator_output_path: str | Path | None = None,
    scenario_eval_matrix_path: str | Path | None = None,
    steps: int = 32,
    duration_seconds: float | None = None,
    render_frames: bool = True,
    render_every_step: bool = False,
    max_rendered_episodes: int = 3,
    max_rendered_steps: int = 24,
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

    matrix_path = (
        Path(scenario_eval_matrix_path).resolve()
        if scenario_eval_matrix_path
        else Path(os.environ["BLUEPRINT_SCENARIO_EVAL_MATRIX"]).resolve()
        if os.environ.get("BLUEPRINT_SCENARIO_EVAL_MATRIX")
        else None
    )
    matrix_runs, matrix_summary = _matrix_runs(matrix_path)
    if not matrix_runs:
        raise RuntimeError(
            "scenario_eval_matrix contains no executable runs; cannot produce MuJoCo coverage"
        )
    matrix_blockers = []
    if matrix_summary.get("missing_scenario_eval_run_id_indexes"):
        matrix_blockers.append(
            "scenario_eval_matrix_missing_scenario_eval_run_id"
        )
    if matrix_summary.get("duplicate_scenario_eval_run_ids"):
        matrix_blockers.append("scenario_eval_matrix_duplicate_scenario_eval_run_id")
    if matrix_summary.get("matrix_declared_count_matches_rows") is False:
        matrix_blockers.append("scenario_eval_matrix_declared_count_mismatch")
    if matrix_blockers:
        raise RuntimeError(
            "scenario_eval_matrix is not executable by MuJoCo command: "
            + ",".join(matrix_blockers)
        )

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
    collision_proxies = mesh_info.get("collision_proxy_geoms")
    _write_mjcf_wrapper(
        scene_obj,
        generated_g1_xml,
        wrapper_xml,
        collision_proxies=collision_proxies if isinstance(collision_proxies, Sequence) else None,
    )

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
    model_timestep = float(getattr(getattr(model, "opt", None), "timestep", 0.0) or 0.0)

    renderer = None
    if render_frames:
        from PIL import Image

        renderer = mujoco.Renderer(model, height=360, width=640)
    else:
        Image = None  # type: ignore[assignment]

    requested_duration_seconds = _number(duration_seconds)
    if requested_duration_seconds is not None and requested_duration_seconds > 0 and model_timestep:
        bounded_steps = max(1, int(round(requested_duration_seconds / model_timestep)))
        step_count_source = "duration_seconds_and_model_timestep"
    else:
        bounded_steps = max(1, int(steps))
        step_count_source = "explicit_steps"
        requested_duration_seconds = (
            round(bounded_steps * model_timestep, 9) if model_timestep else None
        )
    rendered_episode_indexes = _render_episode_indexes(
        total_episodes=len(matrix_runs),
        max_rendered_episodes=max(0, int(max_rendered_episodes)),
    )
    frames: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    spawn_records: list[dict[str, Any]] = []
    all_policy_records: list[dict[str, Any]] = []
    full_contact_trace: list[dict[str, Any]] = []
    full_collision_probe_trace: list[dict[str, Any]] = []
    full_collision_response_events: list[dict[str, Any]] = []
    collision_proxy_count = len(collision_proxies) if isinstance(collision_proxies, Sequence) else 0
    robot_camera = mujoco.MjvCamera()
    robot_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    side_camera = mujoco.MjvCamera()
    side_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    preview_joint_addresses = _g1_preview_joint_addresses(model, mujoco)
    for episode_index, matrix_run in enumerate(matrix_runs):
        navigation = _episode_navigation_spec(
            run=matrix_run,
            mesh_info=mesh_info,
            index=episode_index + 1,
        )
        start = navigation["start"]
        target = navigation["target"]
        route_points, route_strategy = _route_waypoints_from_run(
            run=matrix_run,
            start=start,
            target=target,
        )
        route_distance = _route_distance(route_points)
        scenario_eval_run_id = _string(matrix_run.get("scenario_eval_run_id"))
        attempt_id = f"mujoco_g1_{_safe_id(scenario_eval_run_id or episode_index + 1)}"
        episode_id = _string(matrix_run.get("episode_id")) or f"mujoco_g1_episode_{episode_index + 1:04d}"
        should_render_episode = renderer is not None and episode_index in rendered_episode_indexes
        if should_render_episode:
            capture_steps = (
                set(range(bounded_steps))
                if render_every_step
                else _render_capture_steps(bounded_steps, max_rendered_steps=max_rendered_steps)
            )
        else:
            capture_steps = set()
        actions: list[dict[str, Any]] = []
        episode_frames: list[dict[str, Any]] = []
        episode_contact_trace: list[dict[str, Any]] = []
        episode_collision_probe_trace: list[dict[str, Any]] = []
        episode_collision_response_events: list[dict[str, Any]] = []
        episode_side_camera_selection: dict[str, Any] | None = None
        data.time = 0
        accepted_pose: tuple[float, float, float] | None = None
        accepted_yaw = 0.0
        accepted_phase = 0.0
        accepted_moving = False
        for step in range(bounded_steps):
            alpha = 0.0 if bounded_steps <= 1 else step / float(bounded_steps - 1)
            desired_pose, yaw, route_segment_index = _interpolate_route(route_points, alpha)
            phase = alpha * max(1.0, route_distance) * math.pi * 2.0
            candidate_results = []
            for candidate in _candidate_pose_specs(
                desired_pose=desired_pose,
                previous_pose=accepted_pose,
                yaw=yaw if accepted_pose is None else accepted_yaw,
                previous_yaw=accepted_yaw,
                previous_phase=accepted_phase,
                previous_moving=accepted_moving,
            ):
                candidate_result = _evaluate_preview_candidate(
                    model=model,
                    data=data,
                    mujoco_module=mujoco,
                    base_qpos=base_qpos,
                    root_qpos=root_qpos,
                    joint_addresses=preview_joint_addresses,
                    candidate=candidate,
                    yaw=yaw,
                    phase=phase,
                    moving=route_distance > 0.05,
                )
                candidate_results.append(candidate_result)
                if candidate_result["accepted"]:
                    break
            selected_candidate = next(
                (candidate for candidate in candidate_results if candidate["accepted"]),
                candidate_results[-1],
            )
            rejected_candidates = [
                candidate
                for candidate in candidate_results
                if _scene_collision_contact_count(candidate["contacts"]) > 0
            ]
            for candidate in rejected_candidates:
                for contact_record in candidate["contacts"]:
                    enriched_probe = {
                        **contact_record,
                        "attempt_id": attempt_id,
                        "episode_id": episode_id,
                        "scenario_eval_run_id": scenario_eval_run_id or None,
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                        "candidate_kind": candidate["candidate_kind"],
                        "candidate_pose": list(candidate["pose"]),
                        "lateral_offset_m": candidate.get("lateral_offset_m"),
                    }
                    episode_collision_probe_trace.append(enriched_probe)
                    full_collision_probe_trace.append(enriched_probe)
                event = {
                    "event_type": "candidate_rejected_scene_collision",
                    "step": step,
                    "sim_time_s": round(float(data.time), 9),
                    "candidate_kind": candidate["candidate_kind"],
                    "candidate_pose": list(candidate["pose"]),
                    "scene_collision_contact_count": candidate[
                        "scene_collision_contact_count"
                    ],
                    "lateral_offset_m": candidate.get("lateral_offset_m"),
                }
                episode_collision_response_events.append(event)
                full_collision_response_events.append(event)
            if selected_candidate["accepted"] is False:
                raise RuntimeError(
                    "MuJoCo collision-governed preview could not find a non-colliding "
                    f"pose for {attempt_id} at step {step}; refusing to render a "
                    "pass-through video"
                )
            selected_pose = selected_candidate["pose"]
            selected_yaw = float(selected_candidate.get("yaw", yaw))
            selected_phase = float(selected_candidate.get("phase", phase))
            selected_moving = bool(selected_candidate.get("moving", route_distance > 0.05))
            accepted_pose = selected_pose
            accepted_yaw = selected_yaw
            accepted_phase = selected_phase
            accepted_moving = selected_moving
            if selected_candidate["candidate_kind"] == "direct":
                policy_action = "accepted_direct_collision_checked_motion"
            elif selected_candidate["candidate_kind"] in {"redirect", "spawn_relocation"}:
                policy_action = "redirected_by_collision_probe"
                event = {
                    "event_type": "motion_redirected_by_collision_probe",
                    "step": step,
                    "sim_time_s": round(float(data.time), 9),
                    "accepted_pose": list(selected_pose),
                    "lateral_offset_m": selected_candidate.get("lateral_offset_m"),
                    "relocation_radius_m": selected_candidate.get("relocation_radius_m"),
                    "candidate_kind": selected_candidate["candidate_kind"],
                    "rejected_candidate_count": len(rejected_candidates),
                }
                episode_collision_response_events.append(event)
                full_collision_response_events.append(event)
            else:
                policy_action = "stopped_by_collision_probe"
                event = {
                    "event_type": "motion_stopped_by_collision_probe",
                    "step": step,
                    "sim_time_s": round(float(data.time), 9),
                    "accepted_pose": list(selected_pose),
                    "rejected_candidate_count": len(rejected_candidates),
                }
                episode_collision_response_events.append(event)
                full_collision_response_events.append(event)
            _set_preview_pose(
                data=data,
                base_qpos=base_qpos,
                root_qpos=root_qpos,
                pose=selected_pose,
                yaw=selected_yaw,
                joint_addresses=preview_joint_addresses,
                phase=selected_phase,
                moving=selected_moving,
            )
            mujoco.mj_forward(model, data)
            step_contacts = _contact_records(model, data, mujoco)
            committed_scene_contact_count = _scene_collision_contact_count(step_contacts)
            if committed_scene_contact_count:
                raise RuntimeError(
                    "collision-governed preview committed a scene-colliding pose; "
                    f"attempt={attempt_id} step={step} contacts={committed_scene_contact_count}"
                )
            for contact_record in step_contacts:
                enriched_contact = {
                    **contact_record,
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
                    "step": step,
                    "sim_time_s": round(float(data.time), 9),
                }
                episode_contact_trace.append(enriched_contact)
                full_contact_trace.append(enriched_contact)
            x, y, z = selected_pose
            actions.append(
                {
                    "step": step,
                    "sim_time_s": round(float(data.time), 9),
                    "root_position": [round(x, 6), round(y, 6), round(z, 6)],
                    "desired_root_position": [
                        round(float(desired_pose[0]), 6),
                        round(float(desired_pose[1]), 6),
                        round(float(desired_pose[2]), 6),
                    ],
                    "root_yaw_radians": round(selected_yaw, 6),
                    "target": list(target),
                    "route_segment_index": route_segment_index,
                    "contact_count": len(step_contacts),
                    "scene_collision_contact_count": committed_scene_contact_count,
                    "collision_probe_candidate_count": len(candidate_results),
                    "rejected_collision_probe_count": len(rejected_candidates),
                    "policy_action": policy_action,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
                    "deterministic_seed": navigation["seed"],
                }
            )
            if renderer is not None and step in capture_steps:
                renderer.update_scene(data, camera="overview")
                if len(matrix_runs) == 1:
                    overview_path = frames_dir / f"overview_{step:04d}.png"
                else:
                    overview_path = frames_dir / f"{attempt_id}_overview_{step:04d}.png"
                Image.fromarray(renderer.render()).save(overview_path)
                overview_record = _camera_record_with_time(
                    "overview",
                    overview_path,
                    step,
                    "named_fixed_overview_camera",
                    sim_time_s=float(data.time),
                )
                overview_record.update(
                    {
                        "attempt_id": attempt_id,
                        "episode_id": episode_id,
                        "scenario_eval_run_id": scenario_eval_run_id or None,
                    }
                )
                frames.append(overview_record)
                episode_frames.append(overview_record)

                robot_camera.lookat[:] = [x, y, z + 0.75]
                robot_camera.distance = 2.15
                robot_camera.azimuth = math.degrees(selected_yaw) + 180.0
                robot_camera.elevation = -14
                renderer.update_scene(data, camera=robot_camera)
                robot_render = renderer.render()
                robot_camera_selected = {
                    "azimuth": robot_camera.azimuth,
                    "distance": robot_camera.distance,
                    "elevation": robot_camera.elevation,
                    "fallback_used": False,
                    "scene_detail_score": round(_rendered_array_scene_score(robot_render), 3),
                }
                if robot_camera_selected["scene_detail_score"] <= 0:
                    robot_render = None
                    best_robot_score = -1.0
                    for option_index, (azimuth_offset, distance, elevation) in enumerate(
                        (
                            (180.0, 2.15, -14),
                            (135.0, 2.6, -16),
                            (225.0, 2.6, -16),
                            (90.0, 3.0, -18),
                            (270.0, 3.0, -18),
                            (0.0, 3.4, -20),
                        )
                    ):
                        robot_camera.distance = distance
                        robot_camera.azimuth = math.degrees(selected_yaw) + azimuth_offset
                        robot_camera.elevation = elevation
                        renderer.update_scene(data, camera=robot_camera)
                        candidate = renderer.render()
                        candidate_score = _rendered_array_scene_score(candidate)
                        if robot_render is None or candidate_score > best_robot_score:
                            robot_render = candidate
                            best_robot_score = candidate_score
                            robot_camera_selected = {
                                "azimuth": robot_camera.azimuth,
                                "distance": distance,
                                "elevation": elevation,
                                "fallback_used": True,
                                "fallback_reason": "route_follow_camera_frame_blank",
                                "fallback_option_index": option_index,
                                "scene_detail_score": round(candidate_score, 3),
                            }
                if len(matrix_runs) == 1:
                    robot_path = frames_dir / f"sim_robot_follow_pov_{step:04d}.png"
                else:
                    robot_path = frames_dir / f"{attempt_id}_sim_robot_follow_pov_{step:04d}.png"
                Image.fromarray(robot_render).save(robot_path)
                robot_record = _camera_record_with_time(
                    "sim_robot_follow_pov",
                    robot_path,
                    step,
                    "virtual_free_camera_following_g1_root_not_physical_robot_sensor",
                    sim_time_s=float(data.time),
                )
                robot_record.update(
                    {
                        "attempt_id": attempt_id,
                        "episode_id": episode_id,
                        "scenario_eval_run_id": scenario_eval_run_id or None,
                        "robot_camera_selected": robot_camera_selected,
                    }
                )
                frames.append(robot_record)
                episode_frames.append(robot_record)

                side_camera.lookat[:] = [x, y, z + 0.45]
                if episode_side_camera_selection is None:
                    side_render = None
                    side_selected = {
                        "azimuth": 0,
                        "distance": 2.35,
                        "elevation": -10,
                        "fallback_used": False,
                        "scene_detail_score": 0.0,
                    }
                    best_side_score = -1.0
                    for option_index, (azimuth, distance, elevation) in enumerate(
                        (
                            (0, 2.35, -10),
                            (45, 2.8, -12),
                            (-45, 2.8, -12),
                            (90, 3.2, -14),
                            (-90, 3.2, -14),
                            (180, 3.8, -16),
                        )
                    ):
                        side_camera.distance = distance
                        side_camera.azimuth = azimuth
                        side_camera.elevation = elevation
                        renderer.update_scene(data, camera=side_camera)
                        candidate = renderer.render()
                        candidate_score = _rendered_array_scene_score(candidate)
                        if side_render is None or candidate_score > best_side_score:
                            side_render = candidate
                            best_side_score = candidate_score
                            side_selected = {
                                "azimuth": azimuth,
                                "distance": distance,
                                "elevation": elevation,
                                "fallback_used": option_index > 0,
                                "scene_detail_score": round(candidate_score, 3),
                            }
                    episode_side_camera_selection = side_selected
                else:
                    side_selected = episode_side_camera_selection
                    side_camera.distance = side_selected["distance"]
                    side_camera.azimuth = side_selected["azimuth"]
                    side_camera.elevation = side_selected["elevation"]
                    renderer.update_scene(data, camera=side_camera)
                    side_render = renderer.render()
                    if _rendered_array_scene_score(side_render) <= 0:
                        episode_side_camera_selection = None
                        side_render = None
                        side_selected = {
                            "azimuth": 0,
                            "distance": 2.35,
                            "elevation": -10,
                            "fallback_used": True,
                            "scene_detail_score": 0.0,
                        }
                        best_side_score = -1.0
                        for option_index, (azimuth, distance, elevation) in enumerate(
                            (
                                (0, 2.35, -10),
                                (45, 2.8, -12),
                                (-45, 2.8, -12),
                                (90, 3.2, -14),
                                (-90, 3.2, -14),
                                (180, 3.8, -16),
                            )
                        ):
                            side_camera.distance = distance
                            side_camera.azimuth = azimuth
                            side_camera.elevation = elevation
                            renderer.update_scene(data, camera=side_camera)
                            candidate = renderer.render()
                            candidate_score = _rendered_array_scene_score(candidate)
                            if side_render is None or candidate_score > best_side_score:
                                side_render = candidate
                                best_side_score = candidate_score
                                side_selected = {
                                    "azimuth": azimuth,
                                    "distance": distance,
                                    "elevation": elevation,
                                    "fallback_used": True,
                                    "fallback_reason": "reused_side_camera_frame_blank",
                                    "fallback_option_index": option_index,
                                    "scene_detail_score": round(candidate_score, 3),
                                }
                        episode_side_camera_selection = side_selected
                if len(matrix_runs) == 1:
                    side_path = frames_dir / f"side_{step:04d}.png"
                else:
                    side_path = frames_dir / f"{attempt_id}_side_{step:04d}.png"
                Image.fromarray(side_render).save(side_path)
                side_record = _camera_record_with_time(
                    "side",
                    side_path,
                    step,
                    "virtual_side_profile_camera_tracking_g1_root",
                    sim_time_s=float(data.time),
                )
                side_record.update(
                    {
                        "attempt_id": attempt_id,
                        "episode_id": episode_id,
                        "scenario_eval_run_id": scenario_eval_run_id or None,
                        "side_camera_selected": side_selected,
                    }
                )
                frames.append(side_record)
                episode_frames.append(side_record)
            mujoco.mj_step(model, data)
        scenario_id = _string(matrix_run.get("scenario_id")) or "scenario_walk_to_target"
        task_id = _string(matrix_run.get("task_id")) or "walk_to_target"
        rendered_frame_paths = [frame["path"] for frame in episode_frames]
        episode_collision_summary = _collision_summary(
            episode_contact_trace,
            collision_probe_trace=episode_collision_probe_trace,
            collision_response_events=episode_collision_response_events,
            collision_proxy_count=collision_proxy_count,
        )
        scene_contact_count = int(episode_collision_summary["robot_scene_contact_event_count"])
        collision_free_preview = scene_contact_count == 0
        attempt_status = (
            "completed_collision_governed"
            if collision_free_preview
            else "blocked_collision_overlap_detected"
        )
        attempt_artifact_paths = {
            "scene_trace": str(output_root / "scene_load_trace.json"),
            "spawn_trace": str(output_root / "spawn_trace.json"),
            "policy_trace": str(output_root / "policy_execution_trace.json"),
            "sim_robot_pov_evidence": str(output_root / "sim_robot_pov_evidence_manifest.json"),
            "frames": rendered_frame_paths,
        }
        attempt = {
            "attempt_id": attempt_id,
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "scenario_run_id": _string(matrix_run.get("scenario_run_id"))
            or f"mujoco_g1_scenario_run_{episode_index + 1:04d}",
            "scenario_eval_run_id": scenario_eval_run_id or None,
            "scenario_variation_instance_id": _string(
                matrix_run.get("scenario_variation_instance_id")
            )
            or None,
            "variation_name": _string(matrix_run.get("variation_name")) or None,
            "task_id": task_id,
            "policy_id": _string(matrix_run.get("policy_id")) or POLICY_ID,
            "status": attempt_status,
            "success": collision_free_preview,
            "success_semantics": (
                "scene_collision_contacts_govern_preview_motion_but_not_locomotion_task_success"
                if collision_free_preview
                else "scene_collision_contacts_detected_preview_blocked"
            ),
            "deterministic_seed": navigation["seed"],
            "spawn_pose": list(start),
            "target_pose": list(target),
            "route_source": navigation["route_source"],
            "route_strategy": route_strategy,
            "route_waypoints": [list(point) for point in route_points],
            "locomotion_controller_integrated": False,
            "walking_motion_proven": False,
            "walking_style_preview_animation_rendered": bool(preview_joint_addresses),
            "training_grade_policy_rollout_proven": False,
            "metrics": {
                "cycle_time_seconds": round(bounded_steps * model_timestep, 6)
                if model_timestep
                else None,
                "intervention_count": 0,
                "unsafe_proximity_event_count": 0,
                "collision_risk_event_count": scene_contact_count,
                "collision_risk_status": (
                    "collision_governed_motion_contact_checked"
                ),
                "contact_sample_count": episode_collision_summary["contact_sample_count"],
                "committed_contact_sample_count": episode_collision_summary[
                    "committed_contact_sample_count"
                ],
                "collision_probe_contact_sample_count": episode_collision_summary[
                    "collision_probe_contact_sample_count"
                ],
                "robot_scene_contact_event_count": scene_contact_count,
                "robot_floor_contact_event_count": episode_collision_summary[
                    "robot_floor_contact_event_count"
                ],
                "rejected_scene_collision_probe_count": episode_collision_summary[
                    "rejected_scene_collision_probe_count"
                ],
                "collision_response_event_count": episode_collision_summary[
                    "collision_response_event_count"
                ],
                "object_drop_count": 0,
                "wrong_object_count": 0,
                "timeout_count": 0,
                "simulated_step_count": bounded_steps,
                "rendered_step_count": len(capture_steps),
                "rendered_frame_count": len(rendered_frame_paths),
                "route_distance_m": round(route_distance, 6),
                "direct_start_to_target_distance_m": navigation["route_distance_m"],
                "start_pose_xyz": list(start),
                "target_pose_xyz": list(target),
                "deterministic_seed": navigation["seed"],
            },
            "actions": actions,
            "contact_trace": episode_contact_trace[:100],
            "collision_probe_trace": episode_collision_probe_trace[:100],
            "collision_response_events": episode_collision_response_events[:100],
            "collision_summary": episode_collision_summary,
            "safety_events": [
                {
                    "event_type": "scene_collision_candidate_rejected",
                    "event_count": episode_collision_summary[
                        "rejected_scene_collision_probe_count"
                    ],
                    "severity": "handled_by_collision_governed_preview",
                }
            ]
            if episode_collision_summary["rejected_scene_collision_probe_count"]
            else [],
            "artifact_paths": attempt_artifact_paths,
            "claim_boundary": (
                "mujoco_contact_governed_preview_harness_not_balanced_humanoid_locomotion"
            ),
        }
        attempts.append(attempt)
        spawn_records.append(
            {
                "attempt_id": attempt_id,
                "episode_id": episode_id,
                "scenario_eval_run_id": scenario_eval_run_id or None,
                "scenario_variation_instance_id": attempt["scenario_variation_instance_id"],
                "variation_name": attempt["variation_name"],
                "task_id": task_id,
                "scenario_id": scenario_id,
                "spawn_pose": list(start),
                "target_pose": list(target),
                "deterministic_seed": navigation["seed"],
                "route_source": navigation["route_source"],
                "route_strategy": route_strategy,
                "route_waypoints": [list(point) for point in route_points],
                "route_distance_m": round(route_distance, 6),
                "direct_start_to_target_distance_m": navigation["route_distance_m"],
                "scene_route_frame": navigation["scene_route_frame"],
            }
        )
        all_policy_records.append(
            {
                "attempt_id": attempt_id,
                "episode_id": episode_id,
                "scenario_eval_run_id": scenario_eval_run_id or None,
                "scenario_variation_instance_id": attempt["scenario_variation_instance_id"],
                "variation_name": attempt["variation_name"],
                "task_id": task_id,
                "scenario_id": scenario_id,
                "policy_id": attempt["policy_id"],
                "start_pose": list(start),
                "target_pose": list(target),
                "final_pose": actions[-1]["root_position"],
                "step_count": bounded_steps,
                "route_strategy": route_strategy,
                "route_waypoints": [list(point) for point in route_points],
                "route_distance_m": round(route_distance, 6),
                "rendered_step_count": len(capture_steps),
                "deterministic_seed": navigation["seed"],
                "actions": actions,
                "collision_probe_trace": episode_collision_probe_trace[:100],
                "collision_response_events": episode_collision_response_events[:100],
                "collision_summary": episode_collision_summary,
            }
        )
    if renderer is not None:
        renderer.close()
    visual_artifacts = _visual_artifact_summary(
        frames=frames,
        output_root=output_root,
        mesh_info=mesh_info,
        model_timestep_s=model_timestep,
    )
    collision_summary = _collision_summary(
        full_contact_trace,
        collision_probe_trace=full_collision_probe_trace,
        collision_response_events=full_collision_response_events,
        collision_proxy_count=collision_proxy_count,
    )
    blocked_collision_attempt_count = sum(
        1 for attempt in attempts if _string(attempt.get("status")) == "blocked_collision_overlap_detected"
    )
    collision_free_preview = blocked_collision_attempt_count == 0
    collision_dynamics_validated = bool(collision_summary["collision_dynamics_validated"])
    collision_avoidance_validated = bool(collision_summary["collision_avoidance_validated"])
    physics_controlled_preview_proven = bool(
        collision_summary["physics_controlled_preview_proven"]
    )
    robot_team_handoff_blockers = [
        "balanced_walking_controller_not_integrated_in_default_mujoco_preview",
        "training_grade_unitree_policy_rollout_not_integrated_in_default_preview",
    ]
    if not collision_dynamics_validated:
        robot_team_handoff_blockers.append("collision_dynamics_not_validated")
    if not collision_avoidance_validated:
        robot_team_handoff_blockers.append("collision_avoidance_not_validated")
    if not collision_summary["visible_scene_collision_alignment_validated"]:
        robot_team_handoff_blockers.append("visible_scene_collision_alignment_not_validated")
    official_policy_handoff = {
        "status": "required_for_robot_team_grade_walking_data",
        "entrypoint": "python -m blueprint_pipeline.official_g1_policy_handoff",
        "required_artifacts": [
            "robot_team_handoff_manifest.json",
            "robot_team_timeseries.jsonl",
            "sensor_stream_manifest.json",
            "contact_manifest.json",
            "policy_execution_trace_enriched.jsonl",
            "robot_pov_manifest.json",
        ],
        "boundary": (
            "Use the official Unitree RL Gym G1 handoff path for balanced-controller "
            "rollouts, qpos/qvel/control streams, contact traces, and robot-team review. "
            "The default MuJoCo scene preview is not a substitute."
        ),
    }

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
        "locomotion_controller_integrated": False,
        "walking_motion_proven": False,
        "walking_style_preview_animation_rendered": bool(preview_joint_addresses),
        "training_grade_policy_rollout_proven": False,
        "physical_robot_readiness_proven": False,
        "safety_validated": False,
        "contact_dynamics_validated": collision_dynamics_validated,
        "collision_geometry_loaded": collision_summary["collision_geometry_loaded"],
        "scene_collision_mesh_geom_enabled": collision_summary[
            "scene_collision_mesh_geom_enabled"
        ],
        "scene_visual_mesh_collision_twin_enabled": collision_summary[
            "scene_visual_mesh_collision_twin_enabled"
        ],
        "scene_visual_mesh_collisions_enabled": collision_summary[
            "scene_visual_mesh_collisions_enabled"
        ],
        "scene_collision_proxy_geoms_enabled": collision_summary[
            "scene_collision_proxy_geoms_enabled"
        ],
        "scene_collision_proxy_geom_count": collision_summary[
            "scene_collision_proxy_geom_count"
        ],
        "visible_scene_collision_alignment_validated": collision_summary[
            "visible_scene_collision_alignment_validated"
        ],
        "proxy_collision_model_used": collision_summary["proxy_collision_model_used"],
        "proxy_collision_governed_preview_proven": collision_summary[
            "proxy_collision_governed_preview_proven"
        ],
        "collision_dynamics_validated": collision_dynamics_validated,
        "collision_avoidance_validated": collision_avoidance_validated,
        "collision_response_validated": collision_summary["collision_response_validated"],
        "physics_controlled_preview_proven": physics_controlled_preview_proven,
        "robot_team_handoff_ready": False,
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
        "collision_summary": collision_summary,
        "collision_proxy_summary": mesh_info.get("collision_proxy_summary"),
        "collision_proxy_geoms": mesh_info.get("collision_proxy_geoms", [])[:200],
        "collision_geometry_contract": {
            "visual_geom": "blueprint_scene_visual",
            "visual_geom_collisions_enabled": False,
            "collision_geom": "blueprint_scene_collision"
            if collision_proxy_count == 0
            else None,
            "collision_geom_collisions_enabled": collision_proxy_count == 0,
            "collision_proxy_geoms_enabled": collision_proxy_count > 0,
            "collision_proxy_geom_count": collision_proxy_count,
            "floor_geom": "blueprint_reference_floor",
            "floor_collisions_enabled": True,
        },
    }
    spawn_trace = {
        **common,
        "artifact_type": "spawn_trace",
        "status": "validated",
        "spawn_pose_loaded": True,
        "scenario_eval_matrix_path": matrix_summary.get("scenario_eval_matrix_path"),
        "scenario_eval_run_count": len(matrix_runs),
        "spawn_count": len(spawn_records),
        "spawns": spawn_records,
        "spawn_pose": spawn_records[0]["spawn_pose"] if spawn_records else None,
        "target_pose": spawn_records[0]["target_pose"] if spawn_records else None,
        "keyframe": "stand" if stand_key_id >= 0 else None,
        "deterministic_spawn_target_handling": True,
        "ai_route_selection_used_at_runtime": False,
        "ai_route_selection_boundary": (
            "AI may propose routes upstream, but this command executes only frozen "
            "matrix rows and deterministic seeded route derivation for reproducible evals."
        ),
    }
    policy_trace = {
        **common,
        "artifact_type": "default_sim_policy_trace",
        "status": "completed",
        "policy_id": POLICY_ID,
        "policy_kind": "mujoco_contact_governed_waypoint_preview",
        "task_intent": "walk_to_target",
        "policy_source": "repo_generated_default_smoke_policy",
        "policy_downloaded_from_online": False,
        "default_policy_executed": True,
        "policy_execution_completed": True,
        "locomotion_controller_integrated": False,
        "walking_motion_proven": False,
        "walking_style_preview_animation_rendered": bool(preview_joint_addresses),
        "training_grade_policy_rollout_proven": False,
        "policy_semantics": (
            "mujoco_contact_governed_waypoint_preview_with_joint_pose_animation_"
            "not_balanced_humanoid_locomotion_controller"
        ),
        "data_quality_boundary": (
            "This trace proves MuJoCo command execution, G1 asset placement, and "
            "collision-governed preview motion where scene contacts stop or redirect "
            "the committed preview path. It is still not a balanced humanoid "
            "walking-controller rollout or training-grade Unitree policy data."
        ),
        "scenario_eval_matrix_path": matrix_summary.get("scenario_eval_matrix_path"),
        "scenario_eval_run_count": len(matrix_runs),
        "attempt_count": len(attempts),
        "start_pose": all_policy_records[0]["start_pose"] if all_policy_records else None,
        "target_pose": all_policy_records[0]["target_pose"] if all_policy_records else None,
        "final_pose": all_policy_records[0]["final_pose"] if all_policy_records else None,
        "step_count": bounded_steps,
        "step_count_source": step_count_source,
        "requested_duration_seconds": requested_duration_seconds,
        "model_timestep_s": model_timestep,
        "simulated_duration_s": round(bounded_steps * model_timestep, 9)
        if model_timestep
        else None,
        "rendered_episode_count": len(rendered_episode_indexes) if frames else 0,
        "rendered_step_count": sum(
            int(_mapping(attempt.get("metrics")).get("rendered_step_count") or 0)
            for attempt in attempts
        ),
        "attempts": all_policy_records,
        "actions": all_policy_records[0]["actions"] if all_policy_records else [],
        "collision_summary": collision_summary,
        "collision_probe_trace": full_collision_probe_trace[:500],
        "collision_response_events": full_collision_response_events[:500],
        "physics_controlled_preview_proven": physics_controlled_preview_proven,
        "robot_team_handoff_ready": False,
        "robot_team_handoff_blockers": robot_team_handoff_blockers,
        "official_policy_handoff": official_policy_handoff,
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
        "rendered_episode_count": len(rendered_episode_indexes) if frames else 0,
        "render_every_step": bool(render_every_step),
        "max_rendered_episodes": max_rendered_episodes,
        "max_rendered_steps": max_rendered_steps,
        "model_timestep_s": model_timestep,
        "video_timing_semantics": (
            "Generated MP4 concat durations are derived from rendered frame sim_time_s. "
            "When every step is rendered, source frames cover every simulated physics step."
        ),
        "rendered_scenario_eval_run_ids": sorted(
            {
                _string(frame.get("scenario_eval_run_id"))
                for frame in frames
                if _string(frame.get("scenario_eval_run_id"))
            }
        ),
        "recording_camera_contract": RENDER_CAMERA_CONTRACT,
        "overview_frames": visual_artifacts["overview_frames"],
        "robot_pov_frames": visual_artifacts["robot_pov_frames"],
        "side_frames": visual_artifacts["side_frames"],
        "overview_video": visual_artifacts["overview_video"],
        "robot_pov_video": visual_artifacts["robot_pov_video"],
        "side_video": visual_artifacts["side_video"],
        "blank_scene_checks": visual_artifacts["blank_scene_checks"],
        "texture_material_evidence": visual_artifacts["texture_material_evidence"],
        "limitations": visual_artifacts["limitations"],
        "robot_team_handoff_ready": False,
        "robot_team_handoff_blockers": robot_team_handoff_blockers,
    }
    videos = [
        str(video.get("path"))
        for video in (
            visual_artifacts["overview_video"],
            visual_artifacts["robot_pov_video"],
            visual_artifacts["side_video"],
        )
        if video.get("status") == "complete" and video.get("path")
    ]
    simulator_output = (
        Path(simulator_output_path).resolve()
        if simulator_output_path
        else Path(os.environ["BLUEPRINT_SIMULATOR_OUTPUT"]).resolve()
        if os.environ.get("BLUEPRINT_SIMULATOR_OUTPUT")
        else output_root / "mujoco_g1_simulator_output.json"
    )
    artifact_paths = {
        "scene_trace": str(output_root / "scene_load_trace.json"),
        "spawn_trace": str(output_root / "spawn_trace.json"),
        "policy_trace": str(output_root / "policy_execution_trace.json"),
        "sim_robot_pov_evidence": str(output_root / "sim_robot_pov_evidence_manifest.json"),
        "source_scene_glb": str(scene_glb),
        "converted_scene_obj": str(scene_obj),
        "generated_mjcf": str(wrapper_xml),
        "generated_g1_mjcf": str(generated_g1_xml),
        "scenario_eval_matrix": _string(matrix_summary.get("scenario_eval_matrix_path")) or None,
        "frames": [frame["path"] for frame in frames],
        "overview_frames": visual_artifacts["overview_frames"],
        "robot_pov_frames": visual_artifacts["robot_pov_frames"],
        "side_frames": visual_artifacts["side_frames"],
        "videos": videos,
    }
    artifact_manifest = {
        **common,
        "artifact_type": "artifact_manifest",
        "status": "complete",
        "scenario_eval_matrix": matrix_summary,
        "scenario_eval_run_count": len(matrix_runs),
        "attempt_count": len(attempts),
        "rendered_episode_count": len(rendered_episode_indexes) if frames else 0,
        "recording_camera_contract": RENDER_CAMERA_CONTRACT,
        "artifacts": artifact_paths,
        "artifact_classes": {
            "local_sim": [
                str(output_root / "scene_load_trace.json"),
                str(output_root / "spawn_trace.json"),
                str(output_root / "policy_execution_trace.json"),
                str(simulator_output),
            ],
            "visual": [*artifact_paths["frames"], *videos],
            "provider_proof": [],
            "cost": [],
            "readiness": [],
        },
        "visual_artifacts": visual_artifacts,
        "texture_material_evidence": visual_artifacts["texture_material_evidence"],
        "blank_scene_checks": visual_artifacts["blank_scene_checks"],
        "collision_summary": collision_summary,
        "collision_proxy_summary": mesh_info.get("collision_proxy_summary"),
        "physics_controlled_preview_proven": physics_controlled_preview_proven,
        "robot_team_handoff_ready": False,
        "robot_team_handoff_blockers": robot_team_handoff_blockers,
        "official_policy_handoff": official_policy_handoff,
        "limitations": visual_artifacts["limitations"],
        "files": [str(wrapper_xml), str(generated_g1_xml), str(scene_obj)]
        + [frame["path"] for frame in frames]
        + videos,
    }
    write_json(output_root / "scene_load_trace.json", scene_trace)
    write_json(output_root / "spawn_trace.json", spawn_trace)
    write_json(output_root / "policy_execution_trace.json", policy_trace)
    write_json(output_root / "sim_robot_pov_evidence_manifest.json", pov_manifest)
    write_json(output_root / "artifact_manifest.json", artifact_manifest)

    required_scenario_eval_run_ids = [
        _string(run_id)
        for run_id in matrix_summary.get("required_scenario_eval_run_ids", [])
        if _string(run_id)
    ]
    covered_scenario_eval_run_ids = sorted(
        {
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in attempts
            if _string(attempt.get("scenario_eval_run_id"))
        }
    )
    duplicate_scenario_eval_run_ids = sorted(
        {
            run_id
            for run_id in required_scenario_eval_run_ids
            if required_scenario_eval_run_ids.count(run_id) > 1
        }
    )
    missing_scenario_eval_run_ids = sorted(
        set(required_scenario_eval_run_ids) - set(covered_scenario_eval_run_ids)
    )
    attempt_count_matches_matrix_count = len(attempts) == len(required_scenario_eval_run_ids)
    scenario_eval_run_id_coverage_exact = (
        set(covered_scenario_eval_run_ids) == set(required_scenario_eval_run_ids)
        and len(covered_scenario_eval_run_ids) == len(required_scenario_eval_run_ids)
    )
    scenario_eval_run_coverage_complete = (
        bool(required_scenario_eval_run_ids)
        and attempt_count_matches_matrix_count
        and scenario_eval_run_id_coverage_exact
        and not missing_scenario_eval_run_ids
        and not duplicate_scenario_eval_run_ids
    )
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
        "default_sim_policy_execution_semantics": (
            "repo contact-governed waypoint preview executed with gait-pose animation; "
            "MuJoCo scene contacts stop or redirect committed preview motion, but no "
            "balanced Unitree walking controller is integrated"
        ),
        "robot_policy_execution_proven": False,
        "robot_team_policy_execution_proven": False,
        "locomotion_controller_integrated": False,
        "walking_motion_proven": False,
        "walking_style_preview_animation_rendered": bool(preview_joint_addresses),
        "training_grade_policy_rollout_proven": False,
        "collision_geometry_loaded": collision_summary["collision_geometry_loaded"],
        "scene_collision_mesh_geom_enabled": collision_summary[
            "scene_collision_mesh_geom_enabled"
        ],
        "scene_visual_mesh_collision_twin_enabled": collision_summary[
            "scene_visual_mesh_collision_twin_enabled"
        ],
        "scene_visual_mesh_collisions_enabled": collision_summary[
            "scene_visual_mesh_collisions_enabled"
        ],
        "scene_collision_proxy_geoms_enabled": collision_summary[
            "scene_collision_proxy_geoms_enabled"
        ],
        "scene_collision_proxy_geom_count": collision_summary[
            "scene_collision_proxy_geom_count"
        ],
        "visible_scene_collision_alignment_validated": collision_summary[
            "visible_scene_collision_alignment_validated"
        ],
        "proxy_collision_model_used": collision_summary["proxy_collision_model_used"],
        "proxy_collision_governed_preview_proven": collision_summary[
            "proxy_collision_governed_preview_proven"
        ],
        "collision_dynamics_validated": collision_dynamics_validated,
        "contact_dynamics_validated": collision_dynamics_validated,
        "collision_avoidance_validated": collision_avoidance_validated,
        "collision_response_validated": collision_summary["collision_response_validated"],
        "physics_controlled_preview_proven": physics_controlled_preview_proven,
        "collision_free_preview": collision_free_preview,
        "blocked_collision_attempt_count": blocked_collision_attempt_count,
        "contact_detection_sampled": collision_summary["contact_detection_sampled"],
        "contact_sample_count": collision_summary["contact_sample_count"],
        "committed_contact_sample_count": collision_summary["committed_contact_sample_count"],
        "collision_probe_contact_sample_count": collision_summary[
            "collision_probe_contact_sample_count"
        ],
        "robot_scene_contact_event_count": collision_summary[
            "robot_scene_contact_event_count"
        ],
        "robot_floor_contact_event_count": collision_summary[
            "robot_floor_contact_event_count"
        ],
        "rejected_scene_collision_probe_count": collision_summary[
            "rejected_scene_collision_probe_count"
        ],
        "collision_response_event_count": collision_summary[
            "collision_response_event_count"
        ],
        "collision_semantics": (
            "The external warehouse visual mesh is non-colliding. When proxy collision "
            "geometry is used, MuJoCo obstacle proxies are contact-probed before each "
            "preview pose is committed; candidate poses that touch proxy boxes/shelves are "
            "rejected, redirected, or stopped. Proxy-only success does not validate that "
            "the robot avoided every visible mesh surface, and this is not a balanced "
            "Unitree locomotion policy."
        ),
        "collision_summary": collision_summary,
        "sim_robot_pov_evidence_proven": bool(frames),
        "real_robot_pov_evidence_proven": False,
        "evaluation_mode": "deterministic_mujoco_contact_governed_preview_harness",
        "ai_route_selection_used_at_runtime": False,
        "deterministic_per_episode_spawn_target_seed_handling": True,
        "step_count": bounded_steps,
        "step_count_source": step_count_source,
        "requested_duration_seconds": requested_duration_seconds,
        "model_timestep_s": model_timestep,
        "simulated_duration_s": round(bounded_steps * model_timestep, 9)
        if model_timestep
        else None,
        "scenario_eval_matrix": matrix_summary,
        "scenario_eval_matrix_path": matrix_summary.get("scenario_eval_matrix_path"),
        "scenario_eval_run_count": len(matrix_runs),
        "attempt_count": len(attempts),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_count": len(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_count": len(missing_scenario_eval_run_ids),
        "attempt_count_matches_matrix_count": attempt_count_matches_matrix_count,
        "scenario_eval_run_id_coverage_exact": scenario_eval_run_id_coverage_exact,
        "duplicate_scenario_eval_run_ids": duplicate_scenario_eval_run_ids,
        "required_scenario_eval_run_ids": required_scenario_eval_run_ids,
        "covered_scenario_eval_run_ids": covered_scenario_eval_run_ids,
        "missing_scenario_eval_run_ids": missing_scenario_eval_run_ids,
        "scenario_eval_run_coverage_complete": scenario_eval_run_coverage_complete,
        "rendered_episode_count": len(rendered_episode_indexes) if frames else 0,
        "render_every_step": bool(render_every_step),
        "max_rendered_episodes": max_rendered_episodes,
        "max_rendered_steps": max_rendered_steps,
        "video_timing_semantics": (
            "MP4 frame durations are derived from simulated frame timestamps. A realtime "
            "full-frame export requires --duration-seconds plus --render-every-step and "
            "sufficient storage for every rendered physics step."
        ),
        "robot_team_handoff_ready": False,
        "robot_team_handoff_blockers": robot_team_handoff_blockers,
        "official_policy_handoff": official_policy_handoff,
        "collision_probe_trace": full_collision_probe_trace[:500],
        "collision_response_events": full_collision_response_events[:500],
        "recording_camera_contract": RENDER_CAMERA_CONTRACT,
        "attempts": attempts,
        "artifact_paths": {
            **artifact_paths,
            "artifact_manifest": str(output_root / "artifact_manifest.json"),
        },
        "visual_artifacts": visual_artifacts,
        "texture_material_evidence": visual_artifacts["texture_material_evidence"],
        "blank_scene_checks": visual_artifacts["blank_scene_checks"],
        "claim_boundary": claim_boundary,
        **claim_boundary,
    }
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
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Derive simulator steps from model timestep for a realtime-duration export.",
    )
    parser.add_argument("--skip-render-frames", action="store_true")
    parser.add_argument(
        "--render-every-step",
        action="store_true",
        help="Render every simulated step for selected episodes instead of sampled review frames.",
    )
    parser.add_argument("--max-rendered-episodes", type=int, default=3)
    parser.add_argument("--max-rendered-steps", type=int, default=24)
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
        duration_seconds=args.duration_seconds,
        render_frames=not args.skip_render_frames,
        render_every_step=args.render_every_step,
        max_rendered_episodes=args.max_rendered_episodes,
        max_rendered_steps=args.max_rendered_steps,
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
                "attempt_count": payload.get("attempt_count"),
                "scenario_eval_run_count": payload.get("scenario_eval_run_count"),
                "missing_scenario_eval_run_count": payload.get(
                    "missing_scenario_eval_run_count"
                ),
                "collision_geometry_loaded": payload.get("collision_geometry_loaded"),
                "collision_dynamics_validated": payload.get("collision_dynamics_validated"),
                "robot_team_handoff_ready": payload.get("robot_team_handoff_ready"),
                "output_dir": payload.get("output_dir"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
