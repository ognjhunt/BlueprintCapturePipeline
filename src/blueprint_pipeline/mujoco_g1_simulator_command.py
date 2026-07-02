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
MUJOCO_G1_BATCH_TRACE_PACKAGE_SCHEMA_VERSION = "mujoco_g1_batch_trace_package.v1"
MUJOCO_G1_BATCH_CLOSURE_SCHEMA_VERSION = "mujoco_g1_batch_closure_manifest.v1"
MUJOCO_G1_DIGITAL_TWIN_FIDELITY_QA_SCHEMA_VERSION = "mujoco_g1_digital_twin_fidelity_qa.v1"
DEFAULT_MENAGERIE_REF = "4c358ef9d9d7f32ca58b40b490884a0c1726a440"
G1_SOURCE_URL = "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1"
G1_REPOSITORY_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
POLICY_ID = "blueprint_default_walk_to_target_smoke_policy"
TASK_GOAL_TOLERANCE_M = 0.25
TASK_STUCK_MIN_PROGRESS_RATIO = 0.05
TASK_STUCK_MIN_PROGRESS_M = 0.10
TASK_FALL_ROOT_HEIGHT_M = 0.45
TASK_CLEARANCE_THRESHOLD_M = 0.15
REQUIRED_TASK_METRIC_KEYS = (
    "goal_reached",
    "final_target_error_m",
    "goal_tolerance_m",
    "min_clearance_m",
    "clearance_threshold_m",
    "timeout_count",
    "fall_count",
    "stuck_event_count",
    "near_miss_event_count",
    "robot_scene_contact_event_count",
    "collision_response_event_count",
    "actual_path_distance_m",
    "path_efficiency_ratio",
    "max_path_deviation_m",
    "mean_path_deviation_m",
    "policy_instability_detected",
)
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
MANIPULATION_READY_ARM_SELECTIONS = ("right", "left", "both")
MANIPULATION_READY_ARM_JOINT_DELTAS = {
    "left": {
        "left_shoulder_pitch_joint": -0.85,
        "left_shoulder_roll_joint": 0.15,
        "left_shoulder_yaw_joint": 0.10,
        "left_elbow_joint": -0.23,
        "left_wrist_roll_joint": -0.10,
        "left_wrist_pitch_joint": -0.15,
    },
    "right": {
        "right_shoulder_pitch_joint": -0.85,
        "right_shoulder_roll_joint": -0.15,
        "right_shoulder_yaw_joint": -0.10,
        "right_elbow_joint": -0.23,
        "right_wrist_roll_joint": 0.10,
        "right_wrist_pitch_joint": -0.15,
    },
}
ROBOT_POV_VISIBLE_SELF_BODY_PARTS = ("shoulder", "elbow", "wrist", "hand")


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or fallback


def _float_triplet(value: Any) -> list[float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) < 3:
            return None
        return [float(value[index]) for index in range(3)]
    except (TypeError, ValueError, IndexError):
        return None


def _bounds_payload(bounds: Any) -> dict[str, Any] | None:
    if isinstance(bounds, (str, bytes)):
        return None
    try:
        if len(bounds) < 2:
            return None
        lower = _float_triplet(bounds[0])
        upper = _float_triplet(bounds[1])
    except (TypeError, IndexError):
        return None
    if lower is None or upper is None:
        return None
    extents = [upper[index] - lower[index] for index in range(3)]
    if any(value <= 0.0 for value in extents):
        return None
    return {
        "bounds": [
            [round(value, 6) for value in lower],
            [round(value, 6) for value in upper],
        ],
        "extents": [round(value, 6) for value in extents],
        "volume_m3_estimate": round(extents[0] * extents[1] * extents[2], 6),
    }


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
    mesh_summaries: list[dict[str, Any]] = []
    material_summaries = [
        {
            "material_index": index,
            "name": _string(material.get("name")) or f"material_{index:04d}",
        }
        for index, material in enumerate(gltf_json.get("materials", []) or [])
        if isinstance(material, Mapping)
    ]
    for mesh_index, mesh in enumerate(gltf_json.get("meshes", []) or []):
        if not isinstance(mesh, Mapping):
            continue
        mesh_primitive_count = 0
        mesh_material_indexes: list[int] = []
        for primitive in mesh.get("primitives", []) or []:
            if not isinstance(primitive, Mapping):
                continue
            primitive_count += 1
            mesh_primitive_count += 1
            material_index = primitive.get("material")
            if isinstance(material_index, int):
                mesh_material_indexes.append(material_index)
            for attribute_name in primitive.get("attributes", {}) or {}:
                attribute_usage[attribute_name] = attribute_usage.get(attribute_name, 0) + 1
        mesh_summaries.append(
            {
                "mesh_index": mesh_index,
                "name": _string(mesh.get("name")) or f"mesh_{mesh_index:04d}",
                "primitive_count": mesh_primitive_count,
                "material_indexes": sorted(set(mesh_material_indexes)),
            }
        )
    node_summaries: list[dict[str, Any]] = []
    for node_index, node in enumerate(gltf_json.get("nodes", []) or []):
        if not isinstance(node, Mapping):
            continue
        mesh_index = node.get("mesh")
        node_summaries.append(
            {
                "node_index": node_index,
                "name": _string(node.get("name")) or f"node_{node_index:04d}",
                "mesh_index": mesh_index if isinstance(mesh_index, int) else None,
            }
        )
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
        "meshes": mesh_summaries[:200],
        "nodes": node_summaries[:200],
        "materials": material_summaries[:200],
        "named_mesh_count": sum(
            1
            for mesh in mesh_summaries
            if _string(mesh.get("name")) and not _string(mesh.get("name")).startswith("mesh_")
        ),
        "named_node_count": sum(
            1
            for node in node_summaries
            if _string(node.get("name")) and not _string(node.get("name")).startswith("node_")
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


def _obj_texture_material_summary(obj_path: Path) -> dict[str, Any]:
    default = {
        "status": "no_mtl",
        "mtl_file": None,
        "map_kd_texture_file": None,
        "map_kd_texture_path": None,
        "texture_exists": False,
    }
    try:
        obj_dir = obj_path.parent
        mtl_names: list[str] = []
        if obj_path.is_file():
            with obj_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].lower() == "mtllib":
                        mtl_names.extend(name for name in parts[1].split() if name)
        if mtl_names:
            mtl_paths = [obj_dir / name for name in mtl_names]
        else:
            mtl_paths = sorted(obj_dir.glob("*.mtl"))
        mtl_paths = [path for path in mtl_paths if path.is_file()]
        if not mtl_paths:
            return default
        inspected_mtl_files: list[str] = []
        for mtl_path in mtl_paths:
            inspected_mtl_files.append(mtl_path.name)
            try:
                lines = mtl_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2 or parts[0].lower() != "map_kd":
                    continue
                texture_ref = parts[1].strip().strip('"')
                texture_path = Path(texture_ref)
                if not texture_path.is_absolute():
                    texture_path = mtl_path.parent / texture_path
                texture_exists = texture_path.is_file()
                return {
                    "status": "inspected",
                    "mtl_file": mtl_path.name,
                    "inspected_mtl_files": inspected_mtl_files,
                    "map_kd_texture_file": texture_ref,
                    "map_kd_texture_path": str(texture_path.resolve())
                    if texture_exists
                    else str(texture_path),
                    "texture_exists": texture_exists,
                }
        return {
            **default,
            "status": "inspected",
            "mtl_file": inspected_mtl_files[0] if inspected_mtl_files else None,
            "inspected_mtl_files": inspected_mtl_files,
        }
    except Exception as exc:  # pragma: no cover - defensive artifact inspection.
        return {
            **default,
            "status": "error",
            "error": type(exc).__name__,
        }


def _geometry_material_name(geometry: Any) -> str:
    visual = getattr(geometry, "visual", None)
    material = getattr(visual, "material", None)
    return _string(getattr(material, "name", None))


def _safe_len(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return 0


def _is_generated_semantic_name(value: Any) -> bool:
    name = _safe_id(value, fallback="")
    if not name:
        return True
    generated_exact_names = {
        "combined_scene_mesh",
        "geometry",
        "mesh",
        "node",
        "scene",
        "world",
        "object",
    }
    if name in generated_exact_names:
        return True
    generated_prefixes = (
        "geometry_",
        "mesh_",
        "node_",
        "visible_object_",
        "component_",
        "trimesh_",
    )
    return any(name.startswith(prefix) for prefix in generated_prefixes)


def _geometry_visible_object(
    *,
    index: int,
    geometry_name: str,
    geometry: Any,
    gltf_mesh: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    bounds = _bounds_payload(getattr(geometry, "bounds", None))
    if bounds is None:
        return None
    metadata = _mapping(getattr(geometry, "metadata", None))
    semantic_name = (
        _string(metadata.get("name"))
        or geometry_name
        or _string(gltf_mesh.get("name") if gltf_mesh else None)
        or f"visible_object_{index:04d}"
    )
    semantic_label_available = any(
        _string(candidate) and not _is_generated_semantic_name(candidate)
        for candidate in (
            metadata.get("name"),
            geometry_name,
            gltf_mesh.get("name") if gltf_mesh else None,
        )
    )
    material_name = _geometry_material_name(geometry)
    gltf_material_indexes = []
    if gltf_mesh:
        material_indexes = gltf_mesh.get("material_indexes")
        if isinstance(material_indexes, Sequence) and not isinstance(material_indexes, (str, bytes)):
            gltf_material_indexes = [
                int(value) for value in material_indexes if isinstance(value, int)
            ]
    return {
        "object_id": f"visible_object_{index:04d}_{_safe_id(semantic_name)}",
        "source_component_index": index,
        "name": semantic_name,
        "semantic_label_available": semantic_label_available,
        "geometry_name": geometry_name or None,
        "material_name": material_name or None,
        "gltf_mesh_index": gltf_mesh.get("mesh_index") if gltf_mesh else None,
        "gltf_mesh_name": gltf_mesh.get("name") if gltf_mesh else None,
        "gltf_material_indexes": gltf_material_indexes,
        "vertex_count": _safe_len(getattr(geometry, "vertices", [])),
        "face_count": _safe_len(getattr(geometry, "faces", [])),
        **bounds,
    }


def _visual_object_semantics_summary(
    loaded_scene: Any,
    fallback_mesh: Any,
    visual_summary: Mapping[str, Any],
) -> dict[str, Any]:
    gltf_meshes = [
        dict(mesh)
        for mesh in visual_summary.get("meshes", []) or []
        if isinstance(mesh, Mapping)
    ]
    visible_objects: list[dict[str, Any]] = []
    scene_geometry = getattr(loaded_scene, "geometry", None)
    if isinstance(scene_geometry, Mapping) and scene_geometry:
        for index, (geometry_name, geometry) in enumerate(sorted(scene_geometry.items())):
            gltf_mesh = gltf_meshes[index] if index < len(gltf_meshes) else None
            visible_object = _geometry_visible_object(
                index=index,
                geometry_name=_string(geometry_name),
                geometry=geometry,
                gltf_mesh=gltf_mesh,
            )
            if visible_object:
                visible_objects.append(visible_object)
    if not visible_objects:
        visible_object = _geometry_visible_object(
            index=0,
            geometry_name=_string(getattr(fallback_mesh, "metadata", {}).get("name"))
            if isinstance(getattr(fallback_mesh, "metadata", None), Mapping)
            else "combined_scene_mesh",
            geometry=fallback_mesh,
            gltf_mesh=gltf_meshes[0] if gltf_meshes else None,
        )
        if visible_object:
            visible_objects.append(visible_object)

    semantic_labeled_objects = [
        item for item in visible_objects if item.get("semantic_label_available") is True
    ]
    materialized_objects = [
        item
        for item in visible_objects
        if _string(item.get("material_name")) or item.get("gltf_material_indexes")
    ]
    node_names = [
        _string(node.get("name"))
        for node in visual_summary.get("nodes", []) or []
        if isinstance(node, Mapping) and _string(node.get("name"))
    ]
    mesh_names = [
        _string(mesh.get("name"))
        for mesh in gltf_meshes
        if _string(mesh.get("name"))
    ]
    status = "available" if visible_objects and semantic_labeled_objects else "missing"
    blockers = []
    if not visible_objects:
        blockers.append("no_visible_geometry_objects_detected")
    if visible_objects and not semantic_labeled_objects:
        blockers.append("visible_geometry_object_names_missing")
    return {
        "status": status,
        "visible_object_count": len(visible_objects),
        "named_visible_object_count": len(semantic_labeled_objects),
        "semantic_labeled_visible_object_count": len(semantic_labeled_objects),
        "material_referenced_visible_object_count": len(materialized_objects),
        "visible_objects": visible_objects[:200],
        "gltf_mesh_names": mesh_names[:200],
        "gltf_node_names": node_names[:200],
        "gltf_material_count": int(visual_summary.get("materials_count") or 0),
        "blockers": blockers,
        "proof_boundary": (
            "Visible-object semantics are extracted from GLB mesh/node names and "
            "trimesh geometry components. They prove component identity only to the "
            "extent those source assets expose stable names and bounds."
        ),
    }


def _convert_glb_to_obj(
    glb_path: Path,
    obj_path: Path,
    *,
    collision_proxy_limit: int = 160,
    collision_proxy_mode: str = "aabb",
) -> dict[str, Any]:
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
    obj_texture_material_summary = _obj_texture_material_summary(obj_path)
    glb_visual_summary = _glb_visual_summary(glb_path)
    visual_object_semantics_summary = _visual_object_semantics_summary(
        loaded_scene=loaded,
        fallback_mesh=mesh,
        visual_summary=glb_visual_summary,
    )
    collision_proxy_geoms, collision_proxy_summary = _collision_proxy_geoms_from_mesh(
        mesh, max_proxies=collision_proxy_limit, mode=collision_proxy_mode
    )
    return {
        "source_glb": str(glb_path),
        "converted_obj": str(obj_path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "centroid": mesh.centroid.tolist(),
        "visual_asset_summary": glb_visual_summary,
        "visual_object_semantics_summary": visual_object_semantics_summary,
        "obj_vertex_color_summary": obj_vertex_color_summary,
        "obj_texture_material_summary": obj_texture_material_summary,
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


def _rotation_matrix_to_quaternion(rotation: Any) -> list[float]:
    import numpy as np  # type: ignore[import-not-found]

    matrix = np.asarray(rotation, dtype=float)
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.asarray([w, x, y, z], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-12:
        quat = quat / norm
    return [round(float(value), 8) for value in quat]


def _aabb_collision_proxy(
    component_index: int,
    lower: Sequence[float],
    upper: Sequence[float],
    extents: Sequence[float],
    volume: float,
) -> dict[str, Any]:
    margin = 0.035
    pos = [(lower[index] + upper[index]) / 2.0 for index in range(3)]
    size = [max(0.025, extents[index] / 2.0 + margin) for index in range(3)]
    return {
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


def _obb_proxy_from_component(
    component_index: int,
    component: Any,
    lower: Sequence[float],
    upper: Sequence[float],
    extents: Sequence[float],
    volume: float,
) -> dict[str, Any] | None:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import trimesh  # type: ignore[import-not-found]

        to_origin, obb_extents = trimesh.bounds.oriented_bounds(component)
        transform = np.linalg.inv(np.asarray(to_origin, dtype=float))
        obb_extents_array = np.asarray(obb_extents, dtype=float)
        proxy = _aabb_collision_proxy(component_index, lower, upper, extents, volume)
        proxy["pos"] = [round(float(value), 6) for value in transform[:3, 3]]
        proxy["size"] = [
            round(max(0.025, float(obb_extents_array[index]) / 2.0 + 0.035), 6)
            for index in range(3)
        ]
        proxy["quat"] = _rotation_matrix_to_quaternion(transform[:3, :3])
        return proxy
    except Exception:
        return None


def _convex_decomposition_vertices_from_component(
    component: Any,
    *,
    backend: str | None,
    max_vertices: int = 512,
) -> list[list[float]]:
    if not backend:
        return []
    try:
        import numpy as np  # type: ignore[import-not-found]

        if backend == "coacd":
            import coacd  # type: ignore[import-not-found]

            vertices = np.asarray(getattr(component, "vertices", []), dtype=float)
            faces = np.asarray(getattr(component, "faces", []), dtype=np.int32)
            pieces = coacd.run_coacd(coacd.Mesh(vertices, faces))
        elif backend == "trimesh_vhacd":
            import trimesh  # type: ignore[import-not-found]

            decomposition = getattr(getattr(trimesh, "decomposition", None), "convex_decomposition", None)
            if decomposition is None:
                return []
            pieces = decomposition(component)
        else:
            return []
        if not isinstance(pieces, Sequence) or isinstance(pieces, (str, bytes)):
            pieces = [pieces]
        out: list[list[float]] = []
        for piece in pieces:
            piece_vertices = None
            if isinstance(piece, Sequence) and not isinstance(piece, (str, bytes)) and piece:
                piece_vertices = piece[0]
            if piece_vertices is None:
                piece_vertices = getattr(piece, "vertices", None)
            try:
                for vertex in np.asarray(piece_vertices, dtype=float)[:max_vertices]:
                    out.append([round(float(coord), 6) for coord in vertex[:3]])
                    if len(out) >= max_vertices:
                        return out
            except Exception:
                continue
        return out
    except Exception:
        return []


def _collision_proxy_geoms_from_mesh(
    mesh: Any, *, max_proxies: int = 160, mode: str = "aabb"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_mode = str(mode or "aabb").strip().lower()
    if requested_mode not in {"aabb", "obb", "convex"}:
        requested_mode = "aabb"
    convex_decomposition_status = None
    convex_backend: str | None = None
    if requested_mode == "convex":
        try:
            import coacd  # noqa: F401  # type: ignore[import-not-found]

            convex_backend = "coacd"
        except Exception:
            try:
                import trimesh  # type: ignore[import-not-found]

                if getattr(getattr(trimesh, "decomposition", None), "convex_decomposition", None):
                    convex_backend = "trimesh_vhacd"
            except Exception:
                convex_backend = None
        convex_decomposition_status = (
            "unavailable_fell_back_to_aabb" if convex_backend is None else "available"
        )
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
    skipped_indexes: dict[str, list[int]] = {key: [] for key in skipped}
    obb_fallback_indexes: list[int] = []
    convex_decomposition_generated_count = 0
    for component_index, component in enumerate(components):
        try:
            bounds = component.bounds
            lower = [float(value) for value in bounds[0][:3]]
            upper = [float(value) for value in bounds[1][:3]]
        except Exception:
            skipped["degenerate"] += 1
            skipped_indexes["degenerate"].append(component_index)
            continue
        extents = [upper[index] - lower[index] for index in range(3)]
        if any(value <= 0.0 for value in extents):
            skipped["degenerate"] += 1
            skipped_indexes["degenerate"].append(component_index)
            continue
        z_min = lower[2]
        z_max = upper[2]
        xy_area = extents[0] * extents[1]
        volume = xy_area * extents[2]
        if z_max <= 0.14 and xy_area >= 2.0:
            skipped["floor_like"] += 1
            skipped_indexes["floor_like"].append(component_index)
            continue
        if z_min >= 2.35:
            skipped["overhead"] += 1
            skipped_indexes["overhead"].append(component_index)
            continue
        if extents[0] >= 8.0 and extents[1] >= 8.0 and extents[2] >= 2.0:
            skipped["scene_shell"] += 1
            skipped_indexes["scene_shell"].append(component_index)
            continue
        if volume <= 0.001:
            skipped["degenerate"] += 1
            skipped_indexes["degenerate"].append(component_index)
            continue
        if (
            requested_mode in {"obb", "convex"}
            and convex_decomposition_status != "unavailable_fell_back_to_aabb"
        ):
            proxy = _obb_proxy_from_component(
                component_index, component, lower, upper, extents, volume
            )
            if proxy is None:
                obb_fallback_indexes.append(component_index)
                proxy = _aabb_collision_proxy(component_index, lower, upper, extents, volume)
        else:
            proxy = _aabb_collision_proxy(component_index, lower, upper, extents, volume)
        if (
            requested_mode == "convex"
            and convex_decomposition_status != "unavailable_fell_back_to_aabb"
        ):
            convex_vertices = _convex_decomposition_vertices_from_component(
                component,
                backend=convex_backend,
            )
            proxy["convex_hull_vertices"] = convex_vertices
            if convex_vertices:
                convex_decomposition_generated_count += 1
        proxies.append(proxy)
    proxies.sort(key=lambda item: float(item["volume_m3_estimate"]), reverse=True)
    bounded = proxies[: max(0, max_proxies)]
    covered_component_indexes = sorted(
        int(proxy["source_component_index"]) for proxy in bounded
    )
    truncated_component_indexes = sorted(
        int(proxy["source_component_index"]) for proxy in proxies[max(0, max_proxies) :]
    )
    reference_floor_covered_indexes = sorted(skipped_indexes["floor_like"])
    intentionally_excluded_indexes = sorted(
        [
            *skipped_indexes["overhead"],
            *skipped_indexes["scene_shell"],
        ]
    )
    uncovered_component_indexes = sorted(
        set(range(len(components)))
        - set(covered_component_indexes)
        - set(reference_floor_covered_indexes)
        - set(intentionally_excluded_indexes)
    )
    generation_methods = {
        "aabb": "component_aabb_obstacle_proxies_excluding_floor_overhead_and_scene_shell",
        "obb": "component_obb_obstacle_proxies_excluding_floor_overhead_and_scene_shell",
        "convex": "component_convex_decomposition_obstacle_proxies_excluding_floor_overhead_and_scene_shell",
    }
    summary = {
        "status": "generated" if bounded else "not_generated",
        "source_component_count": len(components),
        "proxy_count": len(bounded),
        "max_proxy_count": max_proxies,
        "collision_proxy_mode": requested_mode,
        "skipped": skipped,
        "skipped_source_component_indexes": skipped_indexes,
        "component_coverage": {
            "covered_source_component_indexes": covered_component_indexes,
            "covered_source_component_count": len(covered_component_indexes),
            "reference_floor_covered_source_component_indexes": (
                reference_floor_covered_indexes
            ),
            "intentionally_excluded_source_component_indexes": intentionally_excluded_indexes,
            "truncated_source_component_indexes": truncated_component_indexes,
            "truncated_source_component_count": len(truncated_component_indexes),
            "uncovered_source_component_indexes": uncovered_component_indexes,
            "uncovered_source_component_count": len(uncovered_component_indexes),
            "component_proxy_coverage_complete": not uncovered_component_indexes
            and not truncated_component_indexes,
        },
        "generation_method": generation_methods[requested_mode],
        "proof_boundary": (
            "Obstacle proxies are conservative MuJoCo box colliders derived from scene "
            "components. They are better than colliding with the entire visual mesh, but "
            "still need robot-team review before customer safety claims."
        ),
    }
    if requested_mode == "obb":
        summary["obb_fallback_component_indexes"] = sorted(obb_fallback_indexes)
    if requested_mode == "convex":
        if convex_decomposition_status != "unavailable_fell_back_to_aabb":
            convex_decomposition_status = (
                "generated"
                if convex_decomposition_generated_count > 0
                else f"{convex_backend}_available_decomposition_failed_box_proxy_emitted"
            )
        summary["convex_decomposition_status"] = convex_decomposition_status
        summary["convex_decomposition_backend"] = convex_backend
        summary["convex_decomposition_generated_proxy_count"] = (
            convex_decomposition_generated_count
        )
        if obb_fallback_indexes:
            summary["obb_fallback_component_indexes"] = sorted(obb_fallback_indexes)
    return bounded, summary


def _xml_float(value: Any) -> str:
    return f"{float(value):.6g}"


def _xml_vec(values: Sequence[Any]) -> str:
    return " ".join(_xml_float(value) for value in values)


def _direction_to_target(origin: Sequence[float], target: Sequence[float]) -> list[float]:
    delta = [float(target[index]) - float(origin[index]) for index in range(3)]
    magnitude = math.sqrt(sum(value * value for value in delta))
    if magnitude <= 1e-9:
        return [0.0, 0.0, -1.0]
    return [round(value / magnitude, 6) for value in delta]


def _mjcf_scene_lighting_assets(
    scene_bounds: Sequence[Sequence[float]] | None,
    scene_centroid: Sequence[float] | None,
) -> tuple[str, str]:
    legacy_headlight = (
        '    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" '
        'specular="0.6 0.6 0.6"/>'
    )
    legacy_light = (
        '    <light name="blueprint_key" pos="0 -4 8" dir="0 0 -1" '
        'directional="true"/>'
    )
    if (
        not isinstance(scene_bounds, Sequence)
        or isinstance(scene_bounds, (str, bytes))
        or len(scene_bounds) < 2
    ):
        return legacy_headlight, legacy_light
    raw_lower = _float_triplet(scene_bounds[0])
    raw_upper = _float_triplet(scene_bounds[1])
    if raw_lower is None or raw_upper is None:
        return legacy_headlight, legacy_light
    lower = [min(raw_lower[index], raw_upper[index]) for index in range(3)]
    upper = [max(raw_lower[index], raw_upper[index]) for index in range(3)]
    spans = [upper[index] - lower[index] for index in range(3)]
    if any(value <= 0.0 for value in spans):
        return legacy_headlight, legacy_light
    centroid = _float_triplet(scene_centroid)
    if centroid is None:
        centroid = [(lower[index] + upper[index]) / 2.0 for index in range(3)]
    span_x = max(spans[0], 1.0)
    span_y = max(spans[1], 1.0)
    span_z = max(spans[2], 1.0)
    scene_diag = max(spans)
    key_pos = [
        centroid[0] - 0.6 * span_x,
        centroid[1] - 0.9 * span_y,
        upper[2] + max(2.0, 0.8 * scene_diag),
    ]
    fill_pos = [
        centroid[0] + 0.75 * span_x,
        centroid[1] + 0.65 * span_y,
        upper[2] + max(1.5, 0.5 * span_z),
    ]
    key_dir = _direction_to_target(key_pos, centroid)
    fill_dir = _direction_to_target(fill_pos, centroid)
    headlight = (
        '    <headlight diffuse="0.45 0.45 0.45" ambient="0.22 0.22 0.22" '
        'specular="0.45 0.45 0.45"/>'
    )
    lights = "\n".join(
        [
            (
                f'    <light name="blueprint_key" pos="{_xml_vec(key_pos)}" '
                f'dir="{_xml_vec(key_dir)}" directional="true" castshadow="true" '
                'diffuse="0.85 0.82 0.76" ambient="0.08 0.08 0.08" '
                'specular="0.35 0.35 0.35"/>'
            ),
            (
                f'    <light name="blueprint_fill" pos="{_xml_vec(fill_pos)}" '
                f'dir="{_xml_vec(fill_dir)}" directional="true" castshadow="false" '
                'diffuse="0.35 0.40 0.48" ambient="0.05 0.05 0.06" '
                'specular="0.1 0.1 0.12"/>'
            ),
        ]
    )
    return headlight, lights


def _write_mjcf_wrapper(
    scene_obj: Path,
    g1_xml: Path,
    wrapper_path: Path,
    *,
    collision_proxies: Sequence[Mapping[str, Any]] | None = None,
    scene_texture_file: str | Path | None = None,
    scene_bounds: Sequence[Sequence[float]] | None = None,
    scene_centroid: Sequence[float] | None = None,
    render_width: int = 640,
    render_height: int = 360,
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
        quat_attr = ""
        quat = proxy.get("quat")
        if (
            isinstance(quat, Sequence)
            and not isinstance(quat, (str, bytes))
            and len(quat) >= 4
        ):
            try:
                quat_attr = f' quat="{_xml_vec(quat[:4])}"'
            except Exception:
                quat_attr = ""
        proxy_geoms.append(
            "    "
            f'<geom name="blueprint_collision_proxy_{index:03d}_{_xml_escape(proxy_id)}" '
            f'type="box" pos="{_xml_vec(pos[:3])}" size="{_xml_vec(size[:3])}"{quat_attr} '
            'rgba="0.05 0.75 0.35 0.18" contype="1" conaffinity="1" group="3"/>'
        )
    if proxy_geoms:
        collision_geometry_block = "\n".join(proxy_geoms)
    else:
        collision_geometry_block = (
            '    <geom name="blueprint_scene_collision" type="mesh" mesh="blueprint_scene_mesh"\n'
            '      material="blueprint_scene_collision_mat" contype="1" conaffinity="1" group="3"/>'
        )
    offwidth = max(1, int(render_width))
    offheight = max(1, int(render_height))
    texture_path = Path(scene_texture_file) if scene_texture_file is not None else None
    if texture_path is not None and texture_path.is_file():
        scene_material_asset = (
            f'    <texture name="blueprint_scene_tex" type="2d" file="{_xml_escape(texture_path.resolve())}"/>\n'
            '    <material name="blueprint_scene_mat" texture="blueprint_scene_tex" texuniform="false"/>'
        )
    else:
        scene_material_asset = (
            '    <material name="blueprint_scene_mat" rgba="0.45 0.50 0.55 1"/>'
        )
    headlight_asset, scene_lights = _mjcf_scene_lighting_assets(
        scene_bounds,
        scene_centroid,
    )
    wrapper = f"""<mujoco model="blueprint_mujoco_g1_simulator_command">
  <include file="{_xml_escape(g1_xml)}"/>
  <visual>
{headlight_asset}
    <global offwidth="{offwidth}" offheight="{offheight}" azimuth="140" elevation="-20"/>
    <quality shadowsize="4096"/>
    <map znear="0.01" zfar="200"/>
  </visual>
  <asset>
    <mesh name="blueprint_scene_mesh" file="{_xml_escape(scene_obj)}"/>
{scene_material_asset}
    <material name="blueprint_scene_collision_mat" rgba="0.05 0.75 0.35 0.18"/>
  </asset>
  <worldbody>
{scene_lights}
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


def _pose_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (float(b[0]) - float(a[0])) ** 2
        + (float(b[1]) - float(a[1])) ** 2
        + (float(b[2]) - float(a[2])) ** 2
    )


def _action_pose(action: Mapping[str, Any], key: str) -> tuple[float, float, float] | None:
    value = action.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    x = _number(value[0])
    y = _number(value[1])
    z = _number(value[2])
    if x is None or y is None or z is None:
        return None
    return (float(x), float(y), float(z))


def _attempt_task_outcome(
    *,
    actions: Sequence[Mapping[str, Any]],
    start: Sequence[float],
    target: Sequence[float],
    route_distance_m: float,
    collision_summary: Mapping[str, Any],
    bounded_steps: int,
    model_timestep_s: float,
) -> dict[str, Any]:
    start_pose = _rounded_pose(start)
    target_pose = _rounded_pose(target)
    root_positions = [
        pose
        for pose in (_action_pose(action, "root_position") for action in actions)
        if pose is not None
    ]
    desired_positions = [
        pose
        for pose in (_action_pose(action, "desired_root_position") for action in actions)
        if pose is not None
    ]
    final_pose = root_positions[-1] if root_positions else start_pose
    direct_distance_m = _pose_distance(start_pose, target_pose)
    final_target_error_m = _pose_distance(final_pose, target_pose)
    actual_path_distance_m = _route_distance(root_positions) if len(root_positions) > 1 else 0.0
    progress_m = max(0.0, direct_distance_m - final_target_error_m)
    progress_ratio = progress_m / direct_distance_m if direct_distance_m > 0 else 1.0
    path_deviations = [
        _pose_distance(root_pose, desired_pose)
        for root_pose, desired_pose in zip(root_positions, desired_positions)
    ]
    max_path_deviation_m = max(path_deviations) if path_deviations else 0.0
    mean_path_deviation_m = (
        sum(path_deviations) / len(path_deviations) if path_deviations else 0.0
    )
    z_values = [pose[2] for pose in root_positions]
    min_root_height_m = min(z_values) if z_values else start_pose[2]
    goal_reached = final_target_error_m <= TASK_GOAL_TOLERANCE_M
    scene_contact_count = int(collision_summary.get("robot_scene_contact_event_count") or 0)
    rejected_probe_count = int(collision_summary.get("rejected_scene_collision_probe_count") or 0)
    near_miss_event_count = int(collision_summary.get("near_miss_event_count") or rejected_probe_count)
    min_clearance_raw = _number(collision_summary.get("min_clearance_m"))
    min_clearance_m = (
        round(float(min_clearance_raw), 6) if min_clearance_raw is not None else None
    )
    clearance_threshold_m = float(
        _number(collision_summary.get("clearance_threshold_m"))
        or TASK_CLEARANCE_THRESHOLD_M
    )
    if min_clearance_m is None and not near_miss_event_count and scene_contact_count == 0:
        min_clearance_m = round(clearance_threshold_m, 6)
    clearance_threshold_violation = bool(
        collision_summary.get("clearance_threshold_violation")
        or near_miss_event_count > 0
        or (min_clearance_m is not None and min_clearance_m < clearance_threshold_m)
    )
    response_count = int(collision_summary.get("collision_response_event_count") or 0)
    stopped_steps = sum(
        1 for action in actions if _string(action.get("policy_action")) == "stopped_by_collision_probe"
    )
    redirected_steps = sum(
        1
        for action in actions
        if _string(action.get("policy_action")) == "redirected_by_collision_probe"
    )
    fall_detected = bool(min_root_height_m < TASK_FALL_ROOT_HEIGHT_M)
    timeout = not goal_reached
    stuck_detected = bool(
        not goal_reached
        and direct_distance_m > TASK_STUCK_MIN_PROGRESS_M
        and (
            progress_m < TASK_STUCK_MIN_PROGRESS_M
            or progress_ratio < TASK_STUCK_MIN_PROGRESS_RATIO
            or stopped_steps >= max(1, int(len(actions) * 0.5))
        )
    )
    endpoint_clean = bool(goal_reached and scene_contact_count == 0)
    spawn_clean = bool(
        not actions
        or _string(actions[0].get("policy_action"))
        != "redirected_by_collision_probe"
    )
    policy_instability = bool(
        len(actions) > 0 and (stopped_steps + redirected_steps) / len(actions) > 0.75
    )
    failure_mode_ids: list[str] = []
    if scene_contact_count:
        failure_mode_ids.append("failure_scene_collision_contact")
    if fall_detected:
        failure_mode_ids.append("failure_robot_fall_detected")
    if not goal_reached:
        failure_mode_ids.append("failure_target_not_reached")
    if not endpoint_clean:
        failure_mode_ids.append("failure_endpoint_not_clean")
    if stuck_detected:
        failure_mode_ids.append("failure_stuck_or_no_progress")
    if timeout:
        failure_mode_ids.append("failure_timeout")
    if policy_instability:
        failure_mode_ids.append("failure_policy_instability")
    if clearance_threshold_violation:
        failure_mode_ids.append("failure_clearance_near_miss")
    success = not failure_mode_ids
    return {
        "task_success": success,
        "task_status": "passed" if success else "failed_task_criteria",
        "failure_mode_ids": failure_mode_ids,
        "failure_reason": ",".join(failure_mode_ids) if failure_mode_ids else None,
        "goal_reached": goal_reached,
        "endpoint_clean": endpoint_clean,
        "spawn_clean": spawn_clean,
        "timeout": timeout,
        "fall_detected": fall_detected,
        "stuck_detected": stuck_detected,
        "policy_instability_detected": policy_instability,
        "final_pose": [round(float(value), 6) for value in final_pose],
        "final_target_error_m": round(final_target_error_m, 6),
        "goal_tolerance_m": TASK_GOAL_TOLERANCE_M,
        "min_clearance_m": min_clearance_m,
        "clearance_threshold_m": clearance_threshold_m,
        "clearance_threshold_violation": clearance_threshold_violation,
        "direct_start_to_target_distance_m": round(direct_distance_m, 6),
        "planned_route_distance_m": round(float(route_distance_m), 6),
        "actual_path_distance_m": round(actual_path_distance_m, 6),
        "path_efficiency_ratio": round(
            actual_path_distance_m / route_distance_m, 6
        )
        if route_distance_m > 0
        else None,
        "progress_to_goal_m": round(progress_m, 6),
        "progress_to_goal_ratio": round(progress_ratio, 6),
        "max_path_deviation_m": round(max_path_deviation_m, 6),
        "mean_path_deviation_m": round(mean_path_deviation_m, 6),
        "min_root_height_m": round(float(min_root_height_m), 6),
        "stopped_step_count": stopped_steps,
        "redirected_step_count": redirected_steps,
        "near_miss_event_count": near_miss_event_count,
        "collision_response_event_count": response_count,
        "robot_scene_contact_event_count": scene_contact_count,
        "simulated_step_count": bounded_steps,
        "cycle_time_seconds": round(bounded_steps * model_timestep_s, 6)
        if model_timestep_s
        else None,
        "success_criteria": {
            "goal_reached_within_tolerance": goal_reached,
            "goal_tolerance_m": TASK_GOAL_TOLERANCE_M,
            "no_committed_scene_collision_contacts": scene_contact_count == 0,
            "no_clearance_near_miss": not clearance_threshold_violation,
            "no_fall_detected": not fall_detected,
            "no_stuck_or_no_progress": not stuck_detected,
            "endpoint_clean": endpoint_clean,
        },
        "proof_boundary": (
            "Task outcome is computed from deterministic MuJoCo preview state, contact "
            "probes, and route traces. It is not physical policy ranking."
        ),
    }


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
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
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


def _manipulation_ready_arm_joint_deltas(arm: str = "both") -> dict[str, float]:
    selection = str(arm or "both").strip().lower()
    if selection not in MANIPULATION_READY_ARM_SELECTIONS:
        raise ValueError(f"unknown manipulation arm selection: {arm!r}")
    sides = ("left", "right") if selection == "both" else (selection,)
    out: dict[str, float] = {}
    for side in sides:
        out.update(MANIPULATION_READY_ARM_JOINT_DELTAS[side])
    return out


def _apply_manipulation_ready_arm_pose(
    *,
    qpos: Any,
    base_qpos: Any,
    joint_addresses: Mapping[str, int],
    arm: str = "both",
) -> list[str]:
    applied: list[str] = []
    for name, delta in _manipulation_ready_arm_joint_deltas(arm).items():
        address = joint_addresses.get(name)
        if address is None or address >= len(qpos) or address >= len(base_qpos):
            continue
        qpos[address] = base_qpos[address] + float(delta)
        applied.append(name)
    return applied


def _is_robot_pov_self_occluding_body_name(body_name: str) -> bool:
    name = str(body_name or "").lower()
    if not name or name == "world":
        return False
    return not any(part in name for part in ROBOT_POV_VISIBLE_SELF_BODY_PARTS)


def _robot_pov_self_occluding_geom_ids(model: Any, mujoco_module: Any) -> list[int]:
    ids: list[int] = []
    for geom_id in range(int(getattr(model, "ngeom", 0) or 0)):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (
            mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if _is_robot_pov_self_occluding_body_name(body_name):
            ids.append(geom_id)
    return ids


def _set_geom_alpha(model: Any, geom_ids: Sequence[int], alpha: float) -> list[tuple[int, float]]:
    previous: list[tuple[int, float]] = []
    for geom_id in geom_ids:
        if 0 <= int(geom_id) < len(model.geom_rgba):
            previous.append((int(geom_id), float(model.geom_rgba[int(geom_id), 3])))
            model.geom_rgba[int(geom_id), 3] = float(alpha)
    return previous


def _restore_geom_alpha(model: Any, previous: Sequence[tuple[int, float]]) -> None:
    for geom_id, alpha in previous:
        if 0 <= int(geom_id) < len(model.geom_rgba):
            model.geom_rgba[int(geom_id), 3] = float(alpha)


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
    manipulation_ready_arms: bool = False,
    manipulation_reach_arm: str = "both",
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
    if manipulation_ready_arms:
        _apply_manipulation_ready_arm_pose(
            qpos=data.qpos,
            base_qpos=base_qpos,
            joint_addresses=joint_addresses,
            arm=manipulation_reach_arm,
        )


def _configure_robot_pov_camera(
    camera: Any,
    *,
    pose: Sequence[float],
    yaw: float,
    manipulation_ready_arms: bool = False,
) -> dict[str, Any]:
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    if manipulation_ready_arms:
        forward_x = math.cos(float(yaw))
        forward_y = math.sin(float(yaw))
        camera.lookat[:] = [x + forward_x * 0.85, y + forward_y * 0.85, z + 0.20]
        camera.distance = 0.92
        camera.azimuth = math.degrees(float(yaw))
        camera.elevation = -18
        return {
            "camera_mode": "virtual_manipulation_pov_near_head_aimed_at_workspace",
            "azimuth": camera.azimuth,
            "distance": camera.distance,
            "elevation": camera.elevation,
            "fallback_used": False,
        }
    camera.lookat[:] = [x, y, z + 0.75]
    camera.distance = 2.15
    camera.azimuth = math.degrees(float(yaw)) + 180.0
    camera.elevation = -14
    return {
        "camera_mode": "virtual_free_camera_following_g1_root_not_physical_robot_sensor",
        "azimuth": camera.azimuth,
        "distance": camera.distance,
        "elevation": camera.elevation,
        "fallback_used": False,
    }


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
    manipulation_ready_arms: bool = False,
    manipulation_reach_arm: str = "both",
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
        manipulation_ready_arms=manipulation_ready_arms,
        manipulation_reach_arm=manipulation_reach_arm,
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
    scene_distance_samples = [
        float(distance)
        for distance in (
            _number(record.get("distance"))
            for record in [*scene_contacts, *rejected_scene_contacts]
        )
        if distance is not None
    ]
    min_clearance_m = (
        round(max(0.0, min(scene_distance_samples)), 6)
        if scene_distance_samples
        else None
    )
    near_miss_event_count = len(rejected_scene_contacts)
    clearance_threshold_violation = bool(
        min_clearance_m is not None and min_clearance_m < TASK_CLEARANCE_THRESHOLD_M
    )
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
        "near_miss_event_count": near_miss_event_count,
        "min_clearance_m": min_clearance_m,
        "clearance_threshold_m": TASK_CLEARANCE_THRESHOLD_M,
        "clearance_threshold_violation": clearance_threshold_violation,
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
    obj_texture_summary = dict(mesh_info.get("obj_texture_material_summary") or {})
    glb_summary = dict(mesh_info.get("visual_asset_summary") or {})
    obj_map_kd_texture_present = bool(obj_texture_summary.get("texture_exists"))
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
        "obj_map_kd_texture_present": obj_map_kd_texture_present,
        "obj_map_kd_texture_file": obj_texture_summary.get("map_kd_texture_file"),
        "mujoco_scene_material_mode": "pbr_texture_bound"
        if obj_map_kd_texture_present
        else "flat_grey_override",
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
        "frames": [dict(frame) for frame in frames if isinstance(frame, Mapping)],
        "overview_frames": groups["overview"],
        "robot_pov_frames": groups["sim_robot_follow_pov"],
        "side_frames": groups["side"],
        **videos,
        "blank_scene_checks": blank_scene_checks,
        "texture_material_evidence": texture_material_evidence,
        "limitations": limitations,
    }


def _relative_path(base_dir: Path, path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except Exception:
        return str(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _file_artifact(path: Path, *, base_dir: Path, required: bool = True) -> dict[str, Any]:
    present = path.is_file()
    return {
        "path": _relative_path(base_dir, path),
        "absolute_path": str(path),
        "required": required,
        "present": present,
        "sha256": _sha256(path) if present else None,
        "size_bytes": path.stat().st_size if present else None,
    }


def _write_visual_media_coverage_manifest(
    *,
    output_root: Path,
    generated_at: str,
    required_scenario_eval_run_ids: Sequence[str],
    visual_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_root / "mujoco_batch_visual_media_coverage.json"
    frames = [
        dict(frame)
        for frame in visual_artifacts.get("frames", []) or []
        if isinstance(frame, Mapping)
    ]
    frames_by_run_camera: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for frame in frames:
        run_id = _string(frame.get("scenario_eval_run_id"))
        camera = _string(frame.get("camera"))
        if not run_id or not camera:
            continue
        frames_by_run_camera.setdefault(run_id, {}).setdefault(camera, []).append(frame)
    video_artifacts = {
        "overview": _mapping(visual_artifacts.get("overview_video")),
        "sim_robot_follow_pov": _mapping(visual_artifacts.get("robot_pov_video")),
        "side": _mapping(visual_artifacts.get("side_video")),
    }
    required_ids = [_string(run_id) for run_id in required_scenario_eval_run_ids if _string(run_id)]
    rows: list[dict[str, Any]] = []
    for run_id in required_ids:
        camera_frames = frames_by_run_camera.get(run_id, {})
        frame_counts = {
            camera: len(camera_frames.get(camera, []))
            for camera in ("overview", "sim_robot_follow_pov", "side")
        }
        frame_paths = {
            camera: [
                _string(frame.get("path"))
                for frame in camera_frames.get(camera, [])
                if _string(frame.get("path"))
            ]
            for camera in ("overview", "sim_robot_follow_pov", "side")
        }
        camera_frame_coverage = {
            camera: frame_counts[camera] > 0
            for camera in ("overview", "sim_robot_follow_pov", "side")
        }
        video_bindings = {
            camera: {
                "status": video_artifacts[camera].get("status"),
                "path": video_artifacts[camera].get("path"),
                "video_contains_run": bool(
                    camera_frame_coverage[camera]
                    and video_artifacts[camera].get("status") == "complete"
                    and video_artifacts[camera].get("path")
                ),
            }
            for camera in ("overview", "sim_robot_follow_pov", "side")
        }
        missing_reasons: list[str] = []
        for camera, covered in camera_frame_coverage.items():
            if not covered:
                missing_reasons.append(f"{camera}_frames_missing")
        for camera, binding in video_bindings.items():
            if not binding["video_contains_run"]:
                missing_reasons.append(f"{camera}_video_missing_or_incomplete")
        rows.append(
            {
                "scenario_eval_run_id": run_id,
                "status": "complete" if not missing_reasons else "incomplete",
                "frame_counts": frame_counts,
                "frame_paths": frame_paths,
                "camera_frame_coverage": camera_frame_coverage,
                "video_bindings": video_bindings,
                "robot_pov_frames_present": camera_frame_coverage["sim_robot_follow_pov"],
                "third_person_frames_present": bool(
                    camera_frame_coverage["overview"] and camera_frame_coverage["side"]
                ),
                "robot_pov_video_present": bool(
                    video_bindings["sim_robot_follow_pov"]["video_contains_run"]
                ),
                "third_person_video_present": bool(
                    video_bindings["overview"]["video_contains_run"]
                    and video_bindings["side"]["video_contains_run"]
                ),
                "missing_reasons": missing_reasons,
            }
        )
    missing_rows = [row for row in rows if row["status"] != "complete"]
    extra_rendered_run_ids = sorted(set(frames_by_run_camera) - set(required_ids))
    manifest = {
        "schema_version": "mujoco_g1_batch_visual_media_coverage.v1",
        "generated_at": generated_at,
        "status": "completed" if rows and not missing_rows else "incomplete",
        "required_scenario_eval_run_count": len(required_ids),
        "required_scenario_eval_run_ids": required_ids,
        "rendered_scenario_eval_run_ids": sorted(frames_by_run_camera),
        "rendered_scenario_eval_run_count": len(frames_by_run_camera),
        "missing_visual_media_run_count": len(missing_rows),
        "missing_visual_media_scenario_eval_run_ids": [
            row["scenario_eval_run_id"] for row in missing_rows
        ],
        "extra_rendered_scenario_eval_run_ids": extra_rendered_run_ids,
        "all_required_runs_have_visual_recording": bool(rows and not missing_rows),
        "all_required_runs_have_robot_pov_video": bool(
            rows and all(row["robot_pov_video_present"] for row in rows)
        ),
        "all_required_runs_have_third_person_video": bool(
            rows and all(row["third_person_video_present"] for row in rows)
        ),
        "video_artifacts": video_artifacts,
        "visual_limitations": list(visual_artifacts.get("limitations") or []),
        "rows": rows,
        "claim_boundary": (
            "This is simulated MuJoCo visual media coverage. It proves only which "
            "scenario_eval_run_id rows have rendered simulator POV and third-person "
            "media in this package; it is not physical robot camera evidence."
        ),
    }
    write_json(manifest_path, manifest)
    return manifest


def _metric_coverage(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for attempt in attempts:
        metrics = _mapping(attempt.get("metrics"))
        missing = [key for key in REQUIRED_TASK_METRIC_KEYS if key not in metrics]
        row = {
            "attempt_id": attempt.get("attempt_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "status": attempt.get("status"),
            "success": bool(attempt.get("success")),
            "task_success": bool(attempt.get("task_success")),
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "missing_metric_keys": missing,
            "metrics": {key: metrics.get(key) for key in REQUIRED_TASK_METRIC_KEYS},
        }
        rows.append(row)
        if missing:
            missing_rows.append(row)
    return {
        "required_metric_keys": list(REQUIRED_TASK_METRIC_KEYS),
        "attempt_metric_rows": rows,
        "attempt_metric_row_count": len(rows),
        "missing_metric_row_count": len(missing_rows),
        "missing_metric_rows": missing_rows,
        "metric_coverage_complete": bool(rows) and not missing_rows,
    }


def _sequence3(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    numbers = [_number(value[index]) for index in range(3)]
    if any(number is None for number in numbers):
        return None
    return [float(number) for number in numbers if number is not None]


def _mesh_bounds_summary(mesh_info: Mapping[str, Any]) -> dict[str, Any]:
    bounds = mesh_info.get("bounds")
    lower = upper = None
    if (
        isinstance(bounds, Sequence)
        and not isinstance(bounds, (str, bytes))
        and len(bounds) >= 2
    ):
        lower = _sequence3(bounds[0])
        upper = _sequence3(bounds[1])
    extents = _sequence3(mesh_info.get("extents"))
    if extents is None and lower and upper:
        extents = [upper[index] - lower[index] for index in range(3)]
    positive_extents = bool(extents and all(value > 0 for value in extents))
    volume = (
        round(float(extents[0] * extents[1] * extents[2]), 6)
        if positive_extents and extents is not None
        else None
    )
    return {
        "bounds": [lower, upper] if lower and upper else None,
        "extents_m": [round(value, 6) for value in extents] if extents else None,
        "positive_extents": positive_extents,
        "volume_m3_estimate": volume,
        "scale_evidence_available": positive_extents,
    }


def _int_set(values: Any) -> set[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _sample_ints(values: Any, *, limit: int = 12) -> list[int]:
    return sorted(_int_set(values))[: max(0, limit)]


def _component_coverage_summary(
    component_coverage: Mapping[str, Any],
    *,
    proxy_summary: Mapping[str, Any],
    visible_object_count: int,
) -> dict[str, Any]:
    skipped = _mapping(proxy_summary.get("skipped"))
    source_component_count = int(proxy_summary.get("source_component_count") or 0)
    proxy_count = int(proxy_summary.get("proxy_count") or 0)
    max_proxy_count = int(proxy_summary.get("max_proxy_count") or 0)
    component_mapping_one_to_one = bool(
        visible_object_count and source_component_count == visible_object_count
    )
    return {
        "source_component_count": source_component_count,
        "visible_object_count": visible_object_count,
        "proxy_count": proxy_count,
        "max_proxy_count": max_proxy_count,
        "covered_source_component_count": int(
            component_coverage.get("covered_source_component_count") or 0
        ),
        "reference_floor_covered_source_component_count": len(
            _int_set(
                component_coverage.get(
                    "reference_floor_covered_source_component_indexes"
                )
            )
        ),
        "intentionally_excluded_source_component_count": len(
            _int_set(
                component_coverage.get("intentionally_excluded_source_component_indexes")
            )
        ),
        "truncated_source_component_count": int(
            component_coverage.get("truncated_source_component_count") or 0
        ),
        "uncovered_source_component_count": int(
            component_coverage.get("uncovered_source_component_count") or 0
        ),
        "component_proxy_coverage_complete": bool(
            component_coverage.get("component_proxy_coverage_complete")
        ),
        "component_mapping_status": "one_to_one"
        if component_mapping_one_to_one
        else "not_one_to_one",
        "component_mapping_one_to_one": component_mapping_one_to_one,
        "sample_covered_source_component_indexes": _sample_ints(
            component_coverage.get("covered_source_component_indexes")
        ),
        "sample_reference_floor_source_component_indexes": _sample_ints(
            component_coverage.get("reference_floor_covered_source_component_indexes")
        ),
        "sample_intentionally_excluded_source_component_indexes": _sample_ints(
            component_coverage.get("intentionally_excluded_source_component_indexes")
        ),
        "sample_truncated_source_component_indexes": _sample_ints(
            component_coverage.get("truncated_source_component_indexes")
        ),
        "sample_uncovered_source_component_indexes": _sample_ints(
            component_coverage.get("uncovered_source_component_indexes")
        ),
        "skipped_component_counts": skipped,
        "boundary": (
            "Coverage counts are over split collision-proxy components. A visible GLB "
            "object is only considered directly mapped when the visible-object count "
            "and proxy source-component count match one-to-one."
        ),
    }


def _artifact_ref(
    artifact_refs: Mapping[str, Any],
    key: str,
    *,
    json_pointer: str | None = None,
    object_id: str | None = None,
) -> dict[str, Any] | None:
    path = _string(artifact_refs.get(key))
    if not path:
        return None
    ref: dict[str, Any] = {"kind": key, "path": path}
    if json_pointer:
        ref["json_pointer"] = json_pointer
    if object_id:
        ref["object_id"] = object_id
    return ref


def _artifact_refs(
    artifact_refs: Mapping[str, Any],
    keys: Sequence[tuple[str, str | None]],
    *,
    object_id: str | None = None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for key, json_pointer in keys:
        ref = _artifact_ref(
            artifact_refs,
            key,
            json_pointer=json_pointer,
            object_id=object_id,
        )
        if ref:
            refs.append(ref)
    return refs


def _visible_object_identity_payload(visible_object: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_id": _string(visible_object.get("object_id")),
        "name": _string(visible_object.get("name")),
        "source_component_index": visible_object.get("source_component_index"),
        "semantic_label_available": bool(
            visible_object.get("semantic_label_available")
        ),
        "geometry_name": visible_object.get("geometry_name"),
        "material_name": visible_object.get("material_name"),
        "gltf_mesh_index": visible_object.get("gltf_mesh_index"),
        "gltf_mesh_name": visible_object.get("gltf_mesh_name"),
        "bounds": visible_object.get("bounds"),
        "extents": visible_object.get("extents"),
        "volume_m3_estimate": visible_object.get("volume_m3_estimate"),
    }


def _component_membership_reason(
    component_index: int | None,
    component_coverage: Mapping[str, Any],
) -> str:
    if component_index is None:
        return "visible_object_missing_source_component_index"
    if component_index in _int_set(component_coverage.get("covered_source_component_indexes")):
        return "source_component_has_collision_proxy"
    if component_index in _int_set(
        component_coverage.get("reference_floor_covered_source_component_indexes")
    ):
        return "source_component_covered_by_reference_floor"
    if component_index in _int_set(
        component_coverage.get("intentionally_excluded_source_component_indexes")
    ):
        return "source_component_intentionally_excluded_from_proxy_generation"
    if component_index in _int_set(component_coverage.get("truncated_source_component_indexes")):
        return "source_component_truncated_by_proxy_limit"
    if component_index in _int_set(component_coverage.get("uncovered_source_component_indexes")):
        return "source_component_uncovered_by_proxy_generation"
    return "source_component_not_found_in_proxy_coverage_sets"


def _missing_semantic_object_records(
    object_semantics_summary: Mapping[str, Any],
    *,
    artifact_refs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for visible_object in object_semantics_summary.get("visible_objects", []) or []:
        if not isinstance(visible_object, Mapping):
            continue
        semantic_label = visible_object.get("semantic_label_available")
        if semantic_label is True:
            continue
        if semantic_label is None and _string(visible_object.get("name")):
            if not _is_generated_semantic_name(_string(visible_object.get("name"))):
                continue
        if semantic_label is None and _string(visible_object.get("gltf_mesh_name")):
            if not _is_generated_semantic_name(
                _string(visible_object.get("gltf_mesh_name"))
            ):
                continue
        if semantic_label is None and _string(visible_object.get("geometry_name")):
            if not _is_generated_semantic_name(
                _string(visible_object.get("geometry_name"))
            ):
                continue
        if semantic_label is None and _string(visible_object.get("material_name")):
            continue
        object_id = _string(visible_object.get("object_id"))
        record = {
            **_visible_object_identity_payload(visible_object),
            "missing_semantics": [
                "non_generated_visible_object_name",
                "semantic_label_or_material_object_reference",
            ],
            "reason": "visible_glb_object_has_only_generated_or_missing_semantic_name",
            "available_name_candidates": {
                "name": visible_object.get("name"),
                "geometry_name": visible_object.get("geometry_name"),
                "gltf_mesh_name": visible_object.get("gltf_mesh_name"),
                "material_name": visible_object.get("material_name"),
                "gltf_material_indexes": visible_object.get("gltf_material_indexes"),
            },
            "evidence_refs": _artifact_refs(
                artifact_refs,
                (
                    (
                        "scene_load_trace",
                        "/mesh_info/visual_object_semantics_summary/visible_objects",
                    ),
                    ("source_scene_glb", None),
                    ("converted_scene_obj", None),
                ),
                object_id=object_id,
            ),
        }
        missing.append(record)
    return missing


def _capture_object_semantics_summary(capture_root: Path) -> dict[str, Any]:
    object_index_path = capture_root / "raw" / "object_index.json"
    task_targets_path = capture_root / "pipeline" / "task_targets.json"
    object_index = (
        _mapping(read_json_any(object_index_path)) if object_index_path.is_file() else {}
    )
    task_targets = (
        _mapping(read_json_any(task_targets_path)) if task_targets_path.is_file() else {}
    )
    objects = [
        dict(item)
        for item in object_index.get("objects", []) or []
        if isinstance(item, Mapping)
    ]
    target_entries = [
        dict(item)
        for item in task_targets.get("object_index_entries", []) or []
        if isinstance(item, Mapping)
    ]

    def compact_object(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "object_id": item.get("object_id"),
            "label": item.get("label") or item.get("category"),
            "confidence": item.get("confidence"),
            "reference_crop": item.get("reference_crop"),
        }

    return {
        "status": "available" if objects or target_entries else "not_available",
        "object_index_path": str(object_index_path) if object_index_path.is_file() else None,
        "task_targets_path": str(task_targets_path) if task_targets_path.is_file() else None,
        "capture_object_count": len(objects),
        "task_target_object_entry_count": len(target_entries),
        "sample_capture_objects": [compact_object(item) for item in objects[:12]],
        "sample_task_target_objects": [compact_object(item) for item in target_entries[:12]],
        "explicit_target_object_ids": [
            _string(item)
            for item in task_targets.get("explicit_target_object_ids", []) or []
            if _string(item)
        ],
        "semantic_link_to_mujoco_visual_objects_verified": False,
        "proof_boundary": (
            "Capture object-index labels are raw/capture-side semantic evidence. They "
            "are supporting refs only until a verified 2D/3D join links each label to "
            "a generated MuJoCo/GLB visible object."
        ),
    }


def _visual_object_physics_coverage(
    *,
    object_semantics_summary: Mapping[str, Any],
    collision_summary: Mapping[str, Any],
    proxy_summary: Mapping[str, Any],
    collider_loaded: bool,
    artifact_refs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_refs = _mapping(artifact_refs)
    visible_objects = [
        dict(item)
        for item in object_semantics_summary.get("visible_objects", []) or []
        if isinstance(item, Mapping)
    ]
    if not visible_objects:
        return {
            "status": "blocked",
            "coverage_complete": False,
            "reason": "visible_object_semantics_missing",
            "visible_object_count": 0,
            "covered_visible_object_count": 0,
            "missing_physics_object_ids": [],
            "missing_physics_objects": [],
        }

    if collider_loaded and bool(collision_summary.get("scene_collision_mesh_geom_enabled")):
        return {
            "status": "complete",
            "coverage_complete": True,
            "coverage_method": "full_scene_collision_mesh_geom",
            "visible_object_count": len(visible_objects),
            "covered_visible_object_count": len(visible_objects),
            "covered_visible_object_ids": [
                _string(item.get("object_id")) for item in visible_objects
            ],
            "missing_physics_object_ids": [],
            "missing_physics_objects": [],
            "boundary": (
                "The full converted scene mesh is loaded as collision geometry, so every "
                "visible object in the converted scene has at least mesh-level physics "
                "coverage."
            ),
        }

    component_coverage = _mapping(proxy_summary.get("component_coverage"))
    covered_component_indexes = _int_set(
        component_coverage.get("covered_source_component_indexes")
    )
    reference_floor_indexes = _int_set(
        component_coverage.get("reference_floor_covered_source_component_indexes")
    )
    physics_supported_indexes = covered_component_indexes | reference_floor_indexes
    coverage_summary = _component_coverage_summary(
        component_coverage,
        proxy_summary=proxy_summary,
        visible_object_count=len(visible_objects),
    )
    component_mapping_one_to_one = bool(
        coverage_summary.get("component_mapping_one_to_one")
    )
    missing_objects: list[dict[str, Any]] = []
    covered_objects: list[dict[str, Any]] = []
    unmapped_objects: list[dict[str, Any]] = []
    missing_physics_object_records: list[dict[str, Any]] = []
    for visible_object in visible_objects:
        component_index = visible_object.get("source_component_index")
        try:
            normalized_component_index = int(component_index)
        except (TypeError, ValueError):
            normalized_component_index = None
        membership_reason = _component_membership_reason(
            normalized_component_index,
            component_coverage,
        )
        if normalized_component_index is None:
            unmapped_objects.append(visible_object)
            missing_objects.append(visible_object)
            coverage_reason = "visible_object_missing_source_component_index"
        elif not component_mapping_one_to_one:
            missing_objects.append(visible_object)
            coverage_reason = "visible_object_proxy_component_mapping_not_one_to_one"
        elif normalized_component_index in physics_supported_indexes:
            covered_objects.append(visible_object)
            continue
        else:
            missing_objects.append(visible_object)
            coverage_reason = membership_reason
        object_id = _string(visible_object.get("object_id"))
        missing_physics_object_records.append(
            {
                **_visible_object_identity_payload(visible_object),
                "coverage_reason": coverage_reason,
                "source_component_membership": membership_reason,
                "component_mapping_status": coverage_summary.get(
                    "component_mapping_status"
                ),
                "component_mapping_one_to_one": component_mapping_one_to_one,
                "component_coverage_summary": coverage_summary,
                "evidence_refs": _artifact_refs(
                    artifact_refs,
                    (
                        (
                            "scene_load_trace",
                            "/mesh_info/visual_object_semantics_summary/visible_objects",
                        ),
                        (
                            "scene_load_trace",
                            "/mesh_info/collision_proxy_summary/component_coverage",
                        ),
                        ("source_scene_glb", None),
                        ("converted_scene_obj", None),
                    ),
                    object_id=object_id,
                ),
            }
        )

    proxy_model_enabled = bool(collision_summary.get("scene_collision_proxy_geoms_enabled"))
    proxy_generation_complete = bool(
        component_coverage.get("component_proxy_coverage_complete")
    )
    coverage_complete = bool(
        collider_loaded
        and proxy_model_enabled
        and proxy_generation_complete
        and not missing_objects
    )
    return {
        "status": "complete" if coverage_complete else "blocked",
        "coverage_complete": coverage_complete,
        "coverage_method": "component_aabb_proxy_geoms"
        if proxy_model_enabled
        else "missing_scene_collision_geometry",
        "visible_object_count": len(visible_objects),
        "covered_visible_object_count": len(covered_objects),
        "covered_visible_object_ids": [
            _string(item.get("object_id")) for item in covered_objects
        ],
        "missing_physics_object_ids": [
            _string(item.get("object_id")) for item in missing_objects
        ],
        "missing_physics_objects": missing_physics_object_records,
        "unmapped_visible_object_ids": [
            _string(item.get("object_id")) for item in unmapped_objects
        ],
        "component_coverage_summary": coverage_summary,
        "component_coverage": component_coverage,
        "proxy_collision_model_used": proxy_model_enabled,
        "proxy_generation_complete": proxy_generation_complete,
        "boundary": (
            "Component proxy coverage proves each named visible component has a "
            "corresponding conservative physics proxy or reference-floor collider. It "
            "does not prove exact mesh collision fidelity or rank fidelity."
        ),
    }


def _build_digital_twin_fidelity_qa(
    *,
    generated_at: str,
    mesh_info: Mapping[str, Any],
    collision_summary: Mapping[str, Any],
    visual_artifacts: Mapping[str, Any],
    artifact_refs: Mapping[str, Any] | None = None,
    capture_object_semantics_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_refs = _mapping(artifact_refs)
    capture_object_semantics_summary = _mapping(capture_object_semantics_summary)
    visual_summary = _mapping(mesh_info.get("visual_asset_summary"))
    object_semantics_summary = _mapping(mesh_info.get("visual_object_semantics_summary"))
    obj_summary = _mapping(mesh_info.get("obj_vertex_color_summary"))
    proxy_summary = _mapping(mesh_info.get("collision_proxy_summary"))
    texture_material_evidence = _mapping(visual_artifacts.get("texture_material_evidence"))
    blank_scene_checks = _mapping(visual_artifacts.get("blank_scene_checks"))
    bounds_summary = _mesh_bounds_summary(mesh_info)
    proxy_count = int(collision_summary.get("scene_collision_proxy_geom_count") or 0)
    source_component_count = int(proxy_summary.get("source_component_count") or 0)
    max_proxy_count = int(proxy_summary.get("max_proxy_count") or 0)
    proxy_generation_truncated = bool(
        max_proxy_count and source_component_count > max_proxy_count and proxy_count >= max_proxy_count
    )
    material_truth_present = bool(
        obj_summary.get("has_vertex_rgb")
        or visual_summary.get("has_embedded_or_referenced_image_textures")
        or int(visual_summary.get("materials_count") or 0) > 0
    )
    blank_scene_passed = bool(
        blank_scene_checks.get("status") in {"not_applicable", "checked"}
        and blank_scene_checks.get("all_frames_nonblank", True) is not False
    )
    visible_collision_alignment_validated = bool(
        collision_summary.get("visible_scene_collision_alignment_validated")
    )
    collider_loaded = bool(collision_summary.get("collision_geometry_loaded"))
    collider_coverage_available = bool(
        collider_loaded
        and (
            collision_summary.get("scene_collision_mesh_geom_enabled")
            or collision_summary.get("scene_collision_proxy_geoms_enabled")
        )
    )
    object_semantics_available = bool(
        object_semantics_summary.get("status") == "available"
        and int(object_semantics_summary.get("visible_object_count") or 0) > 0
        and int(object_semantics_summary.get("named_visible_object_count") or 0) > 0
    )
    missing_semantic_objects = _missing_semantic_object_records(
        object_semantics_summary,
        artifact_refs=artifact_refs,
    )
    object_semantics_evidence = {
        **object_semantics_summary,
        "missing_semantic_objects": missing_semantic_objects,
        "capture_object_semantics_summary": capture_object_semantics_summary,
        "semantic_link_to_capture_object_index_verified": False,
    }
    visual_object_physics_coverage = _visual_object_physics_coverage(
        object_semantics_summary=object_semantics_summary,
        collision_summary=collision_summary,
        proxy_summary=proxy_summary,
        collider_loaded=collider_loaded,
        artifact_refs=artifact_refs,
    )
    visible_objects_have_physics_coverage = bool(
        object_semantics_available
        and visual_object_physics_coverage.get("coverage_complete")
    )
    visual_object_has_matching_physics = bool(
        visible_collision_alignment_validated or visible_objects_have_physics_coverage
    )
    component_coverage = _mapping(proxy_summary.get("component_coverage"))
    full_mesh_collision_loaded = bool(
        collider_loaded and collision_summary.get("scene_collision_mesh_geom_enabled")
    )
    component_coverage_summary = _component_coverage_summary(
        component_coverage,
        proxy_summary=proxy_summary,
        visible_object_count=int(object_semantics_summary.get("visible_object_count") or 0),
    )
    hidden_obstacle_risk_reviewed = bool(
        collider_coverage_available
        and (
            full_mesh_collision_loaded
            or (
                visible_objects_have_physics_coverage
                and component_coverage.get("component_proxy_coverage_complete") is True
                and int(component_coverage.get("uncovered_source_component_count") or 0) == 0
                and int(component_coverage.get("truncated_source_component_count") or 0) == 0
            )
        )
        and not proxy_generation_truncated
    )
    gates = {
        "scale_bounds_available": {
            "passed": bounds_summary["scale_evidence_available"],
            "evidence": bounds_summary,
        },
        "texture_material_truth_available": {
            "passed": material_truth_present,
            "evidence": {
                "visual_asset_summary": visual_summary,
                "obj_vertex_color_summary": obj_summary,
                "texture_material_evidence": texture_material_evidence,
            },
        },
        "nonblank_visual_evidence": {
            "passed": blank_scene_passed,
            "evidence": blank_scene_checks,
        },
        "collider_coverage_available": {
            "passed": collider_coverage_available,
            "evidence": {
                "collision_geometry_loaded": collider_loaded,
                "scene_collision_mesh_geom_enabled": collision_summary.get(
                    "scene_collision_mesh_geom_enabled"
                ),
                "scene_collision_proxy_geoms_enabled": collision_summary.get(
                    "scene_collision_proxy_geoms_enabled"
                ),
                "scene_collision_proxy_geom_count": proxy_count,
                "collision_proxy_summary": proxy_summary,
            },
        },
        "object_semantics_available": {
            "passed": object_semantics_available,
            "evidence": object_semantics_evidence,
        },
        "visible_objects_have_physics_coverage": {
            "passed": visible_objects_have_physics_coverage,
            "evidence": visual_object_physics_coverage,
        },
        "visual_object_has_matching_physics": {
            "passed": visual_object_has_matching_physics,
            "evidence": {
                "visible_scene_collision_alignment_validated": (
                    visible_collision_alignment_validated
                ),
                "visible_objects_have_physics_coverage": (
                    visible_objects_have_physics_coverage
                ),
                "visual_object_physics_coverage": visual_object_physics_coverage,
                "proxy_collision_model_used": collision_summary.get(
                    "proxy_collision_model_used"
                ),
                "boundary": (
                    "Full scene mesh collision proves visual/physics alignment most "
                    "directly. Component proxy coverage can prove each visible object has "
                    "a conservative matching physics body, but it remains a simulator QA "
                    "claim rather than generated-world rank fidelity."
                ),
            },
        },
        "hidden_obstacle_risk_reviewed": {
            "passed": hidden_obstacle_risk_reviewed,
            "evidence": {
                "proxy_generation_truncated": proxy_generation_truncated,
                "source_component_count": source_component_count,
                "proxy_count": proxy_count,
                "max_proxy_count": max_proxy_count,
                "skipped_components": proxy_summary.get("skipped") or {},
                "component_coverage_summary": component_coverage_summary,
                "component_coverage": component_coverage,
                "evidence_refs": _artifact_refs(
                    artifact_refs,
                    (
                        (
                            "scene_load_trace",
                            "/mesh_info/collision_proxy_summary/component_coverage",
                        ),
                        ("source_scene_glb", None),
                        ("converted_scene_obj", None),
                    ),
                ),
            },
        },
    }
    blockers: list[str] = []
    if not gates["scale_bounds_available"]["passed"]:
        blockers.append("digital_twin_scale_bounds_missing")
    if not gates["texture_material_truth_available"]["passed"]:
        blockers.append("digital_twin_texture_material_truth_missing")
    if not gates["nonblank_visual_evidence"]["passed"]:
        blockers.append("digital_twin_visual_frames_blank_or_missing")
    if not gates["collider_coverage_available"]["passed"]:
        blockers.append("digital_twin_collider_coverage_missing")
    if not gates["object_semantics_available"]["passed"]:
        blockers.append("digital_twin_object_semantics_missing")
    if not gates["visible_objects_have_physics_coverage"]["passed"]:
        blockers.append("visible_objects_without_physics_coverage")
    if not gates["visual_object_has_matching_physics"]["passed"]:
        blockers.append("visual_collision_alignment_not_validated")
    if not gates["hidden_obstacle_risk_reviewed"]["passed"]:
        blockers.append("hidden_obstacle_or_proxy_truncation_review_required")
    machine_fidelity_audit_complete = all(
        bool(gates[gate_id]["passed"])
        for gate_id in (
            "scale_bounds_available",
            "texture_material_truth_available",
            "nonblank_visual_evidence",
            "collider_coverage_available",
            "object_semantics_available",
            "visible_objects_have_physics_coverage",
            "hidden_obstacle_risk_reviewed",
        )
    )
    robot_team_grade_fidelity_passed = bool(
        machine_fidelity_audit_complete
        and gates["visual_object_has_matching_physics"]["passed"]
    )
    visual_collision_alignment_gap = {
        "status": "passed"
        if gates["visual_object_has_matching_physics"]["passed"]
        else "review_required",
        "visible_scene_collision_alignment_validated": visible_collision_alignment_validated,
        "visible_objects_have_physics_coverage": visible_objects_have_physics_coverage,
        "reason": "alignment_validated_by_full_collision_mesh_or_complete_component_proxy_coverage"
        if gates["visual_object_has_matching_physics"]["passed"]
        else "neither_full_mesh_collision_nor_complete_component_proxy_coverage_validated",
        "evidence_refs": _artifact_refs(
            artifact_refs,
            (
                ("scene_load_trace", "/collision_summary"),
                (
                    "scene_load_trace",
                    "/mesh_info/collision_proxy_summary/component_coverage",
                ),
                ("source_scene_glb", None),
                ("converted_scene_obj", None),
            ),
        ),
        "boundary": (
            "This is simulator visual/physics alignment evidence only. It is not "
            "physical collision validation or real-world safety evidence."
        ),
    }
    hidden_obstacle_gap = {
        "status": "passed" if hidden_obstacle_risk_reviewed else "review_required",
        "proxy_generation_truncated": proxy_generation_truncated,
        "source_component_count": source_component_count,
        "proxy_count": proxy_count,
        "max_proxy_count": max_proxy_count,
        "component_coverage_summary": component_coverage_summary,
        "reason": "all_proxy_components_covered_or_full_mesh_collision_loaded"
        if hidden_obstacle_risk_reviewed
        else "proxy_generation_truncated_or_split_components_uncovered",
        "evidence_refs": _artifact_refs(
            artifact_refs,
            (
                (
                    "scene_load_trace",
                    "/mesh_info/collision_proxy_summary/component_coverage",
                ),
                ("source_scene_glb", None),
                ("converted_scene_obj", None),
            ),
        ),
        "boundary": (
            "Proxy truncation review can support simulator QA only; it does not "
            "approve physical navigation or safety."
        ),
    }
    return {
        "schema_version": MUJOCO_G1_DIGITAL_TWIN_FIDELITY_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if robot_team_grade_fidelity_passed else "review_required",
        "machine_fidelity_audit_complete": machine_fidelity_audit_complete,
        "robot_team_grade_fidelity_passed": robot_team_grade_fidelity_passed,
        "blockers": sorted(set(blockers)),
        "object_level_fidelity_gaps": {
            "missing_semantic_objects": missing_semantic_objects,
            "visible_objects_without_physics_coverage": (
                visual_object_physics_coverage.get("missing_physics_objects", [])
            ),
            "hidden_obstacle_or_proxy_truncation_review": hidden_obstacle_gap,
            "visual_collision_alignment": visual_collision_alignment_gap,
            "capture_object_semantics_supporting_refs": capture_object_semantics_summary,
            "boundary": (
                "Object-level gap records identify simulator QA gaps. They do not "
                "turn capture-side object labels into generated GLB semantics unless "
                "a verified link is present."
            ),
        },
        "gates": gates,
        "mesh_info_summary": {
            "source_glb": mesh_info.get("source_glb"),
            "converted_obj": mesh_info.get("converted_obj"),
            "vertices": mesh_info.get("vertices"),
            "faces": mesh_info.get("faces"),
            "visual_object_semantics_summary": object_semantics_evidence,
            **bounds_summary,
        },
        "visual_collision_parity": {
            "visual_mesh_collisions_enabled": collision_summary.get(
                "scene_visual_mesh_collisions_enabled"
            ),
            "visible_scene_collision_alignment_validated": (
                visible_collision_alignment_validated
            ),
            "proxy_collision_model_used": collision_summary.get("proxy_collision_model_used"),
            "scene_collision_proxy_geom_count": proxy_count,
        },
        "claim_boundary": (
            "This audit checks simulator digital-twin fidelity evidence. It does not "
            "upgrade generated-world rank fidelity, customer off-scope validation, or real-world "
            "deployment claims."
        ),
    }


def _write_mujoco_batch_trace_package(
    *,
    output_root: Path,
    generated_at: str,
    attempts: Sequence[Mapping[str, Any]],
    full_contact_trace: Sequence[Mapping[str, Any]],
    full_collision_probe_trace: Sequence[Mapping[str, Any]],
    full_collision_response_events: Sequence[Mapping[str, Any]],
    required_scenario_eval_run_ids: Sequence[str],
    covered_scenario_eval_run_ids: Sequence[str],
    missing_scenario_eval_run_ids: Sequence[str],
    duplicate_scenario_eval_run_ids: Sequence[str],
    scenario_eval_run_coverage_complete: bool,
    visual_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_trace_path = output_root / "mujoco_batch_attempt_trace.jsonl"
    contact_stream_path = output_root / "mujoco_batch_contact_stream.jsonl"
    planner_state_path = output_root / "mujoco_batch_planner_state.jsonl"
    control_stream_path = output_root / "mujoco_batch_control_stream.jsonl"
    metrics_path = output_root / "mujoco_batch_metrics.json"
    failure_labels_path = output_root / "mujoco_batch_failure_labels.json"
    visual_review_ledger_path = output_root / "mujoco_batch_visual_review_ledger.json"
    checksums_path = output_root / "mujoco_batch_artifact_checksums.json"
    manifest_path = output_root / "mujoco_batch_trace_package_manifest.json"
    visual_media_coverage_manifest = _write_visual_media_coverage_manifest(
        output_root=output_root,
        generated_at=generated_at,
        required_scenario_eval_run_ids=required_scenario_eval_run_ids,
        visual_artifacts=visual_artifacts,
    )
    visual_media_coverage_path = output_root / "mujoco_batch_visual_media_coverage.json"

    def _label_for_failed_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
        task_outcome = _mapping(attempt.get("task_outcome"))
        failure_mode_ids = list(
            attempt.get("failure_mode_ids")
            or task_outcome.get("failure_mode_ids")
            or []
        )
        artifact_paths = _mapping(attempt.get("artifact_paths"))
        evidence_refs: list[dict[str, Any]] = []
        for key in (
            "scene_trace",
            "spawn_trace",
            "policy_trace",
            "sim_robot_pov_evidence",
        ):
            path = artifact_paths.get(key)
            if path:
                evidence_refs.append({"kind": key, "path": path})
        frames = artifact_paths.get("frames")
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            evidence_refs.append(
                {
                    "kind": "rendered_episode_frames",
                    "frame_count": len(frames),
                    "sample_paths": list(frames[:3]),
                }
            )
        criteria_metric_keys = (
            "goal_reached",
            "endpoint_clean",
            "spawn_clean",
            "timeout",
            "fall_detected",
            "stuck_detected",
            "policy_instability_detected",
            "final_target_error_m",
            "goal_tolerance_m",
            "min_clearance_m",
            "clearance_threshold_m",
            "clearance_threshold_violation",
            "robot_scene_contact_event_count",
            "near_miss_event_count",
            "progress_to_goal_ratio",
            "path_efficiency_ratio",
            "cycle_time_seconds",
        )
        return {
            "label_id": f"mujoco_g1_label_{_safe_id(attempt.get('attempt_id'))}",
            "attempt_id": attempt.get("attempt_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get(
                "scenario_variation_instance_id"
            ),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "label": "failure",
            "label_source": "deterministic_mujoco_state_contact_route_trace",
            "status": "deterministically_labeled_failure",
            "task_success": bool(task_outcome.get("task_success")),
            "task_status": task_outcome.get("task_status") or attempt.get("status"),
            "failure_mode_ids": failure_mode_ids,
            "primary_failure_mode": failure_mode_ids[0] if failure_mode_ids else None,
            "failure_reason": attempt.get("failure_reason")
            or task_outcome.get("failure_reason"),
            "criteria_results": {
                "success_criteria": _mapping(task_outcome.get("success_criteria")),
                "metrics": {
                    key: task_outcome.get(key)
                    for key in criteria_metric_keys
                    if key in task_outcome
                },
            },
            "task_outcome": task_outcome,
            "evidence_refs": evidence_refs,
            "review_status": "available_for_human_audit_not_required_for_sim_only_metric",
            "proof_effect": "sim_only_metric_input_not_real_rank_fidelity",
        }

    def _visual_review_for_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
        task_outcome = _mapping(attempt.get("task_outcome"))
        artifact_paths = _mapping(attempt.get("artifact_paths"))
        frames = artifact_paths.get("frames")
        media_refs: list[dict[str, Any]] = []
        for key in ("overview_video", "robot_pov_video", "side_video", "sim_robot_pov_evidence"):
            value = artifact_paths.get(key)
            if value:
                media_refs.append({"kind": key, "path": value})
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            media_refs.append(
                {
                    "kind": "rendered_episode_frames",
                    "frame_count": len(frames),
                    "sample_paths": list(frames[:3]),
                }
            )
        criteria = _mapping(task_outcome.get("success_criteria"))
        if not criteria:
            criteria = {
                "goal_reached": bool(task_outcome.get("goal_reached")),
                "endpoint_clean": bool(task_outcome.get("endpoint_clean")),
                "spawn_clean": bool(task_outcome.get("spawn_clean", True)),
                "no_timeout": not bool(task_outcome.get("timeout")),
                "no_fall_detected": not bool(task_outcome.get("fall_detected")),
                "no_stuck_or_no_progress": not bool(task_outcome.get("stuck_detected")),
                "no_policy_instability": not bool(
                    task_outcome.get("policy_instability_detected")
                ),
                "no_clearance_near_miss": not bool(
                    task_outcome.get("clearance_threshold_violation")
                ),
            }
        success = bool(attempt.get("task_success"))
        return {
            "review_id": f"mujoco_g1_visual_review_{_safe_id(attempt.get('attempt_id'))}",
            "attempt_id": attempt.get("attempt_id"),
            "episode_id": attempt.get("episode_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "decision": "success" if success else "failure",
            "success": success,
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "criteria_results": {
                "success_criteria": criteria,
                "task_outcome": task_outcome,
            },
            "media_refs": media_refs,
            "media_evidence_present": bool(media_refs),
            "confidence": "high" if media_refs else "medium_trace_only",
            "confidence_score": 0.92 if media_refs else 0.72,
            "review_status": "accepted_deterministic_simulator_visual_review",
            "human_review_status": "not_required_for_sim_only_failure_packaging",
            "claim_boundary": (
                "accepted simulator visual review labels success or failure from criteria; "
                "it does not prove the robot can perform the task in the real world"
            ),
        }

    attempt_records = [
        {
            "attempt_id": attempt.get("attempt_id"),
            "episode_id": attempt.get("episode_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "status": attempt.get("status"),
            "success": bool(attempt.get("success")),
            "task_success": bool(attempt.get("task_success")),
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "failure_reason": attempt.get("failure_reason"),
            "deterministic_seed": attempt.get("deterministic_seed"),
            "spawn_pose": attempt.get("spawn_pose"),
            "target_pose": attempt.get("target_pose"),
            "final_pose": attempt.get("final_pose"),
            "route_waypoints": attempt.get("route_waypoints"),
            "task_outcome": attempt.get("task_outcome"),
            "metrics": attempt.get("metrics"),
            "controls": attempt.get("actions") or [],
            "contact_trace": attempt.get("contact_trace") or [],
            "collision_probe_trace": attempt.get("collision_probe_trace") or [],
            "collision_response_events": attempt.get("collision_response_events") or [],
            "artifact_paths": attempt.get("artifact_paths") or {},
            "claim_boundary": attempt.get("claim_boundary"),
        }
        for attempt in attempts
    ]
    contact_records: list[dict[str, Any]] = []
    for record in full_contact_trace:
        contact_records.append({"stream_type": "committed_contact", **dict(record)})
    for record in full_collision_probe_trace:
        contact_records.append({"stream_type": "collision_probe_candidate", **dict(record)})
    for event in full_collision_response_events:
        contact_records.append({"stream_type": "collision_response_event", **dict(event)})
    planner_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    for attempt in attempts:
        route_waypoints = [
            dict(waypoint)
            for waypoint in attempt.get("route_waypoints", []) or []
            if isinstance(waypoint, Mapping)
        ]
        planner_records.append(
            {
                "stream_type": "planner_state",
                "attempt_id": attempt.get("attempt_id"),
                "episode_id": attempt.get("episode_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": attempt.get(
                    "scenario_variation_instance_id"
                ),
                "task_id": attempt.get("task_id"),
                "scenario_id": attempt.get("scenario_id"),
                "deterministic_seed": attempt.get("deterministic_seed"),
                "spawn_pose": attempt.get("spawn_pose"),
                "target_pose": attempt.get("target_pose"),
                "final_pose": attempt.get("final_pose"),
                "route_waypoint_count": len(route_waypoints),
                "route_waypoints": route_waypoints,
                "planner_status": "completed" if route_waypoints else "not_recorded",
                "runtime_route_mutation_allowed": False,
            }
        )
        for index, action in enumerate(attempt.get("actions", []) or []):
            action_payload = dict(action) if isinstance(action, Mapping) else {"value": action}
            control_records.append(
                {
                    "stream_type": "control_action",
                    "attempt_id": attempt.get("attempt_id"),
                    "episode_id": attempt.get("episode_id"),
                    "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                    "scenario_variation_instance_id": attempt.get(
                        "scenario_variation_instance_id"
                    ),
                    "task_id": attempt.get("task_id"),
                    "scenario_id": attempt.get("scenario_id"),
                    "action_index": index,
                    "action": action_payload,
                    "deterministic_seed": attempt.get("deterministic_seed"),
                }
            )
    metric_coverage = _metric_coverage(attempts)
    failed_attempts = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    failure_labels = {
        "schema_version": "mujoco_g1_batch_failure_labels.v1",
        "generated_at": generated_at,
        "status": "review_required" if failed_attempts else "no_failures_labeled",
        "failed_attempt_count": len(failed_attempts),
        "label_count": len(failed_attempts),
        "failed_run_label_coverage_complete": True,
        "labels": [_label_for_failed_attempt(attempt) for attempt in failed_attempts],
    }
    visual_review_records = [_visual_review_for_attempt(attempt) for attempt in attempts]
    required_run_ids = {
        _string(run_id) for run_id in required_scenario_eval_run_ids if _string(run_id)
    }
    reviewed_run_ids = {
        _string(record.get("scenario_eval_run_id"))
        for record in visual_review_records
        if _string(record.get("scenario_eval_run_id"))
    }
    visual_review_coverage_complete = bool(attempts) and (
        not required_run_ids or required_run_ids.issubset(reviewed_run_ids)
    )
    visual_review_ledger = {
        "schema_version": "mujoco_g1_batch_visual_review_ledger.v1",
        "generated_at": generated_at,
        "status": "accepted" if visual_review_records else "not_available",
        "attempt_count": len(attempts),
        "review_count": len(visual_review_records),
        "accepted_review_count": sum(
            1
            for record in visual_review_records
            if record["review_status"] == "accepted_deterministic_simulator_visual_review"
        ),
        "success_count": sum(1 for record in visual_review_records if record["success"]),
        "failure_count": sum(1 for record in visual_review_records if not record["success"]),
        "media_backed_review_count": sum(
            1 for record in visual_review_records if record["media_evidence_present"]
        ),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "reviewed_scenario_eval_run_ids": sorted(reviewed_run_ids),
        "missing_review_scenario_eval_run_ids": sorted(required_run_ids - reviewed_run_ids),
        "visual_review_coverage_complete": visual_review_coverage_complete,
        "records": visual_review_records,
        "claim_boundary": (
            "Visual review ledger accepts simulator success/failure decisions from "
            "criteria and media refs. It does not claim policy quality, off-scope validation, "
            "or generated-world rank fidelity."
        ),
    }
    metrics = {
        "schema_version": "mujoco_g1_batch_metrics.v1",
        "generated_at": generated_at,
        "attempt_count": len(attempts),
        "passed_attempt_count": sum(1 for attempt in attempts if bool(attempt.get("success"))),
        "failed_attempt_count": len(failed_attempts),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_count": len(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_count": len(missing_scenario_eval_run_ids),
        "scenario_eval_run_coverage_complete": scenario_eval_run_coverage_complete,
        "duplicate_scenario_eval_run_ids": list(duplicate_scenario_eval_run_ids),
        **metric_coverage,
        "claim_boundary": (
            "Metrics are computed from deterministic MuJoCo preview attempts, not "
            "physical robot deployment or robot-team policy quality evidence."
        ),
    }
    _write_jsonl(attempt_trace_path, attempt_records)
    _write_jsonl(contact_stream_path, contact_records)
    _write_jsonl(planner_state_path, planner_records)
    _write_jsonl(control_stream_path, control_records)
    write_json(metrics_path, metrics)
    write_json(failure_labels_path, failure_labels)
    write_json(visual_review_ledger_path, visual_review_ledger)

    checksum_inputs = {
        "attempt_trace_jsonl": attempt_trace_path,
        "contact_stream_jsonl": contact_stream_path,
        "planner_state_jsonl": planner_state_path,
        "control_stream_jsonl": control_stream_path,
        "metrics": metrics_path,
        "failure_labels": failure_labels_path,
        "visual_media_coverage": visual_media_coverage_path,
        "visual_review_ledger": visual_review_ledger_path,
    }
    checksums = {
        "schema_version": "mujoco_g1_batch_artifact_checksums.v1",
        "generated_at": generated_at,
        "artifact_count": len(checksum_inputs),
        "artifacts": {
            name: _file_artifact(path, base_dir=output_root) for name, path in checksum_inputs.items()
        },
    }
    write_json(checksums_path, checksums)
    manifest = {
        "schema_version": MUJOCO_G1_BATCH_TRACE_PACKAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked_missing_attempts",
        "attempt_count": len(attempts),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_count": len(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_count": len(missing_scenario_eval_run_ids),
        "scenario_eval_run_coverage_complete": scenario_eval_run_coverage_complete,
        "metric_coverage_complete": metrics["metric_coverage_complete"],
        "failed_run_label_coverage_complete": failure_labels[
            "failed_run_label_coverage_complete"
        ],
        "contact_stream_record_count": len(contact_records),
        "planner_state_record_count": len(planner_records),
        "control_stream_record_count": len(control_records),
        "planner_state_coverage_complete": len(planner_records) == len(attempts),
        "control_stream_coverage_complete": all(
            bool(attempt.get("actions")) for attempt in attempts
        )
        if attempts
        else False,
        "artifact_paths": {
            "attempt_trace_jsonl": str(attempt_trace_path),
            "contact_stream_jsonl": str(contact_stream_path),
            "planner_state_jsonl": str(planner_state_path),
            "control_stream_jsonl": str(control_stream_path),
            "metrics": str(metrics_path),
            "failure_labels": str(failure_labels_path),
            "visual_media_coverage": str(visual_media_coverage_path),
            "visual_review_ledger": str(visual_review_ledger_path),
            "artifact_checksums": str(checksums_path),
        },
        "visual_media_coverage": {
            "status": visual_media_coverage_manifest.get("status"),
            "all_required_runs_have_visual_recording": visual_media_coverage_manifest.get(
                "all_required_runs_have_visual_recording"
            ),
            "missing_visual_media_run_count": visual_media_coverage_manifest.get(
                "missing_visual_media_run_count"
            ),
            "missing_visual_media_scenario_eval_run_ids": visual_media_coverage_manifest.get(
                "missing_visual_media_scenario_eval_run_ids"
            ),
        },
        "visual_review": {
            "status": visual_review_ledger.get("status"),
            "review_count": visual_review_ledger.get("review_count"),
            "visual_review_coverage_complete": visual_review_ledger.get(
                "visual_review_coverage_complete"
            ),
            "missing_review_scenario_eval_run_ids": visual_review_ledger.get(
                "missing_review_scenario_eval_run_ids"
            ),
        },
        "claim_boundary": (
            "Trace package is simulator evidence and closure input. It does not prove "
            "generated-world rank fidelity or robot-team policy quality."
        ),
    }
    write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "metrics": metrics,
        "failure_labels": failure_labels,
        "visual_media_coverage": visual_media_coverage_manifest,
        "visual_review_ledger": visual_review_ledger,
        "artifact_paths": {
            **manifest["artifact_paths"],
            "visual_review_ledger": str(visual_review_ledger_path),
            "trace_package_manifest": str(manifest_path),
        },
    }


def _build_mujoco_batch_closure_manifest(
    *,
    output_root: Path,
    generated_at: str,
    attempts: Sequence[Mapping[str, Any]],
    required_scenario_eval_run_ids: Sequence[str],
    covered_scenario_eval_run_ids: Sequence[str],
    missing_scenario_eval_run_ids: Sequence[str],
    duplicate_scenario_eval_run_ids: Sequence[str],
    attempt_count_matches_matrix_count: bool,
    scenario_eval_run_id_coverage_exact: bool,
    scenario_eval_run_coverage_complete: bool,
    batch_trace_package: Mapping[str, Any],
    support_artifacts: Mapping[str, Path],
    visual_artifacts: Mapping[str, Any],
    collision_summary: Mapping[str, Any],
    digital_twin_fidelity_qa: Mapping[str, Any],
    robot_team_handoff_blockers: Sequence[str],
    claim_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_presence = {
        key: _file_artifact(path, base_dir=output_root)
        for key, path in support_artifacts.items()
    }
    missing_required_artifacts = [
        key for key, artifact in artifact_presence.items() if not artifact["present"]
    ]
    rendered_run_ids = sorted(
        {
            _string(frame.get("scenario_eval_run_id"))
            for frame in visual_artifacts.get("frames", []) or []
            if isinstance(frame, Mapping) and _string(frame.get("scenario_eval_run_id"))
        }
    )
    video_statuses = {
        "overview_video": _mapping(visual_artifacts.get("overview_video")).get("status"),
        "robot_pov_video": _mapping(visual_artifacts.get("robot_pov_video")).get("status"),
        "side_video": _mapping(visual_artifacts.get("side_video")).get("status"),
    }
    all_video_files_complete = all(status == "complete" for status in video_statuses.values())
    visual_coverage_complete = (
        bool(required_scenario_eval_run_ids)
        and set(rendered_run_ids) == set(required_scenario_eval_run_ids)
        and all_video_files_complete
    )
    trace_manifest = _mapping(batch_trace_package.get("manifest"))
    metrics = _mapping(batch_trace_package.get("metrics"))
    labels = _mapping(batch_trace_package.get("failure_labels"))
    visual_review = _mapping(batch_trace_package.get("visual_review_ledger"))
    if not visual_review:
        visual_review = _mapping(trace_manifest.get("visual_review"))
    metric_coverage_complete = bool(
        trace_manifest.get("metric_coverage_complete")
        and metrics.get("metric_coverage_complete")
    )
    failure_label_coverage_complete = bool(
        trace_manifest.get("failed_run_label_coverage_complete")
        and labels.get("failed_run_label_coverage_complete")
    )
    visual_review_coverage_complete = bool(
        visual_review.get("visual_review_coverage_complete")
        and int(visual_review.get("review_count") or 0) >= len(attempts)
    )
    machine_trace_package_complete = (
        scenario_eval_run_coverage_complete
        and metric_coverage_complete
        and failure_label_coverage_complete
        and not missing_required_artifacts
    )
    robot_team_grade_package_complete = (
        machine_trace_package_complete
        and visual_coverage_complete
        and bool(digital_twin_fidelity_qa.get("robot_team_grade_fidelity_passed"))
        and bool(collision_summary.get("collision_dynamics_validated"))
        and not robot_team_handoff_blockers
    )
    blockers: list[str] = []
    if not scenario_eval_run_coverage_complete:
        blockers.append("scenario_eval_run_coverage_incomplete")
    if missing_required_artifacts:
        blockers.append("batch_required_artifacts_missing")
    if not metric_coverage_complete:
        blockers.append("batch_task_metric_coverage_incomplete")
    if not failure_label_coverage_complete:
        blockers.append("batch_failure_label_coverage_incomplete")
    robot_team_grade_blockers = list(robot_team_handoff_blockers)
    if not visual_coverage_complete:
        robot_team_grade_blockers.append("visual_video_coverage_not_complete_for_all_runs")
    if not visual_review_coverage_complete:
        robot_team_grade_blockers.append("visual_review_coverage_not_complete_for_all_runs")
    if not bool(digital_twin_fidelity_qa.get("robot_team_grade_fidelity_passed")):
        robot_team_grade_blockers.extend(
            _string(blocker)
            for blocker in digital_twin_fidelity_qa.get("blockers", [])
            if _string(blocker)
        )
        robot_team_grade_blockers.append("digital_twin_fidelity_qa_not_passed")
    if not bool(collision_summary.get("collision_dynamics_validated")):
        robot_team_grade_blockers.append("collision_dynamics_not_validated_for_robot_team_grade")
    status = (
        "completed"
        if robot_team_grade_package_complete
        else "completed_with_robot_team_grade_blockers"
        if machine_trace_package_complete
        else "blocked"
    )
    return {
        "schema_version": MUJOCO_G1_BATCH_CLOSURE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "batch_execution_status": "completed" if scenario_eval_run_coverage_complete else "blocked",
        "machine_trace_package_complete": machine_trace_package_complete,
        "robot_team_grade_package_complete": robot_team_grade_package_complete,
        "blockers": blockers,
        "robot_team_grade_blockers": sorted(set(robot_team_grade_blockers)),
        "attempt_count": len(attempts),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_count": len(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_count": len(missing_scenario_eval_run_ids),
        "attempt_count_matches_matrix_count": attempt_count_matches_matrix_count,
        "scenario_eval_run_id_coverage_exact": scenario_eval_run_id_coverage_exact,
        "scenario_eval_run_coverage_complete": scenario_eval_run_coverage_complete,
        "duplicate_scenario_eval_run_ids": list(duplicate_scenario_eval_run_ids),
        "required_scenario_eval_run_ids": list(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_ids": list(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_ids": list(missing_scenario_eval_run_ids),
        "metric_coverage_complete": metric_coverage_complete,
        "failure_label_coverage_complete": failure_label_coverage_complete,
        "visual_review_coverage_complete": visual_review_coverage_complete,
        "failed_attempt_count": int(labels.get("failed_attempt_count") or 0),
        "failure_label_count": int(labels.get("label_count") or 0),
        "visual_review_count": int(visual_review.get("review_count") or 0),
        "artifact_presence": artifact_presence,
        "missing_required_artifacts": missing_required_artifacts,
        "visual_coverage": {
            "rendered_scenario_eval_run_ids": rendered_run_ids,
            "rendered_scenario_eval_run_count": len(rendered_run_ids),
            "missing_rendered_scenario_eval_run_ids": sorted(
                set(required_scenario_eval_run_ids) - set(rendered_run_ids)
            ),
            "all_required_runs_have_visual_recording": visual_coverage_complete,
            "video_statuses": video_statuses,
            "all_video_files_complete": all_video_files_complete,
            "visual_limitations": list(visual_artifacts.get("limitations") or []),
        },
        "visual_review": {
            "status": visual_review.get("status"),
            "review_count": visual_review.get("review_count"),
            "accepted_review_count": visual_review.get("accepted_review_count"),
            "media_backed_review_count": visual_review.get("media_backed_review_count"),
            "missing_review_scenario_eval_run_ids": visual_review.get(
                "missing_review_scenario_eval_run_ids"
            ),
            "visual_review_coverage_complete": visual_review_coverage_complete,
        },
        "digital_twin_fidelity_qa": dict(digital_twin_fidelity_qa),
        "policy_interface_boundary": {
            "default_preview_policy_id": POLICY_ID,
            "robot_team_policy_execution_proven": False,
            "locomotion_controller_integrated": False,
            "training_grade_policy_rollout_proven": False,
            "official_policy_handoff_required_for_robot_team_grade": True,
        },
        "remote_cloud_execution_boundary": {
            "local_or_worker_command_artifacts_emitted": True,
            "signed_output_uris_proven_by_this_command": False,
            "provider_worker_shutdown_proven_by_this_command": False,
        },
        "claim_boundary": dict(claim_boundary),
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
    render_width: int = 640,
    render_height: int = 360,
    max_rendered_episodes: int = 3,
    max_rendered_steps: int = 24,
    allow_fetch_g1_assets: bool = False,
    menagerie_ref: str = DEFAULT_MENAGERIE_REF,
    manipulation_ready_arms: bool = False,
    manipulation_reach_arm: str = "both",
    collision_proxy_mode: str = "aabb",
) -> dict[str, Any]:
    if platform.system().lower() == "linux":
        # EGL is only needed when frames are rendered; forcing it on GL-less hosts
        # (CI runners, --skip-render-frames runs) crashes `import mujoco` at EGL load.
        os.environ.setdefault("MUJOCO_GL", "egl" if render_frames else "disable")
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
    mesh_info = _convert_glb_to_obj(
        scene_glb,
        scene_obj,
        collision_proxy_mode=collision_proxy_mode,
    )

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
    scene_texture_file = None
    obj_texture_summary = _mapping(mesh_info.get("obj_texture_material_summary"))
    if obj_texture_summary.get("texture_exists") is True:
        map_kd_texture_path = _string(obj_texture_summary.get("map_kd_texture_path"))
        if map_kd_texture_path:
            scene_texture_file = Path(map_kd_texture_path)
    _write_mjcf_wrapper(
        scene_obj,
        generated_g1_xml,
        wrapper_xml,
        collision_proxies=collision_proxies if isinstance(collision_proxies, Sequence) else None,
        scene_texture_file=scene_texture_file,
        scene_bounds=mesh_info.get("bounds"),
        scene_centroid=mesh_info.get("centroid"),
        render_width=render_width,
        render_height=render_height,
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

        renderer = mujoco.Renderer(
            model,
            height=max(1, int(render_height)),
            width=max(1, int(render_width)),
        )
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

    def capture_diagnostic_review_frames(
        *,
        attempt_id: str,
        episode_id: str,
        scenario_eval_run_id: str,
        pose: Sequence[float],
        yaw: float,
        step: int,
        reason: str,
    ) -> list[dict[str, Any]]:
        if renderer is None or Image is None:  # pragma: no cover - guarded by should_render_episode.
            return []
        x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
        records: list[dict[str, Any]] = []
        for offset in range(2):
            frame_step = int(step) + offset
            sim_time_s = float(data.time) + offset * float(model_timestep or 0.0)

            renderer.update_scene(data, camera="overview")
            overview_path = frames_dir / f"{attempt_id}_overview_diagnostic_{offset:04d}.png"
            Image.fromarray(renderer.render()).save(overview_path)
            overview_record = _camera_record_with_time(
                "overview",
                overview_path,
                frame_step,
                "named_fixed_overview_camera",
                sim_time_s=sim_time_s,
            )
            overview_record.update(
                {
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
                    "diagnostic_reason": reason,
                }
            )
            records.append(overview_record)

            diagnostic_camera_selected = _configure_robot_pov_camera(
                robot_camera,
                pose=pose,
                yaw=yaw,
                manipulation_ready_arms=manipulation_ready_arms,
            )
            alpha_restore = (
                _set_geom_alpha(model, robot_pov_self_occluding_geom_ids, 0.0)
                if manipulation_ready_arms
                else []
            )
            try:
                renderer.update_scene(data, camera=robot_camera)
                robot_pixels = renderer.render()
            finally:
                _restore_geom_alpha(model, alpha_restore)
            robot_path = frames_dir / f"{attempt_id}_sim_robot_follow_pov_diagnostic_{offset:04d}.png"
            Image.fromarray(robot_pixels).save(robot_path)
            robot_record = _camera_record_with_time(
                "sim_robot_follow_pov",
                robot_path,
                frame_step,
                diagnostic_camera_selected["camera_mode"],
                sim_time_s=sim_time_s,
            )
            robot_record.update(
                {
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
                    "diagnostic_reason": reason,
                    "robot_camera_selected": diagnostic_camera_selected,
                }
            )
            records.append(robot_record)

            side_camera.lookat[:] = [x, y, z + 0.45]
            side_camera.distance = 3.2
            side_camera.azimuth = 90
            side_camera.elevation = -14
            renderer.update_scene(data, camera=side_camera)
            side_path = frames_dir / f"{attempt_id}_side_diagnostic_{offset:04d}.png"
            Image.fromarray(renderer.render()).save(side_path)
            side_record = _camera_record_with_time(
                "side",
                side_path,
                frame_step,
                "diagnostic_side_camera",
                sim_time_s=sim_time_s,
            )
            side_record.update(
                {
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
                    "diagnostic_reason": reason,
                }
            )
            records.append(side_record)
        return records

    preview_joint_addresses = _g1_preview_joint_addresses(model, mujoco)
    robot_pov_self_occluding_geom_ids = _robot_pov_self_occluding_geom_ids(model, mujoco)
    manipulation_ready_pose = {
        "enabled": bool(manipulation_ready_arms),
        "arm_selection": str(manipulation_reach_arm or "both"),
        "applied_joint_names": sorted(
            name
            for name in _manipulation_ready_arm_joint_deltas(manipulation_reach_arm)
            if name in preview_joint_addresses
        )
        if manipulation_ready_arms
        else [],
        "robot_pov_hidden_self_geom_count": (
            len(robot_pov_self_occluding_geom_ids) if manipulation_ready_arms else 0
        ),
        "pose_semantics": (
            "poses the visible simulated G1 forearms into the first-person workspace "
            "for review media; not contact success, manipulation success, real robot "
            "camera proof, or deployment readiness"
        ),
    }
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
        blocked_collision_probe: dict[str, Any] | None = None
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
                    manipulation_ready_arms=manipulation_ready_arms,
                    manipulation_reach_arm=manipulation_reach_arm,
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
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
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
                blocked_collision_probe = {
                    "reason": "no_non_colliding_pose",
                    "step": step,
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
                    "rejected_candidate_count": len(rejected_candidates),
                    "candidate_count": len(candidate_results),
                }
                actions.append(
                    {
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                        "root_position": [round(float(value), 6) for value in start],
                        "desired_root_position": [
                            round(float(desired_pose[0]), 6),
                            round(float(desired_pose[1]), 6),
                            round(float(desired_pose[2]), 6),
                        ],
                        "root_yaw_radians": round(float(yaw), 6),
                        "target": list(target),
                        "route_segment_index": route_segment_index,
                        "contact_count": 0,
                        "scene_collision_contact_count": 0,
                        "collision_probe_candidate_count": len(candidate_results),
                        "rejected_collision_probe_count": len(rejected_candidates),
                        "policy_action": "blocked_no_non_colliding_pose",
                        "status": "blocked_collision_probe_no_safe_pose",
                        "scenario_eval_run_id": scenario_eval_run_id or None,
                        "deterministic_seed": navigation["seed"],
                    }
                )
                break
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
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
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
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "scenario_eval_run_id": scenario_eval_run_id or None,
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
                manipulation_ready_arms=manipulation_ready_arms,
                manipulation_reach_arm=manipulation_reach_arm,
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

                robot_camera_selected = _configure_robot_pov_camera(
                    robot_camera,
                    pose=selected_pose,
                    yaw=selected_yaw,
                    manipulation_ready_arms=manipulation_ready_arms,
                )
                alpha_restore = (
                    _set_geom_alpha(model, robot_pov_self_occluding_geom_ids, 0.0)
                    if manipulation_ready_arms
                    else []
                )
                try:
                    renderer.update_scene(data, camera=robot_camera)
                    robot_render = renderer.render()
                    robot_camera_selected["scene_detail_score"] = round(
                        _rendered_array_scene_score(robot_render), 3
                    )
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
                                    "camera_mode": (
                                        "virtual_free_camera_following_g1_root_not_physical_robot_sensor"
                                    ),
                                    "azimuth": robot_camera.azimuth,
                                    "distance": distance,
                                    "elevation": elevation,
                                    "fallback_used": True,
                                    "fallback_reason": "route_follow_camera_frame_blank",
                                    "fallback_option_index": option_index,
                                    "scene_detail_score": round(candidate_score, 3),
                                }
                finally:
                    _restore_geom_alpha(model, alpha_restore)
                if len(matrix_runs) == 1:
                    robot_path = frames_dir / f"sim_robot_follow_pov_{step:04d}.png"
                else:
                    robot_path = frames_dir / f"{attempt_id}_sim_robot_follow_pov_{step:04d}.png"
                Image.fromarray(robot_render).save(robot_path)
                robot_record = _camera_record_with_time(
                    "sim_robot_follow_pov",
                    robot_path,
                    step,
                    robot_camera_selected["camera_mode"],
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
        if should_render_episode and not episode_frames:
            diagnostic_pose = desired_pose if "desired_pose" in locals() else start
            diagnostic_yaw = yaw if "yaw" in locals() else 0.0
            diagnostic_frames = capture_diagnostic_review_frames(
                attempt_id=attempt_id,
                episode_id=episode_id,
                scenario_eval_run_id=scenario_eval_run_id,
                pose=diagnostic_pose,
                yaw=diagnostic_yaw,
                step=0,
                reason=(
                    "episode_exited_before_regular_render_frames;"
                    f"status={blocked_collision_probe.get('reason') or 'no_actions_recorded'}"
                ),
            )
            frames.extend(diagnostic_frames)
            episode_frames.extend(diagnostic_frames)

        rendered_frame_paths = [frame["path"] for frame in episode_frames]
        episode_collision_summary = _collision_summary(
            episode_contact_trace,
            collision_probe_trace=episode_collision_probe_trace,
            collision_response_events=episode_collision_response_events,
            collision_proxy_count=collision_proxy_count,
        )
        task_outcome = _attempt_task_outcome(
            actions=actions,
            start=start,
            target=target,
            route_distance_m=route_distance,
            collision_summary=episode_collision_summary,
            bounded_steps=bounded_steps,
            model_timestep_s=model_timestep,
        )
        if blocked_collision_probe:
            failure_mode_ids = list(task_outcome["failure_mode_ids"])
            if "failure_collision_probe_no_safe_pose" not in failure_mode_ids:
                failure_mode_ids.append("failure_collision_probe_no_safe_pose")
            task_outcome = {
                **task_outcome,
                "task_success": False,
                "task_status": "blocked_collision_probe_no_safe_pose",
                "failure_mode_ids": failure_mode_ids,
                "failure_reason": ",".join(failure_mode_ids),
                "collision_probe_blocked": blocked_collision_probe,
            }
        scene_contact_count = int(episode_collision_summary["robot_scene_contact_event_count"])
        collision_free_preview = scene_contact_count == 0
        attempt_status = (
            "blocked_collision_probe_no_safe_pose"
            if blocked_collision_probe
            else "passed_task_criteria"
            if task_outcome["task_success"]
            else "failed_task_criteria"
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
            "success": bool(task_outcome["task_success"]),
            "task_success": bool(task_outcome["task_success"]),
            "failure_reason": task_outcome["failure_reason"],
            "failure_mode_ids": task_outcome["failure_mode_ids"],
            "success_semantics": (
                "goal_reached_with_clean_endpoint_no_scene_contacts_no_fall"
                if task_outcome["task_success"]
                else "physics_preview_completed_but_task_success_criteria_failed"
                if collision_free_preview
                else "scene_collision_contacts_detected_preview_blocked"
            ),
            "deterministic_seed": navigation["seed"],
            "spawn_pose": list(start),
            "target_pose": list(target),
            "final_pose": task_outcome["final_pose"],
            "route_source": navigation["route_source"],
            "route_strategy": route_strategy,
            "route_waypoints": [list(point) for point in route_points],
            "locomotion_controller_integrated": False,
            "walking_motion_proven": False,
            "walking_style_preview_animation_rendered": bool(preview_joint_addresses),
            "manipulation_ready_arms_pose_applied": bool(manipulation_ready_arms),
            "manipulation_ready_pose": manipulation_ready_pose,
            "training_grade_policy_rollout_proven": False,
            "collision_probe_blocked": blocked_collision_probe,
            "metrics": {
                "cycle_time_seconds": round(bounded_steps * model_timestep, 6)
                if model_timestep
                else None,
                "intervention_count": 0,
                "unsafe_proximity_event_count": task_outcome["near_miss_event_count"],
                "collision_risk_event_count": scene_contact_count
                + task_outcome["near_miss_event_count"],
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
                "timeout_count": 1 if task_outcome["timeout"] else 0,
                "fall_count": 1 if task_outcome["fall_detected"] else 0,
                "stuck_event_count": 1 if task_outcome["stuck_detected"] else 0,
                "near_miss_event_count": task_outcome["near_miss_event_count"],
                "min_clearance_m": task_outcome["min_clearance_m"],
                "clearance_threshold_m": task_outcome["clearance_threshold_m"],
                "clearance_threshold_violation": task_outcome[
                    "clearance_threshold_violation"
                ],
                "goal_reached": task_outcome["goal_reached"],
                "endpoint_clean": task_outcome["endpoint_clean"],
                "spawn_clean": task_outcome["spawn_clean"],
                "task_success": task_outcome["task_success"],
                "final_target_error_m": task_outcome["final_target_error_m"],
                "goal_tolerance_m": task_outcome["goal_tolerance_m"],
                "actual_path_distance_m": task_outcome["actual_path_distance_m"],
                "path_efficiency_ratio": task_outcome["path_efficiency_ratio"],
                "progress_to_goal_m": task_outcome["progress_to_goal_m"],
                "progress_to_goal_ratio": task_outcome["progress_to_goal_ratio"],
                "max_path_deviation_m": task_outcome["max_path_deviation_m"],
                "mean_path_deviation_m": task_outcome["mean_path_deviation_m"],
                "min_root_height_m": task_outcome["min_root_height_m"],
                "policy_instability_detected": task_outcome[
                    "policy_instability_detected"
                ],
                "simulated_step_count": bounded_steps,
                "rendered_step_count": len(capture_steps),
                "rendered_frame_count": len(rendered_frame_paths),
                "route_distance_m": round(route_distance, 6),
                "direct_start_to_target_distance_m": navigation["route_distance_m"],
                "start_pose_xyz": list(start),
                "target_pose_xyz": list(target),
                "deterministic_seed": navigation["seed"],
            },
            "task_outcome": task_outcome,
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
                "final_pose": task_outcome["final_pose"],
                "step_count": bounded_steps,
                "route_strategy": route_strategy,
                "route_waypoints": [list(point) for point in route_points],
                "route_distance_m": round(route_distance, 6),
                "rendered_step_count": len(capture_steps),
                "deterministic_seed": navigation["seed"],
                "task_success": task_outcome["task_success"],
                "task_outcome": task_outcome,
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
    digital_twin_artifact_refs = {
        "scene_load_trace": str(output_root / "scene_load_trace.json"),
        "source_scene_glb": str(scene_glb),
        "converted_scene_obj": str(scene_obj),
        "capture_object_index": str(root / "raw" / "object_index.json"),
        "task_targets": str(root / "pipeline" / "task_targets.json"),
    }
    digital_twin_fidelity_qa = _build_digital_twin_fidelity_qa(
        generated_at=generated_at,
        mesh_info=mesh_info,
        collision_summary=collision_summary,
        visual_artifacts=visual_artifacts,
        artifact_refs=digital_twin_artifact_refs,
        capture_object_semantics_summary=_capture_object_semantics_summary(root),
    )
    blocked_collision_attempt_count = sum(
        1 for attempt in attempts if _string(attempt.get("status")) == "blocked_collision_overlap_detected"
    )
    collision_free_preview = blocked_collision_attempt_count == 0
    successful_task_attempts = [
        attempt for attempt in attempts if bool(attempt.get("task_success"))
    ]
    failed_task_attempts = [
        attempt for attempt in attempts if not bool(attempt.get("task_success"))
    ]
    task_failure_mode_counts: dict[str, int] = {}
    for attempt in failed_task_attempts:
        failure_modes = attempt.get("failure_mode_ids")
        if not isinstance(failure_modes, Sequence) or isinstance(
            failure_modes, (str, bytes)
        ):
            failure_modes = []
        for failure_mode in failure_modes:
            failure_id = _string(failure_mode)
            if failure_id:
                task_failure_mode_counts[failure_id] = (
                    task_failure_mode_counts.get(failure_id, 0) + 1
                )
    clearance_values = [
        float(value)
        for value in (
            _number(_mapping(attempt.get("task_outcome")).get("min_clearance_m"))
            for attempt in attempts
        )
        if value is not None
    ]
    task_success_summary = {
        "schema_version": "mujoco_g1_task_success_summary.v1",
        "status": "completed" if attempts else "not_available",
        "attempt_count": len(attempts),
        "successful_attempt_count": len(successful_task_attempts),
        "failed_attempt_count": len(failed_task_attempts),
        "task_success_rate": round(len(successful_task_attempts) / len(attempts), 6)
        if attempts
        else None,
        "failed_scenario_eval_run_ids": sorted(
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in failed_task_attempts
            if _string(attempt.get("scenario_eval_run_id"))
        ),
        "failure_mode_counts": dict(sorted(task_failure_mode_counts.items())),
        "near_miss_attempt_count": sum(
            1
            for attempt in attempts
            if int(_mapping(attempt.get("task_outcome")).get("near_miss_event_count") or 0)
            > 0
        ),
        "near_miss_event_count": sum(
            int(_mapping(attempt.get("task_outcome")).get("near_miss_event_count") or 0)
            for attempt in attempts
        ),
        "min_clearance_m": min(clearance_values) if clearance_values else None,
        "clearance_threshold_m": TASK_CLEARANCE_THRESHOLD_M,
        "endpoint_clean_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("endpoint_clean"))
        ),
        "goal_reached_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("goal_reached"))
        ),
        "max_final_target_error_m": max(
            (
                float(_mapping(attempt.get("task_outcome")).get("final_target_error_m"))
                for attempt in attempts
                if _mapping(attempt.get("task_outcome")).get("final_target_error_m")
                is not None
            ),
            default=None,
        ),
        "max_path_deviation_m": max(
            (
                float(_mapping(attempt.get("task_outcome")).get("max_path_deviation_m"))
                for attempt in attempts
                if _mapping(attempt.get("task_outcome")).get("max_path_deviation_m")
                is not None
            ),
            default=None,
        ),
        "task_success_boundary": (
            "Task success requires reaching the target inside tolerance with a clean "
            "endpoint, no committed scene contacts, no clearance near-miss, no fall, "
            "and no stuck/no-progress heuristic. Simulator command completion is tracked separately."
        ),
    }
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
            "navigation_plan_manifest.json",
            "policy_execution_trace_enriched.jsonl",
            "robot_pov_manifest.json",
        ],
        "boundary": (
            "Use the official Unitree RL Gym G1 handoff path for balanced-controller "
            "rollouts, planner waypoint velocity commands, qpos/qvel/control streams, "
            "contact and clearance traces, and robot-team review. "
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
        "manipulation_ready_arms_pose_applied": bool(manipulation_ready_arms),
        "manipulation_ready_pose_is_review_media_not_success_proof": bool(manipulation_ready_arms),
        "training_grade_policy_rollout_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "non_ranking_operational_claim_validated": False,
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
        "render_resolution": {
            "width": max(1, int(render_width)),
            "height": max(1, int(render_height)),
        },
        "robot_asset": robot_asset,
        "asset_source_manifest": source_manifest,
        "manipulation_ready_pose": manipulation_ready_pose,
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
        "manipulation_ready_arms_pose_applied": bool(manipulation_ready_arms),
        "manipulation_ready_pose": manipulation_ready_pose,
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
        "task_success_summary": task_success_summary,
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
        "digital_twin_fidelity_qa": str(output_root / "mujoco_digital_twin_fidelity_qa.json"),
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
                str(output_root / "mujoco_digital_twin_fidelity_qa.json"),
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
        "digital_twin_fidelity_qa": digital_twin_fidelity_qa,
        "task_success_summary": task_success_summary,
        "collision_proxy_summary": mesh_info.get("collision_proxy_summary"),
        "physics_controlled_preview_proven": physics_controlled_preview_proven,
        "robot_team_handoff_ready": False,
        "robot_team_handoff_blockers": robot_team_handoff_blockers,
        "official_policy_handoff": official_policy_handoff,
        "limitations": visual_artifacts["limitations"],
        "files": [
            str(wrapper_xml),
            str(generated_g1_xml),
            str(scene_obj),
            str(output_root / "mujoco_digital_twin_fidelity_qa.json"),
        ]
        + [frame["path"] for frame in frames]
        + videos,
    }
    write_json(output_root / "scene_load_trace.json", scene_trace)
    write_json(output_root / "spawn_trace.json", spawn_trace)
    write_json(output_root / "policy_execution_trace.json", policy_trace)
    write_json(output_root / "sim_robot_pov_evidence_manifest.json", pov_manifest)
    write_json(output_root / "mujoco_digital_twin_fidelity_qa.json", digital_twin_fidelity_qa)
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
    batch_trace_package = _write_mujoco_batch_trace_package(
        output_root=output_root,
        generated_at=generated_at,
        attempts=attempts,
        full_contact_trace=full_contact_trace,
        full_collision_probe_trace=full_collision_probe_trace,
        full_collision_response_events=full_collision_response_events,
        required_scenario_eval_run_ids=required_scenario_eval_run_ids,
        covered_scenario_eval_run_ids=covered_scenario_eval_run_ids,
        missing_scenario_eval_run_ids=missing_scenario_eval_run_ids,
        duplicate_scenario_eval_run_ids=duplicate_scenario_eval_run_ids,
        scenario_eval_run_coverage_complete=scenario_eval_run_coverage_complete,
        visual_artifacts=visual_artifacts,
    )
    batch_artifact_paths = {
        key: str(value) for key, value in _mapping(batch_trace_package.get("artifact_paths")).items()
    }
    batch_closure_path = output_root / "mujoco_batch_closure_manifest.json"
    artifact_paths.update(
            {
                "batch_attempt_trace_jsonl": batch_artifact_paths.get("attempt_trace_jsonl"),
                "batch_contact_stream_jsonl": batch_artifact_paths.get("contact_stream_jsonl"),
                "batch_planner_state_jsonl": batch_artifact_paths.get("planner_state_jsonl"),
                "batch_control_stream_jsonl": batch_artifact_paths.get("control_stream_jsonl"),
            "batch_metrics": batch_artifact_paths.get("metrics"),
            "batch_failure_labels": batch_artifact_paths.get("failure_labels"),
            "batch_visual_media_coverage": batch_artifact_paths.get("visual_media_coverage"),
            "batch_visual_review_ledger": batch_artifact_paths.get("visual_review_ledger"),
            "batch_artifact_checksums": batch_artifact_paths.get("artifact_checksums"),
            "batch_trace_package_manifest": batch_artifact_paths.get("trace_package_manifest"),
            "batch_closure_manifest": str(batch_closure_path),
        }
    )
    artifact_manifest["artifacts"] = artifact_paths
    artifact_manifest["batch_trace_package"] = batch_trace_package.get("manifest")
    artifact_manifest["batch_closure_manifest"] = str(batch_closure_path)
    artifact_manifest["artifact_classes"]["local_sim"] = [
        *artifact_manifest["artifact_classes"]["local_sim"],
        batch_artifact_paths.get("attempt_trace_jsonl"),
        batch_artifact_paths.get("contact_stream_jsonl"),
        batch_artifact_paths.get("planner_state_jsonl"),
        batch_artifact_paths.get("control_stream_jsonl"),
        batch_artifact_paths.get("metrics"),
        batch_artifact_paths.get("failure_labels"),
        batch_artifact_paths.get("visual_review_ledger"),
        batch_artifact_paths.get("artifact_checksums"),
        batch_artifact_paths.get("trace_package_manifest"),
        str(batch_closure_path),
    ]
    artifact_manifest["files"] = [
        *artifact_manifest["files"],
        *[
            value
            for value in (
                batch_artifact_paths.get("attempt_trace_jsonl"),
                batch_artifact_paths.get("contact_stream_jsonl"),
                batch_artifact_paths.get("planner_state_jsonl"),
                batch_artifact_paths.get("control_stream_jsonl"),
                batch_artifact_paths.get("metrics"),
                batch_artifact_paths.get("failure_labels"),
                batch_artifact_paths.get("visual_review_ledger"),
                batch_artifact_paths.get("artifact_checksums"),
                batch_artifact_paths.get("trace_package_manifest"),
                str(batch_closure_path),
            )
            if value
        ],
    ]
    write_json(output_root / "artifact_manifest.json", artifact_manifest)
    batch_closure_manifest = _build_mujoco_batch_closure_manifest(
        output_root=output_root,
        generated_at=generated_at,
        attempts=attempts,
        required_scenario_eval_run_ids=required_scenario_eval_run_ids,
        covered_scenario_eval_run_ids=covered_scenario_eval_run_ids,
        missing_scenario_eval_run_ids=missing_scenario_eval_run_ids,
        duplicate_scenario_eval_run_ids=duplicate_scenario_eval_run_ids,
        attempt_count_matches_matrix_count=attempt_count_matches_matrix_count,
        scenario_eval_run_id_coverage_exact=scenario_eval_run_id_coverage_exact,
        scenario_eval_run_coverage_complete=scenario_eval_run_coverage_complete,
        batch_trace_package=batch_trace_package,
        support_artifacts={
            "scene_trace": output_root / "scene_load_trace.json",
            "spawn_trace": output_root / "spawn_trace.json",
            "policy_trace": output_root / "policy_execution_trace.json",
            "sim_robot_pov_evidence": output_root / "sim_robot_pov_evidence_manifest.json",
            "digital_twin_fidelity_qa": output_root / "mujoco_digital_twin_fidelity_qa.json",
            "artifact_manifest": output_root / "artifact_manifest.json",
            "batch_attempt_trace_jsonl": Path(
                batch_artifact_paths.get("attempt_trace_jsonl") or ""
            ),
            "batch_contact_stream_jsonl": Path(
                batch_artifact_paths.get("contact_stream_jsonl") or ""
            ),
            "batch_metrics": Path(batch_artifact_paths.get("metrics") or ""),
            "batch_failure_labels": Path(
                batch_artifact_paths.get("failure_labels") or ""
            ),
            "batch_visual_review_ledger": Path(
                batch_artifact_paths.get("visual_review_ledger") or ""
            ),
            "batch_artifact_checksums": Path(
                batch_artifact_paths.get("artifact_checksums") or ""
            ),
            "batch_trace_package_manifest": Path(
                batch_artifact_paths.get("trace_package_manifest") or ""
            ),
        },
        visual_artifacts={**visual_artifacts, "frames": frames},
        collision_summary=collision_summary,
        digital_twin_fidelity_qa=digital_twin_fidelity_qa,
        robot_team_handoff_blockers=robot_team_handoff_blockers,
        claim_boundary=claim_boundary,
    )
    write_json(batch_closure_path, batch_closure_manifest)
    payload = {
        "schema_version": MUJOCO_G1_SIMULATOR_COMMAND_OUTPUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "simulator_backend": "mujoco",
        "mujoco_version": _string(getattr(mujoco, "__version__", "")),
        "capture_root": str(root),
        "output_dir": str(output_root),
        "render_resolution": {
            "width": max(1, int(render_width)),
            "height": max(1, int(render_height)),
        },
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
        "manipulation_ready_arms_pose_applied": bool(manipulation_ready_arms),
        "manipulation_ready_pose": manipulation_ready_pose,
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
        "digital_twin_fidelity_qa": digital_twin_fidelity_qa,
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
        "successful_task_attempt_count": task_success_summary["successful_attempt_count"],
        "failed_task_attempt_count": task_success_summary["failed_attempt_count"],
        "task_success_rate": task_success_summary["task_success_rate"],
        "task_success_summary": task_success_summary,
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
        "batch_trace_package": batch_trace_package.get("manifest"),
        "batch_closure_manifest": batch_closure_manifest,
        "batch_closure_manifest_path": str(batch_closure_path),
        "machine_trace_package_complete": batch_closure_manifest.get(
            "machine_trace_package_complete"
        ),
        "robot_team_grade_package_complete": batch_closure_manifest.get(
            "robot_team_grade_package_complete"
        ),
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


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=360)
    parser.add_argument("--max-rendered-episodes", type=int, default=3)
    parser.add_argument("--max-rendered-steps", type=int, default=24)
    parser.add_argument("--allow-fetch-g1-assets", action="store_true")
    parser.add_argument("--no-fetch-g1-assets", action="store_true")
    parser.add_argument("--menagerie-ref", default=DEFAULT_MENAGERIE_REF)
    parser.add_argument(
        "--manipulation-ready-arms",
        action="store_true",
        help=(
            "Pose simulated G1 forearms into the robot-POV workspace for review media; "
            "this does not prove manipulation success."
        ),
    )
    parser.add_argument(
        "--manipulation-reach-arm",
        default="both",
        choices=["right", "left", "both"],
        help="Which simulated arm is posed when --manipulation-ready-arms is enabled.",
    )
    parser.add_argument(
        "--collision-proxy-mode",
        default="aabb",
        choices=["aabb", "obb", "convex"],
        help=(
            "MuJoCo scene collision proxy shape: aabb (fast, coarse, default), obb "
            "(oriented bounding boxes), or convex (coacd/VHACD decomposition when "
            "available); mirrors Isaac --collision-approximation for the MuJoCo lane."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    capture_root_env = os.environ.get("BLUEPRINT_CAPTURE_ROOT")
    capture_root = args.capture_root or (Path(capture_root_env) if capture_root_env else None)
    if capture_root is None:
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
        render_width=args.render_width,
        render_height=args.render_height,
        max_rendered_episodes=args.max_rendered_episodes,
        max_rendered_steps=args.max_rendered_steps,
        allow_fetch_g1_assets=allow_fetch,
        menagerie_ref=args.menagerie_ref,
        manipulation_ready_arms=args.manipulation_ready_arms,
        manipulation_reach_arm=args.manipulation_reach_arm,
        collision_proxy_mode=args.collision_proxy_mode,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
