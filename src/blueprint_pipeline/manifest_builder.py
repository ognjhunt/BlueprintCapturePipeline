"""Build canonical manifest/layout artifacts for downstream BlueprintPipeline jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .capture_bridge import CaptureDescriptor
from .common import ensure_dir, utc_now_iso, write_json


def _room_from_mesh_bounds(nurec_outputs: Mapping[str, Any]) -> Tuple[Dict[str, float], List[float], Dict[str, List[float]]]:
    mesh_stats = nurec_outputs.get("mesh_stats") if isinstance(nurec_outputs.get("mesh_stats"), Mapping) else {}
    bounds = mesh_stats.get("bounds") if isinstance(mesh_stats.get("bounds"), Mapping) else {}
    mins = bounds.get("min") if isinstance(bounds.get("min"), list) else [-3.0, 0.0, -3.0]
    maxs = bounds.get("max") if isinstance(bounds.get("max"), list) else [3.0, 3.0, 3.0]

    mins = [float(mins[idx]) if idx < len(mins) else 0.0 for idx in range(3)]
    maxs = [float(maxs[idx]) if idx < len(maxs) else 0.0 for idx in range(3)]
    width = max(0.1, maxs[0] - mins[0])
    height = max(0.1, maxs[1] - mins[1])
    depth = max(0.1, maxs[2] - mins[2])
    origin = [
        mins[0] + width * 0.5,
        mins[1],
        mins[2] + depth * 0.5,
    ]
    return (
        {"width": width, "height": height, "depth": depth},
        origin,
        {"min": mins, "max": maxs},
    )


def _candidate_transform(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    obb = candidate.get("obb") if isinstance(candidate.get("obb"), Mapping) else {}
    center = obb.get("center") if isinstance(obb.get("center"), list) else [0.0, 0.0, 0.0]
    quat = (
        obb.get("orientationQuaternion")
        if isinstance(obb.get("orientationQuaternion"), list)
        else [1.0, 0.0, 0.0, 0.0]
    )
    return {
        "position": {
            "x": float(center[0]) if len(center) > 0 else 0.0,
            "y": float(center[1]) if len(center) > 1 else 0.0,
            "z": float(center[2]) if len(center) > 2 else 0.0,
        },
        "rotation_quaternion": {
            "w": float(quat[0]) if len(quat) > 0 else 1.0,
            "x": float(quat[1]) if len(quat) > 1 else 0.0,
            "y": float(quat[2]) if len(quat) > 2 else 0.0,
            "z": float(quat[3]) if len(quat) > 3 else 0.0,
        },
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
    }


def _layout_entry(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    object_id = str(candidate["object_id"])
    obb = candidate.get("obb") if isinstance(candidate.get("obb"), Mapping) else {}
    center = obb.get("center") if isinstance(obb.get("center"), list) else [0.0, 0.0, 0.0]
    quat = (
        obb.get("orientationQuaternion")
        if isinstance(obb.get("orientationQuaternion"), list)
        else [1.0, 0.0, 0.0, 0.0]
    )

    entry: Dict[str, Any] = {
        "id": object_id,
        "class_name": str(candidate.get("label") or "object"),
        "center3d": [
            float(center[0]) if len(center) > 0 else 0.0,
            float(center[1]) if len(center) > 1 else 0.0,
            float(center[2]) if len(center) > 2 else 0.0,
        ],
        "rotation_quaternion": {
            "w": float(quat[0]) if len(quat) > 0 else 1.0,
            "x": float(quat[1]) if len(quat) > 1 else 0.0,
            "y": float(quat[2]) if len(quat) > 2 else 0.0,
            "z": float(quat[3]) if len(quat) > 3 else 0.0,
        },
    }
    if obb:
        entry["obb"] = dict(obb)
    return entry


def build_scene_artifacts(
    *,
    storage_root: Path,
    scene_id: str,
    capture_id: str,
    descriptor: CaptureDescriptor,
    descriptor_uri: str,
    nurec_outputs: Mapping[str, Any],
    swap_candidates: List[Mapping[str, Any]],
    assets_prefix: str,
    layout_prefix: str,
    seg_prefix: str,
) -> Dict[str, Any]:
    """Write manifest/layout/inventory expected by BlueprintPipeline jobs."""

    assets_root = storage_root / assets_prefix
    layout_root = storage_root / layout_prefix
    seg_root = storage_root / seg_prefix
    ensure_dir(assets_root)
    ensure_dir(layout_root)
    ensure_dir(seg_root)

    room_bounds, room_origin, room_box = _room_from_mesh_bounds(nurec_outputs)
    nurec_artifacts = nurec_outputs.get("artifacts") if isinstance(nurec_outputs.get("artifacts"), Mapping) else {}
    hallucinated_region_mask = (
        str(nurec_artifacts.get("hallucinated_region_mask") or "").strip()
        if isinstance(nurec_artifacts, Mapping)
        else ""
    )

    manifest_objects: List[Dict[str, Any]] = [
        {
            "id": "scene_shell",
            "name": "scene_shell",
            "category": "scene_shell",
            "description": "NuRec nvblox collision shell",
            "sim_role": "scene_shell",
            "asset": {
                "path": f"{assets_prefix}/obj_scene_shell/model.usd",
                "source": "nurec",
                "format": "usd",
            },
            "transform": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            },
            "physics_hints": {
                "dynamic": False,
                "collision_role": "environment_shell",
            },
            "articulation": {"required": False, "backend_hint": "none"},
            "source": {"type": "capture_nurec"},
        },
        {
            "id": "nurec_visual",
            "name": "nurec_visual",
            "category": "scene_visual",
            "description": "NuRec neural visual context",
            "sim_role": "background",
            "asset": {
                "path": f"{assets_prefix}/obj_nurec_visual/model.usd",
                "source": "nurec",
                "format": "usd",
            },
            "transform": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            },
            "physics_hints": {"dynamic": False, "collision_enabled": False},
            "articulation": {"required": False, "backend_hint": "none"},
            "source": {"type": "capture_nurec"},
        },
    ]

    layout_objects: List[Dict[str, Any]] = []
    inventory_objects: List[Dict[str, Any]] = []

    for candidate in swap_candidates:
        object_id = str(candidate["object_id"])
        label = str(candidate.get("label") or "object")
        sim_role = str(candidate.get("sim_role") or "manipulable_object")
        asset_dir = str(candidate.get("asset_dir") or f"obj_{object_id}")
        articulation = (
            candidate.get("articulation")
            if isinstance(candidate.get("articulation"), Mapping)
            else {"required": False, "requirement_source": "policy"}
        )

        manifest_obj: Dict[str, Any] = {
            "id": object_id,
            "name": label,
            "category": label,
            "description": f"Swappable {label}",
            "sim_role": sim_role,
            "must_be_separate_asset": True,
            "asset": {
                "path": f"{assets_prefix}/{asset_dir}/model.usd",
                "source": "sam3d_first",
                "format": "usd",
            },
            "transform": _candidate_transform(candidate),
            "dimensions_est": dict(candidate.get("dimensions_est") or {}),
            "physics_hints": dict(candidate.get("physics_hints") or {}),
            "articulation": {
                "required": bool(articulation.get("required", False)),
                "backend_hint": "particulate_first"
                if bool(articulation.get("required", False))
                else "none",
                "requirement_source": str(articulation.get("requirement_source") or "policy"),
                "candidate": True,
            },
            "source": {
                "type": "capture_nurec_swap",
                "capture_id": capture_id,
            },
        }

        # Include reference image path in manifest for downstream consumers
        ref_crop = candidate.get("reference_crop")
        if ref_crop:
            manifest_obj["reference_image"] = f"{assets_prefix}/{asset_dir}/reference.png"

        manifest_objects.append(manifest_obj)

        layout_objects.append(_layout_entry(candidate))
        inventory_objects.append(
            {
                "id": object_id,
                "name": label,
                "category": label,
                "sim_role": sim_role,
                "asset_strategy": "generated",
                "articulation_required": bool(articulation.get("required", False)),
                "must_be_separate_asset": True,
            }
        )

    manifest = {
        "version": "1.0.0",
        "scene_id": scene_id,
        "scene": {
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "environment_type": descriptor.environment_type_hint or "unknown",
            "room": {
                "bounds": room_bounds,
                "origin": room_origin,
            },
        },
        "objects": manifest_objects,
        "metadata": {
            "source_pipeline": "capture-nurec-swap",
            "source": {
                "type": "capture_nurec_swap",
                "capture_id": capture_id,
                "descriptor_uri": descriptor_uri,
                "qa_report_uri": descriptor.qa_report_uri,
            },
            "visual_refinement": {
                "hallucinated_region_mask": hallucinated_region_mask,
            },
            "provenance": {
                "descriptor": descriptor.to_dict(),
                "nurec_outputs": dict(nurec_outputs),
                "generated_at": utc_now_iso(),
            },
        },
    }

    layout = {
        "scene_id": scene_id,
        "objects": layout_objects,
        "room_box": room_box,
        "wall_thickness_m": 0.12,
        "openings": [],
    }

    inventory = {
        "scene_id": scene_id,
        "source": "capture_nurec_swap",
        "environment_type": descriptor.environment_type_hint or "unknown",
        "objects": inventory_objects,
    }

    manifest_path = assets_root / "scene_manifest.json"
    layout_path = layout_root / "scene_layout_scaled.json"
    inventory_path = seg_root / "inventory.json"

    write_json(manifest_path, manifest)
    write_json(layout_path, layout)
    write_json(inventory_path, inventory)

    return {
        "manifest_path": manifest_path,
        "layout_path": layout_path,
        "inventory_path": inventory_path,
    }
