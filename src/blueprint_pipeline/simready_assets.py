"""Deterministic local simulator-review asset lane.

This module emits review artifacts for Isaac Sim, MuJoCo, and PyBullet without
running those simulators, downloading models, or calling live providers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .local_capture import resolve_local_capture_context
from .site_memory_utils import load_jsonl

SIMREADY_SCENE_SCHEMA_VERSION = "simready_scene_manifest.v1"
SIMREADY_VALIDATION_SCHEMA_VERSION = "simready_validation.v1"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "simulator_asset_review_only",
    "simulator_execution_proven": False,
    "live_provider_execution_proven": False,
    "model_downloads_performed": False,
    "robot_readiness_proven": False,
    "disallowed_claims": [
        "simulator_execution_completed",
        "robot_policy_ready",
        "robot_deployment_ready",
        "physics_validated",
        "provider_native_world_model_ready",
        "live_runtime_ready",
    ],
    "robot_readiness_requires": [
        "real simulator load trace",
        "simulator action logs",
        "physics/contact validation logs",
        "robot profile or URDF/USD controlled by the robot team",
        "real robot or accepted simulator trial evidence",
    ],
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_exists(base_dir: Path, target: Path) -> str:
    return _relative_to(base_dir, target) if target.exists() else ""


def _stable_slug(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"n_{text}"
    return text[:80]


def _xml_name(value: Any, *, fallback: str) -> str:
    text = _stable_slug(value, fallback=fallback)
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)


def _float_list(value: Any, *, fallback: Sequence[float], minimum: float = 0.0) -> List[float]:
    values: List[float] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value[:3]:
            try:
                parsed = float(item)
            except (TypeError, ValueError):
                parsed = 0.0
            values.append(max(minimum, parsed))
    while len(values) < 3:
        values.append(float(fallback[len(values)]))
    return values[:3]


def _box_from_object(obj: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("placement_bbox", "boundingBox", "bbox", "obb"):
        raw = obj.get(key)
        if isinstance(raw, Mapping):
            center = _float_list(raw.get("center"), fallback=(0.0, 0.0, 0.25))
            extents = _float_list(
                raw.get("extents") or raw.get("size") or raw.get("dimensions"),
                fallback=(0.25, 0.25, 0.25),
                minimum=0.02,
            )
            return {"center": center, "extents": extents}
    return {"center": [0.0, 0.0, 0.25], "extents": [0.25, 0.25, 0.25]}


def _normalize_objects(object_geometry_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_objects = object_geometry_manifest.get("objects")
    if not isinstance(raw_objects, list):
        return []
    objects: List[Dict[str, Any]] = []
    for index, obj in enumerate(raw_objects):
        if not isinstance(obj, Mapping):
            continue
        object_id = str(obj.get("object_id") or obj.get("id") or f"object_{index}").strip()
        if not object_id:
            object_id = f"object_{index}"
        bbox = _box_from_object(obj)
        objects.append(
            {
                "object_id": object_id,
                "label": str(obj.get("label") or obj.get("class_name") or "object"),
                "task_role": str(obj.get("task_role") or ""),
                "center": bbox["center"],
                "extents": bbox["extents"],
                "collision_proxy": "box",
                "has_collision_hulls": bool(obj.get("collision_hulls")),
                "has_support_surfaces": bool(obj.get("support_surfaces")),
                "provenance": dict(obj.get("provenance") or {})
                if isinstance(obj.get("provenance"), Mapping)
                else {},
            }
        )
    return sorted(objects, key=lambda item: item["object_id"])


def _task_list(task_anchor_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tasks = task_anchor_manifest.get("tasks")
    if not isinstance(tasks, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("task_id") or task.get("id") or f"task_{index}").strip()
        out.append(
            {
                "task_id": task_id,
                "task_text": str(task.get("task_text") or task.get("name") or task_id),
                "task_category": str(task.get("task_category") or "generic"),
                "target_object_ids": _string_list(task.get("target_object_ids")),
                "articulation_required_ids": _string_list(task.get("articulation_required_ids")),
                "start_zone": _float_list(task.get("start_zone"), fallback=(0.0, 0.0, 0.0)),
                "goal_zone": _float_list(task.get("goal_zone"), fallback=(0.0, 0.0, 0.0)),
                "task_critical": bool(task.get("task_critical")),
            }
        )
    return out


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _resolve_site_id(*, descriptor: Mapping[str, Any], raw_manifest: Mapping[str, Any]) -> str:
    candidates: List[Any] = [descriptor.get("site_id"), raw_manifest.get("site_id")]
    for payload in (descriptor, raw_manifest, descriptor.get("metadata")):
        if isinstance(payload, Mapping):
            identity = payload.get("site_identity")
            if isinstance(identity, Mapping):
                candidates.append(identity.get("site_id"))
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _site_reference_root(context: Any, site_id: str) -> Optional[Path]:
    if not site_id:
        return None
    candidates = [
        context.storage_root / context.bucket / "sites" / site_id / "reference_memory",
        context.storage_root / "sites" / site_id / "reference_memory",
        context.capture_root / "sites" / site_id / "reference_memory",
    ]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _site_reference_summary(*, context: Any, descriptor: Mapping[str, Any]) -> Dict[str, Any]:
    raw_manifest = _read_optional_mapping(context.raw_root / "manifest.json")
    site_id = _resolve_site_id(descriptor=descriptor, raw_manifest=raw_manifest)
    root = _site_reference_root(context, site_id)
    if not site_id:
        return {
            "schema_version": "simready_site_reference_summary.v1",
            "status": "blocked",
            "site_id": None,
            "blockers": ["missing_site_id"],
            "dense_payload_policy": "summary_only_no_dense_rows",
        }
    manifest_path = (root / "site_reference_manifest.json") if root is not None else Path()
    index_path = (root / "site_reference_index.jsonl") if root is not None else Path()
    validation_path = (root / "retrieval_validation.json") if root is not None else Path()
    manifest = _read_optional_mapping(manifest_path)
    validation = _read_optional_mapping(validation_path)
    rows = load_jsonl(index_path)
    sample: List[Dict[str, Any]] = []
    zones: Dict[str, int] = {}
    chunks: Dict[str, int] = {}
    anchors: Dict[str, int] = {}
    for row in rows:
        zone_id = str(row.get("zone_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        if zone_id:
            zones[zone_id] = zones.get(zone_id, 0) + 1
        if chunk_id:
            chunks[chunk_id] = chunks.get(chunk_id, 0) + 1
        for anchor in row.get("anchor_observations") or []:
            text = str(anchor.get("anchor_id") if isinstance(anchor, Mapping) else anchor).strip()
            if text:
                anchors[text] = anchors.get(text, 0) + 1
        if len(sample) < 5:
            sample.append(
                {
                    "reference_id": row.get("reference_id"),
                    "capture_id": row.get("capture_id"),
                    "frame_id": row.get("frame_id"),
                    "zone_id": row.get("zone_id"),
                    "chunk_id": row.get("chunk_id"),
                    "geometry_source": row.get("geometry_source"),
                    "privacy_source": row.get("privacy_source"),
                }
            )
    blockers: List[str] = []
    if not manifest_path.is_file():
        blockers.append("missing_site_reference_manifest")
    if not index_path.is_file():
        blockers.append("missing_site_reference_index")
    return {
        "schema_version": "simready_site_reference_summary.v1",
        "status": "available" if not blockers else "blocked",
        "site_id": site_id,
        "blockers": blockers,
        "manifest_path": str(manifest_path.resolve()) if manifest_path else "",
        "index_path": str(index_path.resolve()) if index_path else "",
        "validation_path": str(validation_path.resolve()) if validation_path.is_file() else "",
        "counts": {
            "manifest_total_reference_frames": int(manifest.get("total_reference_frames") or 0),
            "index_row_count": len(rows),
            "capture_count": int(manifest.get("capture_count") or 0),
            "chunk_count": int(manifest.get("chunk_count") or len(chunks)),
            "zone_count": len(zones),
            "anchor_count": len(anchors),
        },
        "readiness": dict(manifest.get("readiness") or {})
        if isinstance(manifest.get("readiness"), Mapping)
        else {},
        "validation_status": str(validation.get("status") or validation.get("readiness_state") or ""),
        "sample_references": sample,
        "dense_payload_policy": "summary_only_no_dense_rows",
    }


def _geometry_truth(geometry_summary: Mapping[str, Any]) -> Dict[str, Any]:
    geometry_source = str(geometry_summary.get("geometry_source") or "missing").strip()
    fallback_used = bool(geometry_summary.get("fallback_used"))
    provider_native_result = bool(geometry_summary.get("provider_native_result"))
    geometry_live_ready = bool(geometry_summary.get("geometry_live_ready"))
    ready_for_world_model = bool(geometry_summary.get("ready_for_world_model"))
    return {
        "geometry_source": geometry_source,
        "fallback_used": fallback_used,
        "provider_native_result": provider_native_result,
        "geometry_live_ready": geometry_live_ready,
        "ready_for_world_model": ready_for_world_model,
        "site_frame_available": bool(geometry_summary.get("site_frame_available")),
        "scale_resolved": bool(geometry_summary.get("scale_resolved")),
        "site_faithful_market_ready": bool(geometry_summary.get("site_faithful_market_ready")),
        "launch_blockers": _string_list(geometry_summary.get("launch_blockers")),
    }


def _scene_bounds(objects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not objects:
        return {"center": [0.0, 0.0, 0.0], "extents": [2.0, 2.0, 0.05]}
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for obj in objects:
        center = _float_list(obj.get("center"), fallback=(0.0, 0.0, 0.0))
        extents = _float_list(obj.get("extents"), fallback=(0.25, 0.25, 0.25), minimum=0.02)
        for idx in range(3):
            half = extents[idx] * 0.5
            mins[idx] = min(mins[idx], center[idx] - half)
            maxs[idx] = max(maxs[idx], center[idx] + half)
    center = [round((mins[idx] + maxs[idx]) * 0.5, 6) for idx in range(3)]
    extents = [round(max(0.1, maxs[idx] - mins[idx]), 6) for idx in range(3)]
    return {"center": center, "extents": extents}


def _write_isaac_usda(path: Path, *, scene_name: str, objects: Sequence[Mapping[str, Any]]) -> None:
    def _q(value: Any) -> str:
        return str(value or "").replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "BlueprintSiteScene"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "BlueprintSiteScene" (',
        "    customData = {",
        '        string blueprint_claim_boundary = "simulator_review_artifact_only_no_execution"',
        f'        string blueprint_scene_name = "{_q(scene_name)}"',
        "    }",
        ")",
        "{",
        '    def Cube "floor_proxy"',
        "    {",
        "        double size = 1",
        "        double3 xformOp:translate = (0, 0, -0.025)",
        "        double3 xformOp:scale = (8, 8, 0.05)",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        "    }",
    ]
    for obj in objects:
        name = _stable_slug(obj.get("object_id"), fallback="object")
        center = _float_list(obj.get("center"), fallback=(0.0, 0.0, 0.0))
        extents = _float_list(obj.get("extents"), fallback=(0.25, 0.25, 0.25), minimum=0.02)
        lines.extend(
            [
                f'    def Cube "{name}"',
                "    {",
                "        double size = 1",
                f"        double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})",
                f"        double3 xformOp:scale = ({extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f})",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                f'        custom string blueprint_object_id = "{_q(obj.get("object_id"))}"',
                f'        custom string blueprint_label = "{_q(obj.get("label"))}"',
                "    }",
            ]
        )
    lines.append("}")
    write_text(path, "\n".join(lines) + "\n")


def _indent_xml(root: ET.Element) -> None:
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        pass


def _write_mujoco_mjcf(path: Path, *, scene_name: str, objects: Sequence[Mapping[str, Any]]) -> None:
    root = ET.Element("mujoco", {"model": _xml_name(scene_name, fallback="blueprint_site")})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "review_light", "pos": "0 0 4", "dir": "0 0 -1"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor_proxy",
            "type": "plane",
            "pos": "0 0 0",
            "size": "8 8 0.05",
            "rgba": "0.75 0.78 0.80 1",
        },
    )
    for obj in objects:
        name = _xml_name(obj.get("object_id"), fallback="object")
        center = _float_list(obj.get("center"), fallback=(0.0, 0.0, 0.0))
        extents = _float_list(obj.get("extents"), fallback=(0.25, 0.25, 0.25), minimum=0.02)
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": name,
                "pos": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_collision_proxy",
                "type": "box",
                "size": f"{extents[0] * 0.5:.6f} {extents[1] * 0.5:.6f} {extents[2] * 0.5:.6f}",
                "rgba": "0.2 0.45 0.85 0.55",
            },
        )
    _indent_xml(root)
    ensure_dir(path.parent)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_pybullet_urdf(path: Path, *, scene_name: str, objects: Sequence[Mapping[str, Any]]) -> None:
    robot = ET.Element("robot", {"name": _xml_name(scene_name, fallback="blueprint_site")})
    ET.SubElement(robot, "link", {"name": "world"})
    for obj in objects:
        name = _xml_name(obj.get("object_id"), fallback="object")
        center = _float_list(obj.get("center"), fallback=(0.0, 0.0, 0.0))
        extents = _float_list(obj.get("extents"), fallback=(0.25, 0.25, 0.25), minimum=0.02)
        link = ET.SubElement(robot, "link", {"name": name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", {"value": "1.0"})
        ET.SubElement(
            inertial,
            "inertia",
            {"ixx": "0.01", "ixy": "0", "ixz": "0", "iyy": "0.01", "iyz": "0", "izz": "0.01"},
        )
        for section in ("visual", "collision"):
            node = ET.SubElement(link, section)
            geometry = ET.SubElement(node, "geometry")
            ET.SubElement(
                geometry,
                "box",
                {"size": f"{extents[0]:.6f} {extents[1]:.6f} {extents[2]:.6f}"},
            )
        joint = ET.SubElement(robot, "joint", {"name": f"world_to_{name}", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "world"})
        ET.SubElement(joint, "child", {"link": name})
        ET.SubElement(joint, "origin", {"xyz": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}", "rpy": "0 0 0"})
    _indent_xml(robot)
    ensure_dir(path.parent)
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def _write_framework_artifacts(
    *,
    sim_dir: Path,
    scene_name: str,
    objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    isaac_path = sim_dir / "isaac_sim" / "site_scene.usda"
    mujoco_path = sim_dir / "mujoco" / "site_scene.xml"
    pybullet_path = sim_dir / "pybullet" / "site_scene.urdf"
    _write_isaac_usda(isaac_path, scene_name=scene_name, objects=objects)
    _write_mujoco_mjcf(mujoco_path, scene_name=scene_name, objects=objects)
    _write_pybullet_urdf(pybullet_path, scene_name=scene_name, objects=objects)
    return {
        "isaac_sim": {
            "format": "OpenUSD_USDA",
            "path": str(isaac_path.resolve()),
            "review_target": "Isaac Sim asset import and Asset Validator review",
            "load_status": "not_executed",
            "execution_claim": False,
        },
        "mujoco": {
            "format": "MJCF_XML",
            "path": str(mujoco_path.resolve()),
            "review_target": "MuJoCo MJCF compile/load review",
            "load_status": "not_executed",
            "execution_claim": False,
        },
        "pybullet": {
            "format": "URDF_XML",
            "path": str(pybullet_path.resolve()),
            "review_target": "PyBullet loadURDF/loadSDF-compatible review",
            "load_status": "not_executed",
            "execution_claim": False,
        },
    }


def _validation_payload(
    *,
    objects: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    robot_profiles: Sequence[Mapping[str, Any]],
    site_reference_summary: Mapping[str, Any],
    geometry_truth: Mapping[str, Any],
    framework_artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []
    if not objects:
        blockers.append("missing_object_geometry_manifest")
    if not tasks:
        blockers.append("missing_task_anchor_manifest")
    if not robot_profiles:
        blockers.append("missing_robot_profiles")
    if str(site_reference_summary.get("status") or "") != "available":
        warnings.extend(_string_list(site_reference_summary.get("blockers")))
    if not bool(geometry_truth.get("geometry_live_ready")):
        warnings.append("geometry_not_live_sim_or_video_to_world_proof")
    if bool(geometry_truth.get("fallback_used")):
        warnings.append("fallback_geometry_review_only")
    for framework, detail in framework_artifacts.items():
        if not Path(str(detail.get("path") or "")).is_file():
            blockers.append(f"missing_{framework}_artifact")
    overall_status = "blocked" if blockers else "degraded" if warnings else "prepared_for_review"
    return {
        "schema_version": SIMREADY_VALIDATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "overall_status": overall_status,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "frameworks": {
            key: {
                "format": value.get("format"),
                "path": value.get("path"),
                "load_status": value.get("load_status"),
                "execution_claim": False,
            }
            for key, value in framework_artifacts.items()
        },
    }


def _sha_payload(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(data).hexdigest()


def build_simready_assets(
    *,
    capture_root: str | Path,
    object_geometry_manifest: Optional[Mapping[str, Any]] = None,
    task_anchor_manifest: Optional[Mapping[str, Any]] = None,
    site_world_spec: Optional[Mapping[str, Any]] = None,
    hosted_session_runtime_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    eval_dir = pipeline_dir / "evaluation_prep"
    sim_dir = pipeline_dir / "simready"
    ensure_dir(sim_dir)

    descriptor = _read_optional_mapping(context.descriptor_path)
    object_geometry = dict(
        object_geometry_manifest
        or _read_optional_mapping(eval_dir / "object_geometry_manifest.json")
    )
    task_anchor = dict(
        task_anchor_manifest
        or _read_optional_mapping(eval_dir / "task_anchor_manifest.json")
    )
    site_world = dict(site_world_spec or _read_optional_mapping(eval_dir / "site_world_spec.json"))
    hosted_manifest = dict(
        hosted_session_runtime_manifest
        or _read_optional_mapping(eval_dir / "hosted_session_runtime_manifest.json")
    )
    geometry_summary = _read_optional_mapping(pipeline_dir / "geometry" / "geometry_summary.json")
    geometry_truth = _geometry_truth(geometry_summary)
    site_reference = _site_reference_summary(context=context, descriptor=descriptor)
    objects = _normalize_objects(object_geometry)
    tasks = _task_list(task_anchor)
    robot_profiles = [
        dict(item)
        for item in (
            site_world.get("robot_profiles")
            if isinstance(site_world.get("robot_profiles"), list)
            else hosted_manifest.get("robot_profiles")
            if isinstance(hosted_manifest.get("robot_profiles"), list)
            else []
        )
        if isinstance(item, Mapping)
    ]
    scene_name = f"{context.scene_id}_{context.capture_id}"
    scene_bounds = _scene_bounds(objects)
    framework_artifacts = _write_framework_artifacts(
        sim_dir=sim_dir,
        scene_name=scene_name,
        objects=objects,
    )
    evidence_boundaries = {
        "schema_version": "simready_evidence_boundaries.v1",
        "generated_at": utc_now_iso(),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "raw_capture_authority": {
            "raw_root": str(context.raw_root.resolve()),
            "descriptor_path": str(context.descriptor_path.resolve()),
            "rule": "Raw capture, provenance, rights, privacy, timestamps, poses, and device metadata remain authoritative.",
        },
        "geometry_truth": dict(geometry_truth),
        "site_reference_policy": {
            "summary_only": True,
            "dense_rows_included": False,
            "dense_artifacts_allowed_in_manifest": False,
        },
        "simulator_execution": {
            "isaac_sim": "not_executed",
            "mujoco": "not_executed",
            "pybullet": "not_executed",
        },
    }
    task_scenarios = {
        "schema_version": "simready_task_scenarios.v1",
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "tasks": tasks,
        "robot_profile_ids": [str(item.get("id") or "") for item in robot_profiles],
        "claim_boundary": "scenario_review_only_no_action_trace",
    }
    source_artifacts = {
        "capture_descriptor": str(context.descriptor_path.resolve()),
        "object_geometry_manifest": _relative_if_exists(sim_dir, eval_dir / "object_geometry_manifest.json"),
        "task_anchor_manifest": _relative_if_exists(sim_dir, eval_dir / "task_anchor_manifest.json"),
        "site_world_spec": _relative_if_exists(sim_dir, eval_dir / "site_world_spec.json"),
        "hosted_session_runtime_manifest": _relative_if_exists(
            sim_dir,
            eval_dir / "hosted_session_runtime_manifest.json",
        ),
        "geometry_summary": _relative_if_exists(sim_dir, pipeline_dir / "geometry" / "geometry_summary.json"),
    }
    compact_frameworks = {
        key: {**dict(value), "path": _relative_to(sim_dir, Path(str(value["path"])))}
        for key, value in framework_artifacts.items()
    }
    scene_manifest = {
        "schema_version": SIMREADY_SCENE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "site_id": site_reference.get("site_id"),
        "scene_name": scene_name,
        "scene_bounds": scene_bounds,
        "object_count": len(objects),
        "task_count": len(tasks),
        "robot_profile_count": len(robot_profiles),
        "geometry_truth": geometry_truth,
        "site_reference_summary_path": "site_reference_summary.json",
        "evidence_boundaries_path": "evidence_boundaries.json",
        "task_scenarios_path": "task_scenarios.json",
        "robot_profiles_path": "robot_profiles.json",
        "source_artifacts": source_artifacts,
        "framework_artifacts": compact_frameworks,
        "objects": objects,
        "tasks": tasks,
        "robot_profiles": robot_profiles,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    scene_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": scene_manifest["scene_id"],
            "capture_id": scene_manifest["capture_id"],
            "site_id": scene_manifest["site_id"],
            "objects": objects,
            "tasks": tasks,
            "robot_profiles": robot_profiles,
            "framework_artifacts": compact_frameworks,
            "geometry_truth": geometry_truth,
        }
    )
    validation = _validation_payload(
        objects=objects,
        tasks=tasks,
        robot_profiles=robot_profiles,
        site_reference_summary=site_reference,
        geometry_truth=geometry_truth,
        framework_artifacts=framework_artifacts,
    )
    scene_manifest["status"] = validation["overall_status"]

    write_json(sim_dir / "site_reference_summary.json", site_reference)
    write_json(sim_dir / "evidence_boundaries.json", evidence_boundaries)
    write_json(sim_dir / "task_scenarios.json", task_scenarios)
    write_json(sim_dir / "robot_profiles.json", {"schema_version": "simready_robot_profiles.v1", "robot_profiles": robot_profiles})
    write_json(sim_dir / "framework_review_manifest.json", {"schema_version": "simready_framework_review_manifest.v1", "frameworks": compact_frameworks})
    write_json(sim_dir / "simready_scene_manifest.json", scene_manifest)
    write_json(sim_dir / "simready_validation.json", validation)
    return {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "manifest_path": str((sim_dir / "simready_scene_manifest.json").resolve()),
        "validation_path": str((sim_dir / "simready_validation.json").resolve()),
        "status": validation["overall_status"],
        "framework_artifacts": compact_frameworks,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build local simulator-review artifacts without executing simulators"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    args = parser.parse_args(argv)

    try:
        result = build_simready_assets(capture_root=args.capture_root)
    except (PipelineError, ValueError, OSError) as exc:
        print(f"[simready] FAILED: {exc}")
        return 1
    print(f"[simready] manifest={result['manifest_path']}")
    print(f"[simready] validation={result['validation_path']}")
    print(f"[simready] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
