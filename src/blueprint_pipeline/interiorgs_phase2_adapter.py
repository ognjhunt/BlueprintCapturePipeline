"""Adapt InteriorGS scene exports into a synthetic capture tree for Phase 2 review."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .agent_runtime.openai_phase2 import OpenAIPhase2Config
from .agent_runtime.orchestrator import run_agent_review
from .capture_bridge import CaptureDescriptor
from .common import ensure_dir, to_capture_prefix, to_pipeline_prefix, utc_now_iso, write_json, write_text
from .interiorgs_phase2_summary import write_scene_dashboard_summary, write_scene_deployment_summary
from .qualification import (
    _scene_memory_derived_assets,
    _build_blocker_register,
    _build_capability_checks,
    _build_capture_package_manifest,
    _build_completeness_scorecard,
    _build_geometry_evidence,
    _build_human_actions_required,
    _build_opportunity_handoff,
    _build_pipeline_summary,
    _build_qualification_brief,
    _build_qualification_record,
    _build_readiness_decision,
    _build_route_graph,
    _build_runtime_preflight_report,
    _build_scene_graph,
    _build_task_hypothesis_report,
    _build_task_scope_record,
    _effective_task_metadata,
    _render_readiness_report,
    _write_scene_memory_bundle,
    attach_handoff_package_paths,
)
from .evaluation_prep_stage import run_evaluation_prep_stage
from .simready_stage import run_simready_stage
from .task_targets import write_task_targets
from .webapp_sync import (
    derive_webapp_opportunity_state,
    derive_webapp_qualification_state,
    sync_webapp_pipeline_attachment,
)


_DEFAULT_BUCKET = "localbucket"
_DEFAULT_OUTPUT_ROOT = Path("/tmp/blueprint_interiorgs_phase2")


@dataclass(frozen=True)
class InteriorGSAdaptationResult:
    source_dir: Path
    capture_root: Path
    scene_id: str
    capture_id: str
    provider: Optional[str]
    final_bundle_path: Optional[str]
    final_memo_path: Optional[str]
    evaluation_prep_manifest_path: Optional[str]
    simready_manifest_path: Optional[str]
    simready_scene_path: Optional[str]


@dataclass(frozen=True)
class InteriorGSTaskRunResult:
    category: str
    task_text: str
    capture_root: Path
    capture_id: str
    final_bundle_path: Optional[str]
    final_memo_path: Optional[str]
    evaluation_prep_manifest_path: Optional[str]
    simready_manifest_path: Optional[str]
    simready_scene_path: Optional[str]


def _read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_any(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_name_parts(source_dir: Path) -> tuple[str, str]:
    stem = source_dir.name.strip()
    if "_" in stem:
        scene_id, capture_id = stem.split("_", 1)
        return scene_id or stem, capture_id or "interiorgs"
    return stem or "interiorgs_scene", "interiorgs"


def _bbox_from_points(points: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    xs = [float(point.get("x") or 0.0) for point in points]
    ys = [float(point.get("y") or 0.0) for point in points]
    zs = [float(point.get("z") or 0.0) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    return {
        "center": [
            round((min_x + max_x) * 0.5, 6),
            round((min_y + max_y) * 0.5, 6),
            round((min_z + max_z) * 0.5, 6),
        ],
        "extents": [
            round(max_x - min_x, 6),
            round(max_y - min_y, 6),
            round(max_z - min_z, 6),
        ],
        "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
    }


def _convert_label_entries(labels: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for entry in labels:
        if not isinstance(entry, Mapping):
            continue
        instance_id = str(entry.get("ins_id") or "").strip()
        label = str(entry.get("label") or "object").strip() or "object"
        points = entry.get("bounding_box")
        if not instance_id or not isinstance(points, list) or len(points) < 2:
            continue
        converted.append(
            {
                "id": instance_id,
                "object_id": instance_id,
                "label": label,
                "name": label,
                "boundingBox": _bbox_from_points(points),
                "source": "interiorgs_labels",
            }
        )
    return converted


def _infer_environment_hint(facility_name: str, tasks: Sequence[Mapping[str, Any]]) -> str:
    text = " ".join(
        [facility_name, *[str(item.get("task_id") or "") for item in tasks if isinstance(item, Mapping)]]
    ).lower()
    if "kitchen" in text:
        return "kitchen"
    if "bedroom" in text or "wardrobe" in text:
        return "bedroom"
    return "default"


def _is_residential_environment(environment_hint: str) -> bool:
    return environment_hint in {"default", "bedroom", "kitchen"}


def _room_short_side_spans(structure: Mapping[str, Any]) -> List[float]:
    spans: List[float] = []
    for room in structure.get("rooms", []):
        if not isinstance(room, Mapping):
            continue
        profile = room.get("profile")
        if not isinstance(profile, list) or len(profile) < 3:
            continue
        xs = [float(point[0]) for point in profile if isinstance(point, list) and len(point) >= 2]
        ys = [float(point[1]) for point in profile if isinstance(point, list) and len(point) >= 2]
        if not xs or not ys:
            continue
        spans.append(min(max(xs) - min(xs), max(ys) - min(ys)))
    spans.sort()
    return spans


def _room_route_entries(structure: Mapping[str, Any]) -> List[Dict[str, Any]]:
    synthetic_entries: List[Dict[str, Any]] = []
    for idx, room in enumerate(structure.get("rooms", [])):
        if not isinstance(room, Mapping):
            continue
        profile = room.get("profile")
        if not isinstance(profile, list) or len(profile) < 3:
            continue
        xs = [float(point[0]) for point in profile if isinstance(point, list) and len(point) >= 2]
        ys = [float(point[1]) for point in profile if isinstance(point, list) and len(point) >= 2]
        if not xs or not ys:
            continue
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        depth = max_y - min_y
        synthetic_entries.append(
            {
                "id": f"synthetic_route_room_{idx}",
                "object_id": f"synthetic_route_room_{idx}",
                "label": "aisle",
                "name": "aisle",
                "boundingBox": {
                    "center": [
                        round((min_x + max_x) * 0.5, 6),
                        round((min_y + max_y) * 0.5, 6),
                        1.0,
                    ],
                    "extents": [round(width, 6), round(depth, 6), 2.0],
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
                },
                "source": "interiorgs_structure",
                "synthetic_route_context": True,
            }
        )
    return synthetic_entries


def _qualification_entries(
    *,
    object_index_entries: Sequence[Mapping[str, Any]],
    structure: Mapping[str, Any],
    environment_hint: str,
) -> List[Dict[str, Any]]:
    if not _is_residential_environment(environment_hint):
        return [dict(item) for item in object_index_entries]
    filtered = [
        dict(item)
        for item in object_index_entries
        if str(item.get("label") or "").strip().lower() not in {"door", "window"}
    ]
    filtered.extend(_room_route_entries(structure))
    return filtered


def _residential_route_width_hint(structure: Mapping[str, Any], fallback: Optional[float]) -> Optional[float]:
    spans = _room_short_side_spans(structure)
    if not spans:
        return fallback
    # Use a room-scale lower quartile instead of door-leaf width so residential
    # smoke runs are not dominated by articulation geometry.
    index = max(0, min(len(spans) - 1, len(spans) // 4))
    estimate = round(spans[index], 4)
    if fallback is None:
        return estimate
    return round(max(float(fallback), estimate), 4)


def _target_object_ids(task_targets_payload: Mapping[str, Any], limit: int = 24) -> List[str]:
    ids: List[str] = []
    for collection_name in ("manipulation_candidates", "navigation_hints"):
        for item in task_targets_payload.get(collection_name, []):
            if not isinstance(item, Mapping):
                continue
            instance_id = str(item.get("instance_id") or item.get("id") or "").strip()
            if instance_id and instance_id not in ids:
                ids.append(instance_id)
            if len(ids) >= limit:
                return ids
    return ids


def _articulation_ids(task_targets_payload: Mapping[str, Any], limit: int = 24) -> List[str]:
    ids: List[str] = []
    for item in task_targets_payload.get("articulation_hints", []):
        if not isinstance(item, Mapping):
            continue
        instance_id = str(item.get("instance_id") or item.get("id") or "").strip()
        if instance_id and instance_id not in ids:
            ids.append(instance_id)
        if len(ids) >= limit:
            break
    return ids


def _candidate_lookup(collection: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in collection:
        if not isinstance(item, Mapping):
            continue
        instance_id = str(item.get("instance_id") or item.get("id") or "").strip()
        if instance_id:
            lookup[instance_id] = dict(item)
    return lookup


def _labels_for_text_match(collection: Sequence[Mapping[str, Any]], text: str) -> List[str]:
    lowered = text.lower()
    matched: List[str] = []
    for item in collection:
        if not isinstance(item, Mapping):
            continue
        instance_id = str(item.get("instance_id") or item.get("id") or "").strip()
        label = str(item.get("label") or "").strip().lower()
        if instance_id and label and label in lowered and instance_id not in matched:
            matched.append(instance_id)
    return matched


def _explicit_task_targets(raw_payload: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    manipulation = raw_payload.get("manipulation_candidates", [])
    articulation = raw_payload.get("articulation_hints", [])
    navigation = raw_payload.get("navigation_hints", [])
    manip_lookup = _candidate_lookup(manipulation if isinstance(manipulation, list) else [])
    artic_lookup = _candidate_lookup(articulation if isinstance(articulation, list) else [])
    nav_lookup = _candidate_lookup(navigation if isinstance(navigation, list) else [])
    target_ids: List[str] = []
    articulation_ids: List[str] = []
    for item in raw_payload.get("tasks", []):
        if not isinstance(item, Mapping):
            continue
        task_text = str(item.get("task_id") or "").strip()
        if not task_text:
            continue
        lowered = task_text.lower()
        explicit_ids = re.findall(r"_(\d+)\b", task_text)
        if "open and close" in lowered or lowered.startswith("open "):
            for instance_id in explicit_ids:
                if instance_id in artic_lookup and instance_id not in articulation_ids:
                    articulation_ids.append(instance_id)
            if not explicit_ids:
                for instance_id in _labels_for_text_match(articulation if isinstance(articulation, list) else [], task_text):
                    if instance_id not in articulation_ids:
                        articulation_ids.append(instance_id)
            continue
        if "navigate to" in lowered:
            for instance_id in explicit_ids:
                if instance_id in nav_lookup and instance_id not in target_ids:
                    target_ids.append(instance_id)
            if not explicit_ids:
                for instance_id in _labels_for_text_match(navigation if isinstance(navigation, list) else [], task_text):
                    if instance_id not in target_ids:
                        target_ids.append(instance_id)
            continue
        for instance_id in explicit_ids:
            if instance_id in manip_lookup and instance_id not in target_ids:
                target_ids.append(instance_id)
        if not explicit_ids:
            for instance_id in _labels_for_text_match(manipulation if isinstance(manipulation, list) else [], task_text):
                if instance_id not in target_ids:
                    target_ids.append(instance_id)
    return target_ids, articulation_ids


def _task_zone_center(
    *,
    object_index_entries: Sequence[Mapping[str, Any]],
    target_ids: Sequence[str],
    articulation_ids: Sequence[str],
) -> Optional[List[float]]:
    selected_ids = [item for item in list(target_ids) + list(articulation_ids) if item]
    if not selected_ids:
        return None
    centers: List[List[float]] = []
    for entry in object_index_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_id = str(entry.get("id") or entry.get("object_id") or "").strip()
        if entry_id not in selected_ids:
            continue
        box = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else {}
        center = box.get("center") if isinstance(box.get("center"), list) else None
        if isinstance(center, list) and len(center) >= 3:
            centers.append([float(center[0]), float(center[1]), float(center[2])])
    if not centers:
        return None
    return [
        round(sum(center[idx] for center in centers) / float(len(centers)), 6)
        for idx in range(3)
    ]


def _selected_object_metrics(
    *,
    object_index_entries: Sequence[Mapping[str, Any]],
    target_ids: Sequence[str],
    articulation_ids: Sequence[str],
    task_zone_center: Optional[Sequence[float]],
) -> Dict[str, Optional[float]]:
    by_id: Dict[str, Mapping[str, Any]] = {}
    for entry in object_index_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_id = str(entry.get("id") or entry.get("object_id") or "").strip()
        if entry_id:
            by_id[entry_id] = entry

    def _center(entry: Mapping[str, Any]) -> Optional[List[float]]:
        box = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else {}
        center = box.get("center") if isinstance(box.get("center"), list) else None
        if not isinstance(center, list) or len(center) < 3:
            return None
        return [float(center[0]), float(center[1]), float(center[2])]

    target_centers: List[List[float]] = []
    workcell_centers: List[List[float]] = []
    for instance_id in target_ids:
        entry = by_id.get(str(instance_id))
        if not entry:
            continue
        center = _center(entry)
        if center is None:
            continue
        target_centers.append(center)
        workcell_centers.append(center)
    for instance_id in articulation_ids:
        entry = by_id.get(str(instance_id))
        if not entry:
            continue
        center = _center(entry)
        if center is None:
            continue
        workcell_centers.append(center)

    target_reach: Optional[float] = None
    if task_zone_center is not None and target_centers:
        zone = [float(task_zone_center[0]), float(task_zone_center[1]), float(task_zone_center[2])]
        target_reach = round(
            max(
                (
                    (center[0] - zone[0]) ** 2
                    + (center[1] - zone[1]) ** 2
                    + (center[2] - zone[2]) ** 2
                )
                ** 0.5
                for center in target_centers
            ),
            4,
        )

    workcell_span: Optional[float] = None
    if workcell_centers:
        mins = [min(center[idx] for center in workcell_centers) for idx in range(3)]
        maxs = [max(center[idx] for center in workcell_centers) for idx in range(3)]
        workcell_span = round(max(maxs[idx] - mins[idx] for idx in range(3)), 4)

    return {
        "target_reach_distance_m": target_reach,
        "workcell_span_m": workcell_span,
    }


def _build_task_targets_payload(
    raw_payload: Mapping[str, Any],
    *,
    scene_id: str,
    capture_id: str,
    object_index_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = dict(raw_payload)
    payload.setdefault("schema_version", "v1")
    payload["scene_id"] = scene_id
    payload["capture_id"] = capture_id
    payload.setdefault("generated_at", utc_now_iso())
    payload.setdefault("inference_mode", "synthetic_import")
    payload.setdefault("video_analysis", {"external_inference": {"status": "imported", "reason": "interiorgs_adapter"}})
    explicit_target_ids, explicit_articulation_ids = _explicit_task_targets(payload)
    task_texts = [
        str(item.get("task_id") or "").strip()
        for item in payload.get("tasks", [])
        if isinstance(item, Mapping) and str(item.get("task_id") or "").strip()
    ]
    task_categories = {_task_category(text) for text in task_texts}
    if not explicit_target_ids and explicit_articulation_ids and task_categories == {"open_close"}:
        explicit_target_ids = list(explicit_articulation_ids)
    payload["target_object_ids"] = explicit_target_ids or _target_object_ids(payload, limit=12)
    payload["articulation_required_ids"] = explicit_articulation_ids or _articulation_ids(payload, limit=12)
    payload["object_index_entries"] = [dict(item) for item in object_index_entries]
    return payload


def _build_raw_task_hypothesis(raw_payload: Mapping[str, Any], facility_name: str) -> Dict[str, Any]:
    tasks = [
        str(item.get("task_id") or "").strip()
        for item in raw_payload.get("tasks", [])
        if isinstance(item, Mapping) and str(item.get("task_id") or "").strip()
    ]
    return {
        "schema_version": "v1",
        "source": "bootstrap_synthetic",
        "workflow_name": facility_name,
        "task_steps": tasks[:8],
        "target_kpi": "Grounded object and route coverage",
        "zone": facility_name,
        "owner": "interiorgs_adapter",
        "confidence": 0.95,
        "warnings": [],
    }


def _write_placeholder_raw_files(
    *,
    capture_root: Path,
    scene_id: str,
    capture_id: str,
    object_index_entries: Sequence[Mapping[str, Any]],
    raw_task_hypothesis: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    raw_root = capture_root / "raw"
    frames_root = capture_root / "frames"
    ensure_dir(raw_root)
    ensure_dir(frames_root)
    (raw_root / "walkthrough.mp4").write_bytes(b"interiorgs")
    write_text(
        frames_root / "index.jsonl",
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": scene_id,
                "capture_id": capture_id,
                "source": "interiorgs_adapter",
                "generated_at": utc_now_iso(),
            }
        )
        + "\n",
    )
    write_json(
        raw_root / "manifest.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_source": "iphone",
            "capture_tier_hint": "tier1_iphone",
            "has_lidar": True,
            "pose_match_rate": 0.96,
            "video_uri": "raw/walkthrough.mp4",
            "object_point_cloud_index": "object_index.json",
            "capture_modality": "iphone_arkit_lidar",
            "evidence_tier": "qualified_metric_capture",
        },
    )
    write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": metadata.get("task_statement"),
            "taskSteps": metadata.get("workflow_decomposition", []),
            "targetKPI": (metadata.get("success_criteria") or ["Grounded object and route coverage"])[0],
            "zone": (metadata.get("task_zone") or {}).get("label"),
            "owner": metadata.get("owner"),
        },
    )
    write_json(raw_root / "task_hypothesis.json", dict(raw_task_hypothesis))
    write_json(raw_root / "object_index.json", {"objects": [dict(item) for item in object_index_entries]})


def _copy_or_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _task_category(task_text: str) -> str:
    lowered = task_text.strip().lower()
    if lowered.startswith("pick up "):
        return "pick"
    if lowered.startswith("open and close "):
        return "open_close"
    if lowered.startswith("navigate to "):
        return "navigate"
    return "other"


def _slugify_task(task_text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", task_text.strip().lower()).strip("-")
    return slug[:limit] or "task"


def _payload_center(item: Mapping[str, Any]) -> Optional[List[float]]:
    box = item.get("boundingBox") if isinstance(item.get("boundingBox"), Mapping) else {}
    center = box.get("center") if isinstance(box.get("center"), list) else None
    if not isinstance(center, list) or len(center) < 3:
        return None
    return [float(center[0]), float(center[1]), float(center[2])]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return (
        ((float(a[0]) - float(b[0])) ** 2)
        + ((float(a[1]) - float(b[1])) ** 2)
        + ((float(a[2]) - float(b[2])) ** 2)
    ) ** 0.5


def _candidate_ids(collection: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = _candidate_lookup(collection)
    return [lookup[item] for item in ids if item in lookup]


def _nearby_ids(
    collection: Sequence[Mapping[str, Any]],
    *,
    reference_centers: Sequence[Sequence[float]],
    radius_m: float,
    limit: int,
    exclude_ids: Sequence[str] = (),
) -> List[str]:
    if not reference_centers:
        return []
    selected: List[tuple[float, str]] = []
    excluded = set(str(item) for item in exclude_ids)
    for item in collection:
        if not isinstance(item, Mapping):
            continue
        instance_id = str(item.get("instance_id") or item.get("id") or "").strip()
        if not instance_id or instance_id in excluded:
            continue
        center = _payload_center(item)
        if center is None:
            continue
        best = min(_distance(center, ref) for ref in reference_centers)
        if best <= radius_m:
            selected.append((best, instance_id))
    selected.sort(key=lambda pair: pair[0])
    return [instance_id for _, instance_id in selected[:limit]]


def _filter_labels_for_ids(labels: Sequence[Mapping[str, Any]], keep_ids: Sequence[str]) -> List[Mapping[str, Any]]:
    keep = set(str(item) for item in keep_ids)
    return [
        item
        for item in labels
        if isinstance(item, Mapping) and str(item.get("ins_id") or "").strip() in keep
    ]


def _write_advanced_geometry_bundle(
    *,
    source_dir: Path,
    pipeline_dir: Path,
    scene_id: str,
    capture_id: str,
) -> None:
    advanced_dir = pipeline_dir / "advanced_geometry"
    ensure_dir(advanced_dir)
    copied: Dict[str, str] = {}
    for name in ("labels.json", "structure.json", "task_targets.synthetic.json", "3dgs_compressed.ply"):
        src = source_dir / name
        if not src.is_file():
            continue
        dst = advanced_dir / name
        _copy_or_link(src, dst)
        copied[name] = str(dst)
    write_json(
        advanced_dir / "advanced_geometry_bundle.json",
        {
            "schema_version": "v1",
            "scene_id": scene_id,
            "capture_id": capture_id,
            "generated_at": utc_now_iso(),
            "source": "interiorgs_adapter",
            "files": copied,
        },
    )


def adapt_interiorgs_scene(
    *,
    source_dir: Path,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    run_phase2: bool = False,
    run_evaluation_prep: bool = False,
    run_simready: bool = False,
    provider: str = "openai",
    evaluation_prep_provider: str = "manual",
    simready_provider: str = "manual",
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
) -> InteriorGSAdaptationResult:
    source_dir = source_dir.resolve()
    scene_id, capture_id = _safe_name_parts(source_dir)
    capture_root = output_root / bucket / "scenes" / scene_id / "captures" / capture_id
    pipeline_dir = capture_root / "pipeline"
    ensure_dir(pipeline_dir)

    labels = _read_json_any(source_dir / "labels.json")
    structure = _read_json_any(source_dir / "structure.json")
    synthetic_targets = _read_json_any(source_dir / "task_targets.synthetic.json")
    if not isinstance(labels, list):
        raise ValueError(f"labels.json must contain a list: {source_dir / 'labels.json'}")
    if not isinstance(structure, Mapping):
        raise ValueError(f"structure.json must contain an object: {source_dir / 'structure.json'}")
    if not isinstance(synthetic_targets, Mapping):
        raise ValueError(f"task_targets.synthetic.json must contain an object: {source_dir / 'task_targets.synthetic.json'}")

    object_index_entries = _convert_label_entries(labels)
    facility_name = str(synthetic_targets.get("facility_name") or source_dir.name)
    tasks = synthetic_targets.get("tasks", []) if isinstance(synthetic_targets.get("tasks"), list) else []
    environment_hint = _infer_environment_hint(facility_name, tasks)
    qualification_entries = _qualification_entries(
        object_index_entries=object_index_entries,
        structure=structure,
        environment_hint=environment_hint,
    )
    workflow_decomposition = [
        str(item.get("task_id") or "").strip()
        for item in tasks[:8]
        if isinstance(item, Mapping) and str(item.get("task_id") or "").strip()
    ]
    metadata: Dict[str, Any] = {
        "site_submission_id": f"{scene_id}:{capture_id}",
        "opportunity_id": source_dir.name,
        "task_statement": facility_name,
        "workflow_context": " | ".join(workflow_decomposition) if workflow_decomposition else facility_name,
        "workflow_decomposition": workflow_decomposition,
        "success_criteria": ["Grounded object and route coverage"],
        "task_zone": {"label": facility_name},
        "owner": "interiorgs_adapter",
        "adjacent_systems": [f"{len(structure.get('rooms', []))} room layout"],
        "non_routine_modes": ["Synthetic validation replay"],
        "people_traffic_notes": ["Synthetic InteriorGS validation scene"],
        "privacy_restrictions": [],
        "security_restrictions": [],
        "known_blockers": [],
        "capture_restrictions": [],
        "scene_package_path": str(source_dir),
    }
    raw_task_hypothesis = _build_raw_task_hypothesis(synthetic_targets, facility_name)
    _write_placeholder_raw_files(
        capture_root=capture_root,
        scene_id=scene_id,
        capture_id=capture_id,
        object_index_entries=object_index_entries,
        raw_task_hypothesis=raw_task_hypothesis,
        metadata=metadata,
    )
    qa_report = {
        "schema_version": "v1",
        "status": "passed",
        "uncertainty_score": 0.12,
        "hidden_zone_score": 0.08,
        "hidden_zone_bound": 0.15,
        "generated_at": utc_now_iso(),
        "source": "interiorgs_adapter",
        "structure_room_count": len(structure.get("rooms", [])),
        "object_count": len(object_index_entries),
    }
    qa_report_path = capture_root / "qa_report.json"
    write_json(qa_report_path, qa_report)

    capture_prefix = to_capture_prefix(scene_id, capture_id)
    raw_prefix_uri = f"gs://{bucket}/{capture_prefix}/raw"
    frames_index_uri = f"gs://{bucket}/{capture_prefix}/frames/index.jsonl"
    descriptor_payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_source": "iphone",
        "capture_tier": "tier1_iphone",
        "raw_prefix_uri": raw_prefix_uri,
        "frames_index_uri": frames_index_uri,
        "nurec_mode": "mono_pose_assisted",
        "quality": {
            "pose_match_rate": 0.96,
            "has_metric_geometry": True,
            "intake_complete": True,
        },
        "raw_video_uri": f"{raw_prefix_uri}/walkthrough.mp4",
        "qa_report_uri": f"gs://{bucket}/{capture_prefix}/qa_report.json",
        "qa_status": "passed",
        "environment_type_hint": environment_hint,
        "capture_modality": "iphone_arkit_lidar",
        "evidence_tier": "qualified_metric_capture",
        "intake_packet_uri": f"{raw_prefix_uri}/intake_packet.json",
        "task_hypothesis_uri": f"{raw_prefix_uri}/task_hypothesis.json",
        "coverage_plan": [
            "Traverse all rooms and connect them through doorways.",
            "Preserve object-localized geometry for navigation and manipulation checks.",
        ],
        "calibration_assets": ["synthetic_metric_geometry"],
        "scaffolding_validation": {
            "scale_anchor_count": 1,
            "checkpoint_count": 1,
            "validated_scale_m": 1.0,
            "validated_pose_coverage": 1.0,
            "hidden_zone_bound": 0.15,
            "validated_metric_bundle": True,
        },
        "requested_lanes": ["qualification", "scene_memory", "advanced_geometry"],
        "swap_focus": [environment_hint] if environment_hint != "default" else [],
        "manipulation_candidates": list(synthetic_targets.get("manipulation_candidates", [])),
        "articulation_hints": list(synthetic_targets.get("articulation_hints", [])),
        "metadata": metadata,
    }
    descriptor = CaptureDescriptor.from_dict(descriptor_payload)
    descriptor_path = capture_root / "capture_descriptor.json"
    write_json(descriptor_path, descriptor.to_dict())

    task_targets_payload = _build_task_targets_payload(
        synthetic_targets,
        scene_id=scene_id,
        capture_id=capture_id,
        object_index_entries=object_index_entries,
    )
    task_zone_center = _task_zone_center(
        object_index_entries=object_index_entries,
        target_ids=task_targets_payload.get("target_object_ids", []),
        articulation_ids=task_targets_payload.get("articulation_required_ids", []),
    )
    selected_metrics = _selected_object_metrics(
        object_index_entries=object_index_entries,
        target_ids=task_targets_payload.get("target_object_ids", []),
        articulation_ids=task_targets_payload.get("articulation_required_ids", []),
        task_zone_center=task_zone_center,
    )
    if task_zone_center is not None:
        metadata["task_zone"] = {"label": facility_name, "center": task_zone_center}
        descriptor_payload["metadata"] = metadata
        descriptor = CaptureDescriptor.from_dict(descriptor_payload)
        write_json(descriptor_path, descriptor.to_dict())
    task_hypothesis_report = _build_task_hypothesis_report(
        descriptor=descriptor,
        raw_task_hypothesis=raw_task_hypothesis,
        object_index_entries=object_index_entries,
        task_targets_payload=task_targets_payload,
    )
    normalized_task_hypothesis = (
        dict(task_hypothesis_report.get("normalized_task_hypothesis"))
        if isinstance(task_hypothesis_report.get("normalized_task_hypothesis"), Mapping)
        else {}
    )
    effective_metadata = _effective_task_metadata(
        descriptor,
        task_hypothesis_report=task_hypothesis_report,
    )
    if task_zone_center is not None:
        effective_metadata["task_zone"] = {
            "label": facility_name,
            "center": task_zone_center,
        }
    write_json(pipeline_dir / "task_hypothesis_report.json", task_hypothesis_report)
    write_json(pipeline_dir / "normalized_task_hypothesis.json", normalized_task_hypothesis)

    descriptor_uri = f"gs://{bucket}/{capture_prefix}/capture_descriptor.json"
    object_index_uri = f"{raw_prefix_uri}/object_index.json"
    runtime_preflight_report = _build_runtime_preflight_report(
        descriptor_path=descriptor_path,
        qa_report_path=qa_report_path,
        manifest_path=capture_root / "raw" / "manifest.json",
        object_index_path=capture_root / "raw" / "object_index.json",
        gcs_root=output_root,
    )
    write_json(pipeline_dir / "runtime_preflight_report.json", runtime_preflight_report)

    site_intake = {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "descriptor": descriptor.to_dict(),
        "descriptor_uri": descriptor_uri,
        "qa_report_uri": descriptor.qa_report_uri,
        "task_hypothesis_report_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, capture_id)}/task_hypothesis_report.json",
        "normalized_task_hypothesis_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, capture_id)}/normalized_task_hypothesis.json",
        "site_identity": {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "environment_type_hint": descriptor.environment_type_hint or "unknown",
            "capture_modality": descriptor.capture_modality,
        },
        "task_context": {
            "buyer_type": effective_metadata.get("buyer_type"),
            "task_statement": effective_metadata.get("task_statement"),
            "workflow_context": effective_metadata.get("workflow_context"),
            "operating_hours": effective_metadata.get("operating_hours"),
            "workflow_decomposition": effective_metadata.get("workflow_decomposition", []),
            "task_zone": effective_metadata.get("task_zone", {}),
            "success_criteria": effective_metadata.get("success_criteria", []),
            "owner": effective_metadata.get("owner"),
            "adjacent_systems": effective_metadata.get("adjacent_systems", []),
            "non_routine_modes": effective_metadata.get("non_routine_modes", []),
            "people_traffic_notes": effective_metadata.get("people_traffic_notes", []),
            "task_hypothesis_status": effective_metadata.get("task_hypothesis_status"),
            "task_hypothesis_confidence": effective_metadata.get("task_hypothesis_confidence"),
            "task_hypothesis_source": effective_metadata.get("task_hypothesis_source"),
            "task_hypothesis_warnings": effective_metadata.get("task_hypothesis_warnings", []),
        },
        "constraints": {
            "privacy_restrictions": effective_metadata.get("privacy_restrictions", []),
            "security_restrictions": effective_metadata.get("security_restrictions", []),
            "known_blockers": effective_metadata.get("known_blockers", []),
            "safety_concerns": effective_metadata.get("safety_concerns", []),
            "capture_restrictions": effective_metadata.get("capture_restrictions", []),
        },
        "capture_plan": {
            "scaffolding_used": list(descriptor.scaffolding_used),
            "coverage_plan": list(descriptor.coverage_plan),
            "calibration_assets": list(descriptor.calibration_assets),
            "uncertainty_priors": dict(descriptor.uncertainty_priors),
        },
        "task_hypothesis": dict(raw_task_hypothesis),
    }
    write_json(pipeline_dir / "site_intake.json", site_intake)

    capture_package_manifest = _build_capture_package_manifest(
        descriptor=descriptor,
        descriptor_uri=descriptor_uri,
        qa_report_uri=str(descriptor.qa_report_uri),
        manifest_uri=f"{raw_prefix_uri}/manifest.json",
        object_index_uri=object_index_uri,
        task_hypothesis_uri=descriptor.task_hypothesis_uri,
        storage_root=output_root,
        object_index_entries=object_index_entries,
    )
    write_json(pipeline_dir / "capture_package_manifest.json", capture_package_manifest)

    task_targets_with_index = dict(task_targets_payload)
    task_targets_with_index["object_index_entries"] = [dict(item) for item in object_index_entries]
    write_task_targets(pipeline_dir / "task_targets.json", task_targets_with_index)

    scorecard = _build_completeness_scorecard(
        descriptor=descriptor,
        qa_report=qa_report,
        manifest={"synthetic": True},
        object_index_uri=object_index_uri,
        object_index_entries=object_index_entries,
        metadata_override=effective_metadata,
        task_hypothesis_report=task_hypothesis_report,
    )
    write_json(pipeline_dir / "capture_qa_scorecard.json", scorecard)

    scope_record = _build_task_scope_record(
        descriptor=descriptor,
        task_targets_payload=task_targets_payload,
        completeness_status=str(scorecard.get("completeness_status") or "need_more_evidence"),
        metadata_override=effective_metadata,
    )
    write_json(pipeline_dir / "task_scope_record.json", scope_record)

    qualification_record = _build_qualification_record(
        descriptor=descriptor,
        scorecard=scorecard,
        scope_record=scope_record,
        object_index_entries=qualification_entries,
    )
    qualification_brief = _build_qualification_brief(
        descriptor=descriptor,
        scorecard=scorecard,
        scope_record=scope_record,
        qualification_record=qualification_record,
    )
    scene_graph = _build_scene_graph(
        descriptor=descriptor,
        scope_record=scope_record,
        object_index_entries=qualification_entries,
    )
    route_graph = _build_route_graph(
        descriptor=descriptor,
        scene_graph=scene_graph,
    )
    geometry_evidence = _build_geometry_evidence(
        descriptor=descriptor,
        qa_report=qa_report,
        object_index_entries=qualification_entries,
    )
    geometry_evidence["object_count"] = len(object_index_entries)
    if _is_residential_environment(environment_hint):
        geometry_evidence["measured_route_width_m"] = _residential_route_width_hint(
            structure,
            geometry_evidence.get("measured_route_width_m"),
        )
        geometry_evidence["adapter_route_width_source"] = "room_profile_lower_quartile"
        geometry_evidence["adapter_room_short_side_spans_m"] = _room_short_side_spans(structure)
        if selected_metrics.get("target_reach_distance_m") is not None:
            geometry_evidence["target_reach_distance_m"] = selected_metrics["target_reach_distance_m"]
            geometry_evidence["adapter_target_reach_source"] = "selected_target_centroid"
        if selected_metrics.get("workcell_span_m") is not None:
            geometry_evidence["workcell_span_m"] = selected_metrics["workcell_span_m"]
            geometry_evidence["adapter_workcell_span_source"] = "selected_target_and_articulation_span"
    capability_checks = _build_capability_checks(
        descriptor=descriptor,
        geometry_evidence=geometry_evidence,
        route_graph=route_graph,
        scope_record=scope_record,
    )
    blocker_register = _build_blocker_register(
        descriptor=descriptor,
        qualification_record=qualification_record,
        capability_checks=capability_checks,
        geometry_evidence=geometry_evidence,
    )
    readiness_decision = _build_readiness_decision(
        descriptor=descriptor,
        qualification_record=qualification_record,
        blocker_register=blocker_register,
        capability_checks=capability_checks,
        geometry_evidence=geometry_evidence,
    )
    human_actions_required = _build_human_actions_required(
        descriptor=descriptor,
        scorecard=scorecard,
        qualification_record=qualification_record,
        readiness_decision=readiness_decision,
        blocker_register=blocker_register,
        geometry_evidence=geometry_evidence,
    )
    qualification_record["readiness_state"] = readiness_decision.get("status")
    _write_advanced_geometry_bundle(
        source_dir=source_dir,
        pipeline_dir=pipeline_dir,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    opportunity_handoff = _build_opportunity_handoff(
        descriptor=descriptor,
        scorecard=scorecard,
        scope_record=scope_record,
        qualification_record=qualification_record,
        brief=qualification_brief,
        config=SimpleNamespace(robot_type=None),
        pipeline_dir=pipeline_dir,
        metadata_override=effective_metadata,
    )
    pipeline_prefix = to_pipeline_prefix(scene_id, capture_id)
    opportunity_handoff["evidence_bundle"] = {
        "scene_graph_uri": f"gs://{bucket}/{pipeline_prefix}/scene_graph.json",
        "route_graph_uri": f"gs://{bucket}/{pipeline_prefix}/route_graph.json",
        "geometry_evidence_uri": f"gs://{bucket}/{pipeline_prefix}/geometry_evidence.json",
        "capability_checks_uri": f"gs://{bucket}/{pipeline_prefix}/capability_checks.json",
        "blocker_register_uri": f"gs://{bucket}/{pipeline_prefix}/blocker_register.json",
        "readiness_decision_uri": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
        "task_hypothesis_report_uri": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
        "normalized_task_hypothesis_uri": f"gs://{bucket}/{pipeline_prefix}/normalized_task_hypothesis.json",
    }
    opportunity_handoff["readiness_state"] = readiness_decision.get("status")
    opportunity_handoff["qualification_state"] = readiness_decision.get("status")
    opportunity_handoff["downstream_evaluation_eligibility"] = readiness_decision.get("status") == "ready"
    opportunity_handoff["match_ready"] = bool(
        readiness_decision.get("status") == "ready"
        and opportunity_handoff.get("downstream_evaluation_eligibility") is True
    )

    write_json(pipeline_dir / "qualification_record.json", qualification_record)
    write_json(pipeline_dir / "qualification_brief.json", qualification_brief)
    write_json(pipeline_dir / "scene_graph.json", scene_graph)
    write_json(pipeline_dir / "route_graph.json", route_graph)
    write_json(pipeline_dir / "geometry_evidence.json", geometry_evidence)
    write_json(pipeline_dir / "capability_checks.json", capability_checks)
    write_json(pipeline_dir / "blocker_register.json", blocker_register)
    write_json(pipeline_dir / "readiness_decision.json", readiness_decision)
    write_json(pipeline_dir / "human_actions_required.json", human_actions_required)
    write_text(
        pipeline_dir / "readiness_report.md",
        _render_readiness_report(
            descriptor=descriptor,
            readiness_decision=readiness_decision,
            blocker_register=blocker_register,
            human_actions_required=human_actions_required,
            task_hypothesis_report=task_hypothesis_report,
        ),
    )
    write_json(pipeline_dir / "opportunity_handoff.json", opportunity_handoff)

    pipeline_summary = _build_pipeline_summary(
        bucket=bucket,
        descriptor_uri=descriptor_uri,
        qa_report_uri=str(descriptor.qa_report_uri),
        object_index_uri=object_index_uri,
        pipeline_prefix=pipeline_prefix,
        pipeline_dir=pipeline_dir,
        task_targets_payload=task_targets_with_index,
        scorecard=scorecard,
        qualification_record=qualification_record,
    )
    write_json(pipeline_dir / "pipeline_summary.json", pipeline_summary)
    scene_memory_artifacts = _write_scene_memory_bundle(
        storage_root=output_root,
        bucket=bucket,
        pipeline_prefix=pipeline_prefix,
        pipeline_dir=pipeline_dir,
        descriptor=descriptor,
        scorecard=scorecard,
        qualification_record=qualification_record,
    )
    opportunity_handoff = attach_handoff_package_paths(
        opportunity_handoff,
        pipeline_dir=pipeline_dir,
        metadata=effective_metadata,
    )
    write_json(pipeline_dir / "opportunity_handoff.json", opportunity_handoff)
    quality_report = {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": "passed",
        "generated_at": utc_now_iso(),
        "readiness_state": readiness_decision.get("status"),
        "completeness_status": scorecard.get("completeness_status"),
        "artifacts": {
            "descriptor_uri": descriptor_uri,
            "qa_report_uri": str(descriptor.qa_report_uri),
            "task_targets": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
            "runtime_preflight_report": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
            "site_intake": f"gs://{bucket}/{pipeline_prefix}/site_intake.json",
            "capture_package_manifest": f"gs://{bucket}/{pipeline_prefix}/capture_package_manifest.json",
            "capture_qa_scorecard": f"gs://{bucket}/{pipeline_prefix}/capture_qa_scorecard.json",
            "task_hypothesis_report": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
            "normalized_task_hypothesis": f"gs://{bucket}/{pipeline_prefix}/normalized_task_hypothesis.json",
            "task_scope_record": f"gs://{bucket}/{pipeline_prefix}/task_scope_record.json",
            "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "qualification_brief": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
            "scene_graph": f"gs://{bucket}/{pipeline_prefix}/scene_graph.json",
            "route_graph": f"gs://{bucket}/{pipeline_prefix}/route_graph.json",
            "geometry_evidence": f"gs://{bucket}/{pipeline_prefix}/geometry_evidence.json",
            "capability_checks": f"gs://{bucket}/{pipeline_prefix}/capability_checks.json",
            "blocker_register": f"gs://{bucket}/{pipeline_prefix}/blocker_register.json",
            "readiness_decision": f"gs://{bucket}/{pipeline_prefix}/readiness_decision.json",
            "human_actions_required": f"gs://{bucket}/{pipeline_prefix}/human_actions_required.json",
            "readiness_report": f"gs://{bucket}/{pipeline_prefix}/readiness_report.md",
            "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
            "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
            "scene_memory_manifest": scene_memory_artifacts["scene_memory_manifest_uri"],
            "scene_memory_readiness": scene_memory_artifacts["scene_memory_readiness_uri"],
            "conditioning_bundle": scene_memory_artifacts["conditioning_bundle_uri"],
            "preview_simulation_manifest": scene_memory_artifacts["preview_simulation_manifest_uri"],
            "gen3c_adapter_manifest": scene_memory_artifacts["gen3c_adapter_manifest_uri"],
            "neoverse_adapter_manifest": scene_memory_artifacts["neoverse_adapter_manifest_uri"],
            "cosmos_transfer_adapter_manifest": scene_memory_artifacts["cosmos_transfer_adapter_manifest_uri"],
        },
    }
    write_json(pipeline_dir / "qualification_quality_report.json", quality_report)
    write_json(pipeline_dir / "swap_quality_report.json", quality_report)
    completion_payload = {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": "completed",
        "completed_at": utc_now_iso(),
        "quality_report": f"gs://{bucket}/{pipeline_prefix}/qualification_quality_report.json",
        "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
        "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
        "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
        "scene_memory_manifest": scene_memory_artifacts["scene_memory_manifest_uri"],
        "preview_simulation_manifest": scene_memory_artifacts["preview_simulation_manifest_uri"],
    }
    write_json(pipeline_dir / ".qualification_pipeline_complete", completion_payload)
    write_json(pipeline_dir / ".swap_pipeline_complete", completion_payload)
    write_json(
        pipeline_dir / "interiorgs_adapter_summary.json",
        {
            "schema_version": "v1",
            "source_dir": str(source_dir),
            "capture_root": str(capture_root),
            "scene_id": scene_id,
            "capture_id": capture_id,
            "object_count": len(object_index_entries),
            "target_object_count": len(task_targets_payload.get("target_object_ids", [])),
            "articulation_required_count": len(task_targets_payload.get("articulation_required_ids", [])),
        },
    )

    final_bundle_path: Optional[str] = None
    final_memo_path: Optional[str] = None
    evaluation_prep_manifest_path: Optional[str] = None
    simready_manifest_path: Optional[str] = None
    simready_scene_path: Optional[str] = None
    if run_phase2:
        review = run_agent_review(
            capture_root=capture_root,
            provider_name=provider,
            openai_phase2_config=openai_phase2_config,
        )
        final_bundle_path = str(review.get("final_bundle_path") or "")
        final_memo_path = str(review.get("final_memo_path") or "")
    if run_evaluation_prep:
        evaluation_prep_result = run_evaluation_prep_stage(
            capture_root=capture_root,
            provider_name=evaluation_prep_provider,
        )
        evaluation_prep_manifest_path = str(evaluation_prep_result.get("manifest_path") or "")
    if run_simready:
        simready_result = run_simready_stage(
            capture_root=capture_root,
            provider_name=simready_provider,
        )
        simready_manifest_path = str(simready_result.get("manifest_path") or "")
        simready_scene_path = str(simready_result.get("scene_path") or "")

    return InteriorGSAdaptationResult(
        source_dir=source_dir,
        capture_root=capture_root,
        scene_id=scene_id,
        capture_id=capture_id,
        provider=provider if run_phase2 else None,
        final_bundle_path=final_bundle_path,
        final_memo_path=final_memo_path,
        evaluation_prep_manifest_path=evaluation_prep_manifest_path,
        simready_manifest_path=simready_manifest_path,
        simready_scene_path=simready_scene_path,
    )


def adapt_many(
    *,
    source_dirs: Iterable[Path],
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    run_phase2: bool = False,
    run_evaluation_prep: bool = False,
    run_simready: bool = False,
    provider: str = "openai",
    evaluation_prep_provider: str = "manual",
    simready_provider: str = "manual",
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
) -> List[InteriorGSAdaptationResult]:
    results: List[InteriorGSAdaptationResult] = []
    for source_dir in source_dirs:
        results.append(
            adapt_interiorgs_scene(
                source_dir=source_dir,
                output_root=output_root,
                bucket=bucket,
                run_phase2=run_phase2,
                run_evaluation_prep=run_evaluation_prep,
                run_simready=run_simready,
                provider=provider,
                evaluation_prep_provider=evaluation_prep_provider,
                simready_provider=simready_provider,
                openai_phase2_config=openai_phase2_config,
            )
        )
    return results


def adapt_interiorgs_task_runs(
    *,
    source_dir: Path,
    output_root: Path,
    bucket: str = _DEFAULT_BUCKET,
    run_phase2: bool = False,
    run_evaluation_prep: bool = False,
    run_simready: bool = False,
    provider: str = "openai",
    evaluation_prep_provider: str = "manual",
    simready_provider: str = "manual",
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
) -> List[InteriorGSTaskRunResult]:
    source_dir = source_dir.resolve()
    labels = _read_json_any(source_dir / "labels.json")
    structure = _read_json_any(source_dir / "structure.json")
    synthetic_targets = _read_json_any(source_dir / "task_targets.synthetic.json")
    if not isinstance(labels, list) or not isinstance(structure, Mapping) or not isinstance(synthetic_targets, Mapping):
        raise ValueError(f"InteriorGS source directory is incomplete: {source_dir}")

    scene_id, base_capture_id = _safe_name_parts(source_dir)
    tasks = synthetic_targets.get("tasks", []) if isinstance(synthetic_targets.get("tasks"), list) else []
    facility_name = str(synthetic_targets.get("facility_name") or source_dir.name)
    manipulation = synthetic_targets.get("manipulation_candidates", []) if isinstance(synthetic_targets.get("manipulation_candidates"), list) else []
    articulation = synthetic_targets.get("articulation_hints", []) if isinstance(synthetic_targets.get("articulation_hints"), list) else []
    navigation = synthetic_targets.get("navigation_hints", []) if isinstance(synthetic_targets.get("navigation_hints"), list) else []
    staging_root = output_root / "_interiorgs_task_inputs" / source_dir.name
    results: List[InteriorGSTaskRunResult] = []

    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        task_text = str(task.get("task_id") or "").strip()
        category = _task_category(task_text)
        if category == "other":
            continue

        reduced_payload = {
            "bootstrap_generated": True,
            "generated_at": utc_now_iso(),
            "facility_name": f"{facility_name} :: {task_text}",
            "source": synthetic_targets.get("source"),
            "scene_type": synthetic_targets.get("scene_type"),
            "tasks": [dict(task)],
            "manipulation_candidates": [],
            "articulation_hints": [],
            "navigation_hints": [],
            "interiorgs_source_dir": str(source_dir),
        }

        target_ids, articulation_ids = _explicit_task_targets(
            {
                "tasks": [task],
                "manipulation_candidates": manipulation,
                "articulation_hints": articulation,
                "navigation_hints": navigation,
            }
        )
        selected_targets = _candidate_ids(manipulation, target_ids)
        selected_navigation = _candidate_ids(navigation, target_ids)
        reference_centers = [
            center
            for center in (
                [_payload_center(item) for item in selected_targets + selected_navigation]
            )
            if center is not None
        ]
        if not reference_centers:
            selected_articulation = _candidate_ids(articulation, articulation_ids)
            reference_centers = [
                center for center in [_payload_center(item) for item in selected_articulation] if center is not None
            ]
        nearby_articulation_ids = _nearby_ids(
            articulation,
            reference_centers=reference_centers,
            radius_m=2.5,
            limit=8,
            exclude_ids=articulation_ids,
        )
        nearby_navigation_ids = _nearby_ids(
            navigation,
            reference_centers=reference_centers,
            radius_m=2.0,
            limit=4,
            exclude_ids=target_ids,
        )
        selected_articulation = _candidate_ids(articulation, list(articulation_ids) + nearby_articulation_ids)
        selected_navigation = _candidate_ids(navigation, list(target_ids) + nearby_navigation_ids)
        reduced_payload["manipulation_candidates"] = [dict(item) for item in selected_targets]
        reduced_payload["articulation_hints"] = [dict(item) for item in selected_articulation]
        reduced_payload["navigation_hints"] = [dict(item) for item in selected_navigation]

        keep_ids = [
            *(str(item.get("instance_id") or item.get("id") or "").strip() for item in selected_targets),
            *(str(item.get("instance_id") or item.get("id") or "").strip() for item in selected_articulation),
            *(str(item.get("instance_id") or item.get("id") or "").strip() for item in selected_navigation),
        ]
        reduced_labels = _filter_labels_for_ids(labels, keep_ids)
        task_slug = _slugify_task(task_text)
        staged_dir = staging_root / f"{scene_id}_{base_capture_id}-{category}-{index:03d}-{task_slug}"
        ensure_dir(staged_dir)
        _write_json_any(staged_dir / "labels.json", reduced_labels)
        _copy_or_link(source_dir / "structure.json", staged_dir / "structure.json")
        _write_json_any(staged_dir / "task_targets.synthetic.json", reduced_payload)
        if (source_dir / "3dgs_compressed.ply").is_file():
            _copy_or_link(source_dir / "3dgs_compressed.ply", staged_dir / "3dgs_compressed.ply")

        adapted = adapt_interiorgs_scene(
            source_dir=staged_dir,
            output_root=output_root,
            bucket=bucket,
            run_phase2=run_phase2,
            run_evaluation_prep=run_evaluation_prep,
            run_simready=run_simready,
            provider=provider,
            evaluation_prep_provider=evaluation_prep_provider,
            simready_provider=simready_provider,
            openai_phase2_config=openai_phase2_config,
        )
        results.append(
            InteriorGSTaskRunResult(
                category=category,
                task_text=task_text,
                capture_root=adapted.capture_root,
                capture_id=adapted.capture_id,
                final_bundle_path=adapted.final_bundle_path,
                final_memo_path=adapted.final_memo_path,
                evaluation_prep_manifest_path=adapted.evaluation_prep_manifest_path,
                simready_manifest_path=adapted.simready_manifest_path,
                simready_scene_path=adapted.simready_scene_path,
            )
        )

    scene_capture_root = output_root / bucket / "scenes" / scene_id / "captures" / base_capture_id
    pipeline_dir = scene_capture_root / "pipeline"
    ensure_dir(pipeline_dir)
    grouped: Dict[str, List[Dict[str, Any]]] = {"pick": [], "open_close": [], "navigate": []}
    for item in results:
        grouped.setdefault(item.category, []).append(
            {
                "task_text": item.task_text,
                "capture_root": str(item.capture_root),
                "capture_id": item.capture_id,
                "final_bundle_path": item.final_bundle_path,
                "final_memo_path": item.final_memo_path,
                "evaluation_prep_manifest_path": item.evaluation_prep_manifest_path,
                "simready_manifest_path": item.simready_manifest_path,
                "simready_scene_path": item.simready_scene_path,
            }
        )
    write_json(
        pipeline_dir / "task_run_manifest.json",
        {
            "schema_version": "v1",
            "scene_id": scene_id,
            "base_capture_id": base_capture_id,
            "generated_at": utc_now_iso(),
            "source_dir": str(source_dir),
            "groups": grouped,
        },
    )
    _write_task_run_comparison_report(scene_capture_root=scene_capture_root)
    write_scene_dashboard_summary(scene_capture_root=scene_capture_root, bucket=bucket)
    write_scene_deployment_summary(scene_capture_root=scene_capture_root, bucket=bucket)
    opportunity_handoff = _read_json_any(pipeline_dir / "opportunity_handoff.json")
    completeness_status = _read_json_any(pipeline_dir / "capture_qa_scorecard.json").get("completeness_status")
    qualification_state = derive_webapp_qualification_state(
        readiness_state=opportunity_handoff.get("qualification_state"),
        completeness_status=completeness_status,
    )
    opportunity_state = derive_webapp_opportunity_state(qualification_state=qualification_state)
    pipeline_dir = scene_capture_root / "pipeline"
    scene_memory_manifest_uri = f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_memory/scene_memory_manifest.json"
    scene_memory_readiness_uri = f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_memory/scene_memory_readiness.json"
    conditioning_bundle_uri = f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_memory/conditioning_bundle.json"
    preview_simulation_manifest_uri = f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/preview_simulation/preview_simulation_manifest.json"
    scene_memory_artifacts = {
        "scene_memory_manifest_uri": scene_memory_manifest_uri,
        "scene_memory_readiness_uri": scene_memory_readiness_uri,
        "conditioning_bundle_uri": conditioning_bundle_uri,
        "preview_simulation_manifest_uri": preview_simulation_manifest_uri,
        "scene_memory_status": (
            str((_read_json_any(pipeline_dir / "scene_memory" / "scene_memory_readiness.json") or {}).get("status") or "needs_more_evidence")
        ),
        "preview_simulation_status": (
            str((_read_json_any(pipeline_dir / "preview_simulation" / "preview_simulation_manifest.json") or {}).get("status") or "review_required")
        ),
    }
    sync_webapp_pipeline_attachment(
        site_submission_id=opportunity_handoff.get("site_submission_id"),
        request_id=opportunity_handoff.get("site_submission_id"),
        scene_id=scene_id,
        capture_id=base_capture_id,
        pipeline_prefix=to_pipeline_prefix(scene_id, base_capture_id),
        qualification_state=qualification_state,
        opportunity_state=opportunity_state,
        artifacts={
            "readiness_decision_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/readiness_decision.json",
            "readiness_report_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/readiness_report.md",
            "qualification_quality_report_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/qualification_quality_report.json",
            "opportunity_handoff_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/opportunity_handoff.json",
            "human_actions_required_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/human_actions_required.json",
            "agent_review_bundle_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/agent_review_bundle.json",
            "agent_readiness_memo_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/agent_readiness_memo.md",
            "dashboard_summary_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/dashboard_summary.json",
            "scene_deployment_summary_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_deployment_summary.md",
            "scene_memory_manifest_uri": scene_memory_manifest_uri,
            "scene_memory_readiness_uri": scene_memory_readiness_uri,
            "conditioning_bundle_uri": conditioning_bundle_uri,
            "preview_simulation_manifest_uri": preview_simulation_manifest_uri,
            "gen3c_adapter_manifest_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_memory/adapter_manifests/gen3c.json",
            "neoverse_adapter_manifest_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_memory/adapter_manifests/neoverse.json",
            "cosmos_transfer_adapter_manifest_uri": f"gs://{bucket}/{to_pipeline_prefix(scene_id, base_capture_id)}/scene_memory/adapter_manifests/cosmos_transfer.json",
        },
        derived_assets=_scene_memory_derived_assets(scene_memory_artifacts),
    )
    return results


def _write_task_run_comparison_report(*, scene_capture_root: Path) -> None:
    pipeline_dir = scene_capture_root / "pipeline"
    manifest_path = pipeline_dir / "task_run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json_any(manifest_path)
    whole_ready = _read_json_any(pipeline_dir / "readiness_decision.json")
    whole_memo = pipeline_dir / "agent_readiness_memo.md"
    lines = [
        "# Task Run Comparison Report",
        "",
        f"- Scene: `{manifest.get('scene_id')}`",
        f"- Whole-home capture: `{manifest.get('base_capture_id')}`",
        f"- Whole-home readiness: `{whole_ready.get('status')}`",
        f"- Whole-home memo: `{whole_memo}`",
        "",
    ]
    grouped = manifest.get("groups", {}) if isinstance(manifest, Mapping) else {}
    for category in ("pick", "open_close", "navigate"):
        lines.append(f"## {category}")
        entries = grouped.get(category, []) if isinstance(grouped.get(category), list) else []
        if not entries:
            lines.append("- none")
            lines.append("")
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            capture_root = Path(str(entry.get("capture_root") or ""))
            readiness_path = capture_root / "pipeline" / "readiness_decision.json"
            readiness = _read_json_any(readiness_path) if readiness_path.is_file() else {}
            lines.append(f"- Task: `{entry.get('task_text')}`")
            lines.append(f"  Capture: `{entry.get('capture_id')}`")
            lines.append(f"  Status: `{readiness.get('status', 'unknown')}`")
            lines.append(f"  Memo: `{entry.get('final_memo_path')}`")
        lines.append("")
    write_text(pipeline_dir / "task_run_comparison_report.md", "\n".join(lines).rstrip() + "\n")


def _openai_phase2_config_from_args(args: argparse.Namespace) -> Optional[OpenAIPhase2Config]:
    mode = str(getattr(args, "openai_phase2_mode", "") or "").strip()
    model = str(getattr(args, "openai_phase2_model", "") or "").strip()
    codex_bin = str(getattr(args, "openai_phase2_codex_bin", "") or "").strip()
    timeout_seconds = getattr(args, "openai_phase2_timeout_seconds", None)
    reasoning_effort = str(getattr(args, "openai_phase2_reasoning_effort", "") or "").strip()
    if not any([mode, model, codex_bin, timeout_seconds, reasoning_effort]):
        return None
    env_default = OpenAIPhase2Config.from_env()
    return OpenAIPhase2Config(
        mode=mode or env_default.mode,
        model=model or env_default.model,
        codex_bin=codex_bin or env_default.codex_bin,
        timeout_seconds=int(timeout_seconds or env_default.timeout_seconds),
        reasoning_effort=reasoning_effort or env_default.reasoning_effort,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Adapt InteriorGS scenes into Phase 2 reviewable captures")
    parser.add_argument("source_dirs", nargs="+", help="InteriorGS scene directories")
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bucket", default=_DEFAULT_BUCKET)
    parser.add_argument("--run-phase2", action="store_true")
    parser.add_argument("--run-evaluation-prep", action="store_true")
    parser.add_argument("--run-simready", action="store_true")
    parser.add_argument("--task-runs", action="store_true")
    parser.add_argument("--provider", default="openai", choices=("openai", "claude"))
    parser.add_argument("--evaluation-prep-provider", default="manual")
    parser.add_argument("--simready-provider", default="manual")
    parser.add_argument("--openai-phase2-mode", choices=("disabled", "codex_cli"))
    parser.add_argument("--openai-phase2-model")
    parser.add_argument("--openai-phase2-codex-bin")
    parser.add_argument("--openai-phase2-timeout-seconds", type=int)
    parser.add_argument("--openai-phase2-reasoning-effort")
    args = parser.parse_args(argv)

    results = adapt_many(
        source_dirs=[Path(item) for item in args.source_dirs],
        output_root=Path(args.output_root).resolve(),
        bucket=args.bucket,
        run_phase2=bool(args.run_phase2),
        run_evaluation_prep=bool(args.run_evaluation_prep),
        run_simready=bool(args.run_simready),
        provider=args.provider,
        evaluation_prep_provider=args.evaluation_prep_provider,
        simready_provider=args.simready_provider,
        openai_phase2_config=_openai_phase2_config_from_args(args),
    )
    for result in results:
        print(f"[interiorgs-adapter] source={result.source_dir}")
        print(f"[interiorgs-adapter] capture_root={result.capture_root}")
        if result.final_bundle_path:
            print(f"[interiorgs-adapter] final_bundle={result.final_bundle_path}")
        if result.final_memo_path:
            print(f"[interiorgs-adapter] final_memo={result.final_memo_path}")
        if result.evaluation_prep_manifest_path:
            print(f"[interiorgs-adapter] evaluation_prep={result.evaluation_prep_manifest_path}")
        if result.simready_scene_path:
            print(f"[interiorgs-adapter] simready_scene={result.simready_scene_path}")
        if args.task_runs:
            task_results = adapt_interiorgs_task_runs(
                source_dir=result.source_dir,
                output_root=Path(args.output_root).resolve(),
                bucket=args.bucket,
                run_phase2=bool(args.run_phase2),
                run_evaluation_prep=bool(args.run_evaluation_prep),
                run_simready=bool(args.run_simready),
                provider=args.provider,
                evaluation_prep_provider=args.evaluation_prep_provider,
                simready_provider=args.simready_provider,
                openai_phase2_config=_openai_phase2_config_from_args(args),
            )
            print(f"[interiorgs-adapter] task_runs={len(task_results)}")
            print(
                f"[interiorgs-adapter] task_manifest="
                f"{Path(args.output_root).resolve() / args.bucket / 'scenes' / result.scene_id / 'captures' / result.capture_id / 'pipeline' / 'task_run_manifest.json'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
