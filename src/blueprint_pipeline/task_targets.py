"""Task-target inference and task-aware swap candidate selection."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .capture_bridge import CaptureDescriptor
from .common import (
    join_gs_uri,
    read_json_any,
    resolve_gs_uri_to_path,
    try_parse_float,
    utc_now_iso,
    write_json,
)
from .ios_manifest import IOSManifest
from .swap_candidates import build_swap_candidates_payload

_ARTICULATION_KEYWORDS = {
    "door",
    "drawer",
    "cabinet",
    "cupboard",
    "closet",
    "locker",
    "fridge",
    "refrigerator",
    "microwave",
    "oven",
    "washer",
    "dryer",
    "freezer",
}

_MANIPULATION_KEYWORDS = {
    "tote",
    "bin",
    "box",
    "crate",
    "carton",
    "package",
    "container",
    "pallet",
    "bottle",
    "can",
    "tool",
    "part",
    "object",
    "cup",
    "mug",
}

_STRUCTURAL_KEYWORDS = {
    "wall",
    "floor",
    "ceiling",
    "window",
    "stairs",
    "pillar",
    "column",
    "beam",
    "light_fixture",
    "outlet",
    "rack",
    "shelf",
    "conveyor",
    "safety_barrier",
}

_SELECTION_MODES = {"explicit_only", "hybrid", "policy_only"}
_SOURCE_PRIORITY = {
    "descriptor": 3,
    "video_inference": 2,
    "heuristic": 1,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _normalized_text(*parts: Any) -> str:
    return " ".join(str(part).strip().lower() for part in parts if str(part).strip())


def _candidate_id(entry: Mapping[str, Any]) -> str:
    for key in ("instance_id", "id", "object_id", "uuid"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _candidate_label(entry: Mapping[str, Any]) -> str:
    for key in ("label", "name", "class_name", "category"):
        value = entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "object"


def _mean_box_area_px(entry: Mapping[str, Any]) -> float:
    raw = entry.get("mean_box_px")
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    if isinstance(raw, Mapping):
        area = _safe_float(raw.get("area"), 0.0)
        if area > 0.0:
            return area
        width = _safe_float(raw.get("width"), 0.0)
        height = _safe_float(raw.get("height"), 0.0)
        return max(0.0, width * height)
    if isinstance(raw, list) and len(raw) >= 2:
        width = _safe_float(raw[0], 0.0)
        height = _safe_float(raw[1], 0.0)
        return max(0.0, width * height)
    return 0.0


def _object_salience_score(entry: Mapping[str, Any]) -> float:
    detections = max(
        _safe_int(entry.get("n_total_detections"), 0),
        _safe_int(entry.get("n_frame_detections"), 0),
    )
    confidence = _safe_float(entry.get("mean_confidence"), _safe_float(entry.get("confidence"), 0.0))
    area = _mean_box_area_px(entry)
    crops = len(entry.get("all_crops")) if isinstance(entry.get("all_crops"), list) else 0

    det_score = _clamp01(detections / 20.0)
    conf_score = _clamp01(confidence)
    area_score = _clamp01(area / 30000.0)
    crop_score = _clamp01(crops / 4.0)
    return (det_score + conf_score + area_score + crop_score) / 4.0


def _entry_text(entry: Mapping[str, Any]) -> str:
    return _normalized_text(
        _candidate_label(entry),
        entry.get("name"),
        entry.get("class_name"),
        entry.get("category"),
        entry.get("description"),
    )


def _normalize_hint_entry(
    entry: Mapping[str, Any],
    *,
    source: str,
    role: str,
    default_confidence: float,
) -> Dict[str, Any]:
    instance_id = _candidate_id(entry)
    label = _candidate_label(entry)
    confidence = _safe_float(entry.get("confidence"), default_confidence)
    reason = str(entry.get("reason") or entry.get("rationale") or "").strip()
    return {
        "instance_id": instance_id,
        "label": label,
        "confidence": _clamp01(confidence),
        "source": source,
        "role": role,
        "reason": reason,
    }


def _normalize_hint_list(
    entries: Iterable[Any],
    *,
    source: str,
    role: str,
    default_confidence: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for value in entries:
        if isinstance(value, Mapping):
            out.append(
                _normalize_hint_entry(
                    value,
                    source=source,
                    role=role,
                    default_confidence=default_confidence,
                )
            )
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            out.append(
                {
                    "instance_id": text,
                    "label": "object",
                    "confidence": default_confidence,
                    "source": source,
                    "role": role,
                    "reason": "",
                }
            )
    return out


def _dedupe_hint_entries(entries: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        instance_id = str(entry.get("instance_id") or "").strip()
        label = str(entry.get("label") or "object").strip()
        key = instance_id or f"label:{label.lower()}"
        if not key:
            continue

        existing = by_key.get(key)
        source = str(entry.get("source") or "heuristic").strip() or "heuristic"
        score = (
            _SOURCE_PRIORITY.get(source, 0),
            _safe_float(entry.get("confidence"), 0.0),
        )
        if existing is None:
            by_key[key] = dict(entry)
            continue

        existing_source = str(existing.get("source") or "heuristic").strip() or "heuristic"
        existing_score = (
            _SOURCE_PRIORITY.get(existing_source, 0),
            _safe_float(existing.get("confidence"), 0.0),
        )
        if score > existing_score:
            merged = dict(entry)
        else:
            merged = dict(existing)

        reasons: List[str] = []
        for reason in (existing.get("reason"), entry.get("reason")):
            text = str(reason or "").strip()
            if text and text not in reasons:
                reasons.append(text)
        merged["reason"] = "; ".join(reasons)
        by_key[key] = merged

    return sorted(
        (dict(value) for value in by_key.values()),
        key=lambda item: (
            -_safe_float(item.get("confidence"), 0.0),
            str(item.get("instance_id") or item.get("label") or ""),
        ),
    )


def _descriptor_target_entries(
    descriptor: CaptureDescriptor,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manip = _normalize_hint_list(
        descriptor.manipulation_candidates,
        source="descriptor",
        role="manipulation",
        default_confidence=1.0,
    )
    artic = _normalize_hint_list(
        descriptor.articulation_hints,
        source="descriptor",
        role="articulation",
        default_confidence=1.0,
    )
    return manip, artic


def _heuristic_task_inference(
    *,
    descriptor: CaptureDescriptor,
    object_index_entries: List[Mapping[str, Any]],
    max_targets: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    env_hint = str(descriptor.environment_type_hint or "").strip().lower()
    manip_keywords = set(_MANIPULATION_KEYWORDS)
    artic_keywords = set(_ARTICULATION_KEYWORDS)
    structural_keywords = set(_STRUCTURAL_KEYWORDS)

    if env_hint == "warehouse":
        manip_keywords.update({"parcel", "shipment", "tote", "bin", "carton"})
        artic_keywords.update({"rolling_door", "docking_door"})
    elif env_hint == "kitchen":
        manip_keywords.update({"plate", "bowl", "pot", "pan", "utensil"})
        artic_keywords.update({"pantry", "cabinet_door"})

    manip_entries: List[Dict[str, Any]] = []
    artic_entries: List[Dict[str, Any]] = []

    for entry in object_index_entries:
        if not isinstance(entry, Mapping):
            continue
        obj_id = _candidate_id(entry)
        label = _candidate_label(entry)
        text = _entry_text(entry)
        if not text:
            continue
        if any(token in text for token in structural_keywords):
            continue

        salience = _object_salience_score(entry)
        confidence = _safe_float(entry.get("mean_confidence"), _safe_float(entry.get("confidence"), 0.0))
        blended_conf = _clamp01((salience * 0.65) + (confidence * 0.35))

        has_artic = any(token in text for token in artic_keywords)
        has_manip = any(token in text for token in manip_keywords)
        if has_artic:
            artic_entries.append(
                {
                    "instance_id": obj_id,
                    "label": label,
                    "confidence": max(0.45, blended_conf),
                    "source": "heuristic",
                    "role": "articulation",
                    "reason": f"keyword+salience (score={blended_conf:.2f})",
                }
            )
            continue
        if has_manip:
            manip_entries.append(
                {
                    "instance_id": obj_id,
                    "label": label,
                    "confidence": max(0.35, blended_conf),
                    "source": "heuristic",
                    "role": "manipulation",
                    "reason": f"keyword+salience (score={blended_conf:.2f})",
                }
            )

    manip_entries = _dedupe_hint_entries(manip_entries)[:max_targets]
    artic_entries = _dedupe_hint_entries(artic_entries)[:max_targets]

    tasks: List[Dict[str, Any]] = []
    if artic_entries:
        tasks.append(
            {
                "task_id": "open_close_access_points",
                "confidence": round(
                    sum(_safe_float(item.get("confidence"), 0.0) for item in artic_entries)
                    / max(1, len(artic_entries)),
                    3,
                ),
                "target_object_ids": [
                    str(item.get("instance_id"))
                    for item in artic_entries
                    if str(item.get("instance_id") or "").strip()
                ],
                "rationale": "Detected articulated affordances (doors/drawers/cabinets).",
            }
        )

    if manip_entries:
        tasks.append(
            {
                "task_id": "pick_place_manipulation",
                "confidence": round(
                    sum(_safe_float(item.get("confidence"), 0.0) for item in manip_entries)
                    / max(1, len(manip_entries)),
                    3,
                ),
                "target_object_ids": [
                    str(item.get("instance_id"))
                    for item in manip_entries
                    if str(item.get("instance_id") or "").strip()
                ],
                "rationale": "Detected portable/manipulable object classes.",
            }
        )

    return manip_entries, artic_entries, tasks


def _resolve_video_uri_and_path(
    *,
    descriptor: CaptureDescriptor,
    manifest: IOSManifest,
    storage_root: Path,
) -> Tuple[str, Optional[Path]]:
    candidate_uris: List[str] = []
    for uri in (
        descriptor.raw_video_uri,
        manifest.video_uri,
        join_gs_uri(descriptor.raw_prefix_uri, "video.mp4"),
    ):
        text = str(uri or "").strip()
        if text and text not in candidate_uris:
            candidate_uris.append(text)

    for uri in candidate_uris:
        if uri.startswith("gs://"):
            path = resolve_gs_uri_to_path(uri, storage_root)
            if path.exists():
                return uri, path
        else:
            path = Path(uri)
            if path.exists():
                return uri, path
    return candidate_uris[0] if candidate_uris else "", None


def _run_external_video_task_inference(
    *,
    descriptor: CaptureDescriptor,
    object_index_uri: str,
    storage_root: Path,
    video_uri: str,
    video_path: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    command_template = (os.getenv("TASK_INFERENCE_COMMAND") or "").strip()
    if not command_template:
        return {}, {"status": "skipped", "reason": "TASK_INFERENCE_COMMAND not set"}
    if video_path is None:
        return {}, {"status": "skipped", "reason": "video file not found", "video_uri": video_uri}

    object_index_path = resolve_gs_uri_to_path(object_index_uri, storage_root)
    if not object_index_path.exists():
        return {}, {"status": "skipped", "reason": "object index path not found"}

    timeout_seconds = _safe_int(os.getenv("TASK_INFERENCE_TIMEOUT_SECONDS"), 240)
    if timeout_seconds <= 0:
        timeout_seconds = 240

    fd, tmp_name = tempfile.mkstemp(prefix="task_targets_", suffix=".json")
    os.close(fd)
    tmp_output = Path(tmp_name)
    try:
        substitutions = {
            "VIDEO_PATH": str(video_path),
            "VIDEO_URI": video_uri,
            "OBJECT_INDEX_PATH": str(object_index_path),
            "OUTPUT_JSON": str(tmp_output),
            "SCENE_ID": descriptor.scene_id,
            "CAPTURE_ID": descriptor.capture_id,
            "ENV_HINT": str(descriptor.environment_type_hint or ""),
        }
        rendered = command_template
        for key, value in substitutions.items():
            rendered = rendered.replace("{" + key + "}", value)

        try:
            command = shlex.split(rendered)
        except ValueError as exc:
            return {}, {"status": "failed", "reason": f"invalid command template: {exc}"}
        if not command:
            return {}, {"status": "failed", "reason": "empty command"}

        try:
            proc = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {}, {"status": "failed", "reason": f"command timeout after {timeout_seconds}s"}
        except Exception as exc:  # pragma: no cover - subprocess runtime edge
            return {}, {"status": "failed", "reason": f"command execution failed: {exc}"}

        report: Dict[str, Any] = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "return_code": proc.returncode,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timeout_seconds": timeout_seconds,
        }
        if proc.returncode != 0:
            return {}, report
        if not tmp_output.exists() or tmp_output.stat().st_size == 0:
            report["status"] = "failed"
            report["reason"] = "output JSON missing"
            return {}, report

        payload = read_json_any(tmp_output)
        if isinstance(payload, Mapping):
            return dict(payload), report
        report["status"] = "failed"
        report["reason"] = f"unexpected output type: {type(payload).__name__}"
        return {}, report
    finally:
        try:
            tmp_output.unlink(missing_ok=True)
        except Exception:
            pass


def infer_task_targets(
    *,
    descriptor: CaptureDescriptor,
    manifest: IOSManifest,
    object_index_entries: List[Mapping[str, Any]],
    object_index_uri: str,
    storage_root: Path,
    max_targets: int = 24,
) -> Dict[str, Any]:
    """Infer task-relevant targets from descriptor hints + video/object signals."""

    max_targets = max(1, int(max_targets or 1))
    desc_manip, desc_artic = _descriptor_target_entries(descriptor)

    video_uri, video_path = _resolve_video_uri_and_path(
        descriptor=descriptor,
        manifest=manifest,
        storage_root=storage_root,
    )

    external_payload, external_report = _run_external_video_task_inference(
        descriptor=descriptor,
        object_index_uri=object_index_uri,
        storage_root=storage_root,
        video_uri=video_uri,
        video_path=video_path,
    )
    ext_manip = _normalize_hint_list(
        external_payload.get("manipulation_candidates", []),
        source="video_inference",
        role="manipulation",
        default_confidence=0.7,
    )
    ext_artic = _normalize_hint_list(
        external_payload.get("articulation_hints", []),
        source="video_inference",
        role="articulation",
        default_confidence=0.75,
    )

    heur_manip, heur_artic, heur_tasks = _heuristic_task_inference(
        descriptor=descriptor,
        object_index_entries=object_index_entries,
        max_targets=max_targets,
    )

    manip_entries = _dedupe_hint_entries([*desc_manip, *ext_manip, *heur_manip])[:max_targets]
    artic_entries = _dedupe_hint_entries([*desc_artic, *ext_artic, *heur_artic])[:max_targets]

    target_ids = [
        str(item.get("instance_id"))
        for item in manip_entries
        if str(item.get("instance_id") or "").strip()
    ]
    articulation_ids = [
        str(item.get("instance_id"))
        for item in artic_entries
        if str(item.get("instance_id") or "").strip()
    ]

    tasks_payload = external_payload.get("tasks")
    tasks: List[Dict[str, Any]]
    if isinstance(tasks_payload, list):
        tasks = [dict(item) for item in tasks_payload if isinstance(item, Mapping)]
    else:
        tasks = heur_tasks

    inference_mode = "heuristic"
    if str(external_report.get("status") or "") == "ok":
        inference_mode = "external+heuristic"
    if not heur_manip and not heur_artic and desc_manip and not tasks:
        inference_mode = "descriptor_only"

    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "inference_mode": inference_mode,
        "video_analysis": {
            "video_uri": video_uri,
            "video_path": str(video_path) if video_path is not None else "",
            "external_inference": external_report,
        },
        "manipulation_candidates": manip_entries,
        "articulation_hints": artic_entries,
        "target_object_ids": sorted(set(target_ids)),
        "articulation_required_ids": sorted(set(articulation_ids)),
        "tasks": tasks,
    }


def _merge_descriptor_with_task_targets(
    descriptor: CaptureDescriptor,
    task_targets: Optional[Mapping[str, Any]],
) -> CaptureDescriptor:
    if not isinstance(task_targets, Mapping):
        return descriptor

    desc_manip = [dict(item) for item in descriptor.manipulation_candidates]
    desc_artic = [dict(item) for item in descriptor.articulation_hints]

    def _merge(
        base: List[Dict[str, Any]],
        incoming: Iterable[Any],
        *,
        role: str,
    ) -> List[Dict[str, Any]]:
        merged = _dedupe_hint_entries(
            [
                *_normalize_hint_list(base, source="descriptor", role=role, default_confidence=1.0),
                *_normalize_hint_list(incoming, source="video_inference", role=role, default_confidence=0.75),
            ]
        )
        out: List[Dict[str, Any]] = []
        for entry in merged:
            instance_id = str(entry.get("instance_id") or "").strip()
            label = str(entry.get("label") or "").strip()
            payload: Dict[str, Any] = {}
            if instance_id:
                payload["instance_id"] = instance_id
            if label:
                payload["label"] = label
            if payload:
                out.append(payload)
        return out

    manip = _merge(
        desc_manip,
        task_targets.get("manipulation_candidates", []),
        role="manipulation",
    )
    artic = _merge(
        desc_artic,
        task_targets.get("articulation_hints", []),
        role="articulation",
    )

    # Also ingest plain ID lists when present.
    for obj_id in task_targets.get("target_object_ids", []):
        text = str(obj_id).strip()
        if text and text not in {str(item.get("instance_id") or "") for item in manip}:
            manip.append({"instance_id": text, "label": "object"})
    for obj_id in task_targets.get("articulation_required_ids", []):
        text = str(obj_id).strip()
        if text and text not in {str(item.get("instance_id") or "") for item in artic}:
            artic.append({"instance_id": text, "label": "object"})

    return replace(
        descriptor,
        manipulation_candidates=manip,
        articulation_hints=artic,
    )


def _normalize_selection_mode(mode: str) -> str:
    candidate = str(mode or "").strip().lower()
    return candidate if candidate in _SELECTION_MODES else "hybrid"


def _extract_explicit_sets(
    descriptor: CaptureDescriptor,
    task_targets: Optional[Mapping[str, Any]],
) -> Dict[str, set[str]]:
    descriptor_obj_ids: set[str] = set()
    task_obj_ids: set[str] = set()
    descriptor_labels: set[str] = set()
    task_labels: set[str] = set()

    for entry in descriptor.manipulation_candidates + descriptor.articulation_hints:
        if not isinstance(entry, Mapping):
            continue
        obj_id = _candidate_id(entry)
        label = _candidate_label(entry).strip().lower()
        if obj_id:
            descriptor_obj_ids.add(obj_id)
        if label and label not in {"object", "unknown"}:
            descriptor_labels.add(label)

    if isinstance(task_targets, Mapping):
        for key in ("manipulation_candidates", "articulation_hints"):
            for raw in task_targets.get(key, []):
                if isinstance(raw, Mapping):
                    obj_id = _candidate_id(raw)
                    label = _candidate_label(raw).strip().lower()
                    if obj_id:
                        task_obj_ids.add(obj_id)
                    if label and label not in {"object", "unknown"}:
                        task_labels.add(label)
                elif isinstance(raw, str):
                    text = raw.strip()
                    if text:
                        task_obj_ids.add(text)
        for obj_id in task_targets.get("target_object_ids", []):
            text = str(obj_id).strip()
            if text:
                task_obj_ids.add(text)
        for obj_id in task_targets.get("articulation_required_ids", []):
            text = str(obj_id).strip()
            if text:
                task_obj_ids.add(text)

    return {
        "descriptor_obj_ids": descriptor_obj_ids,
        "task_obj_ids": task_obj_ids,
        "descriptor_labels": descriptor_labels,
        "task_labels": task_labels,
    }


def _candidate_rank_score(
    candidate: Mapping[str, Any],
    object_entry: Optional[Mapping[str, Any]],
    *,
    explicit: bool,
) -> float:
    score = 0.0
    if explicit:
        score += 2.0
    articulation = candidate.get("articulation") if isinstance(candidate.get("articulation"), Mapping) else {}
    if bool(articulation.get("required", False)):
        score += 1.5

    if object_entry is not None:
        score += _object_salience_score(object_entry) * 1.5
        detections = max(
            _safe_int(object_entry.get("n_total_detections"), 0),
            _safe_int(object_entry.get("n_frame_detections"), 0),
        )
        score += min(1.0, detections / 20.0)
        confidence = _safe_float(
            object_entry.get("mean_confidence"),
            _safe_float(object_entry.get("confidence"), 0.0),
        )
        score += _clamp01(confidence)
    else:
        score += 0.5

    return round(score, 6)


def _apply_ranked_cap(
    candidates: List[Dict[str, Any]],
    *,
    max_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_candidates <= 0:
        summary = {
            "pre_cap_count": len(candidates),
            "selected_count": len(candidates),
            "dropped_count": 0,
            "max_candidates": 0,
            "reserved_articulated_count": sum(
                1
                for item in candidates
                if isinstance(item.get("articulation"), Mapping)
                and bool(item["articulation"].get("required"))
            ),
            "cap_overridden_by_required": False,
        }
        return candidates, summary

    required = [
        item
        for item in candidates
        if isinstance(item.get("articulation"), Mapping)
        and bool(item["articulation"].get("required"))
    ]
    optional = [item for item in candidates if item not in required]

    selected: List[Dict[str, Any]] = list(required)
    remaining = max_candidates - len(selected)
    if remaining > 0:
        selected.extend(optional[:remaining])
    cap_overridden = len(required) > max_candidates

    summary = {
        "pre_cap_count": len(candidates),
        "selected_count": len(selected),
        "dropped_count": len(candidates) - len(selected),
        "max_candidates": max_candidates,
        "reserved_articulated_count": len(required),
        "cap_overridden_by_required": cap_overridden,
    }
    return selected, summary


def build_task_aware_swap_candidates_payload(
    *,
    descriptor: CaptureDescriptor,
    object_index_entries: List[Mapping[str, Any]],
    policy_path: Optional[str] = None,
    task_targets: Optional[Mapping[str, Any]] = None,
    selection_mode: str = "hybrid",
    max_candidates: int = 24,
) -> Dict[str, Any]:
    """Build swap candidates using task-aware explicit targets + ranked capping."""

    mode = _normalize_selection_mode(selection_mode)
    max_candidates = int(max_candidates or 0)

    if mode == "policy_only":
        descriptor_for_policy = replace(descriptor, manipulation_candidates=[], articulation_hints=[])
        descriptor_for_selection = descriptor_for_policy
    else:
        descriptor_for_selection = _merge_descriptor_with_task_targets(descriptor, task_targets)

    base_payload = build_swap_candidates_payload(
        descriptor=descriptor_for_selection,
        object_index_entries=object_index_entries,
        policy_path=policy_path,
    )

    if mode == "policy_only":
        explicit_sets = {
            "descriptor_obj_ids": set(),
            "task_obj_ids": set(),
            "descriptor_labels": set(),
            "task_labels": set(),
        }
    else:
        explicit_sets = _extract_explicit_sets(descriptor, task_targets)
    descriptor_obj_ids = explicit_sets["descriptor_obj_ids"]
    task_obj_ids = explicit_sets["task_obj_ids"]
    descriptor_labels = explicit_sets["descriptor_labels"]
    task_labels = explicit_sets["task_labels"]
    all_explicit_ids = descriptor_obj_ids | task_obj_ids
    all_explicit_labels = descriptor_labels | task_labels

    entry_by_id: Dict[str, Mapping[str, Any]] = {}
    for entry in object_index_entries:
        if not isinstance(entry, Mapping):
            continue
        obj_id = _candidate_id(entry)
        if obj_id and obj_id not in entry_by_id:
            entry_by_id[obj_id] = entry

    candidates_raw = base_payload.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError("invalid swap candidates payload")

    annotated: List[Dict[str, Any]] = []
    for item in candidates_raw:
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        obj_id = str(candidate.get("object_id") or "").strip()
        label_text = _normalized_text(candidate.get("label"))

        explicit_by_id = obj_id in all_explicit_ids
        explicit_by_label = any(token in label_text for token in all_explicit_labels if token)
        explicit = explicit_by_id or explicit_by_label

        selected_by = "policy"
        if explicit:
            if obj_id in task_obj_ids or any(token in label_text for token in task_labels if token):
                selected_by = "task_targets"
            elif obj_id in descriptor_obj_ids or any(
                token in label_text for token in descriptor_labels if token
            ):
                selected_by = "descriptor"
            else:
                selected_by = "explicit"

        source_entry = entry_by_id.get(obj_id)
        score = _candidate_rank_score(candidate, source_entry, explicit=explicit)
        selection_meta = {
            "explicit": explicit,
            "selected_by": selected_by,
            "rank_score": score,
        }
        candidate["selection"] = selection_meta
        annotated.append(candidate)

    if mode == "explicit_only":
        annotated = [item for item in annotated if bool(item.get("selection", {}).get("explicit"))]
    elif mode == "hybrid":
        explicit_candidates = [
            item for item in annotated if bool(item.get("selection", {}).get("explicit"))
        ]
        if explicit_candidates:
            if max_candidates > 0:
                policy_candidates = [
                    item for item in annotated if not bool(item.get("selection", {}).get("explicit"))
                ]
                slots = max(0, max_candidates - len(explicit_candidates))
                annotated = explicit_candidates + policy_candidates[:slots]
            else:
                annotated = explicit_candidates

    annotated.sort(
        key=lambda item: (
            bool(
                isinstance(item.get("articulation"), Mapping)
                and bool(item["articulation"].get("required"))
            ),
            bool(item.get("selection", {}).get("explicit")),
            _safe_float(item.get("selection", {}).get("rank_score"), 0.0),
        ),
        reverse=True,
    )

    selected, cap_summary = _apply_ranked_cap(annotated, max_candidates=max_candidates)
    out = dict(base_payload)
    out["candidates"] = selected
    out["selection_mode"] = mode
    out["max_image3d_candidates"] = max_candidates
    out["selection_summary"] = cap_summary
    out["selection_summary"]["explicit_count"] = sum(
        1 for item in selected if bool(item.get("selection", {}).get("explicit"))
    )
    out["selection_summary"]["policy_backfill_count"] = sum(
        1 for item in selected if str(item.get("selection", {}).get("selected_by")) == "policy"
    )
    out["task_targets_attached"] = bool(task_targets)
    out["task_target_object_ids"] = sorted(all_explicit_ids)
    return out


def write_task_targets(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist task-target payload for pipeline observability."""
    write_json(path, payload)
