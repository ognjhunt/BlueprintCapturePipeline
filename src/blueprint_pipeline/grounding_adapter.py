"""Spatial grounding adapter with a Holi-Spatial-compatible placeholder boundary."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import read_json_any, resolve_gs_uri_to_path, utc_now_iso


def normalize_spatial_grounding_backend(raw: str) -> str:
    candidate = (raw or "").strip().lower()
    return candidate if candidate in {"legacy", "holi_adapter"} else "legacy"


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(entry: Mapping[str, Any]) -> Dict[str, Any]:
    raw_box = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else None
    if raw_box is None and isinstance(entry.get("obb"), Mapping):
        raw_box = entry.get("obb")
    box = dict(raw_box) if isinstance(raw_box, Mapping) else {}

    center = box.get("center") if isinstance(box.get("center"), list) else [0.0, 0.0, 0.0]
    extents = box.get("extents") if isinstance(box.get("extents"), list) else [0.25, 0.25, 0.25]
    axes = box.get("axes") if isinstance(box.get("axes"), list) else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    quat = (
        box.get("orientationQuaternion")
        if isinstance(box.get("orientationQuaternion"), list)
        else [1.0, 0.0, 0.0, 0.0]
    )
    return {
        "center": [float(center[idx]) if idx < len(center) else 0.0 for idx in range(3)],
        "extents": [max(0.02, float(extents[idx]) if idx < len(extents) else 0.25) for idx in range(3)],
        "axes": axes,
        "orientationQuaternion": [float(quat[idx]) if idx < len(quat) else 0.0 for idx in range(4)],
    }


def _normalize_grounded_object(entry: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    object_id = _candidate_id(entry)
    label = _candidate_label(entry)
    confidence = _safe_float(entry.get("mean_confidence"), _safe_float(entry.get("confidence"), 0.0))
    merged_ids = entry.get("merged_object_ids") if isinstance(entry.get("merged_object_ids"), list) else []
    return {
        "object_id": object_id,
        "instance_id": object_id,
        "label": label,
        "category": str(entry.get("category") or "object"),
        "caption": str(entry.get("caption") or entry.get("description") or "").strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "source": source,
        "boundingBox": _normalize_bbox(entry),
        "merged_object_ids": [str(value).strip() for value in merged_ids if str(value).strip()],
        "n_total_detections": int(entry.get("n_total_detections", 0) or 0),
        "n_frame_detections": int(entry.get("n_frame_detections", 0) or 0),
    }


def _normalize_grounding_payload(
    payload: Mapping[str, Any],
    *,
    backend: str,
    descriptor: CaptureDescriptor,
    object_index_uri: str,
) -> Dict[str, Any]:
    raw_objects = payload.get("grounded_objects")
    if not isinstance(raw_objects, list):
        for key in ("objects", "instances", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_objects = value
                break
    if not isinstance(raw_objects, list):
        raw_objects = []

    grounded_objects = [
        _normalize_grounded_object(entry, source=str(entry.get("source") or backend))
        for entry in raw_objects
        if isinstance(entry, Mapping)
    ]

    out = {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "backend": backend,
        "backend_status": str(payload.get("backend_status") or "ok"),
        "placeholder": bool(payload.get("placeholder", backend == "holi_adapter")),
        "source_object_index_uri": object_index_uri,
        "grounded_objects": grounded_objects,
    }
    for key in ("manipulation_candidates", "articulation_hints", "navigation_hints", "tasks"):
        value = payload.get(key)
        if isinstance(value, list):
            out[key] = [dict(item) for item in value if isinstance(item, Mapping)]
    backend_report = payload.get("backend_report")
    if isinstance(backend_report, Mapping):
        out["backend_report"] = dict(backend_report)
    return out


def _run_holi_adapter_command(
    *,
    descriptor: CaptureDescriptor,
    storage_root: Path,
    object_index_uri: str,
    nurec_outputs: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    command_template = (os.getenv("HOLI_SPATIAL_COMMAND") or "").strip()
    if not command_template:
        return None

    object_index_path = resolve_gs_uri_to_path(object_index_uri, storage_root)
    if not object_index_path.exists():
        return {
            "backend_status": "fallback",
            "backend_report": {"status": "failed", "reason": f"missing_object_index:{object_index_path}"},
            "grounded_objects": [],
        }

    fd, tmp_name = tempfile.mkstemp(prefix="holi_grounding_", suffix=".json")
    os.close(fd)
    tmp_output = Path(tmp_name)
    try:
        substitutions = {
            "OBJECT_INDEX_PATH": str(object_index_path),
            "OUTPUT_JSON": str(tmp_output),
            "SCENE_ID": descriptor.scene_id,
            "CAPTURE_ID": descriptor.capture_id,
        }
        rendered = command_template
        for key, value in substitutions.items():
            rendered = rendered.replace("{" + key + "}", value)
        command = shlex.split(rendered)
        if not command:
            return {
                "backend_status": "fallback",
                "backend_report": {"status": "failed", "reason": "empty_holi_command"},
                "grounded_objects": [],
            }

        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        report: Dict[str, Any] = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "return_code": proc.returncode,
            "command": command,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        if proc.returncode != 0 or not tmp_output.exists():
            return {
                "backend_status": "fallback",
                "backend_report": report,
                "grounded_objects": [],
            }

        payload = read_json_any(tmp_output)
        if not isinstance(payload, Mapping):
            return {
                "backend_status": "fallback",
                "backend_report": {**report, "reason": f"unexpected_output_type:{type(payload).__name__}"},
                "grounded_objects": [],
            }
        out = dict(payload)
        out["backend_report"] = report
        return out
    finally:
        try:
            tmp_output.unlink(missing_ok=True)
        except Exception:
            pass


def infer_spatial_grounding(
    *,
    descriptor: CaptureDescriptor,
    storage_root: Path,
    object_index_uri: str,
    object_index_entries: List[Mapping[str, Any]],
    nurec_outputs: Optional[Mapping[str, Any]] = None,
    backend: str = "legacy",
) -> Dict[str, Any]:
    """Build normalized grounded-object payloads with an optional Holi adapter hook."""

    resolved_backend = normalize_spatial_grounding_backend(backend)
    legacy_payload = {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "backend": resolved_backend,
        "backend_status": "ok" if resolved_backend == "legacy" else "placeholder_fallback",
        "placeholder": resolved_backend == "holi_adapter",
        "source_object_index_uri": object_index_uri,
        "grounded_objects": [
            _normalize_grounded_object(entry, source="legacy_object_index")
            for entry in object_index_entries
            if isinstance(entry, Mapping)
        ],
    }
    if resolved_backend == "legacy":
        return legacy_payload

    adapter_payload = _run_holi_adapter_command(
        descriptor=descriptor,
        storage_root=storage_root,
        object_index_uri=object_index_uri,
        nurec_outputs=nurec_outputs or {},
    )
    if adapter_payload is None:
        return legacy_payload

    normalized = _normalize_grounding_payload(
        adapter_payload,
        backend=resolved_backend,
        descriptor=descriptor,
        object_index_uri=object_index_uri,
    )
    if not normalized["grounded_objects"]:
        normalized["grounded_objects"] = list(legacy_payload["grounded_objects"])
        normalized["backend_status"] = "placeholder_fallback"
        normalized["placeholder"] = True
    return normalized

