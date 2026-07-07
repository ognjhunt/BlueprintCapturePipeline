"""Build eval-ready task grounding for learned WAM and lightweight state checks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import read_json_any, utc_now_iso, write_json


SCHEMA_VERSION = "eval_ready_task_grounding.v1"
DEFAULT_OUTPUT_RELATIVE_PATH = "pipeline/simulation_automation/eval_ready_task_grounding.json"
DEFAULT_TASK_TEXT = "turn on the sink right handle"
DEFAULT_TARGET_LABEL = "right sink handle"
DEFAULT_HANDLE_AXIS = [0.0, 0.0, 1.0]
DEFAULT_HANDLE_ON_THRESHOLD_DEG = 35.0

_HANDLE_TOKENS = {"handle", "knob", "lever", "faucet"}
_SINK_TOKENS = {"sink", "faucet", "tap"}
_RIGHT_TOKENS = {"right", "rightmost"}
_ACTION_TOKENS = {
    "close",
    "deliver",
    "grasp",
    "inspect",
    "move",
    "open",
    "pick",
    "place",
    "press",
    "pull",
    "push",
    "scan",
    "toggle",
    "turn",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "onto",
    "the",
    "to",
    "with",
}
_SPATIAL_TOKENS = {
    "back",
    "bottom",
    "center",
    "front",
    "left",
    "lower",
    "middle",
    "rear",
    "right",
    "top",
    "upper",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalize_text(*parts: Any) -> str:
    text = " ".join(_string(part).lower() for part in parts if _string(part))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _tokens(*parts: Any) -> set[str]:
    return {token for token in _normalize_text(*parts).split() if token}


def _slug(value: Any, *, fallback: str = "task") -> str:
    text = _normalize_text(value).replace(" ", "_")
    return text.strip("_") or fallback


def _read_optional_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json_any(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_object_index(capture_root: Path) -> tuple[list[dict[str, Any]], Path | None]:
    candidates = [
        capture_root / "raw" / "object_index.json",
        capture_root / "raw" / "arkit" / "objects" / "index.json",
        capture_root / "object_index.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = read_json_any(path)
        raw_objects = payload.get("objects") if isinstance(payload, Mapping) else payload
        if not isinstance(raw_objects, list):
            return [], path
        objects = [dict(item) for item in raw_objects if isinstance(item, Mapping)]
        return objects, path
    return [], None


def _object_id(entry: Mapping[str, Any]) -> str:
    for key in ("object_id", "instance_id", "id", "uuid", "name"):
        value = _string(entry.get(key))
        if value:
            return value
    return ""


def _object_label(entry: Mapping[str, Any]) -> str:
    for key in ("label", "class_name", "category", "name"):
        value = _string(entry.get(key))
        if value:
            return value
    return _object_id(entry) or "object"


def _object_confidence(entry: Mapping[str, Any]) -> float:
    for key in ("mean_confidence", "confidence", "score"):
        if key in entry:
            return max(0.0, min(1.0, _safe_float(entry.get(key), 0.0)))
    return 0.0


def _object_crops(entry: Mapping[str, Any]) -> list[str]:
    crops = []
    reference_crop = _string(entry.get("reference_crop"))
    if reference_crop:
        crops.append(reference_crop)
    for value in _string_list(entry.get("all_crops")):
        if value not in crops:
            crops.append(value)
    for key in ("crop_paths", "image_paths"):
        for value in _string_list(entry.get(key)):
            if value not in crops:
                crops.append(value)
    return crops


def _object_has_mask_or_keypoint(entry: Mapping[str, Any]) -> bool:
    if entry.get("mask") or entry.get("mask_path") or entry.get("segmentation"):
        return True
    keypoints = entry.get("keypoints")
    if isinstance(keypoints, Mapping):
        return bool(keypoints)
    if isinstance(keypoints, list):
        return bool(keypoints)
    return False


def _object_has_tokens(entry: Mapping[str, Any], token_set: set[str]) -> bool:
    return bool(
        _tokens(
            _object_label(entry),
            _object_id(entry),
            entry.get("description"),
            entry.get("source_prompt"),
        )
        & token_set
    )


def _requires_articulated_handle_target(*, task_text: str, target_label: str) -> bool:
    tokens = _tokens(task_text, target_label)
    if tokens & _HANDLE_TOKENS:
        return True
    sink_or_faucet = tokens & _SINK_TOKENS
    manipulation = tokens & {"turn", "toggle", "open", "close", "press", "pull", "push"}
    return bool(sink_or_faucet and manipulation)


def _generic_default_task_from_objects(
    objects: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, dict[str, Any]]:
    for entry in objects:
        label = _object_label(entry)
        if not label:
            continue
        return (
            f"auto_inspect_{_slug(label, fallback='scene_object')}",
            f"inspect the {label}",
            label,
            {
                "default_task_source": "object_index_generic_default",
                "default_task_replaces_legacy_template": True,
                "selected_default_object_id": _object_id(entry),
                "selected_default_label": label,
            },
        )
    return (
        "auto_groundable_task_required",
        "inspect a groundable scene object",
        "groundable scene object",
        {
            "default_task_source": "object_index_missing_groundable_object",
            "default_task_replaces_legacy_template": True,
        },
    )


def _object_bbox(entry: Mapping[str, Any]) -> Any:
    for key in ("boundingBox", "bbox", "box", "mean_box_px", "obb"):
        value = entry.get(key)
        if value:
            return value
    return None


def derive_task_aware_detection_prompts(
    *,
    task_text: str,
    target_label: str = "",
    max_prompts: int = 18,
) -> list[str]:
    """Build detector prompts directly from customer task text.

    This keeps object extraction scene-unspecific: early customers can provide
    arbitrary tasks, and the object-index backends still get concrete task
    targets instead of only broad environment labels.
    """

    prompts: list[str] = []

    def add(value: Any) -> None:
        text = _normalize_text(value)
        if text and text not in prompts:
            prompts.append(text)

    add(target_label)
    add(task_text)
    raw_tokens = [token for token in _normalize_text(task_text, target_label).split() if token]
    content_tokens = [
        token
        for token in raw_tokens
        if token not in _STOPWORDS and not (token in _ACTION_TOKENS and len(raw_tokens) > 2)
    ]
    for size in (4, 3, 2):
        for index in range(0, max(0, len(content_tokens) - size + 1)):
            phrase_tokens = content_tokens[index : index + size]
            if not phrase_tokens:
                continue
            if all(token in _SPATIAL_TOKENS for token in phrase_tokens):
                continue
            add(" ".join(phrase_tokens))
    for token in content_tokens:
        if len(token) >= 3 and token not in _ACTION_TOKENS:
            add(token)
    token_set = set(content_tokens)
    if token_set & {"sink", "faucet", "tap"}:
        add("faucet handle")
        add("water stream")
    if token_set & {"button", "switch", "panel"}:
        for modifier in sorted(token_set & _SPATIAL_TOKENS):
            add(f"{modifier} button")
            add(f"{modifier} switch")
    if token_set & {"door", "drawer", "cabinet"}:
        add("handle")
    return prompts[:max_prompts]


def _target_prompts(*, task_text: str, target_label: str) -> list[str]:
    return derive_task_aware_detection_prompts(task_text=task_text, target_label=target_label)


def _score_object(entry: Mapping[str, Any], *, task_text: str, target_label: str) -> tuple[float, list[str]]:
    label = _object_label(entry)
    obj_id = _object_id(entry)
    haystack_tokens = _tokens(label, obj_id, entry.get("description"), entry.get("source_prompt"))
    target_tokens = _tokens(target_label)
    task_tokens = _tokens(task_text)
    reasons: list[str] = []
    score = 0.0

    if haystack_tokens & target_tokens:
        overlap = haystack_tokens & target_tokens
        score += 0.15 * len(overlap)
        reasons.append(f"target_token_overlap:{','.join(sorted(overlap))}")
    if haystack_tokens & _HANDLE_TOKENS:
        score += 0.45
        reasons.append("handle_semantics")
    if haystack_tokens & _SINK_TOKENS:
        score += 0.25
        reasons.append("sink_semantics")
    if haystack_tokens & _RIGHT_TOKENS:
        score += 0.2
        reasons.append("right_side_semantics")
    if "handle" in task_tokens and haystack_tokens & _HANDLE_TOKENS:
        score += 0.15
        reasons.append("task_requires_handle")
    if "sink" in task_tokens and haystack_tokens & _SINK_TOKENS:
        score += 0.1
        reasons.append("task_requires_sink")
    if _object_has_mask_or_keypoint(entry):
        score += 0.12
        reasons.append("mask_or_keypoint_available")
    if _object_crops(entry):
        score += 0.08
        reasons.append("crop_available")
    score += min(0.1, _object_confidence(entry) * 0.1)
    return min(1.0, score), reasons


def _candidate_row(entry: Mapping[str, Any], *, task_text: str, target_label: str) -> dict[str, Any]:
    score, reasons = _score_object(entry, task_text=task_text, target_label=target_label)
    return {
        "object_id": _object_id(entry),
        "label": _object_label(entry),
        "match_score": round(score, 6),
        "match_reasons": reasons,
        "confidence": _object_confidence(entry),
        "reference_crop": _object_crops(entry)[0] if _object_crops(entry) else None,
        "all_crops": _object_crops(entry),
        "bbox": _object_bbox(entry),
        "keypoints": entry.get("keypoints"),
        "mask_path": entry.get("mask_path"),
        "mask_or_keypoint_available": _object_has_mask_or_keypoint(entry),
        "raw_entry": {
            key: entry.get(key)
            for key in (
                "object_id",
                "instance_id",
                "id",
                "label",
                "class_name",
                "category",
                "source_prompt",
                "mean_confidence",
                "confidence",
                "keypoints",
                "mask_path",
            )
            if key in entry
        },
    }


def _select_task_targets(
    objects: Sequence[Mapping[str, Any]],
    *,
    task_text: str,
    target_label: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    candidates = [
        _candidate_row(entry, task_text=task_text, target_label=target_label)
        for entry in objects
    ]
    candidates.sort(
        key=lambda row: (
            row["match_score"],
            bool(row["mask_or_keypoint_available"]),
            bool(row["all_crops"]),
            row["confidence"],
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates and candidates[0]["match_score"] >= 0.45 else None
    parent_context = None
    if selected is None or "handle_semantics" not in selected.get("match_reasons", []):
        sink_rows = [
            row
            for row in candidates
            if "sink_semantics" in row.get("match_reasons", [])
            and "handle_semantics" not in row.get("match_reasons", [])
        ]
        parent_context = sink_rows[0] if sink_rows else None
    return selected, candidates[:8], parent_context


def _default_existing_path(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _infer_scene_asset_path(capture_root: Path) -> Path | None:
    task_manifest = _read_optional_mapping(
        capture_root / "pipeline" / "scene_wam_policy_episode_packet" / "scene_episode_task_manifest.json"
    )
    scene_asset = _string(task_manifest.get("scene_asset"))
    if scene_asset:
        path = Path(scene_asset).expanduser()
        if not path.is_absolute():
            path = capture_root / path
        if path.is_file():
            return path

    inspection = _read_optional_mapping(
        capture_root / "pipeline" / "simulation_automation" / "scene_asset_inspection.json"
    )
    for key in ("path", "asset_path", "scene_asset"):
        value = _string(inspection.get(key))
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = capture_root / path
        if path.is_file():
            return path

    inventory = _read_optional_mapping(
        capture_root / "pipeline" / "simulation_automation" / "scene_asset_inventory.json"
    )
    raw_assets = inventory.get("assets") if isinstance(inventory.get("assets"), list) else []
    for item in raw_assets:
        if not isinstance(item, Mapping):
            continue
        for key in ("path", "asset_path", "scene_asset"):
            value = _string(item.get(key))
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = capture_root / path
            if path.is_file():
                return path
    return None


def _path_ref(path_value: str | Path | None, *, capture_root: Path) -> dict[str, Any]:
    if path_value is None:
        return {"path": None, "exists": False}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = capture_root / path
    return {"path": str(path), "exists": path.is_file()}


def _parse_axis(value: str | Sequence[float] | None) -> tuple[list[float], str]:
    if value is None:
        return list(DEFAULT_HANDLE_AXIS), "default_unvalidated"
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = [str(part) for part in value]
    axis = [_safe_float(part, 0.0) for part in parts[:3]]
    while len(axis) < 3:
        axis.append(0.0)
    length = math.sqrt(sum(item * item for item in axis))
    if length <= 0:
        return list(DEFAULT_HANDLE_AXIS), "invalid_input_defaulted"
    return [round(item / length, 9) for item in axis], "operator_supplied"


def _nested_mapping(payload: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_number(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _matrix4(value: Any) -> list[list[float]] | None:
    if isinstance(value, list) and len(value) == 16:
        rows = [value[index : index + 4] for index in range(0, 16, 4)]
    elif (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) >= 4 for row in value)
    ):
        rows = value
    else:
        return None
    matrix: list[list[float]] = []
    for row in rows:
        numeric = [_safe_float(item, float("nan")) for item in row[:4]]
        if any(not math.isfinite(item) for item in numeric):
            return None
        matrix.append(numeric)
    return matrix


def _transform_point(matrix: list[list[float]] | None, point: Sequence[float]) -> list[float]:
    x, y, z = [float(point[index]) for index in range(3)]
    if matrix is None:
        return [x, y, z]
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def _camera_calibration_quality_gate(
    *,
    camera_ref: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    path = Path(_string(camera_ref.get("path"))) if camera_ref.get("path") else None
    blockers: list[str] = []
    warnings: list[str] = []
    payload = _read_optional_mapping(path) if path else {}
    intrinsics = _nested_mapping(payload, "intrinsics", "camera_intrinsics") or payload
    fx = _first_number(intrinsics, "fx", "focal_x")
    fy = _first_number(intrinsics, "fy", "focal_y")
    cx = _first_number(intrinsics, "cx", "principal_x")
    cy = _first_number(intrinsics, "cy", "principal_y")
    width = _first_number(intrinsics, "width", "image_width")
    height = _first_number(intrinsics, "height", "image_height")
    if not path or not path.is_file():
        blockers.append("missing_camera_calibration")
    if any(value is None or value <= 0 for value in (fx, fy, width, height)):
        blockers.append("camera_intrinsics_missing_or_invalid")
    if cx is None or cy is None:
        warnings.append("camera_principal_point_missing")
    if width and height and cx is not None and cy is not None:
        if not (0 <= cx <= width and 0 <= cy <= height):
            blockers.append("camera_principal_point_outside_image")
    extrinsics_present = bool(
        _matrix4(payload.get("camera_from_world"))
        or _matrix4(payload.get("T_camera_world"))
        or _matrix4(payload.get("world_from_camera"))
        or _matrix4(payload.get("T_world_camera"))
    )
    if not extrinsics_present:
        warnings.append("camera_extrinsics_missing_identity_projection_assumed")
    reprojection_error = _first_number(
        payload,
        "reprojection_error_px",
        "mean_reprojection_error_px",
        "alignment_error_px",
    )
    frame_alignment_confidence = _first_number(
        payload,
        "frame_alignment_confidence",
        "pose_confidence",
        "calibration_confidence",
    )
    score_parts = [
        0.42 if not any(value is None or value <= 0 for value in (fx, fy, width, height)) else 0.0,
        0.18 if cx is not None and cy is not None else 0.08,
        0.2 if extrinsics_present else 0.08,
    ]
    if reprojection_error is None:
        score_parts.append(0.1)
        warnings.append("camera_reprojection_error_missing")
    elif reprojection_error <= 2.0:
        score_parts.append(0.2)
    elif reprojection_error <= 5.0:
        score_parts.append(0.1)
        warnings.append("camera_reprojection_error_high")
    else:
        score_parts.append(0.0)
        blockers.append("camera_reprojection_error_too_high")
    if frame_alignment_confidence is not None:
        score_parts.append(max(0.0, min(0.2, frame_alignment_confidence * 0.2)))
    confidence = round(min(1.0, sum(score_parts)), 6)
    status = "blocked" if blockers else "passed" if confidence >= 0.75 else "warning"
    gate = {
        "schema_version": "camera_calibration_quality_gate.v1",
        "generated_at": generated_at,
        "status": status,
        "path": str(path) if path else None,
        "confidence": confidence,
        "intrinsics_present": not any(value is None or value <= 0 for value in (fx, fy, width, height)),
        "extrinsics_present": extrinsics_present,
        "reprojection_error_px": reprojection_error,
        "frame_alignment_confidence": frame_alignment_confidence,
        "image_size": {"width": width, "height": height},
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": {
            "calibration_gate_is_input_quality_check_not_success_proof": True,
            "identity_projection_assumption_is_low_confidence": not extrinsics_present,
        },
    }
    write_json(output_dir / "camera_calibration_quality_gate.json", gate)
    return gate


def _point3(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        for key in ("xyz", "world_xyz", "position", "translation", "center"):
            point = _point3(value.get(key))
            if point is not None:
                return point
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 3:
        point = [_safe_float(value[index], float("nan")) for index in range(3)]
        if all(math.isfinite(item) for item in point):
            return point
    return None


def _point2(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        for key in ("uv", "xy", "center", "point"):
            point = _point2(value.get(key))
            if point is not None:
                return point
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        point = [_safe_float(value[index], float("nan")) for index in range(2)]
        if all(math.isfinite(item) for item in point):
            return point
    return None


def _collect_robot_landmarks(robot_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(label: str, value: Any, *, source: str) -> None:
        point = _point3(value)
        if point is not None:
            rows.append({"label": label, "xyz": point, "source": source})

    for key in (
        "end_effector_xyz",
        "right_end_effector_xyz",
        "left_end_effector_xyz",
        "gripper_xyz",
        "right_gripper_xyz",
        "left_gripper_xyz",
    ):
        add(key.replace("_xyz", ""), robot_state.get(key), source=key)
    for key in (
        "end_effector_pose",
        "right_end_effector_pose",
        "left_end_effector_pose",
        "gripper_pose",
        "right_gripper_pose",
        "left_gripper_pose",
    ):
        add(key.replace("_pose", ""), robot_state.get(key), source=key)
    for collection_key in (
        "landmarks_3d",
        "keypoints_3d",
        "fk_landmarks",
        "link_poses",
        "joint_cartesian_positions",
        "joints",
        "frames",
    ):
        value = robot_state.get(collection_key)
        if isinstance(value, Mapping):
            for label, child in value.items():
                add(_string(label), child, source=collection_key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                label = _string(_mapping(child).get("label") or _mapping(child).get("name"))
                add(label or f"{collection_key}_{index}", child, source=collection_key)
    seen: set[tuple[str, tuple[float, float, float]]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row["label"], tuple(round(float(item), 6) for item in row["xyz"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _camera_projection_payload(camera_ref: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(_string(camera_ref.get("path"))) if camera_ref.get("path") else None
    payload = _read_optional_mapping(path) if path else {}
    intrinsics = _nested_mapping(payload, "intrinsics", "camera_intrinsics") or payload
    camera_from_world = (
        _matrix4(payload.get("camera_from_world"))
        or _matrix4(payload.get("T_camera_world"))
        or None
    )
    if camera_from_world is None and not (
        _matrix4(payload.get("world_from_camera")) or _matrix4(payload.get("T_world_camera"))
    ):
        camera_from_world = None
    return {
        "fx": _first_number(intrinsics, "fx", "focal_x"),
        "fy": _first_number(intrinsics, "fy", "focal_y"),
        "cx": _first_number(intrinsics, "cx", "principal_x") or 0.0,
        "cy": _first_number(intrinsics, "cy", "principal_y") or 0.0,
        "width": _first_number(intrinsics, "width", "image_width"),
        "height": _first_number(intrinsics, "height", "image_height"),
        "camera_from_world": camera_from_world,
    }


def _project_landmark(landmark: Mapping[str, Any], camera: Mapping[str, Any]) -> dict[str, Any]:
    point = _point3(landmark.get("xyz")) or [0.0, 0.0, 0.0]
    camera_point = _transform_point(camera.get("camera_from_world"), point)  # type: ignore[arg-type]
    fx = float(camera.get("fx") or 0.0)
    fy = float(camera.get("fy") or 0.0)
    cx = float(camera.get("cx") or 0.0)
    cy = float(camera.get("cy") or 0.0)
    width = camera.get("width")
    height = camera.get("height")
    z = camera_point[2]
    projectable = bool(fx > 0 and fy > 0 and z > 1e-6)
    u = fx * camera_point[0] / z + cx if projectable else None
    v = fy * camera_point[1] / z + cy if projectable else None
    in_frame = bool(
        projectable
        and u is not None
        and v is not None
        and (width is None or 0 <= u <= float(width))
        and (height is None or 0 <= v <= float(height))
    )
    return {
        "label": landmark.get("label"),
        "source": landmark.get("source"),
        "world_xyz": point,
        "camera_xyz": camera_point,
        "uv": [round(float(u), 6), round(float(v), 6)] if u is not None and v is not None else None,
        "projectable": projectable,
        "in_frame": in_frame,
    }


def _build_robot_fk_projection_manifest(
    *,
    robot_model_ref: Mapping[str, Any],
    robot_state_ref: Mapping[str, Any],
    camera_ref: Mapping[str, Any],
    calibration_gate: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    state_path = Path(_string(robot_state_ref.get("path"))) if robot_state_ref.get("path") else None
    robot_state = _read_optional_mapping(state_path) if state_path else {}
    landmarks = _collect_robot_landmarks(robot_state)
    camera = _camera_projection_payload(camera_ref)
    projected = [_project_landmark(row, camera) for row in landmarks]
    projected_count = sum(1 for row in projected if row.get("projectable"))
    in_frame_count = sum(1 for row in projected if row.get("in_frame"))
    trace_path = output_dir / "robot_fk_projected_skeleton_trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "timestamp": robot_state.get("timestamp", 0.0),
                "projected_landmark_count": projected_count,
                "in_frame_landmark_count": in_frame_count,
                "landmarks": projected,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    blockers: list[str] = []
    if not robot_model_ref.get("exists"):
        blockers.append("missing_robot_urdf_or_mjcf")
    if not state_path or not state_path.is_file():
        blockers.append("missing_robot_joint_or_end_effector_state")
    if not landmarks:
        blockers.append("robot_state_missing_cartesian_landmarks_for_fk_projection")
    if calibration_gate.get("status") == "blocked":
        blockers.append("camera_calibration_quality_gate_failed")
    if projected_count <= 0 and landmarks:
        blockers.append("robot_landmarks_not_projectable")
    projection_confidence = 0.0
    if landmarks:
        projection_confidence = (
            float(calibration_gate.get("confidence") or 0.0) * 0.55
            + (projected_count / len(landmarks)) * 0.3
            + (in_frame_count / len(landmarks)) * 0.15
        )
    manifest = {
        "schema_version": "robot_fk_projection_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if not blockers else "blocked",
        "robot_model": dict(robot_model_ref),
        "robot_state": dict(robot_state_ref),
        "camera_calibration": dict(camera_ref),
        "camera_calibration_quality_gate": str(output_dir / "camera_calibration_quality_gate.json"),
        "projection_method": (
            "cartesian_robot_state_projection_with_optional_camera_from_world"
        ),
        "urdf_or_mjcf_fk_solver_executed": False,
        "cartesian_landmark_count": len(landmarks),
        "projected_landmark_count": projected_count,
        "in_frame_landmark_count": in_frame_count,
        "projection_confidence": round(projection_confidence, 6),
        "trace_path": str(trace_path),
        "projected_landmarks": projected,
        "blockers": blockers,
        "warnings": [
            "urdf_mjcf_kinematic_chain_solver_not_executed"
        ] if landmarks else [],
        "claim_boundary": {
            "projection_is_action_conditioning_input_not_policy_success": True,
            "cartesian_state_projection_is_not_physical_contact_proof": True,
            "urdf_or_mjcf_fk_solver_executed": False,
        },
    }
    write_json(output_dir / "robot_fk_projection_manifest.json", manifest)
    return manifest


def _bbox_center_and_radius(target: Mapping[str, Any] | None) -> tuple[list[float] | None, float | None]:
    if not target:
        return None, None
    bbox = target.get("bbox")
    if isinstance(bbox, Mapping):
        if {"x", "y", "width", "height"} <= set(bbox):
            x = _safe_float(bbox.get("x"))
            y = _safe_float(bbox.get("y"))
            width = max(1.0, _safe_float(bbox.get("width"), 1.0))
            height = max(1.0, _safe_float(bbox.get("height"), 1.0))
            return [x + width * 0.5, y + height * 0.5], max(width, height) * 0.75
        if {"cx", "cy"} <= set(bbox):
            width = max(1.0, _safe_float(bbox.get("width"), 40.0))
            height = max(1.0, _safe_float(bbox.get("height"), 40.0))
            return [_safe_float(bbox.get("cx")), _safe_float(bbox.get("cy"))], max(width, height) * 0.75
        if {"width", "height"} <= set(bbox):
            width = max(1.0, _safe_float(bbox.get("width"), 40.0))
            height = max(1.0, _safe_float(bbox.get("height"), 40.0))
            keypoints = _mapping(target.get("raw_entry")).get("keypoints") or target.get("keypoints")
            center = _point2(keypoints) if keypoints else None
            return center, max(width, height) * 0.75
    keypoints = target.get("keypoints") or _mapping(target.get("raw_entry")).get("keypoints")
    if isinstance(keypoints, Mapping):
        for value in keypoints.values():
            point = _point2(value)
            if point:
                return point, 40.0
    return None, None


def _robot_state_numeric(robot_state_ref: Mapping[str, Any], *keys: str) -> float | None:
    path = Path(_string(robot_state_ref.get("path"))) if robot_state_ref.get("path") else None
    payload = _read_optional_mapping(path) if path else {}
    for key in keys:
        value = _first_number(payload, key)
        if value is not None:
            return value
        nested = _nested_mapping(payload, "articulation", "handle_proxy", "action_summary")
        value = _first_number(nested, key)
        if value is not None:
            return value
    return None


def _build_handle_proxy_state_check(
    *,
    proxy: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
    robot_state_ref: Mapping[str, Any],
    fk_projection: Mapping[str, Any],
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    if not proxy.get("state_check_configured"):
        payload = {
            "schema_version": "handle_proxy_state_check.v1",
            "generated_at": generated_at,
            "status": "skipped",
            "state_check_configured": False,
            "blockers": ["articulated_handle_proxy_not_configured"],
        }
        write_json(output_dir / "handle_proxy_state_check.json", payload)
        return payload
    threshold = _safe_float(proxy.get("on_threshold_deg"), DEFAULT_HANDLE_ON_THRESHOLD_DEG)
    measured_angle = _robot_state_numeric(
        robot_state_ref,
        "handle_joint_angle_deg",
        "right_handle_joint_angle_deg",
        "target_joint_angle_deg",
    )
    wrist_rotation = _robot_state_numeric(
        robot_state_ref,
        "wrist_rotation_delta_deg",
        "end_effector_rotation_delta_deg",
        "right_wrist_rotation_delta_deg",
        "commanded_rotation_delta_deg",
    )
    target_center, target_radius = _bbox_center_and_radius(selected)
    projected = [
        row
        for row in fk_projection.get("projected_landmarks", []) or []
        if isinstance(row, Mapping)
        and row.get("uv")
        and any(token in _string(row.get("label")).lower() for token in ("effector", "gripper", "hand"))
    ]
    end_effector_distance_px = None
    if target_center and projected:
        uv = projected[0]["uv"]
        end_effector_distance_px = round(
            math.hypot(float(uv[0]) - target_center[0], float(uv[1]) - target_center[1]),
            6,
        )
    contact_candidate = bool(
        end_effector_distance_px is not None
        and target_radius is not None
        and end_effector_distance_px <= max(40.0, target_radius * 1.75)
    )
    measured_on = bool(measured_angle is not None and abs(measured_angle) >= threshold)
    rotation_candidate = bool(wrist_rotation is not None and abs(wrist_rotation) >= threshold * 0.5)
    on_candidate = bool(measured_on or (contact_candidate and rotation_candidate))
    blockers = []
    if target_center is None:
        blockers.append("target_region_center_unavailable")
    if not projected:
        blockers.append("projected_end_effector_unavailable")
    if measured_angle is None and wrist_rotation is None:
        blockers.append("missing_handle_angle_or_wrist_rotation_delta")
    payload = {
        "schema_version": "handle_proxy_state_check.v1",
        "generated_at": generated_at,
        "status": "completed" if not blockers else "blocked",
        "state_check_configured": True,
        "handle_proxy_state": "on_candidate" if on_candidate else "off_or_unproven",
        "handle_transition": "off_to_on_candidate" if on_candidate else "not_proven",
        "target_object_id": proxy.get("target_object_id"),
        "measured_handle_angle_deg": measured_angle,
        "wrist_rotation_delta_deg": wrist_rotation,
        "on_threshold_deg": threshold,
        "target_center_px": target_center,
        "target_radius_px": target_radius,
        "end_effector_distance_px": end_effector_distance_px,
        "end_effector_target_contact_candidate": contact_candidate,
        "state_success_proven": False,
        "on_candidate": on_candidate,
        "blockers": blockers,
        "claim_boundary": {
            "on_candidate_is_lightweight_proxy_not_task_success": True,
            "measured_joint_state_required_for_exact_success": measured_angle is None,
            "physical_contact_validated": False,
        },
    }
    write_json(output_dir / "handle_proxy_state_check.json", payload)
    return payload


def _build_articulated_handle_proxy(
    *,
    target: Mapping[str, Any] | None,
    task_text: str,
    include_proxy: bool,
    handle_axis: str | Sequence[float] | None,
    on_threshold_deg: float,
) -> dict[str, Any]:
    task_mentions_handle = bool(_tokens(task_text) & _HANDLE_TOKENS)
    target_mentions_handle = bool(target and "handle_semantics" in target.get("match_reasons", []))
    if not include_proxy and not task_mentions_handle and not target_mentions_handle:
        return {
            "available": False,
            "state_check_configured": False,
            "reason": "task_does_not_request_articulated_handle_proxy",
        }
    axis, axis_source = _parse_axis(handle_axis)
    threshold = float(on_threshold_deg) if math.isfinite(float(on_threshold_deg)) else DEFAULT_HANDLE_ON_THRESHOLD_DEG
    target_id = _string(target.get("object_id")) if target else "unresolved_right_handle"
    return {
        "schema_version": "articulated_handle_proxy.v1",
        "available": True,
        "state_check_configured": True,
        "proxy_type": "revolute_sink_handle",
        "target_object_id": target_id,
        "axis": axis,
        "axis_source": axis_source,
        "initial_joint_angle_deg": 0.0,
        "on_threshold_deg": threshold,
        "off_threshold_deg": 5.0,
        "state_success_proven": False,
        "state_truth_source": "proxy_contract_only_no_measured_handle_state",
        "claim_boundary": {
            "proxy_is_not_real_articulation_proof": True,
            "water_flow_visual_label_required_for_visual_success": True,
            "measured_joint_state_required_for_exact_success": True,
        },
    }


def _load_task_anchor(capture_root: Path, *, task_id: str, selected_target_id: str) -> dict[str, Any]:
    manifest = _read_optional_mapping(capture_root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json")
    for task in manifest.get("tasks", []) if isinstance(manifest.get("tasks"), list) else []:
        if not isinstance(task, Mapping):
            continue
        target_ids = [str(item) for item in task.get("target_object_ids") or []]
        if _string(task.get("task_id")) == task_id or selected_target_id in target_ids:
            return dict(task)
    return {}


def build_eval_ready_task_grounding(
    *,
    capture_root: str | Path,
    task_id: str | None = None,
    task_text: str | None = None,
    target_label: str | None = None,
    scene_asset: str | Path | None = None,
    initial_frame: str | Path | None = None,
    camera_calibration: str | Path | None = None,
    robot_model: str | Path | None = None,
    robot_state: str | Path | None = None,
    output_path: str | Path | None = None,
    articulated_handle_proxy: bool = False,
    handle_axis: str | Sequence[float] | None = None,
    handle_on_threshold_deg: float = DEFAULT_HANDLE_ON_THRESHOLD_DEG,
) -> dict[str, Any]:
    """Write a task-grounding packet for WAM rollout requests and exact checks."""

    resolved_capture_root = Path(capture_root).expanduser().resolve()
    generated_at = utc_now_iso()
    object_index, object_index_path = _read_object_index(resolved_capture_root)
    default_task_metadata: dict[str, Any] = {
        "default_task_source": "explicit_task_contract",
        "default_task_replaces_legacy_template": False,
    }
    explicit_task_requested = any(
        value is not None for value in (task_id, task_text, target_label)
    )
    if not explicit_task_requested:
        task_id, task_text, target_label, default_task_metadata = (
            _generic_default_task_from_objects(object_index)
        )
    else:
        task_text = _string(task_text)
        target_label = _string(target_label)
        if not task_text and target_label:
            task_text = f"inspect the {target_label}"
        if not target_label and task_text:
            target_label = task_text
        if not task_text:
            task_text = DEFAULT_TASK_TEXT
        if not target_label:
            target_label = DEFAULT_TARGET_LABEL
        task_id = _string(task_id) or _slug(task_text, fallback="custom_task")
        if (
            task_text == DEFAULT_TASK_TEXT
            and target_label == DEFAULT_TARGET_LABEL
            and not any(
                _object_has_tokens(entry, _SINK_TOKENS | _HANDLE_TOKENS)
                for entry in object_index
            )
        ):
            default_task_metadata["explicit_legacy_sink_template_without_site_target"] = True
    selected, candidates, parent_context = _select_task_targets(
        object_index,
        task_text=task_text,
        target_label=target_label,
    )
    selected_target_id = _string(selected.get("object_id")) if selected else ""
    task_anchor = _load_task_anchor(
        resolved_capture_root,
        task_id=task_id,
        selected_target_id=selected_target_id,
    )

    inferred_scene_asset = _infer_scene_asset_path(resolved_capture_root)
    scene_ref = _path_ref(scene_asset or inferred_scene_asset, capture_root=resolved_capture_root)
    initial_frame_ref = _path_ref(
        initial_frame
        or _default_existing_path(
            resolved_capture_root
            / "pipeline"
            / "scene_wam_policy_episode_packet"
            / "rendered_observations"
            / "initial_policy_observation.jpg",
            resolved_capture_root / "raw" / "object_index_artifacts" / "keyframes" / "frame_0000.jpg",
        ),
        capture_root=resolved_capture_root,
    )
    camera_ref = _path_ref(
        camera_calibration
        or _default_existing_path(
            resolved_capture_root / "pipeline" / "geometry" / "camera" / "intrinsics.json",
            resolved_capture_root / "pipeline" / "evaluation_prep" / "camera_calibration.json",
            resolved_capture_root / "raw" / "camera_calibration.json",
        ),
        capture_root=resolved_capture_root,
    )
    robot_model_ref = _path_ref(robot_model, capture_root=resolved_capture_root)
    robot_state_ref = _path_ref(robot_state, capture_root=resolved_capture_root)
    proxy = _build_articulated_handle_proxy(
        target=selected,
        task_text=task_text,
        include_proxy=articulated_handle_proxy,
        handle_axis=handle_axis,
        on_threshold_deg=handle_on_threshold_deg,
    )
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else resolved_capture_root / DEFAULT_OUTPUT_RELATIVE_PATH
    )
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_gate = _camera_calibration_quality_gate(
        camera_ref=camera_ref,
        output_dir=output_dir,
        generated_at=generated_at,
    )
    fk_projection = _build_robot_fk_projection_manifest(
        robot_model_ref=robot_model_ref,
        robot_state_ref=robot_state_ref,
        camera_ref=camera_ref,
        calibration_gate=calibration_gate,
        output_dir=output_dir,
        generated_at=generated_at,
    )
    handle_proxy_state_check = _build_handle_proxy_state_check(
        proxy=proxy,
        selected=selected,
        robot_state_ref=robot_state_ref,
        fk_projection=fk_projection,
        output_dir=output_dir,
        generated_at=generated_at,
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if object_index_path is None:
        blockers.append("missing_object_index")
    if selected is None:
        blockers.append("missing_task_target_label_or_keypoint")
    else:
        reasons = set(selected.get("match_reasons", []))
        requires_handle_target = _requires_articulated_handle_target(
            task_text=task_text,
            target_label=target_label,
        )
        if requires_handle_target and "handle_semantics" not in reasons:
            blockers.append("missing_task_specific_handle_label_or_keypoint")
        if not selected.get("mask_or_keypoint_available"):
            warnings.append("selected_target_missing_mask_or_keypoint")
        if not selected.get("all_crops"):
            warnings.append("selected_target_missing_crop_refs")
    if not scene_ref["exists"] and not initial_frame_ref["exists"]:
        blockers.append("missing_3dgs_usd_or_initial_visual_scene_ref")
    if not camera_ref["exists"]:
        blockers.append("missing_camera_calibration")
    elif calibration_gate.get("status") == "blocked":
        blockers.append("camera_calibration_quality_gate_failed")
    elif calibration_gate.get("status") == "warning":
        warnings.append("camera_calibration_quality_gate_warning")
    if not robot_model_ref["exists"]:
        blockers.append("missing_robot_urdf_or_mjcf")
    if not robot_state_ref["exists"]:
        blockers.append("missing_robot_joint_or_end_effector_state")
    if fk_projection.get("status") != "completed":
        blockers.append("robot_fk_projection_blocked")
        for blocker in _string_list(fk_projection.get("blockers")):
            if blocker not in blockers:
                blockers.append(blocker)
    if not proxy.get("state_check_configured"):
        warnings.append("articulated_handle_proxy_not_configured")
    elif handle_proxy_state_check.get("status") == "blocked":
        warnings.extend(
            f"handle_proxy_state_check:{blocker}"
            for blocker in _string_list(handle_proxy_state_check.get("blockers"))
        )

    learned_rollout_request_ready = not blockers
    robot_projection_ready = bool(fk_projection.get("status") == "completed")
    requires_handle_target = _requires_articulated_handle_target(
        task_text=task_text,
        target_label=target_label,
    )
    if requires_handle_target:
        vlm_or_human_review_checks = [
            "right handle visibly moved in the intended direction",
            "water appears or faucet state visibly changes",
            "robot end effector interacts with the right handle rather than nearby fixtures",
            "generated rollout preserves scene structure and is not visually inconsistent",
        ]
        deterministic_or_lightweight_checks = [
            "project robot joints or end-effector through camera calibration into the task frame",
            "reject skeleton projections that miss the selected target region",
            "reject impossible rollouts with obvious kinematic discontinuities or target mismatch",
            "score handle-on only from measured/proxy joint angle when that state is available",
        ]
    else:
        vlm_or_human_review_checks = [
            "selected target remains the same object throughout the rollout",
            "robot observation focuses on the selected target rather than an unrelated fixture",
            "generated rollout preserves scene structure and is not visually inconsistent",
        ]
        deterministic_or_lightweight_checks = [
            "project robot joints or end-effector through camera calibration into the task frame",
            "reject skeleton projections that miss the selected target region",
            "reject impossible rollouts with obvious kinematic discontinuities or target mismatch",
        ]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_for_learned_wam_rollout_request" if learned_rollout_request_ready else "blocked",
        "capture_root": str(resolved_capture_root),
        "task": {
            "task_id": task_id,
            "task_text": task_text,
            "target_label": target_label,
            **default_task_metadata,
            "target_prompts_for_object_index_backends": _target_prompts(
                task_text=task_text,
                target_label=target_label,
            ),
            "task_anchor": task_anchor or None,
        },
        "object_index": {
            "path": str(object_index_path) if object_index_path else None,
            "exists": object_index_path is not None,
            "object_count": len(object_index),
            "upstream_crop_backends": [
                "object_index_stage",
                "OBJECT_INDEX_SAM3_COMMAND",
                "OBJECT_INDEX_GROUNDING_DINO_COMMAND",
                "OBJECT_INDEX_YOLO_WORLD_COMMAND",
            ],
            "backend_boundary": (
                "This artifact consumes existing detections/crops. It does not prove a "
                "specific SAM3, Grounding-DINO, or YOLO-World backend ran."
            ),
        },
        "selected_task_target": selected,
        "parent_context_target": parent_context,
        "task_relevant_region_candidates": candidates,
        "scene_and_observation_refs": {
            "scene_asset": scene_ref,
            "initial_visual_frame": initial_frame_ref,
            "camera_calibration": camera_ref,
        },
        "robot_conditioning_refs": {
            "robot_model": robot_model_ref,
            "robot_state": robot_state_ref,
            "required_conditioning_representation": "2d_kinematic_skeleton_projection",
            "projection_ready": robot_projection_ready,
            "robot_fk_projection_manifest": str(output_dir / "robot_fk_projection_manifest.json"),
            "robot_fk_projected_skeleton_trace": fk_projection.get("trace_path"),
            "projection_confidence": fk_projection.get("projection_confidence"),
        },
        "articulated_state_proxy": proxy,
        "camera_calibration_quality_gate": calibration_gate,
        "robot_fk_projection": fk_projection,
        "handle_proxy_state_check": handle_proxy_state_check,
        "generated_artifacts": {
            "camera_calibration_quality_gate": str(output_dir / "camera_calibration_quality_gate.json"),
            "robot_fk_projection_manifest": str(output_dir / "robot_fk_projection_manifest.json"),
            "robot_fk_projected_skeleton_trace": str(output_dir / "robot_fk_projected_skeleton_trace.jsonl"),
            "handle_proxy_state_check": str(output_dir / "handle_proxy_state_check.json"),
        },
        "success_check_plan": {
            "vlm_or_human_review_checks": vlm_or_human_review_checks,
            "deterministic_or_lightweight_checks": deterministic_or_lightweight_checks,
            "uncertain_case_routing": "human_review_required_for_low_confidence_or_visual_state_only_success",
        },
        "readiness": {
            "learned_rollout_request_ready": learned_rollout_request_ready,
            "target_crop_available": bool(selected and selected.get("all_crops")),
            "target_mask_or_keypoint_available": bool(selected and selected.get("mask_or_keypoint_available")),
            "camera_calibration_quality_status": calibration_gate.get("status"),
            "camera_calibration_confidence": calibration_gate.get("confidence"),
            "robot_projection_ready": robot_projection_ready,
            "robot_projection_confidence": fk_projection.get("projection_confidence"),
            "state_check_configured": bool(proxy.get("state_check_configured")),
            "handle_proxy_state": handle_proxy_state_check.get("handle_proxy_state"),
            "handle_proxy_on_candidate": bool(handle_proxy_state_check.get("on_candidate")),
            "exact_task_success_proven": False,
            "physical_contact_validated": False,
            "real_world_readiness_proven": False,
            "blockers": blockers,
            "warnings": warnings,
        },
        "claim_boundary": {
            "artifact_purpose": "task_grounding_preflight_for_learned_wam_and_lightweight_state_checks",
            "generated_or_detected_regions_are_support_artifacts": True,
            "raw_capture_evidence_remains_authoritative": True,
            "learned_wam_rollout_is_not_physical_success_proof": True,
            "vlm_success_label_is_review_signal_not_rank_fidelity_result": True,
            "proxy_articulation_is_not_simready_asset_proof": True,
            "simulator_physics_contact_validated": False,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "output_path": str(output),
    }
    write_json(output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument(
        "--task-id",
        default=None,
        help="Explicit task id. Omit to derive a site-grounded default from the object index.",
    )
    parser.add_argument(
        "--task-text",
        default=None,
        help="Explicit task text. Omit to derive a site-grounded default from the object index.",
    )
    parser.add_argument(
        "--target-label",
        default=None,
        help="Explicit target label. Omit to derive a site-grounded default from the object index.",
    )
    parser.add_argument("--scene-asset")
    parser.add_argument("--initial-frame")
    parser.add_argument("--camera-calibration")
    parser.add_argument("--robot-model")
    parser.add_argument("--robot-state")
    parser.add_argument("--output-path")
    parser.add_argument("--articulated-handle-proxy", action="store_true")
    parser.add_argument("--handle-axis")
    parser.add_argument("--handle-on-threshold-deg", type=float, default=DEFAULT_HANDLE_ON_THRESHOLD_DEG)
    args = parser.parse_args(argv)
    manifest = build_eval_ready_task_grounding(
        capture_root=args.capture_root,
        task_id=args.task_id,
        task_text=args.task_text,
        target_label=args.target_label,
        scene_asset=args.scene_asset,
        initial_frame=args.initial_frame,
        camera_calibration=args.camera_calibration,
        robot_model=args.robot_model,
        robot_state=args.robot_state,
        output_path=args.output_path,
        articulated_handle_proxy=args.articulated_handle_proxy,
        handle_axis=args.handle_axis,
        handle_on_threshold_deg=args.handle_on_threshold_deg,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "eval_ready_task_grounding": manifest["output_path"],
                "learned_rollout_request_ready": manifest["readiness"][
                    "learned_rollout_request_ready"
                ],
                "blockers": manifest["readiness"]["blockers"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
