"""WAM-derived perception and policy-observation adapter artifacts.

The harness turns WAM-generated pixels into derived support observations while
keeping evaluator-controlled state separate from pixel-inferred fields. It is a
deterministic fixture implementation by default; real segmentation, tracking,
depth, or pose backends can replace the fixture layer later without changing the
artifact family.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageStat, UnidentifiedImageError

from .common import ensure_dir, utc_now_iso, write_json


STEP_SCHEMA_VERSION = "wam_derived_observation_step.v1"
BUNDLE_SCHEMA_VERSION = "wam_derived_observation_bundle.v1"
MANIFEST_SCHEMA_VERSION = "wam_derived_observation_manifest.v1"
CHECKS_SCHEMA_VERSION = "wam_perception_harness_checks.v1"
ADAPTER_REPORT_SCHEMA_VERSION = "wam_policy_observation_adapter_report.v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "wam_derived_observation_claim_boundary.v1"
BACKEND_REQUEST_SCHEMA_VERSION = "wam_perception_backend_request.v1"
BACKEND_RESULT_SCHEMA_VERSION = "wam_perception_backend_result.v1"
VALIDATION_REPORT_SCHEMA_VERSION = "wam_perception_harness_validation_report.v1"
FALSE_SUCCESS_METRICS_SCHEMA_VERSION = "wam_false_success_reduction_metrics.v1"
REVIEW_REPORT_SCHEMA_VERSION = "wam_perception_harness_review_report.v1"
VALIDATION_LABEL_KEYS = (
    "expected_object_id",
    "object_id",
    "expected_target_visible",
    "target_visible",
    "expected_contact",
    "contact_expected",
    "actual_success",
    "real_success",
    "capture_success",
    "plain_video_success",
    "generated_video_success",
)
VALIDATION_ACCEPTED_TRUTH_KEYS = (
    "accepted_validation_label",
    "capture_backed",
    "capture_truth",
    "real_labeled_validation",
    "accepted_real_world_anchor",
    "operator_attested",
)
VALIDATION_SOURCE_KEYS = (
    "source_capture_path",
    "source_capture_bundle_path",
    "source_artifact_path",
    "source_manifest_path",
    "source_video_path",
    "source_frame_path",
    "source_label_path",
    "evidence_path",
    "operator_attestation_path",
)
VALIDATION_PROVENANCE_KEYS = (
    "reviewer_id",
    "reviewer",
    "reviewed_by",
    "review_decision",
    "review_status",
    "label_provenance",
    "source_label_path",
    "operator_attestation_path",
)

DEFAULT_EARLY_TERMINATION_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_EXTERNAL_BACKEND_TIMEOUT_SECONDS = 120
EXTERNAL_BACKEND_ENV_GATE = "BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND"
EXTERNAL_BACKEND_COMMAND_ENV = "BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND"

DERIVED_FIELD_NAMES = (
    "objects",
    "depth_estimates",
    "pose_estimates",
    "contact_likelihood",
    "consistency_checks",
    "uncertainty",
)

STATE_FIELD_NAMES = (
    "state",
    "robot_state",
    "proprioception",
    "unitree_g1_sonic_state",
    "base_pose",
    "base_velocity",
    "contact_state",
    "route_task_state",
    "object_state",
    "allowed_action_schema",
    "safety_limits",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = _string(value)
        return [text] if text else []
    if isinstance(value, Sequence):
        return [_string(item) for item in value if _string(item)]
    text = _string(value)
    return [text] if text else []


def _valid_ref_strings(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    refs: list[str] = []
    for key in keys:
        values = _sequence(row.get(key))
        if not values and row.get(key) is not None:
            values = [row.get(key)]
        for value in values:
            text = _string(value)
            if text and text.lower() not in {"none", "null", "n/a", "na", "todo", "tbd"}:
                refs.append(text)
    return refs


def _subprocess_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _redact_secret_text(value: Any) -> str:
    text = _string(value)
    if not text:
        return ""
    text = re.sub(r"hf_[A-Za-z0-9]+", "hf_[REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-[REDACTED]", text)
    text = re.sub(
        r"(token|api[_-]?key|authorization)=([^\s]+)",
        r"\1=[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_jsonable(item) for item in value]
        return str(value)


def _confidence(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(str(value))))
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _load_json_mapping(path_text: Any) -> dict[str, Any]:
    text = _string(path_text)
    if not text or "://" in text:
        return {}
    path = Path(text).expanduser()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_json_value(path_text: Any) -> Any:
    text = _string(path_text)
    if not text or "://" in text:
        return None
    path = Path(text).expanduser()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_json_sequence(path_text: Any) -> list[Any]:
    value = _load_json_value(path_text)
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("rows", "labels", "validation_rows", "anchors", "steps"):
            rows = value.get(key)
            if isinstance(rows, list):
                return rows
    return []


def _path_exists(path_text: Any) -> bool | None:
    text = _string(path_text)
    if not text or "://" in text:
        return None
    return Path(text).expanduser().is_file()


def _frame_stats(frame_path: str | Path | None) -> dict[str, Any]:
    text = _string(frame_path)
    if not text:
        return {
            "available": False,
            "readable": False,
            "blockers": ["missing_generated_frame_path"],
            "confidence": 0.0,
        }
    path = Path(text).expanduser()
    if not path.is_file():
        return {
            "available": False,
            "readable": False,
            "path": str(path),
            "blockers": ["generated_frame_missing"],
            "confidence": 0.0,
        }
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            stat = ImageStat.Stat(rgb)
            mean = sum(stat.mean) / max(len(stat.mean), 1)
            stddev = sum(stat.stddev) / max(len(stat.stddev), 1)
            width, height = rgb.size
    except (UnidentifiedImageError, OSError, ValueError):
        return {
            "available": True,
            "readable": False,
            "path": str(path),
            "width": None,
            "height": None,
            "brightness_mean": None,
            "contrast_stddev": None,
            "blockers": [],
            "warnings": ["generated_frame_not_decodable_for_quality_stats"],
            "confidence": 0.5,
        }
    blockers: list[str] = []
    if mean < 12.0:
        blockers.append("generated_frame_too_dark_for_harness")
    if stddev < 3.0:
        blockers.append("generated_frame_too_flat_for_harness")
    confidence = 0.82
    if blockers:
        confidence = 0.22
    return {
        "available": True,
        "readable": True,
        "path": str(path),
        "width": width,
        "height": height,
        "brightness_mean": round(mean, 4),
        "contrast_stddev": round(stddev, 4),
        "blockers": blockers,
        "confidence": confidence,
    }


def _multiview_items(
    source_generated_multiview_frame_paths: Mapping[str, Any] | Sequence[Any] | None,
) -> list[dict[str, Any]]:
    if isinstance(source_generated_multiview_frame_paths, Mapping):
        return [
            {"view_id": _string(key), "frame_path": _string(value)}
            for key, value in source_generated_multiview_frame_paths.items()
            if _string(value)
        ]
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(_sequence(source_generated_multiview_frame_paths)):
        if isinstance(value, Mapping):
            view_id = _string(value.get("view_id") or value.get("camera_id")) or f"view_{index}"
            frame_path = _string(value.get("frame_path") or value.get("path"))
        else:
            view_id = f"view_{index}"
            frame_path = _string(value)
        if frame_path:
            rows.append({"view_id": view_id, "frame_path": frame_path})
    return rows


def _multiview_summary(
    source_generated_multiview_frame_paths: Mapping[str, Any] | Sequence[Any] | None,
    *,
    primary_frame_path: str | Path | None,
) -> dict[str, Any]:
    items = _multiview_items(source_generated_multiview_frame_paths)
    primary = _string(primary_frame_path)
    if primary and all(_string(row.get("frame_path")) != primary for row in items):
        items.insert(0, {"view_id": "primary", "frame_path": primary})
    view_stats = [
        {
            "view_id": row["view_id"],
            "frame_path": row["frame_path"],
            "frame_quality": _frame_stats(row["frame_path"]),
        }
        for row in items
    ]
    readable = [
        row
        for row in view_stats
        if _mapping(row.get("frame_quality")).get("available")
        and _mapping(row.get("frame_quality")).get("readable")
    ]
    blockers = [
        f"{row.get('view_id')}:{blocker}"
        for row in view_stats
        for blocker in _string_list(_mapping(row.get("frame_quality")).get("blockers"))
    ]
    if len(view_stats) < 2:
        status = "not_evaluated"
        blockers.append("multiview_generated_observation_not_available")
        passed = None
    elif len(readable) == len(view_stats) and not blockers:
        status = "passed"
        passed = True
    else:
        status = "blocked"
        passed = False
        if len(readable) < 2:
            blockers.append("fewer_than_two_readable_generated_views")
    confidence = 0.0
    if view_stats:
        confidence = round(len(readable) / max(len(view_stats), 1), 4)
    return {
        "status": status,
        "passed": passed,
        "view_count": len(view_stats),
        "readable_view_count": len(readable),
        "views": view_stats,
        "confidence": confidence,
        "blockers": sorted(set(blockers)),
        "source": "generated_multiview_frame_quality_and_availability",
        "claim_boundary": {
            "multiview_check_is_generated_media_consistency_support": True,
            "multiview_check_is_not_physical_scene_truth": True,
        },
    }


def _backend_request_payload(
    *,
    generated_at: str,
    step_index: int,
    source_generated_frame_path: str | Path | None,
    source_generated_video_path: str | Path | None,
    source_generated_multiview_frame_paths: Mapping[str, Any] | Sequence[Any] | None,
    source_wam_rollout_id: str | None,
    transition_id: str | None,
    object_index: Mapping[str, Any],
    eval_ready_task_grounding: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
    source_policy_action: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": BACKEND_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "step_index": int(step_index),
        "source_generated_frame_path": _string(source_generated_frame_path) or None,
        "source_generated_video_path": _string(source_generated_video_path) or None,
        "source_generated_multiview_frame_paths": _jsonable(
            source_generated_multiview_frame_paths
        ),
        "source_wam_rollout_id": source_wam_rollout_id,
        "transition_id": transition_id,
        "source_truth": _source_truth(),
        "object_index": _jsonable(object_index),
        "eval_ready_task_grounding": _jsonable(eval_ready_task_grounding),
        "camera_calibration": _jsonable(camera_calibration),
        "source_policy_action": _jsonable(source_policy_action),
        "requested_outputs": [
            "objects",
            "tracks",
            "depth_estimates",
            "pose_estimates",
            "contact_likelihood",
            "uncertainty",
        ],
        "claim_boundary": _claim_boundary(),
    }


def _blocked_backend_result(
    *,
    backend_kind: str,
    command_path: str | None,
    env_gate: str,
    blockers: Sequence[str],
    request_path: Path | None,
    result_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": BACKEND_RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "backend": {
            "kind": backend_kind,
            "status": "blocked",
            "command_path": command_path,
            "env_gate": env_gate,
            "blockers": list(blockers),
            "request_path": str(request_path) if request_path else None,
            "result_path": str(result_path) if result_path else None,
            "real_sam_or_depth_model_ran": False,
            "credentials_persisted": False,
        },
        "objects": [],
        "depth_estimates": [],
        "pose_estimates": [],
        "contact_likelihood": None,
        "blockers": list(blockers),
        "claim_boundary": _claim_boundary(),
    }


def _run_external_backend(
    *,
    output_dir: str | Path,
    generated_at: str,
    step_index: int,
    backend_kind: str,
    backend_command: str | Sequence[str] | None,
    allow_external_backend: bool | None,
    backend_timeout_seconds: int,
    source_generated_frame_path: str | Path | None,
    source_generated_video_path: str | Path | None,
    source_generated_multiview_frame_paths: Mapping[str, Any] | Sequence[Any] | None,
    source_wam_rollout_id: str | None,
    transition_id: str | None,
    object_index: Mapping[str, Any],
    eval_ready_task_grounding: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
    source_policy_action: Mapping[str, Any],
) -> dict[str, Any]:
    if backend_kind == "fixture":
        return {
            "schema_version": BACKEND_RESULT_SCHEMA_VERSION,
            "status": "skipped_fixture_backend",
            "backend": {
                "kind": "fixture",
                "status": "completed",
                "command_path": None,
                "env_gate": None,
                "blockers": [],
                "request_path": None,
                "result_path": None,
                "real_sam_or_depth_model_ran": False,
                "credentials_persisted": False,
            },
            "objects": [],
            "depth_estimates": [],
            "pose_estimates": [],
            "contact_likelihood": None,
            "claim_boundary": _claim_boundary(),
        }

    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    request_path = output / "wam_perception_backend_request.json"
    result_path = output / "wam_perception_backend_result.json"
    request = _backend_request_payload(
        generated_at=generated_at,
        step_index=step_index,
        source_generated_frame_path=source_generated_frame_path,
        source_generated_video_path=source_generated_video_path,
        source_generated_multiview_frame_paths=source_generated_multiview_frame_paths,
        source_wam_rollout_id=source_wam_rollout_id,
        transition_id=transition_id,
        object_index=object_index,
        eval_ready_task_grounding=eval_ready_task_grounding,
        camera_calibration=camera_calibration,
        source_policy_action=source_policy_action,
    )
    write_json(request_path, request)
    command = backend_command or os.environ.get(EXTERNAL_BACKEND_COMMAND_ENV)
    raw_command_path = (
        " ".join(str(item) for item in command)
        if isinstance(command, Sequence) and not isinstance(command, str)
        else _string(command)
    )
    command_path = _redact_secret_text(raw_command_path)
    allowed = _truthy(allow_external_backend) or (
        allow_external_backend is None and _truthy(os.environ.get(EXTERNAL_BACKEND_ENV_GATE))
    )
    if not allowed:
        blocked = _blocked_backend_result(
            backend_kind=backend_kind,
            command_path=command_path or None,
            env_gate=EXTERNAL_BACKEND_ENV_GATE,
            blockers=["external_perception_backend_env_gate_not_enabled"],
            request_path=request_path,
            result_path=result_path,
        )
        write_json(result_path, blocked)
        return blocked
    if not raw_command_path:
        blocked = _blocked_backend_result(
            backend_kind=backend_kind,
            command_path=None,
            env_gate=EXTERNAL_BACKEND_COMMAND_ENV,
            blockers=["external_perception_backend_command_not_configured"],
            request_path=request_path,
            result_path=result_path,
        )
        write_json(result_path, blocked)
        return blocked

    args = (
        [str(item) for item in command]
        if isinstance(command, Sequence) and not isinstance(command, str)
        else shlex.split(raw_command_path)
    )
    if not args:
        blocked = _blocked_backend_result(
            backend_kind=backend_kind,
            command_path=command_path,
            env_gate=EXTERNAL_BACKEND_COMMAND_ENV,
            blockers=["external_perception_backend_command_empty"],
            request_path=request_path,
            result_path=result_path,
        )
        write_json(result_path, blocked)
        return blocked
    env = os.environ.copy()
    env.update(
        {
            "BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT": str(request_path),
            "BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT": str(result_path),
            "BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR": str(output),
        }
    )
    stdout_path = output / "wam_perception_backend.stdout.log"
    stderr_path = output / "wam_perception_backend.stderr.log"
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(backend_timeout_seconds)),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_subprocess_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_subprocess_text(exc.stderr), encoding="utf-8")
        blocked = _blocked_backend_result(
            backend_kind=backend_kind,
            command_path=command_path,
            env_gate=EXTERNAL_BACKEND_ENV_GATE,
            blockers=["external_perception_backend_command_timed_out"],
            request_path=request_path,
            result_path=result_path,
        )
        blocked["backend"]["stdout_path"] = str(stdout_path)
        blocked["backend"]["stderr_path"] = str(stderr_path)
        write_json(result_path, blocked)
        return blocked
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    payload = _load_json_mapping(result_path)
    if completed.returncode != 0 or not payload:
        blocked = _blocked_backend_result(
            backend_kind=backend_kind,
            command_path=command_path,
            env_gate=EXTERNAL_BACKEND_ENV_GATE,
            blockers=[
                "external_perception_backend_command_failed"
                if completed.returncode != 0
                else "external_perception_backend_result_missing_or_invalid"
            ],
            request_path=request_path,
            result_path=result_path,
        )
        blocked["returncode"] = completed.returncode
        write_json(result_path, blocked)
        return blocked
    backend = _mapping(payload.get("backend"))
    payload["schema_version"] = _string(payload.get("schema_version")) or BACKEND_RESULT_SCHEMA_VERSION
    payload["status"] = _string(payload.get("status")) or "completed"
    payload["backend"] = {
        "kind": _string(backend.get("kind")) or backend_kind,
        "status": _string(backend.get("status")) or payload["status"],
        "command_path": command_path,
        "env_gate": EXTERNAL_BACKEND_ENV_GATE,
        "blockers": _string_list(backend.get("blockers") or payload.get("blockers")),
        "request_path": str(request_path),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "real_sam_or_depth_model_ran": bool(
            backend.get("real_sam_or_depth_model_ran") or payload.get("real_model_ran")
        ),
        "provider_statuses": _jsonable(backend.get("provider_statuses")),
        "credentials_persisted": False,
    }
    payload["claim_boundary"] = _claim_boundary()
    write_json(result_path, payload)
    return payload


def _bbox_xyxy(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        if {"x", "y", "width", "height"} <= set(value):
            x = _float_value(value.get("x"), 0.0)
            y = _float_value(value.get("y"), 0.0)
            width = max(1.0, _float_value(value.get("width"), 1.0))
            height = max(1.0, _float_value(value.get("height"), 1.0))
            return [round(x, 3), round(y, 3), round(x + width, 3), round(y + height, 3)]
        if {"cx", "cy", "width", "height"} <= set(value):
            cx = _float_value(value.get("cx"), 0.0)
            cy = _float_value(value.get("cy"), 0.0)
            width = max(1.0, _float_value(value.get("width"), 1.0))
            height = max(1.0, _float_value(value.get("height"), 1.0))
            return [
                round(cx - width * 0.5, 3),
                round(cy - height * 0.5, 3),
                round(cx + width * 0.5, 3),
                round(cy + height * 0.5, 3),
            ]
    row = _sequence(value)
    if len(row) >= 4:
        try:
            values = [float(item) for item in row[:4]]
        except (TypeError, ValueError):
            return None
        x0, y0, x1, y1 = values
        if x1 <= x0 or y1 <= y0:
            x1 = x0 + max(1.0, x1)
            y1 = y0 + max(1.0, y1)
        return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]
    return None


def _bbox_offscreen(bbox: Sequence[float] | None, frame_stats: Mapping[str, Any]) -> bool:
    if not bbox:
        return False
    width = frame_stats.get("width")
    height = frame_stats.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return False
    x0, y0, x1, y1 = [float(item) for item in bbox[:4]]
    return x1 <= 0.0 or y1 <= 0.0 or x0 >= float(width) or y0 >= float(height)


def _bbox_area_ratio(bbox: Sequence[float] | None, frame_stats: Mapping[str, Any]) -> float | None:
    if not bbox:
        return None
    width = frame_stats.get("width")
    height = frame_stats.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = [float(item) for item in bbox[:4]]
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return max(0.0, min(1.0, area / float(width * height)))


def _default_bbox(frame_stats: Mapping[str, Any]) -> list[float]:
    width = frame_stats.get("width") if isinstance(frame_stats.get("width"), int) else 640
    height = frame_stats.get("height") if isinstance(frame_stats.get("height"), int) else 480
    return [
        round(float(width) * 0.35, 3),
        round(float(height) * 0.34, 3),
        round(float(width) * 0.65, 3),
        round(float(height) * 0.66, 3),
    ]


def _target_candidates(
    *,
    object_index: Mapping[str, Any],
    eval_ready_task_grounding: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    selected = _mapping(eval_ready_task_grounding.get("selected_task_target"))
    if selected:
        candidates.append({**selected, "_selection_source": "eval_ready_task_grounding"})
    objects = object_index.get("objects")
    if isinstance(objects, list):
        candidates.extend(
            {**dict(item), "_selection_source": "object_index"}
            for item in objects
            if isinstance(item, Mapping)
        )
    target_id = _string(
        observation.get("target_object_id") or _mapping(observation.get("state")).get("target_object_id")
    )
    if target_id:
        for item in candidates:
            if target_id in {
                _string(item.get("object_id")),
                _string(item.get("id")),
                _string(item.get("label")),
            }:
                return [item]
    return candidates


def _select_target(
    *,
    object_index: Mapping[str, Any],
    eval_ready_task_grounding: Mapping[str, Any],
    observation: Mapping[str, Any],
    frame_stats: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    candidates = _target_candidates(
        object_index=object_index,
        eval_ready_task_grounding=eval_ready_task_grounding,
        observation=observation,
    )
    if candidates:
        return candidates[0], False
    target_id = _string(
        observation.get("target_object_id") or _mapping(observation.get("state")).get("target_object_id")
    ) or "wam_fixture_visual_target"
    return (
        {
            "object_id": target_id,
            "label": _string(observation.get("target_label")) or target_id.replace("_", " "),
            "bbox": _default_bbox(frame_stats),
            "confidence": 0.52,
            "source": "fixture_generated_frame_center_prior",
            "_selection_source": "fixture_generated_frame_center_prior",
        },
        True,
    )


def _mask_ref(target: Mapping[str, Any]) -> dict[str, Any] | None:
    text = _string(target.get("mask_path") or target.get("mask_uri") or target.get("mask"))
    if not text:
        return None
    return {
        "path": text,
        "exists": _path_exists(text),
        "physical_truth": False,
        "derived_or_index_reference": True,
    }


def _asset_ref(target: Mapping[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        text = _string(target.get(key))
        if text:
            return {
                "path": text,
                "exists": _path_exists(text),
                "physical_truth": False,
                "derived_or_index_reference": True,
            }
    return None


def _object_record(
    *,
    target: Mapping[str, Any],
    fallback_target: bool,
    frame_stats: Mapping[str, Any],
    previous_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bbox = _bbox_xyxy(
        target.get("bbox")
        or target.get("boundingBox")
        or target.get("box")
        or target.get("mean_box_px")
    ) or _default_bbox(frame_stats)
    object_id = _string(target.get("object_id") or target.get("id")) or "wam_fixture_target"
    label = _string(target.get("label") or target.get("name")) or object_id
    track_id = _string(target.get("track_id")) or object_id
    confidence = _confidence(target.get("confidence"), 0.52 if fallback_target else 0.72)
    offscreen = _bbox_offscreen(bbox, frame_stats)
    if offscreen:
        confidence = min(confidence, 0.2)
    previous_objects = _sequence(_mapping(previous_steps[-1]).get("objects")) if previous_steps else []
    previous_ids = {
        _string(_mapping(item).get("object_id") or _mapping(item).get("track_id"))
        for item in previous_objects
    }
    if not previous_steps:
        temporal_stability = {
            "status": "first_observation",
            "stable": None,
            "confidence": None,
        }
    else:
        stable = object_id in previous_ids or track_id in previous_ids
        temporal_stability = {
            "status": "stable" if stable else "lost_or_changed",
            "stable": stable,
            "confidence": 0.82 if stable else 0.15,
        }
    return {
        "label": label,
        "object_id": object_id,
        "track_id": track_id,
        "bbox": {
            "xyxy": bbox,
            "source": _string(target.get("_selection_source")) or "fixture",
            "physical_truth": False,
        },
        "mask_ref": _mask_ref(target),
        "crop_ref": _asset_ref(target, "crop_path", "crop_uri", "image_crop_path"),
        "keypoints": _jsonable(
            target.get("keypoints") or target.get("keypoints_2d") or target.get("points")
        ),
        "physical_size_hint": {
            "known_width_m": target.get("known_width_m") or target.get("width_m"),
            "known_height_m": target.get("known_height_m") or target.get("height_m"),
            "known_diameter_m": target.get("diameter_m"),
            "source": _string(target.get("size_source")) or None,
            "truth_source": "object_index_or_task_grounding_metadata",
        },
        "confidence": round(confidence, 4),
        "source_prompt": _string(target.get("source_prompt") or target.get("prompt")) or label,
        "object_index_reference": {
            "source": _string(target.get("_selection_source")) or "fixture",
            "raw_entry_source": _string(target.get("source")) or None,
            "synthetic_label": bool(target.get("synthetic_label")),
            "object_index_path": _string(target.get("object_index_path")) or None,
            "eval_ready_task_grounding_path": _string(
                target.get("eval_ready_task_grounding_path")
            )
            or None,
        },
        "task_grounding_reference": {
            "target_prompt": _string(target.get("target_prompt") or target.get("source_prompt"))
            or None,
            "selected_task_target": bool(target.get("_selection_source") == "eval_ready_task_grounding"),
            "readiness_status": _string(target.get("readiness_status")) or None,
        },
        "temporal_stability": temporal_stability,
        "offscreen": offscreen,
        "claim_boundary": {
            "object_mask_or_bbox_is_derived_observation": True,
            "object_mask_or_bbox_is_not_physical_truth": True,
            "sam_or_detector_physical_truth_proven": False,
        },
    }


def _backend_completed(backend_result: Mapping[str, Any]) -> bool:
    if not backend_result:
        return False
    backend = _mapping(backend_result.get("backend"))
    return _string(backend_result.get("status")) in {"completed", "partial"} or _string(
        backend.get("status")
    ) in {"completed", "partial"}


def _backend_records(value: Any, key: str) -> list[dict[str, Any]]:
    records = _sequence(_mapping(value).get(key))
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _build_object_records(
    *,
    backend_result: Mapping[str, Any],
    target: Mapping[str, Any],
    fallback_target: bool,
    frame_stats: Mapping[str, Any],
    previous_steps: Sequence[Mapping[str, Any]],
    backend_kind: str,
) -> list[dict[str, Any]]:
    backend_objects = _backend_records(backend_result, "objects") if _backend_completed(backend_result) else []
    if backend_objects:
        return [
            _object_record(
                target={
                    **row,
                    "_selection_source": _string(row.get("source"))
                    or _string(_mapping(backend_result.get("backend")).get("kind"))
                    or backend_kind,
                },
                fallback_target=False,
                frame_stats=frame_stats,
                previous_steps=previous_steps,
            )
            for row in backend_objects
        ]
    return [
        _object_record(
            target=target,
            fallback_target=fallback_target,
            frame_stats=frame_stats,
            previous_steps=previous_steps,
        )
    ]


def _calibration_summary(camera_calibration: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(camera_calibration.get("camera_calibration_quality_gate"))
    if not gate:
        gate_path = _string(
            camera_calibration.get("camera_calibration_quality_gate_path")
            or camera_calibration.get("quality_gate_path")
        )
        gate = _load_json_mapping(gate_path)
    intrinsics = _mapping(camera_calibration.get("intrinsics")) or _mapping(
        camera_calibration.get("camera_intrinsics")
    ) or camera_calibration
    width = _float_value(intrinsics.get("width") or intrinsics.get("image_width"), 0.0)
    height = _float_value(intrinsics.get("height") or intrinsics.get("image_height"), 0.0)
    fx = _float_value(intrinsics.get("fx") or intrinsics.get("focal_x"), 0.0)
    fy = _float_value(intrinsics.get("fy") or intrinsics.get("focal_y"), 0.0)
    cx = _float_value(intrinsics.get("cx") or intrinsics.get("principal_x"), width * 0.5)
    cy = _float_value(intrinsics.get("cy") or intrinsics.get("principal_y"), height * 0.5)
    status = _string(gate.get("status")) or _string(camera_calibration.get("status"))
    confidence = _confidence(
        gate.get("confidence") or camera_calibration.get("confidence"),
        0.55 if fx > 0 and fy > 0 else 0.25,
    )
    return {
        "available": bool(camera_calibration),
        "status": status or ("available_without_quality_gate" if camera_calibration else "missing"),
        "confidence": confidence,
        "intrinsics": {
            "fx": fx or None,
            "fy": fy or None,
            "cx": cx if width else None,
            "cy": cy if height else None,
            "width": width or None,
            "height": height or None,
        },
        "source": _string(camera_calibration.get("source"))
        or _string(camera_calibration.get("path"))
        or _string(camera_calibration.get("camera_calibration_path"))
        or None,
        "quality_gate": _jsonable(gate) if gate else None,
        "claim_boundary": {
            "calibration_metadata_is_authoritative_only_when_from_capture_or_sim": True,
            "calibration_does_not_make_generated_pixels_physical_sensor_truth": True,
        },
    }


def _depth_estimate(
    *,
    obj: Mapping[str, Any],
    frame_stats: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    ratio = _bbox_area_ratio(_sequence(_mapping(obj.get("bbox")).get("xyxy")), frame_stats)
    if ratio is None:
        relative_depth = 0.5
        confidence = 0.3
    else:
        relative_depth = max(0.05, min(0.95, 1.0 - math.sqrt(ratio)))
        confidence = 0.48
    calibration = _calibration_summary(camera_calibration)
    metric_depth = None
    metric_source = "not_available"
    bbox = _sequence(_mapping(obj.get("bbox")).get("xyxy"))
    intrinsics = _mapping(calibration.get("intrinsics"))
    size_hint = _mapping(obj.get("physical_size_hint"))
    known_width_m = _float_value(
        size_hint.get("known_width_m")
        or size_hint.get("known_diameter_m")
        or obj.get("known_width_m")
        or obj.get("width_m")
        or obj.get("diameter_m"),
        0.0,
    )
    fx = _float_value(intrinsics.get("fx"), 0.0)
    if len(bbox) >= 4 and known_width_m > 0 and fx > 0:
        bbox_width = max(1.0, float(bbox[2]) - float(bbox[0]))
        metric_depth = round(max(0.001, (known_width_m * fx) / bbox_width), 4)
        metric_source = "calibrated_bbox_size_projection_from_generated_pixels"
        confidence = max(confidence, min(0.72, _confidence(calibration.get("confidence"), 0.45)))
    return {
        "label": obj.get("label"),
        "object_id": obj.get("object_id"),
        "relative_depth": round(relative_depth, 4),
        "metric_depth": metric_depth,
        "confidence": confidence,
        "calibration_source": _string(camera_calibration.get("source"))
        or _string(camera_calibration.get("path"))
        or "not_available_fixture_relative_depth",
        "metric_depth_source": metric_source,
        "metric_depth_truth": False,
        "sensor_depth": False,
        "calibration_summary": calibration,
        "claim_boundary": {
            "estimated_depth_is_not_sensor_depth": True,
            "metric_depth_truth": False,
        },
    }


def _pose_estimate(
    *,
    obj: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    bbox = _sequence(_mapping(obj.get("bbox")).get("xyxy"))
    center = None
    if len(bbox) >= 4:
        center = [round((float(bbox[0]) + float(bbox[2])) * 0.5, 3), round((float(bbox[1]) + float(bbox[3])) * 0.5, 3)]
    calibrated = bool(camera_calibration)
    calibration = _calibration_summary(camera_calibration)
    intrinsics = _mapping(calibration.get("intrinsics"))
    pose_3d = None
    depth = _float_value(obj.get("metric_depth"), 0.0)
    if center and depth > 0:
        fx = _float_value(intrinsics.get("fx"), 0.0)
        fy = _float_value(intrinsics.get("fy"), 0.0)
        cx = _float_value(intrinsics.get("cx"), 0.0)
        cy = _float_value(intrinsics.get("cy"), 0.0)
        if fx > 0 and fy > 0:
            pose_3d = {
                "camera_xyz_m": [
                    round((center[0] - cx) * depth / fx, 4),
                    round((center[1] - cy) * depth / fy, 4),
                    round(depth, 4),
                ],
                "source": "calibrated_bbox_depth_projection_from_generated_pixels",
                "physical_pose_truth": False,
            }
    return {
        "object_id": obj.get("object_id"),
        "pose_2d": {"center_px": center, "bbox_xyxy": bbox or None},
        "pose_3d": pose_3d,
        "confidence": 0.46 if calibrated else 0.34,
        "source": "fixture_bbox_pose_from_generated_frame",
        "calibration_source": _string(camera_calibration.get("source"))
        or _string(camera_calibration.get("path"))
        or None,
        "claim_boundary": {
            "pose_estimate_is_model_derived_support": True,
            "physical_pose_truth": False,
        },
    }


def _normalize_backend_depth_estimates(
    *,
    backend_result: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    frame_stats: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _backend_records(backend_result, "depth_estimates") if _backend_completed(backend_result) else []
    if not rows:
        return [
            _depth_estimate(obj=obj, frame_stats=frame_stats, camera_calibration=camera_calibration)
            for obj in objects
        ]
    normalized: list[dict[str, Any]] = []
    calibration = _calibration_summary(camera_calibration)
    for row in rows:
        normalized.append(
            {
                "label": row.get("label"),
                "object_id": row.get("object_id"),
                "relative_depth": row.get("relative_depth"),
                "metric_depth": row.get("metric_depth"),
                "confidence": _confidence(row.get("confidence"), 0.5),
                "calibration_source": _string(row.get("calibration_source"))
                or _string(calibration.get("source"))
                or "external_backend",
                "metric_depth_source": _string(row.get("metric_depth_source"))
                or "external_backend_estimate_from_generated_pixels",
                "metric_depth_truth": False,
                "sensor_depth": False,
                "calibration_summary": calibration,
                "claim_boundary": {
                    "estimated_depth_is_not_sensor_depth": True,
                    "metric_depth_truth": False,
                },
            }
        )
    return normalized


def _normalize_backend_pose_estimates(
    *,
    backend_result: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    depth_estimates: Sequence[Mapping[str, Any]],
    camera_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _backend_records(backend_result, "pose_estimates") if _backend_completed(backend_result) else []
    if rows:
        return [
            {
                "object_id": row.get("object_id"),
                "pose_2d": _jsonable(row.get("pose_2d")),
                "pose_3d": _jsonable(row.get("pose_3d")),
                "confidence": _confidence(row.get("confidence"), 0.45),
                "source": _string(row.get("source")) or "external_backend_generated_pixel_pose",
                "calibration_source": _string(row.get("calibration_source"))
                or _string(camera_calibration.get("source"))
                or None,
                "claim_boundary": {
                    "pose_estimate_is_model_derived_support": True,
                    "physical_pose_truth": False,
                },
            }
            for row in rows
        ]
    object_depth = {
        _string(row.get("object_id")): row.get("metric_depth") for row in depth_estimates
    }
    enriched_objects = []
    for obj in objects:
        enriched = dict(obj)
        enriched["metric_depth"] = object_depth.get(_string(obj.get("object_id")))
        enriched_objects.append(enriched)
    return [_pose_estimate(obj=obj, camera_calibration=camera_calibration) for obj in enriched_objects]


def _action_type(action: Mapping[str, Any]) -> str:
    return _string(
        action.get("action_type")
        or action.get("type")
        or _mapping(action.get("action")).get("action_type")
        or _mapping(action.get("normalized_action")).get("action_type")
    )


def _gripper_command(action: Mapping[str, Any]) -> Any:
    for key in ("gripper_command", "gripper", "gripper_targets", "hand_targets"):
        if action.get(key) is not None:
            return _jsonable(action.get(key))
    return None


def _projected_skeleton_refs(
    *,
    observation: Mapping[str, Any],
    skeleton_conditioning: Mapping[str, Any],
) -> dict[str, Any]:
    visual = _mapping(observation.get("visual_observation"))
    return {
        "projected_skeleton_trace_path": _string(
            skeleton_conditioning.get("projected_skeleton_trace_path")
            or observation.get("g1_projected_skeleton_trace_jsonl")
            or visual.get("g1_projected_skeleton_trace_jsonl")
        )
        or None,
        "projected_skeleton_trace_used": bool(
            skeleton_conditioning.get("projected_skeleton_trace_used")
        ),
        "landmark_count": skeleton_conditioning.get("projected_landmark_count"),
        "source": "evaluator_controlled_projection_or_action_conditioning",
    }


def _camera_calibration_from_observation(
    observation: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if explicit:
        return dict(explicit)
    visual = _mapping(observation.get("visual_observation"))
    camera = _mapping(observation.get("camera")) or _mapping(visual.get("camera"))
    for value in (
        camera.get("calibration"),
        observation.get("camera_calibration"),
        visual.get("camera_calibration"),
        camera.get("intrinsics"),
        visual.get("camera_intrinsics"),
        observation.get("camera_intrinsics"),
    ):
        mapping = _mapping(value)
        if mapping:
            return mapping
    for path_value in (
        observation.get("camera_calibration_path"),
        visual.get("camera_calibration_path"),
        camera.get("calibration_path"),
        observation.get("camera_calibration_quality_gate"),
        observation.get("camera_calibration_quality_gate_path"),
        visual.get("camera_calibration_quality_gate_path"),
    ):
        mapping = _load_json_mapping(path_value)
        if mapping:
            if "camera_calibration_quality_gate" in _string(path_value):
                return {"camera_calibration_quality_gate": mapping, "quality_gate_path": _string(path_value)}
            mapping.setdefault("path", _string(path_value))
            return mapping
    return {}


def _robot_state(
    *,
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    action_history: Sequence[Mapping[str, Any]],
    skeleton_conditioning: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
    controller_limits: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "action_command": _jsonable(action),
        "action_history": [_jsonable(row) for row in action_history],
        "gripper_command": _gripper_command(action),
        "nominal_joint_state": _jsonable(
            _mapping(observation.get("proprioception"))
            or _mapping(observation.get("unitree_g1_sonic_state"))
        ),
        "nominal_end_effector_pose": _jsonable(
            _mapping(observation.get("end_effector_pose"))
            or _mapping(_mapping(observation.get("state")).get("end_effector_pose"))
        ),
        "projected_skeleton_refs": _projected_skeleton_refs(
            observation=observation,
            skeleton_conditioning=skeleton_conditioning,
        ),
        "camera_calibration": _jsonable(camera_calibration),
        "controller_limits": _jsonable(
            controller_limits or _mapping(observation.get("safety_limits"))
        ),
        "channel_truth": {
            "action_command_evaluator_controlled": bool(action),
            "gripper_command_evaluator_controlled": _gripper_command(action) is not None,
            "nominal_joint_state_evaluator_controlled": bool(
                _mapping(observation.get("proprioception"))
                or _mapping(observation.get("unitree_g1_sonic_state"))
            ),
            "nominal_end_effector_pose_evaluator_controlled_or_fk": bool(
                _mapping(observation.get("end_effector_pose"))
                or _mapping(_mapping(observation.get("state")).get("end_effector_pose"))
            ),
            "projected_skeleton_evaluator_controlled": bool(
                skeleton_conditioning.get("projected_skeleton_trace_used")
            ),
            "camera_calibration_from_capture_or_sim_metadata": bool(camera_calibration),
            "controller_limits_evaluator_controlled": bool(
                controller_limits or _mapping(observation.get("safety_limits"))
            ),
            "robot_state_inferred_by_sam_or_pixel_detector": False,
        },
    }


def _contact_likelihood(
    *,
    action: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    robot_state: Mapping[str, Any],
) -> dict[str, Any]:
    action_name = _action_type(action)
    contact_action = any(token in action_name for token in ("contact", "grasp", "push", "place"))
    object_available = bool(objects)
    skeleton_available = bool(
        _mapping(robot_state.get("projected_skeleton_refs")).get("projected_skeleton_trace_used")
    )
    value = 0.18
    based_on = ["generated_frame_fixture_analysis"]
    if object_available:
        value += 0.18
        based_on.append("object_bbox_or_mask_reference")
    if contact_action:
        value += 0.25
        based_on.append(f"action_type:{action_name}")
    if skeleton_available:
        value += 0.08
        based_on.append("projected_robot_skeleton_ref")
    confidence = 0.45 if object_available else 0.25
    return {
        "value": round(min(0.85, value), 4),
        "confidence": confidence,
        "based_on": based_on,
        "physical_contact_proven": False,
        "stable_grasp_proven": False,
        "claim_boundary": {
            "mask_overlap_or_proximity_is_not_contact_proof": True,
            "physical_contact_proven": False,
        },
    }


def _normalize_backend_contact_likelihood(
    backend_result: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    contact = _mapping(backend_result.get("contact_likelihood"))
    if not (_backend_completed(backend_result) and contact):
        return dict(fallback)
    based_on = _string_list(contact.get("based_on"))
    if not based_on:
        based_on = ["external_backend_generated_pixel_analysis"]
    return {
        "value": _confidence(contact.get("value"), _confidence(fallback.get("value"), 0.0)),
        "confidence": _confidence(
            contact.get("confidence"), _confidence(fallback.get("confidence"), 0.3)
        ),
        "based_on": based_on,
        "physical_contact_proven": False,
        "stable_grasp_proven": False,
        "claim_boundary": {
            "mask_overlap_or_proximity_is_not_contact_proof": True,
            "physical_contact_proven": False,
        },
    }


def _review_acceptance_for_step(
    review_acceptance: Mapping[str, Any] | Sequence[Any] | None,
    *,
    step_index: int,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    if isinstance(review_acceptance, Mapping):
        steps = review_acceptance.get("steps") or review_acceptance.get("per_step")
        if isinstance(steps, Mapping):
            selected = _mapping(steps.get(str(step_index)) or steps.get(step_index))
        elif isinstance(steps, list):
            for row in steps:
                mapping = _mapping(row)
                if int(_float_value(mapping.get("step_index"), -1.0)) == int(step_index):
                    selected = mapping
                    break
        elif not review_acceptance.get("steps"):
            selected = dict(review_acceptance)
    elif isinstance(review_acceptance, Sequence) and not isinstance(
        review_acceptance, (str, bytes, bytearray)
    ):
        for row in review_acceptance:
            mapping = _mapping(row)
            if int(_float_value(mapping.get("step_index"), -1.0)) == int(step_index):
                selected = mapping
                break
    accepted = bool(
        selected.get("accepted_for_success_scoring")
        or selected.get("accepted")
        or selected.get("review_accepted")
    )
    reviewer = _string(
        selected.get("reviewer_id")
        or selected.get("reviewer")
        or selected.get("operator_id")
    )
    evidence_refs = _string_list(
        selected.get("evidence_refs")
        or selected.get("local_evidence_refs")
        or selected.get("artifact_refs")
        or selected.get("evidence_ref")
    )
    blockers: list[str] = []
    if not selected:
        status = "not_provided"
        blockers.append("review_acceptance_not_provided")
    elif not accepted:
        status = "blocked"
        blockers.append("review_acceptance_not_accepted")
    elif not reviewer:
        status = "blocked"
        blockers.append("review_acceptance_reviewer_missing")
    elif not evidence_refs:
        status = "blocked"
        blockers.append("review_acceptance_evidence_refs_missing")
    else:
        status = "accepted"
    return {
        "schema_version": "wam_perception_harness_review_acceptance.v1",
        "status": status,
        "step_index": int(step_index),
        "accepted_for_success_scoring": status == "accepted",
        "reviewer_id": reviewer or None,
        "evidence_refs": evidence_refs,
        "notes": _string(selected.get("notes") or selected.get("rationale")) or None,
        "blockers": blockers,
        "claim_boundary": {
            "review_acceptance_only_allows_generated_rollout_scoring_support": True,
            "review_acceptance_is_not_physical_robot_or_non_ranking_operational_claim": True,
        },
    }


def _external_consistency_rollout_check_passed(row: Mapping[str, Any]) -> bool:
    return bool(
        _truthy(row.get("forward_consistent", row.get("forward_dynamics_consistent")))
        and _truthy(row.get("inverse_consistent", row.get("inverse_dynamics_consistent")))
        and _truthy(row.get("visual_evidence_used"))
        and _truthy(row.get("action_trace_evidence_used"))
    )


def _forward_inverse_consistency_support_boundary() -> dict[str, Any]:
    return {
        "forward_inverse_consistency_is_reliability_review_signal_only": True,
        "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking": True,
        "forward_inverse_consistency_does_not_prove_policy_success": True,
        "forward_inverse_consistency_does_not_prove_task_success": True,
        "forward_inverse_consistency_does_not_prove_rank_fidelity": True,
        "forward_inverse_consistency_does_not_prove_deployment_readiness": True,
        "forward_inverse_consistency_does_not_prove_sensor_truth": True,
        "forward_inverse_consistency_is_not_external_validation": True,
        "consistency_metrics_are_support_signals_only": True,
        "evaluator_bounded_policy_ranking_upgraded_by_consistency": False,
        "policy_success_claimed_from_consistency": False,
        "task_success_claimed_from_consistency": False,
        "rank_fidelity_claimed_from_consistency": False,
        "deployment_readiness_claimed_from_consistency": False,
        "sensor_truth_claimed_from_consistency": False,
        "external_validation_claimed_from_consistency": False,
        "public_claim_upgrade_allowed": False,
    }


def _external_forward_inverse_consistency_status(
    external_consistency: Mapping[str, Any],
) -> dict[str, Any]:
    rollout_checks = [
        _mapping(row) for row in _sequence(external_consistency.get("rollout_checks"))
    ]
    scorer_id = _string(
        external_consistency.get("external_episode_consistency_scorer_id")
        or external_consistency.get("provider")
        or external_consistency.get("scorer")
        or external_consistency.get("label_source")
    )
    scorer_ran = bool(
        external_consistency.get("external_episode_consistency_scorer_ran")
        or external_consistency.get("external_scorer_ran")
        or external_consistency.get("scorer_ran")
    )
    checks_passed = bool(rollout_checks) and all(
        _external_consistency_rollout_check_passed(row) for row in rollout_checks
    )
    claimed = bool(external_consistency.get("forward_inverse_consistency_proven"))
    blockers: list[str] = []
    if claimed and not scorer_ran:
        blockers.append("external_episode_consistency_scorer_run_not_proven")
    if claimed and not scorer_id:
        blockers.append("external_episode_consistency_scorer_id_missing")
    if claimed and not rollout_checks:
        blockers.append("external_episode_consistency_rollout_checks_missing")
    elif claimed and not checks_passed:
        blockers.append("external_episode_consistency_rollout_checks_not_passing")
    proven = bool(claimed and scorer_ran and scorer_id and checks_passed)
    return {
        "proven": proven,
        "scorer_ran": scorer_ran,
        "scorer_id": scorer_id or None,
        "rollout_check_count": len(rollout_checks),
        "rollout_checks_passed": checks_passed,
        "blockers": sorted(set(blockers)),
        "claim_boundary": _forward_inverse_consistency_support_boundary(),
    }


def _consistency_checks(
    *,
    action: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    depth_estimates: Sequence[Mapping[str, Any]],
    contact_likelihood: Mapping[str, Any],
    previous_steps: Sequence[Mapping[str, Any]],
    external_consistency: Mapping[str, Any],
    multiview_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    action_name = _action_type(action)
    robot_motion_matches_command = {
        "status": "passed" if action_name else "requires_review",
        "passed": bool(action_name),
        "source": "evaluator_action_history",
        "blockers": [] if action_name else ["missing_action_command_for_motion_check"],
    }
    object_identity = {
        "status": "passed",
        "passed": True,
        "source": "fixture_track_id_stability",
        "blockers": [],
    }
    for obj in objects:
        stability = _mapping(obj.get("temporal_stability"))
        if stability.get("stable") is False:
            object_identity = {
                "status": "blocked",
                "passed": False,
                "source": "fixture_track_id_stability",
                "blockers": ["object_identity_lost_or_changed"],
            }
            blockers.append("object_identity_lost_or_changed")
            break
    depth_jump = {
        "status": "passed",
        "passed": True,
        "source": "fixture_relative_depth_temporal_check",
        "blockers": [],
    }
    if previous_steps and depth_estimates:
        previous_depths = _sequence(_mapping(previous_steps[-1]).get("depth_estimates"))
        if previous_depths:
            current = _confidence(depth_estimates[0].get("relative_depth"), 0.5)
            previous = _confidence(_mapping(previous_depths[0]).get("relative_depth"), 0.5)
            if abs(current - previous) > 0.35:
                depth_jump = {
                    "status": "blocked",
                    "passed": False,
                    "source": "fixture_relative_depth_temporal_check",
                    "blockers": ["relative_depth_temporal_jump_too_large"],
                }
                blockers.append("relative_depth_temporal_jump_too_large")
    contact_value = _confidence(contact_likelihood.get("value"), 0.0)
    contact_expected = any(token in action_name for token in ("contact", "grasp", "push", "place"))
    contact_plausible = (contact_value >= 0.35) if contact_expected else True
    if not contact_plausible:
        blockers.append("object_motion_contact_not_plausible")
    multiview = dict(multiview_summary)
    if multiview.get("status") == "blocked":
        blockers.extend(_string_list(multiview.get("blockers")))
    external_consistency_status = _external_forward_inverse_consistency_status(
        external_consistency
    )
    checks = {
        "robot_motion_matches_command": robot_motion_matches_command,
        "object_identity_stable": object_identity,
        "depth_temporal_jump_ok": depth_jump,
        "object_motion_contact_plausible": {
            "status": "passed" if contact_plausible else "blocked",
            "passed": contact_plausible,
            "source": "derived_contact_likelihood",
            "blockers": [] if contact_plausible else ["object_motion_contact_not_plausible"],
        },
        "multiview_consistent": multiview,
        "inverse_action_consistency": {
            "status": "external_scorer_passed"
            if external_consistency_status["proven"]
            else "separate_external_scorer_required",
            "passed": True if external_consistency_status["proven"] else None,
            "source": "external_wam_episode_consistency_scorer",
            "external_episode_consistency_scorer_ran": external_consistency_status[
                "scorer_ran"
            ],
            "external_episode_consistency_scorer_id": external_consistency_status[
                "scorer_id"
            ],
            "external_rollout_check_count": external_consistency_status[
                "rollout_check_count"
            ],
            "external_rollout_checks_passed": external_consistency_status[
                "rollout_checks_passed"
            ],
            "forward_inverse_consistency_proven": external_consistency_status["proven"],
            "harness_does_not_prove_forward_inverse_consistency": True,
            **_forward_inverse_consistency_support_boundary(),
            "claim_boundary": external_consistency_status["claim_boundary"],
            "blockers": external_consistency_status["blockers"],
        },
    }
    return checks, blockers


def _uncertainty(
    *,
    frame_stats: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    depth_estimates: Sequence[Mapping[str, Any]],
    robot_state: Mapping[str, Any],
    consistency_blockers: Sequence[str],
    confidence_threshold: float,
) -> dict[str, Any]:
    object_confidence = min(
        (_confidence(_mapping(row).get("confidence"), 0.0) for row in objects),
        default=0.42,
    )
    depth_confidence = min(
        (_confidence(_mapping(row).get("confidence"), 0.0) for row in depth_estimates),
        default=0.35,
    )
    frame_confidence = _confidence(frame_stats.get("confidence"), 0.0)
    robot_confidence = 0.74 if _mapping(robot_state.get("action_command")) else 0.42
    overall = round(
        max(0.0, min(1.0, (frame_confidence * 0.35) + (object_confidence * 0.3) + (depth_confidence * 0.15) + (robot_confidence * 0.2))),
        4,
    )
    reasons = [
        *_string_list(frame_stats.get("blockers")),
        *(str(item) for item in consistency_blockers if str(item)),
    ]
    for obj in objects:
        row = _mapping(obj)
        if row.get("offscreen"):
            reasons.append("target_object_offscreen_in_generated_observation")
        if _confidence(row.get("confidence"), 0.0) < 0.35:
            reasons.append("target_object_low_confidence")
    early = bool(
        reasons
        and (
            overall < confidence_threshold
            or any(
                marker in reason
                for reason in reasons
                for marker in (
                    "missing",
                    "blocked",
                    "not_configured",
                    "not_enabled",
                    "too_dark",
                    "too_flat",
                    "offscreen",
                    "lost",
                    "jump",
                )
            )
        )
    )
    return {
        "overall_confidence": overall,
        "confidence_threshold": confidence_threshold,
        "reasons": sorted(set(reasons)),
        "early_termination_recommended": early,
    }


def _source_truth() -> dict[str, bool]:
    return {
        "capture_truth": False,
        "physical_sensor_truth": False,
        "derived_from_generated_pixels": True,
        "model_derived_support_artifact": True,
    }


def build_wam_derived_observation_step(
    *,
    generated_at: str | None = None,
    step_index: int,
    source_generated_frame_path: str | Path | None = None,
    source_generated_video_path: str | Path | None = None,
    source_generated_multiview_frame_paths: Mapping[str, Any] | Sequence[Any] | None = None,
    source_wam_rollout_id: str | None = None,
    transition_id: str | None = None,
    source_policy_action: Mapping[str, Any] | None = None,
    action_history: Sequence[Mapping[str, Any]] | None = None,
    current_policy_observation: Mapping[str, Any] | None = None,
    object_index: Mapping[str, Any] | None = None,
    eval_ready_task_grounding: Mapping[str, Any] | None = None,
    skeleton_conditioning: Mapping[str, Any] | None = None,
    camera_calibration: Mapping[str, Any] | None = None,
    controller_limits: Mapping[str, Any] | None = None,
    previous_steps: Sequence[Mapping[str, Any]] | None = None,
    backend_kind: str = "fixture",
    external_backend_result: Mapping[str, Any] | None = None,
    external_consistency: Mapping[str, Any] | None = None,
    review_acceptance: Mapping[str, Any] | Sequence[Any] | None = None,
    confidence_threshold: float = DEFAULT_EARLY_TERMINATION_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    observation = _mapping(current_policy_observation)
    visual = _mapping(observation.get("visual_observation"))
    resolved_object_index = _mapping(object_index) or _load_json_mapping(
        observation.get("object_index_path") or visual.get("object_index_path")
    )
    resolved_grounding = _mapping(eval_ready_task_grounding) or _load_json_mapping(
        observation.get("eval_ready_task_grounding_path")
        or visual.get("eval_ready_task_grounding_path")
    )
    action = _mapping(source_policy_action)
    history = [_mapping(row) for row in action_history or []]
    skeleton = _mapping(skeleton_conditioning)
    calibration = _camera_calibration_from_observation(observation, camera_calibration)
    if not calibration:
        calibration = _mapping(resolved_grounding.get("camera_calibration"))
    if resolved_grounding.get("camera_calibration_quality_gate"):
        calibration = {
            **calibration,
            "camera_calibration_quality_gate": _mapping(
                resolved_grounding.get("camera_calibration_quality_gate")
            ),
        }
    frame_stats = _frame_stats(source_generated_frame_path)
    multiview = _multiview_summary(
        source_generated_multiview_frame_paths,
        primary_frame_path=source_generated_frame_path,
    )
    target, fallback_target = _select_target(
        object_index=resolved_object_index,
        eval_ready_task_grounding=resolved_grounding,
        observation=observation,
        frame_stats=frame_stats,
    )
    target = {
        **target,
        "object_index_path": _string(observation.get("object_index_path") or visual.get("object_index_path")) or None,
        "eval_ready_task_grounding_path": _string(
            observation.get("eval_ready_task_grounding_path")
            or visual.get("eval_ready_task_grounding_path")
        )
        or None,
        "readiness_status": _string(_mapping(resolved_grounding.get("readiness")).get("status"))
        or _string(resolved_grounding.get("status"))
        or None,
    }
    backend_result = _mapping(external_backend_result)
    if not backend_result:
        if backend_kind == "fixture":
            backend_result = _run_external_backend(
                output_dir=Path.cwd(),
                generated_at=generated,
                step_index=step_index,
                backend_kind="fixture",
                backend_command=None,
                allow_external_backend=False,
                backend_timeout_seconds=1,
                source_generated_frame_path=source_generated_frame_path,
                source_generated_video_path=source_generated_video_path,
                source_generated_multiview_frame_paths=source_generated_multiview_frame_paths,
                source_wam_rollout_id=source_wam_rollout_id,
                transition_id=transition_id,
                object_index=resolved_object_index,
                eval_ready_task_grounding=resolved_grounding,
                camera_calibration=calibration,
                source_policy_action=_mapping(source_policy_action),
            )
        else:
            backend_result = _blocked_backend_result(
                backend_kind=backend_kind,
                command_path=None,
                env_gate=EXTERNAL_BACKEND_COMMAND_ENV,
                blockers=["external_perception_backend_result_not_supplied"],
                request_path=None,
                result_path=None,
            )
    backend_metadata = _mapping(backend_result.get("backend"))
    backend_blockers: list[str] = []
    if backend_kind != "fixture":
        backend_blockers = _string_list(backend_metadata.get("blockers"))
        if _string(backend_metadata.get("status")) == "blocked" and not backend_blockers:
            backend_blockers.append("external_perception_backend_blocked")
    objects = _build_object_records(
        backend_result=backend_result,
        target=target,
        fallback_target=fallback_target,
        frame_stats=frame_stats,
        previous_steps=previous_steps or [],
        backend_kind=backend_kind,
    )
    depth_estimates = _normalize_backend_depth_estimates(
        backend_result=backend_result,
        objects=objects,
        frame_stats=frame_stats,
        camera_calibration=calibration,
    )
    pose_estimates = _normalize_backend_pose_estimates(
        backend_result=backend_result,
        objects=objects,
        depth_estimates=depth_estimates,
        camera_calibration=calibration,
    )
    robot_state = _robot_state(
        observation=observation,
        action=action,
        action_history=history,
        skeleton_conditioning=skeleton,
        camera_calibration=calibration,
        controller_limits=_mapping(controller_limits),
    )
    contact = _normalize_backend_contact_likelihood(
        backend_result,
        fallback=_contact_likelihood(
            action=action,
            objects=objects,
            robot_state=robot_state,
        ),
    )
    checks, consistency_blockers = _consistency_checks(
        action=action,
        objects=objects,
        depth_estimates=depth_estimates,
        contact_likelihood=contact,
        previous_steps=previous_steps or [],
        external_consistency=_mapping(external_consistency),
        multiview_summary=multiview,
    )
    uncertainty = _uncertainty(
        frame_stats=frame_stats,
        objects=objects,
        depth_estimates=depth_estimates,
        robot_state=robot_state,
        consistency_blockers=[*consistency_blockers, *backend_blockers],
        confidence_threshold=confidence_threshold,
    )
    review = _review_acceptance_for_step(review_acceptance, step_index=step_index)
    blockers = [
        *_string_list(frame_stats.get("blockers")),
        *backend_blockers,
        *consistency_blockers,
    ]
    if uncertainty["early_termination_recommended"]:
        blockers.append("wam_derived_observation_reliability_too_low_for_policy_requery")
    status = "completed" if not blockers else "blocked_reliability"
    if blockers and not frame_stats.get("available"):
        status = "blocked_missing_generated_frame"
    success_scoring_allowed = bool(
        frame_stats.get("available")
        and not uncertainty["early_termination_recommended"]
        and not backend_blockers
        and not consistency_blockers
    )
    success_scoring_review_accepted = bool(review.get("accepted_for_success_scoring"))
    if uncertainty["early_termination_recommended"] and success_scoring_review_accepted:
        success_scoring_allowed = True
    scoring_allowed = {
        "usable_for_diagnostics": bool(frame_stats.get("available")),
        "usable_for_policy_requery": not uncertainty["early_termination_recommended"],
        "usable_for_success_scoring": success_scoring_allowed,
        "success_scoring_review_accepted": success_scoring_review_accepted,
        "success_scoring_requires_explicit_review_acceptance": bool(
            uncertainty["early_termination_recommended"]
        ),
        "claim_boundary_summary": (
            "Harness outputs are derived support artifacts from generated media. They can "
            "guide diagnostics or policy requery gating, but they are not real sensors, "
            "contact proof, or rank fidelity by themselves."
        ),
    }
    return {
        "schema_version": STEP_SCHEMA_VERSION,
        "generated_at": generated,
        "step_index": int(step_index),
        "status": status,
        "source_generated_frame_path": _string(source_generated_frame_path) or None,
        "source_generated_video_path": _string(source_generated_video_path) or None,
        "source_generated_multiview_frame_paths": _jsonable(
            source_generated_multiview_frame_paths
        ),
        "source_wam_rollout_id": source_wam_rollout_id,
        "transition_id": transition_id,
        "source_truth": _source_truth(),
        "harness_backend": backend_metadata,
        "backend_result_path": backend_metadata.get("result_path"),
        "frame_quality": frame_stats,
        "multiview_observation": multiview,
        "task_grounding": {
            "object_index_available": bool(resolved_object_index),
            "eval_ready_task_grounding_available": bool(resolved_grounding),
            "object_index_path": _string(observation.get("object_index_path") or visual.get("object_index_path")) or None,
            "eval_ready_task_grounding_path": _string(
                observation.get("eval_ready_task_grounding_path")
                or visual.get("eval_ready_task_grounding_path")
            )
            or None,
            "target_prompts": _string_list(
                _mapping(resolved_grounding.get("task")).get(
                    "target_prompts_for_object_index_backends"
                )
            ),
            "selected_target_source": _string(target.get("_selection_source")) or None,
            "claim_boundary": {
                "task_grounding_guides_generated_media_analysis": True,
                "task_grounding_does_not_make_wam_pixels_capture_truth": True,
            },
        },
        "objects": objects,
        "depth_estimates": depth_estimates,
        "pose_estimates": pose_estimates,
        "robot_state": robot_state,
        "contact_likelihood": contact,
        "visual_cues": {
            "success_visual_cue": None,
            "failure_visual_cue": None,
            "success_or_failure_visual_cues_are_review_support_only": True,
        },
        "rollout_reviewability": {
            "reviewable_for_policy_requery": not uncertainty["early_termination_recommended"],
            "reviewable_for_success_scoring": success_scoring_allowed,
            "review_acceptance_required_for_success_scoring": bool(
                uncertainty["early_termination_recommended"]
            ),
            "review_acceptance": review,
        },
        "consistency_checks": checks,
        "uncertainty": uncertainty,
        "scoring_allowed": scoring_allowed,
        "blockers": sorted(set(blockers)),
        "claim_boundary": _claim_boundary(),
    }


def _field_set(schema: Mapping[str, Any] | None) -> set[str]:
    declared = _mapping(schema)
    fields: set[str] = set()
    for key in (
        "fields",
        "observation_fields",
        "supported_fields",
        "supported_observation_fields",
        "required_observation_fields",
        "fields_requested_by_policy",
    ):
        fields.update(item.lower() for item in _string_list(declared.get(key)))
    modalities = {item.lower() for item in _string_list(declared.get("modalities"))}
    fields.update(modalities)
    if declared.get("rgb_only"):
        fields.update({"rgb", "camera_frame_path", "visual_observation"})
    if declared.get("supports_depth"):
        fields.update({"depth", "depth_estimates"})
    if declared.get("supports_masks"):
        fields.update({"mask", "masks", "objects"})
    if declared.get("supports_state"):
        fields.update({"state", "robot_state"})
    if not fields:
        fields.update({"rgb", "camera_frame_path", "visual_observation"})
    return fields


def _supports(fields: set[str], *names: str) -> bool:
    return any(name.lower() in fields for name in names)


def adapt_policy_observation_for_declared_schema(
    *,
    policy_id: str | None,
    declared_policy_observation_schema: Mapping[str, Any] | None,
    base_policy_observation: Mapping[str, Any],
    derived_step: Mapping[str, Any],
) -> dict[str, Any]:
    fields = _field_set(declared_policy_observation_schema)
    adapted: dict[str, Any] = {}
    supplied: list[str] = []
    withheld: list[str] = []
    base = _mapping(base_policy_observation)
    visual = _mapping(base.get("visual_observation"))
    camera_frame = (
        derived_step.get("source_generated_frame_path")
        or base.get("camera_frame_path")
        or visual.get("camera_frame_path")
    )
    if _supports(fields, "rgb", "camera_frame_path", "visual_observation", "image"):
        adapted["camera_frame_path"] = camera_frame
        adapted["visual_observation"] = {
            **visual,
            "available": bool(camera_frame),
            "camera_frame_path": camera_frame,
            "wam_generated_observation": True,
            "physical_robot_sensor_proof": False,
        }
        supplied.extend(["camera_frame_path", "visual_observation"])
    for key in ("schema_version", "task_id", "task_prompt", "target_object_id"):
        if key in base:
            adapted[key] = _jsonable(base.get(key))
            supplied.append(key)
    if _supports(fields, "state", "nominal_state", "proprioception", "robot_state"):
        for key in STATE_FIELD_NAMES:
            if key in base:
                adapted[key] = _jsonable(base.get(key))
                supplied.append(key)
        if _supports(fields, "robot_state"):
            adapted["robot_state"] = _jsonable(derived_step.get("robot_state"))
            supplied.append("robot_state")
    if _supports(fields, "objects", "mask", "masks", "segmentation"):
        adapted["objects"] = _jsonable(derived_step.get("objects"))
        supplied.append("objects")
    else:
        withheld.append("objects")
    if _supports(fields, "depth", "rgbd", "rgb-d", "depth_estimates"):
        adapted["depth_estimates"] = _jsonable(derived_step.get("depth_estimates"))
        supplied.append("depth_estimates")
    else:
        withheld.append("depth_estimates")
    if _supports(fields, "pose", "pose_estimates", "object_pose"):
        adapted["pose_estimates"] = _jsonable(derived_step.get("pose_estimates"))
        supplied.append("pose_estimates")
    else:
        withheld.append("pose_estimates")
    if _supports(fields, "contact_likelihood", "contact"):
        adapted["contact_likelihood"] = _jsonable(derived_step.get("contact_likelihood"))
        supplied.append("contact_likelihood")
    else:
        withheld.append("contact_likelihood")
    if _supports(fields, "uncertainty", "consistency_checks"):
        adapted["uncertainty"] = _jsonable(derived_step.get("uncertainty"))
        adapted["consistency_checks"] = _jsonable(derived_step.get("consistency_checks"))
        supplied.extend(["uncertainty", "consistency_checks"])
    else:
        withheld.extend(["uncertainty", "consistency_checks"])
    missing_required = []
    if _supports(fields, "rgb", "camera_frame_path", "visual_observation") and not camera_frame:
        missing_required.append("camera_frame_path")
    early = bool(_mapping(derived_step.get("uncertainty")).get("early_termination_recommended"))
    adapter_status = "completed" if not missing_required else "blocked"
    safe_for_policy_requery = bool(adapter_status == "completed" and not early)
    return {
        "schema_version": ADAPTER_REPORT_SCHEMA_VERSION,
        "policy_id": policy_id,
        "declared_policy_observation_schema": _jsonable(
            _mapping(declared_policy_observation_schema)
        ),
        "fields_requested_by_policy": sorted(fields),
        "fields_supplied_to_policy": sorted(set(supplied)),
        "fields_withheld_due_to_contract": sorted(set(withheld) - set(supplied)),
        "adapter_status": adapter_status,
        "safe_for_policy_requery": safe_for_policy_requery,
        "missing_required_fields": missing_required,
        "early_termination_recommended": early,
        "adapted_policy_observation": adapted,
        "claim_boundary": {
            "policy_receives_only_declared_or_base_contract_fields": True,
            "rgb_only_policy_did_not_receive_masks_or_depth": not (
                _supports(fields, "objects", "mask", "masks", "segmentation")
                or _supports(fields, "depth", "rgbd", "rgb-d", "depth_estimates")
            ),
            "withheld_fields_remain_available_for_diagnostics": True,
        },
    }


def _claim_boundary() -> dict[str, Any]:
    return {
        "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
        "harness_outputs_are_derived_observations_not_real_sensors": True,
        "inferred_depth_is_not_sensor_depth": True,
        "object_masks_are_not_physical_truth": True,
        "mask_overlap_is_not_stable_grasp_or_contact_proof": True,
        "generated_rollout_success_is_not_rank_fidelity_result": True,
        "generated_video_labels_are_not_non_ranking_operational_claim": True,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_success_proven": False,
        "task_evaluation_run_and_post_training_data_package_support_artifact": True,
        **_forward_inverse_consistency_support_boundary(),
        "raw_capture_evidence_remains_authoritative": True,
    }


def _validation_rows(
    validation_set: Mapping[str, Any] | Sequence[Any] | None,
    validation_set_path: str | Path | None,
) -> list[dict[str, Any]]:
    rows: list[Any] = []
    if validation_set_path:
        rows = _load_json_sequence(validation_set_path)
    if not rows and isinstance(validation_set, Mapping):
        for key in ("rows", "labels", "validation_rows", "anchors", "steps"):
            value = validation_set.get(key)
            if isinstance(value, list):
                rows = value
                break
        if not rows and validation_set:
            rows = [validation_set]
    elif not rows:
        rows = _sequence(validation_set)
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _bool_label(row: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            return value
        text = _string(value).lower()
        if text in {"true", "yes", "success", "passed", "visible", "1"}:
            return True
        if text in {"false", "no", "failure", "failed", "blocked", "lost", "0"}:
            return False
    return None


def _row_has_validation_label(row: Mapping[str, Any]) -> bool:
    return any(_string(row.get(key)) for key in VALIDATION_LABEL_KEYS)


def _validation_acceptance_issues(row: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not any(_truthy(row.get(key)) for key in VALIDATION_ACCEPTED_TRUTH_KEYS):
        issues.append("validation_row_not_capture_backed_or_accepted_anchor")
    if not _valid_ref_strings(row, VALIDATION_SOURCE_KEYS):
        issues.append("validation_row_source_reference_missing")
    if not _valid_ref_strings(row, VALIDATION_PROVENANCE_KEYS):
        issues.append("validation_row_label_provenance_missing")
    return issues


def _step_by_index(steps: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for step in steps:
        result[int(_float_value(step.get("step_index"), -1.0))] = step
    return result


def _target_visible_from_step(step: Mapping[str, Any]) -> bool | None:
    objects = _sequence(step.get("objects"))
    if not objects:
        return None
    first = _mapping(objects[0])
    return bool(not first.get("offscreen") and _confidence(first.get("confidence"), 0.0) >= 0.35)


def _build_validation_report(
    *,
    generated_at: str,
    steps: Sequence[Mapping[str, Any]],
    validation_set: Mapping[str, Any] | Sequence[Any] | None,
    validation_set_path: str | Path | None,
) -> dict[str, Any]:
    rows = _validation_rows(validation_set, validation_set_path)
    if not rows:
        return {
            "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_requested",
            "validation_set_path": _string(validation_set_path) or None,
            "row_count": 0,
            "matched_step_count": 0,
            "metrics": {
                "object_id_accuracy": "not_measured",
                "target_visibility_accuracy": "not_measured",
                "contact_likelihood_accuracy": "not_measured",
                "false_success_reduction": "not_measured",
            },
            "blockers": [],
            "diagnostic_issues": [],
            "claim_boundary": {
                **_claim_boundary(),
                "labeled_validation_is_optional_for_sim_only_runs": True,
                "missing_validation_set_does_not_block_sim_only_runs": True,
            },
        }
    by_step = _step_by_index(steps)
    matched: list[dict[str, Any]] = []
    object_matches = 0
    object_total = 0
    visibility_matches = 0
    visibility_total = 0
    contact_matches = 0
    contact_total = 0
    plain_false_success = 0
    harness_false_success = 0
    validation_issues: list[str] = []
    for row in rows:
        step_index = int(_float_value(row.get("step_index"), -1.0))
        step = _mapping(by_step.get(step_index))
        if not step:
            validation_issues.append("validation_step_not_found_in_harness_output")
            matched.append(
                {
                    "step_index": step_index,
                    "status": "unmatched_step",
                    "diagnostic_issues": ["validation_step_not_found_in_harness_output"],
                }
            )
            continue
        if not _row_has_validation_label(row):
            validation_issues.append("validation_row_label_missing")
            matched.append(
                {
                    "step_index": step_index,
                    "status": "missing_validation_label",
                    "diagnostic_issues": ["validation_row_label_missing"],
                }
            )
            continue
        acceptance_issues = _validation_acceptance_issues(row)
        if acceptance_issues:
            validation_issues.extend(acceptance_issues)
            matched.append(
                {
                    "step_index": step_index,
                    "status": "validation_label_not_accepted",
                    "diagnostic_issues": acceptance_issues,
                }
            )
            continue
        objects = [_mapping(item) for item in _sequence(step.get("objects"))]
        first_object = objects[0] if objects else {}
        expected_object_id = _string(row.get("expected_object_id") or row.get("object_id"))
        object_match = None
        if expected_object_id:
            object_total += 1
            object_match = expected_object_id in {
                _string(first_object.get("object_id")),
                _string(first_object.get("track_id")),
                _string(first_object.get("label")),
            }
            object_matches += int(bool(object_match))
        expected_visible = _bool_label(row, "expected_target_visible", "target_visible")
        visible = _target_visible_from_step(step)
        visibility_match = None
        if expected_visible is not None and visible is not None:
            visibility_total += 1
            visibility_match = bool(expected_visible) == bool(visible)
            visibility_matches += int(visibility_match)
        expected_contact = _bool_label(row, "expected_contact", "contact_expected")
        contact_predicted = (
            _confidence(_mapping(step.get("contact_likelihood")).get("value"), 0.0) >= 0.5
        )
        contact_match = None
        if expected_contact is not None:
            contact_total += 1
            contact_match = bool(expected_contact) == bool(contact_predicted)
            contact_matches += int(contact_match)
        actual_success = _bool_label(row, "actual_success", "real_success", "capture_success")
        plain_success = _bool_label(row, "plain_video_success", "generated_video_success")
        harness_allows = bool(
            _mapping(step.get("scoring_allowed")).get("usable_for_success_scoring")
        )
        if actual_success is False and plain_success is True:
            plain_false_success += 1
            if harness_allows:
                harness_false_success += 1
        matched.append(
            {
                "step_index": step_index,
                "status": "matched",
                "object_match": object_match,
                "target_visibility_match": visibility_match,
                "contact_likelihood_match": contact_match,
                "plain_video_false_success": bool(actual_success is False and plain_success is True),
                "harness_false_success_after_gating": bool(
                    actual_success is False and plain_success is True and harness_allows
                ),
                "harness_success_scoring_allowed": harness_allows,
            }
        )
    matched_step_count = sum(1 for row in matched if row.get("status") == "matched")
    false_success_reduction_count = plain_false_success - harness_false_success
    false_success_reduction_rate = (
        round(false_success_reduction_count / plain_false_success, 6)
        if plain_false_success
        else None
    )
    if matched_step_count:
        status = "completed" if not validation_issues else "diagnostic_issues"
    elif validation_issues:
        status = "diagnostic_issues"
    else:
        status = "not_measured"
    return {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "validation_set_path": _string(validation_set_path) or None,
        "row_count": len(rows),
        "matched_step_count": matched_step_count,
        "per_step_validation": matched,
        "metrics": {
            "object_id_accuracy": round(object_matches / object_total, 6)
            if object_total
            else "not_measured",
            "target_visibility_accuracy": round(visibility_matches / visibility_total, 6)
            if visibility_total
            else "not_measured",
            "contact_likelihood_accuracy": round(contact_matches / contact_total, 6)
            if contact_total
            else "not_measured",
            "plain_video_false_success_count": plain_false_success,
            "harness_false_success_after_gating_count": harness_false_success,
            "false_success_reduction_count": false_success_reduction_count,
            "false_success_reduction_rate": false_success_reduction_rate,
        },
        "blockers": [],
        "diagnostic_issues": sorted(set(validation_issues))
        or ([] if matched_step_count else ["validation_steps_did_not_match_harness_steps"]),
        "claim_boundary": {
            **_claim_boundary(),
            "validation_metrics_use_optional_labeled_capture_or_anchor_rows": True,
            "validation_metrics_require_accepted_capture_or_anchor_labels": True,
            "validation_metrics_do_not_prove_evaluation_readiness": True,
            "validation_diagnostic_issues_do_not_block_generated_provider_runs": True,
        },
    }


def _false_success_metrics(validation_report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(validation_report.get("metrics"))
    if validation_report.get("status") == "not_requested":
        return {
            "schema_version": FALSE_SUCCESS_METRICS_SCHEMA_VERSION,
            "status": "not_requested",
            "blockers": [],
            "plain_video_false_success_count": "not_measured",
            "harness_false_success_after_gating_count": "not_measured",
            "false_success_reduction_count": "not_measured",
            "false_success_reduction_rate": "not_measured",
            "diagnostic_issues": [],
            "claim_boundary": {
                **_claim_boundary(),
                "labeled_validation_is_optional_for_sim_only_runs": True,
            },
        }
    if validation_report.get("status") != "completed":
        return {
            "schema_version": FALSE_SUCCESS_METRICS_SCHEMA_VERSION,
            "status": "not_measured",
            "blockers": [],
            "diagnostic_issues": _string_list(validation_report.get("diagnostic_issues"))
            or _string_list(validation_report.get("blockers"))
            or ["validation_report_not_completed"],
            "plain_video_false_success_count": "not_measured",
            "harness_false_success_after_gating_count": "not_measured",
            "false_success_reduction_count": "not_measured",
            "false_success_reduction_rate": "not_measured",
            "claim_boundary": {
                **_claim_boundary(),
                "false_success_reduction_not_measured_without_accepted_labels": True,
                "missing_or_incomplete_labels_do_not_block_generated_provider_runs": True,
            },
        }
    return {
        "schema_version": FALSE_SUCCESS_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "plain_video_false_success_count": metrics.get("plain_video_false_success_count"),
        "harness_false_success_after_gating_count": metrics.get(
            "harness_false_success_after_gating_count"
        ),
        "false_success_reduction_count": metrics.get("false_success_reduction_count"),
        "false_success_reduction_rate": metrics.get("false_success_reduction_rate"),
        "claim_boundary": {
            **_claim_boundary(),
            "false_success_reduction_is_measured_against_supplied_validation_rows": True,
        },
    }


def _review_report_markdown(
    *,
    manifest_path: Path,
    checks: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    adapter_report: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# WAM Perception Harness Review",
        "",
        f"- Status: `{checks.get('status')}`",
        f"- Step count: `{checks.get('step_count')}`",
        f"- Early termination recommended: `{checks.get('early_termination_recommended')}`",
        f"- Success scoring blocked: `{checks.get('success_scoring_blocked')}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Per-Step Reliability",
        "",
    ]
    if not steps:
        lines.append("- No derived observation steps were written.")
    for step in steps:
        uncertainty = _mapping(step.get("uncertainty"))
        scoring = _mapping(step.get("scoring_allowed"))
        reasons = ", ".join(_string_list(uncertainty.get("reasons"))) or "none"
        lines.append(
            "- Step "
            f"`{step.get('step_index')}`: status `{step.get('status')}`, "
            f"confidence `{uncertainty.get('overall_confidence')}`, "
            f"policy requery `{scoring.get('usable_for_policy_requery')}`, "
            f"success scoring `{scoring.get('usable_for_success_scoring')}`, "
            f"reasons `{reasons}`."
        )
    latest_adapter = _mapping(adapter_report.get("latest_policy_adapter_report"))
    lines.extend(
        [
            "",
            "## Policy Adapter",
            "",
            "- Fields supplied: `"
            + ", ".join(_string_list(latest_adapter.get("fields_supplied_to_policy")))
            + "`",
            "- Fields withheld: `"
            + ", ".join(_string_list(latest_adapter.get("fields_withheld_due_to_contract")))
            + "`",
            "",
            "## Validation Metrics",
            "",
            f"- Validation status: `{validation_report.get('status')}`",
            f"- Metrics: `{json.dumps(_mapping(validation_report.get('metrics')), sort_keys=True)}`",
            "",
            "## Claim Boundary",
            "",
            "Harness outputs are derived support artifacts from generated media. They are not "
            "real sensors, physical contact proof, or rank fidelity by themselves.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_checks(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    early = any(_mapping(row.get("uncertainty")).get("early_termination_recommended") for row in steps)
    blockers = sorted({item for row in steps for item in _string_list(row.get("blockers"))})
    scoring_blockers = []
    for row in steps:
        scoring = _mapping(row.get("scoring_allowed"))
        if not scoring.get("usable_for_success_scoring"):
            scoring_blockers.append(
                f"step_{row.get('step_index')}_not_usable_for_success_scoring"
            )
    success_scoring_blocked = bool(not steps or scoring_blockers)
    return {
        "schema_version": CHECKS_SCHEMA_VERSION,
        "status": "blocked" if not steps or early or blockers else "completed",
        "step_count": len(steps),
        "early_termination_recommended": early,
        "success_scoring_blocked": success_scoring_blocked,
        "success_scoring_blockers": [
            *([] if steps else ["derived_observation_steps_missing"]),
            *scoring_blockers,
            *(["wam_derived_observation_reliability_too_low"] if early else []),
        ],
        "forward_inverse_consistency_proven": False,
        "harness_output_does_not_prove_forward_inverse_consistency": True,
        **_forward_inverse_consistency_support_boundary(),
        "blockers": blockers,
        "checks": {
            "derived_observation_steps_present": bool(steps),
            "policy_requery_allowed_by_latest_step": bool(
                steps
                and not _mapping(steps[-1].get("uncertainty")).get(
                    "early_termination_recommended"
                )
            ),
            "success_scoring_allowed_by_all_steps": bool(
                steps and not success_scoring_blocked
            ),
            "physical_contact_proven": False,
            "sensor_depth_proven": False,
        },
        "claim_boundary": _claim_boundary(),
    }


def write_wam_derived_observation_artifacts(
    *,
    output_dir: str | Path,
    generated_at: str | None = None,
    steps: Sequence[Mapping[str, Any]],
    adapter_reports: Sequence[Mapping[str, Any]] | None = None,
    validation_set: Mapping[str, Any] | Sequence[Any] | None = None,
    validation_set_path: str | Path | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    normalized_steps = [dict(row) for row in steps]
    normalized_reports = [dict(row) for row in adapter_reports or []]
    steps_path = output / "wam_derived_observation_steps.jsonl"
    steps_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized_steps),
        encoding="utf-8",
    )
    checks = _aggregate_checks(normalized_steps)
    checks_path = output / "wam_perception_harness_checks.json"
    write_json(checks_path, checks)
    validation_report = _build_validation_report(
        generated_at=generated,
        steps=normalized_steps,
        validation_set=validation_set,
        validation_set_path=validation_set_path,
    )
    validation_report_path = output / "wam_perception_harness_validation_report.json"
    write_json(validation_report_path, validation_report)
    false_success_metrics = _false_success_metrics(validation_report)
    false_success_metrics_path = output / "wam_false_success_reduction_metrics.json"
    write_json(false_success_metrics_path, false_success_metrics)
    bundle_path = output / "wam_derived_observation_bundle.json"
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": checks["status"],
        "step_count": len(normalized_steps),
        "steps": normalized_steps,
        "latest_step": normalized_steps[-1] if normalized_steps else None,
        "validation_report": validation_report,
        "false_success_reduction_metrics": false_success_metrics,
        "source_truth": _source_truth(),
        "claim_boundary": _claim_boundary(),
    }
    write_json(bundle_path, bundle)
    adapter_report_path = output / "wam_policy_observation_adapter_report.json"
    adapter_report = {
        "schema_version": ADAPTER_REPORT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if normalized_reports else "blocked",
        "step_count": len(normalized_reports),
        "per_step_policy_adapter_reports": normalized_reports,
        "latest_policy_adapter_report": normalized_reports[-1] if normalized_reports else None,
        "claim_boundary": {
            "policy_receives_only_declared_or_base_contract_fields": True,
            "harness_outputs_may_be_withheld_from_policy_but_available_for_diagnostics": True,
        },
    }
    write_json(adapter_report_path, adapter_report)
    manifest_path = output / "wam_derived_observation_manifest.json"
    review_report_path = output / "wam_perception_harness_review_report.md"
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated,
        "status": checks["status"],
        "artifact_dir": str(output),
        "artifact_paths": {
            "wam_derived_observation_bundle": str(bundle_path),
            "wam_derived_observation_manifest": str(manifest_path),
            "wam_perception_harness_checks": str(checks_path),
            "wam_policy_observation_adapter_report": str(adapter_report_path),
            "wam_derived_observation_steps": str(steps_path),
            "wam_perception_harness_validation_report": str(validation_report_path),
            "wam_false_success_reduction_metrics": str(false_success_metrics_path),
            "wam_perception_harness_review_report": str(review_report_path),
        },
        "step_count": len(normalized_steps),
        "early_termination_recommended": checks["early_termination_recommended"],
        "success_scoring_blocked": checks["success_scoring_blocked"],
        "validation_status": validation_report.get("status"),
        "false_success_reduction_status": false_success_metrics.get("status"),
        "claim_boundary": _claim_boundary(),
    }
    write_json(manifest_path, manifest)
    review_report_path.write_text(
        _review_report_markdown(
            manifest_path=manifest_path,
            checks=checks,
            validation_report=validation_report,
            adapter_report=adapter_report,
            steps=normalized_steps,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "bundle": bundle,
        "checks": checks,
        "adapter_report": adapter_report,
        "validation_report": validation_report,
        "false_success_reduction_metrics": false_success_metrics,
        "artifact_paths": manifest["artifact_paths"],
    }


def run_wam_derived_observation_harness_step(
    *,
    output_dir: str | Path,
    generated_at: str | None = None,
    step_index: int,
    source_generated_frame_path: str | Path | None = None,
    source_generated_video_path: str | Path | None = None,
    source_generated_multiview_frame_paths: Mapping[str, Any] | Sequence[Any] | None = None,
    source_wam_rollout_id: str | None = None,
    transition_id: str | None = None,
    source_policy_action: Mapping[str, Any] | None = None,
    action_history: Sequence[Mapping[str, Any]] | None = None,
    current_policy_observation: Mapping[str, Any] | None = None,
    object_index: Mapping[str, Any] | None = None,
    eval_ready_task_grounding: Mapping[str, Any] | None = None,
    skeleton_conditioning: Mapping[str, Any] | None = None,
    camera_calibration: Mapping[str, Any] | None = None,
    controller_limits: Mapping[str, Any] | None = None,
    previous_steps: Sequence[Mapping[str, Any]] | None = None,
    previous_adapter_reports: Sequence[Mapping[str, Any]] | None = None,
    backend_kind: str = "fixture",
    backend_command: str | Sequence[str] | None = None,
    allow_external_backend: bool | None = None,
    backend_timeout_seconds: int = DEFAULT_EXTERNAL_BACKEND_TIMEOUT_SECONDS,
    policy_id: str | None = None,
    declared_policy_observation_schema: Mapping[str, Any] | None = None,
    external_consistency: Mapping[str, Any] | None = None,
    review_acceptance: Mapping[str, Any] | Sequence[Any] | None = None,
    validation_set: Mapping[str, Any] | Sequence[Any] | None = None,
    validation_set_path: str | Path | None = None,
    confidence_threshold: float = DEFAULT_EARLY_TERMINATION_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    observation = _mapping(current_policy_observation)
    visual = _mapping(observation.get("visual_observation"))
    resolved_object_index = _mapping(object_index) or _load_json_mapping(
        observation.get("object_index_path") or visual.get("object_index_path")
    )
    resolved_grounding = _mapping(eval_ready_task_grounding) or _load_json_mapping(
        observation.get("eval_ready_task_grounding_path")
        or visual.get("eval_ready_task_grounding_path")
    )
    calibration = _camera_calibration_from_observation(observation, camera_calibration)
    if not calibration:
        calibration = _mapping(resolved_grounding.get("camera_calibration"))
    if resolved_grounding.get("camera_calibration_quality_gate"):
        calibration = {
            **calibration,
            "camera_calibration_quality_gate": _mapping(
                resolved_grounding.get("camera_calibration_quality_gate")
            ),
        }
    backend_result = _run_external_backend(
        output_dir=output_dir,
        generated_at=generated,
        step_index=step_index,
        backend_kind=backend_kind,
        backend_command=backend_command,
        allow_external_backend=allow_external_backend,
        backend_timeout_seconds=backend_timeout_seconds,
        source_generated_frame_path=source_generated_frame_path,
        source_generated_video_path=source_generated_video_path,
        source_generated_multiview_frame_paths=source_generated_multiview_frame_paths,
        source_wam_rollout_id=source_wam_rollout_id,
        transition_id=transition_id,
        object_index=resolved_object_index,
        eval_ready_task_grounding=resolved_grounding,
        camera_calibration=calibration,
        source_policy_action=_mapping(source_policy_action),
    )
    step = build_wam_derived_observation_step(
        generated_at=generated,
        step_index=step_index,
        source_generated_frame_path=source_generated_frame_path,
        source_generated_video_path=source_generated_video_path,
        source_generated_multiview_frame_paths=source_generated_multiview_frame_paths,
        source_wam_rollout_id=source_wam_rollout_id,
        transition_id=transition_id,
        source_policy_action=source_policy_action,
        action_history=action_history,
        current_policy_observation=observation,
        object_index=resolved_object_index,
        eval_ready_task_grounding=resolved_grounding,
        skeleton_conditioning=skeleton_conditioning,
        camera_calibration=calibration,
        controller_limits=controller_limits,
        previous_steps=previous_steps,
        backend_kind=backend_kind,
        external_backend_result=backend_result,
        external_consistency=external_consistency,
        review_acceptance=review_acceptance,
        confidence_threshold=confidence_threshold,
    )
    adapter_report = adapt_policy_observation_for_declared_schema(
        policy_id=policy_id,
        declared_policy_observation_schema=declared_policy_observation_schema,
        base_policy_observation=_mapping(current_policy_observation),
        derived_step=step,
    )
    step["policy_adapter"] = {
        "policy_id": adapter_report.get("policy_id"),
        "declared_policy_observation_schema": adapter_report.get(
            "declared_policy_observation_schema"
        ),
        "fields_requested_by_policy": adapter_report.get("fields_requested_by_policy"),
        "fields_supplied_to_policy": adapter_report.get("fields_supplied_to_policy"),
        "fields_withheld_due_to_contract": adapter_report.get(
            "fields_withheld_due_to_contract"
        ),
        "adapter_status": adapter_report.get("adapter_status"),
        "safe_for_policy_requery": adapter_report.get("safe_for_policy_requery"),
    }
    steps = [*list(previous_steps or []), step]
    reports = [*list(previous_adapter_reports or []), adapter_report]
    artifacts = write_wam_derived_observation_artifacts(
        output_dir=output_dir,
        generated_at=generated,
        steps=steps,
        adapter_reports=reports,
        validation_set=validation_set,
        validation_set_path=validation_set_path,
    )
    return {
        "step_record": step,
        "policy_adapter_report": adapter_report,
        "adapted_policy_observation": adapter_report["adapted_policy_observation"],
        **artifacts,
    }


def summarize_wam_derived_observation_artifacts(
    artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = _mapping(artifacts)
    manifest = _mapping(value.get("manifest"))
    checks = _mapping(value.get("checks"))
    adapter = _mapping(value.get("adapter_report"))
    validation = _mapping(value.get("validation_report"))
    false_success = _mapping(value.get("false_success_reduction_metrics"))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "available": bool(manifest),
        "status": manifest.get("status"),
        "artifact_paths": _mapping(manifest.get("artifact_paths")),
        "step_count": manifest.get("step_count"),
        "early_termination_recommended": manifest.get("early_termination_recommended"),
        "success_scoring_blocked": manifest.get("success_scoring_blocked"),
        "policy_adapter_status": _mapping(
            adapter.get("latest_policy_adapter_report")
        ).get("adapter_status"),
        "policy_adapter_safe_for_policy_requery": _mapping(
            adapter.get("latest_policy_adapter_report")
        ).get("safe_for_policy_requery"),
        "forward_inverse_consistency_proven": checks.get("forward_inverse_consistency_proven"),
        "validation_status": validation.get("status") or manifest.get("validation_status"),
        "false_success_reduction_status": false_success.get("status")
        or manifest.get("false_success_reduction_status"),
        "false_success_reduction_rate": false_success.get("false_success_reduction_rate"),
        "claim_boundary": _claim_boundary(),
    }
