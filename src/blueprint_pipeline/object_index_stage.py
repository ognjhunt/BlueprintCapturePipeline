"""Build a canonical object index from a raw capture bundle."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .capture_enrichment_llm import build_capture_enrichment_runner
from .capture_bridge import CaptureDescriptor
from .common import join_gs_uri, read_json_any, resolve_gs_uri_to_path, utc_now_iso, write_json
from .eval_ready_task_grounding import derive_task_aware_detection_prompts
from .ios_manifest import IOSManifest, load_object_index, load_raw_manifest
from .local_capture import resolve_local_capture_context
from .world_model_policy import WorldModelPolicy, build_output_linkage, build_provenance_record


_DEFAULT_KEYFRAME_COUNT = 12
_DEFAULT_BOX_EXTENTS = [0.45, 0.45, 0.45]
_MIN_BOX_EXTENT = 0.02
_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0p2x0AAAAASUVORK5CYII="
)
_STRUCTURAL_LABELS = {"wall", "floor", "ceiling", "window", "stairs"}
_PROMPT_BANKS: Dict[str, List[str]] = {
    "default": [
        "door",
        "cabinet",
        "drawer",
        "shelf",
        "table",
        "chair",
        "desk",
        "box",
        "container",
        "bin",
        "tote",
        "basket",
        "monitor",
        "tv",
        "lamp",
        "appliance",
    ],
    "warehouse": [
        "rack",
        "shelf",
        "pallet",
        "tote",
        "bin",
        "box",
        "package",
        "cart",
        "forklift",
        "door",
        "container",
    ],
    "kitchen": [
        "cabinet",
        "drawer",
        "refrigerator",
        "fridge",
        "microwave",
        "oven",
        "dishwasher",
        "sink",
        "mug",
        "bottle",
        "bowl",
        "plate",
        "pot",
        "pan",
        "door",
    ],
    "bedroom": [
        "bed",
        "nightstand",
        "dresser",
        "wardrobe",
        "closet",
        "door",
        "desk",
        "chair",
        "lamp",
        "mirror",
        "box",
        "basket",
        "hamper",
    ],
    "office": [
        "desk",
        "chair",
        "monitor",
        "keyboard",
        "mouse",
        "laptop",
        "printer",
        "cabinet",
        "drawer",
        "shelf",
        "box",
        "mug",
    ],
}
_TASK_KEYWORD_PROMPTS: Dict[str, List[str]] = {
    "open": ["door", "drawer", "cabinet", "refrigerator", "fridge"],
    "close": ["door", "drawer", "cabinet", "refrigerator", "fridge"],
    "drawer": ["drawer"],
    "cabinet": ["cabinet"],
    "fridge": ["fridge", "refrigerator"],
    "refrigerator": ["refrigerator", "fridge"],
    "shelf": ["shelf", "rack"],
    "rack": ["rack", "shelf"],
    "tote": ["tote", "bin", "container"],
    "bin": ["bin", "container", "tote"],
    "box": ["box", "package", "container"],
    "package": ["package", "box", "container"],
    "desk": ["desk", "chair", "monitor", "keyboard"],
    "workspace": ["desk", "chair", "monitor", "keyboard", "box"],
    "inventory": ["box", "bin", "container", "shelf", "rack"],
    "organize": ["box", "bin", "container", "drawer", "cabinet", "shelf"],
    "kitchen": ["cabinet", "drawer", "fridge", "sink", "mug"],
    "media": ["tv", "monitor", "shelf", "cabinet"],
}
_LABEL_BUCKETS: Dict[str, Tuple[str, ...]] = {
    "door": ("door",),
    "drawer": ("drawer",),
    "cabinet": ("cabinet", "cupboard", "closet", "wardrobe"),
    "fridge": ("fridge", "refrigerator"),
    "container": ("box", "container", "bin", "tote", "basket", "package", "crate"),
    "desk": ("desk", "table", "workstation"),
    "chair": ("chair", "stool"),
    "monitor": ("monitor", "tv", "screen"),
    "shelf": ("shelf", "rack"),
}
_DEFAULT_EXTENTS_BY_BUCKET: Dict[str, List[float]] = {
    "door": [0.9, 0.12, 2.0],
    "drawer": [0.55, 0.45, 0.22],
    "cabinet": [0.8, 0.45, 0.9],
    "fridge": [0.9, 0.9, 1.9],
    "container": [0.5, 0.4, 0.35],
    "desk": [1.2, 0.7, 0.75],
    "chair": [0.6, 0.6, 0.95],
    "monitor": [0.6, 0.15, 0.4],
    "shelf": [1.0, 0.4, 1.8],
}


@dataclass(frozen=True)
class _Keyframe:
    frame_index: int
    timestamp: float
    image_width: int
    image_height: int
    image_path: Path
    intrinsics: List[float]
    camera_translation: List[float]
    motion_score: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(*values: Any) -> List[str]:
    out: List[str] = []
    for value in values:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple, set)):
            candidates = [str(item) for item in value]
        else:
            candidates = []
        for candidate in candidates:
            text = candidate.strip()
            if text and text not in out:
                out.append(text)
    return out


def _slug(text: str) -> str:
    out = []
    for char in text.strip().lower():
        out.append(char if char.isalnum() else "_")
    normalized = "".join(out).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "object"


def _optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                out.append(dict(payload))
    return out


def _resolve_video_path(context, manifest: IOSManifest) -> Optional[Path]:
    for rel in ("walkthrough.mov", "walkthrough.mp4", "recording.mov", "recording.mp4"):
        path = context.raw_root / rel
        if path.is_file():
            return path
    video_uri = str(manifest.video_uri or "").strip()
    if video_uri.startswith("gs://"):
        try:
            return resolve_gs_uri_to_path(video_uri, context.storage_root)
        except Exception:
            return None
    if video_uri:
        candidate = context.raw_root / video_uri
        if candidate.is_file():
            return candidate
    return None


def _translation_from_matrix(value: Any) -> List[float]:
    if isinstance(value, list) and len(value) == 16:
        return [_safe_float(value[12]), _safe_float(value[13]), _safe_float(value[14])]
    if isinstance(value, list) and len(value) == 4 and all(isinstance(row, list) for row in value):
        return [
            _safe_float(value[0][3] if len(value[0]) > 3 else 0.0),
            _safe_float(value[1][3] if len(value[1]) > 3 else 0.0),
            _safe_float(value[2][3] if len(value[2]) > 3 else 0.0),
        ]
    return [0.0, 0.0, 0.0]


def _nearest_motion_score(timestamp: float, motion_entries: Sequence[Mapping[str, Any]]) -> float:
    if not motion_entries:
        return 0.0
    nearest = min(motion_entries, key=lambda item: abs(_safe_float(item.get("timestamp"), timestamp) - timestamp))
    rotation = nearest.get("rotationRate") if isinstance(nearest.get("rotationRate"), Mapping) else {}
    user_acc = nearest.get("userAcceleration") if isinstance(nearest.get("userAcceleration"), Mapping) else {}
    rot_mag = math.sqrt(sum(_safe_float(rotation.get(axis), 0.0) ** 2 for axis in ("x", "y", "z")))
    acc_mag = math.sqrt(sum(_safe_float(user_acc.get(axis), 0.0) ** 2 for axis in ("x", "y", "z")))
    return round(rot_mag + acc_mag, 6)


def _sample_keyframes(
    *,
    context,
    max_keyframes: int,
    artifact_dir: Path,
) -> List[_Keyframe]:
    frames_entries = _jsonl(context.raw_root / "arkit" / "frames.jsonl")
    poses_entries = _jsonl(context.raw_root / "arkit" / "poses.jsonl")
    motion_entries = _jsonl(context.raw_root / "motion.jsonl")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if frames_entries:
        poses_by_index = {
            _safe_int(entry.get("frameIndex"), -1): entry
            for entry in poses_entries
            if _safe_int(entry.get("frameIndex"), -1) >= 0
        }
        total = len(frames_entries)
        count = min(max(1, max_keyframes), total)
        window = max(1, total // count)
        selected: List[Dict[str, Any]] = []
        for start in range(0, total, window):
            chunk = frames_entries[start : min(total, start + window)]
            if not chunk:  # pragma: no cover - window/range construction keeps slices non-empty.
                continue
            best = min(
                chunk,
                key=lambda entry: _nearest_motion_score(_safe_float(entry.get("timestamp"), 0.0), motion_entries),
            )
            selected.append(best)
            if len(selected) >= count:
                break
        keyframes: List[_Keyframe] = []
        first_ts = _safe_float(selected[0].get("timestamp"), 0.0) if selected else 0.0
        for entry in selected:
            frame_index = _safe_int(entry.get("frameIndex"), len(keyframes))
            ts = _safe_float(entry.get("timestamp"), first_ts)
            resolution = entry.get("imageResolution") if isinstance(entry.get("imageResolution"), list) else [1920, 1080]
            width = _safe_int(resolution[0] if len(resolution) > 0 else 1920, 1920)
            height = _safe_int(resolution[1] if len(resolution) > 1 else 1080, 1080)
            pose = poses_by_index.get(frame_index, {})
            keyframes.append(
                _Keyframe(
                    frame_index=frame_index,
                    timestamp=ts - first_ts,
                    image_width=width,
                    image_height=height,
                    image_path=artifact_dir / f"frame_{frame_index:06d}.png",
                    intrinsics=[_safe_float(value) for value in (entry.get("intrinsics") or [])],
                    camera_translation=_translation_from_matrix(
                        pose.get("transform") or pose.get("T_world_camera") or entry.get("cameraTransform")
                    ),
                    motion_score=_nearest_motion_score(ts, motion_entries),
                )
            )
        return keyframes

    video_path = _resolve_video_path(context, load_raw_manifest(context.raw_prefix_uri, gcs_root=context.storage_root))
    if video_path is None or not video_path.is_file():
        return []
    duration = _ffprobe_duration_seconds(video_path)
    count = max(1, max_keyframes)
    if duration <= 0.0:
        timestamps = [0.0]
    else:
        step = duration / float(count + 1)
        timestamps = [step * float(idx + 1) for idx in range(count)]
    return [
        _Keyframe(
            frame_index=idx,
            timestamp=timestamp,
            image_width=1920,
            image_height=1080,
            image_path=artifact_dir / f"frame_{idx:06d}.png",
            intrinsics=[],
            camera_translation=[0.0, 0.0, 0.0],
            motion_score=0.0,
        )
        for idx, timestamp in enumerate(timestamps)
    ]


def _ffprobe_duration_seconds(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return 0.0
    if result.returncode != 0:
        return 0.0
    return _safe_float(result.stdout.strip(), 0.0)


def _ensure_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_ONE_PIXEL_PNG)


def _extract_keyframe_images(video_path: Optional[Path], keyframes: Sequence[_Keyframe]) -> None:
    for keyframe in keyframes:
        if keyframe.image_path.is_file():
            continue
        if video_path is None or not video_path.is_file():
            _ensure_png(keyframe.image_path)
            continue
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{max(0.0, keyframe.timestamp):.6f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(keyframe.image_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            _ensure_png(keyframe.image_path)
            continue
        if proc.returncode != 0 or not keyframe.image_path.is_file():
            _ensure_png(keyframe.image_path)


def _infer_environment(descriptor: CaptureDescriptor, manifest: IOSManifest) -> str:
    for candidate in (
        descriptor.environment_type_hint,
        *descriptor.swap_focus,
        manifest.intended_space_type,
        descriptor.metadata.get("task_zone") if isinstance(descriptor.metadata.get("task_zone"), str) else "",
    ):
        text = str(candidate or "").strip().lower()
        if not text:
            continue
        if "warehouse" in text:
            return "warehouse"
        if "kitchen" in text:
            return "kitchen"
        if "bedroom" in text:
            return "bedroom"
        if "office" in text or "workspace" in text:
            return "office"
    return "default"


def _build_prompt_bank(descriptor: CaptureDescriptor, intake: Mapping[str, Any], capture_context: Mapping[str, Any], environment: str) -> Dict[str, List[str]]:
    broad = list(_PROMPT_BANKS.get(environment, _PROMPT_BANKS["default"]))
    text_fields = _string_list(
        intake.get("workflowName"),
        intake.get("zone"),
        descriptor.metadata.get("task_statement"),
        descriptor.metadata.get("workflow_context"),
        descriptor.metadata.get("owner"),
        capture_context.get("captureModality"),
        capture_context.get("captureSource"),
        intake.get("taskSteps"),
    )
    joined = " ".join(text_fields).lower()
    task_specific: List[str] = []
    target_label = str(
        descriptor.metadata.get("target_label")
        or descriptor.metadata.get("target_object")
        or intake.get("targetLabel")
        or intake.get("target_label")
        or ""
    )
    explicit_task_text = " ".join(_string_list(intake.get("taskSteps"), intake.get("task_steps"))).lower()
    fallback_task_text = " ".join(
        _string_list(
            descriptor.metadata.get("task_statement"),
            descriptor.metadata.get("workflow_context"),
            intake.get("workflowName"),
        )
    ).lower()
    task_aware = derive_task_aware_detection_prompts(
        task_text=explicit_task_text or fallback_task_text or joined,
        target_label=target_label,
    )
    for prompt in task_aware:
        if prompt not in task_specific:
            task_specific.append(prompt)
    for token, prompts in _TASK_KEYWORD_PROMPTS.items():
        if token in joined:
            for prompt in prompts:
                if prompt not in task_specific:
                    task_specific.append(prompt)
    all_prompts = []
    for value in broad + task_specific:
        if value not in all_prompts:
            all_prompts.append(value)
    return {"broad": broad, "task_specific": task_specific, "all": all_prompts}


def _maybe_expand_prompt_bank(
    *,
    runner,
    descriptor: CaptureDescriptor,
    intake: Mapping[str, Any],
    capture_context: Mapping[str, Any],
    prompt_bank: Mapping[str, List[str]],
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, Any]]]:
    if runner is None:
        return dict(prompt_bank), None
    payload = {
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "task_statement": descriptor.metadata.get("task_statement"),
        "workflow_context": descriptor.metadata.get("workflow_context"),
        "task_zone": descriptor.metadata.get("task_zone"),
        "intake_packet": dict(intake),
        "capture_context": dict(capture_context),
        "existing_prompt_bank": dict(prompt_bank),
    }
    response = runner("prompt_bank_expander", payload)
    if not isinstance(response, Mapping):
        return dict(prompt_bank), None
    expanded = {key: list(value) for key, value in prompt_bank.items()}
    additional = [str(item).strip() for item in response.get("additional_prompts", []) if str(item).strip()] if isinstance(response.get("additional_prompts"), list) else []
    for prompt in additional:
        if prompt not in expanded["task_specific"]:
            expanded["task_specific"].append(prompt)
        if prompt not in expanded["all"]:
            expanded["all"].append(prompt)
    return expanded, dict(response)


def _command_from_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if value:
        return value
    repo_root = Path(__file__).resolve().parents[2]
    defaults = {
        "OBJECT_INDEX_YOLO_WORLD_COMMAND": repo_root / "scripts" / "object_index_yolo_world_runner.py",
        "OBJECT_INDEX_GROUNDING_DINO_COMMAND": repo_root / "scripts" / "object_index_grounding_dino_runner.py",
        "OBJECT_INDEX_SAM3_COMMAND": repo_root / "scripts" / "object_index_sam3_runner.py",
    }
    script_path = defaults.get(name)
    if script_path is not None and script_path.is_file():
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} {{INPUT_JSON}} {{OUTPUT_JSON}}"
    return ""


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _default_sam3_weights_path() -> Path:
    return Path(os.getenv("SAM3_WEIGHTS_PATH") or "/opt/sam3_weights/sam3.pt")


def _backend_runtime_requirements(backend_name: str) -> Dict[str, Any]:
    support_level = "required"
    required_modules: List[str]
    required_paths: List[str] = []
    if backend_name in {"yolo_world", "grounding_dino"}:
        required_modules = ["torch", "ultralytics"]
    elif backend_name == "sam3":
        support_level = "optional"
        required_modules = ["torch", "sam3"]
        required_paths = [str(_default_sam3_weights_path())]
    else:
        required_modules = []
    missing_modules = [name for name in required_modules if not _module_available(name)]
    missing_paths = [path for path in required_paths if not Path(path).is_file()]
    return {
        "support_level": support_level,
        "required_modules": required_modules,
        "missing_modules": missing_modules,
        "required_paths": required_paths,
        "missing_paths": missing_paths,
    }


def _backend_preflight_status(*, backend_name: str, command_template: str) -> Dict[str, Any]:
    requirements = _backend_runtime_requirements(backend_name)
    if not command_template:
        return {
            "configured": False,
            "status": "missing",
            "reason": "command_not_configured",
            **requirements,
        }
    try:
        rendered = (
            command_template.replace("{INPUT_JSON}", "/tmp/input.json")
            .replace("{OUTPUT_JSON}", "/tmp/output.json")
            .replace("{OUTPUT_DIR}", "/tmp")
        )
        command = shlex.split(rendered)
    except ValueError as exc:
        return {
            "configured": True,
            "status": "invalid",
            "reason": f"invalid_command_template:{exc}",
            **requirements,
        }
    executable = command[0] if command else ""
    executable_path = shutil.which(executable) or executable
    status = "ready" if executable else "invalid"
    reason = ""
    if not executable:
        reason = "empty_command"
    elif executable_path and str(executable_path).startswith("/"):
        if not Path(str(executable_path)).exists():
            status = "missing"
            reason = f"missing_executable:{executable_path}"
    if not reason and requirements["missing_modules"]:
        if requirements["support_level"] == "optional":
            status = "optional_unavailable"
        else:
            status = "runtime_missing"
        reason = "missing_modules:" + ",".join(requirements["missing_modules"])
    if not reason and requirements["missing_paths"]:
        if requirements["support_level"] == "optional":
            status = "optional_unavailable"
        else:
            status = "runtime_missing"
        reason = "missing_paths:" + ",".join(requirements["missing_paths"])
    return {
        "configured": True,
        "status": status,
        "reason": reason,
        "command": command,
        **requirements,
    }


def _payload_detection_count(payload: Any) -> int:
    if isinstance(payload, Mapping):
        for key in ("detections", "items", "objects"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _run_backend_command(
    *,
    backend_name: str,
    command_template: str,
    input_payload: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    if not command_template:
        return {"status": "skipped", "backend": backend_name, "reason": "command_not_configured"}
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"{backend_name}_input.json"
    output_path = output_dir / f"{backend_name}_output.json"
    write_json(input_path, input_payload)

    rendered = command_template
    substitutions = {
        "INPUT_JSON": str(input_path),
        "OUTPUT_JSON": str(output_path),
        "OUTPUT_DIR": str(output_dir),
    }
    for key, value in substitutions.items():
        rendered = rendered.replace("{" + key + "}", value)
    try:
        command = shlex.split(rendered)
    except ValueError as exc:
        return {"status": "failed", "backend": backend_name, "reason": f"invalid_command_template:{exc}"}
    if not command:
        return {"status": "failed", "backend": backend_name, "reason": "empty_command"}

    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError as exc:
        return {
            "status": "failed",
            "backend": backend_name,
            "reason": f"failed_to_launch:{exc}",
            "command": command,
        }
    report: Dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "backend": backend_name,
        "return_code": proc.returncode,
        "command": command,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    payload: Any = {}
    if output_path.is_file():
        try:
            payload = read_json_any(output_path)
        except Exception as exc:
            report["status"] = "failed"
            report["reason"] = f"invalid_output_json:{exc}"
    elif proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except Exception:
            payload = {}
    def _tail_reason(raw: Any) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    if isinstance(payload, Mapping):
        report["payload"] = dict(payload)
        backend_status = str(payload.get("backend_status") or "").strip().lower()
        if backend_status in {"ok", "skipped", "failed"}:
            report["status"] = backend_status
        backend_reason = str(payload.get("reason") or "").strip()
        if backend_reason:
            report["reason"] = backend_reason
        elif report["status"] == "failed":
            derived_reason = (
                _tail_reason(payload.get("stderr_tail"))
                or _tail_reason(payload.get("stdout_tail"))
                or _tail_reason(report.get("stderr_tail"))
                or _tail_reason(report.get("stdout_tail"))
            )
            if derived_reason:
                report["reason"] = derived_reason
        if report["status"] == "ok" and _payload_detection_count(payload) == 0:
            report["status"] = "empty"
            report.setdefault("reason", "no_detections")
    elif isinstance(payload, list):
        report["payload"] = {"detections": payload}
        if report["status"] == "ok" and not payload:
            report["status"] = "empty"
            report.setdefault("reason", "no_detections")
    return report


def _backend_reason_indicates_runtime_missing(reason: str) -> bool:
    lowered = str(reason or "").strip().lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "missing",
            "not_installed",
            "unavailable",
            "weights_missing",
            "failed_to_launch",
            "ultralytics_missing",
            "torch_not_installed",
        )
    )


def _bbox_xyxy(entry: Mapping[str, Any], width: int, height: int) -> Optional[List[float]]:
    for key in ("bbox_xyxy", "xyxy", "bbox", "box"):
        value = entry.get(key)
        if isinstance(value, list) and len(value) >= 4:
            coords = [float(value[idx]) for idx in range(4)]
            return [
                max(0.0, min(coords[0], float(width))),
                max(0.0, min(coords[1], float(height))),
                max(0.0, min(coords[2], float(width))),
                max(0.0, min(coords[3], float(height))),
            ]
    return None


def _normalize_detection_payload(
    *,
    backend_name: str,
    payload: Mapping[str, Any],
    keyframes_by_index: Mapping[int, _Keyframe],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    detections_raw = payload.get("detections")
    if not isinstance(detections_raw, list):
        detections_raw = payload.get("items")
    detections: List[Dict[str, Any]] = []
    if isinstance(detections_raw, list):
        for item in detections_raw:
            if not isinstance(item, Mapping):
                continue
            frame_index = _safe_int(item.get("frame_index", item.get("frameIndex", -1)), -1)
            keyframe = keyframes_by_index.get(frame_index)
            if keyframe is None:
                continue
            label = str(item.get("label") or item.get("name") or item.get("class_name") or "").strip()
            if not label:
                continue
            bbox = _bbox_xyxy(item, keyframe.image_width, keyframe.image_height)
            if bbox is None:
                continue
            detections.append(
                {
                    "frame_index": frame_index,
                    "timestamp": keyframe.timestamp,
                    "label": label,
                    "score": max(
                        _safe_float(item.get("score"), 0.0),
                        _safe_float(item.get("confidence"), 0.0),
                        _safe_float(item.get("mean_confidence"), 0.0),
                    ),
                    "bbox_xyxy": bbox,
                    "source": backend_name,
                    "source_prompt": str(item.get("source_prompt") or item.get("prompt") or label).strip(),
                    "mask_path": str(item.get("mask_path") or "").strip() or None,
                    "crop_path": str(item.get("crop_path") or "").strip() or None,
                    "world_center": item.get("world_center") if isinstance(item.get("world_center"), list) else None,
                    "world_extents": item.get("world_extents") if isinstance(item.get("world_extents"), list) else None,
                    "orientation_quaternion": item.get("orientation_quaternion") if isinstance(item.get("orientation_quaternion"), list) else None,
                }
            )
    manip = payload.get("manipulation_candidates")
    artic = payload.get("articulation_hints")
    tasks = payload.get("tasks")
    return (
        detections,
        [dict(item) for item in manip if isinstance(item, Mapping)] if isinstance(manip, list) else [],
        [dict(item) for item in artic if isinstance(item, Mapping)] if isinstance(artic, list) else [],
        [dict(item) for item in tasks if isinstance(item, Mapping)] if isinstance(tasks, list) else [],
    )


def _normalize_existing_objects(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        return []
    objects: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or item.get("name") or "").strip()
        if not label:
            continue
        bbox = item.get("boundingBox") if isinstance(item.get("boundingBox"), Mapping) else {}
        center = bbox.get("center") if isinstance(bbox.get("center"), list) else [float(index), 0.0, 0.0]
        extents = bbox.get("extents") if isinstance(bbox.get("extents"), list) else list(_DEFAULT_BOX_EXTENTS)
        objects.append(
            {
                "id": str(item.get("id") or item.get("object_id") or f"obj_{index+1:04d}"),
                "label": label,
                "boundingBox": {
                    "center": [_safe_float(center[idx] if idx < len(center) else 0.0, 0.0) for idx in range(3)],
                    "extents": [max(_MIN_BOX_EXTENT, _safe_float(extents[idx] if idx < len(extents) else 0.25, 0.25)) for idx in range(3)],
                    "axes": bbox.get("axes") if isinstance(bbox.get("axes"), list) else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "orientationQuaternion": bbox.get("orientationQuaternion") if isinstance(bbox.get("orientationQuaternion"), list) else [1.0, 0.0, 0.0, 0.0],
                },
                "mean_confidence": _safe_float(item.get("mean_confidence"), _safe_float(item.get("confidence"), 0.0)),
                "n_total_detections": _safe_int(item.get("n_total_detections"), 1),
                "n_frame_detections": _safe_int(item.get("n_frame_detections"), 1),
                "reference_crop": str(item.get("reference_crop") or "").strip(),
                "all_crops": [str(value).strip() for value in item.get("all_crops", []) if str(value).strip()] if isinstance(item.get("all_crops"), list) else [],
                "task_relevance": dict(item.get("task_relevance")) if isinstance(item.get("task_relevance"), Mapping) else {},
                "articulation_hints": dict(item.get("articulation_hints")) if isinstance(item.get("articulation_hints"), Mapping) else {},
                "evidence_frames": list(item.get("evidence_frames")) if isinstance(item.get("evidence_frames"), list) else [],
                "source_prompts": list(item.get("source_prompts")) if isinstance(item.get("source_prompts"), list) else [],
                "provenance": dict(item.get("provenance")) if isinstance(item.get("provenance"), Mapping) else {},
                "mean_box_px": dict(item.get("mean_box_px")) if isinstance(item.get("mean_box_px"), Mapping) else {},
            }
        )
    return objects


def _iou2d(a: Sequence[float], b: Sequence[float]) -> float:
    inter_left = max(float(a[0]), float(b[0]))
    inter_top = max(float(a[1]), float(b[1]))
    inter_right = min(float(a[2]), float(b[2]))
    inter_bottom = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    area_b = max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return inter / max(1.0, area_a + area_b - inter)


def _dedupe_same_frame(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        duplicate = False
        for existing in kept:
            if existing["frame_index"] != detection["frame_index"]:
                continue
            if str(existing["label"]).lower() != str(detection["label"]).lower():
                continue
            if _iou2d(existing["bbox_xyxy"], detection["bbox_xyxy"]) >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    kept.sort(key=lambda item: (item["frame_index"], str(item["label"])))
    return kept


def _center_from_bbox(box: Sequence[float], width: int, height: int) -> List[float]:
    cx = ((float(box[0]) + float(box[2])) * 0.5) / float(max(1, width))
    cy = ((float(box[1]) + float(box[3])) * 0.5) / float(max(1, height))
    return [cx, cy]


def _box_area(box: Sequence[float]) -> float:
    return max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))


def _label_bucket(label: str) -> str:
    lowered = label.strip().lower()
    for bucket, tokens in _LABEL_BUCKETS.items():
        if any(token in lowered for token in tokens):
            return bucket
    return lowered or "object"


def _cluster_detections(detections: List[Dict[str, Any]], keyframes_by_index: Mapping[int, _Keyframe]) -> List[List[Dict[str, Any]]]:
    clusters: List[List[Dict[str, Any]]] = []
    for detection in sorted(detections, key=lambda item: (str(item["label"]).lower(), item["frame_index"])):
        placed = False
        det_label = str(detection["label"]).lower()
        keyframe = keyframes_by_index.get(int(detection["frame_index"]))
        if keyframe is None:
            continue
        det_center = _center_from_bbox(detection["bbox_xyxy"], keyframe.image_width, keyframe.image_height)
        det_area = _box_area(detection["bbox_xyxy"])
        for cluster in clusters:
            rep = cluster[0]
            rep_frame = keyframes_by_index.get(int(rep["frame_index"]))
            if rep_frame is None or str(rep["label"]).lower() != det_label:
                continue
            rep_center = _center_from_bbox(rep["bbox_xyxy"], rep_frame.image_width, rep_frame.image_height)
            center_distance = math.sqrt(sum((det_center[idx] - rep_center[idx]) ** 2 for idx in range(2)))
            area_ratio = min(det_area, _box_area(rep["bbox_xyxy"])) / max(det_area, _box_area(rep["bbox_xyxy"]))
            if detection.get("world_center") and rep.get("world_center"):
                world_distance = math.sqrt(
                    sum(
                        (_safe_float(detection["world_center"][idx]) - _safe_float(rep["world_center"][idx])) ** 2
                        for idx in range(3)
                    )
                )
                if world_distance <= 0.85:
                    cluster.append(detection)
                    placed = True
                    break
            if center_distance <= 0.18 and area_ratio >= 0.35:
                cluster.append(detection)
                placed = True
                break
        if not placed:
            clusters.append([detection])
    return clusters


def _copy_crop(frame_path: Path, crop_path: Path, bbox: Sequence[float]) -> None:
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(1, int(round(float(bbox[2]) - float(bbox[0]))))
    height = max(1, int(round(float(bbox[3]) - float(bbox[1]))))
    left = max(0, int(round(float(bbox[0]))))
    top = max(0, int(round(float(bbox[1]))))
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(frame_path),
                "-vf",
                f"crop={width}:{height}:{left}:{top}",
                str(crop_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        shutil.copyfile(frame_path, crop_path)
        return
    if proc.returncode != 0 or not crop_path.is_file():
        shutil.copyfile(frame_path, crop_path)


def _task_relevance(label: str, prompts: Sequence[str], descriptor: CaptureDescriptor) -> Dict[str, Any]:
    del prompts
    label_bucket = _label_bucket(label)
    text = " ".join(
        _string_list(
            descriptor.metadata.get("task_statement"),
            descriptor.metadata.get("workflow_context"),
            descriptor.metadata.get("task_zone", {}).get("label") if isinstance(descriptor.metadata.get("task_zone"), Mapping) else "",
        )
    ).lower()
    matched = []
    for token in _string_list(label, label_bucket):
        if token.lower() in text and token.lower() not in matched:
            matched.append(token.lower())
    score = 0.2 + 0.45 * float(bool(matched))
    if label_bucket in {"door", "drawer", "cabinet", "fridge"} and any(word in text for word in ("open", "close")):
        score += 0.2
    if label_bucket in {"container", "desk", "shelf"} and any(word in text for word in ("organize", "inventory", "move", "pick", "place")):
        score += 0.15
    return {"score": round(min(1.0, score), 4), "matched_terms": matched}


def _articulation_hints(label: str) -> Dict[str, Any]:
    bucket = _label_bucket(label)
    if bucket == "door":
        return {"interactive": True, "kind": "door", "confidence": 0.8}
    if bucket == "drawer":
        return {"interactive": True, "kind": "drawer", "confidence": 0.82}
    if bucket == "cabinet":
        return {"interactive": True, "kind": "cabinet", "confidence": 0.7}
    if bucket == "fridge":
        return {"interactive": True, "kind": "refrigerator_door", "confidence": 0.85}
    return {"interactive": False, "kind": "static", "confidence": 0.35}


def _synthesized_bbox(
    cluster: Sequence[Mapping[str, Any]],
    *,
    keyframes_by_index: Mapping[int, _Keyframe],
    cluster_index: int,
    label: str,
) -> Dict[str, Any]:
    provided_centers = [item.get("world_center") for item in cluster if isinstance(item.get("world_center"), list) and len(item.get("world_center")) >= 3]
    provided_extents = [item.get("world_extents") for item in cluster if isinstance(item.get("world_extents"), list) and len(item.get("world_extents")) >= 3]
    if provided_centers:
        center = [
            round(sum(_safe_float(value[idx]) for value in provided_centers) / float(len(provided_centers)), 6)
            for idx in range(3)
        ]
    else:
        centers = []
        for item in cluster:
            keyframe = keyframes_by_index.get(_safe_int(item.get("frame_index"), -1))
            if keyframe is None:
                continue
            bbox_center = _center_from_bbox(item["bbox_xyxy"], keyframe.image_width, keyframe.image_height)
            translation = keyframe.camera_translation
            if any(abs(value) > 1e-6 for value in translation):
                centers.append(
                    [
                        round(translation[0] + (bbox_center[0] - 0.5) * 2.0, 6),
                        round(translation[1] + (0.5 - bbox_center[1]) * 2.0, 6),
                        round(translation[2] + 1.5, 6),
                    ]
                )
            else:
                centers.append(
                    [
                        round((bbox_center[0] - 0.5) * 4.0, 6),
                        round((0.5 - bbox_center[1]) * 4.0, 6),
                        round(cluster_index * 0.35, 6),
                    ]
                )
        if centers:
            center = [
                round(sum(value[idx] for value in centers) / float(len(centers)), 6)
                for idx in range(3)
            ]
        else:
            center = [round(cluster_index * 0.35, 6), 0.0, 0.0]
    if provided_extents:
        extents = [
            round(max(_MIN_BOX_EXTENT, sum(_safe_float(value[idx]) for value in provided_extents) / float(len(provided_extents))), 6)
            for idx in range(3)
        ]
    else:
        bucket = _label_bucket(label)
        default_extents = _DEFAULT_EXTENTS_BY_BUCKET.get(bucket, _DEFAULT_BOX_EXTENTS)
        areas = [_box_area(item["bbox_xyxy"]) for item in cluster]
        area_scale = min(2.0, max(0.5, math.sqrt(sum(areas) / float(max(1, len(areas)))) / 180.0))
        extents = [round(max(_MIN_BOX_EXTENT, value * area_scale), 6) for value in default_extents]
    return {
        "center": center,
        "extents": extents,
        "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
    }


def _build_objects(
    *,
    clusters: Sequence[Sequence[Mapping[str, Any]]],
    keyframes_by_index: Mapping[int, _Keyframe],
    descriptor: CaptureDescriptor,
    raw_root: Path,
    crops_dir: Path,
) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    privacy_penalty = 0.1 if _string_list(descriptor.metadata.get("privacy_restrictions")) else 0.0
    for cluster_index, cluster in enumerate(clusters, start=1):
        if not cluster:
            continue
        representative = max(cluster, key=lambda item: float(item.get("score") or 0.0))
        label = str(representative["label"])
        object_id = f"{_slug(_label_bucket(label))}_{cluster_index:04d}"
        crop_paths: List[str] = []
        evidence_frames: List[int] = []
        prompts: List[str] = []
        areas: List[float] = []
        scores: List[float] = []
        for obs_index, observation in enumerate(sorted(cluster, key=lambda item: int(item["frame_index"]))):
            frame_index = _safe_int(observation.get("frame_index"), -1)
            keyframe = keyframes_by_index.get(frame_index)
            if keyframe is None:
                continue
            evidence_frames.append(frame_index)
            prompt = str(observation.get("source_prompt") or label).strip()
            if prompt and prompt not in prompts:
                prompts.append(prompt)
            scores.append(_safe_float(observation.get("score"), 0.0))
            areas.append(_box_area(observation["bbox_xyxy"]))
            crop_path = observation.get("crop_path")
            if isinstance(crop_path, str) and crop_path.strip():
                crop_file = Path(crop_path)
                if crop_file.is_file():
                    rel = crop_file.resolve().relative_to(raw_root.resolve()).as_posix() if raw_root.resolve() in crop_file.resolve().parents else crop_file.name
                    crop_paths.append(rel)
                    continue
            crop_file = crops_dir / f"{object_id}_f{frame_index:06d}_{obs_index:02d}.png"
            _copy_crop(keyframe.image_path, crop_file, observation["bbox_xyxy"])
            crop_paths.append(crop_file.relative_to(raw_root).as_posix())
        mean_area = sum(areas) / float(len(areas)) if areas else 0.0
        bbox = _synthesized_bbox(cluster, keyframes_by_index=keyframes_by_index, cluster_index=cluster_index, label=label)
        relevance = _task_relevance(label, prompts, descriptor)
        articulation = _articulation_hints(label)
        objects.append(
            {
                "id": object_id,
                "object_id": object_id,
                "label": label,
                "name": label,
                "boundingBox": bbox,
                "mean_confidence": round(max(0.0, (sum(scores) / float(len(scores) or 1)) - privacy_penalty), 4),
                "confidence": round(max(0.0, (max(scores) if scores else 0.0) - privacy_penalty), 4),
                "n_total_detections": len(cluster),
                "n_frame_detections": len(set(evidence_frames)),
                "reference_crop": crop_paths[0] if crop_paths else "",
                "all_crops": crop_paths,
                "evidence_frames": evidence_frames,
                "source_prompts": prompts,
                "task_relevance": relevance,
                "articulation_hints": articulation,
                "mean_box_px": {
                    "width": round(math.sqrt(mean_area), 4) if mean_area > 0.0 else 0.0,
                    "height": round(math.sqrt(mean_area), 4) if mean_area > 0.0 else 0.0,
                    "area": round(mean_area, 4),
                },
                "merged_object_ids": [],
                "provenance": build_provenance_record(
                    grounding_level="observed",
                    evidence_sources=[
                        *crop_paths,
                        *[str(keyframes_by_index.get(frame_index).image_path) for frame_index in evidence_frames if keyframes_by_index.get(frame_index) is not None],
                    ],
                    observation_coverage={
                        "n_total_detections": len(cluster),
                        "n_frame_detections": len(set(evidence_frames)),
                    },
                    confidence=round(max(0.0, (sum(scores) / float(len(scores) or 1)) - privacy_penalty), 4),
                    canonical_truth=True,
                    presentation_only=False,
                    extra={
                        "stage": "object_index_stage",
                        "sources": sorted({str(item.get("source") or "unknown") for item in cluster}),
                        "capture_id": descriptor.capture_id,
                        "privacy_penalty_applied": privacy_penalty > 0.0,
                    },
                ),
            }
        )
    return objects


def _apply_llm_task_relevance(
    *,
    runner,
    descriptor: CaptureDescriptor,
    objects: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if runner is None or not objects:
        return None
    payload = {
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "task_statement": descriptor.metadata.get("task_statement"),
        "workflow_context": descriptor.metadata.get("workflow_context"),
        "task_zone": descriptor.metadata.get("task_zone"),
        "objects": [
            {
                "object_id": str(item.get("id") or item.get("object_id") or ""),
                "label": str(item.get("label") or "object"),
                "task_relevance": dict(item.get("task_relevance") or {}),
                "articulation_hints": dict(item.get("articulation_hints") or {}),
                "source_prompts": list(item.get("source_prompts") or []),
            }
            for item in objects
        ],
    }
    response = runner("task_relevance_ranker", payload)
    if not isinstance(response, Mapping):
        return None
    scores = response.get("scores")
    if isinstance(scores, list):
        by_id = {str(item.get("object_id") or ""): item for item in scores if isinstance(item, Mapping)}
        for obj in objects:
            object_id = str(obj.get("id") or obj.get("object_id") or "")
            match = by_id.get(object_id)
            if not isinstance(match, Mapping):
                continue
            existing = obj.get("task_relevance") if isinstance(obj.get("task_relevance"), Mapping) else {}
            merged_terms = []
            existing_terms = list(existing.get("matched_terms", [])) if isinstance(existing.get("matched_terms"), list) else []
            match_terms = list(match.get("matched_terms", [])) if isinstance(match.get("matched_terms"), list) else []
            for term in existing_terms + match_terms:
                text = str(term).strip()
                if text and text not in merged_terms:
                    merged_terms.append(text)
            obj["task_relevance"] = {
                "score": round(max(_safe_float(existing.get("score"), 0.0), _safe_float(match.get("score"), 0.0)), 4),
                "matched_terms": merged_terms,
                "reason": str(match.get("reason") or existing.get("reason") or "").strip(),
            }
    return dict(response)


def _apply_llm_articulation_priors(*, runner, descriptor: CaptureDescriptor, objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if runner is None or not objects:
        return None
    payload = {
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "task_statement": descriptor.metadata.get("task_statement"),
        "workflow_context": descriptor.metadata.get("workflow_context"),
        "objects": [
            {
                "object_id": str(item.get("id") or item.get("object_id") or ""),
                "label": str(item.get("label") or "object"),
                "articulation_hints": dict(item.get("articulation_hints") or {}),
                "task_relevance": dict(item.get("task_relevance") or {}),
            }
            for item in objects
        ],
    }
    response = runner("articulation_prior_writer", payload)
    if not isinstance(response, Mapping):
        return None
    priors = response.get("articulation_priors")
    if isinstance(priors, list):
        by_id = {str(item.get("object_id") or item.get("instance_id") or ""): item for item in priors if isinstance(item, Mapping)}
        for obj in objects:
            object_id = str(obj.get("id") or obj.get("object_id") or "")
            prior = by_id.get(object_id)
            if not isinstance(prior, Mapping):
                continue
            existing = obj.get("articulation_hints") if isinstance(obj.get("articulation_hints"), Mapping) else {}
            obj["articulation_hints"] = {
                "interactive": bool(prior.get("interactive", existing.get("interactive"))),
                "kind": str(prior.get("kind") or existing.get("kind") or "static"),
                "confidence": round(max(_safe_float(existing.get("confidence"), 0.0), _safe_float(prior.get("confidence"), 0.0)), 4),
                "reason": str(prior.get("reason") or existing.get("reason") or "").strip(),
            }
    return dict(response)


def _llm_target_resolution(
    *,
    runner,
    descriptor: CaptureDescriptor,
    objects: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if runner is None:
        return None
    payload = {
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "task_statement": descriptor.metadata.get("task_statement"),
        "workflow_context": descriptor.metadata.get("workflow_context"),
        "task_zone": descriptor.metadata.get("task_zone"),
        "success_criteria": descriptor.metadata.get("success_criteria"),
        "known_blockers": descriptor.metadata.get("known_blockers"),
        "objects": [
            {
                "object_id": str(item.get("id") or item.get("object_id") or ""),
                "label": str(item.get("label") or "object"),
                "task_relevance": dict(item.get("task_relevance") or {}),
                "articulation_hints": dict(item.get("articulation_hints") or {}),
                "boundingBox": dict(item.get("boundingBox") or {}),
            }
            for item in objects
        ],
    }
    response = runner("workflow_target_resolver", payload)
    return dict(response) if isinstance(response, Mapping) else None


def _grounding_payload_from_objects(objects: Sequence[Mapping[str, Any]], descriptor: CaptureDescriptor, backend_report: Mapping[str, Any]) -> Dict[str, Any]:
    grounded_objects = []
    manipulation_candidates = []
    articulation_hints = []
    tasks = []
    task_text = " ".join(
        _string_list(
            descriptor.metadata.get("task_statement"),
            descriptor.metadata.get("workflow_context"),
            descriptor.metadata.get("task_zone", {}).get("label") if isinstance(descriptor.metadata.get("task_zone"), Mapping) else "",
        )
    ).lower()
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        grounded_objects.append(
            {
                "object_id": str(obj.get("id") or obj.get("object_id") or ""),
                "label": str(obj.get("label") or "object"),
                "confidence": _safe_float(obj.get("mean_confidence"), 0.0),
                "boundingBox": dict(obj.get("boundingBox") or {}),
                "source": "object_index_stage",
                "provenance": dict(obj.get("provenance") or {}) if isinstance(obj.get("provenance"), Mapping) else {},
            }
        )
        relevance = obj.get("task_relevance") if isinstance(obj.get("task_relevance"), Mapping) else {}
        rel_score = _safe_float(relevance.get("score"), 0.0)
        label_bucket = _label_bucket(str(obj.get("label") or ""))
        if rel_score >= 0.45 and label_bucket not in _STRUCTURAL_LABELS:
            manipulation_candidates.append(
                {
                    "instance_id": str(obj.get("id") or obj.get("object_id") or ""),
                    "label": str(obj.get("label") or "object"),
                    "confidence": rel_score,
                    "reason": "task_relevance",
                    "boundingBox": dict(obj.get("boundingBox") or {}),
                }
            )
        art = obj.get("articulation_hints") if isinstance(obj.get("articulation_hints"), Mapping) else {}
        if art.get("interactive"):
            articulation_hints.append(
                {
                    "instance_id": str(obj.get("id") or obj.get("object_id") or ""),
                    "label": str(obj.get("label") or "object"),
                    "confidence": _safe_float(art.get("confidence"), 0.0),
                    "reason": str(art.get("kind") or "interactive_object"),
                    "boundingBox": dict(obj.get("boundingBox") or {}),
                }
            )
    if articulation_hints and any(token in task_text for token in ("open", "close")):
        tasks.append(
            {
                "task_id": "open_close_primary",
                "target_object_ids": [str(item.get("instance_id") or "") for item in articulation_hints[:2] if str(item.get("instance_id") or "").strip()],
            }
        )
    return {
        "backend": "object_index_stage",
        "backend_status": "ok" if grounded_objects else "empty",
        "backend_report": dict(backend_report),
        "grounded_objects": grounded_objects,
        "manipulation_candidates": manipulation_candidates,
        "articulation_hints": articulation_hints,
        "tasks": tasks,
    }


def _write_descriptor_updates(descriptor_path: Path, descriptor: CaptureDescriptor, object_index_uri: str) -> None:
    payload = descriptor.to_dict()
    payload["object_index_uri"] = object_index_uri
    write_json(descriptor_path, payload)


def _write_manifest_updates(manifest_path: Path) -> None:
    payload = _optional_json(manifest_path)
    if not payload:
        return
    payload["object_index_uri"] = "object_index.json"
    write_json(manifest_path, payload)


def _canonicalize_legacy_index(*, context, descriptor: CaptureDescriptor) -> Optional[Dict[str, Any]]:
    legacy_path = context.raw_root / "arkit" / "objects" / "index.json"
    target_path = context.raw_root / "object_index.json"
    if target_path.is_file():
        loaded = load_object_index(join_gs_uri(context.raw_prefix_uri, "object_index.json"), gcs_root=context.storage_root)
        report = _optional_json(context.raw_root / "object_index_build_report.json")
        if not _existing_index_is_reusable(loaded=loaded, report=report):
            return None
        object_index_uri = join_gs_uri(context.raw_prefix_uri, "object_index.json")
        _write_descriptor_updates(context.descriptor_path, descriptor, object_index_uri)
        _write_manifest_updates(context.raw_root / "manifest.json")
        return {
            "schema_version": "v1",
            "status": "reused",
            "object_index_uri": object_index_uri,
            "manifest_path": str(target_path),
            "report_path": str(context.raw_root / "object_index_build_report.json"),
            "object_count": len(loaded),
            "grounding_payload": _grounding_payload_from_objects(loaded, descriptor, {"status": "reused"}),
        }
    if not legacy_path.is_file():
        return None
    loaded = load_object_index(join_gs_uri(context.raw_prefix_uri, "arkit/objects/index.json"), gcs_root=context.storage_root)
    write_json(target_path, {"objects": [dict(item) for item in loaded]})
    report_path = context.raw_root / "object_index_build_report.json"
    write_json(
        report_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "canonicalized_legacy",
            "legacy_path": str(legacy_path),
            "target_path": str(target_path),
            "object_count": len(loaded),
        },
    )
    object_index_uri = join_gs_uri(context.raw_prefix_uri, "object_index.json")
    _write_descriptor_updates(context.descriptor_path, descriptor, object_index_uri)
    _write_manifest_updates(context.raw_root / "manifest.json")
    return {
        "schema_version": "v1",
        "status": "canonicalized_legacy",
        "object_index_uri": object_index_uri,
        "manifest_path": str(target_path),
        "report_path": str(report_path),
        "object_count": len(loaded),
        "grounding_payload": _grounding_payload_from_objects(loaded, descriptor, {"status": "canonicalized_legacy"}),
    }


def _existing_index_is_reusable(
    *,
    loaded: Sequence[Mapping[str, Any]],
    report: Optional[Mapping[str, Any]],
) -> bool:
    if loaded:
        return True
    if not isinstance(report, Mapping):
        return False

    try:
        report_object_count = int(report.get("object_count"))
    except (TypeError, ValueError):
        return False
    if report_object_count != len(loaded):
        return False

    empty_index_cause = str(report.get("empty_index_cause") or "").strip().lower()
    if empty_index_cause in {"runtime_missing", "backend_skipped"}:
        return False

    runtime_preflight = report.get("runtime_preflight")
    backends = runtime_preflight.get("backends") if isinstance(runtime_preflight, Mapping) else {}
    if isinstance(backends, Mapping):
        for entry in backends.values():
            if not isinstance(entry, Mapping):
                continue
            support_level = str(entry.get("support_level") or "required").strip().lower() or "required"
            status = str(entry.get("status") or "").strip().lower()
            if support_level == "required" and status in {"runtime_missing", "optional_unavailable"}:
                return False

    return True


def run_object_index_stage(
    *,
    capture_root: str | Path,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    policy = WorldModelPolicy.from_env()
    context = resolve_local_capture_context(capture_root)
    manifest = load_raw_manifest(context.raw_prefix_uri, gcs_root=context.storage_root)
    descriptor = CaptureDescriptor.from_file(context.descriptor_path)

    if not force_rebuild:
        existing = _canonicalize_legacy_index(context=context, descriptor=descriptor)
        if existing is not None:
            return existing

    raw_manifest_payload = _optional_json(context.raw_root / "manifest.json")
    intake_payload = _optional_json(context.raw_root / "intake_packet.json")
    capture_context_payload = _optional_json(context.raw_root / "capture_context.json")
    enrichment_runner = build_capture_enrichment_runner(repo_root=Path(__file__).resolve().parents[2])
    artifact_root = context.raw_root / "object_index_artifacts"
    keyframes_dir = artifact_root / "keyframes"
    crops_dir = artifact_root / "crops"
    masks_dir = artifact_root / "masks"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    environment = _infer_environment(descriptor, manifest)
    prompt_bank = _build_prompt_bank(descriptor, intake_payload, capture_context_payload, environment)
    prompt_expansion = None
    prompt_bank, prompt_expansion = _maybe_expand_prompt_bank(
        runner=enrichment_runner,
        descriptor=descriptor,
        intake=intake_payload,
        capture_context=capture_context_payload,
        prompt_bank=prompt_bank,
    )
    keyframes = _sample_keyframes(
        context=context,
        max_keyframes=max(1, _safe_int(os.getenv("OBJECT_INDEX_KEYFRAME_MAX_COUNT"), _DEFAULT_KEYFRAME_COUNT)),
        artifact_dir=keyframes_dir,
    )
    video_path = _resolve_video_path(context, manifest)
    _extract_keyframe_images(video_path, keyframes)
    keyframe_payload = [
        {
            "frame_index": item.frame_index,
            "timestamp": item.timestamp,
            "image_path": str(item.image_path),
            "image_width": item.image_width,
            "image_height": item.image_height,
            "intrinsics": list(item.intrinsics),
            "camera_translation": list(item.camera_translation),
            "motion_score": item.motion_score,
        }
        for item in keyframes
    ]
    write_json(context.raw_root / "object_index_keyframes.json", {"keyframes": keyframe_payload})

    input_payload = {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "raw_root": str(context.raw_root),
        "video_path": str(video_path) if video_path is not None else "",
        "environment": environment,
        "prompt_bank": prompt_bank,
        "keyframes": keyframe_payload,
        "raw_manifest": raw_manifest_payload,
        "intake_packet": intake_payload,
        "capture_context": capture_context_payload,
        "descriptor": descriptor.to_dict(),
    }

    backend_commands = {
        "yolo_world": _command_from_env("OBJECT_INDEX_YOLO_WORLD_COMMAND"),
        "grounding_dino": _command_from_env("OBJECT_INDEX_GROUNDING_DINO_COMMAND"),
        "sam3": _command_from_env("OBJECT_INDEX_SAM3_COMMAND"),
    }
    runtime_preflight = {
        "dependencies": {
            "torch": {"available": _module_available("torch")},
            "ultralytics": {"available": _module_available("ultralytics")},
            "sam3": {"available": _module_available("sam3")},
        },
        "backends": {
            name: _backend_preflight_status(backend_name=name, command_template=command_template)
            for name, command_template in backend_commands.items()
        },
    }

    backend_reports = [
        _run_backend_command(
            backend_name="yolo_world",
            command_template=backend_commands["yolo_world"],
            input_payload=input_payload,
            output_dir=artifact_root,
        ),
        _run_backend_command(
            backend_name="grounding_dino",
            command_template=backend_commands["grounding_dino"],
            input_payload=input_payload,
            output_dir=artifact_root,
        ),
        _run_backend_command(
            backend_name="sam3",
            command_template=backend_commands["sam3"],
            input_payload=input_payload,
            output_dir=artifact_root,
        ),
    ]
    keyframes_by_index = {item.frame_index: item for item in keyframes}

    existing_objects: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    detections_per_backend: Dict[str, int] = {}
    detections_per_keyframe: Dict[str, int] = {}
    manipulation_candidates: List[Dict[str, Any]] = []
    articulation_candidates: List[Dict[str, Any]] = []
    task_candidates: List[Dict[str, Any]] = []
    for report in backend_reports:
        payload = report.get("payload")
        backend_name = str(report.get("backend") or "unknown")
        detections_per_backend[backend_name] = 0
        if not isinstance(payload, Mapping):
            continue
        existing_objects.extend(_normalize_existing_objects(payload))
        parsed_detections, manip, artic, tasks = _normalize_detection_payload(
            backend_name=backend_name,
            payload=payload,
            keyframes_by_index=keyframes_by_index,
        )
        detections_per_backend[backend_name] = len(parsed_detections)
        for item in parsed_detections:
            frame_index = str(_safe_int(item.get("frame_index"), -1))
            detections_per_keyframe[frame_index] = detections_per_keyframe.get(frame_index, 0) + 1
        detections.extend(parsed_detections)
        manipulation_candidates.extend(manip)
        articulation_candidates.extend(artic)
        task_candidates.extend(tasks)

    merged_clusters: List[Sequence[Mapping[str, Any]]] = []
    if existing_objects:
        objects = existing_objects
    else:
        merged_clusters = _cluster_detections(_dedupe_same_frame(detections), keyframes_by_index)
        objects = _build_objects(
            clusters=merged_clusters,
            keyframes_by_index=keyframes_by_index,
            descriptor=descriptor,
            raw_root=context.raw_root,
            crops_dir=crops_dir,
        )
    llm_task_relevance = _apply_llm_task_relevance(
        runner=enrichment_runner,
        descriptor=descriptor,
        objects=objects,
    )
    llm_articulation = _apply_llm_articulation_priors(
        runner=enrichment_runner,
        descriptor=descriptor,
        objects=objects,
    )

    backend_summary = {
        "providers": [
            {
                "backend": str(report.get("backend") or ""),
                "status": str(report.get("status") or "unknown"),
                "reason": str(report.get("reason") or ""),
                "detection_count": detections_per_backend.get(str(report.get("backend") or ""), 0),
            }
            for report in backend_reports
        ],
        "detection_count": len(detections),
        "object_count": len(objects),
    }
    grounding_payload = _grounding_payload_from_objects(objects, descriptor, backend_summary)
    if manipulation_candidates:
        grounding_payload["manipulation_candidates"] = manipulation_candidates
    if articulation_candidates:
        grounding_payload["articulation_hints"] = articulation_candidates
    if task_candidates:
        grounding_payload["tasks"] = task_candidates
    llm_target_resolution = _llm_target_resolution(
        runner=enrichment_runner,
        descriptor=descriptor,
        objects=objects,
    )
    if isinstance(llm_target_resolution, Mapping):
        if isinstance(llm_target_resolution.get("manipulation_candidates"), list):
            grounding_payload["manipulation_candidates"] = [
                *grounding_payload.get("manipulation_candidates", []),
                *[dict(item) for item in llm_target_resolution.get("manipulation_candidates", []) if isinstance(item, Mapping)],
            ]
        if isinstance(llm_target_resolution.get("articulation_hints"), list):
            grounding_payload["articulation_hints"] = [
                *grounding_payload.get("articulation_hints", []),
                *[dict(item) for item in llm_target_resolution.get("articulation_hints", []) if isinstance(item, Mapping)],
            ]
        if isinstance(llm_target_resolution.get("tasks"), list) and llm_target_resolution.get("tasks"):
            grounding_payload["tasks"] = [dict(item) for item in llm_target_resolution.get("tasks", []) if isinstance(item, Mapping)]

    object_index_uri = join_gs_uri(context.raw_prefix_uri, "object_index.json")
    object_index_path = context.raw_root / "object_index.json"
    write_json(
        object_index_path,
        {
            "objects": objects,
            "world_model_policy": policy.to_dict(),
            "canonical_output": build_output_linkage(
                policy=policy,
                canonical_artifact_uri=object_index_uri,
                presentation_artifact_uri=None,
                authoritative_record=True,
            ),
            "provenance": build_provenance_record(
                grounding_level="observed" if objects else "inferred",
                evidence_sources=[str(item.get("image_path") or "") for item in keyframe_payload],
                observation_coverage={
                    "keyframe_count": len(keyframes),
                    "detection_count": len(detections),
                    "object_count": len(objects),
                },
                confidence=1.0 if objects else 0.0,
                canonical_truth=True,
                presentation_only=False,
            ),
        },
    )

    filtered_detection_count = max(0, len(detections) - len(objects))
    clustered_object_count = len(existing_objects) if existing_objects else len(merged_clusters)
    empty_index_cause = None
    provider_reasons = [str(report.get("reason") or "") for report in backend_reports]
    if not objects:
        if any(_backend_reason_indicates_runtime_missing(reason) for reason in provider_reasons if reason):
            empty_index_cause = "runtime_missing"
        elif any(str(report.get("status") or "") == "skipped" for report in backend_reports):
            empty_index_cause = "backend_skipped"
        elif len(detections) == 0:
            empty_index_cause = "zero_detections"
        else:
            empty_index_cause = "all_filtered"

    report_path = context.raw_root / "object_index_build_report.json"
    write_json(
        report_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "built",
            "capture_root": str(context.capture_root),
            "video_path": str(video_path) if video_path is not None else "",
            "environment": environment,
            "prompt_bank": prompt_bank,
            "keyframe_count": len(keyframes),
            "object_count": len(objects),
            "detections_per_backend": detections_per_backend,
            "detections_per_keyframe": detections_per_keyframe,
            "filtered_detection_count": filtered_detection_count,
            "clustered_object_count": clustered_object_count,
            "empty_index_cause": empty_index_cause,
            "runtime_preflight": runtime_preflight,
            "world_model_policy": policy.to_dict(),
            "backend_summary": backend_summary,
            "llm_enrichment": {
                "prompt_bank_expander": prompt_expansion,
                "task_relevance_ranker": llm_task_relevance,
                "articulation_prior_writer": llm_articulation,
                "workflow_target_resolver": llm_target_resolution,
            },
        },
    )
    write_json(context.raw_root / "object_grounding_hints.json", grounding_payload)
    _write_descriptor_updates(context.descriptor_path, descriptor, object_index_uri)
    _write_manifest_updates(context.raw_root / "manifest.json")
    return {
        "schema_version": "v1",
        "status": "built",
        "capture_root": str(context.capture_root),
        "object_index_uri": object_index_uri,
        "manifest_path": str(object_index_path),
        "report_path": str(report_path),
        "object_count": len(objects),
        "grounding_payload": grounding_payload,
    }


def ensure_object_index_stage(
    *,
    capture_root: str | Path,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    return run_object_index_stage(capture_root=capture_root, force_rebuild=force_rebuild)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a canonical object index for a staged capture")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_object_index_stage(
            capture_root=args.capture_root,
            force_rebuild=bool(args.force_rebuild),
        )
    except Exception as exc:
        print(f"[object-index-stage] FAILED: {exc}")
        return 1
    print(f"[object-index-stage] manifest={result['manifest_path']}")
    print(f"[object-index-stage] report={result['report_path']}")
    print(f"[object-index-stage] object_count={result['object_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
