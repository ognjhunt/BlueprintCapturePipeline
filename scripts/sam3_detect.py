#!/usr/bin/env python3
"""SAM3 object detection for the BlueprintCapture swap pipeline.

Replaces ARKit's on-device object detection with Meta's SAM3 (Segment
Anything Model 3) running server-side on a GPU VM.  Produces the
``object_point_cloud_index.json`` consumed by the swap orchestrator's
candidate-selection stage.

The script:
  1. Extracts frames from the capture video (or reads pre-extracted frames)
  2. Runs SAM3 text-prompted detection across multiple sampled frames
  3. Merges per-frame detections into unique scene-level objects
  4. Estimates bounding boxes from 2D detections + COLMAP camera info
  5. Writes ``object_point_cloud_index.json`` and optional per-object masks

Usage:
  python3 sam3_detect.py \
      --frames-dir /workspace/test_scene/images \
      --output /workspace/test_scene/raw/arkit/objects/index.json \
      --environment auto \
      --colmap-sparse /workspace/test_scene/colmap/sparse/0
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import uuid
from urllib.parse import urlparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

# Swap-policy keyword lists used for text prompts
_DETECTION_PROMPTS: Dict[str, List[str]] = {
    "default": [
        "door",
        "bed",
        "nightstand",
        "dresser",
        "closet",
        "cabinet",
        "drawer",
        "desk",
        "chair",
        "table",
        "shelf",
        "lamp",
        "mirror",
        "tv",
        "monitor",
        "box",
        "container",
        "basket",
        "hamper",
    ],
    "warehouse": [
        "shelf", "box", "tote", "bin", "crate", "container",
        "pallet", "carton", "package", "door", "cart",
        "forklift", "rack", "barrel", "drum",
    ],
    "kitchen": [
        "cabinet", "drawer", "refrigerator", "fridge", "microwave",
        "oven", "dishwasher", "door", "mug", "cup", "bowl",
        "plate", "pot", "pan", "bottle",
    ],
    "bedroom": [
        "bed",
        "nightstand",
        "dresser",
        "wardrobe",
        "closet_door",
        "door",
        "desk",
        "chair",
        "lamp",
        "mirror",
        "box",
        "container",
        "basket",
        "hamper",
        "suitcase",
        "backpack",
        "laundry basket",
        "shoes",
        "laptop",
        "mug",
    ],
}

# Generic fallback prompt list for ``environment=auto``.
_AUTO_FALLBACK_PROMPTS: List[str] = [
    "door",
    "bed",
    "nightstand",
    "dresser",
    "closet",
    "cabinet",
    "drawer",
    "desk",
    "chair",
    "table",
    "shelf",
    "lamp",
    "mirror",
    "tv",
    "monitor",
    "box",
    "container",
    "basket",
    "hamper",
]

# Objects that are structural and should be excluded from swap candidates
_STRUCTURAL_LABELS = {"wall", "floor", "ceiling", "window", "stairs"}

# Minimum detection confidence to include
_MIN_CONFIDENCE = 0.45

# How many frames to sample for detection
_DEFAULT_SAMPLE_FRAMES = 8

# IoU threshold for merging detections across frames
_MERGE_IOU_THRESHOLD = 0.35
_INSTANCE_MASK_MAX_ID = 65535


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# Prompt inference settings
_PROMPT_INFERENCE_COMMAND = (os.getenv("PROMPT_INFERENCE_COMMAND") or "").strip()
_PROMPT_INFERENCE_TIMEOUT_SEC = max(10, _env_int("PROMPT_INFERENCE_TIMEOUT_SEC", 120))

# Tracking/association settings
_TRACKING_MODE_DEFAULT = (os.getenv("SAM3_TRACKING_MODE", "auto") or "auto").strip().lower()
_SAM3_FULL_VIDEO_MAX_FRAMES = max(50, _env_int("SAM3_FULL_VIDEO_MAX_FRAMES", 600))
_TRACK_MAX_FRAME_GAP = max(1, _env_int("SAM3_TRACK_MAX_FRAME_GAP", 3))
_TRACK_MIN_ASSOC_SCORE = max(0.0, min(1.0, _env_float("SAM3_TRACK_MIN_ASSOC_SCORE", 0.28)))
_MAX_REFERENCE_CROPS = max(1, _env_int("SAM3_MAX_REFERENCE_CROPS", 12))

# DA3 model source (prefer local snapshot to avoid runtime downloads)
_DA3_MODEL_ID = os.getenv("DA3_MODEL_ID", "depth-anything/DA3Metric-Large")
_DA3_MODEL_PATH = Path(os.getenv("DA3_MODEL_PATH", "/opt/da3/weights/metric_large"))
_DA3_MODEL_NAME = os.getenv("DA3_MODEL_NAME", "da3metric-large")

# SAM3 weights (prefer local snapshot to avoid gated-repo HF download)
_SAM3_WEIGHTS_PATH = Path(os.getenv("SAM3_WEIGHTS_PATH", "/opt/sam3_weights/sam3.pt"))

# Occlusion-aware dimension completion settings
_DIM_COMPLETION_DEFAULT_MODE = (
    os.getenv("SAM3_DIMENSION_COMPLETION_MODE", "auto") or "auto"
).strip().lower()
_DIM_COMPLETION_RUNNER_PATH = Path(__file__).with_name("sam3_dimension_completion_runner.py")
_DIM_COMPLETION_DEFAULT_COMMAND = (
    f"{shlex.quote(os.getenv('SAM3_DIMENSION_COMPLETION_PYTHON', 'python3'))} "
    f"{shlex.quote(str(_DIM_COMPLETION_RUNNER_PATH))}"
    if _DIM_COMPLETION_RUNNER_PATH.is_file()
    else ""
)
_DIM_COMPLETION_COMMAND = (
    os.getenv("SAM3_DIMENSION_COMPLETION_COMMAND")
    or _DIM_COMPLETION_DEFAULT_COMMAND
    or ""
).strip()
_DIM_COMPLETION_GEMINI_MODEL = (
    os.getenv("SAM3_DIMENSION_COMPLETION_GEMINI_MODEL") or "gemini-2.5-flash"
).strip()
_DIM_COMPLETION_TIMEOUT_SEC = max(5, _env_int("SAM3_DIMENSION_COMPLETION_TIMEOUT_SEC", 40))
_DIM_COMPLETION_MAX_OBJECTS = max(1, _env_int("SAM3_DIMENSION_COMPLETION_MAX_OBJECTS", 8))
_DIM_COMPLETION_MAX_IMAGES = max(1, _env_int("SAM3_DIMENSION_COMPLETION_MAX_IMAGES", 2))
_DIM_COMPLETION_MIN_OCCLUSION_SCORE = max(
    0.0,
    min(1.0, _env_float("SAM3_DIMENSION_COMPLETION_MIN_OCCLUSION_SCORE", 0.52)),
)
_DIM_COMPLETION_MIN_CONFIDENCE = max(
    0.0,
    min(1.0, _env_float("SAM3_DIMENSION_COMPLETION_MIN_CONFIDENCE", 0.35)),
)
_DIM_COMPLETION_LOW_FRAME_THRESHOLD = max(
    1,
    _env_int("SAM3_DIMENSION_COMPLETION_LOW_FRAME_THRESHOLD", 4),
)
_DIM_COMPLETION_EDGE_MARGIN_RATIO = max(
    0.005,
    min(0.2, _env_float("SAM3_DIMENSION_COMPLETION_EDGE_MARGIN_RATIO", 0.04)),
)
_DIM_COMPLETION_MAX_EXPAND_RATIO = max(
    1.0,
    min(3.0, _env_float("SAM3_DIMENSION_COMPLETION_MAX_EXPAND_RATIO", 1.75)),
)
_DIM_COMPLETION_ALLOW_SHRINK = _env_bool("SAM3_DIMENSION_COMPLETION_ALLOW_SHRINK", default=False)


def _log(msg: str) -> None:
    print(f"[sam3-detect] {msg}", flush=True)


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


def _validate_local_video_path(raw_video_path: str) -> Path:
    """Validate --video input to ensure it is a local filesystem file."""
    parsed = urlparse((raw_video_path or "").strip())
    if parsed.scheme:
        raise ValueError("--video must be a local filesystem path, not a URI")

    video_path = Path(raw_video_path).expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"--video does not exist or is not a file: {video_path}")

    if video_path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("--video must point to a .mp4 or .mov file")

    return video_path


def _normalize_dimension_completion_mode(value: Optional[str]) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"off", "auto", "always"}:
        return mode
    return "auto"


def _extract_response_text(response: Any) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if not isinstance(parts, list):
            continue
        for part in parts:
            if getattr(part, "thought", False):
                continue
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                return part_text
    return ""


def _extract_json_object(text: str) -> Dict[str, Any]:
    payload_text = (text or "").strip()
    if not payload_text:
        return {}

    try:
        payload = json.loads(payload_text)
        if isinstance(payload, Mapping):
            return dict(payload)
    except Exception:
        pass

    cleaned = re.sub(r"```(?:json)?\s*", "", payload_text).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, Mapping):
            return dict(payload)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _parse_extents_triplet(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, list) and len(payload) >= 3:
        extents = [_safe_float(payload[idx], -1.0) for idx in range(3)]
    elif isinstance(payload, Mapping):
        extents = [
            _safe_float(payload.get("x"), -1.0),
            _safe_float(payload.get("y"), -1.0),
            _safe_float(payload.get("z"), -1.0),
        ]
    else:
        return None
    if any(value <= 0.0 for value in extents):
        return None
    return [max(0.02, min(8.0, float(value))) for value in extents]


def _parse_completion_payload(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    extents: Optional[List[float]] = None
    for key in ("predicted_extents_m", "extents_m", "dimensions_m", "extents"):
        extents = _parse_extents_triplet(payload.get(key))
        if extents is not None:
            break
    if extents is None:
        nested = payload.get("prediction")
        if isinstance(nested, Mapping):
            extents = _parse_extents_triplet(nested.get("extents"))
    if extents is None:
        return None

    confidence = _safe_float(payload.get("confidence", payload.get("score", 0.0)), 0.0)
    confidence = max(0.0, min(1.0, confidence))
    model = str(payload.get("model") or "").strip()
    reason = str(payload.get("reason") or payload.get("rationale") or "").strip()

    return {
        "predicted_extents": extents,
        "confidence": confidence,
        "model": model,
        "reason": reason,
    }


def _object_extents(obj: Mapping[str, Any]) -> Optional[List[float]]:
    bbox = obj.get("boundingBox") if isinstance(obj.get("boundingBox"), Mapping) else {}
    extents = bbox.get("extents") if isinstance(bbox.get("extents"), list) else None
    if extents is None or len(extents) < 3:
        return None
    parsed = [_safe_float(extents[idx], -1.0) for idx in range(3)]
    if any(value <= 0.0 for value in parsed):
        return None
    return [max(0.02, min(8.0, float(value))) for value in parsed]


def _object_image_size(obj: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    raw = obj.get("image_size")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    width = _safe_int(raw[0], 0)
    height = _safe_int(raw[1], 0)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _object_box(obj: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    raw = obj.get("mean_box_px")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    x1 = _safe_float(raw[0], 0.0)
    y1 = _safe_float(raw[1], 0.0)
    x2 = _safe_float(raw[2], 0.0)
    y2 = _safe_float(raw[3], 0.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _compute_occlusion_signals(obj: Mapping[str, Any]) -> Dict[str, Any]:
    frame_count = max(0, _safe_int(obj.get("n_frame_detections"), 0))
    mean_confidence = _safe_float(obj.get("mean_confidence", obj.get("confidence", 0.0)), 0.0)
    mean_confidence = max(0.0, min(1.0, mean_confidence))

    edge_sides_touch = 0
    edge_touch_fraction = 0.0
    image_size = _object_image_size(obj)
    box = _object_box(obj)
    if image_size is not None and box is not None:
        width, height = image_size
        x1, y1, x2, y2 = box
        margin_x = float(width) * _DIM_COMPLETION_EDGE_MARGIN_RATIO
        margin_y = float(height) * _DIM_COMPLETION_EDGE_MARGIN_RATIO
        edge_sides_touch = sum(
            [
                x1 <= margin_x,
                y1 <= margin_y,
                x2 >= float(width) - margin_x,
                y2 >= float(height) - margin_y,
            ]
        )
        edge_touch_fraction = edge_sides_touch / 4.0

    low_frame_score = max(
        0.0,
        min(
            1.0,
            (_DIM_COMPLETION_LOW_FRAME_THRESHOLD - float(frame_count))
            / float(max(1, _DIM_COMPLETION_LOW_FRAME_THRESHOLD)),
        ),
    )
    low_confidence_score = 1.0 - mean_confidence

    refinement = str(obj.get("refinement") or "").strip().lower()
    refinement_uncertainty = {
        "da3_metric_depth": 0.15,
        "gaussian_backprojection": 0.3,
        "focal_length_estimate": 0.65,
        "heuristic_2d": 0.75,
    }.get(refinement, 0.5)

    extents = _object_extents(obj)
    thinness_score = 0.0
    if extents:
        emax = max(extents)
        emin = min(extents)
        if emax > 0:
            thin_ratio = emin / emax
            thinness_score = max(0.0, min(1.0, (0.18 - thin_ratio) / 0.18))

    occlusion_score = (
        (0.45 * edge_touch_fraction)
        + (0.2 * low_frame_score)
        + (0.15 * low_confidence_score)
        + (0.15 * refinement_uncertainty)
        + (0.05 * thinness_score)
    )
    occlusion_score = max(0.0, min(1.0, occlusion_score))

    return {
        "edge_sides_touch": int(edge_sides_touch),
        "edge_touch_fraction": round(float(edge_touch_fraction), 4),
        "low_frame_score": round(float(low_frame_score), 4),
        "low_confidence_score": round(float(low_confidence_score), 4),
        "refinement_uncertainty": round(float(refinement_uncertainty), 4),
        "thinness_score": round(float(thinness_score), 4),
        "occlusion_score": round(float(occlusion_score), 4),
    }


def _resolve_reference_crops(
    obj: Mapping[str, Any],
    *,
    output_path: Path,
    max_images: int,
) -> List[Path]:
    candidates: List[str] = []
    ref_crop = obj.get("reference_crop")
    if isinstance(ref_crop, str) and ref_crop.strip():
        candidates.append(ref_crop.strip())
    all_crops = obj.get("all_crops")
    if isinstance(all_crops, list):
        for crop in all_crops:
            text = str(crop).strip()
            if text:
                candidates.append(text)

    unique: List[Path] = []
    seen: set[str] = set()
    for item in candidates:
        rel = item.strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        path = Path(rel)
        if not path.is_absolute():
            path = output_path.parent / path
        if path.is_file():
            unique.append(path)
        if len(unique) >= max_images:
            break
    return unique


def _run_completion_command(
    *,
    command: str,
    label: str,
    environment: str,
    observed_extents: List[float],
    crop_paths: List[Path],
) -> Optional[Dict[str, Any]]:
    env = os.environ.copy()
    env["SAM3_COMPLETION_LABEL"] = label
    env["SAM3_COMPLETION_ENVIRONMENT"] = environment
    env["SAM3_COMPLETION_OBSERVED_EXTENTS_JSON"] = json.dumps(observed_extents)
    env["SAM3_COMPLETION_CROP_PATHS_JSON"] = json.dumps([str(path) for path in crop_paths])

    command_text = command
    try:
        command_text = command.format(
            label=shlex.quote(label),
            environment=shlex.quote(environment),
            observed_extents_json=shlex.quote(json.dumps(observed_extents)),
            crop_paths_json=shlex.quote(json.dumps([str(path) for path in crop_paths])),
        )
    except Exception:
        command_text = command

    try:
        result = subprocess.run(
            command_text,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_DIM_COMPLETION_TIMEOUT_SEC,
            env=env,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    payload = _extract_json_object(result.stdout or "")
    parsed = _parse_completion_payload(payload)
    if parsed is None:
        return None
    parsed["provider"] = "command"
    return parsed


def _run_completion_gemini(
    *,
    label: str,
    environment: str,
    observed_extents: List[float],
    crop_paths: List[Path],
) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google import genai  # type: ignore
    except Exception:
        return None

    prompt = (
        "Estimate the full object dimensions in meters for a partially visible object.\n"
        f"Object label: {label}\n"
        f"Environment: {environment}\n"
        f"Observed extents (x,y,z meters): {json.dumps(observed_extents)}\n\n"
        "Use the image crop plus common object priors. Return strict JSON:\n"
        "{\"predicted_extents_m\":[x,y,z],\"confidence\":0.0-1.0,"
        "\"reason\":\"short text\"}\n"
        "Rules: x,y,z must each be within [0.02, 8.0]."
    )
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for path in crop_paths[:_DIM_COMPLETION_MAX_IMAGES]:
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:
            continue
        parts.append({"inline_data": {"mime_type": mime, "data": encoded}})

    if len(parts) <= 1:
        return None

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=_DIM_COMPLETION_GEMINI_MODEL,
            contents=[{"parts": parts}],
            config={
                "temperature": 0.15,
                "max_output_tokens": 512,
                "response_mime_type": "application/json",
            },
        )
    except Exception:
        return None

    raw_text = _extract_response_text(response)
    payload = _extract_json_object(raw_text)
    parsed = _parse_completion_payload(payload)
    if parsed is None:
        return None
    parsed["provider"] = "gemini"
    parsed["model"] = parsed.get("model") or _DIM_COMPLETION_GEMINI_MODEL
    return parsed


def _infer_dimension_completion_estimate(
    *,
    obj: Mapping[str, Any],
    environment: str,
    observed_extents: List[float],
    crop_paths: List[Path],
) -> Dict[str, Any]:
    label = str(obj.get("label") or obj.get("id") or "object")
    if _DIM_COMPLETION_COMMAND:
        estimate = _run_completion_command(
            command=_DIM_COMPLETION_COMMAND,
            label=label,
            environment=environment,
            observed_extents=observed_extents,
            crop_paths=crop_paths,
        )
        if estimate is not None:
            return {"ok": True, **estimate}

    estimate = _run_completion_gemini(
        label=label,
        environment=environment,
        observed_extents=observed_extents,
        crop_paths=crop_paths,
    )
    if estimate is not None:
        return {"ok": True, **estimate}

    return {"ok": False, "reason": "no_estimator_available"}


def _fuse_completed_extents(
    *,
    observed_extents: List[float],
    predicted_extents: List[float],
    model_confidence: float,
    occlusion_score: float,
) -> Tuple[List[float], float]:
    observed = np.array(observed_extents, dtype=float)
    predicted = np.array(predicted_extents, dtype=float)
    observed = np.clip(observed, 0.02, 8.0)
    predicted = np.clip(predicted, 0.02, 8.0)

    cap = np.maximum(observed, observed * float(_DIM_COMPLETION_MAX_EXPAND_RATIO))
    if not _DIM_COMPLETION_ALLOW_SHRINK:
        predicted = np.maximum(predicted, observed)
    predicted = np.minimum(predicted, cap)

    confidence = max(0.0, min(1.0, float(model_confidence)))
    occlusion = max(0.0, min(1.0, float(occlusion_score)))
    alpha = max(0.15, min(0.85, confidence * max(0.35, occlusion)))

    fused = observed + (alpha * (predicted - observed))
    if not _DIM_COMPLETION_ALLOW_SHRINK:
        fused = np.maximum(fused, observed)
    fused = np.minimum(fused, cap)
    return [round(float(value), 4) for value in fused], round(float(alpha), 4)


def _apply_occlusion_dimension_completion(
    *,
    objects: List[Dict[str, Any]],
    output_path: Path,
    environment: str,
    mode_override: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = _normalize_dimension_completion_mode(mode_override or _DIM_COMPLETION_DEFAULT_MODE)
    report: Dict[str, Any] = {
        "enabled": mode != "off",
        "mode": mode,
        "objects_considered": len(objects),
        "objects_attempted": 0,
        "objects_completed": 0,
        "objects_updated": 0,
        "min_occlusion_score": round(float(_DIM_COMPLETION_MIN_OCCLUSION_SCORE), 4),
        "min_model_confidence": round(float(_DIM_COMPLETION_MIN_CONFIDENCE), 4),
        "max_objects": int(_DIM_COMPLETION_MAX_OBJECTS),
        "max_expand_ratio": round(float(_DIM_COMPLETION_MAX_EXPAND_RATIO), 4),
        "provider_order": (
            ["command", "gemini"] if _DIM_COMPLETION_COMMAND else ["gemini"]
        ),
    }
    if mode == "off" or not objects:
        report["reason"] = "disabled_or_no_objects"
        return objects, report

    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, obj in enumerate(objects):
        signals = _compute_occlusion_signals(obj)
        completion_info = {
            "status": "skipped",
            "reason": "low_occlusion_score",
            "signals": signals,
            "occlusion_score": signals["occlusion_score"],
        }
        obj["dimension_completion"] = completion_info
        should_attempt = (
            mode == "always"
            or float(signals["occlusion_score"]) >= float(_DIM_COMPLETION_MIN_OCCLUSION_SCORE)
        )
        if should_attempt:
            scored.append((float(signals["occlusion_score"]), idx, signals))

    scored.sort(key=lambda item: item[0], reverse=True)
    budget = min(len(scored), int(_DIM_COMPLETION_MAX_OBJECTS))

    for rank, (_, idx, signals) in enumerate(scored):
        obj = objects[idx]
        if rank >= budget:
            obj["dimension_completion"]["status"] = "skipped"
            obj["dimension_completion"]["reason"] = "max_object_budget_reached"
            continue

        observed_extents = _object_extents(obj)
        if observed_extents is None:
            obj["dimension_completion"]["status"] = "skipped"
            obj["dimension_completion"]["reason"] = "missing_observed_extents"
            continue

        crop_paths = _resolve_reference_crops(
            obj,
            output_path=output_path,
            max_images=_DIM_COMPLETION_MAX_IMAGES,
        )
        if not crop_paths:
            obj["dimension_completion"]["status"] = "skipped"
            obj["dimension_completion"]["reason"] = "missing_reference_crop"
            continue

        report["objects_attempted"] = int(report["objects_attempted"]) + 1
        estimate = _infer_dimension_completion_estimate(
            obj=obj,
            environment=environment,
            observed_extents=observed_extents,
            crop_paths=crop_paths,
        )
        if not bool(estimate.get("ok", False)):
            reason = str(estimate.get("reason") or "inference_failed")
            obj["dimension_completion"]["status"] = (
                "skipped" if reason == "no_estimator_available" else "failed"
            )
            obj["dimension_completion"]["reason"] = reason
            continue

        predicted_extents = estimate.get("predicted_extents")
        if not isinstance(predicted_extents, list) or len(predicted_extents) < 3:
            obj["dimension_completion"]["status"] = "failed"
            obj["dimension_completion"]["reason"] = "invalid_prediction_payload"
            continue

        model_confidence = max(0.0, min(1.0, _safe_float(estimate.get("confidence"), 0.0)))
        if model_confidence < _DIM_COMPLETION_MIN_CONFIDENCE:
            obj["dimension_completion"]["status"] = "skipped"
            obj["dimension_completion"]["reason"] = "model_confidence_below_threshold"
            obj["dimension_completion"]["model_confidence"] = round(float(model_confidence), 4)
            continue

        fused_extents, alpha = _fuse_completed_extents(
            observed_extents=observed_extents,
            predicted_extents=predicted_extents[:3],
            model_confidence=model_confidence,
            occlusion_score=float(signals.get("occlusion_score", 0.0)),
        )

        bbox = obj.get("boundingBox")
        if isinstance(bbox, Mapping):
            obj["boundingBox"]["extents"] = fused_extents

        observed_rounded = [round(float(value), 4) for value in observed_extents]
        predicted_rounded = [round(float(value), 4) for value in predicted_extents[:3]]
        changed = any(
            abs(fused_extents[i] - observed_rounded[i]) > 1e-4
            for i in range(3)
        )

        obj["dimension_completion"] = {
            "status": "completed",
            "reason": "fused_with_model_prior",
            "signals": signals,
            "provider": str(estimate.get("provider") or "unknown"),
            "model": str(estimate.get("model") or ""),
            "model_confidence": round(float(model_confidence), 4),
            "blend_alpha": round(float(alpha), 4),
            "observed_extents": observed_rounded,
            "predicted_extents": predicted_rounded,
            "fused_extents": fused_extents,
            "updated": changed,
        }
        if estimate.get("reason"):
            obj["dimension_completion"]["model_reason"] = str(estimate.get("reason"))

        report["objects_completed"] = int(report["objects_completed"]) + 1
        if changed:
            report["objects_updated"] = int(report["objects_updated"]) + 1

    _log(
        "Dimension completion: "
        f"attempted={report['objects_attempted']} "
        f"completed={report['objects_completed']} "
        f"updated={report['objects_updated']}"
    )
    return objects, report


# ---------------------------------------------------------------------------
# Video predictor helpers: adaptive FPS + frame extraction
# ---------------------------------------------------------------------------

_VIDEO_PREDICTOR_MODEL_VRAM_GB = 3.8  # SAM3 video predictor model weights
_VIDEO_PREDICTOR_PER_FRAME_MB = 11.4  # VRAM per frame in video predictor session
_VIDEO_PREDICTOR_SAFETY_MARGIN_GB = 1.5
_VIDEO_PREDICTOR_MAX_FPS = 15  # Diminishing returns above this
_VIDEO_PREDICTOR_MIN_FPS = 3   # Floor to ensure useful detection


def _compute_safe_fps(
    video_path: Path,
    model_vram_gb: float = _VIDEO_PREDICTOR_MODEL_VRAM_GB,
) -> Tuple[int, str]:
    """Compute maximum safe FPS for video predictor based on available VRAM.

    Returns (fps, reasoning_string).
    """
    # Get video duration via ffprobe
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        duration_sec = float(probe.stdout.strip())
    except Exception as exc:
        _log(f"ffprobe failed ({exc}), assuming 60s video")
        duration_sec = 60.0

    if duration_sec <= 0:
        duration_sec = 60.0

    # Query GPU VRAM
    try:
        free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
        free_vram_gb = free_vram_bytes / (1024 ** 3)
        total_vram_gb = total_vram_bytes / (1024 ** 3)
    except Exception:
        # Fallback: assume 16GB with 14GB free
        free_vram_gb = 14.0
        total_vram_gb = 16.0

    usable_gb = free_vram_gb - model_vram_gb - _VIDEO_PREDICTOR_SAFETY_MARGIN_GB
    usable_mb = usable_gb * 1024
    max_frames = int(usable_mb / _VIDEO_PREDICTOR_PER_FRAME_MB)
    max_fps_raw = max_frames / duration_sec if duration_sec > 0 else _VIDEO_PREDICTOR_MAX_FPS

    fps = int(min(_VIDEO_PREDICTOR_MAX_FPS, max(
        _VIDEO_PREDICTOR_MIN_FPS, max_fps_raw,
    )))

    expected_frames = int(fps * duration_sec)
    expected_vram_gb = (expected_frames * _VIDEO_PREDICTOR_PER_FRAME_MB) / 1024

    reasoning = (
        f"VRAM {total_vram_gb:.0f}GB (free {free_vram_gb:.1f}GB), "
        f"model {model_vram_gb}GB, safety {_VIDEO_PREDICTOR_SAFETY_MARGIN_GB}GB, "
        f"usable {usable_gb:.1f}GB → max {max_frames} frames. "
        f"Video {duration_sec:.1f}s → {fps}fps ({expected_frames} frames, "
        f"~{expected_vram_gb:.1f}GB session VRAM)"
    )
    _log(f"Adaptive FPS: {reasoning}")
    return fps, reasoning


def _extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: int,
) -> Tuple[Path, int]:
    """Extract frames from video at given FPS using ffmpeg.

    Frames are named 00000.jpg, 00001.jpg, ... (SAM3 expected format).
    Skips extraction if output_dir already exists with correct frame count.

    Returns (frames_dir, frame_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already extracted
    existing = sorted(output_dir.glob("*.jpg"))
    if existing:
        _log(f"Frames already extracted: {len(existing)} frames in {output_dir}")
        return output_dir, len(existing)

    _log(f"Extracting frames at {fps}fps from {video_path.name} → {output_dir}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        str(output_dir / "%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed (exit {result.returncode}): "
            f"{result.stderr[-500:]}"
        )

    frames = sorted(output_dir.glob("*.jpg"))
    _log(f"Extracted {len(frames)} frames at {fps}fps")
    return output_dir, len(frames)


def _save_object_crop(
    image_path: Path,
    mask_np: np.ndarray,
    box: List[float],
    crops_dir: Path,
    label: str,
    frame_idx: int,
    det_idx: int,
) -> Optional[Path]:
    """Save a masked RGBA crop of the detected object.

    Extracts the bounding-box region with 5% padding, applies the
    segmentation mask as an alpha channel (transparent background),
    and resizes to max 512×512 preserving aspect ratio.

    Returns a path relative to the object index directory (for portability),
    or None on failure.
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Bounding box with 5% padding
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        pad_x = bw * 0.05
        pad_y = bh * 0.05
        x1 = max(0, int(box[0] - pad_x))
        y1 = max(0, int(box[1] - pad_y))
        x2 = min(w, int(box[2] + pad_x))
        y2 = min(h, int(box[3] + pad_y))

        crop_rgb = img.crop((x1, y1, x2, y2))

        # Build alpha from segmentation mask
        mask_full = (mask_np.astype(np.uint8) * 255)
        mask_crop = mask_full[y1:y2, x1:x2]
        alpha = Image.fromarray(mask_crop, mode="L")

        # Combine into RGBA
        crop_rgba = crop_rgb.copy().convert("RGBA")
        crop_rgba.putalpha(alpha)

        # Resize to max 512×512 preserving aspect ratio
        max_dim = 512
        cw, ch = crop_rgba.size
        if max(cw, ch) > max_dim:
            scale = max_dim / max(cw, ch)
            crop_rgba = crop_rgba.resize(
                (int(cw * scale), int(ch * scale)),
                Image.LANCZOS,
            )

        # Save
        crops_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace("/", "_").replace(" ", "_")
        filename = f"{safe_label}_{frame_idx:03d}_{det_idx:03d}.png"
        out_path = crops_dir / filename
        crop_rgba.save(out_path, "PNG")
        return Path(crops_dir.name) / filename

    except Exception as exc:
        _log(f"    Crop save failed for {label}: {exc}")
        return None


def _accumulate_instance_mask(
    mask_np: np.ndarray,
    object_id: int,
    frame_name: str,
    instance_masks_dir: Path,
) -> None:
    """Accumulate a per-object binary mask into a per-view instance segmentation PNG.

    Each pixel in the output PNG holds the 1-indexed object ID that owns it
    (0 = background). Masks are written as uint16 PNG so object IDs >255 are
    represented losslessly.

    Args:
        mask_np: boolean [H, W] mask for one object.
        object_id: 1-indexed global object ID.
        frame_name: stem of the image file (no extension).
        instance_masks_dir: directory for instance segmentation PNGs.
    """
    if mask_np is None or not mask_np.any():
        return
    try:
        instance_masks_dir.mkdir(parents=True, exist_ok=True)
        out_path = instance_masks_dir / f"{frame_name}.png"
        if out_path.exists():
            from PIL import Image as _PIL
            existing = np.array(_PIL.open(out_path), dtype=np.uint16)
        else:
            existing = np.zeros(mask_np.shape[:2], dtype=np.uint16)

        if existing.shape != mask_np.shape[:2]:
            _log(
                "    Instance mask shape mismatch for "
                f"{frame_name}: existing={existing.shape} new={mask_np.shape[:2]}"
            )
            return

        existing[mask_np] = min(max(int(object_id), 0), _INSTANCE_MASK_MAX_ID)

        from PIL import Image as _PIL
        _PIL.fromarray(existing.astype(np.uint16), mode="I;16").save(out_path)
    except Exception as exc:
        _log(f"    Instance mask write failed for {frame_name} obj {object_id}: {exc}")


def _frame_name_from_detection(det: Mapping[str, Any]) -> str:
    frame_path = str(det.get("frame_path") or "").strip()
    if frame_path:
        stem = Path(frame_path).stem
        if stem:
            return stem
    frame_idx = int(det.get("frame_idx", 0))
    return f"frame_{frame_idx:05d}"


def _write_instance_masks_from_objects(
    *,
    objects: List[Dict[str, Any]],
    instance_masks_dir: Path,
) -> Dict[str, Any]:
    """Compose per-frame instance PNGs from final merged objects.

    Composition order is deterministic and confidence-aware:
    1) lower-confidence objects first, higher-confidence objects later
    2) within each object, lower per-frame detection confidence first
    """
    contributions: List[Tuple[float, float, str, int, np.ndarray]] = []
    for obj in objects:
        instance_mask_id = int(obj.get("instance_mask_id", 0))
        if instance_mask_id <= 0:
            continue
        obj_conf = float(obj.get("confidence", 0.0))
        dets = obj.get("_cluster_detections")
        if not isinstance(dets, list):
            continue
        for det in dets:
            if not isinstance(det, Mapping):
                continue
            mask_np = det.get("_mask_np")
            if not isinstance(mask_np, np.ndarray) or mask_np.size == 0:
                continue
            if mask_np.dtype != np.bool_:
                mask_np = mask_np.astype(bool)
            if not mask_np.any():
                continue
            det_score = float(det.get("score", 0.0))
            frame_name = _frame_name_from_detection(det)
            contributions.append((obj_conf, det_score, frame_name, instance_mask_id, mask_np))

    contributions.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    for _, _, frame_name, instance_mask_id, mask_np in contributions:
        _accumulate_instance_mask(mask_np, instance_mask_id, frame_name, instance_masks_dir)

    frame_count = len({item[2] for item in contributions})
    return {
        "instance_masks_dir": str(instance_masks_dir),
        "instance_mask_dtype": "uint16",
        "instance_masks_frame_count": frame_count,
    }


def _load_sam3():
    """Load SAM3 model and processor."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    ckpt = str(_SAM3_WEIGHTS_PATH) if _SAM3_WEIGHTS_PATH.is_file() else None
    if ckpt:
        _log(f"Loading SAM3 image model from local weights: {ckpt}")
    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=(ckpt is None))
    processor = Sam3Processor(model, confidence_threshold=_MIN_CONFIDENCE)
    _log(f"SAM3 loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return processor


def _load_da3():
    """Load Depth Anything v3 metric depth model."""
    from depth_anything_3.api import DepthAnything3

    if _DA3_MODEL_PATH.exists():
        model_source = str(_DA3_MODEL_PATH)
        _log(f"Loading DA3 from local path: {model_source} (model_name={_DA3_MODEL_NAME})")
        model = DepthAnything3.from_pretrained(model_source, model_name=_DA3_MODEL_NAME)
    else:
        model_source = _DA3_MODEL_ID
        _log(f"Loading DA3 from hub id: {model_source} (model_name={_DA3_MODEL_NAME})")
        model = DepthAnything3.from_pretrained(model_source, model_name=_DA3_MODEL_NAME)
    model = model.to(device=torch.device("cuda"))
    model.eval()
    _log(f"DA3-Metric loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model


def _get_metric_depth(da3_model, image, focal_px: float) -> np.ndarray:
    """Get per-pixel metric depth in meters from DA3.

    Args:
        da3_model: Loaded DA3 model
        image: PIL Image
        focal_px: Focal length in pixels (from COLMAP or estimate)

    Returns:
        depth_map: numpy array (H, W) in meters
    """
    with torch.no_grad():
        pred = da3_model.inference([image])
        raw_depth = pred.depth[0]  # shape (proc_H, proc_W)

        # Convert to metric depth: metric_depth = focal * raw / 300.0
        metric_depth = focal_px * raw_depth / 300.0

        # Resize to original image dimensions
        from PIL import Image as PILImage
        w, h = image.size
        if metric_depth.shape != (h, w):
            depth_img = PILImage.fromarray(metric_depth)
            depth_img = depth_img.resize((w, h), PILImage.BILINEAR)
            metric_depth = np.array(depth_img)

    return metric_depth


def _normalize_prompts(raw_prompts: List[Any]) -> List[str]:
    seen: set[str] = set()
    prompts: List[str] = []
    for raw in raw_prompts:
        label = str(raw or "").strip().lower()
        if not label:
            continue
        label = " ".join(label.replace("_", " ").replace("-", " ").split())
        if not label or label in _STRUCTURAL_LABELS:
            continue
        if label not in seen:
            seen.add(label)
            prompts.append(label)
    return prompts


def _parse_prompt_payload(raw_payload: str) -> List[str]:
    payload_text = (raw_payload or "").strip()
    if not payload_text:
        return []

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return _normalize_prompts(payload_text.split(","))

    if isinstance(payload, list):
        if payload and all(isinstance(item, Mapping) for item in payload):
            labels = []
            for item in payload:
                if not isinstance(item, Mapping):
                    continue
                labels.append(
                    item.get("label")
                    or item.get("object")
                    or item.get("object_id")
                    or item.get("name")
                    or item.get("class_name")
                )
            return _normalize_prompts(labels)
        return _normalize_prompts(payload)

    if isinstance(payload, Mapping):
        for key in ("prompts", "labels", "target_labels", "objects"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                if candidate and all(isinstance(item, Mapping) for item in candidate):
                    labels = []
                    for item in candidate:
                        if not isinstance(item, Mapping):
                            continue
                        labels.append(
                            item.get("label")
                            or item.get("object")
                            or item.get("object_id")
                            or item.get("name")
                            or item.get("class_name")
                        )
                    return _normalize_prompts(labels)
                return _normalize_prompts(candidate)

    return []


def _resolve_detection_prompts(
    *,
    environment: str,
    frames_dir: Path,
    all_frames: List[Path],
) -> Tuple[List[str], str]:
    override = _parse_prompt_payload(os.getenv("SAM3_DETECTION_PROMPTS", ""))
    if override:
        return override, "env:SAM3_DETECTION_PROMPTS"

    if _PROMPT_INFERENCE_COMMAND:
        keyframe_count = max(6, min(18, len(all_frames) // 20 if len(all_frames) > 0 else 6))
        keyframes = _sample_frame_paths(frames_dir, keyframe_count)
        output_hint = frames_dir / "_prompt_inference_output.json"
        if output_hint.exists():
            output_hint.unlink()

        command_template = _PROMPT_INFERENCE_COMMAND
        try:
            command = command_template.format(
                frames_dir=shlex.quote(str(frames_dir)),
                environment=shlex.quote(environment),
                keyframes=" ".join(shlex.quote(str(path)) for path in keyframes),
                keyframes_csv=shlex.quote(",".join(str(path) for path in keyframes)),
                keyframes_json=shlex.quote(json.dumps([str(path) for path in keyframes])),
                output_json=shlex.quote(str(output_hint)),
            )
        except Exception:
            command = command_template

        _log(
            f"Running PROMPT_INFERENCE_COMMAND on {len(keyframes)} keyframes "
            f"(timeout={_PROMPT_INFERENCE_TIMEOUT_SEC}s)"
        )
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=_PROMPT_INFERENCE_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                prompts = _parse_prompt_payload(result.stdout)
                if not prompts and output_hint.exists():
                    prompts = _parse_prompt_payload(output_hint.read_text(encoding="utf-8"))
                if prompts:
                    _log(f"Prompt inference produced {len(prompts)} prompts")
                    return prompts, "command:PROMPT_INFERENCE_COMMAND"
                _log("Prompt inference returned no prompts; using fallback prompts")
            else:
                stderr_lines = (result.stderr or "").strip().splitlines()
                tail = stderr_lines[-1] if stderr_lines else f"exit={result.returncode}"
                _log(f"Prompt inference command failed ({tail}); using fallback prompts")
        except subprocess.TimeoutExpired:
            _log("Prompt inference command timed out; using fallback prompts")
        except Exception as exc:
            _log(f"Prompt inference command errored ({exc}); using fallback prompts")

    env = environment.strip().lower()
    if env in _DETECTION_PROMPTS and env != "auto":
        return list(_DETECTION_PROMPTS[env]), f"environment:{env}"
    return list(_AUTO_FALLBACK_PROMPTS), "auto_fallback"


def _resolve_persistence_thresholds(environment: str, min_frame_detections: int) -> Tuple[int, int]:
    env = environment.strip().lower()
    default_min_frames = 3 if env == "auto" else max(2, min_frame_detections)
    default_min_total = 3 if env == "auto" else 2
    min_frames = _env_int("SAM3_MIN_TRACK_FRAMES", default_min_frames)
    min_total = _env_int("SAM3_MIN_TOTAL_DETECTIONS", default_min_total)
    return max(1, min_frames), max(1, min_total)


def _sample_frame_paths(frames_dir: Path, n_samples: int) -> List[Path]:
    """Select evenly-spaced frames from the directory."""
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No image files in {frames_dir}")

    if len(frames) <= n_samples:
        return frames

    indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    return [frames[i] for i in indices]


def _resolve_sampling_settings(
    *,
    environment: str,
    total_frames: int,
    requested_n_frames: int,
    requested_min_frame_detections: int,
) -> tuple[int, int]:
    """Resolve sampling/filter defaults for robust multi-frame detection."""
    env = environment.strip().lower()
    if env == "warehouse":
        auto_n_frames = 12
        auto_min_detections = 2
    elif env == "kitchen":
        auto_n_frames = 10
        auto_min_detections = 2
    elif env == "bedroom":
        auto_n_frames = 12
        auto_min_detections = 2
    elif env == "auto":
        auto_n_frames = 14
        auto_min_detections = 3
    else:
        auto_n_frames = _DEFAULT_SAMPLE_FRAMES
        auto_min_detections = 2

    if total_frames > 0:
        auto_n_frames = max(auto_n_frames, min(32, max(8, total_frames // 10)))

    n_frames = requested_n_frames if requested_n_frames > 0 else auto_n_frames
    min_frame_detections = (
        requested_min_frame_detections
        if requested_min_frame_detections > 0
        else auto_min_detections
    )

    if total_frames > 0:
        n_frames = max(1, min(n_frames, total_frames))
    min_frame_detections = max(1, min_frame_detections)
    return n_frames, min_frame_detections


def _resolve_tracking_mode(requested_mode: str, total_frames: int) -> tuple[str, str]:
    mode = (requested_mode or "").strip().lower()
    if mode in {"full_video", "sampled"}:
        return mode, f"requested={mode}"
    if mode == "auto" or not mode:
        resolved = "full_video" if total_frames <= _SAM3_FULL_VIDEO_MAX_FRAMES else "sampled"
        reason = (
            "requested=auto "
            f"(total_frames={total_frames} threshold={_SAM3_FULL_VIDEO_MAX_FRAMES} -> {resolved})"
        )
        return resolved, reason
    resolved = "full_video" if total_frames <= _SAM3_FULL_VIDEO_MAX_FRAMES else "sampled"
    return resolved, (
        f"requested={mode} unsupported "
        f"(fallback=auto threshold={_SAM3_FULL_VIDEO_MAX_FRAMES} -> {resolved})"
    )


def _detect_objects_in_frame(
    processor,
    image_path: Path,
    prompts: List[str],
    depth_map: Optional[np.ndarray] = None,
    focal_px: float = 1000.0,
    crops_dir: Optional[Path] = None,
    frame_idx: int = 0,
    include_mask: bool = False,
) -> List[Dict[str, Any]]:
    """Run SAM3 detection on a single frame for all prompts.

    If depth_map is provided (from DA3), computes accurate 3D bounding
    boxes by masking the metric depth with SAM3 segmentation masks.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    state = processor.set_image(img)
    detections = []

    for prompt in prompts:
        processor.reset_all_prompts(state)
        result = processor.set_text_prompt(state=state, prompt=prompt)

        masks = result.get("masks")
        scores = result.get("scores")
        boxes = result.get("boxes")

        if masks is None or scores is None:
            continue

        n = masks.shape[0] if hasattr(masks, "shape") and len(masks.shape) >= 1 else 0
        for i in range(n):
            score = float(scores[i])
            if score < _MIN_CONFIDENCE:
                continue

            box = boxes[i].tolist() if boxes is not None else [0, 0, w, h]
            mask_np = masks[i].squeeze().cpu().numpy().astype(bool) if masks is not None else None

            # Compute mask centroid if available
            if mask_np is not None and mask_np.any():
                ys, xs = np.where(mask_np)
                cx, cy = float(xs.mean()), float(ys.mean())
                mask_area = int(mask_np.sum())
            else:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                mask_area = int((box[2] - box[0]) * (box[3] - box[1]))

            det = {
                "label": prompt,
                "score": score,
                "box": box,  # [x1, y1, x2, y2]
                "centroid_px": [cx, cy],
                "mask_area_px": mask_area,
                "image_size": [w, h],
                "frame_path": str(image_path),
                "frame_idx": int(frame_idx),
            }
            if include_mask and mask_np is not None:
                det["_mask_np"] = mask_np

            # If we have depth, compute 3D extent from mask + depth
            if depth_map is not None and mask_np is not None and mask_np.any():
                # Resize mask to depth map size if needed
                mask_for_depth = mask_np
                if mask_np.shape != depth_map.shape:
                    from PIL import Image as PILImg
                    mask_pil = PILImg.fromarray(mask_np.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize(
                        (depth_map.shape[1], depth_map.shape[0]),
                        PILImg.NEAREST,
                    )
                    mask_for_depth = np.array(mask_pil) > 127

                object_depths = depth_map[mask_for_depth]
                if len(object_depths) > 0:
                    median_depth = float(np.median(object_depths))
                    depth_range = float(np.percentile(object_depths, 90) -
                                       np.percentile(object_depths, 10))

                    # Convert 2D extent to 3D using depth + focal length
                    box_w_px = box[2] - box[0]
                    box_h_px = box[3] - box[1]
                    width_m = box_w_px * median_depth / focal_px
                    height_m = box_h_px * median_depth / focal_px
                    depth_m = max(depth_range, min(width_m, height_m) * 0.3)

                    # 3D center from 2D centroid + depth
                    cx_3d = (cx - w / 2) * median_depth / focal_px
                    cy_3d = (h / 2 - cy) * median_depth / focal_px
                    cz_3d = median_depth

                    det["depth_3d"] = {
                        "center": [round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4)],
                        "extents": [
                            round(max(0.02, width_m), 4),
                            round(max(0.02, height_m), 4),
                            round(max(0.02, depth_m), 4),
                        ],
                        "median_depth_m": round(median_depth, 4),
                        "depth_range_m": round(depth_range, 4),
                    }

            # Save masked RGBA crop if crops_dir is set
            if crops_dir is not None and mask_np is not None and mask_np.any():
                crop_path = _save_object_crop(
                    image_path, mask_np, box, crops_dir,
                    prompt, frame_idx, len(detections),
                )
                if crop_path is not None:
                    det["crop_path"] = crop_path.as_posix()

            detections.append(det)

    return detections


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _box_area(box: List[float]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _detections_match(det_a: Dict[str, Any], det_b: Dict[str, Any]) -> bool:
    """Cross-frame association using IoU + depth + centroid/scale cues."""
    iou = _box_iou(det_a["box"], det_b["box"])
    if iou >= _MERGE_IOU_THRESHOLD:
        return True

    area_a = _box_area(det_a["box"])
    area_b = _box_area(det_b["box"])
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    if area_ratio > 4.0:
        return False

    centroid_a = det_a.get("centroid_px", [0.0, 0.0])
    centroid_b = det_b.get("centroid_px", [0.0, 0.0])
    dx = float(centroid_a[0]) - float(centroid_b[0])
    dy = float(centroid_a[1]) - float(centroid_b[1])
    center_dist = float(np.hypot(dx, dy))
    diag_a = float(np.hypot(det_a["box"][2] - det_a["box"][0], det_a["box"][3] - det_a["box"][1]))
    diag_b = float(np.hypot(det_b["box"][2] - det_b["box"][0], det_b["box"][3] - det_b["box"][1]))
    center_dist_norm = center_dist / max(1.0, diag_a, diag_b)

    depth_a = det_a.get("depth_3d") if isinstance(det_a.get("depth_3d"), dict) else None
    depth_b = det_b.get("depth_3d") if isinstance(det_b.get("depth_3d"), dict) else None
    if depth_a is not None and depth_b is not None:
        center_a = np.array(depth_a.get("center", [0.0, 0.0, 0.0]), dtype=float)
        center_b = np.array(depth_b.get("center", [0.0, 0.0, 0.0]), dtype=float)
        ext_a = np.array(depth_a.get("extents", [0.2, 0.2, 0.2]), dtype=float)
        ext_b = np.array(depth_b.get("extents", [0.2, 0.2, 0.2]), dtype=float)

        dist_3d = float(np.linalg.norm(center_a - center_b))
        size_ref = max(float(np.max(ext_a)), float(np.max(ext_b)), 0.25)
        depth_gap = abs(float(center_a[2] - center_b[2]))

        if dist_3d <= max(0.6, 1.25 * size_ref) and center_dist_norm <= 1.2:
            return True
        if iou >= 0.15 and depth_gap <= max(0.8, 1.5 * size_ref):
            return True
        return False

    return iou >= 0.2 or (iou >= 0.1 and center_dist_norm <= 0.5 and area_ratio <= 2.5)


def _suppress_frame_duplicates(frame_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply per-frame duplicate suppression per label."""
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in frame_detections:
        by_label[str(det.get("label") or "object")].append(det)

    kept: List[Dict[str, Any]] = []
    for _, label_dets in by_label.items():
        selected: List[Dict[str, Any]] = []
        for det in sorted(label_dets, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            duplicate = False
            for prior in selected:
                iou = _box_iou(det["box"], prior["box"])
                if iou >= 0.65:
                    duplicate = True
                    break
                c0 = det.get("centroid_px", [0.0, 0.0])
                c1 = prior.get("centroid_px", [0.0, 0.0])
                d = float(np.hypot(float(c0[0]) - float(c1[0]), float(c0[1]) - float(c1[1])))
                diag = max(
                    1.0,
                    float(np.hypot(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1])),
                    float(np.hypot(prior["box"][2] - prior["box"][0], prior["box"][3] - prior["box"][1])),
                )
                if d / diag <= 0.2:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
        kept.extend(selected)
    return kept


def _track_association_score(track: Dict[str, Any], det: Dict[str, Any]) -> float:
    """Return [0,1] association score for assigning ``det`` to ``track``."""
    last = track["last_det"]
    frame_gap = int(det.get("frame_idx", 0)) - int(track.get("last_frame_idx", -1))
    if frame_gap <= 0 or frame_gap > _TRACK_MAX_FRAME_GAP:
        return -1.0

    iou = _box_iou(last["box"], det["box"])
    area_last = _box_area(last["box"])
    area_det = _box_area(det["box"])
    area_ratio = max(area_last, area_det) / max(1.0, min(area_last, area_det))
    if area_ratio > 6.0:
        return -1.0

    c0 = last.get("centroid_px", [0.0, 0.0])
    c1 = det.get("centroid_px", [0.0, 0.0])
    center_dist = float(np.hypot(float(c0[0]) - float(c1[0]), float(c0[1]) - float(c1[1])))
    diag_last = float(np.hypot(last["box"][2] - last["box"][0], last["box"][3] - last["box"][1]))
    diag_det = float(np.hypot(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]))
    center_dist_norm = center_dist / max(1.0, diag_last, diag_det)
    max_center_norm = 1.1 + (0.4 * float(max(0, frame_gap - 1)))
    center_score = max(0.0, 1.0 - (center_dist_norm / max(1e-6, max_center_norm)))

    size_score = max(0.0, 1.0 - min(6.0, area_ratio) / 6.0)
    depth_score = 0.0

    depth_a = last.get("depth_3d") if isinstance(last.get("depth_3d"), dict) else None
    depth_b = det.get("depth_3d") if isinstance(det.get("depth_3d"), dict) else None
    if depth_a is not None and depth_b is not None:
        center_a = np.array(depth_a.get("center", [0.0, 0.0, 0.0]), dtype=float)
        center_b = np.array(depth_b.get("center", [0.0, 0.0, 0.0]), dtype=float)
        ext_a = np.array(depth_a.get("extents", [0.2, 0.2, 0.2]), dtype=float)
        ext_b = np.array(depth_b.get("extents", [0.2, 0.2, 0.2]), dtype=float)

        # Use horizontal size as primary scale reference (vertical can be noisy).
        size_ref = max(
            float(max(ext_a[0], ext_a[2])),
            float(max(ext_b[0], ext_b[2])),
            0.25,
        )
        dist_3d = float(np.linalg.norm(center_a - center_b))
        depth_gap = abs(float(center_a[2] - center_b[2]))
        max_dist = max(1.1, (2.1 * size_ref) + (0.35 * float(max(0, frame_gap - 1))))
        if dist_3d > max_dist or depth_gap > max(0.9, 1.5 * size_ref):
            return -1.0
        depth_score = max(0.0, 1.0 - (dist_3d / max_dist))
    elif center_dist_norm > max_center_norm and iou < 0.05:
        return -1.0

    score = (0.45 * iou) + (0.25 * center_score) + (0.2 * depth_score) + (0.1 * size_score)
    if _detections_match(last, det):
        score = max(score, 0.33)
    score -= min(0.2, 0.06 * float(max(0, frame_gap - 1)))
    return float(max(0.0, min(1.0, score)))


def _track_label_detections(label_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build temporal tracks for detections of a single label."""
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for det in label_dets:
        by_frame[int(det.get("frame_idx", 0))].append(det)

    tracks: List[Dict[str, Any]] = []
    for frame_idx in sorted(by_frame.keys()):
        frame_dets = _suppress_frame_duplicates(by_frame[frame_idx])
        used_track_ids: set[int] = set()
        for det in sorted(frame_dets, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            best_track_idx = -1
            best_score = _TRACK_MIN_ASSOC_SCORE
            for track_idx, track in enumerate(tracks):
                if track_idx in used_track_ids:
                    continue
                score = _track_association_score(track, det)
                if score > best_score:
                    best_score = score
                    best_track_idx = track_idx

            if best_track_idx >= 0:
                track = tracks[best_track_idx]
                track["detections"].append(det)
                track["last_det"] = det
                track["last_frame_idx"] = frame_idx
                track["frame_indices"].add(frame_idx)
                used_track_ids.add(best_track_idx)
            else:
                tracks.append(
                    {
                        "detections": [det],
                        "last_det": det,
                        "last_frame_idx": frame_idx,
                        "frame_indices": {frame_idx},
                    }
                )
                used_track_ids.add(len(tracks) - 1)

    return tracks


def _track_prototype(track: Dict[str, Any]) -> Dict[str, Any]:
    dets = track.get("detections", [])
    centers = []
    extents = []
    centroids = []
    areas = []
    for det in dets:
        centroids.append(det.get("centroid_px", [0.0, 0.0]))
        areas.append(_box_area(det["box"]))
        depth = det.get("depth_3d") if isinstance(det.get("depth_3d"), dict) else None
        if depth is not None:
            centers.append(depth.get("center", [0.0, 0.0, 0.0]))
            extents.append(depth.get("extents", [0.2, 0.2, 0.2]))

    out: Dict[str, Any] = {
        "frame_indices": set(track.get("frame_indices", set())),
        "centroid_mean": [0.0, 0.0],
        "area_median": 0.0,
    }
    if centroids:
        c = np.array(centroids, dtype=float)
        out["centroid_mean"] = [float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))]
    if areas:
        out["area_median"] = float(np.median(np.array(areas, dtype=float)))
    if centers and extents:
        out["depth_center"] = np.median(np.array(centers, dtype=float), axis=0)
        out["depth_extents"] = np.median(np.array(extents, dtype=float), axis=0)
    return out


def _tracks_mergeable(track_a: Dict[str, Any], track_b: Dict[str, Any]) -> bool:
    proto_a = _track_prototype(track_a)
    proto_b = _track_prototype(track_b)
    frames_a = proto_a.get("frame_indices", set())
    frames_b = proto_b.get("frame_indices", set())
    overlap = set(frames_a).intersection(set(frames_b))
    if overlap:
        # If both tracks are present in same frame(s), treat as distinct objects.
        return False

    depth_center_a = proto_a.get("depth_center")
    depth_center_b = proto_b.get("depth_center")
    depth_ext_a = proto_a.get("depth_extents")
    depth_ext_b = proto_b.get("depth_extents")
    if depth_center_a is not None and depth_center_b is not None:
        center_a = np.array(depth_center_a, dtype=float)
        center_b = np.array(depth_center_b, dtype=float)
        ext_a = np.array(depth_ext_a, dtype=float) if depth_ext_a is not None else np.array([0.2, 0.2, 0.2])
        ext_b = np.array(depth_ext_b, dtype=float) if depth_ext_b is not None else np.array([0.2, 0.2, 0.2])
        size_ref = max(float(max(ext_a[0], ext_a[2])), float(max(ext_b[0], ext_b[2])), 0.3)
        dist_3d = float(np.linalg.norm(center_a - center_b))
        depth_gap = abs(float(center_a[2] - center_b[2]))
        ext_ratio = max(float(np.max(ext_a)), float(np.max(ext_b))) / max(0.05, min(float(np.max(ext_a)), float(np.max(ext_b))))
        return (
            dist_3d <= max(1.2, 1.8 * size_ref)
            and depth_gap <= max(1.0, 1.4 * size_ref)
            and ext_ratio <= 3.5
        )

    c_a = np.array(proto_a.get("centroid_mean", [0.0, 0.0]), dtype=float)
    c_b = np.array(proto_b.get("centroid_mean", [0.0, 0.0]), dtype=float)
    center_dist = float(np.linalg.norm(c_a - c_b))
    area_a = float(proto_a.get("area_median", 0.0))
    area_b = float(proto_b.get("area_median", 0.0))
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    return center_dist <= 80.0 and area_ratio <= 3.0


def _merge_tracklets(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge non-overlapping short track fragments likely from same object."""
    merged = [dict(track) for track in tracks]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            if changed:
                break
            for j in range(i + 1, len(merged)):
                if not _tracks_mergeable(merged[i], merged[j]):
                    continue
                merged[i]["detections"].extend(merged[j]["detections"])
                merged[i]["detections"].sort(key=lambda det: int(det.get("frame_idx", 0)))
                merged[i]["frame_indices"] = set(int(det.get("frame_idx", 0)) for det in merged[i]["detections"])
                merged[i]["last_det"] = merged[i]["detections"][-1]
                merged[i]["last_frame_idx"] = int(merged[i]["last_det"].get("frame_idx", 0))
                del merged[j]
                changed = True
                break
    return merged


def _reference_quality(det: Dict[str, Any]) -> float:
    score = float(det.get("score", 0.0))
    area = max(1.0, _box_area(det["box"]))
    image_size = det.get("image_size", [1, 1])
    image_area = max(1.0, float(image_size[0]) * float(image_size[1]))
    coverage = min(1.0, area / image_area)

    cx, cy = det.get("centroid_px", [0.0, 0.0])
    center_x = float(image_size[0]) * 0.5
    center_y = float(image_size[1]) * 0.5
    dist = float(np.hypot(float(cx) - center_x, float(cy) - center_y))
    center_norm = dist / max(1.0, float(np.hypot(center_x, center_y)))
    center_score = max(0.0, 1.0 - center_norm)
    return (0.65 * score) + (0.25 * np.sqrt(coverage)) + (0.10 * center_score)


def _select_reference_crops(cluster: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    crop_dets = [det for det in cluster if isinstance(det.get("crop_path"), str) and str(det.get("crop_path")).strip()]
    if not crop_dets:
        return None, []

    ranked = sorted(crop_dets, key=_reference_quality, reverse=True)
    seen: set[str] = set()
    ordered: List[str] = []
    for det in ranked:
        crop = str(det.get("crop_path")).strip()
        if crop and crop not in seen:
            seen.add(crop)
            ordered.append(crop)
        if len(ordered) >= _MAX_REFERENCE_CROPS:
            break
    if not ordered:
        return None, []
    return ordered[0], ordered


def _cluster_to_object(
    *,
    label: str,
    cluster_idx: int,
    cluster: List[Dict[str, Any]],
    preserve_detections: bool = False,
) -> Dict[str, Any]:
    scores = [d["score"] for d in cluster]
    boxes = [d["box"] for d in cluster]
    centroids = [d["centroid_px"] for d in cluster]
    n_frames = len(set(int(d.get("frame_idx", 0)) for d in cluster))
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))

    mean_box = [float(np.mean([b[i] for b in boxes])) for i in range(4)]
    mean_centroid = [float(np.mean([c[i] for c in centroids])) for i in range(2)]
    img_w, img_h = cluster[0]["image_size"]

    depth_3d_list = [d["depth_3d"] for d in cluster if "depth_3d" in d]
    has_depth = len(depth_3d_list) > 0
    if has_depth:
        centers = np.array([d["center"] for d in depth_3d_list])
        extents_arr = np.array([d["extents"] for d in depth_3d_list])
        cx_3d = float(np.median(centers[:, 0]))
        cy_3d = float(np.median(centers[:, 1]))
        cz_3d = float(np.median(centers[:, 2]))
        width_m = float(np.median(extents_arr[:, 0]))
        height_m = float(np.median(extents_arr[:, 1]))
        depth_m = float(np.median(extents_arr[:, 2]))
        refinement_source = "da3_metric_depth"
    else:
        box_w = mean_box[2] - mean_box[0]
        box_h = mean_box[3] - mean_box[1]
        scene_depth_est = 3.0
        scale = scene_depth_est / max(img_w, img_h)
        width_m = box_w * scale
        height_m = box_h * scale
        depth_m = min(width_m, height_m) * 0.6
        cx_3d = (mean_centroid[0] / img_w - 0.5) * scene_depth_est
        cy_3d = (0.5 - mean_centroid[1] / img_h) * scene_depth_est
        cz_3d = scene_depth_est * 0.5
        refinement_source = "heuristic_2d"

    best_crop, all_crops = _select_reference_crops(cluster)
    frame_indices = sorted(set(int(det.get("frame_idx", 0)) for det in cluster))
    frame_paths = [str(det.get("frame_path")) for det in sorted(cluster, key=lambda det: int(det.get("frame_idx", 0)))]
    unique_frame_paths = list(dict.fromkeys(frame_paths))

    obj_entry = {
        "id": f"{label}_{cluster_idx + 1}",
        "label": label,
        "confidence": round(max_score, 3),
        "mean_confidence": round(mean_score, 3),
        "n_frame_detections": n_frames,
        "n_total_detections": len(cluster),
        "frame_indices": frame_indices,
        "frame_paths": unique_frame_paths,
        "boundingBox": {
            "center": [round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4)],
            "extents": [
                round(max(0.02, width_m), 4),
                round(max(0.02, height_m), 4),
                round(max(0.02, depth_m), 4),
            ],
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "orientationQuaternion": [1, 0, 0, 0],
        },
        "mean_box_px": [round(v, 1) for v in mean_box],
        "mean_centroid_px": [round(v, 1) for v in mean_centroid],
        "image_size": [int(img_w), int(img_h)],
        "detection_source": "sam3",
        "refinement": refinement_source,
    }
    if best_crop is not None:
        obj_entry["reference_crop"] = best_crop
    if all_crops:
        obj_entry["all_crops"] = all_crops
    if preserve_detections:
        obj_entry["_cluster_detections"] = [
            {
                "frame_idx": int(det.get("frame_idx", 0)),
                "frame_path": str(det.get("frame_path") or ""),
                "score": float(det.get("score", 0.0)),
                "_mask_np": det.get("_mask_np"),
            }
            for det in cluster
            if isinstance(det, Mapping) and isinstance(det.get("_mask_np"), np.ndarray)
        ]
    return obj_entry


def _merge_detections(
    all_detections: List[Dict[str, Any]],
    *,
    preserve_detections: bool = False,
) -> List[Dict[str, Any]]:
    """Temporal association over the sampled sequence into scene-level objects."""
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in all_detections:
        by_label[det["label"]].append(det)

    merged_objects: List[Dict[str, Any]] = []

    for label, dets in by_label.items():
        if label.lower() in _STRUCTURAL_LABELS:
            continue

        tracks = _track_label_detections(dets)
        tracks = _merge_tracklets(tracks)
        tracks.sort(
            key=lambda track: (
                max(float(det.get("score", 0.0)) for det in track["detections"]),
                len(track["frame_indices"]),
            ),
            reverse=True,
        )
        for cluster_idx, track in enumerate(tracks):
            merged_objects.append(
                _cluster_to_object(
                    label=label,
                    cluster_idx=cluster_idx,
                    cluster=track["detections"],
                    preserve_detections=preserve_detections,
                )
            )

    # Sort by confidence descending
    merged_objects.sort(key=lambda x: x["confidence"], reverse=True)
    return merged_objects


def _read_colmap_cameras(cameras_bin: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP cameras.bin file."""
    import struct as st

    cameras = {}
    with open(cameras_bin, "rb") as f:
        n_cameras = st.unpack("<Q", f.read(8))[0]
        for _ in range(n_cameras):
            cam_id = st.unpack("<i", f.read(4))[0]
            model_id = st.unpack("<i", f.read(4))[0]
            width = st.unpack("<Q", f.read(8))[0]
            height = st.unpack("<Q", f.read(8))[0]

            # Number of params per model: SIMPLE_PINHOLE=3, PINHOLE=4
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
            params = st.unpack(f"<{n_params}d", f.read(8 * n_params))

            cam = {"id": cam_id, "model_id": model_id, "width": width,
                   "height": height, "params": params}
            if model_id == 1:  # PINHOLE: fx, fy, cx, cy
                cam["fx"], cam["fy"], cam["cx"], cam["cy"] = params
            elif model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
                cam["fx"] = cam["fy"] = params[0]
                cam["cx"], cam["cy"] = params[1], params[2]
            cameras[cam_id] = cam
    return cameras


def _read_colmap_images(images_bin: Path) -> List[Dict[str, Any]]:
    """Read COLMAP images.bin file (camera poses per image)."""
    import struct as st

    images = []
    with open(images_bin, "rb") as f:
        n_images = st.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            image_id = st.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = st.unpack("<4d", f.read(32))
            tx, ty, tz = st.unpack("<3d", f.read(24))
            camera_id = st.unpack("<i", f.read(4))[0]

            # Read image name (null-terminated)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)

            # Skip 2D points
            n_points = st.unpack("<Q", f.read(8))[0]
            f.read(n_points * 24)  # each: x(8) + y(8) + point3D_id(8)

            # Convert quaternion to rotation matrix
            r = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qz**2)],
            ])
            t = np.array([tx, ty, tz])
            # Camera center in world coordinates: C = -R^T * t
            center = -r.T @ t

            images.append({
                "id": image_id, "name": name, "camera_id": camera_id,
                "R": r, "t": t, "center": center,
            })
    return images


def _load_gaussian_ply(ply_path: Path) -> np.ndarray:
    """Load XYZ coordinates from a Gaussian splat PLY file."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(ply_path))
        v = ply["vertex"]
        return np.column_stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])])
    except ImportError:
        _log("plyfile not available, trying numpy-based PLY reader")
        # Simple ASCII/binary PLY reader fallback
        import struct as st
        xyz = []
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            header_str = header.decode("ascii", errors="ignore")
            n_vertices = 0
            for line in header_str.split("\n"):
                if line.startswith("element vertex"):
                    n_vertices = int(line.split()[-1])
            # Read binary little-endian floats (assuming x,y,z are first 3 floats)
            for _ in range(min(n_vertices, 500000)):
                data = f.read(4 * 3)
                if len(data) < 12:
                    break
                x, y, z = st.unpack("<3f", data)
                xyz.append([x, y, z])
                # Skip remaining properties per vertex
                remaining = f.read(max(0, 62 * 4 - 12))  # approximate
        return np.array(xyz) if xyz else np.zeros((0, 3))


def _project_points_to_image(
    points_3d: np.ndarray, R: np.ndarray, t: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates. Returns (uv, mask)."""
    # Transform to camera coordinates: p_cam = R * p_world + t
    p_cam = (R @ points_3d.T).T + t
    z = p_cam[:, 2]

    # Only keep points in front of camera
    valid = z > 0.1
    u = np.full(len(points_3d), -1.0)
    v = np.full(len(points_3d), -1.0)
    u[valid] = fx * p_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * p_cam[valid, 1] / z[valid] + cy

    # Check image bounds
    in_bounds = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.column_stack([u, v]), in_bounds


def _refine_with_colmap(
    objects: List[Dict[str, Any]],
    colmap_sparse_dir: Optional[Path],
    gaussian_ply_path: Optional[Path] = None,
    all_detections: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Refine 3D bounding boxes using COLMAP cameras + Gaussian PLY.

    Back-projects SAM3 2D masks through COLMAP camera poses into the
    3D Gaussian point cloud to extract accurate oriented bounding boxes.
    """
    if colmap_sparse_dir is None or not colmap_sparse_dir.exists():
        _log("No COLMAP sparse dir provided, using heuristic 3D estimates")
        return objects

    cameras_bin = colmap_sparse_dir / "cameras.bin"
    images_bin = colmap_sparse_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        _log("COLMAP cameras.bin or images.bin not found, using heuristic estimates")
        return objects

    try:
        cameras = _read_colmap_cameras(cameras_bin)
        images = _read_colmap_images(images_bin)
        _log(f"Loaded {len(cameras)} cameras, {len(images)} images from COLMAP")

        if not cameras or not images:
            return objects

        # Get the first camera's intrinsics (single camera assumption)
        cam = list(cameras.values())[0]
        fx = cam.get("fx", 1000)
        fy = cam.get("fy", 1000)
        cx = cam.get("cx", cam["width"] / 2)
        cy = cam.get("cy", cam["height"] / 2)
        img_w, img_h = cam["width"], cam["height"]
        _log(f"Camera: {img_w}x{img_h}, fx={fx:.1f}, fy={fy:.1f}")

        # Build image name → pose lookup
        name_to_pose = {img["name"]: img for img in images}

        # Load Gaussian point cloud if available
        points_3d = None
        if gaussian_ply_path and gaussian_ply_path.exists():
            _log(f"Loading Gaussian PLY for back-projection: {gaussian_ply_path}")
            points_3d = _load_gaussian_ply(gaussian_ply_path)
            _log(f"  Loaded {len(points_3d)} Gaussians")

            # Subsample for performance if very large
            if len(points_3d) > 200000:
                indices = np.random.choice(len(points_3d), 200000, replace=False)
                points_3d = points_3d[indices]
                _log(f"  Subsampled to {len(points_3d)} points")

        # For each object, find its 3D extent by back-projecting its 2D
        # bounding box through all cameras that see it, and intersecting
        # with the Gaussian point cloud
        for obj in objects:
            box_px = obj.get("mean_box_px", [0, 0, 100, 100])
            det_img_w, det_img_h = 1, 1
            # Get detection image size for coordinate scaling
            if obj.get("n_total_detections", 0) > 0:
                # mean_box_px is in detection image coordinates (original frame)
                # Need to scale to COLMAP image coordinates
                pass

            # Scale 2D box from detection resolution to COLMAP resolution
            # Detection was done on original frames, COLMAP uses undistorted frames
            # which may have different resolution
            orig_w = obj.get("_det_img_w", img_w)
            orig_h = obj.get("_det_img_h", img_h)
            scale_x = img_w / max(1, orig_w) if orig_w != img_w else 1.0
            scale_y = img_h / max(1, orig_h) if orig_h != img_h else 1.0

            x1 = box_px[0] * scale_x
            y1 = box_px[1] * scale_y
            x2 = box_px[2] * scale_x
            y2 = box_px[3] * scale_y

            # Skip Gaussian back-projection if DA3 already gave us metric 3D
            if obj.get("refinement") == "da3_metric_depth":
                _log(f"  {obj['id']}: using DA3 metric depth (skipping Gaussian backprojection)")
                continue

            if points_3d is not None and len(points_3d) > 0:
                # For each COLMAP image, project all 3D points and check
                # which fall inside the 2D bounding box. Take the union.
                selected_mask = np.zeros(len(points_3d), dtype=bool)
                n_views = 0

                # Use a tighter box (shrink by 10%) to avoid selecting background
                pad_x = (x2 - x1) * 0.1
                pad_y = (y2 - y1) * 0.1
                x1_tight = x1 + pad_x
                y1_tight = y1 + pad_y
                x2_tight = x2 - pad_x
                y2_tight = y2 - pad_y

                for img_pose in images[:20]:  # Limit to 20 views for speed
                    uv, in_bounds = _project_points_to_image(
                        points_3d, img_pose["R"], img_pose["t"],
                        fx, fy, cx, cy, img_w, img_h,
                    )
                    # Points that project inside the tightened 2D bounding box
                    in_box = (
                        in_bounds &
                        (uv[:, 0] >= x1_tight) & (uv[:, 0] <= x2_tight) &
                        (uv[:, 1] >= y1_tight) & (uv[:, 1] <= y2_tight)
                    )
                    if in_box.any():
                        selected_mask |= in_box
                        n_views += 1

                n_selected = int(selected_mask.sum())
                if n_selected >= 10:
                    obj_points = points_3d[selected_mask]

                    # Compute OBB from selected points
                    center = obj_points.mean(axis=0)
                    # Use PCA for oriented bounding box
                    cov = np.cov(obj_points.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Sort by eigenvalue descending
                    order = eigenvalues.argsort()[::-1]
                    axes = eigenvectors[:, order].T

                    # Project onto principal axes for extents
                    local = (obj_points - center) @ axes.T
                    extents = local.max(axis=0) - local.min(axis=0)

                    obj["boundingBox"] = {
                        "center": [round(float(c), 4) for c in center],
                        "extents": [round(max(0.02, float(e)), 4) for e in extents],
                        "axes": [[round(float(v), 6) for v in ax] for ax in axes],
                        "orientationQuaternion": [1, 0, 0, 0],  # TODO: compute from axes
                    }
                    obj["n_gaussian_points"] = n_selected
                    obj["n_views_matched"] = n_views
                    obj["refinement"] = "gaussian_backprojection"

                    bb = obj["boundingBox"]
                    _log(f"  {obj['id']}: {n_selected} Gaussians from {n_views} views → "
                         f"{bb['extents'][0]:.3f}x{bb['extents'][1]:.3f}x{bb['extents'][2]:.3f}m")
                    continue

            # Fallback: use focal length for depth estimation
            box_w_px = x2 - x1
            box_h_px = y2 - y1
            if box_w_px > 10:
                # Use median camera distance as reference
                cam_centers = np.array([img["center"] for img in images])
                scene_center = cam_centers.mean(axis=0)
                median_dist = float(np.median(np.linalg.norm(cam_centers - scene_center, axis=1)))

                # Estimate real-world size from pixel size + focal length + estimated depth
                est_depth = median_dist * 0.8
                width_m = box_w_px * est_depth / fx
                height_m = box_h_px * est_depth / fy
                depth_m = min(width_m, height_m) * 0.6

                cx_3d = (((x1 + x2) / 2) - cx) * est_depth / fx
                cy_3d = (cy - ((y1 + y2) / 2)) * est_depth / fy

                obj["boundingBox"]["center"] = [round(cx_3d, 4), round(cy_3d, 4), round(est_depth, 4)]
                obj["boundingBox"]["extents"] = [
                    round(max(0.05, width_m), 4),
                    round(max(0.05, height_m), 4),
                    round(max(0.05, depth_m), 4),
                ]
                obj["refinement"] = "focal_length_estimate"

        _log(f"Refined {len(objects)} objects with COLMAP + Gaussian data")
    except Exception as e:
        _log(f"COLMAP refinement failed: {e}, using heuristic estimates")
        import traceback
        traceback.print_exc()

    return objects


# ---------------------------------------------------------------------------
# Video predictor backend: persistent session-based tracking
# ---------------------------------------------------------------------------

def _load_video_predictor():
    """Load SAM3 video predictor model."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from sam3.model_builder import build_sam3_video_predictor

    ckpt = str(_SAM3_WEIGHTS_PATH) if _SAM3_WEIGHTS_PATH.is_file() else None
    if ckpt:
        _log(f"Loading SAM3 video predictor from local weights: {ckpt}")
    predictor = build_sam3_video_predictor(checkpoint_path=ckpt, load_from_HF=(ckpt is None))
    _log(f"SAM3 video predictor loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return predictor


def _collect_propagation_results(
    predictor,
    session_id: str,
    label: str,
    obj_id_to_label: Dict[int, str],
    all_frame_files: List[Path],
    img_w: int,
    img_h: int,
    n_frames: int,
    obj_frames: Dict[int, List[Dict[str, Any]]],
    seed_frame_idx: int = 0,
    instance_masks_dir: Optional[Path] = None,
    global_obj_id_map: Optional[Dict[int, int]] = None,
) -> None:
    """Propagate from seed frame and collect per-object per-frame results.

    Propagates forward from seed_frame_idx, then backward from seed_frame_idx
    if the seed is not frame 0, to cover the entire video.
    """
    directions = [("forward", seed_frame_idx)]
    if seed_frame_idx > 0:
        directions.append(("backward", seed_frame_idx))

    for direction, start_idx in directions:
        _log(f"    Propagating {direction} from frame {start_idx}")
        frame_idx = start_idx
        try:
            for frame_result in predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction=direction,
                start_frame_idx=start_idx,
                max_frame_num_to_track=n_frames,
            ):
                frame_idx = int(frame_result.get("frame_index", 0))
                outputs = frame_result.get("outputs", frame_result)

                out_obj_ids = outputs.get("out_obj_ids", np.array([]))
                out_probs = outputs.get("out_probs", np.array([]))
                out_boxes_xywh = outputs.get("out_boxes_xywh", np.array([]))
                out_masks = outputs.get("out_binary_masks", np.array([]))

                if hasattr(out_obj_ids, "tolist"):
                    out_obj_ids = out_obj_ids.tolist()

                for idx_in_batch, oid in enumerate(out_obj_ids):
                    oid = int(oid)
                    if oid not in obj_id_to_label:
                        continue

                    prob = float(out_probs[idx_in_batch]) if idx_in_batch < len(out_probs) else 0.0

                    # box_xywh is normalized [0,1] — convert to pixel coords [x1,y1,x2,y2]
                    if idx_in_batch < len(out_boxes_xywh):
                        bx, by, bw, bh = out_boxes_xywh[idx_in_batch]
                        box_px = [
                            float(bx) * img_w,
                            float(by) * img_h,
                            float(bx + bw) * img_w,
                            float(by + bh) * img_h,
                        ]
                    else:
                        box_px = [0, 0, img_w, img_h]

                    # Binary mask
                    mask_np = None
                    if idx_in_batch < len(out_masks):
                        mask_np = out_masks[idx_in_batch]
                        if hasattr(mask_np, "cpu"):
                            mask_np = mask_np.cpu().numpy()
                        if mask_np.ndim == 3:
                            mask_np = mask_np.squeeze(0)
                        mask_np = mask_np.astype(bool)

                    # Centroid from mask or box
                    if mask_np is not None and mask_np.any():
                        ys, xs = np.where(mask_np)
                        cx, cy = float(xs.mean()), float(ys.mean())
                        mask_area = int(mask_np.sum())
                    else:
                        cx = (box_px[0] + box_px[2]) / 2
                        cy = (box_px[1] + box_px[3]) / 2
                        mask_area = int((box_px[2] - box_px[0]) * (box_px[3] - box_px[1]))

                    frame_data: Dict[str, Any] = {
                        "frame_idx": frame_idx,
                        "prob": prob,
                        "box": box_px,
                        "centroid_px": [cx, cy],
                        "mask_area_px": mask_area,
                        "image_size": [img_w, img_h],
                    }
                    if mask_np is not None:
                        frame_data["_mask_np"] = mask_np
                    if frame_idx < len(all_frame_files):
                        frame_data["frame_path"] = str(all_frame_files[frame_idx])

                    # Save instance mask contribution if enabled
                    if instance_masks_dir is not None and mask_np is not None and global_obj_id_map is not None:
                        global_id = global_obj_id_map.get(oid)
                        if global_id is not None and frame_idx < len(all_frame_files):
                            _accumulate_instance_mask(
                                mask_np, global_id,
                                all_frame_files[frame_idx].stem,
                                instance_masks_dir,
                            )

                    obj_frames[oid].append(frame_data)

                if (frame_idx + 1) % 100 == 0:
                    _log(f"    Propagated {direction} through frame {frame_idx + 1}/{n_frames}")

        except Exception as exc:
            _log(f"    Propagation {direction} error at frame {frame_idx}: {exc}")
            import traceback
            traceback.print_exc()


def _detect_with_video_predictor(
    frames_dir: Path,
    prompts: List[str],
    *,
    save_crops: bool = True,
    crops_dir: Optional[Path] = None,
    instance_masks_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run SAM3 video predictor for persistent object tracking.

    SAM3's add_prompt resets state each time (it's a semantic prompt model),
    so we run each text prompt as a separate cycle:
      prompt → propagate → collect results → reset → next prompt

    Each prompt may detect multiple instances (obj_ids) which are tracked
    persistently through the video via the 7-frame sliding memory window.

    Returns:
        (objects, metadata) where objects is a list of object dicts compatible
        with the orchestrator contract, and metadata has tracking stats.
    """
    predictor = _load_video_predictor()

    # Count frames and get image dimensions
    all_frame_files = sorted(frames_dir.glob("*.jpg"))
    n_frames = len(all_frame_files)
    if not all_frame_files:
        _log("ERROR: No .jpg frames found in frames_dir")
        predictor.shutdown()
        del predictor
        torch.cuda.empty_cache()
        return [], {"n_objects_detected": 0, "n_frames": 0}

    from PIL import Image as PILImage
    first_frame = PILImage.open(all_frame_files[0])
    img_w, img_h = first_frame.size
    del first_frame

    # Start session — loads all frames into VRAM
    _log(f"Starting video predictor session on {frames_dir} ({n_frames} frames)")
    session_result = predictor.start_session(resource_path=str(frames_dir))
    session_id = session_result["session_id"]
    _log(f"Session {session_id[:8]}... started. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Prepare instance mask output directory if requested
    if instance_masks_dir is not None:
        instance_masks_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Instance masks will be saved to {instance_masks_dir}")

    # Run each prompt as a separate cycle (add_prompt resets state)
    # Try multiple seed frames to catch objects that appear later in the video.
    obj_id_to_label: Dict[int, str] = {}
    prompt_stats: Dict[str, int] = {}
    obj_frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    global_obj_counter = 0  # Assign unique IDs across prompts

    # Seed frames: 0%, 25%, 50%, 75% of the video
    seed_frames = sorted(set([
        0,
        n_frames // 4,
        n_frames // 2,
        3 * n_frames // 4,
    ]))
    _log(f"Seed frames for multi-frame detection: {seed_frames}")

    for prompt_idx, prompt in enumerate(prompts):
        _log(f"  [{prompt_idx+1}/{len(prompts)}] Prompt: '{prompt}'")

        found_on_seed = None
        result = None

        try:
            # Try each seed frame until the object is detected
            for seed_idx in seed_frames:
                result = predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=seed_idx,
                    text=prompt,
                )

                outputs = result.get("outputs", result)
                new_ids = outputs.get("out_obj_ids", np.array([]))
                if hasattr(new_ids, "tolist"):
                    new_ids = new_ids.tolist()

                if new_ids:
                    found_on_seed = seed_idx
                    break

            if not new_ids or found_on_seed is None:
                _log(f"    No instances detected on any seed frame {seed_frames}")
                prompt_stats[prompt] = 0
                continue

            _log(f"    {len(new_ids)} instance(s) detected on seed frame {found_on_seed}")

            # Map SAM3 obj_ids to our global IDs with label
            local_to_global: Dict[int, int] = {}
            for sam_oid in new_ids:
                sam_oid = int(sam_oid)
                global_id = global_obj_counter
                global_obj_counter += 1
                local_to_global[sam_oid] = global_id
                obj_id_to_label[global_id] = prompt

            prompt_stats[prompt] = len(new_ids)

            # Collect seed frame results from add_prompt itself
            out_probs = outputs.get("out_probs", np.array([]))
            out_boxes_xywh = outputs.get("out_boxes_xywh", np.array([]))
            out_masks = outputs.get("out_binary_masks", np.array([]))
            for idx_in_batch, sam_oid in enumerate(new_ids):
                sam_oid = int(sam_oid)
                global_id = local_to_global[sam_oid]
                prob = float(out_probs[idx_in_batch]) if idx_in_batch < len(out_probs) else 0.0

                if idx_in_batch < len(out_boxes_xywh):
                    bx, by, bw, bh = out_boxes_xywh[idx_in_batch]
                    box_px = [float(bx)*img_w, float(by)*img_h, float(bx+bw)*img_w, float(by+bh)*img_h]
                else:
                    box_px = [0, 0, img_w, img_h]

                mask_np = None
                if idx_in_batch < len(out_masks):
                    mask_np = out_masks[idx_in_batch]
                    if hasattr(mask_np, "cpu"):
                        mask_np = mask_np.cpu().numpy()
                    if mask_np.ndim == 3:
                        mask_np = mask_np.squeeze(0)
                    mask_np = mask_np.astype(bool)

                if mask_np is not None and mask_np.any():
                    ys, xs = np.where(mask_np)
                    cx, cy = float(xs.mean()), float(ys.mean())
                    mask_area = int(mask_np.sum())
                else:
                    cx = (box_px[0] + box_px[2]) / 2
                    cy = (box_px[1] + box_px[3]) / 2
                    mask_area = int((box_px[2] - box_px[0]) * (box_px[3] - box_px[1]))

                fd: Dict[str, Any] = {
                    "frame_idx": found_on_seed, "prob": prob, "box": box_px,
                    "centroid_px": [cx, cy], "mask_area_px": mask_area,
                    "image_size": [img_w, img_h],
                }
                if mask_np is not None:
                    fd["_mask_np"] = mask_np
                if found_on_seed < len(all_frame_files):
                    fd["frame_path"] = str(all_frame_files[found_on_seed])
                obj_frames[global_id].append(fd)

                # Save instance mask contribution for seed frame
                if instance_masks_dir is not None and mask_np is not None:
                    if found_on_seed < len(all_frame_files):
                        _accumulate_instance_mask(
                            mask_np, global_id + 1,  # 1-indexed for Inpaint360GS
                            all_frame_files[found_on_seed].stem,
                            instance_masks_dir,
                        )

            # Propagate from seed frame (forward + backward if seed > 0)
            _log(f"    Propagating from seed frame {found_on_seed}...")
            temp_label_map: Dict[int, str] = {
                sam_oid: prompt for sam_oid in [int(x) for x in new_ids]
            }
            temp_obj_frames: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            # Build SAM-oid → 1-indexed global ID map for instance mask writing
            propagation_global_map: Optional[Dict[int, int]] = None
            if instance_masks_dir is not None:
                propagation_global_map = {
                    sam_oid: local_to_global[sam_oid] + 1  # 1-indexed
                    for sam_oid in [int(x) for x in new_ids]
                    if sam_oid in local_to_global
                }
            _collect_propagation_results(
                predictor, session_id, prompt,
                temp_label_map, all_frame_files,
                img_w, img_h, n_frames, temp_obj_frames,
                seed_frame_idx=found_on_seed,
                instance_masks_dir=instance_masks_dir,
                global_obj_id_map=propagation_global_map,
            )

            # Remap SAM3 obj_ids to global IDs
            for sam_oid, frame_data_list in temp_obj_frames.items():
                global_id = local_to_global.get(int(sam_oid))
                if global_id is not None:
                    obj_frames[global_id].extend(frame_data_list)

            n_propagated = sum(len(v) for v in temp_obj_frames.values())
            _log(f"    Collected {n_propagated} frame-detections across {len(temp_obj_frames)} objects")

        except Exception as exc:
            _log(f"    Prompt '{prompt}' failed: {exc}")
            import traceback
            traceback.print_exc()
            prompt_stats[prompt] = 0

    _log(f"All prompts complete. {len(obj_id_to_label)} total tracked objects across {n_frames} frames")

    # Shutdown predictor and free VRAM before DA3
    predictor.close_session(session_id)
    predictor.shutdown()
    del predictor
    torch.cuda.empty_cache()
    _log(f"Video predictor freed. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Convert tracked objects into scene-level object entries
    objects: List[Dict[str, Any]] = []
    for oid, frames_data in obj_frames.items():
        label = obj_id_to_label.get(oid, "object")
        n_obj_frames = len(set(fd["frame_idx"] for fd in frames_data))

        # Skip objects with very few frame appearances
        if n_obj_frames < 2:
            continue

        # Compute aggregate stats
        probs = [fd["prob"] for fd in frames_data]
        boxes = [fd["box"] for fd in frames_data]
        centroids = [fd["centroid_px"] for fd in frames_data]

        mean_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        mean_box = [float(np.mean([b[i] for b in boxes])) for i in range(4)]
        mean_centroid = [float(np.mean([c[i] for c in centroids])) for i in range(2)]

        frame_indices = sorted(set(fd["frame_idx"] for fd in frames_data))
        frame_paths = []
        for fd in sorted(frames_data, key=lambda x: x["frame_idx"]):
            fp = fd.get("frame_path")
            if fp and fp not in frame_paths:
                frame_paths.append(fp)

        # Heuristic 3D from 2D (will be refined by DA3 depth post-processing)
        box_w = mean_box[2] - mean_box[0]
        box_h = mean_box[3] - mean_box[1]
        scene_depth_est = 3.0
        scale = scene_depth_est / max(img_w, img_h)
        width_m = box_w * scale
        height_m = box_h * scale
        depth_m = min(width_m, height_m) * 0.6
        cx_3d = (mean_centroid[0] / img_w - 0.5) * scene_depth_est
        cy_3d = (0.5 - mean_centroid[1] / img_h) * scene_depth_est
        cz_3d = scene_depth_est * 0.5

        # Save reference crop from best-probability frame
        best_crop = None
        all_crops: List[str] = []
        if save_crops and crops_dir is not None:
            ranked = sorted(frames_data, key=lambda fd: fd["prob"], reverse=True)
            for crop_idx, fd in enumerate(ranked[:_MAX_REFERENCE_CROPS]):
                mask = fd.get("_mask_np")
                fp = fd.get("frame_path")
                if mask is not None and fp:
                    cp = _save_object_crop(
                        Path(fp), mask, fd["box"], crops_dir,
                        label, fd["frame_idx"], crop_idx,
                    )
                    if cp is not None:
                        all_crops.append(cp.as_posix())
            if all_crops:
                best_crop = all_crops[0]

        # Determine instance index among same-label objects
        same_label_oids = sorted(
            [k for k, v in obj_id_to_label.items()
             if v == label and k in obj_frames and len(set(fd["frame_idx"] for fd in obj_frames[k])) >= 2]
        )
        instance_idx = same_label_oids.index(oid) if oid in same_label_oids else 0

        obj_entry: Dict[str, Any] = {
            "id": f"{label}_{instance_idx + 1}",
            "label": label,
            "confidence": round(max_prob, 3),
            "mean_confidence": round(mean_prob, 3),
            "n_frame_detections": n_obj_frames,
            "n_total_detections": len(frames_data),
            "frame_indices": frame_indices,
            "frame_paths": frame_paths,
            "boundingBox": {
                "center": [round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4)],
                "extents": [
                    round(max(0.02, width_m), 4),
                    round(max(0.02, height_m), 4),
                    round(max(0.02, depth_m), 4),
                ],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
            "mean_box_px": [round(v, 1) for v in mean_box],
            "mean_centroid_px": [round(v, 1) for v in mean_centroid],
            "image_size": [int(img_w), int(img_h)],
            "detection_source": "sam3",
            "refinement": "heuristic_2d",
            "video_predictor_obj_id": oid,
        }
        if best_crop is not None:
            obj_entry["reference_crop"] = best_crop
        if all_crops:
            obj_entry["all_crops"] = all_crops

        objects.append(obj_entry)

    # Sort by confidence descending
    objects.sort(key=lambda x: x["confidence"], reverse=True)

    # Clean up transient mask data
    for frames_data_list in obj_frames.values():
        for fd in frames_data_list:
            fd.pop("_mask_np", None)

    metadata = {
        "tracking_backend": "video_predictor",
        "n_objects_detected": len(objects),
        "n_frames": n_frames,
        "n_prompts": len(prompts),
        "prompt_stats": prompt_stats,
        "obj_id_to_label": {str(k): v for k, v in obj_id_to_label.items()},
    }

    _log(f"Video predictor produced {len(objects)} tracked objects")
    return objects, metadata


def _add_depth_to_tracked_objects(
    objects: List[Dict[str, Any]],
    frames_dir: Path,
    focal_px: float = 1000.0,
    max_depth_frames_per_object: int = 12,
) -> List[Dict[str, Any]]:
    """Add DA3 metric depth to video-predictor-tracked objects.

    Loads DA3 model (after video predictor is unloaded), samples frames
    per object, computes metric depth maps, and aggregates 3D bounding
    box estimates.

    Returns objects list with updated depth_3d and boundingBox fields.
    """
    if not objects:
        return objects

    # Load DA3
    try:
        da3_model = _load_da3()
    except Exception as exc:
        _log(f"DA3 not available ({exc}), skipping depth post-processing")
        return objects

    from PIL import Image as PILImage

    depth_cache: Dict[str, np.ndarray] = {}  # frame_path → depth_map
    n_refined = 0

    for obj in objects:
        frame_paths = obj.get("frame_paths", [])
        if not frame_paths:
            continue

        # Sample frames evenly (up to max_depth_frames_per_object)
        n_sample = min(max_depth_frames_per_object, len(frame_paths))
        if len(frame_paths) <= n_sample:
            sampled_paths = frame_paths
        else:
            indices = np.linspace(0, len(frame_paths) - 1, n_sample, dtype=int)
            sampled_paths = [frame_paths[i] for i in indices]

        centers_3d = []
        extents_3d = []

        for fp_str in sampled_paths:
            fp = Path(fp_str)
            if not fp.exists():
                continue

            # Get depth map (cached per frame)
            if fp_str not in depth_cache:
                try:
                    img = PILImage.open(fp).convert("RGB")
                    depth_cache[fp_str] = _get_metric_depth(da3_model, img, focal_px)
                except Exception:
                    continue

            depth_map = depth_cache[fp_str]

            # Find which frame data matches this path to get box
            box_px = obj.get("mean_box_px", [0, 0, 100, 100])
            img_w = obj.get("mean_centroid_px", [100, 100])
            image_size = [1920, 1080]  # default

            # Get image dimensions from depth map
            dh, dw = depth_map.shape[:2]

            # Create approximate mask from bounding box
            x1 = max(0, int(box_px[0]))
            y1 = max(0, int(box_px[1]))
            x2 = min(dw, int(box_px[2]))
            y2 = min(dh, int(box_px[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            # Scale box to depth map resolution if needed
            if dw != obj.get("mean_box_px", [0])[0]:
                # Compute based on image_size stored in object
                stored_size = None
                for fi in obj.get("frame_indices", []):
                    break  # We just need the image size

            object_depths = depth_map[y1:y2, x1:x2].flatten()
            object_depths = object_depths[object_depths > 0.05]  # Filter zero/near-zero
            if len(object_depths) < 10:
                continue

            median_depth = float(np.median(object_depths))
            depth_range = float(
                np.percentile(object_depths, 90) - np.percentile(object_depths, 10)
            )

            # Convert 2D extent to 3D
            box_w_px = box_px[2] - box_px[0]
            box_h_px = box_px[3] - box_px[1]
            width_m = box_w_px * median_depth / focal_px
            height_m = box_h_px * median_depth / focal_px
            depth_m = max(depth_range, min(width_m, height_m) * 0.3)

            # 3D center from 2D centroid + depth
            cx_2d = (box_px[0] + box_px[2]) / 2
            cy_2d = (box_px[1] + box_px[3]) / 2
            cx_3d = (cx_2d - dw / 2) * median_depth / focal_px
            cy_3d = (dh / 2 - cy_2d) * median_depth / focal_px
            cz_3d = median_depth

            centers_3d.append([cx_3d, cy_3d, cz_3d])
            extents_3d.append([
                max(0.02, width_m),
                max(0.02, height_m),
                max(0.02, depth_m),
            ])

        if centers_3d:
            centers_arr = np.array(centers_3d)
            extents_arr = np.array(extents_3d)
            center = np.median(centers_arr, axis=0)
            extents = np.median(extents_arr, axis=0)

            obj["boundingBox"]["center"] = [round(float(c), 4) for c in center]
            obj["boundingBox"]["extents"] = [round(float(e), 4) for e in extents]
            obj["refinement"] = "da3_metric_depth"
            n_refined += 1

    # Free DA3
    del da3_model
    del depth_cache
    torch.cuda.empty_cache()
    _log(f"DA3 depth post-processing: refined {n_refined}/{len(objects)} objects")

    return objects


def run_sam3_video_predictor(
    *,
    frames_dir: Path,
    output_path: Path,
    environment: str = "auto",
    detection_prompts_override: Optional[List[str]] = None,
    prompt_source_override: Optional[str] = None,
    environment_source: Optional[str] = None,
    environment_confidence: Optional[float] = None,
    colmap_sparse_dir: Optional[Path] = None,
    gaussian_ply_path: Optional[Path] = None,
    save_crops: bool = True,
    video_path: Optional[Path] = None,
    extraction_fps: int = 0,
    adaptive_fps_reasoning: str = "",
    dimension_completion_mode: Optional[str] = None,
    save_instance_masks: bool = False,
    instance_masks_dir: Optional[Path] = None,
    force_full_video_masks: bool = False,
) -> Dict[str, Any]:
    """Run SAM3 video predictor pipeline and write object index.

    This is the video-predictor-based alternative to run_sam3_detection().
    Instead of per-frame image detection + custom association, it uses
    SAM3's native persistent video tracking with a 7-frame sliding memory.
    """
    _log(f"[VIDEO PREDICTOR MODE]")
    _log(f"Environment: {environment}")
    _log(f"Frames dir: {frames_dir}")
    _log(f"Output: {output_path}")

    all_frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    override_prompts = _normalize_prompts(detection_prompts_override or [])
    if override_prompts:
        prompts = override_prompts
        prompt_source = (prompt_source_override or "scene_semantics_override").strip()
    else:
        prompts, prompt_source = _resolve_detection_prompts(
            environment=environment,
            frames_dir=frames_dir,
            all_frames=all_frames,
        )
    _log(f"Detection prompts ({len(prompts)}) [{prompt_source}]: {', '.join(prompts)}")

    # Set up crops directory
    crops_dir = None
    if save_crops:
        crops_dir = output_path.parent / "object_crops"
        _log(f"Object crops will be saved to: {crops_dir}")

    # Set up instance masks directory for Inpaint360GS scene cleaning
    inst_masks_dir: Optional[Path] = None
    if save_instance_masks:
        inst_masks_dir = instance_masks_dir or (output_path.parent / "instance_masks")
        _log(f"Instance segmentation masks will be saved to: {inst_masks_dir}")

    # Run video predictor
    objects, vp_metadata = _detect_with_video_predictor(
        frames_dir=frames_dir,
        prompts=prompts,
        save_crops=save_crops,
        crops_dir=crops_dir,
        instance_masks_dir=inst_masks_dir,
    )

    # DA3 depth post-processing (video predictor already freed)
    focal_px = 1000.0
    if colmap_sparse_dir and (colmap_sparse_dir / "cameras.bin").exists():
        try:
            cams = _read_colmap_cameras(colmap_sparse_dir / "cameras.bin")
            if cams:
                cam = list(cams.values())[0]
                focal_px = cam.get("fx", 1000.0)
                _log(f"COLMAP focal length: {focal_px:.1f}px")
        except Exception as e:
            _log(f"Could not read COLMAP cameras: {e}")

    objects = _add_depth_to_tracked_objects(objects, frames_dir, focal_px)

    # COLMAP + Gaussian back-projection refinement (for objects not refined by DA3)
    objects = _refine_with_colmap(objects, colmap_sparse_dir, gaussian_ply_path)
    objects, dimension_completion_report = _apply_occlusion_dimension_completion(
        objects=objects,
        output_path=output_path,
        environment=environment,
        mode_override=dimension_completion_mode,
    )
    if save_instance_masks:
        for obj in objects:
            try:
                # Keep mask IDs stable with the IDs written during propagation.
                obj["instance_mask_id"] = int(obj.get("video_predictor_obj_id", -1)) + 1
            except Exception:
                continue

    # Report
    n_with_crops = sum(1 for obj in objects if "reference_crop" in obj)
    _log(f"\nDetected objects ({n_with_crops}/{len(objects)} with reference crops):")
    for obj in objects:
        bb = obj["boundingBox"]
        crop_tag = " [crop]" if "reference_crop" in obj else ""
        _log(f"  {obj['id']:20s}  conf={obj['confidence']:.2f}  "
             f"frames={obj['n_frame_detections']}  "
             f"size={bb['extents'][0]:.2f}x{bb['extents'][1]:.2f}x{bb['extents'][2]:.2f}m"
             f"{crop_tag}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        normalized_env_conf = float(environment_confidence) if environment_confidence is not None else 1.0
    except (TypeError, ValueError):
        normalized_env_conf = 0.0
    normalized_env_conf = max(0.0, min(1.0, normalized_env_conf))

    index_payload = {
        "schema_version": "v1",
        "detection_source": "sam3",
        "tracking_backend": "video_predictor",
        "environment": environment,
        "environment_source": str(environment_source or "").strip() or "environment_input",
        "environment_confidence": round(normalized_env_conf, 4),
        "prompt_source": prompt_source,
        "n_frames_total": len(all_frames),
        "n_raw_detections": sum(obj.get("n_total_detections", 0) for obj in objects),
        "prompts_used": prompts,
        "dimension_completion": dimension_completion_report,
        "objects": objects,
    }
    if save_instance_masks and inst_masks_dir is not None:
        n_mask_frames = len(list(inst_masks_dir.glob("*.png"))) if inst_masks_dir.is_dir() else 0
        index_payload["instance_masks_dir"] = str(inst_masks_dir)
        index_payload["instance_mask_dtype"] = "uint16"
        index_payload["instance_masks_frame_count"] = int(n_mask_frames)
    if force_full_video_masks:
        index_payload["force_full_video_masks"] = True

    # Add video-specific metadata
    if video_path:
        index_payload["input_source"] = f"video:{video_path}"
    if extraction_fps > 0:
        index_payload["extraction_fps"] = extraction_fps
    if adaptive_fps_reasoning:
        index_payload["adaptive_fps_reasoning"] = adaptive_fps_reasoning
    index_payload.update({
        k: v for k, v in vp_metadata.items()
        if k not in index_payload
    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2)

    _log(f"\nWrote {len(objects)} objects to {output_path}")
    return index_payload


def run_sam3_detection(
    *,
    frames_dir: Path,
    output_path: Path,
    environment: str = "auto",
    detection_prompts_override: Optional[List[str]] = None,
    prompt_source_override: Optional[str] = None,
    environment_source: Optional[str] = None,
    environment_confidence: Optional[float] = None,
    colmap_sparse_dir: Optional[Path] = None,
    gaussian_ply_path: Optional[Path] = None,
    n_sample_frames: int = _DEFAULT_SAMPLE_FRAMES,
    min_frame_detections: int = 2,
    save_crops: bool = True,
    dimension_completion_mode: Optional[str] = None,
    save_instance_masks: bool = False,
    instance_masks_dir: Optional[Path] = None,
    force_full_video_masks: bool = False,
) -> Dict[str, Any]:
    """Run full SAM3 detection pipeline and write object index.

    When ``gaussian_ply_path`` is provided (the 3DGRUT Gaussian splat PLY),
    SAM3 2D masks are back-projected through COLMAP cameras into the 3D
    point cloud to produce accurate real-world bounding boxes (position +
    width/height/depth in meters).
    """

    _log(f"Environment: {environment}")
    _log(f"Frames dir: {frames_dir}")
    _log(f"Output: {output_path}")
    if gaussian_ply_path:
        _log(f"Gaussian PLY: {gaussian_ply_path}")

    all_frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    override_prompts = _normalize_prompts(detection_prompts_override or [])
    if override_prompts:
        prompts = override_prompts
        prompt_source = (prompt_source_override or "scene_semantics_override").strip()
    else:
        prompts, prompt_source = _resolve_detection_prompts(
            environment=environment,
            frames_dir=frames_dir,
            all_frames=all_frames,
        )
    _log(f"Detection prompts ({len(prompts)}) [{prompt_source}]: {', '.join(prompts)}")
    n_sample_frames, min_frame_detections = _resolve_sampling_settings(
        environment=environment,
        total_frames=len(all_frames),
        requested_n_frames=n_sample_frames,
        requested_min_frame_detections=min_frame_detections,
    )
    _log(
        f"Sampling settings: total_frames={len(all_frames)} "
        f"n_frames={n_sample_frames} min_frame_detections={min_frame_detections}"
    )

    tracking_mode_raw = (
        os.getenv("SAM3_TRACKING_MODE", _TRACKING_MODE_DEFAULT) or _TRACKING_MODE_DEFAULT
    ).strip().lower()
    if force_full_video_masks:
        tracking_mode = "full_video"
        tracking_mode_reason = "forced_by_force_full_video_masks"
    else:
        tracking_mode, tracking_mode_reason = _resolve_tracking_mode(
            tracking_mode_raw,
            len(all_frames),
        )
    _log(f"Tracking mode resolved: {tracking_mode} ({tracking_mode_reason})")

    if tracking_mode == "full_video":
        frame_paths = all_frames
        _log(f"Tracking mode: full_video (using all {len(frame_paths)} frames)")
    else:
        frame_paths = _sample_frame_paths(frames_dir, n_sample_frames)
        _log(f"Tracking mode: sampled (using {len(frame_paths)} frames)")

    # Load SAM3
    processor = _load_sam3()

    # Load DA3 for metric depth (optional but recommended)
    da3_model = None
    focal_px = 1000.0  # default, overridden by COLMAP if available
    if colmap_sparse_dir and (colmap_sparse_dir / "cameras.bin").exists():
        try:
            cams = _read_colmap_cameras(colmap_sparse_dir / "cameras.bin")
            if cams:
                cam = list(cams.values())[0]
                focal_px = cam.get("fx", 1000.0)
                _log(f"COLMAP focal length: {focal_px:.1f}px")
        except Exception as e:
            _log(f"Could not read COLMAP cameras: {e}")

    try:
        da3_model = _load_da3()
        _log("DA3 metric depth enabled - will compute accurate 3D bounding boxes")
    except Exception as e:
        _log(f"DA3 not available ({e}), using heuristic 3D estimates")

    # Set up crops directory for reference image extraction
    crops_dir = None
    if save_crops:
        crops_dir = output_path.parent / "object_crops"
        _log(f"Object crops will be saved to: {crops_dir}")

    inst_masks_dir: Optional[Path] = None
    if save_instance_masks:
        inst_masks_dir = instance_masks_dir or (output_path.parent / "instance_masks")
        inst_masks_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Instance masks will be saved to: {inst_masks_dir}")

    # Run detection on each frame
    all_detections: List[Dict[str, Any]] = []
    for i, frame_path in enumerate(frame_paths):
        _log(f"  Frame {i+1}/{len(frame_paths)}: {frame_path.name}")

        # Get depth map for this frame
        depth_map = None
        if da3_model is not None:
            try:
                from PIL import Image
                img_for_depth = Image.open(frame_path).convert("RGB")
                depth_map = _get_metric_depth(da3_model, img_for_depth, focal_px)
            except Exception as e:
                _log(f"    DA3 depth failed: {e}")

        dets = _detect_objects_in_frame(
            processor, frame_path, prompts,
            depth_map=depth_map, focal_px=focal_px,
            crops_dir=crops_dir, frame_idx=i,
            include_mask=save_instance_masks,
        )
        n_with_depth = sum(1 for d in dets if "depth_3d" in d)
        _log(f"    {len(dets)} detections ({n_with_depth} with metric depth)")
        all_detections.extend(dets)

    _log(f"Total raw detections: {len(all_detections)}")

    # Free DA3 memory before merge step
    if da3_model is not None:
        del da3_model
        torch.cuda.empty_cache()

    # Merge across frames
    objects = _merge_detections(
        all_detections,
        preserve_detections=save_instance_masks,
    )
    _log(f"Merged into {len(objects)} unique objects")

    # Filter: require detection in multiple frames for robustness
    if min_frame_detections > 1:
        before = len(objects)
        objects = [
            obj for obj in objects
            if obj["n_frame_detections"] >= min_frame_detections
        ]
        _log(f"After multi-frame filter (>={min_frame_detections}): {len(objects)} objects (removed {before - len(objects)})")

    # Refine with COLMAP + Gaussian PLY for accurate 3D bounding boxes
    objects = _refine_with_colmap(objects, colmap_sparse_dir, gaussian_ply_path, all_detections)

    # Persistence filter: drop transient tracks before emitting final objects.
    persistence_min_frames, persistence_min_total = _resolve_persistence_thresholds(
        environment,
        min_frame_detections,
    )
    before_persistence = len(objects)
    objects = [
        obj
        for obj in objects
        if int(obj.get("n_frame_detections", 0)) >= persistence_min_frames
        and int(obj.get("n_total_detections", 0)) >= persistence_min_total
    ]
    _log(
        f"After persistence filter (frames>={persistence_min_frames}, "
        f"total>={persistence_min_total}): {len(objects)} objects "
        f"(removed {before_persistence - len(objects)})"
    )
    objects, dimension_completion_report = _apply_occlusion_dimension_completion(
        objects=objects,
        output_path=output_path,
        environment=environment,
        mode_override=dimension_completion_mode,
    )

    mask_metadata: Dict[str, Any] = {}
    if save_instance_masks and inst_masks_dir is not None:
        for idx, obj in enumerate(objects, start=1):
            obj["instance_mask_id"] = idx
        mask_metadata = _write_instance_masks_from_objects(
            objects=objects,
            instance_masks_dir=inst_masks_dir,
        )

    # Report
    n_with_crops = sum(1 for obj in objects if "reference_crop" in obj)
    _log(f"\nDetected objects ({n_with_crops}/{len(objects)} with reference crops):")
    for obj in objects:
        bb = obj["boundingBox"]
        crop_tag = " [crop]" if "reference_crop" in obj else ""
        _log(f"  {obj['id']:20s}  conf={obj['confidence']:.2f}  "
             f"frames={obj['n_frame_detections']}  "
             f"size={bb['extents'][0]:.2f}x{bb['extents'][1]:.2f}x{bb['extents'][2]:.2f}m"
             f"{crop_tag}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        normalized_env_conf = float(environment_confidence) if environment_confidence is not None else 1.0
    except (TypeError, ValueError):
        normalized_env_conf = 0.0
    normalized_env_conf = max(0.0, min(1.0, normalized_env_conf))

    index_payload = {
        "schema_version": "v1",
        "detection_source": "sam3",
        "environment": environment,
        "environment_source": str(environment_source or "").strip() or "environment_input",
        "environment_confidence": round(normalized_env_conf, 4),
        "prompt_source": prompt_source,
        "tracking_mode": tracking_mode,
        "track_max_frame_gap": _TRACK_MAX_FRAME_GAP,
        "track_min_assoc_score": _TRACK_MIN_ASSOC_SCORE,
        "persistence_min_track_frames": persistence_min_frames,
        "persistence_min_total_detections": persistence_min_total,
        "n_frames_sampled": len(frame_paths),
        "n_raw_detections": len(all_detections),
        "prompts_used": prompts,
        "dimension_completion": dimension_completion_report,
        "objects": objects,
    }
    index_payload.update(mask_metadata)

    for obj in objects:
        obj.pop("_cluster_detections", None)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2)

    _log(f"\nWrote {len(objects)} objects to {output_path}")

    # Free GPU memory
    del processor
    torch.cuda.empty_cache()

    return index_payload


def _sync_descriptor_environment(
    output_path: Path,
    resolved_environment: str,
) -> None:
    """Update capture_descriptor.json with Gemini-resolved environment.

    Searches for capture_descriptor.json near the output path and updates
    the environment_type_hint field so the orchestrator applies correct
    class caps (e.g., residential caps for bedroom instead of warehouse).
    """
    # Search for descriptor in likely locations
    candidates = [
        output_path.parent.parent / "capture_descriptor.json",
        output_path.parent / "capture_descriptor.json",
        output_path.parent.parent.parent / "capture_descriptor.json",
    ]
    descriptor_path = None
    for cp in candidates:
        if cp.exists():
            descriptor_path = cp
            break

    if descriptor_path is None:
        _log("No capture_descriptor.json found for environment sync")
        return

    try:
        with open(descriptor_path, "r", encoding="utf-8") as f:
            descriptor = json.load(f)

        old_env = descriptor.get("environment_type_hint", "")
        if old_env == resolved_environment:
            _log(f"Descriptor environment_type_hint already '{resolved_environment}', no update needed")
            return

        descriptor["environment_type_hint"] = resolved_environment
        with open(descriptor_path, "w", encoding="utf-8") as f:
            json.dump(descriptor, f, indent=2)

        _log(f"Updated descriptor environment_type_hint from '{old_env}' to "
             f"'{resolved_environment}' (Gemini inference) at {descriptor_path}")
    except Exception as exc:
        _log(f"Failed to sync descriptor environment: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 object detection for swap pipeline"
    )
    parser.add_argument("--frames-dir", default=None,
                        help="Directory with extracted video frames")
    parser.add_argument("--video", default=None,
                        help="Path to raw video file (.mp4/.mov). Alternative to --frames-dir")
    parser.add_argument("--fps", type=int, default=0,
                        help="Frame rate for extraction from --video (0=adaptive based on VRAM)")
    parser.add_argument("--tracking-backend", default="video_predictor",
                        choices=["video_predictor", "image_model"],
                        help="Detection backend: video_predictor (persistent tracking) or image_model (legacy per-frame)")
    parser.add_argument("--output", required=True,
                        help="Output path for object_point_cloud_index.json")
    parser.add_argument("--environment", default="auto",
                        choices=["auto", *list(_DETECTION_PROMPTS.keys())],
                        help="Environment type for prompt selection")
    parser.add_argument("--colmap-sparse", default=None,
                        help="Path to COLMAP sparse/0/ for 3D refinement")
    parser.add_argument("--gaussian-ply", default=None,
                        help="Path to 3DGRUT export_last.ply for accurate 3D back-projection")
    parser.add_argument("--n-frames", type=int, default=0,
                        help="Number of frames to sample (0=auto, image_model only)")
    parser.add_argument("--min-frame-detections", type=int, default=0,
                        help="Minimum frames an object must appear in (0=auto, image_model only)")
    parser.add_argument("--no-crops", action="store_true",
                        help="Disable saving per-object reference crops")
    parser.add_argument("--save-instance-masks", action="store_true",
                        help="Save per-view instance segmentation masks for Inpaint360GS scene cleaning")
    parser.add_argument(
        "--instance-masks-dir",
        default=None,
        help="Optional output directory for instance masks (default: <output_dir>/instance_masks)",
    )
    parser.add_argument(
        "--force-full-video-masks",
        action="store_true",
        help="Force all frames to be processed (ignores sampled mode) when exporting instance masks",
    )
    parser.add_argument("--scene-semantics", action="store_true",
                        help="Run Gemini scene semantics before detection to infer environment and prompts")
    parser.add_argument(
        "--dimension-completion-mode",
        default=None,
        choices=["off", "auto", "always"],
        help=(
            "Occlusion-aware dimension completion mode "
            "(default comes from SAM3_DIMENSION_COMPLETION_MODE)."
        ),
    )
    args = parser.parse_args()

    # Validate input: need either --frames-dir or --video
    if not args.frames_dir and not args.video:
        parser.error("Either --frames-dir or --video is required")

    video_path = None
    if args.video:
        try:
            video_path = _validate_local_video_path(args.video)
        except ValueError as exc:
            parser.error(str(exc))
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    # Scene semantics: Gemini-first environment inference
    detection_prompts_override = None
    prompt_source_override = None
    environment_source = None
    environment_confidence = None
    resolved_environment = args.environment

    # For scene semantics with --video, we need frames first.
    # Extract a small set for Gemini analysis, then the full set for detection.
    adaptive_fps_reasoning = ""
    extraction_fps = args.fps

    if args.scene_semantics:
        # If we have a video but no frames yet, extract a small set for Gemini
        sem_frames_dir = frames_dir
        if sem_frames_dir is None and video_path is not None:
            # Extract at low FPS just for scene semantics (fast)
            sem_output = Path(args.output).parent / "scene_semantics_frames"
            try:
                sem_frames_dir, _ = _extract_frames_from_video(video_path, sem_output, fps=3)
            except Exception as exc:
                _log(f"Frame extraction for scene semantics failed: {exc}")
                sem_frames_dir = None

        if sem_frames_dir is not None:
            try:
                import sys as _sys
                _repo_root = Path(__file__).resolve().parents[1]
                _src_root = _repo_root / "src"
                if str(_src_root) not in _sys.path:
                    _sys.path.insert(0, str(_src_root))
                from blueprint_pipeline.scene_semantics import (
                    infer_scene_semantics,
                    write_scene_semantics_report,
                )
                _log("Running Gemini scene semantics...")
                sem_report = infer_scene_semantics(
                    frames_dir=sem_frames_dir,
                    requested_environment=args.environment,
                )
                # Write report next to output
                sem_path = Path(args.output).parent.parent / "pipeline" / "nurec" / "scene_semantics_report.json"
                if not sem_path.parent.exists():
                    sem_path = Path(args.output).parent / "scene_semantics_report.json"
                write_scene_semantics_report(sem_path, sem_report)
                _log(f"Scene semantics report written to {sem_path}")
                _log(f"  environment_source: {sem_report.get('environment_source')}")
                _log(f"  resolved_environment: {sem_report.get('resolved_environment')}")
                _log(f"  environment_confidence: {sem_report.get('environment_confidence')}")
                _log(f"  prompt_source: {sem_report.get('prompt_source')}")

                resolved_environment = str(sem_report.get("resolved_environment") or args.environment)
                prompts = sem_report.get("detection_prompts")
                if isinstance(prompts, list) and prompts:
                    detection_prompts_override = prompts
                prompt_source_override = str(sem_report.get("prompt_source") or "").strip() or None
                environment_source = str(sem_report.get("environment_source") or "").strip() or None
                environment_confidence = sem_report.get("environment_confidence")

                # Sync descriptor environment if Gemini resolved a specific one
                if resolved_environment and resolved_environment != "auto":
                    _sync_descriptor_environment(Path(args.output), resolved_environment)

            except Exception as exc:
                _log(f"Scene semantics failed ({exc}), proceeding with --environment={args.environment}")

    # Handle video input: extract frames at target FPS
    if video_path is not None and frames_dir is None:
        if extraction_fps <= 0:
            extraction_fps, adaptive_fps_reasoning = _compute_safe_fps(video_path)
        else:
            adaptive_fps_reasoning = f"User-specified {extraction_fps}fps"

        # Store frames persistently near the output
        frames_output = Path(args.output).parent.parent / "pipeline" / "nurec" / f"video_frames_{extraction_fps}fps"
        if not frames_output.parent.exists():
            frames_output = Path(args.output).parent / f"video_frames_{extraction_fps}fps"

        frames_dir, n_extracted = _extract_frames_from_video(
            video_path, frames_output, extraction_fps,
        )

    if frames_dir is None:
        _log("ERROR: No frames directory available")
        return 1

    # Choose backend
    use_video_predictor = args.tracking_backend == "video_predictor"

    if use_video_predictor:
        try:
            result = run_sam3_video_predictor(
                frames_dir=frames_dir,
                output_path=Path(args.output),
                environment=resolved_environment,
                detection_prompts_override=detection_prompts_override,
                prompt_source_override=prompt_source_override,
                environment_source=environment_source,
                environment_confidence=environment_confidence,
                colmap_sparse_dir=Path(args.colmap_sparse) if args.colmap_sparse else None,
                gaussian_ply_path=Path(args.gaussian_ply) if args.gaussian_ply else None,
                save_crops=not args.no_crops,
                video_path=video_path,
                extraction_fps=extraction_fps,
                adaptive_fps_reasoning=adaptive_fps_reasoning,
                dimension_completion_mode=args.dimension_completion_mode,
                save_instance_masks=args.save_instance_masks,
                instance_masks_dir=Path(args.instance_masks_dir) if args.instance_masks_dir else None,
                force_full_video_masks=args.force_full_video_masks,
            )
        except Exception as exc:
            _log(f"Video predictor failed: {exc}")
            _log("Falling back to image_model backend...")
            import traceback
            traceback.print_exc()
            use_video_predictor = False

    if not use_video_predictor:
        # Legacy image model path
        result = run_sam3_detection(
            frames_dir=frames_dir,
            output_path=Path(args.output),
            environment=resolved_environment,
            detection_prompts_override=detection_prompts_override,
            prompt_source_override=prompt_source_override,
            environment_source=environment_source,
            environment_confidence=environment_confidence,
            colmap_sparse_dir=Path(args.colmap_sparse) if args.colmap_sparse else None,
            gaussian_ply_path=Path(args.gaussian_ply) if args.gaussian_ply else None,
            n_sample_frames=args.n_frames,
            min_frame_detections=args.min_frame_detections,
            save_crops=not args.no_crops,
            dimension_completion_mode=args.dimension_completion_mode,
            save_instance_masks=args.save_instance_masks,
            instance_masks_dir=Path(args.instance_masks_dir) if args.instance_masks_dir else None,
            force_full_video_masks=args.force_full_video_masks,
        )

    n_objects = len(result.get("objects", []))
    return 0 if n_objects > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
