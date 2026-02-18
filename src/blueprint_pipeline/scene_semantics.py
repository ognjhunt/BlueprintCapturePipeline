"""Scene semantics inference with Gemini-first and local fallback behavior."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_SUPPORTED_ENVIRONMENTS = {"default", "warehouse", "kitchen", "bedroom"}

_PROMPTS_BY_ENV: Dict[str, List[str]] = {
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
        "shelf",
        "box",
        "tote",
        "bin",
        "crate",
        "container",
        "pallet",
        "carton",
        "package",
        "door",
        "cart",
        "forklift",
        "rack",
        "barrel",
        "drum",
    ],
    "kitchen": [
        "cabinet",
        "drawer",
        "refrigerator",
        "fridge",
        "microwave",
        "oven",
        "dishwasher",
        "door",
        "mug",
        "cup",
        "bowl",
        "plate",
        "pot",
        "pan",
        "bottle",
    ],
    "bedroom": [
        "bed",
        "pillow",
        "blanket",
        "nightstand",
        "dresser",
        "wardrobe",
        "closet_door",
        "lamp",
        "desk",
        "chair",
        "mirror",
        "shelf",
        "box",
        "basket",
        "hamper",
        "door",
    ],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_environment(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "default"

    aliases = {
        "auto": "default",
        "livingroom": "default",
        "living_room": "default",
        "residential": "default",
        "home": "default",
        "apt": "default",
        "apartment": "default",
        "bed room": "bedroom",
        "bed-room": "bedroom",
        "bedchamber": "bedroom",
    }
    text = aliases.get(text, text)
    return text if text in _SUPPORTED_ENVIRONMENTS else "default"


def _sample_frame_paths(frames_dir: Path, n_samples: int) -> List[Path]:
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        return []
    if len(frames) <= n_samples:
        return frames
    indices = [
        int(round(i * (len(frames) - 1) / float(max(1, n_samples - 1))))
        for i in range(n_samples)
    ]
    return [frames[idx] for idx in indices]


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
            # Skip thinking parts — they contain reasoning, not the final answer
            if getattr(part, "thought", False):
                continue
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                return part_text
    return ""


@dataclass(frozen=True)
class _GeminiResult:
    environment: str
    confidence: float
    model: str
    raw_text: str
    detected_objects: List[Dict[str, Any]]


_DEFAULT_MODEL_CASCADE = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def _build_image_parts(frames: List[Path], max_frames: int = 8) -> List[Dict[str, Any]]:
    """Encode frame images as inline_data parts for Gemini."""
    parts: List[Dict[str, Any]] = []
    for frame in frames[:max_frames]:
        suffix = frame.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(frame.read_bytes()).decode("ascii"),
                }
            }
        )
    return parts


def _extract_json_array(text: str) -> List[Any]:
    """Extract a JSON array from text, handling markdown fences and nested objects."""
    # Strip markdown code fences (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("objects", "detected_objects", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
    except Exception:
        pass

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, list):
                return payload
        except Exception:
            pass
    return []


def _infer_with_gemini(*, frames: List[Path], timeout_sec: int) -> Optional[_GeminiResult]:
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google import genai  # type: ignore
    except Exception:
        return None

    override = (os.getenv("SCENE_SEMANTICS_GEMINI_MODEL") or "").strip()
    models_to_try = [override] if override else list(_DEFAULT_MODEL_CASCADE)

    image_parts = _build_image_parts(frames, max_frames=8)

    client = genai.Client(api_key=api_key)

    # Single combined call: room classification + object enumeration
    combined_prompt = (
        "Analyze these video frames of an indoor scene. Do TWO things:\n\n"
        "1. CLASSIFY the room type as one of: bedroom, kitchen, warehouse, default.\n"
        "2. LIST ALL distinct physical objects visible that could be manipulated in a "
        "robotics simulator (pick up, open, move, interact with). Include furniture with "
        "movable parts (drawers, doors), containers, items on surfaces, things on the floor, "
        "hanging items, etc.\n\n"
        "Return a single JSON object with these keys:\n"
        "- room_type: string (one of: bedroom, kitchen, warehouse, default)\n"
        "- confidence: number 0-1\n"
        "- rationale: short string explaining classification\n"
        "- objects: array of objects, each with:\n"
        "  - object_id: short snake_case identifier (e.g. 'blue_suitcase')\n"
        "  - category: object category (e.g. 'Furniture', 'Container', 'Clothing')\n"
        "  - sam_prompt: a 1-4 word phrase a segmentation model can use to find this object "
        "(e.g. 'blue suitcase', 'wooden dresser')\n\n"
        "Be thorough — list every distinct manipulatable object you see."
    )
    combined_parts: List[Dict[str, Any]] = [{"text": combined_prompt}] + image_parts

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[{"parts": combined_parts}],
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                },
            )
        except Exception:
            continue

        raw_text = _extract_response_text(response)
        if not raw_text:
            continue

        payload = _extract_json_object(raw_text)
        room_type = _normalize_environment(str(payload.get("room_type") or payload.get("environment") or "default"))
        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        detected_objects: List[Dict[str, Any]] = []
        raw_objects = payload.get("objects", [])
        if isinstance(raw_objects, list):
            detected_objects = raw_objects

        return _GeminiResult(
            environment=room_type, confidence=confidence, model=model,
            raw_text=raw_text, detected_objects=detected_objects,
        )

    return None


def infer_scene_semantics(
    *,
    frames_dir: Path,
    requested_environment: str,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    """Infer room type for detection prompts with Gemini-first behavior.

    Returns a report with normalized environment, provenance, confidence, and
    resolved prompts. If Gemini inference is unavailable, this falls back to a
    local auto prompt set.
    """

    requested = str(requested_environment or "").strip().lower()
    normalized_requested = _normalize_environment(requested)
    has_explicit_hint = requested not in {"", "auto", "default"} and normalized_requested in _SUPPORTED_ENVIRONMENTS
    keyframes = _sample_frame_paths(frames_dir, 8)

    # Always attempt Gemini inference first, regardless of explicit hint.
    gemini_result = _infer_with_gemini(frames=keyframes, timeout_sec=max(5, int(timeout_sec)))
    if gemini_result is not None:
        resolved = _normalize_environment(gemini_result.environment)

        # Use Gemini-enumerated objects as SAM prompts if available,
        # otherwise fall back to hardcoded environment prompts.
        if gemini_result.detected_objects:
            sam_prompts = []
            for obj in gemini_result.detected_objects:
                prompt = (obj.get("sam_prompt") or obj.get("object_id") or "").strip()
                if prompt:
                    # Normalize: replace underscores with spaces for SAM text prompts
                    sam_prompts.append(prompt.replace("_", " "))
            if not sam_prompts:
                sam_prompts = list(_PROMPTS_BY_ENV[resolved])
            prompt_source = "gemini_object_enumeration"
        else:
            sam_prompts = list(_PROMPTS_BY_ENV[resolved])
            prompt_source = "gemini_video_inference"

        return {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "requested_environment": requested or "auto",
            "resolved_environment": resolved,
            "environment_source": "gemini_video_inference",
            "environment_confidence": gemini_result.confidence,
            "prompt_source": prompt_source,
            "detection_prompts": sam_prompts,
            "gemini_model": gemini_result.model,
            "gemini_raw_response": gemini_result.raw_text,
            "gemini_detected_objects": gemini_result.detected_objects,
            "keyframes_used": [str(path) for path in keyframes],
            "fallback_reason": "",
            "explicit_hint": normalized_requested if has_explicit_hint else None,
        }

    # Gemini unavailable — fall back to explicit hint if provided.
    if has_explicit_hint:
        return {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "requested_environment": requested,
            "resolved_environment": normalized_requested,
            "environment_source": "explicit_hint_fallback",
            "environment_confidence": 0.7,
            "prompt_source": "explicit_hint_fallback",
            "detection_prompts": list(_PROMPTS_BY_ENV[normalized_requested]),
            "keyframes_used": [str(path) for path in keyframes],
            "fallback_reason": "gemini_unavailable_used_explicit_hint",
        }

    return {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "requested_environment": requested or "auto",
        "resolved_environment": "default",
        "environment_source": "local_auto_fallback",
        "environment_confidence": 0.35,
        "prompt_source": "auto_fallback",
        "detection_prompts": list(_PROMPTS_BY_ENV["default"]),
        "keyframes_used": [str(path) for path in keyframes],
        "fallback_reason": "gemini_unavailable_or_failed",
    }


def write_scene_semantics_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
