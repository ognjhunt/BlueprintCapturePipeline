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


def _infer_with_gemini(*, frames: List[Path], timeout_sec: int) -> Optional[_GeminiResult]:
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google import genai  # type: ignore
    except Exception:
        return None

    model = (os.getenv("SCENE_SEMANTICS_GEMINI_MODEL") or "gemini-3.0-pro").strip()
    prompt = (
        "Classify this video scene into one of: bedroom, kitchen, warehouse, default. "
        "Return JSON only with keys: room_type (string), confidence (0-1 number), rationale (short string)."
    )
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for frame in frames[:8]:
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

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[{"parts": parts}],
            config={"temperature": 0.1, "max_output_tokens": 300, "timeout": timeout_sec},
        )
    except Exception:
        return None

    raw_text = _extract_response_text(response)
    if not raw_text:
        return None

    payload = _extract_json_object(raw_text)
    room_type = _normalize_environment(str(payload.get("room_type") or payload.get("environment") or "default"))
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return _GeminiResult(environment=room_type, confidence=confidence, model=model, raw_text=raw_text)


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
    keyframes = _sample_frame_paths(frames_dir, 8)

    # Explicit env selections remain authoritative.
    if requested not in {"", "auto", "default"} and normalized_requested in _SUPPORTED_ENVIRONMENTS:
        return {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "requested_environment": requested,
            "resolved_environment": normalized_requested,
            "environment_source": "manual_override",
            "environment_confidence": 1.0,
            "prompt_source": "environment_override",
            "detection_prompts": list(_PROMPTS_BY_ENV[normalized_requested]),
            "keyframes_used": [str(path) for path in keyframes],
            "fallback_reason": "",
        }

    gemini_result = _infer_with_gemini(frames=keyframes, timeout_sec=max(5, int(timeout_sec)))
    if gemini_result is not None:
        resolved = _normalize_environment(gemini_result.environment)
        return {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "requested_environment": requested or "auto",
            "resolved_environment": resolved,
            "environment_source": "gemini_video_inference",
            "environment_confidence": gemini_result.confidence,
            "prompt_source": "gemini_video_inference",
            "detection_prompts": list(_PROMPTS_BY_ENV[resolved]),
            "gemini_model": gemini_result.model,
            "gemini_raw_response": gemini_result.raw_text,
            "keyframes_used": [str(path) for path in keyframes],
            "fallback_reason": "",
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
