"""Scene semantics inference with Gemini-first and local fallback behavior."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .optional_dependencies import log_missing_optional_dependency

logger = logging.getLogger(__name__)


_SUPPORTED_ENVIRONMENTS = {
    "default",
    "warehouse",
    "kitchen",
    "bedroom",
    "industrial_unknown",
    "manufacturing",
    "fulfillment",
    "brownfield_site",
}

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
    "industrial_unknown": [
        "aisle",
        "rack",
        "tote",
        "bin",
        "pallet",
        "threshold",
        "forklift",
        "charger",
        "barrier",
        "workcell",
        "handoff point",
        "traffic zone",
        "door",
        "floor hazard",
    ],
    "manufacturing": [
        "workcell",
        "rack",
        "bin",
        "tote",
        "threshold",
        "forklift",
        "cart",
        "charger",
        "barrier",
        "door",
    ],
    "fulfillment": [
        "aisle",
        "rack",
        "tote",
        "pallet",
        "forklift",
        "traffic zone",
        "handoff point",
        "door",
    ],
    "brownfield_site": [
        "aisle",
        "threshold",
        "door",
        "traffic zone",
        "workcell",
        "floor hazard",
        "rack",
        "tote",
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
        "industrial": "industrial_unknown",
        "factory": "manufacturing",
        "plant": "manufacturing",
    }
    text = aliases.get(text, text)
    return text if text in _SUPPORTED_ENVIRONMENTS else "default"


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
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
_DEFAULT_GEMINI_VIDEO_ANALYSIS_FPS = 5.0
_MAX_GEMINI_VIDEO_ANALYSIS_FPS = 24.0


def _gemini_video_analysis_fps() -> float:
    raw_value = (
        os.getenv("GEMINI_VIDEO_ANALYSIS_FPS")
        or os.getenv("CAPTURE_FIDELITY_GEMINI_VIDEO_FPS")
        or os.getenv("SCENE_SEMANTICS_GEMINI_VIDEO_FPS")
        or ""
    ).strip()
    try:
        fps = float(raw_value) if raw_value else _DEFAULT_GEMINI_VIDEO_ANALYSIS_FPS
    except (TypeError, ValueError):
        fps = _DEFAULT_GEMINI_VIDEO_ANALYSIS_FPS
    return max(0.1, min(_MAX_GEMINI_VIDEO_ANALYSIS_FPS, fps))


def _upload_gemini_video_file(client: Any, raw_video_path: Path, timeout_sec: int) -> Optional[Any]:
    try:
        uploaded = client.files.upload(file=str(raw_video_path))
    except Exception:
        return None

    started = time.time()
    current = uploaded
    while True:
        state = getattr(current, "state", None)
        state_name = str(getattr(state, "name", "") or "").strip().upper()
        if state_name == "ACTIVE":
            return current
        if state_name in {"FAILED", "ERROR"}:
            return None
        if time.time() - started >= max(5, int(timeout_sec)):
            return None
        time.sleep(2)
        try:
            current = client.files.get(name=current.name)
        except Exception:
            return None


def _build_gemini_video_part(types_module: Any, uploaded_file: Any, fps: float) -> Any:
    mime_type = str(
        getattr(uploaded_file, "mime_type", "")
        or getattr(uploaded_file, "mimeType", "")
        or "video/quicktime"
    ).strip() or "video/quicktime"
    file_uri = str(
        getattr(uploaded_file, "uri", "")
        or getattr(uploaded_file, "file_uri", "")
        or getattr(uploaded_file, "fileUri", "")
        or ""
    ).strip()
    return types_module.Part(
        fileData=types_module.FileData(fileUri=file_uri, mimeType=mime_type),
        videoMetadata=types_module.VideoMetadata(fps=fps),
    )


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


def _infer_with_gemini_video(*, raw_video_path: Path, timeout_sec: int) -> Optional[_GeminiResult]:
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or not raw_video_path.is_file():
        return None

    try:
        from google import genai  # type: ignore
    except ImportError:
        log_missing_optional_dependency(
            logger,
            feature="Gemini scene semantics inference",
            package="google-genai",
            extra="llm",
        )
        return None
    except Exception:
        return None

    override = (os.getenv("SCENE_SEMANTICS_GEMINI_MODEL") or "").strip()
    models_to_try = [override] if override else list(_DEFAULT_MODEL_CASCADE)
    client = genai.Client(api_key=api_key)
    uploaded = _upload_gemini_video_file(client, raw_video_path, timeout_sec)
    if uploaded is None:
        return None

    fps = _gemini_video_analysis_fps()
    combined_prompt = (
        "Analyze this walkthrough video of an indoor scene. Do TWO things:\n\n"
        "1. CLASSIFY the environment type as one of: industrial_unknown, manufacturing, fulfillment, warehouse, bedroom, kitchen, default.\n"
        "2. LIST ALL distinct physical objects visible that could be manipulated in a "
        "robotics simulator (pick up, open, move, interact with). Include furniture with "
        "movable parts (drawers, doors), containers, items on surfaces, things on the floor, "
        "hanging items, etc.\n\n"
        "Use the full video context, not just a single moment.\n\n"
        "Return a single JSON object with these keys:\n"
        "- room_type: string (one of: industrial_unknown, manufacturing, fulfillment, warehouse, bedroom, kitchen, default)\n"
        "- confidence: number 0-1\n"
        "- rationale: short string explaining classification\n"
        "- objects: array of objects, each with:\n"
        "  - object_id: short snake_case identifier (e.g. 'blue_suitcase')\n"
        "  - category: object category (e.g. 'Furniture', 'Container', 'Clothing')\n"
        "  - sam_prompt: a 1-4 word phrase a segmentation model can use to find this object "
        "(e.g. 'blue suitcase', 'wooden dresser')\n\n"
        "Be thorough — list every distinct manipulatable object you see."
    )

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[_build_gemini_video_part(genai.types, uploaded, fps), combined_prompt],
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
            detected_objects = [obj for obj in raw_objects if isinstance(obj, dict)]

        return _GeminiResult(
            environment=room_type,
            confidence=confidence,
            model=model,
            raw_text=raw_text,
            detected_objects=detected_objects,
        )

    return None


def infer_scene_semantics(
    *,
    frames_dir: Path,
    raw_video_path: Optional[Path] = None,
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

    # Honor explicit hints to avoid external inference/egress when the
    # operator selected an environment directly.
    if has_explicit_hint:
        return {
            "schema_version": "v1",
            "generated_at": _utc_now_iso(),
            "requested_environment": requested,
            "resolved_environment": normalized_requested,
            "environment_source": "explicit_hint",
            "environment_confidence": 1.0,
            "prompt_source": "explicit_hint",
            "detection_prompts": list(_PROMPTS_BY_ENV[normalized_requested]),
            "keyframes_used": [],
            "fallback_reason": "",
        }

    gemini_result = None
    inference_mode = "video_file_upload"
    if raw_video_path is not None and raw_video_path.is_file():
        gemini_result = _infer_with_gemini_video(
            raw_video_path=raw_video_path,
            timeout_sec=max(5, int(timeout_sec)),
        )

    if gemini_result is not None:
        resolved = _normalize_environment(gemini_result.environment)

        # Use Gemini-enumerated objects as SAM prompts if available,
        # otherwise fall back to hardcoded environment prompts.
        if gemini_result.detected_objects:
            sam_prompts = []
            for obj in gemini_result.detected_objects:
                if not isinstance(obj, dict):
                    continue
                prompt = (obj.get("sam_prompt") or obj.get("object_id") or "")
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                prompt = prompt.strip()
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
            "gemini_inference_mode": inference_mode,
            "gemini_video_analysis_fps": _gemini_video_analysis_fps(),
            "keyframes_used": [],
            "fallback_reason": "",
            "explicit_hint": normalized_requested if has_explicit_hint else None,
        }

    failure_reason = "gemini_video_unavailable_or_failed"
    if raw_video_path is None or not raw_video_path.is_file():
        failure_reason = "raw_walkthrough_video_missing"

    return {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "requested_environment": requested or "auto",
        "resolved_environment": "default",
        "environment_source": "local_auto_fallback",
        "environment_confidence": 0.35,
        "prompt_source": "auto_fallback",
        "detection_prompts": list(_PROMPTS_BY_ENV["default"]),
        "keyframes_used": [],
        "fallback_reason": failure_reason,
    }


def write_scene_semantics_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _bounded_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        return []
    out: List[str] = []
    for item in items:
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _normalize_assessment(
    raw_value: Any,
    *,
    default_score: float,
    default_status: str,
    default_summary: str,
    default_impact: str,
) -> Dict[str, Any]:
    raw = raw_value if isinstance(raw_value, Mapping) else {}
    status = str(raw.get("status") or default_status).strip() or default_status
    summary = str(raw.get("summary") or default_summary).strip() or default_summary
    impact = str(raw.get("impact") or default_impact).strip() or default_impact
    return {
        "status": status,
        "score": _bounded_score(raw.get("score"), default=default_score),
        "summary": summary,
        "impact": impact,
    }


def _gemini_capture_review_prompt(
    *,
    descriptor: Mapping[str, Any],
    qa_report: Mapping[str, Any],
    task_hypothesis_report: Optional[Mapping[str, Any]],
    capture_context: Optional[Mapping[str, Any]] = None,
) -> str:
    return (
        "You are reviewing a real-world capture for Blueprint qualification.\n"
        "Use the raw walkthrough video as the primary evidence source and the structured capture context as supporting evidence.\n"
        "Use the visual evidence conservatively. Do not invent measurements or certainty.\n"
        "Assess whether this capture is good enough to support real-site robot evaluation artifacts and Post-Training Data Package support artifacts.\n"
        "Also assess whether the capture quality supports a stronger capturer payout recommendation.\n\n"
        "Return only a JSON object with this shape:\n"
        "{\n"
        '  "summary": "short string",\n'
        '  "scores": {\n'
        '    "coverage": 0.0,\n'
        '    "visual_clarity": 0.0,\n'
        '    "lighting_stability": 0.0,\n'
        '    "motion_stability": 0.0,\n'
        '    "task_understanding": 0.0,\n'
        '    "world_model_fitness": 0.0,\n'
        '    "payout_quality": 0.0\n'
        "  },\n"
        '  "bonus_signals": {\n'
        '    "complete_coverage": {"score": 0.0, "reason": "..."},\n'
        '    "multi_pass": {"score": 0.0, "reason": "..."},\n'
        '    "lidar_depth": {"score": 0.0, "reason": "..."},\n'
        '    "steady_walkthrough": {"score": 0.0, "reason": "..."}\n'
        "  },\n"
        '  "blur_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "lighting_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "motion_speed_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "doubling_back_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "coverage_completeness_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "task_zone_completeness_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "occlusion_and_hidden_zone_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "depth_and_spatial_conditioning_assessment": {"status": "good|review_required|poor", "score": 0.0, "summary": "...", "impact": "..."},\n'
        '  "missing_views": ["..."],\n'
        '  "blur_observations": ["..."],\n'
        '  "lighting_observations": ["..."],\n'
        '  "occlusion_observations": ["..."],\n'
        '  "task_scope_notes": ["..."],\n'
        '  "blocker_summaries": ["..."],\n'
        '  "recapture_recommendations": ["..."],\n'
        '  "world_model_recommendation": "good_candidate|review_required|not_recommended",\n'
        '  "payout_recommendation": "bonus|baseline|discount|review_required",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Review the actual walkthrough quality from the video. Pay special attention to blur, lighting changes, camera speed, doubling back/rescans, whether all of the scene is covered, task-zone completeness, hidden zones, and whether the depth/spatial evidence is sufficient for evaluation and data-package support artifacts.\n"
        "If quality is poor, recommend recapture and explain why.\n\n"
        f"Descriptor:\n{json.dumps(dict(descriptor), indent=2, sort_keys=True)}\n\n"
        f"QA report:\n{json.dumps(dict(qa_report), indent=2, sort_keys=True)}\n\n"
        f"Task hypothesis:\n{json.dumps(dict(task_hypothesis_report or {}), indent=2, sort_keys=True)}\n\n"
        f"Capture context:\n{json.dumps(dict(capture_context or {}), indent=2, sort_keys=True)}\n"
    )


def _infer_capture_review_with_gemini_video(
    *,
    raw_video_path: Path,
    descriptor: Mapping[str, Any],
    qa_report: Mapping[str, Any],
    task_hypothesis_report: Optional[Mapping[str, Any]],
    capture_context: Optional[Mapping[str, Any]],
    timeout_sec: int,
) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or not raw_video_path.is_file():
        return None

    try:
        from google import genai  # type: ignore
    except ImportError:
        log_missing_optional_dependency(
            logger,
            feature="Gemini raw-video capture fidelity review",
            package="google-genai",
            extra="llm",
        )
        return None
    except Exception:
        return None

    override = (os.getenv("CAPTURE_FIDELITY_GEMINI_MODEL") or os.getenv("SCENE_SEMANTICS_GEMINI_MODEL") or "").strip()
    models_to_try = [override] if override else list(_DEFAULT_MODEL_CASCADE)
    prompt = _gemini_capture_review_prompt(
        descriptor=descriptor,
        qa_report=qa_report,
        task_hypothesis_report=task_hypothesis_report,
        capture_context=capture_context,
    )
    client = genai.Client(api_key=api_key)
    current = _upload_gemini_video_file(client, raw_video_path, timeout_sec)
    if current is None:
        return None
    fps = _gemini_video_analysis_fps()

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[_build_gemini_video_part(genai.types, current, fps), prompt],
                config={
                    "temperature": 0.2,
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
        if not payload:
            continue
        payload["model"] = model
        payload["raw_text"] = raw_text
        payload["video_analysis_fps"] = fps
        payload["video_file_name"] = getattr(current, "name", None)
        payload["video_file_uri"] = getattr(current, "uri", None)
        return payload

    return None


def infer_capture_fidelity_review(
    *,
    capture_root: Path,
    raw_video_path: Optional[Path],
    keyframe_path: Optional[Path],
    descriptor: Mapping[str, Any],
    qa_report: Mapping[str, Any],
    task_hypothesis_report: Optional[Mapping[str, Any]] = None,
    capture_context: Optional[Mapping[str, Any]] = None,
    timeout_sec: int = 45,
) -> Dict[str, Any]:
    qa_quality = qa_report.get("quality") if isinstance(qa_report.get("quality"), Mapping) else {}
    review_mode = "video_file_upload"
    review = None
    if raw_video_path is not None and raw_video_path.is_file():
        review = _infer_capture_review_with_gemini_video(
            raw_video_path=raw_video_path,
            descriptor=descriptor,
            qa_report=qa_report,
            task_hypothesis_report=task_hypothesis_report,
            capture_context=capture_context,
            timeout_sec=max(5, int(timeout_sec)),
        )

    if review is None:
        reasons = []
        if raw_video_path is None or not raw_video_path.is_file():
            reasons.append("raw walkthrough video is missing")
        if not reasons:
            reasons.append("Gemini raw-video review is unavailable or failed")
        return {
            "schema_version": "v1",
            "review_type": "gemini_multimodal_capture_review",
            "status": "failed",
            "generated_at": _utc_now_iso(),
            "provider_name": "gemini",
            "provider_model": None,
            "review_mode": review_mode,
            "confidence": 0.0,
            "summary": "Gemini multimodal review did not complete.",
            "raw_video_present": bool(raw_video_path and raw_video_path.is_file()),
            "raw_video_path": str(raw_video_path) if raw_video_path else None,
            "keyframes_used": [],
            "scores": {
                "coverage": 0.0,
                "visual_clarity": 0.0,
                "lighting_stability": 0.0,
                "motion_stability": 0.0,
                "task_understanding": 0.0,
                "world_model_fitness": 0.0,
                "payout_quality": 0.0,
            },
            "bonus_signals": {
                "complete_coverage": {"score": 0.0, "reason": "Gemini review unavailable."},
                "multi_pass": {"score": 0.0, "reason": "Gemini review unavailable."},
                "lidar_depth": {"score": 0.0, "reason": "Gemini review unavailable."},
                "steady_walkthrough": {"score": 0.0, "reason": "Gemini review unavailable."},
            },
            "assessments": {
                "blur": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Blur quality could not be reviewed.", default_impact="Manual review is required."),
                "lighting": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Lighting quality could not be reviewed.", default_impact="Manual review is required."),
                "motion_speed": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Motion speed could not be reviewed.", default_impact="Manual review is required."),
                "doubling_back": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Doubling-back patterns could not be reviewed.", default_impact="Manual review is required."),
                "coverage_completeness": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Coverage completeness could not be reviewed.", default_impact="Manual review is required."),
                "task_zone_completeness": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Task-zone completeness could not be reviewed.", default_impact="Manual review is required."),
                "occlusion_and_hidden_zone": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Hidden-zone quality could not be reviewed.", default_impact="Manual review is required."),
                "depth_and_spatial_conditioning": _normalize_assessment(None, default_score=0.0, default_status="review_required", default_summary="Depth and spatial conditioning could not be reviewed.", default_impact="Manual review is required."),
            },
            "findings": {
                "missing_views": [],
                "blur_observations": [],
                "lighting_observations": [],
                "occlusion_observations": [],
                "task_scope_notes": [],
                "blocker_summaries": reasons,
                "recapture_recommendations": reasons,
            },
            "recommendations": {
                "world_model_recommendation": "review_required",
                "payout_recommendation": "review_required",
            },
            "provenance": {
                "provider_name": "gemini",
                "provider_model": None,
                "raw_response": None,
                "input_mode": review_mode,
                "keyframes_used": [],
                "raw_video_path": str(raw_video_path) if raw_video_path else None,
                "video_analysis_fps": _gemini_video_analysis_fps(),
            },
        }

    scores_raw = review.get("scores") if isinstance(review.get("scores"), Mapping) else {}
    bonus_signals_raw = review.get("bonus_signals") if isinstance(review.get("bonus_signals"), Mapping) else {}
    findings = {
        "missing_views": _string_list(review.get("missing_views")),
        "blur_observations": _string_list(review.get("blur_observations")),
        "lighting_observations": _string_list(review.get("lighting_observations")),
        "occlusion_observations": _string_list(review.get("occlusion_observations")),
        "task_scope_notes": _string_list(review.get("task_scope_notes")),
        "blocker_summaries": _string_list(review.get("blocker_summaries")),
        "recapture_recommendations": _string_list(review.get("recapture_recommendations")),
    }
    assessments = {
        "blur": _normalize_assessment(
            review.get("blur_assessment"),
            default_score=_bounded_score(scores_raw.get("visual_clarity"), 0.0),
            default_status="good" if _bounded_score(scores_raw.get("visual_clarity"), 0.0) >= 0.7 else "review_required",
            default_summary="Gemini assessed image sharpness and blur from the walkthrough.",
            default_impact="Blur affects the visual quality of evaluation and data-package support artifacts.",
        ),
        "lighting": _normalize_assessment(
            review.get("lighting_assessment"),
            default_score=_bounded_score(scores_raw.get("lighting_stability"), 0.0),
            default_status="good" if _bounded_score(scores_raw.get("lighting_stability"), 0.0) >= 0.7 else "review_required",
            default_summary="Gemini assessed lighting stability through the walkthrough.",
            default_impact="Lighting instability can reduce downstream evaluation and data-package support quality.",
        ),
        "motion_speed": _normalize_assessment(
            review.get("motion_speed_assessment"),
            default_score=_bounded_score(scores_raw.get("motion_stability"), 0.0),
            default_status="good" if _bounded_score(scores_raw.get("motion_stability"), 0.0) >= 0.7 else "review_required",
            default_summary="Gemini assessed camera speed and pacing through the walkthrough.",
            default_impact="Excessive speed can reduce usable reconstruction evidence.",
        ),
        "doubling_back": _normalize_assessment(
            review.get("doubling_back_assessment"),
            default_score=_bounded_score((bonus_signals_raw.get("multi_pass") or {}).get("score") if isinstance(bonus_signals_raw.get("multi_pass"), Mapping) else None, default=0.0),
            default_status="good",
            default_summary="Gemini assessed whether doubling back improved or harmed scene coverage.",
            default_impact="Repeated rescans can either strengthen coverage or indicate inefficient capture.",
        ),
        "coverage_completeness": _normalize_assessment(
            review.get("coverage_completeness_assessment"),
            default_score=_bounded_score(scores_raw.get("coverage"), 0.0),
            default_status="good" if _bounded_score(scores_raw.get("coverage"), 0.0) >= 0.7 else "review_required",
            default_summary="Gemini assessed whether the walkthrough captured the full scene and task area.",
            default_impact="Incomplete coverage often requires recapture before world-model work.",
        ),
        "task_zone_completeness": _normalize_assessment(
            review.get("task_zone_completeness_assessment"),
            default_score=_bounded_score(scores_raw.get("task_understanding"), 0.0),
            default_status="good" if _bounded_score(scores_raw.get("task_understanding"), 0.0) >= 0.7 else "review_required",
            default_summary="Gemini assessed whether the task-relevant zone was fully captured.",
            default_impact="Weak task-zone coverage reduces buyer confidence and world-model fitness.",
        ),
        "occlusion_and_hidden_zone": _normalize_assessment(
            review.get("occlusion_and_hidden_zone_assessment"),
            default_score=1.0 if not findings["occlusion_observations"] else 0.4,
            default_status="good" if not findings["occlusion_observations"] else "review_required",
            default_summary="Gemini assessed occlusions and hidden zones in the walkthrough.",
            default_impact="Hidden or occluded areas can block evaluation and data-package support quality.",
        ),
        "depth_and_spatial_conditioning": _normalize_assessment(
            review.get("depth_and_spatial_conditioning_assessment"),
            default_score=_bounded_score((bonus_signals_raw.get("lidar_depth") or {}).get("score") if isinstance(bonus_signals_raw.get("lidar_depth"), Mapping) else None, default=0.0),
            default_status="good" if str(descriptor.get("capture_modality") or "").strip() == "iphone_arkit_lidar" else "review_required",
            default_summary="Gemini assessed whether depth and spatial evidence are strong enough for downstream evaluation and data-package support work.",
            default_impact="Weak spatial conditioning can block high-quality evaluation and data-package support artifacts.",
        ),
    }
    return {
        "schema_version": "v1",
        "review_type": "gemini_multimodal_capture_review",
        "status": "succeeded",
        "generated_at": _utc_now_iso(),
        "provider_name": "gemini",
        "provider_model": str(review.get("model") or "").strip() or None,
        "review_mode": review_mode,
        "confidence": _bounded_score(review.get("confidence"), default=0.0),
        "summary": str(review.get("summary") or "Gemini reviewed the capture evidence.").strip(),
        "raw_video_present": bool(raw_video_path and raw_video_path.is_file()),
        "raw_video_path": str(raw_video_path) if raw_video_path else None,
        "keyframes_used": [],
        "scores": {
            "coverage": _bounded_score(scores_raw.get("coverage"), default=_bounded_score(qa_quality.get("pose_match_rate"), 0.0)),
            "visual_clarity": _bounded_score(scores_raw.get("visual_clarity"), default=0.7),
            "lighting_stability": _bounded_score(scores_raw.get("lighting_stability"), default=0.7),
            "motion_stability": _bounded_score(scores_raw.get("motion_stability"), default=0.7),
            "task_understanding": _bounded_score(scores_raw.get("task_understanding"), default=0.6),
            "world_model_fitness": _bounded_score(scores_raw.get("world_model_fitness"), default=0.5),
            "payout_quality": _bounded_score(scores_raw.get("payout_quality"), default=0.5),
        },
        "bonus_signals": {
            "complete_coverage": {
                "score": _bounded_score(
                    (bonus_signals_raw.get("complete_coverage") or {}).get("score")
                    if isinstance(bonus_signals_raw.get("complete_coverage"), Mapping)
                    else None,
                    default=_bounded_score(scores_raw.get("coverage"), 0.0),
                ),
                "reason": str(
                    (bonus_signals_raw.get("complete_coverage") or {}).get("reason")
                    if isinstance(bonus_signals_raw.get("complete_coverage"), Mapping)
                    else ""
                ).strip() or "Gemini assessed task-zone coverage from the capture.",
            },
            "multi_pass": {
                "score": _bounded_score(
                    (bonus_signals_raw.get("multi_pass") or {}).get("score")
                    if isinstance(bonus_signals_raw.get("multi_pass"), Mapping)
                    else None,
                    default=0.0,
                ),
                "reason": str(
                    (bonus_signals_raw.get("multi_pass") or {}).get("reason")
                    if isinstance(bonus_signals_raw.get("multi_pass"), Mapping)
                    else ""
                ).strip() or "Gemini estimated whether the capture revisited key areas from multiple angles.",
            },
            "lidar_depth": {
                "score": _bounded_score(
                    (bonus_signals_raw.get("lidar_depth") or {}).get("score")
                    if isinstance(bonus_signals_raw.get("lidar_depth"), Mapping)
                    else None,
                    default=1.0 if str(descriptor.get("capture_modality") or "").strip() == "iphone_arkit_lidar" else 0.0,
                ),
                "reason": str(
                    (bonus_signals_raw.get("lidar_depth") or {}).get("reason")
                    if isinstance(bonus_signals_raw.get("lidar_depth"), Mapping)
                    else ""
                ).strip() or "Gemini assessed whether the capture supports LiDAR/depth-backed review quality.",
            },
            "steady_walkthrough": {
                "score": _bounded_score(
                    (bonus_signals_raw.get("steady_walkthrough") or {}).get("score")
                    if isinstance(bonus_signals_raw.get("steady_walkthrough"), Mapping)
                    else None,
                    default=_bounded_score(scores_raw.get("motion_stability"), 0.0),
                ),
                "reason": str(
                    (bonus_signals_raw.get("steady_walkthrough") or {}).get("reason")
                    if isinstance(bonus_signals_raw.get("steady_walkthrough"), Mapping)
                    else ""
                ).strip() or "Gemini assessed walkthrough steadiness and pacing from the capture.",
            },
        },
        "assessments": assessments,
        "findings": findings,
        "recommendations": {
            "world_model_recommendation": str(review.get("world_model_recommendation") or "review_required").strip() or "review_required",
            "payout_recommendation": str(review.get("payout_recommendation") or "review_required").strip() or "review_required",
        },
        "provenance": {
            "provider_name": "gemini",
            "provider_model": str(review.get("model") or "").strip() or None,
            "raw_response": review.get("raw_text"),
            "input_mode": review_mode,
            "keyframes_used": [],
            "raw_video_path": str(raw_video_path) if raw_video_path else None,
            "video_analysis_fps": review.get("video_analysis_fps"),
            "video_file_name": review.get("video_file_name"),
            "video_file_uri": review.get("video_file_uri"),
        },
    }
