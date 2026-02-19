#!/usr/bin/env python3
"""Dimension-completion runner used by SAM3 occlusion handling.

This command reads object context from CLI args or SAM3_COMPLETION_* env vars,
then returns JSON with full-object size estimates in meters:

{
  "predicted_extents_m": [x, y, z],
  "confidence": 0.0-1.0,
  "model": "...",
  "reason": "..."
}

Strategy:
1. Gemini (if API key + SDK available)
2. Image-prior fallback (label priors + crop alpha coverage)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_DEFAULT_GEMINI_MODEL = (
    os.getenv("SAM3_DIM_COMPLETION_RUNNER_GEMINI_MODEL") or "gemini-2.5-flash"
).strip()
_DEFAULT_MAX_IMAGES = max(1, int(os.getenv("SAM3_DIM_COMPLETION_RUNNER_MAX_IMAGES", "2") or "2"))
_DISABLE_GEMINI = (os.getenv("SAM3_DIM_COMPLETION_RUNNER_DISABLE_GEMINI") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


_LABEL_PRIORS_M: Dict[str, List[float]] = {
    "bed": [2.0, 0.6, 1.5],
    "nightstand": [0.5, 0.55, 0.45],
    "dresser": [1.4, 1.0, 0.55],
    "wardrobe": [1.2, 2.0, 0.6],
    "cabinet": [0.9, 0.9, 0.5],
    "drawer": [0.55, 0.18, 0.4],
    "desk": [1.2, 0.75, 0.6],
    "chair": [0.55, 0.95, 0.55],
    "table": [1.2, 0.75, 0.8],
    "box": [0.55, 0.45, 0.4],
    "container": [0.5, 0.4, 0.35],
    "basket": [0.45, 0.4, 0.35],
    "hamper": [0.45, 0.65, 0.45],
    "suitcase": [0.7, 0.5, 0.25],
    "backpack": [0.35, 0.5, 0.22],
    "door": [0.9, 2.0, 0.08],
    "refrigerator": [0.9, 1.8, 0.8],
    "fridge": [0.9, 1.8, 0.8],
    "microwave": [0.5, 0.33, 0.4],
    "oven": [0.75, 0.9, 0.65],
    "tv": [1.2, 0.75, 0.08],
    "monitor": [0.6, 0.4, 0.08],
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _normalize_extents(values: List[Any]) -> List[float]:
    out = [_clamp(_safe_float(values[idx], 0.35), 0.02, 8.0) for idx in range(3)]
    return [round(v, 4) for v in out]


def _parse_json_object(text: str) -> Dict[str, Any]:
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


def _parse_extents(payload: Mapping[str, Any]) -> Optional[List[float]]:
    keys = ("predicted_extents_m", "extents_m", "dimensions_m", "extents")
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, list) and len(raw) >= 3:
            return _normalize_extents([raw[0], raw[1], raw[2]])
        if isinstance(raw, Mapping):
            return _normalize_extents([raw.get("x"), raw.get("y"), raw.get("z")])
    nested = payload.get("prediction")
    if isinstance(nested, Mapping):
        return _parse_extents(nested)
    return None


def _coerce_observed_extents(text: str) -> List[float]:
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, list) and len(payload) >= 3:
        return _normalize_extents([payload[0], payload[1], payload[2]])
    return [0.35, 0.35, 0.35]


def _coerce_crop_paths(text: str) -> List[Path]:
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if not isinstance(payload, list):
        return []
    out: List[Path] = []
    for item in payload:
        path = Path(str(item).strip())
        if path and path.is_file():
            out.append(path)
    return out


def _alpha_coverage(path: Path) -> Optional[float]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        img = Image.open(path)
    except Exception:
        return None
    if "A" not in img.getbands():
        return None
    alpha = img.getchannel("A")
    data = list(alpha.getdata())
    if not data:
        return None
    nonzero = sum(1 for value in data if int(value) > 0)
    return _clamp(float(nonzero) / float(len(data)), 0.0, 1.0)


def _infer_with_gemini(
    *,
    label: str,
    environment: str,
    observed_extents: List[float],
    crop_paths: List[Path],
    model: str,
    max_images: int,
) -> Optional[Dict[str, Any]]:
    if _DISABLE_GEMINI:
        return None
    api_key = (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        from google import genai  # type: ignore
    except Exception:
        return None

    prompt = (
        "Estimate full object extents in meters for a partially visible object.\n"
        f"Label: {label}\n"
        f"Environment: {environment}\n"
        f"Observed extents meters [x,y,z]: {json.dumps(observed_extents)}\n\n"
        "Return strict JSON only:\n"
        "{\"predicted_extents_m\":[x,y,z],\"confidence\":0.0-1.0,"
        "\"reason\":\"short reason\"}\n"
        "Each extent must be in [0.02, 8.0]."
    )
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for path in crop_paths[:max_images]:
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
            model=model,
            contents=[{"parts": parts}],
            config={
                "temperature": 0.1,
                "max_output_tokens": 512,
                "response_mime_type": "application/json",
            },
        )
    except Exception:
        return None

    payload = _parse_json_object(_extract_response_text(response))
    extents = _parse_extents(payload)
    if extents is None:
        return None
    confidence = _clamp(_safe_float(payload.get("confidence"), 0.0), 0.0, 1.0)
    reason = str(payload.get("reason") or payload.get("rationale") or "").strip()
    return {
        "predicted_extents_m": extents,
        "confidence": round(confidence, 4),
        "model": model,
        "reason": reason or "gemini_dimension_estimate",
    }


def _select_label_prior(label: str) -> Optional[List[float]]:
    text = label.strip().lower()
    if not text:
        return None
    for key, prior in _LABEL_PRIORS_M.items():
        if key in text:
            return list(prior)
    return None


def _infer_with_image_priors(
    *,
    label: str,
    environment: str,
    observed_extents: List[float],
    crop_paths: List[Path],
) -> Dict[str, Any]:
    observed = _normalize_extents(observed_extents)
    prior = _select_label_prior(label)

    if prior is not None:
        base = [max(observed[idx], prior[idx]) for idx in range(3)]
        confidence = 0.54
        reason = "label_prior_fallback"
    else:
        base = [observed[idx] * 1.15 for idx in range(3)]
        confidence = 0.42
        reason = "generic_prior_fallback"

    coverage = None
    if crop_paths:
        coverage = _alpha_coverage(crop_paths[0])
    if coverage is not None:
        occlusion_scale = _clamp(1.0 / max(coverage, 0.45), 1.0, 1.45)
        base = [value * occlusion_scale for value in base]
        confidence = _clamp(confidence + (0.08 if coverage < 0.65 else 0.03), 0.35, 0.72)
        reason = f"{reason}_alpha_coverage_{coverage:.2f}"

    env = environment.strip().lower()
    env_scale = 1.08 if env == "warehouse" else 1.0
    predicted = _normalize_extents(
        [max(observed[idx], base[idx] * env_scale) for idx in range(3)]
    )

    return {
        "predicted_extents_m": predicted,
        "confidence": round(confidence, 4),
        "model": "image_prior_v1",
        "reason": reason,
    }


def run_dimension_completion(
    *,
    label: str,
    environment: str,
    observed_extents: List[float],
    crop_paths: List[Path],
    provider_mode: str,
    model: str,
    max_images: int,
) -> Dict[str, Any]:
    mode = provider_mode.strip().lower()
    if mode not in {"auto", "gemini", "image_prior"}:
        mode = "auto"

    if mode in {"auto", "gemini"}:
        gemini_result = _infer_with_gemini(
            label=label,
            environment=environment,
            observed_extents=observed_extents,
            crop_paths=crop_paths,
            model=model,
            max_images=max_images,
        )
        if gemini_result is not None:
            return gemini_result
        if mode == "gemini":
            return {
                "predicted_extents_m": _normalize_extents(observed_extents),
                "confidence": 0.2,
                "model": model,
                "reason": "gemini_unavailable",
            }

    return _infer_with_image_priors(
        label=label,
        environment=environment,
        observed_extents=observed_extents,
        crop_paths=crop_paths,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM3 dimension completion runner")
    parser.add_argument("--label", default=os.getenv("SAM3_COMPLETION_LABEL", "object"))
    parser.add_argument(
        "--environment",
        default=os.getenv("SAM3_COMPLETION_ENVIRONMENT", "default"),
    )
    parser.add_argument(
        "--observed-extents-json",
        default=os.getenv("SAM3_COMPLETION_OBSERVED_EXTENTS_JSON", "[0.35,0.35,0.35]"),
    )
    parser.add_argument(
        "--crop-paths-json",
        default=os.getenv("SAM3_COMPLETION_CROP_PATHS_JSON", "[]"),
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("SAM3_DIM_COMPLETION_RUNNER_PROVIDER", "auto"),
        choices=["auto", "gemini", "image_prior"],
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SAM3_DIM_COMPLETION_RUNNER_GEMINI_MODEL", _DEFAULT_GEMINI_MODEL),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=max(1, int(os.getenv("SAM3_DIM_COMPLETION_RUNNER_MAX_IMAGES", str(_DEFAULT_MAX_IMAGES)))),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    observed_extents = _coerce_observed_extents(args.observed_extents_json)
    crop_paths = _coerce_crop_paths(args.crop_paths_json)
    result = run_dimension_completion(
        label=str(args.label or "object"),
        environment=str(args.environment or "default"),
        observed_extents=observed_extents,
        crop_paths=crop_paths,
        provider_mode=str(args.provider or "auto"),
        model=str(args.model or _DEFAULT_GEMINI_MODEL),
        max_images=max(1, int(args.max_images)),
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
