"""Agentic Gemini removal-target analysis for the pre-reconstruction clean-plate stage.

Consumes the privacy-safe walkthrough video and produces a per-target removal
plan (``clean_plate_removal_plan.v1``): which people, movable task-objects, and
fixed clutter appear, and which should be removed from the reconstruction stage
plate before Atlas/WorldLabs fuses the frames. This module is the *analysis*
only -- no pixels are edited here (the view-consistent fill is a separate,
deferred stage).

This is a SEPARATE pass from the web app's ``capture-coverage`` analysis, which
deliberately never describes people or removal targets ("that is a separate
review's question"). It fills that gap on the pipeline side.

Video understanding uses Gemini's **agentic video** processing
(``processing="agentic"``; see
https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/),
so the model can search, seek, and time-localize distinct objects across a
multi-minute walkthrough instead of judging a fixed frame sample. The model and
processing mode stay env-configurable.

The paid model call is fail-closed behind
``BLUEPRINT_ALLOW_GEMINI_CLEAN_PLATE_ANALYSIS`` (precedent:
``wam_generated_video_success_label_gemini.py``, PR #180/PR #181): a missing
gate env, API key, or input video collects a blocker and NO provider call is
made -- the analysis returns ``status="blocked"`` with an empty plan.

Claim ceiling: ``development_only``. The plan is a candidate proposal. It is not
metric, collision, hidden-surface, privacy, or physical authority, and it never
authorizes removal of people -- person pixels are only ever removed by
``run_privacy_postprocess``; this pass merely records that people are present so
the clean-plate stage can verify the privacy pipeline handled them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .common import sha256_file, utc_now_iso

GATE_ENV = "BLUEPRINT_ALLOW_GEMINI_CLEAN_PLATE_ANALYSIS"
MODEL_ENV = "BLUEPRINT_GEMINI_CLEAN_PLATE_MODEL"
PROCESSING_ENV = "BLUEPRINT_GEMINI_CLEAN_PLATE_PROCESSING"
# Agentic video keeps the model deciding what to watch, at what speed, and which
# modality -- the right shape for enumerating/time-localizing distinct movers and
# task objects over a long walkthrough. Model + processing mode are overridable.
DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_PROCESSING = "agentic"

REMOVAL_PLAN_SCHEMA_VERSION = "clean_plate_removal_plan.v1"
CLAIM_CEILING = "development_only"

TARGET_CLASSES = ("person", "movable_object", "fixed_clutter")
DISPOSITIONS = ("remove", "keep")
REBUILD_INTENTS = ("rebuild_and_compose", "none")

_API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_AI_API_KEY")
_API_KEY_FILE_ENVS = (
    "GEMINI_API_KEY_FILE",
    "GOOGLE_GENAI_API_KEY_FILE",
    "GOOGLE_AI_API_KEY_FILE",
)
_SECRET_DIR = Path.home() / ".blueprint-secrets"
_SECRET_FILES = ("gemini_api_key", "google_genai_api_key", "google_ai_api_key")

PROMPT_INSTRUCTION = (
    "You are analyzing a walkthrough video of a work environment to plan which "
    "content must be removed BEFORE 3D reconstruction of the immovable scene. "
    "Multi-view reconstruction assumes the world did not move, so two kinds of "
    "content corrupt the fixed stage and must be identified: (1) people or anything "
    "that moved during the pass, and (2) manipulable task objects (totes, bins, "
    "parts, tools, clothing) that should be rebuilt separately as physics-ready "
    "assets rather than baked into the immovable stage. Return compact JSON only, "
    "an object with a single key 'targets' whose value is a list. Enumerate the "
    "DISTINCT targets across the whole video (do not repeat one object per frame). "
    "For each target provide: 'target_id' (stable short slug), 'semantic_label' "
    "(short phrase), 'target_class' (exactly one of person, movable_object, "
    "fixed_clutter), 'disposition' (remove or keep), 'rebuild_intent' "
    "(rebuild_and_compose or none), 'spatial_evidence' (a list of "
    "{timestamp_seconds, box_xywh_normalized:[x,y,w,h]} entries time-localizing the "
    "target), and 'confidence' in [0,1]. Always mark a person as remove. Mark a "
    "movable task-object as remove with rebuild_and_compose. Keep fixed clutter that "
    "is genuinely part of the environment (disposition keep, rebuild_intent none). "
    "Do NOT fabricate targets you cannot actually see: if the space appears already "
    "cleared of movable objects, return an empty targets list. Never invent a person "
    "who is not visibly present."
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_INSTRUCTION.encode("utf-8")).hexdigest()


def build_removal_plan_prompt() -> str:
    """Return the versioned analysis instruction (hashed by PROMPT_TEMPLATE_SHA256)."""

    return PROMPT_INSTRUCTION


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "y", "on"}


def _string_env(name: str, default: str) -> str:
    text = _string(os.getenv(name))
    return text or default


def _api_key() -> tuple[str, str]:
    """Resolve the Gemini API key from env or a perms-checked secret file.

    Returns ``(key, source)``; ``("", "")`` when nothing is configured. Never
    returns the key material as the source label.
    """

    for env_name in _API_KEY_ENVS:
        value = _string(os.getenv(env_name))
        if value:
            return value, env_name
    for env_name in _API_KEY_FILE_ENVS:
        file_path = _string(os.getenv(env_name))
        if file_path:
            candidate = Path(file_path).expanduser()
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text, env_name
    for secret_name in _SECRET_FILES:
        candidate = _SECRET_DIR / secret_name
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text, f"secret_file:{secret_name}"
    return "", ""


def _clamp_confidence(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _normalize_box(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(max(0.0, min(1.0, float(item))))
        except (TypeError, ValueError):
            return None
    return out


def _normalize_spatial_evidence(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return rows
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        try:
            timestamp = float(raw.get("timestamp_seconds"))
        except (TypeError, ValueError):
            timestamp = None
        box = _normalize_box(raw.get("box_xywh_normalized"))
        if timestamp is None and box is None:
            continue
        rows.append(
            {
                "timestamp_seconds": timestamp,
                "box_xywh_normalized": box,
            }
        )
    return rows


def _normalize_target(raw: Mapping[str, Any], index: int) -> Optional[dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    target_class = _string(raw.get("target_class")).lower()
    if target_class not in TARGET_CLASSES:
        return None
    disposition = _string(raw.get("disposition")).lower()
    if disposition not in DISPOSITIONS:
        # Fail safe: default a person to remove, everything else to keep.
        disposition = "remove" if target_class == "person" else "keep"
    rebuild_intent = _string(raw.get("rebuild_intent")).lower()
    if rebuild_intent not in REBUILD_INTENTS:
        rebuild_intent = "rebuild_and_compose" if (
            target_class == "movable_object" and disposition == "remove"
        ) else "none"
    target_id = _string(raw.get("target_id")) or f"{target_class}_{index:03d}"
    return {
        "target_id": target_id,
        "semantic_label": _string(raw.get("semantic_label")) or target_class,
        "target_class": target_class,
        "disposition": disposition,
        "rebuild_intent": rebuild_intent,
        "spatial_evidence": _normalize_spatial_evidence(raw.get("spatial_evidence")),
        "confidence": _clamp_confidence(raw.get("confidence")),
    }


def parse_removal_plan_response(text: str) -> list[dict[str, Any]]:
    """Parse a model JSON response into a normalized list of removal targets.

    Tolerates ```json fenced blocks. Returns only well-formed, class-valid
    targets; malformed rows are dropped rather than trusted.
    """

    cleaned = _string(text)
    if not cleaned:
        return []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if isinstance(data, Mapping):
        rows = data.get("targets")
    elif isinstance(data, list):
        rows = data
    else:
        rows = None
    if not isinstance(rows, (list, tuple)):
        return []
    targets: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        normalized = _normalize_target(raw, index)
        if normalized is not None:
            targets.append(normalized)
    return targets


def build_removal_plan(
    *,
    targets: Sequence[Mapping[str, Any]],
    status: str,
    model: str,
    processing: str,
    video_path: Optional[Path] = None,
    video_digest: Optional[str] = None,
    blockers: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Assemble a ``clean_plate_removal_plan.v1`` record."""

    normalized_targets = [dict(target) for target in targets]
    movable_removals = [
        target
        for target in normalized_targets
        if target.get("target_class") == "movable_object"
        and target.get("disposition") == "remove"
    ]
    people = [
        target for target in normalized_targets if target.get("target_class") == "person"
    ]
    return {
        "schema_version": REMOVAL_PLAN_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "provider": "gemini",
        "model": model,
        "processing": processing,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "claim_ceiling": CLAIM_CEILING,
        "blockers": sorted({_string(item) for item in (blockers or []) if _string(item)}),
        "input_video_path": str(video_path) if video_path else None,
        "input_video_sha256": video_digest,
        "target_count": len(normalized_targets),
        "movable_removal_count": len(movable_removals),
        "person_target_count": len(people),
        "targets": normalized_targets,
        # People removal is never authorized here; the privacy pipeline owns it.
        "authority_boundary": {
            "authorizes_person_pixel_removal": False,
            "is_metric_authority": False,
            "is_collision_authority": False,
            "is_hidden_surface_authority": False,
            "is_physical_evidence": False,
        },
    }


def empty_removal_plan(
    *,
    status: str,
    model: str,
    processing: str,
    blockers: Optional[Sequence[str]] = None,
    video_path: Optional[Path] = None,
    video_digest: Optional[str] = None,
) -> dict[str, Any]:
    """A well-formed plan with no targets (blocked / cleared-space case)."""

    return build_removal_plan(
        targets=[],
        status=status,
        model=model,
        processing=processing,
        video_path=video_path,
        video_digest=video_digest,
        blockers=blockers,
    )


def validate_removal_plan(plan: Mapping[str, Any]) -> list[str]:
    """Return a list of contract errors; empty means the plan is well-formed."""

    errors: list[str] = []
    if plan.get("schema_version") != REMOVAL_PLAN_SCHEMA_VERSION:
        errors.append("removal_plan_schema_version_invalid")
    if plan.get("claim_ceiling") != CLAIM_CEILING:
        errors.append("removal_plan_claim_ceiling_invalid")
    boundary = plan.get("authority_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("authorizes_person_pixel_removal") is not False:
        errors.append("removal_plan_person_authority_boundary_invalid")
    targets = plan.get("targets")
    if not isinstance(targets, list):
        errors.append("removal_plan_targets_not_a_list")
        return errors
    seen_ids: set[str] = set()
    for target in targets:
        if not isinstance(target, Mapping):
            errors.append("removal_plan_target_not_object")
            continue
        if target.get("target_class") not in TARGET_CLASSES:
            errors.append("removal_plan_target_class_invalid")
        if target.get("disposition") not in DISPOSITIONS:
            errors.append("removal_plan_disposition_invalid")
        if target.get("rebuild_intent") not in REBUILD_INTENTS:
            errors.append("removal_plan_rebuild_intent_invalid")
        target_id = _string(target.get("target_id"))
        if not target_id:
            errors.append("removal_plan_target_id_missing")
        elif target_id in seen_ids:
            errors.append("removal_plan_target_id_duplicate")
        else:
            seen_ids.add(target_id)
    return errors


def _provider_error_blocker(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "api_key_invalid" in text or "permission_denied" in text or "unauthenticated" in text:
        return "gemini_clean_plate_authentication_failed"
    if "resource_exhausted" in text or "quota" in text or "429" in text:
        return "gemini_clean_plate_quota_exhausted"
    if "not_found" in text or "unknown model" in text or "unsupported" in text:
        return "gemini_clean_plate_model_or_processing_unavailable"
    return "gemini_clean_plate_provider_error"


def _invoke_agentic_video(
    *,
    api_key: str,
    model: str,
    processing: str,
    video_path: Path,
    genai: Any,
    types: Any,
) -> str:
    """Run the paid agentic-video analysis and return the raw JSON text.

    Prefers the agentic ``interactions`` surface (``processing="agentic"``); if
    the installed SDK does not expose it, falls back to whole-video
    ``generate_content``. This path is live-only and exercised behind the gate;
    the parse/validate/build helpers above are what tests cover.
    """

    client = genai.Client(api_key=api_key)
    video_bytes = video_path.read_bytes()
    mime_type = "video/mp4" if video_path.suffix.lower() == ".mp4" else "video/quicktime"

    interactions = getattr(client, "interactions", None)
    if interactions is not None and hasattr(interactions, "create"):
        result = interactions.create(
            model=model,
            inputs=[
                {"type": "video", "mime_type": mime_type, "data": video_bytes},
                {"type": "text", "text": PROMPT_INSTRUCTION},
            ],
            config={"processing": processing, "response_mime_type": "application/json"},
        )
        text = getattr(result, "text", None)
        if text is None and hasattr(result, "output_text"):
            text = result.output_text
        return _string(text)

    part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model,
        contents=[part, PROMPT_INSTRUCTION],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return _string(getattr(response, "text", ""))


def analyze_removal_targets(
    *,
    video_path: Optional[str | Path],
    model: Optional[str] = None,
    processing: Optional[str] = None,
) -> dict[str, Any]:
    """Analyze the walkthrough and return a ``clean_plate_removal_plan.v1``.

    Fail-closed: without the gate env, an API key, and a readable input video,
    NO provider call is made and a ``status="blocked"`` empty plan is returned.
    """

    model_name = model or _string_env(MODEL_ENV, DEFAULT_MODEL)
    processing_mode = processing or _string_env(PROCESSING_ENV, DEFAULT_PROCESSING)

    blockers: list[str] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    api_key, _api_key_source = _api_key()
    if not api_key:
        blockers.append("missing_gemini_google_genai_or_google_ai_api_key_or_key_file")

    resolved_video: Optional[Path] = None
    if video_path is not None:
        candidate = Path(video_path).expanduser()
        if candidate.is_file():
            resolved_video = candidate
    if resolved_video is None:
        blockers.append("clean_plate_input_video_not_found")

    video_digest = sha256_file(resolved_video) if resolved_video is not None else None

    if blockers:
        return empty_removal_plan(
            status="blocked",
            model=model_name,
            processing=processing_mode,
            blockers=blockers,
            video_path=resolved_video,
            video_digest=video_digest,
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return empty_removal_plan(
            status="blocked",
            model=model_name,
            processing=processing_mode,
            blockers=["missing_google_genai_package"],
            video_path=resolved_video,
            video_digest=video_digest,
        )

    try:
        text = _invoke_agentic_video(
            api_key=api_key,
            model=model_name,
            processing=processing_mode,
            video_path=resolved_video,
            genai=genai,
            types=types,
        )
    except Exception as exc:  # pragma: no cover - live provider behavior
        return empty_removal_plan(
            status="blocked",
            model=model_name,
            processing=processing_mode,
            blockers=[_provider_error_blocker(exc)],
            video_path=resolved_video,
            video_digest=video_digest,
        )

    targets = parse_removal_plan_response(text)
    return build_removal_plan(
        targets=targets,
        status="completed",
        model=model_name,
        processing=processing_mode,
        video_path=resolved_video,
        video_digest=video_digest,
    )
