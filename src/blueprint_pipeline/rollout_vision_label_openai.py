"""OpenAI-backed rollout vision labeling command hook.

This command is intended to be called by ``blueprint-ingest-arena-results`` via
``BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND``. It writes
``rollout_vision_labels.command.json`` in the Arena package output directory.
Labels are always review-required and never upgrade robot, contact, safety, or
readiness proof.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .arena_result_ingest import COMMAND_VISION_LABELS_SCHEMA_VERSION
from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .openai_successor_models import OPENAI_REASONING_EFFORT, OPENAI_TEXT_MODEL


OUTPUT_FILENAME = "rollout_vision_labels.command.json"
GATE_ENV = "BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING"
MODEL_ENV = "BLUEPRINT_ROLLOUT_VISION_OPENAI_MODEL"
DEFAULT_MODEL = OPENAI_TEXT_MODEL


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _clip_by_attempt(clips_manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    clips: Dict[str, Dict[str, Any]] = {}
    for clip in clips_manifest.get("clips", []) or []:
        if not isinstance(clip, Mapping):
            continue
        attempt_id = _string(clip.get("attempt_id"))
        if attempt_id:
            clips[attempt_id] = dict(clip)
    return clips


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.")
    return cleaned or "attempt"


def _extract_keyframe(*, output_dir: Path, clip: Mapping[str, Any], clip_id: str) -> Dict[str, Any]:
    clip_path_text = _string(clip.get("clip_path") or clip.get("source_video_path"))
    if not clip_path_text:
        return {"status": "blocked", "reason": "missing_clip_path", "path": None}
    clip_path = Path(clip_path_text)
    if not clip_path.is_absolute():
        clip_path = output_dir / clip_path
    if not clip_path.is_file():
        return {"status": "blocked", "reason": "clip_path_not_found", "path": str(clip_path)}
    keyframe_dir = output_dir / "vision_keyframes"
    ensure_dir(keyframe_dir)
    keyframe_path = keyframe_dir / f"{_safe_stem(clip_id)}.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(keyframe_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return {"status": "blocked", "reason": "missing_ffmpeg", "path": str(clip_path)}
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "reason": "ffmpeg_timeout", "path": str(clip_path)}
    if completed.returncode != 0 or not keyframe_path.is_file():
        return {
            "status": "blocked",
            "reason": "ffmpeg_failed",
            "path": str(clip_path),
            "stderr": completed.stderr[-500:],
        }
    return {
        "status": "completed",
        "path": str(keyframe_path),
        "relative_path": str(keyframe_path.relative_to(output_dir)),
    }


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_json_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _openai_label(
    *,
    model: str,
    label: Mapping[str, Any],
    clip: Mapping[str, Any],
    keyframe_path: Path,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("missing_openai_package") from exc

    prompt = {
        "instruction": (
            "Inspect this rollout keyframe for a robotics evaluation failure. "
            "Return compact JSON only. The result is review-required evidence, not proof."
        ),
        "required_json_keys": [
            "object_state",
            "contact",
            "occlusion",
            "threshold_miss",
            "failure_evidence",
            "confidence",
        ],
        "label_context": {
            "label_id": label.get("label_id"),
            "attempt_id": label.get("attempt_id"),
            "failure_categories": label.get("failure_categories"),
            "threshold_miss": label.get("threshold_miss"),
            "clip_id": clip.get("clip_id"),
            "scenario_id": clip.get("scenario_id"),
        },
    }
    client = OpenAI()
    response = client.responses.create(
        model=model,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(prompt, sort_keys=True)},
                    {"type": "input_image", "image_url": _data_url(keyframe_path)},
                ],
            }
        ],
    )
    payload = _parse_json_text(getattr(response, "output_text", "") or "{}")
    return payload


def _fallback_label(
    *,
    label: Mapping[str, Any],
    clip: Mapping[str, Any],
    keyframe: Mapping[str, Any],
    model: str,
    openai_payload: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    source_label_id = _string(label.get("label_id"))
    attempt_id = _string(label.get("attempt_id"))
    evidence = openai_payload or {}
    keyframe_completed = keyframe.get("status") == "completed"
    return {
        "vision_label_id": f"vision_{source_label_id or attempt_id}",
        "source_failure_label_id": source_label_id or None,
        "attempt_id": attempt_id or None,
        "status": "review_required",
        "masks": [],
        "object_state": _string(evidence.get("object_state")) or "review_required",
        "contact": _string(evidence.get("contact")) or "review_required",
        "occlusion": _string(evidence.get("occlusion")) or "review_required",
        "threshold_miss": bool(evidence.get("threshold_miss", label.get("threshold_miss"))),
        "failure_evidence": evidence.get("failure_evidence")
        if isinstance(evidence.get("failure_evidence"), list)
        else label.get("failure_categories", []),
        "confidence": evidence.get("confidence") if isinstance(evidence.get("confidence"), (int, float)) else None,
        "label_source": "openai_responses_vision",
        "model": model,
        "reasoning_effort": OPENAI_REASONING_EFFORT,
        "visual_evidence_used": keyframe_completed,
        "evidence_refs": [
            ref
            for ref in (
                clip.get("clip_path"),
                keyframe.get("relative_path"),
            )
            if ref
        ],
        "proof_effect": "none_until_human_review_or_owner_proof",
    }


def build_openai_rollout_vision_labels(
    *,
    output_dir: str | Path = ".",
    model: str | None = None,
    max_labels: int = 20,
    require_visual_evidence: bool = True,
) -> Dict[str, Any]:
    resolved_output = Path(output_dir).resolve()
    generated_at = utc_now_iso()
    model_name = _string(model or os.getenv(MODEL_ENV)) or DEFAULT_MODEL
    failure_labels = _read_mapping(resolved_output / "failure_labels.json")
    clips_manifest = _read_mapping(resolved_output / "clips_manifest.json")
    clips = _clip_by_attempt(clips_manifest)
    raw_labels = [
        item
        for item in failure_labels.get("labels", []) or []
        if isinstance(item, Mapping)
    ][:max_labels]
    blockers: List[str] = []
    labels: List[Dict[str, Any]] = []
    keyframes: List[Dict[str, Any]] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    if not _string(os.getenv("OPENAI_API_KEY")):
        blockers.append("missing_openai_api_key")
    if not raw_labels:
        blockers.append("missing_failure_labels")

    for raw_label in raw_labels:
        attempt_id = _string(raw_label.get("attempt_id"))
        clip = clips.get(attempt_id, {})
        clip_id = _string(clip.get("clip_id")) or f"clip_{attempt_id}"
        keyframe = _extract_keyframe(output_dir=resolved_output, clip=clip, clip_id=clip_id)
        keyframes.append({"attempt_id": attempt_id, "clip_id": clip_id, **keyframe})
        if require_visual_evidence and keyframe["status"] != "completed":
            continue
        if blockers:
            continue
        try:
            openai_payload = _openai_label(
                model=model_name,
                label=raw_label,
                clip=clip,
                keyframe_path=Path(str(keyframe["path"])),
            )
        except Exception as exc:  # pragma: no cover - live provider behavior
            blockers.append(f"openai_labeling_failed:{type(exc).__name__}")
            break
        labels.append(
            _fallback_label(
                label=raw_label,
                clip=clip,
                keyframe=keyframe,
                model=model_name,
                openai_payload=openai_payload,
            )
        )

    if require_visual_evidence and raw_labels and not any(
        item.get("status") == "completed" for item in keyframes
    ):
        blockers.append("missing_visual_evidence_keyframes")

    manifest = {
        "schema_version": COMMAND_VISION_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed_review_required" if labels and not blockers else "blocked_review_required",
        "provider": "openai",
        "model": model_name,
        "reasoning_effort": OPENAI_REASONING_EFFORT,
        "blockers": sorted(set(blockers)),
        "label_count": len(labels),
        "labels": labels,
        "keyframes": keyframes,
        "visual_evidence_used": any(item.get("visual_evidence_used") for item in labels),
        "human_review_required": bool(labels or raw_labels),
        "public_claim_upgrade_allowed": False,
        "proof_effect": "none_until_human_review_or_owner_proof",
    }
    write_json(resolved_output / OUTPUT_FILENAME, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate review-required rollout vision labels with OpenAI Responses"
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-labels", type=int, default=20)
    parser.add_argument(
        "--allow-missing-visual-evidence",
        action="store_true",
        help="Do not require extracted keyframes. Intended only for debugging blocked manifests.",
    )
    args = parser.parse_args(argv)
    result = build_openai_rollout_vision_labels(
        output_dir=args.output_dir,
        model=args.model,
        max_labels=args.max_labels,
        require_visual_evidence=not args.allow_missing_visual_evidence,
    )
    print(f"[rollout-vision-openai] manifest={Path(args.output_dir).resolve() / OUTPUT_FILENAME}")
    print(f"[rollout-vision-openai] status={result['status']}")
    if result["blockers"]:
        print(f"[rollout-vision-openai] blockers={len(result['blockers'])}")
    return 0 if result["status"] == "completed_review_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
