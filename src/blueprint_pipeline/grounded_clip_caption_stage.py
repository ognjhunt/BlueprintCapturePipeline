"""Production-gated grounded clip captions over hashed sampled frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .clip_curation_stage import load_clip_records, load_image_gray
from .common import utc_now_iso, write_json

CAPTION_SCHEMA_VERSION = "blueprint.grounded_clip_caption.v1"
MANIFEST_SCHEMA_VERSION = "blueprint.grounded_clip_caption_manifest.v1"
PROMPT_VERSION = "grounded_clip_caption_prompt.v1"
CLAIM_TYPES = ("visible_object", "task_action", "spatial_relation")

CAPTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "clip_id", "caption", "task_id", "object_ids", "claims"],
    "properties": {
        "schema_version": {"const": CAPTION_SCHEMA_VERSION},
        "clip_id": {"type": "string", "minLength": 1},
        "caption": {"type": "string", "minLength": 1, "maxLength": 500},
        "task_id": {"type": ["string", "null"]},
        "object_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_type", "text", "evidence_frame_sha256"],
                "properties": {
                    "claim_type": {"enum": list(CLAIM_TYPES)},
                    "text": {"type": "string", "minLength": 1, "maxLength": 240},
                    "object_ids": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "evidence_frame_sha256": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    },
                    "geometry_evidence_id": {"type": ["string", "null"]},
                },
            },
        },
    },
}


class CaptionProvider(Protocol):
    name: str
    version: str
    model_id: str
    revision: str
    production_ready: bool

    def caption_clip(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _clip_id(clip: Mapping[str, Any], index: int) -> str:
    return str(clip.get("clip_id") or clip.get("id") or f"clip_{index:06d}").strip()


def _string_ids(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _clip_object_ids(clip: Mapping[str, Any]) -> list[str]:
    direct = _string_ids(clip.get("object_ids"))
    if direct:
        return direct
    objects = clip.get("objects") or []
    return sorted(
        {
            str(item.get("object_id") or item.get("id") or "").strip()
            for item in objects
            if isinstance(item, Mapping)
            and str(item.get("object_id") or item.get("id") or "").strip()
        }
    )


def _sampled_frames(
    clip: Mapping[str, Any], *, bundle_dir: Path, count: int = 3
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates = [
        item
        for item in (clip.get("frames") or [])
        if isinstance(item, Mapping) and str(item.get("image_path") or "").strip()
    ]
    if not candidates:
        return [], ["caption_sample_frames_missing"]
    indices = sorted(
        {
            int(round(position))
            for position in np.linspace(
                0, len(candidates) - 1, min(count, len(candidates))
            )
        }
    )
    sampled: list[dict[str, Any]] = []
    blockers: list[str] = []
    root = bundle_dir.resolve()
    for index in indices:
        raw = Path(str(candidates[index]["image_path"]))
        path = raw if raw.is_absolute() else bundle_dir / raw
        if path.is_symlink():
            blockers.append("caption_frame_symlink_forbidden")
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            blockers.append("caption_frame_outside_bundle")
            continue
        if not resolved.is_file() or load_image_gray(resolved) is None:
            blockers.append("caption_frame_unreadable")
            continue
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        sampled.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": resolved.stat().st_size,
            }
        )
    if len(sampled) < min(3, len(candidates)):
        blockers.append("caption_sample_frame_set_incomplete")
    return sampled, sorted(set(blockers))


def _provider_approved(provider: CaptionProvider | None) -> bool:
    if provider is None or not bool(getattr(provider, "production_ready", False)):
        return False
    model_id = str(getattr(provider, "model_id", "") or "").strip()
    revision = str(getattr(provider, "revision", "") or "").strip().lower()
    return bool(model_id) and len(revision) == 40 and all(
        character in "0123456789abcdef" for character in revision
    )


def _validate_caption(
    payload: Mapping[str, Any],
    *,
    clip_id: str,
    task_id: str | None,
    allowed_object_ids: Sequence[str],
    sampled_frames: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers = [
        f"caption_schema:{error.message}"
        for error in sorted(
            Draft202012Validator(CAPTION_SCHEMA).iter_errors(dict(payload)),
            key=lambda item: list(item.path),
        )
    ]
    if payload.get("clip_id") != clip_id:
        blockers.append("caption_clip_id_mismatch")
    if payload.get("task_id") != task_id:
        blockers.append("caption_task_id_mismatch")
    allowed_objects = set(allowed_object_ids)
    declared_objects = set(_string_ids(payload.get("object_ids")))
    if not declared_objects.issubset(allowed_objects):
        blockers.append("caption_object_id_not_grounded")
    frame_hashes = {str(item.get("sha256") or "") for item in sampled_frames}
    for claim in payload.get("claims") or []:
        if not isinstance(claim, Mapping):
            continue
        if not set(_string_ids(claim.get("object_ids"))).issubset(allowed_objects):
            blockers.append("caption_claim_object_id_not_grounded")
        if not set(_string_ids(claim.get("evidence_frame_sha256"))).issubset(frame_hashes):
            blockers.append("caption_claim_frame_evidence_not_sampled")
        if claim.get("claim_type") == "spatial_relation" and not str(
            claim.get("geometry_evidence_id") or ""
        ).strip():
            blockers.append("caption_spatial_claim_missing_geometry_evidence")
    return sorted(set(blockers))


def run_grounded_clip_caption_stage(
    *,
    bundle_dir: str | Path,
    curation_manifest_path: str | Path,
    dedup_manifest_path: str | Path,
    provider: CaptionProvider | None,
    output_dir: str | Path | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    bundle = Path(bundle_dir).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else bundle / "derived" / "grounded_clip_captions"
    )
    curation = _load_mapping(Path(curation_manifest_path))
    dedup = _load_mapping(Path(dedup_manifest_path))
    clips = load_clip_records(bundle)
    blockers: list[str] = []
    if curation.get("schema_version") != "clip_curation_manifest.v1":
        blockers.append("caption_curation_manifest_missing_or_invalid")
    if dedup.get("schema_version") != "semantic_dedup_manifest.v2":
        blockers.append("caption_dedup_manifest_missing_or_invalid")
    if dedup.get("production_status") != "passed":
        blockers.append("caption_dedup_not_production_passed")
    if not _provider_approved(provider):
        blockers.append("caption_provider_not_production_approved")

    selected_ids = set(_string_ids(curation.get("accepted_clip_ids"))) & set(
        _string_ids(dedup.get("production_accepted_clip_ids"))
    )
    if not selected_ids:
        blockers.append("caption_no_canonical_accepted_clips")

    caption_records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    if not blockers and provider is not None:
        for index, clip in enumerate(clips):
            clip_id = _clip_id(clip, index)
            if clip_id not in selected_ids:
                continue
            sampled, clip_blockers = _sampled_frames(clip, bundle_dir=bundle)
            task_id = str(clip.get("task_id") or "").strip() or None
            object_ids = _clip_object_ids(clip)
            if not task_id:
                clip_blockers.append("caption_task_id_missing")
            if not object_ids:
                clip_blockers.append("caption_grounded_object_ids_missing")
            response: Mapping[str, Any] = {}
            attempts = 0
            preflight_blockers = list(clip_blockers)
            while not preflight_blockers and attempts < max(1, max_attempts):
                attempts += 1
                try:
                    response = provider.caption_clip(
                        {
                            "schema_version": "blueprint.grounded_clip_caption_request.v1",
                            "prompt_version": PROMPT_VERSION,
                            "clip_id": clip_id,
                            "task_id": task_id,
                            "allowed_object_ids": object_ids,
                            "sampled_frames": sampled,
                            "response_schema": CAPTION_SCHEMA,
                        }
                    )
                except Exception as exc:
                    clip_blockers = [f"caption_provider_error:{type(exc).__name__}"]
                    continue
                clip_blockers = _validate_caption(
                    response,
                    clip_id=clip_id,
                    task_id=task_id,
                    allowed_object_ids=object_ids,
                    sampled_frames=sampled,
                )
                if not clip_blockers:
                    break
            if clip_blockers:
                excluded.append(
                    {"clip_id": clip_id, "attempts": attempts, "blockers": sorted(set(clip_blockers))}
                )
                continue
            caption_records.append(
                {
                    **dict(response),
                    "sampled_frames": sampled,
                    "provider": {
                        "name": provider.name,
                        "version": provider.version,
                        "model_id": provider.model_id,
                        "revision": provider.revision,
                    },
                    "prompt_version": PROMPT_VERSION,
                    "attempts": attempts,
                }
            )

    accepted_ids = [str(record["clip_id"]) for record in caption_records]
    missing_ids = sorted(selected_ids - set(accepted_ids))
    if missing_ids:
        blockers.append("one_or_more_canonical_clips_missing_grounded_caption")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if selected_ids and not blockers and not excluded else "blocked",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT_VERSION.encode("utf-8")).hexdigest(),
        "source_curation_manifest": str(Path(curation_manifest_path)),
        "source_dedup_manifest": str(Path(dedup_manifest_path)),
        "selected_clip_ids": sorted(selected_ids),
        "accepted_clip_ids": accepted_ids,
        "excluded_clips": excluded,
        "captions": caption_records,
        "blockers": sorted(set(blockers)),
    }
    write_json(output / "grounded_clip_caption_manifest.json", manifest)
    return manifest


__all__ = [
    "CAPTION_SCHEMA",
    "CAPTION_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "CaptionProvider",
    "run_grounded_clip_caption_stage",
]
