"""Prepare task-aware website images before a single reconstruction.

Website inputs retain original frames and estimated MapAnything cameras, bind
SAM3.1 task masks, and edit selected objects out of original-resolution frames.
Legacy capture keeps its existing opt-in behavior. All inferred geometry and
repairs remain development evidence (ADP-009B, public_scene_day_14).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .clean_plate_removal_analysis_gemini import (
    REMOVAL_PLAN_SCHEMA_VERSION,
    analyze_removal_targets,
    empty_removal_plan,
    validate_removal_plan,
)
from .common import (
    ensure_dir,
    optional_read_json,
    parse_bool,
    utc_now_iso,
    write_json,
)
from .local_capture import resolve_local_capture_context
from .decision_evidence_contracts import canonical_digest
from .website_scene_geometry import run_website_scene_geometry
from .website_task_masks import run_website_task_masks
from .website_object_removal import prepare_object_removal_frames, select_reconstruction_frames, reconstruction_source_frames
from .website_reconstruction_profile import reconstruction_profile
from .website_image_completion import complete_background_images, verify_completed_background

FLAG_ENV = "BLUEPRINT_CLEAN_PLATE_ENABLED"
ADP_ITEM_ENV = "BLUEPRINT_CLEAN_PLATE_ADP_ITEM"
DAY_GATE_ENV = "BLUEPRINT_CLEAN_PLATE_DAY_GATE"

PROGRAM_ID = "arm-decision-proof-v1"
DEFAULT_ADP_ITEM = "ADP-009B"
ALLOWED_ADP_ITEMS = frozenset({"ADP-009B", "ADP-021"})
DEFAULT_DAY_GATE = "public_scene_day_14"
CLAIM_CEILING = "development_only"

STAGE_MANIFEST_SCHEMA_VERSION = "clean_plate_stage_manifest.v1"
REMOVAL_MANIFEST_SCHEMA_VERSION = "clean_plate_removal_manifest.v1"

CLEAN_PLATE_DIRNAME = "clean_plate"
REMOVAL_PLAN_FILENAME = "removal_plan.json"
REMOVAL_MANIFEST_FILENAME = "removal_manifest.json"
STAGE_MANIFEST_FILENAME = "clean_plate_stage_manifest.json"
CURRENT_POINTER_FILENAME = "current.json"

# A privacy status that means people are accounted for. ``failed_closed`` is a
# hard stop; ``not_run`` is tolerated (local/non-delivery) but recorded as
# unverified, and a clean plate is never emitted while unverified.
_PRIVACY_SAFE_STATUSES = frozenset(
    {
        "no_people_detected",
        "person_removed",
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    }
)

# Candidate local locations of the privacy-safe walkthrough that Atlas consumes,
# in preference order. All are pipeline-internal; raw/ is never read for output.
_WORLDLABS_INPUT_RELPATH = ("worldlabs_input", "worldlabs_input.mp4")
_PRIVACY_FINAL_RELPATH = ("final_walkthrough.mov",)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_env(name: str, default: str) -> str:
    text = _string(os.getenv(name))
    return text or default


@dataclass(frozen=True)
class CleanPlatePolicy:
    """Env-driven configuration for the clean-plate stage (serialized into records)."""

    enabled: bool = False
    adp_item: str = DEFAULT_ADP_ITEM
    day_gate: str = DEFAULT_DAY_GATE
    # Forward-safety for the deferred fill machinery: a clean plate may only be
    # emitted (and the reconstruction input redirected) when privacy verified.
    require_privacy_verified: bool = True

    @classmethod
    def from_env(cls) -> "CleanPlatePolicy":
        enabled_raw = os.getenv(FLAG_ENV)
        return cls(
            enabled=parse_bool(enabled_raw, default=False) if enabled_raw is not None else False,
            adp_item=_string_env(ADP_ITEM_ENV, DEFAULT_ADP_ITEM),
            day_gate=_string_env(DAY_GATE_ENV, DEFAULT_DAY_GATE),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "adp_item": self.adp_item,
            "day_gate": self.day_gate,
            "require_privacy_verified": self.require_privacy_verified,
        }


def _boundary_flags() -> Dict[str, bool]:
    """Claim-ceiling boundary booleans; all held False (development_only)."""

    return {
        "reconstruction_qualified": False,
        "metric_authority": False,
        "collision_authority": False,
        "hidden_surface_authority": False,
        "privacy_authority": False,
        "physical_evidence": False,
    }


def _privacy_status(privacy: Mapping[str, Any]) -> str:
    return _string(privacy.get("status")).lower() or "not_run"


def _privacy_is_verified(privacy: Mapping[str, Any]) -> bool:
    return _privacy_status(privacy) in _PRIVACY_SAFE_STATUSES


def _resolve_input_video(pipeline_root: Path, capture_root: Path) -> Optional[Path]:
    worldlabs = pipeline_root.joinpath(*_WORLDLABS_INPUT_RELPATH)
    if worldlabs.is_file():
        return worldlabs
    privacy_final = capture_root.joinpath("privacy", *_PRIVACY_FINAL_RELPATH)
    if privacy_final.is_file():
        return privacy_final
    return None


def apply_clean_plate_to_reconstruction_input(
    worldlabs_input: Mapping[str, Any], clean_plate: Mapping[str, Any], *, required: bool,
) -> Dict[str, Any]:
    """Never let a task-analysis hold fall back to the unedited source video."""
    result = dict(worldlabs_input)
    status = clean_plate.get("status")
    if required and result.get("status") == "blocked":
        return {**result, "output_video_uri": None, "prepared_views": None}
    # Preserve the disabled legacy path; website preparation is mandatory.
    if status == "disabled" and not required:
        return result
    passed = (
        status in {"noop", "objects_removed"}
        and clean_plate.get("privacy_verified") is True
        and not clean_plate.get("blockers")
    )
    views = clean_plate.get("prepared_views")
    image_input_ready = isinstance(views, Mapping) and views.get("status") == "ready"
    if status == "objects_removed" and not clean_plate.get("clean_plate_video_uri") and not image_input_ready:
        passed = False
    if not passed:
        return {**result, "status": "blocked", "output_video_uri": None,
                "reason": clean_plate.get("reason") or "task_scene_preparation_required",
                "clean_plate_blockers": list(clean_plate.get("blockers") or [])}
    if image_input_ready:
        result.update(status="ready", output_video_uri=None, prepared_views=dict(views), clean_plate_applied=True)
    elif status == "objects_removed":
        result.update(output_video_uri=clean_plate["clean_plate_video_uri"], clean_plate_applied=True)
    return result


def _build_removal_manifest(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-target manifest linking a removed source object to its (future) asset.

    The compose-back slots (``replacement_asset_id``, ``pose_world``,
    ``replacement_asset_frame_registration_uri``) are null in the scaffold; the
    Astra rebuild + frame registration fill them downstream.
    """

    entries = []
    for target in plan.get("targets", []) or []:
        if not isinstance(target, Mapping):
            continue
        if target.get("disposition") != "remove":
            continue
        entries.append(
            {
                "target_id": target.get("target_id"),
                "semantic_label": target.get("semantic_label"),
                "target_class": target.get("target_class"),
                "target_role": target.get("target_role"),
                "task_effect": target.get("task_effect"),
                "decision_reason": target.get("decision_reason"),
                "task_basis_quote": target.get("task_basis_quote"),
                "rebuild_intent": target.get("rebuild_intent"),
                "spatial_evidence": target.get("spatial_evidence", []),
                # Filled by the deferred fill machinery / rebuild chain:
                "mask_track_ref": None,
                "observed_bounds_world_m": None,
                "fill_provenance": None,  # per-region observed_recovered | generated_candidate
                "compose_back": {
                    "replacement_asset_id": None,
                    "pose_world": None,
                    "replacement_asset_frame_registration_uri": None,
                },
            }
        )
    return {
        "schema_version": REMOVAL_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "claim_ceiling": CLAIM_CEILING,
        "removal_plan_schema_version": plan.get("schema_version"),
        "removal_plan_status": plan.get("status"),
        "task_context_sha256": plan.get("task_context_sha256"),
        "removed_target_count": len(entries),
        "entries": entries,
    }


def validate_clean_plate_stage_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Deterministic fail-closed validator; empty list means valid.

    Rejects any claim elevation or program/backlog mismatch, mirroring
    ``public_scene_task_selection.py``.
    """

    errors: list[str] = []
    if manifest.get("schema_version") != STAGE_MANIFEST_SCHEMA_VERSION:
        errors.append("clean_plate_stage_manifest_schema_version_invalid")
    if manifest.get("program_id") != PROGRAM_ID:
        errors.append("clean_plate_stage_program_id_invalid")
    if manifest.get("adp_item") not in ALLOWED_ADP_ITEMS:
        errors.append("clean_plate_stage_adp_item_invalid")
    if not _string(manifest.get("day_gate")):
        errors.append("clean_plate_stage_day_gate_missing")
    if manifest.get("claim_ceiling") != CLAIM_CEILING:
        errors.append("clean_plate_stage_claim_ceiling_invalid")
    boundary = manifest.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        errors.append("clean_plate_stage_claim_boundary_missing")
    else:
        for key, value in _boundary_flags().items():
            if boundary.get(key) is not value:
                errors.append(f"clean_plate_stage_boundary_{key}_elevated")
    # This stage emits images; legacy video output is not implemented here.
    if manifest.get("clean_plate_video_uri") not in (None, ""):
        errors.append("clean_plate_stage_video_uri_unexpected_in_scaffold")
    if manifest.get("generated_regions_present") is not False:
        views = manifest.get("prepared_views") or {}
        if views.get("generated_pixels_present") is not True or (views.get("completion_review") or {}).get("status") != "passed":
            errors.append("clean_plate_stage_generated_regions_require_review")
    return errors


def _stage_uri(ctx: Any, filename: str) -> str:
    return (
        f"gs://{ctx.bucket}/scenes/{ctx.scene_id}/captures/{ctx.capture_id}"
        f"/pipeline/{CLEAN_PLATE_DIRNAME}/{filename}"
    )


def run_clean_plate_stage(
    *,
    capture_root: str | Path,
    privacy_processing: Optional[Mapping[str, Any]] = None,
    worldlabs_input: Optional[Mapping[str, Any]] = None,
    task_context: Optional[Mapping[str, Any]] = None,
    website_source_video: Optional[Path] = None,
    image_edit_admission: Optional[Mapping[str, Any]] = None,
    image_edit_admission_grant: Any = None,
    meta_sam_admission: Optional[Mapping[str, Any]] = None,
    meta_sam_admission_grant: Any = None,
    reconstruction_capabilities: Optional[Mapping[str, Any]] = None,
    policy: Optional[CleanPlatePolicy] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Run (or no-op) the clean-plate stage for one capture.

    Website reconstruction consumes only ready prepared images. The original
    source and estimated placement remain available for asset composition.
    """

    policy = policy or (CleanPlatePolicy(enabled=True) if website_source_video is not None else CleanPlatePolicy.from_env())

    if not policy.enabled:
        # Pure opt-in no-op; write nothing, mirror the supervisor-disabled shape.
        return {
            "status": "disabled",
            "mode": "disabled",
            "clean_plate_video_uri": None,
            "program_id": PROGRAM_ID,
            "adp_item": policy.adp_item,
            "day_gate": policy.day_gate,
            "claim_ceiling": CLAIM_CEILING,
            "policy": policy.to_dict(),
            "blockers": [],
        }

    ctx = resolve_local_capture_context(capture_root)
    clean_plate_root = ctx.pipeline_root / CLEAN_PLATE_DIRNAME
    ensure_dir(clean_plate_root)

    if privacy_processing is None:
        privacy_processing = (
            optional_read_json(ctx.pipeline_root / "privacy_processing_manifest.json") or {}
        )
    privacy_status = _privacy_status(privacy_processing)
    privacy_verified = _privacy_is_verified(privacy_processing)

    plan: Dict[str, Any]
    blockers: list[str] = []
    reason: Optional[str] = None
    source_geometry: Optional[Dict[str, Any]] = None
    task_masks: Optional[Dict[str, Any]] = None
    object_removal_frames: Optional[list[Dict[str, Any]]] = None
    prepared_views: Optional[Dict[str, Any]] = None

    if privacy_status == "failed_closed":
        # Fail safe: never proceed on a capture whose privacy pipeline failed.
        status = "failed_closed"
        mode = "privacy_failed_closed"
        reason = "privacy_pipeline_failed_closed"
        plan = empty_removal_plan(
            status="blocked",
            model="",
            processing="",
            blockers=["privacy_pipeline_failed_closed"],
        )
        blockers.append("privacy_pipeline_failed_closed")
        input_video_path = None
    else:
        input_video_path = website_source_video or _resolve_input_video(ctx.pipeline_root, ctx.capture_root)
        plan = analyze_removal_targets(video_path=website_source_video or input_video_path, task_context=task_context)
        plan_errors = validate_removal_plan(plan)
        if plan_errors:
            blockers.extend(plan_errors)
        blockers.extend(plan.get("blockers", []) or [])
        if website_source_video is not None:
            # The website reviews its original source, not a VIP derivative.
            # A complete review with no people can proceed; detected people
            # require actual removal/verification before images leave Pipeline.
            privacy_verified = not blockers and plan.get("status") == "completed" and not int(plan.get("person_target_count") or 0)
            privacy_status = "no_people_detected" if privacy_verified else "pending_website_review"
        movable_removals = int(plan.get("movable_removal_count") or 0)
        if blockers or _string(plan.get("status")) != "completed":
            status = "blocked"
            mode = "analysis_blocked"
            reason = "removal_analysis_blocked"
        elif movable_removals == 0:
            # Cleared space / nothing movable to remove -> near no-op.
            status = "noop"
            mode = "noop"
        else:
            # Analysis found movable objects, but the view-consistent fill
            # machinery (§5) is deferred: emit the plan/manifest, no video.
            status = "blocked"
            mode = "fill_machinery_pending"
            reason = "clean_plate_fill_machinery_not_implemented"

    # The website uses the original video for placement, never generated pixels.
    # Estimated scale is sufficient for this development replica; a missing
    # measured dimension is not a reason to stop preparation.
    if website_source_video is not None and privacy_verified and not blockers:
        try:
            source_geometry = run_website_scene_geometry(
                source_video=website_source_video,
                output_root=clean_plate_root / "source_geometry", capture_id=ctx.capture_id,
            )
        except Exception as exc:
            status, mode = "blocked", "source_geometry_blocked"
            reason = "website_source_geometry_unavailable"
            blockers.append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)

    if source_geometry is not None and not blockers and any(
        target.get("task_effect") in {"manipulated", "static_contact", "static_obstacle"}
        for target in plan.get("targets", [])
    ):
        try:
            task_masks = run_website_task_masks(plan=plan, source_geometry=source_geometry,
                                                output_root=clean_plate_root / "task_masks",
                                                meta_admission=meta_sam_admission,
                                                meta_admission_grant=meta_sam_admission_grant,
                                                task_context=task_context,
                                                source_video=website_source_video)
        except Exception as exc:
            status, mode = "blocked", "task_masks_blocked"
            reason = "website_task_masks_unavailable"
            blockers.append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)

    if task_masks is not None and source_geometry is not None and not blockers:
        try:
            profile = reconstruction_profile(reconstruction_capabilities)
            reconstruction_frames = reconstruction_source_frames(source_geometry=source_geometry, task_masks=task_masks,
                source_video=website_source_video, limit=profile["max_input_images"],
                output_root=clean_plate_root / "reconstruction_source_frames")
            object_removal_frames = prepare_object_removal_frames(
                frames=reconstruction_frames, task_masks=task_masks,
                output_root=clean_plate_root / "object_removal_frames",
            )
        except Exception as exc:
            status, mode = "blocked", "object_removal_frames_blocked"
            reason = "website_object_removal_frames_unavailable"
            blockers.append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)

    if object_removal_frames is not None and not blockers:
        try:
            selected = select_reconstruction_frames(frames=object_removal_frames, task_masks=task_masks,
                                                    limit=profile["max_input_images"])
            completion_review = None
            if any(frame["remaining_pixel_count"] for frame in selected):
                selected = complete_background_images(
                    frames=selected, task_digest=plan["task_context_sha256"],
                    output_root=clean_plate_root / "image_completion", admission=image_edit_admission or {},
                    token=os.getenv("OPENAI_API_KEY", ""), admission_grant=image_edit_admission_grant,
                    targets=plan["targets"], task_context=task_context)
                completion_review = verify_completed_background(
                    frames=selected, original_frames=source_geometry["frames"], plan=plan,
                    output_root=clean_plate_root / "image_completion")
                if completion_review.get("status") != "passed":
                    raise ValueError("website_image_completion_review_failed")
            prepared_views = {"schema_version": "website_prepared_views.v1", "status": "ready", "frames": selected,
                              "reconstruction_profile": profile,
                              "task_context_sha256": plan.get("task_context_sha256"),
                              "source_geometry_digest": (source_geometry or {}).get("digest"),
                              "generated_pixels_present": any(f.get("generated_pixels_present") for f in selected),
                              "completion_review": completion_review, "claim_ceiling": CLAIM_CEILING}
            prepared_views["digest"] = canonical_digest(prepared_views, digest_field="digest")
            status = "objects_removed" if int(plan.get("movable_removal_count") or 0) else "noop"
            mode, reason = "prepared_images", None
        except Exception as exc:
            status, mode, reason = "blocked", "website_images_blocked", "website_prepared_images_unavailable"
            blockers.append(str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__)

    removal_manifest = _build_removal_manifest(plan)

    stage_manifest: Dict[str, Any] = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "program_id": PROGRAM_ID,
        "adp_item": policy.adp_item,
        "day_gate": policy.day_gate,
        "claim_ceiling": CLAIM_CEILING,
        "status": status,
        "mode": mode,
        "reason": reason,
        "scene_id": ctx.scene_id,
        "capture_id": ctx.capture_id,
        "policy": policy.to_dict(),
        "privacy": {
            "status": privacy_status,
            "verified": privacy_verified,
            "world_model_video_uri": privacy_processing.get("world_model_video_uri"),
        },
        "input_video_path": str(input_video_path) if input_video_path else None,
        "input_video_sha256": plan.get("input_video_sha256"),
        "removal_plan_status": plan.get("status"),
        "removal_plan_schema_version": plan.get("schema_version", REMOVAL_PLAN_SCHEMA_VERSION),
        "target_count": int(plan.get("target_count") or 0),
        "movable_removal_count": int(plan.get("movable_removal_count") or 0),
        "person_target_count": int(plan.get("person_target_count") or 0),
        # Generated image repairs are separate from the retained originals.
        "clean_plate_video_uri": None,
        "generated_regions_present": bool((prepared_views or {}).get("generated_pixels_present")),
        "originals_retained": True,
        "source_geometry": source_geometry,
        "task_masks": task_masks,
        "object_removal_frames": object_removal_frames,
        "prepared_views": prepared_views,
        "removal_plan_uri": _stage_uri(ctx, REMOVAL_PLAN_FILENAME),
        "removal_manifest_uri": _stage_uri(ctx, REMOVAL_MANIFEST_FILENAME),
        "stage_manifest_uri": _stage_uri(ctx, STAGE_MANIFEST_FILENAME),
        "blockers": sorted({_string(item) for item in blockers if _string(item)}),
        "claim_boundary": _boundary_flags(),
    }

    write_json(clean_plate_root / REMOVAL_PLAN_FILENAME, plan)
    write_json(clean_plate_root / REMOVAL_MANIFEST_FILENAME, removal_manifest)
    write_json(clean_plate_root / STAGE_MANIFEST_FILENAME, stage_manifest)
    write_json(
        clean_plate_root / CURRENT_POINTER_FILENAME,
        {
            "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
            "generated_at": stage_manifest["generated_at"],
            "status": status,
            "stage_manifest_uri": stage_manifest["stage_manifest_uri"],
        },
    )

    return {
        "status": status,
        "mode": mode,
        "reason": reason,
        "clean_plate_video_uri": None,
        "program_id": PROGRAM_ID,
        "adp_item": policy.adp_item,
        "day_gate": policy.day_gate,
        "claim_ceiling": CLAIM_CEILING,
        "policy": policy.to_dict(),
        "privacy_status": privacy_status,
        "privacy_verified": privacy_verified,
        "source_geometry": source_geometry,
        "task_masks": task_masks,
        "object_removal_frames": object_removal_frames,
        "prepared_views": prepared_views,
        "target_count": stage_manifest["target_count"],
        "movable_removal_count": stage_manifest["movable_removal_count"],
        "person_target_count": stage_manifest["person_target_count"],
        "removal_plan_path": str(clean_plate_root / REMOVAL_PLAN_FILENAME),
        "removal_manifest_path": str(clean_plate_root / REMOVAL_MANIFEST_FILENAME),
        "stage_manifest_path": str(clean_plate_root / STAGE_MANIFEST_FILENAME),
        "removal_plan_uri": stage_manifest["removal_plan_uri"],
        "removal_manifest_uri": stage_manifest["removal_manifest_uri"],
        "stage_manifest_uri": stage_manifest["stage_manifest_uri"],
        "blockers": stage_manifest["blockers"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the conditional clean-plate frame-editing stage for one capture "
            "(scaffold: analysis + manifests; no pixel editing yet)."
        )
    )
    parser.add_argument(
        "--capture-root",
        required=True,
        help="Path under <storage-root>/<bucket>/scenes/<scene>/captures/<capture>.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute even if a current pointer already exists.",
    )
    args = parser.parse_args(argv)

    result = run_clean_plate_stage(
        capture_root=args.capture_root,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    # Terminal contract: a completed no-op or a healthy disabled run is success;
    # a blocked/failed_closed run is nonzero so callers notice.
    return 0 if result.get("status") in {"disabled", "noop", "objects_removed"} else 1


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
