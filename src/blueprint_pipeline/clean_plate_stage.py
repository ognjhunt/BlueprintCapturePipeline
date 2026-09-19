"""Conditional clean-plate frame-editing stage (extract -> Atlas).

This stage sits between the WorldLabs/Atlas input preparation and the
reconstruction submit. When a walkthrough contains movable task-objects, they
should be removed from the immovable "stage plate" before reconstruction (and
rebuilt separately as physics-ready assets from the ORIGINAL frames), because
multi-view reconstruction fuses a static world and bakes movers into ghosts.

**Scope of this module (scaffold).** It lands the contract and wiring only:

- an agentic Gemini removal-target analysis -> ``clean_plate_removal_plan.v1``
  (``clean_plate_removal_analysis_gemini.py``, fail-closed behind its own gate);
- a ``clean_plate_removal_manifest.v1`` linking each removed source object to a
  (later) rebuilt asset + pose for compose-back;
- a ``clean_plate_stage_manifest.v1`` receipt;
- a ``development_only`` fail-closed validator.

The **view-consistent fill machinery** (observed-background recovery + a
video-consistent generative fallback) is deliberately deferred. Until it lands
and a with/without reconstruction comparison on one real capture shows a win,
this stage **never** produces a clean-plate video and **never** redirects the
reconstruction input -- it is a strict no-op on the Atlas path.

**People are not removed here.** They are removed upstream, view-consistently,
by ``run_privacy_postprocess`` (SAM3 -> VIP video-inpaint -> verify). This stage
*verifies* that (fail-safe) and spends its machinery only on movable objects.

Governance: ``program_id="arm-decision-proof-v1"``, ``adp_item="ADP-009B"``
(own-capture, pre-reconstruction analog of the public-splat edit path;
``ADP-021`` is the field consumer), ``claim_ceiling="development_only"``. Behind
``BLUEPRINT_CLEAN_PLATE_ENABLED`` (default off). Design:
``docs/clean_plate_frame_editing_stage_design_2026-09-17.md``.
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
    # Preserve the disabled legacy path; website preparation is mandatory.
    if status == "disabled" and not required:
        return result
    passed = (
        status in {"noop", "objects_removed"}
        and clean_plate.get("privacy_verified") is True
        and not clean_plate.get("blockers")
    )
    if status == "objects_removed" and not clean_plate.get("clean_plate_video_uri"):
        passed = False
    if not passed:
        return {**result, "status": "blocked", "output_video_uri": None,
                "reason": clean_plate.get("reason") or "task_scene_preparation_required",
                "clean_plate_blockers": list(clean_plate.get("blockers") or [])}
    if status == "objects_removed":
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
    # The scaffold must never emit a clean-plate video or claim removed pixels.
    if manifest.get("clean_plate_video_uri") not in (None, ""):
        errors.append("clean_plate_stage_video_uri_unexpected_in_scaffold")
    if manifest.get("generated_regions_present") is not False:
        errors.append("clean_plate_stage_generated_regions_unexpected_in_scaffold")
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
    policy: Optional[CleanPlatePolicy] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Run (or no-op) the clean-plate stage for one capture.

    Returns a receipt dict. In the scaffold ``clean_plate_video_uri`` is always
    ``None`` -- the caller must not redirect the reconstruction input on any
    status other than a future ``objects_removed`` carrying a real video.
    """

    policy = policy or CleanPlatePolicy.from_env()

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
        input_video_path = _resolve_input_video(ctx.pipeline_root, ctx.capture_root)
        plan = analyze_removal_targets(video_path=input_video_path, task_context=task_context)
        plan_errors = validate_removal_plan(plan)
        if plan_errors:
            blockers.extend(plan_errors)
        blockers.extend(plan.get("blockers", []) or [])
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
        # Scaffold never edits pixels: no clean plate, no generated regions.
        "clean_plate_video_uri": None,
        "generated_regions_present": False,
        "originals_retained": True,
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
