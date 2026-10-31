"""Source-bound, read-only preparation findings for the existing website status.

ADP-009D/day-28 precursor: conservative geometry refusals are not an exhaustive
mechanical impossibility proof. No provider call, retry or new spending authority.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_PATH_REFUSAL = re.compile(r"robot_rigid_pick_place_(?:unreachable|scene_strike):[a-z_]+\Z")
_MISSING = frozenset({
    "website_registration_scale_conflicts_declared", "support_surface_not_found_under_subject",
    "task_destination_pose_required", "task_destination_relation_required",
    "task_destination_interior_geometry_required", "task_destination_surface_contact_required",
    "website_owner_success_criteria_translation_required", "website_assembly_whole_object_coverage_required",
    "website_synthetic_subject_start_unplaceable",
})


def _load(path: Path, root: Path, digest_field: str) -> dict[str, Any]:
    absolute = path.absolute()
    if (".." in absolute.parts or not absolute.is_relative_to(root)
            or any(item.is_symlink() for item in (absolute, *absolute.parents))
            or not absolute.is_file()):
        raise ValueError("preparation_outcome_evidence_unavailable")
    value = json.loads(absolute.read_text())
    if (not isinstance(value, dict) or not _DIGEST.fullmatch(str(value.get(digest_field, "")))
            or value[digest_field] != canonical_digest(value, digest_field=digest_field)):
        raise ValueError("preparation_outcome_evidence_invalid")
    return value


def read_preparation_outcome(*, capture_root: Path, task_context_digest: str,
                             state: str) -> dict[str, Any] | None:
    """Only a current stopped attempt may project a verified retained finding."""
    if state not in {"awaiting_inputs", "failed_retryable", "failed_terminal"}:
        return None
    root = capture_root.absolute()
    path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
    if not path.exists():
        return None
    handoff = _load(path, root, "digest")
    if (handoff.get("schema_version") != "website_scene_handoff.v1"
            or handoff.get("capture_id") != root.name or handoff.get("scene_id") != root.parent.parent.name):
        raise ValueError("preparation_outcome_capture_mismatch")
    from .website_target_background_correction import retains_rejected_selection
    from .observed_room_contracts import retains_observed_room_limit
    context = {"capture_id": root.name, "scene_id": root.parent.parent.name,
               "context_digest": task_context_digest}
    if retains_rejected_selection(handoff, context) or retains_observed_room_limit(handoff, context):
        return {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    ref = handoff.get("capability_verdict")
    if isinstance(ref, dict) and ref.get("status") == "abstained":
        from .robot_arm_path_clearance import ROBOT_PROFILE_ARMS

        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        verdict = _load(Path(ref["path"]), root, "verdict_digest")
        runtime_ref = handoff.get("runtime_inputs") or {}
        runtime = _load(Path(runtime_ref["path"]), root, "digest")
        if (preparation.get("schema_version") != "website_scene_preparation.v1"
                or preparation.get("binding", {}).get("task_context_digest") != task_context_digest
                or handoff.get("preparation_digest") != preparation["digest"]
                or runtime.get("schema_version") != "website_scene_runtime_inputs.v1"
                or runtime.get("digest") != runtime_ref.get("digest")
                or runtime.get("preparation_digest") != preparation["digest"]
                or runtime.get("collision", {}).get("digest") != verdict.get("collision_sha256")
                or verdict.get("schema_version") != "website_rigid_task_capability_verdict.v1"
                or verdict.get("preparation_digest") != preparation["digest"]
                or verdict.get("verdict_digest") != ref.get("digest")
                or verdict.get("status") != "abstained"
                or verdict.get("robot_id") not in set(ROBOT_PROFILE_ARMS.values())
                or verdict.get("claim_ceiling") != "development_only"
                or verdict.get("physical_world_truth_claimed") is not False):
            raise ValueError("preparation_outcome_verdict_binding_invalid")
        blockers = verdict.get("blockers")
        if (isinstance(blockers, list) and blockers
                and all(isinstance(code, str) and _PATH_REFUSAL.fullmatch(code) for code in blockers)):
            return {"category": "robot_incompatibility", "reason": "robot_carry_path_rejected",
                    "basis": "conservative_configuration_check", "robot_id": verdict["robot_id"],
                    "evidence_digest": verdict["verdict_digest"]}
        # Failure to find a station is missing evidence, not a reach verdict.
        if isinstance(blockers, list) and blockers and all(isinstance(code, str) and code.startswith("robot_station_")
                                                         for code in blockers):
            return {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                    "basis": "missing_evidence", "robot_id": None, "evidence_digest": verdict["verdict_digest"]}
        return None
    blockers = handoff.get("blockers")
    if isinstance(blockers, list) and blockers and all(code in _MISSING for code in blockers):
        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        if (preparation.get("schema_version") != "website_scene_preparation.v1"
                or preparation.get("binding", {}).get("task_context_digest") != task_context_digest
                or handoff.get("preparation_digest") != preparation["digest"]):
            raise ValueError("preparation_outcome_context_mismatch")
        return {"category": "insufficient_evidence", "reason": "task_evidence_incomplete",
                "basis": "missing_evidence", "robot_id": None, "evidence_digest": handoff["digest"]}
    return None
