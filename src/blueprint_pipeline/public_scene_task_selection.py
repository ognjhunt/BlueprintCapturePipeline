"""Fail-closed preregistration contract for a public-scene robot task.

The contract freezes *how* a scene/task is selected before any new candidate
target inspection or learned-policy outcome is consulted.  It deliberately
does not authorize a candidate from caller-supplied booleans; later admission
receipts must be derived from the bound dataset bytes, renderer outputs, SAGE
collision inspection, and native robot probes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "public_scene_task_selection_preregistration.v1"
SCOPE_AMENDMENT_SCHEMA_VERSION = "public_scene_task_selection_scope_amendment.v2"
REQUIRED_CRITERIA = {
    "admissible_rights_and_disclosure",
    "exact_interiorgs_appearance_identity",
    "exact_sage_collision_identity",
    "shared_metric_frame_or_typed_abstention",
    "complete_enough_observed_task_area",
    "single_joint_articulated_assembly_identity",
    "fixed_and_moving_link_separation",
    "observed_handle_or_contact_region",
    "source_articulation_collision_removal",
    "franka_reachability_envelope",
    "external_and_wrist_policy_observability",
    "review_overview_observability",
    "released_code_inpainting_admissibility",
    "usd_content_agents_joint_topology_admissibility",
    "independent_dynamic_articulation_qualification",
}
REQUIRED_FORBIDDEN_SIGNALS = {
    "learned_policy_outcomes",
    "inpainting_quality_outcomes",
    "manual_mask_or_scene_edit_convenience",
    "replacement_asset_reuse_from_prior_scene",
    "prior_scene_geometry_or_task_coordinates",
}


class PublicSceneTaskSelectionError(ValueError):
    """Stable, sorted validation errors for the selection freeze."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def validate_selection_preregistration(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize an outcome-blind scene/task selection freeze."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("selection_preregistration_schema_invalid")
    if payload.get("program_id") != "arm-decision-proof-v1":
        errors.append("selection_preregistration_program_invalid")
    if payload.get("adp_item") != "ADP-009D":
        errors.append("selection_preregistration_adp_item_invalid")
    if payload.get("day_gate") != "public_scene_day_28":
        errors.append("selection_preregistration_day_gate_invalid")
    if payload.get("frozen_before_new_candidate_inspection") is not True:
        errors.append("selection_preregistration_not_frozen_before_inspection")
    if payload.get("learned_policy_outcomes_accessed") is not False:
        errors.append("selection_preregistration_policy_outcome_leakage")
    if payload.get("new_inpainting_outcomes_accessed") is not False:
        errors.append("selection_preregistration_inpainting_outcome_leakage")
    if payload.get("selection_rule") != "first_fully_passing_candidate_in_frozen_order":
        errors.append("selection_preregistration_rule_invalid")
    if payload.get("scope_amendment") != (
        "user_explicitly_replaced_rigid_pick_place_with_articulated_open_close_"
        "on_2026_08_08"
    ):
        errors.append("selection_preregistration_scope_amendment_invalid")
    if payload.get("task_family") != "single_joint_articulated_open_or_close":
        errors.append("selection_preregistration_task_family_invalid")
    joint_agent = payload.get("usd_content_agents_joint_agent")
    if not isinstance(joint_agent, Mapping):
        errors.append("selection_preregistration_joint_agent_missing")
    else:
        if joint_agent.get("version") != "0.5.2" or joint_agent.get("tag") != "v0.5.2":
            errors.append("selection_preregistration_joint_agent_version_invalid")
        if joint_agent.get("commit") != (
            "36dbf3f274f8e256637230a05a085853f65cc175"
        ):
            errors.append("selection_preregistration_joint_agent_commit_invalid")
        if joint_agent.get("license") != "Apache-2.0":
            errors.append("selection_preregistration_joint_agent_license_invalid")
        if joint_agent.get("adapter") != "owned_core":
            errors.append("selection_preregistration_joint_agent_adapter_invalid")
        if joint_agent.get("supported_joint_types") != ["revolute", "prismatic"]:
            errors.append("selection_preregistration_joint_agent_types_invalid")
        if joint_agent.get("external_disclosure_status") != (
            "blocked_until_exact_scene_derived_bytes_and_render_upload_are_rights_"
            "authorized"
        ):
            errors.append("selection_preregistration_joint_agent_disclosure_invalid")

    criteria = _rows(payload.get("criteria"))
    criterion_ids = [str(row.get("criterion_id") or "") for row in criteria]
    if set(criterion_ids) != REQUIRED_CRITERIA or len(criterion_ids) != len(
        REQUIRED_CRITERIA
    ):
        errors.append("selection_preregistration_criteria_invalid")
    if any(row.get("required") is not True for row in criteria):
        errors.append("selection_preregistration_nonrequired_criterion")
    if any(not str(row.get("evidence") or "").strip() for row in criteria):
        errors.append("selection_preregistration_criterion_evidence_missing")

    forbidden = payload.get("forbidden_selection_signals")
    if not isinstance(forbidden, list) or set(forbidden) != REQUIRED_FORBIDDEN_SIGNALS:
        errors.append("selection_preregistration_forbidden_signals_invalid")

    previous = payload.get("previously_used_scene_ids")
    if (
        not isinstance(previous, list)
        or not previous
        or previous != sorted(set(map(str, previous)))
        or any(not str(scene_id).strip() for scene_id in previous)
    ):
        errors.append("selection_preregistration_prior_scene_identity_invalid")
    candidates = _rows(payload.get("candidate_order"))
    scene_ids = [str(row.get("publisher_scene_id") or "") for row in candidates]
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        errors.append("selection_preregistration_candidate_order_invalid")
    if isinstance(previous, list) and set(scene_ids).intersection(map(str, previous)):
        errors.append("selection_preregistration_prior_scene_reused")
    if scene_ids != sorted(scene_ids):
        errors.append("selection_preregistration_candidate_order_not_deterministic")
    if any(row.get("new_target_inspection_status") != "pending" for row in candidates):
        errors.append("selection_preregistration_candidate_preinspected")
    if any(row.get("method_outcomes_consulted") is not False for row in candidates):
        errors.append("selection_preregistration_candidate_outcome_leakage")
    if payload.get("selected_scene") is not None:
        errors.append("selection_preregistration_contains_selected_scene")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        errors.append("selection_preregistration_thresholds_missing")
    else:
        exact_thresholds = {
            "minimum_target_visible_views": 4,
            "minimum_collider_obb_iou": 0.85,
            "maximum_unrelated_collider_overlap_fraction": 0.0,
            "minimum_handle_grasp_width_m": 0.02,
            "maximum_handle_grasp_width_m": 0.085,
            "maximum_franka_task_radius_m": 0.75,
            "maximum_joint_count": 1,
            "required_articulation_root_count": 1,
            "minimum_revolute_travel_degrees": 30.0,
            "minimum_prismatic_travel_m": 0.1,
        }
        if dict(thresholds) != exact_thresholds:
            errors.append("selection_preregistration_thresholds_invalid")

    if payload.get("claim_ceiling") != "development_only_selection_rule":
        errors.append("selection_preregistration_claim_ceiling_invalid")
    expected = canonical_digest(payload, digest_field="preregistration_digest")
    if payload.get("preregistration_digest") != expected:
        errors.append("selection_preregistration_digest_invalid")
    if errors:
        raise PublicSceneTaskSelectionError(errors)
    return payload


def load_selection_preregistration(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PublicSceneTaskSelectionError(["selection_preregistration_not_mapping"])
    return validate_selection_preregistration(payload)


def validate_selection_scope_amendment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit multi-joint-assembly, one-task-joint amendment."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCOPE_AMENDMENT_SCHEMA_VERSION:
        errors.append("selection_scope_amendment_schema_invalid")
    if payload.get("program_id") != "arm-decision-proof-v1":
        errors.append("selection_scope_amendment_program_invalid")
    if payload.get("adp_item") != "ADP-009D":
        errors.append("selection_scope_amendment_adp_item_invalid")
    if payload.get("base_preregistration_digest") != (
        "sha256:32793a474c30b6be26a03d145dddb3b332d6beadffc525bafcca341bb7040ea0"
    ):
        errors.append("selection_scope_amendment_base_digest_invalid")
    if payload.get("authority") != (
        "user_explicitly_allowed_multi_joint_assembly_with_one_commanded_task_joint_"
        "on_2026_08_08"
    ):
        errors.append("selection_scope_amendment_authority_invalid")
    if payload.get("learned_policy_outcomes_accessed") is not False:
        errors.append("selection_scope_amendment_policy_outcome_leakage")
    if payload.get("new_inpainting_outcomes_accessed") is not False:
        errors.append("selection_scope_amendment_inpainting_outcome_leakage")
    if payload.get("candidate_order_changed") is not False:
        errors.append("selection_scope_amendment_candidate_order_changed")
    if payload.get("task_family") != (
        "one_commanded_joint_in_bounded_multi_joint_articulated_assembly"
    ):
        errors.append("selection_scope_amendment_task_family_invalid")
    exact_joint_scope = {
        "minimum_assembly_joint_count": 1,
        "maximum_assembly_joint_count": 4,
        "commanded_task_joint_count": 1,
        "required_articulation_root_count": 1,
        "non_task_joint_mode": "locked_at_frozen_reset_with_native_readback",
        "non_task_joint_motion_tolerance": 0.001,
    }
    if payload.get("joint_scope") != exact_joint_scope:
        errors.append("selection_scope_amendment_joint_scope_invalid")
    inspection = _rows(payload.get("inspection_disclosure"))
    if [str(row.get("publisher_scene_id") or "") for row in inspection] != [
        "840076",
        "840411",
    ]:
        errors.append("selection_scope_amendment_inspection_disclosure_invalid")
    if any(row.get("method_outcomes_consulted") is not False for row in inspection):
        errors.append("selection_scope_amendment_inspection_outcome_leakage")
    if payload.get("source_link_separation_rule") != (
        "target_link_must_be_separately_observed_and_bound_by_source_evidence_or_"
        "qualified_released_code_never_manual_selection"
    ):
        errors.append("selection_scope_amendment_link_separation_invalid")
    if payload.get("claim_ceiling") != "development_only_selection_rule":
        errors.append("selection_scope_amendment_claim_ceiling_invalid")
    expected = canonical_digest(payload, digest_field="amendment_digest")
    if payload.get("amendment_digest") != expected:
        errors.append("selection_scope_amendment_digest_invalid")
    if errors:
        raise PublicSceneTaskSelectionError(errors)
    return payload


def load_selection_scope_amendment(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PublicSceneTaskSelectionError(["selection_scope_amendment_not_mapping"])
    return validate_selection_scope_amendment(payload)


__all__ = [
    "PublicSceneTaskSelectionError",
    "REQUIRED_CRITERIA",
    "SCHEMA_VERSION",
    "SCOPE_AMENDMENT_SCHEMA_VERSION",
    "load_selection_scope_amendment",
    "load_selection_preregistration",
    "validate_selection_scope_amendment",
    "validate_selection_preregistration",
]
