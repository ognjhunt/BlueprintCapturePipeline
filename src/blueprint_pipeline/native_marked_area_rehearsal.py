"""Freeze an explicitly authorized tabletop-marker revision of a retained task."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .adp_task_scoring import seal_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import validate_native_task_arena_packet_request


def marked_area_request(*, source_request: Mapping[str, Any], authority: Mapping[str, Any], surface_z_m: float, support_prim_path: str) -> dict[str, Any]:
    """Keep the scene, book, robot placement, and strict manipulation thresholds."""
    validate_native_task_arena_packet_request(source_request)
    if (authority.get("schema_version") != "task_evaluation_task_change_authority.v1"
            or authority.get("authority_digest") != canonical_digest(authority, digest_field="authority_digest")
            or authority.get("destination_kind") != "marked_tabletop_area"
            or authority.get("tray_required") is not False
            or authority.get("retain_sealed_scene_and_book") is not True
            or not authority.get("authorized_by")
            or not math.isfinite(surface_z_m)
            or not isinstance(support_prim_path, str) or not support_prim_path.startswith("/")):
        raise ValueError("marked_area_task_change_authority_invalid")
    request = copy.deepcopy(dict(source_request))
    request["task_id"] = authority["new_task_id"]
    request["assets"] = [row for row in request["assets"] if row.get("semantic_role") != "task_support"]
    spec = request["task_spec"]
    retained_destination = {"destination_position_tolerance_m", "destination_position_bounds_world_m", "destination_orientation_xyzw", "destination_orientation_tolerance_rad"}
    for key in list(spec):
        if (key.startswith("destination_") and key not in retained_destination) or (key.startswith("configured_") and key != "configured_success_criteria"):
            spec.pop(key)
    spec.pop("initial_source_support", None)
    goal = list(spec["target_position_world_m"])
    goal[2] = float(spec["start_pose_world"][2])
    tolerance = float(spec["destination_position_tolerance_m"])
    spec["target_position_world_m"] = goal
    for criteria_name in ("configured_success_criteria", "success_criteria"):
        if not isinstance(spec.get(criteria_name), dict):
            continue
        criteria = spec[criteria_name]
        criteria.pop("whole_subject_containment_required", None)
        criteria.pop("object_must_rest_on_destination_support", None)
        criteria["target_center_xyz_m"] = list(goal)
        criteria["forbidden_contact_classes"] = [name for name in criteria.get("forbidden_contact_classes", []) if name != "destination_background"]
    spec["destination_position_bounds_world_m"] = {
        "minimum": [v - tolerance for v in goal], "maximum": [v + tolerance for v in goal]}
    spec["support_height_interval_m"] = [goal[2] - tolerance, goal[2] + tolerance]
    spec["visible_target_label"] = "green target marker"
    spec["prompt"] = "Pick up the open book, place it over the green target marker on the tabletop, release it, and move the gripper clear."
    spec["visible_target_marker"] = {
        "schema_version": "native_task_target_marker.v1", "shape": "flat_green_disc",
        "non_colliding": True, "radius_m": 0.06,
        "surface_position_world_m": [goal[0], goal[1], float(surface_z_m)]}
    spec["task_change_authority_digest"] = authority["authority_digest"]
    affordance = spec.get("interaction_affordance")
    if isinstance(affordance, dict):
        affordance["intended_support_prim_paths"] = [support_prim_path]
        affordance["affordance_digest"] = canonical_digest(affordance, digest_field="affordance_digest")
    previous_contract = spec.get("task_success_contract")
    if isinstance(previous_contract, Mapping):
        criteria = copy.deepcopy(previous_contract["criteria"])
        criteria["destination_containment"]["position_bounds_world_m"] = copy.deepcopy(spec["destination_position_bounds_world_m"])
        criteria["support"]["height_interval_m"] = list(spec["support_height_interval_m"])
        temporal = criteria.get("temporal_invariants", {})
        temporal["forbidden_contact_classes"] = [name for name in temporal.get("forbidden_contact_classes", []) if name != "destination_background"]
        confirmed = seal_rigid_task_success_contract(
            task_spec=spec, site_id=previous_contract["scope"]["site_id"], task_id=request["task_id"],
            author_source="task_owner", author_id=authority["authorized_by"], confirmation_status="confirmed",
            confirmed_by_team_id=previous_contract["provenance"].get("confirmed_by_team_id") or authority["authorized_by"], criteria=criteria)
        spec["task_success_contract"] = confirmed
        spec["task_success_contract_digest"] = confirmed["contract_digest"]
    request.pop("configured_task_template_adapter", None)
    feedback = request.pop("native_construction_feedback", {})
    request["retained_placement_candidate_id"] = feedback.get("selected_placement_candidate_id")
    request["task_change_authority_digest"] = authority["authority_digest"]
    scenario = request["scenario"]
    instance = scenario["context_document"]
    instance.pop("configured_task_source_documents_digest", None)
    instance["task_change_authority_digest"] = authority["authority_digest"]
    instance["cell_id"] = "marked_area_canonical.seed_" + str(scenario["seed"])
    instance["template_id"] = "marked_area_canonical"
    for axis, value in zip("xyz", goal, strict=True):
        instance["resolved_parameters"][f"target_{axis}_m"] = value
    instance["instance_digest"] = canonical_digest(instance, digest_field="instance_digest")
    scenario.update(cell_id=instance["cell_id"], instance_digest=instance["instance_digest"])
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    validate_native_task_arena_packet_request(request)
    return request
