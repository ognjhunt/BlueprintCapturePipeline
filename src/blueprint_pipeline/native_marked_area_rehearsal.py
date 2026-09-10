"""Freeze an explicitly authorized tabletop-marker revision of a retained task."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .adp_task_scoring import seal_rigid_task_success_contract, validate_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
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


def marked_area_control_search_request(*, source_request: Mapping[str, Any], phase_plan: Mapping[str, Any], support_prim_path: str) -> dict[str, Any]:
    """Declare 16 reset/yaw seeds for the existing 64-branch native search."""
    from .native_franka_pose_servo import PINK_GLOBAL_REFERENCE_SEEDS
    from .task_evaluation_native_construction_feedback_controller import (
        CANDIDATE_SCHEMA_VERSION, build_next_native_construction_inventory,
        validate_native_construction_candidate,
    )
    validate_native_task_arena_packet_request(source_request)
    if not support_prim_path.startswith("/") or not phase_plan.get("phases"):
        raise ValueError("marked_area_control_search_inputs_invalid")
    request = copy.deepcopy(dict(source_request))
    first = phase_plan["phases"][0]
    def seal(value, field):
        value[field] = canonical_digest(value, digest_field=field)
        return value
    entry = seal({"schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
        "joins_authored_phase_id": first["phase_id"], "waypoints": [{
            "waypoint_id": "authored-entry", "position_world_m": first["position_world_m"],
            "orientation_world_xyzw": first["orientation_world_xyzw"]}]}, "entry_trajectory_variant_digest")
    camera = seal({"schema_version": "task_evaluation_native_camera_variant.v1",
                   "cameras": request["cameras"]}, "camera_variant_digest")
    names = [f"panda_joint{i}" for i in range(1, 8)]
    reset = request["robot_joint_reset_positions_rad"]
    seeds = [tuple(reset[name] for name in names), *PINK_GLOBAL_REFERENCE_SEEDS]
    candidates = []
    for seed_index, seed in enumerate(seeds):
        for yaw_index, angle in enumerate((0.0, -math.pi/12, math.pi/12, math.pi/6)):
            pose = copy.deepcopy(request["robot_base_pose_world"])
            pose["position_world_m"] = [float(value) for value in pose["position_world_m"]]
            x, y, z, w = pose["orientation_xyzw"]
            s, c = math.sin(angle/2), math.cos(angle/2)
            pose["orientation_xyzw"] = [c*x-s*y, c*y+s*x, c*z+s*w, c*w-s*z]
            reset_variant = seal({"schema_version": "task_evaluation_native_robot_reset_variant.v1",
                "robot_joint_reset_positions_rad": {**reset, **dict(zip(names, seed, strict=True))}}, "reset_variant_digest")
            candidate = seal({"schema_version": CANDIDATE_SCHEMA_VERSION,
                "candidate_id": f"marked-seed-{seed_index}-yaw-{yaw_index}",
                "deterministic_rank": len(candidates), "robot_base_pose_world": pose,
                "support_surface_id": support_prim_path, "reset_variant": reset_variant,
                "entry_trajectory_variant": entry, "camera_variant": camera,
                "maximum_incremental_cost_usd": 0.08, "maximum_runtime_seconds": 300.0,
                "addressed_feedback_codes": []}, "candidate_digest")
            candidates.append(validate_native_construction_candidate(candidate))
    inventory = build_next_native_construction_inventory(
        run_id=request["task_id"]+"-control-search", round_index=0,
        source_native_feedback=None, prior_history=(), candidate_universe=candidates,
        maximum_candidates=64,
    )
    search = seal({"schema_version": "task_evaluation_control_search_authority.v1",
        "enabled": True, "claim_ceiling": "development_only_control_search",
        "provider_allocations_performed": 0, "requested_vector_env_count": 256,
        "maximum_vector_env_count": 1024, "seeds_per_candidate": 1, "shortlist_size": 16,
        "appearance_mode": "omitted", "camera_mode": "disabled",
        "full_fidelity_replay_required": True}, "authority_digest")
    request["native_construction_feedback"] = {
        "candidate_universe": inventory,
        "candidate_generator_authority": {"generator": "remote_curobo_v2_motion_generation",
            "package_version": "0.8.0", "source_revision": "4ea77366ca48ee453e7df139e39fa6532af49f3b",
            "required_on_retained_gpu": True, "deterministic_cpu_prefilter_required": True,
            "silent_fallback_permitted": False},
        "allocator_retry_cap": 0, "maximum_rounds": 8, "native_gates_unchanged": True,
        "control_search": search,
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return validate_native_task_arena_packet_request(request)


def direct_policy_request(*, source_request: Mapping[str, Any], authorized_by: str, authorization_reference: str) -> dict[str, Any]:
    """Record an explicit controls omission for diagnostic policy execution only."""
    from .native_task_arena_policy_canary_session import validate_control_omission_authority
    validate_native_task_arena_packet_request(source_request)
    request = copy.deepcopy(dict(source_request))
    spec = request["task_spec"]
    previous = validate_rigid_task_success_contract(spec["task_success_contract"])
    contract = copy.deepcopy(previous)
    contract["criteria"].pop("controls", None)
    contract["contract_digest"] = cross_runtime_canonical_digest(contract, digest_field="contract_digest")
    for name in ("configured_success_criteria", "success_criteria"):
        if isinstance(spec.get(name), dict):
            spec[name]["per_cell_controls_required"] = False
    validate_rigid_task_success_contract(contract)
    spec["task_success_contract"] = contract
    spec["task_success_contract_digest"] = contract["contract_digest"]
    if spec.get("visible_target_marker") is not None:
        # A marked area has a fixed scoring frame even though it has no tray body.
        spec["destination_pose_world"] = [
            *spec["target_position_world_m"], *spec["destination_orientation_xyzw"],
        ]
    request.pop("native_construction_feedback", None)
    authority = {"schema_version": "task_evaluation_diagnostic_control_omission_authority.v1",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "authorized_by": authorized_by, "authorization_reference": authorization_reference,
        "omitted_controls": ["zero_action_negative", "deterministic_scripted_positive"],
        "source_task_success_contract_digest": previous["contract_digest"],
        "result_task_success_contract_digest": contract["contract_digest"],
        "task_scoring_criteria_changed": False, "qualified_comparison_permitted": False}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    validate_control_omission_authority(authority, contract_digest=contract["contract_digest"])
    request["diagnostic_control_omission_authority"] = authority
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return validate_native_task_arena_packet_request(request)
