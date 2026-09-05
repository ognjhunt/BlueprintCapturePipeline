"""Record builders for a scene-plus-task-objects production submission.

Every record here is derived from admitted scene bytes, the production source
preparation, the destination's SimReady qualification, the deploy identity, and
the owner's task request.  Nothing is hand-authored per scene: object names,
bounds, prims, digests, and identities all come from those inputs, and the
records are shaped exactly as the deployed validators require.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_EXTERNAL_SERVICE_SPEND_USD,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD,
    MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
    MIN_CONTENT_AGENTS_SPEND_USD,
    REQUIRED_PARENT_TTL_SECONDS,
)
from .task_evaluation_scene_construction_recipe import (
    CAPABILITY_ORDER,
    SCHEMA_VERSION as RECIPE_SCHEMA_VERSION,
)


PENDING_STATUS = "candidate_input_pending_exact_deployed_sha_and_publication"
STATIC_CHECKS = {
    "usd_parses": True,
    "meters_per_unit": 1.0,
    "up_axis": "Z",
    "single_movable_rigid_root": True,
    "collision_geometry_present": True,
    "collision_geometry_nonempty_and_finite": True,
    "mass_and_inertia_positive_finite": True,
    "materials_within_preregistered_bounds": True,
    "no_external_unpinned_dependencies": True,
    "no_articulation": True,
    "no_scripts_or_credentials": True,
}
NATIVE_IMPORT_CHECKS = {
    "stage_import": True,
    "rigid_body_enabled": True,
    "collider_enabled": True,
    "gravity_settle_seconds": 3.0,
    "maximum_settle_translation_m": 0.01,
    "maximum_settle_rotation_rad": 0.08,
    "support_contact_required": True,
    "explosion_or_tunneling_forbidden": True,
    "deterministic_reset_required": True,
    "state_digest_repeat_count": 3,
}
IDENTITY_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
COORDINATE_SYSTEM = {
    "handedness": "right_handed",
    "axis_definition": "XYZ_equals_Right_Back_Up",
    "up_axis": "Z",
    "meters_per_unit": 1.0,
    "interiorgs_to_sage_transform": IDENTITY_TRANSFORM,
    "registration_authority": (
        "publisher_scene_id_join_plus_retained_metric_InteriorGS_labels_and_SAGE_stage_metadata"
    ),
    "physical_metrology_claimed": False,
}
PICK_AND_PLACE_TASK_SPACE_TARGETS = [
    "start_pose",
    "pregrasp_pose",
    "grasp_pose",
    "lift_pose",
    "transport_pose",
    "place_pose",
    "release_pose",
    "retreat_pose",
]
PLANAR_PUSH_TASK_SPACE_TARGETS = [
    "start_pose",
    "push_contact_pose",
    "target_entry_pose",
    "retract_pose",
]
OPENAI_STAGE_CAPS_USD = {
    "artifixer_semantic_teacher": MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD,
    "artifixer_visual_review": MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
    "content_agents": round(
        MAX_EXTERNAL_SERVICE_SPEND_USD
        - MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD
        - MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
        6,
    ),
}
assert OPENAI_STAGE_CAPS_USD["content_agents"] >= MIN_CONTENT_AGENTS_SPEND_USD


def _axis_name(index: int) -> str:
    return "xyz"[index]


def stage_sequence() -> list[dict[str, Any]]:
    """The admitted six-stage adapter chain, ordered by capability."""

    by_capability = {
        identity.capability: identity for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
    }
    rows = []
    for index, capability in enumerate(CAPABILITY_ORDER):
        identity = by_capability[capability]
        rows.append(
            {
                "stage_id": f"stage-{index + 1}",
                "capability": capability,
                "adapter": {"id": identity.adapter_id, "version": identity.version},
                "execution_class": identity.execution_class,
                "depends_on": [] if index == 0 else [f"stage-{index}"],
            }
        )
    return rows


def spend_block() -> dict[str, Any]:
    return {
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "hard_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "hard_ttl_seconds": REQUIRED_PARENT_TTL_SECONDS,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "external_service_caps": {
            "openai": {
                "maximum_cost_usd": round(sum(OPENAI_STAGE_CAPS_USD.values()), 6),
                "maximum_requests": 32,
                "stage_max_cost_usd": dict(OPENAI_STAGE_CAPS_USD),
            }
        },
        "retry_cap": 0,
        "selected_provider": "vast",
        "provider_allowlist": ["vast"],
    }


def source_object_selection(
    *,
    scene_id: str,
    instance_id: str,
    semantic_label: str,
    review_label: str,
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
    collision_prim_path: str,
    task_family: str,
) -> dict[str, Any]:
    center = [(float(low) + float(high)) / 2.0 for low, high in zip(aabb_min, aabb_max, strict=True)]
    return {
        "schema_version": "task_evaluation_source_object_selection.v1",
        "status": "frozen_before_scene_configuration_run",
        "scene_id": scene_id,
        "publisher_instance_id": instance_id,
        "publisher_label": semantic_label,
        "review_label": review_label,
        "appearance_source": "InteriorGS",
        "aabb_min_xyz_m": [float(value) for value in aabb_min],
        "aabb_max_xyz_m": [float(value) for value in aabb_max],
        "center_xyz_m": center,
        "collision_candidate_prim": collision_prim_path,
        "selection_reason": (
            "source-visible, rigid, metrically bounded, isolated on a horizontal "
            f"support, and suitable for exact {task_family} evaluation"
        ),
        "outcomes_observed_before_selection": False,
        "source_object_is_physics_authority": False,
    }


def support_plane_input(
    *,
    scene_id: str,
    instance_id: str,
    semantic_label: str,
    sage_prim_path: str,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
) -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_support_plane_input.v1",
        "status": "frozen_candidate_pending_production_validation",
        "scene_id": scene_id,
        "publisher_instance_id": instance_id,
        "publisher_label": semantic_label,
        "sage_prim_path": sage_prim_path,
        "top_z_m": float(bounds_max[2]),
        "bounds_min_xyz_m": [float(value) for value in bounds_min],
        "bounds_max_xyz_m": [float(value) for value in bounds_max],
        "required_validation": [
            "planarity",
            "finite_bounds",
            "support_contact",
            "target_region_inside_bounds",
        ],
    }


def metric_registration_input(*, scene_id: str) -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_metric_registration_input.v1",
        "status": "candidate_registration_input",
        "scene_id": scene_id,
        "handedness": "right_handed",
        "axis_definition": "XYZ_equals_Right_Back_Up",
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "interiorgs_to_sage_transform": IDENTITY_TRANSFORM,
        "registration_authority": (
            "publisher_scene_id_join_plus_metric_InteriorGS_labels_and_SAGE_stage_metadata"
        ),
        "physical_metrology_claimed": False,
        "production_validation_required": True,
    }


def robot_mount_interface_plan(*, scene_id: str, strategy: str) -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_robot_mount_interface_plan.v1",
        "status": "publish_during_scene_configuration_run",
        "scene_id": scene_id,
        "scene_base_frame": f"interiorgs_{scene_id}_world",
        "mount_interface_output": "robot_neutral_scene_mount_frame_and_allowed_base_region",
        "supported_robot_classes": ["fixed_arm", "mobile_manipulator"],
        "robot_specific_base_transform_supplied_per_evaluation": True,
        "robot_specific_kinematics_joint_bounds_and_reachability_supplied_per_evaluation": True,
        "workspace_clearance_envelope_required": True,
        "minimum_non_target_clearance_m": 0.03,
        "configuration_run_must_not_claim_any_robot_qualified": True,
        "task_space_targets": list(
            PICK_AND_PLACE_TASK_SPACE_TARGETS
            if strategy == "pick_and_place"
            else PLANAR_PUSH_TASK_SPACE_TARGETS
        ),
    }


def camera_calibration_plan(*, scene_id: str, strategy: str) -> dict[str, Any]:
    visibility = (
        ["start", "grasp", "lift", "transport", "place", "release", "retreat"]
        if strategy == "pick_and_place"
        else ["start", "first_contact", "maximum_displacement", "target_entry", "retract"]
    )
    return {
        "schema_version": "task_evaluation_scene_camera_calibration_plan.v1",
        "status": "solve_during_scene_configuration_run",
        "scene_id": scene_id,
        "required_cameras": ["policy_external", "wrist", "overview_review_only"],
        "required_fields": [
            "camera_id",
            "pose_scene_from_camera",
            "intrinsics",
            "width",
            "height",
            "frame_convention",
            "calibration_digest",
            "renderer_binding",
        ],
        "target_visibility_required": visibility,
        "policy_inputs_lossless": True,
        "overview_is_policy_input": False,
        "calibration_must_be_digest_bound": True,
    }


def renderer_qualification_plan() -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_renderer_qualification_plan.v1",
        "status": "execute_during_scene_configuration_run",
        "appearance_source": "InteriorGS",
        "browser_preview_qualifies": False,
        "debug_sage_render_qualifies_as_appearance": False,
        "maximum_fidelity_claim_requires_independent_qualification": True,
        "required_bindings": [
            "renderer_name",
            "renderer_version",
            "environment_digest",
            "camera_pose",
            "camera_intrinsics",
            "source_splat_digest",
            "retained_gaussian_count",
            "render_dimensions",
            "supersampling",
            "color_and_alpha_handling",
            "rendered_image_digests",
            "fidelity_qualification_receipt",
        ],
    }


def stage_one_configuration(
    *,
    scene_id: str,
    source_object: Mapping[str, Any],
    support_label: str,
    human_authority: Mapping[str, Any],
) -> dict[str, Any]:
    label = str(source_object["publisher_label"])
    instance = str(source_object["publisher_instance_id"])
    return {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "status": PENDING_STATUS,
        "scene_id": scene_id,
        "appearance_authority": "InteriorGS source splat",
        "source_object": {
            "publisher_instance_id": instance,
            "semantic_label": label,
            "review_label": str(source_object["review_label"]),
            "aabb_min_xyz_m": list(source_object["aabb_min_xyz_m"]),
            "aabb_max_xyz_m": list(source_object["aabb_max_xyz_m"]),
            "center_xyz_m": list(source_object["center_xyz_m"]),
        },
        "edit_instruction": (
            f"Remove only the {label} instance {instance} and reconstruct the "
            f"immediately occluded {support_label} surface. Preserve every other "
            "observed object, surface, and lighting cue exactly."
        ),
        "production_render_required": True,
        "gaussian_cutout": {
            "selection_rule": "gaussian_center_inside_registered_source_object_aabb",
            "aabb_padding_m": 0.0,
            "retained_rows_must_remain_byte_exact": True,
        },
        "required_views": {
            "minimum": 8,
            "lossless_inputs": True,
            "mask_source": "registered_source_object_bounds_projection",
            "must_cover": ["front", "rear", "left", "right", "top_oblique"],
        },
        "render_inputs_must_bind": [
            "renderer_name_and_exact_version",
            "container_or_environment_digest",
            "camera_pose_and_intrinsics",
            "source_splat_digest_and_retained_count",
            "dimensions_supersampling_color_and_alpha",
            "rendered_image_digests",
            "renderer_fidelity_qualification",
        ],
        "provider_disclosure": {
            "raw_interiorgs_bytes": False,
            "source_appearance_bytes": False,
            "derived_rendered_views": True,
            "provider_training": False,
            "public_redistribution": False,
        },
        "output_requirements": {
            "edited_lossless_views": True,
            "per_view_masks": True,
            "generated_pixels_labeled": True,
            "multiview_consistency_report": True,
            "source_and_output_digests_separate": True,
        },
        "human_authority": {
            "accepted_by": str(human_authority["accepted_by"]),
            "accepted_on": str(human_authority["accepted_on"]),
            "authority_reference": str(human_authority["authority_reference"]),
            "private_derived_frame_disclosure_authorized": human_authority["private_derived_frame_disclosure_authorized"],
            "provider_retention_terms_accepted": human_authority["provider_retention_terms_accepted"],
            "provider_training_terms_accepted": human_authority["provider_training_terms_accepted"],
            "provider_training_authorized": human_authority["provider_training_authorized"],
            **{key: human_authority[key] for key in (
                "full_source_provider_disclosure_authority", "full_source_provider_disclosure_authorities")
               if key in human_authority},
        },
        "claim_boundary": (
            "The edit is generated appearance support. It never becomes observed "
            "source truth or physics authority."
        ),
    }


def stage_two_configuration(
    *,
    scene_id: str,
    collision_source_digest: str,
    target_match: Mapping[str, Any],
    support_prim_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": "collision_object_excision_configuration.v1",
        "status": PENDING_STATUS,
        "scene_id": scene_id,
        "operation": "deactivate_exact_prim_only",
        "collision_source_digest": collision_source_digest,
        "exact_target_prim": str(target_match["prim_path"]),
        "expected_target": {
            "aabb_min_xyz_m": list(target_match["world_aabb_min_m"]),
            "aabb_max_xyz_m": list(target_match["world_aabb_max_m"]),
            "point_count": int(target_match["point_count"]),
            "face_count": int(target_match["face_count"]),
        },
        "support_prim_must_remain_active": support_prim_path,
        "neighbor_preservation_required": True,
        "validation": {
            "target_absent_after_excision": True,
            "all_non_target_prim_digests_unchanged": True,
            "stage_units_and_up_axis_unchanged": True,
            "before_and_after_prim_manifests_required": True,
        },
        "claim_boundary": (
            "SAGE geometry is independently validated candidate collision support "
            "and is not observed appearance truth."
        ),
    }


def metric_envelope_tolerance(
    *, source_min: Sequence[float], source_max: Sequence[float], target_match: Mapping[str, Any]
) -> float:
    """Smallest admitted relative dimension error that the measured collider passes."""

    observed = 0.0
    for axis in range(3):
        source_dimension = float(source_max[axis]) - float(source_min[axis])
        target_dimension = float(target_match["world_aabb_max_m"][axis]) - float(
            target_match["world_aabb_min_m"][axis]
        )
        if source_dimension <= 0.0:
            raise ValueError("metric_envelope_source_dimension_invalid")
        observed = max(observed, abs(target_dimension - source_dimension) / source_dimension)
    return round(max(0.05, math.ceil(observed * 100.0 + 1.0) / 100.0), 2)


def stage_three_configuration(
    *,
    scene_id: str,
    replacement_identity: Mapping[str, Any],
    source_instance_id: str,
    authoring_target: str,
    source_min: Sequence[float],
    source_max: Sequence[float],
    dimension_tolerance: float,
    physics_bounds: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        "schema_version": "rigid_replacement_authoring_configuration.v1",
        "status": PENDING_STATUS,
        "scene_id": scene_id,
        "replacement_identity": dict(replacement_identity),
        "source_object_identity": f"publisher-instance-{source_instance_id}",
        "authoring_target": authoring_target,
        "appearance_inputs": "digest_bound_derived_views_from_stage_1_only",
        "geometry_support": "publisher_metric_bounds_and_exact_SAGE_target_bounds",
        "metric_envelope": {
            "minimum_xyz_m": [float(value) for value in source_min],
            "maximum_xyz_m": [float(value) for value in source_max],
            "maximum_dimension_relative_error": float(dimension_tolerance),
        },
        "required_output": {
            "format": "OpenUSD",
            "units": "meters",
            "up_axis": "Z",
            "rigid_body": True,
            "single_movable_root": True,
            "visual_mesh_separate_from_collision": True,
            "mass_kg_bounds": [float(value) for value in physics_bounds["mass_kg_bounds"]],
            "static_friction_bounds": [
                float(value) for value in physics_bounds["static_friction_bounds"]
            ],
            "dynamic_friction_bounds": [
                float(value) for value in physics_bounds["dynamic_friction_bounds"]
            ],
            "restitution_bounds": [float(value) for value in physics_bounds["restitution_bounds"]],
        },
        "physics_authority_granted_by_authoring": False,
        "provider_disclosure": {
            "derived_views_and_metric_envelope": True,
            "raw_interiorgs_bytes": False,
            "provider_training": False,
            "public_redistribution": False,
        },
        "claim_boundary": (
            "Content Agents or equivalent may propose the replacement, but only "
            "independent static and native-import qualification admit it."
        ),
    }


def stage_four_configuration(
    *, replacement_identity: Mapping[str, Any], dimension_tolerance: float
) -> dict[str, Any]:
    return {
        "schema_version": "replacement_static_qualification_configuration.v1",
        "status": PENDING_STATUS,
        "replacement_identity": dict(replacement_identity),
        "required_checks": dict(STATIC_CHECKS),
        "center_of_mass_must_lie_inside_collision_bounds": True,
        "dimension_tolerance_relative": float(dimension_tolerance),
        "physics_authority_on_pass": "static_candidate_only_pending_native_import",
    }


def stage_five_configuration(*, replacement_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "replacement_native_import_qualification_configuration.v1",
        "status": PENDING_STATUS,
        "replacement_identity": dict(replacement_identity),
        "required_checks": dict(NATIVE_IMPORT_CHECKS),
        "runtime": {
            "engine": "Isaac Sim",
            "version_must_match_deployed_runtime": True,
            "container_digest_must_be_pinned": True,
        },
        "physics_authority_on_pass": "qualified_replacement_asset",
    }


def stage_six_configuration(
    *,
    scene_identity: Mapping[str, Any],
    support_plane: Mapping[str, Any],
    start_center: Sequence[float],
    bottom_z: float,
) -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_scene_assembly_configuration.v1",
        "status": PENDING_STATUS,
        "scene_identity": dict(scene_identity),
        "appearance": {
            "source": "InteriorGS",
            "apply_generated_edit_from_stage": "stage-1",
            "generated_region_label_required": True,
        },
        "collision": {
            "source": "SAGE-derived",
            "apply_exact_excision_from_stage": "stage-2",
            "support_prim": str(support_plane["sage_prim_path"]),
        },
        "support_plane": {
            "publisher_instance_id": str(support_plane["publisher_instance_id"]),
            "top_z_m": float(support_plane["top_z_m"]),
            "bounds_min_xyz_m": list(support_plane["bounds_min_xyz_m"]),
            "bounds_max_xyz_m": list(support_plane["bounds_max_xyz_m"]),
        },
        "replacement": {
            "qualified_asset_from_stage": "stage-5",
            "start_center_xyz_m": [float(value) for value in start_center],
            "bottom_z_m": float(bottom_z),
            "initial_orientation_rule": "stable_source_orientation_selected_before_any_episode",
            "source_and_replacement_visual_instances_must_not_coexist": True,
            "source_and_replacement_collision_instances_must_not_coexist": True,
        },
        "camera_registration": {
            "policy_external_camera": True,
            "wrist_camera": True,
            "overview_review_camera": True,
            "all_intrinsics_extrinsics_and_frame_transforms_required": True,
        },
        "robot_mount_interface": {
            "publish_robot_neutral_scene_mount_frame": True,
            "publish_allowed_base_region": True,
            "publish_task_space_targets": True,
            "supported_robot_classes": ["fixed_arm", "mobile_manipulator"],
            "minimum_non_target_clearance_m": 0.03,
            "robot_specific_base_transform_and_reachability_deferred_to_each_evaluation": True,
        },
        "evaluation_episode_executed_in_this_run": False,
        "scene_construction_repeated_per_evaluation": False,
        "terminal_output": "task_evaluation_configured_scene_revision.v1",
    }


def pick_and_place_task_records(
    *,
    task_identity: Mapping[str, Any],
    object_identity: Mapping[str, Any],
    start_center: Sequence[float],
    target_center: Sequence[float],
    source_min: Sequence[float],
    source_max: Sequence[float],
    grasp_axis: int,
    grasp_sign: float,
    success: Mapping[str, Any],
    resolved_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Template, success criteria, and execution spec for one pick-and-place task.

    The grasp is authored on the subject face that points away from the
    destination along the support's long axis (``grasp_sign`` is the sign of that
    face's offset in the scoring frame), with a vertical parallel-jaw axis so the
    thin subject is pinched across its thickness.
    """

    frequency = float(success["control_frequency_hz"])
    seconds = float(success["maximum_episode_seconds"])
    steps = int(round(frequency * seconds))
    if not math.isclose(steps / frequency, seconds, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("pick_and_place_episode_timing_not_integral")
    half_extents = [
        (float(source_max[axis]) - float(source_min[axis])) / 2.0 for axis in range(3)
    ]
    contact = [0.0, 0.0, 0.0]
    contact[grasp_axis] = grasp_sign * half_extents[grasp_axis]
    outward = [0.0, 0.0, 0.0]
    outward[grasp_axis] = grasp_sign
    bounds = {
        "authority": "deterministic_simulator_state",
        "minimum_planar_displacement_m": float(success["minimum_planar_displacement_m"]),
        "maximum_final_planar_target_error_m": float(
            success["maximum_final_planar_target_error_m"]
        ),
        "minimum_lift_m": float(success["minimum_lift_m"]),
        "release_required": True,
        "drop_events_allowed": False,
        "maximum_retries": int(success["maximum_retries"]),
        "maximum_regrasps": int(success["maximum_regrasps"]),
        "forbidden_collision_allowed": False,
        "joint_limit_violation_allowed": False,
        "owner_success_contract_required": True,
        "per_cell_controls_required": True,
        "retreat_clearance_required": True,
        "whole_subject_containment_required": True,
        "object_must_rest_on_destination_support": True,
    }
    for field in ("retreat_clearance_m", "drop_minimum_fall_m",
                  "maximum_task_contact_force_n", "forbidden_contact_classes",
                  "robot_workspace_position_bounds_world_m", "collision_failure_minimum_force_n"):
        if field in success:
            bounds[field] = success[field]
    success_record = {
        "schema_version": "task_evaluation_rigid_relocation_success_criteria.v1",
        "status": "preregistered_before_any_episode",
        "target_center_xyz_m": [float(value) for value in target_center],
        **bounds,
    }
    template = {
        "schema_version": "task_evaluation_rigid_relocation_template.v1",
        "status": "preregistered_candidate_pending_configured_scene_revision",
        "task_identity": dict(task_identity),
        "object_identity": dict(object_identity),
        "strategy": "pick_and_place",
        "start_center_xyz_m": [float(value) for value in start_center],
        "target_center_xyz_m": [float(value) for value in target_center],
        "success": dict(bounds),
        "control_frequency_hz": success["control_frequency_hz"],
        "maximum_step_count": steps,
        "maximum_episode_seconds": seconds,
        "resolved_seed": int(resolved_seed),
        "controls_order": ["zero_action", "deterministic_scripted"],
        "failure_metrics": [
            "insufficient_displacement",
            "final_target_error_exceeded",
            "drop_event",
            "subject_outside_destination",
            "release_incomplete",
            "forbidden_collision",
            "joint_limit_violation",
            "timeout",
        ],
        "interaction_affordance": {
            "contact_point_scoring_frame_m": contact,
            "approach_unit_scoring_frame": outward,
            "jaw_unit_scoring_frame": [0.0, 0.0, 1.0],
            "lift_unit_world": [0.0, 0.0, 1.0],
            "pregrasp_clearance_m": float(success["pregrasp_clearance_m"]),
            "minimum_lift_m": float(success["minimum_lift_m"]),
            "grasp_face_axis": _axis_name(grasp_axis),
            "grasp_face_sign": grasp_sign,
        },
        "preregistration_rule": (
            "Any scientific task, target, tolerance, object, or success-rule change "
            "creates a new immutable template version before any episode runs."
        ),
    }
    execution = {
        "schema_version": "task_evaluation_rigid_relocation_execution_spec.v1",
        "status": "preregistered_before_any_episode",
        "strategy": "pick_and_place",
        "start_center_xyz_m": [float(value) for value in start_center],
        "target_center_xyz_m": [float(value) for value in target_center],
        "control_frequency_hz": success["control_frequency_hz"],
        "maximum_step_count": steps,
        "maximum_episode_seconds": seconds,
        "resolved_seed": int(resolved_seed),
        "action_bounds_m_per_step": {"minimum": -0.02, "maximum": 0.02},
        "collision_exclusions": ["robot_self_collision_pairs_declared_by_robot_configuration"],
        "termination": [
            "success",
            "drop_event",
            "forbidden_collision",
            "joint_limit_violation",
            "timeout",
        ],
    }
    return template, success_record, execution


def rights_admission(
    *,
    scene_id: str,
    publisher_revisions: Mapping[str, str],
    terms_sha256: str,
    interiorgs_readme_sha256: str,
    sage_readme_sha256: str,
    human_authority: Mapping[str, Any],
    provider: str = "vast",
) -> dict[str, Any]:
    reference = str(human_authority["authority_reference"])
    accepted_on = str(human_authority["accepted_on"])
    accepted_by = str(human_authority["accepted_by"])
    return {
        "schema_version": "task_evaluation_scene_rights_admission.v1",
        "status": "admitted_for_internal_development",
        "program_id": "arm-decision-proof-v1",
        "product": "Task Evaluation Run",
        "scene_id": scene_id,
        "publisher_scene_id": scene_id,
        "publication_status": (
            "candidate_pending_exact_deployed_sha_binding_and_full_byte_readback"
        ),
        "declared_use_scope": (
            "internal_noncommercial_research_and_development_Task_Evaluation_Run"
        ),
        "interiorgs": {
            "repository": "spatialverse/InteriorGS",
            "revision": publisher_revisions["interiorgs"],
            "license": "custom_gated_noncommercial_research_terms",
            "permitted_purpose": "noncommercial_research_and_education",
            "commercial_use_allowed": False,
            "attribution_required": True,
            "raw_dataset_redistribution_allowed": False,
            "training_use_authorized": False,
            "terms_text_sha256": terms_sha256,
            "publisher_readme_sha256": interiorgs_readme_sha256,
        },
        "sage": {
            "collision_repository": "spatialverse/SAGE-3D_Collision_Mesh",
            "collision_revision": publisher_revisions["sage_collision"],
            "license": "CC-BY-NC-4.0",
            "commercial_use_allowed": False,
            "attribution_required": True,
            "training_use_authorized": False,
            "publisher_readme_sha256": sage_readme_sha256,
        },
        "private_provider_processing_allowed": human_authority["private_derived_frame_disclosure_authorized"],
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
        "provider_disclosure": {
            "intended_provider": provider,
            "raw_interiorgs_downloaded_bytes_may_be_uploaded": False,
            "source_appearance_downloaded_bytes_may_be_uploaded": False,
            "interiorgs_labels_or_structure_may_be_uploaded": False,
            "sage_collision_runtime_bytes_may_be_privately_processed": True,
            "minimum_digest_bound_derived_appearance_runtime_bytes_may_be_privately_processed": True,
            "qualified_replacement_usd_may_be_privately_processed": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
            "provider_retention_rule": (
                "bounded_to_the_exact_Task_Evaluation_Run_then_governed_teardown_and_provider_zero"
            ),
            "network_egress_rule": (
                "fail_closed_except_pinned_runtime_dependencies_and_governed_result_sync"
            ),
            "claim_boundary": (
                "Only minimum derived runtime inputs may leave the control plane. "
                "Raw publisher bytes remain on the production control plane; "
                "this packet grants no additional publisher or disclosure rights."
            ),
        },
        "amendments": [],
        "authority_records": [
            {
                "authority_kind": "explicit_user_direction_in_current_goal",
                "authority_reference": reference,
                "authorized_by": accepted_by,
                "declared_scope": f"internal_and_development_only_use_of_scene_{scene_id}",
                "recorded_on": accepted_on,
            }
        ],
        "claim_boundary": (
            "Rights are admitted for this internal development-only Task Evaluation Run. "
            "This is not a commercial-use, public-redistribution, dataset-publication, or "
            "customer-delivery license."
        ),
    }


def recipe(
    *,
    recipe_id: str,
    team_namespace: str,
    scene_identity: Mapping[str, Any],
    task_identity: Mapping[str, Any],
    subject_identity: Mapping[str, Any],
    source_manifest_digest: str,
    rights_admission_digest: str,
    output_identity: Mapping[str, Any],
    stage_configuration_references: Sequence[Mapping[str, Any]],
    supplemental_destination: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stages = stage_sequence()
    if len(stage_configuration_references) != len(stages):
        raise ValueError("recipe_stage_configuration_reference_count_invalid")
    for stage, reference in zip(stages, stage_configuration_references, strict=True):
        stage["configuration"] = dict(reference)
    value: dict[str, Any] = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "recipe_id": recipe_id,
        "team_namespace": team_namespace,
        "scene_identity": dict(scene_identity),
        "task_identity": dict(task_identity),
        "subject_identity": dict(subject_identity),
        "source_manifest_digest": source_manifest_digest,
        "rights_admission_digest": rights_admission_digest,
        "stage_sequence": stages,
        "output_identity": dict(output_identity),
        "provider_disclosure": {
            "raw_source_bytes_to_external_provider": False,
            "derived_runtime_processing_allowed": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
        },
        "recipe_digest": "",
    }
    if supplemental_destination is not None:
        value["supplemental_destination"] = dict(supplemental_destination)
    value["recipe_digest"] = canonical_digest(value, digest_field="recipe_digest")
    return value


def runtime_health_protocol(*, source_commit: str) -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_container_health_protocol.v1",
        "identity": {"id": "task-evaluation-scene-configuration-provider", "version": "v1"},
        "startup": {
            "protocol": "file_receipt",
            "relative_path": "task_evaluation_scene_configuration_provider_result.v1.json",
            "maximum_seconds": REQUIRED_PARENT_TTL_SECONDS,
        },
        "terminal": {
            "container_exit_status_retained": True,
            "stdout_stderr_retained": True,
            "provider_result_digest_required": True,
            "watchdog_and_teardown_required": True,
        },
        "network": {"default": "deny", "runtime_allowlist_is_server_owned": True},
        "expected_production_commit": source_commit,
    }


def exact_production_release_binding(
    *,
    team_namespace: str,
    scene_identity: Mapping[str, Any],
    source_commit: str,
    deploy_receipt: Mapping[str, Any],
    deploy_receipt_sha256: str,
    release_environment_sha256: str,
    scene_configuration_publication: Mapping[str, Any],
    splat_render_publication: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = deploy_receipt["release_provenance"]
    return {
        "schema_version": "task_evaluation_exact_production_release_binding.v1",
        "status": "production_commit_proven",
        "program_id": "arm-decision-proof-v1",
        "product": "Task Evaluation Run",
        "team_namespace": team_namespace,
        "scene_identity": dict(scene_identity),
        "source_commit": source_commit,
        "promotion": {
            "workflow": "Full Test Lane",
            "provenance_status": provenance["provenance_status"],
            "promotion_eligible": bool(provenance["promotion_eligible"]),
            "canonical_full_lane_verified": bool(provenance["canonical_full_lane_verified"]),
            "evidence_grade": "development_only",
            "run_id": provenance.get("run_id"),
            "test_count": None,
            "skip_count": None,
            "provenance_sha256": provenance["sha256"],
        },
        "deployment": {
            "initial_receipt_sha256": deploy_receipt_sha256,
            "receipt_sha256": deploy_receipt_sha256,
            "intake_commit_proven": bool(deploy_receipt["intake_runtime"]["commit_proven"]),
            "provider_mutation_performed": bool(deploy_receipt.get("provider_mutation_performed")),
            "exact_source_deployer_used": True,
        },
        "scene_configuration_runtime": {
            "renderer_runtime_digest": splat_render_publication["runtime_digest"],
            "renderer_publication_receipt_digest": splat_render_publication["receipt_digest"],
            "toolchain_digest": scene_configuration_publication["toolchain_digest"],
            "toolchain_publication_receipt_digest": scene_configuration_publication[
                "receipt_digest"
            ],
            "full_byte_service_account_readback_passed": True,
        },
        "object_store": {
            "scheme": "s3",
            "bucket": "blueprint",
            "namespace_prefix": "task-evaluation/production-inputs/",
            "release_environment_sha256": release_environment_sha256,
        },
        "claim_boundary": {
            "scene_inputs_are_bound_to_this_release": True,
            "scene_specific_artifacts_built_by_release": False,
            "simulator_episode_executed": False,
            "physical_world_truth_claimed": False,
        },
    }


__all__ = [
    "COORDINATE_SYSTEM",
    "OPENAI_STAGE_CAPS_USD",
    "camera_calibration_plan",
    "exact_production_release_binding",
    "metric_envelope_tolerance",
    "metric_registration_input",
    "pick_and_place_task_records",
    "recipe",
    "renderer_qualification_plan",
    "rights_admission",
    "robot_mount_interface_plan",
    "runtime_health_protocol",
    "source_object_selection",
    "spend_block",
    "stage_five_configuration",
    "stage_four_configuration",
    "stage_one_configuration",
    "stage_sequence",
    "stage_six_configuration",
    "stage_three_configuration",
    "stage_two_configuration",
    "support_plane_input",
]
