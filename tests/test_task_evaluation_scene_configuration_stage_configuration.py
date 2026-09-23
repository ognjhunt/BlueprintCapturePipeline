from __future__ import annotations

import json

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import (
    TaskEvaluationSceneConfigurationStageConfigurationError,
    validate_immutable_stage_configurations,
)


SUBJECT = {"id": "scene-839873-mug-replacement", "version": "v1"}
SCENE = {"id": "interiorgs-839873", "version": "mug-v1"}
COLLISION_DIGEST = "sha256:" + "c" * 64


def _checks() -> list[dict]:
    return [
        {
            "schema_version": "observed_appearance_object_removal_configuration.v1",
            "source_object": {
                "publisher_instance_id": "104",
                "aabb_min_xyz_m": [0.0, 0.0, 0.0],
                "aabb_max_xyz_m": [0.2, 0.2, 0.3],
            },
            "production_render_required": True,
            "gaussian_cutout": {
                "selection_rule": "gaussian_center_inside_registered_source_object_aabb",
                "aabb_padding_m": 0.01,
                "retained_rows_must_remain_byte_exact": True,
            },
            "required_views": {
                "minimum": 8,
                "lossless_inputs": True,
                "mask_source": "registered_source_object_bounds_projection",
            },
            "provider_disclosure": {
                "raw_interiorgs_bytes": False,
                "derived_rendered_views": True,
            },
            "output_requirements": {"generated_pixels_labeled": True},
            "human_authority": {
                "accepted_by": "fixture-owner",
                "accepted_on": "2026-08-27",
                "authority_reference": "website-submission-839873",
                "private_derived_frame_disclosure_authorized": True,
                "provider_retention_terms_accepted": True,
                "provider_training_terms_accepted": True,
                "provider_training_authorized": False,
            },
        },
        {
            "schema_version": "collision_object_excision_configuration.v1",
            "collision_source_digest": COLLISION_DIGEST,
            "exact_target_prim": "/Root/Target",
            "expected_target": {"point_count": 1, "face_count": 1},
            "operation": "deactivate_exact_prim_only",
            "validation": {
                "target_absent_after_excision": True,
                "all_non_target_prim_digests_unchanged": True,
                "stage_units_and_up_axis_unchanged": True,
                "before_and_after_prim_manifests_required": True,
            },
        },
        {
            "schema_version": "rigid_replacement_authoring_configuration.v1",
            "replacement_identity": SUBJECT,
            "metric_envelope": {
                "minimum_xyz_m": [0.0, 0.0, 0.0],
                "maximum_xyz_m": [0.2, 0.2, 0.3],
                "maximum_dimension_relative_error": 0.05,
            },
            "required_output": {
                "format": "OpenUSD",
                "rigid_body": True,
                "single_movable_root": True,
                "units": "meters",
                "up_axis": "Z",
                "mass_kg_bounds": [0.2, 0.8],
                "static_friction_bounds": [0.3, 0.9],
                "dynamic_friction_bounds": [0.2, 0.8],
                "restitution_bounds": [0.0, 0.15],
            },
            "physics_authority_granted_by_authoring": False,
        },
        {
            "schema_version": "replacement_static_qualification_configuration.v1",
            "replacement_identity": SUBJECT,
            "required_checks": {
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
            },
            "center_of_mass_must_lie_inside_collision_bounds": True,
        },
        {
            "schema_version": "replacement_native_import_qualification_configuration.v1",
            "replacement_identity": SUBJECT,
            "required_checks": {
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
            },
        },
        {
            "schema_version": "task_evaluation_scene_assembly_configuration.v1",
            "scene_identity": SCENE,
            "replacement": {
                "qualified_asset_from_stage": "stage-5",
                "source_and_replacement_visual_instances_must_not_coexist": True,
                "source_and_replacement_collision_instances_must_not_coexist": True,
            },
            "robot_mount_interface": {
                "publish_robot_neutral_scene_mount_frame": True,
                "robot_specific_base_transform_and_reachability_deferred_to_each_evaluation": True,
            },
            "evaluation_episode_executed_in_this_run": False,
            "scene_construction_repeated_per_evaluation": False,
        },
    ]


def _envelope() -> dict:
    capabilities = (
        "observed_appearance_object_removal",
        "collision_object_excision",
        "rigid_replacement_authoring",
        "replacement_static_qualification",
        "replacement_native_import_qualification",
        "scene_assembly",
    )
    return {
        "recipe": {
            "subject_identity": SUBJECT,
            "scene_identity": SCENE,
            "stage_sequence": [
                {"stage_id": f"stage-{index}", "capability": capability}
                for index, capability in enumerate(capabilities, start=1)
            ],
        },
        "materialized_references": [
            {
                "contract_path": "scene.geometry.collision",
                "digest": COLLISION_DIGEST,
            }
        ],
        "render_inputs_result": {},
    }


def _configuration_map() -> dict[str, dict]:
    return {
        f"stage-{index}": value
        for index, value in enumerate(_checks(), start=1)
    }


def test_all_six_immutable_stage_configurations_pass_no_spend_preflight() -> None:
    validate_immutable_stage_configurations(
        envelope=_envelope(), configurations=_configuration_map()
    )


@pytest.mark.parametrize(
    ("stage_id", "key", "capability", "predicate"),
    [
        (
            "stage-1",
            "human_authority",
            "observed_appearance_object_removal",
            "human_authority",
        ),
        ("stage-2", "operation", "collision_object_excision", "operation"),
        (
            "stage-3",
            "physics_authority_granted_by_authoring",
            "rigid_replacement_authoring",
            "physics_authority_granted_by_authoring",
        ),
        (
            "stage-4",
            "required_checks",
            "replacement_static_qualification",
            "required_checks",
        ),
        (
            "stage-5",
            "required_checks",
            "replacement_native_import_qualification",
            "required_checks",
        ),
        ("stage-6", "robot_mount_interface", "scene_assembly", "robot_mount_interface"),
    ],
)
def test_each_stage_refuses_missing_config_before_provider_mutation(
    stage_id: str, key: str, capability: str, predicate: str
) -> None:
    configurations = json.loads(json.dumps(_configuration_map()))
    configurations[stage_id].pop(key)

    with pytest.raises(
        TaskEvaluationSceneConfigurationStageConfigurationError,
        match=(
            "scene_configuration_stage_configuration_preflight_failed:"
            f"{stage_id}:{capability}:{predicate}"
        ),
    ):
        validate_immutable_stage_configurations(
            envelope=_envelope(), configurations=configurations
        )


def _articulated_map() -> dict[str, dict]:
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
        articulated_stage_three_configuration, stage_five_configuration, stage_four_configuration,
    )
    configurations = _configuration_map()
    third = configurations["stage-3"]
    mechanism = {"part_label": "middle drawer", "joint_type": "prismatic", "estimated_usable_stroke_m": 0.3,
                 "estimated_front_normal_world": [0.0, -1.0, 0.0], "lock_status": "unknown"}
    physics = {"mass_kg_bounds": [8.0, 30.0], "task_part_mass_kg_bounds": [0.5, 6.0],
               "static_friction_bounds": [0.3, 0.8], "dynamic_friction_bounds": [0.2, 0.6],
               "restitution_bounds": [0.0, 0.2], "joint_friction_bounds": [1.0, 15.0],
               "joint_damping_bounds": [1.0, 30.0]}
    configurations["stage-3"] = articulated_stage_three_configuration(
        scene_id=str(third.get("scene_id") or "scene-1"), replacement_identity=third["replacement_identity"], source_instance_id="cabinet-1",
        authoring_target="three-drawer cabinet", source_min=third["metric_envelope"]["minimum_xyz_m"],
        source_max=third["metric_envelope"]["maximum_xyz_m"],
        dimension_tolerance=third["metric_envelope"]["maximum_dimension_relative_error"],
        physics_bounds=physics, mechanism=mechanism)
    configurations["stage-4"] = stage_four_configuration(replacement_identity=third["replacement_identity"],
                                                          dimension_tolerance=0.2, articulated=True)
    configurations["stage-5"] = stage_five_configuration(replacement_identity=third["replacement_identity"], articulated=True)
    return configurations


def test_articulated_stage_configurations_pass_and_refuse_a_driven_task_joint() -> None:
    envelope = _envelope()
    configurations = _articulated_map()
    validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)
    driven = json.loads(json.dumps(configurations))
    driven["stage-3"]["mechanism"]["passive_dynamics"]["task_joint_drive"] = "position"
    with pytest.raises(ValueError, match="stage-3"):
        validate_immutable_stage_configurations(envelope=envelope, configurations=driven)
    relaxed = json.loads(json.dumps(configurations))
    relaxed["stage-4"]["required_checks"]["non_target_joints_fixed"] = False
    with pytest.raises(ValueError, match="stage-4"):
        validate_immutable_stage_configurations(envelope=envelope, configurations=relaxed)
    assisted = json.loads(json.dumps(configurations))
    assisted["stage-5"]["required_checks"]["task_joint_drive_forbidden"] = False
    with pytest.raises(ValueError, match="stage-5"):
        validate_immutable_stage_configurations(envelope=envelope, configurations=assisted)


def test_anthropic_stage_requires_the_same_launch_request_provider() -> None:
    envelope = _envelope()
    configurations = _articulated_map()
    authoring = configurations["stage-3"]
    authoring["authoring_backend"] = "astra_cad_blender_v1"
    authoring["authoring_model_provider"] = "anthropic"
    authoring["source_observation_kind"] = "website_capture_frames"
    with pytest.raises(ValueError, match="authoring_provider_budget_binding"):
        validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)
    envelope["request"] = {"replacement_authoring_backend": "astra_cad_blender_v1",
                           "replacement_authoring_model_provider": "anthropic"}
    validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)


def test_stage_three_refuses_physically_impossible_friction_bounds() -> None:
    configurations = _configuration_map()
    required = configurations["stage-3"]["required_output"]
    required["static_friction_bounds"] = [0.1, 0.2]
    required["dynamic_friction_bounds"] = [0.3, 0.4]

    with pytest.raises(
        TaskEvaluationSceneConfigurationStageConfigurationError,
        match=(
            "scene_configuration_stage_configuration_preflight_failed:"
            "stage-3:rigid_replacement_authoring:"
            "required_output.friction_bounds_feasible"
        ),
    ):
        validate_immutable_stage_configurations(
            envelope=_envelope(), configurations=configurations
        )


@pytest.mark.parametrize(
    ("case", "predicate"),
    [
        ("bounds", "source_object.aabb"),
        ("gaussian_cutout", "gaussian_cutout"),
        ("disclosure_intent", "provider_disclosure"),
        ("derived_views", "provider_disclosure"),
        ("view_count", "required_views.minimum"),
        ("lossless_views", "required_views.minimum"),
        ("mask_source", "required_views.minimum"),
        ("tuning", "artifixer_tuning"),
    ],
)
def test_stage_one_refuses_each_late_config_predicate_before_provider_mutation(
    case: str, predicate: str
) -> None:
    configurations = _configuration_map()
    stage = configurations["stage-1"]
    if case == "bounds":
        stage["source_object"]["aabb_max_xyz_m"] = [0.0, 0.2, 0.3]
    elif case == "gaussian_cutout":
        stage["gaussian_cutout"]["aabb_padding_m"] = 0.11
    elif case == "disclosure_intent":
        stage["provider_disclosure"].pop("raw_interiorgs_bytes")
    elif case == "derived_views":
        stage["provider_disclosure"]["derived_rendered_views"] = False
    elif case == "view_count":
        stage["required_views"]["minimum"] = 9
    elif case == "lossless_views":
        stage["required_views"]["lossless_inputs"] = False
    elif case == "mask_source":
        stage["required_views"]["mask_source"] = "unbound_mask"
    elif case == "tuning":
        stage["artifixer3d_steps"] = 30_001

    with pytest.raises(
        TaskEvaluationSceneConfigurationStageConfigurationError,
        match=(
            "scene_configuration_stage_configuration_preflight_failed:"
            "stage-1:observed_appearance_object_removal:"
            f"{predicate}"
        ),
    ):
        validate_immutable_stage_configurations(
            envelope=_envelope(), configurations=configurations
        )


@pytest.mark.parametrize("target", ["", "Root/Target", "/", "/Root//Target"])
def test_stage_two_refuses_invalid_exact_target_syntax_before_provider_mutation(
    target: str,
) -> None:
    configurations = _configuration_map()
    configurations["stage-2"]["exact_target_prim"] = target

    with pytest.raises(
        TaskEvaluationSceneConfigurationStageConfigurationError,
        match=(
            "scene_configuration_stage_configuration_preflight_failed:"
            "stage-2:collision_object_excision:exact_target_prim"
        ),
    ):
        validate_immutable_stage_configurations(
            envelope=_envelope(), configurations=configurations
        )


def test_stage_two_refuses_invalid_removal_id_before_provider_mutation() -> None:
    envelope = _envelope()
    envelope["recipe"]["subject_identity"]["id"] = "scene 839873 replacement"

    with pytest.raises(
        TaskEvaluationSceneConfigurationStageConfigurationError,
        match=(
            "scene_configuration_stage_configuration_preflight_failed:"
            "stage-2:collision_object_excision:recipe.subject_identity.id"
        ),
    ):
        validate_immutable_stage_configurations(
            envelope=envelope, configurations=_configuration_map()
        )
