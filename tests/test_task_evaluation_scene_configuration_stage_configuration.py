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
            "source_object": {"publisher_instance_id": "104"},
            "production_render_required": True,
            "required_views": {"minimum": 8},
            "provider_disclosure": {"raw_interiorgs_bytes": False},
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
