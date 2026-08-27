"""Pure no-spend validation for immutable scene-configuration stage inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

from .task_evaluation_scene_configuration_disclosure import (
    renders_on_provider,
    stage_requests_upload,
)


class TaskEvaluationSceneConfigurationStageConfigurationError(ValueError):
    """One immutable stage configuration would deterministically refuse later."""


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _versioned_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(str(value.get("id") or ""))
        and bool(str(value.get("version") or ""))
    )


def _bounded_pair(value: Any, *, positive_lower: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(lower)
        and math.isfinite(upper)
        and lower <= upper
        and lower >= 0.0
        and (not positive_lower or lower > 0.0)
        and (positive_lower or upper <= 1.0)
    )


def _metric_envelope_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    minimum = value.get("minimum_xyz_m")
    maximum = value.get("maximum_xyz_m")
    tolerance = value.get("maximum_dimension_relative_error")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0.0 <= float(tolerance) <= 1.0
    ):
        return False
    try:
        lower = [float(item) for item in minimum]
        upper = [float(item) for item in maximum]
    except (TypeError, ValueError):
        return False
    return all(
        math.isfinite(lower[index])
        and math.isfinite(upper[index])
        and upper[index] > lower[index]
        for index in range(3)
    )


def native_import_checks_valid(value: Any) -> bool:
    """Match the exact native-import driver and adapter check set."""

    if not isinstance(value, Mapping) or set(value) != {
        "stage_import",
        "rigid_body_enabled",
        "collider_enabled",
        "gravity_settle_seconds",
        "maximum_settle_translation_m",
        "maximum_settle_rotation_rad",
        "support_contact_required",
        "explosion_or_tunneling_forbidden",
        "deterministic_reset_required",
        "state_digest_repeat_count",
    }:
        return False
    return (
        value["stage_import"] is True
        and value["rigid_body_enabled"] is True
        and value["collider_enabled"] is True
        and _positive_finite(value["gravity_settle_seconds"])
        and _positive_finite(value["maximum_settle_translation_m"])
        and _positive_finite(value["maximum_settle_rotation_rad"])
        and value["support_contact_required"] is True
        and value["explosion_or_tunneling_forbidden"] is True
        and value["deterministic_reset_required"] is True
        and value["state_digest_repeat_count"] == 3
    )


def static_qualification_checks_valid(value: Any) -> bool:
    """Match the exact stage-4 adapter check set."""

    expected = {
        "usd_parses",
        "meters_per_unit",
        "up_axis",
        "single_movable_rigid_root",
        "collision_geometry_present",
        "collision_geometry_nonempty_and_finite",
        "mass_and_inertia_positive_finite",
        "materials_within_preregistered_bounds",
        "no_external_unpinned_dependencies",
        "no_articulation",
        "no_scripts_or_credentials",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    return (
        value["usd_parses"] is True
        and value["meters_per_unit"] == 1.0
        and value["up_axis"] == "Z"
        and all(
            value[name] is True
            for name in expected - {"usd_parses", "meters_per_unit", "up_axis"}
        )
    )


def _stage_one_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    source_object = configuration.get("source_object")
    required_views = configuration.get("required_views")
    human = configuration.get("human_authority")
    minimum = required_views.get("minimum") if isinstance(required_views, Mapping) else None
    tuning = (
        configuration.get("transition_radius_pixels"),
        configuration.get("artifixer3d_steps"),
        configuration.get("random_seed"),
    )
    resolved_tuning = (
        3 if tuning[0] is None else tuning[0],
        30_000 if tuning[1] is None else tuning[1],
        839_873 if tuning[2] is None else tuning[2],
    )
    if (
        configuration.get("schema_version")
        != "observed_appearance_object_removal_configuration.v1"
    ):
        return "schema_version"
    if (
        not isinstance(source_object, Mapping)
        or not isinstance(source_object.get("publisher_instance_id"), str)
        or not source_object["publisher_instance_id"].strip()
    ):
        return "source_object.publisher_instance_id"
    if configuration.get("production_render_required") is not True:
        return "production_render_required"
    if stage_requests_upload(configuration) is not renders_on_provider(
            (envelope.get("render_inputs_result") or {}).get("disclosure_decision")
            or {}
    ):
        return "provider_disclosure"
    if (
        configuration.get("output_requirements", {}).get(
            "generated_pixels_labeled"
        )
        is not True
    ):
        return "output_requirements.generated_pixels_labeled"
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        return "required_views.minimum"
    if (
        not isinstance(human, Mapping)
        or not bool(str(human.get("accepted_by") or "").strip())
        or not bool(str(human.get("accepted_on") or "").strip())
        or not bool(str(human.get("authority_reference") or "").strip())
        or human.get("private_derived_frame_disclosure_authorized") is not True
        or human.get("provider_retention_terms_accepted") is not True
        or human.get("provider_training_terms_accepted") is not True
        or human.get("provider_training_authorized") is not False
    ):
        return "human_authority"
    if (
        not isinstance(resolved_tuning[0], int)
        or isinstance(resolved_tuning[0], bool)
        or resolved_tuning[0] < 0
        or not isinstance(resolved_tuning[1], int)
        or isinstance(resolved_tuning[1], bool)
        or not 1 <= resolved_tuning[1] <= 30_000
        or not isinstance(resolved_tuning[2], int)
        or isinstance(resolved_tuning[2], bool)
    ):
        return "artifixer_tuning"
    return None


def _stage_two_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    collision_rows = [
        row
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping)
        and row.get("contract_path") == "scene.geometry.collision"
    ]
    validation = configuration.get("validation")
    if configuration.get("schema_version") != "collision_object_excision_configuration.v1":
        return "schema_version"
    if configuration.get("operation") != "deactivate_exact_prim_only":
        return "operation"
    if (
        len(collision_rows) != 1
        or configuration.get("collision_source_digest")
        != collision_rows[0].get("digest")
    ):
        return "collision_source_digest"
    if (
        not isinstance(configuration.get("exact_target_prim"), str)
        or not str(configuration["exact_target_prim"]).startswith("/")
    ):
        return "exact_target_prim"
    if not isinstance(configuration.get("expected_target"), Mapping):
        return "expected_target"
    if (
        not isinstance(validation, Mapping)
        or validation.get("target_absent_after_excision") is not True
        or validation.get("all_non_target_prim_digests_unchanged") is not True
        or validation.get("stage_units_and_up_axis_unchanged") is not True
        or validation.get("before_and_after_prim_manifests_required") is not True
    ):
        return "validation"
    return None


def _stage_three_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    identity = configuration.get("replacement_identity")
    required = configuration.get("required_output")
    if (
        configuration.get("schema_version")
        != "rigid_replacement_authoring_configuration.v1"
    ):
        return "schema_version"
    if (
        not _versioned_identity(identity)
        or identity != (envelope.get("recipe") or {}).get("subject_identity")
    ):
        return "replacement_identity"
    if not _metric_envelope_valid(configuration.get("metric_envelope")):
        return "metric_envelope"
    if not isinstance(required, Mapping):
        return "required_output"
    static_bounds = required.get("static_friction_bounds")
    dynamic_bounds = required.get("dynamic_friction_bounds")
    if (
        required.get("format") != "OpenUSD"
        or required.get("rigid_body") is not True
        or required.get("single_movable_root") is not True
        or required.get("units") != "meters"
        or required.get("up_axis") != "Z"
    ):
        return "required_output"
    if (
        not _bounded_pair(required.get("mass_kg_bounds"), positive_lower=True)
        or not _bounded_pair(static_bounds)
        or not _bounded_pair(dynamic_bounds)
        or not _bounded_pair(required.get("restitution_bounds"))
    ):
        return "required_output.physics_bounds"
    if float(dynamic_bounds[0]) > float(static_bounds[1]):
        return "required_output.friction_bounds_feasible"
    if configuration.get("physics_authority_granted_by_authoring") is not False:
        return "physics_authority_granted_by_authoring"
    return None


def _stage_four_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    identity = configuration.get("replacement_identity")
    if (
        configuration.get("schema_version")
        != "replacement_static_qualification_configuration.v1"
    ):
        return "schema_version"
    if (
        not _versioned_identity(identity)
        or identity != (envelope.get("recipe") or {}).get("subject_identity")
    ):
        return "replacement_identity"
    if not static_qualification_checks_valid(configuration.get("required_checks")):
        return "required_checks"
    if (
        configuration.get("center_of_mass_must_lie_inside_collision_bounds")
        is not True
    ):
        return "center_of_mass_must_lie_inside_collision_bounds"
    return None


def _stage_five_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    identity = configuration.get("replacement_identity")
    if (
        configuration.get("schema_version")
        != "replacement_native_import_qualification_configuration.v1"
    ):
        return "schema_version"
    if (
        not _versioned_identity(identity)
        or identity != (envelope.get("recipe") or {}).get("subject_identity")
    ):
        return "replacement_identity"
    if not native_import_checks_valid(configuration.get("required_checks")):
        return "required_checks"
    return None


def _stage_six_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    replacement = configuration.get("replacement")
    robot_mount = configuration.get("robot_mount_interface")
    if (
        configuration.get("schema_version")
        != "task_evaluation_scene_assembly_configuration.v1"
    ):
        return "schema_version"
    if configuration.get("scene_identity") != (envelope.get("recipe") or {}).get(
        "scene_identity"
    ):
        return "scene_identity"
    if not isinstance(replacement, Mapping):
        return "replacement"
    if replacement.get("qualified_asset_from_stage") != "stage-5":
        return "replacement.qualified_asset_from_stage"
    if (
        replacement.get("source_and_replacement_visual_instances_must_not_coexist")
        is not True
        or replacement.get(
            "source_and_replacement_collision_instances_must_not_coexist"
        )
        is not True
    ):
        return "replacement.non_coexistence"
    if (
        not isinstance(robot_mount, Mapping)
        or robot_mount.get("publish_robot_neutral_scene_mount_frame") is not True
        or robot_mount.get(
            "robot_specific_base_transform_and_reachability_deferred_to_each_evaluation"
        )
        is not True
    ):
        return "robot_mount_interface"
    if configuration.get("evaluation_episode_executed_in_this_run") is not False:
        return "evaluation_episode_executed_in_this_run"
    if configuration.get("scene_construction_repeated_per_evaluation") is not False:
        return "scene_construction_repeated_per_evaluation"
    return None


_VALIDATORS: dict[
    str, Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]
] = {
    "observed_appearance_object_removal": _stage_one_refusal,
    "collision_object_excision": _stage_two_refusal,
    "rigid_replacement_authoring": _stage_three_refusal,
    "replacement_static_qualification": _stage_four_refusal,
    "replacement_native_import_qualification": _stage_five_refusal,
    "scene_assembly": _stage_six_refusal,
}


def validate_immutable_stage_configurations(
    *,
    envelope: Mapping[str, Any],
    configurations: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject every known deterministic config refusal before bundle creation."""

    for stage in (envelope.get("recipe") or {}).get("stage_sequence") or []:
        stage_id = str(stage.get("stage_id") or "")
        capability = str(stage.get("capability") or "")
        validator = _VALIDATORS.get(capability)
        if validator is None:
            continue
        configuration = configurations.get(stage_id)
        predicate = (
            "configuration"
            if not isinstance(configuration, Mapping)
            else validator(configuration, envelope)
        )
        if predicate is not None:
            raise TaskEvaluationSceneConfigurationStageConfigurationError(
                "scene_configuration_stage_configuration_preflight_failed:"
                f"{stage_id}:{capability}:{predicate}"
            )


__all__ = [
    "TaskEvaluationSceneConfigurationStageConfigurationError",
    "native_import_checks_valid",
    "static_qualification_checks_valid",
    "validate_immutable_stage_configurations",
]
