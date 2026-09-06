"""Pure no-spend validation for immutable scene-configuration stage inputs."""

from __future__ import annotations

import math
import re
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


def _metric_bounds_valid(minimum: Any, maximum: Any) -> bool:
    """Match the camera-ring bounds consumed before the first paid stage."""

    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
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


SAM31_SELECTION_RULE = (
    "union_across_repetitions_of_any_view_target_core_plus_uncertain_"
    "contribution_at_frozen_threshold.v1"
)
SAM31_MASK_SOURCE = "sam31_reviewed_calibrated_object_masks"
SAM31_EVIDENCE_FIELDS = (
    "selection_inputs", "track_selection_review", "calibrated_mask_set",
    "segment_cutout_set", "standard_splat_conversion",
)


def stage_one_gaussian_inputs_refusal(configuration: Mapping[str, Any]) -> str | None:
    """Select exactly one explicit removal method; never substitute AABB for SAM."""
    cutout = configuration.get("gaussian_cutout")
    views = configuration.get("required_views")
    if not isinstance(cutout, Mapping) or not isinstance(views, Mapping):
        return "gaussian_cutout"
    if cutout.get("retained_rows_must_remain_byte_exact") is not True:
        return "gaussian_cutout"
    rule = cutout.get("selection_rule")
    if rule == "gaussian_center_inside_registered_source_object_aabb":
        if views.get("mask_source") != "registered_source_object_bounds_projection":
            return "required_views.minimum"
        padding = cutout.get("aabb_padding_m")
        if (type(padding) not in (int, float) or not math.isfinite(padding)
                or not 0 <= padding <= 0.10):
            return "gaussian_cutout"
    elif rule == SAM31_SELECTION_RULE:
        evidence = configuration.get("sam31_exact_mask_evidence")
        plan = configuration.get("sam31_preparation_plan")
        digests_valid = (
            isinstance(evidence, Mapping) and plan is None
            and set(evidence) == {key + "_digest" for key in SAM31_EVIDENCE_FIELDS}
            and all(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                    for value in evidence.values())
        )
        plan_valid = (
            evidence is None and isinstance(plan, Mapping)
            and set(plan) == {"uri", "digest", "size_bytes"}
            and isinstance(plan.get("uri"), str) and bool(plan["uri"])
            and isinstance(plan.get("digest"), str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", plan["digest"])
            and type(plan.get("size_bytes")) is int and plan["size_bytes"] > 0
        )
        if ("aabb_padding_m" in cutout or views.get("mask_source") != SAM31_MASK_SOURCE
                or configuration.get("sam31_review_kind") not in {"ai", "human"}
                or not (digests_valid or plan_valid)):
            return "sam31_exact_mask_evidence"
    else:
        return "gaussian_cutout"
    return None


def _stage_one_refusal(
    configuration: Mapping[str, Any], envelope: Mapping[str, Any]
) -> str | None:
    if configuration.get("schema_version") == "task_evaluation_provided_mesh_appearance_excision.v1":
        from .task_evaluation_completed_scene_adapters import mesh_appearance_configuration_refusal
        return mesh_appearance_configuration_refusal(configuration, envelope)
    source_object = configuration.get("source_object")
    required_views = configuration.get("required_views")
    disclosure = configuration.get("provider_disclosure")
    output_requirements = configuration.get("output_requirements")
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
    if not _metric_bounds_valid(
        source_object.get("aabb_min_xyz_m"),
        source_object.get("aabb_max_xyz_m"),
    ):
        return "source_object.aabb"
    if configuration.get("production_render_required") is not True:
        return "production_render_required"
    gaussian_refusal = stage_one_gaussian_inputs_refusal(configuration)
    if gaussian_refusal:
        return gaussian_refusal
    disclosure_intent = None
    if isinstance(disclosure, Mapping):
        for key in ("source_appearance_bytes", "raw_interiorgs_bytes"):
            if key in disclosure:
                disclosure_intent = disclosure[key]
                break
    if (
        not isinstance(disclosure, Mapping)
        or not isinstance(disclosure_intent, bool)
        or disclosure.get("derived_rendered_views") is not True
    ):
        return "provider_disclosure"
    if stage_requests_upload(configuration) is not renders_on_provider(
            (envelope.get("render_inputs_result") or {}).get("disclosure_decision")
            or {}
    ):
        return "provider_disclosure"
    if (
        not isinstance(output_requirements, Mapping)
        or output_requirements.get("generated_pixels_labeled") is not True
    ):
        return "output_requirements.generated_pixels_labeled"
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not 1 <= minimum <= (
            16 if (configuration.get("gaussian_cutout") or {}).get("selection_rule")
            == SAM31_SELECTION_RULE else 8)
        or required_views.get("lossless_inputs") is not True
    ):
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
    target = configuration.get("exact_target_prim")
    if (
        not isinstance(target, str)
        or not target.startswith("/")
        or target == "/"
        or "//" in target
    ):
        return "exact_target_prim"
    removal_id = str(
        ((envelope.get("recipe") or {}).get("subject_identity") or {}).get("id")
        or ""
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", removal_id) is None:
        return "recipe.subject_identity.id"
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
