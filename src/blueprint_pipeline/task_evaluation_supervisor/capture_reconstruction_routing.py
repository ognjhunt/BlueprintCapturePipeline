"""Deterministic capture-profile routing for supervisor reconstruction proposals.

The route is a planning observation, not an execution authorization or a
reconstruction result.  Capture metadata selects only a compatible sequence of
method *kinds*; registered adapters, evidence qualification, and claim ceilings
remain independent deterministic gates.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ..capture_intake import CAPTURE_AUTHORITY_PROFILES
from ..decision_evidence_contracts import canonical_digest
from .capture_ingress import validate_capture_build_ingress


CAPTURE_RECONSTRUCTION_ROUTE_SCHEMA_VERSION = "task_evaluation_capture_reconstruction_route.v1"

_PROFILE_KEYS = ("capture_authority_profile", "capture_modality")
_PROFILE_ALIASES = {
    "iphone_video_only": "monocular_video",
    "android_video_only": "monocular_video",
    "glasses_video_only": "monocular_video",
    "glasses_pov": "monocular_video",
}
_VIDEO_PROFILES = {
    "iphone_arkit_lidar",
    "iphone_arkit_non_lidar",
    "camera_360_equirectangular",
    "camera_360_native",
    "monocular_video",
}
_STRICT_360_PROFILES = {"camera_360_equirectangular", "camera_360_native"}
_STRICT_ARKIT_PROFILES = {"iphone_arkit_lidar", "iphone_arkit_non_lidar"}
_PROFILE_VALIDATION_PATHS = {
    "evaluation_prep/capture_profile_validation.json",
    "pipeline/evaluation_prep/capture_profile_validation.json",
}

_PROFILE_STAGES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "iphone_arkit_lidar": (
        ("verify_arkit_raw_contract", "capture_validation", "required_deterministic_gate"),
        (
            "compile_frozen_frame_dataset",
            "capture_reconstruction_dataset_compilation",
            "registered_conditional",
        ),
        ("compile_arkit_metric_scaffold", "lidar_depth_fusion", "registered_conditional"),
        (
            "compile_arkit_observed_surface",
            "observed_depth_surface",
            "registered_conditional",
        ),
        (
            "export_arkit_reconstruction_dataset",
            "calibrated_dataset_export",
            "registered_conditional",
        ),
        (
            "run_pose_refinement",
            "pose_refinement",
            "registered_conditional",
        ),
        ("validate_metric_scale", "metric_scale", "registered_conditional"),
        ("train_gaussian_reconstruction", "gaussian_splat_3d", "registered_conditional"),
        (
            "evaluate_heldout_appearance",
            "independent_appearance_qa",
            "registered_conditional",
        ),
        ("compile_metric_geometry", "metric_geometry", "registered_conditional"),
        ("compile_collision_candidate", "collision_geometry", "registered_conditional"),
        ("qualify_collision_candidate", "collision_qa", "registered_conditional"),
        ("package_nurec_openusd", "openusd_packaging", "registered_conditional"),
        ("verify_isaac_asset", "isaac_verification", "registered_conditional"),
        ("generate_reconstruction_report", "terminal_reporting", "registered_conditional"),
    ),
    "iphone_arkit_non_lidar": (
        ("verify_arkit_pose_intrinsics", "capture_validation", "required_deterministic_gate"),
        (
            "compile_frozen_frame_dataset",
            "capture_reconstruction_dataset_compilation",
            "registered_conditional",
        ),
        (
            "export_arkit_reconstruction_dataset",
            "calibrated_dataset_export",
            "registered_conditional",
        ),
        ("run_pose_refinement", "pose_refinement", "registered_conditional"),
        ("validate_metric_scale", "metric_scaffold", "registered_conditional"),
        ("train_gaussian_reconstruction", "gaussian_splat_3d", "registered_conditional"),
        (
            "evaluate_heldout_appearance",
            "independent_appearance_qa",
            "registered_conditional",
        ),
        ("compile_metric_geometry", "metric_geometry", "registered_conditional"),
        ("compile_collision_candidate", "collision_geometry", "registered_conditional"),
        ("qualify_collision_candidate", "collision_qa", "registered_conditional"),
        ("package_nurec_openusd", "openusd_packaging", "registered_conditional"),
        ("verify_isaac_asset", "isaac_verification", "registered_conditional"),
        ("generate_reconstruction_report", "terminal_reporting", "registered_conditional"),
    ),
    "camera_360_equirectangular": (
        ("verify_equirectangular_metadata", "capture_validation", "required_deterministic_gate"),
        (
            "compile_frozen_frame_dataset",
            "capture_reconstruction_dataset_compilation",
            "registered_conditional",
        ),
        (
            "compile_equirectangular_virtual_rig",
            "equirectangular_normalization",
            "registered_conditional",
        ),
        ("run_pose_estimation", "pose_sfm_estimation", "registered_conditional"),
        ("validate_metric_scale", "metric_scaffold", "registered_conditional"),
        ("train_gaussian_reconstruction", "gaussian_splat_3d", "registered_conditional"),
        (
            "evaluate_heldout_appearance",
            "independent_appearance_qa",
            "registered_conditional",
        ),
        ("compile_metric_geometry", "metric_geometry", "registered_conditional"),
        ("compile_collision_candidate", "collision_geometry", "registered_conditional"),
        ("qualify_collision_candidate", "collision_qa", "registered_conditional"),
        ("package_nurec_openusd", "openusd_packaging", "registered_conditional"),
        ("verify_isaac_asset", "isaac_verification", "registered_conditional"),
        ("generate_reconstruction_report", "terminal_reporting", "registered_conditional"),
    ),
    "camera_360_native": (
        ("retain_native_360_originals", "capture_validation", "required_deterministic_gate"),
        (
            "normalize_native_360_capture",
            "native_360_normalization",
            "registered_conditional",
        ),
        (
            "compile_frozen_frame_dataset",
            "capture_reconstruction_dataset_compilation",
            "registered_conditional",
        ),
        ("validate_camera_rig", "camera_rig_validation", "registered_conditional"),
        (
            "compile_equirectangular_virtual_rig",
            "equirectangular_normalization",
            "registered_conditional",
        ),
        ("run_pose_estimation", "pose_sfm_estimation", "registered_conditional"),
        ("validate_metric_scale", "metric_scaffold", "registered_conditional"),
        ("train_gaussian_reconstruction", "gaussian_splat_3d", "registered_conditional"),
        (
            "evaluate_heldout_appearance",
            "independent_appearance_qa",
            "registered_conditional",
        ),
        ("compile_metric_geometry", "metric_geometry", "registered_conditional"),
        ("compile_collision_candidate", "collision_geometry", "registered_conditional"),
        ("qualify_collision_candidate", "collision_qa", "registered_conditional"),
        ("package_nurec_openusd", "openusd_packaging", "registered_conditional"),
        ("verify_isaac_asset", "isaac_verification", "registered_conditional"),
        ("generate_reconstruction_report", "terminal_reporting", "registered_conditional"),
    ),
    "monocular_video": (
        (
            "compile_frozen_frame_dataset",
            "capture_reconstruction_dataset_compilation",
            "registered_conditional",
        ),
        ("run_pose_estimation", "pose_sfm_estimation", "registered_conditional"),
        ("validate_metric_scale", "metric_scaffold", "registered_conditional"),
        ("train_gaussian_reconstruction", "gaussian_splat_3d", "registered_conditional"),
        (
            "evaluate_heldout_appearance",
            "independent_appearance_qa",
            "registered_conditional",
        ),
        ("compile_metric_geometry", "metric_geometry", "registered_conditional"),
        ("compile_collision_candidate", "collision_geometry", "registered_conditional"),
        ("qualify_collision_candidate", "collision_qa", "registered_conditional"),
        ("package_nurec_openusd", "openusd_packaging", "registered_conditional"),
        ("verify_isaac_asset", "isaac_verification", "registered_conditional"),
        ("generate_reconstruction_report", "terminal_reporting", "registered_conditional"),
    ),
    "precomputed_external_reconstruction": (
        ("verify_source_capture_binding", "capture_validation", "required_deterministic_gate"),
        (
            "import_external_reconstruction",
            "external_reconstruction_import",
            "registered_conditional",
        ),
        ("validate_metric_scale", "metric_scale", "registered_conditional"),
        ("compile_metric_geometry", "metric_geometry", "registered_conditional"),
        ("compile_collision_candidate", "collision_geometry", "registered_conditional"),
        ("qualify_collision_candidate", "collision_qa", "registered_conditional"),
        ("package_nurec_openusd", "openusd_packaging", "registered_conditional"),
        ("verify_isaac_asset", "isaac_verification", "registered_conditional"),
        ("generate_reconstruction_report", "terminal_reporting", "registered_conditional"),
    ),
}


class CaptureReconstructionRouteError(ValueError):
    """Raised when a route artifact violates its deterministic contract."""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _declared_profiles(capture_build: Mapping[str, Any]) -> tuple[str, ...]:
    profiles: set[str] = set()
    for artifact in capture_build.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        projection = artifact.get("approved_projection")
        if not isinstance(projection, Mapping):
            continue
        for key in _PROFILE_KEYS:
            raw = str(projection.get(key) or "").strip().lower()
            normalized = _PROFILE_ALIASES.get(raw, raw)
            if normalized in CAPTURE_AUTHORITY_PROFILES:
                profiles.add(normalized)
    return tuple(sorted(profiles))


def _capture_digest_candidates(capture_build: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    for artifact in capture_build.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        projection = artifact.get("approved_projection")
        if not isinstance(projection, Mapping):
            continue
        value = str(projection.get("capture_digest") or "").strip()
        if value:
            values.add(value)
    return tuple(sorted(values))


def _profile_validation_projection(
    capture_build: Mapping[str, Any], *, declared_profile: str
) -> tuple[dict[str, Any] | None, str | None]:
    projections: list[dict[str, Any]] = []
    for artifact in capture_build.get("artifacts", []):
        if not isinstance(artifact, Mapping) or artifact.get("relative_path") not in (
            _PROFILE_VALIDATION_PATHS
        ):
            continue
        projection = artifact.get("approved_projection")
        if not isinstance(projection, Mapping):
            return None, "invalid"
        value = dict(projection)
        required = {
            "schema_version",
            "source_capture_digest",
            "declared_capture_authority_profile",
            "compatible_capture_authority_profile",
            "validation_status",
            "probe_receipt_digests",
            "probe_source_file_digests",
            "observed_processing_lanes",
            "native_normalization_digest",
            "blockers",
            "agent_selected_capture_profile",
            "agent_may_change_capture_profile",
            "proof_effect",
            "claim_ceiling",
            "legal_next_actions",
            "capture_profile_routing_binding_digest",
            "capture_profile_validation_digest",
        }
        if not required.issubset(value):
            return None, "invalid"
        routing_binding = {key: value[key] for key in required if key not in {
            "capture_profile_routing_binding_digest",
            "capture_profile_validation_digest",
        }}
        if (
            value.get("schema_version") != "capture_profile_validation.v1"
            or value.get("capture_profile_routing_binding_digest")
            != canonical_digest(routing_binding)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(value.get("capture_profile_validation_digest") or ""),
            )
            is None
            or value.get("agent_selected_capture_profile") is not False
            or value.get("agent_may_change_capture_profile") is not False
            or value.get("claim_ceiling") != "capture_profile_compatibility"
        ):
            return None, "invalid"
        projections.append(value)
    if not projections:
        return None, None
    unique = {
        str(value["capture_profile_routing_binding_digest"]): value
        for value in projections
    }
    if len(unique) != 1:
        return None, "invalid"
    projection = next(iter(unique.values()))
    capture_digests = _capture_digest_candidates(capture_build)
    if (
        len(capture_digests) != 1
        or projection.get("source_capture_digest") != capture_digests[0]
        or projection.get("declared_capture_authority_profile") != declared_profile
    ):
        return None, "invalid"
    return projection, None


def _required_representations(claim_types: Sequence[str]) -> list[str]:
    claims = {str(item).strip() for item in claim_types if str(item).strip()}
    required = {"appearance_layer"}
    if claims.intersection({"reachability", "robot_placement", "navigation_clearance"}):
        required.add("metric_reference_layer")
    if claims.intersection(
        {
            "collision_contact",
            "grasp_contact",
            "articulation",
            "containment",
            "mass_inertia",
            "friction_compliance",
            "object_state_transition",
        }
    ):
        required.update({"metric_reference_layer", "collision_geometry", "physics_layer"})
    if "articulation" in claims:
        required.add("articulated_object_asset")
    return sorted(required)


def build_capture_reconstruction_route(
    capture_build_value: Mapping[str, Any],
    *,
    requested_claim_types: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a digest-bound, profile-specific route proposal for one capture build."""

    capture_build = validate_capture_build_ingress(capture_build_value)
    profiles = _declared_profiles(capture_build)
    blockers: list[str] = []
    profile_validation_status = "not_applicable_to_profile"
    profile_validation_digest: str | None = None
    if not profiles:
        status = "capture_profile_required"
        profile: str | None = None
        blockers.append("validated_capture_authority_profile_missing")
    elif len(profiles) > 1:
        status = "ambiguous_capture_profile"
        profile = None
        blockers.append("conflicting_capture_authority_profiles")
    else:
        status = "route_proposed"
        profile = profiles[0]
        if profile in _STRICT_360_PROFILES:
            validation, validation_error = _profile_validation_projection(
                capture_build,
                declared_profile=profile,
            )
            if validation_error is not None:
                status = "capture_profile_validation_invalid"
                profile = None
                profile_validation_status = "invalid"
                blockers.append("deterministic_capture_profile_validation_invalid")
            elif validation is None:
                status = "capture_profile_validation_required"
                profile = None
                profile_validation_status = "required_missing"
                blockers.append("deterministic_capture_profile_validation_missing")
            else:
                profile_validation_digest = str(
                    validation["capture_profile_validation_digest"]
                )
                if (
                    validation.get("validation_status") != "validated"
                    or validation.get("compatible_capture_authority_profile")
                    != profiles[0]
                    or validation.get("blockers") != []
                    or validation.get("proof_effect")
                    != "capture_profile_validation_only"
                ):
                    status = "capture_profile_validation_failed"
                    profile = None
                    profile_validation_status = "blocked"
                    blockers.append("deterministic_capture_profile_validation_failed")
                else:
                    profile_validation_status = "validated"
        elif profile in _STRICT_ARKIT_PROFILES:
            profile_validation_status = "required_raw_contract_gate"

    stages = [
        {
            "ordinal": ordinal,
            "stage_id": stage_id,
            "method_kind": method_kind,
            "implementation_status": implementation_status,
        }
        for ordinal, (stage_id, method_kind, implementation_status) in enumerate(
            _PROFILE_STAGES.get(profile or "", ())
        )
    ]
    executable_adapters = (
        ["local://arkit-metric-scaffold-v1", "local://decoded-observation-index-v1"]
        if profile == "iphone_arkit_lidar"
        else [
            "local://decoded-observation-index-v1",
            "local://equirectangular-virtual-rig-v1",
            "local://native-360-normalization-v1",
        ]
        if profile == "camera_360_native"
        else [
            "local://decoded-observation-index-v1",
            "local://equirectangular-virtual-rig-v1",
        ]
        if profile == "camera_360_equirectangular"
        else ["local://decoded-observation-index-v1"]
        if profile in _VIDEO_PROFILES
        else ["local://external-reconstruction-import-v1"]
        if profile == "precomputed_external_reconstruction"
        else []
    )
    route = {
        "schema_version": CAPTURE_RECONSTRUCTION_ROUTE_SCHEMA_VERSION,
        "capture_build_digest": capture_build["capture_build_digest"],
        "status": status,
        "capture_authority_profile": profile,
        "declared_profile_candidates": list(profiles),
        "capture_profile_validation_status": profile_validation_status,
        "capture_profile_validation_digest": profile_validation_digest,
        "requested_claim_types": _strings(list(requested_claim_types)),
        "required_representations": _required_representations(requested_claim_types),
        "stages": stages,
        "currently_registered_adapters": sorted(executable_adapters),
        "blockers": blockers,
        "next_legal_action": (
            "compile_profile_specific_reconstruction_plan"
            if status == "route_proposed"
            else "request_deterministic_capture_profile_validation"
            if status == "capture_profile_validation_required"
            else "request_corrected_capture_intake"
            if status == "capture_profile_validation_failed"
            else "preserve_evidence_and_stop"
            if status == "capture_profile_validation_invalid"
            else "request_validated_capture_profile"
        ),
        "agent_selected_capture_profile": False,
        "execution_authorized_by_route": False,
        "route_is_reconstruction_evidence": False,
        "appearance_layer_is_metric_or_physics_truth": False,
        "physical_success_established": False,
        "proof_effect": "none",
    }
    route["capture_reconstruction_route_digest"] = canonical_digest(
        route,
        digest_field="capture_reconstruction_route_digest",
    )
    return validate_capture_reconstruction_route(route)


def validate_capture_reconstruction_route(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "capture_build_digest",
        "status",
        "capture_authority_profile",
        "declared_profile_candidates",
        "capture_profile_validation_status",
        "capture_profile_validation_digest",
        "requested_claim_types",
        "required_representations",
        "stages",
        "currently_registered_adapters",
        "blockers",
        "next_legal_action",
        "agent_selected_capture_profile",
        "execution_authorized_by_route",
        "route_is_reconstruction_evidence",
        "appearance_layer_is_metric_or_physics_truth",
        "physical_success_established",
        "proof_effect",
        "capture_reconstruction_route_digest",
    }
    try:
        route = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise CaptureReconstructionRouteError("capture_reconstruction_route_not_json") from exc
    if not isinstance(route, dict) or set(route) != required:
        raise CaptureReconstructionRouteError("capture_reconstruction_route_fields_invalid")
    digest = str(route.get("capture_build_digest") or "")
    expected_digest = canonical_digest(
        route,
        digest_field="capture_reconstruction_route_digest",
    )
    status = str(route.get("status") or "")
    profile = route.get("capture_authority_profile")
    candidates = _strings(route.get("declared_profile_candidates"))
    stages = route.get("stages")
    claims = _strings(route.get("requested_claim_types"))
    representations = _strings(route.get("required_representations"))
    adapters = _strings(route.get("currently_registered_adapters"))
    blockers = _strings(route.get("blockers"))
    if (
        route.get("schema_version") != CAPTURE_RECONSTRUCTION_ROUTE_SCHEMA_VERSION
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or route.get("capture_reconstruction_route_digest") != expected_digest
        or status
        not in {
            "route_proposed",
            "capture_profile_required",
            "ambiguous_capture_profile",
            "capture_profile_validation_required",
            "capture_profile_validation_failed",
            "capture_profile_validation_invalid",
        }
        or not isinstance(stages, list)
        or not all(isinstance(row, Mapping) for row in stages)
        or route.get("declared_profile_candidates") != candidates
        or any(candidate not in CAPTURE_AUTHORITY_PROFILES for candidate in candidates)
        or route.get("requested_claim_types") != claims
        or route.get("required_representations") != representations
        or representations != _required_representations(claims)
        or route.get("currently_registered_adapters") != adapters
        or route.get("blockers") != blockers
        or route.get("agent_selected_capture_profile") is not False
        or route.get("execution_authorized_by_route") is not False
        or route.get("route_is_reconstruction_evidence") is not False
        or route.get("appearance_layer_is_metric_or_physics_truth") is not False
        or route.get("physical_success_established") is not False
        or route.get("proof_effect") != "none"
    ):
        raise CaptureReconstructionRouteError("capture_reconstruction_route_contract_invalid")
    if status == "route_proposed":
        expected_stages = [
            {
                "ordinal": ordinal,
                "stage_id": stage_id,
                "method_kind": method_kind,
                "implementation_status": implementation_status,
            }
            for ordinal, (stage_id, method_kind, implementation_status) in enumerate(
                _PROFILE_STAGES.get(str(profile or ""), ())
            )
        ]
        expected_adapters = (
            ["local://arkit-metric-scaffold-v1", "local://decoded-observation-index-v1"]
            if profile == "iphone_arkit_lidar"
            else [
                "local://decoded-observation-index-v1",
                "local://equirectangular-virtual-rig-v1",
                "local://native-360-normalization-v1",
            ]
            if profile == "camera_360_native"
            else [
                "local://decoded-observation-index-v1",
                "local://equirectangular-virtual-rig-v1",
            ]
            if profile == "camera_360_equirectangular"
            else ["local://decoded-observation-index-v1"]
            if profile in _VIDEO_PROFILES
            else ["local://external-reconstruction-import-v1"]
            if profile == "precomputed_external_reconstruction"
            else []
        )
        if (
            profile not in CAPTURE_AUTHORITY_PROFILES
            or candidates != [profile]
            or stages != expected_stages
            or adapters != sorted(expected_adapters)
            or blockers
            or route.get("next_legal_action") != "compile_profile_specific_reconstruction_plan"
            or (
                profile in _STRICT_360_PROFILES
                and (
                    route.get("capture_profile_validation_status") != "validated"
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(route.get("capture_profile_validation_digest") or ""),
                    )
                    is None
                )
            )
            or (
                profile not in _STRICT_360_PROFILES
                and (
                    route.get("capture_profile_validation_status")
                    != (
                        "required_raw_contract_gate"
                        if profile in _STRICT_ARKIT_PROFILES
                        else "not_applicable_to_profile"
                    )
                    or route.get("capture_profile_validation_digest") is not None
                )
            )
        ):
            raise CaptureReconstructionRouteError("capture_reconstruction_route_profile_invalid")
    else:
        expected_blocker = (
            ["validated_capture_authority_profile_missing"]
            if status == "capture_profile_required"
            else ["conflicting_capture_authority_profiles"]
            if status == "ambiguous_capture_profile"
            else ["deterministic_capture_profile_validation_missing"]
            if status == "capture_profile_validation_required"
            else ["deterministic_capture_profile_validation_failed"]
            if status == "capture_profile_validation_failed"
            else ["deterministic_capture_profile_validation_invalid"]
        )
        expected_validation_status = (
            "required_missing"
            if status == "capture_profile_validation_required"
            else "blocked"
            if status == "capture_profile_validation_failed"
            else "invalid"
            if status == "capture_profile_validation_invalid"
            else "not_applicable_to_profile"
        )
        expected_next_action = (
            "request_deterministic_capture_profile_validation"
            if status == "capture_profile_validation_required"
            else "request_corrected_capture_intake"
            if status == "capture_profile_validation_failed"
            else "preserve_evidence_and_stop"
            if status == "capture_profile_validation_invalid"
            else "request_validated_capture_profile"
        )
        if (
            profile is not None
            or stages
            or adapters
            or blockers != expected_blocker
            or (status == "capture_profile_required" and candidates)
            or (status == "ambiguous_capture_profile" and len(candidates) < 2)
            or (
                status.startswith("capture_profile_validation_")
                and len(candidates) != 1
            )
            or route.get("capture_profile_validation_status")
            != expected_validation_status
            or (
                status in {
                    "capture_profile_validation_required",
                    "capture_profile_validation_invalid",
                }
                and route.get("capture_profile_validation_digest") is not None
            )
            or (
                status == "capture_profile_validation_failed"
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(route.get("capture_profile_validation_digest") or ""),
                )
                is None
            )
            or route.get("next_legal_action") != expected_next_action
        ):
            raise CaptureReconstructionRouteError(
                "capture_reconstruction_route_unresolved_shape_invalid"
            )
    for ordinal, row in enumerate(stages):
        if set(row) != {"ordinal", "stage_id", "method_kind", "implementation_status"}:
            raise CaptureReconstructionRouteError(
                "capture_reconstruction_route_stage_fields_invalid"
            )
        if (
            row.get("ordinal") != ordinal
            or not str(row.get("stage_id") or "").strip()
            or not str(row.get("method_kind") or "").strip()
            or row.get("implementation_status")
            not in {
                "registered",
                "registered_conditional",
                "required_not_registered",
                "required_deterministic_gate",
            }
        ):
            raise CaptureReconstructionRouteError("capture_reconstruction_route_stage_invalid")
    return route


__all__ = [
    "CAPTURE_RECONSTRUCTION_ROUTE_SCHEMA_VERSION",
    "CaptureReconstructionRouteError",
    "build_capture_reconstruction_route",
    "validate_capture_reconstruction_route",
]
