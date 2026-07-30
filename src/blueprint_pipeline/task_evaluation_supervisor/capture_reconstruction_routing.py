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

_PROFILE_STAGES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "iphone_arkit_lidar": (
        ("verify_arkit_raw_contract", "capture_validation", "required_deterministic_gate"),
        ("compile_arkit_metric_scaffold", "lidar_depth_fusion", "registered_conditional"),
        ("train_pose_conditioned_3dgs", "gaussian_splat_3d", "required_not_registered"),
        ("validate_testbed_layers", "testbed_compilation", "required_deterministic_gate"),
    ),
    "iphone_arkit_non_lidar": (
        ("verify_arkit_pose_intrinsics", "capture_validation", "required_deterministic_gate"),
        ("verify_metric_scale_anchor", "metric_scaffold", "required_not_registered"),
        ("train_pose_conditioned_3dgs", "gaussian_splat_3d", "required_not_registered"),
        ("validate_testbed_layers", "testbed_compilation", "required_deterministic_gate"),
    ),
    "camera_360_equirectangular": (
        ("verify_equirectangular_metadata", "capture_validation", "required_deterministic_gate"),
        (
            "project_spherical_to_perspective_views",
            "equirectangular_normalization",
            "required_not_registered",
        ),
        ("estimate_camera_poses_with_sfm", "pose_sfm_estimation", "required_not_registered"),
        ("train_unscaled_appearance_3dgs", "gaussian_splat_3d", "required_not_registered"),
        ("verify_metric_scale_anchor", "metric_scaffold", "required_not_registered"),
        ("validate_testbed_layers", "testbed_compilation", "required_deterministic_gate"),
    ),
    "camera_360_native": (
        ("retain_native_360_originals", "capture_validation", "required_deterministic_gate"),
        (
            "normalize_native_360_capture",
            "native_360_normalization",
            "registered_conditional",
        ),
        (
            "project_spherical_to_perspective_views",
            "equirectangular_normalization",
            "required_not_registered",
        ),
        ("estimate_camera_poses_with_sfm", "pose_sfm_estimation", "required_not_registered"),
        ("train_unscaled_appearance_3dgs", "gaussian_splat_3d", "required_not_registered"),
        ("verify_metric_scale_anchor", "metric_scaffold", "required_not_registered"),
        ("validate_testbed_layers", "testbed_compilation", "required_deterministic_gate"),
    ),
    "monocular_video": (
        ("decode_retained_observation_frames", "decoded_observation_index", "registered"),
        ("estimate_camera_poses_with_sfm", "pose_sfm_estimation", "required_not_registered"),
        ("train_unscaled_appearance_3dgs", "gaussian_splat_3d", "required_not_registered"),
        ("verify_metric_scale_anchor", "metric_scaffold", "required_not_registered"),
        ("validate_testbed_layers", "testbed_compilation", "required_deterministic_gate"),
    ),
    "precomputed_external_reconstruction": (
        ("verify_source_capture_binding", "capture_validation", "required_deterministic_gate"),
        (
            "import_external_reconstruction",
            "external_reconstruction_import",
            "required_not_registered",
        ),
        (
            "independently_validate_metric_semantic_physics_layers",
            "testbed_compilation",
            "required_deterministic_gate",
        ),
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
            "local://native-360-normalization-v1",
        ]
        if profile == "camera_360_native"
        else ["local://decoded-observation-index-v1"]
        if profile in _VIDEO_PROFILES
        else []
    )
    route = {
        "schema_version": CAPTURE_RECONSTRUCTION_ROUTE_SCHEMA_VERSION,
        "capture_build_digest": capture_build["capture_build_digest"],
        "status": status,
        "capture_authority_profile": profile,
        "declared_profile_candidates": list(profiles),
        "requested_claim_types": _strings(list(requested_claim_types)),
        "required_representations": _required_representations(requested_claim_types),
        "stages": stages,
        "currently_registered_adapters": sorted(executable_adapters),
        "blockers": blockers,
        "next_legal_action": (
            "compile_profile_specific_reconstruction_plan"
            if status == "route_proposed"
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
        or status not in {"route_proposed", "capture_profile_required", "ambiguous_capture_profile"}
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
                "local://native-360-normalization-v1",
            ]
            if profile == "camera_360_native"
            else ["local://decoded-observation-index-v1"]
            if profile in _VIDEO_PROFILES
            else []
        )
        if (
            profile not in CAPTURE_AUTHORITY_PROFILES
            or candidates != [profile]
            or stages != expected_stages
            or adapters != sorted(expected_adapters)
            or blockers
            or route.get("next_legal_action") != "compile_profile_specific_reconstruction_plan"
        ):
            raise CaptureReconstructionRouteError("capture_reconstruction_route_profile_invalid")
    else:
        expected_blocker = (
            ["validated_capture_authority_profile_missing"]
            if status == "capture_profile_required"
            else ["conflicting_capture_authority_profiles"]
        )
        if (
            profile is not None
            or stages
            or adapters
            or blockers != expected_blocker
            or (status == "capture_profile_required" and candidates)
            or (status == "ambiguous_capture_profile" and len(candidates) < 2)
            or route.get("next_legal_action") != "request_validated_capture_profile"
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
