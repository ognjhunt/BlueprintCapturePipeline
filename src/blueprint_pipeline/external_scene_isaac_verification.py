"""Fail-closed Isaac request contract for authorized external scene packages.

This lane is intentionally separate from Blueprint raw-capture verification and
from provider-authored NuRec packages.  It accepts a Blueprint-compiled package
whose appearance and collision assets came from an authorized external export,
while preserving the weaker source and metric-scale claim ceiling.
"""

from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .isaac_reconstruction_verification import (
    MAX_PACKAGE_BYTES,
    MAX_RENDER_BYTES,
    IsaacReconstructionVerificationError,
    _render_measurements,
    _safe_artifact,
    build_isaac_runtime_result_v3,
)


REQUEST_SCHEMA = "external_scene_isaac_verification_request.v1"
RUNTIME_SCHEMA = "isaac_splat_nurec_render_result.v3"
AUTHORIZATION_SCHEMA = "blueprint_remote_processing_authorization.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ExternalSceneIsaacVerificationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneIsaacVerificationError(
            ["external_scene_isaac_request_not_json"]
        ) from exc
    if not isinstance(result, dict):
        raise ExternalSceneIsaacVerificationError(["external_scene_isaac_request_invalid"])
    return result


def _is_digest(value: Any) -> bool:
    return _DIGEST.fullmatch(str(value or "")) is not None


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_remote_processing_authorization(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the authorization embedded into an external-scene request."""

    authorization = _clone(value)
    supplied = authorization.pop("authorization_digest", None)
    errors: list[str] = []
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        errors.append("external_scene_authorization_schema_invalid")
    if (
        not isinstance(authorization.get("authorization_id"), str)
        or not str(authorization.get("authorization_id")).strip()
    ):
        errors.append("external_scene_authorization_id_invalid")
    if authorization.get("remote_upload_authorized") is not True:
        errors.append("external_scene_remote_upload_not_authorized")
    if authorization.get("paid_compute_authorized") is not True:
        errors.append("external_scene_paid_compute_not_authorized")
    if "vast" not in (authorization.get("provider_scope") or []):
        errors.append("external_scene_vast_not_authorized")
    required_purposes = {
        "isaac_sim_scene_ingest",
        "collision_candidate_compilation",
        "scene_task_target_analysis",
        "franka_articulated_policy_evaluation",
    }
    if not required_purposes.issubset(set(authorization.get("purpose_scope") or [])):
        errors.append("external_scene_authorization_purpose_incomplete")
    asset_digests = authorization.get("asset_digests")
    if (
        not isinstance(asset_digests, list)
        or not asset_digests
        or any(not _is_digest(item) for item in asset_digests)
    ):
        errors.append("external_scene_authorization_asset_digests_invalid")
    if authorization.get("retention_policy") != "bounded_to_evaluation_then_provider_zero":
        errors.append("external_scene_authorization_retention_invalid")
    for key in (
        "public_disclosure_authorized",
        "model_training_authorized",
        "commercial_benchmarking_authorized",
    ):
        if authorization.get(key) is not False:
            errors.append(f"external_scene_authorization_boundary_invalid:{key}")
    expected = canonical_digest(authorization, digest_field="authorization_digest")
    if supplied != expected:
        errors.append("external_scene_authorization_digest_mismatch")
    if errors:
        raise ExternalSceneIsaacVerificationError(errors)
    authorization["authorization_digest"] = expected
    return authorization


def build_external_scene_isaac_verification_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    request = _clone(value)
    supplied = request.pop("isaac_verification_request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("external_scene_isaac_request_schema_invalid")
    if _COMMIT.fullmatch(str(request.get("source_commit_sha") or "")) is None:
        errors.append("external_scene_isaac_source_commit_invalid")
    if request.get("robot_id") not in {"franka_panda", "unitree_g1"}:
        errors.append("external_scene_isaac_robot_id_invalid")
    for key in (
        "package_digest",
        "package_result_digest",
        "appearance_scene_digest",
        "collision_candidate_digest",
        "scene_frame_binding_digest",
        "target_analysis_digest",
        "target_binding_digest",
        "placement_proposal_digest",
        "render_options_digest",
        "fixed_camera_spec_digest",
        "runtime_implementation_digest",
    ):
        if not _is_digest(request.get(key)):
            errors.append(f"external_scene_isaac_{key}_invalid")
    if _IMAGE.fullmatch(str(request.get("runtime_container_image_digest") or "")) is None:
        errors.append("external_scene_isaac_runtime_image_invalid")
    reference = PurePosixPath(str(request.get("package_artifact_reference") or ""))
    if (
        not str(reference)
        or reference.is_absolute()
        or any(part in {"", ".", ".."} for part in reference.parts)
        or reference.suffix.lower() != ".usdz"
    ):
        errors.append("external_scene_isaac_package_reference_unsafe")
    camera_ids = request.get("fixed_camera_ids")
    if (
        not isinstance(camera_ids, list)
        or not camera_ids
        or len(camera_ids) != len(set(camera_ids))
        or any(_CAMERA_ID.fullmatch(str(item or "")) is None for item in camera_ids)
    ):
        errors.append("external_scene_isaac_camera_ids_invalid")
    elif request.get("robot_id") == "franka_panda" and "head_pov" in camera_ids:
        errors.append("external_scene_isaac_franka_head_pov_forbidden")
    expected_prims = request.get("expected_prim_paths")
    if expected_prims != {
        "appearance": "/World/BlueprintReconstruction/Appearance",
        "collision": "/World/BlueprintReconstruction/Collision",
    }:
        errors.append("external_scene_isaac_expected_prim_paths_invalid")
    probe = request.get("physics_probe_request")
    if not isinstance(probe, Mapping):
        errors.append("external_scene_isaac_physics_probe_missing")
    else:
        if probe.get("ground_collider_prim") != (
            "/World/BlueprintReconstruction/Collision/ExternalSceneMesh"
        ):
            errors.append("external_scene_isaac_probe_collision_prim_invalid")
        if not _finite(probe.get("ground_height_m")):
            errors.append("external_scene_isaac_ground_height_invalid")
        xy = probe.get("probe_xy_m")
        if not isinstance(xy, list) or len(xy) != 2 or any(not _finite(item) for item in xy):
            errors.append("external_scene_isaac_probe_xy_invalid")
        if probe.get("selection_status") != "derived_geometry_candidate_unverified_in_isaac":
            errors.append("external_scene_isaac_probe_status_invalid")
        if probe.get("manufacture_ground_plane") is not False:
            errors.append("external_scene_isaac_manufactured_ground_forbidden")
        if probe.get("require_contact_event") is not True:
            errors.append("external_scene_isaac_contact_event_required")
        if probe.get("test_body") != {
            "shape": "cube",
            "size_m": 0.1,
            "mass_kg": 1.0,
            "spawn_height_above_ground_m": 0.5,
        }:
            errors.append("external_scene_isaac_test_body_invalid")
        if probe.get("gravity_m_s2") != -9.81 or probe.get("physics_dt_seconds") != 1.0 / 60.0:
            errors.append("external_scene_isaac_physics_configuration_invalid")
        if (
            not isinstance(probe.get("steps"), int)
            or isinstance(probe.get("steps"), bool)
            or int(probe.get("steps") or 0) < 2
        ):
            errors.append("external_scene_isaac_probe_steps_invalid")
    authorization_value = request.get("remote_processing_authorization")
    if not isinstance(authorization_value, Mapping):
        errors.append("external_scene_authorization_missing")
    else:
        try:
            authorization = validate_remote_processing_authorization(authorization_value)
            request["remote_processing_authorization"] = authorization
            if authorization["authorization_digest"] != request.get(
                "remote_processing_authorization_digest"
            ):
                errors.append("external_scene_authorization_request_binding_mismatch")
            if request.get("appearance_scene_digest") not in authorization.get("asset_digests", []):
                errors.append("external_scene_authorization_appearance_not_bound")
        except ExternalSceneIsaacVerificationError as exc:
            errors.extend(exc.codes)
    timeout = request.get("timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 60 <= timeout <= 14_400:
        errors.append("external_scene_isaac_timeout_invalid")
    spend = request.get("spend_controls")
    if not isinstance(spend, Mapping) or spend.get("authorized") is not False:
        errors.append("external_scene_isaac_bundle_cannot_authorize_spend")
    elif (
        not _finite(spend.get("estimated_max_spend_usd"))
        or float(spend.get("estimated_max_spend_usd", 0)) <= 0
        or spend.get("hard_ttl_seconds") != timeout
        or spend.get("teardown_required") is not True
        or spend.get("provider_zero_required_before_and_after") is not True
    ):
        errors.append("external_scene_isaac_spend_controls_invalid")
    expected = {
        "external_derived_support_asset": True,
        "source_relationship_to_blueprint_raw_capture": "none",
        "blueprint_raw_capture_truth": False,
        "source_video_available": False,
        "source_video_required_for_candidate_execution": False,
        "independent_metric_scale_proven": False,
        "provider_authored_package": False,
        "blueprint_compiled_package": True,
        "exact_package_required": True,
        "headless": True,
        "display_attached": False,
        "execution_status": "awaiting_canonical_paid_runtime_authorization",
        "provider_allocation_performed": False,
        "expected_runtime_schema": RUNTIME_SCHEMA,
        "proof_effect": "none",
        "claim_ceiling": "request_only",
    }
    for key, expected_value in expected.items():
        if request.get(key) != expected_value:
            errors.append(f"external_scene_isaac_request_boundary_invalid:{key}")
    if errors:
        raise ExternalSceneIsaacVerificationError(errors)
    expected_digest = canonical_digest(request, digest_field="isaac_verification_request_digest")
    if supplied is not None and supplied != expected_digest:
        raise ExternalSceneIsaacVerificationError(
            ["external_scene_isaac_verification_request_digest_mismatch"]
        )
    request["isaac_verification_request_digest"] = expected_digest
    return request


def normalize_external_scene_isaac_verification(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    package_artifact_root: str | Path,
    runtime_artifact_root: str | Path,
) -> dict[str, Any]:
    """Independently rehash external-scene Isaac outputs without upgrading claims."""

    request = build_external_scene_isaac_verification_request(verification_request)
    try:
        runtime = build_isaac_runtime_result_v3(runtime_result)
    except IsaacReconstructionVerificationError as exc:
        raise ExternalSceneIsaacVerificationError(
            [f"external_scene_isaac_runtime_invalid:{code}" for code in exc.codes]
        ) from exc
    blockers: list[str] = []
    if runtime.get("status") != "completed":
        blockers.append("external_scene_isaac_runtime_not_completed")
    for key in (
        "isaac_verification_request_digest",
        "package_digest",
        "fixed_camera_spec_digest",
        "runtime_container_image_digest",
        "runtime_implementation_digest",
    ):
        if runtime.get(key) != request.get(key):
            blockers.append(f"external_scene_isaac_runtime_binding_mismatch:{key}")
    package_root = Path(package_artifact_root)
    runtime_root = Path(runtime_artifact_root)
    if package_root.is_symlink() or not package_root.is_dir():
        blockers.append("external_scene_isaac_package_root_invalid")
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        blockers.append("external_scene_isaac_runtime_root_invalid")
    if blockers:
        raise ExternalSceneIsaacVerificationError(blockers)
    package_root = package_root.resolve()
    runtime_root = runtime_root.resolve()
    try:
        _safe_artifact(
            package_root,
            reference=request["package_artifact_reference"],
            digest=request["package_digest"],
            suffixes={".usdz"},
            maximum_bytes=MAX_PACKAGE_BYTES,
            code="external_scene_isaac_exact_package",
        )
    except IsaacReconstructionVerificationError as exc:
        blockers.extend(exc.codes)
    stage = runtime.get("stage") if isinstance(runtime.get("stage"), Mapping) else {}
    expected_prims = request["expected_prim_paths"]
    stage_checks = {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "transforms_valid": True,
        "dependency_inspection_available": True,
        "missing_asset_count": 0,
        "obvious_scale_mismatch_detected": False,
    }
    for key, expected in stage_checks.items():
        if stage.get(key) != expected:
            blockers.append(f"external_scene_isaac_stage_invalid:{key}")
    for key in ("particlefield_prim_count", "active_collision_prim_count"):
        value = stage.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            blockers.append(f"external_scene_isaac_stage_invalid:{key}")
    if stage.get("expected_prim_paths") != expected_prims:
        blockers.append("external_scene_isaac_expected_prims_not_loaded")
    physics = (
        runtime.get("physics_probe") if isinstance(runtime.get("physics_probe"), Mapping) else {}
    )
    if (
        physics.get("ground_contact_surface_present") is not True
        or physics.get("live_rigid_body_pose_observed") is not True
        or physics.get("test_body_fell_through_floor") is not False
        or not isinstance(physics.get("contact_event_count"), int)
        or isinstance(physics.get("contact_event_count"), bool)
        or int(physics.get("contact_event_count") or 0) < 1
        or not isinstance(physics.get("steps_executed"), int)
        or isinstance(physics.get("steps_executed"), bool)
        or int(physics.get("steps_executed") or 0) < int(request["physics_probe_request"]["steps"])
    ):
        blockers.append("external_scene_isaac_physics_probe_incomplete")
    expected_probe = {
        "test_body": request["physics_probe_request"]["test_body"],
        "gravity_m_s2": request["physics_probe_request"]["gravity_m_s2"],
        "physics_dt_seconds": request["physics_probe_request"]["physics_dt_seconds"],
    }
    if physics.get("probe_configuration") != expected_probe:
        blockers.append("external_scene_isaac_physics_configuration_mismatch")
    robot = runtime.get("robot") if isinstance(runtime.get("robot"), Mapping) else {}
    if robot.get("robot_id") != request["robot_id"]:
        blockers.append("external_scene_isaac_robot_identity_mismatch")
    if robot.get("composited") is not True or robot.get("geometry_streamed") is not True:
        blockers.append("external_scene_isaac_official_robot_geometry_missing")
    resolved_robot = str(robot.get("resolved_usd") or "").lower()
    if request["robot_id"] == "franka_panda" and "frankapanda/franka.usd" not in resolved_robot:
        blockers.append("external_scene_isaac_official_franka_asset_not_observed")
    if request["robot_id"] == "unitree_g1" and not (
        "unitree" in resolved_robot and "/g1/" in resolved_robot
    ):
        blockers.append("external_scene_isaac_official_g1_asset_not_observed")
    cameras = runtime.get("cameras") if isinstance(runtime.get("cameras"), list) else []
    if [row.get("id") for row in cameras if isinstance(row, Mapping)] != request[
        "fixed_camera_ids"
    ]:
        blockers.append("external_scene_isaac_camera_set_mismatch")
    render_refs: list[dict[str, Any]] = []
    for index, row_value in enumerate(cameras):
        row = row_value if isinstance(row_value, Mapping) else {}
        try:
            path = _safe_artifact(
                runtime_root,
                reference=row.get("artifact_reference"),
                digest=row.get("digest"),
                suffixes={".png"},
                maximum_bytes=MAX_RENDER_BYTES,
                code=f"external_scene_isaac_render:{index}",
            )
            width, height, mean, std = _render_measurements(path)
            if width != row.get("width") or height != row.get("height"):
                blockers.append(f"external_scene_isaac_render_dimensions_mismatch:{index}")
            if abs(mean - float(row.get("pixel_mean") or 0.0)) > 0.001:
                blockers.append(f"external_scene_isaac_render_mean_mismatch:{index}")
            if abs(std - float(row.get("pixel_std") or 0.0)) > 0.001 or std <= 3.0:
                blockers.append(f"external_scene_isaac_render_pixels_invalid:{index}")
            render_refs.append(
                {
                    "id": row.get("id"),
                    "artifact_reference": row.get("artifact_reference"),
                    "digest": row.get("digest"),
                }
            )
        except IsaacReconstructionVerificationError as exc:
            blockers.extend(exc.codes)
    if blockers:
        raise ExternalSceneIsaacVerificationError(blockers)
    result = {
        "schema_version": "external_scene_isaac_verification_result.v1",
        "status": "verified_derived_scene_compatibility_only",
        "isaac_verification_request_digest": request["isaac_verification_request_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "package_digest": request["package_digest"],
        "package_result_digest": request["package_result_digest"],
        "remote_processing_authorization_digest": request["remote_processing_authorization_digest"],
        "target_analysis_digest": request["target_analysis_digest"],
        "target_binding_digest": request["target_binding_digest"],
        "placement_proposal_digest": request["placement_proposal_digest"],
        "exact_package_rehash_verified": True,
        "runtime_artifact_rehash_verified": True,
        "fixed_camera_render_references": render_refs,
        "physics_probe": dict(physics),
        "articulated_policy_trace_pair": runtime.get("articulated_policy_trace_pair"),
        "checks": {
            "exact_package_opened": True,
            "expected_prims_present": True,
            "particlefield_loaded": True,
            "collision_geometry_active": True,
            "live_contact_observed": True,
            "test_body_fell_through_floor": False,
            "fixed_camera_renders_nonblank": True,
            "official_franka_loaded": bool(
                request["robot_id"] == "franka_panda"
                and robot.get("composited") is True
                and robot.get("geometry_streamed") is True
            ),
        },
        "source_video_available": False,
        "source_video_required_for_this_evidence": False,
        "independent_metric_scale_proven": False,
        "simulator_task_success_proven": False,
        "comparative_policy_ranking_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": "external_scene_isaac_load_render_contact_presence_only",
        "claim_ceiling": "derived_scene_isaac_compatibility_and_trace_observation",
    }
    result["verification_result_digest"] = canonical_digest(
        result, digest_field="verification_result_digest"
    )
    return result


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "REQUEST_SCHEMA",
    "RUNTIME_SCHEMA",
    "ExternalSceneIsaacVerificationError",
    "build_external_scene_isaac_verification_request",
    "normalize_external_scene_isaac_verification",
    "validate_remote_processing_authorization",
]
