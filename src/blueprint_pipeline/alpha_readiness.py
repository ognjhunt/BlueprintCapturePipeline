"""Alpha-readiness validation and downstream WebApp sync helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import (
    optional_read_json,
    parse_bool,
    parse_gs_uri,
    read_json,
    to_pipeline_prefix,
    utc_now_iso,
    write_json,
)
from .webapp_sync import (
    WebappSyncError,
    derive_webapp_opportunity_state,
    derive_webapp_qualification_state,
    sync_webapp_pipeline_attachment,
)


_COMMON_ENV_VARS = (
    "PIPELINE_PROJECT_ID",
    "PIPELINE_REGION",
    "PIPELINE_BUCKET",
    "GCS_ROOT",
    "PIPELINE_SYNC_WEBAPP_URL",
    "PIPELINE_SYNC_TOKEN",
)
_PRIVACY_ENV_VARS = (
    "PRIVACY_RUNNER_TOKEN",
    "PRIVACY_SAM3_URL",
    "PRIVACY_VIP_URL",
    "PRIVACY_DEEPPRIVACY2_URL",
)


def _read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _check(name: str, passed: bool, detail: str, *, category: str) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "category": category,
    }


def _check_file(path: Path, *, name: str, detail: str, category: str = "artifact") -> Dict[str, Any]:
    return _check(name, path.is_file(), detail if path.is_file() else f"{detail} missing", category=category)


def _bool_env(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    return parse_bool(env.get(name), default=default)


def _env_check(env: Mapping[str, str], name: str, *, expected_value: Optional[str] = None) -> Dict[str, Any]:
    value = str(env.get(name) or "").strip()
    if expected_value is None:
        passed = bool(value)
        detail = f"{name} is configured" if passed else f"{name} is missing"
    else:
        passed = value.lower() == expected_value.lower()
        detail = (
            f"{name}={expected_value}"
            if passed
            else f"{name} must be {expected_value}, got {value or 'unset'}"
        )
    return _check(name.lower(), passed, detail, category="env")


def _mode_payload(descriptor_payload: Mapping[str, Any]) -> Dict[str, Any]:
    capture_mode = descriptor_payload.get("capture_mode")
    if isinstance(capture_mode, Mapping):
        return dict(capture_mode)
    metadata = descriptor_payload.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("capture_mode"), Mapping):
        return dict(metadata.get("capture_mode") or {})
    return {}


def _present_value(payload: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _uri(bucket: str, pipeline_prefix: str, relative_path: str) -> str:
    return f"gs://{bucket}/{pipeline_prefix}/{relative_path}"


def _latest_sync_payload(existing: Mapping[str, Any]) -> Dict[str, Any]:
    syncs = existing.get("syncs")
    if isinstance(syncs, Mapping):
        latest_stage = str(existing.get("latest_stage") or "").strip()
        latest = syncs.get(latest_stage)
        if isinstance(latest, Mapping):
            return dict(latest)
    return dict(existing)


def _runtime_capability_payload(
    *,
    profile: str,
    runtime_launch_expected: bool,
    geometry_summary: Mapping[str, Any],
    site_world_spec: Mapping[str, Any],
    site_world_registration: Mapping[str, Any],
    site_world_health: Mapping[str, Any],
) -> Dict[str, Any]:
    has_site_world_bundle = bool(site_world_spec and site_world_registration and site_world_health)
    geometry_ready = bool(geometry_summary.get("ready_for_world_model"))
    geometry_live_ready = bool(geometry_summary.get("geometry_live_ready"))
    geometry_source = str(geometry_summary.get("geometry_source") or "missing").strip()
    fallback_used = bool(geometry_summary.get("fallback_used"))
    provider_native_result = bool(geometry_summary.get("provider_native_result"))
    site_frame_available = bool(geometry_summary.get("site_frame_available"))
    scale_resolved = bool(geometry_summary.get("scale_resolved"))
    runtime_launchable = bool(site_world_health.get("launchable"))
    runtime_status = str(site_world_health.get("status") or "missing").strip().lower()
    geometry_required = profile in {"meta_glasses", "android_video", "iphone_video_only"}

    blockers: List[str] = []
    if not has_site_world_bundle:
        blockers.append("missing_site_world_bundle")
    if geometry_required and not geometry_ready:
        blockers.append("geometry_not_ready")
    if geometry_required and (
        fallback_used
        or geometry_source != "video_to_world"
        or not geometry_live_ready
        or not provider_native_result
        or not site_frame_available
        or not scale_resolved
    ):
        blockers.append("geometry_not_live_video_to_world")
    if runtime_launch_expected and not runtime_launchable:
        blockers.append("runtime_not_launchable")
    if runtime_launch_expected and runtime_status in {"missing", "", "blocked", "failed"}:
        blockers.append("runtime_health_not_ready")

    return {
        "claim_scope": "native_runtime_capability_only",
        "status": "ready" if not blockers else "blocked",
        "launchable": runtime_launchable,
        "geometry_required": geometry_required,
        "geometry_ready": geometry_ready,
        "geometry_live_ready": geometry_live_ready,
        "geometry_source": geometry_source,
        "fallback_used": fallback_used,
        "provider_native_result": provider_native_result,
        "site_frame_available": site_frame_available,
        "scale_resolved": scale_resolved,
        "non_arkit_geometry_state": (
            "ready"
            if geometry_required
            and geometry_live_ready
            and provider_native_result
            and geometry_source == "video_to_world"
            else "degraded"
            if geometry_required
            and geometry_source == "local_sfm"
            and bool(geometry_summary.get("contract_ready_for_world_model"))
            else "not_applicable"
            if not geometry_required
            else "blocked"
        ),
        "site_world_bundle_ready": has_site_world_bundle,
        "runtime_health_status": runtime_status or "missing",
        "blockers": blockers,
    }


def write_pipeline_sync_result(
    *,
    pipeline_root: Path,
    stage: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    path = pipeline_root / "webapp_sync_result.json"
    existing = _read_json_object(path)
    syncs = existing.get("syncs") if isinstance(existing.get("syncs"), Mapping) else {}
    merged_syncs = {str(key): value for key, value in syncs.items()}
    if existing and not merged_syncs:
        legacy_stage = str(existing.get("latest_stage") or existing.get("stage") or "qualification").strip() or "qualification"
        merged_syncs[legacy_stage] = _latest_sync_payload(existing)
    merged_syncs[stage] = dict(result)
    payload = {
        "status": str(result.get("status") or "unknown"),
        "latest_stage": stage,
        "syncs": merged_syncs,
    }
    write_json(path, payload)
    return payload


def build_alpha_readiness_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    resolved_env = dict(os.environ if env is None else env)
    descriptor_path = capture_root / "capture_descriptor.json"
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    scene_memory_root = pipeline_root / "scene_memory"
    presentation_root = pipeline_root / "presentation_world"
    privacy_depth_root = pipeline_root / "privacy_depth"
    geometry_root = pipeline_root / "geometry"

    descriptor_payload = _read_json_object(descriptor_path)
    descriptor = (
        CaptureDescriptor.from_dict(descriptor_payload)
        if descriptor_payload
        else CaptureDescriptor.from_dict({})
    )
    mode_payload = _mode_payload(descriptor_payload)
    capture_mode_resolved = str(mode_payload.get("resolved_mode") or "").strip() or None
    if capture_mode_resolved is None:
        quality = descriptor_payload.get("quality") if isinstance(descriptor_payload.get("quality"), Mapping) else {}
        rights = descriptor_payload.get("metadata") if isinstance(descriptor_payload.get("metadata"), Mapping) else {}
        rights_block = rights.get("capture_rights") if isinstance(rights.get("capture_rights"), Mapping) else {}
        candidate = bool(
            descriptor.arkit_poses_uri
            or bool(quality.get("world_model_candidate"))
            or (
                bool(quality.get("geometry_ready"))
                and bool(rights_block.get("derived_scene_generation_allowed", False))
            )
        )
        capture_mode_resolved = "site_world_candidate" if candidate else "qualification_only"

    qa_report = _read_json_object(capture_root / "qa_report.json")
    gemini_review = _read_json_object(pipeline_root / "gemini_capture_fidelity_review.json")
    privacy_manifest = _read_json_object(pipeline_root / "privacy_processing_manifest.json")
    webapp_sync = _read_json_object(pipeline_root / "webapp_sync_result.json")
    geometry_summary = _read_json_object(geometry_root / "geometry_summary.json")
    site_world_spec = _read_json_object(eval_root / "site_world_spec.json")
    site_world_registration = _read_json_object(eval_root / "site_world_registration.json")
    site_world_health = _read_json_object(eval_root / "site_world_health.json")

    profile = "unsupported"
    if descriptor.capture_source == "iphone":
        profile = (
            "iphone_arkit_lidar"
            if descriptor.capture_modality == "iphone_arkit_lidar"
            else "iphone_video_only"
        )
    elif descriptor.capture_source == "glasses":
        profile = "meta_glasses"
    elif descriptor.capture_source == "android":
        profile = "android_video"

    requested_outputs = {
        str(value or "").strip().lower()
        for value in descriptor.requested_outputs
        if str(value or "").strip()
    }
    evaluation_requested = bool(
        requested_outputs.intersection({"deeper_evaluation", "evaluation_prep"})
        or eval_root.exists()
    )
    runtime_launch_expected = parse_bool(
        resolved_env.get("PIPELINE_ALPHA_EXPECT_HOSTED_RUNTIME"),
        default=evaluation_requested,
    )
    runtime_capability = _runtime_capability_payload(
        profile=profile,
        runtime_launch_expected=runtime_launch_expected,
        geometry_summary=geometry_summary,
        site_world_spec=site_world_spec,
        site_world_registration=site_world_registration,
        site_world_health=site_world_health,
    )

    env_checks: List[Dict[str, Any]] = [_env_check(resolved_env, name) for name in _COMMON_ENV_VARS]
    env_checks.append(_env_check(resolved_env, "PIPELINE_SYNC_REQUIRED", expected_value="true"))
    genai_present = bool(
        str(resolved_env.get("GOOGLE_GENAI_API_KEY") or "").strip()
        or str(resolved_env.get("GEMINI_API_KEY") or "").strip()
    )
    env_checks.append(
        _check(
            "gemini_api_key",
            genai_present,
            "GOOGLE_GENAI_API_KEY or GEMINI_API_KEY is configured"
            if genai_present
            else "GOOGLE_GENAI_API_KEY or GEMINI_API_KEY is missing",
            category="env",
        )
    )
    env_checks.append(_env_check(resolved_env, "PRIVACY_PIPELINE_ENABLED", expected_value="true"))
    env_checks.append(_env_check(resolved_env, "PRIVACY_FAIL_CLOSED", expected_value="true"))
    env_checks.extend(_env_check(resolved_env, name) for name in _PRIVACY_ENV_VARS)
    if runtime_launch_expected:
        env_checks.append(_env_check(resolved_env, "SITE_WORLD_RUNTIME_SERVICE_URL"))
    if profile in {"iphone_video_only", "meta_glasses", "android_video"}:
        env_checks.append(_env_check(resolved_env, "VIDEO_TO_WORLD_URL"))
        env_checks.append(_env_check(resolved_env, "VIDEO_TO_WORLD_RUNNER_TOKEN"))
    preview_provider = str(resolved_env.get("BLUEPRINT_PREVIEW_PROVIDER") or "world_labs").strip()
    if preview_provider == "world_labs":
        env_checks.append(_env_check(resolved_env, "WORLDLABS_API_KEY"))

    common_checks: List[Dict[str, Any]] = [
        _check_file(descriptor_path, name="capture_descriptor", detail="capture_descriptor.json exists"),
        _check_file(capture_root / "qa_report.json", name="qa_report", detail="qa_report.json exists"),
        _check(
            "gemini_review_succeeded",
            str(gemini_review.get("status") or "").strip().lower() == "succeeded",
            "Gemini capture fidelity review succeeded"
            if str(gemini_review.get("status") or "").strip().lower() == "succeeded"
            else f"Gemini review status is {gemini_review.get('status') or 'missing'}",
            category="status",
        ),
        _check(
            "privacy_completed",
            str(privacy_manifest.get("status") or "").strip().lower()
            in {
                "no_people_detected",
                "person_removed",
                "face_anonymized_fallback",
                "full_frame_redacted_local_proof",
            },
            "privacy produced buyer-safe walkthrough media"
            if str(privacy_manifest.get("status") or "").strip().lower()
            in {
                "no_people_detected",
                "person_removed",
                "face_anonymized_fallback",
                "full_frame_redacted_local_proof",
            }
            else f"privacy status is {privacy_manifest.get('status') or 'not_run'}",
            category="status",
        ),
        _check_file(pipeline_root / "qualification_summary.json", name="qualification_summary", detail="qualification_summary.json exists"),
        _check_file(pipeline_root / "capture_quality_summary.json", name="capture_quality_summary", detail="capture_quality_summary.json exists"),
        _check_file(pipeline_root / "rights_and_compliance_summary.json", name="rights_and_compliance_summary", detail="rights_and_compliance_summary.json exists"),
        _check_file(pipeline_root / "buyer_trust_score.json", name="buyer_trust_score", detail="buyer_trust_score.json exists"),
        _check_file(pipeline_root / "world_model_fit_summary.json", name="world_model_fit_summary", detail="world_model_fit_summary.json exists"),
        _check_file(pipeline_root / "provenance_summary.json", name="provenance_summary", detail="provenance_summary.json exists"),
        _check_file(pipeline_root / "gemini_capture_fidelity_review.json", name="gemini_capture_fidelity_review", detail="gemini_capture_fidelity_review.json exists"),
        _check_file(pipeline_root / "privacy_processing_manifest.json", name="privacy_processing_manifest", detail="privacy_processing_manifest.json exists"),
        _check_file(pipeline_root / "privacy_verification_report.json", name="privacy_verification_report", detail="privacy_verification_report.json exists"),
        _check_file(pipeline_root / "opportunity_handoff.json", name="opportunity_handoff", detail="opportunity_handoff.json exists"),
        _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists"),
        _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists"),
        _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists"),
        _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists"),
        _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists"),
        _check(
            "webapp_sync_succeeded",
            str(webapp_sync.get("status") or "").strip().lower() == "succeeded",
            "webapp sync succeeded"
            if str(webapp_sync.get("status") or "").strip().lower() == "succeeded"
            else f"webapp sync status is {webapp_sync.get('status') or 'missing'}",
            category="status",
        ),
    ]
    if runtime_launch_expected:
        blockers = {
            str(item).strip()
            for item in site_world_health.get("blockers", [])
            if str(item).strip()
        }
        common_checks.append(
            _check(
                "hosted_runtime_configured",
                "missing_runtime_service_url" not in blockers,
                "hosted runtime URL is configured"
                if "missing_runtime_service_url" not in blockers
                else "site world health is blocked by missing runtime service URL",
                category="status",
            )
        )

    common_passed = all(item["passed"] for item in common_checks if item["category"] != "env") and all(
        item["passed"] for item in env_checks
    )

    path_checks: List[Dict[str, Any]] = []
    external_alpha = {"status": "no_go", "reason": "unsupported_capture_path"}
    internal_alpha = {"status": "not_applicable", "reason": "not_meta_glasses"}

    if profile == "iphone_arkit_lidar":
        path_checks = [
            _check(
                "capture_source_iphone",
                descriptor.capture_source == "iphone",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check(
                "capture_modality_iphone_arkit_lidar",
                descriptor.capture_modality == "iphone_arkit_lidar",
                f"capture_modality is {descriptor.capture_modality or 'missing'}",
                category="path",
            ),
            _check(
                "arkit_bundle_complete",
                bool(descriptor.arkit_poses_uri and descriptor.arkit_intrinsics_uri and descriptor.arkit_depth_prefix_uri),
                "ARKit poses, intrinsics, and depth refs are present"
                if descriptor.arkit_poses_uri and descriptor.arkit_intrinsics_uri and descriptor.arkit_depth_prefix_uri
                else "ARKit bundle refs are incomplete",
                category="path",
            ),
            _check(
                "capture_mode_site_world_candidate",
                capture_mode_resolved == "site_world_candidate",
                f"capture_mode resolved to {capture_mode_resolved or 'missing'}",
                category="path",
            ),
            _check(
                "qa_report_passed",
                str(qa_report.get("status") or "").strip().lower() == "passed",
                f"qa_report status is {qa_report.get('status') or 'missing'}",
                category="path",
            ),
            _check_file(presentation_root / "presentation_bundle.json", name="presentation_bundle", detail="presentation_bundle.json exists", category="path"),
            _check_file(presentation_root / "presentation_world_manifest.json", name="presentation_world_manifest", detail="presentation_world_manifest.json exists", category="path"),
            _check_file(presentation_root / "runtime_demo_manifest.json", name="runtime_demo_manifest", detail="runtime_demo_manifest.json exists", category="path"),
            _check_file(eval_root / "hosted_session_runtime_manifest.json", name="hosted_session_runtime_manifest", detail="hosted_session_runtime_manifest.json exists", category="path"),
            _check_file(eval_root / "launchable_export_bundle.json", name="launchable_export_bundle", detail="launchable_export_bundle.json exists", category="path"),
            _check(
                "buyer_safe_walkthrough",
                bool(_present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")),
                "privacy produced buyer-safe walkthrough URI"
                if _present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")
                else "privacy did not produce buyer-safe walkthrough URI",
                category="path",
            ),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        external_alpha = {
            "status": "go" if common_passed and all(item["passed"] for item in path_checks) else "no_go",
            "reason": "all_common_and_iphone_checks_passed"
            if common_passed and all(item["passed"] for item in path_checks)
            else "iphone_alpha_requirements_not_met",
        }
    elif profile == "iphone_video_only":
        path_checks = [
            _check(
                "capture_source_iphone",
                descriptor.capture_source == "iphone",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check(
                "capture_modality_iphone_video_only",
                descriptor.capture_modality == "iphone_video_only",
                f"capture_modality is {descriptor.capture_modality or 'missing'}",
                category="path",
            ),
            _check_file(geometry_root / "geometry_manifest.json", name="geometry_manifest", detail="geometry_manifest.json exists", category="path"),
            _check_file(geometry_root / "geometry_summary.json", name="geometry_summary", detail="geometry_summary.json exists", category="path"),
            _check(
                "geometry_ready_for_world_model",
                bool(geometry_summary.get("ready_for_world_model")),
                "geometry is ready for native world-model conditioning"
                if bool(geometry_summary.get("ready_for_world_model"))
                else "geometry is not ready for native world-model conditioning",
                category="path",
            ),
            _check(
                "geometry_uses_real_video_to_world",
                str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used")),
                "geometry uses video_to_world without fallback"
                if str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used"))
                else "geometry fell back instead of using true video_to_world",
                category="path",
            ),
            _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists", category="path"),
            _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists", category="path"),
            _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists", category="path"),
            _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists", category="path"),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        external_alpha = {
            "status": "go" if common_passed and all(item["passed"] for item in path_checks) else "no_go",
            "reason": "all_common_and_iphone_video_only_checks_passed"
            if common_passed and all(item["passed"] for item in path_checks)
            else "iphone_video_only_alpha_requirements_not_met",
        }
    elif profile == "meta_glasses":
        path_checks = [
            _check(
                "capture_source_glasses",
                descriptor.capture_source == "glasses",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check_file(geometry_root / "geometry_manifest.json", name="geometry_manifest", detail="geometry_manifest.json exists", category="path"),
            _check_file(geometry_root / "geometry_summary.json", name="geometry_summary", detail="geometry_summary.json exists", category="path"),
            _check(
                "geometry_ready_for_world_model",
                bool(geometry_summary.get("ready_for_world_model")),
                "geometry is ready for world-model conditioning"
                if bool(geometry_summary.get("ready_for_world_model"))
                else "geometry is not ready for world-model conditioning",
                category="path",
            ),
            _check(
                "geometry_uses_real_video_to_world",
                str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used")),
                "geometry uses video_to_world without fallback"
                if str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used"))
                else "geometry fell back instead of using true video_to_world",
                category="path",
            ),
            _check_file(privacy_depth_root / "depth_manifest.json", name="privacy_depth_manifest", detail="privacy depth_manifest.json exists", category="path"),
            _check_file(privacy_depth_root / "confidence_manifest.json", name="privacy_confidence_manifest", detail="privacy confidence_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists", category="path"),
            _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists", category="path"),
            _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists", category="path"),
            _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists", category="path"),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        glasses_contract_ready = common_passed and all(item["passed"] for item in path_checks)
        external_alpha = {
            "status": "no_go",
            "reason": "glasses_requires_physical_device_and_operator_launch_evidence"
            if glasses_contract_ready
            else "glasses_external_alpha_requirements_not_met",
            "contract_status": "ready" if glasses_contract_ready else "blocked",
            "contract_reason": "all_common_and_glasses_checks_passed"
            if glasses_contract_ready
            else "glasses_contract_requirements_not_met",
        }
        internal_alpha = {
            "status": "go" if glasses_contract_ready else "no_go",
            "reason": "all_common_and_glasses_checks_passed"
            if glasses_contract_ready
            else "glasses_internal_alpha_requirements_not_met",
        }
    elif profile == "android_video":
        path_checks = [
            _check(
                "capture_source_android",
                descriptor.capture_source == "android",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check_file(geometry_root / "geometry_manifest.json", name="geometry_manifest", detail="geometry_manifest.json exists", category="path"),
            _check_file(geometry_root / "geometry_summary.json", name="geometry_summary", detail="geometry_summary.json exists", category="path"),
            _check(
                "geometry_ready_for_world_model",
                bool(geometry_summary.get("ready_for_world_model")),
                "geometry is ready for native world-model conditioning"
                if bool(geometry_summary.get("ready_for_world_model"))
                else "geometry is not ready for native world-model conditioning",
                category="path",
            ),
            _check(
                "geometry_uses_real_video_to_world",
                str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used")),
                "geometry uses video_to_world without fallback"
                if str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used"))
                else "geometry fell back instead of using true video_to_world",
                category="path",
            ),
            _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists", category="path"),
            _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists", category="path"),
            _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists", category="path"),
            _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists", category="path"),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        android_contract_ready = common_passed and all(item["passed"] for item in path_checks)
        external_alpha = {
            "status": "no_go",
            "reason": "android_requires_physical_device_and_operator_launch_evidence"
            if android_contract_ready
            else "android_external_alpha_requirements_not_met",
            "contract_status": "ready" if android_contract_ready else "blocked",
            "contract_reason": "all_common_and_android_checks_passed"
            if android_contract_ready
            else "android_contract_requirements_not_met",
        }
        internal_alpha = {
            "status": "go" if android_contract_ready else "no_go",
            "reason": "all_common_and_android_checks_passed"
            if android_contract_ready
            else "android_internal_alpha_requirements_not_met",
        }

    external_alpha_go = str(external_alpha.get("status") or "").strip().lower() == "go"
    internal_alpha_go = str(internal_alpha.get("status") or "").strip().lower() == "go"
    if external_alpha_go:
        device_alpha_profile = {
            "status": "ready_for_external_alpha",
            "reason": external_alpha.get("reason"),
        }
    elif internal_alpha_go:
        device_alpha_profile = {
            "status": "internal_only",
            "reason": internal_alpha.get("reason"),
        }
    else:
        device_alpha_profile = {
            "status": "blocked",
            "reason": external_alpha.get("reason") or internal_alpha.get("reason"),
        }

    failed_checks = [
        item["name"]
        for item in [*env_checks, *common_checks, *path_checks]
        if not item["passed"]
    ]
    no_go_reasons = [item["detail"] for item in [*env_checks, *common_checks, *path_checks] if not item["passed"]]

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "capture_source": descriptor.capture_source,
        "capture_modality": descriptor.capture_modality,
        "capture_mode": capture_mode_resolved,
        "profile": profile,
        "runtime_launch_expected": runtime_launch_expected,
        "runtime_capability": runtime_capability,
        "launch_market_readiness": {
            "contract_ready": bool(external_alpha_go or internal_alpha_go),
            "internal_pilot_ready": bool(internal_alpha_go),
            "external_market_ready": bool(external_alpha_go),
            "site_faithful_market_ready": bool(external_alpha_go and profile == "iphone_arkit_lidar"),
            "claim_boundary": (
                "external_market"
                if external_alpha_go
                else "internal_or_blocked_until_live_operator_evidence"
            ),
        },
        "environment_checks": env_checks,
        "common_checks": common_checks,
        "path_checks": path_checks,
        "verdicts": {
            "external_alpha": external_alpha,
            "internal_experimental_alpha": internal_alpha,
        },
        "device_alpha_profile": device_alpha_profile,
        "common_status": "passed" if common_passed else "failed",
        "path_status": "passed" if path_checks and all(item["passed"] for item in path_checks) else "failed",
        "failed_checks": failed_checks,
        "no_go_reasons": no_go_reasons,
        "service_snapshot": {
            "webapp_sync_status": webapp_sync.get("status"),
            "privacy_status": privacy_manifest.get("status") or "not_run",
            "runtime_health_status": site_world_health.get("status") or "missing",
            "runtime_launchable": bool(site_world_health.get("launchable")),
        },
    }


def write_alpha_readiness_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload = build_alpha_readiness_summary(capture_root=capture_root, env=env)
    write_json(capture_root / "pipeline" / "alpha_readiness_summary.json", payload)
    return payload


def build_launch_gate_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    resolved_env = dict(os.environ if env is None else env)
    descriptor = CaptureDescriptor.from_dict(_read_json_object(capture_root / "capture_descriptor.json"))
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"

    alpha_summary = write_alpha_readiness_summary(capture_root=capture_root, env=resolved_env)
    opportunity_handoff = _read_json_object(pipeline_root / "opportunity_handoff.json")
    qualification_record = _read_json_object(pipeline_root / "qualification_record.json")
    scorecard = _read_json_object(pipeline_root / "capture_qa_scorecard.json")
    privacy_manifest = _read_json_object(pipeline_root / "privacy_processing_manifest.json")
    payout_recommendation = _read_json_object(pipeline_root / "capturer_payout_recommendation.json")
    launchable_export_bundle = _read_json_object(eval_root / "launchable_export_bundle.json")
    webapp_sync = _read_json_object(pipeline_root / "webapp_sync_result.json")
    authoritative_qualification_state = derive_webapp_qualification_state(
        readiness_state=qualification_record.get("readiness_state"),
        completeness_status=scorecard.get("completeness_status"),
    )

    site_submission_id = (
        descriptor.site_submission_id
        or str(opportunity_handoff.get("site_submission_id") or "").strip()
    )
    buyer_request_id = (
        descriptor.buyer_request_id
        or str(opportunity_handoff.get("buyer_request_id") or "").strip()
    )
    capture_job_id = (
        descriptor.capture_job_id
        or str(opportunity_handoff.get("capture_job_id") or "").strip()
    )
    payout_eligible = bool(
        payout_recommendation.get("eligible_for_payout")
        if payout_recommendation.get("eligible_for_payout") is not None
        else descriptor.quoted_payout_cents is not None
        or payout_recommendation.get("recommended_payout_cents") is not None
    )
    profile = str(alpha_summary.get("profile") or "unsupported")
    external_alpha = alpha_summary.get("verdicts", {}).get("external_alpha", {})
    internal_alpha = alpha_summary.get("verdicts", {}).get("internal_experimental_alpha", {})
    device_alpha_profile = alpha_summary.get("device_alpha_profile", {})
    runtime_capability = alpha_summary.get("runtime_capability", {})
    external_alpha_go = str(external_alpha.get("status") or "").strip().lower() == "go"
    internal_alpha_go = str(internal_alpha.get("status") or "").strip().lower() == "go"
    launchable_bundle_ready = bool(
        launchable_export_bundle
        and str(launchable_export_bundle.get("status") or "").strip().lower() in {"ready", "launch_ready"}
    )
    if not launchable_bundle_ready and (eval_root / "launchable_export_bundle.json").is_file():
        launchable_bundle_ready = True

    stage_checks = [
        _check(
            "inbound_request_linked",
            bool(site_submission_id),
            f"site_submission_id is {site_submission_id}"
            if site_submission_id
            else "site_submission_id is missing from the captured opportunity handoff",
            category="launch_gate",
        ),
        _check(
            "approved_marketplace_capture_job_linked",
            bool(capture_job_id),
            f"capture_job_id is {capture_job_id}"
            if capture_job_id
            else "capture_job_id is missing from the captured job linkage",
            category="launch_gate",
        ),
        _check(
            "buyer_request_linked",
            bool(buyer_request_id),
            f"buyer_request_id is {buyer_request_id}"
            if buyer_request_id
            else "buyer_request_id is missing from the buyer request linkage",
            category="launch_gate",
        ),
        _check(
            "mobile_claim_context_captured",
            bool(descriptor.capture_source and descriptor.quoted_payout_cents is not None),
            (
                f"capture source {descriptor.capture_source} retained quoted payout {descriptor.quoted_payout_cents}"
                if descriptor.capture_source and descriptor.quoted_payout_cents is not None
                else "capture descriptor is missing source or quoted payout context"
            ),
            category="launch_gate",
        ),
        _check_file(
            capture_root / "raw" / "capture_upload_complete.json",
            name="mobile_upload_completed",
            detail="raw/capture_upload_complete.json exists",
            category="launch_gate",
        ),
        _check(
            "qualification_authoritative",
            authoritative_qualification_state in {"qualified_ready", "qualified_risky"}
            or external_alpha_go
            or internal_alpha_go,
            (
                f"qualification_state is {authoritative_qualification_state or 'not_ready_yet'} and alpha verdict is enforced"
                if authoritative_qualification_state in {"qualified_ready", "qualified_risky"}
                or external_alpha_go
                or internal_alpha_go
                else "authoritative qualification_state did not reach a launchable verdict"
            ),
            category="launch_gate",
        ),
        _check(
            "privacy_safe_buyer_media_ready",
            bool(_present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")),
            "privacy manifest includes buyer-safe walkthrough media"
            if _present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")
            else "privacy-safe walkthrough media is missing",
            category="launch_gate",
        ),
        _check(
            "webapp_sync_completed",
            str(webapp_sync.get("status") or "").strip().lower() == "succeeded",
            "webapp sync succeeded"
            if str(webapp_sync.get("status") or "").strip().lower() == "succeeded"
            else f"webapp sync status is {webapp_sync.get('status') or 'missing'}",
            category="launch_gate",
        ),
        _check(
            "buyer_fulfillment_bundle_ready",
            launchable_bundle_ready,
            "launchable_export_bundle.json is ready for buyer fulfillment"
            if launchable_bundle_ready
            else "launchable_export_bundle.json is missing or not ready",
            category="launch_gate",
        ),
        _check(
            "native_runtime_capability_ready",
            str(runtime_capability.get("status") or "").strip().lower() == "ready",
            "native runtime capability is ready"
            if str(runtime_capability.get("status") or "").strip().lower() == "ready"
            else f"native runtime capability is blocked: {', '.join(runtime_capability.get('blockers') or []) or 'unknown'}",
            category="launch_gate",
        ),
        _check(
            "capturer_payout_transition_ready",
            payout_eligible,
            "capturer payout recommendation is present and payout-eligible"
            if payout_eligible
            else "capturer payout recommendation is missing or not payout-eligible",
            category="launch_gate",
        ),
    ]

    all_stage_checks_passed = all(item["passed"] for item in stage_checks)

    if all_stage_checks_passed and external_alpha_go:
        source_status = "external_beta_contract_ready"
    elif all_stage_checks_passed and internal_alpha_go:
        source_status = "internal_only_contract_ready"
    else:
        source_status = "blocked"

    justified_claims = [
        "Qualification and readiness remain enforced support gates; raw capture and package provenance remain authoritative.",
        "Privacy-safe walkthrough media is the buyer-facing artifact; runtime or world-model outputs stay downstream.",
    ]
    if all_stage_checks_passed:
        justified_claims.extend(
            [
                "Inbound request linkage, marketplace job linkage, upload completion, qualification, privacy processing, and WebApp sync are all contract-verified.",
                "Launchable export packaging exists for buyer fulfillment or buyer access flows.",
                "Capturer payout recommendation is contract-present; live Stripe/provider readiness remains an operator payment checklist item.",
            ]
        )
    if source_status == "external_beta_contract_ready":
        justified_claims.append(
            "This source path is externally marketable for the paid marketplace beta at contract level once operator checks pass."
        )
    elif source_status == "internal_only_contract_ready":
        justified_claims.append(
            "This source path is suitable for internal beta operations, qualification, privacy-safe previews, and workflow orchestration."
        )

    not_justified_claims = [
        "Do not claim runtime or world-model outputs can override raw capture, rights, privacy, provenance, or package truth.",
        "Do not claim strong site-faithful world-model quality; only native runtime capability and downstream packaging are proven here.",
        "Do not claim live buyer payments or live capturer payouts are proven until the operator payment checklist is completed.",
        "Do not claim Stripe, identity/KYC, background-check, instant-pay, or payout-timing readiness from backend URL, publishable key, or mocked tests.",
        "Do not claim real-device discovery and claim UX is proven in production until the device checklist is completed.",
    ]
    if not external_alpha_go:
        not_justified_claims.append(
            "Do not market this source as externally launch-ready while alpha readiness remains blocked."
        )

    operator_required_checks = [
        {
            "id": f"{descriptor.capture_source or 'unknown'}_real_device_claim_flow",
            "scope": "device",
            "required_evidence": "Screenshot or screen recording showing discovery, claim, and upload completion for the same capture_job_id.",
        },
        {
            "id": "buyer_payment_settlement",
            "scope": "payments",
            "required_evidence": "Stripe payment intent or checkout session proving a buyer purchase completed for the launch SKU.",
        },
        {
            "id": "capturer_payout_settlement",
            "scope": "payouts",
            "required_evidence": "Live Stripe connected account state, live payout evidence, webhook reconciliation, and matching creator capture ledger entry for the approved capture.",
        },
        {
            "id": "stripe_connected_account_live_readiness",
            "scope": "payouts",
            "required_evidence": "Backend /v1/stripe/account response showing provider_state_checked=true, provider_mode=live, live_provider_ready=true, payouts_enabled=true, and no blocking requirements.",
        },
        {
            "id": "payout_exception_monitor_live",
            "scope": "ops",
            "required_evidence": "Live monitor or query evidence for payout.failed, payout.canceled, disbursement_failed, and overdue finance_review records.",
        },
        {
            "id": "identity_kyc_provider_decision",
            "scope": "identity",
            "required_evidence": "Document whether Stripe Connect is the only near-term KYC path or provide account/env proof for Persona, Stripe Identity, or another identity provider.",
        },
        {
            "id": "background_check_provider_decision",
            "scope": "background_checks",
            "required_evidence": "Document that no Checkr/background-check provider is integrated yet, or provide provider account/env proof before making screening claims.",
        },
        {
            "id": "human_finance_review_owner",
            "scope": "ops",
            "required_evidence": "Named human finance owner and review queue/route for payout exceptions before any live payout execution flag is enabled.",
        },
        {
            "id": "buyer_artifact_access",
            "scope": "buyer_access",
            "required_evidence": "Authenticated buyer session proving artifact or fulfillment access resolves after purchase.",
        },
    ]

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "capture_source": descriptor.capture_source,
        "capture_modality": descriptor.capture_modality,
        "profile": profile,
        "overall_status": source_status,
        "device_alpha_profile": {
            "status": device_alpha_profile.get("status"),
            "reason": device_alpha_profile.get("reason"),
        },
        "runtime_capability": runtime_capability,
        "qualification_policy": {
            "authoritative_truth": True,
            "detail": "Raw capture, rights, privacy, provenance, and package artifacts are authoritative; qualification and readiness are enforced support gates.",
        },
        "stage_checks": stage_checks,
        "source_acceptance": {
            "status": source_status,
            "external_alpha_status": external_alpha.get("status"),
            "internal_alpha_status": internal_alpha.get("status"),
            "alpha_reason": external_alpha.get("reason") or internal_alpha.get("reason"),
        },
        "launch_claims": {
            "justified": justified_claims,
            "not_justified": not_justified_claims,
        },
        "operator_required_checks": operator_required_checks,
    }


def write_launch_gate_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload = build_launch_gate_summary(capture_root=capture_root, env=env)
    write_json(capture_root / "pipeline" / "launch_gate_summary.json", payload)
    return payload


def sync_webapp_evaluation_prep(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    resolved_env = dict(os.environ if env is None else env)
    descriptor_payload = _read_json_object(capture_root / "capture_descriptor.json")
    descriptor = CaptureDescriptor.from_dict(descriptor_payload)
    parsed = parse_gs_uri(str(descriptor.raw_prefix_uri))
    bucket = parsed.bucket
    pipeline_prefix = to_pipeline_prefix(descriptor.scene_id, descriptor.capture_id)
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    opportunity_handoff = _read_json_object(pipeline_root / "opportunity_handoff.json")
    qualification_record = _read_json_object(pipeline_root / "qualification_record.json")
    scorecard = _read_json_object(pipeline_root / "capture_qa_scorecard.json")
    privacy_manifest = _read_json_object(pipeline_root / "privacy_processing_manifest.json")
    provider_run_manifest = _read_json_object(pipeline_root / "provider_run_manifest.json")
    site_world_health = _read_json_object(eval_root / "site_world_health.json")
    evaluation_prep_summary = _read_json_object(eval_root / "evaluation_prep_summary.json")
    rights_provenance_review = optional_read_json(pipeline_root / "rights_provenance_review.json") or {}
    site_package_manifest = optional_read_json(eval_root / "site_package_manifest.json") or {}
    proof_pack_manifest = optional_read_json(eval_root / "proof_pack_manifest.json") or {}
    hosted_review_readiness = optional_read_json(eval_root / "hosted_review_readiness.json") or {}
    proof_path_status = optional_read_json(eval_root / "proof_path_status.json") or {}
    alpha_summary = write_alpha_readiness_summary(capture_root=capture_root, env=resolved_env)
    launch_gate_summary = write_launch_gate_summary(capture_root=capture_root, env=resolved_env)
    site_submission_id = (
        descriptor.site_submission_id
        or str(opportunity_handoff.get("site_submission_id") or "").strip()
    )
    buyer_request_id = (
        descriptor.buyer_request_id
        or str(opportunity_handoff.get("buyer_request_id") or "").strip()
    )
    capture_job_id = (
        descriptor.capture_job_id
        or str(opportunity_handoff.get("capture_job_id") or "").strip()
    )

    qualification_state = derive_webapp_qualification_state(
        readiness_state=qualification_record.get("readiness_state"),
        completeness_status=scorecard.get("completeness_status"),
    )
    opportunity_state = derive_webapp_opportunity_state(qualification_state=qualification_state)

    def _artifact_if_exists(relative_path: str) -> Optional[str]:
        path = pipeline_root / relative_path
        if path.is_file():
            return _uri(bucket, pipeline_prefix, relative_path)
        return None

    artifacts = {
        "qualification_summary_uri": _artifact_if_exists("qualification_summary.json"),
        "capture_quality_summary_uri": _artifact_if_exists("capture_quality_summary.json"),
        "rights_and_compliance_summary_uri": _artifact_if_exists("rights_and_compliance_summary.json"),
        "buyer_trust_score_uri": _artifact_if_exists("buyer_trust_score.json"),
        "capturer_payout_recommendation_uri": _artifact_if_exists("capturer_payout_recommendation.json"),
        "world_model_fit_summary_uri": _artifact_if_exists("world_model_fit_summary.json"),
        "provenance_summary_uri": _artifact_if_exists("provenance_summary.json"),
        "gemini_capture_fidelity_review_uri": _artifact_if_exists("gemini_capture_fidelity_review.json"),
        "privacy_processing_manifest_uri": _artifact_if_exists("privacy_processing_manifest.json"),
        "privacy_verification_report_uri": _artifact_if_exists("privacy_verification_report.json"),
        "webapp_sync_result_uri": _artifact_if_exists("webapp_sync_result.json"),
        "launch_gate_summary_uri": _artifact_if_exists("launch_gate_summary.json"),
        "preview_manifest_uri": _artifact_if_exists("preview_manifest.json"),
        "worldlabs_request_manifest_uri": _artifact_if_exists("worldlabs_request_manifest.json"),
        "worldlabs_operation_manifest_uri": _artifact_if_exists("worldlabs_operation_manifest.json"),
        "worldlabs_world_manifest_uri": _artifact_if_exists("worldlabs_world_manifest.json"),
        "scene_memory_manifest_uri": _artifact_if_exists("scene_memory/scene_memory_manifest.json"),
        "conditioning_bundle_uri": _artifact_if_exists("scene_memory/conditioning_bundle.json"),
        "preview_simulation_manifest_uri": _artifact_if_exists("preview_simulation/preview_simulation_manifest.json"),
        "presentation_bundle_uri": _artifact_if_exists("presentation_world/presentation_bundle.json"),
        "presentation_world_manifest_uri": _artifact_if_exists("presentation_world/presentation_world_manifest.json"),
        "runtime_demo_manifest_uri": _artifact_if_exists("presentation_world/runtime_demo_manifest.json"),
        "authoritative_runtime_render_manifest_uri": _artifact_if_exists(
            "presentation_world/authoritative_runtime_render_manifest.json"
        ),
        "site_world_spec_uri": _artifact_if_exists("evaluation_prep/site_world_spec.json"),
        "site_world_registration_uri": _artifact_if_exists("evaluation_prep/site_world_registration.json"),
        "site_world_health_uri": _artifact_if_exists("evaluation_prep/site_world_health.json"),
        "hosted_session_runtime_manifest_uri": _artifact_if_exists("evaluation_prep/hosted_session_runtime_manifest.json"),
        "launchable_export_bundle_uri": _artifact_if_exists("evaluation_prep/launchable_export_bundle.json"),
        "evaluation_prep_manifest_uri": _artifact_if_exists("evaluation_prep/evaluation_prep_manifest.json"),
        "evaluation_prep_summary_uri": _artifact_if_exists("evaluation_prep/evaluation_prep_summary.json"),
        "site_package_manifest_uri": _artifact_if_exists("evaluation_prep/site_package_manifest.json"),
        "proof_pack_manifest_uri": _artifact_if_exists("evaluation_prep/proof_pack_manifest.json"),
        "hosted_review_readiness_uri": _artifact_if_exists("evaluation_prep/hosted_review_readiness.json"),
        "proof_path_status_uri": _artifact_if_exists("evaluation_prep/proof_path_status.json"),
        "rights_provenance_review_uri": _artifact_if_exists("rights_provenance_review.json"),
        "geometry_manifest_uri": _artifact_if_exists("geometry/geometry_manifest.json"),
        "geometry_summary_uri": _artifact_if_exists("geometry/geometry_summary.json"),
        "privacy_depth_manifest_uri": _artifact_if_exists("privacy_depth/depth_manifest.json"),
        "privacy_confidence_manifest_uri": _artifact_if_exists("privacy_depth/confidence_manifest.json"),
        "alpha_readiness_summary_uri": _artifact_if_exists("alpha_readiness_summary.json"),
        "worldlabs_launch_url": _present_value(
            provider_run_manifest,
            "worldlabs_launch_url",
            "preview_launch_url",
            "launch_url",
        ),
        "privacy_processed_video_uri": _present_value(privacy_manifest, "privacy_processed_video_uri"),
        "world_model_video_uri": _present_value(privacy_manifest, "world_model_video_uri"),
    }
    derived_assets = {
        key: value
        for key, value in {
            "scene_memory": {
                "status": str(_read_json_object(pipeline_root / "scene_memory" / "scene_memory_readiness.json").get("status") or "missing"),
                "manifest_uri": artifacts.get("scene_memory_manifest_uri"),
                "artifact_uri": artifacts.get("conditioning_bundle_uri"),
            }
            if artifacts.get("scene_memory_manifest_uri")
            else None,
            "presentation_world": {
                "status": str(_read_json_object(pipeline_root / "presentation_world" / "presentation_world_manifest.json").get("status") or "missing"),
                "manifest_uri": artifacts.get("presentation_world_manifest_uri"),
                "artifact_uri": artifacts.get("presentation_bundle_uri"),
            }
            if artifacts.get("presentation_world_manifest_uri")
            else None,
            "site_world_package": {
                "status": str(evaluation_prep_summary.get("site_world_status") or site_world_health.get("status") or "missing"),
                "manifest_uri": artifacts.get("evaluation_prep_manifest_uri"),
                "artifact_uri": artifacts.get("site_world_spec_uri"),
            }
            if artifacts.get("site_world_spec_uri")
            else None,
            "hosted_runtime": {
                "status": str(site_world_health.get("status") or "missing"),
                "manifest_uri": artifacts.get("hosted_session_runtime_manifest_uri"),
                "artifact_uri": artifacts.get("site_world_registration_uri"),
            }
            if artifacts.get("hosted_session_runtime_manifest_uri")
            else None,
        }.items()
        if value
    }
    deployment_readiness = {
        "capture_source": descriptor.capture_source,
        "capture_modality": descriptor.capture_modality,
        "device_alpha_profile_status": alpha_summary.get("device_alpha_profile", {}).get("status"),
        "device_alpha_profile_reason": alpha_summary.get("device_alpha_profile", {}).get("reason"),
        "qualification_state": qualification_state,
        "opportunity_state": opportunity_state,
        "native_world_model_status": str(
            evaluation_prep_summary.get("native_world_model_status")
            or ("primary_ready" if artifacts.get("site_world_spec_uri") and artifacts.get("scene_memory_manifest_uri") else "not_ready")
        ),
        "native_world_model_primary": bool(
            evaluation_prep_summary.get("native_world_model_primary")
            if evaluation_prep_summary.get("native_world_model_primary") is not None
            else artifacts.get("site_world_spec_uri") and artifacts.get("scene_memory_manifest_uri")
        ),
        "provider_fallback_preview_status": (
            str(evaluation_prep_summary.get("provider_fallback_preview_status"))
            if evaluation_prep_summary.get("provider_fallback_preview_status") is not None
            else "fallback_available"
            if artifacts.get("preview_simulation_manifest_uri") or artifacts.get("world_model_video_uri")
            else "not_requested"
        ),
        "provider_fallback_only": bool(
            evaluation_prep_summary.get("provider_fallback_only")
            if evaluation_prep_summary.get("provider_fallback_only") is not None
            else not bool(
                evaluation_prep_summary.get("native_world_model_primary")
                if evaluation_prep_summary.get("native_world_model_primary") is not None
                else artifacts.get("site_world_spec_uri") and artifacts.get("scene_memory_manifest_uri")
            )
            and bool(artifacts.get("preview_simulation_manifest_uri") or artifacts.get("world_model_video_uri"))
        ),
        "runtime_health_status": site_world_health.get("status"),
        "runtime_launchable": bool(site_world_health.get("launchable")),
        "runtime_registration_status": site_world_health.get("runtime_registration_status"),
        "native_runtime_capability_state": alpha_summary.get("runtime_capability", {}).get("status"),
        "native_runtime_capability": alpha_summary.get("runtime_capability"),
        "evaluation_prep_summary": evaluation_prep_summary,
        "alpha_readiness": alpha_summary,
        "launch_gate_summary": launch_gate_summary,
        "rights_provenance_review": rights_provenance_review,
        "site_package_manifest": site_package_manifest,
        "proof_pack_manifest": proof_pack_manifest,
        "hosted_review_readiness": hosted_review_readiness,
        "proof_path_status": proof_path_status,
        "proof_path_events": proof_path_status.get("event_statuses", []),
    }

    try:
        result = sync_webapp_pipeline_attachment(
            site_submission_id=site_submission_id,
            request_id=site_submission_id,
            buyer_request_id=buyer_request_id,
            capture_job_id=capture_job_id,
            scene_id=descriptor.scene_id,
            capture_id=descriptor.capture_id,
            pipeline_prefix=pipeline_prefix,
            qualification_state=qualification_state,
            opportunity_state=opportunity_state,
            authoritative_state_update=True,
            artifacts={str(key): value for key, value in artifacts.items() if value},
            derived_assets=derived_assets,
            deployment_readiness=deployment_readiness,
        )
    except (WebappSyncError, ValueError) as exc:
        result = {
            "status": "failed",
            "reason": str(exc),
            "blocker": "webapp_sync_requires_upstream_request_job_bootstrap",
        }
    return write_pipeline_sync_result(
        pipeline_root=pipeline_root,
        stage="evaluation_prep",
        result=result,
    )
