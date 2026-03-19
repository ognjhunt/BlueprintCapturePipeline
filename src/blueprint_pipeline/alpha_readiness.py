"""Alpha-readiness validation and downstream WebApp sync helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import parse_bool, parse_gs_uri, read_json, to_pipeline_prefix, utc_now_iso, write_json
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
    privacy_verification = _read_json_object(pipeline_root / "privacy_verification_report.json")
    webapp_sync = _read_json_object(pipeline_root / "webapp_sync_result.json")
    scene_memory_manifest = _read_json_object(scene_memory_root / "scene_memory_manifest.json")
    conditioning_bundle = _read_json_object(scene_memory_root / "conditioning_bundle.json")
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
    if str(resolved_env.get("BLUEPRINT_PREVIEW_PROVIDER") or "").strip() == "world_labs":
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
            in {"no_people_detected", "person_removed", "face_anonymized_fallback"},
            "privacy produced buyer-safe walkthrough media"
            if str(privacy_manifest.get("status") or "").strip().lower()
            in {"no_people_detected", "person_removed", "face_anonymized_fallback"}
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
        ]
        external_alpha = {
            "status": "no_go",
            "reason": "meta_glasses_remains_internal_experimental_for_site_faithful_world_model_claims",
        }
        internal_alpha = {
            "status": "go" if common_passed and all(item["passed"] for item in path_checks) else "no_go",
            "reason": "all_common_and_glasses_checks_passed"
            if common_passed and all(item["passed"] for item in path_checks)
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
        ]
        external_alpha = {
            "status": "no_go",
            "reason": "android_remains_internal_until_parity_thresholds_and_marketing_truth_are_met",
        }
        internal_alpha = {
            "status": "go" if common_passed and all(item["passed"] for item in path_checks) else "no_go",
            "reason": "all_common_and_android_checks_passed"
            if common_passed and all(item["passed"] for item in path_checks)
            else "android_internal_alpha_requirements_not_met",
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
        "environment_checks": env_checks,
        "common_checks": common_checks,
        "path_checks": path_checks,
        "verdicts": {
            "external_alpha": external_alpha,
            "internal_experimental_alpha": internal_alpha,
        },
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
    site_world_health = _read_json_object(eval_root / "site_world_health.json")
    evaluation_prep_summary = _read_json_object(eval_root / "evaluation_prep_summary.json")
    alpha_summary = write_alpha_readiness_summary(capture_root=capture_root, env=resolved_env)

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
        "world_model_fit_summary_uri": _artifact_if_exists("world_model_fit_summary.json"),
        "provenance_summary_uri": _artifact_if_exists("provenance_summary.json"),
        "gemini_capture_fidelity_review_uri": _artifact_if_exists("gemini_capture_fidelity_review.json"),
        "privacy_processing_manifest_uri": _artifact_if_exists("privacy_processing_manifest.json"),
        "privacy_verification_report_uri": _artifact_if_exists("privacy_verification_report.json"),
        "webapp_sync_result_uri": _artifact_if_exists("webapp_sync_result.json"),
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
        "geometry_manifest_uri": _artifact_if_exists("geometry/geometry_manifest.json"),
        "geometry_summary_uri": _artifact_if_exists("geometry/geometry_summary.json"),
        "privacy_depth_manifest_uri": _artifact_if_exists("privacy_depth/depth_manifest.json"),
        "privacy_confidence_manifest_uri": _artifact_if_exists("privacy_depth/confidence_manifest.json"),
        "alpha_readiness_summary_uri": _artifact_if_exists("alpha_readiness_summary.json"),
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
        "evaluation_prep_summary": evaluation_prep_summary,
        "alpha_readiness": alpha_summary,
    }

    result = sync_webapp_pipeline_attachment(
        site_submission_id=opportunity_handoff.get("site_submission_id") or descriptor.capture_id,
        request_id=opportunity_handoff.get("site_submission_id") or descriptor.capture_id,
        buyer_request_id=descriptor.buyer_request_id or opportunity_handoff.get("site_submission_id") or descriptor.capture_id,
        capture_job_id=descriptor.capture_job_id or descriptor.capture_id,
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
    return write_pipeline_sync_result(
        pipeline_root=pipeline_root,
        stage="evaluation_prep",
        result=result,
    )
