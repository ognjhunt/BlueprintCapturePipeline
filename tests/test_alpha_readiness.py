from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.alpha_readiness import (
    build_alpha_readiness_summary,
    build_launch_gate_summary,
    validate_operator_launch_evidence,
)
from blueprint_pipeline.evaluation_prep_stage import run_evaluation_prep_stage
from blueprint_pipeline.geometry_stage import build_geometry_stage_contract
from blueprint_pipeline.materialization import materialize_capture_bundle


def _rehash_raw_bundle(raw_root: Path) -> None:
    artifacts = {
        path.relative_to(raw_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_root.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }
    canonical = "\n".join(f"{name}:{artifacts[name]}" for name in sorted(artifacts))
    (raw_root / "hashes.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "bundle_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _successful_capture_review() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "review_type": "gemini_multimodal_capture_review",
        "status": "succeeded",
        "generated_at": "2026-03-15T00:00:00+00:00",
        "provider_name": "gemini",
        "provider_model": "gemini-2.5-pro",
        "review_mode": "video_primary_frames_fallback",
        "confidence": 0.9,
        "summary": "Capture supports downstream work.",
        "scores": {
            "coverage": 0.9,
            "visual_clarity": 0.86,
            "lighting_stability": 0.84,
            "motion_stability": 0.82,
            "task_understanding": 0.86,
            "world_model_fitness": 0.84,
            "payout_quality": 0.8,
        },
        "bonus_signals": {
            "complete_coverage": {"score": 0.9, "reason": "Coverage is complete."},
            "multi_pass": {"score": 0.7, "reason": "Multiple views are present."},
            "lidar_depth": {"score": 1.0, "reason": "Depth-backed capture quality is strong."},
            "steady_walkthrough": {"score": 0.85, "reason": "The walkthrough is steady."},
        },
        "findings": {
            "missing_views": [],
            "blur_observations": [],
            "lighting_observations": [],
            "occlusion_observations": [],
            "task_scope_notes": [],
            "blocker_summaries": [],
            "recapture_recommendations": [],
        },
        "recommendations": {
            "world_model_recommendation": "good_candidate",
            "payout_recommendation": "baseline",
        },
        "provenance": {"provider_name": "gemini", "provider_model": "gemini-2.5-pro"},
    }


class _HealthyRuntimeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def register_site_world_package(self, *, spec, registration, health):  # type: ignore[no-untyped-def]
        site_world_id = str(registration.get("site_world_id") or "siteworld-test")
        return {
            **dict(registration),
            "schema_version": "v1",
            "status": "ready",
            "site_world_id": site_world_id,
            "runtime_base_url": "http://runtime.test",
            "websocket_base_url": "ws://runtime.test",
            "runtime_capabilities": {
                "supports_step_rollout": True,
                "supports_batch_rollout": True,
                "supports_camera_views": True,
                "supports_rlds_export": True,
                "supports_preview_render": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            "health": {
                **dict(health),
                "schema_version": "v1",
                "site_world_id": site_world_id,
                "healthy": True,
                "launchable": True,
                "status": "healthy",
                "blockers": [],
                "warnings": [],
            },
        }

    def get_site_world_health(self, site_world_id: str):  # type: ignore[no-untyped-def]
        return {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "healthy": True,
            "launchable": True,
            "status": "healthy",
            "blockers": [],
            "warnings": [],
        }

    def create_session(self, site_world_id: str, **_kwargs):  # type: ignore[no-untyped-def]
        return {"site_world_id": site_world_id, "session_id": "session-test"}

    def reset_session(self, _session_id: str, **_kwargs):  # type: ignore[no-untyped-def]
        return {"status": "ok"}


def _write_backend_script(path: Path) -> None:
    body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[2])
objects = [
    {
        "id": "cabinet_0001",
        "object_id": "cabinet_0001",
        "label": "cabinet",
        "boundingBox": {
            "center": [0.0, 0.0, 0.75],
            "extents": [0.8, 0.45, 0.9],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "mean_confidence": 0.95,
        "confidence": 0.95,
        "n_total_detections": 3,
        "n_frame_detections": 2,
        "reference_crop": "",
        "all_crops": [],
        "task_relevance": {"score": 0.9, "matched_terms": ["cabinet"]},
        "articulation_hints": {"interactive": True, "kind": "cabinet", "confidence": 0.82},
        "evidence_frames": [0, 1],
        "source_prompts": ["cabinet"],
        "provenance": {"grounding_level": "observed", "canonical_truth": True},
        "mean_box_px": {"area": 64000.0, "width": 240.0, "height": 260.0},
    }
]
output_path.write_text(json.dumps({"backend_status": "ok", "objects": objects}, indent=2), encoding="utf-8")
"""
    path.write_text(body, encoding="utf-8")


def _set_alpha_env(monkeypatch, tmp_path: Path, *, include_runtime: bool, include_video_to_world: bool = False) -> None:  # type: ignore[no-untyped-def]
    values = {
        "PIPELINE_PROJECT_ID": "alpha-project",
        "PIPELINE_REGION": "us-central1",
        "PIPELINE_BUCKET": "local-blueprint",
        "GCS_ROOT": str(tmp_path),
        "PIPELINE_SYNC_WEBAPP_URL": "https://webapp.test/api/pipeline-sync",
        "PIPELINE_SYNC_TOKEN": "sync-token",
        "PIPELINE_SYNC_REQUIRED": "true",
        "GOOGLE_GENAI_API_KEY": "gemini-key",
        "WORLDLABS_API_KEY": "worldlabs-key",
        "PRIVACY_PIPELINE_ENABLED": "true",
        "PRIVACY_FAIL_CLOSED": "true",
        "PRIVACY_RUNNER_TOKEN": "privacy-token",
        "PRIVACY_SAM3_URL": "https://privacy.test/sam3",
        "PRIVACY_VIP_URL": "https://privacy.test/vip",
        "PRIVACY_DEEPPRIVACY2_URL": "https://privacy.test/deepprivacy2",
    }
    if include_runtime:
        values["SITE_WORLD_RUNTIME_SERVICE_URL"] = "http://runtime.test"
        values["SITE_WORLD_RUNTIME_SERVICE_API_KEY"] = "runtime-token"
    if include_video_to_world:
        values["VIDEO_TO_WORLD_URL"] = "https://vtw.test"
        values["VIDEO_TO_WORLD_RUNNER_TOKEN"] = "vtw-token"
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _stub_sync(monkeypatch, sync_calls: list[dict[str, object]]) -> None:  # type: ignore[no-untyped-def]
    def _sync(**kwargs):  # type: ignore[no-untyped-def]
        sync_calls.append(kwargs)
        # Mirror the fields the real sync always emits so the launch gate's
        # sync-truth verification sees the same contract. capture_root is an
        # internal consent-gate input, not part of the webapp attachment payload
        # (the real builder never echoes it), so exclude it here too.
        attachment_payload = {k: v for k, v in kwargs.items() if k != "capture_root"}
        attachment_payload["upstream_links_verified"] = True
        attachment_payload["placeholder_fallback_allowed"] = False
        return {
            "status": "succeeded",
            "attempts": 1,
            "response": {"ok": True},
            "attachment_payload": attachment_payload,
            "evaluation_readiness": kwargs.get("evaluation_readiness"),
        }

    monkeypatch.setattr("blueprint_pipeline.qualification.sync_webapp_pipeline_attachment", _sync)
    monkeypatch.setattr("blueprint_pipeline.alpha_readiness.sync_webapp_pipeline_attachment", _sync)


def _write_privacy_outputs(
    capture_root: Path,
    *,
    depth_source: str,
    include_depth_manifests: bool,
) -> dict[str, object]:
    pipeline_root = capture_root / "pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    prefix = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline"
    depth_conditioning: dict[str, object] = {
        "status": "available",
        "source": depth_source,
        "provider": depth_source,
        "depth_prefix_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/privacy/depth",
        "confidence_prefix_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/privacy/confidence",
        "depth_manifest_uri": None,
        "confidence_manifest_uri": None,
    }
    if include_depth_manifests:
        privacy_depth_root = pipeline_root / "privacy_depth"
        privacy_depth_root.mkdir(parents=True, exist_ok=True)
        (privacy_depth_root / "depth_manifest.json").write_text(
            json.dumps({"schema_version": "v1", "status": "available"}),
            encoding="utf-8",
        )
        (privacy_depth_root / "confidence_manifest.json").write_text(
            json.dumps({"schema_version": "v1", "status": "available"}),
            encoding="utf-8",
        )
        depth_conditioning["depth_manifest_uri"] = f"{prefix}/privacy_depth/depth_manifest.json"
        depth_conditioning["confidence_manifest_uri"] = f"{prefix}/privacy_depth/confidence_manifest.json"

    payload = {
        "schema_version": "v1",
        "status": "person_removed",
        "mode": "removal",
        "fallback_used": False,
        "people_detected": 1,
        "people_removed": 1,
        "face_anonymized_segments": [],
        "raw_retained": True,
        "fail_closed": True,
        "depth_source": depth_source,
        "depth_conditioning": depth_conditioning,
        "privacy_processed_video_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/privacy/final_walkthrough.mov",
        "world_model_video_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/privacy/final_walkthrough.mov",
        "privacy_manifest_uri": f"{prefix}/privacy_processing_manifest.json",
        "privacy_verification_report_uri": f"{prefix}/privacy_verification_report.json",
    }
    (pipeline_root / "privacy_processing_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (pipeline_root / "privacy_verification_report.json").write_text(
        json.dumps({"schema_version": "v1", "status": "passed"}),
        encoding="utf-8",
    )
    return payload


def _write_geometry_lane(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _fake_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        frames_dir = geometry_root / "frames" / "images"
        depth_dir = geometry_root / "depth"
        confidence_dir = geometry_root / "confidence"
        frames_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for frame_index in range(2):
            image_path = frames_dir / f"frame_{frame_index:06d}.npy"
            np.save(image_path, np.full((12, 18, 3), 80, dtype=np.float32))
            depth_path = depth_dir / f"depth_{frame_index:06d}.npy"
            confidence_path = confidence_dir / f"confidence_{frame_index:06d}.npy"
            np.save(depth_path, np.full((12, 18), 1.5, dtype=np.float32))
            np.save(confidence_path, np.full((12, 18), 0.8, dtype=np.float32))
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": float(frame_index) * 0.4,
                    "image_path": str(image_path),
                    "is_keyframe": True,
                    "blur_score": 0.1,
                    "overlap_hint": 0.9,
                    "world_from_camera": [
                        [1.0, 0.0, 0.0, frame_index * 0.15],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "camera_from_world": [
                        [1.0, 0.0, 0.0, -(frame_index * 0.15)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "pose_confidence": 0.9,
                    "depth_path": str(depth_path),
                    "depth_format": "npy",
                    "confidence_path": str(confidence_path),
                    "confidence_format": "npy",
                    "width": 18,
                    "height": 12,
                    "min_depth_m": 1.5,
                    "max_depth_m": 1.5,
                    "depth_unit": "meters",
                    "metric_depth_truth": True,
                    "depth_measurement_source": "provider_metric_reconstruction",
                    "confidence_range": [0.0, 1.0],
                }
            )
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 18,
                "image_height": 12,
                "fx": 16.0,
                "fy": 16.0,
                "cx": 9.0,
                "cy": 6.0,
                "distortion": {"model": "none", "coefficients": []},
            },
            "frames": frames,
            "provider_metrics": {"backend": "test"},
            "provider_native_result": True,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 0.9,
            "p95_pose_delta_sec": 0.033,
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _fake_provider)


def _operator_launch_evidence_fields(check_id: str) -> dict[str, object]:
    base_uri = f"gs://local-blueprint/operator-evidence/{check_id}.json"
    if check_id == "legal_consent_posture_signoff":
        return {"signed_record_uri": base_uri}
    if check_id == "operator_dpa_data_processing_terms":
        return {
            "signed_record_uri": base_uri,
            "document_uri": "gs://local-blueprint/operator-evidence/operator-dpa.pdf",
            "retention_policy_uri": "gs://local-blueprint/operator-evidence/beta-retention-policy.pdf",
            "subprocessor_list_uri": "gs://local-blueprint/operator-evidence/subprocessors.pdf",
            "subprocessors": [
                {
                    "name": "Google Cloud",
                    "service_scope": "storage_and_pipeline_runtime",
                }
            ],
            "access_audit_terms_uri": "gs://local-blueprint/operator-evidence/access-audit-terms.pdf",
            "access_audit_report_uri": "gs://local-blueprint/operator-evidence/access-audit-log.json",
        }
    if check_id == "cross_border_data_residency_posture":
        return {
            "data_residency_policy_uri": "gs://local-blueprint/operator-evidence/data-residency-policy.pdf",
            "us_only_beta_scope_uri": "gs://local-blueprint/operator-evidence/us-only-beta-scope.pdf",
            "allowed_tester_countries": ["US"],
            "allowed_site_countries": ["US"],
            "non_us_participants_blocked": True,
        }
    if check_id == "industrial_site_authorization_ehs_signoff":
        return {
            "signed_record_uri": base_uri,
            "industrial_authorization_record_uri": base_uri,
            "site_authorizer_name": "Plant Manager",
            "site_authorizer_role": "plant_manager",
            "ehs_signoff_uri": "gs://local-blueprint/operator-evidence/ehs-signoff.pdf",
            "worker_pii_consent_posture_uri": (
                "gs://local-blueprint/operator-evidence/worker-pii-posture.pdf"
            ),
            "nda_or_proprietary_data_terms_uri": (
                "gs://local-blueprint/operator-evidence/industrial-nda.pdf"
            ),
            "ppe_requirements_acknowledged": True,
            "escort_requirements_acknowledged": True,
            "restricted_zone_controls_uri": (
                "gs://local-blueprint/operator-evidence/restricted-zones.pdf"
            ),
        }
    if check_id == "paperclip_ops_relay_secret_rotation":
        return {
            "secret_version_ref": "projects/blueprint/secrets/paperclip-ops-relay/versions/7",
            "redeploy_evidence_uri": base_uri,
        }
    if check_id.endswith("_real_device_claim_flow"):
        return {
            "recording_uri": base_uri,
            "capture_job_id": "capture-job-live-123",
        }
    if check_id == "buyer_payment_settlement":
        return {
            "payment_intent_id": "pi_live_123",
            "stripe_event_id": "evt_live_payment_123",
            "stripe_mode": "live",
        }
    if check_id == "capturer_payout_settlement":
        return {
            "payout_id": "po_live_123",
            "transfer_id": "tr_live_123",
            "webhook_reconciliation_uri": base_uri,
            "creator_payout_ledger_ref": "creatorPayouts/live-123",
            "stripe_mode": "live",
        }
    if check_id == "stripe_connected_account_live_readiness":
        return {
            "provider_account_ref": "acct_live_123",
            "provider_state_checked": True,
            "provider_mode": "live",
            "live_provider_ready": True,
            "payouts_enabled": True,
            "blocking_requirements": [],
        }
    if check_id == "payout_exception_monitor_live":
        return {"monitor_uri": base_uri, "alert_policy_uri": "projects/blueprint/alertPolicies/payouts"}
    if check_id in {"identity_kyc_provider_decision", "background_check_provider_decision"}:
        return {"decision_record_uri": base_uri}
    if check_id == "human_finance_review_owner":
        return {"finance_owner": "ops-owner", "review_queue_uri": "https://ops.example/finance-review"}
    if check_id == "buyer_artifact_access":
        return {
            "buyer_session_ref": "buyer-session-live-123",
            "artifact_access_log_uri": base_uri,
            "authenticated_fetch_status": "succeeded",
        }
    return {"evidence_uri": base_uri}


def _write_operator_launch_evidence(capture_root: Path, required_checks: list[dict[str, object]]) -> None:
    checks = {
        str(check["id"]): {
            "status": "verified",
            "evidence_uri": f"gs://local-blueprint/operator-evidence/{check['id']}.json",
            "verified_at": "2026-07-04T00:00:00+00:00",
            "verified_by": "ops-owner",
            **_operator_launch_evidence_fields(str(check["id"])),
        }
        for check in required_checks
    }
    (capture_root / "pipeline" / "operator_launch_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "operator_launch_evidence.v1",
                "checks": checks,
            }
        ),
        encoding="utf-8",
    )


def _build_capture(
    tmp_path: Path,
    *,
    capture_source: str,
    capture_modality: str,
    include_arkit: bool | None = None,
    site_type: str | None = None,
) -> tuple[Path, str]:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    arkit_enabled = include_arkit if include_arkit is not None else capture_modality == "iphone_arkit_lidar"

    manifest_payload = {
        "schema_version": "v3",
        "capture_schema_version": "3.0.0",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": "walkthrough.mov",
        "capture_source": capture_source,
        "capture_tier_hint": "tier1_iphone" if capture_source == "iphone" else "video_only",
        "capture_profile_id": capture_modality,
        "capture_capabilities": {
            "camera_pose": arkit_enabled,
            "camera_intrinsics": arkit_enabled,
            "depth": arkit_enabled,
        },
        "coordinate_frame_session_id": f"cfs-{capture_source}-fixture",
        "capture_start_epoch_ms": 1_700_000_000_000,
        "app_version": "1.0.0-test",
        "app_build": "1",
        "ios_version": "18.0-test",
        "ios_build": "22A-test",
        "hardware_model_identifier": f"{capture_source}-fixture-device",
        "device_model_marketing": f"{capture_source} fixture device",
        "depth_supported": arkit_enabled,
        "rights_profile": "documented_permission",
        "capture_job_id": f"capture-job-{capture_source}",
        "buyer_request_id": f"buyer-request-{capture_source}",
        "site_submission_id": f"site-submission-{capture_source}",
        "width": 1920,
        "height": 1080,
        "fps_source": 30.0,
        "has_lidar": arkit_enabled,
        "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        "quoted_payout_cents": 6500,
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "data_licensing_allowed": False,
            "capture_contributor_payout_eligible": True,
            "consent_status": "documented",
            "permission_document_uri": "gs://local-blueprint/rights/consent-packet.pdf",
            "consent_scope": ["zone-a", "derived_generation", "robot_evaluation"],
            "consent_notes": [],
        },
    }
    if site_type:
        manifest_payload["site_type"] = site_type
        manifest_payload["intended_space_type"] = site_type
    (raw_root / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Open cabinet",
                "taskSteps": ["Walk to cabinet", "Open cabinet"],
                "zone": "cabinet zone",
                "owner": "ops",
            }
        ),
        encoding="utf-8",
    )
    context_payload = {
        "sceneId": scene_id,
        "captureId": capture_id,
        "captureSource": capture_source,
        "captureModality": capture_modality,
    }
    if capture_source == "iphone":
        context_payload["captureOrientation"] = {
            "displayOrientation": "portrait",
            "displayWidth": 1080,
            "displayHeight": 1920,
            "rotationDegrees": 90,
        }
    (raw_root / "capture_context.json").write_text(json.dumps(context_payload), encoding="utf-8")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "sceneId": scene_id,
                "captureId": capture_id,
                "raw_prefix": f"scenes/{scene_id}/captures/{capture_id}/raw",
                "completed_at": "2026-07-09T12:00:00Z",
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    if capture_source == "iphone" and arkit_enabled:
        arkit_root = raw_root / "arkit"
        (arkit_root / "depth").mkdir(parents=True)
        (arkit_root / "poses.jsonl").write_text(
            json.dumps({"frame_id": "000001", "t_device_sec": 0.0, "T_world_camera": np.eye(4).tolist()}) + "\n",
            encoding="utf-8",
        )
        (arkit_root / "intrinsics.json").write_text(
            json.dumps({"width": 1920, "height": 1080, "fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}),
            encoding="utf-8",
        )
        (arkit_root / "depth" / "000001.png").write_bytes(b"depth")

    _rehash_raw_bundle(raw_root)

    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor_payload.update(
        {
            "capture_job_id": manifest_payload["capture_job_id"],
            "buyer_request_id": manifest_payload["buyer_request_id"],
            "site_submission_id": manifest_payload["site_submission_id"],
            "quoted_payout_cents": manifest_payload["quoted_payout_cents"],
        }
    )
    descriptor_path.write_text(json.dumps(descriptor_payload), encoding="utf-8")
    return capture_root, str(materialized["descriptor_uri"])


def test_launch_gate_requires_buyer_request_id(tmp_path: Path) -> None:
    capture_root, _descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_arkit_lidar",
    )
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor_payload["buyer_request_id"] = ""
    descriptor_path.write_text(json.dumps(descriptor_payload), encoding="utf-8")
    pipeline_root = capture_root / "pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    (pipeline_root / "opportunity_handoff.json").write_text(
        json.dumps(
            {
                "site_submission_id": "site-submission-iphone",
                "capture_job_id": "capture-job-iphone",
            }
        ),
        encoding="utf-8",
    )

    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = {check["name"]: check for check in summary["stage_checks"]}

    assert summary["overall_status"] == "blocked"
    assert checks["buyer_request_linked"]["passed"] is False
    assert checks["buyer_request_linked"]["detail"] == "buyer_request_id is missing from the buyer request linkage"


def test_launch_gate_requires_industrial_authorization_for_warehouse_capture(
    tmp_path: Path,
) -> None:
    capture_root, _descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_arkit_lidar",
        site_type="warehouse",
    )

    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    required = {check["id"]: check for check in summary["operator_required_checks"]}

    assert summary["source_acceptance"]["industrial_authorization_required"] is True
    assert "warehouse" in summary["source_acceptance"]["industrial_site_type_candidates"]
    assert "industrial_site_authorization_ehs_signoff" in required
    assert required["industrial_site_authorization_ehs_signoff"]["passed"] is False
    assert "industrial_site_authorization_ehs_signoff_evidence_missing_or_unverified" in (
        summary["operator_evidence_status"]["blockers"]
    )


def test_industrial_authorization_evidence_requires_specific_legal_ehs_fields() -> None:
    incomplete = {
        "schema_version": "operator_launch_evidence.v1",
        "checks": {
            "industrial_site_authorization_ehs_signoff": {
                "status": "verified",
                "evidence_uri": "gs://local-blueprint/operator-evidence/industrial.json",
                "verified_at": "2026-07-04T00:00:00+00:00",
                "verified_by": "ops-owner",
            }
        },
    }
    blocked = validate_operator_launch_evidence(
        incomplete,
        ["industrial_site_authorization_ehs_signoff"],
    )

    assert blocked["status"] == "blocked"
    failures = blocked["checks"][0]["evidence_validation_errors"]
    assert "missing_industrial_site_authorization_record" in failures
    assert "missing_ehs_safety_signoff" in failures
    assert "missing_nda_or_proprietary_data_terms" in failures

    complete = {
        "schema_version": "operator_launch_evidence.v1",
        "checks": {
            "industrial_site_authorization_ehs_signoff": {
                "status": "verified",
                "evidence_uri": "gs://local-blueprint/operator-evidence/industrial.json",
                "verified_at": "2026-07-04T00:00:00+00:00",
                "verified_by": "ops-owner",
                **_operator_launch_evidence_fields("industrial_site_authorization_ehs_signoff"),
            }
        },
    }
    verified = validate_operator_launch_evidence(
        complete,
        ["industrial_site_authorization_ehs_signoff"],
    )

    assert verified["status"] == "verified"
    assert verified["remaining_ids"] == []


def test_operator_dpa_evidence_requires_subprocessor_and_access_audit_terms() -> None:
    incomplete = {
        "schema_version": "operator_launch_evidence.v1",
        "checks": {
            "operator_dpa_data_processing_terms": {
                "status": "verified",
                "signed_record_uri": "gs://local-blueprint/operator-evidence/dpa.pdf",
                "verified_at": "2026-07-04T00:00:00+00:00",
                "verified_by": "ops-owner",
            }
        },
    }
    blocked = validate_operator_launch_evidence(
        incomplete,
        ["operator_dpa_data_processing_terms"],
    )

    assert blocked["status"] == "blocked"
    failures = blocked["checks"][0]["evidence_validation_errors"]
    assert "missing_retention_policy_terms" in failures
    assert "missing_subprocessor_list" in failures
    assert "missing_access_audit_terms" in failures

    complete = {
        "schema_version": "operator_launch_evidence.v1",
        "checks": {
            "operator_dpa_data_processing_terms": {
                "status": "verified",
                "evidence_uri": "gs://local-blueprint/operator-evidence/operator-dpa-record.json",
                "verified_at": "2026-07-04T00:00:00+00:00",
                "verified_by": "ops-owner",
                **_operator_launch_evidence_fields("operator_dpa_data_processing_terms"),
            }
        },
    }
    verified = validate_operator_launch_evidence(
        complete,
        ["operator_dpa_data_processing_terms"],
    )

    assert verified["status"] == "verified"
    assert verified["remaining_ids"] == []


def test_cross_border_residency_evidence_requires_us_scope_or_transfer_terms() -> None:
    incomplete = {
        "schema_version": "operator_launch_evidence.v1",
        "checks": {
            "cross_border_data_residency_posture": {
                "status": "verified",
                "data_residency_policy_uri": "gs://local-blueprint/operator-evidence/residency.pdf",
                "verified_at": "2026-07-04T00:00:00+00:00",
                "verified_by": "ops-owner",
            }
        },
    }
    blocked = validate_operator_launch_evidence(
        incomplete,
        ["cross_border_data_residency_posture"],
    )

    assert blocked["status"] == "blocked"
    assert "missing_us_only_scope_or_signed_transfer_terms" in (
        blocked["checks"][0]["evidence_validation_errors"]
    )

    complete = {
        "schema_version": "operator_launch_evidence.v1",
        "checks": {
            "cross_border_data_residency_posture": {
                "status": "verified",
                "evidence_uri": "gs://local-blueprint/operator-evidence/residency-record.json",
                "verified_at": "2026-07-04T00:00:00+00:00",
                "verified_by": "ops-owner",
                **_operator_launch_evidence_fields("cross_border_data_residency_posture"),
            }
        },
    }
    verified = validate_operator_launch_evidence(
        complete,
        ["cross_border_data_residency_posture"],
    )

    assert verified["status"] == "verified"
    assert verified["remaining_ids"] == []


def test_launch_gate_does_not_treat_blocked_bundle_file_as_ready(tmp_path: Path) -> None:
    capture_root, _descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_arkit_lidar",
    )
    eval_root = capture_root / "pipeline" / "evaluation_prep"
    eval_root.mkdir(parents=True, exist_ok=True)
    (eval_root / "launchable_export_bundle.json").write_text(
        json.dumps({"status": "blocked", "blockers": ["missing_rights"]}),
        encoding="utf-8",
    )

    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = {check["name"]: check for check in summary["stage_checks"]}

    assert summary["overall_status"] == "blocked"
    assert checks["buyer_fulfillment_bundle_ready"]["passed"] is False


def test_launch_gate_rejects_legacy_statusless_bundle_file(tmp_path: Path) -> None:
    capture_root, _descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_arkit_lidar",
    )
    eval_root = capture_root / "pipeline" / "evaluation_prep"
    eval_root.mkdir(parents=True, exist_ok=True)
    # Legacy bundles predate the status field. A bundle without an explicit
    # ready status is missing evidence and must fail closed, not pass as ready.
    (eval_root / "launchable_export_bundle.json").write_text(
        json.dumps({"bundle_uri": "gs://bucket/legacy.zip"}),
        encoding="utf-8",
    )

    summary = build_launch_gate_summary(capture_root=capture_root, env={})
    checks = {check["name"]: check for check in summary["stage_checks"]}

    assert summary["overall_status"] == "blocked"
    assert checks["buyer_fulfillment_bundle_ready"]["passed"] is False


def test_iphone_alpha_readiness_is_go_and_sync_refreshes_after_evaluation_prep(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_arkit_lidar",
    )
    success_backend = tmp_path / "success_backend.py"
    _write_backend_script(success_backend)
    _set_alpha_env(monkeypatch, tmp_path, include_runtime=True)
    sync_calls: list[dict[str, object]] = []
    _stub_sync(monkeypatch, sync_calls)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _write_privacy_outputs(capture_root, depth_source="arkit", include_depth_manifests=False),
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    (capture_root / "pipeline" / "signed_access_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "arena_signed_access_manifest.v1",
                "status": "cloud_artifact_ready_review_required",
                "artifact_uri": (
                    "gs://local-blueprint/scenes/scene-1/captures/capture-1/"
                    "pipeline/archives/post_training_data_package.tar.gz"
                ),
            }
        ),
        encoding="utf-8",
    )
    (capture_root / "pipeline" / "delivery_manifest.json").write_text(
        json.dumps({"schema_version": "arena_delivery_manifest.v1"}),
        encoding="utf-8",
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    alpha_summary = json.loads((capture_root / "pipeline" / "alpha_readiness_summary.json").read_text(encoding="utf-8"))
    launch_gate_summary = json.loads((capture_root / "pipeline" / "launch_gate_summary.json").read_text(encoding="utf-8"))
    sync_result = json.loads((capture_root / "pipeline" / "webapp_sync_result.json").read_text(encoding="utf-8"))

    assert alpha_summary["verdicts"]["external_alpha"]["status"] == "go"
    assert launch_gate_summary["overall_status"] == "automated_contracts_passed_manual_ops_required"
    assert launch_gate_summary["source_acceptance"]["contract_status"] == "external_beta_contract_ready"
    assert launch_gate_summary["source_acceptance"]["operator_evidence_status"] == "blocked"
    assert launch_gate_summary["operator_evidence_status"]["status"] == "blocked"
    assert launch_gate_summary["operator_evidence_status"]["evidence_file_present"] is False
    assert "legal_consent_posture_signoff_evidence_missing_or_unverified" in (
        launch_gate_summary["operator_evidence_status"]["blockers"]
    )
    assert "paperclip_ops_relay_secret_rotation_evidence_missing_or_unverified" in (
        launch_gate_summary["operator_evidence_status"]["blockers"]
    )
    assert "operator_dpa_data_processing_terms_evidence_missing_or_unverified" in (
        launch_gate_summary["operator_evidence_status"]["blockers"]
    )
    assert {
        "legal_consent_posture_signoff",
        "operator_dpa_data_processing_terms",
        "cross_border_data_residency_posture",
        "paperclip_ops_relay_secret_rotation",
        "buyer_payment_settlement",
        "identity_kyc_provider_decision",
    }.issubset({check["id"] for check in launch_gate_summary["operator_required_checks"]})
    assert all(check["passed"] for check in launch_gate_summary["stage_checks"])
    assert sync_result["status"] == "succeeded"
    assert sync_result["latest_stage"] == "evaluation_prep"
    assert (
        sync_result["syncs"]["evaluation_prep"]["attachment_payload"]["evaluation_readiness"]["proof_path_status"]["event_statuses"][0][
            "event_name"
        ]
        == "proof_pack_delivered"
    )
    assert len(sync_calls) == 2
    assert sync_calls[1]["artifacts"]["site_world_spec_uri"].endswith("/evaluation_prep/site_world_spec.json")
    assert sync_calls[1]["artifacts"]["hosted_session_runtime_manifest_uri"].endswith(
        "/evaluation_prep/hosted_session_runtime_manifest.json"
    )
    assert sync_calls[1]["artifacts"]["scene_memory_manifest_uri"].endswith("/scene_memory/scene_memory_manifest.json")
    assert sync_calls[1]["artifacts"]["capturer_payout_recommendation_uri"].endswith(
        "/capturer_payout_recommendation.json"
    )
    assert sync_calls[1]["artifacts"]["launch_gate_summary_uri"].endswith("/launch_gate_summary.json")
    assert sync_calls[1]["artifacts"]["post_training_data_package_uri"].endswith(
        "/pipeline/archives/post_training_data_package.tar.gz"
    )
    assert sync_calls[1]["artifacts"]["delivery_manifest_uri"].endswith("/delivery_manifest.json")
    assert sync_calls[1]["artifacts"]["signed_access_manifest_uri"].endswith(
        "/signed_access_manifest.json"
    )
    assert sync_calls[1]["artifacts"]["site_package_manifest_uri"].endswith(
        "/evaluation_prep/site_package_manifest.json"
    )
    assert sync_calls[1]["artifacts"]["proof_pack_manifest_uri"].endswith(
        "/evaluation_prep/proof_pack_manifest.json"
    )
    assert sync_calls[1]["artifacts"]["hosted_review_readiness_uri"].endswith(
        "/evaluation_prep/hosted_review_readiness.json"
    )
    assert [event["event_name"] for event in sync_calls[1]["evaluation_readiness"]["proof_path_events"]] == [
        "proof_pack_delivered",
        "hosted_review_started",
        "hosted_review_follow_up_sent",
        "human_commercial_handoff_started",
    ]
    assert [event["status"] for event in sync_calls[1]["evaluation_readiness"]["proof_path_events"]] == [
        "verified",
        "pending",
        "pending",
        "pending",
    ]
    assert sync_calls[1]["artifacts"]["rights_provenance_review_uri"].endswith(
        "/rights_provenance_review.json"
    )
    assert sync_calls[1]["evaluation_readiness"]["site_package_manifest"]["status"] in {"ready", "blocked"}
    assert sync_calls[1]["evaluation_readiness"]["proof_pack_manifest"]["status"] in {"ready", "blocked"}

    _write_operator_launch_evidence(capture_root, launch_gate_summary["operator_required_checks"])
    verified_launch_gate = build_launch_gate_summary(capture_root=capture_root)

    assert verified_launch_gate["overall_status"] == "external_beta_live_evidence_ready"
    assert verified_launch_gate["source_acceptance"]["contract_status"] == "external_beta_contract_ready"
    assert verified_launch_gate["operator_evidence_status"]["status"] == "verified"
    assert verified_launch_gate["operator_evidence_status"]["evidence_file_present"] is True
    assert verified_launch_gate["operator_evidence_status"]["blockers"] == []
    assert all(check["passed"] for check in verified_launch_gate["operator_required_checks"])


def test_iphone_video_only_alpha_readiness_is_go_when_geometry_is_ready(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_video_only",
        include_arkit=False,
    )
    success_backend = tmp_path / "success_backend.py"
    _write_backend_script(success_backend)
    _set_alpha_env(monkeypatch, tmp_path, include_runtime=True, include_video_to_world=True)
    sync_calls: list[dict[str, object]] = []
    _stub_sync(monkeypatch, sync_calls)
    _write_geometry_lane(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _write_privacy_outputs(capture_root, depth_source="depth_anything", include_depth_manifests=True),
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    alpha_summary = json.loads((capture_root / "pipeline" / "alpha_readiness_summary.json").read_text(encoding="utf-8"))
    launch_gate_summary = json.loads((capture_root / "pipeline" / "launch_gate_summary.json").read_text(encoding="utf-8"))

    assert alpha_summary["profile"] == "iphone_video_only"
    assert alpha_summary["verdicts"]["external_alpha"]["status"] == "go"
    assert launch_gate_summary["overall_status"] == "automated_contracts_passed_manual_ops_required"
    assert launch_gate_summary["source_acceptance"]["contract_status"] == "external_beta_contract_ready"
    assert launch_gate_summary["operator_evidence_status"]["status"] == "blocked"
    assert sync_calls[1]["artifacts"]["geometry_summary_uri"].endswith("/geometry/geometry_summary.json")


def test_iphone_video_only_alpha_readiness_rejects_fallback_geometry(monkeypatch, tmp_path: Path) -> None:
    capture_root, _descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_video_only",
        include_arkit=False,
    )
    _set_alpha_env(monkeypatch, tmp_path, include_runtime=True, include_video_to_world=True)

    def _failing_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("video_to_world_down")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _failing_provider)
    build_geometry_stage_contract(capture_root)

    alpha_summary = build_alpha_readiness_summary(capture_root=capture_root)
    geometry_checks = {
        item["name"]: item
        for item in alpha_summary["path_checks"]
        if str(item.get("name") or "").startswith("geometry_")
    }

    assert alpha_summary["profile"] == "iphone_video_only"
    assert alpha_summary["verdicts"]["external_alpha"]["status"] == "no_go"
    assert geometry_checks["geometry_ready_for_world_model"]["passed"] is False
    assert geometry_checks["geometry_uses_real_video_to_world"]["passed"] is False
    assert alpha_summary["runtime_capability"]["geometry_ready"] is False
    assert "geometry_not_ready" in alpha_summary["runtime_capability"]["blockers"]


def test_iphone_alpha_readiness_is_no_go_when_runtime_url_missing(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_capture(
        tmp_path,
        capture_source="iphone",
        capture_modality="iphone_arkit_lidar",
    )
    success_backend = tmp_path / "success_backend.py"
    _write_backend_script(success_backend)
    _set_alpha_env(monkeypatch, tmp_path, include_runtime=False)
    sync_calls: list[dict[str, object]] = []
    _stub_sync(monkeypatch, sync_calls)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _write_privacy_outputs(capture_root, depth_source="arkit", include_depth_manifests=False),
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    alpha_summary = json.loads((capture_root / "pipeline" / "alpha_readiness_summary.json").read_text(encoding="utf-8"))
    launch_gate_summary = json.loads((capture_root / "pipeline" / "launch_gate_summary.json").read_text(encoding="utf-8"))

    assert alpha_summary["verdicts"]["external_alpha"]["status"] == "no_go"
    assert launch_gate_summary["overall_status"] == "blocked"
    assert any("missing runtime service URL" in reason for reason in alpha_summary["no_go_reasons"])


def test_meta_glasses_alpha_readiness_is_internal_until_operator_evidence(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_capture(
        tmp_path,
        capture_source="glasses",
        capture_modality="glasses_video_only",
    )
    success_backend = tmp_path / "success_backend.py"
    _write_backend_script(success_backend)
    _set_alpha_env(monkeypatch, tmp_path, include_runtime=True, include_video_to_world=True)
    sync_calls: list[dict[str, object]] = []
    _stub_sync(monkeypatch, sync_calls)
    _write_geometry_lane(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _write_privacy_outputs(capture_root, depth_source="depth_anything", include_depth_manifests=True),
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    alpha_summary = json.loads((capture_root / "pipeline" / "alpha_readiness_summary.json").read_text(encoding="utf-8"))
    launch_gate_summary = json.loads((capture_root / "pipeline" / "launch_gate_summary.json").read_text(encoding="utf-8"))

    assert alpha_summary["verdicts"]["external_alpha"]["status"] == "no_go"
    assert alpha_summary["verdicts"]["external_alpha"]["contract_status"] == "ready"
    assert alpha_summary["verdicts"]["internal_experimental_alpha"]["status"] == "go"
    assert alpha_summary["device_alpha_profile"]["status"] == "internal_only"
    assert alpha_summary["launch_market_readiness"]["internal_pilot_ready"] is True
    assert alpha_summary["launch_market_readiness"]["external_market_ready"] is False
    assert alpha_summary["runtime_capability"]["status"] == "ready"
    assert launch_gate_summary["overall_status"] == "internal_only_contract_ready"
    assert any(
        "Do not claim strong site-faithful world-model quality" in claim
        for claim in launch_gate_summary["launch_claims"]["not_justified"]
    )
    assert any(
        "Do not market this source as externally launch-ready" in claim
        for claim in launch_gate_summary["launch_claims"]["not_justified"]
    )
    assert sync_calls[1]["artifacts"]["geometry_summary_uri"].endswith("/geometry/geometry_summary.json")
    assert sync_calls[1]["artifacts"]["privacy_depth_manifest_uri"].endswith("/privacy_depth/depth_manifest.json")


def test_android_alpha_readiness_is_internal_until_operator_evidence(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_capture(
        tmp_path,
        capture_source="android",
        capture_modality="android_video_only",
    )
    success_backend = tmp_path / "success_backend.py"
    _write_backend_script(success_backend)
    _set_alpha_env(monkeypatch, tmp_path, include_runtime=True, include_video_to_world=True)
    _stub_sync(monkeypatch, [])
    _write_geometry_lane(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _write_privacy_outputs(capture_root, depth_source="depth_anything", include_depth_manifests=True),
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    alpha_summary = json.loads((capture_root / "pipeline" / "alpha_readiness_summary.json").read_text(encoding="utf-8"))
    launch_gate_summary = json.loads((capture_root / "pipeline" / "launch_gate_summary.json").read_text(encoding="utf-8"))

    assert alpha_summary["profile"] == "android_video"
    assert alpha_summary["verdicts"]["external_alpha"]["status"] == "no_go"
    assert alpha_summary["verdicts"]["external_alpha"]["contract_status"] == "ready"
    assert alpha_summary["verdicts"]["internal_experimental_alpha"]["status"] == "go"
    assert alpha_summary["device_alpha_profile"]["status"] == "internal_only"
    assert alpha_summary["launch_market_readiness"]["internal_pilot_ready"] is True
    assert alpha_summary["launch_market_readiness"]["external_market_ready"] is False
    assert alpha_summary["runtime_capability"]["status"] == "ready"
    assert launch_gate_summary["overall_status"] == "internal_only_contract_ready"
    assert any(
        "Do not claim strong site-faithful world-model quality" in claim
        for claim in launch_gate_summary["launch_claims"]["not_justified"]
    )
    assert any(
        "Do not market this source as externally launch-ready" in claim
        for claim in launch_gate_summary["launch_claims"]["not_justified"]
    )
