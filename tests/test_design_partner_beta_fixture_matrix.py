from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from blueprint_pipeline.capture_intake import build_capture_admission
from blueprint_pipeline.capture_qa import build_capture_qa_report


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "design_partner_beta_v1"
    / "fixture_matrix.json"
)
REQUIRED_CASE_IDS = {
    "complete_iphone_arkit_lidar_bundle",
    "iphone_encoder_frame_sync_omissions",
    "supported_360_equirectangular_metadata",
    "360_without_scale_evidence",
    "ordinary_monocular_video",
    "privacy_restricted_capture",
    "provider_forbidden_capture",
    "low_overlap_or_blurred_capture",
    "generated_only_gap_intersects_trajectory",
    "valid_external_reconstruction_import",
    "stale_reconstruction_wrong_source_digest",
    "unqualified_simready_asset",
    "robot_placement_outside_captured_coverage",
    "inferred_task_awaiting_approval",
    "explicit_customer_task",
    "full_decision",
    "partial_decision",
    "explicit_abstention",
    "physical_evidence_request",
}
SECRET_KEY = re.compile(
    r"(^|_)(secret|token|password|private_key|api_key|credential)($|_)", re.I
)


def _matrix() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _secret_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, list):
        return [
            path
            for index, child in enumerate(value)
            for path in _secret_paths(child, f"{prefix}[{index}]")
        ]
    if not isinstance(value, Mapping):
        return []
    found: list[str] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if SECRET_KEY.search(str(key)) and child not in (None, "", False):
            found.append(path)
        found.extend(_secret_paths(child, path))
    return found


def _capture_envelope(case: Mapping[str, Any]) -> dict[str, Any]:
    payload = str(case["case_id"]).encode("utf-8")
    profile = str(case["capture_authority_profile"])
    filename = "capture.mp4"
    return {
        "schema_version": "capture_intake_envelope.v1",
        "intake_id": f"intake-{case['case_id']}",
        "idempotency_key": f"fixture-{case['case_id']}",
        "capture_authority_profile": profile,
        "source_type": profile,
        "original_files": [{
            "original_filename": filename,
            "relative_path": filename,
            "sha256": _digest(payload),
            "size_bytes": len(payload),
            "media_type": "video/mp4",
        }],
        "scene_id": "fixture-tabletop-site",
        "customer_id": "fixture-customer",
        "organization_id": "fixture-organization",
        "capture_device": {"manufacturer": "fixture", "model": "fixture-camera"},
        "timing_declaration": {"clock": "media_pts"},
        "coordinate_frame_declaration": {"status": "declared_by_fixture"},
        "available_sensor_streams": [{
            "stream_type": stream,
            "status": "available",
            "source_relative_path": filename,
        } for stream in case["available_streams"]],
        "governance": {
            "rights": "accepted",
            "consent": "accepted",
            "privacy": case.get("privacy", "cleared"),
            "retention": {"max_days": 30},
            "revocation": {"supported": True, "historical_tombstone_retained": True},
            "provider_constraints": {
                "external_processing_allowed": case.get(
                    "external_processing_allowed", False
                )
            },
            "allowed_uses": ["evaluation"],
        },
        "requested_task_evaluation_run_audience": "design_partner",
        "known_task_specification": None,
        "calibration_board_dimensions": None,
        "operator_notes": [],
        "permitted_reconstruction_providers": case.get(
            "permitted_reconstruction_providers", ["local_only"]
        ),
        "permitted_evidence_uses": ["captured_observation", "task_discovery"],
        "upload_validation": {"status": "passed"},
        "malware_content_validation": {"status": "passed", "scanner": "fixture"},
    }


def test_fixture_matrix_is_finite_complete_redacted_and_keeps_the_frozen_verdict() -> None:
    matrix = _matrix()
    assert matrix["schema_version"] == "design_partner_beta_fixture_matrix.v1"
    assert matrix["synthetic_or_redacted_only"] is True
    assert matrix["real_capture_gate_satisfied"] is False
    assert matrix["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    cases = matrix["cases"]
    case_ids = [row["case_id"] for row in cases]
    assert len(case_ids) == len(set(case_ids))
    assert set(case_ids) == REQUIRED_CASE_IDS
    assert _secret_paths(matrix) == []


def test_capture_matrix_cases_execute_through_fail_closed_admission() -> None:
    cases = {row["case_id"]: row for row in _matrix()["cases"]}
    for case_id in (
        "complete_iphone_arkit_lidar_bundle",
        "iphone_encoder_frame_sync_omissions",
        "supported_360_equirectangular_metadata",
        "360_without_scale_evidence",
        "ordinary_monocular_video",
        "privacy_restricted_capture",
        "provider_forbidden_capture",
    ):
        case = cases[case_id]
        admission = build_capture_admission(_capture_envelope(case))
        assert (
            admission["claim_ceiling"]["comparative_policy_ranking_verdict"]
            == "thesis_not_supported"
        )
        if case.get("expected_metric_authority") is not None:
            assert admission["claim_ceiling"]["metric_geometry"] is case[
                "expected_metric_authority"
            ]
        if case_id == "iphone_encoder_frame_sync_omissions":
            assert admission["status"] == "recapture_required"
            assert admission["missing_required_streams"] == case["expected_missing_evidence"]
        elif case_id == "provider_forbidden_capture":
            assert admission["status"] == "rejected"
            assert case["expected_blocker"] in admission["governance_blockers"]
        else:
            assert admission["status"] == "accepted"
            assert admission["claim_ceiling"]["physical_task_success"] is False
    restricted = build_capture_admission(
        _capture_envelope(cases["privacy_restricted_capture"])
    )
    assert restricted["status"] == "accepted"
    assert restricted["provider_execution_authorized"] is False


def test_low_quality_fixture_returns_specific_recapture_plan(tmp_path: Path) -> None:
    case = next(
        row for row in _matrix()["cases"]
        if row["case_id"] == "low_overlap_or_blurred_capture"
    )
    envelope = _capture_envelope(case)
    payload = str(case["case_id"]).encode("utf-8")
    upload_root = tmp_path / "upload"
    upload_root.mkdir()
    (upload_root / "capture.mp4").write_bytes(payload)
    report = build_capture_qa_report(
        envelope,
        upload_root=upload_root,
        media_probe={
            "status": "ready",
            "tool": "ffprobe",
            "duration_seconds": 20.0,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "codec_name": "h264",
            "width": 3840,
            "height": 1920,
            "frame_rate": 30.0,
            "decoded_frame_count": 4,
            "frame_pts_seconds": [0.0, 1 / 30, 2 / 30, 3 / 30],
            "rotation_degrees": 0,
            "source_file_sha256": _digest(payload),
        },
        quality_observations={
            "schema_version": "capture_quality_observations.v1",
            "source": "local_analyzer",
            "intake_id": envelope["intake_id"],
            "source_file_sha256": _digest(payload),
            "measurements": {
                "sharp_frame_fraction": case["sharp_frame_fraction"],
                "well_exposed_frame_fraction": 0.95,
                "visual_overlap_fraction": case["visual_overlap_fraction"],
                "compression_quality_fraction": 0.95,
                "rolling_shutter_symptom_fraction": 0.01,
                "privacy_sensitive_content_detected": False,
                "dynamic_people_detected": False,
                "moving_task_objects_detected": False,
                "task_critical_occlusion_detected": False,
                "robot_placement_area_covered": True,
                "scale_anchor_verified": False,
            },
        },
    )
    assert report["status"] == "recapture_required"
    codes = {row["code"] for row in report["recapture_plan"]}
    assert set(case["expected_recapture_codes"]) <= codes
    assert report["next_cheapest_experiment"]["kind"] == "targeted_recapture"


def test_non_capture_fixture_expectations_preserve_proof_boundaries() -> None:
    cases = {row["case_id"]: row for row in _matrix()["cases"]}
    assert cases["generated_only_gap_intersects_trajectory"]["expected_physics_use"] is False
    assert cases["valid_external_reconstruction_import"]["expected_raw_capture_authority_upgrade"] is False
    assert cases["stale_reconstruction_wrong_source_digest"]["expected_state"] == "failed"
    assert cases["unqualified_simready_asset"]["expected_physics_use"] is False
    assert cases["robot_placement_outside_captured_coverage"]["expected_verdict"] == "abstention"
    assert cases["inferred_task_awaiting_approval"]["decision_evidence_request_allowed"] is False
    assert cases["explicit_customer_task"]["decision_evidence_request_allowed"] is True
    assert {
        cases[case_id]["overall_outcome"]
        for case_id in ("full_decision", "partial_decision", "explicit_abstention")
    } == {"decision", "partial_decision", "abstention"}
    physical = cases["physical_evidence_request"]
    assert physical["expected_verdict"] == "abstention"
    assert physical["physical_evidence_request_present"] is True
    assert physical["deployment_approval"] is False
