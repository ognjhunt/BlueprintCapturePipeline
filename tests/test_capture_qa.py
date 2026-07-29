from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

import blueprint_pipeline.capture_qa as capture_qa
from blueprint_pipeline.capture_intake import CaptureIntakeError
from blueprint_pipeline.capture_qa import build_capture_qa_report


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _envelope(
    payload: bytes,
    *,
    profile: str = "camera_360_equirectangular",
    streams: list[str] | None = None,
    filename: str = "capture.mp4",
) -> dict:
    streams = streams or ["retained_video", "camera_metadata"]
    return {
        "schema_version": "capture_intake_envelope.v1",
        "intake_id": "intake-qa-1",
        "idempotency_key": "org-1-qa-upload-1",
        "capture_authority_profile": profile,
        "source_type": profile,
        "original_files": [
            {
                "original_filename": filename,
                "relative_path": filename,
                "sha256": _digest(payload),
                "size_bytes": len(payload),
                "media_type": "video/mp4",
            }
        ],
        "scene_id": "scene-1",
        "customer_id": "customer-1",
        "organization_id": "org-1",
        "capture_device": {"manufacturer": "fixture", "model": "360-camera"},
        "timing_declaration": {"clock": "media_pts"},
        "coordinate_frame_declaration": {"status": "not_available_from_video"},
        "available_sensor_streams": [
            {
                "stream_type": stream,
                "status": "available",
                "source_relative_path": filename,
            }
            for stream in streams
        ],
        "governance": {
            "rights": "accepted",
            "consent": "accepted",
            "privacy": "cleared",
            "retention": {"max_days": 30},
            "revocation": {"supported": True, "historical_tombstone_retained": True},
            "provider_constraints": {"external_processing_allowed": False},
            "allowed_uses": ["evaluation"],
        },
        "requested_task_evaluation_run_audience": "design_partner",
        "known_task_specification": None,
        "calibration_board_dimensions": None,
        "operator_notes": [],
        "permitted_reconstruction_providers": ["local_only"],
        "permitted_evidence_uses": ["captured_observation", "task_discovery"],
        "upload_validation": {"status": "passed"},
        "malware_content_validation": {"status": "passed", "scanner": "fixture"},
    }


def _probe(payload: bytes, *, pts: list[float] | None = None, **overrides: object) -> dict:
    value = {
        "status": "ready",
        "tool": "ffprobe",
        "duration_seconds": 20.0,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec_name": "h264",
        "width": 3840,
        "height": 1920,
        "frame_rate": 30.0,
        "decoded_frame_count": 4,
        "frame_pts_seconds": pts or [0.0, 1 / 30, 2 / 30, 3 / 30],
        "rotation_degrees": 0,
        "source_file_sha256": _digest(payload),
    }
    value.update(overrides)
    return value


def _observations(payload: bytes, **overrides: object) -> dict:
    measurements = {
        "sharp_frame_fraction": 0.95,
        "well_exposed_frame_fraction": 0.96,
        "visual_overlap_fraction": 0.82,
        "compression_quality_fraction": 0.94,
        "rolling_shutter_symptom_fraction": 0.02,
        "privacy_sensitive_content_detected": False,
        "dynamic_people_detected": False,
        "moving_task_objects_detected": False,
        "task_critical_occlusion_detected": False,
        "robot_placement_area_covered": True,
        "scale_anchor_verified": False,
    }
    measurements.update(overrides)
    return {
        "schema_version": "capture_quality_observations.v1",
        "source": "local_analyzer",
        "intake_id": "intake-qa-1",
        "source_file_sha256": _digest(payload),
        "measurements": measurements,
    }


def _upload(tmp_path: Path, payload: bytes, filename: str = "capture.mp4") -> Path:
    root = tmp_path / "upload"
    root.mkdir()
    (root / filename).write_bytes(payload)
    return root


def _check(report: dict, check_id: str) -> dict:
    return next(row for row in report["checks"] if row["check_id"] == check_id)


def test_complete_360_media_passes_qa_without_upgrading_metric_or_physical_claims(
    tmp_path: Path,
) -> None:
    payload = b"rights-cleared-360-video"
    report = build_capture_qa_report(
        _envelope(payload),
        upload_root=_upload(tmp_path, payload),
        media_probe=_probe(payload),
        quality_observations=_observations(payload),
    )

    assert report["status"] == "accepted"
    assert report["state"] == "capture_accepted"
    assert report["recapture_plan"] == []
    assert _check(report, "decoded_pts_continuity")["status"] == "pass"
    assert report["claim_ceiling"]["task_candidate_discovery"] is True
    assert report["claim_ceiling"]["metric_geometry"] is False
    assert report["claim_ceiling"]["collision_geometry"] is False
    assert report["claim_ceiling"]["physical_task_success"] is False
    assert report["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert report["missing_evidence"] == [
        "camera_pose_availability",
        "depth_availability",
        "intrinsics_availability",
        "scale_anchor_verified",
    ]
    assert report["next_cheapest_experiment"]["kind"] == "local_or_operator_measurement"


def test_pts_gap_fails_with_specific_export_or_recapture_instruction(tmp_path: Path) -> None:
    payload = b"video-with-gap"
    report = build_capture_qa_report(
        _envelope(payload),
        upload_root=_upload(tmp_path, payload),
        media_probe=_probe(payload, pts=[0.0, 1 / 30, 2 / 30, 0.8]),
        quality_observations=_observations(payload),
    )

    assert report["status"] == "recapture_required"
    pts_check = _check(report, "decoded_pts_continuity")
    assert pts_check["status"] == "fail"
    assert pts_check["measurement"]["maximum_delta_seconds"] > 0.7
    plan = {row["code"]: row for row in report["recapture_plan"]}
    assert "decoded_pts_discontinuity" in plan
    assert "retained original" in plan["decoded_pts_discontinuity"]["instruction"]
    assert report["claim_ceiling"]["captured_observation_review"] is False


def test_quality_failures_return_bounded_task_specific_recapture(tmp_path: Path) -> None:
    payload = b"blurry-occluded-video"
    report = build_capture_qa_report(
        _envelope(payload),
        upload_root=_upload(tmp_path, payload),
        media_probe=_probe(payload),
        quality_observations=_observations(
            payload,
            sharp_frame_fraction=0.2,
            visual_overlap_fraction=0.3,
            task_critical_occlusion_detected=True,
            robot_placement_area_covered=False,
        ),
    )

    assert report["status"] == "recapture_required"
    plan = {row["code"]: row["instruction"] for row in report["recapture_plan"]}
    assert "Repeat the affected pass more slowly" in plan["excessive_blur"]
    assert "overlapping passes" in plan["low_visual_overlap"]
    assert "underside or rear" in plan["task_critical_occlusion"]
    assert "robot placement area" in plan["robot_placement_area_missing"]
    assert report["next_cheapest_experiment"]["kind"] == "targeted_recapture"


@pytest.mark.parametrize(
    ("probe_override", "code"),
    [
        ({"status": "failed", "reason": "ffprobe_rejected_media"}, "media_not_decodable"),
        ({"duration_seconds": 2.0}, "capture_too_short"),
        ({"width": 640, "height": 360}, "resolution_too_low"),
        ({"frame_rate": 10.0}, "frame_rate_unsupported"),
        ({"codec_name": "mpeg2video"}, "codec_unsupported"),
    ],
)
def test_media_contract_failures_produce_exact_recapture_code(
    tmp_path: Path, probe_override: dict, code: str
) -> None:
    payload = code.encode()
    probe = _probe(payload)
    probe.update(probe_override)
    report = build_capture_qa_report(
        _envelope(payload),
        upload_root=_upload(tmp_path, payload),
        media_probe=probe,
        quality_observations=_observations(payload),
    )

    assert report["status"] == "recapture_required"
    assert code in {row["code"] for row in report["recapture_plan"]}


def test_unmeasured_quality_requires_analysis_and_never_becomes_a_pass(tmp_path: Path) -> None:
    payload = b"unmeasured-video"
    report = build_capture_qa_report(
        _envelope(payload, profile="monocular_video", streams=["retained_video"]),
        upload_root=_upload(tmp_path, payload),
        media_probe=_probe(payload),
    )

    assert report["status"] == "analysis_required"
    assert report["state"] == "validating"
    assert _check(report, "sharp_frame_fraction")["status"] == "not_measured"
    assert "sharp_frame_fraction" in report["missing_evidence"]
    assert report["next_cheapest_experiment"]["code"].startswith("measure_")
    assert report["next_cheapest_experiment"]["kind"] == "local_quality_analysis"
    assert "sharp_frame_fraction" in report["required_analysis"]
    assert report["claim_ceiling"]["capture_admitted"] is False
    assert report["claim_ceiling"]["metric_geometry"] is False


def test_quality_observations_require_provenance_and_deterministic_digest(tmp_path: Path) -> None:
    payload = b"video"
    upload = _upload(tmp_path, payload)
    envelope = _envelope(payload)
    invalid = _observations(payload)
    invalid["source"] = "vision_model_said_so"
    with pytest.raises(CaptureIntakeError, match="source:unsupported"):
        build_capture_qa_report(
            envelope,
            upload_root=upload,
            media_probe=_probe(payload),
            quality_observations=invalid,
        )

    wrong_probe = _probe(b"different-bytes")
    with pytest.raises(CaptureIntakeError, match="media_probe.source_file_sha256:mismatch"):
        build_capture_qa_report(
            envelope,
            upload_root=upload,
            media_probe=wrong_probe,
            quality_observations=_observations(payload),
        )

    wrong_observations = _observations(b"different-bytes")
    with pytest.raises(
        CaptureIntakeError, match="quality_observations.source_file_sha256:mismatch"
    ):
        build_capture_qa_report(
            envelope,
            upload_root=upload,
            media_probe=_probe(payload),
            quality_observations=wrong_observations,
        )

    invalid_fraction = _observations(payload, sharp_frame_fraction=1.2)
    with pytest.raises(CaptureIntakeError, match="sharp_frame_fraction:invalid_fraction"):
        build_capture_qa_report(
            envelope,
            upload_root=upload,
            media_probe=_probe(payload),
            quality_observations=invalid_fraction,
        )

    operator_fractions = _observations(payload)
    operator_fractions["source"] = "operator_attested"
    with pytest.raises(CaptureIntakeError, match="quantified_fractions_forbidden"):
        build_capture_qa_report(
            envelope,
            upload_root=upload,
            media_probe=_probe(payload),
            quality_observations=operator_fractions,
        )

    first = build_capture_qa_report(
        envelope,
        upload_root=upload,
        media_probe=_probe(payload),
        quality_observations=_observations(payload),
    )
    second = build_capture_qa_report(
        envelope,
        upload_root=upload,
        media_probe=_probe(payload),
        quality_observations=_observations(payload),
    )
    assert first == second
    assert first["qa_report_digest"].startswith("sha256:")


def test_capture_qa_schemas_accept_the_runtime_contract(tmp_path: Path) -> None:
    payload = b"schema-video"
    observations = _observations(payload)
    report = build_capture_qa_report(
        _envelope(payload),
        upload_root=_upload(tmp_path, payload),
        media_probe=_probe(payload),
        quality_observations=observations,
    )
    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    observations_schema = json.loads(
        (schema_root / "capture_quality_observations.schema.json").read_text(encoding="utf-8")
    )
    report_schema = json.loads(
        (schema_root / "capture_qa_report.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(observations_schema)
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.Draft202012Validator(observations_schema).validate(observations)
    jsonschema.Draft202012Validator(report_schema).validate(report)


def test_media_probe_uses_decoded_frame_pts_instead_of_packet_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "format": {"duration": "6.0", "format_name": "mov,mp4"},
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 1280,
                            "height": 720,
                            "avg_frame_rate": "15/1",
                        }
                    ],
                    "frames": [
                        {"best_effort_timestamp_time": "0.000000"},
                        {"best_effort_timestamp_time": "0.066667"},
                    ],
                    # Packet PTS can be out of presentation order for B-frames
                    # and must not be used by this contract.
                    "packets": [{"pts_time": "0.066667"}, {"pts_time": "0.000000"}],
                }
            ),
        )

    monkeypatch.setattr(capture_qa.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(capture_qa.subprocess, "run", fake_run)

    probe = capture_qa._probe_video(video)

    assert "-show_frames" in commands[0]
    assert "-show_packets" not in commands[0]
    assert probe["frame_pts_seconds"] == [0.0, 0.066667]
    assert probe["decoded_frame_count"] == 2


def test_privacy_and_dynamic_scene_checks_fail_closed_with_separate_reasons(
    tmp_path: Path,
) -> None:
    payload = b"privacy-video"
    report = build_capture_qa_report(
        _envelope(payload),
        upload_root=_upload(tmp_path, payload),
        media_probe=_probe(payload),
        quality_observations=_observations(
            payload,
            privacy_sensitive_content_detected=True,
            dynamic_people_detected=True,
            moving_task_objects_detected=True,
        ),
    )

    codes = {row["code"] for row in report["recapture_plan"]}
    assert report["status"] == "recapture_required"
    assert {
        "privacy_sensitive_content",
        "dynamic_people_present",
        "task_objects_moved_during_capture",
    }.issubset(codes)
    assert report["claim_ceiling"]["task_candidate_discovery"] is False
