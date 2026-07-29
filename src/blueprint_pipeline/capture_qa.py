"""Deterministic, authority-aware capture QA and targeted recapture planning.

The report produced here assesses whether admitted media is usable for the
capture profile and requested task.  It never upgrades video, reconstruction,
or generated content into metric, physics, physical, deployment, safety, or
policy-ranking evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_intake import (
    CaptureIntakeError,
    build_capture_admission,
    verify_capture_intake_bytes,
)


CAPTURE_QA_SCHEMA_VERSION = "capture_qa_report.v1"
CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION = "capture_quality_observations.v1"
_VIDEO_PROFILES = {
    "iphone_arkit_lidar",
    "iphone_arkit_non_lidar",
    "camera_360_equirectangular",
    "camera_360_native",
    "monocular_video",
}
_VIDEO_STREAMS = ("retained_video", "retained_original")
_SUPPORTED_CODECS = {"h264", "hevc", "av1", "vp9"}
_FRACTION_MEASUREMENTS = {
    "sharp_frame_fraction",
    "well_exposed_frame_fraction",
    "visual_overlap_fraction",
    "compression_quality_fraction",
    "rolling_shutter_symptom_fraction",
}
_BOOLEAN_MEASUREMENTS = {
    "privacy_sensitive_content_detected",
    "dynamic_people_detected",
    "moving_task_objects_detected",
    "task_critical_occlusion_detected",
    "robot_placement_area_covered",
    "scale_anchor_verified",
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    normalized = json.loads(json.dumps(value))
    if omit:
        normalized.pop(omit, None)
    return "sha256:" + hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _rate(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        top = _float(numerator)
        bottom = _float(denominator)
        return top / bottom if top is not None and bottom not in {None, 0.0} else None
    return _float(text)


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"status": "unavailable", "reason": "ffprobe_not_found"}
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration,format_name:stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:stream_tags=rotate:stream_side_data=rotation:frame=best_effort_timestamp_time,pkt_duration_time",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "ffprobe_timeout"}
    except OSError:
        return {"status": "failed", "reason": "ffprobe_execution_failed"}
    if process.returncode != 0:
        return {"status": "failed", "reason": "ffprobe_rejected_media"}
    try:
        payload = json.loads(process.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "failed", "reason": "ffprobe_output_not_json"}
    streams = _rows(payload.get("streams"))
    stream = streams[0] if streams else {}
    format_info = payload.get("format")
    format_info = format_info if isinstance(format_info, Mapping) else {}
    frame_rows = _rows(payload.get("frames"))
    pts = [
        number
        for row in frame_rows
        if (number := _float(row.get("best_effort_timestamp_time"))) is not None
    ]
    duration = _float(format_info.get("duration"))
    rotation = (
        _integer((stream.get("tags") or {}).get("rotate"))
        if isinstance(stream.get("tags"), Mapping)
        else None
    )
    if rotation is None:
        side_data = _rows(stream.get("side_data_list"))
        rotation = next(
            (value for row in side_data if (value := _integer(row.get("rotation"))) is not None),
            0,
        )
    return {
        "status": "ready",
        "tool": "ffprobe",
        "duration_seconds": duration,
        "format_name": str(format_info.get("format_name") or "") or None,
        "codec_name": str(stream.get("codec_name") or "").lower() or None,
        "width": _integer(stream.get("width")),
        "height": _integer(stream.get("height")),
        "frame_rate": _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")),
        "declared_frame_count": _integer(stream.get("nb_frames")),
        "decoded_frame_count": len(pts),
        "frame_pts_seconds": pts,
        "rotation_degrees": rotation or 0,
    }


def _validate_observations(
    value: Mapping[str, Any] | None,
    *,
    intake_id: str,
    source_file_digests: set[str],
) -> dict[str, Any]:
    if value is None:
        return {
            "schema_version": CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION,
            "source": "not_supplied",
            "measurements": {},
        }
    try:
        observations = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise CaptureIntakeError(["quality_observations:not_json_serializable"]) from exc
    errors: list[str] = []
    if observations.get("schema_version") != CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION:
        errors.append(
            f"quality_observations.schema_version:must_be:{CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION}"
        )
    if str(observations.get("source") or "") not in {
        "capture_sidecars",
        "local_analyzer",
        "operator_attested",
    }:
        errors.append("quality_observations.source:unsupported")
    if str(observations.get("intake_id") or "") != intake_id:
        errors.append("quality_observations.intake_id:mismatch")
    if str(observations.get("source_file_sha256") or "") not in source_file_digests:
        errors.append("quality_observations.source_file_sha256:mismatch")
    measurements = observations.get("measurements")
    if not isinstance(measurements, Mapping):
        errors.append("quality_observations.measurements:missing")
    else:
        for key in sorted(_FRACTION_MEASUREMENTS & set(measurements)):
            number = _float(measurements.get(key))
            if number is None or not 0.0 <= number <= 1.0:
                errors.append(f"quality_observations.measurements.{key}:invalid_fraction")
        for key in sorted(_BOOLEAN_MEASUREMENTS & set(measurements)):
            if not isinstance(measurements.get(key), bool):
                errors.append(f"quality_observations.measurements.{key}:must_be_boolean")
        if observations.get("source") == "operator_attested" and any(
            key in measurements for key in _FRACTION_MEASUREMENTS
        ):
            errors.append("quality_observations.operator_attested:quantified_fractions_forbidden")
    if errors:
        raise CaptureIntakeError(errors)
    observations["observations_digest"] = _digest(observations, omit="observations_digest")
    return observations


def _check(
    check_id: str,
    status: str,
    *,
    evidence_source: str,
    measurement: Any = None,
    threshold: Any = None,
    claim_impact: Sequence[str] = (),
    recapture_code: str | None = None,
    recapture_instruction: str | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": status,
        "evidence_source": evidence_source,
        "measurement": measurement,
        "threshold": threshold,
        "claim_impact": sorted(set(claim_impact)),
        "recapture_code": recapture_code,
        "recapture_instruction": recapture_instruction,
    }


def _pts_check(probe: Mapping[str, Any]) -> dict[str, Any]:
    pts = [
        number
        for value in probe.get("frame_pts_seconds", [])
        if (number := _float(value)) is not None
    ]
    if len(pts) < 2:
        return _check(
            "decoded_pts_continuity",
            "fail",
            evidence_source="decoded_media_probe",
            measurement={"decoded_timestamp_count": len(pts)},
            threshold={"minimum_timestamp_count": 2},
            claim_impact=("captured_observation_timing", "calibrated_camera_poses"),
            recapture_code="decoded_pts_unavailable",
            recapture_instruction="Re-export the original video without frame-rate conversion; if decoded timestamps are still unavailable, repeat the capture in the supported app or camera mode.",
        )
    deltas = [right - left for left, right in zip(pts, pts[1:])]
    positive = sorted(delta for delta in deltas if delta > 0)
    median = positive[len(positive) // 2] if positive else 0.0
    max_allowed = max(0.25, median * 4.0)
    regressions = sum(delta <= 0 for delta in deltas)
    max_gap = max(deltas)
    passed = regressions == 0 and max_gap <= max_allowed
    return _check(
        "decoded_pts_continuity",
        "pass" if passed else "fail",
        evidence_source="decoded_media_probe",
        measurement={
            "decoded_timestamp_count": len(pts),
            "non_monotonic_delta_count": regressions,
            "median_delta_seconds": round(median, 9),
            "maximum_delta_seconds": round(max_gap, 9),
        },
        threshold={"maximum_gap_seconds": round(max_allowed, 9), "strictly_monotonic": True},
        claim_impact=("captured_observation_timing", "calibrated_camera_poses"),
        recapture_code=None if passed else "decoded_pts_discontinuity",
        recapture_instruction=None
        if passed
        else "Re-export the retained original without dropped or duplicated timestamps; if the source itself contains the gap, repeat the affected pass slowly with the supported capture mode.",
    )


def _measurement_check(
    measurements: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    claim_impact: Sequence[str],
    recapture_code: str,
    recapture_instruction: str,
) -> dict[str, Any]:
    value = measurements.get(key)
    number = _float(value)
    if number is None:
        return _check(
            key,
            "not_measured",
            evidence_source="quality_observations",
            claim_impact=claim_impact,
            recapture_code=recapture_code,
            recapture_instruction=recapture_instruction,
        )
    passed = (minimum is None or number >= minimum) and (maximum is None or number <= maximum)
    return _check(
        key,
        "pass" if passed else "fail",
        evidence_source="quality_observations",
        measurement=number,
        threshold={"minimum": minimum, "maximum": maximum},
        claim_impact=claim_impact,
        recapture_code=None if passed else recapture_code,
        recapture_instruction=None if passed else recapture_instruction,
    )


def _boolean_check(
    measurements: Mapping[str, Any],
    key: str,
    *,
    expected: bool,
    claim_impact: Sequence[str],
    recapture_code: str,
    recapture_instruction: str,
) -> dict[str, Any]:
    value = measurements.get(key)
    if not isinstance(value, bool):
        return _check(
            key,
            "not_measured",
            evidence_source="quality_observations",
            claim_impact=claim_impact,
            recapture_code=recapture_code,
            recapture_instruction=recapture_instruction,
        )
    passed = value is expected
    return _check(
        key,
        "pass" if passed else "fail",
        evidence_source="quality_observations",
        measurement=value,
        threshold={"expected": expected},
        claim_impact=claim_impact,
        recapture_code=None if passed else recapture_code,
        recapture_instruction=None if passed else recapture_instruction,
    )


def _video_source(
    envelope: Mapping[str, Any], verified: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    sources = {
        str(row.get("stream_type") or ""): str(row.get("source_relative_path") or "")
        for row in _rows(envelope.get("available_sensor_streams"))
        if str(row.get("status") or "") == "available"
    }
    relative = next((sources[key] for key in _VIDEO_STREAMS if sources.get(key)), "")
    return next(
        (row for row in verified if str(row.get("source_relative_path") or "") == relative),
        None,
    )


def build_capture_qa_report(
    envelope_value: Mapping[str, Any],
    *,
    upload_root: Path,
    media_probe: Mapping[str, Any] | None = None,
    quality_observations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify bytes and produce deterministic QA, recapture, and missing-evidence output."""

    envelope, verified = verify_capture_intake_bytes(envelope_value, upload_root=upload_root)
    admission = build_capture_admission(envelope)
    source_file_digests = {str(row.get("sha256") or "") for row in verified}
    observations = _validate_observations(
        quality_observations,
        intake_id=str(envelope["intake_id"]),
        source_file_digests=source_file_digests,
    )
    measurements = observations.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    profile = str(envelope["capture_authority_profile"])
    checks: list[dict[str, Any]] = []
    recapture_plan = [dict(row) for row in _rows(admission.get("recapture_plan"))]

    if profile in _VIDEO_PROFILES:
        source = _video_source(envelope, verified)
        expected_source_digest = str(source.get("sha256") or "") if source else ""
        if media_probe is not None:
            probe = dict(media_probe)
            if str(probe.get("source_file_sha256") or "") != expected_source_digest:
                raise CaptureIntakeError(["media_probe.source_file_sha256:mismatch"])
        elif source is not None:
            probe = _probe_video(Path(str(source["source_path"])))
            probe["source_file_sha256"] = expected_source_digest
        else:
            probe = {"status": "failed", "reason": "declared_video_source_not_found"}
        if probe.get("status") != "ready":
            checks.append(
                _check(
                    "media_decodable",
                    "fail",
                    evidence_source="decoded_media_probe",
                    measurement={"reason": str(probe.get("reason") or "probe_not_ready")},
                    claim_impact=("task_candidate_discovery", "captured_observation_review"),
                    recapture_code="media_not_decodable",
                    recapture_instruction="Upload the retained original media file. If the original is corrupt or unsupported, repeat the capture using H.264, HEVC, AV1, or VP9 video in the supported container.",
                )
            )
        else:
            duration = _float(probe.get("duration_seconds"))
            width = _integer(probe.get("width"))
            height = _integer(probe.get("height"))
            frame_rate = _float(probe.get("frame_rate"))
            codec = str(probe.get("codec_name") or "").lower()
            checks.extend(
                [
                    _check(
                        "media_decodable",
                        "pass",
                        evidence_source="decoded_media_probe",
                        measurement={"format_name": probe.get("format_name")},
                        claim_impact=("task_candidate_discovery", "captured_observation_review"),
                    ),
                    _check(
                        "duration",
                        "pass" if duration is not None and duration >= 5.0 else "fail",
                        evidence_source="decoded_media_probe",
                        measurement=duration,
                        threshold={"minimum_seconds": 5.0},
                        claim_impact=("visual_coverage",),
                        recapture_code=None
                        if duration is not None and duration >= 5.0
                        else "capture_too_short",
                        recapture_instruction=None
                        if duration is not None and duration >= 5.0
                        else "Repeat the capture with a complete overlapping pass lasting at least five seconds.",
                    ),
                    _check(
                        "resolution",
                        "pass"
                        if width is not None and height is not None and min(width, height) >= 720
                        else "fail",
                        evidence_source="decoded_media_probe",
                        measurement={"width": width, "height": height},
                        threshold={"minimum_short_edge_pixels": 720},
                        claim_impact=("object_identification", "captured_observation_review"),
                        recapture_code=None
                        if width is not None and height is not None and min(width, height) >= 720
                        else "resolution_too_low",
                        recapture_instruction=None
                        if width is not None and height is not None and min(width, height) >= 720
                        else "Repeat or re-export the capture at 720p or higher without upscaling a lower-resolution source.",
                    ),
                    _check(
                        "frame_rate",
                        "pass"
                        if frame_rate is not None and 15.0 <= frame_rate <= 120.0
                        else "fail",
                        evidence_source="decoded_media_probe",
                        measurement=frame_rate,
                        threshold={"minimum_fps": 15.0, "maximum_fps": 120.0},
                        claim_impact=("captured_observation_timing",),
                        recapture_code=None
                        if frame_rate is not None and 15.0 <= frame_rate <= 120.0
                        else "frame_rate_unsupported",
                        recapture_instruction=None
                        if frame_rate is not None and 15.0 <= frame_rate <= 120.0
                        else "Re-export the retained source at its native 15–120 fps timing; do not interpolate missing frames.",
                    ),
                    _check(
                        "codec",
                        "pass" if codec in _SUPPORTED_CODECS else "fail",
                        evidence_source="decoded_media_probe",
                        measurement=codec or None,
                        threshold={"supported": sorted(_SUPPORTED_CODECS)},
                        claim_impact=("media_processing",),
                        recapture_code=None if codec in _SUPPORTED_CODECS else "codec_unsupported",
                        recapture_instruction=None
                        if codec in _SUPPORTED_CODECS
                        else "Export a hash-bound derivative using H.264, HEVC, AV1, or VP9 while retaining the original file unchanged.",
                    ),
                    _pts_check(probe),
                ]
            )

    checks.extend(
        [
            _measurement_check(
                measurements,
                "sharp_frame_fraction",
                minimum=0.70,
                claim_impact=("object_identification", "captured_observation_review"),
                recapture_code="excessive_blur",
                recapture_instruction="Repeat the affected pass more slowly, keep the camera stable, and pause briefly at task-critical objects.",
            ),
            _measurement_check(
                measurements,
                "well_exposed_frame_fraction",
                minimum=0.80,
                claim_impact=("object_identification", "captured_observation_review"),
                recapture_code="poor_exposure",
                recapture_instruction="Repeat the affected views with even lighting and locked exposure so highlights and dark surfaces retain detail.",
            ),
            _measurement_check(
                measurements,
                "visual_overlap_fraction",
                minimum=0.60,
                claim_impact=("reconstruction", "spatial_coverage"),
                recapture_code="low_visual_overlap",
                recapture_instruction="Slow down and add overlapping passes, keeping previously seen surfaces in view while changing position.",
            ),
            _measurement_check(
                measurements,
                "compression_quality_fraction",
                minimum=0.80,
                claim_impact=("object_identification", "reconstruction"),
                recapture_code="excessive_compression",
                recapture_instruction="Upload the retained original or export once at a higher bitrate; do not repeatedly transcode the capture.",
            ),
            _measurement_check(
                measurements,
                "rolling_shutter_symptom_fraction",
                maximum=0.10,
                claim_impact=("camera_motion", "reconstruction"),
                recapture_code="rolling_shutter_symptoms",
                recapture_instruction="Repeat the affected pass with slower camera rotation and steadier translation under brighter lighting.",
            ),
            _boolean_check(
                measurements,
                "privacy_sensitive_content_detected",
                expected=False,
                claim_impact=("external_provider_use", "artifact_download"),
                recapture_code="privacy_sensitive_content",
                recapture_instruction="Recapture without faces, license plates, screens, documents, or bystanders, or create and approve a privacy-safe derivative before provider use.",
            ),
            _boolean_check(
                measurements,
                "dynamic_people_detected",
                expected=False,
                claim_impact=("reconstruction", "task_scene_stability"),
                recapture_code="dynamic_people_present",
                recapture_instruction="Repeat the affected pass with people outside the task area and preserve the original as restricted evidence if retention permits.",
            ),
            _boolean_check(
                measurements,
                "moving_task_objects_detected",
                expected=False,
                claim_impact=("reconstruction", "task_reset"),
                recapture_code="task_objects_moved_during_capture",
                recapture_instruction="Reset task objects to one documented state and repeat the affected views without moving them during the pass.",
            ),
            _boolean_check(
                measurements,
                "task_critical_occlusion_detected",
                expected=False,
                claim_impact=("task_object_geometry", "task_candidate_discovery"),
                recapture_code="task_critical_occlusion",
                recapture_instruction="Capture the underside or rear of the work surface and a close orbit around each occluded task object.",
            ),
            _boolean_check(
                measurements,
                "robot_placement_area_covered",
                expected=True,
                claim_impact=("robot_placement", "reachability"),
                recapture_code="robot_placement_area_missing",
                recapture_instruction="Capture the full proposed robot placement area, floor support, access path, human clearance, and approach direction.",
            ),
        ]
    )

    stream_state = {
        str(row.get("stream_type") or ""): str(row.get("status") or "")
        for row in _rows(envelope.get("available_sensor_streams"))
    }
    checks.extend(
        [
            _check(
                "camera_pose_availability",
                "pass" if stream_state.get("camera_poses") == "available" else "not_measured",
                evidence_source="capture_intake_envelope",
                measurement=stream_state.get("camera_poses", "undeclared"),
                claim_impact=("calibrated_camera_poses", "metric_geometry"),
            ),
            _check(
                "depth_availability",
                "pass" if stream_state.get("depth") == "available" else "not_measured",
                evidence_source="capture_intake_envelope",
                measurement=stream_state.get("depth", "undeclared"),
                claim_impact=("metric_geometry",),
            ),
            _check(
                "intrinsics_availability",
                "pass" if stream_state.get("camera_intrinsics") == "available" else "not_measured",
                evidence_source="capture_intake_envelope",
                measurement=stream_state.get("camera_intrinsics", "undeclared"),
                claim_impact=("calibrated_camera_poses", "reconstruction"),
            ),
            _check(
                "scale_anchor_verified",
                "pass" if measurements.get("scale_anchor_verified") is True else "not_measured",
                evidence_source="quality_observations",
                measurement=(True if measurements.get("scale_anchor_verified") is True else None),
                threshold={"required_for": ["metric_scale", "metric_geometry"]},
                claim_impact=("metric_scale", "metric_geometry"),
                recapture_code="scale_anchor_missing",
                recapture_instruction="Include the metric calibration board in multiple sharp views and provide its exact dimensions and units.",
            ),
        ]
    )

    existing_codes = {str(row.get("code") or "") for row in recapture_plan}
    for row in checks:
        if row["status"] != "fail" or not row.get("recapture_code"):
            continue
        code = str(row["recapture_code"])
        if code in existing_codes:
            continue
        recapture_plan.append(
            {
                "code": code,
                "instruction": row["recapture_instruction"],
                "reason": f"Capture QA check {row['check_id']} failed for the requested evidence envelope.",
            }
        )
        existing_codes.add(code)

    missing_evidence = sorted(
        {row["check_id"] for row in checks if row["status"] == "not_measured"}
    )
    if admission["status"] == "rejected":
        status = "rejected"
        state = "failed"
    elif recapture_plan:
        status = "recapture_required"
        state = "rejected_or_recapture_required"
    else:
        status = "accepted"
        state = "capture_accepted"
    claim_ceiling = json.loads(json.dumps(admission["claim_ceiling"]))
    if status != "accepted":
        claim_ceiling["capture_admitted"] = False
        claim_ceiling["task_candidate_discovery"] = False
        claim_ceiling["captured_observation_review"] = False
        claim_ceiling["calibrated_camera_poses"] = False
        claim_ceiling["metric_scale"] = False
        claim_ceiling["metric_geometry"] = False
    next_experiment = (
        {
            "kind": "targeted_recapture",
            "code": recapture_plan[0]["code"],
            "instruction": recapture_plan[0]["instruction"],
        }
        if recapture_plan
        else (
            {
                "kind": "local_or_operator_measurement",
                "code": f"measure_{missing_evidence[0]}",
                "instruction": f"Measure and record {missing_evidence[0]} with a provenance-labeled local analyzer or operator review before making claims that depend on it.",
            }
            if missing_evidence
            else None
        )
    )
    report = {
        "schema_version": CAPTURE_QA_SCHEMA_VERSION,
        "intake_id": envelope["intake_id"],
        "envelope_digest": envelope["envelope_digest"],
        "capture_authority_profile": profile,
        "status": status,
        "state": state,
        "checks": sorted(checks, key=lambda row: row["check_id"]),
        "recapture_plan": sorted(recapture_plan, key=lambda row: str(row.get("code") or "")),
        "missing_evidence": missing_evidence,
        "next_cheapest_experiment": next_experiment,
        "quality_observations_digest": observations.get("observations_digest"),
        "claim_ceiling": claim_ceiling,
        "prohibited_claims": [
            "physical_task_success",
            "deployment_readiness",
            "safety_certification",
            "general_policy_ranking_validity",
        ],
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    report["qa_report_digest"] = _digest(report, omit="qa_report_digest")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--quality-observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
    observations = (
        json.loads(args.quality_observations.read_text(encoding="utf-8"))
        if args.quality_observations
        else None
    )
    report = build_capture_qa_report(
        envelope,
        upload_root=args.upload_root,
        quality_observations=observations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_canonical_json(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAPTURE_QA_SCHEMA_VERSION",
    "CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION",
    "build_capture_qa_report",
]
