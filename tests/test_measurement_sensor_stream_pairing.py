from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_sensor_stream_pairing import (
    MeasurementSensorPairingError,
    bind_sensor_pairing_to_site_evidence,
    build_sensor_stream_pairing_record,
    validate_sensor_stream_pairing_record,
)
from blueprint_pipeline.task_site_measurement_routing import (
    MeasurementRoutingError,
    audit_site_evidence_profile,
    validate_site_evidence_profile,
)


ROOT = Path(__file__).parents[1]
CAPTURE_DIGEST = "sha256:" + "1" * 64


def _digest(index: int) -> str:
    return "sha256:" + f"{index:x}"[-1] * 64


def _stream(sensor_id: str, modality: str, index: int) -> dict:
    timestamps = {
        "rgb": [1_000_000, 2_000_000],
        "depth": [1_000_500, 2_000_500],
        "lidar": [999_900, 1_999_900],
    }[modality]
    offsets = {"rgb": 0, "depth": -500, "lidar": 100}
    value = {
        "sensor_id": sensor_id,
        "modality": modality,
        "stream_digest": _digest(index),
        "calibration_digest": _digest(index + 3),
        "extrinsics_digest": _digest(index + 6),
        "clock_domain": "site-ptp-clock",
        "time_offset_ns": offsets[modality],
        "timestamp_uncertainty_ns": 50,
        "independent_calibration": {
            "evaluator_id": f"sensor-calibration-evaluator-{sensor_id}",
            "candidate_method_independent": True,
            "agent_is_evaluator": False,
            "signature_status": "verified",
            "approval_signature_id": f"signed-calibration-{sensor_id}",
        },
        "samples": [
            {
                "sample_id": f"{sensor_id}-{sample_index}",
                "timestamp_ns": timestamp,
                "artifact_digest": _digest(index + 9 + sample_index),
            }
            for sample_index, timestamp in enumerate(timestamps, start=1)
        ],
    }
    if modality in {"rgb", "depth"}:
        value["intrinsics_digest"] = _digest(index + 12)
    return value


def _pairing() -> dict:
    sensor_ids = {"rgb": "camera-rgb", "depth": "camera-depth", "lidar": "lidar-top"}
    return {
        "schema_version": "measurement_sensor_stream_pairing.v1",
        "pairing_id": "site-sensor-pairing-001",
        "source_capture_digest": CAPTURE_DIGEST,
        "site_frame_id": "site-frame-001",
        "clock_domain": "site-ptp-clock",
        "required_modalities": ["rgb", "depth", "lidar"],
        "maximum_pair_delta_ns": 1_000,
        "streams": [
            _stream(sensor_ids[modality], modality, index)
            for index, modality in enumerate(("rgb", "depth", "lidar"), start=2)
        ],
        "pair_groups": [
            {
                "group_id": f"pair-{sample_index}",
                "samples": {
                    modality: {
                        "sensor_id": sensor_id,
                        "sample_id": f"{sensor_id}-{sample_index}",
                    }
                    for modality, sensor_id in sensor_ids.items()
                },
            }
            for sample_index in (1, 2)
        ],
        "physical_measurements_included": True,
        "development_only": False,
        "candidate_self_graded": False,
        "agent_generated_calibration": False,
        "thresholds_modified_after_measurement": False,
        "agent_may_approve": False,
        "q_sensor_qualification_created": False,
        "r5_evidence_created": False,
        "r6_decision_created": False,
        "r7_admission_created": False,
        "physical_success_established": False,
    }


def _site() -> dict:
    return validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": "sensor-site-001",
            "bundle_id": "capture-001",
            "bundle_hash": CAPTURE_DIGEST,
            "source_capture_digest": CAPTURE_DIGEST,
            "provenance_record_id": "provenance-001",
            "rights": {"commercial_evaluation_allowed": True},
            "privacy": {"classification": "internal"},
            "coordinate_system": {"metric_scale_verified": True, "frame": "site-frame-001"},
            "evidence": {
                "sensor_calibration": {
                    "available": True,
                    "validated": False,
                    "record_id": "unvalidated-calibration-candidate",
                },
                "sensor_timing": {
                    "available": True,
                    "validated": False,
                    "record_id": "unvalidated-timing-candidate",
                },
            },
            "limitations": {"known_missing_regions": [], "forbidden_claims": []},
        }
    )


def test_signed_physical_pairing_validates_task_scoped_sensor_site_evidence() -> None:
    raw = _pairing()
    schema = json.loads(
        (ROOT / "docs/schemas/measurement_sensor_stream_pairing.v1.schema.json").read_text()
    )
    jsonschema.validate(raw, schema)
    pairing = build_sensor_stream_pairing_record(raw)
    assert pairing["decision"] == "accepted"
    assert pairing["all_requested_modalities_paired"] is True
    assert pairing["calibrations_independently_verified"] is True
    assert pairing["aggregate_timing"] == {
        "pair_count": 2,
        "maximum_worst_case_span_ns": 100,
        "mean_worst_case_span_ns": 100.0,
    }
    assert validate_sensor_stream_pairing_record(pairing) == pairing

    site = bind_sensor_pairing_to_site_evidence(_site(), pairing)
    assert site["evidence"]["sensor_calibration"]["validated"] is True
    assert site["evidence"]["sensor_timing"]["validated"] is True
    assert site["sensor_pairing_bridge"]["pairing_digest"] == pairing["pairing_digest"]
    assert site["sensor_pairing_bridge"]["q_sensor_qualification_created"] is False
    gaps = {row["evidence_id"] for row in audit_site_evidence_profile(site)["gaps"]}
    assert "sensor_calibration" not in gaps


def test_development_pairing_never_validates_site_sensor_evidence() -> None:
    raw = _pairing()
    raw["development_only"] = True
    raw["physical_measurements_included"] = False
    for stream in raw["streams"]:
        stream["independent_calibration"]["signature_status"] = "unverified"
        stream["independent_calibration"]["approval_signature_id"] = ""
    pairing = build_sensor_stream_pairing_record(raw)
    assert pairing["decision"] == "development_only"
    assert pairing["all_requested_modalities_paired"] is True
    site = bind_sensor_pairing_to_site_evidence(_site(), pairing)
    assert site["evidence"]["sensor_calibration"]["validated"] is False
    assert site["evidence"]["sensor_timing"]["validated"] is False


def test_unsynchronized_or_incomplete_groups_reject_without_inventing_pairs() -> None:
    unsynchronized = _pairing()
    unsynchronized["streams"][2]["samples"][1]["timestamp_ns"] += 20_000
    pairing = build_sensor_stream_pairing_record(unsynchronized)
    assert pairing["decision"] == "rejected"
    assert "sensor_pairing_tolerance_exceeded:pair-2" in pairing["blockers"]
    assert bind_sensor_pairing_to_site_evidence(_site(), pairing)["evidence"][
        "sensor_timing"
    ]["validated"] is False

    incomplete = _pairing()
    incomplete["pair_groups"][0]["samples"].pop("lidar")
    pairing = build_sensor_stream_pairing_record(incomplete)
    assert pairing["decision"] == "rejected"
    assert "sensor_pairing_group_modalities_incomplete:pair-1" in pairing["blockers"]
    assert "sensor_pairing_not_all_groups_resolved" in pairing["blockers"]


def test_pairing_rejects_structural_and_governance_tampering() -> None:
    duplicate = _pairing()
    duplicate["streams"][0]["samples"][1]["timestamp_ns"] = duplicate["streams"][0][
        "samples"
    ][0]["timestamp_ns"]
    with pytest.raises(MeasurementSensorPairingError, match="not_strictly_monotonic"):
        build_sensor_stream_pairing_record(duplicate)

    agent_approved = _pairing()
    agent_approved["agent_may_approve"] = True
    with pytest.raises(MeasurementSensorPairingError, match="agent_may_approve_must_be_false"):
        build_sensor_stream_pairing_record(agent_approved)

    pairing = build_sensor_stream_pairing_record(_pairing())
    tampered = copy.deepcopy(pairing)
    tampered["decision"] = "rejected"
    with pytest.raises(MeasurementSensorPairingError, match="decision_not_deterministic"):
        validate_sensor_stream_pairing_record(tampered)


def test_pairing_bridge_rejects_lineage_and_profile_tampering() -> None:
    pairing = build_sensor_stream_pairing_record(_pairing())
    wrong_site = copy.deepcopy(_site())
    wrong_site.pop("site_evidence_digest")
    wrong_site["source_capture_digest"] = _digest(15)
    wrong_site = validate_site_evidence_profile(wrong_site)
    with pytest.raises(MeasurementSensorPairingError, match="site_capture_mismatch"):
        bind_sensor_pairing_to_site_evidence(wrong_site, pairing)

    bridged = bind_sensor_pairing_to_site_evidence(_site(), pairing)
    bridged.pop("site_evidence_digest")
    bridged["sensor_pairing_bridge"]["q_sensor_qualification_created"] = True
    with pytest.raises(MeasurementRoutingError, match="q_sensor_qualification_created"):
        validate_site_evidence_profile(bridged)
