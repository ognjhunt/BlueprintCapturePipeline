"""Task-scoped RGB/depth/LiDAR pairing and site-evidence bridge.

The pairing record binds raw sensor samples, clocks, calibration artifacts, and
the exact modalities requested by a task.  Timestamp correction is explicit;
the implementation never nearest-neighbor pairs an omitted sample or invents a
calibration.  Only independently verified, signed, physical calibration rows
may validate ``sensor_calibration`` and ``sensor_timing`` in a site profile.

This is supporting site evidence.  It creates no Q-SENSOR result, method
qualification, R5/R6/R7 authority, policy result, task success, or physical
success claim.
"""

from __future__ import annotations

import json
from statistics import fmean
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_site_measurement_routing import validate_site_evidence_profile


PAIRING_SCHEMA_VERSION = "measurement_sensor_stream_pairing.v1"
SUPPORTED_MODALITIES = frozenset({"rgb", "depth", "lidar", "event_camera"})


class MeasurementSensorPairingError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementSensorPairingError("sensor_pairing_not_json") from exc


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _valid_digest(value: Any) -> bool:
    raw = _string(value)
    return bool(
        len(raw) == 71
        and raw.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in raw[7:])
    )


def _integer(value: Any, *, minimum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if minimum is not None and value < minimum:
        return None
    return value


def _required_modalities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(_string(item) for item in value if _string(item)))


def _calibration_verified(stream: Mapping[str, Any]) -> bool:
    calibration = stream.get("independent_calibration")
    return bool(
        isinstance(calibration, Mapping)
        and _string(calibration.get("evaluator_id"))
        and calibration.get("candidate_method_independent") is True
        and calibration.get("agent_is_evaluator") is False
        and calibration.get("signature_status") == "verified"
        and _string(calibration.get("approval_signature_id"))
    )


def _analyze(value: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    record = _clone(value)
    record.pop("pairing_digest", None)
    record.pop("decision", None)
    record.pop("blockers", None)
    record.pop("pair_summaries", None)
    record.pop("aggregate_timing", None)
    errors: list[str] = []
    blockers: list[str] = []
    if record.get("schema_version") != PAIRING_SCHEMA_VERSION:
        errors.append("sensor_pairing_schema_invalid")
    for key in ("pairing_id", "site_frame_id", "clock_domain"):
        if not _string(record.get(key)):
            errors.append(f"sensor_pairing_{key}_missing")
    if not _valid_digest(record.get("source_capture_digest")):
        errors.append("sensor_pairing_source_capture_digest_invalid")
    for key in ("physical_measurements_included", "development_only"):
        if record.get(key) not in {True, False}:
            errors.append(f"sensor_pairing_{key}_invalid")
    for key in (
        "candidate_self_graded",
        "agent_generated_calibration",
        "thresholds_modified_after_measurement",
        "agent_may_approve",
        "q_sensor_qualification_created",
        "r5_evidence_created",
        "r6_decision_created",
        "r7_admission_created",
        "physical_success_established",
    ):
        if record.get(key) is not False:
            errors.append(f"sensor_pairing_{key}_must_be_false")
    tolerance = _integer(record.get("maximum_pair_delta_ns"), minimum=0)
    if tolerance is None:
        errors.append("sensor_pairing_tolerance_invalid")
        tolerance = 0
    required = _required_modalities(record.get("required_modalities"))
    if (
        not required
        or len(required) != len(record.get("required_modalities") or [])
        or not set(required) <= SUPPORTED_MODALITIES
    ):
        errors.append("sensor_pairing_required_modalities_invalid")

    raw_streams = record.get("streams")
    streams = raw_streams if isinstance(raw_streams, list) else []
    if not streams or not all(isinstance(row, Mapping) for row in streams):
        errors.append("sensor_pairing_streams_invalid")
        streams = []
    stream_by_id: dict[str, dict[str, Any]] = {}
    sample_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    modalities_present: set[str] = set()
    for index, raw_stream in enumerate(streams):
        stream = dict(raw_stream)
        sensor_id = _string(stream.get("sensor_id"))
        modality = _string(stream.get("modality"))
        if not sensor_id or sensor_id in stream_by_id:
            errors.append(f"sensor_pairing_sensor_id_invalid:{index}")
            continue
        stream_by_id[sensor_id] = stream
        if modality not in SUPPORTED_MODALITIES:
            errors.append(f"sensor_pairing_modality_invalid:{sensor_id}")
        else:
            modalities_present.add(modality)
        for key in ("stream_digest", "calibration_digest", "extrinsics_digest"):
            if not _valid_digest(stream.get(key)):
                errors.append(f"sensor_pairing_{key}_invalid:{sensor_id}")
        if modality in {"rgb", "depth", "event_camera"} and not _valid_digest(
            stream.get("intrinsics_digest")
        ):
            errors.append(f"sensor_pairing_intrinsics_digest_invalid:{sensor_id}")
        if stream.get("clock_domain") != record.get("clock_domain"):
            blockers.append(f"sensor_pairing_clock_domain_mismatch:{sensor_id}")
        offset = _integer(stream.get("time_offset_ns"))
        uncertainty = _integer(stream.get("timestamp_uncertainty_ns"), minimum=0)
        if offset is None or uncertainty is None:
            errors.append(f"sensor_pairing_clock_correction_invalid:{sensor_id}")
            offset = 0
            uncertainty = 0
        samples = stream.get("samples")
        if not isinstance(samples, list) or not samples:
            errors.append(f"sensor_pairing_samples_invalid:{sensor_id}")
            continue
        previous_timestamp: int | None = None
        for sample_index, raw_sample in enumerate(samples):
            if not isinstance(raw_sample, Mapping):
                errors.append(f"sensor_pairing_sample_invalid:{sensor_id}:{sample_index}")
                continue
            sample = dict(raw_sample)
            sample_id = _string(sample.get("sample_id"))
            timestamp = _integer(sample.get("timestamp_ns"), minimum=0)
            if not sample_id or (sensor_id, sample_id) in sample_by_key:
                errors.append(f"sensor_pairing_sample_id_invalid:{sensor_id}:{sample_index}")
                continue
            if timestamp is None or not _valid_digest(sample.get("artifact_digest")):
                errors.append(f"sensor_pairing_sample_payload_invalid:{sensor_id}:{sample_id}")
                continue
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                errors.append(f"sensor_pairing_timestamps_not_strictly_monotonic:{sensor_id}")
            previous_timestamp = timestamp
            corrected_timestamp = timestamp + offset
            if corrected_timestamp < 0:
                errors.append(f"sensor_pairing_corrected_timestamp_negative:{sensor_id}")
                continue
            sample["corrected_timestamp_ns"] = corrected_timestamp
            sample["timestamp_uncertainty_ns"] = uncertainty
            sample_by_key[(sensor_id, sample_id)] = sample
    missing_modalities = sorted(set(required) - modalities_present)
    blockers.extend(f"sensor_pairing_required_modality_missing:{item}" for item in missing_modalities)

    raw_groups = record.get("pair_groups")
    groups = raw_groups if isinstance(raw_groups, list) else []
    if len(groups) < 2 or not all(isinstance(row, Mapping) for row in groups):
        errors.append("sensor_pairing_pair_groups_invalid")
        groups = []
    group_ids: set[str] = set()
    used_samples: set[tuple[str, str]] = set()
    summaries: list[dict[str, Any]] = []
    for index, raw_group in enumerate(groups):
        group = dict(raw_group)
        group_id = _string(group.get("group_id"))
        if not group_id or group_id in group_ids:
            errors.append(f"sensor_pairing_group_id_invalid:{index}")
            continue
        group_ids.add(group_id)
        bindings = group.get("samples")
        if not isinstance(bindings, Mapping) or set(bindings) != set(required):
            blockers.append(f"sensor_pairing_group_modalities_incomplete:{group_id}")
            continue
        corrected: list[int] = []
        lower_bounds: list[int] = []
        upper_bounds: list[int] = []
        valid_group = True
        normalized_bindings: dict[str, dict[str, str]] = {}
        for modality in required:
            binding = bindings.get(modality)
            if not isinstance(binding, Mapping):
                blockers.append(f"sensor_pairing_binding_invalid:{group_id}:{modality}")
                valid_group = False
                continue
            sensor_id = _string(binding.get("sensor_id"))
            sample_id = _string(binding.get("sample_id"))
            stream = stream_by_id.get(sensor_id)
            sample = sample_by_key.get((sensor_id, sample_id))
            if stream is None or stream.get("modality") != modality or sample is None:
                blockers.append(f"sensor_pairing_sample_binding_missing:{group_id}:{modality}")
                valid_group = False
                continue
            key = (sensor_id, sample_id)
            if key in used_samples:
                blockers.append(f"sensor_pairing_sample_reused:{sensor_id}:{sample_id}")
                valid_group = False
            used_samples.add(key)
            timestamp = int(sample["corrected_timestamp_ns"])
            uncertainty = int(sample["timestamp_uncertainty_ns"])
            corrected.append(timestamp)
            lower_bounds.append(timestamp - uncertainty)
            upper_bounds.append(timestamp + uncertainty)
            normalized_bindings[modality] = {"sensor_id": sensor_id, "sample_id": sample_id}
        if not valid_group or len(corrected) != len(required):
            continue
        observed_span = max(corrected) - min(corrected)
        worst_case_span = max(upper_bounds) - min(lower_bounds)
        synchronized = worst_case_span <= tolerance
        if not synchronized:
            blockers.append(f"sensor_pairing_tolerance_exceeded:{group_id}")
        summaries.append(
            {
                "group_id": group_id,
                "samples": normalized_bindings,
                "observed_corrected_span_ns": observed_span,
                "worst_case_span_ns": worst_case_span,
                "within_tolerance": synchronized,
            }
        )
    if len(summaries) != len(groups):
        blockers.append("sensor_pairing_not_all_groups_resolved")
    if errors:
        raise MeasurementSensorPairingError(*errors)
    return record, sorted(set(blockers)), summaries


def build_sensor_stream_pairing_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record, blockers, summaries = _analyze(value)
    development = record.get("development_only") is True
    physical = record.get("physical_measurements_included") is True
    governance_valid = all(
        record.get(key) is False
        for key in (
            "candidate_self_graded",
            "agent_generated_calibration",
            "thresholds_modified_after_measurement",
            "agent_may_approve",
            "q_sensor_qualification_created",
            "r5_evidence_created",
            "r6_decision_created",
            "r7_admission_created",
            "physical_success_established",
        )
    )
    if not governance_valid:
        raise MeasurementSensorPairingError("sensor_pairing_governance_boundary_invalid")
    calibrations_verified = all(
        _calibration_verified(stream)
        for stream in record["streams"]
        if stream.get("modality") in record["required_modalities"]
    )
    accepted = bool(not blockers and not development and physical and calibrations_verified)
    decision = "accepted" if accepted else "development_only" if development else "rejected"
    spans = [int(row["worst_case_span_ns"]) for row in summaries]
    result = {
        **record,
        "pair_summaries": summaries,
        "aggregate_timing": {
            "pair_count": len(summaries),
            "maximum_worst_case_span_ns": max(spans) if spans else None,
            "mean_worst_case_span_ns": fmean(spans) if spans else None,
        },
        "blockers": blockers,
        "calibrations_independently_verified": calibrations_verified,
        "all_requested_modalities_paired": not blockers,
        "decision": decision,
        "proof_effect": "sensor_stream_pairing_evidence",
        "claim_ceiling": "sensor_pairing_support_only",
    }
    result["pairing_digest"] = canonical_digest(result, digest_field="pairing_digest")
    return validate_sensor_stream_pairing_record(result)


def validate_sensor_stream_pairing_record(value: Mapping[str, Any]) -> dict[str, Any]:
    supplied = _clone(value)
    supplied_digest = supplied.get("pairing_digest")
    rebuilt = build_sensor_stream_pairing_record(
        {
            key: item
            for key, item in supplied.items()
            if key
            not in {
                "pairing_digest",
                "pair_summaries",
                "aggregate_timing",
                "blockers",
                "calibrations_independently_verified",
                "all_requested_modalities_paired",
                "decision",
                "proof_effect",
                "claim_ceiling",
            }
        }
    ) if supplied_digest is None else None
    if rebuilt is not None:
        return rebuilt
    record, blockers, summaries = _analyze(supplied)
    development = record.get("development_only") is True
    physical = record.get("physical_measurements_included") is True
    calibrations_verified = all(
        _calibration_verified(stream)
        for stream in record["streams"]
        if stream.get("modality") in record["required_modalities"]
    )
    expected_decision = (
        "accepted"
        if not blockers and not development and physical and calibrations_verified
        else "development_only"
        if development
        else "rejected"
    )
    spans = [int(row["worst_case_span_ns"]) for row in summaries]
    expected_aggregate = {
        "pair_count": len(summaries),
        "maximum_worst_case_span_ns": max(spans) if spans else None,
        "mean_worst_case_span_ns": fmean(spans) if spans else None,
    }
    errors: list[str] = []
    checks = {
        "pair_summaries": summaries,
        "aggregate_timing": expected_aggregate,
        "blockers": blockers,
        "calibrations_independently_verified": calibrations_verified,
        "all_requested_modalities_paired": not blockers,
        "decision": expected_decision,
        "proof_effect": "sensor_stream_pairing_evidence",
        "claim_ceiling": "sensor_pairing_support_only",
    }
    for key, expected in checks.items():
        if supplied.get(key) != expected:
            errors.append(f"sensor_pairing_{key}_not_deterministic")
    if supplied_digest != canonical_digest(supplied, digest_field="pairing_digest"):
        errors.append("sensor_pairing_digest_mismatch")
    if errors:
        raise MeasurementSensorPairingError(*errors)
    return supplied


def bind_sensor_pairing_to_site_evidence(
    site_value: Mapping[str, Any], pairing_value: Mapping[str, Any]
) -> dict[str, Any]:
    site = validate_site_evidence_profile(site_value)
    pairing = validate_sensor_stream_pairing_record(pairing_value)
    if site.get("source_capture_digest") != pairing.get("source_capture_digest"):
        raise MeasurementSensorPairingError("sensor_pairing_site_capture_mismatch")
    accepted = pairing["decision"] == "accepted"
    evidence = dict(site.get("evidence") or {})
    protected = ("sensor_calibration", "sensor_timing")
    for evidence_id in protected:
        existing = evidence.get(evidence_id)
        if isinstance(existing, Mapping) and existing.get("validated") is True:
            if existing.get("record_id") != pairing["pairing_digest"]:
                raise MeasurementSensorPairingError(
                    f"sensor_pairing_validated_evidence_override:{evidence_id}"
                )
        evidence[evidence_id] = {
            "available": True,
            "validated": accepted,
            "record_id": pairing["pairing_digest"],
            "scope": {
                "site_frame_id": pairing["site_frame_id"],
                "clock_domain": pairing["clock_domain"],
                "modalities": list(pairing["required_modalities"]),
                "maximum_pair_delta_ns": pairing["maximum_pair_delta_ns"],
            },
        }
    updated = _clone(site)
    updated.pop("site_evidence_digest", None)
    updated["evidence"] = evidence
    updated["sensor_pairing_bridge"] = {
        "pairing_digest": pairing["pairing_digest"],
        "required_modalities": list(pairing["required_modalities"]),
        "decision": pairing["decision"],
        "development_only": pairing["development_only"],
        "q_sensor_qualification_created": False,
        "r5_evidence_created": False,
        "r6_decision_created": False,
        "r7_admission_created": False,
        "physical_success_established": False,
        "agent_may_promote": False,
    }
    return validate_site_evidence_profile(updated)


__all__ = [
    "MeasurementSensorPairingError",
    "PAIRING_SCHEMA_VERSION",
    "SUPPORTED_MODALITIES",
    "bind_sensor_pairing_to_site_evidence",
    "build_sensor_stream_pairing_record",
    "validate_sensor_stream_pairing_record",
]
