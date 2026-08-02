"""Validation helpers for optional qualified-geometry site-evidence fields."""

from __future__ import annotations

from typing import Any, Mapping


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def site_geometry_bridge_validation_errors(value: Mapping[str, Any]) -> list[str]:
    """Return stable validation codes without granting routing authority."""

    errors: list[str] = []
    evidence = value.get("evidence")
    bridge = value.get("geometry_bridge")
    if bridge is not None:
        if not isinstance(bridge, Mapping):
            errors.append("geometry_bridge_invalid")
        else:
            bridge_digests = {
                "metric_geometry_manifest_digest": "metric_scale",
                "collider_candidate_manifest_digest": "validated_mesh",
                "collider_qualification_digest": "validated_collider",
                "robot_site_registration_digest": "robot_site_registration",
            }
            for digest_key, evidence_id in bridge_digests.items():
                digest_value = _string(bridge.get(digest_key))
                if not digest_value.startswith("sha256:"):
                    errors.append(f"geometry_bridge_{digest_key}_invalid")
                    continue
                record = evidence.get(evidence_id) if isinstance(evidence, Mapping) else None
                if not isinstance(record, Mapping) or record.get("record_id") != digest_value:
                    errors.append(f"geometry_bridge_{evidence_id}_record_mismatch")
            coordinate = value.get("coordinate_system")
            coordinate = dict(coordinate) if isinstance(coordinate, Mapping) else {}
            if coordinate.get("registration_digest") != bridge.get(
                "robot_site_registration_digest"
            ):
                errors.append("geometry_bridge_coordinate_registration_mismatch")
            for key in (
                "method_qualification_created",
                "r5_evidence_created",
                "r6_decision_created",
                "r7_admission_created",
                "physical_success_established",
                "deployment_readiness_established",
                "safety_established",
                "agent_may_promote",
            ):
                if bridge.get(key) is not False:
                    errors.append(f"geometry_bridge_{key}_must_be_false")
            if bridge.get("development_only") not in {True, False}:
                errors.append("geometry_bridge_development_only_invalid")
            registration = (
                evidence.get("robot_site_registration") if isinstance(evidence, Mapping) else None
            )
            if (
                bridge.get("development_only") is True
                and isinstance(registration, Mapping)
                and registration.get("validated") is True
            ):
                errors.append("geometry_bridge_development_registration_cannot_validate")
    source_digest = value.get("source_capture_digest")
    if source_digest is not None:
        raw_digest = _string(source_digest)
        if (
            len(raw_digest) != 71
            or not raw_digest.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in raw_digest[7:])
        ):
            errors.append("source_capture_digest_invalid")
    return errors


def site_sensor_pairing_bridge_validation_errors(value: Mapping[str, Any]) -> list[str]:
    """Validate the optional sensor-pairing bridge without granting authority."""

    bridge = value.get("sensor_pairing_bridge")
    if bridge is None:
        return []
    if not isinstance(bridge, Mapping):
        return ["sensor_pairing_bridge_invalid"]
    errors: list[str] = []
    pairing_digest = _string(bridge.get("pairing_digest"))
    if (
        len(pairing_digest) != 71
        or not pairing_digest.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in pairing_digest[7:])
    ):
        errors.append("sensor_pairing_bridge_digest_invalid")
    modalities = bridge.get("required_modalities")
    if (
        not isinstance(modalities, list)
        or not modalities
        or len(set(modalities)) != len(modalities)
        or not set(modalities) <= {"rgb", "depth", "lidar", "event_camera"}
    ):
        errors.append("sensor_pairing_bridge_modalities_invalid")
    if bridge.get("decision") not in {"accepted", "rejected", "development_only"}:
        errors.append("sensor_pairing_bridge_decision_invalid")
    if bridge.get("development_only") not in {True, False}:
        errors.append("sensor_pairing_bridge_development_only_invalid")
    evidence = value.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    for evidence_id in ("sensor_calibration", "sensor_timing"):
        record = evidence.get(evidence_id)
        if not isinstance(record, Mapping) or record.get("record_id") != pairing_digest:
            errors.append(f"sensor_pairing_bridge_{evidence_id}_record_mismatch")
            continue
        expected_validated = bridge.get("decision") == "accepted"
        if record.get("validated") is not expected_validated:
            errors.append(f"sensor_pairing_bridge_{evidence_id}_validation_mismatch")
    for key in (
        "q_sensor_qualification_created",
        "r5_evidence_created",
        "r6_decision_created",
        "r7_admission_created",
        "physical_success_established",
        "agent_may_promote",
    ):
        if bridge.get(key) is not False:
            errors.append(f"sensor_pairing_bridge_{key}_must_be_false")
    return errors


def site_evidence_bridge_validation_errors(value: Mapping[str, Any]) -> list[str]:
    """Validate every optional evidence bridge attached to a site profile."""

    return [
        *site_geometry_bridge_validation_errors(value),
        *site_sensor_pairing_bridge_validation_errors(value),
    ]
