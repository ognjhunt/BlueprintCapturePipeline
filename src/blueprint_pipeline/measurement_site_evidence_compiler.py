"""Deterministic compiler from pipeline artifacts to a site evidence profile.

The measurement router consumes ``site_evidence_profile.v1``. Until now those
profiles were hand-authored; this module derives them from the artifacts the
capture pipeline actually produces, fail-closed:

- an evidence record is ``validated=True`` only when the source artifact
  carries an explicit passing/qualified status — presence alone yields
  ``available=True, validated=False``;
- artifact kinds without a mapping rule are surfaced in ``unmapped_artifacts``
  and contribute nothing (unknown inputs are gaps, never guesses);
- SimReady draft manifests contribute their candidate evidence strictly
  ``validated=False`` via the SimReady lane's own invariants;
- the compiler never fabricates rights, privacy, metric scale, or provenance:
  those come from the capture governance inputs it is handed.

The result is what makes routing dynamic per real site: compile, attach to the
maintained testbed, and every claim routes (or abstains with the smallest next
action) against what that site's evidence can actually support.

Layering with ``measurement_site_evidence_bridge``: the bridge is the deep,
lineage-checked path from the reconstruction lane's typed geometry contracts
(metric geometry, collider qualification, SE(3) registration) to a profile;
this compiler is the broad path across the remaining artifact families
(articulation, materials, sensors, appearance, QA, SimReady drafts) plus the
testbed attachment helper. When typed reconstruction contracts exist for a
site, the bridge's geometry records are the stricter authority; records
produced here always carry ``derived_from_artifact`` and claim no lineage.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import MaintainedSiteTaskTestbed
from .simready_asset_lane import validate_simready_asset_manifest
from .task_site_measurement_routing import (
    SITE_EVIDENCE_TAXONOMY,
    validate_site_evidence_profile,
)


SITE_EVIDENCE_COMPILATION_SCHEMA_VERSION = "site_evidence_compilation_report.v1"

_PASS_VALUES = frozenset(
    {"passed", "pass", "qualified", "accepted", "verified", "valid", "ok"}
)


class SiteEvidenceCompilerError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _status_passed(payload: Mapping[str, Any], *fields: str) -> bool:
    return any(
        _string(payload.get(field)).lower() in _PASS_VALUES for field in fields
    )


def _record(
    evidence_id: str,
    record_id: str,
    *,
    available: bool,
    validated: bool,
    source: str,
    **metadata: Any,
) -> tuple[str, dict[str, Any]]:
    row = {
        "available": available,
        "validated": validated,
        "record_id": record_id,
        "derived_from_artifact": source,
    }
    row.update(metadata)
    return evidence_id, row


def _rule_capture_raw_manifest(
    payload: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Surface declared raw streams as available but never calibrated.

    A capture manifest is identity/provenance input.  Its stream declarations
    do not establish calibration, pose accuracy, coverage, or physical truth;
    those require their dedicated reports.
    """

    capabilities = payload.get("capture_capabilities")
    capabilities = dict(capabilities) if isinstance(capabilities, Mapping) else {}
    record_id = _string(
        payload.get("manifest_digest")
        or payload.get("capture_digest")
        or payload.get("capture_id")
    ) or "capture-raw-manifest"
    proof_boundary = payload.get("proof_boundary")
    proof_boundary = (
        dict(proof_boundary) if isinstance(proof_boundary, Mapping) else {}
    )
    fixture_labeled = (
        _string(payload.get("evidence_tier")) == "fixture_only"
        or proof_boundary.get("synthetic_fixture_only") is True
    )
    rows: list[tuple[str, dict[str, Any]]] = []
    if capabilities.get("walkthrough_video") is True or capabilities.get("rgb") is True:
        rows.append(
            _record(
                "calibrated_rgb",
                f"{record_id}-rgb-candidate",
                available=True,
                validated=False,
                source="capture_raw_manifest",
                raw_stream_only=True,
                fixture_labeled=fixture_labeled,
            )
        )
    if capabilities.get("camera_pose") is True:
        rows.append(
            _record(
                "camera_poses",
                f"{record_id}-poses-candidate",
                available=True,
                validated=False,
                source="capture_raw_manifest",
                raw_stream_only=True,
                fixture_labeled=fixture_labeled,
            )
        )
    return rows


def _rule_collider_qualification(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "qualification_status", "overall_status")
    record_id = _string(payload.get("report_id") or payload.get("request_id")) or "collider-qualification"
    return [
        _record(
            "validated_mesh", f"{record_id}-mesh",
            available=True, validated=passed, source="collider_qualification_report",
        ),
        _record(
            "validated_collider", record_id,
            available=True, validated=passed, source="collider_qualification_report",
        )
    ]


def _rule_metric_scale(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "validation_status", "scale_status")
    record_id = _string(payload.get("validation_id") or payload.get("request_id")) or "metric-scale-validation"
    return [
        _record(
            "metric_scale", record_id,
            available=True, validated=passed, source="metric_scale_validation",
        )
    ]


def _rule_registration(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "verification_status")
    record_id = _string(payload.get("registration_id")) or "robot-site-registration"
    return [
        _record(
            "robot_site_registration", record_id,
            available=True, validated=passed, source="robot_site_registration",
        )
    ]


def _rule_metric_geometry(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    # The production metric-geometry contract is explicitly a reference
    # candidate.  Only the collider-qualification/geometry bridge may promote
    # it to validated mesh evidence.
    record_id = _string(
        payload.get("metric_geometry_manifest_digest") or payload.get("manifest_id")
    ) or "metric-geometry-manifest"
    return [
        _record(
            "validated_mesh", record_id,
            available=True,
            validated=False,
            source="metric_geometry_manifest",
            metric_reference_candidate_only=True,
        )
    ]


def _rule_appearance(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "heldout_status", "evaluation_status", "status")
    record_id = _string(payload.get("manifest_id") or payload.get("asset_id")) or "appearance-manifest"
    return [
        _record(
            "gaussian_splat_appearance", record_id,
            available=True, validated=passed, source="appearance_manifest",
        )
    ]


def _rule_capture_qa(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "qa_status", "overall_status")
    record_id = _string(payload.get("report_id")) or "capture-qa-report"
    return [
        _record(
            "calibrated_rgb", f"{record_id}-rgb",
            available=True, validated=passed, source="capture_qa_report",
        ),
        _record(
            "camera_poses", f"{record_id}-poses",
            available=True, validated=passed, source="capture_qa_report",
        ),
    ]


def _rule_articulation(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "measurement_status")
    record_id = _string(payload.get("measurement_id")) or "articulation-measurement"
    rows = [
        _record(
            "articulation_model", f"{record_id}-model",
            available=True, validated=passed, source="articulation_measurement",
        )
    ]
    if payload.get("actuation_measured") is True:
        rows.append(
            _record(
                "articulation_actuation", f"{record_id}-actuation",
                available=True, validated=passed, source="articulation_measurement",
            )
        )
    return rows


def _rule_material_identification(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "identification_status")
    record_id = _string(payload.get("identification_id")) or "material-identification"
    rows = [
        _record(
            "material_parameters", f"{record_id}-parameters",
            available=True, validated=passed, source="material_identification",
        )
    ]
    if payload.get("friction_measured") is True:
        rows.append(
            _record(
                "friction_contact", f"{record_id}-friction",
                available=True, validated=passed, source="material_identification",
            )
        )
    if payload.get("mass_inertia_measured") is True:
        rows.append(
            _record(
                "mass_inertia", f"{record_id}-mass",
                available=True, validated=passed, source="material_identification",
            )
        )
    return rows


def _rule_sensor_calibration(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    passed = _status_passed(payload, "status", "calibration_status")
    record_id = _string(payload.get("calibration_id")) or "sensor-calibration"
    rows = [
        _record(
            "sensor_calibration", f"{record_id}-intrinsics",
            available=True, validated=passed, source="sensor_calibration",
        )
    ]
    if payload.get("timing_verified") is True:
        rows.append(
            _record(
                "sensor_timing", f"{record_id}-timing",
                available=True, validated=passed, source="sensor_calibration",
            )
        )
    return rows


def _rule_kitchen_preflight(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    # A scaling-preflight manifest proves a renderable modeled scene exists.
    # It is an appearance-layer fact only: never metric scale, never colliders.
    passed = _status_passed(payload, "local_preflight_status", "status")
    return [
        _record(
            "appearance_mesh", "kitchen-preflight-scene",
            available=True, validated=passed, source="kitchen_task_scaling_preflight_manifest",
        )
    ]


def _rule_simready_manifest(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    manifest = validate_simready_asset_manifest(payload)
    return [
        (
            row["evidence_id"],
            {
                "available": True,
                "validated": False,
                "record_id": row["record_id"],
                "derived_from_artifact": "simready_asset_manifest",
                "simready_asset_digest": manifest["simready_asset_digest"],
            },
        )
        for row in manifest["candidate_site_evidence"]
    ]


ARTIFACT_EVIDENCE_RULES = {
    "capture_bundle_manifest": _rule_capture_raw_manifest,
    "capture_raw_manifest": _rule_capture_raw_manifest,
    "collider_qualification_report": _rule_collider_qualification,
    "metric_scale_validation": _rule_metric_scale,
    "robot_site_registration": _rule_registration,
    "metric_geometry_manifest": _rule_metric_geometry,
    "appearance_manifest": _rule_appearance,
    "capture_qa_report": _rule_capture_qa,
    "articulation_measurement": _rule_articulation,
    "material_identification": _rule_material_identification,
    "sensor_calibration": _rule_sensor_calibration,
    "kitchen_task_scaling_preflight_manifest": _rule_kitchen_preflight,
    "simready_asset_manifest": _rule_simready_manifest,
}


def compile_site_evidence_profile(
    *,
    profile_id: str,
    bundle_id: str,
    bundle_hash: str,
    provenance_record_id: str,
    rights: Mapping[str, Any],
    privacy: Mapping[str, Any],
    metric_scale_verified: bool,
    artifacts: Mapping[str, Any],
    known_missing_regions: Sequence[str] = (),
    forbidden_claims: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile a validated site evidence profile plus a compilation report."""

    evidence: dict[str, dict[str, Any]] = {}
    mapped: list[dict[str, Any]] = []
    consumed: list[str] = []
    unmapped: list[str] = []
    for artifact_kind in sorted(artifacts):
        payload = artifacts[artifact_kind]
        if not isinstance(payload, Mapping):
            raise SiteEvidenceCompilerError(
                f"site_evidence_artifact_not_object:{artifact_kind}"
            )
        rule = ARTIFACT_EVIDENCE_RULES.get(_string(artifact_kind))
        if rule is None:
            unmapped.append(_string(artifact_kind))
            continue
        consumed.append(_string(artifact_kind))
        for evidence_id, row in rule(payload):
            if evidence_id not in SITE_EVIDENCE_TAXONOMY:
                raise SiteEvidenceCompilerError(
                    f"site_evidence_rule_produced_unknown_id:{evidence_id}"
                )
            existing = evidence.get(evidence_id)
            # Two artifacts about one evidence id: validated wins only when the
            # winning artifact itself passed; never fabricate an upgrade.
            if existing is None or (
                row["validated"] and not existing["validated"]
            ):
                evidence[evidence_id] = row
            mapped.append({"artifact_kind": _string(artifact_kind), "evidence_id": evidence_id})
    profile = validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": profile_id,
            "bundle_id": bundle_id,
            "bundle_hash": bundle_hash,
            "provenance_record_id": provenance_record_id,
            "rights": dict(rights),
            "privacy": dict(privacy),
            "coordinate_system": {"metric_scale_verified": bool(metric_scale_verified)},
            "evidence": evidence,
            "limitations": {
                "known_missing_regions": list(known_missing_regions),
                "forbidden_claims": list(forbidden_claims),
            },
        }
    )
    report = {
        "schema_version": SITE_EVIDENCE_COMPILATION_SCHEMA_VERSION,
        "profile_id": profile_id,
        "site_evidence_digest": profile["site_evidence_digest"],
        "mapped_artifacts": sorted(mapped, key=lambda row: (row["artifact_kind"], row["evidence_id"])),
        "consumed_artifacts": sorted(set(consumed)),
        "unmapped_artifacts": sorted(set(unmapped)),
        "evidence_record_count": len(evidence),
        "validated_record_count": sum(
            1 for row in evidence.values() if row["validated"]
        ),
        "fabricated_records": 0,
        "compiler_may_upgrade_validation": False,
    }
    report["site_evidence_compilation_digest"] = _digest(
        report, "site_evidence_compilation_digest"
    )
    return {"profile": profile, "report": report}


def attach_compiled_site_evidence(
    testbed_value: Mapping[str, Any], profile_value: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a compiled profile into a maintained testbed and revalidate."""

    profile = validate_site_evidence_profile(profile_value)
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    rebound = dict(testbed)
    rebound.pop("testbed_digest", None)
    rebound["site_evidence_profile"] = profile
    return MaintainedSiteTaskTestbed.from_mapping(rebound).to_mapping()


__all__ = [
    "ARTIFACT_EVIDENCE_RULES", "SITE_EVIDENCE_COMPILATION_SCHEMA_VERSION",
    "SiteEvidenceCompilerError", "attach_compiled_site_evidence",
    "compile_site_evidence_profile",
]
