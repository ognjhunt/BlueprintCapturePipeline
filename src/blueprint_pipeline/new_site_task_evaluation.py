"""End-to-end acceptance path for one newly materialized site and explicit task.

This module joins the existing capture-intake, site-evidence, task/site routing,
local evidence execution, Decision Envelope, and WebApp projection contracts.
The development lane is deliberately separate from production measurement
admission: it may execute a zero-spend local adapter only when the production
kernel's sole method blocker is the missing R7/R8 qualification.  Its result is
development evidence and cannot mutate the production routing decision.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from .capture_intake import CaptureIntakeError, materialize_capture_intake
from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
    canonical_digest,
    canonical_json,
)
from .decision_evidence_execution import build_decision_envelope, execute_evidence_plan
from .decision_evidence_router import route_decision_evidence
from .local_evidence_adapters import (
    CAPTURED_VISIBILITY_ADAPTER,
    authorized_local_evidence_adapter_registry,
)
from .measurement_site_evidence_compiler import compile_site_evidence_profile
from .task_evaluation_supervisor.capabilities import (
    DeterministicCaptureTestbedSupervisor,
    DeterministicClaimTaskInterpreter,
    DeterministicEvaluationMethodRouter,
    SupervisorContext,
)
from .task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    audit_site_evidence_profile,
    derive_task_measurement_requirements,
    validate_method_capability_profile,
)

SCHEMA_VERSION = "new_site_task_evaluation_result.v1"
MATERIALIZATION_SCHEMA_VERSION = "new_site_capture_materialization_validation.v1"
DEVELOPMENT_ROUTE_SCHEMA_VERSION = "new_site_development_route.v1"
OBSERVATION_SCHEMA_VERSION = "new_site_observation_manifest.v1"
SITE_ARTIFACTS_SCHEMA_VERSION = "new_site_observed_site_evidence.v1"
SITE_ARTIFACTS_OVERLAY_SCHEMA_VERSION = "new_site_observed_site_evidence_overlay.v1"
TASK_SPEC_SCHEMA_VERSION = "new_site_task_spec.v1"


class NewSiteTaskEvaluationError(ValueError):
    """Stable fail-closed new-site acceptance error."""

    def __init__(self, *codes: str) -> None:
        self.codes = tuple(sorted({str(code) for code in codes if str(code)}))
        super().__init__("; ".join(self.codes))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return (
        [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _digest(value: Mapping[str, Any], field: str) -> str:
    supplied = _text(value.get(field))
    expected = canonical_digest(value, digest_field=field)
    if supplied != expected:
        raise NewSiteTaskEvaluationError(f"digest_mismatch:{field}")
    return supplied


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NewSiteTaskEvaluationError(f"json_missing_or_symlink:{path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NewSiteTaskEvaluationError(f"json_invalid:{path.name}") from exc
    if not isinstance(value, Mapping):
        raise NewSiteTaskEvaluationError(f"json_not_object:{path.name}")
    return dict(value)


def _safe_file(root: Path, relative_path: str) -> Path:
    if (
        not relative_path
        or "\\" in relative_path
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
    ):
        raise NewSiteTaskEvaluationError("observation_reference_path_unsafe")
    candidate = root / relative_path
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise NewSiteTaskEvaluationError("observation_reference_symlink_forbidden")
        cursor = cursor.parent
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise NewSiteTaskEvaluationError("observation_reference_path_escape")
    if not resolved.is_file():
        raise NewSiteTaskEvaluationError(f"observation_reference_missing:{relative_path}")
    return resolved


def _finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def _validated_governance(value: Mapping[str, Any]) -> dict[str, Any]:
    governance = copy.deepcopy(dict(value))
    provider_constraints = _mapping(governance.get("provider_constraints"))
    allowed_uses = governance.get("allowed_uses")
    errors: list[str] = []
    if governance.get("rights") != "accepted":
        errors.append("capture_rights_not_accepted")
    if governance.get("consent") != "accepted":
        errors.append("capture_consent_not_accepted")
    if governance.get("privacy") != "cleared":
        errors.append("capture_privacy_not_cleared")
    if not isinstance(allowed_uses, list) or "evaluation" not in allowed_uses:
        errors.append("capture_evaluation_use_not_allowed")
    if not isinstance(provider_constraints.get("external_processing_allowed"), bool):
        errors.append("capture_external_processing_policy_missing")
    if errors:
        raise NewSiteTaskEvaluationError(*errors)
    return governance


def _validate_observations(
    value: Mapping[str, Any],
    *,
    upload_root: Path,
    declared_paths: set[str],
) -> dict[str, Any]:
    observations = copy.deepcopy(dict(value))
    errors: list[str] = []
    if observations.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        errors.append("observation_schema_version_invalid")
    frames = _rows(observations.get("frames"))
    if len(frames) < 2:
        errors.append("observation_frames_insufficient")
    frame_ids: list[str] = []
    timestamps: list[int] = []
    referenced_paths: list[str] = []
    for index, frame in enumerate(frames):
        frame_id = _text(frame.get("frame_id"))
        timestamp = frame.get("timestamp_ns")
        if not frame_id:
            errors.append(f"observation_frame_id_missing:{index}")
        frame_ids.append(frame_id)
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            errors.append(f"observation_timestamp_invalid:{index}")
        else:
            timestamps.append(timestamp)
        transform = frame.get("camera_to_site_transform")
        if not _finite_vector(transform, 16) or list(transform or [])[12:] != [0, 0, 0, 1]:
            errors.append(f"observation_transform_invalid:{index}")
        for field in ("rgb_path", "depth_path"):
            relative = _text(frame.get(field))
            if relative not in declared_paths:
                errors.append(f"observation_reference_not_declared:{index}:{field}")
                continue
            try:
                _safe_file(upload_root, relative)
            except NewSiteTaskEvaluationError as exc:
                errors.extend(exc.codes)
            referenced_paths.append(relative)
    if len(set(frame_ids)) != len(frame_ids):
        errors.append("observation_frame_ids_duplicate")
    if len(timestamps) == len(frames) and any(
        later <= earlier for earlier, later in pairwise(timestamps)
    ):
        errors.append("observation_timestamps_not_strictly_monotonic")
    coordinate_frame = _mapping(observations.get("coordinate_frame"))
    if (
        coordinate_frame.get("site_frame") != "site"
        or coordinate_frame.get("camera_frame") != "camera"
        or coordinate_frame.get("units") != "m"
        or coordinate_frame.get("handedness") != "right"
        or coordinate_frame.get("up_axis") not in {"Y", "Z"}
    ):
        errors.append("observation_coordinate_frame_invalid")
    volume = _mapping(observations.get("observed_volume"))
    minimum = volume.get("minimum_site_m")
    maximum = volume.get("maximum_site_m")
    supporting = sorted(
        {_text(item) for item in volume.get("supporting_frame_ids", []) if _text(item)}
    )
    if (
        not _finite_vector(minimum, 3)
        or not _finite_vector(maximum, 3)
        or any(float(minimum[i]) >= float(maximum[i]) for i in range(3))
        or not supporting
        or not set(supporting).issubset(frame_ids)
    ):
        errors.append("observation_volume_invalid")
    if observations.get("metric_scale_observed") is not True:
        errors.append("observation_metric_scale_not_observed")
    if errors:
        raise NewSiteTaskEvaluationError(*errors)
    observations["frames"] = frames
    observations["observation_manifest_digest"] = canonical_digest(
        observations, digest_field="observation_manifest_digest"
    )
    return {
        "observation_manifest": observations,
        "frame_count": len(frames),
        "first_timestamp_ns": timestamps[0],
        "last_timestamp_ns": timestamps[-1],
        "timestamps_strictly_monotonic": True,
        "coordinate_frames_verified": True,
        "observed_volume": volume,
        "referenced_paths": sorted(set(referenced_paths)),
    }


def _validate_task_spec(
    value: Mapping[str, Any], *, observation_validation: Mapping[str, Any]
) -> dict[str, Any]:
    task_spec = copy.deepcopy(dict(value))
    errors: list[str] = []
    if task_spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION:
        errors.append("task_spec_schema_version_invalid")
    for field in ("intake_id", "site_id", "site_class", "task_id", "decision_question", "robot_id"):
        if not _text(task_spec.get(field)):
            errors.append(f"task_spec_field_missing:{field}")
    sensor = _mapping(task_spec.get("sensor_binding"))
    if not _text(sensor.get("sensor_id")):
        errors.append("task_spec_sensor_id_missing")
    target = _mapping(task_spec.get("target_region"))
    supporting_frames = sorted(
        {_text(item) for item in target.get("supporting_frames", []) if _text(item)}
    )
    observed_frames = {
        _text(row.get("frame_id"))
        for row in _rows(_mapping(observation_validation.get("observation_manifest")).get("frames"))
    }
    coverage = target.get("captured_coverage")
    if not _text(target.get("region_id")):
        errors.append("task_target_region_id_missing")
    if not _finite_vector(target.get("position_site_m"), 3):
        errors.append("task_target_position_invalid")
    if not supporting_frames or not set(supporting_frames).issubset(observed_frames):
        errors.append("task_target_supporting_frames_not_observed")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or not 0 <= float(coverage) <= 1
    ):
        errors.append("task_target_coverage_invalid")
    if errors:
        raise NewSiteTaskEvaluationError(*errors)
    target["supporting_frames"] = supporting_frames
    target["observation_manifest_digest"] = _mapping(
        observation_validation.get("observation_manifest")
    )["observation_manifest_digest"]
    task_spec["target_region"] = target
    task_spec["task_spec_digest"] = canonical_digest(task_spec, digest_field="task_spec_digest")
    return task_spec


def _validate_site_artifacts(
    value: Mapping[str, Any], *, source_capture_digest: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    container = copy.deepcopy(dict(value))
    if container.get("schema_version") != SITE_ARTIFACTS_SCHEMA_VERSION:
        raise NewSiteTaskEvaluationError("site_artifacts_schema_version_invalid")
    if container.get("source_capture_digest") != source_capture_digest:
        raise NewSiteTaskEvaluationError("site_artifacts_source_capture_mismatch")
    artifacts = _mapping(container.get("artifacts"))
    if not artifacts:
        raise NewSiteTaskEvaluationError("site_artifacts_missing")
    verified: dict[str, Any] = {}
    for kind, raw in sorted(artifacts.items()):
        artifact = _mapping(raw)
        if artifact.get("source_capture_digest") != source_capture_digest:
            raise NewSiteTaskEvaluationError(f"site_artifact_source_mismatch:{kind}")
        if artifact.get("observed_or_independently_measured") is not True:
            raise NewSiteTaskEvaluationError(f"site_artifact_not_observed:{kind}")
        _digest(artifact, "artifact_digest")
        verified[_text(kind)] = artifact
    container["artifacts"] = verified
    container["site_artifacts_digest"] = canonical_digest(
        container, digest_field="site_artifacts_digest"
    )
    collision_scene = _mapping(container.get("collision_scene"))
    if collision_scene:
        if collision_scene.get("source_capture_digest") != source_capture_digest:
            raise NewSiteTaskEvaluationError("collision_scene_source_capture_mismatch")
        _digest(collision_scene, "collision_scene_digest")
    return container, collision_scene


def _load_site_artifact_fixture(fixture: Path, name: str) -> dict[str, Any]:
    value = _load_json(_safe_file(fixture, name))
    if value.get("schema_version") != SITE_ARTIFACTS_OVERLAY_SCHEMA_VERSION:
        return value
    base_name = _text(value.get("base_artifacts_file"))
    omitted = sorted({_text(item) for item in value.get("omit_artifacts", []) if _text(item)})
    if not base_name or not omitted:
        raise NewSiteTaskEvaluationError("site_artifacts_overlay_invalid")
    base = _load_json(_safe_file(fixture, base_name))
    if base.get("schema_version") != SITE_ARTIFACTS_SCHEMA_VERSION:
        raise NewSiteTaskEvaluationError("site_artifacts_overlay_base_invalid")
    if value.get("source_capture_digest") != base.get("source_capture_digest"):
        raise NewSiteTaskEvaluationError("site_artifacts_overlay_source_mismatch")
    artifacts = _mapping(base.get("artifacts"))
    if any(item not in artifacts for item in omitted):
        raise NewSiteTaskEvaluationError("site_artifacts_overlay_omission_unknown")
    for item in omitted:
        artifacts.pop(item)
    base["artifacts"] = artifacts
    base["intentional_fixture_omissions"] = omitted
    base["omission_reason"] = _text(value.get("omission_reason"))
    return base


def _method_capability_profile(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    method_id = _text(_mapping(task_spec.get("development_method")).get("method_id"))
    values: dict[str, Any] = {field: False for field in ALL_CAPABILITY_FIELDS}
    list_fields = {
        "plugin_versions",
        "robot_model_formats",
        "supported_embodiments",
        "supported_end_effectors",
        "action_representation_types",
        "qualification_record_ids",
        "qualified_task_classes",
        "qualified_material_regimes",
        "qualified_robot_ids",
        "qualified_end_effector_ids",
        "qualified_controller_ids",
        "qualified_sensor_ids",
        "qualified_site_classes",
        "qualified_metric_ids",
        "known_failure_modes",
        "prohibited_extrapolations",
        "asset_license_ids",
        "model_license_ids",
        "subprocessor_regions",
        "output_formats",
    }
    for field in list_fields:
        values[field] = []
    for capability in _mapping(task_spec.get("development_method")).get(
        "declared_measurement_capabilities", []
    ):
        if capability not in ALL_CAPABILITY_FIELDS or not str(capability).endswith("_supported"):
            raise NewSiteTaskEvaluationError(f"development_capability_invalid:{capability}")
        values[str(capability)] = True
    sensor_id = _text(_mapping(task_spec.get("sensor_binding")).get("sensor_id"))
    values.update(
        {
            "method_id": method_id,
            "method_family": "captured_real_observation",
            "version": "1",
            "release_date": "2026-08-02",
            "commit_hash": "new-site-development-observation-v1",
            "container_digest": canonical_digest({"runtime": "local-read-only"}),
            "solver_backend": "direct_retained_observation_review",
            "numeric_precision": "not_applicable",
            "deterministic_mode": "strict",
            "operating_system": "local",
            "gpu_model": "none",
            "driver_version": "none",
            "random_seed_policy": "not_applicable",
            "contact_formulation": "none",
            "maximum_control_rate_hz": 0,
            "qualified_parameter_ranges": {},
            "qualified_claim_ceiling": "C2",
            "qualification_expiration": "not_catalog_qualified",
            "harmful_false_negative_bound": 1.0,
            "maximum_latency_class": "interactive",
            "maximum_compute_class": "cpu",
            "estimated_cost_class": "zero_spend",
            "data_retention_days": 0,
            "source_available": True,
            "local_offline_supported": True,
            "api_only": False,
            "commercial_use_allowed": True,
            "redistribution_allowed": False,
            "provider_training_use_allowed": False,
            "deletion_right_supported": True,
            "output_export_supported": True,
            "supported_embodiments": [_text(task_spec.get("robot_id"))],
            "qualified_sensor_ids": [sensor_id],
            "output_formats": ["normalized_evidence_result.v1"],
            "prohibited_extrapolations": [
                "physical_task_success",
                "deployment_readiness",
                "safety_certification",
                "comparative_policy_ranking",
            ],
        }
    )
    return validate_method_capability_profile(
        {
            "schema_version": "method_capability_profile.v1",
            "method_id": method_id,
            "capabilities": values,
            "evidence_quality": {
                "source": "new_site_retained_capture_development_lane",
                "production_qualification_present": False,
                "r7_catalog_entry_created": False,
            },
            "expected_cost_usd": 0.0,
            "expected_latency_seconds": 0.01,
        }
    )


def _method_and_qualification(
    *,
    task_spec: Mapping[str, Any],
    testbed: Mapping[str, Any],
    source_capture_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method_config = _mapping(task_spec.get("development_method"))
    method_id = _text(method_config.get("method_id"))
    implementation_digest = canonical_digest(
        {"adapter_reference": CAPTURED_VISIBILITY_ADAPTER, "version": "1"}
    )
    measurement_profile = _method_capability_profile(task_spec)
    profile = EvidenceMethodProfile.from_mapping(
        {
            "schema_version": "evidence_method_profile.v1",
            "method_id": method_id,
            "version": "1",
            "implementation_digest": implementation_digest,
            "adapter_reference": CAPTURED_VISIBILITY_ADAPTER,
            "method_family": "captured_real_observation",
            "supported_claim_types": ["perception_visibility"],
            "required_inputs": ["captured_rgb_frames"],
            "applicability_envelope": {
                "testbed_ids": [testbed["testbed_id"]],
                "testbed_versions": [testbed["version"]],
                "task_families": [testbed["task_distribution"]["task_family"]],
            },
            "calibration_evidence_references": [source_capture_digest],
            "authority_tier": 1,
            "proof_tier": "development_captured_observation_only",
            "correlation_group": "new-site-retained-capture",
            "shared_dependencies": [source_capture_digest],
            "expected_cost_usd": 0.0,
            "expected_latency_seconds": 0.01,
            "reproducibility_level": "hermetic_local_read_only",
            "constraints": {
                "external_processing": False,
                "development_evidence_only": True,
            },
            "provider_availability": {"status": "available", "provider_spend": False},
            "failure_modes": ["supporting_frames_missing", "coverage_unmeasured"],
            "abstention_modes": ["site_evidence_incomplete", "production_qualification_missing"],
            "disqualifying_conditions": ["r7_required_for_production_route"],
            "self_qualified": False,
            "measurement_capability_profile": measurement_profile,
        }
    ).to_mapping()
    bindings = _mapping(testbed.get("robot_sensor_controller_bindings"))
    qualification = QualificationRecord.from_mapping(
        {
            "schema_version": "evidence_method_qualification.v1",
            "qualification_id": f"{method_id}-development-fixture-review",
            "method_id": method_id,
            "method_version": "1",
            "method_profile_digest": profile["method_profile_digest"],
            "implementation_digest": implementation_digest,
            "claim_type": "perception_visibility",
            "task_family": testbed["task_distribution"]["task_family"],
            "site_domain_conditions": testbed["supported_condition_ranges"],
            "embodiment": _mapping(bindings.get("embodiment")),
            "sensors": _mapping(bindings.get("sensors")),
            "controller_action_representation": _mapping(
                bindings.get("controller_action_representation")
            ),
            "evaluator": {
                "evaluator_id": "new-site-retained-frame-contract-review",
                "version": "1",
            },
            "evaluator_digest": canonical_digest(
                {"evaluator": "new-site-retained-frame-contract-review", "version": "1"}
            ),
            "predictions": [{"prediction_id": "target-visible", "value": True}],
            "accepted_real_outcomes": [
                {
                    "outcome_id": "fixture-observation-contract",
                    "value": True,
                    "physical_robot_outcome": False,
                }
            ],
            "calibration_partition": "heldout",
            "confidence_intervals": {"level": 0.9, "lower": 0.5, "upper": 1.0},
            "coverage": 0.9,
            "abstention_rate": 0.1,
            "false_safe_rate": 0.05,
            "false_reject_rate": 0.1,
            "provenance": {
                "source": "new_site_acceptance_fixture_contract",
                "development_only": True,
                "physical_outcome": False,
                "measurement_r7_qualification": False,
            },
            "owner_evidence": [{"uri": "capture://raw", "digest": source_capture_digest}],
            "status": "qualified",
            "self_grading": False,
            "subject_provider_id": "blueprint-local-captured-observation",
            "evaluator_provider_id": "independent-fixture-contract-review",
        }
    ).to_mapping()
    return profile, qualification


def _build_testbed(
    *,
    task_spec: Mapping[str, Any],
    source_capture_digest: str,
    site_profile: Mapping[str, Any],
    collision_scene: Mapping[str, Any],
    observation_validation: Mapping[str, Any],
    governance: Mapping[str, Any],
) -> dict[str, Any]:
    target = _mapping(task_spec.get("target_region"))
    sensor = _mapping(task_spec.get("sensor_binding"))
    robot_id = _text(task_spec.get("robot_id"))
    evidence_inventory = [
        {"evidence_id": evidence_id, "record_id": row.get("record_id")}
        for evidence_id, row in sorted(_mapping(site_profile.get("evidence")).items())
        if isinstance(row, Mapping) and row.get("available") is True
    ]
    if any(
        row.get("validated") is True
        for key, row in _mapping(site_profile.get("evidence")).items()
        if key == "calibrated_rgb" and isinstance(row, Mapping)
    ):
        evidence_inventory.append({"evidence_id": "captured_rgb_frames"})
    validation_envelope: dict[str, Any] = {
        "site_id": _text(task_spec.get("site_id")),
        "site_class": _text(task_spec.get("site_class")),
        "observation_manifest_digest": _mapping(
            observation_validation.get("observation_manifest")
        ).get("observation_manifest_digest"),
        "observed_volume": observation_validation.get("observed_volume"),
    }
    if collision_scene:
        validation_envelope["reconstruction_layers"] = {
            "physics_layer": [
                {
                    "output": "collision_geometry",
                    "result_id": "new-site-observed-collider",
                    "result_digest": canonical_digest(
                        {"collision_scene_digest": collision_scene["collision_scene_digest"]}
                    ),
                    "asset_references": {"collision_scene": dict(collision_scene)},
                    "generated_regions": [],
                    "claim_ceiling": {"collision_geometry": True},
                }
            ]
        }
        evidence_inventory.append({"evidence_id": "collision_scene"})
    refs = {
        name: {
            "uri": f"new-site://{name}",
            "digest": canonical_digest({"name": name, "source": source_capture_digest}),
        }
        for name in ("site_card", "evaluator", "reset")
    }
    refs.update(
        {
            "task_cards": [{"uri": "new-site://task", "digest": task_spec["task_spec_digest"]}],
            "scenario_cards": [{"uri": "new-site://scenario", "digest": canonical_digest(target)}],
            "eval_cards": [
                {
                    "uri": "new-site://eval",
                    "digest": canonical_digest({"claim": task_spec.get("claim")}),
                }
            ],
        }
    )
    return MaintainedSiteTaskTestbed.from_mapping(
        {
            "schema_version": "maintained_site_task_testbed.v1",
            "testbed_id": f"{_text(task_spec.get('site_id'))}-{_text(task_spec.get('task_id'))}",
            "version": "1",
            "predecessor_testbed_digest": None,
            "supersedes": [],
            "source_capture_bundles": [
                {
                    "bundle_id": _text(task_spec.get("intake_id")),
                    "version": "1",
                    "digest": source_capture_digest,
                }
            ],
            "artifact_references": refs,
            "task_distribution": {
                "task_family": "visual_inspection",
                "measurement_task_class": "visual_perception",
                "tasks": [_text(task_spec.get("task_id"))],
            },
            "supported_condition_ranges": _mapping(task_spec.get("supported_condition_ranges")),
            "robot_sensor_controller_bindings": {
                "embodiment": {"robot_id": robot_id},
                "sensors": {"camera": _text(sensor.get("sensor_id"))},
                "controller_action_representation": {
                    "type": "observation_only",
                    "controller_id": "none",
                },
                "selected_robot_placement": {
                    "candidate_id": "observed-site-registration",
                    "base_position_site_m": list(
                        task_spec.get("robot_base_position_site_m") or [0, 0, 0]
                    ),
                    "captured_coverage": float(target.get("captured_coverage") or 0),
                    "calibration_uncertainty_m": float(
                        task_spec.get("registration_uncertainty_m") or 0
                    ),
                    "method_qualification_status": "analytic_only",
                },
            },
            "governance": dict(governance),
            "evidence_inventory": sorted(evidence_inventory, key=lambda row: row["evidence_id"]),
            "validation_envelope": validation_envelope,
            "target_regions": [target],
            "known_unsupported_conditions": [
                "physical_task_success",
                "deployment_readiness",
                "safety_certification",
                "comparative_policy_ranking",
            ],
            "invalidation_triggers": [
                "layout_change",
                "sensor_calibration_change",
                "capture_revocation",
            ],
            "physical_outcome_history_refs": [],
            "lifecycle_state": "active",
            "site_evidence_profile": dict(site_profile),
        }
    ).to_mapping()


def _build_request(task_spec: Mapping[str, Any], testbed: Mapping[str, Any]) -> dict[str, Any]:
    claim = copy.deepcopy(_mapping(task_spec.get("claim")))
    claim.setdefault("measurement_task_class", "visual_perception")
    claim.setdefault("material_regimes", ["none"])
    claim.setdefault(
        "sensor_scope",
        {
            "required_modalities": ["rgb"],
            "sensor_ids": [_text(_mapping(task_spec.get("sensor_binding")).get("sensor_id"))],
        },
    )
    return DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": f"{_text(task_spec.get('task_id'))}-request",
            "decision_id": f"{_text(task_spec.get('task_id'))}-decision",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": _text(task_spec.get("decision_question")),
            "candidates": [{"robot_id": _text(task_spec.get("robot_id"))}],
            "claims": [claim],
            "budget": {"max_cost_usd": 0.0, "max_latency_seconds": 5.0},
            "deadline": _text(task_spec.get("deadline")),
            "available_physical_evidence": [],
            "permitted_evidence_methods": ["captured_real_observation"],
            "restrictions": {
                "external_processing_allowed": False,
                "local_only": True,
                "max_data_retention_days": 0,
                "provider_training_use_allowed": False,
                "output_portability_required": True,
                "commercial_use_required": True,
                "live_robot_execution_allowed": False,
            },
            "requested_result_audience": "internal_new_site_acceptance",
            "provenance": {"caller_identity": "pipeline:new-site-operator"},
            "idempotency_key": f"{_text(task_spec.get('task_id'))}-request",
        }
    ).to_mapping()


def _development_route(
    *,
    production_plan: Mapping[str, Any],
    request: Mapping[str, Any],
    testbed: Mapping[str, Any],
    requirements: Mapping[str, Any],
    method: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    # Re-validate the exact artifacts at the authority boundary. Callers cannot
    # swap an adapter, method identity, or development review after the
    # production kernel has emitted its digest-bound decision.
    validated_method = EvidenceMethodProfile.from_mapping(method).to_mapping()
    validated_qualification = QualificationRecord.from_mapping(qualification).to_mapping()
    capability_profile = validate_method_capability_profile(
        _mapping(validated_method.get("measurement_capability_profile"))
    )
    claim_plan = _rows(production_plan.get("claim_plans"))[0]
    measurement = _mapping(claim_plan.get("measurement_routing_decision"))
    candidates = [
        row
        for row in _rows(measurement.get("candidates_considered"))
        if row.get("method_id") == method["method_id"]
    ]
    rejection_codes = (
        sorted(set(candidates[0].get("rejection_codes") or [])) if len(candidates) == 1 else []
    )
    candidate_digest_matches = (
        len(candidates) == 1
        and candidates[0].get("capability_profile_digest")
        == capability_profile["capability_profile_digest"]
    )
    capabilities = _mapping(capability_profile.get("capabilities"))
    missing_capabilities = sorted(
        capability
        for capability in requirements.get("required_capabilities", [])
        if capabilities.get(capability) is not True
    )
    missing_alternatives = sorted(
        "|".join(str(capability) for capability in group)
        for group in requirements.get("required_capability_alternatives", [])
        if isinstance(group, list)
        and not any(capabilities.get(str(capability)) is True for capability in group)
    )
    permitted = (
        measurement.get("status") == "abstention"
        and rejection_codes == ["no_exact_verified_qualification"]
        and candidate_digest_matches
        and not missing_capabilities
        and not missing_alternatives
        and validated_qualification["method_id"] == validated_method["method_id"]
        and validated_qualification["method_profile_digest"]
        == validated_method["method_profile_digest"]
        and validated_method.get("expected_cost_usd") == 0.0
        and validated_method.get("adapter_reference") == CAPTURED_VISIBILITY_ADAPTER
        and _mapping(validated_method.get("constraints")).get("development_evidence_only") is True
    )
    result = {
        "schema_version": DEVELOPMENT_ROUTE_SCHEMA_VERSION,
        "status": "development_route_selected" if permitted else "development_route_abstained",
        "request_digest": request["request_digest"],
        "testbed_digest": testbed["testbed_digest"],
        "production_plan_digest": production_plan["plan_digest"],
        "production_measurement_routing_decision_digest": measurement.get(
            "routing_decision_digest"
        ),
        "method_id": method["method_id"] if permitted else None,
        "method_profile_digest": method["method_profile_digest"] if permitted else None,
        "generic_development_qualification_digest": (
            qualification["qualification_digest"] if permitted else None
        ),
        "production_qualification_present": False,
        "r7_catalog_entry_created": False,
        "production_routing_decision_mutated": False,
        "development_evidence_only": True,
        "paid_compute_authorized": False,
        "provider_execution_authorized": False,
        "physical_robot_run_authorized": False,
        "agent_selected_route": False,
        "candidate_capability_digest_matches": candidate_digest_matches,
        "missing_required_capabilities": missing_capabilities,
        "missing_required_capability_alternatives": missing_alternatives,
        "candidate_rejection_codes": rejection_codes,
        "blockers": []
        if permitted
        else sorted(
            {
                *(
                    _text(item)
                    for item in _mapping(measurement.get("abstention")).get("blockers", [])
                    if _text(item)
                ),
                *(f"required_capability_missing:{item}" for item in missing_capabilities),
                *(
                    f"required_capability_alternative_missing:{item}"
                    for item in missing_alternatives
                ),
                *rejection_codes,
            }
        ),
    }
    result["development_route_digest"] = canonical_digest(
        result, digest_field="development_route_digest"
    )
    return result


def _development_plan(
    *,
    request: Mapping[str, Any],
    testbed: Mapping[str, Any],
    method: Mapping[str, Any],
    qualification: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    claim = _rows(request.get("claims"))[0]
    step_id = f"{claim['claim_id']}-{method['method_id']}-development"
    return EvidencePlan.from_mapping(
        {
            "schema_version": "evidence_plan.v1",
            "plan_id": f"{request['request_id']}-development-plan",
            "request_id": request["request_id"],
            "decision_id": request["decision_id"],
            "request_digest": request["request_digest"],
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "claim_plans": [
                {
                    "claim_id": claim["claim_id"],
                    "claim_type": claim["claim_type"],
                    "required_authority_tier": 1,
                    "candidate_methods_considered": [],
                    "selected_methods": [
                        {
                            "step_id": step_id,
                            "claim_id": claim["claim_id"],
                            "method_id": method["method_id"],
                            "method_profile_digest": method["method_profile_digest"],
                            "qualification_digest": qualification["qualification_digest"],
                            "execution_rank": 0,
                            "stop_when_sufficient": True,
                            "escalate_on": [],
                        }
                    ],
                    "escalation_methods": [],
                    "status": "planned",
                    "selection_rationale": "explicit_development_lane_after_production_qualification_abstention",
                    "next_cheapest_experiment": "qualification_benchmark",
                    "measurement_routing_decision": None,
                    "expected_cost_usd": 0.0,
                    "expected_latency_seconds": method["expected_latency_seconds"],
                }
            ],
            "execution_order": [step_id],
            "stop_conditions": ["development_observation_collected"],
            "escalation_conditions": [],
            "physical_evidence_requests": [],
            "compiled_evaluation_run_specs": [],
            "non_evaluation_run_steps": [
                {
                    "step_id": step_id,
                    "claim_id": claim["claim_id"],
                    "method_id": method["method_id"],
                    "method_profile_digest": method["method_profile_digest"],
                    "qualification_digest": qualification["qualification_digest"],
                    "adapter_reference": method["adapter_reference"],
                    "method_family": method["method_family"],
                }
            ],
            "prohibited_claims": [
                "physical_task_success",
                "deployment_readiness",
                "safety_certification",
                "policy_ranking_thesis_upgrade",
                "production_measurement_qualification",
            ],
            "shared_dependency_warnings": [],
            "budget_status": {"max_cost_usd": 0.0, "projected_cost_usd": 0.0},
            "development_route_digest": route["development_route_digest"],
            "development_evidence_only": True,
        }
    ).to_mapping()


def _supervisor_proposals(
    *,
    run_id: str,
    request: Mapping[str, Any],
    testbed: Mapping[str, Any],
    method: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    context = SupervisorContext(
        run_id=run_id,
        customer_question=_text(request.get("decision_question")),
        decision_request=request,
        testbed=testbed,
        method_profiles=[method],
        qualifications=[qualification],
    )
    results = [
        DeterministicClaimTaskInterpreter().propose(context).to_mapping(),
        DeterministicCaptureTestbedSupervisor().propose(context).to_mapping(),
        DeterministicEvaluationMethodRouter().propose(context).to_mapping(),
    ]
    if any(
        row.get("authoritative") is not False
        or row.get("proof_effect") != "none"
        or row.get("proof_booleans_mutable") is not False
        or any(
            proposal.get("requested_proof_effect") != "none"
            or proposal.get("disposition") != "shadow_only"
            for proposal in _rows(row.get("proposals"))
        )
        for row in results
    ):
        raise NewSiteTaskEvaluationError("supervisor_proposal_authority_violation")
    output = {
        "schema_version": "new_site_supervisor_proposals.v1",
        "mode": "hermetic_supervisor_fixture_oracle",
        "results": results,
        "agent_may_lower_requirements": False,
        "agent_may_forge_qualification": False,
        "agent_may_authorize_spend": False,
        "agent_may_substitute_method": False,
    }
    output["supervisor_proposals_digest"] = canonical_digest(
        output, digest_field="supervisor_proposals_digest"
    )
    return output


def _redacted_projection(
    *,
    run_id: str,
    intake_id: str,
    plan: Mapping[str, Any],
    envelope: Mapping[str, Any],
    development_route: Mapping[str, Any],
) -> dict[str, Any]:
    per_claim = _rows(envelope.get("per_claim_verdicts"))
    projection = {
        "schema_version": "new_site_task_evaluation_webapp_projection.v1",
        "run_id": run_id,
        "intake_id": intake_id,
        "state": {
            "decision": "decided",
            "partial_decision": "partially_decided",
            "abstention": "abstained",
        }[envelope["overall_outcome"]],
        "testbed_digest": plan["testbed_digest"],
        "request_digest": plan["request_digest"],
        "plan_digest": plan["plan_digest"],
        "decision_envelope_digest": envelope["decision_envelope_digest"],
        "claim_results": [
            {
                "claim_id": row.get("claim_id"),
                "verdict": row.get("verdict"),
                "claim_ceiling": row.get("claim_ceiling"),
                "next_cheapest_experiment": envelope.get("next_cheapest_experiment"),
            }
            for row in per_claim
        ],
        "development_route_status": development_route["status"],
        "proof_boundary": {
            "development_evidence_only": True,
            "production_measurement_qualification": False,
            "simulation_or_capture_review_is_physical_success": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
        "raw_paths_included": False,
        "raw_frames_included": False,
        "credentials_included": False,
    }
    projection["projection_digest"] = canonical_digest(projection, digest_field="projection_digest")
    return projection


def run_new_site_task_evaluation(
    *,
    fixture_root: str | Path,
    state_root: str | Path,
    site_artifacts_name: str = "site_evidence_complete.json",
    emit_webapp_projection: bool = True,
) -> dict[str, Any]:
    """Run one immutable new-site acceptance fixture through both route lanes."""

    fixture = Path(fixture_root).expanduser().resolve()
    state = Path(state_root).expanduser().resolve()
    envelope_value = _load_json(fixture / "capture_intake_envelope.json")
    task_spec = _load_json(fixture / "task_spec.json")
    if task_spec.get("intake_id") != envelope_value.get("intake_id"):
        raise NewSiteTaskEvaluationError("task_spec_intake_mismatch")
    materialized = materialize_capture_intake(
        envelope_value,
        upload_root=fixture / "raw",
        store_root=state / "capture_store",
    )
    if materialized.admission.get("status") != "accepted":
        raise NewSiteTaskEvaluationError(
            "capture_intake_not_accepted:" + _text(materialized.admission.get("status"))
        )
    source_capture_digest = materialized.envelope["envelope_digest"]
    declared_paths = {
        _text(row.get("relative_path"))
        for row in _rows(materialized.envelope.get("original_files"))
    }
    observations = _load_json(fixture / "raw" / "observations.json")
    observation_validation = _validate_observations(
        observations,
        upload_root=fixture / "raw",
        declared_paths=declared_paths,
    )
    task_spec = _validate_task_spec(task_spec, observation_validation=observation_validation)
    governance = _validated_governance(_mapping(materialized.envelope.get("governance")))
    site_artifacts, collision_scene = _validate_site_artifacts(
        _load_site_artifact_fixture(fixture, site_artifacts_name),
        source_capture_digest=source_capture_digest,
    )
    artifacts = _mapping(site_artifacts.get("artifacts"))
    capture_evidence_tier = _text(
        _mapping(artifacts.get("capture_raw_manifest")).get("evidence_tier")
    )
    if capture_evidence_tier not in {"fixture_only", "raw_site_capture"}:
        raise NewSiteTaskEvaluationError("capture_evidence_tier_invalid")
    external_processing_allowed = _mapping(governance.get("provider_constraints"))[
        "external_processing_allowed"
    ]
    compiled = compile_site_evidence_profile(
        profile_id=f"{_text(task_spec.get('site_id'))}-observed-evidence",
        bundle_id=_text(task_spec.get("intake_id")),
        bundle_hash=source_capture_digest,
        provenance_record_id=materialized.admission["admission_digest"],
        rights={
            "commercial_evaluation_allowed": governance.get("rights") == "accepted"
            and "evaluation" in governance["allowed_uses"]
        },
        privacy={"external_processing_allowed": external_processing_allowed},
        metric_scale_verified=bool(
            observation_validation["observation_manifest"].get("metric_scale_observed")
        ),
        artifacts=artifacts,
    )
    site_profile = compiled["profile"]
    testbed = _build_testbed(
        task_spec=task_spec,
        source_capture_digest=source_capture_digest,
        site_profile=site_profile,
        collision_scene=collision_scene,
        observation_validation=observation_validation,
        governance=governance,
    )
    request = _build_request(task_spec, testbed)
    task_measurement_requirements = derive_task_measurement_requirements(
        _rows(request.get("claims"))[0], testbed
    )
    method, qualification = _method_and_qualification(
        task_spec=task_spec,
        testbed=testbed,
        source_capture_digest=source_capture_digest,
    )
    run_id = f"{_text(task_spec.get('site_id'))}-{_text(task_spec.get('task_id'))}-run"
    supervisor = _supervisor_proposals(
        run_id=run_id,
        request=request,
        testbed=testbed,
        method=method,
        qualification=qualification,
    )
    proposed_requirements = _rows(supervisor["results"])[0]["artifact"]["claims"][0][
        "proposed_task_measurement_requirements"
    ]
    if (
        proposed_requirements.get("requirements_digest")
        != task_measurement_requirements["requirements_digest"]
    ):
        raise NewSiteTaskEvaluationError("supervisor_requirements_do_not_match_kernel")
    production_plan = route_decision_evidence(
        request, testbed, [method], [qualification]
    ).to_mapping()
    measurement_decision = _mapping(
        _mapping(_rows(production_plan.get("claim_plans"))[0]).get("measurement_routing_decision")
    )
    site_audit = _mapping(
        _mapping(measurement_decision.get("abstention")).get("site_evidence_audit")
    )
    if not site_audit:
        site_audit = audit_site_evidence_profile(site_profile, task_measurement_requirements)
    development_route = _development_route(
        production_plan=production_plan,
        request=request,
        testbed=testbed,
        requirements=task_measurement_requirements,
        method=method,
        qualification=qualification,
    )
    execution_manifest: Mapping[str, Any] | None = None
    evidence_results: list[dict[str, Any]] = []
    if development_route["status"] == "development_route_selected":
        final_plan = _development_plan(
            request=request,
            testbed=testbed,
            method=method,
            qualification=qualification,
            route=development_route,
        )
        execution = execute_evidence_plan(
            final_plan,
            request,
            testbed,
            [method],
            [qualification],
            registry=authorized_local_evidence_adapter_registry([CAPTURED_VISIBILITY_ADAPTER]),
            context={"paid_compute_authorized": False, "provider_execution_authorized": False},
        )
        execution_manifest = dict(execution.execution_manifest)
        evidence_results = [row.to_mapping() for row in execution.results]
    else:
        final_plan = production_plan
    envelope = build_decision_envelope(request, testbed, final_plan, evidence_results).to_mapping()
    terminal_state = {
        "decision": "decided",
        "partial_decision": "partially_decided",
        "abstention": "abstained",
    }[envelope["overall_outcome"]]
    materialization_validation = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "intake_id": materialized.envelope["intake_id"],
        "envelope_digest": source_capture_digest,
        "admission_digest": materialized.admission["admission_digest"],
        "admission_status": materialized.admission["status"],
        "raw_objects": list(materialized.content_objects),
        "all_declared_raw_hashes_verified": True,
        "raw_inputs_content_addressed": True,
        "frame_count": observation_validation["frame_count"],
        "first_timestamp_ns": observation_validation["first_timestamp_ns"],
        "last_timestamp_ns": observation_validation["last_timestamp_ns"],
        "timestamps_strictly_monotonic": True,
        "coordinate_frames_verified": True,
        "metric_scale_observed": True,
        "observed_volume": observation_validation["observed_volume"],
        "rights_status": materialized.envelope["governance"]["rights"],
        "privacy_status": materialized.envelope["governance"]["privacy"],
        "provider_execution_authorized": False,
        "paid_compute_authorized": False,
        "raw_capture_truth_rewritten": False,
    }
    materialization_validation["materialization_validation_digest"] = canonical_digest(
        materialization_validation, digest_field="materialization_validation_digest"
    )
    projection = (
        _redacted_projection(
            run_id=run_id,
            intake_id=materialized.envelope["intake_id"],
            plan=final_plan,
            envelope=envelope,
            development_route=development_route,
        )
        if emit_webapp_projection
        else None
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "terminal_state": terminal_state,
        "capture_materialization_validation": materialization_validation,
        "task_specification": task_spec,
        "site_evidence_compilation_report": compiled["report"],
        "site_evidence_profile": site_profile,
        "site_evidence_audit": site_audit,
        "task_measurement_requirements": task_measurement_requirements,
        "supervisor_proposals": supervisor,
        "production_evidence_plan": production_plan,
        "development_route": development_route,
        "final_evidence_plan": final_plan,
        "execution_manifest": execution_manifest,
        "evidence_results": evidence_results,
        "decision_envelope": envelope,
        "webapp_projection": projection,
        "digest_joins": {
            "capture_to_site": source_capture_digest == site_profile["bundle_hash"],
            "task_to_testbed": task_spec["task_spec_digest"]
            == _rows(_mapping(testbed.get("artifact_references")).get("task_cards"))[0]["digest"],
            "observation_to_target": _mapping(_rows(testbed.get("target_regions"))[0]).get(
                "observation_manifest_digest"
            )
            == _mapping(observation_validation.get("observation_manifest")).get(
                "observation_manifest_digest"
            ),
            "site_to_testbed": site_profile["site_evidence_digest"]
            == testbed["site_evidence_profile"]["site_evidence_digest"],
            "testbed_to_request": testbed["testbed_digest"] == request["testbed_digest"],
            "request_to_plan": request["request_digest"] == final_plan["request_digest"],
            "plan_to_result": all(
                row["plan_digest"] == final_plan["plan_digest"] for row in evidence_results
            ),
            "plan_to_envelope": final_plan["plan_digest"] == envelope["plan_digest"],
            "result_to_envelope": all(
                row["result_digest"] in envelope["evidence_accepted"] for row in evidence_results
            ),
        },
        "proof_boundary": {
            "development_evidence_only": True,
            "capture_evidence_tier": capture_evidence_tier,
            "control_plane_fixture_only": capture_evidence_tier == "fixture_only",
            "production_measurement_route_selected": False,
            "r7_catalog_entry_created": False,
            "physical_task_success": False,
            "deployment_readiness": False,
            "safety_certification": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
            "paid_compute_used": False,
        },
    }
    if not all(result["digest_joins"].values()):
        raise NewSiteTaskEvaluationError("digest_join_failure")
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    output_dir = state / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result['result_digest'][7:]}.json"
    payload = canonical_json(result) + "\n"
    try:
        with output_path.open("x", encoding="utf-8", errors="strict", newline="") as handle:
            handle.write(payload)
    except FileExistsError:
        try:
            if output_path.read_text(encoding="utf-8") != payload:
                raise NewSiteTaskEvaluationError("immutable_result_collision")
        except OSError as exc:
            raise NewSiteTaskEvaluationError("result_readback_failed") from exc
    except OSError as exc:
        raise NewSiteTaskEvaluationError("result_write_failed") from exc
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument(
        "--site-artifacts",
        default="site_evidence_complete.json",
        help="Fixture-relative observed-evidence JSON (complete or intentionally incomplete).",
    )
    parser.add_argument("--no-webapp-projection", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_new_site_task_evaluation(
            fixture_root=args.fixture_root,
            state_root=args.state_root,
            site_artifacts_name=args.site_artifacts,
            emit_webapp_projection=not args.no_webapp_projection,
        )
    except (CaptureIntakeError, NewSiteTaskEvaluationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["terminal_state"],
                "run_id": result["run_id"],
                "result_digest": result["result_digest"],
                "development_route_status": result["development_route"]["status"],
                "production_route_status": _rows(result["production_evidence_plan"]["claim_plans"])[
                    0
                ]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NewSiteTaskEvaluationError",
    "run_new_site_task_evaluation",
]
