from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    EvidenceMethodProfile,
    QualificationRecord,
)
from blueprint_pipeline.local_evidence_adapters import ANALYTIC_REACHABILITY_ADAPTER
from blueprint_pipeline.task_evaluation_method_catalog import (
    TaskEvaluationMethodCatalogError,
    load_task_evaluation_method_catalog,
    validate_task_evaluation_method_catalog,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _catalog() -> dict:
    profile = EvidenceMethodProfile.from_mapping({
        "schema_version": "evidence_method_profile.v1",
        "method_id": "local-analytic-reachability",
        "version": "1",
        "implementation_digest": SHA_A,
        "adapter_reference": ANALYTIC_REACHABILITY_ADAPTER,
        "method_family": "analytic_geometry_kinematics",
        "supported_claim_types": ["reachability"],
        "required_inputs": ["metric_geometry"],
        "applicability_envelope": {"task_families": ["rigid_object_pick_place"]},
        "calibration_evidence_references": ["fixture://calibration"],
        "authority_tier": 1,
        "proof_tier": "analytic_only",
        "correlation_group": "metric-scaffold",
        "shared_dependencies": ["capture"],
        "expected_cost_usd": 0,
        "expected_latency_seconds": 0.01,
        "reproducibility_level": "hermetic_local",
        "constraints": {"external_processing": False},
        "provider_availability": {"status": "available"},
        "failure_modes": ["metric_position_missing"],
        "abstention_modes": ["uncertain_boundary"],
        "disqualifying_conditions": [],
        "self_qualified": False,
    }).to_mapping()
    qualification = QualificationRecord.from_mapping({
        "schema_version": "evidence_method_qualification.v1",
        "qualification_id": "qualification-local-reach-1",
        "method_id": profile["method_id"],
        "method_version": profile["version"],
        "method_profile_digest": profile["method_profile_digest"],
        "implementation_digest": profile["implementation_digest"],
        "claim_type": "reachability",
        "task_family": "rigid_object_pick_place",
        "site_domain_conditions": {"lighting_lux": [300, 600]},
        "embodiment": {"robot_id": "fixture-arm"},
        "sensors": {"camera": "rgb-v1"},
        "controller_action_representation": {"type": "joint_position"},
        "evaluator": {"evaluator_id": "independent-check", "version": "1"},
        "evaluator_digest": SHA_B,
        "predictions": [{"prediction_id": "prediction-1", "value": True}],
        "accepted_real_outcomes": [{"outcome_id": "calibration-1", "value": True}],
        "calibration_partition": "heldout",
        "confidence_intervals": {"level": 0.95},
        "coverage": 0.95,
        "abstention_rate": 0.05,
        "false_safe_rate": 0.01,
        "false_reject_rate": 0.02,
        "provenance": {"source": "independent-fixture"},
        "owner_evidence": [{"uri": "fixture://qualification", "digest": SHA_B}],
        "status": "qualified",
        "self_grading": False,
        "subject_provider_id": "blueprint-local-method",
        "evaluator_provider_id": "independent-check",
    }).to_mapping()
    return {
        "schema_version": "task_evaluation_method_catalog.v1",
        "catalog_id": "local-beta-methods",
        "version": "1",
        "method_profiles": [profile],
        "qualifications": [qualification],
    }


def test_catalog_is_deterministic_digest_bound_and_loadable(tmp_path) -> None:
    first = validate_task_evaluation_method_catalog(_catalog())
    second = validate_task_evaluation_method_catalog(_catalog())
    assert first == second
    assert first["proof_boundary"]["catalog_entry_is_execution_authorization"] is False
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    assert load_task_evaluation_method_catalog(path) == first


def test_catalog_rejects_stale_qualification_secrets_and_digest_tampering() -> None:
    stale = _catalog()
    stale["qualifications"][0]["implementation_digest"] = SHA_B
    stale["qualifications"][0].pop("qualification_digest")
    stale["qualifications"][0] = QualificationRecord.from_mapping(
        stale["qualifications"][0]
    ).to_mapping()
    with pytest.raises(TaskEvaluationMethodCatalogError, match="profile_mismatch"):
        validate_task_evaluation_method_catalog(stale)

    secret = _catalog()
    secret["provider_token"] = "must-not-store"
    with pytest.raises(TaskEvaluationMethodCatalogError, match="secret_value"):
        validate_task_evaluation_method_catalog(secret)

    tampered = validate_task_evaluation_method_catalog(_catalog())
    tampered["catalog_id"] = "changed"
    with pytest.raises(TaskEvaluationMethodCatalogError, match="digest_mismatch"):
        validate_task_evaluation_method_catalog(tampered)

    duplicate = copy.deepcopy(_catalog())
    duplicate["qualifications"].append(duplicate["qualifications"][0])
    with pytest.raises(TaskEvaluationMethodCatalogError, match="duplicate_qualification"):
        validate_task_evaluation_method_catalog(duplicate)
