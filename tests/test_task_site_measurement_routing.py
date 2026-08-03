from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    MeasurementRoutingError,
    audit_site_evidence_profile,
    derive_task_measurement_requirements,
    route_task_site_measurement,
    validate_measurement_qualification,
    validate_method_capability_profile,
    validate_site_evidence_profile,
    validate_task_measurement_requirements,
)


SHA_A = "sha256:" + "a" * 64


def _capabilities(method_id: str, *, enabled: set[str], ceiling: str = "C3") -> dict:
    values: dict = {field: False for field in ALL_CAPABILITY_FIELDS}
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
    values.update(
        {
            "method_id": method_id,
            "method_family": "traditional_simulation",
            "version": "1",
            "release_date": "2026-08-01",
            "commit_hash": "fixture-commit",
            "container_digest": SHA_A,
            "solver_backend": "fixture",
            "numeric_precision": "float64",
            "deterministic_mode": "strict",
            "operating_system": "linux",
            "gpu_model": "none",
            "driver_version": "none",
            "random_seed_policy": "frozen",
            "contact_formulation": "fixture",
            "maximum_control_rate_hz": 1000,
            "qualified_parameter_ranges": {},
            "qualified_claim_ceiling": ceiling,
            "qualification_expiration": "2027-08-01",
            "harmful_false_negative_bound": 0.01,
            "maximum_latency_class": "interactive",
            "maximum_compute_class": "cpu",
            "estimated_cost_class": "low",
            "data_retention_days": 0,
            "source_available": True,
            "local_offline_supported": True,
            "commercial_use_allowed": True,
            "provider_training_use_allowed": False,
            "deletion_right_supported": True,
            "output_export_supported": True,
        }
    )
    for field in enabled:
        values[field] = True
    return values


def _profile(
    method_id: str,
    enabled: set[str],
    *,
    accuracy: float,
    cost: float = 1.0,
) -> dict:
    return validate_method_capability_profile(
        {
            "schema_version": "method_capability_profile.v1",
            "method_id": method_id,
            "capabilities": _capabilities(method_id, enabled=enabled),
            "evidence_quality": {
                "source": "independent_fixture",
                "public_research_is_qualification": False,
            },
            "expected_cost_usd": cost,
            "expected_latency_seconds": 1.0,
        }
    )


def _qualification(profile: dict, capabilities: set[str], *, ceiling: str = "C3") -> dict:
    return validate_measurement_qualification(
        {
            "schema_version": "measurement_qualification_record.v1",
            "qualification_id": f"q-{profile['method_id']}",
            "method_id": profile["method_id"],
            "method_version": "1",
            "capability_profile_digest": profile["capability_profile_digest"],
            "admission_record_digest": SHA_A,
            "admission_stage": "R7",
            "status": "approved",
            "qualified_capabilities": sorted(capabilities),
            "claim_ceiling": ceiling,
            "scope": {
                "task_classes": ["rigid_pick_place", "collision_free_motion"],
                "material_regimes": ["none"],
                "robot_ids": [],
                "end_effector_ids": [],
                "controller_ids": [],
                "sensor_ids": [],
                "metric_ids": [],
                "parameter_ranges": {},
            },
            "metrics": {
                "physical_accuracy_error": 0.01,
                "uncertainty": 0.02,
                "scope_distance": 0.0,
                "harmful_false_negative_rate": 0.001,
                "reproducibility_score": 1.0,
                "privacy_preference": 1.0,
            },
            "approval": {
                "signature_status": "verified",
                "signature_id": f"fixture-signature-{profile['method_id']}",
                "approved_by": ["benchmark-owner", "independent-reviewer"],
                "agent_approved": False,
            },
            "expiration_date": "2027-08-01",
            "self_grading": False,
        }
    )


def _site(
    *,
    missing: set[str] | None = None,
    rights: bool = True,
    forbidden_claims: tuple[str, ...] = (),
) -> dict:
    missing = missing or set()
    evidence_ids = {
        "metric_scale",
        "robot_site_registration",
        "validated_collider",
        "mass_inertia",
        "friction_contact",
        "material_parameters",
        "coverage_uncertainty",
    }
    return validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": "site-fixture-v1",
            "bundle_id": "capture-fixture",
            "bundle_hash": SHA_A,
            "provenance_record_id": "provenance-fixture",
            "rights": {"commercial_evaluation_allowed": rights},
            "privacy": {"classification": "internal", "external_processing_allowed": False},
            "coordinate_system": {"metric_scale_verified": True, "frame": "site"},
            "evidence": {
                evidence_id: {
                    "available": evidence_id not in missing,
                    "validated": evidence_id not in missing,
                    "record_id": f"evidence-{evidence_id}",
                }
                for evidence_id in evidence_ids
            },
            "limitations": {
                "known_missing_regions": [],
                "forbidden_claims": list(forbidden_claims),
            },
        }
    )


def _requirements() -> dict:
    return derive_task_measurement_requirements(
        {
            "claim_id": "collision-claim",
            "claim_type": "collision_contact",
            "material_regimes": ["none"],
            "metric_ids": [],
        },
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )


def test_composite_route_covers_exact_site_task_measurements_lexicographically() -> None:
    requirements = _requirements()
    geometry_caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
    }
    contact_caps = {
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    geometry = _profile("exact-geometry", geometry_caps, accuracy=0.01, cost=0.1)
    contact = _profile("qualified-contact", contact_caps, accuracy=0.01, cost=2.0)
    decision = route_task_site_measurement(
        requirements,
        _site(),
        [contact, geometry],
        [_qualification(contact, contact_caps), _qualification(geometry, geometry_caps)],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )

    assert decision["status"] == "route_selected"
    assert decision["selected_route"]["type"] == "composite"
    assert {row["method_id"] for row in decision["selected_route"]["stages"]} == {
        "exact-geometry",
        "qualified-contact",
    }
    assert decision["claim_boundary"]["permitted_claim"] == "C3"
    assert decision["claim_boundary"]["physical_success_established"] is False
    assert decision["agent_selected_route"] is False
    assert decision["execution_authorized"] is False
    assert decision["deterministic_policy_signature"].startswith("sha256:")


def test_missing_site_evidence_abstains_with_smallest_targeted_action() -> None:
    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    profile = _profile("rigid-engine", caps, accuracy=0.01)
    decision = route_task_site_measurement(
        _requirements(),
        _site(missing={"friction_contact"}),
        [profile],
        [_qualification(profile, caps)],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )

    assert decision["status"] == "abstention"
    assert decision["selected_route"]["stages"] == []
    assert (
        decision["abstention"]["smallest_next_action"]["action_type"] == "material_identification"
    )
    assert any(
        "friction_contact" in code for code in decision["abstention"]["blocking_requirements"]
    )
    audit = decision["abstention"]["site_evidence_audit"]
    assert any(
        gap["evidence_id"] == "friction_contact"
        and gap["smallest_next_action"] == "material_identification"
        and gap["required_by_request"] is True
        for gap in audit["gaps"]
    )


def test_explicit_site_evidence_is_a_global_hard_gate() -> None:
    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    profile = _profile("rigid-engine", caps, accuracy=0.01)
    requirements = _requirements()
    requirements.pop("requirements_digest")
    requirements["required_site_evidence"].append("physical_outcomes")
    decision = route_task_site_measurement(
        requirements,
        _site(),
        [profile],
        [_qualification(profile, caps)],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )

    assert decision["status"] == "abstention"
    assert (
        "required_site_evidence_missing:physical_outcomes"
        in (decision["abstention"]["blocking_requirements"])
    )


def test_omitted_constraints_default_to_privacy_safe_policy() -> None:
    requirements = _requirements()
    assert requirements["constraints"]["commercial_use_required"] is True
    assert requirements["constraints"]["output_portability_required"] is True
    assert requirements["constraints"]["provider_training_use_allowed"] is False

    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    profile_value = _profile("training-provider", caps, accuracy=0.01)
    profile_value.pop("capability_profile_digest")
    profile_value["capabilities"]["provider_training_use_allowed"] = True
    profile = validate_method_capability_profile(profile_value)
    decision = route_task_site_measurement(
        requirements,
        _site(),
        [profile],
        [_qualification(profile, caps)],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )

    assert decision["status"] == "abstention"
    assert (
        "provider_training_use_not_prohibited"
        in (decision["candidates_considered"][0]["rejection_codes"])
    )


def test_unverified_agent_qualification_and_rights_fail_closed() -> None:
    caps = {"metric_scale_supported", "continuous_collision_supported"}
    profile = _profile("splat-is-not-collider", caps, accuracy=0.1)
    qualification = _qualification(profile, caps)
    unverified = copy.deepcopy(qualification)
    unverified.pop("measurement_qualification_digest")
    unverified["approval"]["signature_status"] = "agent_asserted"
    unverified["approval"]["agent_approved"] = True
    with pytest.raises(MeasurementRoutingError, match="agent_qualification_approval_forbidden"):
        validate_measurement_qualification(unverified)

    decision = route_task_site_measurement(
        derive_task_measurement_requirements(
            {"claim_id": "motion", "claim_type": "collision_contact", "material_regimes": ["none"]},
            {"task_distribution": {"measurement_task_class": "collision_free_motion"}},
        ),
        _site(rights=False),
        [profile],
        [qualification],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    assert decision["status"] == "abstention"
    assert any(
        "site_rights_not_cleared" in row["rejection_codes"]
        for row in decision["candidates_considered"]
    )


def test_capability_contract_requires_every_report_field() -> None:
    profile = _profile("complete-profile", {"metric_scale_supported"}, accuracy=0.1)
    broken = copy.deepcopy(profile)
    broken.pop("capability_profile_digest")
    broken["capabilities"].pop("gaussian_splat_supported_for_rendering")
    with pytest.raises(MeasurementRoutingError, match="capability_fields_missing"):
        validate_method_capability_profile(broken)


def test_checked_json_schema_accepts_each_core_contract() -> None:
    schema_path = (
        Path(__file__).parents[1] / "docs/schemas/task_site_measurement_routing.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    requirements = _requirements()
    site = _site()
    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    profile = _profile("schema-fixture", caps, accuracy=0.01)
    qualification = _qualification(profile, caps)
    decision = route_task_site_measurement(
        requirements,
        site,
        [profile],
        [qualification],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    for artifact in (requirements, site, profile, qualification, decision):
        jsonschema.validate(artifact, schema)


def test_empty_capability_claim_never_becomes_an_empty_success_route() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "capture-truth", "claim_type": "capture_provenance"},
        {"task_distribution": {"measurement_task_class": "static_reachability"}},
    )
    assert requirements["required_capabilities"] == []
    decision = route_task_site_measurement(
        requirements,
        _site(),
        [],
        [],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    assert decision["status"] == "abstention"
    assert (
        "uncovered_capability:qualified_claim_measurement_method"
        in (decision["abstention"]["blocking_requirements"])
    )


def test_agent_cannot_lower_requirements_below_deterministic_minimum() -> None:
    weakened = copy.deepcopy(_requirements())
    weakened.pop("requirements_digest")
    weakened["required_capabilities"] = ["metric_scale_supported"]
    with pytest.raises(MeasurementRoutingError, match="below_deterministic_minimum"):
        validate_task_measurement_requirements(weakened)


def test_generic_deformable_and_unknown_task_class_are_rejected() -> None:
    with pytest.raises(
        MeasurementRoutingError, match="material_regime_forbidden_generic:deformable"
    ):
        derive_task_measurement_requirements(
            {
                "claim_id": "fold",
                "claim_type": "collision_contact",
                "material_regimes": ["deformable"],
            },
            {"task_distribution": {"measurement_task_class": "garment_manipulation"}},
        )
    with pytest.raises(MeasurementRoutingError, match="task_class_unknown"):
        derive_task_measurement_requirements(
            {"claim_id": "novel", "claim_type": "collision_contact"},
            {"task_distribution": {"measurement_task_class": "unmapped_new_task"}},
        )


def test_world_model_family_never_gains_physics_authority() -> None:
    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    values = _capabilities("video-world-model", enabled=caps)
    values["method_family"] = "learned_world_model"
    with pytest.raises(MeasurementRoutingError, match="world_model_roles_missing"):
        validate_method_capability_profile(
            {
                "schema_version": "method_capability_profile.v1",
                "method_id": "video-world-model",
                "capabilities": values,
                "evidence_quality": {"source": "independent_fixture"},
                "expected_cost_usd": 1.0,
                "expected_latency_seconds": 1.0,
            }
        )
    with pytest.raises(MeasurementRoutingError, match="world_model_role_forbidden"):
        validate_method_capability_profile(
            {
                "schema_version": "method_capability_profile.v1",
                "method_id": "video-world-model",
                "capabilities": values,
                "world_model_roles": ["collision_authority"],
                "evidence_quality": {"source": "independent_fixture"},
                "expected_cost_usd": 1.0,
                "expected_latency_seconds": 1.0,
            }
        )
    ceiling_values = copy.deepcopy(values)
    ceiling_values["qualified_claim_ceiling"] = "C6"
    with pytest.raises(MeasurementRoutingError, match="method_claim_ceiling_exceeds_family_cap"):
        validate_method_capability_profile(
            {
                "schema_version": "method_capability_profile.v1",
                "method_id": "video-world-model",
                "capabilities": ceiling_values,
                "world_model_roles": ["qualitative_rollout"],
                "evidence_quality": {"source": "independent_fixture"},
                "expected_cost_usd": 1.0,
                "expected_latency_seconds": 1.0,
            }
        )
    profile = validate_method_capability_profile(
        {
            "schema_version": "method_capability_profile.v1",
            "method_id": "video-world-model",
            "capabilities": values,
            "world_model_roles": ["qualitative_rollout", "comparative_policy_ranking"],
            "evidence_quality": {"source": "independent_fixture"},
            "expected_cost_usd": 1.0,
            "expected_latency_seconds": 1.0,
        }
    )
    decision = route_task_site_measurement(
        _requirements(),
        _site(),
        [profile],
        [_qualification(profile, caps)],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    assert decision["status"] == "abstention"
    assert (
        "physics_authority_forbidden_for_method_family"
        in (decision["candidates_considered"][0]["rejection_codes"])
    )


def test_fluid_alternative_solver_group_is_satisfied_by_one_member() -> None:
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "pouring",
            "claim_type": "collision_contact",
            "material_regimes": ["fluid_viscous_free_surface"],
        },
        {"task_distribution": {"measurement_task_class": "fluid_manipulation"}},
    )
    assert requirements["required_capability_alternatives"] == [
        ["cfd_supported", "mpm_supported", "sph_supported"]
    ]
    caps = {
        "continuous_collision_supported",
        "fluid_surface_tension_supported",
        "fluid_wetting_supported",
        "metric_scale_supported",
        "sph_supported",
    }
    profile = _profile("sph-solver", caps, accuracy=0.01)
    qualification = validate_measurement_qualification(
        {
            "schema_version": "measurement_qualification_record.v1",
            "qualification_id": "q-sph-solver",
            "method_id": "sph-solver",
            "method_version": "1",
            "capability_profile_digest": profile["capability_profile_digest"],
            "admission_record_digest": SHA_A,
            "admission_stage": "R7",
            "status": "approved",
            "qualified_capabilities": sorted(caps),
            "claim_ceiling": "C3",
            "scope": {
                "task_classes": ["fluid_manipulation"],
                "material_regimes": ["fluid_viscous_free_surface"],
                "robot_ids": [],
                "end_effector_ids": [],
                "controller_ids": [],
                "sensor_ids": [],
                "metric_ids": [],
                "parameter_ranges": {},
            },
            "metrics": {
                "physical_accuracy_error": 0.01,
                "uncertainty": 0.02,
                "scope_distance": 0.0,
                "harmful_false_negative_rate": 0.001,
                "reproducibility_score": 1.0,
                "privacy_preference": 1.0,
            },
            "approval": {
                "signature_status": "verified",
                "signature_id": "fixture-signature-sph",
                "approved_by": ["benchmark-owner", "independent-reviewer"],
                "agent_approved": False,
            },
            "expiration_date": "2027-08-01",
            "self_grading": False,
        }
    )
    decision = route_task_site_measurement(
        requirements,
        _site(),
        [profile],
        [qualification],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    assert decision["status"] == "route_selected"
    assert "sph_supported" in decision["selected_route"]["stages"][0]["covered_capabilities"]


def test_site_forbidden_claims_hard_abstain() -> None:
    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    profile = _profile("rigid-engine", caps, accuracy=0.01)
    decision = route_task_site_measurement(
        _requirements(),
        _site(forbidden_claims=("collision_contact",)),
        [profile],
        [_qualification(profile, caps)],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    assert decision["status"] == "abstention"
    assert decision["abstention"]["abstention_code"] == "claim_forbidden_by_site_limitations"


def test_site_evidence_audit_reports_gaps_with_smallest_actions() -> None:
    audit = audit_site_evidence_profile(_site(missing={"friction_contact"}), _requirements())
    actions = {gap["evidence_id"]: gap["smallest_next_action"] for gap in audit["gaps"]}
    assert actions["friction_contact"] == "material_identification"
    assert actions["articulation_model"] == "articulation_measurement"
    assert actions["sensor_calibration"] == "sensor_calibration"
    assert actions["validated_mesh"] == "collider_validation"
    assert audit["metric_scale_verified"] is True
    assert audit["raw_capture_truth_rewritten"] is False


def test_version_mismatch_is_explicitly_rejected() -> None:
    caps = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    profile = _profile("versioned-engine", caps, accuracy=0.01)
    qualification = _qualification(profile, caps)
    qualification.pop("measurement_qualification_digest")
    qualification["method_version"] = "0"
    qualification = validate_measurement_qualification(qualification)
    decision = route_task_site_measurement(
        _requirements(),
        _site(),
        [profile],
        [qualification],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 1),
    )
    assert decision["status"] == "abstention"
    assert (
        "qualification_method_version_mismatch"
        in (decision["candidates_considered"][0]["rejection_codes"])
    )
