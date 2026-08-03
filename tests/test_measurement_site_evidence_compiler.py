"""Lane 1+2: broad artifact compiler and the any-task routing sweep.

The golden sweep table below was computed against the rich compiled profile
and each outcome hand-verified for doctrinal correctness: measured
articulation moves drawer gaps to force collection, task-level evidence gates
fire for HRI and long-horizon claims, characterized granular routes while
unspecimened food abstains, and appearance-only kitchen evidence can never
produce a collision route.

Layering: ``measurement_site_evidence_bridge`` (typed reconstruction geometry
contracts) is the strict geometry path; this compiler covers the remaining
artifact families and the testbed attachment.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
)
from blueprint_pipeline.decision_evidence_router import route_decision_evidence
from blueprint_pipeline.measurement_site_evidence_compiler import (
    SiteEvidenceCompilerError,
    attach_compiled_site_evidence,
    compile_site_evidence_profile,
)
from blueprint_pipeline.task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    MATERIAL_REGIME_CAPABILITIES,
    TASK_CAPABILITIES,
    derive_task_measurement_requirements,
    route_task_site_measurement,
    validate_measurement_qualification,
    validate_method_capability_profile,
)


ROOT = Path(__file__).parents[1]
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64

_RICH_ARTIFACTS = {
    "collider_qualification_report": {
        "report_id": "collider-q1", "status": "passed", "fixture_only": True,
    },
    "metric_scale_validation": {
        "validation_id": "scale-1", "status": "passed", "fixture_only": True,
    },
    "robot_site_registration": {
        "registration_id": "reg-1", "status": "verified", "fixture_only": True,
    },
    "metric_geometry_manifest": {
        "manifest_id": "geom-1", "status": "passed", "fixture_only": True,
    },
    "appearance_manifest": {
        "manifest_id": "splat-1", "heldout_status": "passed", "fixture_only": True,
    },
    "capture_qa_report": {
        "report_id": "qa-1", "status": "passed", "fixture_only": True,
    },
    "articulation_measurement": {
        "measurement_id": "art-1", "status": "passed", "actuation_measured": True,
        "fixture_only": True,
    },
    "material_identification": {
        "identification_id": "mat-1", "status": "passed",
        "friction_measured": True, "mass_inertia_measured": True,
        "fixture_only": True,
    },
    "sensor_calibration": {
        "calibration_id": "cal-1", "status": "passed", "timing_verified": True,
        "fixture_only": True,
    },
}

_REGIME_BY_CLASS = {
    "garment_manipulation": "garment_cloth",
    "cable_hose_routing": "rope_cable_hose",
    "granular_manipulation": "granular_media",
    "fluid_manipulation": "fluid_viscous_free_surface",
    "food_manipulation": "food_cuttable_multiphase",
}

# task class -> (status, smallest next action when abstaining). Hand-verified.
_GOLDEN_SWEEP = {
    "cable_hose_routing": ("abstention", "targeted_recapture"),
    "collision_free_motion": ("abstention", "targeted_recapture"),
    "contact_rich_dexterous_manipulation": ("route_selected", None),
    "doors_drawers_handles": ("abstention", "force_tactile_collection"),
    "fluid_manipulation": ("route_selected", None),
    "food_manipulation": ("abstention", "material_identification"),
    "garment_manipulation": ("abstention", "targeted_recapture"),
    "granular_manipulation": ("route_selected", None),
    "human_robot_interaction": ("abstention", "targeted_recapture"),
    "insertion_assembly": ("abstention", "force_tactile_collection"),
    "locomotion": ("abstention", "sensor_calibration"),
    "long_horizon_task_execution": ("abstention", "physical_execution"),
    "mobile_manipulation_clutter": ("abstention", "sensor_calibration"),
    "rigid_pick_place": ("route_selected", None),
    "small_thin_occluded_objects": ("abstention", "targeted_recapture"),
    "static_reachability": ("route_selected", None),
    "tactile_manipulation": ("abstention", "force_tactile_collection"),
    "transparent_reflective_objects": ("abstention", "targeted_recapture"),
    "valves_switches_buttons": ("abstention", "force_tactile_collection"),
    "visual_navigation_active_perception": ("abstention", "targeted_recapture"),
    "visual_perception": ("route_selected", None),
}


def _rich_site() -> dict:
    return compile_site_evidence_profile(
        profile_id="rich-compiled-site",
        bundle_id="capture-1",
        bundle_hash=SHA_A,
        provenance_record_id="provenance-fixture",
        rights={"commercial_evaluation_allowed": True},
        privacy={"external_processing_allowed": False},
        metric_scale_verified=True,
        artifacts=_RICH_ARTIFACTS,
    )["profile"]


def _fixture_engine() -> tuple[dict, dict]:
    values: dict = {field: False for field in ALL_CAPABILITY_FIELDS}
    for field in (
        "plugin_versions", "robot_model_formats", "supported_embodiments",
        "supported_end_effectors", "action_representation_types",
        "qualification_record_ids", "qualified_task_classes",
        "qualified_material_regimes", "qualified_robot_ids",
        "qualified_end_effector_ids", "qualified_controller_ids",
        "qualified_sensor_ids", "qualified_site_classes", "qualified_metric_ids",
        "known_failure_modes", "prohibited_extrapolations", "asset_license_ids",
        "model_license_ids", "subprocessor_regions", "output_formats",
    ):
        values[field] = []
    booleans = sorted(
        field for field in ALL_CAPABILITY_FIELDS if field.endswith("_supported")
    )
    for field in booleans:
        values[field] = True
    values.update({
        "method_id": "fixture-universal-development-engine",
        "method_family": "traditional_simulation", "version": "1",
        "release_date": "2026-08-01", "commit_hash": "fixture",
        "container_digest": SHA_A, "solver_backend": "fixture",
        "numeric_precision": "float64", "deterministic_mode": "strict",
        "operating_system": "linux", "gpu_model": "none", "driver_version": "none",
        "random_seed_policy": "frozen", "contact_formulation": "fixture",
        "maximum_control_rate_hz": 1000, "qualified_parameter_ranges": {},
        "qualified_claim_ceiling": "C4", "qualification_expiration": "2027-08-01",
        "harmful_false_negative_bound": 0.01, "maximum_latency_class": "interactive",
        "maximum_compute_class": "cpu", "estimated_cost_class": "low",
        "data_retention_days": 0, "source_available": True,
        "local_offline_supported": True, "api_only": False,
        "commercial_use_allowed": True, "redistribution_allowed": True,
        "provider_training_use_allowed": False, "deletion_right_supported": True,
        "output_export_supported": True,
    })
    profile = validate_method_capability_profile({
        "schema_version": "method_capability_profile.v1",
        "method_id": "fixture-universal-development-engine",
        "capabilities": values,
        "evidence_quality": {
            "source": "development_fixture",
            "fixture_only_never_a_real_engine": True,
        },
        "expected_cost_usd": 1.0,
        "expected_latency_seconds": 1.0,
    })
    qualification = validate_measurement_qualification({
        "schema_version": "measurement_qualification_record.v1",
        "qualification_id": "development-fixture-universal",
        "method_id": "fixture-universal-development-engine",
        "method_version": "1",
        "capability_profile_digest": profile["capability_profile_digest"],
        "admission_record_digest": SHA_A,
        "admission_stage": "R7",
        "status": "approved",
        "qualified_capabilities": booleans,
        "claim_ceiling": "C4",
        "scope": {
            "task_classes": sorted(TASK_CAPABILITIES),
            "material_regimes": sorted(MATERIAL_REGIME_CAPABILITIES),
            "robot_ids": ["fixture-arm"], "end_effector_ids": [],
            "controller_ids": [], "sensor_ids": ["fixture-rgb-v1"],
            "metric_ids": [], "parameter_ranges": {},
        },
        "metrics": {
            "physical_accuracy_error": 0.01, "uncertainty": 0.02,
            "scope_distance": 0.0, "harmful_false_negative_rate": 0.001,
            "reproducibility_score": 1.0, "privacy_preference": 1.0,
        },
        "approval": {
            "signature_status": "verified",
            "signature_id": "development-fixture-signature",
            "approved_by": ["benchmark-owner", "independent-reviewer"],
            "agent_approved": False,
        },
        "expiration_date": "2027-08-01",
        "self_grading": False,
    })
    return profile, qualification


def test_compiler_is_fail_closed_and_never_fabricates() -> None:
    compiled = compile_site_evidence_profile(
        profile_id="fail-closed-site",
        bundle_id="capture-1",
        bundle_hash=SHA_A,
        provenance_record_id="provenance-fixture",
        rights={"commercial_evaluation_allowed": True},
        privacy={"external_processing_allowed": False},
        metric_scale_verified=False,
        artifacts={
            "collider_qualification_report": {"report_id": "c1", "status": "failed"},
            "metric_scale_validation": {"validation_id": "s1", "status": "passed"},
            "totally_unknown_artifact": {"anything": True},
        },
    )
    profile, report = compiled["profile"], compiled["report"]
    collider = profile["evidence"]["validated_collider"]
    assert collider["available"] is True and collider["validated"] is False
    assert profile["evidence"]["metric_scale"]["validated"] is True
    assert report["unmapped_artifacts"] == ["totally_unknown_artifact"]
    assert report["fabricated_records"] == 0
    assert report["compiler_may_upgrade_validation"] is False
    with pytest.raises(SiteEvidenceCompilerError, match="artifact_not_object"):
        compile_site_evidence_profile(
            profile_id="broken", bundle_id="capture-1", bundle_hash=SHA_A,
            provenance_record_id="provenance-fixture",
            rights={}, privacy={}, metric_scale_verified=False,
            artifacts={"collider_qualification_report": "not-a-mapping"},
        )


def test_geometry_manifest_stays_candidate_until_collider_qualification() -> None:
    base = {
        "metric_geometry_manifest": {
            "manifest_id": "generated-geometry-candidate",
            "status": "passed",
            "proof_effect": "metric_reference_candidate_only",
        },
    }
    candidate = compile_site_evidence_profile(
        profile_id="geometry-candidate-site",
        bundle_id="capture-1",
        bundle_hash=SHA_A,
        provenance_record_id="provenance-fixture",
        rights={"commercial_evaluation_allowed": True},
        privacy={"external_processing_allowed": False},
        metric_scale_verified=False,
        artifacts=base,
    )["profile"]
    assert candidate["evidence"]["validated_mesh"]["available"] is True
    assert candidate["evidence"]["validated_mesh"]["validated"] is False

    qualified = compile_site_evidence_profile(
        profile_id="geometry-qualified-fixture-site",
        bundle_id="capture-1",
        bundle_hash=SHA_A,
        provenance_record_id="provenance-fixture",
        rights={"commercial_evaluation_allowed": True},
        privacy={"external_processing_allowed": False},
        metric_scale_verified=False,
        artifacts={
            **base,
            "collider_qualification_report": {
                "report_id": "fixture-collider-qualification",
                "status": "passed",
                "fixture_only": True,
            },
        },
    )["profile"]
    assert qualified["evidence"]["validated_mesh"]["validated"] is True
    assert qualified["evidence"]["validated_collider"]["validated"] is True


def test_kitchen_fixture_compiles_sparse_and_collision_routing_abstains() -> None:
    manifest = json.loads(
        (
            ROOT
            / "tests/fixtures/kitchen_task_min/kitchen_task_scaling_preflight_manifest.json"
        ).read_text(encoding="utf-8")
    )
    capture_manifest = json.loads(
        (
            ROOT / "tests/fixtures/kitchen_task_min/capture_raw_manifest.json"
        ).read_text(encoding="utf-8")
    )
    compiled = compile_site_evidence_profile(
        profile_id="kitchen-task-min-site",
        bundle_id="capture-1",
        bundle_hash=SHA_A,
        provenance_record_id="provenance-kitchen-fixture",
        rights={"commercial_evaluation_allowed": True},
        privacy={"external_processing_allowed": False},
        metric_scale_verified=False,
        artifacts={
            "capture_raw_manifest": capture_manifest,
            "kitchen_task_scaling_preflight_manifest": manifest,
        },
    )
    profile = compiled["profile"]
    # The fixture is a renderable modeled scene: appearance evidence only.
    assert set(profile["evidence"]) == {"appearance_mesh"}
    assert profile["evidence"]["appearance_mesh"]["available"] is True
    assert compiled["report"]["consumed_artifacts"] == [
        "capture_raw_manifest", "kitchen_task_scaling_preflight_manifest",
    ]

    engine_profile, qualification = _fixture_engine()
    requirements = derive_task_measurement_requirements(
        {"claim_id": "kitchen-collision", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    decision = route_task_site_measurement(
        requirements, profile, [engine_profile], [qualification],
        catalog_snapshot_hash=SHA_A, as_of=date(2026, 8, 2),
    )
    assert decision["status"] == "abstention"
    action = decision["abstention"]["smallest_next_action"]
    assert action["action_type"] == "collider_validation"
    assert "metric_scale" in action["exact_scope"]
    assert "validated_collider" in action["exact_scope"]


def test_every_task_class_routes_or_abstains_exactly_as_golden() -> None:
    assert set(_GOLDEN_SWEEP) == set(TASK_CAPABILITIES)
    site = _rich_site()
    engine_profile, qualification = _fixture_engine()
    for task_class, (expected_status, expected_action) in sorted(_GOLDEN_SWEEP.items()):
        claim: dict = {
            "claim_id": f"sweep-{task_class}",
            "claim_type": "collision_contact",
        }
        regime = _REGIME_BY_CLASS.get(task_class)
        if regime:
            claim["material_regimes"] = [regime]
        requirements = derive_task_measurement_requirements(
            claim, {"task_distribution": {"measurement_task_class": task_class}}
        )
        decision = route_task_site_measurement(
            requirements, site, [engine_profile], [qualification],
            catalog_snapshot_hash=SHA_A, as_of=date(2026, 8, 2),
        )
        assert decision["status"] == expected_status, task_class
        if expected_status == "abstention":
            action = decision["abstention"]["smallest_next_action"]
            assert action["action_type"] == expected_action, task_class
        else:
            assert decision["claim_boundary"]["permitted_claim"] == "C3", task_class
            assert decision["execution_authorized"] is False, task_class


def test_compiled_profile_attaches_to_testbed_and_drives_claim_plans() -> None:
    site = _rich_site()
    testbed = MaintainedSiteTaskTestbed.from_mapping({
        "schema_version": "maintained_site_task_testbed.v1",
        "testbed_id": "bridge-testbed",
        "version": "1",
        "predecessor_testbed_digest": None,
        "supersedes": [],
        "source_capture_bundles": [
            {"bundle_id": "capture-1", "version": "1", "digest": SHA_A}
        ],
        "artifact_references": {
            key: {"uri": f"fixture://{key}", "digest": SHA_B}
            for key in (
                "site_card", "task_cards", "scenario_cards", "eval_cards",
                "evaluator", "reset",
            )
        },
        "task_distribution": {"task_family": "rigid_object_pick_place"},
        "supported_condition_ranges": {"lighting_lux": [300, 600]},
        "robot_sensor_controller_bindings": {
            "embodiment": {"robot_id": "fixture-arm"},
            "sensors": {"camera": "fixture-rgb-v1"},
            "controller_action_representation": {"type": "joint_position"},
        },
        "governance": {
            "rights": "accepted", "consent": "accepted", "privacy": "cleared",
            "revocation": "version_invalidates_on_revocation",
            "allowed_uses": ["evaluation"],
        },
        "evidence_inventory": [{"evidence_id": "metric_geometry"}],
        "validation_envelope": {"site_id": "bridge-site"},
        "known_unsupported_conditions": [],
        "invalidation_triggers": [],
        "physical_outcome_history_refs": [],
        "lifecycle_state": "active",
    }).to_mapping()
    bound = attach_compiled_site_evidence(testbed, site)
    assert bound["site_evidence_profile"]["profile_id"] == "rich-compiled-site"

    engine_profile, qualification = _fixture_engine()
    method = {
        "schema_version": "evidence_method_profile.v1",
        "method_id": "fixture-universal-development-engine",
        "version": "1",
        "implementation_digest": SHA_B,
        "adapter_reference": "fixture.adapters:universal",
        "method_family": "traditional_simulation",
        "supported_claim_types": ["collision_contact"],
        "required_inputs": [],
        "applicability_envelope": {"testbed_ids": ["bridge-testbed"]},
        "calibration_evidence_references": [],
        "authority_tier": 2,
        "proof_tier": "tier_2",
        "correlation_group": "fixture-development",
        "shared_dependencies": [],
        "expected_cost_usd": 1.0,
        "expected_latency_seconds": 1.0,
        "reproducibility_level": "hermetic_fixture",
        "constraints": {"external_processing": False, "data_retention_days": 0},
        "provider_availability": {"status": "available"},
        "failure_modes": ["invalid_artifact"],
        "abstention_modes": ["uncertain"],
        "disqualifying_conditions": [],
        "self_qualified": False,
        "measurement_capability_profile": engine_profile,
        "evaluation_run_template": {
            "schema_version": "evaluation_run.v1",
            "run_id": "template",
            "mode": "evaluate",
            "scene_bundle": {
                "adapter_id": "capture_site_scene_bundle",
                "adapter_version": "1",
                "bundle_id": "capture-1",
                "uri": "fixture://capture-1",
                "entrypoint": "scene.usda",
                "content_digest": SHA_A,
            },
            "robot_adapter": {
                "adapter_id": "robot_profile_adapter",
                "adapter_version": "1",
                "robot_profile_id": "fixture-arm",
                "asset_ref": "fixture://robot",
            },
            "task_scenario_pack": {
                "adapter_id": "robot_eval_matrix_task_scenario_pack",
                "adapter_version": "1",
                "pack_id": "rigid-object-pack",
                "tasks": [{"task_id": "pick-red-block"}],
                "scenarios": [{"scenario_id": "base", "task_id": "pick-red-block"}],
            },
            "policy_adapter": {
                "adapter_id": "robot_eval_policy_package",
                "adapter_version": "1",
                "policy_id": "policy-a",
                "observation_schema_ref": "fixture_observation.v1",
                "action_schema_ref": "fixture_action.v1",
            },
            "runtime_provider_profile": {
                "adapter_id": "robot_eval_runtime_provider",
                "adapter_version": "1",
                "profile_id": "fixture-development-engine",
                "providers": ["fixture_local"],
                "simulator": "mujoco",
                "max_spend_usd": 0,
            },
            "proof_contract": {
                "adapter_id": "robot_eval_proof_contract",
                "adapter_version": "1",
                "contract_id": "fixture-proof",
                "required_evidence": ["fixture_result"],
                "claim_ceiling": {"level": "sim_only"},
                "prohibited_claims": ["physical_success", "deployment_readiness"],
            },
            "metadata": {},
        },
    }
    validated_method = EvidenceMethodProfile.from_mapping(method).to_mapping()
    outer_qualification = {
        "schema_version": "evidence_method_qualification.v1",
        "qualification_id": "fixture-outer-qualification",
        "method_id": "fixture-universal-development-engine",
        "method_version": "1",
        "claim_type": "collision_contact",
        "task_family": "rigid_object_pick_place",
        "method_profile_digest": validated_method["method_profile_digest"],
        "implementation_digest": SHA_B,
        "evaluator_digest": SHA_B,
        "site_domain_conditions": {"lighting_lux": [300, 600]},
        "embodiment": {"robot_id": "fixture-arm"},
        "sensors": {"camera": "fixture-rgb-v1"},
        "controller_action_representation": {"type": "joint_position"},
        "evaluator": {"evaluator_id": "fixture-evaluator"},
        "confidence_intervals": {"coverage": [0.6, 0.9]},
        "provenance": {"source": "development_fixture"},
        "predictions": [{"case": "fixture", "value": 1.0}],
        "accepted_real_outcomes": [{"case": "fixture", "value": 1.0}],
        "calibration_partition": "calibration",
        "coverage": 0.9,
        "abstention_rate": 0.0,
        "false_safe_rate": 0.001,
        "false_reject_rate": 0.01,
        "owner_evidence": ["fixture://owner-evidence"],
        "status": "qualified",
        "self_grading": False,
        "measurement_qualification_record": qualification,
    }
    validated_qualification = QualificationRecord.from_mapping(
        outer_qualification
    ).to_mapping()
    request = DecisionEvidenceRequest.from_mapping({
        "schema_version": "decision_evidence_request.v1",
        "request_id": "bridge-request",
        "decision_id": "bridge-decision",
        "testbed_id": bound["testbed_id"],
        "testbed_version": bound["version"],
        "testbed_digest": bound["testbed_digest"],
        "decision_question": "Can the arm pick the part on this site?",
        "candidates": [],
        "claims": [
            {
                "claim_id": "bridge-collision",
                "claim_type": "collision_contact",
                "subject": "fixture:collision",
                "measurable_threshold": {
                    "operator": ">=", "value": 0.8, "units": "ratio",
                },
                "false_safe_consequence": "moderate",
                "acceptable_false_safe_risk": 0.05,
                "desired_confidence_or_coverage": {"minimum_coverage": 0.5},
                "permitted_abstention_behavior": {"allowed": True},
            }
        ],
        "budget": {"max_cost_usd": 20.0},
        "deadline": "2026-08-30T00:00:00Z",
        "available_physical_evidence": [],
        "permitted_evidence_methods": ["traditional_simulation"],
        "restrictions": {"external_processing_allowed": False},
        "requested_result_audience": "robot_team_buyer",
        "provenance": {"caller_identity": "bridge-test"},
        "idempotency_key": "bridge-request",
    }).to_mapping()
    plan = route_decision_evidence(
        request, bound, [validated_method], [validated_qualification]
    ).to_mapping()
    claim_plan = plan["claim_plans"][0]
    measurement = claim_plan["measurement_routing_decision"]
    assert claim_plan["status"] == "planned"
    assert measurement["status"] == "route_selected"
    assert measurement["selected_route"]["stages"][0]["method_id"] == (
        "fixture-universal-development-engine"
    )
    assert measurement["execution_authorized"] is False
