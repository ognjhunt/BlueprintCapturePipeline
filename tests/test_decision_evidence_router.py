from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

import blueprint_pipeline.decision_evidence_contracts as decision_contracts
from blueprint_pipeline.decision_evidence_contracts import (
    DecisionEvidenceContractError,
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
)
from blueprint_pipeline.decision_evidence_router import route_decision_evidence
from blueprint_pipeline.decision_evidence_execution import (
    EvidenceMethodAdapterRegistry,
    build_decision_envelope,
    execute_evidence_plan,
)
from blueprint_pipeline.physical_outcome_learning import join_physical_outcome
from blueprint_pipeline.legacy_wam_evidence import (
    translate_wam_cross_check_plan,
    wam_scorecard_as_debug_evidence,
)
from blueprint_pipeline.legacy_task_evaluation_translation import (
    translate_policy_improvement_request,
    translate_post_training_data_request,
)
from blueprint_pipeline.task_evaluation_evidence_use import determine_evidence_use
from blueprint_pipeline.decision_evidence_cli import main as decision_evidence_cli_main
from blueprint_pipeline.decision_evidence_handoff import verify as verify_webapp_handoff
from blueprint_pipeline.evaluation_run import (
    LegacyEvaluationPackSpec,
    EvaluationRunSpec as LegacyAlias,
    get_evaluation_pack,
    legacy_evaluation_pack_to_leaf_spec,
)
from blueprint_pipeline.evaluation_run_contract import validate_evaluation_run_spec


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _testbed() -> dict:
    return MaintainedSiteTaskTestbed.from_mapping(
        {
            "schema_version": "maintained_site_task_testbed.v1",
            "testbed_id": "rigid-object-site-task",
            "version": "1",
            "predecessor_testbed_digest": None,
            "supersedes": [],
            "source_capture_bundles": [
                {"bundle_id": "capture-1", "version": "3", "digest": SHA_A}
            ],
            "artifact_references": {
                "site_card": {"uri": "fixture://site-card", "digest": SHA_A},
                "task_cards": [{"uri": "fixture://task-card", "digest": SHA_A}],
                "scenario_cards": [{"uri": "fixture://scenario", "digest": SHA_A}],
                "eval_cards": [{"uri": "fixture://eval", "digest": SHA_A}],
                "evaluator": {"uri": "fixture://evaluator", "digest": SHA_B},
                "reset": {"uri": "fixture://reset", "digest": SHA_B},
            },
            "task_distribution": {
                "task_family": "rigid_object_pick_place",
                "tasks": ["pick-red-block"],
            },
            "supported_condition_ranges": {"lighting_lux": [300, 600]},
            "robot_sensor_controller_bindings": {
                "embodiment": {"robot_id": "fixture-arm"},
                "sensors": {"camera": "fixture-rgb-v1"},
                "controller_action_representation": {"type": "joint_position"},
            },
            "governance": {
                "rights": "accepted",
                "consent": "accepted",
                "privacy": "cleared",
                "revocation": "version_invalidates_on_revocation",
                "allowed_uses": ["evaluation"],
            },
            "evidence_inventory": [
                {"evidence_id": "metric_geometry"},
                {"evidence_id": "captured_rgb_frames"},
                {"evidence_id": "collision_scene"},
            ],
            "validation_envelope": {"site_id": "fixture-site", "exact_scope": True},
            "known_unsupported_conditions": ["transparent_objects"],
            "invalidation_triggers": ["layout_change", "camera_change", "robot_change"],
            "physical_outcome_history_refs": [],
            "lifecycle_state": "active",
        }
    ).to_mapping()


def _claim(claim_id: str, claim_type: str, *, coverage: float = 0.8) -> dict:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "subject": f"fixture:{claim_id}",
        "measurable_threshold": {"operator": ">=", "value": 0.8, "units": "ratio"},
        "false_safe_consequence": "critical" if claim_type == "deployment_readiness" else "moderate",
        "acceptable_false_safe_risk": 0.001 if claim_type == "deployment_readiness" else 0.05,
        "desired_confidence_or_coverage": {
            "minimum_coverage": coverage,
            "minimum_independent_methods": 1,
        },
        "permitted_abstention_behavior": {"allowed": True},
        "task_family": "rigid_object_pick_place",
        "site_domain_conditions": {"lighting_lux": [300, 600]},
        "embodiment": {"robot_id": "fixture-arm"},
        "sensors": {"camera": "fixture-rgb-v1"},
        "controller_action_representation": {"type": "joint_position"},
    }


def _request(testbed: dict) -> dict:
    return DecisionEvidenceRequest.from_mapping(
        {
            "schema_version": "decision_evidence_request.v1",
            "request_id": "request-1",
            "decision_id": "decision-1",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "decision_question": "Which claims are supported for the rigid-object task?",
            "candidates": [{"robot_id": "fixture-arm"}, {"policy_id": "policy-a"}],
            "claims": [
                _claim("reach", "reachability"),
                _claim("visible", "perception_visibility"),
                _claim("collision", "collision_contact"),
                _claim("cycle", "cycle_time"),
                _claim("ranking", "comparative_policy_ranking"),
                _claim("deploy", "deployment_readiness"),
            ],
            "budget": {
                "max_cost_usd": 20.0,
                "max_latency_seconds": 120.0,
                "delay_cost_per_second": 0.001,
            },
            "deadline": "2026-07-30T00:00:00Z",
            "available_physical_evidence": [],
            "permitted_evidence_methods": [
                "analytic_geometry_kinematics",
                "captured_real_observation",
                "traditional_simulation",
                "learned_world_model",
                "physical_evidence",
            ],
            "restrictions": {
                "external_processing_allowed": False,
                "max_data_retention_days": 0,
            },
            "requested_result_audience": "robot_team_buyer",
            "provenance": {"caller_identity": "fixture-test"},
            "idempotency_key": "fixture-request-1",
        }
    ).to_mapping()


def _leaf_template() -> dict:
    return {
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
            "profile_id": "fixture-mujoco",
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
    }


def _profile(
    method_id: str,
    family: str,
    claim_types: list[str],
    *,
    required_inputs: list[str],
    authority: int,
    cost: float,
    latency: float,
    correlation_group: str,
    available: bool = True,
) -> dict:
    value = {
        "schema_version": "evidence_method_profile.v1",
        "method_id": method_id,
        "version": "1",
        "implementation_digest": SHA_B,
        "adapter_reference": f"fixture.adapters:{method_id}",
        "method_family": family,
        "supported_claim_types": claim_types,
        "required_inputs": required_inputs,
        "applicability_envelope": {
            "testbed_ids": ["rigid-object-site-task"],
            "testbed_versions": ["1"],
            "task_families": ["rigid_object_pick_place"],
        },
        "calibration_evidence_references": ["fixture://calibration"],
        "authority_tier": authority,
        "proof_tier": f"tier_{authority}",
        "correlation_group": correlation_group,
        "shared_dependencies": ["capture-1"],
        "expected_cost_usd": cost,
        "expected_latency_seconds": latency,
        "reproducibility_level": "hermetic_fixture",
        "constraints": {"external_processing": False, "data_retention_days": 0},
        "provider_availability": {"status": "available" if available else "unavailable"},
        "failure_modes": ["invalid_artifact", "domain_mismatch"],
        "abstention_modes": ["uncertain", "missing_input"],
        "disqualifying_conditions": ["transparent_objects"],
        "self_qualified": False,
    }
    if family in {"traditional_simulation", "learned_world_model", "external_provider_tool"}:
        value["evaluation_run_template"] = _leaf_template()
        value["evaluation_run_template"]["runtime_provider_profile"]["profile_id"] = method_id
    return EvidenceMethodProfile.from_mapping(value).to_mapping()


def _qualification(profile: dict, claim_type: str, *, status: str = "qualified") -> dict:
    value = {
        "schema_version": "evidence_method_qualification.v1",
        "qualification_id": f"qualification-{profile['method_id']}-{claim_type}",
        "method_id": profile["method_id"],
        "method_version": profile["version"],
        "method_profile_digest": profile["method_profile_digest"],
        "implementation_digest": profile["implementation_digest"],
        "claim_type": claim_type,
        "task_family": "rigid_object_pick_place",
        "site_domain_conditions": {"lighting_lux": [300, 600]},
        "embodiment": {"robot_id": "fixture-arm"},
        "sensors": {"camera": "fixture-rgb-v1"},
        "controller_action_representation": {"type": "joint_position"},
        "evaluator": {"evaluator_id": "independent-fixture-evaluator", "version": "1"},
        "evaluator_digest": SHA_C,
        "predictions": [{"prediction_id": "prediction-1", "value": True}],
        "accepted_real_outcomes": [{"outcome_id": "physical-anchor-1", "value": True}],
        "calibration_partition": "heldout",
        "confidence_intervals": {"level": 0.95, "lower": 0.9, "upper": 1.0},
        "coverage": 0.95,
        "abstention_rate": 0.05,
        "false_safe_rate": 0.01,
        "false_reject_rate": 0.02,
        "provenance": {"source": "fixture-anchor"},
        "owner_evidence": [{"uri": "fixture://owner-evidence", "digest": SHA_D}],
        "status": status,
        "self_grading": False,
        "subject_provider_id": "fixture-method-owner",
        "evaluator_provider_id": "independent-fixture-evaluator",
    }
    if claim_type == "comparative_policy_ranking":
        value.update(
            {
                "policy_checkpoint_identity": {
                    "policy_ids": ["policy-a", "policy-b"],
                    "adapted_policy_is_distinct_identity": True,
                },
                "perturbation_sensitivity_metrics": {
                    "spearman": 0.8,
                    "pearson": 0.75,
                    "mae": 0.1,
                    "mmrv": 0.05,
                },
                "simulated_rollout_count": 100,
                "physical_rollout_count": 25,
            }
        )
    return QualificationRecord.from_mapping(value).to_mapping()


def _registry() -> tuple[list[dict], list[dict]]:
    profiles = [
        _profile(
            "analytic-reach",
            "analytic_geometry_kinematics",
            ["reachability"],
            required_inputs=["metric_geometry"],
            authority=1,
            cost=0.1,
            latency=1,
            correlation_group="capture-geometry",
        ),
        _profile(
            "captured-visibility",
            "captured_real_observation",
            ["perception_visibility"],
            required_inputs=["captured_rgb_frames"],
            authority=1,
            cost=0.2,
            latency=2,
            correlation_group="capture-rgb",
        ),
        _profile(
            "fixture-mujoco",
            "traditional_simulation",
            ["collision_contact", "cycle_time"],
            required_inputs=["collision_scene"],
            authority=2,
            cost=1.0,
            latency=10,
            correlation_group="fixture-physics",
        ),
        _profile(
            "expensive-sim",
            "traditional_simulation",
            ["collision_contact"],
            required_inputs=["collision_scene"],
            authority=2,
            cost=4.0,
            latency=30,
            correlation_group="fixture-physics-2",
        ),
        _profile(
            "uncalibrated-world-model",
            "learned_world_model",
            ["comparative_policy_ranking"],
            required_inputs=["captured_rgb_frames"],
            authority=3,
            cost=3.0,
            latency=20,
            correlation_group="capture-label-lineage",
        ),
        _profile(
            "physical-outcome",
            "physical_evidence",
            ["deployment_readiness"],
            required_inputs=["accepted_physical_outcome"],
            authority=4,
            cost=10.0,
            latency=100,
            correlation_group="physical-owner-evidence",
        ),
    ]
    qualifications = [
        _qualification(profiles[0], "reachability"),
        _qualification(profiles[1], "perception_visibility"),
        _qualification(profiles[2], "collision_contact"),
        _qualification(profiles[2], "cycle_time"),
        _qualification(profiles[3], "collision_contact"),
        # The world-model profile is deliberately not qualified.
        _qualification(profiles[5], "deployment_readiness"),
    ]
    return profiles, qualifications


def test_legacy_pack_type_is_explicit_and_alias_remains_compatible() -> None:
    pack = get_evaluation_pack("g1_kitchen")
    assert isinstance(pack, LegacyEvaluationPackSpec)
    assert LegacyAlias is LegacyEvaluationPackSpec


def test_legacy_pack_translation_requires_runtime_identity_and_emits_canonical_leaf() -> None:
    pack = get_evaluation_pack("g1_kitchen")
    with pytest.raises(Exception, match="leaf_scene_content_digest_invalid"):
        legacy_evaluation_pack_to_leaf_spec(
            pack,
            run_id="legacy-1",
            scene_uri="fixture://scene",
            scene_content_digest="missing",
            robot_asset_ref="fixture://g1",
        )
    leaf = legacy_evaluation_pack_to_leaf_spec(
        pack,
        run_id="legacy-1",
        scene_uri="fixture://scene",
        scene_content_digest=SHA_A,
        robot_asset_ref="fixture://g1",
    )
    assert leaf["schema_version"] == "evaluation_run.v1"
    assert leaf["metadata"]["legacy_defaults_are_qualification_evidence"] is False
    assert validate_evaluation_run_spec(leaf)["status"] == "passed"


def test_contracts_reject_secret_values_before_digesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["selected_provider"] = "provider-a"
    request["provenance"]["token"] = "secret-value"
    monkeypatch.setattr(
        decision_contracts,
        "canonical_digest",
        lambda *_args, **_kwargs: pytest.fail("secret-bearing artifact was digested"),
    )
    with pytest.raises(DecisionEvidenceContractError) as excinfo:
        DecisionEvidenceRequest.from_mapping(request)
    assert "request_method_selection_forbidden:selected_provider" in excinfo.value.errors
    assert "secret_value_forbidden:provenance.token" in excinfo.value.errors


def test_router_vertical_plan_is_deterministic_partial_and_multi_leaf() -> None:
    testbed = _testbed()
    request = _request(testbed)
    profiles, qualifications = _registry()
    first = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    second = route_decision_evidence(
        copy.deepcopy(request),
        copy.deepcopy(testbed),
        list(reversed(copy.deepcopy(profiles))),
        list(reversed(copy.deepcopy(qualifications))),
    ).to_mapping()

    assert first == second
    assert len(first["compiled_evaluation_run_specs"]) == 2
    assert len(first["non_evaluation_run_steps"]) == 2
    by_claim = {row["claim_id"]: row for row in first["claim_plans"]}
    assert by_claim["reach"]["selected_methods"][0]["method_id"] == "analytic-reach"
    assert by_claim["visible"]["selected_methods"][0]["method_id"] == "captured-visibility"
    assert by_claim["collision"]["selected_methods"][0]["method_id"] == "fixture-mujoco"
    assert by_claim["cycle"]["selected_methods"][0]["method_id"] == "fixture-mujoco"
    assert by_claim["ranking"]["status"] == "abstention_planned"
    assert by_claim["deploy"]["status"] == "abstention_planned"
    assert first["physical_evidence_requests"][0]["robot_run_initiated"] is False
    world = next(
        row
        for row in by_claim["ranking"]["candidate_methods_considered"]
        if row["method_id"] == "uncalibrated-world-model"
    )
    assert "unqualified_or_out_of_scope" in world["rejection_reasons"]
    expensive = next(
        row
        for row in by_claim["collision"]["candidate_methods_considered"]
        if row["method_id"] == "expensive-sim"
    )
    assert "dominated_by:fixture-mujoco" in expensive["rejection_reasons"]
    assert first["router_policy"]["policy_ranking_thesis_verdict"] == "thesis_not_supported"


def test_router_enforces_false_safe_coverage_budget_rights_inputs_and_availability() -> None:
    testbed = _testbed()
    request = _request(testbed)
    profiles, qualifications = _registry()

    collision_q = next(row for row in qualifications if row["claim_type"] == "collision_contact" and row["method_id"] == "fixture-mujoco")
    collision_q.pop("qualification_digest")
    collision_q["false_safe_rate"] = 0.5
    collision_q = QualificationRecord.from_mapping(collision_q).to_mapping()
    qualifications = [
        collision_q
        if row["claim_type"] == "collision_contact" and row["method_id"] == "fixture-mujoco"
        else row
        for row in qualifications
    ]
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    collision = next(row for row in plan["claim_plans"] if row["claim_id"] == "collision")
    candidate = next(row for row in collision["candidate_methods_considered"] if row["method_id"] == "fixture-mujoco")
    assert "false_safe_limit_exceeded" in candidate["rejection_reasons"]

    restricted = copy.deepcopy(request)
    restricted.pop("request_digest")
    restricted["restrictions"]["allowed_method_families"] = ["captured_real_observation"]
    restricted = DecisionEvidenceRequest.from_mapping(restricted).to_mapping()
    restricted_plan = route_decision_evidence(restricted, testbed, profiles, qualifications).to_mapping()
    reach = next(row for row in restricted_plan["claim_plans"] if row["claim_id"] == "reach")
    assert reach["status"] == "abstention_planned"


def test_abstention_next_experiment_ignores_methods_for_other_claim_types() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("collision", "collision_contact")]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    cheap_wrong_claim = _profile(
        "cheap-reach-only",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=2,
        cost=0.0,
        latency=0.0,
        correlation_group="reach-only",
    )
    compatible_unqualified = _profile(
        "site-contact-simulation",
        "traditional_simulation",
        ["collision_contact"],
        required_inputs=["collision_scene"],
        authority=2,
        cost=1.0,
        latency=10.0,
        correlation_group="site-collision",
    )

    plan = route_decision_evidence(
        request,
        testbed,
        [cheap_wrong_claim, compatible_unqualified],
        [],
    ).to_mapping()

    claim_plan = plan["claim_plans"][0]
    assert claim_plan["status"] == "abstention_planned"
    assert claim_plan["next_cheapest_experiment"].startswith(
        "qualify_or_supply:site-contact-simulation:"
    )
    assert "claim_type_not_supported" not in claim_plan["next_cheapest_experiment"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing_input", "required_input_missing:metric_geometry"),
        ("unavailable", "method_unavailable"),
        ("over_budget", "over_budget"),
        ("over_latency", "over_latency_budget"),
        ("unsupported_domain", "unsupported_domain:testbed_ids"),
    ],
)
def test_router_rejection_gates_are_explicit(mutation: str, expected_reason: str) -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("reach", "reachability")]
    profile = _profile(
        "analytic-reach",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=1,
        cost=0.1,
        latency=1,
        correlation_group="capture-geometry",
    )
    if mutation == "missing_input":
        testbed.pop("testbed_digest")
        testbed["evidence_inventory"] = []
        testbed = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
        request["testbed_digest"] = testbed["testbed_digest"]
    elif mutation == "unavailable":
        profile.pop("method_profile_digest")
        profile["provider_availability"] = {"status": "unavailable"}
        profile = EvidenceMethodProfile.from_mapping(profile).to_mapping()
    elif mutation == "over_budget":
        request["budget"]["max_cost_usd"] = 0.0
    elif mutation == "over_latency":
        request["budget"]["max_latency_seconds"] = 0.5
    elif mutation == "unsupported_domain":
        profile.pop("method_profile_digest")
        profile["applicability_envelope"]["testbed_ids"] = ["different-testbed"]
        profile = EvidenceMethodProfile.from_mapping(profile).to_mapping()
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    qualification = _qualification(profile, "reachability")
    plan = route_decision_evidence(request, testbed, [profile], [qualification]).to_mapping()
    candidate = plan["claim_plans"][0]["candidate_methods_considered"][0]
    assert expected_reason in candidate["rejection_reasons"]
    assert plan["claim_plans"][0]["status"] == "abstention_planned"


def test_provider_privacy_and_retention_restrictions_fail_closed() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("reach", "reachability")]
    request["permitted_evidence_methods"].append("external_provider_tool")
    request["restrictions"]["prohibited_provider_ids"] = ["provider-x"]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    profile = _profile(
        "provider-reach",
        "external_provider_tool",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=2,
        cost=1,
        latency=5,
        correlation_group="provider-x-lineage",
    )
    profile.pop("method_profile_digest")
    profile["constraints"] = {
        "external_processing": True,
        "provider_id": "provider-x",
        "data_retention_days": 30,
    }
    profile = EvidenceMethodProfile.from_mapping(profile).to_mapping()
    plan = route_decision_evidence(
        request, testbed, [profile], [_qualification(profile, "reachability")]
    ).to_mapping()
    reasons = plan["claim_plans"][0]["candidate_methods_considered"][0]["rejection_reasons"]
    assert "external_processing_rights_incompatible" in reasons
    assert "provider_restricted" in reasons
    assert "data_retention_incompatible" in reasons


def test_correlated_methods_do_not_satisfy_independent_evidence_requirement() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    claim = _claim("reach", "reachability")
    claim["desired_confidence_or_coverage"]["minimum_independent_methods"] = 2
    request["claims"] = [claim]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    first = _profile(
        "reach-a",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=1,
        cost=0.1,
        latency=1,
        correlation_group="shared-capture-lineage",
    )
    second = _profile(
        "reach-b",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=2,
        cost=0.2,
        latency=2,
        correlation_group="shared-capture-lineage",
    )
    plan = route_decision_evidence(
        request,
        testbed,
        [first, second],
        [_qualification(first, "reachability"), _qualification(second, "reachability")],
    ).to_mapping()
    assert plan["claim_plans"][0]["status"] == "abstention_planned"
    assert plan["shared_dependency_warnings"][0]["counted_as_independent"] is False


def test_exact_testbed_version_and_digest_binding_rejects_stale_request() -> None:
    testbed = _testbed()
    request = _request(testbed)
    profiles, qualifications = _registry()
    for field, value, error in (
        ("testbed_version", "stale-version", "decision_request_testbed_version_mismatch"),
        ("testbed_digest", SHA_D, "decision_request_testbed_digest_mismatch"),
    ):
        stale = copy.deepcopy(request)
        stale.pop("request_digest")
        stale[field] = value
        stale = DecisionEvidenceRequest.from_mapping(stale).to_mapping()
        with pytest.raises(ValueError, match=error):
            route_decision_evidence(stale, testbed, profiles, qualifications)


def test_all_required_method_families_are_representable() -> None:
    families = {
        "analytic_geometry_kinematics",
        "captured_real_observation",
        "traditional_simulation",
        "learned_world_model",
        "external_provider_tool",
        "physical_evidence",
        "owner_attested_operational_input",
    }
    for family in sorted(families):
        profile = _profile(
            f"representable-{family}",
            family,
            ["operational_input"],
            required_inputs=[],
            authority=1,
            cost=0,
            latency=0,
            correlation_group=f"group-{family}",
        )
        assert profile["method_family"] == family


def test_generated_method_cannot_upgrade_raw_physical_or_deployment_claim_ceiling() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("generated", "perception_visibility")]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    profile = _profile(
        "generated-provider",
        "learned_world_model",
        ["perception_visibility"],
        required_inputs=["captured_rgb_frames"],
        authority=2,
        cost=1,
        latency=1,
        correlation_group="generated-lineage",
    )
    qualification = _qualification(profile, "perception_visibility")
    plan = route_decision_evidence(request, testbed, [profile], [qualification]).to_mapping()
    raw = {
        "status": "valid",
        "supports_claim": True,
        "uncertainty": 0.01,
        "coverage": 0.95,
        "blockers": [],
        "invalid_rollout_reasons": [],
        "raw_artifact_references": [{"uri": "fixture://generated", "digest": SHA_D}],
        "provenance": {"generated": True},
        "claim_ceiling": {
            "physical_success": True,
            "deployment_readiness": True,
            "safety_certification": True,
        },
        "false_safe_risk": 0.01,
    }
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        [profile],
        [qualification],
        registry=EvidenceMethodAdapterRegistry(
            [_RawFixtureAdapter(profile["adapter_reference"], raw)]
        ),
    )
    ceiling = execution.results[0].to_mapping()["claim_ceiling"]
    assert ceiling["physical_success"] is False
    assert ceiling["deployment_readiness"] is False
    assert ceiling["safety_certification"] is False
    assert ceiling["generated_artifact_upgrades_raw_or_physical_claim"] is False


def test_already_recorded_physical_evidence_is_read_only_and_never_starts_robot() -> None:
    testbed = _testbed()
    testbed.pop("testbed_digest")
    testbed["evidence_inventory"].append({"evidence_id": "accepted_physical_outcome"})
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("physical", "physical_task_success")]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    profile = _profile(
        "accepted-physical-outcome",
        "physical_evidence",
        ["physical_task_success"],
        required_inputs=["accepted_physical_outcome"],
        authority=4,
        cost=0,
        latency=0,
        correlation_group="owner-accepted-outcome",
    )
    qualification = _qualification(profile, "physical_task_success")
    plan = route_decision_evidence(request, testbed, [profile], [qualification]).to_mapping()
    assert plan["physical_evidence_requests"] == []
    raw = {
        "status": "valid",
        "supports_claim": True,
        "uncertainty": 0.01,
        "coverage": 0.95,
        "blockers": [],
        "invalid_rollout_reasons": [],
        "raw_artifact_references": [{"uri": "fixture://physical-outcome", "digest": SHA_D}],
        "provenance": {"accepted_outcome_read_only": True},
        "claim_ceiling": {"physical_success": True},
        "false_safe_risk": 0.01,
    }
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        [profile],
        [qualification],
        registry=EvidenceMethodAdapterRegistry(
            [_RawFixtureAdapter(profile["adapter_reference"], raw, physical_read_only=True)]
        ),
    )
    assert execution.results[0].to_mapping()["claim_ceiling"]["physical_success"] is True
    assert execution.execution_manifest["physical_robot_run_initiated"] is False
    decision = build_decision_envelope(
        request, testbed, plan, [execution.results[0].to_mapping()]
    ).to_mapping()
    assert decision["claim_ceiling"]["physical_success"] is True
    assert decision["deployment_approval"] is False


def test_qualification_rejects_provider_self_grading() -> None:
    profile = _profile(
        "self-grader",
        "learned_world_model",
        ["comparative_policy_ranking"],
        required_inputs=[],
        authority=3,
        cost=1,
        latency=1,
        correlation_group="self",
    )
    qualification = _qualification(profile, "comparative_policy_ranking")
    qualification.pop("qualification_digest")
    qualification["evaluator_provider_id"] = qualification["subject_provider_id"]
    with pytest.raises(DecisionEvidenceContractError, match="provider_self_grading_forbidden"):
        QualificationRecord.from_mapping(qualification)


class _FixtureAdapter:
    def __init__(self, adapter_reference: str, *, supports_claim: bool, finding: str):
        self.adapter_reference = adapter_reference
        self._supports_claim = supports_claim
        self._finding = finding

    def execute(self, **kwargs):
        profile = kwargs["method_profile"]
        return {
            "status": "valid",
            "supports_claim": self._supports_claim,
            "observed_value": 0.9 if self._supports_claim else 0.1,
            "categorical_finding": self._finding,
            "uncertainty": 0.02,
            "coverage": 0.95,
            "applicability_envelope": profile["applicability_envelope"],
            "raw_artifact_references": [
                {"uri": f"fixture://{profile['method_id']}/result", "digest": SHA_D}
            ],
            "provenance": {"fixture_backed": True},
            "cost_usd": profile["expected_cost_usd"],
            "duration_seconds": profile["expected_latency_seconds"],
            "blockers": [],
            "invalid_rollout_reasons": [],
            "claim_ceiling": {
                "method_family": profile["method_family"],
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
            "false_safe_risk": 0.01,
        }


class _RawFixtureAdapter:
    def __init__(self, adapter_reference: str, result: dict, *, physical_read_only: bool = False):
        self.adapter_reference = adapter_reference
        self._result = result
        if physical_read_only:
            self.physical_evidence_mode = "read_only"

    def execute(self, **kwargs):
        return {**self._result, "applicability_envelope": kwargs["method_profile"]["applicability_envelope"]}


def _execution_fixture():
    testbed = _testbed()
    request = _request(testbed)
    profiles, qualifications = _registry()
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    adapters = []
    for profile in profiles:
        if profile["method_id"] in {
            "analytic-reach",
            "captured-visibility",
            "fixture-mujoco",
        }:
            adapters.append(
                _FixtureAdapter(
                    profile["adapter_reference"],
                    supports_claim=profile["method_id"] != "captured-visibility" or True,
                    finding=f"{profile['method_id']}:supports",
                )
            )
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=EvidenceMethodAdapterRegistry(adapters),
        context={"ephemeral_fixture_root": "/tmp/not-persisted"},
    )
    envelope = build_decision_envelope(
        request,
        testbed,
        plan,
        [result.to_mapping() for result in execution.results],
    )
    return testbed, request, profiles, qualifications, plan, execution, envelope


def test_execution_normalizes_bindings_and_returns_partial_decision() -> None:
    testbed, request, _, _, plan, execution, envelope = _execution_fixture()
    assert len(execution.results) == 4
    assert execution.execution_manifest["context_values_persisted"] is False
    assert execution.execution_manifest["physical_robot_run_initiated"] is False
    for result in execution.results:
        row = result.to_mapping()
        assert row["request_digest"] == request["request_digest"]
        assert row["plan_digest"] == plan["plan_digest"]
        assert row["testbed_digest"] == testbed["testbed_digest"]
        assert row["raw_policy_values_persisted"] is False
    decision = envelope.to_mapping()
    assert decision["overall_outcome"] == "partial_decision"
    verdicts = {row["claim_id"]: row["verdict"] for row in decision["per_claim_verdicts"]}
    assert verdicts["reach"] == "supported"
    assert verdicts["visible"] == "supported"
    assert verdicts["collision"] == "supported"
    assert verdicts["ranking"] == "abstention"
    assert verdicts["deploy"] == "abstention"
    assert decision["deployment_approval"] is False
    assert decision["safety_certification"] is False
    assert decision["uncertainty"]["ranking_science_boundary"] == "thesis_not_supported"


def test_invalid_primary_result_executes_next_qualified_escalation() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("reach", "reachability")]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    primary = _profile(
        "primary-reach",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=1,
        cost=0.1,
        latency=1,
        correlation_group="primary-geometry",
    )
    escalation = _profile(
        "stronger-reach",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=2,
        cost=0.5,
        latency=2,
        correlation_group="independent-geometry",
    )
    profiles = [primary, escalation]
    qualifications = [
        _qualification(primary, "reachability"),
        _qualification(escalation, "reachability"),
    ]
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    claim_plan = plan["claim_plans"][0]
    assert claim_plan["selected_methods"][0]["method_id"] == "primary-reach"
    assert claim_plan["escalation_methods"][0]["method_id"] == "stronger-reach"
    invalid = {
        "status": "invalid",
        "supports_claim": None,
        "uncertainty": 1.0,
        "coverage": 0.0,
        "blockers": ["invalid_geometry_artifact"],
        "invalid_rollout_reasons": ["checksum_mismatch"],
        "raw_artifact_references": [],
        "provenance": {"fixture": True},
        "false_safe_risk": 1.0,
    }
    valid = {
        "status": "valid",
        "supports_claim": True,
        "uncertainty": 0.01,
        "coverage": 0.95,
        "blockers": [],
        "invalid_rollout_reasons": [],
        "raw_artifact_references": [{"uri": "fixture://stronger", "digest": SHA_D}],
        "provenance": {"fixture": True},
        "false_safe_risk": 0.01,
    }
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=EvidenceMethodAdapterRegistry(
            [
                _RawFixtureAdapter(primary["adapter_reference"], invalid),
                _RawFixtureAdapter(escalation["adapter_reference"], valid),
            ]
        ),
    )
    assert [result.to_mapping()["status"] for result in execution.results] == ["invalid", "valid"]
    envelope = build_decision_envelope(
        request, testbed, plan, [result.to_mapping() for result in execution.results]
    ).to_mapping()
    assert envelope["per_claim_verdicts"][0]["verdict"] == "supported"


def test_sufficient_primary_skips_conditional_escalation() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [_claim("reach", "reachability")]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    primary = _profile(
        "primary-reach",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=1,
        cost=0.1,
        latency=1,
        correlation_group="primary-geometry",
    )
    escalation = _profile(
        "stronger-reach",
        "analytic_geometry_kinematics",
        ["reachability"],
        required_inputs=["metric_geometry"],
        authority=2,
        cost=0.5,
        latency=2,
        correlation_group="independent-geometry",
    )
    qualifications = [_qualification(primary, "reachability"), _qualification(escalation, "reachability")]
    plan = route_decision_evidence(request, testbed, [primary, escalation], qualifications).to_mapping()
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        [primary, escalation],
        qualifications,
        registry=EvidenceMethodAdapterRegistry(
            [
                _FixtureAdapter(primary["adapter_reference"], supports_claim=True, finding="reachable"),
                _FixtureAdapter(escalation["adapter_reference"], supports_claim=True, finding="reachable"),
            ]
        ),
    )
    assert len(execution.results) == 1
    assert execution.execution_manifest["steps"][1]["status"] == "skipped_evidence_already_sufficient"
    assert execution.execution_manifest["status"] == "completed"


def test_disagreement_abstains_instead_of_counting_votes() -> None:
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    pose_claim = _claim("pose", "object_pose")
    pose_claim["desired_confidence_or_coverage"]["minimum_independent_methods"] = 2
    request["claims"].append(pose_claim)
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    profiles, qualifications = _registry()
    pose_a = _profile(
        "pose-captured",
        "captured_real_observation",
        ["object_pose"],
        required_inputs=["captured_rgb_frames"],
        authority=2,
        cost=0.4,
        latency=4,
        correlation_group="pose-camera",
    )
    pose_b = _profile(
        "pose-geometry",
        "analytic_geometry_kinematics",
        ["object_pose"],
        required_inputs=["metric_geometry"],
        authority=2,
        cost=0.5,
        latency=3,
        correlation_group="pose-metric-geometry",
    )
    profiles.extend([pose_a, pose_b])
    qualifications.extend(
        [_qualification(pose_a, "object_pose"), _qualification(pose_b, "object_pose")]
    )
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    registry = EvidenceMethodAdapterRegistry(
        [
            _FixtureAdapter(pose_a["adapter_reference"], supports_claim=True, finding="pose-valid"),
            _FixtureAdapter(pose_b["adapter_reference"], supports_claim=False, finding="pose-invalid"),
        ]
    )
    # Limit the execution order to the two disagreement steps while preserving
    # a valid plan binding; all other selected steps use unavailable adapters
    # and normalize as unavailable evidence.
    execution = execute_evidence_plan(
        plan, request, testbed, profiles, qualifications, registry=registry
    )
    envelope = build_decision_envelope(
        request,
        testbed,
        plan,
        [result.to_mapping() for result in execution.results],
    ).to_mapping()
    pose_verdict = next(row for row in envelope["per_claim_verdicts"] if row["claim_id"] == "pose")
    assert pose_verdict["verdict"] == "abstention"
    assert pose_verdict["rationale"] == "cross_method_disagreement"
    assert envelope["cross_method_disagreements"][0]["resolution"] == "abstain_and_escalate"


def _physical_outcome(testbed: dict, prediction_digest: str, *, partition: str = "calibration") -> dict:
    from blueprint_pipeline.decision_evidence_contracts import PhysicalOutcomeJoin

    return PhysicalOutcomeJoin.from_mapping(
        {
            "schema_version": "physical_outcome_join.v1",
            "outcome_id": f"outcome-{partition}",
            "testbed_id": testbed["testbed_id"],
            "testbed_version": testbed["version"],
            "testbed_digest": testbed["testbed_digest"],
            "site_id": "fixture-site",
            "task_id": "pick-red-block",
            "scenario_id": "base",
            "condition": {"lighting_lux": [300, 600]},
            "robot_embodiment": {"robot_id": "fixture-arm"},
            "sensors": {"camera": "fixture-rgb-v1"},
            "controller": {"type": "joint_position"},
            "policy_checkpoint": {"policy_id": "policy-a", "digest": SHA_A},
            "evaluator": {
                "evaluator_id": "independent-fixture-evaluator",
                "provider_id": "independent-fixture-evaluator",
            },
            "runtime_provider": {"provider_id": "fixture-method-owner", "runtime": "mujoco"},
            "prediction_digest": prediction_digest,
            "prediction": {
                "sample_id": "physical-sample-1",
                "claim_type": "collision_contact",
                "task_family": "rigid_object_pick_place",
                "predicted_safe": True,
            },
            "observed_outcome": {
                "sample_id": "physical-sample-1",
                "actual_safe": True,
                "qualification_metrics": {
                    "qualification_gate_passed": True,
                    "confidence_intervals": {"level": 0.95, "lower": 0.8, "upper": 1.0},
                    "coverage": 1.0,
                    "abstention_rate": 0.0,
                    "false_safe_rate": 0.0,
                    "false_reject_rate": 0.0,
                },
            },
            "owner_evidence": [{"uri": "fixture://physical-owner-proof", "digest": SHA_D}],
            "timestamps": {"predicted_at": "2026-07-29T10:00:00Z", "observed_at": "2026-07-29T11:00:00Z"},
            "runtime_digest": SHA_B,
            "evaluator_digest": SHA_C,
            "partition": partition,
            "mismatch_taxonomy": {"classification": "match"},
            "provenance": {"owner": "fixture-owner"},
        }
    ).to_mapping()


def test_physical_outcome_join_is_append_only_and_versions_testbed() -> None:
    testbed, _, profiles, _, _, execution, envelope = _execution_fixture()
    prediction_digest = execution.results[0].digest
    original = copy.deepcopy(testbed)
    predictor = next(profile for profile in profiles if profile["method_id"] == "fixture-mujoco")
    update = join_physical_outcome(
        testbed_value=testbed,
        decision_value=envelope.to_mapping(),
        outcome_value=_physical_outcome(testbed, prediction_digest),
        method_profile_value=predictor,
    )
    assert testbed == original
    new_testbed = update.new_testbed.to_mapping()
    assert new_testbed["version"] == "2"
    assert new_testbed["predecessor_testbed_digest"] == testbed["testbed_digest"]
    assert new_testbed["physical_outcome_history_refs"][0]["physical_outcome_digest"] == update.physical_outcome.digest
    assert new_testbed["cross_domain_calibration_transfer_enabled"] is False
    calibration = update.calibration_record.to_mapping()
    assert calibration["calibration_partition"] == "calibration"
    assert calibration["status"] == "qualified"
    assert calibration["provenance"]["cross_domain_transfer_enabled"] is False


def test_physical_outcome_join_rejects_calibration_heldout_leakage() -> None:
    testbed, _, profiles, _, _, execution, envelope = _execution_fixture()
    prediction_digest = execution.results[0].digest
    predictor = next(profile for profile in profiles if profile["method_id"] == "fixture-mujoco")
    calibration = _physical_outcome(testbed, prediction_digest, partition="calibration")
    heldout = _physical_outcome(testbed, prediction_digest, partition="heldout")
    with pytest.raises(ValueError, match="calibration_heldout_leakage:physical-sample-1"):
        join_physical_outcome(
            testbed_value=testbed,
            decision_value=envelope.to_mapping(),
            outcome_value=heldout,
            method_profile_value=predictor,
            existing_outcome_values=[calibration],
        )


def test_checked_in_json_schemas_validate_control_plane_and_webapp_examples() -> None:
    root = Path(__file__).parents[1]
    control_schema = json.loads(
        (root / "docs/schemas/decision_evidence_control_plane.schema.json").read_text()
    )
    testbed = _testbed()
    request = _request(testbed)
    profiles, qualifications = _registry()
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    for artifact in [testbed, *profiles, *qualifications, plan]:
        jsonschema.Draft202012Validator(control_schema).validate(artifact)

    handoff = root / "docs/webapp_handoff/decision-evidence-router.v1"
    request_schema = json.loads((handoff / "request.schema.json").read_text())
    result_schema = json.loads((handoff / "result.schema.json").read_text())
    jsonschema.Draft202012Validator(request_schema).validate(request)
    examples = json.loads((handoff / "examples.json").read_text())["examples"]
    for example in examples.values():
        jsonschema.Draft202012Validator(result_schema).validate(example)
        # Runtime validation additionally checks the checked-in digest.
        from blueprint_pipeline.decision_evidence_contracts import DecisionEnvelope

        assert DecisionEnvelope.from_mapping(example).digest == example["decision_envelope_digest"]
    verification = verify_webapp_handoff(handoff)
    assert verification["status"] == "passed"
    assert verification["artifact_count"] == 9
    assert verification["example_count"] == 3


def test_legacy_wam_cross_check_is_candidate_only_not_qualification() -> None:
    translated = translate_wam_cross_check_plan(
        {
            "schema_version": "wam_classical_sim_cross_check_plan.v1",
            "job_id": "job-legacy-wam",
            "primary_evaluation_substrate": "oscar_wam",
            "recommended_cross_checks": ["classical_sim_mujoco", "classical_sim_isaac"],
            "promotion_effect": "none_without_owner_execution_evidence_and_review",
        }
    )
    assert [row["legacy_method_id"] for row in translated["candidate_methods"]] == [
        "classical_sim_isaac",
        "classical_sim_mujoco",
    ]
    assert translated["qualification_granted"] is False
    assert translated["execution_started"] is False
    assert all(row["candidate_only"] for row in translated["candidate_methods"])
    assert all(not row["availability_asserted"] for row in translated["candidate_methods"])


def test_legacy_wam_scorecard_stays_debug_evidence_and_cannot_name_winner() -> None:
    evidence = wam_scorecard_as_debug_evidence(
        {
            "schema_version": "policy_ranking_scorecard.v1",
            "status": "blocked_inconclusive_ranking",
            "top_policy_id": None,
            "single_best_policy_claimed": False,
            "blockers": ["insufficient_qualified_evidence"],
        }
    )
    assert evidence["status"] == "uncertain"
    assert evidence["categorical_finding"] == "thesis_not_supported"
    assert evidence["claim_ceiling"]["comparative_policy_ranking"] is False
    assert evidence["provenance"]["self_grading_used_for_qualification"] is False
    with pytest.raises(ValueError, match="unproven_winner_forbidden"):
        wam_scorecard_as_debug_evidence(
            {
                "schema_version": "policy_ranking_scorecard.v1",
                "status": "completed",
                "top_policy_id": "policy-a",
                "single_best_policy_claimed": True,
            }
        )


@pytest.mark.parametrize(
    ("translator", "legacy_kind", "claim_type"),
    [
        (
            translate_policy_improvement_request,
            "policy_improvement_run",
            "comparative_policy_ranking",
        ),
        (
            translate_post_training_data_request,
            "post_training_data_package",
            "post_training_evidence_use_eligibility",
        ),
    ],
)
def test_legacy_product_request_translation_is_deprecated_and_provider_neutral(
    translator, legacy_kind: str, claim_type: str
) -> None:
    translated = translator(
        {
            "run_id": "legacy-1",
            "policies": [{"policy_id": "policy-a", "api_key": "must-not-copy"}],
            "selected_provider": "legacy-provider-must-not-copy",
            "budget_usd": 5,
            "caller_identity": "compatibility-test",
        },
        _testbed(),
    )
    request = translated.request.to_mapping()
    assert request["claims"][0]["claim_type"] == claim_type
    assert "selected_provider" not in request
    assert "api_key" not in json.dumps(request)
    assert translated.metadata["legacy_contract_kind"] == legacy_kind
    assert translated.metadata["deprecated"] is True
    assert translated.metadata["replacement_product"] == "Task Evaluation Run"
    assert translated.metadata["translation_grants_qualification"] is False


def test_post_training_is_a_gated_evidence_use_not_a_product_or_improvement_claim() -> None:
    *_, envelope = _execution_fixture()
    allowed = determine_evidence_use(
        envelope.to_mapping(),
        rights={
            "evaluation_use_allowed": True,
            "post_training_use_allowed": True,
            "consent_current": True,
            "revocation_clear": True,
        },
        provenance={"complete": True},
        robot_action_alignment={"aligned": True},
        quality={"gate_passed": True},
        leakage={"heldout_leakage_absent": True},
    )
    assert allowed["evaluation_use"]["allowed"] is True
    assert allowed["post_training_use"]["allowed"] is True
    assert allowed["standalone_product_created"] is False
    assert allowed["training_occurred"] is False
    assert allowed["policy_improved"] is False
    blocked = determine_evidence_use(
        envelope.to_mapping(),
        rights={
            "evaluation_use_allowed": True,
            "post_training_use_allowed": False,
            "consent_current": True,
            "revocation_clear": True,
        },
        provenance={"complete": True},
        robot_action_alignment={"aligned": True},
        quality={"gate_passed": True},
        leakage={"heldout_leakage_absent": True},
    )
    assert blocked["evaluation_use"]["allowed"] is True
    assert blocked["post_training_use"]["allowed"] is False


def test_checked_in_rigid_object_vertical_slice_matches_plan_decision_and_learning() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "tests/fixtures/decision_evidence_rigid_object_v1/vertical_slice.json"
        ).read_text()
    )
    testbed, request, profiles, _, plan, execution, envelope = _execution_fixture()
    assert fixture["testbed"]["testbed_id"] == testbed["testbed_id"]
    assert fixture["testbed"]["version"] == testbed["version"]
    assert fixture["testbed"]["raw_capture_digest"] == testbed["source_capture_bundles"][0]["digest"]
    assert fixture["request"]["claim_types"] == [row["claim_type"] for row in request["claims"]]
    dispositions = {
        row["method_id"]: row["expected_disposition"] for row in fixture["method_cases"]
    }
    claim_plans = {row["claim_id"]: row for row in plan["claim_plans"]}
    assert claim_plans["reach"]["selected_methods"][0]["method_id"] == "analytic-reach"
    assert claim_plans["visible"]["selected_methods"][0]["method_id"] == "captured-visibility"
    assert claim_plans["collision"]["selected_methods"][0]["method_id"] == "fixture-mujoco"
    assert any(
        row["method_id"] == "uncalibrated-world-model"
        and row["qualification_result"] == "rejected"
        and "unqualified_or_out_of_scope" in row["rejection_reasons"]
        for row in claim_plans["ranking"]["candidate_methods_considered"]
    )
    assert dispositions["physical-outcome"] == "bounded_evidence_request"
    assert len(plan["compiled_evaluation_run_specs"]) >= fixture["expected_plan"]["minimum_leaf_evaluation_run_specs"]
    assert len(plan["non_evaluation_run_steps"]) >= fixture["expected_plan"]["minimum_non_leaf_steps"]
    decision = envelope.to_mapping()
    assert decision["overall_outcome"] == fixture["expected_decision"]["overall_outcome"]
    verdicts = {row["claim_id"]: row["verdict"] for row in decision["per_claim_verdicts"]}
    assert sorted(key for key, value in verdicts.items() if value != "abstention") == fixture["expected_decision"]["supported_claim_ids"]
    assert sorted(key for key, value in verdicts.items() if value == "abstention") == fixture["expected_decision"]["abstained_claim_ids"]
    prediction_digest = execution.results[0].digest
    predictor = next(profile for profile in profiles if profile["method_id"] == "fixture-mujoco")
    update = join_physical_outcome(
        testbed_value=testbed,
        decision_value=decision,
        outcome_value=_physical_outcome(testbed, prediction_digest),
        method_profile_value=predictor,
    )
    assert update.new_testbed.to_mapping()["version"] == fixture["later_physical_outcome"]["creates_new_testbed_version"]
    assert update.new_testbed.to_mapping()["cross_domain_calibration_transfer_enabled"] is False


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_cli_plan_execute_aggregate_and_fail_closed_authorization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    testbed, request, profiles, qualifications, _, _, _ = _execution_fixture()
    testbed_path = tmp_path / "testbed.json"
    request_path = tmp_path / "request.json"
    _write_json(testbed_path, testbed)
    _write_json(request_path, request)
    profile_paths = []
    qualification_paths = []
    adapters = []
    for index, profile in enumerate(profiles):
        path = tmp_path / f"profile-{index}.json"
        _write_json(path, profile)
        profile_paths.append(path)
        if profile["method_id"] in {"analytic-reach", "captured-visibility", "fixture-mujoco"}:
            adapters.append(
                {
                    "adapter_reference": profile["adapter_reference"],
                    "result": _FixtureAdapter(
                        profile["adapter_reference"], supports_claim=True, finding="fixture-supports"
                    ).execute(method_profile=profile),
                }
            )
    for index, qualification in enumerate(qualifications):
        path = tmp_path / f"qualification-{index}.json"
        _write_json(path, qualification)
        qualification_paths.append(path)
    plan_dir = tmp_path / "plan"
    plan_args = [
        "plan",
        "--request", str(request_path),
        "--testbed", str(testbed_path),
        "--output-dir", str(plan_dir),
    ]
    for path in profile_paths:
        plan_args.extend(["--method-profile", str(path)])
    for path in qualification_paths:
        plan_args.extend(["--qualification", str(path)])
    assert decision_evidence_cli_main(plan_args) == 0
    assert json.loads(capsys.readouterr().out)["execution_started"] is False
    registry_path = tmp_path / "fixture-adapters.json"
    _write_json(
        registry_path,
        {"schema_version": "evidence_fixture_adapter_registry.v1", "adapters": adapters},
    )
    execute_dir = tmp_path / "execute"
    execute_args = [
        "execute",
        "--plan", str(plan_dir / "evidence_plan.json"),
        "--request", str(request_path),
        "--testbed", str(testbed_path),
        "--fixture-adapter-registry", str(registry_path),
        "--output-dir", str(execute_dir),
    ]
    for path in profile_paths:
        execute_args.extend(["--method-profile", str(path)])
    for path in qualification_paths:
        execute_args.extend(["--qualification", str(path)])
    assert decision_evidence_cli_main(execute_args) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    assert blocked["physical_robot_run_initiated"] is False
    assert decision_evidence_cli_main([*execute_args, "--allow-fixture-adapters"]) == 0
    executed = json.loads(capsys.readouterr().out)
    result_paths = executed["results"]
    assert result_paths
    aggregate_dir = tmp_path / "aggregate"
    aggregate_args = [
        "aggregate",
        "--request", str(request_path),
        "--testbed", str(testbed_path),
        "--plan", str(plan_dir / "evidence_plan.json"),
        "--output-dir", str(aggregate_dir),
    ]
    for path in result_paths:
        aggregate_args.extend(["--result", path])
    assert decision_evidence_cli_main(aggregate_args) == 0
    aggregated = json.loads(capsys.readouterr().out)
    assert aggregated["status"] == "partial_decision"
    assert (aggregate_dir / "decision_envelope.json").is_file()
    result_value = json.loads(Path(result_paths[0]).read_text())
    outcome_path = tmp_path / "physical-outcome.json"
    _write_json(
        outcome_path,
        _physical_outcome(testbed, result_value["result_digest"]),
    )
    predictor_path = profile_paths[
        next(index for index, profile in enumerate(profiles) if profile["method_id"] == "fixture-mujoco")
    ]
    learning_dir = tmp_path / "learning"
    assert decision_evidence_cli_main(
        [
            "ingest-outcome",
            "--testbed", str(testbed_path),
            "--decision", str(aggregate_dir / "decision_envelope.json"),
            "--outcome", str(outcome_path),
            "--method-profile", str(predictor_path),
            "--output-dir", str(learning_dir),
        ]
    ) == 0
    learning = json.loads(capsys.readouterr().out)
    assert learning["status"] == "updated_append_only"
    assert learning["historical_decision_mutated"] is False
    assert learning["cross_domain_transfer_enabled"] is False
