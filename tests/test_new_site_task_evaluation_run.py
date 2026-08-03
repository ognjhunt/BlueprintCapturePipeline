from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.new_site_task_evaluation_run import (
    NewSiteTaskEvaluationError,
    compile_new_site_task_evaluation_run,
    main,
)
from blueprint_pipeline.task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    derive_task_measurement_requirements,
    validate_measurement_qualification,
    validate_method_capability_profile,
    validate_site_evidence_profile,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _finalize(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _source(*, status: str = "admitted_provider_derived_support") -> dict:
    is_blueprint_raw = status == "admitted_blueprint_raw_contract"
    value = {
        "schema_version": "fixture_source_profile.v1",
        "status": status,
        "provider_identity": "fixture-provider",
        "smallest_missing_measurement": (
            None
            if status == "admitted_provider_derived_support"
            else {"code": "camera_intrinsics_missing", "instruction": "Export intrinsics."}
        ),
        "claim_boundary": {
            "provider_derived_support": not is_blueprint_raw,
            "blueprint_raw_contract_truth": is_blueprint_raw,
        },
    }
    return _finalize(value, "source_profile_digest")


def _reconstruction(source_digest: str) -> dict:
    return _finalize(
        {
            "schema_version": "registered_site_reconstruction.v1",
            "source_profile_digest": source_digest,
            "appearance_format": "native_3dgs",
            "appearance_asset_digest": _sha("1"),
            "geometry_asset_digest": _sha("2"),
            "scene_registration_digest": _sha("3"),
            "source_scene_digest": _sha("2"),
            "registration_status": "qualified",
            "full_resolution_appearance_preserved": True,
            "presentation_output_used_as_evaluation_evidence": False,
        },
        "reconstruction_digest",
    )


def _target(
    *,
    reconstruction: dict,
    task_class: str = "rigid_pick_place",
    simready_required: bool = False,
) -> dict:
    return _finalize(
        {
            "schema_version": "fixture_target_orchestration.v1",
            "status": "target_ready_for_bounded_sim",
            "reconstruction_digest": reconstruction["reconstruction_digest"],
            "analysis_appearance_digest": reconstruction["appearance_asset_digest"],
            "selected_target": {
                "proposal_id": "visible-object-001",
                "task_family": "pick_visible_object",
                "task_class": task_class,
                "target_binding_digest": _sha("4"),
                "candidate_self_authorized": False,
            },
            "task_zone_asset_requirement": {
                "verified_simready_asset_required": simready_required,
            },
        },
        "target_orchestration_digest",
    )


def _rendered_target(reconstruction: dict) -> dict:
    selected = {
        "proposal_id": "visible-object-001",
        "object_label": "visible object",
        "task_family": "rigid_pick_place",
        "affordances": ["pick"],
        "visual_confidence": 0.95,
        "status": "authorized_metric_sim_target",
    }
    analysis = _finalize(
        {
            "schema_version": "scene_task_target_analysis_result.v1",
            "status": "target_ready_for_bounded_sim",
            "source_scene_digest": reconstruction["source_scene_digest"],
            "selected_target": selected,
        },
        "target_analysis_digest",
    )
    binding = _finalize(
        {
            "schema_version": "splat_bbox_target_binding_result.v1",
            "status": "candidate_bound",
            "source_scene_digest": reconstruction["source_scene_digest"],
            "analysis_splat_digest": reconstruction["appearance_asset_digest"],
            "proposal_id": selected["proposal_id"],
            "candidate_may_self_authorize": False,
        },
        "binding_evidence_digest",
    )
    task_zone_requirement = _finalize(
        {
            "schema_version": "task_zone_asset_requirement_candidate.v1",
            "status": "not_required_for_inspection_only",
            "target_region_id": selected["proposal_id"],
            "interaction_mode": "inspection_only",
            "interaction_mode_source": "fixture",
            "verified_simready_asset_required": False,
            "authoritative_asset_selection_performed": False,
            "next_stage": "robot_placement",
        },
        "requirement_digest",
    )
    return _finalize(
        {
            "schema_version": "rendered_scene_task_target_orchestration.v1",
            "status": "target_ready_for_bounded_sim",
            "source_scene_digest": reconstruction["source_scene_digest"],
            "analysis_splat_digest": reconstruction["appearance_asset_digest"],
            "candidate_may_self_authorize": False,
            "target_analysis": analysis,
            "binding_results": [
                {
                    "proposal_id": selected["proposal_id"],
                    "status": "candidate_bound",
                    "binding": binding,
                    "blockers": [],
                }
            ],
            "task_zone_asset_requirement": task_zone_requirement,
        },
        "orchestration_digest",
    )


def _placement(target: dict, *, robot_id: str = "franka_panda", status: str = "qualified") -> dict:
    return _finalize(
        {
            "schema_version": "qualified_robot_placement.v1",
            "status": status,
            "robot_id": robot_id,
            "target_binding_digest": target["selected_target"]["target_binding_digest"],
            "pose_site": [0.0, 0.0, 0.0, 0.0],
        },
        "placement_digest",
    )


def _external_placement(*, target_binding_digest: str) -> dict:
    return _finalize(
        {
            "schema_version": "external_scene_robot_placement_candidate.v1",
            "status": "runtime_visualization_candidate_only",
            "robot_id": "franka_panda",
            "target_binding_digest": target_binding_digest,
            "metric_reach_qualified": False,
            "collision_status": "candidate_compiled",
            "candidate_may_self_authorize": False,
            "physical_execution_authorized": False,
            "proof_effect": "external_scene_runtime_robot_visualization_candidate",
        },
        "placement_proposal_digest",
    )


def _composition(*, task_zone_status: str = "not_required") -> dict:
    return _finalize(
        {
            "schema_version": "qualified_scene_composition.v1",
            "floor_support_mount": {"status": "not_required"},
            "task_zone_replacement": (
                {"status": "not_required"}
                if task_zone_status == "not_required"
                else {"status": task_zone_status, "qualification_digest": _sha("5")}
            ),
        },
        "scene_composition_digest",
    )


def _capabilities(method_id: str, *, enabled: set[str]) -> dict:
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
            "container_digest": _sha("a"),
            "solver_backend": method_id,
            "numeric_precision": "float64",
            "deterministic_mode": "strict",
            "operating_system": "linux",
            "gpu_model": "fixture",
            "driver_version": "fixture",
            "random_seed_policy": "frozen",
            "contact_formulation": "fixture",
            "maximum_control_rate_hz": 1000,
            "qualified_parameter_ranges": {},
            "qualified_claim_ceiling": "C3",
            "qualification_expiration": "2027-08-01",
            "harmful_false_negative_bound": 0.01,
            "maximum_latency_class": "interactive",
            "maximum_compute_class": "gpu",
            "estimated_cost_class": "low",
            "data_retention_days": 0,
            "source_available": True,
            "local_offline_supported": True,
            "commercial_use_allowed": True,
            "provider_training_use_allowed": False,
            "deletion_right_supported": True,
            "output_export_supported": True,
            "supported_embodiments": ["franka_panda"],
        }
    )
    for field in enabled:
        values[field] = True
    return values


def _profile(method_id: str, capabilities: set[str]) -> dict:
    return validate_method_capability_profile(
        {
            "schema_version": "method_capability_profile.v1",
            "method_id": method_id,
            "capabilities": _capabilities(method_id, enabled=capabilities),
            "evidence_quality": {
                "source": "independent_fixture",
                "public_research_is_qualification": False,
            },
            "expected_cost_usd": 0.1,
            "expected_latency_seconds": 1.0,
        }
    )


def _qualification(profile: dict, capabilities: set[str], *, error: float) -> dict:
    return validate_measurement_qualification(
        {
            "schema_version": "measurement_qualification_record.v1",
            "qualification_id": f"q-{profile['method_id']}",
            "method_id": profile["method_id"],
            "method_version": "1",
            "capability_profile_digest": profile["capability_profile_digest"],
            "admission_record_digest": _sha("b"),
            "admission_stage": "R7",
            "status": "approved",
            "qualified_capabilities": sorted(capabilities),
            "claim_ceiling": "C3",
            "scope": {
                "task_classes": ["rigid_pick_place"],
                "material_regimes": ["none"],
                "robot_ids": ["franka_panda"],
                "end_effector_ids": [],
                "controller_ids": [],
                "sensor_ids": [],
                "metric_ids": [],
                "parameter_ranges": {},
            },
            "metrics": {
                "physical_accuracy_error": error,
                "uncertainty": error,
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


def _routing_inputs(*, missing_metric_scale: bool = False) -> dict:
    capabilities = {
        "metric_scale_supported",
        "continuous_collision_supported",
        "dynamic_collision_supported",
        "static_friction_supported",
        "dynamic_friction_supported",
        "contact_compliance_supported",
    }
    mujoco = _profile("qualified-mujoco-stack", capabilities)
    isaac = _profile("qualified-isaac-stack", capabilities)
    evidence_ids = {
        "metric_scale",
        "robot_site_registration",
        "validated_collider",
        "mass_inertia",
        "friction_contact",
        "material_parameters",
    }
    site = validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": "new-site-fixture-v1",
            "bundle_id": "capture-fixture",
            "bundle_hash": _sha("c"),
            "provenance_record_id": "provenance-fixture",
            "rights": {"commercial_evaluation_allowed": True},
            "privacy": {"classification": "internal", "external_processing_allowed": False},
            "coordinate_system": {
                "metric_scale_verified": not missing_metric_scale,
                "frame": "site",
            },
            "evidence": {
                evidence_id: {
                    "available": not (missing_metric_scale and evidence_id == "metric_scale"),
                    "validated": not (missing_metric_scale and evidence_id == "metric_scale"),
                    "record_id": f"evidence-{evidence_id}",
                }
                for evidence_id in evidence_ids
            },
            "limitations": {"known_missing_regions": [], "forbidden_claims": []},
        }
    )
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "new-site-task-outcome",
            "claim_type": "collision_contact",
            "material_regimes": ["none"],
            "metric_ids": [],
            "robot_scope": {"robot_id": "franka_panda"},
        },
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    return {
        "requirements": requirements,
        "site_evidence_profile": site,
        "method_capability_profiles": [isaac, mujoco],
        "measurement_qualifications": [
            _qualification(isaac, capabilities, error=0.02),
            _qualification(mujoco, capabilities, error=0.01),
        ],
        "catalog_snapshot_hash": _sha("d"),
        "routing_as_of": "2026-08-03",
    }


def _metric() -> dict:
    return _finalize(
        {
            "schema_version": "task_outcome_metric_spec.v1",
            "metric_id": "target_inspection_quality",
            "units": "score",
            "direction": "maximize",
            "fixed_before_execution": True,
            "frozen_at": "2026-08-03T00:00:00Z",
        },
        "metric_spec_digest",
    )


def _candidate(index: int) -> dict:
    return _finalize(
        {
            "schema_version": "learned_policy_candidate_identity.v1",
            "candidate_id": f"policy-{index}",
            "candidate_kind": "learned_policy",
            "checkpoint_digest": _sha(str(index)),
            "endpoint_identity_digest": None,
            "runtime_digest": _sha("e"),
            "observation_schema_digest": _sha("f"),
            "action_schema_digest": _sha("0"),
            "observation_sequence_spec_digest": _sha("9"),
        },
        "policy_identity_digest",
    )


def _attempt(
    candidate: dict,
    *,
    route_digest: str,
    placement_digest: str,
    metric_digest: str,
    value: float,
    supported: bool = True,
    reset_digest: str | None = None,
) -> dict:
    index = int(candidate["candidate_id"].split("-")[-1])
    return _finalize(
        {
            "schema_version": "learned_policy_attempt_receipt.v1",
            "candidate_id": candidate["candidate_id"],
            "policy_identity_digest": candidate["policy_identity_digest"],
            "status": "completed",
            "execution_receipt_digest": _sha(chr(ord("a") + index)),
            "routing_decision_digest": route_digest,
            "placement_digest": placement_digest,
            "matched_reset_digest": reset_digest or _sha("6"),
            "initial_state_observation_digest": _sha("7"),
            "observation_trace_digest": _sha("8"),
            "action_trace_digest": _sha(str(index)),
            "contact_evidence_digest": _sha("a"),
            "collision_evidence_digest": _sha("b"),
            "action_source": "learned_policy",
            "fresh_policy_query_count": 3,
            "learned_policy_action_count": 3,
            "learned_policy_action_proven": True,
            "reset_observed": True,
            "started_at": "2026-08-03T00:01:00Z",
            "ended_at": "2026-08-03T00:02:00Z",
            "task_metric_result": {
                "metric_spec_digest": metric_digest,
                "value": value,
                "supported_for_ranking": supported,
                "blockers": [] if supported else ["terminal_observation_missing"],
            },
        },
        "attempt_digest",
    )


def _request(*, missing_metric_scale: bool = False) -> dict:
    source = _source()
    reconstruction = _reconstruction(source["source_profile_digest"])
    target = _target(reconstruction=reconstruction)
    placement = _placement(target)
    routing = _routing_inputs(missing_metric_scale=missing_metric_scale)
    routing.update(
        {
            "source_profile_digest": source["source_profile_digest"],
            "target_binding_digest": target["selected_target"]["target_binding_digest"],
            "placement_digest": placement["placement_digest"],
            "robot_id": "franka_panda",
            "task_class": target["selected_target"]["task_class"],
        }
    )
    # Compile once to bind attempts to the deterministic route digest.
    from blueprint_pipeline.task_site_measurement_routing import route_task_site_measurement

    route = route_task_site_measurement(
        routing["requirements"],
        routing["site_evidence_profile"],
        routing["method_capability_profiles"],
        routing["measurement_qualifications"],
        catalog_snapshot_hash=routing["catalog_snapshot_hash"],
    )
    metric = _metric()
    candidates = [_candidate(index) for index in range(1, 6)]
    authorization = _finalize(
        {
            "schema_version": "new_site_policy_execution_authorization.v1",
            "policy_execution_authorized": True,
            "physical_robot_execution_authorized": False,
            "routing_decision_digest": route["routing_decision_digest"],
            "placement_digest": placement["placement_digest"],
            "metric_spec_digest": metric["metric_spec_digest"],
            "candidate_set_digest": canonical_digest(
                {
                    "policy_identity_digests": sorted(
                        row["policy_identity_digest"] for row in candidates
                    )
                }
            ),
        },
        "authorization_digest",
    )
    attempts = [
        _attempt(
            candidate,
            route_digest=route["routing_decision_digest"],
            placement_digest=placement["placement_digest"],
            metric_digest=metric["metric_spec_digest"],
            value=float(index),
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    value = {
        "schema_version": "new_site_task_evaluation_request.v1",
        "run_id": "new-site-run-001",
        "source_profile": source,
        "reconstruction": reconstruction,
        "target_orchestration": target,
        "robot_placement": placement,
        "scene_composition": _composition(),
        "routing_inputs": routing,
        "policy_evaluation": {
            "task_metric": metric,
            "policy_candidates": candidates,
            "attempts": attempts,
        },
        "execution_authorization": authorization,
    }
    value["request_digest"] = canonical_digest(value)
    return value


def _rebind_request(value: dict) -> dict:
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def _rebind_attempt(value: dict) -> dict:
    value["attempt_digest"] = canonical_digest(value, digest_field="attempt_digest")
    return value


def _replace_request_target(request: dict, rendered: dict) -> str:
    binding_digest = rendered["binding_results"][0]["binding"]["binding_evidence_digest"]
    request["target_orchestration"] = rendered
    placement = request["robot_placement"]
    placement["target_binding_digest"] = binding_digest
    _finalize(placement, "placement_digest")
    request["routing_inputs"].update(
        {
            "target_binding_digest": binding_digest,
            "placement_digest": placement["placement_digest"],
            "task_class": "rigid_pick_place",
        }
    )
    request["execution_authorization"]["placement_digest"] = placement["placement_digest"]
    _finalize(request["execution_authorization"], "authorization_digest")
    for attempt in request["policy_evaluation"]["attempts"]:
        attempt["placement_digest"] = placement["placement_digest"]
        _rebind_attempt(attempt)
    _rebind_request(request)
    return binding_digest


def test_complete_run_routes_to_best_qualified_non_default_engine_and_ranks_five() -> None:
    request = _request()
    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    jsonschema.Draft202012Validator(
        json.loads((schema_root / "new_site_task_evaluation_request.v1.schema.json").read_text())
    ).validate(request)
    result = compile_new_site_task_evaluation_run(request)
    jsonschema.Draft202012Validator(
        json.loads((schema_root / "new_site_task_evaluation_run.v1.schema.json").read_text())
    ).validate(result)

    assert result["status"] == "completed"
    assert result["robot_id"] == "franka_panda"
    assert [row["method_id"] for row in result["selected_engine_stack"]] == [
        "qualified-mujoco-stack"
    ]
    assert result["policy_attempt_count"] == 5
    assert result["supported_ranking_candidate_count"] == 5
    assert result["winner_candidate_ids"] == ["policy-5"]
    assert result["winner_candidate_id"] == "policy-5"
    assert result["claim_boundary"]["learned_policy_ranking_proven"] is True
    assert result["claim_boundary"]["controller_ranking_is_learned_policy_ranking"] is False
    assert result["claim_boundary"]["provider_source_is_blueprint_raw_truth"] is False
    assert result["claim_boundary"]["physical_success_proven"] is False


def test_existing_rendered_target_orchestration_is_admitted_without_hand_shaping() -> None:
    request = _request()
    rendered = _rendered_target(request["reconstruction"])
    binding_digest = _replace_request_target(request, rendered)
    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    jsonschema.Draft202012Validator(
        json.loads((schema_root / "new_site_task_evaluation_request.v1.schema.json").read_text())
    ).validate(request)

    result = compile_new_site_task_evaluation_run(request)

    assert result["status"] == "completed"
    assert result["target_orchestration_digest"] == rendered["orchestration_digest"]
    assert result["target_binding_digest"] == binding_digest
    assert result["task_class"] == "rigid_pick_place"


def test_existing_external_placement_candidate_abstains_at_qualification() -> None:
    request = _request()
    placement = _external_placement(target_binding_digest=_sha("4"))
    request["robot_placement"] = placement
    _rebind_request(request)

    result = compile_new_site_task_evaluation_run(request)

    assert result["status"] == "abstained"
    assert result["terminal_stage"] == "robot_placement"
    assert result["smallest_missing_measurement"]["code"] == ("qualified_robot_placement_missing")
    assert result["policy_attempt_count"] == 0


def test_rendered_target_nested_analysis_digest_is_independently_verified() -> None:
    request = _request()
    rendered = _rendered_target(request["reconstruction"])
    rendered["target_analysis"]["selected_target"]["object_label"] = "tampered"
    _finalize(rendered, "orchestration_digest")
    _replace_request_target(request, rendered)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "target_analysis_digest_mismatch" in caught.value.codes


def test_source_abstention_propagates_smallest_measurement() -> None:
    request = _request()
    request["source_profile"] = _source(status="abstained")
    _rebind_request(request)

    result = compile_new_site_task_evaluation_run(request)

    assert result["status"] == "abstained"
    assert result["terminal_stage"] == "capture_source_admission"
    assert result["smallest_missing_measurement"]["code"] == "camera_intrinsics_missing"


def test_blueprint_raw_source_cannot_use_provider_derived_truth_boundary() -> None:
    request = _request()
    source = _source(status="admitted_blueprint_raw_contract")
    source["claim_boundary"] = {
        "provider_derived_support": True,
        "blueprint_raw_contract_truth": False,
    }
    _finalize(source, "source_profile_digest")
    request["source_profile"] = source
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "blueprint_raw_source_truth_boundary_invalid" in caught.value.codes


def test_metric_scale_gap_abstains_before_policy_execution() -> None:
    request = _request(missing_metric_scale=True)

    result = compile_new_site_task_evaluation_run(request)

    assert result["status"] == "abstained"
    assert result["terminal_stage"] == "task_site_engine_routing"
    assert result["policy_attempt_count"] == 0
    assert result["routing_decision"]["status"] == "abstention"
    assert result["claim_boundary"]["learned_policy_attempts_observed"] is False


def test_exactly_five_learned_candidates_required() -> None:
    request = _request()
    request["policy_evaluation"]["policy_candidates"].pop()
    request["policy_evaluation"]["attempts"].pop()
    _rebind_request(request)

    result = compile_new_site_task_evaluation_run(request)

    assert result["status"] == "abstained"
    assert result["smallest_missing_measurement"]["code"] == (
        "exactly_five_learned_policy_attempts_required"
    )
    assert result["policy_attempt_count"] == 4


def test_scripted_controller_cannot_impersonate_learned_policy() -> None:
    request = _request()
    candidate = request["policy_evaluation"]["policy_candidates"][0]
    candidate["candidate_kind"] = "scripted_controller"
    _finalize(candidate, "policy_identity_digest")
    request["policy_evaluation"]["attempts"][0]["policy_identity_digest"] = candidate[
        "policy_identity_digest"
    ]
    _rebind_attempt(request["policy_evaluation"]["attempts"][0])
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "scripted_controller_candidate_forbidden" in caught.value.codes


def test_same_checkpoint_cannot_impersonate_two_policy_candidates() -> None:
    request = _request()
    candidates = request["policy_evaluation"]["policy_candidates"]
    candidates[1]["checkpoint_digest"] = candidates[0]["checkpoint_digest"]
    _finalize(candidates[1], "policy_identity_digest")
    request["policy_evaluation"]["attempts"][1]["policy_identity_digest"] = candidates[1][
        "policy_identity_digest"
    ]
    _rebind_attempt(request["policy_evaluation"]["attempts"][1])
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "policy_candidate_immutable_identity_duplicate" in caught.value.codes


@pytest.mark.parametrize(
    "field",
    ["fresh_policy_query_count", "learned_policy_action_count"],
)
def test_boolean_cannot_impersonate_positive_policy_execution_count(field: str) -> None:
    request = _request()
    attempt = request["policy_evaluation"]["attempts"][0]
    attempt[field] = True
    _rebind_attempt(attempt)
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "policy_attempt_real_execution_not_proven" in caught.value.codes


def test_mismatched_reset_blocks_comparative_ranking() -> None:
    request = _request()
    attempt = request["policy_evaluation"]["attempts"][4]
    attempt["matched_reset_digest"] = _sha("c")
    _rebind_attempt(attempt)
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "policy_attempt_matched_reset_mismatch" in caught.value.codes


def test_all_five_attempts_run_but_unsupported_candidate_is_not_ranked() -> None:
    request = _request()
    attempt = request["policy_evaluation"]["attempts"][2]
    attempt["task_metric_result"]["supported_for_ranking"] = False
    attempt["task_metric_result"]["blockers"] = ["terminal_observation_missing"]
    _rebind_attempt(attempt)
    _rebind_request(request)

    result = compile_new_site_task_evaluation_run(request)

    assert result["policy_attempt_count"] == 5
    assert result["supported_ranking_candidate_count"] == 4
    assert result["unsupported_policy_candidate_ids"] == ["policy-3"]
    assert "policy-3" not in {row["candidate_id"] for row in result["ranking"]}


def test_tied_first_place_reports_shared_winners_without_sole_winner() -> None:
    request = _request()
    attempt = request["policy_evaluation"]["attempts"][3]
    attempt["task_metric_result"]["value"] = 5.0
    _rebind_attempt(attempt)
    _rebind_request(request)

    result = compile_new_site_task_evaluation_run(request)
    result_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "new_site_task_evaluation_run.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(result_schema).validate(result)

    assert result["winner_candidate_ids"] == ["policy-4", "policy-5"]
    assert result["winner_candidate_id"] is None
    assert {row["candidate_id"] for row in result["ranking"] if row["rank"] == 1} == {
        "policy-4",
        "policy-5",
    }


def test_simready_task_zone_is_required_only_for_interaction_semantics() -> None:
    request = _request()
    request["target_orchestration"] = _target(
        reconstruction=request["reconstruction"], simready_required=True
    )
    request["robot_placement"] = _placement(request["target_orchestration"])
    placement_digest = request["robot_placement"]["placement_digest"]
    target_digest = request["target_orchestration"]["selected_target"]["target_binding_digest"]
    for attempt in request["policy_evaluation"]["attempts"]:
        attempt["placement_digest"] = placement_digest
        _rebind_attempt(attempt)
    assert target_digest == _sha("4")
    _rebind_request(request)

    result = compile_new_site_task_evaluation_run(request)

    assert result["status"] == "abstained"
    assert result["smallest_missing_measurement"]["code"] == "qualified_simready_task_zone_missing"


def test_humanoid_task_uses_g1_and_rejects_franka_placement() -> None:
    request = _request()
    request["target_orchestration"] = _target(
        reconstruction=request["reconstruction"], task_class="humanoid_locomotion"
    )
    request["robot_placement"] = _placement(request["target_orchestration"])
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "robot_default_binding_mismatch" in caught.value.codes


def test_cli_writes_immutable_abstention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    request.pop("policy_evaluation")
    _rebind_request(request)
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["new-site-task-eval", "--request", str(request_path), "--output", str(output_path)],
    )

    assert main() == 2
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["smallest_missing_measurement"]["code"] == "five_learned_policy_attempts_missing"
    assert main() == 2


def test_request_digest_drift_is_rejected() -> None:
    request = _request()
    request["run_id"] = "changed-after-signing"

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "new_site_request_digest_mismatch" in caught.value.codes


def test_route_scope_must_bind_exact_source_target_placement_robot_and_task() -> None:
    request = _request()
    request["routing_inputs"]["target_binding_digest"] = _sha("e")
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "task_site_engine_route_scope_mismatch" in caught.value.codes


def test_target_orchestration_must_bind_registered_native_appearance() -> None:
    request = _request()
    target = request["target_orchestration"]
    target["analysis_appearance_digest"] = _sha("e")
    _finalize(target, "target_orchestration_digest")
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "target_orchestration_reconstruction_mismatch" in caught.value.codes


def test_execution_authorization_binds_exact_five_candidate_set() -> None:
    request = _request()
    authorization = request["execution_authorization"]
    authorization["candidate_set_digest"] = _sha("e")
    _finalize(authorization, "authorization_digest")
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "policy_authorization_candidate_set_mismatch" in caught.value.codes


def test_frozen_metric_must_precede_every_attempt() -> None:
    request = _request()
    metric = request["policy_evaluation"]["task_metric"]
    metric["frozen_at"] = "2026-08-03T00:01:30Z"
    _finalize(metric, "metric_spec_digest")
    for attempt in request["policy_evaluation"]["attempts"]:
        attempt["task_metric_result"]["metric_spec_digest"] = metric["metric_spec_digest"]
        _rebind_attempt(attempt)
    authorization = request["execution_authorization"]
    authorization["metric_spec_digest"] = metric["metric_spec_digest"]
    _finalize(authorization, "authorization_digest")
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "task_metric_not_frozen_before_attempt" in caught.value.codes


def test_g1_cannot_be_requested_for_non_humanoid_task() -> None:
    request = _request()
    target = request["target_orchestration"]
    target["selected_target"]["required_embodiment"] = "unitree_g1"
    _finalize(target, "target_orchestration_digest")
    request["robot_placement"] = _placement(target, robot_id="unitree_g1")
    _rebind_request(request)

    with pytest.raises(NewSiteTaskEvaluationError) as caught:
        compile_new_site_task_evaluation_run(request)

    assert "g1_requires_humanoid_task" in caught.value.codes
