from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from blueprint_pipeline.policy_evaluation_contracts import (
    MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION,
    POLICY_ADAPTER_SCHEMA_VERSION,
    POLICY_EVALUATION_DESIGN_SCHEMA_VERSION,
    validate_policy_adapter_manifest,
    validate_policy_evaluation_design,
)
from blueprint_pipeline.evaluator_evidence_profiles import (
    COMMON_DIGEST_FIELDS,
    EVALUATOR_EVIDENCE_PROFILES,
    required_evaluator_evidence_digest_fields,
    validate_evaluator_evidence,
)
from blueprint_pipeline.decision_grade_ranking import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_REPLICATES,
    SCHEMA_VERSION as RANKING_REQUEST_SCHEMA_VERSION,
    build_decision_grade_ranking,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _payload_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy(index: int) -> dict:
    return {
        "schema_version": POLICY_ADAPTER_SCHEMA_VERSION,
        "policy_id": f"policy-{index}",
        "checkpoint_id": f"checkpoint-{index}",
        "policy_family": "groot" if index == 0 else f"family-{index}",
        "embodiment_id": "unitree-g1" if index == 0 else f"robot-{index}",
        "version": "1.0.0",
        "policy_sha256": _digest(100 + index),
        "checkpoint_sha256": _digest(200 + index),
        "adapter_code_sha256": _digest(300 + index),
        "embodiment_manifest_sha256": _digest(400 + index),
        "qualification_fixture": "g1_kitchen" if index == 0 else None,
        "action_contract": {
            "dimension": 2,
            "units": ["rad", "rad"],
            "bounds": [
                {"minimum": -1.0, "maximum": 1.0},
                {"minimum": -1.0, "maximum": 1.0},
            ],
            "control_rate_hz": 50.0,
            "timestamp_semantics": "monotonic_chunk_start_and_per_sample_offsets",
            "normalization_manifest_sha256": _digest(500 + index),
            "missing_action_behavior": "block",
            "out_of_bounds_behavior": "block",
        },
    }


def _design() -> dict:
    policies = [_policy(index) for index in range(7)]
    rows = []
    for policy_index, policy in enumerate(policies):
        for seed in range(MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION):
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "site_id": "site-1",
                    "task_id": "open-door",
                    "condition_id": "condition-1",
                    "seed": seed,
                    "site_task_condition_seed_manifest_sha256": _digest(550 + seed),
                    "observation_sha256": _digest(600 + seed),
                    "commanded_action_chunk_sha256": _digest(800 + policy_index * 100 + seed),
                    "policy_runtime_output_sha256": _digest(1600 + policy_index * 100 + seed),
                    "initial_condition_sha256": _digest(600 + seed),
                    "evaluator_profile_manifest_sha256": _digest(2300),
                    "evaluator_backend_manifest_sha256": _digest(2350),
                    "evaluator_backend": {
                        "schema_version": "evaluator_backend_manifest.v1",
                        "backend_id": "cosmos-3-evaluator-adapter",
                        "model_family": "cosmos",
                        "model_version": "3",
                        "adapter_version": "1.0.0",
                        "backend_kind": "world_model",
                        "execution_interface": "provider_worker",
                        "model_artifact_sha256": _digest(2600),
                        "adapter_code_sha256": _digest(2351),
                        "runtime_manifest_sha256": _digest(2352),
                        "license_manifest_sha256": _digest(2353),
                        "backend_is_compute_provider": False,
                    },
                    "evaluator_request_sha256": _digest(2400 + policy_index * 100 + seed),
                    "evaluator_checkpoint_sha256": _digest(2600),
                    "model_output_sha256": _digest(2700 + policy_index * 100 + seed),
                    "provider_execution_sha256": _digest(3500 + policy_index * 100 + seed),
                    "next_policy_query_sha256": _digest(4300 + policy_index * 100 + seed),
                    "action_control_suite_sha256": _digest(5100 + policy_index * 100 + seed),
                    "criterion_result_sha256": _digest(5900 + policy_index * 100 + seed),
                    "authoritative_manifest_sha256": _digest(6700 + policy_index * 100 + seed),
                    "evaluator_profile_id": "generic_evaluator_bounded_v1",
                    "fresh_evaluator_model_execution_proven": True,
                    "fresh_evaluator_model_run_steps": 1,
                    "action_control_suite_status": "passed",
                    "authoritative_manifest_status": "completed",
                    "infrastructure_status": "succeeded",
                    "evaluator_outcome_status": "valid",
                    "criterion_result_status": "valid",
                    "evaluator_identity_is_compute_provider": False,
                    "generic_evaluator_contract_status": "validated",
                    "missing_action": False,
                    "zero_action_substitute_used": False,
                    "scripted_target_motion_used": False,
                    "fallback_policy_used": False,
                    "fixture_or_proxy_model_output_used": False,
                    "policy_specific_scenario_change_used": False,
                    "hidden_shared_state_used": False,
                }
            )
    return {
        "schema_version": POLICY_EVALUATION_DESIGN_SCHEMA_VERSION,
        "policies": policies,
        "rows": rows,
        "hidden_shared_state_prohibited": True,
        "policy_specific_scenario_changes_prohibited": True,
        "minimum_matched_replicates_per_policy_condition": (
            MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION
        ),
        "direct_sc3_comparison_requested": False,
    }


def test_policy_adapter_requires_exact_action_semantics() -> None:
    assert validate_policy_adapter_manifest(_policy(0))["status"] == "validated"

    invalid = _policy(0)
    invalid["action_contract"]["units"] = ["rad"]
    invalid["action_contract"]["missing_action_behavior"] = "zero_fill"
    result = validate_policy_adapter_manifest(invalid)

    assert result["status"] == "blocked"
    assert "policy_action_units_missing_or_dimension_mismatch" in result["blockers"]
    assert "policy_missing_action_behavior_must_block" in result["blockers"]

    malformed_bounds = _policy(0)
    malformed_bounds["action_contract"]["bounds"].append("corrupt-bound")
    assert (
        "policy_action_bounds_payload_invalid"
        in validate_policy_adapter_manifest(malformed_bounds)["blockers"]
    )


def test_generic_policy_design_admits_seven_independent_matched_policies() -> None:
    result = validate_policy_evaluation_design(_design())

    assert result["schema_version"] == "policy_evaluation_design_validation.v2"
    assert result["status"] == "decision_grade"
    assert result["policy_count"] == 7
    assert result["independent_checkpoint_count"] == 7
    assert result["g1_kitchen_fixture_present"] is True
    assert result["g1_kitchen_is_product_architecture"] is False
    assert result["evaluator_profile_ids"] == ["generic_evaluator_bounded_v1"]
    assert result["evaluator_families"] == ["generic_evaluator_bounded"]
    assert result["evaluator_backend_ids"] == ["cosmos-3-evaluator-adapter"]
    assert result["evaluator_model_families"] == ["cosmos"]


def test_legacy_policy_design_schema_does_not_silently_enter_v2_contract() -> None:
    candidate = _design()
    candidate["schema_version"] = "policy_evaluation_design.v1"

    result = validate_policy_evaluation_design(candidate)

    assert result["status"] == "blocked"
    assert "policy_evaluation_design_schema_missing_or_unsupported" in result["blockers"]


def test_policy_design_normalizes_checkpoint_digests_before_independence_count() -> None:
    candidate = _design()
    candidate["policies"][1]["checkpoint_sha256"] = (
        "sha256:" + candidate["policies"][0]["checkpoint_sha256"]
    )

    result = validate_policy_evaluation_design(candidate)

    assert result["status"] == "blocked"
    assert result["independent_checkpoint_count"] == 6
    assert "independent_checkpoint_count_lt_7" in result["blockers"]


def test_policy_design_rejects_asymmetric_cells_and_fallback_injection() -> None:
    candidate = _design()
    candidate["rows"].pop()
    candidate["rows"][0]["fallback_policy_used"] = True

    result = validate_policy_evaluation_design(candidate)

    assert result["status"] == "blocked"
    assert any(
        blocker.startswith("asymmetric_matched_cell_coverage:") for blocker in result["blockers"]
    )
    assert "decision_grade_row_forbidden_or_unproven:0:fallback_policy_used" in result["blockers"]


def test_policy_design_rejects_malformed_registry_and_row_entries() -> None:
    candidate = _design()
    candidate["policies"].append("corrupt-policy")
    candidate["rows"].append("corrupt-row")

    result = validate_policy_evaluation_design(candidate)

    assert result["status"] == "blocked"
    assert "policy_registry_payload_invalid" in result["blockers"]
    assert "evaluation_rows_payload_invalid" in result["blockers"]


def test_policy_design_requires_identical_matched_cell_bindings_across_policies() -> None:
    candidate = _design()
    row = next(
        item for item in candidate["rows"] if item["policy_id"] == "policy-1" and item["seed"] == 0
    )
    row["initial_condition_sha256"] = _digest(9999)
    row["evaluator_profile_manifest_sha256"] = _digest(9998)

    result = validate_policy_evaluation_design(candidate)

    assert result["status"] == "blocked"
    assert (
        "matched_cell_binding_mismatch:site-1:open-door:condition-1:0:initial_condition_sha256"
        in result["blockers"]
    )
    assert (
        "matched_cell_binding_mismatch:site-1:open-door:condition-1:0:evaluator_profile_manifest_sha256"
        in result["blockers"]
    )


def test_policy_design_requires_fresh_profiled_evaluator_chain_and_passed_controls() -> None:
    candidate = _design()
    row = candidate["rows"][0]
    row["fresh_evaluator_model_run_steps"] = 0
    row["fresh_evaluator_model_execution_proven"] = False
    row["action_control_suite_status"] = "blocked"
    row.pop("next_policy_query_sha256")

    result = validate_policy_evaluation_design(candidate)

    assert result["status"] == "blocked"
    assert (
        "evaluation_row_evaluator_evidence:0:fresh_evaluator_model_execution_not_proven"
        in result["blockers"]
    )
    assert (
        "evaluation_row_evaluator_evidence:0:fresh_evaluator_model_run_steps_missing_or_invalid"
        in result["blockers"]
    )
    assert (
        "evaluation_row_evaluator_evidence:0:action_control_suite_not_passed" in result["blockers"]
    )
    assert (
        "evaluation_row_evaluator_evidence:0:evaluator_evidence_digest_missing_or_invalid:next_policy_query_sha256"
        in result["blockers"]
    )


def test_evaluator_profiles_keep_generic_oscar_and_sc3_requirements_separate() -> None:
    generic = _design()["rows"][0]
    assert validate_evaluator_evidence(generic)["status"] == "validated"
    assert set(EVALUATOR_EVIDENCE_PROFILES) == {
        "generic_evaluator_bounded_v1",
        "oscar_roboarena_v2",
        "sc3_eval_v3",
    }
    assert required_evaluator_evidence_digest_fields("generic_evaluator_bounded_v1") == (
        COMMON_DIGEST_FIELDS
    )
    assert "fk_result_sha256" in required_evaluator_evidence_digest_fields("oscar_roboarena_v2")
    assert "synchronized_multiview_manifest_sha256" in (
        required_evaluator_evidence_digest_fields("sc3_eval_v3")
    )

    alternate_backend = deepcopy(generic)
    alternate_backend["evaluator_backend_manifest_sha256"] = _digest(7990)
    alternate_backend["evaluator_checkpoint_sha256"] = _digest(7991)
    alternate_backend["evaluator_backend"].update(
        {
            "backend_id": "future-world-model-adapter",
            "model_family": "future-world-model",
            "model_version": "1",
            "model_artifact_sha256": _digest(7991),
        }
    )
    alternate_result = validate_evaluator_evidence(alternate_backend)
    assert alternate_result["status"] == "validated"
    assert alternate_result["evaluator_model_family"] == "future-world-model"

    blocked_manifest = deepcopy(generic)
    blocked_manifest["generated_episode_results_status"] = "completed"
    blocked_manifest["authoritative_manifest_status"] = "blocked"
    assert (
        "authoritative_manifest_not_completed"
        in validate_evaluator_evidence(blocked_manifest)["blockers"]
    )

    oscar = deepcopy(generic)
    oscar.update(
        {
            "evaluator_profile_id": "oscar_roboarena_v2",
            "official_runtime_contract_sha256": _digest(8000),
            "fk_result_sha256": _digest(8001),
            "camera_projection_sha256": _digest(8002),
            "skeleton_conditioning_sha256": _digest(8003),
            "official_runtime_contract_status": "validated",
            "fk_status": "passed",
            "camera_projection_status": "passed",
            "skeleton_validation_status": "passed",
        }
    )
    assert validate_evaluator_evidence(oscar)["status"] == "validated"
    oscar.pop("fk_result_sha256")
    assert (
        "evaluator_evidence_digest_missing_or_invalid:fk_result_sha256"
        in validate_evaluator_evidence(oscar)["blockers"]
    )

    sc3 = deepcopy(generic)
    sc3.update(
        {
            "evaluator_profile_id": "sc3_eval_v3",
            "synchronized_multiview_manifest_sha256": _digest(8100),
            "recovered_inverse_actions_sha256": _digest(8101),
            "per_chunk_error_sha256": _digest(8102),
            "inverse_calibration_set_sha256": _digest(8103),
            "strict_scorer_request_status": "validated",
            "multiview_consistency_status": "passed",
            "inverse_action_recovery_status": "passed",
            "termination_chunk_index": 3,
            "inverse_error_threshold": 0.05,
            "recovered_inverse_action_dimensions": [
                {"dimension": 0, "unit": "rad", "maximum_error": 0.01}
            ],
        }
    )
    assert validate_evaluator_evidence(sc3)["status"] == "validated"
    malformed_sc3 = deepcopy(sc3)
    malformed_sc3["recovered_inverse_action_dimensions"].append("corrupt-dimension")
    assert (
        "sc3_recovered_inverse_action_dimensions_payload_invalid"
        in validate_evaluator_evidence(malformed_sc3)["blockers"]
    )
    invalid_sc3_dimensions = deepcopy(sc3)
    invalid_sc3_dimensions["recovered_inverse_action_dimensions"] = [
        {"dimension": 0, "unit": "rad", "maximum_error": 0.01},
        {"dimension": 0, "unit": "", "maximum_error": 0.06},
    ]
    invalid_sc3_blockers = validate_evaluator_evidence(invalid_sc3_dimensions)["blockers"]
    assert "sc3_recovered_inverse_action_dimension_invalid:1" in invalid_sc3_blockers
    assert "sc3_recovered_inverse_action_unit_missing:1" in invalid_sc3_blockers
    assert "sc3_recovered_inverse_action_error_exceeds_threshold:1" in invalid_sc3_blockers
    sc3["evaluator_outcome_status"] = "abstained"
    sc3["criterion_result_status"] = "abstained"
    assert (
        "sc3_abstention_requires_inverse_recovery_abstention"
        in validate_evaluator_evidence(sc3)["blockers"]
    )


def test_evaluator_profile_does_not_default_to_sc3_or_oscar() -> None:
    row = deepcopy(_design()["rows"][0])
    row.pop("evaluator_profile_id")

    result = validate_evaluator_evidence(row)

    assert result["status"] == "blocked"
    assert result["blockers"] == ["evaluator_profile_missing_or_unsupported"]


def test_direct_sc3_comparison_requires_36_or_37_replicates() -> None:
    candidate = deepcopy(_design())
    candidate["direct_sc3_comparison_requested"] = True

    result = validate_policy_evaluation_design(candidate)

    assert result["decision_grade_eligible"] is False
    assert "direct_sc3_comparison_requires_36_or_37_matched_replicates" in result["blockers"]
    assert "direct_sc3_comparison_requires_sc3_evaluator_profile" in result["blockers"]


def _ranking_request() -> dict:
    design = _design()
    episode_results = []
    for row in design["rows"]:
        policy_index = int(row["policy_id"].split("-")[-1])
        success = row["seed"] < 14 + (policy_index % 3)
        criterion_results = [
            {
                "criterion_id": "door-angle",
                "outcome": "success" if success else "failure",
                "confidence": 0.95,
                "label_blinded_and_randomized": True,
                "evidence_refs": [{"sha256": "sha256:" + _digest(4000 + row["seed"])}],
                "failure_taxonomy": [] if success else ["criterion_not_reached"],
            }
        ]
        row["criterion_result_sha256"] = _payload_digest(criterion_results)
        episode_results.append(
            {
                "policy_id": row["policy_id"],
                "site_id": row["site_id"],
                "task_id": row["task_id"],
                "condition_id": row["condition_id"],
                "seed": row["seed"],
                "full_ordered_episode_evidence": True,
                "episode_evidence_sha256": "sha256:" + _digest(3000 + row["seed"]),
                "artifact_freshness_status": "current",
                "evaluator_profile_id": row["evaluator_profile_id"],
                "evaluator_backend_id": row["evaluator_backend"]["backend_id"],
                "fresh_evaluator_model_execution_proven": True,
                "fresh_evaluator_model_run_steps": row["fresh_evaluator_model_run_steps"],
                "authoritative_manifest_status": "completed",
                "infrastructure_status": "succeeded",
                "evaluator_outcome_status": "valid",
                "fixture_or_proxy_model_output_used": False,
                "fallback_policy_used": False,
                **{field: row[field] for field in COMMON_DIGEST_FIELDS},
                "criterion_results": criterion_results,
            }
        )
    preferences = [
        {
            "policy_a": f"policy-{index}",
            "policy_b": f"policy-{index + 1}",
            "outcome": "policy_b",
            "label_blinded_and_randomized": True,
            "evidence_refs": [{"sha256": "sha256:" + _digest(6000 + index)}],
        }
        for index in range(6)
    ]
    return {
        "schema_version": RANKING_REQUEST_SCHEMA_VERSION,
        "evaluation_design": design,
        "minimum_calibrated_judge_confidence": 0.8,
        "judge_calibration_set_sha256": "sha256:" + _digest(5000),
        "judge_calibration_status": "accepted",
        "label_authority_independent_of_policy_and_model": True,
        "episode_results": episode_results,
        "pairwise_preferences": preferences,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": 1729,
            "replicate_count": BOOTSTRAP_REPLICATES,
        },
        "ood_axis_results": [
            {
                "axis": axis,
                "coverage": 0.9,
                "abstention_rate": 0.1,
                "sample_count": 20,
                "coverage_95_ci": [0.8, 0.97],
                "abstention_95_ci": [0.03, 0.2],
                "failure_taxonomy": {"criterion_not_reached": 2},
                "split_manifest_sha256": "sha256:" + _digest(7000 + index),
            }
            for index, axis in enumerate(("site", "task", "embodiment", "viewpoint", "appearance"))
        ],
        "accepted_external_anchor_rows": [],
    }


def test_decision_grade_ranking_keeps_correlation_unmeasured_without_real_anchors() -> None:
    result = build_decision_grade_ranking(_ranking_request())

    assert result["schema_version"] == "decision_grade_ranking.v2"
    assert result["status"] == "decision_grade"
    assert result["bradley_terry"]["graph_connected"] is True
    assert len(result["bradley_terry"]["ranking"]) == 7
    assert result["bootstrap"]["replicate_count"] == 10_000
    assert result["bootstrap"]["matched_cells_resampled_jointly_across_policies"] is True
    assert result["correlation_status"] == "correlation_not_measured"
    assert result["pearson"] is None
    assert result["spearman"] is None
    assert result["mmrv"] is None


def test_legacy_ranking_request_schema_does_not_silently_enter_v2_contract() -> None:
    candidate = _ranking_request()
    candidate["schema_version"] = "decision_grade_ranking_request.v1"

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "decision_grade_ranking_schema_missing_or_unsupported" in result["blockers"]


def test_decision_grade_ranking_rejects_low_confidence_silent_failure_and_disconnected_graph() -> (
    None
):
    candidate = _ranking_request()
    candidate["episode_results"][0]["criterion_results"][0]["confidence"] = 0.2
    candidate["episode_results"][0]["criterion_results"][0]["outcome"] = "failure"
    candidate["episode_results"][0]["criterion_results"][0]["failure_taxonomy"] = "invalid"
    criterion_digest = _payload_digest(candidate["episode_results"][0]["criterion_results"])
    candidate["episode_results"][0]["criterion_result_sha256"] = criterion_digest
    candidate["evaluation_design"]["rows"][0]["criterion_result_sha256"] = criterion_digest
    candidate["pairwise_preferences"] = candidate["pairwise_preferences"][:1]

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "low_confidence_criterion_must_abstain:0:0" in result["blockers"]
    assert "criterion_failure_taxonomy_payload_invalid:0:0" in result["blockers"]
    assert "bradley_terry_preference_graph_not_connected" in result["blockers"]


def test_decision_grade_ranking_blocks_all_abstain_policy_conditions() -> None:
    candidate = _ranking_request()
    for row in candidate["episode_results"]:
        for criterion in row["criterion_results"]:
            criterion["outcome"] = "abstain"

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert all(row["coverage"] == 0.0 for row in result["policy_scorecards"])
    assert any(
        blocker.startswith("decided_outcome_count_below_minimum:") for blocker in result["blockers"]
    )

    partial = _ranking_request()
    partial["episode_results"][0]["criterion_results"][0]["outcome"] = "abstain"
    partial_result = build_decision_grade_ranking(partial)
    assert partial_result["status"] == "blocked"
    assert any(
        blocker.endswith(":19<20")
        for blocker in partial_result["blockers"]
        if blocker.startswith("decided_outcome_count_below_minimum:")
    )


def test_decision_grade_ranking_keeps_model_abstention_separate_from_decided_label() -> None:
    candidate = _ranking_request()
    candidate["episode_results"][0]["evaluator_outcome_status"] = "abstained"

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "episode_result_evaluator_outcome_mismatch:0" in result["blockers"]
    assert "abstained_evaluator_cannot_emit_decided_criterion:0:0" in result["blockers"]
    assert any(
        blocker.endswith(":19<20")
        for blocker in result["blockers"]
        if blocker.startswith("decided_outcome_count_below_minimum:")
    )


def test_decision_grade_ranking_rejects_weak_ood_reporting() -> None:
    candidate = _ranking_request()
    candidate["ood_axis_results"][0].pop("coverage_95_ci")
    candidate["ood_axis_results"][0]["failure_taxonomy"] = []

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "ood_axis_coverage_ci_missing_or_invalid:0" in result["blockers"]
    assert "ood_axis_failure_taxonomy_missing:0" in result["blockers"]


def test_decision_grade_ranking_rejects_stale_forged_and_fallback_evidence() -> None:
    candidate = _ranking_request()
    row = candidate["episode_results"][0]
    row["artifact_freshness_status"] = "stale"
    row["fallback_policy_used"] = True
    row["model_output_sha256"] = "sha256:not-a-digest"
    row["provider_execution_sha256"] = "sha256:" + _digest(99999)
    row.pop("policy_runtime_output_sha256")
    row.pop("fresh_evaluator_model_run_steps")
    candidate["episode_results"][1]["fresh_evaluator_model_run_steps"] = 2
    row["criterion_results"][0]["evidence_refs"] = [{"sha256": "forged"}]

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "episode_result_artifact_not_current:0" in result["blockers"]
    assert "episode_result_fallback_policy_not_blocked:0" in result["blockers"]
    assert "episode_result_chain_digest_missing:0:model_output_sha256" in result["blockers"]
    assert "episode_result_chain_digest_mismatch:0:provider_execution_sha256" in result["blockers"]
    assert (
        "episode_result_chain_digest_missing:0:policy_runtime_output_sha256" in result["blockers"]
    )
    assert "episode_result_fresh_evaluator_steps_invalid:0" in result["blockers"]
    assert "episode_result_fresh_evaluator_steps_mismatch:1" in result["blockers"]
    assert "criterion_evidence_digest_invalid:0:0" in result["blockers"]
    assert "criterion_result_payload_digest_mismatch:0" in result["blockers"]


def test_decision_grade_ranking_rejects_non_mapping_criterion_payload_entries() -> None:
    candidate = _ranking_request()
    raw_criteria = candidate["episode_results"][0]["criterion_results"]
    raw_criteria.append("corrupt-extra-criterion")
    digest = _payload_digest(raw_criteria)
    candidate["episode_results"][0]["criterion_result_sha256"] = digest
    candidate["evaluation_design"]["rows"][0]["criterion_result_sha256"] = digest

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "criterion_results_payload_invalid:0" in result["blockers"]


def test_decision_grade_ranking_rejects_non_mapping_criterion_evidence_entries() -> None:
    candidate = _ranking_request()
    criterion = candidate["episode_results"][0]["criterion_results"][0]
    criterion["evidence_refs"].append("corrupt-evidence-reference")
    criteria = candidate["episode_results"][0]["criterion_results"]
    digest = _payload_digest(criteria)
    candidate["episode_results"][0]["criterion_result_sha256"] = digest
    candidate["evaluation_design"]["rows"][0]["criterion_result_sha256"] = digest

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "criterion_evidence_payload_invalid:0:0" in result["blockers"]


def test_decision_grade_ranking_rejects_malformed_top_level_row_collections() -> None:
    candidate = _ranking_request()
    candidate["episode_results"].append("corrupt-episode")
    candidate["pairwise_preferences"].append("corrupt-preference")
    candidate["ood_axis_results"].append("corrupt-ood-row")
    candidate["accepted_external_anchor_rows"].append("corrupt-anchor")

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "episode_results_payload_invalid" in result["blockers"]
    assert "pairwise_preferences_payload_invalid" in result["blockers"]
    assert "ood_axis_results_payload_invalid" in result["blockers"]
    assert "accepted_external_anchor_rows_payload_invalid" in result["blockers"]


def test_decision_grade_ranking_rejects_malformed_pairwise_evidence_entries() -> None:
    candidate = _ranking_request()
    candidate["pairwise_preferences"][0]["evidence_refs"].append("corrupt-evidence-ref")
    candidate["pairwise_preferences"][0]["policy_a"] = "unknown-policy"

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "pairwise_evidence_payload_invalid:0" in result["blockers"]
    assert "pairwise_policy_identity_invalid:0" in result["blockers"]


def test_decision_grade_ranking_normalizes_pairwise_policy_ids() -> None:
    original = _ranking_request()
    padded = deepcopy(original)
    padded["pairwise_preferences"][0]["policy_a"] = (
        f"  {padded['pairwise_preferences'][0]['policy_a']}  "
    )
    padded["pairwise_preferences"][0]["policy_b"] = (
        f"  {padded['pairwise_preferences'][0]['policy_b']}  "
    )

    expected = build_decision_grade_ranking(original)
    result = build_decision_grade_ranking(padded)

    assert result["status"] == "decision_grade"
    assert result["bradley_terry"] == expected["bradley_terry"]


def test_decision_grade_ranking_binds_profile_specific_evidence_digests() -> None:
    candidate = _ranking_request()
    padded_profile_id = " oscar_roboarena_v2 "
    profile_digests = {
        "official_runtime_contract_sha256": _digest(8000),
        "fk_result_sha256": _digest(8001),
        "camera_projection_sha256": _digest(8002),
        "skeleton_conditioning_sha256": _digest(8003),
    }
    for row_index, (design_row, result_row) in enumerate(
        zip(candidate["evaluation_design"]["rows"], candidate["episode_results"])
    ):
        design_row["evaluator_backend"]["backend_id"] = (
            " cosmos-3-evaluator-adapter " if row_index % 2 else "cosmos-3-evaluator-adapter"
        )
        design_row.update(
            {
                "evaluator_profile_id": (
                    padded_profile_id if row_index % 2 else "oscar_roboarena_v2"
                ),
                **profile_digests,
                "official_runtime_contract_status": "validated",
                "fk_status": "passed",
                "camera_projection_status": "passed",
                "skeleton_validation_status": "passed",
            }
        )
        result_row.update(
            {
                "evaluator_profile_id": "oscar_roboarena_v2",
                **profile_digests,
            }
        )
        result_row["fk_result_sha256"] = (
            " SHA256:" + profile_digests["fk_result_sha256"].upper() + " "
        )

    # Admission and final ranking must select the same profile even when
    # serializers disagree about surrounding whitespace in profile/backend
    # identities between matched policies or between design and result rows.
    assert build_decision_grade_ranking(candidate)["status"] == "decision_grade"

    candidate["episode_results"][0].pop("fk_result_sha256")
    candidate["episode_results"][1]["skeleton_conditioning_sha256"] = _digest(8999)
    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "blocked"
    assert "episode_result_chain_digest_missing:0:fk_result_sha256" in result["blockers"]
    assert (
        "episode_result_chain_digest_mismatch:1:skeleton_conditioning_sha256" in result["blockers"]
    )


def test_decision_grade_ranking_is_invariant_to_registered_row_permutation() -> None:
    original = _ranking_request()
    permuted = deepcopy(original)
    permuted["evaluation_design"]["rows"].reverse()
    permuted["episode_results"].reverse()
    permuted["pairwise_preferences"].reverse()

    first = build_decision_grade_ranking(original)
    second = build_decision_grade_ranking(permuted)

    assert first["status"] == second["status"] == "decision_grade"
    assert first["policy_scorecards"] == second["policy_scorecards"]
    assert first["bradley_terry"] == second["bradley_terry"]


def test_decision_grade_ranking_retains_all_equal_ties_as_equal_ability() -> None:
    candidate = _ranking_request()
    candidate["pairwise_preferences"] = [
        {
            **row,
            "outcome": "tie",
        }
        for row in candidate["pairwise_preferences"]
    ]

    result = build_decision_grade_ranking(candidate)

    assert result["status"] == "decision_grade"
    assert {row["ability"] for row in result["bradley_terry"]["ranking"]} == {1.0}
