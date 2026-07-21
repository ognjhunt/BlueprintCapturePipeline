from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.decision_grade_ranking import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_REPLICATES,
    build_decision_grade_ranking,
)
from blueprint_pipeline.evaluator_evidence_profiles import (
    COMMON_DIGEST_FIELDS,
    canonical_evaluator_backend_manifest_sha256,
)
from blueprint_pipeline.evaluator_qualification_workflow import (
    ALLOCATION_SCHEMA_VERSION,
    DELIVERY_SCHEMA_VERSION,
    MEDIA_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    TEARDOWN_SCHEMA_VERSION,
    build_evaluator_qualification_workflow,
    main,
)
from blueprint_pipeline.evaluator_runtime_evidence import (
    EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
    EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION,
    canonical_json_sha256,
)
from blueprint_pipeline.policy_evaluation_contracts import (
    MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION,
    POLICY_ADAPTER_SCHEMA_VERSION,
    POLICY_EVALUATION_DESIGN_SCHEMA_VERSION,
)
from blueprint_pipeline.site_reference_database import EVALUATION_SITE_ADMISSION_SCHEMA_VERSION


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _site_admission(site_index: int) -> dict[str, object]:
    site_id = f"site-{site_index}"
    scene_id = f"scene-{site_index}"
    capture_id = f"capture-{site_index}"
    base = 10_000 + site_index * 100
    task_contracts = [
        {
            "task_id": "open-door",
            "criterion_id": "door-angle",
            "evidence_type": "articulation_state",
            "tolerance": 0.2,
            "tolerance_unit": "radian",
            "evaluator_mapping": "generic.door_transition.v1",
        }
    ]
    return {
        "schema_version": EVALUATION_SITE_ADMISSION_SCHEMA_VERSION,
        "importer_kind": "independent_capture_import",
        "immutable_source_identity": {
            "site_id": site_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "source_bundle_id": f"bundle-{site_index}",
            "capture_sha256": _digest(base),
            "source_bundle_sha256": _digest(base + 1),
            "manifest_sha256": _digest(base + 2),
        },
        "independent_evidence_verification": {
            "status": "verified",
            "independent_of_importer_and_model_backend": True,
            "verifier_is_importer": False,
            "verification_method": "offline_evidence_verifier",
            "verifier_id": "site-verifier",
            "verifier_version": "2.0.0",
            "verification_report_sha256": _digest(base + 3),
            "source_artifact_index_sha256": _digest(base + 4),
            "verified_source_manifest_sha256": _digest(base + 2),
        },
        "rights_privacy_provenance": {
            "consent_active": True,
            "rights_verified": True,
            "privacy_review_passed": True,
            "provenance_verified": True,
            "commercial_sim_evaluation_allowed": True,
            "rights_manifest_sha256": _digest(base + 5),
            "consent_scope_id": f"consent-{site_index}",
            "privacy_policy_id": "privacy-v1",
            "provenance_chain_id": f"provenance-{site_index}",
            "commercial_use_scope": ["sim_evaluation", "buyer_delivery"],
        },
        "metric_coordinate_contract": {
            "scale_status": "verified_metric",
            "length_unit": "m",
            "up_axis": "+Z",
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "coordinate_frame_manifest_sha256": _digest(base + 6),
            "world_frame_id": "world-z-up",
            "site_frame_id": f"{site_id}-frame",
            "capture_frame_id": f"{capture_id}-frame",
            "scale_evidence_sha256": _digest(base + 7),
            "gravity_alignment_sha256": _digest(base + 8),
            "uncertainty": {
                "scale_sigma": 0.001,
                "translation_sigma_m": 0.002,
                "rotation_sigma_deg": 0.1,
            },
        },
        "camera_time_calibration": {
            "intrinsics_calibrated": True,
            "extrinsics_calibrated": True,
            "timestamps_synchronized": True,
            "reprojection_check_passed": True,
            "reprojection_rmse_px": 0.4,
            "maximum_reprojection_rmse_px": 1.0,
            "calibration_manifest_sha256": _digest(base + 9),
            "intrinsics_sha256": _digest(base + 10),
            "extrinsics_sha256": _digest(base + 11),
            "timestamps_sha256": _digest(base + 12),
        },
        "static_robot_evaluation_viewpoints": [
            {
                "viewpoint_id": f"view-{site_index}",
                "camera_profile_id": "camera-1",
                "robot_profile_id": "robot-1",
                "source_capture_id": capture_id,
                "source_frame_id": "frame-1",
                "derived_from_moving_scan": True,
                "status": "calibrated_static_viewpoint",
                "pose_sha256": _digest(base + 13),
                "source_trajectory_sha256": _digest(base + 14),
            }
        ],
        "robot_camera_embodiment": {
            "robot_profile_id": "robot-1",
            "camera_profile_id": "camera-1",
            "embodiment_id": "embodiment-1",
            "robot_profile_sha256": _digest(base + 15),
            "camera_profile_sha256": _digest(base + 16),
            "embodiment_manifest_sha256": _digest(base + 17),
        },
        "task_scene_grounding": {
            "scene_identity": scene_id,
            "task_objects": [{"object_id": "door", "scene_id": scene_id, "capture_id": capture_id}],
            "articulated_parts": [
                {"part_id": "hinge", "scene_id": scene_id, "capture_id": capture_id}
            ],
            "target_zones": [{"zone_id": "open", "scene_id": scene_id, "capture_id": capture_id}],
            "grounding_manifest_sha256": _digest(base + 18),
        },
        "task_contracts": task_contracts,
        "task_contract_manifest_sha256": _digest(base + 19),
        "task_contract_rows_sha256": _canonical_digest(task_contracts),
        "truth_layers": {
            "visual_geometry": {"status": "verified", "evidence_sha256": _digest(base + 20)},
            "collision": {"status": "verified", "evidence_sha256": _digest(base + 21)},
            "contact": {"status": "verified", "evidence_sha256": _digest(base + 22)},
            "dynamics": {"status": "verified", "evidence_sha256": _digest(base + 23)},
        },
        "deduplication": {
            "status": "passed",
            "site_dedup_id": f"site-dedup-{site_index}",
            "task_dedup_id": f"task-dedup-{site_index}",
            "trajectory_dedup_id": f"trajectory-dedup-{site_index}",
            "dedup_report_sha256": _digest(base + 24),
        },
        "frozen_splits": {
            "locked_before_evaluation": True,
            "split_manifest_sha256": _digest(20_000),
            "train_sites": ["site-0", "site-1"],
            "dev_sites": ["site-2"],
            "held_out_sites": ["site-3"],
        },
        "ood_abstention": {
            "abstention_enabled": True,
            "out_of_distribution_behavior": "abstain",
            "calibration_manifest_sha256": _digest(base + 25),
            "axes": [
                {"axis": axis}
                for axis in (
                    "site",
                    "task",
                    "policy_family",
                    "embodiment",
                    "camera",
                    "visual",
                    "dynamics",
                    "contact",
                )
            ],
        },
    }


def _policy(index: int) -> dict[str, object]:
    return {
        "schema_version": POLICY_ADAPTER_SCHEMA_VERSION,
        "policy_id": f"policy-{index}",
        "checkpoint_id": f"checkpoint-{index}",
        "policy_family": f"family-{index}",
        "embodiment_id": "robot-1",
        "version": "1.0.0",
        "policy_sha256": _digest(30_000 + index),
        "checkpoint_sha256": _digest(30_100 + index),
        "adapter_code_sha256": _digest(30_200 + index),
        "embodiment_manifest_sha256": _digest(30_300 + index),
        "action_contract": {
            "dimension": 2,
            "units": ["rad", "rad"],
            "bounds": [
                {"minimum": -1.0, "maximum": 1.0},
                {"minimum": -1.0, "maximum": 1.0},
            ],
            "control_rate_hz": 50.0,
            "timestamp_semantics": "monotonic_chunk_start_and_per_sample_offsets",
            "normalization_manifest_sha256": _digest(30_400 + index),
            "missing_action_behavior": "block",
            "out_of_bounds_behavior": "block",
        },
    }


def _backend() -> dict[str, object]:
    backend = {
        "schema_version": "evaluator_backend_manifest.v1",
        "backend_id": "cosmos-3-evaluator-adapter",
        "model_family": "cosmos3",
        "model_version": "3",
        "adapter_version": "1.0.0",
        "backend_kind": "world_model",
        "execution_interface": "provider_worker",
        "model_artifact_sha256": _digest(40_000),
        "adapter_code_sha256": _digest(40_001),
        "runtime_manifest_sha256": _digest(40_002),
        "license_manifest_sha256": _digest(40_003),
        "backend_is_compute_provider": False,
    }
    return backend


def _qualification_request() -> dict[str, object]:
    policies = [_policy(index) for index in range(7)]
    backend = _backend()
    rows: list[dict[str, object]] = []
    episode_results: list[dict[str, object]] = []
    runtime_requests: list[dict[str, object]] = []
    media_rows: list[dict[str, object]] = []
    row_index = 0
    for policy_index, policy in enumerate(policies):
        for site_index in range(4):
            for seed in range(MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION):
                criterion_results = [
                    {
                        "criterion_id": "door-angle",
                        "outcome": "success" if seed < 12 + policy_index else "failure",
                        "confidence": 0.95,
                        "label_blinded_and_randomized": True,
                        "evidence_refs": [{"sha256": _digest(50_000 + row_index)}],
                        "failure_taxonomy": []
                        if seed < 12 + policy_index
                        else ["criterion_not_reached"],
                    }
                ]
                criterion_digest = _canonical_digest(criterion_results)
                provider_execution_digest_placeholder = _digest(60_000 + row_index)
                row = {
                    "policy_id": policy["policy_id"],
                    "site_id": f"site-{site_index}",
                    "task_id": "open-door",
                    "condition_id": "condition-1",
                    "seed": seed,
                    "site_task_condition_seed_manifest_sha256": _digest(
                        70_000 + site_index * 100 + seed
                    ),
                    "observation_sha256": _digest(71_000 + site_index * 100 + seed),
                    "commanded_action_chunk_sha256": _digest(80_000 + row_index),
                    "policy_runtime_output_sha256": _digest(90_000 + row_index),
                    "initial_condition_sha256": _digest(71_000 + site_index * 100 + seed),
                    "evaluator_profile_manifest_sha256": _digest(100_000),
                    "evaluator_backend_manifest_sha256": canonical_evaluator_backend_manifest_sha256(
                        backend
                    ),
                    "evaluator_backend": backend,
                    "evaluator_request_sha256": _digest(110_000 + row_index),
                    "evaluator_checkpoint_sha256": backend["model_artifact_sha256"],
                    "model_output_sha256": _digest(120_000 + row_index),
                    "provider_execution_sha256": provider_execution_digest_placeholder,
                    "next_policy_query_sha256": _digest(130_000 + row_index),
                    "action_control_suite_sha256": _digest(140_000 + row_index),
                    "criterion_result_sha256": criterion_digest,
                    "authoritative_manifest_sha256": _digest(150_000 + row_index),
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
                runtime_output_sha256 = _digest(160_000 + row_index)
                provider_execution = {
                    "schema_version": "evaluator_provider_execution.v1",
                    "status": "succeeded",
                    "execution_id": "runpod-execution-1",
                    "runtime_id": "cosmos-runtime-1",
                    "provider_id": "runpod",
                    "runtime_output_sha256": runtime_output_sha256,
                    "model_artifact_sha256": backend["model_artifact_sha256"],
                    "adapter_code_sha256": backend["adapter_code_sha256"],
                    "runtime_manifest_sha256": backend["runtime_manifest_sha256"],
                    "provider_is_evaluator_identity": False,
                }
                provider_execution_sha256 = canonical_json_sha256(provider_execution)
                row["provider_execution_sha256"] = provider_execution_sha256
                row["evaluator_runtime_output_sha256"] = runtime_output_sha256
                receipt = {
                    "schema_version": EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION,
                    "status": "validated",
                    "blockers": [],
                    "source_runtime_blockers": [],
                    "runtime_id": "cosmos-runtime-1",
                    "runtime_adapter_id": "cosmos-adapter",
                    "runtime_adapter_version": "1.0.0",
                    "backend_id": backend["backend_id"],
                    "model_family": backend["model_family"],
                    "model_version": backend["model_version"],
                    "runtime_output_sha256": runtime_output_sha256,
                    "model_artifact_sha256": backend["model_artifact_sha256"],
                    "adapter_code_sha256": backend["adapter_code_sha256"],
                    "runtime_manifest_sha256": backend["runtime_manifest_sha256"],
                    "license_manifest_sha256": backend["license_manifest_sha256"],
                    "provider_id": "runpod",
                    "provider_execution": provider_execution,
                    "provider_execution_sha256": provider_execution_sha256,
                    "runtime_status": "completed",
                    "infrastructure_status": "succeeded",
                    "fresh_model_execution_proven": True,
                    "fresh_model_run_steps": 1,
                    "backend_is_compute_provider": False,
                    "model_outputs": [
                        {
                            "output_id": f"output-{row_index}",
                            "model_output_sha256": row["model_output_sha256"],
                            "model_output_status": "completed",
                        }
                    ],
                    "fixture_or_proxy_model_output_used": False,
                    "fallback_model_output_used": False,
                    "stale_model_output_used": False,
                }
                runtime_requests.append(
                    {
                        "policy_id": row["policy_id"],
                        "site_id": row["site_id"],
                        "task_id": row["task_id"],
                        "condition_id": row["condition_id"],
                        "seed": seed,
                        "normalization_request": {
                            "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
                            "runtime_receipt": receipt,
                            "runtime_receipt_sha256": canonical_json_sha256(receipt),
                            "model_output_id": f"output-{row_index}",
                            "evaluator_row": row,
                        },
                    }
                )
                episode_results.append(
                    {
                        "policy_id": row["policy_id"],
                        "site_id": row["site_id"],
                        "task_id": row["task_id"],
                        "condition_id": row["condition_id"],
                        "seed": seed,
                        "full_ordered_episode_evidence": True,
                        "episode_evidence_sha256": _digest(170_000 + row_index),
                        "artifact_freshness_status": "current",
                        "evaluator_profile_id": row["evaluator_profile_id"],
                        "evaluator_backend_id": backend["backend_id"],
                        "fresh_evaluator_model_execution_proven": True,
                        "fresh_evaluator_model_run_steps": 1,
                        "authoritative_manifest_status": "completed",
                        "infrastructure_status": "succeeded",
                        "evaluator_outcome_status": "valid",
                        "fixture_or_proxy_model_output_used": False,
                        "fallback_policy_used": False,
                        **{field: row[field] for field in COMMON_DIGEST_FIELDS},
                        "criterion_results": criterion_results,
                    }
                )
                media_rows.append(
                    {
                        "schema_version": MEDIA_SCHEMA_VERSION,
                        "policy_id": row["policy_id"],
                        "site_id": row["site_id"],
                        "task_id": row["task_id"],
                        "condition_id": row["condition_id"],
                        "seed": seed,
                        "status": "valid",
                        "model_derived": True,
                        "model_output_sha256": row["model_output_sha256"],
                        "media_sha256": row["model_output_sha256"],
                        "validation_report_sha256": _digest(180_000 + row_index),
                    }
                )
                rows.append(row)
                row_index += 1
    design = {
        "schema_version": POLICY_EVALUATION_DESIGN_SCHEMA_VERSION,
        "policies": policies,
        "rows": rows,
        "hidden_shared_state_prohibited": True,
        "policy_specific_scenario_changes_prohibited": True,
        "minimum_matched_replicates_per_policy_condition": MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION,
        "direct_sc3_comparison_requested": False,
    }
    model_set_members = [
        {
            "artifact_kind": artifact_kind,
            "artifact_id": f"{policy['policy_id']}:{artifact_kind}",
            "sha256": policy[field],
        }
        for policy in policies
        for artifact_kind, field in (
            ("policy", "policy_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        )
    ]
    model_set_members.append(
        {
            "artifact_kind": "evaluator_model",
            "artifact_id": str(backend["backend_id"]),
            "sha256": backend["model_artifact_sha256"],
        }
    )
    ranking_inputs = {
        "schema_version": "decision_grade_ranking_request.v2",
        "minimum_calibrated_judge_confidence": 0.8,
        "judge_calibration_set_sha256": _digest(220_000),
        "judge_calibration_status": "accepted",
        "label_authority_independent_of_policy_and_model": True,
        "episode_results": episode_results,
        "pairwise_preferences": [
            {
                "policy_a": f"policy-{index}",
                "policy_b": f"policy-{index + 1}",
                "outcome": "policy_b",
                "label_blinded_and_randomized": True,
                "evidence_refs": [{"sha256": _digest(230_000 + index)}],
            }
            for index in range(6)
        ],
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
                "sample_count": 80,
                "coverage_95_ci": [0.8, 0.97],
                "abstention_95_ci": [0.03, 0.2],
                "failure_taxonomy": {"criterion_not_reached": 2},
                "split_manifest_sha256": _digest(240_000),
            }
            for axis in ("site", "task", "embodiment", "viewpoint", "appearance")
        ],
        "accepted_external_anchor_rows": [],
    }
    ranking_result = build_decision_grade_ranking({**ranking_inputs, "evaluation_design": design})
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "qualification_id": "qualification-1",
        "evaluated_at": "2026-07-21T18:00:00+00:00",
        "marketed_provider_ids": ["runpod", "vast"],
        "release_identity": {
            "source_commit": "a" * 40,
            "source_archive_sha256": _digest(200_000),
            "release_manifest_sha256": _digest(200_001),
            "container_image_sha256": _digest(200_002),
            "model_set_manifest_sha256": _canonical_digest(model_set_members),
            "model_set_members": model_set_members,
            "data_split_manifest_sha256": _digest(20_000),
        },
        "site_admissions": [_site_admission(index) for index in range(4)],
        "evaluation_design": design,
        "runtime_evidence_requests": runtime_requests,
        "provider_allocations": [
            {
                "schema_version": ALLOCATION_SCHEMA_VERSION,
                "allocation_id": "allocation-1",
                "execution_id": "runpod-execution-1",
                "provider_id": "runpod",
                "status": "allocated",
                "source_commit": "a" * 40,
                "allocation_receipt_sha256": _digest(210_000),
                "allocation_command_evidence_sha256": _digest(210_001),
                "budget_admission_sha256": _digest(210_002),
                "container_image_sha256": _digest(200_002),
            }
        ],
        "media_validity": media_rows,
        "ranking_inputs": ranking_inputs,
        "delivery_evidence": {
            "schema_version": DELIVERY_SCHEMA_VERSION,
            "status": "delivered",
            "qualification_id": "qualification-1",
            "source_commit": "a" * 40,
            "authenticated": True,
            "authorized": True,
            "tenant_isolation_verified": True,
            "buyer_scorecard_sha256": _digest(250_000),
            "delivery_receipt_sha256": _digest(250_001),
            "ranking_result_sha256": _canonical_digest(ranking_result),
        },
        "teardown_evidence": {
            "schema_version": TEARDOWN_SCHEMA_VERSION,
            "status": "proven_zero",
            "qualification_id": "qualification-1",
            "source_commit": "a" * 40,
            "exact_attempt_allocation_ids": ["allocation-1"],
            "exact_attempt_active_resource_count": 0,
            "global_provider_inventory": [
                {
                    "provider_id": provider_id,
                    "active_resource_count": 0,
                    "hourly_allocation_burn_usd": 0.0,
                    "inventory_report_sha256": _digest(260_000 + index),
                }
                for index, provider_id in enumerate(("runpod", "vast"))
            ],
            "teardown_report_sha256": _digest(260_010),
            "observed_at": "2026-07-21T18:00:00+00:00",
            "billing_reconciliation": {
                "status": "reconciled",
                "billing_export_sha256": _digest(260_020),
                "total_spend_usd": 12.34,
            },
        },
    }
    return request


def test_workflow_composes_real_contracts_without_inheriting_paper_metrics() -> None:
    result = build_evaluator_qualification_workflow(_qualification_request())

    assert result["status"] == "public_launch_qualified"
    assert result["scientific_qualification_status"] == "decision_grade"
    assert result["public_launch_qualification_status"] == "qualified"
    assert result["matrix"] == {
        "policy_count": 7,
        "site_count": 4,
        "task_count": 1,
        "condition_count": 1,
        "matched_cell_count_per_policy": 80,
        "minimum_matched_replicates_per_policy_condition": 20,
    }
    assert len(result["model_provider_proof"]["runtime_rows"]) == 560
    assert result["ranking"]["correlation_status"] == "correlation_not_measured"
    assert result["ranking"]["pearson"] is None
    assert result["ranking"]["spearman"] is None
    assert result["ranking"]["mmrv"] is None
    assert all(state["status"] != "blocked" for state in result["lifecycle"].values())
    assert result["claim_boundary"]["simulator_ranking_is_not_physical_robot_performance"]
    assert result["claim_boundary"]["paper_metrics_are_not_blueprint_results"]


def test_blocked_manifest_does_not_get_overridden_by_model_or_media_success() -> None:
    request = _qualification_request()
    request["evaluation_design"]["rows"][0]["authoritative_manifest_status"] = "blocked"
    request["runtime_evidence_requests"][0]["normalization_request"]["evaluator_row"][
        "authoritative_manifest_status"
    ] = "blocked"
    request["ranking_inputs"]["episode_results"][0]["authoritative_manifest_status"] = "blocked"

    result = build_evaluator_qualification_workflow(request)

    assert result["lifecycle"]["provider_allocation"]["status"] == "proven"
    assert result["lifecycle"]["media_validity"]["status"] == "valid"
    assert result["lifecycle"]["model_execution"]["status"] == "blocked"
    assert result["lifecycle"]["episode_artifacts_assembled"]["status"] == "blocked"
    assert result["lifecycle"]["rank_result"]["status"] == "blocked"
    assert result["status"] == "blocked"


def test_runtime_normalized_row_must_name_the_outer_envelope_cell() -> None:
    request = _qualification_request()
    request["runtime_evidence_requests"][0]["normalization_request"]["evaluator_row"][
        "policy_id"
    ] = "policy-1"

    result = build_evaluator_qualification_workflow(request)

    assert "runtime_normalized_cell_identity_mismatch:0" in result["lifecycle"][
        "model_execution"
    ]["blockers"]
    assert result["lifecycle"]["model_execution"]["status"] == "blocked"
    assert result["status"] == "blocked"


def test_valid_scientific_ranking_stays_separate_from_billing_and_teardown() -> None:
    request = _qualification_request()
    request["teardown_evidence"]["exact_attempt_active_resource_count"] = 1
    request["teardown_evidence"]["billing_reconciliation"] = {
        "status": "not_reconciled",
        "billing_export_sha256": None,
        "total_spend_usd": None,
    }

    result = build_evaluator_qualification_workflow(request)

    assert result["scientific_qualification_status"] == "decision_grade"
    assert result["ranking"]["status"] == "decision_grade"
    assert result["public_launch_qualification_status"] == "blocked"
    assert result["lifecycle"]["teardown"]["status"] == "blocked"
    assert result["lifecycle"]["billing_reconciliation"]["status"] == "blocked"
    assert result["teardown"]["total_spend_usd"] is None


def test_workflow_rejects_three_site_smoke_fallback_and_sensitive_input() -> None:
    request = _qualification_request()
    request["site_admissions"] = request["site_admissions"][:3]
    request["evaluation_design"]["rows"] = [
        row for row in request["evaluation_design"]["rows"] if row["site_id"] != "site-3"
    ]
    request["runtime_evidence_requests"] = [
        row for row in request["runtime_evidence_requests"] if row["site_id"] != "site-3"
    ]
    request["media_validity"] = [
        row for row in request["media_validity"] if row["site_id"] != "site-3"
    ]
    request["ranking_inputs"]["episode_results"] = [
        row for row in request["ranking_inputs"]["episode_results"] if row["site_id"] != "site-3"
    ]
    request["evaluation_design"]["rows"][0]["fallback_policy_used"] = True
    request["runtime_evidence_requests"][0]["normalization_request"]["evaluator_row"][
        "fallback_policy_used"
    ] = True
    request["api_token"] = "must-not-be-retained"

    result = build_evaluator_qualification_workflow(request)

    assert "qualification_site_count_lt_4" in result["lifecycle"]["site_admission"]["blockers"]
    assert any(
        "fallback_policy_used" in blocker
        for blocker in result["lifecycle"]["policy_registry"]["blockers"]
    )
    assert result["lifecycle"]["request_acceptance"]["status"] == "blocked"
    assert result["sensitive_paths_omitted"] == 1
    assert "must-not-be-retained" not in json.dumps(result)


def test_workflow_rejects_separator_and_prefix_variants_of_sensitive_keys() -> None:
    request = _qualification_request()
    request["metadata"] = {
        "client_secret": "secret-value",
        "refresh_token": "refresh-value",
        "api-token": "api-value",
        "private-key": "private-value",
    }

    result = build_evaluator_qualification_workflow(request)

    assert result["lifecycle"]["request_acceptance"]["status"] == "blocked"
    assert result["sensitive_paths_omitted"] == 4
    serialized = json.dumps(result)
    assert all(
        value not in serialized
        for value in ("secret-value", "refresh-value", "api-value", "private-value")
    )


def test_release_split_container_model_set_and_delivery_bindings_fail_closed() -> None:
    request = _qualification_request()
    request["release_identity"]["model_set_members"][0]["sha256"] = _digest(999_000)
    request["site_admissions"][0]["frozen_splits"]["split_manifest_sha256"] = _digest(999_001)
    request["site_admissions"][1]["immutable_source_identity"]["capture_sha256"] = request[
        "site_admissions"
    ][0]["immutable_source_identity"]["capture_sha256"]
    request["provider_allocations"][0]["container_image_sha256"] = _digest(999_002)
    request["delivery_evidence"]["ranking_result_sha256"] = _digest(999_003)

    result = build_evaluator_qualification_workflow(request)

    request_blockers = result["lifecycle"]["request_acceptance"]["blockers"]
    assert "release_model_set_manifest_digest_mismatch" in request_blockers
    assert "release_model_set_does_not_exactly_bind_run_artifacts" in request_blockers
    assert (
        "site_split_manifest_does_not_match_release"
        in result["lifecycle"]["site_admission"]["blockers"]
    )
    assert (
        "independent_site_capture_identity_not_unique:capture_sha256"
        in result["lifecycle"]["site_admission"]["blockers"]
    )
    assert (
        "provider_allocation_container_image_mismatch:0"
        in result["lifecycle"]["provider_allocation"]["blockers"]
    )
    assert "delivery_ranking_result_digest_mismatch" in result["lifecycle"]["delivery"]["blockers"]


def test_cli_writes_blocked_result_and_returns_two(tmp_path: Path) -> None:
    request = _qualification_request()
    request["delivery_evidence"]["status"] = "blocked"
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert main(["--request", str(request_path), "--output", str(output_path)]) == 2
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["scientific_qualification_status"] == "decision_grade"
    assert result["public_launch_qualification_status"] == "blocked"
    assert result["lifecycle"]["delivery"]["status"] == "blocked"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []
