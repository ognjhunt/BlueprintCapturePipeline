from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from blueprint_pipeline.policy_ranking_roboarena_analysis import (
    analysis_contract,
    analysis_contract_v3,
    freeze_predictions,
    unseal_and_analyze,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def _matrix() -> tuple[dict, dict]:
    evaluator = {
        "model": "gpt-5-mini-2025-08-07",
        "evaluator_digest": "e" * 64,
    }
    requests = []
    results = []
    for session_index in range(4):
        for policy_index in range(7):
            request_id = f"request-{session_index}-{policy_index}"
            request = {
                "request_id": request_id,
                "source_request_id": f"source-{session_index}-{policy_index}",
                "session_id": f"session-{session_index}",
                "policy_id_internal_only": f"policy-{policy_index}",
                "evaluator_digest": evaluator["evaluator_digest"],
            }
            requests.append(request)
            payload = {
                "progress_score_0_to_5": min(5, policy_index),
                "success_probability": policy_index / 6,
                "stable_success_confirmed": policy_index == 6,
                "temporal_consistency": 0.9,
                "action_following_confidence": 0.9,
                "uncertainty": 0.1,
                "artifact_flags": ["none"],
            }
            result = {
                "request_id": request_id,
                "source_request_id": request["source_request_id"],
                "session_id": request["session_id"],
                "policy_id_internal_only": request["policy_id_internal_only"],
                "evaluator_digest": evaluator["evaluator_digest"],
                "model": evaluator["model"],
                "response_id": f"resp-{session_index}-{policy_index}",
                "structured_response": payload,
                "deterministic_collapse_flags": [],
                "evaluator_abstain": False,
                "blueprint_safety_abstain": False,
                "abstention_sources": [],
                "usage": {"estimated_cost_usd": 0.001},
                "latency_seconds": 1.0,
                "policy_identity_sent_to_provider": False,
                "benchmark_outcomes_sent_to_provider": False,
                "physical_ground_truth_pixels_sent_to_provider": False,
            }
            result["result_sha256"] = canonical_sha256(result)
            results.append(result)
    inventory = {
        "status": "ready",
        "protocol_sha256": "p" * 64,
        "inventory_sha256": "i" * 64,
        "evaluator": evaluator,
        "request_count": len(requests),
        "requests": requests,
    }
    run = {
        "status": "completed",
        "inventory_sha256": inventory["inventory_sha256"],
        "results": results,
        "failures": [],
        "provider_called": True,
        "data_uploaded": True,
        "estimated_cost_usd": 0.028,
        "outcome_labels_accessed": False,
        "run_sha256": "r" * 64,
    }
    return inventory, run


def _write_labels(root: Path) -> None:
    for session_index in range(4):
        path = root / "evaluation_sessions" / f"session-{session_index}" / "metadata.yaml"
        path.parent.mkdir(parents=True)
        rows = []
        for policy_index in range(7):
            rows.append(
                {
                    "policy_name": f"policy-{policy_index}",
                    "binary_success": policy_index >= 4,
                    "partial_success": min(5, policy_index) / 5,
                }
            )
        path.write_text(yaml.safe_dump({"policies": rows}), encoding="utf-8")


def test_analysis_contract_freezes_inherited_strict_selectivity() -> None:
    contract = analysis_contract(protocol_sha256="p" * 64, evaluator_digest="e" * 64)
    assert contract["selectivity"] == {
        "evaluator_uncertainty_max": 0.2,
        "action_following_confidence_min": 0.7,
        "temporal_consistency_min": 0.7,
        "pair_score_margin_min": 0.2,
    }
    assert contract["outcome_labels_accessed"] is False
    assert contract["risk_coverage"]["registered_gate_grid"] == [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    artifact = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728"
            / "phase_a_analysis_lock_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact == analysis_contract_v3(
        protocol_sha256="6b41ea618ec290f1c080573e093d26cae03d14a0ecb06b3bb2a4bd016e469066",
        evaluator_digest="6b22136e8708223bd8b8213ca907632659d23949c7a196957af087ae2a197f57",
    )
    v4_artifact = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728"
            / "phase_a_analysis_lock_v4.json"
        ).read_text(encoding="utf-8")
    )
    assert v4_artifact == analysis_contract(
        protocol_sha256="6b41ea618ec290f1c080573e093d26cae03d14a0ecb06b3bb2a4bd016e469066",
        evaluator_digest="079bb46d34ba24d0741784f8df6b25be221081412c26ea998ee8ea40eccf7108",
    )


def test_freeze_requires_exact_valid_complete_result_set() -> None:
    inventory, run = _matrix()
    frozen = freeze_predictions(inventory, run)
    assert frozen["status"] == "frozen"
    assert frozen["prediction_row_count"] == 28
    assert frozen["outcome_labels_accessed"] is False

    changed = copy.deepcopy(run)
    changed["results"][0]["structured_response"]["progress_score_0_to_5"] = 5
    blocked = freeze_predictions(inventory, changed)
    assert blocked["status"] == "blocked"
    assert any(item.startswith("result_digest_invalid:") for item in blocked["blockers"])


def test_unseal_refuses_wrong_prediction_digest_without_reading_labels(tmp_path: Path) -> None:
    inventory, run = _matrix()
    frozen = freeze_predictions(inventory, run)
    report = unseal_and_analyze(
        frozen,
        expected_frozen_predictions_sha256="0" * 64,
        roboarena_root=tmp_path / "does-not-exist",
        dataset_revision="dataset-revision",
    )
    assert report["status"] == "blocked_before_label_access"
    assert report["outcome_labels_accessed"] is False


def test_perfect_known_answer_matrix_passes_rank_and_abstention_gates(tmp_path: Path) -> None:
    inventory, run = _matrix()
    frozen = freeze_predictions(inventory, run)
    _write_labels(tmp_path)
    report = unseal_and_analyze(
        frozen,
        expected_frozen_predictions_sha256=frozen["frozen_predictions_sha256"],
        roboarena_root=tmp_path,
        dataset_revision="dataset-revision",
        unsealed_at="2026-07-28T00:00:00Z",
    )
    assert report["status"] == "completed"
    assert report["rank_metrics"]["spearman_rho"] == 1.0
    assert report["rank_metrics"]["kendall_tau_b"] == 1.0
    assert report["rank_metrics"]["policy_pairwise_ordering_accuracy"] == 1.0
    assert report["rank_metrics"]["mmrv_simpler_pairwise_real_binary_margin"] == 0.0
    assert report["abstention"]["selective_pairwise_coverage"] == 1.0
    assert report["abstention"]["risk_rule_passed"] is True
    assert report["all_registered_gates_passed"] is True
    assert report["rank_metrics"]["exact_permutation_uncertainty"]["permutation_count"] == 5040
