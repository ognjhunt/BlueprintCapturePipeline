from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import (
    PAIR_RESULT_SCHEMA_VERSION,
    DiagnosticContractError,
    analyze_pair_results,
    audit_frozen_gemini_subset_reuse,
    build_complete_pair_inventory,
    build_pair_subset_inventory,
    build_pair_inventory,
    complete_graph_diagnostic_protocol,
    diagnostic_protocol,
    materialize_native_videos,
    pilot_cost_projection,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def _source_inventory(tmp_path: Path) -> dict:
    rows = []
    for session_index in range(63):
        for policy_index in range(7):
            frame = tmp_path / f"s{session_index}-p{policy_index}.jpg"
            frame.write_bytes(f"{session_index}-{policy_index}".encode())
            row = {
                "source_request_id": f"source-{session_index}-{policy_index}",
                "session_id": f"session-{session_index:02d}",
                "policy_id_internal_only": f"policy-{policy_index}",
                "task_instruction": f"task-{session_index}",
                "frames": [
                    {
                        "path": str(frame),
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ],
                "cropped_output_sha256": "a" * 64,
                "deterministic_collapse_flags": [],
            }
            rows.append(row)
    source = {
        "status": "ready",
        "request_count": 441,
        "requests": rows,
    }
    source["inventory_sha256"] = canonical_sha256(source)
    return source


def _result(pair: dict, preference: str) -> dict:
    result = {
        "schema_version": PAIR_RESULT_SCHEMA_VERSION,
        "pair_id": pair["pair_id"],
        "structured_response": {
            "preferred_episode": preference,
            "abstention_factors": ["ambiguous"] if preference == "abstain" else [],
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def test_protocol_freezes_four_diagnostic_only_arms_and_existing_caps() -> None:
    protocol = diagnostic_protocol()
    assert protocol["claim_class"] == "diagnostic_only"
    assert protocol["independent_confirmation_credit"] is False
    assert [arm["arm_id"] for arm in protocol["arms"]] == [
        "gpt5_oscar_comparability",
        "gpt54_mini_challenger",
        "gemini36_flash_native_video",
        "cosmos3_nano_reasoner",
    ]
    assert protocol["comparison_graph"]["total_edges"] == 441
    assert protocol["comparison_graph"]["complete_graph_total_edges"] == 1323
    assert protocol["cost_caps_usd"]["combined_evaluator_api_including_prior_phase_a"] == 25
    assert protocol["arms"][0]["max_output_tokens_including_reasoning"] == 4000
    assert protocol["arms"][1]["max_output_tokens_including_reasoning"] == 3000
    assert protocol["admission_and_stopping"]["seven_pair_cost_projection"]["sample_size"] == 7


def test_cycle_inventory_includes_every_episode_twice_without_labels(tmp_path: Path) -> None:
    inventory = build_pair_inventory(_source_inventory(tmp_path))
    assert inventory["pair_count"] == 441
    assert inventory["outcome_labels_accessed_to_build_pairs"] is False
    appearances = {}
    for pair in inventory["pairs"]:
        for side in ("episode_a", "episode_b"):
            key = (pair["session_id"], pair[side]["policy_id_internal_only"])
            appearances[key] = appearances.get(key, 0) + 1
    assert set(appearances.values()) == {2}
    assert len(appearances) == 441


def test_complete_graph_inventory_includes_all_pairs_without_labels(tmp_path: Path) -> None:
    protocol = complete_graph_diagnostic_protocol()
    inventory = build_complete_pair_inventory(_source_inventory(tmp_path))

    assert protocol["comparison_graph"]["total_edges"] == 1323
    assert protocol["method_attribution"] == (
        "OSCAR_method_inspired_not_exact_code_reproduction"
    )
    assert inventory["pair_count"] == 1323
    assert inventory["comparison_graph_kind"] == "complete_graph_per_session"
    assert inventory["protocol_sha256"] == protocol["protocol_sha256"]
    assert inventory["outcome_labels_accessed_to_build_pairs"] is False
    assert len({pair["pair_id"] for pair in inventory["pairs"]}) == 1323
    appearances = {}
    for pair in inventory["pairs"]:
        for side in ("episode_a", "episode_b"):
            key = (pair["session_id"], pair[side]["policy_id_internal_only"])
            appearances[key] = appearances.get(key, 0) + 1
    assert len(appearances) == 441
    assert set(appearances.values()) == {6}


def test_frozen_cycle_results_map_exactly_into_complete_graph(tmp_path: Path) -> None:
    source = _source_inventory(tmp_path)
    prior_inventory = build_pair_inventory(source)
    complete_inventory = build_complete_pair_inventory(source)
    results = []
    for pair in prior_inventory["pairs"]:
        result = _result(pair, "tie")
        result.update(
            {
                "provider": "google_gemini_api",
                "model": "gemini-3.6-flash",
                "transport": "gemini_batch_api_native_video",
                "policy_identity_sent_to_provider": False,
                "physical_outcome_sent_to_provider": False,
                "physical_ground_truth_pixels_sent_to_provider": False,
            }
        )
        result["result_sha256"] = canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        results.append(result)
    collection = {
        "status": "completed",
        "result_count": 441,
        "error_count": 0,
        "results": results,
    }
    collection["report_sha256"] = canonical_sha256(collection)

    audit = audit_frozen_gemini_subset_reuse(
        complete_inventory, prior_inventory, collection
    )
    missing = build_pair_subset_inventory(
        complete_inventory, audit["missing_pair_ids"]
    )

    assert audit["status"] == "passed"
    assert audit["reused_pair_count"] == 441
    assert audit["same_orientation_count"] + audit["reversed_orientation_count"] == 441
    assert audit["missing_pair_count"] == 882
    assert audit["outcome_labels_accessed"] is False
    assert missing["pair_count"] == 882
    assert missing["parent_inventory_sha256"] == complete_inventory["inventory_sha256"]


def test_bradley_terry_analysis_ranks_consistent_winner_first(tmp_path: Path) -> None:
    inventory = build_pair_inventory(_source_inventory(tmp_path))
    results = []
    for pair in inventory["pairs"]:
        a = pair["episode_a"]["policy_id_internal_only"]
        b = pair["episode_b"]["policy_id_internal_only"]
        results.append(_result(pair, "A" if a < b else "B"))
    report = analyze_pair_results(inventory, results, arm_id="test")
    assert report["predicted_policy_order"][0] == "policy-0"
    assert report["coverage"] == 1.0
    assert report["independent_confirmation_credit"] is False


def test_partial_matrix_cannot_receive_ranking_credit(tmp_path: Path) -> None:
    inventory = build_pair_inventory(_source_inventory(tmp_path))
    with pytest.raises(DiagnosticContractError, match="partial_matrix"):
        analyze_pair_results(inventory, [], arm_id="test")


def test_disconnected_non_abstained_graph_cannot_receive_ranking_credit(
    tmp_path: Path,
) -> None:
    inventory = build_pair_inventory(_source_inventory(tmp_path))
    results = []
    first_component = {"policy-0", "policy-1", "policy-2"}
    for pair in inventory["pairs"]:
        policies = {
            pair["episode_a"]["policy_id_internal_only"],
            pair["episode_b"]["policy_id_internal_only"],
        }
        same_component = policies <= first_component or not (
            policies & first_component
        )
        results.append(_result(pair, "tie" if same_component else "abstain"))
    with pytest.raises(DiagnosticContractError, match="connected"):
        analyze_pair_results(inventory, results, arm_id="test")


def test_native_video_materialization_fails_closed_on_unfrozen_crop_audit(
    tmp_path: Path,
) -> None:
    with pytest.raises(DiagnosticContractError, match="crop_audit_digest_invalid"):
        materialize_native_videos(
            {"audit_sha256": "wrong"},
            visual_review={},
            source_root=tmp_path,
            output_root=tmp_path,
        )


def test_pilot_cost_projection_uses_frozen_conservative_maximum() -> None:
    report = pilot_cost_projection(
        single_canary_batch_equivalent_cost_usd=0.01,
        pilot_batch_costs_usd=[0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.02],
        arm_cap_usd=10,
    )
    assert report["sample_size"] == 7
    assert report["per_request_upper_estimate_usd"] >= 0.02
    assert report["projected_matrix_cost_usd"] == (
        report["per_request_upper_estimate_usd"] * 441
    )
    assert report["arm_cost_admitted"] is True


def test_pilot_cost_projection_supports_complete_graph_request_count() -> None:
    report = pilot_cost_projection(
        single_canary_batch_equivalent_cost_usd=0.01,
        pilot_batch_costs_usd=[0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.02],
        arm_cap_usd=30,
        matrix_request_count=1323,
    )
    assert report["matrix_request_count"] == 1323
    assert report["projected_matrix_cost_usd"] == (
        report["per_request_upper_estimate_usd"] * 1323
    )
