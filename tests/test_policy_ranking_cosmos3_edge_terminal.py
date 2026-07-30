import json
from pathlib import Path


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "policy_ranking_cosmos3_edge_closed_loop_20260729"
)


def _load(name: str) -> dict:
    return json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))


def test_terminal_verdict_keeps_claim_components_and_history_separate() -> None:
    verdict = _load("terminal_verdict_v1.json")
    assert verdict["overall_verdict"] == "inconclusive"
    assert set(verdict["components"]) == {
        "cosmos_wam_qualification",
        "frozen_benchmark_calibration",
        "captured_site_transfer",
        "economics_and_speed",
    }
    assert verdict["historical_frozen_tested_stack"]["verdict"] == "thesis_not_supported"
    assert verdict["historical_frozen_tested_stack"]["immutable"] is True
    assert verdict["independent_confirmation"]["phase_b_measured"] is False


def test_terminal_verdict_does_not_call_public_replay_closed_loop() -> None:
    verdict = _load("terminal_verdict_v1.json")
    truth = verdict["closed_loop_truth"]
    assert truth["public_oscar_policy_evaluation_closed_loop"] is False
    assert truth["oscar_video_autoregression_is_policy_requery"] is False
    assert truth["blueprint_oscar_canary_closed_loop"] is False
    assert truth["blueprint_ctrl_world_canary_closed_loop"] is False
    assert truth["candidate_policy_requeried_on_generated_observation"] is False


def test_evidence_matrix_denies_credit_to_partial_graphs_and_canaries() -> None:
    matrix = _load("final_evidence_matrix_v1.json")
    assert matrix["arms"]["oscar_purpose_built_wam"]["ranking_credit"] is False
    assert matrix["arms"]["ctrl_world"]["ranking_credit"] is False
    assert matrix["arms"]["registered_cosmos3_oscar_skeleton_hybrid"]["hybrid_credit"] is False
    assert matrix["judges"]["gpt5_complete_graph"]["valid_unique_pair_count"] == 355
    assert matrix["judges"]["gpt5_complete_graph"]["bradley_terry_run"] is False
    assert matrix["judges"]["gemini_complete_graph"]["complete_graph_bradley_terry_run"] is False
    assert matrix["phase_b"]["independent_confirmation_credit"] is False
    assert matrix["cost_and_time"]["useful_complete_ranking_completed"] is False
