from __future__ import annotations

from scripts.benchmark_ptdp_scalability import run_benchmark


def test_scalability_benchmark_exercises_bounded_ann_and_streaming_jsonl() -> None:
    result = run_benchmark(clip_count=1_000, dimension=16)

    assert result["status"] == "passed"
    assert result["workload"]["clip_count"] == 1_000
    assert result["measurements"]["known_duplicate_found"] is True
    assert result["measurements"]["pairwise_similarity_matrix_materialized"] is False
    assert result["measurements"]["ann_candidate_comparison_count"] < 499_500
    assert result["claim_boundary"]["full_real_media_ptdp_workload_executed"] is False
