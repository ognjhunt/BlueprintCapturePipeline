"""Scalar scoring for autoresearch eval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .common import read_json, utc_now_iso, write_json


def compute_total_score(
    *,
    test_pass_rate: float,
    structured_output_rate: float,
    rubric_score: float,
    penalties: float,
) -> float:
    return (100.0 * test_pass_rate) + (20.0 * structured_output_rate) + (20.0 * rubric_score) - penalties


def should_accept_candidate(
    *,
    best_score: float,
    best_diff_size_lines: int,
    candidate_score: float,
    candidate_diff_size_lines: int,
) -> tuple[bool, str]:
    if candidate_score > best_score:
        return True, "higher_score"
    if candidate_score == best_score and candidate_diff_size_lines < best_diff_size_lines:
        return True, "tie_smaller_diff"
    if candidate_score == best_score:
        return False, "tie_larger_or_equal_diff"
    return False, "lower_score"


def score_eval_payload(
    eval_payload: Mapping[str, Any],
    *,
    diff_size_lines: int = 0,
    forbidden_mutation_detected: bool = False,
) -> dict[str, Any]:
    pytest_summary = eval_payload.get("pytest", {})
    structured = eval_payload.get("structured_checks", {})
    rubric = eval_payload.get("rubric", {})
    penalties = float(eval_payload.get("penalties", 0.0) or 0.0)
    if forbidden_mutation_detected:
        penalties += 50.0

    test_pass_rate = float(pytest_summary.get("pass_rate", 0.0) or 0.0)
    structured_output_rate = float(structured.get("rate", 0.0) or 0.0)
    rubric_score = float(rubric.get("score", 0.0) or 0.0)
    total_score = compute_total_score(
        test_pass_rate=test_pass_rate,
        structured_output_rate=structured_output_rate,
        rubric_score=rubric_score,
        penalties=penalties,
    )
    return {
        "target_skill": eval_payload.get("target_skill"),
        "generated_at": utc_now_iso(),
        "iteration": eval_payload.get("iteration"),
        "total_score": round(total_score, 4),
        "test_pass_rate": round(test_pass_rate, 6),
        "structured_output_rate": round(structured_output_rate, 6),
        "rubric_score": round(rubric_score, 6),
        "penalties": round(penalties, 4),
        "diff_size_lines": int(diff_size_lines),
        "forbidden_mutation_detected": bool(forbidden_mutation_detected),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an autoresearch eval payload")
    parser.add_argument("--eval-json", required=True, help="Path to eval.json")
    parser.add_argument("--output", required=True, help="Path to write score.json")
    parser.add_argument("--diff-size-lines", type=int, default=0)
    parser.add_argument("--forbidden-mutation-detected", action="store_true")
    args = parser.parse_args(argv)

    payload = read_json(Path(args.eval_json))
    score_payload = score_eval_payload(
        payload,
        diff_size_lines=args.diff_size_lines,
        forbidden_mutation_detected=args.forbidden_mutation_detected,
    )
    write_json(Path(args.output), score_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
