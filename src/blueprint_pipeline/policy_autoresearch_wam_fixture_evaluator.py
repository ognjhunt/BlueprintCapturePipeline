"""Fixture WAM evaluator command for policy-autoresearch split matrices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_any, utc_now_iso, write_json
from .policy_autoresearch import _attempt_for_run, _mapping, _recipe_with_capabilities, _string
from .wam_eval_substrate import normalize_evaluation_substrate


def _claim_boundary(substrate: str) -> dict[str, Any]:
    return {
        "evaluation_substrate": substrate,
        "generated_wam_rollouts_are_model_derived_support_artifacts": True,
        "generated_rollouts_are_raw_capture_evidence": False,
        "fixture_wam_is_deterministic_local_test_substrate": substrate == "fixture_wam",
        "live_provider_calls_performed": False,
        "customer_specific_srcc_claimed": False,
        "customer_specific_srcc_requires_real_world_validation_rollouts": True,
        "simulator_execution_performed": False,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def run_policy_autoresearch_wam_fixture_evaluator(
    *,
    recipe_path: str | Path,
    matrix_path: str | Path,
    output_path: str | Path,
    evaluation_substrate: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    recipe = _recipe_with_capabilities(_mapping(read_json_any(Path(recipe_path))))
    matrix = _mapping(read_json_any(Path(matrix_path)))
    substrate = normalize_evaluation_substrate(
        evaluation_substrate
        or matrix.get("evaluation_substrate")
        or matrix.get("evaluationSubstrate")
        or os.getenv("BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE")
        or "fixture_wam"
    )
    phase = (
        _string(matrix.get("phase"))
        or os.getenv("BLUEPRINT_POLICY_AUTORESEARCH_PHASE")
        or "heldout"
    )
    runs = [dict(run) for run in matrix.get("runs", []) if isinstance(run, Mapping)]
    attempts = []
    for run in runs:
        attempt = _attempt_for_run(
            recipe=recipe,
            run=run,
            phase=phase,
            engine=substrate,
            generated_at=generated,
        )
        attempt["evaluation_substrate"] = substrate
        attempt["simulator_engine"] = substrate
        attempt["metrics"] = {
            **_mapping(attempt.get("metrics")),
            "world_model_uncertainty": 0.12
            if attempt.get("task_success")
            else 0.36,
            "simulator_execution_performed": False,
        }
        attempt["claim_boundary"] = _claim_boundary(substrate)
        attempts.append(attempt)

    successful = [attempt for attempt in attempts if attempt.get("task_success")]
    payload = {
        "schema_version": "policy_autoresearch_wam_fixture_eval_output.v1",
        "generated_at": generated,
        "status": "completed" if attempts else "blocked_missing_eval_runs",
        "evaluation_substrate": substrate,
        "simulator_engine": substrate,
        "phase": phase,
        "attempt_count": len(attempts),
        "task_success_rate": round(len(successful) / len(attempts), 6)
        if attempts
        else 0.0,
        "attempts": attempts,
        "claim_boundary": _claim_boundary(substrate),
    }
    write_json(Path(output_path), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic fixture WAM evaluator for policy autoresearch"
    )
    parser.add_argument(
        "--recipe",
        default=os.getenv("BLUEPRINT_POLICY_AUTORESEARCH_RECIPE"),
    )
    parser.add_argument(
        "--matrix",
        default=os.getenv("BLUEPRINT_POLICY_AUTORESEARCH_MATRIX"),
    )
    parser.add_argument(
        "--output",
        default=os.getenv("BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"),
    )
    parser.add_argument(
        "--evaluation-substrate",
        default=os.getenv("BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE")
        or "fixture_wam",
    )
    args = parser.parse_args(argv)
    if not args.recipe or not args.matrix or not args.output:
        parser.error(
            "--recipe, --matrix, and --output are required unless policy-autoresearch "
            "environment variables are set"
        )
    result = run_policy_autoresearch_wam_fixture_evaluator(
        recipe_path=args.recipe,
        matrix_path=args.matrix,
        output_path=args.output,
        evaluation_substrate=args.evaluation_substrate,
    )
    print(f"[policy-autoresearch-wam-fixture] status={result['status']}")
    print(f"[policy-autoresearch-wam-fixture] output={Path(args.output).resolve()}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
