from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.benchmark_uncertainty import (
    BOOTSTRAP_METHOD,
    REQUEST_SCHEMA_VERSION,
    build_benchmark_uncertainty_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _request() -> dict:
    rows = []
    attempt = 0
    for policy_index, (predicted_base, reference_base) in enumerate(
        ((0.2, 0.1), (0.5, 0.55), (0.85, 0.9)), start=1
    ):
        for site_index in range(2):
            for family_index in range(2):
                attempt += 1
                rows.append(
                    {
                        "attempt_id": f"attempt-{attempt:03d}",
                        "policy_id": f"policy-{policy_index}",
                        "site_id": f"site-{site_index}",
                        "task_family_id": f"family-{family_index}",
                        "task_id": f"task-{family_index}",
                        "initial_condition_id": f"condition-{site_index}-{family_index}",
                        "predicted_score": predicted_base + 0.01 * site_index,
                        "reference_score": reference_base + 0.01 * site_index,
                        "reference_independently_accepted": True,
                        "policy_checkpoint_sha256": f"{policy_index}" * 64,
                        "initial_condition_sha256": f"{site_index + 4}" * 64,
                        "evaluator_output_sha256": f"{family_index + 6}" * 64,
                        "reference_output_sha256": "9" * 64,
                    }
                )
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "study_id": "uncertainty-study-1",
        "study_version": "1.0.0",
        "frozen": True,
        "benchmark_spec_sha256": "a" * 64,
        "attempt_ledger_sha256": "b" * 64,
        "reference_manifest_sha256": "c" * 64,
        "bootstrap": {"seed": 1729, "replicate_count": 40},
        "convergence_trial_counts": [3, 6, 12],
        "convergence_subsample_replicates": 20,
        "rows": rows,
    }


def _schema() -> dict:
    return json.loads(
        (ROOT / "docs" / "schemas" / "benchmark_uncertainty_report.schema.json").read_text(
            encoding="utf-8"
        )
    )


def test_uncertainty_report_covers_hierarchy_convergence_and_leave_one_out() -> None:
    request = _request()
    jsonschema.validate(
        request,
        json.loads(
            (
                ROOT / "docs" / "schemas" / "benchmark_uncertainty_request.schema.json"
            ).read_text(encoding="utf-8")
        ),
    )
    report = build_benchmark_uncertainty_report(request)

    assert report["status"] == "measured"
    assert report["coverage"] == {
        "rollouts": 12,
        "policies": 3,
        "sites": 2,
        "task_families": 2,
        "tasks": 2,
        "initial_conditions": 4,
    }
    assert report["point_metrics"]["pearson"] > 0.99
    assert report["point_metrics"]["spearman"] == 1.0
    assert report["point_metrics"]["kendall_tau_b"] == 1.0
    assert report["point_metrics"]["pairwise_ordering_accuracy"] == 1.0
    assert report["point_metrics"]["mmrv"] == 0.0
    assert report["bootstrap"]["method"] == BOOTSTRAP_METHOD
    assert report["bootstrap"]["resampled_levels"] == [
        "policy",
        "site",
        "task_family",
        "task",
        "initial_condition",
        "trial",
    ]
    assert [row["trial_count"] for row in report["convergence"]] == [3, 6, 12]
    assert len(report["leave_one_policy_out"]) == 3
    assert len(report["leave_one_task_family_out"]) == 2
    assert report["claim_eligibility"]["bootstrap_replicate_count_sufficient"] is False
    assert report["claim_eligibility"]["public_rank_fidelity_claim_eligible"] is False
    jsonschema.validate(report, _schema())


def test_uncertainty_report_is_deterministic_and_input_order_invariant() -> None:
    request = _request()
    first = build_benchmark_uncertainty_report(request)
    reordered = copy.deepcopy(request)
    reordered["rows"] = list(reversed(reordered["rows"]))
    second = build_benchmark_uncertainty_report(reordered)

    assert first == second


def test_uncertainty_report_fails_closed_on_unaccepted_or_duplicate_rows() -> None:
    request = _request()
    request["rows"][0]["reference_independently_accepted"] = False
    request["rows"][1]["attempt_id"] = request["rows"][0]["attempt_id"]

    report = build_benchmark_uncertainty_report(request)

    assert report["status"] == "blocked"
    assert "uncertainty_reference_not_independently_accepted:0" in report["blockers"]
    assert "uncertainty_attempt_ids_duplicate" in report["blockers"]
    assert report["confidence_intervals"] == {}
