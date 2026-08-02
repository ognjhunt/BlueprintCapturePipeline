"""Plan or execute the checked world-model action-fidelity development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from .measurement_adapter_runtime import build_measurement_adapter_descriptor
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)
from .measurement_world_model_action_fidelity_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
)


CORPUS_SCHEMA_VERSION = "world_model_action_fidelity_development_corpus.v1"
SUITE_SCHEMA_VERSION = "world_model_action_fidelity_development_suite.v1"


class WorldModelActionFidelitySuiteError(ValueError):
    pass


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_digest(value: Any) -> bool:
    raw = str(value or "")
    return (
        len(raw) == 71
        and raw.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in raw[7:])
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_corpus_unreadable") from exc
    if not isinstance(value, Mapping):
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_corpus_not_object")
    corpus = dict(value)
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_corpus_schema_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_outcomes_included", False),
        ("policy_ranking_labels_included", False),
        ("provider_execution_included", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
    ):
        if corpus.get(key) is not expected:
            raise WorldModelActionFidelitySuiteError(f"world_model_fidelity_{key}_invalid")
    if not isinstance(corpus.get("cases"), list) or len(corpus["cases"]) < 2:
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_cases_invalid")
    return corpus


def run_world_model_action_fidelity_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    execute: bool = False,
) -> dict[str, Any]:
    if not _valid_digest(qualification_split_digest) or not _valid_digest(controller_scope_digest):
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_input_digest_invalid")
    path = corpus_path.resolve()
    corpus = _load(path)
    corpus_digest = _file_digest(path)
    if corpus_digest == qualification_split_digest:
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="world-model-action-fidelity",
        benchmark_version="development-contract-1",
        method_ids=["gigaworld-wmbench"],
        development_split_digest=corpus_digest,
        qualification_split_digest=qualification_split_digest,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[controller_scope_digest],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 1.0,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 60},
    )
    descriptor = build_measurement_adapter_descriptor("gigaworld-wmbench")
    bundles: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, raw in enumerate(corpus["cases"]):
        row = dict(raw)
        case_id = str(row.pop("case_id", "")).strip()
        manifest = build_benchmark_case_manifest(
            spec,
            case_id=case_id,
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class="long_horizon_task_execution",
            material_regime="none",
            operating_point={**dict(corpus["shared_operating_point"]), **row},
        )
        request = build_measurement_adapter_execution_request(
            descriptor,
            spec,
            manifest,
            execution_id=f"world-model-action-fidelity-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id="blueprint-strict-wam-action-consistency",
            precision="float64",
            seed=41,
            solver_settings={"protocol": "world_model_action_fidelity.v1", "replay_count": 2},
            timeout_seconds=30,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[
                sys.executable,
                "-m",
                "blueprint_pipeline.measurement_world_model_action_fidelity_adapter",
            ],
            execute=execute,
        )
        bundles.append(bundle)
        prediction = bundle["prediction"]
        runtime = bundle["receipt"].get("runtime_observations", {})
        cases.append(
            {
                "case_id": case_id,
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "deterministic_replay_match": runtime.get("deterministic_replay_match") is True,
                "maximum_abs_error": runtime.get("maximum_abs_error"),
                "observed_metrics": (
                    dict(prediction["observed_metrics"]) if isinstance(prediction, Mapping) else {}
                ),
            }
        )
    completed = all(row["receipt_status"] == "completed" for row in cases)
    metrics: dict[str, Any] = {}
    if completed:
        errors = [float(row["maximum_abs_error"]) for row in cases]
        metrics = {
            "mean_maximum_abs_error": fmean(errors),
            "maximum_abs_error": max(errors),
            "within_envelope_case_count": sum(
                row["observed_metrics"].get("task_outcome") == "within_action_fidelity_envelope"
                for row in cases
            ),
            "policy_ranking_case_count": 0,
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "execution_requested": execute is True,
        "status": (
            "completed_development_only"
            if completed
            else "planned_not_executed"
            if not execute
            else "incomplete"
        ),
        "case_count": len(cases),
        "all_cases_completed": completed,
        "all_replays_deterministic": completed
        and all(row["deterministic_replay_match"] for row in cases),
        "cases": cases,
        "aggregate_metrics": metrics,
        "execution_bundle_digests": [row["execution_bundle_digest"] for row in bundles],
        "historical_policy_ranking_verdict": "thesis_not_supported",
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_outcomes_included": False,
        "policy_ranking_labels_included": False,
        "provider_execution_included": False,
        "policy_ranking_scored": False,
        "physics_authority": False,
        "physical_success_established": False,
        "r5_evidence": False,
        "r6_decision": False,
        "r7_admission": False,
        "production_route_eligible": False,
        "agent_may_promote": False,
    }
    suite["suite_digest"] = _digest(suite, "suite_digest")
    return validate_world_model_action_fidelity_development_suite(suite)


def validate_world_model_action_fidelity_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_suite_schema_invalid")
    for key, expected in (
        ("historical_policy_ranking_verdict", "thesis_not_supported"),
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_outcomes_included", False),
        ("policy_ranking_labels_included", False),
        ("provider_execution_included", False),
        ("policy_ranking_scored", False),
        ("physics_authority", False),
        ("physical_success_established", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
        ("production_route_eligible", False),
        ("agent_may_promote", False),
    ):
        if suite.get(key) != expected:
            raise WorldModelActionFidelitySuiteError(f"world_model_fidelity_{key}_invalid")
    if not isinstance(suite.get("cases"), list) or suite.get("case_count") != len(suite["cases"]):
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_suite_cases_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") != expected_digest:
        raise WorldModelActionFidelitySuiteError("world_model_fidelity_suite_digest_mismatch")
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run_world_model_action_fidelity_development_suite(
        args.corpus,
        qualification_split_digest=args.qualification_split_digest,
        controller_scope_digest=args.controller_scope_digest,
        execute=args.execute,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["status"] != "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
