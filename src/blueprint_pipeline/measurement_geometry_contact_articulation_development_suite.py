"""Plan or execute the checked MuJoCo door/drawer development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from .measurement_adapter_runtime import build_measurement_adapter_descriptor
from .measurement_mujoco_articulation_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


SUITE_SCHEMA_VERSION = "capture_to_geometry_contact_articulation_development_suite.v1"
CORPUS_SCHEMA_VERSION = "capture_to_geometry_contact_articulation_development_corpus.v1"


class ArticulationDevelopmentSuiteError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_digest(value: Any) -> bool:
    raw = str(value).strip() if value is not None else ""
    return (
        len(raw) == 71
        and raw.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in raw[7:])
    )


def _number(value: Any, code: str) -> float:
    if isinstance(value, bool):
        raise ArticulationDevelopmentSuiteError(code)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArticulationDevelopmentSuiteError(code) from exc
    if not math.isfinite(result):
        raise ArticulationDevelopmentSuiteError(code)
    return result


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulationDevelopmentSuiteError(
            "articulation_development_corpus_unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArticulationDevelopmentSuiteError("articulation_development_corpus_not_object")
    corpus = dict(value)
    errors: list[str] = []
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("articulation_development_corpus_schema_invalid")
    if corpus.get("lane") != "articulated_rigid_contact":
        errors.append("articulation_development_corpus_lane_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("instrumented_force_included", False),
        ("qualification_labels_included", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
    ):
        if corpus.get(key) is not expected:
            errors.append(f"articulation_development_corpus_{key}_invalid")
    cases = corpus.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) < 2
        or not all(isinstance(row, Mapping) for row in cases)
    ):
        errors.append("articulation_development_corpus_cases_invalid")
    elif len({str(row.get("case_id", "")).strip() for row in cases}) != len(cases):
        errors.append("articulation_development_corpus_case_ids_duplicate")
    if not isinstance(corpus.get("shared_operating_point"), Mapping):
        errors.append("articulation_development_corpus_shared_point_invalid")
    if errors:
        raise ArticulationDevelopmentSuiteError(*errors)
    return corpus


def run_capture_to_geometry_contact_articulation_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    execute: bool = False,
) -> dict[str, Any]:
    if not _valid_digest(qualification_split_digest):
        raise ArticulationDevelopmentSuiteError(
            "articulation_development_qualification_split_digest_invalid"
        )
    if not _valid_digest(controller_scope_digest):
        raise ArticulationDevelopmentSuiteError(
            "articulation_development_controller_scope_digest_invalid"
        )
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise ArticulationDevelopmentSuiteError("articulation_development_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-mujoco-articulation-1",
        method_ids=["mujoco-3"],
        development_split_digest=corpus_digest,
        qualification_split_digest=qualification_split_digest,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[controller_scope_digest],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 0.05,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 90},
        minimum_repeated_trials=2,
    )
    descriptor = build_measurement_adapter_descriptor("mujoco-3")
    bundles: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, raw_case in enumerate(corpus["cases"]):
        row = dict(raw_case)
        case_id = str(row.pop("case_id", "")).strip()
        operating_point = {**dict(corpus["shared_operating_point"]), **row}
        case = build_benchmark_case_manifest(
            spec,
            case_id=case_id,
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class=(
                "articulated_door_opening"
                if operating_point["articulation_type"] == "door_hinge"
                else "articulated_drawer_opening"
            ),
            material_regime="synthetic_rigid_articulation",
            operating_point=operating_point,
        )
        request = build_measurement_adapter_execution_request(
            descriptor,
            spec,
            case,
            execution_id=f"mujoco-articulation-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id="mujoco-cpu-articulation",
            precision="float64",
            seed=31,
            solver_settings={
                "integrator": "implicitfast",
                "solver": "Newton",
                "iterations": 100,
                "tolerance": 1e-10,
            },
            timeout_seconds=45,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[
                sys.executable,
                "-m",
                "blueprint_pipeline.measurement_mujoco_articulation_adapter",
            ],
            execute=execute,
        )
        bundles.append(bundle)
        prediction = bundle["prediction"]
        runtime = bundle["receipt"].get("runtime_observations", {})
        summaries.append(
            {
                "case_id": case_id,
                "articulation_type": operating_point["articulation_type"],
                "case_manifest_digest": case["case_manifest_digest"],
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "deterministic_replay_match": (runtime.get("deterministic_replay_match") is True),
                "final_joint_position": runtime.get("final_joint_position"),
                "peak_absolute_joint_velocity": runtime.get("peak_absolute_joint_velocity"),
                "applied_effort": runtime.get("applied_effort"),
                "observed_metrics": (
                    dict(prediction["observed_metrics"]) if isinstance(prediction, Mapping) else {}
                ),
                "unsafe_condition_predicted": (
                    prediction["unsafe_condition_predicted"]
                    if isinstance(prediction, Mapping)
                    else None
                ),
            }
        )
    completed = all(row["receipt_status"] == "completed" for row in summaries)
    deterministic = completed and all(
        row["deterministic_replay_match"] is True for row in summaries
    )
    aggregate_metrics: dict[str, float | int] = {}
    if completed:
        errors = [
            _number(
                row["observed_metrics"]["drawer_door_force_travel_error"],
                "articulation_development_travel_error_invalid",
            )
            for row in summaries
        ]
        aggregate_metrics = {
            "maximum_travel_error": max(errors),
            "mean_travel_error": fmean(errors),
            "within_envelope_case_count": sum(
                row["unsafe_condition_predicted"] is False for row in summaries
            ),
            "joint_limit_reached_case_count": sum(
                row["observed_metrics"]["contact_sequence"] == "joint_limit_reached"
                for row in summaries
            ),
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "lane": "articulated_rigid_contact",
        "solver_scope": "mujoco-hinge-slide-newton-implicitfast",
        "execution_requested": execute is True,
        "status": (
            "completed_development_only"
            if completed
            else "planned_not_executed"
            if not execute
            else "incomplete"
        ),
        "case_count": len(summaries),
        "minimum_repeated_trials": spec["minimum_repeated_trials"],
        "all_cases_completed": completed,
        "all_replays_deterministic": deterministic,
        "cases": summaries,
        "aggregate_metrics": aggregate_metrics,
        "execution_bundle_digests": [bundle["execution_bundle_digest"] for bundle in bundles],
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
        "instrumented_force_included": False,
        "qualification_labels_included": False,
        "independent_execution": False,
        "r5_evidence": False,
        "r6_decision": False,
        "r7_admission": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "agent_may_promote": False,
    }
    suite["suite_digest"] = _digest(suite, "suite_digest")
    return validate_capture_to_geometry_contact_articulation_development_suite(suite)


def validate_capture_to_geometry_contact_articulation_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        errors.append("articulation_development_suite_schema_invalid")
    if suite.get("lane") != "articulated_rigid_contact":
        errors.append("articulation_development_suite_lane_invalid")
    if suite.get("solver_scope") != "mujoco-hinge-slide-newton-implicitfast":
        errors.append("articulation_development_suite_solver_scope_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("articulation_development_suite_status_invalid")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        errors.append("articulation_development_suite_cases_invalid")
    if suite.get("case_count") != len(cases or []):
        errors.append("articulation_development_suite_case_count_mismatch")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("instrumented_force_included", False),
        ("qualification_labels_included", False),
        ("independent_execution", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
        ("production_route_eligible", False),
        ("physical_success_established", False),
        ("agent_may_promote", False),
    ):
        if suite.get(key) is not expected:
            errors.append(f"articulation_development_suite_{key}_invalid")
    if suite.get("status") == "completed_development_only":
        if suite.get("all_cases_completed") is not True:
            errors.append("articulation_development_suite_completion_invalid")
        if suite.get("all_replays_deterministic") is not True:
            errors.append("articulation_development_suite_replay_invalid")
        if (
            not isinstance(suite.get("aggregate_metrics"), Mapping)
            or not suite["aggregate_metrics"]
        ):
            errors.append("articulation_development_suite_metrics_missing")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"articulation_development_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("articulation_development_suite_digest_mismatch")
    if errors:
        raise ArticulationDevelopmentSuiteError(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run the MuJoCo door/drawer development corpus"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_capture_to_geometry_contact_articulation_development_suite(
        args.corpus,
        qualification_split_digest=args.qualification_split_digest,
        controller_scope_digest=args.controller_scope_digest,
        execute=args.execute,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if suite["status"] != "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArticulationDevelopmentSuiteError",
    "CORPUS_SCHEMA_VERSION",
    "SUITE_SCHEMA_VERSION",
    "main",
    "run_capture_to_geometry_contact_articulation_development_suite",
    "validate_capture_to_geometry_contact_articulation_development_suite",
]
