"""Plan or execute the checked direct-tactile development corpus."""

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
from .measurement_direct_tactile_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


SUITE_SCHEMA_VERSION = "capture_to_tactile_development_suite.v1"
CORPUS_SCHEMA_VERSION = "capture_to_tactile_development_corpus.v1"


class TactileDevelopmentSuiteError(ValueError):
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
        raise TactileDevelopmentSuiteError(code)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TactileDevelopmentSuiteError(code) from exc
    if not math.isfinite(result):
        raise TactileDevelopmentSuiteError(code)
    return result


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TactileDevelopmentSuiteError("tactile_development_corpus_unreadable") from exc
    if not isinstance(value, Mapping):
        raise TactileDevelopmentSuiteError("tactile_development_corpus_not_object")
    corpus = dict(value)
    errors: list[str] = []
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("tactile_development_corpus_schema_invalid")
    if corpus.get("lane") != "tactile":
        errors.append("tactile_development_corpus_lane_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("real_sensor_calibration_included", False),
        ("qualification_labels_included", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
    ):
        if corpus.get(key) is not expected:
            errors.append(f"tactile_development_corpus_{key}_invalid")
    shared = corpus.get("shared_operating_point")
    cases = corpus.get("cases")
    if (
        not isinstance(shared, Mapping)
        or shared.get("data_origin") != "synthetic_development_fixture"
    ):
        errors.append("tactile_development_corpus_shared_point_invalid")
    if (
        not isinstance(cases, list)
        or len(cases) < 2
        or not all(isinstance(row, Mapping) for row in cases)
    ):
        errors.append("tactile_development_corpus_cases_invalid")
    elif len({str(row.get("case_id", "")).strip() for row in cases}) != len(cases):
        errors.append("tactile_development_corpus_case_ids_duplicate")
    if errors:
        raise TactileDevelopmentSuiteError(*errors)
    return corpus


def run_tactile_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    execute: bool = False,
) -> dict[str, Any]:
    if not _valid_digest(qualification_split_digest):
        raise TactileDevelopmentSuiteError("tactile_qualification_split_digest_invalid")
    if not _valid_digest(controller_scope_digest):
        raise TactileDevelopmentSuiteError("tactile_controller_scope_digest_invalid")
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise TactileDevelopmentSuiteError("tactile_development_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-direct-tactile-1",
        method_ids=["direct-captured-observations"],
        development_split_digest=corpus_digest,
        qualification_split_digest=qualification_split_digest,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[controller_scope_digest],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 4 / 6,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 60},
        minimum_repeated_trials=2,
        lane="tactile",
    )
    descriptor = build_measurement_adapter_descriptor("direct-captured-observations")
    bundles: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, raw_case in enumerate(corpus["cases"]):
        case_row = dict(raw_case)
        case_id = str(case_row.pop("case_id", "")).strip()
        operating_point = {**dict(corpus["shared_operating_point"]), **case_row}
        case = build_benchmark_case_manifest(
            spec,
            case_id=case_id,
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class="tactile_manipulation",
            material_regime="elastomer",
            operating_point=operating_point,
        )
        request = build_measurement_adapter_execution_request(
            descriptor,
            spec,
            case,
            execution_id=f"direct-tactile-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id="numpy-direct-tactile-sequence-reduction",
            precision="float64",
            seed=37,
            solver_settings={
                "analysis_method": "deterministic_sequence_reduction",
                "numpy_version": corpus["numpy_version"],
                "replay_count": 2,
            },
            timeout_seconds=30,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[
                sys.executable,
                "-m",
                "blueprint_pipeline.measurement_direct_tactile_adapter",
            ],
            execute=execute,
        )
        bundles.append(bundle)
        prediction = bundle["prediction"]
        runtime = bundle["receipt"].get("runtime_observations", {})
        summaries.append(
            {
                "case_id": case_id,
                "case_manifest_digest": case["case_manifest_digest"],
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "deterministic_replay_match": runtime.get("deterministic_replay_match") is True,
                "maximum_contact_area_mm2": runtime.get("maximum_contact_area_mm2"),
                "peak_shear_to_normal_ratio": runtime.get("peak_shear_to_normal_ratio"),
                "slip_onset_frame": runtime.get("slip_onset_frame"),
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
        displacement = [
            _number(
                row["observed_metrics"]["state_trajectory"],
                "tactile_state_trajectory_metric_invalid",
            )
            for row in summaries
        ]
        areas = [
            _number(row["maximum_contact_area_mm2"], "tactile_contact_area_metric_invalid")
            for row in summaries
        ]
        ratios = [
            _number(row["peak_shear_to_normal_ratio"], "tactile_force_ratio_metric_invalid")
            for row in summaries
        ]
        aggregate_metrics = {
            "mean_peak_marker_displacement_px": fmean(displacement),
            "maximum_contact_area_mm2": max(areas),
            "maximum_shear_to_normal_ratio": max(ratios),
            "slip_case_count": sum(
                row["observed_metrics"].get("task_outcome") == "incipient_slip_observed"
                for row in summaries
            ),
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "lane": "tactile",
        "sensor_scope": "synthetic-optical-tactile-with-synchronized-force",
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
        "real_sensor_calibration_included": False,
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
    return validate_tactile_development_suite(suite)


def validate_tactile_development_suite(value: Mapping[str, Any]) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        errors.append("tactile_development_suite_schema_invalid")
    if suite.get("lane") != "tactile":
        errors.append("tactile_development_suite_lane_invalid")
    if suite.get("sensor_scope") != "synthetic-optical-tactile-with-synchronized-force":
        errors.append("tactile_development_suite_sensor_scope_invalid")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 2 or suite.get("case_count") != len(cases or []):
        errors.append("tactile_development_suite_cases_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("real_sensor_calibration_included", False),
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
            errors.append(f"tactile_development_suite_{key}_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("tactile_development_suite_status_invalid")
    if suite.get("status") == "completed_development_only" and (
        suite.get("all_cases_completed") is not True
        or suite.get("all_replays_deterministic") is not True
        or not suite.get("aggregate_metrics")
    ):
        errors.append("tactile_development_suite_completion_invalid")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"tactile_development_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("tactile_development_suite_digest_mismatch")
    if errors:
        raise TactileDevelopmentSuiteError(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or run the tactile development corpus")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_tactile_development_suite(
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
    "CORPUS_SCHEMA_VERSION",
    "SUITE_SCHEMA_VERSION",
    "TactileDevelopmentSuiteError",
    "main",
    "run_tactile_development_suite",
    "validate_tactile_development_suite",
]
