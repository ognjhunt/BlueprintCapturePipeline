"""Plan or execute the checked PyChrono NSC granular development corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from .measurement_adapter_runtime import build_measurement_adapter_descriptor
from .measurement_chrono_granular_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    WORKER_SCRIPT,
    implementation_digest,
)
from .measurement_deformation_granular_development_suite import (
    GranularDevelopmentSuiteError,
    _digest,
    _file_digest,
    _number,
    _valid_digest,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


SUITE_SCHEMA_VERSION = "capture_to_deformation_granular_chrono_development_suite.v1"
CORPUS_SCHEMA_VERSION = "capture_to_deformation_granular_chrono_development_corpus.v1"
METHOD_ID = "project-chrono-10"
SOLVER_SCOPE = "chrono-nsc-cpu-bullet-psor"
CHRONO_PYTHON_ENV = "BLUEPRINT_CHRONO_PYTHON"


class ChronoGranularDevelopmentSuiteError(GranularDevelopmentSuiteError):
    """Raised when the Chrono development suite fails closed validation."""


def _suite_error(*codes: str) -> ChronoGranularDevelopmentSuiteError:
    return ChronoGranularDevelopmentSuiteError(*codes)


def _worker_python(value: str | Path | None) -> Path:
    raw = str(value or os.environ.get(CHRONO_PYTHON_ENV) or sys.executable).strip()
    if not raw:
        raise _suite_error("chrono_granular_worker_python_missing")
    path = Path(raw).expanduser().absolute()
    if not path.is_file():
        raise _suite_error("chrono_granular_worker_python_invalid")
    return path


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _suite_error("chrono_granular_corpus_unreadable") from exc
    if not isinstance(value, Mapping):
        raise _suite_error("chrono_granular_corpus_not_object")
    corpus = dict(value)
    errors: list[str] = []
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("chrono_granular_corpus_schema_invalid")
    if corpus.get("lane") != "granular":
        errors.append("chrono_granular_corpus_lane_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("physical_material_characterization_included", False),
        ("qualification_labels_included", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
    ):
        if corpus.get(key) is not expected:
            errors.append(f"chrono_granular_corpus_{key}_invalid")
    shared = corpus.get("shared_operating_point")
    if not isinstance(shared, Mapping):
        errors.append("chrono_granular_corpus_shared_point_invalid")
    elif (
        shared.get("contact_method") != "nonsmooth_contact"
        or shared.get("material_characterization_scope") != "synthetic_parameters_only"
    ):
        errors.append("chrono_granular_corpus_scope_invalid")
    cases = corpus.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) < 2
        or not all(isinstance(row, Mapping) for row in cases)
    ):
        errors.append("chrono_granular_corpus_cases_invalid")
    elif len({str(row.get("case_id", "")).strip() for row in cases}) != len(cases):
        errors.append("chrono_granular_corpus_case_ids_duplicate")
    if errors:
        raise _suite_error(*errors)
    return corpus


def run_capture_to_deformation_granular_chrono_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    worker_python: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Run all public Chrono NSC cases through the uniform adapter boundary."""

    if not _valid_digest(qualification_split_digest):
        raise _suite_error("chrono_granular_qualification_split_digest_invalid")
    if not _valid_digest(controller_scope_digest):
        raise _suite_error("chrono_granular_controller_scope_digest_invalid")
    python = _worker_python(worker_python)
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise _suite_error("chrono_granular_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-chrono-nsc-spherical-granular-1",
        method_ids=[METHOD_ID],
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 180},
        minimum_repeated_trials=2,
        lane="granular",
    )
    descriptor = build_measurement_adapter_descriptor(METHOD_ID)
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
            task_class="granular_manipulation",
            material_regime="synthetic_chrono_nsc_granular_media",
            operating_point=operating_point,
        )
        request = build_measurement_adapter_execution_request(
            descriptor,
            spec,
            case,
            execution_id=f"chrono-nsc-spherical-granular-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id=SOLVER_SCOPE,
            precision="float64",
            seed=37,
            solver_settings={
                "collision_system": "bullet",
                "contact_method": "nsc",
                "replay_count": 2,
                "solver": "psor",
                "timestepper": "euler_implicit_linearized",
            },
            timeout_seconds=120,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[str(python), str(WORKER_SCRIPT)],
            execute=execute,
        )
        bundles.append(bundle)
        prediction = bundle["prediction"]
        runtime = bundle["receipt"].get("runtime_observations", {})
        summaries.append(
            {
                "case_id": case_id,
                "synthetic_material_id": operating_point["synthetic_material_id"],
                "case_manifest_digest": case["case_manifest_digest"],
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "failure_codes": list(bundle["receipt"].get("failure_codes") or []),
                "engine_version": runtime.get("engine_version"),
                "package_build": runtime.get("package_build"),
                "package_subdir": runtime.get("package_subdir"),
                "openmp_preload_used": runtime.get("openmp_preload_used"),
                "deterministic_replay_match": runtime.get("deterministic_replay_match") is True,
                "spread_ratio": runtime.get("spread_ratio"),
                "final_settled_fraction": runtime.get("final_settled_fraction"),
                "maximum_normal_contact_force_n": runtime.get("maximum_normal_contact_force_n"),
                "penetration_m": runtime.get("penetration_m"),
                "chrono_granular_gpu_module_used": runtime.get(
                    "chrono_granular_gpu_module_used"
                ),
                "observed_metrics": (
                    dict(prediction["observed_metrics"])
                    if isinstance(prediction, Mapping)
                    else {}
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
        spread = [_number(row["spread_ratio"], "chrono_granular_spread_invalid") for row in summaries]
        settled = [
            _number(row["final_settled_fraction"], "chrono_granular_settled_invalid")
            for row in summaries
        ]
        force = [
            _number(row["maximum_normal_contact_force_n"], "chrono_granular_force_invalid")
            for row in summaries
        ]
        penetration = [
            _number(row["penetration_m"], "chrono_granular_penetration_invalid")
            for row in summaries
        ]
        aggregate_metrics = {
            "mean_spread_ratio": fmean(spread),
            "minimum_spread_ratio": min(spread),
            "maximum_spread_ratio": max(spread),
            "minimum_settled_fraction": min(settled),
            "maximum_normal_contact_force_n": max(force),
            "maximum_penetration_m": max(penetration),
            "within_envelope_case_count": sum(
                row["observed_metrics"].get("task_outcome")
                == "within_chrono_nsc_spherical_particle_envelope"
                for row in summaries
            ),
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "lane": "granular",
        "method_id": METHOD_ID,
        "solver_scope": SOLVER_SCOPE,
        "worker_python_name": python.name,
        "external_worker_python_required": True,
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
        "chrono_granular_gpu_module_used": False,
        "cases": summaries,
        "aggregate_metrics": aggregate_metrics,
        "execution_bundle_digests": [bundle["execution_bundle_digest"] for bundle in bundles],
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
        "physical_material_characterization_included": False,
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
    return validate_capture_to_deformation_granular_chrono_development_suite(suite)


def validate_capture_to_deformation_granular_chrono_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    for key, expected in (
        ("schema_version", SUITE_SCHEMA_VERSION),
        ("corpus_schema_version", CORPUS_SCHEMA_VERSION),
        ("lane", "granular"),
        ("method_id", METHOD_ID),
        ("solver_scope", SOLVER_SCOPE),
        ("external_worker_python_required", True),
        ("chrono_granular_gpu_module_used", False),
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("physical_material_characterization_included", False),
        ("qualification_labels_included", False),
        ("independent_execution", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
        ("production_route_eligible", False),
        ("physical_success_established", False),
        ("agent_may_promote", False),
    ):
        if suite.get(key) is not expected and suite.get(key) != expected:
            errors.append(f"chrono_granular_suite_{key}_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("chrono_granular_suite_status_invalid")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        errors.append("chrono_granular_suite_cases_invalid")
    if suite.get("case_count") != len(cases or []):
        errors.append("chrono_granular_suite_case_count_mismatch")
    if suite.get("status") == "completed_development_only":
        if suite.get("all_cases_completed") is not True:
            errors.append("chrono_granular_suite_completion_invalid")
        if suite.get("all_replays_deterministic") is not True:
            errors.append("chrono_granular_suite_replay_invalid")
        if not isinstance(suite.get("aggregate_metrics"), Mapping) or not suite[
            "aggregate_metrics"
        ]:
            errors.append("chrono_granular_suite_metrics_missing")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"chrono_granular_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("chrono_granular_suite_digest_mismatch")
    if errors:
        raise _suite_error(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_capture_to_deformation_granular_chrono_development_suite(
        args.corpus,
        qualification_split_digest=args.qualification_split_digest,
        controller_scope_digest=args.controller_scope_digest,
        worker_python=args.worker_python,
        execute=args.execute,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if suite["status"] != "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHRONO_PYTHON_ENV",
    "CORPUS_SCHEMA_VERSION",
    "ChronoGranularDevelopmentSuiteError",
    "METHOD_ID",
    "SOLVER_SCOPE",
    "SUITE_SCHEMA_VERSION",
    "main",
    "run_capture_to_deformation_granular_chrono_development_suite",
    "validate_capture_to_deformation_granular_chrono_development_suite",
]
