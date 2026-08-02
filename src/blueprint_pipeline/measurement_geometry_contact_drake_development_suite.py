"""Plan or execute the Drake SAP rigid-contact development corpus."""

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
from .measurement_drake_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    WORKER_SCRIPT,
    implementation_digest,
)
from .measurement_geometry_contact_development_suite import (
    CORPUS_SCHEMA_VERSION,
    GeometryContactDevelopmentSuiteError,
    _digest,
    _file_digest,
    _load_corpus,
    _number,
    _valid_digest,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


SUITE_SCHEMA_VERSION = "capture_to_geometry_contact_drake_development_suite.v1"
METHOD_ID = "drake-1-55"
SOLVER_SCOPE = "drake-multibody-cpu-sap-point"
DRAKE_PYTHON_ENV = "BLUEPRINT_DRAKE_PYTHON"


class GeometryContactDrakeDevelopmentSuiteError(GeometryContactDevelopmentSuiteError):
    """Raised when the Drake development suite fails closed validation."""


def _suite_error(*codes: str) -> GeometryContactDrakeDevelopmentSuiteError:
    return GeometryContactDrakeDevelopmentSuiteError(*codes)


def _worker_python(value: str | Path | None) -> Path:
    raw = str(value or os.environ.get(DRAKE_PYTHON_ENV) or sys.executable).strip()
    if not raw:
        raise _suite_error("geometry_contact_drake_worker_python_missing")
    path = Path(raw).expanduser().absolute()
    # Preserve virtual-environment interpreter symlinks. Resolving one to its
    # base interpreter drops the external Drake environment identity.
    if not path.is_file():
        raise _suite_error("geometry_contact_drake_worker_python_invalid")
    return path


def run_capture_to_geometry_contact_drake_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    worker_python: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Run every public rigid-contact case through Drake's CPU SAP port."""

    if not _valid_digest(qualification_split_digest):
        raise _suite_error("geometry_contact_drake_qualification_split_digest_invalid")
    if not _valid_digest(controller_scope_digest):
        raise _suite_error("geometry_contact_drake_controller_scope_digest_invalid")
    python = _worker_python(worker_python)
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise _suite_error("geometry_contact_drake_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-drake-sap-rigid-contact-1",
        method_ids=[METHOD_ID],
        development_split_digest=corpus_digest,
        qualification_split_digest=qualification_split_digest,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[controller_scope_digest],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 120},
        minimum_repeated_trials=2,
    )
    descriptor = build_measurement_adapter_descriptor(METHOD_ID)
    bundles: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, raw_case in enumerate(corpus["cases"]):
        case_row = dict(raw_case)
        case_id = str(case_row.pop("case_id", "")).strip()
        operating_point = {
            **dict(corpus["shared_operating_point"]),
            "adapter_protocol": PROTOCOL_ID,
            **case_row,
        }
        case = build_benchmark_case_manifest(
            spec,
            case_id=f"{case_id}--{METHOD_ID}",
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class="rigid_pick_place",
            material_regime="synthetic_rigid_body_drop",
            operating_point=operating_point,
        )
        request = build_measurement_adapter_execution_request(
            descriptor,
            spec,
            case,
            execution_id=f"drake-rigid-contact-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id=SOLVER_SCOPE,
            precision="float64",
            seed=47,
            solver_settings={
                "discrete_contact_approximation": "sap",
                "contact_model": "point",
                "penetration_allowance_m": 0.001,
                "stiction_tolerance_m_s": 0.001,
            },
            timeout_seconds=60,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[str(python), str(WORKER_SCRIPT)],
            execute=execute,
        )
        bundles.append(bundle)
        prediction = bundle["prediction"]
        runtime = bundle["receipt"].get("runtime_observations", {})
        timestep = runtime.get("timestep_s")
        contact_step = runtime.get("first_contact_step")
        contact_time = (
            float(contact_step) * float(timestep)
            if isinstance(contact_step, int)
            and not isinstance(contact_step, bool)
            and isinstance(timestep, (int, float))
            and not isinstance(timestep, bool)
            else None
        )
        summaries.append(
            {
                "case_id": case_id,
                "method_case_id": case["case_id"],
                "body_shape": operating_point["body_shape"],
                "case_manifest_digest": case["case_manifest_digest"],
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "failure_codes": list(bundle["receipt"].get("failure_codes") or []),
                "deterministic_replay_match": (runtime.get("deterministic_replay_match") is True),
                "first_contact_time_s": contact_time,
                "final_position_m": runtime.get("final_position_m"),
                "scene_graph_renderer_used": runtime.get("scene_graph_renderer_used"),
                "drake_visualizer_used": runtime.get("drake_visualizer_used"),
                "engine_version": runtime.get("engine_version"),
                "contact_model": runtime.get("contact_model"),
                "discrete_contact_approximation": runtime.get("discrete_contact_approximation"),
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
        penetration = [
            _number(
                row["observed_metrics"]["penetration"],
                "geometry_contact_drake_penetration_metric_invalid",
            )
            for row in summaries
        ]
        contact_times = [
            _number(
                row["first_contact_time_s"],
                "geometry_contact_drake_contact_time_invalid",
            )
            for row in summaries
        ]
        aggregate_metrics = {
            "ground_contact_case_count": sum(
                row["observed_metrics"]["contact_sequence"] == "ground_contact" for row in summaries
            ),
            "maximum_penetration_m": max(penetration),
            "mean_penetration_m": fmean(penetration),
            "mean_first_contact_time_s": fmean(contact_times),
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "lane": "rigid_contact",
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
        "scene_graph_renderer_used": False,
        "drake_visualizer_used": False,
        "cases": summaries,
        "aggregate_metrics": aggregate_metrics,
        "execution_bundle_digests": [bundle["execution_bundle_digest"] for bundle in bundles],
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
        "qualification_labels_included": False,
        "instrumented_contact_included": False,
        "independent_execution": False,
        "r5_evidence": False,
        "r6_decision": False,
        "r7_admission": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "agent_may_promote": False,
    }
    suite["suite_digest"] = _digest(suite, "suite_digest")
    return validate_capture_to_geometry_contact_drake_development_suite(suite)


def validate_capture_to_geometry_contact_drake_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        errors.append("geometry_contact_drake_suite_schema_invalid")
    if suite.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("geometry_contact_drake_corpus_schema_invalid")
    for key, expected in (
        ("lane", "rigid_contact"),
        ("method_id", METHOD_ID),
        ("solver_scope", SOLVER_SCOPE),
        ("external_worker_python_required", True),
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("qualification_labels_included", False),
        ("instrumented_contact_included", False),
        ("independent_execution", False),
        ("scene_graph_renderer_used", False),
        ("drake_visualizer_used", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
        ("production_route_eligible", False),
        ("physical_success_established", False),
        ("agent_may_promote", False),
    ):
        if suite.get(key) != expected:
            errors.append(f"geometry_contact_drake_suite_{key}_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("geometry_contact_drake_suite_status_invalid")
    if not str(suite.get("worker_python_name", "")).strip():
        errors.append("geometry_contact_drake_suite_worker_python_name_invalid")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        errors.append("geometry_contact_drake_suite_cases_invalid")
    if suite.get("case_count") != len(cases or []):
        errors.append("geometry_contact_drake_suite_case_count_mismatch")
    if isinstance(cases, list):
        if any(row.get("scene_graph_renderer_used") is True for row in cases):
            errors.append("geometry_contact_drake_suite_renderer_case_invalid")
        if any(row.get("drake_visualizer_used") is True for row in cases):
            errors.append("geometry_contact_drake_suite_visualizer_case_invalid")
    if suite.get("status") == "completed_development_only":
        if suite.get("all_cases_completed") is not True:
            errors.append("geometry_contact_drake_suite_completion_invalid")
        if suite.get("all_replays_deterministic") is not True:
            errors.append("geometry_contact_drake_suite_replay_invalid")
        if (
            not isinstance(suite.get("aggregate_metrics"), Mapping)
            or not suite["aggregate_metrics"]
        ):
            errors.append("geometry_contact_drake_suite_metrics_missing")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"geometry_contact_drake_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("geometry_contact_drake_suite_digest_mismatch")
    if errors:
        raise _suite_error(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run the Drake Capture-to-Geometry-and-Contact corpus"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_capture_to_geometry_contact_drake_development_suite(
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
    "DRAKE_PYTHON_ENV",
    "GeometryContactDrakeDevelopmentSuiteError",
    "METHOD_ID",
    "SOLVER_SCOPE",
    "SUITE_SCHEMA_VERSION",
    "main",
    "run_capture_to_geometry_contact_drake_development_suite",
    "validate_capture_to_geometry_contact_drake_development_suite",
]
