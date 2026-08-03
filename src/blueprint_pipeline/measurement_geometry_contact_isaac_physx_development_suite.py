"""Plan or execute the Isaac Sim 6.0.1 CPU PhysX rigid-contact corpus."""

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
from .measurement_geometry_contact_development_suite import (
    CORPUS_SCHEMA_VERSION,
    GeometryContactDevelopmentSuiteError,
    _digest,
    _file_digest,
    _load_corpus,
    _number,
    _valid_digest,
)
from .measurement_isaac_physx_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    WORKER_SCRIPT,
    implementation_digest,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


SUITE_SCHEMA_VERSION = "capture_to_geometry_contact_isaac_physx_development_suite.v1"
METHOD_ID = "isaac-sim-6-physx"
SOLVER_SCOPE = "isaac-physx-cpu-tgs-rigid"
ISAAC_PYTHON_ENV = "BLUEPRINT_ISAAC_PYTHON"


class GeometryContactIsaacPhysxDevelopmentSuiteError(GeometryContactDevelopmentSuiteError):
    """Raised when the Isaac/PhysX development suite fails closed validation."""


def _suite_error(*codes: str) -> GeometryContactIsaacPhysxDevelopmentSuiteError:
    return GeometryContactIsaacPhysxDevelopmentSuiteError(*codes)


def _worker_launcher(
    value: str | Path | None,
    *,
    execute: bool,
) -> tuple[Path, bool]:
    explicit = str(value or os.environ.get(ISAAC_PYTHON_ENV) or "").strip()
    if execute and not explicit:
        raise _suite_error("geometry_contact_isaac_physx_exact_runtime_not_configured")
    path = Path(explicit).expanduser().absolute() if explicit else Path(sys.executable).absolute()
    if not path.is_file():
        raise _suite_error("geometry_contact_isaac_physx_worker_launcher_invalid")
    return path, bool(explicit)


def run_capture_to_geometry_contact_isaac_physx_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    worker_launcher: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Run every public rigid-drop case through an exact external Isaac runtime."""

    if not _valid_digest(qualification_split_digest):
        raise _suite_error("geometry_contact_isaac_physx_qualification_split_digest_invalid")
    if not _valid_digest(controller_scope_digest):
        raise _suite_error("geometry_contact_isaac_physx_controller_scope_digest_invalid")
    launcher, runtime_configured = _worker_launcher(worker_launcher, execute=execute)
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise _suite_error("geometry_contact_isaac_physx_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-isaac-physx-tgs-rigid-contact-1",
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 900},
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
            execution_id=f"isaac-physx-rigid-contact-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id=SOLVER_SCOPE,
            precision="float32",
            seed=47,
            solver_settings={
                "solver_type": "TGS",
                "broadphase_type": "SAP",
                "gpu_dynamics": False,
                "enhanced_determinism": True,
                "position_iterations": 8,
                "velocity_iterations": 2,
            },
            timeout_seconds=300,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[str(launcher), str(WORKER_SCRIPT)],
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
                "deterministic_replay_match": runtime.get("deterministic_replay_match") is True,
                "first_contact_time_s": contact_time,
                "final_position_m": runtime.get("final_position_m"),
                "contact_report_event_count": runtime.get("contact_report_event_count"),
                "renderer_used": runtime.get("renderer_used"),
                "rtx_sensor_used": runtime.get("rtx_sensor_used"),
                "engine_version": runtime.get("engine_version"),
                "solver_type": runtime.get("solver_type"),
                "broadphase_type": runtime.get("broadphase_type"),
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
                "geometry_contact_isaac_physx_penetration_metric_invalid",
            )
            for row in summaries
        ]
        contact_times = [
            _number(
                row["first_contact_time_s"],
                "geometry_contact_isaac_physx_contact_time_invalid",
            )
            for row in summaries
        ]
        aggregate_metrics = {
            "ground_contact_case_count": sum(
                row["observed_metrics"]["contact_sequence"] == "ground_contact"
                for row in summaries
            ),
            "contact_report_event_count": sum(
                int(row["contact_report_event_count"] or 0) for row in summaries
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
        "worker_launcher_name": launcher.name,
        "external_isaac_runtime_required": True,
        "external_isaac_runtime_configured": runtime_configured,
        "actual_isaac_execution_verified": completed,
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
        "renderer_used": False,
        "rtx_sensor_used": False,
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
    return validate_capture_to_geometry_contact_isaac_physx_development_suite(suite)


def validate_capture_to_geometry_contact_isaac_physx_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        errors.append("geometry_contact_isaac_physx_suite_schema_invalid")
    if suite.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("geometry_contact_isaac_physx_corpus_schema_invalid")
    for key, expected in (
        ("lane", "rigid_contact"),
        ("method_id", METHOD_ID),
        ("solver_scope", SOLVER_SCOPE),
        ("external_isaac_runtime_required", True),
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("qualification_labels_included", False),
        ("instrumented_contact_included", False),
        ("independent_execution", False),
        ("renderer_used", False),
        ("rtx_sensor_used", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
        ("production_route_eligible", False),
        ("physical_success_established", False),
        ("agent_may_promote", False),
    ):
        if suite.get(key) != expected:
            errors.append(f"geometry_contact_isaac_physx_suite_{key}_invalid")
    for key in (
        "external_isaac_runtime_configured",
        "actual_isaac_execution_verified",
        "execution_requested",
        "all_cases_completed",
        "all_replays_deterministic",
    ):
        if not isinstance(suite.get(key), bool):
            errors.append(f"geometry_contact_isaac_physx_suite_{key}_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("geometry_contact_isaac_physx_suite_status_invalid")
    if not str(suite.get("worker_launcher_name", "")).strip():
        errors.append("geometry_contact_isaac_physx_suite_worker_launcher_name_invalid")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        errors.append("geometry_contact_isaac_physx_suite_cases_invalid")
    if suite.get("case_count") != len(cases or []):
        errors.append("geometry_contact_isaac_physx_suite_case_count_mismatch")
    if isinstance(cases, list):
        if any(row.get("renderer_used") is True for row in cases):
            errors.append("geometry_contact_isaac_physx_suite_renderer_case_invalid")
        if any(row.get("rtx_sensor_used") is True for row in cases):
            errors.append("geometry_contact_isaac_physx_suite_rtx_sensor_case_invalid")
    if suite.get("status") == "completed_development_only":
        if suite.get("external_isaac_runtime_configured") is not True:
            errors.append("geometry_contact_isaac_physx_suite_runtime_configuration_invalid")
        if suite.get("actual_isaac_execution_verified") is not True:
            errors.append("geometry_contact_isaac_physx_suite_execution_verification_invalid")
        if suite.get("all_cases_completed") is not True:
            errors.append("geometry_contact_isaac_physx_suite_completion_invalid")
        if suite.get("all_replays_deterministic") is not True:
            errors.append("geometry_contact_isaac_physx_suite_replay_invalid")
        if not isinstance(suite.get("aggregate_metrics"), Mapping) or not suite["aggregate_metrics"]:
            errors.append("geometry_contact_isaac_physx_suite_metrics_missing")
    elif suite.get("actual_isaac_execution_verified") is not False:
        errors.append("geometry_contact_isaac_physx_suite_unverified_execution_invalid")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"geometry_contact_isaac_physx_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("geometry_contact_isaac_physx_suite_digest_mismatch")
    if errors:
        raise _suite_error(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run the Isaac/PhysX Capture-to-Geometry-and-Contact corpus"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--worker-launcher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_capture_to_geometry_contact_isaac_physx_development_suite(
        args.corpus,
        qualification_split_digest=args.qualification_split_digest,
        controller_scope_digest=args.controller_scope_digest,
        worker_launcher=args.worker_launcher,
        execute=args.execute,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if suite["status"] != "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GeometryContactIsaacPhysxDevelopmentSuiteError",
    "ISAAC_PYTHON_ENV",
    "METHOD_ID",
    "SOLVER_SCOPE",
    "SUITE_SCHEMA_VERSION",
    "main",
    "run_capture_to_geometry_contact_isaac_physx_development_suite",
    "validate_capture_to_geometry_contact_isaac_physx_development_suite",
]
