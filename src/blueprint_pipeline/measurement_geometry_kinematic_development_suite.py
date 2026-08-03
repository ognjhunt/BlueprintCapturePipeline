"""Plan or execute the Pinocchio/Coal Q-KIN development corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from .measurement_adapter_runtime import build_measurement_adapter_descriptor
from .measurement_pinocchio_coal_kinematic_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    implementation_digest,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


CORPUS_SCHEMA_VERSION = "capture_to_geometry_kinematic_development_corpus.v1"
SUITE_SCHEMA_VERSION = "capture_to_geometry_kinematic_development_suite.v1"
METHOD_ID = "exact-geometry-stack"
SOLVER_SCOPE = "pinocchio-analytic-two-link-coal-discrete-gjk"


class GeometryKinematicDevelopmentSuiteError(ValueError):
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


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeometryKinematicDevelopmentSuiteError(
            "geometry_kinematic_corpus_unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise GeometryKinematicDevelopmentSuiteError("geometry_kinematic_corpus_not_object")
    corpus = dict(value)
    errors: list[str] = []
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("geometry_kinematic_corpus_schema_invalid")
    if corpus.get("lane") != "planar_reach_discrete_collision":
        errors.append("geometry_kinematic_corpus_lane_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("captured_mesh_included", False),
        ("captured_registration_included", False),
        ("physical_measurements_included", False),
        ("qualification_labels_included", False),
        ("continuous_collision_evaluated", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
    ):
        if corpus.get(key) is not expected:
            errors.append(f"geometry_kinematic_corpus_{key}_invalid")
    shared = corpus.get("shared_operating_point")
    cases = corpus.get("cases")
    if not isinstance(shared, Mapping):
        errors.append("geometry_kinematic_corpus_shared_point_invalid")
    if (
        not isinstance(cases, list)
        or len(cases) < 3
        or not all(isinstance(row, Mapping) for row in cases)
    ):
        errors.append("geometry_kinematic_corpus_cases_invalid")
    elif len({str(row.get("case_id", "")).strip() for row in cases}) != len(cases):
        errors.append("geometry_kinematic_corpus_case_ids_duplicate")
    if errors:
        raise GeometryKinematicDevelopmentSuiteError(*errors)
    return corpus


def run_capture_to_geometry_kinematic_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    execute: bool = False,
) -> dict[str, Any]:
    if not _valid_digest(qualification_split_digest):
        raise GeometryKinematicDevelopmentSuiteError(
            "geometry_kinematic_qualification_split_digest_invalid"
        )
    if not _valid_digest(controller_scope_digest):
        raise GeometryKinematicDevelopmentSuiteError(
            "geometry_kinematic_controller_scope_digest_invalid"
        )
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise GeometryKinematicDevelopmentSuiteError("geometry_kinematic_split_leakage")
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-pinocchio-coal-planar-1",
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 90},
        minimum_repeated_trials=2,
    )
    descriptor = build_measurement_adapter_descriptor(METHOD_ID)
    bundles: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, raw_case in enumerate(corpus["cases"]):
        case_row = dict(raw_case)
        case_id = str(case_row.pop("case_id", "")).strip()
        expected_class = str(case_row.pop("expected_development_class", "")).strip()
        operating_point = {
            **dict(corpus["shared_operating_point"]),
            "adapter_protocol": PROTOCOL_ID,
            **case_row,
        }
        case = build_benchmark_case_manifest(
            spec,
            case_id=case_id,
            split="development",
            input_artifact_digests=[corpus_digest],
            task_class="static_reachability",
            material_regime="synthetic_rigid_geometry",
            operating_point=operating_point,
        )
        request = build_measurement_adapter_execution_request(
            descriptor,
            spec,
            case,
            execution_id=f"exact-geometry-{index + 1:03d}-{case_id}",
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            implementation_digest=implementation_digest(),
            backend_id="pinocchio-coal-cpu",
            precision="float64",
            seed=0,
            solver_settings={
                "inverse_kinematics": "analytic_two_link",
                "collision_query": "coal_gjk_signed_distance",
                "path_check": "finite_joint_interpolation",
                "continuous_collision": False,
            },
            timeout_seconds=45,
        )
        bundle = run_measurement_adapter_execution(
            request,
            command_argv=[
                sys.executable,
                "-m",
                "blueprint_pipeline.measurement_pinocchio_coal_kinematic_adapter",
            ],
            execute=execute,
        )
        bundles.append(bundle)
        runtime = bundle["receipt"].get("runtime_observations", {})
        prediction = bundle["prediction"]
        summaries.append(
            {
                "case_id": case_id,
                "expected_development_class": expected_class,
                "case_manifest_digest": case["case_manifest_digest"],
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "target_reachable": runtime.get("target_reachable"),
                "target_position_error_m": runtime.get("target_position_error_m"),
                "minimum_discrete_clearance_m": runtime.get("minimum_discrete_clearance_m"),
                "maximum_discrete_penetration_m": runtime.get("maximum_discrete_penetration_m"),
                "first_collision_sample": runtime.get("first_collision_sample"),
                "deterministic_replay_match": (runtime.get("deterministic_replay_match") is True),
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
    aggregate: dict[str, float | int] = {}
    if completed:
        errors = [
            float(row["target_position_error_m"])
            for row in summaries
            if isinstance(row["target_position_error_m"], (int, float))
            and not isinstance(row["target_position_error_m"], bool)
            and math.isfinite(float(row["target_position_error_m"]))
        ]
        aggregate = {
            "reachable_case_count": sum(row["target_reachable"] is True for row in summaries),
            "unreachable_case_count": sum(row["target_reachable"] is False for row in summaries),
            "discrete_collision_case_count": sum(
                row["first_collision_sample"] is not None for row in summaries
            ),
            "unsafe_case_count": sum(
                row["unsafe_condition_predicted"] is True for row in summaries
            ),
            "maximum_target_position_error_m": max(errors, default=0.0),
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "lane": "planar_reach_discrete_collision",
        "method_id": METHOD_ID,
        "solver_scope": SOLVER_SCOPE,
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
        "all_replays_deterministic": completed
        and all(row["deterministic_replay_match"] is True for row in summaries),
        "continuous_collision_evaluated": False,
        "cases": summaries,
        "aggregate_metrics": aggregate,
        "execution_bundle_digests": [bundle["execution_bundle_digest"] for bundle in bundles],
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "captured_mesh_included": False,
        "captured_registration_included": False,
        "physical_measurements_included": False,
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
    return validate_capture_to_geometry_kinematic_development_suite(suite)


def validate_capture_to_geometry_kinematic_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    for key, expected in (
        ("schema_version", SUITE_SCHEMA_VERSION),
        ("lane", "planar_reach_discrete_collision"),
        ("method_id", METHOD_ID),
        ("solver_scope", SOLVER_SCOPE),
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("captured_mesh_included", False),
        ("captured_registration_included", False),
        ("physical_measurements_included", False),
        ("qualification_labels_included", False),
        ("continuous_collision_evaluated", False),
        ("independent_execution", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
        ("production_route_eligible", False),
        ("physical_success_established", False),
        ("agent_may_promote", False),
    ):
        if suite.get(key) != expected:
            errors.append(f"geometry_kinematic_suite_{key}_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("geometry_kinematic_suite_status_invalid")
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        errors.append("geometry_kinematic_suite_cases_invalid")
    if suite.get("case_count") != len(cases or []):
        errors.append("geometry_kinematic_suite_case_count_mismatch")
    if suite.get("status") == "completed_development_only":
        if suite.get("all_cases_completed") is not True:
            errors.append("geometry_kinematic_suite_completion_invalid")
        if suite.get("all_replays_deterministic") is not True:
            errors.append("geometry_kinematic_suite_replay_invalid")
        if (
            not isinstance(suite.get("aggregate_metrics"), Mapping)
            or not suite["aggregate_metrics"]
        ):
            errors.append("geometry_kinematic_suite_metrics_missing")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"geometry_kinematic_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("geometry_kinematic_suite_digest_mismatch")
    if errors:
        raise GeometryKinematicDevelopmentSuiteError(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run the Pinocchio/Coal Q-KIN development corpus"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_capture_to_geometry_kinematic_development_suite(
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
    "GeometryKinematicDevelopmentSuiteError",
    "METHOD_ID",
    "SOLVER_SCOPE",
    "SUITE_SCHEMA_VERSION",
    "main",
    "run_capture_to_geometry_kinematic_development_suite",
    "validate_capture_to_geometry_kinematic_development_suite",
]
