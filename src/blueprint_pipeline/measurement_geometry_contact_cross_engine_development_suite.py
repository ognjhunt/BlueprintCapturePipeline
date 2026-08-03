"""Run the shared rigid-contact corpus through MuJoCo and Newton development ports."""

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
from .measurement_geometry_contact_development_suite import CORPUS_SCHEMA_VERSION
from .measurement_mujoco_adapter import (
    IMPLEMENTATION_ID as MUJOCO_IMPLEMENTATION_ID,
)
from .measurement_mujoco_adapter import (
    IMPLEMENTATION_VERSION as MUJOCO_IMPLEMENTATION_VERSION,
)
from .measurement_mujoco_adapter import PROTOCOL_ID as MUJOCO_PROTOCOL_ID
from .measurement_mujoco_adapter import implementation_digest as mujoco_digest
from .measurement_newton_rigid_adapter import (
    IMPLEMENTATION_ID as NEWTON_IMPLEMENTATION_ID,
)
from .measurement_newton_rigid_adapter import (
    IMPLEMENTATION_VERSION as NEWTON_IMPLEMENTATION_VERSION,
)
from .measurement_newton_rigid_adapter import PROTOCOL_ID as NEWTON_PROTOCOL_ID
from .measurement_newton_rigid_adapter import implementation_digest as newton_digest
from .measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


SUITE_SCHEMA_VERSION = "capture_to_geometry_contact_cross_engine_development_suite.v1"


class CrossEngineGeometryContactSuiteError(ValueError):
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
        raise CrossEngineGeometryContactSuiteError(code)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CrossEngineGeometryContactSuiteError(code) from exc
    if not math.isfinite(result):
        raise CrossEngineGeometryContactSuiteError(code)
    return result


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossEngineGeometryContactSuiteError(
            "cross_engine_geometry_contact_corpus_unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise CrossEngineGeometryContactSuiteError(
            "cross_engine_geometry_contact_corpus_not_object"
        )
    corpus = dict(value)
    errors: list[str] = []
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("cross_engine_geometry_contact_corpus_schema_invalid")
    if corpus.get("lane") != "rigid_contact":
        errors.append("cross_engine_geometry_contact_corpus_lane_invalid")
    shared = corpus.get("shared_operating_point")
    if not isinstance(shared, Mapping) or shared.get("protocol_family") != "rigid_body_drop":
        errors.append("cross_engine_geometry_contact_protocol_family_invalid")
    cases = corpus.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) < 2
        or not all(isinstance(row, Mapping) for row in cases)
    ):
        errors.append("cross_engine_geometry_contact_cases_invalid")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
        ("qualification_labels_included", False),
        ("instrumented_contact_included", False),
        ("r5_evidence", False),
        ("r6_decision", False),
        ("r7_admission", False),
    ):
        if corpus.get(key) is not expected:
            errors.append(f"cross_engine_geometry_contact_corpus_{key}_invalid")
    if errors:
        raise CrossEngineGeometryContactSuiteError(*errors)
    return corpus


def _method_configuration(method_id: str) -> dict[str, Any]:
    if method_id == "mujoco-3":
        return {
            "protocol_id": MUJOCO_PROTOCOL_ID,
            "implementation_id": MUJOCO_IMPLEMENTATION_ID,
            "implementation_version": MUJOCO_IMPLEMENTATION_VERSION,
            "implementation_digest": mujoco_digest(),
            "backend_id": "mujoco-cpu",
            "precision": "float64",
            "seed": 7,
            "solver_settings": {
                "integrator": "implicitfast",
                "solver": "Newton",
                "iterations": 100,
                "tolerance": 1e-10,
            },
            "timeout_seconds": 45,
            "module": "blueprint_pipeline.measurement_mujoco_adapter",
        }
    if method_id == "newton-1-4":
        return {
            "protocol_id": NEWTON_PROTOCOL_ID,
            "implementation_id": NEWTON_IMPLEMENTATION_ID,
            "implementation_version": NEWTON_IMPLEMENTATION_VERSION,
            "implementation_digest": newton_digest(),
            "backend_id": "newton-warp-cpu-xpbd",
            "precision": "float32",
            "seed": 41,
            "solver_settings": {
                "solver": "XPBD",
                "iterations": 10,
                "rigid_contact_relaxation": 0.8,
                "deterministic_mode": "RUN_TO_RUN",
            },
            "timeout_seconds": 120,
            "module": "blueprint_pipeline.measurement_newton_rigid_adapter",
        }
    raise CrossEngineGeometryContactSuiteError(
        f"cross_engine_geometry_contact_method_unknown:{method_id}"
    )


def run_capture_to_geometry_contact_cross_engine_development_suite(
    corpus_path: Path,
    *,
    qualification_split_digest: str,
    controller_scope_digest: str,
    execute: bool = False,
) -> dict[str, Any]:
    if not _valid_digest(qualification_split_digest):
        raise CrossEngineGeometryContactSuiteError(
            "cross_engine_geometry_contact_qualification_split_digest_invalid"
        )
    if not _valid_digest(controller_scope_digest):
        raise CrossEngineGeometryContactSuiteError(
            "cross_engine_geometry_contact_controller_scope_digest_invalid"
        )
    path = corpus_path.resolve()
    corpus = _load_corpus(path)
    corpus_digest = _file_digest(path)
    if qualification_split_digest == corpus_digest:
        raise CrossEngineGeometryContactSuiteError("cross_engine_geometry_contact_split_leakage")
    method_ids = ["mujoco-3", "newton-1-4"]
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-cross-engine-rigid-drop-1",
        method_ids=method_ids,
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 300},
        minimum_repeated_trials=2,
    )
    pair_rows: list[dict[str, Any]] = []
    bundle_digests: list[str] = []
    for case_index, raw_case in enumerate(corpus["cases"]):
        public_case = dict(raw_case)
        case_id = str(public_case.pop("case_id", "")).strip()
        pair_binding = {
            "corpus_digest": corpus_digest,
            "case_id": case_id,
            "shared_operating_point": corpus["shared_operating_point"],
            "case_operating_point": public_case,
        }
        case_pair_digest = _digest(pair_binding, "case_pair_digest")
        method_results: dict[str, dict[str, Any]] = {}
        for method_id in method_ids:
            config = _method_configuration(method_id)
            operating_point = {
                **dict(corpus["shared_operating_point"]),
                "adapter_protocol": config["protocol_id"],
                **public_case,
            }
            case = build_benchmark_case_manifest(
                spec,
                case_id=f"{case_id}--{method_id}",
                split="development",
                input_artifact_digests=[corpus_digest, case_pair_digest],
                task_class="rigid_pick_place",
                material_regime="synthetic_rigid_body_drop",
                operating_point=operating_point,
            )
            request = build_measurement_adapter_execution_request(
                build_measurement_adapter_descriptor(method_id),
                spec,
                case,
                execution_id=f"cross-engine-{case_index + 1:03d}-{method_id}",
                implementation_id=config["implementation_id"],
                implementation_version=config["implementation_version"],
                implementation_digest=config["implementation_digest"],
                backend_id=config["backend_id"],
                precision=config["precision"],
                seed=config["seed"],
                solver_settings=config["solver_settings"],
                timeout_seconds=config["timeout_seconds"],
            )
            bundle = run_measurement_adapter_execution(
                request,
                command_argv=[sys.executable, "-m", config["module"]],
                execute=execute,
            )
            bundle_digests.append(bundle["execution_bundle_digest"])
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
            method_results[method_id] = {
                "case_manifest_digest": case["case_manifest_digest"],
                "execution_bundle_digest": bundle["execution_bundle_digest"],
                "receipt_status": bundle["receipt"]["status"],
                "evidence_class": bundle["receipt"]["evidence_class"],
                "engine_version": runtime.get("engine_version"),
                "backend_id": runtime.get("backend_id"),
                "deterministic_replay_match": (runtime.get("deterministic_replay_match") is True),
                "first_contact_time_s": contact_time,
                "observed_metrics": (
                    dict(prediction["observed_metrics"]) if isinstance(prediction, Mapping) else {}
                ),
                "unsafe_condition_predicted": (
                    prediction["unsafe_condition_predicted"]
                    if isinstance(prediction, Mapping)
                    else None
                ),
            }
        completed = all(row["receipt_status"] == "completed" for row in method_results.values())
        deltas: dict[str, Any] = {}
        if completed:
            mujoco = method_results["mujoco-3"]
            newton = method_results["newton-1-4"]
            deltas = {
                "absolute_penetration_delta_m": abs(
                    _number(
                        mujoco["observed_metrics"]["penetration"],
                        "cross_engine_geometry_contact_mujoco_penetration_invalid",
                    )
                    - _number(
                        newton["observed_metrics"]["penetration"],
                        "cross_engine_geometry_contact_newton_penetration_invalid",
                    )
                ),
                "absolute_first_contact_time_delta_s": abs(
                    _number(
                        mujoco["first_contact_time_s"],
                        "cross_engine_geometry_contact_mujoco_contact_time_invalid",
                    )
                    - _number(
                        newton["first_contact_time_s"],
                        "cross_engine_geometry_contact_newton_contact_time_invalid",
                    )
                ),
                "contact_sequence_match": (
                    mujoco["observed_metrics"]["contact_sequence"]
                    == newton["observed_metrics"]["contact_sequence"]
                ),
                "unsafe_prediction_match": (
                    mujoco["unsafe_condition_predicted"] == newton["unsafe_condition_predicted"]
                ),
            }
        pair_rows.append(
            {
                "case_id": case_id,
                "case_pair_digest": case_pair_digest,
                "body_shape": public_case["body_shape"],
                "all_methods_completed": completed,
                "method_results": method_results,
                "cross_engine_deltas": deltas,
            }
        )
    completed = all(row["all_methods_completed"] is True for row in pair_rows)
    deterministic = completed and all(
        result["deterministic_replay_match"] is True
        for pair in pair_rows
        for result in pair["method_results"].values()
    )
    aggregates: dict[str, Any] = {}
    if completed:
        penetration_deltas = [
            _number(
                row["cross_engine_deltas"]["absolute_penetration_delta_m"],
                "cross_engine_geometry_contact_penetration_delta_invalid",
            )
            for row in pair_rows
        ]
        contact_time_deltas = [
            _number(
                row["cross_engine_deltas"]["absolute_first_contact_time_delta_s"],
                "cross_engine_geometry_contact_time_delta_invalid",
            )
            for row in pair_rows
        ]
        aggregates = {
            "maximum_absolute_penetration_delta_m": max(penetration_deltas),
            "mean_absolute_penetration_delta_m": fmean(penetration_deltas),
            "maximum_absolute_first_contact_time_delta_s": max(contact_time_deltas),
            "contact_sequence_match_count": sum(
                row["cross_engine_deltas"]["contact_sequence_match"] is True for row in pair_rows
            ),
            "unsafe_prediction_match_count": sum(
                row["cross_engine_deltas"]["unsafe_prediction_match"] is True for row in pair_rows
            ),
        }
    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus_digest,
        "benchmark_spec_digest": spec["benchmark_spec_digest"],
        "method_ids": method_ids,
        "execution_requested": execute is True,
        "status": (
            "completed_development_only"
            if completed
            else "planned_not_executed"
            if not execute
            else "incomplete"
        ),
        "case_pair_count": len(pair_rows),
        "method_execution_count": len(bundle_digests),
        "all_methods_completed": completed,
        "all_replays_deterministic": deterministic,
        "case_pairs": pair_rows,
        "aggregate_cross_engine_deltas": aggregates,
        "execution_bundle_digests": bundle_digests,
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
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
    return validate_capture_to_geometry_contact_cross_engine_development_suite(suite)


def validate_capture_to_geometry_contact_cross_engine_development_suite(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    suite = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        errors.append("cross_engine_geometry_contact_suite_schema_invalid")
    if suite.get("method_ids") != ["mujoco-3", "newton-1-4"]:
        errors.append("cross_engine_geometry_contact_suite_method_ids_invalid")
    if suite.get("status") not in {
        "planned_not_executed",
        "completed_development_only",
        "incomplete",
    }:
        errors.append("cross_engine_geometry_contact_suite_status_invalid")
    pairs = suite.get("case_pairs")
    if not isinstance(pairs, list) or len(pairs) < 2:
        errors.append("cross_engine_geometry_contact_suite_pairs_invalid")
    if suite.get("case_pair_count") != len(pairs or []):
        errors.append("cross_engine_geometry_contact_suite_pair_count_mismatch")
    if suite.get("method_execution_count") != 2 * len(pairs or []):
        errors.append("cross_engine_geometry_contact_suite_execution_count_mismatch")
    for key, expected in (
        ("development_only", True),
        ("synthetic_fixture", True),
        ("held_out", False),
        ("physical_measurements_included", False),
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
            errors.append(f"cross_engine_geometry_contact_suite_{key}_invalid")
    if suite.get("status") == "completed_development_only":
        if suite.get("all_methods_completed") is not True:
            errors.append("cross_engine_geometry_contact_suite_completion_invalid")
        if suite.get("all_replays_deterministic") is not True:
            errors.append("cross_engine_geometry_contact_suite_replay_invalid")
        if (
            not isinstance(suite.get("aggregate_cross_engine_deltas"), Mapping)
            or not suite["aggregate_cross_engine_deltas"]
        ):
            errors.append("cross_engine_geometry_contact_suite_deltas_missing")
    for key in ("corpus_digest", "benchmark_spec_digest", "suite_digest"):
        if not _valid_digest(suite.get(key)):
            errors.append(f"cross_engine_geometry_contact_suite_{key}_invalid")
    expected_digest = _digest(suite, "suite_digest")
    if suite.get("suite_digest") is not None and suite.get("suite_digest") != expected_digest:
        errors.append("cross_engine_geometry_contact_suite_digest_mismatch")
    if errors:
        raise CrossEngineGeometryContactSuiteError(*errors)
    suite["suite_digest"] = expected_digest
    return suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run the MuJoCo/Newton rigid-contact development corpus"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qualification-split-digest", required=True)
    parser.add_argument("--controller-scope-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    suite = run_capture_to_geometry_contact_cross_engine_development_suite(
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
    "CrossEngineGeometryContactSuiteError",
    "SUITE_SCHEMA_VERSION",
    "main",
    "run_capture_to_geometry_contact_cross_engine_development_suite",
    "validate_capture_to_geometry_contact_cross_engine_development_suite",
]
