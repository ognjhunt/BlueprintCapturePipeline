"""Model-neutral world-model action-fidelity development adapter.

The adapter materializes strict numeric action-recovery checks from public
synthetic development steps and evaluates them with Blueprint's existing WAM
consistency contract.  It does not generate world-model output, see benchmark
labels, grade policy success, or upgrade the frozen policy-ranking thesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)
from .wam_action_consistency_contract import (
    cross_step_action_motion_replay_blockers,
    strict_action_consistency_blockers,
)


IMPLEMENTATION_ID = "blueprint-world-model-action-fidelity-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "world_model_action_fidelity.v1"
OUTCOME_BLOCKER = "wam_consistency_numeric_threshold_exceeded"


def implementation_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"world_model_fidelity_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"world_model_fidelity_{name}_invalid") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MeasurementAdapterExecutionError(f"world_model_fidelity_{name}_invalid")
    return result


def _vector(value: Any, *, name: str, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or not value or (length is not None and len(value) != length):
        raise MeasurementAdapterExecutionError(f"world_model_fidelity_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("world_model_fidelity_operating_point_invalid")
    point = dict(raw)
    for key, expected in {
        "adapter_protocol": PROTOCOL_ID,
        "data_origin": "synthetic_development_fixture",
        "claim_scope": "evaluator_support_only",
        "historical_policy_ranking_verdict": "thesis_not_supported",
        "policy_ranking_labels_included": False,
        "physical_outcomes_included": False,
        "provider_execution_included": False,
    }.items():
        if point.get(key) != expected:
            raise MeasurementAdapterExecutionError(f"world_model_fidelity_{key}_invalid")
    steps = point.get("action_steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 256:
        raise MeasurementAdapterExecutionError("world_model_fidelity_action_steps_invalid")
    dimension = point.get("action_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 64:
        raise MeasurementAdapterExecutionError("world_model_fidelity_action_dimension_invalid")
    units = point.get("action_units")
    if (
        not isinstance(units, list)
        or len(units) != dimension
        or any(not isinstance(item, str) or not item.strip() for item in units)
    ):
        raise MeasurementAdapterExecutionError("world_model_fidelity_action_units_invalid")
    checked_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise MeasurementAdapterExecutionError("world_model_fidelity_action_step_invalid")
        step = dict(raw_step)
        commanded = _vector(step.get("commanded_action"), name="commanded_action", length=dimension)
        recovered = _vector(step.get("recovered_action"), name="recovered_action", length=dimension)
        uncertainty = _vector(
            step.get("per_dimension_uncertainty"), name="uncertainty", length=dimension
        )
        if any(value < 0 for value in uncertainty):
            raise MeasurementAdapterExecutionError("world_model_fidelity_uncertainty_invalid")
        if step.get("step_index") != index:
            raise MeasurementAdapterExecutionError("world_model_fidelity_step_index_invalid")
        motion_identity = str(step.get("motion_identity", "")).strip()
        if not motion_identity:
            raise MeasurementAdapterExecutionError("world_model_fidelity_motion_identity_invalid")
        checked_steps.append(
            {
                "step_index": index,
                "commanded_action": commanded,
                "recovered_action": recovered,
                "per_dimension_uncertainty": uncertainty,
                "motion_identity": motion_identity,
                "sim_time_s": _number(step.get("sim_time_s"), name="sim_time", minimum=0.0),
                "forward_consistent": step.get("forward_consistent"),
                "inverse_consistent": step.get("inverse_consistent"),
            }
        )
        if not isinstance(step.get("forward_consistent"), bool) or not isinstance(
            step.get("inverse_consistent"), bool
        ):
            raise MeasurementAdapterExecutionError("world_model_fidelity_consistency_flag_invalid")
    return {
        "action_dimension": dimension,
        "action_units": units,
        "action_unit": str(point.get("action_unit", "")).strip(),
        "control_hz": _number(point.get("control_hz"), name="control_hz", minimum=1e-9),
        "maximum_abs_error": _number(
            point.get("maximum_abs_error"), name="maximum_abs_error", minimum=0.0
        ),
        "calibration_id": str(point.get("calibration_id", "")).strip(),
        "calibration_sha256": str(point.get("calibration_sha256", "")).strip(),
        "scorer_runtime_id": str(point.get("scorer_runtime_id", "")).strip(),
        "steps": checked_steps,
    }


def _evaluate(point: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not point["action_unit"]
        or not point["calibration_id"]
        or len(point["calibration_sha256"]) != 64
    ):
        raise MeasurementAdapterExecutionError("world_model_fidelity_identity_invalid")
    checks: list[dict[str, Any]] = []
    errors: list[float] = []
    outcome_failures: list[str] = []
    structural: list[str] = []
    period = 1.0 / point["control_hz"]
    for step in point["steps"]:
        commanded_sha = _sha(step["commanded_action"])
        recovered_sha = _sha(step["recovered_action"])
        per_error = [
            abs(a - b)
            for a, b in zip(step["commanded_action"], step["recovered_action"], strict=True)
        ]
        generated_motion_sha = hashlib.sha256(step["motion_identity"].encode()).hexdigest()
        timing = {
            "step_index": step["step_index"],
            "sim_time_s": step["sim_time_s"],
            "control_hz": point["control_hz"],
            "sample_period_seconds": period,
            "unit": "s",
        }
        expected = {
            "commanded_action_sha256": commanded_sha,
            "commanded_action_vector": step["commanded_action"],
            "action_dimension": point["action_dimension"],
            "action_unit": point["action_unit"],
            "action_units": point["action_units"],
            "action_timing": timing,
            "controller_fk_state_sha256": hashlib.sha256(
                f"controller:{step['step_index']}".encode()
            ).hexdigest(),
            "generated_state_sha256": hashlib.sha256(
                f"state:{step['motion_identity']}".encode()
            ).hexdigest(),
            "generated_motion_sha256": generated_motion_sha,
        }
        check = {
            "commanded_action_sha256": commanded_sha,
            "recovered_action": step["recovered_action"],
            "recovered_action_sha256": recovered_sha,
            "per_dimension_error": per_error,
            "per_dimension_uncertainty": step["per_dimension_uncertainty"],
            "threshold": {
                "max_abs_error": point["maximum_abs_error"],
                "unit": point["action_unit"],
            },
            "calibration_identity": {
                "calibration_id": point["calibration_id"],
                "sha256": point["calibration_sha256"],
            },
            "action_timing": timing,
            "action_units": point["action_units"],
            "controller_fk_state_sha256": expected["controller_fk_state_sha256"],
            "generated_state_sha256": expected["generated_state_sha256"],
            "generated_motion_sha256": generated_motion_sha,
            "scorer_runtime_id": point["scorer_runtime_id"],
            "provider_output_replay_used": False,
            "forward_consistent": step["forward_consistent"],
            "inverse_consistent": step["inverse_consistent"],
            "forward_result": {
                "passed": step["forward_consistent"],
                "method": "synthetic-development-forward-check",
            },
            "inverse_result": {
                "passed": step["inverse_consistent"],
                "method": "synthetic-development-inverse-check",
            },
            "evidence_refs": [f"synthetic-step-{step['step_index']}"],
            "termination_chunk": {
                "step_index": step["step_index"],
                "commanded_action_sha256": commanded_sha,
                "generated_motion_sha256": generated_motion_sha,
            },
        }
        blockers = strict_action_consistency_blockers(check, expected)
        outcome_failures.extend(code for code in blockers if code == OUTCOME_BLOCKER)
        structural.extend(code for code in blockers if code != OUTCOME_BLOCKER)
        checks.append(check)
        errors.extend(per_error)
    structural.extend(cross_step_action_motion_replay_blockers(checks))
    result = {
        "step_count": len(checks),
        "maximum_abs_error": max(errors, default=0.0),
        "mean_abs_error": sum(errors) / len(errors),
        "within_error_threshold": not outcome_failures,
        "forward_inverse_consistent": all(
            step["forward_consistent"] and step["inverse_consistent"] for step in point["steps"]
        ),
        "coverage": 1.0,
        "structural_blockers": sorted(set(structural)),
        "outcome_failures": sorted(set(outcome_failures)),
    }
    result["trace_digest"] = "sha256:" + _sha(result)
    return result


def run_world_model_action_fidelity_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    base = {
        "engine_version": "world-model-action-fidelity-contract.v1",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    implementation = request["implementation"]
    if implementation["implementation_id"] != IMPLEMENTATION_ID:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base,
            failure_codes=["world_model_fidelity_implementation_id_mismatch"],
        )
    if (
        implementation["implementation_version"] != IMPLEMENTATION_VERSION
        or implementation["implementation_digest"] != implementation_digest()
    ):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base,
            failure_codes=["world_model_fidelity_implementation_identity_mismatch"],
        )
    settings = dict(runtime["solver_settings"])
    if settings != {"protocol": PROTOCOL_ID, "replay_count": 2}:
        raise MeasurementAdapterExecutionError("world_model_fidelity_solver_settings_invalid")
    point = _operating_point(request)
    first = _evaluate(point)
    second = _evaluate(point)
    replay = first["trace_digest"] == second["trace_digest"]
    observations = {
        **base,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_digest": implementation_digest(),
        "adapter_protocol": PROTOCOL_ID,
        "solver_settings_digest": runtime["solver_settings_digest"],
        "step_count": first["step_count"],
        "maximum_abs_error": first["maximum_abs_error"],
        "mean_abs_error": first["mean_abs_error"],
        "within_error_threshold": first["within_error_threshold"],
        "forward_inverse_consistent": first["forward_inverse_consistent"],
        "coverage": first["coverage"],
        "trace_digest": first["trace_digest"],
        "repeat_trace_digest": second["trace_digest"],
        "deterministic_replay_match": replay,
        "model_output_generated_by_worker": False,
        "policy_ranking_scored": False,
        "physical_success_established": False,
        "historical_policy_ranking_verdict": "thesis_not_supported",
    }
    if first["structural_blockers"] or not replay:
        codes = first["structural_blockers"] or ["world_model_fidelity_replay_mismatch"]
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=codes,
        )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available: dict[str, Any] = {
        "action_recovery_max_abs_error": first["maximum_abs_error"],
        "forward_inverse_consistency": first["forward_inverse_consistent"],
        "action_motion_correlation": None,
        "policy_ranking_regret": None,
        "coverage": first["coverage"],
        "task_outcome": (
            "within_action_fidelity_envelope"
            if first["within_error_threshold"] and first["forward_inverse_consistent"]
            else "action_fidelity_envelope_exceeded"
        ),
    }
    metrics = {key: value for key, value in available.items() if key in requested}
    return build_measurement_adapter_worker_result(
        request,
        status="completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=None,
        runtime_observations=observations,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = json.loads(args.request.read_text(encoding="utf-8"))
    args.output.write_text(
        json.dumps(run_world_model_action_fidelity_request(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
