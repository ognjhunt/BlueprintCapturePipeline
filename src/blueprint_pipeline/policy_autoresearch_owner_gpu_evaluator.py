"""Owner-GPU evaluator adapter for policy autoresearch candidates.

This adapter is intentionally proof-boundary heavy. It can run an owner
simulator command through ``run_owner_gpu_proof`` and validate that the owner
system produced accepted simulator proof, but it still requires a policy attempt
trace to score task success against the frozen autoresearch matrix.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .owner_gpu_proof_runner import run_owner_gpu_proof


OWNER_GPU_POLICY_EVALUATOR_SCHEMA_VERSION = "policy_autoresearch_owner_gpu_evaluator.v1"

CLAIM_BOUNDARY = {
    "evaluator_kind": "owner_gpu_policy_attempt_execution",
    "source_scenario_eval_matrix_mutated": False,
    "owner_gpu_proof_required": True,
    "policy_attempt_trace_required_for_task_success": True,
    "robot_policy_execution_performed": False,
    "real_world_outcome_proven": False,
    "robot_readiness_proven": False,
    "safety_validation_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_string(item) for item in value if _string(item)]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _attempts_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        attempts = payload.get("attempts") or payload.get("results") or payload.get("episodes")
        if isinstance(attempts, list):
            return [dict(item) for item in attempts if isinstance(item, Mapping)]
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _load_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    return _attempts_from_payload(read_json_any(path))


def _runs_from_matrix(path: Path) -> list[dict[str, Any]]:
    matrix = _mapping(read_json_any(path))
    runs = matrix.get("runs")
    return [dict(run) for run in runs if isinstance(run, Mapping)] if isinstance(runs, list) else []


def _task_success(attempt: Mapping[str, Any]) -> bool:
    outcome = _mapping(attempt.get("task_outcome") or attempt.get("taskOutcome"))
    return bool(
        attempt.get("task_success")
        or attempt.get("taskSuccess")
        or attempt.get("success")
        or outcome.get("task_success")
        or outcome.get("success")
    )


def _failure_modes(attempt: Mapping[str, Any]) -> list[str]:
    direct = _string_list(attempt.get("failure_mode_ids") or attempt.get("failureModeIds"))
    if direct:
        return direct
    outcome = _mapping(attempt.get("task_outcome") or attempt.get("taskOutcome"))
    return _string_list(outcome.get("failure_mode_ids") or outcome.get("failureModeIds"))


def _contact_event_count(attempt: Mapping[str, Any], failure_modes: Sequence[str]) -> int:
    metrics = _mapping(attempt.get("metrics"))
    explicit = _int(
        metrics.get("contact_event_count")
        or metrics.get("robot_scene_contact_event_count")
        or metrics.get("near_miss_event_count")
        or metrics.get("collision_response_event_count")
    )
    if explicit:
        return explicit
    return sum(
        1
        for mode in failure_modes
        if mode
        in {
            "failure_clearance_near_miss",
            "failure_contact_collision",
            "failure_collision_probe_no_safe_pose",
        }
    )


def _safety_event_count(attempt: Mapping[str, Any], failure_modes: Sequence[str]) -> int:
    metrics = _mapping(attempt.get("metrics"))
    explicit = _int(
        metrics.get("safety_event_count")
        or metrics.get("fall_count")
        or metrics.get("unsafe_proximity_event_count")
    )
    if explicit:
        return explicit
    return sum(
        1
        for mode in failure_modes
        if mode in {"failure_dynamic_obstacle", "failure_safety_threshold_violation"}
    )


def _attempt_trace_path(output_path: Path) -> Path:
    explicit = _string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE"))
    if explicit:
        return Path(explicit).expanduser().resolve()
    return output_path.parent / "owner_gpu_policy_attempt_trace.json"


def _normalize_owner_attempts(
    *,
    runs: Sequence[Mapping[str, Any]],
    observed_attempts: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    simulator_engine: str,
    attempt_trace_path: Path,
    proof_result: Mapping[str, Any],
    validation: Mapping[str, Any],
    generated_at: str,
) -> list[dict[str, Any]]:
    proof_proven = bool(proof_result.get("owner_gpu_simulator_execution_proven"))
    observed_by_run_id = {
        _string(attempt.get("scenario_eval_run_id")): attempt
        for attempt in observed_attempts
        if _string(attempt.get("scenario_eval_run_id"))
    }
    policy_id = _string(recipe.get("candidate_id") or recipe.get("policy_id")) or "policy_candidate"
    normalized: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        run_id = _string(run.get("scenario_eval_run_id"))
        observed = observed_by_run_id.get(run_id, {})
        missing_trace = not bool(observed)
        modes = _failure_modes(observed)
        if missing_trace:
            modes = ["owner_gpu_policy_attempt_trace_missing"]
        if not proof_proven:
            modes = sorted(set([*modes, "owner_gpu_simulator_execution_not_proven"]))
        success = proof_proven and bool(observed) and _task_success(observed)
        if not success and not modes:
            modes = ["policy_task_not_successful"]
        metrics = {
            **_mapping(observed.get("metrics")),
            "simulator_execution_performed": proof_proven,
            "owner_gpu_simulator_execution_proven": proof_proven,
            "isaac_sim_execution_proven": bool(validation.get("isaac_sim_execution_proven")),
            "owner_gpu_default_policy_execution_proven": bool(
                validation.get("owner_gpu_default_policy_execution_proven")
            ),
            "owner_gpu_sim_robot_pov_evidence_proven": bool(
                validation.get("owner_gpu_sim_robot_pov_evidence_proven")
            ),
            "policy_attempt_trace_present": not missing_trace,
            "safety_event_count": _safety_event_count(observed, modes),
            "contact_event_count": _contact_event_count(observed, modes),
        }
        normalized.append(
            {
                **dict(observed),
                "attempt_id": _string(observed.get("attempt_id"))
                or f"{_safe_id(policy_id)}_{_safe_id(simulator_engine)}_{index:04d}",
                "scenario_eval_run_id": run_id,
                "scenario_variation_instance_id": observed.get(
                    "scenario_variation_instance_id"
                )
                or run.get("scenario_variation_instance_id"),
                "task_id": _string(observed.get("task_id")) or _string(run.get("task_id")),
                "scenario_id": _string(observed.get("scenario_id"))
                or _string(run.get("scenario_id")),
                "variation_name": observed.get("variation_name") or run.get("variation_name"),
                "simulator_engine": simulator_engine,
                "simulator_backend": validation.get("simulator_backend") or simulator_engine,
                "policy_id": policy_id,
                "policy_kind": _string(recipe.get("policy_kind") or recipe.get("policyKind")),
                "status": "completed" if success else "failed_owner_gpu_policy_attempt",
                "success": success,
                "task_success": success,
                "failure_mode_ids": [] if success else sorted(set(modes)),
                "metrics": metrics,
                "task_outcome": {
                    **_mapping(observed.get("task_outcome") or observed.get("taskOutcome")),
                    "task_success": success,
                    "failure_mode_ids": [] if success else sorted(set(modes)),
                },
                "artifact_paths": _mapping(
                    observed.get("artifact_paths") or observed.get("artifactPaths")
                ),
                "source_attempt_trace_path": str(attempt_trace_path),
                "generated_at": generated_at,
                "claim_boundary": {
                    **CLAIM_BOUNDARY,
                    "simulator_execution_performed": proof_proven,
                    "owner_gpu_simulator_execution_proven": proof_proven,
                    "isaac_sim_execution_proven": bool(
                        validation.get("isaac_sim_execution_proven")
                    ),
                },
            }
        )
    return normalized


def run_owner_gpu_policy_evaluator(
    *,
    recipe_path: str | Path,
    matrix_path: str | Path,
    output_path: str | Path,
    capture_root: str | Path,
    owner_command: str | None = None,
    simulator_engine: str = "isaac_sim",
    owner_system_id: str = "policy-autoresearch-owner-gpu",
    simulator_version: str = "owner-provided",
    gpu_model: str = "owner-provided",
    operator_id: str = "policy-autoresearch",
    operator_attestation: str = "Owner simulator command executed for policy autoresearch.",
    timeout_seconds: int = 1800,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    recipe = _mapping(read_json_any(Path(recipe_path)))
    runs = _runs_from_matrix(Path(matrix_path))
    resolved_output = Path(output_path).resolve()
    ensure_dir(resolved_output.parent)
    trace_path = _attempt_trace_path(resolved_output)
    ensure_dir(trace_path.parent)

    proof_result: dict[str, Any] = {
        "owner_gpu_simulator_execution_proven": False,
        "validation_blockers": ["owner_gpu_policy_evaluator_command_missing"],
    }
    validation: dict[str, Any] = {
        "status": "blocked",
        "simulator_backend": simulator_engine,
        "owner_gpu_simulator_execution_proven": False,
        "isaac_sim_execution_proven": False,
        "blockers": ["owner_gpu_policy_evaluator_command_missing"],
    }
    if _string(owner_command):
        proof_dir = resolved_output.parent / "owner_gpu_proof"
        proof_result = run_owner_gpu_proof(
            capture_root=capture_root,
            command=_string(owner_command),
            proof_dir=proof_dir,
            owner_system_id=owner_system_id,
            simulator_backend=simulator_engine,
            simulator_version=simulator_version,
            gpu_model=gpu_model,
            operator_id=operator_id,
            operator_attestation=operator_attestation,
            timeout_seconds=timeout_seconds,
            extra_env={
                "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE": str(Path(recipe_path).resolve()),
                "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX": str(Path(matrix_path).resolve()),
                "BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE": str(trace_path),
            },
        )
        validation_path = _string(proof_result.get("validation_manifest_path"))
        if validation_path and Path(validation_path).is_file():
            validation = _mapping(read_json_any(Path(validation_path)))

    observed_attempts = _load_attempts(trace_path)
    attempts = _normalize_owner_attempts(
        runs=runs,
        observed_attempts=observed_attempts,
        recipe=recipe,
        simulator_engine=simulator_engine,
        attempt_trace_path=trace_path,
        proof_result=proof_result,
        validation=validation,
        generated_at=generated,
    )
    proof_proven = bool(proof_result.get("owner_gpu_simulator_execution_proven"))
    task_trace_present = bool(observed_attempts)
    status = (
        "completed"
        if proof_proven and task_trace_present
        else "blocked_no_policy_attempt_trace"
        if proof_proven
        else "blocked_owner_gpu_simulator_execution_not_proven"
    )
    payload = {
        "schema_version": OWNER_GPU_POLICY_EVALUATOR_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "phase": _string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_PHASE")),
        "simulator_engine": simulator_engine,
        "simulator_backend": validation.get("simulator_backend") or simulator_engine,
        "frozen_verifier_sha256": _string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256")
        ),
        "policy_id": _string(recipe.get("candidate_id") or recipe.get("policy_id")),
        "owner_gpu_simulator_execution_proven": proof_proven,
        "isaac_sim_execution_proven": bool(validation.get("isaac_sim_execution_proven")),
        "policy_attempt_trace_path": str(trace_path),
        "policy_attempt_trace_present": task_trace_present,
        "owner_gpu_proof_result": proof_result,
        "owner_gpu_validation_manifest": validation,
        "attempts": attempts,
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "simulator_execution_performed": proof_proven,
            "owner_gpu_simulator_execution_proven": proof_proven,
            "isaac_sim_execution_proven": bool(validation.get("isaac_sim_execution_proven")),
        },
    }
    write_json(resolved_output, payload)
    return payload


def _env_path(name: str) -> Path | None:
    value = _string(os.environ.get(name))
    return Path(value).resolve() if value else None


def main(argv: list[str] | None = None) -> int:
    del argv
    recipe_path = _env_path("BLUEPRINT_POLICY_AUTORESEARCH_RECIPE")
    matrix_path = _env_path("BLUEPRINT_POLICY_AUTORESEARCH_MATRIX")
    output_path = _env_path("BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT")
    capture_root = _env_path("BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT")
    missing = []
    if recipe_path is None:
        missing.append("BLUEPRINT_POLICY_AUTORESEARCH_RECIPE")
    if matrix_path is None:
        missing.append("BLUEPRINT_POLICY_AUTORESEARCH_MATRIX")
    if output_path is None:
        missing.append("BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT")
    if capture_root is None:
        missing.append("BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT")
    if missing:
        print(json.dumps({"status": "blocked_missing_env", "missing_env": missing}))
        return 2
    run_owner_gpu_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        capture_root=capture_root,
        owner_command=os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_COMMAND"),
        simulator_engine=_string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE")
        )
        or "isaac_sim",
        owner_system_id=_string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_SYSTEM_ID"))
        or "policy-autoresearch-owner-gpu",
        simulator_version=_string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_SIMULATOR_VERSION")
        )
        or "owner-provided",
        gpu_model=_string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_GPU_MODEL"))
        or "owner-provided",
        operator_id=_string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OPERATOR_ID"))
        or "policy-autoresearch",
        operator_attestation=_string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OPERATOR_ATTESTATION")
        )
        or "Owner simulator command executed for policy autoresearch.",
        timeout_seconds=_int(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_OWNER_TIMEOUT_SECONDS"),
            1800,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
