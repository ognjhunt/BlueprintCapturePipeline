"""Local replay evaluator for the policy-autoresearch lane.

This evaluator is deliberately conservative: it can replay existing MuJoCo or
policy attempt evidence and score a candidate recipe against recorded failure
modes, but it does not claim to have rerun physics or executed a robot policy.
Use it for cheap local smoke tests before wiring a real simulator/controller
evaluator behind ``--evaluator-command``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_any, utc_now_iso, write_json


LOCAL_REPLAY_EVALUATOR_SCHEMA_VERSION = "policy_autoresearch_local_replay_evaluator.v1"

TRACE_CANDIDATE_NAMES = (
    "simulator_command_batch_attempt_trace.jsonl",
    "mujoco_batch_attempt_trace.jsonl",
    "policy_execution_trace.jsonl",
    "policy_execution_trace.json",
    "normalized_attempt_trace.json",
)

CLAIM_BOUNDARY = {
    "evaluator_kind": "local_replay_counterfactual",
    "uses_existing_attempt_evidence": True,
    "simulator_execution_performed": False,
    "robot_policy_execution_performed": False,
    "real_world_outcome_proven": False,
    "rank_fidelity_result_proven": False,
    "non_ranking_operational_claim_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_string(item) for item in value if _string(item)]
    return []


def _derive_policy_capabilities(recipe: Mapping[str, Any]) -> set[str]:
    params = _mapping(recipe.get("mutable_parameters") or recipe.get("mutableParameters"))
    capabilities: set[str] = set()
    planner = _string(params.get("planner")).lower()
    clearance_margin = _float(params.get("clearance_margin_m") or params.get("clearanceMarginM"))
    if planner in {"clearance_aware", "route_replan", "safety_margin"} and clearance_margin >= 0.15:
        capabilities.add("clearance_aware_navigation")
    if bool(params.get("dynamic_obstacle_yield") or params.get("dynamicObstacleYield")):
        capabilities.add("dynamic_obstacle_yield")
    if _int(params.get("perception_vote_count") or params.get("perceptionVoteCount"), 1) >= 2:
        capabilities.add("visual_recheck")
    if _int(params.get("retry_budget") or params.get("retryBudget"), 0) >= 1:
        capabilities.add("retry_recovery")
    if bool(
        params.get("grasp_alignment_correction") or params.get("graspAlignmentCorrection")
    ):
        capabilities.add("grasp_alignment_correction")
    return capabilities


def _failure_requires(mode: str) -> str | None:
    if mode in {
        "failure_clearance_near_miss",
        "failure_contact_collision",
        "failure_collision_probe_no_safe_pose",
    }:
        return "clearance_aware_navigation"
    if mode in {
        "failure_timeout",
        "failure_stuck",
        "failure_stuck_or_no_progress",
        "failure_policy_instability",
    }:
        return "retry_recovery"
    if mode in {"failure_dynamic_obstacle", "failure_safety_threshold_violation"}:
        return "dynamic_obstacle_yield"
    if mode == "failure_perception_uncertainty":
        return "visual_recheck"
    if mode == "failure_grasp_alignment":
        return "grasp_alignment_correction"
    return None


def _remaining_failure_modes(
    failure_modes: Sequence[str],
    capabilities: set[str],
) -> list[str]:
    remaining: list[str] = []
    for mode in failure_modes:
        required = _failure_requires(mode)
        if required is not None and required in capabilities:
            continue
        remaining.append(mode)

    route_recovery_ready = {
        "clearance_aware_navigation",
        "retry_recovery",
    }.issubset(capabilities)
    if route_recovery_ready:
        remaining = [
            mode
            for mode in remaining
            if mode not in {"failure_target_not_reached", "failure_endpoint_not_clean"}
        ]
    return sorted(set(remaining))


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
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    return _attempts_from_payload(read_json_any(path))


def _candidate_trace_paths(job_dir: Path | None) -> list[Path]:
    if job_dir is None:
        return []
    paths = [job_dir / name for name in TRACE_CANDIDATE_NAMES]
    paths.extend(
        [
            job_dir / "simulation_automation" / "mujoco_g1_simulator_command" / name
            for name in TRACE_CANDIDATE_NAMES
        ]
    )
    return paths


def _resolve_attempt_trace() -> Path | None:
    explicit = _string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE"))
    if explicit:
        path = Path(explicit).resolve()
        return path if path.is_file() else None

    job_dir_raw = _string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_JOB_DIR"))
    job_dir = Path(job_dir_raw).resolve() if job_dir_raw else None
    for path in _candidate_trace_paths(job_dir):
        if path.is_file():
            return path
    return None


def _failure_modes_from_attempt(attempt: Mapping[str, Any]) -> list[str]:
    modes = _string_list(attempt.get("failure_mode_ids"))
    if modes:
        return modes
    outcome = _mapping(attempt.get("task_outcome"))
    modes = _string_list(outcome.get("failure_mode_ids"))
    if modes:
        return modes
    metrics = _mapping(attempt.get("metrics"))
    inferred: list[str] = []
    if bool(metrics.get("clearance_threshold_violation")):
        inferred.append("failure_clearance_near_miss")
    if _int(metrics.get("timeout_count")) > 0:
        inferred.append("failure_timeout")
    if _int(metrics.get("stuck_event_count")) > 0:
        inferred.append("failure_stuck_or_no_progress")
    if bool(metrics.get("policy_instability_detected")):
        inferred.append("failure_policy_instability")
    if bool(metrics.get("endpoint_clean")) is False:
        inferred.append("failure_endpoint_not_clean")
    if bool(metrics.get("goal_reached")) is False:
        inferred.append("failure_target_not_reached")
    return sorted(set(inferred))


def _fallback_failure_modes(run: Mapping[str, Any], capabilities: set[str]) -> list[str]:
    required = set(_string_list(run.get("required_policy_capabilities")))
    text = " ".join(
        _string(run.get(key))
        for key in (
            "scenario_eval_run_id",
            "scenario_id",
            "variation_name",
            "task_id",
        )
    ).lower()
    if "blocked" in text or "narrow" in text or "clearance" in text:
        required.add("clearance_aware_navigation")
    if "dynamic" in text or "human" in text or "crossing" in text:
        required.add("dynamic_obstacle_yield")
    if "occlusion" in text or "glare" in text:
        required.add("visual_recheck")
    if "grasp" in text or "insertion" in text:
        required.add("grasp_alignment_correction")

    missing = sorted(required - capabilities)
    failures: list[str] = []
    for capability in missing:
        if capability == "clearance_aware_navigation":
            failures.append("failure_clearance_near_miss")
        elif capability == "dynamic_obstacle_yield":
            failures.append("failure_dynamic_obstacle")
        elif capability == "visual_recheck":
            failures.append("failure_perception_uncertainty")
        elif capability == "grasp_alignment_correction":
            failures.append("failure_grasp_alignment")
    return failures


def _safety_event_count(failure_modes: Sequence[str]) -> int:
    return sum(
        1
        for mode in failure_modes
        if mode in {"failure_dynamic_obstacle", "failure_safety_threshold_violation"}
    )


def _contact_event_count(failure_modes: Sequence[str]) -> int:
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


def _build_attempt(
    *,
    run: Mapping[str, Any],
    observed_attempt: Mapping[str, Any] | None,
    recipe: Mapping[str, Any],
    capabilities: set[str],
    source_attempt_trace_path: Path | None,
    generated_at: str,
) -> dict[str, Any]:
    observed = dict(observed_attempt or {})
    observed_success = bool(observed.get("task_success") or observed.get("success"))
    initial_modes = (
        _failure_modes_from_attempt(observed)
        if observed
        else _fallback_failure_modes(run, capabilities)
    )
    remaining_modes = [] if observed_success else _remaining_failure_modes(initial_modes, capabilities)
    task_success = observed_success or not remaining_modes
    run_id = _string(run.get("scenario_eval_run_id"))
    metrics = {
        **_mapping(observed.get("metrics")),
        "observed_task_success": observed_success,
        "counterfactual_task_success": task_success,
        "counterfactual_replay_from_existing_attempt": bool(observed),
        "simulator_execution_performed": False,
        "robot_policy_execution_performed": False,
        "safety_event_count": _safety_event_count(remaining_modes),
        "contact_event_count": _contact_event_count(remaining_modes),
        "remaining_failure_mode_count": len(remaining_modes),
    }
    return {
        "attempt_id": _string(observed.get("attempt_id")) or f"local_replay_{run_id}",
        "scenario_eval_run_id": run_id,
        "scenario_variation_instance_id": observed.get("scenario_variation_instance_id")
        or run.get("scenario_variation_instance_id"),
        "task_id": _string(observed.get("task_id")) or _string(run.get("task_id")),
        "scenario_id": _string(observed.get("scenario_id")) or _string(run.get("scenario_id")),
        "variation_name": observed.get("variation_name") or run.get("variation_name"),
        "policy_id": _string(recipe.get("policy_id")),
        "policy_kind": _string(recipe.get("policy_kind") or recipe.get("policyKind")),
        "status": "completed" if task_success else "failed_counterfactual_replay",
        "success": task_success,
        "task_success": task_success,
        "observed_task_success": observed_success,
        "counterfactual_replay": True,
        "source_attempt_trace_path": str(source_attempt_trace_path)
        if source_attempt_trace_path
        else None,
        "initial_failure_mode_ids": sorted(set(initial_modes)),
        "failure_mode_ids": remaining_modes,
        "metrics": metrics,
        "task_outcome": {
            **_mapping(observed.get("task_outcome")),
            "task_success": task_success,
            "failure_mode_ids": remaining_modes,
            "proof_boundary": (
                "Counterfactual replay over existing local attempt evidence. "
                "This did not rerun simulator physics or execute a robot policy."
            ),
        },
        "artifact_paths": _mapping(observed.get("artifact_paths") or observed.get("artifactPaths")),
        "generated_at": generated_at,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def run_local_replay_evaluator(
    *,
    recipe_path: str | Path,
    matrix_path: str | Path,
    output_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    recipe = _mapping(read_json_any(Path(recipe_path)))
    matrix = _mapping(read_json_any(Path(matrix_path)))
    runs = [dict(run) for run in matrix.get("runs", []) if isinstance(run, Mapping)]
    capabilities = _derive_policy_capabilities(recipe)
    trace_path = _resolve_attempt_trace()
    observed_attempts = _load_attempts(trace_path) if trace_path is not None else []
    observed_by_run_id = {
        _string(attempt.get("scenario_eval_run_id")): attempt
        for attempt in observed_attempts
        if _string(attempt.get("scenario_eval_run_id"))
    }
    attempts = [
        _build_attempt(
            run=run,
            observed_attempt=observed_by_run_id.get(_string(run.get("scenario_eval_run_id"))),
            recipe=recipe,
            capabilities=capabilities,
            source_attempt_trace_path=trace_path,
            generated_at=generated,
        )
        for run in runs
    ]
    payload = {
        "schema_version": LOCAL_REPLAY_EVALUATOR_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if attempts else "blocked_no_split_matrix_runs",
        "phase": _string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_PHASE")),
        "simulator_engine": _string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE")
        ),
        "frozen_verifier_sha256": _string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256")
        ),
        "source_attempt_trace_path": str(trace_path) if trace_path else None,
        "source_attempt_trace_found": trace_path is not None,
        "observed_attempt_count": len(observed_attempts),
        "split_run_count": len(runs),
        "derived_policy_capabilities": sorted(capabilities),
        "attempts": attempts,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    del argv
    required_env = [
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE",
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX",
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT",
    ]
    missing = [key for key in required_env if not _string(os.environ.get(key))]
    if missing:
        print(json.dumps({"status": "blocked_missing_env", "missing_env": missing}))
        return 2
    run_local_replay_evaluator(
        recipe_path=os.environ["BLUEPRINT_POLICY_AUTORESEARCH_RECIPE"],
        matrix_path=os.environ["BLUEPRINT_POLICY_AUTORESEARCH_MATRIX"],
        output_path=os.environ["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
