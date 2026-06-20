"""Sim-only policy autoresearch lane for robot-eval jobs.

This module bridges Blueprint's bounded autoresearch pattern from skill text to
site policy recipes. It intentionally treats the scenario matrix and verifier as
immutable inputs: candidates may mutate only policy recipe parameters, and a
candidate is promoted only when heldout task success reaches the target while
safety/contact gates stay clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .wam_eval_substrate import WAM_EVALUATION_SUBSTRATES, normalize_evaluation_substrate


POLICY_AUTORESEARCH_REPORT_SCHEMA_VERSION = "policy_autoresearch_report.v1"
POLICY_AUTORESEARCH_VERIFIER_SCHEMA_VERSION = "policy_autoresearch_frozen_verifier.v1"
POLICY_AUTORESEARCH_EVAL_SCHEMA_VERSION = "policy_autoresearch_eval_result.v1"
POLICY_AUTORESEARCH_IDEA_TREE_SCHEMA_VERSION = "policy_autoresearch_agent_idea_tree.v1"
POLICY_AUTORESEARCH_CANDIDATE_PACKAGE_SCHEMA_VERSION = (
    "policy_autoresearch_candidate_package.v1"
)
POLICY_AUTORESEARCH_FOLLOWUP_REQUEST_SCHEMA_VERSION = (
    "policy_autoresearch_real_world_validation_followup_request.v1"
)

DEFAULT_OUTPUT_DIR_NAME = "policy_autoresearch"
DEFAULT_TARGET_SUCCESS_RATE = 1.0
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_AGENT_COUNT = 4
DEFAULT_HELDOUT_RATIO = 0.25
DEFAULT_SIMULATOR_ENGINES = ("mujoco",)
DEFAULT_PARALLEL_BRANCH_LIMIT = 4

FORBIDDEN_RECIPE_KEYS = {
    "reward",
    "reward_function",
    "rewardFunction",
    "success_classifier",
    "successClassifier",
    "verifier",
    "verifier_override",
    "verifierOverride",
    "scenario_eval_matrix",
    "scenarioEvalMatrix",
    "safety_thresholds",
    "safetyThresholds",
    "target_success_rate",
    "targetSuccessRate",
    "frozen_verifier_sha256",
}

CLAIM_BOUNDARY: dict[str, Any] = {
    "artifact_purpose": "policy_autoresearch_support",
    "evaluation_substrate_agnostic": True,
    "policy_recipe_mutation_only": True,
    "verifier_and_reward_frozen_before_mutation": True,
    "scenario_eval_matrix_is_immutable_eval_contract": True,
    "generated_wam_rollouts_are_model_derived_support_artifacts": True,
    "customer_specific_srcc_claimed": False,
    "customer_specific_srcc_requires_real_world_validation_rollouts": True,
    "passing_wam_heldout_eval_is_not_deployment_approval": True,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "real_world_outcome_proven": False,
    "robot_readiness_proven": False,
    "safety_validation_proven": False,
    "public_claim_upgrade_allowed": False,
}

ARTIFACT_PATHS = {
    "policy_autoresearch_report": "policy_autoresearch_report.json",
    "agent_idea_tree": "agent_idea_tree.json",
    "policy_candidate_package": "policy_candidate_package.json",
    "heldout_eval_result": "heldout_eval_result.json",
    "followup_real_world_validation_request": "followup_real_world_validation_request.json",
    "budget_ledger": "budget_ledger.json",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_string(item) for item in value if _string(item)]
    return []


def _engine_key(value: Any) -> str:
    return _safe_id(value, fallback="engine")


def _evaluation_substrate_for_engine(value: Any) -> str:
    text = _string(value)
    if not text:
        return ""
    try:
        return normalize_evaluation_substrate(text, simulator_engine=text, default=text)
    except ValueError:
        return _engine_key(text)


def _wam_substrate_requested(values: Sequence[str]) -> bool:
    return any(_evaluation_substrate_for_engine(value) in WAM_EVALUATION_SUBSTRATES for value in values)


def _requested_evaluation_substrate_cycle(values: Sequence[str] | None) -> list[str]:
    substrates: list[str] = []
    for value in values or ():
        text = _string(value)
        if text:
            substrates.append(normalize_evaluation_substrate(text))
    return substrates


def _parse_engine_evaluator_commands(values: Sequence[str] | None) -> dict[str, str]:
    commands: dict[str, str] = {}
    for value in values or []:
        text = _string(value)
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                "--evaluator-command-by-engine must use ENGINE=COMMAND, "
                f"got {text!r}"
            )
        engine, command = text.split("=", 1)
        engine_name = _string(engine)
        command_text = _string(command)
        if not engine_name or not command_text:
            raise ValueError(
                "--evaluator-command-by-engine requires non-empty engine and command"
            )
        commands[_engine_key(engine_name)] = command_text
    return commands


def _evaluator_command_for_engine(
    *,
    engine: str,
    evaluator_command: str | None,
    evaluator_commands_by_engine: Mapping[str, str] | None,
) -> str | None:
    commands = {
        _engine_key(key): value
        for key, value in (evaluator_commands_by_engine or {}).items()
        if _string(value)
    }
    return commands.get(_engine_key(engine)) or evaluator_command


def _payload_simulator_engines(payload: Any) -> list[str]:
    engines: set[str] = set()

    def add(value: Any) -> None:
        text = _string(value)
        if text:
            engines.add(_engine_key(text))

    if isinstance(payload, Mapping):
        add(payload.get("simulator_engine") or payload.get("simulatorEngine"))
        add(payload.get("simulator_backend") or payload.get("simulatorBackend"))
        add(payload.get("evaluation_substrate") or payload.get("evaluationSubstrate"))
        raw_attempts = payload.get("attempts") or payload.get("results") or payload.get("episodes")
    else:
        raw_attempts = payload
    if isinstance(raw_attempts, list):
        for raw in raw_attempts:
            if not isinstance(raw, Mapping):
                continue
            add(raw.get("simulator_engine") or raw.get("simulatorEngine"))
            add(raw.get("simulator_backend") or raw.get("simulatorBackend"))
            add(raw.get("evaluation_substrate") or raw.get("evaluationSubstrate"))
            metrics = _mapping(raw.get("metrics"))
            add(metrics.get("simulator_engine") or metrics.get("simulatorEngine"))
            add(metrics.get("evaluation_substrate") or metrics.get("evaluationSubstrate"))
            boundary = _mapping(raw.get("claim_boundary") or raw.get("claimBoundary"))
            add(boundary.get("simulator_engine") or boundary.get("simulatorEngine"))
            add(boundary.get("simulator_backend") or boundary.get("simulatorBackend"))
            add(boundary.get("evaluation_substrate") or boundary.get("evaluationSubstrate"))
    return sorted(engines)


def _external_payload_engine_mismatch(payload: Any, *, requested_engine: str) -> list[str]:
    observed = _payload_simulator_engines(payload)
    if not observed:
        return []
    requested = _engine_key(requested_engine)
    if requested in observed:
        return []
    return observed


def _proven_simulator_engines(*eval_results: Mapping[str, Any]) -> list[str]:
    engines: set[str] = set()
    for eval_result in eval_results:
        if not _eval_has_simulator_execution(eval_result):
            continue
        attempts = eval_result.get("attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            engine = _string(attempt.get("simulator_engine") or attempt.get("simulatorEngine"))
            if engine:
                engines.add(engine)
    return sorted(engines)


def _failure_capability_from_mode(mode: str) -> str | None:
    if mode in {"failure_clearance_near_miss", "failure_contact_collision"}:
        return "clearance_aware_navigation"
    if mode in {"failure_dynamic_obstacle", "failure_safety_threshold_violation"}:
        return "dynamic_obstacle_yield"
    if mode == "failure_perception_uncertainty":
        return "visual_recheck"
    if mode == "failure_grasp_alignment":
        return "grasp_alignment_correction"
    if mode in {"failure_timeout", "failure_stuck"}:
        return "retry_recovery"
    return None


def _infer_required_capabilities(run: Mapping[str, Any]) -> list[str]:
    explicit = _string_list(
        run.get("required_policy_capabilities") or run.get("requiredPolicyCapabilities")
    )
    if explicit:
        return sorted(set(explicit))

    text = " ".join(
        _string(run.get(key))
        for key in (
            "task_id",
            "taskId",
            "scenario_id",
            "scenarioId",
            "variation_name",
            "variationName",
            "scenario_eval_run_id",
            "scenarioEvalRunId",
        )
    ).lower()
    required: set[str] = set()
    if any(marker in text for marker in ("blocked", "obstacle", "narrow", "clearance")):
        required.add("clearance_aware_navigation")
    if any(marker in text for marker in ("human", "forklift", "crossing", "dynamic")):
        required.add("dynamic_obstacle_yield")
    if any(marker in text for marker in ("occlusion", "glare", "wrong_object", "missing_label")):
        required.add("visual_recheck")
    if any(marker in text for marker in ("grasp", "place", "insertion", "manipulation")):
        required.add("grasp_alignment_correction")
    return sorted(required)


def _derive_policy_capabilities(recipe: Mapping[str, Any]) -> list[str]:
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
    return sorted(capabilities)


def _recipe_with_capabilities(recipe: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(recipe))
    payload["derived_policy_capabilities"] = _derive_policy_capabilities(payload)
    return payload


def _find_forbidden_recipe_keys(value: Any, *, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FORBIDDEN_RECIPE_KEYS:
                findings.append(path)
            findings.extend(_find_forbidden_recipe_keys(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden_recipe_keys(child, prefix=f"{prefix}[{index}]"))
    return findings


def _load_matrix_runs(matrix_path: Path) -> list[dict[str, Any]]:
    payload = read_json_any(matrix_path)
    matrix = _mapping(payload)
    runs = matrix.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"scenario eval matrix at {matrix_path} does not contain runs[]")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(runs, start=1):
        run = _mapping(raw)
        run_id = _string(run.get("scenario_eval_run_id") or run.get("scenarioEvalRunId"))
        if not run_id:
            run_id = f"scenario_eval_run_{index:04d}"
        normalized.append(
            {
                **run,
                "scenario_eval_run_id": run_id,
                "scenario_variation_instance_id": _string(
                    run.get("scenario_variation_instance_id")
                    or run.get("scenarioVariationInstanceId")
                )
                or None,
                "task_id": _string(run.get("task_id") or run.get("taskId")),
                "scenario_id": _string(run.get("scenario_id") or run.get("scenarioId")),
                "variation_name": _string(run.get("variation_name") or run.get("variationName"))
                or None,
                "required_policy_capabilities": _infer_required_capabilities(run),
            }
        )
    return normalized


def _split_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    heldout_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    explicit_train = [
        dict(run)
        for run in runs
        if _string(run.get("split") or run.get("eval_split") or run.get("evalSplit")).lower()
        == "train"
    ]
    explicit_heldout = [
        dict(run)
        for run in runs
        if _string(run.get("split") or run.get("eval_split") or run.get("evalSplit")).lower()
        in {"heldout", "holdout", "validation"}
    ]
    if explicit_train and explicit_heldout:
        return explicit_train, explicit_heldout, "explicit_matrix_split"

    run_list = [dict(run) for run in runs]
    if len(run_list) <= 1:
        return run_list, run_list, "single_run_reused_as_heldout"
    heldout_count = max(1, round(len(run_list) * max(0.0, min(1.0, heldout_ratio))))
    heldout_count = min(len(run_list) - 1, heldout_count)
    train = run_list[:-heldout_count]
    heldout = run_list[-heldout_count:]
    return train, heldout, "deterministic_tail_holdout_split"


def _build_verifier_manifest(
    *,
    matrix_path: Path,
    reviewed_examples_path: Path | None,
    reviewed_examples_payload: Any,
    runs: Sequence[Mapping[str, Any]],
    train_runs: Sequence[Mapping[str, Any]],
    heldout_runs: Sequence[Mapping[str, Any]],
    split_source: str,
    target_success_rate: float,
    generated_at: str,
) -> dict[str, Any]:
    frozen_runs = [
        {
            "scenario_eval_run_id": _string(run.get("scenario_eval_run_id")),
            "scenario_variation_instance_id": run.get("scenario_variation_instance_id"),
            "task_id": _string(run.get("task_id")),
            "scenario_id": _string(run.get("scenario_id")),
            "variation_name": run.get("variation_name"),
            "split": "heldout"
            if _string(run.get("scenario_eval_run_id"))
            in {_string(item.get("scenario_eval_run_id")) for item in heldout_runs}
            else "train",
            "required_policy_capabilities": sorted(
                _string_list(run.get("required_policy_capabilities"))
            ),
            "max_safety_event_count": 0,
            "max_contact_event_count": 0,
        }
        for run in runs
    ]
    frozen_payload = {
        "scenario_eval_matrix_path": str(matrix_path),
        "reviewed_examples_path": str(reviewed_examples_path) if reviewed_examples_path else None,
        "reviewed_examples_payload": reviewed_examples_payload,
        "target_success_rate": round(float(target_success_rate), 6),
        "max_safety_event_count": 0,
        "max_contact_event_count": 0,
        "split_source": split_source,
        "runs": frozen_runs,
    }
    verifier_sha256 = _json_sha256(frozen_payload)
    return {
        "schema_version": POLICY_AUTORESEARCH_VERIFIER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "frozen",
        "verifier_sha256": verifier_sha256,
        "target_success_rate": round(float(target_success_rate), 6),
        "max_safety_event_count": 0,
        "max_contact_event_count": 0,
        "scenario_eval_matrix_path": str(matrix_path),
        "reviewed_examples_path": str(reviewed_examples_path) if reviewed_examples_path else None,
        "reviewed_examples_frozen": reviewed_examples_payload is not None,
        "reviewed_examples_payload": reviewed_examples_payload,
        "scenario_eval_run_count": len(runs),
        "train_run_count": len(train_runs),
        "heldout_run_count": len(heldout_runs),
        "split_source": split_source,
        "frozen_payload": frozen_payload,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _attempt_for_run(
    *,
    recipe: Mapping[str, Any],
    run: Mapping[str, Any],
    phase: str,
    engine: str,
    generated_at: str,
) -> dict[str, Any]:
    params = _mapping(recipe.get("mutable_parameters") or recipe.get("mutableParameters"))
    capabilities = set(_derive_policy_capabilities(recipe))
    required = set(_string_list(run.get("required_policy_capabilities")))
    missing = sorted(required - capabilities)
    max_speed = _float(params.get("max_speed_mps") or params.get("maxSpeedMps"), 0.0)
    clearance_margin = _float(params.get("clearance_margin_m") or params.get("clearanceMarginM"))
    failure_modes: list[str] = []
    safety_event_count = 0
    contact_event_count = 0

    for capability in missing:
        if capability == "clearance_aware_navigation":
            failure_modes.append("failure_clearance_near_miss")
            contact_event_count += 1
        elif capability == "dynamic_obstacle_yield":
            failure_modes.append("failure_dynamic_obstacle")
            safety_event_count += 1
        elif capability == "visual_recheck":
            failure_modes.append("failure_perception_uncertainty")
        elif capability == "grasp_alignment_correction":
            failure_modes.append("failure_grasp_alignment")
        elif capability == "retry_recovery":
            failure_modes.append("failure_stuck")

    if "dynamic_obstacle_yield" in required and max_speed > 0.65:
        if "failure_safety_threshold_violation" not in failure_modes:
            failure_modes.append("failure_safety_threshold_violation")
        safety_event_count += 1
    if "clearance_aware_navigation" in required and clearance_margin < 0.15:
        if "failure_contact_collision" not in failure_modes:
            failure_modes.append("failure_contact_collision")
        contact_event_count += 1

    success = not missing and safety_event_count == 0 and contact_event_count == 0
    final_target_error_m = 0.04 if success else round(0.35 + 0.05 * len(missing), 4)
    run_id = _string(run.get("scenario_eval_run_id"))
    policy_id = _string(recipe.get("policy_id") or recipe.get("policyId")) or "policy_candidate"
    return {
        "attempt_id": f"{_safe_id(policy_id)}_{phase}_{_safe_id(run_id)}",
        "scenario_eval_run_id": run_id,
        "scenario_variation_instance_id": run.get("scenario_variation_instance_id"),
        "task_id": _string(run.get("task_id")),
        "scenario_id": _string(run.get("scenario_id")),
        "variation_name": run.get("variation_name"),
        "policy_id": policy_id,
        "policy_kind": _string(recipe.get("policy_kind") or recipe.get("policyKind"))
        or "code_as_policy_heuristic",
        "simulator_engine": engine,
        "status": "completed" if success else "failed",
        "success": success,
        "task_success": success,
        "required_policy_capabilities": sorted(required),
        "derived_policy_capabilities": sorted(capabilities),
        "missing_required_policy_capabilities": missing,
        "failure_mode_ids": failure_modes,
        "failure_reason": "missing_or_unsafe_policy_capability" if failure_modes else None,
        "metrics": {
            "task_success": success,
            "final_target_error_m": final_target_error_m,
            "safety_event_count": safety_event_count,
            "contact_event_count": contact_event_count,
            "missing_required_capability_count": len(missing),
            "max_speed_mps": max_speed,
            "clearance_margin_m": clearance_margin,
        },
        "task_outcome": {
            "goal_reached": success,
            "endpoint_clean": success,
            "spawn_clean": True,
            "timeout": "failure_stuck" in failure_modes,
            "fall_detected": False,
            "stuck_detected": "failure_stuck" in failure_modes,
            "policy_instability_detected": False,
            "final_target_error_m": final_target_error_m,
            "goal_tolerance_m": 0.25,
            "safety_event_count": safety_event_count,
            "robot_scene_contact_event_count": contact_event_count,
            "success_criteria": {
                "all_required_policy_capabilities_present": not missing,
                "no_safety_threshold_events": safety_event_count == 0,
                "no_contact_collision_events": contact_event_count == 0,
                "goal_reached_within_tolerance": success,
            },
        },
        "generated_at": generated_at,
        "claim_boundary": "sim_only_policy_autoresearch_attempt_not_robot_readiness_proof",
    }


def _eval_result_from_attempts(
    *,
    recipe: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    phase: str,
    engine: str,
    generated_at: str,
    verifier_sha256: str,
    evaluator_command_used: bool,
    evaluator_detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_substrate = _evaluation_substrate_for_engine(engine)
    normalized_attempts = [dict(attempt) for attempt in attempts]
    success_count = sum(1 for attempt in normalized_attempts if bool(attempt.get("task_success") or attempt.get("success")))
    safety_event_count = sum(
        _int(_mapping(attempt.get("metrics")).get("safety_event_count"))
        for attempt in normalized_attempts
    )
    contact_event_count = sum(
        _int(_mapping(attempt.get("metrics")).get("contact_event_count"))
        for attempt in normalized_attempts
    )
    failed = [
        attempt
        for attempt in normalized_attempts
        if not bool(attempt.get("task_success") or attempt.get("success"))
    ]
    success_rate = round(success_count / len(normalized_attempts), 6) if normalized_attempts else 0.0
    covered_run_ids = sorted(
        _string(attempt.get("scenario_eval_run_id"))
        for attempt in normalized_attempts
        if _string(attempt.get("scenario_eval_run_id"))
    )
    return {
        "schema_version": POLICY_AUTORESEARCH_EVAL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if normalized_attempts else "blocked_missing_eval_runs",
        "phase": phase,
        "simulator_engine": engine,
        "evaluation_substrate": evaluation_substrate,
        "recipe_id": _string(recipe.get("candidate_id") or recipe.get("policy_id")),
        "policy_id": _string(recipe.get("policy_id")),
        "frozen_verifier_sha256": verifier_sha256,
        "evaluator_command_used": evaluator_command_used,
        "evaluator_detail": dict(evaluator_detail or {}),
        "attempt_count": len(normalized_attempts),
        "successful_task_attempt_count": success_count,
        "failed_task_attempt_count": len(normalized_attempts) - success_count,
        "task_success_summary": {
            "attempt_count": len(normalized_attempts),
            "successful_attempt_count": success_count,
            "failed_attempt_count": len(normalized_attempts) - success_count,
            "task_success_rate": success_rate,
        },
        "task_success_rate": success_rate,
        "safety_event_count": safety_event_count,
        "contact_event_count": contact_event_count,
        "safety_contact_gate_passed": safety_event_count == 0 and contact_event_count == 0,
        "covered_scenario_eval_run_ids": covered_run_ids,
        "failed_scenario_eval_run_ids": sorted(
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in failed
            if _string(attempt.get("scenario_eval_run_id"))
        ),
        "failure_mode_ids": sorted(
            {
                mode
                for attempt in failed
                for mode in _string_list(attempt.get("failure_mode_ids"))
            }
        ),
        "attempts": normalized_attempts,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "evaluation_substrate": evaluation_substrate,
            "wam_evaluation_substrate": evaluation_substrate in WAM_EVALUATION_SUBSTRATES,
            "simulator_execution_proven": _eval_has_simulator_execution(
                {"attempts": normalized_attempts}
            ),
        },
    }


def _normalize_external_attempts(
    *,
    payload: Any,
    recipe: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    phase: str,
    engine: str,
    generated_at: str,
) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        raw_attempts = payload.get("attempts") or payload.get("results") or payload.get("episodes")
    else:
        raw_attempts = payload
    if not isinstance(raw_attempts, list):
        raw_attempts = []
    payload_engine = ""
    payload_substrate = ""
    if isinstance(payload, Mapping):
        payload_engine = _string(
            payload.get("simulator_engine")
            or payload.get("simulatorEngine")
            or payload.get("simulator_backend")
            or payload.get("simulatorBackend")
        )
        payload_substrate = _string(
            payload.get("evaluation_substrate") or payload.get("evaluationSubstrate")
        )
    run_by_id = {_string(run.get("scenario_eval_run_id")): dict(run) for run in runs}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_attempts, start=1):
        if not isinstance(raw, Mapping):
            continue
        run_id = _string(raw.get("scenario_eval_run_id") or raw.get("scenarioEvalRunId"))
        run = run_by_id.get(run_id, {})
        metrics = _mapping(raw.get("metrics"))
        success = bool(raw.get("task_success") if "task_success" in raw else raw.get("success"))
        normalized.append(
            {
                "attempt_id": _string(raw.get("attempt_id") or raw.get("attemptId"))
                or f"{_safe_id(recipe.get('policy_id'))}_{phase}_external_{index:04d}",
                "scenario_eval_run_id": run_id,
                "scenario_variation_instance_id": raw.get("scenario_variation_instance_id")
                or raw.get("scenarioVariationInstanceId")
                or run.get("scenario_variation_instance_id"),
                "task_id": _string(raw.get("task_id") or raw.get("taskId")) or _string(run.get("task_id")),
                "scenario_id": _string(raw.get("scenario_id") or raw.get("scenarioId"))
                or _string(run.get("scenario_id")),
                "variation_name": raw.get("variation_name")
                or raw.get("variationName")
                or run.get("variation_name"),
                "policy_id": _string(raw.get("policy_id") or raw.get("policyId"))
                or _string(recipe.get("policy_id")),
                "policy_kind": _string(raw.get("policy_kind") or raw.get("policyKind"))
                or _string(recipe.get("policy_kind")),
                "simulator_engine": _string(
                    raw.get("simulator_engine")
                    or raw.get("simulatorEngine")
                    or raw.get("simulator_backend")
                    or raw.get("simulatorBackend")
                    or payload_engine
                )
                or engine,
                "evaluation_substrate": _string(
                    raw.get("evaluation_substrate")
                    or raw.get("evaluationSubstrate")
                    or payload_substrate
                )
                or _evaluation_substrate_for_engine(engine),
                "status": _string(raw.get("status")) or ("completed" if success else "failed"),
                "success": success,
                "task_success": success,
                "required_policy_capabilities": _string_list(
                    raw.get("required_policy_capabilities")
                    or run.get("required_policy_capabilities")
                ),
                "derived_policy_capabilities": _derive_policy_capabilities(recipe),
                "missing_required_policy_capabilities": _string_list(
                    raw.get("missing_required_policy_capabilities")
                ),
                "failure_mode_ids": _string_list(raw.get("failure_mode_ids")),
                "failure_reason": raw.get("failure_reason"),
                "metrics": metrics,
                "task_outcome": _mapping(raw.get("task_outcome")),
                "artifact_paths": _mapping(raw.get("artifact_paths") or raw.get("artifactPaths")),
                "observed_task_success": raw.get("observed_task_success"),
                "counterfactual_replay": bool(raw.get("counterfactual_replay")),
                "source_attempt_trace_path": raw.get("source_attempt_trace_path"),
                "initial_failure_mode_ids": _string_list(raw.get("initial_failure_mode_ids")),
                "generated_at": generated_at,
                "claim_boundary": raw.get("claim_boundary")
                or "external_policy_autoresearch_eval_output_not_robot_readiness_proof",
            }
        )
    return normalized


def _evaluate_recipe_with_command(
    *,
    recipe: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    phase: str,
    engine: str,
    generated_at: str,
    verifier_sha256: str,
    evaluator_command: str,
    evaluator_timeout_seconds: int,
    eval_root_dir: Path,
    source_capture_root: Path | None = None,
    source_job_dir: Path | None = None,
    source_matrix_path: Path | None = None,
    source_attempt_trace_path: Path | None = None,
) -> dict[str, Any]:
    recipe_id = _safe_id(recipe.get("candidate_id") or recipe.get("policy_id"))
    evaluation_substrate = _evaluation_substrate_for_engine(engine)
    run_dir = eval_root_dir / f"{phase}_{recipe_id}_{_safe_id(engine)}"
    ensure_dir(run_dir)
    recipe_path = run_dir / "policy_recipe.json"
    matrix_path = run_dir / "scenario_eval_matrix.json"
    output_path = run_dir / "evaluator_output.json"
    write_json(recipe_path, _recipe_with_capabilities(recipe))
    write_json(
        matrix_path,
        {
            "schema_version": "policy_autoresearch_split_matrix.v1",
            "phase": phase,
            "simulator_engine": engine,
            "evaluation_substrate": evaluation_substrate,
            "runs": [dict(run) for run in runs],
        },
    )
    env = {
        **os.environ,
        "BLUEPRINT_POLICY_AUTORESEARCH_RECIPE": str(recipe_path),
        "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX": str(matrix_path),
        "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT": str(output_path),
        "BLUEPRINT_POLICY_AUTORESEARCH_PHASE": phase,
        "BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE": engine,
        "BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE": evaluation_substrate,
        "BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256": verifier_sha256,
    }
    if source_capture_root is not None:
        env["BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT"] = str(source_capture_root)
    if source_job_dir is not None:
        env["BLUEPRINT_POLICY_AUTORESEARCH_JOB_DIR"] = str(source_job_dir)
    if source_matrix_path is not None:
        env["BLUEPRINT_POLICY_AUTORESEARCH_SOURCE_MATRIX"] = str(source_matrix_path)
    if source_attempt_trace_path is not None:
        env["BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE"] = str(source_attempt_trace_path)
    command = shlex.split(evaluator_command)
    command_started = time.monotonic()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(1, int(evaluator_timeout_seconds)),
        check=False,
        env=env,
    )
    evaluator_duration_seconds = round(time.monotonic() - command_started, 6)
    detail = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "input_recipe_path": str(recipe_path),
        "input_matrix_path": str(matrix_path),
        "output_path": str(output_path),
        "duration_seconds": evaluator_duration_seconds,
        "evaluation_substrate": evaluation_substrate,
        "source_capture_root": str(source_capture_root) if source_capture_root else None,
        "source_job_dir": str(source_job_dir) if source_job_dir else None,
        "source_matrix_path": str(source_matrix_path) if source_matrix_path else None,
        "source_attempt_trace_path": str(source_attempt_trace_path)
        if source_attempt_trace_path
        else None,
    }
    if completed.returncode != 0 or not output_path.is_file():
        return {
            **_eval_result_from_attempts(
                recipe=recipe,
                attempts=[],
                phase=phase,
                engine=engine,
                generated_at=generated_at,
                verifier_sha256=verifier_sha256,
                evaluator_command_used=True,
                evaluator_detail=detail,
            ),
            "status": "failed_evaluator_command",
            "failure_mode_ids": ["external_evaluator_command_failed"],
        }
    try:
        payload = read_json_any(output_path)
    except Exception as exc:
        return {
            **_eval_result_from_attempts(
                recipe=recipe,
                attempts=[],
                phase=phase,
                engine=engine,
                generated_at=generated_at,
                verifier_sha256=verifier_sha256,
                evaluator_command_used=True,
                evaluator_detail={**detail, "error": str(exc)},
            ),
            "status": "failed_evaluator_output_invalid",
            "failure_mode_ids": ["external_evaluator_output_invalid"],
        }
    mismatched_engines = _external_payload_engine_mismatch(payload, requested_engine=engine)
    if mismatched_engines:
        return {
            **_eval_result_from_attempts(
                recipe=recipe,
                attempts=[],
                phase=phase,
                engine=engine,
                generated_at=generated_at,
                verifier_sha256=verifier_sha256,
                evaluator_command_used=True,
                evaluator_detail={
                    **detail,
                    "error": "external_evaluator_engine_mismatch",
                    "requested_simulator_engine": engine,
                    "observed_simulator_engines": mismatched_engines,
                },
            ),
            "status": "failed_evaluator_engine_mismatch",
            "failure_mode_ids": ["external_evaluator_engine_mismatch"],
        }
    attempts = _normalize_external_attempts(
        payload=payload,
        recipe=recipe,
        runs=runs,
        phase=phase,
        engine=engine,
        generated_at=generated_at,
    )
    return _eval_result_from_attempts(
        recipe=recipe,
        attempts=attempts,
        phase=phase,
        engine=engine,
        generated_at=generated_at,
        verifier_sha256=verifier_sha256,
        evaluator_command_used=True,
        evaluator_detail=detail,
    )


def _evaluate_recipe(
    *,
    recipe: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    phase: str,
    engine: str,
    generated_at: str,
    verifier_sha256: str,
    evaluator_command: str | None = None,
    evaluator_timeout_seconds: int = 120,
    eval_root_dir: Path | None = None,
    source_capture_root: Path | None = None,
    source_job_dir: Path | None = None,
    source_matrix_path: Path | None = None,
    source_attempt_trace_path: Path | None = None,
) -> dict[str, Any]:
    if evaluator_command:
        return _evaluate_recipe_with_command(
            recipe=recipe,
            runs=runs,
            phase=phase,
            engine=engine,
            generated_at=generated_at,
            verifier_sha256=verifier_sha256,
            evaluator_command=evaluator_command,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            eval_root_dir=eval_root_dir or Path.cwd() / ".policy_autoresearch_evals",
            source_capture_root=source_capture_root,
            source_job_dir=source_job_dir,
            source_matrix_path=source_matrix_path,
            source_attempt_trace_path=source_attempt_trace_path,
        )
    attempts = [
        _attempt_for_run(
            recipe=recipe,
            run=run,
            phase=phase,
            engine=engine,
            generated_at=generated_at,
        )
        for run in runs
    ]
    return _eval_result_from_attempts(
        recipe=recipe,
        attempts=attempts,
        phase=phase,
        engine=engine,
        generated_at=generated_at,
        verifier_sha256=verifier_sha256,
        evaluator_command_used=False,
    )


def _apply_capability_mutation(recipe: Mapping[str, Any], capability: str) -> dict[str, Any]:
    candidate = deepcopy(dict(recipe))
    params = _mapping(candidate.get("mutable_parameters") or candidate.get("mutableParameters"))
    if capability == "clearance_aware_navigation":
        params["planner"] = "clearance_aware"
        params["clearance_margin_m"] = max(
            0.15,
            round(_float(params.get("clearance_margin_m"), 0.0) + 0.10, 3),
        )
    elif capability == "dynamic_obstacle_yield":
        params["dynamic_obstacle_yield"] = True
        params["max_speed_mps"] = min(_float(params.get("max_speed_mps"), 0.9), 0.55)
    elif capability == "visual_recheck":
        params["perception_vote_count"] = max(
            2,
            _int(params.get("perception_vote_count"), 1) + 1,
        )
    elif capability == "retry_recovery":
        params["retry_budget"] = max(1, _int(params.get("retry_budget"), 0) + 1)
    elif capability == "grasp_alignment_correction":
        params["grasp_alignment_correction"] = True
    candidate["mutable_parameters"] = params
    return candidate


def _mutation_capabilities_from_failures(
    eval_result: Mapping[str, Any],
    *,
    branch_index: int,
) -> list[str]:
    capabilities: list[str] = []
    for mode in _string_list(eval_result.get("failure_mode_ids")):
        capability = _failure_capability_from_mode(mode)
        if capability and capability not in capabilities:
            capabilities.append(capability)
    if not capabilities:
        capabilities = ["retry_recovery"]
    if branch_index == 0:
        return capabilities[:1]
    if branch_index == 1:
        return capabilities[:2]
    return capabilities


def _mutate_recipe(
    *,
    parent_recipe: Mapping[str, Any],
    parent_train_eval: Mapping[str, Any],
    iteration: int,
    branch_index: int,
    engine: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capabilities = _mutation_capabilities_from_failures(parent_train_eval, branch_index=branch_index)
    candidate = deepcopy(dict(parent_recipe))
    parent_policy_id = _string(parent_recipe.get("policy_id")) or "policy"
    for capability in capabilities:
        candidate = _apply_capability_mutation(candidate, capability)
    candidate["policy_id"] = f"{parent_policy_id}_i{iteration:02d}_a{branch_index + 1:02d}"
    candidate["candidate_id"] = candidate["policy_id"]
    candidate["mutation_parent_policy_id"] = parent_policy_id
    candidate["mutation_iteration"] = iteration
    candidate["mutation_agent_id"] = f"agent_{branch_index + 1:02d}"
    candidate["simulator_engine"] = engine
    candidate["evaluation_substrate"] = _evaluation_substrate_for_engine(engine)
    candidate = _recipe_with_capabilities(candidate)
    idea = {
        "idea_id": candidate["candidate_id"],
        "parent_policy_id": parent_policy_id,
        "iteration": iteration,
        "agent_id": candidate["mutation_agent_id"],
        "simulator_engine": engine,
        "evaluation_substrate": _evaluation_substrate_for_engine(engine),
        "hypothesis": (
            "Add policy capabilities observed in failed train attempts while leaving "
            "the frozen verifier and scenario matrix unchanged."
        ),
        "mutation_summary": {
            "added_or_strengthened_capabilities": capabilities,
            "mutable_parameters": candidate.get("mutable_parameters", {}),
        },
    }
    return candidate, idea


def _rank_eval(eval_result: Mapping[str, Any]) -> tuple[float, int, int]:
    return (
        _float(eval_result.get("task_success_rate")),
        1 if bool(eval_result.get("safety_contact_gate_passed")) else 0,
        -_int(eval_result.get("failed_task_attempt_count")),
    )


def _eval_has_simulator_execution(eval_result: Mapping[str, Any]) -> bool:
    attempts = eval_result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            return False
        metrics = _mapping(attempt.get("metrics"))
        boundary = _mapping(attempt.get("claim_boundary"))
        if not (
            metrics.get("simulator_execution_performed") is True
            or boundary.get("simulator_execution_performed") is True
        ):
            return False
    return True


def _estimated_branch_tokens(candidate_recipe: Mapping[str, Any], idea: Mapping[str, Any]) -> int:
    payload = {
        "candidate_id": candidate_recipe.get("candidate_id"),
        "mutation_summary": idea.get("mutation_summary"),
        "hypothesis": idea.get("hypothesis"),
        "recipe": candidate_recipe,
    }
    return max(1, len(json.dumps(payload, sort_keys=True, default=str)) // 4)


def _budget_limit_reached(ledger: Mapping[str, Any]) -> list[str]:
    limits = _mapping(ledger.get("limits"))
    usage = _mapping(ledger.get("usage"))
    blockers: list[str] = []
    max_candidates = limits.get("max_candidate_evaluations")
    if max_candidates is not None and _int(usage.get("candidate_evaluations")) >= _int(
        max_candidates
    ):
        blockers.append("candidate_evaluation_budget_exhausted")
    token_budget = limits.get("token_budget")
    if token_budget is not None and _int(usage.get("estimated_tokens")) >= _int(token_budget):
        blockers.append("estimated_token_budget_exhausted")
    compute_budget = limits.get("compute_seconds_budget")
    if compute_budget is not None and _float(usage.get("compute_seconds")) >= _float(
        compute_budget
    ):
        blockers.append("compute_seconds_budget_exhausted")
    wall_budget = limits.get("wall_time_budget_seconds")
    if wall_budget is not None and _float(usage.get("wall_time_seconds")) >= _float(wall_budget):
        blockers.append("wall_time_budget_exhausted")
    return blockers


def _update_budget_wall_time(ledger: dict[str, Any], *, start_monotonic: float) -> None:
    ledger["usage"]["wall_time_seconds"] = round(time.monotonic() - start_monotonic, 6)


def _record_eval_budget_usage(
    ledger: dict[str, Any],
    *,
    eval_result: Mapping[str, Any],
    phase: str,
    candidate_id: str,
    branch_id: str,
    start_monotonic: float,
) -> None:
    detail = _mapping(eval_result.get("evaluator_detail"))
    duration = _float(detail.get("duration_seconds"), 0.0)
    ledger["usage"]["compute_seconds"] = round(
        _float(ledger["usage"].get("compute_seconds")) + duration,
        6,
    )
    if phase == "train":
        ledger["usage"]["train_eval_count"] = _int(ledger["usage"].get("train_eval_count")) + 1
    if phase == "heldout":
        ledger["usage"]["heldout_eval_count"] = _int(
            ledger["usage"].get("heldout_eval_count")
        ) + 1
    _update_budget_wall_time(ledger, start_monotonic=start_monotonic)
    ledger["events"].append(
        {
            "event": "eval_completed",
            "phase": phase,
            "candidate_id": candidate_id,
            "branch_id": branch_id,
            "duration_seconds": duration,
            "task_success_rate": eval_result.get("task_success_rate"),
            "safety_contact_gate_passed": bool(eval_result.get("safety_contact_gate_passed")),
        }
    )


def _blocked_artifacts(
    *,
    output_dir: Path,
    generated_at: str,
    blockers: list[str],
    verifier_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    verifier = dict(verifier_manifest or {})
    if verifier:
        write_json(output_dir / "verifier_manifest.json", verifier)
    empty_eval = {
        "schema_version": POLICY_AUTORESEARCH_EVAL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "blockers": blockers,
        "task_success_summary": {
            "attempt_count": 0,
            "successful_attempt_count": 0,
            "failed_attempt_count": 0,
            "task_success_rate": 0.0,
        },
        "safety_contact_gate_passed": False,
        "attempts": [],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    idea_tree = {
        "schema_version": POLICY_AUTORESEARCH_IDEA_TREE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "nodes": [],
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    package = {
        "schema_version": POLICY_AUTORESEARCH_CANDIDATE_PACKAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "blockers": blockers,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    followup = {
        "schema_version": POLICY_AUTORESEARCH_FOLLOWUP_REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "not_ready",
        "blockers": blockers,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    report = {
        "schema_version": POLICY_AUTORESEARCH_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "target_success_reached": False,
        "blockers": blockers,
        "artifact_paths": dict(ARTIFACT_PATHS),
        "support_artifact_paths": {"verifier_manifest": "verifier_manifest.json"} if verifier else {},
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    budget_ledger = {
        "schema_version": "policy_autoresearch_budget_ledger.v1",
        "generated_at": generated_at,
        "status": "blocked_before_loop",
        "limits": {},
        "usage": {
            "estimated_tokens": 0,
            "compute_seconds": 0.0,
            "wall_time_seconds": 0.0,
            "candidate_evaluations": 0,
            "train_eval_count": 0,
            "heldout_eval_count": 0,
        },
        "events": [{"event": "blocked_before_loop", "blockers": list(blockers)}],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / ARTIFACT_PATHS["heldout_eval_result"], empty_eval)
    write_json(output_dir / ARTIFACT_PATHS["agent_idea_tree"], idea_tree)
    write_json(output_dir / ARTIFACT_PATHS["policy_candidate_package"], package)
    write_json(output_dir / ARTIFACT_PATHS["followup_real_world_validation_request"], followup)
    write_json(output_dir / ARTIFACT_PATHS["budget_ledger"], budget_ledger)
    write_json(output_dir / ARTIFACT_PATHS["policy_autoresearch_report"], report)
    return {
        "report": report,
        "verifier_manifest": verifier,
        "heldout_eval_result": empty_eval,
        "agent_idea_tree": idea_tree,
        "policy_candidate_package": package,
        "followup_real_world_validation_request": followup,
        "budget_ledger": budget_ledger,
    }


def run_policy_autoresearch(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    policy_recipe_path: str | Path,
    scenario_eval_matrix_path: str | Path | None = None,
    reviewed_examples_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    agent_count: int = DEFAULT_AGENT_COUNT,
    target_success_rate: float = DEFAULT_TARGET_SUCCESS_RATE,
    heldout_ratio: float = DEFAULT_HELDOUT_RATIO,
    simulator_engines: Sequence[str] = DEFAULT_SIMULATOR_ENGINES,
    evaluation_substrates: Sequence[str] | None = None,
    evaluator_command: str | None = None,
    evaluator_commands_by_engine: Mapping[str, str] | None = None,
    evaluator_timeout_seconds: int = 120,
    evaluator_attempt_trace_path: str | Path | None = None,
    token_budget: int | None = None,
    compute_seconds_budget: float | None = None,
    wall_time_budget_seconds: float | None = None,
    max_candidate_evaluations: int | None = None,
    parallel_branch_limit: int = DEFAULT_PARALLEL_BRANCH_LIMIT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run bounded policy autoresearch for one robot-eval job."""

    generated = generated_at or utc_now_iso()
    started_monotonic = time.monotonic()
    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    matrix_path = (
        Path(scenario_eval_matrix_path).resolve()
        if scenario_eval_matrix_path is not None
        else resolved_job_dir / "scenario_eval_matrix.json"
    )
    recipe_path = Path(policy_recipe_path).resolve()
    examples_path = Path(reviewed_examples_path).resolve() if reviewed_examples_path else None
    attempt_trace_path = (
        Path(evaluator_attempt_trace_path).resolve() if evaluator_attempt_trace_path else None
    )
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else resolved_job_dir / DEFAULT_OUTPUT_DIR_NAME
    )
    ensure_dir(resolved_output_dir)
    budget_ledger: dict[str, Any] = {
        "schema_version": "policy_autoresearch_budget_ledger.v1",
        "generated_at": generated,
        "status": "tracking",
        "limits": {
            "token_budget": int(token_budget) if token_budget is not None else None,
            "compute_seconds_budget": (
                round(float(compute_seconds_budget), 6)
                if compute_seconds_budget is not None
                else None
            ),
            "wall_time_budget_seconds": (
                round(float(wall_time_budget_seconds), 6)
                if wall_time_budget_seconds is not None
                else None
            ),
            "max_candidate_evaluations": (
                int(max_candidate_evaluations)
                if max_candidate_evaluations is not None
                else None
            ),
            "parallel_branch_limit": max(1, int(parallel_branch_limit)),
        },
        "usage": {
            "estimated_tokens": 0,
            "compute_seconds": 0.0,
            "wall_time_seconds": 0.0,
            "candidate_evaluations": 0,
            "train_eval_count": 0,
            "heldout_eval_count": 0,
        },
        "events": [],
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "estimated_tokens_are_branch_planning_proxy": True,
            "compute_seconds_are_local_evaluator_wall_time_proxy": True,
        },
    }

    blockers: list[str] = []
    if not capture_path.exists():
        blockers.append("capture_root_missing")
    if not matrix_path.is_file():
        blockers.append("scenario_eval_matrix_missing")
    if not recipe_path.is_file():
        blockers.append("policy_recipe_missing")
    if examples_path is not None and not examples_path.is_file():
        blockers.append("reviewed_examples_missing")
    if attempt_trace_path is not None and not attempt_trace_path.is_file():
        blockers.append("evaluator_attempt_trace_missing")
    if blockers:
        return _blocked_artifacts(
            output_dir=resolved_output_dir,
            generated_at=generated,
            blockers=blockers,
        )

    try:
        requested_substrate_cycle = _requested_evaluation_substrate_cycle(evaluation_substrates)
    except ValueError as exc:
        return _blocked_artifacts(
            output_dir=resolved_output_dir,
            generated_at=generated,
            blockers=[f"unsupported_evaluation_substrate:{exc}"],
        )

    try:
        runs = _load_matrix_runs(matrix_path)
    except Exception as exc:
        return _blocked_artifacts(
            output_dir=resolved_output_dir,
            generated_at=generated,
            blockers=[f"scenario_eval_matrix_invalid:{exc}"],
        )
    if not runs:
        return _blocked_artifacts(
            output_dir=resolved_output_dir,
            generated_at=generated,
            blockers=["scenario_eval_matrix_empty"],
        )

    train_runs, heldout_runs, split_source = _split_runs(runs, heldout_ratio=heldout_ratio)
    reviewed_examples_payload = read_json_any(examples_path) if examples_path is not None else None
    verifier_manifest = _build_verifier_manifest(
        matrix_path=matrix_path,
        reviewed_examples_path=examples_path,
        reviewed_examples_payload=reviewed_examples_payload,
        runs=runs,
        train_runs=train_runs,
        heldout_runs=heldout_runs,
        split_source=split_source,
        target_success_rate=target_success_rate,
        generated_at=generated,
    )
    write_json(resolved_output_dir / "verifier_manifest.json", verifier_manifest)
    verifier_sha256 = _string(verifier_manifest.get("verifier_sha256"))

    seed_recipe_raw = _mapping(read_json_any(recipe_path))
    forbidden_keys = _find_forbidden_recipe_keys(seed_recipe_raw)
    if forbidden_keys:
        return _blocked_artifacts(
            output_dir=resolved_output_dir,
            generated_at=generated,
            blockers=["forbidden_policy_recipe_keys"],
            verifier_manifest=verifier_manifest
            | {"forbidden_policy_recipe_key_paths": forbidden_keys},
        )

    seed_recipe = _recipe_with_capabilities(seed_recipe_raw)
    engine_cycle = (
        requested_substrate_cycle
        if evaluation_substrates is not None
        else list(simulator_engines or DEFAULT_SIMULATOR_ENGINES)
    )
    if not engine_cycle:
        engine_cycle = list(DEFAULT_SIMULATOR_ENGINES)
    evaluation_substrate_cycle = [_evaluation_substrate_for_engine(engine) for engine in engine_cycle]
    wam_substrate_requested = _wam_substrate_requested(evaluation_substrate_cycle)
    engine = engine_cycle[0]
    baseline_evaluator_command = _evaluator_command_for_engine(
        engine=engine,
        evaluator_command=evaluator_command,
        evaluator_commands_by_engine=evaluator_commands_by_engine,
    )
    baseline_train = _evaluate_recipe(
        recipe=seed_recipe,
        runs=train_runs,
        phase="train",
        engine=engine,
        generated_at=generated,
        verifier_sha256=verifier_sha256,
        evaluator_command=baseline_evaluator_command,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        eval_root_dir=resolved_output_dir / "evaluator_runs",
        source_capture_root=capture_path,
        source_job_dir=resolved_job_dir,
        source_matrix_path=matrix_path,
        source_attempt_trace_path=attempt_trace_path,
    )
    baseline_heldout = _evaluate_recipe(
        recipe=seed_recipe,
        runs=heldout_runs,
        phase="heldout",
        engine=engine,
        generated_at=generated,
        verifier_sha256=verifier_sha256,
        evaluator_command=baseline_evaluator_command,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        eval_root_dir=resolved_output_dir / "evaluator_runs",
        source_capture_root=capture_path,
        source_job_dir=resolved_job_dir,
        source_matrix_path=matrix_path,
        source_attempt_trace_path=attempt_trace_path,
    )
    _record_eval_budget_usage(
        budget_ledger,
        eval_result=baseline_train,
        phase="train",
        candidate_id=_string(seed_recipe.get("policy_id")) or "seed_policy",
        branch_id="seed",
        start_monotonic=started_monotonic,
    )
    _record_eval_budget_usage(
        budget_ledger,
        eval_result=baseline_heldout,
        phase="heldout",
        candidate_id=_string(seed_recipe.get("policy_id")) or "seed_policy",
        branch_id="seed",
        start_monotonic=started_monotonic,
    )

    best_recipe = seed_recipe
    best_train = baseline_train
    best_heldout = baseline_heldout
    idea_nodes: list[dict[str, Any]] = [
        {
            "idea_id": _string(seed_recipe.get("policy_id")) or "seed_policy",
            "parent_policy_id": None,
            "iteration": 0,
            "agent_id": "seed",
            "simulator_engine": engine,
            "evaluation_substrate": evaluation_substrate_cycle[0],
            "hypothesis": "Seed policy recipe supplied by the job owner.",
            "train_success_rate": baseline_train["task_success_rate"],
            "heldout_success_rate": baseline_heldout["task_success_rate"],
            "accepted_for_next_iteration": True,
            "promoted_candidate": False,
        }
    ]
    iteration_records: list[dict[str, Any]] = []

    budget_stop_reasons: list[str] = []

    def evaluate_candidate_branch(
        *,
        candidate_recipe: dict[str, Any],
        idea: dict[str, Any],
        candidate_engine: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        branch_evaluator_command = _evaluator_command_for_engine(
            engine=candidate_engine,
            evaluator_command=evaluator_command,
            evaluator_commands_by_engine=evaluator_commands_by_engine,
        )
        candidate_train = _evaluate_recipe(
            recipe=candidate_recipe,
            runs=train_runs,
            phase="train",
            engine=candidate_engine,
            generated_at=generated,
            verifier_sha256=verifier_sha256,
            evaluator_command=branch_evaluator_command,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            eval_root_dir=resolved_output_dir / "evaluator_runs",
            source_capture_root=capture_path,
            source_job_dir=resolved_job_dir,
            source_matrix_path=matrix_path,
            source_attempt_trace_path=attempt_trace_path,
        )
        candidate_heldout = _evaluate_recipe(
            recipe=candidate_recipe,
            runs=heldout_runs,
            phase="heldout",
            engine=candidate_engine,
            generated_at=generated,
            verifier_sha256=verifier_sha256,
            evaluator_command=branch_evaluator_command,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            eval_root_dir=resolved_output_dir / "evaluator_runs",
            source_capture_root=capture_path,
            source_job_dir=resolved_job_dir,
            source_matrix_path=matrix_path,
            source_attempt_trace_path=attempt_trace_path,
        )
        return candidate_recipe, candidate_train, candidate_heldout, idea

    for iteration in range(1, max(0, int(max_iterations)) + 1):
        _update_budget_wall_time(budget_ledger, start_monotonic=started_monotonic)
        budget_stop_reasons = _budget_limit_reached(budget_ledger)
        if budget_stop_reasons:
            budget_ledger["events"].append(
                {
                    "event": "iteration_not_started_budget_exhausted",
                    "iteration": iteration,
                    "reasons": budget_stop_reasons,
                }
            )
            break
        if (
            _float(best_heldout.get("task_success_rate")) >= target_success_rate
            and bool(best_heldout.get("safety_contact_gate_passed"))
        ):
            break

        planned_branches: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        for branch_index in range(max(1, int(agent_count))):
            _update_budget_wall_time(budget_ledger, start_monotonic=started_monotonic)
            branch_stop_reasons = _budget_limit_reached(budget_ledger)
            if branch_stop_reasons:
                budget_stop_reasons = branch_stop_reasons
                budget_ledger["events"].append(
                    {
                        "event": "branch_not_planned_budget_exhausted",
                        "iteration": iteration,
                        "branch_index": branch_index,
                        "reasons": branch_stop_reasons,
                    }
                )
                break
            candidate_engine = engine_cycle[(iteration + branch_index - 1) % len(engine_cycle)]
            candidate_recipe, idea = _mutate_recipe(
                parent_recipe=best_recipe,
                parent_train_eval=best_train,
                iteration=iteration,
                branch_index=branch_index,
                engine=candidate_engine,
            )
            estimated_tokens = _estimated_branch_tokens(candidate_recipe, idea)
            budget_ledger["usage"]["estimated_tokens"] = _int(
                budget_ledger["usage"].get("estimated_tokens")
            ) + estimated_tokens
            budget_ledger["usage"]["candidate_evaluations"] = _int(
                budget_ledger["usage"].get("candidate_evaluations")
            ) + 1
            budget_ledger["events"].append(
                {
                    "event": "branch_planned",
                    "iteration": iteration,
                    "branch_index": branch_index,
                    "candidate_id": candidate_recipe.get("candidate_id"),
                    "agent_id": candidate_recipe.get("mutation_agent_id"),
                    "simulator_engine": candidate_engine,
                    "evaluation_substrate": _evaluation_substrate_for_engine(candidate_engine),
                    "estimated_tokens": estimated_tokens,
                }
            )
            planned_branches.append((candidate_recipe, idea, candidate_engine))

        candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        if planned_branches:
            max_workers = min(
                max(1, int(parallel_branch_limit)),
                max(1, int(agent_count)),
                len(planned_branches),
            )
            budget_ledger["events"].append(
                {
                    "event": "parallel_branch_batch_started",
                    "iteration": iteration,
                    "branch_count": len(planned_branches),
                    "max_workers": max_workers,
                }
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_branch = {
                    executor.submit(
                        evaluate_candidate_branch,
                        candidate_recipe=candidate_recipe,
                        idea=idea,
                        candidate_engine=candidate_engine,
                    ): (candidate_recipe, idea, candidate_engine)
                    for candidate_recipe, idea, candidate_engine in planned_branches
                }
                for future in as_completed(future_to_branch):
                    candidate_recipe, idea, candidate_engine = future_to_branch[future]
                    try:
                        branch_result = future.result()
                    except Exception as exc:
                        branch_evaluator_command = _evaluator_command_for_engine(
                            engine=candidate_engine,
                            evaluator_command=evaluator_command,
                            evaluator_commands_by_engine=evaluator_commands_by_engine,
                        )
                        candidate_train = _eval_result_from_attempts(
                            recipe=candidate_recipe,
                            attempts=[],
                            phase="train",
                            engine=candidate_engine,
                            generated_at=generated,
                            verifier_sha256=verifier_sha256,
                            evaluator_command_used=bool(branch_evaluator_command),
                            evaluator_detail={
                                "error": f"{type(exc).__name__}:{exc}",
                                "parallel_branch_failed": True,
                            },
                        )
                        candidate_heldout = _eval_result_from_attempts(
                            recipe=candidate_recipe,
                            attempts=[],
                            phase="heldout",
                            engine=candidate_engine,
                            generated_at=generated,
                            verifier_sha256=verifier_sha256,
                            evaluator_command_used=bool(branch_evaluator_command),
                            evaluator_detail={
                                "error": f"{type(exc).__name__}:{exc}",
                                "parallel_branch_failed": True,
                            },
                        )
                        branch_result = (
                            candidate_recipe,
                            candidate_train,
                            candidate_heldout,
                            idea,
                        )
                    candidate_recipe, candidate_train, candidate_heldout, idea = branch_result
                    _record_eval_budget_usage(
                        budget_ledger,
                        eval_result=candidate_train,
                        phase="train",
                        candidate_id=_string(candidate_recipe.get("candidate_id")),
                        branch_id=_string(candidate_recipe.get("mutation_agent_id")),
                        start_monotonic=started_monotonic,
                    )
                    _record_eval_budget_usage(
                        budget_ledger,
                        eval_result=candidate_heldout,
                        phase="heldout",
                        candidate_id=_string(candidate_recipe.get("candidate_id")),
                        branch_id=_string(candidate_recipe.get("mutation_agent_id")),
                        start_monotonic=started_monotonic,
                    )
                    idea.update(
                        {
                            "train_success_rate": candidate_train["task_success_rate"],
                            "heldout_success_rate": candidate_heldout["task_success_rate"],
                            "safety_contact_gate_passed": bool(
                                candidate_train.get("safety_contact_gate_passed")
                            )
                            and bool(candidate_heldout.get("safety_contact_gate_passed")),
                            "accepted_for_next_iteration": False,
                            "promoted_candidate": False,
                        }
                    )
                    candidates.append(
                        (candidate_recipe, candidate_train, candidate_heldout, idea)
                    )
                    iteration_records.append(
                        {
                            "iteration": iteration,
                            "candidate_id": candidate_recipe.get("candidate_id"),
                            "agent_id": candidate_recipe.get("mutation_agent_id"),
                            "simulator_engine": candidate_engine,
                            "evaluation_substrate": _evaluation_substrate_for_engine(
                                candidate_engine
                            ),
                            "train_success_rate": candidate_train["task_success_rate"],
                            "heldout_success_rate": candidate_heldout["task_success_rate"],
                            "train_status": candidate_train.get("status"),
                            "heldout_status": candidate_heldout.get("status"),
                            "train_failure_mode_ids": candidate_train.get(
                                "failure_mode_ids", []
                            ),
                            "heldout_failure_mode_ids": candidate_heldout.get(
                                "failure_mode_ids", []
                            ),
                            "train_safety_contact_gate_passed": candidate_train[
                                "safety_contact_gate_passed"
                            ],
                            "heldout_safety_contact_gate_passed": candidate_heldout[
                                "safety_contact_gate_passed"
                            ],
                            "parallel_branch": True,
                            "recipe": candidate_recipe,
                        }
                    )

        if not candidates:
            break
        candidates.sort(
            key=lambda item: (
                _rank_eval(item[1]),
                _rank_eval(item[2]),
                -len(json.dumps(item[0], sort_keys=True)),
            ),
            reverse=True,
        )
        candidate_recipe, candidate_train, candidate_heldout, selected_idea = candidates[0]
        if _rank_eval(candidate_train) > _rank_eval(best_train):
            best_recipe = candidate_recipe
            best_train = candidate_train
            best_heldout = candidate_heldout
            selected_idea["accepted_for_next_iteration"] = True
        idea_nodes.extend(item[3] for item in candidates)

    target_success_reached = (
        _float(best_heldout.get("task_success_rate")) >= target_success_rate
        and bool(best_heldout.get("safety_contact_gate_passed"))
    )
    promoted = target_success_reached and (
        _float(best_heldout.get("task_success_rate"))
        >= _float(baseline_heldout.get("task_success_rate"))
    )
    for node in idea_nodes:
        if node.get("idea_id") == best_recipe.get("candidate_id") and promoted:
            node["promoted_candidate"] = True

    blockers = []
    if not target_success_reached:
        blockers.append("heldout_target_success_not_reached")
    if budget_stop_reasons and not target_success_reached:
        blockers.extend(reason for reason in budget_stop_reasons if reason not in blockers)
    if not bool(best_heldout.get("safety_contact_gate_passed")):
        blockers.append("heldout_safety_contact_gate_failed")
    if not bool(best_train.get("safety_contact_gate_passed")):
        blockers.append("train_safety_contact_gate_failed")
    best_policy_id = (
        _string(best_recipe.get("candidate_id"))
        or _string(best_recipe.get("policy_id"))
        or "seed_policy"
    )

    simulator_execution_proven = _eval_has_simulator_execution(
        best_train
    ) and _eval_has_simulator_execution(best_heldout)
    proven_simulator_engines = (
        _proven_simulator_engines(best_train, best_heldout)
        if simulator_execution_proven
        else []
    )
    _update_budget_wall_time(budget_ledger, start_monotonic=started_monotonic)
    budget_ledger["status"] = "completed" if not budget_stop_reasons else "budget_exhausted"
    budget_ledger["stop_reasons"] = budget_stop_reasons
    budget_ledger["promoted_before_budget_exhaustion"] = bool(promoted)
    heldout_eval_result = {
        **best_heldout,
        "status": "accepted_for_promotion" if promoted else "not_promoted",
        "target_success_rate": round(float(target_success_rate), 6),
        "target_success_reached": target_success_reached,
    }
    idea_tree = {
        "schema_version": POLICY_AUTORESEARCH_IDEA_TREE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "node_count": len(idea_nodes),
        "iteration_count": max((node.get("iteration") or 0 for node in idea_nodes), default=0),
        "nodes": idea_nodes,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    package_status = (
        "promoted_wam_policy_candidate"
        if promoted and wam_substrate_requested
        else "promoted_sim_only_policy_candidate"
        if promoted
        else "not_promoted"
    )
    policy_candidate_package = {
        "schema_version": POLICY_AUTORESEARCH_CANDIDATE_PACKAGE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": package_status,
        "candidate_policy_id": best_policy_id if promoted else None,
        "budget_ledger_path": ARTIFACT_PATHS["budget_ledger"],
        "recipe": _recipe_with_capabilities(best_recipe),
        "frozen_verifier_sha256": verifier_sha256,
        "train_eval_result_path": "train_eval_result.json",
        "heldout_eval_result_path": ARTIFACT_PATHS["heldout_eval_result"],
        "target_success_rate": round(float(target_success_rate), 6),
        "heldout_success_rate": best_heldout["task_success_rate"],
        "safety_contact_gate_passed": bool(best_heldout.get("safety_contact_gate_passed")),
        "blockers": blockers,
        "sim_only_policy_improvement_support_artifact": not wam_substrate_requested,
        "wam_policy_improvement_support_artifact": wam_substrate_requested,
        "simulator_execution_proven": simulator_execution_proven,
        "requested_simulator_engines": engine_cycle,
        "proven_simulator_engines": proven_simulator_engines,
        "evaluation_substrates": evaluation_substrate_cycle,
        "requested_evaluation_substrates": evaluation_substrate_cycle,
        "wam_evaluation_substrate_requested": wam_substrate_requested,
        "generated_wam_rollouts_are_model_derived_support_artifacts": (
            wam_substrate_requested
        ),
        "customer_specific_srcc_claimed": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "evaluation_substrates": evaluation_substrate_cycle,
            "wam_evaluation_substrate_requested": wam_substrate_requested,
            "simulator_execution_proven": simulator_execution_proven,
        },
    }
    followup_request = {
        "schema_version": POLICY_AUTORESEARCH_FOLLOWUP_REQUEST_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "drafted" if promoted else "not_ready",
        "job_dir": str(resolved_job_dir),
        "policy_candidate_package_path": ARTIFACT_PATHS["policy_candidate_package"],
        "frozen_verifier_sha256": verifier_sha256,
        "required_owner_evidence": [
            "real_robot_pov_manifest_with_exact_scenario_eval_run_ids",
            "deployment_outcome_records_with_owner_evidence_or_operator_attestation",
            "safety_contact_physics_evidence_for_every_promoted_scenario_eval_run",
            "paired_real_world_rollouts_for_customer_specific_srcc_claims",
        ],
        "requested_real_world_validation_run_ids": best_heldout.get(
            "covered_scenario_eval_run_ids", []
        )
        if promoted
        else [],
        "blockers": [] if promoted else blockers,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "evaluation_substrates": evaluation_substrate_cycle,
            "wam_evaluation_substrate_requested": wam_substrate_requested,
        },
    }
    status = "promoted" if promoted else "completed_no_promotion"
    report = {
        "schema_version": POLICY_AUTORESEARCH_REPORT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "scenario_eval_matrix_path": str(matrix_path),
        "seed_policy_recipe_path": str(recipe_path),
        "frozen_verifier_sha256": verifier_sha256,
        "target_success_rate": round(float(target_success_rate), 6),
        "target_success_reached": target_success_reached,
        "baseline_train_success_rate": baseline_train["task_success_rate"],
        "baseline_heldout_success_rate": baseline_heldout["task_success_rate"],
        "best_train_success_rate": best_train["task_success_rate"],
        "best_heldout_success_rate": best_heldout["task_success_rate"],
        "best_policy_id": best_policy_id,
        "promoted_policy_id": best_policy_id if promoted else None,
        "simulator_execution_proven": simulator_execution_proven,
        "max_iterations": int(max_iterations),
        "agent_count": int(agent_count),
        "simulator_engines": engine_cycle,
        "requested_simulator_engines": engine_cycle,
        "proven_simulator_engines": proven_simulator_engines,
        "evaluation_substrates": evaluation_substrate_cycle,
        "requested_evaluation_substrates": evaluation_substrate_cycle,
        "wam_evaluation_substrate_requested": wam_substrate_requested,
        "generated_wam_rollouts_are_model_derived_support_artifacts": (
            wam_substrate_requested
        ),
        "customer_specific_srcc_claimed": False,
        "evaluator_command_used": bool(evaluator_command or evaluator_commands_by_engine),
        "evaluator_commands_by_engine": sorted(
            _engine_key(engine) for engine in (evaluator_commands_by_engine or {})
        ),
        "evaluator_attempt_trace_path": str(attempt_trace_path) if attempt_trace_path else None,
        "budget_ledger": budget_ledger,
        "iteration_records": iteration_records,
        "blockers": blockers,
        "artifact_paths": dict(ARTIFACT_PATHS),
        "support_artifact_paths": {
            "verifier_manifest": "verifier_manifest.json",
            "train_eval_result": "train_eval_result.json",
            "baseline_train_eval_result": "baseline_train_eval_result.json",
        "baseline_heldout_eval_result": "baseline_heldout_eval_result.json",
            "budget_ledger": ARTIFACT_PATHS["budget_ledger"],
        },
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "evaluation_substrates": evaluation_substrate_cycle,
            "wam_evaluation_substrate_requested": wam_substrate_requested,
            "simulator_execution_proven": simulator_execution_proven,
        },
    }

    write_json(resolved_output_dir / "baseline_train_eval_result.json", baseline_train)
    write_json(resolved_output_dir / ARTIFACT_PATHS["budget_ledger"], budget_ledger)
    write_json(resolved_output_dir / "baseline_heldout_eval_result.json", baseline_heldout)
    write_json(resolved_output_dir / "train_eval_result.json", best_train)
    write_json(resolved_output_dir / ARTIFACT_PATHS["heldout_eval_result"], heldout_eval_result)
    write_json(resolved_output_dir / ARTIFACT_PATHS["agent_idea_tree"], idea_tree)
    write_json(
        resolved_output_dir / ARTIFACT_PATHS["policy_candidate_package"],
        policy_candidate_package,
    )
    write_json(
        resolved_output_dir / ARTIFACT_PATHS["followup_real_world_validation_request"],
        followup_request,
    )
    write_json(resolved_output_dir / ARTIFACT_PATHS["policy_autoresearch_report"], report)

    return {
        "report": report,
        "verifier_manifest": verifier_manifest,
        "baseline_train_eval_result": baseline_train,
        "baseline_heldout_eval_result": baseline_heldout,
        "train_eval_result": best_train,
        "heldout_eval_result": heldout_eval_result,
        "agent_idea_tree": idea_tree,
        "policy_candidate_package": policy_candidate_package,
        "followup_real_world_validation_request": followup_request,
        "budget_ledger": budget_ledger,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run sim-only policy autoresearch for a robot-eval job."
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--policy-recipe", required=True)
    parser.add_argument("--scenario-eval-matrix")
    parser.add_argument(
        "--reviewed-examples",
        help=(
            "Optional reviewed success/failure examples JSON to freeze into the verifier "
            "before policy mutation begins."
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--agent-count", type=int, default=DEFAULT_AGENT_COUNT)
    parser.add_argument("--target-success-rate", type=float, default=DEFAULT_TARGET_SUCCESS_RATE)
    parser.add_argument("--heldout-ratio", type=float, default=DEFAULT_HELDOUT_RATIO)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--compute-seconds-budget", type=float)
    parser.add_argument("--wall-time-budget-seconds", type=float)
    parser.add_argument("--max-candidate-evaluations", type=int)
    parser.add_argument(
        "--parallel-branch-limit",
        type=int,
        default=DEFAULT_PARALLEL_BRANCH_LIMIT,
    )
    parser.add_argument(
        "--evaluator-command",
        help=(
            "Optional command that evaluates one candidate recipe against one split matrix. "
            "It receives BLUEPRINT_POLICY_AUTORESEARCH_RECIPE, "
            "BLUEPRINT_POLICY_AUTORESEARCH_MATRIX, and "
            "BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT."
        ),
    )
    parser.add_argument(
        "--evaluator-command-by-engine",
        action="append",
        default=[],
        metavar="ENGINE=COMMAND",
        help=(
            "Engine-specific evaluator command. May be repeated, for example "
            "mujoco='python -m blueprint_pipeline.policy_autoresearch_mujoco_evaluator' "
            "and isaac_sim='python -m blueprint_pipeline.policy_autoresearch_owner_gpu_evaluator'."
        ),
    )
    parser.add_argument("--evaluator-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--evaluator-attempt-trace",
        help=(
            "Optional existing attempt trace JSON/JSONL to expose to the evaluator as "
            "BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE. This is replay evidence only "
            "unless the evaluator separately reruns a simulator."
        ),
    )
    parser.add_argument(
        "--simulator-engine",
        action="append",
        dest="simulator_engines",
        help="Simulator branch label to use for candidate evaluation. May be repeated.",
    )
    parser.add_argument(
        "--evaluation-substrate",
        action="append",
        dest="evaluation_substrates",
        help=(
            "Evaluation substrate branch label such as fixture_wam, cosmos3_wam, "
            "oscar_wam, classical_sim_mujoco, classical_sim_isaac, or recorded_trace. "
            "May be repeated. When provided, it takes precedence over --simulator-engine."
        ),
    )
    args = parser.parse_args(argv)
    try:
        evaluator_commands_by_engine = _parse_engine_evaluator_commands(
            args.evaluator_command_by_engine
        )
    except ValueError as exc:
        parser.error(str(exc))

    result = run_policy_autoresearch(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        policy_recipe_path=args.policy_recipe,
        scenario_eval_matrix_path=args.scenario_eval_matrix,
        reviewed_examples_path=args.reviewed_examples,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        agent_count=args.agent_count,
        target_success_rate=args.target_success_rate,
        heldout_ratio=args.heldout_ratio,
        simulator_engines=tuple(args.simulator_engines or DEFAULT_SIMULATOR_ENGINES),
        evaluation_substrates=tuple(args.evaluation_substrates)
        if args.evaluation_substrates
        else None,
        evaluator_command=args.evaluator_command,
        evaluator_commands_by_engine=evaluator_commands_by_engine,
        evaluator_timeout_seconds=args.evaluator_timeout_seconds,
        evaluator_attempt_trace_path=args.evaluator_attempt_trace,
        token_budget=args.token_budget,
        compute_seconds_budget=args.compute_seconds_budget,
        wall_time_budget_seconds=args.wall_time_budget_seconds,
        max_candidate_evaluations=args.max_candidate_evaluations,
        parallel_branch_limit=args.parallel_branch_limit,
    )
    report = result["report"]
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") in {"promoted", "completed_no_promotion", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
