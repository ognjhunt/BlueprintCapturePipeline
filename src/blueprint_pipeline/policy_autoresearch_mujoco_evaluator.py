"""MuJoCo-backed evaluator for policy autoresearch candidates.

The policy-autoresearch lane freezes the source scenario matrix. This evaluator
does not mutate that source contract; it derives a candidate split matrix that
contains only policy-generated route waypoints and policy ids, then executes the
packaged MuJoCo G1 simulator command against that candidate matrix.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .mujoco_g1_simulator_command import run_mujoco_g1_simulator_command


MUJOCO_POLICY_EVALUATOR_SCHEMA_VERSION = "policy_autoresearch_mujoco_evaluator.v1"

CLAIM_BOUNDARY = {
    "evaluator_kind": "mujoco_policy_route_execution",
    "source_scenario_eval_matrix_mutated": False,
    "candidate_matrix_contains_policy_generated_control_route": True,
    "simulator_execution_performed": True,
    "robot_policy_execution_performed": False,
    "balanced_locomotion_controller_integrated": False,
    "real_world_outcome_proven": False,
    "rank_fidelity_result_proven": False,
    "non_ranking_operational_claim_proven": False,
    "public_claim_upgrade_allowed": False,
}

SimulatorRunner = Callable[..., Mapping[str, Any]]


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


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _pose_triplet(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _run_pose(run: Mapping[str, Any], *keys: str) -> list[float] | None:
    for key in keys:
        pose = _pose_triplet(run.get(key))
        if pose is not None:
            return pose
    mutation = _mapping(run.get("concrete_mutation"))
    for key in keys:
        pose = _pose_triplet(mutation.get(key))
        if pose is not None:
            return pose
    return None


def _dedupe_route(points: Sequence[Sequence[float]]) -> list[list[float]]:
    route: list[list[float]] = []
    for point in points:
        pose = _pose_triplet(point)
        if pose is None:
            continue
        rounded = [round(float(value), 6) for value in pose]
        if route and sum((rounded[index] - route[-1][index]) ** 2 for index in range(3)) < 0.0025:
            continue
        route.append(rounded)
    return route


def _existing_route(run: Mapping[str, Any]) -> list[list[float]]:
    for key in ("route_waypoints", "navigation_waypoints", "waypoints"):
        value = run.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            route = _dedupe_route([point for point in value if _pose_triplet(point) is not None])
            if route:
                return route
    return []


def _route_from_recipe(
    *,
    run: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> tuple[list[list[float]], str]:
    start = _run_pose(
        run,
        "spawn_pose",
        "start_pose",
        "initial_pose",
        "robot_spawn_pose",
        "robot_start_pose",
        "start_xyz",
        "spawn_xyz",
    )
    target = _run_pose(
        run,
        "target_pose",
        "goal_pose",
        "navigation_target_pose",
        "robot_target_pose",
        "target_xyz",
        "goal_xyz",
    )
    existing = _existing_route(run)
    if start is None or target is None:
        return existing, "source_matrix_route_no_pose_available"

    params = _mapping(recipe.get("mutable_parameters") or recipe.get("mutableParameters"))
    planner = _string(params.get("planner")).lower()
    retry_budget = _int(params.get("retry_budget") or params.get("retryBudget"), 0)
    route_style = _string(params.get("route_style") or params.get("routeStyle")).lower()
    detour_y_raw = params.get("detour_y") if "detour_y" in params else params.get("detourY")
    detour_x_raw = params.get("detour_x") if "detour_x" in params else params.get("detourX")
    z = float(start[2])

    if planner not in {"clearance_aware", "route_replan", "safety_margin"}:
        return existing or _dedupe_route([start, target]), "policy_direct_or_source_route"

    if detour_y_raw is not None:
        detour_y = _float(detour_y_raw)
    elif retry_budget > 0 or route_style in {"south", "perimeter_south"}:
        detour_y = -9.0
    else:
        detour_y = 8.8

    if detour_x_raw is not None:
        detour_x = _float(detour_x_raw)
    else:
        detour_x = 0.0

    route: list[list[float]]
    if retry_budget > 0 or route_style.startswith("perimeter"):
        outer_x = max(abs(start[0]), abs(target[0]), 8.0) + 1.5
        signed_outer_x = outer_x if start[0] >= target[0] else -outer_x
        opposite_outer_x = -signed_outer_x
        route = _dedupe_route(
            [
                start,
                [signed_outer_x, detour_y, z],
                [detour_x, detour_y, z],
                [opposite_outer_x, detour_y, z],
                target,
            ]
        )
        return route, "policy_perimeter_clearance_route"

    route = _dedupe_route(
        [
            start,
            [start[0], detour_y, z],
            [detour_x, detour_y, z],
            [target[0], detour_y, z],
            target,
        ]
    )
    return route, "policy_clearance_aware_detour_route"


def build_candidate_matrix(
    *,
    recipe: Mapping[str, Any],
    split_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    runs = [dict(run) for run in split_matrix.get("runs", []) if isinstance(run, Mapping)]
    policy_id = _string(recipe.get("candidate_id") or recipe.get("policy_id")) or "policy_candidate"
    candidate_runs: list[dict[str, Any]] = []
    route_strategies: list[str] = []
    for run in runs:
        route, route_strategy = _route_from_recipe(run=run, recipe=recipe)
        route_strategies.append(route_strategy)
        candidate = {
            **run,
            "policy_id": policy_id,
            "policy_recipe_id": policy_id,
            "policy_generated_route_strategy": route_strategy,
            "policy_generated_route_waypoint_count": len(route),
        }
        if route:
            candidate["route_waypoints"] = route
        candidate_runs.append(candidate)
    return {
        **dict(split_matrix),
        "schema_version": "policy_autoresearch_mujoco_candidate_matrix.v1",
        "source_schema_version": split_matrix.get("schema_version"),
        "policy_id": policy_id,
        "policy_route_strategies": sorted(set(route_strategies)),
        "source_scenario_eval_matrix_mutated": False,
        "runs": candidate_runs,
        "scenario_eval_run_count": len(candidate_runs),
    }


def _contact_event_count(attempt: Mapping[str, Any]) -> int:
    metrics = _mapping(attempt.get("metrics"))
    return (
        _int(metrics.get("robot_scene_contact_event_count"))
        + _int(metrics.get("near_miss_event_count"))
        + _int(metrics.get("collision_response_event_count"))
    )


def _safety_event_count(attempt: Mapping[str, Any]) -> int:
    metrics = _mapping(attempt.get("metrics"))
    return _int(metrics.get("fall_count")) + _int(metrics.get("unsafe_proximity_event_count"))


def _normalize_mujoco_attempts(
    *,
    attempts: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    candidate_matrix_path: Path,
    simulator_output_path: Path,
    generated_at: str,
) -> list[dict[str, Any]]:
    policy_id = _string(recipe.get("candidate_id") or recipe.get("policy_id")) or "policy_candidate"
    normalized: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        metrics = _mapping(attempt.get("metrics"))
        safety_event_count = _safety_event_count(attempt)
        contact_event_count = _contact_event_count(attempt)
        normalized_metrics = {
            **metrics,
            "safety_event_count": safety_event_count,
            "contact_event_count": contact_event_count,
            "simulator_execution_performed": True,
            "robot_policy_execution_performed": False,
            "candidate_matrix_path": str(candidate_matrix_path),
            "simulator_output_path": str(simulator_output_path),
        }
        normalized.append(
            {
                **dict(attempt),
                "attempt_id": _string(attempt.get("attempt_id"))
                or f"{_safe_id(policy_id)}_mujoco_{index:04d}",
                "policy_id": policy_id,
                "policy_kind": _string(recipe.get("policy_kind") or recipe.get("policyKind")),
                "success": bool(attempt.get("task_success") or attempt.get("success")),
                "task_success": bool(attempt.get("task_success") or attempt.get("success")),
                "metrics": normalized_metrics,
                "generated_at": generated_at,
                "claim_boundary": {
                    **CLAIM_BOUNDARY,
                    "mujoco_attempt_claim_boundary": attempt.get("claim_boundary"),
                },
            }
        )
    return normalized


def run_mujoco_policy_evaluator(
    *,
    recipe_path: str | Path,
    matrix_path: str | Path,
    output_path: str | Path,
    capture_root: str | Path,
    g1_model_root: str | Path | None = None,
    steps: int = 64,
    output_root: str | Path | None = None,
    simulator_runner: SimulatorRunner = run_mujoco_g1_simulator_command,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    recipe = _mapping(read_json_any(Path(recipe_path)))
    split_matrix = _mapping(read_json_any(Path(matrix_path)))
    resolved_output = Path(output_path).resolve()
    root = Path(output_root).resolve() if output_root else resolved_output.parent / "mujoco_execution"
    ensure_dir(root)
    candidate_matrix = build_candidate_matrix(recipe=recipe, split_matrix=split_matrix)
    candidate_matrix_path = root / "candidate_scenario_eval_matrix.json"
    simulator_output_path = root / "mujoco_g1_simulator_output.json"
    write_json(candidate_matrix_path, candidate_matrix)

    simulator_payload = simulator_runner(
        capture_root=Path(capture_root),
        g1_model_root=Path(g1_model_root) if g1_model_root else None,
        output_dir=root,
        simulator_output_path=simulator_output_path,
        scenario_eval_matrix_path=candidate_matrix_path,
        steps=max(1, int(steps)),
        render_frames=False,
        max_rendered_episodes=0,
    )
    raw_attempts = simulator_payload.get("attempts")
    attempts = _normalize_mujoco_attempts(
        attempts=raw_attempts if isinstance(raw_attempts, list) else [],
        recipe=recipe,
        candidate_matrix_path=candidate_matrix_path,
        simulator_output_path=simulator_output_path,
        generated_at=generated,
    )
    payload = {
        "schema_version": MUJOCO_POLICY_EVALUATOR_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if attempts else "blocked_no_mujoco_attempts",
        "phase": _string(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_PHASE")),
        "simulator_engine": "mujoco",
        "frozen_verifier_sha256": _string(
            os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256")
        ),
        "policy_id": _string(recipe.get("candidate_id") or recipe.get("policy_id")),
        "candidate_matrix_path": str(candidate_matrix_path),
        "simulator_output_path": str(simulator_output_path),
        "simulator_status": simulator_payload.get("status"),
        "simulator_execution_proven": bool(
            simulator_payload.get("simulator_execution_proven")
            or simulator_payload.get("default_sim_policy_execution_proven")
        ),
        "candidate_route_strategies": candidate_matrix.get("policy_route_strategies", []),
        "attempts": attempts,
        "claim_boundary": dict(CLAIM_BOUNDARY),
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
    g1_root = _env_path("BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_G1_MODEL_ROOT") or _env_path(
        "BLUEPRINT_MUJOCO_G1_MODEL_ROOT"
    )
    output_root = _env_path("BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_OUTPUT_DIR")
    steps = _int(os.environ.get("BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_STEPS"), 64)
    run_mujoco_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        capture_root=capture_root,
        g1_model_root=g1_root,
        steps=steps,
        output_root=output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
