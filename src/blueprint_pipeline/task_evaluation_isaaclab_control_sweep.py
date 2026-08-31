"""Wave scheduler for one booted lightweight Isaac Lab control-search env."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_runtime import build_native_task_arena_environment
from .task_evaluation_control_search_funnel import (
    ControlSearchFunnelError,
    build_control_search_sweep_result,
    validate_control_search_funnel_plan,
)


SCHEDULE_SCHEMA_VERSION = "task_evaluation_isaaclab_control_sweep_schedule.v1"


def _copy(value: Mapping[str, Any], *, blocker: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ControlSearchFunnelError(blocker) from exc
    if not isinstance(result, dict):
        raise ControlSearchFunnelError(blocker)
    return result


def build_isaaclab_control_sweep_schedule(
    *,
    plan: Mapping[str, Any],
    candidate_inventory: Mapping[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    """Assign every candidate/seed pair to deterministic waves and env slots."""

    frozen_plan = validate_control_search_funnel_plan(plan)
    if (
        not isinstance(base_seed, int)
        or isinstance(base_seed, bool)
        or base_seed < 0
        or candidate_inventory.get("inventory_digest")
        != frozen_plan["immutable_inputs"]["candidate_inventory_digest"]
        or candidate_inventory.get("model_authored_candidates") is not False
        or not isinstance(candidate_inventory.get("candidates"), list)
    ):
        raise ControlSearchFunnelError("control_search_schedule_input_invalid")
    candidates = {
        str(row.get("candidate_id") or ""): row
        for row in candidate_inventory["candidates"]
        if isinstance(row, Mapping)
    }
    plan_candidates = frozen_plan["candidate_index"]
    if len(candidates) != len(plan_candidates) or any(
        candidate["candidate_id"] not in candidates
        or candidates[candidate["candidate_id"]].get("candidate_digest")
        != candidate["candidate_digest"]
        or candidates[candidate["candidate_id"]].get("candidate_digest")
        != canonical_digest(
            candidates[candidate["candidate_id"]],
            digest_field="candidate_digest",
        )
        for candidate in plan_candidates
    ):
        raise ControlSearchFunnelError("control_search_schedule_input_invalid")
    vector = frozen_plan["vector_sweep"]
    env_count = vector["resolved_vector_env_count"]
    assignments: list[dict[str, Any]] = []
    for candidate in plan_candidates:
        for seed_index in range(vector["seeds_per_candidate"]):
            ordinal = len(assignments)
            assignments.append(
                {
                    "assignment_index": ordinal,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_digest": candidate["candidate_digest"],
                    "seed_index": seed_index,
                    "resolved_seed": base_seed + ordinal,
                    "wave_index": ordinal // env_count,
                    "environment_index": ordinal % env_count,
                }
            )
    waves = []
    for wave_index in range(vector["wave_count"]):
        rows = [
            row for row in assignments if row["wave_index"] == wave_index
        ]
        waves.append(
            {
                "wave_index": wave_index,
                "active_environment_count": len(rows),
                "reset_before_wave_required": True,
                "assignments": rows,
            }
        )
    schedule: dict[str, Any] = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "status": "scheduled",
        "run_id": frozen_plan["run_id"],
        "plan_digest": frozen_plan["plan_digest"],
        "candidate_inventory_digest": candidate_inventory["inventory_digest"],
        "base_seed": base_seed,
        "vector_env_count": env_count,
        "wave_count": len(waves),
        "assignment_count": len(assignments),
        "boot_once_reuse_across_waves": True,
        "reset_before_every_wave": True,
        "waves": waves,
        "schedule_digest": "",
    }
    schedule["schedule_digest"] = canonical_digest(
        schedule, digest_field="schedule_digest"
    )
    return schedule


def validate_isaaclab_control_sweep_schedule(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact wave coverage before the environment is built."""

    schedule = _copy(value, blocker="control_search_schedule_invalid")
    frozen_plan = validate_control_search_funnel_plan(plan)
    waves = schedule.get("waves")
    if (
        schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or schedule.get("status") != "scheduled"
        or schedule.get("plan_digest") != frozen_plan["plan_digest"]
        or schedule.get("vector_env_count")
        != frozen_plan["vector_sweep"]["resolved_vector_env_count"]
        or schedule.get("wave_count")
        != frozen_plan["vector_sweep"]["wave_count"]
        or schedule.get("assignment_count")
        != frozen_plan["vector_sweep"]["assignment_count"]
        or schedule.get("boot_once_reuse_across_waves") is not True
        or schedule.get("reset_before_every_wave") is not True
        or not isinstance(waves, list)
        or len(waves) != schedule["wave_count"]
        or schedule.get("schedule_digest")
        != canonical_digest(schedule, digest_field="schedule_digest")
    ):
        raise ControlSearchFunnelError("control_search_schedule_invalid")
    assignments = [
        row
        for wave in waves
        if isinstance(wave, Mapping)
        for row in wave.get("assignments") or []
        if isinstance(row, Mapping)
    ]
    if (
        len(assignments) != schedule["assignment_count"]
        or [row.get("assignment_index") for row in assignments]
        != list(range(schedule["assignment_count"]))
        or any(
            any(
                row.get("wave_index") != index
                for row in wave.get("assignments") or []
            )
            or wave.get("reset_before_wave_required") is not True
            or wave.get("active_environment_count")
            != len(wave.get("assignments") or [])
            for index, wave in enumerate(waves)
        )
    ):
        raise ControlSearchFunnelError("control_search_schedule_invalid")
    return schedule


WaveRunner = Callable[..., Mapping[str, Any]]
EnvironmentBuilder = Callable[..., Any]


def execute_isaaclab_control_sweep(
    *,
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    candidate_inventory: Mapping[str, Any],
    scene_plan: Mapping[str, Any],
    bundle_root: str | Path,
    wave_runner: WaveRunner,
    environment_builder: EnvironmentBuilder = build_native_task_arena_environment,
) -> dict[str, Any]:
    """Boot Isaac Lab once, reset per wave, and seal the deterministic result."""

    frozen_plan = validate_control_search_funnel_plan(plan)
    frozen_schedule = validate_isaaclab_control_sweep_schedule(
        schedule, plan=frozen_plan
    )
    built = environment_builder(
        scene_plan,
        bundle_root=bundle_root,
        num_envs=frozen_schedule["vector_env_count"],
        enable_cameras=False,
        include_scene_appearance=False,
        render_mode=None,
    )
    outcomes: list[Mapping[str, Any]] = []
    peak_gpu_memory_bytes = 0
    for wave in frozen_schedule["waves"]:
        observed = wave_runner(
            built=built,
            wave=wave,
            candidate_inventory=candidate_inventory,
            plan=frozen_plan,
        )
        rows = observed.get("outcomes") if isinstance(observed, Mapping) else None
        peak = (
            observed.get("peak_gpu_memory_bytes")
            if isinstance(observed, Mapping)
            else None
        )
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes))
            or len(rows) != wave["active_environment_count"]
            or not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak < 1
        ):
            raise ControlSearchFunnelError(
                "control_search_wave_execution_invalid"
            )
        outcomes.extend(rows)
        peak_gpu_memory_bytes = max(peak_gpu_memory_bytes, peak)
    return build_control_search_sweep_result(
        plan=frozen_plan,
        outcomes=outcomes,
        actual_vector_env_count=frozen_schedule["vector_env_count"],
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
    )


__all__ = [
    "SCHEDULE_SCHEMA_VERSION",
    "build_isaaclab_control_sweep_schedule",
    "execute_isaaclab_control_sweep",
    "validate_isaaclab_control_sweep_schedule",
]
