"""Wave scheduler for one booted lightweight Isaac Lab control-search env."""

from __future__ import annotations

import json
import math
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
WAVE_COMMANDS_SCHEMA_VERSION = (
    "task_evaluation_isaaclab_control_sweep_wave_commands.v1"
)


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


def compile_isaaclab_control_sweep_wave_commands(
    *,
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    candidate_inventory: Mapping[str, Any],
    wave_index: int,
    arm_joint_names: Sequence[str],
) -> dict[str, Any]:
    """Compile cuRobo waypoints into one clone-indexed symbolic action wave."""

    frozen_plan = validate_control_search_funnel_plan(plan)
    frozen_schedule = validate_isaaclab_control_sweep_schedule(
        schedule, plan=frozen_plan
    )
    if (
        not isinstance(wave_index, int)
        or isinstance(wave_index, bool)
        or not 0 <= wave_index < frozen_schedule["wave_count"]
        or not isinstance(arm_joint_names, Sequence)
        or isinstance(arm_joint_names, (str, bytes))
        or len(arm_joint_names) != 7
        or any(not isinstance(name, str) or not name for name in arm_joint_names)
        or len(set(arm_joint_names)) != 7
        or candidate_inventory.get("inventory_digest")
        != frozen_schedule["candidate_inventory_digest"]
    ):
        raise ControlSearchFunnelError("control_search_wave_commands_invalid")
    candidates = {
        str(row.get("candidate_id") or ""): row
        for row in candidate_inventory.get("candidates") or []
        if isinstance(row, Mapping)
    }
    wave = frozen_schedule["waves"][wave_index]
    rows = []
    maximum_waypoint_count = 0
    for assignment in wave["assignments"]:
        candidate = candidates.get(assignment["candidate_id"])
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("candidate_digest")
            != assignment["candidate_digest"]
        ):
            raise ControlSearchFunnelError(
                "control_search_wave_commands_invalid"
            )
        reset = candidate.get("reset_variant")
        base_pose = candidate.get("robot_base_pose_world")
        reset_positions = (
            reset.get("robot_joint_reset_positions_rad")
            if isinstance(reset, Mapping)
            else None
        )
        variants = (
            candidate.get("entry_trajectory_variant"),
            candidate.get("interaction_trajectory_variant"),
        )
        waypoints: list[dict[str, Any]] = []
        if not isinstance(reset_positions, Mapping) or not isinstance(
            base_pose, Mapping
        ):
            raise ControlSearchFunnelError(
                "control_search_wave_commands_invalid"
            )
        try:
            reset_vector = [float(reset_positions[name]) for name in arm_joint_names]
            base_position = [float(value) for value in base_pose["position_world_m"]]
            base_orientation = [float(value) for value in base_pose["orientation_xyzw"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlSearchFunnelError(
                "control_search_wave_commands_invalid"
            ) from exc
        if (
            len(base_position) != 3
            or len(base_orientation) != 4
            or not all(
                math.isfinite(value)
                for value in [*reset_vector, *base_position, *base_orientation]
            )
            or not math.isclose(
                math.sqrt(math.fsum(value * value for value in base_orientation)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-4,
            )
        ):
            raise ControlSearchFunnelError(
                "control_search_wave_commands_invalid"
            )
        for variant in variants:
            if not isinstance(variant, Mapping) or not isinstance(
                variant.get("waypoints"), list
            ):
                raise ControlSearchFunnelError(
                    "control_search_wave_commands_invalid"
                )
            for waypoint in variant["waypoints"]:
                joints = (
                    waypoint.get("robot_joint_positions_rad")
                    if isinstance(waypoint, Mapping)
                    else None
                )
                stage_kind = str(
                    waypoint.get("stage_kind")
                    if isinstance(waypoint, Mapping)
                    else ""
                )
                try:
                    target = [float(joints[name]) for name in arm_joint_names]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ControlSearchFunnelError(
                        "control_search_wave_commands_invalid"
                    ) from exc
                if (
                    stage_kind
                    not in {"entry", "approach", "contact", "release", "retreat"}
                    or not all(math.isfinite(value) for value in target)
                ):
                    raise ControlSearchFunnelError(
                        "control_search_wave_commands_invalid"
                    )
                waypoints.append(
                    {
                        "waypoint_index": len(waypoints),
                        "waypoint_id": str(waypoint.get("waypoint_id") or ""),
                        "stage_kind": stage_kind,
                        "arm_joint_positions_rad": target,
                        "gripper_state": (
                            "closed" if stage_kind == "contact" else "open"
                        ),
                    }
                )
        if not waypoints:
            raise ControlSearchFunnelError(
                "control_search_wave_commands_invalid"
            )
        maximum_waypoint_count = max(maximum_waypoint_count, len(waypoints))
        rows.append(
            {
                **assignment,
                "robot_base_pose_world": {
                    "position_world_m": base_position,
                    "orientation_xyzw": base_orientation,
                },
                "robot_joint_reset_positions_rad": reset_vector,
                "waypoint_count": len(waypoints),
                "waypoints": waypoints,
            }
        )
    commands: dict[str, Any] = {
        "schema_version": WAVE_COMMANDS_SCHEMA_VERSION,
        "status": "compiled",
        "plan_digest": frozen_plan["plan_digest"],
        "schedule_digest": frozen_schedule["schedule_digest"],
        "candidate_inventory_digest": candidate_inventory["inventory_digest"],
        "wave_index": wave_index,
        "active_environment_count": wave["active_environment_count"],
        "vector_env_count": frozen_schedule["vector_env_count"],
        "arm_joint_names": list(arm_joint_names),
        "maximum_waypoint_count": maximum_waypoint_count,
        "inactive_environments_hold_reset_state": True,
        "assignments": rows,
        "commands_digest": "",
    }
    commands["commands_digest"] = canonical_digest(
        commands, digest_field="commands_digest"
    )
    return commands


def _vec3(value: object, *, blocker: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ControlSearchFunnelError(blocker)
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ControlSearchFunnelError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise ControlSearchFunnelError(blocker)
    return result


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(math.fsum(component * component for component in value))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _norm(tuple(left - right for left, right in zip(a, b, strict=True)))


def _segment_distance(
    point: Sequence[float], start: Sequence[float], target: Sequence[float]
) -> float:
    direction = tuple(
        right - left for left, right in zip(start, target, strict=True)
    )
    denominator = math.fsum(component * component for component in direction)
    if denominator <= float.fromhex("0x1.0p-52"):
        return _distance(point, start)
    offset = tuple(
        value - origin for value, origin in zip(point, start, strict=True)
    )
    fraction = min(
        1.0,
        max(
            0.0,
            math.fsum(
                value * component
                for value, component in zip(offset, direction, strict=True)
            )
            / denominator,
        ),
    )
    projection = tuple(
        origin + fraction * component
        for origin, component in zip(start, direction, strict=True)
    )
    return _distance(point, projection)


def build_isaaclab_control_search_outcome(
    *,
    assignment: Mapping[str, Any],
    reset_readback_passed: bool,
    task_position_trace_world_m: Sequence[Sequence[float]],
    forbidden_contact_force_trace_w_n: Sequence[Sequence[float]],
    required_contact_force_trace_w_n: Sequence[Sequence[float]],
    stage_kinds: Sequence[str],
    target_position_world_m: Sequence[float],
    required_contact_minimum_force_n: float,
    settle_sample_count: int,
) -> dict[str, Any]:
    """Reduce raw simulator tensors into one non-learned outcome receipt."""

    positions = [
        _vec3(row, blocker="control_search_measurement_trace_invalid")
        for row in task_position_trace_world_m
    ]
    forbidden = [
        _vec3(row, blocker="control_search_measurement_trace_invalid")
        for row in forbidden_contact_force_trace_w_n
    ]
    required = [
        _vec3(row, blocker="control_search_measurement_trace_invalid")
        for row in required_contact_force_trace_w_n
    ]
    target = _vec3(
        target_position_world_m,
        blocker="control_search_measurement_trace_invalid",
    )
    if (
        not isinstance(reset_readback_passed, bool)
        or not positions
        or len(forbidden) != len(positions)
        or len(required) != len(positions)
        or len(stage_kinds) != len(positions)
        or any(
            stage not in {"reset", "entry", "approach", "contact", "release", "retreat", "settle"}
            for stage in stage_kinds
        )
        or isinstance(required_contact_minimum_force_n, bool)
        or not math.isfinite(float(required_contact_minimum_force_n))
        or float(required_contact_minimum_force_n) < 0.0
        or not isinstance(settle_sample_count, int)
        or isinstance(settle_sample_count, bool)
        or not 1 <= settle_sample_count <= len(positions)
    ):
        raise ControlSearchFunnelError("control_search_measurement_trace_invalid")
    contact_indices = [
        index for index, stage in enumerate(stage_kinds) if stage == "contact"
    ]
    if not contact_indices:
        raise ControlSearchFunnelError("control_search_measurement_trace_invalid")
    contact_coverage = sum(
        _norm(required[index]) >= float(required_contact_minimum_force_n)
        for index in contact_indices
    ) / len(contact_indices)
    path_error = max(
        _segment_distance(positions[index], positions[0], target)
        for index in contact_indices
    )
    settle = positions[-settle_sample_count:]
    outcome: dict[str, Any] = {
        "schema_version": "task_evaluation_control_search_vector_outcome.v1",
        "candidate_id": assignment.get("candidate_id"),
        "candidate_digest": assignment.get("candidate_digest"),
        "seed_index": assignment.get("seed_index"),
        "resolved_seed": assignment.get("resolved_seed"),
        "wave_index": assignment.get("wave_index"),
        "environment_index": assignment.get("environment_index"),
        "reset_readback_passed": reset_readback_passed,
        "forbidden_collision_peak_force_n": max(_norm(row) for row in forbidden),
        "required_task_contact_coverage_fraction": contact_coverage,
        "push_path_tracking_error_m": path_error,
        "destination_error_m": _distance(positions[-1], target),
        "support_stability_error_m": max(
            _distance(position, settle[-1]) for position in settle
        ),
        "task_displacement_m": _distance(positions[-1], positions[0]),
        "physics_steps": len(positions),
        "measurement_authority": (
            "isaac_lab_simulator_state_and_contact_sensors"
        ),
        "learned_grader_used": False,
        "outcome_digest": "",
    }
    outcome["outcome_digest"] = canonical_digest(
        outcome, digest_field="outcome_digest"
    )
    return outcome


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
    "WAVE_COMMANDS_SCHEMA_VERSION",
    "build_isaaclab_control_sweep_schedule",
    "build_isaaclab_control_search_outcome",
    "compile_isaaclab_control_sweep_wave_commands",
    "execute_isaaclab_control_sweep",
    "validate_isaaclab_control_sweep_schedule",
]
