"""What an invited robot team's evaluation of one site job runs: 200 hidden, scored cells.

Beta journey step: L5 team policy evaluation. A team is evaluated once, on 200
cells it has not seen, drawn from the scene's bounded variation families; the
result is the only thing a recommendation may rest on. Ten episodes cannot tell
a 90% policy from a 60% one (9/10 has a 95% Wilson lower bound near 0.60); 200
can (180/200 has one near 0.85). There is no separate practice run: a team's
integration mistakes (joint order, units, cameras) are caught by failing fast,
not by a second kind of run.

- The first shard of 25 cells runs alone. On a scene whose robot-free check is
  still undecided, its first cell records and scores that check before any
  team episode, and no team episode runs unless it passed.
- If the scene's check did not pass, or most of the first shard's episodes
  failed at the interface (the service did not answer, or returned an
  invalid or out-of-limit action) rather than at the task, the evaluation
  stops there and the team gets those episodes' outcomes and reasons, without
  the other seven shards being spent.

Scored cells are generated deterministically from the scene revision and a
per-scene secret salt held on the control plane, so every team on a job faces
the same hidden draws, and the published code cannot be used to compute them
in advance. The variation bounds are the ones the site-task variation request
already derives from the task (``site_task_variation_request``): a start shift
of at most a quarter of the start-to-goal distance (and 3 cm), a yaw of at most
half the orientation tolerance (and 10 degrees), task light 0.8-1.2, external
camera +-2 cm and dynamic friction within 20% of nominal. Nothing here changes
the object, the task or the success contract.

The matrix mixes, for 200 cells: canonical anchors, single-family cells
stratified across each family's range (placement, illumination, camera,
physics), pairwise stress cells and a held-out composition of every family at
once. New team evaluations execute all cells on one GPU allocation. Batches
of at most 25 remain an early interface gate and a checkpoint boundary, not
a reason to rent another machine. Legacy staged shards remain readable.

"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .site_task_variation_request import (
    FRICTION_FRACTION,
    LIGHT_SCALE_RANGE,
    MAXIMUM_CAMERA_SHIFT_M,
    MAXIMUM_START_SHIFT_M,
    MAXIMUM_YAW_DEGREES,
    START_SHIFT_SHARE_OF_REACH,
)

SCORED = "scored"
PURPOSES = (SCORED,)
SCORED_PROFILE = "team_evaluation_scored_v1"
PROFILES = {SCORED: SCORED_PROFILE}
EPISODES = {SCORED: 200}
SHARD_CELLS = 25
# The first shard stops the evaluation when more than this share of its
# episodes failed at the interface rather than at the task.
FIRST_SHARD_INTERFACE_FAILURE_SHARE = 0.5
INTERFACE_FAILURE_REASONS = frozenset({"policy_service_did_not_answer", "policy_action_invalid",
                                       "policy_service_error", "policy_action_beyond_joint_limits"})
# At most this share of a scored run may be void (the simulator, not the
# policy, failed). Void episodes never count; a run with more is refused.
MAXIMUM_VOID_SHARE = 0.05
# Measured: ~1 min to start Isaac and reset a cell, ~140 s per episode at 15 Hz
# for 30 s of simulated time, plus media sealing. Each shard gets this per cell
# and a fixed allowance for provisioning and download.
SECONDS_PER_CELL = 300
SHARD_OVERHEAD_SECONDS = 1800
# Additional admission slack for metadata, sealing and the initial queue handoff.
PRELAUNCH_RESERVE_SECONDS = 1800
# Allocator reserve includes five minutes for its provider-object staging.
ALLOCATOR_STARTUP_RESERVE_SECONDS = 600
PROVIDER_STAGING_RESERVE_SECONDS = 300
SCORED_FAMILY_SHARES = (
    ("canonical_anchor", 0.02),
    ("placement_approach", 0.23),
    ("illumination", 0.12),
    ("camera_sensor", 0.12),
    ("bounded_physics", 0.12),
    ("pairwise_stress", 0.19),
    ("held_out_composition", 0.20),
)
SINGLE_FAMILY_DIMENSIONS = {
    "placement_approach": ("object_start_x_delta_m", "object_start_y_delta_m", "object_yaw_delta_degrees"),
    "illumination": ("task_light_intensity_scale",),
    "camera_sensor": ("external_camera_x_delta_m",),
    "bounded_physics": ("dynamic_friction",),
}


class TeamEvaluationPlanError(ValueError):
    """The scene cannot carry this evaluation plan."""


def profile_purpose(profile_id: str) -> str | None:
    return next((purpose for purpose, value in PROFILES.items() if value == profile_id), None)


def shard_ttl_seconds(cell_count: int, *, control_hz: int = 15) -> int:
    """A shard's hard time limit; cameras render every control step, so a faster robot takes longer per cell."""

    return SHARD_OVERHEAD_SECONDS + SECONDS_PER_CELL * int(cell_count) * max(1, int(control_hz) // 15)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def variation_bounds(task_spec: Mapping[str, Any], *, task_kind: str,
                     nominal_dynamic_friction: float | None) -> dict[str, tuple[float, float]]:
    """Each varied parameter's admitted [low, high], derived from the task; omitted when unknowable."""

    bounds: dict[str, tuple[float, float]] = {
        "task_light_intensity_scale": (float(LIGHT_SCALE_RANGE[0]), float(LIGHT_SCALE_RANGE[1])),
        "external_camera_x_delta_m": (-MAXIMUM_CAMERA_SHIFT_M, MAXIMUM_CAMERA_SHIFT_M),
    }
    start = task_spec.get("start_pose_world")
    box = task_spec.get("destination_position_bounds_world_m")
    shift = MAXIMUM_START_SHIFT_M
    if isinstance(start, list) and len(start) >= 3 and isinstance(box, Mapping):
        low, high = box.get("minimum"), box.get("maximum")
        if isinstance(low, list) and isinstance(high, list) and len(low) >= 3 and len(high) >= 3:
            middle = [(float(low[i]) + float(high[i])) / 2 for i in range(3)]
            reach = math.dist([float(v) for v in start[:3]], middle)
            shift = min(MAXIMUM_START_SHIFT_M, START_SHIFT_SHARE_OF_REACH * reach)
    elif task_kind == "rigid_pick_place":
        shift = 0.0  # a rigid task with no stated start or goal is not varied in place
    if shift >= 0.005:
        bounds["object_start_x_delta_m"] = (-shift, shift)
        bounds["object_start_y_delta_m"] = (-shift, shift)
    yaw = MAXIMUM_YAW_DEGREES
    tolerance = _finite(task_spec.get("destination_orientation_tolerance_rad"))
    if tolerance is not None and tolerance > 0:
        yaw = min(yaw, math.degrees(tolerance) * 0.5)
    if yaw >= 1.0:
        bounds["object_yaw_delta_degrees"] = (-yaw, yaw)
    friction = _finite(nominal_dynamic_friction)
    if task_kind == "rigid_pick_place" and friction is not None and friction > 0:
        bounds["dynamic_friction"] = (friction * (1 - FRICTION_FRACTION), friction * (1 + FRICTION_FRACTION))
    return bounds


def _unit(*parts: Any) -> float:
    """A deterministic number in (0, 1) from ``parts``."""

    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()
    return (int(digest[:15], 16) + 1) / (16**15 + 1)


def _stratified(low: float, high: float, count: int, *, key: Sequence[Any]) -> list[float]:
    """``count`` values covering [low, high] once per stratum, in a seeded order (Latin hypercube)."""

    order = sorted(range(count), key=lambda index: _unit(*key, "order", index))
    return [low + (order[index] + _unit(*key, "jitter", index)) / count * (high - low) for index in range(count)]


def _rounded(name: str, value: float) -> float:
    return round(value, 2 if name.endswith("_degrees") else 4 if name.endswith("_m") else 3)


def scored_cells(*, scene_revision_digest: str, scene_id: str, salt: str, task_spec: Mapping[str, Any],
                 task_kind: str, nominal_dynamic_friction: float | None = None,
                 count: int = EPISODES[SCORED]) -> list[dict[str, Any]]:
    """The scored matrix: ``count`` hidden cells over the task's bounded variation families."""

    if not isinstance(salt, str) or len(salt) < 32:
        raise TeamEvaluationPlanError("team_scored_matrix_salt_invalid")
    bounds = variation_bounds(task_spec, task_kind=task_kind, nominal_dynamic_friction=nominal_dynamic_friction)
    families = [(name, share) for name, share in SCORED_FAMILY_SHARES
                if name not in SINGLE_FAMILY_DIMENSIONS
                or any(dimension in bounds for dimension in SINGLE_FAMILY_DIMENSIONS[name])]
    total = sum(share for _, share in families)
    counts = {name: max(1, math.floor(count * share / total)) for name, share in families}
    counts["held_out_composition"] += count - sum(counts.values())
    root = (salt, scene_revision_digest, "team_scored")
    cells: list[dict[str, Any]] = []
    for family, family_count in counts.items():
        if family == "canonical_anchor":
            draws = [{} for _ in range(family_count)]
        elif family in SINGLE_FAMILY_DIMENSIONS:
            dimensions = [name for name in SINGLE_FAMILY_DIMENSIONS[family] if name in bounds]
            columns = {name: _stratified(*bounds[name], family_count, key=(*root, family, name))
                       for name in dimensions}
            draws = [{name: columns[name][index] for name in dimensions} for index in range(family_count)]
        else:
            varied = sorted(bounds)
            columns = {name: _stratified(*bounds[name], family_count, key=(*root, family, name)) for name in varied}
            draws = []
            for index in range(family_count):
                if family == "pairwise_stress":
                    pair = sorted(varied, key=lambda name: _unit(*root, family, index, "pick", name))[:2]
                    draws.append({name: columns[name][index] for name in pair})
                else:
                    draws.append({name: columns[name][index] for name in varied})
        for draw in draws:
            parameters = {name: _rounded(name, value) for name, value in sorted(draw.items())}
            ordinal = len(cells)
            scenario = {"family": family, "ordinal": ordinal, "parameters": parameters}
            cells.append({
                "cell_id": f"scene{scene_id}.scored{count}.{ordinal:03d}.{family}",
                "seed": int(hashlib.sha256(f"{salt}:{scene_revision_digest}:{ordinal}".encode()).hexdigest()[:8],
                            16) % (2**31),
                "family": family,
                "partition": "held_out" if family == "held_out_composition" else "scored",
                "resolved_scenario": scenario,
                "cell_spec_digest": canonical_digest(scenario),
            })
    seeds = [cell["seed"] for cell in cells]
    if len(cells) != count or len(set(seeds)) != count:
        raise TeamEvaluationPlanError("team_scored_matrix_not_unique")
    return cells


def campaign_ttl_seconds(*, control_hz: int = 15) -> int:
    """One cumulative lease covering every planned cell at the selected rate."""
    return shard_ttl_seconds(EPISODES[SCORED], control_hz=control_hz)


def runtime_cells(cells: Sequence[Mapping[str, Any]], *, control_diagnostic: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generated cells in the runtime-input shape, each carrying the scene's per-cell control binding."""

    rows = []
    for cell in cells:
        row = {key: deepcopy(cell[key]) for key in ("cell_id", "seed", "family", "partition",
                                                     "resolved_scenario", "cell_spec_digest")}
        row["resolved_scenario_digest"] = canonical_digest(row["resolved_scenario"])
        row["control_diagnostic"] = deepcopy(dict(control_diagnostic))
        rows.append(row)
    return rows


def shards(cells: Sequence[Mapping[str, Any]], *, size: int = SHARD_CELLS) -> list[list[dict[str, Any]]]:
    return [[dict(cell) for cell in cells[start:start + size]] for start in range(0, len(cells), size)]


__all__ = [
    "EPISODES",
    "FIRST_SHARD_INTERFACE_FAILURE_SHARE",
    "INTERFACE_FAILURE_REASONS",
    "MAXIMUM_VOID_SHARE",
    "PROFILES",
    "PURPOSES",
    "SCORED",
    "SCORED_PROFILE",
    "SHARD_CELLS",
    "TeamEvaluationPlanError",
    "profile_purpose",
    "runtime_cells",
    "scored_cells",
    "shard_ttl_seconds",
    "shards",
    "variation_bounds",
]
