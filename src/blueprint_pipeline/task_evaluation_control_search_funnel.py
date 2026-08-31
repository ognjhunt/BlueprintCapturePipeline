"""Digest-bound plan for high-throughput development-only control search.

The funnel deliberately separates search from qualification.  cuRobo may
reject impossible candidates and Isaac Lab may compare lightweight physics
clones, but only later full-fidelity replay can qualify controls.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import validate_native_task_arena_packet_request
from .task_evaluation_collision_aware_candidate_generation import (
    INVENTORY_SCHEMA_VERSION,
)
from .task_evaluation_curobo_candidate_generator import CUROBO_BACKEND_IDENTITY


PLAN_SCHEMA_VERSION = "task_evaluation_control_search_funnel_plan.v1"
OUTCOME_SCHEMA_VERSION = "task_evaluation_control_search_vector_outcome.v1"
SWEEP_RESULT_SCHEMA_VERSION = "task_evaluation_control_search_sweep_result.v1"
REPLAY_PLAN_SCHEMA_VERSION = (
    "task_evaluation_control_search_full_fidelity_replay_plan.v1"
)
CLAIM_CEILING = "development_only_control_search"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class ControlSearchFunnelError(ValueError):
    """The search plan or one of its immutable inputs was invalid."""


def _copy(value: Mapping[str, Any], *, blocker: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ControlSearchFunnelError(blocker) from exc
    if not isinstance(result, dict):
        raise ControlSearchFunnelError(blocker)
    return result


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _finite_nonnegative(value: object, *, blocker: str) -> float:
    if isinstance(value, bool):
        raise ControlSearchFunnelError(blocker)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlSearchFunnelError(blocker) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ControlSearchFunnelError(blocker)
    return result


def _candidate_index(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen = _copy(
        inventory, blocker="control_search_candidate_inventory_invalid"
    )
    candidates = frozen.get("candidates")
    if (
        frozen.get("schema_version") != INVENTORY_SCHEMA_VERSION
        or frozen.get("model_authored_candidates") is not False
        or frozen.get("inventory_digest")
        != canonical_digest(frozen, digest_field="inventory_digest")
        or not isinstance(candidates, list)
        or not candidates
        or len(candidates) > 10_000
    ):
        raise ControlSearchFunnelError(
            "control_search_candidate_inventory_invalid"
        )
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ControlSearchFunnelError(
                "control_search_candidate_inventory_invalid"
            )
        candidate_id = candidate.get("candidate_id")
        candidate_digest = candidate.get("candidate_digest")
        deterministic_rank = candidate.get("deterministic_rank")
        if (
            not isinstance(candidate_id, str)
            or not _IDENTIFIER.fullmatch(candidate_id)
            or not _digest(candidate_digest)
            or candidate_digest
            != canonical_digest(candidate, digest_field="candidate_digest")
            or not isinstance(deterministic_rank, int)
            or isinstance(deterministic_rank, bool)
            or deterministic_rank < 0
        ):
            raise ControlSearchFunnelError(
                "control_search_candidate_inventory_invalid"
            )
        result.append(
            {
                "candidate_id": candidate_id,
                "candidate_digest": candidate_digest,
                "deterministic_rank": deterministic_rank,
            }
        )
    ids = [row["candidate_id"] for row in result]
    digests = [row["candidate_digest"] for row in result]
    if len(ids) != len(set(ids)) or len(digests) != len(set(digests)):
        raise ControlSearchFunnelError(
            "control_search_candidate_inventory_invalid"
        )
    return sorted(
        result,
        key=lambda row: (row["deterministic_rank"], row["candidate_id"]),
    )


def build_control_search_funnel_plan(
    *,
    run_id: str,
    source_commit: str,
    packet_request_digest: str,
    candidate_inventory: Mapping[str, Any],
    runtime_source_packet_digest: str,
    scene_collision_digest: str,
    task_object_asset_digest: str,
    robot_configuration_digest: str,
    task_scoring_digest: str,
    requested_vector_env_count: int = 256,
    maximum_vector_env_count: int = 1_024,
    seeds_per_candidate: int = 1,
    shortlist_size: int = 16,
) -> dict[str, Any]:
    """Freeze one cuRobo -> Isaac Lab -> exact replay search plan."""

    if (
        not isinstance(run_id, str)
        or not _IDENTIFIER.fullmatch(run_id)
        or not isinstance(source_commit, str)
        or not _COMMIT.fullmatch(source_commit)
        or not all(
            _digest(value)
            for value in (
                packet_request_digest,
                runtime_source_packet_digest,
                scene_collision_digest,
                task_object_asset_digest,
                robot_configuration_digest,
                task_scoring_digest,
            )
        )
        or not isinstance(requested_vector_env_count, int)
        or isinstance(requested_vector_env_count, bool)
        or requested_vector_env_count < 1
        or not isinstance(maximum_vector_env_count, int)
        or isinstance(maximum_vector_env_count, bool)
        or not 1 <= maximum_vector_env_count <= 1_024
        or requested_vector_env_count > maximum_vector_env_count
        or not isinstance(seeds_per_candidate, int)
        or isinstance(seeds_per_candidate, bool)
        or not 1 <= seeds_per_candidate <= 16
        or not isinstance(shortlist_size, int)
        or isinstance(shortlist_size, bool)
        or not 8 <= shortlist_size <= 32
    ):
        raise ControlSearchFunnelError("control_search_plan_input_invalid")
    candidates = _candidate_index(candidate_inventory)
    assignment_count = len(candidates) * seeds_per_candidate
    vector_env_count = min(requested_vector_env_count, assignment_count)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "planned",
        "run_id": run_id,
        "source_commit": source_commit,
        "claim_ceiling": CLAIM_CEILING,
        "qualification_effect": "none_until_full_fidelity_replay",
        "immutable_inputs": {
            "packet_request_digest": packet_request_digest,
            "candidate_inventory_digest": candidate_inventory["inventory_digest"],
            "runtime_source_packet_digest": runtime_source_packet_digest,
            "scene_collision_digest": scene_collision_digest,
            "task_object_asset_digest": task_object_asset_digest,
            "robot_configuration_digest": robot_configuration_digest,
            "task_scoring_digest": task_scoring_digest,
        },
        "candidate_index": candidates,
        "kinematic_filter": {
            "backend_identity": _copy(
                CUROBO_BACKEND_IDENTITY,
                blocker="control_search_curobo_identity_invalid",
            ),
            "batched_gpu_required": True,
            "full_trajectory_required": True,
            "rejection_authority": [
                "ik_unreachable",
                "joint_limit_invalid",
                "collision_invalid",
            ],
            "native_task_success_unresolved": True,
        },
        "vector_sweep": {
            "backend": "isaac_lab_vectorized_lightweight_physics",
            "requested_vector_env_count": requested_vector_env_count,
            "maximum_vector_env_count": maximum_vector_env_count,
            "resolved_vector_env_count": vector_env_count,
            "seeds_per_candidate": seeds_per_candidate,
            "assignment_count": assignment_count,
            "wave_count": math.ceil(assignment_count / vector_env_count),
            "appearance_mode": "omitted",
            "camera_mode": "disabled",
            "collision_authority": "exact_scene_collision_digest",
            "task_object_authority": "exact_task_object_asset_digest",
            "robot_object_task_scoring_exact": True,
        },
        "shortlist": {
            "requested_size": shortlist_size,
            "resolved_maximum_size": min(shortlist_size, len(candidates)),
            "ranking": "deterministic_lexicographic_physics_v1",
            "learned_grader_used": False,
        },
        "full_fidelity_replay": {
            "required": True,
            "particlefield_required": True,
            "cameras_required": True,
            "reset_readback_required": True,
            "native_gates_unchanged": True,
            "search_result_alone_may_not_qualify_controls": True,
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def validate_control_search_funnel_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen plan before a GPU worker may consume it."""

    plan = _copy(value, blocker="control_search_plan_invalid")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "planned"
        or plan.get("claim_ceiling") != CLAIM_CEILING
        or plan.get("qualification_effect")
        != "none_until_full_fidelity_replay"
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
        or not isinstance(plan.get("candidate_index"), list)
        or not plan["candidate_index"]
        or (plan.get("vector_sweep") or {}).get("appearance_mode") != "omitted"
        or (plan.get("vector_sweep") or {}).get("camera_mode") != "disabled"
        or (plan.get("shortlist") or {}).get("learned_grader_used") is not False
        or (plan.get("full_fidelity_replay") or {}).get(
            "search_result_alone_may_not_qualify_controls"
        )
        is not True
    ):
        raise ControlSearchFunnelError("control_search_plan_invalid")
    return plan


def _validated_outcome(
    value: Mapping[str, Any],
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    seeds_per_candidate: int,
) -> dict[str, Any]:
    outcome = _copy(value, blocker="control_search_vector_outcome_invalid")
    candidate_id = outcome.get("candidate_id")
    candidate = candidates.get(str(candidate_id))
    if (
        outcome.get("schema_version") != OUTCOME_SCHEMA_VERSION
        or candidate is None
        or outcome.get("candidate_digest") != candidate["candidate_digest"]
        or not isinstance(outcome.get("seed_index"), int)
        or isinstance(outcome.get("seed_index"), bool)
        or not 0 <= outcome["seed_index"] < seeds_per_candidate
        or not isinstance(outcome.get("resolved_seed"), int)
        or isinstance(outcome.get("resolved_seed"), bool)
        or outcome["resolved_seed"] < 0
        or not isinstance(outcome.get("wave_index"), int)
        or isinstance(outcome.get("wave_index"), bool)
        or outcome["wave_index"] < 0
        or not isinstance(outcome.get("environment_index"), int)
        or isinstance(outcome.get("environment_index"), bool)
        or outcome["environment_index"] < 0
        or outcome.get("reset_readback_passed") not in {True, False}
        or not isinstance(outcome.get("physics_steps"), int)
        or isinstance(outcome.get("physics_steps"), bool)
        or outcome["physics_steps"] < 1
        or outcome.get("measurement_authority")
        != "isaac_lab_simulator_state_and_contact_sensors"
        or outcome.get("learned_grader_used") is not False
        or outcome.get("outcome_digest")
        != canonical_digest(outcome, digest_field="outcome_digest")
    ):
        raise ControlSearchFunnelError("control_search_vector_outcome_invalid")
    for field in (
        "forbidden_collision_peak_force_n",
        "required_task_contact_coverage_fraction",
        "push_path_tracking_error_m",
        "destination_error_m",
        "support_stability_error_m",
        "task_displacement_m",
    ):
        outcome[field] = _finite_nonnegative(
            outcome.get(field), blocker="control_search_vector_outcome_invalid"
        )
    if outcome["required_task_contact_coverage_fraction"] > 1.0:
        raise ControlSearchFunnelError("control_search_vector_outcome_invalid")
    return outcome


def _aggregate_candidate_outcomes(
    *,
    candidate: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["candidate_digest"],
        "source_deterministic_rank": candidate["deterministic_rank"],
        "seed_count": len(outcomes),
        "all_resets_passed": all(
            row["reset_readback_passed"] is True for row in outcomes
        ),
        "worst_forbidden_collision_peak_force_n": max(
            row["forbidden_collision_peak_force_n"] for row in outcomes
        ),
        "worst_required_task_contact_coverage_fraction": min(
            row["required_task_contact_coverage_fraction"] for row in outcomes
        ),
        "worst_push_path_tracking_error_m": max(
            row["push_path_tracking_error_m"] for row in outcomes
        ),
        "worst_destination_error_m": max(
            row["destination_error_m"] for row in outcomes
        ),
        "worst_support_stability_error_m": max(
            row["support_stability_error_m"] for row in outcomes
        ),
        "worst_task_displacement_m": min(
            row["task_displacement_m"] for row in outcomes
        ),
        "outcome_digests": [row["outcome_digest"] for row in outcomes],
        "aggregate_digest": "",
    }
    aggregate["aggregate_digest"] = canonical_digest(
        aggregate, digest_field="aggregate_digest"
    )
    return aggregate


def _ranking_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value["all_resets_passed"] is not True,
        value["worst_forbidden_collision_peak_force_n"],
        -value["worst_required_task_contact_coverage_fraction"],
        value["worst_push_path_tracking_error_m"],
        value["worst_destination_error_m"],
        value["worst_support_stability_error_m"],
        -value["worst_task_displacement_m"],
        value["source_deterministic_rank"],
        value["candidate_id"],
    )


def build_control_search_sweep_result(
    *,
    plan: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    actual_vector_env_count: int,
    peak_gpu_memory_bytes: int,
) -> dict[str, Any]:
    """Seal raw Isaac Lab outcomes and deterministically choose a shortlist."""

    frozen_plan = validate_control_search_funnel_plan(plan)
    vector = frozen_plan["vector_sweep"]
    if (
        not isinstance(actual_vector_env_count, int)
        or isinstance(actual_vector_env_count, bool)
        or not 1
        <= actual_vector_env_count
        <= vector["maximum_vector_env_count"]
        or not isinstance(peak_gpu_memory_bytes, int)
        or isinstance(peak_gpu_memory_bytes, bool)
        or peak_gpu_memory_bytes < 1
        or not isinstance(outcomes, Sequence)
        or isinstance(outcomes, (str, bytes))
    ):
        raise ControlSearchFunnelError("control_search_sweep_result_invalid")
    candidates = {
        row["candidate_id"]: row for row in frozen_plan["candidate_index"]
    }
    validated = [
        _validated_outcome(
            row,
            candidates=candidates,
            seeds_per_candidate=vector["seeds_per_candidate"],
        )
        for row in outcomes
        if isinstance(row, Mapping)
    ]
    if len(validated) != len(outcomes):
        raise ControlSearchFunnelError("control_search_sweep_result_invalid")
    grouped: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidates
    }
    assignments: set[tuple[int, int]] = set()
    for outcome in validated:
        assignment = (outcome["wave_index"], outcome["environment_index"])
        if assignment in assignments:
            raise ControlSearchFunnelError("control_search_sweep_result_invalid")
        assignments.add(assignment)
        grouped[outcome["candidate_id"]].append(outcome)
    expected_seeds = set(range(vector["seeds_per_candidate"]))
    if any(
        {row["seed_index"] for row in rows} != expected_seeds
        or len(rows) != vector["seeds_per_candidate"]
        for rows in grouped.values()
    ):
        raise ControlSearchFunnelError("control_search_sweep_result_incomplete")
    validated.sort(
        key=lambda row: (
            candidates[row["candidate_id"]]["deterministic_rank"],
            row["candidate_id"],
            row["seed_index"],
        )
    )
    aggregates = [
        _aggregate_candidate_outcomes(
            candidate=candidate,
            outcomes=sorted(
                grouped[candidate["candidate_id"]],
                key=lambda row: row["seed_index"],
            ),
        )
        for candidate in frozen_plan["candidate_index"]
    ]
    aggregates.sort(key=_ranking_key)
    ranked = []
    for rank, aggregate in enumerate(aggregates):
        row = dict(aggregate)
        row["control_search_rank"] = rank
        row["ranking_key"] = {
            "all_resets_passed": row["all_resets_passed"],
            "worst_forbidden_collision_peak_force_n": row[
                "worst_forbidden_collision_peak_force_n"
            ],
            "worst_required_task_contact_coverage_fraction": row[
                "worst_required_task_contact_coverage_fraction"
            ],
            "worst_push_path_tracking_error_m": row[
                "worst_push_path_tracking_error_m"
            ],
            "worst_destination_error_m": row["worst_destination_error_m"],
            "worst_support_stability_error_m": row[
                "worst_support_stability_error_m"
            ],
            "worst_task_displacement_m": row["worst_task_displacement_m"],
        }
        ranked.append(row)
    shortlist_size = frozen_plan["shortlist"]["resolved_maximum_size"]
    result: dict[str, Any] = {
        "schema_version": SWEEP_RESULT_SCHEMA_VERSION,
        "status": "completed_development_only",
        "run_id": frozen_plan["run_id"],
        "source_commit": frozen_plan["source_commit"],
        "plan_digest": frozen_plan["plan_digest"],
        "claim_ceiling": CLAIM_CEILING,
        "qualification_effect": "none_until_full_fidelity_replay",
        "actual_vector_env_count": actual_vector_env_count,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "outcomes": validated,
        "ranked_candidates": ranked,
        "shortlist": [
            {
                "candidate_id": row["candidate_id"],
                "candidate_digest": row["candidate_digest"],
                "aggregate_digest": row["aggregate_digest"],
                "control_search_rank": row["control_search_rank"],
            }
            for row in ranked[:shortlist_size]
        ],
        "learned_grader_used": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


def validate_control_search_sweep_result(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen a sweep receipt and its exact plan binding."""

    result = _copy(value, blocker="control_search_sweep_result_invalid")
    frozen_plan = validate_control_search_funnel_plan(plan)
    if (
        result.get("schema_version") != SWEEP_RESULT_SCHEMA_VERSION
        or result.get("status") != "completed_development_only"
        or result.get("plan_digest") != frozen_plan["plan_digest"]
        or result.get("claim_ceiling") != CLAIM_CEILING
        or result.get("qualification_effect")
        != "none_until_full_fidelity_replay"
        or result.get("learned_grader_used") is not False
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or not isinstance(result.get("shortlist"), list)
        or not 1
        <= len(result["shortlist"])
        <= frozen_plan["shortlist"]["resolved_maximum_size"]
    ):
        raise ControlSearchFunnelError("control_search_sweep_result_invalid")
    return result


def build_full_fidelity_replay_plan(
    *,
    plan: Mapping[str, Any],
    sweep_result: Mapping[str, Any],
    full_fidelity_packet_request: Mapping[str, Any],
    camera_configuration_digest: str,
) -> dict[str, Any]:
    """Bind the development-only shortlist to strict exact-scene replay."""

    frozen_plan = validate_control_search_funnel_plan(plan)
    sweep = validate_control_search_sweep_result(
        sweep_result, plan=frozen_plan
    )
    try:
        packet = validate_native_task_arena_packet_request(
            full_fidelity_packet_request
        )
    except ValueError as exc:
        raise ControlSearchFunnelError(
            "control_search_full_fidelity_packet_invalid"
        ) from exc
    appearance = packet.get("appearance_variant")
    cameras = packet.get("cameras")
    collision_assets = [
        row
        for row in packet.get("assets") or []
        if isinstance(row, Mapping)
        and row.get("semantic_role") == "scene_collision"
    ]
    task_assets = [
        row
        for row in packet.get("assets") or []
        if isinstance(row, Mapping)
        and row.get("semantic_role") == "task_object"
    ]
    if (
        not _digest(camera_configuration_digest)
        or not isinstance(appearance, Mapping)
        or appearance.get("representation")
        != "particlefield_3d_gaussian_splat"
        or not isinstance(cameras, list)
        or {row.get("role") for row in cameras if isinstance(row, Mapping)}
        != {"external", "wrist", "overview"}
        or len(collision_assets) != 1
        or (collision_assets[0].get("source") or {}).get("sha256")
        != frozen_plan["immutable_inputs"]["scene_collision_digest"]
        or len(task_assets) != 1
        or (task_assets[0].get("source") or {}).get("sha256")
        != frozen_plan["immutable_inputs"]["task_object_asset_digest"]
    ):
        raise ControlSearchFunnelError(
            "control_search_full_fidelity_packet_invalid"
        )
    replay_rows = []
    for replay_index, selected in enumerate(sweep["shortlist"]):
        replay_rows.append(
            {
                "replay_index": replay_index,
                "candidate_id": selected["candidate_id"],
                "candidate_digest": selected["candidate_digest"],
                "control_search_rank": selected["control_search_rank"],
                "control_search_aggregate_digest": selected[
                    "aggregate_digest"
                ],
                "full_fidelity_packet_request_digest": packet[
                    "request_digest"
                ],
                "reset_before_replay_required": True,
                "exact_candidate_application_required": True,
            }
        )
    replay: dict[str, Any] = {
        "schema_version": REPLAY_PLAN_SCHEMA_VERSION,
        "status": "ready_for_full_fidelity_replay",
        "run_id": frozen_plan["run_id"],
        "source_commit": frozen_plan["source_commit"],
        "control_search_plan_digest": frozen_plan["plan_digest"],
        "control_search_result_digest": sweep["result_digest"],
        "claim_ceiling_before_replay": CLAIM_CEILING,
        "qualification_effect_before_replay": "none",
        "full_fidelity_bindings": {
            "packet_request_digest": packet["request_digest"],
            "appearance_representation": appearance["representation"],
            "gaussian_field_quality": _copy(
                appearance["gaussian_field_quality"],
                blocker="control_search_full_fidelity_packet_invalid",
            ),
            "scene_collision_digest": frozen_plan["immutable_inputs"][
                "scene_collision_digest"
            ],
            "task_object_asset_digest": frozen_plan["immutable_inputs"][
                "task_object_asset_digest"
            ],
            "camera_configuration_digest": camera_configuration_digest,
            "camera_roles": ["external", "wrist", "overview"],
        },
        "replays": replay_rows,
        "replay_count": len(replay_rows),
        "requirements": {
            "particlefield_render_required": True,
            "camera_evidence_required": True,
            "reset_readback_required": True,
            "native_orientation_collision_contact_and_task_gates_required": True,
            "deterministic_simulator_state_scoring_required": True,
            "learned_grader_used": False,
            "each_replay_must_seal_terminal_evidence": True,
        },
        "replay_plan_digest": "",
    }
    replay["replay_plan_digest"] = canonical_digest(
        replay, digest_field="replay_plan_digest"
    )
    return replay


def validate_full_fidelity_replay_plan(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    sweep_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the bridge without treating readiness as execution evidence."""

    replay = _copy(value, blocker="control_search_replay_plan_invalid")
    frozen_plan = validate_control_search_funnel_plan(plan)
    sweep = validate_control_search_sweep_result(
        sweep_result, plan=frozen_plan
    )
    if (
        replay.get("schema_version") != REPLAY_PLAN_SCHEMA_VERSION
        or replay.get("status") != "ready_for_full_fidelity_replay"
        or replay.get("control_search_plan_digest") != frozen_plan["plan_digest"]
        or replay.get("control_search_result_digest") != sweep["result_digest"]
        or replay.get("claim_ceiling_before_replay") != CLAIM_CEILING
        or replay.get("qualification_effect_before_replay") != "none"
        or replay.get("replay_count") != len(sweep["shortlist"])
        or not isinstance(replay.get("replays"), list)
        or len(replay["replays"]) != replay["replay_count"]
        or (replay.get("requirements") or {}).get("learned_grader_used")
        is not False
        or replay.get("replay_plan_digest")
        != canonical_digest(replay, digest_field="replay_plan_digest")
    ):
        raise ControlSearchFunnelError("control_search_replay_plan_invalid")
    return replay


__all__ = [
    "CLAIM_CEILING",
    "ControlSearchFunnelError",
    "OUTCOME_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "REPLAY_PLAN_SCHEMA_VERSION",
    "SWEEP_RESULT_SCHEMA_VERSION",
    "build_control_search_funnel_plan",
    "build_control_search_sweep_result",
    "build_full_fidelity_replay_plan",
    "validate_control_search_funnel_plan",
    "validate_control_search_sweep_result",
    "validate_full_fidelity_replay_plan",
]
