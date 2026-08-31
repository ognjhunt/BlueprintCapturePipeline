"""Digest-bound plan for high-throughput development-only control search.

The funnel deliberately separates search from qualification.  cuRobo may
reject impossible candidates and Isaac Lab may compare lightweight physics
clones, but only later full-fidelity replay can qualify controls.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_collision_aware_candidate_generation import (
    INVENTORY_SCHEMA_VERSION,
)
from .task_evaluation_curobo_candidate_generator import CUROBO_BACKEND_IDENTITY


PLAN_SCHEMA_VERSION = "task_evaluation_control_search_funnel_plan.v1"
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


__all__ = [
    "CLAIM_CEILING",
    "ControlSearchFunnelError",
    "PLAN_SCHEMA_VERSION",
    "build_control_search_funnel_plan",
    "validate_control_search_funnel_plan",
]
