"""Persist and revalidate deterministic CPU robot-placement inventories."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_robot_placement_geometry import (
    RobotPlacementGeometryIndex,
    validate_robot_placement_geometry_candidate,
)


CANDIDATE_INVENTORY_SCHEMA_VERSION = (
    "task_evaluation_robot_placement_candidate_inventory.v1"
)


def build_candidate_inventory_checkpoint(
    *,
    robot_id: str,
    target_position_world_m: Sequence[float],
    maximum_candidates: int,
    trajectory_digest: str | None,
    geometry_summary_digest: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_rows = [dict(candidate) for candidate in candidates]
    inventory_digest = canonical_digest(
        {"trajectory_digest": trajectory_digest, "candidates": candidate_rows}
    )
    result: dict[str, Any] = {
        "schema_version": CANDIDATE_INVENTORY_SCHEMA_VERSION,
        "status": "complete",
        "robot_id": robot_id,
        "target_position_world_m": [
            float(value) for value in target_position_world_m
        ],
        "maximum_candidates": int(maximum_candidates),
        "trajectory_digest": trajectory_digest,
        "geometry_summary_digest": geometry_summary_digest,
        "candidates": candidate_rows,
        "candidate_inventory_digest": inventory_digest,
        "checkpoint_digest": "",
    }
    result["checkpoint_digest"] = canonical_digest(
        result, digest_field="checkpoint_digest"
    )
    return result


def validate_candidate_inventory_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    index: RobotPlacementGeometryIndex,
    robot_id: str,
    target_position_world_m: Sequence[float],
    maximum_candidates: int,
    trajectory_digest: str | None,
    geometry_summary_digest: str,
) -> list[dict[str, Any]]:
    value = dict(checkpoint)
    candidates_value = value.get("candidates")
    expected_target = [float(row) for row in target_position_world_m]
    if (
        value.get("schema_version") != CANDIDATE_INVENTORY_SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("robot_id") != robot_id
        or value.get("target_position_world_m") != expected_target
        or value.get("maximum_candidates") != int(maximum_candidates)
        or value.get("trajectory_digest") != trajectory_digest
        or value.get("geometry_summary_digest") != geometry_summary_digest
        or value.get("checkpoint_digest")
        != canonical_digest(value, digest_field="checkpoint_digest")
        or not isinstance(candidates_value, list)
        or not 1 <= len(candidates_value) <= int(maximum_candidates)
    ):
        raise ValueError("robot_placement_candidate_inventory_checkpoint_invalid")
    candidates = [dict(row) for row in candidates_value if isinstance(row, Mapping)]
    if len(candidates) != len(candidates_value):
        raise ValueError("robot_placement_candidate_inventory_checkpoint_invalid")
    inventory_digest = canonical_digest(
        {"trajectory_digest": trajectory_digest, "candidates": candidates}
    )
    if value.get("candidate_inventory_digest") != inventory_digest:
        raise ValueError("robot_placement_candidate_inventory_checkpoint_invalid")
    for candidate in candidates:
        trajectory_gate = candidate.get("trajectory_position_ik_gate")
        if not isinstance(trajectory_gate, Mapping):
            raise ValueError("robot_placement_candidate_inventory_checkpoint_invalid")
        gate = validate_robot_placement_geometry_candidate(
            index=index,
            proposal=candidate,
            target_position_world_m=expected_target,
            robot_id=robot_id,
            trajectory_gate_override=trajectory_gate,
        )
        if (
            gate.get("status") != "passed"
            or candidate.get("geometry_gate_digest")
            != gate.get("geometry_gate_digest")
            or candidate.get("trajectory_position_ik_gate_digest")
            != trajectory_gate.get("trajectory_position_ik_gate_digest")
            or candidate.get("trajectory_minimum_manipulability")
            != trajectory_gate.get("minimum_manipulability")
            or candidate.get("shoulder_to_target_distance_m")
            != gate.get("shoulder_to_target_distance_m")
        ):
            raise ValueError("robot_placement_candidate_inventory_checkpoint_invalid")
    return candidates


__all__ = [
    "CANDIDATE_INVENTORY_SCHEMA_VERSION",
    "build_candidate_inventory_checkpoint",
    "validate_candidate_inventory_checkpoint",
]
