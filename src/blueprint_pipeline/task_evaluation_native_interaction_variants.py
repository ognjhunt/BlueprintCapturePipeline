"""Deterministic full-stage solver branches for native construction feedback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest


def build_native_interaction_variants(
    *,
    phases: Sequence[Mapping[str, Any]],
    first_phase_id: str,
    base_rank: int,
    trajectory_digest: str,
) -> list[dict[str, Any]]:
    options = (
        (
            "uniform_seed",
            1009,
            ("gate_failed:base_collision_clearance", "gate_failed:push_contact_maintained"),
        ),
        (
            "contact_ramp",
            8928,
            ("gate_failed:base_collision_clearance", "gate_failed:push_contact_maintained"),
        ),
        (
            "push_contact_dense",
            16847,
            ("gate_failed:push_contact_maintained", "gate_failed:push_path"),
        ),
        (
            "release_retreat_dense",
            24766,
            ("gate_failed:destination_containment", "gate_failed:push_path"),
        ),
    )
    waypoints = []
    for phase in phases:
        phase_id = str(phase["phase_id"])
        if "release" in phase_id:
            stage_kind = "release"
        elif any(token in phase_id for token in ("retreat", "recovery")):
            stage_kind = "retreat"
        elif phase_id != first_phase_id:
            stage_kind = "contact"
        else:
            continue
        waypoints.append(
            {
                "source_native_phase_id": phase_id,
                "stage_kind": stage_kind,
                "target_position_world_m": list(phase["position_world_m"]),
                "target_orientation_world_xyzw": list(
                    phase["orientation_world_xyzw"]
                ),
                "authored_tcp_endpoint": True,
            }
        )
    result = []
    for branch_id, seed_offset, feedback_codes in options:
        variant: dict[str, Any] = {
            "schema_version": "task_evaluation_native_interaction_trajectory_variant.v1",
            "interaction_branch_id": branch_id,
            "solver_seed": int(base_rank) * 65537 + seed_offset,
            "source_normalized_trajectory_digest": trajectory_digest,
            "preserves_authored_tcp_endpoints": True,
            "acceptance_criteria_immutable": True,
            "waypoints": waypoints,
            "interaction_trajectory_variant_digest": "",
        }
        variant["interaction_trajectory_variant_digest"] = canonical_digest(
            variant, digest_field="interaction_trajectory_variant_digest"
        )
        result.append(
            {
                "branch_id": branch_id,
                "feedback_codes": list(feedback_codes),
                "variant": variant,
            }
        )
    return result


__all__ = ["build_native_interaction_variants"]
