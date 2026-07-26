"""Frozen deterministic aggregation for captured-site prospective rankings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .franka_can_tray_feasibility import _TRAY_CENTER
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "captured_site_policy_ranking.v1"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_episode(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered phase-gated progress formula to one episode."""
    if result.get("schema_version") != "franka_droid_closed_loop.v1":
        raise ValueError("unsupported_closed_loop_result_schema")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("closed_loop_metrics_missing")
    initial = result.get("initial_can_position_m")
    final = metrics.get("final_spraycan_center_m")
    if not isinstance(initial, Sequence) or len(initial) != 3:
        raise ValueError("closed_loop_initial_can_position_invalid")
    if not isinstance(final, Sequence) or len(final) != 3:
        raise ValueError("closed_loop_final_can_position_invalid")
    initial_xy_distance = math.hypot(
        float(initial[0]) - _TRAY_CENTER[0],
        float(initial[1]) - _TRAY_CENTER[1],
    )
    final_xy_distance = math.hypot(
        float(final[0]) - _TRAY_CENTER[0],
        float(final[1]) - _TRAY_CENTER[1],
    )
    lift_progress = _clip01(float(metrics.get("lift_delta_m", 0.0)) / 0.05)
    transport_progress = _clip01(
        (initial_xy_distance - final_xy_distance) / initial_xy_distance
        if initial_xy_distance > 0
        else 0.0
    )
    contained = bool(metrics.get("contained_in_tray_interior") is True)
    stable = bool(float(metrics.get("final_linear_speed_m_s", math.inf)) < 0.02)
    contract_valid = bool(
        result.get("status") == "completed"
        and isinstance(result.get("gates"), Mapping)
        and result["gates"].get("contract_valid") is True
    )
    score = 0.0
    if contract_valid:
        score = 0.4 * lift_progress
        if lift_progress == 1.0:
            score += 0.3 * transport_progress + 0.2 * float(contained)
            if contained:
                score += 0.1 * float(stable)
    return {
        "policy_id": str(result.get("policy_id") or ""),
        "contract_valid": contract_valid,
        "lift_progress": lift_progress,
        "transport_progress": transport_progress,
        "containment": contained,
        "stability": stable,
        "initial_xy_distance_to_tray_m": initial_xy_distance,
        "final_xy_distance_to_tray_m": final_xy_distance,
        "episode_progress_score": score,
    }


def aggregate_policy_rankings(
    episodes_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Rank only when all three-variant score intervals are strictly separated."""
    summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for policy_id, episodes in sorted(episodes_by_policy.items()):
        if len(episodes) != 3:
            blockers.append(f"policy_variant_count_not_three:{policy_id}")
            continue
        scored = [score_episode(result) for result in episodes]
        if any(row["policy_id"] != policy_id for row in scored):
            blockers.append(f"policy_episode_identity_mismatch:{policy_id}")
            continue
        scores = [float(row["episode_progress_score"]) for row in scored]
        summaries.append(
            {
                "policy_id": policy_id,
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "episodes": scored,
            }
        )
    ordered = sorted(summaries, key=lambda row: (-float(row["mean_score"]), row["policy_id"]))
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            separated = float(left["min_score"]) > float(right["max_score"])
            pairwise.append(
                {
                    "higher_mean_policy_id": left["policy_id"],
                    "lower_mean_policy_id": right["policy_id"],
                    "intervals_strictly_separated": separated,
                    "decision": "ordered" if separated else "abstain",
                }
            )
    adjacent_separated = all(
        float(ordered[index]["min_score"]) > float(ordered[index + 1]["max_score"])
        for index in range(max(0, len(ordered) - 1))
    )
    total_ranking_emitted = bool(len(ordered) >= 2 and not blockers and adjacent_separated)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "policy_summaries": ordered,
        "pairwise": pairwise,
        "total_ranking_emitted": total_ranking_emitted,
        "ranking": [row["policy_id"] for row in ordered] if total_ranking_emitted else None,
        "abstained": not total_ranking_emitted,
        "blockers": blockers,
        "claim_boundary": {
            "prospective_externally_calibrated_prediction": True,
            "site_specific_physical_success_proven": False,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


__all__ = ["aggregate_policy_rankings", "score_episode"]
