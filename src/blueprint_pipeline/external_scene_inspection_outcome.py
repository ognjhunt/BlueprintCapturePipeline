"""Deterministic outcome contracts for external-scene Franka inspection tasks.

The scorer ranks task-scoped controller or policy candidates only when the
runtime reports target-in-view geometry, stable reset, collision support, and a
nonblank wrist observation.  Scripted controller baselines never become learned
policy evidence, even when their task metric is valid.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


CONTRACT_SCHEMA = "external_scene_inspection_outcome_contract.v1"
RESULT_SCHEMA = "external_scene_inspection_candidate_ranking.v1"
POLICY_ACTION_SOURCES = {"learned_policy", "policy_endpoint", "vla_policy"}


class ExternalSceneInspectionOutcomeError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneInspectionOutcomeError(["inspection_value_not_json"]) from exc


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_franka_inspection_outcome_contract(
    *, target_analysis: Mapping[str, Any], placement_proposal_digest: str
) -> dict[str, Any]:
    analysis = _clone(dict(target_analysis))
    selected = analysis.get("selected_target")
    errors: list[str] = []
    if analysis.get("schema_version") != "scene_task_target_analysis_result.v1":
        errors.append("inspection_target_analysis_schema_invalid")
    if analysis.get("status") != "target_ready_for_bounded_sim":
        errors.append("inspection_target_not_ready")
    if not _digest(analysis.get("target_analysis_digest")):
        errors.append("inspection_target_analysis_digest_invalid")
    if not _digest(placement_proposal_digest):
        errors.append("inspection_placement_digest_invalid")
    if not isinstance(selected, Mapping):
        errors.append("inspection_selected_target_missing")
        selected = {}
    task_family = str(selected.get("task_family") or "")
    if "inspection" not in task_family:
        errors.append("inspection_task_family_required")
    position = selected.get("target_position_scene")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(_finite(item) is None for item in position)
    ):
        errors.append("inspection_target_position_invalid")
    uncertainty = _finite(selected.get("spatial_uncertainty_scene_units"))
    if uncertainty is None or uncertainty <= 0:
        errors.append("inspection_target_uncertainty_invalid")
    if errors:
        raise ExternalSceneInspectionOutcomeError(errors)
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "task_family": task_family,
        "target_region_id": str(selected["proposal_id"]),
        "target_position_scene": [round(float(item), 9) for item in position],
        "target_spatial_uncertainty_scene_units": round(float(uncertainty), 9),
        "target_analysis_digest": analysis["target_analysis_digest"],
        "placement_proposal_digest": placement_proposal_digest,
        "thresholds_frozen_before_candidate_execution": True,
        "thresholds": {
            "minimum_target_in_fov_fraction": 0.6,
            "maximum_view_axis_error_degrees": 30.0,
            "minimum_camera_target_distance_stage_units": 0.20,
            "maximum_camera_target_distance_stage_units": 0.90,
            "minimum_distinct_viewpoints": 2,
            "minimum_nonblank_terminal_observations": 1,
            "stable_reset_required": True,
            "collision_free_required": True,
        },
        "score_weights": {
            "target_in_fov_fraction": 0.4,
            "view_axis_alignment": 0.25,
            "camera_standoff": 0.2,
            "viewpoint_diversity": 0.15,
        },
        "metric_scale_status": analysis.get("metric_scale_status"),
        "candidate_may_self_authorize": False,
        "claim_boundary": {
            "stage_unit_metric_is_independent_metric_scale": False,
            "inspection_score_is_object_state_change": False,
            "controller_ranking_is_learned_policy_ranking": False,
            "simulator_result_is_physical_success": False,
        },
    }
    contract["contract_digest"] = canonical_digest(contract, digest_field="contract_digest")
    return contract


def _score_standoff(distance: float, minimum: float, maximum: float) -> float:
    if not minimum <= distance <= maximum:
        return 0.0
    midpoint = 0.5 * (minimum + maximum)
    half_width = 0.5 * (maximum - minimum)
    return max(0.0, 1.0 - abs(distance - midpoint) / max(half_width, 1e-9))


def rank_franka_inspection_candidates(
    *, contract: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    admitted = _clone(dict(contract))
    supplied_digest = admitted.pop("contract_digest", None)
    if admitted.get("schema_version") != CONTRACT_SCHEMA or supplied_digest != canonical_digest(
        admitted, digest_field="contract_digest"
    ):
        raise ExternalSceneInspectionOutcomeError(["inspection_contract_invalid"])
    admitted["contract_digest"] = supplied_digest
    thresholds = admitted["thresholds"]
    weights = admitted["score_weights"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(candidates):
        candidate = _clone(dict(raw))
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        blockers: list[str] = []
        if not candidate_id or candidate_id in seen:
            blockers.append("inspection_candidate_identity_invalid")
            candidate_id = candidate_id or f"invalid-candidate-{index}"
        seen.add(candidate_id)
        action_source = str(candidate.get("action_source") or "")
        learned = action_source in POLICY_ACTION_SOURCES
        if learned and not _digest(candidate.get("checkpoint_provenance_digest")):
            blockers.append("inspection_learned_policy_provenance_missing")
        if candidate.get("stable_reset_observed") is not True:
            blockers.append("inspection_stable_reset_missing")
        if candidate.get("collision_free_observed") is not True:
            blockers.append("inspection_collision_free_observation_missing")
        if candidate.get("terminal_egocentric_nonblank") is not True:
            blockers.append("inspection_terminal_egocentric_observation_invalid")
        observations = candidate.get("target_view_observations")
        observations = observations if isinstance(observations, list) else []
        valid = [row for row in observations if isinstance(row, Mapping)]
        if not valid:
            blockers.append("inspection_target_view_observations_missing")
        in_fov = [row for row in valid if row.get("target_in_fov") is True]
        in_fov_fraction = len(in_fov) / len(valid) if valid else 0.0
        angles = [_finite(row.get("view_axis_error_degrees")) for row in in_fov]
        distances = [_finite(row.get("camera_target_distance_stage_units")) for row in in_fov]
        if any(value is None for value in angles + distances):
            blockers.append("inspection_target_view_geometry_invalid")
        finite_angles = [float(value) for value in angles if value is not None]
        finite_distances = [float(value) for value in distances if value is not None]
        distinct_viewpoints = len(
            {
                str(row.get("viewpoint_bin") or "").strip()
                for row in in_fov
                if str(row.get("viewpoint_bin") or "").strip()
            }
        )
        if in_fov_fraction < float(thresholds["minimum_target_in_fov_fraction"]):
            blockers.append("inspection_target_in_fov_coverage_below_threshold")
        best_angle = min(finite_angles, default=math.inf)
        if best_angle > float(thresholds["maximum_view_axis_error_degrees"]):
            blockers.append("inspection_view_axis_error_above_threshold")
        distance_min = float(thresholds["minimum_camera_target_distance_stage_units"])
        distance_max = float(thresholds["maximum_camera_target_distance_stage_units"])
        best_standoff = max(
            (_score_standoff(value, distance_min, distance_max) for value in finite_distances),
            default=0.0,
        )
        if best_standoff <= 0.0:
            blockers.append("inspection_camera_standoff_outside_threshold")
        required_viewpoints = int(thresholds["minimum_distinct_viewpoints"])
        if distinct_viewpoints < required_viewpoints:
            blockers.append("inspection_viewpoint_diversity_below_threshold")
        angle_score = max(
            0.0,
            1.0 - best_angle / float(thresholds["maximum_view_axis_error_degrees"]),
        )
        diversity_score = min(1.0, distinct_viewpoints / max(required_viewpoints, 1))
        score = (
            float(weights["target_in_fov_fraction"]) * in_fov_fraction
            + float(weights["view_axis_alignment"]) * angle_score
            + float(weights["camera_standoff"]) * best_standoff
            + float(weights["viewpoint_diversity"]) * diversity_score
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "action_source": action_source,
                "learned_policy_evidence": learned and not blockers,
                "status": "qualified" if not blockers else "abstained",
                "blockers": sorted(set(blockers)),
                "inspection_score": round(score, 9) if not blockers else None,
                "target_in_fov_fraction": round(in_fov_fraction, 9),
                "best_view_axis_error_degrees": (
                    round(best_angle, 9) if math.isfinite(best_angle) else None
                ),
                "best_standoff_score": round(best_standoff, 9),
                "distinct_viewpoints": distinct_viewpoints,
            }
        )
    qualified = [row for row in rows if row["status"] == "qualified"]
    ordered = sorted(qualified, key=lambda row: (-row["inspection_score"], row["candidate_id"]))
    ranking = [
        {"rank": rank, "candidate_id": row["candidate_id"], "score": row["inspection_score"]}
        for rank, row in enumerate(ordered, start=1)
    ]
    controller_ranking = len(ordered) >= 2
    policy_ranking = controller_ranking and all(row["learned_policy_evidence"] for row in ordered)
    result = {
        "schema_version": RESULT_SCHEMA,
        "contract_digest": supplied_digest,
        "status": "completed" if controller_ranking else "abstained",
        "candidate_results": rows,
        "ranking": ranking,
        "controller_candidate_ranking_proven": controller_ranking,
        "learned_policy_ranking_proven": policy_ranking,
        "blockers": (
            [] if controller_ranking else ["fewer_than_two_qualified_inspection_candidates"]
        ),
        "claim_boundary": {
            "ranking_is_task_and_site_bound": True,
            "controller_ranking_is_learned_policy_ranking": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    result["ranking_digest"] = canonical_digest(result, digest_field="ranking_digest")
    return result


__all__ = [
    "CONTRACT_SCHEMA",
    "RESULT_SCHEMA",
    "ExternalSceneInspectionOutcomeError",
    "build_franka_inspection_outcome_contract",
    "rank_franka_inspection_candidates",
]
