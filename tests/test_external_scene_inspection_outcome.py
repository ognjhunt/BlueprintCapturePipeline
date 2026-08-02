from __future__ import annotations

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_scene_inspection_outcome import (
    build_franka_inspection_outcome_contract,
    rank_franka_inspection_candidates,
)


D = ["sha256:" + character * 64 for character in "abc"]


def _analysis() -> dict:
    value = {
        "schema_version": "scene_task_target_analysis_result.v1",
        "status": "target_ready_for_bounded_sim",
        "scene_id": "private-apartment-001",
        "source_scene_digest": D[0],
        "metric_scale_status": "provider_declared_not_independently_validated",
        "selected_target": {
            "proposal_id": "third-person-sink-001",
            "object_label": "sink",
            "task_family": "franka_sink_inspection",
            "target_position_scene": [0.36, 0.48, -3.19],
            "spatial_uncertainty_scene_units": 0.35,
        },
    }
    value["target_analysis_digest"] = canonical_digest(
        value, digest_field="target_analysis_digest"
    )
    return value


def _candidate(candidate_id: str, *, angle: float, distance: float, source: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "action_source": source,
        "stable_reset_observed": True,
        "collision_free_observed": True,
        "terminal_egocentric_nonblank": True,
        "target_view_observations": [
            {
                "target_in_fov": True,
                "view_axis_error_degrees": angle,
                "camera_target_distance_stage_units": distance,
                "viewpoint_bin": "left",
            },
            {
                "target_in_fov": True,
                "view_axis_error_degrees": angle + 2.0,
                "camera_target_distance_stage_units": distance + 0.02,
                "viewpoint_bin": "right",
            },
        ],
    }


def test_ranks_qualified_controller_baselines_without_calling_them_policies() -> None:
    contract = build_franka_inspection_outcome_contract(
        target_analysis=_analysis(), placement_proposal_digest=D[1]
    )
    result = rank_franka_inspection_candidates(
        contract=contract,
        candidates=[
            _candidate("wide-sweep", angle=18.0, distance=0.65, source="scripted_controller"),
            _candidate("aligned-sweep", angle=5.0, distance=0.54, source="scripted_controller"),
        ],
    )

    assert result["status"] == "completed"
    assert result["ranking"][0]["candidate_id"] == "aligned-sweep"
    assert result["controller_candidate_ranking_proven"] is True
    assert result["learned_policy_ranking_proven"] is False


def test_learned_policy_without_checkpoint_provenance_abstains() -> None:
    contract = build_franka_inspection_outcome_contract(
        target_analysis=_analysis(), placement_proposal_digest=D[1]
    )
    candidate = _candidate("unbound-policy", angle=5.0, distance=0.54, source="learned_policy")

    result = rank_franka_inspection_candidates(contract=contract, candidates=[candidate])

    row = result["candidate_results"][0]
    assert row["status"] == "abstained"
    assert "inspection_learned_policy_provenance_missing" in row["blockers"]
    assert result["learned_policy_ranking_proven"] is False


def test_missing_collision_or_viewpoint_diversity_abstains() -> None:
    contract = build_franka_inspection_outcome_contract(
        target_analysis=_analysis(), placement_proposal_digest=D[1]
    )
    candidate = _candidate("unsafe", angle=5.0, distance=0.54, source="scripted_controller")
    candidate["collision_free_observed"] = False
    candidate["target_view_observations"][1]["viewpoint_bin"] = "left"

    result = rank_franka_inspection_candidates(contract=contract, candidates=[candidate])

    blockers = result["candidate_results"][0]["blockers"]
    assert "inspection_collision_free_observation_missing" in blockers
    assert "inspection_viewpoint_diversity_below_threshold" in blockers
    assert result["status"] == "abstained"


def test_two_provenance_bound_learned_candidates_can_support_policy_ranking() -> None:
    contract = build_franka_inspection_outcome_contract(
        target_analysis=_analysis(), placement_proposal_digest=D[1]
    )
    candidates = [
        _candidate("policy-a", angle=8.0, distance=0.52, source="learned_policy"),
        _candidate("policy-b", angle=12.0, distance=0.62, source="policy_endpoint"),
    ]
    candidates[0]["checkpoint_provenance_digest"] = D[0]
    candidates[1]["checkpoint_provenance_digest"] = D[2]

    result = rank_franka_inspection_candidates(contract=contract, candidates=candidates)

    assert result["controller_candidate_ranking_proven"] is True
    assert result["learned_policy_ranking_proven"] is True
