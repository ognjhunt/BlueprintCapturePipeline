from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.scene_task_target_pipeline import compile_scene_task_targets


D = ["sha256:" + character * 64 for character in "abcdef"]
ROOT = Path(__file__).resolve().parents[1]


def _request(**updates):
    value = {
        "schema_version": "scene_task_target_analysis_request.v1",
        "scene_id": "private-apartment-001",
        "source_scene_digest": D[0],
        "source_video_available": False,
        "source_observation_profile": "authorized_external_reconstruction_views",
        "rendered_views": [
            {
                "view_id": "kitchen-wide",
                "rgb_digest": D[1],
                "observation_source": "reconstruction_render",
            },
            {
                "view_id": "sink-close",
                "rgb_digest": D[2],
                "observation_source": "reconstruction_render",
            },
        ],
        "object_affordance_proposals": [
            {
                "proposal_id": "sink-faucet-handle-001",
                "object_label": "faucet handle",
                "task_family": "faucet_handle_reach_and_turn",
                "affordances": ["inspect", "reach", "turn"],
                "visual_confidence": 0.92,
                "supporting_view_ids": ["kitchen-wide", "sink-close"],
                "visual_evidence_digests": [D[1], D[2]],
                "target_binding": {
                    "method": "rendered_depth_backprojection",
                    "position_scene": [1.2, -0.4, 0.9],
                    "spatial_uncertainty_scene_units": 0.08,
                    "binding_evidence_digest": D[3],
                },
            }
        ],
        "minimum_visual_confidence": 0.8,
        "threshold_frozen_before_analysis": True,
        "candidate_may_self_authorize": False,
        "analyzer_provenance": {
            "analyzer_id": "fixture-scene-vision-analyzer",
            "implementation_version": "1",
            "analyzer_contract_digest": D[5],
            "proposal_generation_is_dynamic": True,
            "candidate_may_self_authorize": False,
        },
        "robot_id": "franka_panda",
        "metric_scale_status": "provider_declared_not_independently_validated",
        "collision_support": {
            "status": "candidate_compiled",
            "collision_digest": D[4],
            "source_scene_digest": D[0],
        },
        "reach_support": {"status": "not_checked"},
    }
    value.update(updates)
    return value


def test_external_reconstruction_views_can_bind_bounded_target_without_video() -> None:
    result = compile_scene_task_targets(_request())

    assert result["status"] == "target_ready_for_bounded_sim"
    assert result["source_video_available"] is False
    assert result["source_video_required_for_bounded_sim_target"] is False
    assert result["derived_reconstruction_views_used"] is True
    target = result["selected_target"]
    assert target["object_label"] == "faucet handle"
    assert target["status"] == "authorized_derived_sim_target"
    assert target["claim_ceiling"] == "derived_scene_bounded_sim_target"
    assert target["metric_scale_verified"] is False
    assert "independent_metric_scale_missing" in target["qualification_gaps"]
    assert result["claim_boundary"]["metric_reach_or_clearance"] is False
    request_schema = json.loads(
        (ROOT / "docs/schemas/scene_task_target_analysis_request.v1.schema.json").read_text()
    )
    result_schema = json.loads(
        (ROOT / "docs/schemas/scene_task_target_analysis_result.v1.schema.json").read_text()
    )
    jsonschema.validate({**_request(), "request_digest": result["request_digest"]}, request_schema)
    jsonschema.validate(result, result_schema)


def test_missing_3d_binding_abstains_instead_of_fabricating_target() -> None:
    request = _request()
    request["object_affordance_proposals"][0]["target_binding"] = None
    result = compile_scene_task_targets(request)

    assert result["status"] == "abstained"
    assert result["selected_target"] is None
    assert result["blockers"] == ["no_qualified_3d_task_target"]
    assert "target_proposal_3d_binding_missing" in result["candidate_targets"][0]["blockers"]


def test_unregistered_cross_asset_collider_abstains() -> None:
    request = _request(
        collision_support={
            "status": "candidate_compiled",
            "collision_digest": D[4],
            "collision_source_asset_digest": D[5],
        }
    )

    result = compile_scene_task_targets(request)

    assert result["status"] == "abstained"
    candidate = result["candidate_targets"][0]
    assert candidate["collision_candidate_unbound_available"] is True
    assert candidate["collision_frame_bound"] is False
    assert "collision_candidate_frame_binding_missing" in candidate["qualification_gaps"]


def test_metric_scale_qualified_collision_and_reach_upgrade_only_sim_target() -> None:
    request = _request(
        source_video_available=True,
        source_observation_profile="raw_capture_and_reconstruction_views",
        metric_scale_status="validated",
        collision_support={
            "status": "qualified",
            "collision_digest": D[4],
            "source_scene_digest": D[0],
        },
        reach_support={"status": "reachable", "reach_evidence_digest": D[5]},
    )
    request["rendered_views"][0]["observation_source"] = "raw_capture"
    result = compile_scene_task_targets(request)

    target = result["selected_target"]
    assert target["status"] == "authorized_metric_sim_target"
    assert target["metric_reach_checked"] is True
    assert target["collision_qualified"] is True
    assert result["claim_boundary"]["metric_reach_or_clearance"] is True
    assert result["claim_boundary"]["simulated_task_success"] is False
    assert result["claim_boundary"]["physical_task_success"] is False
