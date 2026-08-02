from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.rendered_scene_task_target_orchestrator import (
    RenderedSceneTaskTargetOrchestratorError,
    compile_rendered_scene_task_target,
)


SCENE_DIGEST = "sha256:" + "a" * 64
COLLISION_DIGEST = "sha256:" + "b" * 64
ANALYZER_DIGEST = "sha256:" + "c" * 64
CAMERA_DIGEST = "sha256:" + "d" * 64
ROOT = Path(__file__).resolve().parents[1]


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _splat(tmp_path):
    grid = np.linspace(-0.12, 0.12, 12)
    front = np.asarray([(x, y, 2.0) for x in grid for y in grid], dtype=np.float32)
    back = np.asarray([(x, y, 5.0) for x in grid for y in grid], dtype=np.float32)
    xyz = np.concatenate([front, back], axis=0)
    count = len(xyz)
    return write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=xyz,
            opacity=np.full(count, 8.0, dtype=np.float32),
            f_dc=np.zeros((count, 3), dtype=np.float32),
            scales=np.zeros((count, 3), dtype=np.float32),
            quats=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (count, 1),
            ),
            properties=(),
        ),
        tmp_path / "analysis.ply",
    )


def _view(tmp_path):
    image = tmp_path / "task-focus.png"
    image.write_bytes(b"digest-bound-derived-render")
    return {
        "view_id": "task_focus",
        "rgb_path": str(image),
        "rgb_digest": _digest(image),
        "observation_source": "reconstruction_render",
        "camera_spec_digest": CAMERA_DIGEST,
        "image_size": {"width": 640, "height": 480},
        "camera": {
            "pos": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 60.0,
        },
    }


def _proposal_set(**updates):
    value = {
        "schema_version": "rendered_scene_task_proposal_set.v1",
        "analyzer_provenance": {
            "analyzer_id": "replaceable-vision-agent",
            "implementation_version": "1",
            "analyzer_contract_digest": ANALYZER_DIGEST,
            "proposal_generation_is_dynamic": True,
            "candidate_may_self_authorize": False,
        },
        "candidate_may_self_authorize": False,
        "proposals": [
            {
                "proposal_id": "visible-work-surface-001",
                "object_label": "visible work surface",
                "task_family": "franka_surface_inspection",
                "affordances": ["inspect", "reach_to_surface"],
                "visual_confidence": 0.92,
                "supporting_view_ids": ["task_focus"],
                "binding_view_id": "task_focus",
                "bbox_xyxy_pixels": [260, 180, 380, 300],
            }
        ],
    }
    value.update(updates)
    return value


def _compile(tmp_path, **updates):
    splat = _splat(tmp_path)
    values = {
        "analysis_splat_path": splat,
        "scene_id": "private-apartment-001",
        "source_scene_digest": SCENE_DIGEST,
        "rendered_views": [_view(tmp_path)],
        "proposal_set": _proposal_set(),
        "source_video_available": False,
        "robot_id": "franka_panda",
        "metric_scale_status": "provider_declared_not_independently_validated",
        "collision_support": {
            "status": "candidate_compiled",
            "collision_digest": COLLISION_DIGEST,
            "source_scene_digest": SCENE_DIGEST,
        },
        "reach_support": {"status": "not_checked"},
    }
    values.update(updates)
    return compile_rendered_scene_task_target(**values)


def test_dynamic_analyzer_proposal_is_bound_and_deterministically_authorized(tmp_path) -> None:
    proposal_set = _proposal_set()
    result = _compile(tmp_path, proposal_set=proposal_set)

    assert result["status"] == "target_ready_for_bounded_sim"
    assert result["source_video_available"] is False
    assert result["source_video_required_for_bounded_sim_target"] is False
    assert result["candidate_may_self_authorize"] is False
    assert result["binding_results"][0]["status"] == "candidate_bound"
    target = result["target_analysis"]["selected_target"]
    assert target["object_label"] == "visible work surface"
    assert target["target_binding_method"] == "rendered_depth_backprojection"
    np.testing.assert_allclose(target["target_position_scene"], [0.0, 0.0, 2.0], atol=0.03)
    assert target["status"] == "authorized_derived_sim_target"
    assert "independent_metric_scale_missing" in target["qualification_gaps"]
    assert result["target_analysis"]["claim_boundary"]["simulated_task_success"] is False
    proposal_set["proposal_set_digest"] = canonical_digest(
        proposal_set, digest_field="proposal_set_digest"
    )
    jsonschema.validate(
        proposal_set,
        json.loads(
            (ROOT / "docs/schemas/rendered_scene_task_proposal_set.v1.schema.json").read_text()
        ),
    )
    jsonschema.validate(
        result,
        json.loads(
            (
                ROOT / "docs/schemas/rendered_scene_task_target_orchestration.v1.schema.json"
            ).read_text()
        ),
    )


def test_weak_visual_to_3d_binding_abstains_without_fabricating_target(tmp_path) -> None:
    proposal_set = _proposal_set()
    proposal_set["proposals"][0]["bbox_xyxy_pixels"] = [0, 0, 80, 80]

    result = _compile(tmp_path, proposal_set=proposal_set)

    assert result["status"] == "abstained"
    assert result["binding_results"][0]["status"] == "abstained"
    assert "bbox_binding_projected_support_insufficient" in result["binding_results"][0]["blockers"]
    assert result["target_analysis"]["selected_target"] is None
    assert result["target_analysis"]["blockers"] == ["no_qualified_3d_task_target"]


def test_rendered_view_digest_mismatch_fails_before_analysis(tmp_path) -> None:
    view = _view(tmp_path)
    view["rgb_digest"] = "sha256:" + "e" * 64

    with pytest.raises(RenderedSceneTaskTargetOrchestratorError) as exc_info:
        _compile(tmp_path, rendered_views=[view])

    assert "rendered_target_view_0_rgb_binding_invalid" in exc_info.value.codes


def test_analyzer_may_never_self_authorize(tmp_path) -> None:
    proposal_set = _proposal_set()
    proposal_set["analyzer_provenance"]["candidate_may_self_authorize"] = True

    with pytest.raises(RenderedSceneTaskTargetOrchestratorError) as exc_info:
        _compile(tmp_path, proposal_set=proposal_set)

    assert "rendered_target_analyzer_self_authorization_forbidden" in exc_info.value.codes


def test_proposal_set_replay_digest_mismatch_fails_closed(tmp_path) -> None:
    proposal_set = _proposal_set()
    proposal_set["proposal_set_digest"] = canonical_digest(
        proposal_set, digest_field="proposal_set_digest"
    )
    proposal_set["proposals"][0]["visual_confidence"] = 0.91

    with pytest.raises(RenderedSceneTaskTargetOrchestratorError) as exc_info:
        _compile(tmp_path, proposal_set=proposal_set)

    assert "rendered_target_proposal_set_digest_mismatch" in exc_info.value.codes
