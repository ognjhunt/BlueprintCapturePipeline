from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.scene_object_discovery_queue import (
    SceneObjectDiscoveryQueueError,
    scene_object_discovery_status,
    seal_scene_object_discovery_result,
    select_scene_object_candidate,
    stage_scene_object_discovery_request,
)
from tests.test_scene_object_discovery_contract import request


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _discovery(*, status: str = "selection_required") -> dict[str, object]:
    candidates = []
    for index in range(2):
        candidates.append(
            {
                "candidate_id": f"sam31-tote-{index + 1:03d}",
                "label": "red tote",
                "backend": "sam31",
                "confidence": 0.9 - index * 0.05,
                "task_match_score": 0.9,
                "eligible_for_automatic_source_object": True,
                "candidate_claim_boundary": "metric_source_object_candidate",
                "metric_geometry_authority": "production_semantic_gaussian_obb",
                "metric_geometry": {
                    "evidence_digest": _digest(str(index + 1)),
                },
                "source_object_artifact": {
                    "uri": f"https://objects.example/source-{index}.json",
                    "digest": _digest(str(index + 3)),
                    "size_bytes": 100,
                },
            }
        )
    return {
        "schema_version": "scene_object_discovery.v1",
        "status": status,
        "discovery_digest": _digest("7"),
        "candidates": candidates,
        "selected_candidate_id": None,
        "source_object": None,
        "coverage": {"unseen_regions": ["behind_partition"]},
    }


def test_queue_is_idempotent_and_conflicting_identity_fails(tmp_path) -> None:
    value = request()
    first = stage_scene_object_discovery_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="webapp"
    )
    replay = stage_scene_object_discovery_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="webapp"
    )
    assert first["already_exists"] is False
    assert replay["already_exists"] is True
    assert (
        scene_object_discovery_status(
            discovery_id=value["discovery_id"], queue_root=tmp_path / "queue"
        )["status"]
        == "pending"
    )

    changed = copy.deepcopy(value)
    changed["task"]["task_statement"] = "Pick another tote"
    with pytest.raises(SceneObjectDiscoveryQueueError) as exc:
        stage_scene_object_discovery_request(
            value=changed, queue_root=tmp_path / "queue", submitted_by="webapp"
        )
    assert str(exc.value) == "scene_object_discovery_id_immutable_conflict"


def test_result_status_is_sanitized_and_selection_is_idempotent(tmp_path) -> None:
    value = request()
    receipt = stage_scene_object_discovery_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="webapp"
    )
    discovery = _discovery()
    discovery["candidates"][0]["preview"] = {"local_path": "/private/secret.png"}
    seal_scene_object_discovery_result(
        queue_root=tmp_path / "queue",
        discovery_id=value["discovery_id"],
        request_digest=receipt["request_digest"],
        source_commit=value["expected_production_commit"],
        discovery=discovery,
    )
    status = scene_object_discovery_status(
        discovery_id=value["discovery_id"], queue_root=tmp_path / "queue"
    )
    assert status["status"] == "selection_required"
    assert status["unseen_regions"] == ["behind_partition"]
    assert "metric_geometry" not in status["candidates"][0]
    assert "preview" not in status["candidates"][0]
    selection = {
        "schema_version": "scene_object_discovery_selection_request.v1",
        "discovery_id": value["discovery_id"],
        "expected_production_commit": value["expected_production_commit"],
        "request_digest": receipt["request_digest"],
        "discovery_digest": _digest("7"),
        "candidate_id": "sam31-tote-001",
        "confirm_selection": True,
    }
    selected = select_scene_object_candidate(value=selection, queue_root=tmp_path / "queue")
    replay = select_scene_object_candidate(value=selection, queue_root=tmp_path / "queue")
    assert selected == replay
    final = scene_object_discovery_status(
        discovery_id=value["discovery_id"], queue_root=tmp_path / "queue"
    )
    assert final["status"] == "ready_auto_selected"
    assert final["source_object"]["source_object_artifact"]["digest"] == _digest("3")


def test_selection_rejects_visual_only_candidate(tmp_path) -> None:
    value = request()
    receipt = stage_scene_object_discovery_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="webapp"
    )
    discovery = _discovery()
    discovery["candidates"][0]["eligible_for_automatic_source_object"] = False
    seal_scene_object_discovery_result(
        queue_root=tmp_path / "queue",
        discovery_id=value["discovery_id"],
        request_digest=receipt["request_digest"],
        source_commit=value["expected_production_commit"],
        discovery=discovery,
    )
    with pytest.raises(SceneObjectDiscoveryQueueError) as exc:
        select_scene_object_candidate(
            value={
                "schema_version": "scene_object_discovery_selection_request.v1",
                "discovery_id": value["discovery_id"],
                "expected_production_commit": value["expected_production_commit"],
                "request_digest": receipt["request_digest"],
                "discovery_digest": _digest("7"),
                "candidate_id": "sam31-tote-001",
                "confirm_selection": True,
            },
            queue_root=tmp_path / "queue",
        )
    assert str(exc.value) == "scene_object_discovery_candidate_not_selectable"
