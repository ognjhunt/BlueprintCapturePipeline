from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.fresh_scene_removal_freezes import (
    REQUEST_SCHEMA_VERSION,
    FreshSceneRemovalFreezeError,
    materialize_fresh_scene_removal_freezes,
)
from blueprint_pipeline.public_scene_segment_contribution_cutout import (
    SWEEP_KIND,
    materialize_segment_contribution_sweep_freeze,
)

from tests.test_adp009d_gaussian_excision_audit import POLICY, _camera, _splat
from tests.test_public_scene_calibrated_object_masks import _task


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _fixture(tmp_path: Path, *, task_count: int = 2) -> dict[str, object]:
    source = _splat(tmp_path / "scene.ply")
    collision = tmp_path / "collision.usda"
    collision.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Root"
{
    def Mesh "Target"
    {
        point3f[] points = [(-0.5, -0.5, 4.9), (0.5, -0.5, 4.9), (0.5, 0.5, 4.9), (-0.5, 0.5, 4.9), (-0.5, -0.5, 5.1), (0.5, -0.5, 5.1), (0.5, 0.5, 5.1), (-0.5, 0.5, 5.1)]
        int[] faceVertexCounts = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0]
    }
}
""",
        encoding="utf-8",
    )
    registered = {
        "schema_version": "interiorgs_sage_shared_frame_candidate.v1",
        "status": "multi_object_identity_alignment_candidate",
        "shared_frame_status": "provider_declared_not_independently_validated",
        "metric_scale_status": "provider_declared_not_independently_validated",
        "provider_transform": {
            "source_to_collision": "identity",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "handedness": "not_independently_proven",
        },
        "claim_boundary": {
            "independent_metric_metrology_completed": False,
            "handedness_independently_proven": False,
        },
        "receipt_digest": "",
    }
    registered["receipt_digest"] = canonical_digest(
        registered, digest_field="receipt_digest"
    )
    registered_path = tmp_path / "registered-frame.json"
    registered_path.write_text(canonical_json(registered) + "\n", encoding="utf-8")
    receipt_root = tmp_path / "reviewed-masks"
    receipt_root.mkdir()
    cameras = [
        _camera("front", 0.0, 0.0),
        _camera("near_left", -0.2, 1.0),
        _camera("near_right", 0.2, -1.0),
        _camera("far_left", -0.5, 3.0),
        _camera("far_right", 0.5, -3.0),
    ]
    task_rows = []
    for slot in range(task_count):
        task_id = f"task_{slot + 1}"
        task_root = receipt_root / "tasks" / task_id
        images = task_root / "images"
        masks = task_root / "masks"
        images.mkdir(parents=True)
        masks.mkdir()
        camera_path = task_root / "cameras.v1.json"
        camera_path.write_text(json.dumps(cameras), encoding="utf-8")
        task_freeze_path = tmp_path / f"{task_id}.json"
        task_freeze = _task(task_id, 1 if slot == 0 else 2)
        task_freeze_path.write_text(canonical_json(task_freeze) + "\n", encoding="utf-8")
        source_tracks_path = tmp_path / f"{task_id}-tracks.json"
        source_tracks_path.write_text("{}\n", encoding="utf-8")
        image_rows = []
        mask_rows = []
        outer = np.zeros((48, 64), dtype=np.uint8)
        outer[8:40, 12:52] = 255
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            image_path = images / f"{camera_id}.png"
            mask_path = masks / f"{camera_id}.png"
            assert cv2.imwrite(str(image_path), np.zeros((48, 64, 3), dtype=np.uint8))
            assert cv2.imwrite(str(mask_path), outer)
            image_rows.append({"camera_id": camera_id, "image": _relative(image_path, receipt_root)})
            mask_rows.append({"camera_id": camera_id, "mask": _relative(mask_path, receipt_root)})
        task_rows.append(
            {
                "task_id": task_id,
                "task_freeze": {
                    "path": str(task_freeze_path),
                    "size_bytes": task_freeze_path.stat().st_size,
                    "sha256": _sha(task_freeze_path),
                    "task_freeze_digest": task_freeze["task_freeze_digest"],
                },
                "source_track_result": {
                    "path": str(source_tracks_path),
                    "size_bytes": source_tracks_path.stat().st_size,
                    "sha256": _sha(source_tracks_path),
                },
                "camera_contract": _relative(camera_path, receipt_root),
                "source_images_root": str(images),
                "source_images": image_rows,
                "mask_root": str(masks),
                "masks": mask_rows,
            }
        )
    mask_receipt = {
        "schema_version": "public_scene_calibrated_object_mask_set.v1",
        "status": "calibrated_inferred_object_masks_materialized_pending_review",
        "task_count": task_count,
        "tasks": task_rows,
        "selection_authority": {
            "all_selected_tracks_human_review_accepted": True,
        },
        "receipt_digest": "",
    }
    mask_receipt["receipt_digest"] = canonical_digest(
        mask_receipt, digest_field="receipt_digest"
    )
    mask_receipt_path = receipt_root / "public_scene_calibrated_object_mask_set.v1.json"
    mask_receipt_path.write_text(canonical_json(mask_receipt) + "\n", encoding="utf-8")
    task_requests = {
        row["task_id"]: {
            "target_collision_prim_path": "/Root/Target",
            "scene": {
                "publisher_scene_id": "fixture",
                "task_id": row["task_id"],
                "target_instance_id": row["task_id"],
                "target_semantic_label": "fixture-object",
            },
            "policy": POLICY,
            "historical_baseline": {
                "method": "center_inside_registered_target_aabb",
                "center_aabb_min_m": [-0.6, -0.6, 4.9],
                "center_aabb_max_m": [0.6, 0.6, 5.1],
                "selected_gaussian_count": 4,
            },
        }
        for row in task_rows
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_standard_splat_path": str(source),
        "source_collision_path": str(collision),
        "registered_frame_receipt_path": str(registered_path),
        "calibrated_mask_set_receipt_path": str(mask_receipt_path),
        "tasks": task_requests,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return {"request": request, "mask_receipt": mask_receipt_path}


def test_materializes_two_task_excision_and_segment_sweep_freezes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = materialize_fresh_scene_removal_freezes(
        request=fixture["request"], output_root=tmp_path / "output"
    )

    assert result["task_count"] == 2
    assert result["paid_execution_started"] is False
    assert result["provider_mutations_performed"] == 0
    assert result["agent_selected_gaussian_indices"] is False
    assert result["canonical_source_altered"] is False
    assert all(row["camera_count"] == 5 for row in result["tasks"])


def test_rejects_mask_bytes_changed_after_review(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mask = tmp_path / "reviewed-masks/tasks/task_1/masks/front.png"
    mask.write_bytes(b"changed")

    with pytest.raises(FreshSceneRemovalFreezeError, match="mask_binding_invalid"):
        materialize_fresh_scene_removal_freezes(
            request=fixture["request"], output_root=tmp_path / "output"
        )


def test_segment_sweep_uses_all_frozen_cameras(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = materialize_fresh_scene_removal_freezes(
        request=fixture["request"], output_root=tmp_path / "output"
    )
    sweep_record = result["tasks"][0]["segment_sweep_freeze"]
    sweep_path = tmp_path / "output" / sweep_record["relative_path"]
    sweep = json.loads(sweep_path.read_text())

    assert sweep["camera_split"]["camera_count"] == 5
    assert sweep["camera_split"]["calibration_camera_count"] == 5
    assert sweep["camera_split"]["heldout_camera_count"] == 0
    assert sweep["camera_split"]["method"] == "all_frozen_segment_views.v1"
    assert sweep["segment_contribution_sweep"]["kind"] == SWEEP_KIND
    assert sweep["learned_policy_outcomes_observed"] is False

    replay_root = tmp_path / "replayed-sweep"
    replay = materialize_segment_contribution_sweep_freeze(
        excision_freeze_path=(
            tmp_path
            / "output"
            / result["tasks"][0]["excision_freeze"]["relative_path"]
        ),
        output_root=replay_root,
    )
    assert replay == sweep
