from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import validate_task_freeze
from blueprint_pipeline.public_scene_calibrated_object_masks import (
    CalibratedObjectMaskError,
    materialize_calibrated_object_mask_set,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)
from blueprint_pipeline.scene_placement.semantic_source_track_import import (
    MASK_ENCODING,
    RESULT_SCHEMA_VERSION,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _task(task_id: str, slot: int) -> dict:
    task_kind = "articulated_interaction" if slot == 1 else "rigid_object_manipulation"
    task = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "prompt": "Open object" if slot == 1 else "Relocate object",
        "task_kind": task_kind,
        "scene_freeze_digest": "sha256:" + "1" * 64,
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "source_object": {
            "instance_id": f"source-{slot}",
            "semantic_label": "washer" if slot == 1 else "laptop",
            "observed_bounds_world_m": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [1.0, 1.0, 1.0],
            },
            "observed_pose_world": {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": f"support-{slot}",
            "collision_identity_receipt_digest": "sha256:" + "2" * 64,
            "support_receipt_digest": "sha256:" + "3" * 64,
            "franka_placement_packet_digest": "sha256:" + "4" * 64,
            "visibility_receipt_digest": "sha256:" + "5" * 64,
        },
        "removal_plan": {
            "removal_id": f"removal-{slot}",
            "mask_set_id": f"masks-{slot}",
            "source_collider_prim_path": f"/World/source_{slot}",
            "collider_deletion_id": f"delete-{slot}",
            "replacement_asset_id": f"replacement-{slot}",
            "replacement_qualification_id": f"qualification-{slot}",
        },
        "cameras": {
            "external": f"external-{slot}",
            "wrist": f"wrist-{slot}",
            "overview": f"overview-{slot}",
        },
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 100,
            "settle_window_steps": 10,
            "seeds": [1],
            "canonical_scenario_cell_id": f"cell-{slot}",
            "reset_state": {"robot": "home"},
        },
        "deterministic_success_predicates": ["complete"],
        "failure_rungs": ["never_moved"],
        "target_configuration": (
            {
                "kind": "joint_interval",
                "target_joint_ids": ["door_hinge"],
                "joint_intervals": {"door_hinge": [0.5, 1.0]},
            }
            if slot == 1
            else {
                "kind": "pose_volume",
                "position_bounds_world_m": {
                    "minimum": [0.0, 0.0, 0.0],
                    "maximum": [1.0, 1.0, 1.0],
                },
                "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                "maximum_orientation_error_rad": 0.2,
                "support_id": "desk",
                "release_required": True,
            }
        ),
        "articulation_graph": None,
        "task_freeze_digest": "",
    }
    if slot == 1:
        drive = {
            "drive_type": "force",
            "stiffness": 0.0,
            "damping": 1.0,
            "maximum_force": 1.0,
        }
        task["articulation_graph"] = {
            "schema_version": "adp_articulation_graph.v1",
            "links": [
                {"link_id": "body", "is_root": True, "semantic_role": "fixed_body"},
                {"link_id": "door", "is_root": False, "semantic_role": "target_door"},
            ],
            "joints": [
                {
                    "joint_id": "door_hinge",
                    "joint_type": "revolute",
                    "parent_link_id": "body",
                    "child_link_id": "door",
                    "role": "target",
                    "axis": [0.0, 0.0, 1.0],
                    "limits": [0.0, 1.0],
                    "reset_position": 0.0,
                    "reset_tolerance": 0.0001,
                    "drive": drive,
                    "dependency": None,
                }
            ],
            "collision_pairs": [
                {"link_a": "body", "link_b": "door", "collision_enabled": True}
            ],
            "success_predicate": {
                "combination": "all",
                "joint_intervals": {"door_hinge": [0.5, 1.0]},
            },
        }
    task["task_freeze_digest"] = canonical_digest(task, digest_field="task_freeze_digest")
    return validate_task_freeze(task)


def _fixture(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cameras = []
    images = tmp_path / "images"
    images.mkdir()
    frame_masks = []
    for index, camera_id in enumerate(("camera_0", "camera_1")):
        image = np.full((3, 4, 3), 10 + index, dtype=np.uint8)
        image_path = images / f"{camera_id}.png"
        Image.fromarray(image, mode="RGB").save(image_path)
        camera = {
            "camera_id": camera_id,
            "T_world_camera_provider_frame": [
                [1.0, 0.0, 0.0, float(index)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "intrinsics": {
                "model": "PINHOLE",
                "fx": 2.0,
                "fy": 2.0,
                "cx": 2.0,
                "cy": 1.5,
                "width": 4,
                "height": 3,
            },
        }
        cameras.append(camera)
        frame_masks.append(
            {
                "source_frame_id": camera_id,
                "source_frame_digest": _sha(image_path),
                "decoded_pts_seconds": float(index),
                "camera_record_digest": canonical_json_digest(camera),
                "width": 4,
                "height": 3,
                "mask_encoding": MASK_ENCODING,
                "track_masks": [
                    {
                        "track_id": "washer-track",
                        "runs": [{"start": index, "length": 2, "probability": 0.95}],
                    },
                    {
                        "track_id": "laptop-track",
                        "runs": [{"start": 8 + index, "length": 2, "probability": 0.9}],
                    },
                ],
            }
        )
    for row in frame_masks:
        row["mask_artifact_digest"] = canonical_json_digest(row["track_masks"])
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(json.dumps(cameras), encoding="utf-8")
    tracks = [
        {
            "track_id": "washer-track",
            "label": "washer",
            "supporting_frame_ids": ["camera_0", "camera_1"],
        },
        {
            "track_id": "laptop-track",
            "label": "laptop",
            "supporting_frame_ids": ["camera_0", "camera_1"],
        },
    ]
    source = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "bindings": {
            "track_registry_digest": canonical_json_digest(tracks),
            "frame_masks_digest": canonical_json_digest(frame_masks),
        },
        "track_registry": tracks,
        "frame_masks": frame_masks,
        "result_digest": "",
    }
    source["result_digest"] = canonical_json_digest(
        {key: value for key, value in source.items() if key != "result_digest"}
    )
    source_path = tmp_path / "source_tracks.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    task_paths = []
    for index, task_id in enumerate(("task_a", "task_b"), start=1):
        path = tmp_path / f"{task_id}.json"
        path.write_text(json.dumps(_task(task_id, index)), encoding="utf-8")
        task_paths.append(path)
    return {
        "source": source_path,
        "cameras": camera_path,
        "images": images,
        "tasks": task_paths,
        "task_inputs": {
            task_id: {
                "source_track_result_path": str(source_path),
                "camera_contract_path": str(camera_path),
                "source_image_root": str(images),
                "camera_frame_map": {
                    camera_id: camera_id for camera_id in ("camera_0", "camera_1")
                },
            }
            for task_id in ("task_a", "task_b")
        },
    }


def test_materializes_two_calibrated_object_masks_without_dilation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = materialize_calibrated_object_mask_set(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task={
            "task_a": ["washer-track"],
            "task_b": ["laptop-track"],
        },
        output_root=tmp_path / "output",
    )

    assert result["task_count"] == 2
    assert result["camera_count_total"] == 4
    assert result["claim_boundary"]["masks_are_model_inferred_candidates"] is True
    assert result["selection_authority"]["mask_dilation_pixels"] == 0
    mask = np.asarray(
        Image.open(tmp_path / "output/tasks/task_a/masks/camera_0.png"),
        dtype=np.uint8,
    )
    assert set(np.unique(mask)) == {0, 255}
    assert int(np.count_nonzero(mask)) == 2
    copied = tmp_path / "output/tasks/task_a/images/camera_0.png"
    assert copied.read_bytes() == (fixture["images"] / "camera_0.png").read_bytes()


def test_rejects_unbound_camera_and_missing_selected_track(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cameras = json.loads(fixture["cameras"].read_text())
    cameras[0]["intrinsics"]["fx"] = 9.0
    fixture["cameras"].write_text(json.dumps(cameras), encoding="utf-8")
    with pytest.raises(
        CalibratedObjectMaskError, match="calibrated_masks_camera_source_binding_invalid"
    ):
        materialize_calibrated_object_mask_set(
            task_freeze_paths=fixture["tasks"],
            task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task={
                "task_a": ["washer-track"],
                "task_b": ["laptop-track"],
            },
            output_root=tmp_path / "bad-output",
        )

    fixture = _fixture(tmp_path / "second")
    with pytest.raises(
        CalibratedObjectMaskError, match="calibrated_masks_selected_tracks_invalid"
    ):
        materialize_calibrated_object_mask_set(
            task_freeze_paths=fixture["tasks"],
            task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task={
                "task_a": ["missing-track"],
                "task_b": ["laptop-track"],
            },
            output_root=tmp_path / "second-output",
        )
