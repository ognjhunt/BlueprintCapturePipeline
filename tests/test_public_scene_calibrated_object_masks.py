from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import validate_task_freeze
from blueprint_pipeline.public_scene_calibrated_object_masks import (
    CalibratedObjectMaskError,
    materialize_calibrated_object_mask_set,
)
from blueprint_pipeline.public_scene_sam31_track_selection_review import (
    AI_RECEIPT_SCHEMA_VERSION,
    AI_REVIEW_METHOD,
    Sam31TrackSelectionReviewError,
    materialize_sam31_track_selection_review_candidate,
    seal_sam31_track_selection_ai_review,
    seal_sam31_track_selection_review,
    validate_sam31_track_selection_review,
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
            "collision_pairs": [{"link_a": "body", "link_b": "door", "collision_enabled": True}],
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


def _review(
    tmp_path: Path,
    fixture: dict[str, object],
    selected: dict[str, list[str]],
) -> Path:
    candidate_root = tmp_path / "review-candidate"
    materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        output_root=candidate_root,
    )
    receipt = tmp_path / "review-receipt.json"
    seal_sam31_track_selection_review(
        candidate_path=(
            candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
        ),
        reviewed_by="fixture-reviewer",
        reviewed_on="2026-08-13",
        output_path=receipt,
    )
    return receipt


def test_materializes_two_calibrated_object_masks_without_dilation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selected = {
        "task_a": ["washer-track"],
        "task_b": ["laptop-track"],
    }
    review = _review(tmp_path, fixture, selected)
    legacy_receipt = json.loads(review.read_text(encoding="utf-8"))
    assert legacy_receipt["schema_version"] == "public_scene_sam31_track_selection_review.v1"
    assert legacy_receipt["status"] == "selected_tracks_human_review_accepted"
    assert "reviewer" not in legacy_receipt
    assert "decision" not in legacy_receipt
    result = materialize_calibrated_object_mask_set(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        reviewed_track_selection_receipt_path=review,
        output_root=tmp_path / "output",
    )

    assert result["task_count"] == 2
    assert result["camera_count_total"] == 4
    assert result["claim_boundary"]["masks_are_model_inferred_candidates"] is True
    assert result["selection_authority"]["mask_dilation_pixels"] == 0
    assert result["selection_authority"]["all_selected_tracks_human_review_accepted"] is True
    mask = np.asarray(
        Image.open(tmp_path / "output/tasks/task_a/masks/camera_0.png"),
        dtype=np.uint8,
    )
    assert set(np.unique(mask)) == {0, 255}
    assert int(np.count_nonzero(mask)) == 2
    copied = tmp_path / "output/tasks/task_a/images/camera_0.png"
    assert copied.read_bytes() == (fixture["images"] / "camera_0.png").read_bytes()


def test_named_ai_visual_review_accepts_exact_media_for_calibrated_masks(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selected = {"task_a": ["washer-track"], "task_b": ["laptop-track"]}
    candidate_root = tmp_path / "ai-review-candidate"
    candidate = materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        output_root=candidate_root,
    )
    receipt_path = tmp_path / "ai-review.json"
    receipt = seal_sam31_track_selection_ai_review(
        candidate_path=(
            candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
        ),
        reviewer_id="codex-visual-reviewer",
        model="gpt-5",
        model_version="2026-08-15",
        review_method=AI_REVIEW_METHOD,
        reviewed_at="2026-08-15T13:30:00Z",
        decision="accepted",
        output_path=receipt_path,
    )

    assert receipt["schema_version"] == AI_RECEIPT_SCHEMA_VERSION
    assert receipt["status"] == "selected_tracks_ai_visual_review_accepted"
    assert receipt["reviewer"] == {
        "kind": "ai",
        "identity": "codex-visual-reviewer",
        "model": "gpt-5",
        "model_version": "2026-08-15",
        "method": AI_REVIEW_METHOD,
    }
    assert receipt["review_scope"]["candidate_digest"] == candidate["candidate_digest"]
    assert receipt["review_scope"]["review_media_digest"] == canonical_json_digest(
        candidate["review_media"]
    )
    assert receipt["review_scope"]["review_frame_count"] == 4
    assert receipt["claim_boundary"]["human_review_completed"] is False
    assert receipt["claim_boundary"]["ai_visual_review_completed"] is True
    assert "reviewed_by" not in receipt
    assert "reviewed_on" not in receipt
    assert "agent_selected_tracks_without_human_review" not in receipt

    validated = validate_sam31_track_selection_review(
        receipt_path=receipt_path,
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
    )
    assert validated["receipt_digest"] == receipt["receipt_digest"]
    calibrated = materialize_calibrated_object_mask_set(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        reviewed_track_selection_receipt_path=receipt_path,
        output_root=tmp_path / "ai-reviewed-masks",
    )
    authority = calibrated["selection_authority"]
    assert authority["reviewer_kind"] == "ai"
    assert authority["all_selected_tracks_review_accepted"] is True
    assert authority["all_selected_tracks_ai_visual_review_accepted"] is True
    assert authority["all_selected_tracks_human_review_accepted"] is False


def test_ai_visual_review_rejection_and_missing_identity_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selected = {"task_a": ["washer-track"], "task_b": ["laptop-track"]}
    candidate_root = tmp_path / "ai-review-candidate"
    materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        output_root=candidate_root,
    )
    candidate_path = candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
    rejected_path = tmp_path / "rejected.json"
    rejected = seal_sam31_track_selection_ai_review(
        candidate_path=candidate_path,
        reviewer_id="codex-visual-reviewer",
        model="gpt-5",
        model_version="2026-08-15",
        review_method=AI_REVIEW_METHOD,
        reviewed_at="2026-08-15T13:30:00Z",
        decision="rejected",
        output_path=rejected_path,
    )
    assert rejected["status"] == "selected_tracks_ai_visual_review_rejected"
    assert rejected["all_selected_tracks_accepted"] is False
    with pytest.raises(Sam31TrackSelectionReviewError, match="receipt_invalid"):
        validate_sam31_track_selection_review(
            receipt_path=rejected_path,
            task_freeze_paths=fixture["tasks"],
            task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task=selected,
        )
    with pytest.raises(CalibratedObjectMaskError, match="calibrated_masks_review_receipt_invalid"):
        materialize_calibrated_object_mask_set(
            task_freeze_paths=fixture["tasks"],
            task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task=selected,
            reviewed_track_selection_receipt_path=rejected_path,
            output_root=tmp_path / "rejected-masks-must-not-exist",
        )

    with pytest.raises(Sam31TrackSelectionReviewError, match="candidate_invalid"):
        seal_sam31_track_selection_ai_review(
            candidate_path=candidate_path,
            reviewer_id="codex-visual-reviewer",
            model="gpt-5",
            model_version="2026-08-15",
            review_method="arbitrary-caller-prose",
            reviewed_at="2026-08-15T13:30:00Z",
            decision="accepted",
            output_path=tmp_path / "must-not-exist.json",
        )
    with pytest.raises(Sam31TrackSelectionReviewError, match="candidate_invalid"):
        seal_sam31_track_selection_ai_review(
            candidate_path=candidate_path,
            reviewer_id="",
            model="gpt-5",
            model_version="2026-08-15",
            review_method=AI_REVIEW_METHOD,
            reviewed_at="2026-08-15T13:30:00Z",
            decision="accepted",
            output_path=tmp_path / "missing-identity.json",
        )
    with pytest.raises(Sam31TrackSelectionReviewError, match="candidate_invalid"):
        seal_sam31_track_selection_ai_review(
            candidate_path=candidate_path,
            reviewer_id="codex-visual-reviewer",
            model="gpt-5",
            model_version="2026-08-15",
            review_method=AI_REVIEW_METHOD,
            reviewed_at="2026-08-15T13:30:00Z",
            decision="",
            output_path=tmp_path / "missing-decision.json",
        )


def test_rejects_unbound_camera_and_missing_selected_track(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selected = {
        "task_a": ["washer-track"],
        "task_b": ["laptop-track"],
    }
    review = _review(tmp_path, fixture, selected)
    cameras = json.loads(fixture["cameras"].read_text())
    cameras[0]["intrinsics"]["fx"] = 9.0
    fixture["cameras"].write_text(json.dumps(cameras), encoding="utf-8")
    with pytest.raises(CalibratedObjectMaskError, match="calibrated_masks_review_receipt_invalid"):
        materialize_calibrated_object_mask_set(
            task_freeze_paths=fixture["tasks"],
            task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task={
                "task_a": ["washer-track"],
                "task_b": ["laptop-track"],
            },
            reviewed_track_selection_receipt_path=review,
            output_root=tmp_path / "bad-output",
        )

    fixture = _fixture(tmp_path / "second")
    second_review = _review(
        tmp_path / "second-review",
        fixture,
        {"task_a": ["washer-track"], "task_b": ["laptop-track"]},
    )
    with pytest.raises(CalibratedObjectMaskError, match="calibrated_masks_review_receipt_invalid"):
        materialize_calibrated_object_mask_set(
            task_freeze_paths=fixture["tasks"],
            task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task={
                "task_a": ["missing-track"],
                "task_b": ["laptop-track"],
            },
            reviewed_track_selection_receipt_path=second_review,
            output_root=tmp_path / "second-output",
        )


def test_review_candidate_supports_five_tasks_and_receipt_rejects_tamper(
    tmp_path: Path,
) -> None:
    base = _fixture(tmp_path / "base")
    tasks: list[Path] = []
    task_inputs: dict[str, dict] = {}
    selected: dict[str, list[str]] = {}
    for index in range(1, 6):
        task_id = f"task_{index}"
        task = _task(task_id, 1 if index == 1 else 2)
        task["source_object"]["instance_id"] = f"source-{index}"
        task["removal_plan"]["removal_id"] = f"removal-{index}"
        task["removal_plan"]["mask_set_id"] = f"masks-{index}"
        task["task_freeze_digest"] = canonical_digest(task, digest_field="task_freeze_digest")
        path = tmp_path / f"{task_id}.json"
        path.write_text(json.dumps(task), encoding="utf-8")
        tasks.append(path)
        task_inputs[task_id] = dict(base["task_inputs"]["task_b"])
        selected[task_id] = ["laptop-track"]
    candidate_root = tmp_path / "five-review"
    candidate = materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=tasks,
        task_inputs=task_inputs,
        selected_track_ids_by_task=selected,
        output_root=candidate_root,
    )
    assert candidate["task_count"] == 5
    assert len(candidate["review_media"]) == 5
    receipt_path = tmp_path / "five-review.json"
    seal_sam31_track_selection_review(
        candidate_path=(
            candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
        ),
        reviewed_by="fixture-reviewer",
        reviewed_on="2026-08-13",
        output_path=receipt_path,
    )
    tampered = json.loads(receipt_path.read_text())
    tampered["selection_bindings"][0]["selected_track_ids"] = ["washer-track"]
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Sam31TrackSelectionReviewError, match="receipt_invalid"):
        validate_sam31_track_selection_review(
            receipt_path=receipt_path,
            task_freeze_paths=tasks,
            task_inputs=task_inputs,
            selected_track_ids_by_task=selected,
        )


def test_review_acceptance_rehashes_exact_overlay_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selected = {"task_a": ["washer-track"], "task_b": ["laptop-track"]}
    candidate_root = tmp_path / "review-candidate"
    materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        output_root=candidate_root,
    )
    overlay = candidate_root / "review_media/task_a/camera_0.png"
    with Image.open(overlay) as image:
        changed = np.asarray(image.convert("RGB")).copy()
    changed[0, 0] = [1, 2, 3]
    Image.fromarray(changed, mode="RGB").save(overlay)
    with pytest.raises(Sam31TrackSelectionReviewError, match="media_record_invalid"):
        seal_sam31_track_selection_review(
            candidate_path=(
                candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
            ),
            reviewed_by="fixture-reviewer",
            reviewed_on="2026-08-13",
            output_path=tmp_path / "must-not-exist.json",
        )


def test_review_candidate_preserves_eight_views_when_one_selected_mask_is_empty(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    cameras = json.loads(fixture["cameras"].read_text(encoding="utf-8"))
    source = json.loads(fixture["source"].read_text(encoding="utf-8"))
    frame_masks = source["frame_masks"]
    images = fixture["images"]
    for index in range(2, 8):
        camera_id = f"camera_{index}"
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
        track_masks = [
            {
                "track_id": "washer-track",
                "runs": [{"start": index, "length": 1, "probability": 0.95}],
            },
            {
                "track_id": "laptop-track",
                "runs": [{"start": 8, "length": 2, "probability": 0.9}],
            },
        ]
        frame_masks.append(
            {
                "source_frame_id": camera_id,
                "source_frame_digest": _sha(image_path),
                "decoded_pts_seconds": float(index),
                "camera_record_digest": canonical_json_digest(camera),
                "width": 4,
                "height": 3,
                "mask_encoding": MASK_ENCODING,
                "track_masks": track_masks,
                "mask_artifact_digest": canonical_json_digest(track_masks),
            }
        )
    frame_masks[0]["track_masks"] = [
        row for row in frame_masks[0]["track_masks"] if row["track_id"] == "washer-track"
    ]
    frame_masks[0]["mask_artifact_digest"] = canonical_json_digest(frame_masks[0]["track_masks"])
    source["bindings"]["frame_masks_digest"] = canonical_json_digest(frame_masks)
    source["result_digest"] = canonical_json_digest(
        {key: value for key, value in source.items() if key != "result_digest"}
    )
    fixture["cameras"].write_text(json.dumps(cameras), encoding="utf-8")
    fixture["source"].write_text(json.dumps(source), encoding="utf-8")
    camera_map = {f"camera_{index}": f"camera_{index}" for index in range(8)}
    for task_input in fixture["task_inputs"].values():
        task_input["camera_frame_map"] = camera_map

    candidate = materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task={
            "task_a": ["washer-track"],
            "task_b": ["laptop-track"],
        },
        output_root=tmp_path / "eight-view-review",
    )

    task_b = next(row for row in candidate["review_media"] if row["task_id"] == "task_b")
    assert task_b["camera_count"] == 8
    assert len(task_b["frames"]) == 8
    empty = next(row for row in task_b["frames"] if row["camera_id"] == "camera_0")
    assert empty["foreground_pixel_count"] == 0

    candidate_path = (
        tmp_path
        / "eight-view-review"
        / "public_scene_sam31_track_selection_review_candidate.v1.json"
    )
    receipt_path = tmp_path / "eight-view-review-accepted.json"
    seal_sam31_track_selection_review(
        candidate_path=candidate_path,
        reviewed_by="fixture-reviewer",
        reviewed_on="2026-08-15",
        output_path=receipt_path,
    )
    calibrated = materialize_calibrated_object_mask_set(
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task={
            "task_a": ["washer-track"],
            "task_b": ["laptop-track"],
        },
        reviewed_track_selection_receipt_path=receipt_path,
        output_root=tmp_path / "eight-view-calibrated-masks",
    )
    assert calibrated["camera_count_total"] == 16
    task_b_masks = tmp_path / "eight-view-calibrated-masks/tasks/task_b/masks"
    assert len(list(task_b_masks.glob("*.png"))) == 8
    assert np.count_nonzero(np.asarray(Image.open(task_b_masks / "camera_0.png"))) == 0


def test_review_candidate_and_accept_clis(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    task_inputs_path = tmp_path / "task-inputs.json"
    selected_path = tmp_path / "selected-tracks.json"
    task_inputs_path.write_text(json.dumps(fixture["task_inputs"]), encoding="utf-8")
    selected_path.write_text(
        json.dumps({"task_a": ["washer-track"], "task_b": ["laptop-track"]}),
        encoding="utf-8",
    )
    candidate_root = tmp_path / "cli-candidate"
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.public_scene_sam31_track_selection_review",
        "candidate",
    ]
    for path in fixture["tasks"]:
        command.extend(["--task-freeze", str(path)])
    command.extend(
        [
            "--task-inputs",
            str(task_inputs_path),
            "--selected-tracks",
            str(selected_path),
            "--output-root",
            str(candidate_root),
        ]
    )
    subprocess.run(command, check=True)
    candidate_path = candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json"
    receipt_path = tmp_path / "cli-review.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.public_scene_sam31_track_selection_review",
            "accept",
            "--candidate",
            str(candidate_path),
            "--reviewed-by",
            "cli-reviewer",
            "--reviewed-on",
            "2026-08-13",
            "--output",
            str(receipt_path),
        ],
        check=True,
    )
    receipt = validate_sam31_track_selection_review(
        receipt_path=receipt_path,
        task_freeze_paths=fixture["tasks"],
        task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task={
            "task_a": ["washer-track"],
            "task_b": ["laptop-track"],
        },
    )
    assert receipt["reviewed_by"] == "cli-reviewer"
