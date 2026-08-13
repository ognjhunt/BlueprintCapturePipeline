from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import binary_dilation

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    canonical_json,
)
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs import (
    SCHEMA_VERSION,
    SEMANTIC_TEACHER_SCHEMA,
    TRANSITION_MORPHOLOGY,
    DualTargetInputError,
    materialize_dual_target_artifixer3d_inputs,
    materialize_whole_frame_semantic_teacher_receipt,
)
from tests.test_public_scene_artifixer3d_candidate_inputs import _preflight


def _semantic_receipts(
    tmp_path: Path,
    *,
    source_root: Path,
    source: dict[str, object],
) -> list[Path]:
    paths: list[Path] = []
    source_receipt_path = (
        source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    )
    for task in source["tasks"]:  # type: ignore[index]
        task_id = task["task_id"]
        task_root = Path(task["scene_directory"])
        semantic_root = tmp_path / f"semantic_{task_id}"
        semantic_root.mkdir()
        for frame in task["frames"]:
            original = np.asarray(
                Image.open(
                    task_root / frame["rendered_rgb"]["relative_path"]
                ).convert("RGB"),
                dtype=np.uint8,
            )
            teacher = np.bitwise_xor(original, np.uint8(31))
            Image.fromarray(teacher, mode="RGB").save(
                semantic_root / f"{frame['frame_index']:05d}.png"
            )
        receipt_path = tmp_path / f"semantic_teacher.{task_id}.json"
        materialize_whole_frame_semantic_teacher_receipt(
            source_candidate_inputs_receipt_path=source_receipt_path,
            task_id=task_id,
            semantic_teacher_frames_root=semantic_root,
            editor_identity={
                "backend": "fixture_semantic_editor",
                "snapshot_pinned": False,
                "formal_api_receipt_available": False,
            },
            prompt_policy="generic_object_absent_room_completion_v1",
            output_path=receipt_path,
        )
        paths.append(receipt_path)
    return paths


def _dual_candidate(
    tmp_path: Path,
    *,
    count: int = 1,
    cameras_per_task: int = 2,
    radius: int = 1,
) -> tuple[Path, dict[str, object], list[Path], dict[str, object]]:
    preflight = _preflight(
        tmp_path / "source_fixture",
        count=count,
        cameras_per_task=cameras_per_task,
    )
    source_root = tmp_path / "source_candidate"
    source = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=source_root,
    )
    semantic_receipts = _semantic_receipts(
        tmp_path,
        source_root=source_root,
        source=source,
    )
    dual_root = tmp_path / "dual"
    dual = materialize_dual_target_artifixer3d_inputs(
        source_candidate_inputs_receipt_path=(
            source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
        ),
        semantic_teacher_receipt_paths=semantic_receipts,
        output_root=dual_root,
        transition_radius_pixels=radius,
    )
    return dual_root, source, semantic_receipts, dual


def test_binds_unreviewed_whole_frame_semantic_teacher_without_locality_claim(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path / "fixture", count=1, cameras_per_task=2)
    source_root = tmp_path / "source"
    source = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=source_root,
    )
    paths = _semantic_receipts(
        tmp_path,
        source_root=source_root,
        source=source,
    )

    receipt = json.loads(paths[0].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == SEMANTIC_TEACHER_SCHEMA
    assert receipt["status"] == "whole_frame_semantic_teacher_candidates_unreviewed"
    assert receipt["frame_count"] == 2
    assert receipt["outside_exact_support_changed_pixels_total"] > 0
    assert receipt["semantic_object_absence_review_passed"] is False
    assert receipt["multiview_consistency_review_passed"] is False
    assert receipt["appearance_repair_qualified"] is False
    assert all(
        frame["whole_frame_candidate_preserved_without_compositing"]
        for frame in receipt["frames"]
    )
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_materializes_same_pose_pairs_masks_review_path_and_unchanged_seed(
    tmp_path: Path,
) -> None:
    dual_root, source, semantic_paths, dual = _dual_candidate(tmp_path, radius=2)

    assert dual["schema_version"] == SCHEMA_VERSION
    assert dual["status"] == "paired_target_inputs_prepared_no_model_no_execution"
    assert dual["pipeline_mode"] == "dual_target_artifixer3d_only"
    assert dual["replacement_object_count"] == 1
    assert dual["transition_support"] == {
        "radius_pixels": 2,
        "morphology": TRANSITION_MORPHOLOGY,
        "anchor_loss_mask_outside_support": 255,
        "anchor_loss_mask_inside_support": 0,
        "semantic_teacher_loss_mask": None,
    }
    assert dual["shared_seed"]["copied_byte_identical"] is True
    assert dual["shared_seed"]["retained_scene"]["sha256"] == source[
        "shared_retained_scene"
    ]["sha256"]
    assert dual["shared_seed"]["colmap_points3D"]["sha256"] == source[
        "shared_colmap_initialization_points3D"
    ]["sha256"]

    task = dual["tasks"][0]
    task_root = Path(task["scene_directory"])
    assert task["camera_count"] == 2
    assert task["physical_camera_count"] == 2
    assert task["training_record_count"] == 4
    assert task["selected_anchor_indices"] == [0, 2]
    assert task["semantic_teacher_indices"] == [1, 3]
    transforms = json.loads(
        (task_root / task["transforms"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    review = json.loads(
        (task_root / task["review_trajectory"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(transforms["frames"]) == 4
    assert len(review["frames"]) == 2
    for index in range(0, 4, 2):
        anchor = transforms["frames"][index]
        teacher = transforms["frames"][index + 1]
        assert anchor["transform_matrix"] == teacher["transform_matrix"]
        for field in ("w", "h", "fl_x", "fl_y", "cx", "cy", "camera_id"):
            assert anchor[field] == teacher[field]
        assert anchor["training_role"] == "original_outside_anchor"
        assert teacher["training_role"] == "whole_frame_semantic_teacher"

    source_task = source["tasks"][0]
    source_task_root = Path(source_task["scene_directory"])
    semantic_receipt = json.loads(semantic_paths[0].read_text(encoding="utf-8"))
    for frame, source_frame, teacher_frame in zip(
        task["frames"], source_task["frames"], semantic_receipt["frames"]
    ):
        anchor_path = task_root / frame["anchor_rgb"]["relative_path"]
        original_path = (
            source_task_root / source_frame["rendered_rgb"]["relative_path"]
        )
        teacher_path = task_root / frame["semantic_teacher_rgb"]["relative_path"]
        assert anchor_path.read_bytes() == original_path.read_bytes()
        assert teacher_path.read_bytes() == Path(
            teacher_frame["whole_frame_semantic_teacher"]["path"]
        ).read_bytes()
        mask = np.asarray(
            Image.open(
                task_root / frame["exact_repair_mask"]["relative_path"]
            ).convert("L"),
            dtype=np.uint8,
        ) > 0
        actual_loss = np.asarray(
            Image.open(
                task_root / frame["anchor_loss_mask"]["relative_path"]
            ).convert("L"),
            dtype=np.uint8,
        )
        axis = np.arange(-2, 3)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        expected_excluded = binary_dilation(
            mask,
            structure=(xx * xx + yy * yy) <= 4,
            border_value=0,
        )
        assert set(actual_loss.ravel()) <= {0, 255}
        assert np.array_equal(actual_loss == 0, expected_excluded)
        teacher_index = frame["semantic_teacher_training_index"]
        assert not (task_root / "images" / f"{teacher_index:05d}_mask.png").exists()
        assert frame["teacher_loss_mask_materialized"] is False

    cameras = task_root / task["source_colmap"]["cameras"]["relative_path"]
    images = task_root / task["source_colmap"]["images"]["relative_path"]
    assert struct.unpack("<Q", cameras.read_bytes()[:8])[0] == 4
    assert struct.unpack("<Q", images.read_bytes()[:8])[0] == 4
    assert dual["execution"]["provider_mutations_performed"] == 0
    assert dual["claim_boundary"]["appearance_repair_qualified"] is False
    assert dual["receipt_digest"] == canonical_digest(
        dual, digest_field="receipt_digest"
    )
    assert (
        dual_root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    ).is_file()


def test_supports_one_to_five_selected_tasks(tmp_path: Path) -> None:
    _root, _source, _semantic_paths, dual = _dual_candidate(
        tmp_path,
        count=5,
        cameras_per_task=1,
        radius=0,
    )

    assert dual["replacement_object_count"] == 5
    assert len(dual["tasks"]) == 5
    assert all(task["camera_count"] == 1 for task in dual["tasks"])
    assert all(task["training_record_count"] == 2 for task in dual["tasks"])
    assert all(task["selected_anchor_indices"] == [0] for task in dual["tasks"])
    assert all(task["semantic_teacher_indices"] == [1] for task in dual["tasks"])


def test_rejects_shape_camera_digest_radius_and_nonempty_output(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path / "fixture", count=1, cameras_per_task=2)
    source_root = tmp_path / "source"
    source = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=source_root,
    )
    task = source["tasks"][0]
    task_root = Path(task["scene_directory"])
    bad_shape_root = tmp_path / "bad_shape"
    bad_shape_root.mkdir()
    for frame in task["frames"]:
        original = Image.open(
            task_root / frame["rendered_rgb"]["relative_path"]
        ).convert("RGB")
        if frame["frame_index"] == 0:
            original = original.crop((0, 0, original.width - 1, original.height))
        original.save(bad_shape_root / f"{frame['frame_index']:05d}.png")
    with pytest.raises(DualTargetInputError, match="semantic_teacher_shape_invalid"):
        materialize_whole_frame_semantic_teacher_receipt(
            source_candidate_inputs_receipt_path=(
                source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
            ),
            task_id=task["task_id"],
            semantic_teacher_frames_root=bad_shape_root,
            editor_identity={"backend": "fixture"},
            prompt_policy="object absent",
            output_path=tmp_path / "bad_shape.json",
        )

    semantic_paths = _semantic_receipts(
        tmp_path,
        source_root=source_root,
        source=source,
    )
    tampered = json.loads(semantic_paths[0].read_text(encoding="utf-8"))
    tampered["frames"][0]["camera_id"] = "wrong_camera"
    tampered["receipt_digest"] = canonical_digest(
        tampered, digest_field="receipt_digest"
    )
    semantic_paths[0].write_text(canonical_json(tampered) + "\n", encoding="utf-8")
    with pytest.raises(
        DualTargetInputError, match="semantic_teacher_camera_set_invalid"
    ):
        materialize_dual_target_artifixer3d_inputs(
            source_candidate_inputs_receipt_path=(
                source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
            ),
            semantic_teacher_receipt_paths=semantic_paths,
            output_root=tmp_path / "camera_mismatch",
            transition_radius_pixels=1,
        )

    with pytest.raises(DualTargetInputError, match="transition_radius_invalid"):
        materialize_dual_target_artifixer3d_inputs(
            source_candidate_inputs_receipt_path=(
                source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
            ),
            semantic_teacher_receipt_paths=semantic_paths,
            output_root=tmp_path / "bad_radius",
            transition_radius_pixels=-1,
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "preserve.txt").write_text("user owned", encoding="utf-8")
    with pytest.raises(DualTargetInputError, match="output_not_empty"):
        materialize_dual_target_artifixer3d_inputs(
            source_candidate_inputs_receipt_path=(
                source_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
            ),
            semantic_teacher_receipt_paths=semantic_paths,
            output_root=occupied,
            transition_radius_pixels=1,
        )
    assert (occupied / "preserve.txt").read_text(encoding="utf-8") == "user owned"
