from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    ArtiFixer3DCandidateInputError,
    SCHEMA_VERSION,
    materialize_artifixer3d_candidate_inputs,
    materialize_object_absent_reference_candidate_receipt,
)
from blueprint_pipeline.public_scene_aura_exact_residual_preflight import (
    materialize_aura_exact_residual_preflight,
)
from tests.test_public_scene_aura_exact_residual_preflight import _packet


def _preflight(tmp_path: Path, *, count: int = 2, cameras_per_task: int = 1) -> Path:
    packet = _packet(tmp_path, count=count)
    output = tmp_path / "preflight.json"
    materialize_aura_exact_residual_preflight(input_packet_path=packet, output_path=output)
    if cameras_per_task == 2:
        value = json.loads(output.read_text())
        expanded = []
        for row in value["camera_inputs"]:
            expanded.append(row)
            duplicate = deepcopy(row)
            duplicate["camera_id"] = f"{row['camera_id']}_second"
            duplicate["calibration"]["spec"]["pose"]["T_world_camera_opencv"][0][3] += 0.25
            expanded.append(duplicate)
        value["camera_inputs"] = expanded
        value["preflight_digest"] = canonical_digest(value, digest_field="preflight_digest")
        output.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return output


def test_prepares_one_to_five_exact_support_candidate_inputs(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path, count=5)

    receipt = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=tmp_path / "artifixer",
    )

    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["status"] == "candidate_inputs_prepared_no_model_no_execution"
    assert receipt["replacement_object_count"] == 5
    assert receipt["maximum_replacement_objects"] == 5
    assert len(receipt["tasks"]) == 5
    assert all(row["camera_count"] == 1 for row in receipt["tasks"])
    assert receipt["execution"]["provider_mutations_performed"] == 0
    assert receipt["adapter"]["opacity_role"] == (
        "binary_exact_repair_support_surrogate_not_native_3dgrut_opacity"
    )
    assert receipt["claim_boundary"]["policy_input_use_permitted"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_masks_references_and_builds_exact_inverse_opacity(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    receipt = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=tmp_path / "artifixer",
    )

    task = receipt["tasks"][0]
    task_root = Path(task["scene_directory"])
    frame = task["frames"][0]
    before = np.asarray(
        Image.open(frame["input_retained_frame"]["path"]).convert("RGB"),
        dtype=np.uint8,
    )
    mask = (
        np.asarray(
            Image.open(task_root / frame["exact_repair_mask"]["relative_path"]).convert("L"),
            dtype=np.uint8,
        )
        > 0
    )
    reference = np.asarray(
        Image.open(task_root / frame["masked_reference_rgb"]["relative_path"]).convert("RGB"),
        dtype=np.uint8,
    )
    opacity = np.asarray(
        Image.open(task_root / frame["binary_opacity_surrogate"]["relative_path"]).convert("L"),
        dtype=np.uint8,
    )
    assert np.array_equal(reference[~mask], before[~mask])
    assert np.count_nonzero(reference[mask]) == 0
    assert np.all(opacity[~mask] == 255)
    assert np.all(opacity[mask] == 0)
    assert frame["repair_pixel_count"] == int(np.count_nonzero(mask))
    assert frame["image_pixel_count"] == int(mask.size)
    assert frame["repair_support_fraction"] == pytest.approx(mask.mean())
    assert frame["outside_support_changed_pixels"] == 0
    assert task["repair_support_coverage"] == {
        "minimum_fraction": pytest.approx(mask.mean()),
        "mean_fraction": pytest.approx(mask.mean()),
        "maximum_fraction": pytest.approx(mask.mean()),
        "interpretation": (
            "pre_execution_large_hole_risk_metric_not_method_quality_or_qualification_verdict"
        ),
    }

    transforms = json.loads(Path(task["transforms"]["path"]).read_text())
    source = json.loads(preflight.read_text())
    source_camera = next(
        row
        for row in source["camera_inputs"]
        if row["task_id"] == task["task_id"] and row["camera_id"] == frame["camera_id"]
    )
    expected = np.asarray(
        source_camera["calibration"]["spec"]["pose"]["T_world_camera_opencv"],
        dtype=np.float64,
    ) @ np.diag([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(transforms["frames"][0]["transform_matrix"], expected)


def test_materializes_colmap_seed_and_complementary_direct_folds(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path, count=1, cameras_per_task=2)
    receipt = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=tmp_path / "artifixer",
    )

    task = receipt["tasks"][0]
    task_root = Path(task["scene_directory"])
    distillation = task["artifixer3d_distillation"]
    assert distillation["camera_partition_eligible"] is True
    assert distillation["execution_eligible"] is False
    assert distillation["required_repaired_input_indices"] == [0, 1]
    assert distillation["masked_reference_placeholders_permitted_as_distillation_images"] is False
    assert distillation["selected_anchor_indices"] == [0]
    assert distillation["generated_prediction_indices"] == [1]
    assert task["direct_prediction_coverage_indices"] == [0, 1]
    assert len(task["direct_inference_folds"]) == 2
    for fold in task["direct_inference_folds"]:
        assert not set(fold["selected_indices"]) & set(fold["target_indices"])
        split = json.loads(Path(fold["split_template"]["path"]).read_text(encoding="utf-8"))
        metadata = split["upstream_split"]["test"][task["task_id"]]
        assert metadata["image_root"] == "."
        assert metadata["target_indices_path"].startswith("target_indices.")

    cameras = task_root / "sparse" / "0" / "cameras.bin"
    images = task_root / "sparse" / "0" / "images.bin"
    points = task_root / "sparse" / "0" / "points3D.bin"
    assert struct.unpack("<Q", cameras.read_bytes()[:8])[0] == 2
    assert struct.unpack("<Q", images.read_bytes()[:8])[0] == 2
    assert (
        struct.unpack("<Q", points.read_bytes()[:8])[0]
        == receipt["shared_retained_scene"]["retained_gaussian_count"]
    )
    assert (
        receipt["repair_target_semantics"]["source_washer_or_notebook_restoration_permitted"]
        is False
    )
    assert (
        receipt["repair_target_semantics"]["black_unknown_placeholder_preservation_permitted"]
        is False
    )


def test_binds_one_selected_tasks_exact_support_object_absent_references(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path, count=2, cameras_per_task=2)
    initial_root = tmp_path / "initial"
    initial = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=initial_root,
    )
    task = initial["tasks"][0]
    task_root = Path(task["scene_directory"])
    generated_root = tmp_path / "generated"
    generated_root.mkdir()
    for frame in task["frames"]:
        source = np.asarray(
            Image.open(task_root / frame["rendered_rgb"]["relative_path"]).convert("RGB"),
            dtype=np.uint8,
        ).copy()
        mask = (
            np.asarray(
                Image.open(task_root / frame["exact_repair_mask"]["relative_path"]).convert("L"),
                dtype=np.uint8,
            )
            > 0
        )
        source[mask] = (90, 100, 110)
        Image.fromarray(source, mode="RGB").save(generated_root / f"{frame['frame_index']:05d}.png")
    reference_path = tmp_path / "object_absent_reference.json"
    reference = materialize_object_absent_reference_candidate_receipt(
        source_candidate_inputs_receipt_path=(
            initial_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
        ),
        task_id=task["task_id"],
        object_absent_frames_root=generated_root,
        editor_identity={"backend": "fixture_editor", "snapshot_pinned": True},
        prompt_policy="generic_object_absent_background_completion_v1",
        output_path=reference_path,
    )
    assert reference["outside_support_changed_pixels_total"] == 0

    rebound = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=tmp_path / "rebound",
        selected_task_ids=[task["task_id"]],
        object_absent_reference_receipt_paths=[reference_path],
    )
    assert rebound["replacement_object_count"] == 1
    assert rebound["source_preflight_replacement_object_count"] == 2
    assert rebound["selected_task_ids"] == [task["task_id"]]
    assert rebound["adapter"]["bound_object_absent_reference_task_count"] == 1
    rebound_task = rebound["tasks"][0]
    assert all(
        frame["reference_source"] == "bound_object_absent_exact_support_candidate"
        for frame in rebound_task["frames"]
    )
    for frame in rebound_task["frames"]:
        actual = np.asarray(
            Image.open(
                Path(rebound_task["scene_directory"])
                / frame["masked_reference_rgb"]["relative_path"]
            ).convert("RGB"),
            dtype=np.uint8,
        )
        expected = np.asarray(
            Image.open(generated_root / f"{frame['frame_index']:05d}.png").convert("RGB"),
            dtype=np.uint8,
        )
        assert np.array_equal(actual, expected)


def test_rejects_tampered_preflight_or_nonempty_output(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    value = json.loads(preflight.read_text())
    value["replacement_object_count"] = 5
    preflight.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ArtiFixer3DCandidateInputError, match="calibrated_preflight_invalid"):
        materialize_artifixer3d_candidate_inputs(
            calibrated_residual_preflight_path=preflight,
            output_root=tmp_path / "artifixer",
        )

    valid = _preflight(tmp_path / "valid")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "user-owned.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ArtiFixer3DCandidateInputError, match="output_not_empty"):
        materialize_artifixer3d_candidate_inputs(
            calibrated_residual_preflight_path=valid, output_root=output
        )
    assert (output / "user-owned.txt").read_text(encoding="utf-8") == "preserve"


def test_rejects_symlinked_preflight(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    link = tmp_path / "preflight-link.json"
    link.symlink_to(preflight)

    with pytest.raises(ArtiFixer3DCandidateInputError, match="preflight_missing"):
        materialize_artifixer3d_candidate_inputs(
            calibrated_residual_preflight_path=link,
            output_root=tmp_path / "artifixer",
        )
