from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_artifixer3d_final_composite import (
    ArtiFixer3DFinalCompositeError,
    materialize_artifixer3d_final_composite,
)


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _packet(root: Path, task_id: str, *, radius: int = 2) -> tuple[Path, Path]:
    scene = root / task_id
    scene.mkdir(parents=True)
    original = np.full((9, 9, 3), 20, dtype=np.uint8)
    generated = np.full((9, 9, 3), 220, dtype=np.uint8)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[4, 4] = 255
    original_path = scene / "original.png"
    generated_path = scene / "generated.png"
    mask_path = scene / "mask.png"
    Image.fromarray(original).save(original_path)
    Image.fromarray(generated).save(generated_path)
    Image.fromarray(mask).save(mask_path)
    dual: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_dual_target_inputs.v1",
        "status": "paired_target_inputs_prepared_no_model_no_execution",
        "pipeline_mode": "dual_target_artifixer3d_only",
        "publisher_scene_id": "840920",
        "transition_support": {
            "radius_pixels": radius,
            "morphology": "euclidean_disk_inclusive_radius_constant_zero_border",
        },
        "tasks": [
            {
                "task_id": task_id,
                "scene_directory": str(scene),
                "physical_camera_count": 1,
                "frames": [
                    {
                        "physical_camera_index": 0,
                        "camera_id": "camera_0",
                        "source_original_frame": _record(original_path),
                        "source_exact_repair_mask": _record(mask_path),
                    }
                ],
            }
        ],
        "receipt_digest": "",
    }
    dual["receipt_digest"] = canonical_digest(dual, digest_field="receipt_digest")
    dual_path = root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    _write_json(dual_path, dual)
    raw: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_raw_result.v1",
        "pipeline_mode": "dual_target_artifixer3d_only",
        "appearance_repair_qualified": False,
        "tasks": [
            {
                "task_id": task_id,
                "artifixer3d_review_frames": [
                    {"frame_index": 0, "camera_id": "camera_0", **_record(generated_path)}
                ],
            }
        ],
        "result_digest": "",
    }
    raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
    raw_path = root / "raw.json"
    _write_json(raw_path, raw)
    return dual_path, raw_path


def test_composes_one_to_five_tasks_and_proves_outside_identity(tmp_path: Path) -> None:
    duals = []
    raws = []
    for index in range(5):
        dual, raw = _packet(tmp_path / f"packet_{index}", f"task_{index}")
        duals.append(dual)
        raws.append(raw)
    result = materialize_artifixer3d_final_composite(
        dual_input_receipt_paths=duals,
        raw_result_paths=raws,
        output_root=tmp_path / "output",
    )
    assert result["replacement_object_count"] == 5
    assert result["outside_support_changed_pixels_total"] == 0
    assert result["outside_support_invariance_proven"] is True
    assert result["appearance_repair_qualified"] is False
    for task in result["tasks"]:
        row = task["frames"][0]
        assert Path(row["path"]).is_file()
        assert row["sha256"] == row["final_frame"]["sha256"]
        frame = np.asarray(Image.open(tmp_path / "output" / task["task_id"] / row["final_frame"]["relative_path"]))
        assert np.array_equal(frame[0, 0], [20, 20, 20])
        assert np.array_equal(frame[4, 4], [220, 220, 220])
        assert 20 < frame[4, 5, 0] < 220


def test_rejects_camera_mismatch_and_digest_tamper(tmp_path: Path) -> None:
    dual, raw = _packet(tmp_path / "packet", "task_a")
    value = json.loads(raw.read_text())
    value["tasks"][0]["artifixer3d_review_frames"][0]["camera_id"] = "wrong"
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    _write_json(raw, value)
    with pytest.raises(ArtiFixer3DFinalCompositeError, match="camera_binding_invalid"):
        materialize_artifixer3d_final_composite(
            dual_input_receipt_paths=[dual],
            raw_result_paths=[raw],
            output_root=tmp_path / "output",
        )


def test_rejects_more_than_five_tasks(tmp_path: Path) -> None:
    duals, raws = zip(
        *[_packet(tmp_path / f"packet_{index}", f"task_{index}") for index in range(6)],
        strict=True,
    )
    with pytest.raises(ArtiFixer3DFinalCompositeError, match="task_count_invalid"):
        materialize_artifixer3d_final_composite(
            dual_input_receipt_paths=duals,
            raw_result_paths=raws,
            output_root=tmp_path / "output",
        )
