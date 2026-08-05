from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.materialize_inpaint360_virtual_masks import materialize_virtual_masks


def _write_mask(path: Path, *, target_id: int = 7, include_target: bool = True) -> None:
    labels = np.zeros((6, 8), dtype=np.uint8)
    if include_target:
        labels[2:4, 3:6] = target_id
    Image.fromarray(labels, mode="L").save(path)


def test_materializes_exact_binary_masks_and_receipt(tmp_path: Path) -> None:
    runtime = tmp_path / "provider_runtime"
    evidence = tmp_path / "runtime_output"
    source = runtime / "virtual" / "objects_pred"
    source.mkdir(parents=True)
    for index in range(3):
        _write_mask(source / f"{index:05d}.png")
    output = runtime / "tracking_results" / "images" / "images_masks"
    receipt_path = evidence / "virtual_mask_receipt.json"

    receipt = materialize_virtual_masks(
        runtime_root=runtime,
        evidence_root=evidence,
        predicted_mask_dir=source,
        output_dir=output,
        receipt_path=receipt_path,
        target_instance_id=7,
        expected_count=3,
    )

    assert receipt["status"] == "completed"
    assert receipt["view_count"] == 3
    assert receipt["handoff_kind"] == "binary_target_mask_without_interactive_refinement"
    assert all(row["foreground_pixels"] == 6 for row in receipt["output_masks"])
    observed = np.asarray(Image.open(output / "00000.png"))
    assert set(np.unique(observed)) == {0, 255}
    assert int(np.count_nonzero(observed)) == 6
    assert receipt_path.is_file()


def test_rejects_view_without_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_mask(source / "00000.png", include_target=False)

    with pytest.raises(ValueError, match="target_missing_from_view"):
        materialize_virtual_masks(
            runtime_root=tmp_path,
            predicted_mask_dir=source,
            output_dir=tmp_path / "output",
            receipt_path=tmp_path / "receipt.json",
            target_instance_id=7,
            expected_count=1,
        )


def test_rejects_paths_outside_runtime_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_mask(source / "00000.png")
    outside = tmp_path.parent / "outside-inpaint360-masks"

    with pytest.raises(ValueError, match="path_outside_runtime_root"):
        materialize_virtual_masks(
            runtime_root=tmp_path,
            predicted_mask_dir=source,
            output_dir=outside,
            receipt_path=tmp_path / "receipt.json",
            target_instance_id=7,
            expected_count=1,
        )


def test_rejects_receipt_outside_evidence_root(tmp_path: Path) -> None:
    runtime = tmp_path / "provider_runtime"
    evidence = tmp_path / "runtime_output"
    source = runtime / "source"
    source.mkdir(parents=True)
    _write_mask(source / "00000.png")

    with pytest.raises(ValueError, match="receipt_outside_evidence_root"):
        materialize_virtual_masks(
            runtime_root=runtime,
            evidence_root=evidence,
            predicted_mask_dir=source,
            output_dir=runtime / "output",
            receipt_path=tmp_path / "unapproved" / "receipt.json",
            target_instance_id=7,
            expected_count=1,
        )
