from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.public_scene_inpainting_locality import (
    PublicSceneInpaintingLocalityError,
    measure_inpainting_locality,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    before_dir = tmp_path / "before"
    mask_dir = tmp_path / "masks"
    after_dir = tmp_path / "after/frames"
    before_dir.mkdir()
    mask_dir.mkdir()
    after_dir.mkdir(parents=True)
    before = np.full((16, 16, 3), 100, dtype=np.uint8)
    after = before.copy()
    after[6:10, 6:10] = 200
    after[0, 0] = 130
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 1
    Image.fromarray(before).save(before_dir / "view.png")
    Image.fromarray(after).save(after_dir / "view.png")
    Image.fromarray(mask).save(mask_dir / "view.png")
    after_path = after_dir / "view.png"
    manifest = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "scene": {
            "publisher_scene_id": "840796",
            "target_instance_id": "ins123",
        },
        "sealed_camera_render_manifest_digest": "sha256:" + "a" * 64,
        "renders": [
            {
                "camera_id": "view",
                "relative_path": "frames/view.png",
                "digest": _sha256(after_path),
            }
        ],
    }
    manifest_path = tmp_path / "after/sealed_camera_render_manifest.v1.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return before_dir, mask_dir, manifest_path


def test_locality_measures_only_outside_mask_and_hashes_actual_files(tmp_path: Path) -> None:
    before, masks, manifest = _fixture(tmp_path)
    receipt = measure_inpainting_locality(
        before_dir=before,
        mask_dir=masks,
        after_render_manifest=manifest,
        output_path=tmp_path / "measurement.json",
        approved_roots=[tmp_path],
        dilation_pixels=0,
    )

    assert receipt["status"] == "measured_no_admission_effect"
    assert receipt["scene"] == {
        "publisher_scene_id": "840796",
        "target_instance_id": "ins123",
    }
    assert receipt["aggregate"]["view_count"] == 1
    assert receipt["rows"][0]["dilated_mask_pixel_count"] == 16
    assert receipt["rows"][0]["outside_mask_pixel_count"] == 240
    assert receipt["rows"][0]["outside_mask_fraction_max_channel_delta_gt_20_255"] == pytest.approx(
        1 / 240
    )
    assert receipt["quality_pass_claimed"] is False
    assert receipt["admission_effect"] == "none"
    assert (tmp_path / "measurement.json").is_file()


def test_locality_rejects_changed_after_bytes_and_paths_outside_roots(tmp_path: Path) -> None:
    before, masks, manifest = _fixture(tmp_path)
    after = tmp_path / "after/frames/view.png"
    after.write_bytes(after.read_bytes() + b"changed")
    with pytest.raises(PublicSceneInpaintingLocalityError, match="digest_mismatch"):
        measure_inpainting_locality(
            before_dir=before,
            mask_dir=masks,
            after_render_manifest=manifest,
            output_path=tmp_path / "measurement.json",
            approved_roots=[tmp_path],
            dilation_pixels=0,
        )
    with pytest.raises(PublicSceneInpaintingLocalityError, match="outside_approved_roots"):
        measure_inpainting_locality(
            before_dir=before,
            mask_dir=masks,
            after_render_manifest=manifest,
            output_path=tmp_path / "measurement.json",
            approved_roots=[tmp_path / "unrelated"],
            dilation_pixels=0,
        )


def test_locality_releases_full_resolution_working_set_after_each_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, masks, manifest = _fixture(tmp_path)
    source = before / "view.png"
    mask = masks / "view.png"
    after = tmp_path / "after/frames/view.png"
    source.replace(before / "view_a.png")
    mask.replace(masks / "view_a.png")
    after.replace(tmp_path / "after/frames/view_a.png")
    Image.open(before / "view_a.png").save(before / "view_b.png")
    Image.open(masks / "view_a.png").save(masks / "view_b.png")
    Image.open(tmp_path / "after/frames/view_a.png").save(
        tmp_path / "after/frames/view_b.png"
    )
    value = json.loads(manifest.read_text())
    value["renders"] = [
        {
            "camera_id": camera_id,
            "relative_path": f"frames/{camera_id}.png",
            "digest": _sha256(tmp_path / f"after/frames/{camera_id}.png"),
        }
        for camera_id in ("view_a", "view_b")
    ]
    manifest.write_text(json.dumps(value), encoding="utf-8")
    collections = 0

    def observed_collect() -> int:
        nonlocal collections
        collections += 1
        return 0

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpainting_locality.gc.collect",
        observed_collect,
    )

    receipt = measure_inpainting_locality(
        before_dir=before,
        mask_dir=masks,
        after_render_manifest=manifest,
        output_path=tmp_path / "measurement.json",
        approved_roots=[tmp_path],
        dilation_pixels=0,
    )

    assert receipt["aggregate"]["view_count"] == 2
    assert collections == 2
