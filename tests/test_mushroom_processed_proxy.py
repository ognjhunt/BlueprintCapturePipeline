from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline import mushroom_processed_proxy as proxy


def _write_capture(root: Path, capture: str, count: int) -> None:
    frames = []
    for index in range(1, count + 1):
        frame_id = f"frame_{index:05d}"
        image_path = root / capture / "images" / f"{frame_id}.jpg"
        depth_path = root / capture / "depth" / f"{frame_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (index, 20, 30)).save(image_path)
        Image.new("I;16", (8, 6), index * 100).save(depth_path)
        frames.append(
            {
                "fl_x": 5.0,
                "fl_y": 5.0,
                "cx": 4.0,
                "cy": 3.0,
                "h": 6,
                "w": 8,
                "file_path": f"./images/{frame_id}.jpg",
                "depth_file_path": f"./depth/{frame_id}.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, index / 10],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (root / capture / "transformations_colmap.json").write_text(
        json.dumps({"camera_model": "OPENCV", "frames": frames}), encoding="utf-8"
    )


def test_real_layout_freezes_author_and_short_views_outside_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "koivu_iphone.tar.gz"
    archive.write_bytes(b"fixture-archive")
    monkeypatch.setattr(proxy, "ARCHIVE_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        proxy,
        "ARCHIVE_SHA256",
        "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(proxy, "PUBLISHER_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    scene = tmp_path / "iphone"
    _write_capture(scene, "long_capture", 8)
    _write_capture(scene, "short_capture", 2)
    (scene / "long_capture" / "test.txt").write_text(
        "frame_00002\nframe_00007\n", encoding="utf-8"
    )

    report = proxy.compile_mushroom_processed_iphone_proxy(
        scene_root=scene,
        archive_path=archive,
        output_root=tmp_path / "output",
        source_commit_sha="a" * 40,
        authority_used={
            "license": "CC-BY-4.0",
            "local_processing_authorized": True,
            "provider_upload_authorized": True,
        },
        timestamp="2026-08-01T00:00:00Z",
    )
    assert report["status"] == "candidate_training_proxy_ready"
    assert report["candidate_count"] == 6
    assert report["author_hidden_count"] == 2
    assert report["independent_short_count"] == 2
    assert report["candidate_may_access_hidden_heldout"] is False
    assert report["raw_contract_3_2_proven"] is False
    export = report["colmap_training_dataset_export_result"]
    assert export["image_count"] == 6
    assert export["hidden_heldout_pixels_included"] is False
    root = next((tmp_path / "output").glob("mushroom_proxy_*"))
    candidates = json.loads((root / "candidate_dataset_manifest.json").read_text())
    assert {row["frame_id"] for row in candidates["frames"]}.isdisjoint(
        {"frame_00002", "frame_00007"}
    )
    hidden = json.loads(
        (root / "evaluator_hidden" / "hidden_evaluator_manifest.json").read_text()
    )
    assert {row["trajectory"] for row in hidden["observations"]} == {
        "long_author_test",
        "independent_short",
    }

    replay = proxy.compile_mushroom_processed_iphone_proxy(
        scene_root=scene,
        archive_path=archive,
        output_root=tmp_path / "output",
        source_commit_sha="a" * 40,
        authority_used={
            "license": "CC-BY-4.0",
            "local_processing_authorized": True,
            "provider_upload_authorized": True,
        },
        timestamp="2026-08-01T00:00:00Z",
    )
    assert replay == report
