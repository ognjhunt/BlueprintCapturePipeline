from __future__ import annotations

import json

from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline import website_mask_view_plan as views


def test_selected_original_views_keep_task_anchor_context_and_full_timeline(tmp_path, monkeypatch):
    video = tmp_path / "source.mov"
    video.write_bytes(b"video")
    geometry_frames = []
    for index in (0, 2, 4, 6, 8):
        image = tmp_path / f"source-{index}.png"
        Image.new("RGB", (16, 12), (index, 0, 0)).save(image)
        geometry_frames.append({"frame_id": f"decoded-{index:09d}", "timestamp_seconds": index / 10,
                                "source_image_path": str(image), "source_image_digest": _sha256_file(image),
                                "display_rotation_degrees": 0})
    source_root = tmp_path / "source_geometry"
    source_root.mkdir()
    split = {"capture_digest": _sha256_file(video),
             "assignments": [{"decoded_frame_index": 5, "split": "held_out"}]}
    split["split_digest"] = canonical_digest(split, digest_field="split_digest")
    (source_root / "frozen_split_manifest.json").write_text(json.dumps(split))
    index = {"capture_digest": _sha256_file(video), "frozen_split_digest": split["split_digest"],
             "decoded_presentation_times_seconds": [i / 10 for i in range(9)]}
    (source_root / "decoded_observation_index.json").write_text(json.dumps(index))
    geometry = {"digest": "geometry", "binding": {"source_video_digest": _sha256_file(video)},
                "frames": geometry_frames}
    task = {"task_context_sha256": "task", "targets": [{"task_effect": "manipulated",
        "disposition": "remove", "spatial_evidence": [{"timestamp_seconds": 0.3}]}]}

    def extract(**kwargs):
        assert kwargs["indexes"] == [3]
        path = kwargs["frame_root"] / "decoded-000000003.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (16, 12), (3, 0, 0)).save(path)
        return [{"frame_id": "decoded-000000003", "digest": _sha256_file(path)}]

    def encode(**kwargs):
        assert len(kwargs["registry"]) == 4
        clip = kwargs["root"] / "retained-frames.mp4"
        clip.write_bytes(b"selected clip")
        return clip

    monkeypatch.setattr(views, "_extract_frames", extract)
    monkeypatch.setattr(views, "encode_clip", encode)
    result = views.prepare_mask_view_plan(source_video=video, source_geometry=geometry, plan=task,
        source_geometry_root=source_root, output_root=tmp_path / "views", limit=4)
    assert result["binding"]["selected_frame_ids"] == [
        "decoded-000000000", "decoded-000000002", "decoded-000000003", "decoded-000000008"]
    assert len(result["sparse_registry"]) == 4
    assert len(result["source_frame_registry"]) == 9
    assert "decoded-000000005" not in result["binding"]["selected_frame_ids"]
    assert views.prepare_mask_view_plan(source_video=video, source_geometry=geometry, plan=task,
        source_geometry_root=source_root, output_root=tmp_path / "views", limit=4) == result
