"""End-to-end tests for synthesize.py — synthesize_view() with local fake storage."""
from __future__ import annotations

import json

import numpy as np

from blueprint_pipeline.synthesis.synthesize import synthesize_view, synthesize_route


class TestSynthesizeView:
    def test_splat_only_completes(self, fake_storage_root, simple_intrinsics, tmp_path):
        """synthesize_view() in splat_only mode should return status=completed."""
        output = tmp_path / "out.jpg"
        result = synthesize_view(
            site_id="site-xyz",
            storage_root=fake_storage_root,
            bucket="bucket",
            target_T_world_camera=np.eye(4),
            target_intrinsics=simple_intrinsics,
            target_h=simple_intrinsics["height"],
            target_w=simple_intrinsics["width"],
            output_path=output,
            mode="splat_only",
            k=1,
            query_mode="spatial",
            depth_scale=0.001,
        )
        assert result["status"] == "completed"
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_splat_only_returns_coverage_frac(self, fake_storage_root, simple_intrinsics, tmp_path):
        """coverage_frac should be in [0, 1]."""
        output = tmp_path / "out.jpg"
        result = synthesize_view(
            site_id="site-xyz",
            storage_root=fake_storage_root,
            bucket="bucket",
            target_T_world_camera=np.eye(4),
            target_intrinsics=simple_intrinsics,
            target_h=simple_intrinsics["height"],
            target_w=simple_intrinsics["width"],
            output_path=output,
            mode="splat_only",
            depth_scale=0.001,
        )
        assert 0.0 <= result["coverage_frac"] <= 1.0

    def test_no_site_index_returns_failed(self, tmp_path, simple_intrinsics):
        """Missing site_reference_index.jsonl → status=failed."""
        # Empty storage root with no index
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        result = synthesize_view(
            site_id="no-such-site",
            storage_root=empty_root,
            bucket="bucket",
            target_T_world_camera=np.eye(4),
            target_intrinsics=simple_intrinsics,
            target_h=simple_intrinsics["height"],
            target_w=simple_intrinsics["width"],
            output_path=tmp_path / "out.jpg",
            mode="splat_only",
        )
        assert result["status"] == "failed"

    def test_reference_used_contains_frame_id(self, fake_storage_root, simple_intrinsics, tmp_path):
        """result['reference_used'] must contain capture_id and frame_id."""
        output = tmp_path / "out.jpg"
        result = synthesize_view(
            site_id="site-xyz",
            storage_root=fake_storage_root,
            bucket="bucket",
            target_T_world_camera=np.eye(4),
            target_intrinsics=simple_intrinsics,
            target_h=simple_intrinsics["height"],
            target_w=simple_intrinsics["width"],
            output_path=output,
            mode="splat_only",
            depth_scale=0.001,
        )
        assert result["status"] == "completed"
        ref = result["reference_used"]
        assert "frame_id" in ref
        assert "capture_id" in ref

    def test_depth_upsampling_does_not_crash(self, tmp_path, simple_intrinsics):
        """
        Explicitly test that depth at 256×192 is upsampled correctly to image resolution.
        Writes a 256×192 uint16 depth PNG but an 1920×1440 JPEG frame — this is the
        actual ARKit mismatch scenario that the _load_ref_depth fix addresses.
        """
        from PIL import Image as PILImage
        import io

        DEPTH_H, DEPTH_W = 192, 256
        IMAGE_H, IMAGE_W = simple_intrinsics["height"], simple_intrinsics["width"]

        root = tmp_path / "gcs"
        depth_dir = root / "bucket" / "depth"
        frames_dir = root / "bucket" / "frames"
        embed_dir = root / "bucket" / "embeddings"
        index_dir = root / "bucket" / "sites" / "site-mismatch" / "reference_memory"
        for d in (depth_dir, frames_dir, embed_dir, index_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 256×192 depth PNG (ARKit resolution)
        depth_arr = np.full((DEPTH_H, DEPTH_W), 2000, dtype=np.uint16)
        depth_img = PILImage.fromarray(depth_arr, mode="I;16")
        buf = io.BytesIO()
        depth_img.save(buf, format="PNG")
        (depth_dir / "000001.png").write_bytes(buf.getvalue())

        # 1920×1440 RGB frame
        rgb = np.full((IMAGE_H, IMAGE_W, 3), 128, dtype=np.uint8)
        rgb_img = PILImage.fromarray(rgb)
        buf2 = io.BytesIO()
        rgb_img.save(buf2, format="JPEG", quality=85)
        (frames_dir / "000001.jpg").write_bytes(buf2.getvalue())

        embed = np.ones(1024, dtype=np.float32)
        embed /= np.linalg.norm(embed)
        (embed_dir / "000001.bin").write_bytes(embed.tobytes())

        rec = {
            "reference_id": "ref-0001",
            "frame_id": "000001",
            "capture_id": "cap-mismatch",
            "scene_id": "scene-mm",
            "site_id": "site-mismatch",
            "pass_id": "pass-001",
            "T_world_camera": np.eye(4).tolist(),
            "intrinsics": simple_intrinsics,
            "depth_uri": "gs://bucket/depth/000001.png",
            "frame_uri": "gs://bucket/frames/000001.jpg",
            "embedding_uri": "gs://bucket/embeddings/000001.bin",
            "site_frame_transform": None,
            "quality": {"tracking_state": "normal", "sharpness_score": 100.0},
        }
        (index_dir / "site_reference_index.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8"
        )

        output = tmp_path / "out.jpg"
        result = synthesize_view(
            site_id="site-mismatch",
            storage_root=root,
            bucket="bucket",
            target_T_world_camera=np.eye(4),
            target_intrinsics=simple_intrinsics,
            target_h=IMAGE_H,
            target_w=IMAGE_W,
            output_path=output,
            mode="splat_only",
            depth_scale=0.001,
        )
        assert result["status"] == "completed", result.get("reason")
        assert output.is_file()

    def test_portrait_reference_frame_is_rotated_to_match_intrinsics(self, tmp_path, simple_intrinsics):
        """
        ARKit reference JPEGs can be saved in display orientation (1440×1920)
        while the indexed intrinsics/depth stay in encoded orientation
        (1920×1440). synthesize_view() should rotate the RGB frame back into the
        indexed camera frame before splatting.
        """
        from PIL import Image as PILImage
        import io

        DEPTH_H, DEPTH_W = 192, 256
        root = tmp_path / "gcs"
        depth_dir = root / "bucket" / "depth"
        frames_dir = root / "bucket" / "frames"
        embed_dir = root / "bucket" / "embeddings"
        index_dir = root / "bucket" / "sites" / "site-portrait" / "reference_memory"
        for d in (depth_dir, frames_dir, embed_dir, index_dir):
            d.mkdir(parents=True, exist_ok=True)

        depth_arr = np.full((DEPTH_H, DEPTH_W), 2000, dtype=np.uint16)
        depth_img = PILImage.fromarray(depth_arr, mode="I;16")
        buf = io.BytesIO()
        depth_img.save(buf, format="PNG")
        (depth_dir / "000001.png").write_bytes(buf.getvalue())

        portrait_rgb = np.zeros((simple_intrinsics["width"], simple_intrinsics["height"], 3), dtype=np.uint8)
        portrait_rgb[:, : simple_intrinsics["height"] // 2] = [255, 0, 0]
        portrait_rgb[:, simple_intrinsics["height"] // 2 :] = [0, 255, 0]
        rgb_img = PILImage.fromarray(portrait_rgb)
        buf2 = io.BytesIO()
        rgb_img.save(buf2, format="JPEG", quality=85)
        (frames_dir / "000001.jpg").write_bytes(buf2.getvalue())

        embed = np.ones(1024, dtype=np.float32)
        embed /= np.linalg.norm(embed)
        (embed_dir / "000001.bin").write_bytes(embed.tobytes())

        rec = {
            "reference_id": "ref-0001",
            "frame_id": "000001",
            "capture_id": "cap-portrait",
            "scene_id": "scene-portrait",
            "site_id": "site-portrait",
            "pass_id": "pass-001",
            "T_world_camera": np.eye(4).tolist(),
            "intrinsics": simple_intrinsics,
            "depth_uri": "gs://bucket/depth/000001.png",
            "frame_uri": "gs://bucket/frames/000001.jpg",
            "embedding_uri": "gs://bucket/embeddings/000001.bin",
            "site_frame_transform": None,
            "quality": {"tracking_state": "normal", "sharpness_score": 100.0},
        }
        (index_dir / "site_reference_index.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8"
        )

        output = tmp_path / "out-portrait.jpg"
        result = synthesize_view(
            site_id="site-portrait",
            storage_root=root,
            bucket="bucket",
            target_T_world_camera=np.eye(4),
            target_intrinsics=simple_intrinsics,
            target_h=simple_intrinsics["height"],
            target_w=simple_intrinsics["width"],
            output_path=output,
            mode="splat_only",
            depth_scale=0.001,
        )
        assert result["status"] == "completed", result.get("reason")
        assert output.is_file()


class TestSynthesizeRoute:
    def test_route_synthesises_all_frames(self, fake_storage_root, simple_intrinsics, tmp_path):
        """synthesize_route() should produce one JPEG per pose."""
        poses = [np.eye(4) for _ in range(3)]
        out_dir = tmp_path / "route"
        result = synthesize_route(
            site_id="site-xyz",
            storage_root=fake_storage_root,
            bucket="bucket",
            target_poses=poses,
            target_intrinsics=simple_intrinsics,
            target_h=simple_intrinsics["height"],
            target_w=simple_intrinsics["width"],
            output_dir=out_dir,
            mode="splat_only",
            depth_scale=0.001,
        )
        assert result["frame_count"] == 3
        assert result["frames_synthesised"] == 3
        for i in range(3):
            assert (out_dir / f"{i:06d}.jpg").is_file()
