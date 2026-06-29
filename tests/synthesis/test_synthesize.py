"""End-to-end tests for synthesize.py — synthesize_view() with local fake storage."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.synthesis import synthesize as synth
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


def test_synthesize_view_failure_and_lookahead_edges(monkeypatch, fake_storage_root, simple_intrinsics, tmp_path):
    index = fake_storage_root / "bucket" / "sites" / "site-xyz" / "reference_memory" / "site_reference_index.jsonl"
    assert index.is_file()

    monkeypatch.setattr(synth, "query_site", lambda **_kwargs: [])
    no_refs = synthesize_view(
        site_id="site-xyz",
        storage_root=fake_storage_root,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        target_intrinsics=simple_intrinsics,
        target_h=simple_intrinsics["height"],
        target_w=simple_intrinsics["width"],
        output_path=tmp_path / "no-ref.jpg",
    )
    assert no_refs["reason"] == "no_reference_frames_found"

    missing_image_rec = {"reference_id": "missing", "frame_uri": str(tmp_path / "missing.jpg"), "T_world_camera": np.eye(4).tolist()}
    monkeypatch.setattr(synth, "query_site", lambda **_kwargs: [missing_image_rec])
    missing_image = synthesize_view(
        site_id="site-xyz",
        storage_root=fake_storage_root,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        target_intrinsics=simple_intrinsics,
        target_h=simple_intrinsics["height"],
        target_w=simple_intrinsics["width"],
        output_path=tmp_path / "missing-image.jpg",
    )
    assert missing_image["reason"] == "could_not_load_reference_image"

    frame = tmp_path / "frame.jpg"
    Image.fromarray(np.full((4, 4, 3), 100, dtype=np.uint8)).save(frame)
    no_pose_rec = {"reference_id": "no-pose", "frame_uri": str(frame)}
    monkeypatch.setattr(synth, "query_site", lambda **_kwargs: [no_pose_rec])
    no_pose = synthesize_view(
        site_id="site-xyz",
        storage_root=fake_storage_root,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 4, "fy": 4, "cx": 2, "cy": 2, "width": 4, "height": 4},
        target_h=4,
        target_w=4,
        output_path=tmp_path / "no-pose.jpg",
    )
    assert no_pose["reason"] == "reference_has_no_pose"

    calls = []
    ref = {
        "reference_id": "ref",
        "capture_id": "cap",
        "frame_id": "frame",
        "frame_uri": str(frame),
        "T_world_camera": np.eye(4).tolist(),
        "site_frame_transform": np.eye(4).tolist(),
    }

    def fake_query_site(**kwargs):
        calls.append(kwargs["target_T_world_camera"])
        return [ref]

    monkeypatch.setattr(synth, "query_site", fake_query_site)
    monkeypatch.setattr(
        synth,
        "depth_splat",
        lambda **_kwargs: (np.full((4, 4, 3), 120, dtype=np.uint8), np.ones((4, 4), dtype=bool)),
    )

    from blueprint_pipeline.synthesis import cosmos_inference

    def fake_generate_view(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"jpg")
        Path(kwargs["output_path"]).with_suffix(".mp4").write_bytes(b"mp4")

    monkeypatch.setattr(cosmos_inference, "generate_view", fake_generate_view)
    tail = tmp_path / "tail.jpg"
    Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8)).save(tail)
    lookahead = tmp_path / "lookahead.jpg"
    Image.fromarray(np.full((4, 4, 3), 80, dtype=np.uint8)).save(lookahead)
    completed = synthesize_view(
        site_id="site-xyz",
        storage_root=fake_storage_root,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_target_T_world_camera=np.eye(4),
        lookahead_ref_uris=[str(lookahead)],
        target_intrinsics={"fx": 4, "fy": 4, "cx": 2, "cy": 2, "width": 4, "height": 4},
        target_h=4,
        target_w=4,
        output_path=tmp_path / "complete.jpg",
        mode="cosmos_i2w",
        previous_tail_path=tail,
        previous_tail_alpha=1.0,
        num_frames=3,
        cosmos_width=4,
        cosmos_height=4,
        cosmos_guidance_scale=2,
        cosmos_num_steps=1,
    )
    assert completed["status"] == "completed"
    assert completed["video_path"]
    assert completed["lookahead_references"][0]["reference_id"] == "ref"
    assert len(calls) == 2


def test_synthesize_private_helpers_and_route_failures(monkeypatch, tmp_path):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = tmp_path / "frame.jpg"
    Image.fromarray(image).save(frame)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"bad")
    storage_root = tmp_path / "gcs"
    gs_frame = storage_root / "bucket" / "frames" / "frame.jpg"
    gs_frame.parent.mkdir(parents=True)
    gs_frame.write_bytes(frame.read_bytes())
    flat_frame = storage_root / "frames" / "flat.jpg"
    flat_frame.parent.mkdir(parents=True)
    flat_frame.write_bytes(frame.read_bytes())

    assert synth._load_ref_image({}, storage_root=storage_root, bucket="bucket") is None
    assert synth._load_ref_image({"frame_uri": str(tmp_path / "missing.jpg")}, storage_root=storage_root, bucket="bucket") is None
    assert synth._load_ref_image({"frame_uri": str(corrupt)}, storage_root=storage_root, bucket="bucket") is None
    assert synth._load_ref_image({"frame_uri": "gs://bucket/frames/frame.jpg"}, storage_root=storage_root, bucket="bucket").shape == (4, 4, 3)
    assert synth._reference_summary({"reference_id": 1, "capture_id": 2, "frame_id": 3, "frame_uri": 4}) == {
        "reference_id": "1",
        "capture_id": "2",
        "frame_id": "3",
        "frame_uri": "4",
    }
    assert synth._uri_to_local(str(frame), storage_root=storage_root, bucket="bucket") == frame
    assert synth._uri_to_local("gs://bucket/frames/frame.jpg", storage_root=storage_root, bucket="bucket") == gs_frame
    assert synth._uri_to_local("gs://other/frames/flat.jpg", storage_root=storage_root, bucket="bucket") == flat_frame
    assert synth._uri_to_local("gs://bucket/missing.jpg", storage_root=storage_root, bucket="bucket") == storage_root / "bucket" / "missing.jpg"

    assert synth._blend_previous_tail(image, previous_tail_path=None, alpha=0.5) is image
    assert np.array_equal(synth._blend_previous_tail(image, previous_tail_path=tmp_path / "missing.jpg", alpha=0.5), image)
    blended = synth._blend_previous_tail(image, previous_tail_path=frame, alpha=0.5)
    assert blended.shape == image.shape
    assert np.array_equal(synth._blend_previous_tail(image, previous_tail_path=corrupt, alpha=0.5), image)

    assert np.array_equal(
        synth._build_conditioning_image(
            image,
            previous_tail_path=None,
            previous_tail_alpha=0,
            lookahead_ref_uris=None,
            storage_root=storage_root,
            bucket="bucket",
        ),
        image,
    )
    assert np.array_equal(
        synth._build_conditioning_image(
            image,
            previous_tail_path=corrupt,
            previous_tail_alpha=1.0,
            lookahead_ref_uris=None,
            storage_root=storage_root,
            bucket="bucket",
        ),
        image,
    )
    assert np.array_equal(
        synth._build_conditioning_image(
            image,
            previous_tail_path=None,
            previous_tail_alpha=0,
            lookahead_ref_uris=[str(tmp_path / "missing.jpg"), str(corrupt)],
            storage_root=storage_root,
            bucket="bucket",
        ),
        image,
    )
    tiled = synth._build_conditioning_image(
        image,
        previous_tail_path=frame,
        previous_tail_alpha=0.5,
        lookahead_ref_uris=[str(frame), str(tmp_path / "missing.jpg"), str(corrupt)],
        storage_root=storage_root,
        bucket="bucket",
    )
    assert tiled.shape == image.shape

    assert synth._load_ref_depth({}, storage_root=storage_root, bucket="bucket", depth_scale=0.001) is None
    assert synth._load_ref_depth({"depth_uri": str(tmp_path / "missing.png")}, storage_root=storage_root, bucket="bucket", depth_scale=0.001) is None
    assert synth._load_ref_depth({"depth_uri": str(corrupt)}, storage_root=storage_root, bucket="bucket", depth_scale=0.001) is None
    assert synth._effective_pose({}) is None
    assert synth._effective_pose({"T_world_camera": [1, 2, 3]}) is None
    transformed = synth._effective_pose({"T_world_camera": np.eye(4).tolist(), "site_frame_transform": (np.eye(4) * 2).tolist()})
    assert transformed[3, 3] == 2

    monkeypatch.setattr(synth, "synthesize_view", lambda **_kwargs: {"status": "failed", "reason": "x"})
    route = synthesize_route(
        site_id="site",
        storage_root=storage_root,
        bucket="bucket",
        target_poses=[np.eye(4)],
        target_intrinsics={"fx": 1, "fy": 1, "cx": 1, "cy": 1},
        target_h=1,
        target_w=1,
        output_dir=tmp_path / "route-failed",
    )
    assert route["frames_synthesised"] == 0
    assert route["mean_coverage_frac"] == 0.0

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    synth._stitch_frames(tmp_path, tmp_path / "route.mp4")


def test_synthesize_cli_edges(monkeypatch, capsys, tmp_path):
    pose = json.dumps(np.eye(4).tolist())
    intrinsics = json.dumps({"fx": 1, "fy": 1, "cx": 1, "cy": 1})
    assert synth.main(["--site-id", "site", "--target-pose", pose, "--target-intrinsics", intrinsics, "--output", str(tmp_path / "out.jpg")]) == 1
    assert "bucket is required" in capsys.readouterr().err
    assert synth.main(["--site-id", "site", "--bucket", "bucket", "--target-pose", "[1,2]", "--target-intrinsics", intrinsics, "--output", str(tmp_path / "out.jpg")]) == 1
    assert "4x4 matrix" in capsys.readouterr().err
    assert synth.main(["--site-id", "site", "--bucket", "bucket", "--target-pose", "{", "--target-intrinsics", intrinsics, "--output", str(tmp_path / "out.jpg")]) == 1
    assert "Invalid --target-pose" in capsys.readouterr().err
    assert synth.main(["--site-id", "site", "--bucket", "bucket", "--target-pose", pose, "--target-intrinsics", "{", "--output", str(tmp_path / "out.jpg")]) == 1
    assert "Invalid --target-intrinsics" in capsys.readouterr().err
    monkeypatch.setattr(synth, "synthesize_view", lambda **_kwargs: {"status": "completed"})
    assert synth.main(["--site-id", "site", "--bucket", "bucket", "--target-pose", pose, "--target-intrinsics", intrinsics, "--output", str(tmp_path / "out.jpg"), "--no-fill-holes"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.setattr(synth, "synthesize_view", lambda **_kwargs: {"status": "failed"})
    assert synth.main(["--site-id", "site", "--bucket", "bucket", "--target-pose", pose, "--target-intrinsics", intrinsics, "--output", str(tmp_path / "out.jpg")]) == 1
