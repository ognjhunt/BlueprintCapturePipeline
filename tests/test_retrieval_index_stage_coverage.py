from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import retrieval_index_stage as ris
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.local_capture import LocalCaptureContext


def _ctx(tmp_path: Path) -> LocalCaptureContext:
    storage_root = tmp_path / "storage"
    bucket = "bucket"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = storage_root / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    pipeline_root = capture_root / "pipeline"
    raw_root.mkdir(parents=True, exist_ok=True)
    pipeline_root.mkdir(parents=True, exist_ok=True)
    return LocalCaptureContext(
        capture_root=capture_root,
        raw_root=raw_root,
        pipeline_root=pipeline_root,
        descriptor_path=capture_root / "capture_descriptor.json",
        raw_complete_path=raw_root / "capture_upload_complete.json",
        storage_root=storage_root,
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
    )


def _write_descriptor(ctx: LocalCaptureContext, payload: dict) -> None:
    ctx.descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.descriptor_path.write_text(json.dumps(payload), encoding="utf-8")


def _pose(frame_id: str, t: float, x: float = 0.0) -> dict:
    matrix = np.eye(4).tolist()
    matrix[0][3] = x
    return {"frame_id": frame_id, "frame_index": int(frame_id), "timestamp": t, "T_world_camera": matrix}


def test_retrieval_index_descriptor_video_selection_and_chunk_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(PipelineError, match="capture_descriptor"):
        ris._load_descriptor(ctx)

    descriptor = {
        "capture_source": "glasses",
        "capture_modality": "glasses_video",
        "raw_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
        "metadata": {
            "site_identity": {"site_id": "site-1"},
            "capture_mode": {"requested_mode": "site_world_candidate"},
            "capture_rights": {"derived_scene_generation_allowed": True},
            "nested_value": "from-meta",
        },
        "capture_bundle": {"bundle_value": "from-bundle"},
    }
    _write_descriptor(ctx, descriptor)
    assert ris._load_descriptor(ctx)["capture_source"] == "glasses"
    assert ris._descriptor_value(descriptor, "nested_value") == "from-meta"
    assert ris._descriptor_value(descriptor, "bundle_value") == "from-bundle"
    assert ris._descriptor_is_android_xr_video_only({"capture_profile_id": "android_xr_glasses"})
    assert ris._reference_media_indexable(descriptor) is True
    assert ris._reference_media_indexable({"capture_profile_id": "android_xr_glasses"}) is False
    assert ris._resolve_site_id({"site_id": "direct-site"}) == "direct-site"

    _write_descriptor(ctx, {"world_model_candidate": False, "metadata": {"site_identity": {"site_id": "site-1"}}})
    assert ris.run_retrieval_index_stage(capture_root=ctx.capture_root)["reason"] == "world_model_candidate=false"
    _write_descriptor(ctx, {"world_model_candidate": True, "metadata": {}})
    assert ris.run_retrieval_index_stage(capture_root=ctx.capture_root)["reason"] == "no_site_id"

    site_index = ctx.storage_root / ctx.bucket / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    site_index.parent.mkdir(parents=True, exist_ok=True)
    site_index.write_text("not-json\n" + json.dumps({"capture_id": ctx.capture_id}) + "\n", encoding="utf-8")
    _write_descriptor(ctx, {"world_model_candidate": True, "metadata": {"site_identity": {"site_id": "site-1"}}})
    assert ris.run_retrieval_index_stage(capture_root=ctx.capture_root)["reason"] == "already_indexed"

    assert ris._geometry_summary_reference_indexable({}) is False
    assert ris._geometry_summary_reference_indexable({"fallback_used": True, "geometry_source": "local_sfm"}) is False
    assert ris._geometry_summary_reference_indexable({"geometry_source": "unknown"}) is False
    assert ris._geometry_summary_reference_indexable({"geometry_source": "video_to_world", "geometry_live_ready": True}) is True
    assert ris._geometry_summary_reference_indexable(
        {
            "geometry_source": "local_sfm",
            "contract_ready_for_world_model": True,
            "intrinsics_available": True,
            "pose_track_count": 1,
        }
    ) is False
    ris._raise_if_geometry_not_reference_indexable({})
    with pytest.raises(PipelineError, match="fallback_geometry"):
        ris._raise_if_geometry_not_reference_indexable({"fallback_used": True, "geometry_source": "local_sfm"})
    with pytest.raises(PipelineError, match="geometry_not_live_video_to_world:local_sfm"):
        ris._raise_if_geometry_not_reference_indexable(
            {
                "geometry_source": "local_sfm",
                "contract_ready_for_world_model": True,
                "intrinsics_available": True,
                "pose_track_count": 1,
            }
        )

    privacy_video = ctx.capture_root / "privacy" / "final_walkthrough.mp4"
    privacy_video.parent.mkdir(parents=True, exist_ok=True)
    privacy_video.write_bytes(b"privacy")
    assert ris._resolve_video_source(ctx, {})["privacy_safe"] is True
    privacy_video.unlink()
    with pytest.raises(PipelineError, match="privacy_safe_video_required"):
        ris._resolve_video_source(ctx, {})
    monkeypatch.setenv("RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO", "0")
    raw_video = ctx.raw_root / "walkthrough.mov"
    raw_video.write_bytes(b"raw")
    raw_source = ris._resolve_video_source(ctx, {"raw_video_uri": "gs://missing-bucket/video.mov"})
    assert raw_source["source"] == "raw/walkthrough.mov"
    raw_video.unlink()
    with pytest.raises(PipelineError, match="No walkthrough video"):
        ris._resolve_video_source(ctx, {})

    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_text(
        json.dumps({"frameIndex": 7, "trackingState": "normal"}) + "\n"
        + json.dumps({"trackingState": "normal"}) + "\n",
        encoding="utf-8",
    )
    assert "000007" in ris._load_frames_quality_index(frames_path)
    assert ris._read_optional_json(tmp_path / "missing.json") == {}
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert ris._read_optional_json(invalid_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert ris._read_optional_json(list_json) == {}

    relocalized = ris._normalized_relocalization_events(
        {"relocalizationEvents": [None, {"startTCaptureSec": "bad"}, {"end_t_capture_sec": "2.5", "frameCount": 4}]}
    )
    assert relocalized == [{"start_t_capture_sec": None, "end_t_capture_sec": 2.5, "frame_count": 4}]
    assert ris._normalized_route_anchors({"routeAnchors": [None, {"anchorId": "entry", "anchorType": "door"}]}) == [
        {"anchor_id": "entry", "anchor_type": "door"}
    ]
    checkpoints = ris._normalized_checkpoint_events(
        {"checkpointEvents": [None, {"anchorId": "entry", "passId": "p1", "tCaptureSec": 1.25, "completed": True}]}
    )
    assert checkpoints[0]["t_capture_sec"] == 1.25

    selected = [{"frame_id": "000001", "t_capture_sec": None}, {"frame_id": "000002", "t_capture_sec": 2.0}]
    ris._attach_anchor_to_nearest_selected(selected=selected, anchor_id="far", t_capture_sec=20.0, max_delta_sec=0.1)
    assert "anchor_observations" not in selected[-1]
    ris._attach_anchor_to_nearest_selected(selected=selected, anchor_id="near", t_capture_sec=2.1)
    assert selected[-1]["anchor_observations"] == ["near"]
    assert ris._anchor_ids("not-a-list") == []
    assert ris._anchor_ids([{"anchorId": "a"}, "a", "b", ""]) == ["a", "b"]

    assert ris._world_mapping_confidence("limited_tracking") == 0.65
    assert ris._world_mapping_confidence("odd") == 0.5
    assert ris._world_mapping_confidence(None) == 0.75
    confidence_entry = {
        "_fq": {"poseConfidence": "bad"},
        "quality": {"sharpness_score": "bad", "world_mapping_status": "mapped"},
    }
    assert ris._capture_confidence(confidence_entry) > 0
    assert ris._staticness_score(
        entry={**confidence_entry, "retrieval_signals": {"route_anchor_density": 3.0}},
        geometry_fingerprint={"valid_fraction": 0.8, "plane_support_ratio": 0.6},
    ) > 0

    route_file = ctx.raw_root / "route_anchors.json"
    route_file.write_text(json.dumps({"routeAnchors": [{"anchorId": "entry"}]}), encoding="utf-8")
    checkpoint_file = ctx.raw_root / "checkpoint_events.json"
    checkpoint_file.write_text(
        json.dumps({"checkpointEvents": [{"anchorId": "entry", "tCaptureSec": 1.0, "completed": False}]}),
        encoding="utf-8",
    )
    selected_for_anchors = [{"frame_id": "000001", "t_capture_sec": 1.0, "zone_id": None, "quality": {}, "_fq": {}}]
    ris._apply_route_anchor_observations(
        selected=selected_for_anchors,
        ctx=ctx,
        descriptor={"metadata": {"site_identity": {"zone_id": "zone-a"}, "capture_topology": {"entry_anchor_id": "entry"}}},
    )
    assert selected_for_anchors[0]["zone_id"] == "zone-a"
    assert selected_for_anchors[0]["anchor_observations"] == ["entry"]
    assert selected_for_anchors[0]["retrieval_signals"]["anchor_observation_count"] == 1
    ris._apply_route_anchor_observations(selected=[], ctx=ctx, descriptor={})

    chunk_entries = [
        {"t_capture_sec": 0.0, "T_world_camera": np.eye(4).tolist()},
        {"t_capture_sec": 2.0, "T_world_camera": np.eye(4).tolist()},
        {"t_capture_sec": 2.1, "T_world_camera": [[1, 0, 0, 3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        {"t_capture_sec": 2.2, "T_world_camera": [[1, 0, 0, 3.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
    ]
    ris._assign_chunk_ids(selected=chunk_entries, relocalization_events=[{"start_t_capture_sec": 2.15}])
    assert [entry["chunk_boundary_reason"] for entry in chunk_entries] == [
        "capture_start",
        "temporal_gap",
        "spatial_jump",
        "relocalization_boundary",
    ]
    ris._assign_chunk_ids(selected=[], relocalization_events=[])

    assert ris._parse_frame_intrinsics({"cameraIntrinsics": {"fx": "bad"}}) is None
    assert ris._parse_frame_intrinsics({"intrinsics": [1, 0, 2, 0, 3, 0, 4, 5, 1], "imageResolution": [640, 480]}) == {
        "fx": 1.0,
        "fy": 3.0,
        "cx": 4.0,
        "cy": 5.0,
        "width": 640,
        "height": 480,
    }

    assert ris._select_frames(poses=[], frames_quality={}) == []
    selected_frames = ris._select_frames(
        poses=[
            {"timestamp": 0.0},
            {"frameIndex": 1, "timestamp": 0.0, "T_world_camera": None},
            _pose("000002", 0.0, 0.0),
            _pose("000003", 0.6, 0.01),
            _pose("000004", 1.2, 0.02),
            _pose("000005", 1.8, 0.03),
            _pose("000006", 2.4, 0.04),
        ],
        frames_quality={
            "000002": {"trackingState": "limited"},
            "000003": {"relocalizationEvent": True},
            "000004": {"sharpnessScore": 10.0},
            "000005": {"trackingState": "normal", "sharpnessScore": 80.0, "anchorObservations": ["a"]},
            "000006": {"trackingState": "normal", "sharpnessScore": 80.0},
        },
    )
    assert [frame["frame_id"] for frame in selected_frames] == ["000005"]


def test_retrieval_index_dense_record_and_media_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(tmp_path)
    source_jpg = tmp_path / "source.jpg"
    Image.new("RGB", (4, 4), color=(20, 30, 40)).save(source_jpg)
    source_npy = tmp_path / "source.npy"
    np.save(source_npy, np.full((2, 2), 128, dtype=np.float32))
    bad_npy = tmp_path / "bad.npy"
    bad_npy.write_text("not npy", encoding="utf-8")

    assert ris._materialize_reference_frame(
        frame_meta={"source_image_path": str(source_jpg)},
        video_path=tmp_path / "video.mov",
        frame_number=1,
        output_path=tmp_path / "copy.jpg",
    )
    assert ris._materialize_reference_frame(
        frame_meta={"source_image_path": str(source_npy)},
        video_path=tmp_path / "video.mov",
        frame_number=1,
        output_path=tmp_path / "npy.jpg",
    )
    assert ris._materialize_reference_frame(
        frame_meta={"source_image_path": str(bad_npy)},
        video_path=tmp_path / "video.mov",
        frame_number=1,
        output_path=tmp_path / "placeholder.jpg",
    )

    bad_png = tmp_path / "bad.png"
    bad_png.write_text("not image", encoding="utf-8")
    assert ris._materialize_reference_frame(
        frame_meta={"source_image_path": str(bad_png)},
        video_path=tmp_path / "video.mov",
        frame_number=1,
        output_path=tmp_path / "bad-out.jpg",
    ) is False

    def ffmpeg_success(cmd, **_kwargs):
        Path(cmd[-1]).write_bytes(b"frame")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ris.subprocess, "run", ffmpeg_success)
    assert ris._ffmpeg_extract_frame(video_path=tmp_path / "video.mov", frame_number=1, output_path=tmp_path / "ffmpeg.jpg")
    monkeypatch.setattr(ris.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    assert ris._ffmpeg_extract_frame(video_path=tmp_path / "video.mov", frame_number=1, output_path=tmp_path / "ffmpeg-fail.jpg") is False

    assert ris._artifact_uri_from_path(None, ctx) is None
    assert ris._artifact_uri_from_path(tmp_path / "missing.depth", ctx) is None
    depth = ctx.capture_root / "pipeline" / "depth.npy"
    depth.parent.mkdir(parents=True, exist_ok=True)
    depth.write_bytes(b"depth")
    assert ris._artifact_uri_from_path(depth, ctx) == "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/depth.npy"
    assert ris._local_to_gs_uri(tmp_path / "outside", ctx) is None
    assert ris._arkit_depth_uri("000001", ctx) is None
    arkit_depth = ctx.raw_root / "arkit" / "depth" / "000001.png"
    arkit_depth.parent.mkdir(parents=True)
    arkit_depth.write_bytes(b"depth")
    assert ris._arkit_depth_uri("000001", ctx)

    class Encoder:
        def encode(self, paths):
            return [np.ones(4, dtype=np.float32) for _ in paths]

    assert len(ris._generate_embeddings(model=Encoder(), image_paths=[source_jpg])) == 1
    assert len(ris._generate_embeddings(model=lambda paths: [np.zeros(4, dtype=np.float32) for _ in paths], image_paths=[source_jpg])) == 1
    emb_path = tmp_path / "embeddings" / "000001.bin"
    ris._save_embedding(np.ones(4, dtype=np.float32), emb_path)
    assert emb_path.is_file()

    selected = [
        {
            "frame_id": "000001",
            "frame_index": 1,
            "t_capture_sec": 0.0,
            "T_world_camera": np.eye(4).tolist(),
            "quality": {"sharpness_score": 80.0},
            "anchor_observations": [],
            "zone_id": None,
            "chunk_id": "chunk_000",
            "chunk_order": 0,
            "chunk_boundary_reason": "capture_start",
            "_fq": {"image_path": str(tmp_path / "missing.jpg"), "intrinsics": [1, 0, 2, 0, 3, 0, 4, 5, 1]},
        },
        {
            "frame_id": "000002",
            "frame_index": 2,
            "t_capture_sec": 1.0,
            "T_world_camera": np.eye(4).tolist(),
            "quality": {"sharpness_score": 90.0},
            "anchor_observations": ["a"],
            "retrieval_signals": {"route_anchor_density": 1.0},
            "zone_id": "zone-a",
            "chunk_id": "chunk_000",
            "chunk_order": 0,
            "chunk_boundary_reason": None,
            "_fq": {"image_path": str(source_jpg), "depth_path": str(depth), "confidence_path": str(depth), "cameraIntrinsics": {"fx": 1, "fy": 2, "cx": 3, "cy": 4}},
        },
    ]
    monkeypatch.setattr(ris, "_generate_embeddings", lambda **_kwargs: [])
    records = ris._build_dense_records(
        selected=selected,
        frames_quality={},
        video_path=tmp_path / "video.mov",
        export_dir=tmp_path / "export",
        model=object(),
        ctx=ctx,
        privacy_source="privacy/final_walkthrough.mov",
        geometry_source="local_sfm",
    )
    assert records[0]["exclude_reason"] == "ffmpeg_failed"
    assert records[1]["exclude_reason"] == "embedding_failed"

    selected_embedding_error = [
        {
            "frame_id": "000002",
            "frame_index": 2,
            "t_capture_sec": 1.0,
            "T_world_camera": np.eye(4).tolist(),
            "quality": {"sharpness_score": 90.0},
            "anchor_observations": [],
            "zone_id": "zone-a",
            "chunk_id": "chunk_000",
            "chunk_order": 0,
            "_fq": {"image_path": str(source_jpg)},
        }
    ]
    monkeypatch.setattr(
        ris,
        "_generate_embeddings",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("not an image")),
    )
    error_records = ris._build_dense_records(
        selected=selected_embedding_error,
        frames_quality={},
        video_path=tmp_path / "video.mov",
        export_dir=tmp_path / "export-error",
        model=object(),
        ctx=ctx,
        privacy_source="privacy/final_walkthrough.mov",
        geometry_source="local_sfm",
    )
    assert error_records[0]["exclude_reason"] == "embedding_failed"

    selected_ok = [
        {
            "frame_id": "000003",
            "frame_index": 3,
            "t_capture_sec": 2.0,
            "T_world_camera": np.eye(4).tolist(),
            "quality": {"sharpness_score": 95.0},
            "anchor_observations": [],
            "zone_id": None,
            "chunk_id": "chunk_001",
            "chunk_order": 1,
            "chunk_boundary_reason": "temporal_gap",
            "_fq": {"image_path": str(source_jpg)},
        }
    ]
    monkeypatch.setattr(ris, "_generate_embeddings", lambda **_kwargs: [np.ones(4, dtype=np.float32)])
    ok_records = ris._build_dense_records(
        selected=selected_ok,
        frames_quality={},
        video_path=tmp_path / "video.mov",
        export_dir=tmp_path / "export-ok",
        model=object(),
        ctx=ctx,
        privacy_source="privacy/final_walkthrough.mov",
        geometry_source="video_to_world",
    )
    assert ok_records[0]["included_in_index"] is True

    selected_existing = [
        {
            "frame_id": "000003",
            "frame_index": 3,
            "t_capture_sec": 2.0,
            "T_world_camera": np.eye(4).tolist(),
            "quality": {"sharpness_score": 95.0},
            "anchor_observations": [],
            "zone_id": None,
            "chunk_id": "chunk_001",
            "chunk_order": 1,
            "chunk_boundary_reason": "temporal_gap",
            "_fq": {},
        }
    ]
    existing_records = ris._build_dense_records(
        selected=selected_existing,
        frames_quality={},
        video_path=tmp_path / "video.mov",
        export_dir=tmp_path / "export-ok",
        model=object(),
        ctx=ctx,
        privacy_source="privacy/final_walkthrough.mov",
        geometry_source="video_to_world",
    )
    assert existing_records[0]["included_in_index"] is True

    ris._write_dense_index(tmp_path / "dense" / "dense_index.jsonl", ok_records)
    ris._write_dense_export_manifest(export_dir=tmp_path / "dense", ctx=ctx, geometry_source="video_to_world", dense_records=ok_records)
    ris._write_pose_alignment_summary(
        export_dir=tmp_path / "dense",
        descriptor={},
        ctx=ctx,
        dense_records=[*ok_records, {"included_in_index": False, "exclude_reason": "privacy_filtered"}],
        coordinate_frame_session_id="session-1",
    )
    assert (tmp_path / "dense" / "dense_export_manifest.json").is_file()

    frames_dir = tmp_path / "thumb-frames"
    thumbs = tmp_path / "thumbs"
    frames_dir.mkdir()
    (frames_dir / "000003.jpg").write_bytes(b"jpg")
    ris._write_thumbnails(
        frames_dir=frames_dir,
        thumbnails_dir=thumbs,
        records=[{}, {"reference_id": "ref-1"}, {"reference_id": "ref-2", "frame_id": "missing"}, {"reference_id": "ref-3", "frame_id": "000003"}],
        ctx=ctx,
    )
    assert thumbs.is_dir()


def test_retrieval_index_stage_site_reference_and_validation_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(tmp_path)
    descriptor = {
        "world_model_candidate": True,
        "captured_at": "2026-01-01T00:00:00Z",
        "metadata": {
            "site_identity": {"site_id": "site-1"},
            "capture_topology": {"captureSessionId": "session-1", "passId": "pass-1", "passIndex": 1},
            "rights_lineage": {"derived_generation_allowed": True},
            "provenance_lineage": {"capture_id": ctx.capture_id},
        },
    }
    _write_descriptor(ctx, descriptor)
    privacy_video = ctx.capture_root / "privacy" / "final_walkthrough.mov"
    privacy_video.parent.mkdir(parents=True, exist_ok=True)
    privacy_video.write_bytes(b"privacy")

    original_ensure_geometry_for_capture = ris._ensure_geometry_for_capture
    monkeypatch.setattr(ris, "_ensure_geometry_for_capture", lambda **kwargs: kwargs["descriptor"])
    monkeypatch.setattr(ris, "load_capture_geometry", lambda **_kwargs: {"frame_meta": {}, "poses": [], "source": "local_sfm"})
    with pytest.raises(PipelineError, match="No geometry poses"):
        ris.run_retrieval_index_stage(capture_root=ctx.capture_root, force_rebuild=True, embedding_model=object())

    monkeypatch.setattr(ris, "_ensure_geometry_for_capture", original_ensure_geometry_for_capture)
    assert ris._ensure_geometry_for_capture(ctx=ctx, descriptor={"capture_modality": "iphone_arkit_lidar"})["capture_modality"] == "iphone_arkit_lidar"

    geometry_summary_path = ctx.pipeline_root / "geometry" / "geometry_summary.json"

    def fake_build_geometry(capture_root: Path, **_kwargs):
        geometry_summary_path.parent.mkdir(parents=True, exist_ok=True)
        geometry_summary_path.write_text(
            json.dumps(
                {
                    "geometry_source": "video_to_world",
                    "fallback_used": False,
                    "provider_native_result": True,
                    "ready_for_world_model": True,
                    "geometry_live_ready": True,
                }
            ),
            encoding="utf-8",
        )
        ctx.descriptor_path.write_text(json.dumps({"geometry_ready": True, "metadata": {"site_identity": {"site_id": "site-1"}}}), encoding="utf-8")

    monkeypatch.setattr(ris, "build_geometry_stage_contract", fake_build_geometry)
    assert ris._ensure_geometry_for_capture(ctx=ctx, descriptor={})["geometry_ready"] is True
    assert ris._capture_already_indexed(ctx.capture_root / "missing.jsonl", ctx.capture_id) is False
    no_match = tmp_path / "no_match.jsonl"
    no_match.write_text("not-json\n" + json.dumps({"capture_id": "other"}) + "\n", encoding="utf-8")
    assert ris._capture_already_indexed(no_match, ctx.capture_id) is False

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

    class FakeParam:
        is_cuda = False

    class FakeVector:
        def cpu(self):
            return self

        def numpy(self):
            return np.ones(4, dtype=np.float32)

    class FakeHidden:
        def __getitem__(self, key):
            if isinstance(key, tuple):
                return self
            return FakeVector()

    class FakeModel:
        def parameters(self):
            yield FakeParam()

        def __call__(self, **_inputs):
            return SimpleNamespace(last_hidden_state=FakeHidden())

    class FakeProcessor:
        def __call__(self, **_kwargs):
            return {"pixels": SimpleNamespace(cuda=lambda: "cuda")}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    embedding_image = tmp_path / "embedding.jpg"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(embedding_image)
    tuple_embeddings = ris._generate_embeddings(model=(FakeModel(), FakeProcessor()), image_paths=[embedding_image])
    assert len(tuple_embeddings) == 1

    site_root = ctx.storage_root / ctx.bucket / "sites" / "site-1" / "reference_memory"
    site_index_path = site_root / "site_reference_index.jsonl"
    records = [
        {
            "reference_id": "ref-1",
            "frame_id": "000001",
            "frame_index": 1,
            "t_capture_sec": 0.0,
            "T_world_camera": np.eye(4).tolist(),
            "intrinsics": {"fx": 1, "fy": 1, "cx": 0.5, "cy": 0.5},
            "depth_uri": "gs://bucket/depth/1.png",
            "confidence_uri": "gs://bucket/conf/1.png",
            "embedding_uri": "gs://bucket/emb/1.bin",
            "frame_uri": "gs://bucket/frame/1.jpg",
            "thumbnail_uri": "gs://bucket/thumb/1.jpg",
            "privacy_source": "privacy/final_walkthrough.mov",
            "geometry_source": "video_to_world",
            "quality": {"sharpness_score": 80.0},
            "anchor_observations": ["entry"],
            "retrieval_signals": {"capture_confidence": 0.8},
            "staticness_score": 0.9,
            "geometry_fingerprint": {"available": True, "valid_fraction": 1.0, "plane_support_ratio": 0.5},
            "visibility_cells": ["0,0"],
            "zone_id": "zone-a",
            "chunk_id": "chunk-a",
            "chunk_order": 0,
        },
        {
            "reference_id": "ref-2",
            "frame_id": "000002",
            "frame_index": 2,
            "t_capture_sec": 1.0,
            "T_world_camera": [[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]],
            "intrinsics": {"fx": 1, "fy": 1, "cx": 0.5, "cy": 0.5},
            "depth_uri": "gs://bucket/depth/2.png",
            "confidence_uri": "gs://bucket/conf/2.png",
            "frame_uri": "gs://bucket/frame/2.jpg",
            "privacy_source": "privacy/final_walkthrough.mov",
            "geometry_source": "video_to_world",
            "quality": {"sharpness_score": 100.0},
            "anchor_observations": ["entry"],
            "retrieval_signals": {"capture_confidence": 0.9},
            "staticness_score": 0.8,
            "geometry_fingerprint": {"available": True, "valid_fraction": 0.8, "plane_support_ratio": 0.5},
            "visibility_cells": ["0,0"],
            "zone_id": "zone-a",
            "chunk_id": "chunk-b",
            "chunk_order": 1,
        },
    ]
    ris._append_to_site_reference_index(
        site_index_path=site_index_path,
        records=records,
        descriptor=descriptor,
        ctx=ctx,
        site_id="site-1",
    )
    assert site_index_path.is_file()

    empty_index = site_root / "empty.jsonl"
    empty_index.write_text("", encoding="utf-8")
    ris._update_coverage_map(site_root=site_root, site_index_path=empty_index, site_id="site-1")
    ris._write_site_memory_indices(site_root=site_root, site_index_path=empty_index, site_id="site-1", storage_root=ctx.storage_root)
    ris._write_overlap_graph(site_root=site_root, site_index_path=empty_index, site_id="site-1", storage_root=ctx.storage_root)
    ris._write_retrieval_validation(site_root=site_root, site_index_path=empty_index, site_id="site-1")

    no_pose_index = site_root / "no_pose.jsonl"
    no_pose_index.write_text(json.dumps({"capture_id": "cap", "chunk_id": "chunk", "T_world_camera": [1, 2, 3]}) + "\n", encoding="utf-8")
    ris._update_coverage_map(site_root=site_root, site_index_path=no_pose_index, site_id="site-1")

    bad_coverage = site_root / "coverage" / "coverage_map.json"
    bad_coverage.parent.mkdir(parents=True, exist_ok=True)
    bad_coverage.write_text("{", encoding="utf-8")
    ris._write_site_manifest(site_root=site_root, site_index_path=no_pose_index, site_id="site-1")
    degraded_manifest = json.loads((site_root / "site_reference_manifest.json").read_text(encoding="utf-8"))
    assert degraded_manifest["readiness"]["state"] == "degraded"

    ris._update_coverage_map(site_root=site_root, site_index_path=site_index_path, site_id="site-1")
    ris._write_site_manifest(site_root=site_root, site_index_path=site_index_path, site_id="site-1")
    ris._write_site_reference_summary_projection(site_root=site_root, site_index_path=site_index_path, site_id="site-1", storage_root=ctx.storage_root)
    ris._write_site_memory_indices(site_root=site_root, site_index_path=site_index_path, site_id="site-1", storage_root=ctx.storage_root)
    ris._write_overlap_graph(site_root=site_root, site_index_path=site_index_path, site_id="site-1", storage_root=ctx.storage_root)
    ris._write_retrieval_validation(site_root=site_root, site_index_path=site_index_path, site_id="site-1")
    assert (site_root / "indices" / "zone_index.json").is_file()
    assert (site_root / "site_overlap_graph.json").is_file()
    assert (site_root / "retrieval_validation.json").is_file()

    assert ris._site_reference_path_to_gs_uri(tmp_path / "outside.json", storage_root=ctx.storage_root).endswith("outside.json")
    assert ris._site_reference_path_to_gs_uri(ctx.storage_root / ctx.bucket, storage_root=ctx.storage_root) is None
    assert ris._site_reference_record_schema_errors([{"bad": "record"}])
    assert ris._site_reference_manifest_schema_error({"bad": "manifest"})
    monkeypatch.setattr(ris, "assert_summary_projection_safe", lambda _payload: (_ for _ in ()).throw(ValueError("unsafe")))
    assert ris._summary_projection_is_safe(
        site_root=tmp_path / "too-shallow",
        site_index_path=site_index_path,
        site_id="site-1",
        manifest_payload={"bad": "manifest"},
    ) is False
    assert ris._retrieval_query_count(site_index_path=site_index_path, records=[{"T_world_camera": [1, 2, 3]}], site_root=site_root) == 0
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.retrieval_query.query_site",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("query failed")),
    )
    loaded_records = [json.loads(line) for line in site_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert ris._retrieval_query_count(site_index_path=site_index_path, records=loaded_records, site_root=site_root) == 0

    assert ris._mat_tx({}) == 0.0
    assert ris._mat_ty({}) == 0.0
    assert ris._mat_tz({}) == 0.0
    assert ris._euclidean((1, 2, 3), None) == float("inf")
    assert ris._p95([1.0, 2.0, 3.0]) >= 1.0
    assert ris._arkit_confidence_uri("000001", ctx) is None
    confidence = ctx.raw_root / "arkit" / "confidence" / "000001.png"
    confidence.parent.mkdir(parents=True, exist_ok=True)
    confidence.write_bytes(b"confidence")
    assert ris._arkit_confidence_uri("000001", ctx)


def test_retrieval_index_stage_remaining_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(tmp_path)
    descriptor = {"world_model_candidate": True, "metadata": {"site_identity": {"site_id": "site-1"}}}
    _write_descriptor(ctx, descriptor)
    privacy_video = ctx.capture_root / "privacy" / "final_walkthrough.mov"
    privacy_video.parent.mkdir(parents=True, exist_ok=True)
    privacy_video.write_bytes(b"privacy")

    export_dir = ctx.capture_root / "world_model_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    dense_index = export_dir / "dense_index.jsonl"
    reusable = {
        "frame_id": "000001",
        "frame_index": 1,
        "t_capture_sec": 0.0,
        "T_world_camera": np.eye(4).tolist(),
        "quality": {},
        "anchor_observations": [],
        "chunk_id": "chunk_000",
        "geometry_fingerprint": {},
        "included_in_index": True,
    }
    dense_index.write_text(json.dumps(reusable) + "\n", encoding="utf-8")
    patched_stage_helpers = (
        "_write_thumbnails",
        "_append_to_site_reference_index",
        "_update_coverage_map",
        "_write_site_manifest",
        "_write_site_reference_summary_projection",
        "_write_site_memory_indices",
        "_write_overlap_graph",
        "_write_retrieval_validation",
    )
    original_stage_helpers = {name: getattr(ris, name) for name in patched_stage_helpers}
    for name in patched_stage_helpers:
        monkeypatch.setattr(ris, name, lambda **_kwargs: None)
    reused = ris.run_retrieval_index_stage(capture_root=ctx.capture_root, embedding_model=object())
    assert reused["status"] == "completed"

    dense_index.write_text(json.dumps({"frame_id": "old-schema"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(ris, "_ensure_geometry_for_capture", lambda **kwargs: kwargs["descriptor"])
    monkeypatch.setattr(ris, "load_capture_geometry", lambda **_kwargs: {"frame_meta": {}, "poses": [], "source": "local_sfm"})
    with pytest.raises(PipelineError, match="No geometry poses"):
        ris.run_retrieval_index_stage(capture_root=ctx.capture_root, embedding_model=object())
    for name, helper in original_stage_helpers.items():
        monkeypatch.setattr(ris, name, helper)

    relocalized = ris._normalized_relocalization_events(
        {"relocalization_events": [{"start_t_capture_sec": "1.0", "end_t_capture_sec": "bad"}]}
    )
    assert relocalized == [{"start_t_capture_sec": 1.0, "end_t_capture_sec": None, "frame_count": 0}]

    selected = [
        {"frame_id": "a", "t_capture_sec": 1.0, "anchor_observations": ["entry"], "quality": {}, "_fq": {}},
        {"frame_id": "b", "t_capture_sec": None, "anchor_observations": ["entry"], "quality": {}, "_fq": {}},
    ]
    ris._annotate_retrieval_signals(
        selected=selected,
        route_anchors=[{"anchor_id": "entry"}],
        checkpoint_events=[{"completed": True, "t_capture_sec": 1.5}],
    )
    assert selected[0]["retrieval_signals"]["checkpoint_proximity_sec"] == 0.5

    selected_for_topology = [{"frame_id": "000001", "t_capture_sec": 1.0, "zone_id": None, "quality": {}, "_fq": {}}]
    ris._apply_route_anchor_observations(
        selected=selected_for_topology,
        ctx=ctx,
        descriptor={"metadata": {"capture_topology": ["not-dict"]}},
    )
    assert selected_for_topology[0]["retrieval_signals"]
    ris._apply_route_anchor_observations(
        selected=selected_for_topology,
        ctx=ctx,
        descriptor={"metadata": {"capture_topology": {"entry_anchor_id": "entry", "entry_anchor_t_capture_sec": 1.0}}},
    )
    assert "entry" in selected_for_topology[0]["anchor_observations"]

    entries = [
        {"t_capture_sec": 0.0, "T_world_camera": np.eye(4).tolist()},
        {"t_capture_sec": 0.1, "T_world_camera": np.eye(4).tolist()},
    ]
    ris._assign_chunk_ids(selected=entries, relocalization_events=[{}])
    assert entries[0]["chunk_boundary_reason"] == "capture_start"

    assert ris._parse_frame_intrinsics({"cameraIntrinsics": {"fx": "bad", "fy": 1, "cx": 1, "cy": 1}}) is None
    assert ris._parse_frame_intrinsics({"intrinsics_payload": {"fx": "bad", "fy": 1, "cx": 1, "cy": 1}}) is None
    selected_frames = ris._select_frames(
        poses=[_pose("000001", 0.0, 0.0), _pose("000002", 0.1, 0.001)],
        frames_quality={"000001": {"trackingState": "normal"}, "000002": {"trackingState": "normal"}},
    )
    assert [frame["frame_id"] for frame in selected_frames] == ["000001"]

    source_png = tmp_path / "source.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(source_png)
    assert ris._materialize_reference_frame(
        frame_meta={"source_image_path": str(source_png)},
        video_path=tmp_path / "video.mov",
        frame_number=1,
        output_path=tmp_path / "from-png.jpg",
    )

    original_resolve = Path.resolve

    def raise_resolve(self: Path, *args, **kwargs):
        if self.name == "raises.depth":
            raise OSError("resolve failed")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(ris.Path, "resolve", raise_resolve)
    assert ris._artifact_uri_from_path(tmp_path / "raises.depth", ctx) is None
    monkeypatch.setattr(ris.Path, "resolve", original_resolve)

    import builtins

    original_import = builtins.__import__

    def fail_transformers_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_transformers_import)
    with pytest.raises(PipelineError, match="Failed to load DINOv3"):
        ris._load_dinov3()
    monkeypatch.setattr(builtins, "__import__", original_import)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeLoadTorch:
        cuda = FakeCuda()

    class FakeLoadedModel:
        def eval(self):
            self.evaluated = True

        def cuda(self):
            self.on_cuda = True
            return self

    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return {"processor_for": model_id, "kwargs": kwargs}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            assert kwargs == {
                "revision": ris._DINOV3_MODEL_REVISION,
                "trust_remote_code": False,
            }
            return FakeLoadedModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
    fake_transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", FakeLoadTorch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    loaded_model, loaded_processor = ris._load_dinov3()
    assert loaded_model.on_cuda is True
    assert loaded_processor["processor_for"] == ris._DINOV3_MODEL_ID
    assert loaded_processor["kwargs"] == {
        "revision": ris._DINOV3_MODEL_REVISION,
        "trust_remote_code": False,
    }

    def fail_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)
    with pytest.raises(PipelineError, match="Missing embedding dependency"):
        ris._generate_embeddings(model=(object(), object()), image_paths=[])
    monkeypatch.setattr(builtins, "__import__", original_import)

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

    class FakeParam:
        is_cuda = True

    class FakeInput:
        def cuda(self):
            return self

    class FakeVector:
        def cpu(self):
            return self

        def numpy(self):
            return np.ones(4, dtype=np.float32)

    class FakeHidden:
        def __getitem__(self, key):
            if isinstance(key, tuple):
                return self
            return FakeVector()

    class FakeModel:
        def parameters(self):
            yield FakeParam()

        def __call__(self, **_inputs):
            return SimpleNamespace(last_hidden_state=FakeHidden())

    class FakeProcessor:
        def __call__(self, **_kwargs):
            return {"pixels": FakeInput()}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    image_path = tmp_path / "cuda-image.jpg"
    Image.new("RGB", (2, 2)).save(image_path)
    assert len(ris._generate_embeddings(model=(FakeModel(), FakeProcessor()), image_paths=[image_path])) == 1

    frames_dir = tmp_path / "thumb-frames"
    thumbs = tmp_path / "thumbs"
    frames_dir.mkdir()
    thumbs.mkdir()
    (frames_dir / "000001.jpg").write_bytes(b"frame")
    (thumbs / "ref-existing.jpg").write_bytes(b"thumb")
    ris._write_thumbnails(
        frames_dir=frames_dir,
        thumbnails_dir=thumbs,
        records=[{"reference_id": "ref-existing", "frame_id": "000001"}],
        ctx=ctx,
    )
    monkeypatch.setattr(ris.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("ffmpeg missing")))
    ris._write_thumbnails(
        frames_dir=frames_dir,
        thumbnails_dir=thumbs,
        records=[{"reference_id": "ref-new", "frame_id": "000001"}],
        ctx=ctx,
    )

    site_root = ctx.storage_root / ctx.bucket / "sites" / "site-1" / "reference_memory"
    site_index_path = site_root / "site_reference_index.jsonl"
    site_index_path.parent.mkdir(parents=True, exist_ok=True)
    site_index_path.write_text("", encoding="utf-8")
    ris._write_site_manifest(site_root=site_root, site_index_path=site_index_path, site_id="site-1")
    assert json.loads((site_root / "site_reference_manifest.json").read_text(encoding="utf-8"))["readiness"]["state"] == "degraded"

    overlap_index = site_root / "overlap.jsonl"
    overlap_records = [
        {"chunk_id": "a", "capture_id": "cap", "coordinate_frame_session_id": "s", "zone_id": "z"},
        {"chunk_id": "b", "capture_id": "cap", "coordinate_frame_session_id": "s", "zone_id": "z"},
    ]
    overlap_index.write_text("\n".join(json.dumps(row) for row in overlap_records) + "\n", encoding="utf-8")
    summaries = {
        "a": {"zone_id": "z", "anchor_ids": [], "record_count": 1, "staticness_score": 0.9, "embedding_centroid": np.ones(2), "geometry_fingerprint": {}},
        "b": {"zone_id": "z", "anchor_ids": [], "record_count": 1, "staticness_score": 0.9, "embedding_centroid": np.ones(2), "geometry_fingerprint": {}},
    }
    monkeypatch.setattr(ris, "aggregate_chunk_summary", lambda chunk_records, storage_root: summaries[str(chunk_records[0]["chunk_id"])])
    ris._write_overlap_graph(site_root=site_root, site_index_path=overlap_index, site_id="site-1", storage_root=ctx.storage_root)
    graph = json.loads((site_root / "site_overlap_graph.json").read_text(encoding="utf-8"))
    assert graph["edges"]

    low_summaries = {
        "a": {"zone_id": "", "anchor_ids": [], "record_count": 1, "staticness_score": 0.9, "geometry_fingerprint": {}},
        "b": {"zone_id": "", "anchor_ids": [], "record_count": 1, "staticness_score": 0.9, "geometry_fingerprint": {}},
    }
    monkeypatch.setattr(ris, "aggregate_chunk_summary", lambda chunk_records, storage_root: low_summaries[str(chunk_records[0]["chunk_id"])])
    ris._write_overlap_graph(site_root=site_root, site_index_path=overlap_index, site_id="site-1", storage_root=ctx.storage_root)

    bad_validation_index = site_root / "bad_validation.jsonl"
    bad_validation_index.write_text(json.dumps({"capture_id": "cap", "privacy_source": "raw_video"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(ris, "_summary_projection_is_safe", lambda **_kwargs: False)
    ris._write_retrieval_validation(site_root=site_root, site_index_path=bad_validation_index, site_id="site-1")
    validation = json.loads((site_root / "retrieval_validation.json").read_text(encoding="utf-8"))
    assert set(validation["runtime_adapter_consumption"]["blockers"]) >= {
        "local_contract_invalid",
        "retrieval_query_not_ready",
        "privacy_safe_source_missing",
        "rights_lineage_missing",
        "provenance_lineage_missing",
    }
