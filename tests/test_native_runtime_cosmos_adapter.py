from __future__ import annotations

import shutil

from blueprint_pipeline.native_runtime_cosmos_adapter import LegacyCosmosRuntimeAdapter


def _adapter(tmp_path, *, environment=None) -> LegacyCosmosRuntimeAdapter:
    return LegacyCosmosRuntimeAdapter(
        storage_root=tmp_path,
        load_site_world=lambda _site_world_id: {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
        },
        process_runner=lambda *_args, **_kwargs: None,
        copy_file=shutil.copy2,
        environment=environment or {},
    )


def test_cosmos_adapter_keeps_prebuilt_discovery_capture_scoped(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    video = (
        tmp_path
        / "vast-local"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
        / "pipeline"
        / "cosmos_single_capture_smoke"
        / "renders"
        / "video_bootstrap_0000.mp4"
    )
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    assert adapter.find_prebuilt_video("site-world-1") == video


def test_cosmos_adapter_discovers_conditioning_frame_and_adapter(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    pipeline_root = (
        tmp_path
        / "vast-local"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
        / "pipeline"
    )
    conditioning = (
        pipeline_root
        / "cosmos_single_capture_smoke"
        / "video_bootstrap_frames"
        / "frame_0000.jpg"
    )
    checkpoint = (
        pipeline_root
        / "cosmos_training_export"
        / "checkpoints"
        / "adapter_model.safetensors"
    )
    conditioning.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    conditioning.write_bytes(b"frame")
    checkpoint.write_bytes(b"checkpoint")

    assert adapter.find_conditioning_frame("site-world-1") == conditioning
    assert adapter.find_lora_adapter("site-world-1") == checkpoint


def test_explicit_cosmos_adapter_does_not_fall_back_when_missing(tmp_path) -> None:
    adapter = _adapter(
        tmp_path,
        environment={"COSMOS_LORA_CHECKPOINT_PATH": str(tmp_path / "missing")},
    )
    capture_checkpoint = (
        tmp_path
        / "vast-local"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
        / "pipeline"
        / "cosmos_training_export"
        / "checkpoints"
        / "adapter_model.safetensors"
    )
    capture_checkpoint.parent.mkdir(parents=True)
    capture_checkpoint.write_bytes(b"checkpoint")

    assert adapter.find_lora_adapter("site-world-1") is None
