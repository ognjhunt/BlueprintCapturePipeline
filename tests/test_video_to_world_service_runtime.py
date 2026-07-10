from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import video_to_world_service_runtime as runtime


def test_video_to_world_runtime_env_templates_and_file_helpers(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("GCS_ROOT", raising=False)
    assert runtime._gcs_root() == Path("/mnt/gcs")
    monkeypatch.setenv("GCS_ROOT", str(tmp_path / "gcs"))
    assert runtime._gcs_root() == tmp_path / "gcs"
    assert runtime._string(None) == ""
    assert runtime._string(" value ") == "value"

    monkeypatch.setenv("VIDEO_TO_WORLD_COMMAND_TIMEOUT_SECONDS", "bad")
    assert runtime._timeout_seconds() == 7200
    monkeypatch.setenv("VIDEO_TO_WORLD_COMMAND_TIMEOUT_SECONDS", "10")
    assert runtime._timeout_seconds() == 60
    monkeypatch.setenv("VIDEO_TO_WORLD_COMMAND_TIMEOUT_SECONDS", "90")
    assert runtime._timeout_seconds() == 90

    monkeypatch.setenv("VIDEO_TO_WORLD_REPO_DIR", "/repo")
    assert runtime._repo_dir() == "/repo"
    monkeypatch.setenv("VIDEO_TO_WORLD_PIPELINE_PRESET", "FULL_FAST")
    assert runtime._command_preset() == "full_fast"
    for preset in (
        "preprocess_only",
        "preprocess_plus_alignment",
        "full_fast",
        "full_extensive",
        "unknown",
    ):
        assert "{INPUT_VIDEO}" in runtime._preset_template(preset)
    monkeypatch.setenv("VIDEO_TO_WORLD_COMMAND_TEMPLATE", "custom {INPUT_VIDEO}")
    assert runtime._command_template() == "custom {INPUT_VIDEO}"
    monkeypatch.delenv("VIDEO_TO_WORLD_COMMAND_TEMPLATE")
    assert "run_reconstruction.py" in runtime._command_template()

    hinted = tmp_path / "hinted.mp4"
    hinted.write_bytes(b"video")
    assert runtime._materialize_file("unused", str(hinted), tmp_path) == hinted
    local = tmp_path / "local.mp4"
    local.write_bytes(b"local")
    assert runtime._materialize_file(str(local), "", tmp_path) == local
    gcs_file = tmp_path / "gcs" / "bucket" / "video.mp4"
    gcs_file.parent.mkdir(parents=True)
    gcs_file.write_bytes(b"gcs")
    assert runtime._materialize_file("gs://bucket/video.mp4", "", tmp_path).read_bytes() == b"gcs"
    with pytest.raises(FileNotFoundError, match="missing_input_video"):
        runtime._materialize_file("", "", tmp_path)
    with pytest.raises(FileNotFoundError, match="input_not_found"):
        runtime._materialize_file(str(tmp_path / "missing.mp4"), "", tmp_path)

    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    assert runtime._copy_to_uri(source, "") == ""
    local_dest = tmp_path / "out" / "copy.bin"
    assert runtime._copy_to_uri(source, str(local_dest)) == str(local_dest)
    assert local_dest.read_bytes() == b"payload"
    assert runtime._copy_to_uri(source, "gs://bucket/copy.bin") == "gs://bucket/copy.bin"
    assert (tmp_path / "gcs" / "bucket" / "copy.bin").read_bytes() == b"payload"

    calls: dict[str, object] = {}

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, "out", "err")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    completed = runtime._run_template(
        "echo {INPUT_VIDEO} {SCENE_ROOT}",
        {"INPUT_VIDEO": "in.mp4", "SCENE_ROOT": "scene"},
        12,
    )
    assert completed.stdout == "out"
    assert calls["argv"] == ["echo", "in.mp4", "scene"]
    assert 0 < calls["kwargs"]["timeout"] <= 12
    assert calls["kwargs"]["shell"] is False

    invalid_command = runtime._run_template('"unterminated', {}, 12)
    assert invalid_command.returncode == 64
    assert invalid_command.stderr == "video_to_world_command_template_invalid_or_empty"

    npy_path = Path(runtime._write_npy(tmp_path / "arrays" / "value.npy", np.asarray([1, 2])))
    assert np.load(npy_path).tolist() == [1, 2]
    world_from_camera = runtime._matrix44_from_w2c(np.eye(3, 4))
    assert world_from_camera.shape == (4, 4)


def test_video_to_world_runtime_normalizes_npz_outputs(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    geometry_root = tmp_path / "geometry"
    with pytest.raises(FileNotFoundError, match="video_to_world_results_missing"):
        runtime._normalize_npz_outputs(scene_root=scene_root, geometry_root=geometry_root)

    npz_path = scene_root / "exports" / "npz" / "results.npz"
    npz_path.parent.mkdir(parents=True)
    np.savez(
        npz_path,
        depth=np.asarray([[[1.0, 2.0], [3.0, 4.0]]]),
        conf=np.asarray([[[0.1, 0.9], [0.2, 0.8]]]),
        extrinsics=np.asarray([np.eye(3, 4)]),
        intrinsics=np.asarray([[[100.0, 0.0, 1.0], [0.0, 120.0, 1.5], [0.0, 0.0, 1.0]]]),
        image=np.asarray([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]),
    )
    pointcloud = (
        scene_root
        / "frame_to_model_icp_50_2_offset0"
        / "after_global_optimization"
        / "aligned_points.ply"
    )
    pointcloud.parent.mkdir(parents=True)
    pointcloud.write_text("ply", encoding="utf-8")

    payload = runtime._normalize_npz_outputs(scene_root=scene_root, geometry_root=geometry_root)

    assert payload["status"] == "succeeded"
    assert payload["intrinsics"]["image_width"] == 2
    assert payload["intrinsics"]["fx"] == 100.0
    assert payload["canonical_pointcloud_source_path"] == str(pointcloud)
    frame = payload["frames"][0]
    assert Path(frame["image_path"]).is_file()
    assert Path(frame["depth_path"]).is_file()
    assert Path(frame["confidence_path"]).is_file()
    assert frame["min_depth_m"] == 1.0
    assert frame["confidence_range"] == [0.1, 0.9]


def test_execute_video_to_world_request_fail_closed_and_success_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")
    geometry_root = tmp_path / "geometry"
    base_body = {
        "input_video_path": str(input_video),
        "geometry_root_path": str(geometry_root),
        "dynamic_mask_manifest_path": "masks.json",
    }
    monkeypatch.setenv("VIDEO_TO_WORLD_PIPELINE_PRESET", "preprocess_only")

    monkeypatch.setattr(runtime, "_command_template", lambda: "")
    assert runtime.execute_video_to_world_request(base_body) == {
        "status": "failed",
        "reason": "video_to_world_command_not_configured",
    }

    def fake_invalid_json(_template, substitutions, _timeout):  # type: ignore[no-untyped-def]
        Path(substitutions["RESULT_JSON"]).write_text("{bad-json", encoding="utf-8")
        return subprocess.CompletedProcess("cmd", 0, "stdout-value", "stderr-value")

    monkeypatch.setattr(runtime, "_command_template", lambda: "cmd {RESULT_JSON}")
    monkeypatch.setattr(runtime, "_run_template", fake_invalid_json)
    invalid = runtime.execute_video_to_world_request(base_body)
    assert invalid["status"] == "failed"
    assert invalid["reason"] == "video_to_world_result_invalid_json"
    assert invalid["stdout"] == "stdout-value"

    def fake_non_dict(_template, substitutions, _timeout):  # type: ignore[no-untyped-def]
        Path(substitutions["RESULT_JSON"]).write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess("cmd", 0, "", "")

    monkeypatch.setattr(runtime, "_run_template", fake_non_dict)
    assert runtime.execute_video_to_world_request(base_body) == {
        "status": "failed",
        "reason": "video_to_world_result_invalid_payload",
    }

    def fake_failed_without_result(_template, _substitutions, _timeout):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess("cmd", 7, "short stdout", "short stderr")

    monkeypatch.setattr(runtime, "_run_template", fake_failed_without_result)
    failed = runtime.execute_video_to_world_request(base_body)
    assert failed == {
        "status": "failed",
        "reason": "video_to_world_command_failed:7",
        "stdout": "short stdout",
        "stderr": "short stderr",
    }

    def fake_normalized(_template, _substitutions, _timeout):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess("cmd", 0, "", "")

    monkeypatch.setattr(runtime, "_run_template", fake_normalized)
    monkeypatch.setattr(
        runtime,
        "_normalize_npz_outputs",
        lambda *, scene_root, geometry_root: {  # type: ignore[no-untyped-def]
            "status": "succeeded",
            "provider_metrics": {"backend": "fake"},
        },
    )
    normalized = runtime.execute_video_to_world_request(base_body)
    assert normalized["status"] == "succeeded"
    assert normalized["provider_metrics"]["backend"] == "fake"
    assert normalized["provider_metrics"]["runner"] == "video_to_world"
    assert normalized["provider_metrics"]["command_preset"] == "preprocess_only"

    def fake_success_json(_template, substitutions, _timeout):  # type: ignore[no-untyped-def]
        Path(substitutions["RESULT_JSON"]).write_text(
            json.dumps({"status": "succeeded", "provider_metrics": {}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess("cmd", 0, "", "")

    monkeypatch.setattr(runtime, "_run_template", fake_success_json)
    success = runtime.execute_video_to_world_request(base_body)
    assert success["provider_metrics"]["repo_dir"] == runtime._repo_dir()
