from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from blueprint_pipeline.nurec_worker_client import NurecWorkerClient, NurecWorkerConfig


def _load_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "loger_contract_adapter.py"
    spec = importlib.util.spec_from_file_location("loger_contract_adapter_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_point_cloud(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 255 0 0",
                "1 0 0 0 255 0",
                "0 1 0 0 0 255",
                "0 0 1 255 255 255",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_loger_adapter_writes_contract_artifacts(monkeypatch, tmp_path: Path) -> None:
    module = _load_adapter_module()
    native_output_dir = tmp_path / "native"
    output_dir = tmp_path / "out"
    native_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_point_cloud(native_output_dir / "point_cloud.ply")
    input_video = tmp_path / "capture.mov"
    input_video.write_bytes(b"video")

    job_spec_path = tmp_path / "nurec_job_spec.json"
    job_spec_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene_demo",
                "capture_id": "cap_demo",
                "capture": {
                    "arkit_poses_uri": "gs://bucket/poses.json",
                    "arkit_intrinsics_uri": "gs://bucket/intrinsics.json",
                },
            }
        ),
        encoding="utf-8",
    )

    def _fake_sam3_detect(*, output_dir: Path, input_video: Path, gaussian_ply: Path) -> Path:
        del input_video, gaussian_ply
        output_path = output_dir / "object_point_cloud_index.json"
        output_path.write_text(json.dumps({"objects": []}), encoding="utf-8")
        return output_path

    monkeypatch.setattr(module, "_run_sam3_detect", _fake_sam3_detect)
    monkeypatch.setattr(
        module,
        "_build_capture_quality_report_for_video",
        lambda _video, _work_dir: {
            "schema_version": "v1",
            "frame_count": 4,
            "blur": {"count": 4},
            "brightness": {"count": 4},
            "motion": {"count": 4},
            "frame_extraction": {"effective_extract_fps": 1.0},
        },
    )
    monkeypatch.setattr(module, "_generate_occupancy_from_ply", lambda _ply, out: out.write_bytes(b"occ"))

    module.adapt_loger_outputs(
        native_output_dir=native_output_dir,
        output_dir=output_dir,
        input_video=input_video,
        job_spec_path=job_spec_path,
        scene_id="scene_demo",
        capture_id="cap_demo",
        native_runtime_sec=12.5,
    )

    assert (output_dir / "export_last.usdz").is_file()
    assert (output_dir / "export_last.ply").is_file()
    assert (output_dir / "visual_pointcloud.ply").is_file()
    assert (output_dir / "nvblox_mesh.ply").is_file()
    assert (output_dir / "visual_mesh.glb").is_file()
    assert (output_dir / "occupancy.bin").is_file()
    assert (output_dir / "object_point_cloud_index.json").is_file()
    assert (output_dir / "mesh_method.txt").read_text(encoding="utf-8").strip() == "loger_poisson"

    manifest = json.loads((output_dir / "mesh_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "loger"
    assert manifest["primary_visual_asset"] == "visual_mesh.glb"

    quality = json.loads((output_dir / "capture_quality_report.json").read_text(encoding="utf-8"))
    assert quality["sfm"]["status"] == "not_applicable"
    assert quality["loger"]["runtime_sec"] == 12.5

    backend_report = json.loads((output_dir / "loger_backend_report.json").read_text(encoding="utf-8"))
    assert backend_report["arkit_available_but_unused"]["poses"] is True
    assert backend_report["arkit_available_but_unused"]["intrinsics"] is True

    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_demo/captures/cap_demo/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )
    nurec_dir = client.nurec_dir
    nurec_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "export_last.usdz",
        "export_last.ply",
        "visual_pointcloud.ply",
        "nvblox_mesh.ply",
        "visual_mesh.glb",
        "occupancy.bin",
        "object_point_cloud_index.json",
        "mesh_manifest.json",
        "capture_quality_report.json",
        "loger_backend_report.json",
    ):
        (nurec_dir / name).write_bytes((output_dir / name).read_bytes())

    payload = client.collect_outputs()
    artifacts = payload["artifacts"]
    assert "visual_usdz" in artifacts
    assert "visual_mesh_glb" in artifacts
    assert "mesh_manifest_json" in artifacts
    assert "visual_pointcloud_ply" in artifacts
