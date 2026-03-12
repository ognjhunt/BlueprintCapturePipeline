from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


def _load_script_module(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{name.replace('.', '_')}", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_reconstruction_job_spec_supports_video_only_neoverse(tmp_path: Path) -> None:
    module = _load_script_module("write_reconstruction_job_spec.py")
    input_video = tmp_path / "capture.mov"
    input_video.write_bytes(b"mov")

    payload = module.build_job_spec(
        Namespace(
            scene_id="scene_a",
            capture_id="capture_a",
            requested_backend="neoverse",
            input_video=str(input_video),
            raw_video_uri="",
            output_dir=str(tmp_path / "out"),
            compare_report_path=str(tmp_path / "compare.json"),
            arkit_poses_path="",
            arkit_intrinsics_path="",
            arkit_depth_dir="",
            arkit_confidence_dir="",
            scene_memory_conditioning_bundle_path="",
            advanced_geometry_bundle_path="",
        )
    )

    assert payload["requested_backend"] == "neoverse"
    assert payload["capture"]["raw_video_path"] == str(input_video)
    assert "arkit_poses_path" not in payload["capture"]


def test_write_reconstruction_job_spec_supports_gen3c_with_arkit(tmp_path: Path) -> None:
    module = _load_script_module("write_reconstruction_job_spec.py")
    input_video = tmp_path / "capture.mov"
    poses = tmp_path / "poses.jsonl"
    intrinsics = tmp_path / "intrinsics.json"
    depth_dir = tmp_path / "depth"
    input_video.write_bytes(b"mov")
    poses.write_text("{}\n", encoding="utf-8")
    intrinsics.write_text("{}", encoding="utf-8")
    depth_dir.mkdir()

    payload = module.build_job_spec(
        Namespace(
            scene_id="scene_a",
            capture_id="capture_a",
            requested_backend="gen3c",
            input_video=str(input_video),
            raw_video_uri="",
            output_dir=str(tmp_path / "out"),
            compare_report_path=str(tmp_path / "compare.json"),
            arkit_poses_path=str(poses),
            arkit_intrinsics_path=str(intrinsics),
            arkit_depth_dir=str(depth_dir),
            arkit_confidence_dir="",
            scene_memory_conditioning_bundle_path="",
            advanced_geometry_bundle_path="",
        )
    )

    assert payload["requested_backend"] == "gen3c"
    assert payload["capture"]["arkit_poses_path"] == str(poses)
    assert payload["capture"]["arkit_intrinsics_path"] == str(intrinsics)
    assert payload["capture"]["arkit_depth_dir"] == str(depth_dir)


def test_write_reconstruction_job_spec_rejects_incomplete_gen3c_conditioning(tmp_path: Path) -> None:
    module = _load_script_module("write_reconstruction_job_spec.py")
    input_video = tmp_path / "capture.mov"
    poses = tmp_path / "poses.jsonl"
    input_video.write_bytes(b"mov")
    poses.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        module.build_job_spec(
            Namespace(
                scene_id="scene_a",
                capture_id="capture_a",
                requested_backend="gen3c",
                input_video=str(input_video),
                raw_video_uri="",
                output_dir=str(tmp_path / "out"),
                compare_report_path=str(tmp_path / "compare.json"),
                arkit_poses_path=str(poses),
                arkit_intrinsics_path="",
                arkit_depth_dir="",
                arkit_confidence_dir="",
                scene_memory_conditioning_bundle_path="",
                advanced_geometry_bundle_path="",
            )
        )


def test_run_gen3c_service_fails_before_remote_submit_when_conditioning_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("run_gen3c_service.py")
    job_spec_path = tmp_path / "job_spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    job_spec_path.write_text(
        json.dumps(
            {
                "scene_id": "scene_a",
                "capture_id": "capture_a",
                "requested_backend": "gen3c",
                "capture": {"raw_video_path": str(tmp_path / "capture.mov")},
            }
        ),
        encoding="utf-8",
    )

    class _ExplodeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("client should not be instantiated")

    monkeypatch.setattr(module, "WorldModelServiceClient", _ExplodeClient)

    with pytest.raises(SystemExit, match="GEN3C requires poses \\+ intrinsics \\+ depth, or advanced geometry bundle"):
        module.main(
            [
                "--job-spec",
                str(job_spec_path),
                "--output-dir",
                str(output_dir),
                "--scene-id",
                "scene_a",
                "--capture-id",
                "capture_a",
            ]
        )


def test_run_neoverse_local_requires_local_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("run_neoverse_local.py")
    input_video = tmp_path / "capture.mov"
    input_video.write_bytes(b"mov")
    job_spec_path = tmp_path / "job_spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    job_spec_path.write_text(
        json.dumps(
            {
                "scene_id": "scene_a",
                "capture_id": "capture_a",
                "requested_backend": "neoverse",
                "capture": {"raw_video_path": str(input_video)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NEOVERSE_CMD_TEMPLATE", raising=False)
    monkeypatch.delenv("NEOVERSE_EXECUTABLE", raising=False)

    with pytest.raises(SystemExit, match="NeoVerse local runtime is not configured"):
        module.main(
            [
                "--job-spec",
                str(job_spec_path),
                "--output-dir",
                str(output_dir),
                "--scene-id",
                "scene_a",
                "--capture-id",
                "capture_a",
            ]
        )


def test_gen3c_contract_adapter_fails_when_required_artifacts_missing(tmp_path: Path) -> None:
    module = _load_script_module("gen3c_contract_adapter.py")
    backend_report = tmp_path / "gen3c_backend_report.json"
    backend_report.write_text("{}", encoding="utf-8")
    usdz = tmp_path / "export_last.usdz"
    mesh = tmp_path / "nvblox_mesh.ply"
    glb = tmp_path / "visual_mesh.glb"
    usdz.write_bytes(b"usdz")
    mesh.write_text("ply\n", encoding="utf-8")
    glb.write_bytes(b"glb")

    with pytest.raises(RuntimeError, match="missing normalized artifact 'occupancy.bin'"):
        module.normalize_backend_manifest(
            result_manifest={
                "normalized_contract": {
                    "export_last.usdz": str(usdz),
                    "nvblox_mesh.ply": str(mesh),
                    "visual_mesh.glb": str(glb),
                }
            },
            output_dir=tmp_path / "normalized",
            backend_report_path=backend_report,
        )
