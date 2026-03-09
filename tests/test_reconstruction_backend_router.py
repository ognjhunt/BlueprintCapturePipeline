from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_router_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "reconstruction_backend_router.py"
    spec = importlib.util.spec_from_file_location("reconstruction_backend_router_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_collision_mesh(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_backend_output(output_dir: Path, *, primary_visual: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_last.usdz").write_bytes(b"usdz")
    (output_dir / "occupancy.bin").write_bytes(b"occ")
    (output_dir / "visual_mesh.glb").write_bytes(b"glb")
    (output_dir / "object_point_cloud_index.json").write_text(json.dumps({"objects": []}), encoding="utf-8")
    (output_dir / "capture_quality_report.json").write_text(json.dumps({"frame_count": 4}), encoding="utf-8")
    _write_collision_mesh(output_dir / "nvblox_mesh.ply")
    (output_dir / "mesh_manifest.json").write_text(
        json.dumps({"primary_visual_asset": primary_visual}),
        encoding="utf-8",
    )


def test_normalize_backend_name_accepts_loger() -> None:
    module = _load_router_module()
    assert module._normalize_backend_name("loger") == module.BACKEND_LOGER


def test_build_loger_command_uses_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_router_module()
    monkeypatch.delenv("LOGER_CMD_TEMPLATE", raising=False)
    monkeypatch.setenv("LOGER_EXECUTABLE", "/tmp/loger_wrapper")

    command, use_shell = module._build_loger_command(
        input_video=tmp_path / "capture.mov",
        output_dir=tmp_path / "out",
        scene_id="scene_demo",
        capture_id="cap_demo",
        job_spec_path=tmp_path / "job.json",
    )

    assert use_shell is False
    assert command[:1] == ["/tmp/loger_wrapper"]
    assert "--input-video" in command
    assert "--job-spec" in command


def test_run_reconstruction_compare_mode_supports_loger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_router_module()

    def _fake_nurec(**kwargs):  # type: ignore[no-untyped-def]
        _write_backend_output(kwargs["output_dir"], primary_visual="export_last.usdz")
        return 0, "nurec", ""

    def _fake_loger(**kwargs):  # type: ignore[no-untyped-def]
        _write_backend_output(kwargs["output_dir"], primary_visual="visual_mesh.glb")
        (kwargs["output_dir"] / "loger_backend_report.json").write_text("{}", encoding="utf-8")
        return 0, "loger", ""

    monkeypatch.setattr(module, "_run_nurec_3dgrut", _fake_nurec)
    monkeypatch.setattr(module, "_run_loger", _fake_loger)

    report, winner = module.run_reconstruction(
        primary_backend=module.BACKEND_NUREC_3DGRUT,
        compare_backends=[module.BACKEND_LOGER],
        compare_winner=module.BACKEND_LOGER,
        job_spec_path=tmp_path / "job.json",
        input_video=tmp_path / "capture.mov",
        output_dir=tmp_path / "selected",
        scene_id="scene_demo",
        capture_id="cap_demo",
        backend_args=[],
        compare_report=tmp_path / "compare_report.json",
    )

    assert winner == module.BACKEND_LOGER
    assert report["selected_winner"] == module.BACKEND_LOGER
    assert module.BACKEND_LOGER in report["runs"]
    assert (tmp_path / "selected" / "visual_mesh.glb").is_file()
    assert (tmp_path / "selected" / "loger_backend_report.json").is_file()
