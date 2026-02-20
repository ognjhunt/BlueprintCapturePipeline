"""Tests for nurec_shim Fixer backend routing."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest


def _load_nurec_shim_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "nurec_shim.py"
    spec = importlib.util.spec_from_file_location("nurec_shim_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixer_auto_prefers_h100(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    output_dir = tmp_path / "out"
    fixed_dir = output_dir / "fixer_output"
    renders_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    call_count = {"local": 0}

    def _fake_h100(*args, **kwargs):  # type: ignore[no-untyped-def]
        fixed_dir.mkdir(parents=True, exist_ok=True)
        (fixed_dir / "frame_00001.png").write_bytes(b"ok")
        return True

    def _fake_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["local"] += 1
        return False

    monkeypatch.setattr(module, "_run_fixer_h100_stage", _fake_h100)
    monkeypatch.setattr(module, "_run_fixer_local_stage", _fake_local)

    result = module.run_fixer_refinement(
        renders_dir,
        output_dir,
        mode="auto",
        h100_script=tmp_path / "dummy.sh",
    )

    assert result == fixed_dir
    assert call_count["local"] == 0


def test_colmap_cuda_detection_true(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()

    class _FakeResult:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "COLMAP 3.10 with CUDA"
            self.stderr = ""

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeResult()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    assert module._colmap_has_cuda() is True


def test_colmap_cuda_detection_false(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()

    class _FakeResult:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "COLMAP 3.7 without CUDA"
            self.stderr = ""

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeResult()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    assert module._colmap_has_cuda() is False


def test_colmap_cuda_detection_legacy_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()

    class _FakeResult:
        def __init__(self, stdout: str = "", stderr: str = "") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        if cmd == ["colmap", "version"]:
            return _FakeResult(stderr="ERROR: Command `version` not recognized.")
        if cmd == ["colmap", "help"]:
            return _FakeResult(stdout="COLMAP 3.7 (Commit abc with CUDA)")
        return _FakeResult(stdout="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    assert module._colmap_has_cuda() is True


def test_fixer_h100_mode_no_fallback_when_h100_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    output_dir = tmp_path / "out"
    renders_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    call_count = {"local": 0}

    def _fake_h100(*args, **kwargs):  # type: ignore[no-untyped-def]
        return False

    def _fake_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["local"] += 1
        return True

    monkeypatch.setattr(module, "_run_fixer_h100_stage", _fake_h100)
    monkeypatch.setattr(module, "_run_fixer_local_stage", _fake_local)

    result = module.run_fixer_refinement(
        renders_dir,
        output_dir,
        mode="h100",
        h100_script=tmp_path / "dummy.sh",
    )

    assert result == renders_dir
    assert call_count["local"] == 0


def test_fixer_auto_falls_back_to_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    output_dir = tmp_path / "out"
    fixed_dir = output_dir / "fixer_output"
    renders_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    def _fake_h100(*args, **kwargs):  # type: ignore[no-untyped-def]
        return False

    def _fake_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        fixed_dir.mkdir(parents=True, exist_ok=True)
        (fixed_dir / "frame_00001.png").write_bytes(b"ok")
        return True

    monkeypatch.setattr(module, "_run_fixer_h100_stage", _fake_h100)
    monkeypatch.setattr(module, "_run_fixer_local_stage", _fake_local)

    result = module.run_fixer_refinement(
        renders_dir,
        output_dir,
        mode="auto",
        h100_script=tmp_path / "dummy.sh",
    )

    assert result == fixed_dir


def test_run_fixer_local_stage_uses_updated_cli_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    fixed_dir = tmp_path / "fixed"
    renders_dir.mkdir(parents=True, exist_ok=True)

    fixer_dir = tmp_path / "Fixer"
    fixer_src = fixer_dir / "src"
    fixer_src.mkdir(parents=True, exist_ok=True)
    inference_script = fixer_src / "inference_pretrained_model.py"
    inference_script.write_text("# test", encoding="utf-8")

    weights_dir = tmp_path / "weights"
    pretrained_path = weights_dir / "pretrained" / "pretrained_fixer.pkl"
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)
    pretrained_path.write_bytes(b"ok")
    base_dit = weights_dir / "base" / "model_fast_tokenizer.pt"
    base_dit.parent.mkdir(parents=True, exist_ok=True)
    base_dit.write_bytes(b"ok")
    base_vae = weights_dir / "base" / "tokenizer_fast.pth"
    base_vae.write_bytes(b"ok")

    monkeypatch.setattr(module, "FIXER_DIR", str(fixer_dir))
    monkeypatch.setattr(module, "FIXER_WEIGHTS_DIR", str(weights_dir))
    monkeypatch.setenv("FIXER_TIMESTEP", "321")
    monkeypatch.setenv("FIXER_RESOLUTION", "576")

    observed: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        observed["cmd"] = cmd
        observed["cwd"] = kwargs.get("cwd")
        fixed_dir.mkdir(parents=True, exist_ok=True)
        (fixed_dir / "frame_00001.png").write_bytes(b"ok")
        return None

    monkeypatch.setattr(module, "_run", _fake_run)

    assert module._run_fixer_local_stage(renders_dir, fixed_dir) is True
    assert observed["cwd"] == str(fixer_src)
    assert observed["cmd"] == [
        "python3",
        str(inference_script),
        "--model",
        str(pretrained_path),
        "--input",
        str(renders_dir),
        "--output",
        str(fixed_dir),
        "--timestep",
        "321",
        "--resolution",
        "576",
    ]


def test_run_fixer_local_stage_skips_when_base_models_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    fixed_dir = tmp_path / "fixed"
    renders_dir.mkdir(parents=True, exist_ok=True)

    fixer_dir = tmp_path / "Fixer"
    fixer_src = fixer_dir / "src"
    fixer_src.mkdir(parents=True, exist_ok=True)
    (fixer_src / "inference_pretrained_model.py").write_text("# test", encoding="utf-8")

    weights_dir = tmp_path / "weights"
    pretrained_path = weights_dir / "pretrained" / "pretrained_fixer.pkl"
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)
    pretrained_path.write_bytes(b"ok")

    monkeypatch.setattr(module, "FIXER_DIR", str(fixer_dir))
    monkeypatch.setattr(module, "FIXER_WEIGHTS_DIR", str(weights_dir))

    assert module._run_fixer_local_stage(renders_dir, fixed_dir) is False


def test_run_colmap_sfm_uses_new_gpu_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    commands: list[list[str]] = []

    def _fake_supports(subcommand: str, option_name: str) -> bool:
        if subcommand == "feature_extractor" and option_name == "--FeatureExtraction.use_gpu":
            return True
        if subcommand == "sequential_matcher" and option_name == "--FeatureMatching.use_gpu":
            return True
        return False

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(cmd)
        if cmd[1] == "mapper":
            (workspace / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(module, "_colmap_supports_option", _fake_supports)
    monkeypatch.setattr(module, "_run", _fake_run)

    module.run_colmap_sfm(frames_dir, workspace, sift_use_gpu=True)

    assert "--FeatureExtraction.use_gpu" in commands[0]
    assert "--FeatureMatching.use_gpu" in commands[1]


def test_run_colmap_sfm_uses_legacy_gpu_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    commands: list[list[str]] = []

    def _fake_supports(subcommand: str, option_name: str) -> bool:
        return False

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(cmd)
        if cmd[1] == "mapper":
            (workspace / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(module, "_colmap_supports_option", _fake_supports)
    monkeypatch.setattr(module, "_run", _fake_run)

    module.run_colmap_sfm(frames_dir, workspace, sift_use_gpu=True)

    assert "--SiftExtraction.use_gpu" in commands[0]
    assert "--SiftMatching.use_gpu" in commands[1]


def test_run_colmap_sfm_selects_reconstruction_with_most_registered_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if cmd[1] == "mapper":
            sparse = workspace / "sparse"
            sparse.mkdir(parents=True, exist_ok=True)
            for name, count in (("0", 4), ("1", 134), ("2", 79)):
                recon = sparse / name
                recon.mkdir(parents=True, exist_ok=True)
                (recon / "images.bin").write_bytes(struct.pack("<Q", count))
        return None

    monkeypatch.setattr(module, "_run", _fake_run)
    monkeypatch.setattr(module, "_colmap_supports_option", lambda *_args, **_kwargs: True)

    best = module.run_colmap_sfm(frames_dir, workspace, sift_use_gpu=True)
    assert best == workspace / "sparse" / "1"


def test_run_colmap_sfm_supports_exhaustive_matcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    commands: list[list[str]] = []

    def _fake_supports(subcommand: str, option_name: str) -> bool:
        if subcommand in {"feature_extractor", "exhaustive_matcher"} and option_name in {
            "--FeatureExtraction.use_gpu",
            "--FeatureMatching.use_gpu",
        }:
            return True
        return False

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(cmd)
        if cmd[1] == "mapper":
            sparse = workspace / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_bytes(struct.pack("<Q", 42))
        return None

    monkeypatch.setattr(module, "_colmap_supports_option", _fake_supports)
    monkeypatch.setattr(module, "_run", _fake_run)

    module.run_colmap_sfm(
        frames_dir,
        workspace,
        sift_use_gpu=True,
        matcher_mode="exhaustive",
    )

    assert commands[1][1] == "exhaustive_matcher"
    assert "--FeatureMatching.use_gpu" in commands[1]


def test_run_colmap_sfm_sequential_overlap_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    commands: list[list[str]] = []

    def _fake_supports(subcommand: str, option_name: str) -> bool:
        return True

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(cmd)
        if cmd[1] == "mapper":
            sparse = workspace / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_bytes(struct.pack("<Q", 42))
        return None

    monkeypatch.setattr(module, "_colmap_supports_option", _fake_supports)
    monkeypatch.setattr(module, "_run", _fake_run)

    module.run_colmap_sfm(
        frames_dir,
        workspace,
        sift_use_gpu=True,
        matcher_mode="sequential",
        sequential_overlap=40,
    )

    seq_cmd = commands[1]
    assert seq_cmd[1] == "sequential_matcher"
    assert "--SequentialMatching.overlap" in seq_cmd
    idx = seq_cmd.index("--SequentialMatching.overlap")
    assert seq_cmd[idx + 1] == "40"


def test_run_colmap_sfm_unknown_matcher_falls_back_to_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    commands: list[list[str]] = []

    def _fake_supports(subcommand: str, option_name: str) -> bool:
        return False

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(cmd)
        if cmd[1] == "mapper":
            (workspace / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(module, "_colmap_supports_option", _fake_supports)
    monkeypatch.setattr(module, "_run", _fake_run)

    module.run_colmap_sfm(frames_dir, workspace, sift_use_gpu=True, matcher_mode="bogus")
    assert commands[1][1] == "sequential_matcher"


def test_resolve_effective_max_frames_scales_for_long_videos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES", "true")
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES_TARGET_FPS", "0.5")
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES_HARD_CAP", "1200")

    max_frames, reason = module._resolve_effective_max_frames(1800.0, 450)
    assert max_frames == 900
    assert "adaptive_max_frames=enabled" in reason


def test_resolve_effective_extract_fps_covers_long_video_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_EXTRACT_FPS", "true")
    fps, reason = module._resolve_effective_extract_fps(1800.0, 6, 900)
    assert fps == pytest.approx(0.5, rel=1e-3)
    assert "adaptive_extract_fps=enabled" in reason


def test_resolve_colmap_matcher_mode_auto_switches_for_large_frame_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("COLMAP_AUTO_EXHAUSTIVE_MAX_FRAMES", "420")

    short_mode, _ = module._resolve_colmap_matcher_mode("auto", 300)
    long_mode, _ = module._resolve_colmap_matcher_mode("auto", 900)
    assert short_mode == "exhaustive"
    assert long_mode == "sequential"


def test_resolve_chunked_sfm_enabled_auto_threshold() -> None:
    module = _load_nurec_shim_module()
    enabled_short, _ = module._resolve_chunked_sfm_enabled("auto", 850, 900)
    enabled_long, _ = module._resolve_chunked_sfm_enabled("auto", 1200, 900)
    assert enabled_short is False
    assert enabled_long is True


def test_build_colmap_chunk_ranges_caps_count_and_covers_tail() -> None:
    module = _load_nurec_shim_module()
    ranges = module._build_colmap_chunk_ranges(
        12000,
        chunk_size=600,
        chunk_overlap=120,
        max_chunks=24,
    )
    assert len(ranges) <= 24
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 12000
    for start, end in ranges:
        assert end > start


def test_resolve_colmap_retry_matcher_mode_auto_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("COLMAP_AUTO_EXHAUSTIVE_MAX_FRAMES", "600")
    short_mode, _ = module._resolve_colmap_retry_matcher_mode("auto", 500)
    long_mode, _ = module._resolve_colmap_retry_matcher_mode("auto", 1400)
    assert short_mode == "exhaustive"
    assert long_mode == "sequential"


def test_robust_occupancy_grid_limits_outlier_impact() -> None:
    module = _load_nurec_shim_module()
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(7)
    core = rng.normal(loc=0.0, scale=1.2, size=(5000, 3)).astype(np.float32)
    outliers = np.array(
        [
            [1200.0, 1100.0, 1300.0],
            [-1300.0, -1400.0, -1250.0],
        ],
        dtype=np.float32,
    )
    xyz = np.vstack([core, outliers])

    grid, center, voxel_size, stats = module._build_robust_occupancy_grid(xyz, 64)
    assert grid.shape == (64, 64, 64)
    assert stats["kept_points"] < stats["total_points"]
    assert abs(float(center[0])) < 1.0
    assert abs(float(center[1])) < 1.0
    assert abs(float(center[2])) < 1.0
    assert float(voxel_size) < 0.2


def test_robust_occupancy_grid_degenerate_points_has_positive_voxel_size() -> None:
    module = _load_nurec_shim_module()
    np = pytest.importorskip("numpy")

    xyz = np.full((256, 3), 5.0, dtype=np.float32)
    grid, center, voxel_size, stats = module._build_robust_occupancy_grid(xyz, 64)
    assert grid.shape == (64, 64, 64)
    assert stats["kept_points"] == stats["total_points"] == 256
    assert tuple(float(v) for v in center) == (5.0, 5.0, 5.0)
    assert float(voxel_size) > 0.0


def test_resolve_sam3_settings_warehouse_auto() -> None:
    module = _load_nurec_shim_module()

    n_frames, min_frames = module._resolve_sam3_settings(
        environment="warehouse",
        frame_count=263,
        requested_n_frames=0,
        requested_min_frame_detections=0,
    )

    assert n_frames == 26
    assert min_frames == 2


def test_resolve_sam3_settings_manual_override() -> None:
    module = _load_nurec_shim_module()

    n_frames, min_frames = module._resolve_sam3_settings(
        environment="warehouse",
        frame_count=263,
        requested_n_frames=14,
        requested_min_frame_detections=3,
    )

    assert n_frames == 14
    assert min_frames == 3


def test_postprocess_collision_mesh_removes_long_edge_spikes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    pytest.importorskip("trimesh")
    pytest.importorskip("numpy")

    mesh_path = tmp_path / "spiky_mesh.ply"
    mesh_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "element face 3",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "100 0 0",
                "3 0 1 2",
                "3 0 2 3",
                "3 0 1 4",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("COLLISION_MAX_EDGE_M", "5.0")
    report = module._postprocess_collision_mesh(mesh_path)
    assert report.get("enabled") is True
    steps = report.get("steps", [])
    assert any(step.get("name") == "spike_face_filter" for step in steps)
    after = report.get("after", {})
    assert int(after.get("face_count", 0)) > 0
    assert int(after.get("face_count", 0)) < 3


def test_build_visual_mesh_artifacts_falls_back_from_gaussian_tsdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    fused_ply = tmp_path / "fused.ply"
    gaussian_ply = tmp_path / "gaussian.ply"
    fused_ply.write_bytes(b"ply")
    gaussian_ply.write_bytes(b"ply")

    def _fake_robust(**_kwargs):  # type: ignore[no-untyped-def]
        return {"ok": False, "method": "gaussian_tsdf", "reason": "not_available"}

    def _fake_quick(*, fused_ply: Path, output_glb: Path, target_faces: int):  # type: ignore[no-untyped-def]
        output_glb.write_bytes(b"glb")
        return {
            "ok": True,
            "method": "quick_poisson_open3d",
            "target_faces": target_faces,
            "path": str(output_glb),
        }

    monkeypatch.setattr(module, "_build_visual_mesh_gaussian_tsdf", _fake_robust)
    monkeypatch.setattr(module, "_build_visual_mesh_quick", _fake_quick)
    monkeypatch.setenv("VISUAL_MESH_METHOD", "gaussian_tsdf")
    monkeypatch.setenv("VISUAL_MESH_ENABLED", "true")

    report = module.build_visual_mesh_artifacts(
        output_dir=tmp_path,
        fused_ply=fused_ply,
        gaussian_ply=gaussian_ply,
    )
    assert report["status"] == "ok"
    assert report["selected_method"] == "quick_poisson_open3d"
    assert (tmp_path / "visual_mesh.glb").is_file()
    assert (tmp_path / "visual_pointcloud.ply").is_file()


def test_build_visual_mesh_artifacts_textured_fallback_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    fused_ply = tmp_path / "fused.ply"
    gaussian_ply = tmp_path / "gaussian.ply"
    fused_ply.write_bytes(b"ply")
    gaussian_ply.write_bytes(b"ply")

    def _fake_textured(**_kwargs):  # type: ignore[no-untyped-def]
        return {"ok": False, "method": "textured_colmap", "reason": "texrecon_unavailable"}

    def _fake_robust(**_kwargs):  # type: ignore[no-untyped-def]
        return {"ok": False, "method": "gaussian_tsdf", "reason": "gaussian_unavailable"}

    def _fake_quick(*, fused_ply: Path, output_glb: Path, target_faces: int):  # type: ignore[no-untyped-def]
        output_glb.write_bytes(b"glb")
        return {
            "ok": True,
            "method": "quick_poisson_open3d",
            "textured": False,
            "texture_image_count": 0,
            "atlas_resolution": 0,
            "uv_coverage_ratio": 0.0,
            "path": str(output_glb),
            "target_faces": target_faces,
        }

    monkeypatch.setattr(module, "_build_visual_mesh_textured_colmap", _fake_textured)
    monkeypatch.setattr(module, "_build_visual_mesh_gaussian_tsdf", _fake_robust)
    monkeypatch.setattr(module, "_build_visual_mesh_quick", _fake_quick)
    monkeypatch.setenv("VISUAL_MESH_METHOD", "textured_colmap")
    monkeypatch.setenv("VISUAL_MESH_ENABLED", "true")

    report = module.build_visual_mesh_artifacts(
        output_dir=tmp_path,
        fused_ply=fused_ply,
        gaussian_ply=gaussian_ply,
        workspace=tmp_path,
    )
    assert report["status"] == "ok"
    assert report["selected_method"] == "quick_poisson_open3d"
    assert "texrecon_unavailable" in report.get("fallback_reason", "")
    assert "gaussian_unavailable" in report.get("fallback_reason", "")
    assert report["textured"] is False


def test_build_visual_mesh_artifacts_textured_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    fused_ply = tmp_path / "fused.ply"
    gaussian_ply = tmp_path / "gaussian.ply"
    fused_ply.write_bytes(b"ply")
    gaussian_ply.write_bytes(b"ply")

    def _fake_textured(**_kwargs):  # type: ignore[no-untyped-def]
        (tmp_path / "visual_mesh.glb").write_bytes(b"glb")
        return {
            "ok": True,
            "method": "textured_colmap_texrecon",
            "textured": True,
            "texture_image_count": 2,
            "atlas_resolution": 4096,
            "uv_coverage_ratio": 0.91,
            "path": str(tmp_path / "visual_mesh.glb"),
        }

    monkeypatch.setattr(module, "_build_visual_mesh_textured_colmap", _fake_textured)
    monkeypatch.setenv("VISUAL_MESH_METHOD", "textured_colmap")
    monkeypatch.setenv("VISUAL_MESH_ENABLED", "true")

    report = module.build_visual_mesh_artifacts(
        output_dir=tmp_path,
        fused_ply=fused_ply,
        gaussian_ply=gaussian_ply,
        workspace=tmp_path,
    )
    assert report["status"] == "ok"
    assert report["selected_method"] == "textured_colmap_texrecon"
    assert report["textured"] is True
    assert report["texture_image_count"] == 2
    assert report["atlas_resolution"] == 4096
    assert report["uv_coverage_ratio"] == pytest.approx(0.91)


def test_write_mesh_manifest_includes_role_entries(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    visual_usdz = tmp_path / "export_last.usdz"
    gaussian_ply = tmp_path / "export_last.ply"
    collision_ply = tmp_path / "nvblox_mesh.ply"
    occupancy = tmp_path / "occupancy.bin"
    visual_mesh = tmp_path / "visual_mesh.glb"
    visual_pointcloud = tmp_path / "visual_pointcloud.ply"
    for p in [visual_usdz, gaussian_ply, collision_ply, occupancy, visual_mesh, visual_pointcloud]:
        p.write_bytes(b"x")

    manifest_path = module.write_mesh_manifest(
        output_dir=tmp_path,
        visual_usdz=visual_usdz,
        gaussian_ply=gaussian_ply,
        collision_mesh_ply=collision_ply,
        occupancy=occupancy,
        visual_report={"selected_method": "quick_poisson_open3d"},
        collision_method="poisson_open3d",
        collision_report={"spike_metrics": {"long_edge_face_ratio": 0.0}},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {entry["role"] for entry in payload.get("assets", [])}
    assert "visual" in roles
    assert "collision" in roles
    assert "volume_visual" in roles
    assert payload["primary_visual_asset"] == "export_last.usdz"
    assert "omniverse_neural" in payload["viewer_compatibility"]
    assert "fallback_vertex_mesh" in payload["viewer_compatibility"]


def test_write_mesh_manifest_prefers_mesh_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    visual_usdz = tmp_path / "export_last.usdz"
    gaussian_ply = tmp_path / "export_last.ply"
    collision_ply = tmp_path / "nvblox_mesh.ply"
    occupancy = tmp_path / "occupancy.bin"
    visual_mesh = tmp_path / "visual_mesh.glb"
    visual_pointcloud = tmp_path / "visual_pointcloud.ply"
    for p in [visual_usdz, gaussian_ply, collision_ply, occupancy, visual_mesh, visual_pointcloud]:
        p.write_bytes(b"x")

    monkeypatch.setenv("NUREC_VISUAL_PRIMARY", "mesh")
    manifest_path = module.write_mesh_manifest(
        output_dir=tmp_path,
        visual_usdz=visual_usdz,
        gaussian_ply=gaussian_ply,
        collision_mesh_ply=collision_ply,
        occupancy=occupancy,
        visual_report={"selected_method": "textured_colmap_texrecon", "textured": True},
        collision_method="poisson_open3d",
        collision_report={"spike_metrics": {"long_edge_face_ratio": 0.0}},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["primary_visual_asset"] == "visual_mesh.glb"
    assert "generic_textured_mesh" in payload["viewer_compatibility"]


def test_build_capture_quality_report_has_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")

    monkeypatch.setattr(
        module,
        "_frame_blur_scores",
        lambda _frames_dir: [  # type: ignore[no-untyped-def]
            (frames_dir / "frame_00001.jpg", 4.5),
            (frames_dir / "frame_00002.jpg", 9.0),
            (frames_dir / "frame_00003.jpg", 12.0),
        ],
    )
    monkeypatch.setattr(
        module,
        "_frame_signal_stats",
        lambda _frames_dir: {  # type: ignore[no-untyped-def]
            "yavg": {1: 90.0, 2: 120.0, 3: 150.0},
            "ydif": {1: 20.0, 2: 30.0, 3: 40.0},
        },
    )

    report = module.build_capture_quality_report(frames_dir)
    assert report["schema_version"] == "v1"
    assert report["frame_count"] == 3
    assert report["blur"]["count"] == 3
    assert report["brightness"]["count"] == 3
    assert report["motion"]["count"] == 3
    assert report["blurriest_frames"][0]["frame"] == "frame_00001.jpg"


def test_sam3_preflight_non_strict_returns_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setattr(module, "_resolve_hf_token", lambda: "")
    monkeypatch.setattr(module, "_sam3_local_cache_present", lambda: False)

    import builtins

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "sam3":
            raise ModuleNotFoundError("sam3 missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    report = module._run_sam3_preflight(strict=False)
    assert report["status"] == "skip"
    assert "sam3_import_failed" in report["reason"]


def test_sam3_preflight_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setattr(module, "_resolve_hf_token", lambda: "")
    monkeypatch.setattr(module, "_sam3_local_cache_present", lambda: False)

    import builtins

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "sam3":
            raise ModuleNotFoundError("sam3 missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="SAM3 preflight failed in strict mode"):
        module._run_sam3_preflight(strict=True)


def test_main_retries_sfm_and_fails_quality_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    job_spec = tmp_path / "job_spec.json"
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"mov")
    job_spec.write_text("{}", encoding="utf-8")

    call_counter = {"sfm": 0}

    def _fake_find_video(raw_prefix: str, storage_root: Path) -> Path:  # noqa: ARG001
        return video_path

    def _fake_extract_frames(video: Path, frames_dir: Path, max_frames: int, target_fps: int) -> int:  # noqa: ARG001
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 101):
            (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
        return 100

    def _fake_run_colmap_sfm(  # type: ignore[no-untyped-def]
        frames_dir,
        workspace,
        *,
        sift_use_gpu,
        mapper_num_threads=0,
        matcher_mode="sequential",
        sequential_overlap=10,
    ):
        del frames_dir, sift_use_gpu, mapper_num_threads, matcher_mode, sequential_overlap
        call_counter["sfm"] += 1
        sparse_dir = workspace / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        registered = 40 if call_counter["sfm"] == 1 else 60
        (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", registered))
        return sparse_dir

    monkeypatch.setattr(module, "find_video", _fake_find_video)
    monkeypatch.setattr(module, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(module, "run_colmap_sfm", _fake_run_colmap_sfm)
    monkeypatch.setattr(module, "_colmap_has_cuda", lambda: False)
    monkeypatch.setattr(
        module,
        "build_capture_quality_report",
        lambda _frames_dir: {"schema_version": "v1", "frame_count": 100},
    )
    monkeypatch.setattr(
        module,
        "_run_sam3_preflight",
        lambda strict: {  # type: ignore[no-untyped-def]
            "schema_version": "v1",
            "generated_at": "now",
            "enabled": True,
            "strict": strict,
            "status": "ok",
            "reason": "",
        },
    )

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "nurec_shim.py",
            "--job-spec",
            str(job_spec),
            "--output-dir",
            str(output_dir),
            "--raw-prefix",
            str(video_path),
            "--no-dependency-preflight",
            "--colmap-min-registered-ratio",
            "0.80",
            "--colmap-retry-min-registered-ratio",
            "0.75",
        ],
    )

    with pytest.raises(RuntimeError, match="COLMAP registration quality gate failed"):
        module.main()

    assert call_counter["sfm"] == 2


def test_resolve_sam3_settings_bedroom_auto() -> None:
    module = _load_nurec_shim_module()

    n_frames, min_frames = module._resolve_sam3_settings(
        environment="bedroom",
        frame_count=263,
        requested_n_frames=0,
        requested_min_frame_detections=0,
    )

    assert n_frames == 26
    assert min_frames == 2


def test_resolve_visual_mesh_poisson_depth_defaults_large_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.delenv("VISUAL_MESH_POISSON_DEPTH", raising=False)
    monkeypatch.delenv("VISUAL_MESH_POISSON_DEPTH_LARGE", raising=False)
    monkeypatch.delenv("VISUAL_MESH_POISSON_LARGE_THRESHOLD", raising=False)
    assert module._resolve_visual_mesh_poisson_depth(700000) == 12
    assert module._resolve_visual_mesh_poisson_depth(200000) == 12


def test_resolve_visual_mesh_poisson_depth_respects_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("VISUAL_MESH_POISSON_DEPTH", "11")
    monkeypatch.setenv("VISUAL_MESH_POISSON_DEPTH_LARGE", "9")
    monkeypatch.setenv("VISUAL_MESH_POISSON_LARGE_THRESHOLD", "250000")
    assert module._resolve_visual_mesh_poisson_depth(260000) == 9
    assert module._resolve_visual_mesh_poisson_depth(120000) == 11


# ---------------------------------------------------------------------------
# Adaptive Long-Capture: _probe_video_duration_seconds
# ---------------------------------------------------------------------------


def test_probe_video_duration_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()

    class _FakeResult:
        returncode = 0
        stdout = "182.45\n"
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: _FakeResult())
    result = module._probe_video_duration_seconds(tmp_path / "video.mov")
    assert result == pytest.approx(182.45)


def test_probe_video_duration_ffprobe_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()

    class _FakeResult:
        returncode = 1
        stdout = ""
        stderr = "No such file"

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: _FakeResult())
    result = module._probe_video_duration_seconds(tmp_path / "video.mov")
    assert result is None


def test_probe_video_duration_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()

    def _raise(*a, **kw):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(module.subprocess, "run", _raise)
    result = module._probe_video_duration_seconds(tmp_path / "video.mov")
    assert result is None


# ---------------------------------------------------------------------------
# Adaptive Long-Capture: _resolve_effective_max_frames edge cases
# ---------------------------------------------------------------------------


def test_resolve_effective_max_frames_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES", "false")
    max_frames, reason = module._resolve_effective_max_frames(1800.0, 450)
    assert max_frames == 450
    assert "adaptive_max_frames=disabled" in reason


def test_resolve_effective_max_frames_duration_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES", "true")
    max_frames, reason = module._resolve_effective_max_frames(None, 450)
    assert max_frames == 450
    assert "duration_unknown" in reason


def test_resolve_effective_max_frames_respects_hard_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES", "true")
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES_TARGET_FPS", "10.0")
    monkeypatch.setenv("ADAPTIVE_MAX_FRAMES_HARD_CAP", "2000")
    max_frames, reason = module._resolve_effective_max_frames(1800.0, 450)
    assert max_frames == 2000
    assert "hard_cap=2000" in reason


# ---------------------------------------------------------------------------
# Adaptive Long-Capture: _resolve_effective_extract_fps edge cases
# ---------------------------------------------------------------------------


def test_resolve_effective_extract_fps_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_EXTRACT_FPS", "false")
    fps, reason = module._resolve_effective_extract_fps(1800.0, 6, 900)
    assert fps == 6.0
    assert "adaptive_extract_fps=disabled" in reason


def test_resolve_effective_extract_fps_duration_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_EXTRACT_FPS", "true")
    fps, reason = module._resolve_effective_extract_fps(None, 6, 900)
    assert fps == 6.0
    assert "duration_unknown" in reason


# ---------------------------------------------------------------------------
# Adaptive Long-Capture: _build_colmap_chunk_ranges edge cases
# ---------------------------------------------------------------------------


def test_build_colmap_chunk_ranges_zero_frames() -> None:
    module = _load_nurec_shim_module()
    assert module._build_colmap_chunk_ranges(0, chunk_size=600, chunk_overlap=120, max_chunks=24) == []


def test_build_colmap_chunk_ranges_total_equals_chunk_size() -> None:
    module = _load_nurec_shim_module()
    ranges = module._build_colmap_chunk_ranges(600, chunk_size=600, chunk_overlap=120, max_chunks=24)
    assert len(ranges) == 1
    assert ranges[0] == (0, 600)


def test_build_colmap_chunk_ranges_total_below_chunk_size() -> None:
    module = _load_nurec_shim_module()
    ranges = module._build_colmap_chunk_ranges(200, chunk_size=600, chunk_overlap=120, max_chunks=24)
    assert len(ranges) == 1
    assert ranges[0] == (0, 200)


# ---------------------------------------------------------------------------
# Adaptive Long-Capture: _run_sfm_with_optional_chunking
# ---------------------------------------------------------------------------


def test_run_sfm_with_optional_chunking_single_pass_below_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When frame count is below chunk threshold, uses single-pass SfM."""
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "mapper":
            sparse = workspace / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_bytes(struct.pack("<Q", 80))
        return None

    monkeypatch.setattr(module, "_run", _fake_run)
    monkeypatch.setattr(module, "_colmap_supports_option", lambda *a, **kw: True)

    sparse_dir, registered, report = module._run_sfm_with_optional_chunking(
        frames_dir=frames_dir,
        workspace=workspace,
        sift_use_gpu=False,
        mapper_num_threads=1,
        matcher_mode="sequential",
        sequential_overlap=30,
        frame_count=400,
        chunked_mode="auto",
        chunk_min_frames=900,
        chunk_size_frames=600,
        chunk_overlap_frames=120,
        chunk_max_chunks=24,
        chunk_matcher_mode="sequential",
    )
    assert report["chunking_enabled"] is False
    assert report["chunking_applied"] is False
    assert registered == 80


def test_run_sfm_with_optional_chunking_falls_back_on_chunk_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When chunking is enabled but fails, falls back to single-pass."""
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    def _fake_chunked(*a, **kw):
        raise RuntimeError("simulated chunk failure")

    def _fake_sfm(frames_dir, workspace, *, sift_use_gpu, mapper_num_threads=0,
                  matcher_mode="sequential", sequential_overlap=10):
        sparse_dir = workspace / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", 500))
        return sparse_dir

    monkeypatch.setattr(module, "run_colmap_sfm_chunked", _fake_chunked)
    monkeypatch.setattr(module, "run_colmap_sfm", _fake_sfm)

    sparse_dir, registered, report = module._run_sfm_with_optional_chunking(
        frames_dir=frames_dir,
        workspace=workspace,
        sift_use_gpu=False,
        mapper_num_threads=1,
        matcher_mode="sequential",
        sequential_overlap=30,
        frame_count=1000,
        chunked_mode="auto",
        chunk_min_frames=900,
        chunk_size_frames=600,
        chunk_overlap_frames=120,
        chunk_max_chunks=24,
        chunk_matcher_mode="sequential",
    )
    assert report["chunking_enabled"] is True
    assert report["chunking_applied"] is False
    assert report["chunking_fallback"] == "single_pass"
    assert registered == 500


# ---------------------------------------------------------------------------
# Adaptive Long-Capture: run_colmap_sfm_chunked smoke test
# ---------------------------------------------------------------------------


def test_run_colmap_sfm_chunked_merges_two_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    workspace = tmp_path / "workspace"
    frames_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    for i in range(1, 801):
        (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")

    def _fake_run(cmd, **kwargs):
        if cmd[1] == "mapper":
            out_idx = cmd.index("--output_path") + 1
            sparse = Path(cmd[out_idx]) / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_bytes(struct.pack("<Q", 50))
            return None
        if cmd[1] == "model_merger":
            out_idx = cmd.index("--output_path") + 1
            out = Path(cmd[out_idx])
            out.mkdir(parents=True, exist_ok=True)
            (out / "cameras.bin").write_bytes(b"\x00")
            (out / "images.bin").write_bytes(struct.pack("<Q", 90))
            (out / "points3D.bin").write_bytes(b"\x00")
            return None
        return None

    monkeypatch.setattr(module, "_run", _fake_run)
    monkeypatch.setattr(module, "_colmap_supports_option", lambda *a, **kw: True)

    sparse_dir, report = module.run_colmap_sfm_chunked(
        frames_dir,
        workspace,
        sift_use_gpu=False,
        mapper_num_threads=1,
        chunk_size_frames=500,
        chunk_overlap_frames=100,
        chunk_max_chunks=10,
        chunk_matcher_mode="sequential",
        sequential_overlap=30,
    )
    assert sparse_dir.exists()
    assert report["enabled"] is True
    assert report["chunk_count_planned"] >= 2
    assert report["chunk_count_successful"] >= 2
