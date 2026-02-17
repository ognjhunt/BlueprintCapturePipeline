"""Tests for nurec_shim Fixer backend routing."""

from __future__ import annotations

import importlib.util
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
