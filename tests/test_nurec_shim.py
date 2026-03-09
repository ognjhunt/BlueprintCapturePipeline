"""Tests for nurec_shim Fixer backend routing."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def test_quality_profile_defaults_set_blur_filter_ratios() -> None:
    module = _load_nurec_shim_module()
    quality_first = module._quality_profile_defaults("quality_first")
    balanced = module._quality_profile_defaults("balanced")
    fast = module._quality_profile_defaults("fast")

    assert quality_first["blur_filter_keep_ratio"] == pytest.approx(0.85)
    assert balanced["blur_filter_keep_ratio"] == pytest.approx(0.90)
    assert fast["blur_filter_keep_ratio"] == pytest.approx(1.0)
    assert quality_first["blur_filter_min_frames"] == 120
    assert balanced["blur_filter_min_frames"] == 120
    assert fast["blur_filter_min_frames"] == 120


def test_apply_blur_filter_required_raises_when_scores_unavailable(
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
        lambda _frames_dir, fail_on_error=False: [],  # type: ignore[no-untyped-def]
    )

    with pytest.raises(RuntimeError, match="blur filtering is required"):
        module._apply_blur_frame_filter(
            frames_dir,
            keep_ratio=0.85,
            min_keep=2,
            fail_on_error=True,
        )


def test_resolve_stage14_resume_rejects_missing_metadata(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    workspace = output_dir / "_colmap_workspace"
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    (output_dir / "export_last.usdz").write_bytes(b"usdz")
    (output_dir / "export_last.ply").write_bytes(b"ply")

    ok, existing, reasons = module._resolve_stage14_resume(
        resume_requested=True,
        quality_guardrails=True,
        output_dir=output_dir,
        workspace=workspace,
        profile="quality_first",
        video_signature={"size_bytes": 123, "mtime_ns": 456},
        requested_max_frames=450,
        effective_max_frames=450,
        requested_extract_fps=6,
        effective_extract_fps=6.0,
        blur_filter_keep_ratio=0.85,
        blur_filter_min_frames=120,
        n_iterations=12000,
    )

    assert ok is False
    assert existing is None
    assert "missing_stage14_resume_metadata" in reasons


def test_resolve_stage14_resume_accepts_matching_metadata(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    workspace = output_dir / "_colmap_workspace"
    frames_dir = workspace / "frames"
    sparse_dir = workspace / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "export_last.usdz").write_bytes(b"usdz")
    (output_dir / "export_last.ply").write_bytes(b"ply")
    for i in range(1, 11):
        (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
    (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", 10))

    metadata = {
        "schema_version": "v1",
        "quality_profile": "quality_first",
        "video": {"size_bytes": 123, "mtime_ns": 456},
        "stage1": {
            "frame_count": 10,
            "requested_max_frames": 450,
            "effective_max_frames": 450,
            "requested_extract_fps": 6,
            "effective_extract_fps": 6.0,
            "blur_filter": {
                "status": "ok",
                "keep_ratio": 0.85,
                "min_frames": 120,
            },
        },
        "stage2": {"registered_images": 10},
        "stage4": {"n_iterations": 12000},
    }
    (output_dir / module.STAGE14_RESUME_METADATA).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    ok, existing, reasons = module._resolve_stage14_resume(
        resume_requested=True,
        quality_guardrails=True,
        output_dir=output_dir,
        workspace=workspace,
        profile="quality_first",
        video_signature={"size_bytes": 123, "mtime_ns": 456},
        requested_max_frames=450,
        effective_max_frames=450,
        requested_extract_fps=6,
        effective_extract_fps=6.0,
        blur_filter_keep_ratio=0.85,
        blur_filter_min_frames=120,
        n_iterations=12000,
    )

    assert ok is True
    assert existing is not None
    assert reasons == ["metadata_match"]


def test_validate_stage9_resume_metadata_rejects_mismatch() -> None:
    module = _load_nurec_shim_module()
    reasons = module._validate_stage9_resume_metadata(
        {
            "schema_version": "v1",
            "video": {"size_bytes": 111, "mtime_ns": 222},
            "gaussian_ply": {"size_bytes": 333, "mtime_ns": 444},
            "requested_environment": "warehouse",
            "requested_n_frames": 20,
            "requested_min_frame_detections": 2,
            "scene_cleaning_mode": "off",
            "sam3_mask_export_space": "undistorted",
        },
        video_signature={"size_bytes": 111, "mtime_ns": 999},
        gaussian_signature={"size_bytes": 333, "mtime_ns": 444},
        requested_environment="warehouse",
        requested_n_frames=20,
        requested_min_frame_detections=2,
        scene_cleaning_mode="off",
        sam3_mask_export_space="undistorted",
    )

    assert "stage9_video_mtime_ns_changed" in reasons


def test_run_stage9_resume_uses_matching_metadata(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    workspace = tmp_path / "ws"
    frames_dir = tmp_path / "frames"
    undistorted_images = tmp_path / "undist"
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    undistorted_images.mkdir(parents=True, exist_ok=True)

    gaussian_ply = output_dir / "export_last.ply"
    gaussian_ply.write_bytes(b"ply")
    (output_dir / "scene_semantics_report.json").write_text("{}", encoding="utf-8")
    index_path = output_dir / "object_point_cloud_index.json"
    index_path.write_text('{"objects": []}', encoding="utf-8")

    video_signature = {"size_bytes": 123, "mtime_ns": 456}
    module._write_stage9_resume_metadata(
        output_dir,
        {
            "schema_version": "v1",
            "video": dict(video_signature),
            "gaussian_ply": module._file_signature(gaussian_ply),
            "requested_environment": "warehouse",
            "requested_n_frames": 20,
            "requested_min_frame_detections": 2,
            "scene_cleaning_mode": "off",
            "sam3_mask_export_space": "undistorted",
        },
    )

    result = module._run_stage9_sam3(
        output_dir=output_dir,
        workspace=workspace,
        frames_dir=frames_dir,
        undistorted_images_dir=undistorted_images,
        frame_count=0,
        requested_environment="warehouse",
        requested_n_frames=20,
        requested_min_frame_detections=2,
        gaussian_ply=gaussian_ply,
        video_signature=video_signature,
        resume=True,
        scene_cleaning_mode="off",
        sam3_mask_export_space="undistorted",
    )

    assert result == index_path

def test_run_3dgrut_training_selects_newest_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    threedgrut_dir = tmp_path / "3dgrut_src"
    train_script = threedgrut_dir / "train.py"
    train_script.parent.mkdir(parents=True, exist_ok=True)
    train_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(module, "THREEDGRUT_DIR", str(threedgrut_dir))

    output_dir = tmp_path / "output"
    undistorted_dir = tmp_path / "undistorted"
    output_dir.mkdir(parents=True, exist_ok=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del cmd, kwargs
        result_root = output_dir / "3dgrut" / "nurec_scene"
        old_dir = result_root / "old"
        new_dir = result_root / "new"
        old_dir.mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)
        (old_dir / "export_last.usdz").write_bytes(b"old")
        (old_dir / "export_last.ply").write_bytes(b"old")
        (old_dir / "export_last.ingp").write_bytes(b"old")
        (new_dir / "export_last.usdz").write_bytes(b"new")
        (new_dir / "export_last.ply").write_bytes(b"new")
        (new_dir / "export_last.ingp").write_bytes(b"new")
        old_ts = 1700000000
        new_ts = 1700003600
        os.utime(old_dir / "export_last.usdz", (old_ts, old_ts))
        os.utime(new_dir / "export_last.usdz", (new_ts, new_ts))
        return None

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.run_3dgrut_training(
        undistorted_dir=undistorted_dir,
        output_dir=output_dir,
        n_iterations=12000,
    )
    assert result["result_dir"].name == "new"


def test_fixer_auto_uses_local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    output_dir = tmp_path / "out"
    fixed_dir = output_dir / "fixer_output"
    renders_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    call_count = {"local": 0, "h100": 0}

    def _fake_h100(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["h100"] += 1
        fixed_dir.mkdir(parents=True, exist_ok=True)
        (fixed_dir / "frame_from_h100.png").write_bytes(b"h100")
        return True

    def _fake_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["local"] += 1
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
    assert call_count["local"] == 1
    assert call_count["h100"] == 0


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

    h100_calls = {"count": 0}

    def _fake_h100(*args, **kwargs):  # type: ignore[no-untyped-def]
        h100_calls["count"] += 1
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
    assert h100_calls["count"] == 0


def test_run_fixer_refinement_writes_completion_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    output_dir = tmp_path / "out"
    fixed_dir = output_dir / "fixer_output"
    renders_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _fake_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        fixed_dir.mkdir(parents=True, exist_ok=True)
        (fixed_dir / "frame_00001.png").write_bytes(b"ok")
        (fixed_dir / "frame_00002.png").write_bytes(b"ok")
        return True

    monkeypatch.setattr(module, "_run_fixer_local_stage", _fake_local)

    result = module.run_fixer_refinement(
        renders_dir,
        output_dir,
        mode="local",
        h100_script=tmp_path / "dummy.sh",
    )

    marker_path = fixed_dir / ".fixer_stage_complete.json"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert result == fixed_dir
    assert payload["backend"] == "local"
    assert payload["image_count"] == 2


def test_run_fixer_refinement_clears_stale_images_when_retry_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    renders_dir = tmp_path / "renders"
    output_dir = tmp_path / "out"
    fixed_dir = output_dir / "fixer_output"
    renders_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    (fixed_dir / "stale.png").write_bytes(b"old")

    def _fake_local(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return False

    monkeypatch.setattr(module, "_run_fixer_local_stage", _fake_local)

    result = module.run_fixer_refinement(
        renders_dir,
        output_dir,
        mode="local",
        h100_script=tmp_path / "dummy.sh",
    )

    assert result == renders_dir
    assert not (fixed_dir / "stale.png").exists()
    assert not (fixed_dir / ".fixer_stage_complete.json").exists()


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


def test_run_fixer_local_stage_skips_when_transformer_engine_extension_missing(
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
    (weights_dir / "base" / "model_fast_tokenizer.pt").parent.mkdir(parents=True, exist_ok=True)
    (weights_dir / "base" / "model_fast_tokenizer.pt").write_bytes(b"ok")
    (weights_dir / "base" / "tokenizer_fast.pth").write_bytes(b"ok")

    monkeypatch.setattr(module, "FIXER_DIR", str(fixer_dir))
    monkeypatch.setattr(module, "FIXER_WEIGHTS_DIR", str(weights_dir))

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        calls.append(list(cmd))
        if "-c" in cmd and "transformer_engine.pytorch" in cmd[-1]:
            raise RuntimeError("missing transformer_engine_torch.so")
        raise AssertionError("inference should not run when preflight fails")

    monkeypatch.setattr(module, "_run", _fake_run)

    assert module._run_fixer_local_stage(renders_dir, fixed_dir) is False
    assert len(calls) == 1
    assert calls[0][0] == "python3"
    assert calls[0][1] == "-c"


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


def test_resolve_post_stage4_refine_mode() -> None:
    module = _load_nurec_shim_module()
    assert module._resolve_post_stage4_refine_mode("off") == "off"
    assert module._resolve_post_stage4_refine_mode("AUTO") == "auto"
    assert module._resolve_post_stage4_refine_mode("force") == "force"
    assert module._resolve_post_stage4_refine_mode("unknown") == "auto"


def test_has_valid_post_stage4_refine_cache(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    (tmp_path / "export_last_refined.usdz").write_bytes(b"u")
    (tmp_path / "export_last_refined.ply").write_bytes(b"p")
    (tmp_path / "refinement_quality_gate.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    assert module._has_valid_post_stage4_refine_cache(tmp_path) is True

    (tmp_path / "refinement_quality_gate.json").write_text(
        json.dumps({"status": "failed_safe_rollback"}),
        encoding="utf-8",
    )
    assert module._has_valid_post_stage4_refine_cache(tmp_path) is False


def test_evaluate_refinement_quality_gate_triggers_rollback() -> None:
    module = _load_nurec_shim_module()
    gate = module._evaluate_refinement_quality_gate(
        baseline_hole_ratio=0.20,
        refined_hole_ratio=0.18,  # only 10% improvement -> fail
        pre_sharpness=100.0,
        post_sharpness=80.0,  # 20% drop -> fail
        baseline_psnr=26.8,
        refined_psnr=26.0,  # 0.8 dB drop > 0.5 threshold -> fail
    )
    assert gate["status"] == "failed_safe_rollback"
    assert gate["metric_basis"] == "candidate_pre_post_repair"
    assert gate["gates"]["hole_improvement"] is False
    assert gate["gates"]["sharpness"] is False
    assert gate["gates"]["psnr"] is False


def test_main_forwards_colmap_images_bin_to_gap_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    job_spec = tmp_path / "job_spec.json"
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"mov")
    job_spec.write_text("{}", encoding="utf-8")
    captured_cmds: list[list[str]] = []

    def _fake_find_video(raw_prefix: str, storage_root: Path) -> Path:  # noqa: ARG001
        return video_path

    def _fake_extract_frames(video: Path, frames_dir: Path, max_frames: int, target_fps: int) -> int:  # noqa: ARG001
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 121):
            (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
        return 120

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
        sparse_dir = workspace / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        (sparse_dir / "cameras.bin").write_bytes(b"\x00")
        (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", 120))
        (sparse_dir / "points3D.bin").write_bytes(b"\x00")
        return sparse_dir

    def _fake_run_colmap_undistort(frames_dir: Path, sparse_dir: Path, workspace: Path) -> Path:  # noqa: ARG001
        undistorted_dir = workspace / "undistorted"
        und_images = undistorted_dir / "images"
        und_sparse = undistorted_dir / "sparse" / "0"
        und_images.mkdir(parents=True, exist_ok=True)
        und_sparse.mkdir(parents=True, exist_ok=True)
        (und_images / "frame_00001.jpg").write_bytes(b"jpg")
        (und_sparse / "images.bin").write_bytes(struct.pack("<Q", 120))
        (und_sparse / "points3D.bin").write_bytes(b"\x00")
        return undistorted_dir

    def _fake_run_3dgrut_training(  # type: ignore[no-untyped-def]
        undistorted_dir,
        output_dir,
        n_iterations,
        max_n_gaussians=0,
        add_end_iteration=0,
        post_plan=None,
    ):
        del undistorted_dir, n_iterations, max_n_gaussians, add_end_iteration, post_plan
        result_dir = output_dir / "3dgrut" / "scene"
        renders_dir = result_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        (renders_dir / "frame_00001.png").write_bytes(b"png")
        usdz = result_dir / "export_last.usdz"
        ply = result_dir / "export_last.ply"
        ingp = result_dir / "export_last.ingp"
        (result_dir / "ckpt_last.pt").write_bytes(b"ckpt")
        usdz.write_bytes(b"usdz")
        ply.write_bytes(b"ply")
        ingp.write_bytes(b"ingp")
        return {
            "usdz": str(usdz),
            "ply": str(ply),
            "ingp": str(ingp),
            "result_dir": str(result_dir),
            "metrics": {"mean_psnr": 26.8},
        }

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        captured_cmds.append([str(part) for part in cmd])
        cmd_path = str(cmd[1]) if len(cmd) > 1 else ""
        if cmd_path.endswith("post_stage4_gap_analyzer.py"):
            (output_dir / "gap_candidate_views.jsonl").write_text(
                json.dumps(
                    {
                        "id": "stage45_virtual_1",
                        "is_virtual": True,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "gap_analysis_report.json").write_text(
                json.dumps({"global_hole_pixel_ratio": 0.4, "virtual_candidates_selected": 1}),
                encoding="utf-8",
            )
        elif cmd_path.endswith("post_stage4_virtual_render.py"):
            work_dir = Path(_cmd_arg(cmd, "--work-dir"))
            renders = work_dir / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            (renders / "00000.png").write_bytes(b"img")
            mapping_path = work_dir / "virtual_render_mapping.jsonl"
            mapping_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "stage45_virtual_1",
                        "render_name": "00000.png",
                        "render_exists": True,
                        "render_image": str((renders / "00000.png").resolve()),
                        "predicted_hole_ratio": 0.2,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (work_dir / "virtual_render_report.json").write_text(
                json.dumps({"rendered_count": 1, "renders_dir": str(renders), "mapping_path": str(mapping_path)}),
                encoding="utf-8",
            )
        elif cmd_path.endswith("post_stage4_view_repair.py"):
            (output_dir / "accepted_repaired_views.jsonl").write_text("", encoding="utf-8")
            (output_dir / "view_repair_report.json").write_text(
                json.dumps(
                    {
                        "pre_sharpness_mean": 100.0,
                        "post_sharpness_mean": 100.0,
                        "post_repair_hole_ratio_mean": 0.2,
                    }
                ),
                encoding="utf-8",
            )
        elif cmd_path.endswith("post_stage4_distill.py"):
            (output_dir / "export_last_refined.usdz").write_bytes(b"refined_usdz")
            (output_dir / "export_last_refined.ply").write_bytes(b"refined_ply")
            (output_dir / "post_stage4_distill_report.json").write_text(
                json.dumps({"refined_metrics": {"mean_psnr": 26.8}}),
                encoding="utf-8",
            )
        return None

    def _fake_mesh_with_open3d_poisson(fused_ply, output_ply, *, force=False):  # type: ignore[no-untyped-def]
        del fused_ply, force
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        # Write minimal valid PLY with a triangle so _validate_collision_mesh passes.
        header = (
            "ply\nformat ascii 1.0\n"
            "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        )
        body = "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
        output_ply.write_text(header + body, encoding="utf-8")
        return True

    def _fake_build_visual(*, output_dir, fused_ply, gaussian_ply, workspace=None, refined_images_dir=None):  # type: ignore[no-untyped-def]
        del fused_ply, gaussian_ply, workspace, refined_images_dir
        visual_mesh = output_dir / "visual_mesh.glb"
        visual_pointcloud = output_dir / "visual_pointcloud.ply"
        visual_mesh.write_bytes(b"glb")
        visual_pointcloud.write_bytes(b"ply")
        return {"enabled": True, "status": "ok", "selected_method": "gaussian_tsdf"}

    def _fake_generate_occupancy(ply_path, output_bin, resolution=64):  # type: ignore[no-untyped-def]
        del ply_path, resolution
        output_bin.write_bytes(b"\x00" * 64)

    monkeypatch.setattr(module, "find_video", _fake_find_video)
    monkeypatch.setattr(module, "extract_frames", _fake_extract_frames)
    monkeypatch.setattr(module, "run_colmap_sfm", _fake_run_colmap_sfm)
    monkeypatch.setattr(module, "run_colmap_undistort", _fake_run_colmap_undistort)
    monkeypatch.setattr(module, "run_3dgrut_training", _fake_run_3dgrut_training)
    monkeypatch.setattr(module, "_colmap_has_cuda", lambda: False)
    monkeypatch.setattr(module, "_run", _fake_run)
    monkeypatch.setattr(module, "_mesh_with_open3d_poisson", _fake_mesh_with_open3d_poisson)
    monkeypatch.setattr(module, "build_visual_mesh_artifacts", _fake_build_visual)
    monkeypatch.setattr(module, "generate_occupancy", _fake_generate_occupancy)
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
        module,
        "build_capture_quality_report",
        lambda _frames_dir: {"schema_version": "v1", "frame_count": 120},
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
            "--no-blur-filter-required",
            "--skip-fixer",
            "--post-stage4-refine",
            "force",
            "--skip-dense",
        ],
    )

    module.main()

    gap_cmd = next(
        cmd for cmd in captured_cmds if len(cmd) > 1 and cmd[1].endswith("post_stage4_gap_analyzer.py")
    )
    assert "--colmap-images-bin" in gap_cmd
    bin_idx = gap_cmd.index("--colmap-images-bin") + 1
    expected_bin = output_dir / "_colmap_workspace" / "undistorted" / "sparse" / "0" / "images.bin"
    assert gap_cmd[bin_idx] == str(expected_bin)
    assert "--colmap-points3d-bin" in gap_cmd
    assert "--max-virtual-candidates" in gap_cmd

    virtual_render_cmd = next(
        cmd for cmd in captured_cmds if len(cmd) > 1 and cmd[1].endswith("post_stage4_virtual_render.py")
    )
    assert "--candidates-jsonl" in virtual_render_cmd
    assert "--checkpoint" in virtual_render_cmd

    view_repair_cmd = next(
        cmd for cmd in captured_cmds if len(cmd) > 1 and cmd[1].endswith("post_stage4_view_repair.py")
    )
    assert "--virtual-render-mapping" in view_repair_cmd
    assert "--model" in view_repair_cmd
    model_idx = view_repair_cmd.index("--model") + 1
    assert view_repair_cmd[model_idx] == "worldforge+gsfix3d"

    distill_cmd = next(
        cmd for cmd in captured_cmds if len(cmd) > 1 and cmd[1].endswith("post_stage4_distill.py")
    )
    assert "--virtual-renders-dir" in distill_cmd
    assert "--virtual-candidates-jsonl" in distill_cmd
    # Verify collision mesh was generated from Gaussian PLY (no PatchMatch).
    assert (output_dir / "nvblox_mesh.ply").exists()
    assert (output_dir / "mesh_method.txt").read_text(encoding="utf-8").strip() == "poisson_open3d"


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
    assert report["blurriest_frames"][0]["frame"] == "frame_00003.jpg"
    assert report["sharpest_frames"][0]["frame"] == "frame_00001.jpg"


def test_frame_blur_scores_handles_non_contiguous_frame_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_num in (1, 3, 7):
        (frames_dir / f"frame_{frame_num:05d}.jpg").write_bytes(b"jpg")

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del cmd, kwargs
        (frames_dir / ".blurdetect_report.txt").write_text(
            "\n".join(
                [
                    "frame:0 pts:0 pts_time:0",
                    "lavfi.blur=1.5",
                    "frame:1 pts:1 pts_time:0.04",
                    "lavfi.blur=2.5",
                    "frame:2 pts:2 pts_time:0.08",
                    "lavfi.blur=3.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    entries = module._frame_blur_scores(frames_dir)
    assert [path.name for path, _score in entries] == [
        "frame_00001.jpg",
        "frame_00003.jpg",
        "frame_00007.jpg",
    ]
    assert [score for _path, score in entries] == [pytest.approx(1.5), pytest.approx(2.5), pytest.approx(3.5)]


def test_frame_signal_stats_handles_non_contiguous_frame_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_num in (2, 5, 9):
        (frames_dir / f"frame_{frame_num:05d}.jpg").write_bytes(b"jpg")

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del cmd, kwargs
        (frames_dir / ".signalstats_report.txt").write_text(
            "\n".join(
                [
                    "frame:0 pts:0 pts_time:0",
                    "lavfi.signalstats.YAVG=100.0",
                    "lavfi.signalstats.YDIF=10.0",
                    "frame:1 pts:1 pts_time:0.04",
                    "lavfi.signalstats.YAVG=110.0",
                    "lavfi.signalstats.YDIF=11.0",
                    "frame:2 pts:2 pts_time:0.08",
                    "lavfi.signalstats.YAVG=120.0",
                    "lavfi.signalstats.YDIF=12.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    stats = module._frame_signal_stats(frames_dir)
    assert stats["yavg"] == {2: pytest.approx(100.0), 5: pytest.approx(110.0), 9: pytest.approx(120.0)}
    assert stats["ydif"] == {2: pytest.approx(10.0), 5: pytest.approx(11.0), 9: pytest.approx(12.0)}


def test_pipeline_mode_photoreal_hallucination_applies_clarity_overrides() -> None:
    module = _load_nurec_shim_module()
    args = SimpleNamespace(
        pipeline_mode="photoreal_hallucination",
        scene_cleaning_mode="auto",
        max_frames=180,
        extract_fps=4,
        n_iterations=9000,
        max_n_gaussians=0,
        blur_filter_keep_ratio=0.90,
        colmap_matcher_mode="auto",
        colmap_sequential_overlap=20,
        post_stage4_refine="auto",
        post_stage4_refine_model="fixer",
        post_stage4_max_pseudoviews=96,
        post_stage4_distill_iters=1600,
        post_stage4_time_budget_min=90,
        void_fill_rounds=2,
    )
    module._apply_pipeline_mode_overrides(args)
    assert args.scene_cleaning_mode == "off"
    assert args.max_frames >= 500
    assert args.extract_fps >= 8
    assert args.n_iterations >= 22000
    assert args.max_n_gaussians >= 500000
    assert args.blur_filter_keep_ratio <= 0.70
    assert args.colmap_matcher_mode == "sequential"
    assert args.colmap_sequential_overlap >= 40
    assert args.post_stage4_refine == "force"
    assert args.post_stage4_refine_model == "worldforge+gsfix3d"
    assert args.void_fill_rounds == 0


def test_refinement_gate_profile_auto_uses_hallucination_for_hallucination_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    monkeypatch.delenv("REFINEMENT_QUALITY_GATE_PROFILE", raising=False)
    monkeypatch.delenv("REFINEMENT_GATE_ENFORCE_PSNR", raising=False)
    gate = module._resolve_refinement_quality_gate_profile(
        pipeline_mode="photoreal_hallucination",
    )
    assert gate["resolved_profile"] == "hallucination"
    assert gate["enforce_psnr"] is False


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
            "--no-blur-filter-required",
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


# ---------------------------------------------------------------------------
# _read_3d_point_count
# ---------------------------------------------------------------------------


def test_read_3d_point_count_valid(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "points3D.bin").write_bytes(struct.pack("<Q", 15432))
    assert module._read_3d_point_count(model_dir) == 15432


def test_read_3d_point_count_missing(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    # No points3D.bin file
    assert module._read_3d_point_count(model_dir) == 0


def test_read_3d_point_count_empty_file(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "points3D.bin").write_bytes(b"")
    assert module._read_3d_point_count(model_dir) == 0


def test_read_3d_point_count_short_file(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "points3D.bin").write_bytes(b"\x01\x02\x03")  # Only 3 bytes, need 8
    assert module._read_3d_point_count(model_dir) == 0


def test_read_3d_point_count_reads_only_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_nurec_shim_module()
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    points3d = model_dir / "points3D.bin"
    points3d.write_bytes(b"placeholder")

    class _Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size: int = -1) -> bytes:
            assert size == 8
            return struct.pack("<Q", 42)

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self == points3d and mode == "rb":
            return _Reader()
        return original_open(self, mode, *args, **kwargs)

    original_open = Path.open
    monkeypatch.setattr(Path, "open", _patched_open)

    assert module._read_3d_point_count(model_dir) == 42


# ---------------------------------------------------------------------------
# _resolve_effective_max_n_gaussians
# ---------------------------------------------------------------------------


def test_resolve_max_n_gaussians_small_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """51s bedroom clip: 15K SfM pts, 134 frames → should be ~268K (min of 300K sfm, 268K frame)."""
    module = _load_nurec_shim_module()
    # Clear env overrides
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=15000,
        n_iterations=9000,
        requested_max_n_gaussians=0,
    )
    # sfm_signal = 15000 * 20 = 300000, frame_signal = 134 * 2000 = 268000
    # min(300000, 268000) = 268000, clamped to [100K, 2M] → 268000
    assert resolved == 268000
    assert end_iter == int(9000 * 0.85)  # 7650
    assert "adaptive_max_n_gaussians=enabled" in reason
    assert "min(sfm,frame)" in reason


def test_resolve_max_n_gaussians_medium_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """51s clip at higher density: 30K SfM pts, 269 frames → ~538K."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=269,
        sfm_point_count=30000,
        n_iterations=12000,
        requested_max_n_gaussians=0,
    )
    # sfm_signal = 30000 * 20 = 600000, frame_signal = 269 * 2000 = 538000
    # min(600000, 538000) = 538000
    assert resolved == 538000
    assert end_iter == int(12000 * 0.85)  # 10200


def test_resolve_max_n_gaussians_large_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """5-min video: 200K SfM pts, 1500 frames → ceiling 2M."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=300.0,
        registered_frame_count=1500,
        sfm_point_count=200000,
        n_iterations=12000,
        requested_max_n_gaussians=0,
    )
    # sfm_signal = 200000 * 20 = 4000000, frame_signal = 1500 * 2000 = 3000000
    # min(4M, 3M) = 3M, clamped to ceiling 2M
    assert resolved == 2_000_000
    assert "ceiling" in reason


def test_resolve_max_n_gaussians_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ADAPTIVE_MAX_N_GAUSSIANS=false, use default 1M or user value."""
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MAX_N_GAUSSIANS", "false")

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=15000,
        n_iterations=9000,
        requested_max_n_gaussians=0,
    )
    assert resolved == 1_000_000
    assert "adaptive_max_n_gaussians=disabled" in reason


def test_resolve_max_n_gaussians_disabled_with_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """When disabled + user override, use user value."""
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MAX_N_GAUSSIANS", "false")

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=15000,
        n_iterations=9000,
        requested_max_n_gaussians=750000,
    )
    assert resolved == 750000
    assert "adaptive_max_n_gaussians=disabled" in reason


def test_resolve_max_n_gaussians_user_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --max-n-gaussians > 0 overrides adaptive calculation."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=15000,
        n_iterations=9000,
        requested_max_n_gaussians=400000,
    )
    assert resolved == 400000
    assert "user_override" in reason


def test_resolve_max_n_gaussians_no_sfm_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """When SfM points not available, use frame signal only."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=0,
        n_iterations=9000,
        requested_max_n_gaussians=0,
    )
    # frame_signal only = 134 * 2000 = 268000
    assert resolved == 268000
    assert "frame_only" in reason


def test_resolve_max_n_gaussians_no_sfm_no_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither signal is available, fall back to floor."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=0,
        sfm_point_count=0,
        n_iterations=9000,
        requested_max_n_gaussians=0,
    )
    assert resolved == 100_000  # hard_floor
    assert "fallback_floor" in reason


def test_resolve_max_n_gaussians_respects_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny scene: 100 SfM pts, 10 frames → floor of 100K."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING", "GRUT_REFINEMENT_TAIL_RATIO"):
        monkeypatch.delenv(key, raising=False)

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=10.0,
        registered_frame_count=10,
        sfm_point_count=100,
        n_iterations=7000,
        requested_max_n_gaussians=0,
    )
    # sfm_signal = 100 * 20 = 2000, frame_signal = 10 * 2000 = 20000
    # min(2000, 20000) = 2000, clamped to floor 100K
    assert resolved == 100_000


def test_resolve_max_n_gaussians_end_iteration_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify add_end_iteration = n_iterations * (1 - refinement_tail)."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MAX_N_GAUSSIANS", "GRUT_SFM_POINT_MULTIPLIER",
                "GRUT_PER_FRAME_GAUSSIAN_BUDGET", "GRUT_MAX_N_GAUSSIANS_FLOOR",
                "GRUT_MAX_N_GAUSSIANS_CEILING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GRUT_REFINEMENT_TAIL_RATIO", "0.20")

    _, end_iter, _ = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=15000,
        n_iterations=10000,
        requested_max_n_gaussians=0,
    )
    # end_iter = int(10000 * (1.0 - 0.20)) = 8000
    assert end_iter == 8000


def test_resolve_max_n_gaussians_custom_env_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify env var overrides for multiplier, budget, floor, ceiling."""
    module = _load_nurec_shim_module()
    monkeypatch.delenv("ADAPTIVE_MAX_N_GAUSSIANS", raising=False)
    monkeypatch.setenv("GRUT_SFM_POINT_MULTIPLIER", "10.0")
    monkeypatch.setenv("GRUT_PER_FRAME_GAUSSIAN_BUDGET", "1000")
    monkeypatch.setenv("GRUT_MAX_N_GAUSSIANS_FLOOR", "50000")
    monkeypatch.setenv("GRUT_MAX_N_GAUSSIANS_CEILING", "500000")
    monkeypatch.setenv("GRUT_REFINEMENT_TAIL_RATIO", "0.10")

    resolved, end_iter, reason = module._resolve_effective_max_n_gaussians(
        video_duration_sec=51.0,
        registered_frame_count=134,
        sfm_point_count=15000,
        n_iterations=9000,
        requested_max_n_gaussians=0,
    )
    # sfm_signal = 15000 * 10 = 150000, frame_signal = 134 * 1000 = 134000
    # min(150000, 134000) = 134000, clamped to [50K, 500K] → 134000
    assert resolved == 134000
    assert end_iter == int(9000 * 0.90)  # 8100


# ---------------------------------------------------------------------------
# run_3dgrut_training: Hydra overrides
# ---------------------------------------------------------------------------


def test_run_3dgrut_training_passes_gaussian_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    threedgrut_dir = tmp_path / "3dgrut_src"
    train_script = threedgrut_dir / "train.py"
    train_script.parent.mkdir(parents=True, exist_ok=True)
    train_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(module, "THREEDGRUT_DIR", str(threedgrut_dir))

    output_dir = tmp_path / "output"
    undistorted_dir = tmp_path / "undistorted"
    output_dir.mkdir(parents=True, exist_ok=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        observed["cmd"] = list(cmd)
        result_root = output_dir / "3dgrut" / "nurec_scene"
        result_dir = result_root / "run"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "export_last.usdz").write_bytes(b"usdz")
        (result_dir / "export_last.ply").write_bytes(b"ply")
        (result_dir / "export_last.ingp").write_bytes(b"ingp")
        return None

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.run_3dgrut_training(
        undistorted_dir=undistorted_dir,
        output_dir=output_dir,
        n_iterations=9000,
        max_n_gaussians=268000,
        add_end_iteration=7650,
    )

    cmd = observed["cmd"]
    assert "strategy.add.max_n_gaussians=268000" in cmd
    assert "strategy.add.end_iteration=7650" in cmd
    assert result["max_n_gaussians"] == 268000
    assert result["add_end_iteration"] == 7650


def test_run_3dgrut_training_omits_overrides_when_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    threedgrut_dir = tmp_path / "3dgrut_src"
    train_script = threedgrut_dir / "train.py"
    train_script.parent.mkdir(parents=True, exist_ok=True)
    train_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(module, "THREEDGRUT_DIR", str(threedgrut_dir))

    output_dir = tmp_path / "output"
    undistorted_dir = tmp_path / "undistorted"
    output_dir.mkdir(parents=True, exist_ok=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        observed["cmd"] = list(cmd)
        result_root = output_dir / "3dgrut" / "nurec_scene"
        result_dir = result_root / "run"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "export_last.usdz").write_bytes(b"usdz")
        (result_dir / "export_last.ply").write_bytes(b"ply")
        (result_dir / "export_last.ingp").write_bytes(b"ingp")
        return None

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.run_3dgrut_training(
        undistorted_dir=undistorted_dir,
        output_dir=output_dir,
        n_iterations=9000,
        max_n_gaussians=0,
        add_end_iteration=0,
    )

    cmd = observed["cmd"]
    assert not any("strategy.add.max_n_gaussians" in str(c) for c in cmd)
    assert not any("strategy.add.end_iteration" in str(c) for c in cmd)
    assert result["max_n_gaussians"] == 0
    assert result["add_end_iteration"] == 0


# ---------------------------------------------------------------------------
# Resume validation: max_n_gaussians
# ---------------------------------------------------------------------------


def test_resolve_stage14_resume_rejects_max_n_gaussians_change(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    workspace = output_dir / "_colmap_workspace"
    frames_dir = workspace / "frames"
    sparse_dir = workspace / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "export_last.usdz").write_bytes(b"usdz")
    (output_dir / "export_last.ply").write_bytes(b"ply")
    for i in range(1, 11):
        (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
    (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", 10))

    metadata = {
        "schema_version": "v1",
        "quality_profile": "quality_first",
        "video": {"size_bytes": 123, "mtime_ns": 456},
        "stage1": {
            "frame_count": 10,
            "requested_max_frames": 450,
            "effective_max_frames": 450,
            "requested_extract_fps": 6,
            "effective_extract_fps": 6.0,
            "blur_filter": {
                "status": "ok",
                "keep_ratio": 0.85,
                "min_frames": 120,
            },
        },
        "stage2": {"registered_images": 10},
        "stage4": {
            "n_iterations": 12000,
            "max_n_gaussians_requested": 268000,
            "max_n_gaussians": 268000,
        },
    }
    (output_dir / module.STAGE14_RESUME_METADATA).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Request different max_n_gaussians → cache should be rejected
    ok, existing, reasons = module._resolve_stage14_resume(
        resume_requested=True,
        quality_guardrails=True,
        output_dir=output_dir,
        workspace=workspace,
        profile="quality_first",
        video_signature={"size_bytes": 123, "mtime_ns": 456},
        requested_max_frames=450,
        effective_max_frames=450,
        requested_extract_fps=6,
        effective_extract_fps=6.0,
        blur_filter_keep_ratio=0.85,
        blur_filter_min_frames=120,
        n_iterations=12000,
        max_n_gaussians=500000,  # CHANGED from 268000
    )

    assert ok is False
    assert "max_n_gaussians_changed" in reasons


def test_resolve_stage14_resume_accepts_matching_max_n_gaussians(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    workspace = output_dir / "_colmap_workspace"
    frames_dir = workspace / "frames"
    sparse_dir = workspace / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "export_last.usdz").write_bytes(b"usdz")
    (output_dir / "export_last.ply").write_bytes(b"ply")
    for i in range(1, 11):
        (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
    (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", 10))

    metadata = {
        "schema_version": "v1",
        "quality_profile": "quality_first",
        "video": {"size_bytes": 123, "mtime_ns": 456},
        "stage1": {
            "frame_count": 10,
            "requested_max_frames": 450,
            "effective_max_frames": 450,
            "requested_extract_fps": 6,
            "effective_extract_fps": 6.0,
            "blur_filter": {
                "status": "ok",
                "keep_ratio": 0.85,
                "min_frames": 120,
            },
        },
        "stage2": {"registered_images": 10},
        "stage4": {
            "n_iterations": 12000,
            "max_n_gaussians_requested": 268000,
            "max_n_gaussians": 268000,
        },
    }
    (output_dir / module.STAGE14_RESUME_METADATA).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    ok, existing, reasons = module._resolve_stage14_resume(
        resume_requested=True,
        quality_guardrails=True,
        output_dir=output_dir,
        workspace=workspace,
        profile="quality_first",
        video_signature={"size_bytes": 123, "mtime_ns": 456},
        requested_max_frames=450,
        effective_max_frames=450,
        requested_extract_fps=6,
        effective_extract_fps=6.0,
        blur_filter_keep_ratio=0.85,
        blur_filter_min_frames=120,
        n_iterations=12000,
        max_n_gaussians=268000,  # Same as cached
    )

    assert ok is True
    assert existing is not None
    assert reasons == ["metadata_match"]


def test_resolve_stage14_resume_accepts_adaptive_requested_with_cached_effective(tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir = tmp_path / "out"
    workspace = output_dir / "_colmap_workspace"
    frames_dir = workspace / "frames"
    sparse_dir = workspace / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "export_last.usdz").write_bytes(b"usdz")
    (output_dir / "export_last.ply").write_bytes(b"ply")
    for i in range(1, 11):
        (frames_dir / f"frame_{i:05d}.jpg").write_bytes(b"jpg")
    (sparse_dir / "images.bin").write_bytes(struct.pack("<Q", 10))

    metadata = {
        "schema_version": "v1",
        "quality_profile": "quality_first",
        "video": {"size_bytes": 123, "mtime_ns": 456},
        "stage1": {
            "frame_count": 10,
            "requested_max_frames": 450,
            "effective_max_frames": 450,
            "requested_extract_fps": 6,
            "effective_extract_fps": 6.0,
            "blur_filter": {
                "status": "ok",
                "keep_ratio": 0.85,
                "min_frames": 120,
            },
        },
        "stage2": {"registered_images": 10},
        "stage4": {
            "n_iterations": 12000,
            "max_n_gaussians_requested": 0,
            "max_n_gaussians": 268000,
        },
    }
    (output_dir / module.STAGE14_RESUME_METADATA).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    ok, existing, reasons = module._resolve_stage14_resume(
        resume_requested=True,
        quality_guardrails=True,
        output_dir=output_dir,
        workspace=workspace,
        profile="quality_first",
        video_signature={"size_bytes": 123, "mtime_ns": 456},
        requested_max_frames=450,
        effective_max_frames=450,
        requested_extract_fps=6,
        effective_extract_fps=6.0,
        blur_filter_keep_ratio=0.85,
        blur_filter_min_frames=120,
        n_iterations=12000,
        max_n_gaussians=0,
    )

    assert ok is True
    assert existing is not None
    assert reasons == ["metadata_match"]


# ---------------------------------------------------------------------------
# Quality profile defaults: max_n_gaussians
# ---------------------------------------------------------------------------


def test_quality_profile_defaults_set_max_n_gaussians() -> None:
    module = _load_nurec_shim_module()
    quality_first = module._quality_profile_defaults("quality_first")
    balanced = module._quality_profile_defaults("balanced")
    fast = module._quality_profile_defaults("fast")

    assert quality_first["max_n_gaussians"] == 0  # adaptive
    assert balanced["max_n_gaussians"] == 0  # adaptive
    assert fast["max_n_gaussians"] == 500_000  # fixed cap for speed


# ---------------------------------------------------------------------------
# Adaptive SfM retry: _resolve_effective_min_registered_ratio
# ---------------------------------------------------------------------------


def test_resolve_min_registered_ratio_relaxes_for_healthy_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """140 registered out of 224: above 100 absolute minimum → relax to 0.50."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MIN_REGISTERED_RATIO", "SFM_ABSOLUTE_MIN_FRAMES",
                "SFM_SMALL_SET_THRESHOLD", "SFM_RELAXED_RATIO"):
        monkeypatch.delenv(key, raising=False)

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=140,
        extracted_frames=224,
    )
    assert ratio == pytest.approx(0.50)
    assert "relaxed" in reason
    assert "registered=140" in reason


def test_resolve_min_registered_ratio_strict_for_low_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only 40 registered out of 224: below 100 minimum → keep strict 0.80."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MIN_REGISTERED_RATIO", "SFM_ABSOLUTE_MIN_FRAMES",
                "SFM_SMALL_SET_THRESHOLD", "SFM_RELAXED_RATIO"):
        monkeypatch.delenv(key, raising=False)

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=40,
        extracted_frames=224,
    )
    assert ratio == pytest.approx(0.80)
    assert "strict" in reason


def test_resolve_min_registered_ratio_strict_for_small_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small capture (30 frames): always strict regardless of absolute count."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MIN_REGISTERED_RATIO", "SFM_ABSOLUTE_MIN_FRAMES",
                "SFM_SMALL_SET_THRESHOLD", "SFM_RELAXED_RATIO"):
        monkeypatch.delenv(key, raising=False)

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=25,
        extracted_frames=30,
    )
    assert ratio == pytest.approx(0.80)
    assert "small_set" in reason


def test_resolve_min_registered_ratio_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ADAPTIVE_MIN_REGISTERED_RATIO=false, keep requested ratio."""
    module = _load_nurec_shim_module()
    monkeypatch.setenv("ADAPTIVE_MIN_REGISTERED_RATIO", "false")

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=140,
        extracted_frames=224,
    )
    assert ratio == pytest.approx(0.80)
    assert "disabled" in reason


def test_resolve_min_registered_ratio_custom_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom env overrides: lower absolute minimum, custom relaxed ratio."""
    module = _load_nurec_shim_module()
    monkeypatch.delenv("ADAPTIVE_MIN_REGISTERED_RATIO", raising=False)
    monkeypatch.setenv("SFM_ABSOLUTE_MIN_FRAMES", "50")
    monkeypatch.setenv("SFM_RELAXED_RATIO", "0.40")

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=60,
        extracted_frames=200,
    )
    assert ratio == pytest.approx(0.40)
    assert "relaxed" in reason


def test_resolve_min_registered_ratio_long_video_low_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long video: 1500 frames, only 80 registered → below absolute min → strict."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MIN_REGISTERED_RATIO", "SFM_ABSOLUTE_MIN_FRAMES",
                "SFM_SMALL_SET_THRESHOLD", "SFM_RELAXED_RATIO"):
        monkeypatch.delenv(key, raising=False)

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=80,
        extracted_frames=1500,
    )
    assert ratio == pytest.approx(0.80)
    assert "strict" in reason


def test_resolve_min_registered_ratio_long_video_healthy_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long video: 1500 frames, 800 registered → above 100 absolute → relax."""
    module = _load_nurec_shim_module()
    for key in ("ADAPTIVE_MIN_REGISTERED_RATIO", "SFM_ABSOLUTE_MIN_FRAMES",
                "SFM_SMALL_SET_THRESHOLD", "SFM_RELAXED_RATIO"):
        monkeypatch.delenv(key, raising=False)

    ratio, reason = module._resolve_effective_min_registered_ratio(
        requested_ratio=0.80,
        registered_images=800,
        extracted_frames=1500,
    )
    assert ratio == pytest.approx(0.50)
    assert "relaxed" in reason


def _cmd_arg(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def _setup_void_fill_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    output_dir = tmp_path / "out"
    workspace = tmp_path / "workspace"
    undistorted = workspace / "undistorted"
    renders_dir = output_dir / "stage4_result" / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    (undistorted / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    (renders_dir / "00000.png").write_bytes(b"render")
    ckpt = output_dir / "stage4_result" / "ckpt_last.pt"
    ckpt.write_bytes(b"ckpt")
    (undistorted / "sparse" / "0" / "points3D.bin").write_bytes(b"\x00")
    (undistorted / "sparse" / "0" / "images.txt").write_text("#\n", encoding="utf-8")

    active_ply = output_dir / "active.ply"
    active_usdz = output_dir / "active.usdz"
    active_ingp = output_dir / "active.ingp"
    active_ply.write_bytes(b"ply")
    active_usdz.write_bytes(b"usdz")
    active_ingp.write_bytes(b"ingp")
    grut_result = {
        "result_dir": str(output_dir / "stage4_result"),
        "metrics": {"mean_psnr": 30.0},
    }
    return output_dir, workspace, undistorted, {
        "active_ply": active_ply,
        "active_usdz": active_usdz,
        "active_ingp": active_ingp,
        "grut_result": grut_result,
    }


def test_void_fill_rejects_round_when_distill_not_ok_even_with_refined_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_nurec_shim_module()
    output_dir, workspace, undistorted, state = _setup_void_fill_fixture(tmp_path)

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        script = Path(cmd[1]).name
        if script == "post_stage4_gap_analyzer.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "gap_analysis_report.json").write_text(
                json.dumps({"global_hole_pixel_ratio": 0.8, "virtual_candidates_selected": 1}),
                encoding="utf-8",
            )
            (round_dir / "gap_candidate_views.jsonl").write_text(
                json.dumps(
                    {
                        "id": "v1",
                        "is_virtual": True,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_virtual_render.py":
            work_dir = Path(_cmd_arg(cmd, "--work-dir"))
            renders = work_dir / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            (renders / "00000.png").write_bytes(b"img")
            (work_dir / "virtual_render_mapping.jsonl").write_text(
                json.dumps(
                    {
                        "candidate_id": "v1",
                        "render_name": "00000.png",
                        "render_exists": True,
                        "render_image": str((renders / "00000.png").resolve()),
                        "predicted_hole_ratio": 0.2,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (work_dir / "virtual_render_report.json").write_text(
                json.dumps({"rendered_count": 1, "renders_dir": str(renders), "mapping_path": str(work_dir / "virtual_render_mapping.jsonl")}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_view_repair.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "accepted_repaired_views.jsonl").write_text(
                json.dumps(
                    {
                        "is_virtual": True,
                        "repaired_image": str(round_dir / "post_stage4_repaired_views" / "00000.png"),
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_distill.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "export_last_refined.ply").write_bytes(b"fallback-copy")
            (round_dir / "post_stage4_distill_report.json").write_text(
                json.dumps(
                    {
                        "status": "fallback_baseline_copy_distill_failed",
                        "distill_ok": False,
                        "virtual_appended_count": 1,
                        "refined_metrics": {"mean_psnr": 29.9},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    report = module._run_void_fill_loop(
        output_dir=output_dir,
        workspace=workspace,
        undistorted_dir=undistorted,
        active_gaussian_ply=state["active_ply"],
        active_visual_usdz=state["active_usdz"],
        active_ingp=state["active_ingp"],
        grut_result=state["grut_result"],
        void_fill_rounds=1,
        void_fill_distill_iters=10,
        void_fill_target_hole_ratio=0.05,
        max_n_gaussians=0,
        time_budget_min=1,
    )
    assert report["rounds"][0]["status"] == "rejected"
    assert report["rounds"][0]["rejection_reason"].startswith("distill_not_ok")
    assert report["best_ply"] == str(state["active_ply"])


def test_void_fill_threshold_filtering_stops_when_no_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir, workspace, undistorted, state = _setup_void_fill_fixture(tmp_path)

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        script = Path(cmd[1]).name
        if script == "post_stage4_gap_analyzer.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "gap_analysis_report.json").write_text(
                json.dumps({"global_hole_pixel_ratio": 0.9, "virtual_candidates_selected": 1}),
                encoding="utf-8",
            )
            (round_dir / "gap_candidate_views.jsonl").write_text(
                json.dumps({"id": "v2", "is_virtual": True, "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_virtual_render.py":
            work_dir = Path(_cmd_arg(cmd, "--work-dir"))
            renders = work_dir / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            (renders / "00000.png").write_bytes(b"img")
            (work_dir / "virtual_render_mapping.jsonl").write_text(
                json.dumps(
                    {
                        "candidate_id": "v2",
                        "render_name": "00000.png",
                        "render_exists": True,
                        "render_image": str((renders / "00000.png").resolve()),
                        "predicted_hole_ratio": 0.995,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (work_dir / "virtual_render_report.json").write_text(
                json.dumps({"rendered_count": 1, "renders_dir": str(renders), "mapping_path": str(work_dir / "virtual_render_mapping.jsonl")}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script in {"post_stage4_view_repair.py", "post_stage4_distill.py"}:
            raise AssertionError("view-repair/distill should not run when threshold filtering removes all candidates")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    report = module._run_void_fill_loop(
        output_dir=output_dir,
        workspace=workspace,
        undistorted_dir=undistorted,
        active_gaussian_ply=state["active_ply"],
        active_visual_usdz=state["active_usdz"],
        active_ingp=state["active_ingp"],
        grut_result=state["grut_result"],
        void_fill_rounds=1,
        void_fill_distill_iters=10,
        void_fill_target_hole_ratio=0.05,
        max_n_gaussians=0,
        time_budget_min=1,
    )
    assert report["rounds"][0]["status"] == "no_candidates_after_threshold"
    assert report["rounds"][0]["filtered_high_hole_count"] == 1


def test_void_fill_target_met_uses_probe_p90_after_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir, workspace, undistorted, state = _setup_void_fill_fixture(tmp_path)

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        script = Path(cmd[1]).name
        if script == "post_stage4_gap_analyzer.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "gap_analysis_report.json").write_text(
                json.dumps({"global_hole_pixel_ratio": 0.8, "virtual_candidates_selected": 1}),
                encoding="utf-8",
            )
            (round_dir / "gap_candidate_views.jsonl").write_text(
                json.dumps({"id": "v_tgt", "is_virtual": True, "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_virtual_render.py":
            work_dir = Path(_cmd_arg(cmd, "--work-dir"))
            renders = work_dir / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            (renders / "00000.png").write_bytes(b"img")
            mapping_path = work_dir / "virtual_render_mapping.jsonl"
            # Probe p90 should be below target (0.05), so loop should stop before repair/distill.
            mapping_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "v_tgt",
                        "render_name": "00000.png",
                        "render_exists": True,
                        "render_image": str((renders / "00000.png").resolve()),
                        "predicted_hole_ratio": 0.01,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (work_dir / "virtual_render_report.json").write_text(
                json.dumps({"rendered_count": 1, "renders_dir": str(renders), "mapping_path": str(mapping_path)}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script in {"post_stage4_view_repair.py", "post_stage4_distill.py"}:
            raise AssertionError("repair/distill should not run when probe p90 already meets target")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    report = module._run_void_fill_loop(
        output_dir=output_dir,
        workspace=workspace,
        undistorted_dir=undistorted,
        active_gaussian_ply=state["active_ply"],
        active_visual_usdz=state["active_usdz"],
        active_ingp=state["active_ingp"],
        grut_result=state["grut_result"],
        void_fill_rounds=1,
        void_fill_distill_iters=10,
        void_fill_target_hole_ratio=0.05,
        max_n_gaussians=0,
        time_budget_min=1,
    )
    assert report["rounds"][0]["status"] == "target_met"
    assert report["rounds"][0]["probe_hole_ratio_p90"] == pytest.approx(0.01, abs=1e-6)


def test_void_fill_no_virtual_candidates_does_not_mark_target_met(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_nurec_shim_module()
    output_dir, workspace, undistorted, state = _setup_void_fill_fixture(tmp_path)

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        script = Path(cmd[1]).name
        if script == "post_stage4_gap_analyzer.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            # Even if global hole ratio is below target, target_met must only come from probe p90.
            (round_dir / "gap_analysis_report.json").write_text(
                json.dumps({"global_hole_pixel_ratio": 0.01, "virtual_candidates_selected": 0}),
                encoding="utf-8",
            )
            (round_dir / "gap_candidate_views.jsonl").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script in {"post_stage4_virtual_render.py", "post_stage4_view_repair.py", "post_stage4_distill.py"}:
            raise AssertionError("virtual-render/repair/distill should not run with zero virtual candidates")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    report = module._run_void_fill_loop(
        output_dir=output_dir,
        workspace=workspace,
        undistorted_dir=undistorted,
        active_gaussian_ply=state["active_ply"],
        active_visual_usdz=state["active_usdz"],
        active_ingp=state["active_ingp"],
        grut_result=state["grut_result"],
        void_fill_rounds=1,
        void_fill_distill_iters=10,
        void_fill_target_hole_ratio=0.05,
        max_n_gaussians=0,
        time_budget_min=1,
    )
    assert report["rounds"][0]["status"] == "no_virtual_candidates"
    assert report["rounds"][0]["probe_hole_ratio_p90"] is None


def test_void_fill_passes_virtual_mapping_to_view_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_nurec_shim_module()
    output_dir, workspace, undistorted, state = _setup_void_fill_fixture(tmp_path)
    seen_repair_cmd: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        script = Path(cmd[1]).name
        if script == "post_stage4_gap_analyzer.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "gap_analysis_report.json").write_text(
                json.dumps({"global_hole_pixel_ratio": 0.9, "virtual_candidates_selected": 1}),
                encoding="utf-8",
            )
            (round_dir / "gap_candidate_views.jsonl").write_text(
                json.dumps({"id": "v3", "is_virtual": True, "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_virtual_render.py":
            work_dir = Path(_cmd_arg(cmd, "--work-dir"))
            renders = work_dir / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            (renders / "00000.png").write_bytes(b"img")
            mapping_path = work_dir / "virtual_render_mapping.jsonl"
            mapping_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "v3",
                        "render_name": "00000.png",
                        "render_exists": True,
                        "render_image": str((renders / "00000.png").resolve()),
                        "predicted_hole_ratio": 0.1,
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (work_dir / "virtual_render_report.json").write_text(
                json.dumps({"rendered_count": 1, "renders_dir": str(renders), "mapping_path": str(mapping_path)}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_view_repair.py":
            seen_repair_cmd["cmd"] = list(cmd)
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            repaired_dir = round_dir / "post_stage4_repaired_views"
            repaired_dir.mkdir(parents=True, exist_ok=True)
            repaired = repaired_dir / "00000.png"
            repaired.write_bytes(b"img")
            (round_dir / "accepted_repaired_views.jsonl").write_text(
                json.dumps(
                    {
                        "is_virtual": True,
                        "repaired_image": str(repaired),
                        "qvec": [1.0, 0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                        "camera_id": 4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_distill.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "export_last_refined.ply").write_bytes(b"ply")
            (round_dir / "post_stage4_distill_report.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "distill_ok": True,
                        "virtual_appended_count": 1,
                        "refined_metrics": {"mean_psnr": 29.8},
                        "result_dir": str(output_dir / "stage4_result"),
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    report = module._run_void_fill_loop(
        output_dir=output_dir,
        workspace=workspace,
        undistorted_dir=undistorted,
        active_gaussian_ply=state["active_ply"],
        active_visual_usdz=state["active_usdz"],
        active_ingp=state["active_ingp"],
        grut_result=state["grut_result"],
        void_fill_rounds=1,
        void_fill_distill_iters=10,
        void_fill_target_hole_ratio=0.05,
        max_n_gaussians=0,
        time_budget_min=1,
    )
    assert report["rounds"][0]["status"] == "ok"
    assert "cmd" in seen_repair_cmd
    repair_cmd = seen_repair_cmd["cmd"]
    assert "--virtual-render-mapping" in repair_cmd


def test_void_fill_continues_when_hole_low_but_virtual_candidates_remain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Void fill loop should NOT exit early when hole_ratio < target but virtual_count > 0."""
    module = _load_nurec_shim_module()
    output_dir, workspace, undistorted, state = _setup_void_fill_fixture(tmp_path)
    rounds_seen: list[int] = []

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        script = Path(cmd[1]).name
        if script == "post_stage4_gap_analyzer.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            round_num = len(rounds_seen) + 1
            rounds_seen.append(round_num)
            # Report low hole ratio BUT non-zero virtual candidates
            (round_dir / "gap_analysis_report.json").write_text(
                json.dumps({
                    "global_hole_pixel_ratio": 0.01,  # well below 0.05 target
                    "virtual_candidates_selected": 3,  # but 3 under-covered directions remain
                }),
                encoding="utf-8",
            )
            (round_dir / "gap_candidate_views.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": f"v{i}", "is_virtual": True, "qvec": [1, 0, 0, 0], "tvec": [0, float(i), 0]})
                    for i in range(3)
                ) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_virtual_render.py":
            work_dir = Path(_cmd_arg(cmd, "--work-dir"))
            renders = work_dir / "renders"
            renders.mkdir(parents=True, exist_ok=True)
            (renders / "00000.png").write_bytes(b"img")
            mapping_path = work_dir / "virtual_render_mapping.jsonl"
            mapping_path.write_text(
                json.dumps({
                    "candidate_id": "v0", "render_name": "00000.png",
                    "render_exists": True, "render_image": str(renders / "00000.png"),
                    "predicted_hole_ratio": 0.1, "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0],
                }) + "\n",
                encoding="utf-8",
            )
            (work_dir / "virtual_render_report.json").write_text(
                json.dumps({"rendered_count": 1, "renders_dir": str(renders), "mapping_path": str(mapping_path)}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_view_repair.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            repaired_dir = round_dir / "post_stage4_repaired_views"
            repaired_dir.mkdir(parents=True, exist_ok=True)
            repaired = repaired_dir / "00000.png"
            repaired.write_bytes(b"img")
            (round_dir / "accepted_repaired_views.jsonl").write_text(
                json.dumps({
                    "is_virtual": True, "repaired_image": str(repaired),
                    "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0], "camera_id": 4,
                }) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if script == "post_stage4_distill.py":
            round_dir = Path(_cmd_arg(cmd, "--output-dir"))
            (round_dir / "export_last_refined.ply").write_bytes(b"ply")
            (round_dir / "post_stage4_distill_report.json").write_text(
                json.dumps({
                    "status": "ok", "distill_ok": True,
                    "virtual_appended_count": 1, "refined_metrics": {"mean_psnr": 29.8},
                    "result_dir": str(output_dir / "stage4_result"),
                }),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", _fake_run)
    report = module._run_void_fill_loop(
        output_dir=output_dir,
        workspace=workspace,
        undistorted_dir=undistorted,
        active_gaussian_ply=state["active_ply"],
        active_visual_usdz=state["active_usdz"],
        active_ingp=state["active_ingp"],
        grut_result=state["grut_result"],
        void_fill_rounds=2,
        void_fill_distill_iters=10,
        void_fill_target_hole_ratio=0.05,
        max_n_gaussians=0,
        time_budget_min=1,
    )
    # With the fix, the loop should NOT stop at round 1 even though hole_ratio < target,
    # because virtual_count > 0 indicates there are still under-covered directions.
    assert len(rounds_seen) == 2, f"Expected 2 rounds but only ran {len(rounds_seen)}"
    assert report["rounds_completed"] == 2



def _write_binary_gaussian_ply(path: Path, *, n_vertices: int, payload: bytes = b"") -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float f_dc_0\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + payload)


def test_safe_read_point_cloud_rejects_excessive_vertex_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    np = pytest.importorskip("numpy")
    ply_path = tmp_path / "gaussian_large_vertices.ply"
    _write_binary_gaussian_ply(ply_path, n_vertices=10)

    monkeypatch.setenv("OPEN3D_GAUSSIAN_MAX_VERTICES", "5")
    with pytest.raises(ValueError, match="exceeds safety limit"):
        module._safe_read_point_cloud(SimpleNamespace(), np, ply_path)


def test_safe_read_point_cloud_rejects_excessive_data_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    np = pytest.importorskip("numpy")
    ply_path = tmp_path / "gaussian_large_payload.ply"
    _write_binary_gaussian_ply(ply_path, n_vertices=10)

    monkeypatch.setenv("OPEN3D_GAUSSIAN_MAX_DATA_BYTES", "100")
    with pytest.raises(ValueError, match="data size .* exceeds safety limit"):
        module._safe_read_point_cloud(SimpleNamespace(), np, ply_path)


def test_safe_read_point_cloud_rejects_truncated_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_nurec_shim_module()
    np = pytest.importorskip("numpy")
    ply_path = tmp_path / "gaussian_truncated.ply"
    # 2 vertices * 16 bytes each expected, but only provide 8 bytes.
    _write_binary_gaussian_ply(ply_path, n_vertices=2, payload=b"\x00" * 8)

    monkeypatch.setenv("OPEN3D_GAUSSIAN_MAX_VERTICES", "10")
    monkeypatch.setenv("OPEN3D_GAUSSIAN_MAX_DATA_BYTES", "1024")
    with pytest.raises(ValueError, match="payload truncated"):
        module._safe_read_point_cloud(SimpleNamespace(), np, ply_path)
