from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from blueprint_pipeline.clip_curation_stage import (
    ClipCurationConfig,
    GATE_CAMERA_STABILITY,
    GATE_CONTENT_NOVELTY,
    GATE_EXPOSURE,
    GATE_MIN_FRAMES,
    GATE_SHARPNESS,
    GATE_STATUS_FAILED,
    GATE_STATUS_NOT_MEASURABLE,
    GATE_STATUS_PASSED,
    curate_clips,
    evaluate_clip,
    laplacian_variance,
    run_clip_curation_stage,
)
from blueprint_pipeline.common import PipelineError


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _pose_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> List[List[float]]:
    yaw = math.radians(yaw_deg)
    mat = np.eye(4)
    mat[0, 0] = math.cos(yaw)
    mat[0, 2] = math.sin(yaw)
    mat[2, 0] = -math.sin(yaw)
    mat[2, 2] = math.cos(yaw)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    return mat.tolist()


def _frames(
    n: int,
    *,
    step_x: float = 0.02,
    jitter_x: float = 0.0,
    yaw_step_deg: float = 0.0,
    image_path: Optional[str] = None,
    sharpness: Optional[float] = None,
    with_poses: bool = True,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for i in range(n):
        frame: Dict[str, Any] = {"frame_id": f"{i:06d}", "timestamp": i / 30.0}
        if with_poses:
            x = step_x * i + jitter_x * ((-1) ** i)
            frame["T_world_camera"] = _pose_matrix(x=x, yaw_deg=yaw_step_deg * i)
        if image_path is not None:
            frame["image_path"] = image_path
        if sharpness is not None:
            frame["sharpness_score"] = sharpness
        frames.append(frame)
    return frames


def _sharp_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(20, 236, size=(32, 32)).astype(np.float64)


def _blurry_image() -> np.ndarray:
    # Linear ramp: Laplacian is ~0 everywhere -> near-zero variance.
    return np.tile(np.linspace(20.0, 235.0, 32), (32, 1))


def _bright_checkerboard() -> np.ndarray:
    # Sharp (high Laplacian variance) but every pixel >= 247 (clipped).
    board = np.indices((32, 32)).sum(axis=0) % 2
    return np.where(board == 0, 249.0, 252.0)


def _dark_checkerboard() -> np.ndarray:
    # Sharp but every pixel <= 8 (crushed).
    board = np.indices((32, 32)).sum(axis=0) % 2
    return np.where(board == 0, 2.0, 5.0)


def _write_image(bundle_dir: Path, rel: str, image: np.ndarray) -> str:
    path = bundle_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image)
    return rel


def _write_bundle(bundle_dir: Path, clips: List[Dict[str, Any]]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "clips_manifest.json").write_text(
        json.dumps({"clips": clips}), encoding="utf-8"
    )


def _good_clip(bundle_dir: Path, clip_id: str = "clip_good", n: int = 80) -> Dict[str, Any]:
    rel = _write_image(bundle_dir, f"frames/{clip_id}_sharp.npy", _sharp_image())
    return {"clip_id": clip_id, "frames": _frames(n, image_path=rel)}


# ---------------------------------------------------------------------------
# Gate: minimum clip length
# ---------------------------------------------------------------------------


def test_short_clip_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())
    short_clip = {"clip_id": "clip_short", "frames": _frames(10, image_path=rel)}

    result = evaluate_clip(short_clip, config=ClipCurationConfig(), bundle_dir=bundle)
    assert result["status"] == "rejected"
    gate = result["gate_results"][GATE_MIN_FRAMES]
    assert gate["status"] == GATE_STATUS_FAILED
    assert gate["value"] == 10
    assert gate["threshold"] == 70
    assert any(GATE_MIN_FRAMES in reason for reason in result["rejection_reasons"])

    long_clip = {"clip_id": "clip_long", "frames": _frames(80, image_path=rel)}
    result = evaluate_clip(long_clip, config=ClipCurationConfig(), bundle_dir=bundle)
    assert result["gate_results"][GATE_MIN_FRAMES]["status"] == GATE_STATUS_PASSED
    assert result["status"] == "accepted"


# ---------------------------------------------------------------------------
# Gate: sharpness — stamped constants are never trusted
# ---------------------------------------------------------------------------


def test_stamped_constant_sharpness_is_unmeasured_and_fails_closed(tmp_path: Path) -> None:
    # Geometry/video lanes stamp sharpness_score=100.0 on every frame.
    clip = {"clip_id": "clip_stamped", "frames": _frames(80, sharpness=100.0)}

    result = evaluate_clip(clip, config=ClipCurationConfig(), bundle_dir=tmp_path)
    gate = result["gate_results"][GATE_SHARPNESS]
    assert gate["status"] == GATE_STATUS_NOT_MEASURABLE
    assert gate["fail_closed"] is True
    assert gate["detail"]["metadata_classification"] == "metadata_stamped_constant"
    assert result["status"] == "rejected"
    assert any(GATE_SHARPNESS in reason for reason in result["rejection_reasons"])


def test_stamped_blur_zero_constant_is_unmeasured(tmp_path: Path) -> None:
    frames = _frames(80)
    for i, frame in enumerate(frames):
        frame["sharpness_score"] = 50.0 + i  # varied, would otherwise be trusted
        frame["blur_score"] = 0.0  # stamped constant from video lane
    clip = {"clip_id": "clip_blur_stamped", "frames": frames}
    result = evaluate_clip(clip, config=ClipCurationConfig(), bundle_dir=tmp_path)
    assert result["gate_results"][GATE_SHARPNESS]["status"] == GATE_STATUS_NOT_MEASURABLE


def test_constant_metadata_remeasured_from_images_when_available(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())
    clip = {
        "clip_id": "clip_stamped_with_images",
        "frames": _frames(80, image_path=rel, sharpness=100.0),
    }
    result = evaluate_clip(clip, config=ClipCurationConfig(), bundle_dir=bundle)
    gate = result["gate_results"][GATE_SHARPNESS]
    assert gate["status"] == GATE_STATUS_PASSED
    assert gate["detail"]["source"] == "measured_laplacian_variance"
    assert gate["value"] != 100.0  # the stamped constant never survives


def test_unmeasured_sharpness_allowed_when_config_opts_in(tmp_path: Path) -> None:
    clip = {"clip_id": "clip_stamped", "frames": _frames(80, sharpness=100.0)}
    config = ClipCurationConfig(
        allow_unmeasured_sharpness=True,
        allow_unmeasured_exposure=True,
    )
    result = evaluate_clip(clip, config=config, bundle_dir=tmp_path)
    assert result["gate_results"][GATE_SHARPNESS]["fail_closed"] is False
    assert result["status"] == "accepted"


def test_varied_metadata_sharpness_is_trusted(tmp_path: Path) -> None:
    frames = _frames(80)
    for i, frame in enumerate(frames):
        frame["sharpness_score"] = 60.0 + (i % 7)
    clip = {"clip_id": "clip_metadata", "frames": frames}
    config = ClipCurationConfig(allow_unmeasured_exposure=True)
    result = evaluate_clip(clip, config=config, bundle_dir=tmp_path)
    gate = result["gate_results"][GATE_SHARPNESS]
    assert gate["status"] == GATE_STATUS_PASSED
    assert gate["detail"]["source"] == "metadata"


# ---------------------------------------------------------------------------
# Gate: sharpness — real Laplacian measurement on synthetic images
# ---------------------------------------------------------------------------


def test_blurry_image_rejected_and_sharp_image_passes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sharp_rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())
    blurry_rel = _write_image(bundle, "frames/blurry.npy", _blurry_image())

    assert laplacian_variance(_sharp_image()) > 40.0
    assert laplacian_variance(_blurry_image()) < 40.0

    sharp_clip = {"clip_id": "clip_sharp", "frames": _frames(80, image_path=sharp_rel)}
    blurry_clip = {"clip_id": "clip_blurry", "frames": _frames(80, image_path=blurry_rel)}
    _write_bundle(bundle, [sharp_clip, blurry_clip])

    manifest = curate_clips([sharp_clip, blurry_clip], bundle_dir=bundle)
    by_id = {c["clip_id"]: c for c in manifest["clips"]}
    assert by_id["clip_sharp"]["status"] == "accepted"
    assert by_id["clip_blurry"]["status"] == "rejected"
    blurry_gate = by_id["clip_blurry"]["gate_results"][GATE_SHARPNESS]
    assert blurry_gate["status"] == GATE_STATUS_FAILED
    assert blurry_gate["detail"]["source"] == "measured_laplacian_variance"
    assert blurry_gate["value"] < blurry_gate["threshold"]


# ---------------------------------------------------------------------------
# Gate: exposure
# ---------------------------------------------------------------------------


def test_overexposed_and_underexposed_clips_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    bright_rel = _write_image(bundle, "frames/bright.npy", _bright_checkerboard())
    dark_rel = _write_image(bundle, "frames/dark.npy", _dark_checkerboard())

    for clip_id, rel in (("clip_over", bright_rel), ("clip_under", dark_rel)):
        clip = {"clip_id": clip_id, "frames": _frames(80, image_path=rel)}
        result = evaluate_clip(clip, config=ClipCurationConfig(), bundle_dir=bundle)
        gate = result["gate_results"][GATE_EXPOSURE]
        assert gate["status"] == GATE_STATUS_FAILED, clip_id
        assert gate["value"] > gate["threshold"]
        # Checkerboards are sharp: the rejection is exposure-specific.
        assert result["gate_results"][GATE_SHARPNESS]["status"] == GATE_STATUS_PASSED
        assert result["status"] == "rejected"


def test_well_exposed_sharp_clip_passes_exposure(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    clip = _good_clip(bundle)
    result = evaluate_clip(clip, config=ClipCurationConfig(), bundle_dir=bundle)
    assert result["gate_results"][GATE_EXPOSURE]["status"] == GATE_STATUS_PASSED
    assert result["status"] == "accepted"


# ---------------------------------------------------------------------------
# Gate: camera stability
# ---------------------------------------------------------------------------


def test_jittery_trajectory_rejected_and_smooth_passes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())

    smooth = {"clip_id": "clip_smooth", "frames": _frames(80, image_path=rel)}
    jittery = {
        "clip_id": "clip_jittery",
        "frames": _frames(80, jitter_x=0.03, image_path=rel),
    }
    config = ClipCurationConfig()

    smooth_result = evaluate_clip(smooth, config=config, bundle_dir=bundle)
    assert smooth_result["gate_results"][GATE_CAMERA_STABILITY]["status"] == GATE_STATUS_PASSED

    jitter_result = evaluate_clip(jittery, config=config, bundle_dir=bundle)
    gate = jitter_result["gate_results"][GATE_CAMERA_STABILITY]
    assert gate["status"] == GATE_STATUS_FAILED
    assert gate["value"] > config.max_pose_jitter_m
    assert jitter_result["status"] == "rejected"


def test_missing_poses_fail_closed_unless_allowed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())
    clip = {"clip_id": "clip_no_poses", "frames": _frames(80, with_poses=False, image_path=rel)}

    result = evaluate_clip(clip, config=ClipCurationConfig(), bundle_dir=bundle)
    stability = result["gate_results"][GATE_CAMERA_STABILITY]
    assert stability["status"] == GATE_STATUS_NOT_MEASURABLE
    assert stability["fail_closed"] is True
    assert result["status"] == "rejected"

    permissive = ClipCurationConfig(
        allow_unmeasured_stability=True,
        allow_unmeasured_novelty=True,
    )
    result = evaluate_clip(clip, config=permissive, bundle_dir=bundle)
    assert result["status"] == "accepted"


def test_robot_pov_static_camera_constraint(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())

    moving = {
        "clip_id": "clip_pov_moving",
        "clip_kind": "robot_pov",
        "frames": _frames(80, step_x=0.02, image_path=rel),
    }
    static = {
        "clip_id": "clip_pov_static",
        "clip_kind": "robot_pov",
        "frames": _frames(80, step_x=0.0, image_path=rel),
    }
    config = ClipCurationConfig()
    moving_result = evaluate_clip(moving, config=config, bundle_dir=bundle)
    assert moving_result["gate_results"][GATE_CAMERA_STABILITY]["status"] == GATE_STATUS_FAILED
    static_result = evaluate_clip(static, config=config, bundle_dir=bundle)
    assert static_result["gate_results"][GATE_CAMERA_STABILITY]["status"] == GATE_STATUS_PASSED
    # Walkthrough novelty gate does not apply to robot_pov clips.
    assert static_result["gate_results"][GATE_CONTENT_NOVELTY]["status"] == "skipped"
    assert static_result["status"] == "accepted"


# ---------------------------------------------------------------------------
# Gate: content novelty
# ---------------------------------------------------------------------------


def test_content_free_clip_rejected_and_rotating_pan_kept(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())

    static = {"clip_id": "clip_static", "frames": _frames(80, step_x=0.0, image_path=rel)}
    result = evaluate_clip(static, config=ClipCurationConfig(), bundle_dir=bundle)
    gate = result["gate_results"][GATE_CONTENT_NOVELTY]
    assert gate["status"] == GATE_STATUS_FAILED
    assert result["status"] == "rejected"

    pan = {
        "clip_id": "clip_pan",
        "frames": _frames(80, step_x=0.0, yaw_step_deg=1.0, image_path=rel),
    }
    result = evaluate_clip(pan, config=ClipCurationConfig(), bundle_dir=bundle)
    gate = result["gate_results"][GATE_CONTENT_NOVELTY]
    assert gate["status"] == GATE_STATUS_PASSED
    assert gate["detail"]["view_direction_spread_deg"] >= 15.0
    assert result["status"] == "accepted"


# ---------------------------------------------------------------------------
# Rejection manifest + stage entry point
# ---------------------------------------------------------------------------


def test_stage_writes_rejection_manifest_with_counts_and_reasons(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())
    blurry_rel = _write_image(bundle, "frames/blurry.npy", _blurry_image())

    clips = [
        _good_clip(bundle, "clip_good"),
        {"clip_id": "clip_short", "frames": _frames(10, image_path=rel)},
        {"clip_id": "clip_blurry", "frames": _frames(80, image_path=blurry_rel)},
    ]
    _write_bundle(bundle, clips)

    result = run_clip_curation_stage(bundle_dir=bundle)
    assert result["status"] == "completed"
    assert result["accepted_clip_ids"] == ["clip_good"]
    assert result["input_clip_count"] == 3
    assert result["rejected_clip_count"] == 2

    manifest_path = Path(result["manifest_path"])
    rejection_path = Path(result["rejection_manifest_path"])
    assert manifest_path.is_file()
    assert rejection_path.is_file()
    # Derived artifacts land under derived/, never next to raw inputs.
    assert "derived" in manifest_path.parts

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "clip_curation_manifest.v1"
    assert manifest["config"]["min_clip_frames"] == 70
    counts = manifest["rejection_manifest"]["gate_rejection_counts"]
    assert counts[GATE_MIN_FRAMES] == 1
    assert counts[GATE_SHARPNESS] == 1
    rejected = {r["clip_id"]: r for r in manifest["rejection_manifest"]["rejected_clips"]}
    assert set(rejected) == {"clip_short", "clip_blurry"}
    assert rejected["clip_short"]["rejection_reasons"]

    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    assert rejection["gate_rejection_counts"] == counts

    # Raw inputs untouched.
    original = json.loads((bundle / "clips_manifest.json").read_text(encoding="utf-8"))
    assert len(original["clips"]) == 3


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_thresholds_are_config_driven(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/sharp.npy", _sharp_image())
    clip = {"clip_id": "clip_short_ok", "frames": _frames(12, image_path=rel)}
    _write_bundle(bundle, [clip])

    # Default floor (70) rejects the 12-frame clip.
    default_result = run_clip_curation_stage(bundle_dir=bundle, output_dir=tmp_path / "out_a")
    assert default_result["accepted_clip_ids"] == []

    # Lowered floor accepts it (also lower the travel floor for the shorter walk).
    config = ClipCurationConfig.from_dict({"min_clip_frames": 5, "min_pose_travel_m": 0.1})
    tuned_result = run_clip_curation_stage(
        bundle_dir=bundle, config=config, output_dir=tmp_path / "out_b"
    )
    assert tuned_result["accepted_clip_ids"] == ["clip_short_ok"]


def test_config_loadable_from_yaml_and_json(tmp_path: Path) -> None:
    yaml_path = tmp_path / "thresholds.yaml"
    yaml_path.write_text(
        "min_clip_frames: 42\nmax_clipped_pixel_fraction: 0.5\n", encoding="utf-8"
    )
    config = ClipCurationConfig.from_file(yaml_path)
    assert config.min_clip_frames == 42
    assert config.max_clipped_pixel_fraction == 0.5

    json_path = tmp_path / "thresholds.json"
    json_path.write_text(json.dumps({"min_sharpness_laplacian_var": 55.0}), encoding="utf-8")
    config = ClipCurationConfig.from_file(json_path)
    assert config.min_sharpness_laplacian_var == 55.0
    # Untouched keys keep documented OSCAR-referencing defaults.
    assert config.min_clip_frames == 70


def test_unknown_config_key_rejected() -> None:
    with pytest.raises(PipelineError, match="Unknown clip curation config"):
        ClipCurationConfig.from_dict({"min_clip_frames": 70, "made_up_knob": 1})
