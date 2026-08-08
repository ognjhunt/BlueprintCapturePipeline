from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image

from blueprint_pipeline import adp009d_aura_frame_comparison as comparison


def _sha(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_probe(root, *, rgb: np.ndarray, semantic: np.ndarray) -> None:
    camera = root / "camera_frames/external_camera"
    camera.mkdir(parents=True)
    rgb_path = camera / "000040.png"
    semantic_path = camera / "000040.semantic.npy"
    depth_path = camera / "000040.distance_to_camera.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(semantic_path, semantic, allow_pickle=False)
    np.save(depth_path, np.ones(semantic.shape, dtype=np.float32), allow_pickle=False)
    probe = {
        "schema_version": "adp009d_frames_only_probe.v1",
        "status": "completed",
        "mode": "frames_only",
        "camera_rows": [
            {
                "camera_id": "external_camera",
                "frame_index": 40,
                "intrinsic_matrix": [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
                "position_world_m": [1.0, 2.0, 3.0],
                "quaternion_world_opengl_xyzw": [0.0, 0.0, 0.0, 1.0],
                "resolution_hw": list(rgb.shape[:2]),
                "sim_time_seconds": 2.0,
                "rgb_png": {"path": str(rgb_path.relative_to(root)), "sha256": _sha(rgb_path)},
                "metric_depth": {"path": str(depth_path.relative_to(root)), "sha256": _sha(depth_path)},
                "semantic_segmentation": {
                    "path": str(semantic_path.relative_to(root)),
                    "sha256": _sha(semantic_path),
                },
            }
        ],
    }
    (root / comparison.PROBE_FILENAME).write_text(
        json.dumps(probe), encoding="utf-8"
    )


def test_rejects_candidate_with_absent_appearance_but_never_names_quality_winner(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    semantic = np.zeros((4, 5), dtype=np.int32)
    semantic[0, 0] = 2
    baseline_rgb = np.full((4, 5, 3), 180, dtype=np.uint8)
    candidate_rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    candidate_rgb[0, 0] = 180
    _write_probe(baseline, rgb=baseline_rgb, semantic=semantic)
    _write_probe(candidate, rgb=candidate_rgb, semantic=semantic)

    receipt = comparison.compare_variants(
        baseline_root=baseline, candidate_root=candidate
    )

    assert receipt["status"] == "completed"
    assert receipt["held_constant"] is True
    assert receipt["candidate_appearance_absent"] is True
    assert receipt["operational_decision"] == (
        "retain_baseline_reject_candidate_as_drop_in"
    )
    assert receipt["quality_winner"] is None
    assert receipt["supports_quality_winner_claim"] is False
    assert receipt["blockers"] == ["independent_quality_reference_missing"]


def test_digest_drift_is_rejected(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    semantic = np.zeros((2, 2), dtype=np.int32)
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    _write_probe(baseline, rgb=rgb, semantic=semantic)
    _write_probe(candidate, rgb=rgb, semantic=semantic)
    (candidate / "camera_frames/external_camera/000040.png").write_bytes(b"drift")

    try:
        comparison.compare_variants(
            baseline_root=baseline, candidate_root=candidate
        )
    except ValueError as exc:
        assert str(exc) == "frames_only_bound_rgb_digest_mismatch"
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("digest drift was accepted")
