from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import adp009d_aura_native_conformance as subject
from blueprint_pipeline.adp009d_aura_native_vast import (
    EXPECTED_AURA_PLY_SHA256,
    PROBE_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TREE,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    probe_rows = []
    result_rows = []
    for index, camera_id in enumerate(("approach_close", "right_translate")):
        native = root / "native" / f"{camera_id}.png"
        native.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.full((16, 16, 3), 20 + index * 30, dtype=np.uint8)
        pixels[4:12, 4:12, index] = 200
        Image.fromarray(pixels).save(native)
        output = root / "execution" / camera_id
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output / "rgb.png")
        depth = np.full((16, 16), 1.0 + index, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        np.save(output / "depth_m.npy", depth, allow_pickle=False)
        np.save(output / "alpha.npy", alpha, allow_pickle=False)
        calibration = {
            "camera_model": "pinhole",
            "intrinsic_matrix": [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]],
            "world_from_camera": [
                [1.0, 0.0, 0.0, float(index)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "resolution": [16, 16],
            "camera_coordinate_convention": "OpenCV_right_down_forward",
        }
        calibration_digest = canonical_digest(calibration)
        native_digest = subject._sha256(native)
        probe_rows.append(
            {
                "camera_id": camera_id,
                "calibration": calibration,
                "calibration_digest": calibration_digest,
                "native_reference_path": str(native),
                "native_reference_sha256": native_digest,
            }
        )
        artifacts = []
        for name, dtype, shape in (
            ("rgb.png", "uint8", [16, 16, 3]),
            ("depth_m.npy", "float32", [16, 16]),
            ("alpha.npy", "float32", [16, 16]),
        ):
            path = output / name
            artifacts.append(
                {
                    "path": path.relative_to(root / "execution").as_posix(),
                    "sha256": subject._sha256(path),
                    "size_bytes": path.stat().st_size,
                    "dtype": dtype,
                    "shape": shape,
                }
            )
        result_rows.append(
            {
                "camera_id": camera_id,
                "valid": True,
                "calibration": calibration,
                "calibration_digest": calibration_digest,
                "native_reference_sha256": native_digest,
                "positive_finite_depth_count": 256,
                "artifacts": artifacts,
            }
        )
    probe = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "materialized_unexecuted",
        "aura_ply_sha256": EXPECTED_AURA_PLY_SHA256,
        "aura_native_render_manifest_digest": "sha256:" + "d" * 64,
        "conformance_thresholds": subject.FROZEN_THRESHOLDS,
        "threshold_definition_commit": subject.THRESHOLD_DEFINITION_COMMIT,
        "thresholds_frozen_before_execution": True,
        "renderer_outcomes_observed_before_freeze": False,
        "camera_configs": probe_rows,
    }
    probe["manifest_digest"] = canonical_digest(probe, digest_field="manifest_digest")
    probe_path = root / "probe.json"
    _write_json(probe_path, probe)
    bundle = {
        "status": "ready",
        "source_probe_manifest_digest": probe["manifest_digest"],
        "conformance_thresholds": subject.FROZEN_THRESHOLDS,
        "threshold_definition_commit": subject.THRESHOLD_DEFINITION_COMMIT,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "aura_ply_sha256": EXPECTED_AURA_PLY_SHA256,
        "input_digest": "sha256:" + "e" * 64,
        "bundle_sha256": "sha256:" + "f" * 64,
    }
    bundle_path = root / "bundle.json"
    _write_json(bundle_path, bundle)
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "blockers": [],
        "input_digest": bundle["input_digest"],
        "source_probe_manifest_digest": probe["manifest_digest"],
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_modified": False,
        "aura_ply_sha256": EXPECTED_AURA_PLY_SHA256,
        "depth_output": "surf_depth_expected_camera_z_m",
        "depth_ratio": 0.0,
        "metric_scene_units": "meters",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "camera_rows": result_rows,
    }
    result_path = root / "execution/result.json"
    _write_json(result_path, result)
    return probe_path, bundle_path, result_path


def test_native_conformance_derives_perfect_rgb_and_metric_depth_receipt(
    tmp_path: Path,
) -> None:
    probe, bundle, result = _fixture(tmp_path)

    receipt = subject.materialize_aura_native_conformance_receipt(
        probe_manifest_path=probe,
        provider_bundle_receipt_path=bundle,
        native_result_path=result,
        evidence_root=tmp_path,
        output_path=tmp_path / "receipt.json",
    )

    assert receipt["passed"] is True
    assert receipt["aggregate"]["mean_psnr_db"] == "infinity"
    assert receipt["aggregate"]["mean_windowed_ssim"] == 1.0
    assert all(row["positive_finite_depth_count"] == 256 for row in receipt["rows"])
    assert subject.validate_aura_native_conformance_receipt(receipt) == receipt


def test_native_conformance_rejects_nonfinite_metric_depth(tmp_path: Path) -> None:
    probe, bundle, result_path = _fixture(tmp_path)
    depth_path = tmp_path / "execution/approach_close/depth_m.npy"
    depth = np.load(depth_path, allow_pickle=False)
    depth[0, 0] = np.nan
    np.save(depth_path, depth, allow_pickle=False)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in result["camera_rows"][0]["artifacts"]
        if item["path"].endswith("depth_m.npy")
    )
    artifact["sha256"] = subject._sha256(depth_path)
    artifact["size_bytes"] = depth_path.stat().st_size
    _write_json(result_path, result)

    with pytest.raises(
        subject.AuraNativeConformanceError,
        match="metric_depth_invalid",
    ):
        subject.materialize_aura_native_conformance_receipt(
            probe_manifest_path=probe,
            provider_bundle_receipt_path=bundle,
            native_result_path=result_path,
            evidence_root=tmp_path,
            output_path=tmp_path / "receipt.json",
        )
