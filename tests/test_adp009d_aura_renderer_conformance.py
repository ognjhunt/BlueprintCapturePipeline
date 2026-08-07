from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.adp009d_aura_renderer_conformance import (
    FROZEN_THRESHOLDS,
    OVRTX_REPOSITORY,
    OVRTX_REVISION,
    REQUEST_SCHEMA_VERSION,
    AuraRendererConformanceError,
    evaluate_aura_renderer_conformance,
    validate_aura_renderer_conformance_receipt,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration() -> dict:
    return {
        "camera_model": "pinhole",
        "intrinsic_matrix": [
            [20.0, 0.0, 7.5],
            [0.0, 20.0, 7.5],
            [0.0, 0.0, 1.0],
        ],
        "world_from_camera": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "resolution": [16, 16],
    }


def _request(tmp_path: Path, *, matching: bool = True) -> dict:
    pairs = []
    for index in range(2):
        native = np.zeros((16, 16, 3), dtype=np.uint8)
        native[:, :, index] = np.arange(16, dtype=np.uint8)[None, :] * 12
        ovrtx = native.copy() if matching else np.flip(native, axis=1).copy()
        native_path = tmp_path / f"native_{index}.png"
        ovrtx_path = tmp_path / f"ovrtx_{index}.png"
        Image.fromarray(native).save(native_path)
        Image.fromarray(ovrtx).save(ovrtx_path)
        pairs.append(
            {
                "camera_id": f"exact_{index}",
                "native_frame_path": native_path.name,
                "native_frame_sha256": _sha256(native_path),
                "ovrtx_frame_path": ovrtx_path.name,
                "ovrtx_frame_sha256": _sha256(ovrtx_path),
                "native_calibration": _calibration(),
                "ovrtx_calibration": _calibration(),
            }
        )
    value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": "frozen_before_ovrtx_execution",
        "thresholds_frozen_before_ovrtx_execution": True,
        "ovrtx_outcomes_observed_before_freeze": False,
        "ovrtx_repository": OVRTX_REPOSITORY,
        "ovrtx_revision": OVRTX_REVISION,
        "thresholds": FROZEN_THRESHOLDS,
        "aura_particlefield_sha256": "sha256:" + "a" * 64,
        "aura_source_ply_sha256": "sha256:" + "b" * 64,
        "aura_native_render_manifest_digest": "sha256:" + "c" * 64,
        "ovrtx_run_input_digest": "sha256:" + "d" * 64,
        "evidence_root": str(tmp_path),
        "pairs": pairs,
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def test_exact_camera_renderer_conformance_passes_matching_bytes(tmp_path: Path) -> None:
    receipt = evaluate_aura_renderer_conformance(_request(tmp_path))

    assert receipt["status"] == "passed_exact_camera_conformance"
    assert receipt["aggregate"]["mean_psnr_db"] == "infinity"
    assert receipt["policy_observation_admitted_by_this_receipt_alone"] is False
    assert validate_aura_renderer_conformance_receipt(receipt) == receipt


def test_exact_camera_renderer_conformance_rejects_structurally_wrong_render(
    tmp_path: Path,
) -> None:
    receipt = evaluate_aura_renderer_conformance(_request(tmp_path, matching=False))

    assert receipt["status"] == "rejected_exact_camera_conformance"
    assert receipt["passed"] is False
    with pytest.raises(AuraRendererConformanceError, match="not_passed"):
        validate_aura_renderer_conformance_receipt(receipt)


def test_exact_camera_renderer_conformance_rejects_camera_or_digest_substitution(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request["pairs"][0]["ovrtx_calibration"]["world_from_camera"][0][3] = 0.1
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    with pytest.raises(AuraRendererConformanceError, match="camera_mismatch"):
        evaluate_aura_renderer_conformance(request)

    request = _request(tmp_path)
    request["pairs"][0]["ovrtx_frame_sha256"] = "sha256:" + "e" * 64
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    with pytest.raises(AuraRendererConformanceError, match="frame_digest_mismatch"):
        evaluate_aura_renderer_conformance(request)


def test_exact_camera_renderer_conformance_rejects_posthoc_threshold_change(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request["thresholds"] = copy.deepcopy(FROZEN_THRESHOLDS)
    request["thresholds"]["minimum_mean_psnr_db"] = 0.0
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(AuraRendererConformanceError, match="thresholds_not_code_frozen"):
        evaluate_aura_renderer_conformance(request)
