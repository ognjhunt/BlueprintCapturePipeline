from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.adp009d_aura_renderer_conformance import (
    FROZEN_THRESHOLDS,
    OVRTX_REPOSITORY,
    OVRTX_REVISION,
    REQUEST_SCHEMA_VERSION,
    THRESHOLD_DEFINITION_COMMIT,
    AuraRendererConformanceError,
    evaluate_aura_renderer_conformance,
    materialize_aura_renderer_conformance_request,
    materialize_aura_ovrtx_failure_diagnostic,
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
        "status": "materialized_from_prospective_probe_and_observed_outputs",
        "thresholds_frozen_before_ovrtx_execution": True,
        "ovrtx_outcomes_observed_before_threshold_freeze": False,
        "threshold_definition_commit": THRESHOLD_DEFINITION_COMMIT,
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


def test_conformance_request_materializer_joins_prospective_probe_to_real_bytes(
    tmp_path: Path,
) -> None:
    base = _request(tmp_path)
    particlefield = tmp_path / "aura.usdc"
    particlefield.write_bytes(b"particlefield")
    particle_receipt = {
        "status": "completed",
        "output_sha256": _sha256(particlefield),
        "source_sha256": "sha256:" + "b" * 64,
    }
    particle_receipt["receipt_digest"] = canonical_digest(
        particle_receipt, digest_field="receipt_digest"
    )
    particle_receipt_path = tmp_path / "particle_receipt.json"
    particle_receipt_path.write_text(json.dumps(particle_receipt), encoding="utf-8")
    probe_rows = []
    result_rows = []
    for pair in base["pairs"]:
        camera_id = pair["camera_id"]
        calibration_digest = canonical_digest(pair["native_calibration"])
        probe_rows.append(
            {
                "camera_id": camera_id,
                "calibration": pair["native_calibration"],
                "calibration_digest": calibration_digest,
                "native_reference_path": str(tmp_path / pair["native_frame_path"]),
                "native_reference_sha256": pair["native_frame_sha256"],
                "native_source_sha256": pair["native_frame_sha256"],
            }
        )
        output = tmp_path / "outputs" / camera_id / "rgb.png"
        output.parent.mkdir(parents=True)
        output.write_bytes((tmp_path / pair["ovrtx_frame_path"]).read_bytes())
        result_rows.append(
            {
                "camera_id": camera_id,
                "valid": True,
                "timed_out": False,
                "artifacts": [
                    {
                        "path": output.relative_to(tmp_path).as_posix(),
                        "sha256": _sha256(output),
                        "size_bytes": output.stat().st_size,
                    }
                ],
            }
        )
    probe = {
        "schema_version": "adp009d_ovrtx_live_camera_probe.v1",
        "status": "materialized_unexecuted",
        "probe_purpose": "aura_ovrtx_exact_camera_visual_conformance",
        "thresholds_frozen_before_ovrtx_execution": True,
        "ovrtx_outcomes_observed_before_freeze": False,
        "conformance_thresholds": FROZEN_THRESHOLDS,
        "particlefield_receipt_path": str(particle_receipt_path),
        "particlefield_sha256": _sha256(particlefield),
        "aura_native_render_manifest_digest": "sha256:" + "c" * 64,
        "camera_configs": probe_rows,
    }
    probe["manifest_digest"] = canonical_digest(probe, digest_field="manifest_digest")
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    input_digest = "sha256:" + "d" * 64
    bundle = {
        "status": "ready",
        "source_probe_manifest_digest": probe["manifest_digest"],
        "conformance_thresholds": FROZEN_THRESHOLDS,
        "thresholds_frozen_before_ovrtx_execution": True,
        "ovrtx_outcomes_observed_before_freeze": False,
        "input_digest": input_digest,
        "bundle_sha256": "sha256:" + "e" * 64,
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    result = {
        "schema_version": "adp009d_ovrtx_live_camera_result.v1",
        "status": "completed",
        "blockers": [],
        "input_digest": input_digest,
        "particlefield_sha256": _sha256(particlefield),
        "implementation_commit": "f" * 40,
        "camera_rows": result_rows,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    request = materialize_aura_renderer_conformance_request(
        probe_manifest_path=probe_path,
        provider_bundle_receipt_path=bundle_path,
        ovrtx_result_path=result_path,
        evidence_root=tmp_path,
        output_path=tmp_path / "request.json",
    )

    assert request["source_probe_manifest_digest"] == probe["manifest_digest"]
    assert request["threshold_definition_commit"] == THRESHOLD_DEFINITION_COMMIT
    assert request["ovrtx_outcomes_observed_before_threshold_freeze"] is False
    assert (tmp_path / "request.json").is_file()


def test_failure_diagnostic_rejects_alpha_only_rgb_and_nonfinite_depth(
    tmp_path: Path,
) -> None:
    camera = tmp_path / "exact_0"
    camera.mkdir()
    rgb = np.zeros((16, 16, 4), dtype=np.uint8)
    rgb[..., 3] = 255
    depth = np.full((16, 16, 1), np.inf, dtype=np.float32)
    np.save(camera / "rgb.npy", rgb, allow_pickle=False)
    np.save(camera / "depth.npy", depth, allow_pickle=False)
    result = {
        "schema_version": "adp009d_ovrtx_live_camera_result.v1",
        "status": "blocked",
        "implementation_commit": "a" * 40,
        "input_digest": "sha256:" + "b" * 64,
        "particlefield_sha256": "sha256:" + "c" * 64,
        "metric_depth_aov": "DistanceToCameraSD",
        "camera_rows": [
            {
                "camera_id": "exact_0",
                "artifacts": [
                    {
                        "path": "exact_0/rgb.npy",
                        "sha256": _sha256(camera / "rgb.npy"),
                    },
                    {
                        "path": "exact_0/depth.npy",
                        "sha256": _sha256(camera / "depth.npy"),
                    },
                ],
            }
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    receipt = materialize_aura_ovrtx_failure_diagnostic(
        ovrtx_result_path=result_path,
        output_path=tmp_path / "diagnostic.json",
    )

    assert receipt["status"] == "blocked"
    assert receipt["rows"][0]["rgb_color_signal_passed"] is False
    assert receipt["rows"][0]["positive_finite_depth_count"] == 0
    assert receipt["smallest_exact_blocker"] == (
        "sealed_aura_hybrid_policy_observation_renderer_missing"
    )
