from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.adp009d_ovrtx_live_camera import (
    materialize_ovrtx_exact_camera_conformance_probe,
    materialize_ovrtx_live_camera_probe,
    opengl_camera_pose_to_usd_row_matrix,
)
from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _camera(camera_id: str) -> dict:
    return {
        "camera_id": camera_id,
        "frame_index": 40,
        "timestamp_ns": 123,
        "sim_time_seconds": 2.0,
        "resolution_hw": [720, 1280],
        "intrinsic_matrix": [[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]],
        "position_world_m": [1.0, 2.0, 3.0],
        "quaternion_world_opengl_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_opengl_pose_identity_has_usd_translation_in_last_row() -> None:
    matrix = opengl_camera_pose_to_usd_row_matrix([1, 2, 3], [0, 0, 0, 1])
    assert matrix == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 2.0, 3.0, 1.0],
    ]


def test_materializer_binds_observed_camera_and_particlefield(tmp_path: Path) -> None:
    particlefield = tmp_path / "aura.usdc"
    particlefield.write_bytes(b"usd")
    particle = {
        "schema_version": "aura_ovrtx_particlefield_receipt.v1",
        "status": "completed",
        "schema": "ParticleField+ParticleFieldKernelGaussianSurfletAPI",
        "output": str(particlefield),
        "output_sha256": f"sha256:{sha256_file(particlefield)}",
    }
    particle["receipt_digest"] = canonical_digest(particle, digest_field="receipt_digest")
    particle_path = tmp_path / "particle.json"
    _write_json(particle_path, particle)
    native = {
        "status": "completed",
        "blockers": [],
        "sealed_source_mutated": False,
        "camera_frames": [_camera("external_camera"), _camera("wrist_camera")],
    }
    native_path = tmp_path / "native.json"
    _write_json(native_path, native)

    result = materialize_ovrtx_live_camera_probe(
        native_result_path=native_path,
        particlefield_receipt_path=particle_path,
        output_dir=tmp_path / "probe",
    )
    assert result["status"] == "materialized_unexecuted"
    assert result["camera_ids"] == ["external", "wrist"]
    assert result["rtpt_warmup_frames"] == 40
    assert result["metric_depth_aov"] == "DistanceToCameraSD"
    assert result["unitless_depth_sd_used"] is False
    external = json.loads((tmp_path / "probe/external.ovrtx.json").read_text())
    assert external["camera_transform_matrix_usd"][3][:3] == [1.0, 2.0, 3.0]
    assert external["_blueprint_required_checks"] == [
        "particlefield_gaussian_surflet_render"
    ]


def test_materializer_rejects_caller_asserted_particlefield_digest(tmp_path: Path) -> None:
    particlefield = tmp_path / "aura.usdc"
    particlefield.write_bytes(b"usd")
    particle = {
        "schema_version": "aura_ovrtx_particlefield_receipt.v1",
        "status": "completed",
        "schema": "ParticleField+ParticleFieldKernelGaussianSurfletAPI",
        "output": str(particlefield),
        "output_sha256": "sha256:" + "0" * 64,
    }
    particle["receipt_digest"] = canonical_digest(particle, digest_field="receipt_digest")
    particle_path = tmp_path / "particle.json"
    _write_json(particle_path, particle)
    native_path = tmp_path / "native.json"
    _write_json(
        native_path,
        {
            "status": "completed",
            "blockers": [],
            "sealed_source_mutated": False,
            "camera_frames": [_camera("external_camera"), _camera("wrist_camera")],
        },
    )
    with pytest.raises(ValueError, match="particlefield_digest_mismatch"):
        materialize_ovrtx_live_camera_probe(
            native_result_path=native_path,
            particlefield_receipt_path=particle_path,
            output_dir=tmp_path / "probe",
        )


def test_exact_camera_conformance_probe_binds_native_frames_and_axis_change(
    tmp_path: Path,
) -> None:
    particlefield = tmp_path / "aura.usdc"
    particlefield.write_bytes(b"usd")
    particle = {
        "schema_version": "aura_ovrtx_particlefield_receipt.v1",
        "status": "completed",
        "output": str(particlefield),
        "output_sha256": f"sha256:{sha256_file(particlefield)}",
    }
    particle["receipt_digest"] = canonical_digest(
        particle, digest_field="receipt_digest"
    )
    particle_path = tmp_path / "particle.json"
    _write_json(particle_path, particle)
    exact_rows = []
    native_rows = []
    native_root = tmp_path / "native"
    frames = native_root / "frames"
    frames.mkdir(parents=True)
    for index, camera_id in enumerate(("approach_close", "right_translate")):
        exact_rows.append(
            {
                "id": camera_id,
                "spec": {
                    "intrinsics": {
                        "width": 32,
                        "height": 24,
                        "fx": 20.0,
                        "fy": 20.0,
                        "cx": 16.0,
                        "cy": 12.0,
                    },
                    "pose": {
                        "T_world_camera_opencv": [
                            [1.0, 0.0, 0.0, float(index)],
                            [0.0, 1.0, 0.0, 2.0],
                            [0.0, 0.0, 1.0, 3.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    },
                },
            }
        )
        frame = frames / f"{camera_id}.png"
        Image.fromarray(np.full((24, 32, 3), 20 + index, dtype=np.uint8)).save(frame)
        native_rows.append(
            {
                "camera_id": camera_id,
                "relative_path": f"frames/{camera_id}.png",
                "digest": f"sha256:{sha256_file(frame)}",
            }
        )
    exact_path = tmp_path / "exact.json"
    exact_path.write_text(json.dumps(exact_rows), encoding="utf-8")
    native = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "rendered_by": "aurafusion360_native_2d_gaussian_rasterizer",
        "renders": native_rows,
    }
    native["sealed_camera_render_manifest_digest"] = canonical_digest(
        native, digest_field="sealed_camera_render_manifest_digest"
    )
    native_path = native_root / "manifest.json"
    _write_json(native_path, native)

    result = materialize_ovrtx_exact_camera_conformance_probe(
        exact_camera_path=exact_path,
        aura_native_manifest_path=native_path,
        particlefield_receipt_path=particle_path,
        output_dir=tmp_path / "probe",
        resolution=(16, 12),
    )

    assert result["camera_ids"] == ["approach_close", "right_translate"]
    assert result["thresholds_frozen_before_ovrtx_execution"] is True
    assert result["ovrtx_outcomes_observed_before_freeze"] is False
    assert result["materializer_source_sha256"].startswith("sha256:")
    first = result["camera_configs"][0]
    assert first["calibration"]["intrinsic_matrix"][0] == [10.0, 0.0, 8.0]
    config = json.loads(Path(first["configuration_path"]).read_text())
    assert config["camera_transform_matrix_usd"] == [
        [1.0, -0.0, -0.0, 0.0],
        [0.0, -1.0, -0.0, 0.0],
        [0.0, -0.0, -1.0, 0.0],
        [0.0, 2.0, 3.0, 1.0],
    ]
    assert Path(first["native_reference_path"]).is_file()
