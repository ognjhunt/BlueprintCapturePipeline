"""Materialize digest-bound OVRTX probes from observed Isaac camera calibration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest


MANIFEST_SCHEMA_VERSION = "adp009d_ovrtx_live_camera_probe.v1"
EXPECTED_CAMERA_IDS = {"external_camera": "external", "wrist_camera": "wrist"}
DEFAULT_HORIZONTAL_APERTURE_MM = 20.955


def _sha256(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _finite_matrix(value: Any, *, rows: int, columns: int, error: str) -> list[list[float]]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == rows
        and all(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == columns
            for row in value
        )
    ):
        raise ValueError(error)
    result = [[float(item) for item in row] for row in value]
    if not all(math.isfinite(item) for row in result for item in row):
        raise ValueError(error)
    return result


def opengl_camera_pose_to_usd_row_matrix(
    position_world_m: Sequence[float], quaternion_world_opengl_xyzw: Sequence[float]
) -> list[list[float]]:
    """Convert an Isaac/OpenGL camera pose to USD/Gf row-matrix storage."""

    if len(position_world_m) != 3 or len(quaternion_world_opengl_xyzw) != 4:
        raise ValueError("ovrtx_live_camera_pose_invalid")
    position = [float(value) for value in position_world_m]
    x, y, z, w = (float(value) for value in quaternion_world_opengl_xyzw)
    if not all(math.isfinite(value) for value in (*position, x, y, z, w)):
        raise ValueError("ovrtx_live_camera_pose_invalid")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-8:
        raise ValueError("ovrtx_live_camera_quaternion_invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    column_matrix = [
        [rotation[row][column] for column in range(3)] + [position[row]]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    return [[column_matrix[column][row] for column in range(4)] for row in range(4)]


def _camera_config(row: Mapping[str, Any]) -> dict[str, Any]:
    resolution = row.get("resolution_hw")
    if not (
        isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution)
    ):
        raise ValueError("ovrtx_live_camera_resolution_invalid")
    height, width = resolution
    intrinsic = _finite_matrix(
        row.get("intrinsic_matrix"),
        rows=3,
        columns=3,
        error="ovrtx_live_camera_intrinsic_invalid",
    )
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    if fx <= 0 or fy <= 0 or abs(fx - fy) > 1.0e-3:
        raise ValueError("ovrtx_live_camera_intrinsic_invalid")
    aperture = DEFAULT_HORIZONTAL_APERTURE_MM
    vertical_aperture = aperture * height / width
    return {
        "camera_id": EXPECTED_CAMERA_IDS[str(row.get("camera_id"))],
        "camera_prim_path": "/BlueprintLive/Camera",
        "camera_transform_matrix_usd": opengl_camera_pose_to_usd_row_matrix(
            row.get("position_world_m") or [],
            row.get("quaternion_world_opengl_xyzw") or [],
        ),
        "camera_coordinate_convention": "OpenGL_equal_to_USD_camera_axes",
        "width": width,
        "height": height,
        "focal_length_mm": fx * aperture / width,
        "horizontal_aperture_mm": aperture,
        "vertical_aperture_mm": vertical_aperture,
        "horizontal_aperture_offset_mm": (cx - width / 2.0) * aperture / width,
        "vertical_aperture_offset_mm": (height / 2.0 - cy) * vertical_aperture / height,
        "clipping_range": [0.01, 100.0],
        "render_mode": "RealTimePathTracing",
        "warmup_frames": 40,
        "quality_steps": 1,
        "delta_time_seconds": 1.0 / 60.0,
        "metric_depth_aov": "DistanceToCameraSD",
        "_blueprint_required_checks": ["particlefield_gaussian_surflet_render"],
    }


def materialize_ovrtx_live_camera_probe(
    *,
    native_result_path: str | Path,
    particlefield_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    native_path = Path(native_result_path).resolve()
    particle_receipt_path = Path(particlefield_receipt_path).resolve()
    output = Path(output_dir).resolve()
    if not native_path.is_file() or not particle_receipt_path.is_file():
        raise ValueError("ovrtx_live_camera_source_receipt_missing")
    native = json.loads(native_path.read_text(encoding="utf-8"))
    particle = json.loads(particle_receipt_path.read_text(encoding="utf-8"))
    if native.get("status") != "completed" or native.get("blockers"):
        raise ValueError("ovrtx_live_camera_native_microcheck_not_admitted")
    if native.get("sealed_source_mutated") is not False:
        raise ValueError("ovrtx_live_camera_sealed_source_mutated")
    if (
        particle.get("schema_version") != "aura_ovrtx_particlefield_receipt.v1"
        or particle.get("status") != "completed"
        or particle.get("schema")
        != "ParticleField+ParticleFieldKernelGaussianSurfletAPI"
        or particle.get("receipt_digest")
        != canonical_digest(particle, digest_field="receipt_digest")
    ):
        raise ValueError("ovrtx_live_camera_particlefield_receipt_invalid")
    particlefield_path = Path(str(particle.get("output") or "")).resolve()
    if not particlefield_path.is_file() or _sha256(particlefield_path) != particle.get(
        "output_sha256"
    ):
        raise ValueError("ovrtx_live_camera_particlefield_digest_mismatch")
    camera_rows = native.get("camera_frames")
    if not isinstance(camera_rows, list):
        raise ValueError("ovrtx_live_camera_frames_missing")
    by_id = {str(row.get("camera_id")): row for row in camera_rows if isinstance(row, Mapping)}
    if set(by_id) != set(EXPECTED_CAMERA_IDS):
        raise ValueError("ovrtx_live_camera_set_invalid")

    output.mkdir(parents=True, exist_ok=True)
    config_rows: list[dict[str, Any]] = []
    for source_id, camera_id in EXPECTED_CAMERA_IDS.items():
        config = _camera_config(by_id[source_id])
        config_path = output / f"{camera_id}.ovrtx.json"
        write_json(config_path, config)
        config_rows.append(
            {
                "camera_id": camera_id,
                "source_camera_id": source_id,
                "configuration_path": str(config_path),
                "configuration_sha256": _sha256(config_path),
                "source_frame_index": int(by_id[source_id].get("frame_index")),
                "source_timestamp_ns": int(by_id[source_id].get("timestamp_ns")),
                "source_sim_time_seconds": float(by_id[source_id].get("sim_time_seconds")),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "materialized_unexecuted",
        "native_result_path": str(native_path),
        "native_result_sha256": _sha256(native_path),
        "particlefield_receipt_path": str(particle_receipt_path),
        "particlefield_receipt_sha256": _sha256(particle_receipt_path),
        "particlefield_path": str(particlefield_path),
        "particlefield_sha256": _sha256(particlefield_path),
        "camera_configs": config_rows,
        "camera_ids": ["external", "wrist"],
        "render_mode": "RealTimePathTracing",
        "rtpt_warmup_frames": 40,
        "metric_depth_aov": "DistanceToCameraSD",
        "unitless_depth_sd_used": False,
        "requested_modalities": ["rgb", "depth"],
        "semantic_contract": "Aura background has no dynamic semantic ownership; composed Isaac override layer owns robot and task-object labels.",
        "sealed_sources_mutated": False,
        "live_renderer_proven": False,
        "blockers": ["sealed_aura_hybrid_policy_observation_renderer_missing"],
        "proof_boundary": "Immutable OVRTX input materialization is not live render evidence.",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    write_json(output / "adp009d_ovrtx_live_camera_probe.v1.json", manifest)
    return manifest


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "materialize_ovrtx_live_camera_probe",
    "opengl_camera_pose_to_usd_row_matrix",
]
