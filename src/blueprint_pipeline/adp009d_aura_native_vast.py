"""Immutable native Aura camera bundle and canonical capped Vast transport."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_aura_author_smoke_vast import (
    DEFAULT_IMAGE,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TREE,
    SUBMODULES,
    _deterministic_zip_directory,
    _deterministic_zip_files,
    _git,
    _sha256,
    _source_files,
    _source_manifest,
)
from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .adp009d_aura_renderer_conformance import (
    FROZEN_THRESHOLDS,
    THRESHOLD_DEFINITION_COMMIT,
)
from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant


PROBE_KIND = "adp009d-aura-native-live-camera"
PROVIDER_BUNDLE_KIND = "adp009d_aura_native"
RESULT_SCHEMA_VERSION = "adp009d_aura_native_vast_run.v1"
PROBE_SCHEMA_VERSION = "adp009d_aura_native_camera_probe.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "adp009d_aura_native_live_camera_result.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/adp009d-aura-native"
DEFAULT_CAMERA_IDS = ("approach_close", "right_translate")
EXPECTED_AURA_PLY_SHA256 = (
    "sha256:cbb05fc8e6da6ecdb72464f3b115f63e8747e2b67e97c309b4e40952b33000bd"
)
AURA_NATIVE_RENDER_MANIFEST_DIGEST = (
    "sha256:dae559070e1df3df58a9778f11ced6295f3810fd2314d9307acfcaf4f70189ca"
)


def _read_mapping(path: Path, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def _finite_matrix(value: Any, *, size: int, error: str) -> list[list[float]]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == size
        and all(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == size
            for row in value
        )
    ):
        raise ValueError(error)
    result = [[float(item) for item in row] for row in value]
    import math

    if not all(math.isfinite(item) for row in result for item in row):
        raise ValueError(error)
    return result


def _opencv_world_from_opengl_pose(
    *, position_world: Any, quaternion_world_opengl_xyzw: Any
) -> list[list[float]]:
    """Convert Isaac's documented OpenGL camera axes to OpenCV/ROS axes."""

    if not (
        isinstance(position_world, Sequence)
        and not isinstance(position_world, (str, bytes))
        and len(position_world) == 3
        and isinstance(quaternion_world_opengl_xyzw, Sequence)
        and not isinstance(quaternion_world_opengl_xyzw, (str, bytes))
        and len(quaternion_world_opengl_xyzw) == 4
    ):
        raise ValueError("aura_native_isaac_camera_pose_invalid")
    position = [float(value) for value in position_world]
    x, y, z, w = (float(value) for value in quaternion_world_opengl_xyzw)
    if not all(math.isfinite(value) for value in (*position, x, y, z, w)):
        raise ValueError("aura_native_isaac_camera_pose_invalid")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("aura_native_isaac_camera_pose_invalid")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    rotation_gl = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    # Isaac/Usd OpenGL: +X right, +Y up, -Z forward. OpenCV/ROS:
    # +X right, +Y down, +Z forward. Flip the camera-local Y and Z axes.
    rotation_cv = [
        [rotation_gl[row][0], -rotation_gl[row][1], -rotation_gl[row][2]]
        for row in range(3)
    ]
    return [
        [*rotation_cv[0], position[0]],
        [*rotation_cv[1], position[1]],
        [*rotation_cv[2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def materialize_aura_native_from_isaac_camera_probe(
    *,
    isaac_result_path: str | Path,
    aura_ply_path: str | Path,
    output_path: str | Path,
    camera_ids: Sequence[str] = ("external_camera", "wrist_camera"),
) -> dict[str, Any]:
    """Freeze exact retained Isaac camera poses and input layers for Aura rendering."""

    result_path = Path(isaac_result_path).expanduser().resolve()
    result = _read_mapping(result_path, "aura_native_isaac_result_invalid")
    ply = Path(aura_ply_path).expanduser().resolve()
    if (
        result.get("schema_version") != "adp009d_native_microcheck.v1"
        or result.get("status") != "completed"
        or result.get("blockers")
        or result.get("workflow") != "isaac_lab_manager_based_via_arena_composition"
        or result.get("sealed_source_mutated") is not False
        or result.get("candidate_policy_queried") is not False
        or result.get("candidate_outcomes_accessed") is not False
        or result.get("arena_revision")
        != "8b4a3a47fc53de23e8205089d71109a2e2348acd"
        or result.get("isaac_lab_revision")
        != "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
    ):
        raise ValueError("aura_native_isaac_result_invalid")
    if not ply.is_file() or _sha256(ply) != EXPECTED_AURA_PLY_SHA256:
        raise ValueError("aura_native_sealed_ply_digest_mismatch")
    if len(camera_ids) != 2 or len(set(camera_ids)) != 2:
        raise ValueError("aura_native_isaac_camera_set_invalid")
    by_id = {
        str(row.get("camera_id")): row
        for row in result.get("camera_frames", [])
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for camera_id in camera_ids:
        row = by_id.get(str(camera_id))
        if not isinstance(row, Mapping):
            raise ValueError("aura_native_isaac_camera_missing")
        intrinsic = _finite_matrix(
            row.get("intrinsic_matrix"),
            size=3,
            error="aura_native_isaac_intrinsic_invalid",
        )
        resolution_hw = row.get("resolution_hw")
        if not (
            isinstance(resolution_hw, Sequence)
            and not isinstance(resolution_hw, (str, bytes))
            and len(resolution_hw) == 2
        ):
            raise ValueError("aura_native_isaac_resolution_invalid")
        height, width = (int(value) for value in resolution_hw)
        if (
            width <= 0
            or height <= 0
            or intrinsic[0][0] <= 0
            or intrinsic[1][1] <= 0
            or abs(intrinsic[0][2] - width / 2.0) > 1.0e-4
            or abs(intrinsic[1][2] - height / 2.0) > 1.0e-4
        ):
            raise ValueError("aura_native_isaac_offcenter_pinhole_unsupported")
        calibration = {
            "camera_model": "pinhole",
            "intrinsic_matrix": intrinsic,
            "world_from_camera": _opencv_world_from_opengl_pose(
                position_world=row.get("position_world_m"),
                quaternion_world_opengl_xyzw=row.get(
                    "quaternion_world_opengl_xyzw"
                ),
            ),
            "resolution": [width, height],
            "camera_coordinate_convention": "OpenCV_right_down_forward",
        }
        input_artifacts: dict[str, Any] = {}
        for key, field in (
            ("dynamic_rgb", "rgb_png"),
            ("dynamic_depth", "metric_depth"),
            ("dynamic_semantic", "semantic_segmentation"),
        ):
            binding = row.get(field)
            if not isinstance(binding, Mapping):
                raise ValueError("aura_native_isaac_input_binding_invalid")
            artifact = (result_path.parent / str(binding.get("path") or "")).resolve()
            try:
                artifact.relative_to(result_path.parent)
            except ValueError as exc:
                raise ValueError("aura_native_isaac_input_path_escape") from exc
            if not artifact.is_file() or _sha256(artifact) != binding.get("sha256"):
                raise ValueError("aura_native_isaac_input_digest_mismatch")
            input_artifacts[key] = {
                "path": str(artifact),
                "sha256": binding["sha256"],
            }
        rows.append(
            {
                "camera_id": str(camera_id),
                "calibration": calibration,
                "calibration_digest": canonical_digest(calibration),
                "native_reference_path": None,
                "native_reference_sha256": None,
                "source_isaac_frame_index": int(row.get("frame_index")),
                "source_isaac_sim_time_seconds": float(row.get("sim_time_seconds")),
                "source_isaac_timestamp_ns": int(row.get("timestamp_ns")),
                "source_isaac_input_artifacts": input_artifacts,
                "source_isaac_semantic_labels": row.get("semantic_segmentation", {}).get(
                    "id_to_labels"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "materialized_unexecuted",
        "program_id": "arm-decision-proof-v1",
        "probe_purpose": "aura_native_from_live_isaac_camera_calibration",
        "aura_repository": SOURCE_REPOSITORY,
        "aura_revision": SOURCE_COMMIT,
        "aura_tree": SOURCE_TREE,
        "aura_submodules": SUBMODULES,
        "aura_ply_path": str(ply),
        "aura_ply_sha256": _sha256(ply),
        "aura_native_render_manifest_digest": AURA_NATIVE_RENDER_MANIFEST_DIGEST,
        "source_isaac_result_path": str(result_path),
        "source_isaac_result_sha256": _sha256(result_path),
        "camera_configs": sorted(rows, key=lambda value: value["camera_id"]),
        "depth_output": "surf_depth_expected_camera_z_m",
        "depth_ratio": 0.0,
        "conformance_thresholds": FROZEN_THRESHOLDS,
        "threshold_definition_commit": THRESHOLD_DEFINITION_COMMIT,
        "thresholds_frozen_before_execution": True,
        "renderer_outcomes_observed_before_freeze": False,
        "candidate_policy_queried": False,
        "retry_cap": 0,
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    write_json(Path(output_path).expanduser().resolve(), manifest)
    return manifest


def materialize_aura_native_exact_camera_probe(
    *,
    exact_camera_path: str | Path,
    aura_native_manifest_path: str | Path,
    aura_ply_path: str | Path,
    output_path: str | Path,
    camera_ids: Sequence[str] = DEFAULT_CAMERA_IDS,
) -> dict[str, Any]:
    """Freeze two exact registered cameras before native renderer execution."""

    if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("aura_native_exact_camera_set_invalid")
    exact_path = Path(exact_camera_path).expanduser().resolve()
    native_path = Path(aura_native_manifest_path).expanduser().resolve()
    ply = Path(aura_ply_path).expanduser().resolve()
    try:
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("aura_native_exact_camera_source_invalid") from exc
    native = _read_mapping(native_path, "aura_native_reference_manifest_invalid")
    if not isinstance(exact, list):
        raise ValueError("aura_native_exact_camera_source_invalid")
    if (
        native.get("schema_version") != "sealed_camera_render_manifest.v1"
        or native.get("status") != "rendered_exact_cameras"
        or native.get("rendered_by")
        != "aurafusion360_native_2d_gaussian_rasterizer"
        or native.get("sealed_camera_render_manifest_digest")
        != canonical_digest(native, digest_field="sealed_camera_render_manifest_digest")
    ):
        raise ValueError("aura_native_reference_manifest_invalid")
    if not ply.is_file() or _sha256(ply) != EXPECTED_AURA_PLY_SHA256:
        raise ValueError("aura_native_sealed_ply_digest_mismatch")
    exact_by_id = {
        str(row.get("id")): row for row in exact if isinstance(row, Mapping)
    }
    native_by_id = {
        str(row.get("camera_id")): row
        for row in native.get("renders", [])
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for camera_id in camera_ids:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", str(camera_id)):
            raise ValueError("aura_native_exact_camera_id_invalid")
        exact_row = exact_by_id.get(str(camera_id))
        native_row = native_by_id.get(str(camera_id))
        if not isinstance(exact_row, Mapping) or not isinstance(native_row, Mapping):
            raise ValueError("aura_native_exact_camera_identity_missing")
        spec = exact_row.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError("aura_native_exact_camera_spec_invalid")
        intrinsics = spec.get("intrinsics")
        pose = spec.get("pose")
        if not isinstance(intrinsics, Mapping) or not isinstance(pose, Mapping):
            raise ValueError("aura_native_exact_camera_spec_invalid")
        width = int(intrinsics.get("width") or 0)
        height = int(intrinsics.get("height") or 0)
        fx = float(intrinsics.get("fx") or 0.0)
        fy = float(intrinsics.get("fy") or 0.0)
        cx = float(intrinsics.get("cx") or 0.0)
        cy = float(intrinsics.get("cy") or 0.0)
        if (
            width <= 0
            or height <= 0
            or fx <= 0
            or fy <= 0
            or abs(cx - width / 2.0) > 1.0e-6
            or abs(cy - height / 2.0) > 1.0e-6
        ):
            raise ValueError("aura_native_offcenter_pinhole_unsupported")
        world_from_camera = _finite_matrix(
            pose.get("T_world_camera_opencv"),
            size=4,
            error="aura_native_exact_camera_transform_invalid",
        )
        if [native_row.get("width"), native_row.get("height")] != [width, height]:
            raise ValueError("aura_native_reference_resolution_mismatch")
        reference = (native_path.parent / str(native_row.get("relative_path"))).resolve()
        try:
            reference.relative_to(native_path.parent)
        except ValueError as exc:
            raise ValueError("aura_native_reference_path_escape") from exc
        if not reference.is_file() or _sha256(reference) != native_row.get("digest"):
            raise ValueError("aura_native_reference_digest_mismatch")
        calibration = {
            "camera_model": "pinhole",
            "intrinsic_matrix": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            "world_from_camera": world_from_camera,
            "resolution": [width, height],
            "camera_coordinate_convention": "OpenCV_right_down_forward",
        }
        rows.append(
            {
                "camera_id": str(camera_id),
                "calibration": calibration,
                "calibration_digest": canonical_digest(calibration),
                "native_reference_path": str(reference),
                "native_reference_sha256": _sha256(reference),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "materialized_unexecuted",
        "program_id": "arm-decision-proof-v1",
        "probe_purpose": "aura_native_exact_camera_rgb_metric_depth_conformance",
        "aura_repository": SOURCE_REPOSITORY,
        "aura_revision": SOURCE_COMMIT,
        "aura_tree": SOURCE_TREE,
        "aura_submodules": SUBMODULES,
        "aura_ply_path": str(ply),
        "aura_ply_sha256": _sha256(ply),
        "aura_native_render_manifest_digest": native[
            "sealed_camera_render_manifest_digest"
        ],
        "camera_configs": sorted(rows, key=lambda row: row["camera_id"]),
        "depth_output": "surf_depth_expected_camera_z_m",
        "depth_ratio": 0.0,
        "conformance_thresholds": FROZEN_THRESHOLDS,
        "threshold_definition_commit": THRESHOLD_DEFINITION_COMMIT,
        "thresholds_frozen_before_execution": True,
        "renderer_outcomes_observed_before_freeze": False,
        "candidate_policy_queried": False,
        "retry_cap": 0,
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    write_json(Path(output_path).expanduser().resolve(), manifest)
    return manifest


def build_aura_native_live_camera_bundle(
    *,
    job_dir: str | Path,
    probe_manifest_path: str | Path,
    aura_root: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an immutable minimal official-source Aura rendering bundle."""

    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise ValueError("adp009d_aura_native_implementation_commit_invalid")
    source = Path(aura_root).expanduser().resolve()
    if (
        _git(source, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(source, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(source, "status", "--porcelain")
    ):
        raise ValueError("adp009d_aura_native_source_identity_mismatch")
    observed_submodules = {
        relative: _git(source / relative, "rev-parse", "HEAD")
        for relative in SUBMODULES
    }
    if observed_submodules != SUBMODULES or any(
        _git(source / relative, "status", "--porcelain") for relative in SUBMODULES
    ):
        raise ValueError("adp009d_aura_native_submodule_identity_mismatch")
    probe_path = Path(probe_manifest_path).expanduser().resolve()
    probe = _read_mapping(probe_path, "adp009d_aura_native_probe_invalid")
    if (
        probe.get("schema_version") != PROBE_SCHEMA_VERSION
        or probe.get("status") != "materialized_unexecuted"
        or probe.get("manifest_digest")
        != canonical_digest(probe, digest_field="manifest_digest")
        or probe.get("aura_revision") != SOURCE_COMMIT
        or probe.get("aura_tree") != SOURCE_TREE
        or probe.get("aura_submodules") != SUBMODULES
        or probe.get("depth_output") != "surf_depth_expected_camera_z_m"
        or probe.get("depth_ratio") != 0.0
        or probe.get("conformance_thresholds") != FROZEN_THRESHOLDS
        or probe.get("threshold_definition_commit")
        != THRESHOLD_DEFINITION_COMMIT
        or probe.get("thresholds_frozen_before_execution") is not True
        or probe.get("renderer_outcomes_observed_before_freeze") is not False
    ):
        raise ValueError("adp009d_aura_native_probe_invalid")
    ply = Path(str(probe.get("aura_ply_path") or "")).expanduser().resolve()
    if not ply.is_file() or _sha256(ply) != probe.get("aura_ply_sha256"):
        raise ValueError("adp009d_aura_native_ply_digest_mismatch")
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp009d_aura_native_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime / "camera_configs")
    source_rows = _source_files(source)
    source_manifest = _source_manifest(source_rows)
    _deterministic_zip_files(source_rows, runtime / "aurafusion360_source.zip")
    shutil.copy2(ply, runtime / "aura_sealed.ply")
    camera_bindings = []
    for row in probe.get("camera_configs", []):
        if not isinstance(row, Mapping):
            raise ValueError("adp009d_aura_native_camera_config_invalid")
        camera_id = str(row.get("camera_id") or "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", camera_id):
            raise ValueError("adp009d_aura_native_camera_id_invalid")
        config = {
            "camera_id": camera_id,
            "calibration": row.get("calibration"),
            "calibration_digest": row.get("calibration_digest"),
            "native_reference_sha256": row.get("native_reference_sha256"),
            "source_isaac_frame_index": row.get("source_isaac_frame_index"),
            "source_isaac_sim_time_seconds": row.get(
                "source_isaac_sim_time_seconds"
            ),
            "source_isaac_timestamp_ns": row.get("source_isaac_timestamp_ns"),
            "source_isaac_input_artifacts": row.get(
                "source_isaac_input_artifacts"
            ),
            "source_isaac_semantic_labels": row.get(
                "source_isaac_semantic_labels"
            ),
        }
        if config["calibration_digest"] != canonical_digest(config["calibration"]):
            raise ValueError("adp009d_aura_native_calibration_digest_mismatch")
        config_path = runtime / "camera_configs" / f"{camera_id}.json"
        write_json(config_path, config)
        camera_bindings.append(
            {
                "camera_id": camera_id,
                "configuration_sha256": _sha256(config_path),
                "calibration_digest": config["calibration_digest"],
                "native_reference_sha256": config["native_reference_sha256"],
                "source_isaac_frame_index": config["source_isaac_frame_index"],
                "source_isaac_sim_time_seconds": config[
                    "source_isaac_sim_time_seconds"
                ],
                "source_isaac_timestamp_ns": config["source_isaac_timestamp_ns"],
            }
        )
    if len(camera_bindings) < 2 or len(camera_bindings) > 8:
        raise ValueError("adp009d_aura_native_camera_set_invalid")
    repo_root = Path(__file__).resolve().parents[2]
    for name in (
        "adp009d_aura_native_provider_runner.py",
        "run_adp009d_aura_native_provider_runtime.sh",
    ):
        source_file = repo_root / "scripts" / name
        destination = runtime / name
        shutil.copy2(source_file, destination)
        if name.endswith(".sh"):
            destination.chmod(0o755)
    manifest: dict[str, Any] = {
        "schema_version": "adp009d_aura_native_provider_manifest.v1",
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "implementation_commit": implementation_commit,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_submodules": SUBMODULES,
        "source_files": source_manifest,
        "source_archive_sha256": _sha256(runtime / "aurafusion360_source.zip"),
        "source_license": "Apache-2.0_with_nested_2DGS_research_only_components",
        "aura_ply_sha256": _sha256(runtime / "aura_sealed.ply"),
        "source_probe_manifest_digest": probe["manifest_digest"],
        "aura_native_render_manifest_digest": probe[
            "aura_native_render_manifest_digest"
        ],
        "camera_configs": sorted(camera_bindings, key=lambda row: row["camera_id"]),
        "container_image": DEFAULT_IMAGE,
        "python_version": "3.10",
        "cuda_toolkit": "12.4",
        "torch_version": "2.5.1+cu124",
        "depth_output": "surf_depth_expected_camera_z_m",
        "depth_ratio": 0.0,
        "conformance_thresholds": FROZEN_THRESHOLDS,
        "threshold_definition_commit": THRESHOLD_DEFINITION_COMMIT,
        "candidate_policy_queried": False,
        "private_data_uploaded": False,
        "provider_zero_required_after_return": True,
        "retry_cap": 0,
        "blockers": [],
    }
    if generated_at is not None:
        manifest["generated_at"] = generated_at
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    write_json(runtime / "adp009d_aura_native_provider_manifest.json", manifest)
    bundle = job / "adp009d_aura_native_live_camera_bundle.zip"
    _deterministic_zip_directory(runtime, bundle)
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
    }
    write_json(job / "adp009d_aura_native_live_camera_bundle_receipt.json", receipt)
    return receipt


def run_aura_native_live_camera_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 1.25,
    hard_ttl_seconds: int = 2700,
) -> dict[str, Any]:
    return run_arena_native_control_vast(
        approval_path=".",
        job_dir=job_dir,
        paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute,
        prepared_bundle=prepared_bundle,
        machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename="adp009d_aura_native_live_camera_result.json",
        container_image=DEFAULT_IMAGE,
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=DEFAULT_KEY_PREFIX,
        instance_label_prefix="blueprint-adp009d-aura-native-",
        blocker_prefix="adp009d_aura_native",
        min_gpu_ram_mb=46_000,
        min_compute_cap=860,
        minimum_driver_version="550.54.14",
        require_known_supported_isaac_driver=False,
        enable_isaac_smoke=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"),
    )


__all__ = [
    "PROBE_KIND",
    "build_aura_native_live_camera_bundle",
    "materialize_aura_native_from_isaac_camera_probe",
    "materialize_aura_native_exact_camera_probe",
    "run_aura_native_live_camera_vast",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-camera", required=True)
    parser.add_argument("--aura-native-manifest", required=True)
    parser.add_argument("--aura-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-id", action="append", dest="camera_ids")
    args = parser.parse_args(argv)
    result = materialize_aura_native_exact_camera_probe(
        exact_camera_path=args.exact_camera,
        aura_native_manifest_path=args.aura_native_manifest,
        aura_ply_path=args.aura_ply,
        output_path=args.output,
        camera_ids=args.camera_ids or DEFAULT_CAMERA_IDS,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
