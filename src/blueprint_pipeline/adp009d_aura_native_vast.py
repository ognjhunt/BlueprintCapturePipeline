"""Immutable native Aura camera bundle and canonical capped Vast transport."""

from __future__ import annotations

import argparse
import json
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
