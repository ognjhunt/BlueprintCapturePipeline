"""Compile a trained standard 3DGS PLY into a provenance-bound appearance asset.

This is the deterministic bridge between candidate-only Gaussian training and
NuRec/OpenUSD packaging.  It authors a ParticleField USD, but it does not render
the asset, evaluate held-out observations, or promote metric/collision claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from typing import Any, Mapping

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .particlefield_usd import write_particlefield_usd
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_training_result,
)


APPEARANCE_ASSET_MANIFEST_SCHEMA = "appearance_asset_manifest.v1"
APPEARANCE_COMPILER_IMPLEMENTATION_VERSION = "blueprint_particlefield_compiler.v1"
MAX_STANDARD_PLY_BYTES = 2_000_000_000
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class AppearanceAssetContractError(ValueError):
    def __init__(self, codes: list[str] | tuple[str, ...]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_relative(value: Any, *, code: str) -> PurePosixPath:
    text = str(value or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise AppearanceAssetContractError([code])
    return relative


def build_appearance_asset_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        artifact = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise AppearanceAssetContractError(["appearance_manifest_not_json_serializable"]) from exc
    if not isinstance(artifact, dict):
        raise AppearanceAssetContractError(["appearance_manifest_not_object"])
    supplied = artifact.pop("appearance_asset_manifest_digest", None)
    errors: list[str] = []
    required_strings = (
        "stable_run_identity",
        "source_capture_identity",
        "producing_method",
        "implementation_version",
        "source_commit_sha",
        "timestamp",
    )
    for key in required_strings:
        if not isinstance(artifact.get(key), str) or not artifact[key]:
            errors.append(f"{key}_missing")
    digest_fields = (
        "source_capture_digest",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
        "reconstruction_training_request_digest",
        "reconstruction_training_result_digest",
        "source_appearance_asset_digest",
        "appearance_asset_digest",
    )
    for key in digest_fields:
        value_digest = artifact.get(key)
        if not isinstance(value_digest, str) or _DIGEST.fullmatch(value_digest) is None:
            errors.append(f"{key}_invalid")
    for key in (
        "original_file_references",
        "input_digests",
        "output_digests",
        "warnings",
        "blockers",
    ):
        if not isinstance(artifact.get(key), list):
            errors.append(f"{key}_invalid")
    for key in (
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "provider_runtime_identity",
        "authority_used",
        "parent_artifact_or_event",
    ):
        if not isinstance(artifact.get(key), Mapping):
            errors.append(f"{key}_invalid")
    if artifact.get("container_image_digest") is not None and _IMAGE.fullmatch(
        str(artifact.get("container_image_digest"))
    ) is None:
        errors.append("container_image_digest_invalid")
    if _COMMIT.fullmatch(str(artifact.get("source_commit_sha") or "")) is None:
        errors.append("source_commit_sha_invalid")
    for key in ("cost_usd", "duration_seconds"):
        measurement = artifact.get(key)
        if (
            isinstance(measurement, bool)
            or not isinstance(measurement, (int, float))
            or not math.isfinite(float(measurement))
            or float(measurement) < 0
        ):
            errors.append(f"{key}_invalid")
    if artifact.get("metric_scale_status") not in {
        "validated",
        "sensor_metric_unvalidated",
        "anchor_required",
        "unknown",
    }:
        errors.append("metric_scale_status_invalid")
    if artifact.get("units") != "meters":
        errors.append("appearance_asset_units_must_be_meters")
    coordinate = artifact.get("coordinate_frame_declaration")
    if isinstance(coordinate, Mapping) and coordinate.get("up_axis") != "Z":
        errors.append("appearance_asset_z_up_frame_required")
    if artifact.get("status") != "completed" or artifact.get("blockers") != []:
        errors.append("appearance_asset_not_completed")
    if artifact.get("source_asset_format") != "standard_3dgs_ply":
        errors.append("appearance_source_format_invalid")
    if artifact.get("appearance_asset_format") != "particlefield_usd":
        errors.append("appearance_output_format_invalid")
    if artifact.get("source_prim_path") != "/World/Appearance":
        errors.append("appearance_source_prim_path_invalid")
    if artifact.get("captured_observation") is not False or artifact.get(
        "raw_evidence"
    ) is not False:
        errors.append("reconstruction_cannot_be_captured_evidence")
    if artifact.get("metric_geometry_proven") is not False or artifact.get(
        "collision_geometry_proven"
    ) is not False:
        errors.append("appearance_cannot_promote_geometry")
    if artifact.get("heldout_evaluated") is not False:
        errors.append("appearance_compiler_cannot_grade_heldout")
    if artifact.get("proof_effect") != "appearance_asset_candidate_only" or artifact.get(
        "claim_ceiling"
    ) != "appearance_reconstruction":
        errors.append("appearance_asset_claim_boundary_invalid")
    if not isinstance(artifact.get("splat_count"), int) or artifact.get("splat_count", 0) < 1:
        errors.append("appearance_splat_count_invalid")
    if not isinstance(artifact.get("sh_degree"), int) or not 0 <= artifact.get("sh_degree", -1) <= 3:
        errors.append("appearance_sh_degree_invalid")
    try:
        source_reference = _safe_relative(
            artifact.get("source_appearance_asset_reference"),
            code="source_appearance_asset_reference_unsafe",
        )
        output_reference = _safe_relative(
            artifact.get("appearance_asset_reference"),
            code="appearance_asset_reference_unsafe",
        )
        if source_reference.suffix.lower() != ".ply":
            errors.append("source_appearance_asset_reference_format_invalid")
        if output_reference.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            errors.append("appearance_asset_reference_format_invalid")
    except AppearanceAssetContractError as exc:
        errors.extend(exc.codes)
    digest_rows_valid = True
    for key in ("original_file_references", "input_digests", "output_digests"):
        rows = artifact.get(key) or []
        if any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("artifact_id"), str)
            or not item.get("artifact_id")
            or not isinstance(item.get("digest"), str)
            or _DIGEST.fullmatch(item["digest"]) is None
            for item in rows
        ):
            errors.append(f"{key}_digest_rows_invalid")
            digest_rows_valid = False
    input_digests = {
        item.get("digest")
        for item in artifact.get("input_digests") or []
        if isinstance(item, Mapping)
    }
    output_digests = {
        item.get("digest")
        for item in artifact.get("output_digests") or []
        if isinstance(item, Mapping)
    }
    if artifact.get("source_appearance_asset_digest") not in input_digests:
        errors.append("appearance_source_digest_not_bound")
    if artifact.get("reconstruction_training_result_digest") not in input_digests:
        errors.append("appearance_training_result_digest_not_bound")
    if artifact.get("appearance_asset_digest") not in output_digests:
        errors.append("appearance_output_digest_not_bound")
    parent = artifact.get("parent_artifact_or_event")
    if isinstance(parent, Mapping) and parent.get("digest") != artifact.get(
        "reconstruction_training_result_digest"
    ):
        errors.append("appearance_parent_training_result_mismatch")
    if digest_rows_valid:
        matching_outputs = [
            item
            for item in artifact.get("output_digests") or []
            if item.get("artifact_id") == artifact.get("appearance_asset_reference")
            and item.get("digest") == artifact.get("appearance_asset_digest")
        ]
        if len(matching_outputs) != 1:
            errors.append("appearance_output_reference_not_exactly_bound")
    if errors:
        raise AppearanceAssetContractError(errors)
    artifact["schema_version"] = APPEARANCE_ASSET_MANIFEST_SCHEMA
    expected = canonical_digest(artifact, digest_field="appearance_asset_manifest_digest")
    if supplied is not None and supplied != expected:
        raise AppearanceAssetContractError(["appearance_asset_manifest_digest_mismatch"])
    artifact["appearance_asset_manifest_digest"] = expected
    return artifact


def _source_ply(training: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    rows = [
        row
        for row in training["output_digests"]
        if isinstance(row, Mapping) and row.get("artifact_id") == "appearance_candidate.ply"
    ]
    if len(rows) != 1:
        raise AppearanceAssetContractError(["trained_appearance_ply_binding_missing_or_ambiguous"])
    relative = _safe_relative(rows[0].get("artifact_id"), code="trained_appearance_ply_path_unsafe")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise AppearanceAssetContractError(["trained_appearance_ply_symlink_forbidden"])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AppearanceAssetContractError(["trained_appearance_ply_missing"]) from exc
    if root != resolved.parent and root not in resolved.parents:
        raise AppearanceAssetContractError(["trained_appearance_ply_path_escape"])
    if not resolved.is_file() or resolved.stat().st_size > MAX_STANDARD_PLY_BYTES:
        raise AppearanceAssetContractError(["trained_appearance_ply_missing_or_oversized"])
    digest = _sha256(resolved)
    if digest != rows[0].get("digest"):
        raise AppearanceAssetContractError(["trained_appearance_ply_digest_mismatch"])
    return resolved, digest


def _replay(final_dir: Path) -> dict[str, Any] | None:
    manifest_path = final_dir / "appearance_asset_manifest.v1.json"
    output_path = final_dir / "appearance.usda"
    if not manifest_path.is_file() or not output_path.is_file():
        return None
    try:
        manifest = build_appearance_asset_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, AppearanceAssetContractError):
        return None
    if _sha256(output_path) != manifest["appearance_asset_digest"]:
        return None
    return manifest


def compile_particlefield_appearance_asset(
    *,
    training_result: Mapping[str, Any],
    training_artifact_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Author and record a replayable ParticleField appearance candidate."""

    try:
        training = build_training_result(training_result)
    except ReconstructionWorkerContractError as exc:
        raise AppearanceAssetContractError(
            [f"training_result_invalid:{code}" for code in exc.codes]
        ) from exc
    if training["status"] != "succeeded":
        raise AppearanceAssetContractError(["successful_training_result_required"])
    if training.get("heldout_labels_included") is not False or training.get(
        "candidate_self_graded"
    ) is not False:
        raise AppearanceAssetContractError(["training_heldout_isolation_invalid"])
    coordinate = training.get("coordinate_frame_declaration")
    if not isinstance(coordinate, Mapping) or coordinate.get("up_axis") != "Z":
        raise AppearanceAssetContractError(["trained_appearance_z_up_frame_unqualified"])

    training_root_path = Path(training_artifact_root)
    if training_root_path.is_symlink() or not training_root_path.is_dir():
        raise AppearanceAssetContractError(["training_artifact_root_invalid"])
    training_root_path = training_root_path.resolve()
    source_ply, source_digest = _source_ply(training, training_root_path)

    destination = Path(output_root)
    if destination.is_symlink():
        raise AppearanceAssetContractError(["appearance_output_root_symlink_forbidden"])
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    configuration = {
        "implementation_version": APPEARANCE_COMPILER_IMPLEMENTATION_VERSION,
        "source_training_result_digest": training["reconstruction_training_result_digest"],
        "source_prim_path": "/World/Appearance",
        "units": "meters",
        "up_axis": "Z",
        "preserve_source_sh": True,
    }
    configuration_digest = canonical_digest(configuration)
    content_id = configuration_digest[7:]
    final_dir = destination / content_id
    replay = _replay(final_dir)
    if replay is not None:
        if replay["reconstruction_training_result_digest"] != training[
            "reconstruction_training_result_digest"
        ]:
            raise AppearanceAssetContractError(["appearance_replay_lineage_mismatch"])
        return replay
    if final_dir.exists() or final_dir.is_symlink():
        raise AppearanceAssetContractError(["appearance_existing_output_incomplete_or_tampered"])

    temporary = Path(tempfile.mkdtemp(prefix=".appearance-", dir=destination))
    started = time.monotonic()
    try:
        output_path = temporary / "appearance.usda"
        try:
            authored = write_particlefield_usd(
                source_ply,
                output_path,
                prim_path="/World/Appearance",
                up_axis="Z",
            )
        except (OSError, ValueError) as exc:
            raise AppearanceAssetContractError(["particlefield_authoring_input_invalid"]) from exc
        if authored.get("status") != "completed" or not output_path.is_file():
            blockers = authored.get("blockers") or ["particlefield_authoring_failed"]
            raise AppearanceAssetContractError(
                [f"particlefield_authoring_blocked:{code}" for code in blockers]
            )
        output_digest = _sha256(output_path)
        artifact_reference = f"{content_id}/appearance.usda"
        original_references = json.loads(json.dumps(training["original_file_references"]))
        original_references.append(
            {"artifact_id": "appearance_candidate.ply", "digest": source_digest}
        )
        input_digests = json.loads(json.dumps(training["input_digests"]))
        input_digests.extend(
            [
                {
                    "artifact_id": "reconstruction_training_result.v1",
                    "digest": training["reconstruction_training_result_digest"],
                },
                {"artifact_id": "appearance_candidate.ply", "digest": source_digest},
            ]
        )
        manifest = build_appearance_asset_manifest(
            {
                "stable_run_identity": training["stable_run_identity"],
                "source_capture_identity": training["source_capture_identity"],
                "source_capture_digest": training["source_capture_digest"],
                "original_file_references": original_references,
                "producing_method": "blueprint.standard_3dgs_to_particlefield",
                "implementation_version": APPEARANCE_COMPILER_IMPLEMENTATION_VERSION,
                "container_image_digest": training.get("container_image_digest"),
                "source_commit_sha": training["source_commit_sha"],
                "deterministic_configuration_digest": configuration_digest,
                "input_digests": input_digests,
                "output_digests": [
                    {"artifact_id": artifact_reference, "digest": output_digest}
                ],
                "train_heldout_split_digest": training["train_heldout_split_digest"],
                "camera_calibration_binding": training["camera_calibration_binding"],
                "coordinate_frame_declaration": training["coordinate_frame_declaration"],
                "units": "meters",
                "metric_scale_status": training["metric_scale_status"],
                "provider_runtime_identity": {
                    "provider": "local",
                    "runtime": "openusd_particlefield_authoring",
                    "source_training_runtime": training["provider_runtime_identity"],
                },
                "cost_usd": 0.0,
                "duration_seconds": round(time.monotonic() - started, 6),
                "authority_used": training["authority_used"],
                "warnings": [
                    "appearance_asset_is_reconstructed_not_captured_evidence",
                    "isaac_render_not_verified",
                ],
                "blockers": [],
                "proof_effect": "appearance_asset_candidate_only",
                "claim_ceiling": "appearance_reconstruction",
                "parent_artifact_or_event": {
                    "digest": training["reconstruction_training_result_digest"]
                },
                "timestamp": utc_now_iso(),
                "status": "completed",
                "reconstruction_training_request_digest": training[
                    "reconstruction_training_request_digest"
                ],
                "reconstruction_training_result_digest": training[
                    "reconstruction_training_result_digest"
                ],
                "source_appearance_asset_reference": "appearance_candidate.ply",
                "source_appearance_asset_digest": source_digest,
                "source_asset_format": "standard_3dgs_ply",
                "appearance_asset_reference": artifact_reference,
                "appearance_asset_digest": output_digest,
                "appearance_asset_format": "particlefield_usd",
                "source_prim_path": "/World/Appearance",
                "splat_count": authored["splat_count"],
                "sh_degree": authored["sh_degree"],
                "captured_observation": False,
                "raw_evidence": False,
                "metric_geometry_proven": False,
                "collision_geometry_proven": False,
                "heldout_evaluated": False,
            }
        )
        write_json(temporary / "appearance_asset_manifest.v1.json", manifest)
        os.replace(temporary, final_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "APPEARANCE_ASSET_MANIFEST_SCHEMA",
    "AppearanceAssetContractError",
    "build_appearance_asset_manifest",
    "compile_particlefield_appearance_asset",
]
