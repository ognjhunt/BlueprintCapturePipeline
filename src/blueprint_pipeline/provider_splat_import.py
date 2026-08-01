"""Typed import and candidate-side alignment for provider splat outputs.

Teleport and Postshot results enter Blueprint only through this lane: the
provider-native bytes are digest-verified, decoded as standard 3DGS data,
validated for finite/bounded values, and preserved unchanged, while alignment
into the candidate camera frame uses candidate cameras only.  Provider success
never becomes Blueprint qualification here; hidden held-out pixels and cameras
are never read, and no metric, collision, Isaac, task, physical, or deployment
claim can be produced by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import read_standard_3dgs_ply
from .rigid_alignment import (
    SimilarityAlignmentError,
    estimate_similarity_transform,
    similarity_residuals,
)


IMPORT_REQUEST_SCHEMA_VERSION = "provider_splat_import_request.v1"
IMPORT_RECEIPT_SCHEMA_VERSION = "provider_splat_import_receipt.v1"
ALIGNMENT_SCHEMA_VERSION = "provider_reconstruction_alignment.v1"

SUPPORTED_PROVIDERS = {"teleport", "postshot"}
SPLAT_ARTIFACT_KINDS = {"splat_ply", "splat_spz"}
SUPPORTED_ARTIFACT_KINDS = SPLAT_ARTIFACT_KINDS | {
    "cameras_metadata",
    "training_log",
    "project_file",
}
MAX_ASSET_BYTES = 4_000_000_000
MAX_SPLAT_COUNT = 60_000_000
MAX_ABS_POSITION = 1.0e6
MIN_ALIGNMENT_PAIRS = 8
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ProviderSplatImportError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_source(root: Path, relative_path: str, *, label: str) -> Path:
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "evaluator_hidden" in relative.parts
        or "held_out" in relative.parts
    ):
        raise ProviderSplatImportError([f"{label}_path_unsafe_or_hidden"])
    resolved_root = root.resolve()
    lexical = resolved_root / Path(*relative.parts)
    if lexical.is_symlink():
        raise ProviderSplatImportError([f"{label}_symlink_forbidden"])
    path = lexical.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ProviderSplatImportError([f"{label}_path_escape"])
    if not path.is_file():
        raise ProviderSplatImportError([f"{label}_missing"])
    if path.stat().st_size > MAX_ASSET_BYTES:
        raise ProviderSplatImportError([f"{label}_oversized"])
    return path


def build_provider_splat_import_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = json.loads(canonical_json(dict(value)))
    errors: list[str] = []
    if request.get("schema_version") != IMPORT_REQUEST_SCHEMA_VERSION:
        errors.append("provider_import_schema_invalid")
    if request.get("provider_identity") not in SUPPORTED_PROVIDERS:
        errors.append("provider_import_provider_unsupported")
    for key in (
        "source_capture_digest",
        "frozen_split_digest",
        "consumed_candidate_dataset_digest",
        "provider_execution_receipt_digest",
    ):
        if not _is_digest(request.get(key)):
            errors.append(f"provider_import_{key}_invalid")
    if _COMMIT.fullmatch(str(request.get("source_commit_sha") or "")) is None:
        errors.append("provider_import_source_commit_invalid")
    for key in ("stable_run_identity", "provider_job_identity", "timestamp"):
        if not str(request.get(key) or "").strip():
            errors.append(f"provider_import_{key}_missing")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("provider_import_authority_missing")
    if request.get("provider_had_hidden_access") is not False:
        errors.append("provider_import_hidden_access_not_false")
    if request.get("hidden_heldout_pixels_included") is not False:
        errors.append("provider_import_hidden_pixels_not_false")
    if (
        request.get("proof_effect") != "provider_output_import_request_only"
        or request.get("claim_ceiling") != "none"
    ):
        errors.append("provider_import_claim_boundary_invalid")
    bindings = request.get("asset_bindings")
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 16:
        errors.append("provider_import_asset_bindings_invalid")
        bindings = []
    splat_count = 0
    seen_ids: set[str] = set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            errors.append(f"provider_import_asset_binding_not_object:{index}")
            continue
        asset_id = str(binding.get("asset_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", asset_id) or asset_id in seen_ids:
            errors.append(f"provider_import_asset_id_invalid:{index}")
        seen_ids.add(asset_id)
        if binding.get("artifact_kind") not in SUPPORTED_ARTIFACT_KINDS:
            errors.append(f"provider_import_artifact_kind_invalid:{index}")
        elif binding["artifact_kind"] in SPLAT_ARTIFACT_KINDS:
            splat_count += 1
        if not _is_digest(binding.get("digest")):
            errors.append(f"provider_import_asset_digest_invalid:{index}")
        relative = PurePosixPath(str(binding.get("relative_path") or "").replace("\\", "/"))
        if (
            not str(binding.get("relative_path") or "")
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            errors.append(f"provider_import_asset_path_invalid:{index}")
    if splat_count != 1:
        errors.append("provider_import_exactly_one_splat_asset_required")
    supplied = request.pop("provider_splat_import_request_digest", None)
    request["provider_splat_import_request_digest"] = canonical_digest(
        request, digest_field="provider_splat_import_request_digest"
    )
    if supplied is not None and supplied != request["provider_splat_import_request_digest"]:
        errors.append("provider_import_request_digest_mismatch")
    if errors:
        raise ProviderSplatImportError(errors)
    return request


def _splat_inventory(path: Path) -> dict[str, Any]:
    try:
        splat = read_standard_3dgs_ply(path)
    except (OSError, ValueError) as exc:
        raise ProviderSplatImportError(["provider_splat_ply_not_standard_3dgs"]) from exc
    if splat.count <= 0 or splat.count > MAX_SPLAT_COUNT:
        raise ProviderSplatImportError(["provider_splat_count_out_of_bounds"])
    arrays = {
        "xyz": splat.xyz,
        "opacity": splat.opacity,
        "f_dc": splat.f_dc,
        "scales": splat.scales,
        "quats": splat.quats,
    }
    if splat.sh_rest is not None:
        arrays["sh_rest"] = splat.sh_rest
    nonfinite = {
        name: int(np.size(values) - np.isfinite(values).sum())
        for name, values in arrays.items()
    }
    if any(count > 0 for count in nonfinite.values()):
        raise ProviderSplatImportError(["provider_splat_values_nonfinite"])
    if float(np.max(np.abs(splat.xyz))) > MAX_ABS_POSITION:
        raise ProviderSplatImportError(["provider_splat_positions_out_of_bounds"])
    quat_norms = np.linalg.norm(splat.quats, axis=1)
    minimum, maximum = splat.aabb()
    sh_degree = 0
    if splat.sh_rest is not None:
        sh_degree = int(round((1 + splat.sh_rest.shape[1] // 3) ** 0.5)) - 1
    return {
        "splat_count": int(splat.count),
        "sh_degree": sh_degree,
        "properties": list(splat.properties),
        "bounds_min": [round(float(v), 6) for v in minimum],
        "bounds_max": [round(float(v), 6) for v in maximum],
        "opacity_sigmoid_mean": round(float(np.mean(splat.opacity_sigmoid)), 6),
        "log_scale_max": round(float(np.max(splat.scales)), 6),
        "log_scale_min": round(float(np.min(splat.scales)), 6),
        "degenerate_quaternion_count": int(np.sum(quat_norms < 1e-8)),
        "nonfinite_counts": nonfinite,
    }


def import_provider_splat(
    *,
    source_artifact: Mapping[str, Any],
    artifact_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify and preserve provider-native splat output, unchanged, with inventory."""

    request = build_provider_splat_import_request(source_artifact)
    root = Path(artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ProviderSplatImportError(["provider_import_artifact_root_invalid"])
    output = Path(output_root)
    if output.is_symlink():
        raise ProviderSplatImportError(["provider_import_output_root_symlink_forbidden"])
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    content_id = request["provider_splat_import_request_digest"][7:23]
    final_dir = output / f"provider_import_{content_id}"
    receipt_path = final_dir / "provider_splat_import_receipt.v1.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("provider_splat_import_receipt_digest") != canonical_digest(
            receipt, digest_field="provider_splat_import_receipt_digest"
        ):
            raise ProviderSplatImportError(["provider_import_replay_receipt_tampered"])
        for asset in receipt.get("imported_assets", []):
            path = output / str(asset.get("relative_path") or "")
            if not path.is_file() or _sha256_file(path) != asset.get("digest"):
                raise ProviderSplatImportError(["provider_import_replay_asset_tampered"])
        return receipt
    if final_dir.exists() or final_dir.is_symlink():
        raise ProviderSplatImportError(["provider_import_existing_output_incomplete"])
    sources = []
    for binding in request["asset_bindings"]:
        path = _safe_source(root, str(binding["relative_path"]), label="provider_asset")
        if _sha256_file(path) != binding["digest"]:
            raise ProviderSplatImportError(["provider_asset_digest_mismatch"])
        sources.append((binding, path))
    inventory: dict[str, Any] | None = None
    temporary = Path(tempfile.mkdtemp(prefix=".provider-import-", dir=output))
    try:
        assets_dir = temporary / "assets"
        assets_dir.mkdir()
        imported_assets = []
        for index, (binding, source) in enumerate(sources):
            destination = assets_dir / f"{index:02d}-{binding['asset_id']}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            digest = _sha256_file(destination)
            if digest != binding["digest"]:
                raise ProviderSplatImportError(["provider_asset_copy_digest_mismatch"])
            if binding["artifact_kind"] == "splat_ply":
                inventory = _splat_inventory(destination)
            imported_assets.append(
                {
                    "asset_id": binding["asset_id"],
                    "artifact_kind": binding["artifact_kind"],
                    "format": destination.suffix.lower(),
                    "digest": digest,
                    "size_bytes": destination.stat().st_size,
                    "relative_path": f"provider_import_{content_id}/assets/{destination.name}",
                    "untrusted_source_filename": source.name,
                    "metadata_treated_as_untrusted": True,
                }
            )
        if inventory is None:
            raise ProviderSplatImportError(["provider_import_splat_ply_required_for_v1"])
        receipt = {
            "schema_version": IMPORT_RECEIPT_SCHEMA_VERSION,
            "stable_run_identity": request["stable_run_identity"],
            "status": "imported_provider_appearance_candidate_only",
            "provider_identity": request["provider_identity"],
            "provider_job_identity": request["provider_job_identity"],
            "provider_execution_receipt_digest": request["provider_execution_receipt_digest"],
            "source_capture_digest": request["source_capture_digest"],
            "frozen_split_digest": request["frozen_split_digest"],
            "consumed_candidate_dataset_digest": request["consumed_candidate_dataset_digest"],
            "provider_splat_import_request_digest": request[
                "provider_splat_import_request_digest"
            ],
            "source_commit_sha": request["source_commit_sha"],
            "imported_assets": imported_assets,
            "splat_inventory": inventory,
            "provider_native_output_preserved_unchanged": True,
            "provider_had_hidden_access": False,
            "hidden_heldout_pixels_included": False,
            "provider_success_is_blueprint_qualification": False,
            "raw_capture_truth": False,
            "metric_scale_proven": False,
            "collision_geometry_validated": False,
            "isaac_compatibility_proven": False,
            "simulator_task_success_proven": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
            "authority_used": dict(request["authority_used"]),
            "cost_usd": 0.0,
            "proof_effect": "provider_output_preserved_for_independent_evaluation",
            "claim_ceiling": "appearance_reconstruction_candidate",
            "parent_artifact_or_event": {
                "request_digest": request["provider_splat_import_request_digest"],
                "provider_execution_receipt_digest": request[
                    "provider_execution_receipt_digest"
                ],
            },
            "timestamp": request["timestamp"],
        }
        receipt["provider_splat_import_receipt_digest"] = canonical_digest(
            receipt, digest_field="provider_splat_import_receipt_digest"
        )
        (temporary / "provider_splat_import_receipt.v1.json").write_text(
            canonical_json(receipt) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def align_provider_reconstruction(
    *,
    import_receipt: Mapping[str, Any],
    provider_cameras: Sequence[Mapping[str, Any]],
    candidate_observations: Sequence[Mapping[str, Any]],
    image_name_to_observation_id: Mapping[str, str],
    alignment_thresholds: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Estimate the provider-frame -> candidate-frame similarity transform.

    Pairing uses provider-declared image names mapped through an explicit
    Blueprint-owned table; list order is never trusted.  Only candidate
    observation ids participate.  Reflection preference and residual gates fail
    closed.  Hidden cameras never enter: the mapping table itself is built from
    the candidate manifest only.
    """

    receipt = json.loads(canonical_json(dict(import_receipt)))
    if receipt.get("schema_version") != IMPORT_RECEIPT_SCHEMA_VERSION or receipt.get(
        "provider_splat_import_receipt_digest"
    ) != canonical_digest(receipt, digest_field="provider_splat_import_receipt_digest"):
        raise ProviderSplatImportError(["provider_alignment_import_receipt_invalid"])
    thresholds = dict(alignment_thresholds)
    for key in ("maximum_rms_residual", "maximum_max_residual"):
        number = thresholds.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise ProviderSplatImportError([f"provider_alignment_threshold_invalid:{key}"])
    candidate_by_id: dict[str, np.ndarray] = {}
    for observation in candidate_observations:
        camera = observation.get("camera")
        camera = camera if isinstance(camera, Mapping) else observation
        observation_id = str(observation.get("observation_id") or "")
        matrix = np.asarray(camera["T_world_camera"], dtype=np.float64)
        if not observation_id or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ProviderSplatImportError(["provider_alignment_candidate_observation_invalid"])
        candidate_by_id[observation_id] = matrix[:3, 3]
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    unmatched_provider = 0
    for camera in provider_cameras:
        if not isinstance(camera, Mapping):
            raise ProviderSplatImportError(["provider_alignment_provider_camera_invalid"])
        image_name = str(camera.get("image_name") or "")
        observation_id = image_name_to_observation_id.get(image_name)
        if observation_id is None:
            unmatched_provider += 1
            continue
        if observation_id not in candidate_by_id:
            raise ProviderSplatImportError(["provider_alignment_noncandidate_observation_claimed"])
        position = camera.get("position")
        if position is None and camera.get("T_world_camera") is not None:
            matrix = np.asarray(camera["T_world_camera"], dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ProviderSplatImportError(["provider_alignment_provider_pose_invalid"])
            position = matrix[:3, 3]
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ProviderSplatImportError(["provider_alignment_provider_pose_invalid"])
        pairs.append((observation_id, position, candidate_by_id[observation_id]))
    if len({observation_id for observation_id, _, _ in pairs}) != len(pairs):
        raise ProviderSplatImportError(["provider_alignment_duplicate_provider_camera"])
    if len(pairs) < MIN_ALIGNMENT_PAIRS:
        raise ProviderSplatImportError(["provider_alignment_insufficient_pairs"])
    source = np.stack([pair[1] for pair in sorted(pairs, key=lambda item: item[0])])
    target = np.stack([pair[2] for pair in sorted(pairs, key=lambda item: item[0])])
    try:
        scale, rotation, translation, reflection = estimate_similarity_transform(source, target)
    except SimilarityAlignmentError as exc:
        raise ProviderSplatImportError(list(exc.codes)) from exc
    if reflection:
        raise ProviderSplatImportError(["provider_alignment_handedness_reflection_detected"])
    rms_residual, max_residual = similarity_residuals(
        source, target, scale=scale, rotation=rotation, translation=translation
    )
    if rms_residual > float(thresholds["maximum_rms_residual"]) or max_residual > float(
        thresholds["maximum_max_residual"]
    ):
        raise ProviderSplatImportError(["provider_alignment_residual_threshold_exceeded"])
    alignment = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "stable_run_identity": receipt["stable_run_identity"],
        "status": "aligned_provider_frame_to_candidate_frame",
        "provider_identity": receipt["provider_identity"],
        "provider_splat_import_receipt_digest": receipt[
            "provider_splat_import_receipt_digest"
        ],
        "source_capture_digest": receipt["source_capture_digest"],
        "frozen_split_digest": receipt["frozen_split_digest"],
        "method": "umeyama_similarity_candidate_camera_centers.v1",
        "pair_count": len(pairs),
        "unmatched_provider_camera_count": unmatched_provider,
        "estimated_scale_factor": round(scale, 12),
        "rotation_matrix": [[round(float(v), 12) for v in row] for row in rotation],
        "translation": [round(float(v), 12) for v in translation],
        "rms_residual": round(rms_residual, 9),
        "max_residual": round(max_residual, 9),
        "thresholds": {key: float(thresholds[key]) for key in sorted(thresholds)},
        "reflection_preferred_by_alignment": False,
        "hidden_cameras_used": False,
        "camera_order_assumed": False,
        "provider_had_hidden_access": False,
        "proof_effect": "provider_alignment_candidate_frame_only",
        "claim_ceiling": "appearance_reconstruction_candidate",
        "timestamp": timestamp,
    }
    alignment["provider_reconstruction_alignment_digest"] = canonical_digest(
        alignment, digest_field="provider_reconstruction_alignment_digest"
    )
    return alignment


__all__ = [
    "ALIGNMENT_SCHEMA_VERSION",
    "IMPORT_RECEIPT_SCHEMA_VERSION",
    "IMPORT_REQUEST_SCHEMA_VERSION",
    "ProviderSplatImportError",
    "align_provider_reconstruction",
    "build_provider_splat_import_request",
    "import_provider_splat",
]
