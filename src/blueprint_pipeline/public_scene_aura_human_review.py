"""Bind project-owner review of the executed Aura InteriorGS visual candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "adp009b_aura_human_visual_review_request.v1"
RECEIPT_SCHEMA_VERSION = "adp009b_aura_human_visual_review.v1"


class AuraHumanReviewError(ValueError):
    """The requested review is unbound or exceeds human-review authority."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuraHumanReviewError(f"json_object_required:{path.name}")
    return value


def _under(root: Path, relative: str) -> Path:
    root = root.expanduser().resolve()
    path = (root / relative).expanduser().resolve()
    if root not in path.parents:
        raise AuraHumanReviewError(f"path_outside_approved_root:{path}")
    return path


def _explicit_under(root: Path, path: Path) -> Path:
    root = root.expanduser().resolve()
    if not path.is_absolute():
        path = root / path
    path = path.expanduser().resolve()
    if path != root and root not in path.parents:
        raise AuraHumanReviewError(f"path_outside_approved_root:{path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verify_file(path: Path, record: Mapping[str, Any], *, error: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise AuraHumanReviewError(error)


def materialize_aura_human_review(
    *, request_path: Path, repo_root: Path, data_root: Path, output_path: Path
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    request_path = _explicit_under(repo_root, request_path)
    output_path = _explicit_under(repo_root, output_path)
    request = _read(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise AuraHumanReviewError("request_schema_invalid")
    if {"status", "admitted", "qualified", "technical_admission"}.intersection(request):
        raise AuraHumanReviewError("caller_asserted_technical_admission_forbidden")
    if (
        request.get("reviewer_role") != "project_owner"
        or request.get("decision") != "accept_for_internal_hybrid_replacement_control"
        or not str(request.get("approval_statement") or "").strip()
    ):
        raise AuraHumanReviewError("project_owner_visual_decision_missing")

    aura_path = _under(repo_root, str(request["aura_execution_receipt_path"]))
    locality_path = _under(data_root, str(request["locality_measurement_path"]))
    native_path = _under(data_root, str(request["native_visual_review_receipt_path"]))
    native_root = _under(data_root, str(request["native_visual_review_root"]))
    aura = _read(aura_path)
    locality = _read(locality_path)
    native = _read(native_path)
    if aura.get("receipt_digest") != canonical_digest(aura, digest_field="receipt_digest"):
        raise AuraHumanReviewError("aura_execution_receipt_digest_mismatch")
    if (
        aura.get("schema_version") != "adp009b_aurafusion360_execution_receipt.v1"
        or aura.get("status") != "executed_candidate"
        or (aura.get("scene") or {}).get("publisher_scene_id") != "840313"
        or (aura.get("scene") or {}).get("target_instance_id") != "ins160"
        or (aura.get("claim_boundary") or {}).get("successful_inpainting_admitted")
        is not False
    ):
        raise AuraHumanReviewError("aura_execution_receipt_invalid")

    if (
        locality.get("locality_measurement_digest")
        != canonical_digest(locality, digest_field="locality_measurement_digest")
        or locality.get("schema_version")
        != "public_scene_inpainting_locality_measurement.v1"
        or locality.get("status") != "measured_no_admission_effect"
        or locality.get("admission_effect") != "none"
        or locality.get("quality_pass_claimed") is not False
        or locality.get("thresholds_frozen_before_evaluation") is not False
        or (locality.get("aggregate") or {}).get("view_count") != 8
    ):
        raise AuraHumanReviewError("aura_locality_measurement_invalid")

    if native.get("receipt_digest") != canonical_digest(native, digest_field="receipt_digest"):
        raise AuraHumanReviewError("native_visual_review_receipt_digest_mismatch")
    artifacts = native.get("artifacts")
    if (
        native.get("schema_version")
        != "adp009b_simready_native_visual_review_receipt.v1"
        or native.get("status") != "rendered_native_visual_review_candidate"
        or native.get("renderer_is_native_ovrtx") is not True
        or native.get("background_renderer")
        != "aurafusion360_native_2d_gaussian_rasterizer"
        or native.get("human_visual_acceptance") != "pending"
        or native.get("technical_admission") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 8
    ):
        raise AuraHumanReviewError("native_visual_review_receipt_invalid")

    aura_frame_hashes = {
        row.get("sha256") for row in (aura.get("execution") or {}).get("final_frames", [])
    }
    native_frame_hashes: set[str] = set()
    camera_ids: list[str] = []
    review_files: list[dict[str, Any]] = []
    for row in artifacts:
        if not isinstance(row, Mapping) or not str(row.get("camera_id") or ""):
            raise AuraHumanReviewError("native_visual_review_artifact_invalid")
        camera_ids.append(str(row["camera_id"]))
        native_frame_hashes.add(str(row.get("source_frame_sha256")))
        for role in ("before", "after", "before_after", "contact_crop_before_after"):
            record = row.get(role)
            if not isinstance(record, Mapping):
                raise AuraHumanReviewError(f"native_visual_review_file_missing:{role}")
            path = _under(native_root, str(record.get("relative_path") or ""))
            _verify_file(path, record, error=f"native_visual_review_file_changed:{role}")
            review_files.append(
                {
                    "camera_id": row["camera_id"],
                    "role": role,
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if len(aura_frame_hashes) != 8 or native_frame_hashes != aura_frame_hashes:
        raise AuraHumanReviewError("aura_native_camera_frame_join_invalid")

    aggregate = locality["aggregate"]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "human_accepted_visual_candidate_for_internal_hybrid_control",
        "reviewer_role": request["reviewer_role"],
        "decision": request["decision"],
        "approval_statement": request["approval_statement"],
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "ins160",
            "camera_ids": camera_ids,
        },
        "bindings": {
            "aura_execution_receipt_digest": aura["receipt_digest"],
            "aura_locality_measurement_digest": locality["locality_measurement_digest"],
            "native_visual_review_receipt_digest": native["receipt_digest"],
            "review_files": review_files,
        },
        "observed_quality": {
            "view_count": aggregate["view_count"],
            "mean_outside_mask_psnr_db": aggregate["mean_outside_mask_psnr_db"],
            "mean_outside_mask_windowed_ssim": aggregate[
                "mean_outside_mask_windowed_ssim"
            ],
            "mean_outside_mask_lpips": aggregate["mean_outside_mask_lpips"],
            "human_visual_acceptance": True,
        },
        "technical_admission": False,
        "successful_inpainting_admitted": False,
        "hidden_background_truth_available": False,
        "claim_ceiling": "project_owner_accepted_internal_visual_candidate_only",
        "claim_boundaries": {
            "human_review_does_not_create_hidden_background_truth": True,
            "native_object_layer_composite_is_not_full_scene_native_render": True,
            "simulation_or_physical_truth": False,
            "digital_twin": False,
        },
        "blockers": ["digest_bound_hybrid_replacement_seal_missing"],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_aura_human_review(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(json.dumps({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
