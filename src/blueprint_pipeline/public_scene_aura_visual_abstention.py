"""Seal a digest-bound Aura visual rejection without creating an admission path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA_VERSION = "adp009b_aura_visual_abstention_request.v1"
RECEIPT_SCHEMA_VERSION = "adp009b_aura_visual_abstention.v1"
ALLOWED_ARTIFACT_CODES = {
    "gaussian_explosion_or_rainbow_splat",
    "semantic_hallucination_in_removed_volume",
    "outside_mask_scene_damage",
    "multiview_background_inconsistency",
}


class AuraVisualAbstentionError(ValueError):
    """The rejection request is unbound or attempts to create an admission."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuraVisualAbstentionError(f"json_object_required:{path.name}")
    return value


def _under(root: Path, value: str | Path) -> Path:
    root = root.expanduser().resolve()
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if path != root and root not in path.parents:
        raise AuraVisualAbstentionError(f"path_outside_approved_root:{path}")
    return path


def materialize_aura_visual_abstention(
    *, request_path: Path, repo_root: Path, data_root: Path, output_path: Path
) -> dict[str, Any]:
    repo = repo_root.expanduser().resolve()
    data = data_root.expanduser().resolve()
    request = _read(_under(repo, request_path))
    output = _under(repo, output_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise AuraVisualAbstentionError("request_schema_invalid")
    if request.get("decision") != "reject_visual_candidate":
        raise AuraVisualAbstentionError("visual_rejection_decision_required")
    if request.get("reviewer_role") != "evidence_operator":
        raise AuraVisualAbstentionError("evidence_operator_role_required")
    forbidden = {"accept", "admit", "qualified", "quality_pass_claimed"}
    if forbidden.intersection(request):
        raise AuraVisualAbstentionError("caller_asserted_admission_forbidden")
    codes = request.get("observed_artifact_codes")
    if (
        not isinstance(codes, list)
        or not codes
        or any(code not in ALLOWED_ARTIFACT_CODES for code in codes)
    ):
        raise AuraVisualAbstentionError("observed_artifact_codes_invalid")

    execution_path = _under(repo, str(request.get("aura_execution_receipt_path") or ""))
    locality_path = _under(data, str(request.get("locality_measurement_path") or ""))
    execution = _read(execution_path)
    locality = _read(locality_path)
    if execution.get("receipt_digest") != canonical_digest(
        execution, digest_field="receipt_digest"
    ):
        raise AuraVisualAbstentionError("aura_execution_receipt_digest_mismatch")
    if locality.get("locality_measurement_digest") != canonical_digest(
        locality, digest_field="locality_measurement_digest"
    ):
        raise AuraVisualAbstentionError("aura_locality_measurement_digest_mismatch")
    scene = execution.get("scene") or {}
    if (
        execution.get("schema_version")
        != "adp009b_aurafusion360_execution_receipt.v1"
        or execution.get("status") != "executed_candidate"
        or (execution.get("claim_boundary") or {}).get("successful_inpainting_admitted")
        is not False
        or locality.get("schema_version")
        != "public_scene_inpainting_locality_measurement.v1"
        or locality.get("status") != "measured_no_admission_effect"
        or locality.get("quality_pass_claimed") is not False
        or locality.get("thresholds_frozen_before_evaluation") is not False
        or locality.get("scene")
        != {
            "publisher_scene_id": scene.get("publisher_scene_id"),
            "target_instance_id": scene.get("target_instance_id"),
        }
    ):
        raise AuraVisualAbstentionError("aura_visual_evidence_contract_invalid")
    final_by_camera = {
        str(row.get("camera_id")): str(row.get("sha256"))
        for row in (execution.get("execution") or {}).get("final_frames", [])
    }
    locality_by_camera = {
        str(row.get("camera_id")): str(row.get("after_sha256"))
        for row in locality.get("rows") or []
    }
    if (
        len(final_by_camera) != int(scene.get("camera_count") or 0)
        or final_by_camera != locality_by_camera
    ):
        raise AuraVisualAbstentionError("aura_locality_frame_join_invalid")

    aggregate = locality.get("aggregate") or {}
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "abstained_visual_artifact_rejection",
        "decision": request["decision"],
        "reviewer_role": request["reviewer_role"],
        "scene": {
            "publisher_scene_id": scene.get("publisher_scene_id"),
            "target_instance_id": scene.get("target_instance_id"),
            "target_semantic_label": scene.get("target_semantic_label"),
            "camera_count": scene.get("camera_count"),
        },
        "bindings": {
            "aura_execution_receipt_digest": execution["receipt_digest"],
            "locality_measurement_digest": locality["locality_measurement_digest"],
        },
        "observed_artifact_codes": sorted(set(str(code) for code in codes)),
        "outside_mask_locality": aggregate,
        "thresholds_frozen_before_evaluation": False,
        "quality_pass_claimed": False,
        "successful_inpainting_admitted": False,
        "failure_localization": (
            "stage_localization_missing_intermediate_inpaint_evidence"
            if (execution.get("quality") or {}).get(
                "intermediate_stage_artifacts_retained"
            )
            is not True
            else "stage_intermediates_retained_for_followup_diagnosis"
        ),
        "smallest_missing_capability": (
            "released_code_multiview_inpainting_that_preserves_observed_scene_outside_"
            "the_target_mask_and_produces_a_consistent_background_inside_it"
        ),
        "claim_ceiling": "rejected_visual_candidate_only",
        "blockers": ["aurafusion360_interiorgs_visual_artifact_rejection"],
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_aura_visual_abstention(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
