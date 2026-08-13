"""Deterministic status ledger for fresh-scene paired-target preparation.

The scientific producers already exist as independent, digest-bound modules.
This ledger makes their ordering a production contract and reports the first
missing producer instead of allowing a later ArtiFixer adapter to emit a vague
calibrated-preflight error.  It is mutation-free: paid stages still launch only
through :mod:`blueprint_pipeline.paid_resource_allocator` and semantic-editor
authorization remains a separate, rights-aware gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import validate_task_freeze_set


SCHEMA_VERSION = "fresh_scene_paired_target_preparation.v1"


STAGE_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "stage_id": "calibrated_scene_views",
        "schemas": ("public_scene_interiorgs_edit_input_receipt.v2",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": ("render_derived_input_packet_materialized",),
        "cardinality": "per_task",
        "producer": "public_scene_inpainting_inputs",
        "implementation": "blueprint_pipeline.public_scene_inpainting_inputs",
        "backend": "purpose-bound exact calibrated 3DGS renderer",
        "next_blocker": "fresh_scene_calibrated_scene_views_missing",
    },
    {
        "stage_id": "sam31_task_inputs",
        "schemas": ("public_scene_sam31_task_input_packet.v1",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": ("prepared_no_upload_no_execution",),
        "cardinality": "per_task",
        "producer": "public_scene_sam31_task_inputs",
        "implementation": "blueprint_pipeline.public_scene_sam31_task_inputs",
        "backend": "deterministic calibrated PNG plus FFV1 and JPEG packet builder",
        "next_blocker": "fresh_scene_sam31_task_inputs_missing",
    },
    {
        "stage_id": "sam31_source_tracks",
        "schemas": ("semantic_source_track_import_result.v1",),
        "digest_fields": ("result_digest",),
        "accepted_statuses": ("completed",),
        "cardinality": "per_task",
        "producer": "semantic_sam31_gpu_canary",
        "implementation": "blueprint_pipeline.sam31_paid_resource_allocator_lane",
        "backend": "Meta SAM 3.1 Object Multiplex",
        "next_blocker": "fresh_scene_sam31_source_tracks_missing",
    },
    {
        "stage_id": "sam31_track_selection_review",
        "schemas": ("public_scene_sam31_track_selection_review.v1",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": ("selected_tracks_human_review_accepted",),
        "cardinality": "one",
        "producer": "sam31_track_selection_review",
        "implementation": (
            "blueprint_pipeline.public_scene_sam31_track_selection_review"
        ),
        "backend": "deterministic selected-mask overlays plus named human acceptance",
        "next_blocker": "fresh_scene_sam31_track_selection_review_missing",
    },
    {
        "stage_id": "calibrated_object_masks",
        "schemas": ("public_scene_calibrated_object_mask_set.v1",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": (
            "calibrated_inferred_object_masks_materialized_pending_review",
        ),
        "cardinality": "one",
        "producer": "calibrated_source_track_mask_bridge",
        "implementation": "blueprint_pipeline.public_scene_calibrated_object_masks",
        "backend": "deterministic sparse-RLE to binary-PNG materializer",
        "next_blocker": "fresh_scene_calibrated_object_masks_missing",
    },
    {
        "stage_id": "excision_freezes",
        "schemas": ("adp009b_gaussian_excision_audit_freeze.v1",),
        "digest_fields": ("freeze_digest",),
        "accepted_statuses": ("frozen_before_excision_execution",),
        "cardinality": "per_task",
        "producer": "registered_excision_freeze",
        "implementation": "blueprint_pipeline.public_scene_gaussian_excision_audit",
        "backend": "OpenUSD plus deterministic calibrated mask projection",
        "next_blocker": "fresh_scene_excision_freezes_missing",
    },
    {
        "stage_id": "segment_sweep_freezes",
        "schemas": ("adp009b_gaussian_excision_audit_freeze.v1",),
        "digest_fields": ("freeze_digest",),
        "accepted_statuses": ("frozen_before_excision_execution",),
        "cardinality": "per_task",
        "producer": "segment_contribution_sweep_freeze",
        "implementation": "blueprint_pipeline.public_scene_segment_contribution_cutout",
        "backend": "deterministic all-calibrated-view freeze",
        "next_blocker": "fresh_scene_segment_sweep_freezes_missing",
    },
    {
        "stage_id": "gaussian_contribution_evidence",
        "schemas": ("adp009b_gaussian_excision_contribution_evidence.v1",),
        "digest_fields": ("manifest_digest",),
        "accepted_statuses": (),
        "cardinality": "per_task",
        "producer": "adp_gaussian_excision",
        "implementation": "blueprint_pipeline.adp_gaussian_excision_vast",
        "backend": "FlashSplat 3e3b147 plus rasterizer 189c483",
        "next_blocker": "fresh_scene_gaussian_contribution_evidence_missing",
    },
    {
        "stage_id": "segment_cutout_set",
        "schemas": ("adp009d_segment_contribution_cutout_set.v1",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": (
            "repair_supported_segment_contribution_cutout_materialized_"
            "pending_full_deleted_layer_projection",
        ),
        "cardinality": "one",
        "producer": "segment_contribution_cutout_set",
        "implementation": "blueprint_pipeline.public_scene_segment_contribution_cutout",
        "backend": "deterministic exact-index standard-3DGS subset writer",
        "next_blocker": "fresh_scene_segment_cutout_set_missing",
    },
    {
        "stage_id": "segment_repair_preflight",
        "schemas": ("public_scene_calibrated_exact_segment_repair_preflight.v1",),
        "digest_fields": ("preflight_digest",),
        "accepted_statuses": ("prepared_no_upload_no_execution",),
        "cardinality": "one",
        "producer": "exact_segment_repair_preflight",
        "implementation": "blueprint_pipeline.public_scene_segment_mask_repair_preflight",
        "backend": "deterministic rights and locality admission",
        "next_blocker": "fresh_scene_segment_repair_preflight_missing",
    },
    {
        "stage_id": "artifixer_candidate_inputs",
        "schemas": ("public_scene_artifixer3d_candidate_inputs.v3",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": ("candidate_inputs_prepared_no_model_no_execution",),
        "cardinality": "one",
        "producer": "artifixer_candidate_input_materializer",
        "implementation": "blueprint_pipeline.public_scene_artifixer3d_candidate_inputs",
        "backend": "deterministic calibrated frame and PLY packaging",
        "next_blocker": "fresh_scene_artifixer_candidate_inputs_missing",
    },
    {
        "stage_id": "semantic_teacher_receipts",
        "schemas": ("public_scene_whole_frame_semantic_teacher_candidates.v1",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": ("whole_frame_semantic_teacher_candidates_unreviewed",),
        "cardinality": "per_task",
        "producer": "rights_admitted_semantic_editor",
        "implementation": "blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs",
        "backend": "gpt-image-2 preferred; pinned local editor fallback",
        "next_blocker": "fresh_scene_semantic_teacher_receipts_missing",
    },
    {
        "stage_id": "dual_target_artifixer_inputs",
        "schemas": ("public_scene_artifixer3d_dual_target_inputs.v1",),
        "digest_fields": ("receipt_digest",),
        "accepted_statuses": ("paired_target_inputs_prepared_no_model_no_execution",),
        "cardinality": "one",
        "producer": "dual_target_artifixer3d_input_materializer",
        "implementation": "blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs",
        "backend": "same-pose original anchors plus whole-frame teachers",
        "next_blocker": "fresh_scene_dual_target_artifixer_inputs_missing",
    },
    {
        "stage_id": "artifixer3d_result",
        "schemas": ("public_scene_artifixer3d_vast_run.v1",),
        "digest_fields": (),
        "accepted_statuses": ("completed",),
        "cardinality": "one",
        "producer": "adp_artifixer3d_exact_support",
        "implementation": "blueprint_pipeline.public_scene_artifixer3d_vast",
        "backend": "ArtiFixer3D paired-target then optional 3D+",
        "next_blocker": "fresh_scene_artifixer3d_result_missing",
    },
)


class FreshScenePreparationError(ValueError):
    """Raised for malformed inventory inputs, never ordinary missing stages."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshScenePreparationError("fresh_scene_artifact_unreadable") from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise FreshScenePreparationError("fresh_scene_artifact_invalid")
    return value


def _paths(value: Any, *, cardinality: str, task_ids: Sequence[str]) -> list[Path]:
    if value is None:
        return []
    if cardinality == "one":
        raw = [value]
    else:
        if not isinstance(value, Mapping) or set(value) != set(task_ids):
            return []
        raw = [value[task_id] for task_id in task_ids]
    result: list[Path] = []
    for item in raw:
        unresolved = Path(str(item or "")).expanduser()
        if unresolved.is_symlink():
            return []
        path = unresolved.resolve()
        if not path.is_file():
            return []
        result.append(path)
    return result


def _validate_stage(
    contract: Mapping[str, Any], paths: Sequence[Path], *, task_count: int
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    expected = task_count if contract["cardinality"] == "per_task" else 1
    if len(paths) != expected:
        return False, [], [contract["next_blocker"]]
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in paths:
        try:
            value = _read(path)
        except FreshScenePreparationError:
            blockers.append(f"fresh_scene_stage_artifact_invalid:{contract['stage_id']}")
            continue
        digest_fields = tuple(contract["digest_fields"])
        digest_valid = True
        if digest_fields:
            present = [field for field in digest_fields if field in value]
            digest_valid = len(present) == 1 and value[present[0]] == canonical_digest(
                value, digest_field=present[0]
            )
        accepted_statuses = tuple(contract["accepted_statuses"])
        if (
            value.get("schema_version") not in contract["schemas"]
            or (accepted_statuses and value.get("status") not in accepted_statuses)
            or not digest_valid
        ):
            blockers.append(f"fresh_scene_stage_artifact_invalid:{contract['stage_id']}")
            continue
        records.append(_record(path))
    return not blockers and len(records) == expected, records, blockers


def materialize_fresh_scene_preparation_status(
    *,
    task_freeze_paths: Sequence[str | Path],
    artifacts: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Inspect the producer chain and seal the earliest actionable blocker."""

    paths = [Path(path).expanduser().resolve() for path in task_freeze_paths]
    try:
        tasks = [_read(path) for path in paths]
        task_set = validate_task_freeze_set(tasks)
    except (FreshScenePreparationError, ValueError) as exc:
        raise FreshScenePreparationError("fresh_scene_task_freeze_set_invalid") from exc
    task_ids = tuple(sorted(str(task["task_id"]) for task in tasks))
    allowed = {contract["stage_id"] for contract in STAGE_CONTRACTS}
    if not isinstance(artifacts, Mapping) or set(artifacts) - allowed:
        raise FreshScenePreparationError("fresh_scene_artifact_inventory_invalid")

    rows: list[dict[str, Any]] = []
    first_blocker: str | None = None
    blocked_upstream = False
    for contract in STAGE_CONTRACTS:
        stage_paths = _paths(
            artifacts.get(contract["stage_id"]),
            cardinality=contract["cardinality"],
            task_ids=task_ids,
        )
        if blocked_upstream:
            ready = False
            records: list[dict[str, Any]] = []
            blockers = ["fresh_scene_stage_waiting_on_upstream"]
            status = "waiting_on_upstream"
        else:
            ready, records, blockers = _validate_stage(
                contract, stage_paths, task_count=len(task_ids)
            )
            status = "completed" if ready else "blocked"
            if not ready:
                first_blocker = blockers[0]
                blocked_upstream = True
        rows.append(
            {
                "ordinal": len(rows) + 1,
                "stage_id": contract["stage_id"],
                "status": status,
                "cardinality": contract["cardinality"],
                "producer": contract["producer"],
                "implementation": contract["implementation"],
                "backend": contract["backend"],
                "artifacts": records,
                "blockers": blockers,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "status": "ready_for_visual_qualification" if first_blocker is None else "blocked",
        "task_count": len(task_ids),
        "task_ids": list(task_ids),
        "task_freeze_set_digest": task_set["set_digest"],
        "task_freezes": [_record(path) for path in paths],
        "stages": rows,
        "first_blocker": first_blocker,
        "next_required_stage": next(
            (row["stage_id"] for row in rows if row["status"] == "blocked"), None
        ),
        "production_contract": {
            "maximum_task_objects": 5,
            "paid_mutations_only_through_paid_resource_allocator": True,
            "automatic_paid_retry_authorized": False,
            "canonical_interiorgs_mutation_permitted": False,
            "raw_nonredistributable_upload_permitted": False,
            "agent_outputs_are_candidate_support_only": True,
            "simulator_outputs_are_physical_evidence": False,
        },
        "status_digest": "",
    }
    payload["status_digest"] = canonical_digest(payload, digest_field="status_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FreshScenePreparationError("fresh_scene_status_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-freeze", action="append", required=True)
    parser.add_argument(
        "--artifact-inventory",
        required=True,
        help="JSON object mapping stage IDs to one path or a task_id-to-path map.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    inventory_path = Path(args.artifact_inventory).expanduser().resolve()
    inventory = _read(inventory_path)
    materialize_fresh_scene_preparation_status(
        task_freeze_paths=args.task_freeze,
        artifacts=inventory,
        output_path=args.output,
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "STAGE_CONTRACTS",
    "FreshScenePreparationError",
    "materialize_fresh_scene_preparation_status",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
