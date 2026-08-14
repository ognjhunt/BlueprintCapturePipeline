"""Prepare ArtiFixer candidate inputs from a reviewed shared segment cutout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from .public_scene_segment_mask_repair_preflight import (
    materialize_segment_mask_repair_preflight,
)


REQUEST_SCHEMA_VERSION = "fresh_scene_artifixer_candidate_preparation_request.v1"
RECEIPT_SCHEMA_VERSION = "fresh_scene_artifixer_candidate_preparation.v1"


class FreshSceneArtiFixerCandidatePreparationError(ValueError):
    """The cutout-to-ArtiFixer candidate request is invalid."""


def materialize_fresh_scene_artifixer_candidate_preparation(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Materialize preflight plus candidate inputs, with no model execution."""

    value = dict(request)
    selected = value.get("selected_task_ids")
    references = value.get("object_absent_reference_receipt_paths", [])
    if (
        value.get("schema_version") != REQUEST_SCHEMA_VERSION
        or value.get("request_digest")
        != canonical_digest(value, digest_field="request_digest")
        or (selected is not None and not isinstance(selected, list))
        or not isinstance(references, list)
    ):
        raise FreshSceneArtiFixerCandidatePreparationError(
            "fresh_scene_artifixer_candidate_request_invalid"
        )
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise FreshSceneArtiFixerCandidatePreparationError(
            "fresh_scene_artifixer_candidate_output_not_empty"
        )
    output.mkdir(parents=True)
    preflight_path = output / "segment_repair_preflight.json"
    preflight = materialize_segment_mask_repair_preflight(
        segment_cutout_set_path=str(value.get("segment_cutout_set_path") or ""),
        execution_authority_path=str(value.get("execution_authority_path") or ""),
        output_path=preflight_path,
    )
    candidate_root = output / "candidate_inputs"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight_path,
        output_root=candidate_root,
        selected_task_ids=(
            [str(task_id) for task_id in selected] if isinstance(selected, list) else None
        ),
        object_absent_reference_receipt_paths=[str(path) for path in references],
    )
    candidate_path = candidate_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "artifixer_candidate_inputs_prepared_no_model_no_execution",
        "task_count": len(candidate["tasks"]),
        "segment_repair_preflight": {
            "path": str(preflight_path),
            "preflight_digest": preflight["preflight_digest"],
        },
        "artifixer_candidate_inputs": {
            "path": str(candidate_path),
            "receipt_digest": candidate["receipt_digest"],
        },
        "semantic_teacher_execution_started": False,
        "artifixer3d_execution_started": False,
        "provider_mutations_performed": 0,
        "canonical_source_altered": False,
        "next_required_stage": "semantic_teacher_receipts",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (output / f"{RECEIPT_SCHEMA_VERSION}.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


__all__: Sequence[str] = (
    "REQUEST_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "FreshSceneArtiFixerCandidatePreparationError",
    "materialize_fresh_scene_artifixer_candidate_preparation",
)
