"""Automatic, claim-limited handoff from published Postshot output.

The post-capture evidence spine is the existing downstream analysis boundary.
This adapter invokes it with the exact admitted Raw V3.2 validation and the
immutable Postshot publication.  Reconstruction is appearance evidence only:
the spine remains responsible for completing or abstaining on geometry,
registration, collision, and task evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .post_capture_evidence_spine import (
    build_native_3dgs_candidate,
    build_source_profile,
    run_post_capture_evidence_spine,
)


class CaptureReconstructionDownstreamError(RuntimeError):
    pass


def _publication_artifact(
    publication: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in publication.get("artifacts", [])
        if isinstance(row, Mapping) and row.get("kind") == kind
    ]
    if len(matches) != 1:
        raise CaptureReconstructionDownstreamError(
            f"capture_reconstruction_publication_{kind}_ambiguous"
        )
    return matches[0]


def dispatch_postshot_to_evidence_spine(
    *,
    capture_id: str,
    capture_digest: str,
    raw_root: str | Path,
    derived_root: str | Path,
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the existing downstream path once over exact published evidence."""

    raw = Path(raw_root).expanduser().resolve(strict=True)
    derived = Path(derived_root).expanduser().resolve(strict=True)
    validation_paths = sorted(derived.rglob("arkit_raw_contract_validation.json"))
    if len(validation_paths) != 1:
        raise CaptureReconstructionDownstreamError(
            "capture_reconstruction_raw_validation_ambiguous"
        )
    validation = json.loads(validation_paths[0].read_text(encoding="utf-8"))
    if validation.get("source_capture_digest") != capture_digest:
        raise CaptureReconstructionDownstreamError(
            "capture_reconstruction_raw_validation_digest_mismatch"
        )
    if publication.get("capture_digest") != capture_digest:
        raise CaptureReconstructionDownstreamError(
            "capture_reconstruction_publication_digest_mismatch"
        )

    source_profile = build_source_profile(
        source_artifact=validation,
        source_root=raw,
    )
    ply = _publication_artifact(publication, kind="standard_3dgs_ply")
    provider_receipt: dict[str, Any] = {
        "schema_version": "postshot_immutable_publication_receipt.v1",
        "status": "candidate_artifacts_published",
        "source_capture_digest": capture_digest,
        "publication_digest": publication.get("publication_digest"),
        "artifacts": list(publication.get("artifacts") or []),
        "provider_identity": "postshot",
        "provider_self_qualified": False,
        "metric_alignment_qualified": False,
        "collision_geometry_qualified": False,
        "physical_task_success_proven": False,
    }
    provider_receipt["provider_receipt_digest"] = canonical_digest(
        provider_receipt, digest_field="provider_receipt_digest"
    )
    appearance = build_native_3dgs_candidate(
        source_profile=source_profile,
        provider_receipt=provider_receipt,
        appearance_asset_digest=str(ply["digest"]),
        provider_identity="postshot",
        provider_receipt_digest_field="provider_receipt_digest",
        full_resolution_appearance_preserved=True,
    )

    result = run_post_capture_evidence_spine(
        run_id=f"capture-{capture_id}-{capture_digest[7:23]}",
        source_artifact=validation,
        source_root=raw,
        output_root=derived / "post-capture-evidence-runs",
        appearance_candidate=appearance,
    )
    terminal = dict(result["terminal"])
    dispatch: dict[str, Any] = {
        "schema_version": "capture_reconstruction_downstream_dispatch.v1",
        "status": "completed" if terminal.get("status") == "completed" else "abstained",
        "entrypoint": "blueprint_pipeline.post_capture_evidence_spine",
        "capture_id": capture_id,
        "capture_digest": capture_digest,
        "publication_digest": publication["publication_digest"],
        "native_3dgs_candidate_digest": appearance["native_3dgs_candidate_digest"],
        "post_capture_evidence_run_digest": result["manifest"][
            "post_capture_evidence_run_digest"
        ],
        "terminal_stage": terminal.get("terminal_stage"),
        "smallest_missing_measurement": terminal.get("smallest_missing_measurement"),
        "run_root": str(result["run_root"]),
        "claim_ceiling": "candidate_artifacts_only",
        "metric_alignment_qualified": False,
        "physical_truth_inferred": False,
    }
    dispatch["dispatch_digest"] = canonical_digest(
        dispatch, digest_field="dispatch_digest"
    )
    return dispatch


__all__ = [
    "CaptureReconstructionDownstreamError",
    "dispatch_postshot_to_evidence_spine",
]
