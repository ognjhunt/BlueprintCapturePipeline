"""Bind a gap review to retained source selection and derived output previews.

ADP-040 / Day 28 development-only preparation. No model-derived preview becomes
a capture observation, scale measurement, or admission decision.
"""
from __future__ import annotations

import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def load_refinement_feedback(path: Path, *, source_digest: str, context_digest: str) -> dict:
    value = json.loads(path.read_text())
    if (value.get("schema_version") != "atlas_reconstruction_feedback.v1"
            or value.get("digest") != canonical_digest(value, digest_field="digest")):
        raise ValueError("atlas_refinement_feedback_invalid")
    parent = Path(value["parent_root"]).resolve()
    result = json.loads((parent / "result.json").read_text())
    selection = json.loads((parent / "pose/atlas_view_selection.json").read_text())
    if (result.get("digest") != canonical_digest(result, digest_field="digest")
            or result["digest"] != value["parent_result_digest"]
            or result["source_video_digest"] != source_digest
            or result["task_context_digest"] != context_digest
            or _sha256_file(parent / "pose/atlas_view_selection.json") != value["parent_selection_sha256"]
            or not result.get("splat_available") or result.get("claim_ceiling") != "development_only"):
        raise ValueError("atlas_refinement_parent_changed")
    selected = [r["frame_id"] for r in selection["frames"]]
    previews = value.get("previews")
    if not isinstance(previews, list) or not previews:
        raise ValueError("atlas_refinement_previews_required")
    resolved = []
    for row in previews:
        image = path.parent / row["relative_path"]
        if (image.is_symlink() or not image.resolve().is_relative_to(path.parent.resolve())
                or image.suffix != ".png" or row["source_frame_id"] not in selected
                or _sha256_file(image) != row["sha256"]):
            raise ValueError("atlas_refinement_preview_invalid")
        resolved.append({**row, "path": str(image.resolve())})
    return {**value, "previews": resolved, "prior_review": selection["review"],
            "prior_selected_frame_ids": selected, "parent_root": str(parent),
            "preview_authority": "derived_reconnaissance_only_not_qualified_method_inputs"}
