"""Select photographed splat context independently of the minimal edit set.

ADP-040 / day 28. Reuse the admitted dense-survey inputs, their overlap selector,
privacy screening and static interval; never reach into the hidden split.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def select_splat_sources(*, frames, inputs, review, output_root: Path):
    import cv2
    import numpy as np
    from .atlas_video_inputs import _features, select_coverage_views, add_distinct_views

    if inputs.get("heldout_pixels_included") is not False:
        raise ValueError("atlas_splat_source_hidden_split_unverified")
    metadata = {row["frame_id"]: row for row in inputs["frames"]}
    rows = []
    for frame in frames:
        row = metadata[frame["frame_id"]]
        path = Path(row["image_path"])
        if path.is_symlink() or _sha256_file(path) != row["image_digest"]:
            raise ValueError("atlas_splat_source_changed")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("atlas_splat_source_image_invalid")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows.append({"frame_id": frame["frame_id"], "timestamp_seconds": row["timestamp_seconds"],
                     "source_image_path": str(path), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                     "clipped_fraction": float(np.mean((gray <= 3) | (gray >= 252))),
                     "image_digest": row["image_digest"]})
    rows.sort(key=lambda row: (row["timestamp_seconds"], row["frame_id"]))
    features = [_features(row, 0) for row in rows]
    requirements = list(review["coverage_requirements"])
    if review.get("additional_input_frame_ids"):
        requirements.append({"role": "support_surface", "label": "Reviewed additional real views",
                             "required_frame_ids": review["additional_input_frame_ids"]})
    cover, diagnostics = select_coverage_views(frames=rows, features=features, maximum_frames=30,
        requirements=requirements, anchor_id=review["anchor_frame_id"], policy=review.get("selection_policy"),
        observed_room_only=review.get("review_scope") == "observed_room_only")
    selected, added = add_distinct_views(frames=rows, features=features, selected=cover,
        maximum_frames=30, policy=review.get("selection_policy"))
    ids = [rows[i]["frame_id"] for i in selected]
    binding = {"input_digest": inputs["digest"], "review_digest": canonical_digest(review),
               "candidate_frames": [{k: row[k] for k in ("frame_id", "timestamp_seconds", "image_digest")}
                                    for row in rows], "target_frame_range": [16, 30]}
    receipt = {"schema_version": "atlas_splat_source_selection.v1", "binding": binding,
               "selected_frame_ids": ids, "selected_real_frame_count": len(ids),
               "minimum_recommended_views_met": len(ids) >= 16, "diagnostics": diagnostics + added,
               "coverage_limitations": ([] if len(ids) >= 16 else
                   ["fewer_than_16_qualified_distinct_overlapping_real_views"]),
               "generated_views_count_as_real": False, "heldout_pixels_included": False,
               "claim_ceiling": "development_only"}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    path = output_root / "splat_source_selection.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("atlas_splat_source_selection_changed")
    write_json(path, receipt)
    return ids, receipt
