"""Refuse to erase furniture the task never named.

A whole-video track is proved on one frame and then trusted on every frame. On
a drawer scene that went wrong in a way no single-frame check could have caught:
the concept resolved the cabinet exactly where it was verified, and elsewhere
the tracker walked off it onto the desk behind it, which is the same light wood.
The editor did as it was told and erased the desk, leaving a phone and its
cables floating against a blank wall. The background review caught that and
failed the run, which is the right outcome and an expensive place to learn it.

The signal is already in the track. A rigid object filmed by a moving camera
changes apparent size smoothly; an observation several times the track's own
typical area is not the same object. That screen is free, so it runs on every
view and decides nothing on its own. Only the outliers are worth a second
opinion, and that opinion is one single-frame segmentation of the same real
pixels, which cannot drift because there is nothing to drift from.

A view whose tracked mask is refuted that way is not used for reconstruction.
Nothing is substituted: one model's mask never replaces another's. The view is
simply not trusted, and a scene with too few trustworthy views refuses rather
than reconstructing a room with its desk removed.

What this does not catch: a track that has left the target on *most* of the
views, because then the median is the drifted scale and the honest view is the
outlier. That case still reaches the background review, which reads the edited
images and fails the run. This screen exists so the common case is caught
before the editor is paid, not so the review can be retired.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .meta_sam31 import run_meta_sam31
from .website_task_masks import decode_track_mask

SCHEMA_VERSION = "website_removal_view_corroboration.v1"

#: An observation this many times the track's own median area is the one worth
#: paying to adjudicate. Below it, apparent size is just the camera moving.
#: Apparent area varies by a factor of sixty across a handheld walkthrough, so
#: no area statistic separates drift from approach on its own. This one only
#: chooses who gets the second opinion; the second opinion decides.
SECOND_OPINION_AREA_FACTOR = 1.5

#: How much larger the tracked mask may be than an independent look at the same
#: frame before the view stops being trustworthy. Measured on the scene that
#: motivated this: the two bad views ran 2.3x and 2.7x, the good one 1.2x.
MAXIMUM_TRACKED_OVER_INDEPENDENT = 1.5

#: Marble's own floor. Below this there is no reconstruction to defend.
MINIMUM_CORROBORATED_VIEWS = 2


def _removal_targets(task_masks: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [target for target in task_masks["targets"]
            if target.get("task_effect") == "manipulated" and target.get("disposition") == "remove"]


def _upright(frame: Mapping[str, Any]) -> Image.Image:
    source = Path(frame["source_image_path"])
    if _sha256_file(source) != frame["source_image_digest"]:
        raise ValueError("website_removal_corroboration_source_changed")
    rotation = float(frame["display_rotation_degrees"])
    if not np.isfinite(rotation) or rotation % 90:
        raise ValueError("website_source_rotation_not_supported")
    with Image.open(source) as image:
        return image.convert("RGB").rotate(rotation, expand=True)


def _independent_share(*, target: Mapping[str, Any], frame: Mapping[str, Any], image: Image.Image,
                       task_context: Mapping[str, Any], output_root: Path) -> float | None:
    """One single-frame segmentation of this view, or None if it finds nothing."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{frame['frame_id']}.png"
    image.save(path)
    digest = _sha256_file(path)
    registry = [{"source_frame_id": frame["frame_id"], "model_frame_index": 0,
                 "width": image.width, "height": image.height,
                 "decoded_pts_seconds": float(frame.get("timestamp_seconds") or 0.0),
                 "removal_view_corroboration": True}]
    result = run_meta_sam31(
        frame_registry=registry,
        frame_artifacts=[{"source_frame_id": frame["frame_id"], "path": str(path), "sha256": digest}],
        prompts=[{"prompt_id": target["target_id"], "output_label": target["target_id"],
                  "text": target["segmentation_prompt"]}],
        output_root=output_root, admission={}, task_context=task_context)
    shares = []
    for track in result["tracks"]:
        for observation in track["observations"]:
            mask = decode_track_mask(observation)
            shares.append(float(mask.sum()) / float(mask.size))
    return max(shares) if shares else None


def corroborate_removal_views(*, frames: Sequence[Mapping[str, Any]], task_masks: Mapping[str, Any],
                              task_context: Mapping[str, Any],
                              output_root: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Return the views whose removal masks are not refuted, and the receipt saying why."""
    targets = _removal_targets(task_masks)
    rows: list[dict[str, Any]] = []
    refuted: set[str] = set()
    for target in targets:
        track = target.get("source_track") or target["track"]
        by_frame = {row["source_frame_id"]: row for row in track["observations"]}
        shares: dict[str, float] = {}
        for frame in frames:
            observation = by_frame.get(frame["frame_id"])
            if observation is None:
                continue
            mask = decode_track_mask(observation)
            shares[frame["frame_id"]] = float(mask.sum()) / float(mask.size)
        if not shares:
            continue
        median = statistics.median(shares.values())
        for frame in frames:
            share = shares.get(frame["frame_id"])
            if share is None:
                continue
            row: dict[str, Any] = {"target_id": target["target_id"], "frame_id": frame["frame_id"],
                                   "tracked_share": round(share, 6),
                                   "track_median_share": round(median, 6)}
            if median <= 0 or share <= median * SECOND_OPINION_AREA_FACTOR:
                row["verdict"] = "within_track_scale"
                rows.append(row)
                continue
            image = _upright(frame)
            independent = _independent_share(target=target, frame=frame, image=image,
                                             task_context=task_context,
                                             output_root=output_root / "second_opinion")
            row["independent_share"] = None if independent is None else round(independent, 6)
            if independent is None or independent <= 0:
                # Nothing found is not a refutation: the object can be clipped
                # by the frame edge or too small to name, and the track may
                # still be right there.
                row["verdict"] = "no_independent_instance"
            elif share > independent * MAXIMUM_TRACKED_OVER_INDEPENDENT:
                row["verdict"] = "refuted"
                row["ratio"] = round(share / independent, 3)
                refuted.add(frame["frame_id"])
            else:
                row["verdict"] = "corroborated"
                row["ratio"] = round(share / independent, 3)
            rows.append(row)
    kept = [frame for frame in frames if frame["frame_id"] not in refuted]
    receipt = {"schema_version": SCHEMA_VERSION,
               "status": "passed" if len(kept) >= MINIMUM_CORROBORATED_VIEWS else "blocked",
               "second_opinion_area_factor": SECOND_OPINION_AREA_FACTOR,
               "maximum_tracked_over_independent": MAXIMUM_TRACKED_OVER_INDEPENDENT,
               "minimum_corroborated_views": MINIMUM_CORROBORATED_VIEWS,
               "reviewed_view_count": len(frames), "retained_view_count": len(kept),
               "refuted_frame_ids": sorted(refuted), "observations": rows,
               "basis": "single_frame_segmentation_of_the_same_pixels",
               "claim_ceiling": "development_only"}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "removal_view_corroboration.json", receipt)
    if receipt["status"] != "passed":
        raise ValueError("website_removal_views_uncorroborated")
    return kept, receipt


__all__ = ["MAXIMUM_TRACKED_OVER_INDEPENDENT", "MINIMUM_CORROBORATED_VIEWS", "SCHEMA_VERSION",
           "SECOND_OPINION_AREA_FACTOR", "corroborate_removal_views"]
