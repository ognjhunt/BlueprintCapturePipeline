"""Atlas video survey and adaptive source-view selection (ADP-040, Day 28).

Only the existing decoder's candidate split is inspected. Scene-state intervals
come from the harness's review of that survey, not a claim that optical flow can
recognize an articulated state. No pixels are uploaded by this module.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import LocalDecodedObservationAdapter, _sha256_file

SURVEY_SCHEMA = "atlas_video_survey.v1"
REVIEW_SCHEMA = "atlas_scene_state_review.v1"


def survey_video(*, source_video: Path, output_root: Path, capture_id: str,
                 sample_count: int = 64) -> dict[str, Any]:
    """Retain candidate-only native images, PTS, quality diagnostics and a sheet."""
    if type(sample_count) is not int or not 4 <= sample_count <= 240:
        raise ValueError("atlas_survey_sample_count_invalid")
    source_video, output_root = source_video.resolve(), output_root.resolve()
    decoded = LocalDecodedObservationAdapter().execute(
        intake_id=capture_id, capture_digest=_sha256_file(source_video),
        capture_authority_profile="monocular_video", capture_root=source_video.parent,
        video_relative_path=source_video.name, output_root=output_root / "source",
        rights_and_retention={"local_processing_authorized": True,
                              "provider_upload_authorized": False, "paid_compute_authorized": False},
        maximum_frames=sample_count)
    refs = decoded["asset_references"]
    source_root = output_root / "source"
    candidate_path = source_root / refs["candidate_dataset_manifest"]["relative_path"]
    candidate = json.loads(candidate_path.read_text())
    index = json.loads((source_root / refs["decoded_observation_index"]["relative_path"]).read_text())
    rotation = float(index["stream_metadata"].get("display_rotation_degrees") or 0)
    if not math.isfinite(rotation) or rotation % 90:
        raise ValueError("atlas_source_rotation_unsupported")
    import cv2
    frames = []
    tiles = []
    for row in sorted(candidate["frames"], key=lambda r: r["t_video_sec"]):
        source = (candidate_path.parent / row["candidate_relative_path"]).resolve()
        if not source.is_relative_to(candidate_path.parent) or _sha256_file(source) != row["frame_digest"]:
            raise ValueError("atlas_source_frame_changed")
        with Image.open(source) as original:
            upright = original.convert("RGB").rotate(rotation, expand=True)
            small = upright.copy()
            small.thumbnail((640, 640))
            gray = np.asarray(small.convert("L"))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            clipped = float(np.mean((gray < 5) | (gray > 250)))
            frames.append({"frame_id": row["frame_id"], "timestamp_seconds": row["t_video_sec"],
                           "source_image_path": str(source), "source_image_digest": row["frame_digest"],
                           "sharpness": round(sharpness, 4), "clipped_fraction": round(clipped, 4),
                           "width": upright.width, "height": upright.height})
            tile = Image.new("RGB", (256, 170), "#181818")
            upright.thumbnail((256, 144))
            tile.paste(upright, (0, 0))
            ImageDraw.Draw(tile).text((4, 148), f'{row["t_video_sec"]:.2f}s  {row["frame_id"]}', fill="white")
            tiles.append(tile)
    sheet = Image.new("RGB", (5 * 256, math.ceil(len(tiles) / 5) * 170), "#181818")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i % 5 * 256, i // 5 * 170))
    output_root.mkdir(parents=True, exist_ok=True)
    sheet.save(output_root / "candidate-survey.jpg", quality=90)
    result = {"schema_version": SURVEY_SCHEMA, "capture_id": capture_id,
              "source_video_digest": _sha256_file(source_video), "sample_count": sample_count,
              "display_rotation_degrees": rotation, "frames": frames,
              "heldout_frame_count": index["hidden_heldout_frame_count"],
              "heldout_pixels_included": False, "claim_ceiling": "development_only",
              "candidate_manifest_digest": _sha256_file(candidate_path),
              "review_required": "same static scene state, moving people, crop and task coverage"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output_root / "atlas_video_survey.json", result)
    return result


def _features(row: Mapping[str, Any], rotation: float):
    import cv2
    with Image.open(row["source_image_path"]) as image:
        image = image.convert("L").rotate(rotation, expand=True)
        image.thumbnail((896, 896))
        return cv2.SIFT_create(nfeatures=4000).detectAndCompute(np.asarray(image), None)


def _overlap(a, b) -> tuple[int, float]:
    """Feature support is a selection heuristic, never camera/geometry proof."""
    import cv2
    ak, ad = a
    bk, bd = b
    if ad is None or bd is None or min(len(ad), len(bd)) < 8:
        return 0, 0.0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ad, bd, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0, 0.0
    source = np.float32([ak[m.queryIdx].pt for m in good])
    target = np.float32([bk[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    _, epipolar = cv2.findFundamentalMat(source, target, cv2.FM_RANSAC, 2.0, .99)
    if epipolar is not None and (mask is None or epipolar.sum() > mask.sum()):
        mask = epipolar
    if mask is None:
        return 0, 0.0
    keep = mask.ravel().astype(bool)
    return int(keep.sum()), float(np.median(np.linalg.norm(target[keep] - source[keep], axis=1)))


def select_views(*, frames: Sequence[Mapping[str, Any]], features: Sequence[Any],
                 maximum_frames: int, anchor_index: int = 0) -> tuple[list[int], list[dict[str, Any]]]:
    """Keep a chronological overlap chain; stop paying for near-duplicate views."""
    if type(maximum_frames) is not int or not 2 <= maximum_frames <= 64:
        raise ValueError("atlas_selection_limit_invalid")
    eligible = [i for i, r in enumerate(frames) if r["sharpness"] >= 25 and r["clipped_fraction"] <= .45]
    if len(eligible) < 2:
        raise ValueError("atlas_insufficient_sharp_static_views")
    anchor = min(eligible, key=lambda i: abs(i - anchor_index))
    selected, diagnostics = [anchor], []
    order = list(reversed([i for i in eligible if i < anchor])) + [i for i in eligible if i > anchor]
    for i in order:
        reference = min(selected) if i < anchor else max(selected)
        support, displacement = _overlap(features[reference], features[i])
        decision = "selected"
        if support < 24:
            decision = "disconnected_view"
        elif displacement < 12:
            decision = "near_duplicate"
        elif len(selected) >= maximum_frames:
            decision = "frame_budget_reached"
        else:
            selected.append(i)
        diagnostics.append({"frame_id": frames[i]["frame_id"], "inlier_matches": support,
                            "median_displacement_preview_pixels": round(displacement, 3), "decision": decision})
    if len(selected) < 2:
        raise ValueError("atlas_insufficient_overlapping_distinct_views")
    return sorted(selected), diagnostics


def prepare_video_inputs(*, survey_path: Path, review: Mapping[str, Any], output_root: Path,
                         maximum_frames: int = 24) -> dict[str, Any]:
    """Build the existing pose producer's pending input from a reviewed survey."""
    from .website_worldlabs import prepare_atlas_pose_inputs
    from .website_scene_geometry import INPUT_SCHEMA
    survey = json.loads(survey_path.read_text())
    if (survey.get("schema_version") != SURVEY_SCHEMA or survey.get("heldout_pixels_included") is not False
            or survey.get("digest") != canonical_digest(survey, digest_field="digest")):
        raise ValueError("atlas_survey_invalid")
    previous_path = output_root / "atlas_view_selection.json"
    if previous_path.is_file() and json.loads(previous_path.read_text()).get("review") != dict(review):
        raise ValueError("atlas_selection_review_conflict")
    intervals = review.get("static_intervals_seconds")
    if (review.get("schema_version") != REVIEW_SCHEMA or review.get("survey_digest") != survey["digest"]
            or review.get("moving_people_excluded") is not True
            or review.get("same_scene_state") is not True or not review.get("reviewed_by")
            or not review.get("scene_state") or not isinstance(intervals, list) or not intervals):
        raise ValueError("atlas_scene_state_review_required")
    for interval in intervals:
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in interval)
                or not 0 <= interval[0] < interval[1]):
            raise ValueError("atlas_static_interval_invalid")
    frames = [r for r in survey["frames"] if any(a <= r["timestamp_seconds"] <= b for a, b in intervals)]
    for r in frames:
        path = Path(r["source_image_path"])
        if not path.resolve().is_relative_to(survey_path.parent.resolve()) or _sha256_file(path) != r["source_image_digest"]:
            raise ValueError("atlas_source_frame_changed")
    rotation = survey["display_rotation_degrees"]
    anchor_id = review.get("anchor_frame_id")
    if anchor_id is not None and anchor_id not in {r["frame_id"] for r in frames}:
        raise ValueError("atlas_anchor_outside_static_candidate_split")
    anchor = next((i for i, row in enumerate(frames) if row["frame_id"] == anchor_id), 0)
    selected, diagnostics = select_views(frames=frames, features=[_features(r, rotation) for r in frames],
                                         maximum_frames=maximum_frames, anchor_index=anchor)
    root = output_root.resolve()
    prepared = []
    for i in selected:
        row = frames[i]
        target = root / "atlas_source_views" / (row["frame_id"] + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        retained_source = root / "atlas_observations" / (row["frame_id"] + ".png")
        retained_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source_image_path"], retained_source)
        with Image.open(row["source_image_path"]) as image:
            # Native resolution, lossless upright RGB; never the 518px model input.
            image.convert("RGB").rotate(rotation, expand=True).save(target)
        prepared.append({**row, "source_image_path": str(retained_source),
                         "image_path": str(target), "image_digest": _sha256_file(target)})
    inputs = {"schema_version": INPUT_SCHEMA,
              "binding": {"source_video_digest": survey["source_video_digest"], "maximum_frames": maximum_frames,
                          "adapter_version": 3, "survey_digest": survey["digest"],
                          "scene_state_review_digest": canonical_digest(review)},
              "frames": prepared, "source_sample_count": survey["sample_count"],
              "heldout_frame_count": survey["heldout_frame_count"], "heldout_pixels_included": False}
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    plan = {"schema_version": "atlas_view_selection.v1", "review": dict(review),
            "input_digest": inputs["digest"], "selected_frame_count": len(prepared),
            "maximum_frames": maximum_frames, "diagnostics": diagnostics,
            "frames": [{k: r[k] for k in ("frame_id", "timestamp_seconds", "image_digest")} for r in prepared],
            "unseen_regions_recovered": False, "geometry_or_static_state_proven": False,
            "excluded_interval_frames": len(survey["frames"]) - len(frames)}
    write_json(root / "atlas_view_selection.json", plan)
    return prepare_atlas_pose_inputs(inputs=inputs, input_root=root, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    survey = sub.add_parser("survey")
    survey.add_argument("--source-video", type=Path, required=True)
    survey.add_argument("--capture-id", required=True)
    survey.add_argument("--sample-count", type=int, default=64)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--survey-path", type=Path, required=True)
    prepare.add_argument("--review", type=Path, required=True)
    prepare.add_argument("--maximum-frames", type=int, default=24)
    for p in (survey, prepare):
        p.add_argument("--output-root", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["review"] = json.loads(args["review"].read_text())
    result = survey_video(**args) if command == "survey" else prepare_video_inputs(**args)
    print(json.dumps({k: result.get(k) for k in ("schema_version", "digest", "planned_provider_input_count")}))


if __name__ == "__main__":
    main()
