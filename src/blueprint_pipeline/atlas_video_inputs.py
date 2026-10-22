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
# The exact pose-input document, beside the posed run that binds its digest.
GEOMETRY_INPUTS_FILE = "atlas_geometry_inputs.json"
#: A view that moved less than this (896-px preview pixels) from an overlapping
#: kept neighbour adds no new viewpoint: the ``select_views`` near-duplicate rule.
DISTINCT_VIEW_MINIMUM_DISPLACEMENT = 12


def initial_survey_count(times, sampling_policy):
    interval = sampling_policy.get("initial_candidate_interval_seconds", .5)
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("atlas_survey_sampling_policy_invalid")
    return min(len(times), max(2, math.ceil((times[-1] - times[0]) / interval) + 1))


def survey_video(*, source_video: Path, output_root: Path, capture_id: str,
                 sample_count: int | None = None, sampling_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Retain candidate-only native images, PTS, quality diagnostics and a sheet."""
    if sample_count is not None and (type(sample_count) is not int or sample_count < 2):
        raise ValueError("atlas_survey_sample_count_invalid")
    source_video, output_root = source_video.resolve(), output_root.resolve()
    sampling_policy = dict(sampling_policy or {})
    if sample_count is None:
        from .local_reconstruction_adapters import _probe_video, _tool_identity
        ffprobe = _tool_identity()[0]
        probe = _probe_video(source_video, ffprobe)
        times = probe["presentation_times_seconds"]
        sample_count = initial_survey_count(times, sampling_policy)
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
              "sampling_policy": sampling_policy, "initial_sampling_is_adaptively_refinable": True,
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
    if type(maximum_frames) is not int or maximum_frames < 2:
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


def select_coverage_views(*, frames, features, maximum_frames, requirements, anchor_id, policy=None):
    """Connect reviewed task and environment anchors before admitting a subset.

    Labels come from source-image review. Feature matches establish only a
    candidate overlap path, never complete or metrically correct geometry.
    """
    from collections import deque
    if (not isinstance(requirements, list) or not requirements
            or not all(isinstance(r, Mapping) for r in requirements)
            or not {"task_object", "surrounding_scene"}.issubset({r.get("role") for r in requirements})):
        raise ValueError("atlas_task_and_scene_coverage_review_required")
    if type(maximum_frames) is not int or maximum_frames < 2:
        raise ValueError("atlas_selection_limit_invalid")
    identifiers = {r["frame_id"]: i for i, r in enumerate(frames)}
    required = []
    for requirement in requirements:
        ids = requirement.get("required_frame_ids")
        if (not requirement.get("label") or not isinstance(ids, list) or not ids
                or any(identifier not in identifiers for identifier in ids)):
            raise ValueError("atlas_coverage_anchor_outside_static_candidate_split")
        required.extend(identifiers[identifier] for identifier in ids)
    if anchor_id not in identifiers:
        raise ValueError("atlas_coverage_task_anchor_missing")
    policy = dict(policy or {})
    minimum_sharpness = policy.get("minimum_sharpness", 25)
    maximum_clipping = policy.get("maximum_clipped_fraction", .45)
    minimum_support = policy.get("minimum_overlap_inliers", 24)
    if (type(minimum_sharpness) not in (int, float) or not math.isfinite(minimum_sharpness) or minimum_sharpness < 0
            or type(maximum_clipping) not in (int, float) or not math.isfinite(maximum_clipping) or not 0 <= maximum_clipping <= 1
            or type(minimum_support) is not int or minimum_support < 8):
        raise ValueError("atlas_scene_selection_policy_invalid")
    eligible = {i for i, r in enumerate(frames) if r["sharpness"] >= minimum_sharpness and r["clipped_fraction"] <= maximum_clipping}
    anchor = identifiers[anchor_id]
    if not set([anchor, *required]).issubset(eligible):
        raise ValueError("atlas_coverage_anchor_quality_insufficient")
    selected, cache, diagnostics = {anchor}, {}, []
    for target in dict.fromkeys(required):
        if target in selected:
            continue
        queue = deque(sorted(selected))
        parent = {i: None for i in selected}
        while queue and target not in parent:
            source = queue.popleft()
            for other in sorted(eligible - parent.keys(), key=lambda i: (abs(i - source), i)):
                pair = tuple(sorted((source, other)))
                if pair not in cache:
                    cache[pair] = _overlap(features[pair[0]], features[pair[1]])
                if cache[pair][0] < minimum_support:
                    continue
                parent[other] = source
                queue.append(other)
                if other == target:
                    break
        if target not in parent:
            raise ValueError("atlas_task_and_scene_coverage_gap:" + frames[target]["frame_id"])
        cursor = target
        while cursor not in selected:
            selected.add(cursor)
            previous = parent[cursor]
            pair = tuple(sorted((cursor, previous)))
            support, displacement = cache[pair]
            diagnostics.append({"from_frame_id": frames[previous]["frame_id"], "frame_id": frames[cursor]["frame_id"],
                                "inlier_matches": support, "median_displacement_preview_pixels": round(displacement, 3),
                                "decision": "required_coverage_or_overlap_bridge"})
            cursor = previous
        if len(selected) > maximum_frames:
            raise ValueError("atlas_coverage_exceeds_frame_budget")
    if len(selected) < 2:
        raise ValueError("atlas_insufficient_overlapping_distinct_views")
    return sorted(selected), diagnostics


def add_distinct_views(*, frames, features, selected, maximum_frames, policy=None):
    """Grow the connected cover with distinct, sharp, overlapping views, widest spread first.

    Owner direction 2026-10-22: accuracy, not cost, decides the provider inputs.
    Each step admits the candidate farthest in time from the kept set that still
    overlaps a kept time neighbour under the cover's support rule and moved enough
    from every overlapping one to be a new viewpoint. Every admitted view therefore
    overlaps a kept view, earlier context is reached through its own bridges, and
    a binding ``maximum_frames`` keeps the widest connected spread. Overlap is a
    selection heuristic, never geometry.
    """
    policy = dict(policy or {})
    minimum_sharpness = policy.get("minimum_sharpness", 25)
    maximum_clipping = policy.get("maximum_clipped_fraction", .45)
    minimum_support = policy.get("minimum_overlap_inliers", 24)
    cover = set(selected)
    if type(maximum_frames) is not int or maximum_frames < len(cover):
        raise ValueError("atlas_coverage_exceeds_frame_budget")
    times = [row["timestamp_seconds"] for row in frames]
    kept, cache, decisions = set(cover), {}, {}
    pending = {i for i, row in enumerate(frames) if i not in kept and row["sharpness"] >= minimum_sharpness
               and row["clipped_fraction"] <= maximum_clipping}

    def overlap(a, b):
        pair = (min(a, b), max(a, b))
        if pair not in cache:
            cache[pair] = _overlap(features[pair[0]], features[pair[1]])
        return cache[pair]
    while pending and len(kept) < maximum_frames:
        best, best_gap = None, -1.0
        for i in sorted(pending, key=lambda index: (times[index], index)):
            before = [k for k in kept if times[k] <= times[i]]
            after = [k for k in kept if times[k] >= times[i]]
            neighbours = [max(before, key=lambda k: times[k])] if before else []
            neighbours += [min(after, key=lambda k: times[k])] if after else []
            connected = [(support, moved) for support, moved in (overlap(i, k) for k in neighbours)
                         if support >= minimum_support]
            if not connected:
                continue  # Not reachable yet; a nearer admitted view may bridge it.
            if min(moved for _, moved in connected) < DISTINCT_VIEW_MINIMUM_DISPLACEMENT:
                decisions[i] = "near_duplicate"
                continue
            gap = min(abs(times[i] - times[k]) for k in neighbours)
            if gap > best_gap:
                best, best_gap = i, gap
        pending -= set(decisions)
        if best is None:
            break
        kept.add(best)
        pending.discard(best)
        decisions[best] = "distinct_view"
    leftover = "beyond_input_capacity" if len(kept) >= maximum_frames else "disconnected_view"
    decisions.update({i: leftover for i in pending})
    return sorted(kept), [{"frame_id": frames[i]["frame_id"], "decision": decision}
                          for i, decision in sorted(decisions.items(), key=lambda item: times[item[0]])]


def takes_reviewed_background(review: Mapping[str, Any]) -> bool:
    """Whether Atlas may receive only reviewed, edited views (the editor-first path).

    A removed object needs the edit, and so does a transient (a person, a carried
    item) in a view the editor processes: the required and additional views. A
    transient seen only in other frames needs no edit; the direct path leaves
    those frames out of its inputs instead.
    """
    automation = review.get("automation") or {}
    if any(row.get("disposition") == "remove" for row in automation.get("objects", [])):
        return True
    edited = {key for group in review.get("coverage_requirements", []) for key in group["required_frame_ids"]}
    edited |= set(review.get("additional_input_frame_ids", []))
    return bool(edited & set(review.get("transient_frame_ids", [])))


def selection_frame_budget(review: Mapping[str, Any]) -> int:
    """A planning ceiling from reviewed dollars, never provider authority."""
    from datetime import datetime, timezone
    from decimal import Decimal
    from .website_atlas_pose import _positive_finite, _reviewed_recently
    budget = review.get("selection_budget")
    if (not isinstance(budget, Mapping) or not _positive_finite(budget.get("maximum_pose_cost_usd"))
            or not _positive_finite(budget.get("input_cost_usd")) or not budget.get("rate_card_reference")
            or not budget.get("reviewed_by")
            or not _reviewed_recently(budget.get("reviewed_on"), datetime.now(timezone.utc).date())):
        raise ValueError("atlas_reviewed_selection_budget_required")
    count = int(Decimal(str(budget["maximum_pose_cost_usd"])) / Decimal(str(budget["input_cost_usd"])))
    if count < 2:
        raise ValueError("atlas_selection_budget_insufficient")
    return count


def prepare_video_inputs(*, survey_path: Path, review: Mapping[str, Any], output_root: Path,
                         maximum_frames: int | None = None) -> dict[str, Any]:
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
    from .website_atlas_pose import MAX_POSE_INPUT_IMAGES
    budget_frames = selection_frame_budget(review)
    if maximum_frames is not None and (type(maximum_frames) is not int or maximum_frames < 2):
        raise ValueError("atlas_selection_limit_invalid")
    # The connected cover is bounded only by the reviewed dollars (or an operator's
    # explicit lower limit); added views by the provider's configured input capacity.
    capacity = MAX_POSE_INPUT_IMAGES if maximum_frames is None else maximum_frames
    cover_limit = budget_frames if maximum_frames is None else min(maximum_frames, budget_frames)
    # Frames the review judged unusable inside a still interval (someone filling
    # the view, another door state) never become candidates. The direct path
    # poses every input as captured, so no frame showing a transient joins it.
    excluded = set(review.get("excluded_frame_ids", []))
    if not takes_reviewed_background(review):
        excluded |= set(review.get("transient_frame_ids", []))
    frames = [r for r in survey["frames"] if any(a <= r["timestamp_seconds"] <= b for a, b in intervals)
              and r["frame_id"] not in excluded]
    for r in frames:
        path = Path(r["source_image_path"])
        if not path.resolve().is_relative_to(survey_path.parent.resolve()) or _sha256_file(path) != r["source_image_digest"]:
            raise ValueError("atlas_source_frame_changed")
    rotation = survey["display_rotation_degrees"]
    anchor_id = review.get("anchor_frame_id")
    if anchor_id is not None and anchor_id not in {r["frame_id"] for r in frames}:
        raise ValueError("atlas_anchor_outside_static_candidate_split")
    requirements = review.get("coverage_requirements")
    if review.get("additional_input_frame_ids"):
        requirements = [*requirements, {"role": "support_surface", "label": "Reviewed reconstruction gap inputs",
            "required_frame_ids": review["additional_input_frame_ids"]}]
    features = [_features(r, rotation) for r in frames]
    try:
        cover, diagnostics = select_coverage_views(frames=frames, features=features,
            maximum_frames=cover_limit, requirements=requirements, anchor_id=anchor_id,
            policy=review.get("selection_policy"))
    except ValueError as error:
        write_json(output_root / "atlas_coverage_gap.json", {"status": "blocked_before_upload", "reason": str(error),
            "survey_digest": survey["digest"], "review_digest": canonical_digest(review),
            "task_and_surrounding_scene_covered": False, "paid_task_started": False})
        raise
    maximum_frames = max(capacity, len(cover))
    selected, added = add_distinct_views(frames=frames, features=features, selected=cover,
                                         maximum_frames=maximum_frames, policy=review.get("selection_policy"))
    diagnostics = [*diagnostics, *added]
    if len(selected) > budget_frames:
        # Dollars only ever refuse the accurate set; they never pose fewer frames.
        raise ValueError("atlas_selection_budget_below_accurate_inputs")
    from .atlas_privacy_screen import blur_faces, screen_identity
    root = output_root.resolve()
    prepared, blurred = [], {}
    for i in selected:
        row = frames[i]
        target = root / "atlas_source_views" / (row["frame_id"] + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        retained_source = root / "atlas_observations" / (row["frame_id"] + ".png")
        retained_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source_image_path"], retained_source)
        with Image.open(row["source_image_path"]) as image:
            # Native resolution, lossless upright RGB; never the 518px model input.
            # Faces are blurred before the digest that the provider upload is bound to.
            screened, faces = blur_faces(image.convert("RGB").rotate(rotation, expand=True))
        screened.save(target)
        blurred[row["frame_id"]] = len(faces)
        prepared.append({**row, "source_image_path": str(retained_source),
                         "image_path": str(target), "image_digest": _sha256_file(target)})
    inputs = {"schema_version": INPUT_SCHEMA,
              "binding": {"source_video_digest": survey["source_video_digest"], "maximum_frames": maximum_frames,
                          "adapter_version": 3, "survey_digest": survey["digest"],
                          "scene_state_review_digest": canonical_digest(review),
                          "privacy_screen": {**screen_identity(), "blurred_faces_by_frame": blurred}},
              "frames": prepared, "source_sample_count": survey["sample_count"],
              "heldout_frame_count": survey["heldout_frame_count"], "heldout_pixels_included": False}
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    # Scene preparation reads the posed views as source geometry, so keep the
    # exact document the pose binds to, not only its digest.
    retained = root / GEOMETRY_INPUTS_FILE
    if retained.is_file() and json.loads(retained.read_text()) != inputs:
        raise ValueError("atlas_geometry_inputs_identity_conflict")
    write_json(retained, inputs)
    plan = {"schema_version": "atlas_view_selection.v1", "review": dict(review),
            "input_digest": inputs["digest"], "selected_frame_count": len(prepared),
            "maximum_frames": maximum_frames, "diagnostics": diagnostics,
            "frame_policy": {"basis": "accuracy_first", "provider_input_capacity": capacity,
                             "frame_target": maximum_frames,
                             "connected_cover_frames": len(cover), "distinct_views_added": len(selected) - len(cover)},
            "task_and_surrounding_scene_covered": True,
            "coverage_requirements": requirements, "coverage_authority": "source_image_review_and_overlap_heuristic",
            "privacy_screen": inputs["binding"]["privacy_screen"],
            "frames": [{k: r[k] for k in ("frame_id", "timestamp_seconds", "image_digest")} for r in prepared],
            "unseen_regions_recovered": False, "geometry_or_static_state_proven": False,
            "excluded_interval_frames": len(survey["frames"]) - len(frames) + survey.get("excluded_base_interval_frame_count", 0)}
    write_json(root / "atlas_view_selection.json", plan)
    return prepare_atlas_pose_inputs(inputs=inputs, input_root=root, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    survey = sub.add_parser("survey")
    survey.add_argument("--source-video", type=Path, required=True)
    survey.add_argument("--capture-id", required=True)
    survey.add_argument("--sample-count", type=int, help="Optional initial survey count; otherwise video duration drives sampling")
    survey.add_argument("--sampling-policy", type=Path, help="Per-video initial candidate cadence; gaps can be refined later")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--survey-path", type=Path, required=True)
    prepare.add_argument("--review", type=Path, required=True)
    prepare.add_argument("--maximum-frames", type=int, help="Optional operator limit below the reviewed dollar budget")
    for p in (survey, prepare):
        p.add_argument("--output-root", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["review"] = json.loads(args["review"].read_text())
    elif args.get("sampling_policy"):
        args["sampling_policy"] = json.loads(args["sampling_policy"].read_text())
    result = survey_video(**args) if command == "survey" else prepare_video_inputs(**args)
    print(json.dumps({k: result.get(k) for k in ("schema_version", "digest", "planned_provider_input_count")}))


if __name__ == "__main__":
    main()
