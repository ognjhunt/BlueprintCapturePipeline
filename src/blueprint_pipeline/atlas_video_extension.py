"""Fill candidate timeline gaps without changing or reading the frozen holdout.

Additional frames are explicitly development-only training observations. The
original frozen reconstruction dataset and its hidden pixels stay untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _extract_frames, _probe_video, _sample_indexes, _sha256_file, _tool_identity


def gap_indexes(*, times, base_indexes, visible_indexes, intervals, step_seconds=None, maximum_extra=None,
                gap_threshold_seconds=None):
    """Densify only holes inside the reviewed state; never promote a holdout."""
    hidden = set(base_indexes) - set(visible_indexes)
    base_steps = [times[b] - times[a] for a, b in zip(base_indexes, base_indexes[1:])]
    cadence = float(np.median(base_steps))
    source_cadence = float(np.median(np.diff(times)))
    step_seconds = max(source_cadence, cadence / 4) if step_seconds is None else step_seconds
    gap_threshold_seconds = cadence * 1.5 if gap_threshold_seconds is None else gap_threshold_seconds
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (step_seconds, gap_threshold_seconds))
            or (maximum_extra is not None and (type(maximum_extra) is not int or maximum_extra < 1))):
        raise ValueError("atlas_candidate_gap_policy_invalid")
    candidates = [i for i in visible_indexes if any(a <= times[i] <= b for a, b in intervals)]
    selected = []
    for left, right in zip(sorted(candidates), sorted(candidates)[1:]):
        if times[right] - times[left] < gap_threshold_seconds or not any(a <= times[left] < times[right] <= b for a, b in intervals):
            continue
        last = times[left]
        for index in range(left + 1, right):
            if index in hidden or index in visible_indexes or times[index] - last < step_seconds:
                continue
            selected.append(index)
            last = times[index]
            if maximum_extra is not None and len(selected) == maximum_extra:
                return selected
    return selected


def extend_survey(*, source_video: Path, base_survey: Path, decoded_index: Path,
                  review: dict, output_root: Path, step_seconds: float | None = None,
                  gap_threshold_seconds: float | None = None) -> dict:
    import cv2
    from .atlas_video_inputs import REVIEW_SCHEMA, SURVEY_SCHEMA
    survey = json.loads(base_survey.read_text())
    index = json.loads(decoded_index.read_text())
    if (survey.get("schema_version") != SURVEY_SCHEMA or survey.get("candidate_extension")
            or survey.get("digest") != canonical_digest(survey, digest_field="digest")
            or survey.get("heldout_pixels_included") is not False
            or review.get("schema_version") != REVIEW_SCHEMA or review.get("survey_digest") != survey["digest"]
            or review.get("moving_people_excluded") is not True or review.get("same_scene_state") is not True):
        raise ValueError("atlas_extension_review_binding_invalid")
    intervals = review.get("static_intervals_seconds")
    if (not isinstance(intervals, list) or not intervals or any(
            not isinstance(row, list) or len(row) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
            or not 0 <= row[0] < row[1] for row in intervals)):
        raise ValueError("atlas_static_interval_invalid")
    if _sha256_file(source_video) != survey["source_video_digest"] or index.get("video_digest") != survey["source_video_digest"]:
        raise ValueError("atlas_extension_source_changed")
    from .website_recording_composition import load_composition, frame_provenance
    composition = load_composition(source_video, survey["source_video_digest"])
    times = index["decoded_presentation_times_seconds"]
    visible = {int(row["frame_id"].split("-")[-1]) for row in survey["frames"]}
    indexed_visible = {int(row["frame_id"].split("-")[-1]) for row in index["sampled_frames"]}
    base = _sample_indexes(times, survey["sample_count"])
    hidden = set(base) - visible
    if visible != indexed_visible or len(hidden) != survey["heldout_frame_count"]:
        raise ValueError("atlas_extension_frozen_split_changed")
    policy = review.get("selection_policy") or {}
    # A caller's explicit cadence (accuracy-first densification) wins over the review's.
    extras = gap_indexes(times=times, base_indexes=base, visible_indexes=visible, intervals=intervals,
        step_seconds=policy.get("candidate_step_seconds") if step_seconds is None else step_seconds,
        gap_threshold_seconds=policy.get("gap_threshold_seconds") if gap_threshold_seconds is None
        else gap_threshold_seconds)
    ffprobe, ffmpeg, runtime, runtime_digest = _tool_identity()
    probe = _probe_video(source_video, ffprobe)
    if probe["presentation_times_seconds"] != times:
        raise ValueError("atlas_extension_timeline_changed")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted = _extract_frames(video_path=source_video, ffmpeg=ffmpeg, indexes=extras,
        presentation_times=times, decoded_frame_metadata=probe["frames"], frame_root=root / "candidate_extension")
    sources = []
    for row in survey["frames"]:
        if not any(a <= row["timestamp_seconds"] <= b for a, b in intervals):
            continue
        source = Path(row["source_image_path"])
        if not source.resolve().is_relative_to(base_survey.parent.resolve()) or _sha256_file(source) != row["source_image_digest"]:
            raise ValueError("atlas_extension_base_frame_changed")
        target = root / "base_candidates" / (row["frame_id"] + ".png")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        sources.append((row["frame_id"], row["timestamp_seconds"], target))
    for row in extracted:
        sources.append((row["frame_id"], row["t_video_sec"], root / "candidate_extension" / (row["frame_id"] + ".png")))
    rotation = survey["display_rotation_degrees"]
    frames, tiles = [], []
    for identifier, timestamp, path in sorted(sources, key=lambda r: r[1]):
        with Image.open(path) as source:
            upright = source.convert("RGB").rotate(rotation, expand=True)
            provenance = frame_provenance(composition, timestamp)
            content = upright.crop(provenance["source_content_bounds"]) if provenance else upright
            small = content.copy()
            small.thumbnail((640, 640))
            gray = np.asarray(small.convert("L"))
            frames.append({"frame_id": identifier, "timestamp_seconds": timestamp,
                "source_image_path": str(path), "source_image_digest": _sha256_file(path),
                "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
                "clipped_fraction": round(float(np.mean((gray < 5) | (gray > 250))), 4),
                "width": upright.width, "height": upright.height, **provenance})
            tile = Image.new("RGB", (256, 170), "#181818")
            content.thumbnail((256, 144))
            tile.paste(content, (0, 0))
            ImageDraw.Draw(tile).text((4, 148), f"{timestamp:.2f}s {identifier}", fill="white")
            tiles.append(tile)
    sheet = Image.new("RGB", (1280, math.ceil(len(tiles) / 5) * 170), "#181818")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i % 5 * 256, i // 5 * 170))
    sheet.save(root / "candidate-survey.jpg", quality=90)
    extension = {"base_survey_digest": survey["digest"], "base_decoded_index_digest": _sha256_file(decoded_index),
                 "frozen_holdout_frame_ids": [f"decoded-{i:09d}" for i in sorted(hidden)],
                 "additional_candidate_indexes": extras, "additional_frames_training_only": True,
                 "base_frozen_dataset_changed": False, "runtime": runtime, "runtime_digest": runtime_digest}
    result = {**survey, "frames": frames, "sample_count": survey["sample_count"] + len(extras),
              "candidate_extension": extension, "candidate_manifest_digest": canonical_digest(extension),
              "excluded_base_interval_frame_count": len(survey["frames"]) - len(frames) + len(extras),
              "claim_ceiling": "development_only", "heldout_pixels_included": False}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(root / "atlas_video_survey.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-video", "base-survey", "decoded-index", "review", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    args["review"] = json.loads(args["review"].read_text())
    result = extend_survey(**args)
    print(json.dumps({"frames": len(result["frames"]), "digest": result["digest"],
        "additional_candidates": len(result["candidate_extension"]["additional_candidate_indexes"])}))


if __name__ == "__main__":
    main()
