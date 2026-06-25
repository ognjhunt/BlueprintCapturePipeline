"""Generated WAM video validation helpers.

These checks distinguish a generated video file that can be decoded for review
from a mere path or placeholder byte blob. Higher-level visual quality checks
still decide whether the decoded rollout is useful for task-success judging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "wam_generated_video_review_validation.v1"
VISUAL_SMOKE_SCHEMA_VERSION = "wam_generated_rollout_visual_smoke.v1"
SOURCE_POLICY_OBSERVATION_VISUAL_QA_SCHEMA_VERSION = "source_policy_observation_visual_qa.v1"
PERSISTENT_WAM_VISUAL_QUALITY_SCHEMA_VERSION = "persistent_policy_wam_visual_quality_report.v1"
PERSISTENT_WAM_FRAME_STATS_SCHEMA_VERSION = "persistent_policy_wam_frame_stats.v1"

REVIEW_QUALITY_MIN_WIDTH = 320
REVIEW_QUALITY_MIN_HEIGHT = 256
REVIEW_QUALITY_MIN_FPS = 8.0
REVIEW_QUALITY_MIN_NUM_FRAMES = 12


def _blocked(path: Path, blockers: list[str], **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "blockers": blockers,
        **fields,
    }


def validate_generated_mp4_for_review(
    path: str | Path,
    *,
    sample_frame_count: int = 6,
) -> dict[str, Any]:
    """Return decode-level review validation for a generated MP4 path."""
    video_path = Path(path).expanduser()
    if not video_path.is_file():
        return _blocked(video_path, ["generated_video_missing"])
    size_bytes = video_path.stat().st_size
    if size_bytes <= 0:
        return _blocked(video_path, ["generated_video_empty"])
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent.
        return _blocked(video_path, [f"opencv_import_failed:{type(exc).__name__}"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return _blocked(video_path, ["generated_video_unreadable"])
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        blockers: list[str] = []
        if frame_count <= 0:
            blockers.append("generated_video_frame_count_unavailable")
        if width <= 0 or height <= 0:
            blockers.append("generated_video_dimensions_unavailable")

        sample_indices: list[int] = []
        if frame_count > 0:
            last = max(0, frame_count - 1)
            wanted = max(1, sample_frame_count)
            sample_indices = sorted(
                {round(index * last / max(wanted - 1, 1)) for index in range(wanted)}
            )
        readable_samples = 0
        sampled_frames: list[dict[str, Any]] = []
        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            readable_samples += 1
            sampled_frames.append(
                {
                    "frame_index": int(frame_index),
                    "height": int(frame.shape[0]),
                    "width": int(frame.shape[1]),
                }
            )
        if frame_count > 0 and readable_samples <= 0:
            blockers.append("generated_video_sample_frames_unreadable")
        status = "completed" if not blockers else "blocked"
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "path": str(video_path),
            "exists": True,
            "size_bytes": size_bytes,
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "width": width,
            "height": height,
            "readable_sampled_frame_count": readable_samples,
            "sampled_frames": sampled_frames,
            "blockers": blockers,
        }
    finally:
        capture.release()


def _safe_component(value: Any, *, fallback: str = "rollout") -> str:
    text = str(value or fallback).strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _rollout_video_path(rollout: Mapping[str, Any]) -> Path | None:
    for key in (
        "generated_video_path",
        "video_path",
        "output_video_path",
        "path",
    ):
        value = rollout.get(key)
        if value:
            return Path(str(value)).expanduser()
    return None


def _round_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _frame_visual_stats(
    path: str | Path,
    *,
    role: str,
    frame_index: int | None = None,
    source_frame_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frame_path = Path(path).expanduser()
    result: dict[str, Any] = {
        "schema_version": PERSISTENT_WAM_FRAME_STATS_SCHEMA_VERSION,
        "role": role,
        "frame_index": frame_index,
        "path": str(frame_path),
        "status": "blocked",
        "blockers": [],
    }
    if not frame_path.is_file():
        result["blockers"] = ["frame_missing_for_visual_quality"]
        return result
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        result["blockers"] = [f"visual_quality_dependency_import_failed:{type(exc).__name__}"]
        return result
    try:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb).astype("float32")
    except Exception as exc:
        result["blockers"] = [f"frame_unreadable_for_visual_quality:{type(exc).__name__}"]
        return result

    luma = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    height, width = luma.shape
    histogram, _ = np.histogram(luma, bins=256, range=(0, 255))
    total = int(histogram.sum())
    probabilities = histogram[histogram > 0] / max(total, 1)
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
    gradient_y, gradient_x = np.gradient(luma)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    edge_density = float((gradient_magnitude > 18.0).mean())
    if height >= 3 and width >= 3:
        laplacian = (
            -4.0 * luma[1:-1, 1:-1]
            + luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
        )
        sharpness_laplacian_variance = float(laplacian.var())
    else:
        sharpness_laplacian_variance = 0.0
    center_y0 = max(0, int(height * 0.25))
    center_y1 = min(height, max(center_y0 + 1, int(height * 0.75)))
    center_x0 = max(0, int(width * 0.25))
    center_x1 = min(width, max(center_x0 + 1, int(width * 0.75)))
    center = luma[center_y0:center_y1, center_x0:center_x1]
    center_gradient = gradient_magnitude[center_y0:center_y1, center_x0:center_x1]
    center_histogram, _ = np.histogram(center, bins=128, range=(0, 255))
    center_probabilities = center_histogram[center_histogram > 0] / max(
        int(center_histogram.sum()),
        1,
    )
    center_entropy = (
        float(-(center_probabilities * np.log2(center_probabilities)).sum())
        if center_histogram.sum()
        else 0.0
    )
    dominant_luma_bin_ratio = float(histogram.max() / max(total, 1)) if total else 0.0

    source = dict(source_frame_stats or {})
    source_edge_density = float(source.get("edge_density") or 0.0)
    source_mean_luma = float(source.get("mean_luma") or 0.0)
    source_std_luma = float(source.get("std_luma") or 0.0)
    source_entropy_bits = float(source.get("entropy_bits") or 0.0)
    drift = {
        "mean_luma_delta_from_source": _round_float(float(luma.mean()) - source_mean_luma, 3)
        if source
        else None,
        "std_luma_ratio_to_source": _round_float(float(luma.std()) / source_std_luma, 6)
        if source_std_luma > 0.0
        else None,
        "edge_density_ratio_to_source": _round_float(edge_density / source_edge_density, 6)
        if source_edge_density > 0.0
        else None,
        "entropy_delta_from_source": _round_float(entropy_bits - source_entropy_bits, 3)
        if source
        else None,
    }
    result.update(
        {
            "status": "completed",
            "width": int(width),
            "height": int(height),
            "mean_luma": _round_float(float(luma.mean()), 3),
            "std_luma": _round_float(float(luma.std()), 3),
            "luma_min": _round_float(float(luma.min()), 3),
            "luma_max": _round_float(float(luma.max()), 3),
            "luma_range": _round_float(float(luma.max() - luma.min()), 3),
            "dark_pixel_ratio": _round_float(float((luma < 32.0).mean()), 6),
            "near_black_pixel_ratio": _round_float(float((luma < 16.0).mean()), 6),
            "bright_pixel_ratio": _round_float(float((luma > 224.0).mean()), 6),
            "dominant_luma_bin_ratio": _round_float(dominant_luma_bin_ratio, 6),
            "entropy_bits": _round_float(entropy_bits, 6),
            "edge_density": _round_float(edge_density, 6),
            "sharpness_laplacian_variance": _round_float(sharpness_laplacian_variance, 3),
            "center_crop": {
                "x0": int(center_x0),
                "y0": int(center_y0),
                "x1": int(center_x1),
                "y1": int(center_y1),
                "mean_luma": _round_float(float(center.mean()), 3),
                "std_luma": _round_float(float(center.std()), 3),
                "dark_pixel_ratio": _round_float(float((center < 32.0).mean()), 6),
                "entropy_bits": _round_float(center_entropy, 6),
                "edge_density": _round_float(float((center_gradient > 18.0).mean()), 6),
            },
            "drift_from_source": drift,
            "blockers": [],
        }
    )
    return result


def _source_policy_observation_blockers(
    stats: Mapping[str, Any],
    *,
    target_object_id: str | None,
    review_quality_required: bool,
) -> list[str]:
    blockers: list[str] = []
    if stats.get("status") != "completed":
        return list(stats.get("blockers") or ["source_policy_observation_visual_probe_failed"])
    width = int(stats.get("width") or 0)
    height = int(stats.get("height") or 0)
    mean_luma = float(stats.get("mean_luma") or 0.0)
    std_luma = float(stats.get("std_luma") or 0.0)
    luma_range = float(stats.get("luma_range") or 0.0)
    dark_ratio = float(stats.get("dark_pixel_ratio") or 0.0)
    near_black_ratio = float(stats.get("near_black_pixel_ratio") or 0.0)
    entropy = float(stats.get("entropy_bits") or 0.0)
    edge_density = float(stats.get("edge_density") or 0.0)
    sharpness = float(stats.get("sharpness_laplacian_variance") or 0.0)
    center = stats.get("center_crop") if isinstance(stats.get("center_crop"), Mapping) else {}
    center_dark_ratio = float(center.get("dark_pixel_ratio") or 0.0)
    center_edge_density = float(center.get("edge_density") or 0.0)
    center_entropy = float(center.get("entropy_bits") or 0.0)
    if review_quality_required and (
        width < REVIEW_QUALITY_MIN_WIDTH or height < REVIEW_QUALITY_MIN_HEIGHT
    ):
        blockers.append("source_policy_observation_resolution_too_low_for_review_quality")
    if mean_luma < 38.0 or dark_ratio > 0.50 or near_black_ratio > 0.45:
        blockers.append("source_policy_observation_too_dark_for_review")
    if std_luma < 12.0 or luma_range < 50.0 or entropy < 2.5:
        blockers.append("source_policy_observation_flat_or_low_contrast")
    if edge_density < 0.012 or sharpness < 20.0:
        blockers.append("source_policy_observation_blurry_or_low_detail")
    if center_dark_ratio > 0.65 or center_edge_density < 0.004 or center_entropy < 1.8:
        blockers.append("source_policy_observation_task_region_low_information")
    if dark_ratio > 0.40 and edge_density < 0.015:
        blockers.append("source_policy_observation_mostly_wall_cabinet_counter_or_occlusion")
    if target_object_id and any(
        item in blockers
        for item in {
            "source_policy_observation_too_dark_for_review",
            "source_policy_observation_blurry_or_low_detail",
            "source_policy_observation_task_region_low_information",
            "source_policy_observation_mostly_wall_cabinet_counter_or_occlusion",
        }
    ):
        blockers.append("target_object_visibility_failed_visual_proxy")
    return sorted(set(blockers))


def assess_source_policy_observation_visual_qa(
    frame_path: str | Path | None,
    *,
    generated_at: str,
    target_object_id: str | None = None,
    task_id: str | None = None,
    visual_profile: str = "smoke",
    review_quality_required: bool = False,
) -> dict[str, Any]:
    """Assess the initial policy POV before WAM spend or rollout claims."""
    if frame_path is None:
        stats = {
            "status": "blocked",
            "path": None,
            "blockers": ["source_policy_observation_frame_missing"],
        }
    else:
        stats = _frame_visual_stats(frame_path, role="source_policy_observation", frame_index=0)
    blockers = _source_policy_observation_blockers(
        stats,
        target_object_id=target_object_id,
        review_quality_required=review_quality_required,
    )
    passed = bool(stats.get("status") == "completed" and not blockers)
    return {
        "schema_version": SOURCE_POLICY_OBSERVATION_VISUAL_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed_visual_quality_gate" if passed else "failed_visual_quality_gate",
        "visual_success": passed,
        "visual_profile": visual_profile,
        "review_quality_required": review_quality_required,
        "source_frame_path": str(Path(frame_path).expanduser()) if frame_path else None,
        "target_object_id": target_object_id,
        "task_id": task_id,
        "target_visibility_status": (
            "failed_visual_proxy"
            if "target_object_visibility_failed_visual_proxy" in blockers
            else "not_declared"
            if not target_object_id
            else "passed_visual_proxy"
            if passed
            else "not_proven"
        ),
        "metrics": stats,
        "blockers": blockers,
        "claim_boundary": {
            "visual_qa_is_not_task_success_proof": True,
            "target_visibility_is_heuristic_without_detector": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "raw_secret_values_recorded": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _rational_to_float(value: Any) -> float:
    text = str(value or "").strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _video_metadata_from_status(video_status: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = (
        video_status.get("ffprobe_metadata")
        if isinstance(video_status, Mapping)
        and isinstance(video_status.get("ffprobe_metadata"), Mapping)
        else {}
    )
    streams = metadata.get("streams") if isinstance(metadata.get("streams"), list) else []
    stream = streams[0] if streams and isinstance(streams[0], Mapping) else {}
    format_metadata = (
        metadata.get("format") if isinstance(metadata.get("format"), Mapping) else {}
    )
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "fps": _rational_to_float(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "nb_frames": int(stream.get("nb_frames") or 0) if str(stream.get("nb_frames") or "").isdigit() else 0,
        "duration_seconds": _round_float(
            stream.get("duration") or format_metadata.get("duration"),
            6,
        ),
        "size_bytes": int(format_metadata.get("size") or 0)
        if str(format_metadata.get("size") or "").isdigit()
        else 0,
    }


def _profile_quality_blockers(
    *,
    visual_profile: str,
    video_metadata: Mapping[str, Any],
    requested_settings: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    requested = dict(requested_settings or {})
    width = int(video_metadata.get("width") or requested.get("width") or 0)
    height = int(video_metadata.get("height") or requested.get("height") or 0)
    fps = float(video_metadata.get("fps") or requested.get("fps") or 0.0)
    num_frames = int(video_metadata.get("nb_frames") or requested.get("num_frames") or 0)
    review_quality_profile = visual_profile == "review_quality"
    below_minimum = bool(
        width < REVIEW_QUALITY_MIN_WIDTH
        or height < REVIEW_QUALITY_MIN_HEIGHT
        or fps < REVIEW_QUALITY_MIN_FPS
        or num_frames < REVIEW_QUALITY_MIN_NUM_FRAMES
    )
    blockers: list[str] = []
    if review_quality_profile and below_minimum:
        blockers.append("review_quality_profile_media_below_minimum")
    profile_contract = {
        "visual_profile": visual_profile,
        "review_quality_profile": review_quality_profile,
        "review_quality_minimum": {
            "width": REVIEW_QUALITY_MIN_WIDTH,
            "height": REVIEW_QUALITY_MIN_HEIGHT,
            "fps": REVIEW_QUALITY_MIN_FPS,
            "num_frames": REVIEW_QUALITY_MIN_NUM_FRAMES,
        },
        "observed_or_requested": {
            "width": width,
            "height": height,
            "fps": _round_float(fps, 3),
            "num_frames": num_frames,
        },
        "review_quality_minimum_satisfied": not below_minimum,
        "smoke_only": bool(not review_quality_profile or below_minimum),
        "bounded_compromise_resolution_used": bool(
            review_quality_profile
            and not below_minimum
            and (width < 640 or height < 480 or fps < 15.0)
        ),
    }
    return blockers, profile_contract


def _generated_frame_quality_blockers(frame_stats: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for stats in frame_stats:
        if stats.get("status") != "completed":
            blockers.extend(str(item) for item in stats.get("blockers") or [])
            continue
        if float(stats.get("mean_luma") or 0.0) < 35.0 or float(
            stats.get("dark_pixel_ratio") or 0.0
        ) > 0.55:
            blockers.append("wam_generated_frame_too_dark_for_review")
        if (
            float(stats.get("std_luma") or 0.0) < 10.0
            or float(stats.get("entropy_bits") or 0.0) < 2.4
            or float(stats.get("dominant_luma_bin_ratio") or 0.0) > 0.70
        ):
            blockers.append("wam_generated_frame_flat_or_low_detail")
        drift = stats.get("drift_from_source") if isinstance(stats.get("drift_from_source"), Mapping) else {}
        edge_ratio = drift.get("edge_density_ratio_to_source")
        entropy_delta = drift.get("entropy_delta_from_source")
        mean_delta = drift.get("mean_luma_delta_from_source")
        if edge_ratio is not None and float(edge_ratio) < 0.25:
            blockers.append("wam_generated_frame_edge_structure_drift")
        if entropy_delta is not None and float(entropy_delta) < -1.5:
            blockers.append("wam_generated_frame_entropy_drift")
        if mean_delta is not None and float(mean_delta) < -35.0:
            blockers.append("wam_generated_frame_darkening_drift")
    return sorted(set(blockers))


def _write_contact_sheet(
    *,
    frame_paths: Sequence[Path],
    output_path: Path,
    labels: Sequence[str],
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        return {"status": "blocked", "blockers": [f"pillow_import_failed:{type(exc).__name__}"]}
    thumbnails = []
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((220, 140))
                canvas = Image.new("RGB", (220, 160), (245, 245, 245))
                x = (220 - thumb.width) // 2
                y = 18 + (140 - thumb.height) // 2
                canvas.paste(thumb, (x, y))
                thumbnails.append(canvas)
        except Exception:
            continue
    if not thumbnails:
        return {"status": "blocked", "blockers": ["no_readable_frames_for_contact_sheet"]}
    columns = min(4, max(1, len(thumbnails)))
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 220, rows * 160), (235, 238, 241))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - Pillow always provides default in normal envs.
        font = None
    for index, thumb in enumerate(thumbnails):
        x = (index % columns) * 220
        y = (index // columns) * 160
        sheet.paste(thumb, (x, y))
        label = labels[index] if index < len(labels) else f"frame {index}"
        draw.rectangle((x, y, x + 219, y + 17), fill=(28, 34, 42))
        draw.text((x + 6, y + 4), label[:36], fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "status": "completed",
        "path": str(output_path),
        "frame_count": len(thumbnails),
        "width": sheet.width,
        "height": sheet.height,
    }


def write_persistent_wam_visual_quality_artifacts(
    *,
    job_dir: str | Path,
    generated_at: str,
    source_frame_path: str | Path | None,
    generated_frame_paths: Sequence[str | Path],
    review_video_path: str | Path | None = None,
    video_status: Mapping[str, Any] | None = None,
    visual_profile: str = "smoke",
    requested_settings: Mapping[str, Any] | None = None,
    provider_status: str | None = None,
    live_wam_generation_success_count: int = 0,
    learned_wam_model_success_count: int = 0,
    structural_fallback_used: bool = False,
    target_object_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Write review-quality visual QA artifacts for a persistent policy/WAM rollout."""
    job = Path(job_dir).expanduser().resolve()
    job.mkdir(parents=True, exist_ok=True)
    normalized_profile = visual_profile if visual_profile in {"smoke", "review_quality"} else "smoke"
    source_qa = assess_source_policy_observation_visual_qa(
        source_frame_path,
        generated_at=generated_at,
        target_object_id=target_object_id,
        task_id=task_id,
        visual_profile=normalized_profile,
        review_quality_required=normalized_profile == "review_quality",
    )
    source_qa_path = job / "source_policy_observation_visual_qa.json"
    source_qa_path.write_text(json.dumps(source_qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_stats = source_qa.get("metrics") if isinstance(source_qa.get("metrics"), Mapping) else {}
    frame_paths = [Path(path).expanduser() for path in generated_frame_paths]
    frame_stats = [
        _frame_visual_stats(
            frame_path,
            role="wam_generated_next_observation",
            frame_index=index + 1,
            source_frame_stats=source_stats,
        )
        for index, frame_path in enumerate(frame_paths)
    ]
    frame_stats_path = job / "wam_rollout_frame_stats.jsonl"
    with frame_stats_path.open("w", encoding="utf-8") as handle:
        for row in frame_stats:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    contact_frame_paths = []
    contact_labels = []
    if source_frame_path:
        contact_frame_paths.append(Path(source_frame_path).expanduser())
        contact_labels.append("source")
    for index, frame_path in enumerate(frame_paths[:15], start=1):
        contact_frame_paths.append(frame_path)
        contact_labels.append(f"step {index}")
    contact_sheet_path = job / "wam_rollout_contact_sheet.jpg"
    contact_sheet = _write_contact_sheet(
        frame_paths=contact_frame_paths,
        output_path=contact_sheet_path,
        labels=contact_labels,
    )

    video_metadata = _video_metadata_from_status(video_status)
    if review_video_path:
        video_metadata["path"] = str(Path(review_video_path).expanduser())
    profile_blockers, profile_contract = _profile_quality_blockers(
        visual_profile=normalized_profile,
        video_metadata=video_metadata,
        requested_settings=requested_settings,
    )
    generated_blockers = _generated_frame_quality_blockers(frame_stats)
    source_blockers = list(source_qa.get("blockers") or [])
    blockers = sorted(set(source_blockers + generated_blockers + profile_blockers))
    generated_frames_pass = bool(frame_stats) and not generated_blockers
    source_pass = source_qa.get("status") == "passed_visual_quality_gate"
    provider_completed = provider_status == "completed"
    visual_success = bool(source_pass and generated_frames_pass and not profile_blockers)
    first_two_frame_blockers = _generated_frame_quality_blockers(frame_stats[:2])
    autoregressive_guard = {
        "autoregressive_chain_used": len(frame_stats) > 1,
        "generated_frame_count": len(frame_stats),
        "first_two_transition_visual_success": bool(frame_stats[:2] and not first_two_frame_blockers),
        "first_two_transition_blockers": first_two_frame_blockers,
        "periodic_reanchor_from_clean_render_used": False,
        "long_horizon_visual_drift_blocker": bool(len(frame_stats) > 2 and not visual_success),
        "long_rollout_should_not_be_overclaimed": bool(len(frame_stats) > 2 and not visual_success),
    }
    if autoregressive_guard["long_horizon_visual_drift_blocker"]:
        blockers.append("autoregressive_chain_visual_drift_or_quality_blocked_long_rollout")
        blockers = sorted(set(blockers))
    report = {
        "schema_version": PERSISTENT_WAM_VISUAL_QUALITY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed_visual_quality_gate" if visual_success else "failed_visual_quality_gate",
        "visual_success": visual_success,
        "visually_useful_rollout": visual_success,
        "visual_profile": normalized_profile,
        "profile_contract": profile_contract,
        "provider_status": provider_status,
        "provider_completed": provider_completed,
        "provider_completed_visual_quality_failed": bool(provider_completed and not visual_success),
        "live_wam_generation_success_count": int(live_wam_generation_success_count),
        "learned_wam_model_success_count": int(learned_wam_model_success_count),
        "live_wam_generation_success": bool(live_wam_generation_success_count > 0),
        "structural_fallback_used": bool(structural_fallback_used),
        "source_policy_observation_visual_qa_path": str(source_qa_path),
        "frame_stats_jsonl_path": str(frame_stats_path),
        "contact_sheet_path": str(contact_sheet_path) if contact_sheet_path.is_file() else None,
        "contact_sheet": contact_sheet,
        "review_video_path": str(Path(review_video_path).expanduser()) if review_video_path else None,
        "review_video_metadata": video_metadata,
        "generated_frame_count": len(frame_stats),
        "generated_frame_paths": [str(path) for path in frame_paths],
        "quality_summary": {
            "source_passed": source_pass,
            "source_mean_luma": source_stats.get("mean_luma"),
            "source_dark_pixel_ratio": source_stats.get("dark_pixel_ratio"),
            "source_edge_density": source_stats.get("edge_density"),
            "generated_frames_passed": generated_frames_pass,
            "minimum_generated_mean_luma": min(
                (float(row.get("mean_luma") or 0.0) for row in frame_stats),
                default=None,
            ),
            "maximum_generated_dark_pixel_ratio": max(
                (float(row.get("dark_pixel_ratio") or 0.0) for row in frame_stats),
                default=None,
            ),
        },
        "autoregressive_chain_guard": autoregressive_guard,
        "blockers": blockers,
        "claim_boundary": {
            "valid_mp4_or_provider_completed_is_not_visual_success": True,
            "live_wam_generation_success_can_coexist_with_visually_useful_rollout_false": True,
            "visual_quality_is_not_task_success_proof": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
            "raw_secret_values_recorded": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    report_path = job / "wam_rollout_visual_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def visual_smoke_generated_rollouts_for_review(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Return a lightweight visual sanity check for generated rollout videos."""
    frame_dir = output_dir / "generated_rollout_frame_review" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rollout_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency/environment edge.
        return {
            "schema_version": VISUAL_SMOKE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked_visual_probe_failed",
            "blockers": [f"opencv_import_failed:{type(exc).__name__}"],
            "rollout_count": len(rollouts),
            "rollouts": [],
            "claim_boundary": {
                "valid_mp4_file_generated": bool(rollouts),
                "visual_rollout_useful_for_task_success_review": False,
                "raw_secret_values_recorded": False,
                "secret_hashes_recorded": False,
            },
        }

    for rollout_index, rollout in enumerate(rollouts):
        video_path = _rollout_video_path(rollout)
        rollout_id = str(rollout.get("rollout_id") or f"rollout_{rollout_index + 1:04d}")
        result: dict[str, Any] = {
            "rollout_id": rollout_id,
            "generated_video_path": str(video_path) if video_path else None,
            "status": "blocked_visual_probe_failed",
            "sampled_frames": [],
            "visual_quality_flags": {
                "first_frame_preserves_source_scene": False,
                "later_frames_flat_or_dark": False,
                "success_review_not_reliable_from_this_rollout": True,
            },
        }
        if not video_path or not video_path.is_file():
            result["blockers"] = ["generated_video_missing_for_visual_smoke"]
            blockers.append("generated_video_missing_for_visual_smoke")
            rollout_results.append(result)
            continue
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            result["blockers"] = ["generated_video_unreadable_for_visual_smoke"]
            blockers.append("generated_video_unreadable_for_visual_smoke")
            rollout_results.append(result)
            continue
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                result["blockers"] = ["generated_video_frame_count_unavailable"]
                blockers.append("generated_video_frame_count_unavailable")
                rollout_results.append(result)
                continue
            sample_indices = sorted(
                {
                    0,
                    min(frame_count - 1, max(0, frame_count // 5)),
                    min(frame_count - 1, max(0, (frame_count * 2) // 5)),
                    min(frame_count - 1, max(0, (frame_count * 3) // 5)),
                    min(frame_count - 1, max(0, (frame_count * 4) // 5)),
                    frame_count - 1,
                }
            )
            samples: list[dict[str, Any]] = []
            safe_rollout_id = _safe_component(rollout_id)
            first_hist = None
            first_edge_density = 0.0
            for sample_order, frame_index in enumerate(sample_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luma_min = int(gray.min())
                luma_max = int(gray.max())
                edges = cv2.Canny(gray, 50, 150)
                edge_density = float((edges > 0).mean())
                hist = cv2.calcHist(
                    [frame],
                    [0, 1, 2],
                    None,
                    [8, 8, 8],
                    [0, 256, 0, 256, 0, 256],
                )
                cv2.normalize(hist, hist)
                if first_hist is None:
                    first_hist = hist
                    first_edge_density = edge_density
                hist_correlation_to_first = float(
                    cv2.compareHist(first_hist, hist, cv2.HISTCMP_CORREL)
                )
                edge_density_ratio_to_first = (
                    edge_density / first_edge_density
                    if first_edge_density > 0.0
                    else 0.0
                )
                sample_path = frame_dir / f"{safe_rollout_id}_frame_{sample_order:03d}.jpg"
                cv2.imwrite(str(sample_path), frame)
                samples.append(
                    {
                        "sample_index": sample_order,
                        "frame_index": int(frame_index),
                        "path": str(sample_path),
                        "mean_luma": round(float(gray.mean()), 3),
                        "luma_min": luma_min,
                        "luma_max": luma_max,
                        "luma_range": luma_max - luma_min,
                        "edge_density": round(edge_density, 6),
                        "edge_density_ratio_to_first": round(
                            edge_density_ratio_to_first,
                            6,
                        ),
                        "histogram_correlation_to_first": round(
                            hist_correlation_to_first,
                            6,
                        ),
                    }
                )
            if not samples:
                result["blockers"] = ["generated_video_frames_unreadable_for_visual_smoke"]
                blockers.append("generated_video_frames_unreadable_for_visual_smoke")
                rollout_results.append(result)
                continue
            first_preserves_scene = (
                samples[0]["luma_range"] >= 100
                and samples[0].get("edge_density", 0.0) >= 0.005
            )
            later_samples = samples[1:]
            later_flat_or_dark = bool(
                later_samples
                and all(sample["luma_range"] < 40 for sample in later_samples)
            )
            later_lost_scene_structure = bool(
                later_samples
                and all(
                    sample.get("edge_density_ratio_to_first", 0.0) < 0.10
                    and sample.get("histogram_correlation_to_first", 0.0) < 0.25
                    for sample in later_samples
                )
            )
            first_frame_not_scene_like = not first_preserves_scene
            quality_blockers = []
            if first_frame_not_scene_like:
                quality_blockers.append(
                    "generated_rollout_first_frame_not_scene_like"
                )
            if later_flat_or_dark:
                quality_blockers.append("generated_rollout_later_frames_flat_or_dark")
            if later_lost_scene_structure:
                quality_blockers.append(
                    "generated_rollout_later_frames_lost_scene_structure"
                )
            result.update(
                {
                    "status": "failed_visual_quality_smoke"
                    if quality_blockers
                    else "passed_visual_quality_smoke",
                    "frame_count": frame_count,
                    "sampled_frames": samples,
                    "visual_quality_flags": {
                        "first_frame_preserves_source_scene": first_preserves_scene,
                        "later_frames_flat_or_dark": later_flat_or_dark,
                        "later_frames_lost_scene_structure": later_lost_scene_structure,
                        "success_review_not_reliable_from_this_rollout": bool(
                            quality_blockers
                        ),
                    },
                    "blockers": quality_blockers,
                }
            )
            blockers.extend(quality_blockers)
        finally:
            capture.release()
        rollout_results.append(result)

    useful = bool(rollout_results) and all(
        row.get("status") == "passed_visual_quality_smoke" for row in rollout_results
    )
    status = (
        "not_applicable_missing_rollouts"
        if not rollouts
        else "passed_visual_quality_smoke"
        if useful
        else "failed_visual_quality_smoke"
        if {
            "generated_rollout_first_frame_not_scene_like",
            "generated_rollout_later_frames_flat_or_dark",
            "generated_rollout_later_frames_lost_scene_structure",
        }.intersection(blockers)
        else "blocked_visual_probe_failed"
    )
    return {
        "schema_version": VISUAL_SMOKE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "rollout_count": len(rollouts),
        "rollouts": rollout_results,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "valid_mp4_file_generated": bool(rollouts),
            "visual_rollout_useful_for_task_success_review": useful,
            "visual_smoke_is_not_forward_inverse_consistency": True,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        },
    }
