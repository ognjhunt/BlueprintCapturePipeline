"""Generated WAM video validation helpers.

These checks distinguish a generated video file that can be decoded for review
from a mere path or placeholder byte blob. Higher-level visual quality checks
still decide whether the decoded rollout is useful for task-success judging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "wam_generated_video_review_validation.v1"
VISUAL_SMOKE_SCHEMA_VERSION = "wam_generated_rollout_visual_smoke.v1"


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
