"""Local, deterministic frame-quality measurements for capture QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_intake import CaptureIntakeError
from .capture_qa import CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION


CAPTURE_QUALITY_ANALYZER_ID = "blueprint_local_frame_quality.v1"
DEFAULT_SAMPLE_COUNT = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    normalized = json.loads(json.dumps(value))
    if omit:
        normalized.pop(omit, None)
    return "sha256:" + hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _resize_for_analysis(frame: Any, cv2: Any) -> Any:
    height, width = frame.shape[:2]
    if width <= 640:
        return frame
    scale = 640.0 / float(width)
    return cv2.resize(frame, (640, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _overlap_supported(left: Any, right: Any, cv2: Any) -> bool | None:
    orb = cv2.ORB_create(nfeatures=600)
    left_points, left_descriptors = orb.detectAndCompute(left, None)
    right_points, right_descriptors = orb.detectAndCompute(right, None)
    if (
        left_descriptors is None
        or right_descriptors is None
        or len(left_points) < 20
        or len(right_points) < 20
    ):
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_descriptors, right_descriptors, k=2)
    good = [
        pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    ratio = len(good) / max(1, min(len(left_points), len(right_points)))
    return len(good) >= 15 and ratio >= 0.06


def _motion_and_rolling_shutter(
    left: Any, right: Any, cv2: Any, np: Any
) -> tuple[float | None, bool | None]:
    points = cv2.goodFeaturesToTrack(
        left,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points is None or len(points) < 20:
        return None, None
    tracked, status, _error = cv2.calcOpticalFlowPyrLK(left, right, points, None)
    if tracked is None or status is None:
        return None, None
    valid = status.reshape(-1) == 1
    source = points.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    if len(source) < 20:
        return None, None
    displacement = target - source
    magnitudes = np.linalg.norm(displacement, axis=1)
    median_motion = float(np.median(magnitudes))
    rows = source[:, 1]
    if float(np.ptp(rows)) < max(10.0, left.shape[0] * 0.25):
        return median_motion, None
    normalized_rows = (rows - float(np.mean(rows))) / max(float(np.ptp(rows)), 1.0)
    slope, intercept = np.polyfit(normalized_rows, displacement[:, 0], 1)
    predicted = slope * normalized_rows + intercept
    residual = float(np.sum((displacement[:, 0] - predicted) ** 2))
    total = float(np.sum((displacement[:, 0] - float(np.mean(displacement[:, 0]))) ** 2))
    r_squared = 1.0 - residual / total if total > 1e-9 else 0.0
    symptom = abs(float(slope)) >= 5.0 and r_squared >= 0.25
    return median_motion, symptom


def measure_sampled_frames(frames: Sequence[Any]) -> dict[str, Any]:
    """Measure deterministic quality fractions from already-decoded BGR frames."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:  # pragma: no cover - canonical install includes both
        raise CaptureIntakeError(["local_quality_analyzer_dependencies_missing"]) from exc
    prepared = [_resize_for_analysis(frame, cv2) for frame in frames if frame is not None]
    if len(prepared) < 2:
        raise CaptureIntakeError(["local_quality_analyzer_insufficient_decoded_frames"])
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in prepared]
    sharp = [float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 80.0 for gray in grays]
    exposed = []
    for gray in grays:
        mean_luma = float(np.mean(gray))
        clipped_fraction = float(np.mean((gray <= 5) | (gray >= 250)))
        exposed.append(30.0 <= mean_luma <= 225.0 and clipped_fraction <= 0.20)

    overlap_results: list[bool] = []
    motions: list[float] = []
    rolling_results: list[bool] = []
    for left, right in zip(grays, grays[1:]):
        overlap = _overlap_supported(left, right, cv2)
        if overlap is not None:
            overlap_results.append(overlap)
        motion, rolling = _motion_and_rolling_shutter(left, right, cv2, np)
        if motion is not None and math.isfinite(motion):
            motions.append(motion)
        if rolling is not None:
            rolling_results.append(rolling)

    measurements: dict[str, Any] = {
        "sharp_frame_fraction": round(sum(sharp) / len(sharp), 6),
        "well_exposed_frame_fraction": round(sum(exposed) / len(exposed), 6),
    }
    if overlap_results:
        measurements["visual_overlap_fraction"] = round(
            sum(overlap_results) / len(overlap_results), 6
        )
    if rolling_results:
        measurements["rolling_shutter_symptom_fraction"] = round(
            sum(rolling_results) / len(rolling_results), 6
        )
    if motions:
        measurements["median_interframe_motion_pixels"] = round(
            float(np.median(np.asarray(motions))), 6
        )
    return {
        "measurements": measurements,
        "sample_count": len(grays),
        "overlap_pair_count": len(overlap_results),
        "rolling_shutter_proxy_pair_count": len(rolling_results),
    }


def _sample_video(path: Path, *, sample_count: int) -> list[Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:  # pragma: no cover - canonical install includes both
        raise CaptureIntakeError(["local_quality_analyzer_dependencies_missing"]) from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise CaptureIntakeError(["local_quality_analyzer_video_open_failed"])
    try:
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count < 2:
            raise CaptureIntakeError(["local_quality_analyzer_insufficient_decoded_frames"])
        count = min(max(2, sample_count), frame_count)
        indices = sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, count)})
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def _compression_sufficiency(path: Path) -> tuple[float | None, float | None]:
    """Return a content-dependent bitrate sufficiency proxy and bpp/frame."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - canonical install includes OpenCV
        raise CaptureIntakeError(["local_quality_analyzer_dependencies_missing"]) from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None, None
    try:
        bitrate_kbps = float(capture.get(cv2.CAP_PROP_BITRATE))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    denominator = fps * width * height
    if not all(math.isfinite(value) and value > 0 for value in (bitrate_kbps, denominator)):
        return None, None
    bits_per_pixel_per_frame = bitrate_kbps * 1000.0 / denominator
    sufficient = 1.0 if bits_per_pixel_per_frame >= 0.035 else 0.0
    return sufficient, round(bits_per_pixel_per_frame, 6)


def analyze_capture_video_quality(
    video_path: Path,
    *,
    intake_id: str,
    source_file_sha256: str,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Decode sampled frames and emit a digest-bound quality-observations packet."""

    video_path = video_path.resolve(strict=True)
    actual_digest = _sha256(video_path)
    if actual_digest != source_file_sha256:
        raise CaptureIntakeError(["local_quality_analyzer_source_digest_mismatch"])
    result = measure_sampled_frames(_sample_video(video_path, sample_count=sample_count))
    compression_fraction, bits_per_pixel_per_frame = _compression_sufficiency(video_path)
    if compression_fraction is not None:
        result["measurements"]["compression_quality_fraction"] = compression_fraction
    packet = {
        "schema_version": CAPTURE_QUALITY_OBSERVATIONS_SCHEMA_VERSION,
        "source": "local_analyzer",
        "intake_id": intake_id,
        "source_file_sha256": actual_digest,
        "analyzer": {
            "analyzer_id": CAPTURE_QUALITY_ANALYZER_ID,
            "sample_count_requested": sample_count,
            "sample_count_decoded": result["sample_count"],
            "overlap_pair_count": result["overlap_pair_count"],
            "rolling_shutter_proxy_pair_count": result["rolling_shutter_proxy_pair_count"],
            "compression_bits_per_pixel_per_frame": bits_per_pixel_per_frame,
            "compression_minimum_bits_per_pixel_per_frame": 0.035,
        },
        "measurements": result["measurements"],
        "limitations": [
            "rolling_shutter_is_an_optical_flow_symptom_proxy_not_sensor_readout",
            "compression_is_a_content_dependent_bitrate_per_pixel_per_frame_proxy",
            "absence_of_privacy_sensitive_content_is_not_certified",
            "spatial_coverage_task_occlusion_and_robot_placement_require_task_aware_review",
        ],
    }
    packet["observations_digest"] = _digest(packet, omit="observations_digest")
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--intake-id", required=True)
    parser.add_argument("--source-file-sha256", required=True)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    packet = analyze_capture_video_quality(
        args.video,
        intake_id=args.intake_id,
        source_file_sha256=args.source_file_sha256,
        sample_count=args.sample_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_canonical_json(packet) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAPTURE_QUALITY_ANALYZER_ID",
    "analyze_capture_video_quality",
    "measure_sampled_frames",
]
