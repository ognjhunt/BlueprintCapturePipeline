"""Build attributable review-only views for a frozen OSCAR rollout.

The generated rollout remains the OSCAR output.  The skeleton-only,
visible-skeleton, binary-mask, and skeleton-region-occluded videos produced by
this module are deterministic review derivatives.  They are never valid WAM
feedback, physical outcomes, or ranking evidence by themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "policy_ranking_oscar_review_views.v1"


def _rgb_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _load_rgb_frames(
    path: str | Path,
    *,
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
    resize_to_contract: bool,
) -> tuple[list[np.ndarray], float]:
    source = Path(path).resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"video_open_failed:{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    try:
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, width) and not resize_to_contract:
                raise ValueError(
                    f"video_dimensions_mismatch:{source}:{rgb.shape[1]}:{rgb.shape[0]}:"
                    f"{width}:{height}"
                )
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
    finally:
        capture.release()
    if len(frames) != frame_count:
        raise ValueError(f"unexpected_frame_count:{source}:{len(frames)}:{frame_count}")
    return frames, fps


def _skeleton_masks(
    skeleton_rgb: np.ndarray,
    *,
    maximum_channel_threshold: int,
    chroma_threshold: int,
    dilation_radius_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(skeleton_rgb, axis=2)
    minimum = np.min(skeleton_rgb, axis=2)
    annotation = (maximum >= maximum_channel_threshold) & (maximum - minimum >= chroma_threshold)
    kernel_size = 2 * dilation_radius_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    occlusion = cv2.dilate(annotation.astype(np.uint8), kernel, iterations=1) > 0
    return annotation, occlusion


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float) -> dict[str, Any]:
    if not frames:
        raise ValueError("no_frames_to_write")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise ValueError(f"video_writer_open_failed:{path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    decoded, decoded_fps = _load_rgb_frames(
        path,
        start_frame=0,
        frame_count=len(frames),
        width=width,
        height=height,
        resize_to_contract=False,
    )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "codec_requested": "mp4v",
        "fps_requested": float(fps),
        "fps_decoded": decoded_fps,
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "pre_encode_rgb_frame_sha256": [_rgb_sha256(frame) for frame in frames],
        "decoded_rgb_frame_sha256": [_rgb_sha256(frame) for frame in decoded],
    }


def build_oscar_review_views(
    *,
    generated_video: str | Path,
    skeleton_video: str | Path,
    expected_generated_sha256: str,
    expected_skeleton_sha256: str,
    output_dir: str | Path,
    start_frame: int,
    frame_count: int = 81,
    width: int = 640,
    height: int = 480,
    fps: float = 14.0,
    maximum_channel_threshold: int = 48,
    chroma_threshold: int = 24,
    dilation_radius_pixels: int = 7,
    overlay_alpha: float = 0.85,
) -> dict[str, Any]:
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha_out_of_range")
    generated_path = Path(generated_video).resolve()
    skeleton_path = Path(skeleton_video).resolve()
    generated_sha256 = file_sha256(generated_path)
    skeleton_sha256 = file_sha256(skeleton_path)
    if generated_sha256 != expected_generated_sha256:
        raise ValueError("generated_video_sha256_mismatch")
    if skeleton_sha256 != expected_skeleton_sha256:
        raise ValueError("skeleton_video_sha256_mismatch")
    generated, generated_fps = _load_rgb_frames(
        generated_path,
        start_frame=0,
        frame_count=frame_count,
        width=width,
        height=height,
        resize_to_contract=False,
    )
    skeleton, skeleton_fps = _load_rgb_frames(
        skeleton_path,
        start_frame=start_frame,
        frame_count=frame_count,
        width=width,
        height=height,
        resize_to_contract=True,
    )
    if abs(generated_fps - fps) > 0.01 or abs(skeleton_fps - fps) > 0.01:
        raise ValueError(f"fps_contract_mismatch:{generated_fps}:{skeleton_fps}:{fps}")

    visible: list[np.ndarray] = []
    occluded: list[np.ndarray] = []
    binary_masks: list[np.ndarray] = []
    annotation_fractions: list[float] = []
    occlusion_fractions: list[float] = []
    for generated_frame, skeleton_frame in zip(generated, skeleton, strict=True):
        annotation_mask, occlusion_mask = _skeleton_masks(
            skeleton_frame,
            maximum_channel_threshold=maximum_channel_threshold,
            chroma_threshold=chroma_threshold,
            dilation_radius_pixels=dilation_radius_pixels,
        )
        annotation_fractions.append(float(np.mean(annotation_mask)))
        occlusion_fractions.append(float(np.mean(occlusion_mask)))

        blended = generated_frame.copy()
        mixed = (
            np.rint(
                (1.0 - overlay_alpha) * generated_frame.astype(np.float32)
                + overlay_alpha * skeleton_frame.astype(np.float32)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        blended[annotation_mask] = mixed[annotation_mask]
        visible.append(blended)

        masked = generated_frame.copy()
        masked[occlusion_mask] = 0
        occluded.append(masked)
        binary_mask = (occlusion_mask.astype(np.uint8) * 255)[:, :, None]
        binary_masks.append(np.repeat(binary_mask, 3, axis=2))

    output = Path(output_dir).resolve()
    media = {
        "skeleton_only": _write_video(output / "skeleton_only.mp4", skeleton, fps=fps),
        "visible_skeleton_review": _write_video(
            output / "visible_skeleton_review.mp4", visible, fps=fps
        ),
        "skeleton_region_occluded_scene": _write_video(
            output / "skeleton_region_occluded_scene.mp4", occluded, fps=fps
        ),
        "binary_occlusion_mask": _write_video(
            output / "binary_occlusion_mask.mp4", binary_masks, fps=fps
        ),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "inputs": {
            "generated_video": str(generated_path),
            "generated_video_sha256": generated_sha256,
            "skeleton_video": str(skeleton_path),
            "skeleton_video_sha256": skeleton_sha256,
            "skeleton_start_frame": start_frame,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
            "decoder": "opencv_videocapture_public_fallback",
            "skeleton_resize": "opencv_inter_linear",
        },
        "mask_contract": {
            "color_space": "decoded_rgb_uint8",
            "annotation_rule": (
                f"max_channel>={maximum_channel_threshold}_and_max_minus_min>={chroma_threshold}"
            ),
            "dilation": "opencv_elliptical_single_iteration",
            "dilation_radius_pixels": dilation_radius_pixels,
            "overlay_alpha": overlay_alpha,
            "occluded_pixel_value_rgb": [0, 0, 0],
            "annotation_fraction_mean": float(np.mean(annotation_fractions)),
            "annotation_fraction_min": float(np.min(annotation_fractions)),
            "annotation_fraction_max": float(np.max(annotation_fractions)),
            "occlusion_fraction_mean": float(np.mean(occlusion_fractions)),
            "occlusion_fraction_min": float(np.min(occlusion_fractions)),
            "occlusion_fraction_max": float(np.max(occlusion_fractions)),
        },
        "media": media,
        "attribution": {
            "generated_video_is_unchanged_oscar_output": True,
            "all_media_in_this_report_are_review_derivatives": True,
            "skeleton_only_is_intended_motion_evidence_only": True,
            "visible_skeleton_review_is_not_native_oscar_output": True,
            "skeleton_region_occluded_scene_is_not_inpainting": True,
            "skeleton_region_occluded_scene_does_not_guarantee_complete_robot_removal": True,
            "review_derivatives_must_not_feed_a_wam_or_candidate_policy": True,
        },
        "evidence_boundaries": {
            "physical_future_rgb_pixels_used": False,
            "physical_outcome_labels_accessed": False,
            "provider_called": False,
            "paid_resource_used": False,
            "ranking_credit": False,
            "wam_qualification_credit": False,
            "physical_success_credit": False,
            "thesis_support_credit": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(output / "review_view_manifest.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-video", required=True)
    parser.add_argument("--skeleton-video", required=True)
    parser.add_argument("--expected-generated-sha256", required=True)
    parser.add_argument("--expected-skeleton-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frame-count", type=int, default=81)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=14.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_oscar_review_views(
        generated_video=args.generated_video,
        skeleton_video=args.skeleton_video,
        expected_generated_sha256=args.expected_generated_sha256,
        expected_skeleton_sha256=args.expected_skeleton_sha256,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
