"""Record one H.264 stream per DROID camera at the true 15 Hz control rate.

The query-cadence policy-input PNGs remain the authoritative record of what a
policy consumed.  What they cannot answer is what a robotics team asks first:
show me the episode, at real rate, per camera.  This recorder produces that
lab-facing stream -- one frame per environment step, per camera, encoded as
``avc1`` H.264 at exactly ``DROID_CONTROL_FPS`` -- while keeping the lossy
video audit-linked to the exact rendered pixels through per-frame raw digests.

The renders themselves are already paid for: with ``render_interval`` equal to
``decimation`` the simulator renders once per environment step, and historic
runs simply discarded 88% of those frames.  Recording adds only buffer reads
and incremental encoding, never an extra render.

Frames are written incrementally so an episode never holds its video in
memory.  Frame ``i`` is the observation *before* control step ``i``, matching
the step trace's pre-step state semantics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest
try:  # flat provider-bundle layout
    from episode_visual_evidence import (
        REVIEW_VIDEO_CODEC,
        REVIEW_VIDEO_CONTAINER,
        REVIEW_VIDEO_FOURCC,
    )
except ModuleNotFoundError:  # repository package
    from .episode_visual_evidence import (
        REVIEW_VIDEO_CODEC,
        REVIEW_VIDEO_CONTAINER,
        REVIEW_VIDEO_FOURCC,
    )

DATASET_CAPTURE_SCHEMA_VERSION = "adp009d_dataset_capture.v1"

# DROID's published control rate; one frame per environment step.
DROID_CONTROL_FPS = 15.0

# DROID view keys as served to policies, mapped to the bare stream ids labs
# know from the DROID/LeRobot releases.
_DROID_VIEW_PREFIX = "observation/"
_KNOWN_DROID_STREAMS = {
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
}

FRAME_ALIGNMENT_STATEMENT = "frame_index_i_is_the_observation_before_control_step_i"


class DatasetCaptureError(ValueError):
    """Fail-closed dataset capture contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def droid_stream_id_for_view(view_key: str) -> str:
    """Map ``observation/<stream>`` to the bare DROID stream id, fail-closed."""

    if not str(view_key).startswith(_DROID_VIEW_PREFIX):
        raise DatasetCaptureError([f"dataset_capture_view_not_droid:{view_key}"])
    stream = str(view_key)[len(_DROID_VIEW_PREFIX) :]
    if stream not in _KNOWN_DROID_STREAMS:
        raise DatasetCaptureError([f"dataset_capture_view_not_droid:{view_key}"])
    return stream


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class _StreamWriter:
    """One camera's incremental encoder plus its per-frame digest ledger."""

    def __init__(self, *, video_path: Path, frames_per_second: float):
        self._video_path = video_path
        self._frames_per_second = float(frames_per_second)
        self._writer: Any = None
        self.shape: tuple[int, int] | None = None
        self.frame_raw_rgb_sha256: list[str] = []

    def write(self, frame: Any) -> None:
        import cv2
        import numpy as np

        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise DatasetCaptureError(
                [f"dataset_capture_frame_not_uint8_rgb:{array.dtype}:{array.shape}"]
            )
        height, width = int(array.shape[0]), int(array.shape[1])
        if self.shape is None:
            self.shape = (height, width)
            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            if self._video_path.exists() or self._video_path.is_symlink():
                raise DatasetCaptureError(
                    [f"dataset_capture_video_overwrite_forbidden:{self._video_path.name}"]
                )
            writer = cv2.VideoWriter(
                str(self._video_path),
                cv2.VideoWriter_fourcc(*REVIEW_VIDEO_FOURCC),
                self._frames_per_second,
                (width, height),
            )
            if not writer.isOpened():
                raise DatasetCaptureError(
                    [f"dataset_capture_encoder_unavailable:{REVIEW_VIDEO_FOURCC}"]
                )
            self._writer = writer
        elif self.shape != (height, width):
            raise DatasetCaptureError(
                [
                    "dataset_capture_frame_shape_changed:"
                    f"{self.shape}!={(height, width)}"
                ]
            )
        contiguous = np.ascontiguousarray(array)
        self.frame_raw_rgb_sha256.append(
            "sha256:" + hashlib.sha256(contiguous.tobytes()).hexdigest()
        )
        self._writer.write(cv2.cvtColor(contiguous, cv2.COLOR_RGB2BGR))

    def close(self) -> dict[str, Any]:
        import cv2

        if self._writer is None or self.shape is None:
            raise DatasetCaptureError(["dataset_capture_stream_never_wrote"])
        self._writer.release()
        self._writer = None
        if not self._video_path.is_file() or self._video_path.stat().st_size <= 0:
            raise DatasetCaptureError(["dataset_capture_video_not_written"])
        capture = cv2.VideoCapture(str(self._video_path))
        if not capture.isOpened():
            raise DatasetCaptureError(["dataset_capture_decode_round_trip_unavailable"])
        decoded_count = 0
        try:
            while True:
                ok, decoded = capture.read()
                if not ok:
                    break
                if decoded is None or decoded.shape[:2] != self.shape:
                    raise DatasetCaptureError(
                        ["dataset_capture_decode_round_trip_shape_mismatch"]
                    )
                decoded_count += 1
        finally:
            capture.release()
        if decoded_count != len(self.frame_raw_rgb_sha256):
            raise DatasetCaptureError(
                [
                    "dataset_capture_decode_round_trip_frame_count_mismatch:"
                    f"{decoded_count}!={len(self.frame_raw_rgb_sha256)}"
                ]
            )
        return {
            "container": REVIEW_VIDEO_CONTAINER,
            "codec": REVIEW_VIDEO_CODEC,
            "fourcc": REVIEW_VIDEO_FOURCC,
            "frames_per_second": self._frames_per_second,
            "decoded_frame_count": decoded_count,
            "decode_round_trip_passed": True,
            "sha256": _file_sha256(self._video_path),
            "size_bytes": self._video_path.stat().st_size,
        }


class DatasetCaptureRecorder:
    """Incrementally record per-camera control-rate video for one episode."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        episode_id: str,
        view_keys: Sequence[str],
        frames_per_second: float = DROID_CONTROL_FPS,
    ):
        if not str(episode_id):
            raise DatasetCaptureError(["dataset_capture_episode_id_missing"])
        streams = {view: droid_stream_id_for_view(view) for view in view_keys}
        if len(set(streams.values())) != len(streams):
            raise DatasetCaptureError(["dataset_capture_duplicate_stream_ids"])
        if not streams:
            raise DatasetCaptureError(["dataset_capture_no_views_requested"])
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._episode_id = str(episode_id)
        self._view_to_stream = streams
        self.episode_id = self._episode_id
        self.view_keys = tuple(sorted(streams))
        self._frames_per_second = float(frames_per_second)
        self._next_step_index = 0
        self._finalized = False
        capture_dir = self._output_dir / "media" / self._episode_id / "dataset"
        self._writers = {
            stream: _StreamWriter(
                video_path=capture_dir / f"{stream}.{REVIEW_VIDEO_CONTAINER}",
                frames_per_second=self._frames_per_second,
            )
            for stream in streams.values()
        }

    def _write_views(self, views: Mapping[str, Any]) -> None:
        if set(views) != set(self._view_to_stream):
            raise DatasetCaptureError(
                [
                    "dataset_capture_view_set_mismatch:"
                    f"{sorted(map(str, views))}!={sorted(self._view_to_stream)}"
                ]
            )
        for view_key, frame in sorted(views.items()):
            self._writers[self._view_to_stream[view_key]].write(frame)

    def record_step(self, *, step_index: int, views: Mapping[str, Any]) -> None:
        """Record the pre-step observation for control step ``step_index``."""

        if self._finalized:
            raise DatasetCaptureError(["dataset_capture_already_finalized"])
        if int(step_index) != self._next_step_index:
            raise DatasetCaptureError(
                [
                    "dataset_capture_step_index_not_contiguous:"
                    f"{step_index}!={self._next_step_index}"
                ]
            )
        self._write_views(views)
        self._next_step_index += 1

    def finalize(self, *, terminal_views: Mapping[str, Any] | None) -> dict[str, Any]:
        """Close the streams, prove the decode round trip, seal the manifest."""

        if self._finalized:
            raise DatasetCaptureError(["dataset_capture_already_finalized"])
        if self._next_step_index < 1:
            raise DatasetCaptureError(["dataset_capture_no_frames_recorded"])
        if terminal_views is not None:
            self._write_views(terminal_views)
        self._finalized = True

        streams: dict[str, dict[str, Any]] = {}
        for view_key, stream_id in sorted(self._view_to_stream.items()):
            writer = self._writers[stream_id]
            video = writer.close()
            video_path = (
                self._output_dir
                / "media"
                / self._episode_id
                / "dataset"
                / f"{stream_id}.{REVIEW_VIDEO_CONTAINER}"
            )
            first_shape = writer.shape
            streams[stream_id] = {
                "view_key": view_key,
                "video": {
                    **video,
                    "relative_path": video_path.relative_to(self._output_dir).as_posix(),
                },
                "width": first_shape[1],
                "height": first_shape[0],
                "frame_raw_rgb_sha256": list(writer.frame_raw_rgb_sha256),
            }

        record: dict[str, Any] = {
            "schema_version": DATASET_CAPTURE_SCHEMA_VERSION,
            "episode_id": self._episode_id,
            "frames_per_second": self._frames_per_second,
            "frame_count": self._next_step_index,
            "terminal_frame_included": terminal_views is not None,
            "frame_alignment": FRAME_ALIGNMENT_STATEMENT,
            "video_is_lossy_frame_digests_are_of_raw_rgb": True,
            "streams": streams,
        }
        record["capture_digest"] = canonical_digest(
            record, digest_field="capture_digest"
        )

        import json

        manifest_path = (
            self._output_dir
            / "media"
            / self._episode_id
            / "dataset"
            / "dataset_capture_manifest.json"
        )
        if manifest_path.exists() or manifest_path.is_symlink():
            raise DatasetCaptureError(["dataset_capture_manifest_overwrite_forbidden"])
        manifest_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record["manifest_relative_path"] = manifest_path.relative_to(
            self._output_dir
        ).as_posix()
        return record


__all__ = [
    "DATASET_CAPTURE_SCHEMA_VERSION",
    "DROID_CONTROL_FPS",
    "FRAME_ALIGNMENT_STATEMENT",
    "DatasetCaptureError",
    "DatasetCaptureRecorder",
    "droid_stream_id_for_view",
]
