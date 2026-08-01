"""Bind decoded dual-fisheye observations to one frozen grouped dataset.

This adapter is downstream of native container probing and pixel decoding. It
does not infer lens identity or synchronization: both must already be present in
the validated native-360 normalization artifacts. Synchronized front/rear
observations receive one immutable split assignment so a hidden counterpart can
never leak into a candidate method's dataset.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image

from .decision_evidence_contracts import canonical_digest
from .reconstruction_frame_dataset import compile_frozen_frame_dataset


NATIVE_360_GROUPED_DATASET_ADAPTER_VERSION = (
    "native_360_grouped_frame_dataset_adapter.v1"
)
_LENS_IDS = {"front", "rear"}
_REQUIRED_LOCAL_AUTHORITY = {
    "source_capture_rights_valid": True,
    "consent_valid": True,
    "privacy_review_valid": True,
    "retention_authorized": True,
    "local_processing_authorized": True,
    "provider_upload_authorized": False,
    "paid_compute_authorized": False,
}


class Native360FrameDatasetError(ValueError):
    """Stable fail-closed error for native paired-observation compilation."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _finite(value: Any, *, code: str) -> float:
    if isinstance(value, bool):
        raise Native360FrameDatasetError([code])
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Native360FrameDatasetError([code]) from exc
    if not math.isfinite(number):
        raise Native360FrameDatasetError([code])
    return number


def _optional_finite(value: Any, *, code: str, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    number = _finite(value, code=code)
    if nonnegative and number < 0:
        raise Native360FrameDatasetError([code])
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validated_image(
    *, root: Path, raw: Mapping[str, Any], declared: Mapping[str, Any], ordinal: int
) -> tuple[str, dict[str, Any]]:
    text = str(raw.get("artifact_relative_path") or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_path_unsafe:{ordinal}"]
        )
    lexical = root / Path(*relative.parts)
    if lexical.is_symlink():
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_artifact_invalid:{ordinal}"]
        )
    path = lexical.resolve()
    if (
        (root != path and root not in path.parents)
        or path.is_symlink()
        or not path.is_file()
        or not _is_digest(raw.get("digest"))
        or _sha256_file(path) != raw.get("digest")
    ):
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_artifact_invalid:{ordinal}"]
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            actual_size = image.size
    except (OSError, SyntaxError) as exc:
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_image_invalid:{ordinal}"]
        ) from exc
    image_metadata = dict(raw.get("image_metadata") or {})
    declared_size = (declared.get("width"), declared.get("height"))
    if (
        actual_size != declared_size
        or image_metadata.get("width") != declared_size[0]
        or image_metadata.get("height") != declared_size[1]
    ):
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_dimensions_invalid:{ordinal}"]
        )
    return relative.as_posix(), image_metadata


def _validated_parent_artifacts(
    *,
    capture_digest: str,
    normalization_result: Mapping[str, Any],
    rig_declaration: Mapping[str, Any],
    dual_fisheye_binding: Mapping[str, Any],
) -> None:
    normalization_digest = normalization_result.get(
        "native_360_normalization_digest"
    )
    if (
        normalization_result.get("schema_version")
        != "native_360_capture_normalization.v1"
        or normalization_result.get("source_capture_digest") != capture_digest
        or normalization_result.get("status") != "normalized"
        or normalization_result.get("blockers") != []
        or normalization_result.get("raw_native_bytes_remain_authoritative")
        is not True
        or normalization_result.get("original_native_bytes_modified") is not False
        or not _is_digest(normalization_digest)
        or normalization_digest
        != canonical_digest(
            normalization_result, digest_field="native_360_normalization_digest"
        )
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_normalization_invalid"]
        )
    if (
        rig_declaration.get("schema_version") != "camera_360_rig_declaration.v1"
        or rig_declaration.get("capture_digest") != capture_digest
        or rig_declaration.get("calibration_status") != "valid"
        or rig_declaration.get("blockers") != []
        or rig_declaration.get("rig_declaration_digest")
        != canonical_digest(rig_declaration, digest_field="rig_declaration_digest")
        or normalization_result.get("rig_declaration_digest")
        != rig_declaration.get("rig_declaration_digest")
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_rig_invalid"]
        )
    if (
        dual_fisheye_binding.get("schema_version")
        != "dual_fisheye_stream_binding.v1"
        or dual_fisheye_binding.get("capture_digest") != capture_digest
        or dual_fisheye_binding.get("all_segments_synchronized") is not True
        or dual_fisheye_binding.get("blockers") != []
        or dual_fisheye_binding.get("dual_fisheye_binding_digest")
        != canonical_digest(
            dual_fisheye_binding, digest_field="dual_fisheye_binding_digest"
        )
        or normalization_result.get("dual_fisheye_binding_digest")
        != dual_fisheye_binding.get("dual_fisheye_binding_digest")
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_stream_binding_invalid"]
        )


def compile_native_360_grouped_frame_dataset(
    *,
    artifact_root: str | Path,
    intake_id: str,
    capture_digest: str,
    normalization_result: Mapping[str, Any],
    rig_declaration: Mapping[str, Any],
    dual_fisheye_binding: Mapping[str, Any],
    decoded_lens_frames: Sequence[Mapping[str, Any]],
    runtime_identity: str,
    runtime_digest: str,
    implementation_digest: str,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Compile one validated single-segment dual-fisheye source into frozen splits."""

    if not _is_digest(capture_digest):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_capture_digest_invalid"]
        )
    if any(
        authority_used.get(key) is not expected
        for key, expected in _REQUIRED_LOCAL_AUTHORITY.items()
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_authority_invalid"]
        )
    _validated_parent_artifacts(
        capture_digest=capture_digest,
        normalization_result=normalization_result,
        rig_declaration=rig_declaration,
        dual_fisheye_binding=dual_fisheye_binding,
    )
    if normalization_result.get("source_capture_identity") != intake_id:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_intake_identity_mismatch"]
        )
    segments = dual_fisheye_binding.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_multisegment_timeline_unavailable"]
        )
    segment = segments[0]
    if not isinstance(segment, Mapping) or segment.get("sequence_index") != 0:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_segment_invalid"]
        )
    files = segment.get("files")
    lens_streams = segment.get("lens_streams")
    frame_pairs = segment.get("frame_pairs")
    if (
        not isinstance(files, list)
        or len(files) != 1
        or not isinstance(lens_streams, list)
        or not isinstance(frame_pairs, list)
        or not frame_pairs
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_segment_invalid"]
        )
    source_reference = files[0]
    if not isinstance(source_reference, Mapping):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_source_invalid"]
        )
    source_relative_path = str(source_reference.get("relative_path") or "")
    source_digest = source_reference.get("digest")
    normalized_source_references = {
        (str(row.get("relative_path") or ""), row.get("digest"))
        for row in normalization_result.get("original_file_references", [])
        if isinstance(row, Mapping)
    }
    streams = {
        str(row.get("lens_id") or ""): row
        for row in lens_streams
        if isinstance(row, Mapping)
    }
    if (
        set(streams) != _LENS_IDS
        or not _is_digest(source_digest)
        or (source_relative_path, source_digest) not in normalized_source_references
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_source_invalid"]
        )

    expected: dict[tuple[int, str], dict[str, Any]] = {}
    for pair in frame_pairs:
        if not isinstance(pair, Mapping):
            raise Native360FrameDatasetError(
                ["native_360_grouped_dataset_frame_pair_invalid"]
            )
        pair_index = pair.get("pair_index")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int):
            raise Native360FrameDatasetError(
                ["native_360_grouped_dataset_frame_pair_invalid"]
            )
        for lens_id in sorted(_LENS_IDS):
            stream = streams[lens_id]
            expected[(pair_index, lens_id)] = {
                "source_relative_path": stream.get("source_relative_path"),
                "source_digest": stream.get("source_digest"),
                "stream_index": stream.get("stream_index"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "source_pts_seconds": pair.get(f"{lens_id}_pts_seconds"),
                "group_reference_pts_seconds": pair.get("front_pts_seconds"),
            }
    if len(expected) != len(frame_pairs) * 2:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_frame_pair_duplicate"]
        )

    first_pair_pts = _finite(
        frame_pairs[0].get("front_pts_seconds"),
        code="native_360_grouped_dataset_frame_pts_invalid",
    )
    selected_frames: list[dict[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    for ordinal, raw in enumerate(decoded_lens_frames):
        if not isinstance(raw, Mapping):
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_frame_invalid:{ordinal}"]
            )
        pair_index = raw.get("pair_index")
        lens_id = str(raw.get("lens_id") or "")
        key = (pair_index, lens_id)
        declared = expected.get(key)
        if declared is None or key in observed:
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_binding_invalid:{ordinal}"]
            )
        observed.add(key)
        pts = _finite(
            raw.get("source_pts_seconds"),
            code=f"native_360_grouped_dataset_frame_pts_invalid:{ordinal}",
        )
        declared_pts = _finite(
            declared["source_pts_seconds"],
            code="native_360_grouped_dataset_frame_pair_invalid",
        )
        group_reference_pts = _finite(
            declared["group_reference_pts_seconds"],
            code="native_360_grouped_dataset_frame_pair_invalid",
        )
        if (
            raw.get("segment_sequence_index") != 0
            or raw.get("source_relative_path") != declared["source_relative_path"]
            or raw.get("source_digest") != declared["source_digest"]
            or raw.get("stream_index") != declared["stream_index"]
            or raw.get("decoded_frame_index") != pair_index
            or not math.isclose(pts, declared_pts, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_binding_invalid:{ordinal}"]
            )
        artifact_relative_path, image_metadata = _validated_image(
            root=Path(artifact_root).expanduser().resolve(),
            raw=raw,
            declared=declared,
            ordinal=ordinal,
        )
        source_dts_seconds = _optional_finite(
            raw.get("source_dts_seconds"),
            code=f"native_360_grouped_dataset_frame_dts_invalid:{ordinal}",
        )
        duration_seconds = _optional_finite(
            raw.get("duration_seconds"),
            code=f"native_360_grouped_dataset_frame_duration_invalid:{ordinal}",
            nonnegative=True,
        )
        group_id = f"segment-0000-pair-{pair_index:09d}"
        selected_frames.append(
            {
                "frame_id": f"{group_id}-{lens_id}",
                "decoded_frame_index": pair_index,
                "t_video_sec": round(group_reference_pts - first_pair_pts, 9),
                "source_pts_seconds": pts,
                "source_dts_seconds": source_dts_seconds,
                "duration_seconds": duration_seconds,
                "key_frame": bool(raw.get("key_frame")),
                "artifact_relative_path": artifact_relative_path,
                "digest": raw.get("digest"),
                "image_metadata": image_metadata,
                "quality_signals": dict(raw.get("quality_signals") or {}),
                "source_camera_identity": lens_id,
                "observation_group_id": group_id,
            }
        )
    if observed != set(expected):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decoded_observations_incomplete"]
        )

    return compile_frozen_frame_dataset(
        artifact_root=artifact_root,
        intake_id=intake_id,
        capture_digest=capture_digest,
        capture_authority_profile="camera_360_native",
        source_video_relative_path=source_relative_path,
        source_video_digest=str(source_digest),
        decoded_frame_count=len(selected_frames),
        selected_frames=selected_frames,
        stream_metadata={
            "camera_representation": "calibrated_dual_fisheye_rig",
            "source_camera_identities": ["front", "rear"],
            "shared_physical_observation_groups": True,
            "group_timestamp_reference": "front_lens_decoded_pts",
            "native_360_normalization_digest": normalization_result[
                "native_360_normalization_digest"
            ],
            "dual_fisheye_binding_digest": dual_fisheye_binding[
                "dual_fisheye_binding_digest"
            ],
            "group_adapter_version": NATIVE_360_GROUPED_DATASET_ADAPTER_VERSION,
        },
        runtime_identity=runtime_identity,
        runtime_digest=runtime_digest,
        implementation_digest=implementation_digest,
        source_commit_sha=source_commit_sha,
        rights_and_retention=authority_used,
        selection_rule="evenly_spaced_actual_decoded_pts_with_endpoints_v1",
        parent_artifact={
            "native_360_normalization_digest": normalization_result[
                "native_360_normalization_digest"
            ],
            "dual_fisheye_binding_digest": dual_fisheye_binding[
                "dual_fisheye_binding_digest"
            ],
        },
        timestamp=timestamp,
        camera_calibration_binding={
            "camera_360_rig_declaration_digest": rig_declaration[
                "rig_declaration_digest"
            ]
        },
        coordinate_frame_declaration=normalization_result[
            "coordinate_frame_declaration"
        ],
    )


__all__ = [
    "NATIVE_360_GROUPED_DATASET_ADAPTER_VERSION",
    "Native360FrameDatasetError",
    "compile_native_360_grouped_frame_dataset",
]
