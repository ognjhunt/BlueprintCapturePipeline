"""Verify file-backed visual evidence for an engineered receptacle twin.

The verifier deliberately separates two claims:

* retained renders can show a source receptacle's rigid exterior and open rim;
* an empty interior in an independently authored twin is authored geometry, not
  an observation of an occupied or incompletely observed source interior.

All external JSON and PNG inputs are opened once, without following the final
path component, and are validated from the exact bytes read from that handle.
The resulting receipt is a visual design-basis receipt only.  It is not native
simulator qualification and does not establish physical equivalence.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .decision_evidence_contracts import canonical_digest
from .semantic_review_attestation import (
    SemanticReviewAttestationError,
    semantic_frame_evidence_digest,
    verify_semantic_review_attestation,
)


SCHEMA_VERSION = "engineered_receptacle_visual_design_basis.v1"
SUPPORTED_VISUAL_REVIEW_SCHEMAS = {
    "adp_deformable_scene_visual_review.v1",
    "adp_deformable_scene_visual_review.v2",
}
SUPPORTED_RENDER_MANIFEST_SCHEMAS = {"splat_scene_render.v1"}
FROZEN_MINIMUM_DIFFERING_LUMINANCE_FRACTION = 0.10
FROZEN_MINIMUM_LUMINANCE_ENTROPY_BITS = 0.50
FROZEN_MINIMUM_LUMINANCE_P05_P95_SPREAD = 8

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class EngineeredReceptacleVisualBasisError(ValueError):
    """A stable, typed failure at the file-backed evidence boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(error for error in errors if error)))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class VisualBasisLimits:
    """Resource limits applied before parsing or decoding untrusted evidence."""

    max_json_bytes: int = 8 * 1024 * 1024
    max_frame_bytes: int = 32 * 1024 * 1024
    max_frame_pixels: int = 16 * 1024 * 1024
    max_cited_frames: int = 256

    def validate(self) -> None:
        values = (
            self.max_json_bytes,
            self.max_frame_bytes,
            self.max_frame_pixels,
            self.max_cited_frames,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise EngineeredReceptacleVisualBasisError(["visual_basis_limits_invalid"])


@dataclass(frozen=True)
class _FileSnapshot:
    path: str
    size_bytes: int
    sha256: str
    data: bytes

    def binding(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _digest(value: Any) -> str:
    text = _string(value)
    return text if _SHA256.fullmatch(text) else ""


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _read_fd_once(
    fd: int,
    *,
    display_path: str,
    max_bytes: int,
    error_prefix: str,
) -> _FileSnapshot:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_not_regular"])
    if before.st_size <= 0 or before.st_size > max_bytes:
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_size_invalid"])
    chunks: list[bytes] = []
    remaining = before.st_size
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    after = os.fstat(fd)
    if len(data) != before.st_size or not _same_identity(before, after):
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_changed_during_read"])
    return _FileSnapshot(
        path=display_path,
        size_bytes=len(data),
        sha256=f"sha256:{hashlib.sha256(data).hexdigest()}",
        data=data,
    )


def _read_regular_file_once(
    path: str | os.PathLike[str], *, max_bytes: int, error_prefix: str
) -> _FileSnapshot:
    if not isinstance(path, (str, os.PathLike)):
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_path_invalid"])
    display_path = os.path.abspath(os.fspath(path))
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_nofollow_unavailable"])
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(display_path, flags)
    except OSError as exc:
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_open_failed"]) from exc
    try:
        return _read_fd_once(
            fd,
            display_path=display_path,
            max_bytes=max_bytes,
            error_prefix=error_prefix,
        )
    finally:
        os.close(fd)


def _lexical_frame_path(
    frame_root: str | os.PathLike[str], manifest_path: str
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(frame_root, (str, os.PathLike)) or not _string(manifest_path):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_path_invalid"])
    root = os.path.abspath(os.fspath(frame_root))
    candidate = (
        os.path.abspath(manifest_path)
        if os.path.isabs(manifest_path)
        else os.path.abspath(os.path.join(root, manifest_path))
    )
    try:
        contained = os.path.commonpath((root, candidate)) == root
    except ValueError as exc:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_outside_root"]) from exc
    if not contained:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_outside_root"])
    relative = os.path.relpath(candidate, root)
    parts = tuple(Path(relative).parts)
    if not parts or parts == (".",) or any(part in {"", ".", ".."} for part in parts):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_path_invalid"])
    return candidate, parts


def _read_frame_once(
    frame_root: str | os.PathLike[str],
    manifest_path: str,
    *,
    max_bytes: int,
) -> _FileSnapshot:
    display_path, parts = _lexical_frame_path(frame_root, manifest_path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_nofollow_unavailable"])
    directory_flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(os.path.abspath(os.fspath(frame_root)), directory_flags))
        for part in parts[:-1]:
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        frame_fd = os.open(parts[-1], file_flags, dir_fd=descriptors[-1])
        try:
            return _read_fd_once(
                frame_fd,
                display_path=display_path,
                max_bytes=max_bytes,
                error_prefix="visual_basis_frame",
            )
        finally:
            os.close(frame_fd)
    except EngineeredReceptacleVisualBasisError:
        raise
    except OSError as exc:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_open_failed"]) from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _strict_json(snapshot: _FileSnapshot, *, error_prefix: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            snapshot.data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_json_invalid"]) from exc
    if not isinstance(value, dict):
        raise EngineeredReceptacleVisualBasisError([f"{error_prefix}_json_invalid"])
    return value


def _validate_renderer(manifest: Mapping[str, Any]) -> dict[str, Any]:
    renderer = manifest.get("renderer_identity")
    if not isinstance(renderer, Mapping):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_renderer_invalid"])
    width = _integer(renderer.get("width"))
    height = _integer(renderer.get("height"))
    required_strings = (
        "name",
        "background_rgb_hex",
        "color_space",
        "node_version",
    )
    valid = (
        all(_string(renderer.get(key)) for key in required_strings)
        and _digest(renderer.get("harness_sha256"))
        and _digest(renderer.get("entry_sha256"))
        and width is not None
        and width > 0
        and height is not None
        and height > 0
        and _number(renderer.get("pixel_ratio")) is not None
        and _number(renderer.get("pixel_ratio")) > 0
        and _number(renderer.get("supersampling")) is not None
        and _number(renderer.get("supersampling")) > 0
        and isinstance(renderer.get("alpha"), bool)
        and renderer.get("output_format") == "lossless_png"
        and isinstance(renderer.get("dependency_versions"), Mapping)
        and bool(renderer.get("dependency_versions"))
    )
    render = manifest.get("render")
    valid = bool(
        valid
        and isinstance(render, Mapping)
        and render.get("status") == "completed"
        and render.get("returncode") == 0
    )
    source_digest = _digest(manifest.get("source_digest"))
    fidelity = manifest.get("appearance_fidelity")
    source_count = (
        _integer(fidelity.get("source_splat_count")) if isinstance(fidelity, Mapping) else None
    )
    retained_count = (
        _integer(fidelity.get("retained_splat_count")) if isinstance(fidelity, Mapping) else None
    )
    valid = bool(
        valid
        and source_digest
        and source_count is not None
        and source_count > 0
        and retained_count is not None
        and retained_count > 0
        and retained_count <= source_count
        and isinstance(fidelity.get("appearance_fidelity_qualified"), bool)
        and isinstance(fidelity.get("evaluation_input_authorized"), bool)
    )
    if not valid:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_renderer_invalid"])
    return {
        "renderer_identity": dict(renderer),
        "renderer_identity_digest": canonical_digest(renderer),
        "source_digest": source_digest,
        "source_splat_count": source_count,
        "retained_splat_count": retained_count,
        "appearance_fidelity_qualified": fidelity.get("appearance_fidelity_qualified"),
        "evaluation_input_authorized": fidelity.get("evaluation_input_authorized"),
    }


def _index_rows(value: Any, *, id_key: str, error: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EngineeredReceptacleVisualBasisError([error])
    result: dict[str, Mapping[str, Any]] = {}
    for row in value:
        row_id = _string(row.get(id_key)) if isinstance(row, Mapping) else ""
        if not row_id or row_id in result:
            raise EngineeredReceptacleVisualBasisError([error])
        result[row_id] = row
    return result


def _validate_calibration(
    calibration: Mapping[str, Any], *, expected_width: int, expected_height: int
) -> dict[str, Any]:
    intrinsics = calibration.get("intrinsics")
    vectors = ("position_world_m", "target_world_m", "up_world")
    vectors_valid = all(
        isinstance(calibration.get(key), list)
        and len(calibration[key]) == 3
        and all(_number(item) is not None for item in calibration[key])
        for key in vectors
    )
    if vectors_valid:
        position = [float(item) for item in calibration["position_world_m"]]
        target = [float(item) for item in calibration["target_world_m"]]
        up = [float(item) for item in calibration["up_world"]]
        vectors_valid = (
            math.dist(position, target) > 0 and math.sqrt(sum(item * item for item in up)) > 0
        )
    if isinstance(intrinsics, Mapping):
        width = _integer(intrinsics.get("width"))
        height = _integer(intrinsics.get("height"))
        numeric = {key: _number(intrinsics.get(key)) for key in ("fx", "fy", "cx", "cy")}
        vertical_fov = _number(intrinsics.get("vertical_fov_deg"))
    else:
        width = height = None
        numeric = {}
        vertical_fov = None
    valid = (
        vectors_valid
        and _string(calibration.get("pose_convention"))
        and isinstance(intrinsics, Mapping)
        and _string(intrinsics.get("model"))
        and width == expected_width
        and height == expected_height
        and numeric.get("fx") is not None
        and numeric["fx"] > 0
        and numeric.get("fy") is not None
        and numeric["fy"] > 0
        and numeric.get("cx") is not None
        and 0 <= numeric["cx"] <= expected_width
        and numeric.get("cy") is not None
        and 0 <= numeric["cy"] <= expected_height
        and vertical_fov is not None
        and 0 < vertical_fov < 180
    )
    if not valid:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_camera_calibration_invalid"])
    normalized = json.loads(json.dumps(calibration, allow_nan=False))
    return {
        "calibration": normalized,
        "calibration_digest": canonical_digest(normalized),
    }


def _decode_png_rgb(
    data: bytes, *, expected_width: int, expected_height: int, max_pixels: int
) -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("not a single-frame PNG")
                width, height = image.size
                if (
                    width != expected_width
                    or height != expected_height
                    or width <= 0
                    or height <= 0
                    or width * height > max_pixels
                ):
                    raise ValueError("PNG dimensions outside bound")
                rgb = image.convert("RGB")
                rgb.load()
                rgb_bytes = rgb.tobytes()
                luminance_histogram = rgb.convert("L").histogram()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_png_invalid"]) from exc
    pixel_count = sum(luminance_histogram)
    dominant_count = max(luminance_histogram)
    differing_fraction = (pixel_count - dominant_count) / pixel_count
    entropy_bits = -sum(
        (count / pixel_count) * math.log2(count / pixel_count)
        for count in luminance_histogram
        if count
    )
    luminance_mean = (
        sum(value * count for value, count in enumerate(luminance_histogram)) / pixel_count
    )
    luminance_variance = max(
        0.0,
        sum(
            ((value - luminance_mean) ** 2) * count
            for value, count in enumerate(luminance_histogram)
        )
        / pixel_count,
    )

    def histogram_quantile(fraction: float) -> int:
        rank = math.floor(fraction * (pixel_count - 1))
        cumulative = 0
        for value, count in enumerate(luminance_histogram):
            cumulative += count
            if cumulative > rank:
                return value
        raise AssertionError("nonempty luminance histogram has no quantile")

    luminance_p05 = histogram_quantile(0.05)
    luminance_p95 = histogram_quantile(0.95)
    luminance_spread = luminance_p95 - luminance_p05
    value_range = max(rgb_bytes) - min(rgb_bytes)
    nontrivial = (
        differing_fraction >= FROZEN_MINIMUM_DIFFERING_LUMINANCE_FRACTION
        and entropy_bits >= FROZEN_MINIMUM_LUMINANCE_ENTROPY_BITS
        and luminance_spread >= FROZEN_MINIMUM_LUMINANCE_P05_P95_SPREAD
    )
    return {
        "decoded_mode": "RGB",
        "decoded_width": width,
        "decoded_height": height,
        "decoded_rgb_sha256": f"sha256:{hashlib.sha256(rgb_bytes).hexdigest()}",
        "decoded_rgb_has_multiple_colors": differing_fraction > 0.0,
        "decoded_rgb_value_range": value_range,
        "decoded_rgb_nontrivial": nontrivial,
        "visual_content_metrics": {
            "pixel_count": pixel_count,
            "dominant_luminance_count": dominant_count,
            "dominant_luminance_fraction": round(dominant_count / pixel_count, 12),
            "differing_luminance_fraction": round(differing_fraction, 12),
            "luminance_entropy_bits": round(entropy_bits, 12),
            "luminance_mean": round(luminance_mean, 12),
            "luminance_standard_deviation": round(math.sqrt(luminance_variance), 12),
            "luminance_p05": luminance_p05,
            "luminance_p95": luminance_p95,
            "luminance_p05_p95_spread": luminance_spread,
        },
        "visual_content_thresholds": {
            "minimum_differing_luminance_fraction": (FROZEN_MINIMUM_DIFFERING_LUMINANCE_FRACTION),
            "minimum_luminance_entropy_bits": FROZEN_MINIMUM_LUMINANCE_ENTROPY_BITS,
            "minimum_luminance_p05_p95_spread": (FROZEN_MINIMUM_LUMINANCE_P05_P95_SPREAD),
        },
    }


def verify_engineered_receptacle_visual_basis(
    *,
    visual_review_receipt_path: str | os.PathLike[str],
    render_manifest_path: str | os.PathLike[str],
    frame_root: str | os.PathLike[str],
    semantic_review_attestation_path: str | os.PathLike[str],
    semantic_authority_selection_path: str | os.PathLike[str],
    current_topology_receipt_digest: str,
    source_instance_id: str,
    limits: VisualBasisLimits = VisualBasisLimits(),
) -> dict[str, Any]:
    """Verify exact visual evidence for an independently authored empty twin.

    Only paths are accepted for evidence artifacts.  Caller-supplied mappings
    cannot cross this boundary because they have no retained byte identity.
    """

    limits.validate()
    topology_digest = _digest(current_topology_receipt_digest)
    instance_id = _string(source_instance_id)
    if not topology_digest:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_topology_digest_invalid"])
    if not instance_id:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_source_instance_id_invalid"])

    try:
        semantic_authority = verify_semantic_review_attestation(
            attestation_path=semantic_review_attestation_path,
            selection_contract_path=semantic_authority_selection_path,
        )
    except SemanticReviewAttestationError as exc:
        raise EngineeredReceptacleVisualBasisError(
            [f"visual_basis_semantic_authority_unqualified:{error}" for error in exc.errors]
        ) from exc

    review_snapshot = _read_regular_file_once(
        visual_review_receipt_path,
        max_bytes=limits.max_json_bytes,
        error_prefix="visual_basis_review",
    )
    manifest_snapshot = _read_regular_file_once(
        render_manifest_path,
        max_bytes=limits.max_json_bytes,
        error_prefix="visual_basis_manifest",
    )
    review = _strict_json(review_snapshot, error_prefix="visual_basis_review")
    manifest = _strict_json(manifest_snapshot, error_prefix="visual_basis_manifest")

    review_digest = _digest(review.get("review_digest"))
    if (
        review.get("schema_version") not in SUPPORTED_VISUAL_REVIEW_SCHEMAS
        or not review_digest
        or review_digest != canonical_digest(review, digest_field="review_digest")
    ):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_review_digest_invalid"])
    if review.get("collision_topology_receipt_digest") != topology_digest:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_topology_join_invalid"])
    if (
        review.get("reconnaissance_only") is not True
        or review.get("learned_policy_outcomes_inspected") is not False
    ):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_review_scope_invalid"])

    manifest_digest = _digest(manifest.get("render_manifest_digest"))
    if (
        manifest.get("schema_version") not in SUPPORTED_RENDER_MANIFEST_SCHEMAS
        or not manifest_digest
        or manifest_digest != canonical_digest(manifest, digest_field="render_manifest_digest")
        or review.get("render_manifest_digest") != manifest_digest
    ):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_manifest_digest_invalid"])
    renderer = _validate_renderer(manifest)
    renderer_width = int(renderer["renderer_identity"]["width"])
    renderer_height = int(renderer["renderer_identity"]["height"])
    if renderer_width * renderer_height > limits.max_frame_pixels:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_renderer_invalid"])

    cameras = _index_rows(
        manifest.get("cameras"), id_key="id", error="visual_basis_cameras_invalid"
    )
    calibrations = _index_rows(
        manifest.get("camera_calibration"),
        id_key="id",
        error="visual_basis_camera_calibrations_invalid",
    )
    targets = review.get("targets")
    if not isinstance(targets, list) or not targets:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_targets_invalid"])

    source_targets = [
        row
        for row in targets
        if isinstance(row, Mapping)
        and row.get("publisher_instance_id") == instance_id
        and row.get("target_kind") == "destination_receptacle"
    ]
    if len(source_targets) != 1:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_source_target_invalid"])
    source_target = source_targets[0]
    source_bools = {
        key: source_target.get(key)
        for key in (
            "rigid_exterior_observed",
            "open_rim_observed",
            "interior_occupied",
            "complete_interior_appearance_observed",
            "source_destination_admitted",
            "engineered_twin_design_basis_admitted",
        )
    }
    if any(not isinstance(value, bool) for value in source_bools.values()):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_source_observation_invalid"])
    claim_boundary = review.get("claim_boundary")
    source_valid = (
        source_bools["rigid_exterior_observed"]
        and source_bools["open_rim_observed"]
        and not source_bools["complete_interior_appearance_observed"]
        and not source_bools["source_destination_admitted"]
        and source_bools["engineered_twin_design_basis_admitted"]
        and source_target.get("selection_role") == "engineered_twin_design_basis"
        and review.get("selected_destination_design_basis_instance_id") == instance_id
        and review.get("source_destination_is_occupied") is source_bools["interior_occupied"]
        and review.get("source_destination_complete_interior_appearance_observed")
        is source_bools["complete_interior_appearance_observed"]
        and review.get("composition_required") is True
        and isinstance(claim_boundary, Mapping)
        and claim_boundary.get("collision_cavity_establishes_hidden_appearance") is False
        and claim_boundary.get("engineered_twin_hidden_geometry_is_source_truth") is False
        and claim_boundary.get("review_is_evaluation_policy_media") is False
        and claim_boundary.get("physical_material_equivalence_proven") is False
    )
    if not source_valid:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_source_observation_invalid"])

    cited: list[tuple[str, str, Mapping[str, Any]]] = []
    target_ids: set[str] = set()
    camera_ids: set[str] = set()
    for target in targets:
        target_id = _string(target.get("target_id")) if isinstance(target, Mapping) else ""
        frames = target.get("cited_frames") if isinstance(target, Mapping) else None
        if not target_id or target_id in target_ids or not isinstance(frames, list) or not frames:
            raise EngineeredReceptacleVisualBasisError(["visual_basis_cited_frames_invalid"])
        target_ids.add(target_id)
        for citation in frames:
            camera_id = _string(citation.get("camera_id")) if isinstance(citation, Mapping) else ""
            if not camera_id or camera_id in camera_ids:
                raise EngineeredReceptacleVisualBasisError(["visual_basis_cited_frames_invalid"])
            camera_ids.add(camera_id)
            cited.append((target_id, camera_id, citation))
    if len(cited) > limits.max_cited_frames:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_cited_frame_limit_exceeded"])

    frame_bindings: list[dict[str, Any]] = []
    frame_snapshots: dict[str, _FileSnapshot] = {}
    for target_id, camera_id, citation in cited:
        camera = cameras.get(camera_id)
        calibration = calibrations.get(camera_id)
        if camera is None or calibration is None or camera.get("nonblank") is not True:
            raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_manifest_join_invalid"])
        camera_size = _integer(camera.get("bytes"))
        camera_digest = _digest(camera.get("digest"))
        manifest_frame_path = _string(camera.get("path"))
        if (
            camera_size is None
            or camera_size <= 0
            or not camera_digest
            or not manifest_frame_path
            or citation.get("size_bytes") != camera_size
            or citation.get("sha256") != camera_digest
        ):
            raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_manifest_join_invalid"])
        normalized_frame_path, _parts = _lexical_frame_path(frame_root, manifest_frame_path)
        frame_snapshot = frame_snapshots.get(normalized_frame_path)
        if frame_snapshot is None:
            frame_snapshot = _read_frame_once(
                frame_root,
                manifest_frame_path,
                max_bytes=limits.max_frame_bytes,
            )
            frame_snapshots[normalized_frame_path] = frame_snapshot
        if frame_snapshot.size_bytes != camera_size or frame_snapshot.sha256 != camera_digest:
            raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_bytes_join_invalid"])
        calibration_binding = _validate_calibration(
            calibration,
            expected_width=renderer_width,
            expected_height=renderer_height,
        )
        decoded = _decode_png_rgb(
            frame_snapshot.data,
            expected_width=renderer_width,
            expected_height=renderer_height,
            max_pixels=limits.max_frame_pixels,
        )
        if decoded["decoded_rgb_nontrivial"] is not True:
            raise EngineeredReceptacleVisualBasisError(["visual_basis_frame_visually_trivial"])
        frame_bindings.append(
            {
                "target_id": target_id,
                "camera_id": camera_id,
                "file": frame_snapshot.binding(),
                **decoded,
                **calibration_binding,
            }
        )

    frame_bindings.sort(key=lambda row: (row["target_id"], row["camera_id"]))
    exact_frame_evidence = [
        {
            "target_id": row["target_id"],
            "camera_id": row["camera_id"],
            "sha256": row["file"]["sha256"],
            "size_bytes": row["file"]["size_bytes"],
            "decoded_rgb_sha256": row["decoded_rgb_sha256"],
        }
        for row in frame_bindings
    ]
    cited_frames_digest = semantic_frame_evidence_digest(exact_frame_evidence)
    semantic_assertions = {
        "rigid_exterior_observed": source_bools["rigid_exterior_observed"],
        "open_rim_observed": source_bools["open_rim_observed"],
        "interior_occupied": source_bools["interior_occupied"],
        "complete_interior_appearance_observed": source_bools[
            "complete_interior_appearance_observed"
        ],
        "source_destination_admitted": source_bools["source_destination_admitted"],
        "engineered_twin_design_basis_admitted": source_bools[
            "engineered_twin_design_basis_admitted"
        ],
        "selection_role": source_target.get("selection_role"),
    }
    expected_semantic_evidence = {
        "visual_review_digest": review_digest,
        "render_manifest_digest": manifest_digest,
        "collision_topology_receipt_digest": topology_digest,
        "cited_frames_digest": cited_frames_digest,
    }
    expected_source_target = {
        "target_id": source_target.get("target_id"),
        "source_instance_id": instance_id,
        "semantic_role": source_target.get("target_kind"),
    }
    if (
        semantic_authority.get("scene_id") != review.get("scene_id")
        or semantic_authority.get("source_target") != expected_source_target
        or semantic_authority.get("evidence") != expected_semantic_evidence
        or semantic_authority.get("learned_policy_outcomes_inspected") is not False
        or semantic_authority.get("semantic_assertions") != semantic_assertions
    ):
        raise EngineeredReceptacleVisualBasisError(["visual_basis_semantic_authority_join_invalid"])
    occupied = bool(source_bools["interior_occupied"])
    observation_state = "occupied_and_incomplete" if occupied else "unoccupied_but_incomplete"
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_visual_design_basis",
        "scene_id": _string(review.get("scene_id")),
        "source_receptacle_instance_id": instance_id,
        "current_topology_receipt_digest": topology_digest,
        "visual_review_receipt": {
            "schema_version": review["schema_version"],
            "review_digest": review_digest,
            "file": review_snapshot.binding(),
        },
        "render_manifest": {
            "schema_version": _string(manifest.get("schema_version")),
            "render_manifest_digest": manifest_digest,
            "file": manifest_snapshot.binding(),
            **renderer,
        },
        "cited_frame_count": len(frame_bindings),
        "cited_frames_digest": cited_frames_digest,
        "cited_frames": frame_bindings,
        "semantic_authority": semantic_authority,
        "source_receptacle_observation": {
            "observation_state": observation_state,
            "rigid_exterior_observed": True,
            "open_rim_observed": True,
            "interior_occupied": occupied,
            "complete_interior_appearance_observed": False,
            "empty_source_interior_observed": False,
            "source_destination_admitted": False,
        },
        "engineered_twin_design_basis": {
            "visual_design_basis_verified": True,
            "authored_empty_interior_is_source_observation": False,
            "authored_empty_interior_native_qualified": False,
            "independent_asset_provenance_required": True,
            "native_simulator_qualification_required": True,
        },
        "claim_boundary": {
            "reconnaissance_render_is_source_capture": False,
            "source_empty_interior_observed": False,
            "authored_twin_hidden_geometry_is_source_truth": False,
            "native_collision_qualified": False,
            "physical_equivalence_proven": False,
            "evaluation_policy_media": False,
            "signed_semantic_authority_is_native_qualification": False,
        },
        "basis_digest": "",
    }
    if not result["scene_id"] or not result["render_manifest"]["schema_version"]:
        raise EngineeredReceptacleVisualBasisError(["visual_basis_identity_invalid"])
    result["basis_digest"] = canonical_digest(result, digest_field="basis_digest")
    return result


__all__ = [
    "EngineeredReceptacleVisualBasisError",
    "FROZEN_MINIMUM_DIFFERING_LUMINANCE_FRACTION",
    "FROZEN_MINIMUM_LUMINANCE_ENTROPY_BITS",
    "FROZEN_MINIMUM_LUMINANCE_P05_P95_SPREAD",
    "SCHEMA_VERSION",
    "SUPPORTED_RENDER_MANIFEST_SCHEMAS",
    "SUPPORTED_VISUAL_REVIEW_SCHEMAS",
    "VisualBasisLimits",
    "verify_engineered_receptacle_visual_basis",
]
