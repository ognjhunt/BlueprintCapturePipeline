"""Object-aware partitioning for captured 3D Gaussian splats.

This module turns a *candidate* Gaussian-level object selection into two exact,
digest-bound standard 3DGS assets:

* a static background with the selected primitives removed; and
* an object-local splat that can be attached to a simulator body transform.

The selection can come from Blueprint's contribution-weighted semantic lifting
contract or from the local multiview-mask projection baseline implemented here.
Neither path upgrades semantic completeness, geometry, collision, or physical
truth.  The mechanical partition is exact; whether the selected primitives are
the whole real object remains a separately qualified perception question.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)
from .scene_placement.semantic_gaussian_lifting import (
    RESULT_SCHEMA_VERSION as SEMANTIC_LIFTING_RESULT_SCHEMA_VERSION,
)
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


GAUSSIAN_OBJECT_SELECTION_SCHEMA_VERSION = "gaussian_object_selection.v1"
GAUSSIAN_OBJECT_PARTITION_SCHEMA_VERSION = "gaussian_object_partition.v1"
MULTIVIEW_MASK_SELECTION_METHOD_ID = "blueprint.multiview_mask_center_visibility"
MULTIVIEW_MASK_SELECTION_METHOD_VERSION = "1.1.0"
WORLD_ALIGNED_OBJECT_FRAME = "world_aligned_translation_only_at_extraction.v1"


class GaussianObjectPartitionError(ValueError):
    def __init__(self, *codes: str) -> None:
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = {key: item for key, item in value.items() if key != field}
    return canonical_json_digest(normalized)


def _same_digest(left: Any, right: Any) -> bool:
    def normalized(value: Any) -> str:
        text = str(value or "").strip().lower()
        return text[7:] if text.startswith("sha256:") else text

    return normalized(left) == normalized(right)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise GaussianObjectPartitionError("gaussian_object_artifact_not_json") from exc
    if not isinstance(result, dict):
        raise GaussianObjectPartitionError("gaussian_object_artifact_not_object")
    return result


def _validate_transform(value: Any, *, code: str) -> list[list[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 4
            for row in value
        )
    ):
        raise GaussianObjectPartitionError(f"{code}_shape_invalid")
    matrix = np.asarray(value, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise GaussianObjectPartitionError(f"{code}_nonfinite")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-9):
        raise GaussianObjectPartitionError(f"{code}_bottom_row_invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6):
        raise GaussianObjectPartitionError(f"{code}_rotation_not_orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-6):
        raise GaussianObjectPartitionError(f"{code}_rotation_not_proper")
    return [[float(component) for component in row] for row in matrix]


def _validate_intrinsics(value: Any, *, view_id: str) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise GaussianObjectPartitionError(f"mask_view_intrinsics_missing:{view_id}")
    numeric: dict[str, float | int] = {}
    for key in ("fx", "fy", "cx", "cy"):
        number = _finite(value.get(key))
        if number is None or (key in {"fx", "fy"} and number <= 0.0):
            raise GaussianObjectPartitionError(
                f"mask_view_intrinsics_invalid:{view_id}:{key}"
            )
        numeric[key] = number
    for key in ("width", "height"):
        number = value.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise GaussianObjectPartitionError(
                f"mask_view_intrinsics_invalid:{view_id}:{key}"
            )
        numeric[key] = number
    return numeric


def _selection_digest(selected: Sequence[int]) -> str:
    return canonical_json_digest([int(index) for index in selected])


def validate_gaussian_object_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    selection = _clone(value)
    errors: list[str] = []
    if selection.get("schema_version") != GAUSSIAN_OBJECT_SELECTION_SCHEMA_VERSION:
        errors.append("gaussian_object_selection_schema_invalid")
    for key in ("selection_id", "object_id", "source_splat_digest"):
        if not str(selection.get(key) or "").strip():
            errors.append(f"gaussian_object_selection_{key}_missing")
    count = selection.get("source_gaussian_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 1:
        errors.append("gaussian_object_selection_source_count_invalid")
        count = 0
    selected = selection.get("selected_gaussian_ids")
    if not isinstance(selected, list) or not selected:
        errors.append("gaussian_object_selection_ids_missing")
        selected = []
    elif any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= count
        for index in selected
    ):
        errors.append("gaussian_object_selection_id_out_of_bounds")
    elif selected != sorted(set(selected)):
        errors.append("gaussian_object_selection_ids_not_unique_sorted")
    method = selection.get("method")
    if not isinstance(method, Mapping):
        errors.append("gaussian_object_selection_method_missing")
    else:
        for key in ("method_id", "method_version", "method_output_digest"):
            if not str(method.get(key) or "").strip():
                errors.append(f"gaussian_object_selection_method_{key}_missing")
    if selection.get("semantic_completeness_validated") is not False:
        errors.append("gaussian_object_selection_semantic_completeness_must_be_false")
    if selection.get("physics_authority_granted") is not False:
        errors.append("gaussian_object_selection_physics_authority_must_be_false")
    if selected and selection.get("selected_gaussian_ids_digest") != _selection_digest(selected):
        errors.append("gaussian_object_selection_ids_digest_mismatch")
    expected = _digest(selection, "gaussian_object_selection_digest")
    supplied = selection.get("gaussian_object_selection_digest")
    if supplied is not None and supplied != expected:
        errors.append("gaussian_object_selection_digest_mismatch")
    if errors:
        raise GaussianObjectPartitionError(*errors)
    selection["gaussian_object_selection_digest"] = expected
    return selection


def selection_from_semantic_lifting(
    lifting_result_value: Mapping[str, Any],
    *,
    gaussian_mapping: Sequence[Mapping[str, Any]],
    source_splat_path: str | Path,
    track_id: str,
) -> dict[str, Any]:
    """Adapt Blueprint semantic lifting output into exact source-Ply row IDs."""

    lifting = _clone(lifting_result_value)
    errors: list[str] = []
    if lifting.get("schema_version") != SEMANTIC_LIFTING_RESULT_SCHEMA_VERSION:
        errors.append("semantic_lifting_result_schema_invalid")
    supplied_result_digest = lifting.get("result_digest")
    expected_result_digest = _digest(lifting, "result_digest")
    if supplied_result_digest != expected_result_digest:
        errors.append("semantic_lifting_result_digest_mismatch")
    source_path = Path(source_splat_path)
    if source_path.is_symlink() or not source_path.is_file():
        errors.append("gaussian_object_source_splat_missing_or_symlink")
        source = None
    else:
        try:
            source = read_standard_3dgs_ply(source_path)
        except ValueError as exc:
            raise GaussianObjectPartitionError(
                "gaussian_object_source_not_standard_3dgs_ply"
            ) from exc
    mapping_digest = canonical_json_digest(list(gaussian_mapping))
    bindings = lifting.get("bindings")
    if not isinstance(bindings, Mapping):
        errors.append("semantic_lifting_bindings_missing")
    else:
        if bindings.get("gaussian_mapping_digest") != mapping_digest:
            errors.append("semantic_lifting_mapping_digest_mismatch")
        if source is not None and not _same_digest(
            bindings.get("analysis_splat_digest"), _sha256_file(source_path)
        ):
            errors.append("semantic_lifting_source_splat_digest_mismatch")
    tracks = lifting.get("tracks")
    matches = (
        [row for row in tracks if isinstance(row, Mapping) and row.get("track_id") == track_id]
        if isinstance(tracks, list)
        else []
    )
    if len(matches) != 1:
        errors.append("semantic_lifting_track_not_exactly_one")
        track: Mapping[str, Any] = {}
    else:
        track = matches[0]
        if track.get("status") != "qualified_semantic_support_candidate":
            errors.append("semantic_lifting_track_not_candidate")

    source_index_by_id: dict[int, int] = {}
    for row in gaussian_mapping:
        if not isinstance(row, Mapping):
            errors.append("semantic_lifting_mapping_row_invalid")
            continue
        gaussian_id = row.get("gaussian_id")
        source_index = row.get("source_index")
        if (
            not isinstance(gaussian_id, int)
            or isinstance(gaussian_id, bool)
            or not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or gaussian_id in source_index_by_id
        ):
            errors.append("semantic_lifting_mapping_id_or_index_invalid")
            continue
        source_index_by_id[gaussian_id] = source_index
    selected_ids = track.get("selected_gaussian_ids")
    selected_source_indices: list[int] = []
    if not isinstance(selected_ids, list) or not selected_ids:
        errors.append("semantic_lifting_selected_ids_missing")
    else:
        for gaussian_id in selected_ids:
            if gaussian_id not in source_index_by_id:
                errors.append("semantic_lifting_selected_id_unmapped")
                continue
            selected_source_indices.append(source_index_by_id[gaussian_id])
    if source is not None and any(
        index < 0 or index >= source.count for index in selected_source_indices
    ):
        errors.append("semantic_lifting_source_index_out_of_bounds")
    selected_source_indices = sorted(set(selected_source_indices))
    if not selected_source_indices:
        errors.append("semantic_lifting_selected_source_indices_empty")
    if errors:
        raise GaussianObjectPartitionError(*errors)

    source_digest = _sha256_file(source_path)
    selection = {
        "schema_version": GAUSSIAN_OBJECT_SELECTION_SCHEMA_VERSION,
        "selection_id": f"selection-{track_id}-{expected_result_digest[-12:]}",
        "object_id": track_id,
        "source_splat_digest": source_digest,
        "source_gaussian_count": source.count,
        "selected_gaussian_ids": selected_source_indices,
        "selected_gaussian_ids_digest": _selection_digest(selected_source_indices),
        "method": {
            "method_id": "blueprint.semantic_gaussian_lifting",
            "method_version": SEMANTIC_LIFTING_RESULT_SCHEMA_VERSION,
            "method_output_digest": expected_result_digest,
            "gaussian_mapping_digest": mapping_digest,
            "source_track_label": str(track.get("label") or ""),
        },
        "selection_evidence": {
            "supporting_view_ids": list(track.get("supporting_view_ids") or []),
            "angular_diversity_degrees": track.get("angular_diversity_degrees"),
            "selected_count": len(selected_source_indices),
        },
        "claim_ceiling": "candidate_object_gaussian_membership",
        "semantic_completeness_validated": False,
        "physics_authority_granted": False,
    }
    selection["gaussian_object_selection_digest"] = _digest(
        selection, "gaussian_object_selection_digest"
    )
    return validate_gaussian_object_selection(selection)


def select_gaussians_from_multiview_masks(
    source_splat_path: str | Path,
    *,
    object_id: str,
    views: Sequence[Mapping[str, Any]],
    min_positive_views: int = 2,
    foreground_probability_threshold: float = 0.75,
    depth_tolerance_m: float = 0.03,
) -> dict[str, Any]:
    """Select Gaussian centers using calibrated masks and a conservative z-buffer.

    This is an executable CPU baseline for already-produced capture masks. It is
    intentionally not called FlashSplat/SAGA/SAGO: those remain replaceable
    higher-quality method adapters. A center contributes only when it projects
    inside a calibrated image and lies within ``depth_tolerance_m`` of that
    pixel's nearest projected Gaussian footprint.  The footprint-aware depth
    test prevents Gaussians on a wall or counter behind the masked object from
    being copied into the movable asset merely because their centers fall in
    the same 2D mask.
    """

    if not str(object_id or "").strip():
        raise GaussianObjectPartitionError("multiview_mask_object_id_missing")
    if (
        not isinstance(min_positive_views, int)
        or isinstance(min_positive_views, bool)
        or min_positive_views < 2
    ):
        raise GaussianObjectPartitionError("multiview_mask_min_positive_views_invalid")
    if not 0.5 <= float(foreground_probability_threshold) <= 1.0:
        raise GaussianObjectPartitionError("multiview_mask_probability_threshold_invalid")
    if not 0.0 <= float(depth_tolerance_m) <= 0.5:
        raise GaussianObjectPartitionError("multiview_mask_depth_tolerance_invalid")
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)) or len(views) < 2:
        raise GaussianObjectPartitionError("multiview_mask_requires_two_views")

    source_path = Path(source_splat_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise GaussianObjectPartitionError("gaussian_object_source_splat_missing_or_symlink")
    try:
        splat = read_standard_3dgs_ply(source_path)
    except ValueError as exc:
        raise GaussianObjectPartitionError(
            "gaussian_object_source_not_standard_3dgs_ply"
        ) from exc

    positive = np.zeros(splat.count, dtype=np.int32)
    negative = np.zeros(splat.count, dtype=np.int32)
    view_receipts: list[dict[str, Any]] = []
    seen_view_ids: set[str] = set()
    for raw_view in views:
        if not isinstance(raw_view, Mapping):
            raise GaussianObjectPartitionError("multiview_mask_view_invalid")
        view_id = str(raw_view.get("view_id") or "").strip()
        if not view_id or view_id in seen_view_ids:
            raise GaussianObjectPartitionError("multiview_mask_view_id_missing_or_duplicate")
        seen_view_ids.add(view_id)
        intrinsics = _validate_intrinsics(raw_view.get("intrinsics"), view_id=view_id)
        transform = np.asarray(
            _validate_transform(
                raw_view.get("T_world_camera_opencv"),
                code=f"mask_view_transform:{view_id}",
            ),
            dtype=np.float64,
        )
        mask_path = Path(str(raw_view.get("mask_path") or ""))
        if mask_path.is_symlink() or not mask_path.is_file():
            raise GaussianObjectPartitionError(f"multiview_mask_missing:{view_id}")
        mask_digest = _sha256_file(mask_path)
        supplied_mask_digest = raw_view.get("mask_digest")
        if supplied_mask_digest is not None and supplied_mask_digest != mask_digest:
            raise GaussianObjectPartitionError(f"multiview_mask_digest_mismatch:{view_id}")
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        height = int(intrinsics["height"])
        width = int(intrinsics["width"])
        if mask.shape != (height, width):
            raise GaussianObjectPartitionError(f"multiview_mask_dimensions_mismatch:{view_id}")

        rotation_world_camera = transform[:3, :3]
        translation_world_camera = transform[:3, 3]
        camera_xyz = (splat.xyz.astype(np.float64) - translation_world_camera) @ (
            rotation_world_camera
        )
        z = camera_xyz[:, 2]
        valid_z = z > 1e-6
        u = np.rint(
            float(intrinsics["fx"]) * camera_xyz[:, 0] / np.maximum(z, 1e-12)
            + float(intrinsics["cx"])
        ).astype(np.int64)
        v = np.rint(
            float(intrinsics["fy"]) * camera_xyz[:, 1] / np.maximum(z, 1e-12)
            + float(intrinsics["cy"])
        ).astype(np.int64)
        inside = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(inside)
        if indices.size == 0:
            view_receipts.append(
                {
                    "view_id": view_id,
                    "mask_digest": mask_digest,
                    "visible_gaussian_count": 0,
                    "positive_gaussian_count": 0,
                }
            )
            continue
        pixels = v[indices] * width + u[indices]
        nearest = np.full(width * height, np.inf, dtype=np.float64)
        np.minimum.at(nearest, pixels, z[indices])
        # A center-only z-buffer leaves holes between primitive centers and can
        # classify background Gaussians through those holes. Approximate the
        # projected 3-sigma support of this view's splats, then erode the depth
        # image (a neighborhood minimum) before testing center visibility.
        world_sigma = np.exp(np.max(splat.scales[indices].astype(np.float64), axis=1))
        projected_sigma = (
            max(float(intrinsics["fx"]), float(intrinsics["fy"]))
            * world_sigma
            / z[indices]
        )
        footprint_radius = int(
            np.clip(np.ceil(3.0 * np.percentile(projected_sigma, 75.0)), 1, 64)
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * footprint_radius + 1, 2 * footprint_radius + 1),
        )
        footprint_nearest = cv2.erode(
            nearest.reshape(height, width),
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=float("inf"),
        ).reshape(-1)
        visible_local = (
            z[indices]
            <= footprint_nearest[pixels] + float(depth_tolerance_m)
        )
        visible_indices = indices[visible_local]
        is_positive = mask[v[visible_indices], u[visible_indices]] >= 128
        positive[visible_indices[is_positive]] += 1
        negative[visible_indices[~is_positive]] += 1
        view_receipts.append(
            {
                "view_id": view_id,
                "mask_digest": mask_digest,
                "camera_record_digest": canonical_json_digest(
                    {
                        "intrinsics": intrinsics,
                        "T_world_camera_opencv": transform.tolist(),
                    }
                ),
                "visible_gaussian_count": int(visible_indices.size),
                "positive_gaussian_count": int(np.count_nonzero(is_positive)),
                "visibility_footprint_radius_px": footprint_radius,
            }
        )

    observations = positive + negative
    probabilities = np.divide(
        positive,
        observations,
        out=np.zeros_like(positive, dtype=np.float64),
        where=observations > 0,
    )
    selected_array = np.flatnonzero(
        (positive >= min_positive_views)
        & (probabilities >= float(foreground_probability_threshold))
    )
    selected = [int(index) for index in selected_array]
    if not selected:
        raise GaussianObjectPartitionError("multiview_mask_selected_no_gaussians")
    method_payload = {
        "source_splat_digest": _sha256_file(source_path),
        "object_id": object_id,
        "views": view_receipts,
        "min_positive_views": min_positive_views,
        "foreground_probability_threshold": float(foreground_probability_threshold),
        "depth_tolerance_m": float(depth_tolerance_m),
        "selected_gaussian_ids_digest": _selection_digest(selected),
    }
    selection = {
        "schema_version": GAUSSIAN_OBJECT_SELECTION_SCHEMA_VERSION,
        "selection_id": f"selection-{object_id}-{canonical_json_digest(method_payload)[-12:]}",
        "object_id": object_id,
        "source_splat_digest": method_payload["source_splat_digest"],
        "source_gaussian_count": splat.count,
        "selected_gaussian_ids": selected,
        "selected_gaussian_ids_digest": _selection_digest(selected),
        "method": {
            "method_id": MULTIVIEW_MASK_SELECTION_METHOD_ID,
            "method_version": MULTIVIEW_MASK_SELECTION_METHOD_VERSION,
            "method_output_digest": canonical_json_digest(method_payload),
            "selection_semantics": "projected_center_visible_by_gaussian_footprint_depth",
        },
        "selection_evidence": {
            "views": view_receipts,
            "selected_count": len(selected),
            "minimum_positive_views": int(positive[selected_array].min()),
            "minimum_foreground_probability": float(probabilities[selected_array].min()),
        },
        "claim_ceiling": "candidate_object_gaussian_membership",
        "semantic_completeness_validated": False,
        "physics_authority_granted": False,
    }
    selection["gaussian_object_selection_digest"] = _digest(
        selection, "gaussian_object_selection_digest"
    )
    return validate_gaussian_object_selection(selection)


def _subset_splat(splat: SplatData, indices: np.ndarray) -> SplatData:
    return SplatData(
        count=int(indices.size),
        xyz=splat.xyz[indices].astype(np.float32, copy=True),
        opacity=splat.opacity[indices].astype(np.float32, copy=True),
        f_dc=splat.f_dc[indices].astype(np.float32, copy=True),
        scales=splat.scales[indices].astype(np.float32, copy=True),
        quats=splat.quats[indices].astype(np.float32, copy=True),
        properties=splat.properties,
        sh_rest=(
            splat.sh_rest[indices].astype(np.float32, copy=True)
            if splat.sh_rest is not None
            else None
        ),
    )


def validate_gaussian_object_partition(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _clone(value)
    errors: list[str] = []
    if manifest.get("schema_version") != GAUSSIAN_OBJECT_PARTITION_SCHEMA_VERSION:
        errors.append("gaussian_object_partition_schema_invalid")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("gaussian_object_partition_counts_missing")
    else:
        source = counts.get("source")
        background = counts.get("background")
        object_count = counts.get("object")
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in (source, background, object_count)
        ):
            errors.append("gaussian_object_partition_counts_invalid")
        elif source != background + object_count:
            errors.append("gaussian_object_partition_not_exhaustive")
    partition = manifest.get("partition")
    if not isinstance(partition, Mapping):
        errors.append("gaussian_object_partition_invariants_missing")
    else:
        for key in (
            "selected_ids_unique",
            "background_object_disjoint",
            "source_exhaustive",
            "background_excludes_selected_object",
        ):
            if partition.get(key) is not True:
                errors.append(f"gaussian_object_partition_invariant_false:{key}")
        selected = partition.get("selected_gaussian_ids")
        object_count = (manifest.get("counts") or {}).get("object")
        source_count = (manifest.get("counts") or {}).get("source")
        if (
            not isinstance(selected, list)
            or len(selected) != object_count
            or selected != sorted(set(selected))
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or not isinstance(source_count, int)
                or index >= source_count
                for index in selected
            )
        ):
            errors.append("gaussian_object_partition_selected_ids_invalid")
        elif partition.get("selected_gaussian_ids_digest") != _selection_digest(
            selected
        ):
            errors.append("gaussian_object_partition_selected_ids_digest_mismatch")
    frame = manifest.get("object_frame")
    if not isinstance(frame, Mapping) or frame.get("convention") != WORLD_ALIGNED_OBJECT_FRAME:
        errors.append("gaussian_object_partition_frame_invalid")
    else:
        try:
            matrix = _validate_transform(
                frame.get("T_world_object_at_extraction"),
                code="gaussian_object_partition_extraction_transform",
            )
        except GaussianObjectPartitionError as exc:
            errors.extend(exc.codes)
        else:
            if not np.allclose(np.asarray(matrix)[:3, :3], np.eye(3), rtol=0, atol=1e-9):
                errors.append("gaussian_object_partition_frame_must_be_world_aligned")
    if manifest.get("semantic_completeness_validated") is not False:
        errors.append("gaussian_object_partition_semantic_completeness_must_be_false")
    if manifest.get("physics_authority_granted") is not False:
        errors.append("gaussian_object_partition_physics_authority_must_be_false")
    expected = _digest(manifest, "gaussian_object_partition_digest")
    supplied = manifest.get("gaussian_object_partition_digest")
    if supplied is not None and supplied != expected:
        errors.append("gaussian_object_partition_digest_mismatch")
    if errors:
        raise GaussianObjectPartitionError(*errors)
    manifest["gaussian_object_partition_digest"] = expected
    return manifest


def partition_gaussian_object(
    source_splat_path: str | Path,
    selection_value: Mapping[str, Any],
    *,
    output_dir: str | Path,
    extraction_origin_world: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Write exhaustive background and object-local standard-Ply partitions."""

    source_path = Path(source_splat_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise GaussianObjectPartitionError("gaussian_object_source_splat_missing_or_symlink")
    selection = validate_gaussian_object_selection(selection_value)
    source_digest = _sha256_file(source_path)
    if selection["source_splat_digest"] != source_digest:
        raise GaussianObjectPartitionError("gaussian_object_partition_source_digest_mismatch")
    try:
        source = read_standard_3dgs_ply(source_path)
    except ValueError as exc:
        raise GaussianObjectPartitionError(
            "gaussian_object_source_not_standard_3dgs_ply"
        ) from exc
    if source.count != selection["source_gaussian_count"]:
        raise GaussianObjectPartitionError("gaussian_object_partition_source_count_mismatch")
    selected = np.asarray(selection["selected_gaussian_ids"], dtype=np.int64)
    if selected.size >= source.count:
        raise GaussianObjectPartitionError("gaussian_object_partition_background_would_be_empty")
    mask = np.zeros(source.count, dtype=bool)
    mask[selected] = True
    background_indices = np.flatnonzero(~mask)
    object_indices = np.flatnonzero(mask)
    background = _subset_splat(source, background_indices)
    object_splat = _subset_splat(source, object_indices)

    if extraction_origin_world is None:
        origin = object_splat.xyz.astype(np.float64).mean(axis=0)
    else:
        if (
            not isinstance(extraction_origin_world, Sequence)
            or isinstance(extraction_origin_world, (str, bytes))
            or len(extraction_origin_world) != 3
        ):
            raise GaussianObjectPartitionError("gaussian_object_extraction_origin_invalid")
        components = [_finite(item) for item in extraction_origin_world]
        if any(item is None for item in components):
            raise GaussianObjectPartitionError("gaussian_object_extraction_origin_nonfinite")
        origin = np.asarray(components, dtype=np.float64)
    object_splat.xyz = (object_splat.xyz.astype(np.float64) - origin).astype(np.float32)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    background_path = write_standard_3dgs_ply(background, destination / "background.ply")
    object_path = write_standard_3dgs_ply(
        object_splat, destination / f"object-{selection['object_id']}.ply"
    )
    extraction_transform = np.eye(4, dtype=np.float64)
    extraction_transform[:3, 3] = origin
    manifest = {
        "schema_version": GAUSSIAN_OBJECT_PARTITION_SCHEMA_VERSION,
        "partition_id": (
            f"partition-{selection['object_id']}-"
            f"{selection['gaussian_object_selection_digest'][-12:]}"
        ),
        "object_id": selection["object_id"],
        "source": {
            "path": str(source_path.resolve()),
            "digest": source_digest,
            "selection_digest": selection["gaussian_object_selection_digest"],
        },
        "artifacts": {
            "background": {
                "path": str(background_path.resolve()),
                "digest": _sha256_file(background_path),
                "kind": "static_background_object_excluded",
            },
            "object": {
                "path": str(object_path.resolve()),
                "digest": _sha256_file(object_path),
                "kind": "movable_object_gaussians_canonical_frame",
            },
        },
        "counts": {
            "source": source.count,
            "background": background.count,
            "object": object_splat.count,
        },
        "partition": {
            "selected_gaussian_ids": selection["selected_gaussian_ids"],
            "selected_gaussian_ids_digest": selection["selected_gaussian_ids_digest"],
            "selected_ids_unique": True,
            "background_object_disjoint": True,
            "source_exhaustive": True,
            "background_excludes_selected_object": True,
        },
        "object_frame": {
            "convention": WORLD_ALIGNED_OBJECT_FRAME,
            "T_world_object_at_extraction": extraction_transform.tolist(),
            "higher_order_spherical_harmonics_preserved": source.sh_rest is not None,
            "view_dependent_appearance_moves_in_object_frame": True,
        },
        "background_completion": {
            "status": "observed_background_only_no_generated_inpainting",
            "generated_gaussians_added": False,
            "vacated_region_visual_completeness_validated": False,
        },
        "claim_ceiling": "exact_mechanical_partition_candidate_semantics",
        "semantic_completeness_validated": False,
        "physics_authority_granted": False,
        "exact_partition_establishes_physical_object_identity": False,
    }
    manifest["gaussian_object_partition_digest"] = _digest(
        manifest, "gaussian_object_partition_digest"
    )
    return validate_gaussian_object_partition(manifest)


def verify_gaussian_object_partition_files(
    manifest_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the persisted partition files and row-count invariants."""

    manifest = validate_gaussian_object_partition(manifest_value)
    errors: list[str] = []
    loaded_counts: dict[str, int] = {}
    loaded: dict[str, SplatData] = {}
    for key in ("background", "object"):
        artifact = dict(manifest["artifacts"][key])
        path = Path(artifact["path"])
        if path.is_symlink() or not path.is_file():
            errors.append(f"gaussian_object_partition_artifact_missing:{key}")
            continue
        if _sha256_file(path) != artifact["digest"]:
            errors.append(f"gaussian_object_partition_artifact_digest_mismatch:{key}")
            continue
        try:
            loaded[key] = read_standard_3dgs_ply(path)
            loaded_counts[key] = loaded[key].count
        except ValueError:
            errors.append(f"gaussian_object_partition_artifact_invalid_ply:{key}")
    if loaded_counts.get("background") != manifest["counts"]["background"]:
        errors.append("gaussian_object_partition_background_count_mismatch")
    if loaded_counts.get("object") != manifest["counts"]["object"]:
        errors.append("gaussian_object_partition_object_count_mismatch")
    source_record = dict(manifest.get("source") or {})
    source_path = Path(str(source_record.get("path") or ""))
    source: SplatData | None = None
    if source_path.is_symlink() or not source_path.is_file():
        errors.append("gaussian_object_partition_source_missing")
    elif _sha256_file(source_path) != source_record.get("digest"):
        errors.append("gaussian_object_partition_source_digest_mismatch")
    else:
        try:
            source = read_standard_3dgs_ply(source_path)
        except ValueError:
            errors.append("gaussian_object_partition_source_invalid_ply")
    if source is not None and source.count != manifest["counts"]["source"]:
        errors.append("gaussian_object_partition_source_count_mismatch")
    if source is not None and set(loaded) == {"background", "object"}:
        selected = np.asarray(
            manifest["partition"]["selected_gaussian_ids"], dtype=np.int64
        )
        selected_mask = np.zeros(source.count, dtype=bool)
        selected_mask[selected] = True
        expected_background = _subset_splat(source, np.flatnonzero(~selected_mask))
        expected_object = _subset_splat(source, np.flatnonzero(selected_mask))
        extraction_origin = np.asarray(
            manifest["object_frame"]["T_world_object_at_extraction"],
            dtype=np.float64,
        )[:3, 3]
        expected_object.xyz = (
            expected_object.xyz.astype(np.float64) - extraction_origin
        ).astype(np.float32)

        def same_splat(left: SplatData, right: SplatData) -> bool:
            arrays_equal = all(
                np.array_equal(getattr(left, name), getattr(right, name))
                for name in ("xyz", "opacity", "f_dc", "scales", "quats")
            )
            if not arrays_equal or left.properties != right.properties:
                return False
            if left.sh_rest is None or right.sh_rest is None:
                return left.sh_rest is None and right.sh_rest is None
            return np.array_equal(left.sh_rest, right.sh_rest)

        if not same_splat(loaded["background"], expected_background):
            errors.append("gaussian_object_partition_background_rows_mismatch_source")
        if not same_splat(loaded["object"], expected_object):
            errors.append("gaussian_object_partition_object_rows_mismatch_source")
    report = {
        "schema_version": "gaussian_object_partition_verification.v1",
        "gaussian_object_partition_digest": manifest[
            "gaussian_object_partition_digest"
        ],
        "status": "passed" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "loaded_counts": loaded_counts,
        "exact_mechanical_partition_verified": not errors,
        "semantic_completeness_verified": False,
        "physical_truth_verified": False,
    }
    report["verification_digest"] = _digest(report, "verification_digest")
    return report


__all__ = [
    "GAUSSIAN_OBJECT_PARTITION_SCHEMA_VERSION",
    "GAUSSIAN_OBJECT_SELECTION_SCHEMA_VERSION",
    "GaussianObjectPartitionError",
    "MULTIVIEW_MASK_SELECTION_METHOD_ID",
    "MULTIVIEW_MASK_SELECTION_METHOD_VERSION",
    "WORLD_ALIGNED_OBJECT_FRAME",
    "partition_gaussian_object",
    "select_gaussians_from_multiview_masks",
    "selection_from_semantic_lifting",
    "validate_gaussian_object_partition",
    "validate_gaussian_object_selection",
    "verify_gaussian_object_partition_files",
]
