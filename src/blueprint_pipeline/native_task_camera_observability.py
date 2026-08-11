"""Measure task-entity visibility and framing from native semantic pixels.

The original public function remains the compatibility boundary for the rigid
and articulated fixtures that tagged one ``class=task_object`` target.  New
multi-entity tasks use the additive entity-keyed API so repeated semantic roles
cannot collapse the movable and destination into one class-level count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "native_task_camera_observability.v1"
ENTITY_SCHEMA_VERSION = "native_task_camera_entity_observability.v1"


class NativeTaskCameraObservabilityError(ValueError):
    """Stable semantic/framing failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _semantic_plane(semantic_ids: Any) -> Any:
    import numpy as np

    semantic = np.asarray(semantic_ids)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if (
        semantic.ndim != 2
        or not semantic.size
        or np.issubdtype(semantic.dtype, np.bool_)
        or not np.issubdtype(semantic.dtype, np.integer)
    ):
        raise NativeTaskCameraObservabilityError(["native_task_camera_semantic_shape_invalid"])
    return semantic.astype(np.int64, copy=False)


def _semantic_identifier(value: Any) -> int:
    if isinstance(value, bool):
        raise NativeTaskCameraObservabilityError(["native_task_camera_semantic_identifier_invalid"])
    if isinstance(value, int):
        identifier = value
    elif isinstance(value, str) and value and value == value.strip():
        try:
            identifier = int(value)
        except ValueError as exc:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_invalid"]
            ) from exc
        if str(identifier) != value:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_invalid"]
            )
    else:
        raise NativeTaskCameraObservabilityError(["native_task_camera_semantic_identifier_invalid"])
    if identifier < 0:
        raise NativeTaskCameraObservabilityError(["native_task_camera_semantic_identifier_invalid"])
    return identifier


def _thresholds(
    *,
    minimum_pixels: Any,
    minimum_pixel_fraction: Any,
    centroid_margin_fraction: Any,
) -> tuple[int, float, float]:
    if (
        isinstance(minimum_pixels, bool)
        or not isinstance(minimum_pixels, int)
        or minimum_pixels < 1
        or isinstance(minimum_pixel_fraction, bool)
        or not isinstance(minimum_pixel_fraction, (int, float))
        or not math.isfinite(float(minimum_pixel_fraction))
        or not 0.0 < float(minimum_pixel_fraction) <= 1.0
        or isinstance(centroid_margin_fraction, bool)
        or not isinstance(centroid_margin_fraction, (int, float))
        or not math.isfinite(float(centroid_margin_fraction))
        or not 0.0 <= float(centroid_margin_fraction) < 0.5
    ):
        raise NativeTaskCameraObservabilityError(["native_task_camera_threshold_invalid"])
    return (
        minimum_pixels,
        float(minimum_pixel_fraction),
        float(centroid_margin_fraction),
    )


def _mask_metrics(
    *,
    mask: Any,
    minimum_pixels: int,
    minimum_pixel_fraction: float,
    centroid_margin_fraction: float,
) -> dict[str, Any]:
    import numpy as np

    count = int(mask.sum())
    height, width = (int(value) for value in mask.shape)
    fraction = count / float(height * width)
    bbox: list[int] | None = None
    centroid: list[float] | None = None
    centroid_framed = False
    bbox_framed = False
    if count:
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [
            float(xs.mean() / max(1, width - 1)),
            float(ys.mean() / max(1, height - 1)),
        ]
        margin = float(centroid_margin_fraction)
        centroid_framed = all(margin <= value <= 1.0 - margin for value in centroid)
        bbox_framed = bool(
            bbox[0] >= margin * max(1, width - 1)
            and bbox[2] <= (1.0 - margin) * max(1, width - 1)
            and bbox[1] >= margin * max(1, height - 1)
            and bbox[3] <= (1.0 - margin) * max(1, height - 1)
        )
    return {
        "pixel_count": count,
        "pixel_fraction": fraction,
        "bbox_xyxy": bbox,
        "centroid_xy_fraction": centroid,
        "centroid_within_margin": centroid_framed,
        "bbox_within_margin": bbox_framed,
        "frame_resolution_hw": [height, width],
        "thresholds": {
            "minimum_pixels": minimum_pixels,
            "minimum_pixel_fraction": minimum_pixel_fraction,
            "centroid_margin_fraction": centroid_margin_fraction,
        },
        "passed": (
            count >= minimum_pixels
            and fraction >= minimum_pixel_fraction
            and centroid_framed
            and bbox_framed
        ),
    }


def measure_native_task_camera_entity_observability(
    *,
    semantic_ids: Any,
    id_to_labels: Mapping[str, Any],
    entity_requirements: Sequence[Mapping[str, Any]],
    centroid_margin_fraction: float = 0.05,
) -> dict[str, Any]:
    """Gate every required task entity by its native ``entity_id`` tag.

    Each requirement has exactly ``entity_id``, ``minimum_pixels``, and
    ``minimum_pixel_fraction``.  Semantic roles are intentionally not accepted:
    two obstacles or two movable objects may share a role while retaining
    independent identity and visibility evidence.
    """

    import numpy as np

    semantic = _semantic_plane(semantic_ids)
    if (
        isinstance(entity_requirements, (str, bytes, Mapping))
        or not isinstance(entity_requirements, Sequence)
        or not entity_requirements
    ):
        raise NativeTaskCameraObservabilityError(["native_task_camera_entity_requirements_invalid"])

    normalized: list[dict[str, Any]] = []
    entity_ids: set[str] = set()
    for index, raw in enumerate(entity_requirements):
        if not isinstance(raw, Mapping) or set(raw) != {
            "entity_id",
            "minimum_pixels",
            "minimum_pixel_fraction",
        }:
            raise NativeTaskCameraObservabilityError(
                [f"native_task_camera_entity_requirement_invalid:{index}"]
            )
        entity_id = raw.get("entity_id")
        if (
            not isinstance(entity_id, str)
            or not entity_id.strip()
            or entity_id != entity_id.strip()
            or entity_id in entity_ids
        ):
            raise NativeTaskCameraObservabilityError(
                [f"native_task_camera_entity_identity_invalid:{index}"]
            )
        entity_ids.add(entity_id)
        minimum_pixels, minimum_fraction, margin = _thresholds(
            minimum_pixels=raw.get("minimum_pixels"),
            minimum_pixel_fraction=raw.get("minimum_pixel_fraction"),
            centroid_margin_fraction=centroid_margin_fraction,
        )
        normalized.append(
            {
                "entity_id": entity_id,
                "minimum_pixels": minimum_pixels,
                "minimum_pixel_fraction": minimum_fraction,
                "centroid_margin_fraction": margin,
            }
        )

    semantic_ids_by_entity = {entity_id: [] for entity_id in entity_ids}
    if not isinstance(id_to_labels, Mapping):
        raise NativeTaskCameraObservabilityError(["native_task_camera_semantic_labels_invalid"])
    canonical_semantic_ids: set[int] = set()
    for identifier, entry in id_to_labels.items():
        if not isinstance(entry, Mapping):
            continue
        tagged_entity_id = entry.get("entity_id")
        if tagged_entity_id is not None and not isinstance(tagged_entity_id, str):
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_entity_identity_invalid"]
            )
        if tagged_entity_id not in semantic_ids_by_entity:
            continue
        semantic_identifier = _semantic_identifier(identifier)
        if semantic_identifier in canonical_semantic_ids:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_alias"]
            )
        canonical_semantic_ids.add(semantic_identifier)
        semantic_ids_by_entity[tagged_entity_id].append(semantic_identifier)

    rows: list[dict[str, Any]] = []
    union_mask = np.zeros(semantic.shape, dtype=bool)
    for requirement in sorted(normalized, key=lambda row: row["entity_id"]):
        entity_id = requirement["entity_id"]
        native_ids = sorted(set(semantic_ids_by_entity[entity_id]))
        mask = np.isin(semantic, native_ids)
        union_mask |= mask
        rows.append(
            {
                "entity_id": entity_id,
                "target_semantic_ids": native_ids,
                **_mask_metrics(
                    mask=mask,
                    minimum_pixels=requirement["minimum_pixels"],
                    minimum_pixel_fraction=requirement["minimum_pixel_fraction"],
                    centroid_margin_fraction=requirement["centroid_margin_fraction"],
                ),
            }
        )

    union = _mask_metrics(
        mask=union_mask,
        minimum_pixels=1,
        minimum_pixel_fraction=0.0,
        centroid_margin_fraction=0.0,
    )
    passed = all(row["passed"] for row in rows)
    return {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "entity_observability": rows,
        "required_entity_ids": sorted(entity_ids),
        "all_entities_passed": passed,
        "pixel_count": union["pixel_count"],
        "pixel_fraction": union["pixel_fraction"],
        "bbox_xyxy": union["bbox_xyxy"],
        "frame_resolution_hw": union["frame_resolution_hw"],
        "passed": passed,
        "measurement_authority": "native_semantic_segmentation_aov_entity_id",
        "rgb_or_model_label_used": False,
        "semantic_roles_used_as_entity_identity": False,
    }


def measure_native_task_camera_observability(
    *,
    semantic_ids: Any,
    id_to_labels: Mapping[str, Any],
    target_label: str = "task_object",
    minimum_pixels: int,
    minimum_pixel_fraction: float,
    centroid_margin_fraction: float = 0.05,
) -> dict[str, Any]:
    """Gate exact target-class pixels without consulting rendered RGB semantics."""

    import numpy as np

    semantic = _semantic_plane(semantic_ids)
    minimum_pixels, minimum_pixel_fraction, centroid_margin_fraction = _thresholds(
        minimum_pixels=minimum_pixels,
        minimum_pixel_fraction=minimum_pixel_fraction,
        centroid_margin_fraction=centroid_margin_fraction,
    )
    target_ids: list[int] = []
    for identifier, entry in id_to_labels.items():
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label != target_label:
            continue
        semantic_identifier = _semantic_identifier(identifier)
        if semantic_identifier in target_ids:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_alias"]
            )
        target_ids.append(semantic_identifier)
    mask = np.isin(semantic, target_ids)
    metrics = _mask_metrics(
        mask=mask,
        minimum_pixels=minimum_pixels,
        minimum_pixel_fraction=minimum_pixel_fraction,
        centroid_margin_fraction=centroid_margin_fraction,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "target_label": target_label,
        "target_semantic_ids": target_ids,
        **metrics,
        "measurement_authority": "native_semantic_segmentation_aov",
        "rgb_or_model_label_used": False,
    }


__all__ = [
    "ENTITY_SCHEMA_VERSION",
    "NativeTaskCameraObservabilityError",
    "SCHEMA_VERSION",
    "measure_native_task_camera_entity_observability",
    "measure_native_task_camera_observability",
]
