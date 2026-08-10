"""Measure task-object visibility and framing from native semantic pixels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "native_task_camera_observability.v1"


class NativeTaskCameraObservabilityError(ValueError):
    """Stable semantic/framing failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


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

    semantic = np.asarray(semantic_ids)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if semantic.ndim != 2 or not semantic.size:
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_shape_invalid"]
        )
    if (
        isinstance(minimum_pixels, bool)
        or int(minimum_pixels) < 1
        or not math.isfinite(float(minimum_pixel_fraction))
        or float(minimum_pixel_fraction) <= 0.0
        or not math.isfinite(float(centroid_margin_fraction))
        or float(centroid_margin_fraction) < 0.0
        or float(centroid_margin_fraction) >= 0.5
    ):
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_threshold_invalid"]
        )
    target_ids: list[int] = []
    for identifier, entry in id_to_labels.items():
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label != target_label:
            continue
        try:
            target_ids.append(int(identifier))
        except (TypeError, ValueError) as exc:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_invalid"]
            ) from exc
    mask = np.isin(semantic.astype(np.int64), target_ids)
    count = int(mask.sum())
    height, width = (int(value) for value in mask.shape)
    fraction = count / float(height * width)
    bbox: list[int] | None = None
    centroid: list[float] | None = None
    centroid_framed = False
    if count:
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [
            float(xs.mean() / max(1, width - 1)),
            float(ys.mean() / max(1, height - 1)),
        ]
        margin = float(centroid_margin_fraction)
        centroid_framed = all(margin <= value <= 1.0 - margin for value in centroid)
    passed = (
        count >= int(minimum_pixels)
        and fraction >= float(minimum_pixel_fraction)
        and centroid_framed
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "target_label": target_label,
        "target_semantic_ids": target_ids,
        "pixel_count": count,
        "pixel_fraction": fraction,
        "bbox_xyxy": bbox,
        "centroid_xy_fraction": centroid,
        "centroid_within_margin": centroid_framed,
        "frame_resolution_hw": [height, width],
        "thresholds": {
            "minimum_pixels": int(minimum_pixels),
            "minimum_pixel_fraction": float(minimum_pixel_fraction),
            "centroid_margin_fraction": float(centroid_margin_fraction),
        },
        "passed": passed,
        "measurement_authority": "native_semantic_segmentation_aov",
        "rgb_or_model_label_used": False,
    }


__all__ = [
    "NativeTaskCameraObservabilityError",
    "SCHEMA_VERSION",
    "measure_native_task_camera_observability",
]
