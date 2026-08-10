"""Measure task-object visibility and framing from native semantic pixels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "native_task_camera_observability.v1"
SEMANTIC_OUTPUT_CONFIGURATION_SCHEMA_VERSION = (
    "native_task_camera_semantic_output_configuration.v1"
)


class NativeTaskCameraObservabilityError(ValueError):
    """Stable semantic/framing failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def configure_native_semantic_id_output(camera_cfg: Any) -> dict[str, Any]:
    """Require integer semantic IDs across legacy and renderer-owned configs.

    Isaac Lab 4.x exposed ``colorize_semantic_segmentation`` directly on
    ``CameraCfg``.  The pinned Arena/Isaac Lab stack retains that field only as
    a deprecated forwarding shim and makes ``renderer_cfg`` authoritative.
    Set every control that exists so config copies made by either API retain
    the ID-valued AOV.  A four-channel colorized image is not interchangeable
    with semantic IDs and is deliberately rejected by the measurement gate.
    """

    configured_controls: list[str] = []
    if hasattr(camera_cfg, "colorize_semantic_segmentation"):
        camera_cfg.colorize_semantic_segmentation = False
        configured_controls.append("camera_cfg.colorize_semantic_segmentation")

    renderer_cfg = getattr(camera_cfg, "renderer_cfg", None)
    if renderer_cfg is not None and hasattr(
        renderer_cfg, "colorize_semantic_segmentation"
    ):
        renderer_cfg.colorize_semantic_segmentation = False
        configured_controls.append(
            "camera_cfg.renderer_cfg.colorize_semantic_segmentation"
        )

    if not configured_controls:
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_id_configuration_control_missing"]
        )

    readback = {
        control: bool(
            getattr(
                renderer_cfg
                if control.startswith("camera_cfg.renderer_cfg")
                else camera_cfg,
                "colorize_semantic_segmentation",
            )
        )
        for control in configured_controls
    }
    if any(readback.values()):
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_id_configuration_not_applied"]
        )
    return {
        "schema_version": SEMANTIC_OUTPUT_CONFIGURATION_SCHEMA_VERSION,
        "requested_representation": "integer_semantic_ids",
        "configured_controls": configured_controls,
        "control_readback": readback,
        "colorized_output_allowed_for_scoring": False,
        "passed": True,
    }


def _normalize_semantic_id_map(semantic_ids: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    semantic = np.asarray(semantic_ids)
    input_shape = [int(value) for value in semantic.shape]
    input_dtype = str(semantic.dtype)
    if semantic.ndim == 3 and semantic.shape[-1] == 4:
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_output_colorized"]
        )
    representation = "integer_id_map_2d"
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
        representation = "integer_id_map_single_channel"
    if semantic.ndim != 2 or not semantic.size:
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_shape_invalid"]
        )
    if semantic.dtype.kind not in {"i", "u"}:
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_dtype_invalid"]
        )
    return semantic, {
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "representation": representation,
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

    semantic, semantic_representation = _normalize_semantic_id_map(semantic_ids)
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
        "semantic_input": semantic_representation,
        "rgb_or_model_label_used": False,
    }


__all__ = [
    "NativeTaskCameraObservabilityError",
    "SCHEMA_VERSION",
    "SEMANTIC_OUTPUT_CONFIGURATION_SCHEMA_VERSION",
    "configure_native_semantic_id_output",
    "measure_native_task_camera_observability",
]
