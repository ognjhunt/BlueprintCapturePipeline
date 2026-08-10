"""Measure task-object visibility and framing from native semantic pixels."""

from __future__ import annotations

import math
import re
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


_RGBA_IDENTIFIER = re.compile(
    r"^\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$"
)


def _rgba_components(identifier: Any) -> tuple[int, int, int, int] | None:
    if isinstance(identifier, tuple) and len(identifier) == 4:
        raw = identifier
    elif isinstance(identifier, str):
        match = _RGBA_IDENTIFIER.fullmatch(identifier)
        if match is None:
            return None
        raw = match.groups()
    else:
        return None
    try:
        components = tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if any(value < 0 or value > 255 for value in components):
        return None
    return components  # type: ignore[return-value]


def _rgba_encodings(components: tuple[int, int, int, int]) -> dict[str, int]:
    r, g, b, a = components
    little = r | (g << 8) | (b << 16) | (a << 24)
    big = (r << 24) | (g << 16) | (b << 8) | a

    def signed(value: int) -> int:
        return value - (1 << 32) if value >= (1 << 31) else value

    return {
        "rgba_little_endian_uint32": little,
        "rgba_little_endian_int32": signed(little),
        "rgba_big_endian_uint32": big,
        "rgba_big_endian_int32": signed(big),
    }


def _semantic_label_ids(
    *, semantic: Any, id_to_labels: Mapping[Any, Any], target_label: str
) -> tuple[list[int], dict[str, Any]]:
    """Resolve Replicator scalar or RGBA-tuple label keys to integer AOV IDs.

    Some pinned Isaac/Replicator combinations return the requested one-channel
    ``int32`` AOV while retaining RGBA tuples as the keys of ``idToLabels``.
    The four bytes are a packed integer in that AOV, but byte order and signed
    representation are runtime details.  Resolve them from values actually
    present in the native frame; never guess from host endianness or RGB.
    """

    import numpy as np

    observed = {int(value) for value in np.unique(semantic.astype(np.int64))}
    parsed: list[tuple[Any, Any, tuple[int, int, int, int] | None]] = []
    tuple_components: list[tuple[int, int, int, int]] = []
    for identifier, entry in id_to_labels.items():
        components = _rgba_components(identifier)
        parsed.append((identifier, entry, components))
        if components is not None:
            tuple_components.append(components)

    tuple_encoding: str | None = None
    matched_count = 0
    if tuple_components:
        scores: dict[str, int] = {}
        signed_suffix = "_int32" if semantic.dtype.kind == "i" else "_uint32"
        encodings = [
            encoding
            for encoding in _rgba_encodings(tuple_components[0])
            if encoding.endswith(signed_suffix)
        ]
        for encoding in encodings:
            scores[encoding] = sum(
                _rgba_encodings(components)[encoding] in observed
                for components in tuple_components
            )
        matched_count = max(scores.values(), default=0)
        winners = [
            encoding for encoding, score in scores.items() if score == matched_count
        ]
        if matched_count < 1:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_tuple_encoding_unresolved"]
            )
        if len(winners) != 1:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_tuple_encoding_ambiguous"]
            )
        tuple_encoding = winners[0]

    target_ids: list[int] = []
    scalar_count = 0
    tuple_count = 0
    for identifier, entry, components in parsed:
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label != target_label:
            continue
        if components is not None:
            assert tuple_encoding is not None
            target_ids.append(_rgba_encodings(components)[tuple_encoding])
            tuple_count += 1
            continue
        try:
            target_ids.append(int(identifier))
            scalar_count += 1
        except (TypeError, ValueError) as exc:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_invalid"]
            ) from exc

    return sorted(set(target_ids)), {
        "scalar_target_identifier_count": scalar_count,
        "rgba_tuple_target_identifier_count": tuple_count,
        "rgba_tuple_encoding": tuple_encoding,
        "rgba_tuple_encoding_evidence_match_count": matched_count,
        "resolution_authority": "native_integer_aov_values_and_id_to_labels",
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
    target_ids, identifier_resolution = _semantic_label_ids(
        semantic=semantic,
        id_to_labels=id_to_labels,
        target_label=target_label,
    )
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
        "semantic_identifier_resolution": identifier_resolution,
        "rgb_or_model_label_used": False,
    }


__all__ = [
    "NativeTaskCameraObservabilityError",
    "SCHEMA_VERSION",
    "SEMANTIC_OUTPUT_CONFIGURATION_SCHEMA_VERSION",
    "configure_native_semantic_id_output",
    "measure_native_task_camera_observability",
]
