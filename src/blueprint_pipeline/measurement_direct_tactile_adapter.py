"""Direct tactile-sequence development measurement worker.

This worker reduces a bounded, synchronized optical-tactile marker/contact
sequence and synthetic normal/shear force channels.  It is the development
port for the real-sensor side of Q-TACT: production use still requires exact
sensor calibration, physical provenance, hidden labels, independent execution,
and R6/R7 approval.  The checked fixture is synthetic and cannot establish
force truth or sensor validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-direct-tactile-sequence-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "direct_tactile_sequence_reduction.v1"


def implementation_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid")
    return result


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid")
    return value


def _numeric_series(value: Any, *, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MeasurementAdapterExecutionError(f"direct_tactile_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("direct_tactile_operating_point_invalid")
    point = dict(raw)
    exact = {
        "adapter_protocol": PROTOCOL_ID,
        "sensor_class": "optical_tactile_with_synchronized_force",
        "data_origin": "synthetic_development_fixture",
        "calibration_scope": "synthetic_identity_only",
        "image_unit": "pixels",
        "force_unit": "newtons",
        "timestamp_unit": "nanoseconds",
    }
    for key, expected in exact.items():
        if point.get(key) != expected:
            raise MeasurementAdapterExecutionError(f"direct_tactile_{key}_invalid")
    frame_count = _integer(point.get("frame_count"), name="frame_count", minimum=3, maximum=256)
    marker_count = _integer(point.get("marker_count"), name="marker_count", minimum=3, maximum=1024)
    grid_width = _integer(
        point.get("contact_grid_width"), name="contact_grid_width", minimum=2, maximum=128
    )
    grid_height = _integer(
        point.get("contact_grid_height"), name="contact_grid_height", minimum=2, maximum=128
    )
    timestamps = point.get("timestamps_ns")
    if (
        not isinstance(timestamps, list)
        or len(timestamps) != frame_count
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in timestamps
        )
        or any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False))
    ):
        raise MeasurementAdapterExecutionError("direct_tactile_timestamps_invalid")
    marker_frames = point.get("marker_displacement_frames_px")
    if not isinstance(marker_frames, list) or len(marker_frames) != frame_count:
        raise MeasurementAdapterExecutionError("direct_tactile_marker_frames_invalid")
    checked_markers: list[list[list[float]]] = []
    for frame in marker_frames:
        if not isinstance(frame, list) or len(frame) != marker_count:
            raise MeasurementAdapterExecutionError("direct_tactile_marker_frames_invalid")
        checked_frame: list[list[float]] = []
        for vector in frame:
            if not isinstance(vector, list) or len(vector) != 2:
                raise MeasurementAdapterExecutionError("direct_tactile_marker_frames_invalid")
            checked_frame.append([_number(item, name="marker_displacement") for item in vector])
        checked_markers.append(checked_frame)
    intensity_frames = point.get("contact_intensity_frames")
    pixel_count = grid_width * grid_height
    if not isinstance(intensity_frames, list) or len(intensity_frames) != frame_count:
        raise MeasurementAdapterExecutionError("direct_tactile_contact_frames_invalid")
    checked_intensities: list[list[float]] = []
    for frame in intensity_frames:
        if not isinstance(frame, list) or len(frame) != pixel_count:
            raise MeasurementAdapterExecutionError("direct_tactile_contact_frames_invalid")
        checked_intensities.append(
            [_number(item, name="contact_intensity", minimum=0.0, maximum=1.0) for item in frame]
        )
    normal_force = _numeric_series(
        point.get("normal_force_n"), name="normal_force", length=frame_count
    )
    shear_force = _numeric_series(
        point.get("shear_force_n"), name="shear_force", length=frame_count
    )
    if any(value < 0 for value in normal_force) or any(value < 0 for value in shear_force):
        raise MeasurementAdapterExecutionError("direct_tactile_force_negative")
    return {
        "frame_count": frame_count,
        "marker_count": marker_count,
        "contact_grid_width": grid_width,
        "contact_grid_height": grid_height,
        "timestamps_ns": timestamps,
        "marker_displacement_frames_px": checked_markers,
        "contact_intensity_frames": checked_intensities,
        "normal_force_n": normal_force,
        "shear_force_n": shear_force,
        "pixel_pitch_mm": _number(
            point.get("pixel_pitch_mm"), name="pixel_pitch", minimum=1e-6, maximum=100.0
        ),
        "contact_intensity_threshold": _number(
            point.get("contact_intensity_threshold"),
            name="contact_intensity_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_normal_force_for_slip_n": _number(
            point.get("minimum_normal_force_for_slip_n"),
            name="minimum_normal_force_for_slip",
            minimum=0.0,
            maximum=1e6,
        ),
        "slip_shear_to_normal_ratio": _number(
            point.get("slip_shear_to_normal_ratio"),
            name="slip_shear_to_normal_ratio",
            minimum=0.0,
            maximum=100.0,
        ),
        "slip_marker_displacement_px": _number(
            point.get("slip_marker_displacement_px"),
            name="slip_marker_displacement",
            minimum=0.0,
            maximum=1e6,
        ),
        "maximum_peak_shear_force_n": _number(
            point.get("maximum_peak_shear_force_n"),
            name="maximum_peak_shear_force",
            minimum=0.0,
            maximum=1e6,
        ),
    }


def _reduce(np: Any, point: Mapping[str, Any]) -> dict[str, Any]:
    markers = np.asarray(point["marker_displacement_frames_px"], dtype=np.float64)
    intensities = np.asarray(point["contact_intensity_frames"], dtype=np.float64)
    normal = np.asarray(point["normal_force_n"], dtype=np.float64)
    shear = np.asarray(point["shear_force_n"], dtype=np.float64)
    marker_magnitudes = np.linalg.norm(markers, axis=2)
    mean_marker_displacement = np.mean(marker_magnitudes, axis=1)
    contact_masks = intensities >= point["contact_intensity_threshold"]
    contact_pixels = np.sum(contact_masks, axis=1)
    contact_area_mm2 = contact_pixels * point["pixel_pitch_mm"] ** 2
    shear_ratio = np.divide(shear, normal, out=np.zeros_like(shear), where=normal > 0)
    slip_candidates = np.flatnonzero(
        (normal >= point["minimum_normal_force_for_slip_n"])
        & (shear_ratio >= point["slip_shear_to_normal_ratio"])
        & (mean_marker_displacement >= point["slip_marker_displacement_px"])
    )
    slip_index = int(slip_candidates[0]) if len(slip_candidates) else None
    trace = {
        "mean_marker_displacement_px": [float(value) for value in mean_marker_displacement],
        "contact_area_mm2": [float(value) for value in contact_area_mm2],
        "shear_to_normal_ratio": [float(value) for value in shear_ratio],
        "peak_marker_displacement_px": float(np.max(marker_magnitudes)),
        "peak_mean_marker_displacement_px": float(np.max(mean_marker_displacement)),
        "maximum_contact_area_mm2": float(np.max(contact_area_mm2)),
        "peak_normal_force_n": float(np.max(normal)),
        "peak_shear_force_n": float(np.max(shear)),
        "peak_shear_to_normal_ratio": float(np.max(shear_ratio)),
        "slip_onset_frame": slip_index,
        "slip_onset_timestamp_ns": (
            point["timestamps_ns"][slip_index] if slip_index is not None else None
        ),
        "contact_observed": bool(np.any(contact_pixels)),
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_direct_tactile_request(request_value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    base_observations = {
        "engine_version": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    implementation = request["implementation"]
    mismatch = next(
        (
            code
            for actual, expected, code in (
                (
                    implementation["implementation_id"],
                    IMPLEMENTATION_ID,
                    "direct_tactile_implementation_id_mismatch",
                ),
                (
                    implementation["implementation_version"],
                    IMPLEMENTATION_VERSION,
                    "direct_tactile_implementation_version_mismatch",
                ),
                (
                    implementation["implementation_digest"],
                    implementation_digest(),
                    "direct_tactile_implementation_digest_mismatch",
                ),
            )
            if actual != expected
        ),
        None,
    )
    if mismatch:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=[mismatch],
        )
    try:
        import numpy as np
    except ImportError:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["direct_tactile_runtime_unavailable"],
        )
    base_observations["engine_version"] = str(np.__version__)
    settings = dict(runtime["solver_settings"])
    if set(settings) != {"analysis_method", "numpy_version", "replay_count"}:
        raise MeasurementAdapterExecutionError("direct_tactile_solver_settings_invalid")
    if settings["analysis_method"] != "deterministic_sequence_reduction":
        raise MeasurementAdapterExecutionError("direct_tactile_analysis_method_invalid")
    if settings["numpy_version"] != np.__version__:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["direct_tactile_numpy_version_mismatch"],
        )
    if settings["replay_count"] != 2:
        raise MeasurementAdapterExecutionError("direct_tactile_replay_count_invalid")
    point = _operating_point(request)
    first = _reduce(np, point)
    second = _reduce(np, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    slip_observed = first["slip_onset_frame"] is not None
    unsafe = slip_observed or first["peak_shear_force_n"] > point["maximum_peak_shear_force_n"]
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics: dict[str, Any] = {
        "state_trajectory": first["peak_mean_marker_displacement_px"],
        "topology_contact": (
            "contact_patch_observed" if first["contact_observed"] else "contact_patch_absent"
        ),
        "force": first["peak_shear_force_n"],
        "task_outcome": "incipient_slip_observed" if slip_observed else "stable_contact_observed",
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    observations = {
        **base_observations,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_digest": implementation_digest(),
        "adapter_protocol": PROTOCOL_ID,
        "solver_settings_digest": runtime["solver_settings_digest"],
        "sensor_class": "optical_tactile_with_synchronized_force",
        "data_origin": "synthetic_development_fixture",
        "calibration_scope": "synthetic_identity_only",
        "frame_count": point["frame_count"],
        "marker_count": point["marker_count"],
        "peak_marker_displacement_px": first["peak_marker_displacement_px"],
        "peak_mean_marker_displacement_px": first["peak_mean_marker_displacement_px"],
        "maximum_contact_area_mm2": first["maximum_contact_area_mm2"],
        "peak_normal_force_n": first["peak_normal_force_n"],
        "peak_shear_force_n": first["peak_shear_force_n"],
        "peak_shear_to_normal_ratio": first["peak_shear_to_normal_ratio"],
        "slip_onset_frame": first["slip_onset_frame"],
        "slip_onset_timestamp_ns": first["slip_onset_timestamp_ns"],
        "trace_digest": first["trace_digest"],
        "repeat_trace_digest": second["trace_digest"],
        "deterministic_replay_match": replay_match,
    }
    if not replay_match:
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics=metrics,
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["direct_tactile_replay_mismatch"],
        )
    return build_measurement_adapter_worker_result(
        request,
        status="completed",
        observed_metrics=metrics,
        unsafe_condition_predicted=unsafe,
        runtime_observations=observations,
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError("direct_tactile_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("direct_tactile_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a direct tactile development case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_direct_tactile_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "implementation_digest",
    "main",
    "run_direct_tactile_request",
]
