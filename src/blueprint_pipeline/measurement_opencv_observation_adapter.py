"""Calibrated Capture-to-Observation development benchmark worker.

The worker runs a real OpenCV PnP/reprojection calculation over a digest-bound
public development case.  It measures calibrated RGB reprojection residuals,
missing-depth fraction, and timestamp alignment.  It deliberately does not
render a scene, infer hidden labels, claim a physical calibration, or authorize
Q-SENSOR/R6/R7.  A repository fixture can exercise this adapter end to end, but
only an independently controlled runner and held-out physical labels can turn
the same protocol into qualification evidence.
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


IMPLEMENTATION_ID = "blueprint-opencv-calibrated-observation-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "opencv_calibrated_observation.v1"
COORDINATE_CONVENTION = "opencv_x_right_y_down_z_forward"


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
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    return result


def _vector(
    value: Any,
    *,
    name: str,
    length: int,
    minimum: float | None = None,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    return [_number(item, name=name, minimum=minimum) for item in value]


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    return value


def _points(value: Any, *, name: str, dimensions: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 6:
        raise MeasurementAdapterExecutionError(f"opencv_observation_{name}_invalid")
    return [_vector(row, name=name, length=dimensions) for row in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("opencv_observation_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("opencv_observation_protocol_invalid")
    if point.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise MeasurementAdapterExecutionError("opencv_observation_coordinate_convention_invalid")
    if point.get("length_unit") != "meters":
        raise MeasurementAdapterExecutionError("opencv_observation_length_unit_invalid")
    if point.get("image_unit") != "pixels":
        raise MeasurementAdapterExecutionError("opencv_observation_image_unit_invalid")
    if point.get("timestamp_unit") != "nanoseconds":
        raise MeasurementAdapterExecutionError("opencv_observation_timestamp_unit_invalid")
    image_size = point.get("image_size_px")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise MeasurementAdapterExecutionError("opencv_observation_image_size_invalid")
    width = int(_number(image_size[0], name="image_width", minimum=2))
    height = int(_number(image_size[1], name="image_height", minimum=2))
    if [width, height] != image_size:
        raise MeasurementAdapterExecutionError("opencv_observation_image_size_invalid")
    matrix_raw = point.get("camera_matrix")
    if not isinstance(matrix_raw, list) or len(matrix_raw) != 3:
        raise MeasurementAdapterExecutionError("opencv_observation_camera_matrix_invalid")
    matrix = [_vector(row, name="camera_matrix", length=3) for row in matrix_raw]
    if matrix[0][0] <= 0 or matrix[1][1] <= 0:
        raise MeasurementAdapterExecutionError("opencv_observation_focal_length_invalid")
    if matrix[0][1] != 0 or matrix[1][0] != 0 or matrix[2] != [0.0, 0.0, 1.0]:
        raise MeasurementAdapterExecutionError("opencv_observation_camera_matrix_noncanonical")
    if not (0 <= matrix[0][2] < width and 0 <= matrix[1][2] < height):
        raise MeasurementAdapterExecutionError("opencv_observation_principal_point_invalid")
    distortion = point.get("distortion_coefficients")
    if not isinstance(distortion, list) or len(distortion) not in {4, 5, 8, 12, 14}:
        raise MeasurementAdapterExecutionError("opencv_observation_distortion_invalid")
    distortion_values = [_number(item, name="distortion") for item in distortion]
    object_points = _points(point.get("object_points_m"), name="object_points", dimensions=3)
    image_points = _points(
        point.get("observed_image_points_px"),
        name="image_points",
        dimensions=2,
    )
    if len(object_points) != len(image_points):
        raise MeasurementAdapterExecutionError("opencv_observation_correspondence_count_mismatch")
    if any(not (0 <= pixel[0] < width and 0 <= pixel[1] < height) for pixel in image_points):
        raise MeasurementAdapterExecutionError("opencv_observation_image_point_out_of_bounds")
    depth_raw = point.get("depth_samples_m")
    if not isinstance(depth_raw, list) or len(depth_raw) != len(object_points):
        raise MeasurementAdapterExecutionError("opencv_observation_depth_samples_invalid")
    depth_samples = [
        None if item is None else _number(item, name="depth_sample", minimum=1e-6)
        for item in depth_raw
    ]
    capture_timestamp = _integer(point.get("capture_timestamp_ns"), name="capture_timestamp")
    reference_timestamp = _integer(
        point.get("reference_timestamp_ns"),
        name="reference_timestamp",
    )
    thresholds = point.get("unsafe_thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "maximum_reprojection_rmse_px",
        "maximum_missing_depth_fraction",
        "maximum_temporal_error_ms",
    }:
        raise MeasurementAdapterExecutionError("opencv_observation_unsafe_thresholds_invalid")
    return {
        "image_size_px": [width, height],
        "camera_matrix": matrix,
        "distortion_coefficients": distortion_values,
        "object_points_m": object_points,
        "observed_image_points_px": image_points,
        "depth_samples_m": depth_samples,
        "capture_timestamp_ns": capture_timestamp,
        "reference_timestamp_ns": reference_timestamp,
        "unsafe_thresholds": {
            key: _number(value, name=key, minimum=0.0) for key, value in thresholds.items()
        },
    }


def _solve(cv2: Any, np: Any, point: Mapping[str, Any]) -> dict[str, Any]:
    object_points = np.asarray(point["object_points_m"], dtype=np.float64)
    image_points = np.asarray(point["observed_image_points_px"], dtype=np.float64)
    if np.linalg.matrix_rank(object_points - object_points.mean(axis=0)) < 3:
        raise MeasurementAdapterExecutionError("opencv_observation_object_points_degenerate")
    camera_matrix = np.asarray(point["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(point["distortion_coefficients"], dtype=np.float64)
    success, rotation, translation = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise MeasurementAdapterExecutionError("opencv_observation_solvepnp_failed")
    projected, _ = cv2.projectPoints(
        object_points, rotation, translation, camera_matrix, distortion
    )
    projected = projected.reshape((-1, 2))
    pixel_residuals = np.linalg.norm(projected - image_points, axis=1)
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    estimated_depths = (object_points @ rotation_matrix.T + translation.reshape((1, 3)))[:, 2]
    if np.any(estimated_depths <= 0):
        raise MeasurementAdapterExecutionError("opencv_observation_estimated_depth_nonpositive")
    depth_residuals = [
        abs(float(estimated_depths[index]) - float(observed))
        for index, observed in enumerate(point["depth_samples_m"])
        if observed is not None
    ]
    trace = {
        "rotation_vector": [float(item) for item in rotation.reshape(-1)],
        "translation_m": [float(item) for item in translation.reshape(-1)],
        "projected_image_points_px": projected.tolist(),
        "estimated_depths_m": [float(item) for item in estimated_depths],
        "reprojection_rmse_px": float(np.sqrt(np.mean(pixel_residuals**2))),
        "maximum_reprojection_error_px": float(np.max(pixel_residuals)),
        "mean_depth_residual_m": (float(np.mean(depth_residuals)) if depth_residuals else None),
    }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_opencv_observation_request(
    request_value: Mapping[str, Any],
) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    base_observations = {
        "engine_version": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    implementation = request["implementation"]
    if implementation["implementation_id"] != IMPLEMENTATION_ID:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["opencv_observation_implementation_id_mismatch"],
        )
    if implementation["implementation_version"] != IMPLEMENTATION_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["opencv_observation_implementation_version_mismatch"],
        )
    if implementation["implementation_digest"] != implementation_digest():
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["opencv_observation_implementation_digest_mismatch"],
        )
    try:
        import cv2
        import numpy as np
    except ImportError:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["opencv_observation_runtime_unavailable"],
        )
    base_observations["engine_version"] = str(cv2.__version__)
    settings = dict(runtime["solver_settings"])
    if set(settings) != {"opencv_version", "solvepnp_flag", "replay_count"}:
        raise MeasurementAdapterExecutionError("opencv_observation_solver_settings_invalid")
    if settings["opencv_version"] != cv2.__version__:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=["opencv_observation_version_mismatch"],
        )
    if settings["solvepnp_flag"] != "SOLVEPNP_ITERATIVE":
        raise MeasurementAdapterExecutionError("opencv_observation_solvepnp_flag_invalid")
    if settings["replay_count"] != 2:
        raise MeasurementAdapterExecutionError("opencv_observation_replay_count_invalid")
    cv2.setRNGSeed(runtime["seed"])
    point = _operating_point(request)
    first = _solve(cv2, np, point)
    cv2.setRNGSeed(runtime["seed"])
    second = _solve(cv2, np, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    missing_depth_fraction = sum(item is None for item in point["depth_samples_m"]) / len(
        point["depth_samples_m"]
    )
    temporal_error_ms = (
        abs(point["capture_timestamp_ns"] - point["reference_timestamp_ns"]) / 1_000_000.0
    )
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics = {
        "calibrated_image_depth_lidar_residuals": first["reprojection_rmse_px"],
        "missing_depth_distribution": missing_depth_fraction,
        "temporal_error": temporal_error_ms,
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    thresholds = point["unsafe_thresholds"]
    unsafe = any(
        (
            first["reprojection_rmse_px"] > thresholds["maximum_reprojection_rmse_px"],
            missing_depth_fraction > thresholds["maximum_missing_depth_fraction"],
            temporal_error_ms > thresholds["maximum_temporal_error_ms"],
        )
    )
    observations = {
        **base_observations,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_digest": implementation_digest(),
        "adapter_protocol": PROTOCOL_ID,
        "coordinate_convention": COORDINATE_CONVENTION,
        "solver_settings_digest": runtime["solver_settings_digest"],
        "correspondence_count": len(point["object_points_m"]),
        "available_depth_sample_count": sum(item is not None for item in point["depth_samples_m"]),
        "reprojection_rmse_px": first["reprojection_rmse_px"],
        "maximum_reprojection_error_px": first["maximum_reprojection_error_px"],
        "mean_depth_residual_m": first["mean_depth_residual_m"],
        "missing_depth_fraction": missing_depth_fraction,
        "temporal_error_ms": temporal_error_ms,
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
            failure_codes=["opencv_observation_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("opencv_observation_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("opencv_observation_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a calibrated observation development measurement case"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_opencv_observation_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COORDINATE_CONVENTION",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "implementation_digest",
    "main",
    "run_opencv_observation_request",
]
