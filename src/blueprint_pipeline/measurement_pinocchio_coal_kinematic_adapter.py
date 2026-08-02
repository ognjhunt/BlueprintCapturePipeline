"""Pinocchio/Coal planar reachability and discrete-collision development worker.

The worker uses Pinocchio for articulated forward kinematics and Coal for
primitive signed-distance queries. It deliberately performs finite-sample
path checking rather than continuous collision detection and consumes only
synthetic public cases. It cannot establish captured-site collider accuracy,
continuous collision safety, physical reachability, or R5-R7 authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-pinocchio-coal-planar-kinematic-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "pinocchio_coal_planar_reach_discrete_collision.v1"
TARGET_VERSION = "pin-4.1.0+coal-3.0.3"
PIN_VERSION = "4.1.0"
COAL_VERSION = "3.0.3"


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
        raise MeasurementAdapterExecutionError(f"exact_geometry_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"exact_geometry_{name}_invalid") from exc
    if not math.isfinite(result):
        raise MeasurementAdapterExecutionError(f"exact_geometry_{name}_invalid")
    if minimum is not None and result < minimum:
        raise MeasurementAdapterExecutionError(f"exact_geometry_{name}_invalid")
    if maximum is not None and result > maximum:
        raise MeasurementAdapterExecutionError(f"exact_geometry_{name}_invalid")
    return result


def _vector(
    value: Any,
    *,
    name: str,
    size: int,
    minimum: float | None = None,
) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise MeasurementAdapterExecutionError(f"exact_geometry_{name}_invalid")
    return [_number(item, name=name, minimum=minimum) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("exact_geometry_operating_point_invalid")
    point = dict(raw)
    for key, expected in (
        ("adapter_protocol", PROTOCOL_ID),
        ("protocol_family", "planar_reach_discrete_collision"),
        ("length_unit", "meters"),
        ("angle_unit", "radians"),
    ):
        if point.get(key) != expected:
            raise MeasurementAdapterExecutionError(f"exact_geometry_{key}_invalid")
    lengths = _vector(point.get("link_lengths_m"), name="link_lengths", size=2, minimum=1e-4)
    half_width = _number(point.get("link_half_width_m"), name="link_half_width", minimum=1e-5)
    limits_raw = point.get("joint_limits_rad")
    if not isinstance(limits_raw, list) or len(limits_raw) != 2:
        raise MeasurementAdapterExecutionError("exact_geometry_joint_limits_invalid")
    limits: list[list[float]] = []
    for raw_limit in limits_raw:
        limit = _vector(raw_limit, name="joint_limits", size=2)
        if limit[0] >= limit[1] or limit[0] < -2 * math.pi or limit[1] > 2 * math.pi:
            raise MeasurementAdapterExecutionError("exact_geometry_joint_limits_invalid")
        limits.append(limit)
    target = _vector(point.get("target_xy_m"), name="target_xy", size=2)
    home = _vector(point.get("home_joint_positions_rad"), name="home_joint_positions", size=2)
    if any(
        value < limits[index][0] or value > limits[index][1] for index, value in enumerate(home)
    ):
        raise MeasurementAdapterExecutionError("exact_geometry_home_joint_positions_invalid")
    obstacle_center = _vector(point.get("obstacle_center_xy_m"), name="obstacle_center_xy", size=2)
    obstacle_half_extents = _vector(
        point.get("obstacle_half_extents_xy_m"),
        name="obstacle_half_extents_xy",
        size=2,
        minimum=1e-5,
    )
    samples = point.get("path_sample_count")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 1001:
        raise MeasurementAdapterExecutionError("exact_geometry_path_sample_count_invalid")
    branch = point.get("elbow_branch")
    if branch not in {"up", "down"}:
        raise MeasurementAdapterExecutionError("exact_geometry_elbow_branch_invalid")
    return {
        "link_lengths_m": lengths,
        "link_half_width_m": half_width,
        "joint_limits_rad": limits,
        "target_xy_m": target,
        "home_joint_positions_rad": home,
        "obstacle_center_xy_m": obstacle_center,
        "obstacle_half_extents_xy_m": obstacle_half_extents,
        "path_sample_count": samples,
        "elbow_branch": branch,
        "clearance_unsafe_threshold_m": _number(
            point.get("clearance_unsafe_threshold_m"),
            name="clearance_unsafe_threshold",
            minimum=0.0,
        ),
        "target_tolerance_m": _number(
            point.get("target_tolerance_m"),
            name="target_tolerance",
            minimum=0.0,
            maximum=0.1,
        ),
    }


def _inverse_kinematics(point: Mapping[str, Any]) -> list[float] | None:
    l1, l2 = point["link_lengths_m"]
    x, y = point["target_xy_m"]
    cosine = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cosine < -1.0 - 1e-12 or cosine > 1.0 + 1e-12:
        return None
    cosine = min(1.0, max(-1.0, cosine))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    if point["elbow_branch"] == "down":
        sine = -sine
    q2 = math.atan2(sine, cosine)
    q1 = math.atan2(y, x) - math.atan2(l2 * sine, l1 + l2 * cosine)
    solution = [q1, q2]
    if any(
        value < point["joint_limits_rad"][index][0] - 1e-12
        or value > point["joint_limits_rad"][index][1] + 1e-12
        for index, value in enumerate(solution)
    ):
        return None
    return solution


def _build_model(pin: Any, point: Mapping[str, Any]) -> tuple[Any, int, int, int]:
    import numpy as np

    model = pin.Model()
    joint_one = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "shoulder")
    model.appendBodyToJoint(
        joint_one, pin.Inertia.FromSphere(1.0, point["link_half_width_m"]), pin.SE3.Identity()
    )
    joint_two = model.addJoint(
        joint_one,
        pin.JointModelRZ(),
        pin.SE3(np.eye(3), np.array([point["link_lengths_m"][0], 0.0, 0.0])),
        "elbow",
    )
    model.appendBodyToJoint(
        joint_two, pin.Inertia.FromSphere(1.0, point["link_half_width_m"]), pin.SE3.Identity()
    )
    tool_frame = model.addFrame(
        pin.Frame(
            "tool",
            joint_two,
            joint_two,
            pin.SE3(np.eye(3), np.array([point["link_lengths_m"][1], 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    model.lowerPositionLimit[:] = [row[0] for row in point["joint_limits_rad"]]
    model.upperPositionLimit[:] = [row[1] for row in point["joint_limits_rad"]]
    return model, joint_one, joint_two, tool_frame


def _coal_transform(coal: Any, placement: Any) -> Any:
    transform = coal.Transform3s()
    transform.setRotation(placement.rotation)
    transform.setTranslation(placement.translation)
    return transform


def _simulate(pin: Any, coal: Any, point: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    solution = _inverse_kinematics(point)
    if solution is None:
        trace = {
            "target_reachable": False,
            "joint_solution_rad": None,
            "target_position_error_m": None,
            "minimum_discrete_clearance_m": None,
            "maximum_discrete_penetration_m": 0.0,
            "first_collision_sample": None,
            "colliding_link_ids": [],
            "sample_count": 0,
        }
    else:
        model, joint_one, joint_two, tool_frame = _build_model(pin, point)
        data = model.createData()
        link_shapes = [
            coal.Box(
                point["link_lengths_m"][0],
                2.0 * point["link_half_width_m"],
                2.0 * point["link_half_width_m"],
            ),
            coal.Box(
                point["link_lengths_m"][1],
                2.0 * point["link_half_width_m"],
                2.0 * point["link_half_width_m"],
            ),
        ]
        obstacle = coal.Box(
            2.0 * point["obstacle_half_extents_xy_m"][0],
            2.0 * point["obstacle_half_extents_xy_m"][1],
            2.0 * point["link_half_width_m"],
        )
        obstacle_transform = coal.Transform3s()
        obstacle_transform.setIdentity()
        obstacle_transform.setTranslation(
            np.array([*point["obstacle_center_xy_m"], 0.0], dtype=float)
        )
        home = np.array(point["home_joint_positions_rad"], dtype=float)
        target = np.array(solution, dtype=float)
        minimum_clearance = math.inf
        maximum_penetration = 0.0
        first_collision: int | None = None
        colliding_links: set[int] = set()
        sample_count = point["path_sample_count"]
        for sample in range(sample_count):
            alpha = sample / (sample_count - 1)
            configuration = home + alpha * (target - home)
            pin.forwardKinematics(model, data, configuration)
            pin.updateFramePlacements(model, data)
            placements = [
                data.oMi[joint_one]
                * pin.SE3(np.eye(3), np.array([point["link_lengths_m"][0] / 2.0, 0.0, 0.0])),
                data.oMi[joint_two]
                * pin.SE3(np.eye(3), np.array([point["link_lengths_m"][1] / 2.0, 0.0, 0.0])),
            ]
            for link_index, (shape, placement) in enumerate(zip(link_shapes, placements)):
                result = coal.DistanceResult()
                signed_distance = float(
                    coal.distance(
                        shape,
                        _coal_transform(coal, placement),
                        obstacle,
                        obstacle_transform,
                        coal.DistanceRequest(),
                        result,
                    )
                )
                minimum_clearance = min(minimum_clearance, signed_distance)
                maximum_penetration = max(maximum_penetration, -signed_distance)
                if signed_distance <= 0.0:
                    colliding_links.add(link_index + 1)
                    if first_collision is None:
                        first_collision = sample
        pin.forwardKinematics(model, data, target)
        pin.updateFramePlacements(model, data)
        tool_xy = [float(item) for item in data.oMf[tool_frame].translation[:2]]
        target_error = math.dist(tool_xy, point["target_xy_m"])
        trace = {
            "target_reachable": True,
            "joint_solution_rad": [float(item) for item in solution],
            "tool_position_xy_m": tool_xy,
            "target_position_error_m": target_error,
            "minimum_discrete_clearance_m": minimum_clearance,
            "maximum_discrete_penetration_m": maximum_penetration,
            "first_collision_sample": first_collision,
            "colliding_link_ids": sorted(colliding_links),
            "sample_count": sample_count,
        }
    trace["trace_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return trace


def run_pinocchio_coal_kinematic_request(
    request_value: Mapping[str, Any],
) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    runtime = request["runtime_configuration"]
    observations = {
        "engine_version": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
    }
    implementation = request["implementation"]
    for key, expected, code in (
        ("implementation_id", IMPLEMENTATION_ID, "implementation_id_mismatch"),
        ("implementation_version", IMPLEMENTATION_VERSION, "implementation_version_mismatch"),
        ("implementation_digest", implementation_digest(), "implementation_digest_mismatch"),
    ):
        if implementation[key] != expected:
            return build_measurement_adapter_worker_result(
                request,
                status="blocked",
                observed_metrics={},
                unsafe_condition_predicted=None,
                runtime_observations=observations,
                failure_codes=[f"exact_geometry_{code}"],
            )
    try:
        if importlib.metadata.version("pin") != PIN_VERSION:
            raise importlib.metadata.PackageNotFoundError
        if importlib.metadata.version("coal") != COAL_VERSION:
            raise importlib.metadata.PackageNotFoundError
        import coal
        import pinocchio as pin
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["exact_geometry_package_or_version_unavailable"],
        )
    observations.update(
        {
            "engine_version": TARGET_VERSION,
            "pinocchio_version": str(pin.__version__),
            "coal_version": str(coal.__version__),
        }
    )
    if runtime["target_engine_version"] != TARGET_VERSION:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["exact_geometry_target_version_mismatch"],
        )
    if runtime["backend_id"] != "pinocchio-coal-cpu":
        raise MeasurementAdapterExecutionError("exact_geometry_backend_invalid")
    if runtime["precision"] != "float64":
        raise MeasurementAdapterExecutionError("exact_geometry_precision_invalid")
    if runtime["solver_settings"] != {
        "inverse_kinematics": "analytic_two_link",
        "collision_query": "coal_gjk_signed_distance",
        "path_check": "finite_joint_interpolation",
        "continuous_collision": False,
    }:
        raise MeasurementAdapterExecutionError("exact_geometry_solver_settings_invalid")
    point = _operating_point(request)
    first = _simulate(pin, coal, point)
    second = _simulate(pin, coal, point)
    replay_match = first["trace_digest"] == second["trace_digest"]
    requested = set(request["case_manifest"]["requested_metric_ids"])
    available_metrics = {
        "contact_sequence": (
            "not_evaluated_unreachable"
            if not first["target_reachable"]
            else "obstacle_contact"
            if first["first_collision_sample"] is not None
            else "collision_free_discrete_path"
        ),
        "penetration": first["maximum_discrete_penetration_m"],
    }
    metrics = {key: value for key, value in available_metrics.items() if key in requested}
    unsafe = (
        not first["target_reachable"]
        or (
            first["target_position_error_m"] is not None
            and first["target_position_error_m"] > point["target_tolerance_m"]
        )
        or first["first_collision_sample"] is not None
        or (
            first["minimum_discrete_clearance_m"] is not None
            and first["minimum_discrete_clearance_m"] < point["clearance_unsafe_threshold_m"]
        )
    )
    observations.update(
        {
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_digest": implementation_digest(),
            "solver_settings_digest": runtime["solver_settings_digest"],
            "device": "cpu",
            "continuous_collision_evaluated": False,
            "captured_mesh_loaded": False,
            "captured_registration_used": False,
            **{key: value for key, value in first.items() if key != "trace_digest"},
            "trace_digest": first["trace_digest"],
            "repeat_trace_digest": second["trace_digest"],
            "deterministic_replay_match": replay_match,
        }
    )
    if not replay_match:
        return build_measurement_adapter_worker_result(
            request,
            status="failed",
            observed_metrics=metrics,
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["exact_geometry_replay_mismatch"],
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
        raise MeasurementAdapterExecutionError("exact_geometry_request_unreadable") from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("exact_geometry_request_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Pinocchio/Coal planar kinematic development case"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_pinocchio_coal_kinematic_request(_load_object(args.request))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COAL_VERSION",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PIN_VERSION",
    "PROTOCOL_ID",
    "TARGET_VERSION",
    "implementation_digest",
    "main",
    "run_pinocchio_coal_kinematic_request",
]
