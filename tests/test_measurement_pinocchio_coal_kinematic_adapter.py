from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys

import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
    probe_measurement_adapter,
)
from blueprint_pipeline.measurement_pinocchio_coal_kinematic_adapter import (
    COAL_VERSION,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PIN_VERSION,
    PROTOCOL_ID,
    TARGET_VERSION,
    implementation_digest,
    run_pinocchio_coal_kinematic_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


pytestmark = pytest.mark.slow
DEVELOPMENT_DIGEST = "sha256:" + hashlib.sha256(b"exact-geometry-development").hexdigest()
QUALIFICATION_DIGEST = "sha256:" + hashlib.sha256(b"exact-geometry-qualification").hexdigest()
CONTROLLER_DIGEST = "sha256:" + hashlib.sha256(b"two-link-controller-scope").hexdigest()


def _request(
    *,
    case_id: str = "reachable-clear-development-001",
    target: list[float] | None = None,
    obstacle: list[float] | None = None,
) -> dict:
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-pinocchio-coal-planar-1",
        method_ids=["exact-geometry-stack"],
        development_split_digest=DEVELOPMENT_DIGEST,
        qualification_split_digest=QUALIFICATION_DIGEST,
        capture_bundle_digests=[DEVELOPMENT_DIGEST],
        robot_controller_digests=[CONTROLLER_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 60},
        minimum_repeated_trials=2,
    )
    case = build_benchmark_case_manifest(
        spec,
        case_id=case_id,
        split="development",
        input_artifact_digests=[DEVELOPMENT_DIGEST],
        task_class="static_reachability",
        material_regime="synthetic_rigid_geometry",
        operating_point={
            "adapter_protocol": PROTOCOL_ID,
            "protocol_family": "planar_reach_discrete_collision",
            "length_unit": "meters",
            "angle_unit": "radians",
            "link_lengths_m": [0.6, 0.4],
            "link_half_width_m": 0.025,
            "joint_limits_rad": [[-3.14, 3.14], [-3.14, 3.14]],
            "target_xy_m": target or [0.6, 0.4],
            "home_joint_positions_rad": [0.0, 0.0],
            "obstacle_center_xy_m": obstacle or [1.5, 1.5],
            "obstacle_half_extents_xy_m": [0.08, 0.08],
            "path_sample_count": 101,
            "elbow_branch": "up",
            "clearance_unsafe_threshold_m": 0.01,
            "target_tolerance_m": 1e-9,
        },
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("exact-geometry-stack"),
        spec,
        case,
        execution_id=f"exact-geometry-{case_id}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="pinocchio-coal-cpu",
        precision="float64",
        seed=0,
        solver_settings={
            "inverse_kinematics": "analytic_two_link",
            "collision_query": "coal_gjk_signed_distance",
            "path_check": "finite_joint_interpolation",
            "continuous_collision": False,
        },
        timeout_seconds=30,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_exact_geometry_extra_and_probe_bind_pinocchio_coal_pair() -> None:
    assert importlib.metadata.version("pin") == PIN_VERSION
    assert importlib.metadata.version("coal") == COAL_VERSION
    descriptor = build_measurement_adapter_descriptor("exact-geometry-stack")
    assert descriptor["target_version"] == TARGET_VERSION
    assert descriptor["probe_contract"]["python_distributions"] == ["pin", "coal"]
    probe = probe_measurement_adapter(descriptor)
    assert probe["status"] in {"available", "partial"}
    assert [
        row["name"] for row in probe["probes"] if row["probe_type"] == "python_distribution"
    ] == [
        "pin",
        "coal",
    ]


def test_exact_geometry_worker_distinguishes_clear_collision_and_unreachable() -> None:
    clear = run_pinocchio_coal_kinematic_request(_request())
    collision = run_pinocchio_coal_kinematic_request(
        _request(case_id="reachable-collision-development-002", obstacle=[0.45, 0.07])
    )
    unreachable = run_pinocchio_coal_kinematic_request(
        _request(case_id="unreachable-development-003", target=[1.2, 0.0])
    )
    assert clear["status"] == collision["status"] == unreachable["status"] == "completed"
    assert clear["unsafe_condition_predicted"] is False
    assert clear["observed_metrics"]["contact_sequence"] == "collision_free_discrete_path"
    assert clear["runtime_observations"]["target_position_error_m"] == 0.0
    assert clear["runtime_observations"]["deterministic_replay_match"] is True
    assert collision["unsafe_condition_predicted"] is True
    assert collision["observed_metrics"]["contact_sequence"] == "obstacle_contact"
    assert collision["observed_metrics"]["penetration"] > 0
    assert collision["runtime_observations"]["colliding_link_ids"] == [1]
    assert unreachable["unsafe_condition_predicted"] is True
    assert unreachable["observed_metrics"]["contact_sequence"] == ("not_evaluated_unreachable")
    assert unreachable["runtime_observations"]["target_reachable"] is False


def test_exact_geometry_worker_runs_through_uniform_subprocess_boundary() -> None:
    bundle = run_measurement_adapter_execution(
        _request(),
        command_argv=[
            sys.executable,
            "-m",
            "blueprint_pipeline.measurement_pinocchio_coal_kinematic_adapter",
        ],
        execute=True,
    )
    assert bundle["receipt"]["status"] == "completed"
    assert bundle["receipt"]["evidence_class"] == "development_execution"
    assert bundle["receipt"]["runtime_observations"]["continuous_collision_evaluated"] is False
    assert bundle["prediction"]["physical_success_established"] is False
    assert bundle["qualification_created"] is False
    assert bundle["catalog_mutated"] is False


def test_exact_geometry_worker_rejects_protocol_and_continuous_claim_tampering() -> None:
    protocol = copy.deepcopy(_request())
    protocol["case_manifest"]["operating_point"]["adapter_protocol"] = (
        "pinocchio_coal_continuous_collision.v1"
    )
    _rehash(protocol["case_manifest"], "case_manifest_digest")
    protocol.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="adapter_protocol_invalid"):
        run_pinocchio_coal_kinematic_request(protocol)

    continuous = copy.deepcopy(_request())
    continuous["runtime_configuration"]["solver_settings"]["continuous_collision"] = True
    encoded = json.dumps(
        continuous["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    continuous["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    continuous.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_settings_invalid"):
        run_pinocchio_coal_kinematic_request(continuous)

    home = copy.deepcopy(_request())
    home["case_manifest"]["operating_point"]["home_joint_positions_rad"] = [4.0, 0.0]
    _rehash(home["case_manifest"], "case_manifest_digest")
    home.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="home_joint_positions_invalid"):
        run_pinocchio_coal_kinematic_request(home)


def test_exact_geometry_implementation_identity_cannot_inherit_physics_worker() -> None:
    request = _request()
    request["implementation"]["implementation_id"] = "blueprint-mujoco-rigid-development-adapter"
    request.pop("execution_request_digest")
    result = run_pinocchio_coal_kinematic_request(request)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["exact_geometry_implementation_id_mismatch"]
