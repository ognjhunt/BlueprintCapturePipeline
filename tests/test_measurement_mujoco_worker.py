from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import measurement_mujoco_worker
from blueprint_pipeline.measurement_adapter_execution import (
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
)
from blueprint_pipeline.measurement_mujoco_worker import (
    execute_mujoco_benchmark_case,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    MeasurementBenchmarkError,
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
    build_r5_stage_data,
    build_sealed_physical_label,
    evaluate_qualification_benchmark,
)

mujoco = pytest.importorskip("mujoco")


SHA_DEV = "sha256:" + "1" * 64
SHA_QUAL = "sha256:" + "2" * 64
SHA_CAPTURE = "sha256:" + "3" * 64
SHA_ROBOT = "sha256:" + "4" * 64
SHA_INPUT = "sha256:" + "5" * 64

WORKER_ARGV = [sys.executable, "-m", "blueprint_pipeline.measurement_mujoco_worker"]
HALF_EXTENT = 0.05


def _worker_implementation_digest() -> str:
    source = Path(measurement_mujoco_worker.__file__).read_bytes()
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _spec() -> dict:
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-dry-run-1",
        method_ids=["mujoco-3"],
        development_split_digest=SHA_DEV,
        qualification_split_digest=SHA_QUAL,
        capture_bundle_digests=[SHA_CAPTURE],
        robot_controller_digests=[SHA_ROBOT],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 0.5,
            "maximum_mismatch_rate": 0.5,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 0.2,
        },
        compute_budget={"usd": 0.0},
    )


def _case(spec: dict, case_id: str, *, drop_height: float, scene: str = "box_settle") -> dict:
    return build_benchmark_case_manifest(
        spec,
        case_id=case_id,
        split="development",
        input_artifact_digests=[SHA_INPUT],
        task_class="rigid_pick_place",
        material_regime="rigid",
        operating_point={
            "scene": scene,
            "drop_height_m": drop_height,
            "box_half_extent_m": HALF_EXTENT,
            "friction": 0.8,
        },
    )


def _request(
    spec: dict,
    case: dict,
    *,
    execution_id: str,
    engine_version_policy: str = "record_actual_development_only",
) -> dict:
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("mujoco-3"),
        spec,
        case,
        execution_id=execution_id,
        implementation_id=measurement_mujoco_worker.MUJOCO_WORKER_ID,
        implementation_version=measurement_mujoco_worker.MUJOCO_WORKER_VERSION,
        implementation_digest=_worker_implementation_digest(),
        backend_id="mujoco_cpu",
        precision="float64",
        seed=0,
        solver_settings={
            "timestep": 0.002,
            "steps": 600,
            "engine_version_policy": engine_version_policy,
        },
        timeout_seconds=120,
    )


def test_real_mujoco_physics_executes_through_the_fail_closed_boundary() -> None:
    spec = _spec()
    case = _case(spec, "dev-box-settle-030", drop_height=0.3)
    bundle = run_measurement_adapter_execution(
        _request(spec, case, execution_id="mujoco-dev-exec-1"),
        command_argv=WORKER_ARGV,
        execute=True,
    )
    receipt = bundle["receipt"]
    assert receipt["status"] == "completed"
    assert receipt["evidence_class"] == "development_execution"
    observations = receipt["runtime_observations"]
    assert observations["engine_version"] == mujoco.__version__
    assert observations["deterministic_rerun_match"] is True
    assert observations["rendering_used"] is False
    metrics = bundle["worker_result"]["observed_metrics"]
    # Real dynamics: the box dropped from 0.3 m must settle at its half-extent
    # height on the plane, with contact only after the free fall. MuJoCo's
    # soft-contact model produces a transient impact penetration spike (about
    # 15 mm here); it must stay well under the box half-extent.
    assert abs(metrics["final_object_pose_error"] - HALF_EXTENT) < 0.005
    assert metrics["contact_sequence"] > 0
    assert 0.0 <= metrics["penetration"] < HALF_EXTENT
    prediction = bundle["prediction"]
    assert prediction is not None
    assert prediction["candidate_id"] == "mujoco-3"
    assert prediction["physical_success_established"] is False
    assert bundle["qualification_created"] is False
    assert bundle["production_route_created"] is False


def test_engine_version_pin_is_explicit_and_fail_closed() -> None:
    spec = _spec()
    case = _case(spec, "dev-box-settle-pin", drop_height=0.2)
    result = execute_mujoco_benchmark_case(
        _request(
            spec, case,
            execution_id="mujoco-dev-exec-pin",
            engine_version_policy="require_target",
        )
    )
    target = build_measurement_adapter_descriptor("mujoco-3")["target_version"]
    if mujoco.__version__ == target:
        assert result["status"] == "completed"
    else:
        assert result["status"] == "blocked"
        assert any(
            code.startswith("engine_version_mismatch:")
            for code in result["failure_codes"]
        )
    no_policy_request = build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("mujoco-3"),
        spec,
        _case(spec, "dev-box-settle-nopolicy", drop_height=0.2),
        execution_id="mujoco-dev-exec-nopolicy",
        implementation_id=measurement_mujoco_worker.MUJOCO_WORKER_ID,
        implementation_version=measurement_mujoco_worker.MUJOCO_WORKER_VERSION,
        implementation_digest=_worker_implementation_digest(),
        backend_id="mujoco_cpu",
        precision="float64",
        seed=0,
        solver_settings={"timestep": 0.002, "steps": 600},
        timeout_seconds=120,
    )
    missing_policy = execute_mujoco_benchmark_case(no_policy_request)
    assert missing_policy["status"] == "blocked"
    assert "engine_version_policy_missing_or_invalid" in missing_policy["failure_codes"]


def test_unknown_scene_and_out_of_bounds_inputs_block_with_typed_codes() -> None:
    spec = _spec()
    unknown = execute_mujoco_benchmark_case(
        _request(
            spec,
            _case(spec, "dev-unknown-scene", drop_height=0.2, scene="teleport_pad"),
            execution_id="mujoco-dev-exec-unknown",
        )
    )
    assert unknown["status"] == "blocked"
    assert "scene_unknown:teleport_pad" in unknown["failure_codes"]

    out_of_bounds_case = build_benchmark_case_manifest(
        spec,
        case_id="dev-too-high",
        split="development",
        input_artifact_digests=[SHA_INPUT],
        task_class="rigid_pick_place",
        material_regime="rigid",
        operating_point={
            "scene": "box_settle",
            "drop_height_m": 50.0,
            "box_half_extent_m": HALF_EXTENT,
            "friction": 0.8,
        },
    )
    blocked = execute_mujoco_benchmark_case(
        _request(spec, out_of_bounds_case, execution_id="mujoco-dev-exec-oob")
    )
    assert blocked["status"] == "blocked"
    assert "operating_point_out_of_bounds:drop_height_m" in blocked["failure_codes"]


def test_repeated_execution_is_bit_identical() -> None:
    spec = _spec()
    case = _case(spec, "dev-box-settle-repeat", drop_height=0.25)
    request = _request(spec, case, execution_id="mujoco-dev-exec-repeat")
    first = execute_mujoco_benchmark_case(request)
    second = execute_mujoco_benchmark_case(request)
    assert first["status"] == "completed"
    assert first["observed_metrics"] == second["observed_metrics"]
    assert (
        first["runtime_observations"]["trajectory_digest"]
        == second["runtime_observations"]["trajectory_digest"]
    )


def test_development_report_evaluates_but_can_never_mint_r5_evidence() -> None:
    spec = _spec()
    predictions = []
    labels = []
    for index, drop_height in enumerate((0.2, 0.3)):
        case = _case(spec, f"dev-report-case-{index}", drop_height=drop_height)
        bundle = run_measurement_adapter_execution(
            _request(spec, case, execution_id=f"mujoco-dev-exec-report-{index}"),
            command_argv=WORKER_ARGV,
            execute=True,
        )
        assert bundle["receipt"]["status"] == "completed"
        prediction = bundle["prediction"]
        predictions.append(prediction)
        labels.append(
            build_sealed_physical_label(
                case,
                expected_metrics={
                    "final_object_pose_error": HALF_EXTENT,
                    "penetration": 0.0,
                    "contact_sequence": prediction["observed_metrics"]["contact_sequence"],
                },
                unsafe_condition_observed=False,
                physical_measurement_ids=[f"physical-measurement-{index}"],
                independent_evaluator_id="blueprint-independent-evaluator",
            )
        )
    report = evaluate_qualification_benchmark(
        spec,
        predictions,
        labels,
        evaluator_id="blueprint-independent-evaluator",
        independent_execution=False,
    )
    assert report["case_count"] == 2
    assert report["thresholds_passed"] is True
    assert report["evidence_status"] == "development_only_not_qualification"
    assert report["production_route_eligible"] is False
    with pytest.raises(
        MeasurementBenchmarkError,
        match="measurement_benchmark_report_not_r5_candidate",
    ):
        build_r5_stage_data(report)
