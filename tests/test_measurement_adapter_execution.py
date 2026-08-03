from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    _safe_environment,
    build_measurement_adapter_execution_request,
    build_measurement_adapter_worker_result,
    run_measurement_adapter_execution,
    validate_measurement_adapter_execution_bundle,
    validate_measurement_adapter_execution_request,
)
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
)
from blueprint_pipeline.measurement_mujoco_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


D = ["sha256:" + char * 64 for char in "abcdef"]


def _spec() -> dict:
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-1",
        method_ids=["mujoco-3"],
        development_split_digest=D[0],
        qualification_split_digest=D[1],
        capture_bundle_digests=[D[2]],
        robot_controller_digests=[D[3]],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 1.0,
            "maximum_harmful_false_negative_rate": 1.0,
            "minimum_coverage": 0.0,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 30},
        minimum_repeated_trials=2,
    )


def _case(*, split: str = "development") -> dict:
    return build_benchmark_case_manifest(
        _spec(),
        case_id=f"mujoco-rigid-drop-{split}",
        split=split,
        input_artifact_digests=[D[4]],
        task_class="rigid_pick_place",
        material_regime="rigid",
        operating_point={
            "adapter_protocol": "mujoco_rigid_drop.v1",
            "body_shape": "sphere",
            "radius_m": 0.05,
            "mass_kg": 0.2,
            "initial_height_m": 0.5,
            "duration_s": 1.0,
            "gravity_m_s2": -9.81,
            "timestep_s": 0.001,
            "friction": [0.8, 0.005, 0.0001],
            "penetration_unsafe_threshold_m": 0.002,
        },
    )


def _request(*, case: dict | None = None) -> dict:
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("mujoco-3"),
        _spec(),
        case or _case(),
        execution_id="mujoco-rigid-drop-development",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="mujoco-cpu",
        precision="float64",
        seed=7,
        solver_settings={
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": 100,
            "tolerance": 1e-10,
        },
        timeout_seconds=30,
    )


def _command() -> list[str]:
    return [sys.executable, "-m", "blueprint_pipeline.measurement_mujoco_adapter"]


def test_plan_only_is_inert_and_cannot_create_a_prediction() -> None:
    bundle = run_measurement_adapter_execution(_request(), command_argv=_command(), execute=False)
    assert bundle["receipt"]["status"] == "planned_not_executed"
    assert bundle["receipt"]["evidence_class"] == "plan_only"
    assert bundle["receipt"]["failure_codes"] == ["explicit_execution_gate_not_set"]
    assert bundle["prediction"] is None
    assert bundle["qualification_created"] is False
    assert bundle["catalog_mutated"] is False


def test_subprocess_environment_isolates_home_and_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/operator-home-must-not-leak")
    monkeypatch.setenv("XDG_CACHE_HOME", "/operator-cache-must-not-leak")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_adapter_execution.platform.system",
        lambda: "Linux",
    )

    environment = _safe_environment(tmp_path)

    assert environment["HOME"] == str(tmp_path)
    assert environment["XDG_CACHE_HOME"] == str(tmp_path / ".cache")
    assert environment["TMPDIR"] == str(tmp_path)
    assert environment["MUJOCO_GL"] == "disable"

    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert _safe_environment(tmp_path)["MUJOCO_GL"] == "disable"


def test_real_mujoco_311_development_adapter_executes_and_replays() -> None:
    bundle = run_measurement_adapter_execution(_request(), command_argv=_command(), execute=True)
    receipt = bundle["receipt"]
    assert receipt["status"] == "completed"
    assert receipt["evidence_class"] == "development_execution"
    assert receipt["runtime_observations"]["engine_version"] == "3.11.0"
    assert receipt["runtime_observations"]["deterministic_replay_match"] is True
    assert receipt["stdout_content_persisted"] is False
    assert receipt["stderr_content_persisted"] is False
    assert receipt["production_route_eligible"] is False
    prediction = bundle["prediction"]
    assert prediction["execution_receipt_digest"] == receipt["execution_receipt_digest"]
    assert prediction["observed_metrics"]["contact_sequence"] == "ground_contact"
    assert prediction["observed_metrics"]["penetration"] >= 0.0
    assert prediction["physical_success_established"] is False


def test_qualification_split_requires_an_independently_controlled_runner() -> None:
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="qualification_execution_requires_independent_runner",
    ):
        _request(case=_case(split="qualification"))


def test_shell_launch_and_rehashed_target_version_drift_fail_closed() -> None:
    with pytest.raises(MeasurementAdapterExecutionError, match="shell_forbidden"):
        run_measurement_adapter_execution(
            _request(), command_argv=["bash", "-lc", "true"], execute=True
        )
    drifted = copy.deepcopy(_request())
    drifted.pop("execution_request_digest")
    drifted["runtime_configuration"]["target_engine_version"] = "3.10.0"
    with pytest.raises(MeasurementAdapterExecutionError, match="target_version_mismatch"):
        validate_measurement_adapter_execution_request(drifted)


def test_worker_result_refuses_sensitive_output_and_unknown_metrics() -> None:
    request = _request()
    with pytest.raises(MeasurementAdapterExecutionError, match="sensitive_output"):
        build_measurement_adapter_worker_result(
            request,
            status="completed",
            observed_metrics={"penetration": 0.0},
            unsafe_condition_predicted=False,
            runtime_observations={
                "engine_version": "3.11.0",
                "backend_id": "mujoco-cpu",
                "precision": "float64",
                "seed": 7,
                "api_key": "must-not-persist",
            },
        )
    with pytest.raises(MeasurementAdapterExecutionError, match="metric_unknown"):
        build_measurement_adapter_worker_result(
            request,
            status="completed",
            observed_metrics={"invented_metric": 1.0},
            unsafe_condition_predicted=False,
            runtime_observations={
                "engine_version": "3.11.0",
                "backend_id": "mujoco-cpu",
                "precision": "float64",
                "seed": 7,
            },
        )


def test_execution_contracts_match_schema_and_tampering_is_detected() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/measurement_adapter_execution.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    bundle = run_measurement_adapter_execution(_request(), command_argv=_command(), execute=True)
    for artifact in (bundle["request"], bundle["worker_result"], bundle["receipt"], bundle):
        jsonschema.validate(artifact, schema)
    tampered = copy.deepcopy(bundle)
    tampered.pop("execution_bundle_digest")
    tampered["receipt"]["case_manifest_digest"] = D[5]
    with pytest.raises(
        MeasurementAdapterExecutionError, match="nested_artifact_invalid|binding_mismatch"
    ):
        validate_measurement_adapter_execution_bundle(tampered)
