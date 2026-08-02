from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import jsonschema
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
from blueprint_pipeline.measurement_opencv_observation_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
    run_opencv_observation_request,
)
from blueprint_pipeline.measurement_observation_development_suite import (
    ObservationDevelopmentSuiteError,
    run_capture_to_observation_development_suite,
    validate_capture_to_observation_development_suite,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_observation_v1/corpus.json"
SCHEMA_PATH = ROOT / "docs/schemas/capture_to_observation_development_corpus.v1.schema.json"
SUITE_SCHEMA_PATH = ROOT / "docs/schemas/capture_to_observation_development_suite.v1.schema.json"
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"capture-to-observation-development-no-controller").hexdigest()
)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _spec() -> dict:
    corpus_digest = _corpus_digest()
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-observation",
        benchmark_version="development-opencv-1",
        method_ids=["direct-captured-observations"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 3 / 7,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 30},
        minimum_repeated_trials=2,
    )


def _case(case_index: int = 0) -> dict:
    corpus = _corpus()
    row = corpus["cases"][case_index]
    operating_point = {**corpus["shared_operating_point"], **row}
    operating_point.pop("case_id")
    return build_benchmark_case_manifest(
        _spec(),
        case_id=row["case_id"],
        split="development",
        input_artifact_digests=[_corpus_digest()],
        task_class="calibrated_visual_perception",
        material_regime="sensor_calibration_target",
        operating_point=operating_point,
    )


def _request(case_index: int = 0) -> dict:
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("direct-captured-observations"),
        _spec(),
        _case(case_index),
        execution_id=f"opencv-observation-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="opencv-cpu-solvepnp",
        precision="float64",
        seed=17,
        solver_settings={
            "opencv_version": "4.11.0",
            "solvepnp_flag": "SOLVEPNP_ITERATIVE",
            "replay_count": 2,
        },
        timeout_seconds=30,
    )


def _command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "blueprint_pipeline.measurement_opencv_observation_adapter",
    ]


def test_development_corpus_is_schema_valid_and_explicitly_nonqualifying() -> None:
    corpus = _corpus()
    jsonschema.validate(corpus, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    assert corpus["development_only"] is True
    assert corpus["synthetic_fixture"] is True
    assert corpus["held_out"] is False
    assert corpus["physical_measurements_included"] is False
    assert corpus["qualification_labels_included"] is False
    assert corpus["r5_evidence"] is False
    assert corpus["r6_decision"] is False
    assert corpus["r7_admission"] is False


def test_direct_observation_probe_finds_opencv_without_claiming_qualification() -> None:
    descriptor = build_measurement_adapter_descriptor("direct-captured-observations")
    probe = probe_measurement_adapter(descriptor)
    assert probe["status"] == "available"
    assert probe["observed_versions"] == ["4.11.0.86"]
    assert probe["package_imported"] is False
    assert probe["process_launched"] is False
    assert probe["qualification_established"] is False
    assert probe["production_route_eligible"] is False


def test_real_opencv_development_corpus_executes_two_deterministic_trials() -> None:
    bundles = [
        run_measurement_adapter_execution(_request(index), command_argv=_command(), execute=True)
        for index in range(2)
    ]
    assert {bundle["receipt"]["status"] for bundle in bundles} == {"completed"}
    assert {bundle["receipt"]["evidence_class"] for bundle in bundles} == {"development_execution"}
    assert all(
        bundle["receipt"]["runtime_observations"]["deterministic_replay_match"]
        for bundle in bundles
    )
    predictions = [bundle["prediction"] for bundle in bundles]
    assert [
        prediction["observed_metrics"]["missing_depth_distribution"] for prediction in predictions
    ] == [0.125, 0.25]
    assert [prediction["observed_metrics"]["temporal_error"] for prediction in predictions] == [
        2.0,
        3.5,
    ]
    assert all(
        prediction["observed_metrics"]["calibrated_image_depth_lidar_residuals"] < 0.2
        for prediction in predictions
    )
    assert all(prediction["unsafe_condition_predicted"] is False for prediction in predictions)
    assert all(bundle["qualification_created"] is False for bundle in bundles)
    assert all(bundle["production_route_created"] is False for bundle in bundles)
    assert all(bundle["physical_success_established"] is False for bundle in bundles)


def test_corpus_runner_plans_and_executes_a_nonqualifying_aggregate_suite() -> None:
    planned = run_capture_to_observation_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=False,
    )
    assert planned["status"] == "planned_not_executed"
    assert planned["all_cases_completed"] is False
    assert planned["aggregate_metrics"] == {}

    completed = run_capture_to_observation_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"]["maximum_reprojection_rmse_px"] < 0.2
    assert completed["aggregate_metrics"]["mean_missing_depth_fraction"] == 0.1875
    assert completed["aggregate_metrics"]["maximum_temporal_error_ms"] == 3.5
    for key in (
        "held_out",
        "physical_measurements_included",
        "qualification_labels_included",
        "independent_execution",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
        "production_route_eligible",
        "physical_success_established",
        "agent_may_promote",
    ):
        assert completed[key] is False
    jsonschema.validate(
        completed,
        json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )

    tampered = copy.deepcopy(completed)
    tampered["r7_admission"] = True
    with pytest.raises(ObservationDevelopmentSuiteError, match="r7_admission"):
        validate_capture_to_observation_development_suite(tampered)


def test_version_drift_calibration_tampering_and_degenerate_geometry_fail_closed() -> None:
    version_drift = copy.deepcopy(_request())
    version_drift.pop("execution_request_digest")
    version_drift["runtime_configuration"]["solver_settings"]["opencv_version"] = "4.10.0"
    settings = version_drift["runtime_configuration"]["solver_settings"]
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    version_drift["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    result = run_opencv_observation_request(version_drift)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["opencv_observation_version_mismatch"]

    noncanonical = copy.deepcopy(_request())
    noncanonical["case_manifest"]["operating_point"]["camera_matrix"][2][2] = 2.0
    noncanonical["case_manifest"].pop("case_manifest_digest")
    case = noncanonical["case_manifest"]
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    case["case_manifest_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    noncanonical.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="camera_matrix_noncanonical",
    ):
        run_opencv_observation_request(noncanonical)

    degenerate = copy.deepcopy(_request())
    for point in degenerate["case_manifest"]["operating_point"]["object_points_m"]:
        point[2] = 0.0
    degenerate["case_manifest"].pop("case_manifest_digest")
    case = degenerate["case_manifest"]
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    case["case_manifest_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    degenerate.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="object_points_degenerate",
    ):
        run_opencv_observation_request(degenerate)
