from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
)
from blueprint_pipeline.measurement_adapter_execution import (
    validate_measurement_adapter_execution_receipt,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    MeasurementBenchmarkError,
    build_benchmark_case_manifest,
    build_benchmark_prediction,
    build_qualification_benchmark_spec,
    build_r4_preregistration_stage_data,
    build_r5_stage_data,
    build_sealed_physical_label,
    evaluate_qualification_benchmark,
    validate_benchmark_case_manifest,
    validate_qualification_benchmark_report,
)


D = ["sha256:" + char * 64 for char in "abcdef"]


def _spec(
    benchmark_id: str = "capture-to-geometry-and-contact",
    *,
    lane: str | None = None,
) -> dict:
    methods = {
        "capture-to-geometry-and-contact": ["mujoco-3", "drake-1-55"],
        "capture-to-observation": ["direct-captured-observations", "isaac-rtx-openusd-sensor-path"],
        "capture-to-deformation": {
            "cloth": ["mujoco-3", "newton-1-4"],
            "cable": ["pyelastica", "sofa-26-06"],
            "granular": ["project-chrono-10"],
            "tactile": ["direct-captured-observations", "tacsl"],
        }[lane or "cloth"],
    }[benchmark_id]
    return build_qualification_benchmark_spec(
        benchmark_id=benchmark_id,
        benchmark_version="1",
        method_ids=methods,
        development_split_digest=D[0],
        qualification_split_digest=D[1],
        capture_bundle_digests=[D[2]],
        robot_controller_digests=[D[3]],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 0.2,
            "maximum_mismatch_rate": 0.1,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 0.1,
        },
        compute_budget={"usd": 100, "maximum_duration_seconds": 3600},
        lane=lane,
    )


def _case(spec: dict, *, split: str = "qualification", trial: int = 1) -> dict:
    return build_benchmark_case_manifest(
        spec,
        case_id=f"{split}-case-{trial}",
        split=split,
        input_artifact_digests=[D[4]],
        task_class="rigid_pick_place",
        material_regime="rigid",
        operating_point={"trial": trial},
    )


def _receipt(descriptor: dict, case: dict) -> dict:
    independent = case["split"] == "qualification"
    receipt = {
        "schema_version": "measurement_adapter_execution_receipt.v1",
        "execution_id": f"execution-{case['case_id']}",
        "execution_request_digest": D[5],
        "candidate_id": descriptor["candidate_id"],
        "adapter_descriptor_digest": descriptor["adapter_descriptor_digest"],
        "benchmark_spec_digest": case["benchmark_spec_digest"],
        "case_manifest_digest": case["case_manifest_digest"],
        "split": case["split"],
        "status": "completed",
        "evidence_class": (
            "independent_qualification_execution" if independent else "development_execution"
        ),
        "executor_id": "independent-execution-lab" if independent else "dev-runner",
        "executor_independent_of_candidate": independent,
        "clean_environment_verified": independent,
        "immutable_runtime_identity_verified": independent,
        "command_digest": D[0],
        "command_executable": "fixture-worker",
        "command_argc": 5,
        "started_at": "2026-08-02T10:00:00+00:00",
        "finished_at": "2026-08-02T10:00:01+00:00",
        "duration_seconds": 1.0,
        "exit_code": 0,
        "worker_result_digest": D[1],
        "stdout_digest": D[2],
        "stdout_bytes": 0,
        "stdout_content_persisted": False,
        "stderr_digest": D[3],
        "stderr_bytes": 0,
        "stderr_content_persisted": False,
        "runtime_observations": {"fixture": True},
        "host_runtime": {"fixture": True},
        "failure_codes": [],
        "host_process_isolation_only": not independent,
        "network_isolation_verified": independent,
        "filesystem_isolation_verified": independent,
        "secrets_persisted": False,
        "qualification_labels_accessed": False,
        "provider_spend_authorized": False,
        "physical_execution_authorized": False,
        "production_route_eligible": False,
        "r6_qualification_decision_created": False,
        "r7_catalog_admission_created": False,
        "agent_authorized": False,
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["execution_receipt_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return validate_measurement_adapter_execution_receipt(receipt)


def _prediction_and_label(spec: dict, case: dict) -> tuple[dict, dict]:
    metric_id = spec["metric_ids"][0]
    descriptor = build_measurement_adapter_descriptor(spec["method_ids"][0])
    prediction = build_benchmark_prediction(
        descriptor,
        case,
        observed_metrics={metric_id: 1.05},
        unsafe_condition_predicted=True,
        execution_receipt=_receipt(descriptor, case),
    )
    label = build_sealed_physical_label(
        case,
        expected_metrics={metric_id: 1.0},
        unsafe_condition_observed=True,
        physical_measurement_ids=["physical-measurement-1"],
        independent_evaluator_id="independent-evaluator",
    )
    return prediction, label


def test_all_three_benchmark_programs_compile_with_separate_deformation_lanes() -> None:
    geometry = _spec()
    observation = _spec("capture-to-observation")
    cloth = _spec("capture-to-deformation", lane="cloth")
    cable = _spec("capture-to-deformation", lane="cable")
    granular = _spec("capture-to-deformation", lane="granular")
    tactile = _spec("capture-to-deformation", lane="tactile")
    assert geometry["protocols"] == ["Q-KIN", "Q-RIGID", "Q-CONTACT", "Q-ART"]
    assert observation["protocols"] == ["Q-SENSOR"]
    assert cloth["lane"] == "cloth"
    assert cable["lane"] == "cable"
    assert granular["lane"] == "granular"
    assert tactile["lane"] == "tactile"
    for spec in (geometry, observation, cloth, cable, granular, tactile):
        assert spec["candidate_may_access_qualification_labels"] is False
        assert spec["vendor_may_grade_qualification"] is False
        assert spec["agent_may_approve"] is False
        assert spec["r6_human_decision_required"] is True
        assert spec["r7_catalog_admission_required"] is True


def test_case_manifest_contains_no_labels_or_physical_values() -> None:
    case = _case(_spec())
    assert case["sealed_labels_included"] is False
    assert case["physical_measurement_values_included"] is False
    leaked = copy.deepcopy(case)
    leaked.pop("case_manifest_digest")
    leaked["sealed_labels_included"] = True
    with pytest.raises(MeasurementBenchmarkError, match="sealed_label_leakage"):
        validate_benchmark_case_manifest(leaked)


def test_independent_qualification_report_is_only_an_r5_evidence_candidate() -> None:
    spec = _spec()
    pairs = [_prediction_and_label(spec, _case(spec, trial=trial)) for trial in (1, 2)]
    report = evaluate_qualification_benchmark(
        spec,
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        evaluator_id="independent-evaluator",
        independent_execution=True,
    )
    assert report["thresholds_passed"] is True
    assert report["evidence_status"] == "r5_evidence_candidate"
    assert report["metrics"]["mean_absolute_error"] == pytest.approx(0.05)
    assert report["metrics"]["harmful_false_negative_rate"] == 0.0
    assert report["confidence_intervals"]["mean_absolute_error"]["sample_size"] == 2
    assert report["independent_execution_receipts_verified"] is True
    assert report["r6_qualification_decision"] is None
    assert report["r7_catalog_admission"] is False
    assert report["production_route_eligible"] is False
    r5 = build_r5_stage_data(report)
    assert r5["vendor_graded_qualification"] is False
    assert r5["heldout_evaluation"]["qualification_split_hash"] == D[1]


def test_development_result_cannot_be_used_as_r5() -> None:
    spec = _spec("capture-to-observation")
    case = _case(spec, split="development")
    prediction, label = _prediction_and_label(spec, case)
    report = evaluate_qualification_benchmark(
        spec,
        [prediction],
        [label],
        evaluator_id="independent-evaluator",
        independent_execution=False,
    )
    assert report["evidence_status"] == "development_only_not_qualification"
    with pytest.raises(MeasurementBenchmarkError, match="not_r5_candidate"):
        build_r5_stage_data(report)


def test_harmful_false_negative_fails_even_when_numeric_error_is_small() -> None:
    spec = _spec()
    case = _case(spec, trial=1)
    prediction, label = _prediction_and_label(spec, case)
    prediction.pop("prediction_digest")
    prediction["unsafe_condition_predicted"] = False
    from blueprint_pipeline.measurement_qualification_benchmarks import (
        validate_benchmark_prediction,
    )

    prediction = validate_benchmark_prediction(prediction)
    second_prediction, second_label = _prediction_and_label(spec, _case(spec, trial=2))
    report = evaluate_qualification_benchmark(
        spec,
        [prediction, second_prediction],
        [label, second_label],
        evaluator_id="independent-evaluator",
        independent_execution=True,
    )
    assert report["metrics"]["harmful_false_negative_rate"] == 0.5
    assert report["threshold_checks"]["harmful_false_negative_rate"] is False
    assert report["thresholds_passed"] is False


def test_r4_preregistration_binds_exact_splits_methods_metrics_and_budget() -> None:
    spec = _spec("capture-to-deformation", lane="cable")
    stage = build_r4_preregistration_stage_data(spec)
    prereg = stage["frozen_benchmark_preregistration"]
    assert prereg["development_split_hash"] == D[0]
    assert prereg["qualification_split_hash"] == D[1]
    assert prereg["comparison_methods"] == ["pyelastica", "sofa-26-06"]
    assert prereg["benchmark_spec_digest"] == spec["benchmark_spec_digest"]
    assert stage["heldout_labels_exposed"] is False


def test_split_leakage_lane_mismatch_and_incomplete_joins_fail_closed() -> None:
    with pytest.raises(MeasurementBenchmarkError, match="split_leakage"):
        build_qualification_benchmark_spec(
            benchmark_id="capture-to-observation",
            benchmark_version="1",
            method_ids=["direct-captured-observations"],
            development_split_digest=D[0],
            qualification_split_digest=D[0],
            capture_bundle_digests=[D[2]],
            robot_controller_digests=[D[3]],
            acceptance_thresholds={
                "maximum_mean_absolute_error": 1,
                "maximum_mismatch_rate": 1,
                "maximum_harmful_false_negative_rate": 1,
                "minimum_coverage": 0,
            },
            compute_budget={"usd": 1},
        )
    with pytest.raises(MeasurementBenchmarkError, match="method_lane_mismatch"):
        build_qualification_benchmark_spec(
            benchmark_id="capture-to-deformation",
            benchmark_version="1",
            method_ids=["pyelastica"],
            development_split_digest=D[0],
            qualification_split_digest=D[1],
            capture_bundle_digests=[D[2]],
            robot_controller_digests=[D[3]],
            acceptance_thresholds={
                "maximum_mean_absolute_error": 1,
                "maximum_mismatch_rate": 1,
                "maximum_harmful_false_negative_rate": 1,
                "minimum_coverage": 0,
            },
            compute_budget={"usd": 1},
            lane="cloth",
        )
    spec = _spec()
    case = _case(spec)
    prediction, _ = _prediction_and_label(spec, case)
    with pytest.raises(MeasurementBenchmarkError, match="case_join_incomplete"):
        evaluate_qualification_benchmark(
            spec,
            [prediction],
            [],
            evaluator_id="independent-evaluator",
            independent_execution=True,
        )


def test_benchmark_contracts_match_checked_schema() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/measurement_qualification_benchmarks.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    spec = _spec()
    case = _case(spec)
    prediction, label = _prediction_and_label(spec, case)
    report = evaluate_qualification_benchmark(
        spec,
        [prediction, _prediction_and_label(spec, _case(spec, trial=2))[0]],
        [label, _prediction_and_label(spec, _case(spec, trial=2))[1]],
        evaluator_id="independent-evaluator",
        independent_execution=True,
    )
    for artifact in (spec, case, prediction, label, report):
        jsonschema.validate(artifact, schema)
    tampered = copy.deepcopy(report)
    tampered["r7_catalog_admission"] = True
    with pytest.raises(MeasurementBenchmarkError, match="must_be_false"):
        validate_qualification_benchmark_report(tampered)
