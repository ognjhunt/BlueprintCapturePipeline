from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
)
from blueprint_pipeline.measurement_adapter_runtime import build_measurement_adapter_descriptor
from blueprint_pipeline.measurement_direct_tactile_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
    run_direct_tactile_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)
from blueprint_pipeline.measurement_tactile_development_suite import (
    TactileDevelopmentSuiteError,
    run_tactile_development_suite,
    validate_tactile_development_suite,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_tactile_v1/corpus.json"
CORPUS_SCHEMA = ROOT / "docs/schemas/capture_to_tactile_development_corpus.v1.schema.json"
SUITE_SCHEMA = ROOT / "docs/schemas/capture_to_tactile_development_suite.v1.schema.json"
Q_DIGEST = "sha256:" + "f" * 64
C_DIGEST = "sha256:" + "e" * 64


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _spec() -> dict:
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-direct-tactile-1",
        method_ids=["direct-captured-observations"],
        development_split_digest=_corpus_digest(),
        qualification_split_digest=Q_DIGEST,
        capture_bundle_digests=[_corpus_digest()],
        robot_controller_digests=[C_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 4 / 6,
        },
        compute_budget={"usd": 0.0},
        lane="tactile",
    )


def _request(index: int = 0) -> dict:
    corpus = _corpus()
    row = dict(corpus["cases"][index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        _spec(),
        case_id=case_id,
        split="development",
        input_artifact_digests=[_corpus_digest()],
        task_class="tactile_manipulation",
        material_regime="elastomer",
        operating_point={**corpus["shared_operating_point"], **row},
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("direct-captured-observations"),
        _spec(),
        case,
        execution_id=f"direct-tactile-development-{index + 1}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="numpy-direct-tactile-sequence-reduction",
        precision="float64",
        seed=37,
        solver_settings={
            "analysis_method": "deterministic_sequence_reduction",
            "numpy_version": corpus["numpy_version"],
            "replay_count": 2,
        },
        timeout_seconds=30,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_tactile_corpus_schema_forces_synthetic_nonqualification_scope() -> None:
    corpus = _corpus()
    jsonschema.validate(corpus, json.loads(CORPUS_SCHEMA.read_text(encoding="utf-8")))
    assert corpus["lane"] == "tactile"
    assert corpus["shared_operating_point"]["calibration_scope"] == "synthetic_identity_only"
    for key in (
        "held_out",
        "physical_measurements_included",
        "real_sensor_calibration_included",
        "qualification_labels_included",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
    ):
        assert corpus[key] is False


def test_direct_tactile_suite_executes_stable_and_slip_sequences() -> None:
    planned = run_tactile_development_suite(
        CORPUS_PATH,
        qualification_split_digest=Q_DIGEST,
        controller_scope_digest=C_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_tactile_development_suite(
        CORPUS_PATH,
        qualification_split_digest=Q_DIGEST,
        controller_scope_digest=C_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"]["slip_case_count"] == 1
    assert [row["observed_metrics"]["task_outcome"] for row in completed["cases"]] == [
        "stable_contact_observed",
        "incipient_slip_observed",
    ]
    assert completed["cases"][1]["slip_onset_frame"] == 3
    for key in (
        "physical_measurements_included",
        "real_sensor_calibration_included",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
        "production_route_eligible",
        "physical_success_established",
        "agent_may_promote",
    ):
        assert completed[key] is False
    jsonschema.validate(completed, json.loads(SUITE_SCHEMA.read_text(encoding="utf-8")))


def test_tactile_worker_rejects_origin_timestamp_and_analysis_tampering() -> None:
    origin = copy.deepcopy(_request())
    origin["case_manifest"]["operating_point"]["data_origin"] = "real_sensor"
    _rehash(origin["case_manifest"], "case_manifest_digest")
    origin.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="data_origin_invalid"):
        run_direct_tactile_request(origin)

    timestamps = copy.deepcopy(_request())
    timestamps["case_manifest"]["operating_point"]["timestamps_ns"][2] = 1010000000
    _rehash(timestamps["case_manifest"], "case_manifest_digest")
    timestamps.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="timestamps_invalid"):
        run_direct_tactile_request(timestamps)

    analysis = copy.deepcopy(_request())
    analysis["runtime_configuration"]["solver_settings"]["analysis_method"] = "learned_grader"
    _rehash(analysis["runtime_configuration"]["solver_settings"], "solver_settings_digest")
    analysis["runtime_configuration"]["solver_settings_digest"] = analysis["runtime_configuration"][
        "solver_settings"
    ].pop("solver_settings_digest")
    analysis.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="analysis_method_invalid"):
        run_direct_tactile_request(analysis)


def test_tactile_suite_rejects_split_and_physical_authority_tampering() -> None:
    with pytest.raises(TactileDevelopmentSuiteError, match="split_leakage"):
        run_tactile_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=C_DIGEST,
        )
    planned = run_tactile_development_suite(
        CORPUS_PATH,
        qualification_split_digest=Q_DIGEST,
        controller_scope_digest=C_DIGEST,
    )
    planned["real_sensor_calibration_included"] = True
    with pytest.raises(TactileDevelopmentSuiteError, match="real_sensor_calibration_included"):
        validate_tactile_development_suite(planned)


def test_tactile_implementation_identity_cannot_inherit_opencv_identity() -> None:
    request = _request()
    request["implementation"]["implementation_id"] = (
        "blueprint-opencv-calibrated-observation-development-adapter"
    )
    request.pop("execution_request_digest")
    result = run_direct_tactile_request(request)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["direct_tactile_implementation_id_mismatch"]
