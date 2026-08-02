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
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
    probe_measurement_adapter,
)
from blueprint_pipeline.measurement_deformation_cable_development_suite import (
    CableDevelopmentSuiteError,
    run_capture_to_deformation_cable_development_suite,
    validate_capture_to_deformation_cable_development_suite,
)
from blueprint_pipeline.measurement_pyelastica_cable_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
    run_pyelastica_cable_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_deformation_cable_v1/corpus.json"
CORPUS_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_deformation_cable_development_corpus.v1.schema.json"
)
SUITE_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_deformation_cable_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:"
    + hashlib.sha256(b"capture-to-deformation-cable-development-no-controller").hexdigest()
)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _spec() -> dict:
    corpus_digest = _corpus_digest()
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-pyelastica-cable-1",
        method_ids=["pyelastica"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 0.5,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 120},
        minimum_repeated_trials=2,
        lane="cable",
    )


def _case(case_index: int = 0) -> dict:
    corpus = _corpus()
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    return build_benchmark_case_manifest(
        _spec(),
        case_id=case_id,
        split="development",
        input_artifact_digests=[_corpus_digest()],
        task_class="cable_hose_routing",
        material_regime="rope_cable_hose",
        operating_point={**corpus["shared_operating_point"], **row},
    )


def _request(case_index: int = 0) -> dict:
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("pyelastica"),
        _spec(),
        _case(case_index),
        execution_id=f"pyelastica-cable-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="pyelastica-numba-cpu",
        precision="float64",
        seed=23,
        solver_settings={"integrator": "PositionVerlet", "replay_count": 2},
        timeout_seconds=90,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_cable_corpus_is_schema_valid_and_explicitly_nonqualifying() -> None:
    corpus = _corpus()
    jsonschema.validate(
        corpus,
        json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    assert corpus["lane"] == "cable"
    assert corpus["development_only"] is True
    assert corpus["synthetic_fixture"] is True
    for key in (
        "held_out",
        "physical_measurements_included",
        "qualification_labels_included",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
    ):
        assert corpus[key] is False


def test_pyelastica_probe_is_exactly_version_bound_without_qualification() -> None:
    descriptor = build_measurement_adapter_descriptor("pyelastica")
    probe = probe_measurement_adapter(descriptor)
    assert descriptor["target_version"] == "0.3.3.post2"
    assert probe["status"] == "available"
    assert probe["observed_versions"] == ["0.3.3.post2"]
    assert probe["target_version_observed"] is True
    assert probe["package_imported"] is False
    assert probe["process_launched"] is False
    assert probe["qualification_established"] is False
    assert probe["production_route_eligible"] is False


def test_real_pyelastica_corpus_executes_two_deterministic_cable_trials() -> None:
    planned = run_capture_to_deformation_cable_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=False,
    )
    assert planned["status"] == "planned_not_executed"
    assert planned["aggregate_metrics"] == {}

    completed = run_capture_to_deformation_cable_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert 0 < completed["aggregate_metrics"]["mean_tip_displacement_m"] < 0.1
    assert completed["aggregate_metrics"]["maximum_tip_displacement_m"] < 0.1
    assert completed["aggregate_metrics"]["maximum_segment_strain"] < 0.01
    assert all(
        row["observed_metrics"]["task_outcome"] == "within_deformation_envelope"
        for row in completed["cases"]
    )
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


def test_cable_worker_rejects_frame_timestep_and_solver_tampering() -> None:
    frame = copy.deepcopy(_request())
    frame["case_manifest"]["operating_point"]["direction"] = [0.0, 1.0, 0.0]
    _rehash(frame["case_manifest"], "case_manifest_digest")
    frame.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="frame_convention_invalid",
    ):
        run_pyelastica_cable_request(frame)

    timestep = copy.deepcopy(_request())
    timestep["case_manifest"]["operating_point"]["duration_s"] = 0.050001
    _rehash(timestep["case_manifest"], "case_manifest_digest")
    timestep.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="timestep_duration_mismatch",
    ):
        run_pyelastica_cable_request(timestep)

    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["integrator"] = "RK4"
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="integrator_invalid",
    ):
        run_pyelastica_cable_request(solver)


def test_cable_suite_rejects_split_leakage_and_authority_tampering() -> None:
    with pytest.raises(CableDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_deformation_cable_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
            execute=False,
        )
    planned = run_capture_to_deformation_cable_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=False,
    )
    planned["r7_admission"] = True
    with pytest.raises(CableDevelopmentSuiteError, match="r7_admission"):
        validate_capture_to_deformation_cable_development_suite(planned)
