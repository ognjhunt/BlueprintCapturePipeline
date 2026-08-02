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
from blueprint_pipeline.measurement_geometry_contact_development_suite import (
    GeometryContactDevelopmentSuiteError,
    run_capture_to_geometry_contact_development_suite,
    validate_capture_to_geometry_contact_development_suite,
)
from blueprint_pipeline.measurement_mujoco_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    implementation_digest,
    run_mujoco_measurement_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
CORPUS_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_contact_development_corpus.v1.schema.json"
)
SUITE_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_contact_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"geometry-contact-development-no-controller").hexdigest()
)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _request() -> dict:
    corpus = _corpus()
    corpus_digest = _corpus_digest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-mujoco-rigid-contact-1",
        method_ids=["mujoco-3"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 90},
        minimum_repeated_trials=2,
    )
    row = dict(corpus["cases"][0])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        spec,
        case_id=case_id,
        split="development",
        input_artifact_digests=[corpus_digest],
        task_class="rigid_pick_place",
        material_regime="synthetic_rigid_body_drop",
        operating_point={
            **corpus["shared_operating_point"],
            "adapter_protocol": PROTOCOL_ID,
            **row,
        },
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("mujoco-3"),
        spec,
        case,
        execution_id="mujoco-rigid-contact-development-001",
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
        timeout_seconds=45,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_geometry_contact_corpus_schema_forces_development_authority() -> None:
    corpus = _corpus()
    jsonschema.validate(
        corpus,
        json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    assert corpus["lane"] == "rigid_contact"
    assert corpus["shared_operating_point"]["protocol_family"] == "rigid_body_drop"
    assert {row["body_shape"] for row in corpus["cases"]} == {"sphere", "box"}
    assert corpus["development_only"] is True
    assert corpus["synthetic_fixture"] is True
    for key in (
        "held_out",
        "physical_measurements_included",
        "qualification_labels_included",
        "instrumented_contact_included",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
    ):
        assert corpus[key] is False


def test_geometry_contact_suite_plans_and_executes_both_shapes() -> None:
    planned = run_capture_to_geometry_contact_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=False,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_capture_to_geometry_contact_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert {row["body_shape"] for row in completed["cases"]} == {"sphere", "box"}
    assert completed["aggregate_metrics"]["ground_contact_case_count"] == 2
    assert 0 < completed["aggregate_metrics"]["maximum_penetration_m"] < 0.03
    assert 0 < completed["aggregate_metrics"]["mean_first_contact_time_s"] < 1
    assert all(row["unsafe_condition_predicted"] is False for row in completed["cases"])
    jsonschema.validate(
        completed,
        json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_geometry_contact_suite_rejects_split_leakage_and_authority_tampering() -> None:
    with pytest.raises(GeometryContactDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_geometry_contact_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )
    planned = run_capture_to_geometry_contact_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["instrumented_contact_included"] = True
    planned["r7_admission"] = True
    with pytest.raises(
        GeometryContactDevelopmentSuiteError,
        match="instrumented_contact_included|r7_admission",
    ):
        validate_capture_to_geometry_contact_development_suite(planned)


def test_geometry_contact_worker_rejects_shape_and_source_identity_tampering() -> None:
    shape = copy.deepcopy(_request())
    shape["case_manifest"]["operating_point"]["body_shape"] = "capsule"
    _rehash(shape["case_manifest"], "case_manifest_digest")
    shape.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="mujoco_adapter_body_shape_invalid"):
        run_mujoco_measurement_request(shape)

    source = copy.deepcopy(_request())
    source["implementation"]["implementation_digest"] = "sha256:" + "0" * 64
    source.pop("execution_request_digest")
    result = run_mujoco_measurement_request(source)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["mujoco_adapter_implementation_digest_mismatch"]


def test_geometry_contact_suite_digest_tampering_fails_closed() -> None:
    planned = run_capture_to_geometry_contact_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["case_count"] = 3
    with pytest.raises(
        GeometryContactDevelopmentSuiteError,
        match="case_count_mismatch|digest_mismatch",
    ):
        validate_capture_to_geometry_contact_development_suite(planned)
