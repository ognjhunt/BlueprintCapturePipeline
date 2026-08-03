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
from blueprint_pipeline.measurement_geometry_contact_articulation_development_suite import (
    ArticulationDevelopmentSuiteError,
    run_capture_to_geometry_contact_articulation_development_suite,
    validate_capture_to_geometry_contact_articulation_development_suite,
)
from blueprint_pipeline.measurement_mujoco_articulation_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
    run_mujoco_articulation_measurement_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = (
    ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_articulation_v1/corpus.json"
)
CORPUS_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_contact_articulation_development_corpus.v1.schema.json"
)
SUITE_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_contact_articulation_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"articulation-development-no-controller").hexdigest()
)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _request(case_index: int = 0) -> dict:
    corpus = _corpus()
    corpus_digest = _corpus_digest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-mujoco-articulation-1",
        method_ids=["mujoco-3"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 0.05,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 90},
        minimum_repeated_trials=2,
    )
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    operating_point = {**corpus["shared_operating_point"], **row}
    case = build_benchmark_case_manifest(
        spec,
        case_id=case_id,
        split="development",
        input_artifact_digests=[corpus_digest],
        task_class=(
            "articulated_door_opening"
            if operating_point["articulation_type"] == "door_hinge"
            else "articulated_drawer_opening"
        ),
        material_regime="synthetic_rigid_articulation",
        operating_point=operating_point,
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("mujoco-3"),
        spec,
        case,
        execution_id=f"mujoco-articulation-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="mujoco-cpu-articulation",
        precision="float64",
        seed=31,
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


def test_articulation_corpus_schema_forces_synthetic_noninstrumented_scope() -> None:
    corpus = _corpus()
    jsonschema.validate(
        corpus,
        json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    assert {row["articulation_type"] for row in corpus["cases"]} == {
        "door_hinge",
        "drawer_slide",
    }
    assert corpus["development_only"] is True
    for key in (
        "held_out",
        "physical_measurements_included",
        "instrumented_force_included",
        "qualification_labels_included",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
    ):
        assert corpus[key] is False


def test_real_mujoco_articulation_suite_executes_door_and_drawer() -> None:
    planned = run_capture_to_geometry_contact_articulation_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_capture_to_geometry_contact_articulation_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"]["within_envelope_case_count"] == 2
    assert completed["aggregate_metrics"]["maximum_travel_error"] < 0.01
    assert completed["aggregate_metrics"]["joint_limit_reached_case_count"] == 0
    assert all(row["unsafe_condition_predicted"] is False for row in completed["cases"])
    jsonschema.validate(
        completed,
        json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_articulation_worker_rejects_physical_scope_and_solver_tampering() -> None:
    physical = copy.deepcopy(_request())
    physical["case_manifest"]["operating_point"]["physical_force_measurement"] = True
    _rehash(physical["case_manifest"], "case_manifest_digest")
    physical.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="physical_force_scope_invalid"):
        run_mujoco_articulation_measurement_request(physical)

    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["solver"] = "CG"
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_invalid"):
        run_mujoco_articulation_measurement_request(solver)


def test_articulation_implementation_identity_is_distinct_from_rigid_worker() -> None:
    request = _request()
    request["implementation"]["implementation_id"] = "blueprint-mujoco-rigid-development-adapter"
    request.pop("execution_request_digest")
    result = run_mujoco_articulation_measurement_request(request)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["mujoco_articulation_implementation_id_mismatch"]


def test_articulation_suite_rejects_split_and_authority_tampering() -> None:
    with pytest.raises(ArticulationDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_geometry_contact_articulation_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )
    planned = run_capture_to_geometry_contact_articulation_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["instrumented_force_included"] = True
    planned["r7_admission"] = True
    with pytest.raises(
        ArticulationDevelopmentSuiteError,
        match="instrumented_force_included|r7_admission",
    ):
        validate_capture_to_geometry_contact_articulation_development_suite(planned)
