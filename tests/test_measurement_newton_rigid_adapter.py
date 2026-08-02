from __future__ import annotations

import copy
import hashlib
import importlib.metadata
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
from blueprint_pipeline.measurement_geometry_contact_cross_engine_development_suite import (
    CrossEngineGeometryContactSuiteError,
    run_capture_to_geometry_contact_cross_engine_development_suite,
    validate_capture_to_geometry_contact_cross_engine_development_suite,
)
from blueprint_pipeline.measurement_newton_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    NEWTON_VERSION,
    PROTOCOL_ID,
    WARP_VERSION,
    implementation_digest,
    run_newton_rigid_measurement_request,
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
    ROOT / "docs/schemas/capture_to_geometry_contact_cross_engine_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"cross-engine-rigid-contact-development-no-controller").hexdigest()
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
        benchmark_version="development-cross-engine-rigid-drop-1",
        method_ids=["mujoco-3", "newton-1-4"],
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 300},
        minimum_repeated_trials=2,
    )
    row = dict(corpus["cases"][0])
    case_id = row.pop("case_id")
    pair_binding = {
        "corpus_digest": corpus_digest,
        "case_id": case_id,
        "shared_operating_point": corpus["shared_operating_point"],
        "case_operating_point": row,
    }
    encoded = json.dumps(pair_binding, sort_keys=True, separators=(",", ":")).encode()
    pair_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    case = build_benchmark_case_manifest(
        spec,
        case_id=f"{case_id}--newton-1-4",
        split="development",
        input_artifact_digests=[corpus_digest, pair_digest],
        task_class="rigid_pick_place",
        material_regime="synthetic_rigid_body_drop",
        operating_point={
            **corpus["shared_operating_point"],
            "adapter_protocol": PROTOCOL_ID,
            **row,
        },
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("newton-1-4"),
        spec,
        case,
        execution_id="newton-rigid-development-001",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="newton-warp-cpu-xpbd",
        precision="float32",
        seed=41,
        solver_settings={
            "solver": "XPBD",
            "iterations": 10,
            "rigid_contact_relaxation": 0.8,
            "deterministic_mode": "RUN_TO_RUN",
        },
        timeout_seconds=120,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_shared_rigid_corpus_and_newton_runtime_are_exact_and_available() -> None:
    corpus = _corpus()
    jsonschema.validate(
        corpus,
        json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    assert corpus["shared_operating_point"]["protocol_family"] == "rigid_body_drop"
    assert "adapter_protocol" not in corpus["shared_operating_point"]
    assert importlib.metadata.version("newton") == NEWTON_VERSION
    assert importlib.metadata.version("warp-lang") == WARP_VERSION
    descriptor = build_measurement_adapter_descriptor("newton-1-4")
    probe = probe_measurement_adapter(descriptor)
    assert probe["status"] == "available"
    assert {row["name"]: row["observed_version"] for row in probe["probes"]} == {
        "newton": NEWTON_VERSION,
        "warp-lang": WARP_VERSION,
    }


def test_cross_engine_suite_executes_same_cases_and_preserves_deltas() -> None:
    planned = run_capture_to_geometry_contact_cross_engine_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_capture_to_geometry_contact_cross_engine_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_pair_count"] == 2
    assert completed["method_execution_count"] == 4
    assert completed["all_methods_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_cross_engine_deltas"]["contact_sequence_match_count"] == 2
    assert completed["aggregate_cross_engine_deltas"]["unsafe_prediction_match_count"] == 2
    assert completed["aggregate_cross_engine_deltas"]["maximum_absolute_penetration_delta_m"] > 0.01
    assert all(
        set(row["method_results"]) == {"mujoco-3", "newton-1-4"} for row in completed["case_pairs"]
    )
    jsonschema.validate(
        completed,
        json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_newton_worker_rejects_protocol_and_solver_tampering() -> None:
    protocol = copy.deepcopy(_request())
    protocol["case_manifest"]["operating_point"]["adapter_protocol"] = "mujoco_rigid_drop.v1"
    _rehash(protocol["case_manifest"], "case_manifest_digest")
    protocol.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="protocol_invalid"):
        run_newton_rigid_measurement_request(protocol)

    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["deterministic_mode"] = "NOT_GUARANTEED"
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="deterministic_mode_invalid"):
        run_newton_rigid_measurement_request(solver)


def test_newton_implementation_identity_is_distinct_from_mujoco() -> None:
    request = _request()
    request["implementation"]["implementation_id"] = "blueprint-mujoco-rigid-development-adapter"
    request.pop("execution_request_digest")
    result = run_newton_rigid_measurement_request(request)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["newton_rigid_implementation_id_mismatch"]


def test_cross_engine_suite_rejects_split_and_authority_tampering() -> None:
    with pytest.raises(CrossEngineGeometryContactSuiteError, match="split_leakage"):
        run_capture_to_geometry_contact_cross_engine_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )
    planned = run_capture_to_geometry_contact_cross_engine_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["r7_admission"] = True
    planned["physical_measurements_included"] = True
    with pytest.raises(
        CrossEngineGeometryContactSuiteError,
        match="r7_admission|physical_measurements_included",
    ):
        validate_capture_to_geometry_contact_cross_engine_development_suite(planned)
