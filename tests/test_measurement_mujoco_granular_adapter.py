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
)
from blueprint_pipeline.measurement_deformation_granular_development_suite import (
    GranularDevelopmentSuiteError,
    run_capture_to_deformation_granular_development_suite,
    validate_capture_to_deformation_granular_development_suite,
)
from blueprint_pipeline.measurement_method_research_catalog import research_intake_catalog
from blueprint_pipeline.measurement_mujoco_granular_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
    run_mujoco_granular_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_deformation_granular_v1/corpus.json"
CORPUS_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_deformation_granular_development_corpus.v1.schema.json"
)
SUITE_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_deformation_granular_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:"
    + hashlib.sha256(b"capture-to-deformation-granular-development-no-controller").hexdigest()
)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _spec() -> dict:
    corpus_digest = _corpus_digest()
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-mujoco-spherical-granular-1",
        method_ids=["mujoco-3"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 4 / 6,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 120},
        minimum_repeated_trials=2,
        lane="granular",
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
        task_class="granular_manipulation",
        material_regime="granular_media",
        operating_point={**corpus["shared_operating_point"], **row},
    )


def _request(case_index: int = 0) -> dict:
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("mujoco-3"),
        _spec(),
        _case(case_index),
        execution_id=f"mujoco-spherical-granular-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="mujoco-cpu-rigid-sphere-contact",
        precision="float64",
        seed=31,
        solver_settings={
            "cone": "elliptic",
            "integrator": "Euler",
            "iterations": 50,
            "particle_model": "rigid_sphere_contact",
            "replay_count": 2,
            "solver": "Newton",
            "tolerance": 1e-8,
        },
        timeout_seconds=90,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_granular_corpus_schema_and_research_scope_are_explicit() -> None:
    corpus = _corpus()
    jsonschema.validate(
        corpus,
        json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    assert corpus["lane"] == "granular"
    assert corpus["shared_operating_point"]["particle_shape"] == "sphere"
    assert corpus["shared_operating_point"]["cohesion_model"] == "none"
    assert (
        corpus["shared_operating_point"]["material_characterization_scope"]
        == "synthetic_parameters_only"
    )
    assert corpus["physical_material_characterization_included"] is False
    mujoco_candidate = next(
        row for row in research_intake_catalog() if row["candidate_id"] == "mujoco-3"
    )
    assert "Q-GRAN" in mujoco_candidate["required_qualification_protocols"]
    assert (
        "rigid_sphere_contact_is_not_dem_or_characterized_granular_authority"
        in mujoco_candidate["known_limitations"]
    )


def test_real_mujoco_granular_corpus_executes_two_regimes() -> None:
    planned = run_capture_to_deformation_granular_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=False,
    )
    assert planned["status"] == "planned_not_executed"
    assert planned["solver_scope"] == "mujoco-rigid-monodisperse-sphere-contact"

    completed = run_capture_to_deformation_granular_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    metrics = completed["aggregate_metrics"]
    assert 1.1 < metrics["minimum_spread_ratio"] < metrics["maximum_spread_ratio"] < 3.0
    assert metrics["minimum_settled_fraction"] >= 0.95
    assert metrics["maximum_penetration_m"] < 0.003
    assert metrics["maximum_normal_contact_force_n"] < 10.0
    assert metrics["within_envelope_case_count"] == 2
    assert all(
        row["observed_metrics"]["topology_contact"] == "particle_ground_and_interparticle_contact"
        for row in completed["cases"]
    )
    assert all(row["warning_count"] == 0 for row in completed["cases"])
    for key in (
        "held_out",
        "physical_measurements_included",
        "physical_material_characterization_included",
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


def test_granular_worker_rejects_material_timestep_and_solver_scope_tampering() -> None:
    cohesive = copy.deepcopy(_request())
    cohesive["case_manifest"]["operating_point"]["cohesion_model"] = "constant_force"
    _rehash(cohesive["case_manifest"], "case_manifest_digest")
    cohesive.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="cohesion_model_invalid"):
        run_mujoco_granular_request(cohesive)

    characterized = copy.deepcopy(_request())
    characterized["case_manifest"]["operating_point"]["material_characterization_scope"] = (
        "physical_calibration"
    )
    _rehash(characterized["case_manifest"], "case_manifest_digest")
    characterized.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="material_characterization_scope_invalid",
    ):
        run_mujoco_granular_request(characterized)

    timestep = copy.deepcopy(_request())
    timestep["case_manifest"]["operating_point"]["duration_s"] = 1.000001
    _rehash(timestep["case_manifest"], "case_manifest_digest")
    timestep.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="timestep_duration_mismatch",
    ):
        run_mujoco_granular_request(timestep)

    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["particle_model"] = "dem"
    _rehash(solver["runtime_configuration"]["solver_settings"], "solver_settings_digest")
    solver["runtime_configuration"]["solver_settings_digest"] = solver["runtime_configuration"][
        "solver_settings"
    ].pop("solver_settings_digest")
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="particle_model_invalid"):
        run_mujoco_granular_request(solver)


def test_granular_suite_rejects_split_leakage_and_authority_tampering() -> None:
    with pytest.raises(GranularDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_deformation_granular_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
            execute=False,
        )
    planned = run_capture_to_deformation_granular_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=False,
    )
    planned["physical_material_characterization_included"] = True
    with pytest.raises(
        GranularDevelopmentSuiteError,
        match="physical_material_characterization_included",
    ):
        validate_capture_to_deformation_granular_development_suite(planned)


def test_granular_implementation_identity_is_distinct_from_other_mujoco_workers() -> None:
    for other_id in (
        "blueprint-mujoco-rigid-development-adapter",
        "blueprint-mujoco-flex-cloth-development-adapter",
    ):
        request = _request()
        request["implementation"]["implementation_id"] = other_id
        request.pop("execution_request_digest")
        result = run_mujoco_granular_request(request)
        assert result["status"] == "blocked"
        assert result["failure_codes"] == ["mujoco_granular_implementation_id_mismatch"]
