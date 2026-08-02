from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_geometry_kinematic_development_suite import (
    GeometryKinematicDevelopmentSuiteError,
    run_capture_to_geometry_kinematic_development_suite,
    validate_capture_to_geometry_kinematic_development_suite,
)


pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_kinematic_v1/corpus.json"
CORPUS_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_kinematic_development_corpus.v1.schema.json"
)
SUITE_SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_kinematic_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"exact-geometry-two-link-controller").hexdigest()
)


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def test_kinematic_corpus_schema_forces_synthetic_discrete_scope() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(
        corpus,
        json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    assert corpus["lane"] == "planar_reach_discrete_collision"
    assert len(corpus["cases"]) == 3
    for key in (
        "held_out",
        "captured_mesh_included",
        "captured_registration_included",
        "physical_measurements_included",
        "qualification_labels_included",
        "continuous_collision_evaluated",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
    ):
        assert corpus[key] is False


def test_kinematic_suite_plans_and_executes_three_boundary_cases() -> None:
    planned = run_capture_to_geometry_kinematic_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_capture_to_geometry_kinematic_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 3
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"] == {
        "reachable_case_count": 2,
        "unreachable_case_count": 1,
        "discrete_collision_case_count": 1,
        "unsafe_case_count": 2,
        "maximum_target_position_error_m": 0.0,
    }
    assert completed["continuous_collision_evaluated"] is False
    jsonschema.validate(
        completed,
        json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_kinematic_suite_rejects_split_leakage_and_authority_tampering() -> None:
    with pytest.raises(GeometryKinematicDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_geometry_kinematic_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )
    planned = run_capture_to_geometry_kinematic_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["continuous_collision_evaluated"] = True
    planned["r7_admission"] = True
    with pytest.raises(
        GeometryKinematicDevelopmentSuiteError,
        match="continuous_collision_evaluated|r7_admission|digest_mismatch",
    ):
        validate_capture_to_geometry_kinematic_development_suite(planned)


def test_kinematic_suite_digest_tampering_fails_closed() -> None:
    planned = run_capture_to_geometry_kinematic_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["case_count"] = 4
    with pytest.raises(
        GeometryKinematicDevelopmentSuiteError,
        match="case_count_mismatch|digest_mismatch",
    ):
        validate_capture_to_geometry_kinematic_development_suite(planned)
