from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_geometry_contact_sapien_development_suite import (
    GeometryContactSapienDevelopmentSuiteError,
    run_capture_to_geometry_contact_sapien_development_suite,
    validate_capture_to_geometry_contact_sapien_development_suite,
)


pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_contact_sapien_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"sapien-rigid-development-no-controller").hexdigest()
)


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def test_sapien_suite_plans_and_executes_method_neutral_corpus() -> None:
    planned = run_capture_to_geometry_contact_sapien_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_capture_to_geometry_contact_sapien_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert completed["renderer_used"] is False
    assert completed["maniskill_runtime_used"] is False
    assert {row["body_shape"] for row in completed["cases"]} == {"sphere", "box"}
    assert completed["aggregate_metrics"]["ground_contact_case_count"] == 2
    assert 0 <= completed["aggregate_metrics"]["maximum_penetration_m"] < 0.001
    assert 0 < completed["aggregate_metrics"]["mean_first_contact_time_s"] < 1
    assert all(row["renderer_created"] is False for row in completed["cases"])
    assert all(row["maniskill_runtime_used"] is False for row in completed["cases"])
    jsonschema.validate(
        completed,
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_sapien_suite_rejects_split_leakage() -> None:
    with pytest.raises(GeometryContactSapienDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_geometry_contact_sapien_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )


def test_sapien_suite_rejects_authority_and_runtime_scope_tampering() -> None:
    planned = run_capture_to_geometry_contact_sapien_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    tampered = copy.deepcopy(planned)
    tampered["r7_admission"] = True
    tampered["renderer_used"] = True
    with pytest.raises(
        GeometryContactSapienDevelopmentSuiteError,
        match="r7_admission|renderer_used|digest_mismatch",
    ):
        validate_capture_to_geometry_contact_sapien_development_suite(tampered)


def test_sapien_suite_digest_tampering_fails_closed() -> None:
    planned = run_capture_to_geometry_contact_sapien_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    planned["case_count"] = 3
    with pytest.raises(
        GeometryContactSapienDevelopmentSuiteError,
        match="case_count_mismatch|digest_mismatch",
    ):
        validate_capture_to_geometry_contact_sapien_development_suite(planned)
