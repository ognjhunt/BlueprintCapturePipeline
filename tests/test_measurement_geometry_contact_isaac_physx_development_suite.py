from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_geometry_contact_isaac_physx_development_suite import (
    GeometryContactIsaacPhysxDevelopmentSuiteError,
    run_capture_to_geometry_contact_isaac_physx_development_suite,
    validate_capture_to_geometry_contact_isaac_physx_development_suite,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
SCHEMA_PATH = (
    ROOT
    / "docs/schemas/capture_to_geometry_contact_isaac_physx_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = "sha256:" + hashlib.sha256(
    b"isaac-physx-rigid-development-no-controller"
).hexdigest()


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _isaac_launcher() -> Path:
    raw = os.environ.get("BLUEPRINT_ISAAC_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_ISAAC_PYTHON exact Isaac 6.0.1 runtime is not configured")
    return Path(raw).absolute()


def test_isaac_suite_plans_without_claiming_external_execution() -> None:
    planned = run_capture_to_geometry_contact_isaac_physx_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    assert planned["external_isaac_runtime_required"] is True
    assert planned["external_isaac_runtime_configured"] is False
    assert planned["actual_isaac_execution_verified"] is False
    assert planned["all_cases_completed"] is False
    assert planned["r7_admission"] is False
    jsonschema.validate(planned, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_isaac_suite_requires_explicit_runtime_for_execution() -> None:
    with pytest.raises(
        GeometryContactIsaacPhysxDevelopmentSuiteError,
        match="exact_runtime_not_configured",
    ):
        run_capture_to_geometry_contact_isaac_physx_development_suite(
            CORPUS_PATH,
            qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
            execute=True,
        )


def test_isaac_suite_rejects_split_authority_and_digest_tampering() -> None:
    with pytest.raises(GeometryContactIsaacPhysxDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_geometry_contact_isaac_physx_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )
    planned = run_capture_to_geometry_contact_isaac_physx_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    tampered = copy.deepcopy(planned)
    tampered["r7_admission"] = True
    tampered["actual_isaac_execution_verified"] = True
    with pytest.raises(
        GeometryContactIsaacPhysxDevelopmentSuiteError,
        match="r7_admission|unverified_execution|digest_mismatch",
    ):
        validate_capture_to_geometry_contact_isaac_physx_development_suite(tampered)


@pytest.mark.slow
def test_isaac_suite_executes_exact_external_runtime() -> None:
    completed = run_capture_to_geometry_contact_isaac_physx_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        worker_launcher=_isaac_launcher(),
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["actual_isaac_execution_verified"] is True
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"]["ground_contact_case_count"] == 2
    assert completed["aggregate_metrics"]["contact_report_event_count"] > 0
    assert all(row["failure_codes"] == [] for row in completed["cases"])
    jsonschema.validate(completed, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
