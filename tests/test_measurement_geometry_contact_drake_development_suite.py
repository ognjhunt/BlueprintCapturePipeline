from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_geometry_contact_drake_development_suite import (
    GeometryContactDrakeDevelopmentSuiteError,
    run_capture_to_geometry_contact_drake_development_suite,
    validate_capture_to_geometry_contact_drake_development_suite,
)


pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
SCHEMA_PATH = (
    ROOT / "docs/schemas/capture_to_geometry_contact_drake_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"drake-rigid-development-no-controller").hexdigest()
)


def _drake_python() -> Path:
    raw = os.environ.get("BLUEPRINT_DRAKE_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_DRAKE_PYTHON exact external runtime is not configured")
    return Path(raw).absolute()


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def test_drake_suite_plans_and_executes_method_neutral_corpus() -> None:
    python = _drake_python()
    planned = run_capture_to_geometry_contact_drake_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        worker_python=python,
    )
    assert planned["status"] == "planned_not_executed"
    completed = run_capture_to_geometry_contact_drake_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        worker_python=python,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["case_count"] == 2
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert completed["scene_graph_renderer_used"] is False
    assert completed["drake_visualizer_used"] is False
    assert completed["aggregate_metrics"]["ground_contact_case_count"] == 2
    assert 0 <= completed["aggregate_metrics"]["maximum_penetration_m"] < 0.003
    assert all(row["failure_codes"] == [] for row in completed["cases"])
    jsonschema.validate(completed, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_drake_suite_preserves_virtual_environment_python_symlink() -> None:
    python = _drake_python()
    if not python.is_symlink():
        pytest.skip("configured Drake interpreter is not a virtual-environment symlink")
    completed = run_capture_to_geometry_contact_drake_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        worker_python=python,
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert {row["engine_version"] for row in completed["cases"]} == {"1.55.0"}


def test_drake_suite_rejects_split_leakage() -> None:
    with pytest.raises(GeometryContactDrakeDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_geometry_contact_drake_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )


def test_drake_suite_rejects_authority_and_digest_tampering() -> None:
    planned = run_capture_to_geometry_contact_drake_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    tampered = copy.deepcopy(planned)
    tampered["r7_admission"] = True
    tampered["scene_graph_renderer_used"] = True
    with pytest.raises(
        GeometryContactDrakeDevelopmentSuiteError,
        match="r7_admission|scene_graph_renderer_used|digest_mismatch",
    ):
        validate_capture_to_geometry_contact_drake_development_suite(tampered)
