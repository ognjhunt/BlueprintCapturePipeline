from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_deformation_granular_chrono_development_suite import (
    ChronoGranularDevelopmentSuiteError,
    run_capture_to_deformation_granular_chrono_development_suite,
    validate_capture_to_deformation_granular_chrono_development_suite,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = (
    ROOT / "tests/fixtures/measurement_capture_to_deformation_granular_chrono_v1/corpus.json"
)
CORPUS_SCHEMA_PATH = (
    ROOT
    / "docs/schemas/capture_to_deformation_granular_chrono_development_corpus.v1.schema.json"
)
SUITE_SCHEMA_PATH = (
    ROOT
    / "docs/schemas/capture_to_deformation_granular_chrono_development_suite.v1.schema.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = "sha256:" + hashlib.sha256(
    b"chrono-granular-development-no-controller"
).hexdigest()


def _chrono_python() -> Path:
    raw = os.environ.get("BLUEPRINT_CHRONO_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_CHRONO_PYTHON exact external runtime is not configured")
    return Path(raw).absolute()


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def test_chrono_granular_corpus_and_plan_are_schema_checked() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(corpus, json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8")))
    planned = run_capture_to_deformation_granular_chrono_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    assert planned["status"] == "planned_not_executed"
    assert planned["chrono_granular_gpu_module_used"] is False
    assert planned["r5_evidence"] is False
    assert planned["r6_decision"] is False
    assert planned["r7_admission"] is False
    jsonschema.validate(planned, json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")))


@pytest.mark.slow
def test_chrono_granular_suite_executes_bounded_deterministic_corpus() -> None:
    completed = run_capture_to_deformation_granular_chrono_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        worker_python=_chrono_python(),
        execute=True,
    )
    assert completed["status"] == "completed_development_only"
    assert completed["all_cases_completed"] is True
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"]["within_envelope_case_count"] == 2
    assert completed["aggregate_metrics"]["minimum_settled_fraction"] >= 0.95
    assert completed["aggregate_metrics"]["maximum_spread_ratio"] <= 3.0
    assert completed["aggregate_metrics"]["maximum_penetration_m"] <= 0.003
    assert completed["aggregate_metrics"]["maximum_normal_contact_force_n"] <= 35.0
    assert all(row["failure_codes"] == [] for row in completed["cases"])
    assert all(row["chrono_granular_gpu_module_used"] is False for row in completed["cases"])
    jsonschema.validate(completed, json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")))


def test_chrono_granular_suite_rejects_split_and_authority_tampering() -> None:
    with pytest.raises(ChronoGranularDevelopmentSuiteError, match="split_leakage"):
        run_capture_to_deformation_granular_chrono_development_suite(
            CORPUS_PATH,
            qualification_split_digest=_corpus_digest(),
            controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
        )
    planned = run_capture_to_deformation_granular_chrono_development_suite(
        CORPUS_PATH,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        controller_scope_digest=CONTROLLER_SCOPE_DIGEST,
    )
    tampered = copy.deepcopy(planned)
    tampered["r7_admission"] = True
    with pytest.raises(
        ChronoGranularDevelopmentSuiteError,
        match="r7_admission|digest_mismatch",
    ):
        validate_capture_to_deformation_granular_chrono_development_suite(tampered)
