from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
    probe_measurement_adapter,
)
from blueprint_pipeline.measurement_drake_rigid_adapter import (
    DRAKE_VERSION,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    WORKER_SCRIPT,
    implementation_digest,
    run_drake_rigid_measurement_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"drake-rigid-development-no-controller").hexdigest()
)


def _drake_python() -> Path:
    raw = os.environ.get("BLUEPRINT_DRAKE_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_DRAKE_PYTHON exact external runtime is not configured")
    return Path(raw).absolute()


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _request(case_index: int = 0) -> dict:
    corpus = _corpus()
    corpus_digest = _corpus_digest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-drake-sap-rigid-drop-1",
        method_ids=["drake-1-55"],
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 120},
        minimum_repeated_trials=2,
    )
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        spec,
        case_id=f"{case_id}--drake-1-55",
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
        build_measurement_adapter_descriptor("drake-1-55"),
        spec,
        case,
        execution_id=f"drake-rigid-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="drake-multibody-cpu-sap-point",
        precision="float64",
        seed=47,
        solver_settings={
            "discrete_contact_approximation": "sap",
            "contact_model": "point",
            "penetration_allowance_m": 0.001,
            "stiction_tolerance_m_s": 0.001,
        },
        timeout_seconds=60,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_drake_descriptor_tracks_current_pip_surface_without_retired_visualizer() -> None:
    descriptor = build_measurement_adapter_descriptor("drake-1-55")
    assert descriptor["target_version"] == DRAKE_VERSION
    assert descriptor["probe_contract"]["python_distributions"] == ["drake"]
    assert descriptor["probe_contract"]["executables"] == []
    probe = probe_measurement_adapter(descriptor)
    try:
        installed = importlib.metadata.version("drake")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    assert probe["status"] == ("available" if installed == DRAKE_VERSION else "unavailable")


@pytest.mark.external_runtime
def test_drake_external_runtime_has_exact_version() -> None:
    python = _drake_python()
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata,pydrake.all;print(importlib.metadata.version('drake'))",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == DRAKE_VERSION


@pytest.mark.external_runtime
def test_drake_worker_executes_shared_cases_through_external_subprocess() -> None:
    python = _drake_python()
    bundles = [
        run_measurement_adapter_execution(
            _request(index),
            command_argv=[str(python), str(WORKER_SCRIPT)],
            execute=True,
        )
        for index in range(2)
    ]
    assert [row["receipt"]["status"] for row in bundles] == ["completed", "completed"]
    assert all(
        row["receipt"]["runtime_observations"]["engine_version"] == DRAKE_VERSION for row in bundles
    )
    assert all(
        row["receipt"]["runtime_observations"]["deterministic_replay_match"] is True
        for row in bundles
    )
    assert all(row["prediction"]["physical_success_established"] is False for row in bundles)
    assert all(row["qualification_created"] is False for row in bundles)
    assert all(row["catalog_mutated"] is False for row in bundles)


def test_drake_worker_rejects_solver_and_identity_tampering() -> None:
    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["contact_model"] = "hydroelastic"
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="contact_model_invalid"):
        run_drake_rigid_measurement_request(solver)

    identity = _request()
    identity["implementation"]["implementation_id"] = "unbound-drake-worker"
    identity.pop("execution_request_digest")
    result = run_drake_rigid_measurement_request(identity)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["drake_rigid_implementation_id_mismatch"]
