from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from blueprint_pipeline.measurement_adapter_runtime import build_measurement_adapter_descriptor
from blueprint_pipeline.measurement_chrono_granular_adapter import (
    EXPECTED_ENGINE_VERSION,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    WORKER_SCRIPT,
    implementation_digest,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = (
    ROOT / "tests/fixtures/measurement_capture_to_deformation_granular_chrono_v1/corpus.json"
)
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"chrono-granular-development-no-controller").hexdigest()
)


def _chrono_python() -> Path:
    raw = os.environ.get("BLUEPRINT_CHRONO_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_CHRONO_PYTHON exact external runtime is not configured")
    return Path(raw).absolute()


def _request(case_index: int = 0) -> dict:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_digest = "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-chrono-nsc-spherical-granular-1",
        method_ids=["project-chrono-10"],
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 180},
        minimum_repeated_trials=2,
        lane="granular",
    )
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        spec,
        case_id=case_id,
        split="development",
        input_artifact_digests=[corpus_digest],
        task_class="granular_manipulation",
        material_regime="synthetic_chrono_nsc_granular_media",
        operating_point={**corpus["shared_operating_point"], **row},
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("project-chrono-10"),
        spec,
        case,
        execution_id=f"chrono-granular-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="chrono-nsc-cpu-bullet-psor",
        precision="float64",
        seed=37,
        solver_settings={
            "collision_system": "bullet",
            "contact_method": "nsc",
            "replay_count": 2,
            "solver": "psor",
            "timestepper": "euler_implicit_linearized",
        },
        timeout_seconds=120,
    )


def test_chrono_descriptor_is_exact_external_conda_without_false_pypi_probe() -> None:
    descriptor = build_measurement_adapter_descriptor("project-chrono-10")
    assert descriptor["target_version"] == EXPECTED_ENGINE_VERSION
    assert descriptor["execution_mode"] == "isolated_external_conda_or_exact_source_build"
    assert descriptor["probe_contract"]["python_distributions"] == []
    assert descriptor["probe_contract"]["executables"] == []


@pytest.mark.slow
def test_chrono_worker_executes_both_nsc_cases_with_exact_replay() -> None:
    python = _chrono_python()
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
        row["receipt"]["runtime_observations"]["engine_version"] == EXPECTED_ENGINE_VERSION
        for row in bundles
    )
    assert all(
        row["receipt"]["runtime_observations"]["deterministic_replay_match"] is True
        for row in bundles
    )
    assert all(
        row["receipt"]["runtime_observations"]["chrono_granular_gpu_module_used"] is False
        for row in bundles
    )
    assert all(row["prediction"]["unsafe_condition_predicted"] is False for row in bundles)
    assert all(row["qualification_created"] is False for row in bundles)
    assert all(row["catalog_mutated"] is False for row in bundles)


def test_chrono_worker_request_rejects_solver_before_runtime_import() -> None:
    from blueprint_pipeline.measurement_chrono_granular_adapter import (
        run_chrono_granular_request,
    )

    request = copy.deepcopy(_request())
    request["runtime_configuration"]["solver_settings"]["contact_method"] = "smc"
    encoded = json.dumps(
        request["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    request["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    request.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_settings_invalid"):
        run_chrono_granular_request(request)
