from __future__ import annotations

import copy
import hashlib
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
from blueprint_pipeline.measurement_adapter_runtime import build_measurement_adapter_descriptor
from blueprint_pipeline.measurement_isaac_physx_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    ISAAC_VERSION,
    PROTOCOL_ID,
    WORKER_SCRIPT,
    implementation_digest,
    run_isaac_physx_rigid_measurement_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = "sha256:" + hashlib.sha256(
    b"isaac-physx-rigid-development-no-controller"
).hexdigest()


def _isaac_launcher() -> Path:
    raw = os.environ.get("BLUEPRINT_ISAAC_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_ISAAC_PYTHON exact Isaac 6.0.1 runtime is not configured")
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
        benchmark_version="development-isaac-physx-tgs-rigid-drop-1",
        method_ids=["isaac-sim-6-physx"],
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
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 900},
        minimum_repeated_trials=2,
    )
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        spec,
        case_id=f"{case_id}--isaac-sim-6-physx",
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
        build_measurement_adapter_descriptor("isaac-sim-6-physx"),
        spec,
        case,
        execution_id=f"isaac-physx-rigid-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="isaac-physx-cpu-tgs-rigid",
        precision="float32",
        seed=47,
        solver_settings={
            "solver_type": "TGS",
            "broadphase_type": "SAP",
            "gpu_dynamics": False,
            "enhanced_determinism": True,
            "position_iterations": 8,
            "velocity_iterations": 2,
        },
        timeout_seconds=300,
    )


def test_isaac_descriptor_is_exact_and_plan_only_request_builds() -> None:
    descriptor = build_measurement_adapter_descriptor("isaac-sim-6-physx")
    assert descriptor["target_version"] == ISAAC_VERSION
    assert descriptor["production_route_eligible"] is False
    request = _request()
    assert request["runtime_configuration"]["backend_id"] == "isaac-physx-cpu-tgs-rigid"
    assert request["implementation"]["implementation_digest"] == implementation_digest()


def test_isaac_worker_rejects_solver_and_identity_tampering_before_import() -> None:
    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["solver_type"] = "PGS"
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_configuration_invalid"):
        run_isaac_physx_rigid_measurement_request(solver)

    identity = _request()
    identity["implementation"]["implementation_id"] = "unbound-isaac-worker"
    identity.pop("execution_request_digest")
    result = run_isaac_physx_rigid_measurement_request(identity)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["isaac_physx_rigid_implementation_id_mismatch"]


@pytest.mark.external_runtime
@pytest.mark.slow
def test_isaac_external_runtime_executes_shared_cases() -> None:
    launcher = _isaac_launcher()
    version = subprocess.run(
        [
            str(launcher),
            "-c",
            "import importlib.metadata;print(importlib.metadata.version('isaacsim'))",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip().splitlines()[-1] == ISAAC_VERSION
    bundles = [
        run_measurement_adapter_execution(
            _request(index),
            command_argv=[str(launcher), str(WORKER_SCRIPT)],
            execute=True,
        )
        for index in range(2)
    ]
    assert [row["receipt"]["status"] for row in bundles] == ["completed", "completed"]
    assert all(
        row["receipt"]["runtime_observations"]["engine_version"] == ISAAC_VERSION
        for row in bundles
    )
    assert all(
        row["receipt"]["runtime_observations"]["deterministic_replay_match"] is True
        for row in bundles
    )
    assert all(row["qualification_created"] is False for row in bundles)
    assert all(row["catalog_mutated"] is False for row in bundles)
