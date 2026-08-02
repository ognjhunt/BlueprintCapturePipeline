from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
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
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)
from blueprint_pipeline.measurement_sapien_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    SAPIEN_VERSION,
    implementation_digest,
    run_sapien_rigid_measurement_request,
)


pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"sapien-rigid-development-no-controller").hexdigest()
)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _request(case_index: int = 0) -> dict:
    corpus = _corpus()
    corpus_digest = _corpus_digest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-sapien-physx-rigid-drop-1",
        method_ids=["sapien-maniskill-3"],
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
        case_id=f"{case_id}--sapien-maniskill-3",
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
        build_measurement_adapter_descriptor("sapien-maniskill-3"),
        spec,
        case,
        execution_id=f"sapien-rigid-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="sapien-physx-cpu",
        precision="float32",
        seed=43,
        solver_settings={
            "enhanced_determinism": True,
            "enable_tgs": True,
            "cpu_workers": 0,
            "position_iterations": 8,
            "velocity_iterations": 2,
        },
        timeout_seconds=60,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_sapien_extra_preserves_headless_opencv_and_exact_probe() -> None:
    assert importlib.metadata.version("sapien") == SAPIEN_VERSION
    assert importlib.metadata.version("opencv-python-headless") == "4.11.0.86"
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("opencv-python")
    descriptor = build_measurement_adapter_descriptor("sapien-maniskill-3")
    probe = probe_measurement_adapter(descriptor)
    assert probe["status"] == "available"
    assert [(row["name"], row["observed_version"]) for row in probe["probes"]] == [
        ("sapien", SAPIEN_VERSION)
    ]


def test_sapien_worker_executes_both_shared_rigid_cases_with_exact_replay() -> None:
    outputs = [run_sapien_rigid_measurement_request(_request(index)) for index in range(2)]
    assert [row["status"] for row in outputs] == ["completed", "completed"]
    assert [row["observed_metrics"]["contact_sequence"] for row in outputs] == [
        "ground_contact",
        "ground_contact",
    ]
    assert all(row["unsafe_condition_predicted"] is False for row in outputs)
    assert all(row["runtime_observations"]["deterministic_replay_match"] is True for row in outputs)
    assert all(row["runtime_observations"]["renderer_created"] is False for row in outputs)
    assert all(row["runtime_observations"]["maniskill_runtime_used"] is False for row in outputs)


def test_sapien_worker_runs_through_uniform_subprocess_boundary() -> None:
    bundle = run_measurement_adapter_execution(
        _request(),
        command_argv=[sys.executable, "-m", "blueprint_pipeline.measurement_sapien_rigid_adapter"],
        execute=True,
    )
    assert bundle["receipt"]["status"] == "completed"
    assert bundle["receipt"]["evidence_class"] == "development_execution"
    assert bundle["receipt"]["runtime_observations"]["physx_version"].startswith("105.1-")
    assert bundle["prediction"]["physical_success_established"] is False
    assert bundle["qualification_created"] is False
    assert bundle["catalog_mutated"] is False


def test_sapien_worker_rejects_protocol_and_determinism_tampering() -> None:
    protocol = copy.deepcopy(_request())
    protocol["case_manifest"]["operating_point"]["adapter_protocol"] = "newton_xpbd_rigid_drop.v1"
    _rehash(protocol["case_manifest"], "case_manifest_digest")
    protocol.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="protocol_invalid"):
        run_sapien_rigid_measurement_request(protocol)

    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["enhanced_determinism"] = False
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="determinism_invalid"):
        run_sapien_rigid_measurement_request(solver)


def test_sapien_implementation_identity_cannot_inherit_newton_or_mujoco() -> None:
    request = _request()
    request["implementation"]["implementation_id"] = (
        "blueprint-newton-xpbd-rigid-development-adapter"
    )
    request.pop("execution_request_digest")
    result = run_sapien_rigid_measurement_request(request)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["sapien_rigid_implementation_id_mismatch"]
