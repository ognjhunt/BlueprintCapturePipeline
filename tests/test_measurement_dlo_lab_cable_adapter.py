from __future__ import annotations

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
)
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
    probe_measurement_adapter,
)
from blueprint_pipeline.measurement_dlo_lab_cable_adapter import (
    EXPECTED_SOURCE_COMMIT,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    PROTOCOL_ID,
    implementation_digest,
    run_dlo_lab_cable_request,
)
from blueprint_pipeline import measurement_dlo_lab_cable_adapter as dlo_adapter
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _request() -> dict:
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-dlo-lab-cable-1",
        method_ids=["dlo-lab"],
        development_split_digest=_digest("dlo-development-split"),
        qualification_split_digest=_digest("dlo-qualification-split"),
        capture_bundle_digests=[_digest("dlo-synthetic-corpus")],
        robot_controller_digests=[_digest("no-controller")],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 0.5,
        },
        compute_budget={"usd": 1.0, "maximum_duration_seconds": 900},
        lane="cable",
    )
    case = build_benchmark_case_manifest(
        spec,
        case_id="dlo-lab-cantilever-development-001",
        split="development",
        input_artifact_digests=[_digest("dlo-synthetic-corpus")],
        task_class="cable_hose_routing",
        material_regime="synthetic_parameterized_rod",
        operating_point={
            "adapter_protocol": PROTOCOL_ID,
            "length_unit": "meters",
            "mass_unit": "kilograms",
            "time_unit": "seconds",
            "n_vertices": 16,
            "start_m": [0.0, 0.0, 0.5],
            "direction": [1.0, 0.0, 0.0],
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "base_length_m": 0.3,
            "base_radius_m": 0.005,
            "segment_mass_kg": 0.002,
            "youngs_modulus_pa": 100000.0,
            "shear_modulus_pa": 10000.0,
            "timestep_s": 0.001,
            "step_count": 100,
            "substeps": 5,
            "damping": 10.0,
            "angular_damping": 5.0,
            "maximum_tip_displacement_m": 0.3,
            "maximum_segment_strain": 0.2,
        },
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("dlo-lab"),
        spec,
        case,
        execution_id="dlo-lab-cable-development-001",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="dlo-lab-genesis-cuda",
        precision="float64",
        seed=31,
        solver_settings={
            "backend": "cuda",
            "replay_count": 2,
            "source_commit": EXPECTED_SOURCE_COMMIT,
        },
        timeout_seconds=900,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_dlo_lab_descriptor_requires_exact_source_checkout_not_package_guess() -> None:
    descriptor = build_measurement_adapter_descriptor("dlo-lab")
    assert descriptor["target_version"] == EXPECTED_SOURCE_COMMIT
    assert descriptor["execution_mode"] == "isolated_source_checkout"
    assert descriptor["probe_contract"]["python_distributions"] == []
    probe = probe_measurement_adapter(descriptor)
    assert probe["status"] == "manual_review"
    assert probe["qualification_established"] is False


def test_dlo_lab_worker_blocks_cleanly_without_exact_gpu_runtime() -> None:
    result = run_dlo_lab_cable_request(_request())
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["dlo_lab_adapter_runtime_unavailable"]
    assert result["physical_success_established"] is False
    assert result["qualification_labels_accessed"] is False


def test_dlo_lab_worker_rejects_protocol_solver_and_implementation_tampering() -> None:
    protocol = copy.deepcopy(_request())
    protocol["case_manifest"]["operating_point"]["adapter_protocol"] = "generic-genesis"
    _rehash(protocol["case_manifest"], "case_manifest_digest")
    protocol.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="protocol_invalid"):
        run_dlo_lab_cable_request(protocol)

    settings = copy.deepcopy(_request())
    settings["runtime_configuration"]["solver_settings"]["backend"] = "cpu"
    _rehash(settings["runtime_configuration"]["solver_settings"], "unused")
    settings["runtime_configuration"]["solver_settings"].pop("unused")
    encoded = json.dumps(
        settings["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    settings["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    settings.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_settings_invalid"):
        run_dlo_lab_cable_request(settings)

    identity = copy.deepcopy(_request())
    identity["implementation"]["implementation_digest"] = _digest("wrong-implementation")
    identity.pop("execution_request_digest")
    result = run_dlo_lab_cable_request(identity)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["dlo_lab_adapter_implementation_identity_mismatch"]


def test_dlo_lab_operating_point_rejects_cpu_fallback_and_unbounded_cases() -> None:
    request = copy.deepcopy(_request())
    request["runtime_configuration"]["backend_id"] = "genesis-auto"
    request.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError,
        match="runtime_configuration_invalid",
    ):
        run_dlo_lab_cable_request(request)

    excessive = copy.deepcopy(_request())
    excessive["case_manifest"]["operating_point"]["step_count"] = 5001
    _rehash(excessive["case_manifest"], "case_manifest_digest")
    excessive.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="step_count_invalid"):
        run_dlo_lab_cable_request(excessive)


def test_exact_cuda_source_double_replay_is_required_for_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_genesis = SimpleNamespace(__file__="/tmp/dlo-lab/genesis/__init__.py")
    fake_torch = SimpleNamespace(
        __version__="2.9.1+cu130",
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "genesis", fake_genesis)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        dlo_adapter.importlib.metadata,
        "version",
        lambda name: "1.0.0" if name == "genesis-world" else "unexpected",
    )
    monkeypatch.setattr(dlo_adapter, "_source_commit", lambda _gs: EXPECTED_SOURCE_COMMIT)
    simulation = {
        "trace_digest": _digest("dlo-trace"),
        "final_tip_position_m": [0.3, 0.0, 0.45],
        "tip_displacement_m": 0.05,
        "maximum_segment_strain": 0.01,
        "applied_gravity_force_n": 0.2943,
    }
    monkeypatch.setattr(dlo_adapter, "_simulate", lambda *_args, **_kwargs: dict(simulation))
    completed = run_dlo_lab_cable_request(_request())
    assert completed["status"] == "completed"
    assert completed["runtime_observations"]["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert completed["runtime_observations"]["deterministic_replay_match"] is True
    assert completed["runtime_observations"]["cpu_fallback_used"] is False
    assert completed["unsafe_condition_predicted"] is False
    assert completed["physical_success_established"] is False

    call_count = 0

    def drifting_simulation(*_args: object, **_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        value = dict(simulation)
        value["trace_digest"] = _digest(f"dlo-trace-{call_count}")
        return value

    monkeypatch.setattr(dlo_adapter, "_simulate", drifting_simulation)
    failed = run_dlo_lab_cable_request(_request())
    assert failed["status"] == "failed"
    assert failed["failure_codes"] == ["dlo_lab_adapter_replay_mismatch"]
