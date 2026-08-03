from __future__ import annotations

import copy
import hashlib
import json
import subprocess
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
            "display_mode": "pyglet_headless_egl",
            "import_diagnostic": "audit_exception_first_case_only",
            "native_diagnostic": "gdb_first_case_only",
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
    phases: list[str] = []
    result = run_dlo_lab_cable_request(_request(), phase_writer=phases.append)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["dlo_lab_adapter_runtime_unavailable"]
    assert result["physical_success_established"] is False
    assert result["qualification_labels_accessed"] is False
    assert phases[0] == "torch_import_started"
    assert "genesis_import_completed" not in phases


def test_dlo_lab_supervisor_contains_native_abort_and_classifies_stderr(
    monkeypatch,
) -> None:
    outcomes = iter(
        [
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(
                returncode=-6,
                stdout=b"",
                stderr=b"OMP: Error libgomp terminate called\n",
            ),
        ]
    )
    monkeypatch.setattr(
        dlo_adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: next(outcomes),
    )
    result = dlo_adapter._run_supervised_worker(_request())
    assert result["status"] == "blocked"
    assert result["failure_codes"] == [
        "dlo_lab_adapter_native_import_probe_failed:torch_then_quadrants",
        "dlo_lab_adapter_native_termination",
        "dlo_lab_adapter_openmp_runtime_failure",
        "dlo_lab_adapter_supervised_worker_exit_nonzero:-6",
        "dlo_lab_adapter_supervised_worker_signal:6",
    ]
    observations = result["runtime_observations"]
    assert observations["supervised_worker_phase"] == ("native_import_probe:torch_then_quadrants")
    assert observations["supervised_worker_native_signal"] == 6
    assert observations["supervised_worker_timed_out"] is False
    assert observations["supervised_worker_import_order"] == ["torch", "genesis"]
    assert observations["python_version"]
    assert observations["quadrants_distribution_version"]
    assert observations["torch_distribution_version"]
    assert observations["supervised_worker_stderr_bytes"] > 0
    assert observations["supervised_worker_stderr_content_persisted"] is False


def test_dlo_lab_supervisor_does_not_report_timeout_as_native_signal(
    monkeypatch,
) -> None:
    def timed_out(*_args, **_kwargs):
        raise dlo_adapter.subprocess.TimeoutExpired(
            cmd=["dlo-worker"],
            timeout=5,
            stderr=b"partial diagnostic",
        )

    monkeypatch.setattr(dlo_adapter.subprocess, "run", timed_out)
    result = dlo_adapter._run_supervised_worker(_request())
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["dlo_lab_adapter_native_import_probe_timed_out"]
    observations = result["runtime_observations"]
    assert observations["supervised_worker_exit_code"] is None
    assert observations["supervised_worker_native_signal"] is None
    assert observations["supervised_worker_timed_out"] is True


def test_native_debugger_observation_sanitizes_paths_and_bounds_output(
    monkeypatch,
) -> None:
    raw = (
        b"Catchpoint 1 (exception thrown), 0x00007f in __cxa_throw () "
        b"from /usr/lib/x86_64-linux-gnu/libstdc++.so.6\n"
        b"#0  0x00007f in __cxa_throw () from /usr/lib/x86_64-linux-gnu/libstdc++.so.6\n"
        b"#1  0x000042 in quadrants::ThreadPool::ThreadPool(int) "
        b"at /build/private/threading.cpp:40\n"
    )
    monkeypatch.setenv("BLUEPRINT_MEASUREMENT_DLO_INPUT_GET_URL", "https://signed/input")
    monkeypatch.setenv("BLUEPRINT_MEASUREMENT_DLO_OUTPUT_GET_URL", "https://signed/get")
    monkeypatch.setenv("BLUEPRINT_MEASUREMENT_DLO_OUTPUT_PUT_URL", "https://signed/put")
    call: dict = {}

    def completed(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

    monkeypatch.setattr(dlo_adapter.shutil, "which", lambda _name: "/usr/bin/gdb")
    monkeypatch.setattr(
        dlo_adapter.subprocess,
        "run",
        completed,
    )
    observation = dlo_adapter._native_debugger_observation(
        import_statements="import quadrants",
        timeout_seconds=90,
    )
    assert observation["status"] == "captured"
    assert observation["timeout_seconds"] == 45
    assert observation["raw_output_content_persisted"] is False
    assert observation["raw_output_bytes"] == len(raw)
    assert observation["frames"] == [
        {"index": 0, "symbol": "__cxa_throw", "module": "libstdc++.so.6"},
        {
            "index": 1,
            "symbol": "quadrants::ThreadPool::ThreadPool",
            "module": None,
        },
    ]
    assert "/build/private" not in json.dumps(observation)
    debugger_command = call["args"][0]
    assert "bt 32" in debugger_command
    assert "thread apply all" not in " ".join(debugger_command)
    diagnostic_env = call["kwargs"]["env"]
    assert "BLUEPRINT_MEASUREMENT_DLO_INPUT_GET_URL" not in diagnostic_env
    assert "BLUEPRINT_MEASUREMENT_DLO_OUTPUT_GET_URL" not in diagnostic_env
    assert "BLUEPRINT_MEASUREMENT_DLO_OUTPUT_PUT_URL" not in diagnostic_env


def test_dlo_lab_supervisor_runs_native_diagnostic_once_for_first_case(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dlo_adapter,
        "_native_debugger_observation",
        lambda **_kwargs: {
            "status": "captured",
            "tool": "gdb",
            "raw_output_content_persisted": False,
            "frames": [{"index": 0, "symbol": "__cxa_throw", "module": None}],
        },
    )
    monkeypatch.setattr(
        dlo_adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=-6,
            stdout=b"",
            stderr=b"terminate called after throwing std::system_error\n",
        ),
    )
    request = _request()
    assert request["execution_id"].endswith("001")
    result = dlo_adapter._run_supervised_worker(request)
    diagnostic = result["runtime_observations"]["supervised_worker_native_diagnostic"]
    assert diagnostic["status"] == "captured"
    assert diagnostic["raw_output_content_persisted"] is False


def test_import_audit_observation_retains_only_sanitized_module_names(
    monkeypatch,
) -> None:
    stdout = (
        b"BLUEPRINT_IMPORT:torch\n"
        b"BLUEPRINT_IMPORT:genesis.engine.solvers\n"
        b"BLUEPRINT_IMPORT:../../private/path\n"
        b"BLUEPRINT_EXCEPTION:builtins.RuntimeError\n"
        b"BLUEPRINT_SYSTEM_EXIT:9999\n"
    )
    stderr = b"Traceback from /tmp/private/source.py\n"
    call: dict = {}

    def completed(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return SimpleNamespace(returncode=87, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(dlo_adapter.subprocess, "run", completed)
    observation = dlo_adapter._import_audit_observation(
        import_statements="import torch\nimport genesis",
        timeout_seconds=90,
    )
    assert observation["status"] == "captured"
    assert observation["tool"] == "python_audit_hook"
    assert observation["return_code"] == 87
    assert observation["timeout_seconds"] == 45
    assert observation["last_modules"] == ["torch", "genesis.engine.solvers"]
    assert observation["exception_type"] == "builtins.RuntimeError"
    assert observation["system_exit_code"] is None
    assert observation["raw_output_content_persisted"] is False
    assert "/tmp/private" not in json.dumps(observation)
    assert "../../private" not in json.dumps(observation)
    assert call["kwargs"]["stdout"] is dlo_adapter.subprocess.PIPE
    assert call["kwargs"]["stderr"] is dlo_adapter.subprocess.PIPE
    assert call["kwargs"]["env"]["PYGLET_HEADLESS"] == "1"
    assert call["kwargs"]["env"]["PYOPENGL_PLATFORM"] == "egl"


def test_import_audit_script_uses_stdout_and_sanitizes_exception_metadata() -> None:
    source = dlo_adapter._import_audit_script("raise RuntimeError('/private/path')")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 87
    assert b"BLUEPRINT_EXCEPTION:builtins.RuntimeError" in completed.stdout
    assert b"/private/path" not in completed.stdout
    assert completed.stderr == b""


def test_dlo_headless_environment_overrides_conflicting_host_values(monkeypatch) -> None:
    monkeypatch.setenv("PYGLET_HEADLESS", "0")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "glx")
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XAUTHORITY", "/tmp/host-xauthority")
    environment = dlo_adapter._headless_subprocess_environment()
    assert environment["PYGLET_HEADLESS"] == "1"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert "DISPLAY" not in environment
    assert "WAYLAND_DISPLAY" not in environment
    assert "XAUTHORITY" not in environment


def test_dlo_lab_supervisor_runs_import_audit_once_for_first_case(
    monkeypatch,
) -> None:
    outcomes = iter(
        [
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
            SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
        ]
    )
    monkeypatch.setattr(
        dlo_adapter,
        "_import_audit_observation",
        lambda **_kwargs: {
            "status": "captured",
            "tool": "python_audit_hook",
            "raw_output_content_persisted": False,
            "last_modules": ["torch", "genesis"],
        },
    )
    monkeypatch.setattr(
        dlo_adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: next(outcomes),
    )
    result = dlo_adapter._run_supervised_worker(_request())
    diagnostic = result["runtime_observations"]["supervised_worker_import_diagnostic"]
    assert diagnostic["status"] == "captured"
    assert diagnostic["last_modules"] == ["torch", "genesis"]


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
