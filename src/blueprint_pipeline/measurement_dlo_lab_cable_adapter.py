"""Exact-source DLO-Lab CUDA cable development worker.

The worker runs a bounded fixed-free parameterized rod twice with the DLO-Lab
fork's GPU ROD solver.  It requires the exact upstream source commit and CUDA;
CPU fallback, a generic Genesis installation, or replay drift blocks the result.
The synthetic case is development evidence only and cannot characterize a real
cable or create Q-DLO/R5/R6/R7 authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


IMPLEMENTATION_ID = "blueprint-dlo-lab-cable-development-adapter"
IMPLEMENTATION_VERSION = "1"
PROTOCOL_ID = "dlo_lab_parameterized_rod_cantilever.v1"
EXPECTED_DISTRIBUTION_VERSION = "1.0.0"
EXPECTED_SOURCE_COMMIT = "c5026a9416b03c6bc5186eba13cd4ffd4c0e7796"
_IMPORT_PROBES = (
    ("quadrants", "import quadrants"),
    ("torch", "import torch"),
    ("quadrants_then_torch", "import quadrants\nimport torch"),
    ("torch_then_quadrants", "import torch\nimport quadrants"),
    ("torch_then_genesis", "import torch\nimport genesis"),
)
_IMPORT_UNAVAILABLE_EXIT_CODE = 86
_NATIVE_DIAGNOSTIC_MODE = "gdb_first_case_only"
_IMPORT_DIAGNOSTIC_MODE = "audit_exception_first_case_only"
_NATIVE_DIAGNOSTIC_CASE_SUFFIX = "001"
_NATIVE_DIAGNOSTIC_MAX_SECONDS = 45
_NATIVE_DIAGNOSTIC_MAX_FRAMES = 32
_IMPORT_DIAGNOSTIC_MAX_SECONDS = 45
_IMPORT_DIAGNOSTIC_MAX_MODULES = 64
_IMPORT_DIAGNOSTIC_MAX_RAW_BYTES = 1_048_576
_IMPORT_DIAGNOSTIC_EXCEPTION_EXIT_CODE = 87
HEADLESS_DISPLAY_MODE = "pyglet_headless_egl"


def implementation_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _number(
    value: Any, *, name: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool):
        raise MeasurementAdapterExecutionError(f"dlo_lab_adapter_{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(f"dlo_lab_adapter_{name}_invalid") from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise MeasurementAdapterExecutionError(f"dlo_lab_adapter_{name}_invalid")
    return result


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeasurementAdapterExecutionError(f"dlo_lab_adapter_{name}_invalid")
    return value


def _vector(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise MeasurementAdapterExecutionError(f"dlo_lab_adapter_{name}_invalid")
    return [_number(item, name=name) for item in value]


def _operating_point(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request["case_manifest"].get("operating_point")
    if not isinstance(raw, Mapping):
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_operating_point_invalid")
    point = dict(raw)
    if point.get("adapter_protocol") != PROTOCOL_ID:
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_protocol_invalid")
    if (
        point.get("length_unit") != "meters"
        or point.get("mass_unit") != "kilograms"
        or point.get("time_unit") != "seconds"
    ):
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_units_invalid")
    direction = _vector(point.get("direction"), name="direction")
    if direction != [1.0, 0.0, 0.0]:
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_direction_invalid")
    timestep = _number(point.get("timestep_s"), name="timestep", minimum=1e-6, maximum=0.01)
    step_count = _integer(point.get("step_count"), name="step_count", minimum=10, maximum=5000)
    n_vertices = _integer(point.get("n_vertices"), name="n_vertices", minimum=6, maximum=256)
    length = _number(point.get("base_length_m"), name="base_length", minimum=0.05, maximum=5.0)
    segment_mass = _number(
        point.get("segment_mass_kg"), name="segment_mass", minimum=1e-6, maximum=10.0
    )
    return {
        "n_vertices": n_vertices,
        "interval_m": length / (n_vertices - 1),
        "base_length_m": length,
        "base_radius_m": _number(
            point.get("base_radius_m"), name="base_radius", minimum=1e-4, maximum=0.1
        ),
        "segment_mass_kg": segment_mass,
        "youngs_modulus_pa": _number(
            point.get("youngs_modulus_pa"), name="youngs_modulus", minimum=1e3, maximum=1e12
        ),
        "shear_modulus_pa": _number(
            point.get("shear_modulus_pa"), name="shear_modulus", minimum=1e3, maximum=1e12
        ),
        "gravity_m_s2": _vector(point.get("gravity_m_s2"), name="gravity"),
        "start_m": _vector(point.get("start_m"), name="start"),
        "direction": direction,
        "timestep_s": timestep,
        "step_count": step_count,
        "substeps": _integer(point.get("substeps"), name="substeps", minimum=1, maximum=32),
        "damping": _number(point.get("damping"), name="damping", minimum=0.0, maximum=1e4),
        "angular_damping": _number(
            point.get("angular_damping"), name="angular_damping", minimum=0.0, maximum=1e4
        ),
        "maximum_tip_displacement_m": _number(
            point.get("maximum_tip_displacement_m"),
            name="maximum_tip_displacement",
            minimum=0.0,
            maximum=10.0,
        ),
        "maximum_segment_strain": _number(
            point.get("maximum_segment_strain"),
            name="maximum_segment_strain",
            minimum=0.0,
            maximum=10.0,
        ),
        "total_mass_kg": segment_mass * (n_vertices - 1),
    }


def _source_commit(genesis: Any) -> str:
    root = Path(genesis.__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip().lower() if result.returncode == 0 else ""


def _simulate(gs: Any, torch: Any, point: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    gs.init(seed=seed, precision="64", logging_level="warning", backend=gs.cuda, debug=True)
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=point["timestep_s"],
                substeps=point["substeps"],
                gravity=tuple(point["gravity_m_s2"]),
            ),
            rod_options=gs.options.RODOptions(
                damping=point["damping"],
                angular_damping=point["angular_damping"],
            ),
            show_viewer=False,
        )
        rod = scene.add_entity(
            material=gs.materials.ROD.Base(
                E=point["youngs_modulus_pa"],
                G=point["shear_modulus_pa"],
                segment_mass=point["segment_mass_kg"],
                segment_radius=point["base_radius_m"],
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=point["n_vertices"],
                interval=point["interval_m"],
                axis="x",
                pos=tuple(point["start_m"]),
            ),
        )
        scene.build(n_envs=1)
        rod.set_fixed_states(fixed_ids=[0, 1])
        sample_steps = {0, point["step_count"] // 2, point["step_count"]}
        trace: list[dict[str, Any]] = []
        for step in range(point["step_count"] + 1):
            if step in sample_steps:
                positions = rod.get_state().pos.detach().cpu().to(torch.float64)[0]
                trace.append(
                    {
                        "step": step,
                        "tip_position_m": [float(item) for item in positions[-1].tolist()],
                    }
                )
            if step < point["step_count"]:
                scene.step()
        final_positions = rod.get_state().pos.detach().cpu().to(torch.float64)[0]
        segment_lengths = torch.linalg.vector_norm(
            final_positions[1:] - final_positions[:-1], dim=1
        )
        initial_tip = torch.tensor(
            [point["start_m"][0] + point["base_length_m"], *point["start_m"][1:]],
            dtype=torch.float64,
            device="cpu",
        )
        displacement = float(torch.linalg.vector_norm(final_positions[-1] - initial_tip))
        maximum_strain = float(torch.max(torch.abs(segment_lengths / point["interval_m"] - 1.0)))
        result = {
            "trace": trace,
            "final_tip_position_m": [float(item) for item in final_positions[-1].tolist()],
            "tip_displacement_m": displacement,
            "maximum_segment_strain": maximum_strain,
            "applied_gravity_force_n": point["total_mass_kg"]
            * math.sqrt(sum(item * item for item in point["gravity_m_s2"])),
        }
        result["trace_digest"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        return result
    finally:
        gs.destroy()


def run_dlo_lab_cable_request(
    request_value: Mapping[str, Any],
    *,
    phase_writer: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    record_phase = phase_writer or (lambda _phase: None)
    runtime = request["runtime_configuration"]
    observations: dict[str, Any] = {
        "engine_version": "unavailable",
        "distribution_version": "unavailable",
        "source_commit": "unavailable",
        "backend_id": runtime["backend_id"],
        "precision": runtime["precision"],
        "seed": runtime["seed"],
        "cpu_fallback_used": False,
        "display_mode": HEADLESS_DISPLAY_MODE,
    }
    implementation = request["implementation"]
    identity_valid = bool(
        implementation["implementation_id"] == IMPLEMENTATION_ID
        and implementation["implementation_version"] == IMPLEMENTATION_VERSION
        and implementation["implementation_digest"] == implementation_digest()
    )
    if not identity_valid:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["dlo_lab_adapter_implementation_identity_mismatch"],
        )
    settings = dict(runtime["solver_settings"])
    if settings != {
        "backend": "cuda",
        "display_mode": HEADLESS_DISPLAY_MODE,
        "import_diagnostic": _IMPORT_DIAGNOSTIC_MODE,
        "native_diagnostic": _NATIVE_DIAGNOSTIC_MODE,
        "replay_count": 2,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }:
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_solver_settings_invalid")
    if runtime["backend_id"] != "dlo-lab-genesis-cuda" or runtime["precision"] != "float64":
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_runtime_configuration_invalid")
    point = _operating_point(request)
    # DLO-Lab's own CUDA examples load PyTorch before Genesis. Preserve that
    # native-library order: Genesis imports Quadrants with RTLD_DEEPBIND on
    # Linux, and loading it first aborted during import in two paid L40S cases.
    record_phase("torch_import_started")
    try:
        import torch
    except (ImportError, OSError):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["dlo_lab_adapter_runtime_unavailable"],
        )
    record_phase("torch_import_completed")
    record_phase("genesis_import_started")
    try:
        import genesis as gs
    except (ImportError, OSError):
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["dlo_lab_adapter_runtime_unavailable"],
        )
    record_phase("genesis_import_completed")
    version = importlib.metadata.version("genesis-world")
    commit = _source_commit(gs)
    observations.update(
        {
            "distribution_version": version,
            "engine_version": version,
            "source_commit": commit,
            "python_version": ".".join(str(item) for item in sys.version_info[:3]),
            "torch_version": str(torch.__version__),
            "torch_distribution_version": importlib.metadata.version("torch"),
            "quadrants_distribution_version": importlib.metadata.version("quadrants"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    )
    if version != EXPECTED_DISTRIBUTION_VERSION or commit != EXPECTED_SOURCE_COMMIT:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["dlo_lab_adapter_source_identity_mismatch"],
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return build_measurement_adapter_worker_result(
            request,
            status="blocked",
            observed_metrics={},
            unsafe_condition_predicted=None,
            runtime_observations=observations,
            failure_codes=["dlo_lab_adapter_cuda_unavailable"],
        )
    record_phase("first_simulation_started")
    first = _simulate(gs, torch, point, seed=runtime["seed"])
    record_phase("first_simulation_completed")
    second = _simulate(gs, torch, point, seed=runtime["seed"])
    record_phase("second_simulation_completed")
    replay_match = first["trace_digest"] == second["trace_digest"]
    unsafe = bool(
        first["tip_displacement_m"] > point["maximum_tip_displacement_m"]
        or first["maximum_segment_strain"] > point["maximum_segment_strain"]
    )
    available_metrics = {
        "state_trajectory": first["tip_displacement_m"],
        "force": first["applied_gravity_force_n"],
        "task_outcome": "deformation_envelope_exceeded"
        if unsafe
        else "within_deformation_envelope",
    }
    requested = set(request["case_manifest"]["requested_metric_ids"])
    metrics = {key: item for key, item in available_metrics.items() if key in requested}
    observations.update(
        {
            "adapter_protocol": PROTOCOL_ID,
            "trace_digest": first["trace_digest"],
            "repeat_trace_digest": second["trace_digest"],
            "deterministic_replay_match": replay_match,
            "final_tip_position_m": first["final_tip_position_m"],
            "tip_displacement_m": first["tip_displacement_m"],
            "maximum_segment_strain": first["maximum_segment_strain"],
            "applied_gravity_force_n": first["applied_gravity_force_n"],
            "q_dlo_qualification_created": False,
            "r5_evidence_created": False,
            "r6_decision_created": False,
            "r7_admission_created": False,
            "physical_success_established": False,
        }
    )
    return build_measurement_adapter_worker_result(
        request,
        status="completed" if replay_match else "failed",
        observed_metrics=metrics,
        unsafe_condition_predicted=unsafe if replay_match else None,
        runtime_observations=observations,
        failure_codes=[] if replay_match else ["dlo_lab_adapter_replay_mismatch"],
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("dlo_lab_adapter_request_not_object")
    return dict(value)


def _phase_writer(path: Path) -> Callable[[str], None]:
    def write(phase: str) -> None:
        path.write_text(
            json.dumps({"phase": phase}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return write


def _classified_native_failure(stderr: bytes, exit_code: int) -> list[str]:
    text = stderr.decode("utf-8", errors="replace").lower()
    codes = [f"dlo_lab_adapter_supervised_worker_exit_nonzero:{exit_code}"]
    if exit_code < 0:
        codes.append(f"dlo_lab_adapter_supervised_worker_signal:{-exit_code}")
    for pattern, code in (
        ("qt.qpa", "dlo_lab_adapter_qt_platform_failure"),
        ("could not load the qt platform plugin", "dlo_lab_adapter_qt_platform_failure"),
        ("omp: error", "dlo_lab_adapter_openmp_runtime_failure"),
        ("libgomp", "dlo_lab_adapter_openmp_runtime_failure"),
        ("cuda_error", "dlo_lab_adapter_cuda_runtime_failure"),
        ("cuda error", "dlo_lab_adapter_cuda_runtime_failure"),
        ("libcuda", "dlo_lab_adapter_cuda_runtime_failure"),
        ("std::system_error", "dlo_lab_adapter_native_system_error"),
        ("invalid argument", "dlo_lab_adapter_native_invalid_argument"),
        ("terminate called", "dlo_lab_adapter_native_termination"),
        ("assertion", "dlo_lab_adapter_native_assertion"),
        ("fatal python error", "dlo_lab_adapter_fatal_python_error"),
    ):
        if pattern in text:
            codes.append(code)
    return sorted(set(codes))


def _probe_script(import_statements: str) -> str:
    indented = "\n".join(f"    {line}" for line in import_statements.splitlines())
    return (
        "try:\n"
        f"{indented}\n"
        "except ImportError:\n"
        f"    raise SystemExit({_IMPORT_UNAVAILABLE_EXIT_CODE})\n"
    )


def _sanitize_native_backtrace(payload: bytes) -> list[dict[str, Any]]:
    """Reduce debugger output to bounded symbols and library basenames.

    Raw debugger output can contain host paths and is therefore never returned.
    """

    frames: list[dict[str, Any]] = []
    for raw_line in payload.decode("utf-8", errors="replace").splitlines():
        match = re.match(r"^#(?P<index>\d+)\s+(?P<body>.+)$", raw_line.strip())
        if match is None:
            continue
        body = match.group("body")
        address = re.match(r"^0x[0-9a-fA-F]+\s+in\s+", body)
        if address is not None:
            body = body[address.end() :]
        module = None
        if " from " in body:
            body, raw_module = body.rsplit(" from ", 1)
            module = Path(raw_module.strip()).name[:160] or None
        if " at " in body:
            body = body.split(" at ", 1)[0]
        symbol = body.split("(", 1)[0].strip()
        symbol = re.sub(r"[^A-Za-z0-9_:+~.<>=,*/& -]", "_", symbol)[:200]
        frame = {
            "index": int(match.group("index")),
            "symbol": symbol or "unknown",
            "module": module,
        }
        frames.append(frame)
        if len(frames) >= _NATIVE_DIAGNOSTIC_MAX_FRAMES:
            break
    return frames


def _native_debugger_observation(
    *, import_statements: str, timeout_seconds: float
) -> dict[str, Any]:
    gdb = shutil.which("gdb")
    if gdb is None:
        return {
            "status": "tool_unavailable",
            "tool": "gdb",
            "raw_output_content_persisted": False,
        }
    bounded_timeout = max(0.001, min(timeout_seconds, _NATIVE_DIAGNOSTIC_MAX_SECONDS))
    command = [
        gdb,
        "--batch",
        "--nx",
        "--nh",
        "-ex",
        "set pagination off",
        "-ex",
        "set print frame-arguments none",
        "-ex",
        "catch throw",
        "-ex",
        "run",
        "-ex",
        "bt 32",
        "--args",
        sys.executable,
        "-c",
        _probe_script(import_statements),
    ]
    diagnostic_env = _headless_subprocess_environment()
    for name in (
        "BLUEPRINT_MEASUREMENT_DLO_INPUT_GET_URL",
        "BLUEPRINT_MEASUREMENT_DLO_OUTPUT_GET_URL",
        "BLUEPRINT_MEASUREMENT_DLO_OUTPUT_PUT_URL",
    ):
        diagnostic_env.pop(name, None)
    try:
        completed = subprocess.run(  # nosec B603 - fixed debugger and probe argv
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=bounded_timeout,
            check=False,
            env=diagnostic_env,
        )
    except subprocess.TimeoutExpired as exc:
        payload = bytes(exc.stdout or b"") + bytes(exc.stderr or b"")
        return {
            "status": "timed_out",
            "tool": "gdb",
            "timeout_seconds": bounded_timeout,
            "raw_output_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "raw_output_bytes": len(payload),
            "raw_output_content_persisted": False,
            "frames": _sanitize_native_backtrace(payload),
        }
    payload = completed.stdout + completed.stderr
    frames = _sanitize_native_backtrace(payload)
    return {
        "status": "captured" if frames else "no_frames",
        "tool": "gdb",
        "return_code": int(completed.returncode),
        "timeout_seconds": bounded_timeout,
        "raw_output_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "raw_output_bytes": len(payload),
        "raw_output_content_persisted": False,
        "frames": frames,
    }


def _import_audit_script(import_statements: str) -> str:
    indented = "\n".join(f"    {line}" for line in import_statements.splitlines())
    return (
        "import os, re, sys\n"
        "def _blueprint_emit(kind, value):\n"
        "    if re.fullmatch(r'[A-Za-z0-9_.]+', value):\n"
        "        os.write(1, ('BLUEPRINT_' + kind + ':' + value + '\\n').encode('ascii'))\n"
        "def _blueprint_import_audit(event, args):\n"
        "    if event != 'import' or not args or not isinstance(args[0], str):\n"
        "        return\n"
        "    _blueprint_emit('IMPORT', args[0])\n"
        "sys.addaudithook(_blueprint_import_audit)\n"
        "try:\n"
        f"{indented}\n"
        "except BaseException as exc:\n"
        "    exc_type = type(exc)\n"
        "    _blueprint_emit('EXCEPTION', exc_type.__module__ + '.' + exc_type.__qualname__)\n"
        "    if isinstance(exc, SystemExit) and isinstance(exc.code, int) and not isinstance(exc.code, bool):\n"
        "        if -255 <= exc.code <= 255:\n"
        "            os.write(1, ('BLUEPRINT_SYSTEM_EXIT:' + str(exc.code) + '\\n').encode('ascii'))\n"
        f"    raise SystemExit({_IMPORT_DIAGNOSTIC_EXCEPTION_EXIT_CODE}) from None\n"
    )


def _import_audit_observation(*, import_statements: str, timeout_seconds: float) -> dict[str, Any]:
    bounded_timeout = max(0.001, min(timeout_seconds, _IMPORT_DIAGNOSTIC_MAX_SECONDS))
    try:
        completed = subprocess.run(  # nosec B603 - fixed interpreter and probe source
            [sys.executable, "-c", _import_audit_script(import_statements)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=bounded_timeout,
            check=False,
            env=_headless_subprocess_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        structured_payload = bytes(exc.stdout or b"")
        stderr_payload = bytes(exc.stderr or b"")
        return_code = None
        status = "timed_out"
    else:
        structured_payload = completed.stdout
        stderr_payload = completed.stderr
        return_code = int(completed.returncode)
        status = "captured"
    payload = structured_payload + stderr_payload
    modules: list[str] = []
    exception_types: list[str] = []
    system_exit_codes: list[int] = []
    for line in structured_payload.decode("utf-8", errors="replace").splitlines():
        if line.startswith("BLUEPRINT_IMPORT:"):
            module = line.removeprefix("BLUEPRINT_IMPORT:")
            if re.fullmatch(r"[A-Za-z0-9_.]+", module):
                modules.append(module[:200])
        elif line.startswith("BLUEPRINT_EXCEPTION:"):
            exception_type = line.removeprefix("BLUEPRINT_EXCEPTION:")
            if re.fullmatch(r"[A-Za-z0-9_.]+", exception_type):
                exception_types.append(exception_type[:200])
        elif line.startswith("BLUEPRINT_SYSTEM_EXIT:"):
            system_exit_code = line.removeprefix("BLUEPRINT_SYSTEM_EXIT:")
            if re.fullmatch(r"-?[0-9]{1,3}", system_exit_code):
                parsed_exit_code = int(system_exit_code)
                if -255 <= parsed_exit_code <= 255:
                    system_exit_codes.append(parsed_exit_code)
    bounded_modules = modules[-_IMPORT_DIAGNOSTIC_MAX_MODULES:]
    bounded_payload = payload[:_IMPORT_DIAGNOSTIC_MAX_RAW_BYTES]
    safe_markers_observed = bool(bounded_modules or exception_types or system_exit_codes)
    return {
        "status": status if status == "timed_out" or safe_markers_observed else "no_safe_markers",
        "tool": "python_audit_hook",
        "return_code": return_code,
        "timeout_seconds": bounded_timeout,
        "raw_output_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "raw_output_bytes": len(payload),
        "raw_output_truncated": len(payload) > len(bounded_payload),
        "raw_output_content_persisted": False,
        "module_count": len(modules),
        "last_modules": bounded_modules,
        "exception_type": exception_types[-1] if exception_types else None,
        "system_exit_code": system_exit_codes[-1] if system_exit_codes else None,
    }


def _supervised_failure_result(
    request: Mapping[str, Any],
    *,
    stderr: bytes,
    exit_code: int | None,
    timed_out: bool,
    phase: str,
    failure_codes: Sequence[str],
    native_diagnostic: Mapping[str, Any] | None = None,
    import_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observations = {
        "engine_version": "unavailable",
        "distribution_version": "unavailable",
        "source_commit": "unavailable",
        "backend_id": request["runtime_configuration"]["backend_id"],
        "precision": request["runtime_configuration"]["precision"],
        "seed": request["runtime_configuration"]["seed"],
        "cpu_fallback_used": False,
        "display_mode": HEADLESS_DISPLAY_MODE,
        "supervised_worker_phase": phase,
        "supervised_worker_exit_code": exit_code,
        "supervised_worker_native_signal": (
            -exit_code if not timed_out and exit_code is not None and exit_code < 0 else None
        ),
        "supervised_worker_timed_out": timed_out,
        "supervised_worker_import_order": ["torch", "genesis"],
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "quadrants_distribution_version": _distribution_version("quadrants"),
        "torch_distribution_version": _distribution_version("torch"),
        "supervised_worker_stderr_digest": "sha256:" + hashlib.sha256(stderr).hexdigest(),
        "supervised_worker_stderr_bytes": len(stderr),
        "supervised_worker_stderr_content_persisted": False,
    }
    if native_diagnostic is not None:
        observations["supervised_worker_native_diagnostic"] = dict(native_diagnostic)
    if import_diagnostic is not None:
        observations["supervised_worker_import_diagnostic"] = dict(import_diagnostic)
    return build_measurement_adapter_worker_result(
        request,
        status="blocked",
        observed_metrics={},
        unsafe_condition_predicted=None,
        runtime_observations=observations,
        failure_codes=sorted(set(failure_codes)),
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _headless_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY"):
        environment.pop(name, None)
    environment["PYGLET_HEADLESS"] = "1"
    environment["PYOPENGL_PLATFORM"] = "egl"
    return environment


def _run_supervised_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="blueprint-dlo-lab-supervisor-") as raw_root:
        root = Path(raw_root)
        request_path = root / "request.json"
        result_path = root / "result.json"
        phase_path = root / "phase.json"
        request_path.write_text(
            json.dumps(dict(request), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + int(request["timeout_seconds"])
        for probe_id, import_statements in _IMPORT_PROBES:
            phase = f"native_import_probe:{probe_id}"
            try:
                probe = subprocess.run(  # nosec B603 - fixed interpreter and probe source
                    [sys.executable, "-c", _probe_script(import_statements)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(0.001, deadline - time.monotonic()),
                    check=False,
                    env=_headless_subprocess_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                return _supervised_failure_result(
                    request,
                    stderr=bytes(exc.stderr or b""),
                    exit_code=None,
                    timed_out=True,
                    phase=phase,
                    failure_codes=["dlo_lab_adapter_native_import_probe_timed_out"],
                )
            probe_exit_code = int(probe.returncode)
            if probe_exit_code == 0:
                continue
            if probe_exit_code == _IMPORT_UNAVAILABLE_EXIT_CODE:
                failure_codes = ["dlo_lab_adapter_runtime_unavailable"]
            else:
                failure_codes = _classified_native_failure(probe.stderr, probe_exit_code)
                failure_codes.append(f"dlo_lab_adapter_native_import_probe_failed:{probe_id}")
            native_diagnostic = None
            import_diagnostic = None
            execution_id = str(request.get("execution_id") or "")
            if (
                probe_id == "quadrants"
                and probe_exit_code < 0
                and request["runtime_configuration"]["solver_settings"].get("native_diagnostic")
                == _NATIVE_DIAGNOSTIC_MODE
                and execution_id.endswith(_NATIVE_DIAGNOSTIC_CASE_SUFFIX)
            ):
                native_diagnostic = _native_debugger_observation(
                    import_statements=import_statements,
                    timeout_seconds=max(0.001, deadline - time.monotonic()),
                )
            if (
                probe_id == "torch_then_genesis"
                and request["runtime_configuration"]["solver_settings"].get("import_diagnostic")
                == _IMPORT_DIAGNOSTIC_MODE
                and execution_id.endswith(_NATIVE_DIAGNOSTIC_CASE_SUFFIX)
            ):
                import_diagnostic = _import_audit_observation(
                    import_statements=import_statements,
                    timeout_seconds=max(0.001, deadline - time.monotonic()),
                )
            return _supervised_failure_result(
                request,
                stderr=probe.stderr,
                exit_code=probe_exit_code,
                timed_out=False,
                phase=phase,
                failure_codes=failure_codes,
                native_diagnostic=native_diagnostic,
                import_diagnostic=import_diagnostic,
            )
        try:
            completed = subprocess.run(  # nosec B603 - fixed module and argv
                [
                    sys.executable,
                    "-m",
                    "blueprint_pipeline.measurement_dlo_lab_cable_adapter",
                    "--direct-worker",
                    "--request",
                    str(request_path),
                    "--output",
                    str(result_path),
                    "--phase-output",
                    str(phase_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(0.001, deadline - time.monotonic()),
                check=False,
                env=_headless_subprocess_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            stderr = bytes(exc.stderr or b"")
            exit_code: int | None = None
            timed_out = True
            failure_codes = ["dlo_lab_adapter_supervised_worker_timed_out"]
        else:
            stderr = completed.stderr
            exit_code = int(completed.returncode)
            timed_out = False
            if exit_code == 0 and result_path.is_file() and not result_path.is_symlink():
                return _load(result_path)
            failure_codes = (
                _classified_native_failure(stderr, exit_code)
                if exit_code != 0
                else ["dlo_lab_adapter_supervised_worker_result_missing"]
            )
        phase = "supervised_worker_not_started"
        try:
            phase_value = json.loads(phase_path.read_text(encoding="utf-8"))
            if isinstance(phase_value, Mapping):
                phase = str(phase_value.get("phase") or phase)
        except (OSError, json.JSONDecodeError):
            pass
        return _supervised_failure_result(
            request,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            phase=phase,
            failure_codes=failure_codes,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an exact-source DLO-Lab CUDA cable case")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-output", type=Path)
    parser.add_argument("--direct-worker", action="store_true")
    args = parser.parse_args(argv)
    if args.direct_worker and args.phase_output is None:
        parser.error("--phase-output is required with --direct-worker")
    request = validate_measurement_adapter_execution_request(_load(args.request))
    result = (
        run_dlo_lab_cable_request(
            request,
            phase_writer=_phase_writer(args.phase_output) if args.phase_output else None,
        )
        if args.direct_worker
        else _run_supervised_worker(request)
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_DISTRIBUTION_VERSION",
    "EXPECTED_SOURCE_COMMIT",
    "HEADLESS_DISPLAY_MODE",
    "IMPLEMENTATION_ID",
    "IMPLEMENTATION_VERSION",
    "PROTOCOL_ID",
    "implementation_digest",
    "main",
    "run_dlo_lab_cable_request",
]
