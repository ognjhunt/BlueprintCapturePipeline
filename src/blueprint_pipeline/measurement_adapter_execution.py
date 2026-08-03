"""Fail-closed execution boundary for measurement benchmark adapters.

The research catalog and adapter descriptors are inert.  This module is the
only repo-local bridge that may launch a measurement adapter for a benchmark
case.  It deliberately supports development execution only: qualification
split execution belongs on an independently controlled runner that can prove
hidden-label isolation, immutable runtime identity, and clean-environment
reruns.

Workers receive one digest-bound JSON request and must write one bounded JSON
result.  Commands are argv-only, run without a shell, inherit only an explicit
environment allowlist, and execute in an isolated temporary working directory.
Receipts bind the request, case, descriptor, command, logs, worker result, and
observed runtime identity.  A receipt is execution evidence, never a method
qualification or a routing authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .measurement_adapter_runtime import (
    validate_measurement_adapter_descriptor,
)
from .measurement_qualification_benchmarks import (
    build_benchmark_prediction,
    validate_benchmark_prediction,
    validate_benchmark_case_manifest,
    validate_qualification_benchmark_spec,
)


EXECUTION_REQUEST_SCHEMA_VERSION = "measurement_adapter_execution_request.v1"
WORKER_RESULT_SCHEMA_VERSION = "measurement_adapter_worker_result.v1"
EXECUTION_RECEIPT_SCHEMA_VERSION = "measurement_adapter_execution_receipt.v1"
EXECUTION_BUNDLE_SCHEMA_VERSION = "measurement_adapter_execution_bundle.v1"

EXECUTION_STATUSES = frozenset(
    {"planned_not_executed", "completed", "blocked", "failed", "timed_out"}
)
EVIDENCE_CLASSES = frozenset(
    {
        "plan_only",
        "development_execution",
        "independent_qualification_execution",
        "failed_execution",
    }
)
WORKER_STATUSES = frozenset({"completed", "blocked", "failed"})
LOCAL_EXECUTION_MODES = frozenset(
    {
        "local_library",
        "isolated_external_conda",
        "isolated_external_conda_or_exact_source_build",
        "isolated_source_checkout",
        "pipeline_native_read_only",
        "dataset_benchmark",
    }
)
MAX_RESULT_BYTES = 10 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 3600
SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "sh",
        "zsh",
    }
)
SAFE_ENV_KEYS = frozenset(
    {
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "MJLIB_PATH",
        "MUJOCO_GL",
        "PATH",
        "PYTHONIOENCODING",
        "PYTHONPATH",
        "SYSTEMROOT",
        "TMPDIR",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)
SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "session",
    "token",
)
SAFE_FALSE_BOUNDARY_KEYS = frozenset({"secrets_persisted"})


class MeasurementAdapterExecutionError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterExecutionError(
            "measurement_adapter_execution_artifact_not_json"
        ) from exc
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bytes_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(_string(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MeasurementAdapterExecutionError(
            "measurement_adapter_execution_timestamp_invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise MeasurementAdapterExecutionError(
            "measurement_adapter_execution_timestamp_timezone_missing"
        )
    return parsed.astimezone(timezone.utc)


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _string(key).lower().replace("-", "_")
            if normalized in SAFE_FALSE_BOUNDARY_KEYS and item is False:
                continue
            if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _validate_digest(value: Any) -> bool:
    raw = _string(value)
    return (
        len(raw) == 71
        and raw.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in raw[7:])
    )


def build_measurement_adapter_execution_request(
    descriptor_value: Mapping[str, Any],
    spec_value: Mapping[str, Any],
    case_value: Mapping[str, Any],
    *,
    execution_id: str,
    implementation_id: str,
    implementation_version: str,
    implementation_digest: str,
    backend_id: str,
    precision: str,
    seed: int,
    solver_settings: Mapping[str, Any],
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Build an inert, exact-bound development execution request."""

    descriptor = validate_measurement_adapter_descriptor(descriptor_value)
    spec = validate_qualification_benchmark_spec(spec_value)
    case = validate_benchmark_case_manifest(case_value)
    errors: list[str] = []
    if not _string(execution_id):
        errors.append("measurement_adapter_execution_id_missing")
    for name, value in (
        ("implementation_id", implementation_id),
        ("implementation_version", implementation_version),
        ("backend_id", backend_id),
        ("precision", precision),
    ):
        if not _string(value):
            errors.append(f"measurement_adapter_execution_{name}_missing")
    if not _validate_digest(implementation_digest):
        errors.append("measurement_adapter_execution_implementation_digest_invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        errors.append("measurement_adapter_execution_seed_invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        errors.append("measurement_adapter_execution_timeout_invalid")
    if case["benchmark_spec_digest"] != spec["benchmark_spec_digest"]:
        errors.append("measurement_adapter_execution_spec_case_mismatch")
    if descriptor["candidate_id"] not in spec["method_ids"]:
        errors.append("measurement_adapter_execution_candidate_outside_spec")
    if case["benchmark_id"] not in descriptor["benchmark_ids"]:
        errors.append("measurement_adapter_execution_benchmark_unsupported")
    if case["split"] != "development":
        errors.append("measurement_adapter_qualification_execution_requires_independent_runner")
    if descriptor["execution_mode"] not in LOCAL_EXECUTION_MODES:
        errors.append("measurement_adapter_execution_mode_not_local")
    if descriptor["access_required"] is True:
        errors.append("measurement_adapter_execution_access_not_admitted")
    settings = dict(solver_settings)
    if not settings or _contains_sensitive_key(settings):
        errors.append("measurement_adapter_execution_solver_settings_invalid")
    if _contains_sensitive_key(case) or _contains_sensitive_key(descriptor):
        errors.append("measurement_adapter_execution_sensitive_input_forbidden")
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    request = {
        "schema_version": EXECUTION_REQUEST_SCHEMA_VERSION,
        "execution_id": execution_id,
        "adapter_descriptor": descriptor,
        "benchmark_spec": spec,
        "case_manifest": case,
        "implementation": {
            "implementation_id": implementation_id,
            "implementation_version": implementation_version,
            "implementation_digest": implementation_digest,
        },
        "runtime_configuration": {
            "target_engine_version": descriptor["target_version"],
            "backend_id": backend_id,
            "precision": precision,
            "seed": seed,
            "solver_settings": settings,
            "solver_settings_digest": _digest(settings, "solver_settings_digest"),
        },
        "timeout_seconds": timeout_seconds,
        "execution_scope": "development_only",
        "qualification_labels_available_to_worker": False,
        "physical_measurements_available_to_worker": False,
        "provider_spend_authorized": False,
        "physical_execution_authorized": False,
        "production_execution_authorized": False,
        "agent_authorized": False,
    }
    request["execution_request_digest"] = _digest(request, "execution_request_digest")
    return validate_measurement_adapter_execution_request(request)


def validate_measurement_adapter_execution_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    request = _clone(value)
    errors: list[str] = []
    if request.get("schema_version") != EXECUTION_REQUEST_SCHEMA_VERSION:
        errors.append("measurement_adapter_execution_request_schema_invalid")
    try:
        descriptor = validate_measurement_adapter_descriptor(request.get("adapter_descriptor", {}))
        spec = validate_qualification_benchmark_spec(request.get("benchmark_spec", {}))
        case = validate_benchmark_case_manifest(request.get("case_manifest", {}))
    except (ValueError, TypeError):
        descriptor, spec, case = {}, {}, {}
        errors.append("measurement_adapter_execution_request_nested_artifact_invalid")
    if not _string(request.get("execution_id")):
        errors.append("measurement_adapter_execution_id_missing")
    implementation = request.get("implementation")
    runtime = request.get("runtime_configuration")
    if not isinstance(implementation, Mapping) or not _validate_digest(
        implementation.get("implementation_digest") if isinstance(implementation, Mapping) else None
    ):
        errors.append("measurement_adapter_execution_implementation_invalid")
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get("solver_settings") if isinstance(runtime, Mapping) else None,
        Mapping,
    ):
        errors.append("measurement_adapter_execution_runtime_invalid")
    elif runtime.get("solver_settings_digest") != _digest(
        dict(runtime["solver_settings"]), "solver_settings_digest"
    ):
        errors.append("measurement_adapter_execution_solver_settings_digest_mismatch")
    if isinstance(runtime, Mapping):
        if descriptor and runtime.get("target_engine_version") != descriptor.get("target_version"):
            errors.append("measurement_adapter_execution_target_version_mismatch")
        if not _string(runtime.get("backend_id")) or not _string(runtime.get("precision")):
            errors.append("measurement_adapter_execution_runtime_identity_missing")
        seed = runtime.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            errors.append("measurement_adapter_execution_seed_invalid")
    timeout = request.get("timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= MAX_TIMEOUT_SECONDS
    ):
        errors.append("measurement_adapter_execution_timeout_invalid")
    if descriptor and spec and case:
        if case.get("benchmark_spec_digest") != spec.get("benchmark_spec_digest"):
            errors.append("measurement_adapter_execution_spec_case_mismatch")
        if descriptor.get("candidate_id") not in spec.get("method_ids", []):
            errors.append("measurement_adapter_execution_candidate_outside_spec")
        if case.get("split") != "development":
            errors.append("measurement_adapter_qualification_execution_requires_independent_runner")
    if request.get("execution_scope") != "development_only":
        errors.append("measurement_adapter_execution_scope_invalid")
    for key in (
        "qualification_labels_available_to_worker",
        "physical_measurements_available_to_worker",
        "provider_spend_authorized",
        "physical_execution_authorized",
        "production_execution_authorized",
        "agent_authorized",
    ):
        if request.get(key) is not False:
            errors.append(f"measurement_adapter_execution_{key}_must_be_false")
    if _contains_sensitive_key(request):
        errors.append("measurement_adapter_execution_sensitive_input_forbidden")
    expected = _digest(request, "execution_request_digest")
    supplied = request.get("execution_request_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_adapter_execution_request_digest_mismatch")
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    request["execution_request_digest"] = expected
    return request


def build_measurement_adapter_worker_result(
    request_value: Mapping[str, Any],
    *,
    status: str,
    observed_metrics: Mapping[str, Any],
    unsafe_condition_predicted: bool | None,
    runtime_observations: Mapping[str, Any],
    failure_codes: Sequence[str] = (),
) -> dict[str, Any]:
    request = validate_measurement_adapter_execution_request(request_value)
    if status not in WORKER_STATUSES:
        raise MeasurementAdapterExecutionError("measurement_adapter_worker_status_invalid")
    metrics = dict(observed_metrics)
    allowed_metrics = set(request["case_manifest"]["requested_metric_ids"])
    errors: list[str] = []
    if not set(metrics) <= allowed_metrics:
        errors.append("measurement_adapter_worker_metric_unknown")
    if unsafe_condition_predicted not in {True, False, None}:
        errors.append("measurement_adapter_worker_unsafe_prediction_invalid")
    observations = dict(runtime_observations)
    for key in ("engine_version", "backend_id", "precision", "seed"):
        if key not in observations:
            errors.append(f"measurement_adapter_worker_runtime_{key}_missing")
    failures = sorted({_string(item) for item in failure_codes if _string(item)})
    if status == "completed" and failures:
        errors.append("measurement_adapter_worker_completed_with_failures")
    if status != "completed" and not failures:
        errors.append("measurement_adapter_worker_failure_code_missing")
    if _contains_sensitive_key(observations) or _contains_sensitive_key(metrics):
        errors.append("measurement_adapter_worker_sensitive_output_forbidden")
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    result = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "execution_id": request["execution_id"],
        "execution_request_digest": request["execution_request_digest"],
        "candidate_id": request["adapter_descriptor"]["candidate_id"],
        "adapter_descriptor_digest": request["adapter_descriptor"]["adapter_descriptor_digest"],
        "benchmark_spec_digest": request["benchmark_spec"]["benchmark_spec_digest"],
        "case_manifest_digest": request["case_manifest"]["case_manifest_digest"],
        "split": request["case_manifest"]["split"],
        "status": status,
        "observed_metrics": metrics,
        "unsafe_condition_predicted": unsafe_condition_predicted,
        "runtime_observations": observations,
        "failure_codes": failures,
        "qualification_labels_accessed": False,
        "physical_measurements_accessed": False,
        "vendor_graded": False,
        "agent_graded": False,
        "physical_success_established": False,
    }
    result["worker_result_digest"] = _digest(result, "worker_result_digest")
    return validate_measurement_adapter_worker_result(result, request=request)


def validate_measurement_adapter_worker_result(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = _clone(value)
    errors: list[str] = []
    if result.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION:
        errors.append("measurement_adapter_worker_result_schema_invalid")
    if result.get("status") not in WORKER_STATUSES:
        errors.append("measurement_adapter_worker_status_invalid")
    if not isinstance(result.get("observed_metrics"), Mapping):
        errors.append("measurement_adapter_worker_metrics_invalid")
    if not isinstance(result.get("runtime_observations"), Mapping):
        errors.append("measurement_adapter_worker_runtime_invalid")
    if result.get("unsafe_condition_predicted") not in {True, False, None}:
        errors.append("measurement_adapter_worker_unsafe_prediction_invalid")
    failures = result.get("failure_codes")
    if not isinstance(failures, list):
        errors.append("measurement_adapter_worker_failure_codes_invalid")
    elif result.get("status") == "completed" and failures:
        errors.append("measurement_adapter_worker_completed_with_failures")
    elif result.get("status") != "completed" and not failures:
        errors.append("measurement_adapter_worker_failure_code_missing")
    for key in (
        "qualification_labels_accessed",
        "physical_measurements_accessed",
        "vendor_graded",
        "agent_graded",
        "physical_success_established",
    ):
        if result.get(key) is not False:
            errors.append(f"measurement_adapter_worker_{key}_must_be_false")
    if request is not None:
        checked = validate_measurement_adapter_execution_request(request)
        bindings = {
            "execution_id": checked["execution_id"],
            "execution_request_digest": checked["execution_request_digest"],
            "candidate_id": checked["adapter_descriptor"]["candidate_id"],
            "adapter_descriptor_digest": checked["adapter_descriptor"]["adapter_descriptor_digest"],
            "benchmark_spec_digest": checked["benchmark_spec"]["benchmark_spec_digest"],
            "case_manifest_digest": checked["case_manifest"]["case_manifest_digest"],
            "split": checked["case_manifest"]["split"],
        }
        for key, expected_value in bindings.items():
            if result.get(key) != expected_value:
                errors.append(f"measurement_adapter_worker_binding_mismatch:{key}")
        if isinstance(result.get("observed_metrics"), Mapping) and not set(
            result["observed_metrics"]
        ) <= set(checked["case_manifest"]["requested_metric_ids"]):
            errors.append("measurement_adapter_worker_metric_unknown")
    if _contains_sensitive_key(result):
        errors.append("measurement_adapter_worker_sensitive_output_forbidden")
    expected_digest = _digest(result, "worker_result_digest")
    supplied = result.get("worker_result_digest")
    if supplied is not None and supplied != expected_digest:
        errors.append("measurement_adapter_worker_result_digest_mismatch")
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    result["worker_result_digest"] = expected_digest
    return result


def validate_measurement_adapter_execution_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    if receipt.get("schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION:
        errors.append("measurement_adapter_execution_receipt_schema_invalid")
    if receipt.get("status") not in EXECUTION_STATUSES:
        errors.append("measurement_adapter_execution_receipt_status_invalid")
    if receipt.get("evidence_class") not in EVIDENCE_CLASSES:
        errors.append("measurement_adapter_execution_evidence_class_invalid")
    for key in (
        "execution_request_digest",
        "adapter_descriptor_digest",
        "benchmark_spec_digest",
        "case_manifest_digest",
        "command_digest",
        "stdout_digest",
        "stderr_digest",
    ):
        if not _validate_digest(receipt.get(key)):
            errors.append(f"measurement_adapter_execution_receipt_{key}_invalid")
    if receipt.get("worker_result_digest") is not None and not _validate_digest(
        receipt.get("worker_result_digest")
    ):
        errors.append("measurement_adapter_execution_receipt_worker_result_digest_invalid")
    try:
        started = _parse_time(receipt.get("started_at"))
        finished = _parse_time(receipt.get("finished_at"))
        if finished < started:
            errors.append("measurement_adapter_execution_receipt_time_order_invalid")
    except MeasurementAdapterExecutionError as exc:
        errors.extend(exc.codes)
    if receipt.get("evidence_class") == "independent_qualification_execution":
        if receipt.get("split") != "qualification":
            errors.append("measurement_adapter_independent_execution_split_invalid")
        for key in (
            "executor_independent_of_candidate",
            "clean_environment_verified",
            "immutable_runtime_identity_verified",
        ):
            if receipt.get(key) is not True:
                errors.append(f"measurement_adapter_independent_execution_{key}_required")
    elif receipt.get("evidence_class") == "development_execution":
        if receipt.get("split") != "development":
            errors.append("measurement_adapter_development_execution_split_invalid")
    status = receipt.get("status")
    failures = receipt.get("failure_codes")
    if not isinstance(failures, list):
        errors.append("measurement_adapter_execution_receipt_failure_codes_invalid")
    elif status == "completed" and failures:
        errors.append("measurement_adapter_execution_completed_with_failures")
    elif status != "completed" and not failures:
        errors.append("measurement_adapter_execution_failure_code_missing")
    if status == "completed" and (
        receipt.get("exit_code") != 0 or receipt.get("worker_result_digest") is None
    ):
        errors.append("measurement_adapter_execution_completed_binding_invalid")
    if status == "planned_not_executed" and (
        receipt.get("evidence_class") != "plan_only"
        or receipt.get("exit_code") is not None
        or receipt.get("worker_result_digest") is not None
    ):
        errors.append("measurement_adapter_execution_plan_only_boundary_invalid")
    for key in (
        "secrets_persisted",
        "qualification_labels_accessed",
        "provider_spend_authorized",
        "physical_execution_authorized",
        "production_route_eligible",
        "r6_qualification_decision_created",
        "r7_catalog_admission_created",
        "agent_authorized",
        "stdout_content_persisted",
        "stderr_content_persisted",
    ):
        if receipt.get(key) is not False:
            errors.append(f"measurement_adapter_execution_receipt_{key}_must_be_false")
    if _contains_sensitive_key(receipt):
        errors.append("measurement_adapter_execution_receipt_sensitive_output_forbidden")
    expected = _digest(receipt, "execution_receipt_digest")
    supplied = receipt.get("execution_receipt_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_adapter_execution_receipt_digest_mismatch")
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    receipt["execution_receipt_digest"] = expected
    return receipt


def _safe_command(command_argv: Sequence[str]) -> tuple[list[str], str]:
    argv = [_string(item) for item in command_argv]
    if not argv or any(not item or "\x00" in item for item in argv):
        raise MeasurementAdapterExecutionError("measurement_adapter_execution_command_invalid")
    executable_name = Path(argv[0]).name.lower()
    if executable_name in SHELL_EXECUTABLES:
        raise MeasurementAdapterExecutionError("measurement_adapter_execution_shell_forbidden")
    resolved = argv[0] if Path(argv[0]).is_absolute() else shutil.which(argv[0])
    if not resolved or not Path(resolved).is_file():
        raise MeasurementAdapterExecutionError(
            "measurement_adapter_execution_executable_unavailable"
        )
    # Preserve a virtual-environment interpreter symlink. Resolving it to the
    # base interpreter changes Python's environment discovery and therefore
    # the observed package/runtime identity.
    argv[0] = str(Path(resolved).absolute())
    command_digest = (
        "sha256:" + hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
    )
    return argv, command_digest


def _safe_environment(temporary_root: Path) -> dict[str, str]:
    result = {
        key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS and _string(value)
    }
    # The worker must import the same source tree that produced the request's
    # implementation digest.  Editable environments can otherwise redirect a
    # child launched from a worktree to a different checkout.  Bind the current
    # package root first and make any explicitly allowed caller entries absolute
    # before changing the child working directory to ``temporary_root``.
    package_root = str(Path(__file__).resolve().parents[1])
    caller_pythonpath: list[str] = []
    for raw_entry in result.get("PYTHONPATH", "").split(os.pathsep):
        entry = raw_entry.strip()
        if not entry:
            continue
        path = Path(entry)
        caller_pythonpath.append(str(path if path.is_absolute() else (Path.cwd() / path).resolve()))
    result["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([package_root, *caller_pythonpath]))
    # Some native runtimes resolve caches during shared-library initialization
    # and abort when HOME is absent.  Never inherit the operator's real home;
    # provide a per-execution sandbox home and cache root instead.
    result["HOME"] = str(temporary_root)
    result["XDG_CACHE_HOME"] = str(temporary_root / ".cache")
    result["TMPDIR"] = str(temporary_root)
    result.setdefault("PATH", os.defpath)
    result.setdefault("PYTHONIOENCODING", "utf-8")
    return result


def _log_summary(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "digest": _bytes_digest(path),
        "bytes": size,
        "content_persisted": False,
    }


def _subprocess_failure_codes(stderr_path: Path, exit_code: int) -> list[str]:
    """Return bounded diagnostic categories without persisting worker stderr."""

    try:
        text = stderr_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        text = ""
    codes = [f"worker_exit_nonzero:{exit_code}"]
    if exit_code < 0:
        codes.append(f"worker_signal:{-exit_code}")
    patterns = (
        ("qt.qpa", "worker_stderr_qt_platform_failure"),
        ("could not load the qt platform plugin", "worker_stderr_qt_platform_failure"),
        ("omp: error", "worker_stderr_openmp_runtime_failure"),
        ("libgomp", "worker_stderr_openmp_runtime_failure"),
        ("cuda_error", "worker_stderr_cuda_runtime_failure"),
        ("cuda error", "worker_stderr_cuda_runtime_failure"),
        ("libcuda", "worker_stderr_cuda_runtime_failure"),
        ("driver version is insufficient", "worker_stderr_cuda_runtime_failure"),
        ("terminate called", "worker_stderr_native_termination"),
        ("assertion", "worker_stderr_native_assertion"),
        ("fatal python error", "worker_stderr_fatal_python_error"),
        ("traceback (most recent call last)", "worker_stderr_python_traceback"),
    )
    codes.extend(code for pattern, code in patterns if pattern in text)
    return sorted(set(codes))


def _receipt(
    request: Mapping[str, Any],
    *,
    status: str,
    command_digest: str,
    executable: str,
    argc: int,
    started: datetime,
    finished: datetime,
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
    exit_code: int | None,
    worker_result: Mapping[str, Any] | None,
    failure_codes: Sequence[str],
) -> dict[str, Any]:
    result = dict(worker_result) if worker_result is not None else None
    evidence_class = "development_execution" if status == "completed" else "failed_execution"
    if status == "planned_not_executed":
        evidence_class = "plan_only"
    receipt = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "execution_id": request["execution_id"],
        "execution_request_digest": request["execution_request_digest"],
        "candidate_id": request["adapter_descriptor"]["candidate_id"],
        "adapter_descriptor_digest": request["adapter_descriptor"]["adapter_descriptor_digest"],
        "benchmark_spec_digest": request["benchmark_spec"]["benchmark_spec_digest"],
        "case_manifest_digest": request["case_manifest"]["case_manifest_digest"],
        "split": request["case_manifest"]["split"],
        "status": status,
        "evidence_class": evidence_class,
        "executor_id": "blueprint-local-measurement-adapter-runner",
        "executor_independent_of_candidate": False,
        "clean_environment_verified": False,
        "immutable_runtime_identity_verified": False,
        "command_digest": command_digest,
        "command_executable": Path(executable).name,
        "command_argc": argc,
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "duration_seconds": max(0.0, (finished - started).total_seconds()),
        "exit_code": exit_code,
        "worker_result_digest": result.get("worker_result_digest") if result else None,
        "stdout_digest": stdout["digest"],
        "stdout_bytes": stdout["bytes"],
        "stdout_content_persisted": stdout["content_persisted"],
        "stderr_digest": stderr["digest"],
        "stderr_bytes": stderr["bytes"],
        "stderr_content_persisted": stderr["content_persisted"],
        "runtime_observations": result.get("runtime_observations", {}) if result else {},
        "host_runtime": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "failure_codes": sorted({_string(item) for item in failure_codes if _string(item)}),
        "host_process_isolation_only": True,
        "network_isolation_verified": False,
        "filesystem_isolation_verified": False,
        "secrets_persisted": False,
        "qualification_labels_accessed": False,
        "provider_spend_authorized": False,
        "physical_execution_authorized": False,
        "production_route_eligible": False,
        "r6_qualification_decision_created": False,
        "r7_catalog_admission_created": False,
        "agent_authorized": False,
    }
    receipt["execution_receipt_digest"] = _digest(receipt, "execution_receipt_digest")
    return validate_measurement_adapter_execution_receipt(receipt)


def run_measurement_adapter_execution(
    request_value: Mapping[str, Any],
    *,
    command_argv: Sequence[str],
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or execute a local development adapter and return a bound bundle."""

    request = validate_measurement_adapter_execution_request(request_value)
    argv, command_digest = _safe_command(command_argv)
    started = _now()
    with tempfile.TemporaryDirectory(prefix="blueprint-measurement-adapter-") as raw_root:
        root = Path(raw_root).resolve()
        request_path = root / "request.json"
        result_path = root / "result.json"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        status = "planned_not_executed"
        exit_code: int | None = None
        worker_result: dict[str, Any] | None = None
        failures: list[str] = ["explicit_execution_gate_not_set"]
        if execute:
            failures = []
            try:
                with (
                    stdout_path.open("wb") as stdout_stream,
                    stderr_path.open("wb") as stderr_stream,
                ):
                    completed = subprocess.run(  # nosec B603 - exact argv, no shell
                        [*argv, "--request", str(request_path), "--output", str(result_path)],
                        cwd=root,
                        env=_safe_environment(root),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        timeout=request["timeout_seconds"],
                        check=False,
                    )
                exit_code = completed.returncode
                if exit_code != 0:
                    status = "failed"
                    failures.extend(_subprocess_failure_codes(stderr_path, exit_code))
                elif not result_path.is_file() or result_path.is_symlink():
                    status = "failed"
                    failures.append("worker_result_missing_or_unsafe")
                elif result_path.stat().st_size > MAX_RESULT_BYTES:
                    status = "failed"
                    failures.append("worker_result_too_large")
                else:
                    try:
                        raw_result = json.loads(result_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        status = "failed"
                        failures.append("worker_result_unreadable")
                    else:
                        if not isinstance(raw_result, Mapping):
                            status = "failed"
                            failures.append("worker_result_not_object")
                        else:
                            try:
                                worker_result = validate_measurement_adapter_worker_result(
                                    raw_result, request=request
                                )
                            except MeasurementAdapterExecutionError as exc:
                                status = "failed"
                                failures.extend(exc.codes)
                            else:
                                status = worker_result["status"]
                                failures.extend(worker_result["failure_codes"])
            except subprocess.TimeoutExpired:
                status = "timed_out"
                failures.append("worker_timeout")
        finished = _now()
        stdout_summary = _log_summary(stdout_path)
        stderr_summary = _log_summary(stderr_path)
        receipt = _receipt(
            request,
            status=status,
            command_digest=command_digest,
            executable=argv[0],
            argc=len(argv) + 4,
            started=started,
            finished=finished,
            stdout=stdout_summary,
            stderr=stderr_summary,
            exit_code=exit_code,
            worker_result=worker_result,
            failure_codes=failures,
        )
    prediction = None
    if status == "completed" and worker_result is not None:
        prediction = build_benchmark_prediction(
            request["adapter_descriptor"],
            request["case_manifest"],
            observed_metrics=worker_result["observed_metrics"],
            unsafe_condition_predicted=worker_result["unsafe_condition_predicted"],
            execution_receipt=receipt,
        )
    bundle = {
        "schema_version": EXECUTION_BUNDLE_SCHEMA_VERSION,
        "request": request,
        "receipt": receipt,
        "worker_result": worker_result,
        "prediction": prediction,
        "qualification_created": False,
        "catalog_mutated": False,
        "production_route_created": False,
        "physical_success_established": False,
    }
    bundle["execution_bundle_digest"] = _digest(bundle, "execution_bundle_digest")
    return validate_measurement_adapter_execution_bundle(bundle)


def validate_measurement_adapter_execution_bundle(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = _clone(value)
    errors: list[str] = []
    if bundle.get("schema_version") != EXECUTION_BUNDLE_SCHEMA_VERSION:
        errors.append("measurement_adapter_execution_bundle_schema_invalid")
    try:
        request = validate_measurement_adapter_execution_request(bundle.get("request", {}))
        receipt = validate_measurement_adapter_execution_receipt(bundle.get("receipt", {}))
    except (ValueError, TypeError):
        request, receipt = {}, {}
        errors.append("measurement_adapter_execution_bundle_nested_artifact_invalid")
    if request and receipt:
        for actual, expected, name in (
            (
                receipt.get("execution_request_digest"),
                request.get("execution_request_digest"),
                "execution_request_digest",
            ),
            (receipt.get("execution_id"), request.get("execution_id"), "execution_id"),
            (
                receipt.get("candidate_id"),
                request["adapter_descriptor"].get("candidate_id"),
                "candidate_id",
            ),
            (
                receipt.get("adapter_descriptor_digest"),
                request["adapter_descriptor"].get("adapter_descriptor_digest"),
                "adapter_descriptor_digest",
            ),
            (
                receipt.get("benchmark_spec_digest"),
                request["benchmark_spec"].get("benchmark_spec_digest"),
                "benchmark_spec_digest",
            ),
            (
                receipt.get("case_manifest_digest"),
                request["case_manifest"].get("case_manifest_digest"),
                "case_manifest_digest",
            ),
        ):
            if actual != expected:
                errors.append(f"measurement_adapter_execution_bundle_binding_mismatch:{name}")
    worker_result = bundle.get("worker_result")
    if receipt and receipt.get("worker_result_digest") is not None:
        if not isinstance(worker_result, Mapping):
            errors.append("measurement_adapter_execution_bundle_worker_result_missing")
        elif request:
            try:
                checked_worker = validate_measurement_adapter_worker_result(
                    worker_result, request=request
                )
            except MeasurementAdapterExecutionError as exc:
                errors.extend(exc.codes)
            else:
                if checked_worker["worker_result_digest"] != receipt.get("worker_result_digest"):
                    errors.append("measurement_adapter_execution_bundle_worker_result_mismatch")
    elif worker_result is not None:
        errors.append("measurement_adapter_execution_bundle_worker_result_forbidden")
    prediction = bundle.get("prediction")
    if receipt and receipt.get("status") == "completed":
        if not isinstance(prediction, Mapping):
            errors.append("measurement_adapter_execution_bundle_prediction_missing")
        else:
            try:
                checked_prediction = validate_benchmark_prediction(prediction)
            except ValueError as exc:
                errors.append("measurement_adapter_execution_bundle_prediction_invalid:" + str(exc))
            else:
                if checked_prediction.get("execution_receipt_digest") != receipt.get(
                    "execution_receipt_digest"
                ):
                    errors.append("measurement_adapter_execution_bundle_prediction_mismatch")
    elif prediction is not None:
        errors.append("measurement_adapter_execution_bundle_prediction_forbidden")
    for key in (
        "qualification_created",
        "catalog_mutated",
        "production_route_created",
        "physical_success_established",
    ):
        if bundle.get(key) is not False:
            errors.append(f"measurement_adapter_execution_bundle_{key}_must_be_false")
    expected = _digest(bundle, "execution_bundle_digest")
    supplied = bundle.get("execution_bundle_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_adapter_execution_bundle_digest_mismatch")
    if errors:
        raise MeasurementAdapterExecutionError(*errors)
    bundle["execution_bundle_digest"] = expected
    return bundle


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementAdapterExecutionError(
            "measurement_adapter_execution_input_unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise MeasurementAdapterExecutionError("measurement_adapter_execution_input_not_object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run a local development measurement adapter"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-arg", action="append", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    bundle = run_measurement_adapter_execution(
        _load_object(args.request),
        command_argv=args.command_arg,
        execute=args.execute,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if bundle["receipt"]["status"] in {"completed", "planned_not_executed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_CLASSES",
    "EXECUTION_BUNDLE_SCHEMA_VERSION",
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "EXECUTION_REQUEST_SCHEMA_VERSION",
    "EXECUTION_STATUSES",
    "MeasurementAdapterExecutionError",
    "WORKER_RESULT_SCHEMA_VERSION",
    "build_measurement_adapter_execution_request",
    "build_measurement_adapter_worker_result",
    "main",
    "run_measurement_adapter_execution",
    "validate_measurement_adapter_execution_bundle",
    "validate_measurement_adapter_execution_receipt",
    "validate_measurement_adapter_execution_request",
    "validate_measurement_adapter_worker_result",
]
