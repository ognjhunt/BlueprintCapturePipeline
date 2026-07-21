"""Fail-closed ovrtx and ovphysx external-worker preflight contracts.

The adapters intentionally do not import prerelease Omniverse wheels into the
core process. An explicitly gated Linux/CUDA worker must write a JSON report;
Blueprint verifies its identity, outputs, checks, and cold/warm repeatability.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, parse_bool, read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256, executable_identity, run_json_worker
from .local_capture import resolve_local_capture_context
from .nvidia_experiment_resource import (
    load_resource_closeout,
    load_resource_context,
    resource_stop_evidence,
)
from .nvidia_siggraph_policy import evaluate_stop_rules


ENABLE_ENV = "BLUEPRINT_ALLOW_OMNIVERSE_EXTERNAL_PREFLIGHT"
SOURCE_URLS = {
    "ovrtx": "https://github.com/NVIDIA-Omniverse/ovrtx",
    "ovphysx": "https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx",
}
DIRECTORIES = {"ovrtx": "sensor_preflight", "ovphysx": "physics_preflight"}
REQUEST_SCHEMAS = {
    "ovrtx": "ovrtx_preflight_request.v1",
    "ovphysx": "ovphysx_preflight_request.v1",
}
RESULT_SCHEMAS = {
    "ovrtx": "ovrtx_preflight_result.v1",
    "ovphysx": "ovphysx_preflight_result.v1",
}
RECEIPT_SCHEMAS = {
    "ovrtx": "ovrtx_preflight_runtime_receipt.v1",
    "ovphysx": "ovphysx_preflight_runtime_receipt.v1",
}
CLAIM_SCHEMAS = {
    "ovrtx": "ovrtx_preflight_claim_boundary.v1",
    "ovphysx": "ovphysx_preflight_claim_boundary.v1",
}
REQUIRED_CHECKS = {
    "ovrtx": (
        "usd_scene_load",
        "requested_sensor_outputs_nonempty",
        "sensor_metadata_complete",
    ),
    "ovphysx": (
        "usd_scene_load",
        "gravity_and_rigid_body_integration",
        "collider_presence_and_penetration",
        "joint_and_limit_inspection",
        "mass_and_friction_bounds",
        "fixed_step_state_snapshot",
    ),
}
DEFAULT_SENSOR_MODALITIES = ("rgb", "depth", "semantic_segmentation")
CHECK_FAILURE_CLASSES = {
    "usd_scene_load": "usd_scene_load",
    "requested_sensor_outputs_nonempty": "empty_sensor_output",
    "sensor_metadata_complete": "sensor_metadata_loss",
    "semantic_id_map": "semantic_id_map_loss",
    "particlefield_gaussian_splat_render": "particlefield_render_failure",
    "dynamic_transform_update": "dynamic_transform_update_failure",
    "robot_and_target_visibility": "robot_or_target_visibility_failure",
    "lidar_structured_output": "lidar_structured_output_failure",
    "radar_structured_output": "radar_structured_output_failure",
    "gravity_and_rigid_body_integration": "nonfinite_or_static_rigid_body_state",
    "collider_presence_and_penetration": "gross_initial_penetration",
    "joint_and_limit_inspection": "invalid_joint_limits",
    "mass_and_friction_bounds": "missing_or_out_of_bounds_physical_properties",
    "fixed_step_state_snapshot": "fixed_step_snapshot_failure",
    "simple_articulation_motion": "articulation_motion_failure",
}

COMMON_CLAIM_BOUNDARY: dict[str, Any] = {
    "raw_capture_authority_preserved": True,
    "support_artifact_only": True,
    "isaac_scene_parity_proven": False,
    "isaac_sim_execution_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_task_success_proven": False,
    "rank_fidelity_result_proven": False,
    "real_sensor_correlation_proven": False,
    "real_world_task_success_proven": False,
    "deployment_ready": False,
    "public_claim_upgrade_allowed": False,
    "advisory_only": True,
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values: Iterable[Any] = (
        [value] if isinstance(value, str) else value if isinstance(value, Iterable) else [value]
    )
    return list(dict.fromkeys(text for item in values if (text := _string(item))))


def _claim_boundary(component: str) -> dict[str, Any]:
    boundary = dict(COMMON_CLAIM_BOUNDARY)
    boundary.update(
        {
            "artifact_purpose": (
                "experimental_rtx_sensor_preflight"
                if component == "ovrtx"
                else "experimental_standalone_physics_smoke_preflight"
            ),
            "component": component,
            "component_pass_proven": False,
        }
    )
    if component == "ovrtx":
        boundary["physics_simulation_proven"] = False
    else:
        boundary["sensor_simulation_proven"] = False
    return boundary


def _path_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def inspect_usd_features(path: Path) -> dict[str, Any]:
    """Identify compatibility-sensitive scene features without importing Omniverse.

    Text inspection works for USDA fixtures. When pxr is available, composed prim
    type names and references cover binary USD and sublayered scenes as well.
    The result only selects required canary checks; it is not a conformance claim.
    """

    text = ""
    try:
        if path.suffix.lower() in {".usda", ".usd"} and path.stat().st_size <= 2_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    type_names: set[str] = set()
    prim_paths: list[str] = []
    references: list[str] = []
    try:
        from pxr import Usd  # type: ignore[import-not-found]

        stage = Usd.Stage.Open(str(path))
        if stage:
            for prim in stage.Traverse():
                type_names.add(str(prim.GetTypeName() or ""))
                prim_paths.append(str(prim.GetPath()))
                references.extend(str(item) for item in prim.GetMetadata("references") or [])
    except (ImportError, RuntimeError, TypeError):
        pass
    haystack = "\n".join([text, *type_names, *prim_paths, *references]).lower()
    return {
        "inspection_method": "text_and_optional_pxr_composition",
        "particlefield_gaussian_splat": (
            "particlefield3dgaussiansplat" in haystack or "gaussiansplat" in haystack
        ),
        "robot_reference_or_prim": any(
            token in haystack for token in ("/robot", "/g1", "unitree", "humanoid")
        ),
        "semantic_schema_or_labels": any(
            token in haystack for token in ("semantic", "semantics:", "class_", "label")
        ),
        "time_samples": "timesamples" in haystack,
        "observed_type_names": sorted(value for value in type_names if value),
    }


def required_checks_for(
    component: str,
    *,
    configuration: Mapping[str, Any],
    usd_features: Mapping[str, Any],
    required_output_kinds: Sequence[str],
) -> tuple[str, ...]:
    checks = list(REQUIRED_CHECKS[component])
    if component == "ovrtx":
        if usd_features.get("particlefield_gaussian_splat") is True:
            checks.append("particlefield_gaussian_splat_render")
        if configuration.get("episode_mode") is True or usd_features.get("time_samples") is True:
            checks.append("dynamic_transform_update")
        if "semantic_segmentation" in required_output_kinds:
            checks.append("semantic_id_map")
        if configuration.get("robot_prim_path") or configuration.get("target_prim_path"):
            checks.append("robot_and_target_visibility")
        if "lidar" in required_output_kinds:
            checks.append("lidar_structured_output")
        if "radar" in required_output_kinds:
            checks.append("radar_structured_output")
    elif configuration.get("articulation_prim_path"):
        checks.append("simple_articulation_motion")
    return tuple(dict.fromkeys(checks))


def validate_runtime_expectations(component: str, configuration: Mapping[str, Any]) -> list[str]:
    expectations = _mapping(configuration.get("runtime_expectations"))
    blockers: list[str] = []
    if not _string(expectations.get("python_version")):
        blockers.append("runtime_expectation_missing:python_version")
    if component == "ovrtx":
        for field in ("cuda_version", "driver_version", "gpu_uuid", "shader_configuration_id"):
            if not _string(expectations.get(field)):
                blockers.append(f"runtime_expectation_missing:{field}")
    else:
        device = _string(configuration.get("device")).lower()
        if device not in {"cpu", "cuda", "gpu"}:
            blockers.append("ovphysx_device_must_be_cpu_cuda_or_gpu")
        if not _string(expectations.get("solver_configuration_id")):
            blockers.append("runtime_expectation_missing:solver_configuration_id")
        if device in {"cuda", "gpu"}:
            for field in ("cuda_version", "driver_version", "gpu_uuid"):
                if not _string(expectations.get(field)):
                    blockers.append(f"runtime_expectation_missing:{field}")
    return blockers


def _normalize_checks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, list):
        return []
    checks: list[dict[str, Any]] = []
    for value in raw_checks:
        if not isinstance(value, Mapping):
            continue
        name = _string(value.get("name") or value.get("check"))
        if not name:
            continue
        checks.append(
            {
                "name": name,
                "status": _string(value.get("status")).lower() or "unknown",
                "details": _mapping(value.get("details")),
                "message": _string(value.get("message")) or None,
            }
        )
    return checks


def _normalize_outputs(
    payload: Mapping[str, Any],
    *,
    output_directory: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_outputs = payload.get("outputs") or payload.get("artifacts")
    if not isinstance(raw_outputs, list):
        return [], ["worker_report_missing_outputs"]
    outputs: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, value in enumerate(raw_outputs):
        if not isinstance(value, Mapping):
            blockers.append(f"worker_output_{index}_not_mapping")
            continue
        kind = _string(value.get("kind") or value.get("modality") or value.get("name"))
        raw_path = _string(value.get("path"))
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else output_directory / candidate
        exists = path.is_file() and path.stat().st_size > 0
        inside = _path_within(path, output_directory)
        if not inside:
            blockers.append(f"worker_output_outside_preflight_directory:{kind or index}")
        if not exists:
            blockers.append(f"worker_output_missing_or_empty:{kind or index}")
        outputs.append(
            {
                "kind": kind or f"output_{index}",
                "path": str(path.resolve()),
                "inside_preflight_directory": inside,
                "exists_nonempty": exists,
                "sha256": sha256_file(path) if exists else None,
                "bytes": path.stat().st_size if exists else 0,
                "metadata": _mapping(value.get("metadata")),
            }
        )
    return outputs, blockers


def _reported_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    tool = _mapping(payload.get("tool") or payload.get("component"))
    runtime = _mapping(payload.get("runtime"))
    return {
        "component": _string(payload.get("component_name") or tool.get("name")) or None,
        "component_version": _string(payload.get("component_version") or tool.get("version"))
        or None,
        "source_revision": _string(payload.get("source_revision") or tool.get("source_revision"))
        or None,
        "python_version": _string(runtime.get("python_version")) or None,
        "cuda_version": _string(runtime.get("cuda_version")) or None,
        "driver_version": _string(runtime.get("driver_version")) or None,
        "gpu_identity": _mapping(runtime.get("gpu_identity") or runtime.get("gpu")),
        "library_versions": _mapping(runtime.get("library_versions")),
    }


def _normalize_run(
    *,
    component: str,
    raw_report_path: Path,
    execution: Mapping[str, Any],
    output_directory: Path,
    expected_version: str,
    expected_revision: str,
    required_checks: Sequence[str],
    required_output_kinds: Sequence[str],
    required_output_metadata: Mapping[str, Sequence[str]],
    expected_configuration_sha256: str,
    expected_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if raw_report_path.is_file():
        loaded = read_json_any(raw_report_path)
        payload = dict(loaded) if isinstance(loaded, Mapping) else {}
    identity = _reported_identity(payload)
    checks = _normalize_checks(payload)
    outputs, output_blockers = _normalize_outputs(payload, output_directory=output_directory)
    blockers = list(output_blockers)
    identity_verified = bool(
        identity["component"] == component
        and identity["component_version"] == expected_version
        and identity["source_revision"] == expected_revision
    )
    if not identity_verified:
        blockers.append("component_runtime_identity_not_verified")
    runtime_verified = True
    for field in ("python_version", "cuda_version", "driver_version"):
        expected_value = _string(expected_runtime.get(field))
        if expected_value and _string(identity.get(field)) != expected_value:
            runtime_verified = False
            blockers.append(f"component_runtime_expectation_mismatch:{field}")
    expected_gpu_uuid = _string(expected_runtime.get("gpu_uuid"))
    observed_gpu_uuid = _string(_mapping(identity.get("gpu_identity")).get("uuid"))
    if expected_gpu_uuid and observed_gpu_uuid != expected_gpu_uuid:
        runtime_verified = False
        blockers.append("component_runtime_expectation_mismatch:gpu_uuid")
    config_sha = _string(
        payload.get("configuration_sha256")
        or payload.get("sensor_configuration_sha256")
        or payload.get("physics_configuration_sha256")
    )
    if config_sha != expected_configuration_sha256:
        blockers.append("worker_configuration_digest_mismatch")
    check_by_name = {check["name"]: check for check in checks}
    for name in required_checks:
        if _string(_mapping(check_by_name.get(name)).get("status")).lower() != "passed":
            blockers.append(f"required_check_not_passed:{name}")
    output_by_kind = {output["kind"]: output for output in outputs if output["exists_nonempty"]}
    output_kinds = set(output_by_kind)
    for kind in required_output_kinds:
        if kind not in output_kinds:
            blockers.append(f"required_output_missing:{kind}")
            continue
        metadata = _mapping(output_by_kind[kind].get("metadata"))
        for field in required_output_metadata.get(kind, ()):
            if metadata.get(field) is None:
                blockers.append(f"required_output_metadata_missing:{kind}:{field}")
    if execution.get("timed_out"):
        blockers.append("worker_timed_out")
    if execution.get("launch_error"):
        blockers.append("worker_launch_error")
    if execution.get("exit_code") != 0:
        blockers.append("worker_nonzero_exit")
    metrics = _mapping(payload.get("metrics") or payload.get("runtime_metrics"))
    failure_classes_detected = set(_string_list(payload.get("failure_classes_detected")))
    failure_classes_detected.update(
        CHECK_FAILURE_CLASSES.get(check["name"], check["name"])
        for check in checks
        if check["status"] != "passed"
    )
    return {
        "status": "passed" if not blockers else "blocked",
        "identity": identity,
        "identity_verified": identity_verified,
        "runtime_expectations_verified": runtime_verified,
        "configuration_sha256": config_sha or None,
        "checks": checks,
        "outputs": outputs,
        "metrics": metrics,
        "failure_classes_checked": _string_list(payload.get("failure_classes_checked")),
        "failure_classes_detected": sorted(failure_classes_detected),
        "required_sensor_metadata_preserved": (
            bool(payload.get("required_sensor_metadata_preserved"))
            if component == "ovrtx"
            else None
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "raw_report_path": str(raw_report_path) if raw_report_path.is_file() else None,
        "raw_report_sha256": sha256_file(raw_report_path) if raw_report_path.is_file() else None,
        "execution": dict(execution),
    }


def _run_external_library_preflight(
    *,
    component: str,
    capture_root: str | Path,
    input_usd: str | Path | None,
    worker_command: str | Sequence[str] | None,
    component_version: str,
    source_revision: str,
    license_id: str,
    license_compatible: bool,
    configuration: Mapping[str, Any],
    required_output_kinds: Sequence[str],
    allow_external_preflight: bool,
    timeout_seconds: int,
    resource_context_path: str | Path | None,
    resource_closeout_path: str | Path | None,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    if component not in {"ovrtx", "ovphysx"}:
        raise ValueError(f"unsupported Omniverse preflight component: {component}")
    context = resolve_local_capture_context(capture_root)
    output_dir = context.pipeline_root / DIRECTORIES[component]
    ensure_dir(output_dir)
    usd_path = Path(
        input_usd or context.pipeline_root / "simready" / "isaac_sim" / "site_scene.usda"
    ).resolve()
    operator_config = dict(configuration)
    usd_features = inspect_usd_features(usd_path) if usd_path.is_file() else {}
    required_checks = required_checks_for(
        component,
        configuration=operator_config,
        usd_features=usd_features,
        required_output_kinds=required_output_kinds,
    )
    # Persist the derived contract beside the operator configuration so an
    # isolated worker can enforce checks selected from composed/binary USD.
    # This is especially important for ParticleField scenes, which cannot be
    # identified reliably by grepping a USDC payload in the worker.
    config = {
        **operator_config,
        "_blueprint_required_checks": list(required_checks),
        "_blueprint_usd_features": usd_features,
    }
    required_output_metadata = {
        str(kind): _string_list(fields)
        for kind, fields in _mapping(config.get("required_output_metadata")).items()
    }
    config_sha = canonical_sha256(config)
    config_path = output_dir / f"{component}_configuration.json"
    write_json(config_path, config)
    request_path = output_dir / f"{component}_request.json"
    result_path = output_dir / f"{component}_result.json"
    receipt_path = output_dir / f"{component}_runtime_receipt.json"
    claim_path = output_dir / f"{component}_claim_boundary.json"
    blockers: list[str] = []
    if not usd_path.is_file():
        blockers.append("input_usd_missing")
    if usd_path.is_file() and not _path_within(usd_path, context.pipeline_root):
        blockers.append("input_must_be_privacy_safe_pipeline_derived_usd")
    if not component_version or not source_revision:
        blockers.append("component_version_or_revision_not_pinned")
    if not license_id or not license_compatible:
        blockers.append("license_not_verified_compatible")
    blockers.extend(validate_runtime_expectations(component, operator_config))
    resource_context, resource_blockers = load_resource_context(resource_context_path)
    blockers.extend(resource_blockers)
    resource_closeout, closeout_blockers = load_resource_closeout(
        resource_context, resource_closeout_path
    )
    blockers.extend(closeout_blockers)
    env_source = os.environ if env is None else env
    gate = bool(allow_external_preflight and parse_bool(env_source.get(ENABLE_ENV), default=False))
    if not gate:
        blockers.append(f"external_preflight_requires_flag_and_{ENABLE_ENV}=true")
    if worker_command is None:
        blockers.append("external_worker_command_not_configured")
    command_identity = executable_identity(worker_command, env=env) if worker_command else {}
    request = {
        "schema_version": REQUEST_SCHEMAS[component],
        "generated_at": utc_now_iso(),
        "component": component,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "input": {
            "path": str(usd_path),
            "sha256": sha256_file(usd_path) if usd_path.is_file() else None,
            "privacy_safe_pipeline_derived": bool(
                usd_path.is_file() and _path_within(usd_path, context.pipeline_root)
            ),
        },
        "component_identity": {
            "version": component_version,
            "source_revision": source_revision,
            "source_url": SOURCE_URLS[component],
            "license_id": license_id,
            "license_compatible": license_compatible,
            "prerelease": True,
            "executable_identity": command_identity,
        },
        "configuration_path": str(config_path),
        "configuration_sha256": config_sha,
        "usd_feature_preflight": usd_features,
        "required_checks": list(required_checks),
        "required_output_kinds": list(required_output_kinds),
        "required_output_metadata": required_output_metadata,
        "execution_policy": {
            "explicit_flag": allow_external_preflight,
            "environment_gate": ENABLE_ENV,
            "gate_satisfied": gate,
            "network_policy": "disabled",
            "isolated_external_worker_required": True,
            "cold_and_warm_runs_required": True,
            "timeout_seconds_per_run": timeout_seconds,
            "paid_resource_allocation_performed_by_this_command": False,
        },
        "resource_context": resource_context,
        "blockers": blockers,
        "claim_boundary_path": claim_path.name,
    }
    request["request_fingerprint"] = canonical_sha256(request)
    write_json(request_path, request)
    boundary = {
        "schema_version": CLAIM_SCHEMAS[component],
        "generated_at": utc_now_iso(),
        **_claim_boundary(component),
    }
    write_json(claim_path, boundary)

    normalized_runs: list[dict[str, Any]] = []
    if not blockers and worker_command is not None:
        for mode in ("cold", "warm"):
            raw_path = output_dir / f"{component}_{mode}_raw_report.json"
            execution = run_json_worker(
                command=worker_command,
                replacements={
                    "input": str(usd_path),
                    "output": str(raw_path),
                    "output_dir": str(output_dir),
                    "config": str(config_path),
                    "mode": mode,
                    "component_version": component_version,
                    "source_revision": source_revision,
                },
                working_directory=output_dir,
                output_directory=output_dir,
                raw_report_path=raw_path,
                timeout_seconds=timeout_seconds,
                network_policy="disabled",
                env=env,
                log_prefix=f"{component}_{mode}",
            )
            normalized_runs.append(
                _normalize_run(
                    component=component,
                    raw_report_path=raw_path,
                    execution=execution,
                    output_directory=output_dir,
                    expected_version=component_version,
                    expected_revision=source_revision,
                    required_checks=required_checks,
                    required_output_kinds=required_output_kinds,
                    required_output_metadata=required_output_metadata,
                    expected_configuration_sha256=config_sha,
                    expected_runtime=_mapping(operator_config.get("runtime_expectations")),
                )
            )
    run_hash_sets = [
        sorted((item["kind"], item["sha256"]) for item in run["outputs"]) for run in normalized_runs
    ]
    repeatable_outputs = bool(len(run_hash_sets) == 2 and run_hash_sets[0] == run_hash_sets[1])
    if normalized_runs and not repeatable_outputs:
        blockers.append("cold_warm_output_digest_mismatch")
    for mode, run in zip(("cold", "warm"), normalized_runs):
        blockers.extend(f"{mode}:{value}" for value in run["blockers"])
    component_ran = bool(normalized_runs)
    status = "passed_advisory" if component_ran and not blockers else "blocked"
    failure_classes = (
        sorted(set().union(*(set(run["failure_classes_checked"]) for run in normalized_runs)))
        if normalized_runs
        else []
    )
    detected_failure_classes = (
        sorted(set().union(*(set(run["failure_classes_detected"]) for run in normalized_runs)))
        if normalized_runs
        else []
    )
    measured_value = bool(failure_classes)
    resource_evidence = resource_stop_evidence(resource_context, resource_closeout)
    stop_evaluation = evaluate_stop_rules(
        component=component,
        require_measured_value=False,
        evidence={
            "component_version_pinned": bool(component_version and source_revision),
            "license_compatible": license_compatible,
            "stable_normalized_receipts": bool(component_ran and repeatable_outputs),
            "privacy_safe_inputs_only": bool(request["input"]["privacy_safe_pipeline_derived"]),
            "dependency_isolated": True,
            "input_output_digests_preserved": bool(
                request["input"]["sha256"] and repeatable_outputs
            ),
            "proof_boundaries_separated": True,
            "useful_failure_class_or_cost_gain": measured_value,
            **resource_evidence,
        },
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMAS[component],
        "generated_at": utc_now_iso(),
        "component": component,
        "component_ran": component_ran,
        "cold_run": normalized_runs[0] if len(normalized_runs) > 0 else None,
        "warm_run": normalized_runs[1] if len(normalized_runs) > 1 else None,
        "repeatable_output_digests": repeatable_outputs,
        "input_sha256": request["input"]["sha256"],
        "configuration_sha256": config_sha,
        "secret_values_in_artifact": False,
        "paid_resource_allocation_performed": False,
        "resource_context": resource_context,
        "resource_closeout": resource_closeout or None,
    }
    write_json(receipt_path, receipt)
    result = {
        "schema_version": RESULT_SCHEMAS[component],
        "generated_at": utc_now_iso(),
        "status": status,
        "component": component,
        "component_ran": component_ran,
        "request_path": request_path.name,
        "runtime_receipt_path": receipt_path.name,
        "claim_boundary_path": claim_path.name,
        "input_sha256": request["input"]["sha256"],
        "configuration_sha256": config_sha,
        "repeatable_output_digests": repeatable_outputs,
        "cold_run_status": normalized_runs[0]["status"] if len(normalized_runs) > 0 else None,
        "warm_run_status": normalized_runs[1]["status"] if len(normalized_runs) > 1 else None,
        "failure_classes_checked": failure_classes,
        "failure_classes_detected": detected_failure_classes,
        "required_sensor_metadata_preserved": (
            all(run["required_sensor_metadata_preserved"] is True for run in normalized_runs)
            if component == "ovrtx" and normalized_runs
            else None
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "stop_rule_evaluation": stop_evaluation,
        "claim_boundary": _claim_boundary(component),
    }
    result["result_fingerprint"] = canonical_sha256(result)
    write_json(result_path, result)
    return {
        "schema_version": f"{component}_preflight_run.v1",
        "status": status,
        "component_ran": component_ran,
        "request_path": str(request_path),
        "result_path": str(result_path),
        "runtime_receipt_path": str(receipt_path),
        "claim_boundary_path": str(claim_path),
        "claim_boundary": _claim_boundary(component),
    }


def run_ovrtx_preflight(
    *,
    capture_root: str | Path,
    worker_command: str | Sequence[str] | None,
    component_version: str,
    source_revision: str,
    license_id: str,
    license_compatible: bool,
    sensor_configuration: Mapping[str, Any],
    input_usd: str | Path | None = None,
    required_modalities: Sequence[str] = DEFAULT_SENSOR_MODALITIES,
    allow_external_preflight: bool = False,
    timeout_seconds: int = 300,
    resource_context_path: str | Path | None = None,
    resource_closeout_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return _run_external_library_preflight(
        component="ovrtx",
        capture_root=capture_root,
        input_usd=input_usd,
        worker_command=worker_command,
        component_version=component_version,
        source_revision=source_revision,
        license_id=license_id,
        license_compatible=license_compatible,
        configuration=sensor_configuration,
        required_output_kinds=required_modalities,
        allow_external_preflight=allow_external_preflight,
        timeout_seconds=timeout_seconds,
        resource_context_path=resource_context_path,
        resource_closeout_path=resource_closeout_path,
        env=env,
    )


def run_ovphysx_preflight(
    *,
    capture_root: str | Path,
    worker_command: str | Sequence[str] | None,
    component_version: str,
    source_revision: str,
    license_id: str,
    license_compatible: bool,
    physics_configuration: Mapping[str, Any],
    input_usd: str | Path | None = None,
    allow_external_preflight: bool = False,
    timeout_seconds: int = 300,
    resource_context_path: str | Path | None = None,
    resource_closeout_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return _run_external_library_preflight(
        component="ovphysx",
        capture_root=capture_root,
        input_usd=input_usd,
        worker_command=worker_command,
        component_version=component_version,
        source_revision=source_revision,
        license_id=license_id,
        license_compatible=license_compatible,
        configuration=physics_configuration,
        required_output_kinds=("state_snapshots",),
        allow_external_preflight=allow_external_preflight,
        timeout_seconds=timeout_seconds,
        resource_context_path=resource_context_path,
        resource_closeout_path=resource_closeout_path,
        env=env,
    )


def _load_mapping(path: str | Path) -> dict[str, Any]:
    loaded = read_json_any(Path(path))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(loaded)


def build_omniverse_preflight_benchmark(
    *,
    output_path: str | Path,
    ovrtx_result_path: str | Path,
    ovrtx_receipt_path: str | Path,
    ovphysx_result_path: str | Path,
    ovphysx_receipt_path: str | Path,
    isaac_baseline_path: str | Path,
    maximum_runtime_ratio: float = 0.8,
) -> dict[str, Any]:
    if not 0.0 < maximum_runtime_ratio < 1.0:
        raise ValueError("maximum_runtime_ratio must be between zero and one")
    ovrtx_result = _load_mapping(ovrtx_result_path)
    ovrtx_receipt = _load_mapping(ovrtx_receipt_path)
    ovphysx_result = _load_mapping(ovphysx_result_path)
    ovphysx_receipt = _load_mapping(ovphysx_receipt_path)
    isaac = _load_mapping(isaac_baseline_path)
    blockers: list[str] = []
    if isaac.get("accepted_fixture") is not True or isaac.get("isaac_execution_proven") is not True:
        blockers.append("accepted_isaac_execution_baseline_required")
    scene_hashes = {
        _string(ovrtx_result.get("input_sha256")),
        _string(ovphysx_result.get("input_sha256")),
        _string(isaac.get("input_sha256")),
    }
    scene_hashes.discard("")
    if len(scene_hashes) != 1:
        blockers.append("benchmark_inputs_do_not_share_one_scene_digest")

    def duration(receipt: Mapping[str, Any], mode: str) -> float | None:
        run = _mapping(receipt.get(f"{mode}_run"))
        execution = _mapping(run.get("execution"))
        value = execution.get("duration_seconds")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def isaac_duration(mode: str) -> float | None:
        value = _mapping(isaac.get("runtime")).get(f"{mode}_start_seconds")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    decisions: dict[str, Any] = {}
    for component, result, receipt in (
        ("ovrtx", ovrtx_result, ovrtx_receipt),
        ("ovphysx", ovphysx_result, ovphysx_receipt),
    ):
        cold = duration(receipt, "cold")
        warm = duration(receipt, "warm")
        baseline_cold = isaac_duration("cold")
        baseline_warm = isaac_duration("warm")
        ratios = {
            "cold": cold / baseline_cold if cold is not None and baseline_cold else None,
            "warm": warm / baseline_warm if warm is not None and baseline_warm else None,
        }
        runtime_gain = all(
            value is not None and value <= maximum_runtime_ratio for value in ratios.values()
        )
        failure_classes = set(_string_list(result.get("failure_classes_checked")))
        baseline_classes = set(_string_list(isaac.get("failure_classes_checked")))
        useful_coverage = bool(failure_classes and failure_classes.intersection(baseline_classes))
        metadata_ok = (
            result.get("required_sensor_metadata_preserved") is True
            if component == "ovrtx"
            else True
        )
        ready = bool(
            result.get("status") == "passed_advisory"
            and receipt.get("repeatable_output_digests") is True
            and runtime_gain
            and useful_coverage
            and metadata_ok
            and not blockers
        )
        decisions[component] = {
            "decision": "candidate_retained" if ready else "reject_or_keep_experimental",
            "cold_start_seconds": cold,
            "warm_start_seconds": warm,
            "isaac_cold_start_seconds": baseline_cold,
            "isaac_warm_start_seconds": baseline_warm,
            "runtime_ratios": ratios,
            "maximum_runtime_ratio": maximum_runtime_ratio,
            "runtime_gain_gate_passed": runtime_gain,
            "useful_failure_coverage_gate_passed": useful_coverage,
            "required_sensor_metadata_gate_passed": metadata_ok,
            "repeatability_gate_passed": receipt.get("repeatable_output_digests") is True,
            "failure_classes_checked": sorted(failure_classes),
        }
    payload = {
        "schema_version": "omniverse_preflight_benchmark.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "input_sha256": next(iter(scene_hashes)) if len(scene_hashes) == 1 else None,
        "accepted_isaac_baseline": not blockers,
        "decisions": decisions,
        "blockers": blockers,
        "paid_resource_allocation_performed": False,
        "provider_inventory_checked": False,
        "claim_boundary": {
            **COMMON_CLAIM_BOUNDARY,
            "benchmark_is_runtime_and_failure_coverage_comparison_only": True,
            "component_retention_is_production_qualification": False,
        },
    }
    payload["benchmark_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def _manifest_relative(path: Any, manifest_path: Path) -> Path:
    candidate = Path(_string(path))
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (manifest_path.parent / candidate).resolve()
    )


def _run_resource_metrics(receipt: Mapping[str, Any], mode: str) -> dict[str, Any]:
    run = _mapping(receipt.get(f"{mode}_run"))
    execution = _mapping(run.get("execution"))
    usage = _mapping(execution.get("resource_usage"))
    metrics = _mapping(run.get("metrics"))
    return {
        "duration_seconds": execution.get("duration_seconds"),
        "child_maximum_resident_set_size_platform_units": usage.get(
            "maximum_resident_set_size_platform_units"
        ),
        "gpu_memory_baseline_bytes": metrics.get("gpu_memory_baseline_bytes"),
        "gpu_memory_peak_observed_bytes": metrics.get("gpu_memory_peak_observed_bytes"),
    }


def build_omniverse_preflight_benchmark_suite(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    maximum_runtime_ratio: float = 0.8,
) -> dict[str, Any]:
    """Require valid and negative same-scene fixtures before retaining a sidecar."""

    if not 0.0 < maximum_runtime_ratio < 1.0:
        raise ValueError("maximum_runtime_ratio must be between zero and one")
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_mapping(manifest_file)
    blockers: list[str] = []
    if manifest.get("schema_version") != "omniverse_preflight_benchmark_suite_manifest.v1":
        blockers.append("omniverse_benchmark_suite_manifest_schema_invalid")
    if manifest.get("frozen") is not True:
        blockers.append("omniverse_benchmark_suite_manifest_not_frozen")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raw_cases = []
        blockers.append("omniverse_benchmark_suite_cases_missing")
    case_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    valid_count = 0
    negative_count = 0
    component_rows: dict[str, list[dict[str, Any]]] = {"ovrtx": [], "ovphysx": []}
    for index, value in enumerate(raw_cases):
        case = _mapping(value)
        fixture_id = _string(case.get("fixture_id"))
        kind = _string(case.get("kind"))
        if not fixture_id or fixture_id in seen:
            blockers.append(f"omniverse_benchmark_fixture_id_missing_or_duplicate:{index}")
        seen.add(fixture_id)
        if kind == "valid":
            valid_count += 1
        elif kind == "negative":
            negative_count += 1
        else:
            blockers.append(f"omniverse_benchmark_fixture_kind_invalid:{fixture_id or index}")
        isaac_path = _manifest_relative(case.get("isaac_baseline_path"), manifest_file)
        isaac = _load_mapping(isaac_path) if isaac_path.is_file() else {}
        if (
            isaac.get("accepted_fixture") is not True
            or isaac.get("isaac_execution_proven") is not True
        ):
            blockers.append(f"omniverse_benchmark_isaac_baseline_unaccepted:{fixture_id or index}")
        expected_by_component = _mapping(case.get("expected_failure_classes"))
        normalized_components: dict[str, Any] = {}
        for component in ("ovrtx", "ovphysx"):
            result_path = _manifest_relative(case.get(f"{component}_result_path"), manifest_file)
            receipt_path = _manifest_relative(case.get(f"{component}_receipt_path"), manifest_file)
            result = _load_mapping(result_path) if result_path.is_file() else {}
            receipt = _load_mapping(receipt_path) if receipt_path.is_file() else {}
            scene_hash_values = [
                _string(result.get("input_sha256")),
                _string(receipt.get("input_sha256")),
                _string(isaac.get("input_sha256")),
            ]
            same_scene = bool(all(scene_hash_values) and len(set(scene_hash_values)) == 1)
            if not same_scene:
                blockers.append(
                    f"omniverse_benchmark_fixture_scene_digest_mismatch:{fixture_id or index}:{component}"
                )
            cold = _run_resource_metrics(receipt, "cold")
            warm = _run_resource_metrics(receipt, "warm")
            isaac_runtime = _mapping(isaac.get("runtime"))
            try:
                cold_ratio = float(cold["duration_seconds"]) / float(
                    isaac_runtime["cold_start_seconds"]
                )
                warm_ratio = float(warm["duration_seconds"]) / float(
                    isaac_runtime["warm_start_seconds"]
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                cold_ratio = warm_ratio = None
            runtime_gain = bool(
                cold_ratio is not None
                and warm_ratio is not None
                and cold_ratio <= maximum_runtime_ratio
                and warm_ratio <= maximum_runtime_ratio
            )
            cpu_memory_recorded = bool(
                cold["child_maximum_resident_set_size_platform_units"] is not None
                and warm["child_maximum_resident_set_size_platform_units"] is not None
            )
            gpu_memory_recorded = bool(
                component != "ovrtx"
                or (
                    cold["gpu_memory_peak_observed_bytes"] is not None
                    and warm["gpu_memory_peak_observed_bytes"] is not None
                )
            )
            expected_failure = _string(expected_by_component.get(component))
            detected = set(_string_list(result.get("failure_classes_detected")))
            isaac_detected = set(_string_list(isaac.get("failure_classes_detected")))
            if kind == "valid":
                case_passed = bool(
                    result.get("status") == "passed_advisory"
                    and receipt.get("repeatable_output_digests") is True
                    and runtime_gain
                    and cpu_memory_recorded
                    and gpu_memory_recorded
                    and (
                        component != "ovrtx"
                        or result.get("required_sensor_metadata_preserved") is True
                    )
                    and same_scene
                )
            else:
                case_passed = bool(
                    expected_failure
                    and expected_failure in detected
                    and expected_failure in isaac_detected
                    and same_scene
                )
            row = {
                "fixture_id": fixture_id or f"fixture_{index}",
                "kind": kind,
                "component": component,
                "same_scene_digest": same_scene,
                "expected_failure_class": expected_failure or None,
                "detected_failure_classes": sorted(detected),
                "isaac_detected_failure_classes": sorted(isaac_detected),
                "cold_runtime_ratio": cold_ratio,
                "warm_runtime_ratio": warm_ratio,
                "runtime_gain_gate_passed": runtime_gain,
                "cpu_memory_recorded": cpu_memory_recorded,
                "gpu_memory_recorded": gpu_memory_recorded,
                "cold_resource_metrics": cold,
                "warm_resource_metrics": warm,
                "case_gate_passed": case_passed,
            }
            component_rows[component].append(row)
            normalized_components[component] = row
        case_rows.append(
            {
                "fixture_id": fixture_id or f"fixture_{index}",
                "kind": kind,
                "isaac_baseline_path": str(isaac_path),
                "components": normalized_components,
            }
        )
    if valid_count == 0 or negative_count == 0:
        blockers.append("omniverse_benchmark_requires_valid_and_negative_fixtures")
    decisions: dict[str, Any] = {}
    for component, rows in component_rows.items():
        retained = bool(
            valid_count > 0
            and negative_count > 0
            and rows
            and all(row["case_gate_passed"] for row in rows)
            and not blockers
        )
        decisions[component] = {
            "decision": "candidate_retained" if retained else "reject_or_keep_experimental",
            "valid_fixture_count": sum(row["kind"] == "valid" for row in rows),
            "negative_fixture_count": sum(row["kind"] == "negative" for row in rows),
            "all_fixture_gates_passed": bool(rows and all(row["case_gate_passed"] for row in rows)),
        }
    payload = {
        "schema_version": "omniverse_preflight_benchmark_suite.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file) if manifest_file.is_file() else None,
        "fixture_count": len(case_rows),
        "valid_fixture_count": valid_count,
        "negative_fixture_count": negative_count,
        "cases": case_rows,
        "decisions": decisions,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            **COMMON_CLAIM_BOUNDARY,
            "candidate_retention_is_production_qualification": False,
            "same_scene_fixture_comparison_only": True,
        },
    }
    payload["benchmark_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def _read_config(path: str) -> dict[str, Any]:
    return _load_mapping(path)


def _common_parser(component: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run gated {component} external preflight")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--input-usd", default=None)
    parser.add_argument("--worker-command", default=None)
    parser.add_argument("--component-version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-compatible", action="store_true")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--allow-external-preflight", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resource-context", default=None)
    parser.add_argument("--resource-closeout", default=None)
    return parser


def ovrtx_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("ovrtx")
    parser.add_argument("--required-modality", action="append", default=[])
    args = parser.parse_args(argv)
    result = run_ovrtx_preflight(
        capture_root=args.capture_root,
        input_usd=args.input_usd,
        worker_command=args.worker_command,
        component_version=args.component_version,
        source_revision=args.source_revision,
        license_id=args.license_id,
        license_compatible=args.license_compatible,
        sensor_configuration=_read_config(args.configuration),
        required_modalities=args.required_modality or DEFAULT_SENSOR_MODALITIES,
        allow_external_preflight=args.allow_external_preflight,
        timeout_seconds=args.timeout_seconds,
        resource_context_path=args.resource_context,
        resource_closeout_path=args.resource_closeout,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed_advisory" else 2


def ovphysx_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("ovphysx")
    args = parser.parse_args(argv)
    result = run_ovphysx_preflight(
        capture_root=args.capture_root,
        input_usd=args.input_usd,
        worker_command=args.worker_command,
        component_version=args.component_version,
        source_revision=args.source_revision,
        license_id=args.license_id,
        license_compatible=args.license_compatible,
        physics_configuration=_read_config(args.configuration),
        allow_external_preflight=args.allow_external_preflight,
        timeout_seconds=args.timeout_seconds,
        resource_context_path=args.resource_context,
        resource_closeout_path=args.resource_closeout,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed_advisory" else 2


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ovrtx/ovphysx preflights to Isaac")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ovrtx-result", required=True)
    parser.add_argument("--ovrtx-receipt", required=True)
    parser.add_argument("--ovphysx-result", required=True)
    parser.add_argument("--ovphysx-receipt", required=True)
    parser.add_argument("--isaac-baseline", required=True)
    parser.add_argument("--maximum-runtime-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)
    result = build_omniverse_preflight_benchmark(
        output_path=args.output,
        ovrtx_result_path=args.ovrtx_result,
        ovrtx_receipt_path=args.ovrtx_receipt,
        ovphysx_result_path=args.ovphysx_result,
        ovphysx_receipt_path=args.ovphysx_receipt,
        isaac_baseline_path=args.isaac_baseline,
        maximum_runtime_ratio=args.maximum_runtime_ratio,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "completed" else 2


def benchmark_suite_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ovrtx/ovphysx and Isaac over valid and negative fixtures"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-runtime-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)
    result = build_omniverse_preflight_benchmark_suite(
        manifest_path=args.manifest,
        output_path=args.output,
        maximum_runtime_ratio=args.maximum_runtime_ratio,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "completed" else 2
