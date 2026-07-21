"""Isolated, advisory NVIDIA SimReady Foundation validation adapter.

This module does not import ``simready-validate``, ``usd-core``, or Omniverse
packages. It invokes an explicit command template in a separately managed
environment and normalizes the JSON report into Blueprint-owned artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import (
    PUBLIC_CLAIM_UPGRADE_KEY,
    canonical_sha256,
    executable_identity,
    run_json_worker,
)
from .local_capture import resolve_local_capture_context
from .nvidia_experiment_resource import (
    load_resource_closeout,
    load_resource_context,
    resource_stop_evidence,
)
from .nvidia_siggraph_policy import evaluate_stop_rules


REQUEST_SCHEMA_VERSION = "external_simready_validation_request.v1"
RESULT_SCHEMA_VERSION = "external_simready_validation_result.v1"
CLAIM_SCHEMA_VERSION = "external_simready_validation_claim_boundary.v1"
DEFAULT_SOURCE_URL = "https://github.com/NVIDIA/simready-foundation"

CLAIM_BOUNDARY: dict[str, Any] = {
    "artifact_purpose": "advisory_external_simready_profile_validation",
    "raw_capture_authority_preserved": True,
    "external_validator_pass_is_capture_truth": False,
    "external_validator_pass_is_simulator_load_proof": False,
    "simulator_execution_proven": False,
    "physics_contact_validated": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "deployment_ready": False,
    "real_world_task_success_proven": False,
    PUBLIC_CLAIM_UPGRADE_KEY: False,
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


def _path_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _normalize_severity(value: Any) -> str:
    text = _string(value).lower()
    if text in {"failed", "failure", "fail", "error", "critical", "blocked"}:
        return "error"
    if text in {"warn", "warning", "degraded"}:
        return "warning"
    if text in {"passed", "pass", "ok", "success", "completed"}:
        return "info"
    return text or "unknown"


def _finding_from_mapping(value: Mapping[str, Any]) -> dict[str, Any] | None:
    rule_id = _string(
        value.get("rule_id")
        or value.get("requirement_id")
        or value.get("requirement")
        or value.get("rule")
    )
    message = _string(value.get("message") or value.get("reason") or value.get("description"))
    status = _string(value.get("status") or value.get("severity") or value.get("result"))
    if not rule_id and not message:
        return None
    return {
        "rule_id": rule_id or "unidentified_rule",
        "severity": _normalize_severity(status),
        "status": status or None,
        "object_path": _string(
            value.get("object_path") or value.get("prim_path") or value.get("path")
        )
        or None,
        "message": message or None,
        "suggested_action": _string(
            value.get("suggested_action") or value.get("suggestion") or value.get("remediation")
        )
        or None,
    }


def normalize_findings(payload: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            candidate = _finding_from_mapping(value)
            if candidate is not None:
                fingerprint = canonical_sha256(candidate)
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    findings.append(candidate)
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)
    return findings


def _reported_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    validator = _mapping(payload.get("validator") or payload.get("tool"))
    profile = _mapping(payload.get("profile"))
    return {
        "validator_version": _string(
            payload.get("validator_version")
            or validator.get("version")
            or payload.get("tool_version")
        )
        or None,
        "profile_name": _string(
            payload.get("profile_name") or profile.get("name") or payload.get("requested_profile")
        )
        or None,
        "profile_version": _string(payload.get("profile_version") or profile.get("version"))
        or None,
    }


def _source_manifest_ids(capture_root: Path, explicit: Sequence[str]) -> list[str]:
    ids = list(_string_list(explicit))
    candidate_paths = (
        capture_root / "capture_descriptor.json",
        capture_root / "raw" / "manifest.json",
        capture_root / "pipeline" / "simready" / "simready_scene_manifest.json",
    )
    for path in candidate_paths:
        if not path.is_file():
            continue
        payload = read_json_any(path)
        if not isinstance(payload, Mapping):
            continue
        for key in ("manifest_id", "capture_id", "scene_id", "deterministic_fingerprint"):
            text = _string(payload.get(key))
            if text:
                ids.append(f"{path.name}:{key}:{text}")
    return list(dict.fromkeys(ids))


def run_external_simready_validation(
    *,
    capture_root: str | Path,
    input_usd: str | Path | None = None,
    validator_command: str | Sequence[str] | None = None,
    requested_profile: str = "Prop-Robotics-Neutral",
    profile_version: str = "1.0.0",
    validator_version: str,
    validator_source_revision: str,
    validator_source_url: str = DEFAULT_SOURCE_URL,
    validator_license_id: str = "Apache-2.0",
    license_compatible: bool = False,
    source_manifest_ids: Sequence[str] = (),
    package_id: str | None = None,
    timeout_seconds: int = 180,
    resource_class: str = "cpu",
    resource_context_path: str | Path | None = None,
    resource_closeout_path: str | Path | None = None,
    network_policy: str = "disabled",
    transformations: Sequence[str] = (),
    allow_transformations: bool = False,
    repeat_runs: int = 2,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    simready_dir = context.pipeline_root / "simready"
    ensure_dir(simready_dir)
    usd_path = Path(input_usd or simready_dir / "isaac_sim" / "site_scene.usda").resolve()
    request_path = simready_dir / "external_validation_request.json"
    result_path = simready_dir / "external_validation_result.json"
    claim_path = simready_dir / "external_validation_claim_boundary.json"
    raw_report_paths = [
        simready_dir / f"external_validation_raw_report_{index}.json"
        for index in range(1, repeat_runs + 1)
    ]
    transform_list = _string_list(transformations)
    blockers: list[str] = []
    if not usd_path.is_file():
        blockers.append("input_usd_missing")
    if usd_path.is_file() and not _path_within(usd_path, context.pipeline_root):
        blockers.append("input_must_be_privacy_safe_pipeline_derived_usd")
    if not requested_profile:
        blockers.append("requested_profile_missing")
    if not profile_version:
        blockers.append("profile_version_missing")
    if not validator_version or not validator_source_revision:
        blockers.append("validator_version_or_revision_not_pinned")
    if not validator_license_id or not license_compatible:
        blockers.append("license_not_verified_compatible")
    if repeat_runs < 2:
        blockers.append("simready_repeatability_requires_at_least_two_runs")
    if transform_list and not allow_transformations:
        blockers.append("transformations_requested_without_explicit_authorization")
    if allow_transformations:
        blockers.append("transformations_not_supported_by_advisory_v1_adapter")
    if network_policy != "disabled":
        blockers.append("base_validation_requires_network_disabled_policy")
    baseline_path = simready_dir / "simready_validation.json"
    baseline_payload = read_json_any(baseline_path) if baseline_path.is_file() else {}
    if not baseline_path.is_file() or not isinstance(baseline_payload, Mapping):
        blockers.append("local_simready_validation_baseline_missing")
        baseline_payload = {}
    resource_context, resource_blockers = load_resource_context(resource_context_path)
    blockers.extend(resource_blockers)
    resource_closeout, closeout_blockers = load_resource_closeout(
        resource_context, resource_closeout_path
    )
    blockers.extend(closeout_blockers)
    command_identity = (
        executable_identity(validator_command, env=env)
        if validator_command
        else {
            "requested_executable": None,
            "resolved_executable": None,
            "executable_found": False,
            "executable_sha256": None,
        }
    )
    resolved_executable = _string(command_identity.get("resolved_executable"))
    requested_executable = _string(command_identity.get("requested_executable"))
    core_venv = Path(__file__).resolve().parents[2] / ".venv"
    if requested_executable or resolved_executable:
        try:
            requested_path = Path(requested_executable).expanduser()
            requested_in_core = bool(
                requested_path.is_absolute()
                and requested_path.absolute().is_relative_to(core_venv.absolute())
            )
            resolved_in_core = bool(
                resolved_executable
                and Path(resolved_executable).absolute().is_relative_to(core_venv.absolute())
            )
            if requested_in_core or resolved_in_core:
                blockers.append("validator_must_not_use_core_environment")
        except OSError:
            blockers.append("validator_executable_identity_unresolvable")
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "package_id": package_id or f"{context.scene_id}:{context.capture_id}",
        "input": {
            "path": str(usd_path),
            "sha256": sha256_file(usd_path) if usd_path.is_file() else None,
            "source_manifest_ids": _source_manifest_ids(context.capture_root, source_manifest_ids),
            "privacy_safe_pipeline_derived": bool(
                usd_path.is_file() and _path_within(usd_path, context.pipeline_root)
            ),
        },
        "local_validation_baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path) if baseline_path.is_file() else None,
            "status": baseline_payload.get("overall_status"),
        },
        "requested_profile": requested_profile,
        "requested_profile_version": profile_version,
        "validator": {
            "package_version": validator_version,
            "source_url": validator_source_url,
            "source_revision": validator_source_revision,
            "license_id": validator_license_id,
            "license_compatible": license_compatible,
            "executable_identity": command_identity,
        },
        "transformations": {
            "requested": transform_list,
            "allowed": allow_transformations,
            "enabled": False,
        },
        "execution_policy": {
            "timeout_seconds": timeout_seconds,
            "resource_class": resource_class,
            "network_policy": network_policy,
            "isolated_external_environment_required": True,
            "core_environment_import_allowed": False,
            "advisory_only": True,
            "repeat_runs": repeat_runs,
        },
        "resource_context": resource_context,
        "preflight_blockers": blockers,
        "claim_boundary_path": claim_path.name,
    }
    request["request_fingerprint"] = canonical_sha256(request)
    write_json(request_path, request)
    write_json(
        claim_path,
        {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            **CLAIM_BOUNDARY,
        },
    )

    executions: list[dict[str, Any]] = []
    raw_payloads: list[dict[str, Any]] = []
    findings_by_run: list[list[dict[str, Any]]] = []
    reported_by_run: list[dict[str, Any]] = []
    external_validator_ran = False
    if validator_command and not blockers:
        for index, raw_report_path in enumerate(raw_report_paths, start=1):
            raw_report_path.unlink(missing_ok=True)
            execution = run_json_worker(
                command=validator_command,
                replacements={
                    "input": str(usd_path),
                    "output": str(raw_report_path),
                    "profile": requested_profile,
                    "profile_version": profile_version,
                },
                working_directory=simready_dir,
                output_directory=simready_dir,
                raw_report_path=raw_report_path,
                timeout_seconds=timeout_seconds,
                network_policy=network_policy,
                env=env,
                log_prefix=f"external_simready_validator_{index}",
            )
            executions.append(execution)
            if raw_report_path.is_file():
                loaded = read_json_any(raw_report_path)
                raw_payload = dict(loaded) if isinstance(loaded, Mapping) else {"raw_value": loaded}
            else:
                raw_payload = {}
            raw_payloads.append(raw_payload)
            findings_by_run.append(normalize_findings(raw_payload))
            reported_by_run.append(_reported_identity(raw_payload))
        external_validator_ran = bool(executions) and all(
            execution.get("exit_code") is not None for execution in executions
        )
    elif not validator_command:
        blockers.append("external_validator_command_not_configured")

    findings = findings_by_run[0] if findings_by_run else []
    reported = reported_by_run[0] if reported_by_run else _reported_identity({})
    version_identity_verified = bool(
        external_validator_ran
        and reported_by_run
        and all(
            row["validator_version"] == validator_version
            and row["profile_name"] == requested_profile
            and row["profile_version"] == profile_version
            for row in reported_by_run
        )
    )
    if external_validator_ran and not version_identity_verified:
        blockers.append("actual_validator_or_profile_identity_not_verified")
    failed_findings = [item for item in findings if item["severity"] == "error"]
    execution_failed = any(
        execution.get("timed_out")
        or execution.get("launch_error")
        or execution.get("exit_code") not in {0, None}
        for execution in executions
    )
    stable_normalized_results = bool(
        len(findings_by_run) == repeat_runs
        and len(reported_by_run) == repeat_runs
        and all(rows == findings_by_run[0] for rows in findings_by_run[1:])
        and all(row == reported_by_run[0] for row in reported_by_run[1:])
    )
    if external_validator_ran and not stable_normalized_results:
        blockers.append("simready_normalized_results_not_repeatable")
    if external_validator_ran and version_identity_verified and stable_normalized_results:
        if failed_findings or execution_failed:
            status = "validation_failed"
        else:
            status = "passed_advisory"
    else:
        status = "blocked"
    resource_evidence = resource_stop_evidence(resource_context, resource_closeout)
    stop_evaluation = evaluate_stop_rules(
        component="simready_foundation",
        require_measured_value=False,
        evidence={
            "component_version_pinned": bool(validator_version and validator_source_revision),
            "license_compatible": license_compatible,
            "stable_normalized_receipts": stable_normalized_results,
            "privacy_safe_inputs_only": bool(request["input"]["privacy_safe_pipeline_derived"]),
            "dependency_isolated": True,
            "input_output_digests_preserved": bool(request["input"]["sha256"]),
            "proof_boundaries_separated": True,
            **resource_evidence,
        },
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "advisory_only": True,
        "request_path": request_path.name,
        "request_fingerprint": request["request_fingerprint"],
        "external_validator_ran": external_validator_ran,
        "process_exit_code": executions[0].get("exit_code") if executions else None,
        "execution": executions[0] if executions else None,
        "executions": executions,
        "normalized_findings": findings,
        "finding_count": len(findings),
        "failed_finding_count": len(failed_findings),
        "reported_identity": reported,
        "requested_identity": {
            "validator_version": validator_version,
            "validator_source_revision": validator_source_revision,
            "profile_name": requested_profile,
            "profile_version": profile_version,
        },
        "version_identity_verified": version_identity_verified,
        "repeatability": {
            "required_runs": repeat_runs,
            "completed_runs": len(executions),
            "stable_normalized_results": stable_normalized_results,
            "normalized_finding_fingerprints": [canonical_sha256(rows) for rows in findings_by_run],
        },
        "raw_report_path": (
            str(raw_report_paths[0]) if raw_report_paths and raw_report_paths[0].is_file() else None
        ),
        "raw_report_sha256": (
            sha256_file(raw_report_paths[0])
            if raw_report_paths and raw_report_paths[0].is_file()
            else None
        ),
        "raw_reports": [
            {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for path in raw_report_paths
        ],
        "local_validation_baseline": request["local_validation_baseline"],
        "resource_context": resource_context,
        "resource_closeout": resource_closeout or None,
        "input_sha256_before": request["input"]["sha256"],
        "input_sha256_after": sha256_file(usd_path) if usd_path.is_file() else None,
        "input_modified": bool(
            usd_path.is_file() and request["input"]["sha256"] != sha256_file(usd_path)
        ),
        "transformations_performed": [],
        "blockers": list(dict.fromkeys(blockers)),
        "stop_rule_evaluation": stop_evaluation,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if result["input_modified"]:
        result["status"] = "blocked"
        result["blockers"].append("input_usd_modified_by_advisory_validator")
    result["result_fingerprint"] = canonical_sha256(result)
    write_json(result_path, result)
    return {
        "schema_version": "external_simready_validation_run.v1",
        "status": result["status"],
        "request_path": str(request_path),
        "result_path": str(result_path),
        "claim_boundary_path": str(claim_path),
        "external_validator_ran": external_validator_ran,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--input-usd", default=None)
    parser.add_argument("--validator-command", default=None)
    parser.add_argument("--profile", default="Prop-Robotics-Neutral")
    parser.add_argument("--profile-version", default="1.0.0")
    parser.add_argument("--validator-version", required=True)
    parser.add_argument("--validator-source-revision", required=True)
    parser.add_argument("--validator-source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--validator-license-id", default="Apache-2.0")
    parser.add_argument("--license-compatible", action="store_true")
    parser.add_argument("--source-manifest-id", action="append", default=[])
    parser.add_argument("--package-id", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--resource-class", default="cpu")
    parser.add_argument("--resource-context", default=None)
    parser.add_argument("--resource-closeout", default=None)
    parser.add_argument("--repeat-runs", type=int, default=2)
    args = parser.parse_args(argv)
    result = run_external_simready_validation(
        capture_root=args.capture_root,
        input_usd=args.input_usd,
        validator_command=args.validator_command,
        requested_profile=args.profile,
        profile_version=args.profile_version,
        validator_version=args.validator_version,
        validator_source_revision=args.validator_source_revision,
        validator_source_url=args.validator_source_url,
        validator_license_id=args.validator_license_id,
        license_compatible=args.license_compatible,
        source_manifest_ids=args.source_manifest_id,
        package_id=args.package_id,
        timeout_seconds=args.timeout_seconds,
        resource_class=args.resource_class,
        resource_context_path=args.resource_context,
        resource_closeout_path=args.resource_closeout,
        repeat_runs=args.repeat_runs,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"passed_advisory", "validation_failed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
