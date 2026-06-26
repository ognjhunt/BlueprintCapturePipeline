"""Shared proof-boundary helpers for policy endpoint setup/eval artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


POLICY_ENDPOINT_BOUNDARY_SCHEMA_VERSION = "policy_endpoint_boundary_manifest.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "y", "passed", "success"}


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return _mapping(payload)


def _resolve_trace_path(
    *,
    policy_execution_manifest: Mapping[str, Any],
    policy_execution_manifest_path: Path | None,
) -> Path | None:
    trace_ref = _string(policy_execution_manifest.get("policy_execution_trace_path"))
    if not trace_ref:
        return None
    trace_path = Path(trace_ref).expanduser()
    if trace_path.is_absolute():
        return trace_path
    if policy_execution_manifest_path is None:
        return None
    return policy_execution_manifest_path.parent / trace_path


def _endpoint_rows(endpoint_discovery: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    discovery = _mapping(endpoint_discovery)
    rows = discovery.get("endpoint_candidates") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def build_policy_endpoint_boundary_manifest(
    *,
    generated_at: str,
    endpoint_discovery: Mapping[str, Any] | None = None,
    selected_runtime: Mapping[str, Any] | None = None,
    endpoint_policy_used: bool = False,
    fixture_policy_used: bool = False,
    endpoint_invocation_count: int = 0,
    endpoint_valid_action_count: int = 0,
    rejected_policy_action_count: int = 0,
    endpoint_setup_configured: bool = False,
    policy_execution_manifest: Mapping[str, Any] | None = None,
    policy_execution_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the explicit endpoint boundary manifest.

    Endpoint setup, endpoint calls, fixture fallback, and provider-output replay
    are integration facts. They do not prove robot policy execution. This helper
    only upgrades `robot_policy_execution_proven` when the separate
    `policy_execution_manifest.json` proves a non-default, gated real trace.
    """

    manifest_path = (
        Path(policy_execution_manifest_path).expanduser()
        if policy_execution_manifest_path
        else None
    )
    loaded_policy_manifest = (
        _mapping(policy_execution_manifest) or _read_optional_json(manifest_path)
    )
    trace_path = _resolve_trace_path(
        policy_execution_manifest=loaded_policy_manifest,
        policy_execution_manifest_path=manifest_path,
    )
    trace_exists = bool(trace_path and trace_path.is_file())
    policy_manifest_present = bool(loaded_policy_manifest)
    allow_policy_execution_flag = _bool(
        loaded_policy_manifest.get("allow_policy_execution_flag")
    )
    env_allows_policy_execution = _bool(
        loaded_policy_manifest.get("env_BLUEPRINT_ALLOW_POLICY_EXECUTION")
    )
    manifest_claims_robot_policy_execution = _bool(
        loaded_policy_manifest.get("robot_policy_execution_proven")
    )
    robot_team_policy_execution_proven = _bool(
        loaded_policy_manifest.get("robot_team_policy_execution_proven")
    )
    default_test_policy_execution_proven = _bool(
        loaded_policy_manifest.get("default_test_policy_execution_proven")
    )
    real_trace_attempt_count = _int(loaded_policy_manifest.get("attempt_count"))
    gated_real_trace_exists = bool(
        policy_manifest_present
        and manifest_claims_robot_policy_execution
        and robot_team_policy_execution_proven
        and not default_test_policy_execution_proven
        and allow_policy_execution_flag
        and env_allows_policy_execution
        and real_trace_attempt_count > 0
        and trace_exists
    )

    endpoint_rows = _endpoint_rows(endpoint_discovery)
    selected = _mapping(selected_runtime)
    endpoint_url_configured = bool(
        selected.get("endpoint_url")
        or any(row.get("endpoint_url_configured") for row in endpoint_rows)
    )
    endpoint_ready = bool(
        endpoint_policy_used
        or selected.get("ready_for_endpoint_call")
        or any(row.get("ready_for_endpoint_call") for row in endpoint_rows)
    )
    discovery_blockers = [
        str(blocker)
        for blocker in _mapping(endpoint_discovery).get("blockers", []) or []
        if str(blocker)
    ]
    missing_auth = bool(
        endpoint_url_configured
        and not endpoint_ready
        and (
            "blocked_missing_policy_auth_token_file" in discovery_blockers
            or any(
                row.get("endpoint_url_configured") and not row.get("auth_token_file_exists")
                for row in endpoint_rows
            )
        )
    )
    endpoint_integration_configured = bool(
        endpoint_setup_configured
        or endpoint_url_configured
        or endpoint_ready
        or endpoint_policy_used
    )
    endpoint_integration_skipped = bool(
        not endpoint_ready and (fixture_policy_used or not endpoint_integration_configured)
    )
    if endpoint_ready or endpoint_policy_used:
        endpoint_integration_status = "configured_ready_for_endpoint_calls"
    elif missing_auth:
        endpoint_integration_status = "configured_missing_credentials"
    elif endpoint_setup_configured:
        endpoint_integration_status = "setup_artifacts_configured"
    elif endpoint_integration_skipped:
        endpoint_integration_status = "skipped_fixture_or_no_endpoint"
    else:
        endpoint_integration_status = "not_configured"

    real_trace_gate_blockers: list[str] = []
    if not policy_manifest_present:
        real_trace_gate_blockers.append("missing_policy_execution_manifest")
    if policy_manifest_present and not allow_policy_execution_flag:
        real_trace_gate_blockers.append("missing_allow_policy_execution_flag")
    if policy_manifest_present and not env_allows_policy_execution:
        real_trace_gate_blockers.append("missing_BLUEPRINT_ALLOW_POLICY_EXECUTION_gate")
    if policy_manifest_present and not manifest_claims_robot_policy_execution:
        real_trace_gate_blockers.append(
            "policy_execution_manifest_does_not_prove_robot_policy_execution"
        )
    if policy_manifest_present and not robot_team_policy_execution_proven:
        real_trace_gate_blockers.append("robot_team_policy_execution_not_proven")
    if default_test_policy_execution_proven:
        real_trace_gate_blockers.append(
            "default_test_policy_execution_is_not_robot_team_policy_proof"
        )
    if policy_manifest_present and real_trace_attempt_count <= 0:
        real_trace_gate_blockers.append("missing_policy_execution_attempts")
    if policy_manifest_present and not trace_exists:
        real_trace_gate_blockers.append("missing_policy_execution_trace_file")

    blockers = sorted(set(discovery_blockers))
    if missing_auth:
        blockers.append("decision_needed_policy_auth_token_file")
    if fixture_policy_used:
        blockers.append("blocked_fixture_policy_is_not_robot_policy_execution_proof")
    if not gated_real_trace_exists:
        blockers.append("blocked_missing_gated_real_policy_execution_trace")
    blockers = sorted(set(blockers))

    if gated_real_trace_exists:
        status = "completed_robot_policy_execution_trace_proven"
    elif missing_auth:
        status = "decision_needed_missing_policy_credentials"
    elif endpoint_policy_used or endpoint_ready:
        status = "endpoint_integration_configured_not_robot_policy_execution"
    elif fixture_policy_used:
        status = "fixture_boundary_only"
    elif endpoint_setup_configured:
        status = "endpoint_setup_boundary_only"
    else:
        status = "blocked_no_endpoint_or_real_trace"

    return {
        "schema_version": POLICY_ENDPOINT_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "endpoint_integration_status": endpoint_integration_status,
        "endpoint_setup_configured": bool(endpoint_setup_configured),
        "endpoint_integration_configured": endpoint_integration_configured,
        "endpoint_integration_skipped": endpoint_integration_skipped,
        "endpoint_url_configured": endpoint_url_configured,
        "endpoint_ready_for_calls": endpoint_ready,
        "endpoint_policy_used": bool(endpoint_policy_used),
        "fixture_policy_used": bool(fixture_policy_used),
        "endpoint_invocation_count": int(endpoint_invocation_count),
        "endpoint_valid_action_count": int(endpoint_valid_action_count),
        "rejected_policy_action_count": int(rejected_policy_action_count),
        "missing_credentials_decision_needed": missing_auth,
        "robot_policy_execution_proven": gated_real_trace_exists,
        "robot_policy_execution_proof_source": (
            "gated_policy_execution_manifest" if gated_real_trace_exists else None
        ),
        "policy_execution_manifest_path": str(manifest_path) if manifest_path else None,
        "policy_execution_trace_path": str(trace_path) if trace_path else None,
        "real_trace_gate": {
            "policy_execution_manifest_present": policy_manifest_present,
            "allow_policy_execution_flag": allow_policy_execution_flag,
            "env_BLUEPRINT_ALLOW_POLICY_EXECUTION": env_allows_policy_execution,
            "manifest_claims_robot_policy_execution": manifest_claims_robot_policy_execution,
            "robot_team_policy_execution_proven": robot_team_policy_execution_proven,
            "default_test_policy_execution_proven": default_test_policy_execution_proven,
            "attempt_count": real_trace_attempt_count,
            "policy_execution_trace_exists": trace_exists,
            "gated_real_trace_exists": gated_real_trace_exists,
            "blockers": [] if gated_real_trace_exists else real_trace_gate_blockers,
        },
        "claim_boundary": {
            "endpoint_setup_is_not_robot_policy_execution": True,
            "endpoint_configuration_is_not_robot_policy_execution": True,
            "endpoint_invocation_is_not_robot_policy_execution": True,
            "fixture_policy_is_not_robot_policy_execution": True,
            "missing_credentials_do_not_upgrade_proof": True,
            "robot_policy_execution_requires_gated_real_trace": True,
            "robot_policy_execution_proven": gated_real_trace_exists,
            "endpoint_setup_is_not_real_world_success": True,
            "endpoint_setup_is_not_safety_validation": True,
            "endpoint_setup_is_not_deployment_approval": True,
            "real_world_success_proven": False,
            "real_world_outcome_proven": False,
            "safety_validation_proven": False,
            "deployment_approval_proven": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
        "blockers": blockers,
        "required_for_robot_policy_execution_proof": [
            "non_default_policy_execution_manifest_with_robot_team_policy_execution_proven_true",
            "BLUEPRINT_ALLOW_POLICY_EXECUTION_gate_true",
            "allow_policy_execution_flag_true",
            "policy_execution_trace_file_present",
        ],
    }
