"""Replayable join between reconstruction routes and executable runtime state.

This is a support artifact.  It records whether a typed route stage has a
registered tool, a trusted runtime binding, and (for the existing local
control-plane adapters) explicit owner authorization.  It never executes a
method and never upgrades reconstruction evidence or claim ceilings.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..decision_evidence_contracts import canonical_digest
from ..local_reconstruction_adapters import (
    LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
    LOCAL_DECODED_OBSERVATION_ADAPTER,
    LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
)
from .capture_ingress import capture_build_source_binding, validate_capture_build_ingress
from .capture_reconstruction_routing import validate_capture_reconstruction_route


RECONSTRUCTION_EXECUTION_READINESS_SCHEMA_VERSION = (
    "task_evaluation_reconstruction_execution_readiness.v1"
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_PLANE_STAGE_BINDINGS = {
    "compile_frozen_frame_dataset": (
        (
            LOCAL_DECODED_OBSERVATION_ADAPTER,
            "local_decoded_observation_index",
        ),
        (
            LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
            "local_arkit_metric_scaffold",
        ),
    ),
    "compile_arkit_metric_scaffold": (
        (
            LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
            "local_arkit_metric_scaffold",
        ),
    ),
    "import_external_reconstruction": (
        (
            LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
            "local_external_reconstruction_import",
        ),
    ),
}


def bound_tool_ids_for_control_plane_inspection(
    value: Mapping[str, Any],
) -> list[str]:
    """Return only repository-registered local tool bindings named by a plan.

    A binding means that the deterministic local implementation is present. It
    does not mean that the method is authorized; authority is evaluated
    separately from the immutable control-plane authorization receipt.
    """

    inspection = _clone(dict(value))
    plan = inspection.get("reconstruction_plan")
    selected = plan.get("selected_methods") if isinstance(plan, Mapping) else None
    if not isinstance(selected, list):
        raise ReconstructionExecutionReadinessError("control_plane_inspection_invalid")
    planned_references = {
        str(row.get("adapter_reference") or "")
        for row in selected
        if isinstance(row, Mapping) and str(row.get("adapter_reference") or "")
    }
    return sorted(
        {
            stage_id
            for stage_id, bindings in _CONTROL_PLANE_STAGE_BINDINGS.items()
            if any(reference in planned_references for reference, _ in bindings)
        }
    )


class ReconstructionExecutionReadinessError(ValueError):
    """Raised when readiness inputs or a recorded artifact fail closed."""


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReconstructionExecutionReadinessError("readiness_not_json") from exc


def _digest(value: Any) -> bool:
    return _DIGEST_RE.fullmatch(str(value or "")) is not None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _timestamp(value: Any) -> bool:
    text = str(value or "")
    if not text.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validated_tool_registry_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _clone(dict(value))
    tools = manifest.get("tools")
    if (
        manifest.get("schema_version") != "task_evaluation_supervisor_tool_registry.v1"
        or not isinstance(tools, list)
        or not all(isinstance(row, Mapping) for row in tools)
        or not _digest(manifest.get("tool_registry_digest"))
        or manifest.get("tool_registry_digest")
        != canonical_digest(manifest, digest_field="tool_registry_digest")
    ):
        raise ReconstructionExecutionReadinessError("tool_registry_manifest_invalid")
    tool_ids = [str(row.get("tool_id") or "") for row in tools]
    if any(not tool_id for tool_id in tool_ids) or len(tool_ids) != len(set(tool_ids)):
        raise ReconstructionExecutionReadinessError("tool_registry_manifest_invalid")
    return manifest


def _control_plane_binding(
    value: Mapping[str, Any] | None,
    *,
    source_binding: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, set[str]], list[str]]:
    state = {
        "planned_adapters": set(),
        "planned_methods": set(),
        "authorized_adapters": set(),
        "completed_methods": set(),
    }
    if value is None:
        return None, state, []
    inspection = _clone(dict(value))
    source = inspection.get("source_binding")
    plan = inspection.get("reconstruction_plan")
    authorization = inspection.get("execution_authorization")
    execution = inspection.get("execution_result")
    if (
        inspection.get("schema_version") != "reconstruction_control_plane_inspection.v1"
        or not str(inspection.get("plan_id") or "").strip()
        or not isinstance(source, Mapping)
        or not isinstance(plan, Mapping)
        or plan.get("reconstruction_plan_digest")
        != canonical_digest(plan, digest_field="reconstruction_plan_digest")
    ):
        raise ReconstructionExecutionReadinessError("control_plane_inspection_invalid")
    plan_source = plan.get("source_capture")
    if (
        not isinstance(plan_source, Mapping)
        or plan_source.get("capture_digest") != source.get("capture_digest")
        or plan_source.get("intake_id") != source.get("intake_id")
        or not _digest(source.get("capture_digest"))
        or not _digest(source.get("context_digest"))
        or any(
            source.get(field) is not None and not _digest(source.get(field))
            for field in ("envelope_digest", "qa_report_digest", "object_manifest_digest")
        )
    ):
        raise ReconstructionExecutionReadinessError("control_plane_source_plan_mismatch")

    blockers: list[str] = []
    for field in ("capture_digest", "envelope_digest", "qa_report_digest"):
        capture_value = source_binding.get(field)
        control_value = source.get(field)
        if (
            capture_value is not None
            and control_value is not None
            and capture_value != control_value
        ):
            blockers.append(f"control_plane_{field}_mismatch")
    if source_binding.get("capture_digest") is None:
        blockers.append("capture_source_digest_missing")
    elif source.get("capture_digest") is None:
        blockers.append("control_plane_capture_digest_missing")

    selected = plan.get("selected_methods")
    if not isinstance(selected, list):
        raise ReconstructionExecutionReadinessError("control_plane_inspection_invalid")
    state["planned_adapters"] = {
        str(row.get("adapter_reference") or "")
        for row in selected
        if isinstance(row, Mapping) and str(row.get("adapter_reference") or "")
    }
    state["planned_methods"] = {
        str(row.get("method_id") or "")
        for row in selected
        if isinstance(row, Mapping) and str(row.get("method_id") or "")
    }

    authorization_digest = None
    if authorization is not None:
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("authorization_digest")
            != canonical_digest(authorization, digest_field="authorization_digest")
            or authorization.get("reconstruction_plan_digest")
            != plan.get("reconstruction_plan_digest")
            or authorization.get("context_digest") != source.get("context_digest")
        ):
            raise ReconstructionExecutionReadinessError("control_plane_authorization_invalid")
        authorization_digest = authorization["authorization_digest"]
        state["authorized_adapters"] = set(
            _strings(authorization.get("authorized_adapter_references"))
        )
        if state["authorized_adapters"] - state["planned_adapters"]:
            raise ReconstructionExecutionReadinessError("control_plane_authorization_unplanned")

    execution_digest = None
    if execution is not None:
        if (
            not isinstance(execution, Mapping)
            or execution.get("execution_result_digest")
            != canonical_digest(execution, digest_field="execution_result_digest")
            or execution.get("reconstruction_plan_digest") != plan.get("reconstruction_plan_digest")
            or authorization_digest is None
            or execution.get("authorization_digest") != authorization_digest
            or execution.get("context_digest") != source.get("context_digest")
        ):
            raise ReconstructionExecutionReadinessError("control_plane_execution_invalid")
        execution_digest = execution["execution_result_digest"]
        results = execution.get("results")
        if not isinstance(results, list):
            raise ReconstructionExecutionReadinessError("control_plane_execution_invalid")
        state["completed_methods"] = {
            str(row.get("method_id") or "")
            for row in results
            if isinstance(row, Mapping) and str(row.get("method_id") or "")
        }
        if state["completed_methods"] - state["planned_methods"]:
            raise ReconstructionExecutionReadinessError("control_plane_execution_unplanned_method")

    binding = {
        "plan_id": inspection["plan_id"],
        "state": inspection.get("state"),
        "capture_digest": source.get("capture_digest"),
        "envelope_digest": source.get("envelope_digest"),
        "qa_report_digest": source.get("qa_report_digest"),
        "context_digest": source.get("context_digest"),
        "reconstruction_plan_digest": plan["reconstruction_plan_digest"],
        "authorization_digest": authorization_digest,
        "execution_result_digest": execution_digest,
    }
    return binding, state, sorted(set(blockers))


def build_reconstruction_execution_readiness(
    *,
    capture_build_value: Mapping[str, Any],
    route_value: Mapping[str, Any],
    tool_registry_manifest: Mapping[str, Any],
    bound_tool_ids: Sequence[str],
    source_commit_sha: str,
    recorded_at: str,
    control_plane_inspection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic, non-executing readiness record."""

    capture_build = validate_capture_build_ingress(capture_build_value)
    route = validate_capture_reconstruction_route(route_value)
    registry = _validated_tool_registry_manifest(tool_registry_manifest)
    if route["capture_build_digest"] != capture_build["capture_build_digest"]:
        raise ReconstructionExecutionReadinessError("readiness_capture_route_mismatch")
    if _COMMIT_RE.fullmatch(str(source_commit_sha or "")) is None:
        raise ReconstructionExecutionReadinessError("readiness_source_commit_invalid")
    if not _timestamp(recorded_at):
        raise ReconstructionExecutionReadinessError("readiness_timestamp_invalid")

    registered = {str(row["tool_id"]) for row in registry["tools"]}
    bound = _strings(bound_tool_ids)
    if set(bound) - registered:
        raise ReconstructionExecutionReadinessError("readiness_bound_tool_unregistered")
    source_binding = capture_build_source_binding(capture_build)
    control_binding, control_state, source_blockers = _control_plane_binding(
        control_plane_inspection,
        source_binding=source_binding,
    )
    source_mismatch = any(blocker.endswith("_mismatch") for blocker in source_blockers)

    blockers = list(source_blockers)
    stages: list[dict[str, Any]] = []
    for route_stage in route["stages"]:
        stage_id = str(route_stage["stage_id"])
        implementation_status = str(route_stage["implementation_status"])
        registered_tool = stage_id in registered
        runtime_bound = stage_id in bound
        adapter_binding = _CONTROL_PLANE_STAGE_BINDINGS.get(stage_id)
        authority_status = "not_applicable"
        readiness_status: str
        stage_blockers: list[str] = []
        if implementation_status == "required_deterministic_gate":
            readiness_status = "deterministic_gate_pending"
        elif source_mismatch:
            readiness_status = "source_binding_mismatch"
            stage_blockers.extend(source_blockers)
        elif not registered_tool:
            readiness_status = "typed_tool_not_registered"
            stage_blockers.append(f"typed_tool_not_registered:{stage_id}")
        elif adapter_binding is not None:
            planned_options = [
                (adapter_reference, method_id)
                for adapter_reference, method_id in adapter_binding
                if adapter_reference in control_state["planned_adapters"]
            ]
            authorized_options = [
                (adapter_reference, method_id)
                for adapter_reference, method_id in planned_options
                if adapter_reference in control_state["authorized_adapters"]
            ]
            if control_binding is None:
                authority_status = "control_plane_plan_missing"
                readiness_status = "awaiting_control_plane_plan"
                stage_blockers.append(f"control_plane_plan_missing:{stage_id}")
            elif not planned_options:
                authority_status = "method_not_planned"
                readiness_status = "control_plane_method_not_planned"
                stage_blockers.append(f"control_plane_method_not_planned:{stage_id}")
            elif not authorized_options:
                authority_status = "explicit_authority_missing"
                readiness_status = "awaiting_control_plane_authority"
                stage_blockers.append(f"control_plane_authority_missing:{stage_id}")
            elif any(
                method_id in control_state["completed_methods"]
                for _, method_id in authorized_options
            ):
                authority_status = "authorized"
                readiness_status = "recorded_support_completed"
            elif runtime_bound:
                authority_status = "authorized"
                readiness_status = "ready_for_bounded_tool_call"
            else:
                authority_status = "authorized"
                readiness_status = "runtime_binding_missing"
                stage_blockers.append(f"runtime_binding_missing:{stage_id}")
        elif runtime_bound:
            readiness_status = "ready_for_bounded_tool_call"
        else:
            readiness_status = "runtime_binding_missing"
            stage_blockers.append(f"runtime_binding_missing:{stage_id}")
        blockers.extend(stage_blockers)
        stages.append(
            {
                "ordinal": route_stage["ordinal"],
                "stage_id": stage_id,
                "method_kind": route_stage["method_kind"],
                "implementation_status": implementation_status,
                "typed_tool_registered": registered_tool,
                "runtime_bound": runtime_bound,
                "control_plane_authority_status": authority_status,
                "readiness_status": readiness_status,
                "blockers": sorted(set(stage_blockers)),
            }
        )

    blockers = sorted(set(blockers))
    if route["status"] != "route_proposed":
        status = "route_unresolved"
        blockers = sorted(set(blockers + list(route["blockers"])))
    elif source_mismatch:
        status = "source_binding_mismatch"
    elif blockers:
        status = "not_ready"
    else:
        status = "ready_for_bounded_execution"
    next_actions = sorted(
        {
            "request_validated_capture_profile"
            if route["status"] != "route_proposed"
            else "bind_capture_to_reconstruction_control_plane"
            if any("control_plane_plan_missing" in blocker for blocker in blockers)
            else "request_explicit_local_method_authority"
            if any("control_plane_authority_missing" in blocker for blocker in blockers)
            else "bind_prequalified_runtime"
            if any("runtime_binding_missing" in blocker for blocker in blockers)
            else "resolve_source_binding_mismatch"
            if source_mismatch
            else "execute_registered_tools_under_recorded_authority"
        }
    )
    readiness = {
        "schema_version": RECONSTRUCTION_EXECUTION_READINESS_SCHEMA_VERSION,
        "readiness_id": (
            "reconstruction-readiness-" + route["capture_reconstruction_route_digest"][7:31]
        ),
        "capture_build_digest": capture_build["capture_build_digest"],
        "source_binding": source_binding,
        "capture_reconstruction_route_digest": route["capture_reconstruction_route_digest"],
        "capture_authority_profile": route["capture_authority_profile"],
        "tool_registry_digest": registry["tool_registry_digest"],
        "bound_tool_ids": bound,
        "control_plane_binding": control_binding,
        "source_commit_sha": source_commit_sha,
        "recorded_at": recorded_at,
        "status": status,
        "stages": stages,
        "blockers": blockers,
        "next_legal_actions": next_actions,
        "proof_boundary": {
            "readiness_is_execution_authority": False,
            "readiness_is_reconstruction_evidence": False,
            "agent_can_grant_authority": False,
            "agent_can_change_proof_state": False,
            "raw_capture_remains_authoritative": True,
            "appearance_is_metric_or_collision_truth": False,
            "isaac_compatibility_is_physical_success": False,
            "proof_effect": "none",
            "claim_ceiling": "execution_readiness_support_only",
        },
    }
    readiness["reconstruction_execution_readiness_digest"] = canonical_digest(
        readiness,
        digest_field="reconstruction_execution_readiness_digest",
    )
    return validate_reconstruction_execution_readiness(readiness)


def validate_reconstruction_execution_readiness(value: Mapping[str, Any]) -> dict[str, Any]:
    readiness = _clone(dict(value))
    required = {
        "schema_version",
        "readiness_id",
        "capture_build_digest",
        "source_binding",
        "capture_reconstruction_route_digest",
        "capture_authority_profile",
        "tool_registry_digest",
        "bound_tool_ids",
        "control_plane_binding",
        "source_commit_sha",
        "recorded_at",
        "status",
        "stages",
        "blockers",
        "next_legal_actions",
        "proof_boundary",
        "reconstruction_execution_readiness_digest",
    }
    boundary = readiness.get("proof_boundary")
    stages = readiness.get("stages")
    if (
        set(readiness) != required
        or readiness.get("schema_version") != RECONSTRUCTION_EXECUTION_READINESS_SCHEMA_VERSION
        or not str(readiness.get("readiness_id") or "").startswith("reconstruction-readiness-")
        or not _digest(readiness.get("capture_build_digest"))
        or not _digest(readiness.get("capture_reconstruction_route_digest"))
        or not _digest(readiness.get("tool_registry_digest"))
        or _COMMIT_RE.fullmatch(str(readiness.get("source_commit_sha") or "")) is None
        or not _timestamp(readiness.get("recorded_at"))
        or readiness.get("status")
        not in {
            "route_unresolved",
            "source_binding_mismatch",
            "not_ready",
            "ready_for_bounded_execution",
        }
        or readiness.get("bound_tool_ids") != _strings(readiness.get("bound_tool_ids"))
        or readiness.get("blockers") != _strings(readiness.get("blockers"))
        or readiness.get("next_legal_actions") != _strings(readiness.get("next_legal_actions"))
        or not isinstance(stages, list)
        or not isinstance(boundary, Mapping)
        or boundary
        != {
            "readiness_is_execution_authority": False,
            "readiness_is_reconstruction_evidence": False,
            "agent_can_grant_authority": False,
            "agent_can_change_proof_state": False,
            "raw_capture_remains_authoritative": True,
            "appearance_is_metric_or_collision_truth": False,
            "isaac_compatibility_is_physical_success": False,
            "proof_effect": "none",
            "claim_ceiling": "execution_readiness_support_only",
        }
        or readiness.get("reconstruction_execution_readiness_digest")
        != canonical_digest(
            readiness,
            digest_field="reconstruction_execution_readiness_digest",
        )
    ):
        raise ReconstructionExecutionReadinessError("execution_readiness_contract_invalid")
    for ordinal, stage in enumerate(stages):
        if (
            not isinstance(stage, Mapping)
            or set(stage)
            != {
                "ordinal",
                "stage_id",
                "method_kind",
                "implementation_status",
                "typed_tool_registered",
                "runtime_bound",
                "control_plane_authority_status",
                "readiness_status",
                "blockers",
            }
            or stage.get("ordinal") != ordinal
            or not str(stage.get("stage_id") or "")
            or not isinstance(stage.get("typed_tool_registered"), bool)
            or not isinstance(stage.get("runtime_bound"), bool)
            or stage.get("blockers") != _strings(stage.get("blockers"))
        ):
            raise ReconstructionExecutionReadinessError("execution_readiness_stage_invalid")
    source_binding = readiness.get("source_binding")
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("capture_build_digest") != readiness.get("capture_build_digest")
        or source_binding.get("source_binding_is_proof_upgrade") is not False
        or source_binding.get("raw_media_included") is not False
        or source_binding.get("source_binding_digest")
        != canonical_digest(source_binding, digest_field="source_binding_digest")
    ):
        raise ReconstructionExecutionReadinessError("execution_readiness_source_invalid")
    return readiness


__all__ = [
    "RECONSTRUCTION_EXECUTION_READINESS_SCHEMA_VERSION",
    "ReconstructionExecutionReadinessError",
    "bound_tool_ids_for_control_plane_inspection",
    "build_reconstruction_execution_readiness",
    "validate_reconstruction_execution_readiness",
]
