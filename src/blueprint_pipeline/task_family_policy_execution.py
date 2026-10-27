"""Data-only bindings for qualified family-native policy execution.

The selection names an operator-qualified catalog record. It never names an
import, executable, or catalog path supplied by a captured job.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_family_native_contracts import validate_task_family_native_runtime_contract
from .task_family_scoring import validate_task_family_contract

FAMILY_BACKEND = "task_family_native.v1"
FAMILY_POLICY_INTERFACE = "task_family_policy_https.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_task_family_runtime_selection(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) != {"runtime_id", "runtime_catalog_digest"}
            or not _ID.fullmatch(str(value.get("runtime_id") or ""))
            or not _DIGEST.fullmatch(str(value.get("runtime_catalog_digest") or ""))):
        raise ValueError("task_family_runtime_selection_invalid")
    return json.loads(json.dumps(dict(value), allow_nan=False))


def task_family_policy_execution_binding(scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact scene bindings copied into each immutable policy spec."""
    native = validate_task_family_native_runtime_contract(scene_plan.get("native_runtime_contract"))
    selection = validate_task_family_runtime_selection(scene_plan.get("task_family_runtime_selection"))
    task_spec = scene_plan.get("task_spec")
    scenario = scene_plan.get("scenario") or {}
    if (not isinstance(task_spec, Mapping)
            or task_spec.get("task_success_contract") != native["task_spec"]["task_success_contract"]
            or scene_plan.get("task_id") != native["task_id"]
            or scene_plan.get("scene_id") != native["scene_id"]
            or scene_plan.get("task_kind") != native["task_kind"]
            or scenario.get("cell_id") != native["scenario_cell_id"]
            or scenario.get("scenario_instance_digest") != native["scenario_instance_digest"]
            or scenario.get("seed") != native["seed"]):
        raise ValueError("task_family_policy_scene_binding_mismatch")
    source_digest = native["contract_digest"]
    if "source_native_runtime_contract" in scene_plan or "source_native_runtime_contract_digest" in scene_plan:
        source = validate_task_family_native_runtime_contract(scene_plan.get("source_native_runtime_contract"))
        source_digest = source["contract_digest"]
        if scene_plan.get("source_native_runtime_contract_digest") != source_digest:
            raise ValueError("task_family_policy_source_contract_digest_invalid")
        scenario_fields = {"scenario_cell_id", "scenario_instance_digest", "seed", "contract_digest"}
        if ({key: value for key, value in source.items() if key not in scenario_fields}
                != {key: value for key, value in native.items() if key not in scenario_fields}):
            raise ValueError("task_family_policy_resolved_contract_changed")
    return {"native_runtime_contract_digest": source_digest,
            "task_family_runtime_selection": selection}


# Short public spelling shared by the planner and worker. Retain the explicit
# spelling for existing callers that materialized specs during this rollout.
family_policy_execution_binding = task_family_policy_execution_binding


def validate_task_family_policy_execution_spec(
    spec: Mapping[str, Any], *, candidate_id: str | None = None,
    scene_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate frozen execution data without granting policy/runtime authority."""
    value = json.loads(json.dumps(spec, allow_nan=False))
    task = validate_task_family_contract(value.get("task_success_contract"))
    if (value.get("schema_version") not in {"native_task_arena_policy_execution_spec.v1",
                                           "native_task_arena_policy_canary_execution_spec.v1"}
            or value.get("evaluation_backend") != FAMILY_BACKEND
            or value.get("policy_interface_id") != FAMILY_POLICY_INTERFACE
            or not _ID.fullmatch(str(value.get("candidate_id") or ""))
            or (candidate_id is not None and value.get("candidate_id") != candidate_id)
            or value.get("execution_spec_digest") != canonical_digest(value, digest_field="execution_spec_digest")
            or value.get("task_success_contract_digest") != task["contract_digest"]
            or not _DIGEST.fullmatch(str(value.get("native_runtime_contract_digest") or ""))):
        raise ValueError("task_family_policy_execution_spec_binding_invalid")
    validate_task_family_runtime_selection(value.get("task_family_runtime_selection"))
    for field in ("checkpoint_digest", "runtime_identity_digest"):
        if not _DIGEST.fullmatch(str(value.get(field) or "")):
            raise ValueError("task_family_policy_identity_invalid")
    for field in ("max_policy_queries", "open_loop_horizon"):
        if type(value.get(field)) is not int or value[field] < 1:
            raise ValueError("task_family_policy_episode_limits_invalid")
    steps, hz = value.get("maximum_action_steps"), value.get("control_frequency_hz")
    if (type(steps) is not int or steps < 1 or type(hz) not in (int, float)
            or not math.isfinite(hz) or hz <= 0 or 1 / hz > task["maximum_sample_gap_seconds"]
            or (steps + value["open_loop_horizon"] - 1) // value["open_loop_horizon"] != value["max_policy_queries"]):
        raise ValueError("task_family_policy_episode_limits_invalid")
    if not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
        raise ValueError("task_family_policy_instruction_missing")
    if not isinstance(value.get("policy_spec"), Mapping) or not isinstance(value.get("candidate_rights_binding"), Mapping):
        raise ValueError("task_family_policy_authority_binding_missing")
    from .task_family_policy_https_client import TaskFamilyPolicySpec

    try:
        policy = TaskFamilyPolicySpec(**value["policy_spec"])
        policy.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("task_family_policy_interface_binding_invalid") from exc
    endpoint = value.get("policy_endpoint")
    if (not isinstance(endpoint, Mapping) or set(endpoint) != {"credential_env"}
            or re.fullmatch(r"[A-Z][A-Z0-9_]{2,58}_FILE", str(endpoint.get("credential_env") or "")) is None
            or value["open_loop_horizon"] != policy.open_loop_horizon
            or value["checkpoint_digest"] != policy.model_identity_digest):
        raise ValueError("task_family_policy_interface_binding_invalid")
    if scene_plan is not None:
        binding = task_family_policy_execution_binding(scene_plan)
        if (any(value.get(key) != expected for key, expected in binding.items())
                or task != scene_plan["task_spec"]["task_success_contract"]):
            raise ValueError("task_family_policy_execution_scene_mismatch")
        native_spec = scene_plan["native_runtime_contract"]["task_spec"]
        if any(field in native_spec and native_spec[field] != value[field]
               for field in ("maximum_action_steps", "control_frequency_hz")):
            raise ValueError("task_family_policy_execution_episode_bounds_mismatch")
    return value
