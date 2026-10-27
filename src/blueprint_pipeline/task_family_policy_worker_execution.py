"""Production worker bridge from qualified family selection to native execution."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_task_family_native_adapter import execute_task_family_episode
from .task_family_native_contracts import validate_task_family_native_runtime_contract
from .task_family_policy_https_client import INTERFACE_ID, TaskFamilyPolicySpec


def is_task_family_scene_plan(plan: Mapping[str, Any]) -> bool:
    return isinstance(plan.get("native_runtime_contract"), Mapping)


def verify_family_operator_inputs(runtime_root: Path, manifest: Mapping[str, Any]) -> Path:
    """Rehash the operator config staged by the authenticated bundle builder."""
    binding = manifest.get("task_family_operator_inputs")
    if (not isinstance(binding, Mapping)
            or binding.get("catalog_relative_path") != "runtime_operator_inputs/catalog.json"
            or not isinstance(binding.get("files"), list) or not binding["files"]):
        raise ValueError("task_family_operator_input_manifest_required")
    seen = set()
    root = Path(runtime_root).resolve()
    for row in binding["files"]:
        if not isinstance(row, Mapping):
            raise ValueError("task_family_operator_input_identity_mismatch")
        relative = str(row.get("relative_path") or "")
        path = root / relative
        if (relative in seen or not relative.startswith("runtime_operator_inputs/")
                or Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.is_symlink() or not path.is_file()
                or any(parent.is_symlink() for parent in path.parents)
                or not path.resolve().is_relative_to(root)
                or path.stat().st_size != row.get("size_bytes")
                or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256")):
            raise ValueError("task_family_operator_input_identity_mismatch")
        seen.add(relative)
    if binding["catalog_relative_path"] not in seen:
        raise ValueError("task_family_operator_input_catalog_missing")
    catalog_path = root / binding["catalog_relative_path"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("catalog_digest") != binding.get("runtime_catalog_digest"):
        raise ValueError("task_family_operator_input_catalog_binding_mismatch")
    return catalog_path


class _NativePolicy:
    def __init__(self, client: Any, spec: TaskFamilyPolicySpec, progress: dict[str, Any]) -> None:
        self.client, self.spec, self.progress = client, spec, progress
        self.queries: list[dict[str, Any]] = []
        self.actions: list[list[float]] = []
        self.pending: list[list[float]] = []

    def act(self, observation: Mapping[str, Any]) -> list[float]:
        if not self.pending:
            self.progress["candidate_policy_query_attempted"] = True
            response = self.client.infer(observation)
            self.progress["candidate_policy_queried"] = True
            self.progress["candidate_action_returned"] = True
            query: dict[str, Any] = {"query_index": len(self.queries)}
            evidence = getattr(self.client, "last_inference_evidence", None)
            if callable(evidence):
                query["policy_inference_evidence"] = evidence()
            self.queries.append(query)
            self.progress["candidate_policy_action_queries"] = self.queries
            raw_values = np.asarray(response)
            if raw_values.dtype.kind not in "iuf":
                raise ValueError("task_family_policy_action_type_invalid")
            values = raw_values.astype(float)
            bounds = np.asarray(self.spec.action_bounds)
            if (values.ndim != 2 or values.shape[1] != len(bounds)
                    or not self.spec.open_loop_horizon <= len(values) <= self.spec.action_chunk_rows):
                raise ValueError("task_family_policy_action_shape_invalid")
            self.progress["candidate_action_shape_validated"] = True
            if not np.isfinite(values).all():
                raise ValueError("task_family_policy_action_nonfinite")
            self.progress["candidate_action_finite_validated"] = True
            if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
                raise ValueError("task_family_policy_action_bounds_invalid")
            self.progress["candidate_action_bounds_validated"] = True
            self.progress["candidate_action_validated"] = True
            query["actions"] = values.tolist()
            self.pending = values[:self.spec.open_loop_horizon].tolist()
        action = self.pending.pop(0)
        self.actions.append(action)
        return action


class _NativeRuntime:
    def __init__(self, runtime: Any, progress: dict[str, Any]) -> None:
        self.runtime, self.progress = runtime, progress
        self.states: list[dict[str, Any]] = []
        self.proprioception: list[list[float]] = []
        self.native_scene_readiness: dict[str, Any] = {}
        self.steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def reset(self, *, seed: int) -> None:
        from .task_family_native_controls import validate_task_family_native_scene_readiness

        self.runtime.reset(seed=seed)
        self.native_scene_readiness = validate_task_family_native_scene_readiness(
            self.runtime, self.runtime.native_runtime_contract, robot_present=True)

    def native_state(self) -> Any:
        state = self.runtime.native_state()
        self.states.append(json.loads(json.dumps(state, allow_nan=False, default=lambda value: value.tolist())))
        if self.steps > 0 and state.get("robot_action_applied") is not True:
            raise ValueError("task_family_policy_native_action_delivery_unproven")
        return state

    def policy_observation(self) -> Any:
        observation = self.runtime.policy_observation()
        self.proprioception.append(np.asarray(observation["proprioception"], dtype=float).tolist())
        return observation

    def step(self, action: Any, *, duration_seconds: float) -> None:
        self.runtime.step(action, duration_seconds=duration_seconds)
        self.steps += 1
        self.progress.update(candidate_action_applied=True, episode_running=True)

    def close(self) -> None:
        self.runtime.close()


def execute_family_policy_episode(
    *, scene_plan: Mapping[str, Any], spec: Mapping[str, Any], policy_client: Any,
    packet_root: Path, output_dir: Path, episode_id: str,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build through the operator's registered factory and retain native evidence.

    This function allocates no resources and grants no execution authority. Its
    callers retain the existing rights, spend, controls and session admission.
    """
    from .task_family_native_runtime import build_task_family_native_runtime
    from .task_family_policy_execution import (
        family_policy_execution_binding, validate_task_family_policy_execution_spec,
    )

    progress = progress if progress is not None else {}
    validated = validate_task_family_policy_execution_spec(spec, candidate_id=str(spec.get("candidate_id") or ""))
    binding = family_policy_execution_binding(scene_plan)
    if any(validated.get(key) != value for key, value in binding.items()):
        raise ValueError("task_family_policy_scene_binding_mismatch")
    native_contract = validate_task_family_native_runtime_contract(scene_plan["native_runtime_contract"])
    native_spec = TaskFamilyPolicySpec(**validated["policy_spec"])
    native_spec.validate()
    configuration = native_contract["robot_configuration"]
    if any(getattr(native_spec, key) != configuration[key] for key in (
            "embodiment_id", "controller", "action_schema_id", "observation_schema_id")):
        raise ValueError("task_family_policy_robot_interface_mismatch")
    catalog_path = os.environ.get("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH")
    if not catalog_path:
        raise ValueError("task_family_qualified_runtime_catalog_not_configured")
    maximum_steps = validated.get("maximum_action_steps")
    control_hz = validated.get("control_frequency_hz")
    if (type(maximum_steps) is not int or maximum_steps < 1
            or not isinstance(control_hz, (int, float)) or isinstance(control_hz, bool)
            or not np.isfinite(control_hz) or control_hz <= 0
            or (maximum_steps + native_spec.open_loop_horizon - 1) // native_spec.open_loop_horizon
            != validated["max_policy_queries"]):
        raise ValueError("task_family_policy_episode_budget_mismatch")
    runtime = build_task_family_native_runtime(native_contract,
        qualified_runtime_selection=binding["task_family_runtime_selection"],
        runtime_catalog_path=Path(catalog_path), asset_root=Path(packet_root))
    try:
        if INTERFACE_ID not in runtime.runtime_registration.get("policy_interfaces", []):
            raise ValueError("task_family_policy_interface_not_qualified_for_runtime")
        readiness = getattr(policy_client, "preflight_readiness", None)
        if not callable(readiness):
            raise ValueError("task_family_policy_control_plane_readiness_unavailable")
        policy_readiness = readiness()
        if (not isinstance(policy_readiness, Mapping)
                or policy_readiness.get("identity_verified") is not True
                or policy_readiness.get("candidate_inference_performed") is not False):
            raise ValueError("task_family_policy_control_plane_not_ready")
    except BaseException:
        runtime.close()
        raise
    tracked_runtime = _NativeRuntime(runtime, progress)
    policy = _NativePolicy(policy_client, native_spec, progress)
    receipt = execute_task_family_episode(runtime=tracked_runtime, policy=policy,
        contract=native_contract["task_spec"]["task_success_contract"], maximum_steps=maximum_steps,
        control_frequency_hz=float(control_hz), output_dir=Path(output_dir), episode_id=episode_id,
        seed=native_contract["seed"], native_runtime_contract=native_contract)
    visual = receipt["visual_evidence"]
    first_frame_retained = (visual.get("status") == "complete"
                            or visual.get("retained_policy_input_frame_count", 0) > 0)
    progressed = (len(tracked_runtime.states) > 1 and tracked_runtime.steps > 0
                  and all(state.get("robot_action_applied") is True for state in tracked_runtime.states[1:]))
    motion = {"actions_reached_robot": progressed,
              "arm_moved": False,
              "policy_state_changed": any(row != tracked_runtime.proprioception[0]
                  for row in tracked_runtime.proprioception[1:]),
              "motion_scope": "native_policy_proprioception",
              "policy_outcome_interpretable": receipt["status"] == "completed" and progressed,
              "native_action_steps": tracked_runtime.steps}
    score = dict(receipt.get("scoring") or {})
    if score:
        score["status"] = "scored"
    lifecycle = {"schema_version": "task_family_episode_lifecycle.v1",
                 "terminal_class": "planned_duration_complete" if receipt["status"] == "completed" else None,
                 "episode_started": first_frame_retained, "planned_action_steps": maximum_steps,
                 "actual_action_steps": tracked_runtime.steps, "actual_policy_queries": len(policy.queries),
                 "runtime_closed": receipt["runtime_closed"], "native_receipt_digest": receipt["receipt_digest"]}
    lifecycle["lifecycle_digest"] = canonical_digest(lifecycle, digest_field="lifecycle_digest")
    progress.update(first_observation_retained=first_frame_retained,
                    exact_policy_observation_retained=visual.get("status") == "complete",
                    candidate_policy_queried=bool(policy.queries) or progress.get("candidate_policy_queried") is True
                        or getattr(policy_client, "candidate_policy_queried", False) is True,
                    episode_started=first_frame_retained,
                    commanded_actions=policy.actions, candidate_policy_action_queries=policy.queries)
    progress["_failure_media_finalizer"] = lambda **_kwargs: (receipt["visual_evidence"], receipt["artifacts"])
    return {"schema_version": "native_task_family_policy_episode.v1", "status": receipt["status"],
            "native_family_receipt": receipt, "score": score, "lifecycle": lifecycle,
            "visual_evidence": visual, "media_artifacts": receipt["artifacts"],
            "motion_evidence": motion, "commanded_actions": policy.actions, "queries": policy.queries,
            "candidate_policy_action_queries": policy.queries, "policy_queries": len(policy.queries),
            "state_trace": tracked_runtime.states, "native_measurements": receipt.get("telemetry"),
            "prestart_readiness": {"policy_control_plane": dict(policy_readiness)},
            "episode_environment": {"adapter": "task_family_native_runtime",
                "native_runtime_contract_digest": native_contract["contract_digest"],
                "native_scene_readiness": tracked_runtime.native_scene_readiness,
                "qualification": dict(runtime.qualification),
                "runtime_registration": dict(runtime.runtime_registration)},
            "failure": receipt["failure"], "claim_ceiling": receipt["claim_ceiling"]}
