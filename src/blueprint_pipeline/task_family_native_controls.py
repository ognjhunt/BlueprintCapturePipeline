"""Qualify one family scene inside its admitted worker before policy queries.

Each retained probe/control uses a fresh registered native runtime, the exact
scene contract and independent scorer/media path. The SDK owns real robot
removal, unactuated physics stepping, scripted actuation and renderer readback.
Missing driver support is a precise preparation gap, never a positive result.
"""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_family_native_contracts import validate_task_family_native_runtime_contract
from .task_family_native_runtime import build_task_family_native_runtime
from .task_evaluation_task_family_native_adapter import execute_task_family_episode, native_runtime_provenance

SCHEMA = "task_family_native_site_qualification.v1"
CONTROLS = ("robot_free_probe", "zero_action_negative", "deterministic_scripted_positive")


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ValueError("task_family_native_site_" + code)


def validate_task_family_native_scene_readiness(runtime: Any, native: Mapping[str, Any], *,
                                                robot_present: bool = True) -> dict[str, Any]:
    """Verify one constructed instance's native scene and calibrated cameras."""
    native_readiness = runtime.native_scene_readiness()
    try:
        row = json.loads(json.dumps(native_readiness, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("task_family_native_site_readiness_invalid") from exc
    _require(isinstance(row, dict) and row.get("native_provenance") == native_runtime_provenance(native)
             and row.get("source") == "native_simulator_state"
             and row.get("camera_measurement_source") == "native_renderer"
             and row.get("robot_present") is robot_present
             and row.get("asset_digests") == {asset["asset_id"]: asset["content_digest"] for asset in native["assets"]},
             "native_scene_readiness_mismatch")
    cameras = row.get("cameras")
    _require(isinstance(cameras, dict) and set(cameras) == {camera["camera_id"] for camera in native["cameras"]},
             "native_camera_readiness_missing")
    appearance_ids = {asset["asset_id"] for asset in native["assets"] if asset.get("semantic_role") == "scene_appearance"}
    for camera in native["cameras"]:
        measured = cameras[camera["camera_id"]]
        _require(isinstance(measured, dict) and all(measured.get(key) == camera[key]
                 for key in ("pose_world", "intrinsics", "resolution"))
                 and measured.get("rendered") is True
                 and isinstance(measured.get("rendered_scene_asset_ids"), list)
                 and appearance_ids <= set(measured["rendered_scene_asset_ids"]), "native_camera_readiness_mismatch")
    return row


class _ControlRuntime:
    def __init__(self, runtime: Any, native: Mapping[str, Any], *, control: str) -> None:
        self.runtime, self.control = runtime, control
        self.native = native
        self.readiness: dict[str, Any] = {}
        self.states: list[dict[str, Any]] = []
        self.steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def reset(self, *, seed: int) -> Any:
        result = self.runtime.reset(seed=seed)
        self.steps = 0
        self.runtime.set_robot_enabled(enabled=self.control != "robot_free_probe")
        self.readiness = validate_task_family_native_scene_readiness(
            self.runtime, self.native, robot_present=self.control != "robot_free_probe")
        return result

    def native_state(self) -> Any:
        row = self.runtime.native_state()
        _require(isinstance(row, Mapping) and row.get("robot_present") is (self.control != "robot_free_probe"),
                 "robot_presence_readback_mismatch")
        if self.control != "deterministic_scripted_positive":
            _require(row.get("robot_action_applied") is False, "unactuated_control_readback_mismatch")
        elif self.steps:
            _require(row.get("robot_action_applied") is True, "scripted_control_action_readback_missing")
        self.states.append(json.loads(json.dumps(row, default=lambda value: value.tolist(), allow_nan=False)))
        return row

    def step(self, action: list[float], *, duration_seconds: float) -> Any:
        if self.control == "deterministic_scripted_positive":
            result = self.runtime.step(action, duration_seconds=duration_seconds)
            self.steps += 1
            return result
        return self.runtime.step_without_robot_action(duration_seconds=duration_seconds)


class _ControlPolicy:
    def __init__(self, runtime: Any, *, control: str) -> None:
        self.runtime, self.control, self.index = runtime, control, 0

    def act(self, observation: Mapping[str, Any]) -> Any:
        # The negative controls advance only native physics, never a robot
        # command. This placeholder vector never reaches an actuator.
        action = (self.runtime.scripted_control_action(copy.deepcopy(observation), step_index=self.index)
                  if self.control == "deterministic_scripted_positive" else [0.0])
        self.index += 1
        return action


def qualify_task_family_site(native_contract: Mapping[str, Any], *,
                             qualified_runtime_selection: Mapping[str, Any],
                             runtime_catalog_path: str | Path, asset_root: str | Path,
                             output_dir: str | Path, maximum_steps: int | None = None,
                             control_frequency_hz: float | None = None) -> dict[str, Any]:
    """Retain mandatory native physics, plus any owner-enabled control trials."""
    from .task_family_control_omission import make_family_control_omission

    native = validate_task_family_native_runtime_contract(native_contract)
    task = native["task_spec"]["task_success_contract"]
    omission = make_family_control_omission(task["contract_digest"])
    required = ("robot_free_probe",) if omission is not None else CONTROLS
    native_steps = native["task_spec"].get("maximum_action_steps")
    native_hz = native["task_spec"].get("control_frequency_hz")
    _require((maximum_steps is None or native_steps is None or maximum_steps == native_steps)
             and (control_frequency_hz is None or native_hz is None or control_frequency_hz == native_hz),
             "episode_bounds_mismatch")
    maximum_steps = native_steps if maximum_steps is None else maximum_steps
    hz = native_hz if control_frequency_hz is None else control_frequency_hz
    _require(type(maximum_steps) is int and maximum_steps > 0
             and type(hz) in (int, float) and math.isfinite(hz) and hz > 0,
             "episode_bounds_required")
    output = Path(output_dir)
    _require(not output.exists() or output.is_dir() and not any(output.iterdir()), "output_exists")
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    failure = None
    for control in required:
        runtime = None
        execution_owns_runtime = False
        try:
            runtime = build_task_family_native_runtime(native,
                qualified_runtime_selection=qualified_runtime_selection,
                runtime_catalog_path=runtime_catalog_path, asset_root=asset_root)
            control_runtime = _ControlRuntime(runtime, native, control=control)
            execution_owns_runtime = True
            episode = execute_task_family_episode(runtime=control_runtime,
                policy=_ControlPolicy(runtime, control=control), contract=task,
                native_runtime_contract=native, maximum_steps=maximum_steps, control_frequency_hz=float(hz),
                output_dir=output / control, episode_id=control, seed=native["seed"])
            expected_success = control == "deterministic_scripted_positive"
            passed = (episode["status"] == "completed" and episode["task_succeeded"] is expected_success
                and episode["scoring"]["status"] == ("succeeded" if expected_success else "failed")
                and all(sample["safety_violation"] is False for sample in episode["telemetry"]["samples"]))
            rows[control] = {"passed": passed, "episode": episode,
                "native_scene_readiness": control_runtime.readiness, "native_state_trace": control_runtime.states,
                "native_application": runtime.construction_receipt,
                "runtime_qualification_digest": runtime.qualification["qualification_digest"]}
            if not passed:
                failure = {"reason": "task_family_native_site_" + control + "_failed", "episode_failure": episode["failure"]}
                break
        except Exception as exc:
            failure = {"type": type(exc).__name__, "reason": str(exc)}
            break
        finally:
            if runtime is not None and not execution_owns_runtime:
                try:
                    runtime.close()
                except Exception as exc:
                    failure = {"type": type(exc).__name__, "reason": "runtime_close_failed:" + str(exc)}
    passed = failure is None and len(rows) == len(required)
    result = {"schema_version": SCHEMA, "status": "passed" if passed else "failed",
        "native_provenance": native_runtime_provenance(native), "task_success_contract_digest": task["contract_digest"],
        "controls": rows, "failure": failure, "candidate_policy_queried": False,
        "controls_qualified": passed and omission is None, "qualified_comparison_permitted": False,
        **({"control_omission_authority": omission, "omitted_controls": list(omission["omitted_controls"])}
           if omission is not None else {}),
        "scoring_source": "deterministic_native_state", "physical_validation": False,
        "claim_ceiling": task["claim_ceiling"]}
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    write_json(output / "task_family_native_site_qualification.v1.json", result)
    return result


__all__ = ["qualify_task_family_site", "validate_task_family_native_scene_readiness"]
