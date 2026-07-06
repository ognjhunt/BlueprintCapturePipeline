"""LeRobot-format policy family loading for real closed-loop evaluation.

A "policy family" is a LeRobot-format checkpoint directory:

- ``config.json`` — LeRobot ``PreTrainedConfig``-style JSON whose ``type`` field
  names the policy class (e.g. ``act``, ``diffusion``, ``pi0``, or the Blueprint
  CPU baseline ``blueprint_scripted_pick_place``).
- weights — ``model.safetensors`` for learned torch policies, or
  ``policy_weights.json`` for the CPU-runnable scripted baseline.

The scripted baseline exists so the full closed-loop eval plumbing
(orchestrator -> rollout -> SC3 scoring -> task_eval_run_report) is proven on a
real, checkpoint-loaded policy without a GPU. Learned LeRobot types are
recognized by the same loader but are never silently emulated: loading reports
``cpu_loadable=False`` and execution must go through a configured inference
command (a config-only swap, no harness code change).

Claim boundary: loading a checkpoint proves nothing about task success,
deployment readiness, or physical-robot behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso, write_json

ADAPTER_SCHEMA_VERSION = "blueprint_lerobot_policy_family_adapter.v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "blueprint_lerobot_policy_checkpoint_manifest.v1"

LEROBOT_CONFIG_FILENAME = "config.json"
SCRIPTED_WEIGHTS_FILENAME = "policy_weights.json"
LEARNED_WEIGHTS_FILENAME = "model.safetensors"

SCRIPTED_PICK_PLACE_TYPE = "blueprint_scripted_pick_place"
SCRIPTED_PICK_PLACE_FAMILY_ID = "blueprint_scripted_pick_place_v1"

# LeRobot policy types that require a torch inference runtime. The loader
# recognizes them so the checkpoint is still a first-class family, but never
# pretends to run them on CPU without their real runtime.
KNOWN_TORCH_POLICY_TYPES = (
    "act",
    "diffusion",
    "pi0",
    "pi0fast",
    "smolvla",
    "tdmpc",
    "vqbet",
    "groot",
)
GROOT_LIBERO_REMOTE_REPO_ID = "nvidia/gr00t17-lerobot-libero_10-640"
KNOWN_REMOTE_LEROBOT_CHECKPOINTS: dict[str, dict[str, Any]] = {
    GROOT_LIBERO_REMOTE_REPO_ID: {
        "type": "groot",
        "family_id": "nvidia_groot_n17_lerobot_libero_10_640",
        "n_obs_steps": 1,
        "input_features": {
            "observation.images.wrist_image": {"type": "VISUAL", "shape": [256, 256, 3]},
            "observation.images.image": {"type": "VISUAL", "shape": [256, 256, 3]},
            "observation.state": {"type": "STATE", "shape": [8]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "device": "cuda",
        "chunk_size": 16,
        "n_action_steps": 16,
        "action_decode_transform": "libero",
        "embodiment_tag": "libero_sim",
        "use_relative_actions": False,
    }
}

ACTION_DIM = 7  # SC3-style 7d delta end-effector action: dxyz, drpy, gripper.


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def checkpoint_sha256(checkpoint_dir: str | Path) -> str:
    """Stable digest over every file in the checkpoint directory."""
    root = Path(checkpoint_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class ScriptedPickPlacePolicy:
    """Deterministic pick-and-place baseline over 7d delta-EE actions.

    All parameters come from the checkpoint's ``policy_weights.json`` — the
    policy is genuinely checkpoint-loaded, not hard-coded. The phase machine is
    re-derived from each observation, so the policy is stateless across
    queries (receding-horizon requeries are safe).
    """

    params: dict[str, Any] = field(default_factory=dict)

    @property
    def policy_id(self) -> str:
        return _string(self.params.get("policy_id")) or SCRIPTED_PICK_PLACE_FAMILY_ID

    @property
    def chunk_size(self) -> int:
        return max(1, int(_number(self.params.get("chunk_size"), 25)))

    def _p(self, key: str, default: float) -> float:
        return _number(self.params.get(key), default)

    def select_action_chunk(
        self, observation: Mapping[str, Any], *, chunk_size: int | None = None
    ) -> list[dict[str, Any]]:
        steps = max(1, int(chunk_size or self.chunk_size))
        ee = _mapping(observation.get("end_effector"))
        ee_pos = [
            _number(v, 0.0) for v in (ee.get("position_xyz") or [0.0, 0.0, 0.0])
        ][:3]
        gripper_opening = _number(ee.get("gripper_opening_m"), 0.05)
        target = self._commanded_target(observation)
        goal = _mapping(observation.get("goal_zone"))
        goal_xyz = [
            _number(v, 0.0) for v in (goal.get("center_xyz") or [0.0, 0.0, 0.0])
        ][:3]

        actions: list[dict[str, Any]] = []
        sim_ee = list(ee_pos)
        sim_opening = gripper_opening
        obj_xyz = [
            _number(v, 0.0) for v in (target.get("position_xyz") or [0.0, 0.0, 0.0])
        ][:3]
        holding = bool(target.get("grasped"))
        # Seed the transition detector from the measured aperture, so a chunk
        # whose first action already flips the gripper (e.g. close right at the
        # grasp pose) still settles for the full chunk instead of assuming the
        # grasp landed and moving on within the same chunk.
        open_cmd = self._p("gripper_open_command", 1.0)
        closed_cmd = self._p("gripper_closed_command", 0.0)
        previous_grip: float | None = (
            open_cmd
            if gripper_opening > self._p("holding_opening_max_m", 0.045)
            else closed_cmd
        )
        for _ in range(steps):
            action, sim_ee, sim_opening, holding, obj_xyz = self._next_action(
                ee_pos=sim_ee,
                gripper_opening=sim_opening,
                holding=holding,
                obj_xyz=obj_xyz,
                goal_xyz=goal_xyz,
            )
            grip = _number(action.get("gripper_command"), 1.0)
            if previous_grip is not None and grip != previous_grip:
                # A grip transition depends on a contact outcome the projection
                # cannot verify (did the grasp/release actually happen?). Hold
                # position with the new grip for the rest of the chunk so the
                # next requery decides the following phase from measured state.
                settle = {
                    "action_type": "delta_end_effector_pose",
                    "delta_xyz_m": [0.0, 0.0, 0.0],
                    "delta_rpy_rad": [0.0, 0.0, 0.0],
                    "gripper_command": round(float(grip), 6),
                    "action_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, round(float(grip), 6)],
                    "settle_after_grip_transition": True,
                }
                while len(actions) < steps:
                    actions.append(dict(settle))
                break
            previous_grip = grip
            actions.append(action)
        return actions

    def _commanded_target(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        objects = [
            _mapping(item)
            for item in observation.get("objects") or []
            if isinstance(item, Mapping)
        ]
        for candidate in objects:
            if _string(candidate.get("role")) == "commanded_target":
                return candidate
        return objects[0] if objects else {}

    def _next_action(
        self,
        *,
        ee_pos: Sequence[float],
        gripper_opening: float,
        holding: bool,
        obj_xyz: Sequence[float],
        goal_xyz: Sequence[float],
    ) -> tuple[dict[str, Any], list[float], float, bool, list[float]]:
        gain = self._p("gain", 0.9)
        max_step = self._p("max_step_m", 0.02)
        approach_height = self._p("approach_height_m", 0.10)
        grasp_height = self._p("grasp_height_m", 0.012)
        grasp_xy_tol = self._p("grasp_xy_tolerance_m", 0.008)
        transport_height = self._p("transport_height_m", 0.12)
        place_xy_tol = self._p("place_xy_tolerance_m", 0.03)
        place_height = self._p("place_height_m", 0.05)
        open_cmd = self._p("gripper_open_command", 1.0)
        closed_cmd = self._p("gripper_closed_command", 0.0)
        holding_opening_max = self._p("holding_opening_max_m", 0.045)
        holding_opening_min = self._p("holding_opening_min_m", 0.02)
        object_width = self._p("object_width_hint_m", 0.04)
        open_opening = self._p("gripper_max_opening_m", 0.06)

        ee = [float(v) for v in ee_pos]
        obj = [float(v) for v in obj_xyz]
        goal = [float(v) for v in goal_xyz]
        xy_err_obj = math.hypot(obj[0] - ee[0], obj[1] - ee[1])
        xy_err_goal = math.hypot(goal[0] - ee[0], goal[1] - ee[1])
        # Holding is inferred from the gripper aperture (an object between the
        # fingers stops them before the empty-close width) plus proximity, so a
        # requery mid-transport never mistakes "airborne with object" for
        # "need to re-approach" and drops it.
        if (
            holding_opening_min <= gripper_opening <= holding_opening_max
            and xy_err_obj <= 0.05
            and abs(ee[2] - obj[2]) <= 0.10
        ):
            holding = True
        elif gripper_opening > holding_opening_max or gripper_opening < holding_opening_min:
            holding = False

        if holding:
            grip = closed_cmd
            transport_z = goal[2] + transport_height
            if xy_err_goal > place_xy_tol:
                if ee[2] < transport_z - 0.01:
                    desired = [ee[0], ee[1], transport_z]
                else:
                    desired = [goal[0], goal[1], transport_z]
            elif ee[2] > goal[2] + place_height + 0.005:
                desired = [goal[0], goal[1], goal[2] + place_height]
            else:
                desired = [goal[0], goal[1], goal[2] + place_height]
                grip = open_cmd
                holding = False
        else:
            grip = open_cmd
            if xy_err_obj > grasp_xy_tol:
                desired = [obj[0], obj[1], obj[2] + approach_height]
            elif ee[2] > obj[2] + grasp_height + 0.003:
                desired = [obj[0], obj[1], obj[2] + grasp_height]
            else:
                desired = [obj[0], obj[1], obj[2] + grasp_height]
                grip = closed_cmd

        delta = [
            _clamp(gain * (desired[axis] - ee[axis]), -max_step, max_step)
            for axis in range(3)
        ]
        next_ee = [ee[axis] + delta[axis] for axis in range(3)]
        next_opening = object_width if grip == closed_cmd else open_opening
        if holding and grip == closed_cmd:
            obj = [next_ee[0], next_ee[1], next_ee[2] + 0.01]
        action = {
            "action_type": "delta_end_effector_pose",
            "delta_xyz_m": [round(v, 6) for v in delta],
            "delta_rpy_rad": [0.0, 0.0, 0.0],
            "gripper_command": round(float(grip), 6),
            "action_7d": [round(v, 6) for v in delta]
            + [0.0, 0.0, 0.0, round(float(grip), 6)],
        }
        return action, next_ee, next_opening, holding, obj


@dataclass
class LoadedPolicyFamily:
    family_id: str
    policy_type: str
    checkpoint_dir: str
    checkpoint_sha256: str
    cpu_loadable: bool
    requires_torch_runtime: bool
    policy: ScriptedPickPlacePolicy | None
    config: dict[str, Any]
    blockers: list[str]
    checkpoint_reference_kind: str = "local_dir"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "family_id": self.family_id,
            "policy_type": self.policy_type,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_reference_kind": self.checkpoint_reference_kind,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_format": "lerobot_pretrained_dir",
            "cpu_loadable": self.cpu_loadable,
            "requires_torch_runtime": self.requires_torch_runtime,
            "blockers": list(self.blockers),
            "claim_boundary": {
                "checkpoint_load_is_not_task_success": True,
                "checkpoint_load_is_not_deployment_readiness": True,
                "scripted_baseline_is_not_a_learned_policy": self.policy_type
                == SCRIPTED_PICK_PLACE_TYPE,
            },
        }


def load_lerobot_policy_checkpoint(checkpoint_dir: str | Path) -> LoadedPolicyFamily:
    checkpoint_text = _string(checkpoint_dir)
    root = Path(checkpoint_text).expanduser().resolve()
    blockers: list[str] = []
    config: dict[str, Any] = {}
    policy_type = ""
    checkpoint_reference_kind = "local_dir"
    if not root.is_dir():
        remote_config = KNOWN_REMOTE_LEROBOT_CHECKPOINTS.get(checkpoint_text)
        if remote_config:
            config = dict(remote_config)
            policy_type = _string(config.get("type"))
            checkpoint_reference_kind = "hf_repo_id"
        else:
            blockers.append("checkpoint_dir_missing")
    else:
        config_path = root / LEROBOT_CONFIG_FILENAME
        if not config_path.is_file():
            blockers.append("lerobot_config_json_missing")
        else:
            try:
                config = _mapping(json.loads(config_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                blockers.append("lerobot_config_json_unreadable")
            policy_type = _string(config.get("type"))
            if not policy_type:
                blockers.append("lerobot_config_type_missing")

    policy: ScriptedPickPlacePolicy | None = None
    cpu_loadable = False
    requires_torch_runtime = False
    if not blockers:
        if policy_type == SCRIPTED_PICK_PLACE_TYPE:
            weights_path = root / SCRIPTED_WEIGHTS_FILENAME
            if not weights_path.is_file():
                blockers.append("scripted_policy_weights_missing")
            else:
                try:
                    params = _mapping(
                        json.loads(weights_path.read_text(encoding="utf-8"))
                    )
                except (OSError, json.JSONDecodeError):
                    params = {}
                    blockers.append("scripted_policy_weights_unreadable")
                if not blockers:
                    policy = ScriptedPickPlacePolicy(params=params)
                    cpu_loadable = True
        elif policy_type in KNOWN_TORCH_POLICY_TYPES:
            requires_torch_runtime = True
            if checkpoint_reference_kind == "local_dir" and not (
                root / LEARNED_WEIGHTS_FILENAME
            ).is_file():
                blockers.append("learned_policy_safetensors_missing")
            blockers.append("policy_type_requires_torch_inference_runtime")
        else:
            blockers.append(f"unsupported_lerobot_policy_type:{policy_type}")

    family_id = (
        _string(config.get("family_id"))
        or (
            _string(_mapping(config).get("policy_id"))
            if config
            else ""
        )
        or (
            SCRIPTED_PICK_PLACE_FAMILY_ID
            if policy_type == SCRIPTED_PICK_PLACE_TYPE
            else f"lerobot_{policy_type}_family"
            if policy_type
            else "unknown_policy_family"
        )
    )
    return LoadedPolicyFamily(
        family_id=family_id,
        policy_type=policy_type,
        checkpoint_dir=str(root),
        checkpoint_sha256=checkpoint_sha256(root) if root.is_dir() else "",
        cpu_loadable=cpu_loadable,
        requires_torch_runtime=requires_torch_runtime,
        policy=policy,
        config=config,
        blockers=blockers,
        checkpoint_reference_kind=checkpoint_reference_kind,
    )


def create_scripted_baseline_checkpoint(
    checkpoint_dir: str | Path,
    *,
    policy_id: str = SCRIPTED_PICK_PLACE_FAMILY_ID,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Write a LeRobot-format checkpoint dir for the scripted CPU baseline."""
    root = Path(checkpoint_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "type": SCRIPTED_PICK_PLACE_TYPE,
        "family_id": policy_id,
        "n_action_steps": 16,
        "chunk_size": 25,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [8]},
            "observation.environment_state": {"type": "ENV", "shape": [9]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [ACTION_DIM]}},
    }
    weights = {
        "family": SCRIPTED_PICK_PLACE_TYPE,
        "policy_id": policy_id,
        "gain": 0.9,
        "max_step_m": 0.02,
        "approach_height_m": 0.10,
        "grasp_height_m": 0.012,
        "grasp_xy_tolerance_m": 0.008,
        "transport_height_m": 0.12,
        "place_xy_tolerance_m": 0.03,
        "place_height_m": 0.05,
        "gripper_open_command": 1.0,
        "gripper_closed_command": 0.0,
        "gripper_max_opening_m": 0.06,
        "holding_opening_max_m": 0.045,
        "holding_opening_min_m": 0.02,
        "object_width_hint_m": 0.04,
        "chunk_size": 25,
    }
    weights.update(_mapping(overrides))
    write_json(root / LEROBOT_CONFIG_FILENAME, config)
    write_json(root / SCRIPTED_WEIGHTS_FILENAME, weights)
    return root


def _adapter_response(
    *,
    loaded: LoadedPolicyFamily,
    observation: Mapping[str, Any],
    chunk_size: int | None,
) -> dict[str, Any]:
    if loaded.blockers or loaded.policy is None:
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "status": "blocked",
            "policy_id": loaded.family_id,
            "policy_type": loaded.policy_type,
            "blockers": list(loaded.blockers) or ["policy_not_cpu_loadable"],
            "claim_boundary": {
                "policy_command_ran": False,
                "single_action_is_not_episode_success": True,
            },
        }
    chunk = loaded.policy.select_action_chunk(observation, chunk_size=chunk_size)
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "completed",
        "policy_id": loaded.policy.policy_id,
        "policy_type": loaded.policy_type,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "action": chunk[0],
        "action_chunk": chunk,
        "chunk_size": len(chunk),
        "claim_boundary": {
            "policy_command_ran": True,
            "single_action_is_not_episode_success": True,
            "scripted_baseline_is_not_a_learned_policy": True,
            "task_success_proven": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="LeRobot checkpoint dir")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--manifest-out", default=None, help="Optional checkpoint manifest JSON path"
    )
    args = parser.parse_args(argv)

    loaded = load_lerobot_policy_checkpoint(args.checkpoint)
    if args.manifest_out:
        manifest = loaded.manifest()
        manifest["generated_at"] = utc_now_iso()
        write_json(Path(args.manifest_out), manifest)

    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    observation = _mapping(payload.get("observation")) or _mapping(payload)

    response = _adapter_response(
        loaded=loaded, observation=observation, chunk_size=args.chunk_size
    )
    encoded = json.dumps(response, sort_keys=True)
    output_path = os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if response.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
