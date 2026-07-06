"""Torch/LeRobot inference adapter for real learned policy families.

Runs an actual learned LeRobot checkpoint (ACT or similar) behind the same
Blueprint policy-adapter contract the scripted baseline uses, so swapping the
GPU family into the closed-loop harness is config-only.

Modes:
- default: one observation in (stdin or BLUEPRINT_POLICY_ACTION_INPUT), one
  action-chunk response out.
- ``--serve``: persistent stdio server — one JSON request line in, one JSON
  response line out; the model loads once (required for closed-loop use).
- ``--batch-observations``: reads BLUEPRINT_POLICY_OBSERVATION_MANIFEST and
  writes BLUEPRINT_POLICY_EXECUTION_OUTPUT (the orchestrator's
  policy-execution-command contract) with one real inference per observation.

Honesty: checkpoints like ``lerobot/act_aloha_sim_transfer_cube_human`` were
trained on other embodiments. Observation/action mapping onto the Blueprint
tabletop proxy is a declared, deterministic projection and every response
carries ``out_of_distribution_embodiment_mapping: true``. Running the model
proves real learned-policy execution — it proves nothing about policy quality
on this task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ADAPTER_SCHEMA_VERSION = "blueprint_lerobot_torch_policy_adapter.v1"
POLICY_EXECUTION_OUTPUT_SCHEMA_VERSION = (
    "blueprint_lerobot_torch_policy_execution_output.v1"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _floats(value: Any, count: int) -> list[float]:
    items: list[float] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value[:count]:
            try:
                items.append(float(item))
            except (TypeError, ValueError):
                items.append(0.0)
    while len(items) < count:
        items.append(0.0)
    return items


def build_state_vector(observation: Mapping[str, Any], dim: int) -> list[float]:
    """Deterministic proprio projection into the checkpoint's state layout.

    Declared mapping (first 12 slots, zero-padded/truncated to ``dim``):
    ee xyz, ee yaw, gripper opening, target-object xyz relative to ee,
    goal xyz relative to ee, step fraction.
    """
    ee = _mapping(observation.get("end_effector"))
    ee_pos = _floats(ee.get("position_xyz"), 3)
    objects = [
        _mapping(item)
        for item in observation.get("objects") or []
        if isinstance(item, Mapping)
    ]
    target = next(
        (item for item in objects if _string(item.get("role")) == "commanded_target"),
        objects[0] if objects else {},
    )
    target_pos = _floats(target.get("position_xyz"), 3)
    goal = _mapping(observation.get("goal_zone"))
    goal_pos = _floats(goal.get("center_xyz"), 3)
    step_index = float(observation.get("step_index") or 0)
    vector = [
        ee_pos[0],
        ee_pos[1],
        ee_pos[2],
        float(ee.get("yaw_rad") or 0.0),
        float(ee.get("gripper_opening_m") or 0.0),
        target_pos[0] - ee_pos[0],
        target_pos[1] - ee_pos[1],
        target_pos[2] - ee_pos[2],
        goal_pos[0] - ee_pos[0],
        goal_pos[1] - ee_pos[1],
        goal_pos[2] - ee_pos[2],
        min(1.0, step_index / 400.0),
    ]
    return (vector + [0.0] * dim)[:dim]


def project_action_to_7d(
    raw_action: Sequence[float],
    *,
    previous_raw: Sequence[float] | None,
    max_step_m: float = 0.02,
) -> list[float]:
    """Declared OOD projection: checkpoint-native action -> 7d delta-EE.

    First three dims (joint/position deltas between consecutive model outputs)
    are squashed to bounded xyz deltas; dim 3 drives yaw; the last dim drives
    the gripper. Deterministic and documented — not a claim of semantic
    compatibility.
    """
    raw = [float(v) for v in raw_action]
    prev = [float(v) for v in previous_raw] if previous_raw is not None else raw
    deltas = [raw[i] - prev[i] if i < len(prev) else 0.0 for i in range(len(raw))]

    def _squash(value: float) -> float:
        return max_step_m * math.tanh(float(value))

    dx = _squash(deltas[0] if len(deltas) > 0 else 0.0)
    dy = _squash(deltas[1] if len(deltas) > 1 else 0.0)
    dz = _squash(deltas[2] if len(deltas) > 2 else 0.0)
    dyaw = 0.1 * math.tanh(deltas[3] if len(deltas) > 3 else 0.0)
    grip_source = raw[-1] if raw else 1.0
    grip = 0.5 * (math.tanh(grip_source) + 1.0)
    return [
        round(dx, 6),
        round(dy, 6),
        round(dz, 6),
        0.0,
        0.0,
        round(dyaw, 6),
        round(grip, 6),
    ]


class LeRobotTorchPolicyRunner:
    """Loads a real LeRobot checkpoint once and serves action chunks."""

    def __init__(self, *, checkpoint: str, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.policy_id = f"lerobot_torch::{checkpoint}"
        self._load()

    def _load(self) -> None:
        import torch  # noqa: F401  (real torch runtime required)

        try:
            from lerobot.policies.factory import get_policy_class
            from lerobot.configs.policies import PreTrainedConfig
        except ImportError:  # older lerobot package layout
            from lerobot.common.policies.factory import get_policy_class
            from lerobot.common.policies.pretrained import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = self.device
        policy_cls = get_policy_class(config.type)
        self.policy = policy_cls.from_pretrained(self.checkpoint, config=config)
        self.policy.eval()
        self.policy_type = config.type
        self.input_features = {
            key: tuple(feature.shape)
            for key, feature in (config.input_features or {}).items()
        }
        output_features = config.output_features or {}
        action_feature = output_features.get("action")
        self.action_dim = int(action_feature.shape[0]) if action_feature else 7

    def _batch(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from PIL import Image
        import numpy as np

        batch: dict[str, Any] = {}
        frame_path = _string(
            _mapping(observation.get("visual_observation")).get("camera_frame_path")
            or observation.get("camera_frame_path")
        )
        for key, shape in self.input_features.items():
            if key.startswith("observation.image"):
                channels, height, width = int(shape[0]), int(shape[1]), int(shape[2])
                if frame_path and Path(frame_path).is_file():
                    image = Image.open(frame_path).convert("RGB").resize(
                        (width, height)
                    )
                    array = np.asarray(image, dtype=np.float32) / 255.0
                else:
                    array = np.zeros((height, width, 3), dtype=np.float32)
                tensor = torch.from_numpy(array).permute(2, 0, 1)[:channels]
                batch[key] = tensor.unsqueeze(0).to(self.device)
            elif key == "observation.state":
                state = build_state_vector(observation, int(shape[0]))
                batch[key] = torch.tensor(
                    [state], dtype=torch.float32, device=self.device
                )
        batch["task"] = [
            _string(observation.get("task_statement"))
            or "Pick up the commanded target item and place it in the goal zone."
        ]
        return batch

    def action_chunk(
        self, observation: Mapping[str, Any], *, chunk_size: int
    ) -> list[dict[str, Any]]:
        import torch

        self.policy.reset()
        batch = self._batch(observation)
        raw_actions: list[list[float]] = []
        with torch.no_grad():
            for _ in range(max(1, int(chunk_size))):
                action = self.policy.select_action(dict(batch))
                raw_actions.append([float(v) for v in action[0].cpu().tolist()])
        chunk: list[dict[str, Any]] = []
        previous: list[float] | None = None
        for raw in raw_actions:
            vector = project_action_to_7d(raw, previous_raw=previous)
            previous = raw
            chunk.append(
                {
                    "action_type": "delta_end_effector_pose",
                    "delta_xyz_m": vector[:3],
                    "delta_rpy_rad": vector[3:6],
                    "gripper_command": vector[6],
                    "action_7d": vector,
                    "raw_model_action": [round(v, 6) for v in raw],
                }
            )
        return chunk


def _claim_boundary() -> dict[str, Any]:
    return {
        "policy_command_ran": True,
        "real_torch_model_inference": True,
        "out_of_distribution_embodiment_mapping": True,
        "action_projection_is_declared_not_semantic": True,
        "single_action_is_not_episode_success": True,
        "task_success_proven": False,
    }


def _response(
    runner: LeRobotTorchPolicyRunner,
    observation: Mapping[str, Any],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    chunk = runner.action_chunk(observation, chunk_size=chunk_size)
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "completed",
        "policy_id": runner.policy_id,
        "policy_type": runner.policy_type,
        "action": chunk[0],
        "action_chunk": chunk,
        "chunk_size": len(chunk),
        "claim_boundary": _claim_boundary(),
    }


def _serve(runner: LeRobotTorchPolicyRunner, *, chunk_size: int) -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = _mapping(json.loads(line))
            observation = _mapping(payload.get("observation")) or payload
            response = _response(runner, observation, chunk_size=chunk_size)
        except Exception as exc:  # pragma: no cover - serve-loop guard
            response = {
                "schema_version": ADAPTER_SCHEMA_VERSION,
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


def _batch_observations(runner: LeRobotTorchPolicyRunner, *, chunk_size: int) -> int:
    manifest_path = os.getenv("BLUEPRINT_POLICY_OBSERVATION_MANIFEST", "").strip()
    output_path = os.getenv("BLUEPRINT_POLICY_EXECUTION_OUTPUT", "").strip()
    if not manifest_path or not output_path:
        print(json.dumps({"status": "blocked", "blockers": ["observation_manifest_or_output_env_missing"]}))
        return 1
    manifest = _mapping(
        json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    )
    observations = [
        _mapping(item)
        for item in manifest.get("observations") or []
        if isinstance(item, Mapping)
    ]
    attempts: list[dict[str, Any]] = []
    for index, observation in enumerate(observations, start=1):
        chunk = runner.action_chunk(observation, chunk_size=chunk_size)
        attempts.append(
            {
                "attempt_id": f"lerobot_torch_policy_attempt_{index:04d}",
                "scenario_eval_run_id": _string(
                    observation.get("scenario_eval_run_id")
                )
                or None,
                "task_id": _string(observation.get("task_id")) or None,
                "scenario_id": _string(observation.get("scenario_id")) or None,
                "policy_id": runner.policy_id,
                "status": "completed",
                "action_source": "learned_policy",
                "action": chunk[0],
                "action_chunk_size": len(chunk),
                "claim_boundary": _claim_boundary(),
            }
        )
    payload = {
        "schema_version": POLICY_EXECUTION_OUTPUT_SCHEMA_VERSION,
        "status": "completed",
        "policy_id": runner.policy_id,
        "policy_type": runner.policy_type,
        "checkpoint": runner.checkpoint,
        "observation_count": len(observations),
        "attempts": attempts,
        "claim_boundary": _claim_boundary(),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "attempt_count": len(attempts)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--batch-observations", action="store_true")
    args = parser.parse_args(argv)

    runner = LeRobotTorchPolicyRunner(checkpoint=args.checkpoint, device=args.device)
    if args.serve:
        return _serve(runner, chunk_size=args.chunk_size)
    if args.batch_observations:
        return _batch_observations(runner, chunk_size=args.chunk_size)

    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    observation = _mapping(_mapping(payload).get("observation")) or _mapping(payload)
    response = _response(runner, observation, chunk_size=args.chunk_size)
    encoded = json.dumps(response, sort_keys=True)
    output_path = os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
