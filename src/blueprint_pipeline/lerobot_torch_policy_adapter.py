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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ADAPTER_SCHEMA_VERSION = "blueprint_lerobot_torch_policy_adapter.v1"
GPU_RUNTIME_CONTRACT_SCHEMA_VERSION = "blueprint_lerobot_torch_gpu_runtime_contract.v1"
POLICY_EXECUTION_OUTPUT_SCHEMA_VERSION = (
    "blueprint_lerobot_torch_policy_execution_output.v1"
)
GROOT_LIBERO_CHECKPOINT_REPO_ID = "nvidia/gr00t17-lerobot-libero_10-640"
GROOT_LIBERO_INTEGRATION_LABEL = "libero_panda_groot_integration_proof"
LIBERO_PANDA_EMBODIMENT_TAG = "libero_sim"
LIBERO_ACTION_DECODE_TRANSFORM = "libero"
LIBERO_VISUAL_FEATURE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
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


def visual_feature_layout(shape: Sequence[Any]) -> dict[str, Any]:
    """Normalize LeRobot visual feature shapes into a CHW tensor layout.

    LeRobot checkpoints are not consistent about whether image features are
    declared as CHW or HWC. The GR00T N1.7 LIBERO checkpoint declares
    ``(256, 256, 3)``; the torch batch must still be ``C,H,W``.
    """
    dims = [int(v) for v in shape[:3]] if len(shape) >= 3 else []
    if len(dims) != 3:
        return {
            "input_shape": list(shape),
            "input_layout": "unknown",
            "channels": 3,
            "height": 256,
            "width": 256,
            "tensor_layout": "CHW",
        }
    first, second, third = dims
    if first in {1, 3, 4} and third not in {1, 3, 4}:
        channels, height, width = first, second, third
        input_layout = "CHW"
    elif third in {1, 3, 4}:
        height, width, channels = first, second, third
        input_layout = "HWC"
    else:
        channels, height, width = first, second, third
        input_layout = "CHW_assumed"
    return {
        "input_shape": dims,
        "input_layout": input_layout,
        "channels": channels,
        "height": height,
        "width": width,
        "tensor_layout": "CHW",
    }


def _path_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("path", "camera_frame_path", "frame_path", "file", "uri"):
            text = _string(value.get(key))
            if text:
                return text
        return ""
    return _string(value)


def _nested_mappings(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(dict(value))
    return rows


def _visual_observation_maps(observation: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    visual = _mapping(observation.get("visual_observation"))
    maps: list[tuple[str, dict[str, Any]]] = []
    for name, value in (
        ("visual_observation.camera_frame_paths", visual.get("camera_frame_paths")),
        ("visual_observation.camera_frames", visual.get("camera_frames")),
        ("visual_observation.images", visual.get("images")),
        ("observation.images", observation.get("images")),
        ("top_level_camera_frame_paths", observation.get("camera_frame_paths")),
        ("top_level_camera_frames", observation.get("camera_frames")),
    ):
        for mapping in _nested_mappings(value):
            maps.append((name, mapping))
    return maps


def _visual_feature_aliases(feature_key: str) -> list[str]:
    suffix = feature_key.rsplit(".", 1)[-1]
    aliases = [feature_key, suffix]
    if suffix == "wrist_image":
        aliases.extend(
            [
                "wrist",
                "wrist_camera",
                "wrist_rgb",
                "wrist_image",
                "wrist_camera_frame_path",
                "wrist_frame_path",
            ]
        )
    elif suffix == "image":
        aliases.extend(
            [
                "image",
                "front",
                "front_camera",
                "scene",
                "scene_camera",
                "camera_frame_path",
                "main_camera_frame_path",
            ]
        )
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _single_frame_fallback(observation: Mapping[str, Any]) -> tuple[str, str]:
    visual = _mapping(observation.get("visual_observation"))
    for source, value in (
        ("visual_observation.camera_frame_path", visual.get("camera_frame_path")),
        ("top_level_camera_frame_path", observation.get("camera_frame_path")),
    ):
        text = _path_text(value)
        if text:
            return text, source
    return "", ""


def resolve_visual_feature_bindings(
    observation: Mapping[str, Any],
    feature_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Resolve policy visual feature names to Blueprint observation frames."""
    rows: list[dict[str, Any]] = []
    fallback_path, fallback_source = _single_frame_fallback(observation)
    maps = _visual_observation_maps(observation)
    for feature_key in feature_keys:
        selected_path = ""
        selected_source = ""
        selected_alias = ""
        for source, mapping in maps:
            for alias in _visual_feature_aliases(feature_key):
                if alias in mapping:
                    selected_path = _path_text(mapping.get(alias))
                    selected_source = source
                    selected_alias = alias
                    break
            if selected_path:
                break
        used_single_frame_fallback = False
        if not selected_path and fallback_path:
            selected_path = fallback_path
            selected_source = fallback_source
            selected_alias = "single_frame_fallback"
            used_single_frame_fallback = True
        path = Path(selected_path).expanduser() if selected_path else None
        rows.append(
            {
                "feature_key": feature_key,
                "source_path": selected_path or None,
                "source_mapping": selected_source or None,
                "source_alias": selected_alias or None,
                "available": bool(path and path.is_file()),
                "used_single_frame_fallback": used_single_frame_fallback,
            }
        )
    counts = Counter(
        row["source_path"] for row in rows if _string(row.get("source_path"))
    )
    for row in rows:
        source_path = _string(row.get("source_path"))
        row["shared_source_path_with_other_visual_feature"] = (
            bool(source_path) and counts[source_path] > 1
        )
    return rows


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
    action_semantics: Mapping[str, Any] | None = None,
    max_step_m: float = 0.02,
) -> list[float]:
    """Declared OOD projection: checkpoint-native action -> 7d delta-EE.

    For generic learned policies, first three dims are deltas between
    consecutive model outputs. For GR00T/LIBERO, the checkpoint is explicitly
    labeled ``libero_sim`` and ``action_decode_transform=libero``; its 7D output
    is treated as LIBERO/Panda-native and projected directly into a bounded
    Blueprint delta-EE control for workflow plumbing only.
    """
    raw = [float(v) for v in raw_action]
    semantics = _mapping(action_semantics)
    projection_mode = _string(semantics.get("projection_mode"))
    if projection_mode == "libero_panda_direct_7d_to_blueprint_delta_ee":
        deltas = list(raw)
    elif semantics.get("source_action_is_relative") is True:
        deltas = list(raw)
    else:
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


def action_semantics_contract(
    *,
    policy_type: str,
    action_decode_transform: str | None,
    embodiment_tag: str | None,
    use_relative_actions: bool | None,
) -> dict[str, Any]:
    is_libero = (
        _string(policy_type) == "groot"
        and (
            _string(action_decode_transform) == LIBERO_ACTION_DECODE_TRANSFORM
            or _string(embodiment_tag) == LIBERO_PANDA_EMBODIMENT_TAG
        )
    )
    if is_libero:
        return {
            "integration_label": GROOT_LIBERO_INTEGRATION_LABEL,
            "source_action_semantics": "libero_panda_7d_action",
            "source_embodiment_tag": _string(embodiment_tag) or LIBERO_PANDA_EMBODIMENT_TAG,
            "action_decode_transform": _string(action_decode_transform)
            or LIBERO_ACTION_DECODE_TRANSFORM,
            "source_action_is_relative": bool(use_relative_actions),
            "projection_mode": "libero_panda_direct_7d_to_blueprint_delta_ee",
            "blueprint_action_semantics": "sc3_7d_delta_end_effector_pose",
            "projection_is_semantic_compatibility_claim": False,
            "panda_compatible_task_evaluator_required_for_quality_claim": True,
            "meaningful_manipulator_scoring_requires": (
                "libero_panda_simulator_bridge_or_panda_task_evaluator"
            ),
        }
    return {
        "integration_label": "generic_lerobot_torch_policy_integration",
        "source_action_semantics": "checkpoint_native_action",
        "source_embodiment_tag": _string(embodiment_tag) or None,
        "action_decode_transform": _string(action_decode_transform) or None,
        "source_action_is_relative": bool(use_relative_actions),
        "projection_mode": (
            "relative_7d_to_blueprint_delta_ee"
            if use_relative_actions
            else "successive_output_delta_to_blueprint_delta_ee"
        ),
        "blueprint_action_semantics": "sc3_7d_delta_end_effector_pose",
        "projection_is_semantic_compatibility_claim": False,
        "panda_compatible_task_evaluator_required_for_quality_claim": False,
        "meaningful_manipulator_scoring_requires": None,
    }


def build_gpu_runtime_contract(
    *,
    checkpoint: str,
    device: str,
    policy_type: str | None = None,
) -> dict[str, Any]:
    checkpoint_text = _string(checkpoint)
    policy_text = _string(policy_type)
    exact_libero = checkpoint_text == GROOT_LIBERO_CHECKPOINT_REPO_ID
    is_groot = policy_text == "groot" or "gr00t" in checkpoint_text.lower() or "groot" in checkpoint_text.lower()
    return {
        "requires_gpu_runtime": bool(is_groot or device.startswith("cuda")),
        "device": device,
        "recommended_device": "cuda" if is_groot else device,
        "adapter_mode": "persistent_stdio_adapter_recommended",
        "model_load_once_for_closed_loop": True,
        "python_package_extra": "lerobot[groot]" if is_groot else "lerobot",
        "checkpoint_repo_id": checkpoint_text,
        "checkpoint_size_class": "large_12gb_plus" if exact_libero else "checkpoint_dependent",
        "model_card_size_gb_approx": 12.6 if exact_libero else None,
        "exact_groot_libero_checkpoint": exact_libero,
        "not_cpu_baseline_lane": bool(is_groot),
    }


def runtime_contract_payload(
    *,
    checkpoint: str,
    device: str,
    policy_type: str | None = None,
) -> dict[str, Any]:
    effective_policy_type = _string(policy_type)
    if not effective_policy_type and _string(checkpoint) == GROOT_LIBERO_CHECKPOINT_REPO_ID:
        effective_policy_type = "groot"
    semantics = action_semantics_contract(
        policy_type=effective_policy_type,
        action_decode_transform=(
            LIBERO_ACTION_DECODE_TRANSFORM
            if _string(checkpoint) == GROOT_LIBERO_CHECKPOINT_REPO_ID
            else None
        ),
        embodiment_tag=(
            LIBERO_PANDA_EMBODIMENT_TAG
            if _string(checkpoint) == GROOT_LIBERO_CHECKPOINT_REPO_ID
            else None
        ),
        use_relative_actions=False,
    )
    return {
        "schema_version": GPU_RUNTIME_CONTRACT_SCHEMA_VERSION,
        "status": "configured",
        "checkpoint": checkpoint,
        "policy_type": effective_policy_type or None,
        "integration_label": semantics["integration_label"],
        "action_semantics": semantics,
        "gpu_runtime_contract": build_gpu_runtime_contract(
            checkpoint=checkpoint,
            device=device,
            policy_type=effective_policy_type,
        ),
        "claim_boundary": _claim_boundary(
            semantics,
            policy_command_ran=False,
            real_torch_model_inference=False,
        ),
    }


class LeRobotTorchPolicyRunner:
    """Loads a real LeRobot checkpoint once and serves action chunks."""

    def __init__(self, *, checkpoint: str, device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.policy_id = f"lerobot_torch::{checkpoint}"
        self.policy_type = ""
        self.input_features: dict[str, tuple[Any, ...]] = {}
        self.output_features: dict[str, Any] = {}
        self.action_dim = 7
        self.action_decode_transform: str | None = None
        self.embodiment_tag: str | None = None
        self.use_relative_actions: bool | None = None
        self.last_visual_feature_bindings: list[dict[str, Any]] = []
        self.last_visual_feature_layouts: dict[str, dict[str, Any]] = {}
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
        self.action_decode_transform = _string(
            getattr(config, "action_decode_transform", None)
        ) or None
        self.embodiment_tag = _string(getattr(config, "embodiment_tag", None)) or None
        if hasattr(config, "use_relative_actions"):
            self.use_relative_actions = bool(getattr(config, "use_relative_actions"))
        output_features = config.output_features or {}
        self.output_features = dict(output_features)
        action_feature = output_features.get("action")
        self.action_dim = int(action_feature.shape[0]) if action_feature else 7

    @property
    def action_semantics(self) -> dict[str, Any]:
        return action_semantics_contract(
            policy_type=self.policy_type,
            action_decode_transform=self.action_decode_transform,
            embodiment_tag=self.embodiment_tag,
            use_relative_actions=self.use_relative_actions,
        )

    @property
    def gpu_runtime_contract(self) -> dict[str, Any]:
        return build_gpu_runtime_contract(
            checkpoint=self.checkpoint,
            device=self.device,
            policy_type=self.policy_type,
        )

    def _batch(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from PIL import Image
        import numpy as np

        batch: dict[str, Any] = {}
        visual_keys = [
            key for key in self.input_features if key.startswith("observation.image")
        ]
        bindings = resolve_visual_feature_bindings(observation, visual_keys)
        bindings_by_key = {row["feature_key"]: row for row in bindings}
        self.last_visual_feature_bindings = bindings
        self.last_visual_feature_layouts = {}
        for key, shape in self.input_features.items():
            if key.startswith("observation.image"):
                layout = visual_feature_layout(shape)
                self.last_visual_feature_layouts[key] = layout
                channels = int(layout["channels"])
                height = int(layout["height"])
                width = int(layout["width"])
                binding = _mapping(bindings_by_key.get(key))
                frame_path = _string(binding.get("source_path"))
                if binding.get("available") is True and frame_path:
                    image = Image.open(frame_path).convert("RGB").resize((width, height))
                    array = np.asarray(image, dtype=np.float32) / 255.0
                else:
                    array = np.zeros((height, width, 3), dtype=np.float32)
                tensor = torch.from_numpy(array).permute(2, 0, 1)[:channels]
                if int(tensor.shape[0]) < channels:
                    pad = torch.zeros(
                        (channels - int(tensor.shape[0]), height, width),
                        dtype=tensor.dtype,
                    )
                    tensor = torch.cat([tensor, pad], dim=0)
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
        semantics = self.action_semantics
        with torch.no_grad():
            for _ in range(max(1, int(chunk_size))):
                action = self.policy.select_action(dict(batch))
                raw_actions.append([float(v) for v in action[0].cpu().tolist()])
        chunk: list[dict[str, Any]] = []
        previous: list[float] | None = None
        for raw in raw_actions:
            vector = project_action_to_7d(
                raw,
                previous_raw=previous,
                action_semantics=semantics,
            )
            previous = raw
            chunk.append(
                {
                    "action_type": "delta_end_effector_pose",
                    "delta_xyz_m": vector[:3],
                    "delta_rpy_rad": vector[3:6],
                    "gripper_command": vector[6],
                    "action_7d": vector,
                    "source_action_semantics": semantics["source_action_semantics"],
                    "blueprint_action_semantics": semantics["blueprint_action_semantics"],
                    "action_projection_mode": semantics["projection_mode"],
                    "action_projection_is_semantic_compatibility_claim": False,
                    "raw_model_action": [round(v, 6) for v in raw],
                }
            )
        return chunk


def _claim_boundary(
    action_semantics: Mapping[str, Any] | None = None,
    *,
    policy_command_ran: bool = True,
    real_torch_model_inference: bool = True,
) -> dict[str, Any]:
    semantics = _mapping(action_semantics)
    libero = _string(semantics.get("integration_label")) == GROOT_LIBERO_INTEGRATION_LABEL
    return {
        "policy_command_ran": bool(policy_command_ran),
        "real_torch_model_inference": bool(real_torch_model_inference),
        "out_of_distribution_embodiment_mapping": True,
        "action_projection_is_declared_not_semantic": True,
        "libero_panda_groot_integration_proof_only": bool(libero),
        "panda_or_libero_task_success_proven": False,
        "meaningful_manipulator_scoring_proven": False,
        "blueprint_site_task_success_proven": False,
        "humanoid_readiness_proven": False,
        "physical_robot_readiness_proven": False,
        "buyer_facing_deployment_claim_allowed": False,
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
    semantics = runner.action_semantics
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": "completed",
        "policy_id": runner.policy_id,
        "policy_type": runner.policy_type,
        "integration_label": semantics["integration_label"],
        "action_semantics": semantics,
        "gpu_runtime_contract": runner.gpu_runtime_contract,
        "visual_feature_bindings": runner.last_visual_feature_bindings,
        "visual_feature_layouts": runner.last_visual_feature_layouts,
        "action": chunk[0],
        "action_chunk": chunk,
        "chunk_size": len(chunk),
        "claim_boundary": _claim_boundary(semantics),
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
                "action_semantics": runner.action_semantics,
                "claim_boundary": _claim_boundary(runner.action_semantics),
            }
        )
    payload = {
        "schema_version": POLICY_EXECUTION_OUTPUT_SCHEMA_VERSION,
        "status": "completed",
        "policy_id": runner.policy_id,
        "policy_type": runner.policy_type,
        "integration_label": runner.action_semantics["integration_label"],
        "checkpoint": runner.checkpoint,
        "observation_count": len(observations),
        "attempts": attempts,
        "gpu_runtime_contract": runner.gpu_runtime_contract,
        "claim_boundary": _claim_boundary(runner.action_semantics),
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
    parser.add_argument("--policy-type")
    parser.add_argument("--print-runtime-contract", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--batch-observations", action="store_true")
    args = parser.parse_args(argv)

    if args.print_runtime_contract:
        print(
            json.dumps(
                runtime_contract_payload(
                    checkpoint=args.checkpoint,
                    device=args.device,
                    policy_type=args.policy_type,
                ),
                sort_keys=True,
            )
        )
        return 0

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
