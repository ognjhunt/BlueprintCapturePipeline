"""Fail-closed contracts for the Cosmos3-Nano successor experiment.

This module contains only local, zero-cost preparation.  It does not download
model weights, contact a model API, or allocate provider resources.  Paid
execution must enter through :mod:`blueprint_pipeline.paid_resource_allocator`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ID = "policy_ranking_successor_experiment_20260727"
ACTION_SCHEMA = "cosmos3_droid_action_stream.v1"
REQUEST_SCHEMA = "cosmos3_successor_forward_dynamics_request.v1"

COSMOS_REPOSITORY = "NVIDIA/cosmos"
COSMOS_REVISION = "bebca76311266941d06c5f5572fb601184ba24fa"
COSMOS_FRAMEWORK_REPOSITORY = "NVIDIA/cosmos-framework"
COSMOS_FRAMEWORK_REVISION = "09f23119ea92c707207bba55565e7a09d16896a2"
CHECKPOINT_REPOSITORY = "nvidia/Cosmos3-Nano"
CHECKPOINT_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
VLLM_IMAGE = "docker.io/vllm/vllm-omni:cosmos3"
VLLM_IMAGE_DIGEST = "sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587"
VLLM_AMD64_DIGEST = "sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c"

DROID_HORIZON = 16
DROID_ACTION_DIM = 10
DROID_FREQUENCY_HZ = 15.0
DROID_DOMAIN = "droid_lerobot"
DROID_EMBODIMENT = "franka_panda_droid"
DROID_VIEWPOINT = "concat_view"
DROID_POSE_CONVENTION = "backward_framewise"
DROID_ROTATION = "rot6d"
DROID_TRANSLATION_UNIT = "meter"
DROID_COORDINATE_CONVERSION = "panda_link8_native_rotation_right_multiply_droid_to_opencv"
DROID_GRIPPER_SEMANTICS = "cosmos_open_close_value_after_dataset_version_flip"
DROID_ACTION_NAMES = (
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "rot_4",
    "rot_5",
    "gripper",
)
DROID_TO_OPENCV = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)
DROID_Q01 = (
    -0.014200,
    -0.013416,
    -0.015206,
    0.998459,
    -0.047659,
    -0.034774,
    -0.047609,
    0.998428,
    -0.035553,
    0.0,
)
DROID_Q99 = (
    0.014515,
    0.011517,
    0.014520,
    1.0,
    0.047596,
    0.034660,
    0.047654,
    1.0,
    0.038888,
    1.0,
)

ALLOWED_CONDITIONS = (
    "recorded",
    "zero",
    "shuffled",
    "reversed",
    "policy_swapped",
)
FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "policy",
        "policy_id",
        "policy_name",
        "performance",
        "score",
        "success_rate",
    }
)


class SuccessorContractError(ValueError):
    """A frozen successor-experiment contract was violated."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SuccessorContractError("action_stream_missing_or_not_a_sequence")
    if len(value) != DROID_HORIZON:
        raise SuccessorContractError("action_horizon_must_be_16")
    matrix: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise SuccessorContractError(f"action_row_not_a_sequence:{index}")
        if len(row) != DROID_ACTION_DIM:
            raise SuccessorContractError(f"action_dimension_must_be_10:{index}")
        try:
            numeric = [float(item) for item in row]
        except (TypeError, ValueError) as exc:
            raise SuccessorContractError(f"action_value_not_numeric:{index}") from exc
        if not all(math.isfinite(item) for item in numeric):
            raise SuccessorContractError(f"action_value_not_finite:{index}")
        matrix.append(numeric)
    return matrix


def validate_droid_action_stream(stream: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the exact external action semantics accepted by the frozen arm."""

    if not isinstance(stream, Mapping):
        raise SuccessorContractError("action_stream_missing")
    leaked = sorted(FORBIDDEN_IDENTITY_KEYS.intersection(str(key).lower() for key in stream))
    if leaked:
        raise SuccessorContractError(f"policy_identity_or_performance_leakage:action:{leaked[0]}")
    expected = {
        "schema_version": ACTION_SCHEMA,
        "embodiment": DROID_EMBODIMENT,
        "domain_name": DROID_DOMAIN,
        "pose_convention": DROID_POSE_CONVENTION,
        "rotation_representation": DROID_ROTATION,
        "translation_unit": DROID_TRANSLATION_UNIT,
        "coordinate_conversion": DROID_COORDINATE_CONVERSION,
        "gripper_semantics": DROID_GRIPPER_SEMANTICS,
        "viewpoint": DROID_VIEWPOINT,
        "action_names": list(DROID_ACTION_NAMES),
        "shape": [DROID_HORIZON, DROID_ACTION_DIM],
        "normalization": "raw_external_actions_no_adapter_normalization",
    }
    for key, expected_value in expected.items():
        if stream.get(key) != expected_value:
            raise SuccessorContractError(f"incompatible_action_semantics:{key}")
    try:
        frequency = float(stream.get("frequency_hz"))
    except (TypeError, ValueError) as exc:
        raise SuccessorContractError("action_frequency_missing_or_invalid") from exc
    if not math.isclose(frequency, DROID_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise SuccessorContractError("action_frequency_must_be_15_hz")
    actions = _finite_matrix(stream.get("actions"))
    rot6d = np.asarray(actions, dtype=np.float64)[:, 3:9]
    first_column = rot6d[:, :3]
    second_column = rot6d[:, 3:]
    if not np.allclose(np.linalg.norm(first_column, axis=1), 1.0, rtol=0.0, atol=1e-4):
        raise SuccessorContractError("rot6d_first_column_not_unit")
    if not np.allclose(np.linalg.norm(second_column, axis=1), 1.0, rtol=0.0, atol=1e-4):
        raise SuccessorContractError("rot6d_second_column_not_unit")
    if not np.allclose(np.sum(first_column * second_column, axis=1), 0.0, rtol=0.0, atol=1e-4):
        raise SuccessorContractError("rot6d_columns_not_orthogonal")
    gripper = [row[-1] for row in actions]
    if any(value < 0.0 or value > 1.0 for value in gripper):
        raise SuccessorContractError("gripper_value_outside_closed_unit_interval")
    payload = {**dict(stream), "frequency_hz": frequency, "actions": actions}
    payload["action_sha256"] = canonical_sha256(actions)
    supplied_hash = str(stream.get("action_sha256") or "")
    if supplied_hash and supplied_hash != payload["action_sha256"]:
        raise SuccessorContractError("action_sha256_mismatch")
    return payload


def droid_action_stream(actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Wrap a raw matrix in the frozen DROID external-semantics contract."""

    payload: dict[str, Any] = {
        "schema_version": ACTION_SCHEMA,
        "embodiment": DROID_EMBODIMENT,
        "domain_name": DROID_DOMAIN,
        "frequency_hz": DROID_FREQUENCY_HZ,
        "pose_convention": DROID_POSE_CONVENTION,
        "rotation_representation": DROID_ROTATION,
        "translation_unit": DROID_TRANSLATION_UNIT,
        "coordinate_conversion": DROID_COORDINATE_CONVERSION,
        "gripper_semantics": DROID_GRIPPER_SEMANTICS,
        "viewpoint": DROID_VIEWPOINT,
        "action_names": list(DROID_ACTION_NAMES),
        "shape": [DROID_HORIZON, DROID_ACTION_DIM],
        "normalization": "raw_external_actions_no_adapter_normalization",
        "actions": [list(row) for row in actions],
    }
    return validate_droid_action_stream(payload)


def _euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    """Match SciPy's extrinsic lowercase ``xyz`` rotation convention."""

    # SciPy computes this path in float64 even for float32 inputs and the pinned
    # upstream DROID adapter casts its result to float32.  Preserve that order so
    # local conversion is bit-identical without requiring SciPy at runtime.
    angles = np.asarray(euler_xyz, dtype=np.float32).astype(np.float64)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise SuccessorContractError("droid_euler_state_shape_invalid")
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    matrices = np.empty((len(angles), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = cz * cy
    matrices[:, 0, 1] = cz * sy * sx - sz * cx
    matrices[:, 0, 2] = cz * sy * cx + sz * sx
    matrices[:, 1, 0] = sz * cy
    matrices[:, 1, 1] = sz * sy * sx + cz * cx
    matrices[:, 1, 2] = sz * sy * cx - cz * sx
    matrices[:, 2, 0] = -sy
    matrices[:, 2, 1] = cy * sx
    matrices[:, 2, 2] = cy * cx
    return matrices.astype(np.float32)


def convert_droid_states_to_action_stream(
    cartesian_states: Sequence[Sequence[float]],
    source_gripper_actions: Sequence[float],
    *,
    source_gripper_action_flipped: bool,
) -> dict[str, Any]:
    """Convert DROID absolute state samples to the official raw 10D actions.

    The input contains 17 absolute Panda link-8 states (position in meters and
    Euler ``xyz`` radians) and 16 source gripper commands.  The conversion is
    the local provider-neutral equivalent of the pinned Cosmos Framework DROID
    dataset path: right-multiply rotations by ``_DROID_TO_OPENCV``, compute
    framewise ``T_i^-1 @ T_(i+1)`` deltas, encode the first two rotation-matrix
    columns as rot6d, then apply the dataset-version gripper flip explicitly.
    """

    try:
        states = np.asarray(cartesian_states, dtype=np.float32)
        gripper = np.asarray(source_gripper_actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SuccessorContractError("droid_raw_action_inputs_not_numeric") from exc
    if states.shape != (DROID_HORIZON + 1, 6):
        raise SuccessorContractError("droid_cartesian_states_must_have_shape_17x6")
    if gripper.shape != (DROID_HORIZON,):
        raise SuccessorContractError("droid_gripper_actions_must_have_shape_16")
    if not np.isfinite(states).all() or not np.isfinite(gripper).all():
        raise SuccessorContractError("droid_raw_action_inputs_not_finite")
    if np.any((gripper < 0.0) | (gripper > 1.0)):
        raise SuccessorContractError("source_gripper_value_outside_closed_unit_interval")

    rotations = _euler_xyz_to_matrix(states[:, 3:6])
    rotations = rotations @ np.asarray(DROID_TO_OPENCV, dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(states), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = states[:, :3]
    deltas = np.linalg.inv(poses[:-1]) @ poses[1:]
    translations = deltas[:, :3, 3]
    rot6d = deltas[:, :3, :2].transpose(0, 2, 1).reshape(DROID_HORIZON, 6)
    cosmos_gripper = 1.0 - gripper if source_gripper_action_flipped else gripper
    actions = np.concatenate([translations, rot6d, cosmos_gripper[:, None]], axis=1)
    return droid_action_stream(actions.astype(np.float32).tolist())


def validate_droid_timestamps(timestamps_seconds: Sequence[float]) -> dict[str, Any]:
    """Validate the 17 observation timestamps that bind a 16-action chunk."""

    try:
        timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SuccessorContractError("droid_timestamps_not_numeric") from exc
    if timestamps.shape != (DROID_HORIZON + 1,) or not np.isfinite(timestamps).all():
        raise SuccessorContractError("droid_timestamps_must_have_shape_17_and_be_finite")
    deltas = np.diff(timestamps)
    expected = 1.0 / DROID_FREQUENCY_HZ
    if not np.allclose(deltas, expected, rtol=0.0, atol=2e-4):
        raise SuccessorContractError("droid_timestamp_alignment_outside_2e_4_seconds")
    return {
        "frequency_hz": DROID_FREQUENCY_HZ,
        "observation_count": DROID_HORIZON + 1,
        "action_count": DROID_HORIZON,
        "maximum_absolute_step_error_seconds": float(np.max(np.abs(deltas - expected))),
        "status": "passed",
    }


def validate_droid_camera_metadata(cameras: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the frozen three-camera ordering before multiview composition."""

    expected_order = (
        "wrist_image_left",
        "exterior_image_1_left",
        "exterior_image_2_left",
    )
    if tuple(cameras) != expected_order:
        raise SuccessorContractError("droid_camera_order_invalid")
    for name in expected_order:
        metadata = cameras[name]
        if metadata.get("shape") != [360, 640, 3]:
            raise SuccessorContractError(f"droid_camera_resolution_invalid:{name}")
        if metadata.get("fps") != 15:
            raise SuccessorContractError(f"droid_camera_frequency_invalid:{name}")
    return {
        "status": "passed",
        "input_order": list(expected_order),
        "composition": "wrist_full_width_top; exterior_1_half_width_bottom_left; exterior_2_half_width_bottom_right",
        "output_shape": [540, 640, 3],
    }


def build_action_controls(
    recorded: Mapping[str, Any],
    policy_swapped: Mapping[str, Any],
    *,
    observation_gripper_hold: float,
    shuffle_seed: int,
) -> dict[str, dict[str, Any]]:
    """Build all smoke-canary controls while preserving action identity."""

    source = validate_droid_action_stream(recorded)
    swapped = validate_droid_action_stream(policy_swapped)
    if source["action_sha256"] == swapped["action_sha256"]:
        raise SuccessorContractError("policy_swapped_action_must_be_distinct")
    hold = float(observation_gripper_hold)
    if not math.isfinite(hold) or hold < 0.0 or hold > 1.0:
        raise SuccessorContractError("observation_gripper_hold_outside_closed_unit_interval")
    actions = source["actions"]
    order = list(range(DROID_HORIZON))
    random.Random(int(shuffle_seed)).shuffle(order)
    if order == list(range(DROID_HORIZON)) or order == list(reversed(range(DROID_HORIZON))):
        order = order[1:] + order[:1]
    controls = {
        "recorded": droid_action_stream(actions),
        "zero": droid_action_stream(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, hold] for _ in actions]
        ),
        "shuffled": droid_action_stream([actions[index] for index in order]),
        "reversed": droid_action_stream(list(reversed(actions))),
        "policy_swapped": droid_action_stream(swapped["actions"]),
    }
    hashes = [controls[name]["action_sha256"] for name in ALLOWED_CONDITIONS]
    if len(set(hashes)) != len(hashes):
        raise SuccessorContractError("action_controls_not_pairwise_distinct")
    return controls


def _reject_identity_leakage(value: Mapping[str, Any], *, location: str) -> None:
    leaked = sorted(FORBIDDEN_IDENTITY_KEYS.intersection(str(key).lower() for key in value))
    if leaked:
        raise SuccessorContractError(
            f"policy_identity_or_performance_leakage:{location}:{leaked[0]}"
        )


def build_forward_dynamics_request(
    *,
    initial_observation_sha256: str,
    task_instruction: str,
    action_stream: Mapping[str, Any],
    condition: str,
    seed: int,
) -> dict[str, Any]:
    """Build a policy-blind request for the official action-conditioned path."""

    if condition not in ALLOWED_CONDITIONS:
        raise SuccessorContractError("unknown_action_condition")
    observation_hash = str(initial_observation_sha256).lower()
    if len(observation_hash) != 64 or any(
        char not in "0123456789abcdef" for char in observation_hash
    ):
        raise SuccessorContractError("initial_observation_sha256_invalid")
    instruction = str(task_instruction).strip()
    if not instruction:
        raise SuccessorContractError("task_instruction_missing")
    action = validate_droid_action_stream(action_stream)
    request_material = {
        "initial_observation_sha256": observation_hash,
        "task_instruction": instruction,
        "action_sha256": action["action_sha256"],
        "seed": int(seed),
        "checkpoint_revision": CHECKPOINT_REVISION,
    }
    request_hash = canonical_sha256(request_material)
    request = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": request_hash,
        "name": request_hash,
        "initial_observation_sha256": observation_hash,
        "prompt": instruction,
        "action": action,
        "action_condition": condition,
        "seed": int(seed),
        "runtime": {
            "model_mode": "forward_dynamics",
            "action_mode": "forward_dynamics",
            "domain_name": DROID_DOMAIN,
            "action_chunk_size": DROID_HORIZON,
            "num_frames": DROID_HORIZON + 1,
            "fps": int(DROID_FREQUENCY_HZ),
            "image_size": 480,
            "view_point": DROID_VIEWPOINT,
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "flow_shift": 10.0,
            "precision": "bf16",
            "guardrails": "required_unless_gated_guardrail_access_is_unavailable_and_exception_is_recorded",
        },
        "source_lock": {
            "cosmos_repository": COSMOS_REPOSITORY,
            "cosmos_revision": COSMOS_REVISION,
            "framework_repository": COSMOS_FRAMEWORK_REPOSITORY,
            "framework_revision": COSMOS_FRAMEWORK_REVISION,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "vllm_image": f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}",
            "vllm_amd64_manifest_digest": VLLM_AMD64_DIGEST,
            "trust_remote_code": False,
        },
        "claim_boundary": {
            "implementation_only": True,
            "runtime_proven": False,
            "generated_media_proven": False,
            "wam_causal_validity_proven": False,
        },
    }
    _reject_identity_leakage(request, location="request")
    _reject_identity_leakage(request["runtime"], location="runtime")
    if request["name"] != request["request_id"] or instruction != request["prompt"]:
        raise SuccessorContractError("request_identity_or_prompt_not_frozen")
    return request


def validate_smoke_inventory_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the frozen 5-condition x 2-seed request inventory."""

    if manifest.get("schema_version") != "policy_ranking_successor_smoke_request_inventory.v1":
        raise SuccessorContractError("smoke_inventory_schema_invalid")
    observation_hash = str(manifest.get("initial_observation_sha256") or "").lower()
    instruction = str(manifest.get("task_instruction") or "").strip()
    actions = manifest.get("action_hashes")
    requests = manifest.get("requests")
    if not isinstance(actions, Mapping) or tuple(actions) != ALLOWED_CONDITIONS:
        raise SuccessorContractError("smoke_inventory_action_conditions_invalid")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes, bytearray)):
        raise SuccessorContractError("smoke_inventory_requests_invalid")
    expected_pairs = {(condition, seed) for condition in ALLOWED_CONDITIONS for seed in (0, 1)}
    observed_pairs: set[tuple[str, int]] = set()
    inventory_rows: list[dict[str, Any]] = []
    for row in requests:
        if not isinstance(row, Mapping):
            raise SuccessorContractError("smoke_inventory_request_row_invalid")
        condition = str(row.get("condition") or "")
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError) as exc:
            raise SuccessorContractError("smoke_inventory_seed_invalid") from exc
        pair = (condition, seed)
        if pair not in expected_pairs or pair in observed_pairs:
            raise SuccessorContractError("smoke_inventory_condition_seed_matrix_invalid")
        observed_pairs.add(pair)
        action_hash = str(actions.get(condition) or "")
        if row.get("action_sha256") != action_hash:
            raise SuccessorContractError("smoke_inventory_action_hash_binding_invalid")
        request_material = {
            "initial_observation_sha256": observation_hash,
            "task_instruction": instruction,
            "action_sha256": action_hash,
            "seed": seed,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }
        if row.get("request_id") != canonical_sha256(request_material):
            raise SuccessorContractError("smoke_inventory_request_id_invalid")
        inventory_rows.append(
            {
                "request_id": row["request_id"],
                "condition": condition,
                "seed": seed,
                "action_sha256": action_hash,
                "observation_sha256": observation_hash,
            }
        )
    if observed_pairs != expected_pairs or len(inventory_rows) != 10:
        raise SuccessorContractError("smoke_inventory_incomplete")
    digest = canonical_sha256(inventory_rows)
    if manifest.get("inventory_sha256") != digest:
        raise SuccessorContractError("smoke_inventory_sha256_invalid")
    return {"status": "passed", "request_count": 10, "inventory_sha256": digest}


def assert_evaluator_eligible(rollout: Mapping[str, Any]) -> None:
    """Prevent any evaluator from scoring a causally invalid WAM rollout."""

    causal = rollout.get("causal_validity")
    if not isinstance(causal, Mapping) or causal.get("status") != "valid":
        raise SuccessorContractError("evaluator_rejects_causally_invalid_rollout")
    if rollout.get("generated_media_valid") is not True:
        raise SuccessorContractError("evaluator_rejects_invalid_generated_media")


@dataclass(frozen=True)
class GPUOffer:
    offer_id: str
    gpu_model: str
    hourly_price_usd: float
    benchmark_seconds_per_rollout: float
    setup_hours: float
    stack_preflight_passed: bool
    expected_validity_rate: float = 1.0

    def projected_all_in_cost_per_valid_rollout(self, rollout_count: int) -> float:
        if rollout_count <= 0:
            raise SuccessorContractError("rollout_count_must_be_positive")
        if not 0.0 < self.expected_validity_rate <= 1.0:
            raise SuccessorContractError("expected_validity_rate_invalid")
        price = float(self.hourly_price_usd)
        seconds = float(self.benchmark_seconds_per_rollout)
        setup = float(self.setup_hours)
        if not all(math.isfinite(value) and value >= 0.0 for value in (price, seconds, setup)):
            raise SuccessorContractError("gpu_projection_input_invalid")
        total = price * (setup + rollout_count * seconds / 3600.0)
        expected_valid = rollout_count * self.expected_validity_rate
        return total / expected_valid


def select_gpu_offer(
    offers: Sequence[GPUOffer],
    *,
    rollout_count: int,
    reasonably_close_fraction: float = 0.15,
) -> dict[str, Any]:
    """Apply the frozen H100-versus-Blackwell selection rule.

    H100 SXM 80GB is preferred when its all-in projected cost per valid rollout
    is within ``reasonably_close_fraction`` of an admitted RTX PRO 6000
    Blackwell offer.  Blackwell can win only after exact-stack preflight.
    """

    h100 = [offer for offer in offers if offer.gpu_model == "H100 SXM 80GB"]
    blackwell = [offer for offer in offers if offer.gpu_model == "RTX PRO 6000 Blackwell"]
    admitted_h100 = [offer for offer in h100 if offer.stack_preflight_passed]
    admitted_blackwell = [offer for offer in blackwell if offer.stack_preflight_passed]
    if not admitted_h100:
        raise SuccessorContractError("h100_sxm_exact_stack_preflight_missing")
    h100_best = min(
        admitted_h100,
        key=lambda item: item.projected_all_in_cost_per_valid_rollout(rollout_count),
    )
    h100_cost = h100_best.projected_all_in_cost_per_valid_rollout(rollout_count)
    if not admitted_blackwell:
        selected = h100_best
        reason = "h100_default_blackwell_exact_stack_not_admitted"
    else:
        blackwell_best = min(
            admitted_blackwell,
            key=lambda item: item.projected_all_in_cost_per_valid_rollout(rollout_count),
        )
        blackwell_cost = blackwell_best.projected_all_in_cost_per_valid_rollout(rollout_count)
        if h100_cost <= blackwell_cost * (1.0 + reasonably_close_fraction):
            selected = h100_best
            reason = "h100_preferred_prices_reasonably_close"
        else:
            selected = blackwell_best
            reason = "blackwell_exact_stack_passed_and_lower_all_in_cost_per_valid_rollout"
    return {
        "selected_offer_id": selected.offer_id,
        "selected_gpu_model": selected.gpu_model,
        "selection_reason": reason,
        "projected_all_in_cost_per_valid_rollout_usd": selected.projected_all_in_cost_per_valid_rollout(
            rollout_count
        ),
        "reasonably_close_fraction": reasonably_close_fraction,
        "rollout_count": rollout_count,
    }


def validate_compute_admission(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed until an explicit compute cap and safety controls exist."""

    blockers: list[str] = []
    if plan.get("allocator_entrypoint") != "python -m blueprint_pipeline.paid_resource_allocator":
        blockers.append("canonical_paid_resource_allocator_required")
    try:
        cap = float(plan.get("authorized_compute_cap_usd"))
    except (TypeError, ValueError):
        cap = math.nan
    try:
        projected = float(plan.get("projected_compute_spend_usd"))
    except (TypeError, ValueError):
        projected = math.nan
    if not math.isfinite(cap) or cap <= 0.0:
        blockers.append("explicit_compute_cap_not_authorized")
    if not math.isfinite(projected) or projected < 0.0:
        blockers.append("projected_compute_spend_invalid")
    elif math.isfinite(cap) and projected > cap:
        blockers.append("projected_compute_spend_exceeds_authorized_cap")
    if not isinstance(plan.get("hard_ttl_seconds"), int) or int(plan["hard_ttl_seconds"]) <= 0:
        blockers.append("hard_ttl_missing_or_invalid")
    if plan.get("watchdog_enabled") is not True:
        blockers.append("watchdog_not_enabled")
    if plan.get("automatic_spend_cutoff") is not True:
        blockers.append("automatic_spend_cutoff_not_enabled")
    if plan.get("teardown_required") is not True:
        blockers.append("teardown_not_required")
    if plan.get("provider_zero_verification_required") is not True:
        blockers.append("provider_zero_verification_not_required")
    return {
        "schema_version": "policy_ranking_successor_compute_admission.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "allocation_authorized": not blockers,
    }


def artifact_claims(*, implementation: bool = False) -> dict[str, bool]:
    """Uniform proof-boundary block required on every successor artifact."""

    return {
        "implementation": implementation,
        "runtime": False,
        "generated_media": False,
        "wam_causal_validity": False,
        "evaluator_validity": False,
        "simulator_outcomes": False,
        "ranking_fidelity": False,
        "captured_site_portability": False,
        "warehouse_portability": False,
        "economic_comparison": False,
        "physical_performance": False,
    }


def validate_artifact_path(path: str | Path, experiment_root: str | Path) -> Path:
    root = Path(experiment_root).resolve()
    candidate = Path(path).resolve()
    if candidate != root and root not in candidate.parents:
        raise SuccessorContractError("artifact_path_outside_successor_namespace")
    return candidate
