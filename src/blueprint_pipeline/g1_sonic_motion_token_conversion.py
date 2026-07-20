"""Convert G1 joint demonstrations into GR00T/SONIC training actions.

The public microwave demonstration candidate records 43 G1 joint targets:
29 body joints and two seven-joint hands.  ``UNITREE_G1_SONIC`` instead trains
on a 78D action: the official GEAR-SONIC encoder's 64D motion token followed by
the two seven-joint hand targets.

This module implements that mechanical conversion without upgrading its proof
boundary.  In particular, the source candidate does not record root pose.  A
caller must explicitly attest that a trajectory is fixed-base and upright
before the identity root-anchor observation can be used.  Converted actions
are training inputs only; they are not a checkpoint or task qualification.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "g1_sonic_motion_token_conversion.v2"
TRAINING_SOURCE_AUDIT_SCHEMA_VERSION = "g1_sonic_training_source_audit.v1"
SOURCE_DATASET_REPO = "niravpanchalmerai/dtwin_g1_microwave_bowl"
SOURCE_DATASET_REVISION = "ae1947850332eb3ad9c4a801f429d2fb27e750ad"
SOURCE_TASK_DESCRIPTION = (
    "Open the microwave door with the left hand, place the can inside with the "
    "right hand, then close the door with the left hand."
)
GEAR_SONIC_ENCODER_REPO = "nvidia/GEAR-SONIC"
GEAR_SONIC_ENCODER_REVISION = "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
GEAR_SONIC_ENCODER_PATH = "model_encoder.onnx"
GEAR_SONIC_ENCODER_SHA256 = (
    "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
)
ISAAC_GROOT_REPO = "NVIDIA/Isaac-GR00T"
ISAAC_GROOT_REVISION = "9c7e746b2cd37a810070a98ef41d290a07e806c2"

SOURCE_FPS = 50.0
ENCODER_INPUT_DIM = 1762
MOTION_TOKEN_DIM = 64
HAND_DIM = 7
SONIC_ACTION_DIM = MOTION_TOKEN_DIM + HAND_DIM + HAND_DIM
FUTURE_FRAME_COUNT = 10
FUTURE_FRAME_STRIDE = 5
BODY_POSITION_SLICE = slice(4, 294)
BODY_VELOCITY_SLICE = slice(294, 584)
ROOT_ANCHOR_ROTATION_SLICE = slice(601, 661)
IDENTITY_ROTATION_6D = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
FIXED_UPRIGHT_PROJECTED_GRAVITY = (0.0, 0.0, -1.0)
APPROVED_TRAINING_LICENSES = frozenset(
    {"apache-2.0", "cc-by-4.0", "cc0-1.0", "bsd-3-clause", "mit"}
)

SOURCE_ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

ISAACLAB_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

LEFT_HAND_JOINT_NAMES = SOURCE_ACTION_JOINT_NAMES[22:29]
RIGHT_HAND_JOINT_NAMES = SOURCE_ACTION_JOINT_NAMES[36:43]


def audit_training_source_metadata(
    *,
    info: Mapping[str, Any],
    modality: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    declared_license: str | None,
    fixed_base_upright_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed before materializing the public candidate for fine-tuning."""

    features = info.get("features")
    feature_map = dict(features) if isinstance(features, Mapping) else {}
    action_feature = feature_map.get("action")
    action_contract = (
        dict(action_feature) if isinstance(action_feature, Mapping) else {}
    )
    state_feature = feature_map.get("observation.state")
    state_contract = (
        dict(state_feature) if isinstance(state_feature, Mapping) else {}
    )
    video_feature = feature_map.get("observation.images.ego_view")
    video_contract = (
        dict(video_feature) if isinstance(video_feature, Mapping) else {}
    )
    video_info = video_contract.get("video_info")
    video_details = dict(video_info) if isinstance(video_info, Mapping) else {}
    episode_rows = [dict(row) for row in episodes if isinstance(row, Mapping)]
    task_rows = [dict(row) for row in tasks if isinstance(row, Mapping)]
    episode_lengths: list[int] = []
    episode_indices: list[int] = []
    episodes_match_task = True
    try:
        for row in episode_rows:
            episode_indices.append(int(row.get("episode_index")))
            episode_lengths.append(int(row.get("length")))
            episodes_match_task = episodes_match_task and row.get("tasks") == [
                SOURCE_TASK_DESCRIPTION
            ]
    except (TypeError, ValueError):
        episode_indices = []
        episode_lengths = []
        episodes_match_task = False

    source_modality = dict(modality)
    source_state = source_modality.get("state")
    source_action = source_modality.get("action")
    source_video = source_modality.get("video")
    source_annotation = source_modality.get("annotation")
    total_episodes = int(info.get("total_episodes") or 0)
    total_frames = int(info.get("total_frames") or 0)
    metadata_checks = {
        "codebase_version_exact": info.get("codebase_version") == "v2.1",
        "robot_type_exact": info.get("robot_type") == "unitree_g1",
        "episode_inventory_exact": bool(
            total_episodes == 200
            and len(episode_rows) == total_episodes
            and episode_indices == list(range(total_episodes))
        ),
        "frame_inventory_exact": bool(
            episode_lengths
            and all(length > 1 for length in episode_lengths)
            and sum(episode_lengths) == total_frames == 97_766
        ),
        "train_split_covers_full_dataset": dict(info.get("splits") or {}).get(
            "train"
        )
        == "0:100",
        "source_fps_exact": float(info.get("fps") or 0.0) == SOURCE_FPS,
        "action_joint_inventory_exact": action_contract.get("shape") == [43]
        and tuple(action_contract.get("names") or ()) == SOURCE_ACTION_JOINT_NAMES,
        "state_joint_inventory_exact": state_contract.get("shape") == [43]
        and tuple(state_contract.get("names") or ()) == SOURCE_ACTION_JOINT_NAMES,
        "ego_video_contract_exact": bool(
            video_contract.get("shape") == [480, 640, 3]
            and video_details.get("video.width") == 640
            and video_details.get("video.height") == 480
            and float(video_details.get("video.fps") or 0.0) == SOURCE_FPS
            and video_details.get("video.codec") == "h264"
        ),
        "single_exact_task": task_rows == [
            {"task_index": 3, "task": SOURCE_TASK_DESCRIPTION}
        ],
        "every_episode_matches_exact_task": episodes_match_task,
        "source_modality_has_required_groups": bool(
            isinstance(source_state, Mapping)
            and isinstance(source_action, Mapping)
            and isinstance(source_video, Mapping)
            and isinstance(source_annotation, Mapping)
            and set(source_state)
            >= {
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            }
            and set(source_action)
            >= {
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            }
            and dict(source_video).get("ego_view", {}).get("original_key")
            == "observation.images.ego_view"
            and "human.task_description" in source_annotation
        ),
    }
    metadata_valid = all(metadata_checks.values())

    normalized_license = str(declared_license or "").strip().lower()
    license_approved = normalized_license in APPROVED_TRAINING_LICENSES
    attestation = (
        dict(fixed_base_upright_attestation)
        if isinstance(fixed_base_upright_attestation, Mapping)
        else {}
    )
    artifact_sha256 = str(attestation.get("artifact_sha256") or "").lower()
    attestation_valid = bool(
        attestation.get("status") == "passed"
        and attestation.get("dataset_repo") == SOURCE_DATASET_REPO
        and attestation.get("dataset_revision") == SOURCE_DATASET_REVISION
        and attestation.get("fixed_base") is True
        and attestation.get("upright") is True
        and len(artifact_sha256) == 64
        and all(character in "0123456789abcdef" for character in artifact_sha256)
    )
    blockers: list[str] = []
    if not metadata_valid:
        blockers.append("source_dataset_metadata_contract_invalid")
    if not license_approved:
        blockers.append("source_dataset_license_undeclared_or_unapproved")
    if not attestation_valid:
        blockers.append("source_dataset_fixed_base_upright_attestation_unproven")
    return {
        "schema_version": TRAINING_SOURCE_AUDIT_SCHEMA_VERSION,
        "status": (
            "admitted_for_sonic_training_materialization"
            if not blockers
            else "blocked"
        ),
        "blockers": blockers,
        "source_dataset": {
            "repo": SOURCE_DATASET_REPO,
            "revision": SOURCE_DATASET_REVISION,
            "declared_license": declared_license,
        },
        "task_description": SOURCE_TASK_DESCRIPTION,
        "metadata_checks": metadata_checks,
        "license_approved": license_approved,
        "approved_training_licenses": sorted(APPROVED_TRAINING_LICENSES),
        "fixed_base_upright_attestation": attestation or None,
        "fixed_base_upright_attestation_valid": attestation_valid,
        "claim_boundary": {
            "technical_convertibility_is_not_rights_clearance": True,
            "dataset_admission_is_not_a_trained_checkpoint": True,
            "trained_checkpoint_is_not_task_qualification": True,
            "task_qualification_is_not_episode_success": True,
        },
    }


def unitree_g1_sonic_training_modality() -> dict[str, Any]:
    """Return the exact modalities consumed by NVIDIA's current SONIC tag."""

    return {
        "state": {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
        },
        "action": {
            "motion_token": {
                "start": 0,
                "end": MOTION_TOKEN_DIM,
                "original_key": "action.motion_token",
            },
            "left_hand_joints": {
                "start": 0,
                "end": HAND_DIM,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": HAND_DIM,
                "original_key": "teleop.right_hand_joints",
            },
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def build_training_columns(
    actions: Any,
    *,
    action_joint_names: Sequence[str],
    encoder: Callable[[np.ndarray], Any],
    fixed_base_upright_attested: bool,
    source_provenance: Mapping[str, Any] | None = None,
    fps: float = SOURCE_FPS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build only the columns read by ``UNITREE_G1_SONIC`` fine-tuning.

    This deliberately does not write a dataset.  The caller must preserve the
    original ego videos and language annotations, regenerate GR00T statistics,
    and retain the returned report beside any derived dataset.
    """

    action_array, _ = _validated_actions(
        actions,
        action_joint_names=action_joint_names,
    )
    anchors = fixed_upright_root_anchor_rotations(
        action_array.shape[0],
        fixed_base_upright_attested=fixed_base_upright_attested,
    )
    sonic_actions, report = convert_to_sonic_actions(
        action_array,
        action_joint_names=action_joint_names,
        root_anchor_rotations_6d=anchors,
        encoder=encoder,
        source_provenance=source_provenance,
        fps=fps,
    )
    columns = {
        "observation.projected_gravity": np.tile(
            np.asarray(FIXED_UPRIGHT_PROJECTED_GRAVITY, dtype=np.float32),
            (action_array.shape[0], 1),
        ),
        "action.motion_token": sonic_actions[:, :MOTION_TOKEN_DIM],
        "teleop.left_hand_joints": sonic_actions[
            :, MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM
        ],
        "teleop.right_hand_joints": sonic_actions[
            :, MOTION_TOKEN_DIM + HAND_DIM : SONIC_ACTION_DIM
        ],
    }
    report["training_schema"] = {
        "isaac_groot_repo": ISAAC_GROOT_REPO,
        "isaac_groot_revision": ISAAC_GROOT_REVISION,
        "embodiment_tag": "UNITREE_G1_SONIC",
        "state_keys": [
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "projected_gravity",
        ],
        "action_keys": [
            "motion_token",
            "left_hand_joints",
            "right_hand_joints",
        ],
        "video_keys": ["ego_view"],
        "language_keys": ["annotation.human.task_description"],
    }
    report["claim_boundary"]["recorded_token_replay_equivalence_proven"] = False
    return columns, report


def _validated_actions(
    actions: Any,
    *,
    action_joint_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    names = tuple(str(name) for name in action_joint_names)
    if len(names) != len(set(names)):
        raise ValueError("g1_sonic_conversion_duplicate_action_joint_names")
    if set(names) != set(SOURCE_ACTION_JOINT_NAMES):
        raise ValueError("g1_sonic_conversion_action_joint_inventory_mismatch")
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(names) or array.shape[0] < 2:
        raise ValueError("g1_sonic_conversion_action_trajectory_shape_invalid")
    if not np.isfinite(array).all():
        raise ValueError("g1_sonic_conversion_action_trajectory_non_finite")
    return array, {name: index for index, name in enumerate(names)}


def fixed_upright_root_anchor_rotations(
    frame_count: int,
    *,
    fixed_base_upright_attested: bool,
) -> np.ndarray:
    """Return identity root anchors only after an explicit source attestation."""

    if not fixed_base_upright_attested:
        raise ValueError("g1_sonic_conversion_root_pose_missing_without_attestation")
    if int(frame_count) < 2:
        raise ValueError("g1_sonic_conversion_frame_count_invalid")
    return np.tile(
        np.asarray(IDENTITY_ROTATION_6D, dtype=np.float32),
        (int(frame_count), 1),
    )


def build_sonic_encoder_inputs(
    actions: Any,
    *,
    action_joint_names: Sequence[str],
    root_anchor_rotations_6d: Any,
    fps: float = SOURCE_FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build official GEAR-SONIC G1-mode encoder inputs and hand targets."""

    action_array, index = _validated_actions(
        actions,
        action_joint_names=action_joint_names,
    )
    if not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("g1_sonic_conversion_fps_invalid")
    anchors = np.asarray(root_anchor_rotations_6d, dtype=np.float32)
    if anchors.shape != (action_array.shape[0], 6) or not np.isfinite(anchors).all():
        raise ValueError("g1_sonic_conversion_root_anchor_rotations_invalid")

    body = action_array[:, [index[name] for name in ISAACLAB_BODY_JOINT_NAMES]]
    velocities = np.gradient(body, 1.0 / float(fps), axis=0).astype(np.float32)
    frame_offsets = np.arange(FUTURE_FRAME_COUNT) * FUTURE_FRAME_STRIDE
    inputs = np.zeros((action_array.shape[0], ENCODER_INPUT_DIM), dtype=np.float32)
    for frame_index in range(action_array.shape[0]):
        window = np.minimum(frame_index + frame_offsets, action_array.shape[0] - 1)
        inputs[frame_index, BODY_POSITION_SLICE] = body[window].reshape(-1)
        inputs[frame_index, BODY_VELOCITY_SLICE] = velocities[window].reshape(-1)
        inputs[frame_index, ROOT_ANCHOR_ROTATION_SLICE] = anchors[window].reshape(-1)

    left_hand = action_array[:, [index[name] for name in LEFT_HAND_JOINT_NAMES]]
    right_hand = action_array[:, [index[name] for name in RIGHT_HAND_JOINT_NAMES]]
    return inputs, left_hand, right_hand


def load_onnx_encoder(
    model_path: str | Path,
    *,
    expected_sha256: str = GEAR_SONIC_ENCODER_SHA256,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load the exact official encoder with hash and tensor-contract checks."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("g1_sonic_conversion_encoder_missing")
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("g1_sonic_conversion_encoder_sha256_mismatch")
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("g1_sonic_conversion_onnxruntime_missing") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "obs_dict"
        or inputs[0].shape != [1, ENCODER_INPUT_DIM]
        or len(outputs) != 1
        or outputs[0].name != "encoded_tokens"
        or outputs[0].shape != [1, MOTION_TOKEN_DIM]
    ):
        raise ValueError("g1_sonic_conversion_encoder_tensor_contract_mismatch")

    def encode(batch: np.ndarray) -> np.ndarray:
        rows = np.asarray(batch, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != ENCODER_INPUT_DIM:
            raise ValueError("g1_sonic_conversion_encoder_batch_shape_invalid")
        encoded = [
            session.run(["encoded_tokens"], {"obs_dict": row[None, :]})[0][0]
            for row in rows
        ]
        return np.asarray(encoded, dtype=np.float32)

    return encode


def convert_to_sonic_actions(
    actions: Any,
    *,
    action_joint_names: Sequence[str],
    root_anchor_rotations_6d: Any,
    encoder: Callable[[np.ndarray], Any],
    source_provenance: Mapping[str, Any] | None = None,
    fps: float = SOURCE_FPS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return 78D SONIC actions plus a truth-bounded conversion report."""

    inputs, left_hand, right_hand = build_sonic_encoder_inputs(
        actions,
        action_joint_names=action_joint_names,
        root_anchor_rotations_6d=root_anchor_rotations_6d,
        fps=fps,
    )
    tokens = np.asarray(encoder(inputs), dtype=np.float32)
    if tokens.shape != (inputs.shape[0], MOTION_TOKEN_DIM):
        raise ValueError("g1_sonic_conversion_motion_token_shape_invalid")
    if not np.isfinite(tokens).all():
        raise ValueError("g1_sonic_conversion_motion_tokens_non_finite")
    sonic_actions = np.concatenate((tokens, left_hand, right_hand), axis=1)
    if sonic_actions.shape != (inputs.shape[0], SONIC_ACTION_DIM):
        raise AssertionError("g1_sonic_conversion_internal_action_shape_error")

    action_digest = hashlib.sha256(sonic_actions.tobytes(order="C")).hexdigest()
    source = (
        dict(source_provenance)
        if isinstance(source_provenance, Mapping)
        else {
            "source_type": "public_dataset_candidate",
            "repo": SOURCE_DATASET_REPO,
            "revision": SOURCE_DATASET_REVISION,
        }
    )
    if not str(source.get("source_type") or "").strip():
        raise ValueError("g1_sonic_conversion_source_provenance_type_missing")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "converted_training_actions",
        "frame_count": int(sonic_actions.shape[0]),
        "source_action_dim": len(SOURCE_ACTION_JOINT_NAMES),
        "motion_token_dim": MOTION_TOKEN_DIM,
        "left_hand_dim": HAND_DIM,
        "right_hand_dim": HAND_DIM,
        "sonic_action_dim": SONIC_ACTION_DIM,
        "sonic_actions_sha256": action_digest,
        "motion_tokens_finite": True,
        "motion_token_unique_rows_rounded_6": int(
            np.unique(tokens.round(6), axis=0).shape[0]
        ),
        "source_provenance": source,
        "encoder": {
            "repo": GEAR_SONIC_ENCODER_REPO,
            "revision": GEAR_SONIC_ENCODER_REVISION,
            "path": GEAR_SONIC_ENCODER_PATH,
            "sha256": GEAR_SONIC_ENCODER_SHA256,
        },
        "claim_boundary": {
            "converted_actions_are_training_inputs_only": True,
            "converted_actions_are_not_a_trained_checkpoint": True,
            "converted_actions_are_not_task_qualification": True,
            "converted_actions_are_not_episode_success": True,
            "root_pose_source_must_be_attested_separately": True,
        },
    }
    json.dumps(report, sort_keys=True)
    return sonic_actions, report


def _read_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"g1_sonic_training_source_metadata_not_object:{path.name}")
    return dict(value)


def _read_jsonl_mappings(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(
                "g1_sonic_training_source_jsonl_row_not_object:"
                f"{path.name}:{line_number}"
            )
        rows.append(dict(value))
    return rows


def audit_training_source_directory(
    metadata_dir: str | Path,
    *,
    declared_license: str | None,
    fixed_base_upright_attestation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit and hash-bind one downloaded LeRobot metadata directory."""

    root = Path(metadata_dir).expanduser().resolve()
    paths = {
        "info": root / "info.json",
        "modality": root / "modality.json",
        "tasks": root / "tasks.jsonl",
        "episodes": root / "episodes.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "g1_sonic_training_source_metadata_files_missing:" + ",".join(missing)
        )
    attestation = (
        _read_json_mapping(Path(fixed_base_upright_attestation_path).expanduser())
        if fixed_base_upright_attestation_path
        else None
    )
    report = audit_training_source_metadata(
        info=_read_json_mapping(paths["info"]),
        modality=_read_json_mapping(paths["modality"]),
        tasks=_read_jsonl_mappings(paths["tasks"]),
        episodes=_read_jsonl_mappings(paths["episodes"]),
        declared_license=declared_license,
        fixed_base_upright_attestation=attestation,
    )
    report["source_metadata_artifacts"] = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a candidate G1 microwave dataset before SONIC fine-tuning."
    )
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--declared-license")
    parser.add_argument("--fixed-base-upright-attestation")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit_training_source_directory(
            args.metadata_dir,
            declared_license=args.declared_license,
            fixed_base_upright_attestation_path=(
                args.fixed_base_upright_attestation
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": TRAINING_SOURCE_AUDIT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [f"training_source_audit_failed:{type(exc).__name__}:{exc}"],
        }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result.get("status") == "admitted_for_sonic_training_materialization" else 1


if __name__ == "__main__":
    raise SystemExit(main())
