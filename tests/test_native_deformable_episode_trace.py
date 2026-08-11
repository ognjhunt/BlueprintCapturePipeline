from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import native_deformable_episode_trace as episode_module
from blueprint_pipeline.native_deformable_episode_trace import (
    ACTION_REPLAY_CONTRACT_ID,
    ACTION_REPLAY_SCHEMA_VERSION,
    CAMERA_IDS,
    FROZEN_RUN_SCHEMA_VERSION,
    H264_DECODE_VERIFIER_CONTRACT_ID,
    MEDIA_MANIFEST_SCHEMA_VERSION,
    MINIMUM_ARM_MOTION_EPSILON_RAD,
    MINIMUM_GRIPPER_MOTION_EPSILON_M,
    NATIVE_TRACE_MANIFEST_SCHEMA_VERSION,
    NativeDeformableEpisodeTraceError,
    RESET_STATE_PROJECTION_SCHEMA_VERSION,
    aggregate_native_deformable_cell_evaluation,
    materialize_native_deformable_episode_trace,
)
from blueprint_pipeline.native_task_entity_contract import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    materialize_native_task_entity_contract,
)
from blueprint_pipeline.trusted_execution_envelope import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    canonical_trusted_execution_envelope_bytes,
    materialize_trusted_execution_envelope,
    materialize_trusted_execution_payload,
    trusted_execution_signature_message,
)


_H264_MP4 = base64.b64decode(
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAANgbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAMgAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAot0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAMgAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAADIAAAEAAABAAAAAAIDbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAACABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABrm1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAW5zdGJsAAAAvnN0c2QAAAAAAAAAAQAAAK5hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAABAAEABIAAAASAAAAAAAAAABFUxhdmM2MS4xOS4xMDEgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAANGF2Y0MBZAAK/+EAF2dkAAqs2V7ARAAAAwAEAAADAKA8SJZYAQAGaOvjyyLA/fj4AAAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAAHRoAAAAAAAAABhzdHRzAAAAAAAAAAEAAAAEAAACAAAAABRzdHNzAAAAAAAAAAEAAAABAAAAKGN0dHMAAAAAAAAAAwAAAAEAAAQAAAAAAQAACAAAAAACAAACAAAAABxzdHNjAAAAAAAAAAEAAAABAAAABAAAAAEAAAAkc3RzegAAAAAAAAAAAAAABAAAAsUAAAAMAAAADAAAAAwAAAAUc3RjbwAAAAAAAAABAAADkAAAAGF1ZHRhAAAAWW1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALGlsc3QAAAAkqXRvbwAAABxkYXRhAAAAAQAAAABMYXZmNjEuNy4xMDAAAAAIZnJlZQAAAvFtZGF0AAACrgYF//+q3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCByMzEwOCAzMWUxOWY5IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyMyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTIwIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAD2WIhAA3//728P4FNlYEwQAAAAhBmiNsQr/+wAAAAAhBnkF4hf/BgQAAAAgBnmJqQr/EgA=="
)

_H264_MP4_BY_CAMERA_ID = {
    "external": base64.b64decode(
        "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"
    ),
    "wrist": base64.b64decode(
        "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"
    ),
    "overview": base64.b64decode(
        "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"
    ),
}


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: object, *, digest_field: str | None = None) -> str:
    assert isinstance(value, dict)
    normalized = dict(value)
    if digest_field is not None:
        normalized.pop(digest_field, None)
    return _bytes_digest(_canonical_bytes(normalized))


_DEFORMABLE_RESET_POSITIONS = [
    [0.8, 0.8, 0.0],
    [0.9, 0.8, 0.0],
    [0.9, 0.9, 0.0],
    [0.8, 0.9, 0.0],
]
_DEFORMABLE_CONTAINED_POSITIONS = [
    [-0.1, -0.1, 0.0],
    [0.1, -0.1, 0.0],
    [0.1, 0.1, 0.0],
    [-0.1, 0.1, 0.0],
]


def _deformable_start_state_sha256(
    positions: list[list[float]] | None = None,
    velocities: list[list[float]] | None = None,
    targets: list[list[float]] | None = None,
) -> str:
    resolved_positions = positions or _DEFORMABLE_RESET_POSITIONS
    resolved_velocities = velocities or [[0.0, 0.0, 0.0] for _ in resolved_positions]
    resolved_targets = targets or [[*point, 1.0] for point in resolved_positions]
    return _digest(
        {
            "schema_version": RESET_STATE_PROJECTION_SCHEMA_VERSION,
            "entity_id": "cloth",
            "physics_type": "deformable_volume",
            "nodal_positions_world_m": resolved_positions,
            "nodal_velocities_world_mps": resolved_velocities,
            "nodal_kinematic_targets": resolved_targets,
        }
    )


def _seal(value: dict, field: str) -> dict:
    value[field] = _digest(value, digest_field=field)
    return value


_PHYSICS = {
    "movable_deformable": "deformable_volume",
    "destination_receptacle": "static_collider",
    "support_surface": "static_collider",
    "obstacle": "static_collider",
    "robot": "robot_articulation",
}
_RESET = {
    "deformable_volume": "native_deformable_state",
    "static_collider": "immutable_scene_state",
    "robot_articulation": "native_robot_state",
}
_CONTACT = {
    "movable_deformable": "manipulated_deformable",
    "destination_receptacle": "destination_volume",
    "support_surface": "supporting_surface",
    "obstacle": "collision_obstacle",
    "robot": "manipulator",
}
_SCORING = {
    "movable_deformable": "deformable_target",
    "destination_receptacle": "destination",
    "support_surface": "support_context",
    "obstacle": "collision_context",
    "robot": "robot_context",
}


def _entity(entity_id: str, role: str, character: str, *, inserted: bool = False) -> dict:
    physics = _PHYSICS[role]
    source_digest = _sha(character)
    asset_digest = _sha(hex((int(character, 16) + 1) % 16)[2:])
    state_digest = _sha(hex((int(character, 16) + 2) % 16)[2:])
    removal = (
        {
            "source_entity_action": "not_present",
            "gaussian_action": "not_applicable",
            "collider_action": "not_applicable",
            "receipt_sha256": _sha("e"),
        }
        if inserted
        else {
            "source_entity_action": "retain",
            "gaussian_action": "retain",
            "collider_action": "retain",
            "receipt_sha256": _sha("e"),
        }
    )
    replacement = (
        {
            "action": "insert_runtime_asset",
            "replacement_required": True,
            "receipt_sha256": _sha("f"),
        }
        if inserted
        else {
            "action": "retain_registered_source",
            "replacement_required": False,
            "receipt_sha256": _sha("f"),
        }
    )
    return {
        "entity_id": entity_id,
        "semantic_role": role,
        "source_observation": {
            "observation_id": f"observation:{entity_id}",
            "source_kind": (
                "runtime_embodiment"
                if role == "robot"
                else "generated_runtime_asset"
                if inserted
                else "registered_scene_geometry"
            ),
            "source_reference": f"sources/{entity_id}",
            "source_sha256": source_digest,
            "observed": role != "robot",
        },
        "physics_type": physics,
        "runtime_asset": {
            "asset_id": f"asset:{entity_id}",
            "binding_kind": (
                "runtime_embodiment"
                if role == "robot"
                else "usd_asset"
                if inserted
                else "registered_scene_geometry"
            ),
            "source_reference": f"assets/{entity_id}.usd",
            "sha256": asset_digest,
        },
        "initial_state": {
            "pose_world": {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "state_sha256": state_digest,
            "settled_state_required": True,
            "initial_penetration_allowed": False,
        },
        "reset_method": {
            "kind": _RESET[physics],
            "state_id": f"reset:{entity_id}",
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        },
        "contact_role": {
            "kind": _CONTACT[role],
            "native_contact_readback_required": True,
        },
        "scoring_role": {
            "kind": _SCORING[role],
            "deterministic_state_readback_required": True,
            "policy_self_grading_allowed": False,
        },
        "removal_policy": removal,
        "replacement_policy": replacement,
        "provenance": {
            "source_id": f"source:{entity_id}",
            "source_revision": "fixture-r1",
            "source_path": f"fixture/{entity_id}",
            "source_size_bytes": 1024,
            "license_id": "fixture-license",
            "public_source_rights_id": "fixture-public-rights",
            "derived_processing_authority_id": "fixture-derived-authority",
            "provider_terms_id": "fixture-provider-terms",
            "output_rights_id": "fixture-output-rights",
            "attribution": "Hermetic fixture",
            "disclosure_class": ("runtime_bundled" if role == "robot" else "generated_derivative"),
            "upload_permitted": True,
            "raw_redistribution_permitted": role == "robot",
            "provider_retention_permitted": False,
            "provider_training_permitted": False,
        },
        "digests": {
            "source_sha256": source_digest,
            "runtime_asset_sha256": asset_digest,
            "initial_state_sha256": state_digest,
            "configuration_sha256": _sha(hex((int(character, 16) + 3) % 16)[2:]),
        },
    }


def _entities() -> list[dict]:
    return [
        _entity("cloth", "movable_deformable", "1", inserted=True),
        _entity("basket", "destination_receptacle", "4", inserted=True),
        _entity("counter", "support_surface", "7"),
        _entity("wall", "obstacle", "a"),
        _entity("pillar", "obstacle", "b"),
        _entity("manipulator_alpha", "robot", "d", inserted=True),
    ]


def _task_spec() -> dict:
    return {
        "deformable_entity_id": "cloth",
        "destination_entity_id": "basket",
        "robot_entity_id": "manipulator_alpha",
        "destination_interior_obb": {
            "center_world_m": [0.0, 0.0, 0.0],
            "half_extents_m": [0.5, 0.5, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "receptacle_reference_pose_world": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "minimum_particle_fraction_inside": 0.75,
        "settle_window_samples": 2,
        "maximum_node_speed_mps": 0.01,
        "maximum_principal_strain": 0.2,
        "minimum_grasp_contact_force_n": 0.1,
        "maximum_release_contact_force_n": 0.01,
        "minimum_robot_clearance_m": 0.2,
        "maximum_receptacle_translation_drift_m": 0.01,
        "maximum_receptacle_rotation_drift_rad": 0.01,
        "maximum_receptacle_linear_speed_mps": 0.01,
        "maximum_receptacle_angular_speed_radps": 0.01,
    }


def _actor(kind: str, *, candidate_id: str = "pi05_droid") -> dict:
    if kind == "learned_policy_evaluation":
        character = "1" if candidate_id == "pi05_droid" else "6"
        return _seal(
            {
                "kind": "learned_policy",
                "candidate_id": candidate_id,
                "source_reference": f"https://example.invalid/{candidate_id}",
                "source_revision": "frozen-r1",
                "checkpoint_reference": f"private/{candidate_id}",
                "checkpoint_sha256": _sha(character),
                "runtime_sha256": _sha("2"),
                "preprocessing_sha256": _sha("3"),
                "action_adapter_sha256": _sha("4"),
                "model_seed": 2026081001,
                "policy_self_grading_allowed": False,
            },
            "identity_digest",
        )
    return _seal(
        {
            "kind": "deterministic_control",
            "control_id": ("zero_action" if kind == "zero_action_control" else "scripted_positive"),
            "source_revision": "fixture-control-r1",
            "controller_sha256": _sha("5"),
            "control_seed": 2026081001,
            "policy_self_grading_allowed": False,
        },
        "identity_digest",
    )


def _thresholds() -> dict:
    return {
        "maximum_frame_sync_skew_ns": 1_000,
        "maximum_frame_simulation_time_skew_s": 1.0e-6,
        "arm_motion_epsilon_rad": MINIMUM_ARM_MOTION_EPSILON_RAD,
        "gripper_motion_epsilon_m": MINIMUM_GRIPPER_MOTION_EPSILON_M,
        "action_epsilon": 1.0e-6,
        "contact_force_epsilon_n": 1.0e-6,
        "minimum_deformable_displacement_m": 0.05,
    }


def _action_replay_contract(
    *,
    arm_scale: list[float] | None = None,
    arm_offset: list[float] | None = None,
    gripper_scale: float = 1.0,
    gripper_offset: float = 0.0,
) -> dict:
    return _seal(
        {
            "schema_version": ACTION_REPLAY_SCHEMA_VERSION,
            "contract_id": ACTION_REPLAY_CONTRACT_ID,
            "command_space": "fixture_joint_delta",
            "source_output_size": 3,
            "arm_source_indices": [0, 1],
            "arm_scale": arm_scale or [1.0, 1.0],
            "arm_offset": arm_offset or [0.0, 0.0],
            "gripper_source_index": 2,
            "gripper_scale": gripper_scale,
            "gripper_offset": gripper_offset,
            "native_action_layout": "arm_then_gripper",
        },
        "contract_digest",
    )


def _camera_contract() -> dict:
    return {
        "camera_calibration_digest_by_camera_id": {
            camera_id: _calibration(camera_id)["calibration_digest"] for camera_id in CAMERA_IDS
        },
        "renderer_identity_sha256_by_camera_id": {camera_id: _sha("b") for camera_id in CAMERA_IDS},
    }


def _cell(family: str) -> dict:
    return {
        "cell_id": f"cell-{family}",
        "family": family,
        "seed": 2026081001,
        "scene_sha256": _sha("1"),
        "asset_bundle_sha256": _sha("2"),
        "resolved_parameters_sha256": _sha("3"),
        "camera_contract_sha256": _digest(_camera_contract()),
        "native_applied_parameters_receipt_sha256": _sha("5"),
    }


def _frozen_run(*, entities: list[dict], spec: dict, prompt: str, cell: dict, reset: dict) -> dict:
    entity_contract = materialize_native_task_entity_contract(
        task_kind=TASK_KIND_DEFORMABLE_TRANSFER,
        task_entities=entities,
    )
    actors = [
        _actor("learned_policy_evaluation", candidate_id=candidate_id)
        for candidate_id in ("pi05_droid", "groot_n17_droid")
    ] + [_actor(kind) for kind in ("zero_action_control", "scripted_positive_control")]
    camera_contract = _camera_contract()
    return _seal(
        {
            "schema_version": FROZEN_RUN_SCHEMA_VERSION,
            "suite_id": "deformable-fixture-suite",
            "entity_contract_digest": entity_contract["contract_digest"],
            "task_spec_sha256": _digest(spec),
            "prompt_sha256": _bytes_digest(prompt.encode()),
            "cell_identity_digest_by_id": {cell["cell_id"]: _digest(cell)},
            "candidate_identity_digest_by_id": {
                candidate_id: _actor("learned_policy_evaluation", candidate_id=candidate_id)[
                    "identity_digest"
                ]
                for candidate_id in ("pi05_droid", "groot_n17_droid")
            },
            "control_identity_digest_by_episode_kind": {
                kind: _actor(kind)["identity_digest"]
                for kind in ("zero_action_control", "scripted_positive_control")
            },
            "trace_thresholds_sha256": _digest(_thresholds()),
            **camera_contract,
            "action_replay_contract_by_actor_identity_digest": {
                actor["identity_digest"]: _action_replay_contract() for actor in actors
            },
            "review_video_codec_by_camera_id": {camera_id: "h264" for camera_id in CAMERA_IDS},
            "review_video_container_by_camera_id": {camera_id: "mp4" for camera_id in CAMERA_IDS},
            "frozen_reset_state_id": "frozen-reset-state-001",
            "frozen_reset_state_sha256": reset["frozen_reset_state_sha256"],
            "frozen_deformable_start_state_sha256": reset["deformable_start_state_sha256"],
        },
        "contract_digest",
    )


def _reset_state_projection(entity: dict) -> dict:
    entity_id = entity["entity_id"]
    physics_type = entity["physics_type"]
    projection = {
        "schema_version": RESET_STATE_PROJECTION_SCHEMA_VERSION,
        "entity_id": entity_id,
        "physics_type": physics_type,
    }
    if physics_type == "deformable_volume":
        positions = [list(point) for point in _DEFORMABLE_RESET_POSITIONS]
        projection.update(
            {
                "nodal_positions_world_m": positions,
                "nodal_velocities_world_mps": [[0.0, 0.0, 0.0] for _ in positions],
                "nodal_kinematic_targets": [[*point, 1.0] for point in positions],
            }
        )
    elif physics_type == "robot_articulation":
        projection.update(
            {
                "joint_positions_rad": [0.0, 0.0],
                "joint_velocities_rad_s": [0.0, 0.0],
                "gripper_width_m": 0.08,
            }
        )
    else:
        projection.update(
            {
                "pose_world": {
                    "position_m": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            }
        )
        if physics_type == "articulation":
            projection.update(
                {
                    "joint_positions_rad": [0.0],
                    "joint_velocities_rad_s": [0.0],
                }
            )
    return _seal(projection, "projection_digest")


def _reset_projection_set_digest(projections: dict[str, dict]) -> str:
    return _digest({"native_reset_state_projection_by_entity_id": projections})


def _reset(*, start_ns: int, actor: dict, entities: list[dict]) -> dict:
    entity_ids = sorted(entity["entity_id"] for entity in entities)
    projections = {
        entity["entity_id"]: _reset_state_projection(entity)
        for entity in sorted(entities, key=lambda item: item["entity_id"])
    }
    deformable = projections["cloth"]
    robot = projections["manipulator_alpha"]
    actor_seed = actor.get("model_seed", actor.get("control_seed"))
    actor_reset_method = "policy.reset" if actor["kind"] == "learned_policy" else "controller.reset"
    reset = _seal(
        {
            "reset_id": "reset-001",
            "frozen_reset_state_id": "frozen-reset-state-001",
            "frozen_reset_state_sha256": _reset_projection_set_digest(projections),
            "reset_timestamp_ns": start_ns - 3_000,
            "actor_reset_timestamp_ns": start_ns - 2_000,
            "native_readback_timestamp_ns": start_ns - 1_000,
            "actor_identity_digest": actor["identity_digest"],
            "actor_seed": actor_seed,
            "actor_reset_method": actor_reset_method,
            "actor_reset_invoked": True,
            "native_reset_write_count_by_entity_id": {
                entity_id: int(entity_id in {"cloth", "manipulator_alpha"})
                for entity_id in entity_ids
            },
            "native_state_readback_sha256_by_entity_id": {
                entity_id: projections[entity_id]["projection_digest"] for entity_id in entity_ids
            },
            "native_reset_state_projection_by_entity_id": projections,
            "deformable_nodal_positions_world_m": deformable["nodal_positions_world_m"],
            "deformable_nodal_velocities_world_mps": deformable["nodal_velocities_world_mps"],
            "deformable_nodal_kinematic_targets": deformable["nodal_kinematic_targets"],
            "deformable_start_state_sha256": deformable["projection_digest"],
            "robot_joint_positions_rad": robot["joint_positions_rad"],
            "robot_joint_velocities_rad_s": robot["joint_velocities_rad_s"],
            "gripper_width_m": robot["gripper_width_m"],
            "native_readback_matches_frozen_state": True,
            "initial_penetration_observed": False,
        },
        "receipt_digest",
    )
    return reset


def _calibration(camera_id: str) -> dict:
    return _seal(
        {
            "camera_id": camera_id,
            "transform_world_from_camera": {
                "position_m": [1.0, 2.0, 3.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "intrinsics": {
                "fx_px": 100.0,
                "fy_px": 100.0,
                "cx_px": 4.0,
                "cy_px": 3.0,
                "width_px": 8,
                "height_px": 6,
            },
        },
        "calibration_digest",
    )


def _frame(
    *,
    root: Path,
    camera_id: str,
    index: int,
    timestamp_ns: int,
    simulation_time_s: float,
    actor_observation: bool,
) -> dict:
    relative_path = f"media/{camera_id}/{index:04d}.png"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    color = {
        "external": (index * 20, 30, 40),
        "wrist": (50, index * 20, 60),
        "overview": (70, 80, index * 20),
    }[camera_id]
    image = Image.new("RGB", (8, 6), color=color)
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=9)
    payload = output.getvalue()
    path.write_bytes(payload)
    policy_eligible = camera_id in {"external", "wrist"}
    return _seal(
        {
            "camera_id": camera_id,
            "frame_sequence_index": index,
            "timestamp_ns": timestamp_ns,
            "simulation_time_s": simulation_time_s,
            "policy_input_eligible": policy_eligible,
            "presented_to_actor": actor_observation and policy_eligible,
            "review_only": camera_id == "overview",
            "used_for_deterministic_scoring": False,
            "encoding": "png",
            "width_px": 8,
            "height_px": 6,
            "channels": 3,
            "relative_path": relative_path,
            "size_bytes": len(payload),
            "lossless_file_sha256": _bytes_digest(payload),
            "raw_rgb_sha256": _bytes_digest(image.tobytes()),
            "calibration": _calibration(camera_id),
            "renderer_identity_sha256": _sha("b"),
        },
        "frame_digest",
    )


def _action(
    *,
    index: int,
    kind: str,
    start_ns: int,
    sample_ns: int,
    arm_command: list[float],
    gripper_delta: float,
    delivered: bool,
    actor_observation: bool,
    frames: dict[str, dict],
) -> dict:
    adapter = _sha("4") if kind == "learned_policy_evaluation" else _sha("5")
    native_action = [*arm_command, gripper_delta]
    delivery = _seal(
        {
            "attempted": True,
            "delivered_to_robot": delivered,
            "delivery_timestamp_ns": sample_ns - 100,
            "native_action_sha256": _bytes_digest(_canonical_bytes(native_action)),
            "adapter_sha256": adapter,
        },
        "receipt_digest",
    )
    source_output_sha256 = _bytes_digest(_canonical_bytes(native_action))
    policy_inference = (
        _seal(
            {
                "actor_identity_digest": _actor(kind)["identity_digest"],
                "policy_input_frame_digest_by_camera_id": {
                    camera_id: frames[camera_id]["frame_digest"]
                    for camera_id in sorted(("external", "wrist"))
                },
                "inference_started_timestamp_ns": sample_ns - 450,
                "inference_completed_timestamp_ns": sample_ns - 250,
                "source_output_sha256": source_output_sha256,
            },
            "receipt_digest",
        )
        if kind == "learned_policy_evaluation" and actor_observation
        else None
    )
    return _seal(
        {
            "action_index": index,
            "origin_kind": (
                "learned_policy"
                if kind == "learned_policy_evaluation" and actor_observation
                else "harness_settle"
                if kind == "learned_policy_evaluation"
                else "zero_action_control"
                if kind == "zero_action_control"
                else "scripted_control"
            ),
            "command_space": "fixture_joint_delta",
            "command_timestamp_ns": max(start_ns, sample_ns - 200),
            "arm_command": arm_command,
            "gripper_delta_command_m": gripper_delta,
            "native_action": native_action,
            "source_output": native_action,
            "source_output_sha256": source_output_sha256,
            "policy_inference": policy_inference,
            "delivery": delivery,
        },
        "action_digest",
    )


def _scored_state_digests(step: dict) -> dict[str, str]:
    result = {entity_id: _digest(state) for entity_id, state in step["entities"].items()}
    for entity_id in ("counter", "wall", "pillar"):
        result[entity_id] = _digest(
            {
                "entity_id": entity_id,
                "sample_index": step["sample_index"],
                "opaque_native_state": True,
            }
        )
    return dict(sorted(result.items()))


def _step(
    *,
    root: Path,
    index: int,
    kind: str,
    start_ns: int,
    move: bool,
    delivered: bool,
    successful_state: bool,
) -> dict:
    timestamp_ns = start_ns + (index + 1) * 1_000_000
    simulation_time = (index + 1) * 0.05
    terminal = index == 3
    actor_observation = kind == "learned_policy_evaluation" and index < 3
    if terminal:
        observation_kind = "terminal"
        actor_observation = False
    elif actor_observation:
        observation_kind = "actor_input"
    elif kind == "learned_policy_evaluation":
        observation_kind = "review_sample"
    else:
        observation_kind = "control_sample"

    active_control = kind == "scripted_positive_control" or (
        kind == "learned_policy_evaluation" and actor_observation
    )
    arm_command = [0.1, 0.0] if active_control else [0.0, 0.0]
    gripper_delta = -0.04 if index == 0 else 0.04 if index == 2 else 0.0
    if not active_control:
        gripper_delta = 0.0
    contact = bool(active_control and move and index in {0, 1})
    contact_pairs = int(contact)
    contact_force = 2.0 if contact else 0.0
    if move and kind == "learned_policy_evaluation":
        joint_position = [0.1 * (min(index, 2) + 1), 0.0]
    elif move and active_control:
        joint_position = [0.1 * (index + 1), 0.0]
    else:
        joint_position = [0.0, 0.0]
    gripper_width = 0.04 if move and active_control and index in {0, 1} else 0.08
    node_positions = [
        list(point)
        for point in (
            _DEFORMABLE_CONTAINED_POSITIONS
            if successful_state and index > 0
            else _DEFORMABLE_RESET_POSITIONS
        )
    ]
    frames = {
        camera_id: _frame(
            root=root,
            camera_id=camera_id,
            index=index,
            timestamp_ns=timestamp_ns - 500,
            simulation_time_s=simulation_time,
            actor_observation=actor_observation,
        )
        for camera_id in CAMERA_IDS
    }
    step = {
        "sample_index": index,
        "timestamp_ns": timestamp_ns,
        "simulation_time_s": simulation_time,
        "observation_kind": observation_kind,
        "actor_observation": actor_observation,
        "action": _action(
            index=index,
            kind=kind,
            start_ns=start_ns,
            sample_ns=timestamp_ns,
            arm_command=arm_command,
            gripper_delta=gripper_delta,
            delivered=delivered,
            actor_observation=actor_observation,
            frames=frames,
        ),
        "entities": {
            "cloth": {
                "nodal_positions_world_m": node_positions,
                "nodal_velocities_world_mps": [[0.0, 0.0, 0.0]] * 4,
                "deformation_gradients": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                "nodal_kinematic_flags": [1.0] * 4,
                "state_write_count_after_episode_start": 0,
                "solver_divergence_count": 0,
                "contact_pair_count_by_entity_id": {
                    "manipulator_alpha": contact_pairs,
                    "basket": 0,
                },
                "contact_normal_force_n_by_entity_id": {
                    "manipulator_alpha": contact_force,
                    "basket": 0.0,
                },
                "hidden_attachment_active": False,
                "grasp_representation": "native_contact_only",
            },
            "basket": {
                "pose_world": {
                    "position_m": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                "angular_velocity_world_radps": [0.0, 0.0, 0.0],
                "state_write_count_after_episode_start": 0,
            },
            "manipulator_alpha": {
                "arm_joint_positions_rad": joint_position,
                "arm_joint_velocities_rad_s": [0.0, 0.0],
                "gripper_width_m": gripper_width,
                "gripper_clearance_points_world_m": [[2.0, 2.0, 2.0]],
                "gripper_contact_pair_count_by_entity_id": {"cloth": contact_pairs},
                "gripper_contact_normal_force_n_by_entity_id": {"cloth": contact_force},
                "gripper_attachment_constraint_count_by_entity_id": {"cloth": 0},
                "state_write_count_after_episode_start": 0,
            },
        },
        "frames": frames,
    }
    step["native_state_sha256_by_entity_id"] = _scored_state_digests(step)
    step["state_write_count_after_episode_start_by_entity_id"] = {
        entity_id: 0
        for entity_id in (
            "basket",
            "cloth",
            "counter",
            "manipulator_alpha",
            "pillar",
            "wall",
        )
    }
    step["observation_digest"] = _digest(step)
    return step


def _media_document(trace: dict) -> dict:
    frames_by_camera: dict[str, list[dict]] = {}
    for camera_id in CAMERA_IDS:
        frames_by_camera[camera_id] = []
        for step in trace["steps"]:
            frame = step["frames"][camera_id]
            frames_by_camera[camera_id].append(
                {
                    "sample_index": step["sample_index"],
                    "frame_sequence_index": frame["frame_sequence_index"],
                    "timestamp_ns": frame["timestamp_ns"],
                    "simulation_time_s": frame["simulation_time_s"],
                    "relative_path": frame["relative_path"],
                    "size_bytes": frame["size_bytes"],
                    "lossless_file_sha256": frame["lossless_file_sha256"],
                    "raw_rgb_sha256": frame["raw_rgb_sha256"],
                    "calibration_digest": frame["calibration"]["calibration_digest"],
                    "renderer_identity_sha256": frame["renderer_identity_sha256"],
                    "presented_to_actor": frame["presented_to_actor"],
                    "review_only": frame["review_only"],
                    "used_for_deterministic_scoring": False,
                    "frame_digest": frame["frame_digest"],
                }
            )
    return {
        "schema_version": MEDIA_MANIFEST_SCHEMA_VERSION,
        "episode_id": trace["episode_id"],
        "frames_by_camera": frames_by_camera,
    }


def _trace_rows(trace: dict) -> list[dict]:
    rows = []
    for step in trace["steps"]:
        cloth = step["entities"]["cloth"]
        robot = step["entities"]["manipulator_alpha"]
        contact_projection = {
            "deformable_contact_pair_count_by_entity_id": cloth["contact_pair_count_by_entity_id"],
            "deformable_contact_normal_force_n_by_entity_id": cloth[
                "contact_normal_force_n_by_entity_id"
            ],
            "robot_gripper_contact_pair_count_by_entity_id": robot[
                "gripper_contact_pair_count_by_entity_id"
            ],
            "robot_gripper_contact_normal_force_n_by_entity_id": robot[
                "gripper_contact_normal_force_n_by_entity_id"
            ],
            "robot_gripper_attachment_constraint_count_by_entity_id": robot[
                "gripper_attachment_constraint_count_by_entity_id"
            ],
            "hidden_attachment_active": cloth["hidden_attachment_active"],
        }
        rows.append(
            {
                "episode_id": trace["episode_id"],
                "sample_index": step["sample_index"],
                "timestamp_ns": step["timestamp_ns"],
                "simulation_time_s": step["simulation_time_s"],
                "action_digest": step["action"]["action_digest"],
                "native_action_sha256": step["action"]["delivery"]["native_action_sha256"],
                "delivery_receipt_digest": step["action"]["delivery"]["receipt_digest"],
                "observation_digest": step["observation_digest"],
                "native_state_sha256_by_entity_id": step["native_state_sha256_by_entity_id"],
                "state_write_count_after_episode_start_by_entity_id": step[
                    "state_write_count_after_episode_start_by_entity_id"
                ],
                "contact_readback_sha256": _digest(contact_projection),
                "frame_digest_by_camera": {
                    camera_id: step["frames"][camera_id]["frame_digest"] for camera_id in CAMERA_IDS
                },
            }
        )
    return rows


def _manifest_record(
    *, root: Path, schema_version: str, relative_path: str, payload: bytes, rows: object
) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _seal(
        {
            "schema_version": schema_version,
            "relative_path": relative_path,
            "file_size_bytes": len(payload),
            "file_sha256": _bytes_digest(payload),
            "row_count": (
                sum(len(value) for value in rows.values()) if isinstance(rows, dict) else len(rows)
            ),
            "rows_sha256": _bytes_digest(_canonical_bytes(rows)),
        },
        "manifest_digest",
    )


def _attach_manifests(fixture: dict) -> None:
    trace = fixture["trace"]
    root = fixture["root"]
    media_document = _media_document(trace)
    media_rows = media_document["frames_by_camera"]
    trace["media_manifest"] = _manifest_record(
        root=root,
        schema_version=MEDIA_MANIFEST_SCHEMA_VERSION,
        relative_path="manifests/media.json",
        payload=_canonical_bytes(media_document) + b"\n",
        rows=media_rows,
    )
    native_rows = _trace_rows(trace)
    trace["native_trace_manifest"] = _manifest_record(
        root=root,
        schema_version=NATIVE_TRACE_MANIFEST_SCHEMA_VERSION,
        relative_path="manifests/native_trace.jsonl",
        payload=b"".join(_canonical_bytes(row) + b"\n" for row in native_rows),
        rows=native_rows,
    )
    trace["review_videos"] = {}
    for camera_id in CAMERA_IDS:
        video_payload = _H264_MP4_BY_CAMERA_ID[camera_id]
        relative_path = f"media/{camera_id}/review.mp4"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(video_payload)
        frame_digests = [step["frames"][camera_id]["frame_digest"] for step in trace["steps"]]
        trace["review_videos"][camera_id] = _seal(
            {
                "camera_id": camera_id,
                "relative_path": relative_path,
                "size_bytes": len(video_payload),
                "file_sha256": _bytes_digest(video_payload),
                "container": "mp4",
                "codec": "h264",
                "source_media_manifest_digest": trace["media_manifest"]["manifest_digest"],
                "source_frame_digest_sequence_sha256": _bytes_digest(
                    _canonical_bytes(frame_digests)
                ),
                "source_frame_count": len(frame_digests),
                "derivation_tool_id": "fixture_ffmpeg",
                "derivation_tool_sha256": _sha("c"),
                "derivation_command_sha256": _sha("d"),
            },
            "video_receipt_digest",
        )


def _reseal_steps(fixture: dict) -> None:
    trace = fixture["trace"]
    for step in trace["steps"]:
        step["native_state_sha256_by_entity_id"] = _scored_state_digests(step)
        step["observation_digest"] = _digest(step, digest_field="observation_digest")
    trace["terminal"]["terminal_observation_digest"] = trace["steps"][-1]["observation_digest"]
    _attach_manifests(fixture)


def _replace_deformable_reset_state(
    fixture: dict,
    positions: list[list[float]],
    *,
    refreeze: bool,
) -> None:
    reset = fixture["trace"]["reset_receipt"]
    projection = reset["native_reset_state_projection_by_entity_id"]["cloth"]
    velocities = [[0.0, 0.0, 0.0] for _ in positions]
    targets = [[*point, 1.0] for point in positions]
    projection["nodal_positions_world_m"] = copy.deepcopy(positions)
    projection["nodal_velocities_world_mps"] = velocities
    projection["nodal_kinematic_targets"] = targets
    projection["projection_digest"] = _digest(projection, digest_field="projection_digest")
    reset["deformable_nodal_positions_world_m"] = copy.deepcopy(positions)
    reset["deformable_nodal_velocities_world_mps"] = velocities
    reset["deformable_nodal_kinematic_targets"] = targets
    reset["deformable_start_state_sha256"] = projection["projection_digest"]
    reset["native_state_readback_sha256_by_entity_id"]["cloth"] = projection["projection_digest"]
    reset["frozen_reset_state_sha256"] = _reset_projection_set_digest(
        reset["native_reset_state_projection_by_entity_id"]
    )
    reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")
    if refreeze:
        fixture["frozen"]["frozen_deformable_start_state_sha256"] = projection["projection_digest"]
        fixture["frozen"]["frozen_reset_state_sha256"] = reset["frozen_reset_state_sha256"]
        fixture["frozen"]["contract_digest"] = _digest(
            fixture["frozen"], digest_field="contract_digest"
        )


def _replace_robot_reset_state(
    fixture: dict,
    *,
    joint_positions_rad: list[float],
    gripper_width_m: float,
    refreeze: bool,
) -> None:
    reset = fixture["trace"]["reset_receipt"]
    projection = reset["native_reset_state_projection_by_entity_id"]["manipulator_alpha"]
    joint_velocities_rad_s = [0.0 for _ in joint_positions_rad]
    projection["joint_positions_rad"] = list(joint_positions_rad)
    projection["joint_velocities_rad_s"] = joint_velocities_rad_s
    projection["gripper_width_m"] = gripper_width_m
    projection["projection_digest"] = _digest(projection, digest_field="projection_digest")
    reset["robot_joint_positions_rad"] = list(joint_positions_rad)
    reset["robot_joint_velocities_rad_s"] = joint_velocities_rad_s
    reset["gripper_width_m"] = gripper_width_m
    reset["native_state_readback_sha256_by_entity_id"]["manipulator_alpha"] = projection[
        "projection_digest"
    ]
    reset["frozen_reset_state_sha256"] = _reset_projection_set_digest(
        reset["native_reset_state_projection_by_entity_id"]
    )
    reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")
    if refreeze:
        fixture["frozen"]["frozen_reset_state_sha256"] = reset["frozen_reset_state_sha256"]
        fixture["frozen"]["contract_digest"] = _digest(
            fixture["frozen"], digest_field="contract_digest"
        )


def _force_release_before_deformable_displacement(fixture: dict) -> None:
    early_release = fixture["trace"]["steps"][1]
    early_release["entities"]["cloth"]["nodal_positions_world_m"] = [
        list(point) for point in _DEFORMABLE_RESET_POSITIONS
    ]
    early_release["entities"]["cloth"]["contact_pair_count_by_entity_id"]["manipulator_alpha"] = 0
    early_release["entities"]["cloth"]["contact_normal_force_n_by_entity_id"][
        "manipulator_alpha"
    ] = 0.0
    early_release["entities"]["manipulator_alpha"]["gripper_contact_pair_count_by_entity_id"][
        "cloth"
    ] = 0
    early_release["entities"]["manipulator_alpha"]["gripper_contact_normal_force_n_by_entity_id"][
        "cloth"
    ] = 0.0
    _reseal_steps(fixture)


def _remove_post_contact_arm_response(fixture: dict) -> None:
    for step in fixture["trace"]["steps"][1:]:
        action = step["action"]
        action["arm_command"] = [0.0, 0.0]
        action["native_action"] = [0.0, 0.0, action["gripper_delta_command_m"]]
        action["source_output"] = list(action["native_action"])
        action["source_output_sha256"] = _bytes_digest(_canonical_bytes(action["source_output"]))
        if action["policy_inference"] is not None:
            action["policy_inference"]["source_output_sha256"] = action["source_output_sha256"]
            action["policy_inference"]["receipt_digest"] = _digest(
                action["policy_inference"], digest_field="receipt_digest"
            )
        delivery = action["delivery"]
        delivery["native_action_sha256"] = _bytes_digest(_canonical_bytes(action["native_action"]))
        delivery["receipt_digest"] = _digest(delivery, digest_field="receipt_digest")
        action["action_digest"] = _digest(action, digest_field="action_digest")
        step["entities"]["manipulator_alpha"]["arm_joint_positions_rad"] = [0.1, 0.0]
    _reseal_steps(fixture)


def _defer_arm_response_until_release(fixture: dict) -> None:
    for step in fixture["trace"]["steps"][:2]:
        step["entities"]["manipulator_alpha"]["arm_joint_positions_rad"] = [0.0, 0.0]
    _reseal_steps(fixture)


def _replace_gripper_commands(fixture: dict, command_by_sample_index: dict[int, float]) -> None:
    for step in fixture["trace"]["steps"]:
        if step["sample_index"] not in command_by_sample_index:
            continue
        action = step["action"]
        gripper_command = command_by_sample_index[step["sample_index"]]
        action["gripper_delta_command_m"] = gripper_command
        action["native_action"] = [*action["arm_command"], gripper_command]
        action["source_output"] = list(action["native_action"])
        action["source_output_sha256"] = _bytes_digest(_canonical_bytes(action["source_output"]))
        if action["policy_inference"] is not None:
            action["policy_inference"]["source_output_sha256"] = action["source_output_sha256"]
            action["policy_inference"]["receipt_digest"] = _digest(
                action["policy_inference"], digest_field="receipt_digest"
            )
        delivery = action["delivery"]
        delivery["native_action_sha256"] = _bytes_digest(_canonical_bytes(action["native_action"]))
        delivery["receipt_digest"] = _digest(delivery, digest_field="receipt_digest")
        action["action_digest"] = _digest(action, digest_field="action_digest")
    _reseal_steps(fixture)


def _fixture(
    root: Path,
    kind: str = "learned_policy_evaluation",
    *,
    family: str = "canonical",
    move: bool = True,
    delivered: bool = True,
    successful_state: bool | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entities = _entities()
    spec = _task_spec()
    prompt = "Pick up the cloth, place it inside the basket, release it, and retreat."
    cell = _cell(family)
    actor = _actor(kind)
    start_ns = 1_000_000_000
    if successful_state is None:
        successful_state = kind != "zero_action_control"
    reset = _reset(start_ns=start_ns, actor=actor, entities=entities)
    steps = [
        _step(
            root=root,
            index=index,
            kind=kind,
            start_ns=start_ns,
            move=move,
            delivered=delivered,
            successful_state=successful_state,
        )
        for index in range(4)
    ]
    trace = {
        "episode_id": f"fixture-{family}-{kind}",
        "episode_kind": kind,
        "episode_start_timestamp_ns": start_ns,
        "prompt": prompt,
        "prompt_sha256": _bytes_digest(prompt.encode()),
        "cell": cell,
        "trace_thresholds": _thresholds(),
        "actor": actor,
        "reset_receipt": reset,
        "steps": steps,
        "terminal": {
            "status": "complete",
            "terminal_step_index": 3,
            "terminal_timestamp_ns": steps[-1]["timestamp_ns"] + 100,
            "terminal_observation_digest": steps[-1]["observation_digest"],
            "media_gap": None,
        },
    }
    fixture = {
        "root": root,
        "entities": entities,
        "spec": spec,
        "frozen": _frozen_run(
            entities=entities,
            spec=spec,
            prompt=prompt,
            cell=cell,
            reset=reset,
        ),
        "trace": trace,
    }
    _attach_manifests(fixture)
    return fixture


def _authorize(fixture: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    authority_root = fixture["root"] / "authority"
    authority_root.mkdir(parents=True, exist_ok=True)
    event_relative_path = "authority/native_event.json"
    event_bytes = _canonical_bytes(fixture["trace"]) + b"\n"
    (fixture["root"] / event_relative_path).write_bytes(event_bytes)

    private_key = Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    runner_fingerprint = _bytes_digest(public_key)
    lifecycle = {
        "admission_receipt": _sha("1"),
        "allocation_receipt": _sha("2"),
        "teardown_receipt": _sha("3"),
        "watchdog_receipt": _sha("4"),
    }
    payload = materialize_trusted_execution_payload(
        nonce="native-episode-runner-nonce-0001",
        run_digest=fixture["frozen"]["contract_digest"],
        package_digest=_sha("5"),
        execution_request_digest=_sha("6"),
        worker_entrypoint="python -m blueprint_pipeline.native_deformable_worker",
        worker_source_tree_digest=_sha("7"),
        worker_container_digest=_sha("8"),
        instance_id="vast-native-episode-fixture",
        return_zip_sha256=_bytes_digest(event_bytes),
        return_zip_size_bytes=len(event_bytes),
        started_at="2026-08-10T10:00:00Z",
        ended_at="2026-08-10T10:01:00Z",
        allocator_lifecycle_artifact_digests=lifecycle,
    )
    envelope = materialize_trusted_execution_envelope(
        payload=payload,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        signature_base64=base64.b64encode(
            private_key.sign(trusted_execution_signature_message(payload))
        ).decode("ascii"),
    )
    envelope_relative_path = "authority/trusted_execution_envelope.json"
    (fixture["root"] / envelope_relative_path).write_bytes(
        canonical_trusted_execution_envelope_bytes(envelope)
    )
    seal = _seal(
        {
            "schema_version": "native_deformable_frozen_run_seal.v1",
            "frozen_run_contract_digest": fixture["frozen"]["contract_digest"],
            "native_event_sha256": _bytes_digest(event_bytes),
            "native_event_size_bytes": len(event_bytes),
            "trusted_runner_public_key_sha256": runner_fingerprint,
            "trusted_execution": {
                "nonce": payload["nonce"],
                "package_digest": payload["package_digest"],
                "execution_request_digest": payload["execution_request_digest"],
                "worker_entrypoint": payload["worker"]["entrypoint"],
                "worker_source_tree_digest": payload["worker"]["source_tree_digest"],
                "worker_container_digest": payload["worker"]["container_digest"],
                "instance_id": payload["instance_id"],
                "allocator_lifecycle_artifact_digests": lifecycle,
            },
        },
        "seal_digest",
    )
    seal_bytes = _canonical_bytes(seal) + b"\n"
    seal_relative_path = "authority/frozen_run_seal.json"
    (fixture["root"] / seal_relative_path).write_bytes(seal_bytes)
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, runner_fingerprint)
    fixture["authority"] = {
        "frozen_run_seal_relative_path": seal_relative_path,
        "expected_frozen_run_seal_sha256": _bytes_digest(seal_bytes),
        "trusted_execution_envelope_relative_path": envelope_relative_path,
        "native_event_relative_path": event_relative_path,
    }


def _materialize_args(fixture: dict) -> dict:
    return {
        "task_entities": fixture["entities"],
        "task_spec": fixture["spec"],
        "frozen_run_contract": fixture["frozen"],
        "episode_trace": fixture["trace"],
        "evidence_root": fixture["root"],
        **fixture.get("authority", {}),
    }


def _materialize(fixture: dict) -> dict:
    return materialize_native_deformable_episode_trace(**_materialize_args(fixture))


@pytest.mark.parametrize("family", ["canonical", "held_out_composed_relocation"])
def test_policy_trace_rehashes_real_media_and_is_task_entity_neutral(
    tmp_path: Path, family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path / family, family=family)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)

    assert receipt["cell"]["family"] == family
    assert receipt["entity_ids"]["robot"] == "manipulator_alpha"
    assert len(receipt["task_entities"]) == 6
    assert receipt["semantic_role_index"]["obstacle"] == ["pillar", "wall"]
    assert receipt["deterministic_task_state_score"]["deterministic_success"] is True
    assert receipt["deterministic_task_state_score"]["predicates"]["grasp_contact_observed"] is True
    assert (
        receipt["deterministic_task_state_score"]["measurements"][
            "maximum_grasp_contact_pair_count"
        ]
        == 1
    )
    assert receipt["native_episode_admitted_deterministic_success"] is True
    assert receipt["evaluation_admitted_deterministic_success"] is False
    assert receipt["evidence"]["interpretability"]["policy_outcome_interpretable"] is True
    assert receipt["evidence"]["interpretability"]["manipulation_sequence_complete"] is True
    assert (
        receipt["evidence"]["deformable_motion"]["initial_deformable_outside_destination"] is True
    )
    assert (
        receipt["evidence"]["deformable_motion"][
            "ordered_contact_displacement_release_final_settle"
        ]
        is True
    )
    assert receipt["evidence"]["deformable_motion"][
        "task_relevant_displacement_sample_indices"
    ] == [1, 2, 3]
    assert receipt["policy_outcome"] is None
    assert (
        receipt["deterministic_score_claim_status"] == "trusted_runner_attested_native_state_score"
    )
    assert (
        receipt["claim_boundary"]
        == "trusted_runner_native_simulator_trace_only_not_physical_material_"
        "equivalence_real_robot_performance_or_deployment_truth"
    )
    assert receipt["native_event_authority"]["native_event_authority_verified"] is True
    assert receipt["media_complete"] is True
    verified_decode = receipt["review_videos"]["external"]["verifier_owned_decode"]
    assert verified_decode["contract_id"] == H264_DECODE_VERIFIER_CONTRACT_ID
    assert verified_decode["decoded_sample_count"] == 4
    assert verified_decode["width_px"] == 8
    assert verified_decode["height_px"] == 6
    assert verified_decode["exact_lossless_rgb_correspondence"] is True
    assert verified_decode["raw_rgb_sha256_by_sample"] == [
        step["frames"]["external"]["raw_rgb_sha256"] for step in receipt["steps"]
    ]
    assert receipt["overview_used_by_policy"] is False
    assert receipt["overview_used_by_deterministic_scorer"] is False
    assert receipt["media_manifest"]["row_count"] == 12
    assert receipt["native_trace_manifest"]["row_count"] == 4
    assert set(fixture["frozen"]["candidate_identity_digest_by_id"]) == {
        "pi05_droid",
        "groot_n17_droid",
    }
    assert receipt["receipt_digest"] == _digest(receipt, digest_field="receipt_digest")


def test_zero_action_is_interpretable_only_as_required_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, "zero_action_control", move=False)
    _authorize(fixture, monkeypatch)
    receipt = _materialize(fixture)

    assert receipt["deterministic_task_state_score"]["deterministic_success"] is False
    assert (
        receipt["deterministic_task_state_score"]["predicates"]["grasp_contact_observed"] is False
    )
    assert (
        receipt["deterministic_task_state_score"]["measurements"][
            "maximum_grasp_contact_pair_count"
        ]
        == 0
    )
    assert receipt["evidence"]["robot_motion"]["arm_moved"] is False
    assert receipt["evidence"]["interpretability"]["zero_control_outcome_interpretable"] is True
    assert receipt["control_outcome"] == "required_zero_action_failure_observed"
    assert receipt["evaluation_admitted_deterministic_success"] is False
    assert receipt["policy_outcome"] is None
    assert (
        receipt["deterministic_score_claim_status"] == "trusted_runner_attested_native_state_score"
    )


def test_static_contained_cloth_cannot_launder_a_transfer_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    static_positions = [list(point) for point in _DEFORMABLE_CONTAINED_POSITIONS]
    _replace_deformable_reset_state(fixture, static_positions, refreeze=True)
    for step in fixture["trace"]["steps"]:
        step["entities"]["cloth"]["nodal_positions_world_m"] = copy.deepcopy(static_positions)
    _reseal_steps(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)

    assert receipt["deterministic_task_state_score"]["deterministic_success"] is True
    motion = receipt["evidence"]["deformable_motion"]
    assert motion["initial_deformable_outside_destination"] is False
    assert motion["maximum_centroid_displacement_from_reset_m"] == 0.0
    assert motion["maximum_post_contact_centroid_displacement_m"] == 0.0
    assert motion["maximum_nodal_displacement_from_reset_m"] == 0.0
    assert motion["maximum_post_contact_nodal_displacement_m"] == 0.0
    assert motion["task_relevant_displacement_sample_indices"] == []
    assert motion["ordered_contact_displacement_release_final_settle"] is False
    assert receipt["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert receipt["native_episode_admitted_deterministic_success"] is False


def test_deformable_displacement_threshold_has_a_physical_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    fixture["trace"]["trace_thresholds"]["minimum_deformable_displacement_m"] = 1.0e-6
    fixture["frozen"]["trace_thresholds_sha256"] = _digest(fixture["trace"]["trace_thresholds"])
    fixture["frozen"]["contract_digest"] = _digest(
        fixture["frozen"], digest_field="contract_digest"
    )
    _authorize(fixture, monkeypatch)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert "deformable_episode_minimum_deformable_displacement_invalid" in exc_info.value.errors


@pytest.mark.parametrize(
    ("threshold_key", "minimum_floor", "expected_error"),
    [
        (
            "arm_motion_epsilon_rad",
            MINIMUM_ARM_MOTION_EPSILON_RAD,
            "deformable_episode_arm_motion_epsilon_invalid",
        ),
        (
            "gripper_motion_epsilon_m",
            MINIMUM_GRIPPER_MOTION_EPSILON_M,
            "deformable_episode_gripper_motion_epsilon_invalid",
        ),
    ],
)
def test_robot_response_thresholds_have_nonoverridable_physical_floors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threshold_key: str,
    minimum_floor: float,
    expected_error: str,
) -> None:
    fixture = _fixture(tmp_path, move=False, successful_state=False)
    fixture["trace"]["trace_thresholds"][threshold_key] = minimum_floor / 10.0
    fixture["frozen"]["trace_thresholds_sha256"] = _digest(fixture["trace"]["trace_thresholds"])
    fixture["frozen"]["contract_digest"] = _digest(
        fixture["frozen"], digest_field="contract_digest"
    )
    for index, step in enumerate(fixture["trace"]["steps"]):
        step["entities"]["manipulator_alpha"]["arm_joint_positions_rad"] = [
            (index + 1) * 1.0e-12,
            0.0,
        ]
        step["entities"]["manipulator_alpha"]["gripper_width_m"] = (
            0.08 - 1.0e-12 if index == 0 else 0.08
        )
    _reseal_steps(fixture)
    _authorize(fixture, monkeypatch)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert expected_error in exc_info.value.errors


def test_subfloor_arm_and_gripper_motion_cannot_prove_action_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    for index, step in enumerate(fixture["trace"]["steps"]):
        step["entities"]["manipulator_alpha"]["arm_joint_positions_rad"] = [
            0.9 * MINIMUM_ARM_MOTION_EPSILON_RAD,
            0.0,
        ]
        step["entities"]["manipulator_alpha"]["gripper_width_m"] = (
            0.08 - 0.9 * MINIMUM_GRIPPER_MOTION_EPSILON_M if index in {0, 1} else 0.08
        )
    _reseal_steps(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)

    assert receipt["evidence"]["robot_motion"]["arm_moved"] is False
    assert receipt["evidence"]["robot_motion"]["gripper_responded"] is False
    assert receipt["evidence"]["action_delivery"]["actions_reached_robot"] is False
    assert receipt["evidence"]["interpretability"]["policy_outcome_interpretable"] is False
    assert receipt["native_episode_admitted_deterministic_success"] is False
    assert receipt["policy_outcome"] is None


def test_zero_action_object_in_destination_does_not_invent_prior_grasp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(
        tmp_path,
        "zero_action_control",
        move=False,
        successful_state=True,
    )
    _authorize(fixture, monkeypatch)
    receipt = _materialize(fixture)

    score = receipt["deterministic_task_state_score"]
    assert score["predicates"]["contained"] is True
    assert score["predicates"]["grasp_contact_observed"] is False
    assert score["deterministic_success"] is False
    assert score["ladder_truncated_at"] == "grasp_contact_observed"
    assert receipt["control_outcome"] == "required_zero_action_failure_observed"
    assert receipt["evaluation_admitted_deterministic_success"] is False


def test_scripted_positive_uses_same_delivery_contact_release_and_retreat_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, "scripted_positive_control")
    _authorize(fixture, monkeypatch)
    receipt = _materialize(fixture)

    assert receipt["control_outcome"] == "required_scripted_positive_success_observed"
    assert receipt["deterministic_task_state_score"]["predicates"]["grasp_contact_observed"] is True
    assert (
        receipt["deterministic_task_state_score"]["measurements"]["maximum_grasp_contact_force_n"]
        == 2.0
    )
    assert receipt["evidence"]["interpretability"]["scripted_control_outcome_interpretable"] is True
    assert receipt["native_episode_admitted_deterministic_success"] is True
    assert receipt["evaluation_admitted_deterministic_success"] is False


@pytest.mark.parametrize(
    "episode_kind",
    ["scripted_positive_control", "learned_policy_evaluation"],
)
def test_release_before_deformable_displacement_cannot_complete_ordered_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    episode_kind: str,
) -> None:
    fixture = _fixture(tmp_path, episode_kind)
    _force_release_before_deformable_displacement(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)
    motion = receipt["evidence"]["deformable_motion"]

    assert receipt["deterministic_task_state_score"]["deterministic_success"] is True
    assert receipt["evidence"]["contact_evidence"]["post_contact_release_sample_indices"] == [
        1,
        2,
        3,
    ]
    assert motion["task_relevant_displacement_sample_indices"] == [2, 3]
    assert motion["first_post_contact_release_sample_index"] == 1
    assert motion["first_task_relevant_displacement_sample_index"] == 2
    assert motion["settle_window_start_sample_index"] == 2
    assert motion["ordered_contact_displacement_release_final_settle"] is False
    assert receipt["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert receipt["native_episode_admitted_deterministic_success"] is False
    if episode_kind == "scripted_positive_control":
        assert receipt["control_outcome"] == "scripted_positive_harness_task_construction_blocker"


def test_never_moved_or_undelivered_learned_trace_stays_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never_fixture = _fixture(tmp_path / "never", move=False)
    delivery_fixture = _fixture(tmp_path / "delivery", delivered=False)
    _authorize(never_fixture, monkeypatch)
    never_moved = _materialize(never_fixture)
    _authorize(delivery_fixture, monkeypatch)
    undelivered = _materialize(delivery_fixture)

    assert never_moved["deterministic_task_state_score"]["deterministic_success"] is False
    assert (
        never_moved["deterministic_task_state_score"]["predicates"]["grasp_contact_observed"]
        is False
    )
    assert never_moved["evaluation_admitted_deterministic_success"] is False
    assert never_moved["policy_outcome"] is None
    assert (
        never_moved["evidence"]["interpretability"]["interpretation"]
        == "arm_motion_not_observed_policy_outcome_uninterpretable"
    )
    assert undelivered["evaluation_admitted_deterministic_success"] is False
    assert undelivered["policy_outcome"] is None
    assert (
        undelivered["evidence"]["interpretability"]["interpretation"]
        == "action_delivery_not_proven_harness_fault"
    )


def test_motion_and_containment_without_native_grasp_contact_cannot_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    for step in fixture["trace"]["steps"]:
        step["entities"]["cloth"]["contact_pair_count_by_entity_id"]["manipulator_alpha"] = 0
        step["entities"]["cloth"]["contact_normal_force_n_by_entity_id"]["manipulator_alpha"] = 0.0
        step["entities"]["manipulator_alpha"]["gripper_contact_pair_count_by_entity_id"][
            "cloth"
        ] = 0
        step["entities"]["manipulator_alpha"]["gripper_contact_normal_force_n_by_entity_id"][
            "cloth"
        ] = 0.0
    _reseal_steps(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)
    score = receipt["deterministic_task_state_score"]

    assert receipt["native_event_authority"]["native_event_authority_verified"] is True
    assert receipt["evidence"]["interpretability"]["policy_outcome_interpretable"] is True
    assert score["predicates"]["contained"] is True
    assert score["predicates"]["grasp_contact_observed"] is False
    assert score["deterministic_success"] is False
    assert score["ladder_truncated_at"] == "grasp_contact_observed"
    assert "qualified_gripper_deformable_contact_not_observed" in score["failure_reasons"]
    assert receipt["native_episode_admitted_deterministic_success"] is False
    assert receipt["evaluation_admitted_deterministic_success"] is False


def test_cross_sample_contact_pair_and_force_cannot_launder_a_grasp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    first = fixture["trace"]["steps"][0]["entities"]
    first["cloth"]["contact_normal_force_n_by_entity_id"]["manipulator_alpha"] = 0.0
    first["manipulator_alpha"]["gripper_contact_normal_force_n_by_entity_id"]["cloth"] = 0.0
    second = fixture["trace"]["steps"][1]["entities"]
    second["cloth"]["contact_pair_count_by_entity_id"]["manipulator_alpha"] = 0
    second["cloth"]["contact_normal_force_n_by_entity_id"]["manipulator_alpha"] = 2.0
    second["manipulator_alpha"]["gripper_contact_pair_count_by_entity_id"]["cloth"] = 0
    second["manipulator_alpha"]["gripper_contact_normal_force_n_by_entity_id"]["cloth"] = 2.0
    _reseal_steps(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)
    score = receipt["deterministic_task_state_score"]

    assert score["measurements"]["maximum_grasp_contact_pair_count"] == 1
    assert score["measurements"]["maximum_grasp_contact_force_n"] == 2.0
    assert score["measurements"]["qualifying_grasp_contact_sample_indices"] == []
    assert score["predicates"]["grasp_contact_observed"] is False
    assert score["deterministic_success"] is False
    assert receipt["evidence"]["contact_evidence"]["bilateral_contact_sample_indices"] == []
    assert receipt["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert receipt["native_episode_admitted_deterministic_success"] is False


def test_static_clearance_without_post_contact_arm_response_is_not_retreat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _remove_post_contact_arm_response(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)

    assert receipt["deterministic_task_state_score"]["deterministic_success"] is True
    assert receipt["evidence"]["interpretability"]["policy_outcome_interpretable"] is True
    assert receipt["evidence"]["robot_motion"]["retreat_response_observed_after_contact"] is False
    assert receipt["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert receipt["evaluation_admitted_deterministic_success"] is False


@pytest.mark.parametrize("mutation", ["direct_write", "obstacle_write", "kinematic", "attachment"])
def test_direct_writes_and_hidden_attachments_cannot_be_admitted(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    for step in fixture["trace"]["steps"]:
        if mutation == "direct_write":
            step["entities"]["cloth"]["state_write_count_after_episode_start"] = 1
            step["state_write_count_after_episode_start_by_entity_id"]["cloth"] = 1
        elif mutation == "obstacle_write":
            step["state_write_count_after_episode_start_by_entity_id"]["pillar"] = 1
        elif mutation == "kinematic":
            step["entities"]["cloth"]["nodal_kinematic_flags"] = [0.0] * 4
        else:
            step["entities"]["cloth"]["hidden_attachment_active"] = True
            step["entities"]["manipulator_alpha"][
                "gripper_attachment_constraint_count_by_entity_id"
            ]["cloth"] = 1
    _reseal_steps(fixture)

    receipt = _materialize(fixture)

    assert receipt["evidence"]["integrity"]["manipulation_integrity_valid"] is False
    assert receipt["evaluation_admitted_deterministic_success"] is False
    assert receipt["policy_outcome"] is None


def test_nonexistent_counterfeit_and_symlink_frame_paths_fail_closed(
    tmp_path: Path,
) -> None:
    missing = _fixture(tmp_path / "missing")
    frame = missing["trace"]["steps"][0]["frames"]["external"]
    frame["relative_path"] = "media/external/nonexistent.png"
    frame["frame_digest"] = _digest(frame, digest_field="frame_digest")
    _reseal_steps(missing)
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(missing)
    assert "deformable_episode_frame_file_invalid:0:external" in exc_info.value.errors

    counterfeit = _fixture(tmp_path / "counterfeit")
    path = (
        counterfeit["root"]
        / counterfeit["trace"]["steps"][0]["frames"]["external"]["relative_path"]
    )
    path.write_bytes(b"not-the-sealed-png")
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(counterfeit)
    assert "deformable_episode_frame_file_mismatch:0:external" in exc_info.value.errors

    symlink = _fixture(tmp_path / "symlink")
    frame_path = (
        symlink["root"] / symlink["trace"]["steps"][0]["frames"]["external"]["relative_path"]
    )
    target = symlink["root"] / "alternate.png"
    target.write_bytes(frame_path.read_bytes())
    frame_path.unlink()
    frame_path.symlink_to(target)
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(symlink)
    assert (
        "deformable_episode_frame_file_invalid:0:external_symlink_forbidden"
        in exc_info.value.errors
    )


def test_actual_media_and_native_trace_manifest_bytes_cannot_be_counterfeited(
    tmp_path: Path,
) -> None:
    media = _fixture(tmp_path / "media")
    media_path = media["root"] / media["trace"]["media_manifest"]["relative_path"]
    media_path.write_bytes(media_path.read_bytes() + b" ")
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(media)
    assert "deformable_episode_media_manifest_invalid_content_mismatch" in exc_info.value.errors

    native = _fixture(tmp_path / "native")
    rows = _trace_rows(native["trace"])
    rows[0]["contact_readback_sha256"] = _sha("f")
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    native["trace"]["native_trace_manifest"] = _manifest_record(
        root=native["root"],
        schema_version=NATIVE_TRACE_MANIFEST_SCHEMA_VERSION,
        relative_path="manifests/native_trace.jsonl",
        payload=payload,
        rows=rows,
    )
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(native)
    assert (
        "deformable_episode_native_trace_manifest_invalid_content_mismatch" in exc_info.value.errors
    )


def test_unilateral_contact_and_native_action_hashes_fail_closed(tmp_path: Path) -> None:
    contact = _fixture(tmp_path / "contact")
    first = contact["trace"]["steps"][0]
    first["entities"]["cloth"]["contact_pair_count_by_entity_id"]["manipulator_alpha"] = 2
    _reseal_steps(contact)
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(contact)
    assert "deformable_episode_bilateral_contact_mismatch:0" in exc_info.value.errors

    action = _fixture(tmp_path / "action")
    delivery = action["trace"]["steps"][0]["action"]["delivery"]
    delivery["native_action_sha256"] = _sha("f")
    delivery["receipt_digest"] = _digest(delivery, digest_field="receipt_digest")
    action["trace"]["steps"][0]["action"]["action_digest"] = _digest(
        action["trace"]["steps"][0]["action"], digest_field="action_digest"
    )
    _reseal_steps(action)
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(action)
    assert "deformable_episode_native_action_digest_mismatch:0" in exc_info.value.errors

    source_output = _fixture(tmp_path / "source-output")
    first_action = source_output["trace"]["steps"][0]["action"]
    first_action["source_output_sha256"] = _sha("f")
    first_action["action_digest"] = _digest(first_action, digest_field="action_digest")
    _reseal_steps(source_output)
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(source_output)
    assert "deformable_episode_source_output_digest_mismatch:0" in exc_info.value.errors


@pytest.mark.parametrize(
    "join", ["cell", "candidate", "reset", "deformable_start", "entity", "task"]
)
def test_every_frozen_identity_join_rejects_post_freeze_substitution(
    tmp_path: Path, join: str
) -> None:
    fixture = _fixture(tmp_path)
    if join == "cell":
        fixture["trace"]["cell"]["seed"] += 1
    elif join == "candidate":
        actor = fixture["trace"]["actor"]
        actor["model_seed"] += 1
        actor["identity_digest"] = _digest(actor, digest_field="identity_digest")
        reset = fixture["trace"]["reset_receipt"]
        reset["actor_seed"] = actor["model_seed"]
        reset["actor_identity_digest"] = actor["identity_digest"]
        reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")
    elif join == "reset":
        reset = fixture["trace"]["reset_receipt"]
        reset["frozen_reset_state_sha256"] = _sha("f")
        reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")
    elif join == "deformable_start":
        reset = fixture["trace"]["reset_receipt"]
        positions = copy.deepcopy(reset["deformable_nodal_positions_world_m"])
        positions[0][0] += 0.1
        _replace_deformable_reset_state(fixture, positions, refreeze=False)
    elif join == "entity":
        fixture["entities"].append(_entity("screen", "obstacle", "c"))
    else:
        fixture["spec"]["minimum_particle_fraction_inside"] = 0.8

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    expected = {
        "cell": "deformable_episode_frozen_cell_identity_mismatch",
        "candidate": "deformable_episode_frozen_actor_identity_mismatch",
        "reset": "deformable_episode_frozen_reset_state_mismatch",
        "deformable_start": "deformable_episode_reset_deformable_start_state_mismatch",
        "entity": "deformable_episode_frozen_entity_contract_mismatch",
        "task": "deformable_episode_frozen_task_spec_mismatch",
    }[join]
    assert expected in exc_info.value.errors


def test_signed_frozen_run_with_only_pi05_candidate_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    candidate_digest = fixture["frozen"]["candidate_identity_digest_by_id"].pop("groot_n17_droid")
    del fixture["frozen"]["action_replay_contract_by_actor_identity_digest"][candidate_digest]
    fixture["frozen"]["contract_digest"] = _digest(
        fixture["frozen"], digest_field="contract_digest"
    )
    _authorize(fixture, monkeypatch)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert "deformable_episode_frozen_run_contract_invalid" in exc_info.value.errors


def test_signed_frozen_run_with_duplicate_candidate_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    candidates = fixture["frozen"]["candidate_identity_digest_by_id"]
    groot_digest = candidates["groot_n17_droid"]
    candidates["groot_n17_droid"] = candidates["pi05_droid"]
    del fixture["frozen"]["action_replay_contract_by_actor_identity_digest"][groot_digest]
    fixture["frozen"]["contract_digest"] = _digest(
        fixture["frozen"], digest_field="contract_digest"
    )
    _authorize(fixture, monkeypatch)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert "deformable_episode_frozen_run_contract_invalid" in exc_info.value.errors


def test_policy_reset_seed_and_all_entity_readbacks_are_mandatory(tmp_path: Path) -> None:
    reset_missing = _fixture(tmp_path / "reset")
    reset = reset_missing["trace"]["reset_receipt"]
    reset["actor_reset_invoked"] = False
    reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(reset_missing)
    assert "deformable_episode_actor_reset_not_proven" in exc_info.value.errors

    missing_entity = _fixture(tmp_path / "entity")
    reset = missing_entity["trace"]["reset_receipt"]
    del reset["native_reset_write_count_by_entity_id"]["pillar"]
    del reset["native_state_readback_sha256_by_entity_id"]["pillar"]
    reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(missing_entity)
    assert "deformable_episode_reset_entity_set_mismatch" in exc_info.value.errors


def test_robot_reset_numeric_values_cannot_borrow_an_unchanged_projection_digest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    reset = fixture["trace"]["reset_receipt"]
    reset["robot_joint_positions_rad"] = [0.03, 0.0]
    reset["robot_joint_velocities_rad_s"] = [0.0, 0.0]
    reset["gripper_width_m"] = 0.05
    reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert "deformable_episode_reset_robot_projection_mismatch" in exc_info.value.errors


@pytest.mark.parametrize(
    ("attack", "expected_error"),
    [
        (
            "deformable_positions",
            "deformable_episode_reset_readback_projection_mismatch:cloth",
        ),
        (
            "deformable_velocity",
            "deformable_episode_reset_state_projection_invalid:cloth",
        ),
        (
            "deformable_kinematic_target",
            "deformable_episode_reset_state_projection_invalid:cloth",
        ),
        (
            "robot_joints",
            "deformable_episode_reset_readback_projection_mismatch:manipulator_alpha",
        ),
        (
            "robot_velocity",
            "deformable_episode_reset_state_projection_invalid:manipulator_alpha",
        ),
        (
            "robot_gripper",
            "deformable_episode_reset_readback_projection_mismatch:manipulator_alpha",
        ),
    ],
)
def test_self_rehashed_numeric_reset_projection_cannot_borrow_old_readback_digest(
    tmp_path: Path,
    attack: str,
    expected_error: str,
) -> None:
    fixture = _fixture(tmp_path)
    reset = fixture["trace"]["reset_receipt"]
    projections = reset["native_reset_state_projection_by_entity_id"]
    if attack.startswith("deformable"):
        projection = projections["cloth"]
        if attack == "deformable_positions":
            projection["nodal_positions_world_m"][0][0] += 0.01
            projection["nodal_kinematic_targets"][0][0] += 0.01
        elif attack == "deformable_velocity":
            projection["nodal_velocities_world_mps"][0][0] = 0.01
        else:
            projection["nodal_kinematic_targets"][0][3] = 0.0
    else:
        projection = projections["manipulator_alpha"]
        if attack == "robot_joints":
            projection["joint_positions_rad"][0] = 0.03
        elif attack == "robot_velocity":
            projection["joint_velocities_rad_s"][0] = 0.01
        else:
            projection["gripper_width_m"] = 0.05
    projection["projection_digest"] = _digest(projection, digest_field="projection_digest")
    reset["receipt_digest"] = _digest(reset, digest_field="receipt_digest")

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert expected_error in exc_info.value.errors


def test_overview_is_rehashed_but_never_policy_input_or_scorer_input(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    overview = fixture["trace"]["steps"][0]["frames"]["overview"]
    overview["presented_to_actor"] = True
    overview["frame_digest"] = _digest(overview, digest_field="frame_digest")
    _reseal_steps(fixture)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert "deformable_episode_camera_role_invalid:0:overview" in exc_info.value.errors


def test_pre_observation_failure_is_typed_and_never_a_policy_outcome(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    trace = fixture["trace"]
    trace["steps"] = []
    trace["media_manifest"] = None
    trace["native_trace_manifest"] = None
    failure_log = b"native runtime import failed before first camera observation\n"
    failure_log_path = fixture["root"] / "logs/pre_observation_failure.log"
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.write_bytes(failure_log)
    trace["terminal"] = {
        "status": "failed_before_first_observation",
        "terminal_timestamp_ns": trace["episode_start_timestamp_ns"] + 10,
        "failure_type": "native_runtime_import_failure",
        "failure_stage": "first_camera_render",
        "media_gap": _seal(
            {
                "gap_type": "no_frames_before_first_observation",
                "required_camera_ids": sorted(CAMERA_IDS),
                "observation_count": 0,
                "first_observation_attempted": True,
                "failure_log_relative_path": "logs/pre_observation_failure.log",
                "failure_log_size_bytes": len(failure_log),
                "failure_log_sha256": _bytes_digest(failure_log),
            },
            "gap_receipt_digest",
        ),
    }

    receipt = _materialize(fixture)

    assert receipt["media_status"] == "typed_gap_before_first_observation"
    assert receipt["deterministic_task_state_score"] is None
    assert receipt["policy_outcome"] is None
    assert receipt["evaluation_admitted_deterministic_success"] is False

    failure_log_path.write_bytes(b"counterfeit replacement log\n")
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)
    assert "deformable_episode_terminal_media_gap_invalid" in exc_info.value.errors


def test_caller_grades_and_raw_state_digest_substitution_are_rejected(
    tmp_path: Path,
) -> None:
    grade = _fixture(tmp_path / "grade")
    grade["trace"]["policy_grade"] = "success"
    with pytest.raises(
        NativeDeformableEpisodeTraceError,
        match="deformable_episode_caller_authored_grade_forbidden",
    ):
        _materialize(grade)

    state = _fixture(tmp_path / "state")
    state["trace"]["steps"][0]["native_state_sha256_by_entity_id"]["cloth"] = _sha("f")
    state["trace"]["steps"][0]["observation_digest"] = _digest(
        state["trace"]["steps"][0], digest_field="observation_digest"
    )
    _attach_manifests(state)
    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(state)
    assert "deformable_episode_native_state_digest_mismatch:0:cloth" in exc_info.value.errors


def test_fixture_mutations_are_isolated(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "first")
    second = copy.deepcopy(first)
    second["trace"]["steps"][0]["frames"]["external"]["relative_path"] = "changed.png"
    assert (
        first["trace"]["steps"][0]["frames"]["external"]["relative_path"]
        != second["trace"]["steps"][0]["frames"]["external"]["relative_path"]
    )


def test_structural_caller_trace_is_never_native_or_evaluation_authority(
    tmp_path: Path,
) -> None:
    receipt = _materialize(_fixture(tmp_path))

    assert receipt["deterministic_task_state_score"]["deterministic_success"] is True
    assert receipt["native_event_authority"] == {
        "status": "untrusted_structural_candidate",
        "native_event_authority_verified": False,
        "frozen_run_seal_sha256": None,
        "trusted_runner_public_key_sha256": None,
        "native_event_sha256": None,
        "native_event_size_bytes": None,
        "blockers": ["trusted_native_event_authority_missing"],
        "claim_scope": "untrusted_structural_candidate_only",
        "does_not_establish": [
            "native_simulator_execution",
            "provider_zero",
            "physical_truth",
        ],
    }
    assert receipt["native_episode_admitted_deterministic_success"] is False
    assert receipt["evaluation_admitted_deterministic_success"] is False
    assert receipt["policy_outcome"] is None
    assert receipt["deterministic_score_claim_status"] == "untrusted_structural_projection"
    assert (
        receipt["claim_boundary"]
        == "untrusted_structural_candidate_no_native_simulator_or_evaluation_claim"
    )
    assert receipt["media_complete"] is False
    assert receipt["media_status"] == "byte_verified_media_untrusted_native_trace"


@pytest.mark.parametrize("substitution", ["bool_as_int", "float_as_int"])
def test_signed_native_event_requires_exact_canonical_source_trace_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    fixture = _fixture(tmp_path / substitution)
    _authorize(fixture, monkeypatch)
    signed_trace = json.loads(
        (fixture["root"] / fixture["authority"]["native_event_relative_path"]).read_text()
    )
    if substitution == "bool_as_int":
        signed_value = signed_trace["steps"][0]["action"]["delivery"]["attempted"]
        fixture["trace"]["steps"][0]["action"]["delivery"]["attempted"] = 1
        caller_value = fixture["trace"]["steps"][0]["action"]["delivery"]["attempted"]
    else:
        signed_value = signed_trace["steps"][0]["entities"]["cloth"]["nodal_positions_world_m"][0][
            2
        ]
        fixture["trace"]["steps"][0]["entities"]["cloth"]["nodal_positions_world_m"][0][2] = 0
        caller_value = fixture["trace"]["steps"][0]["entities"]["cloth"]["nodal_positions_world_m"][
            0
        ][2]
    assert signed_value == caller_value
    assert type(signed_value) is not type(caller_value)

    authority = episode_module._native_authority(
        source_trace=fixture["trace"],
        frozen_run=fixture["frozen"],
        evidence_root=fixture["root"].resolve(),
        **fixture["authority"],
    )

    assert authority["native_event_authority_verified"] is False
    assert "native_event_exact_trace_bytes_mismatch" in authority["blockers"]
    if substitution == "float_as_int":
        receipt = _materialize(fixture)
        assert receipt["native_event_authority"]["native_event_authority_verified"] is False
        assert receipt["native_episode_admitted_deterministic_success"] is False


@pytest.mark.parametrize("contract", ["thresholds", "camera", "renderer"])
def test_threshold_camera_and_renderer_contracts_are_frozen(tmp_path: Path, contract: str) -> None:
    fixture = _fixture(tmp_path)
    if contract == "thresholds":
        fixture["trace"]["trace_thresholds"]["action_epsilon"] = 2.0e-6
    elif contract == "camera":
        for step in fixture["trace"]["steps"]:
            calibration = step["frames"]["external"]["calibration"]
            calibration["transform_world_from_camera"]["position_m"][0] = 2.0
            calibration["calibration_digest"] = _digest(
                calibration, digest_field="calibration_digest"
            )
            step["frames"]["external"]["frame_digest"] = _digest(
                step["frames"]["external"], digest_field="frame_digest"
            )
        _reseal_steps(fixture)
    else:
        for step in fixture["trace"]["steps"]:
            step["frames"]["external"]["renderer_identity_sha256"] = _sha("f")
            step["frames"]["external"]["frame_digest"] = _digest(
                step["frames"]["external"], digest_field="frame_digest"
            )
        _reseal_steps(fixture)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    expected = (
        "deformable_episode_frozen_trace_thresholds_mismatch"
        if contract == "thresholds"
        else "deformable_episode_frozen_camera_contract_mismatch:external"
    )
    assert expected in exc_info.value.errors


def test_action_adapter_join_is_deterministically_replayed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    action = fixture["trace"]["steps"][0]["action"]
    action["source_output"] = [0.2, 0.0, -0.04]
    action["source_output_sha256"] = _bytes_digest(_canonical_bytes(action["source_output"]))
    action["policy_inference"]["source_output_sha256"] = action["source_output_sha256"]
    action["policy_inference"]["receipt_digest"] = _digest(
        action["policy_inference"], digest_field="receipt_digest"
    )
    action["action_digest"] = _digest(action, digest_field="action_digest")
    _reseal_steps(fixture)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert "deformable_episode_action_adapter_replay_mismatch:0" in exc_info.value.errors


def test_non_identity_frozen_affine_action_adapter_is_exactly_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    actor_digest = fixture["trace"]["actor"]["identity_digest"]
    fixture["frozen"]["action_replay_contract_by_actor_identity_digest"][actor_digest] = (
        _action_replay_contract(arm_scale=[2.0, 1.0], gripper_scale=2.0)
    )
    fixture["frozen"]["contract_digest"] = _digest(
        fixture["frozen"], digest_field="contract_digest"
    )
    for step in fixture["trace"]["steps"]:
        action = step["action"]
        if action["origin_kind"] != "learned_policy":
            continue
        action["source_output"] = [
            action["arm_command"][0] / 2.0,
            action["arm_command"][1],
            action["gripper_delta_command_m"] / 2.0,
        ]
        action["source_output_sha256"] = _bytes_digest(_canonical_bytes(action["source_output"]))
        action["policy_inference"]["source_output_sha256"] = action["source_output_sha256"]
        action["policy_inference"]["receipt_digest"] = _digest(
            action["policy_inference"], digest_field="receipt_digest"
        )
        action["action_digest"] = _digest(action, digest_field="action_digest")
    _reseal_steps(fixture)
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)

    assert receipt["native_event_authority"]["native_event_authority_verified"] is True
    assert receipt["steps"][0]["action"]["source_output"] == [0.05, 0.0, -0.02]
    assert receipt["steps"][0]["action"]["native_action"] == [0.1, 0.0, -0.04]


@pytest.mark.parametrize("attack", ["frame_digest", "inference_precedes_frame", "missing"])
def test_policy_actions_bind_exact_input_frames_and_inference_timing(
    tmp_path: Path, attack: str
) -> None:
    fixture = _fixture(tmp_path)
    action = fixture["trace"]["steps"][0]["action"]
    inference = action["policy_inference"]
    if attack == "frame_digest":
        inference["policy_input_frame_digest_by_camera_id"]["external"] = _sha("f")
        inference["receipt_digest"] = _digest(inference, digest_field="receipt_digest")
    elif attack == "inference_precedes_frame":
        inference["inference_started_timestamp_ns"] = (
            fixture["trace"]["steps"][0]["frames"]["external"]["timestamp_ns"] - 1
        )
        inference["receipt_digest"] = _digest(inference, digest_field="receipt_digest")
    else:
        action["policy_inference"] = None
    action["action_digest"] = _digest(action, digest_field="action_digest")
    _reseal_steps(fixture)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    expected = {
        "frame_digest": "deformable_episode_policy_input_frame_join_mismatch:0",
        "inference_precedes_frame": "deformable_episode_policy_inference_precedes_input:0",
        "missing": "deformable_episode_policy_inference_missing:0",
    }[attack]
    assert expected in exc_info.value.errors


@pytest.mark.parametrize(
    "mutation", ["source", "command", "native", "delivered_digest", "subepsilon"]
)
def test_zero_control_requires_every_action_seam_value_zero(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path, "zero_action_control", move=False)
    action = fixture["trace"]["steps"][0]["action"]
    nonzero = [0.1, 0.0, 0.0]
    if mutation == "source":
        action["source_output"] = nonzero
        action["source_output_sha256"] = _bytes_digest(_canonical_bytes(nonzero))
    elif mutation == "command":
        action["arm_command"] = [0.1, 0.0]
        action["source_output"] = nonzero
        action["native_action"] = nonzero
        action["source_output_sha256"] = _bytes_digest(_canonical_bytes(nonzero))
        action["delivery"]["native_action_sha256"] = _bytes_digest(_canonical_bytes(nonzero))
    elif mutation == "native":
        action["native_action"] = nonzero
        action["delivery"]["native_action_sha256"] = _bytes_digest(_canonical_bytes(nonzero))
    elif mutation == "delivered_digest":
        action["delivery"]["native_action_sha256"] = _bytes_digest(_canonical_bytes(nonzero))
    else:
        subepsilon = [1.0e-12, 0.0, 0.0]
        action["source_output"] = subepsilon
        action["arm_command"] = subepsilon[:2]
        action["native_action"] = subepsilon
        action["source_output_sha256"] = _bytes_digest(_canonical_bytes(subepsilon))
        action["delivery"]["native_action_sha256"] = _bytes_digest(_canonical_bytes(subepsilon))
    action["delivery"]["receipt_digest"] = _digest(
        action["delivery"], digest_field="receipt_digest"
    )
    action["action_digest"] = _digest(action, digest_field="action_digest")
    _reseal_steps(fixture)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert any(
        error.startswith(
            (
                "deformable_episode_action_adapter_replay_mismatch",
                "deformable_episode_native_action_digest_mismatch",
                "deformable_episode_zero_action_seam_not_all_zero_and_delivered",
            )
        )
        for error in exc_info.value.errors
    )


@pytest.mark.parametrize(
    "media_attack",
    [
        "missing",
        "counterfeit",
        "marker_injection",
        "zeroed_mdat",
        "frame_count",
        "valid_wrong_content",
        "wrong_derivation",
    ],
)
def test_h264_review_media_is_exactly_joined_or_media_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media_attack: str
) -> None:
    fixture = _fixture(tmp_path)
    if media_attack == "missing":
        fixture["trace"]["review_videos"] = None
    elif media_attack == "counterfeit":
        video = fixture["trace"]["review_videos"]["external"]
        (fixture["root"] / video["relative_path"]).write_bytes(b"not-an-h264-mp4")
    elif media_attack == "marker_injection":
        video = fixture["trace"]["review_videos"]["external"]

        def box(kind: bytes, payload: bytes) -> bytes:
            return (len(payload) + 8).to_bytes(4, "big") + kind + payload

        counterfeit = (
            box(b"ftyp", b"isom\x00\x00\x00\x00isom")
            + box(b"moov", b"avc1-marker-avcC-marker")
            + box(b"mdat", b"counterfeit")
        )
        (fixture["root"] / video["relative_path"]).write_bytes(counterfeit)
        video["size_bytes"] = len(counterfeit)
        video["file_sha256"] = _bytes_digest(counterfeit)
        video["video_receipt_digest"] = _digest(video, digest_field="video_receipt_digest")
    elif media_attack == "zeroed_mdat":
        video = fixture["trace"]["review_videos"]["external"]
        zeroed = bytearray(_H264_MP4_BY_CAMERA_ID["external"])
        mdat_type_offset = zeroed.index(b"mdat")
        mdat_start = mdat_type_offset - 4
        mdat_size = int.from_bytes(zeroed[mdat_start:mdat_type_offset], "big")
        zeroed[mdat_type_offset + 4 : mdat_start + mdat_size] = b"\x00" * (
            mdat_start + mdat_size - (mdat_type_offset + 4)
        )
        zeroed_bytes = bytes(zeroed)
        (fixture["root"] / video["relative_path"]).write_bytes(zeroed_bytes)
        video["size_bytes"] = len(zeroed_bytes)
        video["file_sha256"] = _bytes_digest(zeroed_bytes)
        video["video_receipt_digest"] = _digest(video, digest_field="video_receipt_digest")
    elif media_attack == "frame_count":
        video = fixture["trace"]["review_videos"]["external"]
        wrong_count = bytearray(_H264_MP4_BY_CAMERA_ID["external"])
        stsz_type_offset = wrong_count.index(b"stsz")
        wrong_count[stsz_type_offset + 12 : stsz_type_offset + 16] = (3).to_bytes(4, "big")
        wrong_count_bytes = bytes(wrong_count)
        (fixture["root"] / video["relative_path"]).write_bytes(wrong_count_bytes)
        video["size_bytes"] = len(wrong_count_bytes)
        video["file_sha256"] = _bytes_digest(wrong_count_bytes)
        video["video_receipt_digest"] = _digest(video, digest_field="video_receipt_digest")
    elif media_attack == "valid_wrong_content":
        video = fixture["trace"]["review_videos"]["external"]
        wrong_content = _H264_MP4_BY_CAMERA_ID["wrist"]
        (fixture["root"] / video["relative_path"]).write_bytes(wrong_content)
        video["size_bytes"] = len(wrong_content)
        video["file_sha256"] = _bytes_digest(wrong_content)
        video["video_receipt_digest"] = _digest(video, digest_field="video_receipt_digest")
    else:
        video = fixture["trace"]["review_videos"]["external"]
        video["source_frame_digest_sequence_sha256"] = _sha("f")
        video["video_receipt_digest"] = _digest(video, digest_field="video_receipt_digest")
    _authorize(fixture, monkeypatch)

    receipt = _materialize(fixture)

    assert receipt["native_event_authority"]["native_event_authority_verified"] is True
    assert receipt["media_complete"] is False
    assert receipt["media_status"] == "incomplete_review_video_evidence"
    assert receipt["review_videos"] is None
    assert receipt["media_validation_blockers"]
    assert receipt["native_episode_admitted_deterministic_success"] is False
    assert receipt["evaluation_admitted_deterministic_success"] is False


def test_h264_decode_cache_cannot_be_poisoned_through_a_prior_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_module._H264_DECODE_CACHE.clear()
    first = _fixture(tmp_path / "first")
    _authorize(first, monkeypatch)
    first_receipt = _materialize(first)
    assert first_receipt["media_complete"] is True

    second = _fixture(tmp_path / "second")
    changed_rgb_digests: list[str] = []
    for index, step in enumerate(second["trace"]["steps"]):
        frame = step["frames"]["external"]
        image = Image.new("RGB", (8, 6), color=(200, index + 1, 100))
        output = io.BytesIO()
        image.save(output, format="PNG", compress_level=9)
        payload = output.getvalue()
        (second["root"] / frame["relative_path"]).write_bytes(payload)
        frame["size_bytes"] = len(payload)
        frame["lossless_file_sha256"] = _bytes_digest(payload)
        frame["raw_rgb_sha256"] = _bytes_digest(image.tobytes())
        frame["frame_digest"] = _digest(frame, digest_field="frame_digest")
        changed_rgb_digests.append(frame["raw_rgb_sha256"])
        inference = step["action"]["policy_inference"]
        if inference is not None:
            inference["policy_input_frame_digest_by_camera_id"]["external"] = frame["frame_digest"]
            inference["receipt_digest"] = _digest(inference, digest_field="receipt_digest")
            step["action"]["action_digest"] = _digest(step["action"], digest_field="action_digest")
    _reseal_steps(second)

    poisoned_projection = first_receipt["review_videos"]["external"]["verifier_owned_decode"][
        "raw_rgb_sha256_by_sample"
    ]
    poisoned_projection[:] = changed_rgb_digests
    _authorize(second, monkeypatch)

    second_receipt = _materialize(second)

    assert second_receipt["native_event_authority"]["native_event_authority_verified"] is True
    assert second_receipt["media_complete"] is False
    assert second_receipt["review_videos"] is None
    assert (
        "deformable_episode_review_video_invalid:external_derivation_join_mismatch"
        in second_receipt["media_validation_blockers"]
    )
    assert second_receipt["native_episode_admitted_deterministic_success"] is False


def test_single_descriptor_fstat_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    real_fstat = os.fstat
    regular_observations: dict[tuple[int, int], int] = {}

    def drifting_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        observed = real_fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            return observed
        identity = (observed.st_dev, observed.st_ino)
        regular_observations[identity] = regular_observations.get(identity, 0) + 1
        if regular_observations[identity] == 1:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(episode_module.os, "fstat", drifting_fstat)

    with pytest.raises(NativeDeformableEpisodeTraceError) as exc_info:
        _materialize(fixture)

    assert any("changed_during_read" in error for error in exc_info.value.errors)


def test_artifact_fifo_leaf_fails_closed_without_blocking(tmp_path: Path) -> None:
    fifo_path = tmp_path / "event.fifo"
    os.mkfifo(fifo_path)
    script = """
import json
import sys
from pathlib import Path
from blueprint_pipeline.native_deformable_episode_trace import (
    NativeDeformableEpisodeTraceError,
    _artifact_bytes,
)

try:
    _artifact_bytes(
        evidence_root=Path(sys.argv[1]),
        relative_path="event.fifo",
        maximum_bytes=1024,
        error="fixture_fifo_file_invalid",
    )
except NativeDeformableEpisodeTraceError as exc:
    print(json.dumps(exc.errors))
else:
    raise SystemExit(2)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["fixture_fifo_file_invalid"]


def test_identical_cell_controls_are_required_at_evaluation_aggregator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "zero_action_control": _fixture(tmp_path / "zero", "zero_action_control", move=False),
        "scripted_positive_control": _fixture(tmp_path / "scripted", "scripted_positive_control"),
        "learned_policy_evaluation": _fixture(tmp_path / "policy"),
    }
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)

    policy = receipts["learned_policy_evaluation"]
    assert policy["evaluation_admitted_deterministic_success"] is False
    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=policy,
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["identical_cell_controls_passed"] is True
    assert result["evaluation_admitted"] is True
    assert result["evaluation_admitted_deterministic_success"] is True
    assert result["policy_outcome"] == "succeeded"

    forged_policy = copy.deepcopy(policy)
    forged_policy["deterministic_task_state_score"]["outcome"] = "caller_forged_success"
    forged_policy["receipt_digest"] = _digest(forged_policy, digest_field="receipt_digest")
    forged = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=forged_policy,
        expected_episode_receipt_digest_by_kind={
            "zero_action_control": receipts["zero_action_control"]["receipt_digest"],
            "scripted_positive_control": receipts["scripted_positive_control"]["receipt_digest"],
            "learned_policy_evaluation": forged_policy["receipt_digest"],
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )
    assert forged["evaluation_admitted"] is False
    assert forged["identical_cell_controls_passed"] is False
    assert "episode_cryptographic_replay_mismatch:learned_policy_evaluation" in forged["blockers"]

    failed_fixture = _fixture(tmp_path / "policy-failed", successful_state=False)
    _authorize(failed_fixture, monkeypatch)
    failed_policy = _materialize(failed_fixture)
    failed_result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=failed_policy,
        expected_episode_receipt_digest_by_kind={
            "zero_action_control": receipts["zero_action_control"]["receipt_digest"],
            "scripted_positive_control": receipts["scripted_positive_control"]["receipt_digest"],
            "learned_policy_evaluation": failed_policy["receipt_digest"],
        },
        episode_replay_inputs_by_kind={
            "zero_action_control": _materialize_args(fixtures["zero_action_control"]),
            "scripted_positive_control": _materialize_args(fixtures["scripted_positive_control"]),
            "learned_policy_evaluation": _materialize_args(failed_fixture),
        },
    )
    assert failed_result["evaluation_admitted"] is True
    assert failed_result["evaluation_admitted_deterministic_success"] is False
    assert (
        failed_result["policy_outcome"]
        == failed_policy["deterministic_task_state_score"]["outcome"]
    )
    assert failed_result["policy_outcome"] != "succeeded"

    structural_fixture = _fixture(tmp_path / "structural")
    structural_policy = _materialize(structural_fixture)
    blocked = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=structural_policy,
        expected_episode_receipt_digest_by_kind={
            "zero_action_control": receipts["zero_action_control"]["receipt_digest"],
            "scripted_positive_control": receipts["scripted_positive_control"]["receipt_digest"],
            "learned_policy_evaluation": structural_policy["receipt_digest"],
        },
        episode_replay_inputs_by_kind={
            "zero_action_control": _materialize_args(fixtures["zero_action_control"]),
            "scripted_positive_control": _materialize_args(fixtures["scripted_positive_control"]),
            "learned_policy_evaluation": _materialize_args(structural_fixture),
        },
    )
    assert blocked["evaluation_admitted_deterministic_success"] is False
    assert "trusted_native_event_authority_missing:learned_policy_evaluation" in blocked["blockers"]


def test_aggregator_rejects_scorer_success_without_complete_manipulation_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "zero_action_control": _fixture(tmp_path / "zero", "zero_action_control", move=False),
        "scripted_positive_control": _fixture(tmp_path / "scripted", "scripted_positive_control"),
        "learned_policy_evaluation": _fixture(tmp_path / "policy"),
    }
    _force_release_before_deformable_displacement(fixtures["learned_policy_evaluation"])
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)

    policy = receipts["learned_policy_evaluation"]
    assert policy["deterministic_task_state_score"]["deterministic_success"] is True
    assert policy["native_episode_evidence_admitted"] is True
    assert policy["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert policy["native_episode_admitted_deterministic_success"] is False

    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=policy,
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["identical_cell_controls_passed"] is True
    assert result["evaluation_admitted"] is False
    assert result["evaluation_admitted_deterministic_success"] is False
    assert result["policy_outcome"] is None
    assert "learned_episode_succeeded_without_admitted_manipulation_sequence" in result["blockers"]


@pytest.mark.parametrize(
    ("attack", "command_by_sample_index", "close_response_observed"),
    [
        ("all_zero", {0: 0.0, 2: 0.0}, False),
        ("close_only", {2: 0.0}, True),
    ],
)
def test_aggregator_rejects_success_without_commanded_close_and_release_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    command_by_sample_index: dict[int, float],
    close_response_observed: bool,
) -> None:
    fixtures = {
        "zero_action_control": _fixture(
            tmp_path / attack / "zero", "zero_action_control", move=False
        ),
        "scripted_positive_control": _fixture(
            tmp_path / attack / "scripted", "scripted_positive_control"
        ),
        "learned_policy_evaluation": _fixture(tmp_path / attack / "policy"),
    }
    _replace_gripper_commands(fixtures["learned_policy_evaluation"], command_by_sample_index)
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)

    policy = receipts["learned_policy_evaluation"]
    motion = policy["evidence"]["robot_motion"]
    assert policy["deterministic_task_state_score"]["deterministic_success"] is True
    assert policy["evidence"]["interpretability"]["policy_outcome_interpretable"] is True
    assert policy["native_episode_evidence_admitted"] is True
    assert motion["gripper_close_observed"] is True
    assert motion["gripper_release_observed"] is True
    assert (
        motion["commanded_gripper_close_response_observed_before_or_at_grasp"]
        is close_response_observed
    )
    assert motion["commanded_gripper_release_response_observed_at_release"] is False
    assert motion["commanded_close_then_release_response_ordered"] is False
    assert policy["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert policy["native_episode_admitted_deterministic_success"] is False

    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=policy,
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["identical_cell_controls_passed"] is True
    assert result["evaluation_admitted"] is False
    assert result["evaluation_admitted_deterministic_success"] is False
    assert result["policy_outcome"] is None
    assert "learned_episode_succeeded_without_admitted_manipulation_sequence" in result["blockers"]


def test_aggregator_rejects_contact_that_begins_with_cloth_already_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "zero_action_control": _fixture(tmp_path / "zero", "zero_action_control", move=False),
        "scripted_positive_control": _fixture(tmp_path / "scripted", "scripted_positive_control"),
        "learned_policy_evaluation": _fixture(tmp_path / "policy"),
    }
    policy_fixture = fixtures["learned_policy_evaluation"]
    policy_fixture["trace"]["steps"][0]["entities"]["cloth"]["nodal_positions_world_m"] = [
        [point[0] - 0.2, point[1], point[2]] for point in _DEFORMABLE_CONTAINED_POSITIONS
    ]
    _reseal_steps(policy_fixture)
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)

    policy = receipts["learned_policy_evaluation"]
    motion = policy["evidence"]["deformable_motion"]
    assert policy["deterministic_task_state_score"]["deterministic_success"] is True
    assert motion["initial_deformable_outside_destination"] is True
    assert motion["node_count_inside_destination_by_sample"][0] == 4
    assert motion["first_qualifying_grasp_contact_sample_index"] == 0
    assert motion["all_samples_through_qualifying_contact_outside_destination"] is False
    assert motion["first_task_relevant_displacement_sample_index"] == 1
    assert motion["first_post_contact_containment_sample_index"] == 1
    assert motion["post_contact_displacement_then_containment_transition_observed"] is False
    assert motion["ordered_contact_displacement_release_final_settle"] is False
    assert policy["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert policy["native_episode_admitted_deterministic_success"] is False

    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=policy,
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["identical_cell_controls_passed"] is True
    assert result["evaluation_admitted"] is False
    assert result["evaluation_admitted_deterministic_success"] is False
    assert result["policy_outcome"] is None
    assert "learned_episode_succeeded_without_admitted_manipulation_sequence" in result["blockers"]


def test_aggregator_rejects_arm_response_deferred_until_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "zero_action_control": _fixture(tmp_path / "zero", "zero_action_control", move=False),
        "scripted_positive_control": _fixture(tmp_path / "scripted", "scripted_positive_control"),
        "learned_policy_evaluation": _fixture(tmp_path / "policy"),
    }
    _defer_arm_response_until_release(fixtures["learned_policy_evaluation"])
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)

    policy = receipts["learned_policy_evaluation"]
    motion = policy["evidence"]["robot_motion"]
    assert policy["deterministic_task_state_score"]["deterministic_success"] is True
    assert policy["evidence"]["interpretability"]["policy_outcome_interpretable"] is True
    assert motion["arm_moved"] is True
    assert policy["evidence"]["action_delivery"]["arm_response_sample_indices"] == [2]
    assert motion["retreat_response_observed_after_contact"] is True
    assert motion["transport_arm_response_sample_indices"] == []
    assert motion["delivered_arm_transport_response_observed"] is False
    assert policy["evidence"]["interpretability"]["manipulation_sequence_complete"] is False
    assert policy["native_episode_admitted_deterministic_success"] is False

    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=policy,
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["identical_cell_controls_passed"] is True
    assert result["evaluation_admitted"] is False
    assert result["evaluation_admitted_deterministic_success"] is False
    assert result["policy_outcome"] is None
    assert "learned_episode_succeeded_without_admitted_manipulation_sequence" in result["blockers"]


def test_aggregator_joins_exact_native_reset_readbacks_for_every_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "zero_action_control": _fixture(tmp_path / "zero", "zero_action_control", move=False),
        "scripted_positive_control": _fixture(tmp_path / "scripted", "scripted_positive_control"),
        "learned_policy_evaluation": _fixture(tmp_path / "policy"),
    }
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)
    scripted_receipt = copy.deepcopy(receipts["scripted_positive_control"])
    scripted_reset = scripted_receipt["reset_receipt"]
    scripted_reset["native_state_readback_sha256_by_entity_id"]["pillar"] = _sha("f")
    scripted_reset["receipt_digest"] = _digest(scripted_reset, digest_field="receipt_digest")
    scripted_receipt["receipt_digest"] = _digest(scripted_receipt, digest_field="receipt_digest")
    receipts["scripted_positive_control"] = scripted_receipt

    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=receipts["learned_policy_evaluation"],
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["evaluation_admitted"] is False
    assert result["identical_cell_controls_passed"] is False
    assert result["identical_cell_native_reset_readback_sha256_by_entity_id"] is None
    assert result["native_reset_readback_sha256_by_entity_id_by_episode_kind"][
        "scripted_positive_control"
    ]["pillar"] == _sha("f")
    assert (
        "identical_cell_native_reset_readback_mismatch:scripted_positive_control:pillar"
        in result["blockers"]
    )
    assert "episode_cryptographic_replay_mismatch:scripted_positive_control" in result["blockers"]


def test_aggregator_rejects_independently_refrozen_numeric_robot_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "zero_action_control": _fixture(tmp_path / "zero", "zero_action_control", move=False),
        "scripted_positive_control": _fixture(tmp_path / "scripted", "scripted_positive_control"),
        "learned_policy_evaluation": _fixture(tmp_path / "policy"),
    }
    _replace_robot_reset_state(
        fixtures["scripted_positive_control"],
        joint_positions_rad=[0.03, 0.0],
        gripper_width_m=0.08,
        refreeze=True,
    )
    receipts = {}
    for kind, fixture in fixtures.items():
        _authorize(fixture, monkeypatch)
        receipts[kind] = _materialize(fixture)

    assert (
        receipts["scripted_positive_control"]["control_outcome"]
        == "required_scripted_positive_success_observed"
    )
    result = aggregate_native_deformable_cell_evaluation(
        zero_action_receipt=receipts["zero_action_control"],
        scripted_positive_receipt=receipts["scripted_positive_control"],
        learned_policy_receipt=receipts["learned_policy_evaluation"],
        expected_episode_receipt_digest_by_kind={
            kind: receipt["receipt_digest"] for kind, receipt in receipts.items()
        },
        episode_replay_inputs_by_kind={
            kind: _materialize_args(fixture) for kind, fixture in fixtures.items()
        },
    )

    assert result["evaluation_admitted"] is False
    assert result["identical_cell_controls_passed"] is False
    assert (
        "identical_cell_join_mismatch:scripted_positive_control:"
        "frozen_run_contract_digest" in result["blockers"]
    )
    assert (
        "identical_cell_native_reset_readback_mismatch:"
        "scripted_positive_control:manipulator_alpha" in result["blockers"]
    )


def test_native_event_and_external_seal_bytes_cannot_be_replaced_or_self_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path / "event")
    _authorize(fixture, monkeypatch)
    event_path = fixture["root"] / fixture["authority"]["native_event_relative_path"]
    event_path.write_bytes(event_path.read_bytes() + b" ")
    replaced = _materialize(fixture)
    assert replaced["native_event_authority"]["native_event_authority_verified"] is False
    assert replaced["native_episode_admitted_deterministic_success"] is False

    seal_fixture = _fixture(tmp_path / "seal")
    _authorize(seal_fixture, monkeypatch)
    seal_path = seal_fixture["root"] / seal_fixture["authority"]["frozen_run_seal_relative_path"]
    seal = json.loads(seal_path.read_text())
    seal["native_event_size_bytes"] += 1
    seal["seal_digest"] = _digest(seal, digest_field="seal_digest")
    seal_path.write_bytes(_canonical_bytes(seal) + b"\n")
    self_rehashed = _materialize(seal_fixture)
    assert self_rehashed["native_event_authority"]["native_event_authority_verified"] is False
    assert (
        "frozen_run_seal_external_digest_mismatch"
        in self_rehashed["native_event_authority"]["blockers"]
    )


def test_native_event_symlink_and_wrong_runner_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path / "symlink")
    _authorize(fixture, monkeypatch)
    event_path = fixture["root"] / fixture["authority"]["native_event_relative_path"]
    target = fixture["root"] / "authority/event-target.json"
    event_path.replace(target)
    event_path.symlink_to(target)
    symlinked = _materialize(fixture)
    assert symlinked["native_event_authority"]["native_event_authority_verified"] is False
    assert any(
        "symlink_forbidden" in blocker
        for blocker in symlinked["native_event_authority"]["blockers"]
    )

    wrong_key = _fixture(tmp_path / "key")
    _authorize(wrong_key, monkeypatch)
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, _sha("f"))
    rejected = _materialize(wrong_key)
    assert rejected["native_event_authority"]["native_event_authority_verified"] is False
    assert (
        "trusted_execution_envelope_public_key_not_authorized"
        in rejected["native_event_authority"]["blockers"]
    )
