"""The sealed scene inputs a scripted check's tests stage: the frozen pair's packet, runtime source and inputs.

Built as the retired internal canary sealed them (moved unchanged from its tests), so the scripted check keeps
running on the same real packet and runtime-source fixture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from blueprint_pipeline.adp009d_task_scoring import SUPPORT_PLANE_Z_M
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.policy_canary_setup_support import public_setup
from tests.test_native_task_arena_bundle import _packet, _runtime_source_packet

#: ``native_task_arena_policy_canary_session.CANDIDATE_IDS`` when the canary was retired.
CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")
_DESTINATION = [3.750152333333333, -3.4074919, SUPPORT_PLANE_Z_M]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _scene_plan() -> dict[str, Any]:
    pose = {
        "position_world_m": [3.4681748, -3.3100837, SUPPORT_PLANE_Z_M],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    value: dict[str, Any] = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "task_id": "scene-839873-mug-planar-push",
        "task_kind": "rigid_pick_place",
        # The robot station, as the scene compiler places Blueprint's robot (a team's arm stands here too).
        "robot": {"robot_id": "franka_panda", "base_pose_world": {
            "position_world_m": [2.9681748, -3.3100837, SUPPORT_PLANE_Z_M], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "objects": [
            {
                "name": "task_object",
                "task_subject": True,
                "pose_world": json.loads(json.dumps(pose)),
                "reset_state": {"root_pose_world": json.loads(json.dumps(pose))},
            },
            {
                "name": "scene_appearance",
                "semantic_role": "scene_appearance",
                "task_subject": False,
                "sha256": "1bfd4438e057587c785b8211a70e26b896dd1ef90626e7923c541dbfd7c125cc",
                "pose_world": json.loads(json.dumps(pose)),
            },
            {
                "name": "task_support",
                "semantic_role": "task_support",
                "asset_id": "document_tray",
                "task_subject": False,
                "object_type": "RIGID",
                "pose_world": {
                    "position_world_m": list(_DESTINATION),
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "reset_state": {
                    "root_pose_world": {
                        "position_world_m": list(_DESTINATION),
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": {},
                },
            },
        ],
        "appearance_frame_alignment": {
            "status": "aligned",
            "representation": "particlefield_3d_gaussian_splat",
            "measurement_authority": "particlefield_position_quantiles",
            "source_asset_sha256": (
                "sha256:9193a9de6bd81bd6348065b3cad46ad835b62dcfaa6212285a91bffd8a166445"
            ),
        },
        "cameras": [
            {
                "role": role,
                "frame_from_camera_matrix": [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
                "optical_convention": "opencv",
                "pose_frame": "robot_body" if role == "wrist" else "world",
                "parent_prim_path": "{ENV_REGEX_NS}/Robot/panda_hand" if role == "wrist" else "{ENV_REGEX_NS}",
                "intrinsics": {"fx": 500., "fy": 500., "cx": 639.5, "cy": 359.5, "width": 1280, "height": 720},
                "data_types": ["rgb", "semantic_segmentation"],
                "policy_input": role != "overview",
                "review_only": role == "overview",
            }
            for role in ("external", "wrist", "overview")
        ],
        # The compiled scene still carries the pre-canary 20 Hz cadence; the
        # worker must resolve it to the frozen DROID adapters' 15 Hz.
        "cadence": {
            "control_frequency_hz": 20.0,
            "physics_frequency_hz": 120.0,
            "physics_dt_seconds": 1.0 / 120.0,
            "control_decimation": 6,
            "maximum_action_steps": 240,
            "settle_window_samples": 1,
            "episode_length_seconds": 13.05,
        },
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "rigid_pick_place",
            "start_pose_world": [
                *pose["position_world_m"],
                *pose["orientation_xyzw"],
            ],
            "destination_position_world_m": list(_DESTINATION),
            "support_plane_z_m": SUPPORT_PLANE_Z_M,
            "settle_window_samples": 1,
            "require_sealed_start_pose": True,
            "control_frequency_hz": 20.0,
            "maximum_action_steps": 240,
        },
        "plan_digest": "",
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def _real_canary_pair_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """The frozen pair's sealed Quick-10 inputs: packet, runtime source, inputs and activation paths."""
    public = public_setup()
    packet = _packet(tmp_path, scene_id="839873")
    plan = _scene_plan()
    appearance = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text())
    plan["appearance_frame_alignment"] = appearance["appearance_frame_alignment"]
    plan["objects"] = [row for row in plan["objects"] if row.get("semantic_role") != "scene_appearance"] + appearance["objects"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    _write(packet / "native_task_arena_scene_plan.v1.json", plan)
    packet_receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
    packet_receipt = json.loads(packet_receipt_path.read_text())
    plan_bytes = (packet / "native_task_arena_scene_plan.v1.json").read_bytes()
    for artifact in packet_receipt["artifacts"]:
        if artifact["relative_path"] == "native_task_arena_scene_plan.v1.json":
            artifact.update(size_bytes=len(plan_bytes), sha256="sha256:" + hashlib.sha256(plan_bytes).hexdigest())
    packet_receipt["arena_scene_plan_digest"] = plan["plan_digest"]
    packet_receipt["receipt_digest"] = canonical_digest(packet_receipt, digest_field="receipt_digest")
    _write(packet_receipt_path, packet_receipt)
    runtime_receipt = _runtime_source_packet(tmp_path)
    construction_value = {"schema_version": "native_task_arena_construction_result.v1",
                          "status": "completed", "construction_gate_qualified": True,
                          "scene_plan_digest": plan["plan_digest"], "result_digest": ""}
    construction_value["result_digest"] = canonical_digest(construction_value, digest_field="result_digest")
    construction = _write(tmp_path / "construction.json", construction_value)
    run_id = "scene-839873-canary-closure"
    activation: dict[str, object] = {
        "schema_version": "task_evaluation_policy_campaign_activation.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "task_success_contract": public["task_success_contract"],
        "task_success_contract_digest": public["task_success_contract_digest"],
        "campaign_unit_count": 10,
        "campaign_units": [
            {
                "campaign_unit_id": f"unit-{index}",
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "candidate_ids": list(CANDIDATE_IDS),
            }
            for index in range(10)
        ],
        "activation_digest": "",
    }
    activation["activation_digest"] = canonical_digest(
        activation, digest_field="activation_digest"
    )
    activation_path = _write(tmp_path / "activation.json", activation)
    cells = []
    for index in range(10):
        scenario = {"family": "canonical", "ordinal": index}
        cells.append(
            {
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "family": "canonical_anchor",
                "cell_spec_digest": "sha256:" + f"{index:064x}",
                "resolved_scenario": scenario,
                "resolved_scenario_digest": canonical_digest(scenario),
                "control_diagnostic": {
                    "mode": "nonblocking_diagnostic_pending",
                    "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False,
                },
            }
        )
    inputs: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_revision_digest": "sha256:" + "9" * 64,
        "matrix_digest": "sha256:" + "8" * 64,
        "configuration_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "activation_digest": activation["activation_digest"],
        "base_native_packet": _record(packet / "native_task_arena_packet_receipt.v1.json"),
        "runtime_source": _record(runtime_receipt),
        "construction_result": _record(construction),
        "candidate_ids": list(CANDIDATE_IDS),
        "task_success_contract": public["task_success_contract"],
        "task_success_contract_digest": public["task_success_contract_digest"],
        "cells": cells,
        "execution_authority": {
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "single_warm_provider_session_required": True,
            "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True,
        },
        "runtime_inputs_digest": "",
    }
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )
    inputs_path = _write(tmp_path / "runtime_inputs.json", inputs)
    return packet, runtime_receipt, inputs_path, activation_path
