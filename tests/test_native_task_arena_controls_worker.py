from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_controls_worker import (
    _RigidScoringEnvironment,
    _load_and_verify_manifest,
    _verified_runtime_inputs,
)


def test_controls_worker_source_has_no_scene_task_or_policy_identity() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.native_task_arena_controls_worker",
            fromlist=["x"],
        ).__file__
    ).read_text(encoding="utf-8")

    for forbidden in (
        "840313",
        "840796",
        "refrigerator",
        "approved_can",
        "pi05_droid",
        "groot_n17_droid",
    ):
        assert forbidden not in source


def test_controls_manifest_rejects_policy_or_construction_mode(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "controls",
        "policy_candidate_id": None,
        "candidate_policy_queried": False,
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
    path = tmp_path / "adp_arena_provider_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load_and_verify_manifest(tmp_path)["execution_mode"] == "controls"

    for mode in ("construction_canary", "policy"):
        manifest["execution_mode"] = mode
        manifest["input_digest"] = canonical_digest(
            manifest, digest_field="input_digest"
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="native_task_controls_manifest_invalid"):
            _load_and_verify_manifest(tmp_path)


def test_controls_runtime_inputs_reverify_every_byte(tmp_path: Path) -> None:
    inputs = tmp_path / "runtime_inputs"
    inputs.mkdir()
    rows = []
    for name in (
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    ):
        path = inputs / name
        path.write_text("{}\n", encoding="utf-8")
        rows.append(
            {
                "relative_path": f"runtime_inputs/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    verified = _verified_runtime_inputs(
        tmp_path, {"bound_runtime_inputs": rows}
    )
    assert set(verified) == {
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    }

    (inputs / "adp_task_control_plan.v1.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity_mismatch"):
        _verified_runtime_inputs(tmp_path, {"bound_runtime_inputs": rows})


class _BaseRigidEnvironment:
    def reset(self) -> None:
        return None

    def read_object_sample(self) -> dict:
        return {
            "task_object_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_width_m": 0.071,
            "grasp_frame_position_world_m": [1.0, 2.0, 3.0],
        }


class _ExactRigidReadback:
    def read_task_sample(self) -> dict:
        return {
            "asset_root_pose_world": [1.0, 2.0, 0.7, 0.0, 0.0, 0.0, 1.0],
            "task_scoring_pose_world": [1.02, 1.99, 0.73, 0.0, 0.0, 0.0, 1.0],
            "task_robot_contact_peak_force_n": 0.75,
            "task_support_contact_peak_force_n": 4.0,
            "task_scene_collision_peak_force_n": 0.2,
            "robot_scene_contact_peak_force_n": 0.1,
            "robot_task_forbidden_collision_peak_force_n": 0.0,
            "locked_joint_containment_violation": False,
        }


def _graph_rigid_task_spec() -> dict:
    return {
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "workspace_position_bounds_world_m": {
            "minimum": [0.0, 0.0, 0.0],
            "maximum": [2.0, 3.0, 2.0],
        },
    }


def test_rigid_controls_environment_uses_scoring_frame_and_exact_contacts() -> None:
    environment = _RigidScoringEnvironment(
        environment=_BaseRigidEnvironment(),
        task_readback=_ExactRigidReadback(),
        task_spec=_graph_rigid_task_spec(),
    )

    sample = environment.read_object_sample()

    assert sample["task_object_pose_world"] == [
        1.02,
        1.99,
        0.73,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert sample["asset_root_pose_world"] != sample["task_object_pose_world"]
    assert sample["gripper_width_m"] == pytest.approx(0.071)
    assert sample["task_contact_active"] is True
    assert sample["support_contact_active"] is True
    assert sample["robot_collision_failure"] is False
    assert sample["scene_collision_failure"] is False
    assert sample["forbidden_robot_task_collision_failure"] is False
    assert sample["locked_joint_containment_violation"] is False
    assert sample["containment_violation"] is False
    environment.reset()


def test_rigid_controls_environment_fails_closed_on_missing_native_channel() -> None:
    readback = _ExactRigidReadback()
    readback.read_task_sample = lambda: {"task_scoring_pose_world": [0.0] * 7}
    environment = _RigidScoringEnvironment(
        environment=_BaseRigidEnvironment(),
        task_readback=readback,
        task_spec=_graph_rigid_task_spec(),
    )

    with pytest.raises(RuntimeError, match="rigid_sample_invalid"):
        environment.read_object_sample()
