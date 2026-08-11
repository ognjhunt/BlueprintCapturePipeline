from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_deformable_task_arena_readback import (
    CONTACT_CAPABILITY_BLOCKER,
    NativeDeformableTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_arena_controls_worker import (
    _bind_task_episode_environment,
    _load_and_verify_manifest,
    _verified_runtime_inputs,
    _worker_capability_blockers,
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
    assert "NativeArticulatedTaskArenaReadback" not in source
    assert 'scene_asset_names["task_object"]' not in source


def _deformable_built() -> SimpleNamespace:
    operations = [
        "load_default_nodal_state",
        "zero_nodal_velocities",
        "write_nodal_state_to_sim_index",
        "write_nodal_kinematic_target_to_sim_index",
        "readback_physx_root_view_state_and_kinematic_target",
    ]
    recipe = {
        "reset_kind": "native_deformable_state",
        "state_id": "movable-reset",
        "write_scope": "before_episode_start_only",
        "direct_state_write_after_episode_start_allowed": False,
        "native_readback_required": True,
        "steps": [
            {
                "order": index,
                "operation": operation,
                **({"free_flag_value": 1.0} if index == 4 else {}),
            }
            for index, operation in enumerate(operations, start=1)
        ],
    }
    entities = [
        {
            "entity_id": "movable-alpha",
            "semantic_role": "movable_deformable",
            "physics_type": "deformable_volume",
        },
        {
            "entity_id": "destination-alpha",
            "semantic_role": "destination_receptacle",
            "physics_type": "static_collider",
        },
        {
            "entity_id": "support-alpha",
            "semantic_role": "support_surface",
            "physics_type": "static_collider",
        },
        {
            "entity_id": "obstacle-alpha",
            "semantic_role": "obstacle",
            "physics_type": "static_collider",
        },
        {
            "entity_id": "robot-alpha",
            "semantic_role": "robot",
            "physics_type": "robot_articulation",
        },
    ]
    return SimpleNamespace(
        env=SimpleNamespace(
            unwrapped=SimpleNamespace(
                scene={"robot": object()},
                action_manager=SimpleNamespace(total_action_dim=8),
            ),
            reset=lambda *, seed: None,
        ),
        plan={
            "task_kind": "deformable_transfer",
            "task_spec": {
                "deformable_entity_id": "movable-alpha",
                "destination_entity_id": "destination-alpha",
                "robot_entity_id": "robot-alpha",
            },
            "task_entities": entities,
            "task_entity_role_index": {"robot": ["robot-alpha"]},
            "robot": {
                "grasp_frame": {
                    "kind": "body_midpoint",
                    "body_names": ["finger-left", "finger-right"],
                }
            },
            "scenario": {"seed": 17},
            "cadence": {"control_frequency_hz": 20.0},
        },
        scene_asset_names={
            "movable_deformable": "movable_runtime",
            "destination_receptacle": "destination_runtime",
        },
        scene_asset_names_by_entity_id={
            "movable-alpha": "movable_runtime",
            "destination-alpha": "destination_runtime",
        },
        scene_asset_prim_paths_by_entity_id={
            "movable-alpha": "/World/movable",
            "destination-alpha": "/World/destination",
        },
        entity_reset_recipes_by_entity_id={"movable-alpha": recipe},
        camera_scene_names={
            "external": "external_sensor",
            "wrist": "wrist_sensor",
            "overview": "overview_sensor",
        },
    )


def test_deformable_controls_route_through_shared_environment_to_exact_blocker() -> None:
    built = _deformable_built()

    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match=CONTACT_CAPABILITY_BLOCKER,
    ) as exc_info:
        _bind_task_episode_environment(
            built=built,
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=object(),
            to_tensor=lambda value: value,
        )

    assert _worker_capability_blockers(exc_info.value) == [CONTACT_CAPABILITY_BLOCKER]
    assert "task_object" not in built.scene_asset_names


def test_rigid_controls_keep_the_shared_environment_compatibility_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as environment_module

    captured = {}

    class Adapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class Servo:
        def current_body_pose_world(self):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def reset_command_state(self):
            return None

        def action_for_grasp_target(self, **kwargs):
            return [0.0] * 7 + [float(kwargs["gripper_command"])], {}

    monkeypatch.setattr(environment_module, "IsaacEpisodeAdapter", Adapter)
    legacy_asset = object()
    built = SimpleNamespace(
        env=SimpleNamespace(
            unwrapped=SimpleNamespace(
                scene={"robot": object(), "legacy_target": legacy_asset},
                action_manager=SimpleNamespace(total_action_dim=8),
            ),
            reset=lambda *, seed: None,
        ),
        plan={
            "task_kind": "rigid_pick_place",
            "scenario": {"seed": 17},
            "cadence": {"control_frequency_hz": 15.0},
        },
        scene_asset_names={"task_object": "legacy_target"},
        camera_scene_names={"external": "external_sensor"},
    )

    adapter, binding = _bind_task_episode_environment(
        built=built,
        gripper_convention={
            "closed_command": 1.0,
            "open_command": 0.0,
            "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
        },
        servo=Servo(),
        to_tensor=lambda value: value,
    )

    assert isinstance(adapter, Adapter)
    assert binding["task_readback_binding"]["task_kind"] == "rigid_pick_place"
    assert binding["task_readback_binding"]["blockers"] == []
    assert binding["episode_environment"]["task_state_source"] == ("native_rigid_body_readback")
    assert captured["rigid_task_object"] is legacy_asset
    assert captured["task_sample_callback"] is None


def test_controls_manifest_rejects_policy_or_construction_mode(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "controls",
        "policy_candidate_id": None,
        "candidate_policy_queried": False,
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    path = tmp_path / "adp_arena_provider_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load_and_verify_manifest(tmp_path)["execution_mode"] == "controls"

    for mode in ("construction_canary", "policy"):
        manifest["execution_mode"] = mode
        manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
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
    verified = _verified_runtime_inputs(tmp_path, {"bound_runtime_inputs": rows})
    assert set(verified) == {
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    }

    (inputs / "adp_task_control_plan.v1.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity_mismatch"):
        _verified_runtime_inputs(tmp_path, {"bound_runtime_inputs": rows})
