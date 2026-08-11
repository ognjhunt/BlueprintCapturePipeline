from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_deformable_task_arena_readback import (
    CONTACT_CAPABILITY_BLOCKER,
    NativeDeformableTaskArenaReadback,
    NativeDeformableTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_arena_construction_worker import (
    DEPENDENCY_IMPORTS,
    _bind_native_task_readback,
    _load_and_verify_manifest,
    _native_capability_blockers,
    _requested_arm_reset,
    _task_camera_entity_ids,
)
from blueprint_pipeline.native_task_arena_import_scope import ROBOT_EMBODIMENT_MODULES
from blueprint_pipeline.native_task_runtime_source_provision import TOP_LEVEL_PACKAGES


def test_worker_source_contains_no_scene_or_task_object_identity() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.native_task_arena_construction_worker",
            fromlist=["x"],
        ).__file__
    ).read_text(encoding="utf-8")

    for forbidden in (
        "840313",
        "840796",
        "refrigerator",
        "approved_can",
        "towel",
        "cloth",
        "basket",
    ):
        assert forbidden not in source


def _deformable_reset_recipe() -> dict:
    operations = [
        "load_default_nodal_state",
        "zero_nodal_velocities",
        "write_nodal_state_to_sim_index",
        "write_nodal_kinematic_target_to_sim_index",
        "readback_physx_root_view_state_and_kinematic_target",
    ]
    return {
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


def _deformable_built() -> SimpleNamespace:
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
        entity_reset_recipes_by_entity_id={"movable-alpha": _deformable_reset_recipe()},
    )


def test_multi_entity_deformable_route_binds_exact_readback_and_contact_blocker() -> None:
    built = _deformable_built()

    readback, binding = _bind_native_task_readback(built)

    assert type(readback) is NativeDeformableTaskArenaReadback
    assert binding == {
        "schema_version": "native_task_worker_readback_binding.v1",
        "task_kind": "deformable_transfer",
        "task_entity_ids": [
            "destination-alpha",
            "movable-alpha",
            "obstacle-alpha",
            "robot-alpha",
            "support-alpha",
        ],
        "readback_source": "native_entity_keyed_deformable_readback",
        "legacy_task_object_alias_used": False,
        "evaluation_or_scoring_admitted": False,
        "blockers": [CONTACT_CAPABILITY_BLOCKER],
    }
    assert _task_camera_entity_ids(built.plan) == [
        "destination-alpha",
        "movable-alpha",
    ]
    assert "task_object" not in built.scene_asset_names
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match=CONTACT_CAPABILITY_BLOCKER,
    ) as exc_info:
        readback.ensure_evaluation_capable()
    assert _native_capability_blockers(exc_info.value) == [CONTACT_CAPABILITY_BLOCKER]

    class SelfAttestedBlocker(RuntimeError):
        errors = (CONTACT_CAPABILITY_BLOCKER,)

    assert _native_capability_blockers(SelfAttestedBlocker()) == []


def test_legacy_rigid_and_articulated_readback_routes_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rigid = SimpleNamespace(plan={"task_kind": "rigid_pick_place"})
    rigid_readback, rigid_binding = _bind_native_task_readback(rigid)
    assert rigid_readback is None
    assert rigid_binding["readback_source"] == ("native_rigid_body_environment_readback")
    assert rigid_binding["blockers"] == []

    sentinel = object()
    from blueprint_pipeline import native_task_arena_readback as readback_module

    monkeypatch.setattr(
        readback_module,
        "NativeArticulatedTaskArenaReadback",
        lambda built: sentinel,
    )
    articulated = SimpleNamespace(plan={"task_kind": "articulated_open_close"})
    articulated_readback, articulated_binding = _bind_native_task_readback(articulated)
    assert articulated_readback is sentinel
    assert articulated_binding["readback_source"] == ("native_articulated_task_readback")
    assert articulated_binding["legacy_task_object_alias_used"] is True
    assert articulated_binding["blockers"] == []


def test_dependency_matrix_is_declared_as_one_preflight() -> None:
    assert ROBOT_EMBODIMENT_MODULES == {"franka_panda": "isaaclab_arena.embodiments.droid.droid"}
    assert {
        "torch",
        "pxr.UsdVol",
        "gymnasium",
        "lazy_loader",
        "cloudpickle",
        "farama_notifications",
        "packaging",
        "prettytable",
        "typing_extensions",
        "wcwidth",
        "h5py",
        "yaml",
        "toml",
        "antlr4",
        "omegaconf",
        "hydra",
        "hydra.core",
        "msgpack",
        "zmq",
        "rsl_rl",
        "rsl_rl.runners",
        "tensordict",
        "importlib_metadata",
        "orjson",
        "pyvers",
        "git",
        "gitdb",
        "smmap",
        "lightwheel_sdk",
        "lightwheel_sdk.loader",
        "requests",
        "charset_normalizer",
        "idna",
        "urllib3",
        "certifi",
        "tqdm",
        "termcolor",
        "click",
        "isaaclab_contrib",
        "isaaclab_newton",
        "isaaclab.controllers",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_teleop",
        "isaaclab_arena.environments.arena_env_builder",
    }.issubset(DEPENDENCY_IMPORTS)
    assert set(TOP_LEVEL_PACKAGES).issubset(DEPENDENCY_IMPORTS)
    assert DEPENDENCY_IMPORTS.index("isaaclab_contrib") < DEPENDENCY_IMPORTS.index(
        "isaaclab_arena.environments.arena_env_builder"
    )
    assert DEPENDENCY_IMPORTS.index("isaaclab_newton") < DEPENDENCY_IMPORTS.index(
        "isaaclab_arena.environments.arena_env_builder"
    )


def test_manifest_binding_rejects_tamper_before_isaac(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "construction_canary",
        "implementation_commit": "a" * 40,
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    (tmp_path / "adp_arena_provider_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert _load_and_verify_manifest(tmp_path)["input_digest"].startswith("sha256:")

    manifest["execution_mode"] = "policy"
    (tmp_path / "adp_arena_provider_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    try:
        _load_and_verify_manifest(tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "native_task_construction_manifest_invalid"
    else:  # pragma: no cover - explicit fail message is clearer than pytest magic
        raise AssertionError("tampered execution mode was accepted")


def test_reset_readback_uses_semantic_joint_order_not_json_key_order() -> None:
    resets = {
        "finger_joint": 9.0,
        **{f"panda_joint{index}": float(index) for index in range(7, 0, -1)},
    }

    result = _requested_arm_reset(
        plan={"robot": {"joint_reset_positions_rad": resets}},
        servo_binding={"arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)]},
    )

    assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
