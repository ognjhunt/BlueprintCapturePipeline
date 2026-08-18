from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_construction_worker import (
    DEPENDENCY_IMPORTS,
    _initial_contact_blocked,
    _load_and_verify_manifest,
    _pose_arrival_readback,
    _requested_arm_reset,
    _retain_task_path_samples,
    _task_joint_reset_passed,
    _verified_construction_phase_plan_path,
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

    for forbidden in ("840313", "840796", "refrigerator", "approved_can"):
        assert forbidden not in source


def test_dependency_matrix_is_declared_as_one_preflight() -> None:
    assert ROBOT_EMBODIMENT_MODULES == {
        "franka_panda": "isaaclab_arena.embodiments.droid.droid"
    }
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
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
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
        servo_binding={
            "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)]
        },
    )

    assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_worker_does_not_require_articulated_joint_tolerance_for_rigid_task() -> None:
    assert _task_joint_reset_passed(
        absolute_errors_rad={},
        task_spec={"task_kind": "rigid_pick_place"},
    ) is True


def test_worker_reverifies_frozen_construction_plan_before_native_startup(
    tmp_path: Path,
) -> None:
    runtime_inputs = tmp_path / "runtime_inputs"
    runtime_inputs.mkdir()
    plan = runtime_inputs / "native_task_construction_phase_plan.v1.json"
    plan.write_text('{"plan_digest":"sha256:fixture"}\n', encoding="utf-8")
    manifest = {
        "bound_runtime_inputs": [
            {
                "relative_path": (
                    "runtime_inputs/native_task_construction_phase_plan.v1.json"
                ),
                "size_bytes": plan.stat().st_size,
                "sha256": "sha256:"
                + hashlib.sha256(plan.read_bytes()).hexdigest(),
            }
        ]
    }

    assert _verified_construction_phase_plan_path(tmp_path, manifest) == plan
    plan.write_text("tampered\n", encoding="utf-8")
    try:
        _verified_construction_phase_plan_path(tmp_path, manifest)
    except RuntimeError as exc:
        assert str(exc) == "native_task_construction_phase_plan_identity_mismatch"
    else:
        raise AssertionError("tampered construction phase plan was accepted")


def test_rigid_initial_support_force_is_not_misclassified_as_penetration() -> None:
    sample = {
        "task_robot_contact_peak_force_n": 0.0,
        "task_support_contact_peak_force_n": 20.0,
        "task_scene_collision_peak_force_n": 0.0,
        "robot_scene_contact_peak_force_n": 0.0,
        "robot_task_forbidden_collision_peak_force_n": 0.0,
    }

    assert _initial_contact_blocked(
        task_kind="rigid_pick_place",
        sample=sample,
        collision_threshold_n=1.0,
    ) is False
    sample["task_scene_collision_peak_force_n"] = 2.0
    assert _initial_contact_blocked(
        task_kind="rigid_pick_place",
        sample=sample,
        collision_threshold_n=1.0,
    ) is True
    assert _task_joint_reset_passed(
        absolute_errors_rad={"hinge": 5.0e-5},
        task_spec={"reset_tolerance_rad": 1.0e-4},
    ) is True


def test_worker_arrival_requires_orientation_as_well_as_position() -> None:
    result = _pose_arrival_readback(
        position_world_m=[1.0, 2.0, 3.0],
        target_position_world_m=[1.0, 2.0, 3.0],
        orientation_world_xyzw=[1.0, 0.0, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.05,
    )

    assert result["position_error_m"] == 0.0
    assert result["orientation_error_rad"] > 0.05
    assert result["reached"] is False


def test_worker_retains_path_samples_for_general_graph_and_rigid_tasks() -> None:
    assert _retain_task_path_samples(
        task_kind="articulated_open_close",
        task_spec={"schema_version": "adp_task_spec.v2"},
    )
    assert _retain_task_path_samples(
        task_kind="rigid_pick_place",
        task_spec={"schema_version": "adp_task_spec.v2"},
    )
    assert not _retain_task_path_samples(
        task_kind="articulated_open_close",
        task_spec={"schema_version": "adp_task_spec.v1"},
    )


def test_articulation_device_binding_names_the_asset_on_the_wrong_device() -> None:
    """A cuda/cpu mismatch must name the asset, not just a kernel argument.

    Isaac Lab raises this from inside a Warp kernel launch:

        Error launching kernel 'get_joint_acc_from_joint_vel', device='cuda:0',
        but input array for argument 'joint_vel' is on device=cpu

    That message cannot say which articulation is wrong, and attempts r6-r9
    each spent ~$0.065 failing on it. This reads every articulation directly
    so one run identifies the asset.
    """

    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_device_binding,
    )

    def _asset(joint_device: str, joints: int, actuators: int) -> SimpleNamespace:
        return SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(device=joint_device),
                joint_vel=SimpleNamespace(device=joint_device),
                device="cuda:0",
            ),
            joint_names=[f"j{i}" for i in range(joints)],
            actuators={f"a{i}": None for i in range(actuators)},
        )

    scene = {
        "robot": _asset("cuda:0", 13, 8),
        "task_object": _asset("cpu", 5, 0),
    }
    built = SimpleNamespace(
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)),
        scene_asset_names={"task_object": "task_object"},
    )

    binding = _articulation_device_binding(built, expected_device="cuda:0")

    assert binding["articulations"]["robot"]["on_expected_device"] is True
    offender = binding["articulations"]["task_object"]
    assert offender["on_expected_device"] is False
    assert offender["joint_vel"] == "cpu"
    # the numbers that distinguish the assets are recorded too
    assert offender["num_joints"] == 5
    assert offender["num_actuators"] == 0


def test_device_binding_survives_an_unreachable_scene() -> None:
    """Diagnostics must never replace the real failure with their own."""

    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_device_binding,
    )

    class _Boom:
        @property
        def scene(self):
            raise RuntimeError("scene not ready")

    built = SimpleNamespace(
        env=SimpleNamespace(unwrapped=_Boom()), scene_asset_names={}
    )

    binding = _articulation_device_binding(built, expected_device="cuda:0")

    assert "unavailable" in binding
    assert binding["expected_device"] == "cuda:0"
