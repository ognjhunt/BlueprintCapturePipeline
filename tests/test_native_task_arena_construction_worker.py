from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_construction_worker import (
    DEPENDENCY_IMPORTS,
    _gripper_geometry,
    _load_and_verify_manifest,
    _requested_arm_reset,
    preflight_native_dependency_matrix,
)
from blueprint_pipeline.native_task_arena_import_scope import ROBOT_EMBODIMENT_MODULES
from blueprint_pipeline.native_task_dependency_profiles import (
    CONSTRUCTION_CONTROLS_DEFERRED_MODULES,
    CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE,
    CONSTRUCTION_CONTROLS_REQUIRED_MODULES,
    SIMULATION_APP_OWNED_MODULE_ROOTS,
    construction_controls_deferred_dependencies,
)
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
    assert '_announce("simulation_app")' not in source
    assert '_announce("pre_app_and_simulation_launch")' in source
    assert source.index(
        "simulation_app, launch_receipt = launch_native_task_isaaclab("
    ) < source.index("dependency_matrix = preflight_native_dependency_matrix(")
    assert source.index(
        'result["phase_reached"] = "environment_build"'
    ) < source.index("built = build_native_task_arena_environment(")


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
        "rsl_rl",
        "rsl_rl.runners",
        "tensordict",
        "importlib_metadata",
        "orjson",
        "pyvers",
        "git",
        "gitdb",
        "smmap",
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
        "isaaclab_arena.environments.arena_env_builder_cfg",
    }.issubset(DEPENDENCY_IMPORTS)
    assert set(DEPENDENCY_IMPORTS).isdisjoint(CONSTRUCTION_CONTROLS_DEFERRED_MODULES)
    assert set(CONSTRUCTION_CONTROLS_REQUIRED_MODULES).issubset(DEPENDENCY_IMPORTS)
    assert {row["module"] for row in construction_controls_deferred_dependencies()} == (
        CONSTRUCTION_CONTROLS_DEFERRED_MODULES
    )
    assert set(TOP_LEVEL_PACKAGES).issubset(DEPENDENCY_IMPORTS)
    assert DEPENDENCY_IMPORTS.index("isaaclab_contrib") < DEPENDENCY_IMPORTS.index(
        "isaaclab_arena.environments.arena_env_builder"
    )
    assert DEPENDENCY_IMPORTS.index("isaaclab_newton") < DEPENDENCY_IMPORTS.index(
        "isaaclab_arena.environments.arena_env_builder"
    )


def test_complete_dependency_import_and_cuda_probe_run_only_in_post_app_matrix() -> None:
    attempted: list[str] = []
    torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: f"fixture-gpu-{index}",
        ),
    )

    def importer(name: str):
        attempted.append(name)
        return torch if name == "torch" else SimpleNamespace(__version__="fixture")

    result = preflight_native_dependency_matrix(
        robot_id="franka_panda",
        module_importer=importer,
        run_command=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="fixture tool 1.0\n", stderr=""
        ),
        loaded_module_names_reader=lambda: set(SIMULATION_APP_OWNED_MODULE_ROOTS),
        embodiment_installer=lambda robot_id: {
            "schema_version": "native_task_arena_embodiment_scope.v1",
            "robot_id": robot_id,
            "status": "installed",
        },
    )

    assert result["probe_phase"] == "post_simulation_app"
    assert result["simulation_app_started"] is True
    assert result["post_app_module_execution_performed"] is True
    assert result["all_required_imports_attempted"] is True
    assert result["runtime_owned_namespace_roots_missing"] == []
    assert result["runtime_owned_namespace_roots_present"] == sorted(
        SIMULATION_APP_OWNED_MODULE_ROOTS
    )
    assert result["torch_cuda"] == {
        "probe_phase": "post_simulation_app",
        "available": True,
        "runtime_version": "12.8",
        "device_count": 1,
        "device_name": "fixture-gpu-0",
    }
    assert result["all_required_available"] is True
    assert result["blockers"] == []
    assert attempted == list(DEPENDENCY_IMPORTS)


def test_manifest_binding_rejects_tamper_before_isaac(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "construction_canary",
        "dependency_profile": CONSTRUCTION_CONTROLS_DEPENDENCY_PROFILE,
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


def test_worker_measures_authored_tool_points_and_retains_raw_origin_diagnostic() -> None:
    import math

    import numpy as np

    half = math.sqrt(0.5)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["left_inner_finger", "right_inner_finger"],
            body_pose_w=[
                [
                    [-0.046, 0.0, 0.0, 0.0, half, 0.0, half],
                    [0.046, 0.0, 0.0, 0.0, -half, 0.0, half],
                ]
            ],
        )
    )
    result = _gripper_geometry(
        robot,
        torch=SimpleNamespace(
            as_tensor=np.asarray,
            linalg=SimpleNamespace(vector_norm=np.linalg.norm),
        ),
        grasp_frame={
            "kind": "body_local_point_midpoint",
            "body_names": ["left_inner_finger", "right_inner_finger"],
            "body_local_points_m": {
                "left_inner_finger": [0.0, 0.0, 0.046],
                "right_inner_finger": [0.0, 0.0, 0.046],
            },
        },
    )

    assert result["separation_m"] == pytest.approx(0.0, abs=1.0e-8)
    assert result["body_origin_separation_m"] == pytest.approx(0.092)
    assert result["measurement_authority"] == (
        "native_body_poses_plus_authored_local_tool_points"
    )
