"""Development SDK doubles exercise real source/launch contracts, never GPU proof."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_task_runtime_capabilities as caps
from blueprint_pipeline import native_task_isaaclab_launch as launch
from tests.test_native_task_isaaclab_launch import _receipt, development_runtime_identity


@pytest.mark.parametrize("robot,team,expected", [
    ("franka", None, "pink_franka"), ("franka", {}, "joint"),
    ("unitree_g1", None, "g1_joint"),
])
def test_policy_controller_follows_actual_execution_path(robot, team, expected):
    assert caps.policy_controller(robot_id=robot, team_binding=team) == expected


@pytest.mark.parametrize("render_path,required", [("usd_geometry", False), ("no_site_appearance", False),
    ("plain_nurec_volume", True), ("particlefield_3d_gaussian_splat", True)])
def test_joint_launch_does_not_request_unused_sdk_extensions(tmp_path, render_path, required):
    calls = []
    _, value = launch.launch_native_task_isaaclab(_receipt(tmp_path), device="cuda:0", appearance_render_path=render_path,
        controller_backend="joint", app_launcher_factory=lambda **kw: calls.append(kw) or SimpleNamespace(app=object()),
        runtime_probe_factory=lambda *_a, **_kw: development_runtime_identity(),
        nurec_renderer_probe_factory=lambda: {"extension_enabled": required, "renderer_hints": 3 if required else None,
            "ujitso_geometry_enabled": required, "multi_gpu_enabled": False, "nurec_utils_extension_enabled": False})
    args = calls[0]["kit_args"]
    assert "pink" not in args and "nurec_utils" not in args
    assert ("omni.rtx.spg" in args) == required
    assert value["nurec_renderer"]["extension_required"] == required
    assert value["nurec_renderer"]["status"] == ("qualified" if required else "not_required")
    assert value["post_start_identity"]["gpu_interop"]["shared_pointer"] is True


@pytest.mark.parametrize("change", ["torch_version", "warp_origin", "bundled_warp", "cpu_tensor", "pink_missing"])
def test_actual_loaded_identity_is_required_after_kit(change):
    value = development_runtime_identity()
    backend = "joint"
    if change == "torch_version":
        value["modules"]["torch"]["version"] = "2.11.0+cu128"
    elif change == "warp_origin":
        value["modules"]["warp"]["origin_matches"] = False
    elif change == "bundled_warp":
        value["excluded_extensions"]["omni.warp.core"] = True
    elif change == "cpu_tensor":
        value["gpu_interop"]["torch_device"] = "cpu"
    else:
        backend = "pink_franka"
    with pytest.raises(ValueError):
        caps.validate_post_start_identity(value, controller_backend=backend)


def test_g1_python_provider_is_appended_without_loading_sdk_controller(tmp_path, monkeypatch):
    import sys
    roots = [tmp_path / "exts/isaacsim.robot_motion.pink/pip_prebundle",
             tmp_path / "exts/omni.isaac.core_archive/pip_prebundle"]
    for root in roots:
        root.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", ["retained-pinocchio-torch-warp"])
    assert caps.prepare_g1_python_dependencies(tmp_path) == list(map(str, roots))
    assert sys.path == ["retained-pinocchio-torch-warp", *map(str, roots)]


def test_every_native_launch_names_its_controller_and_franka_consumers_require_pink():
    package = Path(launch.__file__).parent
    required = {"native_task_arena_construction_worker.py", "native_task_arena_controls_worker.py",
                "native_task_arena_policy_worker.py", "native_task_arena_runtime_preflight_worker.py",
                "controlled_native_isaac.py"}
    seen = set()
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "launch_native_task_isaaclab":
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords}
            assert "controller_backend" in keywords, source.name
            if source.name in required:
                assert ast.literal_eval(keywords["controller_backend"]) == "pink_franka", source.name
                seen.add(source.name)
    assert seen == required


def test_selected_dependency_matrix_keeps_g1_pure_pink_but_skips_unused_sdk_pink(monkeypatch):
    from blueprint_pipeline import native_task_arena_construction_worker as worker
    from blueprint_pipeline import native_task_arena_import_scope as scope
    monkeypatch.setattr(scope, "install_scoped_arena_embodiment", lambda _id: {})
    monkeypatch.setattr(worker, "DEPENDENCY_IMPORTS", tuple(sorted(caps.PINK_IMPORTS)))
    requested = []
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: requested.append(name) or SimpleNamespace())
    monkeypatch.setattr(worker.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="development fixture"))
    worker.preflight_native_dependency_matrix(robot_id="franka", controller_backend="joint")
    assert not requested
    worker.preflight_native_dependency_matrix(robot_id="unitree_g1", controller_backend="g1_joint")
    assert set(requested) == {"pink", "pinocchio", "qpsolvers", "osqp"}
    requested.clear()
    worker.preflight_native_dependency_matrix(robot_id="franka", controller_backend="pink_franka")
    assert set(requested) == caps.PINK_IMPORTS


def test_native_source_receipt_is_still_verified_before_sdk_start(tmp_path):
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["runtime_experience"]["source_revision"] = "0" * 40
    receipt.write_text(json.dumps(value))
    with pytest.raises(launch.NativeTaskIsaacLabLaunchError):
        launch.launch_native_task_isaaclab(receipt, device="cuda:0", appearance_render_path="usd_geometry",
            app_launcher_factory=lambda **_kw: pytest.fail("changed source must not reach Kit"))
