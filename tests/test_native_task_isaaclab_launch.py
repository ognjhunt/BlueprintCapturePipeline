from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_isaaclab_launch import (
    NativeTaskIsaacLabLaunchError,
    launch_native_task_isaaclab,
    verify_native_task_isaaclab_launch_contract,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ISAACLAB_REPOSITORY,
    ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
    ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
    ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES,
    RUNTIME_EXPERIENCE_RELATIVE_PATH,
)


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(tmp_path: Path, *, old_conflicting_experience: bool = False) -> Path:
    extraction = tmp_path / "extracted"
    apps = (
        extraction
        / "runtime_sources"
        / "isaaclab_runtime_compatibility"
        / "apps"
    )
    apps.mkdir(parents=True)
    base = (
        '[dependencies]\n"isaacsim.core.simulation_manager" = {}\n'
        '"omni.warp.core" = {}\n'
        if old_conflicting_experience
        else '[dependencies]\n"omni.physics.physx" = {}\n'
        '[settings.app.extensions]\nexcluded = ["omni.warp.core"]\n'
    )
    (apps / "isaaclab.python.kit").write_text(base, encoding="utf-8")
    (apps / "isaaclab.python.headless.kit").write_text(
        '[dependencies]\n"omni.physics.physx" = {}\n'
        '[settings]\napp.extensions.excluded = ["omni.warp.core"]\n',
        encoding="utf-8",
    )
    experience = apps / "isaaclab.python.headless.rendering.kit"
    experience.write_text(
        '[dependencies]\n"isaaclab.python.headless" = {}\n', encoding="utf-8"
    )
    result = {
        "schema_version": "native_task_runtime_source_provisioning.v1",
        "status": "completed",
        "python_executable": "/isaac-sim/python.sh",
        "python_executable_source": "simulator_python_launcher",
        "python_probe_flag": "-P",
        "python_probe_mode": "simulator_wrapper_safe_path",
        "extraction_dir": str(extraction),
        "runtime_experience": {
            "relative_path": RUNTIME_EXPERIENCE_RELATIVE_PATH,
            "repository": ISAACLAB_REPOSITORY,
            "source_revision": ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
            "source_tree": ISAACLAB_RUNTIME_COMPATIBILITY_TREE,
            "upstream_fix_revisions": list(
                ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES
            ),
            "path": str(experience),
            "sha256": _sha256(experience),
        },
        "runtime_dependencies_installed": [
            {
                "package": "warp-lang",
                "version": "1.12.0",
                "pure_python": False,
                "wheel_tag": "py3-none-manylinux_2_28_x86_64",
            }
        ],
        "runtime_import_probe_returncode": 0,
        "runtime_import_probes": [
            {
                "module": "warp",
                "available": True,
                "expected_version": "1.12.0",
                "observed_version": "1.12.0",
                "version_matches": True,
            }
        ],
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    path = tmp_path / "native_task_runtime_source_provisioning.v1.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_launch_uses_exact_compatible_experience_as_a_real_input(
    tmp_path: Path,
) -> None:
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(app=SimpleNamespace(close=lambda: None))

    app, receipt = launch_native_task_isaaclab(
        _receipt(tmp_path), device="cuda:0", app_launcher_factory=factory
    )

    assert app is not None
    assert calls == [
        {
            "headless": True,
            "device": "cuda:0",
            "enable_cameras": True,
            "experience": receipt["experience"]["path"],
        }
    ]
    assert receipt["bundled_isaac_sim_warp_extension_loaded"] is False
    assert receipt["external_warp"]["import_qualified_before_simulation_app"] is True
    assert {row["filename"] for row in receipt["experience_files"]} == {
        "isaaclab.python.kit",
        "isaaclab.python.headless.kit",
        "isaaclab.python.headless.rendering.kit",
    }


def test_old_experience_that_loads_two_warp_runtimes_fails_before_launch(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_experience_warp_contract_invalid",
    ):
        verify_native_task_isaaclab_launch_contract(
            _receipt(tmp_path, old_conflicting_experience=True)
        )


def test_experience_byte_or_revision_drift_fails_closed(tmp_path: Path) -> None:
    receipt_path = _receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["runtime_experience"]["source_revision"] = "0" * 40
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_experience_revision_mismatch",
    ):
        verify_native_task_isaaclab_launch_contract(receipt_path)


def test_missing_external_warp_fails_before_simulation_app_factory(
    tmp_path: Path,
) -> None:
    receipt_path = _receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["runtime_import_probe_returncode"] = 1
    value["runtime_import_probes"] = []
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    called = False

    def forbidden_factory(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("SimulationApp must not launch without external Warp")

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_external_warp_import_unqualified",
    ):
        launch_native_task_isaaclab(
            receipt_path, device="cuda:0", app_launcher_factory=forbidden_factory
        )
    assert called is False


def test_underlying_kit_python_binary_is_rejected_before_launch(
    tmp_path: Path,
) -> None:
    receipt_path = _receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["python_executable"] = "/isaac-sim/kit/python/bin/python3"
    value["python_executable_source"] = "current_interpreter_fallback"
    value["python_probe_flag"] = "-I"
    value["python_probe_mode"] = "isolated_interpreter"
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_runtime_launcher_invalid",
    ):
        verify_native_task_isaaclab_launch_contract(receipt_path)


def test_simulator_wrapper_with_isolated_flag_is_rejected_before_launch(
    tmp_path: Path,
) -> None:
    receipt_path = _receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["python_probe_flag"] = "-I"
    value["python_probe_mode"] = "isolated_interpreter"
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_runtime_launcher_invalid",
    ):
        verify_native_task_isaaclab_launch_contract(receipt_path)


def test_launch_goes_through_app_launcher_not_a_bare_simulation_app() -> None:
    """The launcher is what configures the device; a bare app leaves it undone.

    Attempts r6-r11 (~$0.37) all died with

        Error launching kernel 'get_joint_acc_from_joint_vel', device='cuda:0',
        but input array for argument 'joint_vel' is on device=cpu

    A controlled A/B on one machine (2026-08-19) isolated it to this single
    variable: the same sealed plan, assets and GPU built cleanly with every
    articulation on cuda:0 under `AppLauncher`, and reproduced the production
    failure exactly under `SimulationApp({...}, experience=...)`. Nothing warns
    when the device goes unconfigured -- SimulationContext and PhysicsManager
    both keep reporting cuda:0 while the PhysX views hand back CPU arrays.
    """

    import ast
    import inspect
    import textwrap

    from blueprint_pipeline import native_task_isaaclab_launch

    # inspect the code, not the prose: the docstring names SimulationApp in
    # order to explain why it must not be used
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(native_task_isaaclab_launch.launch_native_task_isaaclab)
        )
    )
    function = tree.body[0]
    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
    ):
        function.body = function.body[1:]  # drop the docstring
    names = {
        node.id for node in ast.walk(ast.Module(body=function.body, type_ignores=[]))
        if isinstance(node, ast.Name)
    } | {
        alias.name for node in ast.walk(ast.Module(body=function.body, type_ignores=[]))
        if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert "AppLauncher" in names
    assert "SimulationApp" not in names
    signature = inspect.signature(
        native_task_isaaclab_launch.launch_native_task_isaaclab
    )
    # the device is required, so no caller can launch without naming one
    assert signature.parameters["device"].default is inspect.Parameter.empty


def test_launch_refuses_an_empty_device(tmp_path: Path) -> None:
    """Fail closed rather than launch a device-less app that looks healthy."""

    def forbidden(**kwargs):  # pragma: no cover - must never run
        raise AssertionError("launched without a device")

    with pytest.raises(NativeTaskIsaacLabLaunchError) as excinfo:
        launch_native_task_isaaclab(
            _receipt(tmp_path), device="  ", app_launcher_factory=forbidden
        )

    assert "native_task_isaaclab_device_missing" in excinfo.value.errors


def test_every_arena_worker_launches_on_the_shared_device() -> None:
    """All three links must name one device; drift is how lanes diverge."""

    import inspect

    from blueprint_pipeline import (
        native_task_arena_construction_worker,
        native_task_arena_controls_worker,
        native_task_arena_policy_worker,
    )
    from blueprint_pipeline.native_task_isaaclab_launch import (
        NATIVE_TASK_ARENA_DEVICE,
    )

    assert NATIVE_TASK_ARENA_DEVICE == "cuda:0"
    for module in (
        native_task_arena_construction_worker,
        native_task_arena_controls_worker,
        native_task_arena_policy_worker,
    ):
        source = inspect.getsource(module)
        assert "launch_native_task_isaaclab(" in source
        assert "device=NATIVE_TASK_ARENA_DEVICE" in source, module.__name__


def test_no_arena_worker_hardcodes_a_device_literal() -> None:
    """One device string, or the components silently disagree about it.

    The launcher configures the device, the preconstruction probe asserts it,
    the environment build binds it and the readback verifies it. If any of them
    carries its own literal, changing the constant leaves them pointing at
    different devices -- and a device disagreement is exactly the failure that
    consumed r6-r11: every component kept reporting cuda:0 while the PhysX
    views handed back CPU arrays, with nothing warning that they had diverged.
    """

    import inspect

    from blueprint_pipeline import (
        native_task_arena_construction_worker,
        native_task_arena_controls_worker,
        native_task_arena_policy_worker,
    )

    for module in (
        native_task_arena_construction_worker,
        native_task_arena_controls_worker,
        native_task_arena_policy_worker,
    ):
        source = inspect.getsource(module)
        assert '"cuda:0"' not in source, f"{module.__name__} hardcodes a device"
        assert "'cuda:0'" not in source, f"{module.__name__} hardcodes a device"
        assert "NATIVE_TASK_ARENA_DEVICE" in source, module.__name__
