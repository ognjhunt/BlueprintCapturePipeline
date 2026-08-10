from __future__ import annotations

import json
import sys
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

    def factory(config, *, experience):
        calls.append((config, experience))
        return SimpleNamespace(close=lambda: None)

    app, receipt = launch_native_task_isaaclab(
        _receipt(tmp_path),
        simulation_app_factory=factory,
        settings_reader=lambda setting: False,
    )

    assert app is not None
    assert calls == [
        (
            {"headless": True, "renderer": "RayTracedLighting"},
            receipt["experience"]["path"],
        )
    ]
    assert receipt["bundled_isaac_sim_warp_extension_loaded"] is False
    assert receipt["external_warp"]["import_qualified_before_simulation_app"] is True
    assert receipt["simulation_manager_lifecycle"]["observed_value"] is False
    assert receipt["simulation_manager_lifecycle"][
        "applied_before_extension_startup"
    ] is True
    assert receipt["launch_receipt_digest"] == canonical_digest(
        receipt, digest_field="launch_receipt_digest"
    )
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
            receipt_path, simulation_app_factory=forbidden_factory
        )
    assert called is False


def test_default_callbacks_are_disabled_before_factory_and_read_back(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.native_task_isaaclab_launch import (
        ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG,
        ISAAC_SIM_DEFAULT_CALLBACKS_SETTING,
    )

    observed_argv = []

    def factory(config, *, experience):
        del config, experience
        observed_argv.extend(sys.argv)
        return SimpleNamespace(close=lambda: None)

    launch_native_task_isaaclab(
        _receipt(tmp_path),
        simulation_app_factory=factory,
        settings_reader=lambda setting: (
            False if setting == ISAAC_SIM_DEFAULT_CALLBACKS_SETTING else None
        ),
    )

    assert ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG in observed_argv
    assert ISAAC_SIM_DEFAULT_CALLBACKS_KIT_ARG not in sys.argv


def test_conflicting_default_callback_setting_fails_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline.native_task_isaaclab_launch import (
        ISAAC_SIM_DEFAULT_CALLBACKS_SETTING,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", f"--{ISAAC_SIM_DEFAULT_CALLBACKS_SETTING}=true"],
    )
    called = False

    def factory(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("factory must not run with lifecycle conflict")

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_default_callbacks_setting_conflict",
    ):
        launch_native_task_isaaclab(
            _receipt(tmp_path), simulation_app_factory=factory
        )
    assert called is False


def test_default_callback_readback_failure_closes_app(tmp_path: Path) -> None:
    closed = False

    def close():
        nonlocal closed
        closed = True

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_default_callbacks_readback_failed",
    ):
        launch_native_task_isaaclab(
            _receipt(tmp_path),
            simulation_app_factory=lambda *args, **kwargs: SimpleNamespace(
                close=close
            ),
            settings_reader=lambda setting: True,
        )
    assert closed is True


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
