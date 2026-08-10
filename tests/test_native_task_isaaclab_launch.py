from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_isaaclab_launch import (
    NativeTaskIsaacLabLaunchError,
    PRE_APP_DEPENDENCY_FILENAME,
    PRE_APP_DEPENDENCY_IMPORTS,
    PRE_APP_DEPENDENCY_SCHEMA_VERSION,
    launch_native_task_isaaclab,
    preflight_native_task_pre_app_dependencies,
    verify_native_task_isaaclab_launch_contract,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_ISAACLAB_SUBMODULE_PATH,
    ARENA_REPOSITORY,
    ISAAC_SIM_BASE_IMAGE,
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
    apps = extraction / "runtime_sources" / "isaaclab" / "apps"
    apps.mkdir(parents=True)
    base = (
        '[dependencies]\n"isaacsim.core.simulation_manager" = {}\n"omni.warp.core" = {}\n'
        if old_conflicting_experience
        else '[dependencies]\n"isaacsim.core.simulation_manager" = {}\n'
        '"omni.physx.bundle" = {}\n'
        '[settings.app.extensions]\nexcluded = ["omni.warp.core"]\n'
    )
    (apps / "isaaclab.python.kit").write_text(base, encoding="utf-8")
    (apps / "isaaclab.python.headless.kit").write_text(
        '[dependencies]\n"omni.physx" = {}\n'
        '[settings]\napp.extensions.excluded = ["omni.warp.core"]\n',
        encoding="utf-8",
    )
    experience = apps / "isaaclab.python.headless.rendering.kit"
    experience.write_text('[dependencies]\n"isaaclab.python.headless" = {}\n', encoding="utf-8")
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
            "upstream_fix_revisions": list(ISAACLAB_RUNTIME_COMPATIBILITY_UPSTREAM_FIXES),
            "path": str(experience),
            "sha256": _sha256(experience),
        },
        "paired_stack": {
            "arena_repository": ARENA_REPOSITORY,
            "arena_revision": ARENA_COMMIT,
            "isaaclab_repository": ISAACLAB_REPOSITORY,
            "isaaclab_revision": ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT,
            "isaaclab_submodule_path": ARENA_ISAACLAB_SUBMODULE_PATH,
            "simulator_base_image": ISAAC_SIM_BASE_IMAGE,
            "simulator_dockerfile_path": "docker/Dockerfile.isaaclab_arena",
            "dockerfile_sha256": "sha256:" + "a" * 64,
            "gitmodules_sha256": "sha256:" + "b" * 64,
        },
        "runtime_dependencies_installed": [
            {
                "package": "warp-lang",
                "version": "1.13.0",
                "pure_python": False,
                "wheel_tag": "py3-none-manylinux_2_28_x86_64",
            }
        ],
        "runtime_import_probe_returncode": 0,
        "runtime_import_probes": [
            {
                "module": "warp",
                "available": True,
                "expected_version": "1.13.0",
                "observed_version": "1.13.0",
                "version_matches": True,
            }
        ],
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    path = tmp_path / "native_task_runtime_source_provisioning.v1.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def _pre_app_result(*, blockers: list[str] | None = None) -> dict:
    errors = list(blockers or [])
    result = {
        "schema_version": PRE_APP_DEPENDENCY_SCHEMA_VERSION,
        "status": "qualified" if not errors else "blocked",
        "imports": [{"module": name, "available": True} for name in PRE_APP_DEPENDENCY_IMPORTS],
        "import_count": len(PRE_APP_DEPENDENCY_IMPORTS),
        "all_declared_imports_attempted": True,
        "torch_cuda": {
            "available": True,
            "runtime_version": "12.8",
            "device_count": 1,
            "device_name": "fixture-gpu",
        },
        "simulation_app_started": False,
        "candidate_policy_queried": False,
        "blockers": errors,
        "raw_secret_values_recorded": False,
        "matrix_digest": "",
    }
    result["matrix_digest"] = canonical_digest(result, digest_field="matrix_digest")
    return result


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
        extension_enabled_reader=lambda extension_id: False,
        pre_app_dependency_probe=_pre_app_result,
    )

    assert app is not None
    assert calls == [
        (
            {"headless": True, "renderer": "RayTracedLighting"},
            receipt["experience"]["path"],
        )
    ]
    assert receipt["bundled_isaac_sim_warp_extension_loaded"] is False
    assert receipt["bundled_warp_extension_readback"]["observed_enabled"] is False
    assert receipt["external_warp"]["import_qualified_before_simulation_app"] is True
    assert receipt["pre_app_dependency_matrix"]["status"] == "qualified"
    assert Path(receipt["pre_app_dependency_matrix"]["path"]).name == (PRE_APP_DEPENDENCY_FILENAME)
    assert receipt["simulation_manager_lifecycle"]["observed_value"] is False
    assert receipt["simulation_manager_lifecycle"]["applied_before_extension_startup"] is True
    assert receipt["launch_receipt_digest"] == canonical_digest(
        receipt, digest_field="launch_receipt_digest"
    )
    assert {row["filename"] for row in receipt["experience_files"]} == {
        "isaaclab.python.kit",
        "isaaclab.python.headless.kit",
        "isaaclab.python.headless.rendering.kit",
    }


def test_pre_app_matrix_attempts_every_declared_dependency_and_reports_all_gaps() -> None:
    missing = {"onnxruntime", "vuer", "sbi"}
    versions = {
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "numpy": "2.4.4",
        "prettytable": "3.3.0",
        "gymnasium": "1.2.1",
        "transformers": "4.57.6",
        "warp-lang": "1.13.0",
        "Pillow": "12.2.0",
        "typing_extensions": "4.12.2",
        "h5py": "3.16.0",
        "packaging": "23.2",
        "pandas": "2.2.3",
        "tqdm": "4.67.1",
    }
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: f"fixture-{index}",
        ),
    )

    def importer(name: str):
        if name in missing:
            raise ModuleNotFoundError(name)
        return torch if name == "torch" else SimpleNamespace(__version__="1.0")

    result = preflight_native_task_pre_app_dependencies(
        module_importer=importer,
        distribution_version_reader=lambda name: versions[name],
    )

    assert result["status"] == "blocked"
    assert result["all_declared_imports_attempted"] is True
    assert result["import_count"] == len(PRE_APP_DEPENDENCY_IMPORTS)
    assert {row["module"] for row in result["imports"] if row["available"] is False} == missing
    assert set(result["blockers"]) == {
        f"native_task_pre_app_dependency_missing:{name}" for name in missing
    }
    assert result["torch_cuda"]["runtime_version"] == "12.8"
    assert result["matrix_digest"] == canonical_digest(result, digest_field="matrix_digest")


def test_pre_app_failure_is_persisted_and_blocks_before_simulation_app(
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_factory(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("SimulationApp must not start")

    blocker = "native_task_pre_app_dependency_missing:torch"
    with pytest.raises(NativeTaskIsaacLabLaunchError, match=blocker):
        launch_native_task_isaaclab(
            _receipt(tmp_path),
            simulation_app_factory=forbidden_factory,
            pre_app_dependency_probe=lambda: _pre_app_result(blockers=[blocker]),
        )

    assert called is False
    retained = json.loads((tmp_path / PRE_APP_DEPENDENCY_FILENAME).read_text())
    assert retained["status"] == "blocked"
    assert retained["simulation_app_started"] is False
    assert retained["blockers"] == [blocker]


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
        launch_native_task_isaaclab(receipt_path, simulation_app_factory=forbidden_factory)
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
        extension_enabled_reader=lambda extension_id: False,
        pre_app_dependency_probe=_pre_app_result,
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
        launch_native_task_isaaclab(_receipt(tmp_path), simulation_app_factory=factory)
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
            simulation_app_factory=lambda *args, **kwargs: SimpleNamespace(close=close),
            settings_reader=lambda setting: True,
            extension_enabled_reader=lambda extension_id: False,
            pre_app_dependency_probe=_pre_app_result,
        )
    assert closed is True


def test_live_bundled_warp_extension_closes_app_and_fails_typed(
    tmp_path: Path,
) -> None:
    closed = False

    def close():
        nonlocal closed
        closed = True

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_bundled_warp_extension_live",
    ):
        launch_native_task_isaaclab(
            _receipt(tmp_path),
            simulation_app_factory=lambda *args, **kwargs: SimpleNamespace(close=close),
            settings_reader=lambda setting: False,
            extension_enabled_reader=lambda extension_id: True,
            pre_app_dependency_probe=_pre_app_result,
        )
    assert closed is True


def test_live_extension_readback_error_closes_app_and_fails_typed(
    tmp_path: Path,
) -> None:
    closed = False

    def close():
        nonlocal closed
        closed = True

    def broken_reader(extension_id: str) -> bool:
        raise RuntimeError(extension_id)

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_live_extension_readback_failed",
    ):
        launch_native_task_isaaclab(
            _receipt(tmp_path),
            simulation_app_factory=lambda *args, **kwargs: SimpleNamespace(close=close),
            settings_reader=lambda setting: False,
            extension_enabled_reader=broken_reader,
            pre_app_dependency_probe=_pre_app_result,
        )
    assert closed is True


def test_unpaired_arena_and_isaaclab_receipt_fails_before_launch(
    tmp_path: Path,
) -> None:
    receipt_path = _receipt(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["paired_stack"]["isaaclab_revision"] = "0" * 40
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        NativeTaskIsaacLabLaunchError,
        match="native_task_isaaclab_paired_stack_mismatch",
    ):
        verify_native_task_isaaclab_launch_contract(receipt_path)


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
