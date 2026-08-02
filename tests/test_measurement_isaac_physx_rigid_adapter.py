from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from blueprint_pipeline.measurement_adapter_runtime import build_measurement_adapter_descriptor
from blueprint_pipeline.measurement_isaac_physx_rigid_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    ISAAC_VERSION,
    PROTOCOL_ID,
    WORKER_SCRIPT,
    _bind_isaac_runtime_environment,
    _enable_installed_simulation_app_extension,
    _import_simulation_app,
    _installed_simulation_app_extension_roots,
    _observe_isaac_runtime_identity,
    implementation_digest,
    run_isaac_physx_rigid_measurement_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_capture_to_geometry_contact_v1/corpus.json"
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"isaac-physx-rigid-development-no-controller").hexdigest()
)


def _isaac_launcher() -> Path:
    raw = os.environ.get("BLUEPRINT_ISAAC_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_ISAAC_PYTHON exact Isaac 6.0.1 runtime is not configured")
    return Path(raw).absolute()


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _corpus_digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _request(case_index: int = 0) -> dict:
    corpus = _corpus()
    corpus_digest = _corpus_digest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="development-isaac-physx-tgs-rigid-drop-1",
        method_ids=["isaac-sim-6-physx"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 2 / 9,
        },
        compute_budget={"usd": 0.0, "maximum_duration_seconds": 900},
        minimum_repeated_trials=2,
    )
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        spec,
        case_id=f"{case_id}--isaac-sim-6-physx",
        split="development",
        input_artifact_digests=[corpus_digest],
        task_class="rigid_pick_place",
        material_regime="synthetic_rigid_body_drop",
        operating_point={
            **corpus["shared_operating_point"],
            "adapter_protocol": PROTOCOL_ID,
            **row,
        },
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("isaac-sim-6-physx"),
        spec,
        case,
        execution_id=f"isaac-physx-rigid-development-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="isaac-physx-cpu-tgs-rigid",
        precision="float32",
        seed=47,
        solver_settings={
            "solver_type": "TGS",
            "broadphase_type": "SAP",
            "gpu_dynamics": False,
            "enhanced_determinism": True,
            "position_iterations": 8,
            "velocity_iterations": 2,
        },
        timeout_seconds=300,
    )


def test_isaac_descriptor_is_exact_and_plan_only_request_builds() -> None:
    descriptor = build_measurement_adapter_descriptor("isaac-sim-6-physx")
    assert descriptor["target_version"] == ISAAC_VERSION
    assert descriptor["production_route_eligible"] is False
    request = _request()
    assert request["runtime_configuration"]["backend_id"] == "isaac-physx-cpu-tgs-rigid"
    assert request["implementation"]["implementation_digest"] == implementation_digest()


def test_isaac_worker_rejects_solver_and_identity_tampering_before_import() -> None:
    solver = copy.deepcopy(_request())
    solver["runtime_configuration"]["solver_settings"]["solver_type"] = "PGS"
    encoded = json.dumps(
        solver["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    solver["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    solver.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_configuration_invalid"):
        run_isaac_physx_rigid_measurement_request(solver)

    identity = _request()
    identity["implementation"]["implementation_id"] = "unbound-isaac-worker"
    identity.pop("execution_request_digest")
    result = run_isaac_physx_rigid_measurement_request(identity)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["isaac_physx_rigid_implementation_id_mismatch"]


def test_isaac_runtime_identity_uses_app_version_file_without_dist_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    class FakeApp:
        @staticmethod
        def get_app_version() -> str:
            return "Isaac-Sim"

        @staticmethod
        def get_build_version() -> str:
            return "6.0.1+release.test"

    class FakeSimulationApp:
        app = FakeApp()

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)
    identity = _observe_isaac_runtime_identity(
        FakeSimulationApp(),
        version_getter=lambda: (ISAAC_VERSION, "", "6", "0", "1", "", "", ""),
    )
    assert identity == {
        "engine_version": ISAAC_VERSION,
        "engine_version_source": "isaacsim.core.version.get_version_app_VERSION_file",
        "observed_package_version": "unavailable",
        "observed_app_version": "Isaac-Sim",
        "observed_build_version": "6.0.1+release.test",
    }


def test_isaac_simulation_app_prefers_concrete_module_and_skips_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CallableSimulationApp:
        pass

    modules = {
        "isaacsim.simulation_app": types.SimpleNamespace(SimulationApp=CallableSimulationApp),
        "isaacsim": types.SimpleNamespace(SimulationApp=None),
    }
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        if name in modules:
            return modules[name]
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert _import_simulation_app() is CallableSimulationApp
    assert imported == ["isaacsim.simulation_app"]


def test_isaac_simulation_app_falls_back_after_noncallable_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacySimulationApp:
        pass

    modules = {
        "isaacsim.simulation_app": types.SimpleNamespace(SimulationApp=None),
        "isaacsim": types.SimpleNamespace(SimulationApp=None),
        "omni.isaac.kit": types.SimpleNamespace(SimulationApp=LegacySimulationApp),
    }
    monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])
    assert _import_simulation_app() is LegacySimulationApp


def test_isaac_simulation_app_failure_carries_safe_candidate_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> object:
        if name == "isaacsim":
            return types.SimpleNamespace(
                __file__="/isaac-sim/isaacsim/__init__.py", SimulationApp=None
            )
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match="simulation_app_not_callable") as captured:
        _import_simulation_app()
    diagnostics = captured.value.diagnostics  # type: ignore[attr-defined]
    assert [row["status"] for row in diagnostics["candidates"]] == [
        "import_failed",
        "noncallable",
        "import_failed",
    ]
    assert diagnostics["candidates"][1] == {
        "module": "isaacsim",
        "status": "noncallable",
        "module_file": "/isaac-sim/isaacsim/__init__.py",
        "symbol_present": True,
        "symbol_type": "NoneType",
        "symbol_callable": False,
    }


def test_isaac_simulation_app_extension_root_is_bounded_and_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_root = tmp_path / "exts/isaacsim.simulation_app"
    package_root = extension_root / "isaacsim/simulation_app"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    assert _installed_simulation_app_extension_roots(tmp_path) == [extension_root]

    root_package = types.SimpleNamespace(__path__=[])
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_physx_rigid_adapter._installed_simulation_app_extension_roots",
        lambda: [extension_root],
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: root_package)
    original_sys_path = list(sys.path)
    try:
        assert _enable_installed_simulation_app_extension() == [str(extension_root)]
        assert root_package.__path__ == [str(extension_root / "isaacsim")]
        assert sys.path[0] == str(extension_root)
    finally:
        sys.path[:] = original_sys_path


def test_isaac_runtime_environment_is_bound_inside_image_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "apps").mkdir()
    (tmp_path / "kit").mkdir()
    for name in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
        monkeypatch.delenv(name, raising=False)
    try:
        bound = _bind_isaac_runtime_environment(tmp_path)
        assert bound == {
            "ISAAC_PATH": str(tmp_path),
            "EXP_PATH": str(tmp_path / "apps"),
            "CARB_APP_PATH": str(tmp_path / "kit"),
        }
        assert {name: os.environ[name] for name in bound} == bound
    finally:
        for name in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
            os.environ.pop(name, None)


def test_isaac_runtime_environment_rejects_existing_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "apps").mkdir()
    (tmp_path / "kit").mkdir()
    mismatch = tmp_path / "other-apps"
    mismatch.mkdir()
    for name in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXP_PATH", str(mismatch))
    with pytest.raises(RuntimeError, match="exp_path_mismatch"):
        _bind_isaac_runtime_environment(tmp_path)


def test_isaac_runtime_identity_rejects_invalid_version_observation() -> None:
    with pytest.raises(RuntimeError, match="runtime_version_observation_invalid"):
        _observe_isaac_runtime_identity(object(), version_getter=lambda: ())


def test_isaac_worker_blocks_runtime_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class FakeSimulationApp:
        def __init__(self, _config: dict) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = FakeSimulationApp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_physx_rigid_adapter._bind_isaac_runtime_environment",
        lambda: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_physx_rigid_adapter._observe_isaac_runtime_identity",
        lambda _app: {
            "engine_version": "6.0.0",
            "engine_version_source": "isaacsim.core.version.get_version_app_VERSION_file",
            "observed_package_version": "unavailable",
            "observed_app_version": "Isaac-Sim",
            "observed_build_version": "test",
        },
    )
    result = run_isaac_physx_rigid_measurement_request(_request())
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["isaac_physx_rigid_runtime_version_mismatch"]
    assert result["runtime_observations"]["engine_version"] == "6.0.0"
    assert closed == [True]


@pytest.mark.slow
def test_isaac_external_runtime_executes_shared_cases() -> None:
    launcher = _isaac_launcher()
    bundles = [
        run_measurement_adapter_execution(
            _request(index),
            command_argv=[str(launcher), str(WORKER_SCRIPT)],
            execute=True,
        )
        for index in range(2)
    ]
    assert [row["receipt"]["status"] for row in bundles] == ["completed", "completed"]
    assert all(
        row["receipt"]["runtime_observations"]["engine_version"] == ISAAC_VERSION for row in bundles
    )
    assert all(
        row["receipt"]["runtime_observations"]["deterministic_replay_match"] is True
        for row in bundles
    )
    assert all(row["qualification_created"] is False for row in bundles)
    assert all(row["catalog_mutated"] is False for row in bundles)
