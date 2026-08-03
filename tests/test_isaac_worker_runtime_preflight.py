from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import isaac_worker_runtime_preflight as worker_preflight
from blueprint_pipeline.isaac_worker_runtime_preflight import (
    ISAAC_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    main as preflight_main,
    run_isaac_worker_runtime_preflight,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_isaac_worker_runtime_preflight_blocks_without_isaacsim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    output_path = tmp_path / "preflight.json"

    result = run_isaac_worker_runtime_preflight(
        output_path=output_path,
        require_nvidia_smi=False,
        require_rtx_render=False,
        smoke_steps=1,
        env={"PATH": ""},
    )

    persisted = _read_json(output_path)
    assert result["status"] == "blocked"
    assert "python_import_isaacsim_failed" in result["blockers"]
    assert persisted["schema_version"] == ISAAC_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    assert persisted["simulator"] == "isaac"
    assert persisted["secret_values_in_artifact"] is False
    proof_boundary = persisted["proof_boundary"]  # type: ignore[index]
    assert proof_boundary["simulator_execution_proven"] is False
    assert proof_boundary["runtime_preflight_is_not_simulator_proof"] is True


def test_isaac_worker_runtime_preflight_blocks_required_nvidia_smi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    output_path = tmp_path / "preflight.json"

    result = run_isaac_worker_runtime_preflight(
        output_path=output_path,
        require_nvidia_smi=True,
        require_rtx_render=False,
        smoke_steps=1,
        env={"PATH": ""},
    )

    assert result["status"] == "blocked"
    assert "nvidia_smi_unavailable" in result["blockers"]
    assert "python_import_isaacsim_failed" in result["blockers"]


def test_isaac_worker_runtime_preflight_fake_rtx_smoke_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    closed: list[bool] = []
    orchestrator_steps: list[bool] = []
    output_path = tmp_path / "preflight.json"

    class FakeSimulationApp:
        def __init__(self, config):
            self.config = config

        def close(self):
            assert output_path.is_file()
            assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "passed"
            closed.append(True)

    class FakeAnnotator:
        def __init__(self, name: str) -> None:
            self.name = name
            self.attached = []

        def attach(self, render_products):
            self.attached.append(list(render_products))

        def get_data(self):
            if self.name == "rgb":
                return np.ones((64, 64, 4), dtype=np.uint8)
            if self.name == "distance_to_camera":
                return np.ones((64, 64), dtype=np.float32)
            return {
                "data": np.ones((64, 64), dtype=np.uint32),
                "info": {"idToLabels": {"1": {"class": "blueprint_smoke_cube"}}},
            }

    fake_annots = {
        name: FakeAnnotator(name)
        for name in ("rgb", "distance_to_camera", "semantic_segmentation")
    }
    fake_core = types.ModuleType("omni.replicator.core")
    fake_core.create = types.SimpleNamespace(
        cube=lambda **_kwargs: object(),
        light=lambda **_kwargs: object(),
        camera=lambda **_kwargs: object(),
        render_product=lambda _camera, _resolution: "render_product",
    )
    fake_core.AnnotatorRegistry = types.SimpleNamespace(
        get_annotator=lambda name: fake_annots[name]
    )
    fake_core.orchestrator = types.SimpleNamespace(step=lambda: orchestrator_steps.append(True))
    fake_replicator = types.ModuleType("omni.replicator")
    fake_replicator.core = fake_core
    fake_omni = types.ModuleType("omni")
    fake_omni.replicator = fake_replicator
    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = FakeSimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", fake_replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", fake_core)
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_physx_rigid_adapter._bind_isaac_runtime_environment",
        lambda: {
            "ISAAC_PATH": "/isaac-sim",
            "EXP_PATH": "/isaac-sim/apps",
            "CARB_APP_PATH": "/isaac-sim/kit",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_physx_rigid_adapter._observe_isaac_runtime_identity",
        lambda _app: {
            "engine_version": "6.0.1",
            "engine_version_source": "test",
            "observed_package_version": "unavailable",
            "observed_app_version": "6.0.1",
            "observed_build_version": "test",
        },
    )

    result = run_isaac_worker_runtime_preflight(
        output_path=output_path,
        require_nvidia_smi=False,
        require_rtx_render=True,
        smoke_steps=2,
        env={"PATH": ""},
    )

    checks = {check["name"]: check for check in result["checks"]}
    assert result["status"] == "passed"
    assert checks["python_import_isaacsim"]["status"] == "passed"
    assert checks["headless_rtx_context_selection"]["renderer"] == "RayTracedLighting"
    assert checks["rtx_smoke_frame_render"]["status"] == "passed"
    assert checks["rtx_smoke_frame_render"]["width"] == 64
    assert checks["rtx_smoke_frame_render"]["height"] == 64
    assert checks["rtx_smoke_frame_render"]["max_steps"] == 2
    assert checks["rtx_smoke_frame_render"]["steps_executed"] == 1
    assert checks["rtx_smoke_frame_render"]["required_output_kinds"] == [
        "rgb",
        "depth",
        "semantic_segmentation",
    ]
    summaries = checks["rtx_smoke_frame_render"]["output_summaries"]
    assert summaries["depth"]["positive_finite_value_count"] == 4096
    assert summaries["semantic_segmentation"]["metadata_present"] is True
    assert orchestrator_steps == [True]
    assert all(annot.attached == [["render_product"]] for annot in fake_annots.values())
    assert result["proof_boundary"]["requested_sensor_modalities_observed"] is True
    assert result["proof_boundary"]["rtx_depth_output_observed"] is True
    assert result["proof_boundary"]["rtx_semantic_segmentation_observed"] is True
    assert closed == [True]


def test_isaac_worker_runtime_preflight_rejects_unknown_output_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isaac_rtx_output_kinds_invalid:lidar"):
        run_isaac_worker_runtime_preflight(
            output_path=tmp_path / "preflight.json",
            required_output_kinds=["rgb", "lidar"],
            env={"PATH": ""},
        )


def test_isaac_worker_runtime_preflight_cli_writes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    output_path = tmp_path / "preflight.json"

    exit_code = preflight_main(["--output", str(output_path), "--smoke-steps", "1"])

    assert exit_code == 2
    assert _read_json(output_path)["status"] == "blocked"


def test_isaac_worker_runtime_preflight_nvidia_check_branches(monkeypatch) -> None:
    monkeypatch.setattr(
        worker_preflight.shutil, "which", lambda _name, path=None: "/bin/nvidia-smi"
    )
    monkeypatch.setattr(
        worker_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0,
            stdout="L40S, 555, 49140 MiB\n",
            stderr="",
        ),
    )
    check, blockers = worker_preflight._nvidia_smi_check(env={"PATH": "/bin"}, required=True)
    assert check["status"] == "passed"
    assert blockers == []
    assert check["gpu_inventory"][0]["gpu_name"] == "L40S"

    monkeypatch.setattr(
        worker_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="driver missing",
        ),
    )
    check, blockers = worker_preflight._nvidia_smi_check(env={"PATH": "/bin"}, required=True)
    assert check["status"] == "blocked"
    assert blockers == ["nvidia_smi_failed"]


def test_isaac_worker_runtime_preflight_rejects_known_bad_isaac_6_rtx_driver() -> None:
    check, blockers = worker_preflight._isaac_rtx_driver_check(
        {
            "gpu_inventory": [
                {
                    "gpu_name": "NVIDIA L40S",
                    "driver_version": "570.124.06",
                    "driver_version_components": [570, 124, 6],
                    "memory_total": "46068 MiB",
                }
            ]
        },
        required=True,
    )

    assert check["status"] == "blocked"
    assert check["reason"] == "known_unsupported_isaac_sim_6_linux_rtx_driver_range"
    assert check["unsupported_range"]["max_exclusive"] == [570, 158, 1]
    assert blockers == ["isaac_sim_6_rtx_driver_unsupported"]


def test_isaac_worker_runtime_preflight_allows_fixed_r570_for_frame_test() -> None:
    check, blockers = worker_preflight._isaac_rtx_driver_check(
        {
            "gpu_inventory": [
                {
                    "gpu_name": "NVIDIA RTX A6000",
                    "driver_version": "570.158.01",
                    "driver_version_components": [570, 158, 1],
                    "memory_total": "49140 MiB",
                }
            ]
        },
        required=True,
    )

    assert check["status"] == "passed_no_known_blocker"
    assert check["rendered_frame_still_required"] is True
    assert blockers == []
