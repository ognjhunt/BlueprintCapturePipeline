from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

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

    class FakeSimulationApp:
        def __init__(self, config):
            self.config = config

        def close(self):
            closed.append(True)

    class FakeAnnotator:
        def __init__(self) -> None:
            self.attached = []

        def attach(self, render_products):
            self.attached.append(list(render_products))

        def get_data(self):
            return np.ones((16, 16, 3), dtype=np.uint8)

    fake_annot = FakeAnnotator()
    fake_core = types.ModuleType("omni.replicator.core")
    fake_core.create = types.SimpleNamespace(
        camera=lambda **_kwargs: object(),
        render_product=lambda _camera, _resolution: "render_product",
    )
    fake_core.AnnotatorRegistry = types.SimpleNamespace(
        get_annotator=lambda _name: fake_annot
    )
    fake_core.orchestrator = types.SimpleNamespace(step=lambda: None)
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

    result = run_isaac_worker_runtime_preflight(
        output_path=tmp_path / "preflight.json",
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
    assert fake_annot.attached == [["render_product"]]
    assert closed == [True]


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
    monkeypatch.setattr(worker_preflight.shutil, "which", lambda _name, path=None: "/bin/nvidia-smi")
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
