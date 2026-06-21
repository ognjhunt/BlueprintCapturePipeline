from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from blueprint_pipeline import mujoco_worker_runtime_preflight as worker_preflight
from blueprint_pipeline.mujoco_worker_runtime_preflight import (
    MUJOCO_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    WORKER_PREFLIGHT_DETAIL_OUTPUT_ENV,
    main as preflight_main,
    run_mujoco_worker_runtime_preflight,
)


pytest.importorskip("mujoco")
import mujoco  # type: ignore[import-not-found]  # noqa: E402


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_mujoco_worker_runtime_preflight_passes_local_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "preflight.json"

    result = run_mujoco_worker_runtime_preflight(
        output_path=output_path,
        require_nvidia_smi=False,
        require_egl_render=False,
        smoke_steps=2,
        env={"PATH": ""},
    )

    persisted = _read_json(output_path)
    assert result["status"] == "passed"
    assert persisted["schema_version"] == MUJOCO_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    assert persisted["proof_boundary"]["runtime_preflight_is_not_simulator_proof"] is True  # type: ignore[index]
    assert persisted["proof_boundary"]["simulator_execution_proven"] is False  # type: ignore[index]
    check_names = {check["name"] for check in persisted["checks"]}  # type: ignore[index]
    assert "python_import_mujoco" in check_names
    assert "blank_model_or_scene_load" in check_names
    assert "short_rollout_smoke" in check_names


def test_mujoco_worker_runtime_preflight_blocks_missing_required_nvidia_smi(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "preflight.json"

    result = run_mujoco_worker_runtime_preflight(
        output_path=output_path,
        require_nvidia_smi=True,
        require_egl_render=False,
        smoke_steps=1,
        env={"PATH": ""},
    )

    assert result["status"] == "blocked"
    assert "nvidia_smi_unavailable" in result["blockers"]
    persisted = _read_json(output_path)
    assert persisted["secret_values_in_artifact"] is False


def test_mujoco_worker_runtime_preflight_cli_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "preflight.json"

    exit_code = preflight_main(
        [
            "--output",
            str(output_path),
            "--smoke-steps",
            "1",
        ]
    )

    assert exit_code == 0
    assert _read_json(output_path)["status"] == "passed"


def test_mujoco_worker_runtime_preflight_covers_runtime_failure_branches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker_preflight.shutil, "which", lambda _name, path=None: "/bin/nvidia-smi")
    monkeypatch.setattr(
        worker_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="L40S, 555, 49140 MiB\n", stderr=""),
    )
    check, blockers = worker_preflight._nvidia_smi_check(env={"PATH": "/bin"}, required=True)
    assert check["status"] == "passed"
    assert blockers == []

    monkeypatch.setattr(
        worker_preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="driver missing"),
    )
    check, blockers = worker_preflight._nvidia_smi_check(env={"PATH": "/bin"}, required=True)
    assert check["status"] == "blocked"
    assert blockers == ["nvidia_smi_failed"]

    original_from_xml = mujoco.MjModel.from_xml_string
    monkeypatch.setattr(
        mujoco.MjModel,
        "from_xml_string",
        staticmethod(lambda _xml: (_ for _ in ()).throw(RuntimeError("bad model"))),
    )
    checks, blockers = worker_preflight._mujoco_smoke_checks(
        smoke_steps=1,
        require_egl_render=True,
        env={},
    )
    assert "blank_model_or_scene_load_failed" in blockers
    assert "egl_context_when_rendering_not_attempted" in blockers
    assert any(check["name"] == "egl_context_when_rendering" for check in checks)

    monkeypatch.setattr(mujoco.MjModel, "from_xml_string", original_from_xml)
    monkeypatch.setattr(
        mujoco,
        "mj_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("step failed")),
    )
    checks, blockers = worker_preflight._mujoco_smoke_checks(
        smoke_steps=1,
        require_egl_render=False,
        env={},
    )
    assert "short_rollout_smoke_failed" in blockers
    assert any(check["name"] == "short_rollout_smoke" and check["status"] == "blocked" for check in checks)


def test_mujoco_worker_runtime_preflight_egl_render_and_default_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeRenderer:
        def __init__(self, _model, *, height: int, width: int) -> None:
            self.height = height
            self.width = width

        def update_scene(self, _data) -> None:
            return None

        def render(self):
            import numpy as np

            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    checks, blockers = worker_preflight._mujoco_smoke_checks(
        smoke_steps=0,
        require_egl_render=True,
        env={"MUJOCO_GL": "egl"},
    )
    assert blockers == []
    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert any(check["name"] == "egl_context_when_rendering" and check["status"] == "passed" for check in checks)

    class FailingRenderer(FakeRenderer):
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("no egl")

    monkeypatch.setattr(mujoco, "Renderer", FailingRenderer)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    checks, blockers = worker_preflight._mujoco_smoke_checks(
        smoke_steps=1,
        require_egl_render=True,
        env={},
    )
    assert "egl_context_when_rendering_failed" in blockers
    assert "MUJOCO_GL" not in os.environ

    output_path = tmp_path / "env-output.json"
    monkeypatch.setenv(WORKER_PREFLIGHT_DETAIL_OUTPUT_ENV, str(output_path))
    assert preflight_main(["--smoke-steps", "1"]) in {0, 2}
    assert output_path.is_file()
