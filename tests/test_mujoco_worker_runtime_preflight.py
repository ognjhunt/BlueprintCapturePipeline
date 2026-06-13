from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.mujoco_worker_runtime_preflight import (
    MUJOCO_WORKER_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    main as preflight_main,
    run_mujoco_worker_runtime_preflight,
)


pytest.importorskip("mujoco")


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
