"""Tests for BlueprintPipeline job env wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping

from blueprint_pipeline.blueprintpipeline_runner import (
    BlueprintPipelineRunner,
    BlueprintPipelineRunnerConfig,
    CommandResult,
)


class _CaptureRunner(BlueprintPipelineRunner):
    def __init__(self, config: BlueprintPipelineRunnerConfig) -> None:
        super().__init__(config)
        self.calls: List[Dict[str, object]] = []

    def _run_job_script(
        self,
        *,
        script_rel_path: str,
        env: MutableMapping[str, str],
    ) -> CommandResult:
        self.calls.append({"script_rel_path": script_rel_path, "env": dict(env)})
        return CommandResult(return_code=0, stdout="", stderr="", command=["python", script_rel_path])


def _last_env(runner: _CaptureRunner) -> Mapping[str, str]:
    assert runner.calls
    env = runner.calls[-1]["env"]
    assert isinstance(env, dict)
    return env


def test_runner_keeps_relative_prefixes_for_mnt_gcs() -> None:
    runner = _CaptureRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=Path("/tmp/blueprintpipeline"),
            gcs_root=Path("/mnt/gcs"),
            bucket="bucket",
        )
    )

    runner.run_simready_job(
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        layout_prefix="scenes/scene_1/layout",
        usd_prefix="scenes/scene_1/usd",
    )
    env = _last_env(runner)
    assert env["ASSETS_PREFIX"] == "scenes/scene_1/assets"
    assert env["LAYOUT_PREFIX"] == "scenes/scene_1/layout"
    assert env["USD_PREFIX"] == "scenes/scene_1/usd"


def test_runner_uses_absolute_prefixes_for_non_mnt_root(tmp_path: Path) -> None:
    gcs_root = tmp_path / "mock_gcs" / "bucket"
    gcs_root.mkdir(parents=True, exist_ok=True)
    runner = _CaptureRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path,
            gcs_root=gcs_root,
            bucket="bucket",
        )
    )

    runner.run_interactive_job(
        scene_id="scene_2",
        assets_prefix="scenes/scene_2/assets",
        regen3d_prefix="scenes/scene_2/assets",
    )
    env = _last_env(runner)
    assert env["ASSETS_PREFIX"] == str((gcs_root / "scenes/scene_2/assets").resolve())
    assert env["REGEN3D_PREFIX"] == str((gcs_root / "scenes/scene_2/assets").resolve())


def test_non_mnt_root_uses_wrapper_for_usd_related_jobs(tmp_path: Path) -> None:
    gcs_root = tmp_path / "mock_gcs" / "bucket"
    gcs_root.mkdir(parents=True, exist_ok=True)
    runner = BlueprintPipelineRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path,
            gcs_root=gcs_root,
            bucket="bucket",
        )
    )

    command = runner._build_job_command(
        script_path=tmp_path / "usd-assembly-job/assemble_scene.py",
        script_rel_path="usd-assembly-job/assemble_scene.py",
    )
    assert len(command) >= 5
    assert command[1] == "-c"
    assert command[-1] == str(gcs_root)


def test_default_mount_root_does_not_use_wrapper() -> None:
    runner = BlueprintPipelineRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=Path("/tmp/blueprintpipeline"),
            gcs_root=Path("/mnt/gcs"),
            bucket="bucket",
        )
    )

    command = runner._build_job_command(
        script_path=Path("/tmp/blueprintpipeline/usd-assembly-job/assemble_scene.py"),
        script_rel_path="usd-assembly-job/assemble_scene.py",
    )
    assert command == [
        sys.executable,
        "/tmp/blueprintpipeline/usd-assembly-job/assemble_scene.py",
    ]
