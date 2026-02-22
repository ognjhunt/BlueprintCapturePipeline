"""Tests for BlueprintPipeline job env wiring."""

from __future__ import annotations

import json
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


def test_standalone_interactive_job_writes_results(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    assets_root = gcs_root / "scenes/scene_1/assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "objects": [
            {
                "id": "drawer_1",
                "name": "drawer",
                "sim_role": "articulated_furniture",
                "articulation": {"required": True},
            }
        ]
    }
    (assets_root / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    runner = BlueprintPipelineRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path / "missing_root",
            gcs_root=gcs_root,
            bucket="bucket",
            standalone_enabled=True,
        )
    )

    result = runner.run_interactive_job(
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        regen3d_prefix="scenes/scene_1/assets",
    )
    assert result.return_code == 0

    output_path = assets_root / "interactive/interactive_results.json"
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["total_objects"] == 1
    assert payload["objects"][0]["id"] == "drawer_1"
    assert payload["objects"][0]["is_articulated"] is True


def test_standalone_materialize_retrieval_writes_joint_asset(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    runner = BlueprintPipelineRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path / "missing_root",
            gcs_root=gcs_root,
            bucket="bucket",
            standalone_enabled=True,
        )
    )

    payload = runner.materialize_text_assets(
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        objects=[{"id": "obj_drawer"}],
        room_type="warehouse",
        generation_enabled=False,
        retrieval_enabled=True,
        retrieval_mode="ann_primary",
        generation_provider_chain="image_to_3d,proxy_box",
    )
    assert payload["provenance_assets"]
    asset_root = gcs_root / "scenes/scene_1/assets/obj_drawer"
    assert (asset_root / "model.usd").is_file()
    assert (asset_root / "mesh.glb").is_file()
    assert (asset_root / "joint.usda").is_file()


def test_standalone_usd_assembly_writes_scene_file(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    runner = BlueprintPipelineRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path / "missing_root",
            gcs_root=gcs_root,
            bucket="bucket",
            standalone_enabled=True,
        )
    )

    result = runner.run_usd_assembly_job(
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        layout_prefix="scenes/scene_1/layout",
        usd_prefix="scenes/scene_1/usd",
    )
    assert result.return_code == 0
    assert (gcs_root / "scenes/scene_1/usd/scene.usda").is_file()


def test_non_mnt_root_uses_wrapper_for_variation_job(tmp_path: Path) -> None:
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
        script_path=tmp_path / "variation-asset-pipeline-job/run_variation_asset_pipeline.py",
        script_rel_path="variation-asset-pipeline-job/run_variation_asset_pipeline.py",
    )
    assert len(command) >= 5
    assert command[1] == "-c"
    assert command[-1] == str(gcs_root)


def test_runner_wires_variation_and_geniesim_env(tmp_path: Path) -> None:
    gcs_root = tmp_path / "mock_gcs" / "bucket"
    gcs_root.mkdir(parents=True, exist_ok=True)
    runner = _CaptureRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path,
            gcs_root=gcs_root,
            bucket="bucket",
        )
    )

    variation_result = runner.run_variation_asset_pipeline_job(
        scene_id="scene_3",
        replicator_prefix="scenes/scene_3/replicator",
        variation_assets_prefix="scenes/scene_3/variation_assets",
    )
    assert variation_result.return_code == 0
    assert runner.calls[-1]["script_rel_path"] == "variation-asset-pipeline-job/run_variation_asset_pipeline.py"
    variation_env = _last_env(runner)
    assert variation_env["BUCKET"] == "bucket"
    assert variation_env["SCENE_ID"] == "scene_3"
    assert variation_env["REPLICATOR_PREFIX"] == str((gcs_root / "scenes/scene_3/replicator").resolve())
    assert variation_env["VARIATION_ASSETS_PREFIX"] == str(
        (gcs_root / "scenes/scene_3/variation_assets").resolve()
    )

    geniesim_result = runner.run_geniesim_export_job(
        scene_id="scene_3",
        assets_prefix="scenes/scene_3/assets",
        replicator_prefix="scenes/scene_3/replicator",
        variation_assets_prefix="scenes/scene_3/variation_assets",
        geniesim_prefix="scenes/scene_3/geniesim",
        filter_commercial=True,
    )
    assert geniesim_result.return_code == 0
    assert runner.calls[-1]["script_rel_path"] == "genie-sim-export-job/export_to_geniesim.py"
    geniesim_env = _last_env(runner)
    assert geniesim_env["BUCKET"] == "bucket"
    assert geniesim_env["SCENE_ID"] == "scene_3"
    assert geniesim_env["ASSETS_PREFIX"] == str((gcs_root / "scenes/scene_3/assets").resolve())
    assert geniesim_env["REPLICATOR_PREFIX"] == str((gcs_root / "scenes/scene_3/replicator").resolve())
    assert geniesim_env["VARIATION_ASSETS_PREFIX"] == str(
        (gcs_root / "scenes/scene_3/variation_assets").resolve()
    )
    assert geniesim_env["GENIESIM_PREFIX"] == str((gcs_root / "scenes/scene_3/geniesim").resolve())
    assert geniesim_env["FILTER_COMMERCIAL"] == "true"
    assert geniesim_env["GCSFUSE_MOUNT_PATH"] == str(gcs_root)


def test_standalone_data_gen_jobs_write_expected_outputs(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    runner = BlueprintPipelineRunner(
        BlueprintPipelineRunnerConfig(
            blueprintpipeline_root=tmp_path / "missing_root",
            gcs_root=gcs_root,
            bucket="bucket",
            standalone_enabled=True,
        )
    )

    replicator_result = runner.run_replicator_job(
        scene_id="scene_1",
        seg_prefix="scenes/scene_1/seg",
        assets_prefix="scenes/scene_1/assets",
        usd_prefix="scenes/scene_1/usd",
        replicator_prefix="scenes/scene_1/replicator",
    )
    assert replicator_result.return_code == 0
    assert (gcs_root / "scenes/scene_1/replicator/.replicator_complete").is_file()

    variation_result = runner.run_variation_asset_pipeline_job(
        scene_id="scene_1",
        replicator_prefix="scenes/scene_1/replicator",
        variation_assets_prefix="scenes/scene_1/variation_assets",
    )
    assert variation_result.return_code == 0
    assert (gcs_root / "scenes/scene_1/variation_assets/.variation_pipeline_complete").is_file()
    assert (gcs_root / "scenes/scene_1/variation_assets/variation_assets.json").is_file()

    geniesim_result = runner.run_geniesim_export_job(
        scene_id="scene_1",
        assets_prefix="scenes/scene_1/assets",
        replicator_prefix="scenes/scene_1/replicator",
        variation_assets_prefix="scenes/scene_1/variation_assets",
        geniesim_prefix="scenes/scene_1/geniesim",
    )
    assert geniesim_result.return_code == 0
    geniesim_root = gcs_root / "scenes/scene_1/geniesim"
    assert (geniesim_root / "scene_graph.json").is_file()
    assert (geniesim_root / "asset_index.json").is_file()
    assert (geniesim_root / "task_config.json").is_file()
    assert (geniesim_root / "merged_scene_manifest.json").is_file()
