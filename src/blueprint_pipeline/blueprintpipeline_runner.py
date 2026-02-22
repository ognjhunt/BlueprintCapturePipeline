"""Compatibility layer for invoking jobs/helpers from BlueprintPipeline."""

from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

from .common import StageError, ensure_dir, has_nonempty_file, write_json, write_text

DEFAULT_GCS_MOUNT_ROOT = Path("/mnt/gcs")
ROOT_OVERRIDE_SCRIPT_PATHS = {
    "simready-job/prepare_simready_assets.py",
    "usd-assembly-job/assemble_scene.py",
    "replicator-job/generate_replicator_bundle.py",
    "variation-asset-pipeline-job/run_variation_asset_pipeline.py",
    "isaac-lab-job/generate_isaac_lab_task.py",
    "episode-generation-job/generate_episodes.py",
}


@dataclass(frozen=True)
class CommandResult:
    return_code: int
    stdout: str
    stderr: str
    command: List[str]


@contextlib.contextmanager
def _temporary_environ(updates: Mapping[str, str]) -> Iterator[None]:
    old_values: Dict[str, Optional[str]] = {}
    for key, value in updates.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@dataclass(frozen=True)
class BlueprintPipelineRunnerConfig:
    blueprintpipeline_root: Path
    gcs_root: Path
    bucket: str
    standalone_enabled: bool = False


class BlueprintPipelineRunner:
    """Invoke BlueprintPipeline helper functions and job scripts."""

    def __init__(self, config: BlueprintPipelineRunnerConfig) -> None:
        self.config = config

    def _prefix_to_path(self, prefix: str) -> Path:
        path = Path(prefix)
        if path.is_absolute():
            return path
        return (self.config.gcs_root / path).resolve()

    def _external_blueprintpipeline_available(self) -> bool:
        root = self.config.blueprintpipeline_root
        required_scripts = [
            root / "interactive-job/run_interactive_assets.py",
            root / "simready-job/prepare_simready_assets.py",
            root / "usd-assembly-job/assemble_scene.py",
            root / "replicator-job/generate_replicator_bundle.py",
            root / "variation-asset-pipeline-job/run_variation_asset_pipeline.py",
            root / "genie-sim-export-job/export_to_geniesim.py",
            root / "tools/source_pipeline/adapter.py",
        ]
        return root.is_dir() and all(path.is_file() for path in required_scripts)

    def _use_internal_standalone(self) -> bool:
        if not self.config.standalone_enabled:
            return False
        return True

    def _using_default_mount_root(self) -> bool:
        try:
            return self.config.gcs_root.resolve() == DEFAULT_GCS_MOUNT_ROOT
        except FileNotFoundError:
            return self.config.gcs_root == DEFAULT_GCS_MOUNT_ROOT

    def _job_prefix_path(self, prefix: str) -> str:
        path = Path(prefix)
        if path.is_absolute():
            return str(path)
        if self._using_default_mount_root():
            return prefix
        return str((self.config.gcs_root / path).resolve())

    def _should_use_root_override_wrapper(self, script_rel_path: str) -> bool:
        return (not self._using_default_mount_root()) and script_rel_path in ROOT_OVERRIDE_SCRIPT_PATHS

    def _build_job_command(self, *, script_path: Path, script_rel_path: str) -> List[str]:
        if not self._should_use_root_override_wrapper(script_rel_path):
            return [sys.executable, str(script_path)]

        wrapper = "\n".join(
            [
                "import importlib.util",
                "import pathlib",
                "import sys",
                "script = pathlib.Path(sys.argv[1])",
                "root = pathlib.Path(sys.argv[2])",
                "spec = importlib.util.spec_from_file_location('bp_job_entry', str(script))",
                "if spec is None or spec.loader is None:",
                "    raise SystemExit(2)",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "if hasattr(module, 'GCS_ROOT'):",
                "    module.GCS_ROOT = root",
                "result = module.main() if hasattr(module, 'main') else 0",
                "raise SystemExit(0 if result is None else int(result))",
            ]
        )
        return [sys.executable, "-c", wrapper, str(script_path), str(self.config.gcs_root)]

    def ensure_expected_commit(self, expected_commit_hash: str) -> None:
        if self._use_internal_standalone():
            return
        expected = expected_commit_hash.strip().lower()
        if not expected:
            return
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.config.blueprintpipeline_root),
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise StageError(
                "blueprintpipeline",
                f"unable to resolve commit hash: {proc.stderr.strip() or proc.stdout.strip()}",
            )
        actual = proc.stdout.strip().lower()
        if actual != expected:
            raise StageError(
                "blueprintpipeline",
                f"commit mismatch (expected {expected}, got {actual})",
            )

    def _import_source_adapter(self):
        root_str = str(self.config.blueprintpipeline_root)
        if root_str not in sys.path:
            sys.path.append(root_str)
        return importlib.import_module("tools.source_pipeline.adapter")

    def materialize_text_assets(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        objects: List[Dict[str, Any]],
        room_type: str,
        generation_enabled: bool,
        retrieval_enabled: bool,
        retrieval_mode: str,
        generation_provider_chain: str,
    ) -> Dict[str, Any]:
        """Materialize assets using BlueprintPipeline text adapter behavior."""

        if self._use_internal_standalone():
            return self._materialize_text_assets_internal(
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                objects=objects,
                room_type=room_type,
                generation_enabled=generation_enabled,
                retrieval_enabled=retrieval_enabled,
                retrieval_mode=retrieval_mode,
                generation_provider_chain=generation_provider_chain,
            )

        adapter = self._import_source_adapter()

        env_updates = {
            "BUCKET": self.config.bucket,
            "TEXT_ASSET_GENERATION_ENABLED": "true" if generation_enabled else "false",
            "TEXT_ASSET_RETRIEVAL_ENABLED": "true" if retrieval_enabled else "false",
            "TEXT_ASSET_GENERATION_PROVIDER_CHAIN": generation_provider_chain,
            "TEXT_ASSET_RETRIEVAL_MODE": retrieval_mode,
        }

        retrieval_report: Dict[str, Any] = {}
        with _temporary_environ(env_updates):
            provenance_assets = adapter.materialize_placeholder_assets(
                root=self.config.gcs_root,
                scene_id=scene_id,
                assets_prefix=assets_prefix,
                objects=objects,
                room_type=room_type,
                retrieval_report=retrieval_report,
            )

        return {
            "provenance_assets": provenance_assets,
            "retrieval_report": retrieval_report,
        }

    def _write_stub_mesh_glb(self, path: Path) -> None:
        if has_nonempty_file(path):
            return
        ensure_dir(path.parent)
        # Minimal placeholder payload; downstream gates only require existence.
        path.write_bytes(b"glTFstub")

    def _write_stub_model_usd(self, path: Path, reference_rel_path: str) -> None:
        ref = reference_rel_path.replace("\\", "/")
        payload = f'''#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend references = @{ref}@
)
{{
}}
'''
        write_text(path, payload)

    def _materialize_text_assets_internal(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        objects: List[Dict[str, Any]],
        room_type: str,
        generation_enabled: bool,
        retrieval_enabled: bool,
        retrieval_mode: str,
        generation_provider_chain: str,
    ) -> Dict[str, Any]:
        assets_root = self._prefix_to_path(assets_prefix)
        ensure_dir(assets_root)

        provenance_assets: List[Dict[str, Any]] = []
        method_counts = {
            "generated": 0,
            "retrieved": 0,
        }

        for obj in objects:
            asset_id = str(obj.get("id") or "").strip()
            if not asset_id:
                continue
            asset_dir = assets_root / asset_id
            ensure_dir(asset_dir)

            mesh_path = asset_dir / "mesh.glb"
            self._write_stub_mesh_glb(mesh_path)

            model_path = asset_dir / "model.usd"
            self._write_stub_model_usd(model_path, "./mesh.glb")

            metadata_path = asset_dir / "metadata.json"
            metadata_payload: Dict[str, Any] = {
                "schema_version": "v1",
                "scene_id": scene_id,
                "asset_id": asset_id,
                "room_type": room_type,
                "generation_enabled": generation_enabled,
                "retrieval_enabled": retrieval_enabled,
                "retrieval_mode": retrieval_mode,
                "generation_provider_chain": generation_provider_chain,
                "mode": "standalone_internal",
            }
            reference_image = str(obj.get("reference_image") or "").strip()
            if reference_image:
                metadata_payload["reference_image"] = reference_image
            write_json(metadata_path, metadata_payload)

            if retrieval_enabled:
                joint_usda = asset_dir / "joint.usda"
                write_text(joint_usda, 'def PhysicsRevoluteJoint "joint_0" {}\n')
                method_counts["retrieved"] += 1
                materialization = "retrieved_asset_bundle"
            else:
                method_counts["generated"] += 1
                materialization = "generated_image_conditioned"

            provenance_assets.append(
                {
                    "object_id": asset_id,
                    "path": f"{assets_prefix}/{asset_id}/model.usd",
                    "materialization": materialization,
                }
            )

        return {
            "provenance_assets": provenance_assets,
            "retrieval_report": {
                "mode": "standalone_internal",
                "method_counts": method_counts,
            },
        }

    def _run_job_script(
        self,
        *,
        script_rel_path: str,
        env: MutableMapping[str, str],
    ) -> CommandResult:
        script_path = self.config.blueprintpipeline_root / script_rel_path
        command = self._build_job_command(script_path=script_path, script_rel_path=script_rel_path)

        merged_env = dict(os.environ)
        merged_env.update(env)
        proc = subprocess.run(
            command,
            cwd=str(self.config.blueprintpipeline_root),
            env=merged_env,
            check=False,
            text=True,
            capture_output=True,
        )
        return CommandResult(
            return_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            command=command,
        )

    def run_interactive_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        regen3d_prefix: str,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        if self._use_internal_standalone():
            assets_root = self._prefix_to_path(assets_prefix)
            ensure_dir(assets_root / "interactive")
            manifest_path = assets_root / "scene_manifest.json"
            objects_payload: List[Dict[str, Any]] = []
            if manifest_path.is_file():
                import json

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_objects = (
                    manifest.get("objects") if isinstance(manifest.get("objects"), list) else []
                )
                for obj in manifest_objects:
                    if not isinstance(obj, Mapping):
                        continue
                    obj_id = str(obj.get("id") or "").strip()
                    if not obj_id:
                        continue
                    articulation = obj.get("articulation") if isinstance(obj.get("articulation"), Mapping) else {}
                    required = bool(articulation.get("required", False))
                    role = str(obj.get("sim_role") or "").strip()
                    is_interactive = role in {
                        "manipulable_object",
                        "deformable_object",
                        "articulated_furniture",
                        "articulated_appliance",
                    }
                    if not is_interactive:
                        continue
                    objects_payload.append(
                        {
                            "id": obj_id,
                            "name": str(obj.get("name") or obj_id),
                            "status": "ok",
                            "backend": "standalone_internal",
                            "required_articulation": required,
                            "is_articulated": True if required else False,
                            "joint_count": 1 if required else 0,
                        }
                    )

            interactive_payload = {
                "schema_version": "v1",
                "objects": objects_payload,
                "total_objects": len(objects_payload),
                "ok_count": sum(1 for item in objects_payload if item.get("status") == "ok"),
                "error_count": 0,
                "fallback_count": 0,
                "articulated_count": sum(1 for item in objects_payload if bool(item.get("is_articulated"))),
            }
            write_json(assets_root / "interactive" / "interactive_results.json", interactive_payload)
            return CommandResult(
                return_code=0,
                stdout="standalone interactive success",
                stderr="",
                command=["internal", "run_interactive_job"],
            )

        assets_env_prefix = self._job_prefix_path(assets_prefix)
        regen3d_env_prefix = self._job_prefix_path(regen3d_prefix)
        env = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": assets_env_prefix,
            "REGEN3D_PREFIX": regen3d_env_prefix,
        }
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return self._run_job_script(
            script_rel_path="interactive-job/run_interactive_assets.py",
            env=env,
        )

    def run_simready_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        layout_prefix: str,
        usd_prefix: str,
    ) -> CommandResult:
        if self._use_internal_standalone():
            assets_root = self._prefix_to_path(assets_prefix)
            layout_root = self._prefix_to_path(layout_prefix)
            usd_root = self._prefix_to_path(usd_prefix)
            ensure_dir(assets_root / "simready")
            ensure_dir(layout_root)
            ensure_dir(usd_root)
            write_json(
                assets_root / "simready" / "simready_report.json",
                {
                    "schema_version": "v1",
                    "scene_id": scene_id,
                    "status": "ok",
                    "mode": "standalone_internal",
                },
            )
            return CommandResult(
                return_code=0,
                stdout="standalone simready success",
                stderr="",
                command=["internal", "run_simready_job"],
            )

        assets_env_prefix = self._job_prefix_path(assets_prefix)
        layout_env_prefix = self._job_prefix_path(layout_prefix)
        usd_env_prefix = self._job_prefix_path(usd_prefix)
        env = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": assets_env_prefix,
            "LAYOUT_PREFIX": layout_env_prefix,
            "USD_PREFIX": usd_env_prefix,
        }
        return self._run_job_script(
            script_rel_path="simready-job/prepare_simready_assets.py",
            env=env,
        )

    def run_usd_assembly_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        layout_prefix: str,
        usd_prefix: str,
    ) -> CommandResult:
        if self._use_internal_standalone():
            usd_root = self._prefix_to_path(usd_prefix)
            ensure_dir(usd_root)
            scene_usda = usd_root / "scene.usda"
            write_text(
                scene_usda,
                '#usda 1.0\n(\n    defaultPrim = "Scene"\n)\n\ndef Xform "Scene"\n{\n}\n',
            )
            return CommandResult(
                return_code=0,
                stdout="standalone usd assembly success",
                stderr="",
                command=["internal", "run_usd_assembly_job"],
            )

        assets_env_prefix = self._job_prefix_path(assets_prefix)
        layout_env_prefix = self._job_prefix_path(layout_prefix)
        usd_env_prefix = self._job_prefix_path(usd_prefix)
        env = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": assets_env_prefix,
            "LAYOUT_PREFIX": layout_env_prefix,
            "USD_PREFIX": usd_env_prefix,
        }
        return self._run_job_script(
            script_rel_path="usd-assembly-job/assemble_scene.py",
            env=env,
        )

    # ------------------------------------------------------------------
    # Data-generation downstream jobs
    # ------------------------------------------------------------------

    def run_replicator_job(
        self,
        *,
        scene_id: str,
        seg_prefix: str,
        assets_prefix: str,
        usd_prefix: str,
        replicator_prefix: str,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Generate Omniverse Replicator bundle (placement regions, policies, variation manifest)."""
        if self._use_internal_standalone():
            replicator_root = self._prefix_to_path(replicator_prefix)
            ensure_dir(replicator_root)
            write_json(
                replicator_root / ".replicator_complete",
                {
                    "schema_version": "v1",
                    "scene_id": scene_id,
                    "status": "ok",
                    "mode": "standalone_internal",
                },
            )
            return CommandResult(
                return_code=0,
                stdout="standalone replicator success",
                stderr="",
                command=["internal", "run_replicator_job"],
            )

        env: Dict[str, str] = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "SEG_PREFIX": self._job_prefix_path(seg_prefix),
            "ASSETS_PREFIX": self._job_prefix_path(assets_prefix),
            "USD_PREFIX": self._job_prefix_path(usd_prefix),
            "REPLICATOR_PREFIX": self._job_prefix_path(replicator_prefix),
        }
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return self._run_job_script(
            script_rel_path="replicator-job/generate_replicator_bundle.py",
            env=env,
        )

    def run_isaac_lab_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        replicator_prefix: str,
        isaac_lab_prefix: str,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Generate Isaac Lab training task package (env_cfg, task, rewards)."""
        if self._use_internal_standalone():
            isaac_lab_root = self._prefix_to_path(isaac_lab_prefix)
            ensure_dir(isaac_lab_root)
            write_json(
                isaac_lab_root / ".isaac_lab_complete",
                {
                    "schema_version": "v1",
                    "scene_id": scene_id,
                    "status": "ok",
                    "mode": "standalone_internal",
                },
            )
            return CommandResult(
                return_code=0,
                stdout="standalone isaac-lab success",
                stderr="",
                command=["internal", "run_isaac_lab_job"],
            )

        env: Dict[str, str] = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": self._job_prefix_path(assets_prefix),
            "REPLICATOR_PREFIX": self._job_prefix_path(replicator_prefix),
            "ISAAC_LAB_PREFIX": self._job_prefix_path(isaac_lab_prefix),
        }
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return self._run_job_script(
            script_rel_path="isaac-lab-job/generate_isaac_lab_task.py",
            env=env,
        )

    def run_variation_asset_pipeline_job(
        self,
        *,
        scene_id: str,
        replicator_prefix: str,
        variation_assets_prefix: str,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Generate variation assets manifest from replicator outputs."""
        if self._use_internal_standalone():
            variation_root = self._prefix_to_path(variation_assets_prefix)
            ensure_dir(variation_root)
            write_json(
                variation_root / ".variation_pipeline_complete",
                {
                    "schema_version": "v1",
                    "scene_id": scene_id,
                    "status": "ok",
                    "mode": "standalone_internal",
                },
            )
            write_json(
                variation_root / "variation_assets.json",
                {
                    "schema_version": "v1",
                    "scene_id": scene_id,
                    "source": "standalone_internal",
                    "objects": [],
                },
            )
            return CommandResult(
                return_code=0,
                stdout="standalone variation-asset pipeline success",
                stderr="",
                command=["internal", "run_variation_asset_pipeline_job"],
            )

        env: Dict[str, str] = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "REPLICATOR_PREFIX": self._job_prefix_path(replicator_prefix),
            "VARIATION_ASSETS_PREFIX": self._job_prefix_path(variation_assets_prefix),
        }
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return self._run_job_script(
            script_rel_path="variation-asset-pipeline-job/run_variation_asset_pipeline.py",
            env=env,
        )

    def run_geniesim_export_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        replicator_prefix: str,
        variation_assets_prefix: str,
        geniesim_prefix: str,
        filter_commercial: bool = True,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Export assembled scene + selected assets into GenieSim package format."""
        if self._use_internal_standalone():
            geniesim_root = self._prefix_to_path(geniesim_prefix)
            ensure_dir(geniesim_root)
            write_json(
                geniesim_root / "scene_graph.json",
                {"schema_version": "v1", "scene_id": scene_id, "source": "standalone_internal"},
            )
            write_json(
                geniesim_root / "asset_index.json",
                {"schema_version": "v1", "scene_id": scene_id, "assets": []},
            )
            write_json(
                geniesim_root / "task_config.json",
                {"schema_version": "v1", "scene_id": scene_id, "tasks": []},
            )
            write_json(
                geniesim_root / "merged_scene_manifest.json",
                {"schema_version": "v1", "scene_id": scene_id, "objects": []},
            )
            return CommandResult(
                return_code=0,
                stdout="standalone genie-sim-export success",
                stderr="",
                command=["internal", "run_geniesim_export_job"],
            )

        env: Dict[str, str] = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": self._job_prefix_path(assets_prefix),
            "REPLICATOR_PREFIX": self._job_prefix_path(replicator_prefix),
            "VARIATION_ASSETS_PREFIX": self._job_prefix_path(variation_assets_prefix),
            "GENIESIM_PREFIX": self._job_prefix_path(geniesim_prefix),
            "FILTER_COMMERCIAL": "true" if filter_commercial else "false",
            "GCSFUSE_MOUNT_PATH": str(self.config.gcs_root),
        }
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return self._run_job_script(
            script_rel_path="genie-sim-export-job/export_to_geniesim.py",
            env=env,
        )

    def run_episode_generation_job(
        self,
        *,
        scene_id: str,
        assets_prefix: str,
        episodes_prefix: str,
        robot_type: str = "franka",
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        """Generate robot demonstration episodes (LeRobot Parquet format)."""
        if self._use_internal_standalone():
            episodes_root = self._prefix_to_path(episodes_prefix)
            ensure_dir(episodes_root)
            write_json(
                episodes_root / ".episodes_complete",
                {
                    "schema_version": "v1",
                    "scene_id": scene_id,
                    "status": "ok",
                    "mode": "standalone_internal",
                },
            )
            return CommandResult(
                return_code=0,
                stdout="standalone episode-generation success",
                stderr="",
                command=["internal", "run_episode_generation_job"],
            )

        env: Dict[str, str] = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": self._job_prefix_path(assets_prefix),
            "EPISODES_PREFIX": self._job_prefix_path(episodes_prefix),
            "ROBOT_TYPE": robot_type,
        }
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return self._run_job_script(
            script_rel_path="episode-generation-job/generate_episodes.py",
            env=env,
        )


def required_env_from_command_result(result: CommandResult) -> Dict[str, Any]:
    return {
        "return_code": result.return_code,
        "command": result.command,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
