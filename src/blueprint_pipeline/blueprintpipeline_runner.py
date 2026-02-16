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

from .common import StageError


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


class BlueprintPipelineRunner:
    """Invoke BlueprintPipeline helper functions and job scripts."""

    def __init__(self, config: BlueprintPipelineRunnerConfig) -> None:
        self.config = config

    def ensure_expected_commit(self, expected_commit_hash: str) -> None:
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

    def _run_job_script(
        self,
        *,
        script_rel_path: str,
        env: MutableMapping[str, str],
    ) -> CommandResult:
        script_path = self.config.blueprintpipeline_root / script_rel_path
        command = [sys.executable, str(script_path)]

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
        env = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": assets_prefix,
            "REGEN3D_PREFIX": regen3d_prefix,
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
        env = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": assets_prefix,
            "LAYOUT_PREFIX": layout_prefix,
            "USD_PREFIX": usd_prefix,
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
        env = {
            "BUCKET": self.config.bucket,
            "SCENE_ID": scene_id,
            "ASSETS_PREFIX": assets_prefix,
            "LAYOUT_PREFIX": layout_prefix,
            "USD_PREFIX": usd_prefix,
        }
        return self._run_job_script(
            script_rel_path="usd-assembly-job/assemble_scene.py",
            env=env,
        )


def required_env_from_command_result(result: CommandResult) -> Dict[str, Any]:
    return {
        "return_code": result.return_code,
        "command": result.command,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
