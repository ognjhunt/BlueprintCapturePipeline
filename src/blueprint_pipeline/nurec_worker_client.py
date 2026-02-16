"""NuRec worker dispatch and output collection helpers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import (
    StageError,
    ensure_dir,
    has_nonempty_file,
    parse_gs_uri,
    relative_scene_path,
    utc_now_iso,
    write_json,
)


@dataclass(frozen=True)
class NurecWorkerConfig:
    timeout_seconds: int = 4 * 60 * 60
    poll_interval_seconds: int = 20
    worker_command: str = ""


class NurecWorkerClient:
    """Thin wrapper around external NuRec GPU worker execution."""

    def __init__(
        self,
        *,
        storage_root: Path,
        bucket: str,
        pipeline_prefix: str,
        config: Optional[NurecWorkerConfig] = None,
    ) -> None:
        self.storage_root = storage_root
        self.bucket = bucket
        self.pipeline_prefix = pipeline_prefix.strip("/")
        self.pipeline_dir = storage_root / self.pipeline_prefix
        self.config = config or NurecWorkerConfig(
            timeout_seconds=int(os.getenv("NUREC_TIMEOUT_SECONDS", "14400") or "14400"),
            poll_interval_seconds=int(os.getenv("NUREC_POLL_SECONDS", "20") or "20"),
            worker_command=(os.getenv("NUREC_WORKER_COMMAND", "") or "").strip(),
        )

    @property
    def nurec_dir(self) -> Path:
        return self.pipeline_dir / "nurec"

    def _marker(self, name: str) -> Path:
        return self.pipeline_dir / name

    def build_job_spec(
        self,
        *,
        descriptor: CaptureDescriptor,
        descriptor_uri: str,
        object_index_uri: str,
    ) -> Dict[str, Any]:
        return {
            "schema_version": "v1",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "descriptor_uri": descriptor_uri,
            "capture": {
                "capture_source": descriptor.capture_source,
                "capture_tier": descriptor.capture_tier,
                "nurec_mode": descriptor.nurec_mode,
                "raw_prefix_uri": descriptor.raw_prefix_uri,
                "frames_index_uri": descriptor.frames_index_uri,
                "arkit_poses_uri": descriptor.arkit_poses_uri,
                "arkit_intrinsics_uri": descriptor.arkit_intrinsics_uri,
                "arkit_depth_prefix_uri": descriptor.arkit_depth_prefix_uri,
                "arkit_confidence_prefix_uri": descriptor.arkit_confidence_prefix_uri,
            },
            "inputs": {
                "object_index_uri": object_index_uri,
            },
            "outputs": {
                "nurec_prefix": f"gs://{self.bucket}/{self.pipeline_prefix}/nurec",
                "complete_marker": f"gs://{self.bucket}/{self.pipeline_prefix}/.nurec_complete",
                "failed_marker": f"gs://{self.bucket}/{self.pipeline_prefix}/.nurec_failed",
            },
            "generated_at": utc_now_iso(),
        }

    def write_job_spec(
        self,
        *,
        descriptor: CaptureDescriptor,
        descriptor_uri: str,
        object_index_uri: str,
    ) -> Path:
        ensure_dir(self.pipeline_dir)
        payload = self.build_job_spec(
            descriptor=descriptor,
            descriptor_uri=descriptor_uri,
            object_index_uri=object_index_uri,
        )
        spec_path = self.pipeline_dir / "nurec_job_spec.json"
        write_json(spec_path, payload)
        return spec_path

    def dispatch(self, *, spec_path: Path) -> None:
        command = self.config.worker_command.strip()
        if not command:
            # External scheduler can pick up nurec_job_spec.json asynchronously.
            return

        command = (
            command.replace("{JOB_SPEC_PATH}", str(spec_path))
            .replace("{PIPELINE_DIR}", str(self.pipeline_dir))
            .replace("{PIPELINE_PREFIX}", self.pipeline_prefix)
            .replace("{BUCKET}", self.bucket)
        )

        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(self.storage_root),
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            error = {
                "schema_version": "v1",
                "scene_prefix": self.pipeline_prefix,
                "status": "failed",
                "stage": "dispatch",
                "command": command,
                "return_code": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "failed_at": utc_now_iso(),
            }
            write_json(self._marker(".nurec_failed"), error)
            raise StageError("nurec", f"worker dispatch failed with code {proc.returncode}")

    def wait_for_completion(self) -> None:
        deadline = time.time() + max(60, self.config.timeout_seconds)
        while time.time() < deadline:
            complete_path = self._marker(".nurec_complete")
            failed_path = self._marker(".nurec_failed")
            if failed_path.is_file():
                details = failed_path.read_text(encoding="utf-8", errors="replace")
                raise StageError("nurec", f"worker failure marker detected: {details[:500]}")
            if complete_path.is_file():
                return
            time.sleep(max(2, self.config.poll_interval_seconds))

        raise StageError("nurec", "timeout waiting for .nurec_complete marker")

    def _mesh_stats(self, mesh_path: Path) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "size_bytes": mesh_path.stat().st_size,
        }
        try:
            import trimesh  # type: ignore

            mesh = trimesh.load_mesh(str(mesh_path))
            if getattr(mesh, "vertices", None) is not None:
                stats["vertex_count"] = int(len(mesh.vertices))
            if getattr(mesh, "faces", None) is not None:
                stats["face_count"] = int(len(mesh.faces))
            bounds = getattr(mesh, "bounds", None)
            if bounds is not None and len(bounds) == 2:
                mins = [float(v) for v in bounds[0]]
                maxs = [float(v) for v in bounds[1]]
                stats["bounds"] = {"min": mins, "max": maxs}
        except Exception:
            # Stats are best-effort; mesh existence is the hard requirement.
            pass
        return stats

    def collect_outputs(self) -> Dict[str, Any]:
        export_usdz = self.nurec_dir / "export_last.usdz"
        mesh_ply = self.nurec_dir / "nvblox_mesh.ply"
        occupancy = sorted(self.nurec_dir.glob("occupancy*"))

        if not has_nonempty_file(export_usdz):
            raise StageError("nurec", f"missing required artifact: {export_usdz}")
        if not has_nonempty_file(mesh_ply):
            raise StageError("nurec", f"missing required artifact: {mesh_ply}")
        if not occupancy:
            raise StageError("nurec", "missing occupancy.* artifact")

        payload = {
            "schema_version": "v1",
            "scene_prefix": self.pipeline_prefix,
            "status": "completed",
            "generated_at": utc_now_iso(),
            "artifacts": {
                "visual_usdz": f"gs://{self.bucket}/{relative_scene_path(export_usdz, self.storage_root)}",
                "collision_mesh_ply": f"gs://{self.bucket}/{relative_scene_path(mesh_ply, self.storage_root)}",
                "occupancy": [
                    f"gs://{self.bucket}/{relative_scene_path(path, self.storage_root)}" for path in occupancy
                ],
            },
            "mesh_stats": self._mesh_stats(mesh_ply),
        }
        write_json(self.pipeline_dir / "nurec_outputs.json", payload)
        return payload

    def run(
        self,
        *,
        descriptor: CaptureDescriptor,
        descriptor_uri: str,
        object_index_uri: str,
    ) -> Dict[str, Any]:
        spec_path = self.write_job_spec(
            descriptor=descriptor,
            descriptor_uri=descriptor_uri,
            object_index_uri=object_index_uri,
        )
        self.dispatch(spec_path=spec_path)
        self.wait_for_completion()
        return self.collect_outputs()


def infer_bucket_from_descriptor_uri(descriptor_uri: str) -> str:
    parsed = parse_gs_uri(descriptor_uri)
    return parsed.bucket
