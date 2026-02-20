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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class NurecWorkerConfig:
    timeout_seconds: int = 4 * 60 * 60
    poll_interval_seconds: int = 20
    worker_mode: str = "local_worker"
    worker_command: str = ""
    worker_python_executable: str = os.getenv("PYTHON_BIN", "python")


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
            worker_mode=(os.getenv("NUREC_WORKER_MODE", "local_worker") or "local_worker").strip(),
            worker_command=(os.getenv("NUREC_WORKER_COMMAND", "") or "").strip(),
            worker_python_executable=(os.getenv("NUREC_WORKER_PYTHON", "python") or "python").strip(),
        )
        self._repo_src = Path(__file__).resolve().parents[1]

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
        worker_mode = self.config.worker_mode.strip().lower()
        if worker_mode == "external_markers":
            # External scheduler picks up nurec_job_spec.json and writes markers.
            return

        if worker_mode == "local_worker":
            command = [
                self.config.worker_python_executable,
                "-m",
                "blueprint_pipeline.nurec_worker",
                "--job-spec",
                str(spec_path),
                "--storage-root",
                str(self.storage_root),
            ]
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH", "")
            parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
            repo_src = str(self._repo_src)
            if repo_src not in parts:
                parts.insert(0, repo_src)
            env["PYTHONPATH"] = os.pathsep.join(parts)
            proc = subprocess.run(
                command,
                cwd=str(self.storage_root),
                env=env,
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
                    "worker_mode": worker_mode,
                    "command": command,
                    "return_code": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                    "failed_at": utc_now_iso(),
                }
                write_json(self._marker(".nurec_failed"), error)
                raise StageError("nurec", f"local worker failed with code {proc.returncode}")
            return

        if worker_mode == "command":
            command_template = self.config.worker_command.strip()
            if not command_template:
                raise StageError(
                    "nurec",
                    "NUREC_WORKER_MODE=command requires NUREC_WORKER_COMMAND",
                )

            command = (
                command_template.replace("{JOB_SPEC_PATH}", str(spec_path))
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
                    "worker_mode": worker_mode,
                    "command": command,
                    "return_code": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                    "failed_at": utc_now_iso(),
                }
                write_json(self._marker(".nurec_failed"), error)
                raise StageError("nurec", f"worker dispatch failed with code {proc.returncode}")
            return

        raise StageError("nurec", f"unsupported NUREC_WORKER_MODE: {worker_mode}")

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

    def _read_ply_face_count(self, mesh_path: Path) -> int:
        with mesh_path.open("rb") as f:
            first = f.readline().decode("ascii", errors="ignore").strip().lower()
            if first != "ply":
                raise StageError("nurec", f"invalid PLY header at {mesh_path}")
            while True:
                line = f.readline()
                if not line:
                    raise StageError("nurec", f"unexpected EOF while reading PLY header at {mesh_path}")
                text = line.decode("ascii", errors="ignore").strip().lower()
                if text.startswith("element face "):
                    return int(text.split()[-1])
                if text == "end_header":
                    break
        return 0

    def collect_outputs(self) -> Dict[str, Any]:
        export_usdz = self.nurec_dir / "export_last.usdz"
        mesh_ply = self.nurec_dir / "nvblox_mesh.ply"
        visual_mesh_glb = self.nurec_dir / "visual_mesh.glb"
        visual_pointcloud_ply = self.nurec_dir / "visual_pointcloud.ply"
        inpainted_visual_mesh_glb = self.nurec_dir / "inpainted_visual_mesh.glb"
        mesh_manifest_json = self.nurec_dir / "mesh_manifest.json"
        instance_masks_dir = self.nurec_dir / "instance_masks"
        undistorted_root = self.nurec_dir / "colmap_undistorted"
        undistorted_sparse_dir = undistorted_root / "sparse" / "0"
        undistorted_images_dir = undistorted_root / "images"
        occupancy = sorted(self.nurec_dir.glob("occupancy*"))
        visual_mesh_enabled = _env_flag("VISUAL_MESH_ENABLED", True)

        if not has_nonempty_file(export_usdz):
            raise StageError("nurec", f"missing required artifact: {export_usdz}")
        if not has_nonempty_file(mesh_ply):
            raise StageError("nurec", f"missing required artifact: {mesh_ply}")
        if visual_mesh_enabled and not has_nonempty_file(visual_mesh_glb):
            raise StageError("nurec", f"missing required visual mesh artifact: {visual_mesh_glb}")
        face_count = self._read_ply_face_count(mesh_ply)
        if face_count <= 0:
            raise StageError("nurec", f"collision mesh must be triangulated (face_count={face_count})")
        if not occupancy:
            raise StageError("nurec", "missing occupancy.* artifact")

        selected_visual_usdz = export_usdz
        hallucinated_region_mask: Path | None = None
        if has_nonempty_file(mesh_manifest_json):
            try:
                manifest_payload = json.loads(mesh_manifest_json.read_text(encoding="utf-8"))
            except Exception:
                manifest_payload = {}
            if isinstance(manifest_payload, dict):
                primary_visual = str(manifest_payload.get("primary_visual_asset") or "").strip()
                if primary_visual.lower().endswith(".usdz"):
                    candidate_visual = self.nurec_dir / primary_visual
                    if has_nonempty_file(candidate_visual):
                        selected_visual_usdz = candidate_visual
                hallucinated_mask_name = str(manifest_payload.get("hallucinated_region_mask") or "").strip()
                if hallucinated_mask_name:
                    candidate_mask = self.nurec_dir / hallucinated_mask_name
                    if has_nonempty_file(candidate_mask):
                        hallucinated_region_mask = candidate_mask

        mesh_stats = self._mesh_stats(mesh_ply)
        mesh_stats.setdefault("face_count", int(face_count))
        artifacts: Dict[str, Any] = {
            "visual_usdz": f"gs://{self.bucket}/{relative_scene_path(selected_visual_usdz, self.storage_root)}",
            "collision_mesh_ply": f"gs://{self.bucket}/{relative_scene_path(mesh_ply, self.storage_root)}",
            "occupancy": [
                f"gs://{self.bucket}/{relative_scene_path(path, self.storage_root)}" for path in occupancy
            ],
        }
        if has_nonempty_file(visual_mesh_glb):
            artifacts["visual_mesh_glb"] = (
                f"gs://{self.bucket}/{relative_scene_path(visual_mesh_glb, self.storage_root)}"
            )
        if has_nonempty_file(inpainted_visual_mesh_glb):
            artifacts["inpainted_visual_mesh_glb"] = (
                f"gs://{self.bucket}/{relative_scene_path(inpainted_visual_mesh_glb, self.storage_root)}"
            )
        if has_nonempty_file(visual_pointcloud_ply):
            artifacts["visual_pointcloud_ply"] = (
                f"gs://{self.bucket}/{relative_scene_path(visual_pointcloud_ply, self.storage_root)}"
            )
        if has_nonempty_file(mesh_manifest_json):
            artifacts["mesh_manifest_json"] = (
                f"gs://{self.bucket}/{relative_scene_path(mesh_manifest_json, self.storage_root)}"
            )
        if instance_masks_dir.is_dir() and any(instance_masks_dir.glob("*.png")):
            artifacts["sam3_instance_masks_dir"] = (
                f"gs://{self.bucket}/{relative_scene_path(instance_masks_dir, self.storage_root)}"
            )
        if undistorted_sparse_dir.is_dir() and any(undistorted_sparse_dir.iterdir()):
            artifacts["colmap_undistorted_sparse_dir"] = (
                f"gs://{self.bucket}/{relative_scene_path(undistorted_sparse_dir, self.storage_root)}"
            )
        if undistorted_images_dir.is_dir() and any(p.is_file() for p in undistorted_images_dir.rglob("*")):
            artifacts["colmap_undistorted_images_dir"] = (
                f"gs://{self.bucket}/{relative_scene_path(undistorted_images_dir, self.storage_root)}"
            )
        if hallucinated_region_mask is not None:
            artifacts["hallucinated_region_mask"] = (
                f"gs://{self.bucket}/{relative_scene_path(hallucinated_region_mask, self.storage_root)}"
            )

        payload = {
            "schema_version": "v1",
            "scene_prefix": self.pipeline_prefix,
            "status": "completed",
            "generated_at": utc_now_iso(),
            "artifacts": artifacts,
            "mesh_stats": mesh_stats,
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
        for marker_name in (".nurec_complete", ".nurec_failed"):
            marker_path = self._marker(marker_name)
            if marker_path.exists():
                marker_path.unlink()
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
