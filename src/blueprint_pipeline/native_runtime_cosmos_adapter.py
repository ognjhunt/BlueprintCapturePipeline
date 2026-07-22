"""Explicit compatibility adapter for legacy Cosmos-Predict2.5 runtime assets.

The native runtime service owns the stable session and media contracts. This
module contains the model-family-specific filesystem layout and subprocess
invocation needed by older Cosmos-backed deployments so those details do not
leak through the runtime store itself.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

from PIL import Image


ProcessRunner = Callable[..., Any]
CopyFile = Callable[[Path, Path], Any]
LoadSiteWorld = Callable[[str], Mapping[str, Any]]
ExtractFrames = Callable[[Path, Path], List[Path]]
ConvertVideo = Callable[[Path, Path], bool]
WriteStatus = Callable[[Path, Mapping[str, Any]], None]


@dataclass(frozen=True)
class LegacyCosmosRuntimeAdapter:
    """Model-specific adapter used only by the explicit ``cosmos_wam`` lane."""

    storage_root: Path
    load_site_world: LoadSiteWorld
    process_runner: ProcessRunner
    copy_file: CopyFile
    environment: Mapping[str, str]

    def _pipeline_root(self, site_world_id: str) -> Optional[Path]:
        try:
            site_world = self.load_site_world(site_world_id)
        except FileNotFoundError:
            return None
        scene_id = str(site_world.get("scene_id") or "").strip()
        capture_id = str(site_world.get("capture_id") or "").strip()
        if not scene_id or not capture_id:
            return None
        # ``vast-local`` is the legacy artifact namespace. It is deliberately
        # confined to this adapter rather than presented as a runtime contract.
        return (
            self.storage_root
            / "vast-local"
            / "scenes"
            / scene_id
            / "captures"
            / capture_id
            / "pipeline"
        )

    def find_prebuilt_video(self, site_world_id: str) -> Optional[Path]:
        """Find an opt-in legacy bootstrap artifact for a site world."""
        pipeline_root = self._pipeline_root(site_world_id)
        if pipeline_root is not None:
            candidates = (
                pipeline_root
                / "cosmos_single_capture_smoke"
                / "renders"
                / "video_bootstrap_0000.mp4",
                pipeline_root
                / "cosmos_single_capture_smoke"
                / "renders"
                / "video_bootstrap_0000.jpg",
            )
            for candidate in candidates:
                try:
                    if candidate.is_file():
                        return candidate
                except OSError:
                    continue
        fallback = (
            self.storage_root
            / "manual_cosmos_probe_official"
            / "blueprint_probe.mp4"
        )
        try:
            return fallback if fallback.is_file() else None
        except OSError:
            return None

    def find_conditioning_frame(self, site_world_id: str) -> Optional[Path]:
        """Find the legacy conditioning frame for on-demand inference."""
        pipeline_root = self._pipeline_root(site_world_id)
        if pipeline_root is None:
            return None
        candidates = (
            pipeline_root
            / "cosmos_single_capture_smoke"
            / "video_bootstrap_frames"
            / "frame_0000.jpg",
            pipeline_root
            / "cosmos_single_capture_smoke"
            / "renders"
            / "video_bootstrap_0000.jpg",
        )
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def extract_frames_from_video(
        self,
        video_path: Path,
        frames_dir: Path,
    ) -> List[Path]:
        """Extract PNG frames from a video at the compatibility playback rate."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.process_runner(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vf",
                    "fps=4",
                    str(frames_dir / "frame_%04d.png"),
                    "-y",
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return []
        return sorted(frames_dir.glob("frame_*.png"))

    def extract_single_frame(
        self,
        image_path: Path,
        frames_dir: Path,
    ) -> List[Path]:
        """Normalize one bootstrap image into the frame-cache layout."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_out = frames_dir / "frame_0001.png"
        try:
            image = Image.open(image_path).convert("RGB")
            image.save(frame_out, format="PNG")
        except Exception:
            try:
                self.copy_file(image_path, frame_out)
            except OSError:
                return []
        return [frame_out] if frame_out.is_file() else []

    def find_lora_adapter(self, site_world_id: str) -> Optional[Path]:
        """Resolve the explicit or legacy per-capture LoRA checkpoint."""
        explicit = str(
            self.environment.get("COSMOS_LORA_CHECKPOINT_PATH") or ""
        ).strip()
        if explicit:
            explicit_path = Path(explicit)
            return explicit_path if explicit_path.is_file() else None
        pipeline_root = self._pipeline_root(site_world_id)
        if pipeline_root is None:
            return None
        adapter = (
            pipeline_root
            / "cosmos_training_export"
            / "checkpoints"
            / "adapter_model.safetensors"
        )
        try:
            return adapter if adapter.is_file() else None
        except OSError:
            return None

    def run_inference_sync(
        self,
        *,
        session_id: str,
        cosmos_repo: Tuple[Path, Path],
        cond_frame: Path,
        frames_dir: Path,
        cosmos_dir: Path,
        status_path: Path,
        extract_frames: ExtractFrames,
        convert_video: ConvertVideo,
        write_status: WriteStatus,
        timestamp: Callable[[], str],
        lora_adapter: Optional[Path] = None,
    ) -> List[Path]:
        """Invoke the legacy Cosmos checkout and normalize its video output."""
        repo_root, python_bin = cosmos_repo
        cosmos_dir.mkdir(parents=True, exist_ok=True)
        sample_name = f"cosmos_{session_id[:8]}"
        asset_path = cosmos_dir / f"{sample_name}.json"
        output_video = cosmos_dir / f"{sample_name}.mp4"
        log_path = cosmos_dir / "inference.log"

        chunk_size = max(
            8,
            int(
                self.environment.get("COSMOS_CHUNK_SIZE")
                or self.environment.get("NATIVE_WORLD_MODEL_CHUNK_FRAMES")
                or "57"
            ),
        )
        chunk_overlap = max(
            1,
            int(self.environment.get("COSMOS_CHUNK_OVERLAP") or "4"),
        )
        asset_path.write_text(
            json.dumps(
                {
                    "inference_type": "image2world",
                    "name": sample_name,
                    "input_path": str(cond_frame.resolve()),
                    "prompt": (
                        "First-person camera moving through a real indoor workspace. "
                        "Preserve the existing geometry and continue the scene naturally."
                    ),
                    "num_output_frames": chunk_size,
                    "num_steps": 35,
                    "seed": 0,
                    "guidance": 7.0,
                    "enable_autoregressive": False,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            ),
            encoding="utf-8",
        )

        process_env = {
            key: value
            for key, value in self.environment.items()
            if isinstance(value, str)
        }
        process_env["PATH"] = (
            str(repo_root / ".venv" / "bin")
            + os.pathsep
            + process_env.get("PATH", "")
        )
        command = [
            str(python_bin),
            "examples/inference.py",
            "-i",
            str(asset_path),
            "-o",
            str(cosmos_dir),
            "--model=2B/post-trained",
            "--disable-guardrails",
        ]
        if lora_adapter and lora_adapter.is_file():
            command += ["--lora-checkpoint", str(lora_adapter)]

        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                result = self.process_runner(
                    command,
                    cwd=str(repo_root),
                    env=process_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError:
                return []
        if result.returncode != 0 or not output_video.is_file():
            return []

        fragmented_video = output_video.with_stem(output_video.stem + "_fmp4")
        if convert_video(output_video, fragmented_video):
            output_video = fragmented_video
        frames = extract_frames(output_video, frames_dir)
        if frames:
            write_status(
                status_path,
                {
                    "source": "on_demand_inference",
                    "video": str(output_video),
                    "frame_count": len(frames),
                    "inferred_at": timestamp(),
                },
            )
        return frames
