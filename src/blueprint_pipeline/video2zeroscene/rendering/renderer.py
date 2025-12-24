"""Main 3D Gaussian Splatting renderer.

This module provides the GaussianRenderer class, which is the primary interface
for rendering 3DGS scenes. It handles:

- Loading Gaussians from PLY files
- Loading camera trajectories from ZeroScene bundles
- Rendering frames using the best available backend
- Saving output as video or image sequences

Usage:
    # Load from ZeroScene bundle
    renderer = GaussianRenderer.from_zeroscene("output/zeroscene")

    # Render all frames
    frames = renderer.render_trajectory()

    # Save as video
    renderer.save_video(frames, "static_scene.mp4")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .camera_utils import Camera, CameraTrajectory
from .gaussian_model import GaussianModel
from .rasterizer import RasterOutput, get_best_rasterizer, available_backends

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class RenderSettings:
    """Settings for 3DGS rendering.

    Attributes:
        background_color: Background color (R, G, B) in [0, 1]
        scaling_modifier: Scale multiplier for Gaussians
        sh_degree: SH degree to use (-1 for auto from model)
        compute_depth: Whether to compute depth maps
        min_opacity: Minimum opacity filter
        max_sh_degree: Maximum SH degree to use (limits quality for speed)
    """
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scaling_modifier: float = 1.0
    sh_degree: int = -1  # -1 = auto (use model's degree)
    compute_depth: bool = False
    min_opacity: float = 0.0
    max_sh_degree: int = 3


class GaussianRenderer:
    """High-level renderer for 3D Gaussian Splatting scenes.

    This class provides a simple interface for rendering 3DGS scenes loaded
    from ZeroScene bundles or individual files.

    Example:
        # From ZeroScene bundle
        renderer = GaussianRenderer.from_zeroscene("output/zeroscene")
        frames = renderer.render_trajectory()
        renderer.save_video(frames, "output.mp4")

        # From individual files
        renderer = GaussianRenderer(
            gaussians_path="gaussians.ply",
            trajectory_path="trajectory.json",
            intrinsics_path="intrinsics.json",
        )
        frames = renderer.render_trajectory()
    """

    def __init__(
        self,
        gaussians_path: Optional[Union[str, Path]] = None,
        trajectory_path: Optional[Union[str, Path]] = None,
        intrinsics_path: Optional[Union[str, Path]] = None,
        model: Optional[GaussianModel] = None,
        trajectory: Optional[CameraTrajectory] = None,
        backend: Optional[str] = None,
        device: str = "cuda",
        settings: Optional[RenderSettings] = None,
    ):
        """Initialize the renderer.

        Args:
            gaussians_path: Path to Gaussians PLY file
            trajectory_path: Path to camera trajectory JSON
            intrinsics_path: Path to camera intrinsics JSON
            model: Pre-loaded GaussianModel (alternative to gaussians_path)
            trajectory: Pre-loaded CameraTrajectory (alternative to files)
            backend: Rasterizer backend name (auto-select if None)
            device: Device to use ('cuda' or 'cpu')
            settings: Render settings (uses defaults if None)
        """
        self.settings = settings or RenderSettings()
        self.device = device

        # Initialize rasterizer
        self._rasterizer = get_best_rasterizer(preferred=backend, device=device)
        print(f"Using rasterizer: {self._rasterizer.name}")

        # Load or use provided model
        if model is not None:
            self.model = model
        elif gaussians_path is not None:
            self.model = GaussianModel(device=device)
            self.model.load_ply(gaussians_path)
        else:
            self.model = None

        # Load or use provided trajectory
        if trajectory is not None:
            self.trajectory = trajectory
        elif trajectory_path is not None and intrinsics_path is not None:
            self.trajectory = CameraTrajectory.from_files(
                intrinsics_path, trajectory_path
            )
        else:
            self.trajectory = None

    @classmethod
    def from_zeroscene(
        cls,
        zeroscene_path: Union[str, Path],
        backend: Optional[str] = None,
        device: str = "cuda",
        settings: Optional[RenderSettings] = None,
    ) -> "GaussianRenderer":
        """Create renderer from ZeroScene bundle.

        Args:
            zeroscene_path: Path to zeroscene/ directory
            backend: Rasterizer backend name (auto-select if None)
            device: Device to use ('cuda' or 'cpu')
            settings: Render settings

        Returns:
            Initialized GaussianRenderer
        """
        zeroscene_path = Path(zeroscene_path)

        # Verify DWM compatibility
        scene_info_path = zeroscene_path / "scene_info.json"
        if scene_info_path.exists():
            scene_info = json.loads(scene_info_path.read_text())
            if not scene_info.get("has_gaussians", False):
                raise ValueError(
                    f"ZeroScene bundle does not contain Gaussians. "
                    f"Run pipeline with DWM-compatible export."
                )
            if not scene_info.get("dwm_compatible", False):
                print("Warning: Scene is not marked as DWM compatible")

        # Find Gaussians file
        gaussians_path = zeroscene_path / "background" / "gaussians.ply"
        if not gaussians_path.exists():
            raise FileNotFoundError(f"Gaussians not found: {gaussians_path}")

        # Load trajectory
        trajectory = CameraTrajectory.from_zeroscene(zeroscene_path)

        # Load model
        model = GaussianModel(device=device)
        model.load_ply(gaussians_path)

        return cls(
            model=model,
            trajectory=trajectory,
            backend=backend,
            device=device,
            settings=settings,
        )

    def render_frame(
        self,
        camera: Camera,
        settings: Optional[RenderSettings] = None,
    ) -> RasterOutput:
        """Render a single frame from a camera viewpoint.

        Args:
            camera: Camera parameters
            settings: Render settings (uses instance settings if None)

        Returns:
            RasterOutput with rendered image
        """
        if self.model is None:
            raise RuntimeError("No Gaussian model loaded")

        settings = settings or self.settings

        # Determine SH degree
        sh_degree = settings.sh_degree
        if sh_degree < 0:
            sh_degree = min(self.model.sh_degree, settings.max_sh_degree)

        # Filter by opacity if needed
        model = self.model
        if settings.min_opacity > 0:
            model = model.filter_by_opacity(settings.min_opacity)

        # Prepare camera dict for rasterizer
        viewpoint_camera = {
            "R": camera.R,
            "T": camera.T,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "width": camera.width,
            "height": camera.height,
            "z_near": camera.z_near,
            "z_far": camera.z_far,
        }

        # Render
        output = self._rasterizer.rasterize(
            means3D=model.xyz,
            opacities=model.opacities,
            scales=model.scales,
            rotations=model.rotations,
            sh_dc=model._sh_dc,
            sh_rest=model._sh_rest,
            sh_degree=sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=settings.background_color,
            scaling_modifier=settings.scaling_modifier,
            compute_depth=settings.compute_depth,
        )

        return output

    def render_trajectory(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        settings: Optional[RenderSettings] = None,
    ) -> List[np.ndarray]:
        """Render all frames along the camera trajectory.

        Args:
            progress_callback: Optional callback(current, total) for progress
            settings: Render settings (uses instance settings if None)

        Returns:
            List of rendered frames as numpy arrays (H, W, 3) in [0, 1]
        """
        if self.trajectory is None:
            raise RuntimeError("No camera trajectory loaded")

        frames = []
        total = len(self.trajectory)

        for i, camera in enumerate(self.trajectory):
            output = self.render_frame(camera, settings)
            frames.append(output.color)

            if progress_callback:
                progress_callback(i + 1, total)
            elif (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"Rendered {i + 1}/{total} frames")

        return frames

    def render_novel_view(
        self,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0, 1, 0),
        fov: float = 60.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        settings: Optional[RenderSettings] = None,
    ) -> RasterOutput:
        """Render from a novel camera viewpoint.

        Args:
            position: Camera position in world coordinates
            look_at: Point to look at in world coordinates
            up: Up vector
            fov: Field of view in degrees
            width: Image width (uses trajectory width if None)
            height: Image height (uses trajectory height if None)
            settings: Render settings

        Returns:
            RasterOutput with rendered image
        """
        import math

        # Get dimensions from trajectory if not specified
        if width is None or height is None:
            if self.trajectory is not None and len(self.trajectory) > 0:
                width = width or self.trajectory.intrinsics["width"]
                height = height or self.trajectory.intrinsics["height"]
            else:
                width = width or 1920
                height = height or 1080

        # Compute view matrix
        position = np.array(position, dtype=np.float32)
        look_at = np.array(look_at, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Rotation matrix (world-to-camera)
        R = np.array([right, -up, forward], dtype=np.float32)
        T = -R @ position

        # Compute focal length from FOV
        fov_rad = math.radians(fov)
        fy = height / (2 * math.tan(fov_rad / 2))
        fx = fy  # Assume square pixels

        # Create camera
        camera = Camera(
            uid=-1,
            R=R,
            T=T,
            fx=fx,
            fy=fy,
            cx=width / 2,
            cy=height / 2,
            width=width,
            height=height,
        )

        return self.render_frame(camera, settings)

    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        """Save rendered frames as video.

        Args:
            frames: List of frames (H, W, 3) in [0, 1]
            output_path: Output video path
            fps: Frames per second
            codec: Video codec (mp4v, avc1, etc.)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video saving: pip install opencv-python")

        if len(frames) == 0:
            raise ValueError("No frames to save")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            # Convert to uint8 BGR
            frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"Saved video to {output_path} ({len(frames)} frames at {fps} FPS)")

    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "frame",
        format: str = "png",
    ) -> None:
        """Save rendered frames as individual images.

        Args:
            frames: List of frames (H, W, 3) in [0, 1]
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format (png, jpg, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

            if PIL_AVAILABLE:
                img = Image.fromarray(frame_uint8)
                img.save(output_dir / f"{prefix}_{i:06d}.{format}")
            elif CV2_AVAILABLE:
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / f"{prefix}_{i:06d}.{format}"), frame_bgr)
            else:
                raise ImportError("PIL or OpenCV required for image saving")

        print(f"Saved {len(frames)} frames to {output_dir}")

    def get_scene_info(self) -> Dict[str, Any]:
        """Get information about the loaded scene.

        Returns:
            Dictionary with scene statistics
        """
        info = {
            "rasterizer": self._rasterizer.name,
            "device": self.device,
        }

        if self.model is not None:
            bounds_min, bounds_max = self.model.bounds
            info.update({
                "num_gaussians": self.model.num_gaussians,
                "sh_degree": self.model.sh_degree,
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "center": self.model.center.tolist(),
                "extent": float(self.model.extent),
            })

        if self.trajectory is not None:
            cam_bounds = self.trajectory.get_bounds()
            info.update({
                "num_cameras": len(self.trajectory),
                "camera_bounds_min": cam_bounds[0].tolist(),
                "camera_bounds_max": cam_bounds[1].tolist(),
            })

            if self.trajectory.intrinsics:
                info.update({
                    "image_width": self.trajectory.intrinsics["width"],
                    "image_height": self.trajectory.intrinsics["height"],
                    "fx": self.trajectory.intrinsics["fx"],
                    "fy": self.trajectory.intrinsics["fy"],
                })

        return info

    def __repr__(self) -> str:
        model_str = f"{self.model.num_gaussians:,} Gaussians" if self.model else "no model"
        traj_str = f"{len(self.trajectory)} cameras" if self.trajectory else "no trajectory"
        return f"GaussianRenderer({model_str}, {traj_str}, backend={self._rasterizer.name})"


def render_zeroscene(
    zeroscene_path: Union[str, Path],
    output_path: Union[str, Path],
    fps: float = 30.0,
    backend: Optional[str] = None,
    device: str = "cuda",
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Convenience function to render a ZeroScene bundle to video.

    Args:
        zeroscene_path: Path to zeroscene/ directory
        output_path: Output video path
        fps: Frames per second
        backend: Rasterizer backend (auto-select if None)
        device: Device to use
        background: Background color
    """
    settings = RenderSettings(background_color=background)

    renderer = GaussianRenderer.from_zeroscene(
        zeroscene_path,
        backend=backend,
        device=device,
        settings=settings,
    )

    print(f"Scene info: {renderer.get_scene_info()}")

    frames = renderer.render_trajectory()
    renderer.save_video(frames, output_path, fps=fps)
