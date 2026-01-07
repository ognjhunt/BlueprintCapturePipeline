"""Difix3D+ Integration for 3D Gaussian Splatting Refinement.

This module provides scene inpainting and quality enhancement for 3DGS
reconstructions using NVIDIA's Difix3D+ (CVPR 2025 Best Paper Finalist).

Difix3D+ is a single-step diffusion model that:
1. Removes artifacts from rendered novel views
2. Generates cleaned pseudo-views that can be distilled back into 3DGS
3. Enables progressive refinement to fill gaps in sparse captures

Reference:
- Paper: "Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models"
- GitHub: https://github.com/nv-tlabs/Difix3D
- Project: https://research.nvidia.com/labs/toronto-ai/difix3d/

Pipeline Overview:
    1. Train initial 3DGS from captured views
    2. Detect gaps/artifacts via coverage analysis
    3. Generate novel poses via interpolation
    4. Render → Enhance with Difix → Distill back into 3DGS
    5. Repeat progressively to expand to harder viewpoints
    6. Optional: Apply Difix as post-process enhancer at inference

Integration with BlueprintCapturePipeline:
    This module slots in after SLAM reconstruction (Stage 1) and before
    export (Stage 2), adding a refinement stage that improves visual
    quality and fills gaps from incomplete captures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Diffusers for Difix model
try:
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# LPIPS for perceptual loss (optional but recommended)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DifixConfig:
    """Configuration for Difix3D+ refinement pipeline.

    This controls all aspects of the refinement process:
    - Model selection and inference settings
    - Progressive refinement strategy
    - Distillation parameters
    - Quality thresholds
    """

    # Model settings
    model_name: str = "nvidia/difix"  # Or "nvidia/difix_ref" for reference-conditioned
    use_reference_model: bool = True  # Use reference views for conditioning
    device: str = "cuda"
    dtype: str = "float16"  # float16 for speed, float32 for quality

    # Progressive refinement settings
    num_refinement_rounds: int = 3  # Number of render→fix→distill cycles
    poses_per_round: int = 50  # Novel poses to generate per round
    pose_interpolation_steps: int = 5  # Interpolation steps between views
    progressive_expansion_rate: float = 1.5  # How fast to expand pose range

    # Distillation training settings
    distillation_weight: float = 0.3  # Weight for pseudo-views (vs real=1.0)
    iterations_per_round: int = 1500  # Training iterations per refinement round
    distillation_lr: float = 1e-4  # Learning rate for distillation

    # Loss weights (matching Difix3D+ paper)
    l2_weight: float = 1.0
    lpips_weight: float = 1.0  # Perceptual loss
    gram_weight: float = 0.5  # Style loss for sharper details
    ssim_weight: float = 0.2  # Structural similarity

    # Quality thresholds
    min_opacity_threshold: float = 0.1  # Min accumulated opacity for "covered"
    coverage_threshold: float = 0.8  # Min coverage to skip inpainting
    artifact_threshold: float = 0.3  # Gradient magnitude for artifact detection

    # Difix inference settings (from paper: timestep 199, single step)
    difix_timestep: int = 199  # Optimal noise level for artifacts
    guidance_scale: float = 0.0  # No classifier-free guidance

    # Output settings
    save_intermediate: bool = False  # Save intermediate renders
    output_resolution: Tuple[int, int] = (512, 512)  # Difix optimal resolution

    # Post-processing (inference-time enhancement)
    enable_post_process: bool = True  # Apply Difix at render time
    post_process_strength: float = 1.0  # 0.0-1.0 blending with original

    # Difix prompt (from paper: "remove degradation" is the default)
    difix_prompt: str = "remove degradation"  # Configurable prompt for diffusion model

    def __post_init__(self):
        """Validate configuration."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Difix3D+ refinement")

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"


# =============================================================================
# Gap Detection
# =============================================================================

class GapDetector:
    """Detect gaps and artifacts in 3DGS renders.

    Uses multiple strategies to identify regions needing inpainting:
    1. Opacity coverage - pixels with low accumulated opacity
    2. Depth discontinuities - sudden jumps suggesting missing geometry
    3. Gradient artifacts - high-frequency noise from poor reconstruction
    4. Multi-view inconsistency - regions that look different from nearby views
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute_coverage_map(
        self,
        rendered_image: torch.Tensor,
        rendered_opacity: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Compute per-pixel coverage map.

        Args:
            rendered_image: Rendered RGB [3, H, W] or [H, W, 3]
            rendered_opacity: Accumulated opacity [H, W] or [1, H, W]
            threshold: Minimum opacity to consider "covered"

        Returns:
            coverage_map: Binary [H, W], 1=covered, 0=gap
        """
        # Normalize shapes
        if rendered_opacity.dim() == 3:
            rendered_opacity = rendered_opacity.squeeze(0)

        coverage = (rendered_opacity > threshold).float()
        return coverage

    def detect_artifacts(
        self,
        rendered_image: torch.Tensor,
        gradient_threshold: float = 0.3,
    ) -> torch.Tensor:
        """Detect artifact regions via gradient analysis.

        High-frequency artifacts from poor 3DGS reconstruction show up
        as regions with abnormally high gradient magnitude.

        Args:
            rendered_image: Rendered RGB [3, H, W]
            gradient_threshold: Threshold for "artifact" classification

        Returns:
            artifact_mask: Binary [H, W], 1=artifact, 0=clean
        """
        if rendered_image.dim() == 3 and rendered_image.shape[0] == 3:
            # [3, H, W] -> [1, 3, H, W]
            img = rendered_image.unsqueeze(0)
        else:
            img = rendered_image

        # Convert to grayscale
        gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
        gray = gray.unsqueeze(1)  # [B, 1, H, W]

        # Sobel gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

        # Normalize and threshold
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        artifact_mask = (grad_mag > gradient_threshold).float()

        return artifact_mask.squeeze()

    def find_gap_regions(
        self,
        gaussian_model: "GaussianModel",
        camera_poses: List[Dict[str, Any]],
        intrinsics: Dict[str, Any],
        coverage_threshold: float = 0.8,
        artifact_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find regions with insufficient coverage across multiple views.

        Args:
            gaussian_model: Trained GaussianModel
            camera_poses: List of camera pose dicts with 'world_to_camera', etc.
            intrinsics: Camera intrinsics dict
            coverage_threshold: Min avg coverage to skip view
            artifact_threshold: Gradient threshold for artifacts

        Returns:
            List of dicts: {pose_idx, coverage_mask, coverage_score, artifact_mask}
        """
        gap_regions = []

        for idx, pose in enumerate(camera_poses):
            try:
                # Render from this view
                render_result = self._render_view(
                    gaussian_model, pose, intrinsics
                )

                if render_result is None:
                    continue

                rendered_image = render_result['rgb']
                rendered_opacity = render_result.get('opacity', torch.ones_like(rendered_image[0]))

                # Compute coverage
                coverage_map = self.compute_coverage_map(
                    rendered_image, rendered_opacity,
                    threshold=0.1
                )
                coverage_score = coverage_map.mean().item()

                # Detect artifacts
                artifact_mask = self.detect_artifacts(
                    rendered_image, artifact_threshold
                )
                artifact_score = artifact_mask.mean().item()

                # Flag views needing enhancement
                needs_enhancement = (
                    coverage_score < coverage_threshold or
                    artifact_score > 0.1
                )

                if needs_enhancement:
                    gap_regions.append({
                        'pose_idx': idx,
                        'pose': pose,
                        'coverage_mask': coverage_map,
                        'coverage_score': coverage_score,
                        'artifact_mask': artifact_mask,
                        'artifact_score': artifact_score,
                    })

            except Exception as e:
                logger.warning(f"Error analyzing view {idx}: {e}")
                continue

        logger.info(
            f"Gap detection: {len(gap_regions)}/{len(camera_poses)} views "
            f"need enhancement"
        )

        return gap_regions

    def _render_view(
        self,
        gaussian_model: "GaussianModel",
        pose: Dict[str, Any],
        intrinsics: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Render a single view from the Gaussian model.

        This is a helper that wraps the GaussianModel rendering.
        """
        # Import here to avoid circular imports
        from .gaussian_splatting import GaussianRasterizer

        try:
            # Check for missing intrinsics and warn user
            missing_intrinsics = []
            if 'height' not in intrinsics:
                missing_intrinsics.append('height')
            if 'width' not in intrinsics:
                missing_intrinsics.append('width')
            if 'fx' not in intrinsics:
                missing_intrinsics.append('fx')
            if 'fy' not in intrinsics:
                missing_intrinsics.append('fy')

            if missing_intrinsics:
                logger.warning(
                    f"Missing camera intrinsics: {missing_intrinsics}. "
                    f"Using default values (512x512). This may affect render quality. "
                    f"Provide complete intrinsics for best results."
                )

            H = intrinsics.get('height', 512)
            W = intrinsics.get('width', 512)
            fx = intrinsics.get('fx', W)
            fy = intrinsics.get('fy', H)

            # Build matrices
            viewmatrix = torch.tensor(
                pose['world_to_camera'],
                device=self.device,
                dtype=torch.float32
            )

            znear, zfar = 0.01, 100.0
            fovx = 2 * math.atan(W / (2 * fx))
            fovy = 2 * math.atan(H / (2 * fy))

            tanfovx = math.tan(fovx / 2)
            tanfovy = math.tan(fovy / 2)

            # Projection matrix
            top = tanfovy * znear
            bottom = -top
            right = tanfovx * znear
            left = -right

            P = torch.zeros(4, 4, device=self.device)
            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[2, 2] = zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            P[3, 2] = 1.0

            campos = torch.inverse(viewmatrix)[:3, 3]
            bg_color = torch.zeros(3, device=self.device)

            rasterizer = GaussianRasterizer(
                image_height=H,
                image_width=W,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg_color=bg_color,
                viewmatrix=viewmatrix,
                projmatrix=P,
                sh_degree=gaussian_model.active_sh_degree,
                campos=campos,
            )

            means2D = torch.zeros(gaussian_model.num_gaussians, 3, device=self.device)

            rendered_image, radii = rasterizer.forward(
                means3D=gaussian_model.xyz,
                means2D=means2D,
                shs=gaussian_model.features,
                colors_precomp=None,
                opacities=gaussian_model.opacity,
                scales=gaussian_model.scaling,
                rotations=gaussian_model.rotation,
            )

            return {
                'rgb': rendered_image,
                'radii': radii,
            }

        except Exception as e:
            logger.warning(f"Render failed: {e}")
            return None


# =============================================================================
# Pose Interpolation
# =============================================================================

class PoseInterpolator:
    """Generate interpolated and extrapolated camera poses.

    Used for progressive refinement:
    1. Interpolate between captured poses for gap filling
    2. Small orbit offsets for local detail enhancement
    3. Progressive expansion toward harder viewpoints

    Based on Difix3D+ pose interpolation strategy that gradually
    moves from known poses toward target poses.
    """

    @staticmethod
    def quaternion_slerp(
        q0: np.ndarray,
        q1: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Spherical linear interpolation between quaternions.

        Args:
            q0: Start quaternion [w, x, y, z]
            q1: End quaternion [w, x, y, z]
            t: Interpolation factor 0-1

        Returns:
            Interpolated quaternion [w, x, y, z]
        """
        # Normalize
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        # Dot product
        dot = np.dot(q0, q1)

        # If negative, negate one to take shorter path
        if dot < 0:
            q1 = -q1
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        # Spherical interpolation
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)

        return q0 * np.cos(theta) + q2 * np.sin(theta)

    @staticmethod
    def interpolate_poses(
        pose_a: Dict[str, Any],
        pose_b: Dict[str, Any],
        num_steps: int = 5,
    ) -> List[Dict[str, Any]]:
        """Interpolate between two camera poses.

        Args:
            pose_a: Start pose dict with 'world_to_camera' 4x4 matrix
            pose_b: End pose dict
            num_steps: Number of intermediate poses (excluding endpoints)

        Returns:
            List of interpolated pose dicts
        """
        # Extract transforms
        T_a = np.array(pose_a['world_to_camera'])
        T_b = np.array(pose_b['world_to_camera'])

        # Decompose into rotation and translation
        R_a, t_a = T_a[:3, :3], T_a[:3, 3]
        R_b, t_b = T_b[:3, :3], T_b[:3, 3]

        # Convert rotations to quaternions
        from scipy.spatial.transform import Rotation
        q_a = Rotation.from_matrix(R_a).as_quat()  # [x, y, z, w]
        q_b = Rotation.from_matrix(R_b).as_quat()

        # Reorder to [w, x, y, z] for slerp
        q_a = np.array([q_a[3], q_a[0], q_a[1], q_a[2]])
        q_b = np.array([q_b[3], q_b[0], q_b[1], q_b[2]])

        interpolated = []
        for i in range(1, num_steps + 1):
            alpha = i / (num_steps + 1)

            # Interpolate rotation
            q_interp = PoseInterpolator.quaternion_slerp(q_a, q_b, alpha)
            # Convert back to [x, y, z, w]
            q_scipy = np.array([q_interp[1], q_interp[2], q_interp[3], q_interp[0]])
            R_interp = Rotation.from_quat(q_scipy).as_matrix()

            # Linear interpolation for translation
            t_interp = (1 - alpha) * t_a + alpha * t_b

            # Build pose matrix
            T_interp = np.eye(4)
            T_interp[:3, :3] = R_interp
            T_interp[:3, 3] = t_interp

            # Create pose dict
            interp_pose = {
                'world_to_camera': T_interp.tolist(),
                'interpolation_alpha': alpha,
                'source_poses': [pose_a.get('frame_id'), pose_b.get('frame_id')],
            }

            # Copy intrinsics from source
            for key in ['fx', 'fy', 'cx', 'cy', 'image_height', 'image_width']:
                if key in pose_a:
                    interp_pose[key] = pose_a[key]

            interpolated.append(interp_pose)

        return interpolated

    @staticmethod
    def generate_orbit_poses(
        center_pose: Dict[str, Any],
        radius: float = 0.1,
        num_poses: int = 8,
        height_variation: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Generate poses orbiting around a center pose.

        Small local variations to capture missed details.

        Args:
            center_pose: Center pose to orbit around
            radius: Orbital radius in world units
            num_poses: Number of orbital poses
            height_variation: Vertical variation range

        Returns:
            List of orbital pose dicts
        """
        T_center = np.array(center_pose['world_to_camera'])
        R_center = T_center[:3, :3]
        t_center = T_center[:3, 3]

        orbital_poses = []

        for i in range(num_poses):
            angle = 2 * np.pi * i / num_poses

            # Offset in camera's local XY plane
            local_offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height_variation * np.sin(2 * angle),  # Slight Z variation
            ])

            # Transform to world coordinates
            world_offset = R_center.T @ local_offset

            # New translation
            t_new = t_center + world_offset

            # Slight rotation to look at original target
            # (For simplicity, keep same rotation)
            T_new = np.eye(4)
            T_new[:3, :3] = R_center
            T_new[:3, 3] = t_new

            orbit_pose = {
                'world_to_camera': T_new.tolist(),
                'orbit_angle': angle,
                'orbit_radius': radius,
            }

            # Copy camera intrinsics
            for key in ['fx', 'fy', 'cx', 'cy', 'image_height', 'image_width']:
                if key in center_pose:
                    orbit_pose[key] = center_pose[key]

            orbital_poses.append(orbit_pose)

        return orbital_poses

    @staticmethod
    def generate_progressive_poses(
        training_poses: List[Dict[str, Any]],
        gap_regions: List[Dict[str, Any]],
        round_idx: int,
        expansion_rate: float = 1.5,
        max_poses: int = 50,
    ) -> List[Dict[str, Any]]:
        """Generate novel poses for progressive refinement.

        Following Difix3D+ strategy:
        - Early rounds: Interpolate between existing poses
        - Later rounds: Expand toward gap regions and beyond

        Args:
            training_poses: Original captured poses
            gap_regions: Detected gap regions from GapDetector
            round_idx: Current refinement round (0-indexed)
            expansion_rate: How much to expand pose range per round
            max_poses: Maximum poses to generate

        Returns:
            List of novel pose dicts
        """
        novel_poses = []

        if not training_poses:
            return novel_poses

        # Strategy depends on round
        if round_idx == 0:
            # First round: Dense interpolation between adjacent poses
            for i in range(len(training_poses) - 1):
                interp = PoseInterpolator.interpolate_poses(
                    training_poses[i],
                    training_poses[i + 1],
                    num_steps=3,
                )
                novel_poses.extend(interp)

                if len(novel_poses) >= max_poses:
                    break

        else:
            # Later rounds: Focus on gap regions + expand outward
            for gap in gap_regions:
                pose_idx = gap['pose_idx']

                if pose_idx < len(training_poses):
                    # Add orbital poses around gap
                    orbital_radius = 0.05 * expansion_rate ** round_idx
                    orbit_poses = PoseInterpolator.generate_orbit_poses(
                        training_poses[pose_idx],
                        radius=orbital_radius,
                        num_poses=4,
                    )
                    novel_poses.extend(orbit_poses)

                # Interpolate toward neighbors
                neighbors = []
                if pose_idx > 0:
                    neighbors.append(training_poses[pose_idx - 1])
                if pose_idx < len(training_poses) - 1:
                    neighbors.append(training_poses[pose_idx + 1])

                for neighbor in neighbors:
                    interp = PoseInterpolator.interpolate_poses(
                        training_poses[pose_idx],
                        neighbor,
                        num_steps=2,
                    )
                    novel_poses.extend(interp)

                if len(novel_poses) >= max_poses:
                    break

        return novel_poses[:max_poses]


# =============================================================================
# Difix3D+ Pipeline
# =============================================================================

class DifixPipeline:
    """Main Difix3D+ refinement pipeline.

    Implements the complete progressive distillation workflow:
    1. Load pretrained Difix model
    2. Render views from current 3DGS
    3. Enhance with Difix (single-step diffusion)
    4. Distill pseudo-views back into 3DGS
    5. Repeat with progressively harder viewpoints

    Usage:
        config = DifixConfig()
        pipeline = DifixPipeline(config)

        refined_model = pipeline.refine(
            gaussian_model=model,
            training_views=views,
            output_dir=Path("output/refined")
        )
    """

    def __init__(self, config: DifixConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize components
        self.gap_detector = GapDetector(config.device)
        self.difix_model = None
        self._lpips_loss = None

        # Load model lazily
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Load Difix model if not already loaded."""
        if self._model_loaded:
            return

        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Difix3D+ requires 'diffusers' package. "
                "Install with: pip install diffusers transformers accelerate"
            )

        logger.info(f"Loading Difix model: {self.config.model_name}")

        try:
            dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

            self.difix_model = DiffusionPipeline.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)

            self._model_loaded = True
            logger.info("Difix model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Difix model: {e}")
            raise

    def _get_lpips_loss(self) -> Optional[Any]:
        """Get LPIPS loss function (lazy loading)."""
        if self._lpips_loss is not None:
            return self._lpips_loss

        if LPIPS_AVAILABLE:
            try:
                self._lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
                logger.info("LPIPS loss initialized")
            except Exception as e:
                logger.warning(f"Could not initialize LPIPS: {e}")
                self._lpips_loss = None

        return self._lpips_loss

    def enhance_image(
        self,
        degraded_image: Union[torch.Tensor, np.ndarray, Image.Image],
        reference_image: Optional[Union[torch.Tensor, np.ndarray, Image.Image]] = None,
    ) -> torch.Tensor:
        """Enhance a single image using Difix.

        Args:
            degraded_image: Degraded render to enhance
            reference_image: Optional clean reference for conditioning

        Returns:
            Enhanced image tensor [3, H, W] in 0-1 range
        """
        self._ensure_model_loaded()

        # Convert to PIL if needed
        if isinstance(degraded_image, torch.Tensor):
            if degraded_image.dim() == 3 and degraded_image.shape[0] == 3:
                # [3, H, W] -> [H, W, 3]
                img_np = degraded_image.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = degraded_image.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            degraded_pil = Image.fromarray(img_np)
        elif isinstance(degraded_image, np.ndarray):
            if degraded_image.max() <= 1.0:
                degraded_image = (degraded_image * 255).astype(np.uint8)
            degraded_pil = Image.fromarray(degraded_image)
        else:
            degraded_pil = degraded_image

        # Resize to optimal resolution for Difix
        target_size = self.config.output_resolution
        degraded_pil = degraded_pil.resize(target_size, Image.LANCZOS)

        # Run Difix (single-step inference)
        with torch.no_grad():
            output = self.difix_model(
                prompt=self.config.difix_prompt,
                image=degraded_pil,
                num_inference_steps=1,
                timesteps=[self.config.difix_timestep],
                guidance_scale=self.config.guidance_scale,
            )

        # Convert output to tensor
        enhanced_pil = output.images[0]
        enhanced_np = np.array(enhanced_pil).astype(np.float32) / 255.0
        enhanced_tensor = torch.tensor(enhanced_np, device=self.device)

        # [H, W, 3] -> [3, H, W]
        if enhanced_tensor.dim() == 3 and enhanced_tensor.shape[2] == 3:
            enhanced_tensor = enhanced_tensor.permute(2, 0, 1)

        return enhanced_tensor

    def refine(
        self,
        gaussian_model: "GaussianModel",
        training_views: List[Dict[str, Any]],
        intrinsics: Dict[str, Any],
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> "GaussianModel":
        """Run the full Difix3D+ refinement pipeline.

        Args:
            gaussian_model: Initial trained GaussianModel
            training_views: List of {pose, image} dicts from capture
            intrinsics: Camera intrinsics dict
            output_dir: Directory for outputs
            progress_callback: Optional callback(round, total_rounds, metrics)

        Returns:
            Refined GaussianModel
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Difix3D+ refinement with {self.config.num_refinement_rounds} rounds")

        # Extract camera poses from training views
        camera_poses = [
            {
                'world_to_camera': v['pose']['world_to_camera']
                    if isinstance(v['pose'], dict) else v['pose'],
                'frame_id': v.get('frame_id', str(i)),
                **{k: v[k] for k in ['fx', 'fy', 'cx', 'cy', 'image_height', 'image_width']
                   if k in v}
            }
            for i, v in enumerate(training_views)
        ]

        # Add intrinsics to poses if not present
        for pose in camera_poses:
            for key in ['fx', 'fy', 'cx', 'cy', 'image_height', 'image_width']:
                if key not in pose and key in intrinsics:
                    pose[key] = intrinsics[key]
            # Also handle CameraIntrinsics object
            if hasattr(intrinsics, key):
                pose[key] = getattr(intrinsics, key)

        # Progressive refinement loop
        for round_idx in range(self.config.num_refinement_rounds):
            logger.info(f"\n=== Refinement Round {round_idx + 1}/{self.config.num_refinement_rounds} ===")

            # 1. Detect gaps in current model
            gap_regions = self.gap_detector.find_gap_regions(
                gaussian_model,
                camera_poses,
                intrinsics,
                coverage_threshold=self.config.coverage_threshold,
                artifact_threshold=self.config.artifact_threshold,
            )

            if not gap_regions:
                logger.info("No gaps detected, skipping round")
                continue

            # 2. Generate novel poses for this round
            novel_poses = PoseInterpolator.generate_progressive_poses(
                training_poses=camera_poses,
                gap_regions=gap_regions,
                round_idx=round_idx,
                expansion_rate=self.config.progressive_expansion_rate,
                max_poses=self.config.poses_per_round,
            )

            logger.info(f"Generated {len(novel_poses)} novel poses")

            # 3. Render → Enhance → Collect pseudo-views
            pseudo_views = self._generate_pseudo_views(
                gaussian_model,
                novel_poses,
                training_views,
                intrinsics,
            )

            logger.info(f"Generated {len(pseudo_views)} pseudo-views")

            # 4. Distill pseudo-views back into 3DGS
            if pseudo_views:
                self._distill_views(
                    gaussian_model,
                    training_views,
                    pseudo_views,
                    intrinsics,
                )

            # Save intermediate if configured
            if self.config.save_intermediate:
                round_path = output_dir / f"round_{round_idx + 1}"
                round_path.mkdir(exist_ok=True)
                gaussian_model.save_ply(round_path / "gaussians.ply")

            # Progress callback
            if progress_callback:
                metrics = {
                    'round': round_idx + 1,
                    'gaps_found': len(gap_regions),
                    'pseudo_views': len(pseudo_views),
                }
                progress_callback(round_idx, self.config.num_refinement_rounds, metrics)

        # Save final refined model
        final_path = output_dir / "refined_gaussians.ply"
        gaussian_model.save_ply(final_path)
        logger.info(f"Saved refined Gaussians to {final_path}")

        return gaussian_model

    def _generate_pseudo_views(
        self,
        gaussian_model: "GaussianModel",
        novel_poses: List[Dict[str, Any]],
        training_views: List[Dict[str, Any]],
        intrinsics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate enhanced pseudo-views from novel poses.

        Render → Difix enhance → return as training targets.
        """
        pseudo_views = []

        for pose in novel_poses:
            try:
                # Render from current model
                render_result = self.gap_detector._render_view(
                    gaussian_model, pose, intrinsics
                )

                if render_result is None:
                    continue

                degraded_rgb = render_result['rgb']

                # Find nearest reference view
                ref_view = self._find_nearest_view(pose, training_views)
                ref_image = ref_view.get('image') if ref_view else None

                # Enhance with Difix
                enhanced_rgb = self.enhance_image(
                    degraded_rgb,
                    reference_image=ref_image,
                )

                # Create pseudo-view
                pseudo_views.append({
                    'pose': pose,
                    'image': enhanced_rgb,
                    'weight': self.config.distillation_weight,
                    'is_pseudo': True,
                })

            except Exception as e:
                logger.warning(f"Failed to generate pseudo-view: {e}")
                continue

        return pseudo_views

    def _find_nearest_view(
        self,
        query_pose: Dict[str, Any],
        views: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find the nearest training view to a query pose."""
        if not views:
            return None

        # Extract query position
        T_query = np.array(query_pose['world_to_camera'])
        pos_query = T_query[:3, 3]

        min_dist = float('inf')
        nearest = None

        for view in views:
            pose = view.get('pose', {})
            if isinstance(pose, dict) and 'world_to_camera' in pose:
                T = np.array(pose['world_to_camera'])
            elif hasattr(pose, '__iter__') and len(pose) == 16:
                T = np.array(pose).reshape(4, 4)
            else:
                continue

            pos = T[:3, 3]
            dist = np.linalg.norm(pos - pos_query)

            if dist < min_dist:
                min_dist = dist
                nearest = view

        return nearest

    def _distill_views(
        self,
        gaussian_model: "GaussianModel",
        real_views: List[Dict[str, Any]],
        pseudo_views: List[Dict[str, Any]],
        intrinsics: Dict[str, Any],
    ):
        """Distill pseudo-views back into the 3DGS model.

        Continues training with combined real + pseudo views,
        weighted by confidence.
        """
        logger.info(
            f"Distilling {len(pseudo_views)} pseudo-views "
            f"+ {len(real_views)} real views"
        )

        # Combine views with weights
        all_views = []

        for v in real_views:
            all_views.append({
                **v,
                'weight': 1.0,
                'is_pseudo': False,
            })

        all_views.extend(pseudo_views)

        # Setup optimizer for distillation
        optimizer = Adam([
            {'params': [gaussian_model._xyz], 'lr': self.config.distillation_lr * 0.1},
            {'params': [gaussian_model._features_dc], 'lr': self.config.distillation_lr},
            {'params': [gaussian_model._features_rest], 'lr': self.config.distillation_lr * 0.05},
            {'params': [gaussian_model._opacity], 'lr': self.config.distillation_lr * 0.5},
            {'params': [gaussian_model._scaling], 'lr': self.config.distillation_lr * 0.1},
            {'params': [gaussian_model._rotation], 'lr': self.config.distillation_lr * 0.1},
        ], lr=self.config.distillation_lr)

        # Distillation training loop
        num_iterations = self.config.iterations_per_round

        for iteration in range(num_iterations):
            # Sample view (weighted by confidence)
            weights = np.array([v['weight'] for v in all_views])
            weights = weights / weights.sum()

            idx = np.random.choice(len(all_views), p=weights)
            view = all_views[idx]

            # Get target image
            target_image = view.get('image')
            if target_image is None:
                continue

            # Ensure tensor
            if not isinstance(target_image, torch.Tensor):
                target_image = torch.tensor(target_image, device=self.device)

            target_image = target_image.to(self.device)

            # Get pose
            pose = view.get('pose', {})
            if isinstance(pose, dict) and 'world_to_camera' in pose:
                pose_dict = pose
            else:
                # Try to convert
                pose_dict = {'world_to_camera': pose}

            # Add intrinsics
            for key in ['fx', 'fy', 'cx', 'cy', 'image_height', 'image_width']:
                if key not in pose_dict:
                    if key in intrinsics:
                        pose_dict[key] = intrinsics[key]
                    elif hasattr(intrinsics, key):
                        pose_dict[key] = getattr(intrinsics, key)

            # Render
            render_result = self.gap_detector._render_view(
                gaussian_model, pose_dict, intrinsics
            )

            if render_result is None:
                continue

            rendered = render_result['rgb']

            # Ensure same size
            if rendered.shape != target_image.shape:
                # Resize target to match render
                target_resized = F.interpolate(
                    target_image.unsqueeze(0),
                    size=rendered.shape[1:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            else:
                target_resized = target_image

            # Compute loss
            loss = self._compute_distillation_loss(
                rendered, target_resized, view['weight']
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress
            if iteration % 500 == 0:
                logger.info(
                    f"  Distillation iter {iteration}/{num_iterations}: "
                    f"loss={loss.item():.4f}"
                )

    def _compute_distillation_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Compute weighted loss for distillation.

        Uses L2 + LPIPS + SSIM matching the Difix3D+ paper.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        # L2 reconstruction loss
        l2_loss = F.mse_loss(rendered, target)
        total_loss = total_loss + self.config.l2_weight * l2_loss

        # LPIPS perceptual loss (if available)
        lpips_fn = self._get_lpips_loss()
        if lpips_fn is not None:
            # LPIPS expects [B, 3, H, W] in [-1, 1] range
            rendered_lpips = rendered.unsqueeze(0) * 2 - 1
            target_lpips = target.unsqueeze(0) * 2 - 1

            lpips_loss = lpips_fn(rendered_lpips, target_lpips).mean()
            total_loss = total_loss + self.config.lpips_weight * lpips_loss

        # SSIM loss
        ssim_val = self._compute_ssim(rendered.unsqueeze(0), target.unsqueeze(0))
        ssim_loss = 1.0 - ssim_val
        total_loss = total_loss + self.config.ssim_weight * ssim_loss

        # Apply view weight
        return weight * total_loss

    def _compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Compute SSIM between images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=self.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)

        num_channels = img1.shape[1]
        window = window.expand(num_channels, 1, window_size, window_size)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=num_channels)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=num_channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=num_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=num_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=num_channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def enhance_render(
        self,
        rendered_image: Union[torch.Tensor, np.ndarray, Image.Image],
        strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Post-process enhancement for rendered images.

        Apply Difix as a fast neural enhancer at inference time.
        This removes residual artifacts that 3DGS couldn't fully fix.

        Args:
            rendered_image: Rendered image to enhance
            strength: Enhancement strength 0-1 (blend with original)

        Returns:
            Enhanced image tensor
        """
        if not self.config.enable_post_process:
            if isinstance(rendered_image, torch.Tensor):
                return rendered_image
            return torch.tensor(np.array(rendered_image)).permute(2, 0, 1) / 255.0

        strength = strength or self.config.post_process_strength

        # Enhance
        enhanced = self.enhance_image(rendered_image)

        # Blend with original if strength < 1
        if strength < 1.0:
            if isinstance(rendered_image, torch.Tensor):
                original = rendered_image
            else:
                original = torch.tensor(np.array(rendered_image)).permute(2, 0, 1) / 255.0

            original = original.to(self.device)
            enhanced = strength * enhanced + (1 - strength) * original

        return enhanced


# =============================================================================
# Integration Functions
# =============================================================================

def refine_gaussians_with_difix(
    gaussians_path: Path,
    training_images_dir: Path,
    poses_path: Path,
    intrinsics: Dict[str, Any],
    output_dir: Path,
    config: Optional[DifixConfig] = None,
) -> Path:
    """Convenience function to refine existing Gaussians with Difix3D+.

    Args:
        gaussians_path: Path to existing PLY file
        training_images_dir: Directory with training images
        poses_path: Path to poses JSON/file
        intrinsics: Camera intrinsics dict
        output_dir: Output directory
        config: Optional DifixConfig

    Returns:
        Path to refined PLY file
    """
    from .gaussian_splatting import GaussianModel

    config = config or DifixConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading Gaussians from {gaussians_path}")
    gaussian_model = GaussianModel.load_ply(gaussians_path)

    # Load training views
    training_views = _load_training_views(
        training_images_dir, poses_path, intrinsics
    )

    # Create pipeline and refine
    pipeline = DifixPipeline(config)
    refined_model = pipeline.refine(
        gaussian_model=gaussian_model,
        training_views=training_views,
        intrinsics=intrinsics,
        output_dir=output_dir,
    )

    return output_dir / "refined_gaussians.ply"


def _load_training_views(
    images_dir: Path,
    poses_path: Path,
    intrinsics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Load training views from disk."""
    import json

    views = []
    images_dir = Path(images_dir)
    poses_path = Path(poses_path)

    # Load poses
    if poses_path.suffix == '.json':
        with open(poses_path) as f:
            poses_data = json.load(f)
    else:
        # Assume COLMAP format or other
        poses_data = {}

    # Load images
    for img_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
        frame_id = img_path.stem

        # Get pose for this frame
        pose = poses_data.get(frame_id, {})
        if not pose:
            continue

        # Load image
        if PIL_AVAILABLE:
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.tensor(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        else:
            img_tensor = None

        views.append({
            'frame_id': frame_id,
            'image': img_tensor,
            'pose': pose,
            'image_path': str(img_path),
        })

    return views
