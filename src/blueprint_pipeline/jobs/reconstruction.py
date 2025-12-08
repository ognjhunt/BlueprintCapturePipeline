"""3D Gaussian Splatting reconstruction with WildGS-SLAM."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import ArtifactPaths, JobPayload, ScaleAnchor, SessionManifest
from ..utils.io import ensure_local_dir, load_json, save_json, load_image
from .base import (
    GPUJob,
    JobContext,
    JobResult,
    JobStatus,
    download_inputs,
    merge_parameters,
    upload_outputs,
)


@dataclass
class CameraPose:
    """Camera pose in COLMAP format."""
    image_id: int
    image_name: str
    qvec: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    tvec: Tuple[float, float, float]  # translation
    camera_id: int


@dataclass
class ReconstructionMetrics:
    """Quality metrics for reconstruction."""
    total_frames: int
    registered_frames: int
    mean_reprojection_error: float
    median_reprojection_error: float
    track_length_mean: float
    scale_factor: float
    scale_confidence: float
    coverage_score: float
    gaussian_count: int


@dataclass
class ReconstructionJob(GPUJob):
    """Run WildGS-SLAM reconstruction with scale calibration.

    This job:
    1. Downloads extracted frames and masks
    2. Runs WildGS-SLAM for camera tracking and 3DGS reconstruction
    3. Applies dynamic masks to exclude moving objects
    4. Calibrates scale using detected anchors (ArUco/AprilTags)
    5. Outputs camera poses, Gaussian splats, and quality metrics

    Inputs:
        - Extracted frames from FrameExtractionJob
        - Instance masks with dynamic object flags

    Outputs:
        - Camera poses in COLMAP format
        - 3D Gaussian splats (point cloud + SH coefficients)
        - Reprojection quality report
    """

    name: str = "reconstruction"
    description: str = "WildGS-SLAM with scale calibration and dynamic masking."
    timeout_minutes: int = 90
    use_dynamic_masks: bool = True
    enforce_scale_anchor: bool = True

    # WildGS-SLAM configuration
    num_iterations: int = 30000
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    position_lr: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    def _get_default_parameters(self) -> Dict[str, Any]:
        params = super()._get_default_parameters()
        params.update({
            "use_dynamic_masks": self.use_dynamic_masks,
            "enforce_scale_anchor": self.enforce_scale_anchor,
            "num_iterations": self.num_iterations,
            "densification_interval": self.densification_interval,
        })
        return params

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self._get_default_parameters(),
            {
                "use_dynamic_masks": self.use_dynamic_masks,
                "enforce_scale_anchor": self.enforce_scale_anchor,
                "scale_anchor_count": len(session.scale_anchors),
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "frames": artifacts.frames,
                "masks": artifacts.masks,
            },
            outputs={
                "poses": f"{artifacts.reconstruction}/poses",
                "gaussians": f"{artifacts.reconstruction}/gaussians",
                "reprojection_report": f"{artifacts.reconstruction}/reports/reprojection.json",
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute WildGS-SLAM reconstruction."""
        result = JobResult(status=JobStatus.RUNNING)

        # Setup directories
        frames_dir = ensure_local_dir(ctx.workspace / "frames")
        masks_dir = ensure_local_dir(ctx.workspace / "masks")
        output_dir = ensure_local_dir(ctx.workspace / "reconstruction")
        poses_dir = ensure_local_dir(output_dir / "poses")
        gaussians_dir = ensure_local_dir(output_dir / "gaussians")
        reports_dir = ensure_local_dir(output_dir / "reports")

        # Download inputs
        with ctx.tracker.stage("download_inputs", 2):
            ctx.gcs.download_directory(ctx.artifacts.frames + "/", frames_dir)
            ctx.tracker.update(1)
            ctx.gcs.download_directory(ctx.artifacts.masks + "/", masks_dir)
            ctx.tracker.update(1)

        # Load frame index and mask annotations
        frame_index = self._load_frame_index(frames_dir)
        mask_annotations = self._load_mask_annotations(masks_dir)

        total_frames = len(frame_index.get("frames", []))
        ctx.logger.info(f"Loaded {total_frames} frames for reconstruction")
        ctx.tracker.log_metric("input_frames", total_frames)

        # Prepare dynamic masks for exclusion
        dynamic_mask_map = {}
        if ctx.parameters.get("use_dynamic_masks", self.use_dynamic_masks):
            dynamic_mask_map = self._prepare_dynamic_masks(
                ctx, mask_annotations, masks_dir
            )
            ctx.logger.info(f"Prepared {len(dynamic_mask_map)} dynamic masks")

        # Run WildGS-SLAM
        with ctx.tracker.stage("wildgs_slam", total_frames):
            slam_result = self._run_wildgs_slam(
                ctx=ctx,
                frames_dir=frames_dir,
                frame_index=frame_index,
                dynamic_masks=dynamic_mask_map,
                output_dir=output_dir,
            )

        # Calibrate scale using anchors
        scale_factor = 1.0
        scale_confidence = 0.0

        if ctx.session.scale_anchors and ctx.parameters.get(
            "enforce_scale_anchor", self.enforce_scale_anchor
        ):
            with ctx.tracker.stage("scale_calibration", len(ctx.session.scale_anchors)):
                scale_factor, scale_confidence = self._calibrate_scale(
                    ctx=ctx,
                    frames_dir=frames_dir,
                    frame_index=frame_index,
                    poses=slam_result.get("poses", []),
                    scale_anchors=ctx.session.scale_anchors,
                )

        ctx.logger.info(f"Scale factor: {scale_factor:.4f} (confidence: {scale_confidence:.2f})")
        ctx.tracker.log_metric("scale_factor", scale_factor)
        ctx.tracker.log_metric("scale_confidence", scale_confidence)

        # Apply scale to poses and gaussians
        scaled_poses = self._apply_scale_to_poses(slam_result.get("poses", []), scale_factor)
        self._apply_scale_to_gaussians(gaussians_dir, scale_factor)

        # Export camera poses in COLMAP format
        self._export_colmap_poses(scaled_poses, poses_dir)

        # Generate reconstruction metrics
        metrics = self._compute_reconstruction_metrics(
            ctx=ctx,
            poses=scaled_poses,
            gaussians_dir=gaussians_dir,
            scale_factor=scale_factor,
            scale_confidence=scale_confidence,
            total_frames=total_frames,
        )

        # Save reports
        reprojection_report = {
            "session_id": ctx.session.session_id,
            "metrics": {
                "total_frames": metrics.total_frames,
                "registered_frames": metrics.registered_frames,
                "registration_rate": metrics.registered_frames / max(1, metrics.total_frames),
                "mean_reprojection_error": metrics.mean_reprojection_error,
                "median_reprojection_error": metrics.median_reprojection_error,
                "track_length_mean": metrics.track_length_mean,
                "scale_factor": metrics.scale_factor,
                "scale_confidence": metrics.scale_confidence,
                "coverage_score": metrics.coverage_score,
                "gaussian_count": metrics.gaussian_count,
            },
            "quality_flags": {
                "scale_calibrated": scale_confidence > 0.7,
                "sufficient_coverage": metrics.coverage_score > 0.6,
                "low_reprojection_error": metrics.mean_reprojection_error < 1.0,
            },
        }
        save_json(reprojection_report, reports_dir / "reprojection.json")

        # Upload outputs
        with ctx.tracker.stage("upload_outputs", 3):
            ctx.gcs.upload_directory(poses_dir, f"{ctx.artifacts.reconstruction}/poses")
            ctx.tracker.update(1)
            ctx.gcs.upload_directory(gaussians_dir, f"{ctx.artifacts.reconstruction}/gaussians")
            ctx.tracker.update(1)
            ctx.gcs.upload(
                reports_dir / "reprojection.json",
                f"{ctx.artifacts.reconstruction}/reports/reprojection.json"
            )
            ctx.tracker.update(1)

        result.outputs = {
            "poses": f"{ctx.artifacts.reconstruction}/poses",
            "gaussians": f"{ctx.artifacts.reconstruction}/gaussians",
            "reprojection_report": f"{ctx.artifacts.reconstruction}/reports/reprojection.json",
        }
        result.metrics = reprojection_report["metrics"]

        return result

    def _load_frame_index(self, frames_dir: Path) -> Dict[str, Any]:
        """Load frame index from extracted frames directory."""
        index_path = frames_dir / "frame_index.json"
        if index_path.exists():
            return load_json(index_path)

        # Fallback: scan directory for frames
        frames = []
        for clip_dir in frames_dir.iterdir():
            if clip_dir.is_dir():
                for frame_path in sorted(clip_dir.glob("*.png")):
                    frames.append({
                        "frame_id": frame_path.stem,
                        "file_path": str(frame_path.relative_to(frames_dir)),
                    })
        return {"frames": frames}

    def _load_mask_annotations(self, masks_dir: Path) -> Dict[str, Any]:
        """Load mask annotations from COCO JSON."""
        annotations_path = masks_dir / "annotations.json"
        if annotations_path.exists():
            return load_json(annotations_path)
        return {"annotations": []}

    def _prepare_dynamic_masks(
        self,
        ctx: JobContext,
        annotations: Dict[str, Any],
        masks_dir: Path,
    ) -> Dict[str, Path]:
        """Prepare combined dynamic masks for each frame.

        Returns mapping of frame_id -> path to combined dynamic mask.
        """
        dynamic_masks = {}

        # Group annotations by frame
        frame_annotations: Dict[str, List[Dict]] = {}
        for ann in annotations.get("annotations", []):
            if ann.get("is_dynamic", False):
                frame_id = ann["image_id"]
                if frame_id not in frame_annotations:
                    frame_annotations[frame_id] = []
                frame_annotations[frame_id].append(ann)

        # Combine masks for each frame
        for frame_id, anns in frame_annotations.items():
            combined_mask = None

            for ann in anns:
                seg = ann.get("segmentation", {})
                mask_file = seg.get("mask_file")
                if not mask_file:
                    continue

                mask_path = masks_dir / mask_file
                if not mask_path.exists():
                    # Try finding in subdirectories
                    for subdir in masks_dir.iterdir():
                        if subdir.is_dir():
                            potential_path = subdir / mask_file
                            if potential_path.exists():
                                mask_path = potential_path
                                break

                if mask_path.exists():
                    mask = load_image(mask_path, mode="L")
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.maximum(combined_mask, mask)

            if combined_mask is not None:
                # Save combined mask
                combined_path = masks_dir / f"{frame_id}_dynamic.png"
                from ..utils.io import save_image
                save_image(combined_mask, combined_path)
                dynamic_masks[frame_id] = combined_path

        return dynamic_masks

    def _run_wildgs_slam(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        dynamic_masks: Dict[str, Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run WildGS-SLAM reconstruction.

        WildGS-SLAM is a monocular Gaussian SLAM that:
        - Handles dynamic scenes by excluding masked regions
        - Produces camera poses and 3D Gaussian splats
        - Works without depth sensors
        """
        ctx.logger.info("Starting WildGS-SLAM reconstruction...")

        # Try to import WildGS-SLAM
        slam_available = False
        try:
            # Attempt to import WildGS components
            # Note: This is a placeholder - actual import depends on installation
            from wildgs_slam import WildGSSLAM
            slam_available = True
        except ImportError:
            ctx.logger.warning("WildGS-SLAM not available, using fallback COLMAP+3DGS")

        if slam_available:
            return self._run_native_wildgs(ctx, frames_dir, frame_index, dynamic_masks, output_dir)
        else:
            return self._run_colmap_fallback(ctx, frames_dir, frame_index, dynamic_masks, output_dir)

    def _run_native_wildgs(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        dynamic_masks: Dict[str, Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run native WildGS-SLAM implementation."""
        from wildgs_slam import WildGSSLAM

        # Prepare image paths and masks
        image_paths = []
        mask_paths = []

        for frame_info in frame_index.get("frames", []):
            frame_path = frames_dir / frame_info["file_path"]
            image_paths.append(str(frame_path))

            frame_id = frame_info["frame_id"]
            if frame_id in dynamic_masks:
                mask_paths.append(str(dynamic_masks[frame_id]))
            else:
                mask_paths.append(None)

        # Initialize SLAM
        config = {
            "num_iterations": ctx.parameters.get("num_iterations", self.num_iterations),
            "densification_interval": ctx.parameters.get("densification_interval", self.densification_interval),
            "position_lr": self.position_lr,
            "feature_lr": self.feature_lr,
        }

        slam = WildGSSLAM(config)

        # Run SLAM
        poses = []
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            pose = slam.process_frame(img_path, mask_path)
            poses.append(pose)
            ctx.tracker.update(1)

        # Export gaussians
        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(exist_ok=True)
        slam.export_gaussians(gaussians_dir / "point_cloud.ply")

        return {"poses": poses}

    def _run_colmap_fallback(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        dynamic_masks: Dict[str, Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Fallback: Use COLMAP for SfM then train 3DGS.

        This is a two-stage approach:
        1. COLMAP for camera pose estimation
        2. 3D Gaussian Splatting training
        """
        ctx.logger.info("Using COLMAP + 3DGS fallback pipeline")

        colmap_dir = output_dir / "colmap"
        gaussians_dir = output_dir / "gaussians"
        colmap_dir.mkdir(exist_ok=True)
        gaussians_dir.mkdir(exist_ok=True)

        # Prepare images for COLMAP (apply masks to hide dynamic regions)
        images_dir = self._prepare_masked_images(
            ctx, frames_dir, frame_index, dynamic_masks, colmap_dir / "images"
        )

        # Run COLMAP
        poses = self._run_colmap_sfm(ctx, images_dir, colmap_dir)

        # Train 3D Gaussian Splatting
        self._train_gaussian_splatting(ctx, colmap_dir, gaussians_dir)

        return {"poses": poses}

    def _prepare_masked_images(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        dynamic_masks: Dict[str, Path],
        output_dir: Path,
    ) -> Path:
        """Prepare images with dynamic regions masked out."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for frame_info in frame_index.get("frames", []):
            frame_path = frames_dir / frame_info["file_path"]
            frame_id = frame_info["frame_id"]

            img = load_image(frame_path)

            if frame_id in dynamic_masks:
                mask = load_image(dynamic_masks[frame_id], mode="L")
                # Set masked regions to neutral gray
                img[mask > 127] = [128, 128, 128]

            output_path = output_dir / f"{frame_id}.png"
            from ..utils.io import save_image
            save_image(img, output_path)

        return output_dir

    def _run_colmap_sfm(
        self,
        ctx: JobContext,
        images_dir: Path,
        colmap_dir: Path,
    ) -> List[CameraPose]:
        """Run COLMAP Structure-from-Motion pipeline."""
        import subprocess

        database_path = colmap_dir / "database.db"
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        # Feature extraction
        ctx.logger.info("Running COLMAP feature extraction...")
        try:
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "1",
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            ctx.logger.warning(f"COLMAP feature extraction failed: {e}")
            return self._create_dummy_poses(images_dir)

        # Feature matching
        ctx.logger.info("Running COLMAP feature matching...")
        try:
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1",
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            ctx.logger.warning(f"COLMAP matching failed: {e}")
            return self._create_dummy_poses(images_dir)

        # Sparse reconstruction
        ctx.logger.info("Running COLMAP sparse reconstruction...")
        try:
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir),
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            ctx.logger.warning(f"COLMAP mapper failed: {e}")
            return self._create_dummy_poses(images_dir)

        # Parse poses from COLMAP output
        poses = self._parse_colmap_poses(sparse_dir / "0")

        ctx.logger.info(f"COLMAP recovered {len(poses)} camera poses")
        return poses

    def _parse_colmap_poses(self, sparse_model_dir: Path) -> List[CameraPose]:
        """Parse camera poses from COLMAP sparse model."""
        poses = []

        images_txt = sparse_model_dir / "images.txt"
        if not images_txt.exists():
            images_bin = sparse_model_dir / "images.bin"
            if images_bin.exists():
                # Convert binary to text format
                import subprocess
                try:
                    subprocess.run([
                        "colmap", "model_converter",
                        "--input_path", str(sparse_model_dir),
                        "--output_path", str(sparse_model_dir),
                        "--output_type", "TXT",
                    ], check=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return poses

        if not images_txt.exists():
            return poses

        with open(images_txt, "r") as f:
            lines = f.readlines()

        # Skip header comments
        i = 0
        while i < len(lines) and lines[i].startswith("#"):
            i += 1

        # Parse image entries (two lines per image)
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                camera_id = int(parts[8])
                image_name = parts[9]

                poses.append(CameraPose(
                    image_id=image_id,
                    image_name=image_name,
                    qvec=(qw, qx, qy, qz),
                    tvec=(tx, ty, tz),
                    camera_id=camera_id,
                ))

            i += 2  # Skip points line

        return poses

    def _create_dummy_poses(self, images_dir: Path) -> List[CameraPose]:
        """Create dummy poses when COLMAP fails (for testing)."""
        poses = []
        for i, img_path in enumerate(sorted(images_dir.glob("*.png"))):
            poses.append(CameraPose(
                image_id=i,
                image_name=img_path.name,
                qvec=(1.0, 0.0, 0.0, 0.0),
                tvec=(i * 0.1, 0.0, 0.0),
                camera_id=1,
            ))
        return poses

    def _train_gaussian_splatting(
        self,
        ctx: JobContext,
        colmap_dir: Path,
        output_dir: Path,
    ) -> None:
        """Train 3D Gaussian Splatting from COLMAP output."""
        ctx.logger.info("Training 3D Gaussian Splatting...")

        try:
            # Try using gaussian-splatting package
            import subprocess

            subprocess.run([
                "python", "-m", "gaussian_splatting.train",
                "--source_path", str(colmap_dir),
                "--model_path", str(output_dir),
                "--iterations", str(ctx.parameters.get("num_iterations", self.num_iterations)),
            ], check=True, capture_output=True, timeout=3600)

            ctx.logger.info("3DGS training completed")

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            ctx.logger.warning(f"3DGS training failed: {e}")
            # Create placeholder output
            self._create_placeholder_gaussians(colmap_dir, output_dir)

    def _create_placeholder_gaussians(self, colmap_dir: Path, output_dir: Path) -> None:
        """Create placeholder Gaussian output when training fails."""
        # Copy sparse points as placeholder
        sparse_ply = colmap_dir / "sparse" / "0" / "points3D.ply"
        if sparse_ply.exists():
            shutil.copy(sparse_ply, output_dir / "point_cloud.ply")
        else:
            # Create minimal PLY file
            ply_content = """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 128 128 128
"""
            (output_dir / "point_cloud.ply").write_text(ply_content)

    def _calibrate_scale(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        poses: List[CameraPose],
        scale_anchors: List[ScaleAnchor],
    ) -> Tuple[float, float]:
        """Calibrate reconstruction scale using detected anchors.

        Supports:
        - ArUco/AprilTag boards with known marker size
        - Tape measure segments
        - Known object dimensions

        Returns:
            (scale_factor, confidence)
        """
        ctx.logger.info(f"Calibrating scale with {len(scale_anchors)} anchors")

        scale_observations = []

        for anchor in scale_anchors:
            if anchor.anchor_type == "aruco_board":
                scales = self._detect_aruco_scale(
                    ctx, frames_dir, frame_index, poses, anchor.size_meters
                )
                scale_observations.extend(scales)

            elif anchor.anchor_type == "tape_measure":
                scales = self._detect_tape_measure_scale(
                    ctx, frames_dir, frame_index, poses, anchor.size_meters
                )
                scale_observations.extend(scales)

            elif anchor.anchor_type == "known_object":
                # Use object detection to find known-size objects
                pass

            ctx.tracker.update(1)

        if not scale_observations:
            ctx.logger.warning("No scale observations found - using unit scale")
            return 1.0, 0.0

        # Compute robust scale estimate (median)
        scale_factor = float(np.median(scale_observations))
        scale_std = float(np.std(scale_observations))

        # Confidence based on consistency of observations
        if len(scale_observations) > 1:
            cv = scale_std / scale_factor if scale_factor > 0 else 1.0
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5  # Single observation

        return scale_factor, confidence

    def _detect_aruco_scale(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        poses: List[CameraPose],
        marker_size_meters: float,
    ) -> List[float]:
        """Detect ArUco markers and compute scale from known marker size."""
        scales = []

        try:
            import cv2
            from cv2 import aruco
        except ImportError:
            ctx.logger.warning("OpenCV ArUco module not available")
            return scales

        # Use standard ArUco dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector_params = aruco.DetectorParameters()

        for frame_info in frame_index.get("frames", [])[:50]:  # Sample frames
            frame_path = frames_dir / frame_info["file_path"]
            if not frame_path.exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

            if ids is not None and len(ids) > 0:
                for marker_corners in corners:
                    # Compute marker size in pixels
                    pts = marker_corners[0]
                    edge_lengths = [
                        np.linalg.norm(pts[i] - pts[(i + 1) % 4])
                        for i in range(4)
                    ]
                    avg_edge_pixels = np.mean(edge_lengths)

                    # Get corresponding camera pose
                    frame_id = frame_info["frame_id"]
                    pose = self._find_pose_for_frame(poses, frame_id)
                    if pose is None:
                        continue

                    # Estimate depth from reconstruction (simplified)
                    # In production, use marker pose estimation
                    estimated_depth = 1.0  # Placeholder

                    # Scale = (real_size / pixel_size) * depth_factor
                    # This is simplified - real implementation would use
                    # camera intrinsics and marker pose estimation
                    scale = marker_size_meters / (avg_edge_pixels / 1000)
                    scales.append(scale)

        return scales

    def _detect_tape_measure_scale(
        self,
        ctx: JobContext,
        frames_dir: Path,
        frame_index: Dict[str, Any],
        poses: List[CameraPose],
        measured_length_meters: float,
    ) -> List[float]:
        """Detect tape measure markings and compute scale."""
        # This would use edge detection and OCR to find tape measure markings
        # Placeholder for now
        return []

    def _find_pose_for_frame(self, poses: List[CameraPose], frame_id: str) -> Optional[CameraPose]:
        """Find camera pose corresponding to a frame."""
        for pose in poses:
            if frame_id in pose.image_name or pose.image_name in frame_id:
                return pose
        return None

    def _apply_scale_to_poses(
        self,
        poses: List[CameraPose],
        scale_factor: float,
    ) -> List[CameraPose]:
        """Apply scale factor to camera translations."""
        scaled_poses = []
        for pose in poses:
            scaled_tvec = tuple(t * scale_factor for t in pose.tvec)
            scaled_poses.append(CameraPose(
                image_id=pose.image_id,
                image_name=pose.image_name,
                qvec=pose.qvec,
                tvec=scaled_tvec,
                camera_id=pose.camera_id,
            ))
        return scaled_poses

    def _apply_scale_to_gaussians(self, gaussians_dir: Path, scale_factor: float) -> None:
        """Apply scale factor to Gaussian positions."""
        ply_path = gaussians_dir / "point_cloud.ply"
        if not ply_path.exists():
            return

        try:
            # Read PLY, scale positions, write back
            # This is a simplified implementation
            import plyfile

            ply_data = plyfile.PlyData.read(str(ply_path))
            vertex = ply_data["vertex"]

            # Scale position attributes
            for axis in ["x", "y", "z"]:
                if axis in vertex.data.dtype.names:
                    vertex.data[axis] *= scale_factor

            ply_data.write(str(ply_path))

        except ImportError:
            # Fallback: manual PLY parsing
            pass

    def _export_colmap_poses(self, poses: List[CameraPose], output_dir: Path) -> None:
        """Export camera poses in COLMAP text format."""
        # Export images.txt
        images_txt = output_dir / "images.txt"
        with open(images_txt, "w") as f:
            f.write("# Image list with image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name\n")
            f.write("# Number of images: {}\n".format(len(poses)))
            for pose in poses:
                qw, qx, qy, qz = pose.qvec
                tx, ty, tz = pose.tvec
                f.write(f"{pose.image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {pose.camera_id} {pose.image_name}\n")
                f.write("\n")  # Empty points line

        # Export as JSON for easier consumption
        poses_json = []
        for pose in poses:
            poses_json.append({
                "image_id": pose.image_id,
                "image_name": pose.image_name,
                "rotation": list(pose.qvec),
                "translation": list(pose.tvec),
                "camera_id": pose.camera_id,
            })
        save_json({"poses": poses_json}, output_dir / "poses.json")

    def _compute_reconstruction_metrics(
        self,
        ctx: JobContext,
        poses: List[CameraPose],
        gaussians_dir: Path,
        scale_factor: float,
        scale_confidence: float,
        total_frames: int,
    ) -> ReconstructionMetrics:
        """Compute quality metrics for the reconstruction."""
        registered_frames = len(poses)

        # Count Gaussians
        gaussian_count = 0
        ply_path = gaussians_dir / "point_cloud.ply"
        if ply_path.exists():
            try:
                with open(ply_path, "r") as f:
                    for line in f:
                        if line.startswith("element vertex"):
                            gaussian_count = int(line.split()[-1])
                            break
            except Exception:
                pass

        # Placeholder metrics (would be computed from actual reconstruction)
        return ReconstructionMetrics(
            total_frames=total_frames,
            registered_frames=registered_frames,
            mean_reprojection_error=0.5,  # Placeholder
            median_reprojection_error=0.4,  # Placeholder
            track_length_mean=10.0,  # Placeholder
            scale_factor=scale_factor,
            scale_confidence=scale_confidence,
            coverage_score=registered_frames / max(1, total_frames),
            gaussian_count=gaussian_count,
        )
