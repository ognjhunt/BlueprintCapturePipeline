"""Object assetization: SAM3 lifting and Hunyuan3D generation."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import ArtifactPaths, JobPayload, SessionManifest
from ..utils.io import ensure_local_dir, load_json, save_json, load_image, save_image
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
class ObjectTrack:
    """Tracked object across multiple frames."""
    track_id: str
    category: str
    frame_ids: List[str]
    bboxes: List[Tuple[int, int, int, int]]  # List of (x, y, w, h) per frame
    masks: List[str]  # Paths to mask files
    confidences: List[float]
    is_dynamic: bool
    coverage_score: float = 0.0
    viewpoint_diversity: float = 0.0


@dataclass
class ObjectAsset:
    """Generated 3D object asset."""
    asset_id: str
    track_id: str
    category: str
    source: str  # "reconstruction" or "generation"
    mesh_path: str
    texture_path: Optional[str]
    bounding_box: Tuple[float, float, float, float, float, float]  # min_x, min_y, min_z, max_x, max_y, max_z
    position_in_scene: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # quaternion
    scale: Tuple[float, float, float]
    quality_score: float


@dataclass
class ObjectAssetizationJob(GPUJob):
    """Lift SAM3 tracks into 3D and generate object assets.

    This job:
    1. Loads SAM3 segmentation tracks from frame extraction
    2. Lifts 2D masks into 3D using camera poses
    3. Computes viewpoint coverage for each object
    4. For objects with good coverage: reconstruct from multi-view
    5. For objects with poor coverage: generate with Hunyuan3D
    6. Outputs individual object USD files

    Inputs:
        - Frames from FrameExtractionJob
        - Masks and annotations from FrameExtractionJob
        - Camera poses from ReconstructionJob
        - Environment mesh from MeshExtractionJob

    Outputs:
        - Individual object USD files
        - Object placement report
    """

    name: str = "object-assetization"
    description: str = (
        "Lift SAM 3 tracks into 3D, reconstruct objects when coverage exists, "
        "and fall back to Hunyuan3D when needed."
    )
    timeout_minutes: int = 120
    coverage_threshold: float = 0.6  # Min coverage for reconstruction
    hunyuan_enabled: bool = True
    min_object_views: int = 5  # Minimum views for reconstruction
    min_object_area: int = 1000  # Minimum mask area in pixels
    max_objects: int = 50  # Maximum objects to process

    # Hunyuan3D configuration
    hunyuan_steps: int = 50
    hunyuan_guidance_scale: float = 7.5
    hunyuan_texture_resolution: int = 1024

    def _get_default_parameters(self) -> Dict[str, Any]:
        params = super()._get_default_parameters()
        params.update({
            "coverage_threshold": self.coverage_threshold,
            "hunyuan_enabled": self.hunyuan_enabled,
            "min_object_views": self.min_object_views,
            "min_object_area": self.min_object_area,
            "max_objects": self.max_objects,
            "hunyuan_steps": self.hunyuan_steps,
            "hunyuan_guidance_scale": self.hunyuan_guidance_scale,
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
                "coverage_threshold": self.coverage_threshold,
                "hunyuan_enabled": self.hunyuan_enabled,
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "frames": artifacts.frames,
                "masks": artifacts.masks,
                "poses": f"{artifacts.reconstruction}/poses",
                "environment_mesh": f"{artifacts.meshes}/environment_mesh.usd",
            },
            outputs={
                "object_usds": f"{artifacts.objects}/",
                "object_reports": f"{artifacts.reports}/objects.json",
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute object assetization pipeline."""
        result = JobResult(status=JobStatus.RUNNING)

        # Setup directories
        frames_dir = ensure_local_dir(ctx.workspace / "frames")
        masks_dir = ensure_local_dir(ctx.workspace / "masks")
        poses_dir = ensure_local_dir(ctx.workspace / "poses")
        objects_dir = ensure_local_dir(ctx.workspace / "objects")
        crops_dir = ensure_local_dir(ctx.workspace / "crops")

        # Download inputs
        with ctx.tracker.stage("download_inputs", 3):
            ctx.gcs.download_directory(ctx.artifacts.frames + "/", frames_dir)
            ctx.tracker.update(1)
            ctx.gcs.download_directory(ctx.artifacts.masks + "/", masks_dir)
            ctx.tracker.update(1)
            ctx.gcs.download_directory(
                f"{ctx.artifacts.reconstruction}/poses/", poses_dir
            )
            ctx.tracker.update(1)

        # Load annotations and poses
        annotations = self._load_annotations(masks_dir)
        poses = self._load_poses(poses_dir)
        frame_index = self._load_frame_index(frames_dir)

        ctx.logger.info(f"Loaded {len(annotations.get('annotations', []))} annotations")
        ctx.logger.info(f"Loaded {len(poses)} camera poses")

        # Build object tracks from annotations
        with ctx.tracker.stage("build_tracks", 1):
            tracks = self._build_object_tracks(ctx, annotations, masks_dir)

        # Filter to static objects (exclude dynamic like people)
        static_tracks = [t for t in tracks if not t.is_dynamic]
        ctx.logger.info(f"Found {len(static_tracks)} static object tracks (from {len(tracks)} total)")

        # Limit number of objects
        max_objects = ctx.parameters.get("max_objects", self.max_objects)
        if len(static_tracks) > max_objects:
            # Sort by coverage and take top N
            static_tracks = sorted(
                static_tracks, key=lambda t: t.coverage_score, reverse=True
            )[:max_objects]
            ctx.logger.info(f"Limited to top {max_objects} objects by coverage")

        # Compute viewpoint coverage for each track
        with ctx.tracker.stage("compute_coverage", len(static_tracks)):
            for track in static_tracks:
                track.coverage_score, track.viewpoint_diversity = self._compute_coverage(
                    ctx=ctx,
                    track=track,
                    poses=poses,
                    frame_index=frame_index,
                )
                ctx.tracker.update(1)

        ctx.tracker.log_metric("total_tracks", len(static_tracks))

        # Process each object track
        assets: List[ObjectAsset] = []
        coverage_threshold = ctx.parameters.get(
            "coverage_threshold", self.coverage_threshold
        )

        with ctx.tracker.stage("process_objects", len(static_tracks)):
            for track in static_tracks:
                ctx.logger.info(
                    f"Processing object {track.track_id}: coverage={track.coverage_score:.2f}"
                )

                # Extract crops for this object
                object_crops_dir = crops_dir / track.track_id
                crops = self._extract_object_crops(
                    ctx=ctx,
                    track=track,
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                    output_dir=object_crops_dir,
                )

                if not crops:
                    ctx.logger.warning(f"No crops extracted for {track.track_id}")
                    ctx.tracker.update(1)
                    continue

                # Decide reconstruction vs generation
                asset = None
                if track.coverage_score >= coverage_threshold and len(crops) >= ctx.parameters.get(
                    "min_object_views", self.min_object_views
                ):
                    # Tier 1: Multi-view reconstruction
                    ctx.logger.info(f"Reconstructing {track.track_id} from {len(crops)} views")
                    asset = self._reconstruct_object(
                        ctx=ctx,
                        track=track,
                        crops_dir=object_crops_dir,
                        poses=poses,
                        output_dir=objects_dir / track.track_id,
                    )
                elif ctx.parameters.get("hunyuan_enabled", self.hunyuan_enabled):
                    # Tier 2: Generate with Hunyuan3D
                    ctx.logger.info(f"Generating {track.track_id} with Hunyuan3D")
                    asset = self._generate_object(
                        ctx=ctx,
                        track=track,
                        crops_dir=object_crops_dir,
                        output_dir=objects_dir / track.track_id,
                    )

                if asset:
                    assets.append(asset)

                ctx.tracker.update(1)

        ctx.logger.info(f"Generated {len(assets)} object assets")
        ctx.tracker.log_metric("objects_generated", len(assets))
        ctx.tracker.log_metric(
            "objects_reconstructed",
            len([a for a in assets if a.source == "reconstruction"])
        )
        ctx.tracker.log_metric(
            "objects_generated_ai",
            len([a for a in assets if a.source == "generation"])
        )

        # Generate object placement report
        report = self._generate_object_report(ctx, assets, static_tracks)
        report_path = ctx.workspace / "objects_report.json"
        save_json(report, report_path)

        # Upload outputs
        with ctx.tracker.stage("upload_outputs", 2):
            if any(objects_dir.iterdir()):
                ctx.gcs.upload_directory(objects_dir, f"{ctx.artifacts.objects}/")
            ctx.tracker.update(1)
            ctx.gcs.upload(report_path, f"{ctx.artifacts.reports}/objects.json")
            ctx.tracker.update(1)

        result.outputs = {
            "object_usds": f"{ctx.artifacts.objects}/",
            "object_reports": f"{ctx.artifacts.reports}/objects.json",
        }
        result.metrics = {
            "total_objects": len(assets),
            "reconstructed": len([a for a in assets if a.source == "reconstruction"]),
            "generated": len([a for a in assets if a.source == "generation"]),
        }

        return result

    def _load_annotations(self, masks_dir: Path) -> Dict[str, Any]:
        """Load COCO-format annotations."""
        annotations_path = masks_dir / "annotations.json"
        if annotations_path.exists():
            return load_json(annotations_path)

        # Try subdirectories
        for subdir in masks_dir.iterdir():
            if subdir.is_dir():
                path = subdir / "annotations.json"
                if path.exists():
                    return load_json(path)

        return {"annotations": [], "images": [], "categories": []}

    def _load_poses(self, poses_dir: Path) -> List[Dict[str, Any]]:
        """Load camera poses."""
        poses_json = poses_dir / "poses.json"
        if poses_json.exists():
            data = load_json(poses_json)
            return data.get("poses", [])
        return []

    def _load_frame_index(self, frames_dir: Path) -> Dict[str, Any]:
        """Load frame index."""
        index_path = frames_dir / "frame_index.json"
        if index_path.exists():
            return load_json(index_path)
        return {"frames": []}

    def _build_object_tracks(
        self,
        ctx: JobContext,
        annotations: Dict[str, Any],
        masks_dir: Path,
    ) -> List[ObjectTrack]:
        """Build object tracks from per-frame annotations.

        Groups annotations by object ID or creates new tracks for untracked objects.
        """
        tracks_by_id: Dict[str, ObjectTrack] = {}

        # Get category mapping
        categories = {
            c["id"]: c["name"]
            for c in annotations.get("categories", [])
        }

        # Process annotations
        for ann in annotations.get("annotations", []):
            # Try to get track/object ID (from SAM3 video tracking)
            track_id = ann.get("track_id") or ann.get("object_id") or ann.get("id", "")
            if isinstance(track_id, int):
                track_id = f"obj_{track_id:04d}"

            frame_id = ann.get("image_id", "")
            category_id = ann.get("category_id", 0)
            category = categories.get(category_id, "object")

            bbox = tuple(ann.get("bbox", [0, 0, 0, 0]))
            area = ann.get("area", 0)
            confidence = ann.get("confidence", 0.9)
            is_dynamic = ann.get("is_dynamic", False)

            # Get mask file path
            seg = ann.get("segmentation", {})
            mask_file = seg.get("mask_file", "")

            # Filter small objects
            min_area = ctx.parameters.get("min_object_area", self.min_object_area)
            if area < min_area:
                continue

            # Create or update track
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = ObjectTrack(
                    track_id=track_id,
                    category=category,
                    frame_ids=[],
                    bboxes=[],
                    masks=[],
                    confidences=[],
                    is_dynamic=is_dynamic,
                )

            track = tracks_by_id[track_id]
            track.frame_ids.append(frame_id)
            track.bboxes.append(bbox)
            track.masks.append(mask_file)
            track.confidences.append(confidence)

            # Update dynamic status (any dynamic annotation makes the track dynamic)
            if is_dynamic:
                track.is_dynamic = True

        return list(tracks_by_id.values())

    def _compute_coverage(
        self,
        ctx: JobContext,
        track: ObjectTrack,
        poses: List[Dict[str, Any]],
        frame_index: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Compute viewpoint coverage and diversity for an object track.

        Coverage: Ratio of frames where object is visible
        Diversity: Angular spread of viewpoints (0-1)
        """
        if not track.frame_ids or not poses:
            return 0.0, 0.0

        # Find poses for frames with this object
        pose_map = {p.get("image_name", ""): p for p in poses}
        object_poses = []

        for frame_id in track.frame_ids:
            # Try to match frame_id to pose
            for pose in poses:
                img_name = pose.get("image_name", "")
                if frame_id in img_name or img_name in frame_id:
                    object_poses.append(pose)
                    break

        if not object_poses:
            return len(track.frame_ids) / max(1, len(poses)), 0.0

        # Coverage: frames with object / total frames
        coverage = len(track.frame_ids) / max(1, len(poses))

        # Viewpoint diversity: angular spread of camera directions
        # Extract camera positions from poses
        positions = []
        for pose in object_poses:
            trans = pose.get("translation", [0, 0, 0])
            positions.append(np.array(trans))

        if len(positions) < 2:
            return coverage, 0.0

        positions = np.array(positions)

        # Compute centroid and angles
        centroid = positions.mean(axis=0)
        directions = positions - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1
        directions = directions / norms

        # Angular diversity: average angle between direction pairs
        # Higher = more diverse viewpoints
        angles = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                dot = np.clip(np.dot(directions[i], directions[j]), -1, 1)
                angle = np.arccos(dot)
                angles.append(angle)

        if angles:
            # Normalize: 0 = all same direction, 1 = 180 degrees spread
            diversity = np.mean(angles) / np.pi
        else:
            diversity = 0.0

        return min(1.0, coverage), min(1.0, diversity)

    def _extract_object_crops(
        self,
        ctx: JobContext,
        track: ObjectTrack,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
    ) -> List[Path]:
        """Extract masked crops of an object from multiple frames."""
        output_dir.mkdir(parents=True, exist_ok=True)
        crops = []

        for i, (frame_id, bbox, mask_file) in enumerate(
            zip(track.frame_ids, track.bboxes, track.masks)
        ):
            # Find frame file
            frame_path = self._find_frame_file(frames_dir, frame_id)
            if not frame_path:
                continue

            # Find mask file
            mask_path = self._find_mask_file(masks_dir, mask_file)

            try:
                # Load frame
                frame = load_image(frame_path)

                # Load and apply mask if available
                if mask_path and mask_path.exists():
                    mask = load_image(mask_path, mode="L")

                    # Resize mask to frame size if needed
                    if mask.shape[:2] != frame.shape[:2]:
                        from PIL import Image
                        mask_pil = Image.fromarray(mask)
                        mask_pil = mask_pil.resize((frame.shape[1], frame.shape[0]))
                        mask = np.array(mask_pil)

                    # Apply mask (set background to white/neutral)
                    mask_3d = np.expand_dims(mask > 127, axis=-1)
                    frame = frame * mask_3d + (1 - mask_3d) * 255
                    frame = frame.astype(np.uint8)

                # Crop to bounding box with padding
                x, y, w, h = bbox
                pad = int(max(w, h) * 0.1)  # 10% padding

                x1 = max(0, int(x) - pad)
                y1 = max(0, int(y) - pad)
                x2 = min(frame.shape[1], int(x + w) + pad)
                y2 = min(frame.shape[0], int(y + h) + pad)

                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    crop_path = output_dir / f"crop_{i:04d}.png"
                    save_image(crop, crop_path)
                    crops.append(crop_path)

            except Exception as e:
                ctx.logger.debug(f"Failed to extract crop for {frame_id}: {e}")

        return crops

    def _find_frame_file(self, frames_dir: Path, frame_id: str) -> Optional[Path]:
        """Find frame file by frame ID."""
        # Direct match
        for ext in [".png", ".jpg", ".jpeg"]:
            path = frames_dir / f"{frame_id}{ext}"
            if path.exists():
                return path

        # Search subdirectories
        for subdir in frames_dir.rglob("*"):
            if subdir.is_file() and frame_id in subdir.stem:
                return subdir

        return None

    def _find_mask_file(self, masks_dir: Path, mask_file: str) -> Optional[Path]:
        """Find mask file."""
        if not mask_file:
            return None

        # Direct path
        path = masks_dir / mask_file
        if path.exists():
            return path

        # Search subdirectories
        for subdir in masks_dir.rglob(mask_file):
            return subdir

        return None

    def _reconstruct_object(
        self,
        ctx: JobContext,
        track: ObjectTrack,
        crops_dir: Path,
        poses: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Optional[ObjectAsset]:
        """Reconstruct object from multi-view crops using 3DGS.

        This is Tier 1: high-fidelity reconstruction when we have good coverage.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        ctx.logger.info(f"Reconstructing {track.track_id} from multi-view images...")

        # Get crop paths
        crop_paths = sorted(crops_dir.glob("*.png"))
        if len(crop_paths) < 3:
            ctx.logger.warning(f"Too few crops ({len(crop_paths)}) for reconstruction")
            return None

        # Try instant-ngp or 3DGS for object reconstruction
        mesh_path = None

        try:
            # Method 1: Try using instant-ngp/nerfstudio for small object
            mesh_path = self._run_object_nerf(ctx, crops_dir, output_dir)
        except Exception as e:
            ctx.logger.debug(f"NeRF reconstruction failed: {e}")

        if mesh_path is None:
            try:
                # Method 2: Try photogrammetry (COLMAP + mesh)
                mesh_path = self._run_object_photogrammetry(ctx, crops_dir, output_dir)
            except Exception as e:
                ctx.logger.debug(f"Photogrammetry failed: {e}")

        if mesh_path is None:
            ctx.logger.warning(f"All reconstruction methods failed for {track.track_id}")
            return None

        # Export to USD
        usd_path = self._export_object_usd(
            ctx=ctx,
            mesh_path=mesh_path,
            output_path=output_dir / f"{track.track_id}.usd",
            category=track.category,
        )

        # Compute object bounding box and placement
        bbox, position = self._compute_object_placement(ctx, track, poses)

        return ObjectAsset(
            asset_id=f"{ctx.session.session_id}_{track.track_id}",
            track_id=track.track_id,
            category=track.category,
            source="reconstruction",
            mesh_path=str(usd_path),
            texture_path=None,
            bounding_box=bbox,
            position_in_scene=position,
            rotation=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion
            scale=(1.0, 1.0, 1.0),
            quality_score=track.coverage_score,
        )

    def _run_object_nerf(
        self,
        ctx: JobContext,
        crops_dir: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run NeRF-based object reconstruction."""
        # Try nerfstudio
        try:
            result = subprocess.run(
                [
                    "ns-train", "instant-ngp",
                    "--data", str(crops_dir),
                    "--output-dir", str(output_dir / "nerf"),
                    "--max-num-iterations", "5000",
                ],
                capture_output=True,
                timeout=600,
            )

            if result.returncode == 0:
                # Export mesh from NeRF
                mesh_result = subprocess.run(
                    [
                        "ns-export", "poisson",
                        "--load-config", str(output_dir / "nerf" / "config.yml"),
                        "--output-dir", str(output_dir),
                    ],
                    capture_output=True,
                    timeout=300,
                )

                mesh_path = output_dir / "mesh.ply"
                if mesh_path.exists():
                    return mesh_path

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _run_object_photogrammetry(
        self,
        ctx: JobContext,
        crops_dir: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run photogrammetry-based object reconstruction."""
        try:
            import subprocess

            colmap_dir = output_dir / "colmap"
            colmap_dir.mkdir(exist_ok=True)

            database_path = colmap_dir / "database.db"
            sparse_dir = colmap_dir / "sparse"
            dense_dir = colmap_dir / "dense"
            sparse_dir.mkdir(exist_ok=True)
            dense_dir.mkdir(exist_ok=True)

            # Feature extraction
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(crops_dir),
            ], capture_output=True, timeout=300)

            # Matching
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
            ], capture_output=True, timeout=300)

            # Reconstruction
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(crops_dir),
                "--output_path", str(sparse_dir),
            ], capture_output=True, timeout=600)

            # Dense reconstruction
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(crops_dir),
                "--input_path", str(sparse_dir / "0"),
                "--output_path", str(dense_dir),
            ], capture_output=True, timeout=300)

            subprocess.run([
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense_dir),
            ], capture_output=True, timeout=600)

            subprocess.run([
                "colmap", "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(output_dir / "fused.ply"),
            ], capture_output=True, timeout=300)

            mesh_path = output_dir / "fused.ply"
            if mesh_path.exists():
                return mesh_path

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _generate_object(
        self,
        ctx: JobContext,
        track: ObjectTrack,
        crops_dir: Path,
        output_dir: Path,
    ) -> Optional[ObjectAsset]:
        """Generate object using Hunyuan3D from best crop image.

        This is Tier 2: AI generation when we don't have enough viewpoints.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get best crop (highest confidence, most centered)
        crop_paths = sorted(crops_dir.glob("*.png"))
        if not crop_paths:
            ctx.logger.warning(f"No crops available for {track.track_id}")
            return None

        # Use first crop for now (could be smarter about selection)
        best_crop = crop_paths[len(crop_paths) // 2]  # Middle crop often best

        ctx.logger.info(f"Generating {track.track_id} with Hunyuan3D from {best_crop.name}")

        # Run Hunyuan3D
        mesh_path = self._run_hunyuan3d(
            ctx=ctx,
            image_path=best_crop,
            output_dir=output_dir,
            track_id=track.track_id,
        )

        if mesh_path is None:
            # Fallback: try other generation methods
            mesh_path = self._run_fallback_generation(
                ctx=ctx,
                image_path=best_crop,
                output_dir=output_dir,
            )

        if mesh_path is None:
            ctx.logger.warning(f"All generation methods failed for {track.track_id}")
            return None

        # Export to USD
        usd_path = self._export_object_usd(
            ctx=ctx,
            mesh_path=mesh_path,
            output_path=output_dir / f"{track.track_id}.usd",
            category=track.category,
        )

        # Compute placement from track bboxes
        bbox, position = self._compute_object_placement(ctx, track, [])

        return ObjectAsset(
            asset_id=f"{ctx.session.session_id}_{track.track_id}",
            track_id=track.track_id,
            category=track.category,
            source="generation",
            mesh_path=str(usd_path),
            texture_path=None,
            bounding_box=bbox,
            position_in_scene=position,
            rotation=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            quality_score=0.5,  # Lower quality for generated
        )

    def _run_hunyuan3d(
        self,
        ctx: JobContext,
        image_path: Path,
        output_dir: Path,
        track_id: str,
    ) -> Optional[Path]:
        """Run Hunyuan3D 2.x for image-to-3D generation.

        Hunyuan3D uses a two-stage process:
        1. Shape generation from image
        2. Texture generation
        """
        ctx.logger.info("Running Hunyuan3D generation...")

        # Try using Hunyuan3D package
        try:
            from hunyuan3d import Hunyuan3D2

            # Initialize model
            model = Hunyuan3D2()

            # Generate mesh
            output_path = output_dir / f"{track_id}_hunyuan.glb"

            mesh = model.generate(
                image_path=str(image_path),
                output_path=str(output_path),
                num_steps=ctx.parameters.get("hunyuan_steps", self.hunyuan_steps),
                guidance_scale=ctx.parameters.get(
                    "hunyuan_guidance_scale", self.hunyuan_guidance_scale
                ),
                texture_resolution=ctx.parameters.get(
                    "hunyuan_texture_resolution", self.hunyuan_texture_resolution
                ),
            )

            if output_path.exists():
                ctx.logger.info(f"Hunyuan3D generated: {output_path}")
                return output_path

        except ImportError:
            ctx.logger.warning("Hunyuan3D package not installed")

        # Fallback: try running as subprocess
        try:
            result = subprocess.run(
                [
                    "python", "-m", "hunyuan3d.generate",
                    "--input", str(image_path),
                    "--output", str(output_dir / f"{track_id}_hunyuan.glb"),
                    "--steps", str(ctx.parameters.get("hunyuan_steps", self.hunyuan_steps)),
                ],
                capture_output=True,
                timeout=600,
            )

            output_path = output_dir / f"{track_id}_hunyuan.glb"
            if result.returncode == 0 and output_path.exists():
                return output_path

            ctx.logger.debug(f"Hunyuan3D CLI output: {result.stderr.decode()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            ctx.logger.debug(f"Hunyuan3D subprocess failed: {e}")

        return None

    def _run_fallback_generation(
        self,
        ctx: JobContext,
        image_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Fallback 3D generation methods when Hunyuan3D fails."""
        ctx.logger.info("Trying fallback generation methods...")

        # Try InstantMesh
        try:
            result = subprocess.run(
                [
                    "python", "-m", "instantmesh.generate",
                    "--input", str(image_path),
                    "--output", str(output_dir / "instantmesh_output.obj"),
                ],
                capture_output=True,
                timeout=300,
            )

            output_path = output_dir / "instantmesh_output.obj"
            if output_path.exists():
                return output_path

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try TripoSR
        try:
            result = subprocess.run(
                [
                    "python", "-m", "triposr.run",
                    str(image_path),
                    "--output-dir", str(output_dir),
                ],
                capture_output=True,
                timeout=300,
            )

            # TripoSR outputs to mesh.obj
            output_path = output_dir / "mesh.obj"
            if output_path.exists():
                return output_path

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Create placeholder mesh as last resort
        ctx.logger.warning("All generation methods failed, creating placeholder")
        return self._create_placeholder_object(output_dir)

    def _create_placeholder_object(self, output_dir: Path) -> Path:
        """Create a placeholder box mesh for failed objects."""
        output_path = output_dir / "placeholder.obj"

        obj_content = """# Placeholder object
v -0.1 -0.1 -0.1
v  0.1 -0.1 -0.1
v  0.1  0.1 -0.1
v -0.1  0.1 -0.1
v -0.1 -0.1  0.1
v  0.1 -0.1  0.1
v  0.1  0.1  0.1
v -0.1  0.1  0.1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
"""
        output_path.write_text(obj_content)
        return output_path

    def _export_object_usd(
        self,
        ctx: JobContext,
        mesh_path: Path,
        output_path: Path,
        category: str,
    ) -> Path:
        """Export object mesh to USD format."""
        ctx.logger.info(f"Exporting object to USD: {output_path}")

        try:
            from pxr import Usd, UsdGeom, Sdf, Gf

            # Create stage
            stage = Usd.Stage.CreateNew(str(output_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            stage.SetMetadata("upAxis", "Y")

            # Define root
            root_path = "/Object"
            root_xform = UsdGeom.Xform.Define(stage, root_path)

            # Add object mesh
            self._add_mesh_to_stage(stage, mesh_path, f"{root_path}/Mesh")

            # Add metadata
            prim = stage.GetPrimAtPath(root_path)
            prim.CreateAttribute("category", Sdf.ValueTypeNames.String).Set(category)

            stage.Save()
            return output_path

        except ImportError:
            ctx.logger.warning("USD not available, copying mesh as-is")
            shutil.copy(mesh_path, output_path.with_suffix(mesh_path.suffix))
            return output_path.with_suffix(mesh_path.suffix)

    def _add_mesh_to_stage(self, stage: Any, mesh_path: Path, prim_path: str) -> None:
        """Add mesh from file to USD stage."""
        from pxr import UsdGeom, Gf

        try:
            import trimesh

            # Load mesh
            mesh = trimesh.load(str(mesh_path))

            # Handle scene vs single mesh
            if hasattr(mesh, "geometry"):
                # Scene with multiple meshes
                meshes = list(mesh.geometry.values())
                if meshes:
                    mesh = meshes[0]
                else:
                    return
            elif not hasattr(mesh, "vertices"):
                return

            # Create USD mesh
            usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

            # Set points
            vertices = mesh.vertices
            points = [Gf.Vec3f(*v) for v in vertices]
            usd_mesh.CreatePointsAttr(points)

            # Set faces
            faces = mesh.faces
            face_counts = [3] * len(faces)
            usd_mesh.CreateFaceVertexCountsAttr(face_counts)

            indices = faces.flatten().tolist()
            usd_mesh.CreateFaceVertexIndicesAttr(indices)

            # Set normals if available
            if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) > 0:
                normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals]
                usd_mesh.CreateNormalsAttr(normals)
                usd_mesh.SetNormalsInterpolation("vertex")

        except ImportError:
            ctx.logger.warning("trimesh not available for USD mesh conversion")

    def _compute_object_placement(
        self,
        ctx: JobContext,
        track: ObjectTrack,
        poses: List[Dict[str, Any]],
    ) -> Tuple[Tuple[float, ...], Tuple[float, float, float]]:
        """Compute object bounding box and position in scene coordinates.

        This is a simplified estimation - real implementation would use
        depth estimation and triangulation.
        """
        # Estimate from average bbox size and assumed depth
        if track.bboxes:
            avg_bbox = np.mean(track.bboxes, axis=0)
            x, y, w, h = avg_bbox

            # Assume object is roughly 0.3m from camera on average
            # and estimate world-space size based on FOV
            assumed_depth = 2.0  # meters
            fov_factor = 0.001  # rough pixels-to-meters at depth

            world_w = w * fov_factor * assumed_depth
            world_h = h * fov_factor * assumed_depth
            world_d = max(world_w, world_h) * 0.5  # Assume depth is half max dimension

            bbox = (-world_w/2, -world_h/2, -world_d/2, world_w/2, world_h/2, world_d/2)

            # Position: use centroid of camera positions that see this object
            if poses:
                visible_poses = poses[:len(track.frame_ids)]
                if visible_poses:
                    positions = [p.get("translation", [0, 0, 0]) for p in visible_poses]
                    centroid = np.mean(positions, axis=0)
                    # Object is in front of average camera position
                    position = tuple(centroid + [0, 0, assumed_depth])
                else:
                    position = (0.0, 0.0, assumed_depth)
            else:
                position = (0.0, 0.0, assumed_depth)
        else:
            bbox = (-0.1, -0.1, -0.1, 0.1, 0.1, 0.1)
            position = (0.0, 0.0, 2.0)

        return bbox, position

    def _generate_object_report(
        self,
        ctx: JobContext,
        assets: List[ObjectAsset],
        tracks: List[ObjectTrack],
    ) -> Dict[str, Any]:
        """Generate detailed report on object assetization."""
        return {
            "session_id": ctx.session.session_id,
            "summary": {
                "total_tracks_found": len(tracks),
                "objects_generated": len(assets),
                "reconstructed_count": len([a for a in assets if a.source == "reconstruction"]),
                "ai_generated_count": len([a for a in assets if a.source == "generation"]),
            },
            "objects": [
                {
                    "asset_id": asset.asset_id,
                    "track_id": asset.track_id,
                    "category": asset.category,
                    "source": asset.source,
                    "mesh_path": asset.mesh_path,
                    "bounding_box": list(asset.bounding_box),
                    "position": list(asset.position_in_scene),
                    "rotation": list(asset.rotation),
                    "scale": list(asset.scale),
                    "quality_score": asset.quality_score,
                }
                for asset in assets
            ],
            "tracks": [
                {
                    "track_id": track.track_id,
                    "category": track.category,
                    "num_frames": len(track.frame_ids),
                    "is_dynamic": track.is_dynamic,
                    "coverage_score": track.coverage_score,
                    "viewpoint_diversity": track.viewpoint_diversity,
                }
                for track in tracks
            ],
        }
