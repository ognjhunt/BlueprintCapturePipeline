"""Articulation detection job using PhysX-Anything.

This job detects and configures articulated joints for interactive objects
(doors, drawers, handles, lids, etc.) from 3D meshes and/or images.

PhysX-Anything Pipeline:
    1. vlm_demo.py - Vision-Language Model processes input images/renders
    2. decoder.py - Generates 3D geometry with part segmentation
    3. split.py - Segments mesh into articulated parts
    4. simready_gen.py - Generates URDF/XML with joint configurations

Integration Flow:
    ZeroScene GLB outputs → Render views → PhysX-Anything → URDF → USD with Articulations
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import numpy as np

from ..models import ArtifactPaths, JobPayload, SessionManifest
from ..utils.io import ensure_local_dir, load_json, save_json
from .base import (
    GPUJob,
    JobContext,
    JobResult,
    JobStatus,
    download_inputs,
    merge_parameters,
    upload_outputs,
)


class JointType(Enum):
    """Supported articulation joint types."""
    REVOLUTE = "revolute"        # Rotational (doors, hinges, lids)
    PRISMATIC = "prismatic"      # Linear (drawers, sliders)
    FIXED = "fixed"              # No movement (base attachment)
    CONTINUOUS = "continuous"    # Unlimited rotation (wheels)
    SPHERICAL = "spherical"      # Ball joint (3 DOF rotation)


class ArticulationType(Enum):
    """Classification of articulated object types."""
    DOOR = "door"                # Cabinet doors, room doors
    DRAWER = "drawer"            # Pull-out drawers
    LID = "lid"                  # Box lids, toilet seats
    HANDLE = "handle"            # Door handles, knobs
    LEVER = "lever"              # Levers, switches
    WHEEL = "wheel"              # Rotating wheels
    FAUCET = "faucet"            # Faucet handles
    APPLIANCE = "appliance"      # Microwave doors, oven doors
    STATIC = "static"            # Non-articulated objects


@dataclass
class JointLimits:
    """Physical limits for joint motion."""
    lower: float = 0.0           # Lower bound (radians for revolute, meters for prismatic)
    upper: float = 0.0           # Upper bound
    effort: float = 100.0        # Maximum force/torque
    velocity: float = 1.0        # Maximum velocity

    def to_dict(self) -> Dict[str, float]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "effort": self.effort,
            "velocity": self.velocity,
        }


@dataclass
class JointDynamics:
    """Dynamic properties for joint simulation."""
    damping: float = 0.1
    friction: float = 0.0
    stiffness: float = 0.0       # For spring-like behavior

    def to_dict(self) -> Dict[str, float]:
        return {
            "damping": self.damping,
            "friction": self.friction,
            "stiffness": self.stiffness,
        }


@dataclass
class ArticulationJoint:
    """Detected articulation joint configuration."""
    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    origin: Tuple[float, float, float]         # Position (x, y, z)
    axis: Tuple[float, float, float]           # Rotation/translation axis
    limits: JointLimits = field(default_factory=JointLimits)
    dynamics: JointDynamics = field(default_factory=JointDynamics)
    confidence: float = 0.0                    # Detection confidence [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "joint_type": self.joint_type.value,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
            "origin": list(self.origin),
            "axis": list(self.axis),
            "limits": self.limits.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "confidence": self.confidence,
        }


@dataclass
class ArticulatedObject:
    """An object with detected articulation."""
    object_id: str
    source_mesh: str                           # Path to source GLB/USD
    articulation_type: ArticulationType
    joints: List[ArticulationJoint] = field(default_factory=list)
    base_link: str = "base_link"
    urdf_path: Optional[str] = None            # Generated URDF path
    usd_path: Optional[str] = None             # Output USD with articulation
    confidence: float = 0.0                    # Overall detection confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "source_mesh": self.source_mesh,
            "articulation_type": self.articulation_type.value,
            "joints": [j.to_dict() for j in self.joints],
            "base_link": self.base_link,
            "urdf_path": self.urdf_path,
            "usd_path": self.usd_path,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# Default joint configurations for common object types
DEFAULT_JOINT_CONFIGS: Dict[ArticulationType, Dict[str, Any]] = {
    ArticulationType.DOOR: {
        "joint_type": JointType.REVOLUTE,
        "limits": JointLimits(lower=0.0, upper=math.pi * 0.6, effort=50.0, velocity=2.0),  # ~108 degrees
        "dynamics": JointDynamics(damping=0.5, friction=0.1),
        "axis": (0.0, 0.0, 1.0),  # Z-axis rotation (vertical hinge)
    },
    ArticulationType.DRAWER: {
        "joint_type": JointType.PRISMATIC,
        "limits": JointLimits(lower=0.0, upper=0.5, effort=30.0, velocity=0.5),  # 50cm travel
        "dynamics": JointDynamics(damping=0.3, friction=0.2),
        "axis": (0.0, 1.0, 0.0),  # Y-axis translation (pull out)
    },
    ArticulationType.LID: {
        "joint_type": JointType.REVOLUTE,
        "limits": JointLimits(lower=0.0, upper=math.pi * 0.5, effort=20.0, velocity=1.5),  # 90 degrees
        "dynamics": JointDynamics(damping=0.2, friction=0.05),
        "axis": (1.0, 0.0, 0.0),  # X-axis rotation (horizontal hinge)
    },
    ArticulationType.HANDLE: {
        "joint_type": JointType.REVOLUTE,
        "limits": JointLimits(lower=-math.pi * 0.25, upper=math.pi * 0.25, effort=10.0, velocity=3.0),  # 45 degrees each way
        "dynamics": JointDynamics(damping=0.1, friction=0.05, stiffness=5.0),  # Spring return
        "axis": (1.0, 0.0, 0.0),
    },
    ArticulationType.LEVER: {
        "joint_type": JointType.REVOLUTE,
        "limits": JointLimits(lower=0.0, upper=math.pi * 0.33, effort=15.0, velocity=2.0),  # 60 degrees
        "dynamics": JointDynamics(damping=0.15, friction=0.1),
        "axis": (1.0, 0.0, 0.0),
    },
    ArticulationType.WHEEL: {
        "joint_type": JointType.CONTINUOUS,
        "limits": JointLimits(effort=100.0, velocity=10.0),  # No position limits
        "dynamics": JointDynamics(damping=0.05, friction=0.02),
        "axis": (1.0, 0.0, 0.0),  # X-axis rotation
    },
    ArticulationType.FAUCET: {
        "joint_type": JointType.REVOLUTE,
        "limits": JointLimits(lower=0.0, upper=math.pi * 0.5, effort=5.0, velocity=2.0),  # 90 degrees
        "dynamics": JointDynamics(damping=0.1, friction=0.05),
        "axis": (0.0, 0.0, 1.0),
    },
    ArticulationType.APPLIANCE: {
        "joint_type": JointType.REVOLUTE,
        "limits": JointLimits(lower=0.0, upper=math.pi * 0.7, effort=40.0, velocity=1.5),  # ~126 degrees
        "dynamics": JointDynamics(damping=0.4, friction=0.1),
        "axis": (0.0, 0.0, 1.0),
    },
}


@dataclass
class ArticulationJob(GPUJob):
    """Detect and configure articulations using PhysX-Anything.

    This job processes 3D meshes (GLB/USD) from ZeroScene or other sources
    to detect articulated parts and generate simulation-ready assets with
    proper joint configurations.

    Pipeline:
        1. Classify objects that may have articulation
        2. Render reference views from meshes (if images not provided)
        3. Run PhysX-Anything VLM to detect articulation
        4. Generate URDF with joint configurations
        5. Convert to USD with PhysX articulation APIs

    Inputs:
        - Object meshes (GLB/USD) from objects/ directory
        - Optional reference images from frames/
        - Object metadata from reports/objects.json

    Outputs:
        - Articulated USD files in articulations/ directory
        - Articulation report in reports/articulation.json
    """

    name: str = "articulation"
    description: str = "Detect articulations using PhysX-Anything and generate simulation-ready joints."
    timeout_minutes: int = 60
    uses_gpu: bool = True
    gpu_type: str = "L4"
    gpu_memory_gb: int = 24
    min_gpu_memory_gb: float = 16.0

    # PhysX-Anything configuration
    physx_anything_path: str = "/opt/PhysX-Anything"  # Installation path
    voxel_resolution: int = 32
    enable_post_processing: bool = True
    fixed_base: bool = True                          # Fix base link to world
    enable_deformable: bool = False                  # Deformable body support

    # Articulation detection thresholds
    min_confidence: float = 0.5                      # Minimum detection confidence
    max_objects_per_batch: int = 10                  # Batch processing limit

    # Rendering configuration (for mesh-to-image)
    render_resolution: Tuple[int, int] = (512, 512)
    render_views: int = 4                            # Number of views to render
    render_fov: float = 60.0                         # Field of view

    # Output configuration
    output_format: str = "usda"                      # usda, usdc, usd
    include_urdf: bool = True                        # Also output URDF files
    merge_into_scene: bool = True                    # Merge articulated objects into scene

    def _get_default_parameters(self) -> Dict[str, Any]:
        return {
            "physx_anything_path": self.physx_anything_path,
            "voxel_resolution": self.voxel_resolution,
            "enable_post_processing": self.enable_post_processing,
            "fixed_base": self.fixed_base,
            "min_confidence": self.min_confidence,
            "render_resolution": list(self.render_resolution),
            "render_views": self.render_views,
            "output_format": self.output_format,
            "include_urdf": self.include_urdf,
        }

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self._get_default_parameters(),
            parameters,
        )
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "objects": f"{artifacts.objects}/",
                "object_report": f"{artifacts.reports}/objects.json",
                "frames": f"{artifacts.frames}/",         # Optional reference images
                "scene_usd": f"{artifacts.session_root}/scene.usdc",  # For scene integration
            },
            outputs={
                "articulations": f"{artifacts.session_root}/articulations/",
                "report": f"{artifacts.reports}/articulation.json",
                "scene_articulated": f"{artifacts.session_root}/scene_articulated.usdc",
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute articulation detection pipeline."""
        result = JobResult(status=JobStatus.RUNNING)

        # Setup directories
        objects_dir = ensure_local_dir(ctx.workspace / "objects")
        frames_dir = ensure_local_dir(ctx.workspace / "frames")
        reports_dir = ensure_local_dir(ctx.workspace / "reports")
        output_dir = ensure_local_dir(ctx.workspace / "articulations")
        physx_work_dir = ensure_local_dir(ctx.workspace / "physx_anything")

        # Download inputs
        with ctx.tracker.stage("download_inputs", 4):
            # Object meshes
            try:
                ctx.gcs.download_directory(f"{ctx.artifacts.objects}/", objects_dir)
            except Exception as e:
                ctx.logger.warning(f"Could not download objects: {e}")
            ctx.tracker.update(1)

            # Object report
            try:
                ctx.gcs.download(
                    f"{ctx.artifacts.reports}/objects.json",
                    reports_dir / "objects.json"
                )
            except Exception as e:
                ctx.logger.warning(f"Could not download objects report: {e}")
            ctx.tracker.update(1)

            # Reference frames (optional)
            try:
                ctx.gcs.download_directory(f"{ctx.artifacts.frames}/", frames_dir)
            except Exception as e:
                ctx.logger.info(f"No reference frames available: {e}")
            ctx.tracker.update(1)

            # Scene USD (for merging)
            try:
                ctx.gcs.download(
                    f"{ctx.artifacts.session_root}/scene.usdc",
                    ctx.workspace / "scene.usdc"
                )
            except Exception as e:
                ctx.logger.info(f"No scene USD available: {e}")
            ctx.tracker.update(1)

        # Load object metadata
        object_report = {}
        if (reports_dir / "objects.json").exists():
            object_report = load_json(reports_dir / "objects.json")

        objects_info = object_report.get("objects", [])
        ctx.logger.info(f"Processing {len(objects_info)} objects for articulation detection")

        # Phase 1: Classify objects for articulation
        with ctx.tracker.stage("classify_objects", len(objects_info)):
            articulation_candidates = self._classify_articulation_candidates(
                ctx, objects_info, objects_dir
            )
            ctx.logger.info(f"Found {len(articulation_candidates)} articulation candidates")

        # Phase 2: Run PhysX-Anything on candidates
        articulated_objects: List[ArticulatedObject] = []

        with ctx.tracker.stage("detect_articulation", len(articulation_candidates)):
            for candidate in articulation_candidates:
                try:
                    articulated = self._process_articulation_candidate(
                        ctx=ctx,
                        candidate=candidate,
                        objects_dir=objects_dir,
                        frames_dir=frames_dir,
                        work_dir=physx_work_dir,
                        output_dir=output_dir,
                    )
                    if articulated and articulated.confidence >= ctx.parameters.get(
                        "min_confidence", self.min_confidence
                    ):
                        articulated_objects.append(articulated)
                        ctx.logger.info(
                            f"Detected articulation for {candidate['track_id']}: "
                            f"{articulated.articulation_type.value} with {len(articulated.joints)} joints "
                            f"(confidence: {articulated.confidence:.2f})"
                        )
                except Exception as e:
                    ctx.logger.error(f"Failed to process {candidate.get('track_id')}: {e}")
                ctx.tracker.update(1)

        # Phase 3: Convert to USD with articulation
        with ctx.tracker.stage("convert_to_usd", len(articulated_objects)):
            for obj in articulated_objects:
                try:
                    usd_path = self._convert_to_articulated_usd(
                        ctx=ctx,
                        articulated_obj=obj,
                        output_dir=output_dir,
                    )
                    obj.usd_path = str(usd_path)
                except Exception as e:
                    ctx.logger.error(f"Failed to convert {obj.object_id} to USD: {e}")
                ctx.tracker.update(1)

        # Phase 4: Optionally merge into scene
        scene_articulated_path = None
        if ctx.parameters.get("merge_into_scene", self.merge_into_scene):
            with ctx.tracker.stage("merge_scene", 1):
                scene_path = ctx.workspace / "scene.usdc"
                if scene_path.exists():
                    try:
                        scene_articulated_path = self._merge_articulations_into_scene(
                            ctx=ctx,
                            scene_path=scene_path,
                            articulated_objects=articulated_objects,
                            output_path=ctx.workspace / "scene_articulated.usdc",
                        )
                    except Exception as e:
                        ctx.logger.error(f"Failed to merge articulations into scene: {e}")

        # Generate articulation report
        report = {
            "session_id": ctx.session.session_id,
            "total_objects_analyzed": len(objects_info),
            "articulation_candidates": len(articulation_candidates),
            "articulated_objects": len(articulated_objects),
            "objects": [obj.to_dict() for obj in articulated_objects],
            "metrics": {
                "average_confidence": (
                    sum(o.confidence for o in articulated_objects) / len(articulated_objects)
                    if articulated_objects else 0.0
                ),
                "joint_types": self._count_joint_types(articulated_objects),
                "articulation_types": self._count_articulation_types(articulated_objects),
            },
            "parameters": ctx.parameters,
        }
        save_json(report, output_dir / "articulation_report.json")

        # Upload outputs
        with ctx.tracker.stage("upload_outputs", 3):
            # Upload articulation directory
            ctx.gcs.upload_directory(
                output_dir,
                f"{ctx.artifacts.session_root}/articulations/"
            )
            ctx.tracker.update(1)

            # Upload report
            ctx.gcs.upload(
                output_dir / "articulation_report.json",
                f"{ctx.artifacts.reports}/articulation.json"
            )
            ctx.tracker.update(1)

            # Upload articulated scene if created
            if scene_articulated_path and scene_articulated_path.exists():
                ctx.gcs.upload(
                    scene_articulated_path,
                    f"{ctx.artifacts.session_root}/scene_articulated.usdc"
                )
            ctx.tracker.update(1)

        result.outputs = {
            "articulations": f"{ctx.artifacts.session_root}/articulations/",
            "report": f"{ctx.artifacts.reports}/articulation.json",
        }
        if scene_articulated_path:
            result.outputs["scene_articulated"] = f"{ctx.artifacts.session_root}/scene_articulated.usdc"

        result.metrics = report["metrics"]
        result.metrics["articulated_objects_count"] = len(articulated_objects)

        return result

    def _classify_articulation_candidates(
        self,
        ctx: JobContext,
        objects_info: List[Dict[str, Any]],
        objects_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Classify which objects are candidates for articulation detection.

        Uses category labels and heuristics to identify objects that likely
        have articulated parts (doors, drawers, etc.).
        """
        candidates = []

        # Keywords that suggest articulation
        articulation_keywords = {
            "door": ArticulationType.DOOR,
            "drawer": ArticulationType.DRAWER,
            "cabinet": ArticulationType.DOOR,
            "cupboard": ArticulationType.DOOR,
            "lid": ArticulationType.LID,
            "box": ArticulationType.LID,
            "chest": ArticulationType.LID,
            "handle": ArticulationType.HANDLE,
            "knob": ArticulationType.HANDLE,
            "lever": ArticulationType.LEVER,
            "switch": ArticulationType.LEVER,
            "wheel": ArticulationType.WHEEL,
            "faucet": ArticulationType.FAUCET,
            "tap": ArticulationType.FAUCET,
            "refrigerator": ArticulationType.APPLIANCE,
            "fridge": ArticulationType.APPLIANCE,
            "microwave": ArticulationType.APPLIANCE,
            "oven": ArticulationType.APPLIANCE,
            "washer": ArticulationType.APPLIANCE,
            "dryer": ArticulationType.APPLIANCE,
            "dishwasher": ArticulationType.APPLIANCE,
            "toilet": ArticulationType.LID,
        }

        for obj in objects_info:
            track_id = obj.get("track_id", "")
            category = obj.get("category", "").lower()
            label = obj.get("label", "").lower()

            # Check for mesh file
            obj_dir = objects_dir / track_id
            has_mesh = False
            if obj_dir.exists():
                for ext in (".glb", ".gltf", ".usd", ".usda", ".usdc"):
                    if list(obj_dir.glob(f"*{ext}")):
                        has_mesh = True
                        break

            if not has_mesh:
                continue

            # Match keywords to determine articulation type
            matched_type = None
            for keyword, art_type in articulation_keywords.items():
                if keyword in category or keyword in label:
                    matched_type = art_type
                    break

            if matched_type:
                obj["suggested_articulation_type"] = matched_type.value
                candidates.append(obj)
            elif obj.get("may_be_interactive", False):
                # If object was flagged as potentially interactive
                obj["suggested_articulation_type"] = ArticulationType.DOOR.value  # Default guess
                candidates.append(obj)

        return candidates

    def _process_articulation_candidate(
        self,
        ctx: JobContext,
        candidate: Dict[str, Any],
        objects_dir: Path,
        frames_dir: Path,
        work_dir: Path,
        output_dir: Path,
    ) -> Optional[ArticulatedObject]:
        """Process a single articulation candidate through PhysX-Anything."""
        track_id = candidate.get("track_id", "")
        obj_dir = objects_dir / track_id
        obj_work_dir = ensure_local_dir(work_dir / track_id)

        # Find mesh file
        mesh_path = None
        for ext in (".glb", ".gltf", ".usd", ".usda", ".usdc"):
            matches = list(obj_dir.glob(f"*{ext}"))
            if matches:
                mesh_path = matches[0]
                break

        if not mesh_path:
            ctx.logger.warning(f"No mesh found for {track_id}")
            return None

        # Step 1: Get or render reference images
        image_path = self._get_reference_image(
            ctx=ctx,
            track_id=track_id,
            mesh_path=mesh_path,
            frames_dir=frames_dir,
            work_dir=obj_work_dir,
        )

        if not image_path:
            ctx.logger.warning(f"Could not get reference image for {track_id}")
            return None

        # Step 2: Run PhysX-Anything pipeline
        urdf_path, physx_result = self._run_physx_anything(
            ctx=ctx,
            image_path=image_path,
            work_dir=obj_work_dir,
        )

        if not urdf_path:
            # PhysX-Anything may have determined object is not articulated
            # In this case, we fall back to heuristics
            ctx.logger.info(f"PhysX-Anything did not detect articulation for {track_id}, using heuristics")
            return self._create_heuristic_articulation(
                ctx=ctx,
                candidate=candidate,
                mesh_path=mesh_path,
                output_dir=output_dir,
            )

        # Step 3: Parse URDF to extract joint information
        joints = self._parse_urdf_joints(ctx, urdf_path)

        if not joints:
            ctx.logger.info(f"No joints found in URDF for {track_id}")
            return None

        # Create ArticulatedObject
        suggested_type_str = candidate.get("suggested_articulation_type", "door")
        try:
            suggested_type = ArticulationType(suggested_type_str)
        except ValueError:
            suggested_type = ArticulationType.DOOR

        articulated = ArticulatedObject(
            object_id=track_id,
            source_mesh=str(mesh_path),
            articulation_type=suggested_type,
            joints=joints,
            base_link=physx_result.get("base_link", "base_link"),
            urdf_path=str(urdf_path),
            confidence=physx_result.get("confidence", 0.7),
            metadata={
                "physx_anything_result": physx_result,
                "original_category": candidate.get("category", ""),
            },
        )

        # Copy URDF to output if requested
        if ctx.parameters.get("include_urdf", self.include_urdf):
            urdf_output = ensure_local_dir(output_dir / track_id) / f"{track_id}.urdf"
            shutil.copy(urdf_path, urdf_output)
            articulated.urdf_path = str(urdf_output)

        return articulated

    def _get_reference_image(
        self,
        ctx: JobContext,
        track_id: str,
        mesh_path: Path,
        frames_dir: Path,
        work_dir: Path,
    ) -> Optional[Path]:
        """Get or render a reference image for the object.

        Tries to find an existing frame, otherwise renders views from mesh.
        """
        # Try to find existing cropped image for this object
        for img_file in frames_dir.glob(f"*{track_id}*.png"):
            return img_file
        for img_file in frames_dir.glob(f"*{track_id}*.jpg"):
            return img_file

        # Render views from mesh
        ctx.logger.info(f"Rendering reference views for {track_id} from {mesh_path}")
        return self._render_mesh_views(ctx, mesh_path, work_dir)

    def _render_mesh_views(
        self,
        ctx: JobContext,
        mesh_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Render reference views from a 3D mesh using Python rendering.

        Uses trimesh + pyrender for offline rendering, or falls back to
        a simpler approach if those aren't available.
        """
        try:
            import trimesh

            # Load mesh
            mesh = trimesh.load(str(mesh_path), force='mesh')

            # Get render resolution
            res = ctx.parameters.get("render_resolution", list(self.render_resolution))
            if isinstance(res, list):
                res = tuple(res)

            # Try pyrender first
            try:
                import pyrender
                from PIL import Image

                # Create scene
                scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])

                # Add mesh
                if isinstance(mesh, trimesh.Trimesh):
                    mesh_node = pyrender.Mesh.from_trimesh(mesh)
                    scene.add(mesh_node)
                else:
                    for m in mesh.geometry.values():
                        mesh_node = pyrender.Mesh.from_trimesh(m)
                        scene.add(mesh_node)

                # Setup camera looking at mesh center
                bounds = mesh.bounds
                center = (bounds[0] + bounds[1]) / 2
                extent = np.max(bounds[1] - bounds[0])
                camera_distance = extent * 2.0

                camera = pyrender.PerspectiveCamera(
                    yfov=np.radians(ctx.parameters.get("render_fov", self.render_fov))
                )
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = center + np.array([0, 0, camera_distance])
                scene.add(camera, pose=camera_pose)

                # Add light
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                scene.add(light, pose=camera_pose)

                # Render
                renderer = pyrender.OffscreenRenderer(res[0], res[1])
                color, _ = renderer.render(scene)
                renderer.delete()

                # Save image
                output_path = output_dir / "reference.png"
                Image.fromarray(color).save(output_path)
                return output_path

            except ImportError:
                ctx.logger.warning("pyrender not available, using trimesh export")

            # Fallback: Export scene as image using trimesh's built-in
            output_path = output_dir / "reference.png"
            scene = mesh.scene()
            png_data = scene.save_image(resolution=res)
            with open(output_path, 'wb') as f:
                f.write(png_data)
            return output_path

        except ImportError:
            ctx.logger.warning("trimesh not available for mesh rendering")
            return None
        except Exception as e:
            ctx.logger.error(f"Failed to render mesh: {e}")
            return None

    def _run_physx_anything(
        self,
        ctx: JobContext,
        image_path: Path,
        work_dir: Path,
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Run the PhysX-Anything pipeline on an input image.

        PhysX-Anything consists of 4 sequential scripts:
            1. 1_vlm_demo.py - VLM processes input image
            2. 2_decoder.py - Generates 3D geometry
            3. 3_split.py - Segments mesh into parts
            4. 4_simready_gen.py - Generates URDF/XML

        Returns:
            Tuple of (urdf_path, result_dict) or (None, {}) if failed.
        """
        physx_path = Path(ctx.parameters.get("physx_anything_path", self.physx_anything_path))
        result: Dict[str, Any] = {}

        if not physx_path.exists():
            ctx.logger.warning(f"PhysX-Anything not found at {physx_path}")
            return None, result

        # Setup environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            # Step 1: VLM Demo
            ctx.logger.info("Running PhysX-Anything Step 1: VLM Demo")
            vlm_cmd = [
                "python", str(physx_path / "1_vlm_demo.py"),
                "--demo_path", str(image_path),
                "--save_folder", str(work_dir / "vlm_output"),
            ]
            subprocess.run(vlm_cmd, env=env, cwd=str(physx_path), check=True, capture_output=True)

            # Step 2: Decoder
            ctx.logger.info("Running PhysX-Anything Step 2: Decoder")
            decoder_cmd = [
                "python", str(physx_path / "2_decoder.py"),
                "--basepath", str(work_dir / "vlm_output"),
                "--save_folder", str(work_dir / "decoder_output"),
            ]
            subprocess.run(decoder_cmd, env=env, cwd=str(physx_path), check=True, capture_output=True)

            # Step 3: Split
            ctx.logger.info("Running PhysX-Anything Step 3: Split")
            split_cmd = [
                "python", str(physx_path / "3_split.py"),
                "--basepath", str(work_dir / "decoder_output"),
                "--save_folder", str(work_dir / "split_output"),
            ]
            subprocess.run(split_cmd, env=env, cwd=str(physx_path), check=True, capture_output=True)

            # Step 4: SimReady Generation
            ctx.logger.info("Running PhysX-Anything Step 4: SimReady Generation")
            simready_cmd = [
                "python", str(physx_path / "4_simready_gen.py"),
                "--basepath", str(work_dir / "split_output"),
                "--save_folder", str(work_dir / "simready_output"),
                "--voxel_define", str(ctx.parameters.get("voxel_resolution", self.voxel_resolution)),
            ]

            if ctx.parameters.get("enable_post_processing", self.enable_post_processing):
                simready_cmd.append("--process")
            if ctx.parameters.get("fixed_base", self.fixed_base):
                simready_cmd.append("--fixed_base")
            if ctx.parameters.get("enable_deformable", self.enable_deformable):
                simready_cmd.append("--deformable")

            subprocess.run(simready_cmd, env=env, cwd=str(physx_path), check=True, capture_output=True)

            # Find output URDF
            simready_dir = work_dir / "simready_output"
            urdf_files = list(simready_dir.glob("**/*.urdf"))

            if urdf_files:
                urdf_path = urdf_files[0]
                result["confidence"] = 0.8  # PhysX-Anything detected articulation
                result["base_link"] = "base_link"
                result["urdf_generated"] = True

                # Try to load any metadata
                metadata_files = list(simready_dir.glob("**/*.json"))
                if metadata_files:
                    try:
                        result["physx_metadata"] = load_json(metadata_files[0])
                    except Exception:
                        pass

                return urdf_path, result

            else:
                ctx.logger.info("PhysX-Anything did not generate URDF (object may not be articulated)")
                result["confidence"] = 0.0
                result["urdf_generated"] = False
                return None, result

        except subprocess.CalledProcessError as e:
            ctx.logger.error(f"PhysX-Anything failed: {e.stderr.decode() if e.stderr else str(e)}")
            return None, {"error": str(e)}
        except Exception as e:
            ctx.logger.error(f"PhysX-Anything pipeline error: {e}")
            return None, {"error": str(e)}

    def _parse_urdf_joints(
        self,
        ctx: JobContext,
        urdf_path: Path,
    ) -> List[ArticulationJoint]:
        """Parse URDF file to extract joint configurations."""
        joints = []

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(urdf_path)
            root = tree.getroot()

            for joint_elem in root.findall("joint"):
                joint_name = joint_elem.get("name", "joint")
                joint_type_str = joint_elem.get("type", "fixed")

                # Map URDF joint type to our enum
                type_map = {
                    "revolute": JointType.REVOLUTE,
                    "prismatic": JointType.PRISMATIC,
                    "continuous": JointType.CONTINUOUS,
                    "fixed": JointType.FIXED,
                    "floating": JointType.SPHERICAL,
                    "planar": JointType.PRISMATIC,
                }
                joint_type = type_map.get(joint_type_str, JointType.FIXED)

                # Skip fixed joints for articulation
                if joint_type == JointType.FIXED:
                    continue

                # Parse parent and child links
                parent_elem = joint_elem.find("parent")
                child_elem = joint_elem.find("child")
                parent_link = parent_elem.get("link", "base_link") if parent_elem is not None else "base_link"
                child_link = child_elem.get("link", "link") if child_elem is not None else "link"

                # Parse origin
                origin_elem = joint_elem.find("origin")
                if origin_elem is not None:
                    xyz_str = origin_elem.get("xyz", "0 0 0")
                    origin = tuple(float(x) for x in xyz_str.split())
                else:
                    origin = (0.0, 0.0, 0.0)

                # Parse axis
                axis_elem = joint_elem.find("axis")
                if axis_elem is not None:
                    xyz_str = axis_elem.get("xyz", "1 0 0")
                    axis = tuple(float(x) for x in xyz_str.split())
                else:
                    axis = (1.0, 0.0, 0.0)

                # Parse limits
                limit_elem = joint_elem.find("limit")
                if limit_elem is not None:
                    limits = JointLimits(
                        lower=float(limit_elem.get("lower", 0)),
                        upper=float(limit_elem.get("upper", 0)),
                        effort=float(limit_elem.get("effort", 100)),
                        velocity=float(limit_elem.get("velocity", 1)),
                    )
                else:
                    limits = JointLimits()

                # Parse dynamics
                dynamics_elem = joint_elem.find("dynamics")
                if dynamics_elem is not None:
                    dynamics = JointDynamics(
                        damping=float(dynamics_elem.get("damping", 0.1)),
                        friction=float(dynamics_elem.get("friction", 0)),
                    )
                else:
                    dynamics = JointDynamics()

                joints.append(ArticulationJoint(
                    name=joint_name,
                    joint_type=joint_type,
                    parent_link=parent_link,
                    child_link=child_link,
                    origin=origin,
                    axis=axis,
                    limits=limits,
                    dynamics=dynamics,
                    confidence=0.8,  # From PhysX-Anything detection
                ))

        except Exception as e:
            ctx.logger.error(f"Failed to parse URDF {urdf_path}: {e}")

        return joints

    def _create_heuristic_articulation(
        self,
        ctx: JobContext,
        candidate: Dict[str, Any],
        mesh_path: Path,
        output_dir: Path,
    ) -> Optional[ArticulatedObject]:
        """Create articulation using heuristics when PhysX-Anything fails.

        Falls back to predefined joint configurations based on object category.
        """
        track_id = candidate.get("track_id", "")
        suggested_type_str = candidate.get("suggested_articulation_type", "door")

        try:
            art_type = ArticulationType(suggested_type_str)
        except ValueError:
            art_type = ArticulationType.DOOR

        # Skip if no default config
        if art_type not in DEFAULT_JOINT_CONFIGS:
            return None

        config = DEFAULT_JOINT_CONFIGS[art_type]

        # Create a simple articulation with one joint
        joint = ArticulationJoint(
            name=f"{track_id}_joint",
            joint_type=config["joint_type"],
            parent_link="base_link",
            child_link="moving_link",
            origin=(0.0, 0.0, 0.0),
            axis=config["axis"],
            limits=config["limits"],
            dynamics=config["dynamics"],
            confidence=0.5,  # Lower confidence for heuristic
        )

        return ArticulatedObject(
            object_id=track_id,
            source_mesh=str(mesh_path),
            articulation_type=art_type,
            joints=[joint],
            base_link="base_link",
            confidence=0.5,
            metadata={
                "method": "heuristic",
                "original_category": candidate.get("category", ""),
            },
        )

    def _convert_to_articulated_usd(
        self,
        ctx: JobContext,
        articulated_obj: ArticulatedObject,
        output_dir: Path,
    ) -> Path:
        """Convert articulated object to USD with PhysX articulation APIs."""
        obj_output_dir = ensure_local_dir(output_dir / articulated_obj.object_id)
        output_format = ctx.parameters.get("output_format", self.output_format)
        output_path = obj_output_dir / f"{articulated_obj.object_id}_articulated.{output_format}"

        try:
            from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf

            # Create stage
            stage = Usd.Stage.CreateNew(str(output_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            stage.SetMetadata("upAxis", "Y")

            # Create root
            root_path = "/World"
            root_xform = UsdGeom.Xform.Define(stage, root_path)
            stage.SetDefaultPrim(root_xform.GetPrim())

            # Create articulation root
            art_root_path = f"{root_path}/{articulated_obj.object_id}"
            art_xform = UsdGeom.Xform.Define(stage, art_root_path)

            # Apply articulation root API
            art_root_api = UsdPhysics.ArticulationRootAPI.Apply(art_xform.GetPrim())

            # Try to apply PhysX articulation API for more control
            try:
                physx_art_api = PhysxSchema.PhysxArticulationAPI.Apply(art_xform.GetPrim())
                physx_art_api.CreateEnabledSelfCollisionsAttr().Set(False)
                physx_art_api.CreateSolverPositionIterationCountAttr().Set(32)
                physx_art_api.CreateSolverVelocityIterationCountAttr().Set(1)
            except Exception:
                ctx.logger.info("PhysxSchema not available, using basic USD Physics")

            # Create base link
            base_link_path = f"{art_root_path}/{articulated_obj.base_link}"
            base_link_xform = UsdGeom.Xform.Define(stage, base_link_path)

            # Reference source mesh in base link
            if articulated_obj.source_mesh:
                mesh_ref_path = f"{base_link_path}/Mesh"
                mesh_prim = stage.DefinePrim(mesh_ref_path)
                mesh_prim.GetReferences().AddReference(articulated_obj.source_mesh)

            # Add rigid body to base (fixed if configured)
            if ctx.parameters.get("fixed_base", self.fixed_base):
                # Don't add rigid body API to keep it fixed
                collision_api = UsdPhysics.CollisionAPI.Apply(base_link_xform.GetPrim())
            else:
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(base_link_xform.GetPrim())

            # Create joints and child links
            for joint in articulated_obj.joints:
                # Create child link
                child_link_path = f"{art_root_path}/{joint.child_link}"
                child_link_xform = UsdGeom.Xform.Define(stage, child_link_path)

                # Apply rigid body to child link
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(child_link_xform.GetPrim())

                # Create joint
                joint_path = f"{art_root_path}/{joint.name}"

                if joint.joint_type == JointType.REVOLUTE:
                    usd_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                    usd_joint.CreateAxisAttr().Set(self._axis_to_usd(joint.axis))
                    usd_joint.CreateLowerLimitAttr().Set(math.degrees(joint.limits.lower))
                    usd_joint.CreateUpperLimitAttr().Set(math.degrees(joint.limits.upper))

                elif joint.joint_type == JointType.PRISMATIC:
                    usd_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
                    usd_joint.CreateAxisAttr().Set(self._axis_to_usd(joint.axis))
                    usd_joint.CreateLowerLimitAttr().Set(joint.limits.lower)
                    usd_joint.CreateUpperLimitAttr().Set(joint.limits.upper)

                elif joint.joint_type == JointType.CONTINUOUS:
                    # Continuous is revolute without limits
                    usd_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                    usd_joint.CreateAxisAttr().Set(self._axis_to_usd(joint.axis))
                    # No limits for continuous

                elif joint.joint_type == JointType.SPHERICAL:
                    usd_joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)

                else:  # Fixed or unknown
                    usd_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

                # Set joint relationships
                usd_joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{art_root_path}/{joint.parent_link}")])
                usd_joint.CreateBody1Rel().SetTargets([Sdf.Path(child_link_path)])

                # Set joint origin/pose
                usd_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*joint.origin))
                usd_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))

                # Apply drive if available
                try:
                    drive_api = UsdPhysics.DriveAPI.Apply(usd_joint.GetPrim(), "angular" if joint.joint_type in [JointType.REVOLUTE, JointType.CONTINUOUS] else "linear")
                    drive_api.CreateDampingAttr().Set(joint.dynamics.damping)
                    if joint.dynamics.stiffness > 0:
                        drive_api.CreateStiffnessAttr().Set(joint.dynamics.stiffness)
                except Exception:
                    pass

            stage.Save()
            ctx.logger.info(f"Created articulated USD: {output_path}")
            return output_path

        except ImportError as e:
            ctx.logger.error(f"USD libraries not available: {e}")
            # Create placeholder JSON
            placeholder = {
                "format": "placeholder",
                "object_id": articulated_obj.object_id,
                "articulation": articulated_obj.to_dict(),
                "message": "USD libraries not available",
            }
            json_path = output_path.with_suffix(".json")
            save_json(placeholder, json_path)
            return json_path

    def _axis_to_usd(self, axis: Tuple[float, float, float]) -> str:
        """Convert axis tuple to USD axis string."""
        # Find dominant axis
        abs_axis = [abs(a) for a in axis]
        max_idx = abs_axis.index(max(abs_axis))

        if max_idx == 0:
            return "X"
        elif max_idx == 1:
            return "Y"
        else:
            return "Z"

    def _merge_articulations_into_scene(
        self,
        ctx: JobContext,
        scene_path: Path,
        articulated_objects: List[ArticulatedObject],
        output_path: Path,
    ) -> Path:
        """Merge articulated objects back into the main scene USD."""
        try:
            from pxr import Usd, UsdGeom, Sdf

            # Copy scene to output
            shutil.copy(scene_path, output_path)

            # Open for editing
            stage = Usd.Stage.Open(str(output_path))

            # Create articulations group
            art_group_path = "/World/Articulations"
            if not stage.GetPrimAtPath(art_group_path):
                UsdGeom.Xform.Define(stage, art_group_path)

            # Reference each articulated object
            for art_obj in articulated_objects:
                if art_obj.usd_path and Path(art_obj.usd_path).exists():
                    ref_path = f"{art_group_path}/{art_obj.object_id}"
                    ref_prim = stage.DefinePrim(ref_path)
                    ref_prim.GetReferences().AddReference(art_obj.usd_path)

                    ctx.logger.info(f"Added articulated object to scene: {art_obj.object_id}")

            stage.Save()
            return output_path

        except ImportError:
            ctx.logger.error("USD libraries not available for scene merging")
            return scene_path

    def _count_joint_types(self, objects: List[ArticulatedObject]) -> Dict[str, int]:
        """Count joint types across all articulated objects."""
        counts: Dict[str, int] = {}
        for obj in objects:
            for joint in obj.joints:
                jtype = joint.joint_type.value
                counts[jtype] = counts.get(jtype, 0) + 1
        return counts

    def _count_articulation_types(self, objects: List[ArticulatedObject]) -> Dict[str, int]:
        """Count articulation types across all objects."""
        counts: Dict[str, int] = {}
        for obj in objects:
            atype = obj.articulation_type.value
            counts[atype] = counts.get(atype, 0) + 1
        return counts
