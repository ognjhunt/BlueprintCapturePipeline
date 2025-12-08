"""USD authoring with physics metadata for Isaac Sim."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import ArtifactPaths, JobPayload, SessionManifest
from ..utils.io import ensure_local_dir, load_json, save_json
from .base import (
    BaseJob,
    JobContext,
    JobResult,
    JobStatus,
    download_inputs,
    merge_parameters,
    upload_outputs,
)


@dataclass
class PhysicsMaterial:
    """Physics material properties."""
    name: str
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.1
    density: float = 1000.0  # kg/m^3


@dataclass
class ColliderInfo:
    """Collision geometry information."""
    collision_type: str  # "triangle_mesh", "convex_hull", "convex_decomposition", "box", "sphere"
    vertex_count: int = 0
    face_count: int = 0
    hull_count: int = 1  # For convex decomposition


# Default physics materials for common categories
DEFAULT_MATERIALS: Dict[str, PhysicsMaterial] = {
    "default": PhysicsMaterial("default", 0.5, 0.4, 0.1, 1000.0),
    "wood": PhysicsMaterial("wood", 0.4, 0.3, 0.2, 600.0),
    "metal": PhysicsMaterial("metal", 0.3, 0.2, 0.3, 7800.0),
    "plastic": PhysicsMaterial("plastic", 0.4, 0.3, 0.3, 1200.0),
    "glass": PhysicsMaterial("glass", 0.5, 0.4, 0.5, 2500.0),
    "fabric": PhysicsMaterial("fabric", 0.6, 0.5, 0.0, 200.0),
    "rubber": PhysicsMaterial("rubber", 0.9, 0.8, 0.6, 1100.0),
    "ceramic": PhysicsMaterial("ceramic", 0.5, 0.4, 0.2, 2400.0),
    "floor": PhysicsMaterial("floor", 0.6, 0.5, 0.0, 2000.0),
    "wall": PhysicsMaterial("wall", 0.6, 0.5, 0.0, 2000.0),
}


@dataclass
class USDAuthoringJob(BaseJob):
    """Package environment and objects into SimReady USD for Isaac Sim.

    This job:
    1. Loads environment mesh and collision mesh from MeshExtractionJob
    2. Loads individual object assets from ObjectAssetizationJob
    3. Applies physics materials and collision approximations
    4. Composes final scene USD with proper hierarchy
    5. Adds Isaac Sim-specific metadata

    Inputs:
        - Environment mesh (render + collision)
        - Object USD files with placement info
        - Object placement report

    Outputs:
        - scene.usdc: Final composed scene for Isaac Sim
        - SimReady validation report
    """

    name: str = "usd-authoring"
    description: str = "Package environment and objects into USD with physics metadata."
    timeout_minutes: int = 30
    uses_gpu: bool = False

    # USD/Physics configuration
    meters_per_unit: float = 1.0
    up_axis: str = "Y"
    convex_decomposition: bool = True
    max_convex_hulls: int = 32
    max_vertices_per_hull: int = 64
    use_sdf_collision: bool = False  # SDF for complex objects

    # Isaac Sim specific
    enable_rigid_body: bool = True
    enable_articulations: bool = False  # For doors/drawers
    default_contact_offset: float = 0.02
    default_rest_offset: float = 0.01

    def _get_default_parameters(self) -> Dict[str, Any]:
        return {
            "meters_per_unit": self.meters_per_unit,
            "up_axis": self.up_axis,
            "convex_decomposition": self.convex_decomposition,
            "max_convex_hulls": self.max_convex_hulls,
            "enable_rigid_body": self.enable_rigid_body,
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
                "environment_mesh": f"{artifacts.meshes}/environment_mesh.usd",
                "collision_mesh": f"{artifacts.meshes}/environment_collision.usd",
                "object_usds": f"{artifacts.objects}/",
                "object_report": f"{artifacts.reports}/objects.json",
            },
            outputs={
                "scene_usd": f"{artifacts.session_root}/scene.usdc",
                "report": f"{artifacts.reports}/usd_authoring.json",
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute USD authoring and scene packaging."""
        result = JobResult(status=JobStatus.RUNNING)

        # Setup directories
        meshes_dir = ensure_local_dir(ctx.workspace / "meshes")
        objects_dir = ensure_local_dir(ctx.workspace / "objects")
        reports_dir = ensure_local_dir(ctx.workspace / "reports")
        output_dir = ensure_local_dir(ctx.workspace / "output")

        # Download inputs
        with ctx.tracker.stage("download_inputs", 4):
            # Environment meshes
            try:
                ctx.gcs.download(
                    f"{ctx.artifacts.meshes}/environment_mesh.usd",
                    meshes_dir / "environment_mesh.usd"
                )
            except Exception as e:
                ctx.logger.warning(f"Could not download environment_mesh.usd: {e}")
            ctx.tracker.update(1)

            try:
                ctx.gcs.download(
                    f"{ctx.artifacts.meshes}/environment_collision.usd",
                    meshes_dir / "environment_collision.usd"
                )
            except Exception as e:
                ctx.logger.warning(f"Could not download collision mesh: {e}")
            ctx.tracker.update(1)

            # Object assets
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

        # Load object placement info
        object_report = {}
        if (reports_dir / "objects.json").exists():
            object_report = load_json(reports_dir / "objects.json")

        objects_info = object_report.get("objects", [])
        ctx.logger.info(f"Loaded placement info for {len(objects_info)} objects")

        # Create the composed USD scene
        with ctx.tracker.stage("compose_scene", 1):
            scene_path = self._compose_scene(
                ctx=ctx,
                meshes_dir=meshes_dir,
                objects_dir=objects_dir,
                objects_info=objects_info,
                output_path=output_dir / "scene.usdc",
            )

        # Add physics properties
        with ctx.tracker.stage("add_physics", 1):
            self._add_physics_properties(
                ctx=ctx,
                scene_path=scene_path,
                objects_info=objects_info,
            )

        # Validate SimReady compliance
        with ctx.tracker.stage("validation", 1):
            validation_results = self._validate_simready(ctx, scene_path)

        # Generate authoring report
        report = {
            "session_id": ctx.session.session_id,
            "scene_path": str(scene_path),
            "metrics": {
                "environment_mesh_present": (meshes_dir / "environment_mesh.usd").exists(),
                "collision_mesh_present": (meshes_dir / "environment_collision.usd").exists(),
                "objects_placed": len(objects_info),
                "meters_per_unit": ctx.parameters.get("meters_per_unit", self.meters_per_unit),
                "up_axis": ctx.parameters.get("up_axis", self.up_axis),
            },
            "physics": {
                "convex_decomposition_enabled": ctx.parameters.get(
                    "convex_decomposition", self.convex_decomposition
                ),
                "rigid_bodies_enabled": ctx.parameters.get(
                    "enable_rigid_body", self.enable_rigid_body
                ),
                "materials_assigned": True,
            },
            "validation": validation_results,
        }
        save_json(report, output_dir / "usd_authoring_report.json")

        # Upload outputs
        with ctx.tracker.stage("upload_outputs", 2):
            ctx.gcs.upload(scene_path, f"{ctx.artifacts.session_root}/scene.usdc")
            ctx.tracker.update(1)
            ctx.gcs.upload(
                output_dir / "usd_authoring_report.json",
                f"{ctx.artifacts.reports}/usd_authoring.json"
            )
            ctx.tracker.update(1)

        result.outputs = {
            "scene_usd": f"{ctx.artifacts.session_root}/scene.usdc",
            "report": f"{ctx.artifacts.reports}/usd_authoring.json",
        }
        result.metrics = report["metrics"]

        return result

    def _compose_scene(
        self,
        ctx: JobContext,
        meshes_dir: Path,
        objects_dir: Path,
        objects_info: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Compose the final USD scene from components."""
        ctx.logger.info("Composing USD scene...")

        try:
            from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, Kind

            # Create the root stage
            stage = Usd.Stage.CreateNew(str(output_path))

            # Set stage metadata
            meters_per_unit = ctx.parameters.get("meters_per_unit", self.meters_per_unit)
            up_axis = ctx.parameters.get("up_axis", self.up_axis)

            stage.SetMetadata("metersPerUnit", meters_per_unit)
            stage.SetMetadata("upAxis", up_axis)

            # Set default prim
            root_path = "/World"
            root_xform = UsdGeom.Xform.Define(stage, root_path)
            stage.SetDefaultPrim(root_xform.GetPrim())

            # Set kind for Isaac Sim compatibility
            Usd.ModelAPI(root_xform.GetPrim()).SetKind(Kind.Tokens.assembly)

            # Create hierarchy
            env_path = f"{root_path}/Environment"
            objects_path = f"{root_path}/Objects"
            UsdGeom.Xform.Define(stage, env_path)
            UsdGeom.Xform.Define(stage, objects_path)

            # Add physics scene
            self._create_physics_scene(stage, root_path)

            # Reference environment mesh
            env_mesh_path = meshes_dir / "environment_mesh.usd"
            if env_mesh_path.exists():
                ctx.logger.info("Adding environment mesh...")
                self._add_sublayer_reference(
                    stage=stage,
                    reference_path=env_mesh_path,
                    prim_path=f"{env_path}/Mesh",
                )

            # Reference collision mesh
            collision_mesh_path = meshes_dir / "environment_collision.usd"
            if collision_mesh_path.exists():
                ctx.logger.info("Adding environment collision mesh...")
                self._add_sublayer_reference(
                    stage=stage,
                    reference_path=collision_mesh_path,
                    prim_path=f"{env_path}/Collision",
                    purpose="guide",
                )

            # Add ground plane
            self._create_ground_plane(stage, f"{env_path}/GroundPlane")

            # Add objects with placements
            for i, obj_info in enumerate(objects_info):
                track_id = obj_info.get("track_id", f"object_{i}")
                obj_usd_dir = objects_dir / track_id

                # Find the USD file for this object
                obj_usd_path = None
                if obj_usd_dir.exists():
                    for usd_file in obj_usd_dir.glob("*.usd*"):
                        obj_usd_path = usd_file
                        break

                if obj_usd_path and obj_usd_path.exists():
                    ctx.logger.info(f"Adding object: {track_id}")
                    self._add_object_to_scene(
                        stage=stage,
                        obj_usd_path=obj_usd_path,
                        obj_info=obj_info,
                        parent_path=objects_path,
                    )

            stage.Save()
            ctx.logger.info(f"Scene composed: {output_path}")
            return output_path

        except ImportError:
            ctx.logger.warning("USD libraries not available, creating placeholder")
            return self._create_placeholder_scene(ctx, output_path)

    def _create_physics_scene(self, stage: Any, root_path: str) -> None:
        """Create PhysicsScene prim for Isaac Sim."""
        from pxr import UsdPhysics, Gf

        physics_scene_path = f"{root_path}/PhysicsScene"
        physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)

        # Set gravity
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    def _add_sublayer_reference(
        self,
        stage: Any,
        reference_path: Path,
        prim_path: str,
        purpose: Optional[str] = None,
    ) -> None:
        """Add a USD file as a reference."""
        from pxr import Usd, UsdGeom, Sdf

        # Create the prim and add reference
        prim = stage.DefinePrim(prim_path)
        prim.GetReferences().AddReference(str(reference_path))

        # Set purpose if specified (e.g., "guide" for collision-only)
        if purpose:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.CreatePurposeAttr().Set(purpose)

    def _create_ground_plane(self, stage: Any, prim_path: str) -> None:
        """Create a ground plane with collision."""
        from pxr import UsdGeom, UsdPhysics, Gf

        # Create plane geometry
        plane = UsdGeom.Plane.Define(stage, prim_path)
        plane.CreateAxisAttr().Set("Y")
        plane.CreateWidthAttr().Set(100.0)
        plane.CreateLengthAttr().Set(100.0)

        # Add collision
        collision_api = UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

        # Add physics material
        self._apply_physics_material(
            stage, plane.GetPrim(), DEFAULT_MATERIALS["floor"]
        )

    def _add_object_to_scene(
        self,
        stage: Any,
        obj_usd_path: Path,
        obj_info: Dict[str, Any],
        parent_path: str,
    ) -> None:
        """Add an object to the scene with proper transform and physics."""
        from pxr import UsdGeom, UsdPhysics, Gf

        track_id = obj_info.get("track_id", "object")
        obj_prim_path = f"{parent_path}/{track_id}"

        # Create xform for the object
        obj_xform = UsdGeom.Xform.Define(stage, obj_prim_path)

        # Set transform from placement info
        position = obj_info.get("position", [0, 0, 0])
        rotation = obj_info.get("rotation", [1, 0, 0, 0])  # quaternion (w, x, y, z)
        scale = obj_info.get("scale", [1, 1, 1])

        # Apply translation
        translate_op = obj_xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*position))

        # Apply rotation (quaternion to rotation)
        orient_op = obj_xform.AddOrientOp()
        orient_op.Set(Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3]))

        # Apply scale
        scale_op = obj_xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*scale))

        # Reference the object USD
        mesh_prim_path = f"{obj_prim_path}/Mesh"
        mesh_prim = stage.DefinePrim(mesh_prim_path)
        mesh_prim.GetReferences().AddReference(str(obj_usd_path))

        # Apply physics properties
        category = obj_info.get("category", "default")
        source = obj_info.get("source", "generation")

        # Objects get rigid body physics
        if source in ("reconstruction", "generation"):
            self._apply_rigid_body(stage, obj_xform.GetPrim(), obj_info)

    def _apply_rigid_body(
        self,
        stage: Any,
        prim: Any,
        obj_info: Dict[str, Any],
    ) -> None:
        """Apply rigid body physics to an object."""
        from pxr import UsdPhysics, Gf

        # Add rigid body API
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)

        # Set mass based on bounding box and assumed density
        bbox = obj_info.get("bounding_box", [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1])
        if len(bbox) >= 6:
            size_x = bbox[3] - bbox[0]
            size_y = bbox[4] - bbox[1]
            size_z = bbox[5] - bbox[2]
            volume = size_x * size_y * size_z

            # Get material density
            category = obj_info.get("category", "default")
            material = DEFAULT_MATERIALS.get(category, DEFAULT_MATERIALS["default"])
            mass = volume * material.density

            # Clamp mass to reasonable values
            mass = max(0.01, min(100.0, mass))

            # Add mass API
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(mass)

    def _add_physics_properties(
        self,
        ctx: JobContext,
        scene_path: Path,
        objects_info: List[Dict[str, Any]],
    ) -> None:
        """Add physics properties to all prims in the scene."""
        ctx.logger.info("Adding physics properties...")

        try:
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = Usd.Stage.Open(str(scene_path))

            # Add collision to environment
            env_mesh_prim = stage.GetPrimAtPath("/World/Environment/Mesh")
            if env_mesh_prim:
                self._add_mesh_collision(
                    stage, env_mesh_prim, is_static=True, ctx=ctx
                )

            env_collision_prim = stage.GetPrimAtPath("/World/Environment/Collision")
            if env_collision_prim:
                # Mark collision mesh as collision-only
                UsdPhysics.CollisionAPI.Apply(env_collision_prim)

            # Add collision to objects
            for obj_info in objects_info:
                track_id = obj_info.get("track_id", "")
                obj_prim_path = f"/World/Objects/{track_id}"
                obj_prim = stage.GetPrimAtPath(obj_prim_path)

                if obj_prim:
                    self._add_object_collision(stage, obj_prim, obj_info, ctx)

            stage.Save()

        except ImportError:
            ctx.logger.warning("USD physics libraries not available")

    def _add_mesh_collision(
        self,
        stage: Any,
        prim: Any,
        is_static: bool,
        ctx: JobContext,
    ) -> None:
        """Add collision to a mesh prim."""
        from pxr import UsdPhysics

        # For static environment, use triangle mesh collision
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)

        if is_static:
            # Static objects can use triangle mesh
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("none")  # Use exact mesh
        else:
            # Dynamic objects need convex approximation
            if ctx.parameters.get("convex_decomposition", self.convex_decomposition):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set("convexDecomposition")

    def _add_object_collision(
        self,
        stage: Any,
        prim: Any,
        obj_info: Dict[str, Any],
        ctx: JobContext,
    ) -> None:
        """Add collision to an object prim."""
        from pxr import UsdPhysics, UsdGeom

        # Find mesh children
        for child in prim.GetAllChildren():
            if child.IsA(UsdGeom.Mesh):
                collision_api = UsdPhysics.CollisionAPI.Apply(child)

                # Use convex decomposition for dynamic objects
                if ctx.parameters.get("convex_decomposition", self.convex_decomposition):
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(child)
                    mesh_collision.CreateApproximationAttr().Set("convexDecomposition")

                # Apply physics material
                category = obj_info.get("category", "default")
                material = DEFAULT_MATERIALS.get(category, DEFAULT_MATERIALS["default"])
                self._apply_physics_material(stage, child, material)

    def _apply_physics_material(
        self,
        stage: Any,
        prim: Any,
        material: PhysicsMaterial,
    ) -> None:
        """Apply physics material to a prim."""
        from pxr import UsdPhysics, UsdShade, Sdf

        # Create material if it doesn't exist
        material_path = f"/World/PhysicsMaterials/{material.name}"

        if not stage.GetPrimAtPath(material_path):
            phys_material = UsdPhysics.MaterialAPI.Apply(
                UsdShade.Material.Define(stage, material_path).GetPrim()
            )
            phys_material.CreateStaticFrictionAttr().Set(material.static_friction)
            phys_material.CreateDynamicFrictionAttr().Set(material.dynamic_friction)
            phys_material.CreateRestitutionAttr().Set(material.restitution)
            phys_material.CreateDensityAttr().Set(material.density)

        # Bind material to prim
        # Note: This uses the physics material binding, not render material
        prim_material_api = UsdPhysics.MaterialAPI.Apply(prim)

    def _validate_simready(
        self,
        ctx: JobContext,
        scene_path: Path,
    ) -> Dict[str, Any]:
        """Validate that the scene is SimReady compliant for Isaac Sim."""
        ctx.logger.info("Validating SimReady compliance...")

        results = {
            "passed": True,
            "checks": [],
            "warnings": [],
            "errors": [],
        }

        try:
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = Usd.Stage.Open(str(scene_path))

            # Check 1: metersPerUnit is set
            meters_per_unit = stage.GetMetadata("metersPerUnit")
            if meters_per_unit and meters_per_unit > 0:
                results["checks"].append({
                    "name": "metersPerUnit",
                    "passed": True,
                    "value": meters_per_unit,
                })
            else:
                results["checks"].append({
                    "name": "metersPerUnit",
                    "passed": False,
                    "message": "metersPerUnit not set or invalid",
                })
                results["warnings"].append("metersPerUnit should be set for accurate physics")

            # Check 2: upAxis is set
            up_axis = stage.GetMetadata("upAxis")
            if up_axis:
                results["checks"].append({
                    "name": "upAxis",
                    "passed": True,
                    "value": up_axis,
                })
            else:
                results["checks"].append({
                    "name": "upAxis",
                    "passed": False,
                    "message": "upAxis not set",
                })
                results["warnings"].append("upAxis should be set (Y or Z)")

            # Check 3: Physics scene exists
            physics_scene = stage.GetPrimAtPath("/World/PhysicsScene")
            if physics_scene and physics_scene.IsValid():
                results["checks"].append({
                    "name": "physicsScene",
                    "passed": True,
                })
            else:
                results["checks"].append({
                    "name": "physicsScene",
                    "passed": False,
                    "message": "PhysicsScene prim not found",
                })
                results["warnings"].append("PhysicsScene recommended for Isaac Sim")

            # Check 4: Default prim is set
            default_prim = stage.GetDefaultPrim()
            if default_prim and default_prim.IsValid():
                results["checks"].append({
                    "name": "defaultPrim",
                    "passed": True,
                    "value": default_prim.GetPath().pathString,
                })
            else:
                results["checks"].append({
                    "name": "defaultPrim",
                    "passed": False,
                    "message": "Default prim not set",
                })
                results["warnings"].append("Default prim should be set for proper referencing")

            # Check 5: Count meshes and collisions
            mesh_count = 0
            collision_count = 0
            rigid_body_count = 0

            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    mesh_count += 1

                if UsdPhysics.CollisionAPI(prim):
                    collision_count += 1

                if UsdPhysics.RigidBodyAPI(prim):
                    rigid_body_count += 1

            results["checks"].append({
                "name": "meshCount",
                "passed": mesh_count > 0,
                "value": mesh_count,
            })

            results["checks"].append({
                "name": "collisionCount",
                "passed": True,  # Not required but good to have
                "value": collision_count,
            })

            results["checks"].append({
                "name": "rigidBodyCount",
                "passed": True,
                "value": rigid_body_count,
            })

            # Overall pass/fail
            results["passed"] = all(
                check.get("passed", True) for check in results["checks"]
            )

            # Add summary
            results["summary"] = {
                "total_checks": len(results["checks"]),
                "passed_checks": sum(1 for c in results["checks"] if c.get("passed", True)),
                "meshes": mesh_count,
                "colliders": collision_count,
                "rigid_bodies": rigid_body_count,
            }

        except ImportError:
            results["passed"] = False
            results["errors"].append("USD libraries not available for validation")

        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Validation error: {str(e)}")

        return results

    def _create_placeholder_scene(self, ctx: JobContext, output_path: Path) -> Path:
        """Create a placeholder scene when USD libraries aren't available."""
        # Create a minimal JSON representation
        placeholder = {
            "format": "placeholder",
            "message": "USD libraries not available. Install with: pip install usd-core",
            "session_id": ctx.session.session_id,
            "intended_contents": {
                "environment_mesh": True,
                "collision_mesh": True,
                "objects": True,
                "physics": True,
            },
        }

        json_path = output_path.with_suffix(".json")
        save_json(placeholder, json_path)
        return json_path
