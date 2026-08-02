"""Isaac Sim 6.0 GPU runner: render a Gaussian-splat scene via RTX ParticleField.

Executes **on the Isaac Sim 6.0 GPU worker**. It opens an Isaac-renderable splat asset
and RTX-renders the eval's free cameras, capturing per-camera PNGs + an MP4 and uploading
them. Preferred input is a pre-authored ``ParticleField3DGaussianSplat`` USD (``--usdc``),
authored locally by ``blueprint_pipeline.particlefield_usd`` — Isaac 6.0 renders that schema
natively (no ncore/3dgrut/NRE).

Hardening applied from an adversarial pre-spend review (avoids the documented black-frame /
zero-frame failure modes on this image):
- author the camera transform with ``Xformable.AddTransformOp().Set`` (XformCommonAPI has no
  matrix transform setter), and author vertical/horizontal aperture so the requested FOV is real;
- assert a ``ParticleField3DGaussianSplat`` prim is present before paying to render;
- force single-GPU (multi-GPU yields empty annotator frames) and enable+log the ParticleField
  render extensions (omni.hydra.rtx / omni.ujitso.* / omni.kit.converter.gsplat);
- capture with the proven ``BasicWriter`` + settle + ``wait_until_complete`` pattern and gate on
  per-frame **pixel variance**, not byte size (a black 1080p PNG can exceed 30 KB).

Run with Isaac's python:
  ./python.sh run_isaac_splat_nurec_render.py --usdc scene_particlefield.usdc \\
      --cameras cameras.json --out-dir /workspace/out

Truth boundary: Isaac RTX render evidence of the captured scene only — not physics,
navigation, control, or robot readiness.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

LEGACY_RESULT_SCHEMA = "isaac_splat_nurec_render_result.v1"
QUALIFICATION_RESULT_SCHEMA = "isaac_splat_nurec_render_result.v3"
PROVIDER_QUALIFICATION_RESULT_SCHEMA = "provider_nurec_isaac_runtime_result.v1"
PARTICLEFIELD_TYPE = "ParticleField3DGaussianSplat"
NUREC_FIELD_TYPE = "OmniNuRecFieldAsset"
_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_USD_PRIM_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _write_json(path: Path, payload: dict) -> None:
    payload = dict(payload)
    if payload.get("schema_version") in {
        QUALIFICATION_RESULT_SCHEMA,
        PROVIDER_QUALIFICATION_RESULT_SCHEMA,
    }:
        payload.pop("isaac_runtime_result_digest", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        payload["isaac_runtime_result_digest"] = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(result_path: Path, base: dict, phase: str, **extra) -> None:
    _write_json(
        result_path, {**base, "status": "running", "provider_runtime_phase": phase, **extra}
    )


def _camera_usd_prim_names(camera_ids: list[str]) -> list[str]:
    """Map portable external IDs to valid, collision-free USD identifiers."""

    names = []
    for camera_id in camera_ids:
        name = re.sub(r"[^A-Za-z0-9_]", "_", camera_id)
        if not name or name[0].isdigit():
            name = "_" + name
        if _USD_PRIM_IDENTIFIER.fullmatch(name) is None:
            raise ValueError("isaac_camera_usd_prim_name_invalid")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("isaac_camera_usd_prim_names_collide")
    return names


def _transcode_ply_to_usd(ply: Path, usd: Path, *, python: str, fmt: str = "lightfield") -> dict:
    # fmt 'lightfield' => ParticleField3DGaussianSplat (matches the validated authoring path).
    cmd = [
        python,
        "-m",
        "threedgrut.export.scripts.transcode",
        str(ply),
        "-o",
        str(usd),
        "--format",
        fmt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "blocked",
            "blockers": ["threedgrut_transcode_exception"],
            "error": repr(exc),
        }
    if proc.returncode != 0 or not usd.is_file():
        return {
            "status": "blocked",
            "blockers": ["threedgrut_transcode_failed"],
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    return {"status": "completed", "usd": str(usd), "bytes": usd.stat().st_size, "format": fmt}


def _load_render_options(cameras_path: Path) -> dict:
    """Optional ``render_options.json`` next to cameras.json (bundle-driven knobs).

    Bundle-side options survive warm pod restarts (container env is fixed at
    create; the bundle is re-fetched every boot), so robot compositing config
    rides here instead of argv/env. Absent file -> empty options.
    """
    options_path = Path(cameras_path).parent / "render_options.json"
    if not options_path.is_file():
        return {}
    try:
        payload = json.loads(options_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _robot_usd_candidates(value: str) -> list:
    """Resolve a robot USD reference: assets-root expansion + Isaac-6 short-path variants.

    Mirrors the kitchen parity runner's proven resolution (relative asset paths
    resolve against the worker's Isaac assets root; '/Isaac/Robots/Unitree/G1/'
    also ships as the short '/Unitree/G1/' layout on Isaac 6 workers).
    """
    raw = str(value or "").strip()
    if not raw:
        return []
    resolved = [raw]
    if "://" not in raw and not raw.startswith("/") and not raw.startswith("omniverse:"):
        try:
            from isaacsim.storage.native import get_assets_root_path  # type: ignore

            root = get_assets_root_path()
            if root:
                resolved.insert(0, root.rstrip("/") + "/" + raw.lstrip("/"))
        except Exception:  # noqa: BLE001
            pass
    out = []
    for cand in resolved:
        for variant in (cand, cand.replace("/Isaac/Robots/Unitree/G1/", "/Unitree/G1/")):
            if variant not in out:
                out.append(variant)
    return out


def _composite_robot(stage, options: dict, *, Gf, UsdGeom, Sdf) -> dict:
    """Reference the robot USD at the solved stance pose (visual compositing only).

    ``options`` carries ``robot_usd`` (asset path/URI) and ``robot_pose``
    ``[x, y, z, yaw_rad]`` — the pelvis-frame stance the placement preflight
    validated. Returns a report dict; never raises (a missing robot asset is a
    recorded blocker, not a wasted render).
    """
    robot_usd = str(options.get("robot_usd") or "").strip()
    pose = options.get("robot_pose")
    if not robot_usd or not isinstance(pose, (list, tuple)) or len(pose) != 4:
        return {"requested": False}
    report = {"requested": True, "robot_usd": robot_usd, "robot_pose": [float(v) for v in pose]}
    try:
        from pxr import Usd  # type: ignore

        prim_path = str(options.get("robot_prim_path") or "/World/RobotVisual")
        chosen = None
        tried = []
        pre_authored = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if pre_authored and pre_authored.IsValid() and pre_authored.HasAuthoredReferences():
            # The reference was authored into the stage BEFORE open (runtime-added
            # instanceable references can miss the Fabric render index entirely).
            prim = pre_authored
            composed = any(
                True for child in Usd.PrimRange(prim) if child.GetPath() != prim.GetPath()
            )
            if composed:
                chosen = "(pre_authored_in_stage)"
                report["pre_authored"] = True
        else:
            prim = stage.DefinePrim(Sdf.Path(prim_path), "Xform")
            for cand in _robot_usd_candidates(robot_usd):
                tried.append(cand)
                prim.GetReferences().ClearReferences()
                prim.GetReferences().AddReference(cand)
                composed = any(
                    True for child in Usd.PrimRange(prim) if child.GetPath() != prim.GetPath()
                )
                if composed:
                    chosen = cand
                    break
        report["candidates_tried"] = tried
        if chosen is None:
            prim.GetReferences().ClearReferences()
            stage.RemovePrim(Sdf.Path(prim_path))
            report.update(composited=False, blocker="robot_usd_unresolvable")
            return report
        x, y, z, yaw = (float(v) for v in pose)
        import math as _math

        matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), _math.degrees(yaw)))
        matrix.SetTranslateOnly(Gf.Vec3d(x, y, z))
        xformable = UsdGeom.Xformable(prim)
        if not report.get("pre_authored"):
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(matrix)
        child_count = sum(1 for child in Usd.PrimRange(prim)) - 1
        report.update(
            composited=True,
            resolved_usd=chosen,
            prim_path=prim_path,
            composed_prim_count=child_count,
        )
        return report
    except Exception as exc:  # noqa: BLE001
        report.update(composited=False, blocker="robot_composite_exception", error=repr(exc)[:400])
        return report


def _ensure_robot_only_lights(stage, lights_path: str, *, Sdf, UsdGeom, UsdLux):
    """Return a hidden light rig for the robot evidence renders.

    Provider-authored NuRec packages are often self-emissive and contain no
    lights. A referenced mesh robot then renders black when the splat is
    hidden. Reuse an authored rig when present; otherwise author a deterministic
    dome+distant rig at runtime and enable it for both the scene-plus-robot and
    robot-only evidence passes. The exact source package on disk is never edited.
    """

    requested_path = str(lights_path or "/World/BlueprintRobotOnlyLights").strip()
    path = Sdf.Path(requested_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError("robot_only_lights_prim_path_invalid")
    existing = stage.GetPrimAtPath(path)
    authored = not (existing and existing.IsValid())
    if authored:
        root = UsdGeom.Xform.Define(stage, path)
        dome = UsdLux.DomeLight.Define(stage, path.AppendChild("Dome"))
        dome.CreateIntensityAttr(400.0)
        distant = UsdLux.DistantLight.Define(stage, path.AppendChild("Distant"))
        distant.CreateIntensityAttr(2500.0)
        prim = root.GetPrim()
    else:
        prim = existing
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim, {
        "lights_path": str(path),
        "authored_for_robot_only_pass": authored,
        "enabled_for_robot_scene_and_robot_only_passes": True,
        "dome_intensity": 400.0 if authored else None,
        "distant_intensity": 2500.0 if authored else None,
        "claim_boundary": "render_lighting_support_only_not_scene_or_task_evidence",
    }


def _author_robot_evidence_material(stage, robot_prim, *, Sdf, UsdShade) -> dict:
    """Bind a deterministic preview material for placement evidence renders.

    Some Isaac robot assets resolve their geometry but not their remote MDL
    materials in a bounded headless canary.  A stronger inherited preview
    material keeps the exact referenced geometry visible without editing the
    robot asset or pretending that the fallback proves production appearance.
    """

    path = Sdf.Path("/World/BlueprintRobotEvidenceMaterial")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.72, 0.74, 0.78))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(robot_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    return {
        "material_path": str(path),
        "authored": True,
        "binding_strength": "strongerThanDescendants",
        "claim_boundary": "render_material_support_only_not_robot_asset_or_task_evidence",
    }


def _exclude_robot_from_environment_physics_probe(stage, robot_report: dict, *, Sdf) -> None:
    """Keep the environment collision probe independent of the visual robot.

    The Franka reference contains articulation and collision APIs.  Its solved
    stance can coincide with the precommitted ground-probe point, so leaving it
    active changes a scene-collision test into an accidental robot-contact
    test.  Robot placement is visual evidence in this lane; deactivate it only
    after all robot renders and before Physics is initialized.
    """

    path = str(robot_report.get("prim_path") or "")
    if not robot_report.get("composited") or not path:
        return
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim or not prim.IsValid():
        robot_report["excluded_from_environment_physics_probe"] = False
        return
    prim.SetActive(False)
    robot_report.update(
        excluded_from_environment_physics_probe=True,
        physics_probe_claim_boundary=(
            "visual_robot_excluded_so_probe_measures_provider_environment_collision_only"
        ),
    )


def _camera_xform(Gf, position, target, up):
    """Camera local-to-world transform (USD camera looks down local -Z, +Y up)."""
    eye = Gf.Vec3d(*[float(x) for x in position])
    center = Gf.Vec3d(*[float(x) for x in target])
    up_v = Gf.Vec3d(*[float(x) for x in (up or [0.0, 0.0, 1.0])])
    view = Gf.Matrix4d().SetLookAt(eye, center, up_v)  # world -> camera
    return view.GetInverse()  # camera -> world (the prim's local transform)


def _pixel_std(png_path: Path) -> float:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        arr = np.asarray(Image.open(str(png_path)).convert("RGB"), dtype="float32")
        return float(arr.std())
    except Exception:  # noqa: BLE001
        # fall back to a coarse byte-size heuristic if numpy/PIL are unavailable
        try:
            return 100.0 if png_path.stat().st_size > 60000 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0


def _pixel_stats(png_path: Path) -> tuple[int, int, float, float]:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    with Image.open(str(png_path)) as image:
        image.load()
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype="float32")
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise ValueError("render_pixels_invalid")
    height, width = array.shape[:2]
    return int(width), int(height), float(array.mean()), float(array.std())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _is_sha256_digest(value) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


def _is_image_digest(value) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", value))


def _select_ground_surface(collision_prims, *, requested_path="", declared_height=None):
    """Choose an existing package collider without manufacturing a ground plane.

    Automatic selection is intentionally conservative: a collider must either
    be named like a floor/ground surface or have a broad, thin horizontal world
    bound. A caller may bind an exact collider path, but a combined room mesh
    must also declare the measured probe height because its bounding-box top is
    not necessarily the floor.
    """
    requested = str(requested_path or "").strip()
    candidates = [
        item
        for item in collision_prims
        if item.get("active") is True and item.get("static") is True
    ]
    if requested:
        candidates = [item for item in candidates if item.get("prim_path") == requested]
        if not candidates:
            return None, "declared_ground_collider_not_active"
    ranked = []
    for item in candidates:
        bounds = item.get("world_bounds") or {}
        low = bounds.get("min")
        high = bounds.get("max")
        if (
            not isinstance(low, list)
            or not isinstance(high, list)
            or len(low) != 3
            or len(high) != 3
        ):
            continue
        values = [*low, *high]
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values
        ):
            continue
        dx, dy, dz = (float(high[index]) - float(low[index]) for index in range(3))
        name = str(item.get("prim_path") or "").lower()
        semantic_floor = "floor" in name or "ground" in name
        broad_thin_surface = dx >= 0.25 and dy >= 0.25 and dz <= max(0.25, 0.25 * min(dx, dy))
        if not requested and not semantic_floor and not broad_thin_surface:
            continue
        if requested and declared_height is None and not semantic_floor and not broad_thin_surface:
            return None, "combined_ground_collider_requires_declared_height"
        surface = dict(item)
        surface["probe_height_m"] = (
            float(declared_height) if declared_height is not None else float(high[2])
        )
        surface["selection_reason"] = (
            "declared_active_collider"
            if requested
            else "floor_semantic"
            if semantic_floor
            else "broad_thin_horizontal_bound"
        )
        ranked.append((dx * dy, surface))
    if not ranked:
        return None, "ground_contact_surface_not_identified"
    ranked.sort(key=lambda row: (-row[0], str(row[1].get("prim_path") or "")))
    return ranked[0][1], None


def _classify_physics_probe(
    *,
    ground_surface,
    requested_steps,
    executed_steps,
    initial_position,
    final_position,
    contact_event_count,
    errors,
):
    ground_height = (
        ground_surface.get("probe_height_m") if isinstance(ground_surface, dict) else None
    )
    positions_valid = (
        isinstance(initial_position, list)
        and len(initial_position) == 3
        and isinstance(final_position, list)
        and len(final_position) == 3
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in [*initial_position, *final_position]
        )
    )
    fell_through = None
    if (
        positions_valid
        and isinstance(ground_height, (int, float))
        and math.isfinite(float(ground_height))
    ):
        fell_through = float(final_position[2]) < float(ground_height) - 0.1
    return {
        "ground_contact_surface_present": bool(ground_surface),
        "ground_surface": ground_surface,
        "requested_steps": int(requested_steps),
        "steps_executed": int(executed_steps),
        "initial_position_xyz": initial_position,
        "final_position_xyz": final_position,
        "live_rigid_body_pose_observed": positions_valid,
        "test_body_fell_through_floor": fell_through,
        "contact_event_count": int(contact_event_count),
        "probe_configuration": {
            "test_body": {
                "shape": "cube",
                "size_m": 0.1,
                "mass_kg": 1.0,
                "spawn_height_above_ground_m": 0.5,
            },
            "gravity_m_s2": -9.81,
            "physics_dt_seconds": 1.0 / 60.0,
        },
        "errors": list(errors),
        "claim_boundary": (
            "One bounded test-body contact probe verifies physics presence only; it does not prove "
            "navigation, manipulation, task success, physical transfer, or deployment readiness."
        ),
    }


def _qualification_blockers(*, package_digest, stage, physics_probe, cameras):
    blockers = []
    if not _is_sha256_digest(package_digest):
        blockers.append("isaac_package_digest_invalid")
    if stage.get("meters_per_unit") != 1.0 or stage.get("up_axis") != "Z":
        blockers.append("isaac_stage_units_invalid")
    if stage.get("transforms_valid") is not True:
        blockers.append("isaac_stage_transforms_invalid")
    if stage.get("dependency_inspection_available") is not True:
        blockers.append("isaac_dependency_inspection_unavailable")
    if stage.get("missing_asset_count") != 0:
        blockers.append("isaac_missing_assets")
    if int(stage.get("particlefield_prim_count") or 0) < 1:
        blockers.append("isaac_particlefield_not_loaded")
    if int(stage.get("active_collision_prim_count") or 0) < 1:
        blockers.append("isaac_collision_geometry_inactive")
    if stage.get("obvious_scale_mismatch_detected") is not False:
        blockers.append("isaac_obvious_scale_mismatch")
    expected_prims = stage.get("expected_prim_paths")
    if (
        not isinstance(expected_prims, dict)
        or not expected_prims.get("appearance")
        or not expected_prims.get("collision")
    ):
        blockers.append("isaac_expected_prims_not_loaded")
    if physics_probe.get("ground_contact_surface_present") is not True:
        blockers.append("isaac_ground_contact_surface_missing")
    if int(physics_probe.get("steps_executed") or 0) < 2:
        blockers.append("isaac_physics_probe_not_executed")
    if physics_probe.get("live_rigid_body_pose_observed") is not True:
        blockers.append("isaac_test_body_pose_unavailable")
    if physics_probe.get("test_body_fell_through_floor") is not False:
        blockers.append("isaac_test_body_fell_through_floor")
    if int(physics_probe.get("contact_event_count") or 0) < 1:
        blockers.append("isaac_test_body_contact_not_observed")
    if not cameras:
        blockers.append("isaac_fixed_camera_renders_missing")
    for index, camera in enumerate(cameras):
        pixel_std = camera.get("pixel_std")
        if (
            camera.get("nonblank") is not True
            or isinstance(pixel_std, bool)
            or not isinstance(pixel_std, (int, float))
            or not math.isfinite(float(pixel_std))
            or float(pixel_std) <= 3.0
            or not _is_sha256_digest(camera.get("digest"))
        ):
            blockers.append(f"isaac_fixed_render_invalid:{index}")
    return sorted(set(blockers))


def _inspect_qualification_stage(
    stage, stage_path, particlefields, *, Usd, UsdGeom, UsdPhysics, UsdUtils
):
    invalid_transforms = []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        try:
            if not prim.IsA(UsdGeom.Xformable):
                continue
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            values = [float(matrix[row][column]) for row in range(4) for column in range(4)]
            if not all(math.isfinite(value) for value in values):
                invalid_transforms.append(str(prim.GetPath()))
        except Exception:  # noqa: BLE001
            invalid_transforms.append(str(prim.GetPath()))

    dependency_inspection_available = False
    unresolved_assets = []
    dependency_error = None
    try:
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(stage_path))
        unresolved_assets = sorted(str(value) for value in unresolved)
        dependency_inspection_available = True
    except Exception as exc:  # noqa: BLE001
        dependency_error = repr(exc)[:400]

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    collision_prims = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        ancestor = prim
        rigid_body_ancestor = False
        while ancestor and ancestor.IsValid() and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_ancestor = True
                break
            ancestor = ancestor.GetParent()
        record = {
            "prim_path": str(prim.GetPath()),
            "active": enabled is not False,
            "static": not rigid_body_ancestor,
        }
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not aligned.IsEmpty():
                low = [float(aligned.GetMin()[index]) for index in range(3)]
                high = [float(aligned.GetMax()[index]) for index in range(3)]
                if all(math.isfinite(value) for value in [*low, *high]):
                    record["world_bounds"] = {"min": low, "max": high}
                else:
                    record["bounds_error"] = "nonfinite_world_bounds"
        except Exception as exc:  # noqa: BLE001
            record["bounds_error"] = repr(exc)[:240]
        collision_prims.append(record)
    active = [item for item in collision_prims if item["active"]]
    bounded = [item["world_bounds"] for item in active if "world_bounds" in item]
    stage_bounds = None
    obvious_scale_mismatch = None
    if bounded:
        low = [min(float(bounds["min"][index]) for bounds in bounded) for index in range(3)]
        high = [max(float(bounds["max"][index]) for bounds in bounded) for index in range(3)]
        extents = [high[index] - low[index] for index in range(3)]
        stage_bounds = {"min": low, "max": high, "extents_m": extents}
        maximum_extent = max(extents)
        obvious_scale_mismatch = maximum_extent < 0.25 or maximum_extent > 1000.0
    return {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "transforms_valid": not invalid_transforms,
        "invalid_transform_prim_paths": invalid_transforms[:40],
        "dependency_inspection_available": dependency_inspection_available,
        "missing_asset_count": len(unresolved_assets) if dependency_inspection_available else None,
        "unresolved_assets": unresolved_assets[:40],
        "dependency_inspection_error": dependency_error,
        "particlefield_prim_count": len(particlefields),
        "active_collision_prim_count": len(active),
        "active_collision_world_bounds": stage_bounds,
        "obvious_scale_mismatch_detected": obvious_scale_mismatch,
        "collision_prims": collision_prims,
    }


def _live_rigid_body_position(prim_path):
    import omni.physx  # type: ignore

    state = omni.physx.get_physx_interface().get_rigidbody_transformation(prim_path)
    if not hasattr(state, "get") or state.get("ret_val") is not True:
        raise RuntimeError("physx_live_rigid_body_pose_unavailable")
    position = state.get("position")
    values = [float(position[index]) for index in range(3)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("physx_live_rigid_body_position_nonfinite")
    return values


def _contact_event_count(prim_path, ground_prim_path):
    try:
        import omni.physx  # type: ignore
        from pxr import PhysicsSchemaTools  # type: ignore

        report = omni.physx.get_physx_simulation_interface().get_contact_report()
        headers = report[0] if report else []
        count = 0
        for header in headers:
            encoded = [
                getattr(header, name, "") for name in ("actor0", "actor1", "collider0", "collider1")
            ]
            paths = []
            for value in encoded:
                try:
                    paths.append(str(PhysicsSchemaTools.intToSdfPath(int(value))))
                except Exception:  # noqa: BLE001
                    paths.append(str(value))
            probe_seen = prim_path in paths or any(
                path.startswith(prim_path + "/") for path in paths
            )
            ground_seen = ground_prim_path in paths or any(
                path.startswith(ground_prim_path + "/") for path in paths
            )
            if probe_seen and ground_seen:
                count += 1
        return count
    except Exception:  # noqa: BLE001
        return 0


def _run_qualification_physics_probe(
    stage,
    ground_surface,
    *,
    steps,
    probe_xy,
    Gf,
    UsdGeom,
    Sdf,
):
    requested_steps = max(0, int(steps))
    errors = []
    executed_steps = 0
    contacts = 0
    initial_position = None
    final_position = None
    if not ground_surface:
        return _classify_physics_probe(
            ground_surface=None,
            requested_steps=requested_steps,
            executed_steps=0,
            initial_position=None,
            final_position=None,
            contact_event_count=0,
            errors=["ground_contact_surface_not_identified"],
        )
    probe_path = "/World/BlueprintReconstructionQualificationProbe"
    try:
        from isaacsim.core.api import SimulationContext  # type: ignore
        from pxr import PhysxSchema, UsdPhysics  # type: ignore

        bounds = ground_surface["world_bounds"]
        low, high = bounds["min"], bounds["max"]
        if probe_xy is None:
            x = (float(low[0]) + float(high[0])) * 0.5
            y = (float(low[1]) + float(high[1])) * 0.5
        else:
            x, y = (float(probe_xy[0]), float(probe_xy[1]))
        ground_height = float(ground_surface["probe_height_m"])
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(probe_path))
        cube.CreateSizeAttr(0.1)
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, ground_height + 0.5))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(1.0)
        contact_api = PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim())
        contact_api.CreateThresholdAttr().Set(0.0)

        context = SimulationContext(
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0,
        )
        context.initialize_physics()
        context.get_physics_context().set_gravity(-9.81)
        context.play()
        initial_position = _live_rigid_body_position(probe_path)
        for _ in range(requested_steps):
            try:
                context.step(render=False)
            except TypeError:
                context.step()
            executed_steps += 1
            contacts += _contact_event_count(probe_path, ground_surface["prim_path"])
        final_position = _live_rigid_body_position(probe_path)
        context.stop()
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc)[:600])
    return _classify_physics_probe(
        ground_surface=ground_surface,
        requested_steps=requested_steps,
        executed_steps=executed_steps,
        initial_position=initial_position,
        final_position=final_position,
        contact_event_count=contacts,
        errors=errors,
    )


def _render(args) -> int:
    started = time.monotonic()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "isaac_runtime_result.json"
    qualification_mode = bool(args.qualification_mode)
    provider_package_mode = bool(args.provider_package_mode)
    base = {
        "schema_version": (
            PROVIDER_QUALIFICATION_RESULT_SCHEMA
            if provider_package_mode
            else QUALIFICATION_RESULT_SCHEMA
            if qualification_mode
            else LEGACY_RESULT_SCHEMA
        ),
        "renderer": "isaac_rtx_particlefield",
        "raw_secret_values_recorded": False,
    }
    if qualification_mode:
        base.update(
            {
                "package_digest": args.package_digest,
                "isaac_verification_request_digest": args.verification_request_digest,
                "fixed_camera_spec_digest": args.camera_spec_digest,
                "runtime_container_image_digest": args.runtime_container_image_digest,
                "runtime_implementation_digest": args.runtime_implementation_digest,
                "runtime_identity": {
                    "runtime": "isaac_sim",
                    "renderer": "RayTracedLighting",
                    "python_version": sys.version.split()[0],
                    "headless": True,
                },
            }
        )
    _phase(result_path, base, "runner_started")

    python = args.python or sys.executable or "python3"
    transcode = None
    if args.usdc:
        stage_path = Path(args.usdc).expanduser().resolve()
        transcode = {"status": "skipped_direct_usdc"}
        if not stage_path.is_file():
            _write_json(
                result_path,
                {**base, "status": "blocked", "blockers": ["particlefield_usdc_missing"]},
            )
            return 2
    elif args.usdz:
        stage_path = Path(args.usdz).expanduser().resolve()
        transcode = {"status": "skipped_precomputed_usdz"}
        if not stage_path.is_file():
            _write_json(result_path, {**base, "status": "blocked", "blockers": ["usdz_missing"]})
            return 2
    else:
        ply = Path(args.ply).expanduser().resolve()
        if not ply.is_file():
            _write_json(
                result_path, {**base, "status": "blocked", "blockers": ["standard_ply_missing"]}
            )
            return 2
        stage_path = out_dir / "scene_particlefield.usd"
        _phase(result_path, base, "runner_transcoding_ply")
        transcode = _transcode_ply_to_usd(ply, stage_path, python=python, fmt="lightfield")
        if transcode.get("status") != "completed":
            _write_json(
                result_path,
                {
                    **base,
                    "status": "blocked",
                    "transcode": transcode,
                    "blockers": transcode.get("blockers", ["transcode_failed"]),
                },
            )
            return 2

    if qualification_mode:
        observed_package_digest = _sha256_file(stage_path)
        if observed_package_digest != args.package_digest:
            _write_json(
                result_path,
                {
                    **base,
                    "status": "blocked",
                    "observed_package_digest": observed_package_digest,
                    "blockers": ["isaac_exact_package_digest_mismatch"],
                },
            )
            return 2

    cameras_path = Path(args.cameras).expanduser().resolve()
    if qualification_mode:
        if _sha256_file(cameras_path) != args.camera_spec_digest:
            _write_json(
                result_path,
                {**base, "status": "blocked", "blockers": ["isaac_camera_spec_digest_mismatch"]},
            )
            return 2
        if _sha256_file(Path(__file__).resolve()) != args.runtime_implementation_digest:
            _write_json(
                result_path,
                {
                    **base,
                    "status": "blocked",
                    "blockers": ["isaac_runtime_implementation_digest_mismatch"],
                },
            )
            return 2
    cameras = json.loads(cameras_path.read_text(encoding="utf-8")) if args.cameras else []
    camera_ids = [str(row.get("id") or "") for row in cameras if isinstance(row, dict)]
    if (
        len(camera_ids) != len(cameras)
        or any(_CAMERA_ID.fullmatch(value) is None for value in camera_ids)
        or len(set(camera_ids)) != len(camera_ids)
    ):
        _write_json(
            result_path,
            {**base, "status": "blocked", "blockers": ["isaac_camera_ids_invalid"]},
        )
        return 2
    try:
        camera_prim_names = _camera_usd_prim_names(camera_ids)
    except ValueError as exc:
        _write_json(
            result_path,
            {**base, "status": "blocked", "blockers": [str(exc)]},
        )
        return 2
    _phase(
        result_path,
        base,
        "runner_importing_isaacsim",
        camera_count=len(cameras),
        stage_path=str(stage_path),
    )

    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _write_json(
            result_path,
            {
                **base,
                "status": "blocked",
                "blockers": ["isaacsim_module_unavailable"],
                "error": repr(exc),
            },
        )
        return 2

    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    rendered = []
    try:
        _phase(result_path, base, "runner_simulation_app_started")
        import omni.usd  # type: ignore
        import omni.replicator.core as rep  # type: ignore
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils  # type: ignore

        try:
            import carb  # type: ignore

            settings = carb.settings.get_settings()
            settings.set("/renderer/multiGpu/enabled", False)  # multi-GPU => empty annotator frames
            settings.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
        except Exception:  # noqa: BLE001
            pass
        try:
            from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

            loaded = {}
            for ext in (
                "omni.hydra.rtx",
                "omni.ujitso.client",
                "omni.ujitso.default",
                "omni.kit.converter.gsplat",
            ):
                try:
                    loaded[ext] = bool(enable_extension(ext))
                except Exception as exc:  # noqa: BLE001
                    loaded[ext] = f"err:{type(exc).__name__}"
            _phase(result_path, base, "runner_extensions", extensions=loaded)
        except Exception:  # noqa: BLE001
            pass

        context = omni.usd.get_context()
        context.open_stage(str(stage_path))
        stage = context.get_stage()
        if stage is None:
            _write_json(
                result_path, {**base, "status": "blocked", "blockers": ["isaac_stage_open_failed"]}
            )
            return 2
        pf = [
            p
            for p in stage.Traverse()
            if str(p.GetTypeName()) == PARTICLEFIELD_TYPE
            or (
                provider_package_mode
                and (
                    str(p.GetTypeName()) == NUREC_FIELD_TYPE
                    or any(
                        str(attr.GetName()) == "omni:nurec:isNuRecVolume" and bool(attr.Get())
                        for attr in p.GetAttributes()
                    )
                )
            )
        ]
        prim_types = sorted({str(p.GetTypeName()) for p in stage.Traverse() if p.GetTypeName()})[
            :40
        ]
        _phase(
            result_path,
            base,
            "runner_stage_opened",
            stage_path=str(stage_path),
            particlefield_prim_count=len(pf),
            prim_types=prim_types,
        )
        if not pf and not args.allow_any_stage:
            _write_json(
                result_path,
                {
                    **base,
                    "status": "blocked",
                    "blockers": ["no_particlefield_prim_in_stage"],
                    "prim_types": prim_types,
                },
            )
            return 2

        stage_evidence = {}
        ground_surface = None
        ground_surface_error = None
        if qualification_mode:
            try:
                stage.Load()
            except Exception:  # noqa: BLE001
                pass
            for _ in range(10):
                simulation_app.update()
            stage_evidence = _inspect_qualification_stage(
                stage,
                stage_path,
                pf,
                Usd=Usd,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                UsdUtils=UsdUtils,
            )
            expected_paths = {
                "appearance": args.expected_appearance_prim,
                "collision": args.expected_collision_prim,
            }
            stage_evidence["expected_prim_paths"] = {
                name: path if stage.GetPrimAtPath(path).IsValid() else None
                for name, path in expected_paths.items()
            }
            ground_surface, ground_surface_error = _select_ground_surface(
                stage_evidence.get("collision_prims") or [],
                requested_path=args.ground_collider_prim,
                declared_height=args.ground_height,
            )
            _phase(
                result_path,
                base,
                "runner_qualification_stage_inspected",
                stage=stage_evidence,
                ground_surface=ground_surface,
                ground_surface_error=ground_surface_error,
            )

        # Optional robot compositing at the validated stance (bundle-driven).
        render_options = _load_render_options(Path(args.cameras)) if args.cameras else {}
        robot_report = _composite_robot(stage, render_options, Gf=Gf, UsdGeom=UsdGeom, Sdf=Sdf)
        robot_lights_prim = None
        if robot_report.get("requested"):
            _phase(result_path, base, "runner_robot_composited", robot=robot_report)
        if robot_report.get("composited"):
            # A composed reference is only STRUCTURE — the payload meshes stream
            # asynchronously (here from the Isaac cloud assets bucket). Capturing
            # before they arrive renders an invisible robot with bit-identical
            # frames to a robot-less run. Load payloads explicitly and pump
            # updates until the robot's world bound is non-empty (or a bounded
            # timeout), recording the bound as placement ground truth.
            from pxr import Usd  # type: ignore

            robot_prim = stage.GetPrimAtPath(robot_report["prim_path"])
            try:
                robot_prim.Load()
                stage.Load()  # payloads stage-wide: instanceable robot meshes ride payloads
            except Exception:  # noqa: BLE001
                pass

            def _mesh_point_stats(prim):
                # REAL geometry evidence: extentsHint composes with structure long
                # before the mesh payload downloads, so bounds alone cannot prove
                # the robot will draw. Count meshes whose `points` attribute holds
                # actual data (traverse into instance prototypes, where
                # instanceable robot assets keep their geometry).
                meshes = 0
                points = 0
                it = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
                for child in it:
                    if child.IsA(UsdGeom.Mesh):
                        arr = UsdGeom.Mesh(child).GetPointsAttr().Get()
                        if arr:
                            meshes += 1
                            points += len(arr)
                return meshes, points

            streamed = False
            for i in range(90):  # up to ~90 x 20 updates waiting for mesh payload data
                for _ in range(20):
                    simulation_app.update()
                meshes, points = _mesh_point_stats(robot_prim)
                if meshes > 0 and points > 100:
                    streamed = True
                    robot_report["mesh_prims_with_points"] = meshes
                    robot_report["mesh_point_total"] = points
                    robot_report["updates_until_mesh_data"] = (i + 1) * 20
                    break
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            rng = cache.ComputeWorldBound(robot_prim).GetRange()
            if not rng.IsEmpty():
                robot_report["world_bound_min"] = [round(float(v), 4) for v in rng.GetMin()]
                robot_report["world_bound_max"] = [round(float(v), 4) for v in rng.GetMax()]
            robot_report["geometry_streamed"] = streamed
            ground_z = render_options.get("robot_ground_z")
            if streamed and ground_z is not None:
                # Ground snap: place the robot's lowest point exactly on the
                # floor, independent of whether the asset's root origin is the
                # pelvis or the feet.
                shift = float(ground_z) - float(robot_report["world_bound_min"][2])
                if abs(shift) > 1e-4:
                    x, y, z, yaw = (float(v) for v in robot_report["robot_pose"])
                    import math as _math

                    matrix = Gf.Matrix4d().SetRotate(
                        Gf.Rotation(Gf.Vec3d(0, 0, 1), _math.degrees(yaw))
                    )
                    matrix.SetTranslateOnly(Gf.Vec3d(x, y, z + shift))
                    xformable = UsdGeom.Xformable(robot_prim)
                    xformable.ClearXformOpOrder()
                    xformable.AddTransformOp().Set(matrix)
                    robot_report["ground_snap_shift_z"] = round(shift, 4)
                    for _ in range(10):
                        simulation_app.update()
                    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                    rng = cache.ComputeWorldBound(robot_prim).GetRange()
                    if not rng.IsEmpty():
                        robot_report["world_bound_min"] = [round(float(v), 4) for v in rng.GetMin()]
                        robot_report["world_bound_max"] = [round(float(v), 4) for v in rng.GetMax()]
            robot_report["visual_material_support"] = _author_robot_evidence_material(
                stage,
                robot_prim,
                Sdf=Sdf,
                UsdShade=UsdShade,
            )
            robot_lights_prim, lighting_report = _ensure_robot_only_lights(
                stage,
                str(render_options.get("lights_path") or ""),
                Sdf=Sdf,
                UsdGeom=UsdGeom,
                UsdLux=UsdLux,
            )
            UsdGeom.Imageable(robot_lights_prim).MakeVisible()
            robot_report["robot_only_lighting"] = lighting_report
            for _ in range(10):
                simulation_app.update()
            _phase(result_path, base, "runner_robot_geometry", robot=robot_report)

        # warm up so the splat/materials upload to the GPU before any capture
        for _ in range(int(args.warmup_frames)):
            simulation_app.update()

        cam_root = "/World/BlueprintRenderCameras"
        UsdGeom.Xform.Define(stage, Sdf.Path(cam_root))
        for idx, cam in enumerate(cameras):
            cid = str(cam.get("id") or f"cam_{idx}")
            spec = cam.get("spec") or {}
            cam_path = f"{cam_root}/{camera_prim_names[idx]}"
            camera = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
            xform = _camera_xform(
                Gf,
                spec.get("pos", [0, 0, 0]),
                spec.get("target", [0, 0, -1]),
                spec.get("up", [0, 0, 1]),
            )
            xf = UsdGeom.Xformable(camera.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(xform)
            fov = float(spec.get("fov", 50))
            aperture = 24.0
            aspect = float(args.width) / float(args.height)
            camera.CreateFocalLengthAttr(
                float((aperture * 0.5) / math.tan(math.radians(fov) * 0.5))
            )
            camera.CreateVerticalApertureAttr(float(aperture))
            camera.CreateHorizontalApertureAttr(float(aperture) * aspect)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100000.0))

            cam_out = frames_dir / cid
            cam_out.mkdir(parents=True, exist_ok=True)
            render_product = rep.create.render_product(
                cam_path, (int(args.width), int(args.height))
            )
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(cam_out), rgb=True)
            writer.attach([render_product])
            for _ in range(10):
                simulation_app.update()
            for _ in range(int(args.subframes)):
                rep.orchestrator.step(rt_subframes=int(args.rt_subframes))
                simulation_app.update()
            try:
                rep.orchestrator.wait_until_complete()
            except Exception:  # noqa: BLE001
                pass

            pngs = sorted(glob.glob(str(cam_out / "*.png")))
            canonical = frames_dir / f"{cid}.png"
            std = 0.0
            if pngs:
                src = Path(pngs[-1])
                canonical.write_bytes(src.read_bytes())
                width, height, mean, std = _pixel_stats(canonical)
            else:
                width, height, mean = 0, 0, 0.0
            rendered.append(
                {
                    "id": cid,
                    "png": str(canonical),
                    "artifact_reference": f"frames/{cid}.png",
                    "digest": _sha256_file(canonical) if canonical.is_file() else None,
                    "width": width,
                    "height": height,
                    "pixel_mean": mean,
                    "pixel_std": std,
                    "nonblank": std > 3.0 and math.isfinite(std),
                    "frame_count": len(pngs),
                }
            )
            _phase(
                result_path,
                base,
                "runner_camera_rendered",
                camera_id=cid,
                pixel_std=round(std, 3),
                rendered_count=len(rendered),
            )
            try:
                writer.detach()
            except Exception:  # noqa: BLE001
                pass
            try:
                render_product.destroy()
            except Exception:  # noqa: BLE001
                pass

        # Optional probe/composite pass: hide the splat and capture the ROBOT ONLY
        # (RGB + distance_to_camera). Answers "does the mesh render at all in this
        # pipeline" and provides the inputs for a local depth-composite fallback
        # when the splat pass does not depth-composite with mesh geometry.
        robot_only = []
        if render_options.get("robot_only_pass") and robot_report.get("composited"):
            ro_dir = out_dir / "frames_robot_only"
            ro_dir.mkdir(parents=True, exist_ok=True)
            pf_imageables = [UsdGeom.Imageable(p) for p in pf]
            for imageable in pf_imageables:
                imageable.MakeInvisible()
            # The splat is self-emissive; the mesh robot needs the same support
            # lights used by the scene-plus-robot evidence pass.
            lights_prim = robot_lights_prim
            if lights_prim is None:
                lights_prim, lighting_report = _ensure_robot_only_lights(
                    stage,
                    str(render_options.get("lights_path") or ""),
                    Sdf=Sdf,
                    UsdGeom=UsdGeom,
                    UsdLux=UsdLux,
                )
                UsdGeom.Imageable(lights_prim).MakeVisible()
                robot_report["robot_only_lighting"] = lighting_report
            for _ in range(10):
                simulation_app.update()
            for idx, cam in enumerate(cameras):
                cid = str(cam.get("id") or f"cam_{idx}")
                cam_path = f"{cam_root}/{camera_prim_names[idx]}"
                cam_out = ro_dir / cid
                cam_out.mkdir(parents=True, exist_ok=True)
                render_product = rep.create.render_product(
                    cam_path, (int(args.width), int(args.height))
                )
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(output_dir=str(cam_out), rgb=True, distance_to_camera=True)
                writer.attach([render_product])
                for _ in range(10):
                    simulation_app.update()
                for _ in range(int(args.subframes)):
                    rep.orchestrator.step(rt_subframes=int(args.rt_subframes))
                    simulation_app.update()
                try:
                    rep.orchestrator.wait_until_complete()
                except Exception:  # noqa: BLE001
                    pass
                pngs = sorted(glob.glob(str(cam_out / "*.png")))
                npys = sorted(glob.glob(str(cam_out / "*.npy")))
                canonical = ro_dir / f"{cid}.png"
                distance = ro_dir / f"{cid}_distance.npy"
                std = 0.0
                if pngs:
                    canonical.write_bytes(Path(pngs[-1]).read_bytes())
                    std = _pixel_std(canonical)
                if npys:
                    distance.write_bytes(Path(npys[-1]).read_bytes())
                robot_only.append(
                    {
                        "id": cid,
                        "pixel_std": round(std, 3),
                        "nonblank": std > 3.0,
                        "depth_npy": bool(npys),
                        "rgb_artifact_reference": (
                            f"frames_robot_only/{cid}.png" if canonical.is_file() else None
                        ),
                        "rgb_digest": (_sha256_file(canonical) if canonical.is_file() else None),
                        "distance_artifact_reference": (
                            f"frames_robot_only/{cid}_distance.npy" if distance.is_file() else None
                        ),
                        "distance_digest": (_sha256_file(distance) if distance.is_file() else None),
                    }
                )
                _phase(
                    result_path,
                    base,
                    "runner_robot_only_rendered",
                    camera_id=cid,
                    pixel_std=round(std, 3),
                )
                try:
                    writer.detach()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    render_product.destroy()
                except Exception:  # noqa: BLE001
                    pass
            for imageable in pf_imageables:
                imageable.MakeVisible()
            if lights_prim and lights_prim.IsValid():
                UsdGeom.Imageable(lights_prim).MakeInvisible()
            robot_report["robot_only_pass"] = robot_only

        nonblank = sum(1 for r in rendered if r["nonblank"])
        threshold = max(1, round(0.6 * len(rendered))) if rendered else 1
        visual_ok = nonblank >= threshold
        mp4 = _encode_mp4(
            [Path(r["png"]) for r in rendered if r["nonblank"]], out_dir / "scene_render.mp4"
        )
        physics_probe = {}
        blockers = []
        if qualification_mode:
            _exclude_robot_from_environment_physics_probe(
                stage,
                robot_report,
                Sdf=Sdf,
            )
            physics_probe = _run_qualification_physics_probe(
                stage,
                ground_surface,
                steps=args.physics_probe_steps,
                probe_xy=args.probe_xy,
                Gf=Gf,
                UsdGeom=UsdGeom,
                Sdf=Sdf,
            )
            if ground_surface_error:
                physics_probe.setdefault("errors", []).append(ground_surface_error)
            blockers = _qualification_blockers(
                package_digest=args.package_digest,
                stage=stage_evidence,
                physics_probe=physics_probe,
                cameras=rendered,
            )
            ok = not blockers
        else:
            ok = visual_ok
            blockers = [] if ok else ["isaac_particlefield_render_produced_blank_or_few_frames"]
        final_result = {
            **base,
            "status": "completed" if ok else "blocked",
            "isaac_runtime_executed": True,
            "isaac_particlefield_rendered": bool(visual_ok),
            "rendered_by_isaac_rtx": True,
            "stage_path": str(stage_path),
            "particlefield_prim_count": len(pf),
            "stage": stage_evidence if qualification_mode else None,
            "physics_probe": physics_probe if qualification_mode else None,
            "transcode": transcode,
            "cameras": rendered,
            "nonblank_camera_count": nonblank,
            "nonblank_threshold": threshold,
            "mp4": mp4,
            "robot": robot_report,
            "blockers": blockers,
            "cost_usd": 0.0,
            "duration_seconds": max(0.0, time.monotonic() - started),
            "proof_boundary": {
                "captured_scene_displayed_in_isaac_rtx": bool(visual_ok),
                "robot_visual_composited_at_stance": bool(robot_report.get("composited")),
                "isaac_load_render_physics_presence_compatibility": bool(qualification_mode and ok),
                "simulator_task_success_proven": False,
                "physics_navigation_control_proven": False,
                "physical_success_proven": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
            },
        }
        _write_json(result_path, final_result)
        _bundle_and_upload(out_dir, result_path)
        return 0 if ok else 2
    except Exception as exc:  # noqa: BLE001
        _write_json(
            result_path,
            {
                **base,
                "status": "blocked",
                "blockers": ["isaac_render_exception"],
                "error": repr(exc),
                "cameras": rendered,
            },
        )
        return 2
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass


def _encode_mp4(frames, out_path: Path) -> dict:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    frames = [p for p in frames if p and Path(p).is_file()]
    if not ffmpeg or not frames:
        return {"status": "blocked", "blockers": ["ffmpeg_or_frames_unavailable"]}
    listing = out_path.with_suffix(".concat.txt")
    lines = []
    for p in frames:
        lines.append(f"file '{Path(p).resolve()}'")
        lines.append("duration 1.6")
    lines.append(f"file '{Path(frames[-1]).resolve()}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["ffmpeg_exception"], "error": repr(exc)}
    if proc.returncode != 0 or not out_path.is_file():
        return {"status": "blocked", "blockers": ["ffmpeg_failed"]}
    return {"status": "completed", "mp4": str(out_path), "bytes": out_path.stat().st_size}


def _bundle_and_upload(out_dir: Path, result_path: Path) -> None:
    signed = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "").strip()
    zip_path = out_dir / "isaac_particlefield_runtime_output.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(out_dir.rglob("*")):
                if item.is_file() and item != zip_path:
                    zf.write(item, item.relative_to(out_dir).as_posix())
    except Exception:  # noqa: BLE001
        return
    if not signed:
        return
    try:
        import urllib.parse
        import urllib.request

        parsed = urllib.parse.urlsplit(signed)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return
        data = zip_path.read_bytes()
        req = urllib.request.Request(
            signed, data=data, method="PUT", headers={"Content-Type": "application/zip"}
        )
        # The exact presigned URL is accepted only after the HTTPS and authority
        # checks above; redirects remain within urllib's standard handler.
        urllib.request.urlopen(req, timeout=300).read()  # nosec B310
    except Exception:  # noqa: BLE001
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--usdc", help="ParticleField3DGaussianSplat .usdc/.usd opened directly (preferred)"
    )
    ap.add_argument("--usdz", help="precomputed NuRec/ParticleField USDZ")
    ap.add_argument(
        "--ply", help="standard 3DGS PLY (transcoded to ParticleField USD on the worker)"
    )
    ap.add_argument(
        "--cameras", required=True, help="cameras.json: [{id, spec:{pos,target,fov,up}}]"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--python", default=None, help="python for threedgrut transcode (only with --ply)"
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--subframes", type=int, default=8)
    ap.add_argument("--rt-subframes", type=int, default=64)
    ap.add_argument(
        "--qualification-mode",
        action="store_true",
        help="emit strict v3 load/render/physics-presence evidence for an exact packaged asset",
    )
    ap.add_argument(
        "--package-digest",
        help="sha256 digest of the exact --usdc/--usdz bytes; required in qualification mode",
    )
    ap.add_argument("--verification-request-digest")
    ap.add_argument("--camera-spec-digest")
    ap.add_argument("--runtime-container-image-digest")
    ap.add_argument("--runtime-implementation-digest")
    ap.add_argument(
        "--ground-collider-prim",
        default="",
        help="optional exact active package collider prim used as the contact surface",
    )
    ap.add_argument(
        "--ground-height",
        type=float,
        default=None,
        help="measured floor Z in stage meters; required when the declared collider is a combined room mesh",
    )
    ap.add_argument(
        "--probe-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="optional measured collision-probe XY in stage meters",
    )
    ap.add_argument("--physics-probe-steps", type=int, default=240)
    ap.add_argument(
        "--provider-package-mode",
        action="store_true",
        help="qualify an exact provider-authored NuRec package and emit its versioned runtime result",
    )
    ap.add_argument(
        "--expected-appearance-prim",
        default="/World/BlueprintReconstruction/Appearance",
        help="exact appearance prim expected in qualification mode",
    )
    ap.add_argument(
        "--expected-collision-prim",
        default="/World/BlueprintReconstruction/Collision",
        help="exact collision prim expected in qualification mode",
    )
    ap.add_argument(
        "--allow-any-stage",
        action="store_true",
        help="skip the ParticleField prim assertion (e.g. rendering a textured USD de-risk scene)",
    )
    args = ap.parse_args(argv)
    if not args.usdc and not args.usdz and not args.ply:
        ap.error("one of --usdc, --usdz, or --ply is required")
    if args.qualification_mode and not _is_sha256_digest(args.package_digest):
        ap.error(
            "--qualification-mode requires --package-digest sha256:<64 lowercase or uppercase hex>"
        )
    if args.package_digest:
        args.package_digest = args.package_digest.lower()
    if args.qualification_mode and not _is_sha256_digest(args.verification_request_digest):
        ap.error("--qualification-mode requires --verification-request-digest")
    if args.qualification_mode and not _is_sha256_digest(args.camera_spec_digest):
        ap.error("--qualification-mode requires --camera-spec-digest")
    if args.qualification_mode and not _is_image_digest(args.runtime_container_image_digest):
        ap.error("--qualification-mode requires --runtime-container-image-digest")
    if args.qualification_mode and not _is_sha256_digest(args.runtime_implementation_digest):
        ap.error("--qualification-mode requires --runtime-implementation-digest")
    if args.qualification_mode and args.ply:
        ap.error(
            "--qualification-mode requires an exact packaged --usdc or --usdz, not --ply transcode"
        )
    if args.qualification_mode and args.allow_any_stage:
        ap.error("--qualification-mode cannot be combined with --allow-any-stage")
    if args.provider_package_mode and not args.qualification_mode:
        ap.error("--provider-package-mode requires --qualification-mode")
    for option, value in (
        ("--expected-appearance-prim", args.expected_appearance_prim),
        ("--expected-collision-prim", args.expected_collision_prim),
    ):
        if not isinstance(value, str) or not value.startswith("/") or ".." in value.split("/"):
            ap.error(f"{option} must be an absolute safe prim path")
    if args.physics_probe_steps < 2:
        ap.error("--physics-probe-steps must be at least 2")
    return _render(args)


if __name__ == "__main__":
    raise SystemExit(main())
