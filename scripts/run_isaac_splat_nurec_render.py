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
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

RESULT_SCHEMA = "isaac_splat_nurec_render_result.v1"
PARTICLEFIELD_TYPE = "ParticleField3DGaussianSplat"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(result_path: Path, base: dict, phase: str, **extra) -> None:
    _write_json(result_path, {**base, "status": "running", "provider_runtime_phase": phase, **extra})


def _transcode_ply_to_usd(ply: Path, usd: Path, *, python: str, fmt: str = "lightfield") -> dict:
    # fmt 'lightfield' => ParticleField3DGaussianSplat (matches the validated authoring path).
    cmd = [python, "-m", "threedgrut.export.scripts.transcode", str(ply), "-o", str(usd), "--format", fmt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["threedgrut_transcode_exception"], "error": repr(exc)}
    if proc.returncode != 0 or not usd.is_file():
        return {"status": "blocked", "blockers": ["threedgrut_transcode_failed"],
                "returncode": proc.returncode, "stderr_tail": (proc.stderr or "")[-2000:]}
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
        prim = stage.DefinePrim(Sdf.Path(prim_path), "Xform")
        chosen = None
        tried = []
        for cand in _robot_usd_candidates(robot_usd):
            tried.append(cand)
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(cand)
            composed = any(True for child in Usd.PrimRange(prim) if child.GetPath() != prim.GetPath())
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
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(matrix)
        child_count = sum(1 for child in Usd.PrimRange(prim)) - 1
        report.update(composited=True, resolved_usd=chosen, prim_path=prim_path,
                      composed_prim_count=child_count)
        return report
    except Exception as exc:  # noqa: BLE001
        report.update(composited=False, blocker="robot_composite_exception", error=repr(exc)[:400])
        return report


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


def _render(args) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "isaac_runtime_result.json"
    base = {"schema_version": RESULT_SCHEMA, "renderer": "isaac_rtx_particlefield", "raw_secret_values_recorded": False}
    _phase(result_path, base, "runner_started")

    python = args.python or sys.executable or "python3"
    transcode = None
    if args.usdc:
        stage_path = Path(args.usdc).expanduser().resolve()
        transcode = {"status": "skipped_direct_usdc"}
        if not stage_path.is_file():
            _write_json(result_path, {**base, "status": "blocked", "blockers": ["particlefield_usdc_missing"]})
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
            _write_json(result_path, {**base, "status": "blocked", "blockers": ["standard_ply_missing"]})
            return 2
        stage_path = out_dir / "scene_particlefield.usd"
        _phase(result_path, base, "runner_transcoding_ply")
        transcode = _transcode_ply_to_usd(ply, stage_path, python=python, fmt="lightfield")
        if transcode.get("status") != "completed":
            _write_json(result_path, {**base, "status": "blocked", "transcode": transcode,
                                      "blockers": transcode.get("blockers", ["transcode_failed"])})
            return 2

    cameras = json.loads(Path(args.cameras).read_text(encoding="utf-8")) if args.cameras else []
    _phase(result_path, base, "runner_importing_isaacsim", camera_count=len(cameras), stage_path=str(stage_path))

    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _write_json(result_path, {**base, "status": "blocked", "blockers": ["isaacsim_module_unavailable"],
                                  "error": repr(exc)})
        return 2

    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    rendered = []
    try:
        _phase(result_path, base, "runner_simulation_app_started")
        import omni.usd  # type: ignore
        import omni.replicator.core as rep  # type: ignore
        from pxr import Gf, UsdGeom, Sdf  # type: ignore

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
            for ext in ("omni.hydra.rtx", "omni.ujitso.client", "omni.ujitso.default", "omni.kit.converter.gsplat"):
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
            _write_json(result_path, {**base, "status": "blocked", "blockers": ["isaac_stage_open_failed"]})
            return 2
        pf = [p for p in stage.Traverse() if str(p.GetTypeName()) == PARTICLEFIELD_TYPE]
        prim_types = sorted({str(p.GetTypeName()) for p in stage.Traverse() if p.GetTypeName()})[:40]
        _phase(result_path, base, "runner_stage_opened", stage_path=str(stage_path),
               particlefield_prim_count=len(pf), prim_types=prim_types)
        if not pf and not args.allow_any_stage:
            _write_json(result_path, {**base, "status": "blocked",
                                      "blockers": ["no_particlefield_prim_in_stage"], "prim_types": prim_types})
            return 2

        # Optional robot compositing at the validated stance (bundle-driven).
        render_options = _load_render_options(Path(args.cameras)) if args.cameras else {}
        robot_report = _composite_robot(stage, render_options, Gf=Gf, UsdGeom=UsdGeom, Sdf=Sdf)
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
            _phase(result_path, base, "runner_robot_geometry", robot=robot_report)

        # warm up so the splat/materials upload to the GPU before any capture
        for _ in range(int(args.warmup_frames)):
            simulation_app.update()

        cam_root = "/World/BlueprintRenderCameras"
        UsdGeom.Xform.Define(stage, Sdf.Path(cam_root))
        for idx, cam in enumerate(cameras):
            cid = str(cam.get("id") or f"cam_{idx}")
            spec = cam.get("spec") or {}
            cam_path = f"{cam_root}/{cid}"
            camera = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
            xform = _camera_xform(Gf, spec.get("pos", [0, 0, 0]), spec.get("target", [0, 0, -1]),
                                  spec.get("up", [0, 0, 1]))
            xf = UsdGeom.Xformable(camera.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(xform)
            fov = float(spec.get("fov", 50))
            aperture = 24.0
            aspect = float(args.width) / float(args.height)
            camera.CreateFocalLengthAttr(float((aperture * 0.5) / math.tan(math.radians(fov) * 0.5)))
            camera.CreateVerticalApertureAttr(float(aperture))
            camera.CreateHorizontalApertureAttr(float(aperture) * aspect)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100000.0))

            cam_out = frames_dir / cid
            cam_out.mkdir(parents=True, exist_ok=True)
            render_product = rep.create.render_product(cam_path, (int(args.width), int(args.height)))
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
                std = _pixel_std(canonical)
            rendered.append({"id": cid, "png": str(canonical), "pixel_std": round(std, 3),
                             "nonblank": std > 3.0, "frame_count": len(pngs)})
            _phase(result_path, base, "runner_camera_rendered", camera_id=cid, pixel_std=round(std, 3),
                   rendered_count=len(rendered))
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
            for _ in range(10):
                simulation_app.update()
            for idx, cam in enumerate(cameras):
                cid = str(cam.get("id") or f"cam_{idx}")
                cam_path = f"{cam_root}/{cid}"
                cam_out = ro_dir / cid
                cam_out.mkdir(parents=True, exist_ok=True)
                render_product = rep.create.render_product(cam_path, (int(args.width), int(args.height)))
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
                std = 0.0
                if pngs:
                    canonical.write_bytes(Path(pngs[-1]).read_bytes())
                    std = _pixel_std(canonical)
                if npys:
                    (ro_dir / f"{cid}_distance.npy").write_bytes(Path(npys[-1]).read_bytes())
                robot_only.append({"id": cid, "pixel_std": round(std, 3), "nonblank": std > 3.0,
                                   "depth_npy": bool(npys)})
                _phase(result_path, base, "runner_robot_only_rendered", camera_id=cid,
                       pixel_std=round(std, 3))
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
            robot_report["robot_only_pass"] = robot_only

        nonblank = sum(1 for r in rendered if r["nonblank"])
        threshold = max(1, round(0.6 * len(rendered))) if rendered else 1
        ok = nonblank >= threshold
        mp4 = _encode_mp4([Path(r["png"]) for r in rendered if r["nonblank"]], out_dir / "scene_render.mp4")
        _bundle_and_upload(out_dir, result_path)
        _write_json(result_path, {
            **base,
            "status": "completed" if ok else "blocked",
            "isaac_runtime_executed": True,
            "isaac_particlefield_rendered": bool(ok),
            "rendered_by_isaac_rtx": True,
            "stage_path": str(stage_path),
            "particlefield_prim_count": len(pf),
            "transcode": transcode,
            "cameras": rendered,
            "nonblank_camera_count": nonblank,
            "nonblank_threshold": threshold,
            "mp4": mp4,
            "robot": robot_report,
            "blockers": [] if ok else ["isaac_particlefield_render_produced_blank_or_few_frames"],
            "proof_boundary": {
                "captured_scene_displayed_in_isaac_rtx": bool(ok),
                "robot_visual_composited_at_stance": bool(robot_report.get("composited")),
                "physics_navigation_control_proven": False,
                "physical_robot_readiness_proven": False,
            },
        })
        return 0 if ok else 2
    except Exception as exc:  # noqa: BLE001
        _write_json(result_path, {**base, "status": "blocked", "blockers": ["isaac_render_exception"],
                                  "error": repr(exc), "cameras": rendered})
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
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", str(out_path)]
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
        import urllib.request

        data = zip_path.read_bytes()
        req = urllib.request.Request(signed, data=data, method="PUT",
                                     headers={"Content-Type": "application/zip"})
        urllib.request.urlopen(req, timeout=300).read()
    except Exception:  # noqa: BLE001
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usdc", help="ParticleField3DGaussianSplat .usdc/.usd opened directly (preferred)")
    ap.add_argument("--usdz", help="precomputed NuRec/ParticleField USDZ")
    ap.add_argument("--ply", help="standard 3DGS PLY (transcoded to ParticleField USD on the worker)")
    ap.add_argument("--cameras", required=True, help="cameras.json: [{id, spec:{pos,target,fov,up}}]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=None, help="python for threedgrut transcode (only with --ply)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--subframes", type=int, default=8)
    ap.add_argument("--rt-subframes", type=int, default=64)
    ap.add_argument("--allow-any-stage", action="store_true",
                    help="skip the ParticleField prim assertion (e.g. rendering a textured USD de-risk scene)")
    args = ap.parse_args(argv)
    if not args.usdc and not args.usdz and not args.ply:
        ap.error("one of --usdc, --usdz, or --ply is required")
    return _render(args)


if __name__ == "__main__":
    raise SystemExit(main())
