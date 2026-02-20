#!/usr/bin/env python3
"""Inpaint360GS scene cleaning orchestrator.

Removes detected objects from a 3D Gaussian Splatting model and inpaints the
background behind them.  Produces a clean visual mesh (GLB) that replaces
``obj_nurec_visual`` so swapped USD assets sit in a ghost-geometry-free scene.

Pipeline stages:
  1. Prepare data layout (symlinks + SAM3 instance masks)
  2. Train vanilla 3DGS from COLMAP reconstruction
  3. Distill SAM3 masks into per-Gaussian object embeddings
  4. Remove target object Gaussians
  5. Generate virtual camera poses around removal regions
  6. Run LaMa 2D inpainting (color + depth)
  7. PLY fusion + 3DGS inpainting optimization
  8. Convert final PLY → GLB mesh

Requires:  https://github.com/dfki-av/Inpaint360GS  installed at INPAINT360GS_DIR.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration (env-var overridable)
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


INPAINT360GS_DIR = Path(os.getenv("INPAINT360GS_DIR", "/opt/Inpaint360GS"))
INPAINT360GS_PYTHON = os.getenv("INPAINT360GS_PYTHON", "python3.10")
INPAINT360GS_RESOLUTION = max(1, _env_int("INPAINT360GS_RESOLUTION", 2))
INPAINT360GS_TRAIN_ITERS = max(1000, _env_int("INPAINT360GS_TRAIN_ITERS", 30000))
INPAINT360GS_DISTILL_ITERS = max(100, _env_int("INPAINT360GS_DISTILL_ITERS", 2000))
INPAINT360GS_FINETUNE_ITERS = max(100, _env_int("INPAINT360GS_FINETUNE_ITERS", 3000))
INPAINT360GS_REMOVAL_THRESH = max(0.1, min(1.0, _env_float("INPAINT360GS_REMOVAL_THRESH", 0.7)))
INPAINT360GS_LAMA_EXPAND_PX = max(5, _env_int("INPAINT360GS_LAMA_EXPAND_PX", 15))
INPAINT360GS_MAX_OBJECTS = _env_int("INPAINT360GS_MAX_OBJECTS", 0)  # 0 = all
INPAINT360GS_MAX_MESH_FACES = _env_int("INPAINT360GS_MAX_MESH_FACES", 500000)


def _log(msg: str) -> None:
    print(f"[inpaint360gs] {msg}", flush=True)


def probe_installation(*, install_dir: Path = INPAINT360GS_DIR) -> Dict[str, Any]:
    """Validate Inpaint360GS install and command surface before execution."""
    required_scripts = [
        "train.py",
        "train_finetune.py",
        "edit_object_removal.py",
        "edit_object_inpaint.py",
        "predict_color.py",
        "predict_depth.py",
    ]
    missing = [name for name in required_scripts if not (install_dir / name).is_file()]
    status = "ok" if install_dir.is_dir() and not missing else "failed"
    return {
        "status": status,
        "install_dir": str(install_dir),
        "python": INPAINT360GS_PYTHON,
        "required_scripts": required_scripts,
        "missing_scripts": missing,
    }


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
    label: str = "",
) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    label_str = f" ({label})" if label else ""
    _log(f"Running{label_str}: {' '.join(str(c) for c in cmd)}")
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        _log(f"  FAILED (rc={proc.returncode})")
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-10:]:
                _log(f"  stderr: {line}")
    return proc


# ---------------------------------------------------------------------------
# Stage 1: Data layout preparation
# ---------------------------------------------------------------------------

def prepare_data_layout(
    *,
    colmap_sparse_dir: Path,
    images_dir: Path,
    instance_masks_dir: Path,
    object_index_path: Path,
    workspace: Path,
    resolution: int = INPAINT360GS_RESOLUTION,
) -> Dict[str, Any]:
    """Create Inpaint360GS data directory layout with symlinks.

    Creates::

        workspace/
          images/              → symlinks to undistorted images
          sparse/0/            → symlinks to COLMAP files
          associated_hqsam/    → instance masks (rescaled if needed)
          associated_hqsam/scene.json

    Returns:
        dict with ``num_objects``, ``num_images``, ``object_ids``.
    """
    _log("Preparing Inpaint360GS data layout...")
    workspace.mkdir(parents=True, exist_ok=True)

    # Symlink images
    ws_images = workspace / "images"
    if not ws_images.exists():
        ws_images.symlink_to(images_dir.resolve())
    image_files = sorted(ws_images.glob("*.*"))
    _log(f"  Images: {len(image_files)} files")

    # Symlink COLMAP sparse
    ws_sparse = workspace / "sparse" / "0"
    ws_sparse.parent.mkdir(parents=True, exist_ok=True)
    if not ws_sparse.exists():
        ws_sparse.symlink_to(colmap_sparse_dir.resolve())

    # Load object index to count objects
    with open(object_index_path, "r", encoding="utf-8") as f:
        obj_index = json.load(f)
    objects = obj_index.get("objects", [])
    if INPAINT360GS_MAX_OBJECTS > 0:
        objects = objects[:INPAINT360GS_MAX_OBJECTS]
    num_objects = len(objects)
    # Object IDs are 1-indexed in the instance masks
    object_ids = list(range(1, num_objects + 1))

    # Copy instance masks to associated_hqsam/
    ws_masks = workspace / "associated_hqsam"
    ws_masks.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(instance_masks_dir.glob("*.png"))
    if resolution > 1 and mask_files:
        # Need to downscale masks
        _log(f"  Rescaling {len(mask_files)} instance masks to 1/{resolution} resolution")
        try:
            from PIL import Image as PILImage
            import numpy as np
            for mf in mask_files:
                img = PILImage.open(mf)
                new_w = max(1, img.width // resolution)
                new_h = max(1, img.height // resolution)
                # Use NEAREST to preserve integer object IDs
                resized = img.resize((new_w, new_h), PILImage.NEAREST)
                resized.save(ws_masks / mf.name)
        except ImportError:
            _log("  WARNING: PIL not available, copying masks without rescaling")
            for mf in mask_files:
                shutil.copy2(mf, ws_masks / mf.name)
    else:
        for mf in mask_files:
            shutil.copy2(mf, ws_masks / mf.name)

    _log(f"  Instance masks: {len(mask_files)} files → {ws_masks}")

    # Write scene.json (num_classes = num_objects + 1 for background)
    scene_json = {"num_classes": num_objects + 1}
    (ws_masks / "scene.json").write_text(json.dumps(scene_json, indent=2), encoding="utf-8")
    _log(f"  scene.json: num_classes={num_objects + 1}")

    return {
        "num_objects": num_objects,
        "num_images": len(image_files),
        "object_ids": object_ids,
    }


# ---------------------------------------------------------------------------
# Stage 2: Vanilla 3DGS training
# ---------------------------------------------------------------------------

def run_training(
    *,
    workspace: Path,
    resolution: int = INPAINT360GS_RESOLUTION,
    iterations: int = INPAINT360GS_TRAIN_ITERS,
) -> Dict[str, Any]:
    """Train vanilla 3DGS on the COLMAP data."""
    _log(f"Training 3DGS (resolution=1/{resolution}, iters={iterations})...")
    t0 = time.monotonic()

    train_script = INPAINT360GS_DIR / "train.py"
    if not train_script.is_file():
        return {"status": "failed", "reason": "train.py not found"}

    model_path = workspace / "output"

    proc = _run(
        [
            INPAINT360GS_PYTHON,
            str(train_script),
            "-s", str(workspace),
            "--model_path", str(model_path),
            "-r", str(resolution),
            "--iterations", str(iterations),
            "--eval",
        ],
        cwd=INPAINT360GS_DIR,
        label="3DGS training",
        timeout=3600,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"train.py rc={proc.returncode}", "duration_s": duration}

    return {
        "status": "ok",
        "model_path": str(model_path),
        "iterations": iterations,
        "duration_s": round(duration, 1),
    }


# ---------------------------------------------------------------------------
# Stage 3: Semantic distillation
# ---------------------------------------------------------------------------

def run_distillation(
    *,
    workspace: Path,
    model_path: Path,
    iterations: int = INPAINT360GS_DISTILL_ITERS,
) -> Dict[str, Any]:
    """Distill 2D instance masks into per-Gaussian 16-dim object embeddings."""
    _log(f"Distilling semantic masks ({iterations} iters)...")
    t0 = time.monotonic()

    finetune_script = INPAINT360GS_DIR / "train_finetune.py"
    if not finetune_script.is_file():
        return {"status": "failed", "reason": "train_finetune.py not found"}

    proc = _run(
        [
            INPAINT360GS_PYTHON,
            str(finetune_script),
            "-s", str(workspace),
            "--model_path", str(model_path),
            "--finetune_semantic",
            "--iterations", str(iterations),
        ],
        cwd=INPAINT360GS_DIR,
        label="semantic distillation",
        timeout=1200,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"train_finetune.py rc={proc.returncode}", "duration_s": duration}

    return {"status": "ok", "duration_s": round(duration, 1)}


# ---------------------------------------------------------------------------
# Stage 4: Object removal
# ---------------------------------------------------------------------------

def run_object_removal(
    *,
    workspace: Path,
    model_path: Path,
    target_ids: List[int],
) -> Dict[str, Any]:
    """Remove Gaussians belonging to target objects."""
    _log(f"Removing {len(target_ids)} object(s): {target_ids}")
    t0 = time.monotonic()

    removal_script = INPAINT360GS_DIR / "edit_object_removal.py"
    if not removal_script.is_file():
        return {"status": "failed", "reason": "edit_object_removal.py not found"}

    # Inpaint360GS expects comma-separated target IDs
    target_str = ",".join(str(i) for i in target_ids)

    proc = _run(
        [
            INPAINT360GS_PYTHON,
            str(removal_script),
            "--model_path", str(model_path),
            "--target_id", target_str,
        ],
        cwd=INPAINT360GS_DIR,
        label="object removal",
        timeout=600,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"edit_object_removal.py rc={proc.returncode}", "duration_s": duration}

    return {"status": "ok", "target_ids": target_ids, "duration_s": round(duration, 1)}


# ---------------------------------------------------------------------------
# Stage 5: Virtual poses + LaMa inpainting
# ---------------------------------------------------------------------------

def run_virtual_poses_and_inpaint(
    *,
    workspace: Path,
    model_path: Path,
    expand_pixels: int = INPAINT360GS_LAMA_EXPAND_PX,
) -> Dict[str, Any]:
    """Generate virtual camera poses around removal regions and run LaMa 2D inpainting."""
    _log("Generating virtual poses + LaMa inpainting...")
    t0 = time.monotonic()

    # Virtual pose generation
    vpose_script = INPAINT360GS_DIR / "tools" / "virtual_pose.py"
    if vpose_script.is_file():
        proc = _run(
            [INPAINT360GS_PYTHON, str(vpose_script), "--model_path", str(model_path)],
            cwd=INPAINT360GS_DIR,
            label="virtual pose generation",
            timeout=300,
        )
        if proc.returncode != 0:
            _log(f"  Virtual pose generation failed (rc={proc.returncode}), continuing anyway...")

    # LaMa color inpainting
    color_script = INPAINT360GS_DIR / "predict_color.py"
    if color_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(color_script),
                "--model_path", str(model_path),
                "--expand", str(expand_pixels),
            ],
            cwd=INPAINT360GS_DIR,
            label="LaMa color inpainting",
            timeout=1200,
        )
        if proc.returncode != 0:
            return {"status": "failed", "reason": f"predict_color.py rc={proc.returncode}"}

    # LaMa depth inpainting
    depth_script = INPAINT360GS_DIR / "predict_depth.py"
    if depth_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(depth_script),
                "--model_path", str(model_path),
                "--expand", str(expand_pixels),
            ],
            cwd=INPAINT360GS_DIR,
            label="LaMa depth inpainting",
            timeout=1200,
        )
        if proc.returncode != 0:
            _log(f"  Depth inpainting failed, continuing with color only...")

    duration = time.monotonic() - t0
    return {"status": "ok", "duration_s": round(duration, 1)}


# ---------------------------------------------------------------------------
# Stage 6: PLY fusion + inpaint optimization
# ---------------------------------------------------------------------------

def run_inpaint_optimization(
    *,
    workspace: Path,
    model_path: Path,
    resolution: int = INPAINT360GS_RESOLUTION,
    iterations: int = INPAINT360GS_FINETUNE_ITERS,
) -> Dict[str, Any]:
    """Run PLY fusion and 3DGS inpainting optimization.

    Returns dict with path to the final inpainted PLY.
    """
    _log(f"Running inpaint optimization ({iterations} iters)...")
    t0 = time.monotonic()

    # PLY fusion
    fusion_script = INPAINT360GS_DIR / "edit_object_removal_plyfusion.py"
    if fusion_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(fusion_script),
                "--model_path", str(model_path),
            ],
            cwd=INPAINT360GS_DIR,
            label="PLY fusion",
            timeout=600,
        )
        if proc.returncode != 0:
            _log(f"  PLY fusion failed (rc={proc.returncode}), trying direct inpainting...")

    # Inpainting optimization
    inpaint_script = INPAINT360GS_DIR / "edit_object_inpaint.py"
    if not inpaint_script.is_file():
        return {"status": "failed", "reason": "edit_object_inpaint.py not found"}

    proc = _run(
        [
            INPAINT360GS_PYTHON,
            str(inpaint_script),
            "-s", str(workspace),
            "--model_path", str(model_path),
            "-r", str(resolution),
            "--iterations", str(iterations),
        ],
        cwd=INPAINT360GS_DIR,
        label="inpaint optimization",
        timeout=1800,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"edit_object_inpaint.py rc={proc.returncode}", "duration_s": duration}

    # Find the output PLY
    inpaint_dirs = sorted(model_path.glob("point_cloud_object_inpaint*/iteration_*/point_cloud.ply"))
    if not inpaint_dirs:
        # Fallback: look for any PLY in the model directory
        inpaint_dirs = sorted(model_path.glob("**/point_cloud.ply"))

    if not inpaint_dirs:
        return {"status": "failed", "reason": "no output PLY found after inpainting", "duration_s": duration}

    final_ply = inpaint_dirs[-1]  # Use the latest iteration
    _log(f"  Final inpainted PLY: {final_ply} ({final_ply.stat().st_size / 1024 / 1024:.1f}MB)")

    return {
        "status": "ok",
        "ply_path": str(final_ply),
        "duration_s": round(duration, 1),
    }


# ---------------------------------------------------------------------------
# Stage 7: PLY → GLB mesh conversion
# ---------------------------------------------------------------------------

def convert_gaussians_to_mesh(
    *,
    ply_path: Path,
    output_glb: Path,
    max_faces: int = INPAINT360GS_MAX_MESH_FACES,
) -> Dict[str, Any]:
    """Convert 3DGS PLY → triangle mesh → GLB via Poisson reconstruction.

    Uses Open3D point cloud → Poisson surface reconstruction, matching the
    existing ``build_gaussian_visual_mesh()`` pattern in nurec_shim.py.
    """
    _log(f"Converting PLY → GLB mesh (max_faces={max_faces})...")
    t0 = time.monotonic()

    try:
        import numpy as np
    except ImportError:
        return {"status": "failed", "reason": "numpy not available"}

    # Read PLY and extract positions + colors
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.io.read_point_cloud(str(ply_path))
        n_points = len(pcd.points)
        _log(f"  Loaded {n_points} points from PLY")

        if n_points < 100:
            return {"status": "failed", "reason": f"too few points ({n_points})"}

        # Estimate normals for Poisson reconstruction
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, scale=1.1, linear_fit=False
        )
        _log(f"  Poisson mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

        # Remove low-density vertices (floating artifacts)
        densities_np = np.asarray(densities)
        density_threshold = np.quantile(densities_np, 0.02)
        vertices_to_remove = densities_np < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Simplify if over budget
        n_faces = len(mesh.triangles)
        if n_faces > max_faces:
            mesh = mesh.simplify_quadric_decimation(max_faces)
            _log(f"  Simplified: {n_faces} → {len(mesh.triangles)} faces")

        # Export as GLB via trimesh (Open3D doesn't export GLB directly)
        try:
            import trimesh  # type: ignore
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            colors = None
            if mesh.has_vertex_colors():
                colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

            tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
            tm.export(str(output_glb))
        except ImportError:
            # Fallback: export as PLY and let downstream handle it
            o3d.io.write_triangle_mesh(str(output_glb.with_suffix(".ply")), mesh)
            return {"status": "failed", "reason": "trimesh not available for GLB export"}

    except ImportError:
        # Try trimesh directly as fallback
        try:
            import trimesh  # type: ignore
            mesh = trimesh.load_mesh(str(ply_path))
            if hasattr(mesh, "faces") and len(mesh.faces) > max_faces:
                mesh = mesh.simplify_quadric_decimation(max_faces)
            mesh.export(str(output_glb))
        except Exception as exc:
            return {"status": "failed", "reason": f"mesh conversion failed: {exc}"}

    duration = time.monotonic() - t0

    if not output_glb.is_file():
        return {"status": "failed", "reason": "output GLB not created"}

    size_mb = output_glb.stat().st_size / 1024 / 1024
    _log(f"  Output: {output_glb} ({size_mb:.1f}MB)")

    try:
        n_verts = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
        n_faces_out = len(mesh.triangles) if hasattr(mesh, "triangles") else (
            len(mesh.faces) if hasattr(mesh, "faces") else 0
        )
    except Exception:
        n_verts = 0
        n_faces_out = 0

    return {
        "status": "ok",
        "vertices": n_verts,
        "faces": n_faces_out,
        "file_size_mb": round(size_mb, 1),
        "duration_s": round(duration, 1),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_scene_cleaning(
    *,
    colmap_sparse_dir: Path,
    images_dir: Path,
    instance_masks_dir: Path,
    object_index_path: Path,
    output_dir: Path,
    resolution: int = INPAINT360GS_RESOLUTION,
    resume: bool = False,
    target_instance_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Top-level entry point for Stage 9.5 Scene Cleaning.

    Orchestrates all Inpaint360GS substeps. Returns a report dict:

    - ``status``: ``"ok"`` | ``"skipped"`` | ``"failed"``
    - ``inpainted_visual_glb``: path to clean GLB mesh (or ``None``)
    - ``timing``: per-substep durations
    - ``metrics``: training/removal stats

    On failure: logs a warning and returns gracefully so the pipeline
    can fall back to the original visual layer.
    """
    report_path = output_dir / "scene_cleaning_report.json"
    output_glb = output_dir / "inpainted_visual_mesh.glb"

    # Resume check
    if resume and report_path.is_file() and output_glb.is_file() and output_glb.stat().st_size > 0:
        _log("Resuming: using existing inpainted visual mesh")
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Validate Inpaint360GS installation
    probe = probe_installation()
    if probe.get("status") != "ok":
        report = {
            "status": "skipped",
            "reason": f"Inpaint360GS probe failed: {probe}",
            "probe": probe,
        }
        _log(f"Skipped: {report['reason']}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    # Validate instance masks
    if not instance_masks_dir.is_dir() or not any(instance_masks_dir.glob("*.png")):
        report = {"status": "skipped", "reason": "no instance masks available"}
        _log(f"Skipped: {report['reason']}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    # Validate object index
    if not object_index_path.is_file():
        report = {"status": "skipped", "reason": "no object index"}
        _log(f"Skipped: {report['reason']}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    try:
        with open(object_index_path, "r", encoding="utf-8") as f:
            obj_index = json.load(f)
        objects = obj_index.get("objects", [])
        if not objects:
            report = {"status": "skipped", "reason": "no objects in index"}
            _log(f"Skipped: {report['reason']}")
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
    except Exception as exc:
        report = {"status": "failed", "reason": f"failed to read object index: {exc}"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    timing: Dict[str, float] = {}
    t_total = time.monotonic()

    # Working directory for Inpaint360GS
    inpaint_workspace = output_dir / "_inpaint360gs_workspace"

    try:
        # Stage 1: Prepare data layout
        layout = prepare_data_layout(
            colmap_sparse_dir=colmap_sparse_dir,
            images_dir=images_dir,
            instance_masks_dir=instance_masks_dir,
            object_index_path=object_index_path,
            workspace=inpaint_workspace,
            resolution=resolution,
        )
        if target_instance_ids:
            resolved_targets: List[int] = []
            for value in target_instance_ids:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    resolved_targets.append(parsed)
            object_ids = sorted(set(resolved_targets))
        else:
            object_ids = layout["object_ids"]
        if not object_ids:
            report = {"status": "skipped", "reason": "no target instance IDs"}
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        _log(f"Data layout ready: {layout['num_images']} images, {layout['num_objects']} objects")

        # Stage 2: Train 3DGS
        train_result = run_training(workspace=inpaint_workspace, resolution=resolution)
        timing["training"] = train_result.get("duration_s", 0)
        if train_result["status"] != "ok":
            raise RuntimeError(f"3DGS training failed: {train_result.get('reason')}")
        model_path = Path(train_result["model_path"])

        # Stage 3: Semantic distillation
        distill_result = run_distillation(workspace=inpaint_workspace, model_path=model_path)
        timing["distillation"] = distill_result.get("duration_s", 0)
        if distill_result["status"] != "ok":
            raise RuntimeError(f"Distillation failed: {distill_result.get('reason')}")

        # Stage 4: Object removal
        removal_result = run_object_removal(
            workspace=inpaint_workspace,
            model_path=model_path,
            target_ids=object_ids,
        )
        timing["removal"] = removal_result.get("duration_s", 0)
        if removal_result["status"] != "ok":
            raise RuntimeError(f"Object removal failed: {removal_result.get('reason')}")

        # Stage 5: Virtual poses + LaMa inpainting
        inpaint_2d_result = run_virtual_poses_and_inpaint(
            workspace=inpaint_workspace,
            model_path=model_path,
        )
        timing["lama_inpainting"] = inpaint_2d_result.get("duration_s", 0)
        if inpaint_2d_result["status"] != "ok":
            raise RuntimeError(f"LaMa inpainting failed: {inpaint_2d_result.get('reason')}")

        # Stage 6: Inpaint optimization
        opt_result = run_inpaint_optimization(
            workspace=inpaint_workspace,
            model_path=model_path,
            resolution=resolution,
        )
        timing["inpaint_optimization"] = opt_result.get("duration_s", 0)
        if opt_result["status"] != "ok":
            raise RuntimeError(f"Inpaint optimization failed: {opt_result.get('reason')}")

        final_ply = Path(opt_result["ply_path"])

        # Stage 7: Convert PLY → GLB mesh
        mesh_result = convert_gaussians_to_mesh(
            ply_path=final_ply,
            output_glb=output_glb,
        )
        timing["mesh_conversion"] = mesh_result.get("duration_s", 0)
        if mesh_result["status"] != "ok":
            raise RuntimeError(f"Mesh conversion failed: {mesh_result.get('reason')}")

        total_duration = time.monotonic() - t_total
        timing["total"] = round(total_duration, 1)

        report = {
            "status": "ok",
            "inpainted_visual_glb": str(output_glb),
            "num_objects_removed": len(object_ids),
            "target_instance_ids": object_ids,
            "timing": timing,
            "metrics": {
                "training": train_result,
                "mesh": mesh_result,
                "probe": probe,
            },
        }
        _log(f"Scene cleaning complete in {total_duration:.0f}s")

    except Exception as exc:
        total_duration = time.monotonic() - t_total
        timing["total"] = round(total_duration, 1)
        _log(f"WARNING: Scene cleaning failed ({exc})")
        report = {
            "status": "failed",
            "reason": str(exc),
            "inpainted_visual_glb": None,
            "timing": timing,
        }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inpaint360GS scene cleaning")
    parser.add_argument("--probe", action="store_true", help="Validate Inpaint360GS installation and exit")
    parser.add_argument("--colmap-sparse-dir", required=False, type=Path, default=None,
                        help="Path to COLMAP sparse/0/ directory")
    parser.add_argument("--images-dir", required=False, type=Path, default=None,
                        help="Path to undistorted images directory")
    parser.add_argument("--instance-masks-dir", required=False, type=Path, default=None,
                        help="Path to SAM3 instance segmentation masks directory")
    parser.add_argument("--object-index", required=False, type=Path, default=None,
                        help="Path to object_point_cloud_index.json")
    parser.add_argument("--output-dir", required=False, type=Path, default=None,
                        help="Output directory for results")
    parser.add_argument("--resolution", type=int, default=INPAINT360GS_RESOLUTION,
                        help="Image downscale factor (1=full, 2=half, 4=quarter)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output if available")
    parser.add_argument(
        "--target-instance-ids",
        default="",
        help="Comma-separated list of instance_mask_id values to remove (default: all objects)",
    )

    args = parser.parse_args()

    if args.probe:
        result = probe_installation()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result.get("status") == "ok" else 1)

    missing = []
    if args.colmap_sparse_dir is None:
        missing.append("--colmap-sparse-dir")
    if args.images_dir is None:
        missing.append("--images-dir")
    if args.instance_masks_dir is None:
        missing.append("--instance-masks-dir")
    if args.object_index is None:
        missing.append("--object-index")
    if args.output_dir is None:
        missing.append("--output-dir")
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    target_ids: Optional[List[int]] = None
    raw_targets = [part.strip() for part in str(args.target_instance_ids or "").split(",") if part.strip()]
    if raw_targets:
        target_ids = []
        for value in raw_targets:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                target_ids.append(parsed)

    result = run_scene_cleaning(
        colmap_sparse_dir=args.colmap_sparse_dir,
        images_dir=args.images_dir,
        instance_masks_dir=args.instance_masks_dir,
        object_index_path=args.object_index,
        output_dir=args.output_dir,
        resolution=args.resolution,
        resume=args.resume,
        target_instance_ids=target_ids,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 1)
