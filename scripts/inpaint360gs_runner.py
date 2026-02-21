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
# cfg_args patcher
# ---------------------------------------------------------------------------

def _patch_cfg_args(
    cfg_args_path: Path,
    extra: Dict[str, Any],
    *,
    overwrite: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge *extra* keys into an existing ``cfg_args`` Namespace file.

    ``edit_object_removal.py`` uses ``ModelParams(parser, sentinel=True)``
    which sets every default to ``None``.  ``get_combined_args`` starts
    from ``cfg_args`` (written by vanilla ``train.py``) and only overwrites
    with non-None CLI values.  Because ``train.py``'s ModelParams lacks
    keys like ``object_path``, ``n_views``, ``random_init``, etc., they
    end up absent from the merged Namespace and Scene.__init__ raises
    ``AttributeError``.  Patching cfg_args is the safest fix — it does
    not depend on whether argparse treats the value as None.

    *extra* keys are only added when missing.  *overwrite* keys are always set.
    """
    if not cfg_args_path.is_file():
        _log(f"  WARNING: cfg_args not found at {cfg_args_path}, skipping patch")
        return

    from argparse import Namespace  # needed for eval and for writing back

    text = cfg_args_path.read_text(encoding="utf-8").strip()
    try:
        ns = eval(text)  # noqa: S307 — trusted file written by train.py
        ns_dict = vars(ns)
    except Exception as exc:
        _log(f"  WARNING: could not parse cfg_args ({exc}), skipping patch")
        return

    changed = False
    added_keys: list = []
    for key, value in extra.items():
        if key not in ns_dict:
            ns_dict[key] = value
            added_keys.append(key)
            changed = True

    overwritten_keys: list = []
    for key, value in (overwrite or {}).items():
        if ns_dict.get(key) != value:
            ns_dict[key] = value
            overwritten_keys.append(key)
            changed = True

    if changed:
        patched_ns = Namespace(**ns_dict)
        cfg_args_path.write_text(repr(patched_ns) + "\n", encoding="utf-8")
        if added_keys:
            _log(f"  Patched cfg_args — added: {added_keys}")
        if overwritten_keys:
            _log(f"  Patched cfg_args — overwritten: {overwritten_keys}")


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
    # Ensure Inpaint360GS root is on PYTHONPATH so `from utils import ...`
    # works even for scripts in subdirectories (e.g. seg/distillation.py).
    if cwd and str(INPAINT360GS_DIR) in str(cwd):
        existing = merged_env.get("PYTHONPATH", "")
        inpaint_root = str(INPAINT360GS_DIR)
        if inpaint_root not in existing:
            merged_env["PYTHONPATH"] = f"{inpaint_root}:{existing}" if existing else inpaint_root
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


def _python_has_module(
    *,
    module: str,
    python: Path = INPAINT360GS_PYTHON,
    extra_pythonpath: Optional[str] = None,
) -> bool:
    """Return whether *module* can be imported in the given interpreter."""
    try:
        env = os.environ.copy()
        if extra_pythonpath:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{extra_pythonpath}:{existing}" if existing else extra_pythonpath
            )
        proc = subprocess.run(
            [str(python), "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module}') else 1)"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=env,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _python_module_version(
    *,
    module: str,
    python: Path = INPAINT360GS_PYTHON,
    extra_pythonpath: Optional[str] = None,
) -> Optional[str]:
    """Return installed module version if importable, otherwise None."""
    try:
        env = os.environ.copy()
        if extra_pythonpath:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{extra_pythonpath}:{existing}" if existing else extra_pythonpath
            )
        proc = subprocess.run(
            [str(python), "-c", f"import importlib.metadata as m; print(m.version('{module}'))"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=env,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or None
    except Exception:
        return None


def _ensure_python_module(
    *,
    module: str,
    expected_version: Optional[str] = None,
    python: Path = INPAINT360GS_PYTHON,
    timeout: int = 1200,
) -> bool:
    """Ensure *module* is importable (and version matches optional constraint)."""
    import_name = module
    package_spec = module
    if expected_version:
        package_spec = f"{module}=={expected_version}"
        import_name = module
    if _python_has_module(module=import_name, python=python):
        if expected_version is None:
            return True
        version = _python_module_version(module=import_name, python=python)
        if version == expected_version:
            return True
        _log(f"  Module '{import_name}' version mismatch: got {version}, expected {expected_version}")

    _log(f"  Missing/unsupported python module '{module}' — attempting pip install --user {package_spec}")
    install_proc = _run(
        [str(python), "-m", "pip", "install", "--user", package_spec],
        label=f"install python module {package_spec}",
        timeout=timeout,
    )
    if install_proc.returncode != 0:
        _log(f"  pip install for '{package_spec}' failed (rc={install_proc.returncode})")
        return False

    if not _python_has_module(module=import_name, python=python):
        return False
    if expected_version is not None:
        return _python_module_version(module=import_name, python=python) == expected_version
    return True


def _ensure_minimal_easydict_stub(workspace: Path) -> None:
    """Create a tiny local easydict shim if external dependency cannot be installed."""
    stub_path = workspace / "easydict.py"
    if stub_path.exists():
        return
    stub_path.write_text(
        "\n".join(
            [
                "class EasyDict(dict):",
                "    def __getattr__(self, name):",
                "        try:",
                "            return self[name]",
                "        except KeyError as exc:",
                "            raise AttributeError(name) from exc",
                "    def __setattr__(self, name, value):",
                "        self[name] = value",
                "    __setitem__ = dict.__setitem__",
                "    def __delattr__(self, name):",
                "        try:",
                "            del self[name]",
                "        except KeyError as exc:",
                "            raise AttributeError(name) from exc",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Stage 5 helpers
# ---------------------------------------------------------------------------

def _build_lama_masks_from_virtual_outputs(
    *,
    workspace: Path,
    model_path: Path,
    iteration: int,
    lama_workspace: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> None:
    """Populate ``Segment-and-Track-Anything/tracking_results/images/images_masks``.

    Upstream ``prepare_lama_data.py`` expects masks under a hardcoded
    path in the Inpaint360GS repo root. The repository does not populate
    this path in the local setup, so we mirror masks from the virtual
    pose outputs that Inpaint360GS just produced.
    """
    if lama_workspace is None:
        lama_workspace = INPAINT360GS_DIR
    target_dir = lama_workspace / "Segment-and-Track-Anything" / "tracking_results" / "images" / "images_masks"
    target_dir.mkdir(parents=True, exist_ok=True)
    for existing_mask in target_dir.glob("*.png"):
        try:
            existing_mask.unlink()
        except OSError:
            pass

    # Resolve selected object IDs from the config, if available.
    selected_ids: List[int] = []
    if config_path is not None and config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            select_obj_id = cfg.get("select_obj_id", [])
            for value in select_obj_id:
                if isinstance(value, (int, str)):
                    try:
                        selected_ids.append(int(value))
                    except ValueError:
                        continue
        except Exception:
            selected_ids = []
    selected_ids = sorted(set([v for v in selected_ids if v > 0]))

    # Candidates are created by virtual_pose.py and contain per-frame masks.
    candidates = [
        workspace / "inpaint_2d_unseen_mask_virtual",
        model_path / "virtual" / "ours_object_removal" / f"iteration_{iteration}" / "objects_pred",
        model_path / "virtual" / "ours_object_removal" / f"iteration_{iteration}" / "gt_objects_color",
    ]
    source_dir: Optional[Path] = None
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.png")):
            source_dir = candidate
            break

    if source_dir is None:
        # Try one more place: full-view renders (fallback to no-op inpainting masks).
        render_dir = model_path / "virtual" / "ours_object_removal" / f"iteration_{iteration}" / "renders"
        if render_dir.is_dir() and any(render_dir.glob("*.png")):
            source_dir = render_dir

    if source_dir is None:
        _log(f"  WARN: no virtual mask source found for LaMa under iteration_{iteration}")
        return

    # Build hardcoded-tracking masks from the best source available.
    try:
        from PIL import Image
        import numpy as np

        for mask_file in source_dir.glob("*.png"):
            src_img = Image.open(mask_file).convert("L")
            arr = np.array(src_img, dtype=np.int32)
            if source_dir.name == "renders" and arr.ndim == 3:
                # Render RGB -> convert to a non-zero mask.
                arr = arr.mean(axis=2)

            if selected_ids:
                # If IDs are within range and masks were actually encoded as IDs,
                # keep only selected IDs. Otherwise use any non-zero object.
                if arr.max() <= 255 and all(value <= 255 for value in selected_ids):
                    keep = np.isin(arr, selected_ids)
                else:
                    keep = arr != 0
            else:
                keep = arr != 0

            mask = (keep.astype(np.uint8) * 255)
            dst = target_dir / mask_file.name
            Image.fromarray(mask).save(dst)
    except Exception:
        # Fall back to raw copy if image libs are unavailable.
        for mask_file in source_dir.glob("*.png"):
            shutil.copy2(mask_file, target_dir / mask_file.name)


def _ensure_lama_workspace(workspace: Path) -> Path:
    """Create a writable LaMa staging directory with the expected relative layout."""
    lama_workspace = workspace / "_lama_workspace"
    lama_workspace.mkdir(exist_ok=True)

    def _safe_link(name: str, target: Path) -> None:
        link = lama_workspace / name
        if link.is_symlink() or link.exists():
            return
        if target.is_dir():
            link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target)

    # Inpaint360GS expects these relative paths:
    # - Segment-and-Track-Anything/.../images_masks
    # - LaMa/data
    # - LaMa/output
    # - configs (for default prediction config)
    # - big-lama weights
    # - config/object_distill/train_distill.json
    _safe_link("Segment-and-Track-Anything", INPAINT360GS_DIR / "Segment-and-Track-Anything")
    _safe_link("configs", INPAINT360GS_DIR / "LaMa" / "configs")
    _safe_link("big-lama", INPAINT360GS_DIR / "LaMa" / "big-lama")
    _safe_link("config", INPAINT360GS_DIR / "config")

    lama_repo_dir = lama_workspace / "LaMa"
    lama_repo_dir.mkdir(exist_ok=True)
    (lama_repo_dir / "data").mkdir(parents=True, exist_ok=True)
    (lama_repo_dir / "output").mkdir(parents=True, exist_ok=True)

    # Keep helper aliases consistent with prepare/predict expectations.
    for alias in ["data", "output"]:
        alias_path = lama_workspace / alias
        target = lama_repo_dir / alias
        if not alias_path.exists():
            alias_path.symlink_to(target, target_is_directory=True)

    return lama_workspace


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
        # Need to downscale masks to match Inpaint360GS render resolution.
        # camera_utils.py uses math.ceil(dim / resolution) for images but
        # never resizes object masks, so masks must match exactly.
        import math
        _log(f"  Rescaling {len(mask_files)} instance masks to 1/{resolution} resolution")
        try:
            from PIL import Image as PILImage
            import numpy as np
            for mf in mask_files:
                img = PILImage.open(mf)
                new_w = max(1, math.ceil(img.width / resolution))
                new_h = max(1, math.ceil(img.height / resolution))
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
    model_path = workspace / "output"
    ckpt_dir = model_path / "point_cloud" / f"iteration_{iterations}"
    if ckpt_dir.is_dir() and any(ckpt_dir.glob("point_cloud.*")):
        _log(f"Reusing existing 3DGS training at {ckpt_dir}")
        return {"status": "ok", "model_path": str(model_path), "duration_s": 0.0, "reused": True}

    _log(f"Training 3DGS (resolution=1/{resolution}, iters={iterations})...")
    t0 = time.monotonic()

    train_script = INPAINT360GS_DIR / "train.py"
    if not train_script.is_file():
        return {"status": "failed", "reason": "train.py not found"}

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
    """Distill 2D instance masks into per-Gaussian object embeddings.

    Runs ``seg/distillation.py`` which trains a lightweight classifier
    on top of frozen Gaussian object features and saves ``classifier.pth``
    alongside the point cloud checkpoint.  This is required before
    ``edit_object_removal.py`` can identify which Gaussians to remove.
    """
    _log(f"Running semantic distillation ({iterations} iters)...")

    # Check for existing classifier (resume)
    # distillation saves classifier at iteration_{iterations}/classifier.pth
    classifier_path = model_path / "point_cloud" / f"iteration_{iterations}" / "classifier.pth"
    if classifier_path.is_file():
        _log(f"  Reusing existing classifier at {classifier_path}")
        return {"status": "ok", "duration_s": 0.0, "reused": True}

    # Also check if classifier exists at the vanilla training iteration
    for ckpt_dir in sorted(model_path.glob("point_cloud/iteration_*")):
        existing_cls = ckpt_dir / "classifier.pth"
        if existing_cls.is_file():
            _log(f"  Reusing existing classifier at {existing_cls}")
            return {"status": "ok", "duration_s": 0.0, "reused": True}

    distill_script = INPAINT360GS_DIR / "seg" / "distillation.py"
    if not distill_script.is_file():
        return {"status": "failed", "reason": "seg/distillation.py not found"}

    t0 = time.monotonic()

    # Write distillation config
    config_path = workspace / "distill_config.json"
    config = {
        "reg3d_interval": 50,
        "reg3d_k": 5,
        "reg3d_lambda_val": 2,
        "reg3d_max_points": 200000,
        "reg3d_sample_size": 1000,
        "iterations": iterations,
        "train_distill": True,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Patch cfg_args for distillation (needs vanilla_3dgs_path etc.)
    cfg_args_path = model_path / "cfg_args"
    _patch_cfg_args(cfg_args_path, {
        "object_path": "associated_hqsam",
        "vanilla_3dgs_path": str(model_path),
        "n_views": 100,
        "random_init": False,
        "train_split": False,
        "num_classes": -1,
        "train_distill": True,
    })

    proc = _run(
        [
            INPAINT360GS_PYTHON,
            str(distill_script),
            "-s", str(workspace),
            "--model_path", str(model_path),
            "--vanilla_3dgs_path", str(model_path),
            "--object_path", "associated_hqsam",
            "-r", str(INPAINT360GS_RESOLUTION),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--train_distill",
            "--config_file", str(config_path),
        ],
        cwd=INPAINT360GS_DIR,
        label="semantic distillation",
        timeout=3600,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"distillation.py rc={proc.returncode}", "duration_s": duration}

    # Verify classifier was produced
    found_classifiers = list(model_path.glob("point_cloud/iteration_*/classifier.pth"))
    if not found_classifiers:
        return {"status": "failed", "reason": "classifier.pth not produced", "duration_s": duration}

    latest_classifier = sorted(found_classifiers)[-1]
    _log(f"  Classifier saved at {latest_classifier}")

    # Copy classifier.pth to ALL checkpoint directories that lack it.
    # edit_object_removal.py loads the classifier from iteration_{loaded_iter}
    # which may be the vanilla training iteration (e.g. 30000), not the
    # distillation iteration (e.g. 2000).
    import shutil as _shutil
    for ckpt_dir in sorted(model_path.glob("point_cloud/iteration_*")):
        cls_dst = ckpt_dir / "classifier.pth"
        if not cls_dst.is_file():
            _shutil.copy2(str(latest_classifier), str(cls_dst))
            _log(f"  Copied classifier.pth → {ckpt_dir.name}")

    return {
        "status": "ok",
        "classifier_path": str(latest_classifier),
        "iterations": iterations,
        "duration_s": round(duration, 1),
    }


# ---------------------------------------------------------------------------
# Stage 4: Object removal
# ---------------------------------------------------------------------------

def run_object_removal(
    *,
    workspace: Path,
    model_path: Path,
    target_ids: List[int],
) -> Dict[str, Any]:
    """Run distillation + object removal via edit_object_removal.py."""
    _log(f"Removing {len(target_ids)} object(s) (with inline distillation)")
    t0 = time.monotonic()

    # Check for existing removal output (resume from previous run)
    for ckpt_dir in sorted(model_path.glob("point_cloud_object_removal/iteration_*")):
        removal_ply = ckpt_dir / "point_cloud.ply"
        if removal_ply.is_file() and removal_ply.stat().st_size > 1_000_000:
            _log(f"  Reusing existing removal output at {removal_ply} ({removal_ply.stat().st_size / 1024 / 1024:.0f}MB)")
            return {"status": "ok", "target_ids": target_ids, "duration_s": 0.0, "reused": True}

    removal_script = INPAINT360GS_DIR / "edit_object_removal.py"
    if not removal_script.is_file():
        return {"status": "failed", "reason": "edit_object_removal.py not found"}

    # Write a config file with target IDs for removal.
    # NOTE: train_distill=False here — removal loads the DISTILLED model
    # directly (with object features), not the vanilla model.
    config_path = workspace / "removal_config.json"
    config = {
        "removal_thresh": INPAINT360GS_REMOVAL_THRESH,
        "select_obj_id": target_ids,
        # target_id == select_obj_id means "remove ALL selected objects."
        # When target_id is a strict subset of select_obj_id, the script
        # re-combines the non-targeted objects back into the scene.
        "target_id": target_ids,
        # Downstream inpaint stages:
        "object_path": "inpaint_2d_unseen_mask_virtual",
        "images": "images_inpaint_unseen_virtual",
        "surrounding_ids": [],
        "lambda_dssim": 0.8,
        "opacity_init": 0.1,
        "lambda_lpips": 0.0005,
        "finetune_iteration": INPAINT360GS_FINETUNE_ITERS,
        "circle_radius": 1.0,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Read num_classes from scene.json (needed by the removal script).
    scene_json = workspace / "associated_hqsam" / "scene.json"
    num_classes = -1
    if scene_json.is_file():
        try:
            num_classes = json.loads(scene_json.read_text(encoding="utf-8")).get("num_classes", -1)
        except Exception:
            pass

    # Patch cfg_args so get_combined_args() merges the extra keys that
    # edit_object_removal.py's ModelParams (sentinel=True) needs but
    # vanilla train.py never wrote.  Without this, Scene.__init__
    # throws AttributeError for object_path / n_views / etc.
    # train_distill=False for removal (we load the distilled checkpoint).
    cfg_args_path = model_path / "cfg_args"
    _patch_cfg_args(
        cfg_args_path,
        {
            "object_path": "associated_hqsam",
            "vanilla_3dgs_path": str(model_path),
            "n_views": 100,
            "random_init": False,
            "train_split": False,
        },
        overwrite={
            "num_classes": num_classes,
            # MUST be False for removal — we load the distilled model
            # (with object features), not the vanilla model.
            "train_distill": False,
        },
    )

    # Use the distillation checkpoint which contains per-Gaussian object
    # features.  The vanilla training checkpoint (iteration_30000) only has
    # geometry — no object features — so the classifier can't map Gaussians
    # to semantic IDs.  Always prefer the distillation iteration.
    distill_iter = INPAINT360GS_DISTILL_ITERS
    distill_ckpt = model_path / "point_cloud" / f"iteration_{distill_iter}"
    if not (distill_ckpt / "point_cloud.ply").is_file():
        # Fallback: scan for any checkpoint with classifier.pth, prefer
        # the *lowest* iteration (distillation runs fewer iters than vanilla).
        for ckpt_dir in sorted(model_path.glob("point_cloud/iteration_*")):
            if (ckpt_dir / "classifier.pth").is_file() and (ckpt_dir / "point_cloud.ply").is_file():
                try:
                    distill_iter = int(ckpt_dir.name.split("_")[-1])
                except ValueError:
                    pass
                break
    _log(f"  Loading distilled checkpoint at iteration {distill_iter}")

    proc = _run(
        [
            INPAINT360GS_PYTHON,
            str(removal_script),
            "-s", str(workspace),
            "--model_path", str(model_path),
            "--object_path", "associated_hqsam",
            "--num_classes", str(num_classes),
            "--n_views", "100",
            "--iteration", str(distill_iter),
            "--config_file", str(config_path),
        ],
        cwd=INPAINT360GS_DIR,
        label="object removal",
        timeout=1800,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"edit_object_removal.py rc={proc.returncode}", "duration_s": duration}

    return {"status": "ok", "target_ids": target_ids, "duration_s": round(duration, 1)}


def _lama_data_name(workspace: Path) -> str:
    """Build a deterministic LaMa data_name for the current workspace."""
    safe_name = workspace.name or "scene"
    safe_name = safe_name.replace(" ", "_")
    return f"360_{safe_name}_virtual"


# ---------------------------------------------------------------------------
# Stage 5: Virtual poses + LaMa inpainting
# ---------------------------------------------------------------------------

def run_virtual_poses_and_inpaint(
    *,
    workspace: Path,
    model_path: Path,
    config_path: Optional[Path] = None,
    expand_pixels: int = INPAINT360GS_LAMA_EXPAND_PX,
) -> Dict[str, Any]:
    """Generate virtual camera poses around removal regions and run LaMa 2D inpainting."""
    _log("Generating virtual poses + LaMa inpainting...")
    t0 = time.monotonic()

    # LaMa lives inside Inpaint360GS/LaMa — add to PYTHONPATH.
    lama_dir = INPAINT360GS_DIR / "LaMa"
    lama_env: Dict[str, str] = {}
    lama_workspace = _ensure_lama_workspace(workspace)
    if lama_dir.is_dir():
        user_site = subprocess.run(
            [INPAINT360GS_PYTHON, "-c", "import site,sys; print(site.getusersitepackages() or '')"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        paths = [str(lama_dir)]
        if user_site:
            paths.append(user_site)
        paths.append(str(lama_workspace))
        lama_env["PYTHONPATH"] = ":".join(p for p in paths if p)

    # Build config_file arg (needed by virtual_pose.py)
    if config_path is None:
        config_path = workspace / "removal_config.json"
    config_args = ["--config_file", str(config_path)] if config_path.is_file() else []
    data_name = _lama_data_name(workspace)
    virtual_iteration = INPAINT360GS_DISTILL_ITERS

    # Virtual pose generation (needs to run first so virtual masks exist locally).
    vpose_script = INPAINT360GS_DIR / "tools" / "virtual_pose.py"
    if vpose_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(vpose_script),
                "-s", str(workspace),
                "--model_path", str(model_path),
                "--object_path", "associated_hqsam",
                "--iteration", str(virtual_iteration),
                *config_args,
            ],
            cwd=INPAINT360GS_DIR,
            label="virtual pose generation",
            timeout=600,
        )
        if proc.returncode != 0:
            _log(f"  Virtual pose generation failed (rc={proc.returncode}), continuing anyway...")

    # Step 0: ensure legacy tracking_results path exists for prepare_lama_data.py.
    _build_lama_masks_from_virtual_outputs(
        workspace=workspace,
        model_path=model_path,
        iteration=virtual_iteration,
        lama_workspace=lama_workspace,
        config_path=config_path if config_path.is_file() else None,
    )

    # Prepare inpaint inputs for LaMa using the prepared masks.
    prepare_script = INPAINT360GS_DIR / "tools" / "prepare_lama_data.py"
    if prepare_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(prepare_script),
                "-s", str(workspace),
                "-m", str(model_path),
                "-r", str(INPAINT360GS_RESOLUTION),
                "--inpaint2lama",
            ],
            cwd=lama_workspace,
            label="prepare LaMa input",
            timeout=600,
        )
        if proc.returncode != 0:
            return {"status": "failed", "reason": f"prepare_lama_data.py rc={proc.returncode}"}

    # LaMa color inpainting
    color_script = INPAINT360GS_DIR / "predict_color.py"
    if color_script.is_file():
        for module in ("easydict", "kornia", "albumentations"):
            if not _ensure_python_module(module=module):
                _log(f"  Warning: failed to install {module}; attempting local fallback where available")
        if not _ensure_python_module(module="easydict"):
            _ensure_minimal_easydict_stub(lama_workspace)
            if not _python_has_module(
                module="easydict",
                python=INPAINT360GS_PYTHON,
                extra_pythonpath=lama_env.get("PYTHONPATH", ""),
            ):
                return {"status": "failed", "reason": "easydict dependency unavailable and shim creation failed"}
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(color_script),
                "--model_path", str(model_path),
                "--expand", str(expand_pixels),
                "--data_name", data_name,
            ],
            cwd=lama_workspace,
            env=lama_env,
            label="LaMa color inpainting",
            timeout=1200,
        )
        if proc.returncode != 0:
            return {"status": "failed", "reason": f"predict_color.py rc={proc.returncode}"}

    # LaMa depth inpainting
    depth_script = INPAINT360GS_DIR / "predict_depth.py"
    if depth_script.is_file():
        for module in ("easydict", "kornia", "albumentations"):
            if not _ensure_python_module(module=module):
                _log(f"  Warning: failed to install {module}; continuing with existing environment")
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(depth_script),
                "--model_path", str(model_path),
                "--expand", str(expand_pixels),
                "--data_name", data_name,
            ],
            cwd=lama_workspace,
            env=lama_env,
            label="LaMa depth inpainting",
            timeout=1200,
        )
        if proc.returncode != 0:
            _log(f"  Depth inpainting failed, continuing with color only...")

    # Step 3: copy LaMa outputs back to workspace for fusion/optimization.
    if prepare_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(prepare_script),
                "-s", str(workspace),
                "-m", str(model_path),
                "-r", str(INPAINT360GS_RESOLUTION),
            ],
            cwd=lama_workspace,
            label="copy LaMa outputs",
            timeout=600,
        )
        if proc.returncode != 0:
            return {"status": "failed", "reason": f"prepare_lama_data.py rc={proc.returncode}"}

    duration = time.monotonic() - t0
    return {"status": "ok", "duration_s": round(duration, 1)}


# ---------------------------------------------------------------------------
# Stage 6: PLY fusion + inpaint optimization
# ---------------------------------------------------------------------------

def run_inpaint_optimization(
    *,
    workspace: Path,
    model_path: Path,
    config_path: Optional[Path] = None,
    resolution: int = INPAINT360GS_RESOLUTION,
    iterations: int = INPAINT360GS_FINETUNE_ITERS,
    num_classes: int = -1,
    distillation_iteration: int = INPAINT360GS_DISTILL_ITERS,
) -> Dict[str, Any]:
    """Run PLY fusion and 3DGS inpainting optimization.

    Returns dict with path to the final inpainted PLY.
    """
    _log(f"Running inpaint optimization ({iterations} iters)...")
    t0 = time.monotonic()

    if config_path is None:
        config_path = workspace / "removal_config.json"

    if num_classes < 0:
        scene_json = workspace / "associated_hqsam" / "scene.json"
        num_classes = -1
        if scene_json.is_file():
            try:
                num_classes = json.loads(scene_json.read_text(encoding="utf-8")).get("num_classes", -1)
            except Exception:
                pass

    # PLY fusion
    fusion_script = INPAINT360GS_DIR / "edit_object_removal_plyfusion.py"
    if fusion_script.is_file():
        proc = _run(
            [
                INPAINT360GS_PYTHON,
                str(fusion_script),
                "--model_path", str(model_path),
                "-s", str(workspace),
                "--config_file", str(config_path),
                "--iteration", str(distillation_iteration),
            ],
            cwd=INPAINT360GS_DIR,
            label="PLY fusion",
            timeout=600,
        )
        if proc.returncode != 0:
            _log(f"  PLY fusion failed (rc={proc.returncode}), trying direct inpainting...")

    # Ensure downstream stage has dataset model args required by inpaint parser.
    cfg_args_path = model_path / "cfg_args"
    _patch_cfg_args(
        cfg_args_path,
        {
            "object_path": "associated_hqsam",
            "vanilla_3dgs_path": str(model_path),
            "n_views": 100,
            "random_init": False,
            "train_split": False,
        },
        overwrite={
            "num_classes": num_classes,
            "train_distill": False,
        },
    )

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
            "--iteration", str(distillation_iteration),
            "--iterations", str(iterations),
            "--config_file", str(config_path),
        ],
        cwd=INPAINT360GS_DIR,
        label="inpaint optimization",
        timeout=1800,
    )

    duration = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"edit_object_inpaint.py rc={proc.returncode}", "duration_s": duration}

    # Find the output PLY written by edit_object_inpaint.py.
    candidate_plys = [
        model_path / "point_cloud" / "_object_inpaint_virtual" / f"iteration_{iterations}" / "point_cloud.ply",
        model_path / "point_cloud_object_inpaint_virtual" / f"iteration_{iterations}" / "point_cloud.ply",
        model_path / "point_cloud_object_inpaint" / f"iteration_{iterations}" / "point_cloud.ply",
    ]
    inpaint_dirs = [path for path in candidate_plys if path.is_file()]

    # Backward-compatible fallback for any inpaint output naming.
    if not inpaint_dirs:
        inpaint_dirs = sorted(model_path.glob("point_cloud/**/*/point_cloud.ply"))

    if not inpaint_dirs:
        return {"status": "failed", "reason": "no output PLY found after inpainting", "duration_s": duration}

    final_ply = inpaint_dirs[-1]  # Use the latest candidate
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
        import open3d as o3d  # type: ignore
    except ImportError:
        return {"status": "failed", "reason": "numpy or open3d not available"}

    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_points = len(pcd.points)
    _log(f"  Loaded {n_points} points from PLY")

    if n_points < 100:
        return {"status": "failed", "reason": f"too few points ({n_points})"}

    # Keep Poisson reconstruction tractable for dense Gaussians. Use voxel
    # downsampling (NOT random) to preserve uniform spatial coverage — random
    # sampling creates holes that cause Poisson to produce tiny meshes.
    if n_points > 300_000:
        voxel_size = 0.01
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        _log(f"  Downsampled for mesh conversion: {len(pcd.points)} points (voxel={voxel_size})")

    # Clear pre-existing normals — Gaussian PLY files store rotation normals
    # (arbitrary orientation) that corrupt Poisson surface reconstruction.
    pcd.normals = o3d.utility.Vector3dVector()

    # Estimate proper surface normals for Poisson reconstruction
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson surface reconstruction (depth=10 for detailed mesh)
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
    output_ply = output_dir / "inpainted_gaussian_splat.ply"

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
        num_classes = layout["num_objects"] + 1
        config_path = inpaint_workspace / "removal_config.json"
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

        warnings: List[str] = []

        # Stage 5: Virtual poses + LaMa inpainting
        inpaint_2d_result = run_virtual_poses_and_inpaint(
            workspace=inpaint_workspace,
            model_path=model_path,
            config_path=config_path,
        )
        timing["lama_inpainting"] = inpaint_2d_result.get("duration_s", 0)
        if inpaint_2d_result["status"] != "ok":
            warnings.append(f"LaMa stage skipped: {inpaint_2d_result.get('reason')}")

        final_ply: Optional[Path] = None
        if inpaint_2d_result["status"] == "ok":
            # Stage 6: Inpaint optimization
            opt_result = run_inpaint_optimization(
                workspace=inpaint_workspace,
                model_path=model_path,
                config_path=config_path,
                resolution=resolution,
                iterations=INPAINT360GS_FINETUNE_ITERS,
                num_classes=num_classes,
                distillation_iteration=INPAINT360GS_DISTILL_ITERS,
            )
            timing["inpaint_optimization"] = opt_result.get("duration_s", 0)
            if opt_result["status"] == "ok":
                final_ply = Path(opt_result["ply_path"])
            else:
                warnings.append(f"Inpaint optimization failed: {opt_result.get('reason')}")
        else:
            timing["inpaint_optimization"] = 0

        if final_ply is None:
            final_ply_fallback = model_path / "point_cloud_object_removal" / f"iteration_{INPAINT360GS_DISTILL_ITERS}" / "point_cloud.ply"
            if not final_ply_fallback.is_file():
                raise RuntimeError("No optimized PLY available and no removal fallback PLY was found")
            final_ply = final_ply_fallback
            warnings.append("Using point_cloud_object_removal output as final artifact")
            _log(f"  Fallback PLY for final output: {final_ply}")

        # Copy inpainted PLY to output directory as a first-class artifact
        shutil.copy2(str(final_ply), str(output_ply))
        _log(f"Copied inpainted PLY to {output_ply} ({output_ply.stat().st_size / 1024 / 1024:.1f}MB)")

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
            "warnings": warnings,
            "inpainted_visual_glb": str(output_glb),
            "inpainted_gaussian_ply": str(output_ply) if output_ply.is_file() else None,
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
            "inpainted_gaussian_ply": None,
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
