"""Swappable 3DGS backend registry (decode / render / export / enhance).

Per ``WORLD_MODEL_STRATEGY_CONTEXT`` the model/render backends must stay swappable. This
module is the single registration + discovery point for the Gaussian-splat frameworks the
pipeline can use, so new ones plug in without touching core code. Each backend exposes a
uniform, fail-closed ``run()`` contract and an ``available()`` probe.

Built-in backends:
* ``splat_transform`` (decoder)  — PlayCanvas: compressed PLY/SPZ -> standard 3DGS PLY
* ``spark``          (renderer)  — Spark.js/three.js headless reference render (local)
* ``threedgrut``     (exporter)  — NVIDIA 3dgrut: standard PLY -> NuRec USDZ / ParticleField
* ``isaac_nurec``    (renderer)  — Isaac Sim RTX/NuRec-3DGUT render (GPU worker)
* ``artifixer``      (enhancer)  — NVIDIA ArtiFixer: diffusion artifact-fix / novel-view
                                   frames from sparse 3DGRUT reconstructions (GPU)

Adding a framework = ``register_backend(SplatBackend(...))`` with an availability probe and
a fail-closed ``run`` — no core change required.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BACKEND_KINDS = ("decoder", "renderer", "exporter", "enhancer")


@dataclass
class SplatBackend:
    name: str
    kind: str
    summary: str
    requires: tuple[str, ...]
    available: Callable[[], bool]
    run: Callable[..., dict]

    def describe(self) -> dict:
        try:
            avail = bool(self.available())
        except Exception:  # noqa: BLE001 - availability probes must never raise
            avail = False
        return {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "requires": list(self.requires),
            "available": avail,
        }


_REGISTRY: dict[str, SplatBackend] = {}


def register_backend(backend: SplatBackend) -> None:
    if backend.kind not in BACKEND_KINDS:
        raise ValueError(f"unknown backend kind: {backend.kind}")
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> SplatBackend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown splat backend: {name} (have: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def list_backends(kind: str | None = None) -> list[dict]:
    return [b.describe() for b in _REGISTRY.values() if kind is None or b.kind == kind]


def _node_available() -> bool:
    return shutil.which("node") is not None


# ---------------- built-in backend wrappers (delegate to existing fail-closed impls) ---

def _splat_transform_available() -> bool:
    from .gaussian_splat_decode import find_splat_transform_cli

    return _node_available() and find_splat_transform_cli() is not None


def _splat_transform_run(src, dst, **kwargs) -> dict:
    from .gaussian_splat_decode import convert_to_standard_ply

    return convert_to_standard_ply(src, dst, **kwargs)


def _spark_available() -> bool:
    from .splat_scene_render import RENDER_HARNESS_REL

    root = Path(__file__).resolve().parents[2]
    return _node_available() and (root / RENDER_HARNESS_REL).is_file()


def _spark_run(source, out_dir, **kwargs) -> dict:
    from .splat_scene_render import render_splat_scene

    return render_splat_scene(source, out_dir, **kwargs)


def _threedgrut_available() -> bool:
    from .isaac_nurec_export import threedgrut_available

    return threedgrut_available()


def _threedgrut_run(ply, out, *, fmt="nurec", **kwargs) -> dict:
    from .isaac_nurec_export import convert_ply_to_isaac_usd

    return convert_ply_to_isaac_usd(ply, out, fmt=fmt, **kwargs)


def _particlefield_available() -> bool:
    return importlib.util.find_spec("pxr") is not None


def _particlefield_run(source, out, **kwargs) -> dict:
    from .particlefield_usd import write_particlefield_usd

    return write_particlefield_usd(source, out, **kwargs)


def _isaac_nurec_available() -> bool:
    return importlib.util.find_spec("isaacsim") is not None


def _isaac_nurec_run(**kwargs) -> dict:
    # Isaac RTX/NuRec render runs on the GPU worker via the provider bundle, not in-process.
    return {
        "status": "blocked",
        "blockers": ["isaac_nurec_render_is_gpu_worker_only"],
        "runner": "scripts/run_isaac_splat_nurec_render.py",
        "note": "stage the standard PLY + cameras.json into the provider bundle and run the "
        "runner on the Isaac GPU worker",
    }


ARTIFIXER_INFERENCE_MODULE = "model_eval.run_inference"


def _artifixer_available() -> bool:
    # nv-tlabs/artifixer exposes the model_eval package when installed/on PYTHONPATH.
    try:
        return importlib.util.find_spec("model_eval") is not None
    except (ImportError, ValueError):
        return False


def _artifixer_run(
    *,
    checkpoint_pt,
    save_dir,
    split_path,
    evalset: str = "reconstructed_colmap",
    render_trajectory: str = "all_frames",
    python: str = "python",
    timeout_seconds: int = 7200,
) -> dict:
    """NVIDIA ArtiFixer: diffusion artifact-fix / novel-view frames from a sparse 3DGRUT
    reconstruction. Fail-closed wrapper around its documented inference CLI."""
    if not _artifixer_available():
        return {
            "status": "blocked",
            "blockers": ["artifixer_unavailable"],
            "remediation": "install nv-tlabs/artifixer (GPU, CUDA 12/13) and supply a checkpoint",
        }
    exe = shutil.which(python) or python
    cmd = [
        exe, "-m", ARTIFIXER_INFERENCE_MODULE,
        "--evalset", evalset,
        "--checkpoint_pt", str(checkpoint_pt),
        "--save_dir", str(save_dir),
        "--split_path", str(split_path),
        "--render_trajectory", render_trajectory,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["artifixer_exception"], "error": repr(exc)}
    if proc.returncode != 0:
        return {
            "status": "blocked",
            "blockers": ["artifixer_failed"],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "command": " ".join(cmd),
        }
    return {"status": "completed", "save_dir": str(save_dir), "command": " ".join(cmd), "enhancer": "artifixer"}


def _register_builtins() -> None:
    register_backend(SplatBackend(
        "splat_transform", "decoder",
        "PlayCanvas splat-transform: compressed PLY/SPZ -> standard 3DGS PLY",
        ("node",), _splat_transform_available, _splat_transform_run,
    ))
    register_backend(SplatBackend(
        "spark", "renderer",
        "Spark.js (three.js) headless reference renderer of the captured scene (local, no GPU)",
        ("node",), _spark_available, _spark_run,
    ))
    register_backend(SplatBackend(
        "threedgrut", "exporter",
        "NVIDIA 3dgrut transcode: standard 3DGS PLY -> NuRec USDZ / ParticleField USD",
        ("threedgrut",), _threedgrut_available, _threedgrut_run,
    ))
    register_backend(SplatBackend(
        "particlefield_usd", "exporter",
        "Author ParticleField3DGaussianSplat USD (Isaac Sim 6.0 RTX-native) from a standard "
        "3DGS PLY in pure pxr -- no ncore/3dgrut/NRE",
        ("pxr",), _particlefield_available, _particlefield_run,
    ))
    register_backend(SplatBackend(
        "isaac_nurec", "renderer",
        "Isaac Sim RTX / NuRec-3DGUT render of the captured scene (GPU worker)",
        ("gpu", "isaacsim"), _isaac_nurec_available, _isaac_nurec_run,
    ))
    register_backend(SplatBackend(
        "artifixer", "enhancer",
        "NVIDIA ArtiFixer: diffusion artifact-fix / novel-view frames from sparse 3DGRUT recon",
        ("gpu", "artifixer", "checkpoint"), _artifixer_available, _artifixer_run,
    ))


_register_builtins()
