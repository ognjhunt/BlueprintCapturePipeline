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
import hashlib
import re
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
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def _artifixer_available() -> bool:
    # nv-tlabs/artifixer exposes the model_eval package when installed/on PYTHONPATH.
    try:
        return importlib.util.find_spec("model_eval") is not None
    except (ImportError, ValueError):
        return False


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _artifixer_run(
    *,
    checkpoint_pt,
    save_dir,
    split_path,
    evalset: str = "reconstructed_colmap",
    render_trajectory: str = "all_frames",
    python: str = "python",
    timeout_seconds: int = 7200,
    held_out_manifest=None,
    model_id: str | None = None,
    checkpoint_digest: str | None = None,
    base_model_digest: str | None = None,
    source_commit_sha: str | None = None,
    container_image_digest: str | None = None,
    license_receipt_digest: str | None = None,
    baseline_reconstruction_digest: str | None = None,
    frozen_split_digest: str | None = None,
) -> dict:
    """NVIDIA ArtiFixer: diffusion artifact-fix / novel-view frames from a sparse 3DGRUT
    reconstruction. Fail-closed wrapper around its documented inference CLI."""
    if not _artifixer_available():
        return {
            "status": "blocked",
            "blockers": ["artifixer_unavailable"],
            "remediation": "install nv-tlabs/artifixer (GPU, CUDA 12/13) and supply a checkpoint",
        }
    from .reconstruction_enhancement_audit import enhancement_method_audit

    qualification_audit = enhancement_method_audit("artifixer")
    pins = {
        "checkpoint_digest": checkpoint_digest,
        "base_model_digest": base_model_digest,
        "license_receipt_digest": license_receipt_digest,
        "baseline_reconstruction_digest": baseline_reconstruction_digest,
        "frozen_split_digest": frozen_split_digest,
    }
    blockers = list(qualification_audit["blockers"])
    blockers.extend(
        [
        f"artifixer_{key}_missing_or_invalid"
        for key, value in pins.items()
        if not _DIGEST.fullmatch(str(value or ""))
        ]
    )
    if _COMMIT.fullmatch(str(source_commit_sha or "")) is None:
        blockers.append("artifixer_source_commit_missing_or_invalid")
    if _IMAGE.fullmatch(str(container_image_digest or "")) is None:
        blockers.append("artifixer_container_image_missing_or_invalid")
    if model_id not in {
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    }:
        blockers.append("artifixer_base_model_identity_missing_or_invalid")
    if not held_out_manifest:
        blockers.append("artifixer_frozen_real_heldout_manifest_required")
    checkpoint = Path(checkpoint_pt)
    split = Path(split_path)
    heldout = Path(held_out_manifest) if held_out_manifest else None
    if checkpoint.is_symlink() or not checkpoint.is_file():
        blockers.append("artifixer_checkpoint_missing_or_symlink")
    elif checkpoint_digest and _sha256_path(checkpoint) != checkpoint_digest:
        blockers.append("artifixer_checkpoint_digest_mismatch")
    if split.is_symlink() or not split.is_file():
        blockers.append("artifixer_split_missing_or_symlink")
    if heldout is None or heldout.is_symlink() or not heldout.is_file():
        blockers.append("artifixer_heldout_manifest_missing_or_symlink")
    if blockers:
        return {
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "enhancer": "artifixer",
            "enhancement_method_audit": qualification_audit,
            "proof_effect": "none",
            "claim_ceiling": "generated_visual_support",
        }
    exe = shutil.which(python) or python
    cmd = [
        exe,
        "-m",
        ARTIFIXER_INFERENCE_MODULE,
        "--evalset",
        evalset,
        "--checkpoint_pt",
        str(checkpoint_pt),
        "--model_id",
        str(model_id),
        "--save_dir",
        str(save_dir),
        "--split_path",
        str(split_path),
        "--render_trajectory",
        render_trajectory,
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
    from .artifixer_heldout_evaluation import evaluate_artifixer_heldout_views

    evaluation = evaluate_artifixer_heldout_views(
        manifest_path=held_out_manifest,
        generated_root=save_dir,
        output_path=Path(save_dir) / "artifixer_heldout_real_view_evaluation.json",
    )
    return {
        "status": (
            "completed_generated_support_advisory"
            if evaluation["status"] == "passed_advisory"
            else "generated_support_failed_heldout_evaluation"
        ),
        "save_dir": str(save_dir),
        "command": " ".join(cmd),
        "enhancer": "artifixer",
        "heldout_evaluation": evaluation,
        "claim_boundary": evaluation["claim_boundary"],
    }


def _rejected_enhancer_run(method_id: str, **_kwargs) -> dict:
    from .reconstruction_enhancement_audit import enhancement_method_audit

    audit = enhancement_method_audit(method_id)
    return {
        "status": "blocked",
        "blockers": list(audit["blockers"]),
        "enhancer": method_id,
        "enhancement_method_audit": audit,
        "proof_effect": "none",
        "claim_ceiling": "generated_visual_support",
    }


def _register_builtins() -> None:
    register_backend(
        SplatBackend(
            "splat_transform",
            "decoder",
            "PlayCanvas splat-transform: compressed PLY/SPZ -> standard 3DGS PLY",
            ("node",),
            _splat_transform_available,
            _splat_transform_run,
        )
    )
    register_backend(
        SplatBackend(
            "spark",
            "renderer",
            "Spark.js (three.js) headless reference renderer of the captured scene (local, no GPU)",
            ("node",),
            _spark_available,
            _spark_run,
        )
    )
    register_backend(
        SplatBackend(
            "threedgrut",
            "exporter",
            "NVIDIA 3dgrut transcode: standard 3DGS PLY -> NuRec USDZ / ParticleField USD",
            ("threedgrut",),
            _threedgrut_available,
            _threedgrut_run,
        )
    )
    register_backend(
        SplatBackend(
            "particlefield_usd",
            "exporter",
            "Author ParticleField3DGaussianSplat USD (Isaac Sim 6.0 RTX-native) from a standard "
            "3DGS PLY in pure pxr -- no ncore/3dgrut/NRE",
            ("pxr",),
            _particlefield_available,
            _particlefield_run,
        )
    )
    register_backend(
        SplatBackend(
            "isaac_nurec",
            "renderer",
            "Isaac Sim RTX / NuRec-3DGUT render of the captured scene (GPU worker)",
            ("gpu", "isaacsim"),
            _isaac_nurec_available,
            _isaac_nurec_run,
        )
    )
    register_backend(
        SplatBackend(
            "artifixer",
            "enhancer",
            "NVIDIA ArtiFixer: diffusion artifact-fix / novel-view frames from sparse 3DGRUT recon",
            ("gpu", "artifixer", "checkpoint"),
            _artifixer_available,
            _artifixer_run,
        )
    )
    register_backend(
        SplatBackend(
            "difix3d",
            "enhancer",
            "NVIDIA Difix3D+: deterministic rejection pending commercial license and runtime qualification",
            ("commercial_license_receipt", "pinned_worker", "checkpoint"),
            lambda: False,
            lambda **kwargs: _rejected_enhancer_run("difix3d", **kwargs),
        )
    )
    register_backend(
        SplatBackend(
            "harmonizer",
            "enhancer",
            "NVIDIA DiffusionHarmonizer: deterministic rejection pending pinned runtime qualification",
            ("pinned_worker", "checkpoint", "cosmos_base_model"),
            lambda: False,
            lambda **kwargs: _rejected_enhancer_run("harmonizer", **kwargs),
        )
    )


_register_builtins()
