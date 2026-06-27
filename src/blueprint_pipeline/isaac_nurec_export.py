"""Convert a standard 3DGS PLY into an Isaac-renderable NuRec/ParticleField USD.

Isaac Sim 5.0+/6.0 renders 3D Gaussian splats via the RTX **NuRec / 3DGUT** path. An
existing trained splat (our standard INRIA 3DGS PLY) is made Isaac-renderable with
NVIDIA's ``3dgrut`` standalone transcoder (no retraining):

    python -m threedgrut.export.scripts.transcode <model.ply> -o <out.usdz> --format nurec
    python -m threedgrut.export.scripts.transcode <model.ply> -o <out.usd>  --format lightfield

(``nurec`` -> NuRec USDZ, broadly supported on 5.0+; ``lightfield`` -> ParticleField USD,
the schema preferred on Isaac Sim 6.0+). The produced asset loads via
``add_reference_to_stage`` / ``omni.usd open_stage`` and renders with the RTX renderer.

This wrapper is location-agnostic: it runs wherever ``threedgrut`` is importable (the
GPU worker image, or a local install). It is fail-closed — a clean status dict, never a
fabricated success. It performs the *conversion* only; it claims no rendering.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TRANSCODE_MODULE = "threedgrut.export.scripts.transcode"
PLY_TO_USD_MODULE = "threedgrut.export.scripts.ply_to_usd"
ISAAC_FORMATS = {"nurec", "lightfield"}  # nurec=USDZ, lightfield=ParticleField USD


def threedgrut_available(python: str = "python") -> bool:
    """True if ``threedgrut.export`` is importable by the given interpreter."""
    exe = shutil.which(python) or python
    try:
        proc = subprocess.run(
            [exe, "-c", "import threedgrut.export.scripts.transcode"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def convert_ply_to_isaac_usd(
    ply: str | Path,
    out: str | Path,
    *,
    fmt: str = "nurec",
    half: bool = False,
    max_sh_degree: int | None = None,
    python: str = "python",
    timeout_seconds: int = 1800,
    check_available: bool = True,
) -> dict:
    """Transcode a standard 3DGS PLY -> Isaac NuRec USDZ (``fmt='nurec'``) or
    ParticleField USD (``fmt='lightfield'``). Returns a fail-closed status dict;
    ``status == 'completed'`` means ``out`` exists."""
    ply = Path(ply)
    out = Path(out)
    if fmt not in ISAAC_FORMATS:
        return {"status": "blocked", "blockers": ["unsupported_isaac_splat_format"], "format": fmt}
    if not ply.is_file():
        return {"status": "blocked", "blockers": ["standard_ply_missing"], "input": str(ply)}
    if check_available and not threedgrut_available(python):
        return {
            "status": "blocked",
            "blockers": ["threedgrut_unavailable"],
            "remediation": "install nv-tlabs/3dgrut (GPU worker) — pip install or clone; "
            "this transcode runs where threedgrut is importable",
            "input": str(ply),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which(python) or python
    cmd = [exe, "-m", TRANSCODE_MODULE, str(ply), "-o", str(out), "--format", fmt]
    if half:
        cmd.append("--half")
    if max_sh_degree is not None:
        cmd += ["--max-sh-degree", str(int(max_sh_degree))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError:
        return {"status": "blocked", "blockers": ["python_runtime_unavailable"], "input": str(ply)}
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "blockers": ["threedgrut_transcode_timeout"], "input": str(ply)}
    if proc.returncode != 0 or not out.is_file():
        return {
            "status": "blocked",
            "blockers": ["threedgrut_transcode_failed"],
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-2000:],
            "command": " ".join(cmd),
            "input": str(ply),
        }
    return {
        "status": "completed",
        "input": str(ply),
        "output": str(out),
        "output_bytes": out.stat().st_size,
        "format": fmt,
        "schema": "nurec_usdz" if fmt == "nurec" else "particlefield_usd",
        "converter": "threedgrut_transcode",
        "command": " ".join(cmd),
    }
