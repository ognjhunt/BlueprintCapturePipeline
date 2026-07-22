"""Fail-closed 3DGS PLY to Isaac NuRec/ParticleField export adapter."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


TRANSCODE_MODULE = "threedgrut.export.scripts.transcode"
ISAAC_FORMATS = {"nurec", "lightfield"}


def threedgrut_available(python: str = "python") -> bool:
    """Return whether ``threedgrut.export`` is importable by the interpreter."""
    executable = shutil.which(python) or python
    try:
        process = subprocess.run(
            [executable, "-c", "import threedgrut.export.scripts.transcode"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0


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
    """Transcode standard 3DGS PLY to NuRec USDZ or ParticleField USD."""
    ply_path = Path(ply)
    output_path = Path(out)
    if fmt not in ISAAC_FORMATS:
        return {
            "status": "blocked",
            "blockers": ["unsupported_isaac_splat_format"],
            "format": fmt,
        }
    if not ply_path.is_file():
        return {
            "status": "blocked",
            "blockers": ["standard_ply_missing"],
            "input": str(ply_path),
        }
    if check_available and not threedgrut_available(python):
        return {
            "status": "blocked",
            "blockers": ["threedgrut_unavailable"],
            "remediation": (
                "install nv-tlabs/3dgrut on the GPU worker; this transcode "
                "runs only where threedgrut is importable"
            ),
            "input": str(ply_path),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(python) or python
    command = [
        executable,
        "-m",
        TRANSCODE_MODULE,
        str(ply_path),
        "-o",
        str(output_path),
        "--format",
        fmt,
    ]
    if half:
        command.append("--half")
    if max_sh_degree is not None:
        command += ["--max-sh-degree", str(int(max_sh_degree))]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {
            "status": "blocked",
            "blockers": ["python_runtime_unavailable"],
            "input": str(ply_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "blocked",
            "blockers": ["threedgrut_transcode_timeout"],
            "input": str(ply_path),
        }
    if process.returncode != 0 or not output_path.is_file():
        return {
            "status": "blocked",
            "blockers": ["threedgrut_transcode_failed"],
            "returncode": process.returncode,
            "stderr_tail": (process.stderr or "")[-2000:],
            "command": " ".join(command),
            "input": str(ply_path),
        }
    return {
        "status": "completed",
        "input": str(ply_path),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "format": fmt,
        "schema": "nurec_usdz" if fmt == "nurec" else "particlefield_usd",
        "converter": "threedgrut_transcode",
        "command": " ".join(command),
    }
