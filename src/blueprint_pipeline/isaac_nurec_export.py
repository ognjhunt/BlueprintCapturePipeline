"""Fail-closed 3DGS PLY to Isaac NuRec/ParticleField export adapter."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


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


# ---------------------------------------------------------------------------
# Direct NuRec USDZ -> LightField ParticleField transcode (Scene 839873 audit)
# ---------------------------------------------------------------------------
#
# NVIDIA 3DGRUT documents a direct ``nurec.usd -> lightfield.usdz`` transcode
# whose importer owns the private ``.nurec`` tensor semantics
# (threedgrut/export/importers/nurec_usd.py) and whose LightField writer owns
# activation, SH layout, hints and colour-space metadata
# (threedgrut/export/usd/writers/lightfield.py).  This wrapper invokes that
# route in a pinned source revision, never decodes ``.nurec`` state itself,
# and validates the output before a receipt is sealed.

THREEDGRUT_PINNED_REVISION = "a37ef721012dea0f29c0fcfff2d525023b4e854a"
THREEDGRUT_TRANSCODE_SOURCE_URL = (
    "https://github.com/nv-tlabs/3dgrut/blob/"
    f"{THREEDGRUT_PINNED_REVISION}/threedgrut/export/scripts/transcode.py"
)
NUREC_TRANSCODE_RECEIPT_SCHEMA_VERSION = "nurec_particlefield_direct_transcode_receipt.v1"
#: Environment variables the transcode subprocess may inherit.  Credentials
#: (NGC, cloud, provider tokens) are never forwarded and never retained.
TRANSCODE_FORWARDED_ENVIRONMENT = ("PATH", "HOME", "PYTHONPATH", "LANG", "LC_ALL", "TMPDIR")
PARTICLEFIELD_SORTING_MODE_HINTS = frozenset({"zDepth", "cameraDistance", "rayHitDistance"})
PARTICLEFIELD_DISPLAY_COLOR_SPACE = "srgb_rec709_display"
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[=:]\s*\S+"
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _scrub(text: str | None, *, limit: int = 4000) -> str:
    return _SECRET_PATTERN.sub(r"\1=<redacted>", (text or "")[-limit:])


def build_nurec_transcode_command(
    source_usdz: str | Path,
    out_path: str | Path,
    *,
    python: str = "python",
    render_order_hint: str | None = None,
) -> list[str]:
    """The exact pinned command; digest-free, credential-free, no reinterpretation.

    ``--apply-coordinate-transform`` is deliberately absent: the NuRec importer
    already applies the Volume's authored local-to-world transform, and the
    3DGRUT-to-USDZ axis flip is for fields trained in 3DGRUT's own frame, not
    for a NuRec asset that already renders in Omniverse space.
    """

    command = [
        shutil.which(python) or python,
        "-m",
        TRANSCODE_MODULE,
        str(source_usdz),
        "-o",
        str(out_path),
        "--format",
        "lightfield",
    ]
    if render_order_hint is not None:
        if render_order_hint not in PARTICLEFIELD_SORTING_MODE_HINTS:
            raise ValueError("nurec_transcode_render_order_hint_invalid")
        command += ["--render-order-hint", render_order_hint]
    return command


def validate_transcoded_particlefield(
    path: str | Path,
    *,
    expected_gaussian_count: int | None = None,
    expected_sh_degree: int = 3,
) -> dict[str, Any]:
    """Validate one LightField ParticleField and digest every learned attribute.

    Checks the schema, counts, SH degree and element size, the renderer hints
    and colour space 3DGRUT authors, the absence of a material binding, and the
    composed world transform.  Returns per-attribute sha256 digests of the
    float32 arrays so two conversions can be compared without a renderer.
    """

    path = Path(path)
    blockers: list[str] = []
    try:
        import numpy as np
        from pxr import Usd, UsdGeom
    except Exception:  # noqa: BLE001
        return {
            "passed": False,
            "blockers": ["nurec_transcode_validation_runtime_unavailable"],
        }
    stage = Usd.Stage.Open(str(path)) if path.is_file() else None
    if not stage:
        return {"passed": False, "blockers": ["nurec_transcode_output_unreadable"]}
    prims = [p for p in stage.Traverse() if p.GetTypeName() == "ParticleField3DGaussianSplat"]
    if len(prims) != 1:
        return {
            "passed": False,
            "blockers": ["nurec_transcode_particlefield_prim_not_exact"],
            "particlefield_prim_count": len(prims),
        }
    prim = prims[0]

    def _array(name: str) -> np.ndarray | None:
        attr = prim.GetAttribute(name)
        value = attr.Get() if attr and attr.HasAuthoredValueOpinion() else None
        if value is None:
            return None
        if name == "orientations":
            return np.array(
                [[q.GetReal(), *q.GetImaginary()] for q in value], dtype=np.float32
            )
        return np.asarray(value, dtype=np.float32)

    arrays = {name: _array(name) for name in ("positions", "scales", "orientations", "opacities")}
    sh_attr = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
    sh = _array("radiance:sphericalHarmonicsCoefficients")
    degree_attr = prim.GetAttribute("radiance:sphericalHarmonicsDegree")
    degree = degree_attr.Get() if degree_attr else None
    element_size = sh_attr.GetMetadata("elementSize") if sh_attr else None
    count = int(arrays["positions"].shape[0]) if arrays["positions"] is not None else 0
    if any(value is None for value in arrays.values()) or sh is None:
        blockers.append("nurec_transcode_particlefield_attribute_missing")
    else:
        if count < 1 or any(int(value.shape[0]) != count for value in arrays.values()):
            blockers.append("nurec_transcode_particlefield_count_disagreement")
        if expected_gaussian_count is not None and count != int(expected_gaussian_count):
            blockers.append("nurec_transcode_particlefield_count_unexpected")
        expected_elements = (int(expected_sh_degree) + 1) ** 2
        if degree != int(expected_sh_degree) or element_size != expected_elements:
            blockers.append("nurec_transcode_sh_degree_unexpected")
        if int(sh.shape[0]) != count * expected_elements:
            blockers.append("nurec_transcode_sh_coefficient_count_unexpected")
        if not all(np.isfinite(value).all() for value in arrays.values()) or not np.isfinite(sh).all():
            blockers.append("nurec_transcode_particlefield_nonfinite")
        if (arrays["opacities"] < 0).any() or (arrays["opacities"] > 1).any():
            blockers.append("nurec_transcode_opacity_out_of_range")
        if (arrays["scales"] <= 0).any():
            blockers.append("nurec_transcode_scale_not_positive")
    projection = prim.GetAttribute("projectionModeHint")
    sorting = prim.GetAttribute("sortingModeHint")
    projection_hint = projection.Get() if projection and projection.HasAuthoredValueOpinion() else None
    sorting_hint = sorting.Get() if sorting and sorting.HasAuthoredValueOpinion() else None
    if projection_hint != "perspective":
        blockers.append("nurec_transcode_projection_hint_unexpected")
    if sorting_hint not in PARTICLEFIELD_SORTING_MODE_HINTS:
        blockers.append("nurec_transcode_sorting_hint_unexpected")
    color_space_attr = prim.GetAttribute("colorSpace:name")
    color_space = color_space_attr.Get() if color_space_attr else None
    if color_space != PARTICLEFIELD_DISPLAY_COLOR_SPACE:
        blockers.append("nurec_transcode_color_space_unexpected")
    binding = prim.GetRelationship("material:binding")
    if binding and binding.GetTargets():
        blockers.append("nurec_transcode_material_binding_unexpected")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world = [[float(matrix[row][column]) for column in range(4)] for row in range(4)]
    digests = {
        name: "sha256:" + hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()
        for name, value in {**arrays, "sh_coefficients": sh}.items()
        if value is not None
    }
    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "prim_path": str(prim.GetPath()),
        "gaussian_count": count,
        "sh_degree": degree,
        "sh_element_size": element_size,
        "projection_mode_hint": projection_hint,
        "sorting_mode_hint": sorting_hint,
        "color_space": color_space,
        "material_binding_targets": [str(target) for target in binding.GetTargets()] if binding else [],
        "world_transform_row_major": world,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "render_settings": dict(stage.GetRootLayer().customLayerData.get("renderSettings") or {}),
        "attribute_digests": digests,
    }


def transcode_nurec_usdz_to_particlefield(
    source_usdz: str | Path,
    out_path: str | Path,
    *,
    expected_source_sha256: str,
    expected_gaussian_count: int | None = None,
    expected_sh_degree: int = 3,
    threedgrut_revision: str = THREEDGRUT_PINNED_REVISION,
    container_image_digest: str | None = None,
    python: str = "python",
    timeout_seconds: int = 3600,
    render_order_hint: str | None = None,
    runner: Callable[..., Any] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run NVIDIA's direct NuRec -> LightField transcode and seal a receipt.

    Fails closed before any subprocess when the source identity does not
    match; forwards only a fixed environment whitelist; retains scrubbed
    stdout/stderr tails; validates the output; and never reads ``.nurec``
    state tensors in Blueprint code.
    """

    source = Path(source_usdz)
    out = Path(out_path)
    observed = _sha256_path(source) if source.is_file() and not source.is_symlink() else None
    base: dict[str, Any] = {
        "schema_version": NUREC_TRANSCODE_RECEIPT_SCHEMA_VERSION,
        "converter": "threedgrut_transcode_direct_nurec_to_lightfield",
        "threedgrut_revision": threedgrut_revision,
        "threedgrut_source_url": THREEDGRUT_TRANSCODE_SOURCE_URL,
        "container_image_digest": container_image_digest,
        "expected_source_sha256": expected_source_sha256,
        "observed_source_sha256": observed,
        "nurec_state_reinterpreted_by_blueprint": False,
    }
    if observed is None or observed != expected_source_sha256:
        return {**base, "status": "blocked", "blockers": ["nurec_transcode_source_identity_mismatch"]}
    if not re.fullmatch(r"[0-9a-f]{40}", str(threedgrut_revision or "")):
        return {**base, "status": "blocked", "blockers": ["nurec_transcode_revision_unpinned"]}
    try:
        command = build_nurec_transcode_command(
            source, out, python=python, render_order_hint=render_order_hint
        )
    except ValueError as exc:
        return {**base, "status": "blocked", "blockers": [str(exc)]}
    environment = {
        key: os.environ[key] for key in TRANSCODE_FORWARDED_ENVIRONMENT if key in os.environ
    }
    base.update(
        command=list(command),
        environment_keys_forwarded=sorted(environment),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    run = runner or (
        lambda cmd, *, env, timeout: subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    )
    try:
        process = run(command, env=environment, timeout=timeout_seconds)
    except FileNotFoundError:
        return {**base, "status": "blocked", "blockers": ["python_runtime_unavailable"]}
    except subprocess.TimeoutExpired:
        return {**base, "status": "blocked", "blockers": ["threedgrut_transcode_timeout"]}
    base.update(
        returncode=int(getattr(process, "returncode", -1)),
        stdout_tail=_scrub(getattr(process, "stdout", "")),
        stderr_tail=_scrub(getattr(process, "stderr", "")),
    )
    if base["returncode"] != 0 or not out.is_file():
        return {**base, "status": "blocked", "blockers": ["threedgrut_transcode_failed"]}
    if _sha256_path(source) != expected_source_sha256:
        return {**base, "status": "blocked", "blockers": ["nurec_transcode_sealed_source_mutated"]}
    validation = validate_transcoded_particlefield(
        out,
        expected_gaussian_count=expected_gaussian_count,
        expected_sh_degree=expected_sh_degree,
    )
    base["output_validation"] = validation
    if not validation.get("passed"):
        return {**base, "status": "blocked", "blockers": list(validation.get("blockers") or [])}
    result = {
        **base,
        "status": "completed",
        "output": str(out),
        "output_bytes": out.stat().st_size,
        "output_sha256": _sha256_path(out),
        "schema": "particlefield_usd",
        "format": "lightfield",
        "representation_conversion_only": True,
    }
    from .decision_evidence_contracts import canonical_digest

    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if receipt_path is not None:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
