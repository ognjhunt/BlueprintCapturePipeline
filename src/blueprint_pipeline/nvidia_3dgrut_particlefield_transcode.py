"""Thin, digest-bound adapter for NVIDIA's direct NuRec to LightField transcode."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_field_quality import (
    gaussian_quality_is_qualified,
    measure_gaussian_field_quality,
)
from .nurec_volume_codec import describe_volume
from .particlefield_usd import _nurec_payload_and_transform


AUTHORING_IMPLEMENTATION = "nvidia_3dgrut_direct_nurec_transcode"
RECEIPT_SCHEMA_VERSION = "nvidia_3dgrut_particlefield_transcode.v1"
UPSTREAM_REPOSITORY = "https://github.com/nv-tlabs/3dgrut.git"
UPSTREAM_SOURCE_REVISION = "a37ef721012dea0f29c0fcfff2d525023b4e854a"
UPSTREAM_MODULE = "threedgrut.export.scripts.transcode"
OUTPUT_FORMAT = "lightfield"
PROJECTION_MODE_HINT = "perspective"
SORTING_MODE_HINT = "cameraDistance"
COLOR_SPACE = "srgb_rec709_display"


class Nvidia3DGrutTranscodeError(ValueError):
    """Stable direct-transcode refusal."""


def _identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _git_output(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _load_pinned_transcode(
    *, source_root: str | Path | None
) -> tuple[Callable[..., Any], dict[str, Any]]:
    root_text = str(
        source_root
        or os.environ.get("BLUEPRINT_NVIDIA_3DGRUT_SOURCE_ROOT")
        or ""
    ).strip()
    if not root_text:
        raise Nvidia3DGrutTranscodeError("nvidia_3dgrut_source_root_missing")
    root = Path(root_text).expanduser().resolve()
    if (
        not root.is_dir()
        or _git_output(root, "rev-parse", "HEAD") != UPSTREAM_SOURCE_REVISION
        or _git_output(root, "status", "--porcelain")
    ):
        raise Nvidia3DGrutTranscodeError("nvidia_3dgrut_source_identity_invalid")
    from threedgrut.export.scripts import transcode as transcode_module

    module_path = Path(transcode_module.__file__).resolve()
    if not module_path.is_relative_to(root):
        raise Nvidia3DGrutTranscodeError("nvidia_3dgrut_import_root_mismatch")
    return transcode_module.transcode, {
        "repository": UPSTREAM_REPOSITORY,
        "source_revision": UPSTREAM_SOURCE_REVISION,
        "module": UPSTREAM_MODULE,
        "module_sha256": "sha256:" + sha256_file(module_path),
        "source_identity_verified": True,
    }


def validate_direct_particlefield(path: str | Path) -> dict[str, Any]:
    """Validate the exact LightField contract authored by pinned 3DGRUT."""

    from pxr import Usd, UsdGeom

    asset = Path(path).expanduser().resolve()
    stage = Usd.Stage.Open(str(asset)) if asset.is_file() else None
    fields = (
        [
            prim
            for prim in stage.Traverse()
            if prim.GetTypeName() == "ParticleField3DGaussianSplat"
        ]
        if stage and stage.GetDefaultPrim()
        else []
    )
    nurec = (
        [
            prim
            for prim in stage.Traverse()
            if bool(prim.GetAttribute("omni:nurec:isNuRecVolume").Get())
        ]
        if stage
        else []
    )
    if len(fields) != 1 or nurec:
        raise Nvidia3DGrutTranscodeError(
            "nvidia_3dgrut_particlefield_stage_invalid"
        )
    field = fields[0]
    positions = field.GetAttribute("positions").Get()
    scales = field.GetAttribute("scales").Get()
    opacities = field.GetAttribute("opacities").Get()
    orientations = field.GetAttribute("orientations").Get()
    coefficients = field.GetAttribute(
        "radiance:sphericalHarmonicsCoefficients"
    ).Get()
    degree = field.GetAttribute("radiance:sphericalHarmonicsDegree").Get()
    projection = field.GetAttribute("projectionModeHint")
    sorting = field.GetAttribute("sortingModeHint")
    color_space = field.GetAttribute("colorSpace:name")
    display_color = field.GetAttribute("primvars:displayColor")
    count = len(positions or [])
    if (
        count < 1
        or len(scales or []) != count
        or len(opacities or []) != count
        or len(orientations or []) != count
        or degree != 3
        or len(coefficients or []) != count * 16
        or field.GetRelationship("material:binding").GetTargets()
        or not projection.HasAuthoredValueOpinion()
        or projection.Get() != PROJECTION_MODE_HINT
        or not sorting.HasAuthoredValueOpinion()
        or sorting.Get() != SORTING_MODE_HINT
        or not color_space.HasAuthoredValueOpinion()
        or color_space.Get() != COLOR_SPACE
        or display_color.HasAuthoredValueOpinion()
    ):
        raise Nvidia3DGrutTranscodeError(
            "nvidia_3dgrut_particlefield_contract_invalid"
        )
    field_quality = measure_gaussian_field_quality(
        positions=positions,
        activated_scales=scales,
        opacities=opacities,
    )
    if not gaussian_quality_is_qualified(field_quality):
        raise Nvidia3DGrutTranscodeError(
            "nvidia_3dgrut_particlefield_quality_invalid"
        )
    digest, size = _identity(asset)
    return {
        "output_sha256": digest,
        "output_bytes": size,
        "schema": "ParticleField3DGaussianSplat",
        "prim_path": str(field.GetPath()),
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "stage_up_axis": stage.GetMetadata("upAxis"),
        "stage_meters_per_unit": stage.GetMetadata("metersPerUnit"),
        "splat_count": count,
        "sh_degree": degree,
        "sh_coefficient_count": len(coefficients),
        "sh_primvar_element_size": 16,
        "sh_primvar_interpolation": str(
            UsdGeom.Primvar(
                field.GetAttribute("radiance:sphericalHarmonicsCoefficients")
            ).GetInterpolation()
        ),
        "particlefield_emissive_material_binding_authored": False,
        "particlefield_custom_render_hints_authored": False,
        "upstream_projection_mode_hint": projection.Get(),
        "upstream_sorting_mode_hint": sorting.Get(),
        "upstream_color_space": color_space.Get(),
        "display_color_fallback_authored": False,
        "gaussian_field_quality": field_quality,
    }


def write_direct_particlefield_from_nurec(
    source_path: str | Path,
    output_path: str | Path,
    *,
    expected_source_sha256: str,
    receipt_path: str | Path | None = None,
    source_root: str | Path | None = None,
    transcode_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run NVIDIA's direct NuRec-to-LightField path without tensor remapping."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    observed_source_sha256 = (
        "sha256:" + sha256_file(source) if source.is_file() else None
    )
    if (
        source.is_symlink()
        or not source.is_file()
        or observed_source_sha256 != expected_source_sha256
        or output.exists()
        or output.is_symlink()
    ):
        return {
            "status": "blocked",
            "blockers": ["nvidia_3dgrut_transcode_input_or_output_invalid"],
            "source_sha256": observed_source_sha256,
            "provider_mutation_performed": False,
        }
    try:
        document, transform, source_prim_path, payload = (
            _nurec_payload_and_transform(source)
        )
        description = describe_volume(document)
        if transcode_runner is None:
            transcode_runner, upstream = _load_pinned_transcode(
                source_root=source_root
            )
        else:
            upstream = {
                "repository": UPSTREAM_REPOSITORY,
                "source_revision": UPSTREAM_SOURCE_REVISION,
                "module": UPSTREAM_MODULE,
                "module_sha256": None,
                "source_identity_verified": False,
                "test_injected_runner": True,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="nvidia-3dgrut-transcode-", dir=output.parent
        ) as temporary_directory:
            candidate = Path(temporary_directory) / "scene.lightfield.usdc"
            transcode_runner(
                source,
                candidate,
                output_format=OUTPUT_FORMAT,
                render_order_hint=SORTING_MODE_HINT,
                validate_usd=True,
            )
            contract = validate_direct_particlefield(candidate)
            if contract["splat_count"] != description["gaussian_count"]:
                raise Nvidia3DGrutTranscodeError(
                    "nvidia_3dgrut_particlefield_count_mismatch"
                )
            os.replace(candidate, output)
        contract = validate_direct_particlefield(output)
    except Exception as exc:  # noqa: BLE001 - typed authoring refusal
        return {
            "status": "blocked",
            "blockers": [
                f"nvidia_3dgrut_direct_transcode_failed:{type(exc).__name__}:{exc}"
            ],
            "source_sha256": observed_source_sha256,
            "provider_mutation_performed": False,
        }

    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": observed_source_sha256,
        "source_kind": "nurec_usdz",
        "source_nurec_payload_sha256": "sha256:"
        + hashlib.sha256(payload).hexdigest(),
        "source_nurec_prim_path": source_prim_path,
        "source_nurec_description": description,
        "source_local_to_world_transform": transform,
        "output": str(output),
        **contract,
        "particlefield_authoring_implementation": AUTHORING_IMPLEMENTATION,
        "upstream_converter": upstream,
        "exact_learned_arrays_preserved": True,
        "representation_conversion_only": True,
        "provider_mutation_performed": False,
        "proof_boundary": (
            "NVIDIA 3DGRUT direct NuRec-to-LightField transcode and structural "
            "validation only; live Isaac camera fidelity remains required."
        ),
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    if receipt_path is not None:
        write_json(Path(receipt_path), result)
    return result


__all__ = [
    "AUTHORING_IMPLEMENTATION",
    "COLOR_SPACE",
    "Nvidia3DGrutTranscodeError",
    "PROJECTION_MODE_HINT",
    "RECEIPT_SCHEMA_VERSION",
    "SORTING_MODE_HINT",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_SOURCE_REVISION",
    "validate_direct_particlefield",
    "write_direct_particlefield_from_nurec",
]
