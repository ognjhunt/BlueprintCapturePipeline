"""Deterministic render-appearance contracts for policy-visible SimReady assets.

Physics materials control contact and are not render materials.  This module
keeps those claims separate: it can bind a minimal evidence-labeled
UsdPreviewSurface appearance without touching physics bindings, and it rejects
policy-visible geometry that would fall back to a renderer's default gray.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "simready_render_appearance.v1"
AUTHORING_SCHEMA_VERSION = "simready_minimal_render_appearance_authoring.v1"
COLOR_OBSERVATION_SCHEMA_VERSION = "simready_masked_color_observation.v1"
PROVENANCE_ATTRIBUTE = "blueprint:articulatedReplacement:provenance"
GENERATED_PROVENANCE_VALUE = "generated_candidate_geometry"
OBSERVED_APPEARANCE_LABEL = "observed_reference_derived_color_candidate"
GENERATED_APPEARANCE_LABEL = "generated_candidate_unobserved"


class SimReadyRenderAppearanceError(ValueError):
    """Stable, aggregate render-appearance errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> str | None:
    text = str(value or "")
    if (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    ):
        return text
    return None


def _unit_interval(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _color(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    result = tuple(_unit_interval(item) for item in value)
    if any(item is None for item in result):
        return None
    return tuple(float(item) for item in result)  # type: ignore[arg-type]


def _surface_shader_ids(material: Any) -> list[str]:
    try:
        connected, _invalid = material.GetSurfaceOutput().GetConnectedSources()
    except Exception:  # pragma: no cover - pxr version guard
        return []
    shader_ids: list[str] = []
    for connection in connected:
        source = connection.source.GetPrim()
        identifier = source.GetAttribute("info:id").Get() if source.IsValid() else None
        if identifier:
            shader_ids.append(str(identifier))
    return sorted(set(shader_ids))


def inspect_simready_render_appearance(asset_path: str | Path) -> dict[str, Any]:
    """Require render-purpose geometry to resolve to an authored surface shader."""

    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_openusd_runtime_missing"]
        ) from exc

    path = Path(asset_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise SimReadyRenderAppearanceError(["simready_render_appearance_asset_missing"])
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_asset_unreadable"]
        )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = str(imageable.ComputePurpose())
        visibility = str(imageable.ComputeVisibility())
        if purpose in {"guide", "proxy"} or visibility == "invisible":
            continue
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        material_path = str(material.GetPath()) if material else None
        shader_ids = _surface_shader_ids(material) if material else []
        prim_path = str(prim.GetPath())
        provenance_attr = prim.GetAttribute(PROVENANCE_ATTRIBUTE)
        provenance = (
            str(provenance_attr.Get())
            if provenance_attr and provenance_attr.HasAuthoredValue()
            else None
        )
        if not material_path:
            errors.append(f"simready_render_material_unbound:{prim_path}")
        elif not shader_ids:
            errors.append(f"simready_render_surface_shader_missing:{prim_path}")
        rows.append(
            {
                "prim_path": prim_path,
                "purpose": purpose,
                "provenance": provenance,
                "material_path": material_path,
                "surface_shader_ids": shader_ids,
            }
        )
    if not rows:
        errors.append("simready_policy_visible_geometry_missing")
    if errors:
        raise SimReadyRenderAppearanceError(errors)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "policy_visible_render_materials_statically_admitted",
        "asset_path": str(path),
        "asset_sha256": _sha256(path),
        "policy_visible_gprim_count": len(rows),
        "material_rows": rows,
        "default_neutral_renderer_fallback_required": False,
        "coverage_silhouette_audit_satisfies_policy_render_gate": False,
        "native_renderer_honored_materials_observed": False,
        "claim_boundary": {
            "static_shader_binding_is_not_native_render_readback": True,
            "material_binding_is_not_observed_site_texture_truth": True,
            "generated_interior_is_observed_site_truth": False,
            "policy_evaluation_authorized_by_this_receipt": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _validated_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        spec = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_spec_invalid"]
        ) from exc
    errors: list[str] = []
    if _digest(spec.get("observed_exterior_reference_sha256")) is None:
        errors.append("simready_observed_exterior_reference_digest_missing")
    observation = spec.get("observed_exterior_color_observation")
    if not isinstance(observation, Mapping):
        errors.append("simready_observed_exterior_color_observation_missing")
    else:
        if observation.get("schema_version") != COLOR_OBSERVATION_SCHEMA_VERSION:
            errors.append("simready_observed_exterior_color_observation_invalid")
        if observation.get("image_sha256") != spec.get(
            "observed_exterior_reference_sha256"
        ):
            errors.append("simready_observed_exterior_reference_mismatch")
        if observation.get("median_srgb") != spec.get(
            "observed_exterior_base_color_rgb"
        ):
            errors.append("simready_observed_exterior_color_not_derived")
        if observation.get("receipt_digest") != canonical_digest(
            dict(observation), digest_field="receipt_digest"
        ):
            errors.append("simready_observed_exterior_color_observation_digest_mismatch")
    for prefix in ("observed_exterior", "generated_surface"):
        if _color(spec.get(f"{prefix}_base_color_rgb")) is None:
            errors.append(f"simready_{prefix}_base_color_invalid")
        if _unit_interval(spec.get(f"{prefix}_roughness")) is None:
            errors.append(f"simready_{prefix}_roughness_invalid")
        if _unit_interval(spec.get(f"{prefix}_metallic")) is None:
            errors.append(f"simready_{prefix}_metallic_invalid")
    if spec.get("generated_surface_claim") != GENERATED_APPEARANCE_LABEL:
        errors.append("simready_generated_surface_claim_invalid")
    if errors:
        raise SimReadyRenderAppearanceError(errors)
    return spec


def derive_masked_observed_color(
    *, image_path: str | Path, mask_path: str | Path
) -> dict[str, Any]:
    """Measure median sRGB only from exact, digest-bound observed mask pixels."""

    import numpy as np
    from PIL import Image

    image_file = Path(image_path).expanduser().resolve()
    mask_file = Path(mask_path).expanduser().resolve()
    if any(not path.is_file() or path.is_symlink() for path in (image_file, mask_file)):
        raise SimReadyRenderAppearanceError(
            ["simready_observed_exterior_color_input_missing"]
        )
    try:
        image = np.asarray(Image.open(image_file).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_file).convert("L"), dtype=np.uint8) > 127
    except (OSError, ValueError) as exc:
        raise SimReadyRenderAppearanceError(
            ["simready_observed_exterior_color_input_unreadable"]
        ) from exc
    if image.shape[:2] != mask.shape or not bool(mask.any()):
        raise SimReadyRenderAppearanceError(
            ["simready_observed_exterior_color_mask_invalid"]
        )
    median = np.median(image[mask], axis=0) / 255.0
    receipt: dict[str, Any] = {
        "schema_version": COLOR_OBSERVATION_SCHEMA_VERSION,
        "status": "masked_observed_color_measured",
        "image_sha256": _sha256(image_file),
        "mask_sha256": _sha256(mask_file),
        "mask_threshold_8bit": 127,
        "masked_pixel_count": int(mask.sum()),
        "estimator": "per_channel_median_srgb_v1",
        "median_srgb": [float(value) for value in median],
        "claim_boundary": {
            "masked_color_is_not_texture_equivalence": True,
            "appearance_is_lighting_dependent": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def bind_minimal_render_materials_to_stage(
    *, stage: Any, appearance_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind deterministic PreviewSurface materials without changing physics purpose."""

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    spec = _validated_spec(appearance_spec)
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise SimReadyRenderAppearanceError(["simready_render_default_prim_missing"])
    root = str(default_prim.GetPath())

    def _material(name: str, prefix: str, label: str) -> Any:
        material = UsdShade.Material.Define(stage, f"{root}/render_materials/{name}")
        shader = UsdShade.Shader.Define(
            stage, f"{root}/render_materials/{name}_shader"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*_color(spec[f"{prefix}_base_color_rgb"]))
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(spec[f"{prefix}_roughness"])
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(spec[f"{prefix}_metallic"])
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        material.GetPrim().CreateAttribute(
            "blueprint:appearance:provenance", Sdf.ValueTypeNames.String
        ).Set(label)
        return material

    observed = _material(
        "observed_exterior", "observed_exterior", OBSERVED_APPEARANCE_LABEL
    )
    generated = _material(
        "generated_unobserved", "generated_surface", GENERATED_APPEARANCE_LABEL
    )
    UsdShade.MaterialBindingAPI.Apply(default_prim).Bind(observed)
    generated_paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        provenance = prim.GetAttribute(PROVENANCE_ATTRIBUTE)
        if (
            provenance
            and provenance.HasAuthoredValue()
            and provenance.Get() == GENERATED_PROVENANCE_VALUE
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(generated)
            generated_paths.append(str(prim.GetPath()))
    return {
        "observed_exterior_material_path": str(observed.GetPath()),
        "texture_agent_target_material_path": str(observed.GetPath()),
        "generated_unobserved_material_path": str(generated.GetPath()),
        "generated_surface_prim_paths": sorted(generated_paths),
        "observed_exterior_reference_sha256": spec[
            "observed_exterior_reference_sha256"
        ],
        "generated_surface_claim": GENERATED_APPEARANCE_LABEL,
    }


def author_minimal_render_appearance(
    *,
    source_asset_path: str | Path,
    output_asset_path: str | Path,
    appearance_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic materialized derivative while preserving source bytes."""

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_openusd_runtime_missing"]
        ) from exc
    source = Path(source_asset_path).expanduser().resolve()
    output = Path(output_asset_path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise SimReadyRenderAppearanceError(["simready_render_appearance_asset_missing"])
    if source == output:
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_in_place_mutation_forbidden"]
        )
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_asset_unreadable"]
        )
    layer = source_stage.Flatten()
    layer.documentation = ""
    layer.comment = ""
    stage = Usd.Stage.Open(layer)
    binding = bind_minimal_render_materials_to_stage(
        stage=stage, appearance_spec=appearance_spec
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise SimReadyRenderAppearanceError(
            ["simready_render_appearance_export_failed"]
        )
    validation = inspect_simready_render_appearance(output)
    receipt: dict[str, Any] = {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "status": "minimal_render_appearance_authored",
        "source_asset_path": str(source),
        "source_asset_sha256": _sha256(source),
        "output_asset_path": str(output),
        "output_asset_sha256": _sha256(output),
        "binding": binding,
        "static_validation": validation,
        "native_material_render_readback_required": True,
        "texture_enrichment_optional": True,
        "claim_boundary": {
            "observed_reference_color_is_not_texture_equivalence": True,
            "generated_interior_is_labeled_unobserved": True,
            "physical_equivalence_proven": False,
            "policy_evaluation_authorized_by_this_receipt": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "COLOR_OBSERVATION_SCHEMA_VERSION",
    "GENERATED_APPEARANCE_LABEL",
    "OBSERVED_APPEARANCE_LABEL",
    "SCHEMA_VERSION",
    "SimReadyRenderAppearanceError",
    "author_minimal_render_appearance",
    "bind_minimal_render_materials_to_stage",
    "derive_masked_observed_color",
    "inspect_simready_render_appearance",
]
