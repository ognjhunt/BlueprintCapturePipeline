"""Give an articulated candidate a render material the texture stage can paint.

Blueprint authors physics: masses, inertias, colliders, joints, and physics
materials. None of that is visible. The 840796 Content Agents pass made the
consequence concrete - the Physics Agent succeeded while the Texture Agent
rejected its plan outright, because the candidate had seven materials, zero
shaders, and nothing bound at the default material purpose. A texture stage
needs a surface to paint onto, and there was none.

This module authors that surface: one ``UsdPreviewSurface`` per named
appearance role, bound at the default purpose to the requested geometry, seeded
with a base colour. It writes a new stage and never touches the source, and it
touches only appearance - physics-purpose bindings, joints, articulation roots,
and rigid bodies are carried through untouched and counted in the receipt.

A flat base colour is not a texture pass. Even when the colour is the measured
albedo of the geometry it replaces, the receipt records the appearance as a
candidate, never as observed site truth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


RENDER_MATERIAL_SCAFFOLD_SCHEMA_VERSION = "articulated_render_material_scaffold.v1"
RENDER_MATERIAL_SCOPE_SUFFIX = "Looks/Render"
DEFAULT_BASE_COLOR = (0.72, 0.72, 0.74)
DEFAULT_ROUGHNESS = 0.5
DEFAULT_METALLIC = 0.0


class ArticulatedRenderMaterialError(ValueError):
    """Stable, sorted render-material authoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _colour(value: Any, fallback: tuple[float, float, float]) -> list[float]:
    if value is None:
        return [float(item) for item in fallback]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArticulatedRenderMaterialError(["render_material_base_color_invalid"])
    if len(value) != 3:
        raise ArticulatedRenderMaterialError(["render_material_base_color_invalid"])
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ArticulatedRenderMaterialError(["render_material_base_color_invalid"])
        out.append(min(max(float(item), 0.0), 1.0))
    return out


def _scalar(value: Any, fallback: float, error: str) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArticulatedRenderMaterialError([error])
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ArticulatedRenderMaterialError([error])
    return number


def ensure_render_material_scaffold(
    *,
    source_usd_path: str | Path,
    destination: str | Path,
    surfaces: Sequence[Mapping[str, Any]],
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Author bound preview surfaces on a copy of an articulated candidate."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedRenderMaterialError(
            ["render_material_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ArticulatedRenderMaterialError(["render_material_source_missing"])
    if output == source:
        raise ArticulatedRenderMaterialError(["render_material_destination_is_source"])
    if not surfaces:
        raise ArticulatedRenderMaterialError(["render_material_surfaces_missing"])

    errors: list[str] = []
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, Mapping):
            errors.append(f"render_material_surface_{index}_invalid")
            continue
        material_id = str(surface.get("material_id") or "")
        if not material_id or "/" in material_id or material_id in seen:
            errors.append(
                f"render_material_material_id_duplicated_or_invalid:{material_id}"
            )
            continue
        seen.add(material_id)
        paths = [str(item) for item in (surface.get("prim_paths") or [])]
        if not paths:
            errors.append(f"render_material_surface_{index}_prim_paths_missing")
            continue
        specs.append(
            {
                "material_id": material_id,
                "prim_paths": paths,
                "base_color": _colour(surface.get("base_color"), DEFAULT_BASE_COLOR),
                "roughness": _scalar(
                    surface.get("roughness"),
                    DEFAULT_ROUGHNESS,
                    "render_material_roughness_invalid",
                ),
                "metallic": _scalar(
                    surface.get("metallic"),
                    DEFAULT_METALLIC,
                    "render_material_metallic_invalid",
                ),
                "observed_albedo": bool(surface.get("observed_albedo", False)),
            }
        )
    if errors:
        raise ArticulatedRenderMaterialError(errors)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise ArticulatedRenderMaterialError(["render_material_source_unreadable"])

    def _resolved_physics_materials(current: Any) -> dict[str, str]:
        """Physics-purpose material per mesh, counting only real physics materials.

        A render material bound at the all-purpose slot becomes the fallback for
        every purpose, physics included. That is harmless when the bound
        material carries no physics API - PhysX uses its defaults either way -
        but it must never displace a material that does.
        """

        resolved: dict[str, str] = {}
        for prim in current.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                materialPurpose="physics"
            )
            target = bound.GetPrim() if bound else None
            if target and target.IsValid() and target.HasAPI(UsdPhysics.MaterialAPI):
                resolved[str(prim.GetPath())] = str(target.GetPath())
        return resolved

    physics_before = _resolved_physics_materials(stage)

    for spec in specs:
        for path in spec["prim_paths"]:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                errors.append(f"render_material_target_prim_missing:{path}")
            elif not prim.IsA(UsdGeom.Mesh):
                errors.append(f"render_material_target_prim_not_a_mesh:{path}")
    if errors:
        output.unlink(missing_ok=True)
        raise ArticulatedRenderMaterialError(errors)

    # The material scope hangs off whatever the stage calls its default prim,
    # so an asset that does not happen to be named /Asset still works.
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        output.unlink(missing_ok=True)
        raise ArticulatedRenderMaterialError(
            ["render_material_default_prim_missing"]
        )
    scope = f"{default_prim.GetPath()}/{RENDER_MATERIAL_SCOPE_SUFFIX}"
    for spec in specs:
        material_path = f"{scope}/{spec['material_id']}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*spec["base_color"])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(spec["roughness"])
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(spec["metallic"])
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        for path in spec["prim_paths"]:
            prim = stage.GetPrimAtPath(path)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            existing, _ = binding.ComputeBoundMaterial(materialPurpose="physics")
            binding.Bind(material)
            if (
                existing
                and existing.GetPrim().IsValid()
                and existing.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
            ):
                binding.Bind(existing, materialPurpose="physics")
        spec["material_path"] = material_path
        spec["bound_prim_count"] = len(spec["prim_paths"])

    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output))

    physics_after = _resolved_physics_materials(stage)
    if physics_after != physics_before:
        output.unlink(missing_ok=True)
        raise ArticulatedRenderMaterialError(
            ["render_material_physics_bindings_changed"]
        )

    receipt: dict[str, Any] = {
        "schema_version": RENDER_MATERIAL_SCAFFOLD_SCHEMA_VERSION,
        "status": "render_material_scaffold_authored",
        "source_usd_path": str(source),
        "source_usd_sha256": _sha256(source),
        "render_ready_usd_path": str(output),
        "render_ready_usd_sha256": _sha256(output),
        "surfaces": specs,
        "preserved": {
            "assembly_joint_count": len(
                [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
            ),
            "articulation_root_count": len(
                [
                    p
                    for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI)
                ]
            ),
            "rigid_body_count": len(
                [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            ),
            "physics_bindings_unchanged": True,
        },
        "claim_boundary": {
            "flat_colour_is_not_a_texture_pass": True,
            "appearance_is_candidate_not_observed_truth": True,
            "source_usd_modified": False,
            "native_simulator_qualified": False,
        },
        "receipt_path": str(
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else output.with_name(output.stem + "_render_material_receipt.json")
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["receipt_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ArticulatedRenderMaterialError",
    "DEFAULT_BASE_COLOR",
    "RENDER_MATERIAL_SCAFFOLD_SCHEMA_VERSION",
    "RENDER_MATERIAL_SCOPE_SUFFIX",
    "ensure_render_material_scaffold",
]
