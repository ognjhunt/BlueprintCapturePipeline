"""Connect generated PBR maps to the surfaces that are supposed to show them.

The Content Agents produce a texture set and the render scaffold produces a
bound UsdPreviewSurface, and nothing joins them. Left that way the twin renders
in flat scaffold colours - which looks deliberate rather than broken, so it
survives review, and every camera observation a policy ever sees is of an
object that does not resemble the one that was captured.

Three ways this goes wrong quietly, all guarded here:

A dangling asset path renders as flat magenta or black rather than raising, so
every map is checked to exist before anything is authored.

A texture reader with no ``st`` input samples a single texel and paints the
whole surface one colour. It looks like a solid material, not a missing UV
connection, so the primvar reader is mandatory rather than implied.

And colour space. Albedo is authored sRGB; roughness, metalness and normals are
data and must be read raw. Reading a data map as sRGB gamma-shifts it - nothing
errors, the render still looks like a render, and the surface response is
simply wrong in a direction nobody can eyeball. It is the easiest texture
mistake to make and the hardest to catch afterwards.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


RENDER_TEXTURE_SCHEMA_VERSION = "articulated_render_textures.v1"
UV_PRIMVAR_NAME = "st"
# Colour data is display-referred; everything else is measurement.
CHANNELS = (
    ("albedo", "diffuseColor", "rgb", "sRGB"),
    ("roughness", "roughness", "r", "raw"),
    ("metallic", "metallic", "r", "raw"),
    ("normal", "normal", "rgb", "raw"),
)


class ArticulatedRenderTextureError(ValueError):
    """Stable, sorted texture-binding failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bind_render_textures(
    *,
    source_usd_path: str | Path,
    destination: str | Path,
    bindings: Sequence[dict[str, Any]],
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Author UV-connected, correctly-encoded texture readers on a copy."""

    try:
        from pxr import Sdf, Usd, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedRenderTextureError(
            ["articulated_render_texture_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ArticulatedRenderTextureError(
            ["articulated_render_texture_source_missing"]
        )
    if output == source:
        raise ArticulatedRenderTextureError(
            ["articulated_render_texture_destination_is_source"]
        )
    if not bindings:
        raise ArticulatedRenderTextureError(
            ["articulated_render_texture_bindings_missing"]
        )

    probe = Usd.Stage.Open(str(source))
    if probe is None:
        raise ArticulatedRenderTextureError(
            ["articulated_render_texture_source_unreadable"]
        )

    errors: list[str] = []
    resolved: list[dict[str, Any]] = []
    for index, raw in enumerate(bindings):
        material_path = str(raw.get("material_path") or "")
        if not material_path:
            errors.append(f"articulated_render_texture_material_path_missing:{index}")
            continue
        material = UsdShade.Material.Get(probe, material_path)
        if not material or not material.GetPrim().IsValid():
            errors.append(
                f"articulated_render_texture_material_missing:{material_path}"
            )
            continue
        maps: list[dict[str, Any]] = []
        for name, shader_input, channel, colour_space in CHANNELS:
            value = raw.get(f"{name}_path")
            if not value:
                continue
            texture = Path(str(value)).expanduser().resolve()
            if not texture.is_file():
                # A dangling path paints flat magenta instead of raising.
                errors.append(
                    f"articulated_render_texture_file_missing:{name}:{texture}"
                )
                continue
            maps.append(
                {
                    "channel_name": name,
                    "shader_input": shader_input,
                    "output_channel": channel,
                    "source_colour_space": colour_space,
                    "texture_path": str(texture),
                    "texture_sha256": _sha256(texture),
                }
            )
        if not maps:
            errors.append(
                f"articulated_render_texture_no_maps_for_material:{material_path}"
            )
            continue
        resolved.append({"material_path": material_path, "maps": maps})
    if errors:
        raise ArticulatedRenderTextureError(errors)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        output.unlink(missing_ok=True)
        raise ArticulatedRenderTextureError(
            ["articulated_render_texture_source_unreadable"]
        )

    for row in resolved:
        material = UsdShade.Material.Get(stage, row["material_path"])
        surface = material.ComputeSurfaceSource()[0]
        if not surface or not surface.GetPrim().IsValid():
            errors.append(
                "articulated_render_texture_material_has_no_surface:"
                f"{row['material_path']}"
            )
            continue
        # One reader per material: without an st connection every texture
        # samples a single texel and the surface renders flat.
        reader = UsdShade.Shader.Define(
            stage, f"{row['material_path']}/uv_reader"
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(UV_PRIMVAR_NAME)
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        for entry in row["maps"]:
            texture = UsdShade.Shader.Define(
                stage,
                f"{row['material_path']}/{entry['channel_name']}_texture",
            )
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                entry["texture_path"]
            )
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                entry["source_colour_space"]
            )
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            channel = entry["output_channel"]
            value_type = (
                Sdf.ValueTypeNames.Float3
                if channel == "rgb"
                else Sdf.ValueTypeNames.Float
            )
            texture.CreateOutput(channel, value_type)
            surface_input = surface.GetInput(entry["shader_input"])
            if not surface_input:
                surface_input = surface.CreateInput(
                    entry["shader_input"],
                    Sdf.ValueTypeNames.Color3f
                    if channel == "rgb"
                    else Sdf.ValueTypeNames.Float,
                )
            surface_input.ConnectToSource(texture.ConnectableAPI(), channel)
    if errors:
        output.unlink(missing_ok=True)
        raise ArticulatedRenderTextureError(errors)

    stage.GetRootLayer().Save()

    receipt: dict[str, Any] = {
        "schema_version": RENDER_TEXTURE_SCHEMA_VERSION,
        "status": "render_textures_bound",
        "source_usd_path": str(source),
        "source_usd_sha256": _sha256(source),
        "textured_usd_path": str(output),
        "textured_usd_sha256": _sha256(output),
        "uv_primvar_name": UV_PRIMVAR_NAME,
        "uv_reader_authored": True,
        "bindings": resolved,
        "claim_boundary": {
            "maps_are_generated_not_photographed": True,
            "colour_maps_srgb_data_maps_raw": True,
            "appearance_is_candidate_not_observed_truth": True,
        },
        "receipt_path": str(
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else output.with_name(output.stem + "_texture_receipt.json")
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["receipt_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ArticulatedRenderTextureError",
    "CHANNELS",
    "RENDER_TEXTURE_SCHEMA_VERSION",
    "UV_PRIMVAR_NAME",
    "bind_render_textures",
]
