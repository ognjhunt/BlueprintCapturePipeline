"""Package an authored NuRec volume as a USDZ Isaac can reference.

The layout mirrors the shipped InteriorGS package exactly -- a ``Volume`` prim
carrying ``omni:nurec:isNuRecVolume`` with two ``OmniNuRecFieldAsset`` children
pointing at the same ``.nurec`` payload -- because that is the arrangement Isaac
has demonstrably rendered.

Two things are deliberately *not* copied from it.

``defaultPrim`` is set and asserted.  Arena brings an asset in with
``Object(usd_path=...)``, a USD reference, and a reference into a layer without
a default prim resolves to nothing: the previous appearance shipped, composed
into no geometry at all, and five runs were spent looking for a render bug that
was a missing metadata field.

The transform is identity.  The shipped package carries a mirroring
``xformOp:transform`` because its stored positions are in a NuRec-internal
frame -- its ``extent`` matches the raw positions, and the matrix maps those to
world.  Aura's positions are already in the admitted world frame, so copying
that matrix would mirror and rotate the room while looking entirely plausible.
"""

from __future__ import annotations

import struct
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .nurec_volume_codec import encode_nurec_bytes, gaussian_arrays

AURA_NUREC_USDZ_SCHEMA_VERSION = "aura_nurec_usdz.v1"

DEFAULT_PRIM = "World"
# The reference nests: the root layer's over "gauss" pulls in gauss.usda
# whose default prim is World, so the Volume lands one level deeper.  This
# is the path the shipped InteriorGS package composes to as well.
VOLUME_PRIM_PATH = "/World/gauss/gauss"

# USDZ requires stored (uncompressed) members aligned to 64 bytes so the
# runtime can memory-map them in place.
USDZ_ALIGNMENT = 64

_RENDER_SETTINGS = """(
    customLayerData = {
        dictionary renderSettings = {
            int "rtx:directLighting:sampledLighting:samplesPerPixel" = 8
            bool "rtx:material:enableRefraction" = 0
            bool "rtx:matteObject:visibility:secondaryRays" = 1
            bool "rtx:post:histogram:enabled" = 0
            bool "rtx:post:registeredCompositing:invertColorCorrection" = 1
            bool "rtx:post:registeredCompositing:invertToneMap" = 1
            int "rtx:post:tonemap:op" = 2
            bool "rtx:raytracing:fractionalCutoutOpacity" = 0
            string "rtx:rendermode" = "RaytracedLighting"
        }
    }
    defaultPrim = "{default_prim}"
    metersPerUnit = 1
    upAxis = "Z"
)"""
"""Carried verbatim from the shipped package.

Omitting these produced a render that was unmistakably the right room and
35 percent saturated: the geometry was correct and the tone mapping was not.
``invertToneMap`` and ``tonemap:op`` are how a NuRec volume's stored radiance
is meant to reach display, and a splat authored without them is being asked to
survive a tone curve its reconstruction already accounted for.
"""

_ROOT_LAYER = """#usda 1.0
{render_settings}

def Xform "{default_prim}"
{{
    over "gauss" (
        prepend references = @gauss.usda@
    )
    {{
    }}
}}
"""

_VOLUME_LAYER = """#usda 1.0
{render_settings}

def Xform "{default_prim}"
{{
    def Volume "gauss"
    {{
        float3[] extent = [({x0}, {y0}, {z0}), ({x1}, {y1}, {z1})]
        custom rel field:density = </World/gauss/density_field>
        custom rel field:emissiveColor = </World/gauss/emissive_color_field>
        custom float3 omni:nurec:crop:maxBounds = ({x1}, {y1}, {z1})
        custom float3 omni:nurec:crop:minBounds = ({x0}, {y0}, {z0})
        custom bool omni:nurec:isNuRecVolume = 1
        custom float3 omni:nurec:offset = (0, 0, 0)
        custom bool omni:nurec:useProxyTransform = 0
        custom rel proxy
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]

        def OmniNuRecFieldAsset "density_field"
        {{
            custom token fieldDataType = "float"
            custom token fieldName = "density"
            custom token fieldRole = "density"
            custom asset filePath = @./{payload_name}@
        }}

        def OmniNuRecFieldAsset "emissive_color_field"
        {{
            custom token fieldDataType = "float3"
            custom token fieldName = "emissiveColor"
            custom token fieldRole = "emissiveColor"
            custom asset filePath = @./{payload_name}@
            custom float4 omni:nurec:ccmB = (0, 0, 1, 0)
            custom float4 omni:nurec:ccmG = (0, 1, 0, 0)
            custom float4 omni:nurec:ccmR = (1, 0, 0, 0)
        }}
    }}
}}
"""


def _write_aligned_usdz(out_path: Path, members: Sequence[tuple[str, bytes]]) -> None:
    """Write a USDZ: stored members, each payload aligned to 64 bytes."""

    with out_path.open("wb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in members:
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_STORED
                # Header length is fixed by the name; pad with an extra field
                # so the data itself lands on the alignment boundary.
                header = 30 + len(name.encode("utf-8"))
                position = handle.tell() + header
                padding = (-position) % USDZ_ALIGNMENT
                if padding:
                    if padding < 4:
                        padding += USDZ_ALIGNMENT
                    info.extra = struct.pack("<hh", 0x1986, padding - 4) + b"\0" * (padding - 4)
                archive.writestr(info, data)


def write_aura_nurec_usdz(
    document: Mapping[str, Any],
    out_path: str | Path,
    *,
    payload_name: str = "aura_appearance.nurec",
) -> dict[str, Any]:
    """Package a NuRec container document as a referenceable USDZ."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = encode_nurec_bytes(document)

    positions = gaussian_arrays(document)["positions"].astype(np.float32)
    low = positions.min(axis=0)
    high = positions.max(axis=0)

    render_settings = _RENDER_SETTINGS.replace("{default_prim}", DEFAULT_PRIM)
    volume_layer = _VOLUME_LAYER.format(
        default_prim=DEFAULT_PRIM,
        render_settings=render_settings,
        payload_name=payload_name,
        x0=float(low[0]), y0=float(low[1]), z0=float(low[2]),
        x1=float(high[0]), y1=float(high[1]), z1=float(high[2]),
    )
    root_layer = _ROOT_LAYER.format(
        render_settings=render_settings, default_prim=DEFAULT_PRIM
    )
    _write_aligned_usdz(
        out_path,
        [
            ("default.usda", root_layer.encode("utf-8")),
            ("gauss.usda", volume_layer.encode("utf-8")),
            (payload_name, payload),
        ],
    )
    return {
        "schema_version": AURA_NUREC_USDZ_SCHEMA_VERSION,
        "status": "completed",
        "output": str(out_path),
        "output_bytes": out_path.stat().st_size,
        "payload_name": payload_name,
        "payload_bytes": len(payload),
        "default_prim": f"/{DEFAULT_PRIM}",
        "volume_prim": VOLUME_PRIM_PATH,
        "gaussian_count": int(positions.shape[0]),
        "extent_min": [float(v) for v in low],
        "extent_max": [float(v) for v in high],
        # Identity, because Aura's positions are already in the admitted world
        # frame.  The shipped InteriorGS package mirrors instead, and copying
        # that would rotate the room while looking entirely plausible.
        "world_transform": "identity",
        "render_settings": "shipped_interiorgs_verbatim",
        "authoring": dict(document.get("_blueprint_authoring") or {}),
    }


__all__ = [
    "AURA_NUREC_USDZ_SCHEMA_VERSION",
    "DEFAULT_PRIM",
    "VOLUME_PRIM_PATH",
    "write_aura_nurec_usdz",
]
