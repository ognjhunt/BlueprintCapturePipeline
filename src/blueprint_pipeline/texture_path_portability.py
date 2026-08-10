"""Refuse USD layers whose texture references cannot survive transport.

A USD layer references its textures through ``asset`` attributes. Absolute
paths resolve on the machine that authored them and nowhere else; a provider
instance renders the geometry untextured and calls it a warning. The twin
carried five such references for the whole v12-v19 lineage, undetonated only
because the appliance had never yet appeared on camera.

The audit is textual on purpose: it reads the ``@...@`` asset references out
of the layer, refuses absolute ones, and proves each relative one resolves -
against the layer's own directory by default, or against the directory the
layer will occupy on the provider when the bundle relocates it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence


TEXTURE_PATH_PORTABILITY_SCHEMA_VERSION = "texture_path_portability.v1"
# @path@ references; textures are the ones that end in image extensions.
_ASSET_REFERENCE = re.compile(r"@([^@]+)@")
_TEXTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga", ".bmp")


class TexturePathPortabilityError(ValueError):
    """Stable, sorted portability failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def audit_texture_path_portability(
    *,
    asset_path: str | Path,
    resolve_as_if_layer_lived_in: str | Path | None = None,
) -> dict[str, Any]:
    """Prove every texture reference is relative and resolves after transport."""

    layer = Path(asset_path).expanduser()
    if not layer.is_file():
        raise TexturePathPortabilityError(
            [f"texture_path_portability_layer_missing:{layer}"]
        )
    base = (
        Path(resolve_as_if_layer_lived_in).expanduser()
        if resolve_as_if_layer_lived_in is not None
        else layer.parent
    )

    references = [
        match.group(1).strip()
        for match in _ASSET_REFERENCE.finditer(layer.read_text(encoding="utf-8"))
    ]
    textures = [
        ref for ref in references if ref.lower().endswith(_TEXTURE_SUFFIXES)
    ]

    errors: list[str] = []
    resolved: list[dict[str, Any]] = []
    for ref in textures:
        if Path(ref).is_absolute():
            errors.append(
                f"texture_path_portability_absolute_texture_path:{Path(ref).name}"
            )
            continue
        candidate = (base / ref).resolve()
        if not candidate.is_file():
            errors.append(
                f"texture_path_portability_texture_unresolvable:{ref}"
            )
            continue
        resolved.append({"reference": ref, "resolved": str(candidate)})
    if errors:
        raise TexturePathPortabilityError(errors)

    return {
        "schema_version": TEXTURE_PATH_PORTABILITY_SCHEMA_VERSION,
        "layer": str(layer),
        "resolved_against": str(base),
        "texture_count": len(textures),
        "resolved_textures": resolved,
        "all_relative_and_resolvable": True,
        "claim_boundary": {
            "resolution_is_checked_not_rendering": True,
            "the_audit_reads_the_layer_text_not_the_composed_stage": True,
        },
    }


__all__ = [
    "TEXTURE_PATH_PORTABILITY_SCHEMA_VERSION",
    "TexturePathPortabilityError",
    "audit_texture_path_portability",
]
