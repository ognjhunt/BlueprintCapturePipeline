"""Pin an exported NuRec appearance volume's layer transform to identity.

The pinned upstream 3DGRUT USDZ exporter bakes a fixed axis-convention matrix
onto the volume prim unconditionally::

    ( (-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1) )   # p -> (-x, -z, -y)

That matrix maps the exporter's OpenGL/Y-up internal frame to USD's Z-up world,
so it is correct only for a tensor that is still in that internal frame.  When
the reconstruction is fit to metric capture poses the tensor is *already* in the
world frame, and the matrix becomes a spurious rigid motion: it mirrors the room
and turns it upside down.  For scene 840920 it moved an 11 m room roughly 13 m
away and entirely below the floor, and fourteen paid arena runs rendered black.

``export_usdz.apply_normalizing_transform = False`` does **not** prevent this.
That flag governs a different thing -- the camera-derived recenter/upright
transform the exporter can bake into the *point data* -- and the runner already
sets it.  The axis matrix is authored separately, into the USD layer, and no
exporter setting suppresses it.  Turning off the normalizing transform and
assuming the layer was therefore clean is exactly how the defect shipped.

``aura_nurec_usdz`` states the same fact from the other side: it pins an
identity transform because its positions are already in the admitted world
frame, and copying the shipped package's matrix "would mirror and rotate the
room while looking entirely plausible".  This module applies that decision to
the export path, so the placement of an appearance volume is the spawn pose the
plan authored and nothing else.

Nothing here renders, and nothing here decides that a volume looks correct.
The rewrite is verified through the frame-alignment gate's own measurement
rather than by trusting the edit.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from .aura_nurec_usdz import write_aligned_usdz
from .native_task_appearance_frame_alignment import (
    NativeTaskAppearanceFrameAlignmentError,
    measure_native_task_appearance_frame,
)

SCHEMA_VERSION = "nurec_usdz_layer_transform.v1"

# The matrix the pinned upstream exporter bakes in, recorded so a receipt can
# name what was removed instead of only what remains.
EXPORTER_AXIS_MATRIX: tuple[tuple[float, ...], ...] = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

IDENTITY_MATRIX: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

IDENTITY_LITERAL = "( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )"

NUREC_VOLUME_MARKER = "omni:nurec:isNuRecVolume"

# A four-row ``matrix4d`` value.  ``.timeSamples`` is matched so an animated
# transform is refused by name rather than silently left in place.
_TRANSFORM = re.compile(
    r"[ \t]*matrix4d[ \t]+xformOp:transform(?P<samples>\.timeSamples)?[ \t]*=[ \t]*"
    r"\([ \t]*\([^()]*\)[ \t]*,[ \t]*\([^()]*\)[ \t]*,[ \t]*\([^()]*\)[ \t]*,"
    r"[ \t]*\([^()]*\)[ \t]*\)"
)

_XFORM_OP_ORDER = re.compile(
    r"uniform[ \t]+token\[\][ \t]+xformOpOrder[ \t]*=[ \t]*\[(?P<ops>[^\]]*)\]"
)

_ANY_XFORM_OP = re.compile(r"[ \t]*(?:custom[ \t]+)?\w+(?:\[\])?[ \t]+xformOp:(?P<op>[\w:]+)")


class NuRecUsdzLayerTransformError(ValueError):
    """Fail-closed errors from rewriting an appearance volume's transform."""

    def __init__(self, errors):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _volume_block(text: str) -> tuple[int, int]:
    """Return the character span of the ``def Volume`` block that is NuRec.

    Scoped deliberately: only the prim carrying the NuRec marker is rewritten,
    so a transform authored anywhere else in the package is reported by the
    verification step rather than silently erased.
    """

    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bdef\b[^\n{]*\bVolume\b[^\n{]*", text):
        brace = text.find("{", match.end())
        if brace < 0:
            continue
        depth = 0
        index = brace
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth != 0:
            raise NuRecUsdzLayerTransformError(
                ["nurec_usdz_layer_volume_block_unterminated"]
            )
        block = text[match.start() : index + 1]
        if NUREC_VOLUME_MARKER in block:
            spans.append((match.start(), index + 1))
    if len(spans) != 1:
        raise NuRecUsdzLayerTransformError(["nurec_usdz_layer_volume_not_exact"])
    return spans[0]


def _rewrite_volume_layer(text: str) -> tuple[str, str | None]:
    """Replace the NuRec volume's transform with identity.

    Returns the new layer text and the matrix literal that was replaced, or
    ``None`` when the prim carried no transform at all (already identity).
    """

    start, end = _volume_block(text)
    block = text[start:end]

    order = _XFORM_OP_ORDER.search(block)
    if order is not None:
        ops = [op.strip().strip('"') for op in order.group("ops").split(",") if op.strip()]
        if ops not in ([], ["xformOp:transform"]):
            # Translate/orient/scale stacks compose to a placement this module
            # cannot rewrite to identity by editing one value, and guessing
            # would reintroduce exactly the class of silent mis-placement this
            # exists to remove.
            raise NuRecUsdzLayerTransformError(
                ["nurec_usdz_layer_xform_op_order_unsupported:" + ",".join(ops)]
            )
    other = {
        match.group("op")
        for match in _ANY_XFORM_OP.finditer(block)
        if match.group("op") != "transform"
    }
    if other:
        raise NuRecUsdzLayerTransformError(
            ["nurec_usdz_layer_xform_op_unsupported:" + ",".join(sorted(other))]
        )

    matches = list(_TRANSFORM.finditer(block))
    if any(match.group("samples") for match in matches):
        raise NuRecUsdzLayerTransformError(["nurec_usdz_layer_transform_time_sampled"])
    if len(matches) > 1:
        raise NuRecUsdzLayerTransformError(["nurec_usdz_layer_transform_not_exact"])
    if not matches:
        return text, None

    match = matches[0]
    original = match.group(0).strip()
    indent = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip(" \t"))]
    replaced = block[: match.start()] + indent + "matrix4d xformOp:transform = " + IDENTITY_LITERAL + block[match.end() :]
    return text[:start] + replaced + text[end:], original


def _members(path: Path) -> list[tuple[str, bytes]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                not infos
                or len(names) != len(set(names))
                or any(info.is_dir() or info.flag_bits & 0x1 for info in infos)
            ):
                raise ValueError
            return [(info.filename, archive.read(info)) for info in infos]
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise NuRecUsdzLayerTransformError(["nurec_usdz_layer_archive_unreadable"]) from exc


def pin_nurec_usdz_layer_transform_to_identity(usdz_path: str | Path) -> dict[str, Any]:
    """Rewrite an appearance USDZ so its NuRec volume composes at identity.

    The ``.nurec`` payload bytes are never touched -- only the USD layer that
    places it -- so the gaussians the capture recorded are carried through
    unchanged and only their claimed frame-to-world mapping is corrected.

    The result is verified by re-measuring the packaged asset through
    :func:`measure_native_task_appearance_frame`, the same oracle the arena
    plan gate uses.  A rewrite that did not actually reach the composed
    transform fails here rather than at a rented GPU.
    """

    path = Path(usdz_path).expanduser()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise NuRecUsdzLayerTransformError(["nurec_usdz_layer_asset_missing"])

    try:
        before = measure_native_task_appearance_frame(path)
    except NativeTaskAppearanceFrameAlignmentError as exc:
        raise NuRecUsdzLayerTransformError(list(exc.errors)) from exc

    members = _members(path)
    rewritten: list[tuple[str, bytes]] = []
    replaced_in: list[str] = []
    replaced_literal: str | None = None
    for name, body in members:
        if not name.lower().endswith((".usda", ".usd")):
            rewritten.append((name, body))
            continue
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            # A binary crate layer.  The exporter changing format is a real
            # possibility and it must not be mistaken for "no transform".
            raise NuRecUsdzLayerTransformError(
                ["nurec_usdz_layer_binary_crate_unsupported:" + name]
            ) from None
        if NUREC_VOLUME_MARKER not in text:
            rewritten.append((name, body))
            continue
        new_text, literal = _rewrite_volume_layer(text)
        if literal is not None:
            replaced_in.append(name)
            replaced_literal = literal
        rewritten.append((name, new_text.encode("utf-8")))

    if len(replaced_in) > 1:
        raise NuRecUsdzLayerTransformError(["nurec_usdz_layer_volume_not_exact"])

    write_aligned_usdz(path, rewritten)

    try:
        after = measure_native_task_appearance_frame(path)
    except NativeTaskAppearanceFrameAlignmentError as exc:
        raise NuRecUsdzLayerTransformError(list(exc.errors)) from exc
    if not after["layer_transform_is_identity"]:
        # Something outside the volume prim still places the volume.  Refuse
        # rather than ship an asset whose receipt would claim identity.
        raise NuRecUsdzLayerTransformError(
            ["nurec_usdz_layer_transform_not_identity_after_rewrite"]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "identity_pinned",
        "volume_prim_path": after["volume_prim_path"],
        "layer_transform_before_row_major": before["layer_transform_row_major"],
        "layer_transform_after_row_major": after["layer_transform_row_major"],
        "layer_transform_was_identity": bool(before["layer_transform_is_identity"]),
        "layer_transform_is_identity": True,
        "exporter_axis_matrix_removed": bool(
            before["layer_transform_row_major"]
            == [list(row) for row in EXPORTER_AXIS_MATRIX]
        ),
        "rewritten_layers": sorted(replaced_in),
        "replaced_transform_literal": replaced_literal,
        "gaussian_count": int(after["gaussian_count"]),
        "stored_tensor_occupied_bounds_m": after["stored_tensor_occupied_bounds_m"],
        "payload_bytes_preserved": True,
        "verified_by": "native_task_appearance_frame_alignment.measure",
    }


__all__ = [
    "EXPORTER_AXIS_MATRIX",
    "IDENTITY_MATRIX",
    "NUREC_VOLUME_MARKER",
    "NuRecUsdzLayerTransformError",
    "SCHEMA_VERSION",
    "pin_nurec_usdz_layer_transform_to_identity",
]
