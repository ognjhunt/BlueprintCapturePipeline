"""Hermetic contract tests for pinning an exported NuRec volume to identity.

Every fixture is a real USDZ carrying a real gzip+MessagePack ``.nurec``
container, read back through pxr and the shipped codec.  No GPU, no Isaac, no
network.
"""

from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.native_task_appearance_frame_alignment import (
    measure_native_task_appearance_frame,
    qualify_native_task_appearance_frame_alignment,
)
from blueprint_pipeline.nurec_usdz_layer_transform import (
    EXPORTER_AXIS_MATRIX,
    NuRecUsdzLayerTransformError,
    pin_nurec_usdz_layer_transform_to_identity,
)
from blueprint_pipeline.nurec_volume_codec import decode_nurec_bytes, gaussian_arrays
from tests.test_native_task_appearance_frame_alignment import (
    IDENTITY,
    INSIDE,
    _matrix_literal,
    nurec_payload,
    room_positions,
    write_appearance_usdz,
)


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_the_exporter_axis_matrix_is_replaced_with_identity(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "repaired_scene.usdz",
        room_positions(),
        matrix=EXPORTER_AXIS_MATRIX,
    )
    # The defect, stated before the fix: the room composes outside the scene.
    before = qualify_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )
    assert before["status"] == "misaligned"

    receipt = pin_nurec_usdz_layer_transform_to_identity(asset)

    assert receipt["status"] == "identity_pinned"
    assert receipt["layer_transform_was_identity"] is False
    assert receipt["exporter_axis_matrix_removed"] is True
    assert receipt["layer_transform_is_identity"] is True
    assert receipt["rewritten_layers"] == ["gauss.usda"]
    assert receipt["layer_transform_before_row_major"] == [
        list(row) for row in EXPORTER_AXIS_MATRIX
    ]
    assert receipt["layer_transform_after_row_major"] == [list(row) for row in IDENTITY]
    # Measured out of the shipped bytes, not taken from the receipt.
    assert measure_native_task_appearance_frame(asset)["layer_transform_is_identity"]
    after = qualify_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )
    assert after["status"] == "aligned"
    assert after["blockers"] == []


def test_the_gaussian_payload_is_carried_through_byte_identical(tmp_path):
    """Only the layer that places the volume may change."""

    asset = write_appearance_usdz(
        tmp_path / "repaired_scene.usdz",
        room_positions(),
        matrix=EXPORTER_AXIS_MATRIX,
        payload_name="repaired_scene.nurec",
        payload_first=True,
    )
    before = _members(asset)
    pin_nurec_usdz_layer_transform_to_identity(asset)
    after = _members(asset)

    assert hashlib.sha256(after["repaired_scene.nurec"]).hexdigest() == hashlib.sha256(
        before["repaired_scene.nurec"]
    ).hexdigest()
    assert after["default.usda"] == before["default.usda"]
    assert after["gauss.usda"] != before["gauss.usda"]
    positions = gaussian_arrays(decode_nurec_bytes(after["repaired_scene.nurec"]))[
        "positions"
    ]
    assert positions.shape == room_positions().shape


def test_member_names_order_and_alignment_survive_the_rewrite(tmp_path):
    """Three separate downstream validators assert exactly this shape."""

    asset = write_appearance_usdz(
        tmp_path / "repaired_scene.usdz",
        room_positions(),
        matrix=EXPORTER_AXIS_MATRIX,
        payload_name="repaired_scene.nurec",
        payload_first=True,
    )
    pin_nurec_usdz_layer_transform_to_identity(asset)

    with zipfile.ZipFile(asset) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == [
            "default.usda",
            "repaired_scene.nurec",
            "gauss.usda",
        ]
        for info in infos:
            assert info.compress_type == zipfile.ZIP_STORED
            header = 30 + len(info.filename.encode("utf-8")) + len(info.extra)
            assert (info.header_offset + header) % 64 == 0


def test_pinning_an_already_identity_volume_changes_nothing_it_claims(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "repaired_scene.usdz", room_positions(), matrix=IDENTITY
    )
    receipt = pin_nurec_usdz_layer_transform_to_identity(asset)

    assert receipt["layer_transform_was_identity"] is True
    assert receipt["exporter_axis_matrix_removed"] is False
    assert receipt["layer_transform_is_identity"] is True


def test_pinning_is_idempotent(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "repaired_scene.usdz",
        room_positions(),
        matrix=EXPORTER_AXIS_MATRIX,
    )
    first = pin_nurec_usdz_layer_transform_to_identity(asset)
    second = pin_nurec_usdz_layer_transform_to_identity(asset)

    assert first["exporter_axis_matrix_removed"] is True
    assert second["layer_transform_was_identity"] is True
    assert second["layer_transform_after_row_major"] == first[
        "layer_transform_after_row_major"
    ]
    assert _members(asset)["gauss.usda"] == _members(asset)["gauss.usda"]


def test_a_transform_outside_the_volume_prim_is_refused_not_ignored(tmp_path):
    """The rewrite is scoped to the volume; anything else must fail closed.

    A package that places the room from an ancestor prim would still compose
    away from the scene after the volume's own matrix is cleared, and a
    receipt claiming identity would then be false.
    """

    volume_layer = (
        '#usda 1.0\n(\n    defaultPrim = "World"\n    upAxis = "Z"\n)\n\n'
        'def Xform "World"\n{\n'
        "    matrix4d xformOp:transform = "
        + _matrix_literal(EXPORTER_AXIS_MATRIX)
        + '\n    uniform token[] xformOpOrder = ["xformOp:transform"]\n\n'
        '    def Volume "gauss"\n    {\n'
        "        custom rel field:density = </World/gauss/density_field>\n"
        "        custom bool omni:nurec:isNuRecVolume = 1\n"
        "        matrix4d xformOp:transform = " + _matrix_literal(IDENTITY) + "\n"
        '        uniform token[] xformOpOrder = ["xformOp:transform"]\n\n'
        '        def OmniNuRecFieldAsset "density_field"\n        {\n'
        '            custom asset filePath = @./scene.nurec@\n        }\n'
        "    }\n}\n"
    )
    asset = _write_usdz(
        tmp_path / "repaired_scene.usdz",
        [
            (
                "default.usda",
                '#usda 1.0\n(\n    defaultPrim = "World"\n)\n\n'
                'def Xform "World"\n{\n    over "gauss" (\n'
                "        prepend references = @gauss.usda@\n    )\n    {\n    }\n}\n",
            ),
            ("gauss.usda", volume_layer),
        ],
        room_positions(),
    )
    with pytest.raises(NuRecUsdzLayerTransformError) as excinfo:
        pin_nurec_usdz_layer_transform_to_identity(asset)
    assert "nurec_usdz_layer_transform_not_identity_after_rewrite" in excinfo.value.errors


def test_a_translate_orient_stack_is_refused_by_name(tmp_path):
    """Editing one value cannot clear a composed op stack, so refuse it."""

    volume_layer = (
        '#usda 1.0\n(\n    defaultPrim = "World"\n    upAxis = "Z"\n)\n\n'
        'def Xform "World"\n{\n    def Volume "gauss"\n    {\n'
        "        custom rel field:density = </World/gauss/density_field>\n"
        "        custom bool omni:nurec:isNuRecVolume = 1\n"
        "        double3 xformOp:translate = (1, 2, 3)\n"
        "        quatd xformOp:orient = (1, 0, 0, 0)\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n\n'
        '        def OmniNuRecFieldAsset "density_field"\n        {\n'
        '            custom asset filePath = @./scene.nurec@\n        }\n'
        "    }\n}\n"
    )
    asset = _write_usdz(
        tmp_path / "repaired_scene.usdz",
        [
            (
                "default.usda",
                '#usda 1.0\n(\n    defaultPrim = "World"\n)\n\n'
                'def Xform "World"\n{\n    over "gauss" (\n'
                "        prepend references = @gauss.usda@\n    )\n    {\n    }\n}\n",
            ),
            ("gauss.usda", volume_layer),
        ],
        room_positions(),
    )
    with pytest.raises(NuRecUsdzLayerTransformError) as excinfo:
        pin_nurec_usdz_layer_transform_to_identity(asset)
    assert any(
        error.startswith("nurec_usdz_layer_xform_op_unsupported")
        or error.startswith("nurec_usdz_layer_xform_op_order_unsupported")
        for error in excinfo.value.errors
    )


def test_a_missing_asset_fails_closed(tmp_path):
    with pytest.raises(NuRecUsdzLayerTransformError) as excinfo:
        pin_nurec_usdz_layer_transform_to_identity(tmp_path / "absent.usdz")
    assert "nurec_usdz_layer_asset_missing" in excinfo.value.errors


def _write_usdz(path: Path, layers, positions: np.ndarray) -> Path:
    members = [(name, text.encode("utf-8")) for name, text in layers]
    members.append(("scene.nurec", nurec_payload(positions)))
    with path.open("wb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in members:
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_STORED
                header = 30 + len(name.encode("utf-8"))
                padding = (-(handle.tell() + header)) % 64
                if padding:
                    if padding < 4:
                        padding += 64
                    info.extra = struct.pack("<hh", 0x1986, padding - 4) + b"\0" * (
                        padding - 4
                    )
                archive.writestr(info, data)
    return path
