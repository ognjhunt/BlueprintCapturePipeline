"""Hermetic contract tests for the NuRec appearance frame-alignment gate.

Every fixture here is authored in-process: a real gzip+MessagePack ``.nurec``
container inside a real USDZ, read back through pxr and the shipped codec.  No
GPU, no Isaac, no network.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.native_task_appearance_frame_alignment import (
    NativeTaskAppearanceFrameAlignmentError,
    measure_native_task_appearance_frame,
    qualify_native_task_appearance_frame_alignment,
    require_native_task_appearance_frame_alignment,
)
from blueprint_pipeline.nurec_volume_codec import (
    build_state_dict,
    encode_nurec_bytes,
)


IDENTITY = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
# The matrix the pinned upstream 3DGRUT USDZ exporter bakes in: p -> (-x, -z, -y).
EXPORTER_AXIS_MATRIX = ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1))

_ROOT_LAYER = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    over "gauss" (
        prepend references = @gauss.usda@
    )
    {
    }
}
"""

_VOLUME_LAYER = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Volume "gauss"
    {{
        custom rel field:density = </World/gauss/density_field>
        custom rel field:emissiveColor = </World/gauss/emissive_color_field>
        custom bool omni:nurec:isNuRecVolume = 1
        matrix4d xformOp:transform = {matrix}
        uniform token[] xformOpOrder = ["xformOp:transform"]

        def OmniNuRecFieldAsset "density_field"
        {{
            custom token fieldRole = "density"
            custom asset filePath = @./{payload}@
        }}

        def OmniNuRecFieldAsset "emissive_color_field"
        {{
            custom token fieldRole = "emissiveColor"
            custom asset filePath = @./{payload}@
        }}
    }}
}}
"""


def _matrix_literal(rows) -> str:
    return "( " + ", ".join(
        "(" + ", ".join(str(float(value)) for value in row) + ")" for row in rows
    ) + " )"


def nurec_payload(positions: np.ndarray) -> bytes:
    """Encode a valid NuRec container carrying the given gaussian centres."""

    count = int(positions.shape[0])
    arrays = {
        "positions": np.asarray(positions, dtype=np.float32),
        "rotations": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], np.float32), (count, 1)),
        "scales": np.full((count, 3), -3.0, np.float32),
        "densities": np.full((count, 1), 2.0, np.float32),
        "features_albedo": np.full((count, 3), 0.5, np.float32),
        "features_specular": np.zeros((count, 45), np.float32),
    }
    return encode_nurec_bytes(
        {
            "version": "0.2.576",
            "model": "nre",
            "config": {"layers": {"gaussians": {"precision": 16}}},
            "state_dict": build_state_dict(arrays, precision=16),
        }
    )


def write_appearance_usdz(
    path: Path,
    positions: np.ndarray,
    *,
    matrix=IDENTITY,
    payload_name: str = "scene.nurec",
    volume_marker: bool = True,
    default_prim: bool = True,
    payload_first: bool = False,
) -> Path:
    """Author a referenceable USDZ wrapping the given gaussian centres."""

    volume_layer = _VOLUME_LAYER.format(
        matrix=_matrix_literal(matrix), payload=payload_name
    )
    if not volume_marker:
        volume_layer = volume_layer.replace(
            "custom bool omni:nurec:isNuRecVolume = 1",
            "custom bool omni:nurec:isNuRecVolume = 0",
        )
    root_layer = _ROOT_LAYER
    if not default_prim:
        root_layer = root_layer.replace('    defaultPrim = "World"\n', "")
    members = [
        ("default.usda", root_layer.encode("utf-8")),
        ("gauss.usda", volume_layer.encode("utf-8")),
        (payload_name, nurec_payload(positions)),
    ]
    if payload_first:
        # The order the real 3DGRUT export writes, and the one three separate
        # archive validators assert.  Kept as an option rather than the default
        # so the existing fixtures stay byte-stable.
        members = [members[0], members[2], members[1]]
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


def room_positions(seed: int = 20260819, count: int = 4096) -> np.ndarray:
    """A metric Z-up room spanning roughly x 2..14, y 0.5..10.5, z 0..2.8."""

    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(2.0, 14.0, count),
            rng.uniform(0.5, 10.5, count),
            rng.uniform(0.0, 2.8, count),
        ],
        axis=1,
    ).astype(np.float32)


INSIDE = {"task_object": [3.5, 9.7, 0.0], "robot_base": [3.5, 9.2, 0.09]}


def test_metric_tensor_with_identity_transform_qualifies(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions(), matrix=IDENTITY
    )
    receipt = require_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )
    assert receipt["status"] == "aligned"
    assert receipt["blockers"] == []
    assert receipt["layer_transform_is_identity"] is True
    assert receipt["layer_transform_is_spurious"] is False
    assert receipt["volume_prim_path"] == "/World/gauss/gauss"
    assert receipt["spawned_frame_contains"] == {
        "robot_base": True,
        "task_object": True,
    }
    assert receipt["measurement_is_not_render_evidence"] is True


def test_particlefield_positions_use_the_same_containment_gate(tmp_path) -> None:
    from pxr import Sdf, Usd, UsdGeom, Vt

    asset = tmp_path / "scene_appearance.usdc"
    positions = room_positions()
    stage = Usd.Stage.CreateNew(str(asset))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    field = stage.DefinePrim("/World/gauss/gauss", "ParticleField3DGaussianSplat")
    field.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(positions, dtype=np.float32))
    )
    stage.GetRootLayer().Save()

    receipt = require_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )

    assert receipt["status"] == "aligned"
    assert receipt["representation"] == "particlefield_3d_gaussian_splat"
    assert receipt["appearance_prim_path"] == "/World/gauss/gauss"
    assert receipt["gaussian_count"] == positions.shape[0]
    assert receipt["measurement_authority"] == "particlefield_position_quantiles"


def test_exporter_axis_matrix_on_metric_tensor_is_refused_as_spurious(tmp_path):
    """The r23 defect: a metric tensor plus the exporter's fixed axis matrix.

    The room composes at negative x and below the floor, so every camera in the
    metric scene sees empty space while the task object stays in frame.
    """

    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz",
        room_positions(),
        matrix=EXPORTER_AXIS_MATRIX,
    )
    receipt = qualify_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )
    assert receipt["status"] == "misaligned"
    assert receipt["blockers"] == [
        "native_task_appearance_frame_excludes_scene_position:robot_base",
        "native_task_appearance_frame_excludes_scene_position:task_object",
        "native_task_appearance_layer_transform_spurious",
    ]
    assert receipt["layer_transform_is_spurious"] is True
    # The measurement names which frame the stored tensor is really in.
    assert receipt["layer_transform_omitted_frame_contains"] == {
        "robot_base": True,
        "task_object": True,
    }
    spawned = receipt["spawned_world_occupied_bounds_m"]
    assert spawned["maximum"][0] < 0.0
    assert spawned["maximum"][2] < 0.0

    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        require_native_task_appearance_frame_alignment(
            asset, required_world_positions_m=INSIDE
        )
    assert "native_task_appearance_layer_transform_spurious" in excinfo.value.errors


def test_axis_matrix_is_not_called_spurious_when_it_is_the_correct_mapping(tmp_path):
    """A tensor genuinely in the exporter's internal frame must still qualify.

    The gate must not learn "this matrix is always wrong": applied to a tensor
    the matrix really does map to world, it is required, not spurious.
    """

    metric = room_positions()
    internal = np.stack([-metric[:, 0], -metric[:, 2], -metric[:, 1]], axis=1)
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", internal, matrix=EXPORTER_AXIS_MATRIX
    )
    receipt = require_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )
    assert receipt["status"] == "aligned"
    assert receipt["layer_transform_is_identity"] is False
    assert receipt["layer_transform_is_spurious"] is False


def test_translated_spawn_pose_is_applied_on_top_of_the_layer_transform(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions(), matrix=IDENTITY
    )
    receipt = qualify_native_task_appearance_frame_alignment(
        asset,
        required_world_positions_m=INSIDE,
        spawn_position_world_m=(40.0, 0.0, 0.0),
    )
    assert receipt["status"] == "misaligned"
    assert receipt["layer_transform_is_spurious"] is False
    assert receipt["spawned_world_occupied_bounds_m"]["minimum"][0] > 40.0


def test_spawn_orientation_rotates_the_measured_bounds(tmp_path):
    """A 180 degree yaw is measured as one, in the plan's declared xyzw order."""

    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions(), matrix=IDENTITY
    )
    receipt = measure_native_task_appearance_frame(
        asset, spawn_orientation_xyzw=(0.0, 0.0, 1.0, 0.0)
    )
    spawned = receipt["spawned_world_occupied_bounds_m"]
    stored = receipt["stored_tensor_occupied_bounds_m"]
    assert spawned["maximum"][0] == pytest.approx(-stored["minimum"][0], abs=1e-6)
    assert spawned["minimum"][1] == pytest.approx(-stored["maximum"][1], abs=1e-6)
    assert spawned["maximum"][2] == pytest.approx(stored["maximum"][2], abs=1e-6)


def test_floater_gaussians_do_not_widen_the_occupied_bounds(tmp_path):
    """The shipped volume's raw box is 1.2 km wide around an 11 m room.

    Bounds taken from the extremes would contain every point ever tested, so
    the gate would pass a volume placed anywhere at all.
    """

    positions = room_positions()
    positions[:8] = np.array([[-900.0, -260.0, -1300.0]], np.float32)
    positions[8:16] = np.array([[340.0, 430.0, 900.0]], np.float32)
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", positions, matrix=EXPORTER_AXIS_MATRIX
    )
    receipt = qualify_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )
    assert receipt["status"] == "misaligned"
    bounds = receipt["stored_tensor_occupied_bounds_m"]
    assert bounds["minimum"][0] > 0.0
    assert bounds["maximum"][2] < 10.0


def test_missing_default_prim_is_refused(tmp_path):
    """A reference into a layer with no default prim composes into nothing."""

    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions(), default_prim=False
    )
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        measure_native_task_appearance_frame(asset)
    assert excinfo.value.errors == ("native_task_appearance_default_prim_missing",)


def test_asset_without_a_nurec_volume_is_refused(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions(), volume_marker=False
    )
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        measure_native_task_appearance_frame(asset)
    assert excinfo.value.errors == ("native_task_appearance_nurec_volume_not_exact",)


def test_missing_payload_member_is_refused(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions()
    )
    stripped = tmp_path / "stripped.usdz"
    with zipfile.ZipFile(asset) as source, zipfile.ZipFile(
        stripped, "w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            if info.filename.endswith(".nurec"):
                continue
            target.writestr(info.filename, source.read(info.filename))
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        measure_native_task_appearance_frame(stripped)
    assert excinfo.value.errors == ("native_task_appearance_nurec_payload_missing",)


def test_undecodable_payload_is_refused(tmp_path):
    asset = tmp_path / "scene_appearance.usdz"
    write_appearance_usdz(asset, room_positions())
    corrupt = tmp_path / "corrupt.usdz"
    with zipfile.ZipFile(asset) as source, zipfile.ZipFile(
        corrupt, "w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".nurec"):
                data = b"not a nurec container"
            target.writestr(info.filename, data)
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        measure_native_task_appearance_frame(corrupt)
    assert excinfo.value.errors == ("native_task_appearance_nurec_payload_undecodable",)


def test_missing_asset_is_refused(tmp_path):
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        measure_native_task_appearance_frame(tmp_path / "absent.usdz")
    assert excinfo.value.errors == ("native_task_appearance_asset_missing",)


def test_required_positions_cannot_be_empty(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions()
    )
    with pytest.raises(NativeTaskAppearanceFrameAlignmentError) as excinfo:
        qualify_native_task_appearance_frame_alignment(
            asset, required_world_positions_m={}
        )
    assert excinfo.value.errors == (
        "native_task_appearance_required_positions_missing",
    )


# The exporter's matrix is judged on its linear part, independently of whether
# containment happened to survive.  These two cases are the reason: one is the
# defect hiding inside a passing containment check, the other is a legitimate
# non-identity transform that must not be called spurious.


def symmetric_room_positions(seed: int = 20260820, count: int = 4096) -> np.ndarray:
    """A room centred on the axis matrix's fixed point.

    ``p -> (-x, -z, -y)`` maps this box onto itself, so a mirrored, upside-down
    volume still contains every position a plan asks about.
    """

    rng = np.random.default_rng(seed)
    return rng.uniform(-5.0, 5.0, (count, 3)).astype(np.float32)


def test_a_mirrored_volume_that_still_contains_the_task_is_refused(tmp_path):
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz",
        symmetric_room_positions(),
        matrix=EXPORTER_AXIS_MATRIX,
    )
    required = {"task_object": [1.0, 2.0, 0.5], "robot_base": [1.0, 1.5, 0.09]}
    receipt = qualify_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=required
    )

    # Containment alone is satisfied both ways -- which is exactly why it is
    # not enough on its own.
    assert all(receipt["spawned_frame_contains"].values())
    assert all(receipt["layer_transform_omitted_frame_contains"].values())
    assert receipt["layer_transform_linear_is_identity"] is False
    assert receipt["layer_transform_is_spurious"] is True
    assert receipt["status"] == "misaligned"
    assert receipt["blockers"] == ["native_task_appearance_layer_transform_spurious"]

    with pytest.raises(NativeTaskAppearanceFrameAlignmentError):
        require_native_task_appearance_frame_alignment(
            asset, required_world_positions_m=required
        )


def test_a_translation_only_transform_is_not_spurious(tmp_path):
    """``aura_nurec_usdz`` authors one, from the recentre offset it applied."""

    translation = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.25, -0.4, 0.1, 1))
    asset = write_appearance_usdz(
        tmp_path / "scene_appearance.usdz", room_positions(), matrix=translation
    )
    receipt = qualify_native_task_appearance_frame_alignment(
        asset, required_world_positions_m=INSIDE
    )

    assert receipt["layer_transform_is_identity"] is False
    assert receipt["layer_transform_linear_is_identity"] is True
    assert receipt["layer_transform_is_spurious"] is False
    assert receipt["status"] == "aligned"
    assert receipt["blockers"] == []
