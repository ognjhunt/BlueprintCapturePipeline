"""Prove a captured appearance volume lands where the scene it depicts is.

A NuRec appearance asset carries its gaussians in one frame and an
``xformOp:transform`` that maps that frame to world.  Which frame the stored
positions are already in is a property of the *trainer*, not of the format, so
the mapping matrix is only correct for the exporter that authored it.

``aura_nurec_usdz`` states the same fact from the other side: it pins an
identity transform precisely because its positions are already in the admitted
world frame, and copying the shipped package's mirroring matrix "would mirror
and rotate the room while looking entirely plausible".

The pinned upstream 3DGRUT USDZ exporter bakes a fixed axis-convention matrix
unconditionally.  When the trained tensor is already in the capture's metric
Z-up frame -- which is what happens when the reconstruction is fit to metric
capture poses -- that matrix is a spurious rigid motion.  The volume then
composes into the stage tens of metres from the room it depicts, every camera
sees empty space, and every existing gate still passes: the camera
observability gate measures the task object's semantic-segmentation pixels, and
a task object that is genuinely in frame reports ``passed`` against a render
whose captured scene is absent.

So this module measures, on CPU and without a renderer, where the volume's
occupied region actually lands once the layer transform and the planned spawn
pose are applied, and refuses a plan whose appearance does not contain the task
the cameras are pointed at.  Refusing costs one plan build.  Not refusing costs
a paid GPU run that returns black frames and reports success.

Nothing here renders, and nothing here decides that a volume looks correct.
Containment is necessary, not sufficient.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .nurec_volume_codec import NuRecCodecError, decode_nurec_bytes, gaussian_arrays

SCHEMA_VERSION = "native_task_appearance_frame_alignment.v1"

# The attribute the shipped NuRec packages carry on the volume prim.  It is a
# custom attribute, so a stage that never loaded the NuRec schema still reports
# it -- which is exactly the case on the CPU control plane.
NUREC_VOLUME_MARKER = "omni:nurec:isNuRecVolume"

# Floaters are a normal artifact of gaussian training and they push the raw
# bounding box out by two orders of magnitude: the shipped washer volume spans
# 1.2 km on its widest axis while its occupied room is 11 m across.  A bound
# taken from the extremes therefore contains every point in the building and
# would pass any containment test ever written.  Trim symmetrically instead.
DEFAULT_OCCUPANCY_QUANTILE = 0.01

# How far outside the measured occupied box a spawn position may sit and still
# count as inside it.  A room's occupied gaussians stop at its walls, and a
# task object standing against a wall is legitimately at the boundary.
DEFAULT_CONTAINMENT_MARGIN_M = 0.5


class NativeTaskAppearanceFrameAlignmentError(ValueError):
    """Stable fail-closed appearance-frame errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _volume_prim_and_transform(asset_path: Path) -> tuple[str, list[list[float]], str]:
    """Locate the NuRec volume prim and its composed layer transform.

    Returns the prim path, the row-major 4x4 local-to-world matrix in USD's
    row-vector convention (``p' = p * M``), and the payload asset path the
    volume's fields point at.
    """

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_usd_runtime_unavailable"]
        ) from exc
    try:
        stage = Usd.Stage.Open(str(asset_path))
    except Exception as exc:  # noqa: BLE001 - pxr raises bare exceptions
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_asset_unreadable"]
        ) from exc
    if stage is None:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_asset_unreadable"]
        )
    # A reference into a layer with no default prim resolves to nothing, so the
    # asset would compose into an empty scene no matter how well aligned it is.
    if not stage.GetDefaultPrim():
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_default_prim_missing"]
        )
    volumes = [
        prim
        for prim in stage.Traverse()
        if bool(prim.GetAttribute(NUREC_VOLUME_MARKER).Get())
    ]
    if len(volumes) != 1:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_volume_not_exact"]
        )
    volume = volumes[0]
    payloads = {
        str(child.GetAttribute("filePath").Get().path)
        for child in volume.GetChildren()
        if child.GetAttribute("filePath").IsValid()
        and child.GetAttribute("filePath").Get() is not None
    }
    if len(payloads) != 1:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_payload_not_exact"]
        )
    xformable = UsdGeom.Xformable(volume)
    if not xformable:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_volume_not_transformable"]
        )
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rows = [[float(matrix[row][column]) for column in range(4)] for row in range(4)]
    return str(volume.GetPath()), rows, payloads.pop()


def _gaussian_positions(asset_path: Path, payload_asset_path: str) -> np.ndarray:
    """Read the volume payload's gaussian centres, in their stored frame."""

    name = Path(payload_asset_path.replace("\\", "/")).name
    if not name:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_payload_not_exact"]
        )
    if zipfile.is_zipfile(asset_path):
        with zipfile.ZipFile(asset_path) as archive:
            members = [
                info.filename
                for info in archive.infolist()
                if Path(info.filename).name == name
            ]
            if len(members) != 1:
                raise NativeTaskAppearanceFrameAlignmentError(
                    ["native_task_appearance_nurec_payload_missing"]
                )
            raw = archive.read(members[0])
    else:
        candidate = (asset_path.parent / name).resolve()
        if candidate.parent != asset_path.parent.resolve() or not candidate.is_file():
            raise NativeTaskAppearanceFrameAlignmentError(
                ["native_task_appearance_nurec_payload_missing"]
            )
        raw = candidate.read_bytes()
    try:
        positions = gaussian_arrays(decode_nurec_bytes(raw))["positions"]
    except (NuRecCodecError, KeyError, ValueError) as exc:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_payload_undecodable"]
        ) from exc
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not array.shape[0]:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_positions_invalid"]
        )
    if not np.isfinite(array).all():
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_nurec_positions_invalid"]
        )
    return array


def _occupied_bounds(
    positions: np.ndarray, *, quantile: float
) -> tuple[list[float], list[float]]:
    lower = np.quantile(positions, quantile, axis=0)
    upper = np.quantile(positions, 1.0 - quantile, axis=0)
    return [float(value) for value in lower], [float(value) for value in upper]


def _apply(matrix: Sequence[Sequence[float]], points: np.ndarray) -> np.ndarray:
    """Transform row vectors by a USD row-major matrix (``p' = p * M``)."""

    m = np.asarray(matrix, dtype=np.float64)
    return points @ m[:3, :3] + m[3, :3]


def _spawn_matrix(
    position_world_m: Sequence[float], orientation_xyzw: Sequence[float]
) -> list[list[float]]:
    """Build the prim-level transform Isaac Lab spawns the asset with.

    ``AssetBaseCfg.InitialStateCfg.rot`` is documented ``(x, y, z, w)`` in the
    pinned Isaac Lab, and reaches the stage through ``spawn_from_usd``'s
    ``orientation`` argument, documented the same way.  Building the matrix
    through ``Gf`` rather than by hand keeps this on USD's own composition
    convention instead of a second hand-rolled one.
    """

    from pxr import Gf

    try:
        x, y, z, w = (float(value) for value in orientation_xyzw)
        tx, ty, tz = (float(value) for value in position_world_m)
    except (TypeError, ValueError) as exc:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_spawn_pose_invalid"]
        ) from exc
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if not norm or not np.isfinite(norm):
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_spawn_pose_invalid"]
        )
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(w / norm, Gf.Vec3d(x / norm, y / norm, z / norm)))
    matrix.SetTranslateOnly(Gf.Vec3d(tx, ty, tz))
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _contains(
    lower: Sequence[float],
    upper: Sequence[float],
    point: Sequence[float],
    *,
    margin_m: float,
) -> bool:
    return all(
        float(lower[axis]) - margin_m <= float(point[axis]) <= float(upper[axis]) + margin_m
        for axis in range(3)
    )


def measure_native_task_appearance_frame(
    appearance_asset_path: str | Path,
    *,
    spawn_position_world_m: Sequence[float] = (0.0, 0.0, 0.0),
    spawn_orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    occupancy_quantile: float = DEFAULT_OCCUPANCY_QUANTILE,
) -> dict[str, Any]:
    """Report where a NuRec appearance volume's occupied region lands."""

    if not 0.0 <= float(occupancy_quantile) < 0.5:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_occupancy_quantile_invalid"]
        )
    path = Path(appearance_asset_path).expanduser()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_asset_missing"]
        )
    prim_path, layer_transform, payload = _volume_prim_and_transform(path)
    positions = _gaussian_positions(path, payload)
    spawn_transform = _spawn_matrix(spawn_position_world_m, spawn_orientation_xyzw)
    stored_lower, stored_upper = _occupied_bounds(
        positions, quantile=float(occupancy_quantile)
    )
    layer_lower, layer_upper = _occupied_bounds(
        _apply(layer_transform, positions), quantile=float(occupancy_quantile)
    )
    spawned = _apply(spawn_transform, _apply(layer_transform, positions))
    spawned_lower, spawned_upper = _occupied_bounds(
        spawned, quantile=float(occupancy_quantile)
    )
    # What the volume would occupy if the exporter's fixed axis matrix were not
    # applied.  This is not a proposal to drop it -- it is how a run says which
    # of the two frames the stored tensor is actually in.
    unmapped_lower, unmapped_upper = _occupied_bounds(
        _apply(spawn_transform, positions), quantile=float(occupancy_quantile)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "volume_prim_path": prim_path,
        "gaussian_count": int(positions.shape[0]),
        "occupancy_quantile": float(occupancy_quantile),
        "layer_transform_row_major": layer_transform,
        "layer_transform_is_identity": bool(
            np.allclose(np.asarray(layer_transform, dtype=np.float64), np.eye(4))
        ),
        # The rotate/mirror/scale part, separate from the translation.  A
        # translation re-places a volume that is otherwise in the world frame
        # -- ``aura_nurec_usdz`` authors exactly one, from the recentre offset
        # it applied -- while a non-identity linear part re-*orients* the room,
        # which only an exporter's frame convention ever wants.
        "layer_transform_linear_is_identity": bool(
            np.allclose(
                np.asarray(layer_transform, dtype=np.float64)[:3, :3], np.eye(3)
            )
        ),
        "spawn_transform_row_major": spawn_transform,
        "stored_tensor_occupied_bounds_m": {
            "minimum": stored_lower,
            "maximum": stored_upper,
        },
        "layer_frame_occupied_bounds_m": {
            "minimum": layer_lower,
            "maximum": layer_upper,
        },
        "spawned_world_occupied_bounds_m": {
            "minimum": spawned_lower,
            "maximum": spawned_upper,
        },
        "layer_transform_omitted_world_occupied_bounds_m": {
            "minimum": unmapped_lower,
            "maximum": unmapped_upper,
        },
        "measurement_authority": "nurec_gaussian_centre_quantiles",
        "measurement_is_not_render_evidence": True,
    }


def qualify_native_task_appearance_frame_alignment(
    appearance_asset_path: str | Path,
    *,
    required_world_positions_m: Mapping[str, Sequence[float]],
    spawn_position_world_m: Sequence[float] = (0.0, 0.0, 0.0),
    spawn_orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    occupancy_quantile: float = DEFAULT_OCCUPANCY_QUANTILE,
    containment_margin_m: float = DEFAULT_CONTAINMENT_MARGIN_M,
) -> dict[str, Any]:
    """Fail closed unless the appearance contains what the cameras look at.

    ``required_world_positions_m`` maps a stable name -- a semantic role, a
    camera role -- to a world position the run depends on being inside the
    captured scene.  Every one of them must fall inside the volume's spawned
    occupied bounds.
    """

    if not required_world_positions_m:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_required_positions_missing"]
        )
    if not float(containment_margin_m) >= 0.0:
        raise NativeTaskAppearanceFrameAlignmentError(
            ["native_task_appearance_containment_margin_invalid"]
        )
    measurement = measure_native_task_appearance_frame(
        appearance_asset_path,
        spawn_position_world_m=spawn_position_world_m,
        spawn_orientation_xyzw=spawn_orientation_xyzw,
        occupancy_quantile=occupancy_quantile,
    )
    spawned = measurement["spawned_world_occupied_bounds_m"]
    omitted = measurement["layer_transform_omitted_world_occupied_bounds_m"]
    containment: dict[str, bool] = {}
    omitted_containment: dict[str, bool] = {}
    for name in sorted(required_world_positions_m):
        point = required_world_positions_m[name]
        try:
            resolved = [float(value) for value in point]
        except (TypeError, ValueError) as exc:
            raise NativeTaskAppearanceFrameAlignmentError(
                [f"native_task_appearance_required_position_invalid:{name}"]
            ) from exc
        if len(resolved) != 3:
            raise NativeTaskAppearanceFrameAlignmentError(
                [f"native_task_appearance_required_position_invalid:{name}"]
            )
        containment[name] = _contains(
            spawned["minimum"],
            spawned["maximum"],
            resolved,
            margin_m=float(containment_margin_m),
        )
        omitted_containment[name] = _contains(
            omitted["minimum"],
            omitted["maximum"],
            resolved,
            margin_m=float(containment_margin_m),
        )
    outside = sorted(name for name, inside in containment.items() if not inside)
    blockers = [
        f"native_task_appearance_frame_excludes_scene_position:{name}"
        for name in outside
    ]
    # The exporter's fixed axis matrix maps a NuRec-internal frame to world.
    # When the stored tensor is already in the world frame that matrix is a
    # spurious rigid motion, and the same measurement that refuses the plan can
    # say so -- without which the next run relitigates the renderer instead.
    #
    # Judged on the linear part alone and independently of whether containment
    # happened to survive.  The first version of this rule only fired once a
    # required position had already fallen outside, which made it a commentary
    # on a refusal rather than a refusal: a volume whose occupied box is
    # roughly symmetric about the mapping's fixed point stays "inside" while
    # being mirrored and upside down, and would have shipped.  Containment is
    # a coincidence-tolerant test; the frame question is not.
    #
    # A tensor genuinely in the exporter's internal frame is still admitted --
    # dropping its matrix moves the room away from the scene, so
    # ``omitted_containment`` is false and nothing here fires.  A pure
    # translation is admitted too, because its linear part is identity.
    layer_transform_spurious = bool(
        not measurement["layer_transform_linear_is_identity"]
        and all(omitted_containment.values())
    )
    if layer_transform_spurious:
        blockers.append("native_task_appearance_layer_transform_spurious")
    receipt = {
        **measurement,
        "schema_version": SCHEMA_VERSION,
        "containment_margin_m": float(containment_margin_m),
        "required_world_positions_m": {
            name: [float(value) for value in required_world_positions_m[name]]
            for name in sorted(required_world_positions_m)
        },
        "spawned_frame_contains": containment,
        "layer_transform_omitted_frame_contains": omitted_containment,
        "layer_transform_is_spurious": layer_transform_spurious,
        "status": "aligned" if not blockers else "misaligned",
        "blockers": sorted(set(blockers)),
    }
    return receipt


def require_native_task_appearance_frame_alignment(
    appearance_asset_path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    """Qualify and raise, so a caller cannot seal a misaligned appearance."""

    receipt = qualify_native_task_appearance_frame_alignment(
        appearance_asset_path, **kwargs
    )
    if receipt["blockers"]:
        raise NativeTaskAppearanceFrameAlignmentError(receipt["blockers"])
    return receipt


__all__ = [
    "DEFAULT_CONTAINMENT_MARGIN_M",
    "DEFAULT_OCCUPANCY_QUANTILE",
    "NUREC_VOLUME_MARKER",
    "NativeTaskAppearanceFrameAlignmentError",
    "SCHEMA_VERSION",
    "measure_native_task_appearance_frame",
    "qualify_native_task_appearance_frame_alignment",
    "require_native_task_appearance_frame_alignment",
]
