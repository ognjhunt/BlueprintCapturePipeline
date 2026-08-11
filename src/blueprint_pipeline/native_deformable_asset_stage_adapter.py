"""OpenUSD implementation of the native deformable preparation stage seam.

The adapter is intentionally source-provider neutral.  It reconstructs one
inspected triangle surface and its bound render material into a clean stage,
bakes a frozen metric scale into points, and derives readback from the authored
USD and cooked PhysX arrays.  Isaac Lab imports remain lazy so the deterministic
geometry/material path is testable with ``usd-core`` alone.

Native execution authority is not established here.  The enclosing trusted
worker envelope remains responsible for proving that the pinned Isaac/PhysX
runtime executed these bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import sys
import tempfile
import weakref
from collections.abc import Callable, Mapping, Sequence
from collections import Counter
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any


_DIGEST_PREFIX = "sha256:"
_PROVIDER_SCHEMA_PREFIXES = ("omniphysics", "physxauto", "physxbase", "newton")
_PROVIDER_ATTRIBUTE_PREFIXES = (
    "omniphysics:",
    "lightwheelusd:",
    "physxauto",
    "physxbase",
    "newton:",
)
_BODY_SCHEMA_NAMES = {
    "PhysxCollisionAPI": "pxr.PhysxSchema.PhysxCollisionAPI",
    "PhysxDeformableAPI": "pxr.PhysxSchema.PhysxDeformableAPI",
    "PhysxDeformableBodyAPI": "pxr.PhysxSchema.PhysxDeformableBodyAPI",
}
_MATERIAL_SCHEMA_NAMES = {
    "PhysxDeformableBodyMaterialAPI": ("pxr.PhysxSchema.PhysxDeformableBodyMaterialAPI")
}
_COOKING_FIELDS = frozenset(
    {
        "collision_simplification",
        "collision_simplification_remeshing",
        "collision_simplification_remeshing_resolution",
        "collision_simplification_target_triangle_count",
        "collision_simplification_force_conforming",
        "simulation_hexahedral_resolution",
    }
)
_MAX_SOURCE_USD_BYTES = 64 * 1024 * 1024
_MAX_TEXTURE_BYTES = 64 * 1024 * 1024
_MAX_TEXTURE_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_POINTS = 1_000_000
_MAX_TRIANGLES = 2_000_000
_MAX_TET_POINTS = 2_000_000
_MAX_TETS = 4_000_000
_MAX_MATERIAL_PRIMS = 4096
_MAX_TEXTURES = 4096
_MAX_MATERIAL_PROPERTIES = 100_000
_MAX_MATERIAL_CONNECTIONS = 100_000
_MAX_STAGE_PRIMS = 100_000
_MAX_STAGE_PROPERTIES = 250_000
_MAX_PRIMVARS = 4096
_MAX_ARRAY_VALUES = 6_000_000
_MAX_PATH_CHARACTERS = 1024
_MAX_PATH_COMPONENTS = 32
_MAX_ABSOLUTE_COORDINATE_M = 1_000_000.0
_MIN_SCALE = 1.0e-6
_MAX_SCALE = 1.0e6
_MIN_TRANSFORM_DETERMINANT = 1.0e-12
_MIN_TET_VOLUME_M3 = 1.0e-18
_MAX_SIMULATION_TO_SURFACE_VOLUME_RELATIVE_ERROR = 0.25
_READ_CHUNK_BYTES = 1024 * 1024
_ALLOWED_STAGE_SCHEMA_NAMES = frozenset(
    {
        "MaterialBindingAPI",
        "NodeDefAPI",
        "PhysicsCollisionAPI",
        "PhysicsMassAPI",
        *_BODY_SCHEMA_NAMES,
        *_MATERIAL_SCHEMA_NAMES,
    }
)
_ALLOWED_MATERIAL_DESCENDANT_TYPES = frozenset({"Shader", "NodeGraph"})


class NativeDeformableAssetStageAdapterError(ValueError):
    """Stable adapter failure."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _adapter_boundary(error: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convert dependency and malformed-input failures into one stable boundary."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return function(*args, **kwargs)
            except NativeDeformableAssetStageAdapterError:
                raise
            except Exception as exc:
                raise NativeDeformableAssetStageAdapterError([error]) from exc

        return wrapped

    return decorate


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field, None) == getattr(right, field, None)
        for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _absolute_path_parts(path_value: object, *, error: str) -> tuple[str, tuple[str, ...], Path]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise NativeDeformableAssetStageAdapterError([error])
    display = os.path.abspath(os.path.expanduser(os.fspath(path_value)))
    if sys.platform == "darwin":
        if display == "/var" or display.startswith("/var/"):
            display = f"/private{display}"
        elif display == "/tmp" or display.startswith("/tmp/"):
            display = f"/private{display}"
    path = Path(display)
    if (
        not path.anchor
        or not path.name
        or len(path.parts) > _MAX_PATH_COMPONENTS
        or len(display) > _MAX_PATH_CHARACTERS
    ):
        raise NativeDeformableAssetStageAdapterError([error])
    return path.anchor, tuple(path.parts[1:]), path


def _read_regular_file_snapshot(
    path_value: object,
    *,
    maximum_size: int,
    error: str,
) -> tuple[bytes, Path]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_nofollow_unavailable"]
        )
    anchor, parts, display = _absolute_path_parts(path_value, error=error)
    descriptors: list[int] = []
    directory_flags = os.O_RDONLY | int(nofollow) | int(directory) | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | int(nofollow) | getattr(os, "O_CLOEXEC", 0)
    try:
        parent = os.open(anchor, directory_flags)
        descriptors.append(parent)
        for component in parts[:-1]:
            parent = os.open(component, directory_flags, dir_fd=parent)
            descriptors.append(parent)
        descriptor = os.open(parts[-1], file_flags, dir_fd=parent)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not (0 < before.st_size <= maximum_size):
            raise NativeDeformableAssetStageAdapterError([error])
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(content) != before.st_size or not _same_file_snapshot(before, after):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_changed_while_reading"]
            )
        return content, display
    except NativeDeformableAssetStageAdapterError:
        raise
    except OSError as exc:
        raise NativeDeformableAssetStageAdapterError([error]) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _safe_relative_path(value: object, *, error: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_PATH_CHARACTERS:
        raise NativeDeformableAssetStageAdapterError([error])
    if any(token in value for token in ("\\", "\x00", ":", "[", "]")):
        raise NativeDeformableAssetStageAdapterError([error])
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) > _MAX_PATH_COMPONENTS
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise NativeDeformableAssetStageAdapterError([error])
    return value


def _pxr() -> tuple[Any, ...]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    except ImportError as exc:  # pragma: no cover - exercised only in native packaging
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_openusd_unavailable"]
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def _listop_applied_items(value: object) -> tuple[object, ...]:
    getter = getattr(value, "GetAppliedItems", None)
    if getter is None:
        return ()
    return tuple(getter())


def _preflight_single_source_layer(layer: object, *, sdf: object) -> None:
    """Reject composition before ``Usd.Stage.Open`` can resolve another file."""

    if list(getattr(layer, "subLayerPaths", ())):
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_source_composition_forbidden"]
        )
    spec_count = 0

    def visit(path: object) -> bool:
        nonlocal spec_count
        spec_count += 1
        if spec_count > _MAX_STAGE_PROPERTIES:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_layer_resource_limit_exceeded"]
            )
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            return True
        for attribute_name in ("referenceList", "payloadList"):
            if _listop_applied_items(getattr(spec, attribute_name, None)):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_composition_forbidden"]
                )
        for key in spec.ListInfoKeys():
            if str(key).casefold().startswith("clip"):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_composition_forbidden"]
                )
        return True

    layer.Traverse(sdf.Path.absoluteRootPath, visit)


def _open_source_stage_snapshot(source_path_value: object) -> tuple[object, Path, str]:
    _, Sdf, Usd, _, _, _ = _pxr()
    content, source_path = _read_regular_file_snapshot(
        source_path_value,
        maximum_size=_MAX_SOURCE_USD_BYTES,
        error="native_deformable_stage_source_usd_invalid",
    )
    suffix = source_path.suffix.casefold()
    if suffix not in {".usd", ".usda", ".usdc"}:
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_source_usd_suffix_invalid"]
        )
    with tempfile.TemporaryDirectory(prefix="blueprint-native-source-") as temporary:
        snapshot_path = Path(temporary) / f"asset{suffix}"
        descriptor = os.open(
            snapshot_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(content):
                offset += os.write(descriptor, content[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        layer = Sdf.Layer.FindOrOpen(str(snapshot_path))
        if layer is None:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_usd_open_failed"]
            )
        _preflight_single_source_layer(layer, sdf=Sdf)
        source_stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_usd_open_failed"]
            )
    return source_stage, source_path, _sha256_bytes(content)


def _finite_vector3(value: Sequence[float], *, error: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise NativeDeformableAssetStageAdapterError([error])
    result: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise NativeDeformableAssetStageAdapterError([error])
        number = float(component)
        if not math.isfinite(number):
            raise NativeDeformableAssetStageAdapterError([error])
        result.append(number)
    return result[0], result[1], result[2]


def _positive_vector3(value: Sequence[float], *, error: str) -> tuple[float, float, float]:
    result = _finite_vector3(value, error=error)
    if any(not (_MIN_SCALE <= component <= _MAX_SCALE) for component in result):
        raise NativeDeformableAssetStageAdapterError([error])
    return result


def _sha256_bytes(content: bytes) -> str:
    return f"{_DIGEST_PREFIX}{hashlib.sha256(content).hexdigest()}"


def _topology_sha256(counts: Sequence[int], indices: Sequence[int]) -> str:
    payload = bytearray(b"blueprint-triangle-topology-v1\0")
    payload.extend(struct.pack("<Q", len(counts)))
    for count in counts:
        payload.extend(struct.pack("<q", int(count)))
    payload.extend(struct.pack("<Q", len(indices)))
    for index in indices:
        payload.extend(struct.pack("<q", int(index)))
    return _sha256_bytes(bytes(payload))


def _point_sha256(points: Sequence[Sequence[float]]) -> str:
    payload = bytearray(b"blueprint-point3f-array-v1\0")
    payload.extend(struct.pack("<Q", len(points)))
    for point in points:
        payload.extend(struct.pack("<fff", *(float(component) for component in point)))
    return _sha256_bytes(bytes(payload))


def _normalized_points(
    values: Sequence[Sequence[float]],
    *,
    maximum_count: int,
    error: str,
) -> list[tuple[float, float, float]]:
    if isinstance(values, (str, bytes)) or not values or len(values) > maximum_count:
        raise NativeDeformableAssetStageAdapterError([error])
    points: list[tuple[float, float, float]] = []
    for value in values:
        point = _finite_vector3(value, error=error)
        if any(abs(component) > _MAX_ABSOLUTE_COORDINATE_M for component in point):
            raise NativeDeformableAssetStageAdapterError([error])
        points.append(point)
    return points


def _triangle_mesh_measurements(
    points_value: Sequence[Sequence[float]],
    counts_value: Sequence[int],
    indices_value: Sequence[int],
) -> dict[str, Any]:
    points = _normalized_points(
        points_value,
        maximum_count=_MAX_POINTS,
        error="native_deformable_stage_surface_points_invalid",
    )
    if (
        isinstance(counts_value, (str, bytes))
        or isinstance(indices_value, (str, bytes))
        or not counts_value
        or len(counts_value) > _MAX_TRIANGLES
        or len(indices_value) != 3 * len(counts_value)
        or len(indices_value) > _MAX_ARRAY_VALUES
    ):
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_source_topology_invalid"]
        )
    counts: list[int] = []
    indices: list[int] = []
    for value in counts_value:
        if isinstance(value, bool) or not isinstance(value, int) or int(value) != 3:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_surface_not_triangulated"]
            )
        counts.append(3)
    for value in indices_value:
        if isinstance(value, bool) or not isinstance(value, int):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_topology_invalid"]
            )
        index = int(value)
        if index < 0 or index >= len(points):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_topology_invalid"]
            )
        indices.append(index)

    parents = list(range(len(points)))
    ranks = [0] * len(points)

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1

    edge_counts: Counter[tuple[int, int]] = Counter()
    edge_balance: Counter[tuple[int, int]] = Counter()
    contributions: list[tuple[int, float]] = []
    used_vertices: set[int] = set()
    for cursor in range(0, len(indices), 3):
        a_index, b_index, c_index = indices[cursor : cursor + 3]
        if len({a_index, b_index, c_index}) != 3:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_topology_invalid"]
            )
        a, b, c = points[a_index], points[b_index], points[c_index]
        ab = tuple(b[axis] - a[axis] for axis in range(3))
        ac = tuple(c[axis] - a[axis] for axis in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        area_squared = sum(component * component for component in cross)
        if not math.isfinite(area_squared) or area_squared <= 1.0e-30:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_surface_degenerate_triangle"]
            )
        contribution = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if not math.isfinite(contribution):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_surface_volume_invalid"]
            )
        contributions.append((a_index, contribution))
        used_vertices.update((a_index, b_index, c_index))
        union(a_index, b_index)
        union(b_index, c_index)
        for left, right in ((a_index, b_index), (b_index, c_index), (c_index, a_index)):
            edge = (min(left, right), max(left, right))
            edge_counts[edge] += 1
            edge_balance[edge] += 1 if left < right else -1
    if len(used_vertices) != len(points) or any(
        edge_counts[edge] != 2 or edge_balance[edge] != 0 for edge in edge_counts
    ):
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_surface_not_closed_or_oriented"]
        )
    component_signed_sums: dict[int, float] = {}
    for vertex, contribution in contributions:
        root = find(vertex)
        component_signed_sums[root] = component_signed_sums.get(root, 0.0) + contribution
    component_volumes = [abs(value) / 6.0 for value in component_signed_sums.values()]
    if not component_volumes or any(
        not math.isfinite(value) or value <= _MIN_TET_VOLUME_M3 for value in component_volumes
    ):
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_surface_volume_invalid"]
        )
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    dimensions = [maximum[axis] - minimum[axis] for axis in range(3)]
    if any(not math.isfinite(value) or value <= 0.0 for value in dimensions):
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_surface_dimensions_invalid"]
        )
    return {
        "points": points,
        "counts": counts,
        "indices": indices,
        "point_positions_sha256": _point_sha256(points),
        "face_topology_sha256": _topology_sha256(counts, indices),
        "dimensions_m": dimensions,
        "aabb_center_m": [(maximum[axis] + minimum[axis]) * 0.5 for axis in range(3)],
        "closed_volume_m3": sum(component_volumes),
    }


def _tet_topology_measurements(
    points_value: Sequence[Sequence[float]],
    indices_value: Sequence[int],
    *,
    label: str,
) -> dict[str, Any]:
    points = _normalized_points(
        points_value,
        maximum_count=_MAX_TET_POINTS,
        error=f"native_deformable_stage_{label}_points_invalid",
    )
    if (
        isinstance(indices_value, (str, bytes))
        or not indices_value
        or len(indices_value) % 4
        or len(indices_value) // 4 > _MAX_TETS
    ):
        raise NativeDeformableAssetStageAdapterError(
            [f"native_deformable_stage_{label}_topology_invalid"]
        )
    indices: list[int] = []
    for value in indices_value:
        if isinstance(value, bool) or not isinstance(value, int):
            raise NativeDeformableAssetStageAdapterError(
                [f"native_deformable_stage_{label}_topology_invalid"]
            )
        index = int(value)
        if index < 0 or index >= len(points):
            raise NativeDeformableAssetStageAdapterError(
                [f"native_deformable_stage_{label}_topology_invalid"]
            )
        indices.append(index)
    observed_tets: set[tuple[int, int, int, int]] = set()
    used_vertices: set[int] = set()
    total_volume = 0.0
    for cursor in range(0, len(indices), 4):
        tet = tuple(indices[cursor : cursor + 4])
        canonical = tuple(sorted(tet))
        if len(set(tet)) != 4 or canonical in observed_tets:
            raise NativeDeformableAssetStageAdapterError(
                [f"native_deformable_stage_{label}_topology_invalid"]
            )
        observed_tets.add(canonical)
        used_vertices.update(tet)
        a, b, c, d = (points[index] for index in tet)
        ab = tuple(b[axis] - a[axis] for axis in range(3))
        ac = tuple(c[axis] - a[axis] for axis in range(3))
        ad = tuple(d[axis] - a[axis] for axis in range(3))
        determinant = (
            ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
            - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
            + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
        )
        volume = abs(determinant) / 6.0
        if not math.isfinite(volume) or volume <= _MIN_TET_VOLUME_M3:
            raise NativeDeformableAssetStageAdapterError(
                [f"native_deformable_stage_{label}_topology_invalid"]
            )
        total_volume += volume
    if len(used_vertices) != len(points) or not math.isfinite(total_volume):
        raise NativeDeformableAssetStageAdapterError(
            [f"native_deformable_stage_{label}_topology_invalid"]
        )
    return {
        "node_count": len(points),
        "element_count": len(indices) // 4,
        "topology_sha256": _topology_sha256([4] * (len(indices) // 4), indices),
        "volume_m3": total_volume,
    }


def _matrix_is_finite(matrix: object) -> bool:
    try:
        return (
            all(
                math.isfinite(float(matrix[row][column])) for row in range(4) for column in range(4)
            )
            and all(abs(float(matrix[row][3])) <= 1.0e-12 for row in range(3))
            and abs(float(matrix[3][3]) - 1.0) <= 1.0e-12
        )
    except (IndexError, TypeError, ValueError):
        return False


def _xform_chain_is_time_varying(prim: object, *, usd_geom: object) -> bool:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        xformable = usd_geom.Xformable(current)
        if xformable:
            for operation in xformable.GetOrderedXformOps():
                attribute = operation.GetAttr()
                if attribute.ValueMightBeTimeVarying() or attribute.GetNumTimeSamples():
                    return True
        current = current.GetParent()
    return False


def _normal_cardinality_valid(
    *, interpolation: str, count: int, point_count: int, face_count: int, corner_count: int
) -> bool:
    expected = {
        "constant": 1,
        "uniform": face_count,
        "vertex": point_count,
        "varying": point_count,
        "faceVarying": corner_count,
    }.get(interpolation)
    return expected is not None and count == expected


def _jsonable_usd_value(value: object, *, sdf: object, nodes: list[int], depth: int = 0) -> Any:
    nodes[0] += 1
    if nodes[0] > _MAX_ARRAY_VALUES or depth > 64:
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_material_value_resource_limit_exceeded"]
        )
    if isinstance(value, sdf.AssetPath):
        return {"asset_path": str(value.path or "")}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_value_nonfinite"]
            )
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_usd_value(nested, sdf=sdf, nodes=nodes, depth=depth + 1)
            for key, nested in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > _MAX_ARRAY_VALUES:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_value_resource_limit_exceeded"]
            )
        return [
            _jsonable_usd_value(nested, sdf=sdf, nodes=nodes, depth=depth + 1) for nested in value
        ]
    return str(value)


def _contains_asset_path(value: object, *, sdf: object, depth: int = 0) -> bool:
    if depth > 64:
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_material_value_resource_limit_exceeded"]
        )
    if isinstance(value, sdf.AssetPath):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_asset_path(nested, sdf=sdf, depth=depth + 1) for nested in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > _MAX_ARRAY_VALUES:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_value_resource_limit_exceeded"]
            )
        return any(_contains_asset_path(nested, sdf=sdf, depth=depth + 1) for nested in value)
    return False


def _material_network_digest(stage: object, roots: Sequence[str]) -> str:
    _, Sdf, Usd, _, _, _ = _pxr()
    rows: list[dict[str, Any]] = []
    prim_count = 0
    property_count = 0
    value_nodes = [0]
    for root in sorted(set(roots)):
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_network_missing"]
            )
        for prim in Usd.PrimRange(root_prim):
            prim_count += 1
            if prim_count > _MAX_MATERIAL_PRIMS:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_material_prim_limit_exceeded"]
                )
            attributes: list[dict[str, Any]] = []
            relationships: list[dict[str, Any]] = []
            for attribute in prim.GetAttributes():
                property_count += 1
                if property_count > _MAX_MATERIAL_PROPERTIES:
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_material_property_limit_exceeded"]
                    )
                if attribute.ValueMightBeTimeVarying() or attribute.GetNumTimeSamples():
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_material_time_varying"]
                    )
                attributes.append(
                    {
                        "name": str(attribute.GetName()),
                        "type": str(attribute.GetTypeName()),
                        "value": _jsonable_usd_value(attribute.Get(), sdf=Sdf, nodes=value_nodes),
                        "connections": sorted(str(path) for path in attribute.GetConnections()),
                    }
                )
            for relationship in prim.GetRelationships():
                property_count += 1
                if property_count > _MAX_MATERIAL_PROPERTIES:
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_material_property_limit_exceeded"]
                    )
                relationships.append(
                    {
                        "name": str(relationship.GetName()),
                        "targets": sorted(str(path) for path in relationship.GetTargets()),
                    }
                )
            rows.append(
                {
                    "path": str(prim.GetPath()),
                    "type": str(prim.GetTypeName()),
                    "schemas": sorted(_schema_names(prim)),
                    "attributes": sorted(attributes, key=lambda row: row["name"]),
                    "relationships": sorted(relationships, key=lambda row: row["name"]),
                }
            )
    return _sha256_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def _camel_case(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def _path_below(path: Any, prefix: Any) -> bool:
    return path == prefix or path.HasPrefix(prefix)


def _replace_prefix(path: Any, mappings: Sequence[tuple[Any, Any]]) -> Any:
    for _source, destination in mappings:
        if _path_below(path.GetPrimPath(), destination):
            return path
    for source, destination in sorted(mappings, key=lambda row: len(str(row[0])), reverse=True):
        if _path_below(path.GetPrimPath(), source):
            return path.ReplacePrefix(source, destination)
    raise NativeDeformableAssetStageAdapterError(
        ["native_deformable_stage_material_connection_outside_allowlist"]
    )


def _asset_relative_key(asset: Any, source_textures: Mapping[str, Path]) -> str:
    authored = str(getattr(asset, "path", "") or "")
    if authored.startswith("./"):
        authored = authored[2:]
    try:
        normalized = PurePosixPath(
            _safe_relative_path(
                authored,
                error="native_deformable_stage_material_asset_outside_allowlist",
            )
        )
    except NativeDeformableAssetStageAdapterError as exc:
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_material_asset_outside_allowlist"]
        ) from exc
    for relative in source_textures:
        candidate = PurePosixPath(relative)
        if normalized == candidate or normalized == PurePosixPath("textures") / candidate:
            return relative
    raise NativeDeformableAssetStageAdapterError(
        ["native_deformable_stage_material_asset_outside_allowlist"]
    )


def _schema_names(prim: object) -> set[str]:
    names = {str(schema) for schema in prim.GetAppliedSchemas()}
    authored = prim.GetMetadata("apiSchemas")
    if authored is not None and hasattr(authored, "GetAppliedItems"):
        names.update(str(schema) for schema in authored.GetAppliedItems())
    return names


def _registered_physx_schema_names(
    prim: object,
    expected: Mapping[str, str],
    *,
    error: str,
) -> list[str]:
    """Production-default proof that schema tokens resolve to registered APIs."""

    try:
        from pxr import PhysxSchema
    except ImportError as exc:  # pragma: no cover - native runtime owns this path
        raise NativeDeformableAssetStageAdapterError(
            ["native_deformable_stage_physx_schema_runtime_unavailable"]
        ) from exc
    observed: list[str] = []
    for token, qualified_name in expected.items():
        schema_type = getattr(PhysxSchema, token, None)
        if schema_type is None:
            raise NativeDeformableAssetStageAdapterError([error])
        try:
            valid = bool(prim.HasAPI(schema_type)) and bool(schema_type(prim))
        except Exception as exc:
            raise NativeDeformableAssetStageAdapterError([error]) from exc
        if not valid:
            raise NativeDeformableAssetStageAdapterError([error])
        observed.append(qualified_name)
    return sorted(observed)


class OpenUsdNativeDeformableStageAdapter:
    """Concrete clean-stage implementation for the preparation protocol."""

    def __init__(self, *, stage_factory: Callable[[], object] | None = None):
        self._stage_factory = stage_factory
        self._state: weakref.WeakKeyDictionary[object, dict[str, Any]] = weakref.WeakKeyDictionary()

    def _entry(self, stage: object) -> dict[str, Any]:
        entry = self._state.get(stage)
        if entry is None:
            raise NativeDeformableAssetStageAdapterError(["native_deformable_stage_unknown_stage"])
        return entry

    @_adapter_boundary("native_deformable_stage_create_failed")
    def create_clean_stage(
        self,
        *,
        output_path: Path,
        default_prim_path: str,
        meters_per_unit: float,
        up_axis: str,
    ) -> object:
        _, Sdf, Usd, UsdGeom, _, _ = _pxr()
        _, _, output = _absolute_path_parts(
            output_path, error="native_deformable_stage_output_path_invalid"
        )
        if (
            output.suffix.casefold() not in {".usd", ".usda", ".usdc"}
            or not output.parent.is_dir()
            or output.parent.is_symlink()
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_output_path_invalid"]
            )
        if output.exists() or output.is_symlink():
            raise NativeDeformableAssetStageAdapterError(["native_deformable_stage_output_exists"])
        if not Sdf.Path(default_prim_path).IsAbsoluteRootOrPrimPath():
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_default_prim_path_invalid"]
            )
        if (
            isinstance(meters_per_unit, bool)
            or not isinstance(meters_per_unit, (int, float))
            or not math.isfinite(float(meters_per_unit))
            or float(meters_per_unit) != 1.0
            or up_axis != "Z"
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_metadata_invalid"]
            )
        if self._stage_factory is None:
            try:
                from isaaclab.sim.utils.stage import create_new_stage
            except ImportError as exc:  # pragma: no cover - native-only path
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_factory_unavailable"]
                ) from exc
            stage = create_new_stage()
        else:
            stage = self._stage_factory()
        if not isinstance(stage, Usd.Stage):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_factory_return_invalid"]
            )
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, default_prim_path).GetPrim()
        stage.SetDefaultPrim(root)
        entry = {
            "output_path": output,
            "default_prim_path": default_prim_path,
            "surface": None,
            "material": None,
            "physics_configuration": None,
            "current_stage_context": None,
        }
        self._state[stage] = entry
        return stage

    @_adapter_boundary("native_deformable_stage_surface_copy_failed")
    def copy_surface_mesh_baking_points(
        self,
        *,
        stage: object,
        source_usd_path: Path,
        source_prim_path: str,
        output_prim_path: str,
        source_world_bounds_center_m: Sequence[float],
        recenter_to_output_origin: bool,
        bake_scale_xyz: Sequence[float],
        flatten_source_xform: bool,
    ) -> None:
        Gf, Sdf, Usd, UsdGeom, _, Vt = _pxr()
        entry = self._entry(stage)
        if entry["surface"] is not None:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_surface_already_copied"]
            )
        if recenter_to_output_origin is not True or flatten_source_xform is not True:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_surface_transform_contract_invalid"]
            )
        center = _finite_vector3(
            source_world_bounds_center_m,
            error="native_deformable_stage_source_center_invalid",
        )
        scale = _positive_vector3(
            bake_scale_xyz, error="native_deformable_stage_bake_scale_invalid"
        )
        source_stage, source_path, source_file_sha256 = _open_source_stage_snapshot(source_usd_path)
        if (
            UsdGeom.GetStageUpAxis(source_stage) != UsdGeom.Tokens.z
            or float(UsdGeom.GetStageMetersPerUnit(source_stage)) != 1.0
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_axes_or_units_unsupported"]
            )
        source_sdf_path = Sdf.Path(source_prim_path)
        if not source_sdf_path.IsAbsoluteRootOrPrimPath():
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_surface_invalid"]
            )
        source_prim = source_stage.GetPrimAtPath(source_sdf_path)
        if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Mesh):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_surface_invalid"]
            )
        source_mesh = UsdGeom.Mesh(source_prim)
        points_attribute = source_mesh.GetPointsAttr()
        normals_attribute = source_mesh.GetNormalsAttr()
        if (
            points_attribute.ValueMightBeTimeVarying()
            or points_attribute.GetNumTimeSamples()
            or normals_attribute.ValueMightBeTimeVarying()
            or normals_attribute.GetNumTimeSamples()
            or _xform_chain_is_time_varying(source_prim, usd_geom=UsdGeom)
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_rest_state_time_varying"]
            )
        if list(source_mesh.GetHoleIndicesAttr().Get() or []):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_holes_forbidden"]
            )
        if (
            source_mesh.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none
        ) != UsdGeom.Tokens.none:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_subdivision_forbidden"]
            )
        orientation = source_mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded
        if orientation not in {UsdGeom.Tokens.rightHanded, UsdGeom.Tokens.leftHanded}:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_orientation_invalid"]
            )
        points = _normalized_points(
            list(points_attribute.Get() or []),
            maximum_count=_MAX_POINTS,
            error="native_deformable_stage_surface_points_invalid",
        )
        counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(source_prim)
        determinant = float(transform.GetDeterminant())
        if (
            not _matrix_is_finite(transform)
            or not math.isfinite(determinant)
            or determinant <= _MIN_TRANSFORM_DETERMINANT
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_transform_invalid"]
            )
        world_points = [transform.Transform(Gf.Vec3d(point)) for point in points]
        normalized_world_points = _normalized_points(
            world_points,
            maximum_count=_MAX_POINTS,
            error="native_deformable_stage_source_transform_invalid",
        )
        minimum = [min(float(point[axis]) for point in world_points) for axis in range(3)]
        maximum = [max(float(point[axis]) for point in world_points) for axis in range(3)]
        measured_center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
        if any(abs(measured_center[axis] - center[axis]) > 1.0e-9 for axis in range(3)):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_center_mismatch"]
            )
        baked = Vt.Vec3fArray(
            [
                Gf.Vec3f(*((float(point[axis]) - center[axis]) * scale[axis] for axis in range(3)))
                for point in normalized_world_points
            ]
        )
        baked_measurements = _triangle_mesh_measurements(baked, counts, indices)
        counts = baked_measurements["counts"]
        indices = baked_measurements["indices"]
        output_sdf_path = Sdf.Path(output_prim_path)
        default_sdf_path = Sdf.Path(entry["default_prim_path"])
        if (
            not output_sdf_path.IsAbsoluteRootOrPrimPath()
            or output_sdf_path == default_sdf_path
            or not output_sdf_path.HasPrefix(default_sdf_path)
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_output_surface_path_invalid"]
            )
        parent = output_sdf_path.GetParentPath()
        UsdGeom.Scope.Define(stage, parent)
        output_mesh = UsdGeom.Mesh.Define(stage, output_sdf_path)
        output_mesh.CreatePointsAttr(baked)
        output_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        output_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        output_mesh.CreateOrientationAttr(orientation)
        output_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        output_mesh.CreateDoubleSidedAttr(bool(source_mesh.GetDoubleSidedAttr().Get()))
        output_mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(baked))

        source_normals = source_mesh.GetNormalsAttr().Get()
        normal_digest: str | None = None
        normal_interpolation: str | None = None
        if source_normals:
            normalized_normals = _normalized_points(
                list(source_normals),
                maximum_count=_MAX_ARRAY_VALUES,
                error="native_deformable_stage_source_normal_invalid",
            )
            normal_interpolation = str(source_mesh.GetNormalsInterpolation())
            if not _normal_cardinality_valid(
                interpolation=normal_interpolation,
                count=len(normalized_normals),
                point_count=len(points),
                face_count=len(counts),
                corner_count=len(indices),
            ):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_normal_cardinality_invalid"]
                )
            normal_transform = transform.GetInverse().GetTranspose()
            transformed_normals = []
            for normal in normalized_normals:
                world_normal = normal_transform.TransformDir(Gf.Vec3d(normal))
                baked_normal = Gf.Vec3d(
                    *(float(world_normal[axis]) / scale[axis] for axis in range(3))
                )
                length = baked_normal.GetLength()
                if not math.isfinite(length) or length <= 0.0:
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_source_normal_invalid"]
                    )
                transformed_normals.append(Gf.Vec3f(baked_normal / length))
            output_mesh.CreateNormalsAttr(Vt.Vec3fArray(transformed_normals))
            output_mesh.SetNormalsInterpolation(normal_interpolation)
            normal_digest = _point_sha256(transformed_normals)

        source_primvars = UsdGeom.PrimvarsAPI(source_prim)
        output_primvars = UsdGeom.PrimvarsAPI(output_mesh.GetPrim())
        source_primvar_rows = source_primvars.GetPrimvars()
        if len(source_primvar_rows) > _MAX_PRIMVARS:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_primvar_limit_exceeded"]
            )
        for primvar in source_primvar_rows:
            name = str(primvar.GetPrimvarName())
            if not name or name in {"points", "normals"}:
                continue
            if primvar.GetAttr().ValueMightBeTimeVarying() or primvar.GetAttr().GetNumTimeSamples():
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_rest_state_time_varying"]
                )
            value = primvar.Get()
            if _contains_asset_path(value, sdf=Sdf):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_primvar_asset_forbidden"]
                )
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and len(value) > _MAX_ARRAY_VALUES
            ):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_primvar_limit_exceeded"]
                )
            primvar_indices = list(primvar.GetIndices() or []) if primvar.IsIndexed() else []
            if len(primvar_indices) > _MAX_ARRAY_VALUES:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_source_primvar_limit_exceeded"]
                )
            output_primvar = output_primvars.CreatePrimvar(
                name,
                primvar.GetTypeName(),
                primvar.GetInterpolation(),
                primvar.GetElementSize(),
            )
            if value is not None:
                output_primvar.Set(value)
            if primvar.IsIndexed():
                output_primvar.SetIndices(primvar_indices)

        entry["surface"] = {
            "source_stage": source_stage,
            "source_path": source_path,
            "source_file_sha256": source_file_sha256,
            "source_prim_path": source_prim_path,
            "output_prim_path": output_prim_path,
            "source_world_bounds_center_m": list(center),
            "bake_scale_xyz": list(scale),
            "counts": counts,
            "indices": indices,
            "source_face_topology_sha256": baked_measurements["face_topology_sha256"],
            "output_face_topology_sha256": baked_measurements["face_topology_sha256"],
            "point_positions_sha256": baked_measurements["point_positions_sha256"],
            "dimensions_m": baked_measurements["dimensions_m"],
            "aabb_center_m": baked_measurements["aabb_center_m"],
            "closed_volume_m3": baked_measurements["closed_volume_m3"],
            "normal_positions_sha256": normal_digest,
            "normal_interpolation": normal_interpolation,
            "orientation": str(orientation),
        }

    @_adapter_boundary("native_deformable_stage_material_copy_failed")
    def copy_bound_material_network(
        self,
        *,
        stage: object,
        source_usd_path: Path,
        material_prim_path_map: Mapping[str, str],
        output_looks_prim_path: str,
        output_visual_prim_path: str,
        source_texture_paths: Mapping[str, Path],
        output_texture_asset_paths: Mapping[str, str],
    ) -> None:
        _, Sdf, Usd, UsdGeom, UsdShade, _ = _pxr()
        entry = self._entry(stage)
        surface = entry.get("surface")
        if surface is None or surface["output_prim_path"] != output_visual_prim_path:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_surface_join_invalid"]
            )
        if entry["material"] is not None:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_already_copied"]
            )
        if (
            not material_prim_path_map
            or len(material_prim_path_map) > _MAX_MATERIAL_PRIMS
            or set(source_texture_paths) != set(output_texture_asset_paths)
            or len(source_texture_paths) > _MAX_TEXTURES
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_allowlist_invalid"]
            )
        _, _, source_path = _absolute_path_parts(
            source_usd_path, error="native_deformable_stage_material_source_mismatch"
        )
        if source_path != surface["source_path"]:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_source_mismatch"]
            )
        default_sdf = Sdf.Path(entry["default_prim_path"])
        looks_sdf = Sdf.Path(output_looks_prim_path)
        if (
            not looks_sdf.IsAbsoluteRootOrPrimPath()
            or looks_sdf == default_sdf
            or not looks_sdf.HasPrefix(default_sdf)
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_mapping_invalid"]
            )

        normalized_source_textures: dict[str, Path] = {}
        normalized_output_textures: dict[str, str] = {}
        texture_inventory: list[dict[str, Any]] = []
        total_texture_bytes = 0
        for relative_value, source_texture in sorted(source_texture_paths.items()):
            relative = _safe_relative_path(
                relative_value,
                error="native_deformable_stage_material_allowlist_invalid",
            )
            expected_output = (PurePosixPath("textures") / relative).as_posix()
            if output_texture_asset_paths.get(relative_value) != expected_output:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_output_texture_path_invalid"]
                )
            content, path = _read_regular_file_snapshot(
                source_texture,
                maximum_size=_MAX_TEXTURE_BYTES,
                error="native_deformable_stage_texture_invalid",
            )
            expected_source = source_path.parent / "textures" / Path(*PurePosixPath(relative).parts)
            if path != expected_source:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_texture_outside_source_root"]
                )
            total_texture_bytes += len(content)
            if total_texture_bytes > _MAX_TEXTURE_TOTAL_BYTES:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_texture_total_size_exceeded"]
                )
            normalized_source_textures[relative] = path
            normalized_output_textures[relative] = expected_output
            texture_inventory.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256_bytes(content),
                    "size_bytes": len(content),
                }
            )
        if len(set(normalized_output_textures.values())) != len(normalized_output_textures):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_output_texture_path_invalid"]
            )
        source_stage = surface["source_stage"]
        mappings: list[tuple[Any, Any]] = []
        output_material_paths: list[Any] = []
        for source_material, output_material in sorted(material_prim_path_map.items()):
            source_sdf = Sdf.Path(source_material)
            output_sdf = Sdf.Path(output_material)
            prim = source_stage.GetPrimAtPath(source_sdf)
            if (
                not prim.IsValid()
                or not prim.IsA(UsdShade.Material)
                or not source_sdf.IsAbsoluteRootOrPrimPath()
                or not output_sdf.IsAbsoluteRootOrPrimPath()
                or output_sdf == looks_sdf
                or not output_sdf.HasPrefix(looks_sdf)
            ):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_material_mapping_invalid"]
                )
            mappings.append((source_sdf, output_sdf))
            output_material_paths.append(output_sdf)
        if len({str(path) for path in output_material_paths}) != len(output_material_paths) or any(
            left != right and (_path_below(left, right) or _path_below(right, left))
            for index, left in enumerate(output_material_paths)
            for right in output_material_paths[index + 1 :]
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_mapping_invalid"]
            )
        source_connection_count = 0
        for source_sdf, _output_sdf in mappings:
            for source_network_prim in Usd.PrimRange(source_stage.GetPrimAtPath(source_sdf)):
                for source_attribute in source_network_prim.GetAttributes():
                    for connection in source_attribute.GetConnections():
                        source_connection_count += 1
                        if source_connection_count > _MAX_MATERIAL_CONNECTIONS or not any(
                            _path_below(connection.GetPrimPath(), allowed_source)
                            for allowed_source, _destination in mappings
                        ):
                            raise NativeDeformableAssetStageAdapterError(
                                ["native_deformable_stage_material_connection_outside_allowlist"]
                            )
                for source_relationship in source_network_prim.GetRelationships():
                    for target in source_relationship.GetTargets():
                        source_connection_count += 1
                        if source_connection_count > _MAX_MATERIAL_CONNECTIONS or not any(
                            _path_below(target.GetPrimPath(), allowed_source)
                            for allowed_source, _destination in mappings
                        ):
                            raise NativeDeformableAssetStageAdapterError(
                                ["native_deformable_stage_material_connection_outside_allowlist"]
                            )
        UsdGeom.Scope.Define(stage, output_looks_prim_path)
        flattened = source_stage.Flatten()
        for source_sdf, output_sdf in mappings:
            if not Sdf.CopySpec(flattened, source_sdf, stage.GetRootLayer(), output_sdf):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_material_copy_failed"]
                )

        # Rebind connections and relationships copied from the source namespace.
        material_prim_count = 0
        material_property_count = 0
        material_connection_count = 0
        for _source_sdf, output_sdf in mappings:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(output_sdf)):
                material_prim_count += 1
                if material_prim_count > _MAX_MATERIAL_PRIMS:
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_material_prim_limit_exceeded"]
                    )
                if any(
                    str(schema).casefold().startswith(_PROVIDER_SCHEMA_PREFIXES)
                    for schema in _schema_names(prim)
                ):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_material_schema_forbidden"]
                    )
                if _contains_asset_path(prim.GetAllMetadata(), sdf=Sdf):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_material_metadata_asset_forbidden"]
                    )
                for attribute in prim.GetAttributes():
                    material_property_count += 1
                    if material_property_count > _MAX_MATERIAL_PROPERTIES:
                        raise NativeDeformableAssetStageAdapterError(
                            ["native_deformable_stage_material_property_limit_exceeded"]
                        )
                    if _contains_asset_path(attribute.GetAllMetadata(), sdf=Sdf):
                        raise NativeDeformableAssetStageAdapterError(
                            ["native_deformable_stage_material_metadata_asset_forbidden"]
                        )
                    connections = attribute.GetConnections()
                    if connections:
                        material_connection_count += len(connections)
                        if material_connection_count > _MAX_MATERIAL_CONNECTIONS:
                            raise NativeDeformableAssetStageAdapterError(
                                ["native_deformable_stage_material_connection_limit_exceeded"]
                            )
                        rewritten_connections = [
                            _replace_prefix(path, mappings) for path in connections
                        ]
                        if any(
                            not stage.GetPropertyAtPath(path).IsValid()
                            for path in rewritten_connections
                        ):
                            raise NativeDeformableAssetStageAdapterError(
                                ["native_deformable_stage_material_connection_invalid"]
                            )
                        attribute.SetConnections(rewritten_connections)
                    value = attribute.Get()
                    if isinstance(value, Sdf.AssetPath):
                        key = _asset_relative_key(value, normalized_source_textures)
                        attribute.Set(Sdf.AssetPath(normalized_output_textures[key]))
                    elif (
                        value is not None
                        and attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray
                    ):
                        rewritten = []
                        for asset in value:
                            key = _asset_relative_key(asset, normalized_source_textures)
                            rewritten.append(Sdf.AssetPath(normalized_output_textures[key]))
                        attribute.Set(rewritten)
                for relationship in prim.GetRelationships():
                    material_property_count += 1
                    if material_property_count > _MAX_MATERIAL_PROPERTIES:
                        raise NativeDeformableAssetStageAdapterError(
                            ["native_deformable_stage_material_property_limit_exceeded"]
                        )
                    if _contains_asset_path(relationship.GetAllMetadata(), sdf=Sdf):
                        raise NativeDeformableAssetStageAdapterError(
                            ["native_deformable_stage_material_metadata_asset_forbidden"]
                        )
                    targets = relationship.GetTargets()
                    if targets:
                        material_connection_count += len(targets)
                        rewritten_targets = [_replace_prefix(path, mappings) for path in targets]
                        if any(
                            not stage.GetPrimAtPath(path.GetPrimPath()).IsValid()
                            for path in rewritten_targets
                        ):
                            raise NativeDeformableAssetStageAdapterError(
                                ["native_deformable_stage_material_connection_invalid"]
                            )
                        relationship.SetTargets(rewritten_targets)

        source_mesh = source_stage.GetPrimAtPath(surface["source_prim_path"])
        bound, _relationship = UsdShade.MaterialBindingAPI(source_mesh).ComputeBoundMaterial()
        if not bound or str(bound.GetPath()) not in material_prim_path_map:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_source_material_binding_invalid"]
            )
        output_material_path = material_prim_path_map[str(bound.GetPath())]
        output_mesh = stage.GetPrimAtPath(output_visual_prim_path)
        UsdShade.MaterialBindingAPI.Apply(output_mesh).Bind(
            UsdShade.Material(stage.GetPrimAtPath(output_material_path))
        )
        material_roots = sorted(material_prim_path_map.values())
        entry["material"] = {
            "output_looks_prim_path": output_looks_prim_path,
            "material_prim_paths": material_roots,
            "texture_asset_paths": sorted(normalized_output_textures.values()),
            "texture_inventory": texture_inventory,
            "network_digest": _material_network_digest(stage, material_roots),
        }

    @_adapter_boundary("native_deformable_stage_current_stage_activation_failed")
    def activate_and_verify_current_stage(self, *, stage: object) -> bool:
        """Hold the pinned Isaac current-stage context through native authoring."""

        entry = self._entry(stage)
        try:
            from isaaclab.sim.utils.stage import get_current_stage, use_stage
        except ImportError as exc:  # pragma: no cover - native-only path
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_current_stage_api_unavailable"]
            ) from exc
        context = None
        try:
            if get_current_stage() is not stage:
                context = use_stage(stage)
                context.__enter__()
                entry["current_stage_context"] = context
            if get_current_stage() is not stage:
                self.release_current_stage(stage=stage)
                return False
            return True
        except Exception:
            owned = entry.get("current_stage_context")
            if owned is not None:
                self.release_current_stage(stage=stage)
            elif context is not None:
                context.__exit__(*sys.exc_info())
            raise

    @_adapter_boundary("native_deformable_stage_physics_configuration_invalid")
    def record_native_configuration(
        self,
        *,
        stage: object,
        body_and_cooking_properties: Mapping[str, Any],
        material_properties: Mapping[str, Any],
    ) -> None:
        entry = self._entry(stage)
        if entry["physics_configuration"] is not None:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_physics_configuration_already_recorded"]
            )
        if (
            not isinstance(body_and_cooking_properties, Mapping)
            or not isinstance(material_properties, Mapping)
            or not body_and_cooking_properties
            or not material_properties
            or len(body_and_cooking_properties) > 128
            or len(material_properties) > 128
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_physics_configuration_invalid"]
            )
        for source in (body_and_cooking_properties, material_properties):
            for key, value in source.items():
                if not isinstance(key, str) or not key or len(key) > 128:
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_physics_configuration_invalid"]
                    )
                if isinstance(value, bool):
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_physics_configuration_invalid"]
                    )
        body = {
            str(key): value
            for key, value in body_and_cooking_properties.items()
            if key not in _COOKING_FIELDS
        }
        cooking = {
            str(key): value
            for key, value in body_and_cooking_properties.items()
            if key in _COOKING_FIELDS
        }
        entry["physics_configuration"] = {
            "body_properties": body,
            "cooking_properties": cooking,
            "material_properties": dict(material_properties),
        }

    @_adapter_boundary("native_deformable_stage_context_release_failed")
    def release_current_stage(self, *, stage: object) -> None:
        """Idempotently release a current-stage context after success or failure."""

        entry = self._entry(stage)
        context = entry.get("current_stage_context")
        if context is not None:
            context.__exit__(None, None, None)
            entry["current_stage_context"] = None

    @_adapter_boundary("native_deformable_stage_save_failed")
    def save_stage(self, *, stage: object) -> None:
        entry = self._entry(stage)
        temporary_path: Path | None = None
        try:
            if entry["surface"] is None or entry["material"] is None:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_save_incomplete"]
                )
            output = entry["output_path"]
            if output.exists() or output.is_symlink():
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_output_exists"]
                )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            if not stage.GetRootLayer().Export(str(temporary_path)):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_export_failed"]
                )
            with temporary_path.open("rb+") as stream:
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, output)
            except FileExistsError as exc:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_output_exists"]
                ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.release_current_stage(stage=stage)

    def _read_native_configuration(
        self, *, stage: object, schema_prim: object, physics_material_prim: object
    ) -> dict[str, Any]:
        entry = self._entry(stage)
        configuration = entry.get("physics_configuration")
        if not isinstance(configuration, dict):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_physics_configuration_missing"]
            )
        errors: list[str] = []
        actual_configuration: dict[str, dict[str, Any]] = {
            "body_properties": {},
            "cooking_properties": {},
            "material_properties": {},
        }

        def normalized_actual(attribute: object, expected: object, *, mismatch: str) -> Any:
            if not attribute or not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
                errors.append(mismatch)
                return None
            actual = attribute.Get()
            if isinstance(expected, bool):
                if not isinstance(actual, bool) or actual is not expected:
                    errors.append(mismatch)
                    return None
                return bool(actual)
            if isinstance(expected, int):
                if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                    errors.append(mismatch)
                    return None
                return int(actual)
            if isinstance(expected, float):
                if (
                    isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                    or not math.isfinite(float(actual))
                    or abs(float(actual) - expected) > 1.0e-8
                ):
                    errors.append(mismatch)
                    return None
                return float(actual)
            errors.append(mismatch)
            return None

        for group in ("body_properties", "cooking_properties"):
            for field, expected in configuration[group].items():
                namespace = (
                    "physxCollision"
                    if field in {"contact_offset", "rest_offset"}
                    else "physxDeformable"
                )
                attribute = schema_prim.GetAttribute(f"{namespace}:{_camel_case(field)}")
                mismatch = f"native_deformable_stage_body_readback_mismatch:{field}"
                actual_configuration[group][field] = normalized_actual(
                    attribute, expected, mismatch=mismatch
                )
        for field, expected in configuration["material_properties"].items():
            attribute = physics_material_prim.GetAttribute(
                f"physxDeformableBodyMaterial:{_camel_case(field)}"
            )
            mismatch = f"native_deformable_stage_material_readback_mismatch:{field}"
            actual_configuration["material_properties"][field] = normalized_actual(
                attribute, expected, mismatch=mismatch
            )
        if errors:
            raise NativeDeformableAssetStageAdapterError(errors)
        return actual_configuration

    @_adapter_boundary("native_deformable_stage_readback_failed")
    def readback_prepared_stage(
        self,
        *,
        stage: object,
        output_authoring_root_prim_path: str,
        output_deformable_schema_prim_path: str,
        output_visual_prim_path: str,
    ) -> Mapping[str, Any]:
        Gf, Sdf, Usd, UsdGeom, UsdShade, _ = _pxr()
        entry = self._entry(stage)
        surface = entry.get("surface")
        material = entry.get("material")
        if (
            surface is None
            or material is None
            or output_authoring_root_prim_path != entry["default_prim_path"]
            or output_deformable_schema_prim_path != output_visual_prim_path
            or output_visual_prim_path != surface["output_prim_path"]
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_readback_join_invalid"]
            )
        schema_prim = stage.GetPrimAtPath(output_deformable_schema_prim_path)
        physics_material_path = f"{output_authoring_root_prim_path}/PhysicsMaterial"
        physics_material_prim = stage.GetPrimAtPath(physics_material_path)
        if (
            not schema_prim.IsValid()
            or not schema_prim.IsA(UsdGeom.Mesh)
            or not physics_material_prim.IsValid()
            or not physics_material_prim.IsA(UsdShade.Material)
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_native_prims_missing"]
            )
        stage_metadata = {
            "default_prim_path": str(stage.GetDefaultPrim().GetPath()),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        }
        if stage_metadata != {
            "default_prim_path": entry["default_prim_path"],
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        }:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_metadata_readback_invalid"]
            )
        body_schemas = _registered_physx_schema_names(
            schema_prim,
            _BODY_SCHEMA_NAMES,
            error="native_deformable_stage_native_schema_readback_invalid",
        )
        material_schemas = _registered_physx_schema_names(
            physics_material_prim,
            _MATERIAL_SCHEMA_NAMES,
            error="native_deformable_stage_native_schema_readback_invalid",
        )

        live_mesh = UsdGeom.Mesh(schema_prim)
        if (
            live_mesh.GetPointsAttr().ValueMightBeTimeVarying()
            or live_mesh.GetPointsAttr().GetNumTimeSamples()
            or live_mesh.GetNormalsAttr().ValueMightBeTimeVarying()
            or live_mesh.GetNormalsAttr().GetNumTimeSamples()
            or _xform_chain_is_time_varying(schema_prim, usd_geom=UsdGeom)
            or list(live_mesh.GetHoleIndicesAttr().Get() or [])
            or (live_mesh.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none)
            != UsdGeom.Tokens.none
            or str(live_mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
            != surface["orientation"]
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_visual_mesh_readback_mismatch"]
            )
        live_transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            schema_prim
        )
        identity = Gf.Matrix4d(1.0)
        if not _matrix_is_finite(live_transform) or any(
            abs(float(live_transform[row][column]) - float(identity[row][column])) > 1.0e-12
            for row in range(4)
            for column in range(4)
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_visual_mesh_readback_mismatch"]
            )
        live_visual = _triangle_mesh_measurements(
            list(live_mesh.GetPointsAttr().Get() or []),
            list(live_mesh.GetFaceVertexCountsAttr().Get() or []),
            list(live_mesh.GetFaceVertexIndicesAttr().Get() or []),
        )
        if (
            live_visual["point_positions_sha256"] != surface["point_positions_sha256"]
            or live_visual["face_topology_sha256"] != surface["output_face_topology_sha256"]
            or any(
                abs(live_visual["dimensions_m"][axis] - surface["dimensions_m"][axis]) > 1.0e-8
                for axis in range(3)
            )
            or any(
                abs(live_visual["aabb_center_m"][axis] - surface["aabb_center_m"][axis]) > 1.0e-8
                for axis in range(3)
            )
            or abs(live_visual["closed_volume_m3"] - surface["closed_volume_m3"])
            > max(1.0e-12, surface["closed_volume_m3"] * 1.0e-8)
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_visual_mesh_readback_mismatch"]
            )
        live_normals = list(live_mesh.GetNormalsAttr().Get() or [])
        if surface["normal_positions_sha256"] is None:
            if live_normals:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_visual_normal_readback_mismatch"]
                )
        else:
            normalized_live_normals = _normalized_points(
                live_normals,
                maximum_count=_MAX_ARRAY_VALUES,
                error="native_deformable_stage_visual_normal_readback_mismatch",
            )
            if (
                str(live_mesh.GetNormalsInterpolation()) != surface["normal_interpolation"]
                or not _normal_cardinality_valid(
                    interpolation=str(live_mesh.GetNormalsInterpolation()),
                    count=len(normalized_live_normals),
                    point_count=len(live_visual["points"]),
                    face_count=len(live_visual["counts"]),
                    corner_count=len(live_visual["indices"]),
                )
                or _point_sha256(normalized_live_normals) != surface["normal_positions_sha256"]
            ):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_visual_normal_readback_mismatch"]
                )

        topology_attributes = {
            "simulation": (
                "physxDeformable:simulationPoints",
                "physxDeformable:simulationIndices",
            ),
            "collision": (
                "physxDeformable:collisionPoints",
                "physxDeformable:collisionIndices",
            ),
        }
        topology_measurements: dict[str, dict[str, Any]] = {}
        for label, (points_name, indices_name) in topology_attributes.items():
            points_attribute = schema_prim.GetAttribute(points_name)
            indices_attribute = schema_prim.GetAttribute(indices_name)
            if (
                not points_attribute
                or not indices_attribute
                or not points_attribute.HasAuthoredValueOpinion()
                or not indices_attribute.HasAuthoredValueOpinion()
                or points_attribute.ValueMightBeTimeVarying()
                or indices_attribute.ValueMightBeTimeVarying()
            ):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_cooked_topology_missing"]
                )
            topology_measurements[label] = _tet_topology_measurements(
                list(points_attribute.Get() or []),
                list(indices_attribute.Get() or []),
                label=label,
            )
        simulation_volume_error = (
            abs(topology_measurements["simulation"]["volume_m3"] - live_visual["closed_volume_m3"])
            / live_visual["closed_volume_m3"]
        )
        if simulation_volume_error > _MAX_SIMULATION_TO_SURFACE_VOLUME_RELATIVE_ERROR:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_simulation_volume_readback_mismatch"]
            )

        if (
            _material_network_digest(stage, material["material_prim_paths"])
            != material["network_digest"]
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_network_readback_mismatch"]
            )
        render_bound, render_relationship = UsdShade.MaterialBindingAPI(
            schema_prim
        ).ComputeBoundMaterial()
        physics_bound, physics_relationship = UsdShade.MaterialBindingAPI(
            schema_prim
        ).ComputeBoundMaterial("physics")
        strength = (
            physics_relationship.GetMetadata("bindMaterialAs") if physics_relationship else None
        )
        if (
            not render_bound
            or str(render_bound.GetPath()) not in material["material_prim_paths"]
            or not render_relationship
            or render_relationship.GetPrim() != schema_prim
            or not physics_bound
            or str(physics_bound.GetPath()) != physics_material_path
            or not physics_relationship
            or physics_relationship.GetPrim() != schema_prim
            or str(strength or "") != "strongerThanDescendants"
        ):
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_binding_readback_invalid"]
            )
        physics_configuration = self._read_native_configuration(
            stage=stage,
            schema_prim=schema_prim,
            physics_material_prim=physics_material_prim,
        )

        expected_root = stage.GetPrimAtPath(output_authoring_root_prim_path)
        expected_scopes: set[str] = {material["output_looks_prim_path"]}
        current_parent = schema_prim.GetParent()
        while current_parent.IsValid() and current_parent != expected_root:
            expected_scopes.add(str(current_parent.GetPath()))
            current_parent = current_parent.GetParent()
        material_roots = [stage.GetPrimAtPath(path) for path in material["material_prim_paths"]]
        prim_count = 0
        property_count = 0
        for prim in stage.TraverseAll():
            prim_count += 1
            if prim_count > _MAX_STAGE_PRIMS:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_prim_limit_exceeded"]
                )
            path = str(prim.GetPath())
            if not prim.GetPath().HasPrefix(expected_root.GetPath()):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_unexpected_prim_inventory"]
                )
            type_name = str(prim.GetTypeName())
            material_root = next(
                (root for root in material_roots if prim.GetPath().HasPrefix(root.GetPath())),
                None,
            )
            allowed_inventory = (
                (path == output_authoring_root_prim_path and type_name == "Xform")
                or (path in expected_scopes and type_name == "Scope")
                or (path == output_visual_prim_path and type_name == "Mesh")
                or (path == physics_material_path and type_name == "Material")
                or (
                    material_root is not None
                    and (
                        (prim == material_root and type_name == "Material")
                        or (
                            prim != material_root
                            and type_name in _ALLOWED_MATERIAL_DESCENDANT_TYPES
                        )
                    )
                )
            )
            if not allowed_inventory or type_name == "TetMesh":
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_unexpected_prim_inventory"]
                )
            if path == output_visual_prim_path:
                allowed_schemas = {
                    "MaterialBindingAPI",
                    "PhysicsCollisionAPI",
                    "PhysicsMassAPI",
                    *_BODY_SCHEMA_NAMES,
                }
            elif path == physics_material_path:
                allowed_schemas = set(_MATERIAL_SCHEMA_NAMES)
            elif material_root is not None:
                allowed_schemas = {"MaterialBindingAPI", "NodeDefAPI"}
            else:
                allowed_schemas = set()
            authored_schemas = _schema_names(prim)
            if not authored_schemas.issubset(allowed_schemas):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_forbidden_source_content_present"]
                )
            for schema in authored_schemas:
                schema_text = str(schema)
                if (
                    schema_text.casefold().startswith(_PROVIDER_SCHEMA_PREFIXES)
                    or schema_text.split(":", 1)[0] not in allowed_schemas
                ):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_forbidden_source_content_present"]
                    )
            if _contains_asset_path(prim.GetAllMetadata(), sdf=Sdf):
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_forbidden_source_content_present"]
                )
            for attribute in prim.GetAttributes():
                property_count += 1
                if property_count > _MAX_STAGE_PROPERTIES:
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_property_limit_exceeded"]
                    )
                if str(attribute.GetName()).casefold().startswith(_PROVIDER_ATTRIBUTE_PREFIXES):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_forbidden_source_content_present"]
                    )
                if (
                    (material_root is None and _contains_asset_path(attribute.Get(), sdf=Sdf))
                    or _contains_asset_path(attribute.GetAllMetadata(), sdf=Sdf)
                    or any(
                        not connection.GetPrimPath().HasPrefix(expected_root.GetPath())
                        or not stage.GetPropertyAtPath(connection).IsValid()
                        for connection in attribute.GetConnections()
                    )
                ):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_forbidden_source_content_present"]
                    )
            relationships = prim.GetRelationships()
            property_count += len(relationships)
            if property_count > _MAX_STAGE_PROPERTIES:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_property_limit_exceeded"]
                )
            for relationship in relationships:
                if _contains_asset_path(relationship.GetAllMetadata(), sdf=Sdf) or any(
                    not target.GetPrimPath().HasPrefix(expected_root.GetPath())
                    or not stage.GetPrimAtPath(target.GetPrimPath()).IsValid()
                    for target in relationship.GetTargets()
                ):
                    raise NativeDeformableAssetStageAdapterError(
                        ["native_deformable_stage_forbidden_source_content_present"]
                    )
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputePurpose() == UsdGeom.Tokens.guide:
                raise NativeDeformableAssetStageAdapterError(
                    ["native_deformable_stage_forbidden_source_content_present"]
                )
        density = float(physics_configuration["material_properties"]["density"])
        if not math.isfinite(density) or density <= 0.0:
            raise NativeDeformableAssetStageAdapterError(
                ["native_deformable_stage_material_readback_mismatch:density"]
            )
        result = {
            "stage_metadata": stage_metadata,
            "visual_mesh": {
                "prim_path": output_visual_prim_path,
                "point_count": len(live_visual["points"]),
                "triangle_count": len(live_visual["counts"]),
                "source_face_topology_sha256": surface["source_face_topology_sha256"],
                "output_face_topology_sha256": live_visual["face_topology_sha256"],
                "dimensions_m": live_visual["dimensions_m"],
                "authored_scale_xyz": [1.0, 1.0, 1.0],
                "metric_scale_baked_into_points": True,
                "source_xform_flattened": True,
                "source_world_bounds_center_m": surface["source_world_bounds_center_m"],
                "recentered_before_scale": True,
                "aabb_center_m": live_visual["aabb_center_m"],
                "authored_pivot_m": [0.0, 0.0, 0.0],
                "placement_origin_semantics": "body_pose_translation_is_replacement_aabb_center",
                "point_positions_sha256": live_visual["point_positions_sha256"],
                "closed_volume_m3": live_visual["closed_volume_m3"],
            },
            "authoring_root_prim_path": output_authoring_root_prim_path,
            "deformable_schema_prim_path": output_deformable_schema_prim_path,
            "body_api_schemas": body_schemas,
            "physics_material": {
                "prim_path": physics_material_path,
                "api_schemas": material_schemas,
                "properties": physics_configuration["material_properties"],
            },
            "mass_properties": {
                "density_kg_m3": density,
                "closed_volume_m3": live_visual["closed_volume_m3"],
                "derived_mass_kg": density * live_visual["closed_volume_m3"],
                "mass_tolerance_kg": max(
                    1.0e-12, density * live_visual["closed_volume_m3"] * 1.0e-6
                ),
                "development_configuration_not_observed_material_truth": True,
            },
            "physics_material_binding": {
                "prim_path": output_visual_prim_path,
                "material_prim_path": physics_material_path,
                "material_purpose": "physics",
                "binding_strength": str(strength),
            },
            "material_binding": {
                "visual_prim_path": output_visual_prim_path,
                "material_prim_paths": material["material_prim_paths"],
                "texture_asset_paths": material["texture_asset_paths"],
            },
            "simulation_topology": {
                key: topology_measurements["simulation"][key]
                for key in ("node_count", "element_count", "topology_sha256")
            },
            "collision_topology": {
                key: topology_measurements["collision"][key]
                for key in ("node_count", "element_count", "topology_sha256")
            },
            "physics_configuration": physics_configuration,
            "texture_inventory": material["texture_inventory"],
            "experimental_api_schemas": [],
            "empty_tet_mesh_prim_paths": [],
            "guide_prim_paths": [],
            "light_prim_paths": [],
            "source_provider_prim_paths": [],
            "source_provider_attributes": [],
        }
        self._state.pop(stage, None)
        return result


__all__ = [
    "NativeDeformableAssetStageAdapterError",
    "OpenUsdNativeDeformableStageAdapter",
]
