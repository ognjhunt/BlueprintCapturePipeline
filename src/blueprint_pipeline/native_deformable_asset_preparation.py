"""Prepare inspected surface assets for pinned native PhysX cooking.

This module deliberately separates three authorities:

* :func:`materialize_native_deformable_asset_preparation_plan` validates exact
  external source bytes and creates a deterministic, static preparation plan.
* :func:`build_native_deformable_asset_source_package` copies only those exact
  bytes into a portable input package.  It does not import USD, Isaac Lab, or
  PhysX and cannot claim a successful deformable cook.
* :func:`execute_native_deformable_asset_preparation` is a small native-worker
  seam.  Stage operations, typed configuration constructors, authoring, and
  physics-material binding are dependency injected so their orchestration is
  hermetically testable.  Its returned payload remains worker-authored evidence.
* :func:`verify_native_deformable_asset_preparation_return` verifies the
  payload and returned bytes, but intentionally leaves native execution
  authentication and simulator qualification to the enclosing trusted canary.

The output stage is rebuilt from an allowlist.  Provider-authored experimental
schemas, empty ``TetMesh`` placeholders, guides, and lights are evidence about
what must be excluded; none are copied into the clean output stage.  Metric
scale is baked into visual points and the output visual transform must read
back as identity.  Materials and referenced textures remain digest-bound.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import resource
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .decision_evidence_contracts import canonical_digest
from .external_simready_deformable_asset import (
    ExternalSimreadyDeformableAssetError,
)
from .external_simready_deformable_asset import (
    PENDING_STATUS as INSPECTION_STATUS,
)
from .external_simready_deformable_asset import (
    SCHEMA_VERSION as INSPECTION_SCHEMA_VERSION,
)
from .external_simready_deformable_asset import (
    verify_external_simready_deformable_asset_inspection,
)
from .native_task_entity_asset_authoring_bundle import (
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_COOKING_API,
)
from .native_task_runtime_source_packet import (
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)


PLAN_SCHEMA_VERSION = "native_deformable_asset_preparation_plan.v1"
PACKAGE_SCHEMA_VERSION = "native_deformable_asset_source_package.v1"
WORKER_RETURN_SCHEMA_VERSION = "native_deformable_asset_preparation_worker_return.v1"
RETURN_VERIFICATION_SCHEMA_VERSION = "native_deformable_asset_preparation_return_verification.v1"

PLAN_FILENAME = "native_deformable_asset_preparation_plan.v1.json"
PACKAGE_RECEIPT_FILENAME = "native_deformable_asset_source_package.v1.json"
SOURCE_USD_PACKAGE_PATH = "source/asset.usd"
SOURCE_TEXTURE_PACKAGE_ROOT = PurePosixPath("source/textures")
OUTPUT_USD_PACKAGE_PATH = "prepared/deformable.usda"

OUTPUT_BODY_PRIM_PATH = "/Deformable"
OUTPUT_VISUAL_PRIM_PATH = "/Deformable/Visuals/Surface"
OUTPUT_LOOKS_PRIM_PATH = "/Deformable/Looks"
OUTPUT_PHYSICS_MATERIAL_PRIM_PATH = "/Deformable/PhysicsMaterial"

DEFORMABLE_MATERIAL_API = (
    "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"
)
DEFORMABLE_MATERIAL_CFG = (
    "isaaclab.sim.spawners.materials.physics_materials_cfg:DeformableBodyMaterialCfg"
)
DEFORMABLE_BODY_CFG = "isaaclab.sim.schemas.schemas_cfg:DeformableBodyPropertiesCfg"
DEFORMABLE_PHYSICS_BINDING_API = "isaaclab.sim.utils.prims:bind_physics_material"
DEFORMABLE_BODY_SCHEMAS = (
    "pxr.OmniPhysicsSchema.OmniPhysicsDeformableBodyAPI",
    "pxr.PhysxSchema.PhysxBaseDeformableBodyAPI",
    "pxr.PhysxSchema.PhysxCollisionAPI",
)
DEFORMABLE_MATERIAL_SCHEMAS = ("pxr.PhysxSchema.PhysxDeformableBodyMaterialAPI",)
NATIVE_REQUIRED_API_SYMBOLS = (
    DEFORMABLE_MATERIAL_CFG,
    DEFORMABLE_MATERIAL_API,
    DEFORMABLE_BODY_CFG,
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_PHYSICS_BINDING_API,
)
NATIVE_EXECUTED_API_SYMBOLS = (
    DEFORMABLE_MATERIAL_API,
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_PHYSICS_BINDING_API,
)

MATERIAL_API_STATUS = "positive_native_prim_returned"
AUTHORING_API_STATUS = "pinned_none_returned_readback_required"
PHYSICS_BINDING_API_STATUS = "pinned_none_returned_readback_required"
_DIGEST_PREFIX = "sha256:"
_IDENTITY_SCALE = [1.0, 1.0, 1.0]
_DIMENSION_TOLERANCE_M = 1.0e-6
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024
_MAX_TEXTURE_COUNT = 4096
_MAX_TEXTURE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_OUTPUT_FILE_COUNT = _MAX_TEXTURE_COUNT + 2
_MAX_OUTPUT_TOTAL_BYTES = 768 * 1024 * 1024
_MAX_OUTPUT_DIRECTORY_COUNT = _MAX_OUTPUT_FILE_COUNT * 2
_MAX_OUTPUT_ENTRY_COUNT = _MAX_OUTPUT_FILE_COUNT + _MAX_OUTPUT_DIRECTORY_COUNT
_MAX_OUTPUT_DEPTH = 64
_MAX_OUTPUT_RELATIVE_PATH_BYTES = 4096
_MAX_WORKER_RETURN_BYTES = 16 * 1024 * 1024
_MAX_OUTPUT_SNAPSHOT_DESCRIPTOR_COUNT = _MAX_OUTPUT_FILE_COUNT + _MAX_OUTPUT_DIRECTORY_COUNT + 1
_OUTPUT_SNAPSHOT_DESCRIPTOR_RESERVE = 64

_BODY_CFG_FIELDS = frozenset(
    {
        "deformable_body_enabled",
        "kinematic_enabled",
        "mass",
        "self_collision",
        "self_collision_filter_distance",
        "settling_threshold",
        "settling_damping",
        "sleep_threshold",
        "solver_position_iteration_count",
        "linear_damping",
        "max_linear_velocity",
        "contact_offset",
        "rest_offset",
        "max_depenetration_velocity",
        "enable_speculative_c_c_d",
        "disable_gravity",
        "collision_pair_update_frequency",
        "collision_iteration_multiplier",
    }
)
_SOURCE_COOKING_FIELDS = frozenset(
    {
        "simulation_hexahedral_resolution",
        "collision_simplification",
        "collision_simplification_remeshing",
        "collision_simplification_remeshing_resolution",
        "collision_simplification_target_triangle_count",
        "collision_simplification_force_conforming",
    }
)
_MATERIAL_CFG_FIELDS = frozenset(
    {
        "density",
        "static_friction",
        "dynamic_friction",
        "youngs_modulus",
        "poissons_ratio",
        "elasticity_damping",
    }
)
PINNED_NATIVE_CALL_CONTRACT = {
    "material_spawn": {
        "symbol": DEFORMABLE_MATERIAL_API,
        "source_relative_path": (
            "source/isaaclab/isaaclab/sim/spawners/materials/physics_materials.py"
        ),
        "source_git_blob_sha1": "8c12bee9442dbf4122b67234ff9ccca40cc02a74",
        "parameters": ["prim_path", "cfg"],
        "stage_keyword_supported": False,
        "configuration_symbol": DEFORMABLE_MATERIAL_CFG,
        "positive_prim_return_required": True,
    },
        "deformable_authoring": {
        "symbol": DEFORMABLE_AUTHORING_API,
        "source_relative_path": "source/isaaclab/isaaclab/sim/schemas/schemas.py",
        "source_git_blob_sha1": "8bd2c314bf931afe160759fb1ac3f92e24358ff3",
        "parameters": [
            "prim_path",
            "cfg",
            "stage",
            "deformable_type",
            "sim_mesh_prim_path",
        ],
        "configuration_symbol": DEFORMABLE_BODY_CFG,
        "embedded_cooking_owner": "isaaclab.sim.schemas.schemas:define_deformable_body_properties",
        "body_cfg_constructor_receives_cooking_fields": False,
        "source_cooking_properties_recorded_not_constructor_kwargs": True,
        "explicit_success_return": None,
        "direct_duplicate_cook_forbidden": True,
        "schema_is_applied_to_authoring_root": True,
    },
    "physics_material_binding": {
        "symbol": DEFORMABLE_PHYSICS_BINDING_API,
        "source_relative_path": "source/isaaclab/isaaclab/sim/utils/prims.py",
        "source_git_blob_sha1": "d0f0e8d9042a531ce617645cdc158fa4ac81f754",
        "parameters": [
            "prim_path",
            "material_path",
            "stage",
            "stronger_than_descendants",
        ],
        "decorator_return": None,
        "readback_required": True,
        "material_purpose": "physics",
    },
    "configuration_sources": {
        DEFORMABLE_BODY_CFG: {
            "source_relative_path": "source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py",
            "source_git_blob_sha1": "d6dc99a847482a96fc7db07df023ad4f16584138",
            "allowed_fields": sorted(_BODY_CFG_FIELDS),
        },
        DEFORMABLE_MATERIAL_CFG: {
            "source_relative_path": (
                "source/isaaclab/isaaclab/sim/spawners/materials/physics_materials_cfg.py"
            ),
            "source_git_blob_sha1": "5c88731cf8d5b056812eb4713e534312eab1dc68",
            "allowed_fields": sorted(_MATERIAL_CFG_FIELDS),
        },
    },
}


class NativeDeformableAssetPreparationError(ValueError):
    """Stable, sorted preparation-contract failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class _RegularFileSnapshot:
    """One descriptor-anchored regular-file observation."""

    size_bytes: int
    sha256: str
    content: bytes | None


@dataclass(frozen=True)
class _OutputTreeSnapshot:
    """One bounded descriptor-relative observation of an output tree."""

    files: Mapping[str, _RegularFileSnapshot]
    directories: frozenset[str]


@dataclass(frozen=True)
class _HeldRegularFile:
    descriptor: int
    parent_descriptor: int
    name: str
    identity: os.stat_result
    snapshot: _RegularFileSnapshot


@dataclass(frozen=True)
class _HeldDirectory:
    descriptor: int
    parent_descriptor: int | None
    name: str | None
    identity: os.stat_result


class NativeDeformableStageAPI(Protocol):
    """Minimal clean-stage seam implemented inside the pinned native runtime."""

    def create_clean_stage(
        self,
        *,
        output_path: Path,
        default_prim_path: str,
        meters_per_unit: float,
        up_axis: str,
    ) -> object: ...

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
    ) -> None: ...

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
    ) -> None: ...

    def activate_and_verify_current_stage(self, *, stage: object) -> bool:
        """Make ``stage`` current for native helpers that expose no stage argument."""

        ...

    def record_native_configuration(
        self,
        *,
        stage: object,
        body_and_cooking_properties: Mapping[str, Any],
        material_properties: Mapping[str, Any],
    ) -> None: ...

    def release_current_stage(self, *, stage: object) -> None:
        """Idempotently release a context acquired for native helper calls."""

        ...

    def save_stage(self, *, stage: object) -> None: ...

    def readback_prepared_stage(
        self,
        *,
        stage: object,
        output_authoring_root_prim_path: str,
        output_deformable_schema_prim_path: str,
        output_visual_prim_path: str,
    ) -> Mapping[str, Any]: ...


def _json_clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeDeformableAssetPreparationError([error]) from exc
    if not isinstance(cloned, dict):
        raise NativeDeformableAssetPreparationError([error])
    return cloned


def _valid_digest(value: Any) -> bool:
    text = str(value or "")
    return bool(
        text.startswith(_DIGEST_PREFIX)
        and len(text) == len(_DIGEST_PREFIX) + 64
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _descriptor_flags(*, directory: bool, error: str) -> int:
    """Return the fail-closed flags used by every trusted path traversal."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or (directory and directory_flag is None):
        raise NativeDeformableAssetPreparationError([error])
    flags = os.O_RDONLY | nofollow
    if directory:
        flags |= directory_flag
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _absolute_path_components(path: Path, *, error: str) -> tuple[str, ...]:
    absolute = path.expanduser().absolute()
    parts = absolute.parts
    if (
        not absolute.is_absolute()
        or not parts
        or parts[0] != os.sep
        or any(part in {"", ".", ".."} or os.sep in part for part in parts[1:])
    ):
        raise NativeDeformableAssetPreparationError([error])
    return tuple(parts[1:])


def _open_directory_descriptor(path: Path, *, error: str) -> int:
    """Open a directory through anchored ``openat`` steps without symlinks."""

    components = _absolute_path_components(path, error=error)
    flags = _descriptor_flags(directory=True, error=error)
    descriptor = -1
    try:
        descriptor = os.open(os.sep, flags)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise NativeDeformableAssetPreparationError([error])
        for component in components:
            child = -1
            try:
                entry_before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                child = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(child)
                entry_after = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(entry_before.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or not _same_identity(entry_before, opened)
                    or not _same_identity(opened, entry_after)
                ):
                    raise NativeDeformableAssetPreparationError([error])
            except NativeDeformableAssetPreparationError:
                if child >= 0:
                    os.close(child)
                raise
            except (OSError, TypeError, NotImplementedError) as exc:
                if child >= 0:
                    os.close(child)
                raise NativeDeformableAssetPreparationError([error]) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except NativeDeformableAssetPreparationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (OSError, TypeError, NotImplementedError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise NativeDeformableAssetPreparationError([error]) from exc


def _open_regular_file_descriptors(path: Path, *, error: str) -> tuple[int, int, str]:
    """Open a regular file and its parent through one anchored component walk."""

    components = _absolute_path_components(path, error=error)
    if not components:
        raise NativeDeformableAssetPreparationError([error])
    parent_path = Path(os.sep, *components[:-1])
    parent_descriptor = _open_directory_descriptor(parent_path, error=error)
    name = components[-1]
    file_descriptor = -1
    try:
        try:
            entry_before = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            file_descriptor = os.open(
                name,
                _descriptor_flags(directory=False, error=error),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(file_descriptor)
            entry_after = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except (OSError, TypeError, NotImplementedError) as exc:
            raise NativeDeformableAssetPreparationError([error]) from exc
        if (
            not stat.S_ISREG(entry_before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_identity(entry_before, opened)
            or not _same_identity(opened, entry_after)
        ):
            raise NativeDeformableAssetPreparationError([error])
        return file_descriptor, parent_descriptor, name
    except Exception:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(parent_descriptor)
        raise


def _snapshot_open_regular_file(
    *,
    file_descriptor: int,
    parent_descriptor: int,
    name: str,
    maximum_size: int,
    expected_digest: Any | None,
    expected_size: Any | None,
    retain_content: bool,
    error: str,
) -> _RegularFileSnapshot:
    """Consume one already anchored descriptor and verify its stable identity."""

    try:
        if isinstance(maximum_size, bool) or not isinstance(maximum_size, int) or maximum_size <= 0:
            raise NativeDeformableAssetPreparationError([error])
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum_size:
            raise NativeDeformableAssetPreparationError([error])
        if expected_size is not None and (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or before.st_size != expected_size
        ):
            raise NativeDeformableAssetPreparationError([error])
        chunks: list[bytes] = []
        total = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(
                file_descriptor,
                min(1024 * 1024, maximum_size + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_size:
                raise NativeDeformableAssetPreparationError([error])
            digest.update(chunk)
            if retain_content:
                chunks.append(chunk)
        after = os.fstat(file_descriptor)
        try:
            current = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except (OSError, TypeError, NotImplementedError) as exc:
            raise NativeDeformableAssetPreparationError([error]) from exc
        actual_digest = f"sha256:{digest.hexdigest()}"
        if (
            total != before.st_size
            or not _same_identity(before, after)
            or not _same_identity(after, current)
            or (expected_digest is not None and actual_digest != expected_digest)
        ):
            raise NativeDeformableAssetPreparationError([error])
        return _RegularFileSnapshot(
            size_bytes=total,
            sha256=actual_digest,
            content=b"".join(chunks) if retain_content else None,
        )
    except (OSError, TypeError, NotImplementedError) as exc:
        raise NativeDeformableAssetPreparationError([error]) from exc


def _read_regular_file_once(
    path: Path,
    *,
    maximum_size: int,
    expected_digest: Any | None,
    expected_size: Any | None,
    error: str,
) -> bytes:
    """Read one immutable component-anchored file snapshot.

    Every parent and the final file are opened relative to an already-open
    directory with ``O_NOFOLLOW``.  Path-component replacement therefore
    cannot redirect later operations, and downstream consumers use only the
    returned bytes.
    """

    file_descriptor, parent_descriptor, name = _open_regular_file_descriptors(
        path,
        error=error,
    )
    try:
        snapshot = _snapshot_open_regular_file(
            file_descriptor=file_descriptor,
            parent_descriptor=parent_descriptor,
            name=name,
            maximum_size=maximum_size,
            expected_digest=expected_digest,
            expected_size=expected_size,
            retain_content=True,
            error=error,
        )
        if snapshot.content is None:
            raise NativeDeformableAssetPreparationError([error])
        return snapshot.content
    finally:
        os.close(file_descriptor)
        os.close(parent_descriptor)


def _mapping(value: Any, *, error: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(error)
        return {}
    return value


def _rows(value: Any, *, error: str, errors: list[str]) -> list[Mapping[str, Any]]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or any(not isinstance(row, Mapping) for row in value)
    ):
        errors.append(error)
        return []
    return list(value)


def _strings(value: Any, *, error: str, errors: list[str]) -> list[str]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        errors.append(error)
        return []
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        errors.append(error)
    return result


def _positive_int(value: Any, *, error: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(error)
        return 0
    return value


def _positive_vector3(value: Any, *, error: str, errors: list[str]) -> list[float]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        errors.append(error)
        return []
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            errors.append(error)
            return []
        try:
            number = float(item)
        except (TypeError, ValueError):
            errors.append(error)
            return []
        if not math.isfinite(number) or number <= 0.0:
            errors.append(error)
            return []
        result.append(number)
    return result


def _finite_vector3(value: Any, *, error: str, errors: list[str]) -> list[float]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        errors.append(error)
        return []
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            errors.append(error)
            return []
        try:
            number = float(item)
        except (TypeError, ValueError):
            errors.append(error)
            return []
        if not math.isfinite(number):
            errors.append(error)
            return []
        result.append(number)
    return result


def _safe_relative_path(value: Any, *, error: str, errors: list[str]) -> str:
    text = str(value or "").strip()
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in text
        or relative.name in {"", ".", ".."}
    ):
        errors.append(error)
        return ""
    return relative.as_posix()


def _path_has_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    parts = candidate.parts
    if not parts:
        return True
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _regular_file_identity(
    path: Path,
    *,
    expected_digest: Any,
    expected_size: Any,
    error: str,
    errors: list[str],
) -> None:
    try:
        if not _valid_digest(expected_digest):
            raise NativeDeformableAssetPreparationError([error])
        _read_regular_file_once(
            path,
            maximum_size=_MAX_SOURCE_FILE_BYTES,
            expected_digest=expected_digest,
            expected_size=expected_size,
            error=error,
        )
    except NativeDeformableAssetPreparationError:
        errors.append(error)


def _normalize_texture_rows(
    value: Any,
    *,
    texture_root: Path,
    errors: list[str],
) -> list[dict[str, Any]]:
    rows = _rows(value, error="native_deformable_inspection_textures_invalid", errors=errors)
    if len(rows) > _MAX_TEXTURE_COUNT:
        errors.append("native_deformable_inspection_texture_count_exceeded")
        rows = rows[:_MAX_TEXTURE_COUNT]
    normalized: list[dict[str, Any]] = []
    relative_paths: set[str] = set()
    total_size = 0
    for index, row in enumerate(rows):
        relative_path = _safe_relative_path(
            row.get("relative_path"),
            error=f"native_deformable_inspection_texture_path_invalid:{index}",
            errors=errors,
        )
        source_path = texture_root / Path(*PurePosixPath(relative_path).parts)
        expected_path = str(source_path.resolve()) if source_path.exists() else str(source_path)
        if row.get("source_path") != expected_path:
            errors.append(f"native_deformable_inspection_texture_path_mismatch:{index}")
        if relative_path in relative_paths:
            errors.append("native_deformable_inspection_texture_path_duplicate")
        relative_paths.add(relative_path)
        size_value = row.get("size_bytes")
        if isinstance(size_value, int) and not isinstance(size_value, bool):
            total_size += size_value
        _regular_file_identity(
            source_path,
            expected_digest=row.get("sha256"),
            expected_size=row.get("size_bytes"),
            error=f"native_deformable_inspection_texture_identity_mismatch:{index}",
            errors=errors,
        )
        normalized.append(
            {
                "relative_path": relative_path,
                "source_path": expected_path,
                "package_path": (
                    SOURCE_TEXTURE_PACKAGE_ROOT / PurePosixPath(relative_path)
                ).as_posix(),
                "sha256": row.get("sha256"),
                "size_bytes": row.get("size_bytes"),
            }
        )
    if not normalized:
        errors.append("native_deformable_inspection_textures_missing")
    if total_size > _MAX_TEXTURE_TOTAL_BYTES:
        errors.append("native_deformable_inspection_texture_total_size_exceeded")
    return sorted(normalized, key=lambda row: row["relative_path"])


def _normalize_schema_rows(value: Any, *, errors: list[str]) -> list[dict[str, str]]:
    rows = _rows(
        value,
        error="native_deformable_inspection_experimental_schemas_invalid",
        errors=errors,
    )
    result: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        prim_path = str(row.get("prim_path") or "").strip()
        schema = str(row.get("schema") or "").strip()
        if not prim_path.startswith("/") or not schema:
            errors.append(f"native_deformable_inspection_experimental_schema_invalid:{index}")
            continue
        identity = (prim_path, schema)
        if identity in identities:
            errors.append("native_deformable_inspection_experimental_schema_duplicate")
        identities.add(identity)
        result.append({"prim_path": prim_path, "schema": schema})
    return sorted(result, key=lambda row: (row["prim_path"], row["schema"]))


def _normalize_physics_configuration(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(
        value,
        error="native_deformable_physics_configuration_invalid",
        errors=errors,
    )
    expected_keys = {"body_properties", "cooking_properties", "material_properties"}
    if set(source) != expected_keys:
        errors.append("native_deformable_physics_configuration_fields_invalid")
    result: dict[str, Any] = {}
    for key in sorted(expected_keys):
        mapping = _mapping(
            source.get(key),
            error=f"native_deformable_physics_{key}_invalid",
            errors=errors,
        )
        if not mapping:
            errors.append(f"native_deformable_physics_{key}_empty")
        cloned = _json_clone(
            mapping,
            error=f"native_deformable_physics_{key}_not_json",
        )
        if any(
            token in str(field).lower()
            for field in cloned
            for token in ("qualified", "success", "equivalent", "physical_truth")
        ):
            errors.append("native_deformable_physics_claim_field_forbidden")
        result[key] = cloned
    body = result.get("body_properties", {})
    cooking = result.get("cooking_properties", {})
    material = result.get("material_properties", {})
    if set(body) & set(cooking):
        errors.append("native_deformable_physics_body_cooking_field_overlap")
    if not set(body).issubset(_BODY_CFG_FIELDS):
        errors.append("native_deformable_physics_body_cfg_fields_unsupported")
    if not set(cooking).issubset(_SOURCE_COOKING_FIELDS):
        errors.append("native_deformable_physics_cooking_fields_unsupported")
    if not set(material).issubset(_MATERIAL_CFG_FIELDS):
        errors.append("native_deformable_physics_material_cfg_fields_unsupported")
    if body.get("deformable_body_enabled") is not True:
        errors.append("native_deformable_physics_deformable_enabled_required")
    if body.get("kinematic_enabled") is not False:
        errors.append("native_deformable_physics_kinematic_disabled_required")
    if not {
        "deformable_body_enabled",
        "kinematic_enabled",
        "self_collision",
        "solver_position_iteration_count",
        "linear_damping",
        "contact_offset",
        "rest_offset",
    }.issubset(body):
        errors.append("native_deformable_physics_body_required_fields_missing")
    if not {
        "collision_simplification",
        "collision_simplification_remeshing",
        "collision_simplification_remeshing_resolution",
        "simulation_hexahedral_resolution",
    }.issubset(cooking):
        errors.append("native_deformable_physics_cooking_required_fields_missing")
    for field in (
        "deformable_body_enabled",
        "kinematic_enabled",
        "self_collision",
        "enable_speculative_c_c_d",
        "disable_gravity",
        "collision_simplification",
        "collision_simplification_remeshing",
        "collision_simplification_force_conforming",
    ):
        if field in body or field in cooking:
            candidate = body[field] if field in body else cooking[field]
            if not isinstance(candidate, bool):
                errors.append(f"native_deformable_physics_{field}_type_invalid")
    for field in (
        "solver_position_iteration_count",
        "collision_pair_update_frequency",
        "simulation_hexahedral_resolution",
        "collision_simplification_remeshing_resolution",
        "collision_simplification_target_triangle_count",
    ):
        if field in body or field in cooking:
            candidate = body[field] if field in body else cooking[field]
            minimum = (
                1
                if field
                in {
                    "solver_position_iteration_count",
                    "simulation_hexahedral_resolution",
                }
                else 0
            )
            maximum = 255 if field == "solver_position_iteration_count" else 1_000_000
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, int)
                or candidate < minimum
                or candidate > maximum
            ):
                errors.append(f"native_deformable_physics_{field}_range_invalid")
    numeric_fields = {
        "self_collision_filter_distance",
        "settling_threshold",
        "settling_damping",
        "sleep_threshold",
        "linear_damping",
        "max_linear_velocity",
        "contact_offset",
        "rest_offset",
        "max_depenetration_velocity",
        "collision_iteration_multiplier",
        "mass",
    }
    for field in sorted(numeric_fields & (set(body) | set(cooking))):
        candidate = body[field] if field in body else cooking[field]
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (int, float))
            or not math.isfinite(float(candidate))
            or float(candidate) < 0.0
        ):
            errors.append(f"native_deformable_physics_{field}_range_invalid")
    for field in sorted(set(material)):
        candidate = material[field]
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (int, float))
            or not math.isfinite(float(candidate))
        ):
            errors.append(f"native_deformable_physics_material_{field}_range_invalid")
            continue
        number = float(candidate)
        if field == "poissons_ratio":
            if number <= 0.0 or number >= 0.5:
                errors.append(f"native_deformable_physics_material_{field}_range_invalid")
        elif field in {"density", "youngs_modulus"}:
            if number <= 0.0:
                errors.append(f"native_deformable_physics_material_{field}_range_invalid")
        elif number < 0.0:
            errors.append(f"native_deformable_physics_material_{field}_range_invalid")
    if not {
        "density",
        "static_friction",
        "dynamic_friction",
        "youngs_modulus",
        "poissons_ratio",
        "elasticity_damping",
    }.issubset(material):
        errors.append("native_deformable_physics_material_required_fields_missing")
    contact_offset = body.get("contact_offset")
    rest_offset = body.get("rest_offset")
    if (
        isinstance(contact_offset, (int, float))
        and not isinstance(contact_offset, bool)
        and isinstance(rest_offset, (int, float))
        and not isinstance(rest_offset, bool)
        and float(rest_offset) > float(contact_offset)
    ):
        errors.append("native_deformable_physics_rest_offset_exceeds_contact_offset")
    filter_distance = body.get("self_collision_filter_distance")
    if (
        body.get("self_collision") is True
        and filter_distance is not None
        and isinstance(filter_distance, (int, float))
        and not isinstance(filter_distance, bool)
        and isinstance(rest_offset, (int, float))
        and not isinstance(rest_offset, bool)
        and float(filter_distance) < 2.0 * float(rest_offset)
    ):
        errors.append("native_deformable_physics_self_collision_filter_distance_invalid")
    return result


def _load_replayed_inspection_receipt(
    *,
    inspection_receipt_path: str | Path,
    expected_inspection_receipt_digest: str,
) -> dict[str, Any]:
    receipt_path = Path(inspection_receipt_path).expanduser().absolute()
    if not _valid_digest(expected_inspection_receipt_digest):
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_expected_inspection_receipt_digest_invalid"]
        )
    content = _read_regular_file_once(
        receipt_path,
        maximum_size=_MAX_RECEIPT_BYTES,
        expected_digest=None,
        expected_size=None,
        error="native_deformable_inspection_receipt_file_invalid",
    )
    try:
        persisted = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_inspection_receipt_file_invalid"]
        ) from exc
    if not isinstance(persisted, dict):
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_inspection_receipt_file_invalid"]
        )
    if (
        persisted.get("receipt_digest") != expected_inspection_receipt_digest
        or canonical_digest(persisted, digest_field="receipt_digest")
        != expected_inspection_receipt_digest
    ):
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_inspection_receipt_expected_digest_mismatch"]
        )
    input_paths = persisted.get("input_paths")
    if not isinstance(input_paths, Mapping) or set(input_paths) != {
        "archive",
        "expanded_root",
        "usd",
        "observation",
        "source_topology_receipt",
    }:
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_inspection_input_paths_invalid"]
        )
    try:
        replayed = verify_external_simready_deformable_asset_inspection(
            archive_path=str(input_paths["archive"]),
            expanded_root=str(input_paths["expanded_root"]),
            usd_path=str(input_paths["usd"]),
            observation_path=str(input_paths["observation"]),
            expected_receipt_digest=expected_inspection_receipt_digest,
        )
    except ExternalSimreadyDeformableAssetError as exc:
        raise NativeDeformableAssetPreparationError(
            [f"native_deformable_inspection_replay_failed:{error}" for error in exc.errors]
        ) from exc
    replayed_json = _json_clone(
        replayed,
        error="native_deformable_inspection_replay_not_json",
    )
    if replayed_json != persisted:
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_inspection_persisted_replay_mismatch"]
        )
    return replayed_json


def materialize_native_deformable_asset_preparation_plan(
    *,
    preparation_id: str,
    inspection_receipt_path: str | Path,
    expected_inspection_receipt_digest: str,
    source_usd_path: str | Path,
    source_texture_root: str | Path,
    target_metric_dimensions_m: Sequence[float],
    physics_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact inspected bytes and freeze a clean native preparation plan."""

    receipt = _load_replayed_inspection_receipt(
        inspection_receipt_path=inspection_receipt_path,
        expected_inspection_receipt_digest=expected_inspection_receipt_digest,
    )
    errors: list[str] = []
    identifier = str(preparation_id or "").strip()
    if not identifier:
        errors.append("native_deformable_preparation_id_missing")
    if receipt.get("schema_version") != INSPECTION_SCHEMA_VERSION:
        errors.append("native_deformable_inspection_schema_unsupported")
    if receipt.get("status") != INSPECTION_STATUS:
        errors.append("native_deformable_inspection_not_verified")
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        errors.append("native_deformable_inspection_receipt_digest_invalid")
    input_paths = _mapping(
        receipt.get("input_paths"),
        error="native_deformable_inspection_input_paths_invalid",
        errors=errors,
    )
    source_package = _mapping(
        receipt.get("source_package"),
        error="native_deformable_inspection_source_package_invalid",
        errors=errors,
    )
    stage = _mapping(
        receipt.get("openusd_inspection"),
        error="native_deformable_inspection_openusd_invalid",
        errors=errors,
    )
    conversion = _mapping(
        receipt.get("pinned_physx_conversion"),
        error="native_deformable_inspection_conversion_invalid",
        errors=errors,
    )
    claim_ceiling = _mapping(
        receipt.get("claim_ceiling"),
        error="native_deformable_inspection_claim_ceiling_invalid",
        errors=errors,
    )
    if (
        source_package.get("archive_expanded_identity_verified") is not True
        or source_package.get("source_bytes_mutated_by_inspector") is not False
        or source_package.get("inspection_bound_to_snapshot_not_live_path") is not True
        or not _valid_digest(source_package.get("archive_sha256"))
        or not _valid_digest(source_package.get("expanded_tree_digest"))
    ):
        errors.append("native_deformable_inspection_source_identity_unverified")
    if (
        conversion.get("required") is not True
        or conversion.get("source_representation") != "closed_surface_mesh_pending_native_cook"
        or conversion.get("derived_runtime_usd_must_be_separate") is not True
        or conversion.get("source_usd_must_remain_immutable") is not True
        or conversion.get("bake_metric_scale_before_cooking") is not True
        or conversion.get("exclude_embedded_lighting_and_provider_auto_cook_prims") is not True
        or conversion.get("repository") != ISAACLAB_REPOSITORY
        or conversion.get("commit") != ISAACLAB_COMMIT
        or conversion.get("tree") != ISAACLAB_TREE
        or conversion.get("authoring_api") != DEFORMABLE_AUTHORING_API
        or conversion.get("cooking_api") != DEFORMABLE_COOKING_API
    ):
        errors.append("native_deformable_inspection_pinned_conversion_mismatch")
    if (
        claim_ceiling.get("maximum_claim") != "static_external_deformable_candidate_only"
        or claim_ceiling.get("geometry_and_material_candidate_inspected") is not True
        or claim_ceiling.get("standard_physx_runtime_asset") is not False
        or claim_ceiling.get("simready_asset_admitted") is not False
        or claim_ceiling.get("native_simulator_qualified") is not False
        or claim_ceiling.get("physically_equivalent_real_material") is not False
        or claim_ceiling.get("provider_or_gpu_execution") != "not_performed"
    ):
        errors.append("native_deformable_inspection_claim_boundary_invalid")

    source_usd = Path(source_usd_path).expanduser().absolute()
    texture_root = Path(source_texture_root).expanduser().absolute()
    expanded_root = Path(str(input_paths.get("expanded_root") or "")).expanduser().absolute()
    if source_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        errors.append("native_deformable_source_usd_extension_invalid")
    try:
        resolved_usd = source_usd.resolve(strict=True)
    except (FileNotFoundError, OSError):
        resolved_usd = source_usd
        errors.append("native_deformable_source_usd_missing")
    if _path_has_symlink_component(source_usd):
        errors.append("native_deformable_source_usd_symlink_forbidden")
    try:
        resolved_texture_root = texture_root.resolve(strict=True)
    except (FileNotFoundError, OSError):
        resolved_texture_root = texture_root
        errors.append("native_deformable_source_texture_root_missing")
    if not resolved_texture_root.is_dir() or _path_has_symlink_component(texture_root):
        errors.append("native_deformable_source_texture_root_invalid")
    try:
        resolved_expanded_root = expanded_root.resolve(strict=True)
    except (FileNotFoundError, OSError):
        resolved_expanded_root = expanded_root
        errors.append("native_deformable_source_expanded_root_missing")
    if (
        not resolved_expanded_root.is_dir()
        or _path_has_symlink_component(expanded_root)
        or input_paths.get("expanded_root") != str(expanded_root)
    ):
        errors.append("native_deformable_source_expanded_root_invalid")
    if input_paths.get("usd") != str(source_usd):
        errors.append("native_deformable_source_usd_path_mismatch")
    try:
        usd_relative = resolved_usd.relative_to(resolved_expanded_root).as_posix()
        texture_root_relative = resolved_texture_root.relative_to(resolved_expanded_root).as_posix()
    except ValueError:
        usd_relative = ""
        texture_root_relative = ""
        errors.append("native_deformable_source_paths_outside_inspected_root")
    if usd_relative != stage.get("root_layer_relative_path"):
        errors.append("native_deformable_source_usd_relative_path_mismatch")

    expanded_file_rows = _rows(
        source_package.get("expanded_files"),
        error="native_deformable_inspection_file_inventory_invalid",
        errors=errors,
    )
    expanded_files = {str(row.get("relative_path") or ""): row for row in expanded_file_rows}
    if len(expanded_files) != len(expanded_file_rows) or usd_relative not in expanded_files:
        errors.append("native_deformable_inspection_file_inventory_incomplete")
    usd_row = expanded_files.get(usd_relative, {})
    if usd_row.get("sha256") != stage.get("root_layer_sha256") or usd_row.get(
        "size_bytes"
    ) != stage.get("root_layer_size_bytes"):
        errors.append("native_deformable_inspection_usd_inventory_mismatch")
    _regular_file_identity(
        resolved_usd,
        expected_digest=usd_row.get("sha256"),
        expected_size=usd_row.get("size_bytes"),
        error="native_deformable_source_usd_identity_mismatch",
        errors=errors,
    )
    meters_per_unit = stage.get("meters_per_unit")
    if (
        isinstance(meters_per_unit, bool)
        or not isinstance(meters_per_unit, (int, float))
        or not math.isfinite(float(meters_per_unit))
        or float(meters_per_unit) <= 0.0
    ):
        errors.append("native_deformable_source_meters_per_unit_invalid")
    elif abs(float(meters_per_unit) - 1.0) > 1.0e-12:
        errors.append("native_deformable_source_nonmetric_stage_requires_explicit_conversion")
    up_axis = str(stage.get("up_axis") or "").upper()
    if up_axis != "Z":
        errors.append("native_deformable_source_non_z_up_requires_explicit_conversion")

    surface = _mapping(
        stage.get("surface_mesh"),
        error="native_deformable_inspection_surface_mesh_invalid",
        errors=errors,
    )
    source_prim_path = str(surface.get("prim_path") or "").strip()
    if not source_prim_path.startswith("/"):
        errors.append("native_deformable_inspection_surface_prim_invalid")
    source_world_bounds = _mapping(
        surface.get("world_bounds"),
        error="native_deformable_inspection_surface_world_bounds_invalid",
        errors=errors,
    )
    source_dimensions = _positive_vector3(
        source_world_bounds.get("dimensions"),
        error="native_deformable_inspection_surface_dimensions_invalid",
        errors=errors,
    )
    source_world_minimum = _finite_vector3(
        source_world_bounds.get("minimum"),
        error="native_deformable_inspection_surface_minimum_invalid",
        errors=errors,
    )
    source_world_maximum = _finite_vector3(
        source_world_bounds.get("maximum"),
        error="native_deformable_inspection_surface_maximum_invalid",
        errors=errors,
    )
    source_world_center = (
        [(source_world_minimum[index] + source_world_maximum[index]) / 2.0 for index in range(3)]
        if len(source_world_minimum) == len(source_world_maximum) == 3
        else []
    )
    if (
        source_dimensions
        and source_world_center
        and any(
            abs(
                source_world_maximum[index] - source_world_minimum[index] - source_dimensions[index]
            )
            > 1.0e-12
            for index in range(3)
        )
    ):
        errors.append("native_deformable_inspection_surface_bounds_inconsistent")
    requested_target_dimensions = _positive_vector3(
        target_metric_dimensions_m,
        error="native_deformable_target_dimensions_invalid",
        errors=errors,
    )
    observed = _mapping(
        receipt.get("observation"),
        error="native_deformable_inspection_observation_invalid",
        errors=errors,
    )
    target_dimensions = _positive_vector3(
        observed.get("dimensions_m"),
        error="native_deformable_inspection_observed_dimensions_invalid",
        errors=errors,
    )
    if (
        requested_target_dimensions
        and target_dimensions
        and any(
            abs(requested_target_dimensions[index] - target_dimensions[index]) > 1.0e-12
            for index in range(3)
        )
    ):
        errors.append("native_deformable_target_dimensions_observation_mismatch")
    point_count = _positive_int(
        surface.get("vertex_count"),
        error="native_deformable_inspection_surface_point_count_invalid",
        errors=errors,
    )
    triangle_count = _positive_int(
        surface.get("triangulated_face_count"),
        error="native_deformable_inspection_surface_triangle_count_invalid",
        errors=errors,
    )
    source_volume_m3 = surface.get("absolute_volume_m3")
    if (
        isinstance(source_volume_m3, bool)
        or not isinstance(source_volume_m3, (int, float))
        or not math.isfinite(float(source_volume_m3))
        or float(source_volume_m3) <= 0.0
    ):
        errors.append("native_deformable_inspection_surface_volume_invalid")
    if (
        surface.get("closed_oriented_manifold") is not True
        or surface.get("boundary_edge_count") != 0
        or surface.get("nonmanifold_edge_count") != 0
        or surface.get("orientation_mismatch_edge_count") != 0
        or surface.get("connected_component_count") != 1
        or surface.get("topology_errors") != []
    ):
        errors.append("native_deformable_inspection_surface_not_closed_manifold")

    relationships = _rows(
        stage.get("physics_relationships"),
        error="native_deformable_inspection_relationships_invalid",
        errors=errors,
    )
    material_rows = _rows(
        stage.get("materials"),
        error="native_deformable_inspection_materials_invalid",
        errors=errors,
    )
    inspected_material_prim_paths = {
        str(row.get("prim_path") or "")
        for row in material_rows
        if row.get("type_name") == "Material"
    }
    surface_binding_rows = [
        row
        for row in relationships
        if str(row.get("prim_path") or "") == source_prim_path
        and str(row.get("name") or "") == "material:binding"
    ]
    material_prim_paths = sorted(
        {
            str(target)
            for row in surface_binding_rows
            for target in row.get("targets") or []
            if str(target).startswith("/")
        }
    )
    if (
        len(surface_binding_rows) != 1
        or len(material_prim_paths) != 1
        or any(path not in inspected_material_prim_paths for path in material_prim_paths)
    ):
        errors.append("native_deformable_inspection_surface_material_binding_invalid")
    material_prim_path_map = {
        path: f"{OUTPUT_LOOKS_PRIM_PATH}/Material_{index:03d}"
        for index, path in enumerate(sorted(material_prim_paths))
    }

    dependencies = _rows(
        stage.get("dependencies"),
        error="native_deformable_inspection_dependencies_invalid",
        errors=errors,
    )
    light_prim_paths = {
        str(row.get("prim_path") or "")
        for row in _rows(
            stage.get("dome_lights"),
            error="native_deformable_inspection_dome_lights_invalid",
            errors=errors,
        )
    }
    material_dependencies: list[dict[str, Any]] = []
    for index, row in enumerate(dependencies):
        relative = str(row.get("relative_path") or "")
        consumer = str(row.get("consumer_prim_path") or "")
        if consumer in light_prim_paths:
            continue
        if not any(
            consumer == material_path or consumer.startswith(f"{material_path}/")
            for material_path in material_prim_paths
        ):
            continue
        try:
            relative_to_texture_root = PurePosixPath(relative).relative_to(
                PurePosixPath(texture_root_relative)
            )
        except ValueError:
            errors.append(
                f"native_deformable_inspection_material_dependency_outside_texture_root:{index}"
            )
            continue
        inventory_row = expanded_files.get(relative, {})
        if inventory_row.get("sha256") != row.get("sha256") or inventory_row.get(
            "size_bytes"
        ) != row.get("size_bytes"):
            errors.append(
                f"native_deformable_inspection_material_dependency_identity_mismatch:{index}"
            )
        material_dependencies.append(
            {
                "relative_path": relative_to_texture_root.as_posix(),
                "source_path": str(resolved_texture_root / Path(*relative_to_texture_root.parts)),
                "sha256": row.get("sha256"),
                "size_bytes": row.get("size_bytes"),
            }
        )
    texture_rows = _normalize_texture_rows(
        material_dependencies,
        texture_root=resolved_texture_root,
        errors=errors,
    )
    raw_schema_bindings = _rows(
        stage.get("raw_schema_bindings"),
        error="native_deformable_inspection_schema_bindings_invalid",
        errors=errors,
    )
    experimental_prefixes = (
        "OmniPhysics",
        "PhysxAuto",
        "PhysxBase",
        "Newton",
    )
    experimental_schemas = _normalize_schema_rows(
        [
            {"prim_path": row.get("prim_path"), "schema": schema}
            for row in raw_schema_bindings
            for schema in row.get("raw_authored_schemas") or []
            if str(schema).startswith(experimental_prefixes)
        ],
        errors=errors,
    )
    provider_attributes = [
        {
            "prim_path": str(row.get("prim_path") or ""),
            "name": str(row.get("name") or ""),
        }
        for row in _rows(
            stage.get("physics_attributes"),
            error="native_deformable_inspection_physics_attributes_invalid",
            errors=errors,
        )
        if str(row.get("name") or "").startswith(
            ("physics:", "physx", "omniphysics:", "lightwheelusd:")
        )
    ]
    tet_mesh_rows = _rows(
        stage.get("tetmeshes"),
        error="native_deformable_inspection_tetmeshes_invalid",
        errors=errors,
    )
    empty_tet_meshes = [
        str(row.get("prim_path") or "")
        for row in tet_mesh_rows
        if row.get("point_count") == 0 or row.get("tetrahedron_count") == 0
    ]
    guide_prims: list[str] = []
    light_prims = sorted(light_prim_paths)
    if any(not path.startswith("/") for path in [*empty_tet_meshes, *guide_prims, *light_prims]):
        errors.append("native_deformable_inspection_provider_prim_path_invalid")

    normalized_physics = _normalize_physics_configuration(physics_configuration, errors=errors)
    if errors:
        raise NativeDeformableAssetPreparationError(errors)

    bake_scale = [target_dimensions[index] / source_dimensions[index] for index in range(3)]
    dimension_alignment = _mapping(
        stage.get("dimension_alignment"),
        error="native_deformable_inspection_dimension_alignment_invalid",
        errors=errors,
    )
    inspected_bake_scale = _positive_vector3(
        dimension_alignment.get("required_bake_scale_xyz"),
        error="native_deformable_inspection_bake_scale_invalid",
        errors=errors,
    )
    if inspected_bake_scale and any(
        abs(inspected_bake_scale[index] - bake_scale[index]) > 1.0e-12 for index in range(3)
    ):
        errors.append("native_deformable_inspection_bake_scale_mismatch")
    if errors:
        raise NativeDeformableAssetPreparationError(errors)
    define_arguments = {
        "prim_path": OUTPUT_BODY_PRIM_PATH,
        "cfg_kwargs": normalized_physics["body_properties"],
    }
    material_arguments = {
        "prim_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
        "cfg_kwargs": normalized_physics["material_properties"],
    }
    physics_binding_arguments = {
        "prim_path": OUTPUT_BODY_PRIM_PATH,
        "material_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
        "stronger_than_descendants": True,
    }
    bake_scale_determinant = math.prod(bake_scale)
    expected_baked_volume_m3 = float(source_volume_m3) * bake_scale_determinant
    expected_mass_kg = expected_baked_volume_m3 * float(
        normalized_physics["material_properties"]["density"]
    )
    volume_tolerance_m3 = max(1.0e-12, expected_baked_volume_m3 * 1.0e-6)
    mass_tolerance_kg = max(1.0e-12, expected_mass_kg * 1.0e-6)
    result: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "preparation_id": identifier,
        "inspection_id": receipt["receipt_digest"],
        "inspection_receipt_digest": receipt["receipt_digest"],
        "source_asset": {
            "usd_path": str(resolved_usd),
            "usd_package_path": SOURCE_USD_PACKAGE_PATH,
            "usd_sha256": usd_row["sha256"],
            "usd_size_bytes": usd_row["size_bytes"],
            "texture_root_path": str(resolved_texture_root),
            "meters_per_unit": float(meters_per_unit),
            "up_axis": up_axis,
            "source_archive_sha256": source_package["archive_sha256"],
            "source_expanded_tree_digest": source_package["expanded_tree_digest"],
        },
        "source_surface_mesh": {
            "prim_path": source_prim_path,
            "point_count": point_count,
            "triangle_count": triangle_count,
            "dimensions_m": source_dimensions,
            "world_bounds_minimum_m": source_world_minimum,
            "world_bounds_maximum_m": source_world_maximum,
            "world_bounds_center_m": source_world_center,
            "closed_surface_volume_m3": float(source_volume_m3),
            "closed_surface": True,
            "manifold_surface": True,
            "material_prim_paths": sorted(material_prim_paths),
        },
        "textures": texture_rows,
        "clean_stage_rebuild": {
            "strategy": "allowlisted_surface_and_bound_material_reconstruction",
            "output_usd_package_path": OUTPUT_USD_PACKAGE_PATH,
            "output_default_prim_path": OUTPUT_BODY_PRIM_PATH,
            "output_visual_prim_path": OUTPUT_VISUAL_PRIM_PATH,
            "output_looks_prim_path": OUTPUT_LOOKS_PRIM_PATH,
            "material_prim_path_map": material_prim_path_map,
            "output_meters_per_unit": 1.0,
            "output_up_axis": "Z",
            "flatten_source_xform_to_points": True,
            "recenter_source_world_bounds_to_output_origin": True,
            "source_world_bounds_center_m": source_world_center,
            "output_authored_pivot_m": [0.0, 0.0, 0.0],
            "placement_origin_semantics": "body_pose_translation_is_replacement_aabb_center",
            "bake_metric_scale_into_points": True,
            "point_bake_scale_xyz": bake_scale,
            "point_bake_scale_determinant": bake_scale_determinant,
            "target_metric_dimensions_m": target_dimensions,
            "expected_baked_closed_volume_m3": expected_baked_volume_m3,
            "expected_density_kg_m3": float(normalized_physics["material_properties"]["density"]),
            "expected_mass_kg": expected_mass_kg,
            "authored_visual_scale_xyz_after_bake": list(_IDENTITY_SCALE),
            "copy_source_prim_subtree": False,
            "copy_source_api_schemas": False,
            "copy_empty_source_tet_meshes": False,
            "copy_guides": False,
            "copy_lights": False,
        },
        "source_content_exclusions": {
            "experimental_api_schemas": experimental_schemas,
            "provider_authored_attributes": sorted(
                provider_attributes, key=lambda row: (row["prim_path"], row["name"])
            ),
            "empty_tet_mesh_prim_paths": sorted(empty_tet_meshes),
            "guide_prim_paths": sorted(guide_prims),
            "light_prim_paths": sorted(light_prims),
        },
        "native_runtime": {
            "source_repository": ISAACLAB_REPOSITORY,
            "source_revision": ISAACLAB_COMMIT,
            "source_tree": ISAACLAB_TREE,
            "pinned_source_call_contract": _json_clone(
                PINNED_NATIVE_CALL_CONTRACT,
                error="native_deformable_pinned_call_contract_not_json",
            ),
            "required_api_symbols": list(NATIVE_REQUIRED_API_SYMBOLS),
            "executed_api_symbols": list(NATIVE_EXECUTED_API_SYMBOLS),
            "embedded_cooking_contract": {
                "owner_symbol": DEFORMABLE_AUTHORING_API,
                "legacy_external_cooking_symbol_not_required": DEFORMABLE_COOKING_API,
                "direct_cooking_call_forbidden": True,
                "body_cfg_constructor_receives_cooking_fields": False,
                "source_cooking_properties_recorded_not_constructor_kwargs": True,
                "pinned_authoring_return": None,
            },
            "api_calls_in_order": [
                {
                    "symbol": DEFORMABLE_MATERIAL_API,
                    "configuration_symbol": DEFORMABLE_MATERIAL_CFG,
                    "arguments": material_arguments,
                    "arguments_digest": canonical_digest(material_arguments),
                },
                {
                    "symbol": DEFORMABLE_AUTHORING_API,
                    "configuration_symbol": DEFORMABLE_BODY_CFG,
                    "arguments": define_arguments,
                    "arguments_digest": canonical_digest(define_arguments),
                },
                {
                    "symbol": DEFORMABLE_PHYSICS_BINDING_API,
                    "configuration_symbol": None,
                    "arguments": physics_binding_arguments,
                    "arguments_digest": canonical_digest(physics_binding_arguments),
                },
            ],
        },
        "physics_configuration": normalized_physics,
        "required_native_readback": {
            "stage_metadata": {
                "default_prim_path": OUTPUT_BODY_PRIM_PATH,
                "meters_per_unit": 1.0,
                "up_axis": "Z",
            },
            "visual_mesh": {
                "prim_path": OUTPUT_VISUAL_PRIM_PATH,
                "point_count": point_count,
                "triangle_count": triangle_count,
                "source_face_topology_sha256_required": True,
                "output_face_topology_sha256_must_match_source": True,
                "dimensions_m": target_dimensions,
                "authored_scale_xyz": list(_IDENTITY_SCALE),
                "metric_scale_baked_into_points": True,
                "source_xform_flattened": True,
                "source_world_bounds_center_m": source_world_center,
                "recentered_before_scale": True,
                "aabb_center_m": [0.0, 0.0, 0.0],
                "authored_pivot_m": [0.0, 0.0, 0.0],
                "placement_origin_semantics": ("body_pose_translation_is_replacement_aabb_center"),
                "point_positions_sha256_required": True,
                "closed_volume_m3": expected_baked_volume_m3,
                "closed_volume_tolerance_m3": volume_tolerance_m3,
                "dimension_tolerance_m": _DIMENSION_TOLERANCE_M,
            },
            "material_binding": {
                "visual_prim_path": OUTPUT_VISUAL_PRIM_PATH,
                "material_prim_paths": sorted(material_prim_path_map.values()),
                "texture_asset_paths": sorted(
                    f"textures/{row['relative_path']}" for row in texture_rows
                ),
            },
            "authoring_root_prim_path": OUTPUT_BODY_PRIM_PATH,
            "deformable_schema_prim_path": OUTPUT_BODY_PRIM_PATH,
            "body_api_schemas": sorted(DEFORMABLE_BODY_SCHEMAS),
            "physics_material": {
                "prim_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
                "api_schemas": sorted(DEFORMABLE_MATERIAL_SCHEMAS),
                "properties": normalized_physics["material_properties"],
            },
            "mass_properties": {
                "density_kg_m3": float(normalized_physics["material_properties"]["density"]),
                "closed_volume_m3": expected_baked_volume_m3,
                "derived_mass_kg": expected_mass_kg,
                "mass_tolerance_kg": mass_tolerance_kg,
                "development_configuration_not_observed_material_truth": True,
            },
            "physics_material_binding": {
                "prim_path": OUTPUT_BODY_PRIM_PATH,
                "material_prim_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
                "material_purpose": "physics",
                "binding_strength": "strongerThanDescendants",
            },
            "simulation_topology": {
                "node_count_minimum": 1,
                "element_count_minimum": 1,
                "topology_sha256_required": True,
            },
            "collision_topology": {
                "node_count_minimum": 1,
                "element_count_minimum": 1,
                "topology_sha256_required": True,
            },
            "forbidden_experimental_api_schemas": sorted(
                {row["schema"] for row in experimental_schemas}
            ),
            "empty_tet_mesh_prim_paths": [],
            "guide_prim_paths": [],
            "light_prim_paths": [],
            "source_provider_prim_paths": [],
            "source_provider_attributes": [],
            "physics_configuration": normalized_physics,
            "texture_inventory": [
                {
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
                for row in texture_rows
            ],
        },
        "claim_boundary": {
            "verified_external_static_inspection_consumed": True,
            "local_source_package_constructed": False,
            "native_worker_executed": False,
            "native_cook_qualified": False,
            "native_simulator_qualified": False,
            "physical_material_equivalence": False,
        },
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def _verify_plan(value: Mapping[str, Any], *, expected_plan_digest: str) -> dict[str, Any]:
    plan = _json_clone(value, error="native_deformable_preparation_plan_invalid")
    errors: list[str] = []
    if not _valid_digest(expected_plan_digest) or plan.get("plan_digest") != expected_plan_digest:
        errors.append("native_deformable_preparation_expected_plan_digest_mismatch")
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        errors.append("native_deformable_preparation_plan_digest_invalid")
    if (
        set(plan)
        != {
            "schema_version",
            "preparation_id",
            "inspection_id",
            "inspection_receipt_digest",
            "source_asset",
            "source_surface_mesh",
            "textures",
            "clean_stage_rebuild",
            "source_content_exclusions",
            "native_runtime",
            "physics_configuration",
            "required_native_readback",
            "claim_boundary",
            "plan_digest",
        }
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        errors.append("native_deformable_preparation_plan_fields_invalid")
    if (
        not str(plan.get("preparation_id") or "").strip()
        or not _valid_digest(plan.get("inspection_id"))
        or plan.get("inspection_id") != plan.get("inspection_receipt_digest")
    ):
        errors.append("native_deformable_preparation_inspection_join_invalid")

    source = _mapping(
        plan.get("source_asset"),
        error="native_deformable_preparation_source_asset_invalid",
        errors=errors,
    )
    if set(source) != {
        "usd_path",
        "usd_package_path",
        "usd_sha256",
        "usd_size_bytes",
        "texture_root_path",
        "meters_per_unit",
        "up_axis",
        "source_archive_sha256",
        "source_expanded_tree_digest",
    }:
        errors.append("native_deformable_preparation_source_asset_fields_invalid")
    source_usd_path = Path(str(source.get("usd_path") or "")).expanduser().absolute()
    texture_root_path = Path(str(source.get("texture_root_path") or "")).expanduser().absolute()
    if (
        source.get("usd_path") != str(source_usd_path)
        or source.get("texture_root_path") != str(texture_root_path)
        or source.get("usd_package_path") != SOURCE_USD_PACKAGE_PATH
        or not _valid_digest(source.get("usd_sha256"))
        or isinstance(source.get("usd_size_bytes"), bool)
        or not isinstance(source.get("usd_size_bytes"), int)
        or not 0 < source.get("usd_size_bytes", 0) <= _MAX_SOURCE_FILE_BYTES
        or source.get("meters_per_unit") != 1.0
        or source.get("up_axis") != "Z"
        or not _valid_digest(source.get("source_archive_sha256"))
        or not _valid_digest(source.get("source_expanded_tree_digest"))
    ):
        errors.append("native_deformable_preparation_source_asset_contract_invalid")

    surface = _mapping(
        plan.get("source_surface_mesh"),
        error="native_deformable_preparation_source_surface_invalid",
        errors=errors,
    )
    if set(surface) != {
        "prim_path",
        "point_count",
        "triangle_count",
        "dimensions_m",
        "world_bounds_minimum_m",
        "world_bounds_maximum_m",
        "world_bounds_center_m",
        "closed_surface_volume_m3",
        "closed_surface",
        "manifold_surface",
        "material_prim_paths",
    }:
        errors.append("native_deformable_preparation_source_surface_fields_invalid")
    source_dimensions = _positive_vector3(
        surface.get("dimensions_m"),
        error="native_deformable_preparation_source_dimensions_invalid",
        errors=errors,
    )
    source_volume = surface.get("closed_surface_volume_m3")
    source_minimum = _finite_vector3(
        surface.get("world_bounds_minimum_m"),
        error="native_deformable_preparation_source_minimum_invalid",
        errors=errors,
    )
    source_maximum = _finite_vector3(
        surface.get("world_bounds_maximum_m"),
        error="native_deformable_preparation_source_maximum_invalid",
        errors=errors,
    )
    source_center = _finite_vector3(
        surface.get("world_bounds_center_m"),
        error="native_deformable_preparation_source_center_invalid",
        errors=errors,
    )
    material_prim_paths = _strings(
        surface.get("material_prim_paths"),
        error="native_deformable_preparation_material_prim_paths_invalid",
        errors=errors,
    )
    if (
        not str(surface.get("prim_path") or "").startswith("/")
        or _positive_int(
            surface.get("point_count"),
            error="native_deformable_preparation_source_point_count_invalid",
            errors=errors,
        )
        == 0
        or _positive_int(
            surface.get("triangle_count"),
            error="native_deformable_preparation_source_triangle_count_invalid",
            errors=errors,
        )
        == 0
        or isinstance(source_volume, bool)
        or not isinstance(source_volume, (int, float))
        or not math.isfinite(float(source_volume or 0.0))
        or float(source_volume or 0.0) <= 0.0
        or surface.get("closed_surface") is not True
        or surface.get("manifold_surface") is not True
        or len(material_prim_paths) != 1
        or any(not path.startswith("/") for path in material_prim_paths)
        or len(source_dimensions) != 3
        or len(source_minimum) != 3
        or len(source_maximum) != 3
        or len(source_center) != 3
        or any(
            abs(source_maximum[index] - source_minimum[index] - source_dimensions[index]) > 1.0e-12
            or abs((source_minimum[index] + source_maximum[index]) / 2.0 - source_center[index])
            > 1.0e-12
            for index in range(3)
        )
    ):
        errors.append("native_deformable_preparation_source_surface_contract_invalid")

    texture_rows = _rows(
        plan.get("textures"),
        error="native_deformable_preparation_textures_invalid",
        errors=errors,
    )
    if not texture_rows or len(texture_rows) > _MAX_TEXTURE_COUNT:
        errors.append("native_deformable_preparation_texture_count_invalid")
    normalized_texture_rows: list[dict[str, Any]] = []
    texture_total = 0
    seen_texture_paths: set[str] = set()
    for index, row in enumerate(texture_rows[:_MAX_TEXTURE_COUNT]):
        relative = _safe_relative_path(
            row.get("relative_path"),
            error=f"native_deformable_preparation_texture_path_invalid:{index}",
            errors=errors,
        )
        expected_source_path = texture_root_path.joinpath(*PurePosixPath(relative).parts)
        expected_package_path = (SOURCE_TEXTURE_PACKAGE_ROOT / PurePosixPath(relative)).as_posix()
        size_value = row.get("size_bytes")
        if (
            set(row) != {"relative_path", "source_path", "package_path", "sha256", "size_bytes"}
            or relative in seen_texture_paths
            or row.get("source_path") != str(expected_source_path)
            or row.get("package_path") != expected_package_path
            or not _valid_digest(row.get("sha256"))
            or isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or not 0 < size_value <= _MAX_SOURCE_FILE_BYTES
        ):
            errors.append(f"native_deformable_preparation_texture_contract_invalid:{index}")
        if isinstance(size_value, int) and not isinstance(size_value, bool):
            texture_total += size_value
        seen_texture_paths.add(relative)
        normalized_texture_rows.append(dict(row))
    if texture_total > _MAX_TEXTURE_TOTAL_BYTES:
        errors.append("native_deformable_preparation_texture_total_size_exceeded")
    if normalized_texture_rows != sorted(
        normalized_texture_rows, key=lambda row: row["relative_path"]
    ):
        errors.append("native_deformable_preparation_texture_order_invalid")

    physics_errors: list[str] = []
    physics = _normalize_physics_configuration(
        plan.get("physics_configuration"), errors=physics_errors
    )
    if physics_errors or physics != plan.get("physics_configuration"):
        errors.append("native_deformable_preparation_physics_contract_invalid")
    rebuild = _mapping(
        plan.get("clean_stage_rebuild"),
        error="native_deformable_preparation_rebuild_invalid",
        errors=errors,
    )
    expected_rebuild_fields = {
        "strategy",
        "output_usd_package_path",
        "output_default_prim_path",
        "output_visual_prim_path",
        "output_looks_prim_path",
        "material_prim_path_map",
        "output_meters_per_unit",
        "output_up_axis",
        "flatten_source_xform_to_points",
        "recenter_source_world_bounds_to_output_origin",
        "source_world_bounds_center_m",
        "output_authored_pivot_m",
        "placement_origin_semantics",
        "bake_metric_scale_into_points",
        "point_bake_scale_xyz",
        "point_bake_scale_determinant",
        "target_metric_dimensions_m",
        "expected_baked_closed_volume_m3",
        "expected_density_kg_m3",
        "expected_mass_kg",
        "authored_visual_scale_xyz_after_bake",
        "copy_source_prim_subtree",
        "copy_source_api_schemas",
        "copy_empty_source_tet_meshes",
        "copy_guides",
        "copy_lights",
    }
    if set(rebuild) != expected_rebuild_fields:
        errors.append("native_deformable_preparation_rebuild_fields_invalid")
    target_dimensions = _positive_vector3(
        rebuild.get("target_metric_dimensions_m"),
        error="native_deformable_preparation_target_dimensions_invalid",
        errors=errors,
    )
    bake_scale = _positive_vector3(
        rebuild.get("point_bake_scale_xyz"),
        error="native_deformable_preparation_bake_scale_invalid",
        errors=errors,
    )
    expected_scale = (
        [target_dimensions[index] / source_dimensions[index] for index in range(3)]
        if len(target_dimensions) == len(source_dimensions) == 3
        else []
    )
    expected_determinant = math.prod(expected_scale) if expected_scale else 0.0
    source_volume_number = (
        float(source_volume)
        if isinstance(source_volume, (int, float))
        and not isinstance(source_volume, bool)
        and math.isfinite(float(source_volume))
        else 0.0
    )
    expected_volume = source_volume_number * expected_determinant
    density = physics.get("material_properties", {}).get("density", 0.0)
    density_number = (
        float(density)
        if isinstance(density, (int, float))
        and not isinstance(density, bool)
        and math.isfinite(float(density))
        else 0.0
    )
    expected_mass = expected_volume * density_number
    expected_material_map = {
        path: f"{OUTPUT_LOOKS_PRIM_PATH}/Material_{index:03d}"
        for index, path in enumerate(sorted(material_prim_paths))
    }
    if (
        rebuild.get("strategy") != "allowlisted_surface_and_bound_material_reconstruction"
        or rebuild.get("output_usd_package_path") != OUTPUT_USD_PACKAGE_PATH
        or rebuild.get("output_default_prim_path") != OUTPUT_BODY_PRIM_PATH
        or rebuild.get("output_visual_prim_path") != OUTPUT_VISUAL_PRIM_PATH
        or rebuild.get("output_looks_prim_path") != OUTPUT_LOOKS_PRIM_PATH
        or rebuild.get("material_prim_path_map") != expected_material_map
        or rebuild.get("output_meters_per_unit") != 1.0
        or rebuild.get("output_up_axis") != "Z"
        or rebuild.get("flatten_source_xform_to_points") is not True
        or rebuild.get("recenter_source_world_bounds_to_output_origin") is not True
        or rebuild.get("source_world_bounds_center_m") != source_center
        or rebuild.get("output_authored_pivot_m") != [0.0, 0.0, 0.0]
        or rebuild.get("placement_origin_semantics")
        != "body_pose_translation_is_replacement_aabb_center"
        or rebuild.get("bake_metric_scale_into_points") is not True
        or rebuild.get("authored_visual_scale_xyz_after_bake") != _IDENTITY_SCALE
        or any(
            rebuild.get(field) is not False
            for field in (
                "copy_source_prim_subtree",
                "copy_source_api_schemas",
                "copy_empty_source_tet_meshes",
                "copy_guides",
                "copy_lights",
            )
        )
        or len(bake_scale) != 3
        or any(
            abs(bake_scale[index] - expected_scale[index]) > 1.0e-12
            for index in range(len(expected_scale))
        )
        or not isinstance(rebuild.get("point_bake_scale_determinant"), (int, float))
        or isinstance(rebuild.get("point_bake_scale_determinant"), bool)
        or abs(float(rebuild.get("point_bake_scale_determinant")) - expected_determinant) > 1.0e-12
        or not isinstance(rebuild.get("expected_baked_closed_volume_m3"), (int, float))
        or isinstance(rebuild.get("expected_baked_closed_volume_m3"), bool)
        or abs(float(rebuild.get("expected_baked_closed_volume_m3")) - expected_volume)
        > max(1.0e-12, expected_volume * 1.0e-12)
        or rebuild.get("expected_density_kg_m3") != density_number
        or not isinstance(rebuild.get("expected_mass_kg"), (int, float))
        or isinstance(rebuild.get("expected_mass_kg"), bool)
        or abs(float(rebuild.get("expected_mass_kg")) - expected_mass)
        > max(1.0e-12, expected_mass * 1.0e-12)
    ):
        errors.append("native_deformable_preparation_rebuild_contract_invalid")

    material_arguments = {
        "prim_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
        "cfg_kwargs": physics.get("material_properties"),
    }
    body_arguments = {
        "prim_path": OUTPUT_BODY_PRIM_PATH,
        "cfg_kwargs": physics.get("body_properties", {}),
    }
    physics_binding_arguments = {
        "prim_path": OUTPUT_BODY_PRIM_PATH,
        "material_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
        "stronger_than_descendants": True,
    }
    expected_calls = [
        {
            "symbol": DEFORMABLE_MATERIAL_API,
            "configuration_symbol": DEFORMABLE_MATERIAL_CFG,
            "arguments": material_arguments,
            "arguments_digest": canonical_digest(material_arguments),
        },
        {
            "symbol": DEFORMABLE_AUTHORING_API,
            "configuration_symbol": DEFORMABLE_BODY_CFG,
            "arguments": body_arguments,
            "arguments_digest": canonical_digest(body_arguments),
        },
        {
            "symbol": DEFORMABLE_PHYSICS_BINDING_API,
            "configuration_symbol": None,
            "arguments": physics_binding_arguments,
            "arguments_digest": canonical_digest(physics_binding_arguments),
        },
    ]
    expected_runtime = {
        "source_repository": ISAACLAB_REPOSITORY,
        "source_revision": ISAACLAB_COMMIT,
        "source_tree": ISAACLAB_TREE,
        "pinned_source_call_contract": _json_clone(
            PINNED_NATIVE_CALL_CONTRACT,
            error="native_deformable_pinned_call_contract_not_json",
        ),
        "required_api_symbols": list(NATIVE_REQUIRED_API_SYMBOLS),
        "executed_api_symbols": list(NATIVE_EXECUTED_API_SYMBOLS),
        "embedded_cooking_contract": {
            "owner_symbol": DEFORMABLE_AUTHORING_API,
            "legacy_external_cooking_symbol_not_required": DEFORMABLE_COOKING_API,
            "direct_cooking_call_forbidden": True,
            "body_cfg_constructor_receives_cooking_fields": False,
            "source_cooking_properties_recorded_not_constructor_kwargs": True,
            "pinned_authoring_return": None,
        },
        "api_calls_in_order": expected_calls,
    }
    if plan.get("native_runtime") != expected_runtime:
        errors.append("native_deformable_preparation_runtime_contract_invalid")

    exclusions = _mapping(
        plan.get("source_content_exclusions"),
        error="native_deformable_preparation_exclusions_invalid",
        errors=errors,
    )
    if set(exclusions) != {
        "experimental_api_schemas",
        "provider_authored_attributes",
        "empty_tet_mesh_prim_paths",
        "guide_prim_paths",
        "light_prim_paths",
    }:
        errors.append("native_deformable_preparation_exclusion_fields_invalid")
    schema_errors: list[str] = []
    normalized_schemas = _normalize_schema_rows(
        exclusions.get("experimental_api_schemas"), errors=schema_errors
    )
    if schema_errors or normalized_schemas != exclusions.get("experimental_api_schemas"):
        errors.append("native_deformable_preparation_experimental_schemas_invalid")
    provider_rows = _rows(
        exclusions.get("provider_authored_attributes"),
        error="native_deformable_preparation_provider_attributes_invalid",
        errors=errors,
    )
    normalized_provider_rows: list[dict[str, str]] = []
    for index, row in enumerate(provider_rows):
        normalized_row = {
            "prim_path": str(row.get("prim_path") or ""),
            "name": str(row.get("name") or ""),
        }
        if (
            set(row) != {"prim_path", "name"}
            or not normalized_row["prim_path"].startswith("/")
            or not normalized_row["name"]
        ):
            errors.append(f"native_deformable_preparation_provider_attribute_invalid:{index}")
        normalized_provider_rows.append(normalized_row)
    if normalized_provider_rows != sorted(
        normalized_provider_rows, key=lambda row: (row["prim_path"], row["name"])
    ) or len({(row["prim_path"], row["name"]) for row in normalized_provider_rows}) != len(
        normalized_provider_rows
    ):
        errors.append("native_deformable_preparation_provider_attribute_order_invalid")
    for field in ("empty_tet_mesh_prim_paths", "guide_prim_paths", "light_prim_paths"):
        paths = _strings(
            exclusions.get(field),
            error=f"native_deformable_preparation_{field}_invalid",
            errors=errors,
        )
        if paths != sorted(paths) or any(not path.startswith("/") for path in paths):
            errors.append(f"native_deformable_preparation_{field}_order_invalid")

    expected_texture_inventory = [
        {
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in normalized_texture_rows
    ]
    expected_readback = {
        "stage_metadata": {
            "default_prim_path": OUTPUT_BODY_PRIM_PATH,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "visual_mesh": {
            "prim_path": OUTPUT_VISUAL_PRIM_PATH,
            "point_count": surface.get("point_count"),
            "triangle_count": surface.get("triangle_count"),
            "source_face_topology_sha256_required": True,
            "output_face_topology_sha256_must_match_source": True,
            "dimensions_m": target_dimensions,
            "authored_scale_xyz": list(_IDENTITY_SCALE),
            "metric_scale_baked_into_points": True,
            "source_xform_flattened": True,
            "source_world_bounds_center_m": source_center,
            "recentered_before_scale": True,
            "aabb_center_m": [0.0, 0.0, 0.0],
            "authored_pivot_m": [0.0, 0.0, 0.0],
            "placement_origin_semantics": "body_pose_translation_is_replacement_aabb_center",
            "point_positions_sha256_required": True,
            "closed_volume_m3": expected_volume,
            "closed_volume_tolerance_m3": max(1.0e-12, expected_volume * 1.0e-6),
            "dimension_tolerance_m": _DIMENSION_TOLERANCE_M,
        },
        "material_binding": {
            "visual_prim_path": OUTPUT_VISUAL_PRIM_PATH,
            "material_prim_paths": sorted(expected_material_map.values()),
            "texture_asset_paths": sorted(
                f"textures/{row['relative_path']}" for row in normalized_texture_rows
            ),
        },
        "authoring_root_prim_path": OUTPUT_BODY_PRIM_PATH,
        "deformable_schema_prim_path": OUTPUT_BODY_PRIM_PATH,
        "body_api_schemas": sorted(DEFORMABLE_BODY_SCHEMAS),
        "physics_material": {
            "prim_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
            "api_schemas": sorted(DEFORMABLE_MATERIAL_SCHEMAS),
            "properties": physics.get("material_properties"),
        },
        "mass_properties": {
            "density_kg_m3": density_number,
            "closed_volume_m3": expected_volume,
            "derived_mass_kg": expected_mass,
            "mass_tolerance_kg": max(1.0e-12, expected_mass * 1.0e-6),
            "development_configuration_not_observed_material_truth": True,
        },
        "physics_material_binding": {
            "prim_path": OUTPUT_BODY_PRIM_PATH,
            "material_prim_path": OUTPUT_PHYSICS_MATERIAL_PRIM_PATH,
            "material_purpose": "physics",
            "binding_strength": "strongerThanDescendants",
        },
        "simulation_topology": {
            "node_count_minimum": 1,
            "element_count_minimum": 1,
            "topology_sha256_required": True,
        },
        "collision_topology": {
            "node_count_minimum": 1,
            "element_count_minimum": 1,
            "topology_sha256_required": True,
        },
        "forbidden_experimental_api_schemas": sorted({row["schema"] for row in normalized_schemas}),
        "empty_tet_mesh_prim_paths": [],
        "guide_prim_paths": [],
        "light_prim_paths": [],
        "source_provider_prim_paths": [],
        "source_provider_attributes": [],
        "physics_configuration": physics,
        "texture_inventory": expected_texture_inventory,
    }
    if plan.get("required_native_readback") != expected_readback:
        errors.append("native_deformable_preparation_readback_contract_invalid")
    expected_claim_boundary = {
        "verified_external_static_inspection_consumed": True,
        "local_source_package_constructed": False,
        "native_worker_executed": False,
        "native_cook_qualified": False,
        "native_simulator_qualified": False,
        "physical_material_equivalence": False,
    }
    if plan.get("claim_boundary") != expected_claim_boundary:
        errors.append("native_deformable_preparation_claim_boundary_invalid")
    if errors:
        raise NativeDeformableAssetPreparationError(errors)
    return plan


def build_native_deformable_asset_source_package(
    *, output_dir: str | Path, plan: Mapping[str, Any], expected_plan_digest: str
) -> dict[str, Any]:
    """Copy exact inspected inputs to a local, native-worker source package."""

    normalized = _verify_plan(plan, expected_plan_digest=expected_plan_digest)
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_source_package_output_exists"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        inventory: list[dict[str, Any]] = []
        source_rows = [
            {
                "role": "source_usd",
                "source_path": normalized["source_asset"]["usd_path"],
                "package_path": normalized["source_asset"]["usd_package_path"],
                "sha256": normalized["source_asset"]["usd_sha256"],
                "size_bytes": normalized["source_asset"]["usd_size_bytes"],
            },
            *[
                {
                    "role": "texture",
                    "source_path": row["source_path"],
                    "package_path": row["package_path"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
                for row in normalized["textures"]
            ],
        ]
        for index, row in enumerate(source_rows):
            source = Path(row["source_path"])
            content = _read_regular_file_once(
                source,
                maximum_size=_MAX_SOURCE_FILE_BYTES,
                expected_digest=row["sha256"],
                expected_size=row["size_bytes"],
                error=f"native_deformable_source_changed_after_plan:{index}",
            )
            relative = PurePosixPath(row["package_path"])
            destination = staging / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            if (
                destination.stat().st_size != row["size_bytes"]
                or _sha256(destination) != row["sha256"]
            ):
                raise NativeDeformableAssetPreparationError(
                    [f"native_deformable_staged_source_identity_mismatch:{index}"]
                )
            inventory.append(
                {
                    "role": row["role"],
                    "package_path": row["package_path"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
            )

        plan_path = staging / PLAN_FILENAME
        plan_bytes = (
            json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(plan_bytes) > _MAX_RECEIPT_BYTES:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_preparation_plan_size_exceeded"]
            )
        plan_path.write_bytes(plan_bytes)
        inventory.append(
            {
                "role": "preparation_plan",
                "package_path": PLAN_FILENAME,
                "sha256": f"sha256:{hashlib.sha256(plan_bytes).hexdigest()}",
                "size_bytes": len(plan_bytes),
            }
        )
        inventory.sort(key=lambda row: (row["package_path"], row["role"]))
        receipt: dict[str, Any] = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "preparation_id": normalized["preparation_id"],
            "plan_digest": normalized["plan_digest"],
            "files": inventory,
            "package_content_digest": canonical_digest({"files": inventory}),
            "claim_boundary": {
                "exact_inspected_source_bytes_packaged": True,
                "native_worker_executed": False,
                "native_cook_qualified": False,
                "native_simulator_qualified": False,
                "physical_material_equivalence": False,
            },
            "package_root": str(output),
            "receipt_path": str(output / PACKAGE_RECEIPT_FILENAME),
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        receipt_path = staging / PACKAGE_RECEIPT_FILENAME
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _package_file_snapshot(
    package_root: Path,
    relative_path: str,
    *,
    expected_digest: str,
    expected_size: int,
    error: str,
) -> bytes:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in relative_path
    ):
        raise NativeDeformableAssetPreparationError([error])
    path = package_root / Path(*relative.parts)
    return _read_regular_file_once(
        path,
        maximum_size=_MAX_SOURCE_FILE_BYTES,
        expected_digest=expected_digest,
        expected_size=expected_size,
        error=error,
    )


def execute_native_deformable_asset_preparation(
    *,
    plan: Mapping[str, Any],
    expected_plan_digest: str,
    package_root: str | Path,
    output_root: str | Path,
    stage_api: NativeDeformableStageAPI,
    native_api_registry: Mapping[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Execute the frozen plan inside a caller-owned pinned native runtime.

    The returned result contains observations, never a self-authored
    qualification boolean.  Authenticating that this function ran inside the
    admitted Vast/Isaac environment is outside this module.
    """

    normalized = _verify_plan(plan, expected_plan_digest=expected_plan_digest)
    root = Path(package_root).expanduser().absolute()
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise NativeDeformableAssetPreparationError(["native_deformable_preparation_output_exists"])
    if set(native_api_registry) != set(NATIVE_REQUIRED_API_SYMBOLS) or any(
        not callable(value) for value in native_api_registry.values()
    ):
        raise NativeDeformableAssetPreparationError(
            ["native_deformable_preparation_api_registry_invalid"]
        )

    source = normalized["source_asset"]
    source_usd_content = _package_file_snapshot(
        root,
        source["usd_package_path"],
        expected_digest=source["usd_sha256"],
        expected_size=source["usd_size_bytes"],
        error="native_deformable_packaged_source_usd_invalid",
    )
    texture_contents: dict[str, bytes] = {}
    for row in normalized["textures"]:
        texture_contents[row["relative_path"]] = _package_file_snapshot(
            root,
            row["package_path"],
            expected_digest=row["sha256"],
            expected_size=row["size_bytes"],
            error=f"native_deformable_packaged_texture_invalid:{row['relative_path']}",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    stage: object | None = None
    stage_context_active = False
    try:
        source_snapshot_root = staging / ".source_snapshot"
        source_snapshot_root.mkdir()
        source_usd = source_snapshot_root / "asset.usd"
        source_usd.write_bytes(source_usd_content)
        texture_map: dict[str, Path] = {}
        for row in normalized["textures"]:
            snapshot = (
                source_snapshot_root / "textures" / Path(*PurePosixPath(row["relative_path"]).parts)
            )
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(texture_contents[row["relative_path"]])
            texture_map[row["relative_path"]] = snapshot
        prepared_root = staging / "prepared"
        prepared_root.mkdir()
        output_usd = prepared_root / "deformable.usda"
        output_textures: list[dict[str, Any]] = []
        output_texture_asset_paths: dict[str, str] = {}
        for row in normalized["textures"]:
            destination = (
                prepared_root / "textures" / Path(*PurePosixPath(row["relative_path"]).parts)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(texture_contents[row["relative_path"]])
            asset_path = f"textures/{row['relative_path']}"
            output_texture_asset_paths[row["relative_path"]] = asset_path
            output_textures.append(
                {
                    "relative_path": row["relative_path"],
                    "package_path": (
                        PurePosixPath("prepared/textures") / PurePosixPath(row["relative_path"])
                    ).as_posix(),
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
            )
        rebuild = normalized["clean_stage_rebuild"]
        stage = stage_api.create_clean_stage(
            output_path=output_usd,
            default_prim_path=rebuild["output_default_prim_path"],
            meters_per_unit=rebuild["output_meters_per_unit"],
            up_axis=rebuild["output_up_axis"],
        )
        stage_api.copy_surface_mesh_baking_points(
            stage=stage,
            source_usd_path=source_usd,
            source_prim_path=normalized["source_surface_mesh"]["prim_path"],
            output_prim_path=rebuild["output_visual_prim_path"],
            source_world_bounds_center_m=rebuild["source_world_bounds_center_m"],
            recenter_to_output_origin=rebuild["recenter_source_world_bounds_to_output_origin"],
            bake_scale_xyz=rebuild["point_bake_scale_xyz"],
            flatten_source_xform=rebuild["flatten_source_xform_to_points"],
        )
        stage_api.copy_bound_material_network(
            stage=stage,
            source_usd_path=source_usd,
            material_prim_path_map=rebuild["material_prim_path_map"],
            output_looks_prim_path=rebuild["output_looks_prim_path"],
            output_visual_prim_path=rebuild["output_visual_prim_path"],
            source_texture_paths=texture_map,
            output_texture_asset_paths=output_texture_asset_paths,
        )

        # The pinned material spawner has no ``stage`` argument and resolves
        # ``get_current_stage()`` internally.  Require the stage adapter to
        # activate and verify this exact clean output stage before invoking it;
        # the later physics-material readback proves the prim landed here.
        if stage_api.activate_and_verify_current_stage(stage=stage) is not True:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_material_spawn_stage_context_invalid"]
            )
        stage_context_active = True

        api_calls: list[dict[str, Any]] = []
        material_row, authoring_row, binding_row = normalized["native_runtime"][
            "api_calls_in_order"
        ]
        material_cfg = native_api_registry[DEFORMABLE_MATERIAL_CFG](
            **material_row["arguments"]["cfg_kwargs"]
        )
        if material_cfg is None:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_material_cfg_construction_failed"]
            )
        material_prim = native_api_registry[DEFORMABLE_MATERIAL_API](
            prim_path=material_row["arguments"]["prim_path"],
            cfg=material_cfg,
        )
        try:
            material_prim_valid = bool(material_prim)
        except Exception as exc:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_material_api_return_invalid"]
            ) from exc
        if not material_prim_valid:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_material_api_return_invalid"]
            )
        api_calls.append(
            {
                "symbol": DEFORMABLE_MATERIAL_API,
                "configuration_symbol": DEFORMABLE_MATERIAL_CFG,
                "arguments_digest": material_row["arguments_digest"],
                "status": MATERIAL_API_STATUS,
            }
        )
        body_cfg = native_api_registry[DEFORMABLE_BODY_CFG](
            **authoring_row["arguments"]["cfg_kwargs"]
        )
        if body_cfg is None:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_body_cfg_construction_failed"]
            )
        authoring_return = native_api_registry[DEFORMABLE_AUTHORING_API](
            prim_path=authoring_row["arguments"]["prim_path"],
            cfg=body_cfg,
            stage=stage,
        )
        if authoring_return is not None:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_authoring_api_return_contract_invalid"]
            )
        api_calls.append(
            {
                "symbol": DEFORMABLE_AUTHORING_API,
                "configuration_symbol": DEFORMABLE_BODY_CFG,
                "arguments_digest": authoring_row["arguments_digest"],
                "status": AUTHORING_API_STATUS,
            }
        )
        binding_return = native_api_registry[DEFORMABLE_PHYSICS_BINDING_API](
            stage=stage,
            **binding_row["arguments"],
        )
        if binding_return is not None:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_physics_material_binding_return_contract_invalid"]
            )
        api_calls.append(
            {
                "symbol": DEFORMABLE_PHYSICS_BINDING_API,
                "configuration_symbol": None,
                "arguments_digest": binding_row["arguments_digest"],
                "status": PHYSICS_BINDING_API_STATUS,
            }
        )
        stage_api.record_native_configuration(
            stage=stage,
            body_and_cooking_properties={
                **normalized["physics_configuration"]["body_properties"],
                **normalized["physics_configuration"]["cooking_properties"],
            },
            material_properties=material_row["arguments"]["cfg_kwargs"],
        )
        stage_api.save_stage(stage=stage)
        stage_api.release_current_stage(stage=stage)
        stage_context_active = False
        readback = _json_clone(
            stage_api.readback_prepared_stage(
                stage=stage,
                output_authoring_root_prim_path=rebuild["output_default_prim_path"],
                output_deformable_schema_prim_path=rebuild["output_default_prim_path"],
                output_visual_prim_path=rebuild["output_visual_prim_path"],
            ),
            error="native_deformable_preparation_readback_not_json",
        )
        output_usd_content = _read_regular_file_once(
            output_usd,
            maximum_size=_MAX_OUTPUT_TOTAL_BYTES,
            expected_digest=None,
            expected_size=None,
            error="native_deformable_prepared_usd_missing",
        )

        shutil.rmtree(source_snapshot_root)
        result: dict[str, Any] = {
            "schema_version": WORKER_RETURN_SCHEMA_VERSION,
            "preparation_id": normalized["preparation_id"],
            "plan_digest": normalized["plan_digest"],
            "runtime_identity": {
                "source_repository": ISAACLAB_REPOSITORY,
                "source_revision": ISAACLAB_COMMIT,
                "source_tree": ISAACLAB_TREE,
                "pinned_source_call_contract_digest": canonical_digest(PINNED_NATIVE_CALL_CONTRACT),
                "required_api_symbols": list(NATIVE_REQUIRED_API_SYMBOLS),
                "executed_api_symbols": list(NATIVE_EXECUTED_API_SYMBOLS),
                "single_cook_owner_symbol": DEFORMABLE_AUTHORING_API,
            },
            "api_calls": api_calls,
            "output_artifacts": {
                "runtime_usd": {
                    "package_path": OUTPUT_USD_PACKAGE_PATH,
                    "sha256": (f"sha256:{hashlib.sha256(output_usd_content).hexdigest()}"),
                    "size_bytes": len(output_usd_content),
                },
                "textures": sorted(output_textures, key=lambda row: row["relative_path"]),
            },
            "readback": readback,
            "worker_result_digest": "",
        }
        result["worker_result_digest"] = canonical_digest(
            result, digest_field="worker_result_digest"
        )
        result_path = staging / "worker_return.json"
        result_bytes = (
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(result_bytes) > _MAX_WORKER_RETURN_BYTES:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_worker_return_size_exceeded"]
            )
        result_path.write_bytes(result_bytes)
        expected_output_members = {
            "worker_return.json",
            OUTPUT_USD_PACKAGE_PATH,
            *(row["package_path"] for row in output_textures),
        }
        output_snapshot = _snapshot_output_tree(
            staging,
            retained_content_limits={"worker_return.json": _MAX_WORKER_RETURN_BYTES},
            invalid_error="native_deformable_preparation_output_member_set_invalid",
            resource_error="native_deformable_preparation_output_resource_limit_exceeded",
        )
        if set(
            output_snapshot.files
        ) != expected_output_members or output_snapshot.directories != _expected_output_directories(
            expected_output_members
        ):
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_preparation_output_member_set_invalid"]
            )
        _snapshot_member(
            output_snapshot,
            OUTPUT_USD_PACKAGE_PATH,
            expected_digest=result["output_artifacts"]["runtime_usd"]["sha256"],
            expected_size=result["output_artifacts"]["runtime_usd"]["size_bytes"],
            retain_content=False,
            error="native_deformable_prepared_usd_identity_mismatch",
        )
        for index, row in enumerate(output_textures):
            _snapshot_member(
                output_snapshot,
                row["package_path"],
                expected_digest=row["sha256"],
                expected_size=row["size_bytes"],
                retain_content=False,
                error=f"native_deformable_prepared_texture_identity_mismatch:{index}",
            )
        persisted_snapshot = _snapshot_member(
            output_snapshot,
            "worker_return.json",
            expected_digest=f"sha256:{hashlib.sha256(result_bytes).hexdigest()}",
            expected_size=len(result_bytes),
            retain_content=True,
            error="native_deformable_worker_return_persisted_payload_invalid",
        )
        if persisted_snapshot.content != result_bytes:
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_worker_return_persisted_payload_invalid"]
            )
        os.replace(staging, output)
        return result
    except Exception as exc:
        if stage is not None and stage_context_active:
            try:
                stage_api.release_current_stage(stage=stage)
            except Exception as cleanup_exc:
                shutil.rmtree(staging, ignore_errors=True)
                errors = ["native_deformable_stage_context_release_failed"]
                if isinstance(exc, NativeDeformableAssetPreparationError):
                    errors.extend(exc.errors)
                raise NativeDeformableAssetPreparationError(errors) from cleanup_exc
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_topology_readback(value: Any, *, label: str, errors: list[str]) -> dict[str, Any]:
    source = _mapping(
        value,
        error=f"native_deformable_return_{label}_topology_invalid",
        errors=errors,
    )
    if set(source) != {"node_count", "element_count", "topology_sha256"}:
        errors.append(f"native_deformable_return_{label}_topology_fields_invalid")
    node_count = _positive_int(
        source.get("node_count"),
        error=f"native_deformable_return_{label}_node_count_invalid",
        errors=errors,
    )
    element_count = _positive_int(
        source.get("element_count"),
        error=f"native_deformable_return_{label}_element_count_invalid",
        errors=errors,
    )
    digest = source.get("topology_sha256")
    if not _valid_digest(digest):
        errors.append(f"native_deformable_return_{label}_topology_digest_invalid")
    return {
        "node_count": node_count,
        "element_count": element_count,
        "topology_sha256": digest,
    }


def _output_relative_path(value: Any, *, error: str) -> str:
    text = str(value or "")
    relative = PurePosixPath(text)
    if (
        not text
        or text != relative.as_posix()
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in text
        or len(os.fsencode(text)) > _MAX_OUTPUT_RELATIVE_PATH_BYTES
    ):
        raise NativeDeformableAssetPreparationError([error])
    return text


def _output_snapshot_descriptor_cap(*, resource_error: str) -> int:
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        raise NativeDeformableAssetPreparationError([resource_error]) from exc
    if soft_limit == resource.RLIM_INFINITY:
        usable_limit = _MAX_OUTPUT_SNAPSHOT_DESCRIPTOR_COUNT
    else:
        usable_limit = max(0, int(soft_limit) - _OUTPUT_SNAPSHOT_DESCRIPTOR_RESERVE)
    return min(_MAX_OUTPUT_SNAPSHOT_DESCRIPTOR_COUNT, usable_limit)


def _snapshot_output_tree(
    output: Path,
    *,
    retained_content_limits: Mapping[str, int],
    invalid_error: str,
    resource_error: str,
) -> _OutputTreeSnapshot:
    """Read one bounded tree while retaining every observed descriptor.

    A pathname-anchored walk alone is not a coherent multi-file snapshot: an
    earlier file can be changed in place while a later file is read without
    changing either directory's entries.  Every file and directory descriptor
    therefore remains open until a whole-tree identity and digest recheck has
    completed.  The retained-descriptor count is capped by both this contract
    and the process ``RLIMIT_NOFILE`` reserve.
    """

    descriptor_cap = _output_snapshot_descriptor_cap(resource_error=resource_error)
    if descriptor_cap < 1:
        raise NativeDeformableAssetPreparationError([resource_error])
    try:
        root_descriptor = _open_directory_descriptor(output, error=invalid_error)
    except NativeDeformableAssetPreparationError as exc:
        if isinstance(exc.__cause__, OSError) and exc.__cause__.errno in {
            errno.EMFILE,
            errno.ENFILE,
        }:
            raise NativeDeformableAssetPreparationError([resource_error]) from exc
        raise
    try:
        root_identity = os.fstat(root_descriptor)
    except OSError as exc:
        os.close(root_descriptor)
        raise NativeDeformableAssetPreparationError([invalid_error]) from exc
    files: dict[str, _RegularFileSnapshot] = {}
    directories: set[str] = set()
    counters = {"entries": 0, "directories": 0, "files": 0, "bytes": 0}
    held_files: list[_HeldRegularFile] = []
    held_directories: list[_HeldDirectory] = [
        _HeldDirectory(
            descriptor=root_descriptor,
            parent_descriptor=None,
            name=None,
            identity=root_identity,
        )
    ]

    def require_descriptor_slot() -> None:
        if len(held_files) + len(held_directories) >= descriptor_cap:
            raise NativeDeformableAssetPreparationError([resource_error])

    def translate_open_error(exc: OSError) -> NativeDeformableAssetPreparationError:
        error = resource_error if exc.errno in {errno.EMFILE, errno.ENFILE} else invalid_error
        return NativeDeformableAssetPreparationError([error])

    def walk(directory_descriptor: int, prefix: PurePosixPath) -> None:
        directory_before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise NativeDeformableAssetPreparationError([invalid_error])
        names: list[str] = []
        try:
            with os.scandir(directory_descriptor) as iterator:
                for entry in iterator:
                    name = entry.name
                    if (
                        not isinstance(name, str)
                        or name in {"", ".", ".."}
                        or os.sep in name
                        or "\\" in name
                    ):
                        raise NativeDeformableAssetPreparationError([invalid_error])
                    counters["entries"] += 1
                    if counters["entries"] > _MAX_OUTPUT_ENTRY_COUNT:
                        raise NativeDeformableAssetPreparationError([resource_error])
                    names.append(name)
        except NativeDeformableAssetPreparationError:
            raise
        except OSError as exc:
            raise translate_open_error(exc) from exc
        except (TypeError, NotImplementedError) as exc:
            raise NativeDeformableAssetPreparationError([invalid_error]) from exc

        for name in sorted(names):
            relative = (prefix / name) if prefix.parts else PurePosixPath(name)
            relative_text = relative.as_posix()
            if (
                len(relative.parts) > _MAX_OUTPUT_DEPTH
                or len(os.fsencode(relative_text)) > _MAX_OUTPUT_RELATIVE_PATH_BYTES
            ):
                raise NativeDeformableAssetPreparationError([resource_error])
            try:
                entry_before = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except (OSError, TypeError, NotImplementedError) as exc:
                raise NativeDeformableAssetPreparationError([invalid_error]) from exc
            if stat.S_ISDIR(entry_before.st_mode):
                counters["directories"] += 1
                if counters["directories"] > _MAX_OUTPUT_DIRECTORY_COUNT:
                    raise NativeDeformableAssetPreparationError([resource_error])
                require_descriptor_slot()
                child_descriptor = -1
                try:
                    child_descriptor = os.open(
                        name,
                        _descriptor_flags(directory=True, error=invalid_error),
                        dir_fd=directory_descriptor,
                    )
                    opened = os.fstat(child_descriptor)
                    entry_after_open = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or not _same_identity(entry_before, opened)
                        or not _same_identity(opened, entry_after_open)
                    ):
                        raise NativeDeformableAssetPreparationError([invalid_error])
                    directories.add(relative_text)
                    held_directories.append(
                        _HeldDirectory(
                            descriptor=child_descriptor,
                            parent_descriptor=directory_descriptor,
                            name=name,
                            identity=opened,
                        )
                    )
                    child_descriptor = -1
                    held_child_descriptor = held_directories[-1].descriptor
                    walk(held_child_descriptor, relative)
                    entry_after_walk = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if not _same_identity(os.fstat(held_child_descriptor), entry_after_walk):
                        raise NativeDeformableAssetPreparationError([invalid_error])
                except NativeDeformableAssetPreparationError:
                    raise
                except OSError as exc:
                    raise translate_open_error(exc) from exc
                except (TypeError, NotImplementedError) as exc:
                    raise NativeDeformableAssetPreparationError([invalid_error]) from exc
                finally:
                    if child_descriptor >= 0:
                        os.close(child_descriptor)
                continue
            if not stat.S_ISREG(entry_before.st_mode):
                raise NativeDeformableAssetPreparationError([invalid_error])
            counters["files"] += 1
            if counters["files"] > _MAX_OUTPUT_FILE_COUNT:
                raise NativeDeformableAssetPreparationError([resource_error])
            require_descriptor_slot()
            remaining_bytes = _MAX_OUTPUT_TOTAL_BYTES - counters["bytes"]
            if entry_before.st_size <= 0:
                raise NativeDeformableAssetPreparationError([invalid_error])
            if entry_before.st_size > remaining_bytes:
                raise NativeDeformableAssetPreparationError([resource_error])
            file_descriptor = -1
            try:
                retained_content_limit = retained_content_limits.get(relative_text)
                if retained_content_limit is not None and (
                    isinstance(retained_content_limit, bool)
                    or not isinstance(retained_content_limit, int)
                    or retained_content_limit <= 0
                ):
                    raise NativeDeformableAssetPreparationError([invalid_error])
                file_descriptor = os.open(
                    name,
                    _descriptor_flags(directory=False, error=invalid_error),
                    dir_fd=directory_descriptor,
                )
                opened = os.fstat(file_descriptor)
                entry_after_open = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not _same_identity(entry_before, opened) or not _same_identity(
                    opened, entry_after_open
                ):
                    raise NativeDeformableAssetPreparationError([invalid_error])
                snapshot = _snapshot_open_regular_file(
                    file_descriptor=file_descriptor,
                    parent_descriptor=directory_descriptor,
                    name=name,
                    maximum_size=remaining_bytes,
                    expected_digest=None,
                    expected_size=entry_before.st_size,
                    retain_content=(
                        retained_content_limit is not None
                        and entry_before.st_size <= retained_content_limit
                    ),
                    error=invalid_error,
                )
                stable_identity = os.fstat(file_descriptor)
                held_files.append(
                    _HeldRegularFile(
                        descriptor=file_descriptor,
                        parent_descriptor=directory_descriptor,
                        name=name,
                        identity=stable_identity,
                        snapshot=snapshot,
                    )
                )
                file_descriptor = -1
            except NativeDeformableAssetPreparationError:
                raise
            except OSError as exc:
                raise translate_open_error(exc) from exc
            except (TypeError, NotImplementedError) as exc:
                raise NativeDeformableAssetPreparationError([invalid_error]) from exc
            finally:
                if file_descriptor >= 0:
                    os.close(file_descriptor)
            counters["bytes"] += snapshot.size_bytes
            files[relative_text] = snapshot

        directory_after = os.fstat(directory_descriptor)
        if not _same_identity(directory_before, directory_after):
            raise NativeDeformableAssetPreparationError([invalid_error])

    try:
        walk(root_descriptor, PurePosixPath())
        for held in held_files:
            if not _same_identity(held.identity, os.fstat(held.descriptor)):
                raise NativeDeformableAssetPreparationError([invalid_error])
            os.lseek(held.descriptor, 0, os.SEEK_SET)
            rechecked = _snapshot_open_regular_file(
                file_descriptor=held.descriptor,
                parent_descriptor=held.parent_descriptor,
                name=held.name,
                maximum_size=held.snapshot.size_bytes,
                expected_digest=held.snapshot.sha256,
                expected_size=held.snapshot.size_bytes,
                retain_content=False,
                error=invalid_error,
            )
            if (
                rechecked.sha256 != held.snapshot.sha256
                or rechecked.size_bytes != held.snapshot.size_bytes
            ):
                raise NativeDeformableAssetPreparationError([invalid_error])
        for held in held_files:
            current = os.fstat(held.descriptor)
            current_entry = os.stat(
                held.name,
                dir_fd=held.parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_identity(held.identity, current) or not _same_identity(
                current, current_entry
            ):
                raise NativeDeformableAssetPreparationError([invalid_error])
        for held in held_directories:
            current = os.fstat(held.descriptor)
            if not _same_identity(held.identity, current):
                raise NativeDeformableAssetPreparationError([invalid_error])
            if held.parent_descriptor is not None and held.name is not None:
                current_entry = os.stat(
                    held.name,
                    dir_fd=held.parent_descriptor,
                    follow_symlinks=False,
                )
                if not _same_identity(current, current_entry):
                    raise NativeDeformableAssetPreparationError([invalid_error])
        return _OutputTreeSnapshot(files=dict(files), directories=frozenset(directories))
    except NativeDeformableAssetPreparationError:
        raise
    except (OSError, TypeError, NotImplementedError) as exc:
        raise NativeDeformableAssetPreparationError([invalid_error]) from exc
    finally:
        for held in reversed(held_files):
            try:
                os.close(held.descriptor)
            except OSError:
                pass
        for held in reversed(held_directories):
            try:
                os.close(held.descriptor)
            except OSError:
                pass


def _expected_output_directories(file_paths: set[str]) -> frozenset[str]:
    directories: set[str] = set()
    for file_path in file_paths:
        parent = PurePosixPath(file_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(directories)


def _snapshot_member(
    snapshot: _OutputTreeSnapshot,
    relative_path: Any,
    *,
    expected_digest: Any,
    expected_size: Any,
    retain_content: bool,
    error: str,
) -> _RegularFileSnapshot:
    normalized = _output_relative_path(relative_path, error=error)
    member = snapshot.files.get(normalized)
    if (
        member is None
        or not _valid_digest(expected_digest)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or member.sha256 != expected_digest
        or member.size_bytes != expected_size
        or (retain_content and member.content is None)
    ):
        raise NativeDeformableAssetPreparationError([error])
    return member


def verify_native_deformable_asset_preparation_return(
    *,
    plan: Mapping[str, Any],
    expected_plan_digest: str,
    worker_return: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify worker payload/readback without authenticating native execution."""

    normalized = _verify_plan(plan, expected_plan_digest=expected_plan_digest)
    returned = _json_clone(
        worker_return,
        error="native_deformable_preparation_worker_return_not_json",
    )
    errors: list[str] = []
    if set(returned) != {
        "schema_version",
        "preparation_id",
        "plan_digest",
        "runtime_identity",
        "api_calls",
        "output_artifacts",
        "readback",
        "worker_result_digest",
    }:
        errors.append("native_deformable_return_fields_invalid")
    if returned.get("schema_version") != WORKER_RETURN_SCHEMA_VERSION:
        errors.append("native_deformable_return_schema_unsupported")
    if returned.get("preparation_id") != normalized["preparation_id"]:
        errors.append("native_deformable_return_preparation_id_mismatch")
    if returned.get("plan_digest") != normalized["plan_digest"]:
        errors.append("native_deformable_return_plan_digest_mismatch")
    if returned.get("worker_result_digest") != canonical_digest(
        returned, digest_field="worker_result_digest"
    ):
        errors.append("native_deformable_return_digest_invalid")
    expected_runtime = {
        "source_repository": ISAACLAB_REPOSITORY,
        "source_revision": ISAACLAB_COMMIT,
        "source_tree": ISAACLAB_TREE,
        "pinned_source_call_contract_digest": canonical_digest(PINNED_NATIVE_CALL_CONTRACT),
        "required_api_symbols": list(NATIVE_REQUIRED_API_SYMBOLS),
        "executed_api_symbols": list(NATIVE_EXECUTED_API_SYMBOLS),
        "single_cook_owner_symbol": DEFORMABLE_AUTHORING_API,
    }
    if returned.get("runtime_identity") != expected_runtime:
        errors.append("native_deformable_return_runtime_identity_mismatch")
    expected_calls = [
        {
            "symbol": normalized["native_runtime"]["api_calls_in_order"][0]["symbol"],
            "configuration_symbol": DEFORMABLE_MATERIAL_CFG,
            "arguments_digest": normalized["native_runtime"]["api_calls_in_order"][0][
                "arguments_digest"
            ],
            "status": MATERIAL_API_STATUS,
        },
        {
            "symbol": normalized["native_runtime"]["api_calls_in_order"][1]["symbol"],
            "configuration_symbol": DEFORMABLE_BODY_CFG,
            "arguments_digest": normalized["native_runtime"]["api_calls_in_order"][1][
                "arguments_digest"
            ],
            "status": AUTHORING_API_STATUS,
        },
        {
            "symbol": normalized["native_runtime"]["api_calls_in_order"][2]["symbol"],
            "configuration_symbol": None,
            "arguments_digest": normalized["native_runtime"]["api_calls_in_order"][2][
                "arguments_digest"
            ],
            "status": PHYSICS_BINDING_API_STATUS,
        },
    ]
    if returned.get("api_calls") != expected_calls:
        errors.append("native_deformable_return_api_calls_mismatch")

    artifacts = _mapping(
        returned.get("output_artifacts"),
        error="native_deformable_return_artifacts_invalid",
        errors=errors,
    )
    if set(artifacts) != {"runtime_usd", "textures"}:
        errors.append("native_deformable_return_artifact_fields_invalid")
    runtime_usd = _mapping(
        artifacts.get("runtime_usd"),
        error="native_deformable_return_runtime_usd_invalid",
        errors=errors,
    )
    if set(runtime_usd) != {"package_path", "sha256", "size_bytes"}:
        errors.append("native_deformable_return_runtime_usd_fields_invalid")
    output = Path(output_root).expanduser().absolute()
    runtime_package_path = str(runtime_usd.get("package_path") or "")
    if runtime_package_path != OUTPUT_USD_PACKAGE_PATH:
        errors.append("native_deformable_return_runtime_usd_identity_mismatch")
    expected_textures = [
        {
            "relative_path": row["relative_path"],
            "package_path": (
                PurePosixPath("prepared/textures") / PurePosixPath(row["relative_path"])
            ).as_posix(),
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in normalized["textures"]
    ]
    if artifacts.get("textures") != expected_textures:
        errors.append("native_deformable_return_texture_inventory_mismatch")
    expected_output_files = {
        "worker_return.json",
        OUTPUT_USD_PACKAGE_PATH,
        *(row["package_path"] for row in expected_textures),
    }
    try:
        output_snapshot = _snapshot_output_tree(
            output,
            retained_content_limits={"worker_return.json": _MAX_WORKER_RETURN_BYTES},
            invalid_error="native_deformable_return_output_member_set_invalid",
            resource_error="native_deformable_return_output_resource_limit_exceeded",
        )
    except NativeDeformableAssetPreparationError as exc:
        errors.extend(exc.errors)
        output_snapshot = _OutputTreeSnapshot(files={}, directories=frozenset())
    if set(
        output_snapshot.files
    ) != expected_output_files or output_snapshot.directories != _expected_output_directories(
        expected_output_files
    ):
        errors.append("native_deformable_return_output_member_set_invalid")
    try:
        _snapshot_member(
            output_snapshot,
            runtime_package_path,
            expected_digest=str(runtime_usd.get("sha256") or ""),
            expected_size=runtime_usd.get("size_bytes"),
            retain_content=False,
            error="native_deformable_return_runtime_usd_identity_mismatch",
        )
    except NativeDeformableAssetPreparationError as exc:
        errors.extend(exc.errors)
    for index, row in enumerate(expected_textures):
        try:
            _snapshot_member(
                output_snapshot,
                row["package_path"],
                expected_digest=row["sha256"],
                expected_size=row["size_bytes"],
                retain_content=False,
                error=f"native_deformable_return_texture_identity_mismatch:{index}",
            )
        except NativeDeformableAssetPreparationError as exc:
            errors.extend(exc.errors)

    try:
        persisted_snapshot = output_snapshot.files.get("worker_return.json")
        if (
            persisted_snapshot is None
            or persisted_snapshot.content is None
            or persisted_snapshot.size_bytes <= 0
            or persisted_snapshot.size_bytes > _MAX_WORKER_RETURN_BYTES
        ):
            raise NativeDeformableAssetPreparationError(
                ["native_deformable_return_persisted_payload_invalid"]
            )
        persisted_return = json.loads(persisted_snapshot.content.decode("utf-8"))
    except NativeDeformableAssetPreparationError as exc:
        errors.extend(exc.errors)
        persisted_return = None
    except (UnicodeDecodeError, json.JSONDecodeError):
        persisted_return = None
    if persisted_return != returned:
        errors.append("native_deformable_return_persisted_payload_mismatch")

    readback = _mapping(
        returned.get("readback"),
        error="native_deformable_return_readback_invalid",
        errors=errors,
    )
    if set(readback) != {
        "stage_metadata",
        "visual_mesh",
        "authoring_root_prim_path",
        "deformable_schema_prim_path",
        "body_api_schemas",
        "physics_material",
        "mass_properties",
        "physics_material_binding",
        "material_binding",
        "simulation_topology",
        "collision_topology",
        "physics_configuration",
        "texture_inventory",
        "experimental_api_schemas",
        "empty_tet_mesh_prim_paths",
        "guide_prim_paths",
        "light_prim_paths",
        "source_provider_prim_paths",
        "source_provider_attributes",
    }:
        errors.append("native_deformable_return_readback_fields_invalid")
    required = normalized["required_native_readback"]
    if readback.get("stage_metadata") != required["stage_metadata"]:
        errors.append("native_deformable_return_stage_metadata_mismatch")
    visual = _mapping(
        readback.get("visual_mesh"),
        error="native_deformable_return_visual_mesh_invalid",
        errors=errors,
    )
    required_visual = required["visual_mesh"]
    if set(visual) != {
        "prim_path",
        "point_count",
        "triangle_count",
        "source_face_topology_sha256",
        "output_face_topology_sha256",
        "dimensions_m",
        "authored_scale_xyz",
        "metric_scale_baked_into_points",
        "source_xform_flattened",
        "source_world_bounds_center_m",
        "recentered_before_scale",
        "aabb_center_m",
        "authored_pivot_m",
        "placement_origin_semantics",
        "point_positions_sha256",
        "closed_volume_m3",
    }:
        errors.append("native_deformable_return_visual_mesh_fields_invalid")
    for field in (
        "prim_path",
        "point_count",
        "triangle_count",
        "authored_scale_xyz",
    ):
        if visual.get(field) != required_visual[field]:
            errors.append(f"native_deformable_return_visual_{field}_mismatch")
    for field in (
        "metric_scale_baked_into_points",
        "source_xform_flattened",
        "recentered_before_scale",
    ):
        if visual.get(field) is not True:
            errors.append(f"native_deformable_return_visual_{field}_mismatch")
    if not _valid_digest(visual.get("point_positions_sha256")):
        errors.append("native_deformable_return_visual_point_positions_digest_invalid")
    source_face_digest = visual.get("source_face_topology_sha256")
    output_face_digest = visual.get("output_face_topology_sha256")
    if (
        not _valid_digest(source_face_digest)
        or not _valid_digest(output_face_digest)
        or source_face_digest != output_face_digest
    ):
        errors.append("native_deformable_return_visual_face_topology_mismatch")
    observed_dimensions = _positive_vector3(
        visual.get("dimensions_m"),
        error="native_deformable_return_visual_dimensions_invalid",
        errors=errors,
    )
    if observed_dimensions and any(
        abs(observed_dimensions[index] - required_visual["dimensions_m"][index])
        > required_visual["dimension_tolerance_m"]
        for index in range(3)
    ):
        errors.append("native_deformable_return_visual_dimensions_mismatch")
    for field in (
        "source_world_bounds_center_m",
        "aabb_center_m",
        "authored_pivot_m",
        "placement_origin_semantics",
    ):
        if visual.get(field) != required_visual[field]:
            errors.append(f"native_deformable_return_visual_{field}_mismatch")
    closed_volume = visual.get("closed_volume_m3")
    if (
        isinstance(closed_volume, bool)
        or not isinstance(closed_volume, (int, float))
        or not math.isfinite(float(closed_volume))
        or abs(float(closed_volume) - required_visual["closed_volume_m3"])
        > required_visual["closed_volume_tolerance_m3"]
    ):
        errors.append("native_deformable_return_visual_closed_volume_mismatch")
    if readback.get("authoring_root_prim_path") != required["authoring_root_prim_path"]:
        errors.append("native_deformable_return_authoring_root_prim_mismatch")
    if readback.get("deformable_schema_prim_path") != required["deformable_schema_prim_path"]:
        errors.append("native_deformable_return_schema_prim_mismatch")
    if readback.get("body_api_schemas") != required["body_api_schemas"]:
        errors.append("native_deformable_return_body_schemas_mismatch")
    if readback.get("physics_material") != required["physics_material"]:
        errors.append("native_deformable_return_physics_material_mismatch")
    if readback.get("physics_material_binding") != required["physics_material_binding"]:
        errors.append("native_deformable_return_physics_material_binding_mismatch")
    mass_properties = _mapping(
        readback.get("mass_properties"),
        error="native_deformable_return_mass_properties_invalid",
        errors=errors,
    )
    required_mass = required["mass_properties"]
    if set(mass_properties) != set(required_mass):
        errors.append("native_deformable_return_mass_property_fields_invalid")
    if (
        mass_properties.get("density_kg_m3") != required_mass["density_kg_m3"]
        or mass_properties.get("closed_volume_m3") != required_mass["closed_volume_m3"]
        or mass_properties.get("mass_tolerance_kg") != required_mass["mass_tolerance_kg"]
        or mass_properties.get("development_configuration_not_observed_material_truth") is not True
        or isinstance(mass_properties.get("derived_mass_kg"), bool)
        or not isinstance(mass_properties.get("derived_mass_kg"), (int, float))
        or not math.isfinite(float(mass_properties.get("derived_mass_kg") or 0.0))
        or abs(
            float(mass_properties.get("derived_mass_kg") or 0.0) - required_mass["derived_mass_kg"]
        )
        > required_mass["mass_tolerance_kg"]
    ):
        errors.append("native_deformable_return_mass_properties_mismatch")
    if readback.get("material_binding") != required["material_binding"]:
        errors.append("native_deformable_return_material_binding_mismatch")
    simulation_topology = _verify_topology_readback(
        readback.get("simulation_topology"), label="simulation", errors=errors
    )
    collision_topology = _verify_topology_readback(
        readback.get("collision_topology"), label="collision", errors=errors
    )
    if readback.get("physics_configuration") != required["physics_configuration"]:
        errors.append("native_deformable_return_physics_configuration_mismatch")
    if readback.get("texture_inventory") != required["texture_inventory"]:
        errors.append("native_deformable_return_texture_readback_mismatch")
    for field in (
        "empty_tet_mesh_prim_paths",
        "guide_prim_paths",
        "light_prim_paths",
        "source_provider_prim_paths",
        "source_provider_attributes",
    ):
        if readback.get(field) != []:
            errors.append(f"native_deformable_return_forbidden_{field}_present")
    if readback.get("experimental_api_schemas") != []:
        errors.append("native_deformable_return_experimental_schema_present")
    if errors:
        raise NativeDeformableAssetPreparationError(errors)

    verification: dict[str, Any] = {
        "schema_version": RETURN_VERIFICATION_SCHEMA_VERSION,
        "status": "worker_payload_verified_pending_trusted_execution_join",
        "preparation_id": normalized["preparation_id"],
        "plan_digest": normalized["plan_digest"],
        "worker_result_digest": returned["worker_result_digest"],
        "runtime_usd_sha256": runtime_usd["sha256"],
        "simulation_topology_sha256": simulation_topology["topology_sha256"],
        "collision_topology_sha256": collision_topology["topology_sha256"],
        "readback_contract_satisfied": True,
        "claim_boundary": {
            "worker_payload_and_returned_bytes_structurally_verified": True,
            "trusted_native_execution_join_present": False,
            "native_cook_qualified": False,
            "native_simulator_qualified": False,
            "visual_alignment_qualified": False,
            "physical_material_equivalence": False,
        },
        "verification_digest": "",
    }
    verification["verification_digest"] = canonical_digest(
        verification, digest_field="verification_digest"
    )
    return verification


__all__ = [
    "DEFORMABLE_BODY_CFG",
    "DEFORMABLE_MATERIAL_API",
    "DEFORMABLE_MATERIAL_CFG",
    "DEFORMABLE_PHYSICS_BINDING_API",
    "INSPECTION_SCHEMA_VERSION",
    "NativeDeformableAssetPreparationError",
    "NativeDeformableStageAPI",
    "PLAN_SCHEMA_VERSION",
    "PINNED_NATIVE_CALL_CONTRACT",
    "RETURN_VERIFICATION_SCHEMA_VERSION",
    "WORKER_RETURN_SCHEMA_VERSION",
    "build_native_deformable_asset_source_package",
    "execute_native_deformable_asset_preparation",
    "materialize_native_deformable_asset_preparation_plan",
    "verify_native_deformable_asset_preparation_return",
]
