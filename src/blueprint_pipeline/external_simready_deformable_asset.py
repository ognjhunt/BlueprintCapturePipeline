"""Fail-closed static intake for externally generated deformable USD assets.

This boundary inspects bytes; it never qualifies a simulator asset.  The only
inputs are paths.  Every external file is read once through ``O_NOFOLLOW``
file descriptors, the ZIP and expanded directory must contain identical bytes,
and OpenUSD parses a temporary mirror made from those exact snapshots.  A
caller cannot supply a qualification boolean or a pre-computed digest.

The receipt deliberately keeps four claims separate:

* externally generated geometry/material candidate;
* candidate with authored deformable-looking metadata;
* standard pinned-PhysX runtime asset; and
* natively qualified or physically equivalent object.

Only the first two can be described here.  The receipt's canonical digest is
an integrity checksum, not proof of origin; downstream consumers must replay
the exact paths against an independently frozen expected digest.  A clean derived runtime USD may be
authored later from a retained immutable source mesh, but it must receive its
own digest and native cooking/contact/reset/render qualification evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import posixpath
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_source_packet import (
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)


SCHEMA_VERSION = "external_simready_deformable_asset_inspection.v2"
OBSERVATION_SCHEMA_VERSION = "external_simready_deformable_observation.v2"
SOURCE_TOPOLOGY_SCHEMA_VERSION = "interiorgs_sage_collision_component_topology.v2"

PENDING_STATUS = "pending_pinned_physx_conversion_and_native_qualification"
REJECTED_STATUS = "rejected_static_deformable_asset"
CLAIM_CEILING = "static_external_deformable_candidate_only"

PINNED_DEFORMABLE_LOADER_SOURCE = (
    "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py"
)
PINNED_DEFORMABLE_LOADER_SCHEMA_LINE = 544
PINNED_DEFORMABLE_LOADER_ERROR_LINES = (547, 550)
PINNED_AUTHORING_SOURCE = "source/isaaclab/isaaclab/sim/schemas/schemas.py"
PINNED_AUTHORING_LINES = (852, 1003)
PINNED_AUTHORING_API = "isaaclab.sim.schemas.schemas:define_deformable_body_properties"
PINNED_COOKING_API = "omni.physx.scripts.deformableUtils:add_physx_deformable_body"
PINNED_RUNTIME_CLASS = "isaaclab_physx.assets.deformable_object.deformable_object:DeformableObject"

STANDARD_PHYSX_BODY_SCHEMA = "PhysxDeformableBodyAPI"
STANDARD_PHYSX_MATERIAL_SCHEMA = "PhysxDeformableBodyMaterialAPI"
PALATIAL_SHELL_SCHEMAS = frozenset({"NewtonShellAPI", "NewtonClothAPI"})

_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_EXPANDED_BYTES = 512 * 1024 * 1024
_MAX_OBSERVATION_BYTES = 128 * 1024
_MAX_SOURCE_TOPOLOGY_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_EXPANDED_FILES = 4096
_MAX_EXPANDED_ENTRIES = 8192
_MAX_PATH_COMPONENTS = 32
_MAX_PATH_CHARACTERS = 1024
_MAX_USD_LAYER_BYTES = 64 * 1024 * 1024
_MAX_USD_LAYERS = 256
_MAX_DEPENDENCIES = 4096
_MAX_STAGE_PRIMS = 100_000
_MAX_AUTHORED_PROPERTIES_PER_PRIM = 4096
_MAX_TOTAL_AUTHORED_PROPERTIES = 250_000
_MAX_MESH_POINTS = 1_000_000
_MAX_MESH_FACES = 2_000_000
_MAX_MESH_FACE_VERTEX_INDICES = 6_000_000
_MAX_TET_POINTS = 2_000_000
_MAX_TETRAHEDRA = 4_000_000
_MAX_SDF_VALUE_NODES = 500_000
_MAX_ABSOLUTE_WORLD_COORDINATE_M = 1_000_000.0
_MIN_OBSERVED_DIMENSION_M = 1.0e-6
_MAX_OBSERVED_DIMENSION_M = 1_000.0
_MIN_BAKE_SCALE = 1.0e-6
_MAX_BAKE_SCALE = 1.0e6
_READ_CHUNK_BYTES = 1024 * 1024
_DIGEST_PREFIX = "sha256:"
_DIMENSION_RELATIVE_TOLERANCE = 0.01
_USD_LAYER_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})


class ExternalSimreadyDeformableAssetError(ValueError):
    """Stable, sorted failures at the immutable static-ingest boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(left, field, None) == getattr(right, field, None) for field in fields)


def _require_nofollow() -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_nofollow_unavailable"]
        )
    return int(nofollow), int(directory)


def _absolute_parts(
    path_value: str | os.PathLike[str], *, label: str
) -> tuple[str, tuple[str, ...], str]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_path_invalid"]
        )
    display = os.path.abspath(os.path.expanduser(os.fspath(path_value)))
    # macOS exposes these stable system aliases as symlinks.  Canonicalize only
    # the two OS-owned aliases before the component-wise O_NOFOLLOW walk; never
    # resolve an arbitrary caller-controlled symlink.
    if sys.platform == "darwin":
        if display == "/var" or display.startswith("/var/"):
            display = f"/private{display}"
        elif display == "/tmp" or display.startswith("/tmp/"):
            display = f"/private{display}"
    path = Path(display)
    if not path.anchor or not path.name:
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_path_invalid"]
        )
    return path.anchor, tuple(path.parts[1:]), display


def _open_absolute_directory(
    path_value: str | os.PathLike[str], *, label: str
) -> tuple[int, list[int], str]:
    nofollow, directory = _require_nofollow()
    anchor, parts, display = _absolute_parts(path_value, label=label)
    descriptors: list[int] = []
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        current = os.open(anchor, flags)
        descriptors.append(current)
        for part in parts:
            current = os.open(part, flags, dir_fd=current)
            descriptors.append(current)
        return current, descriptors, display
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_directory_invalid"]
        ) from exc


def _read_descriptor_once(
    descriptor: int,
    *,
    label: str,
    maximum_size: int,
    allow_empty: bool,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    invalid_size = before.st_size > maximum_size or (before.st_size == 0 and not allow_empty)
    if not stat.S_ISREG(before.st_mode) or invalid_size:
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_file_invalid"]
        )
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
    if len(content) != before.st_size or not _same_identity(before, after):
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_changed_while_reading"]
        )
    return content, before


def _read_absolute_file_once(
    path_value: str | os.PathLike[str],
    *,
    label: str,
    maximum_size: int,
) -> tuple[bytes, str, os.stat_result]:
    nofollow, directory = _require_nofollow()
    anchor, parts, display = _absolute_parts(path_value, label=label)
    descriptors: list[int] = []
    directory_flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        parent = os.open(anchor, directory_flags)
        descriptors.append(parent)
        for part in parts[:-1]:
            parent = os.open(part, directory_flags, dir_fd=parent)
            descriptors.append(parent)
        descriptor = os.open(parts[-1], file_flags, dir_fd=parent)
        descriptors.append(descriptor)
        content, metadata = _read_descriptor_once(
            descriptor,
            label=label,
            maximum_size=maximum_size,
            allow_empty=False,
        )
        return content, display, metadata
    except ExternalSimreadyDeformableAssetError:
        raise
    except OSError as exc:
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_file_invalid"]
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _valid_component(name: str) -> bool:
    return bool(
        name
        and name not in {".", ".."}
        and len(name) <= 255
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
    )


def _snapshot_directory(
    root_value: str | os.PathLike[str],
) -> tuple[dict[str, bytes], str]:
    nofollow, directory = _require_nofollow()
    root_descriptor, descriptors, display = _open_absolute_directory(
        root_value, label="expanded_root"
    )
    directory_flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    files: dict[str, bytes] = {}
    total_bytes = 0
    entry_count = 0

    def visit(descriptor: int, prefix: tuple[str, ...]) -> None:
        nonlocal entry_count, total_bytes
        if len(prefix) > _MAX_PATH_COMPONENTS:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_expanded_path_depth_exceeded"]
            )
        before = os.fstat(descriptor)
        try:
            entries = []
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    entry_count += 1
                    if entry_count > _MAX_EXPANDED_ENTRIES:
                        raise ExternalSimreadyDeformableAssetError(
                            ["external_simready_deformable_expanded_entry_count_exceeded"]
                        )
                    entries.append(entry)
            entries.sort(key=lambda entry: entry.name)
        except ExternalSimreadyDeformableAssetError:
            raise
        except OSError as exc:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_expanded_root_scan_failed"]
            ) from exc
        for entry in entries:
            name = entry.name
            if not _valid_component(name):
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_member_path_invalid"]
                )
            relative_parts = (*prefix, name)
            relative = PurePosixPath(*relative_parts).as_posix()
            if len(relative_parts) > _MAX_PATH_COMPONENTS or len(relative) > _MAX_PATH_CHARACTERS:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_member_path_invalid"]
                )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_member_invalid"]
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_symlink_forbidden"]
                )
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child = os.open(name, directory_flags, dir_fd=descriptor)
                except OSError as exc:
                    raise ExternalSimreadyDeformableAssetError(
                        ["external_simready_deformable_expanded_directory_invalid"]
                    ) from exc
                try:
                    visit(child, relative_parts)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_special_file_forbidden"]
                )
            try:
                child = os.open(name, file_flags, dir_fd=descriptor)
            except OSError as exc:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_file_invalid"]
                ) from exc
            try:
                content, _ = _read_descriptor_once(
                    child,
                    label="expanded_member",
                    maximum_size=_MAX_MEMBER_BYTES,
                    allow_empty=True,
                )
            finally:
                os.close(child)
            total_bytes += len(content)
            if total_bytes > _MAX_EXPANDED_BYTES:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_size_exceeded"]
                )
            if len(files) >= _MAX_EXPANDED_FILES:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_expanded_file_count_exceeded"]
                )
            files[relative] = content
        after = os.fstat(descriptor)
        if not _same_identity(before, after):
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_expanded_directory_changed_while_reading"]
            )

    try:
        visit(root_descriptor, ())
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    if not files:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_expanded_root_empty"]
        )
    return files, display


def _safe_archive_name(name: str, *, directory: bool) -> str:
    logical = name[:-1] if directory and name.endswith("/") else name
    if not logical or "\\" in logical or "\x00" in logical:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_archive_member_path_invalid"]
        )
    path = PurePosixPath(logical)
    normalized = posixpath.normpath(logical)
    if (
        len(logical) > _MAX_PATH_CHARACTERS
        or len(path.parts) > _MAX_PATH_COMPONENTS
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or normalized != logical
        or path.as_posix() != logical
    ):
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_archive_member_path_invalid"]
        )
    return logical


def _archive_mode(member: zipfile.ZipInfo) -> int:
    return (member.external_attr >> 16) & 0xFFFF


def _snapshot_archive(content: bytes) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_archive_invalid"]
        ) from exc
    files: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    declared_total = 0
    try:
        members = archive.infolist()
        if len(members) > _MAX_ARCHIVE_MEMBERS:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_archive_member_count_exceeded"]
            )
        for member in members:
            is_directory = member.is_dir() or stat.S_ISDIR(_archive_mode(member))
            name = _safe_archive_name(member.filename, directory=is_directory)
            if name in names:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_duplicate_member"]
                )
            names.add(name)
            mode = _archive_mode(member)
            if member.flag_bits & 0x1:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_encryption_forbidden"]
                )
            if is_directory:
                rows.append({"relative_path": name, "kind": "directory", "size_bytes": 0})
                continue
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_special_member_forbidden"]
                )
            if member.file_size > _MAX_MEMBER_BYTES:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_member_size_exceeded"]
                )
            declared_total += member.file_size
            if declared_total > _MAX_EXPANDED_BYTES:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_size_exceeded"]
                )
            if member.file_size and not member.compress_size:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_ratio_invalid"]
                )
            if member.compress_size and member.file_size / member.compress_size > 1000:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_ratio_invalid"]
                )
            try:
                member_content = archive.read(member)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_member_invalid"]
                ) from exc
            if len(member_content) != member.file_size:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_archive_member_invalid"]
                )
            files[name] = member_content
            rows.append(
                {
                    "relative_path": name,
                    "kind": "file",
                    "size_bytes": len(member_content),
                    "sha256": _sha256_bytes(member_content),
                    "crc32": f"{member.CRC:08x}",
                    "unix_mode": mode,
                }
            )
    finally:
        archive.close()
    return files, sorted(rows, key=lambda row: (row["relative_path"], row["kind"]))


def _strict_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_invalid"]
        ) from exc
    if not isinstance(value, dict):
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_invalid"]
        )
    return value


def _valid_digest(value: Any) -> bool:
    text = str(value or "")
    return bool(
        text.startswith(_DIGEST_PREFIX)
        and len(text) == len(_DIGEST_PREFIX) + 64
        and all(character in "0123456789abcdef" for character in text[len(_DIGEST_PREFIX) :])
    )


def _parse_observation(content: bytes) -> dict[str, Any]:
    value = _strict_json_object(content, label="observation")
    expected = {
        "schema_version",
        "entity_id",
        "source_topology_receipt_relative_path",
        "source_topology_receipt_file_sha256",
        "source_topology_receipt_digest",
        "source_instance_id",
        "source_component_geometry_digest",
        "source_semantic_label",
        "units",
        "dimensions_m",
    }
    errors: list[str] = []
    if set(value) != expected or value.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        errors.append("external_simready_deformable_observation_schema_invalid")
    for field in ("entity_id", "source_instance_id", "source_semantic_label"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip() or len(item) > 192:
            errors.append(f"external_simready_deformable_observation_{field}_invalid")
    relative_path = value.get("source_topology_receipt_relative_path")
    if not isinstance(relative_path, str):
        errors.append("external_simready_deformable_observation_source_path_invalid")
    else:
        source_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or len(relative_path) > _MAX_PATH_CHARACTERS
            or len(source_path.parts) > _MAX_PATH_COMPONENTS
            or source_path.is_absolute()
            or "\\" in relative_path
            or "\x00" in relative_path
            or posixpath.normpath(relative_path) != relative_path
            or any(not _valid_component(part) for part in source_path.parts)
        ):
            errors.append("external_simready_deformable_observation_source_path_invalid")
    for field in (
        "source_topology_receipt_file_sha256",
        "source_topology_receipt_digest",
        "source_component_geometry_digest",
    ):
        if not _valid_digest(value.get(field)):
            errors.append(f"external_simready_deformable_observation_{field}_invalid")
    if value.get("units") != "m":
        errors.append("external_simready_deformable_observation_units_invalid")
    dimensions = value.get("dimensions_m")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not (_MIN_OBSERVED_DIMENSION_M <= float(item) <= _MAX_OBSERVED_DIMENSION_M)
            for item in dimensions
        )
    ):
        errors.append("external_simready_deformable_observation_dimensions_invalid")
    if errors:
        raise ExternalSimreadyDeformableAssetError(errors)
    return {
        **value,
        "entity_id": value["entity_id"].strip(),
        "source_instance_id": value["source_instance_id"].strip(),
        "source_semantic_label": value["source_semantic_label"].strip(),
        "dimensions_m": [float(item) for item in dimensions],
    }


def _join_source_topology_observation(
    observation: Mapping[str, Any], *, observation_display: str
) -> tuple[dict[str, Any], bytes, str]:
    relative = PurePosixPath(str(observation["source_topology_receipt_relative_path"]))
    source_path = os.path.join(os.path.dirname(observation_display), *relative.parts)
    source_content, source_display, _ = _read_absolute_file_once(
        source_path,
        label="source_topology_receipt",
        maximum_size=_MAX_SOURCE_TOPOLOGY_BYTES,
    )
    errors: list[str] = []
    observed_file_digest = _sha256_bytes(source_content)
    if observed_file_digest != observation["source_topology_receipt_file_sha256"]:
        errors.append("external_simready_deformable_source_topology_file_digest_mismatch")
    topology = _strict_json_object(source_content, label="source_topology_receipt")
    if topology.get("schema_version") != SOURCE_TOPOLOGY_SCHEMA_VERSION:
        errors.append("external_simready_deformable_source_topology_schema_invalid")
    topology_digest = topology.get("receipt_digest")
    if (
        not _valid_digest(topology_digest)
        or topology_digest != canonical_digest(topology, digest_field="receipt_digest")
        or topology_digest != observation["source_topology_receipt_digest"]
    ):
        errors.append("external_simready_deformable_source_topology_receipt_digest_mismatch")
    coordinate_frame = topology.get("coordinate_frame")
    if (
        not isinstance(coordinate_frame, Mapping)
        or coordinate_frame.get("meters_per_unit") != 1.0
        or coordinate_frame.get("up_axis") != "Z"
    ):
        errors.append("external_simready_deformable_source_topology_coordinate_frame_invalid")
    targets = topology.get("targets")
    matching_targets = (
        [
            row
            for row in targets
            if isinstance(row, Mapping)
            and str(row.get("interiorgs_instance_id") or "") == observation["source_instance_id"]
        ]
        if isinstance(targets, list)
        else []
    )
    if len(matching_targets) != 1:
        errors.append("external_simready_deformable_source_topology_target_missing_or_ambiguous")
        target: Mapping[str, Any] = {}
    else:
        target = matching_targets[0]
    best = target.get("best_component") if isinstance(target, Mapping) else None
    if not isinstance(best, Mapping):
        errors.append("external_simready_deformable_source_topology_component_missing")
        best = {}
    if (
        target.get("component_collision_identity_passed") is not True
        or best.get("collision_api_applied") is not True
    ):
        errors.append("external_simready_deformable_source_topology_collision_identity_invalid")
    if target.get("semantic_label") != observation["source_semantic_label"]:
        errors.append("external_simready_deformable_source_topology_semantic_label_mismatch")
    if best.get("geometry_digest") != observation["source_component_geometry_digest"]:
        errors.append("external_simready_deformable_source_topology_geometry_digest_mismatch")
    source_dimensions = best.get("world_aabb_size_m")
    if (
        not isinstance(source_dimensions, list)
        or len(source_dimensions) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in source_dimensions
        )
    ):
        errors.append("external_simready_deformable_source_topology_dimensions_invalid")
        normalized_source_dimensions: list[float] = []
    else:
        normalized_source_dimensions = [float(item) for item in source_dimensions]
        if any(
            abs(normalized_source_dimensions[index] - observation["dimensions_m"][index]) > 1.0e-12
            for index in range(3)
        ):
            errors.append("external_simready_deformable_source_topology_dimensions_mismatch")
    if errors:
        raise ExternalSimreadyDeformableAssetError(errors)
    joined = {
        **dict(observation),
        "source_topology_receipt_size_bytes": len(source_content),
        "source_topology_receipt_observed_file_sha256": observed_file_digest,
        "source_target": {
            "interiorgs_instance_id": str(target["interiorgs_instance_id"]),
            "semantic_label": str(target["semantic_label"]),
            "component_collision_identity_passed": True,
            "best_component_prim_path": str(best.get("prim_path") or ""),
            "best_component_geometry_digest": str(best["geometry_digest"]),
            "best_component_world_aabb_size_m": normalized_source_dimensions,
        },
    }
    return joined, source_content, source_display


def _relative_path_within(root: str, path_value: str | os.PathLike[str], *, label: str) -> str:
    if not isinstance(path_value, (str, os.PathLike)):
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_path_invalid"]
        )
    _, _, candidate = _absolute_parts(path_value, label=label)
    try:
        if os.path.commonpath((root, candidate)) != root:
            raise ValueError("outside_root")
    except ValueError as exc:
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_outside_root"]
        ) from exc
    relative = os.path.relpath(candidate, root)
    parts = Path(relative).parts
    if not parts or parts == (".",) or any(not _valid_component(part) for part in parts):
        raise ExternalSimreadyDeformableAssetError(
            [f"external_simready_deformable_{label}_path_invalid"]
        )
    return PurePosixPath(*parts).as_posix()


def _listop_items(value: Any) -> list[str]:
    return [str(item) for item in _listop_values(value) if str(item)]


def _listop_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if bool(getattr(value, "isExplicit", False)):
        source = getattr(value, "explicitItems", ())
    else:
        source = (
            *getattr(value, "prependedItems", ()),
            *getattr(value, "addedItems", ()),
            *getattr(value, "appendedItems", ()),
        )
    result: list[Any] = []
    observed: set[str] = set()
    for item in source:
        key = str(item)
        if key and key not in observed:
            observed.add(key)
            result.append(item)
    return result


def _resolve_preflight_asset_path(
    authored_path: str,
    *,
    authoring_layer_relative_path: str,
    snapshots: Mapping[str, bytes],
    composition_asset: bool,
) -> str:
    authored = str(authored_path or "")
    if not authored:
        return ""
    if (
        len(authored) > _MAX_PATH_CHARACTERS
        or "\\" in authored
        or "\x00" in authored
        or ":" in authored
        or "[" in authored
        or "]" in authored
    ):
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_path_invalid"]
        )
    authored_posix = PurePosixPath(authored)
    if authored_posix.is_absolute():
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_outside_root"]
        )
    candidate = PurePosixPath(authoring_layer_relative_path).parent / authored_posix
    normalized = posixpath.normpath(candidate.as_posix())
    if normalized in {"", ".", ".."} or normalized.startswith("../") or normalized not in snapshots:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_missing_or_outside_root"]
        )
    if composition_asset and PurePosixPath(normalized).suffix.lower() not in _USD_LAYER_SUFFIXES:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_composition_asset_type_forbidden"]
        )
    if not snapshots[normalized]:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_empty"]
        )
    return normalized


def _preflight_snapshot_layers(
    snapshots: Mapping[str, bytes],
    *,
    temporary_root: Path,
    root_layer_relative_path: str,
    sdf: Any,
) -> dict[str, Any]:
    """Inspect every local USD layer without composing any external arc."""

    layer_paths = sorted(
        relative
        for relative in snapshots
        if PurePosixPath(relative).suffix.lower() in _USD_LAYER_SUFFIXES
    )
    if root_layer_relative_path not in layer_paths:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_root_layer_missing"]
        )
    if len(layer_paths) > _MAX_USD_LAYERS:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_layer_count_exceeded"]
        )
    dependency_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    value_nodes = 0

    def add_dependency(
        authored_path: str,
        *,
        layer_relative: str,
        spec_path: str,
        field_name: str,
        composition_asset: bool,
    ) -> None:
        if not authored_path:
            return
        resolved = _resolve_preflight_asset_path(
            authored_path,
            authoring_layer_relative_path=layer_relative,
            snapshots=snapshots,
            composition_asset=composition_asset,
        )
        dependency_rows.append(
            {
                "authoring_layer_relative_path": layer_relative,
                "spec_path": spec_path,
                "field_name": field_name,
                "authored_path": authored_path,
                "resolved_relative_path": resolved,
                "composition_asset": composition_asset,
                "sha256": _sha256_bytes(snapshots[resolved]),
                "size_bytes": len(snapshots[resolved]),
            }
        )
        if len(dependency_rows) > _MAX_DEPENDENCIES:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_dependency_count_exceeded"]
            )

    def visit_value(
        value: Any,
        *,
        layer_relative: str,
        spec_path: str,
        field_name: str,
        composition_context: bool,
        depth: int = 0,
    ) -> None:
        nonlocal value_nodes
        value_nodes += 1
        if value_nodes > _MAX_SDF_VALUE_NODES or depth > 64:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_metadata_resource_limit_exceeded"]
            )
        if isinstance(value, sdf.AssetPath):
            add_dependency(
                str(value.path),
                layer_relative=layer_relative,
                spec_path=spec_path,
                field_name=field_name,
                composition_asset=composition_context,
            )
            return
        if isinstance(value, (sdf.Reference, sdf.Payload)):
            add_dependency(
                str(value.assetPath or ""),
                layer_relative=layer_relative,
                spec_path=spec_path,
                field_name=field_name,
                composition_asset=True,
            )
            return
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                visit_value(
                    nested,
                    layer_relative=layer_relative,
                    spec_path=spec_path,
                    field_name=f"{field_name}.{key_text}",
                    composition_context=composition_context or "clip" in key_text.lower(),
                    depth=depth + 1,
                )
            return
        listop = _listop_values(value)
        if listop:
            for nested in listop:
                visit_value(
                    nested,
                    layer_relative=layer_relative,
                    spec_path=spec_path,
                    field_name=field_name,
                    composition_context=composition_context,
                    depth=depth + 1,
                )
            return
        if isinstance(value, Sequence):
            for nested in value:
                visit_value(
                    nested,
                    layer_relative=layer_relative,
                    spec_path=spec_path,
                    field_name=field_name,
                    composition_context=composition_context,
                    depth=depth + 1,
                )

    for layer_relative in layer_paths:
        layer_content = snapshots[layer_relative]
        if len(layer_content) > _MAX_USD_LAYER_BYTES:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_layer_size_exceeded"]
            )
        layer_path = temporary_root.joinpath(*PurePosixPath(layer_relative).parts)
        try:
            layer = sdf.Layer.FindOrOpen(str(layer_path))
        except Exception as exc:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_layer_parse_failed"]
            ) from exc
        if layer is None:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_layer_parse_failed"]
            )
        for sublayer in layer.subLayerPaths:
            add_dependency(
                str(sublayer),
                layer_relative=layer_relative,
                spec_path="/",
                field_name="subLayers",
                composition_asset=True,
            )
        spec_count = 0

        def inspect_spec(path: Any) -> None:
            nonlocal spec_count
            spec_count += 1
            if spec_count > _MAX_STAGE_PRIMS + _MAX_TOTAL_AUTHORED_PROPERTIES:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_spec_count_exceeded"]
                )
            spec = layer.GetObjectAtPath(path)
            if spec is None:
                return
            for key in spec.ListInfoKeys():
                value = spec.GetInfo(key)
                key_text = str(key)
                visit_value(
                    value,
                    layer_relative=layer_relative,
                    spec_path=str(path),
                    field_name=key_text,
                    composition_context=key_text in {"references", "payload"}
                    or "clip" in key_text.lower(),
                )

        try:
            layer.Traverse("/", inspect_spec)
        except ExternalSimreadyDeformableAssetError:
            raise
        except Exception as exc:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_layer_traversal_failed"]
            ) from exc
        layer_rows.append(
            {
                "relative_path": layer_relative,
                "sha256": _sha256_bytes(layer_content),
                "size_bytes": len(layer_content),
                "spec_count": spec_count,
            }
        )
    unique_dependencies = {
        (
            row["authoring_layer_relative_path"],
            row["spec_path"],
            row["field_name"],
            row["authored_path"],
            row["resolved_relative_path"],
        ): row
        for row in dependency_rows
    }
    return {
        "layers": layer_rows,
        "dependencies": sorted(
            unique_dependencies.values(),
            key=lambda row: (
                row["authoring_layer_relative_path"],
                row["spec_path"],
                row["field_name"],
                row["authored_path"],
            ),
        ),
        "composition_confined_before_stage_open": True,
    }


def _open_composed_stage(usd: Any, path: Path) -> Any:
    return usd.Stage.Open(str(path), load=usd.Stage.LoadAll)


def _bounded_stage_prims(stage: Any) -> list[Any]:
    result: list[Any] = []
    for prim in stage.TraverseAll():
        if prim.IsPseudoRoot():
            continue
        result.append(prim)
        if len(result) > _MAX_STAGE_PRIMS:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_prim_count_exceeded"]
            )
    return result


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "authoredPath") and hasattr(value, "resolvedPath"):
        return {
            "authored_path": str(value.authoredPath),
            "resolved_path_recorded": False,
        }
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return [float(value.GetReal()), *_json_value(value.GetImaginary())]
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted = [_json_value(item) for item in value]
        if len(converted) <= 64:
            return converted
        encoded = json.dumps(
            converted,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "element_count": len(converted),
            "sha256": _sha256_bytes(encoded),
        }
    try:
        converted = list(value)
    except (TypeError, ValueError):
        return str(value)
    return _json_value(converted)


def _bounds(
    points: Sequence[tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float]]:
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return minimum, maximum, [maximum[axis] - minimum[axis] for axis in range(3)]


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(value, value))


def _inspect_surface_mesh(prim: Any, *, usd: Any, usd_geom: Any, gf: Any) -> dict[str, Any]:
    mesh = usd_geom.Mesh(prim)
    raw_points = mesh.GetPointsAttr().Get()
    raw_counts = mesh.GetFaceVertexCountsAttr().Get()
    raw_indices = mesh.GetFaceVertexIndicesAttr().Get()
    errors: list[str] = []
    if raw_points is None or raw_counts is None or raw_indices is None:
        errors.append("surface_topology_missing")
        raw_points = raw_points or []
        raw_counts = raw_counts or []
        raw_indices = raw_indices or []
    if len(raw_points) > _MAX_MESH_POINTS:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_surface_point_count_exceeded"]
        )
    if len(raw_counts) > _MAX_MESH_FACES:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_surface_face_count_exceeded"]
        )
    if len(raw_indices) > _MAX_MESH_FACE_VERTEX_INDICES:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_surface_index_count_exceeded"]
        )
    points = [tuple(float(component) for component in point) for point in raw_points]
    counts = [int(item) for item in raw_counts]
    indices = [int(item) for item in raw_indices]
    if not points or not counts or sum(counts) != len(indices):
        errors.append("surface_topology_invalid")
    if any(count < 3 for count in counts):
        errors.append("surface_face_size_invalid")
    if any(index < 0 or index >= len(points) for index in indices):
        errors.append("surface_index_out_of_range")
    points_are_finite = all(math.isfinite(component) for point in points for component in point)
    if not points_are_finite:
        errors.append("surface_point_non_finite")
    if mesh.GetPointsAttr().ValueMightBeTimeVarying() or any(
        operation.GetAttr().ValueMightBeTimeVarying()
        for operation in usd_geom.Xformable(prim).GetOrderedXformOps()
    ):
        errors.append("surface_rest_state_time_varying")

    faces: list[tuple[int, ...]] = []
    cursor = 0
    if not errors:
        for count in counts:
            face = tuple(indices[cursor : cursor + count])
            cursor += count
            if len(set(face)) != len(face):
                errors.append("surface_face_repeated_vertex")
            faces.append(face)

    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_counts: Counter[tuple[int, int]] = Counter()
    adjacency: list[set[int]] = [set() for _ in points]
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            edge_counts[tuple(sorted((first, second)))] += 1
            directed_counts[(first, second)] += 1
            adjacency[first].add(second)
            adjacency[second].add(first)
        triangles.extend(
            (face[0], face[index], face[index + 1]) for index in range(1, len(face) - 1)
        )

    world_points: list[tuple[float, float, float]] = []
    if points_are_finite:
        xform = usd_geom.Xformable(prim).ComputeLocalToWorldTransform(usd.TimeCode.Default())
        meters_per_unit = float(usd_geom.GetStageMetersPerUnit(prim.GetStage()))
        if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
            errors.append("surface_stage_units_invalid")
            meters_per_unit = 1.0
        world_points = [
            tuple(
                float(component) * meters_per_unit
                for component in xform.Transform(gf.Vec3d(*point))
            )
            for point in points
        ]
        if any(not math.isfinite(component) for point in world_points for component in point):
            errors.append("surface_world_point_non_finite")
            world_points = []
        elif any(
            abs(component) > _MAX_ABSOLUTE_WORLD_COORDINATE_M
            for point in world_points
            for component in point
        ):
            errors.append("surface_world_coordinate_magnitude_exceeded")
            world_points = []
    total_area = 0.0
    signed_volume = 0.0
    degenerate_triangles = 0
    for first, second, third in triangles if world_points else ():
        a, b, c = world_points[first], world_points[second], world_points[third]
        cross = _cross(_subtract(b, a), _subtract(c, a))
        area = 0.5 * _norm(cross)
        total_area += area
        if area <= 1e-12:
            degenerate_triangles += 1
        signed_volume += _dot(a, _cross(b, c)) / 6.0
    if not math.isfinite(total_area):
        errors.append("surface_area_non_finite")
    if not math.isfinite(signed_volume):
        errors.append("surface_volume_non_finite")

    used = {index for face in faces for index in face}
    component_sizes: list[int] = []
    remaining = set(used)
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = adjacency[current] & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        component_sizes.append(size)

    boundary_edges = sum(count == 1 for count in edge_counts.values())
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    orientation_mismatches = sum(
        directed_counts[(first, second)] != 1 or directed_counts[(second, first)] != 1
        for first, second in edge_counts
        if edge_counts[(first, second)] == 2
    )
    duplicate_faces = len(faces) - len({tuple(sorted(face)) for face in faces})
    isolated_vertices = sum(not neighbors for neighbors in adjacency)
    if boundary_edges:
        errors.append("surface_boundary_edges_present")
    if nonmanifold_edges:
        errors.append("surface_nonmanifold_edges_present")
    if orientation_mismatches:
        errors.append("surface_orientation_inconsistent")
    if len(component_sizes) != 1:
        errors.append("surface_component_count_invalid")
    if isolated_vertices:
        errors.append("surface_isolated_vertices_present")
    if duplicate_faces:
        errors.append("surface_duplicate_faces_present")
    if degenerate_triangles:
        errors.append("surface_degenerate_triangles_present")
    if math.isfinite(signed_volume) and abs(signed_volume) <= 1e-12:
        errors.append("surface_volume_zero")

    local_minimum: list[float] = []
    local_maximum: list[float] = []
    local_dimensions: list[float] = []
    world_minimum: list[float] = []
    world_maximum: list[float] = []
    world_dimensions: list[float] = []
    if points and points_are_finite:
        local_minimum, local_maximum, local_dimensions = _bounds(points)
    if world_points:
        world_minimum, world_maximum, world_dimensions = _bounds(world_points)
        if any(dimension <= 0 for dimension in world_dimensions):
            errors.append("surface_dimensions_invalid")

    return {
        "prim_path": str(prim.GetPath()),
        "vertex_count": len(points),
        "face_count": len(faces),
        "triangulated_face_count": len(triangles),
        "edge_count": len(edge_counts),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "orientation_mismatch_edge_count": orientation_mismatches,
        "connected_component_count": len(component_sizes),
        "connected_component_vertex_counts": sorted(component_sizes, reverse=True),
        "isolated_vertex_count": isolated_vertices,
        "duplicate_face_count": duplicate_faces,
        "degenerate_triangle_count": degenerate_triangles,
        "local_bounds": {
            "minimum": local_minimum,
            "maximum": local_maximum,
            "dimensions": local_dimensions,
        },
        "world_bounds": {
            "minimum": world_minimum,
            "maximum": world_maximum,
            "dimensions": world_dimensions,
        },
        "surface_area_m2": total_area,
        "signed_volume_m3": signed_volume,
        "absolute_volume_m3": abs(signed_volume),
        "closed_oriented_manifold": not errors,
        "topology_errors": sorted(set(errors)),
    }


def _resolve_snapshot_dependency(
    authored: str,
    resolved: str,
    *,
    temporary_root: Path,
    snapshots: Mapping[str, bytes],
    preflight_dependencies: Sequence[Mapping[str, Any]],
) -> str:
    if not authored or "\\" in authored or "\x00" in authored:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_path_invalid"]
        )
    preflight_matches = {
        str(row["resolved_relative_path"])
        for row in preflight_dependencies
        if row.get("authored_path") == authored
    }
    if resolved:
        resolved_path = Path(resolved)
        try:
            resolved_relative = resolved_path.relative_to(temporary_root).as_posix()
        except ValueError as exc:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_dependency_outside_root"]
            ) from exc
        if resolved_relative not in preflight_matches:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_dependency_resolution_mismatch"]
            )
        normalized = resolved_relative
    elif len(preflight_matches) == 1:
        normalized = next(iter(preflight_matches))
    else:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_resolution_ambiguous"]
        )
    if normalized not in snapshots or not snapshots[normalized]:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_dependency_empty"]
        )
    return normalized


def _inspect_stage_snapshot(
    snapshots: Mapping[str, bytes],
    *,
    usd_relative_path: str,
    observed_dimensions_m: Sequence[float],
) -> dict[str, Any]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils
    except ImportError as exc:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_openusd_unavailable"]
        ) from exc

    with tempfile.TemporaryDirectory(prefix="external-simready-snapshot-") as directory:
        temporary_root = Path(directory)
        for relative, content in snapshots.items():
            target = temporary_root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        usd_path = temporary_root.joinpath(*PurePosixPath(usd_relative_path).parts)
        preflight = _preflight_snapshot_layers(
            snapshots,
            temporary_root=temporary_root,
            root_layer_relative_path=usd_relative_path,
            sdf=Sdf,
        )
        try:
            stage = _open_composed_stage(Usd, usd_path)
        except Exception as exc:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_open_failed"]
            ) from exc
        if stage is None:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_open_failed"]
            )

        stage_prims = _bounded_stage_prims(stage)
        raw_schema_rows: list[dict[str, Any]] = []
        physics_attributes: list[dict[str, Any]] = []
        physics_relationships: list[dict[str, Any]] = []
        material_prims: list[dict[str, Any]] = []
        shader_prims: list[dict[str, Any]] = []
        dependencies: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        composition_arcs: list[dict[str, Any]] = []
        schemas_by_prim: dict[str, set[str]] = {}
        dome_lights: list[dict[str, Any]] = []
        mesh_prims: list[Any] = []
        cooking_bindings: list[dict[str, Any]] = []
        static_rigid_contract_paths: set[str] = set()
        palatial_shell_intent_paths: set[str] = set()

        total_authored_properties = 0
        for prim in stage_prims:
            prim_path = str(prim.GetPath())
            schemas = _listop_items(prim.GetMetadata("apiSchemas"))
            schemas_by_prim[prim_path] = set(schemas)
            if schemas:
                raw_schema_rows.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "raw_authored_schemas": schemas,
                        "registered_composed_schemas": list(prim.GetAppliedSchemas()),
                    }
                )
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)
            if prim.GetTypeName() in {"Material", "Shader"}:
                authored_properties: list[dict[str, Any]] = []
                authored_properties_for_prim = prim.GetAuthoredProperties()
                if len(authored_properties_for_prim) > _MAX_AUTHORED_PROPERTIES_PER_PRIM:
                    raise ExternalSimreadyDeformableAssetError(
                        ["external_simready_deformable_usd_property_count_exceeded"]
                    )
                for authored_property in authored_properties_for_prim:
                    if isinstance(authored_property, Usd.Attribute):
                        authored_properties.append(
                            {
                                "kind": "attribute",
                                "name": authored_property.GetName(),
                                "type": str(authored_property.GetTypeName()),
                                "custom": authored_property.IsCustom(),
                                "value": _json_value(authored_property.Get()),
                                "connections": [
                                    str(path) for path in authored_property.GetConnections()
                                ],
                            }
                        )
                    elif isinstance(authored_property, Usd.Relationship):
                        authored_properties.append(
                            {
                                "kind": "relationship",
                                "name": authored_property.GetName(),
                                "custom": authored_property.IsCustom(),
                                "targets": [str(path) for path in authored_property.GetTargets()],
                            }
                        )
                material_or_shader = {
                    "prim_path": str(prim.GetPath()),
                    "type_name": prim.GetTypeName(),
                    "raw_authored_schemas": schemas,
                    "authored_properties": sorted(
                        authored_properties,
                        key=lambda row: (row["kind"], row["name"]),
                    ),
                }
                if prim.GetTypeName() == "Material":
                    material_prims.append(material_or_shader)
                else:
                    shader_prims.append(material_or_shader)
            if prim.GetTypeName() == "DomeLight":
                texture = prim.GetAttribute("inputs:texture:file").Get()
                dome_lights.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "intensity": _json_value(prim.GetAttribute("inputs:intensity").Get()),
                        "texture": _json_value(texture),
                    }
                )
            intent = prim.GetAttribute("newton:deformable:simulationIntent").Get()
            body_type = prim.GetAttribute("newton:bodyType").Get()
            if schemas and PALATIAL_SHELL_SCHEMAS.intersection(schemas):
                palatial_shell_intent_paths.add(prim_path)
            if str(intent or "") in {"cloth", "shell"} or body_type == "cloth":
                palatial_shell_intent_paths.add(prim_path)

            authored_properties_for_prim = prim.GetAuthoredProperties()
            if len(authored_properties_for_prim) > _MAX_AUTHORED_PROPERTIES_PER_PRIM:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_property_count_exceeded"]
                )
            total_authored_properties += len(authored_properties_for_prim)
            if total_authored_properties > _MAX_TOTAL_AUTHORED_PROPERTIES:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_property_count_exceeded"]
                )
            for prop in authored_properties_for_prim:
                if isinstance(prop, Usd.Attribute):
                    name = prop.GetName()
                    value = prop.Get()
                    if name.startswith(("physics:", "physx", "omniphysics:", "lightwheelusd:")):
                        physics_attributes.append(
                            {
                                "prim_path": str(prim.GetPath()),
                                "name": name,
                                "type": str(prop.GetTypeName()),
                                "custom": prop.IsCustom(),
                                "value": _json_value(value),
                            }
                        )
                        if name == "lightwheelusd:assetFormat" and value == "static_rigid_usd":
                            static_rigid_contract_paths.add(prim_path)
                    if str(prop.GetTypeName()) in {"asset", "asset[]"} and value is not None:
                        asset_values = (
                            [value] if str(prop.GetTypeName()) == "asset" else list(value)
                        )
                        for asset_value in asset_values:
                            authored = str(asset_value.authoredPath)
                            relative = _resolve_snapshot_dependency(
                                authored,
                                str(asset_value.resolvedPath),
                                temporary_root=temporary_root,
                                snapshots=snapshots,
                                preflight_dependencies=preflight["dependencies"],
                            )
                            dependency_key = (relative, prim_path, name, authored)
                            dependencies[dependency_key] = {
                                "relative_path": relative,
                                "sha256": _sha256_bytes(snapshots[relative]),
                                "size_bytes": len(snapshots[relative]),
                                "consumer_prim_path": str(prim.GetPath()),
                                "consumer_attribute": name,
                                "authored_path": authored,
                            }
                elif isinstance(prop, Usd.Relationship):
                    name = prop.GetName()
                    targets = [str(target) for target in prop.GetTargets()]
                    if name.startswith(("physics:", "physx", "omniphysics:", "material:")):
                        physics_relationships.append(
                            {
                                "prim_path": str(prim.GetPath()),
                                "name": name,
                                "custom": prop.IsCustom(),
                                "targets": targets,
                            }
                        )
                    if name == "physxDeformableBody:cookingSourceMesh":
                        cooking_bindings.append({"body_prim_path": prim_path, "targets": targets})

            for arc_type in ("references", "payload"):
                for arc in _listop_values(prim.GetMetadata(arc_type)):
                    asset_path = str(getattr(arc, "assetPath", "") or "")
                    row = {
                        "prim_path": str(prim.GetPath()),
                        "arc_type": arc_type,
                        "asset_path": asset_path,
                        "target_prim_path": str(getattr(arc, "primPath", "") or ""),
                    }
                    if asset_path:
                        relative = _resolve_snapshot_dependency(
                            asset_path,
                            "",
                            temporary_root=temporary_root,
                            snapshots=snapshots,
                            preflight_dependencies=preflight["dependencies"],
                        )
                        row["resolved_relative_path"] = relative
                    composition_arcs.append(row)

        try:
            dependency_layers, dependency_assets, unresolved_dependencies = (
                UsdUtils.ComputeAllDependencies(str(usd_path))
            )
        except Exception as exc:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_dependency_scan_failed"]
            ) from exc
        if unresolved_dependencies:
            raise ExternalSimreadyDeformableAssetError(
                ["external_simready_deformable_usd_dependency_unresolved"]
            )
        computed_dependency_layers: list[str] = []
        for layer in dependency_layers:
            resolved = Path(str(layer.resolvedPath))
            try:
                relative = resolved.relative_to(temporary_root).as_posix()
            except ValueError as exc:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_layer_outside_root"]
                ) from exc
            if relative not in snapshots:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_layer_missing"]
                )
            computed_dependency_layers.append(relative)
        computed_dependency_assets: list[str] = []
        for asset in dependency_assets:
            resolved = Path(str(asset))
            try:
                relative = resolved.relative_to(temporary_root).as_posix()
            except ValueError as exc:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_dependency_outside_root"]
                ) from exc
            if relative not in snapshots:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_dependency_missing_or_outside_root"]
                )
            computed_dependency_assets.append(relative)

        used_layers: list[dict[str, Any]] = []
        for layer in stage.GetUsedLayers(includeClipLayers=True):
            if not layer.resolvedPath:
                continue
            layer_path = Path(str(layer.resolvedPath))
            try:
                relative = layer_path.relative_to(temporary_root).as_posix()
            except ValueError as exc:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_layer_outside_root"]
                ) from exc
            if relative not in snapshots:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_usd_layer_missing"]
                )
            used_layers.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256_bytes(snapshots[relative]),
                    "size_bytes": len(snapshots[relative]),
                }
            )

        selected_mesh = None
        selected_body = None
        selection_method = ""
        cooking_relationship_join_valid = False
        if len(cooking_bindings) == 1 and len(cooking_bindings[0]["targets"]) == 1:
            candidate_body = stage.GetPrimAtPath(cooking_bindings[0]["body_prim_path"])
            candidate_mesh = stage.GetPrimAtPath(cooking_bindings[0]["targets"][0])
            if (
                candidate_body
                and candidate_mesh
                and candidate_mesh.GetTypeName() == "Mesh"
                and (
                    str(candidate_mesh.GetPath()) == str(candidate_body.GetPath())
                    or str(candidate_mesh.GetPath()).startswith(
                        f"{str(candidate_body.GetPath()).rstrip('/')}/"
                    )
                )
            ):
                selected_body = candidate_body
                selected_mesh = candidate_mesh
                selection_method = "authored_body_scoped_cooking_source_relationship"
                cooking_relationship_join_valid = True
        elif not cooking_bindings and len(mesh_prims) == 1:
            selected_mesh = mesh_prims[0]
            selection_method = "only_mesh_prim_without_body_binding"

        if selected_mesh is None:
            surface = {
                "prim_path": None,
                "closed_oriented_manifold": False,
                "topology_errors": ["surface_mesh_selection_ambiguous_or_missing"],
                "world_bounds": {"dimensions": []},
            }
        else:
            surface = _inspect_surface_mesh(
                selected_mesh,
                usd=Usd,
                usd_geom=UsdGeom,
                gf=Gf,
            )
        surface["selection_method"] = selection_method or None

        tetmeshes: list[dict[str, Any]] = []
        for prim in stage_prims:
            if prim.GetTypeName() != "TetMesh":
                continue
            points = prim.GetAttribute("points").Get()
            tetrahedra = prim.GetAttribute("tetVertexIndices").Get()
            point_count = len(points) if points is not None else 0
            tetrahedron_count = len(tetrahedra) if tetrahedra is not None else 0
            if point_count > _MAX_TET_POINTS:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_tet_point_count_exceeded"]
                )
            if tetrahedron_count > _MAX_TETRAHEDRA:
                raise ExternalSimreadyDeformableAssetError(
                    ["external_simready_deformable_tetrahedron_count_exceeded"]
                )
            topology_valid = bool(point_count and tetrahedron_count)
            if topology_valid:
                topology_valid = all(
                    len(tuple(tet)) == 4
                    and all(0 <= int(index) < point_count for index in tuple(tet))
                    for tet in tetrahedra
                ) and all(
                    all(math.isfinite(float(component)) for component in point) for point in points
                )
            tetmeshes.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "point_count": point_count,
                    "tetrahedron_count": tetrahedron_count,
                    "topology_valid": topology_valid,
                    "raw_authored_schemas": _listop_items(prim.GetMetadata("apiSchemas")),
                }
            )

        selected_body_path = str(selected_body.GetPath()) if selected_body else None
        selected_mesh_path = str(selected_mesh.GetPath()) if selected_mesh else None

        def nearest_material_binding(
            start_path: str | None, *, relationship_name: str
        ) -> dict[str, Any] | None:
            if not start_path:
                return None
            current = Sdf.Path(start_path)
            while current != Sdf.Path.absoluteRootPath:
                matches = [
                    row
                    for row in physics_relationships
                    if row["prim_path"] == str(current) and row["name"] == relationship_name
                ]
                if matches:
                    if len(matches) != 1 or len(matches[0]["targets"]) != 1:
                        return None
                    target_path = matches[0]["targets"][0]
                    target_prim = stage.GetPrimAtPath(target_path)
                    if not target_prim or target_prim.GetTypeName() != "Material":
                        return None
                    return {
                        "relationship_owner_prim_path": str(current),
                        "relationship_name": relationship_name,
                        "material_prim_path": target_path,
                    }
                current = current.GetParentPath()
            return None

        visual_material_binding = nearest_material_binding(
            selected_mesh_path, relationship_name="material:binding"
        )
        physics_material_binding = nearest_material_binding(
            selected_body_path, relationship_name="material:binding:physics"
        )
        body_tetmeshes = [
            row
            for row in tetmeshes
            if selected_body_path
            and row["prim_path"].startswith(f"{selected_body_path.rstrip('/')}/")
        ]

        dimensions = list(surface.get("world_bounds", {}).get("dimensions", []))
        required_bake_scale: list[float] = []
        relative_errors: list[float] = []
        dimensions_match = False
        if len(dimensions) == 3 and all(
            math.isfinite(float(value)) and float(value) > 0 for value in dimensions
        ):
            required_bake_scale = [
                float(observed_dimensions_m[index]) / float(dimensions[index]) for index in range(3)
            ]
            relative_errors = [
                abs(float(dimensions[index]) - float(observed_dimensions_m[index]))
                / float(observed_dimensions_m[index])
                for index in range(3)
            ]
            if all(
                math.isfinite(value) and _MIN_BAKE_SCALE <= value <= _MAX_BAKE_SCALE
                for value in required_bake_scale
            ):
                dimensions_match = max(relative_errors) <= _DIMENSION_RELATIVE_TOLERANCE
            else:
                required_bake_scale = []
                relative_errors = []

        default_prim = stage.GetDefaultPrim()
        default_path = str(default_prim.GetPath()) if default_prim else None
        dome_lights_in_default = [
            row
            for row in dome_lights
            if default_path
            and (
                row["prim_path"] == default_path or row["prim_path"].startswith(f"{default_path}/")
            )
        ]
        static_rigid_contract = bool(
            default_path
            and any(
                path == default_path or path.startswith(f"{default_path}/")
                for path in static_rigid_contract_paths
            )
        )
        palatial_shell_intent = bool(
            selected_body_path and selected_body_path in palatial_shell_intent_paths
        )
        has_standard_body_schema = bool(
            selected_body_path
            and STANDARD_PHYSX_BODY_SCHEMA in schemas_by_prim.get(selected_body_path, set())
        )
        bound_physics_material_path = (
            physics_material_binding["material_prim_path"] if physics_material_binding else None
        )
        has_standard_material_schema = bool(
            bound_physics_material_path
            and STANDARD_PHYSX_MATERIAL_SCHEMA
            in schemas_by_prim.get(bound_physics_material_path, set())
        )
        has_nonempty_tetmesh = bool(body_tetmeshes) and all(
            row["topology_valid"] for row in body_tetmeshes
        )

        blockers = ["external_simready_deformable_native_qualification_missing"]
        hazards: list[str] = []
        if not default_path:
            blockers.append("external_simready_deformable_default_prim_missing")
        if not surface.get("closed_oriented_manifold"):
            blockers.append("external_simready_deformable_closed_surface_topology_invalid")
        if not cooking_relationship_join_valid:
            blockers.append(
                "external_simready_deformable_cooking_source_binding_missing_or_ambiguous"
            )
        if visual_material_binding is None:
            blockers.append(
                "external_simready_deformable_visual_material_binding_missing_or_ambiguous"
            )
        if physics_material_binding is None:
            blockers.append(
                "external_simready_deformable_physics_material_binding_missing_or_ambiguous"
            )
        if not has_standard_body_schema:
            blockers.append("external_simready_deformable_pinned_physx_body_schema_missing")
        if not has_standard_material_schema:
            blockers.append("external_simready_deformable_pinned_physx_material_schema_missing")
        if not has_nonempty_tetmesh:
            blockers.append("external_simready_deformable_cooked_tetmesh_topology_missing")
        if dome_lights_in_default:
            blockers.append(
                "external_simready_deformable_default_prim_dome_light_requires_exclusion"
            )
            hazards.append("embedded_lighting_would_change_frozen_evaluation_illumination")
        if static_rigid_contract:
            blockers.append("external_simready_deformable_static_rigid_contract_conflict")
            hazards.append("provider_root_declares_static_rigid_asset_format")
        if not dimensions_match:
            blockers.append("external_simready_deformable_frozen_dimensions_require_baked_scale")
        if not palatial_shell_intent:
            hazards.append("palatial_shell_loader_will_not_detect_cloth")

        stage_up_axis = str(UsdGeom.GetStageUpAxis(stage))
        stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        stage_time_values = {
            "time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
            "frames_per_second": float(stage.GetFramesPerSecond()),
            "start_time_code": float(stage.GetStartTimeCode()),
            "end_time_code": float(stage.GetEndTimeCode()),
        }
        if (
            stage_up_axis not in {"X", "Y", "Z"}
            or not math.isfinite(stage_meters_per_unit)
            or stage_meters_per_unit <= 0.0
            or any(not math.isfinite(value) for value in stage_time_values.values())
        ):
            blockers.append("external_simready_deformable_stage_metadata_invalid")
        dependency_values = sorted(
            dependencies.values(),
            key=lambda row: (
                row["relative_path"],
                row["consumer_prim_path"],
                row["consumer_attribute"],
                row["authored_path"],
            ),
        )
        bound_visual_material_path = (
            visual_material_binding["material_prim_path"] if visual_material_binding else None
        )
        selected_visual_material_dependencies = [
            row
            for row in dependency_values
            if bound_visual_material_path
            and (
                row["consumer_prim_path"] == bound_visual_material_path
                or row["consumer_prim_path"].startswith(
                    f"{bound_visual_material_path.rstrip('/')}/"
                )
            )
        ]
        selected_entity_paths = {
            path for path in (selected_body_path, selected_mesh_path) if path is not None
        }
        authored_deformable_metadata_on_selected_entity = any(
            "Deformable" in schema
            for path in selected_entity_paths
            for schema in schemas_by_prim.get(path, set())
        ) or any(
            row["prim_path"] in selected_entity_paths
            and row["name"].startswith(
                ("physxDeformableBody:", "omniphysics:deformable", "newton:deformable:")
            )
            for row in physics_attributes
        )

        stage_result = {
            "openusd_version": list(Usd.GetVersion()),
            "root_layer_relative_path": usd_relative_path,
            "root_layer_sha256": _sha256_bytes(snapshots[usd_relative_path]),
            "root_layer_size_bytes": len(snapshots[usd_relative_path]),
            "default_prim_path": default_path,
            "up_axis": stage_up_axis,
            "meters_per_unit": stage_meters_per_unit,
            "meters_per_unit_authored": bool(
                stage.HasAuthoredMetadata(UsdGeom.Tokens.metersPerUnit)
            ),
            **stage_time_values,
            "documentation": str(stage.GetRootLayer().documentation or ""),
            "stage_metadata": _json_value(stage.GetPseudoRoot().GetAllMetadata()),
            "sublayers": list(stage.GetRootLayer().subLayerPaths),
            "used_layers": sorted(used_layers, key=lambda row: row["relative_path"]),
            "prim_count": len(stage_prims),
            "authored_property_count": total_authored_properties,
            "raw_schema_bindings": sorted(raw_schema_rows, key=lambda row: row["prim_path"]),
            "physics_attributes": sorted(
                physics_attributes, key=lambda row: (row["prim_path"], row["name"])
            ),
            "physics_relationships": sorted(
                physics_relationships, key=lambda row: (row["prim_path"], row["name"])
            ),
            "materials": sorted(material_prims, key=lambda row: row["prim_path"]),
            "shaders": sorted(shader_prims, key=lambda row: row["prim_path"]),
            "source_metadata": {
                "documentation": str(stage.GetRootLayer().documentation or ""),
                "provider_authored_attributes": sorted(
                    (row for row in physics_attributes if row["name"].startswith("lightwheelusd:")),
                    key=lambda row: (row["prim_path"], row["name"]),
                ),
                "generator_identity_inferred": False,
            },
            "dependencies": dependency_values,
            "precomposition_dependency_preflight": preflight,
            "composition_arcs": sorted(
                composition_arcs,
                key=lambda row: (row["prim_path"], row["arc_type"], row["asset_path"]),
            ),
            "computed_dependency_layers": sorted(set(computed_dependency_layers)),
            "computed_dependency_assets": sorted(set(computed_dependency_assets)),
            "surface_mesh": surface,
            "tetmeshes": sorted(tetmeshes, key=lambda row: row["prim_path"]),
            "deformable_entity_binding": {
                "selected_body_prim_path": selected_body_path,
                "selected_surface_mesh_prim_path": selected_mesh_path,
                "selection_method": selection_method or None,
                "cooking_source_relationship_join_valid": cooking_relationship_join_valid,
                "cooking_source_relationships": sorted(
                    cooking_bindings,
                    key=lambda row: (row["body_prim_path"], tuple(row["targets"])),
                ),
                "visual_material_binding": visual_material_binding,
                "physics_material_binding": physics_material_binding,
                "body_scoped_tetmeshes": sorted(body_tetmeshes, key=lambda row: row["prim_path"]),
                "selected_visual_material_dependencies": selected_visual_material_dependencies,
                "authored_deformable_metadata_on_selected_entity": (
                    authored_deformable_metadata_on_selected_entity
                ),
            },
            "dome_lights": sorted(dome_lights, key=lambda row: row["prim_path"]),
            "dome_lights_inside_default_prim": sorted(
                dome_lights_in_default, key=lambda row: row["prim_path"]
            ),
            "compatibility": {
                "literal_physx_deformable_body_api_authored": has_standard_body_schema,
                "literal_physx_deformable_material_api_authored": has_standard_material_schema,
                "nonempty_tetmesh_authored": has_nonempty_tetmesh,
                "palatial_shell_intent_authored": palatial_shell_intent,
                "static_rigid_contract_authored": static_rigid_contract,
                "evidence_scoped_to_selected_body_mesh_and_material": True,
            },
            "dimension_alignment": {
                "asset_dimensions_m": dimensions,
                "observed_dimensions_m": list(observed_dimensions_m),
                "relative_errors": relative_errors,
                "within_one_percent": dimensions_match,
                "required_bake_scale_xyz": required_bake_scale,
            },
            "hazards": sorted(set(hazards)),
            "blockers": sorted(set(blockers)),
        }
        return stage_result


def inspect_external_simready_deformable_asset(
    *,
    archive_path: str | os.PathLike[str],
    expanded_root: str | os.PathLike[str],
    usd_path: str | os.PathLike[str],
    observation_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Inspect one external candidate without accepting caller-authored claims.

    Args:
        archive_path: ZIP received from the external authoring service.
        expanded_root: Expanded package root whose files must exactly match the ZIP.
        usd_path: USD entrypoint underneath ``expanded_root``.
        observation_path: Frozen metric-observation JSON using
            :data:`OBSERVATION_SCHEMA_VERSION`.

    Returns:
        A deterministic, digest-bound inspection receipt.  Passing inspection
        never upgrades the source to a native or physically equivalent asset.
    """

    archive_content, archive_display, archive_metadata = _read_absolute_file_once(
        archive_path,
        label="archive",
        maximum_size=_MAX_ARCHIVE_BYTES,
    )
    expanded_files, expanded_display = _snapshot_directory(expanded_root)
    usd_relative_path = _relative_path_within(expanded_display, usd_path, label="usd")
    _, _, usd_display = _absolute_parts(usd_path, label="usd")
    if PurePosixPath(usd_relative_path).suffix.lower() not in _USD_LAYER_SUFFIXES:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_usd_suffix_invalid"]
        )
    if usd_relative_path not in expanded_files or not expanded_files[usd_relative_path]:
        raise ExternalSimreadyDeformableAssetError(["external_simready_deformable_usd_missing"])

    archive_files, archive_rows = _snapshot_archive(archive_content)
    if set(archive_files) != set(expanded_files) or any(
        archive_files[name] != expanded_files[name]
        for name in archive_files.keys() & expanded_files.keys()
    ):
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_archive_expanded_identity_mismatch"]
        )

    observation_content, observation_display, _ = _read_absolute_file_once(
        observation_path,
        label="observation",
        maximum_size=_MAX_OBSERVATION_BYTES,
    )
    observation = _parse_observation(observation_content)
    observation, source_topology_content, source_topology_display = (
        _join_source_topology_observation(
            observation,
            observation_display=observation_display,
        )
    )
    stage = _inspect_stage_snapshot(
        expanded_files,
        usd_relative_path=usd_relative_path,
        observed_dimensions_m=observation["dimensions_m"],
    )

    file_rows = [
        {
            "relative_path": name,
            "size_bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
        for name, content in sorted(expanded_files.items())
    ]
    tree_digest = canonical_digest({"files": file_rows})
    topology_admitted = bool(stage["surface_mesh"].get("closed_oriented_manifold"))
    material_binding_admitted = bool(
        stage["deformable_entity_binding"].get("visual_material_binding")
    )
    static_candidate_admitted = topology_admitted and material_binding_admitted
    status = PENDING_STATUS if static_candidate_admitted else REJECTED_STATUS
    blockers = sorted(
        {
            *stage["blockers"],
            "external_simready_deformable_rights_and_provider_output_terms_unresolved",
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "input_paths": {
            "archive": archive_display,
            "expanded_root": expanded_display,
            "usd": usd_display,
            "observation": observation_display,
            "source_topology_receipt": source_topology_display,
        },
        "source_package": {
            "archive_size_bytes": archive_metadata.st_size,
            "archive_sha256": _sha256_bytes(archive_content),
            "archive_member_count": len(archive_rows),
            "archive_members": archive_rows,
            "expanded_file_count": len(file_rows),
            "expanded_total_size_bytes": sum(row["size_bytes"] for row in file_rows),
            "expanded_tree_digest": tree_digest,
            "expanded_files": file_rows,
            "archive_expanded_identity_verified": True,
            "source_bytes_mutated_by_inspector": False,
            "inspection_bound_to_snapshot_not_live_path": True,
            "source_retention_rights_evaluated": False,
            "source_may_be_retained_as_immutable_conversion_input": False,
            "source_is_technically_snapshot_eligible_for_later_rights_gated_conversion": True,
        },
        "observation": {
            **observation,
            "observation_file_size_bytes": len(observation_content),
            "observation_file_sha256": _sha256_bytes(observation_content),
            "source_topology_receipt_file_size_bytes": len(source_topology_content),
            "source_topology_receipt_file_sha256": _sha256_bytes(source_topology_content),
        },
        "openusd_inspection": stage,
        "pinned_physx_conversion": {
            "required": True,
            "source_representation": "closed_surface_mesh_pending_native_cook",
            "derived_runtime_usd_must_be_separate": True,
            "source_usd_must_remain_immutable": True,
            "bake_metric_scale_before_cooking": True,
            "exclude_embedded_lighting_and_provider_auto_cook_prims": True,
            "repository": ISAACLAB_REPOSITORY,
            "commit": ISAACLAB_COMMIT,
            "tree": ISAACLAB_TREE,
            "loader_source": PINNED_DEFORMABLE_LOADER_SOURCE,
            "loader_literal_schema_predicate_line": PINNED_DEFORMABLE_LOADER_SCHEMA_LINE,
            "loader_missing_schema_error_lines": list(PINNED_DEFORMABLE_LOADER_ERROR_LINES),
            "authoring_source": PINNED_AUTHORING_SOURCE,
            "authoring_source_lines": list(PINNED_AUTHORING_LINES),
            "authoring_api": PINNED_AUTHORING_API,
            "cooking_api": PINNED_COOKING_API,
            "runtime_class": PINNED_RUNTIME_CLASS,
            "required_runtime_schema": STANDARD_PHYSX_BODY_SCHEMA,
            "required_material_schema": STANDARD_PHYSX_MATERIAL_SCHEMA,
            "native_gates_after_conversion": [
                "usd_composition_and_deformable_cooking",
                "soft_body_tensor_view_and_nodal_readback",
                "genuine_gripper_contact_lift_release",
                "nodal_reset_repeatability",
                "settling_strain_and_solver_stability",
                "external_wrist_overview_camera_capture",
                "applied_parameter_readback",
            ],
        },
        "claim_ceiling": {
            "maximum_claim": CLAIM_CEILING,
            "geometry_and_material_candidate_inspected": static_candidate_admitted,
            "authored_deformable_metadata_observed": stage["deformable_entity_binding"][
                "authored_deformable_metadata_on_selected_entity"
            ],
            "standard_physx_runtime_asset": False,
            "simready_asset_admitted": False,
            "native_simulator_qualified": False,
            "visually_aligned_replacement": False,
            "physically_equivalent_real_material": False,
            "rights_and_provider_output_terms": "not_evaluated",
            "provider_or_gpu_execution": "not_performed",
        },
        "receipt_integrity": {
            "canonical_self_digest_only": True,
            "authenticity_or_origin_proven_by_self_digest": False,
            "downstream_replay_with_independently_frozen_expected_digest_required": True,
        },
        "blockers": blockers,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


def verify_external_simready_deformable_asset_inspection(
    *,
    archive_path: str | os.PathLike[str],
    expanded_root: str | os.PathLike[str],
    usd_path: str | os.PathLike[str],
    observation_path: str | os.PathLike[str],
    expected_receipt_digest: str,
) -> dict[str, Any]:
    """Replay exact source paths and match an independently frozen receipt digest."""

    if not _valid_digest(expected_receipt_digest):
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_expected_receipt_digest_invalid"]
        )
    replay = inspect_external_simready_deformable_asset(
        archive_path=archive_path,
        expanded_root=expanded_root,
        usd_path=usd_path,
        observation_path=observation_path,
    )
    if replay["receipt_digest"] != expected_receipt_digest:
        raise ExternalSimreadyDeformableAssetError(
            ["external_simready_deformable_receipt_replay_mismatch"]
        )
    return replay


__all__ = [
    "CLAIM_CEILING",
    "ExternalSimreadyDeformableAssetError",
    "OBSERVATION_SCHEMA_VERSION",
    "PENDING_STATUS",
    "REJECTED_STATUS",
    "SCHEMA_VERSION",
    "SOURCE_TOPOLOGY_SCHEMA_VERSION",
    "inspect_external_simready_deformable_asset",
    "verify_external_simready_deformable_asset_inspection",
]
