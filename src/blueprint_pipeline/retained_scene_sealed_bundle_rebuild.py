"""Rebuild a retained-scene bundle from its sealed scientific predecessor.

The retained-scene provider bundle deliberately omits the complete source PLY
and source-index arrays.  That keeps provider disclosure narrow, but it means a
new code release cannot simply reuse the old commit-bound bundle.  This module
reopens the sealed subset PLYs, reconstructs the omitted index arrays from an
exact host-resident source PLY, proves the historical bytes, and then invokes
the normal current-checkout bundle builder.

This is input materialization only.  It neither authorizes nor executes paid
compute, and it does not upgrade the predecessor's scientific or website claim.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .adp_retained_scene_render_packet import (
    PROBE_KIND,
    RetainedSceneRenderPacketError,
    _camera_contract,
    _validate_authority,
    _validate_candidate_set,
    build_retained_scene_gpu_render_bundle,
    build_retained_scene_gpu_render_request,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS, validate_task_freeze
from .gaussian_splat_decode import (
    _parse_ply_header,
    verify_standard_3dgs_ply_subset_exact,
)


REBUILD_SCHEMA = "adp009d_retained_scene_sealed_bundle_rebuild.v1"
MANIFEST_MEMBER = "provider_runtime/adp_retained_scene_gpu_render_manifest.json"
CANDIDATE_MEMBER = "provider_runtime/input/direct_evidence_successor_set.json"
REQUEST_MEMBER = "provider_runtime/source_request_manifest.json"
AUTHORITY_MEMBER = "provider_runtime/execution_authority.json"
VENDOR_MEMBER_ROOT = "provider_runtime/renderer/node_modules"
REQUIRED_VENDOR_PACKAGES = (
    "@sparkjsdev/spark",
    "fflate",
    "playwright",
    "playwright-core",
    "three",
)
DEFAULT_PRODUCTION_ROOTS = (Path("/var/lib/blueprint"),)
MAX_ARCHIVE_MEMBERS = 4_096
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024**2
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_JSON_MEMBER_BYTES = 4 * 1024**2


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if relative_to is None:
        record["path"] = str(path)
    else:
        record["relative_path"] = path.relative_to(relative_to).as_posix()
    return record


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _under(path: str | Path, roots: Sequence[str | Path], *, code: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    allowed = tuple(Path(root).expanduser().resolve() for root in roots)
    if not any(candidate == root or root in candidate.parents for root in allowed):
        raise RetainedSceneRenderPacketError([code])
    return candidate


def _source_file(
    value: str | Path,
    roots: Sequence[str | Path],
    *,
    code: str,
) -> Path:
    path = _under(value, roots, code=code)
    if path.is_symlink() or not path.is_file():
        raise RetainedSceneRenderPacketError([code])
    return path


def _prepare_output_root(value: str | Path, roots: Sequence[str | Path]) -> Path:
    path = _under(value, roots, code="retained_scene_sealed_rebuild_output_root_invalid")
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_output_root_not_empty"]
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkout_commit(repo: Path) -> str:
    result = subprocess.run(  # nosec B603 - fixed executable and arguments
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    commit = result.stdout.strip()
    if result.returncode or len(commit) != 40:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_repository_identity_unavailable"]
        )
    return commit


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_archive_member_count_invalid"]
        )
    total = 0
    result: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        path = Path(info.filename)
        mode = info.external_attr >> 16
        if (
            info.filename.startswith("/")
            or ".." in path.parts
            or stat.S_ISLNK(mode)
            or info.file_size < 0
            or info.file_size > MAX_ARCHIVE_MEMBER_BYTES
            or info.filename in result
        ):
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_archive_member_invalid"]
            )
        total += info.file_size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_archive_size_invalid"]
            )
        result[info.filename] = info
    return result


def _member_bytes(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    name: str,
    *,
    maximum_size: int = MAX_JSON_MEMBER_BYTES,
) -> bytes:
    info = members.get(name)
    if info is None or info.file_size <= 0 or info.file_size > maximum_size:
        raise RetainedSceneRenderPacketError(
            [f"retained_scene_sealed_rebuild_member_invalid:{name}"]
        )
    value = archive.read(info)
    if len(value) != info.file_size:
        raise RetainedSceneRenderPacketError(
            [f"retained_scene_sealed_rebuild_member_invalid:{name}"]
        )
    return value


def _json_member(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    name: str,
) -> dict[str, Any]:
    try:
        value = json.loads(_member_bytes(archive, members, name))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RetainedSceneRenderPacketError(
            [f"retained_scene_sealed_rebuild_json_invalid:{name}"]
        ) from exc
    if not isinstance(value, dict):
        raise RetainedSceneRenderPacketError(
            [f"retained_scene_sealed_rebuild_json_invalid:{name}"]
        )
    return value


def _expected_record(value: Any, *, code: str) -> tuple[str, int]:
    if not isinstance(value, Mapping):
        raise RetainedSceneRenderPacketError([code])
    digest = str(value.get("sha256") or "")
    size = value.get("size_bytes")
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size > MAX_ARCHIVE_MEMBER_BYTES
    ):
        raise RetainedSceneRenderPacketError([code])
    return digest, size


def _extract_member(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    name: str,
    destination: Path,
    expected: Any,
    *,
    code: str,
) -> dict[str, Any]:
    digest, size = _expected_record(expected, code=code)
    info = members.get(name)
    if info is None or info.file_size != size:
        raise RetainedSceneRenderPacketError([code])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RetainedSceneRenderPacketError([code])
    hasher = hashlib.sha256()
    written = 0
    try:
        with archive.open(info) as source, destination.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                hasher.update(chunk)
                written += len(chunk)
    except OSError as exc:
        raise RetainedSceneRenderPacketError([code]) from exc
    observed = "sha256:" + hasher.hexdigest()
    if written != size or observed != digest:
        raise RetainedSceneRenderPacketError([code])
    destination.chmod(0o440)
    return _record(destination)


def _extract_vendor_tree(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    manifest: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Reopen the exact sealed renderer dependencies without a host npm install."""

    identity = manifest.get("renderer_identity")
    expected_counts = (
        identity.get("vendor_packages") if isinstance(identity, Mapping) else None
    )
    if (
        not isinstance(expected_counts, Mapping)
        or set(expected_counts) != set(REQUIRED_VENDOR_PACKAGES)
        or any(
            not isinstance(expected_counts[package], int)
            or isinstance(expected_counts[package], bool)
            or expected_counts[package] <= 0
            for package in REQUIRED_VENDOR_PACKAGES
        )
    ):
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_vendor_manifest_invalid"]
        )
    if destination.exists() or destination.is_symlink():
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_vendor_destination_collision"]
        )

    extracted: dict[str, list[dict[str, Any]]] = {}
    for package in REQUIRED_VENDOR_PACKAGES:
        prefix = f"{VENDOR_MEMBER_ROOT}/{package}/"
        package_members = sorted(
            (name, info)
            for name, info in members.items()
            if name.startswith(prefix) and not name.endswith("/")
        )
        if len(package_members) != expected_counts[package]:
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_vendor_manifest_invalid"]
            )
        package_records: list[dict[str, Any]] = []
        for name, info in package_members:
            relative = Path(name).relative_to(VENDOR_MEMBER_ROOT)
            mode = info.external_attr >> 16
            if not relative.parts or (mode and not stat.S_ISREG(mode)):
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_vendor_member_invalid"]
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_vendor_destination_collision"]
                )
            digest = hashlib.sha256()
            written = 0
            try:
                with archive.open(info) as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_vendor_member_invalid"]
                ) from exc
            if written != info.file_size:
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_vendor_member_invalid"]
                )
            # The host-side reopened tree is immutable input. Preserve only
            # executable bits needed by package entrypoints; never inherit a
            # writable mode from the predecessor archive.
            target.chmod(0o440 | (stat.S_IMODE(mode) & 0o111))
            package_records.append(
                {
                    "relative_path": relative.as_posix(),
                    "sha256": "sha256:" + digest.hexdigest(),
                    "size_bytes": written,
                }
            )
        extracted[package] = package_records

    value: dict[str, Any] = {
        "schema_version": "adp009d_retained_scene_renderer_vendor_reopen.v1",
        "source_bundle_member_root": VENDOR_MEMBER_ROOT,
        "package_file_counts": {
            package: len(extracted[package]) for package in REQUIRED_VENDOR_PACKAGES
        },
        "files": extracted,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _link_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_destination_collision"]
        )
    # Copy instead of hard-linking because the production copy has a stricter
    # service-read mode.  chmod on a hard link would mutate the sealed source
    # inode and violate the no-overwrite predecessor contract.
    shutil.copy2(source, destination)
    destination.chmod(0o440)


def _raw_rows(path: Path) -> tuple[np.memmap, int, tuple[tuple[str, str], ...]]:
    with path.open("rb") as stream:
        fmt, count, properties, offset = _parse_ply_header(stream)
    if fmt != "binary_little_endian" or count < 1:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_standard_ply_invalid"]
        )
    row_size = 4 * len(properties)
    expected_size = offset + count * row_size
    if path.stat().st_size != expected_size:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_standard_ply_invalid"]
        )
    return (
        np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(count, row_size)),
        count,
        tuple(properties),
    )


def reconstruct_subset_indices(source: Path, subset: Path) -> np.ndarray:
    """Return the exact increasing source-row indices for a byte-exact subset."""

    source_rows, source_count, source_properties = _raw_rows(source)
    subset_rows, subset_count, subset_properties = _raw_rows(subset)
    if source_properties != subset_properties or subset_count >= source_count:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_subset_layout_invalid"]
        )
    indices = np.empty(subset_count, dtype=np.int64)
    cursor = 0
    for row_index in range(subset_count):
        target = subset_rows[row_index]
        while cursor < source_count and not np.array_equal(source_rows[cursor], target):
            cursor += 1
        if cursor >= source_count:
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_subset_not_source_subsequence"]
            )
        indices[row_index] = cursor
        cursor += 1
    return indices


def _write_exact_indices(path: Path, indices: np.ndarray, expected: Any, *, code: str) -> None:
    expected_digest, expected_size = _expected_record(expected, code=code)
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(indices, dtype=np.int64), allow_pickle=False)
    value = buffer.getvalue()
    if len(value) != expected_size or _sha256_bytes(value) != expected_digest:
        raise RetainedSceneRenderPacketError([code])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(0o440)


def _strip_absolute_path_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_absolute_path_fields(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "path" and isinstance(item, str) and Path(item).is_absolute():
            continue
        result[str(key)] = _strip_absolute_path_fields(item)
    return result


def _assert_emitted_paths_local(value: Any, allowed_roots: Sequence[str | Path]) -> None:
    roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "path" and isinstance(item, str) and Path(item).is_absolute():
                resolved = Path(item).resolve()
                if not any(resolved == root or root in resolved.parents for root in roots):
                    raise RetainedSceneRenderPacketError(
                        ["retained_scene_sealed_rebuild_non_host_binding_emitted"]
                    )
            _assert_emitted_paths_local(item, roots)
    elif isinstance(value, list):
        for item in value:
            _assert_emitted_paths_local(item, roots)


def rebuild_retained_scene_bundle_from_sealed_predecessor(
    *,
    predecessor_bundle_path: str | Path,
    source_standard_splat_path: str | Path,
    repo_root: str | Path,
    output_root: str | Path,
    allowed_host_roots: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Rehydrate exact source records and seal a current-checkout bundle."""

    host_roots = tuple(allowed_host_roots or DEFAULT_PRODUCTION_ROOTS)
    predecessor = _source_file(
        predecessor_bundle_path,
        host_roots,
        code="retained_scene_sealed_rebuild_predecessor_missing",
    )
    source = _source_file(
        source_standard_splat_path,
        host_roots,
        code="retained_scene_sealed_rebuild_source_splat_missing",
    )
    repo = Path(repo_root).expanduser().resolve()
    if repo.is_symlink() or not repo.is_dir():
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_repository_invalid"]
        )
    root = _prepare_output_root(output_root, host_roots)
    source_root = root / "rehydrated_scene"
    source_root.mkdir()

    predecessor_record = _record(predecessor)
    with zipfile.ZipFile(predecessor) as archive:
        members = _archive_members(archive)
        manifest = _json_member(archive, members, MANIFEST_MEMBER)
        candidate = _json_member(archive, members, CANDIDATE_MEMBER)
        old_request = _json_member(archive, members, REQUEST_MEMBER)

        if (
            manifest.get("schema_version") != "adp009d_retained_scene_gpu_render_bundle.v1"
            or manifest.get("probe_kind") != PROBE_KIND
            or manifest.get("status") != "ready"
            or manifest.get("blockers") not in (None, [])
            or candidate.get("receipt_digest")
            != canonical_digest(candidate, digest_field="receipt_digest")
            or candidate.get("receipt_digest") != manifest.get("candidate_set_digest")
        ):
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_predecessor_manifest_invalid"]
            )

        expected_source = candidate.get("source_standard_splat")
        source_digest, source_size = _expected_record(
            expected_source,
            code="retained_scene_sealed_rebuild_source_splat_binding_invalid",
        )
        if source.stat().st_size != source_size or _sha256_file(source) != source_digest:
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_source_splat_binding_invalid"]
            )
        source_copy = source_root / "standard_splat" / "scene_standard.ply"
        _link_exact(source, source_copy)

        union = candidate.get("shared_scene_union")
        outputs = union.get("outputs") if isinstance(union, Mapping) else None
        if not isinstance(outputs, Mapping):
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_candidate_outputs_invalid"]
            )
        deleted_record = outputs.get("deleted_source_gaussians")
        retained_record = outputs.get("retained_scene_gaussians")
        deleted_path = source_root / "shared_scene_union" / "deleted_source_gaussians.ply"
        retained_path = source_root / "shared_scene_union" / "retained_scene_gaussians.ply"
        _extract_member(
            archive,
            members,
            "provider_runtime/input/shared_deleted_source_layer.ply",
            deleted_path,
            deleted_record,
            code="retained_scene_sealed_rebuild_deleted_splat_invalid",
        )
        _extract_member(
            archive,
            members,
            "provider_runtime/input/shared_retained_scene.ply",
            retained_path,
            retained_record,
            code="retained_scene_sealed_rebuild_retained_splat_invalid",
        )

        deleted_indices = reconstruct_subset_indices(source_copy, deleted_path)
        retained_indices = reconstruct_subset_indices(source_copy, retained_path)
        expected_count = int((union.get("counts") or {}).get("source") or 0)
        if (
            expected_count <= 0
            or np.intersect1d(deleted_indices, retained_indices, assume_unique=True).size
            or not np.array_equal(
                np.union1d(deleted_indices, retained_indices),
                np.arange(expected_count, dtype=np.int64),
            )
        ):
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_indices_not_disjoint_exhaustive"]
            )
        deleted_indices_path = source_root / "shared_scene_union" / "deleted_source_indices.npy"
        retained_indices_path = source_root / "shared_scene_union" / "retained_source_indices.npy"
        _write_exact_indices(
            deleted_indices_path,
            deleted_indices,
            outputs.get("deleted_source_indices"),
            code="retained_scene_sealed_rebuild_deleted_indices_digest_mismatch",
        )
        _write_exact_indices(
            retained_indices_path,
            retained_indices,
            outputs.get("retained_source_indices"),
            code="retained_scene_sealed_rebuild_retained_indices_digest_mismatch",
        )
        if (
            verify_standard_3dgs_ply_subset_exact(
                source_copy, deleted_path, deleted_indices
            ).get("retained_rows_byte_exact")
            is not True
            or verify_standard_3dgs_ply_subset_exact(
                source_copy, retained_path, retained_indices
            ).get("retained_rows_byte_exact")
            is not True
        ):
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_subset_validation_failed"]
            )

        manifest_lanes = manifest.get("task_lanes")
        candidate_tasks = candidate.get("task_candidates")
        if (
            not isinstance(manifest_lanes, list)
            or not isinstance(candidate_tasks, list)
            or not 1 <= len(manifest_lanes) <= MAX_REPLACEMENT_OBJECTS
            or len(manifest_lanes) != len(candidate_tasks)
        ):
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_task_set_invalid"]
            )
        tasks_by_id = {
            str(row.get("task_id") or ""): row
            for row in candidate_tasks
            if isinstance(row, Mapping)
        }
        rebound_tasks: dict[str, dict[str, Any]] = {}
        request_lanes: list[dict[str, str]] = []
        for lane in manifest_lanes:
            if not isinstance(lane, Mapping):
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_task_set_invalid"]
                )
            task_id = str(lane.get("task_id") or "")
            task = tasks_by_id.get(task_id)
            if not task or task_id in rebound_tasks:
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_task_set_invalid"]
                )
            freeze_member = f"provider_runtime/input/task_freezes/{task_id}.json"
            camera_member = f"provider_runtime/input/cameras/{task_id}/cameras.v1.json"
            freeze_path = source_root / "task_freezes" / f"{task_id}.json"
            camera_path = source_root / "cameras" / task_id / "cameras.v1.json"
            _extract_member(
                archive,
                members,
                freeze_member,
                freeze_path,
                lane.get("task_freeze"),
                code="retained_scene_sealed_rebuild_task_freeze_invalid",
            )
            _extract_member(
                archive,
                members,
                camera_member,
                camera_path,
                lane.get("camera_contract"),
                code="retained_scene_sealed_rebuild_camera_contract_invalid",
            )
            freeze = validate_task_freeze(json.loads(freeze_path.read_text(encoding="utf-8")))
            if freeze.get("task_id") != task_id or freeze.get("task_freeze_digest") != task.get(
                "task_freeze_digest"
            ):
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_task_freeze_invalid"]
                )
            _camera_contract(camera_path)
            rebound_tasks[task_id] = _record(freeze_path)
            request_lanes.append(
                {
                    "task_id": task_id,
                    "camera_contract_path": camera_path.relative_to(root).as_posix(),
                }
            )

        authority_expected = manifest.get("execution_authority")
        authority_path = source_root / "execution_authority.json"
        _extract_member(
            archive,
            members,
            AUTHORITY_MEMBER,
            authority_path,
            authority_expected,
            code="retained_scene_sealed_rebuild_execution_authority_invalid",
        )
        authority = _validate_authority(authority_path)

        vendor_root = root / "rehydrated_renderer_vendor" / "node_modules"
        vendor_receipt = _extract_vendor_tree(
            archive,
            members,
            manifest,
            vendor_root,
        )
        vendor_receipt_path = root / "rehydrated_renderer_vendor" / "receipt.json"
        _write_json(vendor_receipt_path, vendor_receipt)

        request_expected = manifest.get("request")
        request_raw = _member_bytes(archive, members, REQUEST_MEMBER)
        request_digest, request_size = _expected_record(
            request_expected,
            code="retained_scene_sealed_rebuild_source_request_invalid",
        )
        if len(request_raw) != request_size or _sha256_bytes(request_raw) != request_digest:
            raise RetainedSceneRenderPacketError(
                ["retained_scene_sealed_rebuild_source_request_invalid"]
            )

    rebound = _strip_absolute_path_fields(copy.deepcopy(candidate))
    rebound["source_standard_splat"] = _record(source_copy)
    rebound_outputs = rebound["shared_scene_union"]["outputs"]
    rebound_outputs["deleted_source_gaussians"] = _record(
        deleted_path, relative_to=source_root
    )
    rebound_outputs["retained_scene_gaussians"] = _record(
        retained_path, relative_to=source_root
    )
    rebound_outputs["deleted_source_indices"] = _record(
        deleted_indices_path, relative_to=source_root
    )
    rebound_outputs["retained_source_indices"] = _record(
        retained_indices_path, relative_to=source_root
    )
    for task in rebound["task_candidates"]:
        task["task_freeze"] = rebound_tasks[str(task["task_id"])]
    rebound["sealed_predecessor_rebuild"] = {
        "predecessor_bundle_sha256": predecessor_record["sha256"],
        "predecessor_blueprint_commit": manifest.get("blueprint_commit"),
        "original_candidate_digest": candidate.get("receipt_digest"),
        "absolute_authoring_paths_followed": False,
        "source_rows_reopened": True,
        "indices_reconstructed_byte_identical": True,
        "renderer_vendor_reopened_from_sealed_predecessor": True,
    }
    rebound["receipt_digest"] = canonical_digest(rebound, digest_field="receipt_digest")
    _assert_emitted_paths_local(rebound, (root,))
    candidate_path = source_root / "direct_evidence_successor_set.json"
    _write_json(candidate_path, rebound)
    _validate_candidate_set(candidate_path)

    request_value = {
        "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_render_execution": True,
        "learned_policy_outcomes_accessed": False,
        "render_scope": str(old_request.get("render_scope") or "shared_union"),
        "candidate_set_path": candidate_path.relative_to(root).as_posix(),
        "execution_authority_path": authority_path.relative_to(root).as_posix(),
        "renderer_vendor_root": "rehydrated_renderer_vendor/node_modules",
        "task_lanes": sorted(request_lanes, key=lambda row: row["task_id"]),
        "private_upload_policy": copy.deepcopy(old_request.get("private_upload_policy")),
    }
    request = build_retained_scene_gpu_render_request(request_value)
    _assert_emitted_paths_local(request, (root,))
    request_path = root / "retained_scene_gpu_render_request.current.json"
    _write_json(request_path, request)

    bundle_root = root / "current_bundle"
    bundle = build_retained_scene_gpu_render_bundle(
        request_path=request_path,
        repo_root=repo,
        job_dir=bundle_root,
        scene_input_root=root,
    )
    current_commit = _checkout_commit(repo)
    if bundle.get("blueprint_commit") != current_commit or bundle.get("status") != "ready":
        raise RetainedSceneRenderPacketError(
            ["retained_scene_sealed_rebuild_current_bundle_invalid"]
        )

    result: dict[str, Any] = {
        "schema_version": REBUILD_SCHEMA,
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "probe_kind": PROBE_KIND,
        "source_commit_sha": current_commit,
        "predecessor_bundle": predecessor_record,
        "source_standard_splat": _record(source_copy),
        "reconstructed_indices": {
            "deleted": {
                **_record(deleted_indices_path),
                "count": int(deleted_indices.size),
            },
            "retained": {
                **_record(retained_indices_path),
                "count": int(retained_indices.size),
            },
            "disjoint": True,
            "exhaustive": True,
        },
        "candidate": {
            **_record(candidate_path),
            "receipt_digest": rebound["receipt_digest"],
        },
        "task_count": len(request_lanes),
        "task_ids": sorted(rebound_tasks),
        "camera_contracts_reopened": True,
        "task_freezes_reopened": True,
        "execution_authority": {
            **_record(authority_path),
            "authority_digest": authority["authority_digest"],
        },
        "renderer_vendor": {
            **_record(vendor_receipt_path),
            "package_file_counts": vendor_receipt["package_file_counts"],
            "receipt_digest": vendor_receipt["receipt_digest"],
            "reopened_from_sealed_predecessor": True,
        },
        "request": {
            **_record(request_path),
            "request_digest": request["request_digest"],
        },
        "bundle_receipt": {
            **_record(bundle_root / "adp_retained_scene_gpu_render_bundle_receipt.json"),
            "bundle_sha256": bundle["bundle_sha256"],
            "blueprint_commit": bundle["blueprint_commit"],
        },
        "absolute_authoring_paths_followed": False,
        "provider_mutation_performed": False,
        "paid_resource_used": False,
        "scientific_execution_performed": False,
        "website_trigger_proven": False,
        "claim_ceiling": (
            "current_commit_input_bundle_rebuilt_from_byte_exact_sealed_predecessor_"
            "not_executed_not_website_proof"
        ),
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    _assert_emitted_paths_local(result, host_roots)
    receipt_path = root / "adp009d_retained_scene_sealed_bundle_rebuild.v1.json"
    _write_json(receipt_path, result)
    return {**result, "receipt_path": str(receipt_path)}


__all__ = [
    "REBUILD_SCHEMA",
    "rebuild_retained_scene_bundle_from_sealed_predecessor",
    "reconstruct_subset_indices",
]
