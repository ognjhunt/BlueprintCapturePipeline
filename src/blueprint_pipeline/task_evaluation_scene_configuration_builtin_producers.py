"""Execute the three admitted GPU stage tools from one sealed toolchain.

The scene recipe cannot name commands.  A production-published toolchain binds
one executable to each admitted adapter identity, and this module invokes those
fixed bytes with file-path environment variables.  Secret values are read only
for log redaction and never placed in argv, JSON inputs, receipts, or outputs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404 - executable is full-byte toolchain-bound
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest, canonical_json
from .immutable_file_digest import sha256_file
from .immutable_tree_walk import path_order, walk_tree
from .task_evaluation_scene_configuration_component_package import (
    TaskEvaluationSceneConfigurationComponentPackageError,
    validate_scene_configuration_component_package,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    GPU_STAGE_TIMEOUT_SECONDS,
    STAGE_DEADLINE_EPOCH_ENV,
)
from .task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
    SceneConfigurationStageProducerIdentity,
    SceneConfigurationStageProducerRegistry,
    TaskEvaluationSceneConfigurationStageProducerError,
)


TOOLCHAIN_SCHEMA_VERSION = "task_evaluation_scene_configuration_toolchain.v1"
TOOLCHAIN_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
# Compatibility alias retained for focused tests and existing callers.  The
# canonical mapping lives with the serialized parent-runtime contract so the
# parent budget cannot drift from the child allowances it must cover.
_STAGE_TIMEOUTS_SECONDS = GPU_STAGE_TIMEOUT_SECONDS
_EXPECTED_ARTIFACT_ROLES = {
    "artifixer3d_observed_object_removal": frozenset(
        {
            "configured_appearance_without_source_object",
            "appearance_removal_receipt",
            "appearance_visual_review_receipt",
            "configured_task_thumbnail",
            "provider_render_reference_manifest",
        }
    ),
    "content_agents_rigid_replacement": frozenset(
        {
            "replacement_asset",
            "replacement_authoring_receipt",
            "replacement_graph_spec",
        }
    ),
    "simready_native_import_qualification": frozenset(
        {"native_import_runtime_result"}
    ),
}
_DIAGNOSTIC_REJECTED_ARTIFACT_ROLES = frozenset(
    {
        "diagnostic_rejected_appearance_candidate",
        "appearance_rejection_receipt",
        "appearance_visual_review_execution",
        "provider_render_reference_manifest",
    }
)
_SUPPLEMENTAL_DESTINATION_ARTIFACT_ROLES = {
    "simready_native_import_qualification": frozenset(
        {"destination_native_import_runtime_result"}
    ),
}


def expected_stage_artifact_roles(
    adapter_id: str, *, supplemental_destination: bool
) -> frozenset[str]:
    """Exact artifact roles one GPU stage component must emit.

    A recipe that binds a supplemental passive destination makes the native
    import component emit the destination's runtime result too; no other
    component grows an extra role, and an undeclared destination is refused.
    """

    roles = _EXPECTED_ARTIFACT_ROLES[adapter_id]
    if supplemental_destination:
        roles = roles | _SUPPLEMENTAL_DESTINATION_ARTIFACT_ROLES.get(
            adapter_id, frozenset()
        )
    return roles


def envelope_declares_supplemental_destination(envelope: Any) -> bool:
    recipe = envelope.get("recipe") if isinstance(envelope, Mapping) else None
    return isinstance(recipe, Mapping) and isinstance(
        recipe.get("supplemental_destination"), Mapping
    )


_SECRET_ENVIRONMENT_FILES = (
    "OPENAI_API_KEY_FILE",
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE",
)
_RAW_SECRET_ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "BLUEPRINT_OPENAI_ADMIN_KEY",
)
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationStageProducerError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationStageProducerError(code)
    return dict(value)


def _relative_file(root: Path, value: Any, *, code: str) -> Path:
    relative = str(value or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise TaskEvaluationSceneConfigurationStageProducerError(code)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationStageProducerError(code) from exc
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationSceneConfigurationStageProducerError(code)
    return path


def _read_toolchain_manifest(
    *, root: Path, expected_source_commit: str
) -> tuple[dict[str, Any], Path]:
    """The sealed manifest of a read-only published toolchain, bytes unread."""
    if root.is_symlink() or not root.is_dir():
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_root_invalid"
        )
    if root.stat().st_mode & 0o222:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_root_invalid:not_read_only"
        )
    manifest_path = root / f"{TOOLCHAIN_SCHEMA_VERSION}.json"
    manifest = _read(
        manifest_path, code="scene_configuration_toolchain_manifest_invalid"
    )
    _check_toolchain_manifest(manifest, expected_source_commit=expected_source_commit)
    return manifest, manifest_path


def _check_toolchain_manifest(manifest: Mapping[str, Any], *, expected_source_commit: str) -> None:
    stages = manifest.get("stages")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != TOOLCHAIN_SCHEMA_VERSION
        or manifest.get("status") != "published_full_byte_readback_passed"
        or manifest.get("source_commit") != expected_source_commit
        or _COMMIT.fullmatch(str(expected_source_commit or "")) is None
        or manifest.get("full_byte_service_account_readback_passed") is not True
        or manifest.get("toolchain_digest")
        != canonical_digest(manifest, digest_field="toolchain_digest")
        or not isinstance(stages, Mapping)
        or not isinstance(files, list)
        or not files
    ):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_manifest_invalid"
        )


def _toolchain_inventory(files: Sequence[Any]) -> dict[str, tuple[str, int, bool]]:
    """The manifest's file rows as relative path -> (sha256, size, executable)."""
    inventory: dict[str, tuple[str, int, bool]] = {}
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_inventory_invalid:row={index}"
            )
        relative = str(row.get("relative_path") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size_bytes")
        executable = row.get("executable")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in inventory
            or _DIGEST.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            # An empty file is legitimate (py.typed, REQUESTED, __init__.py: 675 in
            # the ARTISAN runtime, 2026-10-13); its digest pins it like any other.
            or size < 0
            or not isinstance(executable, bool)
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_inventory_invalid:row={index}"
            )
        inventory[relative] = (digest, size, executable)
    return inventory


def _validate_toolchain(
    *, root: Path, expected_source_commit: str, verify_bytes: bool = True
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest, manifest_path = _read_toolchain_manifest(
        root=root, expected_source_commit=expected_source_commit
    )
    stages = manifest["stages"]
    inventory = _toolchain_inventory(manifest["files"])
    # Every refusal names the entry it refused. These are paths inside our own
    # sealed toolchain, never site data, and a run that stops here must say
    # what changed: the machine is usually gone by the time anyone looks.
    observed: set[str] = set()
    for relative, entry in walk_tree(root):
        if relative == manifest_path.name:
            continue
        if entry.is_symlink():
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_symlink_forbidden:{relative}"
            )
        info = entry.stat(follow_symlinks=False)
        if entry.is_dir(follow_symlinks=False):
            if info.st_mode & 0o222:
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    f"scene_configuration_toolchain_not_read_only:{relative}"
                )
            continue
        if not entry.is_file(follow_symlinks=False):
            continue
        expected = inventory.get(relative)
        if (
            expected is None
            or info.st_size != expected[1]
            # Manifests are hashed even when bytes are not: the bundle builder
            # reads them from this tree to decide what the receipt claims.
            or ((verify_bytes or relative.endswith(".json"))
                and _sha256(Path(entry.path)) != expected[0])
            or bool(info.st_mode & 0o111) != expected[2]
            or info.st_mode & 0o222
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_file_invalid:{relative}"
            )
        observed.add(relative)
    if observed != set(inventory):
        missing = sorted(set(inventory) - observed, key=path_order)
        raise TaskEvaluationSceneConfigurationStageProducerError(
            f"scene_configuration_toolchain_inventory_incomplete:{missing[0]}:"
            f"missing_count={len(missing)}"
        )
    entrypoints: dict[str, Path] = {}
    expected_ids = {identity.adapter_id for identity in ADMITTED_PRODUCER_IDENTITIES}
    if set(stages) != expected_ids:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_stage_set_invalid"
        )
    for adapter_id, row in stages.items():
        if (
            not isinstance(row, Mapping)
            or row.get("network_policy")
            not in {"openai_api_only", "provider_and_openai_api", "disabled"}
            or row.get("secrets_via_files_only") is not True
            or row.get("raw_secret_values_in_argv_or_logs") is not False
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            )
        executable = _relative_file(
            root,
            row.get("entrypoint"),
            code=f"scene_configuration_toolchain_stage_invalid:{adapter_id}",
        )
        relative = executable.relative_to(root).as_posix()
        if relative not in inventory or inventory[relative][2] is not True:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            )
        entrypoints[str(adapter_id)] = executable
        component = _relative_file(
            root,
            row.get("component_entrypoint"),
            code=f"scene_configuration_toolchain_stage_invalid:{adapter_id}",
        )
        component_relative = component.relative_to(root).as_posix()
        package_manifest = _relative_file(
            root,
            row.get("component_package_manifest"),
            code=f"scene_configuration_toolchain_stage_invalid:{adapter_id}",
        )
        package_manifest_relative = package_manifest.relative_to(root).as_posix()
        try:
            package = validate_scene_configuration_component_package(
                root=package_manifest.parent,
                expected_adapter_id=str(adapter_id),
                verify_bytes=verify_bytes,
            )
        except TaskEvaluationSceneConfigurationComponentPackageError as exc:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            ) from exc
        if (
            component_relative not in inventory
            or inventory[component_relative][2] is not True
            or package_manifest_relative not in inventory
            or not _DIGEST.fullmatch(
                str(row.get("component_package_digest") or "")
            )
            or row.get("component_package_digest") != package["package_digest"]
            or component
            != (package_manifest.parent / package["driver_entrypoint"]).resolve()
            or row.get("network_policy") != package["network_policy"]
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            )
    return manifest, entrypoints


#: The provider bundle manifest the bundle builder writes beside the toolchain
#: it ships (``provider_runtime/<this>`` next to ``provider_runtime/toolchain``).
_PROVIDER_BUNDLE_MANIFEST = "task_evaluation_scene_configuration_provider_bundle.v1.json"
TOOLCHAIN_RESTORATION_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_toolchain_restoration.v1"
)
_RESTORATION_LIST_LIMIT = 32


def _loaded_runtime_root() -> Path:
    """The directory this ``blueprint_pipeline`` package was imported from."""
    return Path(__file__).resolve().parent.parent


def is_run_toolchain_copy(root: str | Path) -> bool:
    """Whether ``root`` is the toolchain unpacked into this run's own bundle.

    A run (the GPU container, or the control plane's CPU prestage) unpacks its
    bundle to ``provider_runtime/``: the toolchain at ``provider_runtime/toolchain``,
    the bundle manifest beside it, and this package imported from
    ``provider_runtime/blueprint_pipeline``. A published toolchain on the control
    plane has none of that: it lives under ``system-runtimes/scene-configuration/``
    and the deployed code is imported from the release checkout. All must hold,
    so restoration can never reach a published toolchain.

    The root itself must not be a link, but an ancestor may be (``/workspace``
    on a provider image): the parent is compared with the import location by
    real path on both sides, since the entrypoint passes the unresolved path.
    """
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or path.name != "toolchain":
        return False
    try:
        info = path.lstat()
        bundle = (path.parent / _PROVIDER_BUNDLE_MANIFEST).lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and stat.S_ISREG(bundle.st_mode)
        and Path(os.path.realpath(path.parent)) == Path(os.path.realpath(_loaded_runtime_root()))
    )


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _open_at(name: str, directory: int, info: os.stat_result, *, flags: int) -> int | None:
    """``name`` under the ``directory`` handle, if it is still the entry ``info`` saw.

    Opened without following a link, relative to a handle we already hold, so
    no path string is resolved again and a name swapped meanwhile is refused.
    """
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError:
        return None
    try:
        opened = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
        os.close(descriptor)
        return None
    return descriptor


def _entries(directory: int) -> list[tuple[str, os.stat_result]]:
    # scandir on a handle stats each name relative to it (no link followed).
    with os.scandir(directory) as listing:
        return [(entry.name, entry.stat(follow_symlinks=False)) for entry in listing]


def _mount_points_under(root: Path) -> frozenset[str] | None:
    """Mount points below ``root``, relative to it, from this process's mount table.

    A bind mount of the same filesystem keeps its ``st_dev``, so the device
    alone does not show it. Paths here are only compared, never acted on. The
    table is read as bytes (a mount path need not be UTF-8); ``None`` means it
    could not be read or parsed, and then no directory is removed at all.
    """
    try:
        table = Path("/proc/self/mountinfo").read_bytes()
        base = os.fsencode(os.path.realpath(root))
        points = set()
        for line in table.splitlines():
            fields = line.split(b" ")
            if len(fields) < 5:
                continue
            point = re.sub(rb"\\([0-7]{3})", lambda match: bytes([int(match.group(1), 8) & 0xFF]), fields[4])
            if point.startswith(base + b"/"):
                points.add(os.fsdecode(point[len(base) + 1:]))
        return frozenset(points)
    except Exception:  # noqa: BLE001 - any failure means the table is not known
        return None


class _Removal:
    """Remove one foreign directory's contents and then the directory, iteratively.

    Every step is relative to a handle; a name on another filesystem or one
    swapped meanwhile stops the removal of that directory (it is left for the
    full check), never followed.
    """

    def __init__(self, *, device: int, mounts: frozenset[str], unprivileged: bool) -> None:
        self.device = device
        self.mounts = mounts
        self.unprivileged = unprivileged
        self.count = 0

    def remove(self, parent: int, name: str, info: os.stat_result, relative: str) -> str | None:
        top = _open_at(name, parent, info, flags=_DIRECTORY_FLAGS)
        if top is None:
            return "entry_changed"
        stack: list[tuple[int, str | None, int | None, list[tuple[str, os.stat_result]], str]] = []
        stack.append((top, name, parent, [], relative))
        try:
            if self.unprivileged:
                os.fchmod(top, 0o700)
            stack[-1][3].extend(_entries(top))
            while stack:
                handle, own_name, own_parent, pending, own_relative = stack[-1]
                if not pending:
                    stack.pop()
                    os.close(handle)
                    if own_parent is not None and own_name is not None:
                        os.rmdir(own_name, dir_fd=own_parent)
                    continue
                child, child_info = pending.pop()
                child_relative = own_relative + "/" + child
                if child_info.st_dev != self.device or child_relative in self.mounts:
                    return "other_filesystem"
                self.count += 1
                if stat.S_ISDIR(child_info.st_mode):
                    opened = _open_at(child, handle, child_info, flags=_DIRECTORY_FLAGS)
                    if opened is None:
                        return "entry_changed"
                    stack.append((opened, child, handle, [], child_relative))
                    if self.unprivileged:
                        os.fchmod(opened, 0o700)
                    stack[-1][3].extend(_entries(opened))
                else:
                    # Removes the name itself; a link's target is never touched.
                    os.unlink(child, dir_fd=handle)
            return None
        finally:
            for handle, _name, _parent, _pending, _relative in stack:
                os.close(handle)


def restore_run_toolchain_copy(
    *, root: str | Path, expected_source_commit: str, expected_toolchain_digest: str
) -> dict[str, Any]:
    """Return this run's own toolchain copy to its sealed inventory.

    A process the run itself started as container root can create entries in
    the copy (root ignores read-only modes), and the stage-start check would
    then end the job over a tree whose sealed files are untouched. This removes
    every entry that is neither an inventory file, an ancestor directory of one,
    nor the manifest, and clears write bits that the root, a directory or an
    inventory file gained. It never writes an inventory file's bytes and never
    acts outside ``root``: the root is opened once without following a link and
    every later step is relative to a directory handle. An entry on another
    filesystem, a sealed file with more than one link, or a name that changed
    while we looked is left alone and recorded. The caller then runs the full
    byte check, which still refuses any changed, resized or missing sealed file.

    Only the run's unpacked copy is accepted (``is_run_toolchain_copy``), and only
    when its manifest is the one the stage input is bound to; otherwise nothing
    is changed. Returns a bounded record of what was removed, restored and left.
    """

    root = Path(root)
    if not is_run_toolchain_copy(root):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_restore_not_run_copy"
        )
    seen = root.lstat()
    try:
        root_handle = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_restore_not_run_copy"
        ) from exc
    try:
        return _restore_from_handle(root_handle, seen, mounts=_mount_points_under(root),
                                    expected_source_commit=expected_source_commit,
                                    expected_toolchain_digest=expected_toolchain_digest)
    finally:
        os.close(root_handle)


def _restore_from_handle(
    root_handle: int, seen: os.stat_result, *, mounts: frozenset[str] | None, expected_source_commit: str,
    expected_toolchain_digest: str,
) -> dict[str, Any]:
    root_info = os.fstat(root_handle)
    if (root_info.st_dev, root_info.st_ino) != (seen.st_dev, seen.st_ino):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_restore_entry_changed:."
        )
    manifest_name = f"{TOOLCHAIN_SCHEMA_VERSION}.json"
    try:
        descriptor = os.open(manifest_name, _FILE_FLAGS, dir_fd=root_handle)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("manifest is not a regular file")
            manifest = json.loads(stream.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_manifest_invalid"
        ) from exc
    if not isinstance(manifest, dict):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_manifest_invalid"
        )
    _check_toolchain_manifest(manifest, expected_source_commit=expected_source_commit)
    record: dict[str, Any] = {
        "schema_version": TOOLCHAIN_RESTORATION_SCHEMA_VERSION,
        "status": "unchanged",
        "toolchain_digest": manifest["toolchain_digest"],
        "removed": [],
        "removed_count": 0,
        "removed_entry_count": 0,
        "modes_restored": [],
        "modes_restored_count": 0,
        "left": [],
        "left_count": 0,
    }
    if manifest["toolchain_digest"] != expected_toolchain_digest:
        record["status"] = "skipped_manifest_not_bound_to_stage_input"
        return record
    inventory = _toolchain_inventory(manifest["files"])
    keep = {manifest_name, *inventory}
    for relative in inventory:
        parts = relative.split("/")
        keep.update("/".join(parts[:index]) for index in range(1, len(parts)))
    device = root_info.st_dev
    unprivileged = os.geteuid() != 0

    def note(field: str, row: dict[str, Any]) -> None:
        record[f"{field}_count"] += 1
        if len(record[field]) < _RESTORATION_LIST_LIMIT:
            record[field].append(row)

    def restored(relative: str, kind: str, mode: int) -> None:
        note("modes_restored", {"relative_path": relative, "kind": kind,
                                "before_mode": oct(mode), "after_mode": oct(mode & ~0o222)})

    # One frame per kept directory on the current path: its handle, relative
    # name, mode as found, whether we opened it for writing, and its entries.
    frames: list[dict[str, Any]] = [{
        "handle": root_handle, "relative": ".", "prefix": "",
        "mode": stat.S_IMODE(root_info.st_mode), "opened": False, "entries": _entries(root_handle)}]
    try:
        while frames:
            frame = frames[-1]
            if not frame["entries"]:
                frames.pop()
                mode = frame["mode"]
                if mode & 0o222:
                    os.fchmod(frame["handle"], mode & ~0o222)
                    restored(frame["relative"], "directory", mode)
                elif frame["opened"]:
                    os.fchmod(frame["handle"], mode)
                if frame["handle"] != root_handle:
                    os.close(frame["handle"])
                continue
            name, info = frame["entries"].pop()
            relative = frame["prefix"] + name
            if info.st_dev != device or relative in (mounts or ()):
                # A mount inside the tree: never act on another filesystem.
                note("left", {"relative_path": relative, "reason": "other_filesystem"})
                continue
            if relative in keep:
                # A kept name of the wrong kind (a link, a file where a directory
                # belongs) is left for the full check to refuse by name.
                if stat.S_ISDIR(info.st_mode):
                    handle = _open_at(name, frame["handle"], info, flags=_DIRECTORY_FLAGS)
                    if handle is None:
                        note("left", {"relative_path": relative, "reason": "entry_changed"})
                        continue
                    # Held by the frame before listing, so a failed listing closes it too.
                    frames.append({"handle": handle, "relative": relative, "prefix": relative + "/",
                                   "mode": stat.S_IMODE(info.st_mode), "opened": False, "entries": []})
                    frames[-1]["entries"] = _entries(handle)
                elif stat.S_ISREG(info.st_mode) and relative in inventory and info.st_mode & 0o222:
                    if info.st_nlink > 1:
                        # A mode change would reach every other name of this inode.
                        note("left", {"relative_path": relative, "reason": "hard_linked"})
                        continue
                    handle = _open_at(name, frame["handle"], info, flags=_FILE_FLAGS)
                    if handle is None:
                        note("left", {"relative_path": relative, "reason": "entry_changed"})
                        continue
                    try:
                        os.fchmod(handle, stat.S_IMODE(info.st_mode) & ~0o222)
                    finally:
                        os.close(handle)
                    restored(relative, "file", stat.S_IMODE(info.st_mode))
                continue
            kind = ("symlink" if stat.S_ISLNK(info.st_mode) else
                    "directory" if stat.S_ISDIR(info.st_mode) else
                    "file" if stat.S_ISREG(info.st_mode) else "other")
            if kind == "directory" and mounts is None:
                # Without the mount table a same-filesystem bind mount below this
                # directory cannot be seen, so no directory is removed.
                note("left", {"relative_path": relative, "reason": "mount_table_unavailable"})
                continue
            if unprivileged and not frame["opened"] and not frame["mode"] & 0o200:
                # An unprivileged owner must open the parent to remove a name;
                # it is sealed again when the directory is done.
                os.fchmod(frame["handle"], frame["mode"] | 0o700)
                frame["opened"] = True
            contained = 0
            if kind == "directory":
                removal = _Removal(device=device, mounts=mounts or frozenset(), unprivileged=unprivileged)
                reason = removal.remove(frame["handle"], name, info, relative)
                if reason is not None:
                    note("left", {"relative_path": relative, "reason": reason})
                    continue
                contained = removal.count
            else:
                os.unlink(name, dir_fd=frame["handle"])
            record["removed_entry_count"] += 1 + contained
            note("removed", {"relative_path": relative, "kind": kind, "contained_entries": contained})
    finally:
        for frame in frames:
            if frame["handle"] != root_handle:
                os.close(frame["handle"])
    if record["removed_count"] or record["modes_restored_count"]:
        record["status"] = "restored"
    elif record["left_count"]:
        record["status"] = "left_for_full_check"
    return record


def read_scene_configuration_toolchain_manifest(
    *, root: str | Path, expected_source_commit: str
) -> dict[str, Any]:
    """Check the published manifest's seal and release binding, not its bytes.

    For callers that only bind ``toolchain_digest`` and hand the tree to a later
    step that validates every byte against it (the provider bundle builder).
    """

    manifest, _path = _read_toolchain_manifest(
        root=Path(root).expanduser().resolve(),
        expected_source_commit=expected_source_commit,
    )
    return manifest


def validate_scene_configuration_toolchain(
    *, root: str | Path, expected_source_commit: str, verify_bytes: bool = True
) -> dict[str, Any]:
    """Validate every toolchain byte without constructing executable handlers.

    ``verify_bytes=False`` keeps every manifest, stage, inventory, size, mode
    and completeness check and skips the per-file sha256 of every file except
    ``.json`` manifests, which callers read to make decisions. It is for the
    bundle builder, whose shipped toolchain bytes come from an archive hashed
    against this same inventory.
    """

    manifest, _entrypoints = _validate_toolchain(
        root=Path(root).expanduser().resolve(),
        expected_source_commit=expected_source_commit,
        verify_bytes=verify_bytes,
    )
    return manifest


def _secret_values(environment: Mapping[str, str]) -> list[str]:
    values: list[str] = []
    capsule_url = str(environment.get("BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_URL") or "")
    if capsule_url:
        values.append(capsule_url)
    for name in _SECRET_ENVIRONMENT_FILES:
        unresolved = str(environment.get(name) or "").strip()
        if not unresolved:
            continue
        path = Path(unresolved).expanduser()
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_secret_file_invalid:{name}"
            )
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_secret_file_invalid:{name}"
            )
        values.append(value)
    return values


def _redact(value: str, secrets: Sequence[str]) -> str:
    result = str(value or "")
    for secret in secrets:
        result = result.replace(secret, "REDACTED_SECRET")
    return redacted_failure_text(result)


def _captured_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _handler(
    *,
    identity: SceneConfigurationStageProducerIdentity,
    toolchain_root: Path,
    expected_source_commit: str,
    diagnostic_only: bool,
    runner: Runner,
    environment: Mapping[str, str],
    validated_toolchain: tuple[dict[str, Any], dict[str, Path]] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    manifest, entrypoints = validated_toolchain or _validate_toolchain(
        root=toolchain_root, expected_source_commit=expected_source_commit
    )
    executable = entrypoints[identity.adapter_id]
    toolchain_digest = str(manifest["toolchain_digest"])

    def execute(
        *,
        stage: Mapping[str, Any],
        envelope: Mapping[str, Any],
        configuration: Mapping[str, Any],
        configuration_path: Path,
        dependency_results: tuple[Mapping[str, Any], ...],
        output_root: Path,
    ) -> Mapping[str, Any]:
        input_path = output_root / "stage_production_input.v1.json"
        dependency_path = output_root / "dependency_results.v1.json"
        result_path = output_root / f"{PRODUCTION_RESULT_SCHEMA_VERSION}.json"
        input_value = {
            "schema_version": "task_evaluation_scene_configuration_stage_production_input.v1",
            "run_id": envelope["run_id"],
            "stage": dict(stage),
            "configuration": dict(configuration),
            "configuration_sha256": _sha256(configuration_path),
            "source_commit": expected_source_commit,
            "construction_source_commit": str(
                envelope.get("expected_production_commit")
                or expected_source_commit
            ),
            "execution_mode": (
                "diagnostic_only" if diagnostic_only else "production"
            ),
            "toolchain_digest": toolchain_digest,
            "construction_envelope": dict(envelope),
        }
        if configuration.get("authoring_backend") in {"astra_cad_blender_v1", "artisan_v1"} and input_path.exists():
            if (_read(input_path, code="astra_producer_retained_input_invalid") != input_value
                    or _read(dependency_path, code="astra_producer_retained_dependencies_invalid") != list(dependency_results)):
                raise TaskEvaluationSceneConfigurationStageProducerError("astra_producer_retained_input_changed")
            previous = output_root / "retained_producer_attempts"
            previous.mkdir(exist_ok=True)
            snapshot = previous / f"attempt-{len(list(previous.iterdir())):04d}"
            snapshot.mkdir()
            for path in (input_path, dependency_path, output_root / "stage_producer.log"):
                if path.exists():
                    shutil.copyfile(path, snapshot / path.name)
        input_path.write_text(canonical_json(input_value) + "\n", encoding="utf-8")
        dependency_path.write_text(
            canonical_json(list(dependency_results)) + "\n", encoding="utf-8"
        )
        run_environment = {
            **dict(environment),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(input_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(
                dependency_path
            ),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output_root),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT": str(result_path),
        }
        if diagnostic_only:
            run_environment[
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY"
            ] = "1"
        else:
            run_environment.pop(
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY", None
            )
        if any(str(run_environment.get(name) or "").strip() for name in _RAW_SECRET_ENVIRONMENT_NAMES):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_raw_secret_environment_forbidden"
            )
        secrets = _secret_values(run_environment)
        log_path = output_root / "stage_producer.log"
        timeout_seconds = _STAGE_TIMEOUTS_SECONDS[identity.adapter_id]
        run_environment[STAGE_DEADLINE_EPOCH_ENV] = str(time.time() + timeout_seconds)
        if configuration.get("authoring_backend") == "artisan_v1":
            deadline_path = output_root / "artisan_stage_deadline.v1.json"
            if deadline_path.exists():
                retained = _read(deadline_path, code="artisan_stage_deadline_invalid")
                if retained.get("configuration_sha256") != input_value["configuration_sha256"]:
                    raise TaskEvaluationSceneConfigurationStageProducerError("artisan_deadline_input_changed")
            else:
                retained = {"configuration_sha256": input_value["configuration_sha256"],
                            "deadline_epoch": float(run_environment[STAGE_DEADLINE_EPOCH_ENV])}
                deadline_path.write_text(canonical_json(retained) + "\n")
            remaining = float(retained["deadline_epoch"]) - time.time()
            if remaining <= 0:
                raise TaskEvaluationSceneConfigurationStageProducerError("artisan_stage_deadline_exhausted")
            timeout_seconds = min(timeout_seconds, remaining)
            run_environment[STAGE_DEADLINE_EPOCH_ENV] = str(retained["deadline_epoch"])
        try:
            completed = runner(
                [str(executable)],
                cwd=toolchain_root,
                env=run_environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(
                _redact(
                    f"{_captured_text(exc.stdout)}\n{_captured_text(exc.stderr)}",
                    secrets,
                ),
                encoding="utf-8",
            )
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_stage_producer_timeout:{identity.adapter_id}:"
                f"{timeout_seconds}"
            ) from exc
        log_path.write_text(
            _redact(f"{completed.stdout}\n{completed.stderr}", secrets),
            encoding="utf-8",
        )
        if completed.returncode != 0 or not result_path.is_file():
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_stage_producer_failed:{identity.adapter_id}:"
                f"{completed.returncode}"
            )
        result = _read(
            result_path, code="scene_configuration_stage_production_result_invalid"
        )
        observed_roles = {
            str(row.get("role") or "")
            for row in result.get("artifacts") or []
            if isinstance(row, Mapping)
        }
        diagnostic_rejection = (
            diagnostic_only
            and identity.adapter_id == "artifixer3d_observed_object_removal"
            and observed_roles == _DIAGNOSTIC_REJECTED_ARTIFACT_ROLES
        )
        if diagnostic_rejection:
            if (
                result.get("diagnostic_only") is not True
                or result.get("qualification_eligible") is not False
                or result.get("configured_revision_publication_permitted") is not False
                or result.get("offering_publication_permitted") is not False
                or result.get("terminal_e2e_completion_permitted") is not False
            ):
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    "scene_configuration_stage_production_diagnostic_claim_invalid"
                )
        elif observed_roles != expected_stage_artifact_roles(
            identity.adapter_id,
            supplemental_destination=envelope_declares_supplemental_destination(
                envelope
            ),
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_production_artifact_roles_invalid"
            )
        return result

    return execute


def builtin_scene_configuration_stage_producer_registry(
    *,
    expected_source_commit: str,
    toolchain_root: str | Path | None = None,
    runner: Runner = subprocess.run,
    environment: Mapping[str, str] | None = None,
    diagnostic_only: bool = False,
) -> SceneConfigurationStageProducerRegistry:
    """Resolve all GPU stage producers from one exact published toolchain."""

    values = os.environ if environment is None else environment
    unresolved = str(
        toolchain_root or values.get(TOOLCHAIN_ROOT_ENV) or ""
    ).strip()
    if not unresolved:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_environment_missing"
        )
    root = Path(unresolved).expanduser().resolve()
    # Every handler belongs to this same published tree. Validate its full
    # inventory once for this construction, rather than once per handler.
    # A later registry construction performs a fresh validation.
    validated = _validate_toolchain(root=root, expected_source_commit=expected_source_commit)
    handlers = {
        identity: _handler(
            identity=identity,
            toolchain_root=root,
            expected_source_commit=expected_source_commit,
            diagnostic_only=diagnostic_only,
            runner=runner,
            environment=values,
            validated_toolchain=validated,
        )
        for identity in ADMITTED_PRODUCER_IDENTITIES
    }
    return SceneConfigurationStageProducerRegistry(handlers)


__all__ = [
    "TOOLCHAIN_ROOT_ENV",
    "TOOLCHAIN_SCHEMA_VERSION",
    "TOOLCHAIN_RESTORATION_SCHEMA_VERSION",
    "builtin_scene_configuration_stage_producer_registry",
    "is_run_toolchain_copy",
    "restore_run_toolchain_copy",
    "validate_scene_configuration_toolchain",
]
