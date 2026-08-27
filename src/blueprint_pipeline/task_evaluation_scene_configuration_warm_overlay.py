"""Immutable exact-SHA source-overlay contract for warm scene diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    validate_scene_configuration_diagnostic_checkpoint,
)


OVERLAY_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_source_overlay.v1"
OVERLAY_RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_source_overlay_receipt.v1"
)
MAX_OVERLAY_FILES = 4_096
MAX_OVERLAY_BYTES = 64 * 1024**2
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OVERLAY_SINGLE_FILES = (
    (
        "scripts/task_evaluation_scene_configuration_diagnostic_provider_runner.py",
        "provider_runtime/task_evaluation_scene_configuration_provider_runner.py",
    ),
    (
        "scripts/run_task_evaluation_scene_configuration_provider.sh",
        "provider_runtime/run_task_evaluation_scene_configuration_provider.sh",
    ),
)
_OVERLAY_REPLACEMENT_ROOTS = ("provider_runtime/blueprint_pipeline",)
_OVERLAY_EXACT_REPLACEMENT_FILES = tuple(
    provider_relative for _source_relative, provider_relative in _OVERLAY_SINGLE_FILES
)


class SceneConfigurationWarmDiagnosticError(ValueError):
    """A warm session, overlay, iteration, or closeout failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _record_path(value: Mapping[str, Any], *, code: str) -> Path:
    unresolved = Path(str(value.get("path") or "")).expanduser()
    if (
        not unresolved.is_absolute()
        or unresolved.is_symlink()
        or not unresolved.is_file()
        or unresolved.stat().st_size != value.get("size_bytes")
        or _sha256(unresolved) != value.get("sha256")
    ):
        raise SceneConfigurationWarmDiagnosticError(code)
    return unresolved.resolve()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("unsafe JSON path")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigurationWarmDiagnosticError(code) from exc
    if not isinstance(value, Mapping):
        raise SceneConfigurationWarmDiagnosticError(code)
    return dict(value)


def _absolute_directory(value: str | Path, *, code: str) -> Path:
    unresolved = Path(value).expanduser()
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(code)
    path = unresolved.resolve()
    if not path.is_dir():
        raise SceneConfigurationWarmDiagnosticError(code)
    return path


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable warm receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
    finally:
        os.close(descriptor)


def _validated_release_receipt(
    receipt_path: str | Path, *, source_commit: str
) -> dict[str, Any]:
    # Imported lazily so this module remains separable from the source-only
    # diagnostic-release PR while both are developed in disjoint worktrees.
    from .task_evaluation_scene_configuration_diagnostic_release import (  # noqa: PLC0415
        validate_scene_configuration_diagnostic_release_receipt,
    )

    return validate_scene_configuration_diagnostic_release_receipt(
        receipt_path,
        expected_source_commit=source_commit,
    )


def _overlay_sources(release_root: Path) -> list[tuple[Path, PurePosixPath, str]]:
    rows: list[tuple[Path, PurePosixPath, str]] = []
    package_root = release_root / "src/blueprint_pipeline"
    if package_root.is_symlink() or not package_root.is_dir():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_package_root_invalid"
        )
    for source in sorted(package_root.rglob("*")):
        if source.is_symlink():
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_source_symlink"
            )
        if not source.is_file() or source.name == "__pycache__" or source.suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        relative = source.relative_to(package_root).as_posix()
        rows.append(
            (
                source,
                PurePosixPath("provider_runtime/blueprint_pipeline") / relative,
                f"src/blueprint_pipeline/{relative}",
            )
        )
    for source_relative, provider_relative in _OVERLAY_SINGLE_FILES:
        source = release_root / source_relative
        if source.is_symlink() or not source.is_file():
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_entrypoint_invalid"
            )
        rows.append((source, PurePosixPath(provider_relative), source_relative))
    if not rows or len(rows) > MAX_OVERLAY_FILES:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_file_count_invalid"
        )
    return rows


def build_scene_configuration_warm_source_overlay(
    *,
    diagnostic_release_receipt_path: str | Path,
    source_commit: str,
    checkpoint_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build one deterministic, inventory-bound provider source overlay."""

    if _COMMIT.fullmatch(source_commit) is None:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_source_commit_invalid"
        )
    release = _validated_release_receipt(
        diagnostic_release_receipt_path, source_commit=source_commit
    )
    release_root = _absolute_directory(
        str(release.get("release_path") or ""),
        code="scene_configuration_warm_overlay_release_invalid",
    )
    checkpoint_path = _absolute_directory(
        checkpoint_root,
        code="scene_configuration_warm_overlay_checkpoint_invalid",
    )
    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_path
    )
    output = Path(output_root).expanduser()
    if (
        not output.is_absolute()
        or output.is_symlink()
        or output.exists()
        or output.parent.is_symlink()
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_output_invalid"
        )
    output.mkdir(parents=True, mode=0o750)
    try:
        inventory: list[dict[str, Any]] = []
        total = 0
        sources = _overlay_sources(release_root)
        for source, provider_relative, source_relative in sources:
            size = source.stat().st_size
            total += size
            mode = stat.S_IMODE(source.stat().st_mode)
            if total > MAX_OVERLAY_BYTES or mode & 0o022:
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_overlay_inventory_invalid"
                )
            inventory.append(
                {
                    "source_relative_path": source_relative,
                    "provider_relative_path": provider_relative.as_posix(),
                    "sha256": _sha256(source),
                    "size_bytes": size,
                    "mode": 0o555 if mode & 0o111 else 0o444,
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "status": "ready",
            "source_commit": source_commit,
            "remote_ref": release["remote_ref"],
            "remote_ref_tip_commit": release["remote_ref_tip_commit"],
            "diagnostic_release_receipt_digest": release["receipt_digest"],
            "source_checkpoint_digest": checkpoint["checkpoint_digest"],
            "scientific_binding_digest": checkpoint["scientific_bindings"][
                "binding_digest"
            ],
            "completed_stage_prefix_count": checkpoint[
                "completed_stage_prefix_count"
            ],
            "inventory": inventory,
            "replacement_roots": list(_OVERLAY_REPLACEMENT_ROOTS),
            "exact_replacement_files": list(_OVERLAY_EXACT_REPLACEMENT_FILES),
            "file_count": len(inventory),
            "total_bytes": total,
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "arbitrary_command_permitted": False,
            "raw_secret_values_recorded": False,
            "manifest_digest": "",
        }
        manifest["manifest_digest"] = canonical_digest(
            manifest, digest_field="manifest_digest"
        )
        manifest_path = output / f"{OVERLAY_SCHEMA_VERSION}.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        archive_path = output / "scene_configuration_warm_source_overlay.zip"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            manifest_info = zipfile.ZipInfo(
                f"overlay/{OVERLAY_SCHEMA_VERSION}.json",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            manifest_info.create_system = 3
            manifest_info.external_attr = (stat.S_IFREG | 0o444) << 16
            archive.writestr(
                manifest_info, (canonical_json(manifest) + "\n").encode("utf-8")
            )
            for (source, _provider_relative, _), row in zip(
                sources, inventory, strict=True
            ):
                info = zipfile.ZipInfo(
                    "overlay/" + row["provider_relative_path"],
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | row["mode"]) << 16
                archive.writestr(info, source.read_bytes())
        sealed_release = _validated_release_receipt(
            diagnostic_release_receipt_path, source_commit=source_commit
        )
        if sealed_release.get("receipt_digest") != release.get("receipt_digest"):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_release_changed_during_build"
            )
        receipt_path = output / f"{OVERLAY_RECEIPT_SCHEMA_VERSION}.json"
        receipt: dict[str, Any] = {
            "schema_version": OVERLAY_RECEIPT_SCHEMA_VERSION,
            "status": "ready",
            "source_commit": source_commit,
            "remote_ref": release["remote_ref"],
            "diagnostic_release_receipt": _record(
                Path(diagnostic_release_receipt_path).expanduser().resolve()
            ),
            "manifest": _record(manifest_path),
            "manifest_digest": manifest["manifest_digest"],
            "source_checkpoint_digest": checkpoint["checkpoint_digest"],
            "scientific_binding_digest": checkpoint["scientific_bindings"][
                "binding_digest"
            ],
            "completed_stage_prefix_count": checkpoint[
                "completed_stage_prefix_count"
            ],
            "overlay_archive": _record(archive_path),
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "arbitrary_command_permitted": False,
            "raw_secret_values_recorded": False,
            "receipt_path": str(receipt_path),
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        _write_exclusive(receipt_path, receipt)
        return receipt
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def validate_scene_configuration_warm_source_overlay(
    receipt_path: str | Path,
    *,
    expected_source_commit: str | None = None,
    expected_checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    """Reopen every overlay byte and reject extra or path-escaping members."""

    unresolved_receipt = Path(receipt_path).expanduser()
    if (
        not unresolved_receipt.is_absolute()
        or unresolved_receipt.is_symlink()
        or not unresolved_receipt.is_file()
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_receipt_invalid"
        )
    path = unresolved_receipt.resolve()
    receipt = _read(
        path, code="scene_configuration_warm_overlay_receipt_invalid"
    )
    archive_record = receipt.get("overlay_archive")
    manifest_record = receipt.get("manifest")
    if not isinstance(archive_record, Mapping) or not isinstance(
        manifest_record, Mapping
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_receipt_invalid"
        )
    archive_path = _record_path(
        archive_record, code="scene_configuration_warm_overlay_receipt_invalid"
    )
    manifest_path = _record_path(
        manifest_record, code="scene_configuration_warm_overlay_receipt_invalid"
    )
    manifest = _read(
        manifest_path, code="scene_configuration_warm_overlay_manifest_invalid"
    )
    if (
        receipt.get("schema_version") != OVERLAY_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "ready"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("diagnostic_only") is not True
        or receipt.get("development_only") is not True
        or receipt.get("qualification_eligible") is not False
        or receipt.get("configured_revision_publication_permitted") is not False
        or receipt.get("offering_publication_permitted") is not False
        or receipt.get("terminal_e2e_completion_permitted") is not False
        or receipt.get("arbitrary_command_permitted") is not False
        or receipt.get("raw_secret_values_recorded") is not False
        or receipt.get("receipt_path") != str(path)
        or manifest.get("schema_version") != OVERLAY_SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("manifest_digest") != receipt.get("manifest_digest")
        or manifest.get("source_commit") != receipt.get("source_commit")
        or manifest.get("source_checkpoint_digest")
        != receipt.get("source_checkpoint_digest")
        or manifest.get("scientific_binding_digest")
        != receipt.get("scientific_binding_digest")
        or manifest.get("diagnostic_only") is not True
        or manifest.get("development_only") is not True
        or manifest.get("qualification_eligible") is not False
        or manifest.get("configured_revision_publication_permitted") is not False
        or manifest.get("offering_publication_permitted") is not False
        or manifest.get("terminal_e2e_completion_permitted") is not False
        or manifest.get("arbitrary_command_permitted") is not False
        or manifest.get("raw_secret_values_recorded") is not False
        or manifest.get("replacement_roots")
        != list(_OVERLAY_REPLACEMENT_ROOTS)
        or manifest.get("exact_replacement_files")
        != list(_OVERLAY_EXACT_REPLACEMENT_FILES)
        or (
            expected_source_commit is not None
            and receipt.get("source_commit") != expected_source_commit
        )
        or (
            expected_checkpoint_digest is not None
            and receipt.get("source_checkpoint_digest")
            != expected_checkpoint_digest
        )
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_receipt_invalid"
        )
    rows = manifest.get("inventory")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) != manifest.get("file_count")
        or len(rows) > MAX_OVERLAY_FILES
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_inventory_invalid"
        )
    expected_members = {f"overlay/{OVERLAY_SCHEMA_VERSION}.json"}
    total = 0
    expected_paths: set[str] = set()
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_archive_invalid"
        ) from exc
    with archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_duplicate_member"
            )
        manifest_info = archive.getinfo(
            f"overlay/{OVERLAY_SCHEMA_VERSION}.json"
        )
        if (
            stat.S_IFMT(manifest_info.external_attr >> 16) != stat.S_IFREG
            or stat.S_IMODE(manifest_info.external_attr >> 16) != 0o444
        ):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_manifest_mode_invalid"
            )
        for row in rows:
            relative = str(row.get("provider_relative_path") or "") if isinstance(row, Mapping) else ""
            posix = PurePosixPath(relative)
            size = row.get("size_bytes") if isinstance(row, Mapping) else None
            mode = row.get("mode") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or not relative.startswith("provider_runtime/")
                or posix.is_absolute()
                or ".." in posix.parts
                or relative in expected_paths
                or type(size) is not int
                or size < 0
                or mode not in {0o444, 0o555}
                or _DIGEST.fullmatch(str(row.get("sha256") or "")) is None
            ):
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_overlay_inventory_invalid"
                )
            expected_paths.add(relative)
            member = "overlay/" + relative
            expected_members.add(member)
            try:
                info = archive.getinfo(member)
                body = archive.read(info)
            except (KeyError, OSError, zipfile.BadZipFile) as exc:
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_overlay_inventory_invalid"
                ) from exc
            total += len(body)
            if (
                info.is_dir()
                or info.file_size != size
                or "sha256:" + hashlib.sha256(body).hexdigest()
                != row.get("sha256")
                or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG
                or stat.S_IMODE(info.external_attr >> 16) != mode
            ):
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_overlay_inventory_invalid"
                )
        if set(names) != expected_members:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_extra_member"
            )
        archived_manifest = json.loads(
            archive.read(f"overlay/{OVERLAY_SCHEMA_VERSION}.json")
        )
        if archived_manifest != manifest:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_overlay_manifest_mismatch"
            )
    if total != manifest.get("total_bytes") or total > MAX_OVERLAY_BYTES:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_inventory_invalid"
        )
    release_record = receipt.get("diagnostic_release_receipt")
    if not isinstance(release_record, Mapping):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_release_receipt_invalid"
        )
    release_path = _record_path(
        release_record,
        code="scene_configuration_warm_overlay_release_receipt_invalid",
    )
    release = _validated_release_receipt(
        release_path, source_commit=str(receipt["source_commit"])
    )
    if (
        release.get("receipt_digest")
        != manifest.get("diagnostic_release_receipt_digest")
        or release.get("remote_ref") != receipt.get("remote_ref")
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_overlay_release_receipt_invalid"
        )
    return receipt

