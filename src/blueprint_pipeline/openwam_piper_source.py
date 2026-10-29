"""Verify the retained pinned upstream source without requiring Git on a worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

SOURCE_REVISION = "3950b6d9ea848dc980a82a89b4554af2c082544a"
SOURCE_ARCHIVE = f"openwam-source-{SOURCE_REVISION}.zip"
SOURCE_MANIFEST = "openwam-piper-source.v1.json"
SOURCE_MANIFEST_SHA256 = "d804f8d0e1735f95b7ce51ab2eebfd9b84d2468991f6e04f799dcaf5b91bab68"
SOURCE_ARCHIVE_SHA256 = "sha256:58e3901bc1b15643828e17350fe27d0f4d2493a0b7076251d3d6b5354c980ab7"
SOURCE_ARCHIVE_BYTES = 7_417_458


def source_manifest(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else Path(__file__).parent / "_vendor" / SOURCE_MANIFEST
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("openwam_source_manifest_changed")
    return json.loads(raw)


def stage_source(runtime: Path) -> dict:
    vendor = Path(__file__).parent / "_vendor"
    manifest = source_manifest()
    archive = vendor / SOURCE_ARCHIVE
    data = archive.read_bytes()
    if (len(data) != manifest["archive_size_bytes"]
            or "sha256:" + hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]):
        raise ValueError("openwam_source_archive_changed")
    for name in (SOURCE_ARCHIVE, SOURCE_MANIFEST):
        shutil.copyfile(vendor / name, runtime / name)
    return {"source_revision": SOURCE_REVISION, "archive_sha256": manifest["archive_sha256"],
            "archive_size_bytes": manifest["archive_size_bytes"],
            "manifest_sha256": "sha256:" + SOURCE_MANIFEST_SHA256}


def verify_source_directory(directory: str | Path, manifest_path: str | Path) -> Path:
    root = Path(directory).resolve(strict=True)
    manifest = source_manifest(manifest_path)
    expected = {row["path"] for row in manifest["files"]}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if not path.is_dir()}
    if actual != expected:
        raise ValueError("openwam_source_directory_members_changed")
    for row in manifest["files"]:
        path = root / row["path"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != row["size_bytes"]
                or any(parent.is_symlink() for parent in path.parents)):
            raise ValueError("openwam_source_file_changed")
        if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("openwam_source_file_changed")
    return root


def extract_source(archive_path: str | Path, manifest_path: str | Path, destination: Path) -> Path:
    manifest = source_manifest(manifest_path)
    data = Path(archive_path).read_bytes()
    if (len(data) != manifest["archive_size_bytes"]
            or "sha256:" + hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]):
        raise ValueError("openwam_source_archive_changed")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        rows = {row["path"]: row for row in manifest["files"]}
        if len(archive.namelist()) != len(rows) or set(archive.namelist()) != set(rows):
            raise ValueError("openwam_source_archive_members_changed")
        for name, row in rows.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("openwam_source_archive_path_invalid")
            content = archive.read(name)
            if (len(content) != row["size_bytes"]
                    or "sha256:" + hashlib.sha256(content).hexdigest() != row["sha256"]):
                raise ValueError("openwam_source_file_changed")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
                raise ValueError("openwam_source_archive_path_invalid")
            target.write_bytes(content)
    return verify_source_directory(destination, manifest_path)
