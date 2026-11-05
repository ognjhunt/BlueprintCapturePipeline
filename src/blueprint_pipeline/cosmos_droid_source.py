"""Verify the retained pinned upstream source without requiring Git on a worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote
import zipfile

SOURCE_REVISION = "2f603cb114ff8b335e116060444d0b6caee3a85e"
SOURCE_ARCHIVE = f"cosmos-source-{SOURCE_REVISION}.zip"
SOURCE_MANIFEST = "cosmos-droid-source.v1.json"
SOURCE_MANIFEST_SHA256 = "ee795732f0ce4cd7ccbcf2b4bbd5ad16990a943ce5233ad42224b49b31b833c7"
SOURCE_ARCHIVE_SHA256 = "sha256:d0825b630396f62b58f4c8875066cc52ceadaf6432c9491602e83ef808a9eddc"
SOURCE_ARCHIVE_BYTES = 4756151
SOURCE_PART_BYTES = 1024**2
SOURCE_PARTS = tuple(
    f"{SOURCE_ARCHIVE}.part-{index:03d}"
    for index in range((SOURCE_ARCHIVE_BYTES + SOURCE_PART_BYTES - 1) // SOURCE_PART_BYTES)
)
# Files this source's checkpoint registry (register_checkpoints in
# cosmos_framework/inference/common/checkpoints.py) fetches while either DROID
# policy model loads: both snapshots' video tokenizer resolves to the Wan2.2 VAE.
# The source fetches them with `uvx hf download`, which the GPU image lacks and
# the offline server must not run, so provisioning fetches them exactly and the
# server answers the registry from that store. Repository, revision and filename
# are the registration's; size and sha256 are the publisher's LFS record there.
CHECKPOINT_REGISTRY_FILES = (
    {"repository": "Wan-AI/Wan2.2-TI2V-5B", "revision": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
     "filename": "Wan2.2_VAE.pth", "size_bytes": 2_818_839_170, "license": "apache-2.0",
     "sha256": "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"},
)
CHECKPOINT_REGISTRY_BYTES = sum(row["size_bytes"] for row in CHECKPOINT_REGISTRY_FILES)
CHECKPOINT_REGISTRY_STORE = "checkpoint-registry"


def checkpoint_registry_path(store: str | Path, row: dict) -> Path:
    return Path(store) / row["repository"] / row["revision"] / row["filename"]


def checkpoint_registry_downloads(store: str | Path) -> list[tuple[dict, Path]]:
    """Exact fetch rows for the registry files, each with the path the server answers."""
    return [({**row, "url": f"https://huggingface.co/{row['repository']}/resolve/{row['revision']}/"
              f"{quote(row['filename'])}?download=true"}, checkpoint_registry_path(store, row))
            for row in CHECKPOINT_REGISTRY_FILES]


def source_manifest(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else Path(__file__).parent / "_vendor" / SOURCE_MANIFEST
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("cosmos_source_manifest_changed")
    return json.loads(raw)


def stage_source(runtime: Path) -> dict:
    vendor = Path(__file__).parent / "_vendor"
    manifest = source_manifest()
    # Retain the archive in source-admissible leaves. Reassembly preserves the
    # publisher archive's exact bytes and hash; privileged release limits stay
    # unchanged, and no network fetch is needed when a bundle is prepared.
    pieces = []
    for index, name in enumerate(SOURCE_PARTS):
        part = vendor / name
        expected = min(SOURCE_PART_BYTES, SOURCE_ARCHIVE_BYTES - index * SOURCE_PART_BYTES)
        if part.is_symlink() or not part.is_file() or part.stat().st_size != expected:
            raise ValueError("cosmos_source_archive_changed")
        with part.open("rb") as stream:
            data = stream.read(expected + 1)
        if len(data) != expected:
            raise ValueError("cosmos_source_archive_changed")
        pieces.append(data)
    data = b"".join(pieces)
    if (len(data) != manifest["archive_size_bytes"]
            or "sha256:" + hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]):
        raise ValueError("cosmos_source_archive_changed")
    (runtime / SOURCE_ARCHIVE).write_bytes(data)
    shutil.copyfile(vendor / SOURCE_MANIFEST, runtime / SOURCE_MANIFEST)
    return {"source_revision": SOURCE_REVISION, "archive_sha256": manifest["archive_sha256"],
            "archive_size_bytes": manifest["archive_size_bytes"],
            "manifest_sha256": "sha256:" + SOURCE_MANIFEST_SHA256}


def verify_source_directory(directory: str | Path, manifest_path: str | Path) -> Path:
    root = Path(directory).resolve(strict=True)
    manifest = source_manifest(manifest_path)
    expected = {row["path"] for row in manifest["files"]}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if not path.is_dir()}
    if actual != expected:
        raise ValueError("cosmos_source_directory_members_changed")
    for row in manifest["files"]:
        path = root / row["path"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != row["size_bytes"]
                or any(parent.is_symlink() for parent in path.parents)):
            raise ValueError("cosmos_source_file_changed")
        if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("cosmos_source_file_changed")
    return root


def extract_source(archive_path: str | Path, manifest_path: str | Path, destination: Path) -> Path:
    manifest = source_manifest(manifest_path)
    data = Path(archive_path).read_bytes()
    if (len(data) != manifest["archive_size_bytes"]
            or "sha256:" + hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]):
        raise ValueError("cosmos_source_archive_changed")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        rows = {row["path"]: row for row in manifest["files"]}
        if len(archive.namelist()) != len(rows) or set(archive.namelist()) != set(rows):
            raise ValueError("cosmos_source_archive_members_changed")
        for name, row in rows.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("cosmos_source_archive_path_invalid")
            content = archive.read(name)
            if (len(content) != row["size_bytes"]
                    or "sha256:" + hashlib.sha256(content).hexdigest() != row["sha256"]):
                raise ValueError("cosmos_source_file_changed")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
                raise ValueError("cosmos_source_archive_path_invalid")
            target.write_bytes(content)
    return verify_source_directory(destination, manifest_path)
