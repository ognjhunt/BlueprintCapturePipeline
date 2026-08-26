#!/usr/bin/env python3
"""Publish one exact-release InteriorGS renderer runtime.

The large Linux execution prerequisites (Node, Chromium, and lockfile-resolved
``node_modules``) are installed once in a governed read-only prerequisite root.
Every control-plane release then copies those bytes together with its exact
renderer sources, inventories every byte, performs a service-account readback,
and publishes an immutable runtime.  No scene bytes are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.immutable_directory_publication import (
    publish_staged_immutable_directory,
)
from blueprint_pipeline.task_evaluation_splat_render_runtime import (
    SCHEMA_VERSION,
    validate_splat_render_runtime,
)


RECEIPT_SCHEMA_VERSION = "task_evaluation_splat_render_runtime_publication.v1"
_RENDERER_FILES = (
    "tools/splat_render/render_splat.mjs",
    "tools/splat_render/src/render_entry.mjs",
    "tools/splat_render/harness.html",
    "tools/splat_render/package.json",
    "tools/splat_render/package-lock.json",
)
Readback = Callable[[Path], bytes]


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - fixed git argv
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError("splat_render_runtime_repository_invalid")
    return completed.stdout.strip()


def _copy_file(source: Path, destination: Path, *, executable: bool) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("splat_render_runtime_prerequisite_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o555 if executable else 0o444)


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("splat_render_runtime_prerequisite_invalid")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("splat_render_runtime_prerequisite_symlink_forbidden")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            _copy_file(
                path,
                target,
                executable=bool(path.stat().st_mode & 0o111),
            )


def build_published_splat_render_runtime(
    *,
    repository_root: str | Path,
    source_commit: str,
    node_executable: str | Path,
    browser_root: str | Path,
    browser_executable: str | Path,
    node_modules_root: str | Path,
    output_root: str | Path,
    readback: Readback,
    readback_actor: str,
) -> dict[str, Any]:
    """Publish exact renderer bytes and prove unprivileged full-byte readback."""

    repository = Path(repository_root).expanduser().resolve()
    node = Path(node_executable).expanduser().resolve()
    browser_bundle = Path(browser_root).expanduser().resolve()
    browser = Path(browser_executable).expanduser().resolve()
    modules = Path(node_modules_root).expanduser().resolve()
    destination = Path(output_root).expanduser().absolute()
    if (
        not readback_actor.strip()
        or _git(repository, "rev-parse", "HEAD") != source_commit
        or _git(repository, "status", "--porcelain=v1")
        or node.is_symlink()
        or not node.is_file()
        or not node.stat().st_mode & 0o111
        or browser_bundle.is_symlink()
        or not browser_bundle.is_dir()
        or browser_bundle not in browser.parents
        or browser.is_symlink()
        or not browser.is_file()
        or not browser.stat().st_mode & 0o111
        or modules.is_symlink()
        or not modules.is_dir()
    ):
        raise ValueError("splat_render_runtime_publication_input_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    installed = False
    try:
        _copy_file(node, staging / "node/bin/node", executable=True)
        _copy_tree(browser_bundle, staging / "browser")
        browser_relative = browser.relative_to(browser_bundle)
        renderer = staging / "renderer"
        for relative in _RENDERER_FILES:
            _copy_file(
                repository / relative,
                renderer / relative,
                executable=False,
            )
        _copy_tree(
            modules,
            renderer / "tools/splat_render/node_modules",
        )
        files = [
            {
                "relative_path": path.relative_to(staging).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        staging.chmod(0o755)
        for row in files:
            observed = readback(staging / row["relative_path"])
            if (
                len(observed) != row["size_bytes"]
                or _sha256_bytes(observed) != row["sha256"]
            ):
                raise ValueError("splat_render_runtime_service_readback_failed")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "qualified_for_production_method_input",
            "platform": "linux-x86_64",
            "source_commit": source_commit,
            "full_byte_service_account_readback_passed": True,
            "readback_actor": readback_actor,
            "entrypoints": {
                "node": "node/bin/node",
                "browser": (Path("browser") / browser_relative).as_posix(),
                "renderer_root": "renderer",
            },
            "files": files,
            "runtime_digest": "",
        }
        manifest["runtime_digest"] = canonical_digest(
            manifest, digest_field="runtime_digest"
        )
        manifest_path = staging / f"{SCHEMA_VERSION}.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        manifest_path.chmod(0o444)
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        staging.chmod(0o555)
        publish_staged_immutable_directory(
            staging=staging,
            destination=destination,
            manifest_name=manifest_path.name,
            output_exists_code="splat_render_runtime_publication_output_exists",
        )
        installed = True
        installed_manifest = destination / manifest_path.name
        if readback(installed_manifest) != installed_manifest.read_bytes():
            raise ValueError("splat_render_runtime_service_readback_failed")
        validate_splat_render_runtime(
            runtime_root=destination,
            repo_root=repository,
            allowed_roots=(destination.parent,),
        )
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "published_and_read_back",
            "source_commit": source_commit,
            "runtime_root": str(destination),
            "runtime_digest": manifest["runtime_digest"],
            "file_count": len(files),
            "readback_actor": readback_actor,
            "full_byte_service_account_readback_passed": True,
            "provider_mutation_performed": False,
            "paid_resource_allocated": False,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path = destination.parent / f"{destination.name}.publication.v1.json"
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((canonical_json(receipt) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        return receipt
    except Exception:
        owned = destination if installed else staging
        if owned.exists() and not owned.is_symlink():
            for path in sorted(owned.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                path.chmod(0o700 if path.is_dir() else 0o600)
            owned.chmod(0o700)
            shutil.rmtree(owned)
        raise


def _service_account_readback(user: str) -> Readback:
    def read(path: Path) -> bytes:
        return subprocess.run(
            ["sudo", "-n", "-u", user, "--", "dd", f"if={path}", "status=none"],
            check=True,
            capture_output=True,
        ).stdout

    return read


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--node-executable", required=True)
    parser.add_argument("--browser-root", required=True)
    parser.add_argument("--browser-executable", required=True)
    parser.add_argument("--node-modules-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--readback-user", required=True)
    args = parser.parse_args()
    value = build_published_splat_render_runtime(
        repository_root=args.repository_root,
        source_commit=args.source_commit,
        node_executable=args.node_executable,
        browser_root=args.browser_root,
        browser_executable=args.browser_executable,
        node_modules_root=args.node_modules_root,
        output_root=args.output_root,
        readback=_service_account_readback(args.readback_user),
        readback_actor=f"service-account:{args.readback_user}",
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
