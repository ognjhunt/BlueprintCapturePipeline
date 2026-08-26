#!/usr/bin/env python3
"""Install pinned Linux splat-render prerequisites without a paid resource.

This networked bootstrap is a one-time platform release action.  It downloads
the exact Node archive named below, lets the lockfile-pinned Playwright package
download its matching Chromium build, inventories every retained byte, and
publishes a root-owned read-only prerequisite tree.  Website scene runs never
invoke this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.immutable_directory_publication import (
    publish_staged_immutable_directory,
)


SCHEMA_VERSION = "task_evaluation_splat_render_prerequisites.v1"
NODE_VERSION = "22.21.1"
NODE_ARCHIVE_URL = (
    f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz"
)
NODE_ARCHIVE_SHA256 = (
    "sha256:680d3f30b24a7ff24b98db5e96f294c0070f8f9078df658da1bce1b9c9873c88"
)
Readback = Callable[[Path], bytes]
Downloader = Callable[[str], bytes]
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_tree_without_links(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".bin" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("splat_render_prerequisite_symlink_forbidden")
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            target.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)


def _default_download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310
        return response.read()


def validate_splat_render_prerequisites(
    *,
    root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Full-byte reopen one immutable prerequisite publication."""

    prerequisite = Path(root).expanduser().resolve()
    repository = Path(repository_root).expanduser().resolve()
    manifest_path = prerequisite / f"{SCHEMA_VERSION}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("splat_render_prerequisite_manifest_invalid") from exc
    entrypoints = manifest.get("entrypoints") if isinstance(manifest, dict) else None
    rows = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        prerequisite.is_symlink()
        or not prerequisite.is_dir()
        or prerequisite.stat().st_mode & 0o222
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "published_full_byte_readback_passed"
        or manifest.get("platform") != "linux-x86_64"
        or manifest.get("node_version") != NODE_VERSION
        or manifest.get("node_archive_url") != NODE_ARCHIVE_URL
        or manifest.get("node_archive_sha256") != NODE_ARCHIVE_SHA256
        or manifest.get("package_lock_sha256")
        != _sha256(repository / "tools/splat_render/package-lock.json")
        or manifest.get("full_byte_service_account_readback_passed") is not True
        or manifest.get("prerequisite_digest")
        != canonical_digest(manifest, digest_field="prerequisite_digest")
        or not isinstance(entrypoints, dict)
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("splat_render_prerequisite_manifest_invalid")
    inventory: dict[str, tuple[str, int, bool]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("splat_render_prerequisite_inventory_invalid")
        relative = str(row.get("relative_path") or "")
        record = (row.get("sha256"), row.get("size_bytes"), row.get("executable"))
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in inventory
            or not isinstance(record[0], str)
            or not record[0].startswith("sha256:")
            or not isinstance(record[1], int)
            or isinstance(record[1], bool)
            or record[1] <= 0
            or not isinstance(record[2], bool)
        ):
            raise ValueError("splat_render_prerequisite_inventory_invalid")
        inventory[relative] = record  # type: ignore[assignment]
    observed: set[str] = set()
    for path in prerequisite.rglob("*"):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise ValueError("splat_render_prerequisite_symlink_forbidden")
        if path.is_dir():
            if path.stat().st_mode & 0o222:
                raise ValueError("splat_render_prerequisite_not_read_only")
            continue
        relative = path.relative_to(prerequisite).as_posix()
        expected = inventory.get(relative)
        if (
            expected is None
            or path.stat().st_size != expected[1]
            or _sha256(path) != expected[0]
            or bool(path.stat().st_mode & 0o111) != expected[2]
            or path.stat().st_mode & 0o222
        ):
            raise ValueError(f"splat_render_prerequisite_file_invalid:{relative}")
        observed.add(relative)
    if observed != set(inventory):
        raise ValueError("splat_render_prerequisite_inventory_incomplete")
    resolved: dict[str, str] = {}
    for name in ("node", "browser_root", "browser", "node_modules"):
        relative = str(entrypoints.get(name) or "")
        path = (prerequisite / relative).resolve()
        if prerequisite not in path.parents:
            raise ValueError("splat_render_prerequisite_entrypoint_invalid")
        if name in {"browser_root", "node_modules"}:
            valid = path.is_dir()
        else:
            valid = path.is_file() and bool(path.stat().st_mode & 0o111)
        if not valid:
            raise ValueError("splat_render_prerequisite_entrypoint_invalid")
        resolved[name] = str(path)
    return {"manifest": manifest, "entrypoints": resolved}


def bootstrap_splat_render_prerequisites(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    readback: Readback,
    readback_actor: str,
    downloader: Downloader = _default_download,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Publish pinned Node, Chromium, and npm dependency bytes."""

    repository = Path(repository_root).expanduser().resolve()
    lockfile = repository / "tools/splat_render/package-lock.json"
    package = repository / "tools/splat_render/package.json"
    destination = Path(output_root).expanduser().absolute()
    if (
        not readback_actor.strip()
        or lockfile.is_symlink()
        or not lockfile.is_file()
        or package.is_symlink()
        or not package.is_file()
    ):
        raise ValueError("splat_render_prerequisite_input_invalid")
    if destination.exists():
        return validate_splat_render_prerequisites(
            root=destination,
            repository_root=repository,
        )["manifest"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    build = Path(tempfile.mkdtemp(prefix="splat-render-prerequisite-build-"))
    installed = False
    try:
        archive_bytes = downloader(NODE_ARCHIVE_URL)
        if _sha256_bytes(archive_bytes) != NODE_ARCHIVE_SHA256:
            raise ValueError("splat_render_prerequisite_node_digest_mismatch")
        archive = build / "node.tar.xz"
        archive.write_bytes(archive_bytes)
        with tarfile.open(archive, mode="r:xz") as bundle:
            bundle.extractall(build / "node-extracted", filter="data")
        node_root = build / "node-extracted" / f"node-v{NODE_VERSION}-linux-x64"
        node = node_root / "bin/node"
        npm_cli = node_root / "lib/node_modules/npm/bin/npm-cli.js"
        renderer = build / "renderer"
        renderer.mkdir()
        shutil.copyfile(lockfile, renderer / "package-lock.json")
        shutil.copyfile(package, renderer / "package.json")
        completed = runner(
            [
                str(node),
                str(npm_cli),
                "ci",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
            ],
            cwd=renderer,
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        if completed.returncode:
            raise ValueError("splat_render_prerequisite_npm_ci_failed")
        browser_cache = (build / "playwright-browsers").resolve()
        environment = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(browser_cache)}
        completed = runner(
            [
                str(node),
                str(renderer / "node_modules/playwright/cli.js"),
                "install",
                "chromium",
            ],
            cwd=renderer,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if completed.returncode:
            raise ValueError("splat_render_prerequisite_browser_install_failed")
        probe = runner(
            [
                str(node),
                "-e",
                "const {chromium}=require('playwright');"
                "process.stdout.write(chromium.executablePath())",
            ],
            cwd=renderer,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        browser = Path(probe.stdout.strip()).resolve()
        if (
            probe.returncode
            or browser.is_symlink()
            or not browser.is_file()
            or browser_cache not in browser.parents
        ):
            raise ValueError("splat_render_prerequisite_browser_invalid")
        browser_root = browser.parent
        (staging / "node/bin").mkdir(parents=True)
        shutil.copyfile(node, staging / "node/bin/node")
        (staging / "node/bin/node").chmod(0o555)
        _copy_tree_without_links(browser_root, staging / "browser")
        _copy_tree_without_links(renderer / "node_modules", staging / "node_modules")
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
        browser_relative = browser.relative_to(browser_root)
        staging.chmod(0o755)
        for row in files:
            observed = readback(staging / row["relative_path"])
            if len(observed) != row["size_bytes"] or _sha256_bytes(observed) != row["sha256"]:
                raise ValueError("splat_render_prerequisite_service_readback_failed")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "published_full_byte_readback_passed",
            "platform": "linux-x86_64",
            "node_version": NODE_VERSION,
            "node_archive_url": NODE_ARCHIVE_URL,
            "node_archive_sha256": NODE_ARCHIVE_SHA256,
            "package_lock_sha256": _sha256(lockfile),
            "entrypoints": {
                "node": "node/bin/node",
                "browser_root": "browser",
                "browser": (Path("browser") / browser_relative).as_posix(),
                "node_modules": "node_modules",
            },
            "files": files,
            "readback_actor": readback_actor,
            "full_byte_service_account_readback_passed": True,
            "prerequisite_digest": "",
        }
        manifest["prerequisite_digest"] = canonical_digest(
            manifest, digest_field="prerequisite_digest"
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
            output_exists_code="splat_render_prerequisite_output_exists",
        )
        installed = True
        if readback(destination / manifest_path.name) != (destination / manifest_path.name).read_bytes():
            raise ValueError("splat_render_prerequisite_service_readback_failed")
        return manifest
    except Exception:
        owned = destination if installed else staging
        if owned.exists() and not owned.is_symlink():
            for path in sorted(owned.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                path.chmod(0o700 if path.is_dir() else 0o600)
            owned.chmod(0o700)
            shutil.rmtree(owned)
        raise
    finally:
        shutil.rmtree(build, ignore_errors=True)


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
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--readback-user", required=True)
    args = parser.parse_args()
    value = bootstrap_splat_render_prerequisites(
        repository_root=args.repository_root,
        output_root=args.output_root,
        readback=_service_account_readback(args.readback_user),
        readback_actor=f"service-account:{args.readback_user}",
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
