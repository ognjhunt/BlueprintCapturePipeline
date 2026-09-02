from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blueprint_pipeline.task_evaluation_splat_render_runtime import (
    validate_splat_render_runtime,
)
from scripts.build_task_evaluation_splat_render_runtime import (
    build_published_splat_render_runtime,
)


def _repository(root: Path) -> tuple[Path, str]:
    repository = root / "repository"
    for relative in (
        "tools/splat_render/render_splat.mjs",
        "tools/splat_render/src/render_entry.mjs",
        "tools/splat_render/harness.html",
        "tools/splat_render/package.json",
        "tools/splat_render/package-lock.json",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Blueprint Tests",
            "-c",
            "user.email=tests@blueprint.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def test_publishes_exact_release_renderer_with_full_byte_readback(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path)
    prerequisites = tmp_path / "prerequisites"
    node = prerequisites / "node"
    browser_root = prerequisites / "chromium"
    browser = browser_root / "chrome"
    node.parent.mkdir()
    node.write_bytes(b"linux-node")
    browser_root.mkdir()
    browser.write_bytes(b"linux-browser")
    node.chmod(0o555)
    browser.chmod(0o555)
    modules = prerequisites / "node_modules"
    for package in (
        "@sparkjsdev/spark",
        "fflate",
        "playwright",
        "playwright-core",
        "three",
    ):
        marker = modules / package / "index.js"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(package, encoding="utf-8")
        marker.chmod(0o444)
    (modules / "three/empty.js").write_bytes(b"")
    (modules / "three/empty.js").chmod(0o444)
    destination = tmp_path / "system-runtimes" / "splat-render" / commit

    def readback(path: Path) -> bytes:
        publication_root = next(
            parent for parent in path.parents if parent.parent == destination.parent
        )
        assert publication_root.stat().st_mode & 0o001
        assert path.stat().st_mode & 0o004
        return path.read_bytes()

    receipt = build_published_splat_render_runtime(
        repository_root=repository,
        source_commit=commit,
        node_executable=node,
        browser_root=browser_root,
        browser_executable=browser,
        node_modules_root=modules,
        output_root=destination,
        readback=readback,
        readback_actor="service-account:blueprint",
    )

    reopened = validate_splat_render_runtime(
        runtime_root=destination,
        repo_root=repository,
        allowed_roots=(tmp_path / "system-runtimes",),
    )
    assert receipt["status"] == "published_and_read_back"
    assert receipt["runtime_digest"] == reopened["identity"]["runtime_digest"]
    assert receipt["full_byte_service_account_readback_passed"] is True
    manifest = json.loads(
        (destination / "task_evaluation_splat_render_runtime.v1.json").read_text(
            encoding="utf-8"
        )
    )
    empty_row = next(
        row
        for row in manifest["files"]
        if row["relative_path"]
        == "renderer/tools/splat_render/node_modules/three/empty.js"
    )
    assert empty_row["size_bytes"] == 0
    assert empty_row["sha256"] == (
        "sha256:e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert destination.stat().st_mode & 0o222 == 0
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    assert (destination / "node/bin/node").stat().st_ino == node.stat().st_ino
    assert (destination / "browser/chrome").stat().st_ino == browser.stat().st_ino
    assert (
        destination / "renderer/tools/splat_render/node_modules/three/index.js"
    ).stat().st_ino == (modules / "three/index.js").stat().st_ino
