from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_splat_render_runtime as runtime_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_splat_render_runtime import (
    SCHEMA_VERSION,
    TaskEvaluationSplatRenderRuntimeError,
    validate_diagnostic_splat_render_runtime,
    validate_splat_render_runtime,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime(tmp_path: Path) -> Path:
    root = tmp_path / "allowed" / "runtime"
    renderer = root / "renderer"
    for relative in (
        "tools/splat_render/render_splat.mjs",
        "tools/splat_render/src/render_entry.mjs",
        "tools/splat_render/harness.html",
        "tools/splat_render/package.json",
        "tools/splat_render/package-lock.json",
    ):
        destination = renderer / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for package in (
        "@sparkjsdev/spark",
        "playwright",
        "playwright-core",
        "three",
    ):
        marker = renderer / "tools/splat_render/node_modules" / package / "index.js"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"// {package}\n", encoding="utf-8")
    node = root / "node/bin/node"
    browser = root / "browser/chrome"
    node.parent.mkdir(parents=True)
    browser.parent.mkdir(parents=True)
    node.write_bytes(b"linux-node")
    browser.write_bytes(b"linux-chromium")
    node.chmod(0o555)
    browser.chmod(0o555)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified_for_production_method_input",
        "platform": "linux-x86_64",
        "source_commit": _commit(),
        "full_byte_service_account_readback_passed": True,
        "entrypoints": {
            "node": "node/bin/node",
            "browser": "browser/chrome",
            "renderer_root": "renderer",
        },
        "files": files,
        "runtime_digest": "",
    }
    manifest["runtime_digest"] = canonical_digest(
        manifest, digest_field="runtime_digest"
    )
    (root / f"{SCHEMA_VERSION}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o555 if path in (node, browser) else 0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)
    return root


def test_validates_every_runtime_byte_and_release_binding(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    result = validate_splat_render_runtime(
        runtime_root=runtime,
        repo_root=ROOT,
        allowed_roots=[tmp_path / "allowed"],
    )
    assert result["node"] == str(runtime / "node/bin/node")
    assert result["browser_executable"] == str(runtime / "browser/chrome")
    assert result["identity"]["source_commit"] == _commit()
    assert result["identity"]["full_byte_service_account_readback_passed"] is True


def test_repository_identity_probe_admits_exact_release_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="a" * 40 + "\n", stderr="")

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    release = Path("/opt/blueprint/task-evaluation-control-plane-releases") / ("a" * 40)

    assert runtime_module._repository_commit(release) == "a" * 40
    assert observed["argv"] == [
        "git",
        "-c",
        f"safe.directory={release}",
        "-C",
        str(release),
        "rev-parse",
        "HEAD",
    ]
    assert observed["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": 30,
    }


def test_diagnostic_runtime_binds_retained_commit_without_relabeling_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)

    def unexpected_repository_probe(_repo: Path) -> str:
        raise AssertionError("diagnostic resolver must use the explicit runtime commit")

    monkeypatch.setattr(
        runtime_module, "_repository_commit", unexpected_repository_probe
    )
    result = validate_diagnostic_splat_render_runtime(
        runtime_root=runtime,
        repo_root=ROOT,
        expected_runtime_source_commit=_commit(),
        allowed_roots=[tmp_path / "allowed"],
    )

    assert result["identity"]["source_commit"] == _commit()
    assert result["identity"]["full_byte_service_account_readback_passed"] is True


def test_rejects_byte_tamper_even_when_path_still_exists(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    node = runtime / "node/bin/node"
    node.chmod(0o755)
    node.write_bytes(b"tampered")
    node.chmod(0o555)
    with pytest.raises(
        TaskEvaluationSplatRenderRuntimeError,
        match="splat_render_runtime_file_mismatch:node/bin/node",
    ):
        validate_splat_render_runtime(
            runtime_root=runtime,
            repo_root=ROOT,
            allowed_roots=[tmp_path / "allowed"],
        )


def test_rejects_runtime_tree_writable_by_service_account(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.chmod(0o755)
    with pytest.raises(
        TaskEvaluationSplatRenderRuntimeError,
        match="splat_render_runtime_root_not_read_only",
    ):
        validate_splat_render_runtime(
            runtime_root=runtime,
            repo_root=ROOT,
            allowed_roots=[tmp_path / "allowed"],
        )
