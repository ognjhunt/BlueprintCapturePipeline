"""Validate the immutable production runtime for exact InteriorGS renders.

The launch-preparation service deliberately has no Docker access.  Granting
that account access to the Docker socket would defeat its systemd sandbox.
Instead, production publishes one read-only runtime tree containing a Linux
Node executable, Chromium, the lockfile-resolved JavaScript dependencies, and
byte-identical renderer sources.  This module performs a full-byte inventory
readback before any source splat is opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_splat_render_runtime.v1"
DEFAULT_ENVIRONMENT_VARIABLE = "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT"
DEFAULT_ALLOWED_ROOTS = (Path("/var/lib/blueprint/task-evaluation-inputs/system-runtimes"),)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_RENDERER_FILES = (
    "tools/splat_render/render_splat.mjs",
    "tools/splat_render/src/render_entry.mjs",
    "tools/splat_render/harness.html",
    "tools/splat_render/package.json",
    "tools/splat_render/package-lock.json",
)
_REQUIRED_PACKAGES = (
    "@sparkjsdev/spark",
    "playwright",
    "playwright-core",
    "three",
)


class TaskEvaluationSplatRenderRuntimeError(ValueError):
    """The runtime is not an immutable, release-bound execution input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: Path, roots: Sequence[str | Path]) -> bool:
    resolved_roots = tuple(Path(root).expanduser().resolve() for root in roots)
    return any(path == root or root in path.parents for root in resolved_roots)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_manifest_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_manifest_invalid"
        )
    return dict(value)


def _repository_commit(repo: Path) -> str:
    result = subprocess.run(  # nosec B603 B607 - fixed read-only git argv
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    value = result.stdout.strip()
    if result.returncode or _COMMIT.fullmatch(value) is None:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_repository_identity_unavailable"
        )
    return value


def validate_splat_render_runtime(
    *,
    runtime_root: str | Path,
    repo_root: str | Path,
    allowed_roots: Sequence[str | Path] = DEFAULT_ALLOWED_ROOTS,
) -> dict[str, Any]:
    """Full-byte validate one immutable runtime and return execution paths."""

    raw_root = Path(runtime_root).expanduser()
    if raw_root.is_symlink():
        raise TaskEvaluationSplatRenderRuntimeError("splat_render_runtime_root_invalid")
    root = raw_root.resolve()
    repo = Path(repo_root).expanduser().resolve()
    if not _under(root, allowed_roots) or not root.is_dir():
        raise TaskEvaluationSplatRenderRuntimeError("splat_render_runtime_root_invalid")
    if root.stat().st_mode & 0o222:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_root_not_read_only"
        )
    if _under(root, DEFAULT_ALLOWED_ROOTS) and root.stat().st_uid != 0:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_root_not_root_owned"
        )
    manifest_path = root / f"{SCHEMA_VERSION}.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_manifest_invalid"
        )
    manifest = _read_json(manifest_path)
    supplied_digest = manifest.get("runtime_digest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "qualified_for_production_method_input"
        or manifest.get("platform") != "linux-x86_64"
        or manifest.get("full_byte_service_account_readback_passed") is not True
        or _COMMIT.fullmatch(str(manifest.get("source_commit") or "")) is None
        or _DIGEST.fullmatch(str(supplied_digest or "")) is None
        or supplied_digest
        != canonical_digest(manifest, digest_field="runtime_digest")
        or manifest.get("source_commit") != _repository_commit(repo)
    ):
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_manifest_invalid"
        )
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_inventory_invalid"
        )
    expected: dict[str, tuple[str, int, bool]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSplatRenderRuntimeError(
                "splat_render_runtime_inventory_invalid"
            )
        relative = str(row.get("relative_path") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size_bytes")
        executable = row.get("executable")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in expected
            or _DIGEST.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(executable, bool)
        ):
            raise TaskEvaluationSplatRenderRuntimeError(
                "splat_render_runtime_inventory_invalid"
            )
        expected[relative] = (digest, size, executable)
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise TaskEvaluationSplatRenderRuntimeError(
                "splat_render_runtime_symlink_forbidden"
            )
        if path.is_dir():
            if path.stat().st_mode & 0o222:
                raise TaskEvaluationSplatRenderRuntimeError(
                    "splat_render_runtime_directory_not_read_only"
                )
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        record = expected.get(relative)
        if record is None:
            raise TaskEvaluationSplatRenderRuntimeError(
                "splat_render_runtime_uninventoried_file"
            )
        digest, size, executable = record
        if (
            path.stat().st_size != size
            or _sha256(path) != digest
            or bool(path.stat().st_mode & 0o111) != executable
            or bool(path.stat().st_mode & 0o222)
        ):
            raise TaskEvaluationSplatRenderRuntimeError(
                f"splat_render_runtime_file_mismatch:{relative}"
            )
        observed.add(relative)
    if observed != set(expected):
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_inventory_incomplete"
        )
    entrypoints = manifest.get("entrypoints")
    if not isinstance(entrypoints, Mapping):
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_entrypoints_invalid"
        )
    node_relative = str(entrypoints.get("node") or "")
    browser_relative = str(entrypoints.get("browser") or "")
    renderer_root_relative = str(entrypoints.get("renderer_root") or "")
    for relative in (node_relative, browser_relative):
        if relative not in expected or expected[relative][2] is not True:
            raise TaskEvaluationSplatRenderRuntimeError(
                "splat_render_runtime_entrypoints_invalid"
            )
    renderer_root = (root / renderer_root_relative).resolve()
    if not renderer_root.is_dir() or root not in renderer_root.parents:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_entrypoints_invalid"
        )
    for relative in _RENDERER_FILES:
        source = repo / relative
        runtime = renderer_root / relative
        if (
            source.is_symlink()
            or runtime.is_symlink()
            or not source.is_file()
            or not runtime.is_file()
            or source.read_bytes() != runtime.read_bytes()
        ):
            raise TaskEvaluationSplatRenderRuntimeError(
                f"splat_render_runtime_renderer_source_mismatch:{relative}"
            )
    for package in _REQUIRED_PACKAGES:
        package_root = renderer_root / "tools/splat_render/node_modules" / package
        if package_root.is_symlink() or not package_root.is_dir():
            raise TaskEvaluationSplatRenderRuntimeError(
                f"splat_render_runtime_package_missing:{package}"
            )
    return {
        "node": str(root / node_relative),
        "browser_executable": str(root / browser_relative),
        "renderer_root": str(renderer_root),
        "identity": {
            "mode": "immutable_host_runtime",
            "schema_version": SCHEMA_VERSION,
            "runtime_digest": supplied_digest,
            "source_commit": manifest["source_commit"],
            "platform": manifest["platform"],
            "file_count": len(expected),
            "full_byte_service_account_readback_passed": True,
        },
    }


def runtime_from_environment(
    *,
    repo_root: str | Path,
    environment: Mapping[str, str] | None = None,
    allowed_roots: Sequence[str | Path] = DEFAULT_ALLOWED_ROOTS,
) -> dict[str, Any]:
    values = os.environ if environment is None else environment
    root = str(values.get(DEFAULT_ENVIRONMENT_VARIABLE) or "").strip()
    if not root:
        raise TaskEvaluationSplatRenderRuntimeError(
            "splat_render_runtime_environment_missing"
        )
    return validate_splat_render_runtime(
        runtime_root=root,
        repo_root=repo_root,
        allowed_roots=allowed_roots,
    )


__all__ = [
    "DEFAULT_ENVIRONMENT_VARIABLE",
    "SCHEMA_VERSION",
    "TaskEvaluationSplatRenderRuntimeError",
    "runtime_from_environment",
    "validate_splat_render_runtime",
]
