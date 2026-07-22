"""Seal and verify the OSCAR source tree shipped in the thin GPU image.

The thin foundation intentionally omits Git and ``.git`` metadata.  A runtime
gate therefore cannot use ``git rev-parse`` as its source authority.  This
module records the reviewed Git origin/commit while Git is still available in
the builder, binds that identity to the post-patch runtime tree, and verifies
the tree again before OSCAR execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
# Build-only fixed-argv verification; the module never invokes a shell.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Sequence

from .oscar_official_release import (
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    OFFICIAL_OSCAR_SOURCE_URL,
    source_ref_is_official,
    source_url_is_official,
)


SEAL_SCHEMA_VERSION = "blueprint.oscar_runtime_source_seal.v1"
RUNTIME_ARTIFACT_SCHEMA_VERSION = "single_g1_kitchen_oscar_runtime_provenance.v1"
DEFAULT_RUNTIME_SOURCE_ROOT = "/opt/OSCAR"
DEFAULT_SEAL_PATH = "/opt/blueprint/oscar_source_provenance.json"
GIT_EXECUTABLE = "/usr/bin/git"
# Exact tree produced by OFFICIAL_OSCAR_SOURCE_COMMIT after the reviewed
# Foundation TransformerEngine compatibility patch.  This independent runtime
# expectation prevents a mutable launch environment or self-consistent forged
# seal from choosing a different tree.
OFFICIAL_OSCAR_RUNTIME_TREE_SHA256 = (
    "319f4d415f54afa05159783f388b844363e87b721c38c78e6cbb756162b29f1a"
)
_IGNORED_DIRECTORY_NAMES = {".git", ".mypy_cache", ".pytest_cache"}
_FORBIDDEN_BYTECODE_DIRECTORY_NAME = "__pycache__"
_FORBIDDEN_BYTECODE_FILE_SUFFIXES = {".pyc", ".pyo"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _tree_records(source_root: str | Path) -> list[dict[str, Any]]:
    unresolved_root = Path(source_root)
    if unresolved_root.is_symlink():
        raise ValueError("oscar_runtime_source_root_missing_or_unsafe")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise ValueError("oscar_runtime_source_root_missing_or_unsafe")
    records: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = candidate.relative_to(root)
        if any(part in _IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if (
            _FORBIDDEN_BYTECODE_DIRECTORY_NAME in relative.parts
            or candidate.suffix.lower() in _FORBIDDEN_BYTECODE_FILE_SUFFIXES
        ):
            raise ValueError("oscar_runtime_source_tree_python_bytecode_forbidden")
        if candidate.is_symlink():
            if not candidate.resolve().is_relative_to(root):
                raise ValueError("oscar_runtime_source_tree_external_symlink_forbidden")
            records.append(
                {
                    "kind": "symlink",
                    "path": relative.as_posix(),
                    "target": os.readlink(candidate),
                }
            )
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError("oscar_runtime_source_tree_special_file_forbidden")
        mode = candidate.stat().st_mode
        payload = candidate.read_bytes()
        records.append(
            {
                "executable": bool(mode & stat.S_IXUSR),
                "kind": "file",
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    if not records:
        raise ValueError("oscar_runtime_source_tree_empty")
    return records


def source_tree_evidence(source_root: str | Path) -> dict[str, Any]:
    records = _tree_records(source_root)
    return {
        "file_count": sum(row["kind"] == "file" for row in records),
        "record_count": len(records),
        "tree_sha256": hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest(),
    }


def _git_value(source_root: Path, *arguments: str) -> str:
    # The executable and both call-site argument tuples are fixed by this module.
    completed = subprocess.run(  # nosec B603
        [GIT_EXECUTABLE, "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def seal_source_tree(
    *,
    source_root: str | Path,
    output_path: str | Path,
    source_url: str = OFFICIAL_OSCAR_SOURCE_URL,
    source_commit: str = OFFICIAL_OSCAR_SOURCE_COMMIT,
    runtime_source_root: str = DEFAULT_RUNTIME_SOURCE_ROOT,
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    resolved_url = _git_value(root, "config", "--get", "remote.origin.url")
    resolved_commit = _git_value(root, "rev-parse", "HEAD")
    if not source_url_is_official(source_url) or not source_url_is_official(resolved_url):
        raise ValueError("oscar_build_source_url_not_official")
    if not source_ref_is_official(source_commit) or resolved_commit != source_commit:
        raise ValueError("oscar_build_source_commit_mismatch")
    tree = source_tree_evidence(root)
    if tree["tree_sha256"] != OFFICIAL_OSCAR_RUNTIME_TREE_SHA256:
        raise ValueError("oscar_build_runtime_tree_digest_mismatch")
    payload = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed",
        "source_url": OFFICIAL_OSCAR_SOURCE_URL,
        "source_commit": OFFICIAL_OSCAR_SOURCE_COMMIT,
        "runtime_source_root": runtime_source_root,
        "runtime_tree": tree,
        "runtime_tree_stage": "reviewed_source_plus_foundation_runtime_patch",
        "git_metadata_required_at_runtime": False,
        "raw_secret_values_recorded": False,
    }
    target = Path(output_path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("oscar_source_seal_output_path_unsafe")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_source_tree(
    *,
    source_root: str | Path,
    seal_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    seal_file = Path(seal_path)
    blockers: list[str] = []
    try:
        if seal_file.is_symlink() or not seal_file.is_file():
            raise OSError("unsafe seal path")
        loaded = json.loads(seal_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        loaded = {}
    seal = dict(loaded) if isinstance(loaded, dict) else {}
    expected_tree = dict(seal.get("runtime_tree") or {})
    tree_scan_error: str | None = None
    try:
        actual_tree = source_tree_evidence(root)
    except (OSError, ValueError) as exc:
        actual_tree = {}
        tree_scan_error = str(exc)
    checks = {
        "official_source_url_verified_from_sealed_build_provenance": (
            seal.get("source_url") == OFFICIAL_OSCAR_SOURCE_URL
        ),
        "reviewed_source_commit_verified_from_sealed_build_provenance": (
            seal.get("source_commit") == OFFICIAL_OSCAR_SOURCE_COMMIT
        ),
        "runtime_tree_sha256_verified": (
            bool(actual_tree)
            and actual_tree == expected_tree
        ),
        "runtime_tree_contains_no_unsealed_python_bytecode": (
            bool(actual_tree) and tree_scan_error is None
        ),
        "reviewed_runtime_tree_digest_verified": (
            actual_tree.get("tree_sha256") == OFFICIAL_OSCAR_RUNTIME_TREE_SHA256
            and expected_tree.get("tree_sha256")
            == OFFICIAL_OSCAR_RUNTIME_TREE_SHA256
        ),
        "sealed_source_root_resolved": (
            str(root) == str(seal.get("runtime_source_root") or "")
            and str(root) == DEFAULT_RUNTIME_SOURCE_ROOT
        ),
        "seal_schema_verified": (
            seal.get("schema_version") == SEAL_SCHEMA_VERSION
            and seal.get("status") == "sealed"
            and seal.get("git_metadata_required_at_runtime") is False
        ),
    }
    if not all(checks.values()):
        blockers.append("official_oscar_runtime_provenance_mismatch")
    passed = not blockers
    payload = {
        "schema_version": RUNTIME_ARTIFACT_SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "checks": checks,
        "expected_source_root": DEFAULT_RUNTIME_SOURCE_ROOT,
        "expected_source_url": OFFICIAL_OSCAR_SOURCE_URL,
        "expected_source_commit": OFFICIAL_OSCAR_SOURCE_COMMIT,
        "resolved_source_root": str(root) if checks["sealed_source_root_resolved"] else None,
        "resolved_source_url": (
            OFFICIAL_OSCAR_SOURCE_URL
            if checks["official_source_url_verified_from_sealed_build_provenance"]
            else None
        ),
        "resolved_source_commit": (
            OFFICIAL_OSCAR_SOURCE_COMMIT
            if checks["reviewed_source_commit_verified_from_sealed_build_provenance"]
            else None
        ),
        "runtime_tree": actual_tree or None,
        "sealed_runtime_tree": expected_tree or None,
        "blockers": blockers,
        "claim_boundary": {
            "build_time_git_identity_bound_to_runtime_tree": passed,
            "git_executable_or_metadata_required_at_runtime": False,
            "provenance_is_not_model_execution": True,
        },
        "raw_secret_values_recorded": False,
    }
    target = Path(artifact_path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("oscar_runtime_provenance_artifact_path_unsafe")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--source-root", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--source-url", default=OFFICIAL_OSCAR_SOURCE_URL)
    seal.add_argument("--source-commit", default=OFFICIAL_OSCAR_SOURCE_COMMIT)
    seal.add_argument("--runtime-source-root", default=DEFAULT_RUNTIME_SOURCE_ROOT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--source-root", default=DEFAULT_RUNTIME_SOURCE_ROOT)
    verify.add_argument("--seal", default=DEFAULT_SEAL_PATH)
    verify.add_argument("--artifact", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal":
        seal_source_tree(
            source_root=args.source_root,
            output_path=args.output,
            source_url=args.source_url,
            source_commit=args.source_commit,
            runtime_source_root=args.runtime_source_root,
        )
        return 0
    result = verify_source_tree(
        source_root=args.source_root,
        seal_path=args.seal,
        artifact_path=args.artifact,
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
