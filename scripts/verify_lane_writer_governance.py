#!/usr/bin/env python3
"""Check reviewed lane-writer bytes and explicitly bounded discovery patterns.

This is a source-review gate. It cannot enforce arbitrary same-UID filesystem
writes, resolve dynamic CLI/env paths, or prove lease provenance for plain paths.
See docs/architecture/lane-scratch-writer-governance.md for syntax coverage.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import stat
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

SCHEMA = "lane_scratch_writer_manifest.v1"
REQUIRED_SOURCES = (
    "src/blueprint_pipeline/control_plane_lane_scratch.py",
    "src/blueprint_pipeline/control_plane_leased_scratch.py",
    "src/blueprint_pipeline/control_plane_arena_scratch.py",
    "src/blueprint_pipeline/native_g1_development_pair.py",
    "scripts/arena_construction_launch_chain.sh",
    "scripts/install_live_pipeline_control_plane.sh",
)
LANE_ROOTS = ("/mnt/blueprint-work/lanes", "/var/lib/blueprint/task-evaluation-inputs/lanes")
CONSTRUCTORS = {"create_lane_scratch", "create_leased_lane_scratch"}
DIRECTORY_CALLS = {"mkdir", "makedirs", "mkdtemp", "copytree"}
ROOT_VARIABLES = ("WORK_VOLUME_ROOT", "TASK_EVALUATION_INPUT_ROOT")
MAX_FILE_BYTES = 2 * 1024 * 1024


def _read_file(root: Path, relative: object) -> bytes:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
            or PurePosixPath(relative).as_posix() != relative):
        raise ValueError("path_unsafe")
    parts = PurePosixPath(relative).parts
    if not parts:
        raise ValueError("path_unsafe")
    with ExitStack() as fds:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.callback(os.close, fd)
        for component in parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            fds.callback(os.close, fd)
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            stream = os.fdopen(leaf, "rb")
        except BaseException:
            os.close(leaf)
            raise
        with stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                raise ValueError("path_unsafe")
            payload = stream.read(MAX_FILE_BYTES + 1)
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError("path_unsafe")
        return payload


def _python_writer(source: str) -> bool:
    # Avoid parsing the rest of the repository for this always-on sentinel.
    if not any(token in source for token in ("lanes", *CONSTRUCTORS)):
        return False
    tree = ast.parse(source)
    assignments: dict[str, ast.expr] = {}
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module or ''}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            assignments[node.target.id] = node.value

    def symbol(node: ast.expr, seen: frozenset[str] = frozenset()) -> str:
        if isinstance(node, ast.Name):
            if node.id in assignments and node.id not in seen:
                return symbol(assignments[node.id], seen | {node.id})
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{symbol(node.value, seen)}.{node.attr}"
        return ""

    def text(node: ast.expr, seen: frozenset[str] = frozenset()) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return text(assignments[node.id], seen | {node.id})
        if isinstance(node, ast.Call) and symbol(node.func).split('.')[-1] == "Path" and node.args:
            return text(node.args[0], seen)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            left, right = text(node.left, seen), text(node.right, seen)
            if left is not None and right is not None:
                return left.rstrip('/') + '/' + right.lstrip('/') if isinstance(node.op, ast.Div) else left + right
        if isinstance(node, ast.JoinedStr):
            pieces = [text(piece.value if isinstance(piece, ast.FormattedValue) else piece, seen)
                      for piece in node.values]
            if all(piece is not None for piece in pieces):
                return ''.join(pieces)
        return None

    calls = {symbol(node.func).split('.')[-1] for node in ast.walk(tree) if isinstance(node, ast.Call)}
    if calls & CONSTRUCTORS:
        return True
    if not calls & DIRECTORY_CALLS:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.expr):
            value = text(node)
            if value is not None and any(value == root or value.startswith(root + '/') for root in LANE_ROOTS):
                return True
    return False


def _shell_writer(source: str) -> bool:
    words = shlex.split(source, comments=True)
    if any(words[index:index + 2] == ["blueprint_pipeline.control_plane_arena_scratch", "prepare"]
           for index in range(len(words) - 1)):
        return True
    mutation = "mkdir" in words or "install" in words and "-d" in words
    references = (*LANE_ROOTS, *(f"${{{name}}}/lanes" for name in ROOT_VARIABLES),
                  *(f"${name}/lanes" for name in ROOT_VARIABLES))
    return mutation and any(reference in word for reference in references for word in words)


def validate_lane_writer_manifest(
    *, root: Path, manifest: Mapping[str, Any], required_sources: tuple[str, ...] = REQUIRED_SOURCES,
) -> list[str]:
    blockers: list[str] = []
    if manifest.get("schema_version") != SCHEMA:
        blockers.append("lane_writer_manifest_schema_invalid")
    entries = manifest.get("sources")
    if not isinstance(entries, list):
        return blockers + ["lane_writer_manifest_sources_invalid"]
    registered: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            blockers.append(f"lane_writer_entry_invalid:{index}")
            continue
        relative = entry.get("path")
        if isinstance(relative, str):
            if relative in registered:
                blockers.append(f"lane_writer_duplicate:{relative}")
            registered.add(relative)
        try:
            payload = _read_file(root, relative)
        except (ValueError, OSError):
            blockers.append(f"lane_writer_source_path_unsafe:{index}")
        else:
            if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                blockers.append(f"lane_writer_digest_changed:{relative}")
        if (entry.get("responsibility") not in {"constructor", "payload_consumer", "parent_provisioning"}
                or entry.get("output_class") not in {"lane_scratch", "container"}
                or not isinstance(entry.get("admission_method"), str) or not entry["admission_method"].strip()
                or not isinstance(entry.get("review_reference"), str) or not entry["review_reference"].strip()
                or not isinstance(entry.get("historical_exceptions"), list)
                or any(not isinstance(item, str) for item in entry.get("historical_exceptions", []))):
            blockers.append(f"lane_writer_contract_invalid:{index}")
        tests = entry.get("characterization_tests")
        if not isinstance(tests, list) or not tests:
            blockers.append(f"lane_writer_characterization_missing:{index}")
        else:
            for test in tests:
                try:
                    _read_file(root, test)
                except (ValueError, OSError):
                    blockers.append(f"lane_writer_test_path_unsafe:{index}")
    for missing in sorted(set(required_sources) - registered):
        blockers.append(f"lane_writer_required_missing:{missing}")
    for directory in ("src/blueprint_pipeline", "scripts", "deploy"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in {".py", ".sh"}:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                source = _read_file(root, relative).decode("utf-8")
                discovered = _python_writer(source) if path.suffix == ".py" else _shell_writer(source)
            except (OSError, ValueError, UnicodeError, SyntaxError):
                blockers.append(f"lane_writer_discovery_unreadable:{relative}")
                continue
            if discovered and relative not in registered:
                blockers.append(f"lane_writer_unregistered:{relative}")
    return sorted(set(blockers))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        manifest = json.loads(_read_file(root, "docs/architecture/lane-scratch-writer-manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
    except (OSError, ValueError):
        print(json.dumps({"blockers": ["lane_writer_manifest_unreadable"]}))
        return 1
    blockers = validate_lane_writer_manifest(root=root, manifest=manifest)
    print(json.dumps({"blockers": blockers}, sort_keys=True))
    return bool(blockers)


if __name__ == "__main__":
    raise SystemExit(main())
