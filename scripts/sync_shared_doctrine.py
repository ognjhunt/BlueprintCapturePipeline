#!/usr/bin/env python3
"""Single-source generator for cross-repo shared doctrine blocks.

Shared doctrine lives in exactly one editable place: the `doctrine/` directory
of the canonical repo, one plain Markdown fragment per block, with no wrappers
and no markers.  Every repo's `PLATFORM_CONTEXT.md`, `VISION.md`, and
`WORLD_MODEL_STRATEGY_CONTEXT.md` keeps its own repo-specific header and footer;
only the region between the shared markers is generated from the fragment.

That split is why a git submodule does not solve this on its own: the shared
content is a fragment spliced *inside* a larger repo-specific file, not a
standalone file a submodule could provide.

Workflow:

    # edit exactly one file
    $EDITOR doctrine/platform-context.md

    # write it into every repo this checkout can see, and re-lock
    python3 scripts/sync_shared_doctrine.py --write

    # verify without writing (CI in the canonical repo)
    python3 scripts/sync_shared_doctrine.py --check

`verify_shared_doctrine.py` is the enforcement half and runs in every repo: it
compares committed content against the lock without needing this script, a
sibling checkout, or a network.  This script is the propagation half.

Sibling repos are located by the sibling-checkout convention in `AGENTS.md`.
A repo that is not present is reported and skipped, never guessed around.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:  # imported as `scripts.sync_shared_doctrine` (tests, pythonpath = ["."])
    from scripts.verify_shared_doctrine import (
        LOCK_RELATIVE_PATH,
        LOCK_SCHEMA_VERSION,
        STATUS_LOCKED,
        DoctrineVerificationError,
        digest_block,
        extract_block,
        normalize_newlines,
    )
except ImportError:  # run directly as `python3 scripts/sync_shared_doctrine.py`
    from verify_shared_doctrine import (  # type: ignore[import-not-found, no-redef]
        LOCK_RELATIVE_PATH,
        LOCK_SCHEMA_VERSION,
        STATUS_LOCKED,
        DoctrineVerificationError,
        digest_block,
        extract_block,
        normalize_newlines,
    )


CANONICAL_REPO = "BlueprintCapturePipeline"
DOCTRINE_DIRECTORY = "doctrine"

# block name -> (fragment filename, target file relative to each repo root)
BLOCK_SOURCES = {
    "SHARED_PLATFORM_CONTEXT": ("platform-context.md", "PLATFORM_CONTEXT.md"),
    "SHARED_VISION": ("vision.md", "VISION.md"),
    "SHARED_WORLD_MODEL_STRATEGY": ("world-model-strategy.md", "WORLD_MODEL_STRATEGY_CONTEXT.md"),
}


class DoctrineSyncError(RuntimeError):
    """Raised when a fragment or target cannot be read, spliced, or written."""


def read_fragment(root: Path, block: str) -> str:
    """Return a canonical fragment normalized to exactly one trailing newline."""

    filename, _ = BLOCK_SOURCES[block]
    path = root / DOCTRINE_DIRECTORY / filename
    if not path.is_file():
        raise DoctrineSyncError(f"{block}: missing canonical fragment {path}")
    body = normalize_newlines(path.read_text(encoding="utf-8"))
    if not body.strip():
        raise DoctrineSyncError(f"{block}: canonical fragment {path} is empty")
    return body.rstrip("\n") + "\n"


def splice(text: str, block: str, body: str) -> str:
    """Replace the marked region of `text` with `body`, preserving the wrapper.

    Marker lines are preserved exactly as written, including any indentation,
    so a target file's surrounding structure is never rewritten.
    """

    start_marker = f"<!-- {block}_START -->"
    end_marker = f"<!-- {block}_END -->"
    lines = normalize_newlines(text).split("\n")

    start_hits = [i for i, line in enumerate(lines) if start_marker in line]
    end_hits = [i for i, line in enumerate(lines) if end_marker in line]
    if len(start_hits) != 1 or len(end_hits) != 1:
        raise DoctrineSyncError(
            f"{block}: expected exactly one start and one end marker, "
            f"found {len(start_hits)} and {len(end_hits)}"
        )
    if end_hits[0] <= start_hits[0]:
        raise DoctrineSyncError(f"{block}: end marker precedes start marker")

    replacement = body.rstrip("\n").split("\n")
    return "\n".join(lines[: start_hits[0] + 1] + replacement + lines[end_hits[0] :])


def discover_repos(root: Path, explicit: Iterable[str] = ()) -> dict[str, Path]:
    """Map repo name -> checkout path, for repos actually present on disk.

    The canonical repo is always included.  Siblings are looked for next to it
    per the sibling-checkout convention; absent ones are simply not returned.
    """

    found: dict[str, Path] = {CANONICAL_REPO: root}
    for raw in explicit:
        path = Path(raw).resolve()
        if not path.is_dir():
            raise DoctrineSyncError(f"explicit repo path is not a directory: {path}")
        found[path.name] = path

    lock = json.loads((root / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    for name in lock.get("repos") or []:
        if name in found:
            continue
        candidate = root.parent / name
        if candidate.is_dir():
            found[name] = candidate
    return found


def sync(root: Path, *, write: bool, explicit: Iterable[str] = ()) -> dict[str, Any]:
    """Splice canonical fragments into every discovered repo.

    Returns a report.  With `write=False` nothing is modified and the report
    records which targets would change, so CI can fail on drift.
    """

    lock_path = root / LOCK_RELATIVE_PATH
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise DoctrineSyncError(f"unsupported lock schema_version: {lock.get('schema_version')!r}")

    repos = discover_repos(root, explicit)
    missing = [name for name in (lock.get("repos") or []) if name not in repos]
    changed: list[str] = []
    rows: list[dict[str, Any]] = []

    for block, (_, target_relative) in sorted(BLOCK_SOURCES.items()):
        fragment = read_fragment(root, block)
        canonical_sha = digest_block(fragment)

        for repo_name, repo_root in sorted(repos.items()):
            target = repo_root / target_relative
            if not target.is_file():
                raise DoctrineSyncError(f"{repo_name}: missing target {target_relative}")
            original = target.read_text(encoding="utf-8")
            updated = splice(original, block, fragment)
            differs = updated != original
            if differs:
                changed.append(f"{repo_name}:{target_relative}")
                if write:
                    target.write_text(updated, encoding="utf-8")
            rows.append(
                {
                    "block": block,
                    "repo": repo_name,
                    "file": target_relative,
                    "changed": differs,
                    "sha256": canonical_sha,
                }
            )

        entry = lock["blocks"].setdefault(block, {"file": target_relative})
        entry["file"] = target_relative
        entry["canonical_sha256"] = canonical_sha
        entry.pop("observed_sha256", None)

    if write:
        lock["status"] = STATUS_LOCKED
        lock["canonical_source"] = f"{CANONICAL_REPO} {DOCTRINE_DIRECTORY}/"
        lock.pop("status_note", None)
        rendered = json.dumps(lock, indent=2, sort_keys=False) + "\n"
        for repo_root in repos.values():
            (repo_root / LOCK_RELATIVE_PATH).write_text(rendered, encoding="utf-8")

    return {
        "canonical_repo": CANONICAL_REPO,
        "repos_synced": sorted(repos),
        "repos_absent": missing,
        "changed_targets": sorted(set(changed)),
        "rows": rows,
        "wrote": write,
    }


def verify_roundtrip(root: Path) -> None:
    """Confirm each spliced target re-extracts to exactly its fragment.

    Guards the splice/extract pair against ever disagreeing: whatever this
    script writes must be what the enforcement verifier reads back.
    """

    for block in BLOCK_SOURCES:
        fragment = read_fragment(root, block)
        _, target_relative = BLOCK_SOURCES[block]
        target = root / target_relative
        spliced = splice(target.read_text(encoding="utf-8"), block, fragment)
        extracted = extract_block(spliced, block)
        if digest_block(extracted) != digest_block(fragment):
            raise DoctrineSyncError(f"{block}: splice/extract roundtrip disagrees")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo", action="append", default=[], help="explicit repo checkout path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write fragments into every repo")
    mode.add_argument("--check", action="store_true", help="fail if any target is out of date")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        verify_roundtrip(root)
        report = sync(root, write=args.write, explicit=args.repo)
    except (DoctrineSyncError, DoctrineVerificationError) as exc:
        print(f"shared doctrine sync FAILED: {exc}", file=sys.stderr)
        return 1

    for name in report["repos_absent"]:
        print(f"note: {name} is not checked out; skipped", file=sys.stderr)

    if args.check and report["changed_targets"]:
        print(
            "shared doctrine sync FAILED: targets are out of date with "
            f"{DOCTRINE_DIRECTORY}/: {', '.join(report['changed_targets'])}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
