#!/usr/bin/env python3
"""Fail-closed verifier for cross-repo shared doctrine blocks.

Shared doctrine blocks are required to be byte-identical across
`BlueprintCapture`, `BlueprintCapturePipeline`, and `Blueprint-WebApp`.  The
prior mechanism was a convention plus a sibling-checkout comparison, which
passes trivially in CI because siblings are not checked out there.  Two repos
diverged on 2026-07-29 without any gate firing.

This verifier compares local block content against digests committed in
`contracts/shared-doctrine.lock.json`.  It needs no sibling checkout, no
network, and no provider, so it runs identically on a laptop and in CI.

Extraction is defined exactly, so the Python and TypeScript implementations
agree byte for byte:

  * locate the single line containing `<!-- <BLOCK>_START -->`
  * locate the single line containing `<!-- <BLOCK>_END -->`
  * take the lines strictly between them, join with "\\n", append one "\\n"
  * hash the UTF-8 bytes of that string with SHA-256

Exit status is 0 only when every tracked block matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_NAME = "BlueprintCapturePipeline"
LOCK_RELATIVE_PATH = "contracts/shared-doctrine.lock.json"
LOCK_SCHEMA_VERSION = "blueprint.shared_doctrine_lock.v1"
STATUS_LOCKED = "locked"
STATUS_UNRECONCILED = "unreconciled"


class DoctrineVerificationError(RuntimeError):
    """Raised when a block cannot be extracted or does not match the lock."""


def extract_block(text: str, block: str) -> str:
    """Return the exact shared-block body between its markers.

    Raises when the markers are missing, duplicated, or out of order, so a
    malformed file fails closed rather than hashing an empty string.
    """

    start_marker = f"<!-- {block}_START -->"
    end_marker = f"<!-- {block}_END -->"
    lines = text.splitlines()

    start_hits = [i for i, line in enumerate(lines) if start_marker in line]
    end_hits = [i for i, line in enumerate(lines) if end_marker in line]

    if len(start_hits) != 1:
        raise DoctrineVerificationError(
            f"{block}: expected exactly one {start_marker}, found {len(start_hits)}"
        )
    if len(end_hits) != 1:
        raise DoctrineVerificationError(
            f"{block}: expected exactly one {end_marker}, found {len(end_hits)}"
        )
    if end_hits[0] <= start_hits[0]:
        raise DoctrineVerificationError(f"{block}: end marker precedes start marker")

    body = lines[start_hits[0] + 1 : end_hits[0]]
    return "\n".join(body) + "\n"


def digest_block(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_lock(root: Path) -> dict[str, Any]:
    lock_path = root / LOCK_RELATIVE_PATH
    if not lock_path.is_file():
        raise DoctrineVerificationError(f"missing lock file: {LOCK_RELATIVE_PATH}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise DoctrineVerificationError(
            f"unsupported lock schema_version: {lock.get('schema_version')!r}"
        )
    status = lock.get("status")
    if status not in {STATUS_LOCKED, STATUS_UNRECONCILED}:
        raise DoctrineVerificationError(f"unsupported lock status: {status!r}")
    return lock


def expected_digest(block_name: str, entry: dict[str, Any], status: str) -> str:
    """Resolve the digest this repo's copy must match.

    Under `locked`, every repo matches one canonical digest.  Under
    `unreconciled`, each repo matches the divergent baseline recorded for it, so
    no repo can drift further while reconciliation is pending.  A repo with no
    recorded baseline fails closed rather than being silently skipped.
    """

    if status == STATUS_LOCKED:
        canonical = entry.get("canonical_sha256")
        if not canonical:
            raise DoctrineVerificationError(
                f"{block_name}: lock status is {STATUS_LOCKED} but canonical_sha256 is absent"
            )
        return str(canonical)

    observed = entry.get("observed_sha256") or {}
    if REPO_NAME not in observed or not observed[REPO_NAME]:
        raise DoctrineVerificationError(
            f"{block_name}: no baseline recorded for {REPO_NAME}; "
            "measure this repo's block and add it to the lock before merging"
        )
    return str(observed[REPO_NAME])


def verify(root: Path) -> list[dict[str, Any]]:
    lock = load_lock(root)
    status = str(lock["status"])
    blocks = lock.get("blocks") or {}
    if not blocks:
        raise DoctrineVerificationError("lock declares no blocks")

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for block_name, entry in sorted(blocks.items()):
        relative_file = str(entry.get("file") or "")
        source = root / relative_file
        if not source.is_file():
            failures.append(f"{block_name}: missing source file {relative_file}")
            continue
        try:
            body = extract_block(source.read_text(encoding="utf-8"), block_name)
            actual = digest_block(body)
            wanted = expected_digest(block_name, entry, status)
        except DoctrineVerificationError as exc:
            failures.append(str(exc))
            continue

        matched = actual == wanted
        if not matched:
            failures.append(
                f"{block_name}: {relative_file} does not match the lock "
                f"(expected {wanted}, found {actual})"
            )
        results.append(
            {
                "block": block_name,
                "file": relative_file,
                "expected_sha256": wanted,
                "actual_sha256": actual,
                "matched": matched,
            }
        )

    if failures:
        raise DoctrineVerificationError("; ".join(failures))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root (defaults to this script's repository)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        results = verify(root)
    except DoctrineVerificationError as exc:
        print(f"shared doctrine verification FAILED: {exc}", file=sys.stderr)
        return 1

    lock = load_lock(root)
    status = str(lock["status"])
    if args.json:
        print(json.dumps({"repo": REPO_NAME, "status": status, "blocks": results}, indent=2))
    else:
        for row in results:
            print(f"ok  {row['block']}  {row['file']}  {row['actual_sha256']}")
        if status == STATUS_UNRECONCILED:
            print(
                "\nNOTE: lock status is 'unreconciled'. Blocks are frozen at their "
                "current per-repo baselines so no new variants can appear, but the "
                "repos do not yet agree. See "
                "docs/doctrine-shared-block-divergence-2026-07-31.md.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
