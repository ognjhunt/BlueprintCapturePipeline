#!/usr/bin/env python3
"""Refresh worktree artifact digests in the fail-closed SC3 quality ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def rebind(ledger: dict[str, Any]) -> int:
    changed = 0
    for gap in ledger.get("gaps", []):
        for criterion in gap.get("criteria", []):
            for artifact in criterion.get("evidence_artifacts", []):
                relative = Path(str(artifact.get("path") or ""))
                candidate = (ROOT / relative).resolve()
                try:
                    candidate.relative_to(ROOT)
                except ValueError as exc:
                    raise ValueError(f"artifact_path_outside_repository:{relative}") from exc
                if not candidate.is_file():
                    raise FileNotFoundError(f"artifact_missing:{relative}")
                digest = _sha256(candidate)
                if artifact.get("sha256") != digest:
                    artifact["sha256"] = digest
                    changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    ledger_path = args.ledger.expanduser().resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    changed = rebind(ledger)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ledger": str(ledger_path), "digests_rebound": changed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
