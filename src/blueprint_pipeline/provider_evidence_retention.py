"""Reclaim derived provider-bundle bytes without touching paid evidence.

A single paid run leaves roughly 210 MB behind, and only about 40 MB of that
is the result. The rest is the provider bundle, stored twice: the zip that was
uploaded, and the same zip unpacked beside it. Across one campaign that reached
27 GB, of which 21 GB was bundles and 10 GB was pure duplication - enough to
fill the disk and fail the next run before it started.

Three tiers, in increasing cost:

``unpacked_duplicate``
    A ``provider_runtime`` tree whose every file is also in the zip next to it.
    Deleting it loses nothing at all, and the check is per-file rather than by
    size, so a tree that has drifted from its zip is never touched.

``superseded_bundle_zip``
    The zip itself, for runs older than the newest few. It is derived from a
    pinned commit and its digest is recorded in the run receipt, so the claim
    chain survives - what is lost is the ability to re-inspect the exact bytes,
    which is why it is a separate, later tier.

Attempt evidence, receipts, and results are never eligible. Nothing is deleted
without an explicit apply and acknowledgement.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


RETENTION_SCHEMA_VERSION = "provider_evidence_retention.v1"
EXECUTE_ACK = "reap-derived-bundles"
PROTECTED_DIRECTORY_NAMES = ("attempts", "immutable_execution", "runtime_output")
DEFAULT_KEEP_NEWEST = 3


@dataclass
class Candidate:
    """One reclaimable path, with the reason it is safe to remove."""

    path: Path
    tier: str
    bytes_reclaimed: int
    reason: str
    run_directory: Path
    deleted: bool = False


@dataclass
class RetentionPlan:
    candidates: list[Candidate] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)

    @property
    def bytes_reclaimed(self) -> int:
        return sum(row.bytes_reclaimed for row in self.candidates)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_unpacked_duplicate(unpacked: Path, archive: Path) -> bool:
    """True only when every file under ``unpacked`` also exists in ``archive``."""

    try:
        with zipfile.ZipFile(archive) as handle:
            names = {name for name in handle.namelist() if not name.endswith("/")}
    except (OSError, zipfile.BadZipFile):
        return False
    if not names:
        return False
    root = unpacked.parent
    on_disk = {
        str(item.relative_to(root)) for item in unpacked.rglob("*") if item.is_file()
    }
    return bool(on_disk) and not (on_disk - names)


def plan_provider_evidence_retention(
    *,
    evidence_root: str | Path,
    keep_newest: int = DEFAULT_KEEP_NEWEST,
    include_superseded_zips: bool = False,
) -> RetentionPlan:
    """Find derived bundle bytes that can be reclaimed, newest runs excluded."""

    root = Path(evidence_root).expanduser().resolve()
    plan = RetentionPlan()
    if not root.is_dir():
        return plan

    runs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    keep = {path for path in runs[: max(int(keep_newest), 0)]}
    for run in runs:
        if run in keep:
            plan.protected.append(str(run))
            continue
        bundle = run / "bundle"
        if not bundle.is_dir():
            continue
        archives = sorted(bundle.glob("*.zip"))
        unpacked = bundle / "provider_runtime"
        if unpacked.is_dir() and archives:
            if _is_unpacked_duplicate(unpacked, archives[0]):
                plan.candidates.append(
                    Candidate(
                        path=unpacked,
                        tier="unpacked_duplicate",
                        bytes_reclaimed=_directory_bytes(unpacked),
                        reason=f"every file also present in {archives[0].name}",
                        run_directory=run,
                    )
                )
            else:
                plan.protected.append(str(unpacked))
        if include_superseded_zips:
            for archive in archives:
                plan.candidates.append(
                    Candidate(
                        path=archive,
                        tier="superseded_bundle_zip",
                        bytes_reclaimed=archive.stat().st_size,
                        reason="rebuildable from the pinned commit; digest recorded in the run receipt",
                        run_directory=run,
                    )
                )
    return plan


def apply_provider_evidence_retention(
    plan: RetentionPlan, *, apply: bool, ack: str | None
) -> dict[str, Any]:
    """Delete planned candidates, but only with an explicit acknowledgement."""

    import shutil

    authorised = bool(apply) and ack == EXECUTE_ACK
    for candidate in plan.candidates:
        if not authorised:
            continue
        # Never remove a path that carries run evidence, whatever the plan says.
        if any(part in PROTECTED_DIRECTORY_NAMES for part in candidate.path.parts):
            continue
        if candidate.path.is_dir():
            shutil.rmtree(candidate.path, ignore_errors=True)
        else:
            candidate.path.unlink(missing_ok=True)
        candidate.deleted = True
    return {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "status": "applied" if authorised else "dry_run",
        "applied": authorised,
        "candidate_count": len(plan.candidates),
        "deleted_count": sum(1 for row in plan.candidates if row.deleted),
        "bytes_reclaimed": plan.bytes_reclaimed
        if authorised
        else 0,
        "bytes_reclaimable": plan.bytes_reclaimed,
        "protected_paths": sorted(plan.protected),
        "candidates": [
            {
                "path": str(row.path),
                "tier": row.tier,
                "bytes": row.bytes_reclaimed,
                "reason": row.reason,
                "deleted": row.deleted,
            }
            for row in plan.candidates
        ],
        "claim_boundary": {
            "attempt_evidence_never_eligible": True,
            "receipts_never_eligible": True,
            "superseded_zip_digest_remains_in_receipt": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--keep-newest", type=int, default=DEFAULT_KEEP_NEWEST)
    parser.add_argument("--include-superseded-zips", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default=None, help=f"Must equal '{EXECUTE_ACK}'.")
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    plan = plan_provider_evidence_retention(
        evidence_root=arguments.evidence_root,
        keep_newest=arguments.keep_newest,
        include_superseded_zips=arguments.include_superseded_zips,
    )
    result = apply_provider_evidence_retention(
        plan, apply=arguments.apply, ack=arguments.ack
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "Candidate",
    "DEFAULT_KEEP_NEWEST",
    "EXECUTE_ACK",
    "RETENTION_SCHEMA_VERSION",
    "RetentionPlan",
    "apply_provider_evidence_retention",
    "plan_provider_evidence_retention",
]


if __name__ == "__main__":
    raise SystemExit(main())
