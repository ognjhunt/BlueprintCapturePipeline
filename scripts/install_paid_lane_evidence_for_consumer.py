#!/usr/bin/env python3
"""Hand a locally produced paid-lane evidence directory to the service account.

Some paid-lane inputs cannot be staged from a workstation. A config preflight
binds `orchestrator_source_identity.commit`, so it is only meaningful when it
runs on the control plane at the deployed commit -- and it drives Docker, so it
runs as an account with the docker socket, which the service account
deliberately is not.

What that leaves behind is a directory of correct bytes the pipeline cannot
open: owner root, mode 0640 under the default umask. Every digest checks out,
the receipt reads `passed`, and the lane fails at the paid boundary on a file
that is sitting right there.

`stage_paid_lane_bundle.py` already answers this for bytes that arrive over the
wire (#485): hand the tree over, then read every digest back *as the consumer*.
This is that same check for bytes that were produced in place, using the same
transport object rather than a second implementation of it -- a per-situation
copy is a per-situation opportunity to skip the read-back.

Reads and re-permissions retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.host_resident_launch_inputs import (
    PRODUCTION_LAUNCH_INPUT_ROOTS,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_paid_lane_bundle import (  # noqa: E402
    DEFAULT_OWNER,
    LocalTransport,
    StagingError,
)

SCHEMA_VERSION = "paid_lane_evidence_install_receipt.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def install_paid_lane_evidence_for_consumer(
    *,
    evidence_dir: str | Path,
    owner: str = DEFAULT_OWNER,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Give the evidence directory to `owner`, then prove `owner` can read it."""

    directory = Path(evidence_dir).expanduser().resolve()
    if not directory.is_dir():
        raise StagingError("evidence_dir_missing")
    if not any(
        str(directory) == root or str(directory).startswith(root.rstrip("/") + "/")
        for root in PRODUCTION_LAUNCH_INPUT_ROOTS
    ):
        # Installing outside a control-plane root only defers the residency
        # refusal to the paid boundary.
        raise StagingError(f"evidence_dir_outside_control_plane:{directory}")

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise StagingError("evidence_dir_empty")

    # Digest before handing over: reading as the consumer afterwards then proves
    # both that the bytes are unchanged and that the account can open them.
    expected = {path: _sha256(path) for path in files}

    link = transport if transport is not None else LocalTransport()
    link.finalize(str(directory), owner)

    installed: list[dict[str, Any]] = []
    for path in files:
        observed = link.digest(str(path), as_user=owner)
        relative = str(path.relative_to(directory))
        if not observed:
            # Distinct from a mismatch: the bytes may be perfect and simply
            # unopenable by the account that has to use them.
            raise StagingError(f"evidence_consumer_cannot_read:{relative}")
        if observed != expected[path]:
            raise StagingError(f"evidence_install_digest_mismatch:{relative}")
        installed.append(
            {
                "relative_path": relative,
                "sha256": expected[path],
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "installed",
        "evidence_dir": str(directory),
        "installed_files": installed,
        "verified_readable_as": owner,
        "provider_mutation_performed": False,
        "claim_boundary": (
            "This receipt proves the named files are host-resident and readable "
            "by the consuming account with matching digests. It says nothing "
            "about what they assert, whether a launch profile references them, "
            "or whether any provider ran."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        help="The account the control-plane units run as.",
    )
    parser.add_argument("--receipt-out")
    args = parser.parse_args(argv)

    try:
        receipt = install_paid_lane_evidence_for_consumer(
            evidence_dir=args.evidence_dir, owner=args.owner
        )
    except (OSError, StagingError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    if args.receipt_out:
        out = Path(args.receipt_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
