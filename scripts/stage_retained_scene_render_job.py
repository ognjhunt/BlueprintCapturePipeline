#!/usr/bin/env python3
"""Stage a built retained-scene render job onto a control-plane host.

The bundle is built where the private scene bytes are -- a workstation with a
120 MB source PLY on it -- and run from the control plane, which has neither the
scene bytes nor any way to obtain them. Something has to carry the archive
across, and on 2026-08-12 that something was a person with `scp`, which is how
the live receipt ended up naming `/Users/<author>/...` and how an authoring
directory tree ended up recreated on the droplet.

The transfer is not the problem; an unrecorded transfer is. This stages the
exact files the receipt references, verifies every digest on both ends, refuses
any destination outside a control-plane root, and leaves a staging receipt that
says what was placed, from which commit, and with which digests.

What it stages is deliberately a subset. The receipt, the archive, and the small
documents the receipt references by relative path are what the control plane
opens; the bulk PLY inputs are already inside the archive and are not copied a
second time.

Performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess  # nosec B404 - fixed ssh/scp argv over validated paths
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.host_resident_launch_inputs import (
    PRODUCTION_LAUNCH_INPUT_ROOTS,
    HostResidentInputError,
    resolve_host_resident_bundle_receipt,
)

SCHEMA_VERSION = "retained_scene_render_job_staging_receipt.v1"
RECEIPT_NAME = "adp_retained_scene_gpu_render_bundle_receipt.json"
REHEARSAL_NAME = "adp_retained_scene_gpu_render_exact_bundle_rehearsal.json"
DEFAULT_REMOTE_ROOT = "/var/lib/blueprint/task-evaluation-inputs"


class StagingError(ValueError):
    """The job cannot be staged as a host-resident, digest-verified input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class SshTransport:
    """Place bytes on a remote host and read back their digests."""

    def __init__(self, host: str) -> None:
        self.host = host

    def _ssh(self, command: str) -> str:
        result = subprocess.run(  # nosec B603 B607 - fixed argv, quoted command
            ["ssh", "-o", "BatchMode=yes", self.host, command],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise StagingError(f"staging_remote_command_failed:{result.returncode}")
        return result.stdout

    def mkdir(self, remote_dir: str) -> None:
        self._ssh(f"mkdir -p {shlex.quote(remote_dir)}")

    def put(self, local: Path, remote: str) -> None:
        result = subprocess.run(  # nosec B603 B607 - fixed argv
            ["scp", "-o", "BatchMode=yes", "-q", str(local), f"{self.host}:{remote}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise StagingError(f"staging_transfer_failed:{Path(remote).name}")

    def digest(self, remote: str) -> str:
        output = self._ssh(f"sha256sum {shlex.quote(remote)} 2>/dev/null || true").strip()
        if not output:
            return ""
        return "sha256:" + output.split()[0]


def _referenced_relative_paths(receipt: Mapping[str, Any]) -> list[str]:
    """The receipt, the archive, and every document the receipt resolves by
    relative path. Anything else is already sealed inside the archive."""

    relatives = [RECEIPT_NAME, REHEARSAL_NAME]
    bundle_relative = str(receipt.get("bundle_relative_path") or "").strip()
    if not bundle_relative:
        raise StagingError("staging_receipt_has_no_portable_bundle_reference")
    relatives.append(bundle_relative)
    for name in ("execution_authority", "request"):
        record = receipt.get(name)
        if not isinstance(record, Mapping):
            continue
        relative = str(record.get("relative_path") or "").strip()
        if not relative:
            raise StagingError(f"staging_receipt_reference_not_portable:{name}")
        relatives.append(relative)
    ordered: list[str] = []
    for relative in relatives:
        if relative not in ordered:
            ordered.append(relative)
    return ordered


def stage_retained_scene_render_job(
    *,
    job_dir: str | Path,
    lane_id: str,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    transport: Any | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Copy the receipt-referenced files to the host and verify them there."""

    job = Path(job_dir).expanduser().resolve()
    receipt_path = job / RECEIPT_NAME
    if not receipt_path.is_file():
        raise StagingError("staging_job_receipt_missing")
    if not lane_id or "/" in lane_id or lane_id.startswith("."):
        raise StagingError("staging_lane_id_invalid")
    if not any(
        remote_root == root or remote_root.startswith(root.rstrip("/") + "/")
        for root in PRODUCTION_LAUNCH_INPUT_ROOTS
    ):
        # Staging somewhere the residency gate will later refuse only defers the
        # failure to the paid boundary.
        raise StagingError(f"staging_remote_root_outside_control_plane:{remote_root}")

    # The receipt must already resolve against its own directory here, or it
    # cannot resolve against the staged one there either.
    try:
        resolution = resolve_host_resident_bundle_receipt(receipt_path, roots=[job])
    except HostResidentInputError as exc:
        raise StagingError(str(exc)) from exc
    if resolution["blockers"]:
        raise StagingError(
            "staging_receipt_not_self_resolving:" + ",".join(resolution["blockers"])
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    relatives = _referenced_relative_paths(receipt)
    link = transport if transport is not None else SshTransport(str(host or ""))
    if transport is None and not host:
        raise StagingError("staging_host_required")

    remote_dir = f"{remote_root.rstrip('/')}/{lane_id}"
    link.mkdir(remote_dir)

    staged: list[dict[str, Any]] = []
    for relative in relatives:
        local = job / relative
        if local.is_symlink() or not local.is_file():
            raise StagingError(f"staging_local_file_missing:{relative}")
        remote = f"{remote_dir}/{relative}"
        parent = str(Path(remote).parent)
        if parent != remote_dir:
            link.mkdir(parent)
        local_digest = _sha256(local)
        link.put(local, remote)
        remote_digest = link.digest(remote)
        if remote_digest != local_digest:
            raise StagingError(f"staging_remote_digest_mismatch:{relative}")
        staged.append(
            {
                "relative_path": relative,
                "sha256": local_digest,
                "size_bytes": local.stat().st_size,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "staged",
        "lane_id": lane_id,
        "remote_dir": remote_dir,
        "blueprint_commit": receipt.get("blueprint_commit"),
        "bundle_sha256": receipt.get("bundle_sha256"),
        "receipt_sha256": resolution["receipt_sha256"],
        "staged_files": staged,
        "provider_mutation_performed": False,
        "claim_boundary": (
            "This receipt proves the named files were placed on the host and "
            "read back with matching digests. It is not proof that a launch "
            "profile references them, that a provider ran, or that any render "
            "completed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--host", required=True, help="ssh destination, e.g. root@<host>")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--receipt-out", help="Write the staging receipt here as well.")
    args = parser.parse_args(argv)

    try:
        receipt = stage_retained_scene_render_job(
            job_dir=args.job_dir,
            lane_id=args.lane_id,
            remote_root=args.remote_root,
            host=args.host,
        )
    except (OSError, StagingError, json.JSONDecodeError) as exc:
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
