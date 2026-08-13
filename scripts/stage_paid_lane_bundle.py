#!/usr/bin/env python3
"""Stage a built paid-lane bundle onto a control-plane host.

A bundle is built where its private source bytes are -- a workstation with a
120 MB source PLY or a CAD agent's sealed inputs on it -- and run from the
control plane, which has neither those bytes nor any way to obtain them.
Something has to carry the archive across, and on 2026-08-12 that something was
a person with `scp`, which is how the live receipt ended up naming
`/Users/<author>/...` and how an authoring directory tree ended up recreated on
the droplet.

Deliberately lane-neutral. Every paid lane has this same problem and the same
shape of answer -- a receipt beside its archive -- so a per-lane copy of this
would be a per-lane opportunity to omit a digest check.

The transfer is not the problem; an unrecorded transfer is. This stages the
exact files the receipt references, verifies every digest on both ends, refuses
any destination outside a control-plane root, and leaves a staging receipt that
says what was placed, from which commit, and with which digests.

What it stages is deliberately a subset. The receipt, the archive, and the small
documents the receipt references by relative path are what the control plane
opens; bulk inputs are already inside the archive and are not copied a second
time.

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

SCHEMA_VERSION = "paid_lane_bundle_staging_receipt.v1"
DEFAULT_REMOTE_ROOT = "/var/lib/blueprint/task-evaluation-inputs"
#: The account the control-plane units run as. Staged bytes are useless to
#: the pipeline unless this user can read them.
DEFAULT_OWNER = "blueprint"


class StagingError(ValueError):
    """The job cannot be staged as a host-resident, digest-verified input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class LocalTransport:
    """Hand already-present bytes to the consuming account, on this host.

    Not every paid-lane input arrives over the wire. A config preflight has to
    be produced *on* the control plane, because it binds the deployed commit,
    and it needs Docker, so it runs as an account the service is not. What it
    leaves behind is then owned by whoever ran it and mode 0640 under the
    default umask -- correct-looking bytes the pipeline cannot open.

    That is the same defect the ssh transport already had (#485), so it gets the
    same answer rather than a second one: hand the tree over, then read every
    digest back *as the consumer*.
    """

    def _run(self, argv: list[str]) -> str:
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise StagingError(f"staging_local_command_failed:{result.returncode}")
        return result.stdout

    def mkdir(self, local_dir: str) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    def put(self, local: Path, remote: str) -> None:
        destination = Path(remote)
        if destination.resolve() == local.resolve():
            return
        destination.write_bytes(local.read_bytes())

    def finalize(self, local_dir: str, owner: str) -> None:
        self._run(["chown", "-R", f"{owner}:{owner}", local_dir])
        self._run(["chmod", "-R", "u=rwX,g=rX,o=", local_dir])

    def digest(self, remote: str, *, as_user: str | None = None) -> str:
        argv = ["sha256sum", remote]
        if as_user:
            argv = ["sudo", "-n", "-u", as_user, *argv]
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            return ""
        return "sha256:" + result.stdout.split()[0]


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

    def finalize(self, remote_dir: str, owner: str) -> None:
        """Hand the staged tree to the account that will consume it.

        `scp` preserves the source mode, so a receipt that happened to be 0600
        on the authoring machine lands 0600 and root-owned. Everything then
        looks correct -- the bytes are there and their digests match -- while
        the service account cannot open a single one of them.
        """

        quoted = shlex.quote(remote_dir)
        user = shlex.quote(owner)
        self._ssh(
            f"chown -R {user}:{user} {quoted} && chmod -R u=rwX,g=rX,o= {quoted}"
        )

    def digest(self, remote: str, *, as_user: str | None = None) -> str:
        command = f"sha256sum {shlex.quote(remote)} 2>/dev/null || true"
        if as_user:
            command = f"sudo -u {shlex.quote(as_user)} " + command
        output = self._ssh(command).strip()
        if not output:
            return ""
        return "sha256:" + output.split()[0]


def _referenced_relative_paths(
    receipt: Mapping[str, Any], *, receipt_name: str, job: Path
) -> list[str]:
    """The receipt, the archive, and every document the receipt resolves by
    relative path. Anything else is already sealed inside the archive.

    A receipt that predates portable references still stages: the archive is
    taken from the basename of its recorded path, which is where the resolver
    looks first anyway. Refusing those would strand every bundle built before
    the receipt format carried relative paths, for no gain in safety -- the
    digest is checked either way.
    """

    relatives = [receipt_name]
    bundle_relative = str(receipt.get("bundle_relative_path") or "").strip()
    if not bundle_relative:
        recorded = str(receipt.get("bundle_path") or "").strip()
        bundle_relative = Path(recorded).name if recorded else ""
    if not bundle_relative:
        raise StagingError("staging_receipt_names_no_bundle")
    relatives.append(bundle_relative)
    for name in ("execution_authority", "request"):
        record = receipt.get(name)
        if not isinstance(record, Mapping):
            continue
        relative = str(record.get("relative_path") or "").strip()
        if relative:
            relatives.append(relative)
    # Anything else sitting beside the receipt that the lane wrote as its own
    # evidence: rehearsal receipts, preflights. Small, and absent exactly when
    # a reader most wants them.
    for sibling in sorted(job.glob("*.json")):
        if sibling.name not in relatives and sibling.is_file():
            relatives.append(sibling.name)
    ordered: list[str] = []
    for relative in relatives:
        if relative not in ordered:
            ordered.append(relative)
    return ordered


def stage_paid_lane_bundle(
    *,
    receipt_path: str | Path,
    lane_id: str,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    transport: Any | None = None,
    host: str | None = None,
    extra_paths: Sequence[str | Path] = (),
    owner: str = DEFAULT_OWNER,
) -> dict[str, Any]:
    """Copy the receipt-referenced files to the host and verify them there."""

    receipt_file = Path(receipt_path).expanduser().resolve()
    if not receipt_file.is_file():
        raise StagingError("staging_job_receipt_missing")
    job = receipt_file.parent
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
        resolution = resolve_host_resident_bundle_receipt(receipt_file, roots=[job])
    except HostResidentInputError as exc:
        raise StagingError(str(exc)) from exc
    if resolution["blockers"]:
        raise StagingError(
            "staging_receipt_not_self_resolving:" + ",".join(resolution["blockers"])
        )

    # Use the resolver's launch-contract view.  Lane-native receipts (currently
    # ArtiFixer3D) keep their scientific status and bundle record nested in the
    # retained bytes; the view exposes their portable bundle reference without
    # rewriting the evidence the allocator will later validate.
    receipt = resolution["receipt"]
    relatives = _referenced_relative_paths(
        receipt, receipt_name=receipt_file.name, job=job
    )
    # Receipts a lane binds but keeps outside the bundle directory -- config
    # preflights are the case here, and the allocator refuses to run without
    # one. They are staged flat under their own name so the destination layout
    # stays a directory of files rather than a copy of somebody's tree.
    staged_extras: dict[str, Path] = {}
    for item in extra_paths:
        extra = Path(item).expanduser().resolve()
        if extra.is_symlink() or not extra.is_file():
            raise StagingError(f"staging_extra_file_missing:{extra.name}")
        if extra.name in relatives or extra.name in staged_extras:
            raise StagingError(f"staging_extra_file_name_collides:{extra.name}")
        staged_extras[extra.name] = extra
    link = transport if transport is not None else SshTransport(str(host or ""))
    if transport is None and not host:
        raise StagingError("staging_host_required")

    remote_dir = f"{remote_root.rstrip('/')}/{lane_id}"
    link.mkdir(remote_dir)

    staged: list[dict[str, Any]] = []
    for relative in [*relatives, *sorted(staged_extras)]:
        local = staged_extras.get(relative) or (job / relative)
        if local.is_symlink() or not local.is_file():
            raise StagingError(f"staging_local_file_missing:{relative}")
        remote = f"{remote_dir}/{relative}"
        parent = str(Path(remote).parent)
        if parent != remote_dir:
            link.mkdir(parent)
        local_digest = _sha256(local)
        link.put(local, remote)
        staged.append(
            {
                "relative_path": relative,
                "remote_path": remote,
                "sha256": local_digest,
                "size_bytes": local.stat().st_size,
            }
        )

    # Hand the tree over first, then read every digest back as the account that
    # will consume it. Verifying as the transfer user proves the bytes arrived
    # and nothing about whether the pipeline can open them.
    if hasattr(link, "finalize"):
        link.finalize(remote_dir, owner)
    for row in staged:
        observed = link.digest(row["remote_path"], as_user=owner)
        if not observed:
            # Distinct from a mismatch: the bytes may be perfect and simply
            # unopenable by the account that has to use them.
            raise StagingError(f"staging_consumer_cannot_read:{row['relative_path']}")
        if observed != row["sha256"]:
            raise StagingError(f"staging_remote_digest_mismatch:{row['relative_path']}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "staged",
        "lane_id": lane_id,
        "remote_dir": remote_dir,
        "receipt_name": receipt_file.name,
        "blueprint_commit": receipt.get("blueprint_commit")
        or receipt.get("implementation_commit"),
        "bundle_sha256": receipt.get("bundle_sha256"),
        "receipt_sha256": resolution["receipt_sha256"],
        "staged_files": [
            {k: v for k, v in row.items() if k != "remote_path"} for row in staged
        ],
        "verified_readable_as": owner,
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
    parser.add_argument(
        "--receipt",
        required=True,
        help="The lane's bundle receipt. Its directory is what gets staged.",
    )
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--host", required=True, help="ssh destination, e.g. root@<host>")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        help="Account the control plane runs as; staged bytes are verified as it.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help=(
            "A receipt the lane binds but keeps outside the bundle directory, "
            "such as a config preflight. Repeat for each; staged flat by name."
        ),
    )
    parser.add_argument("--receipt-out", help="Write the staging receipt here as well.")
    args = parser.parse_args(argv)

    try:
        receipt = stage_paid_lane_bundle(
            receipt_path=args.receipt,
            lane_id=args.lane_id,
            remote_root=args.remote_root,
            host=args.host,
            extra_paths=args.extra,
            owner=args.owner,
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
