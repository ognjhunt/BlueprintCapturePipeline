"""Keep the WebApp launch-profile catalog equal to the published directory.

The catalog is the document the WebApp reads to resolve a ``profile_id``. It is
derived from the profile directory, and a derived artifact can drift from its
source.

PR #454 fixed the writer: the publisher now enumerates the directory instead of
using the profiles named on its command line, so publishing B no longer drops A.
That corrects every catalog written after it deploys and none of the ones
already on disk. Observed on the live control plane immediately after #454
deployed -- seven profiles in the directory, four in the served catalog, and the
three missing ones unreachable by any launch. The repair was a manual publish,
which is exactly the kind of remembered step that does not survive a host
rebuild.

Reconciling here, at start-up and from the directory, makes the drift
self-correcting: the directory is the published state, the catalog is a
projection of it, and a projection that disagrees with its source is rebuilt
rather than served. A directory that cannot be projected -- an invalid or
symlinked profile -- fails closed instead, because a catalog that silently omits
a published file is the defect this exists to prevent.

Reads and rewrites retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .host_resident_launch_inputs import launch_profile_residency_blockers
from .task_evaluation_launch_dispatcher import (
    PUBLIC_LAUNCH_PROFILE_CATALOG_MAX_BYTES,
    PUBLIC_LAUNCH_PROFILE_CATALOG_MAX_PROFILES,
    TaskEvaluationLaunchError,
    public_launch_profile_descriptor,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)

SCHEMA_VERSION = "task_evaluation_launch_catalog_reconciliation.v1"


class LaunchCatalogError(TaskEvaluationLaunchError):
    """The published directory cannot be projected into a catalog."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LaunchCatalogError(f"published_profile_invalid:{path.name}:not_an_object")
    return value


def build_catalog_payload(profile_dir: str | Path) -> bytes:
    """Project every published profile into the public catalog's bytes.

    Shared with the publisher so the artifact written at publish time and the
    one reconciled at start-up cannot disagree about what the catalog *is*.
    """
    root = Path(profile_dir).expanduser().resolve()
    if not root.is_dir():
        raise LaunchCatalogError(f"profile_dir_missing:{root}")

    descriptors: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.is_symlink():
            # Published profiles are immutable; a symlink lets the bytes behind
            # a published id change without the id changing.
            raise LaunchCatalogError(f"launch_profile_source_invalid:{path}")
        try:
            profile = _read(path)
        except json.JSONDecodeError as exc:
            raise LaunchCatalogError(
                f"published_profile_invalid:{path.name}:unparseable"
            ) from exc
        # A malformed profile cannot be projected at all and still fails the
        # whole catalog. Missing bytes are a different kind of problem: the
        # document is well formed and this host simply does not have what it
        # names. Raising on that meant one stale profile stopped every other
        # profile from being served -- and worse, it is guaranteed on a host
        # restored from an image, where none of the inputs a previous host
        # accumulated exist yet. The catalog reconciler is an ExecStartPre for
        # intake, so that is a total outage with no obvious cause.
        blockers = validate_launch_profile(profile)
        if blockers:
            raise LaunchCatalogError(
                f"published_profile_invalid:{path.name}:" + ",".join(sorted(set(blockers)))
            )
        descriptor = public_launch_profile_descriptor(profile)
        unavailable = verify_profile_immutable_inputs(profile)
        # A profile whose inputs are not on this host cannot start a run here,
        # so serving it as live is the lie. Demote it in the projection rather
        # than refusing the whole catalog: the published bytes stay immutable
        # evidence, and one stale profile must not make every other profile
        # unreachable -- the catalog reconciler is an ExecStartPre for intake,
        # so raising here would take the website path down with it.
        unrunnable = [*unavailable, *launch_profile_residency_blockers(profile)]
        if unrunnable:
            admission = dict(descriptor.get("execution_admission") or {})
            admission["live_enabled"] = False
            admission["blockers"] = sorted(
                {*(admission.get("blockers") or []), *unrunnable}
            )
            descriptor["execution_admission"] = admission
        descriptors.append(descriptor)

    if len(descriptors) > PUBLIC_LAUNCH_PROFILE_CATALOG_MAX_PROFILES:
        raise LaunchCatalogError("published_profile_catalog_profile_limit_exceeded")
    payload = (
        json.dumps(
            sorted(descriptors, key=lambda row: str(row["profile_id"])),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    if len(payload) > PUBLIC_LAUNCH_PROFILE_CATALOG_MAX_BYTES:
        raise LaunchCatalogError("published_profile_catalog_size_limit_exceeded")
    return payload


def _catalog_ids(payload: bytes) -> set[str]:
    try:
        rows = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # An unreadable catalog is not evidence of what is published; the
        # directory is. Report nothing rather than trusting the bytes.
        return set()
    if not isinstance(rows, list):
        return set()
    return {
        str(row["profile_id"])
        for row in rows
        if isinstance(row, dict) and "profile_id" in row
    }


def reconcile_public_catalog(
    *,
    profile_dir: str | Path,
    catalog_path: str | Path,
) -> dict[str, Any]:
    """Rebuild the catalog when it disagrees with the published directory."""
    catalog = Path(catalog_path).expanduser().resolve()
    expected = build_catalog_payload(profile_dir)

    try:
        current = catalog.read_bytes()
    except FileNotFoundError:
        current = b""
    except OSError as exc:
        raise LaunchCatalogError(f"catalog_unreadable:{catalog}") from exc

    if current == expected:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "consistent",
            "catalog_path": str(catalog),
            "profile_ids_added": [],
            "profile_ids_removed": [],
            "provider_mutation_performed": False,
        }

    before = _catalog_ids(current)
    after = _catalog_ids(expected)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    try:
        catalog.write_bytes(expected)
    except OSError as exc:
        raise LaunchCatalogError(f"catalog_unwritable:{catalog}") from exc

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "repaired",
        "catalog_path": str(catalog),
        "profile_ids_added": sorted(after - before),
        "profile_ids_removed": sorted(before - after),
        "provider_mutation_performed": False,
    }


__all__ = [
    "SCHEMA_VERSION",
    "LaunchCatalogError",
    "build_catalog_payload",
    "reconcile_public_catalog",
]
