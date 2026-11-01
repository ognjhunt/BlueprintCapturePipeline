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
import os
import stat
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from .host_resident_launch_inputs import (
    configured_launch_input_roots,
    launch_profile_residency_blockers,
)
from .task_evaluation_launch_dispatcher import (
    LAUNCH_PROFILE_CATALOG_SCHEMA_VERSION,
    PUBLIC_LAUNCH_PROFILE_CATALOG_MAX_BYTES,
    PUBLIC_LAUNCH_PROFILE_CATALOG_MAX_PROFILES,
    TaskEvaluationLaunchError,
    catalog_launch_profile_descriptor,
    validate_launch_profile,
    validate_public_launch_profile_descriptor,
    verify_profile_immutable_inputs,
)

SCHEMA_VERSION = "task_evaluation_launch_catalog_reconciliation.v1"


class LaunchCatalogError(TaskEvaluationLaunchError):
    """The published directory cannot be projected into a catalog."""


@lru_cache(maxsize=1)
def _validated_public_launch_profile_catalog(
    payload: bytes, max_profiles: int,
    validate_descriptor: Callable[[Mapping[str, Any]], list[str]],
) -> dict[str, Any]:
    """Reuse full validation only for identical bytes, limits and validator."""
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, list) or len(value) > max_profiles:
        raise TaskEvaluationLaunchError("launch_profile_public_catalog_invalid")
    profiles: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        descriptor = dict(item) if isinstance(item, Mapping) else {}
        blockers = validate_descriptor(descriptor)
        key = (str(descriptor.get("profile_id") or ""), str(descriptor.get("profile_digest") or ""))
        if blockers or key in seen:
            raise TaskEvaluationLaunchError("launch_profile_public_catalog_invalid")
        seen.add(key)
        profiles.append(descriptor)
    return {"schema_version": LAUNCH_PROFILE_CATALOG_SCHEMA_VERSION, "profiles": profiles}


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
    descriptors, _unjudged = _project_published_profiles(profile_dir)
    return _catalog_bytes(descriptors)


def _project_published_profiles(
    profile_dir: str | Path,
    timings: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Every published profile's descriptor, and the ones whose owner state this
    process could not judge: ``scene_execution_owner_store_missing`` means its
    environment names no scene-intake owner store, not that the owner is gone."""
    root = Path(profile_dir).expanduser().resolve()
    if not root.is_dir():
        raise LaunchCatalogError(f"profile_dir_missing:{root}")

    descriptors: list[dict[str, Any]] = []
    unjudged: set[tuple[str, str]] = set()
    # One look per input file and per bound path for the whole pass, and one
    # validation of each kind per profile. On 2026-10-12 the 539 published profiles
    # listed 26,015 inputs that are 3,784 distinct files; the pass checked every
    # reference and validated each profile two or three times, and it is the
    # ExecStartPre of every guarded unit.
    observed: dict[str, str | None] = {}
    resolved: dict[tuple[str, tuple[Path, ...]], bool] = {}
    roots = tuple(configured_launch_input_roots())
    # Where a pass spends its time, reported by the runtime env guard: the
    # catalog is reconciled in the ExecStartPre of units that start every minute.
    spent = {"profile_parse": 0.0, "immutable_inputs": 0.0, "profile_validation": 0.0, "input_residency": 0.0}

    def lap(name: str, started: float) -> float:
        now = time.monotonic()
        spent[name] += now - started
        return now

    for path in sorted(root.glob("*.json")):
        started = time.monotonic()
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
        try:
            # The structural pass, and the public fields every outcome below serves.
            descriptor = catalog_launch_profile_descriptor(profile)
        except TaskEvaluationLaunchError as exc:
            raise LaunchCatalogError(f"published_profile_invalid:{path.name}:{exc}") from exc
        started = lap("profile_parse", started)
        unavailable = verify_profile_immutable_inputs(profile, observed=observed)
        started = lap("immutable_inputs", started)
        inactive_authority: list[str] = []
        if not unavailable:
            # Once all declared bytes are present and digest-bound, retain the
            # existing fail-closed cross-document validation. An inconsistent
            # or tampered lineage is malformed evidence, not host unavailability.
            blockers = validate_launch_profile(profile)
            inactive_owner_states = {
                "scene_execution_owner_revoked",
                "scene_execution_owner_expired",
                "scene_execution_owner_attempt_cancelled_before_execution",
                # An owner record this host cannot read is host unavailability,
                # the same category as missing immutable inputs above, not
                # corrupted evidence. Raising made one stale profile abort the
                # whole catalog, and the reconciler is an ExecStartPre for
                # intake. Scene 840938, 2026-09-16: a single profile from a
                # retired source (7756dd48) failed this way for hours, the
                # catalog never reconciled, and launch activation starved --
                # so a healthy retargeted run sat at awaiting_execution with
                # nothing wrong with it.
                "scene_execution_owner_store_missing",
            }
            if blockers and not set(blockers).issubset(inactive_owner_states):
                raise LaunchCatalogError(
                    f"published_profile_invalid:{path.name}:"
                    + ",".join(sorted(set(blockers)))
                )
            # Revoking an owner or cancelling an unstarted attempt is a normal
            # lifecycle transition, not corrupted published evidence. Keep the
            # immutable profile in history and explicitly disable its catalog
            # projection.
            inactive_authority = blockers
            if "scene_execution_owner_store_missing" in blockers:
                unjudged.add((str(descriptor["profile_id"]), str(descriptor["profile_digest"])))
        # A profile whose inputs are not on this host cannot start a run here,
        # so serving it as live is the lie. Demote it in the projection rather
        # than refusing the whole catalog: the published bytes stay immutable
        # evidence, and one stale profile must not make every other profile
        # unreachable -- the catalog reconciler is an ExecStartPre for intake,
        # so raising here would take the website path down with it.
        started = lap("profile_validation", started)
        unrunnable = [
            *unavailable,
            *inactive_authority,
            *launch_profile_residency_blockers(profile, roots=roots, resolved=resolved),
        ]
        lap("input_residency", started)
        if unrunnable:
            admission = dict(descriptor.get("execution_admission") or {})
            admission["live_enabled"] = False
            admission["blockers"] = sorted(
                {*(admission.get("blockers") or []), *unrunnable}
            )
            descriptor["execution_admission"] = admission
        descriptors.append(descriptor)
    if timings is not None:
        timings.update({name: round(value, 3) for name, value in spent.items()})
        timings["profiles"] = len(descriptors)
        timings["distinct_inputs"] = len(observed)
    return descriptors, unjudged


def _catalog_bytes(descriptors: list[dict[str, Any]]) -> bytes:
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
    """Rebuild the catalog when it disagrees with the published directory.

    Every guarded unit reconciles, each from its own environment, and only some
    name the scene-intake owner store. On 2026-10-12 the served catalog flipped
    every few minutes: the control-plane guard rewrote each owner-bound profile as
    ``scene_execution_owner_store_missing`` and the intake and listener guards
    wrote the real owner state back. A row this process cannot judge therefore
    keeps the served row for the same profile bytes; every other row, and any row
    nothing has judged yet, is still written from this pass.
    """
    catalog = Path(catalog_path).expanduser().resolve()
    timings: dict[str, float] = {}
    descriptors, unjudged = _project_published_profiles(profile_dir, timings)

    try:
        current = catalog.read_bytes()
    except FileNotFoundError:
        current = b""
    except OSError as exc:
        raise LaunchCatalogError(f"catalog_unreadable:{catalog}") from exc

    carried = 0
    if unjudged:
        served = _served_rows(current, unjudged)
        for index, row in enumerate(descriptors):
            key = (str(row["profile_id"]), str(row["profile_digest"]))
            if key in unjudged and key in served:
                descriptors[index] = served[key]
                carried += 1
    expected = _catalog_bytes(descriptors)
    owner_state = {"owner_state_unjudged": len(unjudged), "owner_state_carried_forward": carried,
                   "pass_timing_seconds": timings}

    if current == expected:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "consistent",
            "catalog_path": str(catalog),
            "profile_ids_added": [],
            "profile_ids_removed": [],
            **owner_state,
            "provider_mutation_performed": False,
        }

    before = _catalog_ids(current)
    after = _catalog_ids(expected)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    # Publication installs the catalog read-only (0440) so no consumer can
    # edit it in place. Its own owner cannot reopen that for writing either,
    # so repairing drift has to replace the inode rather than truncate it --
    # otherwise every repair raises catalog_unwritable, and because this runs
    # from the intake unit's ExecStartPre, intake stops starting at all.
    try:
        previous = catalog.stat() if catalog.exists() else None
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{catalog.name}.", suffix=".tmp", dir=catalog.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            if previous is not None:
                os.chmod(temporary, stat.S_IMODE(previous.st_mode))
                try:
                    os.chown(temporary, -1, previous.st_gid)
                except PermissionError:
                    # The replacement already carries the service account's
                    # own group; only a foreign-gid catalog needs the chown,
                    # and that case is caught by the readback below.
                    pass
            os.replace(temporary, catalog)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if catalog.read_bytes() != expected:
            raise OSError(f"catalog readback mismatch: {catalog}")
    except OSError as exc:
        raise LaunchCatalogError(f"catalog_unwritable:{catalog}") from exc

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "repaired",
        "catalog_path": str(catalog),
        "profile_ids_added": sorted(after - before),
        "profile_ids_removed": sorted(before - after),
        **owner_state,
        "provider_mutation_performed": False,
    }


def _served_rows(
    payload: bytes, wanted: set[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """The valid served rows for ``wanted`` profile ids and digests; none from unreadable bytes."""
    try:
        rows = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(rows, list):
        return {}
    served: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("profile_id")), str(row.get("profile_digest")))
        if key in wanted and not validate_public_launch_profile_descriptor(row):
            served[key] = row
    return served


__all__ = [
    "SCHEMA_VERSION",
    "LaunchCatalogError",
    "build_catalog_payload",
    "reconcile_public_catalog",
]
