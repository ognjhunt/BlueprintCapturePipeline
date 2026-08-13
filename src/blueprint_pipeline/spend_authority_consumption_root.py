"""Resolve the root that holds single-use paid-attempt consumption records.

Every paid lane enforces "one provider allocation per authorization" by writing
an exclusive record and treating ``FileExistsError`` as an already-consumed
attempt. Seven modules independently computed that record's location as
``Path.home() / ".blueprint-spend-authority" / "consumed"``, evaluated at import
time.

That location cannot be written on a correctly hardened host. The deployed
dispatcher runs as a service account whose home is ``/nonexistent`` under a unit
that sets ``ProtectHome=true``, so ``mkdir`` fails and every paid attempt is
refused with a consumption-write blocker *after* its authority has already
validated. The failure looks like a spend-authority problem and is really a
filesystem-layout problem, which is what made it expensive to diagnose.

Resolving the root here, at call time and from configuration, fixes all seven
lanes at once and lets a deployment place the ledger inside the writable paths
its unit already grants. The default is unchanged, so developer environments
behave exactly as before.

The ledger is security-relevant: it is the only thing preventing one signed
authorization from funding repeated provider allocations. Callers keep their own
ownership and permission checks on the returned directory.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Absolute path to the spend-authority tree. A deployment sets this to a
#: directory inside its unit's ``ReadWritePaths``.
SPEND_AUTHORITY_ROOT_ENV = "BLUEPRINT_SPEND_AUTHORITY_ROOT"

_DEFAULT_DIRECTORY_NAME = ".blueprint-spend-authority"
_CONSUMED_DIRECTORY_NAME = "consumed"
_AUTHORIZATIONS_DIRECTORY_NAME = "authorizations"


class SpendAuthorityRootError(ValueError):
    """The configured spend-authority root is unusable."""


def spend_authority_root() -> Path:
    """Return the spend-authority tree, honouring deployment configuration.

    Resolved on every call rather than at import so a process that sets the
    variable during start-up is not silently bound to the value that existed
    when the module happened to be imported -- the import-time binding is what
    made the original defect survive configuration changes.
    """
    configured = str(os.environ.get(SPEND_AUTHORITY_ROOT_ENV) or "").strip()
    if not configured:
        return Path.home() / _DEFAULT_DIRECTORY_NAME
    root = Path(configured).expanduser()
    if not root.is_absolute():
        # A relative root would depend on the working directory, so the same
        # authorization could be consumed once per directory.
        raise SpendAuthorityRootError(
            f"spend_authority_root_must_be_absolute:{SPEND_AUTHORITY_ROOT_ENV}"
        )
    return root


def consumption_root() -> Path:
    """Return the directory holding single-use consumption records."""
    return spend_authority_root() / _CONSUMED_DIRECTORY_NAME


def prepare_consumption_root() -> Path:
    """Return the consumption directory, created and tightened to 0o700.

    Two components disagreed about this directory's mode. The ledger
    reconciler created it with the process umask -- 0o755 on the deployed host
    -- while every paid lane refuses a consumption root with any group or other
    bit set. ``mkdir(mode=0o700, exist_ok=True)`` does not change an existing
    directory, so on any host where the reconciler ran first, which is every
    deployed host, every paid attempt was refused.

    The mode check exists so nothing but this process can read or alter the
    ledger that stops one authorization funding repeated allocations. Where we
    own the directory, tightening it enforces that property; refusing instead
    leaves a state we could have fixed and could not act on, reported as a
    write failure that names the symptom rather than the cause. Ownership is
    still required, and a symlink is still refused outright: those we cannot
    make safe.
    """

    root = consumption_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise SpendAuthorityRootError(f"spend_authority_consumption_root_unwritable:{exc.errno}")
    if root.is_symlink():
        raise SpendAuthorityRootError("spend_authority_consumption_root_is_symlink")
    stat_result = root.stat()
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        raise SpendAuthorityRootError("spend_authority_consumption_root_not_owned")
    if stat_result.st_mode & 0o077:
        try:
            root.chmod(0o700)
        except OSError:
            raise SpendAuthorityRootError(
                "spend_authority_consumption_root_permissions_unsafe"
            ) from None
    return root


def authorizations_root() -> Path:
    """Return the directory holding externally supplied authorizations."""
    return spend_authority_root() / _AUTHORIZATIONS_DIRECTORY_NAME


__all__ = [
    "SPEND_AUTHORITY_ROOT_ENV",
    "SpendAuthorityRootError",
    "authorizations_root",
    "consumption_root",
    "prepare_consumption_root",
    "spend_authority_root",
]
