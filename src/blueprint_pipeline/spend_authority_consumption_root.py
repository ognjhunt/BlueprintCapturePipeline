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


def authorizations_root() -> Path:
    """Return the directory holding externally supplied authorizations."""
    return spend_authority_root() / _AUTHORIZATIONS_DIRECTORY_NAME


__all__ = [
    "SPEND_AUTHORITY_ROOT_ENV",
    "SpendAuthorityRootError",
    "authorizations_root",
    "consumption_root",
    "spend_authority_root",
]
