"""Policy clients for the Quick-10 canary: the frozen pair, or one admitted team candidate.

Beta journey step: L5 team policy evaluation. The canary worker resolves every
candidate's client here. A spec on a team HTTPS interface gets the team's own
authenticated HTTPS client. Its bearer token is read from the private file
named by the ``*_FILE`` environment variable the sealed execution spec names;
the provider bootstrap writes that file at 0600 from the instance environment.
Every other spec keeps the frozen pair's pinned OpenPI and GR00T clients. Only the canary bundle binds
this factory, so the other policy bundles never ship the team transport.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .policy_interface_binding import TEAM_HTTPS_INTERFACES, resolve_policy_interface


def _bearer_token(credential_env: str) -> str | None:
    """Read the token from the private file the provider bootstrap wrote; never from argv or bundle bytes."""

    if not credential_env.endswith("_FILE"):
        return None
    raw = os.environ.get(credential_env)
    if not raw:
        return None
    path = Path(raw)
    try:
        status = path.lstat()
    except OSError:
        return None
    if (not path.is_absolute() or path.is_symlink() or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o077 or not 0 < status.st_size <= 65_536):
        return None
    return path.read_text(encoding="utf-8").strip() or None


def canary_policy_client(spec: Mapping[str, Any], *,
                         groot_worker_identity_receipt: Mapping[str, Any] | None = None) -> Any:
    if resolve_policy_interface(spec) in TEAM_HTTPS_INTERFACES:
        from .droid_https_policy_client import DroidHttpsPolicySpec, DroidHttpsTeamPolicyClient

        return DroidHttpsTeamPolicyClient(
            spec=DroidHttpsPolicySpec(**spec["policy_spec"]),
            bearer_token=_bearer_token(str(spec["policy_endpoint"]["credential_env"])),
        )
    from .native_task_arena_policy_worker import _policy_client

    return _policy_client(spec, groot_worker_identity_receipt=groot_worker_identity_receipt)


__all__ = ["canary_policy_client"]
