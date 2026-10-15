"""Policy clients for the Quick-10 canary: the frozen pair, or one admitted team candidate.

Beta journey step: L5 team policy evaluation. The canary worker resolves every
candidate's client here. A spec on a team HTTPS interface gets the team's own
authenticated HTTPS client, with its bearer token read from the environment
variable the sealed execution spec names; every other spec keeps the frozen
pair's pinned OpenPI and GR00T clients unchanged. Only the canary bundle binds
this factory, so the other policy bundles never ship the team transport.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .policy_interface_binding import TEAM_HTTPS_INTERFACES, resolve_policy_interface


def canary_policy_client(spec: Mapping[str, Any], *,
                         groot_worker_identity_receipt: Mapping[str, Any] | None = None) -> Any:
    if resolve_policy_interface(spec) in TEAM_HTTPS_INTERFACES:
        from .droid_https_policy_client import DroidHttpsPolicySpec, DroidHttpsTeamPolicyClient

        return DroidHttpsTeamPolicyClient(
            spec=DroidHttpsPolicySpec(**spec["policy_spec"]),
            bearer_token=os.environ.get(str(spec["policy_endpoint"]["credential_env"])),
        )
    from .native_task_arena_policy_worker import _policy_client

    return _policy_client(spec, groot_worker_identity_receipt=groot_worker_identity_receipt)


__all__ = ["canary_policy_client"]
