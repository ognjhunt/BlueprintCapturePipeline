"""Policy clients for the Quick-10 canary: the frozen pair, or one admitted team candidate.

Beta journey step: L5 team policy evaluation. The canary worker resolves every
candidate's client here. A spec on a team HTTPS interface gets the team's own
authenticated HTTPS client: the DROID contract, or the team's own robot
(``blueprint_policy_https.v1``). Its bearer token is read from the private file
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
    interface = resolve_policy_interface(spec)
    if interface == "openwam_piper_eef20.v1":
        from .openwam_piper_client_dependencies import activate_client_dependencies
        from .openwam_piper_policy import OpenWAMPiperPolicyClient
        from .openwam_piper_runtime import policy_binding, validate_execution_spec

        validate_execution_spec(spec)
        activate_client_dependencies()
        client = OpenWAMPiperPolicyClient(binding=policy_binding(spec["policy_spec"]))
        client.execution_binding_digest = spec["execution_spec_digest"]
        return client
    if interface == "molmoact2_yam_joint14.v1":
        from .molmoact2_yam_policy import MolmoAct2YAMPolicyClient
        from .molmoact2_yam_runtime import policy_binding, validate_execution_spec

        validate_execution_spec(spec)
        client = MolmoAct2YAMPolicyClient(binding=policy_binding(spec["policy_spec"]))
        client.execution_binding_digest = spec["execution_spec_digest"]
        return client
    if interface == "task_family_policy_https.v1":
        from .task_family_policy_https_client import TaskFamilyPolicyHttpsClient, TaskFamilyPolicySpec

        return TaskFamilyPolicyHttpsClient(
            spec=TaskFamilyPolicySpec(**spec["policy_spec"]),
            prompt=str(spec["prompt"]),
            bearer_token=_bearer_token(str(spec["policy_endpoint"]["credential_env"])),
        )
    if interface == "dreamlabs_dream1.v1":
        from .dreamlabs_policy_client import DreamLabsPolicyClient, DreamLabsPolicySpec

        return DreamLabsPolicyClient(
            spec=DreamLabsPolicySpec(**spec["policy_spec"]),
            bearer_token=_bearer_token(str(spec["policy_endpoint"]["credential_env"])),
        )
    if interface == "blueprint_policy_https.v1":
        from .team_policy_https_client import TeamPolicyHttpsClient, TeamPolicySpec

        return TeamPolicyHttpsClient(
            spec=TeamPolicySpec(**spec["policy_spec"]),
            bearer_token=_bearer_token(str(spec["policy_endpoint"]["credential_env"])),
        )
    if interface in TEAM_HTTPS_INTERFACES:
        from .droid_https_policy_client import DroidHttpsPolicySpec, DroidHttpsTeamPolicyClient

        return DroidHttpsTeamPolicyClient(
            spec=DroidHttpsPolicySpec(**spec["policy_spec"]),
            bearer_token=_bearer_token(str(spec["policy_endpoint"]["credential_env"])),
        )
    from .native_task_arena_policy_worker import _policy_client

    return _policy_client(spec, groot_worker_identity_receipt=groot_worker_identity_receipt)


__all__ = ["canary_policy_client"]
