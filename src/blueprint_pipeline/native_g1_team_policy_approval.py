"""Bind one team-owned G1 policy to operator-reviewed runtime authority.

WebApp registration records a delivery description, not permission to send
site observations or allocate a GPU. This separate, digest-sealed approval is
supplied by the task operator and rechecked before a team policy is opened.
It contains a secret reference but never the credential value.
"""

from __future__ import annotations

import re
import time
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_packet_planning_setup import validate_packet_planning_setup
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


SCHEMA = "native_g1_team_policy_approval.v1"
OBJECTIVES = frozenset({"task_success", "g1_navigation_goal"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FIELDS = frozenset({
    "schema_version", "owner", "profile_digest", "source_setup_digest",
    "source_packet_receipt_digest", "objective_ids", "runtime_binding",
    "site_observation_exchange_authorized", "model_execution_rights_reviewed",
    "operator_reviewer", "expires_at_epoch", "claim_ceiling",
    "public_redistribution_authorized", "approval_digest",
})


def _binding_valid(binding: Any, delivery: Mapping[str, Any], profile_digest: str) -> bool:
    if not isinstance(binding, Mapping) or binding.get("profile_digest") != profile_digest:
        return False
    mode = delivery["mode"]
    if binding.get("mode") != mode:
        return False
    if mode == "authenticated_endpoint":
        if set(binding) != {
            "mode", "profile_digest", "approved_origin", "resolved_secret_ref"
        } or binding.get("resolved_secret_ref") != delivery["auth_secret_ref"]:
            return False
        endpoint = urlsplit(delivery["endpoint_url"])
        return binding.get("approved_origin") == f"https://{endpoint.hostname}"
    if mode == "container":
        gpu_device = binding.get("gpu_device")
        return (
            set(binding) == {"mode", "profile_digest", "image_ref", "gpu_device"}
            and binding.get("image_ref") == delivery["image_ref"]
            and (gpu_device is None or type(gpu_device) is int and gpu_device >= 0)
        )
    if mode == "noncontainer_artifact":
        path = binding.get("staged_artifact_path")
        return (
            set(binding) == {
                "mode", "profile_digest", "artifact_sha256", "staged_artifact_path"
            }
            and binding.get("artifact_sha256") == delivery["artifact_sha256"]
            and isinstance(path, str)
            and 1 <= len(path) <= 4096
            and not any(char in path for char in "\x00\n\r")
            and Path(path).is_absolute()
            and ".." not in Path(path).parts
            and not Path(path).is_symlink()
        )
    return False


def validate_g1_team_policy_approval(
    value: Mapping[str, Any] | Any,
    *,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    objective_id: str,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Require exact owner, packet, objective, delivery, rights and expiry."""

    setup = validate_packet_planning_setup(trusted_setup)
    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=setup, authenticated_owner=authenticated_owner
    )
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("g1_team_policy_approval_shape_invalid")
    approval = dict(value)
    objectives = approval.get("objective_ids")
    expiry = approval.get("expires_at_epoch")
    now = time.time() if now_epoch is None else now_epoch
    reviewer = approval.get("operator_reviewer")
    if (
        approval.get("schema_version") != SCHEMA
        or approval.get("owner") != dict(authenticated_owner)
        or approval.get("profile_digest") != bound["profile_digest"]
        or approval.get("source_setup_digest") != setup["setup_digest"]
        or approval.get("source_packet_receipt_digest")
        != setup["source_packet_receipt_digest"]
        or not isinstance(objectives, list)
        or not 1 <= len(objectives) <= 2
        or not all(isinstance(item, str) for item in objectives)
        or len(set(objectives)) != len(objectives)
        or not set(objectives) <= OBJECTIVES
        or objective_id not in objectives
        or not _binding_valid(
            approval.get("runtime_binding"), bound["delivery"], bound["profile_digest"]
        )
        or approval.get("site_observation_exchange_authorized") is not True
        or approval.get("model_execution_rights_reviewed") is not True
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or type(expiry) not in (int, float)
        or not math.isfinite(expiry)
        or not now < expiry <= now + 90 * 86400
        or approval.get("claim_ceiling") != "development_only"
        or approval.get("public_redistribution_authorized") is not False
        or not isinstance(approval.get("approval_digest"), str)
        or _DIGEST.fullmatch(approval["approval_digest"]) is None
        or approval["approval_digest"]
        != canonical_digest(approval, digest_field="approval_digest")
    ):
        raise ValueError("g1_team_policy_approval_binding_invalid")
    return approval


__all__ = ["SCHEMA", "validate_g1_team_policy_approval"]
