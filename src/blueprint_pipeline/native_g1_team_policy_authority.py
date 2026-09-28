"""Recheck a selected G1 team policy and operator approval before dispatch.

The signed intake is an immutable request, not runtime authority. A dispatcher
must reopen the current packet registry and a separately supplied approval at
each admission boundary, including after a slow spend-guard refresh.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_intake import REGISTRY_SCHEMA, _read, _selected_binding
from .native_g1_team_policy_approval import validate_g1_team_policy_approval
from .native_g1_team_policy_run_intake import INTENT_SCHEMA
from .native_g1_team_policy_run_request import validate_g1_team_policy_run_request
from .task_evaluation_packet_planning_setup import make_packet_planning_setup


_INTENT_ID = re.compile(r"g1-team-policy-[0-9a-f]{64}\Z")


def verify_g1_team_policy_authority(
    *,
    intent_path: Path,
    registry_path: Path,
    approval_path: Path,
    trusted_clients: set[str],
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Return exact sealed authority or fail before any provider mutation."""

    paths = (Path(intent_path), Path(registry_path), Path(approval_path))
    if any(not path.is_absolute() or path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("g1_team_policy_authority_path_invalid")
    intent = _read(paths[0], field="intent_digest")
    request = intent.get("request")
    intent_id = intent.get("intent_id")
    if (
        intent.get("schema_version") != INTENT_SCHEMA
        or intent.get("status") != "accepted_pending_operator_approval"
        or intent.get("claim_ceiling") != "development_only"
        or intent.get("provider_mutation_performed") is not False
        or intent.get("authenticated_issuer") not in trusted_clients
        or not isinstance(request, dict)
        or not isinstance(intent_id, str)
        or _INTENT_ID.fullmatch(intent_id) is None
        or paths[0].parent.name != intent_id
        or intent_id != "g1-team-policy-" + digest({
            "owner": request.get("owner"), "run_id": request.get("run_id"),
        }).removeprefix("sha256:")
    ):
        raise ValueError("g1_team_policy_intent_invalid")
    registry = _read(paths[1], field="registry_digest")
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA
        or registry["registry_digest"] != intent.get("registry_digest")
    ):
        raise ValueError("g1_team_policy_registry_changed")
    binding = _selected_binding(registry, request)
    if (
        binding != intent.get("binding")
        or digest(binding) != intent.get("binding_digest")
    ):
        raise ValueError("g1_team_policy_binding_changed")
    setup = make_packet_planning_setup(source_packet_dir=Path(binding["source_packet_dir"]))
    if (
        setup["scene_id"] != binding["scene_id"]
        or setup["task_id"] != binding["task_id"]
        or setup["source_packet_receipt_digest"] != binding["source_packet_receipt_digest"]
    ):
        raise ValueError("g1_team_policy_packet_changed")
    now = time.time() if now_epoch is None else now_epoch
    selected = validate_g1_team_policy_run_request(
        request, trusted_setup=setup, authenticated_owner=binding["owner"],
        now_epoch=now,
    )
    if paths[2].stat().st_size > 1024 * 1024:
        raise ValueError("g1_team_policy_approval_file_invalid")
    try:
        approval_value = json.loads(paths[2].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("g1_team_policy_approval_file_invalid") from exc
    approval = validate_g1_team_policy_approval(
        approval_value,
        profile=selected["policy_profile"],
        trusted_setup=setup,
        authenticated_owner=binding["owner"],
        objective_id=selected["objective_id"],
        now_epoch=now,
    )
    return {
        "intent": intent,
        "registry_binding": binding,
        "trusted_setup": setup,
        "operator_approval": approval,
    }


__all__ = ["verify_g1_team_policy_authority"]
