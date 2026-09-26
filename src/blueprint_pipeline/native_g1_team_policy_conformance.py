"""Exercise a team policy with synthetic G1 inputs before any site episode.

This calls the configured wire client once. A passing response proves only
that the live interface accepted a synthetic G1 observation and returned a
semantic-v3 action. Runtime identity, rights and paid launch remain separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .native_g1_humanoidarena_policy_client import validate_semantic_v3_infer_response
from .native_g1_team_policy_https_client import NativeG1TeamPolicyHttpsClient
from .native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID, G1_PRESET_ID
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


SCHEMA = "native_g1_team_policy_synthetic_conformance.v1"
TASK = "Synthetic G1 interface check; no site observation or task outcome."


def run_g1_team_policy_synthetic_conformance(
    *,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    policy_client: Any,
) -> dict[str, Any]:
    """Send one zero observation through the real client without GPU launch."""

    bound = validate_team_policy_delivery_profile(
        profile,
        trusted_setup=trusted_setup,
        authenticated_owner=authenticated_owner,
    )
    if (
        bound["robot_preset_id"] != G1_PRESET_ID
        or bound["embodiment_id"] != G1_EMBODIMENT_ID
        or bound["observation_schema_id"] != "humanoidarena_head_rgb_state64_v1"
        or bound["action_schema_id"] != "humanoidarena_semantic_v3"
        or not callable(getattr(policy_client, "reset", None))
        or not callable(getattr(policy_client, "infer_chunk", None))
    ):
        raise ValueError("g1_team_policy_conformance_interface_invalid")
    client_profile_digest = getattr(policy_client, "profile_digest", None)
    if client_profile_digest is not None and client_profile_digest != bound["profile_digest"]:
        raise ValueError("g1_team_policy_conformance_client_binding_invalid")
    policy_client.reset(seed=0)
    response = policy_client.infer_chunk(
        front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        observation_state=[0.0] * 64,
        task=TASK,
    )
    actions = validate_semantic_v3_infer_response({"action_chunk": response})
    receipt = {
        "schema_version": SCHEMA,
        "status": "synthetic_wire_compatible",
        "profile_digest": bound["profile_digest"],
        "source_setup_digest": bound["source_setup_digest"],
        "robot_preset_id": bound["robot_preset_id"],
        "delivery_mode": bound["delivery"]["mode"],
        "synthetic_policy_query_count": 1,
        "returned_action_count": len(actions),
        "site_observation_sent": False,
        "site_policy_query_count": 0,
        "task_scored": False,
        "runtime_identity_verified": False,
        "rights_authorized": False,
        "paid_launch_authorized": False,
        "public_redistribution_authorized": False,
        "claim_ceiling": "planning_only",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def run_g1_team_endpoint_synthetic_conformance(
    *,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    approved_origin: str,
    resolved_secret_ref: str,
    credential: str,
    fetcher: Any = None,
) -> dict[str, Any]:
    """Exercise the registered HTTPS endpoint with bound synthetic G1 input."""

    bound = validate_team_policy_delivery_profile(
        profile,
        trusted_setup=trusted_setup,
        authenticated_owner=authenticated_owner,
    )
    robots = [
        robot
        for robot in trusted_setup["robot_presets"]
        if robot["robot_preset_id"] == bound["robot_preset_id"]
    ]
    robot = robots[0]
    interface = {
        "robot_preset_id": robot["robot_preset_id"],
        "embodiment_id": robot["embodiment_id"],
        "observation_schema_id": robot["observation_schema"]["schema_id"],
        "action_schema_id": robot["action_schema"]["schema_id"],
    }
    options = {"fetcher": fetcher} if fetcher is not None else {}
    client = NativeG1TeamPolicyHttpsClient(
        profile=bound,
        expected_owner=authenticated_owner,
        expected_setup_digest=trusted_setup["setup_digest"],
        expected_interface=interface,
        approved_origin=approved_origin,
        resolved_secret_ref=resolved_secret_ref,
        credential=credential,
        **options,
    )
    return run_g1_team_policy_synthetic_conformance(
        profile=bound,
        trusted_setup=trusted_setup,
        authenticated_owner=authenticated_owner,
        policy_client=client,
    )


def run_g1_team_process_synthetic_conformance(
    *,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    process: Any,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Probe an already isolated OCI or noncontainer JSONL process."""

    bound = validate_team_policy_delivery_profile(
        profile,
        trusted_setup=trusted_setup,
        authenticated_owner=authenticated_owner,
    )
    if bound["delivery"]["mode"] not in {"container", "noncontainer_artifact"}:
        raise ValueError("g1_team_policy_conformance_process_mode_invalid")
    client = NativeG1TeamPolicyJsonlClient(
        process,
        timeout_seconds=timeout_seconds,
        profile_digest=bound["profile_digest"],
    )
    return run_g1_team_policy_synthetic_conformance(
        profile=bound,
        trusted_setup=trusted_setup,
        authenticated_owner=authenticated_owner,
        policy_client=client,
    )
