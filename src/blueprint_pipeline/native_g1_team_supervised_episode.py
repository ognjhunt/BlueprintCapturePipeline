"""Run one reviewed team policy in the retained G1 scene and seal its lifecycle.

The caller must first admit scene/model rights, the operator's exact runtime
binding, provider spend, and the pinned SONIC bridge. This module runs synthetic
wire conformance before sharing site observations, then scores one episode and
closes the team policy process even when inference or scoring fails. Provider
teardown and billing remain the paid controller's responsibility.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_shared_scene_episode import team_policy_candidate_id
from .native_g1_team_runtime_session import open_g1_team_runtime_session
from .native_g1_team_scored_scene_episode import run_g1_team_scored_scene_episode
from .task_evaluation_packet_planning_setup import validate_packet_planning_setup
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


SCHEMA = "native_g1_team_supervised_episode.v1"
FILENAME = SCHEMA + ".json"


def run_g1_team_supervised_episode(
    *,
    built: Any,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    approved_binding: Mapping[str, Any],
    sonic_bridge: Any,
    objective_id: str,
    max_steps: int,
    output_dir: Path,
    to_tensor: Callable[[Any], Any],
    make_action_tensor: Callable[..., Any],
    credential: str | None = None,
    fetcher: Any = None,
) -> dict[str, Any]:
    """Connect a bound runtime session to one scored G1 scene episode.

    A blocked attempt is returned with a terminal receipt after the output
    directory is created. The exception type is recorded without its message
    because transport exceptions may contain a private URL or credential.
    """

    setup = validate_packet_planning_setup(trusted_setup)
    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=setup, authenticated_owner=authenticated_owner
    )
    plan = getattr(built, "plan", None)
    if (
        not isinstance(plan, Mapping)
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
        or plan.get("scene_id") != bound["source_scene_id"]
        or plan.get("task_id") != bound["source_task_id"]
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or objective_id not in {"task_success", "g1_navigation_goal"}
        or type(max_steps) is not int
        or not 1 <= max_steps <= 3000
        or not isinstance(output_dir, Path)
        or not output_dir.is_absolute()
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise ValueError("g1_team_supervised_episode_admission_invalid")
    output_dir.mkdir(mode=0o700)
    session = None
    episode = None
    closed = None
    blocker = None
    phase = "synthetic_conformance"
    try:
        session = open_g1_team_runtime_session(
            profile=bound,
            trusted_setup=setup,
            authenticated_owner=authenticated_owner,
            approved_binding=approved_binding,
            output_dir=output_dir / "runtime",
            credential=credential,
            fetcher=fetcher,
        )
        phase = "scored_scene_episode"
        episode = run_g1_team_scored_scene_episode(
            built=built,
            profile=bound,
            trusted_setup=setup,
            authenticated_owner=authenticated_owner,
            policy_client=session.client,
            sonic_bridge=sonic_bridge,
            objective_id=objective_id,
            max_steps=max_steps,
            output_dir=output_dir / "episode",
            to_tensor=to_tensor,
            make_action_tensor=make_action_tensor,
        )
        session.link_scored_episode(episode)
        phase = "runtime_teardown"
    except BaseException as exc:  # noqa: BLE001 - retain terminal evidence for GPU attempts
        blocker = type(exc).__name__
    finally:
        if session is not None:
            try:
                closed = session.close()
            except BaseException as exc:  # noqa: BLE001 - failed teardown blocks completion
                closed = {"status": "teardown_blocked"}
                blocker = blocker or type(exc).__name__
    complete = (
        blocker is None
        and episode is not None
        and episode.get("status") == "development_only_scored_episode"
        and episode.get("profile_digest") == bound["profile_digest"]
        and episode.get("candidate_id") == team_policy_candidate_id(bound["profile_digest"])
        and isinstance(episode.get("score"), Mapping)
        and episode["score"].get("status") == "scored"
        and type(episode.get("policy_query_count")) is int
        and episode["policy_query_count"] > 0
        and episode.get("result_digest") == canonical_digest(episode, digest_field="result_digest")
        and session is not None
        and session.conformance.get("receipt_digest")
        == canonical_digest(session.conformance, digest_field="receipt_digest")
        and closed is not None
        and closed.get("status") == "closed"
        and closed.get("receipt_digest") == canonical_digest(closed, digest_field="receipt_digest")
        and closed.get("linked_scored_episode_result_digest") == episode["result_digest"]
    )
    result = {
        "schema_version": SCHEMA,
        "status": "completed_development_only" if complete else "blocked",
        "claim_ceiling": "development_only",
        "profile_digest": bound["profile_digest"],
        "candidate_id": team_policy_candidate_id(bound["profile_digest"]),
        "objective_id": objective_id,
        "delivery_mode": bound["delivery"]["mode"],
        "source_setup_digest": setup["setup_digest"],
        "scene_plan_digest": plan["plan_digest"],
        "synthetic_conformance_digest": (
            session.conformance.get("receipt_digest") if session is not None else None
        ),
        "scored_episode_result_digest": episode.get("result_digest") if episode else None,
        "policy_query_count": episode.get("policy_query_count") if episode else 0,
        "runtime_teardown_digest": closed.get("receipt_digest") if closed else None,
        "phase_reached": phase,
        "blocker_type": blocker if blocker is not None else (None if complete else "IncompleteEvidence"),
        "runtime_identity_attested_by_scored_episode": False,
        "provider_teardown_verified": False,
        "official_billing_reconciled": False,
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
        "public_redistribution_authorized": False,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with (output_dir / FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


__all__ = ["SCHEMA", "run_g1_team_supervised_episode"]
