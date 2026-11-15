"""Outcome-blind admission to the exact checkpoint inventory this lane executes.

Owner artifact identities use checkpoint *inventory* digests, as the canonical
policy-canary setup does; they are neither model-name hashes nor downloaded
file hashes.  The persistent scene-intake schema intentionally remains
generic, so this module is the active ADP-009D capability gate.  It is called
by scene progression before source resolution, attempt reservation, or any
construction work.

The exact inventory digests are frozen here, as the retired admission registry
held them, while the intake module owns the currently frozen candidate order.
A request is never repaired by replacing a digest or candidate; it receives a
typed refusal instead.
"""

from collections.abc import Mapping
from typing import Any

from .task_evaluation_scene_intent_contracts import SUPPORTED_POLICY_CANDIDATE_IDS

CAPABILITY_BLOCKER = "scene_policy_checkpoint_capability_unavailable"
# The checkpoint inventory digests the retired policy lane admitted, frozen unchanged so the same
# intents pass this gate and the same intents are refused.
EXPECTED_CANDIDATES = {
    "pi05_droid": {
        "checkpoint_inventory_digest": "sha256:492ef95fa2e0ea8c026fda4bf6a2662758e7958ab5223ecb270cde5bc3797063",
    },
    "groot_n17_droid": {
        "checkpoint_inventory_digest": "sha256:5d1d83ab34215da2dcaa049d70e93ccec18687591ad5760c5183fc1fd6e035fd",
    },
}


def supported_policy_candidates() -> list[dict[str, str]]:
    """Return the exact ordered pair admitted by the live canary lane.

    ``EXPECTED_CANDIDATES`` is the frozen, outcome-blind candidate registry;
    its ``checkpoint_inventory_digest`` is the digest carried by the real
    scene setup and policy specs.  Constructing fresh dictionaries here keeps
    callers from mutating the registry or accidentally retaining a reference
    to an owner request.
    """

    result: list[dict[str, str]] = []
    for candidate_id in SUPPORTED_POLICY_CANDIDATE_IDS:
        candidate = EXPECTED_CANDIDATES.get(candidate_id)
        digest = candidate.get("checkpoint_inventory_digest") if candidate else None
        if not isinstance(candidate_id, str) or not isinstance(digest, str):
            # A damaged source registry must fail closed at the same gate as an
            # unsupported owner request.  Do not let a malformed constant turn
            # into an exception after an attempt has already been reserved.
            return []
        result.append({"id": candidate_id, "artifact_digest": digest})
    return result


def policy_capability_blockers(request: Mapping[str, Any] | Any) -> list[str]:
    """Return a typed blocker when the request names an unsupported pair.

    Scene intents have already passed structural intake by the time this is
    called, but keeping this function total prevents malformed retained input
    from falling through to source or reservation code if a caller reuses the
    capability gate directly.
    """

    execution = request.get("execution") if isinstance(request, Mapping) else None
    requested = execution.get("policy_candidates") if isinstance(execution, Mapping) else None
    if isinstance(execution, Mapping) and 'purpose' in execution:
        from .task_evaluation_scene_execution_scope import scene_preparation_only
        task = request.get('task')
        return [] if (scene_preparation_only(request) and requested == [] and isinstance(task, Mapping)
                      and 'robot_binding_id' not in task) else [CAPABILITY_BLOCKER]
    return [] if requested == supported_policy_candidates() else [CAPABILITY_BLOCKER]


__all__ = ["CAPABILITY_BLOCKER", "policy_capability_blockers", "supported_policy_candidates"]
