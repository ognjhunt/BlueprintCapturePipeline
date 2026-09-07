"""Outcome-blind admission to the exact checkpoint inventory this lane executes.

Owner artifact identities use checkpoint *inventory* digests, as the canonical
policy-canary setup does; they are neither model-name hashes nor downloaded
file hashes.  The persistent scene-intake schema intentionally remains
generic, so this module is the active ADP-009D capability gate.  It is called
by scene progression before source resolution, attempt reservation, or any
construction work.

Keep the supported pair in one direction: the canonical admission registry is
the source of the exact inventory digest, while the intake module owns the
currently frozen candidate order.  A request is never repaired by replacing a
digest or candidate; it receives a typed refusal instead.
"""

from collections.abc import Mapping
from typing import Any

from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from .task_evaluation_scene_intake import SUPPORTED_POLICY_CANDIDATE_IDS


CAPABILITY_BLOCKER = "scene_policy_checkpoint_capability_unavailable"


def supported_policy_candidates() -> list[dict[str, str]]:
    """Return the exact ordered pair admitted by the live canary lane.

    ``EXPECTED_CANDIDATES`` is the canonical, outcome-blind candidate registry;
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
    return [] if requested == supported_policy_candidates() else [CAPABILITY_BLOCKER]


__all__ = ["CAPABILITY_BLOCKER", "policy_capability_blockers", "supported_policy_candidates"]
