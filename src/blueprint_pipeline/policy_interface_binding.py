"""Explicit admitted client interfaces, separate from product candidate IDs.

Each interface names the frozen candidate whose observation and action
contract it uses. ``openpi_droid_https.v1`` is that same pi0.5-DROID contract
(two 224x224 views, joint and gripper state, prompt; joint-position action
chunk) delivered by a robot team's own authenticated HTTPS service instead of
Blueprint's pinned OpenPI server. Beta journey step: L5 team policy evaluation.
"""
from collections.abc import Mapping
from typing import Any

INTERFACES = {
    "openpi_droid.v1": "pi05_droid",
    "groot_n17_droid.v1": "groot_n17_droid",
    "openpi_droid_https.v1": "pi05_droid",
    # A team's own robot and cameras (``team_embodiment``); no frozen candidate shares it.
    "blueprint_policy_https.v1": "team_embodiment",
}
# Written out, not inverted: two interfaces share pi05_droid's contract, and
# only the pinned OpenPI one may ever serve the frozen pi05_droid candidate.
LEGACY_INTERFACES = {"pi05_droid": "openpi_droid.v1", "groot_n17_droid": "groot_n17_droid.v1"}
TEAM_HTTPS_INTERFACES = frozenset({"openpi_droid_https.v1", "blueprint_policy_https.v1"})


def resolve_policy_interface(spec: Mapping[str, Any]) -> str:
    candidate = spec.get("candidate_id")
    interface = spec.get("policy_interface_id") or LEGACY_INTERFACES.get(candidate)
    if interface not in INTERFACES:
        raise ValueError("policy_interface_not_admitted")
    if candidate in LEGACY_INTERFACES and interface != LEGACY_INTERFACES[candidate]:
        raise ValueError("policy_interface_frozen_candidate_mismatch")
    return interface


def observation_interface(spec: Mapping[str, Any]) -> str:
    return INTERFACES[resolve_policy_interface(spec)]
