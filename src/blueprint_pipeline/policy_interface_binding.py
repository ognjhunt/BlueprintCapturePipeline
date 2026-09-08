"""Explicit admitted client interfaces, separate from product candidate IDs."""
from collections.abc import Mapping
from typing import Any

INTERFACES = {"openpi_droid.v1": "pi05_droid", "groot_n17_droid.v1": "groot_n17_droid"}
LEGACY_INTERFACES = {value: key for key, value in INTERFACES.items()}


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
