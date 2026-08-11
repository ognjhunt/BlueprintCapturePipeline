"""Shared fail-closed bindings for single-use paid-attempt authorities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ALLOWLIST_GROUPS = ("external_provider_owned", "same_goal_concurrent")


def _normalize_instance_ids(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            return None
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        return None
    return tuple(sorted(normalized))


def normalize_external_instance_allowlist(value: Any) -> tuple[int, ...] | None:
    """Normalize the v1 external-only allowlist for compatibility callers."""

    return _normalize_instance_ids(value)


def normalize_active_instance_allowlist(
    value: Any,
) -> dict[str, tuple[int, ...]] | None:
    """Normalize a bound external and same-goal concurrent instance allowlist.

    Legacy list values represent only externally owned instances. New paid
    authorities use the two explicit groups so a bounded 1..N-object goal can
    admit known same-goal sibling instances without relabeling them external.
    """

    if isinstance(value, Mapping):
        if set(value) != set(ALLOWLIST_GROUPS):
            return None
        external = _normalize_instance_ids(value.get("external_provider_owned"))
        same_goal = _normalize_instance_ids(value.get("same_goal_concurrent"))
        if external is None or same_goal is None or set(external) & set(same_goal):
            return None
        return {
            "external_provider_owned": external,
            "same_goal_concurrent": same_goal,
        }
    external = _normalize_instance_ids(value)
    if external is None:
        return None
    return {"external_provider_owned": external, "same_goal_concurrent": ()}


def flatten_active_instance_allowlist(
    value: Mapping[str, Sequence[int]],
) -> tuple[int, ...]:
    return tuple(sorted({item for group in ALLOWLIST_GROUPS for item in value[group]}))


def active_instance_allowlist_metadata_error(
    authority: Mapping[str, Any],
    *,
    allowlist: Mapping[str, Sequence[int]],
) -> str | None:
    """Fail closed unless each same-goal instance has a bound authority digest.

    The provider inventory guard receives a flattened ID set, while the paid
    authority must preserve why each pre-existing instance is admitted.  The
    mapping is deliberately independent of scene/task identity and supports a
    bounded 1..N object campaign without treating a sibling goal instance as
    an externally owned workload.
    """

    same_goal = tuple(allowlist["same_goal_concurrent"])
    has_metadata = any(
        key in authority
        for key in (
            "concurrent_goal_id",
            "same_goal_concurrent_members",
            "concurrent_member_authority_digests",
        )
    )
    if not same_goal:
        return "same_goal_concurrent_allowlist_metadata_unexpected" if has_metadata else None
    goal_id = authority.get("concurrent_goal_id")
    members = authority.get("same_goal_concurrent_members")
    if not isinstance(goal_id, str) or not goal_id.strip() or not isinstance(members, list):
        return "same_goal_concurrent_allowlist_metadata_invalid"
    expected_ids = set(same_goal)
    observed_ids: list[int] = []
    for member in members:
        if not isinstance(member, Mapping) or set(member) != {
            "instance_id",
            "paid_attempt_authority_digest",
        }:
            return "same_goal_concurrent_allowlist_metadata_invalid"
        instance_id = member.get("instance_id")
        digest = member.get("paid_attempt_authority_digest")
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
            return "same_goal_concurrent_allowlist_metadata_invalid"
        if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
            return "same_goal_concurrent_allowlist_metadata_invalid"
        observed_ids.append(instance_id)
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        return "same_goal_concurrent_allowlist_metadata_invalid"
    return None


__all__ = [
    "ALLOWLIST_GROUPS",
    "active_instance_allowlist_metadata_error",
    "flatten_active_instance_allowlist",
    "normalize_active_instance_allowlist",
    "normalize_external_instance_allowlist",
]
