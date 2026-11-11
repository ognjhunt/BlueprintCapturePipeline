"""The catalog row of a session-served registry entry (``session_served_policy``), control plane only.

Derived from the native row it re-serves: the same policy identity, licence and
interface (``openpi_droid_https.v1`` in place of the native interface id), and
at least the native memory floor. Shipped in no provider bundle.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .session_served_policy import _CANDIDATE, INTERFACE_ID, registry_entries, validate_entry


def catalog_row(native_row: Mapping[str, Any], *, candidate_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The native row's identity and licence, served over the contract."""

    if native_row["interface"] is None or native_row["runtime"] is None:
        raise ValueError("session_served_policy_native_row_not_runnable")
    row = deepcopy(dict(native_row))
    row.pop("configuration_digest", None)
    row.pop("deployment", None)
    row.update(candidate_id=candidate_id, display_name=entry["display_name"])
    row["interface"]["policy_interface_id"] = INTERFACE_ID
    floor = native_row["runtime"]["minimum_gpu_memory_gb"]
    declared = entry.get("minimum_gpu_memory_gb")
    row["runtime"]["minimum_gpu_memory_gb"] = floor if declared is None or floor is None else max(floor, declared)
    row["eligibility"]["requires_credentials"] = []  # The session makes its own loopback token.
    return row


def catalog_for_profile(profile: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """Add one row per registry entry; an entry that cannot be derived is listed with its reason."""

    entries = registry_entries(profile)
    if not entries:
        return dict(base)
    from .beta_policy_candidate_catalog import seal_catalog

    rows = {row["candidate_id"]: row for row in base["candidates"]}
    native_rows = dict(rows)
    revisions = {}
    for key, raw in sorted(entries.items()):
        try:
            if key in native_rows:
                raise ValueError("session_served_policy_candidate_id_taken")
            entry = validate_entry(raw, candidate_id=key)
            native = native_rows.get(entry["native_candidate_id"])
            if native is None:
                raise ValueError("session_served_policy_native_candidate_missing")
            row = catalog_row(native, candidate_id=key, entry=entry)
            row = seal_catalog({**base, "candidates": [row]})["candidates"][0]
            revisions[key] = entry
        except (ValueError, KeyError, TypeError) as exc:
            code = str(exc).split(":", 1)[0]
            reason = code if re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code) else "session_served_policy_entry_invalid"
            if key in native_rows:
                # Never disable a packaged row because a registry entry reused its id.
                revisions[key] = {"status": reason}
                continue
            row = {"candidate_id": key if _CANDIDATE.fullmatch(str(key)) else "invalid-session-entry",
                   "display_name": key, "policy": {"kind": "checkpoint", "source": "operator-registration",
                   "revision": None, "digest": None, "digest_kind": "unverified"}, "interface": None, "runtime": None,
                   "supported_task_kinds": [], "eligibility": {"enabled": False, "commercial_use": "under_review",
                   "license_id": "unverified", "license_note": "The session-served registry entry could not be validated.",
                   "requires_credentials": [], "blockers": [reason]}}
            revisions[key] = {"status": reason}
        rows[row["candidate_id"]] = row
    return seal_catalog({**base, "candidates": list(rows.values()),
                         "revision": base["revision"] + ":session-served:" + canonical_digest(revisions)[7:]})


def interface_matches(configured: Mapping[str, Any], spec: Mapping[str, Any], native_row: Mapping[str, Any]) -> bool:
    """The registered row is the native row over the contract, with the served spec's horizons."""

    expected = deepcopy(dict(native_row["interface"]))
    expected.update(policy_interface_id=INTERFACE_ID, execution_horizon=spec["open_loop_horizon"],
                    action_horizon=spec["policy_spec"]["action_chunk_rows"])
    runtime = configured.get("runtime") or {}
    floor = native_row["runtime"]["minimum_gpu_memory_gb"]
    return (configured.get("interface") == expected and configured.get("policy") == native_row["policy"]
            and runtime.get("backend") == native_row["runtime"]["backend"]
            and runtime.get("runtime_id") == native_row["runtime"]["runtime_id"]
            and (floor is None or (runtime.get("minimum_gpu_memory_gb") or 0) >= floor))


__all__ = ["catalog_for_profile", "catalog_row", "interface_matches"]
