"""Authenticated historical target observations; every target remains kept.

ADP-009D/day28. An exact observed version is neither a registration generation
nor an action grant. This leaf never creates, renews or removes target payload.
"""
from __future__ import annotations

import math

from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_scratch_lifetime import LANE_ROOTS

EXPECTED_SCHEMA = "control_plane_lane_expected_target_version.v1"
ATTESTATION_SCHEMA = "control_plane_lane_owner_target_version_attestation.v1"
REPORT_SCHEMA = "control_plane_lane_owner_target_version_report.v1"
MAX_EXPECTED_BYTES = 32768
MAX_REPORT_BYTES = 65536
MAX_OUTCOME_BYTES = 8192
MAX_LOCAL_RAW = 2 * 1024 * 1024
_DIRECTORY_FIELDS = frozenset({"dev", "ino", "type"})
_FILE_FIELDS = _DIRECTORY_FIELDS | {"mode", "uid", "gid", "nlink", "size_bytes", "mtime_ns", "ctime_ns"}
_LEASE_FIELDS = frozenset({"owner", "reference_kind", "reference_value", "reason", "class_intent",
                          "cleanup", "consumer_lifetime_contract", "created_at_epoch", "expires_at_epoch",
                          "renewed_at_epoch", "released_at_epoch", "size_budget_bytes"})
_EXPECTED_FIELDS = frozenset({"schema_version", "root", "lane", "name", "root_identity", "lane_identity",
                             "folder_identity", "lease_file_identity", "lease_raw_sha256", "lease_raw_size_bytes",
                             "lease_digest", "lease"})
_NO_AUTHORITY = dict(execution_authorized=False, apply_supported=False,
                    target_generation_bound=False, action_generation_bound=False,
                    general_reference_inventory_complete=False, general_process_inventory_complete=False,
                    consumer_fence_checked=False, references_clear=False, retirement_admission_checked=False,
                    budget_enforced=False, admission_checked=False, registration_applied=False,
                    mutations=0, candidate_bytes=None, estimated_reclaimable_bytes=None,
                    eta_contribution_bytes=None, eta_seconds=None)


class OwnerTargetVersionError(ValueError):
    """Fixed screened code, never input, paths, policy or OS exception text."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnerTargetVersionError(code)


def _counter(value, *, maximum=2**63 - 1, positive=False):
    return type(value) is int and (0 < value if positive else 0 <= value) and value <= maximum


def _epoch(value):
    return (type(value) in (int, float) and (type(value) is not int or value.bit_length() <= 63)
            and math.isfinite(value) and 0 <= value <= 2**63 - 1)


def _valid_id(value):
    return isinstance(value, str) and len(value) <= 80 and scratch._ID.fullmatch(value) is not None


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 71 and scratch._DIGEST.fullmatch(value) is not None


def _expected_version(raw, budget):
    code = "owner_target_expected_invalid"
    try:
        value = retained._document(raw, MAX_EXPECTED_BYTES, _work_budget=budget)
    except retained.CensusDecisionError:
        raise OwnerTargetVersionError(code) from None
    _require(set(value) == _EXPECTED_FIELDS and value["schema_version"] == EXPECTED_SCHEMA
             and isinstance(value["root"], str) and value["root"] in ("work", "inputs")
             and _valid_id(value["lane"]) and _valid_id(value["name"]), code)
    for key in ("root_identity", "lane_identity", "folder_identity", "lease_file_identity"):
        budget.charge("facts")
        info = value[key]
        file = key == "lease_file_identity"
        _require(isinstance(info, dict) and set(info) == (_FILE_FIELDS if file else _DIRECTORY_FIELDS), code)
        _require(info["type"] == ("regular" if file else "directory")
                 and all(_counter(info[k], maximum=2**64 - 1) for k in ("dev", "ino")), code)
        if file:
            _require(all(_counter(info[k]) for k in _FILE_FIELDS - _DIRECTORY_FIELDS)
                     and info["mode"] <= 0o7777 and info["nlink"] == 1
                     and 0 < info["size_bytes"] <= scratch.MAX_LEASE_BYTES, code)
    _require(_valid_digest(value["lease_raw_sha256"]) and _valid_digest(value["lease_digest"])
             and _counter(value["lease_raw_size_bytes"], positive=True)
             and value["lease_raw_size_bytes"] == value["lease_file_identity"]["size_bytes"], code)
    lease = value["lease"]
    _require(isinstance(lease, dict) and set(lease) == _LEASE_FIELDS, code)
    _require(_valid_id(lease["owner"]) and _valid_id(lease["reference_value"])
             and isinstance(lease["reference_kind"], str) and lease["reference_kind"] in ("run_ref", "scene_ref")
             and isinstance(lease["reason"], str) and scratch._REASON.fullmatch(lease["reason"]) is not None
             and isinstance(lease["class_intent"], str) and lease["class_intent"] in ("cache", "scratch", "evidence")
             and isinstance(lease["cleanup"], str) and lease["cleanup"] in ("delete", "offload", "owner_review")
             and lease["consumer_lifetime_contract"] == scratch.CONSUMER_LIFETIME_PROTOCOL, code)
    created, renewed, expires = (lease[k] for k in ("created_at_epoch", "renewed_at_epoch", "expires_at_epoch"))
    released, size_budget = lease["released_at_epoch"], lease["size_budget_bytes"]
    _require(all(_epoch(v) for v in (created, renewed, expires)) and created <= renewed < expires
             and expires - renewed <= scratch.MAX_TTL_SECONDS
             and (released is None or _epoch(released) and released >= created)
             and (size_budget is None or _counter(size_budget, positive=True))
             and (lease["class_intent"] != "cache" or size_budget is not None), code)
    budget.retain(value)
    return value


def _target_path(expected):
    root = LANE_ROOTS[0 if expected["root"] == "work" else 1]
    return root / expected["lane"] / expected["name"]


def _match_intent(*, expected, selected, policy, consent_issued, consent_expires,
                  expires_at_epoch, now, budget):
    """Called only after the whole protected consent validates against current policy."""
    budget.tick()
    _require(_epoch(now) and _epoch(expires_at_epoch), "owner_target_expiry_invalid")
    decision, lease = selected["decision"], expected["lease"]
    action = decision["action"]
    _require(action in ("keep", "register"), "owner_target_intent_unsupported")
    _require(decision["path"] == str(_target_path(expected)) and decision["owner"] == lease["owner"]
             and lease["owner"] in policy["owners"] and action in policy["allowed_actions"],
             "owner_target_intent_mismatch")
    _require(lease["released_at_epoch"] is None and now < lease["expires_at_epoch"], "owner_target_inactive")
    ceiling = min(consent_expires, now + policy["max_consent_seconds"], lease["expires_at_epoch"])
    if action == "keep":
        ceiling = min(ceiling, decision["expires_at_epoch"])
    else:
        ceiling = min(ceiling, consent_issued + decision["ttl_seconds"])
        for key in ("lane", "name"):
            _require(decision[key] == expected[key], "owner_target_intent_mismatch")
        for key in ("reason", "class_intent", "cleanup", "size_budget_bytes"):
            _require(decision.get(key) == lease[key], "owner_target_intent_mismatch")
        reference = lease["reference_kind"]
        other = "run_ref" if reference == "scene_ref" else "scene_ref"
        _require(decision.get(reference) == lease["reference_value"] and other not in decision,
                 "owner_target_intent_mismatch")
    _require(_epoch(now) and _epoch(expires_at_epoch) and now < expires_at_epoch <= ceiling,
             "owner_target_expiry_invalid")
    approved = lease["size_budget_bytes"] if action == "register" and lease["class_intent"] == "cache" else None
    result = dict(_NO_AUTHORITY, approved_size_budget_bytes=approved,
                  budget_source="protected_register_intent" if approved is not None else "lease_metadata_only",
                  historical_references=list(selected["census_row"]["references"]),
                  kept_reasons=["current_references_unknown", "consumer_participation_unproven"])
    if lease["class_intent"] == "cache" and approved is None:
        result["kept_reasons"].append("cache_budget_owner_approval_missing")
    budget.retain(result)
    return result
