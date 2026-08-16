"""Shared fail-closed bindings for single-use paid-attempt authorities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


ALLOWLIST_GROUPS = ("external_provider_owned", "same_goal_concurrent")
SAME_GOAL_RECONCILIATION_SCHEMA = "adp_same_goal_spend_reconciliation.v1"
SAME_GOAL_ENTRY_SCHEMA = "adp_same_goal_spend_entry.v1"
SAME_GOAL_RECONCILIATION_STATUS = (
    "all_same_goal_paid_attempts_terminal_and_provider_zero"
)
_DIGEST_FIELDS = (
    "receipt_digest",
    "result_digest",
    "execution_result_digest",
    "execution_digest",
    "provider_zero_digest",
    "provider_zero_receipt_digest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_record(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _json_path(value: Any, path: Any) -> Any:
    if not isinstance(path, list) or not path:
        raise ValueError("same_goal_spend_binding_path_invalid")
    current = value
    for component in path:
        if isinstance(component, str) and isinstance(current, Mapping):
            if component not in current:
                raise ValueError("same_goal_spend_binding_path_invalid")
            current = current[component]
        elif (
            isinstance(component, int)
            and not isinstance(component, bool)
            and isinstance(current, list)
            and 0 <= component < len(current)
        ):
            current = current[component]
        else:
            raise ValueError("same_goal_spend_binding_path_invalid")
    return current


def validate_same_goal_spend_reconciliation(
    path: str | Path, *, expected_total_cost_usd: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen the canonical same-goal ledger and every bound source receipt."""

    source = Path(path).expanduser().resolve()
    value = _read_json(source, code="same_goal_spend_reconciliation_invalid")
    entries = value.get("entries")
    total = value.get("total_cost_usd")
    if (
        value.get("schema_version") != SAME_GOAL_RECONCILIATION_SCHEMA
        or value.get("status") != SAME_GOAL_RECONCILIATION_STATUS
        or value.get("goal_id") != "arm-decision-proof-v1"
        or not isinstance(entries, list)
        or not entries
        or value.get("entry_count") != len(entries)
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not _finite(total)
        or (
            expected_total_cost_usd is not None
            and float(total) != float(expected_total_cost_usd)
        )
    ):
        raise ValueError("same_goal_spend_reconciliation_invalid")
    costs: list[float] = []
    attempt_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("same_goal_spend_entry_invalid")
        attempt_id = str(entry.get("attempt_id") or "")
        cost = entry.get("cost_usd")
        if (
            entry.get("schema_version") != SAME_GOAL_ENTRY_SCHEMA
            or entry.get("goal_id") != "arm-decision-proof-v1"
            or not attempt_id
            or attempt_id in attempt_ids
            or not str(entry.get("lane") or "").strip()
            or entry.get("evidence_kind") != "fully_bound_official_billing"
            or not _finite(cost)
            or entry.get("continuing_spend_from_this_run") is not False
            or entry.get("provider_zero_confirmed") is not True
            or entry.get("entry_digest")
            != canonical_digest(entry, digest_field="entry_digest")
        ):
            raise ValueError("same_goal_spend_entry_invalid")
        attempt_ids.add(attempt_id)
        sources = entry.get("source_receipts")
        bindings = entry.get("bindings")
        if not isinstance(sources, list) or not sources or not isinstance(bindings, list):
            raise ValueError("same_goal_spend_entry_invalid")
        reopened: dict[str, dict[str, Any]] = {}
        for source_record in sources:
            if not isinstance(source_record, Mapping):
                raise ValueError("same_goal_spend_source_invalid")
            role = str(source_record.get("role") or "")
            source_path = _bound_record(
                source_record.get("record"), code="same_goal_spend_source_unbound"
            )
            source_value = _read_json(
                source_path, code="same_goal_spend_source_invalid"
            )
            digest_field = source_record.get("digest_field")
            if (
                not role
                or role in reopened
                or source_value.get("schema_version")
                != source_record.get("schema_version")
            ):
                raise ValueError("same_goal_spend_source_invalid")
            if digest_field is None:
                if source_record.get("legacy_digest_gap") != (
                    "exact_source_bytes_sha256_bound_no_canonical_digest"
                ):
                    raise ValueError("same_goal_spend_source_invalid")
            elif digest_field not in _DIGEST_FIELDS:
                raise ValueError("same_goal_spend_digest_field_invalid")
            else:
                digest = source_value.get(str(digest_field))
                if (
                    digest
                    != canonical_digest(source_value, digest_field=str(digest_field))
                    or digest
                    != (source_record.get("record") or {}).get("receipt_digest")
                ):
                    raise ValueError("same_goal_spend_source_invalid")
            reopened[role] = source_value
        observed_kinds: set[str] = set()
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("same_goal_spend_binding_invalid")
            kind = str(binding.get("kind") or "")
            source_value = reopened.get(str(binding.get("source_role") or ""))
            if source_value is None or kind in observed_kinds:
                raise ValueError("same_goal_spend_binding_invalid")
            observed = _json_path(source_value, binding.get("json_path"))
            if observed != binding.get("expected_value"):
                raise ValueError("same_goal_spend_binding_invalid")
            observed_kinds.add(kind)
            if kind == "cost_usd" and (
                not _finite(observed) or float(observed) != float(cost)
            ):
                raise ValueError("same_goal_spend_binding_invalid")
            if kind == "provider_zero" and observed not in {
                True,
                "PASS",
                "provider_zero",
                "provider_zero_confirmed",
            }:
                raise ValueError("same_goal_spend_binding_invalid")
            if kind == "continuing_spend" and observed is not False:
                raise ValueError("same_goal_spend_binding_invalid")
            if kind == "authority_digest" and observed != entry.get(
                "authority_digest"
            ):
                raise ValueError("same_goal_spend_binding_invalid")
            if kind == "bundle_sha256" and observed != entry.get("bundle_sha256"):
                raise ValueError("same_goal_spend_binding_invalid")
        required = {
            "cost_usd",
            "provider_zero",
            "continuing_spend",
            "instance_id",
            "authority_digest",
            "bundle_sha256",
        }
        if observed_kinds != required:
            raise ValueError("same_goal_spend_binding_incomplete")
        costs.append(float(cost))
    if not math.isclose(math.fsum(costs), float(total), rel_tol=0, abs_tol=1e-9):
        raise ValueError("same_goal_spend_total_invalid")
    return value, {
        **_record(source),
        "receipt_digest": value["receipt_digest"],
        "entry_count": len(entries),
        "total_cost_usd": float(total),
    }


def bind_lane_prior_spend(
    *,
    prior_result_paths: Sequence[str | Path],
    reconciliation_path: str | Path | None,
    lane: str,
) -> dict[str, Any]:
    """Bind one lane-local ledger's posted charges to exact terminal results.

    Mixed-lane or campaign-wide reconciliations are intentionally rejected.
    Each paid lane must rematerialize its own canonical reconciliation so its
    authority cannot inherit unrelated costs or evidence by implication.
    """

    result_paths = [Path(item).expanduser().resolve() for item in prior_result_paths]
    if not result_paths:
        if reconciliation_path is not None:
            raise ValueError("prior_spend_reconciliation_without_prior_attempt")
        return {"prior_terminal_attempts": [], "reconciliation": None, "actual_total_usd": 0.0}
    if any(path.is_symlink() or not path.is_file() for path in result_paths):
        raise ValueError("prior_terminal_attempt_missing")
    if reconciliation_path is None:
        raise ValueError("prior_spend_reconciliation_required")
    reconciliation, record = validate_same_goal_spend_reconciliation(
        reconciliation_path
    )
    entries = reconciliation["entries"]
    if len(entries) != len(result_paths) or any(
        entry.get("lane") != lane for entry in entries
    ):
        raise ValueError("prior_spend_reconciliation_lane_or_count_mismatch")
    rows: list[dict[str, Any]] = []
    used_entries: set[str] = set()
    required_roles = {
        "terminal_result",
        "teardown_manifest",
        "provider_zero",
        "official_billing_response",
        "provider_billing_source_receipt",
    }
    for result_path in result_paths:
        result = _read_json(result_path, code="prior_terminal_attempt_invalid")
        if (
            result.get("status") not in {
                "completed",
                "blocked",
                "sealed_completed_attempt",
                "sealed_blocked_attempt",
            }
            or result.get("continuing_spend_from_this_run", result.get("continuing_spend"))
            is not False
        ):
            raise ValueError("prior_terminal_attempt_invalid")
        result_sha = _sha256(result_path)
        matches = []
        for entry in entries:
            sources = {
                str(source.get("role") or ""): source
                for source in entry.get("source_receipts") or []
                if isinstance(source, Mapping)
            }
            terminal = sources.get("terminal_result")
            terminal_record = terminal.get("record") if isinstance(terminal, Mapping) else None
            if (
                isinstance(terminal_record, Mapping)
                and terminal_record.get("sha256") == result_sha
                and Path(str(terminal_record.get("path") or "")).expanduser().resolve()
                == result_path
            ):
                matches.append((entry, sources))
        if len(matches) != 1:
            raise ValueError("prior_terminal_attempt_reconciliation_match_invalid")
        entry, sources = matches[0]
        attempt_id = str(entry["attempt_id"])
        if attempt_id in used_entries or not required_roles.issubset(sources):
            raise ValueError("prior_terminal_attempt_reconciliation_sources_incomplete")
        used_entries.add(attempt_id)
        teardown_path = _bound_record(
            sources["teardown_manifest"]["record"], code="prior_teardown_unbound"
        )
        zero_path = _bound_record(
            sources["provider_zero"]["record"], code="prior_provider_zero_unbound"
        )
        billing_path = _bound_record(
            sources["official_billing_response"]["record"],
            code="prior_billing_response_unbound",
        )
        billing_source_path = _bound_record(
            sources["provider_billing_source_receipt"]["record"],
            code="prior_billing_source_receipt_unbound",
        )
        teardown = _read_json(teardown_path, code="prior_teardown_invalid")
        zero = _read_json(zero_path, code="prior_provider_zero_invalid")
        billing = _read_json(billing_path, code="prior_billing_response_invalid")
        billing_source = _read_json(
            billing_source_path, code="prior_billing_source_receipt_invalid"
        )
        instance_id = entry.get("provider_instance_id")
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("prior_provider_instance_id_invalid")
        linked_sources = [
            row
            for row in billing_source.get("sources") or []
            if isinstance(row, Mapping)
            and row.get("provider") == "vast"
            and row.get("response_digest") == _sha256(billing_path)
            and row.get("response_size_bytes") == billing_path.stat().st_size
            and Path(str(row.get("retained_path") or "")).expanduser().resolve()
            == billing_path
        ]
        charge_rows = [
            row
            for row in billing.get("results") or []
            if isinstance(row, Mapping)
            and row.get("source") == f"instance-{instance_id}"
        ]
        teardown_instance_ids = teardown.get("vast_instance_ids")
        if teardown_instance_ids is None:
            single_instance_id = teardown.get("instance_id")
            if (
                isinstance(single_instance_id, int)
                and not isinstance(single_instance_id, bool)
                and single_instance_id > 0
            ):
                teardown_instance_ids = [single_instance_id]
            elif (
                isinstance(single_instance_id, str)
                and single_instance_id.isdigit()
                and int(single_instance_id) > 0
            ):
                teardown_instance_ids = [int(single_instance_id)]
        if (
            len(linked_sources) != 1
            or len(charge_rows) != 1
            or billing_source.get("status") != "reconciled"
            or teardown.get("status") not in {"completed", "PASS"}
            or teardown.get("continuing_spend_from_this_run", False) is not False
            or not isinstance(teardown_instance_ids, list)
            or instance_id not in teardown_instance_ids
            or zero.get(
                "provider_zero_verified",
                zero.get(
                    "provider_zero_confirmed",
                    zero.get("provider_zero_api_confirmed"),
                ),
            )
            is not True
            or zero.get("continuing_spend_from_this_run", False) is not False
            or float(charge_rows[0].get("amount", -1)) != float(entry["cost_usd"])
        ):
            raise ValueError("prior_terminal_billing_or_zero_invalid")
        estimate = result.get("estimated_cost_usd", result.get("cost_usd"))
        if not _finite(estimate):
            raise ValueError("prior_terminal_attempt_estimate_invalid")
        rows.append(
            {
                # Preserve the v1 flat identity while adding the strict bound
                # record used by all new validators.
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "receipt_digest": result.get("receipt_digest"),
                "result": {
                    **_record(result_path),
                    "receipt_digest": result.get("receipt_digest"),
                },
                "estimated_cost_usd": float(estimate),
                "actual_provider_charge_usd": float(entry["cost_usd"]),
                "provider_instance_id": instance_id,
                "reconciliation_entry_digest": entry["entry_digest"],
            }
        )
    return {
        "prior_terminal_attempts": rows,
        "reconciliation": record,
        "actual_total_usd": float(reconciliation["total_cost_usd"]),
    }


def validate_bound_lane_prior_spend(
    authority: Mapping[str, Any], *, lane: str
) -> dict[str, Any]:
    """Reopen and compare the canonical prior-spend binding in an authority."""

    attempts = authority.get("prior_terminal_attempts", [])
    reconciliation = authority.get("prior_spend_reconciliation")
    actual_total = authority.get("prior_actual_provider_spend_usd", 0.0)
    if not isinstance(attempts, list):
        raise ValueError("prior_terminal_attempts_invalid")
    if not attempts:
        if reconciliation is not None or actual_total != 0.0:
            raise ValueError("prior_spend_reconciliation_without_prior_attempt")
        return {
            "prior_terminal_attempts": [],
            "reconciliation": None,
            "actual_total_usd": 0.0,
        }
    if not isinstance(reconciliation, Mapping):
        raise ValueError("prior_spend_reconciliation_required")
    observed = bind_lane_prior_spend(
        prior_result_paths=[
            str((row.get("result") or {}).get("path") or "")
            for row in attempts
            if isinstance(row, Mapping)
        ],
        reconciliation_path=str(reconciliation.get("path") or ""),
        lane=lane,
    )
    if (
        observed["prior_terminal_attempts"] != attempts
        or observed["reconciliation"] != reconciliation
        or observed["actual_total_usd"] != actual_total
    ):
        raise ValueError("prior_spend_reconciliation_record_mismatch")
    return observed


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
    "SAME_GOAL_ENTRY_SCHEMA",
    "SAME_GOAL_RECONCILIATION_SCHEMA",
    "SAME_GOAL_RECONCILIATION_STATUS",
    "active_instance_allowlist_metadata_error",
    "bind_lane_prior_spend",
    "flatten_active_instance_allowlist",
    "normalize_active_instance_allowlist",
    "normalize_external_instance_allowlist",
    "validate_same_goal_spend_reconciliation",
    "validate_bound_lane_prior_spend",
]
