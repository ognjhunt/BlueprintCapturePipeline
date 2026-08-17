"""Seal exact posted Vast instance charges from retained official billing bytes.

The provider billing reconciler intentionally exports only cohort totals.  This
module reopens its digest-bound raw Vast responses and extracts named instance
charges without making a provider request or accepting an operator-entered
cost.  Reconciliations can be extended from one previously sealed result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .provider_billing_reconciler import (
    BILLING_SOURCE_SCHEMA_VERSION,
    MAX_RESPONSE_BYTES,
    VAST_CHARGES_URL,
)


RECONCILIATION_SCHEMA_VERSION = "blueprint.vast_official_same_goal_reconciliation.v1"
ENTRY_SCHEMA_VERSION = "blueprint.vast_official_instance_charge.v1"
RECONCILIATION_STATUS = "reconciled_official_posted_charges"
GOAL_ID = "arm-decision-proof-v1"
MAX_EXPECTED_INSTANCES = 256
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_LAUNCH_LABEL = re.compile(r"blueprint-[a-z0-9][a-z0-9._-]{1,254}")
_ITEM_NAMES = {
    "gpu": "gpu",
    "disk": "disk",
    "bwd": "bandwidth_download",
    "bwu": "bandwidth_upload",
}


class VastOfficialBillingExtractionError(ValueError):
    """The retained billing evidence was incomplete, ambiguous, or altered."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_file(path: str | Path, *, code: str) -> tuple[Path, bytes]:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
        payload = resolved.read_bytes()
    except OSError as exc:
        raise VastOfficialBillingExtractionError(code) from exc
    if (
        candidate.is_symlink()
        or resolved != absolute
        or not resolved.is_file()
        or metadata.st_size > MAX_RESPONSE_BYTES
    ):
        raise VastOfficialBillingExtractionError(code)
    return resolved, payload


def _json_file(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any], bytes]:
    source, payload = _strict_file(path, code=code)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VastOfficialBillingExtractionError(code) from exc
    if not isinstance(value, dict):
        raise VastOfficialBillingExtractionError(code)
    return source, value, payload


def _record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _prepare_output(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    ancestor = absolute.parent
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise VastOfficialBillingExtractionError("vast_official_output_invalid")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve() != ancestor:
        raise VastOfficialBillingExtractionError("vast_official_output_invalid")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if (
        absolute.parent.resolve() != absolute.parent
        or absolute.exists()
        or absolute.is_symlink()
    ):
        raise VastOfficialBillingExtractionError("vast_official_output_invalid")
    return absolute


def _money(value: Any, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VastOfficialBillingExtractionError(code)
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise VastOfficialBillingExtractionError(code)
    return amount


def _decimal(value: float, *, code: str) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:  # pragma: no cover - guarded by _money
        raise VastOfficialBillingExtractionError(code) from exc


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _expected_instances(
    values: Sequence[tuple[int, str]],
) -> list[tuple[int, str]]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not 1 <= len(values) <= MAX_EXPECTED_INSTANCES
    ):
        raise VastOfficialBillingExtractionError("vast_official_expected_instances_invalid")
    normalized: list[tuple[int, str]] = []
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise VastOfficialBillingExtractionError(
                "vast_official_expected_instances_invalid"
            )
        instance_id, launch_label = value
        if (
            isinstance(instance_id, bool)
            or not isinstance(instance_id, int)
            or instance_id <= 0
            or not isinstance(launch_label, str)
            or _LAUNCH_LABEL.fullmatch(launch_label) is None
        ):
            raise VastOfficialBillingExtractionError(
                "vast_official_expected_instances_invalid"
            )
        normalized.append((instance_id, launch_label))
    if (
        len({instance_id for instance_id, _label in normalized}) != len(normalized)
        or len({_label for _instance_id, _label in normalized}) != len(normalized)
    ):
        raise VastOfficialBillingExtractionError(
            "vast_official_expected_instances_duplicate"
        )
    return sorted(normalized)


def _validate_source_receipt(
    path: str | Path,
) -> tuple[Path, dict[str, Any], bytes]:
    source_path, receipt, source_bytes = _json_file(
        path, code="vast_official_billing_source_receipt_invalid"
    )
    totals = receipt.get("provider_totals_usd")
    if (
        receipt.get("schema_version") != BILLING_SOURCE_SCHEMA_VERSION
        or receipt.get("status") != "reconciled"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("provider_mutation_performed") is not False
        or receipt.get("raw_secret_values_recorded") is not False
        or not isinstance(receipt.get("sources"), list)
        or not isinstance(totals, Mapping)
    ):
        raise VastOfficialBillingExtractionError(
            "vast_official_billing_source_receipt_invalid"
        )
    _money(totals.get("vast"), code="vast_official_billing_source_total_invalid")
    return source_path, receipt, source_bytes


def _load_vast_responses(
    *, source_receipt_path: Path, source_receipt: Mapping[str, Any]
) -> list[tuple[int, Path, bytes, dict[str, Any]]]:
    responses: list[tuple[int, Path, bytes, dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    seen_digests: set[str] = set()
    for source_index, source in enumerate(source_receipt.get("sources") or []):
        if not isinstance(source, Mapping):
            raise VastOfficialBillingExtractionError("vast_official_source_row_invalid")
        if source.get("provider") != "vast":
            continue
        retained_path = source.get("retained_path")
        if not isinstance(retained_path, str) or not Path(retained_path).is_absolute():
            raise VastOfficialBillingExtractionError("vast_official_response_path_invalid")
        response_path, response_bytes = _strict_file(
            retained_path, code="vast_official_response_path_invalid"
        )
        response_digest = source.get("response_digest")
        if (
            response_path.parent != source_receipt_path.parent
            or response_path in seen_paths
            or not _valid_digest(response_digest)
            or response_digest in seen_digests
            or response_digest != _sha256_bytes(response_bytes)
            or isinstance(source.get("response_size_bytes"), bool)
            or source.get("response_size_bytes") != len(response_bytes)
            or source.get("endpoint") != VAST_CHARGES_URL
        ):
            raise VastOfficialBillingExtractionError(
                "vast_official_response_binding_invalid"
            )
        try:
            response = json.loads(response_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VastOfficialBillingExtractionError(
                "vast_official_response_invalid"
            ) from exc
        if (
            not isinstance(response, dict)
            or response.get("success") is not True
            or not isinstance(response.get("results"), list)
        ):
            raise VastOfficialBillingExtractionError("vast_official_response_invalid")
        seen_paths.add(response_path)
        seen_digests.add(str(response_digest))
        responses.append((source_index, response_path, response_bytes, response))
    if not responses:
        raise VastOfficialBillingExtractionError("vast_official_response_missing")
    return responses


def _line_items(row: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    items = row.get("items")
    if not isinstance(items, list) or len(items) != len(_ITEM_NAMES):
        raise VastOfficialBillingExtractionError("vast_official_charge_items_invalid")
    amounts: dict[str, float] = {}
    for item in items:
        if not isinstance(item, Mapping) or item.get("type") not in _ITEM_NAMES:
            raise VastOfficialBillingExtractionError("vast_official_charge_items_invalid")
        normalized = _ITEM_NAMES[str(item["type"])]
        if (
            normalized in amounts
            or item.get("source") not in {None, ""}
            or item.get("items") not in (None, [])
        ):
            raise VastOfficialBillingExtractionError("vast_official_charge_items_invalid")
        amounts[normalized] = _money(
            item.get("amount"), code="vast_official_charge_item_amount_invalid"
        )
    if set(amounts) != set(_ITEM_NAMES.values()):
        raise VastOfficialBillingExtractionError("vast_official_charge_items_invalid")
    bandwidth = float(
        _decimal(amounts["bandwidth_download"], code="vast_official_charge_items_invalid")
        + _decimal(amounts["bandwidth_upload"], code="vast_official_charge_items_invalid")
    )
    return amounts, bandwidth


def _entry(
    *,
    instance_id: int,
    launch_label: str,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    source_receipt_bytes: bytes,
    source_index: int,
    response_path: Path,
    response_bytes: bytes,
    result_index: int,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = row.get("metadata")
    if (
        row.get("source") != f"instance-{instance_id}"
        or row.get("type") != "instance"
        or not isinstance(metadata, Mapping)
        or metadata.get("label") != launch_label
    ):
        raise VastOfficialBillingExtractionError("vast_official_charge_identity_invalid")
    amount = _money(row.get("amount"), code="vast_official_charge_amount_invalid")
    items, bandwidth = _line_items(row)
    item_total = sum(
        (_decimal(value, code="vast_official_charge_items_invalid") for value in items.values()),
        Decimal("0"),
    )
    if item_total != _decimal(amount, code="vast_official_charge_amount_invalid"):
        raise VastOfficialBillingExtractionError("vast_official_charge_total_contradiction")
    source_receipt_record = _record(source_receipt_path, source_receipt_bytes)
    source_receipt_record["receipt_digest"] = source_receipt["receipt_digest"]
    response_record = _record(response_path, response_bytes)
    response_record["source_index"] = source_index
    response_record["result_index"] = result_index
    entry: dict[str, Any] = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "provider": "vast",
        "currency": "USD",
        "evidence_kind": "official_provider_charge",
        "official_charge_posted": True,
        "provider_instance_id": instance_id,
        "launch_label": launch_label,
        "official_charge_usd": amount,
        "official_line_items_usd": items,
        "bandwidth_total_usd": bandwidth,
        "line_item_total_usd": float(item_total),
        "provider_billing_source_receipt": source_receipt_record,
        "official_billing_response": response_record,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "entry_digest": "",
    }
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    return entry


def _validate_entry(entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    items = entry.get("official_line_items_usd")
    expected_id = entry.get("provider_instance_id")
    if (
        isinstance(expected_id, bool)
        or not isinstance(expected_id, int)
        or expected_id <= 0
        or entry.get("schema_version") != ENTRY_SCHEMA_VERSION
        or entry.get("provider") != "vast"
        or entry.get("currency") != "USD"
        or entry.get("evidence_kind") != "official_provider_charge"
        or entry.get("official_charge_posted") is not True
        or not isinstance(entry.get("launch_label"), str)
        or _LAUNCH_LABEL.fullmatch(entry["launch_label"]) is None
        or not isinstance(items, Mapping)
        or set(items) != set(_ITEM_NAMES.values())
        or entry.get("provider_mutation_performed") is not False
        or entry.get("raw_secret_values_recorded") is not False
        or entry.get("entry_digest")
        != canonical_digest(entry, digest_field="entry_digest")
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    amount = _money(
        entry.get("official_charge_usd"), code="vast_official_prior_entry_invalid"
    )
    normalized_items = {
        key: _money(value, code="vast_official_prior_entry_invalid")
        for key, value in items.items()
    }
    item_total = sum((Decimal(str(value)) for value in normalized_items.values()), Decimal("0"))
    if (
        item_total != Decimal(str(amount))
        or _money(
            entry.get("line_item_total_usd"), code="vast_official_prior_entry_invalid"
        )
        != amount
        or Decimal(
            str(
                _money(
                    entry.get("bandwidth_total_usd"),
                    code="vast_official_prior_entry_invalid",
                )
            )
        )
        != Decimal(str(normalized_items["bandwidth_download"]))
        + Decimal(str(normalized_items["bandwidth_upload"]))
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    for role in ("provider_billing_source_receipt", "official_billing_response"):
        record = entry.get(role)
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] <= 0
            or not _valid_digest(record.get("sha256"))
        ):
            raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    source_record = entry["provider_billing_source_receipt"]
    response_record = entry["official_billing_response"]
    if (
        not _valid_digest(source_record.get("receipt_digest"))
        or isinstance(response_record.get("source_index"), bool)
        or not isinstance(response_record.get("source_index"), int)
        or response_record["source_index"] < 0
        or isinstance(response_record.get("result_index"), bool)
        or not isinstance(response_record.get("result_index"), int)
        or response_record["result_index"] < 0
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    source_path, source_receipt, source_bytes = _validate_source_receipt(
        source_record["path"]
    )
    if (
        _record(source_path, source_bytes)
        | {"receipt_digest": source_receipt["receipt_digest"]}
        != dict(source_record)
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    response_path, response_bytes = _strict_file(
        response_record["path"], code="vast_official_prior_entry_invalid"
    )
    if _record(response_path, response_bytes) != {
        key: response_record[key] for key in ("path", "size_bytes", "sha256")
    }:
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    source_index = response_record["source_index"]
    result_index = response_record["result_index"]
    sources = source_receipt.get("sources") or []
    if source_index >= len(sources) or not isinstance(sources[source_index], Mapping):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    linked = sources[source_index]
    if (
        linked.get("provider") != "vast"
        or linked.get("endpoint") != VAST_CHARGES_URL
        or linked.get("retained_path") != str(response_path)
        or linked.get("response_size_bytes") != len(response_bytes)
        or linked.get("response_digest") != _sha256_bytes(response_bytes)
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    try:
        response = json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid") from exc
    results = response.get("results") if isinstance(response, Mapping) else None
    if (
        not isinstance(results, list)
        or result_index >= len(results)
        or not isinstance(results[result_index], Mapping)
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")
    row = results[result_index]
    metadata = row.get("metadata")
    source_items, source_bandwidth = _line_items(row)
    if (
        row.get("source") != f"instance-{expected_id}"
        or row.get("type") != "instance"
        or not isinstance(metadata, Mapping)
        or metadata.get("label") != entry["launch_label"]
        or _money(row.get("amount"), code="vast_official_prior_entry_invalid")
        != amount
        or source_items != normalized_items
        or source_bandwidth
        != _money(
            entry.get("bandwidth_total_usd"), code="vast_official_prior_entry_invalid"
        )
    ):
        raise VastOfficialBillingExtractionError("vast_official_prior_entry_invalid")


def _validate_reconciliation_value(value: Mapping[str, Any]) -> None:
    entries = value.get("entries")
    if (
        value.get("schema_version") != RECONCILIATION_SCHEMA_VERSION
        or value.get("status") != RECONCILIATION_STATUS
        or value.get("goal_id") != GOAL_ID
        or value.get("provider") != "vast"
        or value.get("currency") != "USD"
        or not isinstance(entries, list)
        or isinstance(value.get("entry_count"), bool)
        or not isinstance(value.get("entry_count"), int)
        or value.get("entry_count") != len(entries)
        or isinstance(value.get("new_entry_count"), bool)
        or not isinstance(value.get("new_entry_count"), int)
        or value.get("new_entry_count") <= 0
        or isinstance(value.get("prior_entry_count"), bool)
        or not isinstance(value.get("prior_entry_count"), int)
        or value.get("prior_entry_count") < 0
        or value.get("new_entry_count") + value.get("prior_entry_count")
        != len(entries)
        or value.get("provider_mutation_performed") is not False
        or value.get("paid_resource_allocated") is not False
        or value.get("raw_secret_values_recorded") is not False
        or value.get("blockers") != []
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
    ):
        raise VastOfficialBillingExtractionError(
            "vast_official_reconciliation_invalid"
        )
    for entry in entries:
        _validate_entry(entry)
    current_source = value.get("current_provider_billing_source_receipt")
    if (
        not isinstance(current_source, Mapping)
        or not isinstance(current_source.get("path"), str)
        or isinstance(current_source.get("size_bytes"), bool)
        or not isinstance(current_source.get("size_bytes"), int)
        or current_source["size_bytes"] <= 0
        or not _valid_digest(current_source.get("sha256"))
        or not _valid_digest(current_source.get("receipt_digest"))
        or sum(
            entry.get("provider_billing_source_receipt") == current_source
            for entry in entries
        )
        < value["new_entry_count"]
    ):
        raise VastOfficialBillingExtractionError(
            "vast_official_reconciliation_source_binding_invalid"
        )
    predecessor = value.get("predecessor_reconciliation")
    prior_entry_count = value["prior_entry_count"]
    if (
        (prior_entry_count == 0 and predecessor is not None)
        or (
            prior_entry_count > 0
            and (
                not isinstance(predecessor, Mapping)
                or not isinstance(predecessor.get("path"), str)
                or isinstance(predecessor.get("size_bytes"), bool)
                or not isinstance(predecessor.get("size_bytes"), int)
                or predecessor["size_bytes"] <= 0
                or not _valid_digest(predecessor.get("sha256"))
                or not _valid_digest(predecessor.get("receipt_digest"))
            )
        )
    ):
        raise VastOfficialBillingExtractionError(
            "vast_official_reconciliation_predecessor_invalid"
        )
    if prior_entry_count > 0:
        predecessor_path, predecessor_value, predecessor_bytes = _json_file(
            predecessor["path"],
            code="vast_official_reconciliation_predecessor_invalid",
        )
        if (
            _record(predecessor_path, predecessor_bytes)
            | {"receipt_digest": predecessor_value.get("receipt_digest")}
            != dict(predecessor)
            or predecessor_value.get("schema_version")
            != RECONCILIATION_SCHEMA_VERSION
            or predecessor_value.get("receipt_digest")
            != canonical_digest(predecessor_value, digest_field="receipt_digest")
            or not isinstance(predecessor_value.get("entries"), list)
            or len(predecessor_value["entries"]) != prior_entry_count
            or len(
                {
                    entry.get("entry_digest")
                    for entry in predecessor_value["entries"]
                    if isinstance(entry, Mapping)
                }
            )
            != prior_entry_count
            or not {
                entry.get("entry_digest")
                for entry in predecessor_value["entries"]
                if isinstance(entry, Mapping)
            }.issubset({entry["entry_digest"] for entry in entries})
        ):
            raise VastOfficialBillingExtractionError(
                "vast_official_reconciliation_predecessor_invalid"
            )
    ids = [entry["provider_instance_id"] for entry in entries]
    labels = [entry["launch_label"] for entry in entries]
    if (
        ids != sorted(ids)
        or len(ids) != len(set(ids))
        or len(labels) != len(set(labels))
        or value.get("provider_instance_ids") != ids
        or value.get("launch_labels") != labels
    ):
        raise VastOfficialBillingExtractionError(
            "vast_official_reconciliation_identity_invalid"
        )
    official_total = _money(
        value.get("official_total_usd"), code="vast_official_reconciliation_total_invalid"
    )
    expected_total = sum(
        (Decimal(str(entry["official_charge_usd"])) for entry in entries), Decimal("0")
    )
    if Decimal(str(official_total)) != expected_total:
        raise VastOfficialBillingExtractionError(
            "vast_official_reconciliation_total_invalid"
        )


def validate_vast_official_same_goal_reconciliation(
    path: str | Path,
) -> dict[str, Any]:
    """Reopen and validate one sealed reconciliation without provider access."""

    _source, value, _payload = _json_file(
        path, code="vast_official_reconciliation_invalid"
    )
    _validate_reconciliation_value(value)
    return value


def materialize_vast_official_same_goal_reconciliation(
    *,
    provider_billing_source_receipt_path: str | Path,
    expected_instances: Sequence[tuple[int, str]],
    output_path: str | Path,
    prior_reconciliation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract official rows and exclusively seal one cumulative reconciliation."""

    expected = _expected_instances(expected_instances)
    source_path, source_receipt, source_bytes = _validate_source_receipt(
        provider_billing_source_receipt_path
    )
    responses = _load_vast_responses(
        source_receipt_path=source_path, source_receipt=source_receipt
    )
    candidates: dict[int, list[tuple[int, Path, bytes, int, Mapping[str, Any]]]] = {
        instance_id: [] for instance_id, _label in expected
    }
    expected_by_id = dict(expected)
    expected_by_label = {label: instance_id for instance_id, label in expected}
    for source_index, response_path, response_bytes, response in responses:
        for result_index, row in enumerate(response["results"]):
            if not isinstance(row, Mapping):
                raise VastOfficialBillingExtractionError("vast_official_charge_row_invalid")
            source = str(row.get("source") or "")
            metadata = row.get("metadata")
            label = metadata.get("label") if isinstance(metadata, Mapping) else None
            source_id: int | None = None
            if source.startswith("instance-") and source.removeprefix("instance-").isdigit():
                source_id = int(source.removeprefix("instance-"))
            related_ids = set()
            if source_id in expected_by_id:
                related_ids.add(source_id)
            if label in expected_by_label:
                related_ids.add(expected_by_label[str(label)])
            for instance_id in related_ids:
                candidates[instance_id].append(
                    (source_index, response_path, response_bytes, result_index, row)
                )
    entries: list[dict[str, Any]] = []
    for instance_id, launch_label in expected:
        matches = candidates[instance_id]
        if not matches:
            raise VastOfficialBillingExtractionError("vast_official_charge_unposted")
        if len(matches) != 1:
            raise VastOfficialBillingExtractionError("vast_official_charge_duplicate")
        source_index, response_path, response_bytes, result_index, row = matches[0]
        entries.append(
            _entry(
                instance_id=instance_id,
                launch_label=launch_label,
                source_receipt_path=source_path,
                source_receipt=source_receipt,
                source_receipt_bytes=source_bytes,
                source_index=source_index,
                response_path=response_path,
                response_bytes=response_bytes,
                result_index=result_index,
                row=row,
            )
        )

    predecessor: dict[str, Any] | None = None
    prior_entries: list[dict[str, Any]] = []
    if prior_reconciliation_path is not None:
        prior_path, prior, prior_bytes = _json_file(
            prior_reconciliation_path, code="vast_official_prior_reconciliation_invalid"
        )
        _validate_reconciliation_value(prior)
        prior_entries = [dict(entry) for entry in prior["entries"]]
        predecessor = _record(prior_path, prior_bytes)
        predecessor["receipt_digest"] = prior["receipt_digest"]
    combined = sorted(
        [*prior_entries, *entries], key=lambda entry: entry["provider_instance_id"]
    )
    ids = [entry["provider_instance_id"] for entry in combined]
    labels = [entry["launch_label"] for entry in combined]
    if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise VastOfficialBillingExtractionError("vast_official_prior_overlap")
    official_total = sum(
        (Decimal(str(entry["official_charge_usd"])) for entry in combined), Decimal("0")
    )
    current_source_record = _record(source_path, source_bytes)
    current_source_record["receipt_digest"] = source_receipt["receipt_digest"]
    value: dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "status": RECONCILIATION_STATUS,
        "goal_id": GOAL_ID,
        "provider": "vast",
        "currency": "USD",
        "entries": combined,
        "entry_count": len(combined),
        "new_entry_count": len(entries),
        "prior_entry_count": len(prior_entries),
        "provider_instance_ids": ids,
        "launch_labels": labels,
        "official_total_usd": float(official_total),
        "current_provider_billing_source_receipt": current_source_record,
        "predecessor_reconciliation": predecessor,
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    _validate_reconciliation_value(value)

    destination = _prepare_output(output_path)
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o440)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        validate_vast_official_same_goal_reconciliation(temporary)
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise VastOfficialBillingExtractionError("vast_official_output_invalid") from exc
    finally:
        temporary.unlink(missing_ok=True)
    validate_vast_official_same_goal_reconciliation(destination)
    return value


def _parse_expected(value: str) -> tuple[int, str]:
    instance, separator, label = value.partition("=")
    if not separator or not instance.isdigit():
        raise argparse.ArgumentTypeError("expected INSTANCE_ID=EXACT_LAUNCH_LABEL")
    return int(instance), label


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-billing-source-receipt", required=True)
    parser.add_argument(
        "--expected-instance",
        action="append",
        type=_parse_expected,
        required=True,
        metavar="INSTANCE_ID=EXACT_LAUNCH_LABEL",
    )
    parser.add_argument("--prior-reconciliation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = materialize_vast_official_same_goal_reconciliation(
            provider_billing_source_receipt_path=args.provider_billing_source_receipt,
            expected_instances=args.expected_instance,
            output_path=args.output,
            prior_reconciliation_path=args.prior_reconciliation,
        )
    except (OSError, VastOfficialBillingExtractionError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "output": str(Path(args.output).expanduser().resolve()),
                "receipt_digest": value["receipt_digest"],
                "entry_count": value["entry_count"],
                "official_total_usd": value["official_total_usd"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ENTRY_SCHEMA_VERSION",
    "RECONCILIATION_SCHEMA_VERSION",
    "VastOfficialBillingExtractionError",
    "main",
    "materialize_vast_official_same_goal_reconciliation",
    "validate_vast_official_same_goal_reconciliation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
