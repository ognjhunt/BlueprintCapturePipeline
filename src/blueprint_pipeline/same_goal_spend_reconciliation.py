"""Materialize lane-local official-billing spend reconciliations.

This is the producer paired with :mod:`blueprint_pipeline.paid_attempt_authority`.
It derives every ledger value from retained terminal, teardown, provider-zero,
and official billing evidence; operators provide paths, never costs or digests.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import (
    SAME_GOAL_ENTRY_SCHEMA,
    SAME_GOAL_RECONCILIATION_SCHEMA,
    SAME_GOAL_RECONCILIATION_STATUS,
    bind_lane_prior_spend,
    validate_same_goal_spend_reconciliation,
)


SUPPORTED_LANES = frozenset(
    {
        "content_agents",
        "gaussian_excision",
        "native_task_arena",
        "retained_scene_render",
        "simready_isaac",
    }
)
_DIGEST_FIELDS = (
    "receipt_digest",
    "result_digest",
    "execution_result_digest",
    "provider_zero_digest",
    "provider_zero_receipt_digest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, dict):
        raise ValueError(code)
    return source, value


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return True


def _source(role: str, path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = _record(path)
    digest_fields = _DIGEST_FIELDS
    if role == "provider_zero":
        digest_fields = (
            "provider_zero_receipt_digest",
            "provider_zero_digest",
            "receipt_digest",
        )
    digest_field = next((field for field in digest_fields if field in value), None)
    source: dict[str, Any] = {
        "role": role,
        "schema_version": value.get("schema_version"),
        "digest_field": digest_field,
        "record": record,
    }
    if digest_field is None:
        source["legacy_digest_gap"] = (
            "exact_source_bytes_sha256_bound_no_canonical_digest"
        )
    else:
        digest = value.get(digest_field)
        if digest != canonical_digest(value, digest_field=digest_field):
            raise ValueError(f"same_goal_spend_source_digest_invalid:{role}")
        record["receipt_digest"] = digest
    return source


def _json_path(value: Mapping[str, Any], *paths: tuple[str, ...]) -> tuple[list[str], Any]:
    for path in paths:
        current: Any = value
        for component in path:
            if not isinstance(current, Mapping) or component not in current:
                break
            current = current[component]
        else:
            return list(path), current
    raise ValueError("same_goal_spend_required_binding_missing")


def _instance_ids(teardown: Mapping[str, Any]) -> list[int]:
    raw = teardown.get("vast_instance_ids")
    if (
        not isinstance(raw, list)
        or not raw
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError("same_goal_spend_teardown_instance_ids_invalid")
    return raw


def _attempt_id(result_path: Path, result: Mapping[str, Any]) -> str:
    for key in ("launch_id", "run_id", "attempt_id"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if result_path.parent.name == "allocator" and result_path.parent.parent.name:
        return result_path.parent.parent.name
    if result_path.parent.name:
        return result_path.parent.name
    raise ValueError("same_goal_spend_attempt_id_unavailable")


def _entry(
    *,
    lane: str,
    terminal_result_path: str | Path,
    teardown_manifest_path: str | Path,
    provider_zero_path: str | Path,
    official_billing_response_path: str | Path,
    provider_billing_source_receipt_path: str | Path,
) -> dict[str, Any]:
    result_path, result = _read(
        terminal_result_path, code="same_goal_spend_terminal_result_invalid"
    )
    teardown_path, teardown = _read(
        teardown_manifest_path, code="same_goal_spend_teardown_invalid"
    )
    zero_path, zero = _read(
        provider_zero_path, code="same_goal_spend_provider_zero_invalid"
    )
    billing_path, billing = _read(
        official_billing_response_path, code="same_goal_spend_billing_response_invalid"
    )
    billing_source_path, billing_source = _read(
        provider_billing_source_receipt_path,
        code="same_goal_spend_billing_source_invalid",
    )
    continuing_path, continuing = _json_path(
        result, ("continuing_spend_from_this_run",), ("continuing_spend",)
    )
    authority_path, authority_digest = _json_path(
        result,
        ("authorization_consumption", "authorization_digest"),
        ("authority_digest",),
    )
    bundle_path, bundle_sha256 = _json_path(result, ("bundle_sha256",))
    zero_binding_path, zero_confirmed = _json_path(
        zero, ("provider_zero_verified",), ("provider_zero_confirmed",)
    )
    estimate_path, estimated_cost = _json_path(result, ("estimated_cost_usd",))
    del estimate_path
    if (
        result.get("status")
        not in {"completed", "blocked", "sealed_completed_attempt", "sealed_blocked_attempt"}
        or continuing is not False
        or zero_confirmed is not True
        or teardown.get("status") not in {"completed", "PASS"}
        or teardown.get("continuing_spend_from_this_run", False) is not False
        or zero.get("continuing_spend_from_this_run", False) is not False
        or isinstance(estimated_cost, bool)
        or not isinstance(estimated_cost, (int, float))
        or not math.isfinite(float(estimated_cost))
        or float(estimated_cost) < 0
        or not _digest(authority_digest)
        or not _digest(bundle_sha256)
    ):
        raise ValueError("same_goal_spend_terminal_or_zero_invalid")

    teardown_ids = _instance_ids(teardown)
    results = billing.get("results")
    if not isinstance(results, list):
        raise ValueError("same_goal_spend_billing_results_invalid")
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, row in enumerate(results):
        if not isinstance(row, Mapping):
            continue
        source = str(row.get("source") or "")
        if not source.startswith("instance-"):
            continue
        try:
            instance_id = int(source.removeprefix("instance-"))
        except ValueError:
            continue
        if instance_id in teardown_ids:
            candidates.append((index, instance_id, row))
    if len(candidates) != 1:
        raise ValueError("same_goal_spend_billing_instance_match_invalid")
    billing_index, instance_id, charge = candidates[0]
    amount = charge.get("amount")
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(float(amount))
        or float(amount) < 0
    ):
        raise ValueError("same_goal_spend_billing_amount_invalid")
    linked = [
        row
        for row in billing_source.get("sources") or []
        if isinstance(row, Mapping)
        and row.get("provider") == "vast"
        and Path(str(row.get("retained_path") or "")).expanduser().resolve() == billing_path
        and row.get("response_digest") == _sha256(billing_path)
        and row.get("response_size_bytes") == billing_path.stat().st_size
    ]
    if billing_source.get("status") != "reconciled" or len(linked) != 1:
        raise ValueError("same_goal_spend_billing_source_unbound")

    sources = [
        _source("terminal_result", result_path, result),
        _source("teardown_manifest", teardown_path, teardown),
        _source("provider_zero", zero_path, zero),
        _source("official_billing_response", billing_path, billing),
        _source("provider_billing_source_receipt", billing_source_path, billing_source),
    ]
    entry: dict[str, Any] = {
        "schema_version": SAME_GOAL_ENTRY_SCHEMA,
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": _attempt_id(result_path, result),
        "lane": lane,
        "evidence_kind": "fully_bound_official_billing",
        "provider_instance_id": instance_id,
        "cost_usd": float(amount),
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_sha256,
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "source_receipts": sources,
        "bindings": [
            {
                "kind": "cost_usd",
                "source_role": "official_billing_response",
                "json_path": ["results", billing_index, "amount"],
                "expected_value": amount,
            },
            {
                "kind": "continuing_spend",
                "source_role": "terminal_result",
                "json_path": continuing_path,
                "expected_value": False,
            },
            {
                "kind": "instance_id",
                "source_role": "official_billing_response",
                "json_path": ["results", billing_index, "source"],
                "expected_value": f"instance-{instance_id}",
            },
            {
                "kind": "authority_digest",
                "source_role": "terminal_result",
                "json_path": authority_path,
                "expected_value": authority_digest,
            },
            {
                "kind": "provider_zero",
                "source_role": "provider_zero",
                "json_path": zero_binding_path,
                "expected_value": True,
            },
            {
                "kind": "bundle_sha256",
                "source_role": "terminal_result",
                "json_path": bundle_path,
                "expected_value": bundle_sha256,
            },
        ],
    }
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    return entry


def materialize_same_goal_spend_reconciliation(
    *,
    lane: str,
    terminal_result_paths: Sequence[str | Path],
    teardown_manifest_paths: Sequence[str | Path],
    provider_zero_paths: Sequence[str | Path],
    official_billing_response_paths: Sequence[str | Path],
    provider_billing_source_receipt_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive and exclusively write one lane-local reconciliation."""

    counts = {
        len(terminal_result_paths),
        len(teardown_manifest_paths),
        len(provider_zero_paths),
        len(official_billing_response_paths),
        len(provider_billing_source_receipt_paths),
    }
    if lane not in SUPPORTED_LANES or counts == {0} or len(counts) != 1:
        raise ValueError("same_goal_spend_materialization_arguments_invalid")
    entries = [
        _entry(
            lane=lane,
            terminal_result_path=result,
            teardown_manifest_path=teardown,
            provider_zero_path=zero,
            official_billing_response_path=billing,
            provider_billing_source_receipt_path=billing_source,
        )
        for result, teardown, zero, billing, billing_source in zip(
            terminal_result_paths,
            teardown_manifest_paths,
            provider_zero_paths,
            official_billing_response_paths,
            provider_billing_source_receipt_paths,
            strict=True,
        )
    ]
    if len({entry["attempt_id"] for entry in entries}) != len(entries):
        raise ValueError("same_goal_spend_attempt_id_duplicate")
    value: dict[str, Any] = {
        "schema_version": SAME_GOAL_RECONCILIATION_SCHEMA,
        "status": SAME_GOAL_RECONCILIATION_STATUS,
        "goal_id": "arm-decision-proof-v1",
        "entries": entries,
        "entry_count": len(entries),
        "total_cost_usd": math.fsum(float(entry["cost_usd"]) for entry in entries),
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError("same_goal_spend_output_exists")
    payload = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        validate_same_goal_spend_reconciliation(temporary)
        bind_lane_prior_spend(
            prior_result_paths=terminal_result_paths,
            reconciliation_path=temporary,
            lane=lane,
        )
        os.link(temporary, destination)
        os.chmod(destination, 0o440)
    finally:
        temporary.unlink(missing_ok=True)
    validate_same_goal_spend_reconciliation(destination)
    bind_lane_prior_spend(
        prior_result_paths=terminal_result_paths,
        reconciliation_path=destination,
        lane=lane,
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(SUPPORTED_LANES), required=True)
    parser.add_argument("--terminal-result", action="append", required=True)
    parser.add_argument("--teardown-manifest", action="append", required=True)
    parser.add_argument("--provider-zero", action="append", required=True)
    parser.add_argument("--official-billing-response", action="append", required=True)
    parser.add_argument(
        "--provider-billing-source-receipt", action="append", required=True
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = materialize_same_goal_spend_reconciliation(
            lane=args.lane,
            terminal_result_paths=args.terminal_result,
            teardown_manifest_paths=args.teardown_manifest,
            provider_zero_paths=args.provider_zero,
            official_billing_response_paths=args.official_billing_response,
            provider_billing_source_receipt_paths=args.provider_billing_source_receipt,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "output": str(Path(args.output).expanduser().resolve()),
                "receipt_digest": value["receipt_digest"],
                "entry_count": value["entry_count"],
                "total_cost_usd": value["total_cost_usd"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
