"""Conservatively roll a sealed project-spend baseline forward.

A brand-new paid lane has no terminal predecessor, but it still shares the
Arm Decision Proof project ceiling.  Lane-local billing reconciliations cannot
produce that project total by themselves.  This module binds one previously
sealed project baseline, every posted attempt after that baseline, and the full
caps of any admitted-but-unposted authorities into one reopenable receipt.

The producer performs no provider mutation.  Its opening total comes either
from a prior paid-attempt authority or from a digest-bound human authorization
that explicitly adopts a conservative project exposure; every later increment
is derived from digest-bound source bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import validate_same_goal_spend_reconciliation


SCHEMA_VERSION = "adp_project_spend_reconciliation.v1"
STATUS = "project_spend_conservatively_reconciled"
HUMAN_BASELINE_SCHEMA_VERSION = "blueprint_project_spend_human_authorization.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Seal one new path without any check-then-replace overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o000)
    except FileExistsError as exc:
        raise ValueError("project_spend_output_exists") from exc
    reserved = os.fstat(descriptor)
    try:
        payload = json.dumps(dict(value), indent=2).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            observed = path.stat(follow_symlinks=False)
            if observed.st_dev == reserved.st_dev and observed.st_ino == reserved.st_ino:
                path.unlink()
        except FileNotFoundError:
            pass
        raise


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, dict):
        raise ValueError(code)
    return source, value


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _positive_gpu_hours(description: Any) -> float | None:
    """Read Vast's retained GPU-hours description without trusting charge rounding."""

    words = str(description or "").strip().split()
    if len(words) < 2 or words[1] not in {"hour", "hours"}:
        return None
    try:
        hours = float(words[0])
    except ValueError:
        return None
    return hours if math.isfinite(hours) and hours > 0 else None


def _official_billing_maturity(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Prove a posted Vast row includes nonzero GPU time for an allocated run.

    Vast can publish a disk-only row before its GPU charge matures.  Such a row
    is official provider output, but it is not yet a safe replacement for the
    authority's full-cap reserve.  Project-wide reconciliation therefore
    accepts a ``fully_bound_official_billing`` entry only after the exact bound
    response contains a matching instance row with positive GPU hours.
    """

    if entry.get("evidence_kind") != "fully_bound_official_billing":
        return None
    instance_id = entry.get("provider_instance_id")
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("project_spend_official_billing_immature")
    sources = entry.get("source_receipts")
    if not isinstance(sources, list):
        raise ValueError("project_spend_official_billing_immature")
    billing_record: Mapping[str, Any] | None = None
    for source in sources:
        if not isinstance(source, Mapping) or source.get("role") != "official_billing_response":
            continue
        record = source.get("record")
        if billing_record is not None or not isinstance(record, Mapping):
            raise ValueError("project_spend_official_billing_immature")
        billing_record = record
    if billing_record is None:
        raise ValueError("project_spend_official_billing_immature")
    billing_path = str(billing_record.get("path") or "")
    source_path, billing = _read(
        billing_path, code="project_spend_official_billing_immature"
    )
    if (
        billing_record.get("path") != str(source_path)
        or billing_record.get("size_bytes") != source_path.stat().st_size
        or billing_record.get("sha256") != _sha256(source_path)
    ):
        raise ValueError("project_spend_official_billing_immature")
    expected_source = f"instance-{instance_id}"
    matches = [
        row
        for row in billing.get("results") or []
        if isinstance(row, Mapping) and row.get("source") == expected_source
    ]
    observed_amount = matches[0].get("amount") if len(matches) == 1 else None
    entry_cost = entry.get("cost_usd")
    if (
        len(matches) != 1
        or not _finite(observed_amount)
        or not _finite(entry_cost)
        or float(observed_amount) != float(entry_cost)
    ):
        raise ValueError("project_spend_official_billing_immature")
    gpu_hours = math.fsum(
        hours
        for item in matches[0].get("items") or []
        if isinstance(item, Mapping)
        and item.get("type") == "gpu"
        and (hours := _positive_gpu_hours(item.get("description"))) is not None
    )
    if gpu_hours <= 0:
        raise ValueError("project_spend_official_billing_immature")
    return {
        "attempt_id": str(entry.get("attempt_id") or ""),
        "provider_instance_id": instance_id,
        "official_billing_response_sha256": billing_record["sha256"],
        "cost_usd": float(entry["cost_usd"]),
        "gpu_hours": gpu_hours,
        "status": "mature_nonzero_gpu_time",
    }


def _baseline(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source, value = _read(path, code="project_spend_baseline_invalid")
    if value.get("schema_version") == HUMAN_BASELINE_SCHEMA_VERSION:
        text = value.get("authorization_text")
        opening = value.get("opening_project_exposure_usd")
        ceiling = value.get("aggregate_project_ceiling_usd")
        attempt = value.get("authorized_attempt")
        bounded = value.get("maximum_bounded_exposure_after_full_attempt_reserve_usd")
        headroom = value.get("minimum_guaranteed_headroom_after_full_attempt_reserve_usd")
        text_digest = (
            "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            if isinstance(text, str)
            else ""
        )
        if (
            value.get("status") != "authorized"
            or value.get("program_id") != "arm-decision-proof-v1"
            or not isinstance(text, str)
            or not text.strip()
            or value.get("authorization_text_sha256") != text_digest
            or not _finite(opening)
            or not _finite(ceiling)
            or not isinstance(attempt, Mapping)
            or attempt.get("count") != 1
            or attempt.get("retry_cap") != 0
            or not _finite(attempt.get("maximum_spend_usd"))
            or float(attempt.get("maximum_spend_usd", 0)) <= 0
            or not _finite(attempt.get("maximum_hourly_rate_usd"))
            or float(attempt.get("maximum_hourly_rate_usd", 0)) <= 0
            or isinstance(attempt.get("hard_ttl_seconds"), bool)
            or not isinstance(attempt.get("hard_ttl_seconds"), int)
            or int(attempt.get("hard_ttl_seconds", 0)) <= 0
            or not _finite(bounded)
            or not _finite(headroom)
            or round(float(opening) + float(attempt["maximum_spend_usd"]), 6)
            != float(bounded)
            or round(float(ceiling) - float(bounded), 6) != float(headroom)
            or float(opening) > float(ceiling)
            or value.get("production_standing_authorization") is not False
            or value.get("launch_request") is not False
            or value.get("provider_mutation_performed") is not False
        ):
            raise ValueError("project_spend_baseline_invalid")
        return value, _record(
            source,
            baseline_kind="human_authorized_conservative_opening",
            authorization_text_sha256=text_digest,
            aggregate_goal_spend_before_attempt_usd=float(opening),
        )

    total = value.get("aggregate_goal_spend_before_attempt_usd")
    digest = value.get("authorization_digest")
    if (
        value.get("schema_version") != "native_task_arena_paid_attempt_authority.v1"
        or value.get("provider") != "vast"
        or value.get("paid_compute_authorized") is not True
        or value.get("maximum_automatic_retries") != 0
        or value.get("automatic_paid_retry_authorized") is not False
        or not _finite(total)
        or not isinstance(value.get("authorized_on"), str)
        or not value["authorized_on"].strip()
        or digest != canonical_digest(value, digest_field="authorization_digest")
    ):
        raise ValueError("project_spend_baseline_invalid")
    return value, _record(
        source,
        authorization_digest=digest,
        aggregate_goal_spend_before_attempt_usd=float(total),
    )


def _unposted_authority(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source, value = _read(path, code="project_spend_unposted_authority_invalid")
    cap = value.get("hard_attempt_spend_cap_usd")
    digest = value.get("authorization_digest")
    if (
        value.get("schema_version") != "native_task_arena_paid_attempt_authority.v1"
        or value.get("provider") != "vast"
        or value.get("paid_compute_authorized") is not True
        or value.get("maximum_paid_attempts") != 1
        or value.get("maximum_provider_allocations") != 1
        or value.get("maximum_automatic_retries") != 0
        or value.get("automatic_paid_retry_authorized") is not False
        or not _finite(cap)
        or float(cap) <= 0
        or digest != canonical_digest(value, digest_field="authorization_digest")
    ):
        raise ValueError("project_spend_unposted_authority_invalid")
    return value, _record(
        source,
        authorization_digest=digest,
        hard_attempt_spend_cap_usd=float(cap),
    )


def materialize_project_spend_reconciliation(
    *,
    baseline_authority_path: str | Path,
    posted_reconciliation_paths: Sequence[str | Path],
    unposted_authority_paths: Sequence[str | Path] = (),
    expected_coverage_ids: Sequence[str],
    completeness_reference: str,
    authorized_by: str,
    authorized_on: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one conservative project total from already-authorized evidence."""

    if (
        not completeness_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
    ):
        raise ValueError("project_spend_completeness_authority_missing")
    expected = tuple(str(item).strip() for item in expected_coverage_ids)
    if any(not item for item in expected) or len(set(expected)) != len(expected):
        raise ValueError("project_spend_expected_coverage_ids_invalid")

    baseline, baseline_record = _baseline(baseline_authority_path)
    if not expected and baseline.get("schema_version") != HUMAN_BASELINE_SCHEMA_VERSION:
        raise ValueError("project_spend_expected_coverage_ids_invalid")
    posted_records: list[dict[str, Any]] = []
    posted_entries: list[dict[str, Any]] = []
    posted_billing_maturity: list[dict[str, Any]] = []
    attempt_ids: set[str] = set()
    authority_digests: set[str] = set()
    for raw_path in posted_reconciliation_paths:
        value, record = validate_same_goal_spend_reconciliation(raw_path)
        posted_records.append(record)
        for entry in value["entries"]:
            attempt_id = str(entry["attempt_id"])
            authority_digest = str(entry.get("authority_digest") or "")
            if attempt_id in attempt_ids or (
                authority_digest and authority_digest in authority_digests
            ):
                raise ValueError("project_spend_posted_attempt_duplicate")
            attempt_ids.add(attempt_id)
            if authority_digest:
                authority_digests.add(authority_digest)
            retained_entry = json.loads(json.dumps(entry, allow_nan=False))
            maturity = _official_billing_maturity(retained_entry)
            if maturity is not None:
                posted_billing_maturity.append(maturity)
            posted_entries.append(retained_entry)

    unposted_records: list[dict[str, Any]] = []
    unposted_cap_total = 0.0
    for raw_path in unposted_authority_paths:
        value, record = _unposted_authority(raw_path)
        digest = str(value["authorization_digest"])
        if digest in authority_digests:
            raise ValueError("project_spend_unposted_authority_already_posted")
        authority_digests.add(digest)
        unposted_records.append(record)
        unposted_cap_total += float(value["hard_attempt_spend_cap_usd"])

    observed_coverage = attempt_ids | {
        str(record["authorization_digest"]) for record in unposted_records
    }
    if observed_coverage != set(expected):
        raise ValueError("project_spend_expected_attempt_coverage_mismatch")

    baseline_total = float(
        baseline_record["aggregate_goal_spend_before_attempt_usd"]
    )
    posted_total = math.fsum(float(entry["cost_usd"]) for entry in posted_entries)
    conservative_total = math.fsum((baseline_total, posted_total, unposted_cap_total))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "goal_id": "arm-decision-proof-v1",
        "baseline_authority": baseline_record,
        "baseline_total_cost_usd": baseline_total,
        "posted_reconciliations": posted_records,
        "posted_entries": posted_entries,
        "posted_entry_count": len(posted_entries),
        "posted_billing_maturity": posted_billing_maturity,
        "posted_increment_total_cost_usd": posted_total,
        "unposted_authorities": unposted_records,
        "unposted_full_cap_total_usd": unposted_cap_total,
        "covered_post_baseline_attempt_ids": sorted(observed_coverage),
        "total_cost_usd": conservative_total,
        "completeness_authority": {
            "authority_kind": "caller_supplied_human_authorized_coverage",
            "authority_reference": completeness_reference.strip(),
            "authorized_by": authorized_by.strip(),
            "authorized_on": authorized_on.strip(),
            "expected_post_baseline_coverage_ids": sorted(set(expected)),
        },
        "continuing_spend_conservatively_counted_at_full_cap": bool(
            unposted_records
        ),
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "The baseline is inherited from one sealed project authority; "
            "posted increments are official-billing reconciliations and "
            "admitted unposted attempts are counted at their full caps. The "
            "post-baseline coverage set is supplied under explicit human "
            "authority; this materializer does not discover global billing "
            "or launch-queue completeness."
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    _write_json_exclusive(destination, receipt)
    validate_project_spend_reconciliation(destination)
    return receipt


def validate_project_spend_reconciliation(
    path: str | Path, *, expected_total_cost_usd: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen a project receipt and every baseline, billing, and cap source."""

    source, value = _read(path, code="project_spend_reconciliation_invalid")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("goal_id") != "arm-decision-proof-v1"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not _finite(value.get("total_cost_usd"))
        or (
            # The receipt stores the raw ``math.fsum`` accumulation (its own
            # revalidation below recomputes that identical arithmetic, so the
            # stored representation must stay untouched), while an authority
            # retains the canonical six-decimal currency total.  Comparing the
            # two representations as exact binary floats refused a valid
            # launch (79.26491399999999 vs 79.264914), so cross-document
            # equality is at currency precision: one whole micro-dollar of
            # genuine disagreement still refuses.
            expected_total_cost_usd is not None
            and round(float(value["total_cost_usd"]), 6)
            != round(float(expected_total_cost_usd), 6)
        )
    ):
        raise ValueError("project_spend_reconciliation_invalid")
    baseline_record = value.get("baseline_authority")
    if not isinstance(baseline_record, Mapping):
        raise ValueError("project_spend_reconciliation_invalid")
    baseline, observed_baseline = _baseline(str(baseline_record.get("path") or ""))
    if dict(baseline_record) != observed_baseline:
        raise ValueError("project_spend_baseline_record_mismatch")

    posted_records = value.get("posted_reconciliations")
    posted_entries = value.get("posted_entries")
    posted_billing_maturity = value.get("posted_billing_maturity")
    unposted_records = value.get("unposted_authorities")
    if not all(
        isinstance(item, list)
        for item in (
            posted_records,
            posted_entries,
            posted_billing_maturity,
            unposted_records,
        )
    ):
        raise ValueError("project_spend_reconciliation_invalid")
    reopened_entries: list[dict[str, Any]] = []
    for record in posted_records:
        if not isinstance(record, Mapping):
            raise ValueError("project_spend_posted_record_invalid")
        reopened, observed = validate_same_goal_spend_reconciliation(
            str(record.get("path") or "")
        )
        if dict(record) != observed:
            raise ValueError("project_spend_posted_record_mismatch")
        reopened_entries.extend(reopened["entries"])
    if reopened_entries != posted_entries or value.get("posted_entry_count") != len(
        posted_entries
    ):
        raise ValueError("project_spend_posted_entries_mismatch")
    reopened_maturity = [
        maturity
        for entry in posted_entries
        if (maturity := _official_billing_maturity(entry)) is not None
    ]
    if reopened_maturity != posted_billing_maturity:
        raise ValueError("project_spend_billing_maturity_mismatch")

    authority_digests = {
        str(entry.get("authority_digest") or "")
        for entry in posted_entries
        if str(entry.get("authority_digest") or "")
    }
    reopened_unposted: list[dict[str, Any]] = []
    unposted_total = 0.0
    for record in unposted_records:
        if not isinstance(record, Mapping):
            raise ValueError("project_spend_unposted_record_invalid")
        authority, observed = _unposted_authority(str(record.get("path") or ""))
        if dict(record) != observed or authority["authorization_digest"] in authority_digests:
            raise ValueError("project_spend_unposted_record_mismatch")
        authority_digests.add(str(authority["authorization_digest"]))
        reopened_unposted.append(authority)
        unposted_total += float(authority["hard_attempt_spend_cap_usd"])

    attempt_ids = [str(entry["attempt_id"]) for entry in posted_entries]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("project_spend_posted_attempt_duplicate")
    coverage = set(attempt_ids) | {
        str(authority["authorization_digest"]) for authority in reopened_unposted
    }
    completeness = value.get("completeness_authority")
    expected = (
        set(completeness.get("expected_post_baseline_coverage_ids") or [])
        if isinstance(completeness, Mapping)
        else set()
    )
    if (
        not isinstance(completeness, Mapping)
        or completeness.get("authority_kind")
        != "caller_supplied_human_authorized_coverage"
        or not str(completeness.get("authority_reference") or "").strip()
        or not str(completeness.get("authorized_by") or "").strip()
        or not str(completeness.get("authorized_on") or "").strip()
        or coverage != expected
        or sorted(coverage) != value.get("covered_post_baseline_attempt_ids")
    ):
        raise ValueError("project_spend_completeness_authority_invalid")

    baseline_total = float(
        observed_baseline["aggregate_goal_spend_before_attempt_usd"]
    )
    posted_total = math.fsum(float(entry["cost_usd"]) for entry in posted_entries)
    total = math.fsum((baseline_total, posted_total, unposted_total))
    if (
        value.get("baseline_total_cost_usd") != baseline_total
        or value.get("posted_increment_total_cost_usd") != posted_total
        or value.get("unposted_full_cap_total_usd") != unposted_total
        or value.get("total_cost_usd") != total
        or value.get("continuing_spend_conservatively_counted_at_full_cap")
        is not bool(unposted_records)
        or value.get("provider_mutation_performed") is not False
        or value.get("raw_secret_values_recorded") is not False
    ):
        raise ValueError("project_spend_total_invalid")
    return value, _record(
        source,
        receipt_digest=value["receipt_digest"],
        total_cost_usd=total,
        posted_entry_count=len(posted_entries),
        unposted_authority_count=len(unposted_records),
    )


def project_spend_dependency_records(
    value: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return every byte record recursively required to reopen a project receipt."""

    dependencies: list[tuple[str, Mapping[str, Any]]] = []
    baseline = value.get("baseline_authority")
    if not isinstance(baseline, Mapping):
        raise ValueError("project_spend_dependency_invalid")
    dependencies.append(("baseline_authority", baseline))
    for index, record in enumerate(value.get("posted_reconciliations") or []):
        if not isinstance(record, Mapping):
            raise ValueError("project_spend_dependency_invalid")
        dependencies.append((f"posted_reconciliation_{index}", record))
    for entry_index, entry in enumerate(value.get("posted_entries") or []):
        if not isinstance(entry, Mapping):
            raise ValueError("project_spend_dependency_invalid")
        for source in entry.get("source_receipts") or []:
            if not isinstance(source, Mapping) or not isinstance(
                source.get("record"), Mapping
            ):
                raise ValueError("project_spend_dependency_invalid")
            role = str(source.get("role") or "source")
            dependencies.append(
                (f"posted_entry_{entry_index}_{role}", source["record"])
            )
    for index, record in enumerate(value.get("unposted_authorities") or []):
        if not isinstance(record, Mapping):
            raise ValueError("project_spend_dependency_invalid")
        dependencies.append((f"unposted_authority_{index}", record))
    return dependencies


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roll one sealed project-spend baseline forward without spending."
    )
    parser.add_argument("--baseline-authority", required=True)
    parser.add_argument("--posted-reconciliation", action="append", default=[])
    parser.add_argument("--unposted-authority", action="append", default=[])
    parser.add_argument("--expected-coverage-id", action="append", default=[])
    parser.add_argument("--completeness-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = materialize_project_spend_reconciliation(
            baseline_authority_path=args.baseline_authority,
            posted_reconciliation_paths=args.posted_reconciliation,
            unposted_authority_paths=args.unposted_authority,
            expected_coverage_ids=args.expected_coverage_id,
            completeness_reference=args.completeness_reference,
            authorized_by=args.authorized_by,
            authorized_on=args.authorized_on,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "output": str(Path(args.output).expanduser().resolve()),
                "total_cost_usd": value["total_cost_usd"],
                "receipt_digest": value["receipt_digest"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "materialize_project_spend_reconciliation",
    "project_spend_dependency_records",
    "validate_project_spend_reconciliation",
]
