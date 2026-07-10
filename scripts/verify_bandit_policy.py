#!/usr/bin/env python3
"""Enforce the reviewed Bandit high/medium-finding policy.

Bandit findings are matched by exact code-and-location fingerprints so a moved
or edited finding re-enters review instead of silently inheriting an exception.
High findings can never be baselined. Medium exceptions are temporary,
owner-bound, and expire.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "blueprint.bandit_triage.v1"
REPORT_SCHEMA_VERSION = "blueprint.bandit_policy_gate.v1"
EXPECTED_SCANNER_VERSION = "1.8.6"
ALLOWED_DISPOSITIONS = {"accepted_risk", "false_positive", "mitigated_by_control"}


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _canonical_code(value: object) -> str:
    lines: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = re.sub(r"^\s*\d+\s+", "", raw_line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _relative_filename(value: object, root: Path) -> str:
    path = Path(str(value or ""))
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def finding_fingerprint(finding: Mapping[str, Any], *, root: Path) -> str:
    identity = {
        "filename": _relative_filename(finding.get("filename"), root),
        "test_id": str(finding.get("test_id") or ""),
        # Bandit can report multiple findings from the same short context block.
        # Keeping the exact scanner line distinguishes those sites and makes an
        # edit re-enter review instead of inheriting a nearby exception.
        "line_number": finding.get("line_number"),
        "issue_text": " ".join(str(finding.get("issue_text") or "").split()),
        "code": _canonical_code(finding.get("code")),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_finding(finding: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    return {
        "fingerprint": finding_fingerprint(finding, root=root),
        "test_id": str(finding.get("test_id") or ""),
        "severity": str(finding.get("issue_severity") or "").upper(),
        "confidence": str(finding.get("issue_confidence") or "").upper(),
        "filename": _relative_filename(finding.get("filename"), root),
        "line_number": finding.get("line_number"),
        "issue_text": " ".join(str(finding.get("issue_text") or "").split()),
    }


def validate_policy(
    *,
    bandit_report: Mapping[str, Any],
    triage: Mapping[str, Any],
    root: Path,
    today: date,
) -> dict[str, Any]:
    blockers: list[str] = []
    if triage.get("schema_version") != SCHEMA_VERSION:
        blockers.append("triage_schema_version_invalid")
    if triage.get("scanner_version") != EXPECTED_SCANNER_VERSION:
        blockers.append("triage_scanner_version_invalid")
    raw_entries = triage.get("entries")
    entries = raw_entries if isinstance(raw_entries, list) else []
    if not isinstance(raw_entries, list):
        blockers.append("triage_entries_missing")

    baselined: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry)
        fingerprint = str(entry.get("fingerprint") or "")
        prefix = f"triage_entry:{index}"
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            blockers.append(f"{prefix}:fingerprint_invalid")
            continue
        if fingerprint in baselined:
            blockers.append(f"{prefix}:fingerprint_duplicate")
        if str(entry.get("severity") or "").upper() != "MEDIUM":
            blockers.append(f"{prefix}:only_medium_findings_may_be_baselined")
        if str(entry.get("disposition") or "") not in ALLOWED_DISPOSITIONS:
            blockers.append(f"{prefix}:disposition_invalid")
        if len(str(entry.get("owner") or "").strip()) < 3:
            blockers.append(f"{prefix}:owner_missing")
        if len(str(entry.get("reason") or "").strip()) < 20:
            blockers.append(f"{prefix}:reason_missing")
        expires_on: date | None = None
        reviewed_on: date | None = None
        try:
            expires_on = date.fromisoformat(str(entry.get("expires_on") or ""))
        except ValueError:
            blockers.append(f"{prefix}:expires_on_invalid")
        else:
            if expires_on < today:
                blockers.append(f"triage_expired:{fingerprint}")
        try:
            reviewed_on = date.fromisoformat(str(entry.get("reviewed_on") or ""))
        except ValueError:
            blockers.append(f"{prefix}:reviewed_on_invalid")
        else:
            if reviewed_on > today:
                blockers.append(f"{prefix}:reviewed_on_future")
        if expires_on is not None and reviewed_on is not None:
            if expires_on < reviewed_on:
                blockers.append(f"{prefix}:expires_before_review")
            if expires_on > reviewed_on + timedelta(days=90):
                blockers.append(f"{prefix}:review_window_exceeds_90_days")
        baselined[fingerprint] = entry

    raw_findings = bandit_report.get("results")
    findings = raw_findings if isinstance(raw_findings, list) else []
    if not isinstance(raw_findings, list):
        blockers.append("bandit_results_missing")
    normalized = [normalized_finding(_mapping(item), root=root) for item in findings]
    high = [item for item in normalized if item["severity"] == "HIGH"]
    medium = [item for item in normalized if item["severity"] == "MEDIUM"]
    low = [item for item in normalized if item["severity"] == "LOW"]
    for item in high:
        blockers.append(
            f"high_finding:{item['test_id']}:{item['filename']}:{item['line_number']}"
        )
    current_medium = {item["fingerprint"] for item in medium}
    for item in medium:
        matched_entry = baselined.get(item["fingerprint"])
        if matched_entry is None:
            blockers.append(
                f"untriaged_medium:{item['test_id']}:{item['filename']}:{item['line_number']}"
            )
            continue
        metadata = {
            "test_id": item["test_id"],
            "filename": item["filename"],
            "line_number": item["line_number"],
        }
        for key, expected in metadata.items():
            if matched_entry.get(key) != expected:
                blockers.append(
                    f"triage_metadata_mismatch:{item['fingerprint']}:{key}"
                )
    for fingerprint in sorted(set(baselined) - current_medium):
        blockers.append(f"orphaned_triage_entry:{fingerprint}")

    blockers = sorted(set(blockers))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "scanner": "bandit",
        "finding_counts": {
            "high": len(high),
            "medium": len(medium),
            "low": len(low),
            "total": len(normalized),
        },
        "triaged_medium_count": len(current_medium & set(baselined)),
        "blockers": blockers,
        "claim_boundary": {
            "high_findings_cannot_be_baselined": True,
            "medium_exceptions_are_owner_bound_and_expiring": True,
            "passing_static_analysis_is_not_absence_of_vulnerabilities": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandit-report", type=Path, required=True)
    parser.add_argument(
        "--triage",
        type=Path,
        default=Path("docs/bandit_triage.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        bandit_report = json.loads(args.bandit_report.read_text(encoding="utf-8"))
        triage = json.loads(args.triage.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[bandit-policy] ERROR unreadable_input:{exc}", file=sys.stderr)
        return 1
    result = validate_policy(
        bandit_report=_mapping(bandit_report),
        triage=_mapping(triage),
        root=args.root.resolve(),
        today=date.today(),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[bandit-policy] "
        f"status={result['status']} high={result['finding_counts']['high']} "
        f"medium={result['finding_counts']['medium']}"
    )
    for blocker in result["blockers"]:
        print(f"[bandit-policy] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
