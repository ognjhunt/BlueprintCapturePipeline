#!/usr/bin/env python3
"""Build fail-closed CPU full-lane evidence from canonical pytest artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from defusedxml import ElementTree as ET

if __package__ in {None, ""}:  # Direct ``python scripts/...`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_full_lane_collection import verify as verify_collection


EVIDENCE_SCHEMA = "blueprint.critical_capability_lane_evidence.v1"
PAYLOAD_SCHEMA = "blueprint.full_pytest_lane.v1"
COLLECTION_SCHEMA = "blueprint_full_lane_collection.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_NAMES = {
    "planned": "full-test-lane-planned.json",
    "executed": "full-test-lane-executed.json",
    "junit": "full-test-lane-junit.xml",
}
NODEID_PROPERTY = "blueprint_nodeid"
DETERMINISTIC_FIELDS = {
    "schema_version",
    "lane_id",
    "evidence_schema_version",
    "repository_sha",
    "status",
    "executed",
    "skipped_count",
    "test_count",
    "passed_count",
    "failure_count",
    "error_count",
    "planned_test_count",
    "executed_test_count",
    "nodeids_sha256",
    "testcase_outcomes_sha256",
    "skipped_testcases",
    "skipped_testcases_truncated",
    "artifact_digests",
    "artifact_sizes",
    "blockers",
    "claim_boundary",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path, *, phase: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{phase}_manifest_not_object")
    result = dict(payload)
    if result.get("schema_version") != COLLECTION_SCHEMA:
        raise ValueError(f"{phase}_manifest_schema_invalid")
    if result.get("phase") != phase:
        raise ValueError(f"{phase}_manifest_phase_invalid")
    return result


def _testcase_identifier(testcase: Any) -> str:
    classname = str(testcase.attrib.get("classname") or "").strip()
    name = str(testcase.attrib.get("name") or "").strip()
    return f"{classname}::{name}" if classname else name


def _junit_outcomes(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError("junit_invalid") from exc
    testcases = list(root.iter("testcase"))
    if not testcases:
        raise ValueError("junit_has_no_testcases")
    counts = {"passed": 0, "failures": 0, "errors": 0, "skipped": 0}
    outcome_rows: list[str] = []
    skip_details: list[dict[str, str]] = []
    junit_nodeids: list[str] = []
    nodeid_errors: list[str] = []
    for testcase in testcases:
        fallback_identifier = _testcase_identifier(testcase)
        properties = testcase.find("properties")
        property_rows = list(properties) if properties is not None else []
        nodeid_properties = [
            prop
            for prop in property_rows
            if prop.tag == "property" and prop.attrib.get("name") == NODEID_PROPERTY
        ]
        nodeid_values = [str(prop.attrib.get("value") or "").strip() for prop in nodeid_properties]
        if len(nodeid_properties) != 1 or not nodeid_values[0]:
            nodeid_errors.append(
                f"{fallback_identifier}:"
                f"{'missing' if not nodeid_properties else 'multiple' if len(nodeid_properties) > 1 else 'empty'}"
            )
            identifier = fallback_identifier
        else:
            identifier = nodeid_values[0]
            junit_nodeids.append(identifier)
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")
        if failure is not None:
            outcome = "failure"
            counts["failures"] += 1
        elif error is not None:
            outcome = "error"
            counts["errors"] += 1
        elif skipped is not None:
            outcome = "skipped"
            counts["skipped"] += 1
            if len(skip_details) < 100:
                reason = str(skipped.attrib.get("message") or skipped.text or "").strip()
                skip_details.append({"testcase": identifier, "reason": reason[:500]})
        else:
            outcome = "passed"
            counts["passed"] += 1
        outcome_rows.append(f"{identifier}\t{outcome}")

    suite_rows = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suite_rows:
        raise ValueError("junit_has_no_suites")
    declared = {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suite_rows)
        for key in ("tests", "failures", "errors", "skipped")
    }
    observed = {
        "tests": len(testcases),
        "failures": counts["failures"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
    }
    return {
        **observed,
        "passed": counts["passed"],
        "declared_counts": declared,
        "counts_match_declared": declared == observed,
        "testcase_outcomes_sha256": hashlib.sha256(
            "\n".join(sorted(outcome_rows)).encode("utf-8")
        ).hexdigest(),
        "junit_nodeids": sorted(junit_nodeids),
        "nodeid_errors": nodeid_errors,
        "duplicate_nodeids": sorted(
            nodeid for nodeid, count in Counter(junit_nodeids).items() if count > 1
        ),
        "skipped_testcases": skip_details,
        "skipped_testcases_truncated": counts["skipped"] > len(skip_details),
    }


def build_cpu_full_lane_evidence(
    *,
    planned: Path,
    executed: Path,
    junit: Path,
    repository_sha: str,
) -> dict[str, Any]:
    repository_sha = repository_sha.strip().lower()
    blockers: list[str] = []
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("cpu_full_repository_sha_invalid")

    paths = {"planned": planned, "executed": executed, "junit": junit}
    artifact_digests: dict[str, str] = {}
    artifact_sizes: dict[str, int] = {}
    for label, path in paths.items():
        if path.is_symlink():
            blockers.append(f"cpu_full_source_symlink:{label}")
        elif not path.is_file():
            blockers.append(f"cpu_full_source_missing:{label}")
        else:
            artifact_digests[SOURCE_NAMES[label]] = _sha256(path)
            artifact_sizes[SOURCE_NAMES[label]] = path.stat().st_size

    planned_payload: dict[str, Any] = {}
    executed_payload: dict[str, Any] = {}
    outcomes: dict[str, Any] = {}
    if len(artifact_digests) == len(paths):
        try:
            collection_blockers = verify_collection(planned, executed)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            blockers.append(f"cpu_full_collection_unreadable:{type(exc).__name__}")
        else:
            blockers.extend(f"cpu_full_collection:{item}" for item in collection_blockers)
        try:
            planned_payload = _read_manifest(planned, phase="planned")
            executed_payload = _read_manifest(executed, phase="executed")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            blockers.append(f"cpu_full_manifest_invalid:{exc}")
        try:
            outcomes = _junit_outcomes(junit)
        except ValueError as exc:
            blockers.append(f"cpu_full_{exc}")

    planned_count = planned_payload.get("test_count", 0)
    executed_count = executed_payload.get("test_count", 0)
    nodeids_digest = str(planned_payload.get("nodeids_sha256") or "")
    testcase_digest = str(outcomes.get("testcase_outcomes_sha256") or "")
    test_count = outcomes.get("tests", 0)
    failure_count = outcomes.get("failures", 0)
    error_count = outcomes.get("errors", 0)
    skipped_count = outcomes.get("skipped", 0)
    if outcomes:
        if outcomes.get("counts_match_declared") is not True:
            blockers.append("cpu_full_junit_declared_counts_mismatch")
        if type(test_count) is not int or test_count <= 0:
            blockers.append("cpu_full_junit_test_count_invalid")
        if test_count != planned_count or test_count != executed_count:
            blockers.append("cpu_full_junit_collection_count_mismatch")
        if outcomes.get("nodeid_errors"):
            blockers.append("cpu_full_junit_nodeid_property_invalid")
        if outcomes.get("duplicate_nodeids"):
            blockers.append("cpu_full_junit_duplicate_nodeids")
        executed_nodeids = executed_payload.get("nodeids")
        if outcomes.get("junit_nodeids") != executed_nodeids:
            blockers.append("cpu_full_junit_nodeids_mismatch")
        if failure_count:
            blockers.append(f"cpu_full_junit_failures:{failure_count}")
        if error_count:
            blockers.append(f"cpu_full_junit_errors:{error_count}")
        if skipped_count:
            blockers.append(f"cpu_full_junit_skipped:{skipped_count}")
    if nodeids_digest and DIGEST_PATTERN.fullmatch(nodeids_digest) is None:
        blockers.append("cpu_full_nodeids_digest_invalid")
    if testcase_digest and DIGEST_PATTERN.fullmatch(testcase_digest) is None:
        blockers.append("cpu_full_testcase_outcomes_digest_invalid")

    blockers = sorted(set(blockers))
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "lane_id": "cpu_full",
        "evidence_schema_version": PAYLOAD_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_sha": repository_sha,
        "status": "passed" if not blockers else "blocked",
        "executed": bool(executed_payload and outcomes),
        "skipped_count": skipped_count,
        "test_count": test_count,
        "passed_count": outcomes.get("passed", 0),
        "failure_count": failure_count,
        "error_count": error_count,
        "planned_test_count": planned_count,
        "executed_test_count": executed_count,
        "nodeids_sha256": nodeids_digest or None,
        "testcase_outcomes_sha256": testcase_digest or None,
        "skipped_testcases": outcomes.get("skipped_testcases", []),
        "skipped_testcases_truncated": outcomes.get("skipped_testcases_truncated", False),
        "artifact_digests": artifact_digests,
        "artifact_sizes": artifact_sizes,
        "blockers": blockers,
        "claim_boundary": {
            "collection_identity_is_not_test_success": True,
            "zero_failures_errors_and_skips_required": True,
            "cpu_lane_is_not_gpu_provider_execution": True,
            "cpu_lane_is_not_native_lerobot_export_proof": True,
        },
    }


def validate_cpu_full_lane_evidence(
    evidence: Mapping[str, Any],
    *,
    planned: Path,
    executed: Path,
    junit: Path,
    repository_sha: str,
) -> list[str]:
    """Recompute the canonical envelope and reject stale or self-asserted fields."""

    expected = build_cpu_full_lane_evidence(
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=repository_sha,
    )
    blockers: list[str] = []
    expected_fields = DETERMINISTIC_FIELDS | {"generated_at"}
    if set(evidence) != expected_fields:
        blockers.append("cpu_full_evidence_fields_invalid")
    if expected["status"] != "passed":
        blockers.extend(f"cpu_full_source:{item}" for item in expected["blockers"])
    for field in sorted(DETERMINISTIC_FIELDS):
        if evidence.get(field) != expected.get(field):
            blockers.append(f"cpu_full_evidence_field_mismatch:{field}")
    generated_at = evidence.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        blockers.append("cpu_full_evidence_generated_at_invalid")
    else:
        try:
            parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError:
            blockers.append("cpu_full_evidence_generated_at_invalid")
        else:
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                blockers.append("cpu_full_evidence_generated_at_invalid")
    return sorted(set(blockers))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned", type=Path, required=True)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_cpu_full_lane_evidence(
        planned=args.planned.expanduser().absolute(),
        executed=args.executed.expanduser().absolute(),
        junit=args.junit.expanduser().absolute(),
        repository_sha=args.repository_sha,
    )
    _write_json_atomic(args.output.resolve(), result)
    print(f"[cpu-full-evidence] status={result['status']}")
    for blocker in result["blockers"]:
        print(f"[cpu-full-evidence] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
