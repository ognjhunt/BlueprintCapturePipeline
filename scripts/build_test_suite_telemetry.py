#!/usr/bin/env python3
"""Summarize full-lane duration and parametrization evidence from JUnit XML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from defusedxml import ElementTree as ET


SCHEMA_VERSION = "blueprint.test_suite_telemetry.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
NODEID_PROPERTY = "blueprint_nodeid"
TOP_LIMIT = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(value, 6)


def _nodeid(testcase: Any) -> str:
    properties = testcase.find("properties")
    property_rows = list(properties) if properties is not None else []
    values = [
        str(prop.attrib.get("value") or "").strip()
        for prop in property_rows
        if prop.tag == "property" and prop.attrib.get("name") == NODEID_PROPERTY
    ]
    if len(values) != 1 or not values[0]:
        raise ValueError("junit_nodeid_property_invalid")
    return values[0]


def _duration(testcase: Any) -> float:
    try:
        value = float(testcase.attrib.get("time", "0"))
    except (TypeError, ValueError) as exc:
        raise ValueError("junit_testcase_duration_invalid") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("junit_testcase_duration_invalid")
    return value


def _is_parametrized(nodeid: str) -> bool:
    test_name = nodeid.rsplit("::", 1)[-1]
    return "[" in test_name and test_name.endswith("]")


def _family(nodeid: str) -> str:
    return nodeid.split("[", 1)[0] if _is_parametrized(nodeid) else nodeid


def _shard_files(
    file_rows: list[dict[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = [
        {"index": index, "estimated_case_duration_seconds": 0.0, "files": []}
        for index in range(workers)
    ]
    for row in sorted(
        file_rows,
        key=lambda item: (-float(item["case_duration_seconds"]), str(item["path"])),
    ):
        target = min(
            shards,
            key=lambda shard: (
                float(shard["estimated_case_duration_seconds"]),
                int(shard["index"]),
            ),
        )
        target["files"].append(row["path"])
        target["estimated_case_duration_seconds"] += float(
            row["case_duration_seconds"]
        )
    for shard in shards:
        shard["estimated_case_duration_seconds"] = _round(
            float(shard["estimated_case_duration_seconds"])
        )
        shard["file_count"] = len(shard["files"])
    return shards


def build_telemetry(
    *, junit: Path, repository_sha: str, workers: int
) -> dict[str, Any]:
    repository_sha = repository_sha.strip().lower()
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        raise ValueError("repository_sha_invalid")
    if workers < 1:
        raise ValueError("workers_invalid")
    try:
        root = ET.parse(junit).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError("junit_invalid") from exc
    testcases = list(root.iter("testcase"))
    if not testcases:
        raise ValueError("junit_has_no_testcases")
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    try:
        suite_wall_seconds = sum(float(suite.attrib["time"]) for suite in suites)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("junit_suite_duration_invalid") from exc
    if not suites or not math.isfinite(suite_wall_seconds) or suite_wall_seconds < 0:
        raise ValueError("junit_suite_duration_invalid")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for testcase in testcases:
        nodeid = _nodeid(testcase)
        if nodeid in seen:
            raise ValueError(f"junit_duplicate_nodeid:{nodeid}")
        seen.add(nodeid)
        rows.append(
            {
                "nodeid": nodeid,
                "path": nodeid.split("::", 1)[0],
                "family": _family(nodeid),
                "parametrized": _is_parametrized(nodeid),
                "duration": _duration(testcase),
            }
        )

    files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        files[str(row["path"])].append(row)
        families[str(row["family"])].append(row)

    file_rows = [
        {
            "path": path,
            "test_count": len(values),
            "parametrized_case_count": sum(
                bool(value["parametrized"]) for value in values
            ),
            "case_duration_seconds": _round(
                sum(float(value["duration"]) for value in values)
            ),
        }
        for path, values in files.items()
    ]
    file_rows.sort(key=lambda row: (-float(row["case_duration_seconds"]), row["path"]))

    family_rows = [
        {
            "family": family,
            "case_count": len(values),
            "case_duration_seconds": _round(
                sum(float(value["duration"]) for value in values)
            ),
        }
        for family, values in families.items()
        if len(values) > 1 and any(bool(value["parametrized"]) for value in values)
    ]
    family_rows.sort(
        key=lambda row: (-int(row["case_count"]), -float(row["case_duration_seconds"]), row["family"])
    )

    total_duration = sum(float(row["duration"]) for row in rows)
    parametrized_count = sum(bool(row["parametrized"]) for row in rows)
    shards = _shard_files(file_rows, workers=workers)
    max_file_duration = max(float(row["case_duration_seconds"]) for row in file_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_sha": repository_sha,
        "junit_sha256": _sha256(junit),
        "summary": {
            "test_count": len(rows),
            "test_file_count": len(file_rows),
            "parametrized_case_count": parametrized_count,
            "parametrized_case_fraction": _round(parametrized_count / len(rows)),
            "summed_case_duration_seconds": _round(total_duration),
            "reported_suite_wall_seconds": _round(suite_wall_seconds),
            "maximum_test_file_duration_seconds": _round(max_file_duration),
            "line_coverage_collected": False,
        },
        "parallelization": {
            "strategy": "pytest_xdist_loadfile",
            "workers": workers,
            "serial_case_duration_seconds": _round(total_duration),
            "theoretical_lower_bound_seconds": _round(
                max(total_duration / workers, max_file_duration)
            ),
            "lpt_file_assignment_estimate_seconds": _round(
                max(float(shard["estimated_case_duration_seconds"]) for shard in shards)
            ),
            "shards": shards,
        },
        "top_testcases_by_duration": [
            {
                "nodeid": row["nodeid"],
                "duration_seconds": _round(float(row["duration"])),
            }
            for row in sorted(rows, key=lambda row: (-float(row["duration"]), row["nodeid"]))[
                :TOP_LIMIT
            ]
        ],
        "top_test_files_by_duration": file_rows[:TOP_LIMIT],
        "top_parametrized_families_by_case_count": family_rows[:TOP_LIMIT],
    }


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_telemetry(
            junit=args.junit.resolve(),
            repository_sha=args.repository_sha,
            workers=args.workers,
        )
        _write_json_atomic(args.output.resolve(), result)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[test-suite-telemetry] ERROR {exc}", file=sys.stderr)
        return 1
    print(
        "[test-suite-telemetry] ok "
        f"tests={result['summary']['test_count']} "
        f"files={result['summary']['test_file_count']} "
        f"workers={result['parallelization']['workers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
