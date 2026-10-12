#!/usr/bin/env python3
"""Plan, verify, and aggregate deterministic serial full-test-lane shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as WritableElementTree

from defusedxml import ElementTree as SafeElementTree


COLLECTION_SCHEMA = "blueprint_full_lane_collection.v1"
BASELINE_SCHEMA = "blueprint.full_lane_duration_baseline.v2"
PLAN_SCHEMA = "blueprint.full_lane_shard_plan.v2"
PLAN_STRATEGY = "lpt_heavy_file_split_serial_shards"
SHARD_RECEIPT_SCHEMA = "blueprint.full_lane_shard_verification.v1"
AGGREGATE_SCHEMA = "blueprint.full_lane_shard_aggregate.v1"
NODEID_PROPERTY = "blueprint_nodeid"
# 2026-10-12: four shards planned against a two-month-old baseline ran
# 68/51/35/28 minutes and gated every deploy on the slowest. Twelve shards
# balanced on measured durations finish together in about nineteen minutes.
SHARD_COUNT = 12
# Tests at least this slow are recorded individually in the baseline, so a
# split file schedules its few long tests by their own measured durations.
HEAVY_TEST_SECONDS = 5.0
# A file estimated above this share of one shard's mean load is scheduled by
# node ID; every other file stays whole on one shard.
SPLIT_SHARE_OF_MEAN_SHARD = 0.5
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FullLaneShardError(RuntimeError):
    """Raised when shard evidence is missing, inconsistent, or non-green."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nodeids_digest(nodeids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(nodeids).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FullLaneShardError(f"json_not_object:{path.name}")
    return payload


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


def _collection(payload: Mapping[str, Any], *, phase: str) -> list[str]:
    blockers: list[str] = []
    if payload.get("schema_version") != COLLECTION_SCHEMA:
        blockers.append(f"{phase}_schema_invalid")
    if payload.get("phase") != phase:
        blockers.append(f"{phase}_phase_invalid")
    raw_nodeids = payload.get("nodeids")
    nodeids = [str(value) for value in raw_nodeids] if isinstance(raw_nodeids, list) else []
    if not nodeids:
        blockers.append(f"{phase}_nodeids_empty")
    if len(nodeids) != len(set(nodeids)):
        blockers.append(f"{phase}_nodeids_duplicate")
    if payload.get("test_count") != len(nodeids):
        blockers.append(f"{phase}_test_count_mismatch")
    if payload.get("nodeids_sha256") != _nodeids_digest(nodeids):
        blockers.append(f"{phase}_nodeids_digest_mismatch")
    if blockers:
        raise FullLaneShardError(",".join(blockers))
    return nodeids


def _nodeid(testcase: Any) -> str:
    properties = testcase.find("properties")
    rows = list(properties) if properties is not None else []
    values = [
        str(row.attrib.get("value") or "").strip()
        for row in rows
        if row.tag == "property" and row.attrib.get("name") == NODEID_PROPERTY
    ]
    if len(values) != 1 or not values[0]:
        raise FullLaneShardError("junit_nodeid_property_invalid")
    return values[0]


def _duration(testcase: Any) -> float:
    try:
        value = float(testcase.attrib.get("time", "0"))
    except (TypeError, ValueError) as exc:
        raise FullLaneShardError("junit_testcase_duration_invalid") from exc
    if not math.isfinite(value) or value < 0:
        raise FullLaneShardError("junit_testcase_duration_invalid")
    return value


def _junit(path: Path) -> dict[str, Any]:
    try:
        root = SafeElementTree.parse(path).getroot()
    except (SafeElementTree.ParseError, OSError) as exc:
        raise FullLaneShardError("junit_invalid") from exc
    testcases = list(root.iter("testcase"))
    if not testcases:
        raise FullLaneShardError("junit_has_no_testcases")
    rows: list[dict[str, Any]] = []
    for testcase in testcases:
        nodeid = _nodeid(testcase)
        failure = testcase.find("failure") is not None
        error = testcase.find("error") is not None
        skipped = testcase.find("skipped") is not None
        rows.append(
            {
                "nodeid": nodeid,
                "duration_seconds": _duration(testcase),
                "failure": failure,
                "error": error,
                "skipped": skipped,
                "element": testcase,
            }
        )
    duplicates = sorted(
        nodeid
        for nodeid, count in Counter(str(row["nodeid"]) for row in rows).items()
        if count > 1
    )
    return {
        "rows": rows,
        "nodeids": sorted(str(row["nodeid"]) for row in rows),
        "duplicates": duplicates,
        "failures": sum(bool(row["failure"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "skipped": sum(bool(row["skipped"]) for row in rows),
        "duration_seconds": sum(float(row["duration_seconds"]) for row in rows),
    }


def build_duration_baseline(
    *, junit: Path, source_sha: str, source_run_id: int
) -> dict[str, Any]:
    source_sha = source_sha.strip().lower()
    if SHA_PATTERN.fullmatch(source_sha) is None:
        raise FullLaneShardError("duration_baseline_source_sha_invalid")
    if source_run_id <= 0:
        raise FullLaneShardError("duration_baseline_source_run_id_invalid")
    outcomes = _junit(junit)
    if outcomes["duplicates"]:
        raise FullLaneShardError("duration_baseline_junit_duplicates")
    files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes["rows"]:
        files[str(row["nodeid"]).split("::", 1)[0]].append(row)
    file_rows = [
        {
            "path": path,
            "test_count": len(rows),
            "duration_seconds": round(
                sum(float(row["duration_seconds"]) for row in rows), 6
            ),
        }
        for path, rows in sorted(files.items())
    ]
    total_duration = float(outcomes["duration_seconds"])
    test_count = len(outcomes["rows"])
    heavy_tests = [
        {
            "nodeid": str(row["nodeid"]),
            "duration_seconds": round(float(row["duration_seconds"]), 6),
        }
        for row in sorted(outcomes["rows"], key=lambda row: str(row["nodeid"]))
        if float(row["duration_seconds"]) >= HEAVY_TEST_SECONDS
    ]
    return {
        "schema_version": BASELINE_SCHEMA,
        "source_repository_sha": source_sha,
        "source_run_id": source_run_id,
        "source_junit_sha256": _sha256(junit),
        "test_count": test_count,
        "file_count": len(file_rows),
        "total_duration_seconds": round(total_duration, 6),
        "default_seconds_per_test": round(total_duration / test_count, 9),
        "heavy_test_threshold_seconds": HEAVY_TEST_SECONDS,
        "heavy_tests": heavy_tests,
        "files": file_rows,
    }


def _validate_heavy_tests(
    payload: Mapping[str, Any], by_path: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """Return recorded per-test durations, or raise when any row is inconsistent."""
    if payload.get("heavy_test_threshold_seconds") != HEAVY_TEST_SECONDS:
        raise FullLaneShardError("duration_baseline_heavy_test_threshold_invalid")
    raw_rows = payload.get("heavy_tests")
    if not isinstance(raw_rows, list):
        raise FullLaneShardError("duration_baseline_heavy_test_invalid")
    heavy: dict[str, float] = {}
    per_file: dict[str, float] = defaultdict(float)
    per_file_count: Counter[str] = Counter()
    for row in raw_rows:
        nodeid = str(row.get("nodeid") or "") if isinstance(row, Mapping) else ""
        duration = row.get("duration_seconds") if isinstance(row, Mapping) else None
        path = nodeid.split("::", 1)[0]
        if (
            "::" not in nodeid
            or path not in by_path
            or nodeid in heavy
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < HEAVY_TEST_SECONDS
        ):
            raise FullLaneShardError("duration_baseline_heavy_test_invalid")
        heavy[nodeid] = float(duration)
        per_file[path] += float(duration)
        per_file_count[path] += 1
    for path, seconds in per_file.items():
        file_row = by_path[path]
        # A file's recorded tests can never outlast the whole file, and can
        # never outnumber its tests (rounding is per row, so allow 1 us each).
        if (
            seconds > float(file_row["duration_seconds"]) + 1e-6 * per_file_count[path]
            or per_file_count[path] > int(file_row["test_count"])
        ):
            raise FullLaneShardError("duration_baseline_heavy_test_invalid")
    return heavy


def _validate_baseline(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    blockers: list[str] = []
    if payload.get("schema_version") != BASELINE_SCHEMA:
        blockers.append("duration_baseline_schema_invalid")
    if SHA_PATTERN.fullmatch(str(payload.get("source_repository_sha") or "")) is None:
        blockers.append("duration_baseline_source_sha_invalid")
    if not isinstance(payload.get("source_run_id"), int) or int(
        payload.get("source_run_id") or 0
    ) <= 0:
        blockers.append("duration_baseline_source_run_id_invalid")
    if DIGEST_PATTERN.fullmatch(str(payload.get("source_junit_sha256") or "")) is None:
        blockers.append("duration_baseline_junit_digest_invalid")
    raw_files = payload.get("files")
    file_rows = raw_files if isinstance(raw_files, list) else []
    by_path: dict[str, dict[str, Any]] = {}
    for row in file_rows:
        if not isinstance(row, Mapping):
            blockers.append("duration_baseline_file_row_invalid")
            continue
        path = str(row.get("path") or "")
        count = row.get("test_count")
        duration = row.get("duration_seconds")
        if (
            not path.startswith("tests/")
            or path in by_path
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            blockers.append("duration_baseline_file_row_invalid")
            continue
        by_path[path] = dict(row)
    if payload.get("file_count") != len(by_path):
        blockers.append("duration_baseline_file_count_mismatch")
    if payload.get("test_count") != sum(
        int(row["test_count"]) for row in by_path.values()
    ):
        blockers.append("duration_baseline_test_count_mismatch")
    total_duration = payload.get("total_duration_seconds")
    total_duration_value = (
        float(total_duration) if isinstance(total_duration, (int, float)) else 0.0
    )
    if (
        not isinstance(total_duration, (int, float))
        or not math.isfinite(total_duration_value)
        or total_duration_value <= 0
        or round(sum(float(row["duration_seconds"]) for row in by_path.values()), 6)
        != total_duration_value
    ):
        blockers.append("duration_baseline_total_duration_mismatch")
    default_seconds = payload.get("default_seconds_per_test")
    test_count = payload.get("test_count")
    if (
        not isinstance(default_seconds, (int, float))
        or not math.isfinite(float(default_seconds))
        or float(default_seconds) <= 0
        or not isinstance(test_count, int)
        or test_count <= 0
        or round(total_duration_value / test_count, 9)
        != float(default_seconds)
    ):
        blockers.append("duration_baseline_default_mismatch")
    if blockers:
        raise FullLaneShardError(",".join(sorted(set(blockers))))
    _validate_heavy_tests(payload, by_path)
    return by_path


def build_shard_plan(
    *,
    planned: Mapping[str, Any],
    duration_baseline: Mapping[str, Any],
    repository_sha: str,
    shard_count: int = SHARD_COUNT,
) -> dict[str, Any]:
    repository_sha = repository_sha.strip().lower()
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        raise FullLaneShardError("shard_plan_repository_sha_invalid")
    if shard_count != SHARD_COUNT:
        raise FullLaneShardError("shard_count_invalid")
    nodeids = _collection(planned, phase="planned")
    baseline = _validate_baseline(duration_baseline)
    heavy = _validate_heavy_tests(duration_baseline, baseline)
    default_seconds = duration_baseline.get("default_seconds_per_test")
    if (
        not isinstance(default_seconds, (int, float))
        or not math.isfinite(float(default_seconds))
        or float(default_seconds) <= 0
    ):
        raise FullLaneShardError("duration_baseline_default_invalid")

    by_file: dict[str, list[str]] = defaultdict(list)
    file_order: list[str] = []
    for nodeid in nodeids:
        path = nodeid.split("::", 1)[0]
        if path not in by_file:
            file_order.append(path)
        by_file[path].append(nodeid)
    order_index = {path: index for index, path in enumerate(file_order)}

    file_estimates: dict[str, float] = {}
    for path, rows in by_file.items():
        baseline_row = baseline.get(path)
        if baseline_row is None:
            estimate = float(default_seconds) * len(rows)
        else:
            estimate = float(baseline_row["duration_seconds"]) * (
                len(rows) / int(baseline_row["test_count"])
            )
        file_estimates[path] = max(estimate, 0.000001)

    # One 21-minute file would otherwise set the floor for the whole lane:
    # schedule a file heavier than half a shard by node ID, with its recorded
    # long tests at their own measured durations and the rest sharing what is
    # left of the file's estimate.
    mean_load = sum(file_estimates.values()) / shard_count
    split_files = sorted(
        path
        for path, estimate in file_estimates.items()
        if estimate > SPLIT_SHARE_OF_MEAN_SHARD * mean_load and len(by_file[path]) > 1
    )
    units: list[tuple[str, str, float]] = []
    for path, rows in by_file.items():
        if path not in split_files:
            units.append(("files", path, file_estimates[path]))
            continue
        recorded = {nodeid: heavy[nodeid] for nodeid in rows if nodeid in heavy}
        light = [nodeid for nodeid in rows if nodeid not in recorded]
        residual = max(file_estimates[path] - sum(recorded.values()), 0.0)
        per_light = residual / len(light) if light else 0.0
        units.extend(
            ("nodeids", nodeid, max(recorded.get(nodeid, per_light), 0.000001))
            for nodeid in rows
        )

    shards: list[dict[str, Any]] = [
        {"index": index, "estimated_duration_seconds": 0.0, "files": [], "nodeids": []}
        for index in range(shard_count)
    ]
    for kind, unit, estimate in sorted(units, key=lambda row: (-row[2], row[1])):
        target = min(
            shards,
            key=lambda shard: (
                float(shard["estimated_duration_seconds"]),
                int(shard["index"]),
            ),
        )
        target[kind].append(unit)
        target["estimated_duration_seconds"] += estimate

    for shard in shards:
        shard["files"].sort(key=order_index.__getitem__)
        shard["nodeids"].sort()
        expected = _assigned_nodeids(
            planned_nodeids=nodeids, files=shard["files"], split_nodeids=shard["nodeids"]
        )
        if not expected:
            raise FullLaneShardError("shard_plan_empty_shard")
        shard["run_paths"] = sorted(
            {*shard["files"], *(nodeid.split("::", 1)[0] for nodeid in shard["nodeids"])},
            key=order_index.__getitem__,
        )
        shard["file_count"] = len(shard["run_paths"])
        shard["expected_test_count"] = len(expected)
        shard["expected_nodeids_sha256"] = _nodeids_digest(expected)
        shard["estimated_duration_seconds"] = round(
            float(shard["estimated_duration_seconds"]), 6
        )

    core = {
        "schema_version": PLAN_SCHEMA,
        "repository_sha": repository_sha,
        "strategy": PLAN_STRATEGY,
        "shard_count": shard_count,
        "planned_test_count": len(nodeids),
        "planned_nodeids_sha256": _nodeids_digest(nodeids),
        "duration_baseline_sha256": _canonical_digest(duration_baseline),
        "duration_baseline_source_sha": duration_baseline[
            "source_repository_sha"
        ],
        "duration_baseline_source_run_id": duration_baseline["source_run_id"],
        "split_files": split_files,
        "shards": shards,
    }
    return {**core, "plan_digest": _canonical_digest(core)}


def _assigned_nodeids(
    *, planned_nodeids: Sequence[str], files: Sequence[str], split_nodeids: Sequence[str]
) -> list[str]:
    file_set = set(files)
    split_set = set(split_nodeids)
    return [
        nodeid
        for nodeid in planned_nodeids
        if nodeid.split("::", 1)[0] in file_set or nodeid in split_set
    ]


def _plan_expected_nodeids(
    *, plan: Mapping[str, Any], planned_nodeids: Sequence[str], shard_index: int
) -> list[str]:
    raw_shards = plan.get("shards")
    shards = raw_shards if isinstance(raw_shards, list) else []
    matches = [
        shard
        for shard in shards
        if isinstance(shard, Mapping) and shard.get("index") == shard_index
    ]
    if len(matches) != 1:
        raise FullLaneShardError("shard_plan_index_invalid")
    raw_files = matches[0].get("files")
    raw_split = matches[0].get("nodeids")
    files = [str(value) for value in raw_files] if isinstance(raw_files, list) else []
    split = [str(value) for value in raw_split] if isinstance(raw_split, list) else []
    if (
        not (files or split)
        or len(files) != len(set(files))
        or len(split) != len(set(split))
    ):
        raise FullLaneShardError("shard_plan_files_invalid")
    return _assigned_nodeids(
        planned_nodeids=planned_nodeids, files=files, split_nodeids=split
    )


def verify_shard(
    *,
    planned_path: Path,
    duration_baseline_path: Path,
    plan_path: Path,
    executed_path: Path,
    junit_path: Path,
    repository_sha: str,
    shard_index: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    input_paths = (
        planned_path,
        duration_baseline_path,
        plan_path,
        executed_path,
        junit_path,
    )
    symlinks = sorted(path.name for path in input_paths if path.is_symlink())
    if symlinks:
        blockers.append("shard_artifact_symlinks_forbidden:" + ",".join(symlinks))
    planned_payload = _read_json(planned_path)
    baseline_payload = _read_json(duration_baseline_path)
    plan_payload = _read_json(plan_path)
    try:
        planned_nodeids = _collection(planned_payload, phase="planned")
        expected_plan = build_shard_plan(
            planned=planned_payload,
            duration_baseline=baseline_payload,
            repository_sha=repository_sha,
        )
        if plan_payload != expected_plan:
            blockers.append("shard_plan_recomputation_mismatch")
        expected_nodeids = _plan_expected_nodeids(
            plan=expected_plan,
            planned_nodeids=planned_nodeids,
            shard_index=shard_index,
        )
    except FullLaneShardError as exc:
        blockers.append(str(exc))
        expected_nodeids = []

    executed_nodeids: list[str] = []
    try:
        executed_nodeids = _collection(_read_json(executed_path), phase="executed")
    except (OSError, UnicodeError, json.JSONDecodeError, FullLaneShardError) as exc:
        blockers.append(f"shard_executed_invalid:{exc}")
    if executed_nodeids != expected_nodeids:
        blockers.append("shard_executed_nodeids_mismatch")

    outcomes: dict[str, Any] = {}
    try:
        outcomes = _junit(junit_path)
    except FullLaneShardError as exc:
        blockers.append(str(exc))
    if outcomes:
        if outcomes["duplicates"]:
            blockers.append("shard_junit_duplicate_nodeids")
        if outcomes["nodeids"] != sorted(expected_nodeids):
            blockers.append("shard_junit_nodeids_mismatch")
        if outcomes["failures"]:
            blockers.append(f"shard_junit_failures:{outcomes['failures']}")
        if outcomes["errors"]:
            blockers.append(f"shard_junit_errors:{outcomes['errors']}")
        if outcomes["skipped"]:
            blockers.append(f"shard_junit_skipped:{outcomes['skipped']}")

    blockers = sorted(set(blockers))
    artifacts = {
        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in input_paths
        if path.is_file() and not path.is_symlink()
    }
    return {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "passed" if not blockers else "blocked",
        "repository_sha": repository_sha,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "expected_test_count": len(expected_nodeids),
        "executed_test_count": len(executed_nodeids),
        "junit_test_count": len(outcomes.get("rows", [])),
        "failure_count": int(outcomes.get("failures", 0)),
        "error_count": int(outcomes.get("errors", 0)),
        "skipped_count": int(outcomes.get("skipped", 0)),
        "artifacts": artifacts,
        "blockers": blockers,
    }


def _write_aggregate_junit(
    *, paths: Sequence[Path], nodeids: Sequence[str], output: Path
) -> None:
    rows: dict[str, Any] = {}
    total_duration = 0.0
    for path in paths:
        outcomes = _junit(path)
        for row in outcomes["rows"]:
            nodeid = str(row["nodeid"])
            if nodeid in rows:
                raise FullLaneShardError(f"aggregate_junit_duplicate_nodeid:{nodeid}")
            rows[nodeid] = row["element"]
            total_duration += float(row["duration_seconds"])
    if set(rows) != set(nodeids):
        raise FullLaneShardError("aggregate_junit_nodeids_mismatch")
    root = WritableElementTree.Element("testsuites")
    suite = WritableElementTree.SubElement(
        root,
        "testsuite",
        {
            "name": "full-test-lane-sharded",
            "tests": str(len(nodeids)),
            "failures": "0",
            "errors": "0",
            "skipped": "0",
            "time": f"{total_duration:.6f}",
        },
    )
    for nodeid in nodeids:
        suite.append(copy.deepcopy(rows[nodeid]))
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = WritableElementTree.ElementTree(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def aggregate_shards(
    *, shard_root: Path, output_dir: Path, repository_sha: str
) -> dict[str, Any]:
    receipt_paths = sorted(shard_root.rglob("full-test-lane-shard-verification.json"))
    if len(receipt_paths) != SHARD_COUNT:
        raise FullLaneShardError("aggregate_shard_receipt_count_invalid")
    by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    for receipt_path in receipt_paths:
        receipt = _read_json(receipt_path)
        index = receipt.get("shard_index")
        if not isinstance(index, int) or index in by_index:
            raise FullLaneShardError("aggregate_shard_indices_invalid")
        by_index[index] = (receipt_path.parent, receipt)
    if sorted(by_index) != list(range(SHARD_COUNT)):
        raise FullLaneShardError("aggregate_shard_indices_invalid")

    first_dir = by_index[0][0]
    planned_path = first_dir / "full-test-lane-planned.json"
    baseline_path = first_dir / "full-test-lane-duration-baseline.json"
    plan_path = first_dir / "full-test-lane-shard-plan.json"
    planned_payload = _read_json(planned_path)
    planned_nodeids = _collection(planned_payload, phase="planned")
    baseline_payload = _read_json(baseline_path)
    plan_payload = _read_json(plan_path)
    expected_plan = build_shard_plan(
        planned=planned_payload,
        duration_baseline=baseline_payload,
        repository_sha=repository_sha,
    )
    if plan_payload != expected_plan:
        raise FullLaneShardError("aggregate_shard_plan_invalid")

    shard_receipts: list[dict[str, Any]] = []
    junit_paths: list[Path] = []
    union: list[str] = []
    seen: set[str] = set()
    for index in range(SHARD_COUNT):
        directory, retained_receipt = by_index[index]
        local_planned = directory / "full-test-lane-planned.json"
        local_baseline = directory / "full-test-lane-duration-baseline.json"
        local_plan = directory / "full-test-lane-shard-plan.json"
        executed = directory / "full-test-lane-shard-executed.json"
        junit = directory / "full-test-lane-shard-junit.xml"
        if _sha256(local_planned) != _sha256(planned_path):
            raise FullLaneShardError("aggregate_planned_manifest_mismatch")
        if _sha256(local_baseline) != _sha256(baseline_path):
            raise FullLaneShardError("aggregate_duration_baseline_mismatch")
        if _sha256(local_plan) != _sha256(plan_path):
            raise FullLaneShardError("aggregate_shard_plan_mismatch")
        recomputed = verify_shard(
            planned_path=local_planned,
            duration_baseline_path=local_baseline,
            plan_path=local_plan,
            executed_path=executed,
            junit_path=junit,
            repository_sha=repository_sha,
            shard_index=index,
        )
        if retained_receipt != recomputed or recomputed["status"] != "passed":
            raise FullLaneShardError(f"aggregate_shard_verification_invalid:{index}")
        nodeids = _collection(_read_json(executed), phase="executed")
        duplicates = seen.intersection(nodeids)
        if duplicates:
            raise FullLaneShardError("aggregate_shard_nodeids_duplicate")
        seen.update(nodeids)
        union.extend(nodeids)
        shard_receipts.append(recomputed)
        junit_paths.append(junit)

    if seen != set(planned_nodeids) or len(union) != len(planned_nodeids):
        raise FullLaneShardError("aggregate_shard_union_mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_planned = output_dir / "full-test-lane-planned.json"
    canonical_executed = output_dir / "full-test-lane-executed.json"
    canonical_junit = output_dir / "full-test-lane-junit.xml"
    canonical_baseline = output_dir / "full-test-lane-duration-baseline.json"
    canonical_plan = output_dir / "full-test-lane-shard-plan.json"
    _write_json_atomic(canonical_planned, planned_payload)
    _write_json_atomic(
        canonical_executed,
        {
            "schema_version": COLLECTION_SCHEMA,
            "phase": "executed",
            "test_count": len(planned_nodeids),
            "nodeids_sha256": _nodeids_digest(planned_nodeids),
            "nodeids": planned_nodeids,
        },
    )
    _write_json_atomic(canonical_baseline, baseline_payload)
    _write_json_atomic(canonical_plan, plan_payload)
    _write_aggregate_junit(
        paths=junit_paths, nodeids=planned_nodeids, output=canonical_junit
    )

    retained_shards = output_dir / "shards"
    for index in range(SHARD_COUNT):
        destination = retained_shards / f"shard-{index}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(by_index[index][0], destination)

    canonical_files = (
        canonical_planned,
        canonical_executed,
        canonical_junit,
        canonical_baseline,
        canonical_plan,
    )
    receipt = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "passed",
        "repository_sha": repository_sha,
        "strategy": PLAN_STRATEGY,
        "shard_count": SHARD_COUNT,
        "test_count": len(planned_nodeids),
        "planned_nodeids_sha256": _nodeids_digest(planned_nodeids),
        "plan_digest": plan_payload["plan_digest"],
        "zero_duplicates": True,
        "zero_omissions": True,
        "zero_failures_errors_and_skips": True,
        "shards": shard_receipts,
        "canonical_artifacts": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in canonical_files
        },
        "blockers": [],
    }
    _write_json_atomic(output_dir / "full-test-lane-shard-aggregate.json", receipt)
    return receipt


def validate_sharded_artifact(
    artifact_dir: Path, *, repository_sha: str
) -> dict[str, Any]:
    retained = artifact_dir / "full-test-lane-shard-aggregate.json"
    if retained.is_symlink() or not retained.is_file():
        raise FullLaneShardError("shard_aggregate_receipt_missing")
    retained_payload = _read_json(retained)
    with tempfile.TemporaryDirectory(prefix="blueprint-shard-recompute-") as temporary:
        recomputed = aggregate_shards(
            shard_root=artifact_dir / "shards",
            output_dir=Path(temporary),
            repository_sha=repository_sha,
        )
    if retained_payload != recomputed:
        raise FullLaneShardError("shard_aggregate_receipt_mismatch")
    for name, metadata in retained_payload["canonical_artifacts"].items():
        path = artifact_dir / str(name)
        if path.is_symlink() or not path.is_file():
            raise FullLaneShardError(f"shard_aggregate_artifact_missing:{name}")
        if (
            _sha256(path) != metadata.get("sha256")
            or path.stat().st_size != metadata.get("size_bytes")
        ):
            raise FullLaneShardError(f"shard_aggregate_artifact_mismatch:{name}")
    return retained_payload


def _command_baseline(args: argparse.Namespace) -> int:
    payload = build_duration_baseline(
        junit=args.junit.resolve(),
        source_sha=args.source_sha,
        source_run_id=args.source_run_id,
    )
    _write_json_atomic(args.output.resolve(), payload)
    print(
        f"[full-lane-shards] baseline tests={payload['test_count']} "
        f"files={payload['file_count']}"
    )
    return 0


def _command_plan(args: argparse.Namespace) -> int:
    payload = build_shard_plan(
        planned=_read_json(args.planned.resolve()),
        duration_baseline=_read_json(args.duration_baseline.resolve()),
        repository_sha=args.repository_sha,
    )
    _write_json_atomic(args.output.resolve(), payload)
    print(
        "[full-lane-shards] plan "
        + " ".join(
            f"shard{row['index']}={row['expected_test_count']}tests/"
            f"{row['estimated_duration_seconds']}s"
            for row in payload["shards"]
        )
    )
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    try:
        payload = verify_shard(
            planned_path=args.planned.resolve(),
            duration_baseline_path=args.duration_baseline.resolve(),
            plan_path=args.plan.resolve(),
            executed_path=args.executed.resolve(),
            junit_path=args.junit.resolve(),
            repository_sha=args.repository_sha,
            shard_index=args.shard_index,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, FullLaneShardError) as exc:
        _write_json_atomic(
            output,
            {
                "schema_version": SHARD_RECEIPT_SCHEMA,
                "status": "blocked",
                "repository_sha": args.repository_sha,
                "shard_index": args.shard_index,
                "shard_count": SHARD_COUNT,
                "expected_test_count": 0,
                "executed_test_count": 0,
                "junit_test_count": 0,
                "failure_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "artifacts": {},
                "blockers": [f"shard_verification_error:{exc}"],
            },
        )
        raise
    _write_json_atomic(output, payload)
    print(
        f"[full-lane-shards] shard={args.shard_index} status={payload['status']} "
        f"tests={payload['executed_test_count']}"
    )
    for blocker in payload["blockers"]:
        print(f"[full-lane-shards] blocker={blocker}", file=sys.stderr)
    return 0 if payload["status"] == "passed" else 1


def _command_aggregate(args: argparse.Namespace) -> int:
    shard_root = args.shard_root.resolve()
    output_dir = args.output_dir.resolve()
    try:
        payload = aggregate_shards(
            shard_root=shard_root,
            output_dir=output_dir,
            repository_sha=args.repository_sha,
        )
    except FullLaneShardError as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        retained_shards = output_dir / "shards"
        for receipt_path in sorted(
            shard_root.rglob("full-test-lane-shard-verification.json")
        ):
            try:
                receipt = _read_json(receipt_path)
            except (OSError, UnicodeError, json.JSONDecodeError, FullLaneShardError):
                continue
            index = receipt.get("shard_index")
            if not isinstance(index, int) or index < 0 or index >= SHARD_COUNT:
                continue
            destination = retained_shards / f"shard-{index}"
            if not destination.exists():
                shutil.copytree(receipt_path.parent, destination)
        _write_json_atomic(
            output_dir / "full-test-lane-shard-aggregate.json",
            {
                "schema_version": AGGREGATE_SCHEMA,
                "status": "blocked",
                "repository_sha": args.repository_sha,
                "strategy": PLAN_STRATEGY,
                "shard_count": SHARD_COUNT,
                "blockers": [str(exc)],
            },
        )
        raise
    print(
        f"[full-lane-shards] aggregate status={payload['status']} "
        f"tests={payload['test_count']}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--junit", type=Path, required=True)
    baseline.add_argument("--source-sha", required=True)
    baseline.add_argument("--source-run-id", type=int, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.set_defaults(handler=_command_baseline)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--planned", type=Path, required=True)
    plan.add_argument("--duration-baseline", type=Path, required=True)
    plan.add_argument("--repository-sha", required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(handler=_command_plan)

    verify = subparsers.add_parser("verify-shard")
    verify.add_argument("--planned", type=Path, required=True)
    verify.add_argument("--duration-baseline", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--executed", type=Path, required=True)
    verify.add_argument("--junit", type=Path, required=True)
    verify.add_argument("--repository-sha", required=True)
    verify.add_argument("--shard-index", type=int, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(handler=_command_verify)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--shard-root", type=Path, required=True)
    aggregate.add_argument("--repository-sha", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.set_defaults(handler=_command_aggregate)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        FullLaneShardError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"[full-lane-shards] ERROR {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
