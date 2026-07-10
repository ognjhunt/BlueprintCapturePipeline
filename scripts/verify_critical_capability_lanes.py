#!/usr/bin/env python3
"""Validate critical-lane policy or evaluate scope-bound lane evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast


POLICY_SCHEMA = "blueprint.critical_capability_lanes.v1"
EVIDENCE_SCHEMA = "blueprint.critical_capability_lane_evidence.v1"
KNOWN_SCOPES = {"BASE", "SIM", "PTDP", "SC3", "PAID", "LIVE"}
LANE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
REPOSITORY_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ARTIFACT_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
RAW_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_URI_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:([0-9a-f]{64})$")
EXPECTED_LANE_SCOPES = {
    "cpu_full": KNOWN_SCOPES,
    "container_production": KNOWN_SCOPES,
    "gpu_provider_canary": {"SC3", "PAID", "LIVE"},
    "pubsub_emulator_integration": {"PAID", "LIVE"},
    "native_lerobot_export": {"PTDP", "SC3", "PAID", "LIVE"},
}
COMMON_EVIDENCE_FIELDS = {
    "schema_version",
    "lane_id",
    "evidence_schema_version",
    "generated_at",
    "repository_sha",
    "status",
    "executed",
    "skipped_count",
    "artifact_digests",
    "artifact_sizes",
    "blockers",
    "claim_boundary",
}
LANE_FIELDS = {
    "cpu_full": {
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
    },
    "container_production": {
        "production_image_id",
        "development_image_id",
        "production_nonroot_user",
        "development_nonroot_user",
        "compose_config_valid",
        "systemd_security_passed",
        "production_runtime_smoke_passed",
        "development_runtime_smoke_passed",
    },
    "gpu_provider_canary": {
        "image_uri",
        "approved_image_uri",
        "image_digest",
        "canary_result_schema_version",
        "canary_status",
        "spend_limits",
        "result_contract",
        "estimated_cost_usd",
        "selected_machine_id",
        "selected_gpu_ram_mb",
        "selected_hourly_rate_usd",
        "actual_live_runtime_seconds",
        "raw_artifact_digests",
        "source_manifest_digest",
        "source_bundle_digest",
    },
    "pubsub_emulator_integration": {
        "emulator_loopback_only",
        "published_message_id",
        "round_trip_payload_received",
        "message_acknowledged",
        "cleanup_succeeded",
        "round_trip_payload_sha256",
    },
    "native_lerobot_export": {
        "export_dir",
        "export_file_count",
        "export_total_bytes",
        "export_tree_sha256",
        "validation_report",
    },
}
EXPECTED_ARTIFACT_KEYS = {
    "cpu_full": {
        "full-test-lane-planned.json",
        "full-test-lane-executed.json",
        "full-test-lane-junit.xml",
    },
    "container_production": {
        "production_image",
        "development_image",
        "production_user",
        "development_user",
        "compose_config",
        "compose_sentinel",
        "systemd_security_log",
        "systemd_security_sentinel",
        "production_runtime_smoke",
        "development_runtime_smoke",
    },
    "gpu_provider_canary": {
        "vast_unitree_groot_sonic_image_canary_result.json",
        "vast_unitree_groot_sonic_image_canary_safe_summary.json",
        "vast_provider_adapter_result.json",
        "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report.json",
        "vast_provider_command_result.json",
        "vast_teardown_manifest.json",
        "vast_budget_ledger.json",
        "vast_provider_runtime_output.json",
    },
    "pubsub_emulator_integration": {"pubsub_emulator_round_trip_source.json"},
    "native_lerobot_export": {"native_lerobot_export_source_manifest.json"},
}
GPU_RAW_ARTIFACT_KEYS = (
    EXPECTED_ARTIFACT_KEYS["gpu_provider_canary"] - {"vast_provider_runtime_output.json"}
) | {"vast_provider_runtime_output.zip"}
GPU_SOURCE_MANIFEST_NAME = "gpu-canary-source-manifest.json"
GPU_SOURCE_BUNDLE_NAME = "gpu-provider-canary-sanitized-evidence.zip"
CONTAINER_SOURCE_DIR_NAME = "container-production-sources"
MAX_EVIDENCE_AGE = timedelta(hours=24)
MAX_CLOCK_SKEW = timedelta(minutes=5)
MAX_SANITIZED_SOURCE_SIZE = 4 * 1024 * 1024
MAX_SANITIZED_BUNDLE_SIZE = 40 * 1024 * 1024
MAX_RAW_GPU_SOURCE_SIZE = 64 * 1024 * 1024
PUBSUB_SOURCE_NAME = "pubsub_emulator_round_trip_source.json"
PUBSUB_SOURCE_SCHEMA = "blueprint.pubsub_emulator_round_trip_source.v1"
MAX_PUBSUB_SOURCE_SIZE = 64 * 1024
NATIVE_SOURCE_NAME = "native_lerobot_export_source_manifest.json"
NATIVE_SOURCE_SCHEMA = "blueprint.native_lerobot_export_source_manifest.v1"
MAX_NATIVE_SOURCE_SIZE = 32 * 1024 * 1024


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _workflow_job_check_names(text: str) -> set[str]:
    """Read job display names without requiring the repository dependencies.

    This policy gate deliberately runs on the stock GitHub runner Python. The
    constrained parser recognizes the checked-in workflow shape and, unlike a
    loose ``line.strip()`` scan, cannot mistake a step display name for a
    branch-protection check name.
    """

    names: set[str] = set()
    in_jobs = False
    current_job = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            in_jobs = stripped == "jobs:"
            current_job = False
            continue
        if not in_jobs:
            continue
        if indent == 2:
            current_job = re.fullmatch(r"[A-Za-z0-9_-]+:", stripped) is not None
            continue
        if indent == 4 and current_job and stripped.startswith("name:"):
            name = stripped.removeprefix("name:").strip().strip("'\"")
            if name:
                names.add(name)
    return names


def validate_policy(*, root: Path, policy: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if policy.get("schema_version") != POLICY_SCHEMA:
        blockers.append("policy_schema_version_invalid")
    lanes = _mapping(policy.get("lanes"))
    if not lanes:
        blockers.append("critical_lanes_missing")
    check_names: set[str] = set()
    for path in sorted((root / ".github" / "workflows").glob("*.yml")):
        try:
            check_names.update(_workflow_job_check_names(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError):
            blockers.append(f"critical_lane_workflow_unreadable:{path.name}")
    for lane_id, raw_lane in sorted(lanes.items()):
        lane = _mapping(raw_lane)
        if LANE_ID_PATTERN.fullmatch(lane_id) is None:
            blockers.append(f"lane_id_invalid:{lane_id}")
        scopes = lane.get("required_for_scopes")
        if (
            not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(scope, str) and scope in KNOWN_SCOPES for scope in scopes)
            or len(scopes) != len(set(scopes))
        ):
            blockers.append(f"lane_scopes_invalid:{lane_id}")
        if lane.get("skip_is_blocker") is not True:
            blockers.append(f"lane_skip_policy_not_fail_closed:{lane_id}")
        if not str(lane.get("evidence_schema_version") or "").strip():
            blockers.append(f"lane_evidence_schema_missing:{lane_id}")
        check_name = str(lane.get("required_check_name") or "")
        if not check_name or check_name not in check_names:
            blockers.append(f"lane_required_check_missing:{lane_id}:{check_name or 'unset'}")
        command = str(lane.get("local_contract_command") or "")
        if not command:
            blockers.append(f"lane_contract_command_missing:{lane_id}")
        else:
            referenced_scripts = re.findall(r"\b(scripts/[A-Za-z0-9_./-]+)", command)
            for script in referenced_scripts:
                if not (root / script).is_file():
                    blockers.append(f"lane_contract_script_missing:{lane_id}:{script}")
    for lane_id, expected_scopes in sorted(EXPECTED_LANE_SCOPES.items()):
        lane = _mapping(lanes.get(lane_id))
        if not lane:
            blockers.append(f"mandatory_critical_lane_missing:{lane_id}")
            continue
        raw_scopes = lane.get("required_for_scopes")
        actual_scopes = (
            set(raw_scopes)
            if isinstance(raw_scopes, list) and all(isinstance(scope, str) for scope in raw_scopes)
            else set()
        )
        if actual_scopes != expected_scopes:
            blockers.append(f"mandatory_critical_lane_scope_mismatch:{lane_id}")
    return blockers


def _parse_aware_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _valid_aware_timestamp(value: object) -> bool:
    return _parse_aware_timestamp(value) is not None


def _file_sha256(path: Path, *, max_size: int) -> tuple[str, int] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if not 0 < size <= max_size:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return f"sha256:{digest.hexdigest()}", size


def _validate_artifacts(lane_id: str, row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected = EXPECTED_ARTIFACT_KEYS[lane_id]
    digests = row.get("artifact_digests")
    sizes = row.get("artifact_sizes")
    if not isinstance(digests, Mapping) or set(digests) != expected:
        blockers.append(f"critical_lane_artifact_digest_keys_invalid:{lane_id}")
    else:
        for name, value in digests.items():
            if ARTIFACT_DIGEST_PATTERN.fullmatch(str(value or "")) is None:
                blockers.append(f"critical_lane_artifact_digest_invalid:{lane_id}:{name}")
    if not isinstance(sizes, Mapping) or set(sizes) != expected:
        blockers.append(f"critical_lane_artifact_size_keys_invalid:{lane_id}")
    else:
        for name, value in sizes.items():
            if type(value) is not int or value < 0:
                blockers.append(f"critical_lane_artifact_size_invalid:{lane_id}:{name}")
    return blockers


def _validate_cpu_full(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    counts = {
        name: row.get(name)
        for name in (
            "test_count",
            "passed_count",
            "failure_count",
            "error_count",
            "planned_test_count",
            "executed_test_count",
        )
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        blockers.append("critical_lane_cpu_counts_invalid")
    else:
        typed_counts = {name: cast(int, value) for name, value in counts.items()}
        if typed_counts["test_count"] <= 0:
            blockers.append("critical_lane_cpu_test_count_empty")
        if not (
            typed_counts["test_count"]
            == typed_counts["passed_count"]
            == typed_counts["planned_test_count"]
            == typed_counts["executed_test_count"]
        ):
            blockers.append("critical_lane_cpu_test_counts_mismatch")
        if typed_counts["failure_count"] or typed_counts["error_count"]:
            blockers.append("critical_lane_cpu_nonpassing_outcomes")
    for name in ("nodeids_sha256", "testcase_outcomes_sha256"):
        if RAW_DIGEST_PATTERN.fullmatch(str(row.get(name) or "")) is None:
            blockers.append(f"critical_lane_cpu_digest_invalid:{name}")
    if row.get("skipped_testcases") != []:
        blockers.append("critical_lane_cpu_skip_details_not_empty")
    if row.get("skipped_testcases_truncated") is not False:
        blockers.append("critical_lane_cpu_skip_details_truncated")
    return blockers


def _validate_container(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for name in ("production_image_id", "development_image_id"):
        if ARTIFACT_DIGEST_PATTERN.fullmatch(str(row.get(name) or "")) is None:
            blockers.append(f"critical_lane_container_image_id_invalid:{name}")
    for name in (
        "production_nonroot_user",
        "development_nonroot_user",
        "compose_config_valid",
        "systemd_security_passed",
        "production_runtime_smoke_passed",
        "development_runtime_smoke_passed",
    ):
        if row.get(name) is not True:
            blockers.append(f"critical_lane_container_check_not_passed:{name}")
    return blockers


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_gpu_source_bundle(row: Mapping[str, Any], *, evidence_dir: Path) -> list[str]:
    blockers: list[str] = []
    raw_digests = row.get("raw_artifact_digests")
    if not isinstance(raw_digests, Mapping) or set(raw_digests) != GPU_RAW_ARTIFACT_KEYS:
        blockers.append("critical_lane_gpu_raw_digest_keys_invalid")
    else:
        for name, value in raw_digests.items():
            if ARTIFACT_DIGEST_PATTERN.fullmatch(str(value or "")) is None:
                blockers.append(f"critical_lane_gpu_raw_digest_invalid:{name}")

    source_manifest_digest = str(row.get("source_manifest_digest") or "")
    source_bundle_digest = str(row.get("source_bundle_digest") or "")
    if ARTIFACT_DIGEST_PATTERN.fullmatch(source_manifest_digest) is None:
        blockers.append("critical_lane_gpu_source_manifest_digest_invalid")
    if ARTIFACT_DIGEST_PATTERN.fullmatch(source_bundle_digest) is None:
        blockers.append("critical_lane_gpu_source_bundle_digest_invalid")
    manifest_path = evidence_dir / GPU_SOURCE_MANIFEST_NAME
    bundle_path = evidence_dir / GPU_SOURCE_BUNDLE_NAME
    manifest_file = _file_sha256(manifest_path, max_size=MAX_SANITIZED_SOURCE_SIZE)
    bundle_file = _file_sha256(bundle_path, max_size=MAX_SANITIZED_BUNDLE_SIZE)
    if manifest_file is None:
        blockers.append("critical_lane_gpu_source_manifest_missing_or_unsafe")
        return blockers
    if bundle_file is None:
        blockers.append("critical_lane_gpu_source_bundle_missing_or_unsafe")
        return blockers
    if manifest_file[0] != source_manifest_digest:
        blockers.append("critical_lane_gpu_source_manifest_digest_mismatch")
    if bundle_file[0] != source_bundle_digest:
        blockers.append("critical_lane_gpu_source_bundle_digest_mismatch")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        blockers.append("critical_lane_gpu_source_manifest_malformed")
        return blockers
    manifest = _mapping(manifest_value)
    if set(manifest) != {
        "schema_version",
        "sanitized_artifact_digests",
        "sanitized_artifact_sizes",
        "raw_artifact_digests",
        "raw_artifact_sizes",
        "claim_boundary",
    }:
        blockers.append("critical_lane_gpu_source_manifest_fields_invalid")
    if manifest.get("schema_version") != "blueprint.gpu_canary_sanitized_source_manifest.v1":
        blockers.append("critical_lane_gpu_source_manifest_schema_invalid")
    sanitized_digests = _mapping(manifest.get("sanitized_artifact_digests"))
    sanitized_sizes = _mapping(manifest.get("sanitized_artifact_sizes"))
    if sanitized_digests != _mapping(row.get("artifact_digests")):
        blockers.append("critical_lane_gpu_manifest_sanitized_digests_mismatch")
    if sanitized_sizes != _mapping(row.get("artifact_sizes")):
        blockers.append("critical_lane_gpu_manifest_sanitized_sizes_mismatch")
    if _mapping(manifest.get("raw_artifact_digests")) != _mapping(raw_digests):
        blockers.append("critical_lane_gpu_manifest_raw_digests_mismatch")
    raw_sizes = manifest.get("raw_artifact_sizes")
    if not isinstance(raw_sizes, Mapping) or set(raw_sizes) != GPU_RAW_ARTIFACT_KEYS:
        blockers.append("critical_lane_gpu_manifest_raw_sizes_invalid")
    else:
        for name, value in raw_sizes.items():
            if type(value) is not int or not 0 < value <= MAX_RAW_GPU_SOURCE_SIZE:
                blockers.append(f"critical_lane_gpu_manifest_raw_size_invalid:{name}")
    if _mapping(manifest.get("claim_boundary")) != {
        "sanitized_projection_retained": True,
        "raw_signed_urls_and_commands_excluded": True,
        "manifest_does_not_prove_policy_inference": True,
    }:
        blockers.append("critical_lane_gpu_source_manifest_boundary_invalid")

    source_payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(EXPECTED_ARTIFACT_KEYS["gpu_provider_canary"]):
        path = evidence_dir / name
        actual = _file_sha256(path, max_size=MAX_SANITIZED_SOURCE_SIZE)
        if actual is None:
            blockers.append(f"critical_lane_gpu_sanitized_source_missing_or_unsafe:{name}")
            continue
        if actual[0] != sanitized_digests.get(name):
            blockers.append(f"critical_lane_gpu_sanitized_source_digest_mismatch:{name}")
        if actual[1] != sanitized_sizes.get(name):
            blockers.append(f"critical_lane_gpu_sanitized_source_size_mismatch:{name}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append(f"critical_lane_gpu_sanitized_source_malformed:{name}")
        else:
            source_payloads[name] = _mapping(value)

    expected_bundle_names = EXPECTED_ARTIFACT_KEYS["gpu_provider_canary"] | {
        GPU_SOURCE_MANIFEST_NAME
    }
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != expected_bundle_names:
                blockers.append("critical_lane_gpu_source_bundle_members_invalid")
            for info in infos:
                if (
                    info.is_dir()
                    or info.file_size <= 0
                    or info.file_size > MAX_SANITIZED_SOURCE_SIZE
                ):
                    blockers.append(
                        f"critical_lane_gpu_source_bundle_member_size_invalid:{info.filename}"
                    )
                    continue
                content = archive.read(info)
                if info.filename == GPU_SOURCE_MANIFEST_NAME:
                    try:
                        manifest_bytes = manifest_path.read_bytes()
                    except OSError:
                        manifest_bytes = b""
                    if content != manifest_bytes:
                        blockers.append("critical_lane_gpu_source_bundle_manifest_mismatch")
                elif info.filename in EXPECTED_ARTIFACT_KEYS["gpu_provider_canary"]:
                    digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
                    if digest != sanitized_digests.get(info.filename):
                        blockers.append(
                            f"critical_lane_gpu_source_bundle_digest_mismatch:{info.filename}"
                        )
                    if len(content) != sanitized_sizes.get(info.filename):
                        blockers.append(
                            f"critical_lane_gpu_source_bundle_size_mismatch:{info.filename}"
                        )
    except (OSError, zipfile.BadZipFile, RuntimeError):
        blockers.append("critical_lane_gpu_source_bundle_malformed")

    raw_result = source_payloads.get("vast_unitree_groot_sonic_image_canary_result.json", {})
    expected_source_values = {
        "schema_version": row.get("canary_result_schema_version"),
        "status": row.get("canary_status"),
        "public_image": row.get("image_uri"),
        "selected_hourly_rate_usd": row.get("selected_hourly_rate_usd"),
        "actual_live_runtime_seconds": row.get("actual_live_runtime_seconds"),
        "estimated_cost_usd": row.get("estimated_cost_usd"),
        "continuing_spend_from_this_run": False,
        "blocker_count": 0,
        "raw_secret_values_recorded": False,
    }
    for name, expected in expected_source_values.items():
        if raw_result.get(name) != expected:
            blockers.append(f"critical_lane_gpu_retained_result_mismatch:{name}")
    selected_offer = _mapping(raw_result.get("selected_offer"))
    if selected_offer != {
        "machine_id": row.get("selected_machine_id"),
        "gpu_ram_mb": row.get("selected_gpu_ram_mb"),
        "hourly_rate_usd": row.get("selected_hourly_rate_usd"),
    }:
        blockers.append("critical_lane_gpu_retained_offer_mismatch")
    for name, expected in _mapping(row.get("result_contract")).items():
        if raw_result.get(name) != expected:
            blockers.append(f"critical_lane_gpu_retained_contract_mismatch:{name}")

    provider_output = source_payloads.get("vast_provider_runtime_output.json", {})
    if provider_output.get("schema_version") != (
        "unitree_groot_n17_sonic_policy_provider_output.v1"
    ):
        blockers.append("critical_lane_gpu_retained_provider_output_schema_invalid")
    if provider_output.get("status") != "completed":
        blockers.append("critical_lane_gpu_retained_provider_output_status_invalid")
    if (
        provider_output.get("canary_only") is not True
        or provider_output.get("canary_marker")
        != "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_IMAGE_CANARY_OK"
    ):
        blockers.append("critical_lane_gpu_retained_provider_output_marker_invalid")
    if provider_output.get("blocker_count") != 0:
        blockers.append("critical_lane_gpu_retained_provider_output_blocked")
    for name in (
        "unitree_groot_n17_sonic_model_executed",
        "unitree_groot_n17_sonic_policy_action_command_ran",
        "policy_action_model_command_ran",
        "raw_credentials_written_to_artifacts",
        "secret_hashes_written_to_artifacts",
    ):
        if provider_output.get(name) is not False:
            blockers.append(f"critical_lane_gpu_retained_provider_boundary_invalid:{name}")
    for name in ("python", "nvidia_smi"):
        check = _mapping(_mapping(provider_output.get("checks")).get(name))
        if check.get("returncode") != 0 or _finite_number(check.get("duration_seconds")) is None:
            blockers.append(f"critical_lane_gpu_retained_provider_check_invalid:{name}")
    return blockers


def _validate_gpu_canary(
    row: Mapping[str, Any], *, expected_image_uri: str | None, evidence_dir: Path
) -> list[str]:
    blockers: list[str] = []
    image_uri = str(row.get("image_uri") or "")
    approved_image_uri = str(row.get("approved_image_uri") or "")
    image_match = IMAGE_URI_PATTERN.fullmatch(image_uri)
    if image_match is None:
        blockers.append("critical_lane_gpu_image_invalid")
    if approved_image_uri != image_uri:
        blockers.append("critical_lane_gpu_image_not_approved")
    if expected_image_uri is None:
        blockers.append("critical_lane_gpu_expected_image_missing")
    elif image_uri != expected_image_uri:
        blockers.append("critical_lane_gpu_wrong_expected_image")
    expected_digest = f"sha256:{image_match.group(1)}" if image_match else None
    if row.get("image_digest") != expected_digest:
        blockers.append("critical_lane_gpu_image_digest_mismatch")
    if row.get("canary_result_schema_version") != ("unitree_groot_n17_sonic_vast_image_canary.v1"):
        blockers.append("critical_lane_gpu_result_schema_invalid")
    if row.get("canary_status") != "completed":
        blockers.append("critical_lane_gpu_canary_not_completed")
    limits = _mapping(row.get("spend_limits"))
    hourly_limit = _finite_number(limits.get("max_hourly_rate_usd"))
    target = _finite_number(limits.get("target_spend_usd"))
    hard_cap = _finite_number(limits.get("hard_cap_usd"))
    live_minutes = limits.get("max_live_minutes")
    timeout_seconds = limits.get("startup_timeout_seconds")
    if hourly_limit is None or not 0 < hourly_limit <= 1.0:
        blockers.append("critical_lane_gpu_hourly_limit_invalid")
    if target is None or hard_cap is None or not 0 < target <= hard_cap <= 1.0:
        blockers.append("critical_lane_gpu_spend_limits_invalid")
    if type(live_minutes) is not int or not 1 <= live_minutes <= 30:
        blockers.append("critical_lane_gpu_live_minutes_invalid")
    if (
        hourly_limit is not None
        and hard_cap is not None
        and type(live_minutes) is int
        and live_minutes > 0
        and hourly_limit * live_minutes / 60.0 > hard_cap
    ):
        blockers.append("critical_lane_gpu_projected_cost_exceeds_hard_cap")
    if (
        type(timeout_seconds) is not int
        or not 60 <= timeout_seconds <= 1800
        or type(live_minutes) is not int
        or timeout_seconds > live_minutes * 60
    ):
        blockers.append("critical_lane_gpu_timeout_invalid")
    result_contract = _mapping(row.get("result_contract"))
    required_true = {
        "heartbeat_completed",
        "gpu_sanity_completed",
        "provider_bundle_downloaded_and_ran",
        "provider_output_upload_ok",
        "provider_runtime_output_zip_produced",
        "canary_marker_observed",
    }
    for name in required_true:
        if result_contract.get(name) is not True:
            blockers.append(f"critical_lane_gpu_result_not_true:{name}")
    if result_contract.get("continuing_spend_from_this_run") is not False:
        blockers.append("critical_lane_gpu_continuing_spend")
    hourly = _finite_number(row.get("selected_hourly_rate_usd"))
    duration = _finite_number(row.get("actual_live_runtime_seconds"))
    cost = _finite_number(row.get("estimated_cost_usd"))
    if hourly is None or hourly_limit is None or not 0 < hourly <= hourly_limit:
        blockers.append("critical_lane_gpu_selected_hourly_rate_invalid")
    if duration is None or type(live_minutes) is not int or not 0 < duration <= live_minutes * 60:
        blockers.append("critical_lane_gpu_runtime_invalid")
    if cost is None or hard_cap is None or not 0 <= cost <= hard_cap:
        blockers.append("critical_lane_gpu_cost_invalid")
    elif hourly is not None and duration is not None:
        if abs(cost - (hourly * duration / 3600.0)) > 0.00001:
            blockers.append("critical_lane_gpu_cost_runtime_mismatch")
    if type(row.get("selected_machine_id")) is not int:
        blockers.append("critical_lane_gpu_machine_invalid")
    if (
        type(row.get("selected_gpu_ram_mb")) is not int
        or int(row.get("selected_gpu_ram_mb") or 0) < 48000
    ):
        blockers.append("critical_lane_gpu_ram_invalid")
    blockers.extend(_validate_gpu_source_bundle(row, evidence_dir=evidence_dir))
    return blockers


def _validate_pubsub(
    row: Mapping[str, Any], *, evidence_dir: Path, repository_sha: str
) -> list[str]:
    blockers: list[str] = []
    for name in (
        "emulator_loopback_only",
        "round_trip_payload_received",
        "message_acknowledged",
        "cleanup_succeeded",
    ):
        if row.get(name) is not True:
            blockers.append(f"critical_lane_pubsub_check_not_passed:{name}")
    if not str(row.get("published_message_id") or "").strip():
        blockers.append("critical_lane_pubsub_message_id_missing")
    if ARTIFACT_DIGEST_PATTERN.fullmatch(str(row.get("round_trip_payload_sha256") or "")) is None:
        blockers.append("critical_lane_pubsub_payload_digest_invalid")
    path = evidence_dir / PUBSUB_SOURCE_NAME
    actual = _file_sha256(path, max_size=MAX_PUBSUB_SOURCE_SIZE)
    if actual is None:
        blockers.append("critical_lane_pubsub_source_missing_or_unsafe")
        return blockers
    if actual[0] != _mapping(row.get("artifact_digests")).get(PUBSUB_SOURCE_NAME):
        blockers.append("critical_lane_pubsub_source_digest_mismatch")
    if actual[1] != _mapping(row.get("artifact_sizes")).get(PUBSUB_SOURCE_NAME):
        blockers.append("critical_lane_pubsub_source_size_mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        blockers.append("critical_lane_pubsub_source_malformed")
        return blockers
    source = _mapping(value)
    if set(source) != {
        "schema_version",
        "generated_at",
        "repository_sha",
        "emulator_loopback_only",
        "expected_payload",
        "received_payload",
        "published_message_id",
        "message_acknowledged",
        "cleanup_succeeded",
        "claim_boundary",
    }:
        blockers.append("critical_lane_pubsub_source_fields_invalid")
    if source.get("schema_version") != PUBSUB_SOURCE_SCHEMA:
        blockers.append("critical_lane_pubsub_source_schema_invalid")
    for name, expected in (
        ("generated_at", row.get("generated_at")),
        ("repository_sha", repository_sha),
        ("emulator_loopback_only", True),
        ("published_message_id", row.get("published_message_id")),
        ("message_acknowledged", True),
        ("cleanup_succeeded", True),
    ):
        if source.get(name) != expected:
            blockers.append(f"critical_lane_pubsub_source_mismatch:{name}")
    expected_payload = _mapping(source.get("expected_payload"))
    received_payload = _mapping(source.get("received_payload"))
    if expected_payload != received_payload:
        blockers.append("critical_lane_pubsub_source_payload_mismatch")
    if set(expected_payload) != {"probe_id", "kind"}:
        blockers.append("critical_lane_pubsub_source_payload_fields_invalid")
    if (
        expected_payload.get("kind") != "pipeline_handoff"
        or re.fullmatch(r"[0-9a-f]{32}", str(expected_payload.get("probe_id") or "")) is None
    ):
        blockers.append("critical_lane_pubsub_source_probe_invalid")
    canonical = json.dumps(received_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if digest != row.get("round_trip_payload_sha256"):
        blockers.append("critical_lane_pubsub_source_payload_digest_mismatch")
    if _mapping(source.get("claim_boundary")) != {
        "local_emulator_transcript_only": True,
        "not_deployed_pubsub_proof": True,
    }:
        blockers.append("critical_lane_pubsub_source_boundary_invalid")
    return blockers


def _validate_native_lerobot(
    row: Mapping[str, Any], *, evidence_dir: Path, repository_sha: str
) -> list[str]:
    blockers: list[str] = []
    if type(row.get("export_file_count")) is not int or int(row.get("export_file_count") or 0) <= 0:
        blockers.append("critical_lane_native_export_file_count_invalid")
    export_dir = str(row.get("export_dir") or "")
    if not export_dir or Path(export_dir).name != export_dir:
        blockers.append("critical_lane_native_export_label_invalid")
    report = _mapping(row.get("validation_report"))
    if report.get("status") != "passed":
        blockers.append("critical_lane_native_validation_not_passed")
    if report.get("loader") != "lerobot_native+hermetic":
        blockers.append("critical_lane_native_loader_invalid")
    if _mapping(report.get("checks")).get("lerobot_native_load") != "passed":
        blockers.append("critical_lane_native_load_not_passed")
    export_total_bytes = row.get("export_total_bytes")
    if type(export_total_bytes) is not int or export_total_bytes < 0:
        blockers.append("critical_lane_native_export_total_bytes_invalid")
    if ARTIFACT_DIGEST_PATTERN.fullmatch(str(row.get("export_tree_sha256") or "")) is None:
        blockers.append("critical_lane_native_export_tree_digest_invalid")
    path = evidence_dir / NATIVE_SOURCE_NAME
    actual = _file_sha256(path, max_size=MAX_NATIVE_SOURCE_SIZE)
    if actual is None:
        blockers.append("critical_lane_native_source_missing_or_unsafe")
        return blockers
    if actual[0] != _mapping(row.get("artifact_digests")).get(NATIVE_SOURCE_NAME):
        blockers.append("critical_lane_native_source_digest_mismatch")
    if actual[1] != _mapping(row.get("artifact_sizes")).get(NATIVE_SOURCE_NAME):
        blockers.append("critical_lane_native_source_size_mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        blockers.append("critical_lane_native_source_malformed")
        return blockers
    source = _mapping(value)
    if set(source) != {
        "schema_version",
        "generated_at",
        "repository_sha",
        "export_dir",
        "export_file_count",
        "export_total_bytes",
        "export_tree_sha256",
        "files",
        "validation_report",
        "claim_boundary",
    }:
        blockers.append("critical_lane_native_source_fields_invalid")
    if source.get("schema_version") != NATIVE_SOURCE_SCHEMA:
        blockers.append("critical_lane_native_source_schema_invalid")
    for name, expected in (
        ("generated_at", row.get("generated_at")),
        ("repository_sha", repository_sha),
        ("export_dir", row.get("export_dir")),
        ("export_file_count", row.get("export_file_count")),
        ("export_total_bytes", row.get("export_total_bytes")),
        ("export_tree_sha256", row.get("export_tree_sha256")),
        ("validation_report", row.get("validation_report")),
    ):
        if source.get(name) != expected:
            blockers.append(f"critical_lane_native_source_mismatch:{name}")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        blockers.append("critical_lane_native_source_files_invalid")
        file_rows: list[str] = []
        total_bytes = 0
    else:
        file_rows = []
        total_bytes = 0
        seen: set[str] = set()
        for index, raw_file in enumerate(files):
            file = _mapping(raw_file)
            if set(file) != {"path", "size", "sha256"}:
                blockers.append(f"critical_lane_native_source_file_fields_invalid:{index}")
                continue
            relative = str(file.get("path") or "")
            relative_path = PurePosixPath(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or str(relative_path) != relative
                or any(part in {"", ".", ".."} for part in relative_path.parts)
                or relative in seen
            ):
                blockers.append(f"critical_lane_native_source_file_path_invalid:{index}")
                continue
            size = file.get("size")
            digest = str(file.get("sha256") or "")
            if type(size) is not int or size < 0:
                blockers.append(f"critical_lane_native_source_file_size_invalid:{index}")
                continue
            if ARTIFACT_DIGEST_PATTERN.fullmatch(digest) is None:
                blockers.append(f"critical_lane_native_source_file_digest_invalid:{index}")
                continue
            seen.add(relative)
            total_bytes += size
            file_rows.append(f"{relative}\t{size}\t{digest.removeprefix('sha256:')}")
        if len(file_rows) != row.get("export_file_count"):
            blockers.append("critical_lane_native_source_file_count_mismatch")
    if total_bytes != row.get("export_total_bytes"):
        blockers.append("critical_lane_native_source_total_bytes_mismatch")
    tree_digest = f"sha256:{hashlib.sha256(chr(10).join(file_rows).encode('utf-8')).hexdigest()}"
    if tree_digest != row.get("export_tree_sha256"):
        blockers.append("critical_lane_native_source_tree_digest_mismatch")
    if _mapping(source.get("claim_boundary")) != {
        "relative_file_manifest_only": True,
        "source_manifest_is_not_dataset_quality_proof": True,
    }:
        blockers.append("critical_lane_native_source_boundary_invalid")
    return blockers


def _validate_cpu_sources(
    row: Mapping[str, Any], *, evidence_dir: Path, repository_sha: str
) -> list[str]:
    try:
        from scripts.build_cpu_full_lane_evidence import validate_cpu_full_lane_evidence
    except ImportError:
        return ["critical_lane_cpu_source_validator_unavailable"]
    return [
        f"critical_lane_cpu_source_invalid:{blocker}"
        for blocker in validate_cpu_full_lane_evidence(
            row,
            planned=evidence_dir / "full-test-lane-planned.json",
            executed=evidence_dir / "full-test-lane-executed.json",
            junit=evidence_dir / "full-test-lane-junit.xml",
            repository_sha=repository_sha,
        )
    ]


def _validate_container_sources(
    row: Mapping[str, Any], *, evidence_dir: Path, repository_sha: str
) -> list[str]:
    from scripts.build_container_production_evidence import (
        build_container_production_evidence,
    )

    expected = build_container_production_evidence(
        source_dir=evidence_dir / CONTAINER_SOURCE_DIR_NAME,
        repository_sha=repository_sha,
    )
    blockers = [
        f"critical_lane_container_source_invalid:{blocker}" for blocker in expected["blockers"]
    ]
    deterministic_fields = (COMMON_EVIDENCE_FIELDS | LANE_FIELDS["container_production"]) - {
        "generated_at"
    }
    for field in sorted(deterministic_fields):
        if row.get(field) != expected.get(field):
            blockers.append(f"critical_lane_container_source_field_mismatch:{field}")
    return blockers


def _validate_lane_payload(
    lane_id: str,
    row: Mapping[str, Any],
    *,
    expected_gpu_image_uri: str | None,
    evidence_dir: Path,
    repository_sha: str,
    now: datetime,
) -> list[str]:
    blockers: list[str] = []
    allowed = COMMON_EVIDENCE_FIELDS | LANE_FIELDS[lane_id]
    missing = allowed - set(row)
    unexpected = set(row) - allowed
    blockers.extend(f"critical_lane_field_missing:{lane_id}:{name}" for name in sorted(missing))
    blockers.extend(
        f"critical_lane_field_unexpected:{lane_id}:{name}" for name in sorted(unexpected)
    )
    generated_at = _parse_aware_timestamp(row.get("generated_at"))
    if generated_at is None:
        blockers.append(f"critical_lane_generated_at_invalid:{lane_id}")
    elif generated_at > now + MAX_CLOCK_SKEW:
        blockers.append(f"critical_lane_generated_at_future:{lane_id}")
    elif now - generated_at > MAX_EVIDENCE_AGE:
        blockers.append(f"critical_lane_evidence_stale:{lane_id}")
    if row.get("blockers") != []:
        blockers.append(f"critical_lane_embedded_blockers_invalid:{lane_id}")
    if not isinstance(row.get("claim_boundary"), Mapping):
        blockers.append(f"critical_lane_claim_boundary_invalid:{lane_id}")
    blockers.extend(_validate_artifacts(lane_id, row))
    if lane_id == "cpu_full":
        blockers.extend(_validate_cpu_full(row))
        blockers.extend(
            _validate_cpu_sources(
                row,
                evidence_dir=evidence_dir,
                repository_sha=repository_sha,
            )
        )
    elif lane_id == "container_production":
        blockers.extend(_validate_container(row))
        blockers.extend(
            _validate_container_sources(
                row,
                evidence_dir=evidence_dir,
                repository_sha=repository_sha,
            )
        )
    elif lane_id == "gpu_provider_canary":
        blockers.extend(
            _validate_gpu_canary(
                row,
                expected_image_uri=expected_gpu_image_uri,
                evidence_dir=evidence_dir,
            )
        )
    elif lane_id == "pubsub_emulator_integration":
        blockers.extend(
            _validate_pubsub(
                row,
                evidence_dir=evidence_dir,
                repository_sha=repository_sha,
            )
        )
    elif lane_id == "native_lerobot_export":
        blockers.extend(
            _validate_native_lerobot(
                row,
                evidence_dir=evidence_dir,
                repository_sha=repository_sha,
            )
        )
    return blockers


def evaluate_scope(
    *,
    root: Path,
    policy: Mapping[str, Any],
    scope: str,
    evidence_dir: Path,
    repository_sha: str,
    expected_gpu_image_uri: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    blockers = validate_policy(root=root, policy=policy)
    evaluation_time = now or datetime.now(timezone.utc)
    if evaluation_time.tzinfo is None or evaluation_time.utcoffset() is None:
        blockers.append("critical_lane_evaluation_time_invalid")
        evaluation_time = datetime.now(timezone.utc)
    scope = scope.strip().upper()
    repository_sha = repository_sha.strip().lower()
    if scope not in KNOWN_SCOPES:
        blockers.append(f"critical_lane_scope_invalid:{scope or 'missing'}")
    if REPOSITORY_SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("critical_lane_repository_sha_invalid")
    lanes = _mapping(policy.get("lanes"))
    required = [
        lane_id
        for lane_id, raw_lane in lanes.items()
        if scope in (_mapping(raw_lane).get("required_for_scopes") or [])
    ]
    normalized_expected_gpu_image = (
        expected_gpu_image_uri.strip() if expected_gpu_image_uri else None
    )
    if "gpu_provider_canary" in required and (
        normalized_expected_gpu_image is None
        or IMAGE_URI_PATTERN.fullmatch(normalized_expected_gpu_image) is None
    ):
        blockers.append("critical_lane_expected_gpu_image_invalid")
    evidence_rows: list[dict[str, Any]] = []
    for lane_id in required:
        path = evidence_dir / f"{lane_id}.json"
        if path.is_symlink():
            blockers.append(f"critical_lane_evidence_symlink:{lane_id}")
            evidence_rows.append({"lane_id": lane_id, "status": "symlink", "path": path.name})
            continue
        if not path.is_file():
            blockers.append(f"critical_lane_evidence_missing:{lane_id}")
            evidence_rows.append({"lane_id": lane_id, "status": "missing", "path": path.name})
            continue
        try:
            evidence = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append(f"critical_lane_evidence_malformed:{lane_id}")
            continue
        row = _mapping(evidence)
        evidence_rows.append(row)
        lane = _mapping(lanes[lane_id])
        if row.get("schema_version") != EVIDENCE_SCHEMA:
            blockers.append(f"critical_lane_envelope_schema_invalid:{lane_id}")
        if row.get("lane_id") != lane_id:
            blockers.append(f"critical_lane_id_mismatch:{lane_id}")
        if row.get("evidence_schema_version") != lane.get("evidence_schema_version"):
            blockers.append(f"critical_lane_payload_schema_invalid:{lane_id}")
        if row.get("status") != "passed":
            blockers.append(f"critical_lane_not_passed:{lane_id}:{row.get('status') or 'missing'}")
        if row.get("executed") is not True:
            blockers.append(f"critical_lane_not_executed:{lane_id}")
        skipped_count = row.get("skipped_count")
        if type(skipped_count) is not int or skipped_count != 0:
            blockers.append(f"critical_lane_skipped:{lane_id}:{row.get('skipped_count')}")
        if str(row.get("repository_sha") or "") != repository_sha:
            blockers.append(f"critical_lane_wrong_sha:{lane_id}")
        blockers.extend(
            _validate_lane_payload(
                lane_id,
                row,
                expected_gpu_image_uri=normalized_expected_gpu_image,
                evidence_dir=evidence_dir,
                repository_sha=repository_sha,
                now=evaluation_time,
            )
        )
    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint.critical_capability_lane_gate.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "repository_sha": repository_sha,
        "expected_gpu_image_uri": normalized_expected_gpu_image,
        "status": "passed" if not blockers else "blocked",
        "required_lane_ids": sorted(required),
        "evidence": evidence_rows,
        "blockers": blockers,
        "claim_boundary": {
            "missing_or_skipped_critical_lane_is_a_scope_blocker": True,
            "sim_only_scope_excludes_physical_robot_lane": True,
            "policy_validation_alone_is_not_lane_execution_proof": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--policy", type=Path, default=Path("docs/critical_capability_lanes.json"))
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--scope")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--repository-sha")
    parser.add_argument(
        "--expected-gpu-image-uri",
        help="Required exact approved @sha256 image URI for SC3/PAID/LIVE evaluation.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[critical-lanes] ERROR unreadable_policy:{exc}", file=sys.stderr)
        return 1
    root = args.root.resolve()
    if args.policy_only:
        blockers = validate_policy(root=root, policy=_mapping(policy))
        result = {
            "schema_version": "blueprint.critical_capability_lane_policy_check.v1",
            "status": "passed" if not blockers else "blocked",
            "blockers": sorted(set(blockers)),
            "claim_boundary": {"policy_check_is_not_execution_proof": True},
        }
    else:
        scope = str(args.scope or "").upper()
        repository_sha = str(args.repository_sha or "").lower()
        if (
            scope not in KNOWN_SCOPES
            or args.evidence_dir is None
            or REPOSITORY_SHA_PATTERN.fullmatch(repository_sha) is None
        ):
            print(
                "[critical-lanes] ERROR evaluation requires --scope, --evidence-dir, and a full --repository-sha",
                file=sys.stderr,
            )
            return 2
        result = evaluate_scope(
            root=root,
            policy=_mapping(policy),
            scope=scope,
            evidence_dir=args.evidence_dir.resolve(),
            repository_sha=repository_sha,
            expected_gpu_image_uri=args.expected_gpu_image_uri,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"[critical-lanes] status={result['status']}")
    for blocker in result["blockers"]:
        print(f"[critical-lanes] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
