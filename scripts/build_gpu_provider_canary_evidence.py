#!/usr/bin/env python3
"""Validate bounded GPU-canary inputs and convert one real result to lane evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


EVIDENCE_SCHEMA = "blueprint.critical_capability_lane_evidence.v1"
PAYLOAD_SCHEMA = "blueprint.provider_canary_evidence.v1"
RESULT_SCHEMA = "unitree_groot_n17_sonic_vast_image_canary.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:([0-9a-f]{64})$")
MAX_HOURLY_RATE_USD = 1.0
MAX_HARD_CAP_USD = 1.0
MAX_LIVE_MINUTES = 30
MAX_STARTUP_TIMEOUT_SECONDS = 1800
MIN_GPU_RAM_MB = 48000
RAW_RESULT_NAME = "vast_unitree_groot_sonic_image_canary_result.json"
SAFE_SUMMARY_NAME = "vast_unitree_groot_sonic_image_canary_safe_summary.json"
SANITIZED_OUTPUT_NAME = "vast_provider_runtime_output.json"
SOURCE_MANIFEST_NAME = "gpu-canary-source-manifest.json"
SOURCE_BUNDLE_NAME = "gpu-provider-canary-sanitized-evidence.zip"
PROVIDER_OUTPUT_SCHEMA = "unitree_groot_n17_sonic_policy_provider_output.v1"
CANARY_MARKER = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_VAST_IMAGE_CANARY_OK"
MAX_RAW_JSON_SIZE = 8 * 1024 * 1024
MAX_PROVIDER_OUTPUT_ZIP_SIZE = 64 * 1024 * 1024
MAX_TOTAL_RAW_SIZE = 96 * 1024 * 1024
MAX_PROVIDER_ZIP_MEMBERS = 256
MAX_PROVIDER_ZIP_MEMBER_SIZE = 16 * 1024 * 1024
MAX_PROVIDER_OUTPUT_MEMBER_SIZE = 4 * 1024 * 1024
MAX_PROVIDER_ZIP_TOTAL_UNCOMPRESSED = 64 * 1024 * 1024
MAX_PROVIDER_ZIP_COMPRESSION_RATIO = 200


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


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


def validate_canary_inputs(
    *,
    image_uri: str,
    approved_image_uri: str,
    repository_sha: str,
    max_hourly_rate: float,
    target_spend_usd: float,
    hard_cap_usd: float,
    max_live_minutes: int,
    startup_timeout_seconds: int,
) -> list[str]:
    blockers: list[str] = []
    if IMAGE_PATTERN.fullmatch(image_uri.strip()) is None:
        blockers.append("gpu_canary_image_not_exact_digest")
    if IMAGE_PATTERN.fullmatch(approved_image_uri.strip()) is None:
        blockers.append("gpu_canary_approved_image_not_exact_digest")
    elif image_uri.strip() != approved_image_uri.strip():
        blockers.append("gpu_canary_image_not_approved")
    if SHA_PATTERN.fullmatch(repository_sha.strip().lower()) is None:
        blockers.append("gpu_canary_repository_sha_invalid")
    numeric = {
        "max_hourly_rate": max_hourly_rate,
        "target_spend_usd": target_spend_usd,
        "hard_cap_usd": hard_cap_usd,
    }
    for name, value in numeric.items():
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            blockers.append(f"gpu_canary_{name}_invalid")
    if math.isfinite(max_hourly_rate) and max_hourly_rate > MAX_HOURLY_RATE_USD:
        blockers.append("gpu_canary_max_hourly_rate_exceeds_contract")
    if math.isfinite(hard_cap_usd) and hard_cap_usd > MAX_HARD_CAP_USD:
        blockers.append("gpu_canary_hard_cap_exceeds_contract")
    if (
        math.isfinite(target_spend_usd)
        and math.isfinite(hard_cap_usd)
        and target_spend_usd > hard_cap_usd
    ):
        blockers.append("gpu_canary_target_spend_exceeds_hard_cap")
    if type(max_live_minutes) is not int or not 1 <= max_live_minutes <= MAX_LIVE_MINUTES:
        blockers.append("gpu_canary_max_live_minutes_out_of_bounds")
    if (
        type(startup_timeout_seconds) is not int
        or not 60 <= startup_timeout_seconds <= MAX_STARTUP_TIMEOUT_SECONDS
    ):
        blockers.append("gpu_canary_startup_timeout_out_of_bounds")
    if startup_timeout_seconds > max_live_minutes * 60:
        blockers.append("gpu_canary_startup_timeout_exceeds_live_window")
    if (
        math.isfinite(max_hourly_rate)
        and math.isfinite(hard_cap_usd)
        and type(max_live_minutes) is int
        and max_live_minutes > 0
        and max_hourly_rate * max_live_minutes / 60.0 > hard_cap_usd
    ):
        blockers.append("gpu_canary_projected_max_cost_exceeds_hard_cap")
    return sorted(set(blockers))


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _project(payload: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    def sanitize(value: object) -> object:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if (
                len(value) > 256
                or value.startswith(("/", "~"))
                or "://" in value
                or any(character.isspace() for character in value)
            ):
                return None
            return value
        if isinstance(value, list) and len(value) <= 100:
            sanitized = [sanitize(item) for item in value]
            return sanitized if all(item is not None for item in sanitized) else None
        return None

    return {field: sanitize(payload.get(field)) for field in fields}


def _blocker_count(payload: Mapping[str, Any]) -> int | None:
    blockers = payload.get("blockers")
    return len(blockers) if isinstance(blockers, list) else None


def _sanitized_json_sources(
    artifact_paths: Mapping[str, Path], *, provider_output_zip: Path | None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for label, path in artifact_paths.items():
        if label.endswith(".json") and path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                blockers.append(f"gpu_canary_raw_json_unreadable:{label}")
                payloads[label] = {}
                continue
            if not 0 < size <= MAX_RAW_JSON_SIZE:
                blockers.append(f"gpu_canary_raw_json_oversize:{label}")
                payloads[label] = {}
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payloads[label] = {}
            else:
                payloads[label] = _mapping(value)
    projections = {
        RAW_RESULT_NAME: _project(
            payloads.get(RAW_RESULT_NAME, {}),
            (
                "schema_version",
                "generated_at",
                "status",
                "public_image",
                "min_gpu_ram_mb",
                "vast_instance_ids",
                "selected_hourly_rate_usd",
                "actual_live_runtime_seconds",
                "estimated_cost_usd",
                "continuing_spend_from_this_run",
                "heartbeat_completed",
                "gpu_sanity_completed",
                "provider_bundle_downloaded_and_ran",
                "provider_output_upload_ok",
                "provider_runtime_output_zip_produced",
                "canary_marker_observed",
                "raw_secret_values_recorded",
            ),
        ),
        SAFE_SUMMARY_NAME: _project(
            payloads.get(SAFE_SUMMARY_NAME, {}),
            (
                "schema_version",
                "generated_at",
                "status",
                "public_image",
                "min_gpu_ram_mb",
                "vast_instance_ids",
                "estimated_cost_usd",
                "continuing_spend_from_this_run",
                "heartbeat_completed",
                "gpu_sanity_completed",
                "provider_bundle_downloaded_and_ran",
                "provider_output_upload_ok",
                "canary_marker_observed",
                "raw_secret_values_recorded",
            ),
        ),
        "vast_provider_adapter_result.json": _project(
            payloads.get("vast_provider_adapter_result.json", {}),
            (
                "schema_version",
                "generated_at",
                "status",
                "vast_instance_ids",
                "estimated_cost_usd",
                "continuing_spend_from_this_run",
            ),
        ),
        "vast_startup_probe_manifest.json": _project(
            payloads.get("vast_startup_probe_manifest.json", {}),
            ("schema_version", "generated_at", "status", "heartbeat_completed"),
        ),
        "vast_gpu_sanity_report.json": _project(
            payloads.get("vast_gpu_sanity_report.json", {}),
            ("schema_version", "generated_at", "status"),
        ),
        "vast_provider_command_result.json": _project(
            payloads.get("vast_provider_command_result.json", {}),
            (
                "schema_version",
                "generated_at",
                "status",
                "provider_command_path_remote_proven",
                "blueprint_provider_bundle_execution_proven",
                "provider_output_upload_ok",
                "provider_runtime_output_zip_produced",
            ),
        ),
        "vast_teardown_manifest.json": _project(
            payloads.get("vast_teardown_manifest.json", {}),
            (
                "schema_version",
                "generated_at",
                "status",
                "vast_instance_ids",
                "runner_gpu_teardown_completed",
                "continuing_spend_from_this_run",
                "raw_secret_values_recorded",
            ),
        ),
        "vast_budget_ledger.json": _project(
            payloads.get("vast_budget_ledger.json", {}),
            (
                "schema_version",
                "generated_at",
                "status",
                "target_spend_usd",
                "hard_cap_usd",
                "max_hourly_rate_usd",
                "max_live_runtime_minutes",
                "selected_hourly_rate_usd",
                "vast_instance_ids",
                "actual_live_runtime_seconds_observed_by_adapter",
                "estimated_cost_usd",
                "actual_cost_usd",
                "estimated_spend_under_target",
                "estimated_spend_under_hard_cap",
                "continuing_spend_from_this_run",
                "raw_secret_values_recorded",
            ),
        ),
    }
    raw_result = payloads.get(RAW_RESULT_NAME, {})
    projections[RAW_RESULT_NAME].update(
        {
            "selected_offer": _project(
                _mapping(raw_result.get("selected_offer")),
                ("machine_id", "gpu_ram_mb", "hourly_rate_usd"),
            ),
            "blocker_count": _blocker_count(raw_result),
            "claim_boundary": _project(
                _mapping(raw_result.get("claim_boundary")),
                (
                    "canary_is_not_policy_inference",
                    "custom_image_startup_proof_only",
                    "generated_world_rank_fidelity_result_proven",
                    "generated_world_policy_evaluation_scope_proven",
                    "accepted_anchor_manipulation_success_proven",
                ),
            ),
        }
    )
    safe_summary = payloads.get(SAFE_SUMMARY_NAME, {})
    projections[SAFE_SUMMARY_NAME].update(
        {
            "selected_offer": _project(
                _mapping(safe_summary.get("selected_offer")),
                ("machine_id", "gpu_ram_mb", "hourly_rate_usd"),
            ),
            "blocker_count": _blocker_count(safe_summary),
        }
    )
    for name in (
        "vast_provider_adapter_result.json",
        "vast_startup_probe_manifest.json",
        "vast_gpu_sanity_report.json",
        "vast_provider_command_result.json",
    ):
        projections[name]["blocker_count"] = _blocker_count(payloads.get(name, {}))
    provider_output: dict[str, Any] = {}
    if provider_output_zip is None:
        blockers.append("gpu_canary_provider_output_zip_not_validated")
    else:
        try:
            archive_context = zipfile.ZipFile(provider_output_zip)
        except (OSError, zipfile.BadZipFile):
            blockers.append("gpu_canary_provider_output_zip_invalid")
        else:
            with archive_context as archive:
                infos = archive.infolist()
                if len(infos) > MAX_PROVIDER_ZIP_MEMBERS:
                    blockers.append("gpu_canary_provider_output_too_many_members")
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    blockers.append("gpu_canary_provider_output_duplicate_members")
                total_uncompressed = 0
                for info in infos:
                    member = PurePosixPath(info.filename)
                    if (
                        info.is_dir()
                        or member.is_absolute()
                        or str(member) != info.filename
                        or any(part in {"", ".", ".."} for part in member.parts)
                        or info.flag_bits & 0x1
                    ):
                        blockers.append("gpu_canary_provider_output_unsafe_member")
                    if info.file_size < 0 or info.file_size > MAX_PROVIDER_ZIP_MEMBER_SIZE:
                        blockers.append("gpu_canary_provider_output_member_oversize")
                    total_uncompressed += info.file_size
                    if info.file_size > 0 and (
                        info.file_size / max(1, info.compress_size)
                        > MAX_PROVIDER_ZIP_COMPRESSION_RATIO
                    ):
                        blockers.append("gpu_canary_provider_output_compression_ratio_exceeded")
                if total_uncompressed > MAX_PROVIDER_ZIP_TOTAL_UNCOMPRESSED:
                    blockers.append("gpu_canary_provider_output_uncompressed_total_oversize")
                candidates = [
                    info
                    for info in infos
                    if PurePosixPath(info.filename).name
                    == "unitree_groot_n17_sonic_policy_provider_output.json"
                ]
                if len(candidates) != 1:
                    blockers.append("gpu_canary_provider_output_member_missing_or_duplicate")
                elif candidates[0].file_size > MAX_PROVIDER_OUTPUT_MEMBER_SIZE:
                    blockers.append("gpu_canary_provider_output_member_oversize")
                elif not blockers:
                    try:
                        value = json.loads(archive.read(candidates[0]).decode("utf-8"))
                    except (UnicodeError, json.JSONDecodeError, RuntimeError):
                        blockers.append("gpu_canary_provider_output_zip_invalid")
                    else:
                        provider_output = _mapping(value)
    checks = _mapping(provider_output.get("checks"))
    projections[SANITIZED_OUTPUT_NAME] = {
        **_project(
            provider_output,
            (
                "schema_version",
                "status",
                "canary_only",
                "canary_marker",
                "unitree_groot_n17_sonic_model_executed",
                "unitree_groot_n17_sonic_policy_action_command_ran",
                "policy_action_model_command_ran",
                "raw_credentials_written_to_artifacts",
                "secret_hashes_written_to_artifacts",
            ),
        ),
        "checks": {
            name: _project(_mapping(checks.get(name)), ("returncode", "duration_seconds"))
            for name in ("python", "nvidia_smi")
        },
        "blocker_count": _blocker_count(provider_output),
    }
    return projections, sorted(set(blockers))


def _write_sanitized_bundle(
    *,
    evidence_dir: Path,
    sources: Mapping[str, Mapping[str, Any]],
    raw_digests: Mapping[str, str],
    raw_sizes: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, int], str, str]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name, payload in sorted(sources.items()):
        path = evidence_dir / name
        _write_json_atomic(path, payload)
        digests[name] = _sha256(path)
        sizes[name] = path.stat().st_size
    manifest = {
        "schema_version": "blueprint.gpu_canary_sanitized_source_manifest.v1",
        "sanitized_artifact_digests": digests,
        "sanitized_artifact_sizes": sizes,
        "raw_artifact_digests": dict(raw_digests),
        "raw_artifact_sizes": dict(raw_sizes),
        "claim_boundary": {
            "sanitized_projection_retained": True,
            "raw_signed_urls_and_commands_excluded": True,
            "manifest_does_not_prove_policy_inference": True,
        },
    }
    manifest_path = evidence_dir / SOURCE_MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    bundle_path = evidence_dir / SOURCE_BUNDLE_NAME
    with tempfile.NamedTemporaryFile(
        prefix=f".{bundle_path.name}.", suffix=".tmp", dir=evidence_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(sources):
                archive.write(evidence_dir / name, name)
            archive.write(manifest_path, SOURCE_MANIFEST_NAME)
        os.replace(temporary, bundle_path)
    finally:
        temporary.unlink(missing_ok=True)
    return digests, sizes, _sha256(manifest_path), _sha256(bundle_path)


def build_gpu_provider_canary_evidence(
    *,
    result_path: Path,
    job_dir: Path,
    evidence_dir: Path,
    image_uri: str,
    approved_image_uri: str,
    repository_sha: str,
    max_hourly_rate: float,
    target_spend_usd: float,
    hard_cap_usd: float,
    max_live_minutes: int,
    startup_timeout_seconds: int,
) -> dict[str, Any]:
    image_uri = image_uri.strip()
    approved_image_uri = approved_image_uri.strip()
    repository_sha = repository_sha.strip().lower()
    job_dir = job_dir.resolve()
    blockers = validate_canary_inputs(
        image_uri=image_uri,
        approved_image_uri=approved_image_uri,
        repository_sha=repository_sha,
        max_hourly_rate=max_hourly_rate,
        target_spend_usd=target_spend_usd,
        hard_cap_usd=hard_cap_usd,
        max_live_minutes=max_live_minutes,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    result_path_is_canonical = result_path.resolve() == job_dir / RAW_RESULT_NAME
    if not result_path_is_canonical:
        blockers.append("gpu_canary_result_path_not_canonical")
        result: dict[str, Any] = {}
    elif result_path.is_symlink():
        blockers.append("gpu_canary_result_symlink")
        result = {}
    elif not result_path.is_file():
        blockers.append("gpu_canary_result_unreadable")
        result = {}
    elif (result_size := _file_size(result_path)) is None:
        blockers.append("gpu_canary_result_unreadable")
        result = {}
    elif not 0 < result_size <= MAX_RAW_JSON_SIZE:
        blockers.append("gpu_canary_result_oversize")
        result = {}
    else:
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append("gpu_canary_result_unreadable")
            result = {}
        else:
            result = _mapping(raw)
            if not result:
                blockers.append("gpu_canary_result_not_object")

    if result.get("schema_version") != RESULT_SCHEMA:
        blockers.append("gpu_canary_result_schema_invalid")
    if result.get("status") != "completed":
        blockers.append("gpu_canary_result_not_completed")
    raw_blockers = result.get("blockers")
    if raw_blockers != []:
        blockers.append("gpu_canary_result_has_blockers")
    if result.get("public_image") != image_uri:
        blockers.append("gpu_canary_result_image_mismatch")
    if result.get("min_gpu_ram_mb") != MIN_GPU_RAM_MB:
        blockers.append("gpu_canary_min_gpu_ram_mismatch")
    required_true = {
        "heartbeat_completed",
        "gpu_sanity_completed",
        "provider_bundle_downloaded_and_ran",
        "provider_output_upload_ok",
        "provider_runtime_output_zip_produced",
        "canary_marker_observed",
    }
    for field in sorted(required_true):
        if result.get(field) is not True:
            blockers.append(f"gpu_canary_result_field_not_true:{field}")
    if result.get("continuing_spend_from_this_run") is not False:
        blockers.append("gpu_canary_teardown_not_proven")
    if result.get("raw_secret_values_recorded") is not False:
        blockers.append("gpu_canary_secret_disclosure_not_false")
    instance_ids = result.get("vast_instance_ids")
    if (
        not isinstance(instance_ids, list)
        or not instance_ids
        or not all(type(item) is int and item > 0 for item in instance_ids)
    ):
        blockers.append("gpu_canary_instance_ids_invalid")
    estimated_cost = result.get("estimated_cost_usd")
    if (
        isinstance(estimated_cost, bool)
        or not isinstance(estimated_cost, (int, float))
        or not math.isfinite(float(estimated_cost))
        or float(estimated_cost) < 0
        or float(estimated_cost) > hard_cap_usd
    ):
        blockers.append("gpu_canary_estimated_cost_invalid")
    selected_offer = _mapping(result.get("selected_offer"))
    if type(selected_offer.get("machine_id")) is not int:
        blockers.append("gpu_canary_selected_machine_invalid")
    if (
        type(selected_offer.get("gpu_ram_mb")) is not int
        or int(selected_offer.get("gpu_ram_mb") or 0) < MIN_GPU_RAM_MB
    ):
        blockers.append("gpu_canary_selected_gpu_ram_insufficient")
    selected_hourly_rate = result.get("selected_hourly_rate_usd")
    if (
        isinstance(selected_hourly_rate, bool)
        or not isinstance(selected_hourly_rate, (int, float))
        or not math.isfinite(float(selected_hourly_rate))
        or not 0 < float(selected_hourly_rate) <= max_hourly_rate
    ):
        blockers.append("gpu_canary_selected_hourly_rate_invalid")
    offer_hourly_rate = selected_offer.get("hourly_rate_usd")
    if (
        not isinstance(offer_hourly_rate, (int, float))
        or isinstance(offer_hourly_rate, bool)
        or not math.isfinite(float(offer_hourly_rate))
        or not isinstance(selected_hourly_rate, (int, float))
        or float(offer_hourly_rate) != float(selected_hourly_rate)
    ):
        blockers.append("gpu_canary_offer_hourly_rate_mismatch")
    actual_live_runtime = result.get("actual_live_runtime_seconds")
    if (
        isinstance(actual_live_runtime, bool)
        or not isinstance(actual_live_runtime, (int, float))
        or not math.isfinite(float(actual_live_runtime))
        or not 0 < float(actual_live_runtime) <= max_live_minutes * 60
    ):
        blockers.append("gpu_canary_live_runtime_invalid")
    if (
        isinstance(estimated_cost, (int, float))
        and not isinstance(estimated_cost, bool)
        and isinstance(selected_hourly_rate, (int, float))
        and not isinstance(selected_hourly_rate, bool)
        and isinstance(actual_live_runtime, (int, float))
        and not isinstance(actual_live_runtime, bool)
        and all(
            math.isfinite(float(value))
            for value in (estimated_cost, selected_hourly_rate, actual_live_runtime)
        )
    ):
        expected_estimate = float(selected_hourly_rate) * float(actual_live_runtime) / 3600.0
        if abs(float(estimated_cost) - expected_estimate) > 0.00001:
            blockers.append("gpu_canary_estimated_cost_runtime_mismatch")
    claim_boundary = _mapping(result.get("claim_boundary"))
    if claim_boundary.get("canary_is_not_policy_inference") is not True:
        blockers.append("gpu_canary_claim_boundary_missing")
    if claim_boundary.get("custom_image_startup_proof_only") is not True:
        blockers.append("gpu_canary_startup_only_boundary_missing")
    for field in (
        "generated_world_rank_fidelity_result_proven",
        "generated_world_policy_evaluation_scope_proven",
        "accepted_anchor_manipulation_success_proven",
    ):
        if claim_boundary.get(field) is not False:
            blockers.append(f"gpu_canary_claim_boundary_not_false:{field}")

    artifact_paths = {
        RAW_RESULT_NAME: result_path,
        SAFE_SUMMARY_NAME: job_dir / SAFE_SUMMARY_NAME,
        "vast_provider_adapter_result.json": Path(
            str(result.get("vast_provider_adapter_result_path") or "")
        ),
        "vast_startup_probe_manifest.json": Path(
            str(result.get("vast_startup_probe_manifest_path") or "")
        ),
        "vast_gpu_sanity_report.json": Path(str(result.get("vast_gpu_sanity_report_path") or "")),
        "vast_provider_command_result.json": Path(
            str(result.get("vast_provider_command_result_path") or "")
        ),
        "vast_teardown_manifest.json": Path(str(result.get("vast_teardown_manifest_path") or "")),
        "vast_budget_ledger.json": Path(str(result.get("vast_budget_ledger_path") or "")),
        "vast_provider_runtime_output.zip": Path(str(result.get("provider_output_zip_path") or "")),
    }
    raw_artifact_digests: dict[str, str] = {}
    raw_artifact_sizes: dict[str, int] = {}
    validated_artifact_paths: dict[str, Path] = {}
    total_raw_size = 0
    for label, path in artifact_paths.items():
        if not str(path) or not _path_within(path, job_dir):
            blockers.append(f"gpu_canary_artifact_outside_job:{label}")
        elif path.is_symlink():
            blockers.append(f"gpu_canary_artifact_symlink:{label}")
        elif not path.is_file():
            blockers.append(f"gpu_canary_artifact_missing:{label}")
        else:
            try:
                size = path.stat().st_size
            except OSError:
                blockers.append(f"gpu_canary_artifact_unreadable:{label}")
                continue
            maximum = (
                MAX_PROVIDER_OUTPUT_ZIP_SIZE
                if label == "vast_provider_runtime_output.zip"
                else MAX_RAW_JSON_SIZE
            )
            if not 0 < size <= maximum:
                blockers.append(f"gpu_canary_artifact_oversize:{label}")
                continue
            try:
                raw_artifact_digests[label] = _sha256(path)
            except OSError:
                blockers.append(f"gpu_canary_artifact_unreadable:{label}")
                continue
            raw_artifact_sizes[label] = size
            validated_artifact_paths[label] = path
            total_raw_size += size
    if total_raw_size > MAX_TOTAL_RAW_SIZE:
        blockers.append("gpu_canary_artifact_total_oversize")

    sanitized_sources, sanitization_blockers = _sanitized_json_sources(
        validated_artifact_paths,
        provider_output_zip=validated_artifact_paths.get("vast_provider_runtime_output.zip"),
    )
    blockers.extend(sanitization_blockers)
    sanitized_provider_output = _mapping(sanitized_sources.get(SANITIZED_OUTPUT_NAME))
    if sanitized_provider_output.get("schema_version") != PROVIDER_OUTPUT_SCHEMA:
        blockers.append("gpu_canary_provider_output_schema_invalid")
    if sanitized_provider_output.get("status") != "completed":
        blockers.append("gpu_canary_provider_output_not_completed")
    if sanitized_provider_output.get("canary_only") is not True:
        blockers.append("gpu_canary_provider_output_not_canary_only")
    if sanitized_provider_output.get("canary_marker") != CANARY_MARKER:
        blockers.append("gpu_canary_provider_output_marker_invalid")
    if sanitized_provider_output.get("blocker_count") != 0:
        blockers.append("gpu_canary_provider_output_has_blockers")
    for field in (
        "unitree_groot_n17_sonic_model_executed",
        "unitree_groot_n17_sonic_policy_action_command_ran",
        "policy_action_model_command_ran",
        "raw_credentials_written_to_artifacts",
        "secret_hashes_written_to_artifacts",
    ):
        if sanitized_provider_output.get(field) is not False:
            blockers.append(f"gpu_canary_provider_output_boundary_invalid:{field}")
    sanitized_checks = _mapping(sanitized_provider_output.get("checks"))
    for check_name in ("python", "nvidia_smi"):
        check = _mapping(sanitized_checks.get(check_name))
        duration = check.get("duration_seconds")
        if check.get("returncode") != 0:
            blockers.append(f"gpu_canary_provider_output_check_failed:{check_name}")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not 0 <= float(duration) <= max_live_minutes * 60
        ):
            blockers.append(f"gpu_canary_provider_output_duration_invalid:{check_name}")
    if len(sanitized_sources) != len(artifact_paths) or any(
        not payload for payload in sanitized_sources.values()
    ):
        blockers.append("gpu_canary_sanitized_sources_incomplete")
        artifact_digests: dict[str, str] = {}
        artifact_sizes: dict[str, int] = {}
        source_manifest_digest = None
        source_bundle_digest = None
    else:
        (
            artifact_digests,
            artifact_sizes,
            source_manifest_digest,
            source_bundle_digest,
        ) = _write_sanitized_bundle(
            evidence_dir=evidence_dir,
            sources=sanitized_sources,
            raw_digests=raw_artifact_digests,
            raw_sizes=raw_artifact_sizes,
        )

    image_match = IMAGE_PATTERN.fullmatch(image_uri)
    blockers = sorted(set(blockers))
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "lane_id": "gpu_provider_canary",
        "evidence_schema_version": PAYLOAD_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_sha": repository_sha,
        "status": "passed" if not blockers else "blocked",
        "executed": bool(result),
        "skipped_count": 0,
        "image_uri": image_uri,
        "approved_image_uri": approved_image_uri,
        "image_digest": f"sha256:{image_match.group(1)}" if image_match else None,
        "canary_result_schema_version": result.get("schema_version"),
        "canary_status": result.get("status"),
        "spend_limits": {
            "max_hourly_rate_usd": max_hourly_rate,
            "target_spend_usd": target_spend_usd,
            "hard_cap_usd": hard_cap_usd,
            "max_live_minutes": max_live_minutes,
            "startup_timeout_seconds": startup_timeout_seconds,
        },
        "result_contract": {
            field: result.get(field)
            for field in sorted(required_true | {"continuing_spend_from_this_run"})
        },
        "estimated_cost_usd": estimated_cost,
        "selected_machine_id": selected_offer.get("machine_id"),
        "selected_gpu_ram_mb": selected_offer.get("gpu_ram_mb"),
        "selected_hourly_rate_usd": selected_hourly_rate,
        "actual_live_runtime_seconds": actual_live_runtime,
        "artifact_digests": artifact_digests,
        "artifact_sizes": artifact_sizes,
        "raw_artifact_digests": raw_artifact_digests,
        "source_manifest_digest": source_manifest_digest,
        "source_bundle_digest": source_bundle_digest,
        "blockers": blockers,
        "claim_boundary": {
            "gpu_canary_is_custom_image_startup_proof_only": True,
            "gpu_canary_is_not_policy_inference": True,
            "gpu_canary_is_not_sc3_rank_fidelity_proof": True,
            "evid_03_remains_external": True,
            "continuing_spend_must_be_false": True,
        },
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--approved-image-uri", required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--max-hourly-rate", type=float, required=True)
    parser.add_argument("--target-spend-usd", type=float, required=True)
    parser.add_argument("--hard-cap-usd", type=float, required=True)
    parser.add_argument("--max-live-minutes", type=int, required=True)
    parser.add_argument("--startup-timeout-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_common_arguments(preflight)
    convert = subparsers.add_parser("convert")
    _add_common_arguments(convert)
    convert.add_argument("--result", type=Path, required=True)
    convert.add_argument("--job-dir", type=Path, required=True)
    convert.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    common = {
        "image_uri": args.image_uri,
        "approved_image_uri": args.approved_image_uri,
        "repository_sha": args.repository_sha,
        "max_hourly_rate": args.max_hourly_rate,
        "target_spend_usd": args.target_spend_usd,
        "hard_cap_usd": args.hard_cap_usd,
        "max_live_minutes": args.max_live_minutes,
        "startup_timeout_seconds": args.startup_timeout_seconds,
    }
    if args.command == "preflight":
        blockers = validate_canary_inputs(**common)
        result: dict[str, Any] = {
            "schema_version": "blueprint.gpu_provider_canary_preflight.v1",
            "status": "passed" if not blockers else "blocked",
            "repository_sha": args.repository_sha.strip().lower(),
            "image_uri": args.image_uri.strip(),
            "approved_image_uri": args.approved_image_uri.strip(),
            "blockers": blockers,
            "claim_boundary": {
                "preflight_is_not_provider_execution": True,
                "preflight_authorizes_no_spend_by_itself": True,
            },
        }
    else:
        result = build_gpu_provider_canary_evidence(
            result_path=args.result.expanduser().absolute(),
            job_dir=args.job_dir.resolve(),
            evidence_dir=args.evidence_dir.expanduser().absolute(),
            **common,
        )
    _write_json_atomic(args.output.resolve(), result)
    print(f"[gpu-canary-evidence] status={result['status']}")
    for blocker in result["blockers"]:
        print(f"[gpu-canary-evidence] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
