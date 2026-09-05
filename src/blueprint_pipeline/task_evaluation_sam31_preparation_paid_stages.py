"""Canonical paid SAM and FlashSplat child stages for scene preparation.

The parent Website preparation request fixes the scene and task.  This module
accepts no command or provider arguments from that request: it derives two
closed allocator invocations from exact host-resident inputs and a server-owned
profile, then requires terminal teardown evidence before exposing artifacts to
the next phase.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .paid_attempt_authority import bind_lane_prior_spend
from .task_evaluation_scene_configuration_submission_inputs import (
    beneath,
    checked_file,
    read,
    sha,
)

STAGES = {"sam31_tracking", "contribution_sweep"}
SAM31_SOURCE_PROFILE = "render_derived_synthetic_method_inputs"
SAM31_MAX_SPEND_USD = 1.0
SAM31_MAX_TTL_SECONDS = 1_800
GAUSSIAN_MAX_SPEND_USD = 1.5
GAUSSIAN_TTL_SECONDS = 3_600
ALLOCATOR_PREFIX = (
    "-m",
    "blueprint_pipeline.paid_resource_allocator",
    "gpu-canary",
)


class Sam31PreparationPaidStageError(ValueError):
    """A paid precursor could not be derived or closed safely."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Sam31PreparationPaidStageError("sam31_preparation_paid_" + code)


def _record(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "artifact_missing")
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(value)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resident(path: str | Path, roots: tuple[Path, ...], code: str) -> Path:
    source = Path(path)
    _require(source.is_absolute() and not any(item.is_symlink() for item in (source, *source.parents)), code)
    resolved = source.resolve()
    _require(any(resolved.is_relative_to(root.resolve()) for root in roots), code)
    return resolved


def _input(job: Mapping[str, Any], name: str, roots: tuple[Path, ...]) -> Path:
    row = (job.get("inputs") or {}).get(name)
    _require(isinstance(row, Mapping), "input_missing:" + name)
    path = _resident(str(row.get("path") or ""), roots, "input_path_invalid:" + name)
    return checked_file(path, dict(row))


def _task_authority(job: Mapping[str, Any], roots: tuple[Path, ...]) -> dict[str, Any]:
    host = (job.get("plan") or {}).get("host_inputs") or {}
    row = host.get("task_request")
    _require(isinstance(row, Mapping), "task_authority_missing")
    path = _resident(str(row.get("path") or ""), roots, "task_authority_path_invalid")
    checked_file(path, dict(row))
    task = json.loads(path.read_text(encoding="utf-8"))
    authority = task.get("human_authority") if isinstance(task, Mapping) else None
    _require(isinstance(authority, Mapping), "task_authority_missing")
    for field in ("accepted_by", "accepted_on", "authority_reference"):
        _require(bool(str(authority.get(field) or "").strip()), "task_authority_invalid")
    _require(
        authority.get("private_derived_frame_disclosure_authorized") is True
        and authority.get("provider_retention_terms_accepted") is True
        and authority.get("provider_training_terms_accepted") is True
        and authority.get("provider_training_authorized") is False,
        "task_authority_invalid",
    )
    return dict(authority)


def _secret_file(value: Any, code: str) -> Path:
    path = Path(str(value or "")).expanduser()
    _require(path.is_absolute() and path.is_file() and not path.is_symlink(), code)
    mode = path.stat().st_mode & 0o777
    _require(mode & 0o027 == 0, code)
    return path.resolve()


def _positive_number(value: Any, *, maximum: float, code: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0 < float(value) <= maximum,
        code,
    )
    return float(value)


def _default_allocator_runner(argv: list[str], *, cwd: Path, timeout: int) -> int:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124
    return int(completed.returncode)


def _invoke_allocator(
    *,
    argv: list[str],
    result_path: Path,
    job: Mapping[str, Any],
    repo: Path,
    timeout: int,
    allocator_runner: Callable[..., int],
) -> tuple[dict[str, Any] | None, int | None]:
    _require(argv[:4] == [sys.executable, *ALLOCATOR_PREFIX], "allocator_entrypoint_invalid")
    _require(argv[-1] == "--execute" and "--experimental-branch-diagnostic" not in argv,
             "allocator_execution_shape_invalid")
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8")), None
    if job.get("resume_only"):
        return None, None
    return_code = allocator_runner(argv, cwd=repo, timeout=timeout)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    return result, return_code


def _sam31_stage(
    job: Mapping[str, Any],
    *,
    roots: tuple[Path, ...],
    output: Path,
    repo: Path,
    config: Mapping[str, Any],
    allocator_runner: Callable[..., int],
) -> dict[str, Any]:
    from .sam31_paid_attempt_authority import materialize_sam31_paid_attempt_authority
    from .sam31_provider_launch_packet import materialize_sam31_gpu_canary_request
    from .sam31_source_track_canary_worker import build_sam31_source_track_input_bundle

    _require(config.get("source_profile") == SAM31_SOURCE_PROFILE, "sam31_source_profile_invalid")
    cap = _positive_number(config.get("max_spend_usd"), maximum=SAM31_MAX_SPEND_USD,
                           code="sam31_spend_cap_invalid")
    hourly = _positive_number(config.get("max_hourly_rate_usd"), maximum=cap,
                              code="sam31_hourly_rate_invalid")
    ttl = config.get("hard_ttl_seconds")
    _require(type(ttl) is int and 1 <= ttl <= SAM31_MAX_TTL_SECONDS
             and hourly * ttl / 3_600 <= cap, "sam31_ttl_invalid")
    _require(config.get("retry_cap") == 0 and config.get("allowed_active_instance_ids") == [],
             "sam31_retry_or_allowlist_invalid")
    authority = _task_authority(job, roots)
    hf_token = _secret_file(config.get("hf_token_file"), "sam31_hf_token_file_invalid")
    run_request = _input(job, "sam31_run_request", roots)
    provider_profile = _input(job, "sam31_provider_profile", roots)
    request_value = read(run_request)
    frames = request_value.get("frame_registry")
    _require(isinstance(frames, list) and 1 <= len(frames) <= 128, "sam31_frame_count_invalid")

    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    bundle = prepared / "sam31-input-bundle.zip"
    bundle_receipt = prepared / "sam31-input-bundle-receipt.json"
    if not bundle_receipt.exists():
        build_sam31_source_track_input_bundle(
            request_path=run_request,
            bundle_path=bundle,
            receipt_path=bundle_receipt,
        )
    checked_file(bundle, read(bundle_receipt, digest_field="receipt_digest")["bundle"])
    gpu_request = prepared / "sam31-gpu-request.json"
    child_id = str(job["child_id"])
    if not gpu_request.exists():
        materialize_sam31_gpu_canary_request(
            provider_profile_path=provider_profile,
            source_track_run_request_path=run_request,
            input_bundle_path=bundle,
            input_bundle_receipt_path=bundle_receipt,
            source_profile=SAM31_SOURCE_PROFILE,
            source_commit_sha=str(job["expected_source_commit"]),
            expected_camera_count=len(frames),
            expected_frame_count=len(frames),
            max_spend_usd=cap,
            hard_ttl_seconds=ttl,
            retry_cap=0,
            authority_id=child_id,
            output_path=gpu_request,
        )
    paid_authority = prepared / "sam31-paid-attempt-authority.json"
    if not paid_authority.exists():
        materialize_sam31_paid_attempt_authority(
            request_path=gpu_request,
            bundle_path=bundle,
            bundle_receipt_path=bundle_receipt,
            authorization_reference=str(authority["authority_reference"]),
            authorized_by=str(authority["accepted_by"]),
            authorized_on=str(authority["accepted_on"]),
            blueprint_commit=str(job["expected_source_commit"]),
            max_hourly_rate_usd=hourly,
            hard_cap_usd=cap,
            hard_ttl_seconds=ttl,
            aggregate_goal_spend_before_attempt_usd=float(config.get("aggregate_goal_spend_before_attempt_usd", 0)),
            aggregate_goal_spend_cap_usd=float(config.get("aggregate_goal_spend_cap_usd", cap)),
            output_path=paid_authority,
            allowed_active_instance_ids=(),
        )
    allocator = output / "allocator"
    allocator.mkdir(parents=True, exist_ok=True)
    result_path = allocator / "result.json"
    argv = [
        sys.executable,
        *ALLOCATOR_PREFIX,
        "--admission-out", str(allocator / "admission.json"),
        "--bound-request-out", str(allocator / "bound-request.json"),
        "--adapter-output", str(result_path),
        "--pod-name", "blueprint-sam31-scene-preparation-" + child_id[-24:],
        "--expected-source-commit", str(job["expected_source_commit"]),
        "--provider", "vast",
        "--probe-kind", "semantic-sam31-source-tracks",
        "--provider-launch-request", str(gpu_request),
        "--preflight-bundle", str(allocator / "sam31-execution-preflight.json"),
        "--sam31-input-bundle", str(bundle),
        "--sam31-input-bundle-receipt", str(bundle_receipt),
        "--sam31-attempt-authority", str(paid_authority),
        "--sam31-hf-token-file", str(hf_token),
        "--sam31-max-hourly-rate-usd", str(hourly),
        "--sam31-max-spend-usd", str(cap),
        "--sam31-hard-ttl-seconds", str(ttl),
        "--sam31-retry-cap", "0",
        "--sam31-authority-id", child_id,
        "--execute",
    ]
    result, return_code = _invoke_allocator(
        argv=argv,
        result_path=result_path,
        job=job,
        repo=repo,
        timeout=ttl + 900,
        allocator_runner=allocator_runner,
    )
    if result is None:
        return {"status": "failed", "stage_id": "sam31_tracking", "artifacts": {},
                "blockers": ["sam31_paid_stage_started_without_terminal_reconciliation"],
                "candidate_policy_queried": False}
    raw_tracks = Path(str(result.get("source_track_import_result_path") or ""))
    tracks = (
        _resident(raw_tracks, (output,), "sam31_result_path_invalid")
        if raw_tracks.is_absolute() and raw_tracks.is_file()
        else raw_tracks
    )
    tracks_valid = False
    if tracks.is_file() and not tracks.is_symlink():
        try:
            track_result = read(tracks, digest_field="result_digest")
            tracks_valid = (
                track_result.get("schema_version") == "semantic_source_track_import_result.v1"
                and track_result.get("status") == "completed"
                and track_result.get("result_digest") == result.get("source_track_import_result_digest")
            )
        except (OSError, ValueError, KeyError, TypeError):
            tracks_valid = False
    result_digest_valid = result.get("execution_result_digest") == canonical_digest(
        result, digest_field="execution_result_digest"
    )
    bundle_identity = sha(bundle)
    request_binding_valid = False
    try:
        request_packet = read(gpu_request, digest_field="request_digest")
        bound_packet = read(allocator / "bound-request.json", digest_field="bound_request_digest")
        request_binding_valid = (
            result.get("request_digest") == request_packet["request_digest"]
            and bound_packet.get("request_digest") == request_packet["request_digest"]
            and result.get("bound_request_digest") == bound_packet["bound_request_digest"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    terminal_records = {}
    for name, field in (
        ("sam31_teardown", "teardown_manifest_path"),
        ("sam31_artifact_manifest", "artifact_manifest_path"),
    ):
        raw = Path(str(result.get(field) or ""))
        if raw.is_absolute() and raw.is_file():
            path = _resident(raw, (output,), "sam31_terminal_path_invalid")
            terminal_records[name] = _record(path)
    zero = allocator / "sam31_vast_source_track_canary" / "provider_zero_verification.json"
    zero_valid = False
    try:
        zero_value = read(zero, digest_field="provider_zero_digest")
        zero_valid = (
            zero_value.get("schema_version") == "semantic_sam31_vast_provider_zero.v1"
            and zero_value.get("status") == "PASS" and zero_value.get("api_confirmed") is True
            and zero_value.get("provider") == "vast"
            and zero_value.get("scoped_live_resource_count") == 0
            and zero_value.get("global_live_resource_count") == 0
            and zero_value.get("request_digest") == result.get("request_digest")
            and zero_value.get("bound_request_digest") == result.get("bound_request_digest")
            and zero_value.get("provider_zero_digest") == result.get("provider_zero_digest")
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    terminal_valid = False
    if set(terminal_records) == {"sam31_teardown", "sam31_artifact_manifest"}:
        teardown = Path(terminal_records["sam31_teardown"]["path"])
        terminal_valid = _teardown_valid(teardown) and _terminal_manifest_valid(
            Path(terminal_records["sam31_artifact_manifest"]["path"]),
            lane="semantic_sam31_source_tracks", required_paths=(tracks, teardown, zero),
        )
    complete = (
        result.get("schema_version") == "semantic_sam31_vast_source_track_execution.v1"
        and return_code in (None, 0)
        and result.get("status") == "completed"
        and result.get("provider_zero_verified") is True
        and result.get("retry_cap") == 0
        and result.get("blockers") in (None, [])
        and result.get("source_commit_sha") == job["expected_source_commit"]
        and result.get("input_bundle_digest") == bundle_identity
        and result.get("source_track_run_request_digest") == read(bundle_receipt)["source_track_run_request_digest"]
        and result.get("continuing_spend_from_this_run") is False
        and result.get("all_staged_objects_absent") is True
        and (result.get("independent_watchdog") or {}).get("status") == "provider_terminal"
        and zero_valid and terminal_valid and request_binding_valid
        and result_digest_valid
        and tracks_valid
        and set(terminal_records) == {"sam31_teardown", "sam31_artifact_manifest"}
    )
    artifacts = {"sam31_allocator_result": _record(result_path)}
    artifacts.update(terminal_records)
    if tracks_valid:
        artifacts["sam31_source_tracks"] = _record(tracks.resolve())
    if zero_valid:
        artifacts["sam31_provider_zero"] = _record(zero)
    return {
        "status": "completed" if complete else "failed",
        "stage_id": "sam31_tracking",
        "artifacts": artifacts,
        "blockers": [] if complete else list(result.get("blockers") or ["sam31_tracking_not_terminal"]),
        "provider_compute_allocated": bool(result.get("provider_mutations_performed")),
        "candidate_policy_queried": False,
        "raw_source_uploaded": False,
    }



def _terminal_manifest_valid(
    path: Path, *, lane: str, required_paths: tuple[Path, ...], binding: Mapping[str, Any] | None = None
) -> bool:
    """Rehash retained files using the canonical allocator's actual inventory format."""
    try:
        manifest = read(path, digest_field="manifest_digest")
        rows = manifest.get("files")
        actual_binding = manifest.get("binding") or {}
        if (
            manifest.get("schema_version") != "task_evaluation_artifact_manifest.v1"
            or manifest.get("status") != "completed" or manifest.get("blockers") != []
            or actual_binding.get("allocator_lane") != lane
            or actual_binding.get("retry_cap") != 0
            or any(actual_binding.get(key) != value for key, value in (binding or {}).items())
            or not isinstance(rows, list) or not rows or manifest.get("file_count") != len(rows)
        ):
            return False
        seen: set[Path] = set()
        total = 0
        for row in rows:
            artifact = beneath(path.parent, row["relative_path"])
            size = row.get("size_bytes")
            if (artifact in seen or type(size) is not int or size < 0
                    or not artifact.is_file() or artifact.stat().st_size != size
                    or sha(artifact) != row.get("sha256")):
                return False
            seen.add(artifact)
            total += size
        return (manifest.get("total_size_bytes") == total
                and set(required_paths).issubset(seen))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _teardown_valid(path: Path) -> bool:
    try:
        teardown = read(path)
        return (teardown.get("schema_version") == "vast_teardown_manifest.v1"
                and teardown.get("continuing_spend_from_this_run") is False)
    except (OSError, ValueError, KeyError, TypeError):
        return False

def _gaussian_execution_authority(
    *, freeze: Mapping[str, Any], authority: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    scene = freeze.get("scene") or {}
    value = {
        "schema_version": "public_scene_gaussian_excision_execution_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": str(authority["authority_reference"]),
        "authorized_by": str(authority["accepted_by"]),
        "authorized_on": str(authority["accepted_on"]),
        "purpose": "released_code_segment_contribution_sweep",
        "publisher_scene_id": str(scene.get("publisher_scene_id") or ""),
        "target_instance_id": str(scene.get("target_instance_id") or ""),
        "freeze_digest": freeze.get("freeze_digest"),
        "private_scene_derived_standard_splat_upload_authorized": True,
        "paid_compute_authorized": True,
        "provider_zero_required_before_and_after": True,
        "teardown_required": True,
        "raw_interiorgs_downloaded_bytes_upload_authorized": False,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "automatic_paid_retry_authorized": False,
        "retention_policy": "bounded_to_goal_then_provider_zero",
        "hard_attempt_spend_cap_usd": GAUSSIAN_MAX_SPEND_USD,
        "maximum_single_resource_ttl_seconds": GAUSSIAN_TTL_SECONDS,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "authorization_digest": "",
    }
    value["authorization_digest"] = canonical_digest(value, digest_field="authorization_digest")
    _write(path, value)
    return value


def _gaussian_paid_authority(
    *, receipt: Mapping[str, Any], authority: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    prior = bind_lane_prior_spend(
        prior_result_paths=(), reconciliation_path=None, lane="gaussian_excision"
    )
    value = {
        "schema_version": "public_scene_gaussian_excision_paid_attempt_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": str(authority["authority_reference"]),
        "authorized_by": str(authority["accepted_by"]),
        "authorized_on": str(authority["accepted_on"]),
        "purpose": "released_code_segment_contribution_sweep",
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": receipt.get("execution_authority_digest"),
        "freeze_digest": receipt.get("freeze_digest"),
        "bundle_sha256": receipt.get("bundle_sha256"),
        "corrective_blueprint_commit": receipt.get("blueprint_commit"),
        "paid_attempt_ordinal": 1,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": GAUSSIAN_MAX_SPEND_USD,
        "maximum_single_resource_ttl_seconds": GAUSSIAN_TTL_SECONDS,
        "active_instance_allowlist": [],
        "prior_terminal_attempts": prior["prior_terminal_attempts"],
        "prior_spend_reconciliation": prior["reconciliation"],
        "prior_actual_provider_spend_usd": prior["actual_total_usd"],
        "authorization_digest": "",
    }
    value["authorization_digest"] = canonical_digest(value, digest_field="authorization_digest")
    _write(path, value)
    return value


def _contribution_stage(
    job: Mapping[str, Any],
    *,
    roots: tuple[Path, ...],
    output: Path,
    repo: Path,
    config: Mapping[str, Any],
    allocator_runner: Callable[..., int],
) -> dict[str, Any]:
    from .adp_gaussian_excision_vast import (
        build_gaussian_excision_vast_bundle,
        validate_gaussian_excision_paid_attempt_authority,
    )

    _require(config.get("max_spend_usd") == GAUSSIAN_MAX_SPEND_USD
             and config.get("hard_ttl_seconds") == GAUSSIAN_TTL_SECONDS
             and config.get("retry_cap") == 0
             and config.get("allowed_active_instance_ids") == [],
             "gaussian_limits_invalid")
    hourly = _positive_number(config.get("max_hourly_rate_usd"), maximum=0.6,
                              code="gaussian_hourly_rate_invalid")
    authority = _task_authority(job, roots)
    freeze_path = _input(job, "segment_sweep_freeze", roots)
    source = _input(job, "standard_splat", roots)
    cameras = _input(job, "camera_contract", roots)
    freeze = read(freeze_path, digest_field="freeze_digest")
    _require((freeze.get("segment_contribution_sweep") or {}).get("kind")
             == "repair_supported_full_view_segment_contribution_sweep.v1",
             "gaussian_freeze_invalid")
    flashsplat = _resident(str(config.get("flashsplat_root") or ""), roots,
                           "flashsplat_root_invalid")
    wheelhouse = _resident(str(config.get("dependency_wheelhouse_path") or ""), roots,
                           "gaussian_wheelhouse_invalid")
    dependency_manifest = _resident(str(config.get("dependency_manifest_path") or ""), roots,
                                    "gaussian_dependency_manifest_invalid")
    _require(flashsplat.is_dir() and wheelhouse.is_dir() and dependency_manifest.is_file(),
             "gaussian_dependencies_missing")
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    execution_authority = prepared / "gaussian-execution-authority.json"
    if not execution_authority.exists():
        _gaussian_execution_authority(freeze=freeze, authority=authority, path=execution_authority)
    bundle_root = prepared / "bundle"
    bundle_receipt_path = bundle_root / "adp_gaussian_excision_bundle_receipt.json"
    if not bundle_receipt_path.exists():
        build_gaussian_excision_vast_bundle(
            repo_root=repo,
            flashsplat_root=flashsplat,
            freeze_path=freeze_path,
            source_standard_splat_path=source,
            camera_contract_path=cameras,
            execution_authority_path=execution_authority,
            dependency_wheelhouse_path=wheelhouse,
            dependency_manifest_path=dependency_manifest,
            job_dir=bundle_root,
        )
    bundle_receipt = read(bundle_receipt_path)
    _require(bundle_receipt.get("status") == "ready"
             and bundle_receipt.get("schema_version") == "adp009b_gaussian_excision_vast_bundle.v1"
             and bundle_receipt.get("blueprint_commit") == job["expected_source_commit"]
             and bundle_receipt.get("freeze_digest") == freeze["freeze_digest"],
             "gaussian_bundle_not_ready")
    bundle_path = _resident(str(bundle_receipt.get("bundle_path") or ""), (output,),
                            "gaussian_bundle_path_invalid")
    checked_file(bundle_path, {"sha256": bundle_receipt.get("bundle_sha256"),
                              "size_bytes": bundle_receipt.get("bundle_size_bytes")})
    paid_authority_path = prepared / "gaussian-paid-attempt-authority.json"
    paid_authority = (
        read(paid_authority_path, digest_field="authorization_digest")
        if paid_authority_path.exists()
        else _gaussian_paid_authority(
            receipt=bundle_receipt, authority=authority, path=paid_authority_path
        )
    )
    validate_gaussian_excision_paid_attempt_authority(
        paid_authority,
        prepared_bundle=bundle_receipt,
        previous_attempt_receipt=None,
        allowed_active_instance_ids=(),
    )
    allocator = output / "allocator"
    allocator.mkdir(parents=True, exist_ok=True)
    result_path = allocator / "result.json"
    argv = [
        sys.executable,
        *ALLOCATOR_PREFIX,
        "--admission-out", str(allocator / "admission.json"),
        "--bound-request-out", str(allocator / "bound-request.json"),
        "--adapter-output", str(result_path),
        "--pod-name", "blueprint-gaussian-scene-preparation-" + str(job["child_id"])[-20:],
        "--expected-source-commit", str(job["expected_source_commit"]),
        "--provider", "vast",
        "--probe-kind", "adp-gaussian-excision",
        "--adp-gaussian-excision-bundle-receipt", str(bundle_receipt_path),
        "--adp-gaussian-excision-attempt-authority", str(paid_authority_path),
        "--adp-job-dir", str(allocator / "gaussian-excision-job"),
        "--adp-max-hourly-rate-usd", str(hourly),
        "--adp-max-spend-usd", str(GAUSSIAN_MAX_SPEND_USD),
        "--adp-hard-ttl-seconds", str(GAUSSIAN_TTL_SECONDS),
    ]
    avoidlist = config.get("machine_avoidlist_path")
    if avoidlist:
        avoidlist_path = _resident(str(avoidlist), roots, "machine_avoidlist_invalid")
        _require(avoidlist_path.is_file(), "machine_avoidlist_invalid")
        argv.extend(("--adp-machine-avoidlist", str(avoidlist_path)))
    argv.append("--execute")
    result, return_code = _invoke_allocator(
        argv=argv,
        result_path=result_path,
        job=job,
        repo=repo,
        timeout=GAUSSIAN_TTL_SECONDS + 900,
        allocator_runner=allocator_runner,
    )
    if result is None:
        return {"status": "failed", "stage_id": "contribution_sweep", "artifacts": {},
                "blockers": ["gaussian_paid_stage_started_without_terminal_reconciliation"],
                "candidate_policy_queried": False}
    raw_execution_path = Path(str(result.get("execution_result_path") or ""))
    execution_path = (
        _resident(raw_execution_path, (output,), "gaussian_result_path_invalid")
        if raw_execution_path.is_absolute() and raw_execution_path.is_file()
        else raw_execution_path
    )
    try:
        execution = read(execution_path, digest_field="result_digest")
    except (OSError, ValueError, KeyError, TypeError):
        execution = {}
    manifest_row = execution.get("contribution_manifest") if isinstance(execution, Mapping) else None
    manifest_path = (
        beneath(execution_path.parent, str(manifest_row.get("relative_path") or ""))
        if isinstance(manifest_row, Mapping)
        else Path()
    )
    manifest_valid = False
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest = read(manifest_path, digest_field="manifest_digest")
            checked_file(manifest_path, dict(manifest_row))
            manifest_valid = (
                manifest.get("schema_version") == "adp009b_gaussian_excision_contribution_evidence.v1"
                and manifest.get("freeze_digest") == freeze["freeze_digest"]
                and manifest.get("manifest_digest") == execution.get("contribution_manifest_digest")
                and manifest.get("heldout_cameras_accessed_for_classification") is False
                and bool(manifest.get("repetitions")) and bool(manifest.get("calibration_renders"))
            )
            for row in [*manifest.get("repetitions", []), *manifest.get("calibration_renders", [])]:
                checked_file(beneath(manifest_path.parent, row["relative_path"]), row)
        except (OSError, ValueError, KeyError, TypeError):
            manifest_valid = False
    raw_artifact_manifest = Path(str(result.get("artifact_manifest_path") or ""))
    artifact_manifest = (
        _resident(raw_artifact_manifest, (output,), "gaussian_artifact_manifest_path_invalid")
        if raw_artifact_manifest.is_absolute() and raw_artifact_manifest.is_file()
        else raw_artifact_manifest
    )
    complete = (
        result.get("schema_version") == "adp009b_gaussian_excision_vast_run.v1"
        and result.get("bundle_sha256") == bundle_receipt["bundle_sha256"]
        and result.get("all_staged_objects_absent") is True
        and (result.get("independent_watchdog") or {}).get("status") == "provider_terminal"
        and return_code in (None, 0)
        and result.get("status") == "completed"
        and result.get("continuing_spend_from_this_run") is False
        and result.get("retry_cap") == 0
        and result.get("blockers") in (None, [])
        and execution.get("schema_version") == "adp009b_gaussian_excision_result.v1"
        and execution.get("status") == "completed"
        and execution.get("freeze_digest") == freeze["freeze_digest"]
        and execution.get("released_code_executed") is True
        and execution.get("heldout_cameras_accessed_for_classification") is False
        and execution.get("provider_zero_required_after_return") is True
        and execution.get("depth_anything_3_used") is False
        and execution.get("retry_cap") == 0
        and execution.get("blockers") == []
        and manifest_valid
        and artifact_manifest.is_file()
        and not artifact_manifest.is_symlink()
    )
    artifacts = {"gaussian_allocator_result": _record(result_path)}
    if artifact_manifest.is_file() and not artifact_manifest.is_symlink():
        artifacts["gaussian_artifact_manifest"] = _record(artifact_manifest)
    if execution_path.is_file() and not execution_path.is_symlink():
        artifacts["gaussian_provider_execution_result"] = _record(execution_path.resolve())
    if manifest_valid:
        artifacts["gaussian_contribution_evidence"] = _record(manifest_path.resolve())
    raw_teardown = Path(str(result.get("teardown_manifest_path") or ""))
    teardown = (
        _resident(raw_teardown, (output,), "gaussian_teardown_path_invalid")
        if raw_teardown.is_absolute() and raw_teardown.is_file()
        else raw_teardown
    )
    complete = complete and _teardown_valid(teardown) and _terminal_manifest_valid(
        artifact_manifest, lane="adp_gaussian_excision",
        required_paths=(execution_path, manifest_path, teardown),
        binding={"bundle_sha256": bundle_receipt["bundle_sha256"], "provider": "vast"},
    )
    if teardown.is_file() and not teardown.is_symlink():
        artifacts["gaussian_teardown"] = _record(teardown.resolve())
    return {
        "status": "completed" if complete else "failed",
        "stage_id": "contribution_sweep",
        "artifacts": artifacts,
        "blockers": [] if complete else list(result.get("blockers") or ["gaussian_contribution_not_terminal"]),
        "provider_compute_allocated": bool(result.get("provider_mutations_performed")),
        "candidate_policy_queried": False,
        "raw_source_uploaded": False,
    }



def validate_retained_paid_stage(outcome: Mapping[str, Any], *, stage_id: str) -> None:
    """Read-only validation of a completed phase's exact retained artifact set."""
    _require(stage_id in STAGES and outcome.get("stage_id") == stage_id
             and outcome.get("status") == "completed", "replay_stage_invalid")
    prefix, lane, names = (
        ("sam31", "semantic_sam31_source_tracks", ("source_tracks", "provider_zero"))
        if stage_id == "sam31_tracking"
        else ("gaussian", "adp_gaussian_excision", ("provider_execution_result", "contribution_evidence"))
    )
    required = {prefix + "_" + name for name in ("allocator_result", "teardown", "artifact_manifest", *names)}
    records = outcome.get("artifacts")
    _require(isinstance(records, Mapping) and set(records) == required,
             "replay_artifact_set_changed")
    paths = {}
    for name, row in records.items():
        _require(isinstance(row, Mapping), "replay_artifact_invalid")
        path = Path(str(row.get("path") or ""))
        _require(path.is_absolute(), "replay_artifact_invalid")
        paths[name] = checked_file(path, dict(row))
    manifest = paths[prefix + "_artifact_manifest"]
    teardown = paths[prefix + "_teardown"]
    retained = tuple(paths[prefix + "_" + name] for name in (*names, "teardown"))
    _require(_teardown_valid(teardown) and _terminal_manifest_valid(
        manifest, lane=lane, required_paths=retained,
    ), "replay_terminal_artifacts_changed")

def execute_paid_stage(
    job: Mapping[str, Any],
    *,
    allocator_runner: Callable[..., int] = _default_allocator_runner,
) -> dict[str, Any]:
    """Execute one fixed paid phase or adopt its retained terminal result."""

    stage = job.get("stage_id")
    _require(stage in STAGES, "stage_invalid")
    profile = job.get("server_profile")
    _require(isinstance(profile, Mapping), "server_profile_missing")
    config = (profile.get("paid_stages") or {}).get(stage)
    _require(isinstance(config, Mapping), "stage_profile_missing:" + str(stage))
    output = Path(str(job.get("output_root") or ""))
    data_root = Path(str(job.get("server_data_root") or ""))
    repo = Path(str(job.get("repo_root") or ""))
    _require(output.is_absolute() and data_root.is_absolute() and repo.is_absolute()
             and data_root.is_dir() and repo.is_dir(), "runtime_root_invalid")
    roots = tuple(
        Path(value).resolve()
        for value in profile.get("approved_paid_input_roots", [str(data_root), str(repo)])
    )
    _require(roots and all(root.is_absolute() and root.exists() and not root.is_symlink()
                           for root in roots), "approved_roots_invalid")
    output = _resident(output, roots, "output_root_invalid")
    commit = str(job.get("expected_source_commit") or "")
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None
             and profile.get("source_commit") == commit, "source_commit_invalid")
    # The queue owns the phase directory; this handler owns its artifact child.
    # Require an existing directory only on reconciliation, never manufacture
    # a missing prior attempt as a fresh execution.
    if job.get("resume_only"):
        _require(output.is_dir(), "prior_output_missing")
    else:
        output.mkdir(parents=True, exist_ok=True)
    if stage == "sam31_tracking":
        return _sam31_stage(
            job, roots=roots, output=output, repo=repo, config=config,
            allocator_runner=allocator_runner,
        )
    return _contribution_stage(
        job, roots=roots, output=output, repo=repo, config=config,
        allocator_runner=allocator_runner,
    )


__all__ = ["execute_paid_stage", "validate_retained_paid_stage"]
