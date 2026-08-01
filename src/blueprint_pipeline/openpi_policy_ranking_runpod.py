"""Provider request binding for the frozen OpenPI policy-ranking campaign.

This module deliberately exposes request construction and a no-mutation shape
check only. Paid creation remains owned by ``paid_resource_allocator`` so the
independent watchdog, budget reservation, pending-teardown record, and exact
protected-main checks cannot be bypassed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .gpu_render_providers import RenderLaunchSpec, get_render_provider
from .groot_oscar_runpod_watchdog import (
    terminate_canary_resources,
    write_owner_teardown_cancel_request,
)
from .new_site_diagnostic_canary_gpu import (
    INPUT_RECEIPT_SCHEMA_VERSION as CANARY_INPUT_RECEIPT_SCHEMA_VERSION,
)
from .openpi_current_reference_gpu_bundle import (
    INPUT_RECEIPT_SCHEMA_VERSION as CURRENT_REFERENCE_INPUT_RECEIPT_SCHEMA_VERSION,
)
from .openpi_policy_ranking_gpu_bootstrap import (
    EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY,
    EXECUTION_MODE_FULL_CAMPAIGN,
    EXECUTION_MODE_NEW_SITE_CANARY,
    POLICY_IDS,
)
from .openpi_policy_ranking_gpu_admission import (
    build_openpi_policy_ranking_gpu_admission,
    collect_openpi_policy_ranking_runpod_preflight,
    collect_openpi_policy_ranking_vast_preflight,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    mark_pending_teardown_ambiguous,
    load_pending_teardowns,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    release_paid_provider_lane_lease,
    transfer_paid_provider_compute_lane_lease_to_watchdog,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission,
)
from .production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)
from .runpod_provider_adapter import run_runpod_provider_adapter
from .safe_outbound_http import presigned_transfer_policy
from .safe_outbound_http import request as safe_http_request


SCHEMA_VERSION = "openpi_policy_ranking_runpod_launch.v1"
INPUT_SECRET_URL_ENV = "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL"
INPUT_SHA256_ENV = "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SHA256"
EXECUTION_MODE_ENV = "BLUEPRINT_OPENPI_EXECUTION_MODE"
RUNTIME_SOURCE_ARCHIVE_URL_ENV = "BLUEPRINT_RUNTIME_SOURCE_ARCHIVE_URL"
RUNTIME_SOURCE_ARCHIVE_SHA256_ENV = "BLUEPRINT_RUNTIME_SOURCE_ARCHIVE_SHA256"
RUNTIME_SOURCE_COMMIT_ENV = "BLUEPRINT_RUNTIME_SOURCE_COMMIT"
OUTPUT_SECRET_PUT_URL_ENV = "BLUEPRINT_OPENPI_POLICY_RANKING_OUTPUT_SECRET_PUT_URL"
GENERIC_OUTPUT_SECRET_URL_ENV = "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"
FORWARD_SECRET_ENV_NAMES_ENV = "BLUEPRINT_RUNPOD_FORWARD_SECRET_ENV_VARS"
WATCHDOG_EVIDENCE_NAME = "groot_oscar_runpod_canary_watchdog.json"
CANARY_NAME_PREFIX = "blueprint-groot-oscar-canary-openpi-ranking-"
PAID_LANE = "openpi_policy_ranking_gpu_canary"
MAX_OUTPUT_ARCHIVE_BYTES = 2 * 1024**3
MAX_OUTPUT_ARCHIVE_MEMBERS = 50_000
MAX_OUTPUT_UNCOMPRESSED_BYTES = 12 * 1024**3
MAX_CONSECUTIVE_TRANSIENT_OUTPUT_ERRORS = 3
CURRENT_REFERENCE_AUTHORIZATION_SCHEMA = (
    "policy_ranking_openpi_current_reference_compute_authorization.v1"
)
AUTHORIZATION_CONSUMPTION_ROOT = Path.home() / ".blueprint-spend-authority" / "consumed"


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _current_reference_authorization_blockers(
    authorization: Mapping[str, Any],
    *,
    input_bundle: Mapping[str, Any],
    expected_source_commit: str,
) -> list[str]:
    blockers: list[str] = []
    authorization_id = str(authorization.get("authorization_id") or "")
    manifest = input_bundle.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    runtime_source = manifest.get("runtime_source")
    runtime_source = runtime_source if isinstance(runtime_source, Mapping) else {}
    policy_ids = manifest.get("policy_ids")
    policy_ids = policy_ids if isinstance(policy_ids, list) else []
    if authorization.get("schema_version") != CURRENT_REFERENCE_AUTHORIZATION_SCHEMA:
        blockers.append("openpi_current_reference_authorization_schema_invalid")
    if authorization.get("experiment_id") != (
        "policy_ranking_real_policy_closed_loop_confirmation_20260730"
    ):
        blockers.append("openpi_current_reference_authorization_experiment_invalid")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{15,159}", authorization_id):
        blockers.append("openpi_current_reference_authorization_id_invalid")
    if (
        authorization.get("paid_mutation_authorized") is not True
        or authorization.get("single_use_consumption_required") is not True
        or authorization.get("maximum_provider_allocations") != 1
        or authorization.get("maximum_policy_requests") != 1
    ):
        blockers.append("openpi_current_reference_authorization_limits_invalid")
    maximum_concurrent_gpus = authorization.get("maximum_concurrent_gpus")
    if (
        isinstance(maximum_concurrent_gpus, bool)
        or not isinstance(maximum_concurrent_gpus, int)
        or maximum_concurrent_gpus not in {1, 2}
    ):
        blockers.append("openpi_current_reference_authorization_concurrency_invalid")
    if (
        authorization.get("runtime_source_commit") != expected_source_commit
        or runtime_source.get("commit") != expected_source_commit
    ):
        blockers.append("openpi_current_reference_authorization_runtime_mismatch")
    if authorization.get("input_bundle_sha256") != input_bundle.get("bundle_sha256"):
        blockers.append("openpi_current_reference_authorization_input_mismatch")
    policy_id = str(authorization.get("policy_id") or "")
    if len(policy_ids) == 1 and policy_id != str(policy_ids[0]):
        blockers.append("openpi_current_reference_authorization_policy_mismatch")
    query_index = authorization.get("query_index")
    if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 0:
        blockers.append("openpi_current_reference_authorization_query_index_invalid")
    if (
        authorization.get("prior_allocation_closed_provider_zero") is not True
        or authorization.get("physical_robot_endpoint_access_allowed") is not False
        or authorization.get("evaluator_or_vlm_spend_authorized_by_this_record") is not False
    ):
        blockers.append("openpi_current_reference_authorization_boundary_invalid")
    return sorted(set(blockers))


def _consume_current_reference_authorization_once(
    authorization: Mapping[str, Any],
    *,
    expected_source_commit: str,
    input_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = _current_reference_authorization_blockers(
        authorization,
        input_bundle=input_bundle,
        expected_source_commit=expected_source_commit,
    )
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    authorization_id = str(authorization["authorization_id"])
    root = AUTHORIZATION_CONSUMPTION_ROOT
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_mode = root.stat().st_mode
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["openpi_authorization_consumption_root_unavailable"],
        }
    if root_mode & 0o077:
        return {
            "status": "blocked",
            "blockers": ["openpi_authorization_consumption_root_insecure"],
        }
    record_path = root / f"{authorization_id}.json"
    record = {
        "schema_version": "openpi_current_reference_authorization_consumption.v1",
        "authorization_id": authorization_id,
        "authorization_sha256": hashlib.sha256(
            (json.dumps(dict(authorization), sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        ).hexdigest(),
        "experiment_id": authorization["experiment_id"],
        "source_commit": expected_source_commit,
        "input_bundle_sha256": input_bundle["bundle_sha256"],
        "consumed_at_epoch": time.time(),
        "maximum_provider_allocations": 1,
        "maximum_policy_requests": 1,
    }
    record_bytes = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    temporary_path = root / f".{authorization_id}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    try:
        descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["openpi_authorization_consumption_write_failed"],
            "authorization_id": authorization_id,
        }
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(record_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, record_path)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["openpi_current_reference_authorization_already_consumed"],
            "authorization_id": authorization_id,
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["openpi_authorization_consumption_write_failed"],
            "authorization_id": authorization_id,
        }
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "status": "consumed",
        "authorization_id": authorization_id,
        "consumption_record_sha256": record_sha256,
        "record_location_disclosed": False,
    }


def _read_private_https_url(path: str | Path, *, field: str) -> str:
    candidate = Path(path).expanduser()
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError(f"{field}_no_follow_unavailable")
    descriptor = os.open(candidate, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field}_not_regular")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError(f"{field}_not_private_0600")
        value = os.read(descriptor, 64 * 1024).decode("utf-8").strip()
    finally:
        os.close(descriptor)
    if not value.startswith("https://") or any(char.isspace() for char in value):
        raise ValueError(f"{field}_not_https_url")
    return value


def _wait_for_watchdog(
    *, root: Path, process: subprocess.Popen[Any], prefix: str, deadline: float
) -> dict[str, Any]:
    evidence_path = root / WATCHDOG_EVIDENCE_NAME
    until = time.time() + 10
    while time.time() < until:
        if process.poll() is not None:
            break
        try:
            value = _read_object(evidence_path)
        except (OSError, UnicodeError, ValueError):
            time.sleep(0.05)
            continue
        if (
            value.get("status") == "armed"
            and value.get("independent_process") is True
            and value.get("pid") == process.pid
            and value.get("pod_name_prefix") == prefix
            and value.get("deadline_epoch") == deadline
        ):
            return value
        time.sleep(0.05)
    raise RuntimeError("openpi_runpod_watchdog_not_confirmed_armed")


def _stop_watchdog_after_provider_zero(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.kill(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output_archive(archive_bytes: bytes) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile:
        return {
            "status": "blocked",
            "blockers": ["openpi_output_archive_invalid_zip"],
            "terminal_output_present": True,
        }
    with archive:
        members = archive.infolist()
        names: set[str] = set()
        total_uncompressed = 0
        if not members or len(members) > MAX_OUTPUT_ARCHIVE_MEMBERS:
            blockers.append("openpi_output_archive_member_count_invalid")
        for member in members:
            parts = Path(member.filename).parts
            if (
                member.filename in names
                or Path(member.filename).is_absolute()
                or ".." in parts
                or member.flag_bits & 0x1
            ):
                blockers.append("openpi_output_archive_member_unsafe")
            names.add(member.filename)
            total_uncompressed += int(member.file_size)
        if total_uncompressed > MAX_OUTPUT_UNCOMPRESSED_BYTES:
            blockers.append("openpi_output_archive_uncompressed_size_exceeded")
        current_manifest_name = "openpi_current_reference_policy_canary.json"
        if current_manifest_name in names:
            try:
                value = json.loads(archive.read(current_manifest_name).decode("utf-8"))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                value = {}
                blockers.append("openpi_output_current_reference_manifest_unreadable")
            manifest = dict(value) if isinstance(value, Mapping) else {}
            if manifest.get("schema_version") != "openpi_current_reference_policy_canary.v1":
                blockers.append("openpi_output_current_reference_schema_invalid")
            declared_sha = str(manifest.get("manifest_sha256") or "")
            digest_payload = dict(manifest)
            digest_payload.pop("manifest_sha256", None)
            from .policy_ranking_thesis import canonical_sha256

            if declared_sha != canonical_sha256(digest_payload):
                blockers.append("openpi_output_current_reference_manifest_sha256_mismatch")
            declared_policies = tuple(manifest.get("frozen_policy_order") or [])
            initial_policies = ("pi0_droid", "pi0_fast_droid", "pi05_droid")
            query_mode = manifest.get("query_mode")
            expected_policies: tuple[str, ...]
            if query_mode in {None, "initial_identity_canary"} and (
                declared_policies == initial_policies
            ):
                expected_policies = initial_policies
            elif (
                query_mode == "same_candidate_policy_requery"
                and len(declared_policies) == 1
                and declared_policies[0] in initial_policies
                and manifest.get("same_candidate_policy_id") == declared_policies[0]
            ):
                expected_policies = declared_policies
            else:
                expected_policies = ()
                blockers.append("openpi_output_current_reference_policy_scope_invalid")
            policy_results = manifest.get("policy_results")
            policy_results = policy_results if isinstance(policy_results, list) else []
            status = str(manifest.get("status") or "")
            if status == "completed":
                if (
                    declared_policies != expected_policies
                    or manifest.get("requests_per_policy") != 1
                    or manifest.get("artifact_path_mode") != "result_root_relative"
                    or len(policy_results) != len(expected_policies)
                    or any(
                        not isinstance(row, Mapping) or row.get("status") != "completed"
                        for row in policy_results
                    )
                ):
                    blockers.append("openpi_output_current_reference_completion_invalid")
                result_by_policy = {
                    str(row.get("policy_id") or ""): row
                    for row in policy_results
                    if isinstance(row, Mapping)
                }
                for policy_id in expected_policies:
                    action_name = f"{policy_id}_native_action.npy"
                    receipt_name = f"{policy_id}_policy_receipt.json"
                    if action_name not in names or receipt_name not in names:
                        blockers.append(
                            f"openpi_output_current_reference_policy_artifacts_missing:{policy_id}"
                        )
                        continue
                    try:
                        receipt_value = json.loads(archive.read(receipt_name).decode("utf-8"))
                        receipt = dict(receipt_value) if isinstance(receipt_value, Mapping) else {}
                        receipt_digest_payload = dict(receipt)
                        receipt_manifest_sha256 = str(
                            receipt_digest_payload.pop("manifest_sha256", "")
                        )
                        receipt_bytes = archive.read(receipt_name)
                        action_bytes = archive.read(action_name)
                        action = np.load(io.BytesIO(action_bytes), allow_pickle=False)
                        expected_shape = (15, 8) if policy_id == "pi05_droid" else (10, 8)
                        result_row = result_by_policy.get(policy_id, {})
                        if (
                            receipt.get("schema_version")
                            != "openpi_current_reference_policy_query_receipt.v1"
                            or receipt.get("policy_id") != policy_id
                            or receipt.get("artifact_path_mode") != "result_root_relative"
                            or receipt.get("native_action_path") != action_name
                            or receipt.get("native_action_file_sha256")
                            != hashlib.sha256(action_bytes).hexdigest()
                            or receipt.get("physical_outcome_accessed") is not False
                            or receipt.get("wam_called") is not False
                            or receipt_manifest_sha256 != canonical_sha256(receipt_digest_payload)
                            or action.shape != expected_shape
                            or action.dtype != np.float64
                            or not np.isfinite(action).all()
                            or result_row.get("artifact_path_mode") != "result_root_relative"
                            or result_row.get("receipt_path") != receipt_name
                            or result_row.get("receipt_file_sha256")
                            != hashlib.sha256(receipt_bytes).hexdigest()
                            or result_row.get("receipt_manifest_sha256") != receipt_manifest_sha256
                            or result_row.get("native_action_shape") != list(expected_shape)
                            or result_row.get("native_action_file_sha256")
                            != hashlib.sha256(action_bytes).hexdigest()
                        ):
                            blockers.append(
                                "openpi_output_current_reference_policy_artifacts_invalid:"
                                f"{policy_id}"
                            )
                    except (UnicodeError, ValueError, json.JSONDecodeError):
                        blockers.append(
                            f"openpi_output_current_reference_policy_artifacts_unreadable:{policy_id}"
                        )
            elif status == "blocked":
                if not manifest.get("blockers"):
                    blockers.append("openpi_output_current_reference_blocker_missing")
            else:
                blockers.append("openpi_output_current_reference_not_terminal")
            if (
                manifest.get("wam_called") is not False
                or manifest.get("judge_called") is not False
                or manifest.get("physical_outcome_accessed") is not False
            ):
                blockers.append("openpi_output_current_reference_claim_boundary_invalid")
            return {
                "schema_version": "openpi_policy_ranking_output_validation.v1",
                "status": "completed" if not blockers else "blocked",
                "execution_mode": EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY,
                "blockers": sorted(set(blockers)),
                "terminal_output_present": True,
                "campaign_status": manifest.get("status"),
                "campaign_manifest": manifest,
                "policy_result_count": len(policy_results),
                "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "archive_size_bytes": len(archive_bytes),
                "archive_member_count": len(members),
                "raw_secret_values_recorded": False,
            }
        canary_manifest_name = "new_site_diagnostic_canary_gpu.json"
        if canary_manifest_name in names:
            try:
                value = json.loads(archive.read(canary_manifest_name).decode("utf-8"))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                value = {}
                blockers.append("openpi_output_canary_manifest_unreadable")
            manifest = dict(value) if isinstance(value, Mapping) else {}
            if manifest.get("schema_version") != "new_site_diagnostic_canary_gpu.v1":
                blockers.append("openpi_output_canary_schema_invalid")
            declared_sha = str(manifest.get("manifest_sha256") or "")
            digest_payload = dict(manifest)
            digest_payload.pop("manifest_sha256", None)
            from .policy_ranking_thesis import canonical_sha256

            if declared_sha != canonical_sha256(digest_payload):
                blockers.append("openpi_output_canary_manifest_sha256_mismatch")
            canary_status = str(manifest.get("status") or "")
            canary = manifest.get("canary")
            canary = canary if isinstance(canary, Mapping) else {}
            if canary_status == "completed":
                if (
                    manifest.get("arm_id") != "skeleton_only"
                    or canary.get("status") not in {"passed", "failed"}
                    or canary.get("label_free") is not True
                    or canary.get("model_invoked") is not True
                ):
                    blockers.append("openpi_output_completed_canary_invalid")
                expected_media = {
                    "external": any(
                        name.endswith("query_000_external_skeleton.mp4") for name in names
                    ),
                    "wrist": any(name.endswith("query_000_wrist_skeleton.mp4") for name in names),
                }
                if not all(expected_media.values()):
                    blockers.append("openpi_output_canary_individual_camera_media_missing")
            elif canary_status == "blocked":
                if not manifest.get("blockers"):
                    blockers.append("openpi_output_blocked_canary_reason_missing")
                expected_media = {}
            else:
                expected_media = {}
                blockers.append("openpi_output_canary_not_terminal")
            claim = manifest.get("claim_boundary")
            claim = claim if isinstance(claim, Mapping) else {}
            if (
                claim.get("ranking_accuracy") is not False
                or claim.get("physical_success") is not False
                or claim.get("captured_site_transfer_validation") is not False
                or claim.get("phase_b_confirmation") is not False
            ):
                blockers.append("openpi_output_canary_claim_boundary_invalid")
            return {
                "schema_version": "openpi_policy_ranking_output_validation.v1",
                "status": "completed" if not blockers else "blocked",
                "execution_mode": EXECUTION_MODE_NEW_SITE_CANARY,
                "blockers": sorted(set(blockers)),
                "terminal_output_present": True,
                "campaign_status": manifest.get("status"),
                "campaign_manifest": manifest,
                "episode_record_count": 0,
                "scene_ids": [str((canary.get("freeze_bindings") or {}).get("scene_id") or "")],
                "individual_camera_media_present": expected_media,
                "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "archive_size_bytes": len(archive_bytes),
                "archive_member_count": len(members),
                "raw_secret_values_recorded": False,
            }
        manifest_name = "openpi_policy_ranking_gpu_job.json"
        if manifest_name not in names:
            blockers.append("openpi_output_campaign_manifest_missing")
            manifest: dict[str, Any] = {}
        else:
            try:
                value = json.loads(archive.read(manifest_name).decode("utf-8"))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                value = {}
                blockers.append("openpi_output_campaign_manifest_unreadable")
            manifest = dict(value) if isinstance(value, Mapping) else {}
        if manifest.get("schema_version") != "openpi_policy_ranking_gpu_job.v1":
            blockers.append("openpi_output_campaign_schema_invalid")
        declared_sha = str(manifest.get("manifest_sha256") or "")
        digest_payload = dict(manifest)
        digest_payload.pop("manifest_sha256", None)
        from .policy_ranking_thesis import canonical_sha256

        if declared_sha != canonical_sha256(digest_payload):
            blockers.append("openpi_output_campaign_manifest_sha256_mismatch")
        inputs = manifest.get("inputs")
        inputs = inputs if isinstance(inputs, Mapping) else {}
        scenes = inputs.get("scenes")
        scenes = scenes if isinstance(scenes, list) else []
        scene_ids = {str(row.get("scene_id") or "") for row in scenes if isinstance(row, Mapping)}
        scene_kinds = {
            str(row.get("scene_kind") or "") for row in scenes if isinstance(row, Mapping)
        }
        if len(scene_ids) != 2 or scene_kinds != {
            "captured_3dgs",
            "controlled_nvidia_usd",
        }:
            blockers.append("openpi_output_scene_cohort_invalid")
        policy_ids = inputs.get("policy_ids")
        policy_ids = policy_ids if isinstance(policy_ids, list) else []
        if tuple(policy_ids) != tuple(POLICY_IDS):
            blockers.append("openpi_output_policy_cohort_invalid")
        policy_runs = manifest.get("policy_runs")
        policy_runs = policy_runs if isinstance(policy_runs, list) else []
        if len(policy_runs) != len(POLICY_IDS):
            blockers.append("openpi_output_policy_run_count_invalid")
        completed_records = 0
        for run in policy_runs:
            run = run if isinstance(run, Mapping) else {}
            records = run.get("episode_records")
            records = records if isinstance(records, list) else []
            completed_records += len(records)
            for record in records:
                record = record if isinstance(record, Mapping) else {}
                policy_id = str(run.get("policy_id") or "")
                scene_id = str(record.get("scene_id") or "")
                variant_id = str(record.get("variant_id") or "")
                episode_name = f"{policy_id}/{scene_id}/{variant_id}/franka_droid_closed_loop.json"
                if episode_name not in names:
                    blockers.append("openpi_output_episode_manifest_missing")
                    continue
                try:
                    episode = json.loads(archive.read(episode_name).decode("utf-8"))
                except (UnicodeError, ValueError, json.JSONDecodeError):
                    blockers.append("openpi_output_episode_manifest_unreadable")
                    continue
                if episode.get("manifest_sha256") != record.get("episode_manifest_sha256"):
                    blockers.append("openpi_output_episode_manifest_binding_mismatch")
        campaign_status = str(manifest.get("status") or "")
        if campaign_status == "completed":
            if completed_records != 24 or any(
                not isinstance(run, Mapping) or run.get("status") != "completed"
                for run in policy_runs
            ):
                blockers.append("openpi_output_completed_campaign_episode_count_invalid")
            rankings = manifest.get("rankings")
            rankings = rankings if isinstance(rankings, Mapping) else {}
            if set(rankings) != scene_ids:
                blockers.append("openpi_output_completed_campaign_rankings_invalid")
        elif campaign_status == "blocked":
            if not manifest.get("blockers"):
                blockers.append("openpi_output_blocked_campaign_reason_missing")
        else:
            blockers.append("openpi_output_campaign_not_terminal")
        claim = manifest.get("claim_boundary")
        claim = claim if isinstance(claim, Mapping) else {}
        if (
            claim.get("site_specific_physical_success_proven") is not False
            or claim.get("physical_robot_endpoint_contacted") is not False
            or claim.get("physical_robot_operated") is not False
        ):
            blockers.append("openpi_output_physical_claim_boundary_invalid")
    return {
        "schema_version": "openpi_policy_ranking_output_validation.v1",
        "status": "completed" if not blockers else "blocked",
        "execution_mode": EXECUTION_MODE_FULL_CAMPAIGN,
        "blockers": sorted(set(blockers)),
        "terminal_output_present": True,
        "campaign_status": manifest.get("status"),
        "campaign_manifest": manifest,
        "episode_record_count": completed_records,
        "scene_ids": sorted(scene_ids),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size_bytes": len(archive_bytes),
        "archive_member_count": len(members),
        "raw_secret_values_recorded": False,
    }


def _wait_for_watchdog_terminal(root: Path, *, timeout_seconds: float = 45.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    evidence_path = root / WATCHDOG_EVIDENCE_NAME
    observed: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            observed = _read_object(evidence_path)
        except (OSError, UnicodeError, ValueError):
            time.sleep(0.25)
            continue
        settlement = observed.get("campaign_budget_settlement")
        settlement = settlement if isinstance(settlement, Mapping) else {}
        if (
            observed.get("provider_absence_confirmed") is True
            and observed.get("control_plane_terminal") is True
            and settlement.get("status") == "settled"
        ):
            return observed
        time.sleep(0.25)
    return observed


DEFAULT_PROVIDER_STARTUP_TIMEOUT_SECONDS = 480.0
_PROVIDER_TERMINAL_STATES = {
    "cancelled",
    "destroyed",
    "error",
    "exited",
    "failed",
    "stopped",
    "terminated",
}


def _provider_runtime_inspection(provider: Any, pod_id: str) -> dict[str, Any]:
    inspect = getattr(provider, "inspect", None)
    if not callable(inspect):
        return {"status": "not_supported", "raw_secret_values_recorded": False}
    try:
        value = inspect(pod_id)
    except Exception as exc:  # noqa: BLE001 - the output watchdog remains authoritative
        return {
            "status": "inspection_failed",
            "error_type": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return (
        dict(value)
        if isinstance(value, Mapping)
        else {
            "status": "invalid_response",
            "raw_secret_values_recorded": False,
        }
    )


def _close_provider_without_output(
    *,
    root: Path,
    provider: Any,
    armed: Mapping[str, Any],
    pod_id: str,
    provider_name: str,
    blocker: str,
    inspection: Mapping[str, Any],
) -> dict[str, Any]:
    teardown = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=CANARY_NAME_PREFIX,
        armed=armed,
        provider_name=provider_name,
    )
    global_inventory = provider.billable_inventory(name_prefix="")
    global_absent = bool(
        global_inventory.get("api_confirmed") is True
        and global_inventory.get("live_resource_count") == 0
    )
    absence_proven = bool(teardown.get("provider_absence_confirmed") is True and global_absent)
    if absence_proven:
        write_owner_teardown_cancel_request(
            root=root,
            pod_name_prefix=CANARY_NAME_PREFIX,
            provider_name=provider_name,
            instance_id=pod_id,
        )
    watchdog_terminal = _wait_for_watchdog_terminal(root) if absence_proven else {}
    settlement = watchdog_terminal.get("campaign_budget_settlement")
    settlement = settlement if isinstance(settlement, Mapping) else {}
    control_terminal = bool(
        absence_proven
        and watchdog_terminal.get("control_plane_terminal") is True
        and settlement.get("status") == "settled"
    )
    result = {
        "schema_version": "openpi_policy_ranking_monitor.v1",
        "status": (
            "provider_terminal_without_output"
            if control_terminal
            else "provider_terminal_without_output_watchdog_retained"
        ),
        "blockers": sorted(
            set(
                [
                    blocker,
                    *([] if absence_proven else ["openpi_provider_absence_unverified"]),
                    *([] if control_terminal else ["openpi_control_plane_not_terminal"]),
                ]
            )
        ),
        "provider_runtime_inspection": dict(inspection),
        "teardown": teardown,
        "final_global_inventory": global_inventory,
        "provider_absence_confirmed": absence_proven,
        "control_plane_terminal": control_terminal,
        "campaign_budget_settlement": dict(settlement),
        "continuing_spend": not control_terminal,
        "raw_secret_values_recorded": False,
    }
    write_json(root / "openpi_policy_ranking_monitor.json", result)
    return result


def _monitor_openpi_output_and_teardown(
    *,
    root: Path,
    output_secret_get_url: str,
    provider: Any,
    armed: Mapping[str, Any],
    pod_id: str,
    provider_name: str,
    deadline_epoch: float,
    poll_interval_seconds: float = 15.0,
    provider_startup_timeout_seconds: float = DEFAULT_PROVIDER_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_path = root / "openpi_policy_ranking_provider_output.zip"
    response_bytes: bytes | None = None
    http_status: int | None = None
    consecutive_transient_errors = 0
    monitor_started_epoch = time.time()
    while time.time() < deadline_epoch - 60:
        try:
            response = safe_http_request(
                output_secret_get_url,
                method="GET",
                timeout_seconds=60,
                policy=presigned_transfer_policy(
                    output_secret_get_url,
                    max_response_bytes=MAX_OUTPUT_ARCHIVE_BYTES,
                ),
                max_response_bytes=MAX_OUTPUT_ARCHIVE_BYTES,
            )
            http_status = response.status
            consecutive_transient_errors = 0
            if response.status == 200 and response.body:
                response_bytes = response.body
                break
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            consecutive_transient_errors = 0
            if exc.code not in {403, 404}:
                break
        except urllib.error.URLError as exc:
            consecutive_transient_errors += 1
            if consecutive_transient_errors >= MAX_CONSECUTIVE_TRANSIENT_OUTPUT_ERRORS:
                return {
                    "status": "monitor_failed_watchdog_retained",
                    "blockers": [f"openpi_output_monitor_failed:{type(exc).__name__}"],
                    "transient_error_attempts": consecutive_transient_errors,
                    "continuing_spend": True,
                    "watchdog_deadline_epoch": deadline_epoch,
                    "raw_secret_values_recorded": False,
                }
        except Exception as exc:  # noqa: BLE001 - teardown still owns the deadline
            return {
                "status": "monitor_failed_watchdog_retained",
                "blockers": [f"openpi_output_monitor_failed:{type(exc).__name__}"],
                "continuing_spend": True,
                "watchdog_deadline_epoch": deadline_epoch,
                "raw_secret_values_recorded": False,
            }
        inspection = _provider_runtime_inspection(provider, pod_id)
        actual_status = (
            str(inspection.get("actual_status") or inspection.get("desiredStatus") or "")
            .strip()
            .lower()
        )
        provider_absent = inspection.get("provider_absence_confirmed") is True
        terminal = provider_absent or actual_status in _PROVIDER_TERMINAL_STATES
        runtime_ready = bool(
            inspection.get("direct_port_ready") is True
            or inspection.get("runtime_ready") is True
            or actual_status == "running"
        )
        startup_timed_out = bool(
            provider_startup_timeout_seconds >= 0
            and time.time() - monitor_started_epoch >= provider_startup_timeout_seconds
            and not runtime_ready
            and actual_status in {"", "created", "initializing", "loading", "queued", "starting"}
        )
        if terminal or startup_timed_out:
            return _close_provider_without_output(
                root=root,
                provider=provider,
                armed=armed,
                pod_id=pod_id,
                provider_name=provider_name,
                blocker=(
                    "openpi_provider_terminal_without_output"
                    if terminal
                    else "openpi_provider_startup_timeout_without_output"
                ),
                inspection=inspection,
            )
        time.sleep(max(0.1, poll_interval_seconds))
    if response_bytes is None:
        return {
            "status": "output_not_observed_watchdog_retained",
            "blockers": [f"openpi_output_not_observed:http_{http_status}"],
            "continuing_spend": True,
            "watchdog_deadline_epoch": deadline_epoch,
            "raw_secret_values_recorded": False,
        }
    output_path.write_bytes(response_bytes)
    validation = _validate_output_archive(response_bytes)
    write_json(root / "openpi_policy_ranking_output_validation.json", validation)
    teardown = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=CANARY_NAME_PREFIX,
        armed=armed,
        provider_name=provider_name,
    )
    global_inventory = provider.billable_inventory(name_prefix="")
    global_absent = bool(
        global_inventory.get("api_confirmed") is True
        and global_inventory.get("live_resource_count") == 0
    )
    absence_proven = bool(teardown.get("provider_absence_confirmed") is True and global_absent)
    if absence_proven:
        write_owner_teardown_cancel_request(
            root=root,
            pod_name_prefix=CANARY_NAME_PREFIX,
            provider_name=provider_name,
            instance_id=pod_id,
        )
    watchdog_terminal = _wait_for_watchdog_terminal(root) if absence_proven else {}
    settlement = watchdog_terminal.get("campaign_budget_settlement")
    settlement = settlement if isinstance(settlement, Mapping) else {}
    control_terminal = bool(
        absence_proven
        and watchdog_terminal.get("control_plane_terminal") is True
        and settlement.get("status") == "settled"
    )
    result = {
        "schema_version": "openpi_policy_ranking_monitor.v1",
        "status": (
            "completed"
            if validation.get("status") == "completed"
            and validation.get("campaign_status") == "completed"
            and control_terminal
            else "blocked"
        ),
        "blockers": sorted(
            set(
                [
                    *(validation.get("blockers") or []),
                    *([] if absence_proven else ["openpi_provider_absence_unverified"]),
                    *([] if control_terminal else ["openpi_control_plane_not_terminal"]),
                ]
            )
        ),
        "campaign_status": validation.get("campaign_status"),
        "output_validation": validation,
        "teardown": teardown,
        "final_global_inventory": global_inventory,
        "provider_absence_confirmed": absence_proven,
        "control_plane_terminal": control_terminal,
        "campaign_budget_settlement": dict(settlement),
        "continuing_spend": not control_terminal,
        "raw_secret_values_recorded": False,
    }
    write_json(root / "openpi_policy_ranking_monitor.json", result)
    return result


def _handoff_cleanup_to_watchdog(
    *,
    root: Path,
    provider: Any,
    cleanup: Mapping[str, Any],
    instance_id: str,
    provider_name: str,
) -> dict[str, Any]:
    global_inventory = provider.billable_inventory(name_prefix="")
    absence_proven = bool(
        cleanup.get("provider_absence_confirmed") is True
        and global_inventory.get("api_confirmed") is True
        and global_inventory.get("live_resource_count") == 0
    )
    cancel_request = (
        write_owner_teardown_cancel_request(
            root=root,
            pod_name_prefix=CANARY_NAME_PREFIX,
            provider_name=provider_name,
            instance_id=instance_id,
        )
        if absence_proven
        else {}
    )
    watchdog_terminal = _wait_for_watchdog_terminal(root) if cancel_request else {}
    settlement = watchdog_terminal.get("campaign_budget_settlement")
    settlement = settlement if isinstance(settlement, Mapping) else {}
    control_terminal = bool(
        watchdog_terminal.get("control_plane_terminal") is True
        and settlement.get("status") == "settled"
    )
    return {
        "provider_absence_confirmed": absence_proven,
        "global_inventory": global_inventory,
        "watchdog_cancel_requested": bool(cancel_request),
        "watchdog_terminal": watchdog_terminal,
        "control_plane_terminal": control_terminal,
        "continuing_spend": not control_terminal,
    }


def _execution_mode_for_input_bundle(input_bundle: Mapping[str, Any]) -> str:
    schema = input_bundle.get("schema_version")
    if schema == CURRENT_REFERENCE_INPUT_RECEIPT_SCHEMA_VERSION:
        return EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY
    if schema == CANARY_INPUT_RECEIPT_SCHEMA_VERSION:
        return EXECUTION_MODE_NEW_SITE_CANARY
    return EXECUTION_MODE_FULL_CAMPAIGN


def _runtime_source_bootstrap_script(execution_mode: str) -> str:
    if execution_mode != EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY:
        return (
            "exec /.venv/bin/python -m blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap run"
        )
    return r"""set -euo pipefail
exec /.venv/bin/python - <<'PY'
import hashlib
import os
import pathlib
import shutil
import tarfile
import urllib.request

url = os.environ["BLUEPRINT_RUNTIME_SOURCE_ARCHIVE_URL"]
expected_sha = os.environ["BLUEPRINT_RUNTIME_SOURCE_ARCHIVE_SHA256"]
commit = os.environ["BLUEPRINT_RUNTIME_SOURCE_COMMIT"]
expected_url = (
    "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/"
    + commit
)
if url != expected_url or len(commit) != 40 or len(expected_sha) != 64:
    raise SystemExit("runtime_source_identity_invalid")
archive_path = pathlib.Path("/workspace/blueprint-runtime-source.tar.gz")
request = urllib.request.Request(url, method="GET")
digest = hashlib.sha256()
size = 0
with urllib.request.urlopen(request, timeout=120) as response:
    if response.geturl() != url:
        raise SystemExit("runtime_source_redirect_forbidden")
    with archive_path.open("xb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 32 * 1024 * 1024:
                raise SystemExit("runtime_source_archive_too_large")
            digest.update(chunk)
            handle.write(chunk)
if digest.hexdigest() != expected_sha:
    raise SystemExit("runtime_source_archive_sha256_mismatch")
destination = pathlib.Path("/workspace/blueprint-runtime-source")
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir()
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    roots = {pathlib.PurePosixPath(member.name).parts[0] for member in members}
    if len(roots) != 1 or not next(iter(roots)).endswith(commit):
        raise SystemExit("runtime_source_archive_root_invalid")
    root_name = next(iter(roots))
    skipped_symlinks = {
        ".agents/skills/gstack",
        ".claude/skills/gstack",
    }
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root_name:
            raise SystemExit("runtime_source_archive_member_unsafe")
        relative = pathlib.PurePosixPath(*path.parts[1:]).as_posix()
        if member.issym():
            if relative not in skipped_symlinks:
                raise SystemExit("runtime_source_archive_symlink_not_allowlisted")
            continue
        if member.islnk() or not (member.isdir() or member.isfile()):
            raise SystemExit("runtime_source_archive_member_unsafe")
        target = destination.joinpath(*path.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit("runtime_source_archive_file_unreadable")
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
root = destination / next(iter(roots))
module = root / "src/blueprint_pipeline/openpi_policy_ranking_gpu_bootstrap.py"
if not module.is_file():
    raise SystemExit("runtime_source_bootstrap_module_missing")
env = dict(os.environ)
env["PYTHONPATH"] = str(root / "src") + (
    ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
)
os.execve(
    "/.venv/bin/python",
    [
        "/.venv/bin/python",
        "-m",
        "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap",
        "run",
    ],
    env,
)
PY"""


def _build_vast_launch_request(
    *,
    provider: Any,
    root: Path,
    pod_name: str,
    release: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    input_secret_url: str,
    output_secret_put_url: str,
) -> dict[str, Any]:
    capacity_request = preflight.get("capacity_request")
    capacity_request = capacity_request if isinstance(capacity_request, Mapping) else {}
    execution_mode = _execution_mode_for_input_bundle(input_bundle)
    manifest = input_bundle.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    runtime_source = manifest.get("runtime_source")
    runtime_source = runtime_source if isinstance(runtime_source, Mapping) else {}
    environment = {
        INPUT_SECRET_URL_ENV: input_secret_url,
        INPUT_SHA256_ENV: str(input_bundle["bundle_sha256"]),
        EXECUTION_MODE_ENV: execution_mode,
        OUTPUT_SECRET_PUT_URL_ENV: output_secret_put_url,
        GENERIC_OUTPUT_SECRET_URL_ENV: output_secret_put_url,
    }
    if execution_mode == EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY:
        environment.update(
            {
                RUNTIME_SOURCE_ARCHIVE_URL_ENV: str(runtime_source["archive_url"]),
                RUNTIME_SOURCE_ARCHIVE_SHA256_ENV: str(runtime_source["archive_sha256"]),
                RUNTIME_SOURCE_COMMIT_ENV: str(runtime_source["commit"]),
            }
        )
    spec = RenderLaunchSpec(
        name=pod_name,
        image=str(release["resolved_digest_ref"]),
        env=environment,
        bootstrap_argv=[
            "-lc",
            _runtime_source_bootstrap_script(execution_mode),
        ],
        entrypoint=["bash"],
        container_disk_gb=int(preflight["container_disk_bytes"]) // 1024**3,
        volume_gb=0,
        max_hourly_rate_usd=float(preflight["on_demand_price_usd_per_hour"]),
        # Keep offer fallback bound to the preregistered floor.  Using the
        # snapshot offer's exact VRAM here would silently make a later launch
        # more restrictive than the frozen campaign contract.
        min_gpu_ram_mb=int(
            capacity_request.get("min_gpu_ram_mb")
            or int(preflight["gpu_memory_bytes"]) // 1_000_000
        ),
        requires_rtx=False,
        vast_launch_mode="args",
    )
    request = provider.build_request(spec, root)
    request.update(
        {
            "prelaunch_spend_guard": {
                "required_before_provider_launch": True,
                "can_launch": True,
                "blockers": [],
            },
            "min_reliability": capacity_request.get("min_reliability"),
            "require_avx": True,
            "require_known_supported_isaac_driver": False,
            "preferred_gpu_keywords": capacity_request.get("preferred_gpu_keywords"),
        }
    )
    return request


def build_openpi_policy_ranking_provider_request(
    *,
    release: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    spend: Mapping[str, Any],
    expected_source_commit: str,
    job_id: str,
) -> dict[str, Any]:
    admission = build_openpi_policy_ranking_gpu_admission(
        release=release,
        input_bundle=input_bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=expected_source_commit,
    )
    if admission["status"] != "admitted":
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": admission["blockers"],
            "admission": admission,
            "provider_mutations_performed": 0,
        }
    ttl = int(spend["hard_ttl_seconds"])
    image_ref = str(release["resolved_digest_ref"])
    bundle_sha = str(input_bundle["bundle_sha256"])
    gpu_type_id = str(preflight["gpu_type_id"])
    provider_name = str(preflight["provider"])
    execution_mode = str(admission["execution_mode"])
    operation = {
        EXECUTION_MODE_NEW_SITE_CANARY: "execute_frozen_new_site_diagnostic_canary",
        EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY: (
            "execute_current_reference_real_policy_identity_canary"
        ),
    }.get(execution_mode, "execute_frozen_openpi_policy_ranking_campaign")
    manifest = input_bundle.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    runtime_source = manifest.get("runtime_source")
    runtime_source = runtime_source if isinstance(runtime_source, Mapping) else {}
    source_overlay = execution_mode == EXECUTION_MODE_CURRENT_REFERENCE_POLICY_CANARY
    plaintext_values = {
        INPUT_SHA256_ENV: bundle_sha,
        EXECUTION_MODE_ENV: execution_mode,
    }
    if source_overlay:
        plaintext_values.update(
            {
                RUNTIME_SOURCE_ARCHIVE_URL_ENV: str(runtime_source["archive_url"]),
                RUNTIME_SOURCE_ARCHIVE_SHA256_ENV: str(runtime_source["archive_sha256"]),
                RUNTIME_SOURCE_COMMIT_ENV: str(runtime_source["commit"]),
            }
        )
    request: dict[str, Any] = {
        "schema_version": "robot_eval_gpu_provider_launch_request.v1",
        "job_id": job_id,
        "provider": provider_name,
        "status": "request_manifest_ready",
        "operation": operation,
        "prelaunch_spend_guard": {
            "required_before_provider_launch": True,
            "can_launch": True,
            "blockers": [],
        },
        "provider_request_shape": {
            "api_payload_is_provider_adapter_template": True,
            "api_payload_values_are_redacted": True,
            "operation": operation,
            "image": {
                "configured_image_ref": image_ref,
                "configured_image_ref_is_versioned": True,
                "configured_image_ref_fetchable_by_provider": True,
            },
            "docker_entrypoint": (
                ["bash", "-lc"]
                if source_overlay
                else [
                    "/.venv/bin/python",
                    "-m",
                    "blueprint_pipeline.openpi_policy_ranking_gpu_bootstrap",
                    "run",
                ]
            ),
            "docker_start_cmd": (
                [_runtime_source_bootstrap_script(execution_mode)] if source_overlay else []
            ),
            "environment": {
                "secret_env_var_names": [
                    INPUT_SECRET_URL_ENV,
                    OUTPUT_SECRET_PUT_URL_ENV,
                    GENERIC_OUTPUT_SECRET_URL_ENV,
                ],
                "plaintext_env_var_names": sorted(plaintext_values),
                "plaintext_env_values": plaintext_values,
                "secret_values_in_artifact": False,
                "customer_visible_secret_values_allowed": False,
            },
            "inputs": {
                "manifest_uri_required_for_provider": True,
                "manifest_uri": "private-signed-input-url:not-persisted",
                "manifest_uri_fetchable_by_provider": True,
                "capture_root_bundle_uri_required_for_provider": True,
                "capture_root_bundle_uri": "private-signed-input-url:not-persisted",
                "capture_root_bundle_uri_fetchable_by_provider": True,
                "artifact_output_uri_required": False,
            },
            "runtime_source": {
                "overlay_required": source_overlay,
                "commit": admission.get("runtime_source_commit"),
                "archive_sha256": runtime_source.get("archive_sha256"),
                "archive_url_is_exact_commit_codeload": source_overlay,
                "image_source_commit": admission.get("source_commit"),
            },
            "gpu": {
                "preferred_gpu_class": gpu_type_id,
                "preferred_gpu_type_id": gpu_type_id,
                "provider_gpu_priority": [gpu_type_id],
                "gpu_count": 1,
                "container_disk_in_gb": int(preflight["container_disk_bytes"]) // 1024**3,
                # OpenPI stores the four frozen checkpoints under /workspace.
                # RunPod mounts this requested ephemeral volume at that path;
                # 80 GiB safely exceeds the 47.3 GB checkpoint cohort.  Vast
                # uses the admitted 100 GiB container disk and no volume.
                "volume_in_gb": 80 if provider_name == "runpod" else 0,
                "min_vcpu_count": 4,
                "min_memory_in_gb": 24,
            },
            "limits": {
                "max_active_workers": 1,
                "requested_budget_usd": float(spend["max_spend_usd"]),
                "hard_timeout_seconds": max(60, ttl - 120),
                "idle_timeout_seconds": 60,
                "startup_artifact_watchdog_required": True,
                "startup_artifact_timeout_seconds": max(60, ttl - 180),
                "idle_shutdown_required": True,
                "external_watchdog_ttl_required": True,
                "external_watchdog_ttl_seconds": ttl,
                "external_watchdog_owner": (f"independent_name_bound_{provider_name}_watchdog"),
                "scale_to_zero_default": True,
            },
            "artifact_finalizer": {
                "upload_before_shutdown_required": True,
                "record_actual_gpu_time_required": True,
            },
            "local_sim_only_prerequisite": {
                "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
                "required_before_provider_spend": True,
                "status": "passed",
                "source_artifact": "interiorgs_0787_dynamic_hybrid_control.json",
                "local_sim_only_evidence_clean": True,
                "sim_only_beta_core_complete": True,
                "sim_only_beta_blocked_requirement_ids": [],
                "blockers": [],
                "claim_boundary": {
                    "controls_are_not_learned_policy_results": True,
                    "local_sim_only_clean_does_not_prove_remote_execution": True,
                },
            },
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted",
        "blockers": [],
        "admission": admission,
        "bound_request": request,
        "provider_mutations_performed": 0,
    }


def shape_openpi_policy_ranking_request_without_mutation(
    *,
    prepared: Mapping[str, Any],
    output_path: str | Path,
    input_secret_url: str,
    output_secret_put_url: str,
    pod_name: str,
) -> dict[str, Any]:
    """Validate the exact redacted provider payload without making an API call."""

    if prepared.get("status") != "admitted" or not isinstance(
        prepared.get("bound_request"), Mapping
    ):
        return {
            "status": "blocked",
            "blockers": ["openpi_runpod_request_not_admitted"],
            "provider_mutations_performed": 0,
        }
    output = Path(output_path).expanduser().resolve()
    bound_request_path = output.parent / "openpi_provider_launch_request.json"
    write_json(bound_request_path, dict(prepared["bound_request"]))
    names = (INPUT_SECRET_URL_ENV, OUTPUT_SECRET_PUT_URL_ENV)
    previous = {
        key: os.environ.get(key)
        for key in (*names, GENERIC_OUTPUT_SECRET_URL_ENV, FORWARD_SECRET_ENV_NAMES_ENV)
    }
    os.environ[INPUT_SECRET_URL_ENV] = input_secret_url
    os.environ[OUTPUT_SECRET_PUT_URL_ENV] = output_secret_put_url
    # The generic adapter contract requires an output sink whenever the worker
    # owns finalization. It is never persisted and the worker uses the OpenPI-
    # specific alias above.
    os.environ[GENERIC_OUTPUT_SECRET_URL_ENV] = output_secret_put_url
    os.environ[FORWARD_SECRET_ENV_NAMES_ENV] = ",".join(names)
    try:
        return run_runpod_provider_adapter(
            provider_launch_request_path=bound_request_path,
            output_path=output,
            mode="dry-run",
            pod_name=pod_name,
            gpu_type_id=str(prepared["admission"]["gpu_type_id"]),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_openpi_policy_ranking_campaign(
    *,
    release_evidence: str | Path,
    input_bundle_receipt: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    input_secret_url_file: str | Path,
    output_secret_put_url_file: str | Path,
    pod_name: str,
    expected_source_commit: str,
    execute: bool,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    campaign_budget_ledger: str | Path | None = None,
    campaign_initial_spent_usd: float | None = None,
    campaign_initial_used_gpu_seconds: int | None = None,
    campaign_total_spend_cap_usd: float = 20.0,
    campaign_wall_cap_seconds: int = 36_000,
    output_secret_get_url_file: str | Path | None = None,
    provider_name: str = "vast",
    compute_authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate or launch one frozen OpenPI ranking campaign.

    The execute path refreshes provider facts, reserves worst-case campaign
    budget, confirms an independent watchdog, opens a pending-teardown record,
    and only then reaches the selected provider mutation adapter.
    """

    root = Path(adapter_output).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    if not pod_name.startswith(CANARY_NAME_PREFIX):
        result = {
            "status": "blocked",
            "blockers": ["openpi_runpod_pod_name_outside_watchdog_scope"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    release = _read_object(release_evidence)
    input_bundle = _read_object(input_bundle_receipt)
    preflight = _read_object(preflight_bundle)
    current_reference_authorization: dict[str, Any] = {}
    if input_bundle.get("schema_version") == CURRENT_REFERENCE_INPUT_RECEIPT_SCHEMA_VERSION:
        if compute_authorization_path is None:
            authorization_blockers = ["openpi_current_reference_authorization_missing"]
        else:
            try:
                current_reference_authorization = _read_object(compute_authorization_path)
            except (OSError, ValueError, json.JSONDecodeError):
                authorization_blockers = ["openpi_current_reference_authorization_unreadable"]
            else:
                authorization_blockers = _current_reference_authorization_blockers(
                    current_reference_authorization,
                    input_bundle=input_bundle,
                    expected_source_commit=expected_source_commit,
                )
        if authorization_blockers:
            result = {
                "status": "blocked",
                "blockers": authorization_blockers,
                "provider_mutations_performed": 0,
            }
            write_json(Path(admission_out), result)
            write_json(Path(adapter_output), result)
            return result
    resolved_provider = str(provider_name or "vast").strip().lower()
    if resolved_provider not in {"vast", "runpod"}:
        result = {
            "status": "blocked",
            "blockers": ["openpi_gpu_provider_unsupported"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    provider = get_render_provider(resolved_provider) if execute else None
    if execute and provider is not None:
        maximum_concurrent_gpus = int(
            current_reference_authorization.get("maximum_concurrent_gpus") or 1
        )
        max_existing_live_resources = maximum_concurrent_gpus - 1
        preflight = (
            collect_openpi_policy_ranking_vast_preflight(
                name_prefix=CANARY_NAME_PREFIX,
                container_disk_bytes=int(preflight.get("container_disk_bytes") or 0),
                capacity_probe=provider.capacity_preflight,
                inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
                max_existing_live_resources=max_existing_live_resources,
            )
            if resolved_provider == "vast"
            else collect_openpi_policy_ranking_runpod_preflight(
                name_prefix=CANARY_NAME_PREFIX,
                gpu_type_ids=list(preflight.get("requested_gpu_types") or []),
                container_disk_bytes=int(preflight.get("container_disk_bytes") or 0),
                capacity_probe=provider.capacity_preflight,
                inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
                max_existing_live_resources=max_existing_live_resources,
            )
        )
        write_json(root / "openpi_provider_preflight_launch_refresh.json", preflight)
    if str(preflight.get("provider") or "") != resolved_provider:
        result = {
            "status": "blocked",
            "blockers": ["openpi_gpu_preflight_provider_mismatch"],
            "expected_provider": resolved_provider,
            "observed_provider": preflight.get("provider"),
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result

    price = float(preflight.get("on_demand_price_usd_per_hour") or 0.0)
    spend = {
        "paid_mutation_authorized": True,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "hard_ttl_seconds": int(hard_ttl_seconds),
        "max_spend_usd": float(max_spend_usd),
        "physical_robot_endpoint_access_allowed": False,
    }
    prepared = build_openpi_policy_ranking_provider_request(
        release=release,
        input_bundle=input_bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=expected_source_commit,
        job_id=pod_name,
    )
    write_json(Path(admission_out), prepared)
    if prepared.get("status") != "admitted":
        return prepared
    write_json(Path(bound_request_out), dict(prepared["bound_request"]))
    if not execute:
        result = {
            **prepared,
            "status": "dry_run_ready",
            "provider_mutations_performed": 0,
            "watchdog_process_started": False,
            "budget_reservation_created": False,
        }
        write_json(Path(adapter_output), result)
        return result
    if (
        campaign_budget_ledger is None
        or campaign_initial_spent_usd is None
        or campaign_initial_used_gpu_seconds is None
        or output_secret_get_url_file is None
    ):
        result = {
            **prepared,
            "status": "blocked",
            "blockers": ["openpi_runpod_campaign_budget_arguments_missing"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result

    input_secret_url = _read_private_https_url(
        input_secret_url_file, field="openpi_input_secret_url_file"
    )
    output_secret_put_url = _read_private_https_url(
        output_secret_put_url_file, field="openpi_output_secret_put_url_file"
    )
    output_secret_get_url = _read_private_https_url(
        output_secret_get_url_file, field="openpi_output_secret_get_url_file"
    )
    budget = ProductionGpuCampaignBudget(
        campaign_budget_ledger,
        initial_spent_usd=campaign_initial_spent_usd,
        initial_used_gpu_seconds=campaign_initial_used_gpu_seconds,
        total_spend_cap_usd=campaign_total_spend_cap_usd,
        combined_gpu_wall_cap_seconds=campaign_wall_cap_seconds,
    )
    try:
        reservation = budget.reserve(
            reservation_id=pod_name,
            gpu_seconds=hard_ttl_seconds,
            max_hourly_rate_usd=price,
        )
    except CampaignBudgetExceeded as exc:
        result = {
            **prepared,
            "status": "blocked",
            "blockers": [str(exc)],
            "provider_mutations_performed": 0,
            "campaign_budget_admission": exc.admission,
        }
        write_json(Path(adapter_output), result)
        return result
    reserved_at_epoch = time.time()
    budget_context = {
        "status": "reserved",
        "ledger_path": str(Path(campaign_budget_ledger).expanduser().resolve()),
        "reservation_id": pod_name,
        "reserved_at_epoch": reserved_at_epoch,
        "reservation": reservation,
        "identity": {
            "initial_spent_usd": campaign_initial_spent_usd,
            "initial_used_gpu_seconds": campaign_initial_used_gpu_seconds,
            "total_spend_cap_usd": campaign_total_spend_cap_usd,
            "combined_gpu_wall_cap_seconds": campaign_wall_cap_seconds,
        },
    }
    write_json(root / "openpi_campaign_budget_reservation.json", budget_context)

    deadline = time.time() + hard_ttl_seconds
    watchdog = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--out-dir",
            str(root),
            "--pod-name-prefix",
            CANARY_NAME_PREFIX,
            "--deadline-epoch",
            str(deadline),
            "--provider",
            resolved_provider,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        armed = _wait_for_watchdog(
            root=root,
            process=watchdog,
            prefix=CANARY_NAME_PREFIX,
            deadline=deadline,
        )
    except Exception:
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="watchdog_not_armed_no_mutation",
        )
        raise
    write_json(root / "openpi_watchdog_armed_receipt.json", armed)

    inventory = provider.billable_inventory(name_prefix=CANARY_NAME_PREFIX)
    reconciliation = build_paid_provider_lane_reconciliation(
        provider=resolved_provider,
        lane=PAID_LANE,
        provider_inventory=inventory,
        open_pending_teardowns=load_pending_teardowns(),
    )
    lease = acquire_paid_provider_lane_lease(
        provider=resolved_provider,
        lane=PAID_LANE,
        job_dir=str(root),
        ttl_seconds=hard_ttl_seconds + 600,
        reconciliation=reconciliation,
    )
    write_json(root / "openpi_paid_provider_lane_lease.json", lease)
    if lease.get("status") != "acquired":
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="paid_provider_lane_not_acquired_no_mutation",
        )
        result = {
            **prepared,
            "status": "blocked",
            "blockers": list(lease.get("blockers") or []),
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result

    try:
        pending = open_pending_teardown(
            provider=resolved_provider,
            lane=PAID_LANE,
            run_id=pod_name,
            resource_kind="compute_instance",
            resource_name=pod_name,
            job_dir=root,
            max_age_seconds=hard_ttl_seconds + 600,
        )
        grant = require_paid_resource_admission(
            prepared["admission"]["shared_paid_lane_admission"],
            resource_class=str(prepared["admission"]["provider_resource_class"]),
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except Exception:  # noqa: BLE001 - provider boundary has not been crossed
        release_paid_provider_lane_lease(
            lease,
            reason="local_pre_provider_failure_no_mutation",
            provider_mutation_started=False,
        )
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="local_pre_provider_failure_no_mutation",
        )
        raise
    receipt_path = root / "provider_lane_handoff_receipt.json"
    receipt = {
        "status": "pending_watchdog_transfer",
        "lease_path": lease["path"],
        "owner_pid": watchdog.pid,
        "provider_lane_release_mode": "watchdog_direct_compute",
        "pod_pending_teardown_record": pending["path"],
        "pod_id": None,
        "pod_name_prefix": CANARY_NAME_PREFIX,
        "campaign_kind": "openpi_policy_ranking",
        "campaign_budget": budget_context,
    }
    _write_private_json(receipt_path, receipt)
    handoff = transfer_paid_provider_compute_lane_lease_to_watchdog(
        lease,
        watchdog_pid=watchdog.pid,
        pending_teardown_record=pending["path"],
        watchdog_deadline_epoch=deadline,
        resource_name_prefix=CANARY_NAME_PREFIX,
    )
    write_json(root / "openpi_paid_provider_lane_handoff.json", handoff)
    if handoff.get("status") != "accepted":
        cancel_pending_teardown(
            pending["path"],
            reason="compute_lane_handoff_failed_no_mutation",
            evidence={"provider_mutations_performed": 0},
        )
        release_paid_provider_lane_lease(
            lease,
            reason="compute_lane_handoff_failed_no_mutation",
            provider_mutation_started=False,
        )
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="compute_lane_handoff_failed_no_mutation",
        )
        receipt_path.unlink(missing_ok=True)
        result = {
            **prepared,
            "status": "blocked",
            "blockers": list(handoff.get("blockers") or []),
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    receipt = {
        **handoff,
        "provider_lane_release_mode": "watchdog_direct_compute",
        "pod_pending_teardown_record": pending["path"],
        "pod_id": None,
        "pod_name_prefix": CANARY_NAME_PREFIX,
        "campaign_kind": "openpi_policy_ranking",
        "campaign_budget": budget_context,
    }
    _write_private_json(receipt_path, receipt)
    authorization_consumption: dict[str, Any] = {"status": "not_required"}
    if current_reference_authorization:
        authorization_consumption = _consume_current_reference_authorization_once(
            current_reference_authorization,
            expected_source_commit=expected_source_commit,
            input_bundle=input_bundle,
        )
        if authorization_consumption.get("status") != "consumed":
            cancel_pending_teardown(
                pending["path"],
                reason="compute_authorization_not_consumed_no_mutation",
                evidence={"provider_mutations_performed": 0},
            )
            release_paid_provider_lane_lease(
                lease,
                reason="compute_authorization_not_consumed_no_mutation",
                provider_mutation_started=False,
            )
            _stop_watchdog_after_provider_zero(watchdog)
            budget.settle(
                reservation_id=pod_name,
                charged_gpu_seconds=0,
                charged_usd=0.0,
                outcome="compute_authorization_not_consumed_no_mutation",
            )
            receipt_path.unlink(missing_ok=True)
            result = {
                **prepared,
                "status": "blocked",
                "blockers": list(authorization_consumption.get("blockers") or []),
                "authorization_consumption": authorization_consumption,
                "provider_mutations_performed": 0,
            }
            write_json(Path(adapter_output), result)
            return result
        prepared["authorization_consumption"] = authorization_consumption
        write_json(Path(admission_out), prepared)
        bound_request = dict(prepared["bound_request"])
        bound_request["authorization_consumption"] = authorization_consumption
        write_json(Path(bound_request_out), bound_request)
    previous = {
        key: os.environ.get(key)
        for key in (
            INPUT_SECRET_URL_ENV,
            OUTPUT_SECRET_PUT_URL_ENV,
            GENERIC_OUTPUT_SECRET_URL_ENV,
            FORWARD_SECRET_ENV_NAMES_ENV,
        )
    }
    os.environ[INPUT_SECRET_URL_ENV] = input_secret_url
    os.environ[OUTPUT_SECRET_PUT_URL_ENV] = output_secret_put_url
    os.environ[GENERIC_OUTPUT_SECRET_URL_ENV] = output_secret_put_url
    os.environ[FORWARD_SECRET_ENV_NAMES_ENV] = ",".join(
        (INPUT_SECRET_URL_ENV, OUTPUT_SECRET_PUT_URL_ENV)
    )
    try:
        if resolved_provider == "vast":
            vast_request = _build_vast_launch_request(
                provider=provider,
                root=root,
                pod_name=pod_name,
                release=release,
                input_bundle=input_bundle,
                preflight=preflight,
                input_secret_url=input_secret_url,
                output_secret_put_url=output_secret_put_url,
            )
            launch = provider.launch(
                root,
                vast_request,
                cold=True,
                paid_resource_admission_grant=grant,
            )
            adapter = {
                **dict(launch),
                "status": ("submitted" if launch.get("status") == "launched" else "blocked"),
                "provider": "vast",
                "raw_secret_values_recorded": False,
            }
            adapter.pop("error", None)
            write_json(Path(adapter_output), adapter)
        else:
            adapter = run_runpod_provider_adapter(
                provider_launch_request_path=bound_request_out,
                output_path=adapter_output,
                mode="on-demand-pod",
                allow_runpod_api_call=True,
                pod_name=pod_name,
                gpu_type_id=str(preflight["gpu_type_id"]),
                paid_resource_admission_grant=grant,
            )
    except Exception as exc:
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="openpi_runpod_adapter_raised_after_create_boundary",
            evidence={"error_type": type(exc).__name__},
        )
        cleanup = terminate_canary_resources(
            provider=provider,
            pod_name_prefix=CANARY_NAME_PREFIX,
            armed=armed,
            provider_name=resolved_provider,
        )
        cleanup_handoff = _handoff_cleanup_to_watchdog(
            root=root,
            provider=provider,
            cleanup=cleanup,
            instance_id=pod_name,
            provider_name=resolved_provider,
        )
        result = {
            "status": "failed",
            "blockers": ["openpi_runpod_adapter_failed_or_ambiguous"],
            "error_type": type(exc).__name__,
            "immediate_cleanup": cleanup,
            "cleanup_handoff": cleanup_handoff,
            "continuing_spend": cleanup_handoff["continuing_spend"],
            "raw_secret_values_recorded": False,
            "authorization_consumption": authorization_consumption,
        }
        write_json(Path(adapter_output), result)
        return result
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if resolved_provider == "vast":
        pod_id = str(adapter.get("instance_id") or "").strip()
    else:
        response = adapter.get("runpod_response")
        response = response if isinstance(response, Mapping) else {}
        pod_id = str(response.get("id") or "").strip()
    if adapter.get("status") == "submitted" and pod_id:
        bind_pending_teardown_instance(pending["path"], pod_id)
        receipt["pod_id"] = pod_id
        _write_private_json(receipt_path, receipt)
        monitor = _monitor_openpi_output_and_teardown(
            root=root,
            output_secret_get_url=output_secret_get_url,
            provider=provider,
            armed=armed,
            pod_id=pod_id,
            provider_name=resolved_provider,
            deadline_epoch=deadline,
        )
        result = {
            **adapter,
            "status": monitor["status"],
            "watchdog_pid": watchdog.pid,
            "watchdog_deadline_epoch": deadline,
            "campaign_budget_reservation": reservation,
            "pending_teardown_record": pending["path"],
            "monitor": monitor,
            "continuing_spend": monitor.get("continuing_spend") is True,
            "raw_secret_values_recorded": False,
            "authorization_consumption": authorization_consumption,
        }
        write_json(Path(adapter_output), result)
        return result

    if (
        adapter.get("runpod_side_effects_may_have_occurred") is True
        or adapter.get("allocation_outcome_ambiguous") is True
    ):
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="openpi_runpod_create_result_missing_pod_id",
            evidence={"adapter_status": adapter.get("status")},
        )
    cleanup = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=CANARY_NAME_PREFIX,
        armed=armed,
        provider_name=resolved_provider,
    )
    cleanup_handoff = _handoff_cleanup_to_watchdog(
        root=root,
        provider=provider,
        cleanup=cleanup,
        instance_id=pod_name,
        provider_name=resolved_provider,
    )
    result = {
        **adapter,
        "status": "failed",
        "blockers": sorted(set([*(adapter.get("blockers") or []), "openpi_runpod_pod_id_missing"])),
        "immediate_cleanup": cleanup,
        "cleanup_handoff": cleanup_handoff,
        "continuing_spend": cleanup_handoff["continuing_spend"],
        "authorization_consumption": authorization_consumption,
    }
    write_json(Path(adapter_output), result)
    return result


__all__ = [
    "build_openpi_policy_ranking_provider_request",
    "run_openpi_policy_ranking_campaign",
    "shape_openpi_policy_ranking_request_without_mutation",
]
