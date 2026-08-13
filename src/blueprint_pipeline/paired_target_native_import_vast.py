"""One-shot paid native Isaac import for a bound 1--5 replacement bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from typing import Any

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE
from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import (
    active_instance_allowlist_metadata_error,
    flatten_active_instance_allowlist,
    normalize_active_instance_allowlist,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .paired_target_native_import_bundle import (
    RESULT_FILENAME,
    RESULT_SCHEMA_VERSION as RUNTIME_RESULT_SCHEMA_VERSION,
    validate_paired_target_native_import_bundle,
)
from .public_scene_artifixer3d_vast import validate_artifixer3d_terminal_spend_chain
from .spend_authority_consumption_root import consumption_root
from .task_evaluation_artifact_manifest import seal_lane_terminal_artifacts
from .vast_independent_watchdog_control import (
    EVIDENCE_NAME as WATCHDOG_EVIDENCE_NAME,
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-paired-target-native-import"
PROVIDER_BUNDLE_KIND = "paired_target_native_import"
RESULT_SCHEMA_VERSION = "paired_target_native_import_vast_run.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "paired_target_native_import_paid_attempt_authority.v1"
)
POST_ATTEMPT_PROVIDER_ZERO_SCHEMA_VERSION = (
    "paired_target_native_import_provider_zero.v1"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/paired-target-native-import"
INSTANCE_LABEL_PREFIX = "blueprint-adp-paired-native-import-"
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 7_200
MAX_HARD_CAP_USD = 2.0
AGGREGATE_GOAL_SPEND_CAP_USD = 12.0
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound_record(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def materialize_paired_target_native_import_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    prior_artifixer_authority_path: str | Path,
    prior_artifixer_result_path: str | Path,
    prior_artifixer_cleanup_path: str | Path,
    prior_artifixer_provider_zero_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Seal one new authority chained to the exact prior scene spend and zero receipt."""

    bundle = validate_paired_target_native_import_bundle(bundle_receipt_path)
    terminal = validate_artifixer3d_terminal_spend_chain(
        authority_path=prior_artifixer_authority_path,
        result_path=prior_artifixer_result_path,
        cleanup_path=prior_artifixer_cleanup_path,
        provider_zero_path=prior_artifixer_provider_zero_path,
    )
    prior_spend = float(terminal["aggregate_goal_spend_after_attempt_usd"])
    aggregate_cap = min(
        AGGREGATE_GOAL_SPEND_CAP_USD,
        float(terminal["aggregate_goal_spend_cap_usd"]),
    )
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or blueprint_commit != bundle.get("implementation_commit")
        or DEFAULT_IMAGE != bundle.get("container_image")
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
        or prior_spend + hard_cap_usd > aggregate_cap
        or any(value <= 0 for value in allowed)
    ):
        raise ValueError("paired_target_native_import_authority_configuration_invalid")
    receipt = Path(str(bundle["receipt_path"])).resolve()
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_paired_target_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt": _record(receipt),
        "bundle_receipt_digest": bundle["receipt_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "probe_spec_sha256": bundle["probe_spec_sha256"],
        "source_request_digest": bundle["source_request_digest"],
        "replacement_count": bundle["replacement_count"],
        "blueprint_commit": blueprint_commit,
        "container_image": DEFAULT_IMAGE,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": prior_spend,
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "prior_terminal_artifixer": {
            **terminal["records"],
            "authority_digest": terminal["authority_digest"],
            "attempt_cost_usd": terminal["attempt_cost_usd"],
            "lineage_cost_usd": terminal["lineage_cost_usd"],
        },
        "active_instance_allowlist": {
            "external_provider_owned": list(allowed),
            "same_goal_concurrent": [],
        },
        "native_simulator_import_probe_only": True,
        "candidate_policy_queried": False,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "physical_success_established": False,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_native_import_authority_output_exists")
    ensure_dir(output.parent)
    write_json(output, authority)
    validate_paired_target_native_import_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed,
    )
    return authority


def validate_paired_target_native_import_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    value = dict(authority)
    allowlist = normalize_active_instance_allowlist(value.get("active_instance_allowlist"))
    expected = normalize_active_instance_allowlist(list(allowed_active_instance_ids))
    errors: list[str] = []
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION:
        errors.append("schema_invalid")
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("digest_invalid")
    expected_fields = {
        "purpose": "one_shot_paired_target_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt_digest": prepared_bundle.get("receipt_digest"),
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "probe_spec_sha256": prepared_bundle.get("probe_spec_sha256"),
        "source_request_digest": prepared_bundle.get("source_request_digest"),
        "replacement_count": prepared_bundle.get("replacement_count"),
        "blueprint_commit": prepared_bundle.get("implementation_commit"),
        "container_image": DEFAULT_IMAGE,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "native_simulator_import_probe_only": True,
        "candidate_policy_queried": False,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "physical_success_established": False,
    }
    errors.extend(f"{key}_mismatch" for key, expected_value in expected_fields.items() if value.get(key) != expected_value)
    if allowlist is None or expected is None or flatten_active_instance_allowlist(
        allowlist or {"external_provider_owned": (), "same_goal_concurrent": ()}
    ) != flatten_active_instance_allowlist(
        expected or {"external_provider_owned": (), "same_goal_concurrent": ()}
    ):
        errors.append("active_instance_allowlist_mismatch")
    elif active_instance_allowlist_metadata_error(value, allowlist=allowlist) is not None:
        errors.append("active_instance_allowlist_metadata_invalid")
    if (
        not isinstance(value.get("aggregate_goal_spend_before_attempt_usd"), (int, float))
        or isinstance(value.get("aggregate_goal_spend_before_attempt_usd"), bool)
        or value.get("aggregate_goal_spend_before_attempt_usd", 0) + hard_cap_usd
        > value.get("aggregate_goal_spend_cap_usd", 0)
    ):
        errors.append("aggregate_spend_invalid")
    try:
        receipt_path, _ = _bound_record(
            value.get("bundle_receipt"), "paired_target_native_import_bundle_unbound"
        )
        if receipt_path != Path(str(prepared_bundle.get("receipt_path") or "")).resolve():
            errors.append("bundle_receipt_path_mismatch")
        predecessor = value.get("prior_terminal_artifixer")
        if not isinstance(predecessor, Mapping):
            raise ValueError("predecessor_invalid")
        paths = {
            key: _bound_record(predecessor.get(key), "predecessor_unbound")[0]
            for key in (
                "authority",
                "terminal_result",
                "object_store_cleanup",
                "provider_zero",
            )
        }
        terminal = validate_artifixer3d_terminal_spend_chain(
            authority_path=paths["authority"],
            result_path=paths["terminal_result"],
            cleanup_path=paths["object_store_cleanup"],
            provider_zero_path=paths["provider_zero"],
        )
        if (
            predecessor.get("authority_digest") != terminal["authority_digest"]
            or predecessor.get("attempt_cost_usd") != terminal["attempt_cost_usd"]
            or predecessor.get("lineage_cost_usd") != terminal["lineage_cost_usd"]
            or value.get("aggregate_goal_spend_before_attempt_usd")
            != terminal["aggregate_goal_spend_after_attempt_usd"]
        ):
            errors.append("prior_terminal_spend_mismatch")
    except ValueError:
        errors.append("prior_terminal_spend_invalid")
    if errors:
        raise ValueError(
            "paired_target_native_import_authority_invalid:" + ",".join(sorted(set(errors)))
        )
    return value


def consume_paired_target_native_import_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["paired_target_native_import_authority_identity_invalid"]}
    root = consumption_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = root.stat()
        if root.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise OSError("insecure_root")
        destination = root / f"paired-target-native-import-{digest[7:]}.json"
        payload = {
            "schema_version": "paired_target_native_import_authority_consumption.v1",
            "authorization_digest": digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "blueprint_commit": blueprint_commit,
            "consumed_at": utc_now_iso(),
            "maximum_provider_allocations": 1,
        }
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = root / f".paired-target-native-import-{digest[7:]}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["paired_target_native_import_authority_consumed"]}
    except OSError:
        return {"status": "blocked", "blockers": ["paired_target_native_import_authority_consumption_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


@contextmanager
def _mutation_authority():
    previous = {name: os.environ.get(name) for name in _MUTATION_ENV}
    try:
        for name in _MUTATION_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract_result(source: Path, destination: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError("path_traversal")
            archive.extractall(destination)
    except (OSError, ValueError, zipfile.BadZipFile):
        blockers.append("paired_target_native_import_output_zip_invalid")
    result_path = destination / RESULT_FILENAME
    try:
        execution = _read(result_path, "paired_target_native_import_runtime_result_missing")
    except ValueError:
        execution = {}
        blockers.append("paired_target_native_import_runtime_result_missing")
    if execution and (
        execution.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or execution.get("result_digest")
        != canonical_digest(execution, digest_field="result_digest")
    ):
        blockers.append("paired_target_native_import_runtime_result_invalid")
    return execution, blockers


def run_paired_target_native_import_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    paid_attempt_authority: Mapping[str, Any] | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 1.0,
    hard_ttl_seconds: int = 3_600,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    result_path = job / "paired_target_native_import_vast_result.v1.json"
    receipt_value = prepared_bundle.get("receipt_path")
    if not receipt_value:
        raise ValueError("paired_target_native_import_prepared_bundle_receipt_missing")
    bundle = validate_paired_target_native_import_bundle(receipt_value)
    if bundle.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        raise ValueError("paired_target_native_import_prepared_bundle_mismatch")
    if paid_attempt_authority is not None:
        authority = validate_paired_target_native_import_paid_attempt_authority(
            paid_attempt_authority,
            prepared_bundle=bundle,
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
            allowed_active_instance_ids=allowed_active_instance_ids,
        )
    else:
        authority = None
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle_sha256": bundle["bundle_sha256"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result
    if paid_resource_admission_grant is None or authority is None:
        raise ValueError("paired_target_native_import_paid_execution_authority_missing")
    consumption = consume_paired_target_native_import_authority_once(
        authority, blueprint_commit=authority["blueprint_commit"]
    )
    if consumption.get("status") != "consumed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
        write_json(result_path, result)
        return result
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=Path(str(bundle["bundle_path"])),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=hard_ttl_seconds + 1_800,
    )
    if staging.get("status") != "completed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "blockers": staging.get("blockers") or ["paired_target_native_import_staging_blocked"],
        }
        write_json(result_path, result)
        return result
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    handoff, handle = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=hard_ttl_seconds // 60,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed,
        pod_name_prefix=INSTANCE_LABEL_PREFIX,
    )
    if handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": handoff,
            "blockers": ["paired_target_native_import_watchdog_not_armed"],
        }
        write_json(result_path, result)
        return result
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any]
    try:
        with _mutation_authority():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=hard_ttl_seconds // 60,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=DEFAULT_IMAGE,
                isaac_image=DEFAULT_IMAGE,
                ngc_image_login_mode="always",
                provider_bundle=bundle["bundle_path"],
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=True,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=True,
                min_cold_isaac_pull_live_minutes=30,
                disk_gb=120,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=1_800,
                session_budget_ledger_path=job / "paired_target_native_import_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=True,
                preferred_gpu_keywords=("L40S", "RTX 4090", "RTX A6000"),
                prefer_isaac_rt=True,
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed,
                vast_launch_lock_file=job.parent / "paired_target_native_import_paid_launch.lock",
                instance_label_prefix=INSTANCE_LABEL_PREFIX,
                started_instance_id_path=handle.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [
                f"paired_target_native_import_adapter_failed:{redacted_failure_detail(exc)}"
            ],
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path, "paired_target_native_import_teardown_missing") if teardown_path.is_file() else {}
    instance_ids = [value for value in teardown.get("vast_instance_ids") or [] if isinstance(value, int) and value > 0]
    watchdog = close_independent_vast_watchdog(
        job_dir=job,
        handle=handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=not instance_ids and adapter.get("provider_create_attempted") is not True,
    )
    execution, blockers = _extract_result(output_zip, job / "immutable_execution")
    if adapter.get("status") != "completed":
        blockers.append("paired_target_native_import_provider_adapter_not_completed")
    if (
        execution.get("status") != "completed"
        or execution.get("native_isaac_executed") is not True
        or execution.get("all_replacements_import_qualified") is not True
        or execution.get("replacement_count") != bundle.get("replacement_count")
        or execution.get("request_digest") != bundle.get("request_digest")
        or execution.get("candidate_policy_queried") is not False
        or execution.get("physical_equivalence_claimed") is not False
    ):
        blockers.append("paired_target_native_import_runtime_not_qualified")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("paired_target_native_import_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        blockers.append("paired_target_native_import_watchdog_not_terminal")
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    final = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "request_digest": bundle["request_digest"],
        "replacement_count": bundle["replacement_count"],
        "native_result_path": str(job / "immutable_execution" / RESULT_FILENAME),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(teardown_path),
        "watchdog_receipt_path": str(watchdog_path),
        "object_store_cleanup_path": str(staging_dir / "wam_provider_object_store_cleanup.json"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_mutations_performed": 1 if adapter.get("provider_create_attempted") is True else 0,
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "authorization_consumption": consumption,
        "independent_watchdog": watchdog,
        "candidate_policy_queried": False,
        "physical_success_established": False,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    final = seal_lane_terminal_artifacts(final, attempt_root=job, lane="paired_target_native_import")
    write_json(result_path, final)
    return final


def materialize_paired_target_native_import_provider_zero(
    *,
    attempt_authority_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    authority_path = Path(attempt_authority_path).expanduser().resolve()
    terminal_path = Path(result_path).expanduser().resolve()
    authority = _read(authority_path, "paired_target_native_import_authority_unreadable")
    result = _read(terminal_path, "paired_target_native_import_result_unreadable")
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = Path(str(result.get("object_store_cleanup_path") or "")).resolve()
    adapter_path = Path(str(result.get("adapter_result_path") or "")).resolve()
    watchdog = _read(watchdog_path, "paired_target_native_import_watchdog_unreadable")
    cleanup = _read(cleanup_path, "paired_target_native_import_cleanup_unreadable")
    adapter = _read(adapter_path, "paired_target_native_import_adapter_unreadable")
    inventory = watchdog.get("final_global_inventory")
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") not in {"completed", "blocked"}
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
        or adapter.get("continuing_spend_from_this_run") is not False
    ):
        raise ValueError("paired_target_native_import_provider_zero_invalid")
    receipt = {
        "schema_version": POST_ATTEMPT_PROVIDER_ZERO_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed",
        "attempt_authority": _record(authority_path),
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(terminal_path),
        "provider_adapter": _record(adapter_path),
        "watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_zero_confirmed": True,
        "inventory": inventory,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_native_import_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


__all__ = [
    "PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "consume_paired_target_native_import_authority_once",
    "materialize_paired_target_native_import_paid_attempt_authority",
    "materialize_paired_target_native_import_provider_zero",
    "run_paired_target_native_import_vast",
    "validate_paired_target_native_import_paid_attempt_authority",
]
