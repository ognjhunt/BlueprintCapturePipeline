"""Canonical zero-retry Vast execution for the ADP-009B exact SimReady probe."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import zipfile

from .task_evaluation_artifact_manifest import (
    seal_lane_terminal_artifacts,
    seal_unallocated_provider_teardown,
)
from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import (
    active_instance_allowlist_metadata_error,
    flatten_active_instance_allowlist,
    normalize_active_instance_allowlist,
    validate_bound_lane_prior_spend,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .public_scene_simready_isaac_bundle import (
    DEFAULT_IMAGE,
    RIGID_PROBE_NAMES,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_independent_watchdog_control import (
    WATCHDOG_DIR_NAME,
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


from .spend_authority_consumption_root import consumption_root

PROBE_KIND = "adp009b-exact-simready-isaac"
RESULT_SCHEMA_VERSION = "adp009b_simready_isaac_vast_run.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA = "adp_simready_isaac_paid_attempt_authority.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/exact-simready-isaac"
INSTANCE_LABEL_PREFIX = "blueprint-adp009b-simready-"
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _next_attempt(job: Path) -> tuple[int, Path]:
    attempts = job / "attempts"
    observed = 0
    if attempts.is_dir():
        for path in attempts.glob("attempt_*"):
            try:
                observed = max(observed, int(path.name.removeprefix("attempt_")))
            except ValueError:
                continue
    number = observed + 1
    return number, attempts / f"attempt_{number:03d}"


def _remaining_minutes(
    *, ledger_path: Path, cap_usd: float, ttl_seconds: int, hourly_rate: float
) -> int:
    ledger = _read_json(ledger_path)
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    time_minutes = math.floor(max(0.0, ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(max(0.0, cap_usd - prior_cost) * 60.0 / hourly_rate)
    return max(0, min(time_minutes, spend_minutes))


@contextmanager
def _mutation_authority():
    prior = {name: os.environ.get(name) for name in _MUTATION_ENV}
    try:
        for name in _MUTATION_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract_result(source: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not source.is_file():
        return {"status": "blocked", "execution": {}, "blockers": ["simready_isaac_output_zip_missing"]}
    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("simready_isaac_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("simready_isaac_output_zip_invalid")
    result_path = destination / "isaac_runtime_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("simready_isaac_runtime_result_missing")
    elif not _runtime_result_digest_valid(execution):
        blockers.append("simready_isaac_runtime_result_digest_invalid")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }




def _runtime_result_digest_valid(execution: Mapping[str, Any]) -> bool:
    """Verify the canonical result and one retained, verifiable legacy encoding.

    One articulated-controls worker wrote the same digest into both
    ``result_digest`` and ``_canonical_digest`` after calculating it over the
    payload that contained neither field.  The values remain independently
    verifiable, but the extra alias makes the normal single-field verifier
    reject them.  Accept exactly that shape so a retained paid result can be
    adjudicated without rewriting it or paying for a retry.  New workers emit
    only ``result_digest``.
    """

    observed = execution.get("result_digest")
    if not isinstance(observed, str):
        return False
    if canonical_digest(execution, digest_field="result_digest") == observed:
        return True
    legacy = execution.get("_canonical_digest")
    if legacy != observed:
        return False
    payload = dict(execution)
    payload.pop("result_digest", None)
    payload.pop("_canonical_digest", None)
    return canonical_digest(payload) == observed


def _probe_name(row: Mapping[str, Any]) -> str:
    """Read the current probe key, or the one retained legacy worker emitted."""

    return str(row.get("probe") or row.get("name") or "")


def _execution_blockers(
    execution: Mapping[str, Any],
    expected_probe_names: frozenset[str] = RIGID_PROBE_NAMES,
) -> list[str]:
    blockers: list[str] = []
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["simready_isaac_execution_not_completed"])
    if execution.get("native_isaac_executed") is not True:
        blockers.append("simready_isaac_native_execution_not_proven")
    if execution.get("physical_success_established") is not False:
        blockers.append("simready_isaac_physical_claim_boundary_invalid")
    if execution.get("source_target_collider_active") is not False:
        blockers.append("simready_isaac_source_collider_not_inactive")
    if execution.get("replacement_count") != 1:
        blockers.append("simready_isaac_replacement_count_invalid")
    probes = execution.get("probe_results")
    if not isinstance(probes, list) or {
        _probe_name(row) for row in probes if isinstance(row, Mapping)
    } != set(expected_probe_names):
        blockers.append("simready_isaac_probe_set_invalid")
    elif any(not isinstance(row, Mapping) or row.get("passed") is not True for row in probes):
        blockers.append("simready_isaac_probe_failure")
    return blockers


def adjudicate_retained_simready_isaac_execution(
    *,
    execution_path: str | Path,
    bundle_receipt_path: str | Path,
    destination: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Independently adjudicate an immutable result the outer lane misread.

    The source result is never changed.  This derives a new receipt that binds
    its byte digest, verifies its self-attestation, and applies the exact probe
    set frozen in the bundle receipt.  It is useful only for bookkeeping
    failures after scientific execution; a missing or failed probe remains a
    blocker and cannot be caller-overridden.
    """

    execution_file = Path(execution_path).expanduser().resolve()
    bundle_file = Path(bundle_receipt_path).expanduser().resolve()
    execution = _read_json(execution_file)
    bundle = _read_json(bundle_file)
    blockers: list[str] = []
    if not execution:
        blockers.append("simready_isaac_retained_execution_missing")
    elif not _runtime_result_digest_valid(execution):
        blockers.append("simready_isaac_runtime_result_digest_invalid")
    if not bundle:
        blockers.append("simready_isaac_bundle_receipt_missing")
    elif bundle.get("receipt_digest") != canonical_digest(
        bundle, digest_field="receipt_digest"
    ):
        blockers.append("simready_isaac_bundle_receipt_digest_invalid")
    probe_names = frozenset(str(name) for name in (bundle.get("probe_names") or []))
    if not probe_names:
        blockers.append("simready_isaac_expected_probe_set_missing")
    if execution and probe_names:
        blockers.extend(_execution_blockers(execution, probe_names))

    receipt: dict[str, Any] = {
        "schema_version": "simready_isaac_retained_execution_adjudication.v1",
        "generated_at": generated_at or utc_now_iso(),
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "source_execution": {
            "path": str(execution_file),
            "sha256": _sha256(execution_file) if execution_file.is_file() else None,
            "result_digest": execution.get("result_digest"),
            "retained_legacy_encoding": "_canonical_digest" in execution,
        },
        "source_bundle_receipt": {
            "path": str(bundle_file),
            "sha256": _sha256(bundle_file) if bundle_file.is_file() else None,
            "receipt_digest": bundle.get("receipt_digest"),
            "bundle_sha256": bundle.get("bundle_sha256"),
        },
        "expected_probe_names": sorted(probe_names),
        "observed_probe_results": list(execution.get("probe_results") or []),
        "native_isaac_executed": execution.get("native_isaac_executed") is True,
        "physical_success_established": False,
        "claim_boundary": {
            "derived_after_execution": True,
            "source_result_bytes_unchanged": True,
            "adjudication_is_not_a_paid_retry": True,
            "adjudication_is_not_robot_task_success": True,
            "simulator_execution_is_not_physical_truth": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    write_json(Path(destination), receipt)
    return receipt


def validate_simready_isaac_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    bundle_receipt_sha256: str | None,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Bind one explicit paid native-import probe attempt to exact bundle bytes."""

    value = dict(authority)
    errors: list[str] = []
    structured_allowlist = "active_instance_allowlist" in value
    predecessor = prepared_bundle.get("paired_native_predecessor") or {}
    authority_allowlist = normalize_active_instance_allowlist(
        value.get("active_instance_allowlist", value.get("external_instance_allowlist"))
    )
    expected_allowlist = normalize_active_instance_allowlist(allowed_active_instance_ids)
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA:
        errors.append("schema_invalid")
    if value.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("authority_kind_invalid")
    if not str(value.get("authority_reference") or "").strip():
        errors.append("authority_reference_invalid")
    if not str(value.get("authorized_by") or "").strip():
        errors.append("authorized_by_invalid")
    if not str(value.get("authorized_on") or "").strip():
        errors.append("authorized_on_invalid")
    if value.get("purpose") != "simready_native_import_probe":
        errors.append("purpose_invalid")
    if value.get("provider") != "vast":
        errors.append("provider_invalid")
    if value.get("paid_compute_authorized") is not True:
        errors.append("paid_compute_not_authorized")
    if value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        errors.append("bundle_sha256_mismatch")
    if value.get("bundle_receipt_sha256") != bundle_receipt_sha256:
        errors.append("bundle_receipt_sha256_mismatch")
    if value.get("probe_spec_sha256") != prepared_bundle.get("probe_spec_sha256"):
        errors.append("probe_spec_sha256_mismatch")
    if value.get("scene_id") != prepared_bundle.get("scene_id"):
        errors.append("scene_id_mismatch")
    if value.get("candidate_usd_sha256") != prepared_bundle.get(
        "candidate_usd_sha256"
    ):
        errors.append("candidate_usd_sha256_mismatch")
    if value.get("native_probe_manifest_sha256") != prepared_bundle.get(
        "native_probe_manifest_sha256"
    ):
        errors.append("native_probe_manifest_sha256_mismatch")
    if value.get("native_probe_manifest_digest") != prepared_bundle.get(
        "native_probe_manifest_digest"
    ):
        errors.append("native_probe_manifest_digest_mismatch")
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("binding_digest")
        != canonical_digest(predecessor, digest_field="binding_digest")
        or predecessor.get("scene_id") != prepared_bundle.get("scene_id")
        or predecessor.get("candidate_usd_sha256")
        != prepared_bundle.get("candidate_usd_sha256")
        or predecessor.get("binding_digest")
        != prepared_bundle.get("predecessor_binding_digest")
        or value.get("predecessor_binding_digest")
        != prepared_bundle.get("predecessor_binding_digest")
    ):
        errors.append("predecessor_binding_digest_mismatch")
    if value.get("container_image") != DEFAULT_IMAGE:
        errors.append("container_image_mismatch")
    if value.get("maximum_paid_attempts") != 1:
        errors.append("maximum_paid_attempts_invalid")
    if value.get("maximum_automatic_retries") != 0:
        errors.append("maximum_automatic_retries_invalid")
    if value.get("automatic_paid_retry_authorized") is not False:
        errors.append("automatic_paid_retry_authorized_invalid")
    if value.get("zero_retry") is not True:
        errors.append("zero_retry_invalid")
    if value.get("hard_attempt_spend_cap_usd") != hard_cap_usd:
        errors.append("hard_attempt_spend_cap_mismatch")
    if value.get("maximum_hourly_rate_usd") != max_hourly_rate_usd:
        errors.append("maximum_hourly_rate_mismatch")
    if value.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds:
        errors.append("maximum_single_resource_ttl_mismatch")
    if value.get("native_simulator_import_probe_only") is not True:
        errors.append("native_import_probe_scope_invalid")
    if value.get("physical_success_established") is not False:
        errors.append("physical_claim_boundary_invalid")
    if value.get("candidate_policy_queried") is not False:
        errors.append("candidate_policy_query_claim_invalid")
    if authority_allowlist is None:
        errors.append(
            "active_instance_allowlist_invalid"
            if structured_allowlist
            else "external_instance_allowlist_invalid"
        )
    elif expected_allowlist is None:
        errors.append("allowed_active_instance_ids_invalid")
    elif flatten_active_instance_allowlist(
        authority_allowlist
    ) != flatten_active_instance_allowlist(expected_allowlist):
        errors.append(
            "active_instance_allowlist_mismatch"
            if structured_allowlist
            else "external_instance_allowlist_mismatch"
        )
    elif (metadata_error := active_instance_allowlist_metadata_error(
        value, allowlist=authority_allowlist
    )) is not None:
        errors.append(metadata_error)
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("authorization_digest_invalid")
    try:
        validate_bound_lane_prior_spend(value, lane="simready_isaac")
    except ValueError:
        errors.append("prior_spend_reconciliation_invalid")
    if errors:
        raise ValueError(
            "simready_isaac_paid_attempt_authority_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return value


def consume_simready_isaac_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    """Atomically consume a validated native-import grant before provider mutation."""

    authorization_digest = str(authority.get("authorization_digest") or "")
    if not authorization_digest.startswith("sha256:") or len(authorization_digest) != 71:
        return {
            "status": "blocked",
            "blockers": ["simready_isaac_paid_attempt_authority_identity_invalid"],
        }
    root = consumption_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = root.stat()
        if (
            root.is_symlink()
            or root_stat.st_uid != os.getuid()
            or root_stat.st_mode & 0o077
        ):
            return {
                "status": "blocked",
                "blockers": ["simready_isaac_authority_consumption_root_insecure"],
            }
        identity = authorization_digest.removeprefix("sha256:")
        destination = root / f"simready-isaac-{identity}.json"
        record = {
            "schema_version": "simready_isaac_paid_attempt_consumption.v1",
            "authorization_digest": authorization_digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "probe_spec_sha256": authority.get("probe_spec_sha256"),
            "blueprint_commit": blueprint_commit,
            "consumed_at": utc_now_iso(),
            "maximum_provider_allocations": 1,
        }
        raw = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        temporary = root / f".simready-isaac-{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            directory_descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["simready_isaac_paid_attempt_authority_consumed"],
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["simready_isaac_authority_consumption_write_failed"],
        }
    return {
        "status": "consumed",
        "authorization_digest": authorization_digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def run_simready_isaac_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    paid_attempt_authority: Mapping[str, Any] | None = None,
    bundle_receipt_sha256: str | None = None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    expected_probe_names: frozenset[str] | None = None,
    allowed_active_instance_ids: Sequence[int] = (),
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 3.0,
    hard_ttl_seconds: int = 10_800,
) -> dict[str, Any]:
    """Run the frozen exact-scene packet with hard cumulative spend and time caps."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    bundle_path = Path(str(bundle.get("bundle_path") or "")).expanduser().resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("simready_isaac_prepared_bundle_binding_invalid")
    generated = utc_now_iso()
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "dry_run_ready",
            "bundle_sha256": bundle["bundle_sha256"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp009b_simready_isaac_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("simready_isaac_paid_resource_admission_grant_missing")
    if paid_attempt_authority is None:
        raise ValueError("simready_isaac_paid_attempt_authority_missing")
    validated_attempt_authority = validate_simready_isaac_paid_attempt_authority(
        paid_attempt_authority,
        prepared_bundle=bundle,
        bundle_receipt_sha256=bundle_receipt_sha256,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed_active_instance_ids,
    )
    blueprint_commit = str(
        bundle.get("source_commit_sha")
        or validated_attempt_authority.get("blueprint_commit")
        or ""
    )
    authorization_consumption = consume_simready_isaac_paid_attempt_authority_once(
        validated_attempt_authority, blueprint_commit=blueprint_commit
    )
    if authorization_consumption.get("status") != "consumed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "paid_attempt_authority_digest": validated_attempt_authority.get(
                "authorization_digest"
            ),
            "authorization_consumption": authorization_consumption,
            "blockers": authorization_consumption.get("blockers")
            or ["simready_isaac_paid_attempt_authority_consumption_blocked"],
        }
        write_json(job / "adp009b_simready_isaac_vast_result.json", result)
        return result

    number, attempt_root = _next_attempt(job)
    ensure_dir(attempt_root)
    ledger_path = job / "adp009b_simready_isaac_session_budget.json"
    remaining = _remaining_minutes(
        ledger_path=ledger_path,
        cap_usd=hard_cap_usd,
        ttl_seconds=hard_ttl_seconds,
        hourly_rate=max_hourly_rate_usd,
    )
    if remaining < 30:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "attempt_number": number,
            "provider_mutations_performed": 0,
            "paid_attempt_authority_digest": validated_attempt_authority.get(
                "authorization_digest"
            ),
            "authorization_consumption": authorization_consumption,
            "blockers": ["simready_isaac_cumulative_budget_below_minimum_live_window"],
        }
        write_json(job / "adp009b_simready_isaac_vast_result.json", result)
        return result
    staging_dir = attempt_root / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix=os.getenv("BLUEPRINT_ADP009B_ISAAC_OBJECT_STORE_PREFIX", DEFAULT_KEY_PREFIX),
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
        generated_at=generated,
    )
    if staging.get("status") != "completed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "attempt_number": number,
            "provider_mutations_performed": 0,
            "paid_attempt_authority_digest": validated_attempt_authority.get(
                "authorization_digest"
            ),
            "authorization_consumption": authorization_consumption,
            "blockers": staging.get("blockers") or ["simready_isaac_object_store_staging_blocked"],
        }
        write_json(job / "adp009b_simready_isaac_vast_result.json", result)
        return result
    bundle_url = (staging_dir / "provider_bundle_url.txt").read_text().strip()
    output_put = (staging_dir / "provider_output_put_url.txt").read_text().strip()
    output_get = (staging_dir / "provider_output_get_url.txt").read_text().strip()
    provider_run = attempt_root / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    local_avoidlist = job / "adp009b_simready_isaac_machine_avoidlist.json"
    resolved_avoidlist = (
        Path(machine_avoidlist_path).expanduser().resolve()
        if machine_avoidlist_path is not None
        else local_avoidlist
    )
    watchdog_handoff, watchdog_handle = arm_independent_vast_watchdog(
        job_dir=attempt_root,
        max_live_minutes=remaining,
        generated_at=generated,
        pod_name_prefix=INSTANCE_LABEL_PREFIX,
        allowed_active_instance_ids=allowed_active_instance_ids,
    )
    if watchdog_handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "attempt_number": number,
            "attempt_root": str(attempt_root),
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "paid_attempt_authority_digest": validated_attempt_authority.get(
                "authorization_digest"
            ),
            "authorization_consumption": authorization_consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_handoff,
            "blockers": ["simready_isaac_independent_watchdog_not_armed"],
        }
        write_json(attempt_root / "adp009b_simready_isaac_vast_result.json", result)
        write_json(job / "adp009b_simready_isaac_vast_result.json", result)
        return result
    adapter: dict[str, Any] = {}
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
                max_live_minutes=remaining,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=DEFAULT_IMAGE,
                isaac_image=DEFAULT_IMAGE,
                ngc_image_login_mode="always",
                provider_bundle=bundle_path,
                provider_bundle_url=bundle_url,
                provider_output_put_url=output_put,
                provider_output_get_url=output_get,
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=True,
                enable_blueprint_bundle=True,
                provider_bundle_kind="adp_simready_isaac",
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=True,
                min_cold_isaac_pull_live_minutes=30,
                disk_gb=200,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=remaining * 60,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=ledger_path,
                verify_staging_urls=True,
                require_known_supported_isaac_driver=True,
                preferred_gpu_keywords=("L40S", "RTX 4090", "RTX A6000", "RTX A5000"),
                prefer_isaac_rt=True,
                allowed_active_instance_ids=allowed_active_instance_ids,
                machine_avoidlist_path=resolved_avoidlist,
                # The exact collision-free label the independent process
                # watches, not merely the same broad lane family.
                instance_label_prefix=watchdog_handle.pod_name_prefix,
                started_instance_id_path=watchdog_handle.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        recorded_adapter = _read_json(provider_run / "vast_provider_adapter_result.json")
        adapter = {
            **recorded_adapter,
            "status": "blocked",
            "blockers": sorted(
                {
                    *(str(item) for item in recorded_adapter.get("blockers") or []),
                    "simready_isaac_vast_adapter_failed:"
                    + redacted_failure_detail(exc),
                }
            ),
            "raw_secret_values_recorded": False,
        }
        if (
            adapter.get("provider_create_attempted") is False
            and not watchdog_handle.started_instance_id_path.exists()
        ):
            seal_unallocated_provider_teardown(
                provider_run, reason="simready_isaac_vast_adapter_failed_before_create"
            )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract_result(output_zip, attempt_root / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    instance_ids: list[int] = []
    for value in (
        teardown.get("vast_instance_ids") or adapter.get("vast_instance_ids") or []
    ):
        if isinstance(value, bool):
            continue
        try:
            instance_id = int(value)
        except (TypeError, ValueError):
            continue
        if instance_id > 0 and instance_id not in instance_ids:
            instance_ids.append(instance_id)
    started_instance_path = watchdog_handle.started_instance_id_path
    if (
        not started_instance_path.is_symlink()
        and started_instance_path.is_file()
    ):
        try:
            started_instance_id = int(
                started_instance_path.read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            started_instance_id = 0
        if started_instance_id > 0 and started_instance_id not in instance_ids:
            instance_ids.append(started_instance_id)
    watchdog_close = close_independent_vast_watchdog(
        job_dir=attempt_root,
        handle=watchdog_handle,
        instance_ids=instance_ids,
        provider_teardown_completed=(
            teardown.get("continuing_spend_from_this_run") is False
        ),
        provider_allocation_impossible=(
            not instance_ids
            and not started_instance_path.exists()
            and adapter.get("provider_create_attempted") is False
        ),
    )
    blockers = [
        *(adapter.get("blockers") or []),
        *(extracted.get("blockers") or []),
        *_execution_blockers(
            execution,
            # An articulated bundle declares its own readbacks; validating
            # against the bundle's set keeps the check as strict as the rigid
            # one without assuming drop/slide/tip/gripper.
            frozenset(
                expected_probe_names
                or prepared_bundle.get("probe_names")
                or RIGID_PROBE_NAMES
            ),
        ),
    ]
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("simready_isaac_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("simready_isaac_object_store_zero_not_proven")
    if watchdog_close.get("status") not in {
        "provider_terminal",
        "cancelled_no_allocation",
    }:
        blockers.append("simready_isaac_independent_watchdog_not_closed")
    recorded_provider_mutations = adapter.get("provider_mutations_performed")
    provider_mutations_performed = (
        recorded_provider_mutations
        if isinstance(recorded_provider_mutations, int)
        and not isinstance(recorded_provider_mutations, bool)
        and recorded_provider_mutations >= 0
        else (1 if instance_ids else 0)
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "attempt_number": number,
        "attempt_root": str(attempt_root),
        "bundle_sha256": bundle["bundle_sha256"],
        "probe_spec_sha256": bundle.get("probe_spec_sha256"),
        "paid_attempt_authority_digest": validated_attempt_authority.get(
            "authorization_digest"
        ),
        "authorization_consumption": authorization_consumption,
        "provider_mutations_performed": provider_mutations_performed,
        "vast_instance_ids": instance_ids,
        "native_result_path": extracted.get("result_path"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "independent_watchdog": watchdog_close,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_json(attempt_root / "adp009b_simready_isaac_vast_result.json", result)
    # Seal the two terminal artifacts every production launch profile asks
    # this result for. Without them the run ends
    # `allocator_terminal_artifact_missing:` whatever happened on the provider.
    result = seal_lane_terminal_artifacts(
        result,
        # The provider run lives under this attempt, not under the job root:
        # this lane numbers its attempts. Sealing the job root found nothing
        # and said nothing, so the first live run reported `completed` with a
        # torn-down instance and no terminal artifacts.
        attempt_root=attempt_root,
        lane="public_scene_simready_isaac",
        extra_artifact_roots={
            "independent_watchdog": attempt_root / WATCHDOG_DIR_NAME,
        },
        binding={"provider": "vast"},
    )
    write_json(job / "adp009b_simready_isaac_vast_result.json", result)
    return result


__all__ = [
    "PAID_ATTEMPT_AUTHORITY_SCHEMA",
    "INSTANCE_LABEL_PREFIX",
    "PROBE_KIND",
    "consume_simready_isaac_paid_attempt_authority_once",
    "run_simready_isaac_vast",
    "validate_simready_isaac_paid_attempt_authority",
]
