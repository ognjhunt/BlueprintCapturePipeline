"""Canonical zero-retry Vast execution for the ADP-009B exact SimReady probe."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import zipfile

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .public_scene_simready_isaac_bundle import DEFAULT_IMAGE
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp009b-exact-simready-isaac"
RESULT_SCHEMA_VERSION = "adp009b_simready_isaac_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/exact-simready-isaac"
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


RIGID_PROBE_NAMES = frozenset({"drop", "slide", "tip", "gripper"})


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


def run_simready_isaac_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    expected_probe_names: frozenset[str] | None = None,
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
    if machine_avoidlist_path is not None:
        shutil.copy2(Path(machine_avoidlist_path).expanduser().resolve(), local_avoidlist)
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
                machine_avoidlist_path=local_avoidlist,
                instance_label_prefix="blueprint-adp009b-simready-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract_result(output_zip, attempt_root / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
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
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "attempt_number": number,
        "attempt_root": str(attempt_root),
        "bundle_sha256": bundle["bundle_sha256"],
        "probe_spec_sha256": bundle.get("probe_spec_sha256"),
        "native_result_path": extracted.get("result_path"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_json(attempt_root / "adp009b_simready_isaac_vast_result.json", result)
    write_json(job / "adp009b_simready_isaac_vast_result.json", result)
    return result


__all__ = ["PROBE_KIND", "run_simready_isaac_vast"]
