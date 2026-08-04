"""Canonical Vast execution adapter for the pinned SIMPLER ADP reference.

This is deliberately a thin wrapper around the existing provider bundle,
object-store, paid-admission, Vast watchdog/teardown, and provider-zero seams.
It transfers only public, digest-bound inputs and never bundles the separately
held physical-reference outcome values.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .paid_resource_admission import PaidResourceAdmissionGrant
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-simpler-public-reference"
RESULT_SCHEMA_VERSION = "adp_simpler_vast_run.v1"
DEFAULT_IMAGE = (
    "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@"
    "sha256:131e238d724ee145317f10d6c8eba0d301439c6c8764b02473510e7035756e81"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/simpler"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)


ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_SIMPLER_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
mkdir -p "$OUT_DIR"
python3 -m venv "$RUNTIME_DIR/.venv"
venv_rc=$?
if [ $venv_rc -eq 0 ]; then
  "$RUNTIME_DIR/.venv/bin/python" "$RUNTIME_DIR/adp_simpler_provider_runner.py" \
    --manifest "$RUNTIME_DIR/public_reference_manifest.json" \
    --output-dir "$OUT_DIR"
  runner_rc=$?
else
  runner_rc=$venv_rc
fi
if [ $runner_rc -ne 0 ] && [ ! -f "$OUT_DIR/adp_simpler_closed_loop_execution.json" ]; then
python3 - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "adp_simpler_closed_loop_execution.json").write_text(json.dumps({
    "schema_version": "simpler_closed_loop_execution.v1",
    "status": "blocked",
    "blockers": [
        "adp_simpler_runner_failed_without_runtime_result",
        "blocked_adp_simpler_process_exited_without_result"
    ],
    "physical_outcome_values_accessed": False,
    "phase_label": "retrospective_external_reference",
    "claim_ceiling": "development_only"
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi
exit $runner_rc
'''


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _adp_session_budget_ledger(job: Path) -> Path:
    return job.resolve() / "adp_vast_session_budget.json"


@contextmanager
def _vast_authority_environment():
    """Bridge the canonical grant to the adapter's defense-in-depth env gates."""

    previous = {name: os.environ.get(name) for name in _VAST_MUTATION_ENV}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_simpler_public_vast_bundle(
    *, manifest_path: str | Path, job_dir: str | Path, generated_at: str | None = None
) -> dict[str, Any]:
    """Build the small public-input runtime bundle without outcome values."""

    source_manifest = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(source_manifest)
    job = Path(job_dir).expanduser().resolve()
    if job.exists():
        shutil.rmtree(job)
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    generated = generated_at or utc_now_iso()
    blockers: list[str] = []
    if manifest.get("schema_version") != "public_reference_admission.v1":
        blockers.append("adp_public_reference_manifest_invalid")
    if manifest.get("reference_id") != "simpler-google-robot-pick-coke-can-v1":
        blockers.append("adp_public_reference_identity_invalid")
    if len(manifest.get("candidates") or []) != 2:
        blockers.append("adp_public_reference_must_have_exactly_two_candidates")
    if len(manifest.get("conditions") or []) != 3:
        blockers.append("adp_public_reference_condition_matrix_invalid")
    environment = dict(dict(manifest.get("runtime") or {}).get("environment_lock") or {})
    if environment.get("container_image") != DEFAULT_IMAGE:
        blockers.append("adp_public_reference_container_image_not_exact")

    worker_source = Path(__file__).with_name("simpler_public_runtime_worker.py")
    shutil.copy2(worker_source, runtime / "adp_simpler_provider_runner.py")
    shutil.copy2(source_manifest, runtime / "public_reference_manifest.json")
    _write_executable(runtime / "run_adp_simpler_provider_runtime.sh", ENTRYPOINT)
    readiness = {
        "schema_version": "adp_simpler_provider_bundle.v1",
        "generated_at": generated,
        "status": "ready" if not blockers else "blocked",
        "reference_id": manifest.get("reference_id"),
        "source_identity_digest": manifest.get("source_identity_digest"),
        "source_manifest_digest": manifest.get("manifest_digest"),
        "candidate_ids": [
            row.get("candidate_id") for row in manifest.get("candidates") or []
        ],
        "condition_ids": [
            row.get("condition_id") for row in manifest.get("conditions") or []
        ],
        "container_image": environment.get("container_image"),
        "runtime_entrypoint": "provider_runtime/run_adp_simpler_provider_runtime.sh",
        "expected_output_filename": "adp_simpler_closed_loop_execution.json",
        "physical_outcome_values_bundled": False,
        "local_bundle_ready_for_remote_staging": not blockers,
        "blockers": sorted(set(blockers)),
        "phase_label": "retrospective_external_reference",
        "claim_ceiling": "development_only",
        "raw_secret_values_recorded": False,
    }
    readiness_path = runtime / "adp_simpler_provider_manifest.json"
    write_json(readiness_path, readiness)
    bundle_path = job / "adp_simpler_provider_runtime_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    path.relative_to(job).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.create_system = 3
                info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
                archive.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    receipt = {
        **readiness,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _file_sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "manifest_path": str(readiness_path),
    }
    write_json(job / "adp_simpler_bundle_receipt.json", receipt)
    return receipt


def _extract_provider_output(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {"status": "blocked", "blockers": ["adp_provider_output_zip_missing"]}
    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("adp_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        blockers.append(f"adp_provider_output_zip_invalid:{type(exc).__name__}")
    execution_path = destination / "adp_simpler_closed_loop_execution.json"
    execution = _read_json(execution_path)
    if not execution:
        blockers.append("adp_provider_execution_package_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "execution_path": str(execution_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


def run_simpler_public_vast(
    *,
    manifest_path: str | Path,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any] | None = None,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 2.00,
    hard_ttl_seconds: int = 7200,
) -> dict[str, Any]:
    """Run one zero-retry, two-candidate SIMPLER acquisition on Vast."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    generated = utc_now_iso()
    bundle = (
        dict(prepared_bundle)
        if prepared_bundle is not None
        else build_simpler_public_vast_bundle(
            manifest_path=manifest_path, job_dir=job / "bundle", generated_at=generated
        )
    )
    bundle_path = Path(str(bundle.get("bundle_path") or "")).expanduser().resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _file_sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("adp_simpler_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "dry_run_ready" if bundle.get("status") == "ready" else "blocked",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": bundle.get("blockers", []),
        }
        write_json(job / "adp_simpler_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_simpler_paid_resource_admission_grant_missing")

    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=os.getenv("BLUEPRINT_ADP_SIMPLER_OBJECT_STORE_PREFIX", DEFAULT_KEY_PREFIX),
        expiration_seconds=max(hard_ttl_seconds + 1800, 10800),
        generated_at=generated,
    )
    if staging.get("status") != "completed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "bundle": bundle,
            "object_store_staging": staging,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": staging.get("blockers") or ["adp_object_store_staging_blocked"],
        }
        write_json(job / "adp_simpler_vast_result.json", result)
        return result

    bundle_url = (staging_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
    output_put_url = (staging_dir / "provider_output_put_url.txt").read_text(
        encoding="utf-8"
    ).strip()
    output_get_url = (staging_dir / "provider_output_get_url.txt").read_text(
        encoding="utf-8"
    ).strip()
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    local_avoidlist = job / "adp_vast_machine_avoidlist.json"
    if machine_avoidlist_path is not None:
        source_avoidlist = Path(machine_avoidlist_path).expanduser().resolve()
        shutil.copy2(source_avoidlist, local_avoidlist)
    adapter: dict[str, Any] = {}
    try:
        with _vast_authority_environment():
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
                provider_bundle=str(bundle_path),
                provider_bundle_url=bundle_url,
                provider_output_put_url=output_put_url,
                provider_output_get_url=output_get_url,
                provider_runtime_output_zip=output_zip,
                enable_blueprint_bundle=True,
                provider_bundle_kind="adp_simpler",
                vast_launch_mode="ssh_direct",
                disk_gb=50,
                min_gpu_ram_mb=16000,
                poll_interval_seconds=15,
                startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=900,
                session_budget_ledger_path=_adp_session_budget_ledger(job),
                verify_staging_urls=True,
                preferred_gpu_keywords=("RTX 4090", "RTX 3090", "RTX A5000"),
                machine_avoidlist_path=local_avoidlist,
                instance_label_prefix="blueprint-adp-simpler-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract_provider_output(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or [])
    blockers.extend(extracted.get("blockers") or [])
    if execution.get("status") not in {"completed", "completed_with_retained_failures"}:
        blockers.extend(execution.get("blockers") or ["adp_execution_not_completed"])
    if execution and execution.get("physical_outcome_values_accessed") is not False:
        blockers.append("adp_execution_outcome_firebreak_violated")
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("adp_vast_provider_zero_not_proven")
    if cleanup.get("status") != "completed" or cleanup.get("all_objects_absent") is not True:
        blockers.append("adp_object_store_provider_zero_not_proven")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "source_identity_digest": bundle.get("source_identity_digest"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "execution_path": extracted.get("execution_path"),
        "execution_digest": execution.get("execution_digest"),
        "runtime_lock_digest": execution.get("runtime_lock_digest"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(provider_run / "vast_teardown_manifest.json"),
        "object_store_cleanup_path": str(
            staging_dir / "wam_provider_object_store_cleanup.json"
        ),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "phase_label": "retrospective_external_reference",
        "claim_ceiling": "development_only",
        "raw_secret_values_recorded": False,
    }
    write_json(job / "adp_simpler_vast_result.json", result)
    return result


__all__ = [
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "build_simpler_public_vast_bundle",
    "run_simpler_public_vast",
]
