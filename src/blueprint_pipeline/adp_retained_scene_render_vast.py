"""One-watchdog Vast execution for an immutable retained-scene render bundle."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp_retained_scene_render_packet import (
    BUNDLE_SCHEMA,
    DEFAULT_IMAGE,
    ENTRYPOINT,
    PROBE_KIND,
)
from .common import ensure_dir, utc_now_iso, write_json, redacted_failure_detail
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_bundle_rehearsal import provider_bundle_rehearsal_blockers
from .task_evaluation_artifact_manifest import (
    seal_unallocated_provider_teardown,
    TaskEvaluationArtifactManifestError,
    build_task_evaluation_artifact_manifest,
)
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


from .spend_authority_consumption_root import (
    SpendAuthorityRootError,
    prepare_consumption_root,
)

PROVIDER_BUNDLE_KIND = "adp_retained_scene_render"
RESULT_SCHEMA = "adp009d_retained_scene_gpu_render_vast_run.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA = "adp009d_retained_scene_gpu_render_paid_attempt_authority.v1"
ATTEMPT_RECEIPT_SCHEMA = "adp009d_retained_scene_gpu_render_attempt_receipt.v1"
OUTPUT_RELOCATION_SCHEMA = "adp009d_retained_scene_gpu_render_output_relocation.v1"
_VAST_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")
_VAST_STALE_OFFER_RETRY_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("retained_scene_render_json_unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("retained_scene_render_json_not_object")
    return value


def validate_retained_scene_render_bundle(
    bundle: Mapping[str, Any], *, expected_commit: str | None = None
) -> dict[str, Any]:
    """Fail closed on every immutable input and dry-run binding."""

    value = dict(bundle)
    path = Path(str(value.get("bundle_path") or "")).expanduser().resolve()
    errors: list[str] = []
    if value.get("schema_version") != BUNDLE_SCHEMA or value.get("status") != "ready":
        errors.append("schema_or_status_invalid")
    if value.get("probe_kind") != PROBE_KIND or value.get("container_image") != DEFAULT_IMAGE:
        errors.append("probe_or_image_invalid")
    if expected_commit and value.get("blueprint_commit") != expected_commit:
        errors.append("blueprint_commit_mismatch")
    if not path.is_file() or _sha256(path) != value.get("bundle_sha256"):
        errors.append("bundle_sha256_mismatch")
    runtime_manifest: dict[str, Any] = {}
    if path.is_file():
        try:
            with zipfile.ZipFile(path) as archive:
                runtime_manifest = json.loads(
                    archive.read(
                        "provider_runtime/adp_retained_scene_gpu_render_manifest.json"
                    ).decode("utf-8")
                )
        except (KeyError, OSError, UnicodeError, ValueError, zipfile.BadZipFile):
            errors.append("runtime_manifest_unreadable")
    if (
        not isinstance(runtime_manifest, dict)
        or runtime_manifest.get("schema_version") != BUNDLE_SCHEMA
        or runtime_manifest.get("status") != "ready"
        or runtime_manifest.get("container_image") != DEFAULT_IMAGE
        or runtime_manifest.get("candidate_set_digest") != value.get("candidate_set_digest")
        or runtime_manifest.get("blueprint_commit") != value.get("blueprint_commit")
        or runtime_manifest.get("task_lanes") != value.get("task_lanes")
    ):
        errors.append("runtime_manifest_binding_invalid")
    if (
        value.get("raw_interiorgs_downloaded_bytes_included") is not False
        or value.get("private_scene_derived_standard_splats_included") is not True
        or value.get("provider_network_dependency_install_required") is not False
        or value.get("automatic_paid_retry_allowed") is not False
        or value.get("provider_zero_required_after_return") is not True
        or value.get("source_pair_per_task") is not True
        or value.get("deleted_source_layer_pair_per_task") is not True
        or value.get("retained_frame_per_task") is not True
        or not isinstance(value.get("shared_deleted_source_layer"), Mapping)
        or value["shared_deleted_source_layer"].get("source_layer_role")
        != "shared_deleted_source_union"
        or not str(value["shared_deleted_source_layer"].get("sha256") or "").startswith(
            "sha256:"
        )
        or value.get("maximum_replacement_objects") != 5
        or not str(value.get("candidate_set_digest") or "").startswith("sha256:")
        or not str(
            (value.get("execution_authority") or {}).get("authority_digest") or ""
        ).startswith("sha256:")
    ):
        errors.append("bundle_contract_invalid")
    lanes = value.get("task_lanes")
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= 5:
        errors.append("task_lane_count_invalid")
    elif any(
        not isinstance(row, Mapping)
        or not str(row.get("task_id") or "")
        or not str(row.get("task_freeze_digest") or "").startswith("sha256:")
        or not str(row.get("removal_id") or "")
        or not str(row.get("mask_set_id") or "")
        or not str(row.get("replacement_asset_id") or "")
        or int(row.get("camera_count") or 0) < 1
        for row in lanes
    ):
        errors.append("task_lane_binding_invalid")
    if provider_bundle_rehearsal_blockers(
        value.get("exact_bundle_entrypoint_rehearsal"),
        bundle_sha256=str(value.get("bundle_sha256") or ""),
        entrypoint_relative_path=ENTRYPOINT,
    ):
        errors.append("exact_bundle_rehearsal_invalid")
    if errors:
        raise ValueError("retained_scene_render_bundle_invalid:" + ",".join(sorted(set(errors))))
    return value


def validate_retained_scene_render_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int],
) -> dict[str, Any]:
    """Bind one human-authorized paid execution to these exact bundle bytes."""

    value = dict(authority)
    errors: list[str] = []
    expected_allowlist = sorted({int(item) for item in allowed_active_instance_ids})
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA:
        errors.append("schema_invalid")
    if value.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("authority_kind_invalid")
    if value.get("purpose") != "exact_retained_scene_gpu_render":
        errors.append("purpose_invalid")
    if value.get("provider") != "vast" or value.get("paid_compute_authorized") is not True:
        errors.append("provider_or_paid_authority_invalid")
    if value.get("parent_execution_authority_digest") != (
        prepared_bundle.get("execution_authority") or {}
    ).get("authority_digest"):
        errors.append("parent_execution_authority_digest_mismatch")
    if value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        errors.append("bundle_sha256_mismatch")
    if value.get("blueprint_commit") != prepared_bundle.get("blueprint_commit"):
        errors.append("blueprint_commit_mismatch")
    if value.get("maximum_paid_attempts") != 1 or value.get("maximum_automatic_retries") != 0:
        errors.append("single_attempt_retry_contract_invalid")
    if value.get("automatic_paid_retry_authorized") is not False:
        errors.append("automatic_retry_authorized")
    if value.get("hard_attempt_spend_cap_usd") != prepared_bundle.get("hard_total_spend_cap_usd"):
        errors.append("hard_spend_cap_mismatch")
    if value.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds:
        errors.append("hard_ttl_mismatch")
    if value.get("maximum_hourly_rate_usd") != max_hourly_rate_usd:
        errors.append("hourly_rate_mismatch")
    if (
        sorted({int(item) for item in value.get("external_active_instance_allowlist") or []})
        != expected_allowlist
    ):
        errors.append("external_active_instance_allowlist_mismatch")
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("authorization_digest_invalid")
    prior_attempts = value.get("prior_terminal_attempts", [])
    prior_spend_usd = 0.0
    if not isinstance(prior_attempts, list):
        errors.append("prior_terminal_attempts_invalid")
    else:
        if prior_attempts and value.get("manual_reissue_after_prior_terminal_attempt") is not True:
            errors.append("prior_terminal_attempt_manual_reissue_missing")
        seen_receipts: set[str] = set()
        for row in prior_attempts:
            if not isinstance(row, Mapping):
                errors.append("prior_terminal_attempts_invalid")
                continue
            result_path = Path(str(row.get("result_path") or "")).expanduser().resolve()
            expected_sha256 = str(row.get("result_sha256") or "")
            expected_digest = str(row.get("receipt_digest") or "")
            expected_cost = row.get("estimated_cost_usd")
            if (
                not result_path.is_file()
                or result_path.is_symlink()
                or expected_sha256 != _sha256(result_path)
                or expected_digest in seen_receipts
                or isinstance(expected_cost, bool)
                or not isinstance(expected_cost, (int, float))
                or float(expected_cost) < 0.0
            ):
                errors.append("prior_terminal_attempt_receipt_invalid")
                continue
            seen_receipts.add(expected_digest)
            try:
                result = _read(result_path)
            except ValueError:
                errors.append("prior_terminal_attempt_receipt_invalid")
                continue
            if (
                result.get("schema_version") != RESULT_SCHEMA
                or result.get("status") not in {"completed", "blocked"}
                or result.get("receipt_digest")
                != canonical_digest(result, digest_field="receipt_digest")
                or result.get("receipt_digest") != expected_digest
                or result.get("continuing_spend_from_this_run") is not False
                or result.get("all_staged_objects_absent") is not True
                or result.get("estimated_cost_usd") != expected_cost
            ):
                errors.append("prior_terminal_attempt_receipt_invalid")
                continue
            prior_spend_usd += float(expected_cost)
    if prior_spend_usd + max_hourly_rate_usd * hard_ttl_seconds / 3600.0 > float(
        prepared_bundle.get("hard_total_spend_cap_usd") or 0.0
    ):
        errors.append("aggregate_spend_cap_exceeded")
    if errors:
        raise ValueError(
            "retained_scene_render_paid_attempt_authority_invalid:" + ",".join(sorted(set(errors)))
        )
    return value


def consume_retained_scene_render_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    """Atomically reject any second provider allocation for this authority."""

    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["attempt_authority_identity_invalid"]}
    identity = digest.removeprefix("sha256:")
    try:
        # Creates the directory and tightens its mode where we own it, rather
        # than refusing a too-permissive one it could fix. A refusal there
        # surfaced as `attempt_authority_consumption_write_failed`, which named
        # the symptom and not the cause.
        root = prepare_consumption_root()
    except SpendAuthorityRootError as exc:
        return {"status": "blocked", "blockers": [str(exc)]}
    try:
        destination = root / f"retained-scene-render-{identity}.json"
        record = {
            "schema_version": "retained_scene_render_paid_attempt_consumption.v1",
            "authorization_digest": digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "blueprint_commit": blueprint_commit,
            "consumed_at": utc_now_iso(),
            "maximum_provider_allocations": 1,
        }
        raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = root / f".{identity}.{os.getpid()}.tmp"
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
        return {"status": "blocked", "blockers": ["attempt_authority_already_consumed"]}
    except OSError:
        return {"status": "blocked", "blockers": ["attempt_authority_consumption_write_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def _render_manifest_path(
    *, destination: Path, task_id: str, layer: str, background_rgb: str
) -> Path:
    if (
        not task_id
        or "/" in task_id
        or "\\" in task_id
        or layer not in {"shared_deleted_source_layer", "shared_retained_scene"}
        or background_rgb not in {"#000000", "#ffffff"}
    ):
        raise ValueError("retained_scene_render_manifest_reference_invalid")
    background = "black" if background_rgb == "#000000" else "white"
    path = (
        destination
        / "renders"
        / task_id
        / f"{task_id}_{layer}_{background}"
        / "sealed_camera_render_manifest.v1.json"
    ).resolve()
    root = destination.resolve()
    if root not in path.parents:
        raise ValueError("retained_scene_render_manifest_reference_invalid")
    return path


def materialize_retained_scene_render_output_relocation(
    *, result_path: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Bind extracted renderer manifests to the untouched provider result bytes.

    Provider runtime manifests contain paths meaningful only inside the
    container.  This receipt neither edits the exported result nor trusts
    those path strings: it deterministically resolves the extracted filename,
    reopens the local manifest, and verifies its digest before exposing a local
    path for the next file-backed evidence gate.
    """

    result_file = Path(result_path).expanduser().resolve()
    root = Path(destination).expanduser().resolve()
    if not result_file.is_file() or result_file.is_symlink() or not root.is_dir():
        raise ValueError("retained_scene_render_output_relocation_input_missing")
    if result_file.parent != root:
        raise ValueError("retained_scene_render_output_relocation_result_outside_root")
    result = _read(result_file)
    if result.get("schema_version") != "adp009d_retained_scene_gpu_render_result.v1":
        raise ValueError("retained_scene_render_output_relocation_result_invalid")
    rows = result.get("render_manifests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("retained_scene_render_output_relocation_manifests_missing")
    local_manifests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("retained_scene_render_output_relocation_manifests_invalid")
        task_id = str(row.get("task_id") or "")
        layer = str(row.get("layer") or "")
        background_rgb = str(row.get("background_rgb") or "")
        key = (task_id, layer, background_rgb)
        if key in seen:
            raise ValueError("retained_scene_render_output_relocation_manifest_duplicate")
        seen.add(key)
        manifest_path = _render_manifest_path(
            destination=root,
            task_id=task_id,
            layer=layer,
            background_rgb=background_rgb,
        )
        manifest = _read(manifest_path)
        digest = manifest.get("sealed_camera_render_manifest_digest")
        expected_role = (
            "shared_deleted_source_union"
            if layer == "shared_deleted_source_layer"
            else "shared_retained_scene"
        )
        if (
            manifest.get("schema_version") != "sealed_camera_render_manifest.v1"
            or manifest.get("status") != "rendered_exact_cameras"
            or manifest.get("source_layer_role") != expected_role
            or manifest.get("render_settings", {}).get("background_rgb") != background_rgb
            or digest != row.get("manifest_digest")
            or digest != canonical_digest(
                manifest, digest_field="sealed_camera_render_manifest_digest"
            )
        ):
            raise ValueError("retained_scene_render_output_relocation_manifest_invalid")
        local_manifests.append(
            {
                "task_id": task_id,
                "layer": layer,
                "background_rgb": background_rgb,
                "remote_manifest_path": row.get("manifest_path"),
                "local_manifest": {
                    "path": str(manifest_path),
                    "size_bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                    "manifest_digest": digest,
                },
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": OUTPUT_RELOCATION_SCHEMA,
        "status": "extracted_manifest_paths_verified",
        "provider_result": {
            "path": str(result_file),
            "size_bytes": result_file.stat().st_size,
            "sha256": _sha256(result_file),
        },
        "render_manifests": local_manifests,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = root / "adp009d_retained_scene_gpu_render_output_relocation.v1.json"
    write_json(receipt_path, receipt)
    return receipt


def _extract_provider_output(
    path: Path, destination: Path
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    blockers: list[str] = []
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    blockers.append("provider_output_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("provider_output_zip_invalid")
    result_path = destination / "adp009d_retained_scene_gpu_render_result.v1.json"
    result = _read(result_path) if result_path.is_file() else {}
    if not result:
        blockers.append("provider_result_missing")
    relocation: dict[str, Any] | None = None
    if not blockers:
        try:
            relocation = materialize_retained_scene_render_output_relocation(
                result_path=result_path, destination=destination
            )
        except ValueError:
            blockers.append("provider_output_manifest_relocation_invalid")
    return result, blockers, relocation


@contextmanager
def _authority_environment():
    environment_names = (*_VAST_MUTATION_ENV, _VAST_STALE_OFFER_RETRY_ENV)
    previous = {name: os.environ.get(name) for name in environment_names}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_STALE_OFFER_RETRY_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_retained_scene_render_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    paid_attempt_authority: Mapping[str, Any] | None = None,
    max_hourly_rate_usd: float = 2.0,
    hard_ttl_seconds: int = 10_800,
    allowed_active_instance_ids: Sequence[int] = (),
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one exact derived-only GPU render, then require provider zero."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = validate_retained_scene_render_bundle(prepared_bundle)
    if max_hourly_rate_usd <= 0 or hard_ttl_seconds < 1800:
        raise ValueError("retained_scene_render_budget_or_ttl_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle_sha256": bundle["bundle_sha256"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "retained_scene_render_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None or paid_attempt_authority is None:
        raise ValueError("retained_scene_render_paid_admission_or_authority_missing")
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class="vast_provider_adapter",
        require_allocation_binding=True,
    )
    authority = validate_retained_scene_render_paid_attempt_authority(
        paid_attempt_authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed_active_instance_ids,
    )
    bundle_path = Path(str(bundle["bundle_path"])).resolve()
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix="blueprint/arm-decision-proof-v1/retained-scene-render",
        expiration_seconds=hard_ttl_seconds + 1800,
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": list(staging.get("blockers") or ["object_store_staging_blocked"]),
        }
    watchdog_handoff, watchdog = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=hard_ttl_seconds // 60,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed_active_instance_ids,
        pod_name_prefix="blueprint-groot-oscar-canary-adp-retained-render-",
    )
    if watchdog is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        return {
            "schema_version": RESULT_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_handoff,
            "blockers": ["independent_watchdog_not_armed"],
        }
    consumption = consume_retained_scene_render_paid_attempt_authority_once(
        authority, blueprint_commit=str(bundle["blueprint_commit"])
    )
    if consumption.get("status") != "consumed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        watchdog_close = close_independent_vast_watchdog(
            job_dir=job,
            handle=watchdog,
            instance_ids=[],
            provider_teardown_completed=False,
            provider_allocation_impossible=True,
        )
        return {
            "schema_version": RESULT_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "authorization_consumption": consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_close,
            "blockers": list(consumption.get("blockers") or []),
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=float(bundle["hard_total_spend_cap_usd"]),
                hard_cap_usd=float(bundle["hard_total_spend_cap_usd"]),
                max_live_minutes=hard_ttl_seconds // 60,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=DEFAULT_IMAGE,
                isaac_image=DEFAULT_IMAGE,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt")
                .read_text()
                .strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt")
                .read_text()
                .strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=64,
                min_gpu_ram_mb=16_000,
                poll_interval_seconds=10,
                startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=1200,
                session_budget_ledger_path=job / "retained_scene_render_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090", "L40S", "RTX A6000"),
                prefer_isaac_rt=False,
                allowed_active_instance_ids=allowed_active_instance_ids,
                machine_avoidlist_path=machine_avoidlist_path,
                vast_launch_lock_file=job.parent / "retained_scene_render_paid_launch.lock",
                instance_label_prefix="blueprint-adp-retained-render-",
                started_instance_id_path=watchdog.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {"status": "blocked", "blockers": [f"vast_adapter_failed:{redacted_failure_detail(exc)}"]}
        # The adapter may never have been entered -- resolving a secret or a
        # staged URL raises before it. Record the absence of any allocation so
        # the run can close; the sealer declines whenever the evidence does not
        # support that claim.
        seal_unallocated_provider_teardown(
            provider_run, reason="vast_adapter_failed"
        )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path) if teardown_path.is_file() else {}
    instance_ids = [
        int(value)
        for value in (teardown.get("vast_instance_ids") or adapter.get("vast_instance_ids") or [])
        if isinstance(value, int) and value > 0
    ]
    watchdog_close = close_independent_vast_watchdog(
        job_dir=job,
        handle=watchdog,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    execution, blockers, relocation = _extract_provider_output(
        output_zip, job / "immutable_execution"
    )
    if execution.get("status") != "completed":
        blockers.append("provider_render_not_completed")
    if (
        execution.get("released_renderer_executed") is not True
        or execution.get("gpu_runtime_started") is not True
        or execution.get("paid_inference_performed") is not False
        or execution.get("provider_mutations_performed") != 0
    ):
        blockers.append("provider_render_execution_contract_invalid")
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("object_store_provider_zero_not_proven")
    if watchdog_close.get("status") not in {"provider_terminal", "cancelled_no_allocation"}:
        blockers.append("independent_watchdog_not_closed")
    # The terminal contract asks the result for these two paths, and this lane
    # named neither. Every run therefore ended
    # `allocator_terminal_artifact_missing:teardown_manifest_path` and
    # `:artifact_manifest_path` no matter what happened on the provider -- the
    # teardown manifest had been written next to the adapter result the whole
    # time, and nothing pointed at it. A teardown that is not referenced cannot
    # be checked, so provider-zero could never be verified from this lane.
    # The shared manifest, not a lane-local one. `adp009d_live_readiness` and
    # every future consumer validate `task_evaluation_artifact_manifest.v1`, so
    # a second schema here would mean each lane's evidence had to be read by a
    # reader written for it. Roles also state what coverage is *required*
    # rather than sweeping whatever happens to be on disk.
    artifact_manifest_path = job / "artifact_manifest.json"
    try:
        artifact_manifest = build_task_evaluation_artifact_manifest(
            attempt_root=job,
            artifact_roots={
                "provider_runtime_evidence": job / "immutable_execution",
                "allocator_adapter_result": (
                    provider_run / "vast_provider_adapter_result.json"
                ),
                "teardown_manifest": provider_run / "vast_teardown_manifest.json",
                # Retained but not required: a blocked attempt is diagnosed from
                # these, and they are exactly what is absent when it failed
                # early.
                "provider_run_diagnostics": provider_run,
            },
            required_roles=(
                "provider_runtime_evidence",
                "allocator_adapter_result",
                "teardown_manifest",
            ),
            binding={
                "allocator_lane": PROVIDER_BUNDLE_KIND,
                "blueprint_commit": bundle.get("blueprint_commit"),
                "bundle_sha256": bundle.get("bundle_sha256"),
                "provider": "vast",
                "result_schema_version": RESULT_SCHEMA,
                "retry_cap": 0,
            },
            output_path=artifact_manifest_path,
        )
    except TaskEvaluationArtifactManifestError as exc:
        artifact_manifest = {"status": "blocked", "blockers": [str(exc)]}
        blockers.append("retained_scene_render_artifact_manifest_invalid")
    if artifact_manifest.get("status") != "completed":
        blockers.extend(str(item) for item in artifact_manifest.get("blockers") or [])
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "authorization_consumption": consumption,
        "provider_adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "artifact_manifest_path": str(artifact_manifest_path)
        if artifact_manifest_path.is_file()
        else None,
        # Null rather than a path that is not there: an unwritten teardown
        # manifest is the absence of teardown evidence, and naming a
        # nonexistent file would let a later reader think one was produced.
        "teardown_manifest_path": str(teardown_path) if teardown_path.is_file() else None,
        "execution_result_path": str(
            job / "immutable_execution/adp009d_retained_scene_gpu_render_result.v1.json"
        ),
        "output_relocation_receipt": (
            {
                "path": str(
                    job
                    / "immutable_execution"
                    / "adp009d_retained_scene_gpu_render_output_relocation.v1.json"
                ),
                "receipt_digest": relocation.get("receipt_digest"),
            }
            if relocation is not None
            else None
        ),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": bundle["hard_total_spend_cap_usd"],
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "independent_watchdog": watchdog_close,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    write_json(job / "retained_scene_render_vast_result.json", result)
    return result


__all__ = [
    "ATTEMPT_RECEIPT_SCHEMA",
    "PAID_ATTEMPT_AUTHORITY_SCHEMA",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_SCHEMA",
    "materialize_retained_scene_render_output_relocation",
    "consume_retained_scene_render_paid_attempt_authority_once",
    "run_retained_scene_render_vast",
    "validate_retained_scene_render_bundle",
    "validate_retained_scene_render_paid_attempt_authority",
]
