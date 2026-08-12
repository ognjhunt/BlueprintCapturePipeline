"""Execute one sealed ArtiFixer/3D/3D+ candidate packet on Vast.

This adapter is deliberately narrow.  It accepts a rehearsed 1--5 object
packet containing only private-derived frames, exact masks, and released
source; consumes one file-backed authority; arms an independent watchdog
before allocation; and returns candidate appearance only after object-store
cleanup and API-confirmed provider zero.  It never promotes generated pixels
to observed hidden-background or physical evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .public_scene_artifixer3d_bundle import (
    DEFAULT_IMAGE,
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION as BUNDLE_SCHEMA_VERSION,
    USE_ATTESTATION_SCHEMA_VERSION,
)
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


PROBE_KIND = "adp-artifixer3d-exact-support"
PROVIDER_BUNDLE_KIND = "adp_artifixer3d"
RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_vast_run.v1"
RAW_RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_raw_result.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "public_scene_artifixer3d_paid_attempt_authority.v1"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/artifixer3d-exact-support"
INSTANCE_LABEL_PREFIX = "blueprint-adp-artifixer3d-canary-"
MIN_TTL_SECONDS = 7_200
MAX_TTL_SECONDS = 21_600
MAX_HARD_CAP_USD = 10.0
MIN_GPU_RAM_MB = 78_000
MIN_COMPUTE_CAP = 800
GPU_SELECTION_POLICY = {
    "policy_id": "artifixer3d_a100_80gb_author_control",
    "allowed_gpu_keywords": ("A100",),
    "denied_gpu_keywords": ("A100X",),
    "reason": (
        "the released 1.3B author workflow is documented for one 80 GB GPU; "
        "A100 uses the released cuDNN SDPA fallback without Hopper-only kernels"
    ),
}
AUTHORIZATION_CONSUMPTION_ROOT = Path.home() / ".blueprint-spend-authority" / "consumed"
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")
_RETRY_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(path: Path, *, code: str = "artifixer3d_receipt_unreadable") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _zip_json(archive: zipfile.ZipFile, name: str, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _zip_text(archive: zipfile.ZipFile, name: str, *, code: str) -> str:
    try:
        return archive.read(name).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError(code) from exc


def _validate_parent_execution_authority(
    attestation: Mapping[str, Any], *, publisher_scene_id: str
) -> tuple[Path, dict[str, Any]]:
    path = _bound(
        attestation.get("parent_execution_authority"),
        code="artifixer3d_parent_execution_authority_unbound",
    )
    authority = _read(path, code="artifixer3d_parent_execution_authority_unreadable")
    paid = authority.get("paid_compute")
    if (
        authority.get("schema_version") != "third_scene_dual_task_execution_authority.v1"
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or not publisher_scene_id
        or authority.get("publisher_scene_id") != publisher_scene_id
        or not isinstance(paid, Mapping)
        or paid.get("provider") != "vast"
        or paid.get("zero_retry") is not True
        or paid.get("provider_zero_required_for_lane") is not True
        or paid.get("hard_total_spend_cap_usd") != 12.0
    ):
        raise ValueError("artifixer3d_parent_execution_authority_invalid")
    return path, authority


def validate_artifixer3d_bundle(receipt_path: str | Path) -> dict[str, Any]:
    """Re-open every immutable bundle binding before paid admission."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = _read(path, code="artifixer3d_bundle_receipt_unreadable")
    if (
        path.is_symlink()
        or receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "sealed_rehearsal_passed_no_upload_no_execution"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("provider_mutations_performed") != 0
        or receipt.get("private_derived_upload_performed") is not False
        or not 1 <= int(receipt.get("replacement_object_count") or 0) <= 5
        or receipt.get("local_rehearsal", {}).get("status") != "passed"
        or receipt.get("local_rehearsal", {}).get("provider_mutations_performed") != 0
    ):
        raise ValueError("artifixer3d_bundle_receipt_invalid")
    bundle_path = _bound(receipt.get("bundle"), code="artifixer3d_bundle_unbound")
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("artifixer3d_bundle_zip_integrity_failed")
            names = set(archive.namelist())
            required = {
                "provider_runtime/run_public_scene_artifixer3d.sh",
                "provider_runtime/public_scene_artifixer3d_runner.py",
                "provider_runtime/artifixer3d_bundle_manifest.json",
                "provider_runtime/artifixer3d_runtime_request.json",
                "provider_runtime/input/public_scene_artifixer3d_candidate_inputs.v3.json",
                "provider_runtime/artifixer3d_use_attestation.json",
            }
            if not required.issubset(names):
                raise ValueError("artifixer3d_bundle_required_entries_missing")
            manifest = _zip_json(
                archive,
                "provider_runtime/artifixer3d_bundle_manifest.json",
                code="artifixer3d_bundle_manifest_invalid",
            )
            request = _zip_json(
                archive,
                "provider_runtime/artifixer3d_runtime_request.json",
                code="artifixer3d_bundle_request_invalid",
            )
            candidate = _zip_json(
                archive,
                "provider_runtime/input/public_scene_artifixer3d_candidate_inputs.v3.json",
                code="artifixer3d_bundle_candidate_invalid",
            )
            attestation = _zip_json(
                archive,
                "provider_runtime/artifixer3d_use_attestation.json",
                code="artifixer3d_bundle_attestation_invalid",
            )
            blockers = provider_runtime_contract_blockers(
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                entrypoint_text=_zip_text(
                    archive,
                    "provider_runtime/run_public_scene_artifixer3d.sh",
                    code="artifixer3d_bundle_entrypoint_invalid",
                ),
                runner_text=_zip_text(
                    archive,
                    "provider_runtime/public_scene_artifixer3d_runner.py",
                    code="artifixer3d_bundle_runner_invalid",
                ),
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("artifixer3d_bundle_zip_invalid") from exc
    if blockers:
        raise ValueError("artifixer3d_bundle_runtime_contract_invalid")
    identity = manifest.get("blueprint_source_identity")
    tasks = candidate.get("tasks")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or request.get("runtime_request_digest")
        != canonical_digest(request, digest_field="runtime_request_digest")
        or request.get("runtime_request_digest") != receipt.get("runtime_request_digest")
        or manifest.get("manifest_digest") != receipt.get("manifest_digest")
        or candidate.get("receipt_digest") != receipt.get("candidate_input_receipt_digest")
        or candidate.get("replacement_object_count") != receipt.get("replacement_object_count")
        or not isinstance(tasks, list)
        or len(tasks) != receipt.get("replacement_object_count")
        or request.get("task_ids") != receipt.get("task_ids")
        or request.get("source_object_restoration_permitted") is not False
        or request.get("outside_exact_support_changed_pixels_permitted") != 0
        or request.get("repair_target")
        != "plausible_object_free_background_inside_exact_support_only"
        or manifest.get("contains_raw_dataset_bytes") is not False
        or manifest.get("contains_model_weights") is not False
        or manifest.get("container_image") != DEFAULT_IMAGE
        or receipt.get("container_image") != DEFAULT_IMAGE
        or not isinstance(identity, Mapping)
        or identity != receipt.get("blueprint_source_identity")
        or identity.get("tracked_files_clean") is not True
        or len(str(identity.get("commit") or "")) != 40
        or len(str(identity.get("tree") or "")) != 40
        or attestation.get("schema_version") != USE_ATTESTATION_SCHEMA_VERSION
        or attestation.get("attestation_digest")
        != canonical_digest(attestation, digest_field="attestation_digest")
        or attestation.get("attestation_digest") != receipt.get("use_attestation_digest")
        or attestation.get("private_derived_input_upload_authorized") is not True
        or attestation.get("raw_dataset_bytes_upload_authorized") is not False
        or attestation.get("provider_training_authorized") is not False
        or attestation.get("internal_noncommercial_research_and_development_only")
        is not True
        or attestation.get("simulator_or_generated_output_is_physical_evidence") is not False
    ):
        raise ValueError("artifixer3d_bundle_binding_invalid")
    task_ids: list[str] = []
    task_camera_counts: dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("artifixer3d_bundle_task_invalid")
        task_id = str(task.get("task_id") or "")
        camera_count = task.get("camera_count")
        if (
            not task_id
            or task_id in task_camera_counts
            or isinstance(camera_count, bool)
            or not isinstance(camera_count, int)
            or camera_count < 2
            or len(task.get("frames") or []) != camera_count
        ):
            raise ValueError("artifixer3d_bundle_task_invalid")
        task_ids.append(task_id)
        task_camera_counts[task_id] = camera_count
    if task_ids != receipt.get("task_ids"):
        raise ValueError("artifixer3d_bundle_task_order_invalid")
    publisher_scene_id = str(candidate.get("publisher_scene_id") or "")
    if attestation.get("publisher_scene_id") != publisher_scene_id:
        raise ValueError("artifixer3d_bundle_scene_binding_invalid")
    parent_path, parent = _validate_parent_execution_authority(
        attestation, publisher_scene_id=publisher_scene_id
    )
    parent_allowlist = sorted(
        set(
            int(value)
            for value in parent.get("paid_compute", {}).get(
                "external_instance_allowlist", []
            )
        )
    )
    allowed_active_instance_ids = sorted(
        set(int(value) for value in manifest.get("allowed_active_instance_ids") or [])
    )
    return {
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "receipt_digest": receipt["receipt_digest"],
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
        "manifest_digest": manifest["manifest_digest"],
        "runtime_request_digest": request["runtime_request_digest"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "use_attestation_digest": attestation["attestation_digest"],
        "parent_execution_authority_path": str(parent_path),
        "parent_execution_authority_digest": parent["authority_digest"],
        "aggregate_goal_spend_cap_usd": parent["paid_compute"]["hard_total_spend_cap_usd"],
        "blueprint_source_identity": dict(identity),
        "container_image": DEFAULT_IMAGE,
        "publisher_scene_id": publisher_scene_id,
        "allowed_active_instance_ids": allowed_active_instance_ids,
        "forbidden_external_instance_ids": sorted(
            set(parent_allowlist) - set(allowed_active_instance_ids)
        ),
        "replacement_object_count": receipt["replacement_object_count"],
        "task_ids": task_ids,
        "task_camera_counts": task_camera_counts,
    }


def _validate_prior_authority_chain(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    """Recursively re-open the predecessor authority's complete spend chain."""

    seen = set() if seen is None else seen
    resolved = path.expanduser().resolve()
    if resolved in seen or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("artifixer3d_prior_authority_chain_invalid")
    seen.add(resolved)
    value = _read(resolved, code="artifixer3d_prior_authority_unreadable")
    if (
        value.get("schema_version")
        != "public_scene_aura_exact_residual_paid_attempt_authority.v1"
        or value.get("authorization_digest")
        != canonical_digest(value, digest_field="authorization_digest")
        or value.get("automatic_paid_retry_authorized") is not False
        or value.get("maximum_automatic_retries") != 0
        or value.get("maximum_paid_attempts") != 1
        or value.get("aggregate_goal_spend_cap_usd") != 12.0
        or isinstance(value.get("prior_goal_spend_usd"), bool)
        or not isinstance(value.get("prior_goal_spend_usd"), (int, float))
    ):
        raise ValueError("artifixer3d_prior_authority_invalid")
    for field in (
        "previous_terminal_execution_result",
        "previous_runtime_result",
        "previous_teardown",
        "previous_watchdog",
        "previous_object_store_cleanup",
        "prior_provider_runtime_campaign",
    ):
        _bound(value.get(field), code="artifixer3d_prior_authority_dependency_unbound")
    for record in value.get("additional_terminal_spend_receipts") or []:
        _bound(record, code="artifixer3d_prior_authority_spend_unbound")
    parent = value.get("prior_manual_corrected_attempt_authority")
    if parent is not None:
        _validate_prior_authority_chain(
            _bound(parent, code="artifixer3d_prior_authority_parent_unbound"),
            seen=seen,
        )
    return value


def _validate_prior_terminal_result(
    path: Path, *, prior_authority: Mapping[str, Any]
) -> tuple[dict[str, Any], float]:
    result = _read(path, code="artifixer3d_prior_terminal_result_unreadable")
    cost = result.get("estimated_cost_usd")
    if (
        result.get("schema_version") != "public_scene_aura_exact_residual_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != prior_authority.get("authorization_digest")
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError("artifixer3d_prior_terminal_result_invalid")
    teardown_path = Path(str(result.get("teardown_manifest_path") or "")).resolve()
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = path.parent / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    teardown = _read(teardown_path, code="artifixer3d_prior_teardown_unreadable")
    watchdog = _read(watchdog_path, code="artifixer3d_prior_watchdog_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_prior_cleanup_unreadable")
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("final_global_inventory", {}).get("live_resource_count") != 0
        or watchdog.get("final_global_inventory", {}).get("api_confirmed") is not True
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
    ):
        raise ValueError("artifixer3d_prior_terminal_closeout_invalid")
    return result, round(float(cost), 6)


def materialize_artifixer3d_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    prior_aura_authority_path: str | Path,
    prior_terminal_result_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one zero-retry authority chained through all previous scene spend."""

    bundle = validate_artifixer3d_bundle(bundle_receipt_path)
    prior_authority_path = Path(prior_aura_authority_path).expanduser().resolve()
    prior_authority = _validate_prior_authority_chain(prior_authority_path)
    terminal_path = Path(prior_terminal_result_path).expanduser().resolve()
    _, latest_cost = _validate_prior_terminal_result(
        terminal_path, prior_authority=prior_authority
    )
    prior_spend = round(float(prior_authority["prior_goal_spend_usd"]) + latest_cost, 6)
    aggregate_cap = float(bundle["aggregate_goal_spend_cap_usd"])
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or blueprint_commit != bundle["blueprint_source_identity"]["commit"]
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
        or prior_spend + hard_cap_usd > aggregate_cap
    ):
        raise ValueError("artifixer3d_paid_attempt_authority_configuration_invalid")
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_artifixer3d_object_free_exact_support_candidate_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "automatic_paid_retry_authorized": False,
        "maximum_automatic_retries": 0,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "zero_retry": True,
        "bundle_receipt": _record(Path(bundle["receipt_path"])),
        "bundle_receipt_digest": bundle["receipt_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "candidate_input_receipt_digest": bundle["candidate_input_receipt_digest"],
        "use_attestation_digest": bundle["use_attestation_digest"],
        "parent_execution_authority_digest": bundle[
            "parent_execution_authority_digest"
        ],
        "blueprint_commit": blueprint_commit,
        "blueprint_tree": bundle["blueprint_source_identity"]["tree"],
        "container_image": bundle["container_image"],
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": prior_spend,
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "prior_aura_authority": _record(prior_authority_path),
        "prior_aura_authority_digest": prior_authority["authorization_digest"],
        "prior_terminal_result": _record(terminal_path),
        "prior_terminal_cost_usd": latest_cost,
        "external_active_instance_allowlist": bundle["allowed_active_instance_ids"],
        "forbidden_external_instance_ids": bundle[
            "forbidden_external_instance_ids"
        ],
        "private_derived_upload_only": True,
        "raw_interiorgs_upload_authorized": False,
        "raw_dataset_bytes_upload_authorized": False,
        "provider_training_authorized": False,
        "publication_authorized": False,
        "commercial_use_authorized": False,
        "exact_mask_only_edits_required": True,
        "source_object_restoration_authorized": False,
        "generated_output_is_physical_evidence": False,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("artifixer3d_paid_attempt_authority_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, authority)
    validate_artifixer3d_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=bundle["allowed_active_instance_ids"],
    )
    return authority


def validate_artifixer3d_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int],
) -> dict[str, Any]:
    """Fail closed if any bundle, spend-chain, cap, or rights byte drifts."""

    value = dict(authority)
    if (
        value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or value.get("authorization_digest")
        != canonical_digest(value, digest_field="authorization_digest")
        or value.get("paid_compute_authorized") is not True
        or value.get("automatic_paid_retry_authorized") is not False
        or value.get("maximum_automatic_retries") != 0
        or value.get("maximum_paid_attempts") != 1
        or value.get("maximum_provider_allocations") != 1
        or value.get("zero_retry") is not True
        or value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or value.get("bundle_receipt_digest") != prepared_bundle.get("receipt_digest")
        or value.get("manifest_digest") != prepared_bundle.get("manifest_digest")
        or value.get("runtime_request_digest")
        != prepared_bundle.get("runtime_request_digest")
        or value.get("blueprint_commit")
        != prepared_bundle.get("blueprint_source_identity", {}).get("commit")
        or value.get("blueprint_tree")
        != prepared_bundle.get("blueprint_source_identity", {}).get("tree")
        or value.get("container_image") != prepared_bundle.get("container_image")
        or value.get("hard_attempt_spend_cap_usd") != hard_cap_usd
        or value.get("maximum_hourly_rate_usd") != max_hourly_rate_usd
        or value.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds
        or value.get("external_active_instance_allowlist")
        != sorted(set(int(item) for item in allowed_active_instance_ids))
        or set(value.get("external_active_instance_allowlist", []))
        & set(value.get("forbidden_external_instance_ids", []))
        or value.get("forbidden_external_instance_ids")
        != prepared_bundle.get("forbidden_external_instance_ids")
        or value.get("private_derived_upload_only") is not True
        or value.get("raw_interiorgs_upload_authorized") is not False
        or value.get("raw_dataset_bytes_upload_authorized") is not False
        or value.get("provider_training_authorized") is not False
        or value.get("commercial_use_authorized") is not False
        or value.get("exact_mask_only_edits_required") is not True
        or value.get("source_object_restoration_authorized") is not False
        or value.get("generated_output_is_physical_evidence") is not False
    ):
        raise ValueError("artifixer3d_paid_attempt_authority_invalid")
    bundle_receipt_path = _bound(
        value.get("bundle_receipt"), code="artifixer3d_authority_bundle_unbound"
    )
    if bundle_receipt_path != Path(str(prepared_bundle["receipt_path"])).resolve():
        raise ValueError("artifixer3d_authority_bundle_mismatch")
    prior_path = _bound(
        value.get("prior_aura_authority"),
        code="artifixer3d_authority_prior_chain_unbound",
    )
    prior = _validate_prior_authority_chain(prior_path)
    if prior.get("authorization_digest") != value.get("prior_aura_authority_digest"):
        raise ValueError("artifixer3d_authority_prior_chain_mismatch")
    terminal_path = _bound(
        value.get("prior_terminal_result"),
        code="artifixer3d_authority_terminal_unbound",
    )
    _, terminal_cost = _validate_prior_terminal_result(
        terminal_path, prior_authority=prior
    )
    prior_spend = round(float(prior["prior_goal_spend_usd"]) + terminal_cost, 6)
    if (
        terminal_cost != value.get("prior_terminal_cost_usd")
        or prior_spend != value.get("aggregate_goal_spend_before_attempt_usd")
        or value.get("aggregate_goal_spend_cap_usd")
        != prepared_bundle.get("aggregate_goal_spend_cap_usd")
        or prior_spend + hard_cap_usd > float(value["aggregate_goal_spend_cap_usd"])
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
    ):
        raise ValueError("artifixer3d_paid_attempt_authority_budget_invalid")
    return value


def consume_artifixer3d_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if digest != canonical_digest(dict(authority), digest_field="authorization_digest"):
        return {"status": "blocked", "blockers": ["artifixer3d_authority_identity_invalid"]}
    identity = digest.removeprefix("sha256:")
    if blueprint_commit != authority.get("blueprint_commit") or len(identity) != 64:
        return {"status": "blocked", "blockers": ["artifixer3d_authority_identity_invalid"]}
    try:
        AUTHORIZATION_CONSUMPTION_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = AUTHORIZATION_CONSUMPTION_ROOT.stat()
        if (
            AUTHORIZATION_CONSUMPTION_ROOT.is_symlink()
            or root_stat.st_uid != os.getuid()
            or root_stat.st_mode & 0o077
        ):
            raise PermissionError
        destination = AUTHORIZATION_CONSUMPTION_ROOT / f"artifixer3d-{identity}.json"
        record = {
            "schema_version": "artifixer3d_paid_attempt_consumption.v1",
            "authorization_digest": digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "blueprint_commit": blueprint_commit,
            "maximum_provider_allocations": 1,
            "consumed_at": utc_now_iso(),
        }
        raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = AUTHORIZATION_CONSUMPTION_ROOT / f".{identity}.{os.getpid()}.tmp"
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
        return {"status": "blocked", "blockers": ["artifixer3d_paid_attempt_authority_consumed"]}
    except (OSError, PermissionError):
        return {"status": "blocked", "blockers": ["artifixer3d_authority_consumption_write_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


@contextmanager
def _authority_environment():
    previous = {name: os.environ.get(name) for name in (*_MUTATION_ENV, _RETRY_ENV)}
    for name in _MUTATION_ENV:
        os.environ[name] = "true"
    os.environ[_RETRY_ENV] = "0"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract_output(path: Path | None, destination: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if path is None or not path.is_file() or path.is_symlink():
        return {}, ["artifixer3d_provider_output_missing"]
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(path) as archive:
            total = 0
            for info in archive.infolist():
                member = Path(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or mode == 0o120000
                    or info.file_size > 2_000_000_000
                ):
                    raise ValueError("artifixer3d_provider_output_member_invalid")
                total += info.file_size
                if total > 4_000_000_000:
                    raise ValueError("artifixer3d_provider_output_too_large")
            archive.extractall(destination)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {}, [f"artifixer3d_provider_output_extract_failed:{type(exc).__name__}"]
    result_path = destination / "public_scene_artifixer3d_runtime_result.json"
    if not result_path.is_file():
        blockers.append("artifixer3d_runtime_result_missing")
        return {}, blockers
    return _read(result_path, code="artifixer3d_runtime_result_unreadable"), blockers


def _local_runtime_path(root: Path, provider_path: Any, *, code: str) -> Path:
    value = str(provider_path or "").replace("\\", "/")
    marker = "/runtime_output/"
    if marker not in value:
        raise ValueError(code)
    relative = Path(value.split(marker, 1)[1])
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(code)
    return path


def _materialize_raw_result(
    *,
    execution: Mapping[str, Any],
    execution_root: Path,
    bundle: Mapping[str, Any],
    closeout: Mapping[str, Any],
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in execution.get("tasks") or []:
        task_id = str(task.get("task_id") or "")
        if task_id not in bundle["task_ids"] or task_id in seen:
            raise ValueError("artifixer3d_runtime_task_invalid")
        seen.add(task_id)
        frames: list[dict[str, Any]] = []
        for row in task.get("final_candidate_frames") or []:
            path = _local_runtime_path(
                execution_root,
                row.get("path"),
                code="artifixer3d_runtime_frame_unbound",
            )
            if (
                path.stat().st_size != row.get("size_bytes")
                or _sha256(path) != row.get("sha256")
                or row.get("outside_support_changed_pixels") != 0
            ):
                raise ValueError("artifixer3d_runtime_frame_invalid")
            frames.append(
                {
                    "frame_index": row.get("frame_index"),
                    "camera_id": row.get("camera_id"),
                    "repair_pixel_count": row.get("repair_pixel_count"),
                    "outside_support_changed_pixels": 0,
                    **_record(path),
                }
            )
        checkpoint = _local_runtime_path(
            execution_root,
            task.get("artifixer3d_checkpoint", {}).get("path"),
            code="artifixer3d_runtime_checkpoint_unbound",
        )
        if (
            checkpoint.stat().st_size
            != task.get("artifixer3d_checkpoint", {}).get("size_bytes")
            or _sha256(checkpoint)
            != task.get("artifixer3d_checkpoint", {}).get("sha256")
            or len(frames) != bundle["task_camera_counts"][task_id]
            or task.get("outside_support_changed_pixels_total") != 0
        ):
            raise ValueError("artifixer3d_runtime_task_outputs_invalid")
        tasks.append(
            {
                "task_id": task_id,
                "final_candidate_frames": frames,
                "artifixer3d_checkpoint": _record(checkpoint),
                "outside_support_changed_pixels_total": 0,
                "semantic_object_free_review_passed": False,
                "multiview_consistency_review_passed": False,
            }
        )
    if seen != set(bundle["task_ids"]):
        raise ValueError("artifixer3d_runtime_task_coverage_invalid")
    raw: dict[str, Any] = {
        "schema_version": RAW_RESULT_SCHEMA_VERSION,
        "status": "candidate_frames_ready_for_external_visual_and_multiview_review",
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "replacement_object_count": bundle["replacement_object_count"],
        "tasks": tasks,
        "source_object_restoration_permitted": False,
        "outside_exact_support_changed_pixels_total": 0,
        "appearance_repair_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "provider_closeout": dict(closeout),
        "result_digest": "",
    }
    raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
    return raw


def run_artifixer3d_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    machine_avoidlist_path: str | Path | None = None,
    paid_attempt_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute exactly once through the canonical paid provider adapter."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if (
        not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
    ):
        raise ValueError("artifixer3d_budget_invalid")
    result_path = job / "public_scene_artifixer3d_vast_result.json"
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "prepared_bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("artifixer3d_paid_resource_admission_grant_missing")
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class="vast_provider_adapter",
        require_allocation_binding=True,
    )
    if paid_attempt_authority is None:
        raise ValueError("artifixer3d_paid_attempt_authority_missing")
    authority = validate_artifixer3d_paid_attempt_authority(
        paid_attempt_authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=bundle["allowed_active_instance_ids"],
    )
    consumption = consume_artifixer3d_paid_attempt_authority_once(
        authority, blueprint_commit=str(authority["blueprint_commit"])
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
    bundle_path = Path(str(bundle["bundle_path"])).resolve()
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=hard_ttl_seconds + 1800,
    )
    if staging.get("status") != "completed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "authorization_consumption": consumption,
            "blockers": staging.get("blockers")
            or ["artifixer3d_object_store_staging_blocked"],
        }
        write_json(result_path, result)
        return result
    allowed = tuple(int(value) for value in bundle["allowed_active_instance_ids"])
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
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "authorization_consumption": consumption,
            "independent_watchdog": handoff,
            "blockers": ["artifixer3d_independent_watchdog_not_armed"],
        }
        write_json(result_path, result)
        return result
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
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=hard_ttl_seconds // 60,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=bundle["container_image"],
                isaac_image=bundle["container_image"],
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=192,
                min_gpu_ram_mb=MIN_GPU_RAM_MB,
                min_compute_cap=MIN_COMPUTE_CAP,
                poll_interval_seconds=15,
                startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=3600,
                session_budget_ledger_path=job / "artifixer3d_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("A100",),
                prefer_isaac_rt=False,
                gpu_selection_policy=GPU_SELECTION_POLICY,
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed,
                vast_launch_lock_file=job.parent / "artifixer3d_paid_launch.lock",
                instance_label_prefix=INSTANCE_LABEL_PREFIX,
                started_instance_id_path=handle.started_instance_id_path,
                forward_hf_token=True,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"artifixer3d_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path) if teardown_path.is_file() else {}
    instance_ids = [
        value
        for value in teardown.get("vast_instance_ids") or []
        if isinstance(value, int) and value > 0
    ]
    watchdog = close_independent_vast_watchdog(
        job_dir=job,
        handle=handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    execution_root = job / "immutable_execution"
    execution, blockers = _extract_output(output_zip, execution_root)
    adapter_path = provider_run / "vast_provider_adapter_result.json"
    final_path = provider_run / "vast_final_validation.json"
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    cleanup_path = staging_dir / "wam_provider_object_store_cleanup.json"
    closeout = {
        "adapter_result": _record(adapter_path) if adapter_path.is_file() else None,
        "teardown_manifest": _record(teardown_path) if teardown_path.is_file() else None,
        "final_validation": _record(final_path) if final_path.is_file() else None,
        "watchdog_receipt": _record(watchdog_path) if watchdog_path.is_file() else None,
        "object_store_cleanup": _record(cleanup_path) if cleanup_path.is_file() else None,
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_zero_confirmed": watchdog.get("status") == "provider_terminal",
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
    }
    if adapter.get("status") != "completed":
        blockers.append("artifixer3d_provider_adapter_not_completed")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("artifixer3d_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        blockers.append("artifixer3d_watchdog_not_terminal")
    if (
        execution.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or execution.get("status")
        != "candidate_completed_requires_visual_and_multiview_review"
        or execution.get("model_loaded") is not True
        or execution.get("artifixer_direct_inference_executed") is not True
        or execution.get("artifixer3d_distillation_executed") is not True
        or execution.get("artifixer3d_plus_inference_executed") is not True
        or execution.get("provider_zero_required_after_return") is not True
        or execution.get("source_object_restoration_permitted") is not False
        or execution.get("outside_exact_support_changed_pixels_permitted") != 0
    ):
        blockers.append("artifixer3d_runtime_not_completed")
    raw_path: Path | None = None
    if not blockers:
        try:
            raw = _materialize_raw_result(
                execution=execution,
                execution_root=execution_root,
                bundle=bundle,
                closeout=closeout,
            )
            raw_path = job / "public_scene_artifixer3d_raw_result.json"
            write_json(raw_path, raw)
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(
                f"artifixer3d_raw_result_materialization_failed:{type(exc).__name__}"
            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "execution_result_path": str(
            execution_root / "public_scene_artifixer3d_runtime_result.json"
        ),
        "raw_result_path": str(raw_path) if raw_path else None,
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "final_validation_path": str(final_path),
        "watchdog_receipt_path": str(watchdog_path),
        "object_store_cleanup_path": str(cleanup_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "authorization_consumption": consumption,
        "independent_watchdog": watchdog,
        "appearance_repair_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    write_json(result_path, result)
    return result


__all__ = [
    "MAX_HARD_CAP_USD",
    "MAX_TTL_SECONDS",
    "PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "consume_artifixer3d_paid_attempt_authority_once",
    "materialize_artifixer3d_paid_attempt_authority",
    "run_artifixer3d_vast",
    "validate_artifixer3d_bundle",
    "validate_artifixer3d_paid_attempt_authority",
]
