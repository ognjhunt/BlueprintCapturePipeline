"""Run one rights-admitted, exact-mask Aura residual packet on Vast.

This adapter is intentionally narrower than the historical Aura InteriorGS
lane: it accepts a sealed 1--5 replacement packet, uploads only its
private-derived ZIP, arms an independent watchdog before create, and makes a
raw Aura result available only after provider-zero and object-store absence
are independently retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .task_evaluation_artifact_manifest import (
    seal_lane_terminal_artifacts,
    seal_unallocated_provider_teardown,
)
from .common import ensure_dir, utc_now_iso, write_json, redacted_failure_detail
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .public_scene_aura_exact_residual_bundle import DEFAULT_IMAGE, SCHEMA_VERSION as BUNDLE_SCHEMA
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


from .spend_authority_consumption_root import consumption_root

PROBE_KIND = "adp-aurafusion360-exact-residual"
PROVIDER_BUNDLE_KIND = "adp_aura_exact_residual"
RESULT_SCHEMA_VERSION = "public_scene_aura_exact_residual_vast_run.v1"
RAW_RESULT_SCHEMA_VERSION = "public_scene_aura_exact_residual_raw_result.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "public_scene_aura_exact_residual_paid_attempt_authority.v1"
)
RUNTIME_ABSTENTION_SCHEMA_VERSION = "public_scene_aura_exact_residual_runtime_abstention.v1"
CAMPAIGN_ABSTENTION_SCHEMA_VERSION = (
    "public_scene_aura_exact_residual_provider_runtime_campaign_abstention.v1"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/aura-exact-residual"
MAX_TTL_SECONDS = 14_400
MIN_TTL_SECONDS = 7_200
MAX_HARD_CAP_USD = 12.0
MIN_RASTERIZER_COMPUTE_CAP = 890
GPU_SELECTION_POLICY = {
    "policy_id": "aura_exact_residual_observed_cuda_control",
    "allowed_gpu_keywords": ("L40S", "RTX 4090"),
    "denied_gpu_keywords": (),
    "reason": (
        "released Aura author controls previously reached their entrypoint on "
        "both L40S and RTX 4090; no task input or scene claim depends on GPU class"
    ),
}
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")
_RETRY_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
AUTHORIZATION_CONSUMPTION_ROOT = Path.home() / ".blueprint-spend-authority" / "consumed"
CORRECTED_ATTEMPT_PURPOSE = "manual_corrected_aura_exact_residual_execution"
SCIENTIFIC_SUCCESSOR_PURPOSE = "manual_successor_aura_exact_residual_execution"
ADDITIONAL_TERMINAL_SPEND_SCHEMAS = frozenset(
    {"adp009d_retained_scene_gpu_render_vast_run.v1"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("aura_exact_residual_receipt_unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("aura_exact_residual_receipt_unreadable")
    return value


def _bound(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _bound_json(record: Any, *, code: str) -> tuple[Path, dict[str, Any]]:
    """Open a file-backed authority dependency; digest-shaped paths never suffice."""

    path = _bound(record, code=code)
    return path, _read(path)


def _record_existing_file(value: str | Path, *, code: str) -> dict[str, Any]:
    """Create a receipt from a local file instead of trusting caller digests."""

    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(code)
    return _record(path)


def _is_known_correctable_aura_runtime_failure(runtime: Mapping[str, Any]) -> bool:
    """Accept only the sealed, code-correctable failures for a manual reissue."""

    blockers = runtime.get("blockers")
    if not isinstance(blockers, list):
        return False
    observed = {str(item) for item in blockers}
    return (
        "aura_exact_residual_runtime_wonderworld_bytes_changed" in observed
        or (
            "aura_exact_residual_runtime_exception:ValueError" in observed
            and "aura_exact_residual_runtime_native_point_cloud_missing" in observed
        )
        or (
            "aura_exact_residual_runtime_exception:FileNotFoundError" in observed
            and (
                "[Errno 2] No such file or directory: "
                "'/workspace/adp_aura_exact_residual_provider_bundle/runtime_output/logs/"
                "train_shared_retained_scene.log'"
            )
            in observed
        )
    )


def _authority_attempt_mode(authority: Mapping[str, Any]) -> str | None:
    """Return the explicit manual-attempt mode without inferring intent."""

    scientific_successor = authority.get("scientific_input_changed_after_terminal_attempt")
    if (
        authority.get("purpose") == CORRECTED_ATTEMPT_PURPOSE
        and authority.get("manual_corrected_reissue_after_terminal_attempt") is True
        and authority.get("manual_successor_after_terminal_attempt") in (None, False)
        and scientific_successor in (None, False)
    ):
        return "corrected_reissue"
    if (
        authority.get("purpose") == SCIENTIFIC_SUCCESSOR_PURPOSE
        and authority.get("manual_corrected_reissue_after_terminal_attempt") is False
        and authority.get("manual_successor_after_terminal_attempt") is True
        and scientific_successor is True
    ):
        return "scientific_successor"
    return None


def _bound_additional_terminal_spend_receipts(
    records: Any,
) -> tuple[float, frozenset[str]]:
    """Re-open independent terminal spend receipts and reject duplicates."""

    if records is None:
        records = []
    if isinstance(records, (str, bytes)) or not isinstance(records, list):
        raise ValueError("additional_terminal_spend_receipts_invalid")
    total = 0.0
    digests: set[str] = set()
    for record in records:
        _, receipt = _bound_json(record, code="additional_terminal_spend_receipt_unbound")
        receipt_digest = receipt.get("receipt_digest")
        cost = receipt.get("estimated_cost_usd")
        if (
            receipt.get("schema_version") not in ADDITIONAL_TERMINAL_SPEND_SCHEMAS
            or receipt.get("status") not in {"blocked", "completed"}
            or receipt.get("continuing_spend_from_this_run") is not False
            or receipt.get("all_staged_objects_absent") is not True
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0
            or not isinstance(receipt_digest, str)
            or receipt_digest != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt_digest in digests
        ):
            raise ValueError("additional_terminal_spend_receipt_invalid")
        digests.add(receipt_digest)
        total += float(cost)
    return round(total, 6), frozenset(digests)


def _terminal_attempt_evidence_valid(
    *,
    authority: Mapping[str, Any],
    previous: Mapping[str, Any],
    runtime: Mapping[str, Any],
    expected_new_preflight_digest: str | None = None,
) -> bool:
    """Validate either a corrected failure or a completed-input successor."""

    mode = _authority_attempt_mode(authority)
    previous_preflight = previous.get("preflight_digest")
    common = (
        previous.get("schema_version") == RESULT_SCHEMA_VERSION
        and previous.get("retry_cap") == 0
        and previous.get("continuing_spend_from_this_run") is False
        and previous.get("all_staged_objects_absent") is True
        and previous.get("bundle_sha256") == authority.get("previous_bundle_sha256")
        and runtime.get("schema_version")
        == "public_scene_aura_exact_residual_runtime_result.v1"
    )
    if not common:
        return False
    if mode == "corrected_reissue":
        return (
            previous.get("status") == "blocked"
            and previous.get("raw_result_path") is None
            and (
                expected_new_preflight_digest is None
                or previous_preflight == expected_new_preflight_digest
            )
            and runtime.get("status") == "blocked"
            and runtime.get("aura_inpainting_executed") is False
            and _is_known_correctable_aura_runtime_failure(runtime)
            and authority.get("previous_raw_result") is None
        )
    if mode == "scientific_successor":
        try:
            raw_path = _bound(
                authority.get("previous_raw_result"),
                code="previous_raw_result_unbound",
            )
        except ValueError:
            return False
        return (
            previous.get("status") == "completed"
            and isinstance(previous.get("raw_result_path"), str)
            and raw_path == Path(previous["raw_result_path"]).expanduser().resolve()
            and authority.get("previous_preflight_digest") == previous_preflight
            and isinstance(previous_preflight, str)
            and (
                expected_new_preflight_digest is None
                or previous_preflight != expected_new_preflight_digest
            )
            and runtime.get("status") == "completed"
            and runtime.get("aura_inpainting_executed") is True
            and not runtime.get("blockers")
        )
    return False


def _bound_prior_manual_authority(
    record: Any, *, ancestor_paths: frozenset[Path] = frozenset()
) -> tuple[dict[str, Any], float, frozenset[str]]:
    """Re-open a prior manual authority and its complete, acyclic spend chain."""

    authority_path, authority = _bound_json(
        record, code="prior_manual_corrected_attempt_authority_unbound"
    )
    if authority_path in ancestor_paths:
        raise ValueError("prior_manual_corrected_attempt_authority_cycle")
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority.get("authority_kind") != "explicit_user_direction_in_current_goal"
        or _authority_attempt_mode(authority) is None
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("maximum_paid_attempts") != 1
        or authority.get("zero_retry") is not True
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
    ):
        raise ValueError("prior_manual_corrected_attempt_authority_invalid")
    _, previous = _bound_json(
        authority.get("previous_terminal_execution_result"),
        code="prior_manual_corrected_attempt_terminal_unbound",
    )
    _, runtime = _bound_json(
        authority.get("previous_runtime_result"),
        code="prior_manual_corrected_attempt_runtime_unbound",
    )
    _, teardown = _bound_json(
        authority.get("previous_teardown"),
        code="prior_manual_corrected_attempt_teardown_unbound",
    )
    _, watchdog = _bound_json(
        authority.get("previous_watchdog"),
        code="prior_manual_corrected_attempt_watchdog_unbound",
    )
    _, cleanup = _bound_json(
        authority.get("previous_object_store_cleanup"),
        code="prior_manual_corrected_attempt_cleanup_unbound",
    )
    _, campaign = _bound_json(
        authority.get("prior_provider_runtime_campaign"),
        code="prior_manual_corrected_attempt_campaign_unbound",
    )
    previous_cost = previous.get("estimated_cost_usd")
    campaign_cost = campaign.get("total_estimated_cost_usd")
    if (
        not _terminal_attempt_evidence_valid(
            authority=authority,
            previous=previous,
            runtime=runtime,
            expected_new_preflight_digest=authority.get("preflight_digest"),
        )
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or (watchdog.get("final_inventory") or {}).get("live_resource_count") != 0
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or campaign.get("schema_version") != CAMPAIGN_ABSTENTION_SCHEMA_VERSION
        or campaign.get("provider_zero_confirmed_all") is not True
        or campaign.get("aura_inpainting_executed") is not False
        or isinstance(previous_cost, bool)
        or not isinstance(previous_cost, (int, float))
        or isinstance(campaign_cost, bool)
        or not isinstance(campaign_cost, (int, float))
    ):
        raise ValueError("prior_manual_corrected_attempt_evidence_invalid")
    additional_cost, additional_digests = _bound_additional_terminal_spend_receipts(
        authority.get("additional_terminal_spend_receipts")
    )
    parent_record = authority.get("prior_manual_corrected_attempt_authority")
    if parent_record is None:
        prior_total = float(campaign_cost)
        ancestor_additional_digests: frozenset[str] = frozenset()
    else:
        parent, prior_total, ancestor_additional_digests = _bound_prior_manual_authority(
            parent_record, ancestor_paths=ancestor_paths | {authority_path}
        )
        if authority.get("prior_provider_runtime_campaign") != parent.get(
            "prior_provider_runtime_campaign"
        ):
            raise ValueError("prior_manual_corrected_attempt_campaign_mismatch")
    if additional_digests & ancestor_additional_digests:
        raise ValueError("prior_manual_corrected_attempt_spend_duplicate")
    total = round(float(previous_cost) + prior_total + additional_cost, 6)
    if authority.get("prior_goal_spend_usd") != total:
        raise ValueError("prior_manual_corrected_attempt_spend_invalid")
    return authority, total, additional_digests | ancestor_additional_digests


def validate_aura_exact_residual_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: tuple[int, ...] | list[int] = (),
) -> dict[str, Any]:
    """Validate one human-directed, corrected Aura execution before staging.

    A terminal failed run is not an automatic retry.  The authority must bind
    the prior zero-closed receipts and the exact corrected bundle that will be
    launched; it is then consumed atomically immediately before a provider
    mutation.
    """

    value = dict(authority)
    errors: list[str] = []
    try:
        expected_allowed = tuple(sorted({int(item) for item in allowed_active_instance_ids}))
    except (TypeError, ValueError):
        expected_allowed = ()
        errors.append("allowed_active_instance_ids_invalid")
    observed_allowed = value.get("external_active_instance_allowlist")
    if (
        isinstance(observed_allowed, (str, bytes))
        or not isinstance(observed_allowed, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in observed_allowed)
        or len(set(observed_allowed)) != len(observed_allowed)
        or tuple(sorted(observed_allowed)) != expected_allowed
    ):
        errors.append("external_active_instance_allowlist_mismatch")
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION:
        errors.append("schema_invalid")
    if value.get("authority_kind") != "explicit_user_direction_in_current_goal":
        errors.append("authority_kind_invalid")
    if not isinstance(value.get("authority_reference"), str) or not value["authority_reference"].strip():
        errors.append("authority_reference_invalid")
    if not isinstance(value.get("authorized_by"), str) or not value["authorized_by"].strip():
        errors.append("authorized_by_invalid")
    if not isinstance(value.get("authorized_on"), str) or not value["authorized_on"].strip():
        errors.append("authorized_on_invalid")
    attempt_mode = _authority_attempt_mode(value)
    if attempt_mode is None:
        errors.append("purpose_invalid")
    if value.get("provider") != "vast" or value.get("paid_compute_authorized") is not True:
        errors.append("provider_or_paid_authority_invalid")
    if value.get("automatic_paid_retry_authorized") is not False or value.get("maximum_automatic_retries") != 0:
        errors.append("automatic_retry_contract_invalid")
    if value.get("maximum_paid_attempts") != 1 or value.get("zero_retry") is not True:
        errors.append("single_attempt_contract_invalid")
    if value.get("parent_execution_authority_digest") != prepared_bundle.get("execution_authority_digest"):
        errors.append("parent_execution_authority_digest_mismatch")
    if value.get("bundle_receipt_sha256") != prepared_bundle.get("receipt_sha256"):
        errors.append("bundle_receipt_sha256_mismatch")
    if value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        errors.append("bundle_sha256_mismatch")
    if value.get("preflight_digest") != prepared_bundle.get("preflight_digest"):
        errors.append("preflight_digest_mismatch")
    if value.get("hard_attempt_spend_cap_usd") != hard_cap_usd:
        errors.append("hard_attempt_spend_cap_mismatch")
    if value.get("maximum_hourly_rate_usd") != max_hourly_rate_usd:
        errors.append("maximum_hourly_rate_mismatch")
    if value.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds:
        errors.append("maximum_single_resource_ttl_mismatch")
    if value.get("private_derived_upload_only") is not True or value.get("raw_interiorgs_upload_authorized") is not False:
        errors.append("upload_scope_invalid")
    if value.get("provider_training_authorized") is not False or value.get("publication_authorized") is not False:
        errors.append("retention_or_training_scope_invalid")
    if value.get("exact_mask_only_edits_required") is not True:
        errors.append("exact_mask_scope_invalid")
    if value.get("authorization_digest") != canonical_digest(value, digest_field="authorization_digest"):
        errors.append("authorization_digest_invalid")

    prior_cost = 0.0
    historic_additional_digests: frozenset[str] = frozenset()
    prior_manual_authority = value.get("prior_manual_corrected_attempt_authority")
    if prior_manual_authority is not None:
        try:
            historic_authority, historic_cost, historic_additional_digests = _bound_prior_manual_authority(
                prior_manual_authority
            )
        except ValueError:
            errors.append("prior_manual_corrected_attempt_authority_invalid")
        else:
            if value.get("prior_provider_runtime_campaign") != historic_authority.get(
                "prior_provider_runtime_campaign"
            ):
                errors.append("prior_manual_corrected_attempt_campaign_mismatch")
            prior_cost = historic_cost
    try:
        _, previous = _bound_json(
            value.get("previous_terminal_execution_result"),
            code="previous_terminal_execution_result_unbound",
        )
        _, prior_runtime = _bound_json(
            value.get("previous_runtime_result"), code="previous_runtime_result_unbound"
        )
        _, prior_teardown = _bound_json(
            value.get("previous_teardown"), code="previous_teardown_unbound"
        )
        _, prior_watchdog = _bound_json(
            value.get("previous_watchdog"), code="previous_watchdog_unbound"
        )
        _, prior_cleanup = _bound_json(
            value.get("previous_object_store_cleanup"), code="previous_object_store_cleanup_unbound"
        )
    except ValueError:
        errors.append("previous_terminal_evidence_unbound")
    else:
        prior_cost_value = previous.get("estimated_cost_usd")
        if isinstance(prior_cost_value, bool) or not isinstance(prior_cost_value, (int, float)):
            errors.append("previous_terminal_cost_invalid")
        else:
            prior_cost += float(prior_cost_value)
        if not _terminal_attempt_evidence_valid(
            authority=value,
            previous=previous,
            runtime=prior_runtime,
            expected_new_preflight_digest=prepared_bundle.get("preflight_digest"),
        ):
            errors.append("previous_terminal_execution_invalid")
        if (
            prior_teardown.get("status") != "completed"
            or prior_teardown.get("continuing_spend_from_this_run") is not False
            or prior_watchdog.get("status") != "provider_terminal"
            or prior_watchdog.get("provider_absence_confirmed") is not True
            or (prior_watchdog.get("final_inventory") or {}).get("live_resource_count") != 0
            or prior_cleanup.get("status") != "completed"
            or prior_cleanup.get("all_objects_absent") is not True
        ):
            errors.append("previous_terminal_zero_close_invalid")

    try:
        additional_cost, additional_digests = _bound_additional_terminal_spend_receipts(
            value.get("additional_terminal_spend_receipts")
        )
    except ValueError:
        errors.append("additional_terminal_spend_receipts_invalid")
    else:
        if additional_digests & historic_additional_digests:
            errors.append("additional_terminal_spend_receipt_duplicate")
        prior_cost += additional_cost

    try:
        _, campaign = _bound_json(
            value.get("prior_provider_runtime_campaign"),
            code="prior_provider_runtime_campaign_unbound",
        )
    except ValueError:
        errors.append("prior_provider_runtime_campaign_unbound")
    else:
        campaign_cost = campaign.get("total_estimated_cost_usd")
        expected_campaign_preflight = (
            value.get("previous_preflight_digest")
            if attempt_mode == "scientific_successor"
            else prepared_bundle.get("preflight_digest")
        )
        if (
            campaign.get("schema_version") != CAMPAIGN_ABSTENTION_SCHEMA_VERSION
            or campaign.get("preflight_digest") != expected_campaign_preflight
            or campaign.get("provider_zero_confirmed_all") is not True
            or campaign.get("aura_inpainting_executed") is not False
            or isinstance(campaign_cost, bool)
            or not isinstance(campaign_cost, (int, float))
        ):
            errors.append("prior_provider_runtime_campaign_invalid")
        elif prior_manual_authority is None:
            prior_cost += float(campaign_cost)
    # Provider ledgers and campaign receipts are sealed to six decimal places.
    # Reconcile at that same evidence precision, not binary-float accident.
    prior_cost = round(prior_cost, 6)
    if value.get("prior_goal_spend_usd") != prior_cost:
        errors.append("prior_goal_spend_mismatch")
    aggregate = prior_cost + hard_cap_usd
    total_cap = value.get("aggregate_goal_spend_cap_usd")
    if (
        isinstance(total_cap, bool)
        or not isinstance(total_cap, (int, float))
        or total_cap < aggregate
        or total_cap > MAX_HARD_CAP_USD
    ):
        errors.append("aggregate_goal_spend_cap_invalid")
    if errors:
        raise ValueError(
            "aura_exact_residual_paid_attempt_authority_invalid:" + ",".join(sorted(set(errors)))
        )
    return value


def consume_aura_exact_residual_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    """Atomically consume the one corrected execution before staging provider bytes."""

    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["aura_exact_residual_authority_identity_invalid"]}
    identity = digest.removeprefix("sha256:")
    try:
        consumption_root().mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = consumption_root().stat()
        if (
            consumption_root().is_symlink()
            or root_stat.st_uid != os.getuid()
            or root_stat.st_mode & 0o077
        ):
            raise PermissionError
        destination = consumption_root() / f"aura-exact-residual-{identity}.json"
        record = {
            "schema_version": "aura_exact_residual_paid_attempt_consumption.v1",
            "authorization_digest": digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "blueprint_commit": blueprint_commit,
            "maximum_provider_allocations": 1,
            "consumed_at": utc_now_iso(),
        }
        raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = consumption_root() / f".{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            directory_descriptor = os.open(consumption_root(), os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["aura_exact_residual_paid_attempt_authority_consumed"]}
    except (OSError, PermissionError):
        return {"status": "blocked", "blockers": ["aura_exact_residual_authority_consumption_write_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def materialize_aura_exact_residual_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    previous_terminal_execution_result_path: str | Path,
    previous_runtime_result_path: str | Path,
    previous_teardown_path: str | Path,
    previous_watchdog_path: str | Path,
    previous_object_store_cleanup_path: str | Path,
    prior_provider_runtime_campaign_path: str | Path,
    prior_manual_corrected_attempt_authority_path: str | Path | None = None,
    scientific_input_changed_after_terminal_attempt: bool = False,
    additional_terminal_spend_receipt_paths: Sequence[str | Path] = (),
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    corrective_blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Materialize one manual corrected or scientific-successor authority.

    This performs no provider mutation.  It makes the user's current explicit
    continuation authority reviewable and validates the generated authority
    against the exact previous terminal run before it can reach an allocator.
    """

    bundle = validate_aura_exact_residual_bundle(bundle_receipt_path)
    record_paths = {
        "previous_terminal_execution_result": previous_terminal_execution_result_path,
        "previous_runtime_result": previous_runtime_result_path,
        "previous_teardown": previous_teardown_path,
        "previous_watchdog": previous_watchdog_path,
        "previous_object_store_cleanup": previous_object_store_cleanup_path,
        "prior_provider_runtime_campaign": prior_provider_runtime_campaign_path,
    }
    records = {
        key: _record_existing_file(value, code=f"{key}_unbound")
        for key, value in record_paths.items()
    }
    prior_manual_authority_record: dict[str, Any] | None = None
    prior_manual_cost = 0.0
    if prior_manual_corrected_attempt_authority_path is not None:
        prior_manual_authority_record = _record_existing_file(
            prior_manual_corrected_attempt_authority_path,
            code="prior_manual_corrected_attempt_authority_unbound",
        )
        _, prior_manual_cost, prior_additional_digests = _bound_prior_manual_authority(
            prior_manual_authority_record
        )
    else:
        prior_additional_digests = frozenset()
    previous = _read(Path(previous_terminal_execution_result_path).expanduser().resolve())
    campaign = _read(Path(prior_provider_runtime_campaign_path).expanduser().resolve())
    additional_records = [
        _record_existing_file(path, code="additional_terminal_spend_receipt_unbound")
        for path in additional_terminal_spend_receipt_paths
    ]
    additional_cost, additional_digests = _bound_additional_terminal_spend_receipts(
        additional_records
    )
    if additional_digests & prior_additional_digests:
        raise ValueError("aura_exact_residual_paid_attempt_authority_spend_duplicate")
    previous_cost = previous.get("estimated_cost_usd")
    campaign_cost = campaign.get("total_estimated_cost_usd")
    if (
        isinstance(previous_cost, bool)
        or not isinstance(previous_cost, (int, float))
        or isinstance(campaign_cost, bool)
        or not isinstance(campaign_cost, (int, float))
        or not isinstance(authorization_reference, str)
        or not authorization_reference.strip()
        or not isinstance(authorized_by, str)
        or not authorized_by.strip()
        or not isinstance(authorized_on, str)
        or not authorized_on.strip()
        or len(corrective_blueprint_commit) != 40
        or any(character not in "0123456789abcdef" for character in corrective_blueprint_commit)
    ):
        raise ValueError("aura_exact_residual_paid_attempt_authority_materialization_invalid")
    parent = _read(Path(bundle["execution_authority_path"]).expanduser().resolve())
    aggregate_cap = (parent.get("paid_compute") or {}).get("hard_total_spend_cap_usd")
    if (
        isinstance(aggregate_cap, bool)
        or not isinstance(aggregate_cap, (int, float))
        or aggregate_cap > MAX_HARD_CAP_USD
    ):
        raise ValueError("aura_exact_residual_paid_attempt_authority_parent_cap_invalid")
    previous_raw_result_record: dict[str, Any] | None = None
    if scientific_input_changed_after_terminal_attempt:
        raw_result_path = previous.get("raw_result_path")
        if not isinstance(raw_result_path, str) or not raw_result_path:
            raise ValueError(
                "aura_exact_residual_paid_attempt_authority_previous_raw_result_invalid"
            )
        previous_raw_result_record = _record_existing_file(
            raw_result_path, code="previous_raw_result_unbound"
        )
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference,
        "authorized_by": authorized_by,
        "authorized_on": authorized_on,
        "purpose": (
            SCIENTIFIC_SUCCESSOR_PURPOSE
            if scientific_input_changed_after_terminal_attempt
            else CORRECTED_ATTEMPT_PURPOSE
        ),
        "provider": "vast",
        "paid_compute_authorized": True,
        "manual_corrected_reissue_after_terminal_attempt": not scientific_input_changed_after_terminal_attempt,
        "manual_successor_after_terminal_attempt": scientific_input_changed_after_terminal_attempt,
        "scientific_input_changed_after_terminal_attempt": scientific_input_changed_after_terminal_attempt,
        "automatic_paid_retry_authorized": False,
        "maximum_automatic_retries": 0,
        "maximum_paid_attempts": 1,
        "zero_retry": True,
        "parent_execution_authority_digest": bundle["execution_authority_digest"],
        "bundle_receipt_sha256": bundle["receipt_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "preflight_digest": bundle["preflight_digest"],
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "external_active_instance_allowlist": list(bundle["allowed_active_instance_ids"]),
        "private_derived_upload_only": True,
        "raw_interiorgs_upload_authorized": False,
        "provider_training_authorized": False,
        "publication_authorized": False,
        "exact_mask_only_edits_required": True,
        "previous_bundle_sha256": previous.get("bundle_sha256"),
        "previous_preflight_digest": previous.get("preflight_digest"),
        "previous_raw_result": previous_raw_result_record,
        **records,
        "prior_manual_corrected_attempt_authority": prior_manual_authority_record,
        "additional_terminal_spend_receipts": additional_records,
        "prior_goal_spend_usd": round(
            prior_manual_cost
            + float(previous_cost)
            + additional_cost
            + (0.0 if prior_manual_authority_record else float(campaign_cost)),
            6,
        ),
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "corrective_blueprint_commit": corrective_blueprint_commit,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    validate_aura_exact_residual_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=list(bundle["allowed_active_instance_ids"]),
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("aura_exact_residual_paid_attempt_authority_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, authority)
    return authority


def _zip_member_bytes(
    archive: zipfile.ZipFile, *, record: Any, root: str, code: str
) -> bytes:
    """Read one digest-bound ZIP member without trusting a caller path."""

    if not isinstance(record, Mapping):
        raise ValueError(code)
    relative = str(record.get("relative_path") or "")
    member = f"{root}/{relative}" if relative else ""
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or member not in archive.namelist()
    ):
        raise ValueError(code)
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError(code) from exc
    if (
        len(payload) != record.get("size_bytes")
        or "sha256:" + hashlib.sha256(payload).hexdigest() != record.get("sha256")
    ):
        raise ValueError(code)
    return payload


def _zip_member_json(
    archive: zipfile.ZipFile, *, record: Any, root: str, code: str
) -> dict[str, Any]:
    try:
        value = json.loads(
            _zip_member_bytes(archive, record=record, root=root, code=code).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _authority(
    preflight: Mapping[str, Any], *, backend_value: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    if (
        backend_value.get("schema_version") != "public_scene_released_code_inpainting_admission.v1"
        or backend_value.get("status") != "rights_admitted_for_private_derived_inpainting"
        or backend_value.get("backend_id") != "aurafusion360_exact_residual_multiview"
        or backend_value.get("strict_exact_residual_masks_required") is not True
        or backend_value.get("outside_mask_pixel_delta_required") != 0
        or backend_value.get("private_derived_upload_policy", {}).get("private_derived_upload")
        is not True
        or backend_value.get("private_derived_upload_policy", {}).get("raw_dataset_bytes_upload")
        is not False
        or backend_value.get("private_derived_upload_policy", {}).get("provider_training")
        is not False
        or backend_value.get("receipt_digest")
        != canonical_digest(backend_value, digest_field="receipt_digest")
    ):
        raise ValueError("aura_exact_residual_backend_admission_invalid")
    authority_record = backend_value.get("execution_authority")
    authority_path = _bound(authority_record, code="aura_exact_residual_execution_authority_unbound")
    authority = _read(authority_path)
    paid = authority.get("paid_compute")
    if (
        authority.get("schema_version") != "third_scene_dual_task_execution_authority.v1"
        or authority.get("authority_kind") != "explicit_user_direction_in_current_goal"
        or authority.get("publisher_scene_id") != "840920"
        or authority.get("private_rights_admitted_scene_derived_uploads_authorized") is not True
        or authority.get("raw_interiorgs_upload_authorized") is not False
        or authority.get("training_authorized") is not False
        or authority.get("retention") != "bounded_to_goal_then_provider_zero"
        or not isinstance(paid, Mapping)
        or paid.get("provider") != "vast"
        or paid.get("hard_total_spend_cap_usd") != MAX_HARD_CAP_USD
        or paid.get("zero_retry") is not True
        or paid.get("provider_zero_required_for_lane") is not True
        or paid.get("external_instance_allowlist") != [47373597]
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or authority_record.get("authority_digest") != authority.get("authority_digest")
    ):
        raise ValueError("aura_exact_residual_execution_authority_invalid")
    return authority, authority_path


def validate_aura_exact_residual_bundle(receipt_path: str | Path) -> dict[str, Any]:
    """Load file-backed receipts; never accept digest-shaped caller assertions."""

    receipt_file = Path(receipt_path).expanduser().resolve()
    receipt = _read(receipt_file)
    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    if (
        receipt.get("schema_version") != BUNDLE_SCHEMA
        or receipt.get("status") != "ready"
        or not bundle_path.is_file()
        or bundle_path.is_symlink()
        or _sha256(bundle_path) != receipt.get("bundle_sha256")
        or receipt.get("container_image") != DEFAULT_IMAGE
        or not 1 <= receipt.get("replacement_object_count", 0) <= 5
        or receipt.get("shared_camera_count", 0) < receipt.get("replacement_object_count", 0)
        or receipt.get("task_count", 0) < 1
        or receipt.get("private_derived_upload_only") is not True
        or receipt.get("raw_interiorgs_bytes_included") is not False
        or receipt.get("stock_inpaint360gs_code_or_author_data_included") is not False
        or receipt.get("automatic_paid_retry_allowed") is not False
        or receipt.get("provider_zero_required_after_return") is not True
    ):
        raise ValueError("aura_exact_residual_bundle_receipt_invalid")
    rehearsal = receipt.get("exact_bundle_entrypoint_rehearsal")
    if (
        not isinstance(rehearsal, Mapping)
        or rehearsal.get("status") != "passed"
        or rehearsal.get("provider_mutations_performed") != 0
        or rehearsal.get("gpu_runtime_started") is not False
    ):
        raise ValueError("aura_exact_residual_bundle_rehearsal_invalid")
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("aura_exact_residual_bundle_integrity_invalid")
            runtime_request_path = _bound(
                receipt.get("runtime_request"), code="aura_exact_residual_runtime_request_unbound"
            )
            request_bytes = archive.read("provider_runtime/aura_exact_residual_runtime_request.json")
            if request_bytes != runtime_request_path.read_bytes():
                raise ValueError("aura_exact_residual_runtime_request_bundle_mismatch")
            request = json.loads(request_bytes.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("aura_exact_residual_runtime_request_unbound")
            preflight = _zip_member_json(
                archive,
                record=request.get("preflight"),
                root="provider_runtime",
                code="aura_exact_residual_preflight_unbound",
            )
            backend = _zip_member_json(
                archive,
                record=request.get("backend_admission"),
                root="provider_runtime",
                code="aura_exact_residual_backend_receipt_unbound",
            )
            _zip_member_bytes(
                archive,
                record=request.get("shared_retained_scene"),
                root="provider_runtime",
                code="aura_exact_residual_shared_ply_unbound",
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("aura_exact_residual_bundle_integrity_invalid") from exc
    if (
        request.get("schema_version") != "public_scene_aura_exact_residual_runtime_request.v1"
        or request.get("request_digest") != canonical_digest(request, digest_field="request_digest")
        or request.get("private_derived_upload_only") is not True
        or request.get("raw_dataset_bytes_included") is not False
        or request.get("provider_training_authorized") is not False
        or request.get("automatic_paid_retry_allowed") is not False
        or request.get("provider_zero_required_after_return") is not True
        or request.get("learned_policy_outcomes_accessed") is not False
    ):
        raise ValueError("aura_exact_residual_runtime_request_invalid")
    if (
        preflight.get("schema_version") != "public_scene_aura_exact_residual_preflight.v1"
        or preflight.get("status") != "prepared_no_upload_no_execution"
        or preflight.get("preflight_digest") != receipt.get("preflight_digest")
        or preflight.get("preflight_digest") != canonical_digest(preflight, digest_field="preflight_digest")
        or preflight.get("replacement_object_count") != receipt.get("replacement_object_count")
        or preflight.get("execution", {}).get("provider_mutations_performed") != 0
        or preflight.get("execution", {}).get("aura_inpainting_executed") is not False
        or preflight.get("backend_admission", {}).get("sha256")
        != request.get("backend_admission", {}).get("sha256")
        or preflight.get("backend_admission", {}).get("size_bytes")
        != request.get("backend_admission", {}).get("size_bytes")
        or preflight.get("shared_retained_scene", {}).get("sha256")
        != request.get("shared_retained_scene", {}).get("sha256")
        or preflight.get("shared_retained_scene", {}).get("retained_gaussian_count")
        != request.get("shared_retained_scene", {}).get("retained_gaussian_count")
        or len(request.get("camera_inputs") or []) != receipt.get("shared_camera_count")
        or len(request.get("task_plans") or []) != receipt.get("task_count")
    ):
        raise ValueError("aura_exact_residual_preflight_invalid")
    authority, authority_path = _authority(preflight, backend_value=backend)
    return {
        "receipt_path": str(receipt_file),
        "receipt_sha256": _sha256(receipt_file),
        "bundle_path": str(bundle_path),
        "bundle_sha256": receipt["bundle_sha256"],
        "container_image": DEFAULT_IMAGE,
        "preflight_path": None,
        "preflight_digest": preflight["preflight_digest"],
        "replacement_object_count": receipt["replacement_object_count"],
        "shared_camera_count": receipt["shared_camera_count"],
        "task_count": receipt["task_count"],
        "execution_authority_path": str(authority_path),
        "execution_authority_digest": authority["authority_digest"],
        "allowed_active_instance_ids": list(authority["paid_compute"]["external_instance_allowlist"]),
    }


@contextmanager
def _authority_environment():
    names = (*_MUTATION_ENV, _RETRY_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_RETRY_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract(path: Path, destination: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["aura_exact_residual_provider_output_zip_missing"]
    ensure_dir(destination)
    root = destination.resolve()
    blockers: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("aura_exact_residual_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(root)
    except (OSError, zipfile.BadZipFile):
        blockers.append("aura_exact_residual_provider_output_zip_invalid")
    result_path = root / "public_scene_aura_exact_residual_runtime_result.json"
    if not result_path.is_file() or result_path.is_symlink():
        blockers.append("aura_exact_residual_runtime_result_missing")
        return {}, blockers
    try:
        return _read(result_path), blockers
    except ValueError:
        return {}, [*blockers, "aura_exact_residual_runtime_result_unreadable"]


def _absolute_runtime_rows(execution: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def local(record: Any) -> Path:
        if not isinstance(record, Mapping):
            raise ValueError("aura_exact_residual_runtime_output_record_invalid")
        relative = str(record.get("relative_path") or "")
        path = (root / relative).resolve()
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or root not in path.parents
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError("aura_exact_residual_runtime_output_record_invalid")
        return path

    frames: list[dict[str, Any]] = []
    for row in execution.get("frames") or []:
        if not isinstance(row, Mapping):
            raise ValueError("aura_exact_residual_runtime_frame_invalid")
        path = local(row.get("native_aura_frame"))
        frames.append({
            "task_id": row.get("task_id"), "camera_id": row.get("camera_id"),
            "native_aura_frame": _record(path),
            "native_aura_point_cloud_sha256": row.get("native_aura_point_cloud_sha256"),
        })
    outputs: list[dict[str, Any]] = []
    for row in execution.get("task_outputs") or []:
        if not isinstance(row, Mapping):
            raise ValueError("aura_exact_residual_runtime_task_output_invalid")
        path = local(row.get("native_aura_point_cloud"))
        outputs.append({
            "task_id": row.get("task_id"), "native_aura_point_cloud": _record(path),
            "native_aura_point_cloud_sha256": row.get("native_aura_point_cloud_sha256"),
            "native_aura_gaussian_count": row.get("native_aura_gaussian_count"),
            "native_aura_representation": row.get("native_aura_representation"),
            "render_camera_ids": row.get("render_camera_ids"),
        })
    if not frames or not outputs:
        raise ValueError("aura_exact_residual_runtime_outputs_missing")
    return frames, outputs


def run_aura_exact_residual_vast(
    *, job_dir: str | Path, paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool, prepared_bundle: Mapping[str, Any], max_hourly_rate_usd: float = 1.5,
    hard_cap_usd: float = 6.0, hard_ttl_seconds: int = MAX_TTL_SECONDS,
    machine_avoidlist_path: str | Path | None = None,
    paid_attempt_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute once. The only paid path is through the canonical allocator."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if (
        not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
    ):
        raise ValueError("aura_exact_residual_budget_invalid")
    if not execute:
        result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(),
                  "status": "dry_run_ready", "prepared_bundle": bundle,
                  "provider_mutations_performed": 0, "retry_cap": 0, "blockers": []}
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("aura_exact_residual_paid_resource_admission_grant_missing")
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class="vast_provider_adapter",
        require_allocation_binding=True,
    )
    if paid_attempt_authority is None:
        raise ValueError("aura_exact_residual_paid_attempt_authority_missing")
    validated_attempt_authority = validate_aura_exact_residual_paid_attempt_authority(
        paid_attempt_authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=list(bundle["allowed_active_instance_ids"]),
    )
    authorization_consumption = consume_aura_exact_residual_paid_attempt_authority_once(
        validated_attempt_authority,
        blueprint_commit=str(validated_attempt_authority.get("corrective_blueprint_commit") or ""),
    )
    if authorization_consumption.get("status") != "consumed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": authorization_consumption,
            "blockers": list(authorization_consumption.get("blockers") or []),
        }
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    bundle_path = Path(str(bundle["bundle_path"])).resolve()
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir, bundle_path=bundle_path, key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=hard_ttl_seconds + 1800,
    )
    if staging.get("status") != "completed":
        result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(), "status": "blocked",
                  "provider_mutations_performed": 0, "retry_cap": 0,
                  "blockers": staging.get("blockers") or ["aura_exact_residual_object_store_staging_blocked"]}
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    allowed = tuple(int(value) for value in bundle["allowed_active_instance_ids"])
    handoff, handle = arm_independent_vast_watchdog(
        job_dir=job, max_live_minutes=hard_ttl_seconds // 60, generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed, pod_name_prefix="blueprint-adp-aura-exact-residual-",
    )
    if handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(), "status": "blocked",
                  "provider_mutations_performed": 0, "retry_cap": 0,
                  "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                  "independent_watchdog": handoff,
                  "blockers": ["aura_exact_residual_independent_watchdog_not_armed"]}
        write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
        return result
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run, mode="live-startup-probe", allow_vast_api_call=True,
                allow_instance_launch=True, max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd, hard_cap_usd=hard_cap_usd,
                max_live_minutes=hard_ttl_seconds // 60, session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=bundle["container_image"], isaac_image=bundle["container_image"],
                ngc_image_login_mode="never", provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip, enable_isaac_smoke=False,
                enable_blueprint_bundle=True, provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct", allow_cold_isaac_image_pull=False, disk_gb=192,
                min_gpu_ram_mb=24_000, min_compute_cap=MIN_RASTERIZER_COMPUTE_CAP,
                poll_interval_seconds=15, startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "aura_exact_residual_vast_session_budget.json",
                verify_staging_urls=True, require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("L40S", "RTX 4090"), prefer_isaac_rt=False,
                gpu_selection_policy=GPU_SELECTION_POLICY, machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed,
                vast_launch_lock_file=job.parent / "aura_exact_residual_paid_launch.lock",
                instance_label_prefix="blueprint-adp-aura-exact-residual-",
                started_instance_id_path=handle.started_instance_id_path, forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {"status": "blocked", "blockers": [f"aura_exact_residual_adapter_failed:{redacted_failure_detail(exc)}"],
                   "raw_secret_values_recorded": False}
        # The adapter may never have been entered -- resolving a secret or a
        # staged URL raises before it. Record the absence of any allocation so
        # the run can close; the sealer declines whenever the evidence does not
        # support that claim.
        seal_unallocated_provider_teardown(
            provider_run, reason="aura_exact_residual_adapter_failed"
        )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path) if teardown_path.is_file() else {}
    instance_ids = [value for value in teardown.get("vast_instance_ids") or [] if isinstance(value, int) and value > 0]
    watchdog = close_independent_vast_watchdog(
        job_dir=job, handle=handle, instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=not instance_ids and adapter.get("provider_create_attempted") is not True,
    )
    execution_root = job / "immutable_execution"
    execution, blockers = _extract(output_zip, execution_root)
    adapter_path = provider_run / "vast_provider_adapter_result.json"
    final_path = provider_run / "vast_final_validation.json"
    # Bind the watchdog's terminal provider-inventory observation, not the
    # owner-to-watchdog cancellation request.  The compositor requires its
    # independent exact-id and global-zero evidence.
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    closeout_adapter = {
        "schema_version": "public_scene_aura_exact_residual_adapter_closeout.v1",
        "source_adapter_result": _record(adapter_path) if adapter_path.is_file() else None,
        "api_call_performed": adapter.get("api_call_performed"),
        "provider_create_attempted": adapter.get("provider_create_attempted"),
        "final_validation_status": adapter.get("final_validation_status"),
        "continuing_spend_from_this_run": adapter.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_ttl_seconds": hard_ttl_seconds,
    }
    closeout_adapter_path = job / "aura_exact_residual_adapter_closeout.json"
    write_json(closeout_adapter_path, closeout_adapter)
    if adapter.get("status") != "completed":
        blockers.append("aura_exact_residual_provider_adapter_not_completed")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("aura_exact_residual_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        blockers.append("aura_exact_residual_watchdog_not_terminal")
    if (
        execution.get("schema_version") != "public_scene_aura_exact_residual_runtime_result.v1"
        or execution.get("status") != "completed"
    ):
        blockers.append("aura_exact_residual_runtime_not_completed")
    if (
        execution.get("aura_inpainting_executed") is not True
        or execution.get("provider_mutations_performed") != 0
    ):
        blockers.append("aura_exact_residual_runtime_claim_invalid")
    raw_path: Path | None = None
    if not blockers:
        try:
            frames, task_outputs = _absolute_runtime_rows(execution, execution_root)
            raw: dict[str, Any] = {
                "schema_version": RAW_RESULT_SCHEMA_VERSION, "status": "aura_native_residual_frames_rendered",
                "preflight_digest": bundle["preflight_digest"], "aura_inpainting_executed": True,
                "provider_mutations_performed": 1, "learned_policy_outcomes_accessed": False,
                "provider_closeout": {"adapter_result": _record(closeout_adapter_path),
                    "teardown_manifest": _record(teardown_path), "final_validation": _record(final_path),
                    "watchdog_receipt": _record(watchdog_path)},
                "task_outputs": task_outputs, "frames": frames, "result_digest": "",
            }
            raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
            raw_path = job / "public_scene_aura_exact_residual_raw_result.json"
            write_json(raw_path, raw)
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(f"aura_exact_residual_raw_result_materialization_failed:{redacted_failure_detail(exc)}")
    result = {"schema_version": RESULT_SCHEMA_VERSION, "generated_at": utc_now_iso(),
              "status": "completed" if not blockers else "blocked", "bundle_sha256": bundle["bundle_sha256"],
              "preflight_digest": bundle["preflight_digest"], "execution_result_path": str(execution_root / "public_scene_aura_exact_residual_runtime_result.json"),
              "raw_result_path": str(raw_path) if raw_path else None,
              "adapter_result_path": str(adapter_path), "teardown_manifest_path": str(teardown_path),
              "final_validation_path": str(final_path), "watchdog_receipt_path": str(watchdog_path),
              "estimated_cost_usd": adapter.get("estimated_cost_usd"), "hard_cap_usd": hard_cap_usd,
              "hard_ttl_seconds": hard_ttl_seconds, "retry_cap": 0,
              "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
              "all_staged_objects_absent": cleanup.get("all_objects_absent"),
              "authorization_consumption": authorization_consumption,
              "independent_watchdog": watchdog, "blockers": sorted(set(str(item) for item in blockers if str(item))),
              "raw_secret_values_recorded": False}
    # Seal the two terminal artifacts every production launch profile asks
    # this result for. Without them the run ends
    # `allocator_terminal_artifact_missing:` whatever happened on the provider.
    result = seal_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane="public_scene_aura_exact_residual",
        binding={
            "bundle_sha256": bundle.get("bundle_sha256")
            if isinstance(bundle, Mapping)
            else None,
            "provider": "vast",
        },
    )
    write_json(job / "public_scene_aura_exact_residual_vast_result.json", result)
    return result


def materialize_aura_exact_residual_runtime_abstention(
    *,
    execution_result_path: str | Path,
    paid_admission_path: str | Path,
    bundle_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal a pre-entrypoint provider null without treating it as Aura execution.

    This is intentionally narrower than a general run failure.  It applies only
    when a rights-admitted, exact-mask packet created one provider instance but
    the provider failed before the sealed Aura bundle or entrypoint could run,
    and when both the owner and independent watchdog prove resource zero.
    """

    result_path = Path(execution_result_path).expanduser().resolve()
    admission_path = Path(paid_admission_path).expanduser().resolve()
    bundle = validate_aura_exact_residual_bundle(bundle_receipt_path)
    result = _read(result_path)
    admission = _read(admission_path)
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "blocked"
        or result.get("retry_cap") != 0
        or result.get("raw_result_path") is not None
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or admission.get("schema_version") != "paid_lane_admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("retry_cap") != 0
        or admission.get("private_derived_upload_only") is not True
        or admission.get("raw_interiorgs_upload_authorized") is not False
        or admission.get("provider_training_authorized") is not False
        or admission.get("exact_mask_only_edits_required") is not True
        or (admission.get("allocation_binding") or {}).get("bundle_receipt_sha256")
        != bundle["receipt_sha256"]
        or result.get("bundle_sha256") != bundle["bundle_sha256"]
        or result.get("preflight_digest") != bundle["preflight_digest"]
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_result_invalid")

    root = result_path.parent.resolve()

    def result_member(field: str, relative: str, code: str) -> Path:
        path = Path(str(result.get(field) or "")).expanduser().resolve()
        expected = (root / relative).resolve()
        if path != expected or not path.is_file() or path.is_symlink():
            raise ValueError(code)
        return path

    adapter_path = result_member(
        "adapter_result_path",
        "vast_provider_run/vast_provider_adapter_result.json",
        "aura_exact_residual_runtime_abstention_adapter_missing",
    )
    teardown_path = result_member(
        "teardown_manifest_path",
        "vast_provider_run/vast_teardown_manifest.json",
        "aura_exact_residual_runtime_abstention_teardown_missing",
    )
    watchdog_path = result_member(
        "watchdog_receipt_path",
        f"independent_vast_watchdog/{WATCHDOG_EVIDENCE_NAME}",
        "aura_exact_residual_runtime_abstention_watchdog_missing",
    )
    adapter = _read(adapter_path)
    teardown = _read(teardown_path)
    watchdog = _read(watchdog_path)
    classification = adapter.get("provider_attempt_classification")
    instance_ids = adapter.get("vast_instance_ids")
    if (
        adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("status") != "failed"
        or adapter.get("reason") != "vast_probe_failed"
        or adapter.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or adapter.get("api_call_performed") is not True
        or adapter.get("provider_create_attempted") is not True
        or adapter.get("continuing_spend_from_this_run") is not False
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or not isinstance(instance_ids[0], int)
        or instance_ids[0] <= 0
        or not isinstance(classification, Mapping)
        or classification.get("classification") != "pre_execution_provider_null"
        or classification.get("provider_bundle_started") is not False
        or classification.get("provider_entrypoint_started") is not False
        or classification.get("provider_output_returned") is not False
        or classification.get("automatic_requeue_authorized") is not False
        or classification.get("automatic_requeue_executed") is not False
        or classification.get("maximum_automatic_requeues") != 0
        or "vast_heartbeat_instance_exited" not in (adapter.get("blockers") or [])
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("vast_instance_ids") != instance_ids
        or watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or (watchdog.get("recorded_vast_instance") or {}).get("instance_id")
        != str(instance_ids[0])
        or (watchdog.get("recorded_vast_instance_teardown") or {}).get("status") != "absent"
        or (watchdog.get("final_inventory") or {}).get("live_resource_count") != 0
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_provider_evidence_invalid")

    cleanup_path = root / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    # A deliberate later attempt may reuse the prior attempt's immutable
    # avoidlist so its failed machine cannot be reselected. Bind exactly the
    # path the adapter recorded, while keeping it inside this shared-scene
    # execution parent and refusing an arbitrary caller-supplied path.
    avoidlist_path = Path(str(adapter.get("machine_avoidlist_path") or "")).expanduser().resolve()
    if not cleanup_path.is_file() or cleanup_path.is_symlink():
        raise ValueError("aura_exact_residual_runtime_abstention_object_store_cleanup_missing")
    if (
        avoidlist_path.name != "vast_machine_avoidlist.json"
        or (root not in avoidlist_path.parents and root.parent not in avoidlist_path.parents)
        or not avoidlist_path.is_file()
        or avoidlist_path.is_symlink()
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_machine_avoidlist_missing")
    cleanup = _read(cleanup_path)
    avoidlist = _read(avoidlist_path)
    budget_path = root / "vast_provider_run" / "vast_budget_ledger.json"
    artifacts = adapter.get("artifacts")
    if (
        not budget_path.is_file()
        or budget_path.is_symlink()
        or not isinstance(artifacts, Mapping)
        or Path(str(artifacts.get("vast_budget_ledger") or "")).expanduser().resolve()
        != budget_path.resolve()
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_budget_missing")
    budget = _read(budget_path)
    entries = avoidlist.get("entries")
    if (
        cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
        or avoidlist.get("status") != "completed"
        or not isinstance(entries, list)
        or not any(
            isinstance(entry, Mapping)
            and entry.get("instance_id") == instance_ids[0]
            and entry.get("reason") == "vast_startup_control_plane_did_not_reach_onstart_heartbeat"
            and entry.get("retry_policy")
            == "exclude_persistently_across_sibling_jobs_until_manual_review"
            for entry in entries
        )
        or budget.get("schema_version") != "vast_budget_ledger.v1"
        or budget.get("status") != "completed"
        or budget.get("continuing_spend_from_this_run") is not False
        or budget.get("vast_instance_ids") != instance_ids
        or isinstance(budget.get("estimated_cost_usd"), bool)
        or not isinstance(budget.get("estimated_cost_usd"), (int, float))
        or not math.isfinite(float(budget["estimated_cost_usd"]))
        or float(budget["estimated_cost_usd"]) < 0
        or budget.get("estimated_cost_usd") != adapter.get("estimated_cost_usd")
        or isinstance(result.get("hard_cap_usd"), bool)
        or not isinstance(result.get("hard_cap_usd"), (int, float))
        or float(budget["estimated_cost_usd"]) > float(result["hard_cap_usd"])
    ):
        raise ValueError("aura_exact_residual_runtime_abstention_closeout_invalid")

    receipt: dict[str, Any] = {
        "schema_version": RUNTIME_ABSTENTION_SCHEMA_VERSION,
        "status": "abstained_provider_runtime_before_aura_entrypoint",
        "bundle_sha256": result.get("bundle_sha256"),
        "preflight_digest": result.get("preflight_digest"),
        "replacement_object_count": bundle["replacement_object_count"],
        "shared_camera_count": bundle["shared_camera_count"],
        "task_count": bundle["task_count"],
        "bundle_receipt": _record(Path(bundle["receipt_path"])),
        "paid_admission": _record(admission_path),
        "execution_result": _record(result_path),
        "provider_adapter": _record(adapter_path),
        "teardown": _record(teardown_path),
        "independent_watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "machine_avoidlist": _record(avoidlist_path),
        "provider_budget_ledger": _record(budget_path),
        "provider_instance_id": instance_ids[0],
        "estimated_cost_usd": float(budget["estimated_cost_usd"]),
        "actual_live_runtime_seconds_observed_by_adapter": budget.get(
            "actual_live_runtime_seconds_observed_by_adapter"
        ),
        "aura_inpainting_executed": False,
        "provider_bundle_started": False,
        "provider_entrypoint_started": False,
        "provider_output_returned": False,
        "automatic_paid_retry_allowed": False,
        "automatic_paid_retry_executed": False,
        "provider_mutations_performed": 1,
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "smallest_missing_capability": (
            "rights_admitted_gpu_provider_runtime_that_reaches_the_sealed_Aura_exact_"
            "residual_container_entrypoint"
        ),
        "blockers": ["aura_exact_residual_provider_runtime_pre_entrypoint_null"],
        "claim_boundary": {
            "rights_admitted_backend_is_not_executed_backend": True,
            "inpainting_output_exists": False,
            "native_aura_frames_exist": False,
            "outside_mask_locality_measured": False,
            "multi_view_consistency_measured": False,
            "simready_or_policy_gate_unlocked": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    return receipt


def materialize_aura_exact_residual_provider_runtime_campaign_abstention(
    *,
    runtime_abstention_paths: list[str | Path],
    bundle_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal two distinct provider-null attempts without inventing Aura results.

    This is deliberately a campaign closeout, not a retry controller.  It can
    only describe exactly two independently zero-closed attempts against one
    sealed 1--5 replacement packet.  Any future provider mutation needs a new
    explicit authority path; this materializer performs no mutation itself.
    """

    if len(runtime_abstention_paths) != 2:
        raise ValueError("aura_exact_residual_campaign_requires_exactly_two_attempts")
    bundle = validate_aura_exact_residual_bundle(bundle_receipt_path)
    bundle_receipt = Path(bundle["receipt_path"]).resolve()
    rows: list[dict[str, Any]] = []
    seen_receipts: set[Path] = set()
    for value in runtime_abstention_paths:
        receipt_path = Path(value).expanduser().resolve()
        if (
            receipt_path in seen_receipts
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
        ):
            raise ValueError("aura_exact_residual_campaign_runtime_abstention_path_invalid")
        seen_receipts.add(receipt_path)
        abstention = _read(receipt_path)
        if (
            abstention.get("schema_version") != RUNTIME_ABSTENTION_SCHEMA_VERSION
            or abstention.get("status")
            != "abstained_provider_runtime_before_aura_entrypoint"
            or abstention.get("receipt_digest")
            != canonical_digest(abstention, digest_field="receipt_digest")
            or abstention.get("bundle_sha256") != bundle["bundle_sha256"]
            or abstention.get("preflight_digest") != bundle["preflight_digest"]
            or abstention.get("replacement_object_count")
            != bundle["replacement_object_count"]
            or abstention.get("shared_camera_count") != bundle["shared_camera_count"]
            or abstention.get("task_count") != bundle["task_count"]
            or abstention.get("aura_inpainting_executed") is not False
            or abstention.get("provider_bundle_started") is not False
            or abstention.get("provider_entrypoint_started") is not False
            or abstention.get("provider_output_returned") is not False
            or abstention.get("automatic_paid_retry_allowed") is not False
            or abstention.get("automatic_paid_retry_executed") is not False
            or abstention.get("continuing_spend_from_this_run") is not False
            or abstention.get("provider_zero_confirmed") is not True
        ):
            raise ValueError("aura_exact_residual_campaign_runtime_abstention_invalid")
        bound_bundle = _bound(
            abstention.get("bundle_receipt"),
            code="aura_exact_residual_campaign_bundle_receipt_unbound",
        )
        execution_result = _bound(
            abstention.get("execution_result"),
            code="aura_exact_residual_campaign_execution_result_unbound",
        )
        adapter_path = _bound(
            abstention.get("provider_adapter"),
            code="aura_exact_residual_campaign_adapter_unbound",
        )
        teardown_path = _bound(
            abstention.get("teardown"),
            code="aura_exact_residual_campaign_teardown_unbound",
        )
        watchdog_path = _bound(
            abstention.get("independent_watchdog"),
            code="aura_exact_residual_campaign_watchdog_unbound",
        )
        cleanup_path = _bound(
            abstention.get("object_store_cleanup"),
            code="aura_exact_residual_campaign_cleanup_unbound",
        )
        avoidlist_path = _bound(
            abstention.get("machine_avoidlist"),
            code="aura_exact_residual_campaign_avoidlist_unbound",
        )
        budget_path = _bound(
            abstention.get("provider_budget_ledger"),
            code="aura_exact_residual_campaign_budget_unbound",
        )
        root = execution_result.parent.resolve()
        if (
            bound_bundle != bundle_receipt
            or execution_result.name != "public_scene_aura_exact_residual_vast_result.json"
            or adapter_path != root / "vast_provider_run" / "vast_provider_adapter_result.json"
            or teardown_path != root / "vast_provider_run" / "vast_teardown_manifest.json"
            or watchdog_path
            != root / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
            or cleanup_path
            != root / "object_store_staging" / "wam_provider_object_store_cleanup.json"
            or budget_path != root / "vast_provider_run" / "vast_budget_ledger.json"
            or avoidlist_path.name != "vast_machine_avoidlist.json"
            # The second sealed attempt may deliberately inherit its sibling's
            # avoidlist.  It must still be within the same shared-scene parent,
            # never an arbitrary caller-selected file.
            or (
                root not in avoidlist_path.parents
                and root.parent not in avoidlist_path.parents
            )
        ):
            raise ValueError("aura_exact_residual_campaign_artifact_layout_invalid")
        adapter = _read(adapter_path)
        teardown = _read(teardown_path)
        watchdog = _read(watchdog_path)
        cleanup = _read(cleanup_path)
        budget = _read(budget_path)
        instance_id = abstention.get("provider_instance_id")
        session_attempts = (adapter.get("session_budget_summary") or {}).get("attempts")
        matching_attempts = [
            item
            for item in session_attempts or []
            if isinstance(item, Mapping) and item.get("vast_instance_ids") == [instance_id]
        ]
        if (
            not isinstance(instance_id, int)
            or instance_id <= 0
            or adapter.get("status") != "failed"
            or adapter.get("reason") != "vast_probe_failed"
            or adapter.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
            or (adapter.get("provider_attempt_classification") or {}).get("classification")
            != "pre_execution_provider_null"
            or adapter.get("vast_instance_ids") != [instance_id]
            or adapter.get("continuing_spend_from_this_run") is not False
            or teardown.get("status") != "completed"
            or teardown.get("continuing_spend_from_this_run") is not False
            or teardown.get("vast_instance_ids") != [instance_id]
            or watchdog.get("status") != "provider_terminal"
            or watchdog.get("provider_absence_confirmed") is not True
            or (watchdog.get("recorded_vast_instance") or {}).get("instance_id")
            != str(instance_id)
            or cleanup.get("all_objects_absent") is not True
            or budget.get("status") != "completed"
            or budget.get("continuing_spend_from_this_run") is not False
            or budget.get("vast_instance_ids") != [instance_id]
            or budget.get("estimated_cost_usd") != abstention.get("estimated_cost_usd")
            or not isinstance(matching_attempts, list)
            or len(matching_attempts) != 1
            or not isinstance(matching_attempts[0].get("machine_id"), int)
            or matching_attempts[0]["machine_id"] <= 0
        ):
            raise ValueError("aura_exact_residual_campaign_attempt_evidence_invalid")
        estimated_cost = budget.get("estimated_cost_usd")
        if (
            isinstance(estimated_cost, bool)
            or not isinstance(estimated_cost, (int, float))
            or not math.isfinite(float(estimated_cost))
            or float(estimated_cost) < 0
        ):
            raise ValueError("aura_exact_residual_campaign_attempt_cost_invalid")
        rows.append(
            {
                "runtime_abstention": _record(receipt_path),
                "provider_instance_id": instance_id,
                "machine_id": matching_attempts[0]["machine_id"],
                "estimated_cost_usd": float(estimated_cost),
                "actual_live_runtime_seconds_observed_by_adapter": budget.get(
                    "actual_live_runtime_seconds_observed_by_adapter"
                ),
                "provider_adapter": _record(adapter_path),
                "teardown": _record(teardown_path),
                "independent_watchdog": _record(watchdog_path),
                "object_store_cleanup": _record(cleanup_path),
                "machine_avoidlist": _record(avoidlist_path),
                "provider_budget_ledger": _record(budget_path),
            }
        )
    if (
        len({row["provider_instance_id"] for row in rows}) != 2
        or len({row["machine_id"] for row in rows}) != 2
    ):
        raise ValueError("aura_exact_residual_campaign_attempts_not_independent")
    receipt: dict[str, Any] = {
        "schema_version": CAMPAIGN_ABSTENTION_SCHEMA_VERSION,
        "status": "abstained_shared_provider_runtime_before_aura_entrypoint",
        "bundle_receipt": _record(bundle_receipt),
        "bundle_sha256": bundle["bundle_sha256"],
        "preflight_digest": bundle["preflight_digest"],
        "replacement_object_count": bundle["replacement_object_count"],
        "shared_camera_count": bundle["shared_camera_count"],
        "task_count": bundle["task_count"],
        "attempt_count": 2,
        "attempts": rows,
        "total_estimated_cost_usd": round(
            sum(row["estimated_cost_usd"] for row in rows), 6
        ),
        "aura_inpainting_executed": False,
        "native_aura_frames_exist": False,
        "automatic_paid_retry_executed": False,
        "provider_zero_confirmed_all": True,
        "smallest_missing_capability": (
            "rights_admitted_gpu_provider_runtime_that_reaches_the_sealed_Aura_exact_"
            "residual_container_entrypoint"
        ),
        "blockers": [
            "aura_exact_residual_provider_runtime_pre_entrypoint_null_on_two_distinct_vast_hosts"
        ],
        "claim_boundary": {
            "two_provider_nulls_are_not_aura_execution": True,
            "inpainting_output_exists": False,
            "outside_mask_locality_measured": False,
            "multi_view_consistency_measured": False,
            "simready_or_policy_gate_unlocked": False,
            "further_paid_mutation_requires_new_explicit_authority": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    return receipt


__all__ = [
    "DEFAULT_IMAGE",
    "CAMPAIGN_ABSTENTION_SCHEMA_VERSION",
    "MAX_HARD_CAP_USD",
    "MAX_TTL_SECONDS",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "RUNTIME_ABSTENTION_SCHEMA_VERSION",
    "consume_aura_exact_residual_paid_attempt_authority_once",
    "materialize_aura_exact_residual_paid_attempt_authority",
    "materialize_aura_exact_residual_provider_runtime_campaign_abstention",
    "materialize_aura_exact_residual_runtime_abstention",
    "run_aura_exact_residual_vast",
    "validate_aura_exact_residual_paid_attempt_authority",
    "validate_aura_exact_residual_bundle",
]
