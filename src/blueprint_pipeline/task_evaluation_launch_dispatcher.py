"""Immutable WebApp launch queue for canonical paid Task Evaluation Runs.

The WebApp may authorize and enqueue a Pipeline-owned launch profile.  It may
not choose a provider command, local path, secret, or allocator argument.  The
dispatcher resolves the exact profile from local Pipeline state and invokes
``paid_resource_allocator gpu-canary`` as the only paid mutation boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LAUNCH_REQUEST_SCHEMA_VERSION = "task_evaluation_launch_request.v1"
LAUNCH_PROFILE_SCHEMA_VERSION = "task_evaluation_launch_profile.v1"
LAUNCH_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_receipt.v1"
LAUNCH_PROFILE_CATALOG_SCHEMA_VERSION = "task_evaluation_launch_profile_catalog.v1"
CANONICAL_ALLOCATOR_ENTRYPOINT = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
EXECUTE_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE"
SECRET_PROFILE_ID_ENV = "BLUEPRINT_TASK_EVALUATION_SECRET_PROFILE_ID"
LAUNCH_RUN_ROOT_PLACEHOLDER = "{launch_run_root}"
_DIGEST_PREFIX = "sha256:"
_URI_SCHEMES = ("gs://", "s3://", "https://")
_AUTHORITY_URI_SCHEMES = (*_URI_SCHEMES, "firestore://")
_SECRET_KEY_FRAGMENTS = (
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_ALLOWED_RUNTIME_ENV_KEYS = frozenset(
    {
        "BLUEPRINT_ADP009D_CAMERA_RESOLUTION",
        "BLUEPRINT_ADP009D_CAMERA_WARMUP_FRAMES",
        "BLUEPRINT_ADP009D_CONTROLS",
        "BLUEPRINT_ADP009D_EPISODES",
        "BLUEPRINT_ADP009D_FIRST_RENDER_BUDGET_SECONDS",
        "BLUEPRINT_ADP009D_MAX_GAUSSIANS_TO_ACCUMULATE",
        "BLUEPRINT_ADP009D_PROVISION_TIMEOUT_SECONDS",
        "BLUEPRINT_ADP009D_STOP_AFTER_FRAMES",
    }
)
PUBLIC_PROFILE_DESCRIPTOR_FIELDS = (
    "profile_id",
    "profile_digest",
    "source_bundle",
    "evaluation_run_spec",
    "required_controls",
    "execution_admission",
    "claim_ceiling",
)


class TaskEvaluationLaunchError(ValueError):
    """Raised when a launch request or profile fails closed."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return _DIGEST_PREFIX + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        text.startswith(_DIGEST_PREFIX)
        and len(text) == len(_DIGEST_PREFIX) + 64
        and all(character in "0123456789abcdef" for character in text[len(_DIGEST_PREFIX) :])
    )


def _is_identifier(value: Any) -> bool:
    text = str(value or "")
    return (
        bool(text)
        and len(text) <= 192
        and all(character.isalnum() or character in "._-" for character in text)
    )


def _is_uri(value: Any, *, authority: bool = False) -> bool:
    text = str(value or "")
    return text.startswith(_AUTHORITY_URI_SCHEMES if authority else _URI_SCHEMES)


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
                # References name a canonical secret profile; raw secret-bearing
                # fields are never accepted from the website.
                if normalized != "secret_profile_id":
                    return True
            if _contains_secret_key(nested):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_secret_key(item) for item in value)
    return False


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _validate_reference(
    value: Any, *, field: str, blockers: list[str], authority: bool = False
) -> None:
    reference = _mapping(value)
    if not _is_uri(reference.get("uri"), authority=authority):
        blockers.append(f"{field}_uri_invalid")
    if not _is_digest(reference.get("digest")):
        blockers.append(f"{field}_digest_invalid")


def validate_launch_request(value: Mapping[str, Any]) -> list[str]:
    request = _mapping(value)
    blockers: list[str] = []
    if request.get("schema_version") != LAUNCH_REQUEST_SCHEMA_VERSION:
        blockers.append("launch_request_schema_version_mismatch")
    for field in ("launch_id", "run_id", "launch_profile_id", "idempotency_key"):
        if not _is_identifier(request.get(field)):
            blockers.append(f"{field}_invalid")
    if not _is_digest(request.get("launch_profile_digest")):
        blockers.append("launch_profile_digest_invalid")
    _validate_reference(request.get("source_bundle"), field="source_bundle", blockers=blockers)
    source_bundle = _mapping(request.get("source_bundle"))
    if not _is_identifier(source_bundle.get("bundle_id")):
        blockers.append("source_bundle_id_invalid")
    if source_bundle.get("source_kind") not in {
        "interiorgs_sage",
        "raw_v3_2_capture",
        "scaniverse_derived",
    }:
        blockers.append("source_bundle_kind_invalid")
    _validate_reference(
        request.get("evaluation_run_spec"), field="evaluation_run_spec", blockers=blockers
    )

    authorization = _mapping(request.get("authorization"))
    actor = _mapping(authorization.get("actor"))
    if actor.get("role") not in {"admin", "ops"} or not _is_identifier(actor.get("id")):
        blockers.append("launch_actor_invalid")
    if _parse_timestamp(authorization.get("authorized_at")) is None:
        blockers.append("launch_authorized_at_invalid")
    rights = _mapping(authorization.get("rights"))
    if rights.get("approved") is not True or not str(rights.get("scope") or "").strip():
        blockers.append("rights_authority_missing")
    _validate_reference(
        rights.get("evidence"), field="rights_authority_evidence", blockers=blockers, authority=True
    )
    spend = _mapping(authorization.get("spend"))
    max_spend = spend.get("max_spend_usd")
    if (
        spend.get("approved") is not True
        or spend.get("currency") != "USD"
        or not isinstance(max_spend, (int, float))
        or isinstance(max_spend, bool)
        or max_spend <= 0
    ):
        blockers.append("spend_authority_missing")
    expires_at = _parse_timestamp(spend.get("expires_at"))
    if expires_at is None:
        blockers.append("spend_authority_expiry_invalid")
    execution = _mapping(authorization.get("execution"))
    if execution.get("approved") is not True:
        blockers.append("execution_authority_missing")

    controls = _mapping(request.get("required_controls"))
    expected_controls = {
        "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "watchdog_required": True,
        "artifact_storage_required": True,
        "teardown_required": True,
        "provider_zero_required": True,
        "webapp_status_sync_required": True,
        "retry_cap": 0,
    }
    for field, expected in expected_controls.items():
        if controls.get(field) != expected:
            blockers.append(f"required_control_mismatch:{field}")
    if not _is_identifier(controls.get("secret_profile_id")):
        blockers.append("secret_profile_id_invalid")
    if request.get("claim_ceiling") not in {
        "development_only",
        "partner_run_pending_physical_join",
    }:
        blockers.append("launch_claim_ceiling_invalid")
    if _contains_secret_key(request):
        blockers.append("launch_request_secret_value_forbidden")
    if request.get("request_digest") != canonical_digest(request, digest_field="request_digest"):
        blockers.append("launch_request_digest_mismatch")
    return sorted(set(blockers))


def validate_launch_profile(value: Mapping[str, Any]) -> list[str]:
    profile = _mapping(value)
    blockers: list[str] = []
    if profile.get("schema_version") != LAUNCH_PROFILE_SCHEMA_VERSION:
        blockers.append("launch_profile_schema_version_mismatch")
    if not _is_identifier(profile.get("profile_id")):
        blockers.append("launch_profile_id_invalid")
    if profile.get("program_id") != "arm-decision-proof-v1":
        blockers.append("launch_profile_program_mismatch")
    _validate_reference(profile.get("source_bundle"), field="source_bundle", blockers=blockers)
    profile_source = _mapping(profile.get("source_bundle"))
    if not _is_identifier(profile_source.get("bundle_id")):
        blockers.append("launch_profile_source_bundle_id_invalid")
    if profile_source.get("source_kind") not in {
        "interiorgs_sage",
        "raw_v3_2_capture",
        "scaniverse_derived",
    }:
        blockers.append("launch_profile_source_kind_invalid")
    _validate_reference(
        profile.get("evaluation_run_spec"), field="evaluation_run_spec", blockers=blockers
    )
    immutable_inputs = profile.get("immutable_inputs")
    if not isinstance(immutable_inputs, list) or len(immutable_inputs) < 2:
        blockers.append("launch_profile_immutable_inputs_invalid")
    else:
        input_names: set[str] = set()
        for item in immutable_inputs:
            immutable_input = _mapping(item)
            name = str(immutable_input.get("name") or "")
            path_value = str(immutable_input.get("path") or "")
            if not _is_identifier(name) or name in input_names:
                blockers.append("launch_profile_immutable_input_name_invalid")
            input_names.add(name)
            if not path_value or not Path(path_value).expanduser().is_absolute():
                blockers.append(f"launch_profile_immutable_input_path_invalid:{name}")
            if not _is_digest(immutable_input.get("digest")):
                blockers.append(f"launch_profile_immutable_input_digest_invalid:{name}")
        if not {"source_bundle_manifest", "evaluation_run_spec"}.issubset(input_names):
            blockers.append("launch_profile_immutable_input_roles_missing")
    allocator = _mapping(profile.get("allocator"))
    if allocator.get("entrypoint") != CANONICAL_ALLOCATOR_ENTRYPOINT:
        blockers.append("launch_profile_allocator_entrypoint_invalid")
    if allocator.get("subcommand") != "gpu-canary":
        blockers.append("launch_profile_allocator_subcommand_invalid")
    argv = allocator.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
        blockers.append("launch_profile_allocator_argv_invalid")
    elif "--execute" in argv:
        blockers.append("launch_profile_execute_flag_forbidden")
    elif any(_has_unknown_placeholder(item) for item in argv):
        blockers.append("launch_profile_allocator_argv_placeholder_invalid")
    max_spend = allocator.get("max_spend_usd")
    if not isinstance(max_spend, (int, float)) or isinstance(max_spend, bool) or max_spend <= 0:
        blockers.append("launch_profile_max_spend_invalid")
    hard_ttl = allocator.get("hard_ttl_seconds")
    if not isinstance(hard_ttl, int) or isinstance(hard_ttl, bool) or hard_ttl <= 0:
        blockers.append("launch_profile_hard_ttl_invalid")
    if allocator.get("retry_cap") != 0:
        blockers.append("launch_profile_retry_cap_invalid")
    runtime_environment = profile.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        blockers.append("launch_profile_runtime_environment_invalid")
    else:
        for key, raw_value in runtime_environment.items():
            normalized_key = str(key)
            normalized_lower = normalized_key.lower()
            if normalized_key not in _ALLOWED_RUNTIME_ENV_KEYS or any(
                fragment in normalized_lower for fragment in _SECRET_KEY_FRAGMENTS
            ):
                blockers.append(f"launch_profile_runtime_environment_key_invalid:{normalized_key}")
            if (
                not isinstance(raw_value, str)
                or len(raw_value) > 1024
                or "\n" in raw_value
                or "\r" in raw_value
                or "\x00" in raw_value
            ):
                blockers.append(
                    f"launch_profile_runtime_environment_value_invalid:{normalized_key}"
                )
    execution_admission = _mapping(profile.get("execution_admission"))
    if not isinstance(profile.get("execution_admission"), Mapping):
        blockers.append("launch_profile_execution_admission_invalid")
    if not isinstance(execution_admission.get("live_enabled"), bool):
        blockers.append("launch_profile_live_enabled_invalid")
    _validate_reference(
        execution_admission.get("readiness_receipt"),
        field="launch_profile_readiness_receipt",
        blockers=blockers,
    )
    readiness_blockers = execution_admission.get("blockers")
    if not isinstance(readiness_blockers, list) or any(
        not isinstance(item, str) or not item.strip() for item in readiness_blockers
    ):
        blockers.append("launch_profile_readiness_blockers_invalid")
    elif execution_admission.get("live_enabled") is True and readiness_blockers:
        blockers.append("launch_profile_live_enabled_with_blockers")
    elif execution_admission.get("live_enabled") is False and not readiness_blockers:
        blockers.append("launch_profile_live_disabled_without_blocker")
    webapp_sync = _mapping(profile.get("webapp_sync"))
    sync_attempts = webapp_sync.get("max_attempts")
    if (
        not isinstance(sync_attempts, int)
        or isinstance(sync_attempts, bool)
        or sync_attempts < 1
        or sync_attempts > 100
    ):
        blockers.append("launch_profile_webapp_sync_attempts_invalid")
    reconciliation = _mapping(profile.get("reconciliation"))
    required_providers = reconciliation.get("required_providers")
    if (
        not isinstance(required_providers, list)
        or not required_providers
        or any(item not in {"vast", "runpod", "gcp", "aws"} for item in required_providers)
    ):
        blockers.append("launch_profile_reconciliation_providers_invalid")
    max_guard_age = reconciliation.get("max_guard_age_seconds")
    if not isinstance(max_guard_age, int) or isinstance(max_guard_age, bool) or max_guard_age <= 0:
        blockers.append("launch_profile_reconciliation_guard_age_invalid")
    terminal = _mapping(profile.get("terminal_contract"))
    terminal_result_path = str(terminal.get("result_path") or "").strip()
    if not terminal_result_path:
        blockers.append("launch_profile_terminal_result_path_missing")
    elif _has_unknown_placeholder(terminal_result_path):
        blockers.append("launch_profile_terminal_result_path_placeholder_invalid")
    if not isinstance(terminal.get("success_statuses"), list) or not terminal.get(
        "success_statuses"
    ):
        blockers.append("launch_profile_terminal_statuses_missing")
    required_values = _mapping(terminal.get("required_values"))
    if required_values.get("continuing_spend_from_this_run") is not False:
        blockers.append("launch_profile_provider_zero_check_missing")
    required_paths = terminal.get("required_path_fields")
    if not isinstance(required_paths, list) or not {
        "teardown_manifest_path",
        "artifact_manifest_path",
    }.issubset(set(str(item) for item in required_paths or [])):
        blockers.append("launch_profile_terminal_artifact_checks_missing")
    controls = _mapping(profile.get("required_controls"))
    if not _is_identifier(controls.get("secret_profile_id")):
        blockers.append("launch_profile_secret_profile_missing")
    for field in (
        "canonical_allocator",
        "watchdog_required",
        "artifact_storage_required",
        "teardown_required",
        "provider_zero_required",
        "webapp_status_sync_required",
    ):
        expected = CANONICAL_ALLOCATOR_ENTRYPOINT if field == "canonical_allocator" else True
        if controls.get(field) != expected:
            blockers.append(f"launch_profile_control_missing:{field}")
    if controls.get("retry_cap") != 0:
        blockers.append("launch_profile_control_missing:retry_cap")
    if profile.get("claim_ceiling") not in {
        "development_only",
        "partner_run_pending_physical_join",
    }:
        blockers.append("launch_profile_claim_ceiling_invalid")
    if profile.get("profile_digest") != canonical_digest(profile, digest_field="profile_digest"):
        blockers.append("launch_profile_digest_mismatch")
    if _contains_secret_key(profile):
        blockers.append("launch_profile_secret_value_forbidden")
    return sorted(set(blockers))


def _has_unknown_placeholder(value: str) -> bool:
    remainder = value.replace(LAUNCH_RUN_ROOT_PLACEHOLDER, "")
    return "{" in remainder or "}" in remainder


def _render_launch_path(value: str, *, run_root: Path) -> str:
    if _has_unknown_placeholder(value):
        raise TaskEvaluationLaunchError("launch_profile_runtime_placeholder_invalid")
    return value.replace(LAUNCH_RUN_ROOT_PLACEHOLDER, str(run_root))


def public_launch_profile_descriptor(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only profile fields a public WebApp selector may observe."""

    blockers = validate_launch_profile(profile)
    if blockers:
        raise TaskEvaluationLaunchError(",".join(blockers))
    return {field: profile[field] for field in PUBLIC_PROFILE_DESCRIPTOR_FIELDS}


def validate_public_launch_profile_descriptor(value: Mapping[str, Any]) -> list[str]:
    """Fail closed on a public projection that could smuggle execution details."""

    descriptor = _mapping(value)
    blockers: list[str] = []
    if set(descriptor) != set(PUBLIC_PROFILE_DESCRIPTOR_FIELDS):
        blockers.append("launch_profile_public_descriptor_fields_invalid")
    if not _is_identifier(descriptor.get("profile_id")):
        blockers.append("launch_profile_public_id_invalid")
    if not _is_digest(descriptor.get("profile_digest")):
        blockers.append("launch_profile_public_digest_invalid")
    _validate_reference(
        descriptor.get("source_bundle"), field="launch_profile_public_source", blockers=blockers
    )
    source = _mapping(descriptor.get("source_bundle"))
    if not _is_identifier(source.get("bundle_id")) or source.get("source_kind") not in {
        "interiorgs_sage",
        "raw_v3_2_capture",
        "scaniverse_derived",
    }:
        blockers.append("launch_profile_public_source_invalid")
    _validate_reference(
        descriptor.get("evaluation_run_spec"),
        field="launch_profile_public_evaluation_run_spec",
        blockers=blockers,
    )
    controls = _mapping(descriptor.get("required_controls"))
    expected_controls = {
        "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "watchdog_required": True,
        "artifact_storage_required": True,
        "teardown_required": True,
        "provider_zero_required": True,
        "webapp_status_sync_required": True,
        "retry_cap": 0,
    }
    if set(controls) != {*expected_controls, "secret_profile_id"}:
        blockers.append("launch_profile_public_controls_fields_invalid")
    for field, expected in expected_controls.items():
        if controls.get(field) != expected:
            blockers.append(f"launch_profile_public_control_invalid:{field}")
    if not _is_identifier(controls.get("secret_profile_id")):
        blockers.append("launch_profile_public_secret_profile_invalid")
    execution = _mapping(descriptor.get("execution_admission"))
    if set(execution) != {"live_enabled", "readiness_receipt", "blockers"}:
        blockers.append("launch_profile_public_execution_fields_invalid")
    if not isinstance(execution.get("live_enabled"), bool):
        blockers.append("launch_profile_public_live_enabled_invalid")
    _validate_reference(
        execution.get("readiness_receipt"),
        field="launch_profile_public_readiness_receipt",
        blockers=blockers,
    )
    readiness_blockers = execution.get("blockers")
    if not isinstance(readiness_blockers, list) or any(
        not isinstance(item, str) or not item.strip() for item in readiness_blockers
    ):
        blockers.append("launch_profile_public_blockers_invalid")
    elif execution.get("live_enabled") is True and readiness_blockers:
        blockers.append("launch_profile_public_live_enabled_with_blockers")
    elif execution.get("live_enabled") is False and not readiness_blockers:
        blockers.append("launch_profile_public_dry_without_blocker")
    if descriptor.get("claim_ceiling") not in {
        "development_only",
        "partner_run_pending_physical_join",
    }:
        blockers.append("launch_profile_public_claim_ceiling_invalid")
    if _contains_secret_key(descriptor):
        blockers.append("launch_profile_public_secret_value_forbidden")
    return sorted(set(blockers))


def load_public_launch_profile_catalog(
    path_value: str | Path, *, max_bytes: int = 512 * 1024, max_profiles: int = 100
) -> dict[str, Any]:
    """Load a publisher-generated catalog without exposing its filesystem path."""

    source_input = Path(path_value).expanduser()
    if source_input.is_symlink():
        raise TaskEvaluationLaunchError("launch_profile_public_catalog_invalid")
    source = source_input.resolve()
    if not source.is_file() or source.stat().st_size > max_bytes:
        raise TaskEvaluationLaunchError("launch_profile_public_catalog_invalid")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) > max_profiles:
        raise TaskEvaluationLaunchError("launch_profile_public_catalog_invalid")
    profiles: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        descriptor = _mapping(item)
        blockers = validate_public_launch_profile_descriptor(descriptor)
        key = (str(descriptor.get("profile_id") or ""), str(descriptor.get("profile_digest") or ""))
        if blockers or key in seen:
            raise TaskEvaluationLaunchError("launch_profile_public_catalog_invalid")
        seen.add(key)
        profiles.append(descriptor)
    return {
        "schema_version": LAUNCH_PROFILE_CATALOG_SCHEMA_VERSION,
        "profiles": profiles,
    }


def _write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TaskEvaluationLaunchError(f"immutable_launch_conflict:{path.name}")
        return False


def stage_launch_request(*, value: Mapping[str, Any], queue_root: str | Path) -> dict[str, Any]:
    request = dict(value)
    blockers = validate_launch_request(request)
    if blockers:
        raise TaskEvaluationLaunchError(",".join(blockers))
    digest = str(request["request_digest"])
    queue = Path(queue_root).expanduser().resolve()
    filename = (
        f"{request['launch_id']}-{digest[len(_DIGEST_PREFIX) : len(_DIGEST_PREFIX) + 16]}.json"
    )
    existing: Path | None = None
    for state in ("pending", "processing", "completed", "blocked"):
        candidate = queue / state / filename
        if candidate.exists():
            if existing is not None:
                raise TaskEvaluationLaunchError(f"duplicate_launch_queue_state:{filename}")
            existing = candidate
    path = existing or queue / "pending" / filename
    created = _write_immutable(path, request)
    return {
        "schema_version": "task_evaluation_launch_queue_receipt.v1",
        "status": path.parent.name if not created else "queued",
        "already_exists": not created,
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": digest,
        "launch_profile_id": request["launch_profile_id"],
        "launch_profile_digest": request["launch_profile_digest"],
        "queue_path": str(path),
        "provider_mutation_performed": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"json_object_required:{path.name}")
    return dict(value)


def _file_digest(path: Path) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "digest": _file_digest(path) if path.is_file() else None,
    }


def verify_profile_immutable_inputs(profile: Mapping[str, Any]) -> list[str]:
    """Verify that every profile-bound local input is a regular, non-symlink file."""

    blockers: list[str] = []
    for item in profile.get("immutable_inputs") or []:
        immutable_input = _mapping(item)
        name = str(immutable_input.get("name") or "invalid")
        raw_path = Path(str(immutable_input.get("path") or "")).expanduser()
        if raw_path.is_symlink() or not raw_path.is_file():
            blockers.append(f"launch_profile_immutable_input_missing:{name}")
            continue
        if _file_digest(raw_path.resolve()) != immutable_input.get("digest"):
            blockers.append(f"launch_profile_immutable_input_digest_mismatch:{name}")
    return sorted(set(blockers))


@contextlib.contextmanager
def _scoped_runtime_environment(values: Mapping[str, Any]):
    """Apply an already-validated profile environment only around the allocator call."""

    prior = {str(key): os.environ.get(str(key)) for key in values}
    try:
        for key, value in values.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _terminal_evidence(
    profile: Mapping[str, Any], *, execute: bool, run_root: Path
) -> dict[str, Any]:
    terminal = _mapping(profile.get("terminal_contract"))
    result_path = Path(
        _render_launch_path(str(terminal.get("result_path") or ""), run_root=run_root)
    ).expanduser().resolve()
    if not execute:
        return {
            "status": "not_required_for_dry_run",
            "result": _artifact(result_path),
            "blockers": [],
        }
    blockers: list[str] = []
    result: dict[str, Any] = {}
    if not result_path.is_file():
        blockers.append("allocator_terminal_result_missing")
    else:
        try:
            result = _read_json(result_path)
        except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
            blockers.append("allocator_terminal_result_invalid")
    if result:
        if result.get("status") not in terminal.get("success_statuses", []):
            blockers.append("allocator_terminal_status_not_success")
        for field, expected in _mapping(terminal.get("required_values")).items():
            if result.get(field) != expected:
                blockers.append(f"allocator_terminal_value_mismatch:{field}")
        artifacts: dict[str, Any] = {}
        for field in terminal.get("required_path_fields") or []:
            artifact_path = Path(str(result.get(field) or "")).expanduser().resolve()
            artifacts[str(field)] = _artifact(artifact_path)
            if not artifact_path.is_file():
                blockers.append(f"allocator_terminal_artifact_missing:{field}")
    else:
        artifacts = {}
    return {
        "status": "passed" if not blockers else "blocked",
        "result": _artifact(result_path),
        "artifacts": artifacts,
        "blockers": sorted(set(blockers)),
    }


def dispatch_launch_request(
    *,
    request_path: str | Path,
    profile_dir: str | Path,
    state_root: str | Path,
    execute: bool = False,
    allocator_runner: Callable[[Sequence[str]], int] | None = None,
) -> dict[str, Any]:
    request_source = Path(request_path).expanduser().resolve()
    request = _read_json(request_source)
    blockers = validate_launch_request(request)
    profile_path = (
        Path(profile_dir).expanduser().resolve() / f"{request.get('launch_profile_id', '')}.json"
    )
    profile: dict[str, Any] = {}
    if not profile_path.is_file():
        blockers.append("launch_profile_missing")
    else:
        try:
            profile = _read_json(profile_path)
        except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
            blockers.append("launch_profile_invalid_json")
    if profile:
        blockers.extend(validate_launch_profile(profile))
        blockers.extend(verify_profile_immutable_inputs(profile))
        if request.get("launch_profile_digest") != profile.get("profile_digest"):
            blockers.append("launch_profile_binding_mismatch")
        if _canonical_json(_mapping(request.get("source_bundle"))) != _canonical_json(
            _mapping(profile.get("source_bundle"))
        ):
            blockers.append("source_bundle_profile_binding_mismatch")
        if _canonical_json(_mapping(request.get("evaluation_run_spec"))) != _canonical_json(
            _mapping(profile.get("evaluation_run_spec"))
        ):
            blockers.append("evaluation_run_spec_profile_binding_mismatch")
        request_controls = _mapping(request.get("required_controls"))
        profile_controls = _mapping(profile.get("required_controls"))
        if request_controls != profile_controls:
            blockers.append("launch_required_controls_profile_binding_mismatch")
        if request.get("claim_ceiling") != profile.get("claim_ceiling"):
            blockers.append("launch_claim_ceiling_profile_binding_mismatch")
        approved_spend = _mapping(_mapping(request.get("authorization")).get("spend")).get(
            "max_spend_usd"
        )
        profile_spend = _mapping(profile.get("allocator")).get("max_spend_usd")
        if isinstance(approved_spend, (int, float)) and isinstance(profile_spend, (int, float)):
            if profile_spend > approved_spend:
                blockers.append("launch_profile_exceeds_approved_spend")

    live_requested = bool(execute)
    live_allowed = _is_truthy(os.getenv(EXECUTE_ENV))
    if live_requested and not live_allowed:
        blockers.append(f"missing_env_{EXECUTE_ENV}")
    authorization = _mapping(request.get("authorization"))
    spend_expiry = _parse_timestamp(_mapping(authorization.get("spend")).get("expires_at"))
    if live_requested and spend_expiry is not None and spend_expiry <= datetime.now(timezone.utc):
        blockers.append("spend_authority_expired")
    profile_controls = _mapping(profile.get("required_controls"))
    configured_secret_profile_id = str(os.getenv(SECRET_PROFILE_ID_ENV) or "").strip()
    expected_secret_profile_id = str(profile_controls.get("secret_profile_id") or "")
    secret_profile_match = bool(
        configured_secret_profile_id and configured_secret_profile_id == expected_secret_profile_id
    )
    if live_requested and not secret_profile_match:
        blockers.append("canonical_secret_profile_mismatch")
    execution_admission = _mapping(profile.get("execution_admission"))
    if live_requested and execution_admission.get("live_enabled") is not True:
        blockers.append("launch_profile_live_execution_disabled")

    run_root = Path(state_root).expanduser().resolve() / str(request.get("launch_id") or "invalid")
    run_root.mkdir(parents=True, exist_ok=True)
    prior_receipt_path = run_root / "launch_receipt.json"
    if prior_receipt_path.is_file():
        prior_receipt = _read_json(prior_receipt_path)
        if prior_receipt.get("request_digest") != request.get("request_digest"):
            raise TaskEvaluationLaunchError("launch_receipt_request_binding_mismatch")
        return prior_receipt
    _write_immutable(run_root / "launch_request.json", request)
    if profile:
        _write_immutable(run_root / "launch_profile.json", profile)
    bound = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": request.get("launch_id"),
        "run_id": request.get("run_id"),
        "request_digest": request.get("request_digest"),
        "profile_digest": profile.get("profile_digest") if profile else None,
        "source_bundle_digest": _mapping(request.get("source_bundle")).get("digest"),
        "evaluation_run_spec_digest": _mapping(request.get("evaluation_run_spec")).get("digest"),
        "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "execute_requested": live_requested,
        "execute_env_allowed": live_allowed,
        "secret_profile_id_match": secret_profile_match,
        "profile_live_enabled": execution_admission.get("live_enabled"),
    }
    bound["binding_digest"] = canonical_digest(bound, digest_field="binding_digest")
    _write_immutable(run_root / "launch_binding.json", bound)
    started = {
        "schema_version": "task_evaluation_launch_started.v1",
        "launch_id": request.get("launch_id"),
        "run_id": request.get("run_id"),
        "request_digest": request.get("request_digest"),
        "binding_digest": bound["binding_digest"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hard_ttl_seconds": _mapping(profile.get("allocator")).get("hard_ttl_seconds")
        if profile
        else None,
        "process_id": os.getpid(),
        "automatic_retry_authorized": False,
    }
    started["started_digest"] = canonical_digest(started, digest_field="started_digest")
    _write_immutable(run_root / "launch_started.json", started)

    allocator_exit_code: int | None = None
    stdout_text = ""
    stderr_text = ""
    if not blockers and profile:
        allocator = _mapping(profile.get("allocator"))
        argv = [
            "gpu-canary",
            *[
                _render_launch_path(item, run_root=run_root)
                for item in list(allocator.get("argv") or [])
            ],
        ]
        if live_requested:
            argv.append("--execute")
        if allocator_runner is None:
            try:
                with _scoped_runtime_environment(
                    _mapping(profile.get("runtime_environment"))
                ):
                    completed = subprocess.run(  # nosec B603 - fixed module plus validated profile argv
                        [
                            sys.executable,
                            "-m",
                            "blueprint_pipeline.paid_resource_allocator",
                            *argv,
                        ],
                        shell=False,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=(
                            int(
                                _mapping(profile.get("allocator")).get(
                                    "hard_ttl_seconds"
                                )
                                or 1
                            )
                            + 300
                        ),
                    )
                allocator_exit_code = completed.returncode
                stdout_text = completed.stdout
                stderr_text = completed.stderr
            except subprocess.TimeoutExpired as exc:
                allocator_exit_code = 124
                stdout_text = str(exc.stdout or "")
                stderr_text = str(exc.stderr or "") + "\ncanonical_allocator_timeout\n"
        else:
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with (
                    _scoped_runtime_environment(
                        _mapping(profile.get("runtime_environment"))
                    ),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    allocator_exit_code = int(allocator_runner(argv))
            finally:
                stdout_text = stdout.getvalue()
                stderr_text = stderr.getvalue()
        (run_root / "allocator.stdout.log").write_text(stdout_text, encoding="utf-8")
        (run_root / "allocator.stderr.log").write_text(stderr_text, encoding="utf-8")
        if allocator_exit_code != 0:
            blockers.append("canonical_allocator_nonzero_exit")

    terminal = (
        _terminal_evidence(profile, execute=live_requested, run_root=run_root)
        if profile
        else {
            "status": "blocked",
            "blockers": ["launch_profile_missing"],
        }
    )
    blockers.extend(terminal.get("blockers") or [])
    if blockers:
        status = "blocked"
    elif live_requested:
        status = "completed"
    else:
        status = "dry_run_completed"
    receipt = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "launch_id": request.get("launch_id"),
        "run_id": request.get("run_id"),
        "request_digest": request.get("request_digest"),
        "launch_profile_digest": profile.get("profile_digest") if profile else None,
        "binding_digest": bound["binding_digest"],
        "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
        "allocator_exit_code": allocator_exit_code,
        "execute_requested": live_requested,
        "provider_mutation_attempted": bool(live_requested and allocator_exit_code is not None),
        "terminal_evidence": terminal,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
        "agent_operator_used": False,
        "claim_ceiling": request.get("claim_ceiling"),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_immutable(run_root / "launch_receipt.json", receipt)
    from .task_evaluation_launch_webapp_sync import sync_launch_receipt_to_webapp

    sync_result = sync_launch_receipt_to_webapp(receipt=receipt)
    sync_attempt = {
        **sync_result,
        "attempt_number": 1,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "provider_mutation_performed": False,
    }
    sync_attempt["sync_result_digest"] = canonical_digest(
        sync_attempt, digest_field="sync_result_digest"
    )
    _write_immutable(
        run_root
        / "webapp_sync_attempts"
        / f"{sync_attempt['sync_result_digest'][len(_DIGEST_PREFIX) :]}.json",
        sync_attempt,
    )
    if sync_result.get("status") == "succeeded":
        _write_immutable(run_root / "webapp_sync_succeeded.json", sync_attempt)
    return receipt


def process_launch_queue(
    *,
    queue_root: str | Path,
    profile_dir: str | Path,
    state_root: str | Path,
    execute: bool = False,
    max_messages: int = 1,
    allocator_runner: Callable[[Sequence[str]], int] | None = None,
) -> dict[str, Any]:
    queue = Path(queue_root).expanduser().resolve()
    pending = queue / "pending"
    processing = queue / "processing"
    processed: list[dict[str, Any]] = []
    pending.mkdir(parents=True, exist_ok=True)
    processing.mkdir(parents=True, exist_ok=True)
    for source in sorted(pending.glob("*.json"))[: max(0, max_messages)]:
        claimed = processing / source.name
        os.replace(source, claimed)
        try:
            receipt = dispatch_launch_request(
                request_path=claimed,
                profile_dir=profile_dir,
                state_root=state_root,
                execute=execute,
                allocator_runner=allocator_runner,
            )
        except Exception as exc:  # noqa: BLE001 - the queue must retain a terminal receipt
            receipt = {
                "schema_version": LAUNCH_RECEIPT_SCHEMA_VERSION,
                "status": "blocked",
                "launch_id": claimed.stem,
                "blockers": ["launch_dispatcher_unhandled_error"],
                "error_type": type(exc).__name__,
                "provider_mutation_attempted": None,
                "provider_mutation_state": "unknown_requires_independent_reconciliation",
                "retain_processing_for_reconciliation": True,
                "raw_error_message_recorded": False,
            }
        if receipt.get("retain_processing_for_reconciliation") is True:
            processed.append(receipt)
            continue
        destination_dir = queue / (
            "completed"
            if receipt.get("status")
            in {
                "completed",
                "dry_run_completed",
            }
            else "blocked"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        os.replace(claimed, destination_dir / claimed.name)
        processed.append(receipt)
    return {
        "schema_version": "task_evaluation_launch_queue_run.v1",
        "status": "completed"
        if all(row.get("status") != "blocked" for row in processed)
        else "blocked",
        "processed_count": len(processed),
        "receipts": processed,
        "automatic_retry_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--max-messages", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = process_launch_queue(
        queue_root=args.queue_root,
        profile_dir=args.profile_dir,
        state_root=args.state_root,
        execute=args.execute,
        max_messages=args.max_messages,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
