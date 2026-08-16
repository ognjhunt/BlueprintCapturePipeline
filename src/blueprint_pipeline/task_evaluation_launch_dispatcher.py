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

from .host_resident_launch_inputs import launch_profile_residency_blockers
from .task_evaluation_standing_launch_authorization import (
    STANDING_AUTHORIZATION_DIR_ENV,
    StandingAuthorizationError,
    consume_standing_authorization_once,
    consumption_totals,
    standing_authorization_admits,
)


LAUNCH_REQUEST_SCHEMA_VERSION = "task_evaluation_launch_request.v1"
LAUNCH_PROFILE_SCHEMA_VERSION = "task_evaluation_launch_profile.v1"
LAUNCH_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_receipt.v1"
LAUNCH_PROFILE_CATALOG_SCHEMA_VERSION = "task_evaluation_launch_profile_catalog.v1"
CANONICAL_ALLOCATOR_ENTRYPOINT = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"

EXECUTE_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE"
EXECUTE_LAUNCH_ID_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID"
SECRET_PROFILE_ID_ENV = "BLUEPRINT_TASK_EVALUATION_SECRET_PROFILE_ID"
LAUNCH_RUN_ROOT_PLACEHOLDER = "{launch_run_root}"
_DIGEST_PREFIX = "sha256:"
_URI_SCHEMES = ("gs://", "s3://", "r2://", "https://")
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
QUEUE_RUN_SCHEMA_VERSION = "task_evaluation_launch_queue_run.v1"
#: Where the scene a launch runs over actually came from. This is a provenance
#: claim, not a label, so a lane whose scene is none of these does not get to
#: borrow the nearest one -- the fresh-site camera lane runs over a public
#: NVIDIA SimReady warehouse dataset and is neither a capture nor InteriorGS.
#:
#: Stated once. It was written out three times -- request, profile, and public
#: descriptor -- which is three chances for one of them to fall behind.
LAUNCH_PROFILE_SOURCE_KINDS = frozenset(
    {
        "interiorgs_sage",
        "raw_v3_2_capture",
        "scaniverse_derived",
        "nvidia_simready_warehouse",
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


def standing_authorization_directory(state_root: str | Path) -> str:
    """Where this host keeps standing authorizations.

    The environment variable stays authoritative, but it is not required. The
    deployed control plane never set it, so the standing authorization shipped
    in #462 could not admit anything there: the dispatcher read an unset
    variable and fell straight back to the per-run `--execute-launch-id`
    handshake it was meant to replace. Deriving a default beside the launch
    state root is what makes the capability work on a host restored from an
    image, which runs no installer and applies no remembered env edit.

    Defaulting is safe in the direction that matters: a host with no
    authorization on disk is still refused, with no blocker of its own.
    """

    configured = str(os.getenv(STANDING_AUTHORIZATION_DIR_ENV) or "").strip()
    if configured:
        return configured
    return str(Path(state_root).expanduser().resolve().parent / "standing-authorizations")


def _standing_authorization_decision(
    profile: Mapping[str, Any], live_requested: bool, state_root: str | Path
) -> dict[str, Any]:
    """Consult the standing per-profile authorization, if this host has one."""
    if not live_requested:
        return {"admitted": False, "blockers": []}
    directory = standing_authorization_directory(state_root)
    if not directory:
        return {"admitted": False, "blockers": []}
    profile_id = str(profile.get("profile_id") or "")
    try:
        launches, spend = consumption_totals(directory=directory, profile_id=profile_id)
    except StandingAuthorizationError as exc:
        # Spend we cannot account for must not be treated as zero.
        return {"admitted": False, "blockers": [str(exc)]}
    return standing_authorization_admits(
        profile=profile,
        directory=directory,
        launches_consumed=launches,
        spend_consumed_usd=spend,
    )


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
    if source_bundle.get("source_kind") not in LAUNCH_PROFILE_SOURCE_KINDS:
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
    if profile_source.get("source_kind") not in LAUNCH_PROFILE_SOURCE_KINDS:
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
    if "prelaunch_skill_plan" in profile:
        # Keep legacy profiles independent of optional adapter dependencies.
        # New profiles gain a profile-bound plan, never a caller-provided path.
        from .task_evaluation_prelaunch_skills import validate_profile_prelaunch_skill_plan

        blockers.extend(validate_profile_prelaunch_skill_plan(profile))
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
    standing_requirement = profile.get("standing_launch_authorization")
    if standing_requirement is not None:
        requirement = _mapping(standing_requirement)
        expected_requirement = {
            "schema_version": (
                "task_evaluation_standing_launch_authorization_requirement.v1"
            ),
            "required_for_live_execution": True,
            "maximum_launches": 1,
            "consumption_must_precede_allocator": True,
        }
        if requirement != expected_requirement:
            blockers.append("launch_profile_standing_authorization_requirement_invalid")
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
    descriptor = {field: profile[field] for field in PUBLIC_PROFILE_DESCRIPTOR_FIELDS}
    allocator = _mapping(profile.get("allocator"))
    descriptor["required_authorization"] = {
        "max_spend_usd": allocator.get("max_spend_usd"),
        "hard_ttl_seconds": allocator.get("hard_ttl_seconds"),
    }
    return descriptor


def validate_public_launch_profile_descriptor(value: Mapping[str, Any]) -> list[str]:
    """Fail closed on a public projection that could smuggle execution details."""

    descriptor = _mapping(value)
    blockers: list[str] = []
    if set(descriptor) != {*PUBLIC_PROFILE_DESCRIPTOR_FIELDS, "required_authorization"}:
        blockers.append("launch_profile_public_descriptor_fields_invalid")
    authorization = _mapping(descriptor.get("required_authorization"))
    if set(authorization) != {"max_spend_usd", "hard_ttl_seconds"}:
        blockers.append("launch_profile_public_required_authorization_fields_invalid")
    public_spend = authorization.get("max_spend_usd")
    if (
        not isinstance(public_spend, (int, float))
        or isinstance(public_spend, bool)
        or public_spend <= 0
    ):
        blockers.append("launch_profile_public_required_spend_invalid")
    public_ttl = authorization.get("hard_ttl_seconds")
    if not isinstance(public_ttl, int) or isinstance(public_ttl, bool) or public_ttl <= 0:
        blockers.append("launch_profile_public_required_ttl_invalid")
    if not _is_identifier(descriptor.get("profile_id")):
        blockers.append("launch_profile_public_id_invalid")
    if not _is_digest(descriptor.get("profile_digest")):
        blockers.append("launch_profile_public_digest_invalid")
    _validate_reference(
        descriptor.get("source_bundle"), field="launch_profile_public_source", blockers=blockers
    )
    source = _mapping(descriptor.get("source_bundle"))
    if (
        not _is_identifier(source.get("bundle_id"))
        or source.get("source_kind") not in LAUNCH_PROFILE_SOURCE_KINDS
    ):
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


def validate_launch_request_against_public_catalog(
    value: Mapping[str, Any], *, catalog_path: str | Path
) -> list[str]:
    """Require the signed request to name one currently published profile.

    The profile directory retains immutable historical profiles for evidence and
    replay.  It is not a selector.  Only the publisher-generated catalog is
    allowed to authorize a new Website or signed intake request.
    """

    request = _mapping(value)
    try:
        catalog = load_public_launch_profile_catalog(catalog_path)
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
        return ["launch_profile_public_catalog_invalid"]
    descriptor = next(
        (
            row
            for row in catalog["profiles"]
            if row["profile_id"] == request.get("launch_profile_id")
            and row["profile_digest"] == request.get("launch_profile_digest")
        ),
        None,
    )
    if descriptor is None:
        return ["launch_profile_not_published"]
    blockers: list[str] = []
    for request_field, descriptor_field in (
        ("source_bundle", "source_bundle"),
        ("evaluation_run_spec", "evaluation_run_spec"),
        ("required_controls", "required_controls"),
        ("claim_ceiling", "claim_ceiling"),
    ):
        if request.get(request_field) != descriptor.get(descriptor_field):
            blockers.append(f"launch_profile_public_catalog_{request_field}_mismatch")
    return blockers


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
            raw = str(result.get(field) or "").strip()
            if not raw:
                # `Path("").resolve()` is the process working directory, so an
                # unset field used to be recorded as a descriptor naming the
                # release checkout -- evidence that a reader could mistake for
                # a real artifact that had merely gone missing. A field the
                # result never set has no path.
                artifacts[str(field)] = {"path": None, "exists": False, "digest": None}
                blockers.append(f"allocator_terminal_artifact_missing:{field}")
                continue
            artifact_path = Path(raw).expanduser().resolve()
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
    execute_launch_id: str | None = None,
    public_catalog_path: str | Path | None = None,
    allocator_runner: Callable[[Sequence[str]], int] | None = None,
) -> dict[str, Any]:
    request_source = Path(request_path).expanduser().resolve()
    request = _read_json(request_source)
    blockers = validate_launch_request(request)
    if public_catalog_path is not None:
        blockers.extend(
            validate_launch_request_against_public_catalog(
                request, catalog_path=public_catalog_path
            )
        )
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
        # Existence is not residency. The 2026-08-12 retained-scene run passed
        # every existence check and still reached the provider with two
        # authoring paths in argv, because the authoring tree had been
        # recreated on the host by hand.
        blockers.extend(launch_profile_residency_blockers(profile))
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
    execution_scope_launch_id = str(execute_launch_id or "").strip()
    # A standing per-profile authorization is the alternative to copying a
    # freshly minted launch id into the host's environment file before every
    # run. It is consulted only when the per-launch handshake is not satisfied,
    # so a launch carrying a matching id is admitted exactly as before and a
    # launch carrying neither is still refused.
    standing = _standing_authorization_decision(profile, live_requested, state_root)
    standing_requirement = _mapping(profile.get("standing_launch_authorization"))
    one_use_standing_required = bool(standing_requirement)
    if live_requested and one_use_standing_required:
        if not standing.get("admitted"):
            blockers.append("one_use_standing_authorization_required")
        elif standing.get("max_launches") != standing_requirement.get(
            "maximum_launches"
        ):
            blockers.append("one_use_standing_authorization_launch_limit_mismatch")
    if live_requested and not standing.get("admitted"):
        if not execution_scope_launch_id:
            blockers.append("execute_launch_id_required")
        elif not _is_identifier(execution_scope_launch_id):
            blockers.append("execute_launch_id_invalid")
        elif execution_scope_launch_id != request.get("launch_id"):
            blockers.append("execute_launch_scope_mismatch")
        # Only surface authorization faults when one was actually present:
        # a host with none is refused for the missing id, not for a document
        # it was never asked to have.
        blockers.extend(standing.get("blockers") or [])
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
        "execute_launch_id": execution_scope_launch_id if live_requested else None,
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

    prelaunch_skill_execution: dict[str, Any] = {
        "schema_version": "task_evaluation_prelaunch_skill_execution.v1",
        "status": "not_configured" if "prelaunch_skill_plan" not in profile else "not_started",
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_authorized": False,
        "agent_operator_used": False,
        "steps": [],
        "blockers": [],
    }
    if not blockers and profile and "prelaunch_skill_plan" in profile:
        try:
            from .task_evaluation_prelaunch_skills import execute_prelaunch_skill_plan

            prelaunch_skill_execution = execute_prelaunch_skill_plan(
                profile=profile,
                run_root=run_root,
            )
            if prelaunch_skill_execution.get("status") != "passed":
                blockers.append("prelaunch_skill_execution_blocked")
                blockers.extend(prelaunch_skill_execution.get("blockers") or [])
        except Exception:  # fail closed and retain a typed terminal launch receipt
            prelaunch_skill_execution = {
                "schema_version": "task_evaluation_prelaunch_skill_execution.v1",
                "status": "blocked",
                "provider_mutation_performed": False,
                "allocator_invoked": False,
                "automatic_retry_authorized": False,
                "agent_operator_used": False,
                "steps": [],
                "blockers": ["prelaunch_skill_execution_internal_error"],
            }
            blockers.append("prelaunch_skill_execution_blocked")
            blockers.append("prelaunch_skill_execution_internal_error")

    allocator_exit_code: int | None = None
    stdout_text = ""
    stderr_text = ""
    if not blockers and profile and live_requested and standing.get("admitted"):
        # Count the launch before the allocator runs, not after. The bounds are
        # read back from these records on every later admission, so a run that
        # dies mid-flight must still count: over-counting refuses a launch that
        # might have been allowed, under-counting spends past an approval.
        # Nothing recorded these until now -- `max_launches` and
        # `max_total_spend_usd` were declared and never consumed, which made a
        # bounded authorization unbounded in practice.
        try:
            consumed = consume_standing_authorization_once(
                profile=profile,
                directory=standing_authorization_directory(state_root),
                launch_id=str(request.get("launch_id") or ""),
            )
            if not consumed.get("consumed"):
                blockers.append("standing_authorization_consumption_not_recorded")
                blockers.extend(consumed.get("blockers") or [])
            elif one_use_standing_required and consumed.get(
                "max_launches"
            ) != standing_requirement.get("maximum_launches"):
                blockers.append("one_use_standing_authorization_launch_limit_mismatch")
        except (OSError, StandingAuthorizationError, TypeError, ValueError):
            blockers.append("standing_authorization_consumption_not_recorded")

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
        "execute_launch_id": execution_scope_launch_id if live_requested else None,
        "provider_mutation_attempted": bool(live_requested and allocator_exit_code is not None),
        "prelaunch_skill_execution": prelaunch_skill_execution,
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
    execute_launch_id: str | None = None,
    public_catalog_path: str | Path | None = None,
    max_messages: int = 1,
    allocator_runner: Callable[[Sequence[str]], int] | None = None,
) -> dict[str, Any]:
    queue = Path(queue_root).expanduser().resolve()
    pending = queue / "pending"
    processing = queue / "processing"
    processed: list[dict[str, Any]] = []
    pending.mkdir(parents=True, exist_ok=True)
    processing.mkdir(parents=True, exist_ok=True)
    execution_scope_launch_id = str(execute_launch_id or "").strip()
    armed = _is_identifier(execution_scope_launch_id)
    ignored_terminal_execute_launch_id: str | None = None
    # A standing per-profile authorization is the other way a paid launch is
    # admitted. It was unreachable from here: the queue refused for a missing
    # launch id before `dispatch_launch_request` -- the only place that reads a
    # standing authorization -- was ever called. So every paid run still needed
    # the hand-edited env var the standing authorization exists to replace, and
    # a stale one silently filtered every newer request out of the queue.
    standing_root = Path(standing_authorization_directory(state_root)).expanduser()
    standing_present = standing_root.is_dir() and any(standing_root.glob("*.json"))
    sources = sorted(pending.glob("*.json"))
    if execute and armed and standing_present:
        scoped_sources = [
            source
            for source in sources
            if source.name.startswith(f"{execution_scope_launch_id}-")
        ]
        terminal_scope_exists = any(
            directory.is_dir()
            and any(directory.glob(f"{execution_scope_launch_id}-*.json"))
            for directory in (queue / "completed", queue / "blocked")
        )
        if not scoped_sources and terminal_scope_exists:
            # Exact launch windows are one-shot. A completed/blocked request can
            # remain in a host EnvironmentFile after its window closes; letting
            # that stale value keep filtering the queue strands every newer
            # website request even when it carries a valid standing authority.
            # Ignore only a terminal scope and keep each newer request subject
            # to its own digest-bound standing-authorization decision.
            ignored_terminal_execute_launch_id = execution_scope_launch_id
            execution_scope_launch_id = ""
            armed = False
    if execute and not armed and not standing_present:
        return {
            "schema_version": QUEUE_RUN_SCHEMA_VERSION,
            "status": "blocked",
            "processed_count": 0,
            "receipts": [],
            "blockers": ["execute_launch_id_required"],
            "execute_requested": True,
            "execute_launch_id": None,
            "ignored_terminal_execute_launch_id": None,
            "automatic_retry_performed": False,
        }
    if execute and armed:
        # A paid activation scoped to one immutable launch ID stays scoped to
        # it: do not claim, dry-run, or mutate any other pending request while
        # that one-shot window is open. With no ID armed there is no such window
        # to protect, and `dispatch_launch_request` still refuses any launch its
        # profile's standing authorization does not admit.
        sources = [
            source
            for source in sources
            if source.name.startswith(f"{execution_scope_launch_id}-")
        ]
    for source in sources[: max(0, max_messages)]:
        claimed = processing / source.name
        os.replace(source, claimed)
        try:
            receipt = dispatch_launch_request(
                request_path=claimed,
                profile_dir=profile_dir,
                state_root=state_root,
                execute=execute,
                execute_launch_id=execution_scope_launch_id if execute and armed else None,
                public_catalog_path=public_catalog_path,
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
        "schema_version": QUEUE_RUN_SCHEMA_VERSION,
        "status": "completed"
        if all(row.get("status") != "blocked" for row in processed)
        else "blocked",
        "processed_count": len(processed),
        "receipts": processed,
        "execute_requested": bool(execute),
        "execute_launch_id": execution_scope_launch_id if execute else None,
        "ignored_terminal_execute_launch_id": ignored_terminal_execute_launch_id,
        "automatic_retry_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--max-messages", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-launch-id")
    parser.add_argument("--public-catalog")
    args = parser.parse_args(argv)
    result = process_launch_queue(
        queue_root=args.queue_root,
        profile_dir=args.profile_dir,
        state_root=args.state_root,
        execute=args.execute,
        execute_launch_id=args.execute_launch_id,
        public_catalog_path=args.public_catalog,
        max_messages=args.max_messages,
    )
    print(json.dumps(result, sort_keys=True))
    # The queue status describes the launch outcome, not whether dispatch
    # worked. Exiting non-zero on a blocked launch made systemd mark the unit
    # failed and deactivate the path unit that watches the queue, so one
    # blocked run -- an entirely normal scientific outcome -- silently stopped
    # every later website trigger from ever dispatching. Reserve a non-zero
    # exit for a dispatcher that could not process the queue at all.
    return 0 if result.get("schema_version") == QUEUE_RUN_SCHEMA_VERSION else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
