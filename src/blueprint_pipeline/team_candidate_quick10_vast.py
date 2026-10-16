"""Run ONE admitted team candidate's Quick-10 bundle through one canonical Vast allocation.

Beta journey step: L5 team policy evaluation. The canonical allocator
(``paid_resource_allocator gpu-canary`` → ``policy_canary_allocator_lane``)
calls this for a ``team_candidate_evaluation`` session authority, after the
lane's paid admission. It mirrors the frozen-pair session transport with the
team's differences:

- the bearer token reaches the GPU host only through the provider's private
  runtime-secret channel (the instance environment carries it base64-encoded;
  the bootstrap writes it to a 0600 file, exports its path under the
  ``*_FILE`` name the sealed spec names, and unsets the encoded variable);
  provider outputs redact it, and it is never in the bundle, argv or receipts;
- no Hugging Face token, no paired witness, no checkpoint download;
- the provider output is downloaded (streaming stays the pair's path).

Egress is bounded in the application: the team client talks to exactly one
allowlisted origin, with no redirects (``fetch_bounded_https``). The provider
host itself has no egress firewall here.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_isaac_lab_arena_vast import run_arena_native_control_vast
from .native_task_arena_policy_canary_session import (
    PROVIDER_RESULT_FILENAME,
    RESULT_SCHEMA_VERSION,
    TEAM_RUN_KIND,
    consume_session_authority_once,
    validate_provider_bundle,
    validate_session_authority,
)
from .native_task_arena_vast import (
    DEFAULT_KEY_PREFIX,
    MINIMUM_DRIVER_VERSION,
    NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES,
    POLICY_MIN_COMPUTE_CAP,
    POLICY_PROVIDER_BUNDLE_DOWNLOAD_BYTES,
    POLICY_RESULT_UPLOAD_BYTES,
    POLICY_RUNTIME_DEPENDENCY_DOWNLOAD_BYTES,
    _validated_policy_provider_runtime_environment,
)
from .paid_resource_admission import PaidResourceAdmissionGrant

_SECRET_FILE_ENV = re.compile(r"^[A-Z][A-Z0-9_]{2,58}_FILE$")
# Ten episodes of per-query frames and review media (the pair budgets 8 GB for twenty).
TEAM_RESULT_UPLOAD_BYTES = 4 * POLICY_RESULT_UPLOAD_BYTES


def team_result_upload_bytes(episodes: int) -> int:
    """The download budget for one team session, scaled from ten episodes to its own cell count."""
    return TEAM_RESULT_UPLOAD_BYTES * max(1, -(-int(episodes) // 10))


def credential_file_blockers(path: str | Path | None) -> list[str]:
    """The private token file the lane hands to the provider channel: regular, 0600, owned, bounded."""

    if not path:
        return ["team_candidate_credential_file_missing"]
    candidate = Path(path)
    try:
        status = candidate.lstat()
    except OSError:
        return ["team_candidate_credential_file_missing"]
    if (not candidate.is_absolute() or stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o077 or status.st_uid != os.getuid() or not 0 < status.st_size <= 65_536):
        return ["team_candidate_credential_file_not_private"]
    return []


def _blocked(blockers: list[str], **extra: Any) -> dict[str, Any]:
    return {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked", "provider_mutations_performed": 0,
            "provider_allocations_observed": 0, "retry_cap": 0, "blockers": blockers, **extra}


def run_team_candidate_session_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    session_authority: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    credential_file: str | Path | None,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 0.80,
    hard_cap_usd: float = 4.00,
    hard_ttl_seconds: int = 14_400,
    allowed_active_instance_ids: Sequence[int] = (),
    provider_runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One zero-retry allocation; the authority is consumed once, only when executing."""

    authority = validate_session_authority(session_authority)
    bundle = validate_provider_bundle(prepared_bundle, authority=authority)
    if authority["run_kind"] != TEAM_RUN_KIND:
        raise ValueError("team_candidate_session_run_kind_invalid")
    if float(authority["hard_cap_usd"]) != float(hard_cap_usd) or int(authority["hard_ttl_seconds"]) != int(
            hard_ttl_seconds):
        raise ValueError("policy_canary_session_resource_bounds_mismatch")
    credential_env = str(bundle.get("team_credential_env") or "")
    blockers = credential_file_blockers(credential_file)
    if _SECRET_FILE_ENV.fullmatch(credential_env) is None or bundle.get("secret_material_bundled") is not False:
        blockers.append("team_candidate_bundle_credential_binding_invalid")
    if blockers:
        return _blocked(sorted(set(blockers)))
    environment = _validated_policy_provider_runtime_environment(provider_runtime_environment)
    consumption = consume_session_authority_once(
        authority, consumption_path=Path(job_dir) / "policy_canary_session_consumption.json") if execute else None
    if consumption is not None and consumption.get("status") != "consumed":
        return _blocked(list(consumption.get("blockers") or []), authorization_consumption=consumption)
    return run_arena_native_control_vast(
        approval_path=".", job_dir=job_dir, paid_resource_admission_grant=paid_resource_admission_grant,
        execute=execute, prepared_bundle=bundle, machine_avoidlist_path=machine_avoidlist_path,
        max_hourly_rate_usd=max_hourly_rate_usd, hard_cap_usd=hard_cap_usd, hard_ttl_seconds=hard_ttl_seconds,
        expected_output_filename=PROVIDER_RESULT_FILENAME, container_image=str(bundle["container_image"]),
        provider_bundle_kind="native_task_arena_policy_canary_session", result_schema_version=RESULT_SCHEMA_VERSION,
        object_store_key_prefix=f"{DEFAULT_KEY_PREFIX}/team-candidate-session",
        instance_label_prefix="blueprint-native-task-policy-canary-",
        instance_label_exact=str(authority["resource_name"]), blocker_prefix="native_task_arena_policy_canary_session",
        min_gpu_ram_mb=46_000, min_compute_cap=POLICY_MIN_COMPUTE_CAP,
        allowed_active_instance_ids=tuple(sorted({int(value) for value in allowed_active_instance_ids})),
        vast_launch_lock_file=None, candidate_policy_query_expected=True, forward_hf_token=False,
        preferred_gpu_keywords=("L40S", "RTX 6000 Ada", "RTX A6000"), minimum_driver_version=MINIMUM_DRIVER_VERSION,
        require_independent_watchdog=True,
        **({"hard_deadline_epoch": authority["hard_deadline_epoch"]} if "hard_deadline_epoch" in authority else {}),
        allowed_geolocation_country_codes=NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES,
        authorization_consumption=consumption, stale_offer_create_retry_limit=0,
        expected_provider_download_bytes=POLICY_RUNTIME_DEPENDENCY_DOWNLOAD_BYTES + POLICY_PROVIDER_BUNDLE_DOWNLOAD_BYTES,
        expected_provider_upload_bytes=team_result_upload_bytes(authority["episodes_per_policy"]),
        provider_runtime_environment=environment,
        # Outputs download after the session (the launcher default); only the frozen
        # Quick-10 session streams provider outputs.
        runtime_secret_file_paths={credential_env: Path(str(credential_file))},
    )


__all__ = ["credential_file_blockers", "run_team_candidate_session_vast", "team_result_upload_bytes"]
