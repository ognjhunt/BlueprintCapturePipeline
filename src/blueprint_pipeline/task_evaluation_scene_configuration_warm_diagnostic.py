"""Retain one Vast scene diagnostic worker for immutable source-only iterations.

The warm session is deliberately non-qualifying.  One ordinary scene diagnostic
bundle performs the expensive image, toolchain, checkpoint, and dependency
bootstrap.  Later iterations may replace only an inventory-bound source overlay
from the exact tip of a pushed ``codex/*`` branch and resume the exact scientific
checkpoint.  No interface accepts an arbitrary command.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import time
import urllib.error
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .retained_gpu_session_lifecycle import record_retained_gpu_state
from .gpu_render_providers import VastRenderProvider, enroll_vast_ssh_host_key
from .native_task_arena_warm_vast import (
    _dispatch_warm_script_over_ssh,
    _run_pinned_ssh,
)
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_warm_overlay import (
    OVERLAY_RECEIPT_SCHEMA_VERSION,
    OVERLAY_SCHEMA_VERSION,
    SceneConfigurationWarmDiagnosticError,
    _absolute_directory,
    _read,
    _record_path,
    _write_exclusive,
    build_scene_configuration_warm_source_overlay,
    validate_scene_configuration_warm_source_overlay,
)
from .task_evaluation_scene_configuration_warm_remote_protocol import (
    _remote_iteration_script,
    artifixer_warm_secret_envelope,
    artifixer_warm_secret_install_remote_argv,
)
from .vast_scene_warm_secret_probe import (
    probe_fresh_ssh_secret_environment_absent,
)
from .task_evaluation_scene_configuration_warm_execution_contract import (
    warm_execution_binding_blockers as _warm_execution_binding_blockers,
)
from .task_evaluation_scene_configuration_warm_allocation import (
    scene_configuration_warm_closeout_allocation_binding,
    scene_configuration_warm_iteration_allocation_binding,
)
from .task_evaluation_scene_configuration_warm_transport import (
    SIGNED_URL_RETRIEVAL_RESERVE_SECONDS,
    _download_bounded_when_ready,
    _output_object_ready,
    validated_warm_staging_urls,
)
from .task_evaluation_scene_configuration_vast import (
    _extract_provider_output,
)
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)
from .watchdog_owner_teardown_contract import write_owner_teardown_cancel_request


from .task_evaluation_scene_configuration_warm_contract import (
    ARTIFIXER_POST_TRAINING_CONTINUATION_KIND,
    ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES as ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES,
    ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES,
    BASE_RUNTIME_ROOT as BASE_RUNTIME_ROOT,
    CLOSEOUT_SCHEMA_VERSION,
    ITERATION_AUTHORITY_SCHEMA_VERSION,
    ITERATION_RESULT_SCHEMA_VERSION,
    REMOTE_ROOT,
    SESSION_AUTHORITY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    SESSION_STATE_NAME,
    SESSION_STATE_SCHEMA_VERSION,
    WARM_CARRIED_PAID_MODEL_STAGES,
    _ITERATION_ID,
    _append_session_event,
    _remote_artifixer_checkpoint_root as _remote_artifixer_checkpoint_root,
    _remote_checkpoint_root,
    _sha256_file,
    _write_state,
    materialize_scene_configuration_warm_iteration_authority,
    materialize_scene_configuration_warm_session,
    materialize_scene_configuration_warm_session_authority,
    scene_configuration_warm_session_owner_lock,
    validate_scene_configuration_warm_iteration_authority,
    validate_scene_configuration_warm_session,
    validate_scene_configuration_warm_session_authority,
)


def _consume_iteration_authority(
    *, root: Path, state: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    digest = str(authority["authority_digest"])
    consumed_root = root / "consumed_iterations"
    consumed_root.mkdir(mode=0o700, exist_ok=True)
    destination = consumed_root / f"{digest[7:]}.json"
    payload = {
        "schema_version": "task_evaluation_scene_configuration_warm_iteration_consumption.v1",
        "authority_digest": digest,
        "session_digest": authority["session_digest"],
        "iteration_id": authority["iteration_id"],
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 0,
        "consumption_digest": "",
    }
    payload["consumption_digest"] = canonical_digest(payload, digest_field="consumption_digest")
    try:
        _write_exclusive(destination, payload)
    except FileExistsError as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_authority_replayed"
        ) from exc
    next_state = {
        **dict(state),
        "status": "iteration_running",
        "attempted_iteration_count": int(state["attempted_iteration_count"]) + 1,
        "active_iteration_id": authority["iteration_id"],
        "active_iteration_authority_digest": digest,
        "state_digest": "",
    }
    next_state["state_digest"] = canonical_digest(next_state, digest_field="state_digest")
    _write_state(root, next_state)
    _append_session_event(
        root,
        {
            "schema_version": SESSION_STATE_SCHEMA_VERSION,
            "event": "iteration_authority_consumed",
            "iteration_id": authority["iteration_id"],
            "authority_digest": digest,
            "state_digest": next_state["state_digest"],
            "recorded_at": utc_now_iso(),
        },
    )
    return payload


def _install_artifixer_warm_secret_files(
    *, session: Mapping[str, Any], authority: Mapping[str, Any], job: Path
) -> dict[str, Any]:
    expected_digests = authority.get("fresh_openai_cost_scope_attestation_digests")
    if authority.get(
        "continuation_kind"
    ) != ARTIFIXER_POST_TRAINING_CONTINUATION_KIND or not isinstance(expected_digests, Mapping):
        return {
            "status": "blocked",
            "blockers": ["scene_configuration_artifixer_warm_secret_scope_invalid"],
            "raw_secret_values_recorded": False,
        }
    values: dict[str, str] = {}
    for name in ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES:
        path = Path(str(os.environ.get(name) or "")).expanduser()
        try:
            valid = (
                path.is_absolute()
                and not path.is_symlink()
                and path.is_file()
                and path.stat().st_size > 0
                and path.stat().st_size <= 65_536
                and path.stat().st_mode & 0o077 == 0
            )
            value = path.read_text(encoding="utf-8") if valid else ""
        except (OSError, UnicodeError):
            value = ""
            valid = False
        if (
            not valid
            or not value.strip()
            or "\x00" in value
            or (
                "COST_SCOPE_ATTESTATION" in name
                and _sha256_file(path) != expected_digests.get(name)
            )
        ):
            return {
                "status": "blocked",
                "blockers": ["scene_configuration_artifixer_warm_secret_scope_invalid"],
                "raw_secret_values_recorded": False,
            }
        values[name] = value
    enrollment = enroll_vast_ssh_host_key(
        session, attempt_dir=job / "artifixer_warm_ssh_trust", timeout_seconds=15
    )
    if enrollment.get("status") != "enrolled":
        return {
            "status": "blocked",
            "blockers": list(enrollment.get("blockers") or [])
            or ["scene_configuration_artifixer_warm_secret_install_failed"],
            "raw_secret_values_recorded": False,
        }
    try:
        try:
            envelope = artifixer_warm_secret_envelope(
                iteration_id=str(authority["iteration_id"]), secret_values=values
            )
            installed = _run_pinned_ssh(
                session=session,
                known_hosts_file=str(enrollment["known_hosts_file"]),
                remote_argv=artifixer_warm_secret_install_remote_argv(
                    iteration_id=str(authority["iteration_id"])
                ),
                stdin=envelope,
                timeout_seconds=30,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            installed = {
                "status": "blocked",
                "blockers": ["scene_configuration_artifixer_warm_secret_install_failed"],
                "raw_secret_values_recorded": False,
            }
    finally:
        values.clear()
    if (
        installed.get("status") != "completed"
        or installed.get("stdout") != "BLUEPRINT_SCENE_ARTIFIXER_WARM_SECRET_FILES_READY\n"
    ):
        return {
            "status": "blocked",
            "blockers": ["scene_configuration_artifixer_warm_secret_install_failed"],
            "host_key_enrollment": enrollment,
            "raw_secret_values_recorded": False,
        }
    return {
        "status": "completed",
        "blockers": [],
        "host_key_enrollment": enrollment,
        "secret_file_count": len(ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES),
        "transport": "strict_pinned_ssh_stdin_private_files.v1",
        "raw_secret_values_recorded": False,
    }


def _scrub_artifixer_warm_secret_files(
    *, session: Mapping[str, Any], authority: Mapping[str, Any], install: Mapping[str, Any]
) -> dict[str, Any]:
    enrollment = install.get("host_key_enrollment")
    iteration_id = str(authority.get("iteration_id") or "")
    if not isinstance(enrollment, Mapping) or _ITERATION_ID.fullmatch(iteration_id) is None:
        return {
            "status": "blocked",
            "secret_files_absent": False,
            "blockers": ["scene_configuration_artifixer_warm_secret_scrub_unproven"],
            "raw_secret_values_recorded": False,
        }
    pending = f"{REMOTE_ROOT}/pending-secrets/{iteration_id}"
    installed = f"{REMOTE_ROOT}/iterations/{iteration_id}/.runtime-secrets"
    command = (
        "set -euo pipefail; "
        f"rm -rf -- {shlex.quote(pending)} {shlex.quote(installed)}; "
        f"test ! -e {shlex.quote(pending)}; "
        f"test ! -e {shlex.quote(installed)}; "
        "printf 'BLUEPRINT_SCENE_ARTIFIXER_WARM_SECRET_FILES_ABSENT\\n'"
    )
    try:
        result = _run_pinned_ssh(
            session=session,
            known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
            remote_argv=["/bin/bash", "-c", command],
            timeout_seconds=30,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        result = {
            "status": "blocked",
            "blockers": ["scene_configuration_artifixer_warm_secret_scrub_unproven"],
            "raw_secret_values_recorded": False,
        }
    absent = bool(
        result.get("status") == "completed"
        and result.get("stdout") == "BLUEPRINT_SCENE_ARTIFIXER_WARM_SECRET_FILES_ABSENT\n"
    )
    return {
        "status": "completed" if absent else "blocked",
        "secret_files_absent": absent,
        "blockers": [] if absent else ["scene_configuration_artifixer_warm_secret_scrub_unproven"],
        "raw_secret_values_recorded": False,
    }


def _wait_for_remote_output_or_exit(
    *,
    session: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    output_get_url: str,
    attempt_key: str,
    deadline_monotonic: float,
) -> dict[str, Any]:
    pid = dispatch.get("remote_pid")
    enrollment = dispatch.get("host_key_enrollment")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(enrollment, Mapping)
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_dispatch_identity_invalid"
        )
    remote_log = f"/workspace/native_task_arena_warm_dispatches/{attempt_key}/run.log"
    started = time.monotonic()
    overlay_latency: float | None = None
    entrypoint_latency: float | None = None
    last_probe: dict[str, Any] = {}
    while time.monotonic() < deadline_monotonic:
        if _output_object_ready(output_get_url):
            return {
                "status": "output_ready",
                "overlay_applied_latency_seconds": overlay_latency,
                "entrypoint_started_latency_seconds": entrypoint_latency,
                "remote_probe": last_probe,
            }
        command = (
            f"if kill -0 {pid} 2>/dev/null; then state=running; else state=exited; fi; "
            "printf 'STATE:%s\\n' \"$state\"; "
            "grep -aoE 'BLUEPRINT_SCENE_WARM_(OVERLAY_APPLIED|ENTRYPOINT_STARTED|PROVIDER_OUTPUT_UPLOAD_OK|BLOCKED:[A-Za-z0-9:_.-]+)' -- "
            + shlex.quote(remote_log)
            + " | tail -n 40 || true"
        )
        last_probe = _run_pinned_ssh(
            session=session,
            known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
            remote_argv=["sh", "-c", command],
            timeout_seconds=30,
        )
        stdout = str(last_probe.get("stdout") or "")
        elapsed = max(0.0, time.monotonic() - started)
        if overlay_latency is None and "BLUEPRINT_SCENE_WARM_OVERLAY_APPLIED" in stdout:
            overlay_latency = elapsed
        if entrypoint_latency is None and "BLUEPRINT_SCENE_WARM_ENTRYPOINT_STARTED" in stdout:
            entrypoint_latency = elapsed
        if "STATE:exited" in stdout:
            # One final object read closes the race between process exit and PUT
            # visibility before failing immediately instead of waiting to TTL.
            ready = _output_object_ready(output_get_url)
            return {
                "status": "output_ready" if ready else "remote_exited_without_output",
                "overlay_applied_latency_seconds": overlay_latency,
                "entrypoint_started_latency_seconds": entrypoint_latency,
                "remote_probe": last_probe,
            }
        time.sleep(2)
    return {
        "status": "timeout",
        "overlay_applied_latency_seconds": overlay_latency,
        "entrypoint_started_latency_seconds": entrypoint_latency,
        "remote_probe": last_probe,
    }


def _quiesce_remote_dispatch(
    *,
    session: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    attempt_key: str,
) -> dict[str, Any]:
    """Boundedly stop the exact remote dispatch session before URL cleanup."""

    pid = dispatch.get("remote_pid")
    pgid = dispatch.get("remote_process_group_id")
    sid = dispatch.get("remote_session_id")
    enrollment = dispatch.get("host_key_enrollment")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (pid, pgid, sid)
        )
        or pid != pgid
        or pid != sid
        or not isinstance(enrollment, Mapping)
        or re.fullmatch(r"[0-9a-f]{16}", attempt_key) is None
    ):
        return {"status": "unproven", "remote_session_absent": False}
    expected_script = f"/workspace/native_task_arena_warm_dispatches/{attempt_key}/run.sh"
    command = f"""
set -eu
root={pid}
sid={sid}
attempt={shlex.quote(attempt_key)}
expected={shlex.quote(expected_script)}
session_members() {{
  for proc in /proc/[0-9]*; do
    [ -r "$proc/stat" ] || continue
    proc_pid=${{proc##*/}}
    stat=$(cat "$proc/stat" 2>/dev/null || true)
    [ -n "$stat" ] || continue
    rest=${{stat##*) }}
    set -- $rest
    [ "$#" -ge 4 ] || continue
    [ "$1" != Z ] || continue
    [ "$4" = "$sid" ] && printf '%s\\n' "$proc_pid"
  done
}}
members=$(session_members || true)
[ -n "$members" ] || {{ printf 'REMOTE_SESSION:absent\\n'; exit 0; }}
if kill -0 "$root" 2>/dev/null; then
  cmdline=$(tr '\\000' ' ' < "/proc/$root/cmdline" 2>/dev/null || true)
  case "$cmdline" in
    *"bash $expected"*) ;;
    *) printf 'REMOTE_SESSION:identity_mismatch\\n'; exit 74;;
  esac
fi
for member in $members; do
  kill -0 "$member" 2>/dev/null || continue
  tr '\\000' '\\n' < "/proc/$member/environ" 2>/dev/null \
    | grep -Fqx "BLUEPRINT_SCENE_WARM_DISPATCH_ATTEMPT=$attempt" \
    || {{ printf 'REMOTE_SESSION:identity_mismatch\\n'; exit 74; }}
done
for _pass in 1 2; do
  members=$(session_members || true)
  [ -z "$members" ] || kill -TERM $members 2>/dev/null || true
  for _wait in 1 2 3 4 5; do
    [ -n "$(session_members || true)" ] || break
    sleep 1
  done
  members=$(session_members || true)
  [ -z "$members" ] || kill -KILL $members 2>/dev/null || true
done
if [ -n "$(session_members || true)" ]; then
  printf 'REMOTE_SESSION:running\\n'
  exit 75
fi
printf 'REMOTE_SESSION:absent\\n'
"""
    observation = _run_pinned_ssh(
        session=session,
        known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
        remote_argv=["/bin/bash", "-c", command],
        timeout_seconds=20,
    )
    absent = bool(
        observation.get("status") == "completed"
        and "REMOTE_SESSION:absent" in str(observation.get("stdout") or "")
    )
    return {
        "status": "quiesced" if absent else "unproven",
        "remote_pid": pid,
        "remote_process_group_id": pgid,
        "remote_session_id": sid,
        "remote_session_absent": absent,
        "observation": observation,
    }


def _mark_iteration_state(
    *,
    root: Path,
    state: Mapping[str, Any],
    authority: Mapping[str, Any],
    status: str,
    result_digest: str,
    advanced_checkpoint_digest: str | None = None,
    advanced_checkpoint_prefix_count: int | None = None,
    advanced_checkpoint_stage_ids: list[str] | None = None,
    advanced_remote_checkpoint_root: str | None = None,
) -> dict[str, Any]:
    completed = int(state["completed_iteration_count"])
    next_state = {
        **dict(state),
        "status": status,
        "active_iteration_id": None,
        "active_iteration_authority_digest": None,
        "last_iteration_authority_digest": authority["authority_digest"],
        "last_iteration_result_digest": result_digest,
        "completed_iteration_count": completed + (1 if status == "ready" else 0),
        "state_digest": "",
    }
    if advanced_checkpoint_digest is not None:
        if (
            advanced_checkpoint_prefix_count is None
            or advanced_checkpoint_stage_ids is None
            or len(advanced_checkpoint_stage_ids) != advanced_checkpoint_prefix_count
            or advanced_checkpoint_prefix_count < 3
            or advanced_remote_checkpoint_root is None
        ):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_advanced_checkpoint_state_invalid"
            )
        next_state["current_checkpoint_digest"] = advanced_checkpoint_digest
        next_state["current_remote_checkpoint_root"] = _remote_checkpoint_root(
            advanced_remote_checkpoint_root
        )
        next_state["current_completed_stage_prefix_count"] = advanced_checkpoint_prefix_count
        next_state["current_completed_stage_ids"] = list(advanced_checkpoint_stage_ids)
        next_state["current_carried_paid_model_stages"] = list(WARM_CARRIED_PAID_MODEL_STAGES)
    next_state["state_digest"] = canonical_digest(next_state, digest_field="state_digest")
    _write_state(root, next_state)
    _append_session_event(
        root,
        {
            "schema_version": SESSION_STATE_SCHEMA_VERSION,
            "event": "iteration_completed" if status == "ready" else "iteration_failed",
            "iteration_id": authority["iteration_id"],
            "authority_digest": authority["authority_digest"],
            "result_digest": result_digest,
            "state_digest": next_state["state_digest"],
            "recorded_at": utc_now_iso(),
        },
    )
    return next_state


def run_scene_configuration_warm_iteration(
    *,
    session_root: str | Path,
    authority_path: str | Path,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Dispatch one fixed checkpoint resume to the retained exact instance."""

    job = Path(job_dir).expanduser()
    if not job.is_absolute() or job.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_job_invalid"
        )
    ensure_dir(job)
    with scene_configuration_warm_session_owner_lock(session_root):
        session, state, authority = validate_scene_configuration_warm_iteration_authority(
            session_root=session_root, authority_path=authority_path
        )
        if not execute:
            result = {
                "schema_version": ITERATION_RESULT_SCHEMA_VERSION,
                "status": "dry_run_ready",
                "provider_allocations_performed": 0,
                "instance_lifecycle_mutations_performed": 0,
                "remote_workload_dispatches_performed": 0,
                "iteration_id": authority["iteration_id"],
                "diagnostic_only": True,
                "development_only": True,
                "qualification_eligible": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
                "arbitrary_command_permitted": False,
                "raw_secret_values_recorded": False,
                "blockers": [],
                "result_digest": "",
            }
            result["result_digest"] = canonical_digest(result, digest_field="result_digest")
            write_json(job / f"{ITERATION_RESULT_SCHEMA_VERSION}.json", result)
            return result
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="vast_provider_adapter",
            allocation_binding_digest=canonical_digest(
                scene_configuration_warm_iteration_allocation_binding(
                    session=session, authority=authority
                )
            ),
            require_allocation_binding=True,
        )
        consumption = _consume_iteration_authority(
            root=_absolute_directory(
                session_root, code="scene_configuration_warm_session_root_invalid"
            ),
            state=state,
            authority=authority,
        )
        retained_provider = provider or VastRenderProvider()
        blockers: list[str] = []
        dispatch: dict[str, Any] = {}
        cleanup: dict[str, Any] = {}
        execution: dict[str, Any] = {}
        remote_dispatches = 0
        dispatch_start_latency_seconds: float | None = None
        stage_completion_seconds: float | None = None
        remote_progress: dict[str, Any] = {}
        remote_quiescence: dict[str, Any] = {}
        remote_writer_absent = False
        remote_setup_to_entrypoint_seconds: float | None = None
        secret_install: dict[str, Any] = {}
        secret_scrub: dict[str, Any] = {}
        fresh_secret_probe: dict[str, Any] = {}
        artifixer_continuation = (
            authority.get("continuation_kind") == ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
        )
        try:
            observation = retained_provider.inspect(str(session["provider_instance_id"]))
            if (
                observation.get("status") != "observed"
                or observation.get("api_confirmed") is not True
                or observation.get("provider_absence_confirmed") is not False
                or observation.get("instance_id") != str(session["provider_instance_id"])
                or observation.get("ssh_host") != session["ssh_host"]
                or observation.get("ssh_port") != session["ssh_port"]
                or str(
                    observation.get("actual_status") or observation.get("cur_state") or ""
                ).lower()
                not in {"running", "active"}
            ):
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_instance_identity_not_live"
                )
            overlay_record = authority["source_overlay_receipt"]
            overlay_receipt = validate_scene_configuration_warm_source_overlay(
                _record_path(
                    overlay_record,
                    code="scene_configuration_warm_iteration_overlay_invalid",
                ),
                expected_source_commit=authority["source_commit"],
                expected_checkpoint_digest=authority["source_checkpoint_digest"],
            )
            archive = _record_path(
                overlay_receipt["overlay_archive"],
                code="scene_configuration_warm_iteration_overlay_invalid",
            )
            staging_dir = job / "object_store_staging"
            staging = stage_wam_provider_bundle_object_store(
                job_dir=staging_dir,
                bundle_path=archive,
                key_prefix="blueprint/arm-decision-proof-v1/scene-configuration/warm",
                expiration_seconds=max(
                    600,
                    math.ceil(float(session["watchdog_deadline_epoch"]) - time.time())
                    + SIGNED_URL_RETRIEVAL_RESERVE_SECONDS,
                ),
            )
            if staging.get("status") != "completed":
                blockers.extend(
                    staging.get("blockers")
                    or ["scene_configuration_warm_object_store_staging_blocked"]
                )
            else:
                staging_urls = validated_warm_staging_urls(
                    staging_dir=staging_dir,
                    staging=staging,
                    overlay_archive=archive,
                    watchdog_deadline_epoch=float(session["watchdog_deadline_epoch"]),
                )
                remote_script = _remote_iteration_script(
                    authority=authority,
                    session=session,
                    overlay_url=staging_urls["overlay_url"],
                    output_put_url=staging_urls["output_put_url"],
                )
                attempt_key = str(authority["authority_digest"])[7:23]
                if artifixer_continuation:
                    secret_install = _install_artifixer_warm_secret_files(
                        session=session, authority=authority, job=job
                    )
                    if secret_install.get("status") != "completed":
                        blockers.extend(
                            secret_install.get("blockers")
                            or ["scene_configuration_artifixer_warm_secret_install_failed"]
                        )
                        remote_writer_absent = True
                dispatch_started = time.monotonic()
                if not artifixer_continuation or secret_install.get("status") == "completed":
                    dispatch = _dispatch_warm_script_over_ssh(
                        job=job,
                        session=session,
                        remote_script=remote_script,
                        attempt_key=attempt_key,
                        require_dedicated_session=True,
                    )
                dispatch_start_latency_seconds = max(0.0, time.monotonic() - dispatch_started)
                if not dispatch and artifixer_continuation:
                    pass
                elif dispatch.get("status") != "completed":
                    blockers.extend(
                        dispatch.get("blockers") or ["scene_configuration_warm_dispatch_blocked"]
                    )
                    remote_writer_absent = True
                else:
                    remote_dispatches = 1
                    output_zip = job / "vast_provider_runtime_output.zip"
                    output_get_url = staging_urls["output_get_url"]
                    stage_started = time.monotonic()
                    remote_progress = _wait_for_remote_output_or_exit(
                        session=session,
                        dispatch=dispatch,
                        output_get_url=output_get_url,
                        attempt_key=attempt_key,
                        deadline_monotonic=(
                            time.monotonic()
                            + max(
                                1,
                                int(float(session["watchdog_deadline_epoch"]) - time.time() - 120),
                            )
                        ),
                    )
                    remote_quiescence = _quiesce_remote_dispatch(
                        session=session,
                        dispatch=dispatch,
                        attempt_key=attempt_key,
                    )
                    remote_writer_absent = remote_quiescence.get("remote_session_absent") is True
                    if not remote_writer_absent:
                        blockers.append(
                            "scene_configuration_warm_remote_writer_quiescence_unproven"
                        )
                    remote_stdout = str(
                        (remote_progress.get("remote_probe") or {}).get("stdout") or ""
                    )
                    for marker in re.findall(
                        r"BLUEPRINT_SCENE_WARM_BLOCKED:([A-Za-z0-9:_.-]+)",
                        remote_stdout,
                    ):
                        blockers.append(f"scene_configuration_warm_remote_blocked:{marker}")
                    setup_match = re.search(
                        r"BLUEPRINT_SCENE_WARM_REMOTE_SETUP_STARTED_EPOCH_NS:([0-9]+)",
                        remote_stdout,
                    )
                    entrypoint_match = re.search(
                        r"BLUEPRINT_SCENE_WARM_ENTRYPOINT_STARTED_EPOCH_NS:([0-9]+)",
                        remote_stdout,
                    )
                    if setup_match and entrypoint_match:
                        elapsed_ns = int(entrypoint_match.group(1)) - int(setup_match.group(1))
                        if elapsed_ns >= 0:
                            remote_setup_to_entrypoint_seconds = elapsed_ns / 1e9
                    output_ready = remote_progress.get("status") == "output_ready"
                    if output_ready and remote_writer_absent:
                        output_ready = _download_bounded_when_ready(
                            url=output_get_url,
                            destination=output_zip,
                            maximum_bytes=int(session["maximum_warm_output_archive_bytes"]),
                            deadline_monotonic=time.monotonic() + 60,
                        )
                    stage_completion_seconds = max(0.0, time.monotonic() - stage_started)
                    if not output_ready:
                        blockers.append("scene_configuration_warm_output_timeout")
                    else:
                        execution, extraction_blockers = _extract_provider_output(
                            output_zip,
                            job / "immutable_execution",
                            maximum_archive_bytes=int(session["maximum_warm_output_archive_bytes"]),
                            diagnostic_only=True,
                        )
                        blockers.extend(extraction_blockers)
                        blockers.extend(
                            _warm_execution_binding_blockers(
                                execution=execution,
                                session=session,
                                authority=authority,
                            )
                        )
        except (OSError, RuntimeError, TypeError, ValueError, urllib.error.URLError) as exc:
            blockers.append(
                "scene_configuration_warm_iteration_failed:" + redacted_failure_detail(exc)
            )
        finally:
            if artifixer_continuation and secret_install:
                secret_scrub = _scrub_artifixer_warm_secret_files(
                    session=session,
                    authority=authority,
                    install=secret_install,
                )
                if secret_scrub.get("secret_files_absent") is not True:
                    blockers.append("scene_configuration_artifixer_warm_secret_scrub_unproven")
                fresh_secret_probe = probe_fresh_ssh_secret_environment_absent(
                    {
                        "ssh_host": session["ssh_host"],
                        "ssh_port": session["ssh_port"],
                    },
                    attempt_dir=job / "artifixer_warm_fresh_ssh_secret_probe",
                )
                if (
                    fresh_secret_probe.get("fresh_ssh_runtime_secret_environment_absent")
                    is not True
                ):
                    blockers.append("scene_configuration_artifixer_warm_secret_scrub_unproven")
            if (job / "object_store_staging").is_dir() and remote_writer_absent:
                cleanup = cleanup_staged_wam_provider_objects(job / "object_store_staging")
                if cleanup.get("all_objects_absent") is not True:
                    blockers.append("scene_configuration_warm_object_store_cleanup_unproven")
        advanced = execution.pop("_validated_advanced_checkpoint", None)
        unsafe_checkpoint_blockers = [
            item for item in blockers if not str(item).startswith("provider_result_blocker:")
        ]
        if not unsafe_checkpoint_blockers and not isinstance(advanced, Mapping):
            blockers.append("scene_configuration_warm_advanced_checkpoint_missing")
            unsafe_checkpoint_blockers.append(
                "scene_configuration_warm_advanced_checkpoint_missing"
            )
        advanced_checkpoint_value: dict[str, Any] | None = None
        advanced_remote_checkpoint_root: str | None = None
        if not unsafe_checkpoint_blockers and isinstance(advanced, Mapping):
            try:
                advanced_checkpoint_value = validate_scene_configuration_diagnostic_checkpoint(
                    checkpoint_root=Path(str(advanced["checkpoint_root"]))
                )
                advanced_relative_root = PurePosixPath(
                    str(advanced["provider_output_relative_root"])
                )
                if advanced_relative_root.is_absolute() or ".." in advanced_relative_root.parts:
                    raise ValueError("unsafe advanced checkpoint root")
                advanced_remote_checkpoint_root = (
                    PurePosixPath(REMOTE_ROOT)
                    / "iterations"
                    / str(authority["iteration_id"])
                    / "output"
                    / advanced_relative_root
                ).as_posix()
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                blockers.append("scene_configuration_warm_advanced_checkpoint_state_invalid")
                unsafe_checkpoint_blockers.append(
                    "scene_configuration_warm_advanced_checkpoint_state_invalid"
                )
        safe_checkpoint_advanced = bool(
            not unsafe_checkpoint_blockers
            and isinstance(advanced, Mapping)
            and advanced_checkpoint_value is not None
            and advanced_remote_checkpoint_root is not None
        )
        result: dict[str, Any] = {
            "schema_version": ITERATION_RESULT_SCHEMA_VERSION,
            "status": "completed_diagnostic_only" if not blockers else "blocked_diagnostic_only",
            "iteration_id": authority["iteration_id"],
            "iteration_index": authority["iteration_index"],
            "session_digest": session["session_digest"],
            "authority_digest": authority["authority_digest"],
            "authorization_consumption": consumption,
            "provider_instance_id": session["provider_instance_id"],
            "provider_allocations_performed": 0,
            "instance_lifecycle_mutations_performed": 0,
            "remote_workload_dispatches_performed": remote_dispatches,
            "fix_overlay_dispatch_start_latency_seconds": (
                round(dispatch_start_latency_seconds, 6)
                if dispatch_start_latency_seconds is not None
                else None
            ),
            "stage_completion_wait_seconds": (
                round(stage_completion_seconds, 6) if stage_completion_seconds is not None else None
            ),
            "remote_setup_to_provider_entrypoint_seconds": (
                round(remote_setup_to_entrypoint_seconds, 6)
                if remote_setup_to_entrypoint_seconds is not None
                else None
            ),
            "remote_setup_to_provider_entrypoint_measured": (
                remote_setup_to_entrypoint_seconds is not None
            ),
            "runtime_clone_strategy": "reflink_auto_with_full_copy_fallback",
            "persistent_isaac_process_reused": False,
            "remote_progress": remote_progress,
            "remote_quiescence": remote_quiescence,
            "base_bundle_source_commit": session["source_commit"],
            "diagnostic_source_overlay_commit": authority["source_commit"],
            "source_overlay_manifest_digest": authority["source_overlay_manifest_digest"],
            "source_checkpoint_digest": authority["source_checkpoint_digest"],
            "advanced_checkpoint_digest": (
                advanced.get("checkpoint_digest") if isinstance(advanced, Mapping) else None
            ),
            "checkpoint_advanced_despite_stage_blocker": bool(
                safe_checkpoint_advanced and blockers
            ),
            "scientific_binding_digest": authority["scientific_binding_digest"],
            "carried_completed_stage_prefix_count": authority[
                "carried_completed_stage_prefix_count"
            ],
            "carried_completed_stage_ids": authority["carried_completed_stage_ids"],
            "carried_paid_model_stages": authority["carried_paid_model_stages"],
            "rerun_paid_model_stages": list(authority.get("rerun_paid_model_stages") or []),
            "fresh_openai_cost_scope_attestation_digests": dict(
                authority.get("fresh_openai_cost_scope_attestation_digests") or {}
            ),
            "warm_openai_external_service_spend_permitted": (
                authority.get("warm_openai_external_service_spend_permitted") is True
            ),
            "artifixer_warm_secret_install": secret_install,
            "artifixer_warm_secret_scrub": secret_scrub,
            "artifixer_warm_fresh_ssh_secret_probe": fresh_secret_probe,
            "continuing_spend_from_this_run": True,
            "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
            "dispatch": dispatch,
            "object_store_cleanup": cleanup,
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "arbitrary_command_permitted": False,
            "raw_secret_values_recorded": False,
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        write_json(job / f"{ITERATION_RESULT_SCHEMA_VERSION}.json", result)
        _mark_iteration_state(
            root=_absolute_directory(
                session_root, code="scene_configuration_warm_session_root_invalid"
            ),
            state=_read(
                _absolute_directory(
                    session_root, code="scene_configuration_warm_session_root_invalid"
                )
                / SESSION_STATE_NAME,
                code="scene_configuration_warm_session_state_invalid",
            ),
            authority=authority,
            status=(
                "ready"
                if not blockers
                else (
                    "teardown_required"
                    if (
                        not remote_writer_absent
                        or "scene_configuration_artifixer_warm_secret_scrub_unproven" in blockers
                        or (artifixer_continuation and not safe_checkpoint_advanced)
                    )
                    else "iteration_failed"
                )
            ),
            result_digest=result["result_digest"],
            advanced_checkpoint_digest=(
                str(advanced["checkpoint_digest"])
                if safe_checkpoint_advanced and isinstance(advanced, Mapping)
                else None
            ),
            advanced_checkpoint_prefix_count=(
                int(advanced_checkpoint_value["completed_stage_prefix_count"])
                if safe_checkpoint_advanced and advanced_checkpoint_value is not None
                else None
            ),
            advanced_checkpoint_stage_ids=(
                [
                    str(row["stage_id"])
                    for row in advanced_checkpoint_value["completed_stage_results"]
                ]
                if safe_checkpoint_advanced and advanced_checkpoint_value is not None
                else None
            ),
            advanced_remote_checkpoint_root=(
                advanced_remote_checkpoint_root if safe_checkpoint_advanced else None
            ),
        )
        return result


def close_scene_configuration_warm_session(
    *,
    session_root: str | Path,
    output_path: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    provider: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Destroy the exact retained instance and prove global provider-zero."""

    destination = Path(output_path).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_closeout_output_invalid"
        )
    with scene_configuration_warm_session_owner_lock(session_root):
        session, state = validate_scene_configuration_warm_session(
            session_root=session_root,
            require_iteration_window=False,
            allowed_state_statuses=frozenset(
                {
                    "ready",
                    "iteration_failed",
                    "iteration_running",
                    "teardown_required",
                }
            ),
        )
        if not execute:
            result = {
                "schema_version": CLOSEOUT_SCHEMA_VERSION,
                "status": "dry_run_ready",
                "session_digest": session["session_digest"],
                "provider_instance_id": session["provider_instance_id"],
                "instance_lifecycle_mutations_performed": 0,
                "provider_instance_absent": False,
                "global_provider_zero_proven": False,
                "continuing_spend_from_this_run": True,
                "diagnostic_only": True,
                "qualification_eligible": False,
                "raw_secret_values_recorded": False,
                "closeout_digest": "",
            }
            result["closeout_digest"] = canonical_digest(result, digest_field="closeout_digest")
            _write_exclusive(destination, result)
            return result
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="vast_provider_adapter",
            allocation_binding_digest=canonical_digest(
                scene_configuration_warm_closeout_allocation_binding(session=session)
            ),
            require_allocation_binding=True,
        )
        retained_provider = provider or VastRenderProvider()
        instance_id = str(session["provider_instance_id"])
        lifecycle_manifest_path = Path(session_root) / "retained_gpu_session_manifest.json"
        lifecycle_manifest = (
            _read(
                lifecycle_manifest_path,
                code="scene_configuration_warm_lifecycle_invalid",
            )
            if lifecycle_manifest_path.is_file()
            else {}
        )
        if lifecycle_manifest.get("state") != "teardown_requested":
            record_retained_gpu_state(
                session_root,
                "teardown_requested",
                evidence={
                    "provider": "vast",
                    "provider_instance_id": session["provider_instance_id"],
                    "session_digest": session["session_digest"],
                },
            )
        try:
            termination = retained_provider.terminate(instance_id)
        except Exception as exc:  # noqa: BLE001 - closeout must still inspect/seal
            termination = {
                "status": "terminate_failed",
                "error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
        observations: list[dict[str, Any]] = []
        deadline = monotonic() + max(1.0, float(timeout_seconds))
        consecutive_absent = 0
        while monotonic() < deadline and consecutive_absent < 2:
            try:
                observed = retained_provider.inspect(instance_id)
            except Exception as exc:  # noqa: BLE001 - seal blocked closeout
                observed = {
                    "status": "observation_failed",
                    "api_confirmed": False,
                    "provider_absence_confirmed": False,
                    "blockers": [redacted_failure_detail(exc)],
                    "raw_secret_values_recorded": False,
                }
            observations.append(dict(observed))
            if (
                observed.get("status") == "absent"
                and observed.get("provider_absence_confirmed") is True
                and observed.get("api_confirmed") is True
            ):
                consecutive_absent += 1
            else:
                consecutive_absent = 0
            if consecutive_absent < 2:
                sleep(2.0)
        global_inventories: list[dict[str, Any]] = []
        for index in range(2):
            try:
                inventory = retained_provider.billable_inventory(name_prefix="")
            except Exception as exc:  # noqa: BLE001 - preserve blocked closeout
                inventory = {
                    "status": "blocked",
                    "api_confirmed": False,
                    "blockers": [redacted_failure_detail(exc)],
                    "raw_secret_values_recorded": False,
                }
            global_inventories.append(dict(inventory))
            if index == 0:
                sleep(2.0)
        exact_absent = consecutive_absent >= 2
        global_zero = bool(
            len(global_inventories) == 2
            and all(
                row.get("api_confirmed") is True
                and row.get("live_resource_count") == 0
                and row.get("resources") == []
                for row in global_inventories
            )
        )
        blockers: list[str] = []
        if not exact_absent:
            blockers.append("scene_configuration_warm_provider_absence_unproven")
        if not global_zero:
            blockers.append("scene_configuration_warm_global_provider_zero_unproven")
        watchdog_cancel: dict[str, Any] = {}
        if exact_absent and global_zero:
            watchdog_cancel = write_owner_teardown_cancel_request(
                root=Path(str(session["watchdog_out_dir"])),
                pod_name_prefix=str(session["watchdog_pod_name_prefix"]),
                provider_name="vast",
                instance_id=instance_id,
            )
            record_retained_gpu_state(
                session_root,
                "provider_absent",
                evidence={
                    "provider": "vast",
                    "provider_instance_id": session["provider_instance_id"],
                    "global_provider_zero_proven": True,
                },
            )
        result = {
            "schema_version": CLOSEOUT_SCHEMA_VERSION,
            "status": "completed" if not blockers else "blocked",
            "session_digest": session["session_digest"],
            "provider_instance_id": session["provider_instance_id"],
            "termination": termination,
            "provider_absence_observations": observations,
            "global_billable_inventory_observations": global_inventories,
            "watchdog_cancel_request": watchdog_cancel,
            "instance_lifecycle_mutations_performed": 1,
            "provider_instance_absent": exact_absent,
            "global_provider_zero_proven": global_zero,
            "continuing_spend_from_this_run": not (exact_absent and global_zero),
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "raw_secret_values_recorded": False,
            "blockers": blockers,
            "closeout_digest": "",
        }
        result["closeout_digest"] = canonical_digest(result, digest_field="closeout_digest")
        _write_exclusive(destination, result)
        if not blockers:
            terminal_state = {
                **dict(state),
                "status": "closed",
                "continuing_spend": False,
                "active_iteration_id": None,
                "active_iteration_authority_digest": None,
                "closeout_digest": result["closeout_digest"],
                "state_digest": "",
            }
            terminal_state["state_digest"] = canonical_digest(
                terminal_state, digest_field="state_digest"
            )
            _write_state(
                _absolute_directory(
                    session_root,
                    code="scene_configuration_warm_session_root_invalid",
                ),
                terminal_state,
            )
        return result


__all__ = [
    "CLOSEOUT_SCHEMA_VERSION",
    "ITERATION_AUTHORITY_SCHEMA_VERSION",
    "ITERATION_RESULT_SCHEMA_VERSION",
    "OVERLAY_RECEIPT_SCHEMA_VERSION",
    "OVERLAY_SCHEMA_VERSION",
    "SESSION_AUTHORITY_SCHEMA_VERSION",
    "SESSION_SCHEMA_VERSION",
    "SceneConfigurationWarmDiagnosticError",
    "build_scene_configuration_warm_source_overlay",
    "close_scene_configuration_warm_session",
    "materialize_scene_configuration_warm_iteration_authority",
    "materialize_scene_configuration_warm_session",
    "materialize_scene_configuration_warm_session_authority",
    "run_scene_configuration_warm_iteration",
    "scene_configuration_warm_closeout_allocation_binding",
    "scene_configuration_warm_iteration_allocation_binding",
    "scene_configuration_warm_session_owner_lock",
    "validate_scene_configuration_warm_iteration_authority",
    "validate_scene_configuration_warm_session",
    "validate_scene_configuration_warm_session_authority",
    "validate_scene_configuration_warm_source_overlay",
]
