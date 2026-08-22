"""Attach one digest-bound controls bundle to a retained Vast Arena worker."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

from .adp_isaac_lab_arena_vast import (
    _extract_provider_output,
    _vast_authority_environment,
)
from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .gpu_render_providers import (
    _latest_redacted_vast_ssh_output_fields,
    _validated_vast_known_hosts_pin,
    enroll_vast_ssh_host_key,
)
from .native_task_arena_controls_bundle import RESULT_FILENAME
from .native_task_arena_warm_authority import (
    consume_native_task_arena_warm_authority_once,
    validate_native_task_arena_warm_attempt_authority,
    validate_native_task_arena_warm_session,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .task_evaluation_artifact_manifest import seal_lane_terminal_artifacts
from .vast_provider_adapter import (
    DEFAULT_VAST_API_KEY_FILE,
    VAST_API_GATE_ENV,
    VAST_API_KEY_FILE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    _api_json,
    _instance_liveness_from_payload,
)
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


RESULT_SCHEMA_VERSION = "native_task_arena_warm_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/adp/native-task-arena/warm-controls"
MINIMUM_REMOTE_TIMEOUT_SECONDS = 600
POLL_SECONDS = 10
DEFAULT_SSH_IDENTITY_FILE = "~/.ssh/id_ed25519"
MAX_REMOTE_SCRIPT_BYTES = 128 * 1024


def _truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_api_key() -> str:
    path = Path(
        os.getenv(VAST_API_KEY_FILE_ENV, DEFAULT_VAST_API_KEY_FILE)
    ).expanduser()
    if path.is_symlink() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_url(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value.startswith("https://"):
        raise ValueError("native_task_arena_warm_signed_url_invalid")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _remote_attempt_script(
    *,
    bundle_url: str,
    output_put_url: str,
    bundle_sha256: str,
    runtime_dependency_sha256: str,
    runtime_dependency_size_bytes: int,
    attempt_key: str,
) -> str:
    """Build the small remote dispatcher; the 4.4 GB layer must already exist."""

    values = {
        "BUNDLE_URL": bundle_url,
        "OUTPUT_PUT_URL": output_put_url,
        "BUNDLE_SHA": bundle_sha256,
        "DEPENDENCY_SHA": runtime_dependency_sha256,
        "DEPENDENCY_SIZE": str(runtime_dependency_size_bytes),
        "ATTEMPT_KEY": attempt_key,
    }
    assignments = "\n".join(
        f"{key}={shlex.quote(value)}" for key, value in values.items()
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
{assignments}
exec 9>/workspace/native_task_arena_warm_attempt.lock
if ! flock -n 9; then
  echo BLUEPRINT_ARENA_WARM_BLOCKED:another_attempt_running
  exit 73
fi
ATTEMPT_DIR="/workspace/native_task_arena_warm_attempts/$ATTEMPT_KEY"
rm -rf "$ATTEMPT_DIR"
mkdir -p "$ATTEMPT_DIR/bundle" "$ATTEMPT_DIR/runtime_output"
echo BLUEPRINT_ARENA_WARM_ATTEMPT_STARTED:$ATTEMPT_KEY
curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 2 \
  "$BUNDLE_URL" -o "$ATTEMPT_DIR/bundle.zip"
ACTUAL_BUNDLE_SHA="sha256:$(sha256sum "$ATTEMPT_DIR/bundle.zip" | cut -d' ' -f1)"
if [ "$ACTUAL_BUNDLE_SHA" != "$BUNDLE_SHA" ]; then
  echo BLUEPRINT_ARENA_WARM_BLOCKED:bundle_digest_mismatch
  exit 74
fi
/isaac-sim/python.sh -m zipfile -e "$ATTEMPT_DIR/bundle.zip" "$ATTEMPT_DIR/bundle"
DEPENDENCY_HEX="${{DEPENDENCY_SHA#sha256:}}"
CACHED_DEPENDENCY="/workspace/native_task_runtime_dependency_cache/$DEPENDENCY_HEX.zip"
DEPENDENCY_PACKET="$ATTEMPT_DIR/bundle/provider_runtime/native_task_runtime_sources/native_task_runtime_sources.zip"
if [ ! -f "$CACHED_DEPENDENCY" ]; then
  echo BLUEPRINT_ARENA_WARM_BLOCKED:dependency_cache_miss
  exit 75
fi
ACTUAL_DEPENDENCY_SHA="sha256:$(sha256sum "$CACHED_DEPENDENCY" | cut -d' ' -f1)"
ACTUAL_DEPENDENCY_SIZE="$(wc -c < "$CACHED_DEPENDENCY" | tr -d ' ')"
if [ "$ACTUAL_DEPENDENCY_SHA" != "$DEPENDENCY_SHA" ] || \
   [ "$ACTUAL_DEPENDENCY_SIZE" != "$DEPENDENCY_SIZE" ]; then
  echo BLUEPRINT_ARENA_WARM_BLOCKED:dependency_cache_identity_mismatch
  exit 76
fi
ln -s "$CACHED_DEPENDENCY" "$DEPENDENCY_PACKET"
echo BLUEPRINT_ARENA_WARM_DEPENDENCY_CACHE_HIT:$DEPENDENCY_SHA
export BLUEPRINT_ADP_ARENA_OUTPUT_DIR="$ATTEMPT_DIR/runtime_output"
set +e
bash "$ATTEMPT_DIR/bundle/provider_runtime/run_adp_arena_provider_runtime.sh"
PROVIDER_RC=$?
set -e
echo BLUEPRINT_ARENA_WARM_ENTRYPOINT_EXIT_CODE:$PROVIDER_RC
export BLUEPRINT_ARENA_WARM_ATTEMPT_DIR="$ATTEMPT_DIR"
/isaac-sim/python.sh - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["BLUEPRINT_ARENA_WARM_ATTEMPT_DIR"])
output = root / "provider_runtime_output.zip"
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((root / "runtime_output").rglob("*")):
        if path.is_file() and path.stat().st_size <= 100_000_000:
            archive.write(path, path.relative_to(root / "runtime_output").as_posix())
print(f"BLUEPRINT_ARENA_WARM_OUTPUT_ZIP_WRITTEN:{{output.stat().st_size}}")
PY
curl -fsS --http1.1 --retry 5 --retry-all-errors --retry-delay 2 \
  -X PUT -H 'Content-Type: application/zip' \
  --upload-file "$ATTEMPT_DIR/provider_runtime_output.zip" "$OUTPUT_PUT_URL"
echo BLUEPRINT_ARENA_WARM_PROVIDER_OUTPUT_UPLOAD_OK
"""


def _run_pinned_ssh(
    *,
    session: Mapping[str, Any],
    known_hosts_file: str | Path,
    remote_argv: list[str],
    stdin: bytes | None = None,
    identity_file: str | Path = DEFAULT_SSH_IDENTITY_FILE,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run one bounded command on the retained worker over strict pinned SSH."""

    host = str(session.get("ssh_host") or "").strip()
    try:
        port = int(session.get("ssh_port") or 0)
    except (TypeError, ValueError):
        port = 0
    pin = (
        _validated_vast_known_hosts_pin(known_hosts_file, host=host, port=port)
        if host and 0 < port <= 65535
        else None
    )
    if pin is None:
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_ssh_host_pin_invalid"],
            "raw_secret_values_recorded": False,
        }
    known_hosts, known_hosts_sha256 = pin
    identity = Path(identity_file).expanduser()
    try:
        identity_mode = identity.stat().st_mode & 0o777
    except OSError:
        identity_mode = -1
    if (
        identity.is_symlink()
        or not identity.is_file()
        or identity_mode < 0
        or identity_mode & 0o077
    ):
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_ssh_identity_invalid"],
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    if not remote_argv or any("\x00" in value for value in remote_argv):
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_ssh_command_invalid"],
            "known_hosts_sha256": known_hosts_sha256,
            "raw_secret_values_recorded": False,
        }
    timeout = min(300.0, max(1.0, float(timeout_seconds)))
    command = [
        "ssh",
        "-i",
        str(identity.resolve(strict=True)),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=2",
        "--",
        f"root@{host}",
        shlex.join(remote_argv),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            input=stdin,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_ssh_timeout"],
            **_latest_redacted_vast_ssh_output_fields(
                stdout=exc.stdout, stderr=exc.stderr
            ),
            "known_hosts_sha256": known_hosts_sha256,
            "strict_host_key_checking": True,
            "raw_secret_values_recorded": False,
        }
    returncode = int(completed.returncode)
    return {
        "status": "completed" if returncode == 0 else "blocked",
        "blockers": (
            [] if returncode == 0 else ["native_task_arena_warm_ssh_command_failed"]
        ),
        "returncode": returncode,
        **_latest_redacted_vast_ssh_output_fields(
            stdout=completed.stdout, stderr=completed.stderr
        ),
        "known_hosts_sha256": known_hosts_sha256,
        "strict_host_key_checking": True,
        "raw_secret_values_recorded": False,
    }


def _dispatch_warm_script_over_ssh(
    *,
    job: Path,
    session: Mapping[str, Any],
    remote_script: str,
    attempt_key: str,
) -> dict[str, Any]:
    """Stream the dispatcher over SSH; Vast's execute API caps commands at 512 B."""

    if not re.fullmatch(r"[0-9a-f]{16}", attempt_key):
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_attempt_key_invalid"],
            "raw_secret_values_recorded": False,
        }
    script_bytes = remote_script.encode("utf-8")
    if (
        not remote_script.startswith("#!/usr/bin/env bash\n")
        or len(script_bytes) > MAX_REMOTE_SCRIPT_BYTES
    ):
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_warm_remote_script_invalid"],
            "raw_secret_values_recorded": False,
        }
    enrollment = enroll_vast_ssh_host_key(
        session,
        attempt_dir=job / "vast_ssh_trust",
        timeout_seconds=15,
    )
    if enrollment.get("status") != "enrolled":
        return {
            "status": "blocked",
            "blockers": list(enrollment.get("blockers") or [])
            or ["native_task_arena_warm_ssh_host_enrollment_failed"],
            "host_key_enrollment": enrollment,
            "raw_secret_values_recorded": False,
        }
    remote_dir = f"/workspace/native_task_arena_warm_attempts/{attempt_key}"
    wrapper = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(remote_dir)}; "
        f"cat > {shlex.quote(remote_dir + '/run.sh')}; "
        f"chmod 700 {shlex.quote(remote_dir + '/run.sh')}; "
        f"nohup bash {shlex.quote(remote_dir + '/run.sh')} > "
        f"{shlex.quote(remote_dir + '/run.log')} 2>&1 < /dev/null & "
        "pid=$!; case \"$pid\" in ''|*[!0-9]*) exit 78;; esac; printf '%s\\n' \"$pid\""
    )
    result = _run_pinned_ssh(
        session=session,
        known_hosts_file=str(enrollment["known_hosts_file"]),
        remote_argv=["/bin/bash", "-c", wrapper],
        stdin=script_bytes,
        timeout_seconds=30,
    )
    stdout = str(result.get("stdout") or "")
    pid_match = re.fullmatch(r"([1-9][0-9]*)\n?", stdout)
    if result.get("status") != "completed" or pid_match is None:
        result["status"] = "blocked"
        result["blockers"] = sorted(
            set(
                list(result.get("blockers") or [])
                + ["native_task_arena_warm_dispatch_pid_unproven"]
            )
        )
    else:
        result["remote_pid"] = int(pid_match.group(1))
    result["host_key_enrollment"] = enrollment
    result["transport"] = "strict_pinned_ssh_stdin.v1"
    (job / "warm_dispatch.log").write_text(
        (
            f"status={result.get('status')}\n"
            f"remote_pid={result.get('remote_pid', '')}\n"
            f"transport={result['transport']}\n"
        ),
        encoding="utf-8",
    )
    return result


def _fetch_warm_runtime_log_over_ssh(
    *,
    job: Path,
    session: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    attempt_key: str,
) -> tuple[dict[str, Any], str]:
    enrollment = dict(dispatch.get("host_key_enrollment") or {})
    remote_log_path = (
        f"/workspace/native_task_arena_warm_attempts/{attempt_key}/run.log"
    )
    result = _run_pinned_ssh(
        session=session,
        known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
        remote_argv=["tail", "-n", "500", "--", remote_log_path],
        timeout_seconds=30,
    )
    text = str(result.get("stdout") or "")
    (job / "warm_runtime.log").write_text(text, encoding="utf-8")
    return result, text


def _download_when_ready(
    *, url: str, destination: Path, deadline_monotonic: float
) -> bool:
    while time.monotonic() < deadline_monotonic:
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BlueprintArenaWarm/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            if payload:
                destination.write_bytes(payload)
                return True
        except urllib.error.HTTPError as exc:
            if exc.code not in {403, 404}:
                raise
        except urllib.error.URLError:
            pass
        time.sleep(POLL_SECONDS)
    return False


def _close_warm_instance(
    *, instance_id: int, api_key: str, timeout_seconds: int = 180
) -> dict[str, Any]:
    """Destroy the retained worker after controls passes and prove its absence."""

    status_code, _response = _api_json(
        method="DELETE",
        path=f"/instances/{instance_id}/",
        api_key=api_key,
        timeout_seconds=30,
    )
    observations: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            read_status, payload = _api_json(
                method="GET",
                path=f"/instances/{instance_id}/",
                api_key=api_key,
                timeout_seconds=30,
            )
            liveness = (
                _instance_liveness_from_payload(payload, instance_id=instance_id)
                if 200 <= read_status < 300
                else {"status": "unknown", "observed": False}
            )
        except urllib.error.HTTPError as exc:
            # Vast may report a destroyed instance either as an empty 200
            # payload or as a terminal 404.  Both are provider observations of
            # absence; other HTTP failures remain unknown and fail closed.
            liveness = (
                {"status": "absent", "observed": True}
                if exc.code == 404
                else {"status": "unknown", "observed": False}
            )
        except (OSError, ValueError, urllib.error.URLError):
            liveness = {"status": "unknown", "observed": False}
        observations.append(
            {
                "observed_at": utc_now_iso(),
                "status": liveness.get("status"),
                "observed": liveness.get("observed"),
            }
        )
        if liveness.get("status") == "absent" and liveness.get("observed") is True:
            return {
                "status": "completed",
                "destroy_http_status_code": status_code,
                "provider_instance_absent": True,
                "continuing_spend_from_this_run": False,
                "observations": observations,
            }
        time.sleep(5)
    return {
        "status": "blocked",
        "destroy_http_status_code": status_code,
        "provider_instance_absent": False,
        "continuing_spend_from_this_run": True,
        "observations": observations,
        "blockers": ["native_task_arena_warm_instance_absence_unproven"],
    }


def _execute_staged_warm_attempt(
    *,
    job: Path,
    staging_dir: Path,
    prepared_bundle: Mapping[str, Any],
    session: Mapping[str, Any],
    instance_id: int,
    api_key: str,
) -> dict[str, Any]:
    bundle_url = _read_url(staging_dir / "provider_bundle_url.txt")
    output_put_url = _read_url(staging_dir / "provider_output_put_url.txt")
    output_get_url = _read_url(staging_dir / "provider_output_get_url.txt")
    runtime_source = prepared_bundle.get("runtime_source_packet") or {}
    attempt_key = str(prepared_bundle["bundle_sha256"])[7:23]
    remote_script = _remote_attempt_script(
        bundle_url=bundle_url,
        output_put_url=output_put_url,
        bundle_sha256=str(prepared_bundle["bundle_sha256"]),
        runtime_dependency_sha256=str(runtime_source["packet_sha256"]),
        runtime_dependency_size_bytes=int(runtime_source["packet_size_bytes"]),
        attempt_key=attempt_key,
    )
    dispatch = _dispatch_warm_script_over_ssh(
        job=job,
        session=session,
        remote_script=remote_script,
        attempt_key=attempt_key,
    )
    if dispatch.get("status") != "completed":
        return {
            "dispatch": dispatch,
            "remote_log": {},
            "runtime_log_text": "",
            "extracted": {},
            "elapsed": 0.0,
            "blockers": list(dispatch.get("blockers") or [])
            or ["native_task_arena_warm_dispatch_failed"],
        }
    output_zip = job / "vast_provider_runtime_output.zip"
    remaining_seconds = float(session["watchdog_deadline_epoch"]) - time.time() - 120
    timeout_seconds = max(
        MINIMUM_REMOTE_TIMEOUT_SECONDS,
        min(1800, int(remaining_seconds)),
    )
    started = time.monotonic()
    output_ready = _download_when_ready(
        url=output_get_url,
        destination=output_zip,
        deadline_monotonic=started + timeout_seconds,
    )
    remote_log, runtime_log_text = _fetch_warm_runtime_log_over_ssh(
        job=job,
        session=session,
        dispatch=dispatch,
        attempt_key=attempt_key,
    )
    extracted = (
        _extract_provider_output(
            output_zip,
            job / "immutable_execution",
            result_name=RESULT_FILENAME,
            blocker_prefix="native_task_arena_warm_controls",
        )
        if output_ready
        else {"execution": {}, "blockers": ["native_task_arena_warm_output_timeout"]}
    )
    return {
        "dispatch": dispatch,
        "remote_log": remote_log,
        "runtime_log_text": runtime_log_text,
        "extracted": extracted,
        "elapsed": max(0.0, time.monotonic() - started),
        "blockers": [],
    }


def run_native_task_arena_warm_controls_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    warm_session: Mapping[str, Any],
    warm_attempt_authority: Mapping[str, Any] | None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    object_store_key_prefix: str = DEFAULT_KEY_PREFIX,
    close_on_success: bool = True,
) -> dict[str, Any]:
    """Run through the same scoped Vast mutation gates as a cold Arena run."""

    kwargs = {
        "job_dir": job_dir,
        "prepared_bundle": prepared_bundle,
        "warm_session": warm_session,
        "warm_attempt_authority": warm_attempt_authority,
        "paid_resource_admission_grant": paid_resource_admission_grant,
        "execute": execute,
        "object_store_key_prefix": object_store_key_prefix,
        "close_on_success": close_on_success,
    }
    if not execute:
        return _run_native_task_arena_warm_controls_vast(**kwargs)
    # The cold Arena path deliberately opens these process-local gates only
    # after paid-resource admission and restores them afterwards. Warm attach
    # has the same provider GET/execute/DELETE surface, so it must use the same
    # scoped authority instead of depending on persistent service env flags.
    with _vast_authority_environment():
        return _run_native_task_arena_warm_controls_vast(**kwargs)


def _run_native_task_arena_warm_controls_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    warm_session: Mapping[str, Any],
    warm_attempt_authority: Mapping[str, Any] | None,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    object_store_key_prefix: str = DEFAULT_KEY_PREFIX,
    close_on_success: bool = True,
) -> dict[str, Any]:
    """Run one controls bundle on an existing instance; allocate no provider."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    generated = utc_now_iso()
    bundle_path = Path(str(prepared_bundle.get("bundle_path") or "")).resolve()
    if (
        prepared_bundle.get("schema_version")
        != "native_task_arena_provider_bundle.v1"
        or prepared_bundle.get("execution_mode") != "controls"
        or prepared_bundle.get("expected_output_filename") != RESULT_FILENAME
        or not bundle_path.is_file()
        or _file_sha256(bundle_path) != prepared_bundle.get("bundle_sha256")
    ):
        raise ValueError("native_task_arena_warm_prepared_bundle_invalid")
    session = validate_native_task_arena_warm_session(
        warm_session, prepared_bundle=prepared_bundle
    )
    authority = (
        validate_native_task_arena_warm_attempt_authority(
            warm_attempt_authority,
            warm_session=session,
            prepared_bundle=prepared_bundle,
        )
        if warm_attempt_authority is not None
        else None
    )
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "dry_run_ready",
            "provider_allocations_performed": 0,
            "warm_session_digest": session["session_digest"],
            "bundle_sha256": prepared_bundle.get("bundle_sha256"),
            "blockers": [],
        }
        write_json(job / "native_task_arena_warm_vast_result.json", result)
        return result
    if authority is None or paid_resource_admission_grant is None:
        raise ValueError("native_task_arena_warm_paid_authority_missing")
    api_key = _read_api_key()
    if (
        not _truthy(VAST_API_GATE_ENV)
        or not _truthy(VAST_INSTANCE_LAUNCH_GATE_ENV)
        or not api_key
    ):
        raise ValueError("native_task_arena_warm_vast_api_gate_closed")
    # Do not burn the single-use authority on a local credential/gate defect.
    # Consumption immediately precedes the first provider request.
    consumption = consume_native_task_arena_warm_authority_once(authority)
    if consumption.get("status") != "consumed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "provider_allocations_performed": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
    instance_id = int(session["instance_id"])
    status_code, instance_payload = _api_json(
        method="GET",
        path=f"/instances/{instance_id}/",
        api_key=api_key,
        timeout_seconds=30,
    )
    liveness = (
        _instance_liveness_from_payload(instance_payload, instance_id=instance_id)
        if 200 <= status_code < 300
        else {"status": "unknown", "observed": False}
    )
    if (
        liveness.get("status") != "running"
        or liveness.get("ssh_host") != session.get("ssh_host")
        or liveness.get("ssh_port") != session.get("ssh_port")
    ):
        raise ValueError("native_task_arena_warm_instance_identity_not_live")
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix=object_store_key_prefix,
        # The signed attempt URLs must not outlive the retained worker's
        # independent watchdog.  Session validation leaves at least 900 s.
        expiration_seconds=max(
            600, int(float(session["watchdog_deadline_epoch"]) - time.time())
        ),
        generated_at=generated,
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "provider_allocations_performed": 0,
            "authorization_consumption": consumption,
            "blockers": list(staging.get("blockers") or []),
        }
    try:
        outcome = _execute_staged_warm_attempt(
            job=job,
            staging_dir=staging_dir,
            prepared_bundle=prepared_bundle,
            session=session,
            instance_id=instance_id,
            api_key=api_key,
        )
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
        outcome = {
            "dispatch": {},
            "remote_log": {},
            "runtime_log_text": "",
            "extracted": {},
            "elapsed": 0.0,
            "blockers": [
                "native_task_arena_warm_dispatch_failed:"
                + redacted_failure_detail(exc)
            ],
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = dict(outcome.get("extracted") or {})
    runtime_log_text = str(outcome.get("runtime_log_text") or "")
    execution = dict(extracted.get("execution") or {})
    blockers = list(outcome.get("blockers") or []) + list(
        extracted.get("blockers") or []
    )
    if "BLUEPRINT_ARENA_WARM_DEPENDENCY_CACHE_HIT:" not in runtime_log_text:
        blockers.append("native_task_arena_warm_dependency_cache_hit_unproven")
    if "BLUEPRINT_ARENA_WARM_PROVIDER_OUTPUT_UPLOAD_OK" not in runtime_log_text:
        blockers.append("native_task_arena_warm_output_upload_unproven")
    if execution.get("status") != "completed":
        blockers.extend(
            execution.get("blockers") or ["native_task_arena_warm_controls_not_completed"]
        )
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("native_task_arena_warm_object_store_cleanup_unproven")
    closeout: dict[str, Any] = {
        "status": "retained_for_next_attempt",
        "provider_instance_absent": False,
        "continuing_spend_from_this_run": True,
    }
    if not blockers and close_on_success:
        closeout = _close_warm_instance(instance_id=instance_id, api_key=api_key)
        if closeout.get("provider_instance_absent") is not True:
            blockers.extend(closeout.get("blockers") or [])
    elapsed = float(outcome.get("elapsed") or 0.0)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "provider_allocations_performed": 0,
        "provider_instance_id": instance_id,
        "warm_session_digest": session["session_digest"],
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "authorization_consumption": consumption,
        "dispatch": outcome.get("dispatch"),
        "remote_log_fetch": outcome.get("remote_log"),
        "native_control_result_path": extracted.get("result_path"),
        "runtime_seconds": round(elapsed, 3),
        "incremental_cost_upper_bound_usd": round(
            float(session.get("max_hourly_rate_usd") or 0.0) * elapsed / 3600,
            6,
        ),
        "warm_session_closeout": closeout,
        "continuing_spend_from_this_run": closeout.get(
            "continuing_spend_from_this_run"
        ),
        "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
        "object_store_cleanup": cleanup,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    # A warm attachment allocates no *new* GPU, but it still controls a paid
    # provider resource.  Normalize its exact instance and close/retention
    # decision into the same terminal-artifact convention as every allocating
    # transport, so the production reconciler can account for continuing spend.
    provider_run = job / "vast_provider_run"
    ensure_dir(provider_run)
    write_json(provider_run / "vast_provider_adapter_result.json", result)
    if closeout.get("provider_instance_absent") is True:
        teardown = {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": utc_now_iso(),
            "status": "completed",
            "vast_instance_ids": [instance_id],
            "teardown_actions_performed": [
                {
                    "action": "destroy_retained_instance",
                    "instance_id": instance_id,
                    "http_status_code": closeout.get(
                        "destroy_http_status_code"
                    ),
                }
            ],
            "provider_instance_absent": True,
            "continuing_spend_from_this_run": False,
            "zero_continuing_spend_scope": (
                "The exact retained Vast instance was destroyed and provider "
                "absence was observed after the successful warm controls attempt."
            ),
        }
    else:
        teardown = {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": utc_now_iso(),
            "status": "retained_owned",
            "vast_instance_ids": [instance_id],
            "teardown_actions_performed": [],
            "provider_instance_absent": False,
            "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
            "continuing_spend_from_this_run": True,
            "zero_continuing_spend_scope": None,
        }
    write_json(provider_run / "vast_teardown_manifest.json", teardown)
    result = seal_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane="native_task_arena_warm_controls",
        binding={
            "provider": "vast",
            "provider_instance_id": instance_id,
            "warm_session_digest": session["session_digest"],
            "bundle_sha256": prepared_bundle.get("bundle_sha256"),
            "provider_allocations_performed": 0,
        },
    )
    write_json(job / "native_task_arena_warm_vast_result.json", result)
    return result


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "run_native_task_arena_warm_controls_vast",
]
