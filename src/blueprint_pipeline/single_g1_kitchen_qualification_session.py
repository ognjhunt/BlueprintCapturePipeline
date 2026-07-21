"""Allocator-owned persistent Vast qualification session for one G1 kitchen episode.

The ordinary single-episode lane is intentionally terminal: it destroys the GPU
after one attempt.  This module keeps the exact same sealed image, bundle, worker
bootstrap, readiness gates, and semantic-success contracts, but installs that
bootstrap behind a fixed SSH control script.  Component failure therefore ends an
attempt, not the provider allocation.  The independent hard-TTL watchdog and the
open pending-teardown record remain authoritative while the allocation is retained.

No public provider-adapter launcher is exposed here.  Every action is routed by
``python -m blueprint_pipeline.paid_resource_allocator gpu-canary``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .gear_sonic_isaac_dds_bridge import (
    BRIDGE_BINARY_PATH,
    BRIDGE_HEARTBEAT_PATH,
    BRIDGE_LOG_PATH,
    BRIDGE_REQUIRED_ENV,
    NATIVE_BRIDGE_SOURCE_SHA256,
    SNAPSHOT_DEFAULT_PATH,
    SNAPSHOT_ENV,
)
from .gpu_render_providers import (
    VAST_SSH_HOST_KEY_FINGERPRINT_NAME,
    get_render_provider,
)
from .groot_oscar_digitalocean_closed_loop_job import build_launch_spec
from .groot_oscar_runpod_watchdog import arm_watchdog, terminate_canary_resources
from .groot_oscar_episode_review import _collect_isaac_execution_frames
from .g1_microwave_groot_finetune_component import (
    REMOTE_FINAL_CHECKPOINT,
    build_finetune_component,
)
from .g1_kitchen_leaf_evidence import load_attempt_identity
from .g1_kitchen_proof_row_validation import (
    WORKER_PROOF_ROW_SPECS,
    load_attestation_pins,
    validate_worker_proof_rows,
)
from .isaac_particlefield_render_job import _extract_provider_output_snapshot
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission,
)
from .qualification_control_admission import admit_qualification_control_mutation
from .single_g1_kitchen_episode_runpod import (
    BUNDLE_SHA256,
    GPU_TYPES,
    ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV,
    MAX_HOURLY_RATE_USD,
    RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV,
    TASK_PROMPT,
    VAST_BOOTSTRAP_SHA256_ENV,
    VAST_BOOTSTRAP_URL_ENV,
    VAST_MIN_RELIABILITY,
    VAST_PREFERRED_GPU_KEYWORDS,
    VAST_REQUIRE_KNOWN_SUPPORTED_ISAAC_DRIVER,
    WALL_SECONDS,
    _load_single_episode_inputs,
    _materialize_launch_session_nonce,
    _read_secret_url_file,
    _require_signed_output_staging_proof,
    _validate_collected_final_review,
    _vast_remote_bootstrap_script,
    _vast_signed_bootstrap_downloader_script,
    _write_materialized_inputs,
)
from .safe_outbound_http import presigned_transfer_policy
from .safe_outbound_http import request as safe_http_request
from .single_g1_kitchen_qualification_admission import (
    qualification_pre_spend_preflight,
    write_standard_artifacts,
)
from .single_g1_kitchen_qualification_contract import (
    build_release_binding as _release_binding,
    qualification_gate_matrix,
    session_claim_boundary as _session_claim_boundary,
    valid_image_binding,
    valid_source_commit,
)


SCHEMA_VERSION = "single_g1_kitchen_qualification_session.v1"
BOUND_REQUEST_SCHEMA_VERSION = "single_g1_kitchen_qualification_bound_request.v1"
PREFLIGHT_SCHEMA_VERSION = "single_g1_kitchen_qualification_preflight.v1"
REFRESH_PAYLOAD_SCHEMA_VERSION = "single_g1_kitchen_qualification_refresh_payload.v1"
REFRESH_REQUEST_SCHEMA_VERSION = "single_g1_kitchen_qualification_refresh_request.v1"
OVERLAY_BINDING_SCHEMA_VERSION = "single_g1_kitchen_qualification_overlay_binding.v1"
IMMUTABLE_BINDING_SCHEMA_VERSION = "single_g1_kitchen_qualification_immutable_binding.v1"
CONTROL_CONTRACT_VERSION = "fixed_qualification_control_script.v6"
REFRESH_COMPATIBLE_CONTROL_CONTRACT_VERSIONS = frozenset(
    {
        "fixed_qualification_control_script.v4",
        "fixed_qualification_control_script.v5",
        CONTROL_CONTRACT_VERSION,
    }
)
PROBE_KIND = "single-kitchen-qualification"
SESSION_ACTIONS = (
    "allocate",
    "refresh-bootstrap",
    "run",
    "collect",
    "status",
    "tail",
    "gpu-status",
    "restart-component",
    "stop-component",
    "teardown",
)
COMPONENT_ALIASES = {
    "bootstrap": "bootstrap",
    "episode": "episode",
    "groot": "groot_server",
    "controller": "gear_sonic_controller",
    "isaac": "isaac_task_executor",
    "bridge": "gear_sonic_isaac_dds_bridge",
    "finetune": "groot_microwave_finetune",
}
RESTARTABLE_COMPONENTS = ("groot", "controller", "isaac", "bridge")
STOPPABLE_COMPONENTS = (*RESTARTABLE_COMPONENTS, "finetune")
SESSION_MANIFEST_NAME = "qualification_session.json"
QUALIFICATION_BOOTSTRAP_NAME = "provider_qualification_bootstrap.sh"
QUALIFICATION_REFRESH_PAYLOAD_NAME = "qualification_refresh_payload.json"
REMOTE_ROOT = "/workspace/runtime_overlay"
REMOTE_CONTROL_SCRIPT = f"{REMOTE_ROOT}/blueprint_qualification_control.sh"
REMOTE_REFRESH_INSTALLER = f"{REMOTE_ROOT}/blueprint_qualification_refresh.py"
REMOTE_REVISIONS_DIR = f"{REMOTE_ROOT}/qualification_revisions"
REMOTE_ACTIVE_OVERLAY = f"{REMOTE_ROOT}/qualification_active"
REMOTE_EPISODE_BOOTSTRAP = f"{REMOTE_ACTIVE_OVERLAY}/qualification_episode_bootstrap.sh"
REMOTE_BINDING = f"{REMOTE_ACTIVE_OVERLAY}/qualification_overlay_binding.json"
REMOTE_IMMUTABLE_BINDING = f"{REMOTE_ROOT}/qualification_immutable_binding.json"
REMOTE_STATE_DIR = f"{REMOTE_ROOT}/qualification_state"
REMOTE_RUNTIME_ENV = f"{REMOTE_STATE_DIR}/qualification_runtime_env.sh"
DEFAULT_IDENTITY_FILE = "~/.ssh/id_ed25519"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_COLLECTED_JSON_BYTES = 128 * 1024 * 1024
MAX_TAIL_LINES = 2_000
MAX_PROVIDER_OUTPUT_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXTRACTED_PROVIDER_OUTPUT_BYTES = 8 * 1024 * 1024 * 1024
MAX_PROVIDER_OUTPUT_MEMBERS = 100_000
COLLECTIONS_DIR_NAME = "qualification_collections"
SSH_READY_TIMEOUT_SECONDS = 600
SSH_READY_POLL_SECONDS = 5
LANE = "single_g1_kitchen_qualification"
NAME_PREFIX_ROOT = "blueprint-groot-oscar-canary-qualification-"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _private_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a session-bearing artifact with mode 0600."""

    ensure_dir(path.parent)
    if path.exists() and path.is_symlink():
        raise ValueError("qualification_session_manifest_symlink_forbidden")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_private_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ValueError("qualification_session_manifest_missing") from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("qualification_session_manifest_unsafe")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("qualification_session_manifest_permissions_not_0600")
    if metadata.st_size <= 0 or metadata.st_size > MAX_MANIFEST_BYTES:
        raise ValueError("qualification_session_manifest_size_invalid")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("qualification_session_manifest_unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("qualification_session_manifest_not_object")
    manifest = dict(value)
    _validate_manifest_binding(source, manifest)
    return source, manifest


def _valid_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _validate_manifest_binding(path: Path, manifest: Mapping[str, Any]) -> None:
    blockers: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        blockers.append("schema")
    if manifest.get("provider") != "vast":
        blockers.append("provider")
    release_binding_status = manifest.get("release_binding_status")
    preallocation_blocked = bool(
        manifest.get("status") == "blocked"
        and manifest.get("instance_id") is None
        and manifest.get("continuing_spend") is False
    )
    if release_binding_status == "blocked":
        if (
            manifest.get("status") != "blocked"
            or manifest.get("instance_id") is not None
            or manifest.get("continuing_spend") is not False
        ):
            blockers.append("blocked_release_binding_state")
    elif release_binding_status == "bound":
        if not valid_image_binding(manifest.get("image_ref"), manifest.get("image_digest")):
            blockers.append("image")
        if not valid_source_commit(manifest.get("source_commit")):
            blockers.append("source_commit")
    else:
        # Retained pre-contract sessions remain readable for status/teardown.
        if not valid_image_binding(manifest.get("image_ref"), manifest.get("image_digest")):
            blockers.append("image")
        source_commit = manifest.get("source_commit")
        if source_commit is not None and not valid_source_commit(source_commit):
            blockers.append("source_commit")
    if manifest.get("bundle_sha256") != BUNDLE_SHA256:
        blockers.append("bundle")
    instance_id = str(manifest.get("instance_id") or "")
    if instance_id and (not instance_id.isdigit() or int(instance_id) <= 0):
        blockers.append("instance_id")
    prefix = str(manifest.get("resource_name_prefix") or "")
    name = str(manifest.get("resource_name") or "")
    if not prefix.startswith(NAME_PREFIX_ROOT) or not name.startswith(prefix):
        blockers.append("resource_scope")
    nonce = str(manifest.get("launch_session_id") or "")
    if not nonce or manifest.get("launch_session_nonce_sha256") != _sha256_bytes(
        nonce.encode("utf-8")
    ):
        blockers.append("launch_nonce")
    bootstrap = manifest.get("bootstrap")
    bootstrap = dict(bootstrap) if isinstance(bootstrap, Mapping) else {}
    if not preallocation_blocked:
        for key in (
            "provider_bootstrap_sha256",
            "episode_bootstrap_sha256",
            "control_script_sha256",
            "refresh_installer_sha256",
        ):
            if not _valid_sha256(bootstrap.get(key)):
                blockers.append(key)
        if (
            bootstrap.get("control_contract_version")
            not in REFRESH_COMPATIBLE_CONTROL_CONTRACT_VERSIONS
        ):
            blockers.append("control_contract_version")
    if bootstrap.get("episode_auto_run") is not False:
        blockers.append("episode_auto_run")
    if not preallocation_blocked and (
        not isinstance(bootstrap.get("overlay_revision"), int)
        or int(bootstrap.get("overlay_revision") or 0) < 1
    ):
        blockers.append("overlay_revision")
    job_dir = str(manifest.get("job_dir") or "")
    if not job_dir or Path(job_dir).expanduser().resolve() != path.parent:
        blockers.append("job_dir")
    if blockers:
        raise ValueError("qualification_session_manifest_binding_invalid:" + ",".join(blockers))


def _component_wrapper(*, env: Mapping[str, Any], command: list[str]) -> str:
    exports: list[str] = []
    for key, value in sorted(env.items()):
        name = str(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"qualification_component_environment_name_invalid:{name}")
        exports.append(f"export {name}={shlex.quote(str(value))}")
    if not command:
        raise ValueError("qualification_component_command_missing")
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n" + "\n".join(exports) + "\n"
        "if [ -f /workspace/.runtime-secrets/trust_env.sh ] && "
        "[ ! -L /workspace/.runtime-secrets/trust_env.sh ]; then\n"
        "  source /workspace/.runtime-secrets/trust_env.sh\n"
        "fi\n"
        f"exec {shlex.join(command)}\n"
    )


def _bridge_wrapper(env: Mapping[str, Any]) -> str:
    bridge_env = dict(env)
    bridge_env[SNAPSHOT_ENV] = str(bridge_env.get(SNAPSHOT_ENV) or SNAPSHOT_DEFAULT_PATH)
    bridge_env[BRIDGE_REQUIRED_ENV] = "true"
    bridge_env["BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_SOURCE_SHA256"] = NATIVE_BRIDGE_SOURCE_SHA256
    return _component_wrapper(
        env=bridge_env,
        command=[
            BRIDGE_BINARY_PATH,
            str(bridge_env[SNAPSHOT_ENV]),
            BRIDGE_HEARTBEAT_PATH,
        ],
    )


def _apply_trained_checkpoint_override(
    inputs: dict[str, Any], checkpoint_path: str | Path | None
) -> dict[str, Any]:
    """Pin the qualification policy server to the same-session trained output."""

    if checkpoint_path in {None, ""}:
        return inputs
    resolved = str(checkpoint_path)
    if resolved != REMOTE_FINAL_CHECKPOINT:
        raise ValueError("qualification_trained_checkpoint_path_not_fixed")
    plan = inputs.get("plan")
    plan = dict(plan) if isinstance(plan, Mapping) else {}
    command = [str(item) for item in plan.get("groot_server_command") or []]
    positions = [index for index, item in enumerate(command) if item == "--model-path"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("qualification_groot_model_path_option_invalid")
    command[positions[0] + 1] = resolved
    plan["groot_server_command"] = command
    plan["qualification_checkpoint_override"] = {
        "schema_version": "single_g1_kitchen_qualification_checkpoint_override.v1",
        "checkpoint_path": resolved,
        "same_session_training_required": True,
        "open_loop_qualification_required": True,
        "isaac_registered_transition_required": True,
        "task_compatibility_claimed": False,
    }
    updated = dict(inputs)
    updated["plan"] = plan
    return updated


def _qualification_control_script_v1(
    *,
    launch_session_id: str,
    episode_bootstrap_sha256: str,
    bundle_sha256: str,
    component_sha256s: Mapping[str, str],
) -> str:
    """Return a shell controller with a closed action/component command set."""

    sha_cases = "\n".join(
        f"    {component}) printf '%s\\n' {shlex.quote(digest)} ;;"
        for component, digest in sorted(component_sha256s.items())
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT={shlex.quote(REMOTE_ROOT)}
STATE={shlex.quote(REMOTE_STATE_DIR)}
BINDING={shlex.quote(REMOTE_BINDING)}
RUNTIME_ENV={shlex.quote(REMOTE_RUNTIME_ENV)}
EPISODE_BOOTSTRAP={shlex.quote(REMOTE_EPISODE_BOOTSTRAP)}
PROVIDER_BOOTSTRAP=/tmp/blueprint-provider-bootstrap.sh
EXPECTED_EPISODE_BOOTSTRAP_SHA256={shlex.quote(episode_bootstrap_sha256)}
EXPECTED_BUNDLE_SHA256={shlex.quote(bundle_sha256)}
EXPECTED_LAUNCH_SESSION_ID={shlex.quote(launch_session_id)}
mkdir -p "$STATE"
ATTEMPT_ARCHIVE=/workspace/qualification_attempts
ATTEMPT_SEQUENCE=""
ATTEMPT_NONCE_SHA256=""

ACTION="${{1:-}}"
COMPONENT="${{2:-}}"
TAIL_LINES="${{3:-200}}"
case "$ACTION" in status|tail|run|restart) ;; *) echo qualification_action_forbidden >&2; exit 64 ;; esac
case "$COMPONENT" in
  bootstrap|episode|groot_server|gear_sonic_controller|isaac_task_executor|gear_sonic_isaac_dds_bridge|groot_microwave_finetune) ;;
  *) echo qualification_component_forbidden >&2; exit 64 ;;
esac
case "$TAIL_LINES" in ''|*[!0-9]*) echo qualification_tail_lines_invalid >&2; exit 64 ;; esac
if [ "$TAIL_LINES" -lt 1 ] || [ "$TAIL_LINES" -gt {MAX_TAIL_LINES} ]; then
  echo qualification_tail_lines_invalid >&2; exit 64
fi
if [ ! -f "$RUNTIME_ENV" ] || [ -L "$RUNTIME_ENV" ] || [ "$(stat -c '%a' "$RUNTIME_ENV")" != 600 ]; then
  echo qualification_runtime_environment_missing_or_unsafe >&2; exit 65
fi
source "$RUNTIME_ENV"
if [ "$ACTION" = run ] || [ "$ACTION" = restart ]; then
  if ! mkdir "$STATE/control_mutation.lock" 2>/dev/null; then
    echo qualification_control_mutation_already_running >&2; exit 68
  fi
  trap 'rmdir "$STATE/control_mutation.lock" 2>/dev/null || true' EXIT
fi

python3 - "$BINDING" "$PROVIDER_BOOTSTRAP" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$EXPECTED_BUNDLE_SHA256" "$EXPECTED_LAUNCH_SESSION_ID" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
provider_bootstrap = pathlib.Path(sys.argv[2])
value = json.loads(path.read_text(encoding="utf-8"))
expected = {{
    "provider_bootstrap_sha256": hashlib.sha256(provider_bootstrap.read_bytes()).hexdigest(),
    "episode_bootstrap_sha256": sys.argv[3],
    "bundle_sha256": sys.argv[4],
    "launch_session_id": sys.argv[5],
}}
if value != expected:
    raise SystemExit("qualification_remote_binding_mismatch")
PY

actual_episode_sha256=$(sha256sum "$EPISODE_BOOTSTRAP" | awk '{{print $1}}')
if [ "$actual_episode_sha256" != "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" ]; then
  echo qualification_episode_bootstrap_sha256_mismatch >&2; exit 65
fi

script_path() {{
  case "$1" in
    bootstrap|episode) printf '%s\n' "$EPISODE_BOOTSTRAP" ;;
    groot_server) printf '%s\n' "$ROOT/qualification_groot_server.sh" ;;
    gear_sonic_controller) printf '%s\n' "$ROOT/qualification_gear_sonic_controller.sh" ;;
    isaac_task_executor) printf '%s\n' "$ROOT/qualification_isaac_task_executor.sh" ;;
    gear_sonic_isaac_dds_bridge) printf '%s\n' "$ROOT/qualification_gear_sonic_isaac_dds_bridge.sh" ;;
    groot_microwave_finetune) printf '%s\n' "$ROOT/qualification_groot_microwave_finetune.sh" ;;
  esac
}}
expected_script_sha() {{
  case "$1" in
{sha_cases}
  esac
}}
log_path() {{
  case "$1" in
    bootstrap|episode) printf '%s\n' /workspace/closed_loop_out/qualification_episode.log ;;
    groot_server) printf '%s\n' /workspace/groot_server.log ;;
    gear_sonic_controller) printf '%s\n' /workspace/gear_sonic_controller.log ;;
    isaac_task_executor) printf '%s\n' /workspace/isaac_task_executor.log ;;
    gear_sonic_isaac_dds_bridge) printf '%s\n' {shlex.quote(BRIDGE_LOG_PATH)} ;;
    groot_microwave_finetune) printf '%s\n' /workspace/microwave_finetune.log ;;
  esac
}}
fixed_process_pattern() {{
  case "$1" in
    groot_server) printf '%s\n' '/opt/gr00t/gr00t/eval/run_gr00t_server.py' ;;
    gear_sonic_controller) printf '%s\n' '/opt/wbc/gear_sonic_deploy' ;;
    isaac_task_executor) printf '%s\n' '/workspace/runtime_overlay/run_patched_isaac_executor.py' ;;
    gear_sonic_isaac_dds_bridge) printf '%s\n' '/workspace/runtime_overlay/gear_sonic_isaac_dds_bridge/gear_sonic_isaac_dds_bridge' ;;
    groot_microwave_finetune) printf '%s\n' '/opt/gr00t/gr00t/experiment/launch_finetune.py' ;;
    bootstrap|episode) printf '%s\n' "$EPISODE_BOOTSTRAP" ;;
  esac
}}
verify_component_script() {{
  selected=$(script_path "$1")
  expected=$(expected_script_sha "$1")
  actual=$(sha256sum "$selected" | awk '{{print $1}}')
  if [ -z "$expected" ] || [ "$actual" != "$expected" ]; then
    echo qualification_component_script_sha256_mismatch >&2; exit 65
  fi
}}
live_pids() {{
  pid_file="$STATE/$1.pid"
  if [ -f "$pid_file" ]; then
    pid=$(tr -cd '0-9' < "$pid_file")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then printf '%s\n' "$pid"; fi
  fi
  pattern=$(fixed_process_pattern "$1")
  pgrep -f -- "$pattern" 2>/dev/null || true
}}
stop_component() {{
  pids=$(live_pids "$1" | sort -u)
  if [ -n "$pids" ]; then
    while IFS= read -r pid; do kill -TERM "$pid" 2>/dev/null || true; done <<< "$pids"
    sleep 2
    while IFS= read -r pid; do kill -KILL "$pid" 2>/dev/null || true; done <<< "$pids"
  fi
  rm -f "$STATE/$1.pid"
}}
prepare_episode_attempt() {{
  counter="$STATE/episode_attempt_sequence.txt"
  previous=0
  if [ -f "$counter" ] && [ ! -L "$counter" ]; then
    previous=$(tr -cd '0-9' < "$counter")
  fi
  case "$previous" in ''|*[!0-9]*) echo qualification_attempt_sequence_invalid >&2; exit 69 ;; esac
  mkdir -p "$ATTEMPT_ARCHIVE"
  chmod 700 "$ATTEMPT_ARCHIVE"
  if [ "$previous" -gt 0 ]; then
    previous_slug=$(printf 'attempt_%04d' "$previous")
  else
    previous_slug=preexisting_before_attempt_0001
  fi
  archive="$ATTEMPT_ARCHIVE/$previous_slug"
  if [ -e "$archive" ] || [ -L "$archive" ]; then
    echo qualification_prior_attempt_archive_already_exists >&2; exit 69
  fi
  mkdir "$archive"
  chmod 700 "$archive"
  for stale in \
    /workspace/closed_loop_out /workspace/out /workspace/bootstrap.json \
    /workspace/initial_policy_frame.png \
    /workspace/controller_fk_camera_projection_context.json \
    /workspace/runtime_ephemeral_trust.json \
    /workspace/isaac_runtime_result.json /workspace/groot_oscar_image_healthcheck.json \
    /workspace/groot_oscar_image_healthcheck.stderr.log /workspace/groot_server.log \
    /workspace/gear_sonic_controller.log /workspace/gear_sonic_isaac_dds_bridge.log \
    /workspace/isaac_task_executor.log /workspace/closed_loop_stdout.log \
    /workspace/closed_loop_stderr.log /workspace/initial_g1_sonic_state.json \
    /workspace/qualification_episode.log /workspace/input_bundle.zip \
    /workspace/attempt_input_manifest.json; do
    if [ -e "$stale" ] || [ -L "$stale" ]; then mv -- "$stale" "$archive/"; fi
  done
  ATTEMPT_SEQUENCE=$((previous + 1))
  counter_tmp="$STATE/.episode_attempt_sequence.$$"
  printf '%s\n' "$ATTEMPT_SEQUENCE" > "$counter_tmp"
  chmod 600 "$counter_tmp"
  mv "$counter_tmp" "$counter"
  attempt_slug=$(printf 'attempt_%04d' "$ATTEMPT_SEQUENCE")
  rm -rf /workspace/.runtime-secrets
  mkdir -p /workspace/closed_loop_out
  ATTEMPT_NONCE="$EXPECTED_LAUNCH_SESSION_ID:$attempt_slug"
  ATTEMPT_NONCE_SHA256=$(printf '%s' "$ATTEMPT_NONCE" | sha256sum | awk '{{print $1}}')
  export BLUEPRINT_QUALIFICATION_ATTEMPT_SEQUENCE="$ATTEMPT_SEQUENCE"
  export BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE="$ATTEMPT_NONCE"
  export BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE_SHA256="$ATTEMPT_NONCE_SHA256"
  python3 - /workspace/closed_loop_out/qualification_attempt.json "$ATTEMPT_SEQUENCE" "$ATTEMPT_NONCE" "$ATTEMPT_NONCE_SHA256" "$EXPECTED_LAUNCH_SESSION_ID" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$EXPECTED_BUNDLE_SHA256" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {{
    "schema_version": "single_g1_kitchen_qualification_attempt.v1",
    "attempt_sequence": int(sys.argv[2]),
    "attempt_nonce": sys.argv[3],
    "attempt_nonce_sha256": sys.argv[4],
    "launch_session_id": sys.argv[5],
    "episode_bootstrap_sha256": sys.argv[6],
    "bundle_sha256": sys.argv[7],
    "stale_outputs_reused": False,
    "raw_secret_values_recorded": False,
}}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}}
start_component() {{
  verify_component_script "$1"
  if [ -n "$(live_pids "$1" | head -n 1)" ]; then
    echo qualification_component_already_running >&2; exit 66
  fi
  if [ "$1" = episode ] || [ "$1" = bootstrap ]; then prepare_episode_attempt; fi
  selected=$(script_path "$1")
  log=$(log_path "$1")
  if [ "$1" = episode ] || [ "$1" = bootstrap ]; then
    nohup "$selected" > "$log" 2>&1 < /dev/null &
  else
    nohup "$selected" >> "$log" 2>&1 < /dev/null &
  fi
  pid=$!
  tmp="$STATE/.$1.pid.$$"
  printf '%s\n' "$pid" > "$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$STATE/$1.pid"
  printf 'action=run component=%s pid=%s bootstrap_sha256=%s attempt_sequence=%s attempt_nonce_sha256=%s\n' "$1" "$pid" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "${{ATTEMPT_SEQUENCE:-}}" "${{ATTEMPT_NONCE_SHA256:-}}"
}}

verify_component_script "$COMPONENT"
case "$ACTION" in
  status)
    pids=$(live_pids "$COMPONENT" | sort -u | paste -sd, -)
    if [ -n "$pids" ]; then state=running; else state=stopped; fi
    attempt_sequence=""
    attempt_nonce_sha256=""
    counter="$STATE/episode_attempt_sequence.txt"
    if [ -L "$counter" ]; then echo qualification_attempt_sequence_unsafe >&2; exit 69; fi
    if [ -f "$counter" ]; then
      attempt_sequence=$(tr -cd '0-9' < "$counter")
      case "$attempt_sequence" in ''|*[!0-9]*) echo qualification_attempt_sequence_invalid >&2; exit 69 ;; esac
      attempt_slug=$(printf 'attempt_%04d' "$attempt_sequence")
      attempt_nonce_sha256=$(printf '%s' "$EXPECTED_LAUNCH_SESSION_ID:$attempt_slug" | sha256sum | awk '{{print $1}}')
    fi
    printf 'action=status component=%s state=%s pids=%s bootstrap_sha256=%s attempt_sequence=%s attempt_nonce_sha256=%s\n' "$COMPONENT" "$state" "$pids" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$attempt_sequence" "$attempt_nonce_sha256"
    ;;
  tail)
    log=$(log_path "$COMPONENT")
    if [ ! -f "$log" ] || [ -L "$log" ]; then echo qualification_log_missing >&2; exit 67; fi
    tail -n "$TAIL_LINES" -- "$log"
    ;;
  run)
    start_component "$COMPONENT"
    ;;
  restart)
    if [ "$COMPONENT" = episode ] || [ "$COMPONENT" = bootstrap ]; then
      stop_component episode
      stop_component groot_server
      stop_component gear_sonic_controller
      stop_component isaac_task_executor
      stop_component gear_sonic_isaac_dds_bridge
    else
      stop_component "$COMPONENT"
    fi
    start_component "$COMPONENT"
    ;;
esac
"""


def _qualification_overlay_sources(inputs: Mapping[str, Any]) -> dict[str, str]:
    route = inputs.get("route")
    if not isinstance(route, Mapping):
        raise ValueError("qualification_route_overlay_missing")
    route_bytes = (json.dumps(dict(route), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    route_base64 = base64.b64encode(route_bytes).decode("ascii")
    # The retained Vast container keeps its original launch environment across
    # refreshes.  Override only the non-secret route variable inside the
    # revision-bound episode process so the bootstrap's existing safe writer
    # materializes the newly reviewed camera-safe stance.
    episode_script = (
        "#!/usr/bin/env bash\n"
        f"export BLUEPRINT_ROUTE_JSON_B64={shlex.quote(route_base64)}\n"
        + _vast_remote_bootstrap_script(inputs)
    )
    plan = dict(inputs.get("plan") or {})
    env = dict(plan.get("env") or {})
    finetune = inputs.get("finetune_component")
    finetune = dict(finetune) if isinstance(finetune, Mapping) else {}
    finetune_script = str(finetune.get("script") or "")
    if not finetune_script:
        finetune_script = (
            "#!/usr/bin/env bash\n"
            "echo qualification_finetune_dataset_not_bound >&2\n"
            "exit 64\n"
        )
    return {
        "qualification_episode_bootstrap.sh": episode_script,
        "qualification_groot_server.sh": _component_wrapper(
            env=env,
            command=[str(item) for item in plan.get("groot_server_command") or []],
        ),
        "qualification_gear_sonic_controller.sh": _component_wrapper(
            env=env,
            command=[str(item) for item in plan.get("gear_sonic_controller_command") or []],
        ),
        "qualification_isaac_task_executor.sh": _component_wrapper(
            env=env,
            command=[str(item) for item in plan.get("isaac_task_executor_command") or []],
        ),
        "qualification_gear_sonic_isaac_dds_bridge.sh": _bridge_wrapper(env),
        "qualification_groot_microwave_finetune.sh": finetune_script,
    }


def _qualification_refresh_installer_source() -> str:
    constants = {
        "refresh_request_schema": REFRESH_REQUEST_SCHEMA_VERSION,
        "refresh_payload_schema": REFRESH_PAYLOAD_SCHEMA_VERSION,
        "overlay_binding_schema": OVERLAY_BINDING_SCHEMA_VERSION,
        "immutable_binding_schema": IMMUTABLE_BINDING_SCHEMA_VERSION,
        "allowed_files": sorted(
            (
                "qualification_episode_bootstrap.sh",
                "qualification_groot_server.sh",
                "qualification_gear_sonic_controller.sh",
                "qualification_isaac_task_executor.sh",
                "qualification_gear_sonic_isaac_dds_bridge.sh",
                "qualification_groot_microwave_finetune.sh",
            )
        ),
        "max_payload_bytes": 128 * 1024 * 1024,
    }
    return (
        "#!/usr/bin/env python3\n"
        "import base64, hashlib, json, os, re, shutil, sys, urllib.parse, "
        "urllib.request, uuid\n"
        f"CONSTANTS = {constants!r}\n"
        r"""
from pathlib import Path


def fail(reason):
    raise SystemExit(reason)


def load_object(path, reason):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail(reason)
    if not isinstance(value, dict):
        fail(reason)
    return value


if len(sys.argv) != 4:
    fail("qualification_refresh_installer_arguments_invalid")
immutable_path = Path(sys.argv[1])
active_link = Path(sys.argv[2])
revisions_dir = Path(sys.argv[3])
raw_request = sys.stdin.buffer.read(64 * 1024 + 1)
if not raw_request or len(raw_request) > 64 * 1024:
    fail("qualification_refresh_request_size_invalid")
try:
    request = json.loads(raw_request)
except (UnicodeError, json.JSONDecodeError):
    fail("qualification_refresh_request_invalid")
if not isinstance(request, dict) or set(request) != {
    "schema_version",
    "signed_get_url",
    "refresh_payload_sha256",
    "target_revision",
    "immutable_binding",
}:
    fail("qualification_refresh_request_invalid")
if request.get("schema_version") != CONSTANTS["refresh_request_schema"]:
    fail("qualification_refresh_request_schema_invalid")
url = request.get("signed_get_url")
parsed = urllib.parse.urlsplit(url if isinstance(url, str) else "")
if (
    parsed.scheme != "https"
    or not parsed.netloc
    or parsed.username is not None
    or parsed.password is not None
    or bool(parsed.fragment)
):
    fail("qualification_refresh_url_not_safe_https")
expected_payload_sha = str(request.get("refresh_payload_sha256") or "")
if not re.fullmatch(r"[0-9a-f]{64}", expected_payload_sha):
    fail("qualification_refresh_payload_sha256_invalid")
target_revision = request.get("target_revision")
if not isinstance(target_revision, int) or target_revision < 2:
    fail("qualification_refresh_target_revision_invalid")
if (
    immutable_path.is_symlink()
    or not immutable_path.is_file()
    or (immutable_path.stat().st_mode & 0o777) != 0o400
):
    fail("qualification_immutable_binding_missing_or_unsafe")
immutable = load_object(immutable_path, "qualification_immutable_binding_invalid")
if (
    immutable.get("schema_version") != CONSTANTS["immutable_binding_schema"]
    or request.get("immutable_binding") != immutable
):
    fail("qualification_refresh_immutable_binding_mismatch")
if not active_link.is_symlink():
    fail("qualification_active_overlay_link_missing_or_unsafe")
active_target = os.readlink(active_link)
if not re.fullmatch(r"qualification_revisions/revision_[0-9]{4,}", active_target):
    fail("qualification_active_overlay_target_invalid")
root = active_link.parent.resolve()
revisions = revisions_dir.resolve()
if revisions.parent != root:
    fail("qualification_revisions_scope_invalid")
active_dir = (root / active_target).resolve()
if active_dir.parent != revisions or not active_dir.is_dir():
    fail("qualification_active_overlay_scope_invalid")
active_binding = load_object(
    active_dir / "qualification_overlay_binding.json",
    "qualification_active_overlay_binding_invalid",
)
current_revision = active_binding.get("revision")
if not isinstance(current_revision, int) or current_revision < 1:
    fail("qualification_active_overlay_revision_invalid")


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            fail("qualification_refresh_redirect_not_https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


download_path = revisions / (".refresh." + uuid.uuid4().hex + ".json")
stage = revisions / (".revision.%d.%s.tmp" % (target_revision, uuid.uuid4().hex))
try:
    opener = urllib.request.build_opener(HttpsOnlyRedirect())
    try:
        response = opener.open(url, timeout=60)
    except Exception:
        fail("qualification_refresh_download_failed")
    if urllib.parse.urlsplit(response.geturl()).scheme != "https":
        fail("qualification_refresh_final_url_not_https")
    digest = hashlib.sha256()
    total = 0
    with download_path.open("xb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > CONSTANTS["max_payload_bytes"]:
                fail("qualification_refresh_payload_too_large")
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(download_path, 0o600)
    if digest.hexdigest() != expected_payload_sha:
        fail("qualification_refresh_payload_sha256_mismatch")
    payload = load_object(download_path, "qualification_refresh_payload_invalid")
    if set(payload) != {
        "schema_version", "target_revision", "immutable_binding", "files"
    }:
        fail("qualification_refresh_payload_shape_invalid")
    if payload.get("schema_version") != CONSTANTS["refresh_payload_schema"]:
        fail("qualification_refresh_payload_schema_invalid")
    if payload.get("target_revision") != target_revision:
        fail("qualification_refresh_payload_revision_mismatch")
    if payload.get("immutable_binding") != immutable:
        fail("qualification_refresh_payload_immutable_binding_mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or sorted(files) != CONSTANTS["allowed_files"]:
        fail("qualification_refresh_payload_file_allowlist_mismatch")
    expected_binding = {
        "schema_version": CONSTANTS["overlay_binding_schema"],
        "revision": target_revision,
        "source_payload_sha256": expected_payload_sha,
        "files": {},
    }
    for name in CONSTANTS["allowed_files"]:
        row = files.get(name)
        if not isinstance(row, dict) or set(row) != {"base64", "sha256"}:
            fail("qualification_refresh_payload_file_row_invalid")
        expected = str(row.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail("qualification_refresh_payload_file_sha256_invalid")
        expected_binding["files"][name] = expected
    final = revisions / ("revision_%04d" % target_revision)
    if current_revision == target_revision:
        if active_binding != expected_binding or active_dir != final:
            fail("qualification_refresh_idempotent_revision_mismatch")
    elif current_revision != target_revision - 1:
        fail("qualification_refresh_revision_not_sequential")
    else:
        stage.mkdir(mode=0o700)
        for name in CONSTANTS["allowed_files"]:
            row = files[name]
            try:
                data = base64.b64decode(row["base64"], validate=True)
            except Exception:
                fail("qualification_refresh_payload_file_base64_invalid")
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                fail("qualification_refresh_payload_file_digest_mismatch")
            destination = stage / name
            with destination.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(destination, 0o500)
        binding_path = stage / "qualification_overlay_binding.json"
        with binding_path.open("x", encoding="utf-8") as handle:
            json.dump(expected_binding, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(binding_path, 0o400)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if final.exists() or final.is_symlink():
            fail("qualification_refresh_target_revision_already_exists")
        os.replace(stage, final)
        temporary_link = root / (".qualification_active." + uuid.uuid4().hex)
        os.symlink(str(final.relative_to(root)), temporary_link)
        os.replace(temporary_link, active_link)
        root_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    print(
        "action=refresh component=bootstrap overlay_revision=%d "
        "refresh_payload_sha256=%s episode_bootstrap_sha256=%s"
        % (
            target_revision,
            expected_payload_sha,
            expected_binding["files"]["qualification_episode_bootstrap.sh"],
        )
    )
finally:
    try:
        download_path.unlink()
    except FileNotFoundError:
        pass
    if stage.exists():
        shutil.rmtree(stage)
"""
    )


def _qualification_control_script(
    *,
    launch_session_id: str,
    bundle_sha256: str,
    image_digest: str,
    source_commit: str,
) -> str:
    """Return immutable fixed control; mutable overlay digests live in the active binding."""

    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT={shlex.quote(REMOTE_ROOT)}
STATE={shlex.quote(REMOTE_STATE_DIR)}
ACTIVE={shlex.quote(REMOTE_ACTIVE_OVERLAY)}
BINDING={shlex.quote(REMOTE_BINDING)}
IMMUTABLE_BINDING={shlex.quote(REMOTE_IMMUTABLE_BINDING)}
RUNTIME_ENV={shlex.quote(REMOTE_RUNTIME_ENV)}
EPISODE_BOOTSTRAP={shlex.quote(REMOTE_EPISODE_BOOTSTRAP)}
PROVIDER_BOOTSTRAP=/tmp/blueprint-provider-bootstrap.sh
REFRESH_INSTALLER={shlex.quote(REMOTE_REFRESH_INSTALLER)}
REVISIONS={shlex.quote(REMOTE_REVISIONS_DIR)}
EXPECTED_IMAGE_DIGEST={shlex.quote(image_digest)}
EXPECTED_SOURCE_COMMIT={shlex.quote(source_commit)}
EXPECTED_BUNDLE_SHA256={shlex.quote(bundle_sha256)}
EXPECTED_LAUNCH_SESSION_ID={shlex.quote(launch_session_id)}
EXPECTED_CONTROL_CONTRACT={shlex.quote(CONTROL_CONTRACT_VERSION)}
mkdir -p "$STATE"
ATTEMPT_ARCHIVE=/workspace/qualification_attempts
ATTEMPT_SEQUENCE=""
ATTEMPT_NONCE_SHA256=""

ACTION="${{1:-}}"
COMPONENT="${{2:-}}"
TAIL_LINES="${{3:-200}}"
case "$ACTION" in status|tail|gpu-status|run|restart|stop|refresh) ;; *) echo qualification_action_forbidden >&2; exit 64 ;; esac
case "$COMPONENT" in
  bootstrap|episode|groot_server|gear_sonic_controller|isaac_task_executor|gear_sonic_isaac_dds_bridge|groot_microwave_finetune) ;;
  *) echo qualification_component_forbidden >&2; exit 64 ;;
esac
if [ "$ACTION" = refresh ] && [ "$COMPONENT" != bootstrap ]; then
  echo qualification_refresh_component_forbidden >&2; exit 64
fi
case "$TAIL_LINES" in ''|*[!0-9]*) echo qualification_tail_lines_invalid >&2; exit 64 ;; esac
if [ "$TAIL_LINES" -lt 1 ] || [ "$TAIL_LINES" -gt {MAX_TAIL_LINES} ]; then
  echo qualification_tail_lines_invalid >&2; exit 64
fi
if [ ! -f "$RUNTIME_ENV" ] || [ -L "$RUNTIME_ENV" ] || [ "$(stat -c '%a' "$RUNTIME_ENV")" != 600 ]; then
  echo qualification_runtime_environment_missing_or_unsafe >&2; exit 65
fi
source "$RUNTIME_ENV"
if [ "$ACTION" = run ] || [ "$ACTION" = restart ] || [ "$ACTION" = refresh ]; then
  if ! mkdir "$STATE/control_mutation.lock" 2>/dev/null; then
    echo qualification_control_mutation_already_running >&2; exit 68
  fi
  trap 'rmdir "$STATE/control_mutation.lock" 2>/dev/null || true' EXIT
fi

read -r ACTIVE_REVISION EXPECTED_EPISODE_BOOTSTRAP_SHA256 < <(
python3 - "$IMMUTABLE_BINDING" "$BINDING" "$PROVIDER_BOOTSTRAP" "$0" "$REFRESH_INSTALLER" "$ACTIVE" "$EXPECTED_IMAGE_DIGEST" "$EXPECTED_BUNDLE_SHA256" "$EXPECTED_LAUNCH_SESSION_ID" "$EXPECTED_CONTROL_CONTRACT" "$EXPECTED_SOURCE_COMMIT" <<'PY'
import hashlib, json, os, pathlib, re, sys
immutable_path, binding_path, provider_path, control_path, refresh_path, active_path = map(pathlib.Path, sys.argv[1:7])
for path, mode, reason in (
    (immutable_path, 0o400, "qualification_immutable_binding_missing_or_unsafe"),
    (provider_path, None, "qualification_provider_bootstrap_missing_or_unsafe"),
    (control_path, 0o500, "qualification_control_script_missing_or_unsafe"),
    (refresh_path, 0o500, "qualification_refresh_installer_missing_or_unsafe"),
):
    if path.is_symlink() or not path.is_file() or (mode is not None and path.stat().st_mode & 0o777 != mode):
        raise SystemExit(reason)
immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
expected_immutable = {{
    "schema_version": {IMMUTABLE_BINDING_SCHEMA_VERSION!r},
    "provider_bootstrap_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
    "image_digest": sys.argv[7],
    "source_commit": sys.argv[11],
    "bundle_sha256": sys.argv[8],
    "launch_session_id": sys.argv[9],
    "control_contract_version": sys.argv[10],
    "control_script_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
    "refresh_installer_sha256": hashlib.sha256(refresh_path.read_bytes()).hexdigest(),
}}
if immutable != expected_immutable:
    raise SystemExit("qualification_remote_immutable_binding_mismatch")
if not active_path.is_symlink():
    raise SystemExit("qualification_active_overlay_link_missing_or_unsafe")
target = os.readlink(active_path)
if not re.fullmatch(r"qualification_revisions/revision_[0-9]{{4,}}", target):
    raise SystemExit("qualification_active_overlay_target_invalid")
active_resolved = (active_path.parent / target).resolve()
if active_resolved.parent != (active_path.parent / "qualification_revisions").resolve():
    raise SystemExit("qualification_active_overlay_scope_invalid")
binding = json.loads(binding_path.read_text(encoding="utf-8"))
if binding.get("schema_version") != {OVERLAY_BINDING_SCHEMA_VERSION!r}:
    raise SystemExit("qualification_overlay_binding_schema_invalid")
revision = binding.get("revision")
files = binding.get("files")
if not isinstance(revision, int) or revision < 1 or not isinstance(files, dict):
    raise SystemExit("qualification_overlay_binding_invalid")
allowed = {{
    "qualification_episode_bootstrap.sh",
    "qualification_groot_server.sh",
    "qualification_gear_sonic_controller.sh",
    "qualification_isaac_task_executor.sh",
    "qualification_gear_sonic_isaac_dds_bridge.sh",
    "qualification_groot_microwave_finetune.sh",
}}
if set(files) != allowed:
    raise SystemExit("qualification_overlay_binding_file_allowlist_mismatch")
for name, expected in files.items():
    path = active_resolved / name
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o500:
        raise SystemExit("qualification_overlay_file_missing_or_unsafe")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("qualification_overlay_file_sha256_mismatch")
print(revision, files["qualification_episode_bootstrap.sh"])
PY
)

script_path() {{
  case "$1" in
    bootstrap|episode) printf '%s\n' "$EPISODE_BOOTSTRAP" ;;
    groot_server) printf '%s\n' "$ACTIVE/qualification_groot_server.sh" ;;
    gear_sonic_controller) printf '%s\n' "$ACTIVE/qualification_gear_sonic_controller.sh" ;;
    isaac_task_executor) printf '%s\n' "$ACTIVE/qualification_isaac_task_executor.sh" ;;
    gear_sonic_isaac_dds_bridge) printf '%s\n' "$ACTIVE/qualification_gear_sonic_isaac_dds_bridge.sh" ;;
    groot_microwave_finetune) printf '%s\n' "$ACTIVE/qualification_groot_microwave_finetune.sh" ;;
  esac
}}
expected_script_sha() {{
  name=$(basename "$(script_path "$1")")
  python3 - "$BINDING" "$name" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["files"][sys.argv[2]])
PY
}}
log_path() {{
  case "$1" in
    bootstrap|episode) printf '%s\n' /workspace/closed_loop_out/qualification_episode.log ;;
    groot_server) printf '%s\n' /workspace/groot_server.log ;;
    gear_sonic_controller) printf '%s\n' /workspace/gear_sonic_controller.log ;;
    isaac_task_executor) printf '%s\n' /workspace/isaac_task_executor.log ;;
    gear_sonic_isaac_dds_bridge) printf '%s\n' {shlex.quote(BRIDGE_LOG_PATH)} ;;
    groot_microwave_finetune) printf '%s\n' /workspace/microwave_finetune.log ;;
  esac
}}
fixed_process_pattern() {{
  case "$1" in
    groot_server) printf '%s\n' '/opt/gr00t/gr00t/eval/run_gr00t_server.py' ;;
    gear_sonic_controller) printf '%s\n' '/opt/wbc/gear_sonic_deploy' ;;
    isaac_task_executor) printf '%s\n' '/workspace/runtime_overlay/run_patched_isaac_executor.py' ;;
    gear_sonic_isaac_dds_bridge) printf '%s\n' '/workspace/runtime_overlay/gear_sonic_isaac_dds_bridge/gear_sonic_isaac_dds_bridge' ;;
    groot_microwave_finetune) printf '%s\n' '/opt/gr00t/gr00t/experiment/launch_finetune.py' ;;
    bootstrap|episode) printf '%s\n' "$EPISODE_BOOTSTRAP" ;;
  esac
}}
verify_component_script() {{
  selected=$(script_path "$1")
  expected=$(expected_script_sha "$1")
  actual=$(sha256sum "$selected" | awk '{{print $1}}')
  if [ -z "$expected" ] || [ "$actual" != "$expected" ]; then
    echo qualification_component_script_sha256_mismatch >&2; exit 65
  fi
}}
live_pids() {{
  live_non_zombie_pid() {{
    candidate="$1"
    if ! kill -0 "$candidate" 2>/dev/null; then return 1; fi
    if [ -r "/proc/$candidate/stat" ]; then
      process_state=$(awk '{{print $3}}' "/proc/$candidate/stat" 2>/dev/null || true)
      if [ "$process_state" = Z ]; then return 1; fi
    fi
    return 0
  }}
  pid_file="$STATE/$1.pid"
  if [ -f "$pid_file" ]; then
    pid=$(tr -cd '0-9' < "$pid_file")
    if [ -n "$pid" ] && live_non_zombie_pid "$pid"; then printf '%s\n' "$pid"; fi
  fi
  pattern=$(fixed_process_pattern "$1")
  while IFS= read -r pid; do
    if [ -n "$pid" ] && live_non_zombie_pid "$pid"; then printf '%s\n' "$pid"; fi
  done < <(pgrep -f -- "$pattern" 2>/dev/null || true)
}}
stop_component() {{
  pids=$(live_pids "$1" | sort -u)
  if [ -n "$pids" ]; then
    while IFS= read -r pid; do kill -TERM "$pid" 2>/dev/null || true; done <<< "$pids"
    sleep 2
    while IFS= read -r pid; do kill -KILL "$pid" 2>/dev/null || true; done <<< "$pids"
  fi
  rm -f "$STATE/$1.pid"
}}
prepare_episode_attempt() {{
  counter="$STATE/episode_attempt_sequence.txt"
  previous=0
  if [ -f "$counter" ] && [ ! -L "$counter" ]; then previous=$(tr -cd '0-9' < "$counter"); fi
  case "$previous" in ''|*[!0-9]*) echo qualification_attempt_sequence_invalid >&2; exit 69 ;; esac
  mkdir -p "$ATTEMPT_ARCHIVE"
  chmod 700 "$ATTEMPT_ARCHIVE"
  if [ "$previous" -gt 0 ]; then
    previous_slug=$(printf 'attempt_%04d' "$previous")
  else
    previous_slug=preexisting_before_attempt_0001
  fi
  archive="$ATTEMPT_ARCHIVE/$previous_slug"
  if [ -e "$archive" ] || [ -L "$archive" ]; then
    echo qualification_prior_attempt_archive_already_exists >&2; exit 69
  fi
  mkdir "$archive"
  chmod 700 "$archive"
  for stale in \
    /workspace/closed_loop_out /workspace/out /workspace/bootstrap.json \
    /workspace/initial_policy_frame.png \
    /workspace/controller_fk_camera_projection_context.json \
    /workspace/runtime_ephemeral_trust.json \
    /workspace/isaac_runtime_result.json /workspace/groot_oscar_image_healthcheck.json \
    /workspace/groot_oscar_image_healthcheck.stderr.log /workspace/groot_server.log \
    /workspace/gear_sonic_controller.log /workspace/gear_sonic_isaac_dds_bridge.log \
    /workspace/isaac_task_executor.log /workspace/closed_loop_stdout.log \
    /workspace/closed_loop_stderr.log /workspace/initial_g1_sonic_state.json \
    /workspace/qualification_episode.log /workspace/input_bundle.zip \
    /workspace/attempt_input_manifest.json; do
    if [ -e "$stale" ] || [ -L "$stale" ]; then mv -- "$stale" "$archive/"; fi
  done
  ATTEMPT_SEQUENCE=$((previous + 1))
  counter_tmp="$STATE/.episode_attempt_sequence.$$"
  printf '%s\n' "$ATTEMPT_SEQUENCE" > "$counter_tmp"
  chmod 600 "$counter_tmp"
  mv "$counter_tmp" "$counter"
  attempt_slug=$(printf 'attempt_%04d' "$ATTEMPT_SEQUENCE")
  rm -rf /workspace/.runtime-secrets
  mkdir -p /workspace/closed_loop_out
  ATTEMPT_NONCE="$EXPECTED_LAUNCH_SESSION_ID:$attempt_slug"
  ATTEMPT_NONCE_SHA256=$(printf '%s' "$ATTEMPT_NONCE" | sha256sum | awk '{{print $1}}')
  export BLUEPRINT_QUALIFICATION_ATTEMPT_SEQUENCE="$ATTEMPT_SEQUENCE"
  export BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE="$ATTEMPT_NONCE"
  export BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE_SHA256="$ATTEMPT_NONCE_SHA256"
  python3 - /workspace/closed_loop_out/qualification_attempt.json "$ATTEMPT_SEQUENCE" "$ATTEMPT_NONCE" "$ATTEMPT_NONCE_SHA256" "$EXPECTED_LAUNCH_SESSION_ID" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$EXPECTED_BUNDLE_SHA256" "$ACTIVE_REVISION" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {{
    "schema_version": "single_g1_kitchen_qualification_attempt.v1",
    "attempt_sequence": int(sys.argv[2]),
    "attempt_nonce": sys.argv[3],
    "attempt_nonce_sha256": sys.argv[4],
    "launch_session_id": sys.argv[5],
    "episode_bootstrap_sha256": sys.argv[6],
    "bundle_sha256": sys.argv[7],
    "overlay_revision": int(sys.argv[8]),
    "stale_outputs_reused": False,
    "raw_secret_values_recorded": False,
}}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
}}
start_component() {{
  verify_component_script "$1"
  if [ -n "$(live_pids "$1" | head -n 1)" ]; then
    echo qualification_component_already_running >&2; exit 66
  fi
  if [ "$1" = episode ] || [ "$1" = bootstrap ]; then prepare_episode_attempt; fi
  selected=$(script_path "$1")
  log=$(log_path "$1")
  if [ "$1" = episode ] || [ "$1" = bootstrap ]; then
    nohup "$selected" > "$log" 2>&1 < /dev/null &
  else
    nohup "$selected" >> "$log" 2>&1 < /dev/null &
  fi
  pid=$!
  tmp="$STATE/.$1.pid.$$"
  printf '%s\n' "$pid" > "$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$STATE/$1.pid"
  printf 'action=run component=%s pid=%s bootstrap_sha256=%s overlay_revision=%s attempt_sequence=%s attempt_nonce_sha256=%s\n' "$1" "$pid" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$ACTIVE_REVISION" "${{ATTEMPT_SEQUENCE:-}}" "${{ATTEMPT_NONCE_SHA256:-}}"
}}

if [ "$ACTION" = refresh ]; then
  for selected_component in episode groot_server gear_sonic_controller isaac_task_executor gear_sonic_isaac_dds_bridge groot_microwave_finetune; do
    if [ -n "$(live_pids "$selected_component" | head -n 1)" ]; then
      echo qualification_refresh_requires_all_components_stopped >&2; exit 70
    fi
  done
  python3 "$REFRESH_INSTALLER" "$IMMUTABLE_BINDING" "$ACTIVE" "$REVISIONS"
  exit 0
fi

verify_component_script "$COMPONENT"
case "$ACTION" in
  status)
    pids=$(live_pids "$COMPONENT" | sort -u | paste -sd, -)
    if [ -n "$pids" ]; then state=running; else state=stopped; fi
    attempt_sequence=""
    attempt_nonce_sha256=""
    counter="$STATE/episode_attempt_sequence.txt"
    if [ -L "$counter" ]; then echo qualification_attempt_sequence_unsafe >&2; exit 69; fi
    if [ -f "$counter" ]; then
      attempt_sequence=$(tr -cd '0-9' < "$counter")
      case "$attempt_sequence" in ''|*[!0-9]*) echo qualification_attempt_sequence_invalid >&2; exit 69 ;; esac
      attempt_slug=$(printf 'attempt_%04d' "$attempt_sequence")
      attempt_nonce_sha256=$(printf '%s' "$EXPECTED_LAUNCH_SESSION_ID:$attempt_slug" | sha256sum | awk '{{print $1}}')
    fi
    printf 'action=status component=%s state=%s pids=%s bootstrap_sha256=%s overlay_revision=%s attempt_sequence=%s attempt_nonce_sha256=%s\n' "$COMPONENT" "$state" "$pids" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$ACTIVE_REVISION" "$attempt_sequence" "$attempt_nonce_sha256"
    ;;
  tail)
    log=$(log_path "$COMPONENT")
    if [ ! -f "$log" ] || [ -L "$log" ]; then echo qualification_log_missing >&2; exit 67; fi
    tail -n "$TAIL_LINES" -- "$log"
    ;;
  gpu-status)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo qualification_nvidia_smi_missing >&2; exit 67
    fi
    printf '%s\n' 'gpu_snapshot_schema=qualification_gpu_snapshot.v1'
    nvidia-smi --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,pstate --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
    ;;
  run)
    start_component "$COMPONENT"
    ;;
  restart)
    if [ "$COMPONENT" = episode ] || [ "$COMPONENT" = bootstrap ]; then
      stop_component episode
      stop_component groot_server
      stop_component gear_sonic_controller
      stop_component isaac_task_executor
      stop_component gear_sonic_isaac_dds_bridge
      stop_component groot_microwave_finetune
    else
      stop_component "$COMPONENT"
    fi
    start_component "$COMPONENT"
    ;;
  stop)
    stop_component "$COMPONENT"
    printf 'action=stop component=%s bootstrap_sha256=%s overlay_revision=%s\n' "$COMPONENT" "$EXPECTED_EPISODE_BOOTSTRAP_SHA256" "$ACTIVE_REVISION"
    ;;
esac
"""


def _qualification_bootstrap_payload_v1(
    inputs: Mapping[str, Any], launch_session_id: str
) -> tuple[bytes, dict[str, Any]]:
    episode_script = "#!/usr/bin/env bash\n" + _vast_remote_bootstrap_script(inputs)
    plan = dict(inputs.get("plan") or {})
    env = dict(plan.get("env") or {})
    finetune = inputs.get("finetune_component")
    finetune = dict(finetune) if isinstance(finetune, Mapping) else {}
    components = {
        "groot_server": _component_wrapper(
            env=env,
            command=[str(item) for item in plan.get("groot_server_command") or []],
        ),
        "gear_sonic_controller": _component_wrapper(
            env=env,
            command=[str(item) for item in plan.get("gear_sonic_controller_command") or []],
        ),
        "isaac_task_executor": _component_wrapper(
            env=env,
            command=[str(item) for item in plan.get("isaac_task_executor_command") or []],
        ),
        "gear_sonic_isaac_dds_bridge": _bridge_wrapper(env),
        "groot_microwave_finetune": str(finetune.get("script") or ""),
    }
    episode_sha = _sha256_bytes(episode_script.encode("utf-8"))
    component_sha256s = {
        name: _sha256_bytes(script.encode("utf-8")) for name, script in components.items()
    }
    component_sha256s["episode"] = episode_sha
    component_sha256s["bootstrap"] = episode_sha

    control_script = _qualification_control_script(
        launch_session_id=launch_session_id,
        episode_bootstrap_sha256=episode_sha,
        bundle_sha256=BUNDLE_SHA256,
        component_sha256s=component_sha256s,
    )
    files = {
        REMOTE_EPISODE_BOOTSTRAP: episode_script,
        REMOTE_CONTROL_SCRIPT: control_script,
        f"{REMOTE_ROOT}/qualification_groot_server.sh": components["groot_server"],
        f"{REMOTE_ROOT}/qualification_gear_sonic_controller.sh": components[
            "gear_sonic_controller"
        ],
        f"{REMOTE_ROOT}/qualification_isaac_task_executor.sh": components["isaac_task_executor"],
        f"{REMOTE_ROOT}/qualification_gear_sonic_isaac_dds_bridge.sh": components[
            "gear_sonic_isaac_dds_bridge"
        ],
        f"{REMOTE_ROOT}/qualification_groot_microwave_finetune.sh": components[
            "groot_microwave_finetune"
        ],
    }
    encoded_files = {
        path: {
            "base64": base64.b64encode(source.encode("utf-8")).decode("ascii"),
            "sha256": _sha256_bytes(source.encode("utf-8")),
        }
        for path, source in files.items()
    }
    binding = {
        # Filled from the downloaded signed artifact itself by the installer.
        "provider_bootstrap_sha256": None,
        "episode_bootstrap_sha256": episode_sha,
        "bundle_sha256": BUNDLE_SHA256,
        "launch_session_id": launch_session_id,
    }
    installer_payload = base64.b64encode(
        json.dumps({"files": encoded_files, "binding": binding}, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    installer = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
mkdir -p {shlex.quote(REMOTE_ROOT)} {shlex.quote(REMOTE_STATE_DIR)}
python3 - <<'PY'
import base64, hashlib, json, os, shlex
from pathlib import Path
payload = json.loads(base64.b64decode({installer_payload!r}, validate=True))
provider_bootstrap = Path("/tmp/blueprint-provider-bootstrap.sh")
if not provider_bootstrap.is_file() or provider_bootstrap.is_symlink():
    raise SystemExit("qualification_provider_bootstrap_missing_or_unsafe")
payload["binding"]["provider_bootstrap_sha256"] = hashlib.sha256(
    provider_bootstrap.read_bytes()
).hexdigest()
runtime_environment_names = (
    "ACCEPT_EULA",
    "PRIVACY_CONSENT",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "BLUEPRINT_EVAL_MANIFEST_URI",
    "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
    "BLUEPRINT_ROUTE_JSON_B64",
    "BLUEPRINT_TASK_PROMPT",
    "BLUEPRINT_SEALED_LAUNCH_PLAN_B64",
    "BLUEPRINT_SEED_PROVENANCE_B64",
    "BLUEPRINT_LAUNCH_SESSION_ID",
    "BLUEPRINT_WORKER_IMAGE_DIGEST",
    "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED",
    "BLUEPRINT_SOURCE_COMMIT",
    "BLUEPRINT_SINGLE_EPISODE_ATTEMPT_ID",
    "CONTAINER_ID",
    "VAST_CONTAINERLABEL",
)
runtime_environment = Path({REMOTE_RUNTIME_ENV!r})
runtime_environment_temporary = runtime_environment.with_name(
    "." + runtime_environment.name + "." + str(os.getpid()) + ".tmp"
)
with runtime_environment_temporary.open("x", encoding="utf-8") as handle:
    for name in runtime_environment_names:
        if name in os.environ:
            handle.write("export " + name + "=" + shlex.quote(os.environ[name]) + "\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(runtime_environment_temporary, 0o600)
os.replace(runtime_environment_temporary, runtime_environment)
for raw_path, row in payload["files"].items():
    path = Path(raw_path)
    data = base64.b64decode(row["base64"], validate=True)
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SystemExit("qualification_staged_file_sha256_mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o500)
    os.replace(temporary, path)
binding = Path({REMOTE_BINDING!r})
temporary = binding.with_name("." + binding.name + "." + str(os.getpid()) + ".tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload["binding"], handle, indent=2, sort_keys=True)
    handle.write("\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o400)
os.replace(temporary, binding)
PY
"""
    metadata = {
        "episode_bootstrap_sha256": episode_sha,
        "control_script_sha256": _sha256_bytes(control_script.encode("utf-8")),
        "component_script_sha256s": component_sha256s,
        "remote_control_script": REMOTE_CONTROL_SCRIPT,
        "remote_episode_bootstrap": REMOTE_EPISODE_BOOTSTRAP,
        "fixed_actions": ["run", "status", "tail", "restart"],
        "fixed_components": sorted(component_sha256s),
        "arbitrary_remote_command_allowed": False,
        "episode_auto_run": False,
    }
    return installer.encode("utf-8"), metadata


def _qualification_bootstrap_payload(
    inputs: Mapping[str, Any],
    launch_session_id: str,
    *,
    image_digest: str,
    source_commit: str,
) -> tuple[bytes, dict[str, Any]]:
    overlay_sources = _qualification_overlay_sources(inputs)
    episode_sha = _sha256_bytes(
        overlay_sources["qualification_episode_bootstrap.sh"].encode("utf-8")
    )
    component_names = {
        "groot_server": "qualification_groot_server.sh",
        "gear_sonic_controller": "qualification_gear_sonic_controller.sh",
        "isaac_task_executor": "qualification_isaac_task_executor.sh",
        "gear_sonic_isaac_dds_bridge": "qualification_gear_sonic_isaac_dds_bridge.sh",
        "groot_microwave_finetune": "qualification_groot_microwave_finetune.sh",
    }
    component_sha256s = {
        component: _sha256_bytes(overlay_sources[name].encode("utf-8"))
        for component, name in component_names.items()
    }
    component_sha256s["episode"] = episode_sha
    component_sha256s["bootstrap"] = episode_sha
    control_script = _qualification_control_script(
        launch_session_id=launch_session_id,
        bundle_sha256=BUNDLE_SHA256,
        image_digest=image_digest,
        source_commit=source_commit,
    )
    refresh_installer = _qualification_refresh_installer_source()
    control_sha = _sha256_bytes(control_script.encode("utf-8"))
    refresh_installer_sha = _sha256_bytes(refresh_installer.encode("utf-8"))
    encoded_overlay_files = {
        name: {
            "base64": base64.b64encode(source.encode("utf-8")).decode("ascii"),
            "sha256": _sha256_bytes(source.encode("utf-8")),
        }
        for name, source in sorted(overlay_sources.items())
    }
    immutable_binding = {
        "schema_version": IMMUTABLE_BINDING_SCHEMA_VERSION,
        "provider_bootstrap_sha256": None,
        "image_digest": image_digest,
        "source_commit": source_commit,
        "bundle_sha256": BUNDLE_SHA256,
        "launch_session_id": launch_session_id,
        "control_contract_version": CONTROL_CONTRACT_VERSION,
        "control_script_sha256": control_sha,
        "refresh_installer_sha256": refresh_installer_sha,
    }
    payload = {
        "immutable_binding": immutable_binding,
        "control_script": {
            "base64": base64.b64encode(control_script.encode("utf-8")).decode("ascii"),
            "sha256": control_sha,
        },
        "refresh_installer": {
            "base64": base64.b64encode(refresh_installer.encode("utf-8")).decode("ascii"),
            "sha256": refresh_installer_sha,
        },
        "overlay_files": encoded_overlay_files,
    }
    installer_payload = base64.b64encode(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    installer = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
mkdir -p {shlex.quote(REMOTE_ROOT)} {shlex.quote(REMOTE_STATE_DIR)} {shlex.quote(REMOTE_REVISIONS_DIR)}
python3 - <<'PY'
import base64, hashlib, json, os, shlex
from pathlib import Path
payload = json.loads(base64.b64decode({installer_payload!r}, validate=True))
provider_bootstrap = Path("/tmp/blueprint-provider-bootstrap.sh")
if not provider_bootstrap.is_file() or provider_bootstrap.is_symlink():
    raise SystemExit("qualification_provider_bootstrap_missing_or_unsafe")
provider_sha = hashlib.sha256(provider_bootstrap.read_bytes()).hexdigest()
payload["immutable_binding"]["provider_bootstrap_sha256"] = provider_sha
runtime_environment_names = (
    "ACCEPT_EULA",
    "PRIVACY_CONSENT",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "BLUEPRINT_EVAL_MANIFEST_URI",
    "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
    "BLUEPRINT_ROUTE_JSON_B64",
    "BLUEPRINT_TASK_PROMPT",
    "BLUEPRINT_SEALED_LAUNCH_PLAN_B64",
    "BLUEPRINT_SEED_PROVENANCE_B64",
    "BLUEPRINT_LAUNCH_SESSION_ID",
    "BLUEPRINT_WORKER_IMAGE_DIGEST",
    "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED",
    "BLUEPRINT_SOURCE_COMMIT",
    "BLUEPRINT_SINGLE_EPISODE_ATTEMPT_ID",
    "CONTAINER_ID",
    "VAST_CONTAINERLABEL",
)
runtime_environment = Path({REMOTE_RUNTIME_ENV!r})
runtime_environment_temporary = runtime_environment.with_name(
    "." + runtime_environment.name + "." + str(os.getpid()) + ".tmp"
)
with runtime_environment_temporary.open("x", encoding="utf-8") as handle:
    for name in runtime_environment_names:
        if name in os.environ:
            handle.write("export " + name + "=" + shlex.quote(os.environ[name]) + "\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(runtime_environment_temporary, 0o600)
os.replace(runtime_environment_temporary, runtime_environment)

def install_fixed(path_value, row):
    path = Path(path_value)
    data = base64.b64decode(row["base64"], validate=True)
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SystemExit("qualification_fixed_file_sha256_mismatch")
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o500)
    os.replace(temporary, path)

install_fixed({REMOTE_CONTROL_SCRIPT!r}, payload["control_script"])
install_fixed({REMOTE_REFRESH_INSTALLER!r}, payload["refresh_installer"])
revision = Path({REMOTE_REVISIONS_DIR!r}) / "revision_0001"
revision.mkdir(mode=0o700)
file_digests = {{}}
for name, row in payload["overlay_files"].items():
    if "/" in name or name.startswith("."):
        raise SystemExit("qualification_initial_overlay_file_name_invalid")
    data = base64.b64decode(row["base64"], validate=True)
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SystemExit("qualification_initial_overlay_file_sha256_mismatch")
    path = revision / name
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o500)
    file_digests[name] = row["sha256"]
overlay_binding = {{
    "schema_version": {OVERLAY_BINDING_SCHEMA_VERSION!r},
    "revision": 1,
    "source_payload_sha256": provider_sha,
    "files": file_digests,
}}
overlay_binding_path = revision / "qualification_overlay_binding.json"
with overlay_binding_path.open("x", encoding="utf-8") as handle:
    json.dump(overlay_binding, handle, indent=2, sort_keys=True)
    handle.write("\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(overlay_binding_path, 0o400)
immutable_path = Path({REMOTE_IMMUTABLE_BINDING!r})
with immutable_path.open("x", encoding="utf-8") as handle:
    json.dump(payload["immutable_binding"], handle, indent=2, sort_keys=True)
    handle.write("\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(immutable_path, 0o400)
active = Path({REMOTE_ACTIVE_OVERLAY!r})
temporary_link = active.with_name("." + active.name + "." + str(os.getpid()))
os.symlink("qualification_revisions/revision_0001", temporary_link)
os.replace(temporary_link, active)
PY
"""
    metadata = {
        "episode_bootstrap_sha256": episode_sha,
        "control_script_sha256": control_sha,
        "refresh_installer_sha256": refresh_installer_sha,
        "component_script_sha256s": component_sha256s,
        "overlay_revision": 1,
        "control_contract_version": CONTROL_CONTRACT_VERSION,
        "remote_control_script": REMOTE_CONTROL_SCRIPT,
        "remote_refresh_installer": REMOTE_REFRESH_INSTALLER,
        "remote_episode_bootstrap": REMOTE_EPISODE_BOOTSTRAP,
        "fixed_actions": ["run", "status", "tail", "gpu-status", "restart", "refresh"],
        "fixed_components": sorted(component_sha256s),
        "arbitrary_remote_command_allowed": False,
        "episode_auto_run": False,
    }
    return installer.encode("utf-8"), metadata


def _materialize_qualification_bootstrap(
    root: Path,
    inputs: Mapping[str, Any],
    launch_session_id: str,
    *,
    image_digest: str,
    source_commit: str,
) -> dict[str, Any]:
    payload, metadata = _qualification_bootstrap_payload(
        inputs,
        launch_session_id,
        image_digest=image_digest,
        source_commit=source_commit,
    )
    path = root / QUALIFICATION_BOOTSTRAP_NAME
    path.write_bytes(payload)
    path.chmod(0o600)
    return {
        "path": str(path),
        "provider_bootstrap_sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode_is_0600": stat.S_IMODE(path.stat().st_mode) == 0o600,
        **metadata,
    }


def _qualification_immutable_binding_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = dict(manifest.get("bootstrap") or {})
    binding = {
        "schema_version": IMMUTABLE_BINDING_SCHEMA_VERSION,
        "provider_bootstrap_sha256": bootstrap["provider_bootstrap_sha256"],
        "image_digest": manifest["image_digest"],
        "bundle_sha256": manifest["bundle_sha256"],
        "launch_session_id": manifest["launch_session_id"],
        "control_contract_version": bootstrap["control_contract_version"],
        "control_script_sha256": bootstrap["control_script_sha256"],
        "refresh_installer_sha256": bootstrap["refresh_installer_sha256"],
    }
    if manifest.get("source_commit") is not None:
        binding["source_commit"] = manifest["source_commit"]
    return binding


def _materialize_qualification_refresh_payload(
    root: Path,
    *,
    inputs: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = dict(manifest.get("bootstrap") or {})
    current_revision = int(bootstrap.get("overlay_revision") or 0)
    if current_revision < 1:
        raise ValueError("qualification_refresh_current_revision_invalid")
    if (
        bootstrap.get("control_contract_version")
        not in REFRESH_COMPATIBLE_CONTROL_CONTRACT_VERSIONS
    ):
        raise ValueError("qualification_refresh_control_contract_incompatible")
    target_revision = current_revision + 1
    overlay_sources = _qualification_overlay_sources(inputs)
    encoded_files = {
        name: {
            "base64": base64.b64encode(source.encode("utf-8")).decode("ascii"),
            "sha256": _sha256_bytes(source.encode("utf-8")),
        }
        for name, source in sorted(overlay_sources.items())
    }
    immutable = _qualification_immutable_binding_from_manifest(manifest)
    payload = {
        "schema_version": REFRESH_PAYLOAD_SCHEMA_VERSION,
        "target_revision": target_revision,
        "immutable_binding": immutable,
        "files": encoded_files,
    }
    path = root / QUALIFICATION_REFRESH_PAYLOAD_NAME
    _private_write_json(path, payload)
    episode_sha = encoded_files["qualification_episode_bootstrap.sh"]["sha256"]
    component_sha256s = {
        "groot_server": encoded_files["qualification_groot_server.sh"]["sha256"],
        "gear_sonic_controller": encoded_files["qualification_gear_sonic_controller.sh"]["sha256"],
        "isaac_task_executor": encoded_files["qualification_isaac_task_executor.sh"]["sha256"],
        "gear_sonic_isaac_dds_bridge": encoded_files[
            "qualification_gear_sonic_isaac_dds_bridge.sh"
        ]["sha256"],
        "groot_microwave_finetune": encoded_files[
            "qualification_groot_microwave_finetune.sh"
        ]["sha256"],
        "episode": episode_sha,
        "bootstrap": episode_sha,
    }
    return {
        "path": str(path),
        "refresh_payload_sha256": _sha256_bytes(path.read_bytes()),
        "size_bytes": path.stat().st_size,
        "mode_is_0600": stat.S_IMODE(path.stat().st_mode) == 0o600,
        "from_revision": current_revision,
        "target_revision": target_revision,
        "episode_bootstrap_sha256": episode_sha,
        "component_script_sha256s": component_sha256s,
        "immutable_binding": immutable,
        "control_script_unchanged": True,
        "image_bundle_instance_scope_unchanged": True,
        "arbitrary_remote_command_allowed": False,
    }


def _append_refresh_audit_record(
    manifest: dict[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    chain = list(manifest.get("refresh_audit_chain") or [])
    previous = str(chain[-1].get("audit_sha256") or "") if chain else "0" * 64
    if not _valid_sha256(previous):
        raise ValueError("qualification_refresh_audit_chain_invalid")
    body = {**dict(record), "previous_audit_sha256": previous}
    audit_sha = _sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    entry = {**body, "audit_sha256": audit_sha}
    chain.append(entry)
    manifest["refresh_audit_chain"] = chain
    return entry


def _safe_connection(inspected: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "instance_id",
        "ssh_host",
        "ssh_port",
        "ssh_endpoint_source",
        "public_ipaddr",
        "image_runtype",
        "direct_port_count",
        "direct_port_ready",
        "direct_port_metadata",
    )
    return {key: inspected.get(key) for key in allowed if inspected.get(key) is not None}


def _wait_for_qualification_attach(
    provider: Any,
    *,
    instance_id: str,
    resource_name: str,
    attempt_dir: Path,
    identity_file: str,
    timeout_seconds: int = SSH_READY_TIMEOUT_SECONDS,
    clock: Any = time.monotonic,
    sleeper: Any = time.sleep,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Retry the full metadata -> host-key -> authenticated-control chain."""

    from .gpu_render_providers import (
        enroll_vast_ssh_host_key,
        run_vast_ssh_control,
    )

    deadline = clock() + max(1, int(timeout_seconds))
    observations: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    connection: dict[str, Any] = {}
    host_key: dict[str, Any] = {}
    control_probe: dict[str, Any] = {}
    while clock() < deadline:
        inspected = provider.inspect(instance_id)
        last = dict(inspected) if isinstance(inspected, Mapping) else {}
        observation = {
            "status": last.get("status"),
            "actual_status": last.get("actual_status"),
            "cur_state": last.get("cur_state"),
            "name_matches": last.get("name") == resource_name,
            "image_runtype": last.get("image_runtype"),
            "ssh_direct_mode_confirmed": last.get("image_runtype") == "ssh_direct",
            "direct_port_ready": last.get("direct_port_ready") is True,
            "ssh_endpoint_present": bool(last.get("ssh_host") and last.get("ssh_port")),
        }
        metadata_ready = bool(
            last.get("status") == "observed"
            and last.get("actual_status") == "running"
            and str(last.get("instance_id") or "") == instance_id
            and last.get("name") == resource_name
            and last.get("image_runtype") == "ssh_direct"
            and last.get("direct_port_ready") is True
            and last.get("ssh_host")
            and last.get("ssh_port")
        )
        if metadata_ready:
            connection = _safe_connection(last)
            host_key = enroll_vast_ssh_host_key(
                connection,
                attempt_dir=attempt_dir,
                timeout_seconds=15.0,
            )
            observation["host_key_enrollment_status"] = host_key.get("status")
            observation["host_key_enrollment_blockers"] = list(host_key.get("blockers") or [])
            if host_key.get("status") == "enrolled" and host_key.get("known_hosts_file"):
                control_probe = run_vast_ssh_control(
                    connection,
                    action="status",
                    component="episode",
                    known_hosts_file=str(host_key["known_hosts_file"]),
                    identity_file=identity_file,
                    timeout_seconds=30.0,
                    tail_lines=1,
                )
                observation["authenticated_control_status"] = control_probe.get("status")
                observation["authenticated_control_returncode"] = control_probe.get("returncode")
                observation["authenticated_control_blockers"] = list(
                    control_probe.get("blockers") or []
                )
                observations.append(observation)
                if control_probe.get("status") == "completed":
                    return connection, observations, host_key, control_probe
            else:
                observations.append(observation)
        else:
            observations.append(observation)
        if last.get("status") == "absent":
            break
        sleeper(SSH_READY_POLL_SECONDS)
    return connection, observations, host_key, control_probe


def _manifest_base(
    *,
    root: Path,
    resource_name: str,
    resource_name_prefix: str,
    launch_session_id: str,
    bootstrap: Mapping[str, Any],
    deadline_epoch: float,
    image_ref: str,
    image_digest: str,
    source_commit: str,
    release_binding_status: str = "bound",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "bound_before_allocation",
        "provider": "vast",
        "instance_id": None,
        "resource_name": resource_name,
        "resource_name_prefix": resource_name_prefix,
        "job_dir": str(root),
        "image_ref": image_ref,
        "image_digest": image_digest,
        "source_commit": source_commit,
        "release_binding_status": release_binding_status,
        "bundle_sha256": BUNDLE_SHA256,
        "launch_session_id": launch_session_id,
        "launch_session_nonce_sha256": _sha256_bytes(launch_session_id.encode("utf-8")),
        "bootstrap": {
            "provider_bootstrap_sha256": bootstrap["provider_bootstrap_sha256"],
            "episode_bootstrap_sha256": bootstrap["episode_bootstrap_sha256"],
            "control_script_sha256": bootstrap["control_script_sha256"],
            "refresh_installer_sha256": bootstrap["refresh_installer_sha256"],
            "component_script_sha256s": bootstrap["component_script_sha256s"],
            "overlay_revision": bootstrap["overlay_revision"],
            "control_contract_version": bootstrap["control_contract_version"],
            "remote_control_script": REMOTE_CONTROL_SCRIPT,
            "remote_refresh_installer": REMOTE_REFRESH_INSTALLER,
            "remote_episode_bootstrap": REMOTE_EPISODE_BOOTSTRAP,
            "arbitrary_remote_command_allowed": False,
            "episode_auto_run": False,
        },
        "session_ttl_seconds": WALL_SECONDS,
        "watchdog_deadline_epoch": float(deadline_epoch),
        "watchdog": None,
        "pending_teardown_record": None,
        "pending_teardown_status": None,
        "ssh_connection": None,
        "ssh_host_key": None,
        "continuing_spend": False,
        "provider_absence_confirmed": False,
        "qualification_gate_matrix": qualification_gate_matrix(),
        "history": [],
        "latest_attempt": None,
        "collections": [],
        "pending_refresh": None,
        "refresh_audit_chain": [],
        "claim_boundary": _session_claim_boundary(),
        "raw_secret_values_recorded": False,
    }


def _allocate(
    *,
    session_manifest: str | Path,
    episode_bundle: str | Path,
    provider_bundle_url_file: str | Path,
    provider_output_put_url_file: str | Path,
    provider_output_get_url_file: str | Path,
    provider_bootstrap_url_file: str | Path | None,
    release_evidence: str | Path,
    provider_launch_request: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    execute: bool,
    identity_file: str,
    expected_source_commit: str,
    training_dataset: str | Path | None,
    trained_checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    result_path = Path(adapter_output).expanduser().resolve()
    manifest_path = Path(session_manifest).expanduser().resolve()
    root = manifest_path.parent
    ensure_dir(root)
    blockers: list[str] = []
    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists() or manifest_path.is_symlink():
        _, existing_manifest = _load_private_manifest(manifest_path)
        if (
            existing_manifest.get("instance_id")
            or existing_manifest.get("continuing_spend") is True
        ):
            raise ValueError("qualification_session_already_allocated")
    try:
        inputs = _load_single_episode_inputs(Path(episode_bundle).expanduser().resolve())
        inputs = _apply_trained_checkpoint_override(inputs, trained_checkpoint_path)
        if training_dataset not in {None, ""}:
            inputs["finetune_component"] = build_finetune_component(training_dataset)
        bundle_url = _read_secret_url_file(provider_bundle_url_file)
        put_url = _read_secret_url_file(provider_output_put_url_file)
        get_url = _read_secret_url_file(provider_output_get_url_file)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        inputs = {}
        bundle_url = put_url = get_url = ""
        blockers.append(str(exc))
    signed_output_staging_proof: dict[str, Any] = {
        "status": "not_checked_dry_run",
        "required_before_provider_allocation": True,
        "raw_signed_urls_recorded": False,
    }
    if execute and inputs and put_url and get_url:
        try:
            signed_output_staging_proof = _require_signed_output_staging_proof(
                provider_output_put_url_file=provider_output_put_url_file,
                provider_output_get_url_file=provider_output_get_url_file,
                put_url=put_url,
                get_url=get_url,
            )
        except (OSError, ValueError) as exc:
            signed_output_staging_proof = {
                "status": "blocked",
                "blockers": str(exc).split(";"),
                "raw_signed_urls_recorded": False,
            }
            blockers.extend(signed_output_staging_proof["blockers"])
    release_path = Path(release_evidence).expanduser().resolve()
    try:
        release_value = json.loads(release_path.read_text(encoding="utf-8"))
        release = dict(release_value) if isinstance(release_value, Mapping) else {}
        if not release:
            blockers.append("qualification_release_evidence_not_object")
    except (OSError, json.JSONDecodeError):
        release = {}
        blockers.append("qualification_release_evidence_missing_or_unreadable")
    release_binding, release_blockers = _release_binding(
        release, expected_source_commit=expected_source_commit
    )
    blockers.extend(release_blockers)
    image_ref = str(release_binding.get("image_ref") or "")
    image_digest = str(release_binding.get("image_digest") or "")
    source_commit = str(release_binding.get("source_commit") or "")
    existing_release_mismatch = (
        bool(existing_manifest)
        and existing_manifest.get("release_binding_status") != "blocked"
        and any(
            existing_manifest.get(key) != release_binding.get(key)
            for key in ("image_ref", "image_digest", "source_commit")
        )
    )
    if existing_release_mismatch:
        blockers.append("qualification_existing_manifest_release_binding_mismatch")

    if existing_manifest:
        prefix = str(existing_manifest["resource_name_prefix"])
        resource_name = str(existing_manifest["resource_name"])
        launch_session_id = str(existing_manifest["launch_session_id"])
    else:
        suffix = uuid.uuid4().hex[:10]
        prefix = f"{NAME_PREFIX_ROOT}{suffix}"
        resource_name = pod_name.strip() or f"{prefix}-pod"
        if not resource_name.startswith(prefix):
            resource_name = f"{prefix}-pod"
        launch_session_id = f"single-g1-kitchen-qualification-{suffix}"
    deadline_epoch = time.time() + WALL_SECONDS
    bootstrap: dict[str, Any] = {}
    bootstrap_url = ""
    bootstrap_staging_required = False
    if inputs and image_digest and not existing_release_mismatch:
        bootstrap = _materialize_qualification_bootstrap(
            root,
            inputs,
            launch_session_id,
            image_digest=image_digest,
            source_commit=source_commit,
        )
        prior_bootstrap = dict(existing_manifest.get("bootstrap") or {})
        bootstrap_changed_since_staging = bool(
            prior_bootstrap
            and any(
                bootstrap.get(key) != prior_bootstrap.get(key)
                for key in (
                    "provider_bootstrap_sha256",
                    "episode_bootstrap_sha256",
                    "control_script_sha256",
                )
            )
        )
        if bootstrap_changed_since_staging:
            bootstrap_staging_required = True
        elif provider_bootstrap_url_file:
            try:
                bootstrap_url = _read_secret_url_file(provider_bootstrap_url_file)
            except (OSError, ValueError) as exc:
                blockers.append(str(exc))
        else:
            bootstrap_staging_required = True

    provider = get_render_provider("vast")
    request: dict[str, Any] = {}
    capacity: dict[str, Any] = {}
    pre_inventory: dict[str, Any] = {}
    pre_spend_preflight: dict[str, Any] = {"status": "not_evaluated", "blockers": ["qualification_inputs_not_ready_for_pre_spend_preflight"]}
    nonce_artifact: dict[str, Any] = {}
    if inputs and not blockers and not bootstrap_staging_required:
        for name, value in (
            ("provider_bundle_url.txt", bundle_url),
            ("provider_output_put_url.txt", put_url),
            ("provider_output_get_url.txt", get_url),
        ):
            path = root / name
            path.write_text(value, encoding="utf-8")
            path.chmod(0o600)
        start_frame, _ = _write_materialized_inputs(root, inputs)
        nonce_artifact = _materialize_launch_session_nonce(root, launch_session_id)
        spec = build_launch_spec(
            job_dir=root,
            image_ref=image_ref,
            start_frame=start_frame,
            route_payload=inputs["route"],
            task_prompt=TASK_PROMPT,
            plan=inputs["plan"],
            launch_nonce=launch_session_id,
            seed_provenance=inputs["seed"],
            container_disk_gb=220,
            volume_gb=20,
            max_hourly_rate_usd=MAX_HOURLY_RATE_USD,
        )
        spec.name = resource_name
        spec.gpu_types = GPU_TYPES
        spec.min_gpu_ram_mb = 40_000
        spec.vast_launch_mode = "ssh_direct"
        spec.bootstrap_argv = ["-lc", _vast_signed_bootstrap_downloader_script()]
        spec.env.pop("BLUEPRINT_INITIAL_POLICY_FRAME_B64", None)
        spec.env.update(
            {
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "BLUEPRINT_SOURCE_COMMIT": source_commit,
                "BLUEPRINT_SINGLE_EPISODE_ATTEMPT_ID": "episode_001",
                VAST_BOOTSTRAP_URL_ENV: bootstrap_url,
                VAST_BOOTSTRAP_SHA256_ENV: bootstrap["provider_bootstrap_sha256"],
            }
        )
        spec.env.pop(RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV, None)
        spec.env.pop(ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV, None)
        request = provider.build_request(spec, root)
        request.update(
            {
                "min_gpu_ram_mb": 40_000,
                "requires_rtx": True,
                "bootstrap_transport": "signed_https_sha256",
                "remote_bootstrap_sha256": bootstrap["provider_bootstrap_sha256"],
                "remote_bootstrap_size_bytes": bootstrap["size_bytes"],
                "require_avx": True,
                "min_reliability": VAST_MIN_RELIABILITY,
                "require_known_supported_isaac_driver": (VAST_REQUIRE_KNOWN_SUPPORTED_ISAAC_DRIVER),
                "preferred_gpu_keywords": list(VAST_PREFERRED_GPU_KEYWORDS),
            }
        )
        pre_inventory = provider.billable_inventory(name_prefix="")
        capacity = provider.capacity_preflight(request)
        viable = [
            row
            for row in capacity.get("viable_gpu_types", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("on_demand_price_usd_per_hour"), (int, float))
            and float(row["on_demand_price_usd_per_hour"]) <= MAX_HOURLY_RATE_USD
        ]
        if pre_inventory.get("api_confirmed") is not True:
            blockers.append("qualification_prelaunch_inventory_unverified")
        elif pre_inventory.get("live_resource_count") != 0:
            blockers.append("qualification_prelaunch_vast_inventory_not_zero")
        if capacity.get("status") != "available" or not viable:
            blockers.append("qualification_48gb_rtx_capacity_unavailable")
        pre_spend_preflight, pre_spend_blockers = qualification_pre_spend_preflight(
            root=root, capacity=capacity,
            pre_inventory=pre_inventory,
            image_ref=image_ref,
            execute=execute,
        )
        blockers.extend(pre_spend_blockers)
        request["pre_spend_preflight"] = pre_spend_preflight
        request["prelaunch_spend_guard"] = {
            "required_before_provider_launch": True,
            "can_launch": not blockers,
            "blockers": sorted(set(blockers)),
            "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
            "maximum_live_seconds": WALL_SECONDS,
            "maximum_estimated_spend_usd": round(MAX_HOURLY_RATE_USD * WALL_SECONDS / 3600.0, 2),
        }

    reported_blockers = sorted(
        set(
            blockers
            + (["qualification_bootstrap_staging_required"] if bootstrap_staging_required else [])
        )
    )
    preserve_existing_bound_release = (
        bool(existing_manifest)
        and existing_manifest.get("release_binding_status") == "bound"
    )
    manifest = _manifest_base(
        root=root,
        resource_name=resource_name,
        resource_name_prefix=prefix,
        launch_session_id=launch_session_id,
        bootstrap=bootstrap
        or existing_manifest.get("bootstrap")
        or {
            "provider_bootstrap_sha256": "0" * 64,
            "episode_bootstrap_sha256": "0" * 64,
            "control_script_sha256": "0" * 64,
            "refresh_installer_sha256": "0" * 64, "component_script_sha256s": {},
            "overlay_revision": 0, "control_contract_version": CONTROL_CONTRACT_VERSION,
        },
        deadline_epoch=deadline_epoch,
        image_ref=(
            str(existing_manifest.get("image_ref"))
            if preserve_existing_bound_release
            else image_ref
        ),
        image_digest=(
            str(existing_manifest.get("image_digest"))
            if preserve_existing_bound_release
            else image_digest
        ),
        source_commit=(
            str(existing_manifest.get("source_commit"))
            if preserve_existing_bound_release
            else source_commit
        ),
        release_binding_status=(
            "bound"
            if preserve_existing_bound_release or not release_blockers
            else "blocked"
        ),
    )
    manifest["status"] = (
        "bootstrap_staging_required"
        if bootstrap_staging_required and not blockers
        else ("dry_run_bound" if not reported_blockers else "blocked")
    )
    manifest["signed_output_staging_proof"] = signed_output_staging_proof
    manifest["history"] = list(existing_manifest.get("history") or []) + [
        {
            "action": "allocate",
            "status": manifest["status"],
            "recorded_at": utc_now_iso(),
            "provider_mutation_performed": False,
        }
    ]
    _private_write_json(manifest_path, manifest)
    artifact_release_binding = dict(release_binding)
    if preserve_existing_bound_release:
        artifact_release_binding.update(
            {
                "image_ref": manifest["image_ref"],
                "image_digest": manifest["image_digest"],
                "source_commit": manifest["source_commit"],
                "preserved_from_session_manifest": True,
            }
        )
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": manifest["status"],
        "provider": "vast",
        "image_ref": manifest["image_ref"],
        "image_digest": manifest["image_digest"],
        "source_commit": manifest["source_commit"],
        "release_binding": artifact_release_binding,
        "bundle_sha256": inputs.get("bundle_sha256"),
        "launch_mode": "ssh_direct",
        "capacity": capacity,
        "pre_inventory": pre_inventory,
        "bootstrap": bootstrap or None,
        "launch_session_nonce_artifact": nonce_artifact or None,
        "signed_output_staging_proof": signed_output_staging_proof,
        "pre_spend_preflight": pre_spend_preflight,
        "blockers": reported_blockers,
    }
    admission = {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
        "status": "admitted" if not reported_blockers else "blocked",
        "resource_class": "gpu_render",
        "scope": "persistent_single_g1_kitchen_qualification_session",
        "retained_until_explicit_teardown_or_hard_ttl": True,
        "provider_mutations_performed": 0,
        "pre_spend_preflight": pre_spend_preflight,
        "blockers": reported_blockers,
        "raw_secret_values_recorded": False,
    }
    bound = {
        "schema_version": BOUND_REQUEST_SCHEMA_VERSION,
        "status": "bound" if not reported_blockers else manifest["status"],
        "provider": "vast",
        "resource_name": resource_name,
        "resource_name_prefix": prefix,
        "image_ref": manifest["image_ref"],
        "image_digest": manifest["image_digest"],
        "source_commit": manifest["source_commit"],
        "release_binding": artifact_release_binding,
        "bundle_sha256": inputs.get("bundle_sha256"),
        "launch_session_id": launch_session_id,
        "launch_session_nonce_sha256": manifest["launch_session_nonce_sha256"],
        "bootstrap": manifest["bootstrap"],
        "session_manifest": str(manifest_path),
        "session_manifest_mode": "0600",
        "session_ttl_seconds": WALL_SECONDS,
        "continuing_spend_after_allocate": True,
        "qualification_gate_matrix": manifest["qualification_gate_matrix"],
        "claim_boundary": manifest["claim_boundary"],
        "signed_output_staging_proof": signed_output_staging_proof,
        "pre_spend_preflight": pre_spend_preflight,
        "blockers": reported_blockers,
        "raw_secret_values_recorded": False,
    }
    write_standard_artifacts(
        provider_launch_request=provider_launch_request,
        preflight_bundle=preflight_bundle,
        admission_out=admission_out,
        bound_request_out=bound_request_out,
        bound=bound,
        preflight=preflight,
        admission=admission,
    )
    if not execute or reported_blockers:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": manifest["status"],
            "action": "allocate",
            "execute": bool(execute),
            "session_manifest": str(manifest_path),
            "continuing_spend": False,
            "provider_mutations_performed": 0,
            "signed_output_staging_proof": signed_output_staging_proof,
            "pre_spend_preflight": pre_spend_preflight,
            "blockers": reported_blockers,
        }
        write_json(result_path, result)
        return result

    grant = require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    armed = arm_watchdog(
        out_dir=root,
        pod_name_prefix=prefix,
        deadline_epoch=deadline_epoch,
        provider_name="vast",
    )
    watchdog = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--out-dir",
            str(root),
            "--pod-name-prefix",
            prefix,
            "--deadline-epoch",
            str(deadline_epoch),
            "--provider",
            "vast",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    run_id = launch_session_id
    pending = open_pending_teardown(
        provider="vast",
        lane=LANE,
        run_id=run_id,
        resource_name=resource_name,
        job_dir=root,
        max_age_seconds=WALL_SECONDS + 1_800,
    )
    manifest.update(
        {
            "status": "watchdog_armed_before_allocation",
            "watchdog": {**armed, "pid": watchdog.pid},
            "pending_teardown_record": pending.get("path"),
            "pending_teardown_status": pending.get("status"),
        }
    )
    _private_write_json(manifest_path, manifest)
    launch: dict[str, Any] = {}
    try:
        launch = provider.launch(
            root,
            request,
            cold=True,
            paid_resource_admission_grant=grant,
        )
    except BaseException as exc:
        launch = {
            "status": "launch_interrupted",
            "allocation_outcome_ambiguous": True,
            "error_type": type(exc).__name__,
        }
        pending = mark_pending_teardown_ambiguous(
            pending["path"],
            reason="qualification_provider_create_outcome_ambiguous",
            evidence={"status": launch["status"]},
        )
        manifest.update(
            {
                "status": "allocation_outcome_ambiguous_continuing_spend_unknown",
                "continuing_spend": True,
                "pending_teardown_status": pending.get("status"),
                "last_error_type": type(exc).__name__,
            }
        )
        _private_write_json(manifest_path, manifest)
        raise

    instance_id = str(launch.get("instance_id") or "")
    if not instance_id:
        ambiguous = launch.get("allocation_outcome_ambiguous") is True
        if ambiguous:
            pending = mark_pending_teardown_ambiguous(
                pending["path"],
                reason="qualification_provider_create_outcome_ambiguous",
                evidence={"status": launch.get("status")},
            )
        else:
            pending = cancel_pending_teardown(
                pending["path"],
                reason="qualification_launch_returned_no_allocation",
                evidence={"status": launch.get("status")},
            )
        manifest.update(
            {
                "status": (
                    "allocation_outcome_ambiguous_continuing_spend_unknown"
                    if ambiguous
                    else "blocked_no_allocation"
                ),
                "continuing_spend": ambiguous,
                "pending_teardown_status": pending.get("status"),
            }
        )
        _private_write_json(manifest_path, manifest)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": manifest["status"],
            "action": "allocate",
            "launch": launch,
            "session_manifest": str(manifest_path),
            "continuing_spend": manifest["continuing_spend"],
            "provider_mutations_performed": 0,
            "blockers": ["qualification_vast_instance_not_created"],
        }
        write_json(result_path, result)
        return result

    pending = bind_pending_teardown_instance(pending["path"], instance_id)
    manifest.update(
        {
            "status": "allocated_attach_pending_continuing_spend",
            "instance_id": instance_id,
            "continuing_spend": True,
            "pending_teardown_status": pending.get("status"),
            "launch": {
                "status": launch.get("status"),
                "mode": launch.get("mode"),
                "instance_id": instance_id,
            },
        }
    )
    manifest["history"].append(
        {
            "action": "allocate",
            "status": "provider_allocation_bound",
            "recorded_at": utc_now_iso(),
            "provider_mutation_performed": True,
            "instance_id": instance_id,
        }
    )
    _private_write_json(manifest_path, manifest)

    connection, observations, host_key, control_probe = _wait_for_qualification_attach(
        provider,
        instance_id=instance_id,
        resource_name=resource_name,
        attempt_dir=root,
        identity_file=identity_file,
    )
    host_key_ready = bool(
        host_key.get("status") == "enrolled"
        and host_key.get("known_hosts_file")
        and host_key.get("fingerprint_artifact")
        and host_key.get("tofu_pinned") is True
    )
    authenticated_control_ready = control_probe.get("status") == "completed"
    manifest.update(
        {
            "status": (
                "allocated_ready_continuing_spend"
                if connection and host_key_ready and authenticated_control_ready
                else "allocated_attach_blocked_continuing_spend"
            ),
            "ssh_connection": connection or None,
            "ssh_host_key": host_key or None,
            "ssh_attach_probe": {
                "status": control_probe.get("status"),
                "returncode": control_probe.get("returncode"),
                "action": control_probe.get("action"),
                "component": control_probe.get("component"),
                "blockers": list(control_probe.get("blockers") or []),
                "strict_host_key_checking": control_probe.get("strict_host_key_checking"),
            }
            if control_probe
            else None,
            "ssh_readiness_observations": observations,
            "continuing_spend": True,
        }
    )
    manifest["history"].append(
        {
            "action": "allocate",
            "status": manifest["status"],
            "recorded_at": utc_now_iso(),
            "provider_mutation_performed": False,
        }
    )
    _private_write_json(manifest_path, manifest)
    result_blockers = []
    if not connection:
        result_blockers.append("qualification_vast_ssh_direct_not_ready")
    if connection and not host_key_ready:
        result_blockers.append("qualification_vast_ssh_host_key_not_enrolled")
    if host_key_ready and not authenticated_control_ready:
        result_blockers.append("qualification_vast_ssh_control_not_ready")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": manifest["status"],
        "action": "allocate",
        "provider": "vast",
        "instance_id": instance_id,
        "session_manifest": str(manifest_path),
        "watchdog_deadline_epoch": deadline_epoch,
        "continuing_spend": True,
        "provider_mutations_performed": 1,
        "qualification_gate_matrix": manifest["qualification_gate_matrix"],
        "claim_boundary": manifest["claim_boundary"],
        "blockers": result_blockers,
    }
    write_json(result_path, result)
    return result


def _refresh_bootstrap(
    *,
    session_manifest: str | Path,
    episode_bundle: str | Path,
    provider_bootstrap_url_file: str | Path | None,
    adapter_output: str | Path,
    execute: bool,
    identity_file: str,
    training_dataset: str | Path | None,
    trained_checkpoint_path: str | Path | None,
    admission_out: str | Path | None,
) -> dict[str, Any]:
    result_path = Path(adapter_output).expanduser().resolve()
    manifest_path, manifest = _load_private_manifest(session_manifest)
    if manifest.get("continuing_spend") is not True:
        raise ValueError("qualification_session_not_live")
    if time.time() >= float(manifest.get("watchdog_deadline_epoch") or 0):
        raise ValueError("qualification_session_ttl_expired")
    inputs = _load_single_episode_inputs(Path(episode_bundle).expanduser().resolve())
    inputs = _apply_trained_checkpoint_override(inputs, trained_checkpoint_path)
    if training_dataset not in {None, ""}:
        inputs["finetune_component"] = build_finetune_component(training_dataset)
    refresh = _materialize_qualification_refresh_payload(
        manifest_path.parent,
        inputs=inputs,
        manifest=manifest,
    )
    pending = {
        "schema_version": REFRESH_REQUEST_SCHEMA_VERSION,
        "status": "artifact_staged_signed_https_url_required",
        "path": refresh["path"],
        "refresh_payload_sha256": refresh["refresh_payload_sha256"],
        "size_bytes": refresh["size_bytes"],
        "from_revision": refresh["from_revision"],
        "target_revision": refresh["target_revision"],
        "episode_bootstrap_sha256": refresh["episode_bootstrap_sha256"],
        "component_script_sha256s": refresh["component_script_sha256s"],
        "immutable_binding": refresh["immutable_binding"],
        "signed_get_url_stored": False,
        "control_script_unchanged": True,
        "image_bundle_instance_scope_unchanged": True,
    }
    existing_pending = manifest.get("pending_refresh")
    pending_matches = isinstance(existing_pending, Mapping) and all(
        existing_pending.get(key) == value for key, value in pending.items() if key != "status"
    )
    if not pending_matches:
        manifest["pending_refresh"] = pending
        manifest["status"] = "refresh_bootstrap_staging_required_continuing_spend"
        manifest.setdefault("history", []).append(
            {
                "action": "refresh-bootstrap-stage",
                "status": manifest["status"],
                "recorded_at": utc_now_iso(),
                "from_revision": refresh["from_revision"],
                "target_revision": refresh["target_revision"],
                "refresh_payload_sha256": refresh["refresh_payload_sha256"],
                "provider_mutation_performed": False,
            }
        )
        _private_write_json(manifest_path, manifest)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": manifest["status"],
            "action": "refresh-bootstrap",
            "instance_id": manifest.get("instance_id"),
            "session_manifest": str(manifest_path),
            "refresh_payload": {
                key: value for key, value in refresh.items() if key != "immutable_binding"
            },
            "continuing_spend": True,
            "provider_mutations_performed": 0,
            "blockers": ["qualification_refresh_payload_upload_and_signed_url_required"],
        }
        write_json(result_path, result)
        return result
    if provider_bootstrap_url_file in {None, ""}:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "refresh_bootstrap_staging_required_continuing_spend",
            "action": "refresh-bootstrap",
            "instance_id": manifest.get("instance_id"),
            "session_manifest": str(manifest_path),
            "refresh_payload": {
                key: value for key, value in refresh.items() if key != "immutable_binding"
            },
            "continuing_spend": True,
            "provider_mutations_performed": 0,
            "blockers": ["qualification_refresh_payload_signed_url_required"],
        }
        write_json(result_path, result)
        return result
    signed_get_url = _read_secret_url_file(provider_bootstrap_url_file)
    if not execute:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_refresh_bound_continuing_spend",
            "action": "refresh-bootstrap",
            "instance_id": manifest.get("instance_id"),
            "session_manifest": str(manifest_path),
            "from_revision": refresh["from_revision"],
            "target_revision": refresh["target_revision"],
            "refresh_payload_sha256": refresh["refresh_payload_sha256"],
            "control_script_unchanged": True,
            "image_bundle_instance_scope_unchanged": True,
            "signed_get_url_stored": False,
            "continuing_spend": True,
            "provider_mutations_performed": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result

    instance_id = str(manifest.get("instance_id") or "")
    provider = get_render_provider("vast")
    inspected = provider.inspect(instance_id)
    if (
        inspected.get("status") != "observed"
        or str(inspected.get("instance_id") or "") != instance_id
        or inspected.get("name") != manifest.get("resource_name")
        or inspected.get("image_runtype") != "ssh_direct"
        or inspected.get("direct_port_ready") is not True
    ):
        raise ValueError("qualification_session_provider_binding_not_observed")
    connection = _safe_connection(inspected)
    bound_connection = dict(manifest.get("ssh_connection") or {})
    if connection.get("ssh_host") != bound_connection.get("ssh_host") or connection.get(
        "ssh_port"
    ) != bound_connection.get("ssh_port"):
        raise ValueError("qualification_session_ssh_endpoint_changed")
    known_hosts_file = str(dict(manifest.get("ssh_host_key") or {}).get("known_hosts_file") or "")
    if not known_hosts_file:
        raise ValueError("qualification_session_known_hosts_missing")
    from .gpu_render_providers import run_vast_ssh_control

    admit_qualification_control_mutation(admission_out, manifest, inspected, instance_id, "refresh", "bootstrap")
    immutable_before = _qualification_immutable_binding_from_manifest(manifest)
    control = run_vast_ssh_control(
        connection,
        action="refresh",
        component="bootstrap",
        known_hosts_file=known_hosts_file,
        identity_file=identity_file,
        timeout_seconds=180.0,
        tail_lines=1,
        refresh_request={
            "schema_version": REFRESH_REQUEST_SCHEMA_VERSION,
            "signed_get_url": signed_get_url,
            "refresh_payload_sha256": refresh["refresh_payload_sha256"],
            "target_revision": refresh["target_revision"],
            "immutable_binding": immutable_before,
        },
    )
    match = re.search(
        r"\baction=refresh component=bootstrap overlay_revision=(\d+) "
        r"refresh_payload_sha256=([0-9a-f]{64}) "
        r"episode_bootstrap_sha256=([0-9a-f]{64})\b",
        str(control.get("stdout") or ""),
    )
    completed = bool(
        control.get("status") == "completed"
        and match
        and int(match.group(1)) == refresh["target_revision"]
        and match.group(2) == refresh["refresh_payload_sha256"]
        and match.group(3) == refresh["episode_bootstrap_sha256"]
    )
    recorded_at = utc_now_iso()
    if completed:
        bootstrap = dict(manifest["bootstrap"])
        prior_episode_sha = bootstrap["episode_bootstrap_sha256"]
        bootstrap.update(
            {
                "episode_bootstrap_sha256": refresh["episode_bootstrap_sha256"],
                "component_script_sha256s": refresh["component_script_sha256s"],
                "overlay_revision": refresh["target_revision"],
            }
        )
        manifest["bootstrap"] = bootstrap
        if _qualification_immutable_binding_from_manifest(manifest) != immutable_before:
            raise ValueError("qualification_refresh_changed_immutable_binding")
        audit = _append_refresh_audit_record(
            manifest,
            {
                "action": "refresh-bootstrap",
                "recorded_at": recorded_at,
                "instance_id": instance_id,
                "resource_name": manifest["resource_name"],
                "from_revision": refresh["from_revision"],
                "target_revision": refresh["target_revision"],
                "prior_episode_bootstrap_sha256": prior_episode_sha,
                "episode_bootstrap_sha256": refresh["episode_bootstrap_sha256"],
                "refresh_payload_sha256": refresh["refresh_payload_sha256"],
                "control_script_sha256": bootstrap["control_script_sha256"],
                "image_digest": manifest["image_digest"],
                "bundle_sha256": manifest["bundle_sha256"],
                "launch_session_nonce_sha256": manifest["launch_session_nonce_sha256"],
                "provider_mutation_performed": True,
            },
        )
        manifest["pending_refresh"] = None
        manifest["status"] = "bootstrap_refreshed_continuing_spend"
    else:
        audit = None
        manifest["status"] = "refresh_bootstrap_blocked_continuing_spend"
    manifest["continuing_spend"] = True
    manifest["last_refresh"] = {
        "status": manifest["status"],
        "recorded_at": recorded_at,
        "from_revision": refresh["from_revision"],
        "target_revision": refresh["target_revision"],
        "refresh_payload_sha256": refresh["refresh_payload_sha256"],
        "episode_bootstrap_sha256": refresh["episode_bootstrap_sha256"],
        "control_status": control.get("status"),
        "audit_sha256": audit.get("audit_sha256") if audit else None,
        "signed_get_url_stored": False,
    }
    manifest.setdefault("history", []).append(
        {
            "action": "refresh-bootstrap",
            **manifest["last_refresh"],
            "provider_mutation_performed": True,
        }
    )
    _private_write_json(manifest_path, manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": manifest["status"],
        "action": "refresh-bootstrap",
        "instance_id": instance_id,
        "session_manifest": str(manifest_path),
        "from_revision": refresh["from_revision"],
        "target_revision": refresh["target_revision"],
        "refresh_payload_sha256": refresh["refresh_payload_sha256"],
        "control": control,
        "audit": audit,
        "control_script_unchanged": True,
        "image_bundle_instance_scope_unchanged": True,
        "signed_get_url_stored": False,
        "continuing_spend": True,
        "provider_mutations_performed": 1,
        "qualification_gate_matrix": manifest["qualification_gate_matrix"],
        "claim_boundary": manifest["claim_boundary"],
        "blockers": [] if completed else ["qualification_fixed_refresh_failed"],
    }
    write_json(result_path, result)
    return result


def _download_provider_output_archive(signed_get_url: str) -> bytes:
    """Download one exact signed output object without persisting its URL."""

    response = safe_http_request(
        signed_get_url,
        method="GET",
        timeout_seconds=300,
        policy=presigned_transfer_policy(
            signed_get_url,
            max_response_bytes=MAX_PROVIDER_OUTPUT_ARCHIVE_BYTES,
        ),
        max_response_bytes=MAX_PROVIDER_OUTPUT_ARCHIVE_BYTES,
    )
    if response.status != 200:
        raise ValueError(f"qualification_provider_output_http_status:{response.status}")
    if not response.body:
        raise ValueError("qualification_provider_output_archive_empty")
    return response.body


def _validate_provider_output_archive_limits(archive: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as value:
            members = value.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError("qualification_provider_output_archive_invalid") from exc
    if not members or len(members) > MAX_PROVIDER_OUTPUT_MEMBERS:
        raise ValueError("qualification_provider_output_member_count_invalid")
    names: set[str] = set()
    extracted_size = 0
    for member in members:
        if member.filename in names:
            raise ValueError("qualification_provider_output_duplicate_member")
        names.add(member.filename)
        if member.flag_bits & 0x1:
            raise ValueError("qualification_provider_output_encrypted_member_forbidden")
        member_type = (member.external_attr >> 16) & 0o170000
        if member_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("qualification_provider_output_special_member_forbidden")
        extracted_size += int(member.file_size)
        if extracted_size > MAX_EXTRACTED_PROVIDER_OUTPUT_BYTES:
            raise ValueError("qualification_provider_output_extracted_size_exceeded")


def _read_collected_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"qualification_collected_{label}_missing") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"qualification_collected_{label}_unsafe")
    if metadata.st_size <= 0 or metadata.st_size > MAX_COLLECTED_JSON_BYTES:
        raise ValueError(f"qualification_collected_{label}_size_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"qualification_collected_{label}_unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"qualification_collected_{label}_not_object")
    return dict(value)


def _require_latest_attempt_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = manifest.get("latest_attempt")
    latest = dict(value) if isinstance(value, Mapping) else {}
    try:
        sequence = int(latest.get("attempt_sequence"))
        overlay_revision = int(latest.get("overlay_revision"))
    except (TypeError, ValueError) as exc:
        raise ValueError("qualification_latest_attempt_binding_missing") from exc
    attempt_slug = f"attempt_{sequence:04d}"
    launch_session_id = str(manifest.get("launch_session_id") or "")
    attempt_nonce = f"{launch_session_id}:{attempt_slug}"
    expected = {
        "schema_version": "single_g1_kitchen_qualification_attempt_binding.v1",
        "attempt_sequence": sequence,
        "attempt_slug": attempt_slug,
        "attempt_nonce": attempt_nonce,
        "attempt_nonce_sha256": _sha256_bytes(attempt_nonce.encode("utf-8")),
        "launch_session_id": launch_session_id,
        "episode_bootstrap_sha256": str(latest.get("episode_bootstrap_sha256") or ""),
        "bundle_sha256": str(manifest.get("bundle_sha256") or ""),
        "overlay_revision": overlay_revision,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if key != "schema_version" and latest.get(key) != expected_value
    ]
    if sequence < 1:
        mismatches.append("attempt_sequence")
    if not _valid_sha256(expected["episode_bootstrap_sha256"]):
        mismatches.append("episode_bootstrap_sha256")
    if mismatches:
        raise ValueError(
            "qualification_latest_attempt_binding_invalid:" + ",".join(sorted(set(mismatches)))
        )
    return {**latest, **expected}


def _validate_collected_attempt_binding(
    snapshot: Path,
    *,
    manifest: Mapping[str, Any],
    latest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap = _read_collected_json(snapshot / "bootstrap.json", label="bootstrap")
    attempt = _read_collected_json(
        snapshot / "closed_loop_out" / "qualification_attempt.json",
        label="attempt_identity",
    )
    expected_attempt = {
        "schema_version": "single_g1_kitchen_qualification_attempt.v1",
        "attempt_sequence": latest["attempt_sequence"],
        "attempt_nonce": latest["attempt_nonce"],
        "attempt_nonce_sha256": latest["attempt_nonce_sha256"],
        "launch_session_id": latest["launch_session_id"],
        "episode_bootstrap_sha256": latest["episode_bootstrap_sha256"],
        "bundle_sha256": latest["bundle_sha256"],
        "overlay_revision": latest["overlay_revision"],
        "stale_outputs_reused": False,
        "raw_secret_values_recorded": False,
    }
    mismatches = [
        f"attempt.{key}"
        for key, expected_value in expected_attempt.items()
        if attempt.get(key) != expected_value
    ]
    if bootstrap.get("schema_version") != "groot_oscar_closed_loop_bootstrap.v1":
        mismatches.append("bootstrap.schema_version")
    if bootstrap.get("launch_session_id") != manifest.get("launch_session_id"):
        mismatches.append("bootstrap.launch_session_id")
    phase = str(bootstrap.get("phase") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,127}", phase):
        mismatches.append("bootstrap.phase")
    if mismatches:
        raise ValueError(
            "qualification_collected_output_stale_or_unbound:"
            + ",".join(sorted(set(mismatches)))
        )
    return bootstrap, attempt


def _ensure_collection_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("qualification_collection_directory_unsafe")
        return
    path.mkdir(mode=0o700, parents=True)


def _collected_tree_integrity(root: Path) -> dict[str, Any]:
    """Hash every immutable snapshot file by relative path, size, and bytes."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("qualification_collection_integrity_root_unsafe")
    rows: list[dict[str, Any]] = []
    total_size_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        metadata = path.lstat()
        if path.is_symlink():
            raise ValueError("qualification_collection_integrity_symlink_forbidden")
        if path.is_dir():
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("qualification_collection_integrity_special_file_forbidden")
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        size_bytes = len(payload)
        total_size_bytes += size_bytes
        rows.append(
            {
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": _sha256_bytes(payload),
            }
        )
    canonical_rows = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "single_g1_kitchen_collection_tree_integrity.v1",
        "file_count": len(rows),
        "total_size_bytes": total_size_bytes,
        "tree_sha256": _sha256_bytes(canonical_rows),
    }


def _require_collected_tree_integrity(
    root: Path, expected: Mapping[str, Any] | None
) -> dict[str, Any]:
    observed = _collected_tree_integrity(root)
    if not isinstance(expected, Mapping) or dict(expected) != observed:
        raise ValueError("qualification_collection_snapshot_tree_integrity_mismatch")
    return observed


def _copy_initial_artifact(source: Path, destination: Path) -> dict[str, Any] | None:
    if not source.is_file() or source.is_symlink():
        return None
    payload = source.read_bytes()
    digest = _sha256_bytes(payload)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("qualification_initial_artifact_destination_unsafe")
        if _sha256_bytes(destination.read_bytes()) != digest:
            raise ValueError("qualification_initial_artifact_changed_between_snapshots")
    else:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return {"path": str(destination), "sha256": digest, "size_bytes": len(payload)}


def _collected_isaac_frames_directory(collected_root: Path) -> Path | None:
    candidates = (
        collected_root / "closed_loop_out" / "isaac_task_state" / "frames",
        collected_root
        / "closed_loop_out"
        / "episode_001"
        / "isaac_task_state"
        / "frames",
    )
    existing = [path for path in candidates if path.is_dir() and not path.is_symlink()]
    if len(existing) > 1:
        raise ValueError("qualification_collected_isaac_frames_directory_ambiguous")
    return existing[0] if existing else None


def _preserve_initial_frames(
    *,
    collected_root: Path,
    attempt_dir: Path,
    phase: str,
) -> dict[str, Any]:
    initial_dir = attempt_dir / "initial"
    _ensure_collection_directory(initial_dir)
    frames = _collected_isaac_frames_directory(collected_root)
    root_robot_pov = collected_root / "initial_policy_frame.png"
    frame_robot_pov = frames / "robot_pov_0000.png" if frames else None
    frame_overview = frames / "overview_0000.png" if frames else None
    if (
        phase == "isaac_task_executor_ready"
        and root_robot_pov.is_file()
        and frame_robot_pov is not None
        and frame_robot_pov.is_file()
        and _sha256_bytes(root_robot_pov.read_bytes())
        != _sha256_bytes(frame_robot_pov.read_bytes())
    ):
        raise ValueError("qualification_initial_robot_pov_frame_mismatch")
    robot_source = (
        root_robot_pov
        if root_robot_pov.is_file()
        else frame_robot_pov
        if frame_robot_pov is not None
        else None
    )
    artifacts: dict[str, Any] = {}
    robot = (
        _copy_initial_artifact(robot_source, initial_dir / "initial_robot_pov.png")
        if robot_source is not None
        else None
    )
    if robot:
        artifacts["robot_pov"] = robot
    if phase == "isaac_task_executor_ready" and frame_overview is not None:
        overview = _copy_initial_artifact(
            frame_overview,
            initial_dir / "initial_overview.png",
        )
        if overview:
            artifacts["overview"] = overview
    return artifacts


def _relative_collected_artifact_paths(collected_root: Path) -> dict[str, Any]:
    frames = _collected_isaac_frames_directory(collected_root)

    def relative_files(pattern: str) -> list[str]:
        if frames is None:
            return []
        return [
            str(path.relative_to(collected_root))
            for path in sorted(frames.glob(pattern))
            if path.is_file() and not path.is_symlink()
        ]

    episode_dir = collected_root / "closed_loop_out" / "episode_001"

    def relative_file(path: Path) -> str | None:
        return (
            str(path.relative_to(collected_root))
            if path.is_file() and not path.is_symlink()
            else None
        )

    log_candidates = (
        collected_root / "closed_loop_out" / "qualification_episode.log",
        collected_root / "groot_server.log",
        collected_root / "gear_sonic_controller.log",
        collected_root / "gear_sonic_isaac_dds_bridge.log",
        collected_root / "isaac_task_executor.log",
        collected_root / "closed_loop_stdout.log",
        collected_root / "closed_loop_stderr.log",
    )
    return {
        "overview_frames": relative_files("overview_*.png"),
        "robot_pov_frames": relative_files("robot_pov_*.png"),
        "final_review_video": relative_file(episode_dir / "final_review.mp4"),
        "overview_review_video": relative_file(episode_dir / "isaac_overview_review.mp4"),
        "robot_pov_review_video": relative_file(episode_dir / "isaac_robot_pov_review.mp4"),
        "wam_prediction_review_video": relative_file(
            episode_dir / "wam_prediction_review.mp4"
        ),
        "final_review_validation": relative_file(
            episode_dir / "final_review_validation.json"
        ),
        "runner_result": relative_file(collected_root / "isaac_runtime_result.json"),
        "qualification_attempt": relative_file(
            collected_root / "closed_loop_out" / "qualification_attempt.json"
        ),
        "logs": [
            str(path.relative_to(collected_root))
            for path in log_candidates
            if path.is_file() and not path.is_symlink()
        ],
    }


def _absolute_collected_artifact_paths(
    collected_root: Path,
    relative_paths: Mapping[str, Any],
) -> dict[str, Any]:
    absolute: dict[str, Any] = {}
    for key, value in relative_paths.items():
        if isinstance(value, list):
            absolute[key] = [str(collected_root / str(item)) for item in value]
        elif value:
            absolute[key] = str(collected_root / str(value))
        else:
            absolute[key] = None
    return absolute


def _validate_terminal_collection(
    snapshot_dir: Path,
    phase: str,
    *,
    manifest: Mapping[str, Any],
    latest: Mapping[str, Any],
) -> dict[str, Any]:
    collected_root = snapshot_dir / "closed_loop_output"
    episode_dir = collected_root / "closed_loop_out" / "episode_001"
    blockers: list[str] = []
    runner: dict[str, Any] = {}
    standalone_manifest: dict[str, Any] = {}
    attempt_input: dict[str, Any] = {}
    attempt_identity: dict[str, str] = {}
    trusted_proof: dict[str, Any] | None = None
    independent_frame_evidence: dict[str, Any] | None = None
    runner_path = collected_root / "isaac_runtime_result.json"
    try:
        runner = _read_collected_json(runner_path, label="runner_result")
    except ValueError as exc:
        blockers.append(str(exc))
    if runner:
        if runner.get("schema_version") != "groot_oscar_closed_loop_worker_result.v1":
            blockers.append("qualification_runner_result_schema_invalid")
        if runner.get("status") != "completed":
            blockers.append("qualification_runner_result_not_completed")
        if list(runner.get("blockers") or []):
            blockers.append("qualification_runner_result_reports_blockers")
        closed_loop = runner.get("closed_loop_manifest")
        closed_loop = dict(closed_loop) if isinstance(closed_loop, Mapping) else {}
        if not closed_loop:
            blockers.append("qualification_closed_loop_manifest_missing")
        else:
            if closed_loop.get("status") != "completed":
                blockers.append("qualification_closed_loop_manifest_not_completed")
            if list(closed_loop.get("blockers") or []):
                blockers.append("qualification_closed_loop_manifest_reports_blockers")
            if closed_loop.get("manipulation_success_proven") is not True:
                blockers.append("qualification_semantic_manipulation_success_not_proven")
            success_proof = closed_loop.get("success_proof")
            success_proof = (
                dict(success_proof) if isinstance(success_proof, Mapping) else {}
            )
            if (
                success_proof.get("manipulation_success_proven") is not True
                or success_proof.get("did_target_manipulation_succeed") is not True
            ):
                blockers.append("qualification_semantic_success_proof_not_passed")
            proof = closed_loop.get("proof")
            proof = dict(proof) if isinstance(proof, Mapping) else {}
            transition = proof.get("registered_task_completion_transition")
            transition = dict(transition) if isinstance(transition, Mapping) else {}
            if (
                transition.get("registered_transition_passed") is not True
                or transition.get("computed_transition_passed") is not True
                or list(transition.get("validation_blockers") or [])
            ):
                blockers.append("qualification_registered_task_transition_not_passed")
            termination = closed_loop.get("episode_termination")
            termination = dict(termination) if isinstance(termination, Mapping) else {}
            reason = str(termination.get("reason") or "")
            if not re.fullmatch(r"task_criterion_.+_passed_at_step_[1-9][0-9]*", reason):
                blockers.append("qualification_dynamic_semantic_stop_not_proven")
            if termination.get("task_completion_evidence_status") != "passed":
                blockers.append("qualification_task_completion_evidence_not_passed")

    standalone_manifest_path = episode_dir / "oscar_isaac_closed_loop_manifest.json"
    try:
        standalone_manifest = _read_collected_json(
            standalone_manifest_path,
            label="standalone_closed_loop_manifest",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    embedded_manifest = runner.get("closed_loop_manifest")
    embedded_manifest = (
        dict(embedded_manifest) if isinstance(embedded_manifest, Mapping) else {}
    )
    if standalone_manifest:
        if standalone_manifest.get("schema_version") != "oscar_isaac_closed_loop_eval.v1":
            blockers.append("qualification_standalone_closed_loop_manifest_schema_invalid")
        if standalone_manifest != embedded_manifest:
            blockers.append("qualification_embedded_and_standalone_manifest_mismatch")

    trace_rows: list[dict[str, Any]] = []
    trace_path = episode_dir / "oscar_isaac_closed_loop_trace.jsonl"
    try:
        trace_metadata = trace_path.lstat()
        if trace_path.is_symlink() or not stat.S_ISREG(trace_metadata.st_mode):
            raise ValueError("unsafe")
        for line_number, line in enumerate(
            trace_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"row_{line_number}_not_object")
            trace_rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        blockers.append(
            "qualification_closed_loop_trace_missing_or_invalid:"
            f"{type(exc).__name__}"
        )
        trace_rows = []
    trace_indices: list[int] = []
    try:
        trace_indices = [int(row.get("step_index")) for row in trace_rows]
    except (TypeError, ValueError):
        trace_indices = []
    expected_trace_count = int(standalone_manifest.get("steps_executed") or 0)
    termination_manifest = standalone_manifest.get("episode_termination")
    termination_manifest = (
        dict(termination_manifest)
        if isinstance(termination_manifest, Mapping)
        else {}
    )
    if (
        expected_trace_count < 1
        or len(trace_rows) != expected_trace_count
        or trace_indices != list(range(1, expected_trace_count + 1))
        or int(termination_manifest.get("steps_executed") or 0)
        != expected_trace_count
    ):
        blockers.append("qualification_closed_loop_trace_horizon_invalid")

    evaluator: dict[str, Any] = {}
    try:
        evaluator = _read_collected_json(
            episode_dir / "manipulation_success_evaluator_results.json",
            label="manipulation_success_evaluator",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    success_proof = standalone_manifest.get("success_proof")
    success_proof = dict(success_proof) if isinstance(success_proof, Mapping) else {}
    if (
        evaluator.get("schema_version")
        != "isaac_manipulation_success_evaluator_results.v1"
        or evaluator.get("manipulation_success_proven") is not True
        or evaluator.get("did_target_manipulation_succeed") is not True
        or standalone_manifest.get("manipulation_success_proven")
        is not evaluator.get("manipulation_success_proven")
        or success_proof.get("manipulation_success_proven")
        is not evaluator.get("manipulation_success_proven")
        or success_proof.get("did_target_manipulation_succeed")
        is not evaluator.get("did_target_manipulation_succeed")
    ):
        blockers.append("qualification_manipulation_evaluator_crosscheck_failed")

    attempt_input_path = collected_root / "closed_loop_out" / "attempt_input_manifest.json"
    try:
        attempt_input = _read_collected_json(
            attempt_input_path,
            label="attempt_input_manifest",
        )
        attempt_identity = load_attempt_identity(
            attempt_input_path,
            provider_allocation_id=str(manifest.get("instance_id") or ""),
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        blockers.append(f"qualification_attempt_input_identity_invalid:{type(exc).__name__}")
    if attempt_input:
        expected_qualification = {
            "launch_nonce": latest.get("attempt_nonce"),
            "allocation_launch_session_id": manifest.get("launch_session_id"),
            "qualification_attempt_bound": True,
            "qualification_attempt_sequence": latest.get("attempt_sequence"),
            "qualification_attempt_nonce": latest.get("attempt_nonce"),
            "qualification_attempt_nonce_sha256": latest.get("attempt_nonce_sha256"),
        }
        mismatches = [
            key
            for key, expected in expected_qualification.items()
            if attempt_input.get(key) != expected
        ]
        if mismatches:
            blockers.append(
                "qualification_attempt_input_binding_mismatch:"
                + ",".join(sorted(mismatches))
            )

    if expected_trace_count > 0:
        try:
            independent_frame_evidence = _collect_isaac_execution_frames(
                episode_dir,
                trace_step_count=expected_trace_count,
            )
        except Exception as exc:  # noqa: BLE001 - convert collected bytes to blocker
            blockers.append(
                "qualification_independent_isaac_frame_validation_failed:"
                f"{type(exc).__name__}"
            )
            independent_frame_evidence = None
        if independent_frame_evidence is not None:
            if independent_frame_evidence.get("status") != "passed":
                blockers.extend(
                    "qualification_independent_isaac_frame:"
                    + str(value)
                    for value in independent_frame_evidence.get("blockers") or []
                )
            final_validation = _read_collected_json(
                episode_dir / "final_review_validation.json",
                label="final_review_validation_crosscheck",
            )
            worker_frame_evidence = final_validation.get("isaac_frame_evidence")
            worker_frame_evidence = (
                dict(worker_frame_evidence)
                if isinstance(worker_frame_evidence, Mapping)
                else {}
            )
            for key in (
                "simulator_session_id",
                "stage_id",
                "attempt_id",
                "launch_nonce",
                "ordered_execution_frame_indices",
                "ordered_review_frame_indices",
                "ordered_review_control_frame_indices",
                "ordered_review_frame_count",
                "terminal_execution_frame_indices",
            ):
                if independent_frame_evidence.get(key) != worker_frame_evidence.get(key):
                    blockers.append(
                        f"qualification_worker_frame_evidence_crosscheck_mismatch:{key}"
                    )
            if attempt_identity and (
                independent_frame_evidence.get("attempt_id")
                != attempt_identity.get("attempt_id")
                or independent_frame_evidence.get("launch_nonce")
                != latest.get("attempt_nonce")
            ):
                blockers.append("qualification_isaac_frame_attempt_identity_mismatch")

    trust_path = collected_root / "runtime_ephemeral_trust.json"
    pins = load_attestation_pins(trust_path)
    if pins is None:
        blockers.append("qualification_runtime_attestation_pins_missing_or_invalid")
    elif attempt_identity and dict(pins.get("identity_binding") or {}) != attempt_identity:
        blockers.append("qualification_runtime_attestation_pins_identity_mismatch")
    worker_rows = standalone_manifest.get("g1_kitchen_proof_rows")
    worker_rows = dict(worker_rows) if isinstance(worker_rows, Mapping) else {}
    if set(worker_rows) != set(WORKER_PROOF_ROW_SPECS):
        blockers.append("qualification_worker_proof_row_set_invalid")
    if worker_rows and attempt_identity:
        trusted_proof = validate_worker_proof_rows(
            worker_rows=worker_rows,
            worker_manifest_path=standalone_manifest_path,
            collected_root=collected_root,
            identity=attempt_identity,
            attestation_pins=pins,
        )
        optional_consistency_cross_blockers = {
            "cross_row_action_sequence_mismatch:forward_consistency",
            "cross_row_action_sequence_mismatch:inverse_consistency",
        }
        unsafe_top_proof_blockers = [
            str(value)
            for value in trusted_proof.get("blockers") or []
            if str(value) not in optional_consistency_cross_blockers
        ]
        if unsafe_top_proof_blockers:
            blockers.extend(
                "qualification_trusted_proof:" + str(value)
                for value in unsafe_top_proof_blockers
            )
        trusted_rows = trusted_proof.get("rows")
        trusted_rows = dict(trusted_rows) if isinstance(trusted_rows, Mapping) else {}
        for row_id, row_value in trusted_rows.items():
            row = dict(row_value) if isinstance(row_value, Mapping) else {}
            allowed_row_blockers = set(optional_consistency_cross_blockers)
            if row_id == "forward_consistency":
                allowed_row_blockers.update(
                    {
                        "forward_consistency_proven_not_proven_by_leaf_evidence",
                        "forward_consistency:worker_status_contradicts_leaf_evidence",
                    }
                )
            elif row_id == "inverse_consistency":
                allowed_row_blockers.update(
                    {
                        "inverse_consistency_proven_not_proven_by_leaf_evidence",
                        "inverse_consistency:worker_status_contradicts_leaf_evidence",
                    }
                )
            unsafe = [
                str(value)
                for value in row.get("blockers") or []
                if str(value) not in allowed_row_blockers
            ]
            status_is_optional_consistency_cascade = bool(row.get("blockers")) and not unsafe
            if unsafe or (
                row_id not in {"forward_consistency", "inverse_consistency"}
                and row.get("status") != "passed"
                and not status_is_optional_consistency_cascade
            ):
                blockers.append(f"qualification_trusted_proof_row_invalid:{row_id}")
        expected_action_sha256s = [
            str(row.get("source_action_sha256") or "")
            for row in termination_manifest.get("task_completion_results") or []
            if isinstance(row, Mapping)
        ]
        for row_id in ("forward_consistency", "inverse_consistency"):
            raw_row = worker_rows.get(row_id)
            raw_row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
            refs = raw_row.get("leaf_artifacts")
            refs = (
                list(refs)
                if isinstance(refs, list)
                else []
            )
            for raw_ref in refs:
                ref = dict(raw_ref) if isinstance(raw_ref, Mapping) else {}
                relative = str(ref.get("path") or "")
                candidate = (collected_root / relative).resolve()
                if (
                    Path(relative).is_absolute()
                    or not candidate.is_relative_to(collected_root.resolve())
                    or not candidate.is_file()
                ):
                    blockers.append(
                        f"qualification_optional_consistency_leaf_path_invalid:{row_id}"
                    )
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    payload = {}
                observed_actions = (
                    [str(value) for value in payload.get("source_action_sha256s") or []]
                    if isinstance(payload, Mapping)
                    else []
                )
                if observed_actions != expected_action_sha256s:
                    blockers.append(
                        f"qualification_optional_consistency_action_binding_invalid:{row_id}"
                    )
    if phase != "runner_done":
        blockers.append("qualification_runner_done_phase_not_observed")
    final_review = _validate_collected_final_review(snapshot_dir)
    blockers.extend(str(value) for value in final_review.get("blockers") or [])
    unique_blockers = sorted(set(blockers))
    return {
        "status": "passed" if not unique_blockers else "blocked",
        "blockers": unique_blockers,
        "runner_result": runner or None,
        "standalone_closed_loop_manifest": standalone_manifest or None,
        "trace_step_count": len(trace_rows),
        "manipulation_success_evaluator": evaluator or None,
        "attempt_input_manifest": attempt_input or None,
        "attempt_identity": attempt_identity or None,
        "independent_isaac_frame_evidence": independent_frame_evidence,
        "trusted_proof_validation": trusted_proof,
        "final_review": final_review,
    }


def _collect(
    *,
    session_manifest: str | Path,
    adapter_output: str | Path,
    execute: bool,
) -> dict[str, Any]:
    """Collect one immutable, exact-attempt output snapshot without provider mutation."""

    result_path = Path(adapter_output).expanduser().resolve()
    manifest_path, manifest = _load_private_manifest(session_manifest)
    if not execute:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_ready",
            "action": "collect",
            "session_manifest": str(manifest_path),
            "continuing_spend": manifest.get("continuing_spend") is True,
            "provider_mutations_performed": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result

    latest = _require_latest_attempt_binding(manifest)
    signed_get_url = _read_secret_url_file(manifest_path.parent / "provider_output_get_url.txt")
    archive = _download_provider_output_archive(signed_get_url)
    _validate_provider_output_archive_limits(archive)
    archive_sha256 = _sha256_bytes(archive)
    collections_root = manifest_path.parent / COLLECTIONS_DIR_NAME
    _ensure_collection_directory(collections_root)
    attempt_dir = collections_root / str(latest["attempt_slug"])
    _ensure_collection_directory(attempt_dir)
    snapshots_dir = attempt_dir / "snapshots"
    _ensure_collection_directory(snapshots_dir)
    extracted = _extract_provider_output_snapshot(archive, collections_root)
    try:
        bootstrap, _attempt_identity = _validate_collected_attempt_binding(
            extracted,
            manifest=manifest,
            latest=latest,
        )
        phase = str(bootstrap["phase"])
        prior_terminal = latest.get("collection_status") in {
            "collected_terminal_passed",
            "collected_terminal_blocked",
        }
        prior_collection = latest.get("latest_collection")
        prior_collection = (
            dict(prior_collection) if isinstance(prior_collection, Mapping) else {}
        )
        if prior_terminal and prior_collection.get("archive_sha256") != archive_sha256:
            raise ValueError("qualification_collection_regressed_after_terminal")
        snapshot_dir = snapshots_dir / f"{phase}-{archive_sha256[:16]}"
        already_collected = snapshot_dir.exists() or snapshot_dir.is_symlink()
        if already_collected:
            if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
                raise ValueError("qualification_collection_snapshot_destination_unsafe")
            marker = _read_collected_json(
                snapshot_dir / "collection.json",
                label="collection_marker",
            )
            if (
                marker.get("archive_sha256") != archive_sha256
                or marker.get("attempt_nonce_sha256") != latest["attempt_nonce_sha256"]
                or marker.get("phase") != phase
            ):
                raise ValueError("qualification_collection_snapshot_binding_mismatch")
            _require_collected_tree_integrity(
                snapshot_dir / "closed_loop_output",
                marker.get("closed_loop_output_integrity"),
            )
        else:
            temporary = attempt_dir / f".snapshot-{uuid.uuid4().hex}"
            temporary.mkdir(mode=0o700)
            try:
                shutil.copytree(extracted, temporary / "closed_loop_output")
                relative_paths = _relative_collected_artifact_paths(
                    temporary / "closed_loop_output"
                )
                tree_integrity = _collected_tree_integrity(
                    temporary / "closed_loop_output"
                )
                _private_write_json(
                    temporary / "collection.json",
                    {
                        "schema_version": "single_g1_kitchen_qualification_collection.v1",
                        "archive_sha256": archive_sha256,
                        "phase": phase,
                        "attempt_sequence": latest["attempt_sequence"],
                        "attempt_nonce_sha256": latest["attempt_nonce_sha256"],
                        "episode_bootstrap_sha256": latest["episode_bootstrap_sha256"],
                        "bundle_sha256": latest["bundle_sha256"],
                        "overlay_revision": latest["overlay_revision"],
                        "relative_artifact_paths": relative_paths,
                        "closed_loop_output_integrity": tree_integrity,
                        "raw_signed_url_recorded": False,
                        "provider_mutations_performed": 0,
                        "collected_at": utc_now_iso(),
                    },
                )
                os.replace(temporary, snapshot_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        marker = _read_collected_json(
            snapshot_dir / "collection.json",
            label="collection_marker",
        )
        collected_root = snapshot_dir / "closed_loop_output"
        tree_integrity = _require_collected_tree_integrity(
            collected_root,
            marker.get("closed_loop_output_integrity"),
        )
        initial_artifacts = _preserve_initial_frames(
            collected_root=collected_root,
            attempt_dir=attempt_dir,
            phase=phase,
        )
        relative_paths = marker.get("relative_artifact_paths")
        relative_paths = (
            dict(relative_paths) if isinstance(relative_paths, Mapping) else {}
        )
        artifact_paths = _absolute_collected_artifact_paths(collected_root, relative_paths)
        runner_result_present = (collected_root / "isaac_runtime_result.json").is_file()
        stopped_after_dispatch = latest.get("remote_process_state") == "stopped"
        terminal = bool(
            phase in {"runner_done", "runner_timeout"}
            or runner_result_present
            or stopped_after_dispatch
        )
        validation: dict[str, Any] | None = None
        if terminal:
            validation = _validate_terminal_collection(
                snapshot_dir,
                phase,
                manifest=manifest,
                latest=latest,
            )
            passed = validation["status"] == "passed"
            status = (
                "episode_collected_passed_continuing_spend"
                if passed
                else "episode_collected_blocked_continuing_spend"
            )
            collection_status = (
                "collected_terminal_passed" if passed else "collected_terminal_blocked"
            )
            blockers = list(validation["blockers"])
        else:
            status = "episode_snapshot_collected_continuing_spend"
            collection_status = "pending"
            blockers = []
        collection_record = {
            "schema_version": "single_g1_kitchen_qualification_collection_record.v1",
            "attempt_sequence": latest["attempt_sequence"],
            "attempt_nonce_sha256": latest["attempt_nonce_sha256"],
            "archive_sha256": archive_sha256,
            "phase": phase,
            "snapshot_dir": str(snapshot_dir),
            "collection_status": collection_status,
            "artifact_paths": artifact_paths,
            "initial_artifacts": initial_artifacts,
            "validation_status": validation.get("status") if validation else None,
            "collected_at": utc_now_iso(),
            "provider_mutation_performed": False,
            "raw_signed_url_recorded": False,
            "closed_loop_output_integrity": tree_integrity,
        }
        persisted_latest = dict(manifest["latest_attempt"])
        persisted_latest.update(
            {
                "collection_status": collection_status,
                "latest_collection": collection_record,
            }
        )
        manifest["latest_attempt"] = persisted_latest
        prior_collections = list(manifest.get("collections") or [])
        if not any(
            isinstance(row, Mapping)
            and row.get("archive_sha256") == archive_sha256
            and row.get("attempt_nonce_sha256") == latest["attempt_nonce_sha256"]
            for row in prior_collections
        ):
            prior_collections.append(collection_record)
        manifest["collections"] = prior_collections
        manifest["status"] = status
        manifest.setdefault("history", []).append(
            {
                "action": "collect",
                "status": status,
                "recorded_at": utc_now_iso(),
                "attempt_sequence": latest["attempt_sequence"],
                "archive_sha256": archive_sha256,
                "phase": phase,
                "already_collected": already_collected,
                "provider_mutation_performed": False,
            }
        )
        _private_write_json(manifest_path, manifest)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "action": "collect",
            "instance_id": manifest.get("instance_id"),
            "session_manifest": str(manifest_path),
            "attempt_sequence": latest["attempt_sequence"],
            "attempt_nonce_sha256": latest["attempt_nonce_sha256"],
            "archive_sha256": archive_sha256,
            "phase": phase,
            "snapshot_dir": str(snapshot_dir),
            "already_collected": already_collected,
            "artifact_paths": artifact_paths,
            "initial_artifacts": initial_artifacts,
            "validation": validation,
            "continuing_spend": manifest.get("continuing_spend") is True,
            "provider_mutations_performed": 0,
            "raw_signed_url_recorded": False,
            "blockers": blockers,
        }
        write_json(result_path, result)
        return result
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


def _control(
    *,
    action: str,
    component: str | None,
    session_manifest: str | Path,
    adapter_output: str | Path,
    execute: bool,
    identity_file: str,
    tail_lines: int,
    admission_out: str | Path | None,
) -> dict[str, Any]:
    result_path = Path(adapter_output).expanduser().resolve()
    manifest_path, manifest = _load_private_manifest(session_manifest)
    resolved_component = COMPONENT_ALIASES.get(str(component or "episode"))
    if not resolved_component:
        raise ValueError("qualification_component_invalid")
    if action == "restart-component" and str(component or "") not in RESTARTABLE_COMPONENTS:
        raise ValueError("qualification_restart_component_invalid")
    if action == "stop-component" and str(component or "") not in STOPPABLE_COMPONENTS:
        raise ValueError("qualification_stop_component_invalid")
    remote_action = {
        "restart-component": "restart",
        "stop-component": "stop",
    }.get(action, action)
    if remote_action not in {"run", "status", "tail", "gpu-status", "restart", "stop"}:
        raise ValueError("qualification_control_action_invalid")
    if not 1 <= int(tail_lines) <= MAX_TAIL_LINES:
        raise ValueError("qualification_tail_lines_invalid")
    expected_attempt_sequence: int | None = None
    expected_attempt_nonce = ""
    expected_attempt_nonce_sha256 = ""
    if action == "run" and resolved_component in {"episode", "bootstrap"}:
        prior = manifest.get("latest_attempt")
        prior = dict(prior) if isinstance(prior, Mapping) else {}
        if prior and prior.get("collection_status") not in {
            "collected_terminal_passed",
            "collected_terminal_blocked",
        }:
            raise ValueError("qualification_collect_required_before_episode_rerun")
        try:
            previous_sequence = int(prior.get("attempt_sequence") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("qualification_prior_attempt_sequence_invalid") from exc
        expected_attempt_sequence = previous_sequence + 1
        expected_attempt_nonce = (
            f"{manifest['launch_session_id']}:attempt_{expected_attempt_sequence:04d}"
        )
        expected_attempt_nonce_sha256 = _sha256_bytes(
            expected_attempt_nonce.encode("utf-8")
        )
    if not execute:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_ready",
            "action": action,
            "component": resolved_component,
            "session_manifest": str(manifest_path),
            "continuing_spend": manifest.get("continuing_spend") is True,
            "provider_mutations_performed": 0,
            "qualification_gate_matrix": manifest["qualification_gate_matrix"],
            "blockers": [],
        }
        write_json(result_path, result)
        return result
    if manifest.get("continuing_spend") is not True:
        raise ValueError("qualification_session_not_live")
    if time.time() >= float(manifest.get("watchdog_deadline_epoch") or 0):
        raise ValueError("qualification_session_ttl_expired")
    instance_id = str(manifest.get("instance_id") or "")
    provider = get_render_provider("vast")
    inspected = provider.inspect(instance_id)
    if (
        inspected.get("status") != "observed"
        or str(inspected.get("instance_id") or "") != instance_id
        or inspected.get("name") != manifest.get("resource_name")
        or inspected.get("image_runtype") != "ssh_direct"
        or inspected.get("direct_port_ready") is not True
    ):
        raise ValueError("qualification_session_provider_binding_not_observed")
    connection = _safe_connection(inspected)
    bound_connection = dict(manifest.get("ssh_connection") or {})
    host_key = dict(manifest.get("ssh_host_key") or {})
    known_hosts_file = str(host_key.get("known_hosts_file") or "")
    prior_attach_ready = any(
        isinstance(row, Mapping)
        and row.get("status") == "allocated_ready_continuing_spend"
        for row in manifest.get("history") or []
    )
    root_pin_endpoint_mismatch = False
    root_pin_path = manifest_path.parent / VAST_SSH_HOST_KEY_FINGERPRINT_NAME
    try:
        root_pin_mode = stat.S_IMODE(root_pin_path.stat().st_mode)
        root_pin = json.loads(root_pin_path.read_text(encoding="utf-8"))
        root_pin_endpoint_mismatch = bool(
            not root_pin_path.is_symlink()
            and root_pin_mode == 0o600
            and isinstance(root_pin, Mapping)
            and root_pin.get("ssh_host")
            and root_pin.get("ssh_port")
            and (
                root_pin.get("ssh_host") != connection.get("ssh_host")
                or root_pin.get("ssh_port") != connection.get("ssh_port")
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        root_pin_endpoint_mismatch = False
    endpoint_changed_after_loading = bool(
        action == "status"
        and inspected.get("actual_status") == "running"
        and bound_connection
        and (
            connection.get("ssh_host") != bound_connection.get("ssh_host")
            or connection.get("ssh_port") != bound_connection.get("ssh_port")
        )
        and any(
            isinstance(row, Mapping) and row.get("actual_status") == "loading"
            for row in manifest.get("ssh_readiness_observations") or []
        )
        and not prior_attach_ready
        and (manifest.get("ssh_attach_probe") or {}).get("status") != "completed"
    )
    running_endpoint_trust_recovery = bool(
        action == "status"
        and inspected.get("actual_status") == "running"
        and not prior_attach_ready
        and root_pin_endpoint_mismatch
        and (
            not known_hosts_file
            or "vast_ssh_existing_host_key_pin_invalid"
            in set(host_key.get("blockers") or [])
        )
    )
    if action == "status" and (
        not bound_connection or not known_hosts_file or endpoint_changed_after_loading
        or running_endpoint_trust_recovery
    ):
        (
            recovered_connection,
            observations,
            recovered_host_key,
            control_probe,
        ) = _wait_for_qualification_attach(
            provider,
            instance_id=instance_id,
            resource_name=str(manifest["resource_name"]),
            attempt_dir=(
                manifest_path.parent / "ssh_attach_running_endpoint"
                if endpoint_changed_after_loading or running_endpoint_trust_recovery
                else manifest_path.parent
            ),
            identity_file=identity_file,
            timeout_seconds=180,
        )
        host_key_ready = bool(
            recovered_host_key.get("status") == "enrolled"
            and recovered_host_key.get("known_hosts_file")
            and recovered_host_key.get("fingerprint_artifact")
            and recovered_host_key.get("tofu_pinned") is True
        )
        authenticated_control_ready = control_probe.get("status") == "completed"
        recovered = bool(
            recovered_connection and host_key_ready and authenticated_control_ready
        )
        manifest.update(
            {
                "status": (
                    "allocated_ready_continuing_spend"
                    if recovered
                    else "allocated_attach_blocked_continuing_spend"
                ),
                "ssh_connection": recovered_connection or None,
                "ssh_host_key": recovered_host_key or None,
                "ssh_attach_probe": {
                    "status": control_probe.get("status"),
                    "returncode": control_probe.get("returncode"),
                    "action": control_probe.get("action"),
                    "component": control_probe.get("component"),
                    "blockers": list(control_probe.get("blockers") or []),
                    "strict_host_key_checking": control_probe.get(
                        "strict_host_key_checking"
                    ),
                }
                if control_probe
                else None,
                "ssh_readiness_observations": observations,
                "continuing_spend": True,
            }
        )
        manifest.setdefault("history", []).append(
            {
                "action": "recover-attach-via-status",
                "status": manifest["status"],
                "recorded_at": utc_now_iso(),
                "provider_mutation_performed": False,
            }
        )
        _private_write_json(manifest_path, manifest)
        if not recovered:
            raise ValueError("qualification_vast_ssh_reattach_not_ready")
        connection = recovered_connection
        bound_connection = dict(manifest["ssh_connection"])
        host_key = dict(manifest["ssh_host_key"])
        known_hosts_file = str(host_key["known_hosts_file"])
    if connection.get("ssh_host") != bound_connection.get("ssh_host") or connection.get(
        "ssh_port"
    ) != bound_connection.get("ssh_port"):
        raise ValueError("qualification_session_ssh_endpoint_changed")
    if not known_hosts_file:
        raise ValueError("qualification_session_known_hosts_missing")
    from .gpu_render_providers import run_vast_ssh_control

    if remote_action in {"run", "restart", "stop"}:
        admit_qualification_control_mutation(
            admission_out, manifest, inspected, instance_id, remote_action, resolved_component
        )
    control = run_vast_ssh_control(
        connection,
        action=remote_action,
        component=resolved_component,
        known_hosts_file=known_hosts_file,
        identity_file=identity_file,
        timeout_seconds=30.0,
        tail_lines=int(tail_lines),
    )
    control_status = str(control.get("status") or "")
    completed = control_status in {
        "completed",
        "passed",
        "observed",
        "running",
        "dispatched",
    }
    status_by_action = {
        "run": "episode_dispatched_continuing_spend",
        "status": "status_observed_continuing_spend",
        "tail": "tail_collected_continuing_spend",
        "gpu-status": "gpu_status_collected_continuing_spend",
        "restart-component": "component_restarted_continuing_spend",
        "stop-component": "component_stopped_continuing_spend",
    }
    attempt_match = re.search(
        r"\battempt_sequence=(\d+)\s+attempt_nonce_sha256=([0-9a-f]{64})\b",
        str(control.get("stdout") or ""),
    )
    attempt_identity_blocker: str | None = None
    if expected_attempt_sequence is not None:
        if not attempt_match:
            attempt_identity_blocker = "qualification_run_attempt_identity_missing"
        elif (
            int(attempt_match.group(1)) != expected_attempt_sequence
            or attempt_match.group(2) != expected_attempt_nonce_sha256
        ):
            attempt_identity_blocker = "qualification_run_attempt_identity_mismatch"
        if attempt_identity_blocker:
            completed = False
    manifest["status"] = (
        status_by_action[action] if completed else "control_blocked_continuing_spend"
    )
    manifest["continuing_spend"] = True
    recorded_at = utc_now_iso()
    manifest["last_control"] = {
        "action": action,
        "remote_action": remote_action,
        "component": resolved_component,
        "status": control_status,
        "recorded_at": recorded_at,
        "bootstrap_sha256": manifest["bootstrap"]["episode_bootstrap_sha256"],
        "attempt_sequence": int(attempt_match.group(1)) if attempt_match else None,
        "attempt_nonce_sha256": attempt_match.group(2) if attempt_match else None,
    }
    if completed and expected_attempt_sequence is not None:
        manifest["latest_attempt"] = {
            "schema_version": "single_g1_kitchen_qualification_attempt_binding.v1",
            "attempt_sequence": expected_attempt_sequence,
            "attempt_slug": f"attempt_{expected_attempt_sequence:04d}",
            "attempt_nonce": expected_attempt_nonce,
            "attempt_nonce_sha256": expected_attempt_nonce_sha256,
            "launch_session_id": manifest["launch_session_id"],
            "episode_bootstrap_sha256": manifest["bootstrap"][
                "episode_bootstrap_sha256"
            ],
            "bundle_sha256": manifest["bundle_sha256"],
            "overlay_revision": manifest["bootstrap"]["overlay_revision"],
            "dispatched_at": recorded_at,
            "remote_process_state": "running",
            "collection_status": "pending",
        }
    status_match = re.search(
        r"\baction=status component=(?:episode|bootstrap) state=(running|stopped)\b.*?"
        r"bootstrap_sha256=([0-9a-f]{64})\s+overlay_revision=(\d+)\s+"
        r"attempt_sequence=([0-9]*)\s+attempt_nonce_sha256=([0-9a-f]*)\b",
        str(control.get("stdout") or ""),
    )
    if action == "status" and resolved_component in {"episode", "bootstrap"} and status_match:
        latest_value = manifest.get("latest_attempt")
        latest_attempt = (
            dict(latest_value) if isinstance(latest_value, Mapping) else {}
        )
        if (
            latest_attempt
            and latest_attempt.get("episode_bootstrap_sha256") == status_match.group(2)
            and latest_attempt.get("overlay_revision") == int(status_match.group(3))
            and str(latest_attempt.get("attempt_sequence")) == status_match.group(4)
            and latest_attempt.get("attempt_nonce_sha256") == status_match.group(5)
        ):
            latest_attempt["remote_process_state"] = status_match.group(1)
            latest_attempt["remote_process_state_observed_at"] = recorded_at
            manifest["latest_attempt"] = latest_attempt
    manifest.setdefault("history", []).append(
        {
            **manifest["last_control"],
            "provider_mutation_performed": action
            in {"run", "restart-component", "stop-component"},
        }
    )
    _private_write_json(manifest_path, manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": manifest["status"],
        "action": action,
        "component": resolved_component,
        "instance_id": instance_id,
        "session_manifest": str(manifest_path),
        "control": control,
        "continuing_spend": True,
        "provider_mutations_performed": 1
        if action in {"run", "restart-component", "stop-component"}
        else 0,
        "qualification_gate_matrix": manifest["qualification_gate_matrix"],
        "claim_boundary": manifest["claim_boundary"],
        "blockers": (
            []
            if completed
            else [
                attempt_identity_blocker or "qualification_fixed_control_failed"
            ]
        ),
    }
    write_json(result_path, result)
    return result


def _teardown(
    *,
    session_manifest: str | Path,
    adapter_output: str | Path,
    execute: bool,
) -> dict[str, Any]:
    result_path = Path(adapter_output).expanduser().resolve()
    manifest_path, manifest = _load_private_manifest(session_manifest)
    if not execute:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_ready",
            "action": "teardown",
            "session_manifest": str(manifest_path),
            "continuing_spend": manifest.get("continuing_spend") is True,
            "provider_mutations_performed": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result
    provider = get_render_provider("vast")
    armed = dict(manifest.get("watchdog") or {})
    instance_id = str(manifest.get("instance_id") or "")
    if not instance_id or str(armed.get("pod_name_prefix") or "") != manifest.get(
        "resource_name_prefix"
    ):
        raise ValueError("qualification_teardown_binding_invalid")
    teardown = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=str(manifest["resource_name_prefix"]),
        armed=armed,
        provider_name="vast",
    )
    global_inventory = provider.billable_inventory(name_prefix="")
    exact_and_prefix_absent = teardown.get("provider_absence_confirmed") is True
    global_absent = bool(
        global_inventory.get("api_confirmed") is True
        and global_inventory.get("live_resource_count") == 0
    )
    absence_proven = exact_and_prefix_absent and global_absent
    pending_path = str(manifest.get("pending_teardown_record") or "")
    pending: dict[str, Any] = {}
    if absence_proven and pending_path:
        pending = close_pending_teardown(
            pending_path,
            {
                "status": "PASS",
                "provider": "vast",
                "allocation_id": instance_id,
                "exact_id_absence_confirmed": True,
                "name_prefix_absence_confirmed": True,
                "global_inventory_absence_confirmed": True,
                "status_source": "provider_api_exact_id_prefix_and_global_inventory",
            },
        )
    manifest.update(
        {
            "status": (
                "teardown_completed_provider_zero"
                if absence_proven
                else "teardown_unverified_continuing_spend_unknown"
            ),
            "continuing_spend": not absence_proven,
            "provider_absence_confirmed": absence_proven,
            "pending_teardown_status": pending.get("status")
            if pending
            else manifest.get("pending_teardown_status"),
            "teardown": teardown,
            "final_global_inventory": global_inventory,
        }
    )
    manifest.setdefault("history", []).append(
        {
            "action": "teardown",
            "status": manifest["status"],
            "recorded_at": utc_now_iso(),
            "provider_mutation_performed": True,
            "provider_absence_confirmed": absence_proven,
        }
    )
    _private_write_json(manifest_path, manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": manifest["status"],
        "action": "teardown",
        "instance_id": instance_id,
        "session_manifest": str(manifest_path),
        "teardown": teardown,
        "final_global_inventory": global_inventory,
        "pending_teardown_status": manifest.get("pending_teardown_status"),
        "continuing_spend": manifest["continuing_spend"],
        "provider_mutations_performed": teardown.get("provider_mutations_performed", 0),
        "blockers": []
        if absence_proven
        else ["qualification_teardown_exact_prefix_global_absence_not_proven"],
    }
    write_json(result_path, result)
    return result


def run_qualification_session(
    *,
    action: str,
    session_manifest: str | Path,
    provider_name: str = "vast",
    component: str | None = None,
    tail_lines: int = 200,
    identity_file: str = DEFAULT_IDENTITY_FILE,
    episode_bundle: str | Path | None = None,
    training_dataset: str | Path | None = None,
    trained_checkpoint_path: str | Path | None = None,
    provider_bundle_url_file: str | Path | None = None,
    provider_output_put_url_file: str | Path | None = None,
    provider_output_get_url_file: str | Path | None = None,
    provider_bootstrap_url_file: str | Path | None = None,
    release_evidence: str | Path | None = None,
    expected_source_commit: str | None = None,
    provider_launch_request: str | Path | None = None,
    preflight_bundle: str | Path | None = None,
    admission_out: str | Path | None = None,
    bound_request_out: str | Path | None = None,
    adapter_output: str | Path,
    pod_name: str = "",
    execute: bool = False,
) -> dict[str, Any]:
    """Execute one canonical qualification lifecycle action."""

    if provider_name != "vast":
        raise ValueError("qualification_session_requires_vast")
    if action not in SESSION_ACTIONS:
        raise ValueError("qualification_session_action_invalid")
    if action == "allocate":
        required = {
            "episode_bundle": episode_bundle,
            "provider_bundle_url_file": provider_bundle_url_file,
            "provider_output_put_url_file": provider_output_put_url_file,
            "provider_output_get_url_file": provider_output_get_url_file,
            "release_evidence": release_evidence,
            "expected_source_commit": expected_source_commit,
            "provider_launch_request": provider_launch_request,
            "preflight_bundle": preflight_bundle,
            "admission_out": admission_out,
            "bound_request_out": bound_request_out,
        }
        missing = sorted(name for name, value in required.items() if value in {None, ""})
        if missing:
            raise ValueError("qualification_allocate_arguments_missing:" + ",".join(missing))
        return _allocate(
            session_manifest=session_manifest,
            episode_bundle=episode_bundle,
            provider_bundle_url_file=provider_bundle_url_file,
            provider_output_put_url_file=provider_output_put_url_file,
            provider_output_get_url_file=provider_output_get_url_file,
            provider_bootstrap_url_file=provider_bootstrap_url_file,
            release_evidence=release_evidence,
            provider_launch_request=provider_launch_request,
            preflight_bundle=preflight_bundle,
            admission_out=admission_out,
            bound_request_out=bound_request_out,
            adapter_output=adapter_output,
            pod_name=pod_name,
            execute=execute,
            identity_file=identity_file,
            expected_source_commit=expected_source_commit,
            training_dataset=training_dataset,
            trained_checkpoint_path=trained_checkpoint_path,
        )
    if action == "teardown":
        return _teardown(session_manifest=session_manifest, adapter_output=adapter_output, execute=execute)
    if action == "collect":
        return _collect(
            session_manifest=session_manifest,
            adapter_output=adapter_output,
            execute=execute,
        )
    if action == "refresh-bootstrap":
        if episode_bundle in {None, ""}:
            raise ValueError("qualification_refresh_episode_bundle_missing")
        return _refresh_bootstrap(
            session_manifest=session_manifest,
            episode_bundle=episode_bundle,
            provider_bootstrap_url_file=provider_bootstrap_url_file,
            adapter_output=adapter_output, execute=execute, identity_file=identity_file,
            training_dataset=training_dataset, trained_checkpoint_path=trained_checkpoint_path, admission_out=admission_out,
        )
    return _control(
        action=action, component=component, session_manifest=session_manifest,
        adapter_output=adapter_output, execute=execute, identity_file=identity_file,
        tail_lines=tail_lines, admission_out=admission_out,
    )
